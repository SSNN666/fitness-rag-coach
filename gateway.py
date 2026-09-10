"""
轻量化网关 —— 限流 / 降噪 / 令牌预算 / 内存降级 / 结构化日志
============================================================
所有组件遵循 fail-open 模式：内部异常 → logging.ERROR → 返回安全默认值。

用法:
    from gateway import Gateway, GatewayConfig
    _gate = Gateway(GatewayConfig.from_module())

    # Hook 0: 速率限制
    allowed, reason = _gate.check_rate_limit(session_id)
    # Hook 1: 请求降噪
    if _gate.is_duplicate(query, session_id): ...
    # Hook 2: 令牌预算守卫
    ctx = _gate.guard_token_budget(system_prompt, history_store, session_id, ctx, query)
    # Hook 3: 内存守卫
    active_llm = _gate.get_active_llm(base_llm)
    # Hook 4: 结构化日志
    _gate.log_cycle("response", query=..., len_in=..., len_out=...)
"""

import logging
import re
import time
import json
import hashlib
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

from cost_guard import DEGRADED, EXHAUSTED, CostGuard


# ============================================================
# GatewayConfig
# ============================================================

@dataclass
class GatewayConfig:
    """网关全部可调参数。通过 GatewayConfig.from_module(config) 从 config.py 读取。"""

    # 主开关
    enabled: bool = True
    log_path: str = "gateway.log"
    log_level: str = "INFO"

    # 速率限制
    rate_limit_enabled: bool = True
    rate_limit_max: int = 10
    rate_limit_window_s: int = 60
    rate_limit_cooldown_s: int = 30

    # 请求降噪
    noise_reduction_enabled: bool = True
    noise_similarity: float = 0.85
    noise_window: int = 5
    noise_ttl_s: float = 600.0

    # 会话级状态回收（限流/降噪按 session_id 建键）
    state_max_keys: int = 512
    state_sweep_interval_s: float = 60.0

    # 成本上限（进程级累计 token）
    cost_guard_enabled: bool = True
    cost_soft_limit_tokens: int = 5_000_000
    cost_hard_limit_tokens: int = 20_000_000
    cost_window_s: float = 86400.0
    # None = 仅内存（测试/本地调试）；from_module 时取 config.COST_GUARD_PATH
    cost_guard_path: str | None = None

    # 令牌预算 (Strategy A)
    token_budget_enabled: bool = True
    token_budget_ratio: float = 0.55
    token_answer_min: int = 500
    chat_max_history_tokens: int = 2048
    ollama_num_ctx: int = 8192
    token_budget_truncate_warn: float = 0.30

    # 内存守卫 (Strategy B)
    memory_guard_enabled: bool = True
    memory_threshold_gb: float = 3.0
    memory_recovery_gb: float = 4.0
    memory_degraded_model: str = "qwen2.5:1.5b"

    @classmethod
    def from_module(cls, config_module=None):
        """从 config 模块读取所有值，缺失时使用默认值。"""
        if config_module is None:
            import config as config_module
        cfg = config_module
        g = getattr  # shorthand

        return cls(
            enabled=g(cfg, "GATEWAY_ENABLED", True),
            log_path=g(cfg, "GATEWAY_LOG_PATH", "gateway.log"),
            log_level=g(cfg, "GATEWAY_LOG_LEVEL", "INFO"),

            rate_limit_enabled=g(cfg, "RATE_LIMIT_ENABLED", True),
            rate_limit_max=g(cfg, "RATE_LIMIT_MAX", 10),
            rate_limit_window_s=g(cfg, "RATE_LIMIT_WINDOW", 60),
            rate_limit_cooldown_s=g(cfg, "RATE_LIMIT_COOLDOWN", 30),

            noise_reduction_enabled=g(cfg, "NOISE_REDUCTION_ENABLED", True),
            noise_similarity=g(cfg, "NOISE_SIMILARITY", 0.85),
            noise_window=g(cfg, "NOISE_WINDOW", 5),
            noise_ttl_s=g(cfg, "NOISE_TTL", 600),

            state_max_keys=g(cfg, "GW_STATE_MAX_KEYS", 512),
            state_sweep_interval_s=g(cfg, "GW_STATE_SWEEP_INTERVAL", 60),

            cost_guard_enabled=g(cfg, "COST_GUARD_ENABLED", True),
            cost_soft_limit_tokens=g(cfg, "COST_SOFT_LIMIT_TOKENS", 5_000_000),
            cost_hard_limit_tokens=g(cfg, "COST_HARD_LIMIT_TOKENS", 20_000_000),
            cost_window_s=g(cfg, "COST_WINDOW", 86400),
            cost_guard_path=g(cfg, "COST_GUARD_PATH", None),

            token_budget_enabled=g(cfg, "TOKEN_BUDGET_ENABLED", True),
            token_budget_ratio=g(cfg, "CONTEXT_BUDGET_RATIO", 0.55),
            token_answer_min=g(cfg, "ANSWER_BUDGET_MIN", 500),
            chat_max_history_tokens=g(cfg, "CHAT_MAX_HISTORY_TOKENS", 2048),
            ollama_num_ctx=g(cfg, "OLLAMA_NUM_CTX", 8192),
            token_budget_truncate_warn=g(cfg, "TOKEN_BUDGET_TRUNCATE_WARN", 0.30),

            memory_guard_enabled=g(cfg, "MEMORY_GUARD_ENABLED", True),
            memory_threshold_gb=g(cfg, "MEMORY_GUARD_THRESHOLD", 3.0),
            memory_recovery_gb=g(cfg, "MEMORY_GUARD_RECOVERY", 4.0),
            memory_degraded_model=g(cfg, "MEMORY_GUARD_DEGRADED_MODEL", "qwen2.5:1.5b"),
        )


# ============================================================
# 令牌估算（复用 app.py 逻辑，独立副本避免循环导入）
# ============================================================

def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文 ~1.5 tokens/char，英文 ~0.3 tokens/char。"""
    if not text:
        return 0
    cn_chars = len(re.findall(r'[一-鿿]', text))
    other_chars = max(0, len(text) - cn_chars)
    return int(cn_chars * 1.5 + other_chars * 0.3)


# ============================================================
# _StructuredLogger
# ============================================================

class _StructuredLogger:
    """JSON Lines 日志 → 文件，RotatingFileHandler 5MB × 3 备份。

    llm_adapter / content_moderation 等底层模块的日志共用同一文件句柄
    （同一 handler 实例 → 共享轮转锁），不再散落控制台明文。
    """

    # 并入 gateway.log JSON 的底层模块 logger 名
    EXTRA_JSON_LOGGERS = ("llm_adapter", "content_moderation")

    def __init__(self, path: str, level: str = "INFO"):
        self._logger = logging.getLogger("gateway")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        fh = RotatingFileHandler(
            path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(_JsonFormatter())
        fh.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.addHandler(fh)

        for name in self.EXTRA_JSON_LOGGERS:
            lg = logging.getLogger(name)
            lg.handlers.clear()
            lg.setLevel(logging.DEBUG)
            lg.addHandler(fh)
            lg.propagate = False   # 不再冒泡到 root(避免控制台明文重复输出)

    def log(self, level: str, event: str, **kwargs):
        """写入一条结构化日志。"""
        try:
            extra = {"extra_fields": kwargs} if kwargs else {}
            getattr(self._logger, level.lower())(event, extra=extra)
        except Exception:
            pass  # 日志失败不阻断业务

    def debug(self, event: str, **kwargs):
        self.log("debug", event, **kwargs)

    def info(self, event: str, **kwargs):
        self.log("info", event, **kwargs)

    def warning(self, event: str, **kwargs):
        self.log("warning", event, **kwargs)

    def error(self, event: str, **kwargs):
        self.log("error", event, **kwargs)

    # ---------- 康养 Demo 扩展：完整日志（query/检索片段/prompt/输出/token 消耗） ----------

    def log_usage(self, request_id: str, role: str, usage) -> None:
        """LLM 调用 token 消耗（llm_adapter 的 on_usage 回调目标）。usage 为 UsageInfo。"""
        self.info("llm_usage", request_id=request_id, role=role,
                  provider=usage.provider, model=usage.model,
                  prompt_tokens=usage.prompt_tokens,
                  completion_tokens=usage.completion_tokens,
                  total_tokens=usage.total_tokens,
                  latency_ms=usage.latency_ms)

    def log_retrieval(self, request_id: str, query: str, docs: list) -> None:
        """检索片段：原始 query + 每条文档的来源/页码/得分/摘要（JSON 可序列化 dict）。"""
        self.info("retrieval", request_id=request_id, query=query, docs=docs)

    def log_prompt(self, request_id: str, role: str, prompt: str) -> None:
        """最终拼装的 prompt 全文（用户画像脱敏；问题本身保留用于质量审计）。"""
        self.info("prompt", request_id=request_id, role=role, prompt=_redact(prompt))

    def log_answer(self, request_id: str, answer: str, censor: dict | None = None) -> None:
        """模型最终输出 + 内容审核结果（用户画像脱敏）。"""
        self.info("answer", request_id=request_id, answer=_redact(answer), censor=censor)


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        obj = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            obj.update(extra)
        return json.dumps(obj, ensure_ascii=False)


# 用户画像脱敏（身高/体重/目标属个人信息，日志明文落盘需替换）
_PROFILE_PATTERN = re.compile(r"用户画像[：:]\s*[^\n]*")


def _redact(text: str) -> str:
    """日志脱敏：用户画像行替换为占位符；其余（问题/回答正文）保留用于审计。"""
    if not text:
        return text
    return _PROFILE_PATTERN.sub("用户画像：[已脱敏]", text)


def _slog(logger, level: str, event: str, **kwargs):
    """安全日志调用：logger 为 None 时静默跳过。"""
    if logger is not None:
        getattr(logger, level)(event, **kwargs)


# ============================================================
# _SlidingWindowLimiter
# ============================================================

class _SlidingWindowLimiter:
    """基于 st.session_state 的滑动窗口速率限制器。"""

    def __init__(self, cfg: GatewayConfig, logger: _StructuredLogger):
        self._cfg = cfg
        self._log = logger

    def check(self, session_id: str, session_state: dict) -> tuple[bool, str]:
        """返回 (allowed, reason)。"""
        if not self._cfg.rate_limit_enabled:
            return True, ""

        try:
            now = time.time()
            key = f"_gw_rate_{session_id}"
            state = session_state.get(key, {"timestamps": [], "blocked_until": 0})

            # 冷却期内直接拒绝
            if now < state.get("blocked_until", 0):
                remain = int(state["blocked_until"] - now)
                _slog(self._log, 'warning', "rate_limit_block", session=session_id,
                                  cooldown_remaining=remain)
                return False, f"请求过于频繁，请在 {remain} 秒后重试"

            # 清除窗口外的旧记录
            window_start = now - self._cfg.rate_limit_window_s
            state["timestamps"] = [t for t in state["timestamps"] if t > window_start]

            if len(state["timestamps"]) >= self._cfg.rate_limit_max:
                state["blocked_until"] = now + self._cfg.rate_limit_cooldown_s
                session_state[key] = state
                _slog(self._log, 'warning', "rate_limit_block", session=session_id,
                                  count=len(state["timestamps"]),
                                  max=self._cfg.rate_limit_max,
                                  cooldown=self._cfg.rate_limit_cooldown_s)
                return False, f"请求过于频繁，请稍后再试"

            state["timestamps"].append(now)
            session_state[key] = state
            _slog(self._log, 'info', "rate_limit", session=session_id,
                           count=len(state["timestamps"]),
                           max=self._cfg.rate_limit_max)
            return True, ""

        except Exception as e:
            _slog(self._log, 'error', "gateway_error", component="rate_limiter", error=str(e))
            return True, ""  # fail-open


# ============================================================
# _NoiseReducer
# ============================================================

class _NoiseReducer:
    """Bigram Jaccard 相似度检测近重复查询。"""

    def __init__(self, cfg: GatewayConfig, logger: _StructuredLogger):
        self._cfg = cfg
        self._log = logger

    @staticmethod
    def _bigrams(text: str) -> set:
        """提取 bigram 集合，用于 Jaccard 计算。"""
        chars = list(text)
        return {"".join(chars[i:i+2]) for i in range(len(chars) - 1)}

    def is_duplicate(self, query: str, session_id: str, session_state: dict,
                     tag: str = "") -> bool:
        """与最近 N 条查询比较，超过相似度阈值返回 True。

        tag 区分请求模式（如 deep_thinking 开关）：同 query 不同 tag 不算重复——
        模式切换是用户有意行为（如提示引导的"开深度思考重问"），不应被降噪拦截。
        """
        if not self._cfg.noise_reduction_enabled:
            return False

        try:
            key = f"_gw_noise_{session_id}"
            recent = session_state.get(key, [])
            # ts 用于状态回收（无时间戳则无法判断该键是否已过期，只能永久驻留）
            now = time.time()

            if not recent:
                recent.append({"q": query, "tag": tag, "ts": now})
                if len(recent) > self._cfg.noise_window:
                    recent.pop(0)
                session_state[key] = recent
                return False

            q_bigrams = self._bigrams(query)
            if not q_bigrams:
                return False

            for past in recent:
                if past.get("tag") != tag:
                    continue  # 模式不同 → 切换模式重问是合法行为
                p_bigrams = self._bigrams(past.get("q", ""))
                if not p_bigrams:
                    continue
                intersection = len(q_bigrams & p_bigrams)
                union = len(q_bigrams | p_bigrams)
                if union == 0:
                    continue
                similarity = intersection / union
                if similarity >= self._cfg.noise_similarity:
                    _slog(self._log, 'info', "noise_duplicate", session=session_id,
                                       similarity=round(similarity, 3),
                                       query_hash=hashlib.md5(query.encode()).hexdigest()[:8])
                    return True

            recent.append({"q": query, "tag": tag, "ts": now})
            if len(recent) > self._cfg.noise_window:
                recent.pop(0)
            session_state[key] = recent
            return False

        except Exception as e:
            _slog(self._log, 'error', "gateway_error", component="noise_reducer", error=str(e))
            return False  # fail-open: 放行


# ============================================================
# _TokenBudgetGuard (Strategy A)
# ============================================================

class _TokenBudgetGuard:
    """级联截断：聊天历史 → 检索上下文。禁忌黑名单在 context 头部，始终保留。"""

    # 中英文句子边界分隔符（与 build_index.py 的文本分割器一致）
    # ⚠️ 捕获组是必需的：split 用捕获组时会把分隔符保留为独立元素，
    #    截断重组才能还原原标点与换行（见 _truncate_context）。
    _SENTENCE_SEPS = re.compile(r"([。！？；\n](?![」』）\)]))")

    def __init__(self, cfg: GatewayConfig, logger: _StructuredLogger):
        self._cfg = cfg
        self._log = logger

    def _resolve_ctx_window(self, provider: str | None) -> int:
        """按**实际激活的供应商**取上下文窗口；未知/None 回退本地配置。

        原实现恒用 ollama_num_ctx（8192），云端主链（128k）的上下文预算只有 4505
        tokens，等于把可用窗口白扔 16 倍。见 config.LLM_CONTEXT_WINDOWS。
        """
        if provider and provider != "ollama":
            try:
                from config import LLM_CONTEXT_WINDOWS
                win = LLM_CONTEXT_WINDOWS.get(provider)
                if win:
                    return int(win)
            except Exception:
                pass
        return int(self._cfg.ollama_num_ctx)

    def guard(
        self,
        system_prompt: str,
        history_store: dict,
        session_id: str,
        context: str,
        query: str,
        provider: str | None = None,
    ) -> tuple[str, dict | None]:
        """
        级联截断，返回 (safe_context, budget_info | None)。

        provider：当前实际激活的供应商（如 "dashscope"）；None → 按本地配置估算。
        budget_info 包含 before/after 令牌数，供 UI 展示。
        """
        if not self._cfg.token_budget_enabled:
            return context, None

        try:
            budget = int(self._resolve_ctx_window(provider)
                         * self._cfg.token_budget_ratio)
            sys_tokens = _estimate_tokens(system_prompt)
            q_tokens = _estimate_tokens(query)
            hist_tokens = self._estimate_history(history_store, session_id)
            ctx_tokens = _estimate_tokens(context)

            total_before = sys_tokens + hist_tokens + q_tokens + ctx_tokens

            if total_before <= budget:
                _slog(self._log, 'debug', "token_budget_ok", total=total_before,
                                budget=budget, margin=budget - total_before)
                return context, None

            action = "none"

            # Step 1: 裁剪聊天历史
            hist_limit = self._cfg.chat_max_history_tokens
            if hist_tokens > hist_limit:
                self._trim_history(history_store, session_id, hist_limit)
                hist_tokens = self._estimate_history(history_store, session_id)
                action = "trim_history"

            # Step 2: 仍超预算 → 截断 context 尾部
            remaining = budget - sys_tokens - hist_tokens - q_tokens
            if remaining < self._cfg.token_answer_min:
                remaining = self._cfg.token_answer_min

            if _estimate_tokens(context) > remaining:
                context = self._truncate_context(context, remaining)
                action = "trim_both" if action == "trim_history" else "trim_context"

            total_after = sys_tokens + hist_tokens + q_tokens + _estimate_tokens(context)
            truncate_ratio = 1.0 - (total_after / max(total_before, 1))

            _slog(self._log, 'warning', "token_budget_trim",
                              total_before=total_before, total_after=total_after,
                              budget=budget, action=action,
                              truncate_ratio=round(truncate_ratio, 2))

            budget_info = {
                "total_before": total_before,
                "total_after": total_after,
                "budget": budget,
                "action": action,
            }

            # 截断超过阈值时标记 UI 警告
            if truncate_ratio >= self._cfg.token_budget_truncate_warn:
                budget_info["show_warning"] = True

            return context, budget_info

        except Exception as e:
            _slog(self._log, 'error', "gateway_error", component="token_budget_guard", error=str(e))
            return context, None  # fail-open

    @staticmethod
    def _msg_content(m) -> str:
        """兼容 dict（{type, content} 存储格式）与 BaseMessage。"""
        if isinstance(m, dict):
            return m.get("content", "") or ""
        return getattr(m, "content", "") or ""

    def _estimate_history(self, history_store: dict, session_id: str) -> int:
        """估算指定 session 的聊天历史总 tokens。"""
        try:
            session = history_store.get(session_id)
            if session is None:
                return 0
            msgs = list(session.messages)
            return sum(_estimate_tokens(self._msg_content(m)) for m in msgs)
        except Exception:
            return 0

    def _trim_history(self, history_store: dict, session_id: str, max_tokens: int):
        """裁剪旧消息，从最早开始删除。"""
        try:
            session = history_store.get(session_id)
            if session is None:
                return
            msgs = list(session.messages)
            while len(msgs) > 2:
                total = sum(_estimate_tokens(self._msg_content(m)) for m in msgs)
                if total <= max_tokens:
                    break
                msgs.pop(0)
            session.clear()
            for m in msgs:
                session.add_message(m)
        except Exception:
            pass  # 裁剪失败不影响

    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """按句子边界截断 context 尾部，保留头部（禁忌黑名单在前）。"""
        if not context or max_tokens <= 0:
            return ""

        # 快速路径：context 本身很短
        if _estimate_tokens(context) <= max_tokens:
            return context

        # 按句子边界拆分（捕获组 → 分隔符保留为独立元素：[句1, 分隔符1, 句2, ...]）
        parts = self._SENTENCE_SEPS.split(context)
        kept: list[str] = []
        current_tokens = 0

        for i in range(0, len(parts), 2):
            seg = parts[i]
            sep = parts[i + 1] if i + 1 < len(parts) else ""
            t = _estimate_tokens(seg + sep)
            if current_tokens + t > max_tokens:
                break
            kept.append(seg + sep)
            current_tokens += t

        if not kept:
            # 连一个句子都放不下：字符级截断
            ratio = max_tokens / max(_estimate_tokens(context), 1)
            char_limit = max(50, int(len(context) * ratio))
            return context[:char_limit] + "…"

        # 原样拼回（保留原标点与换行）。原实现用 "。".join(kept) 重组，
        # 而 split 已吃掉分隔符 → ！？；与换行**全部变成「。」**：
        # 禁忌黑名单按 \n 分条，被压成一行跑文后模型可读性显著下降。
        return "".join(kept) + "[内容已截断]"


# ============================================================
# _MemoryGuard (Strategy B)
# ============================================================

class _MemoryGuard:
    """内存看守：可用内存 < 阈值 → 切换到小模型；恢复后切回。"""

    def __init__(self, cfg: GatewayConfig, logger: _StructuredLogger):
        self._cfg = cfg
        self._log = logger
        self._degraded_llm = None
        self._is_degraded = False
        self._banner_shown = False

    @property
    def is_degraded(self) -> bool:
        return self._is_degraded

    def get_active_llm(self, base_llm):
        """返回当前应使用的 LLM 实例（base_llm 或降级小模型）。"""
        if not self._cfg.memory_guard_enabled:
            return base_llm

        try:
            from config import get_available_memory_gb
            avail = get_available_memory_gb()

            _slog(self._log, 'debug', "memory_check", avail_gb=round(avail, 2),
                            threshold=self._cfg.memory_threshold_gb)

            # 触发降级
            if avail < self._cfg.memory_threshold_gb and not self._is_degraded:
                self._is_degraded = True
                self._banner_shown = False
                if self._degraded_llm is None:
                    self._degraded_llm = self._create_degraded_llm()
                _slog(self._log, 'warning', "memory_degrade", avail_gb=round(avail, 2),
                                  threshold=self._cfg.memory_threshold_gb,
                                  model=self._cfg.memory_degraded_model)
                return self._degraded_llm

            # 恢复
            if avail >= self._cfg.memory_recovery_gb and self._is_degraded:
                self._is_degraded = False
                self._banner_shown = False
                _slog(self._log, 'info', "memory_recover", avail_gb=round(avail, 2),
                               recovery=self._cfg.memory_recovery_gb,
                               model="qwen2.5:7b")
                return base_llm

            # 维持当前状态
            if self._is_degraded:
                return self._degraded_llm if self._degraded_llm else base_llm
            return base_llm

        except Exception as e:
            _slog(self._log, 'error', "gateway_error", component="memory_guard", error=str(e))
            return base_llm  # fail-open

    def _create_degraded_llm(self):
        """懒加载降级小模型（统一适配器 OllamaAdapter，去 langchain_ollama 依赖）。"""
        try:
            from llm_adapter import OllamaAdapter
            import config as _cfg
            return OllamaAdapter(
                model=self._cfg.memory_degraded_model,
                base_url=getattr(_cfg, "OLLAMA_BASE_URL", "http://localhost:11434"),
                num_ctx=getattr(_cfg, "OLLAMA_NUM_CTX", 8192),
            )
        except Exception as e:
            _slog(self._log, 'error', "gateway_error", component="memory_guard_degraded_init", error=str(e))
            return None

    def get_banner(self) -> str | None:
        """返回应在 UI 显示的横幅文本（仅首次降级时）。"""
        if self._is_degraded and not self._banner_shown:
            self._banner_shown = True
            return ("⚡ 轻量化应答模式：系统可用内存不足，已自动切换至 1.5B 模型。"
                    "回答可能较简短，内存恢复后将自动切回 7B 模型。")
        return None


# ============================================================
# Gateway 门面
# ============================================================

class Gateway:
    """轻量化网关门面：限流 / 降噪 / 令牌预算 / 内存降级 / 成本上限 / 结构化日志。"""

    # 会话级状态键前缀（回收只处理自己的键，不动调用方放在同一 dict 里的其他东西）
    _STATE_PREFIXES = ("_gw_rate_", "_gw_noise_")

    def __init__(self, cfg: GatewayConfig):
        self._cfg = cfg
        self._log = _StructuredLogger(cfg.log_path, cfg.log_level)
        self._rate_limiter = _SlidingWindowLimiter(cfg, self._log)
        self._noise_reducer = _NoiseReducer(cfg, self._log)
        self._token_guard = _TokenBudgetGuard(cfg, self._log)
        self._memory_guard = _MemoryGuard(cfg, self._log)
        self._cost_guard = CostGuard(
            soft_limit=cfg.cost_soft_limit_tokens,
            hard_limit=cfg.cost_hard_limit_tokens,
            window_s=cfg.cost_window_s,
            path=cfg.cost_guard_path,
            enabled=cfg.cost_guard_enabled,
        )
        self._last_sweep = 0.0

    # ---- Hook 6: 成本上限 ----

    @property
    def cost_state(self) -> str:
        """当前用量档位：ok / degraded（强制 cheapest 层）/ exhausted（拒绝新请求）。"""
        return self._cost_guard.state

    def cost_snapshot(self) -> dict:
        return self._cost_guard.snapshot()

    def check_cost_budget(self) -> tuple[bool, str]:
        """返回 (allowed, reason)。

        独立于 _cfg.enabled —— 成本闸门是安全设施，不该跟着日志/网关开关一起被关掉。
        """
        if not self._cfg.cost_guard_enabled:
            return True, ""
        if self._cost_guard.state == EXHAUSTED:
            snap = self._cost_guard.snapshot()
            _slog(self._log, 'warning', "cost_limit_exhausted",
                  used=snap["total_tokens"], limit=snap["hard_limit"])
            return False, ("服务今日用量已达上限，为避免产生更多费用已暂停应答。"
                           "请稍后再试或联系管理员。")
        return True, ""

    # ---- 会话级状态回收 ----

    @staticmethod
    def _entry_last_active(v) -> float:
        """键的最后活动时间；无法判定返回 0（视为最旧，优先回收）。"""
        try:
            if isinstance(v, dict):                    # 限流状态 {timestamps, blocked_until}
                stamps = [t for t in (v.get("timestamps") or []) if t]
                return max(stamps) if stamps else 0.0
            if isinstance(v, list) and v:              # 降噪状态 [{q, tag, ts}, ...]
                return max(float(e.get("ts", 0.0)) for e in v if isinstance(e, dict))
        except Exception:
            pass
        return 0.0

    def _sweep_state(self, session_state: dict) -> None:
        """回收过期的会话级状态键（限流 / 降噪）。

        为什么必须有：会话隔离后 session_id 按访客生成，每个键都活到进程结束。
        原实现 session_id 恒为 "default_user"，字典永远只有一组键，问题被掩盖——
        修会话隔离而不同步回收，等于把一个隐私 bug 换成内存泄漏。

        两条触发路径（把 O(n) 从每请求摊薄到每 interval 一次）：
          - 超过 interval 未清扫
          - 键数超过硬上限（此时立即清扫，仍超则按最久未活动驱逐）
        """
        try:
            now = time.time()
            over_cap = len(session_state) > self._cfg.state_max_keys
            if not over_cap and now - self._last_sweep < self._cfg.state_sweep_interval_s:
                return
            self._last_sweep = now

            rate_window = self._cfg.rate_limit_window_s
            removed = 0
            for key in list(session_state):
                if not key.startswith(self._STATE_PREFIXES):
                    continue
                v = session_state[key]
                last = self._entry_last_active(v)
                # 限流键：冷却期未过不能回收（否则封禁被提前解除）
                if key.startswith("_gw_rate_"):
                    cooling = isinstance(v, dict) and now < v.get("blocked_until", 0)
                    if not cooling and (last == 0.0 or now - last > rate_window):
                        session_state.pop(key, None)
                        removed += 1
                else:
                    if last == 0.0 or now - last > self._cfg.noise_ttl_s:
                        session_state.pop(key, None)
                        removed += 1

            # 硬上限兜底：清扫后仍超限（如短时大量不同 session_id）→ 驱逐最久未活动
            evicted = 0
            if len(session_state) > self._cfg.state_max_keys:
                ours = [k for k in session_state if k.startswith(self._STATE_PREFIXES)]
                ours.sort(key=lambda k: self._entry_last_active(session_state[k]))
                for key in ours[: len(session_state) - self._cfg.state_max_keys]:
                    session_state.pop(key, None)
                    evicted += 1

            if removed or evicted:
                _slog(self._log, 'info', "gw_state_sweep", expired=removed,
                      evicted=evicted, remaining=len(session_state))
        except Exception as e:
            _slog(self._log, 'error', "gateway_error", component="state_sweep", error=str(e))

    # ---- Hook 0: 速率限制 ----

    def check_rate_limit(self, session_id: str, session_state: dict = None) -> tuple[bool, str]:
        """返回 (allowed, reason)。需要传入 st.session_state。"""
        if not self._cfg.enabled:
            return True, ""
        if session_state is None:
            return True, ""
        # 每个请求都经过这里（限流+降噪的公共入口）→ 作为状态回收的挂载点
        self._sweep_state(session_state)
        return self._rate_limiter.check(session_id, session_state)

    # ---- Hook 1: 请求降噪 ----

    def is_duplicate(self, query: str, session_id: str, session_state: dict = None,
                     tag: str = "") -> bool:
        """返回 True 表示检测到近重复查询。需要传入 st.session_state。

        tag: 请求模式标签（如 "fast"/"deep"）——不同模式的同 query 不判重。
        """
        if not self._cfg.enabled:
            return False
        if session_state is None:
            return False
        return self._noise_reducer.is_duplicate(query, session_id, session_state, tag=tag)

    # ---- Hook 2: 令牌预算守卫 (Strategy A) ----

    def guard_token_budget(
        self,
        system_prompt: str,
        history_store: dict,
        session_id: str,
        context: str,
        query: str,
        provider: str | None = None,
    ) -> str:
        """
        级联截断 → 返回安全 context。
        history_store 应为会话 store（_SessionHistory 字典，与 langchain ChatMessageHistory 接口兼容）。
        provider: 当前实际激活的供应商，决定上下文预算（见 _TokenBudgetGuard._resolve_ctx_window）。
        内部可能裁剪 history_store 中的消息。
        """
        if not self._cfg.enabled:
            return context
        safe_ctx, budget_info = self._token_guard.guard(
            system_prompt, history_store, session_id, context, query, provider
        )
        self._last_budget_info = budget_info
        return safe_ctx

    # ---- Hook 3: 内存守卫 (Strategy B) ----

    def get_active_llm(self, base_llm):
        """返回当前应使用的 LLM 实例（7B 或降级 1.5B）。"""
        if not self._cfg.enabled:
            return base_llm
        return self._memory_guard.get_active_llm(base_llm)

    # ---- Hook 4: 结构化日志 ----

    def log_cycle(self, event: str, **kwargs):
        """记录一次请求周期的结构化日志。

        约定：**第一个参数是事件名**（写入日志的 `event` 字段），不是日志级别——
        级别固定为 INFO（要分级请用 _StructuredLogger 的 debug/warning/error）。

        ⚠️ 曾有 4 处调用点误把 "info"/"error"/"warning" 当作第一个参数传进来，
        真实事件名塞在 `event_detail` 里 → 日志里出现 `event: "error"` 这种条目，
        `grep 'event=="retrieval_error"'` 一条都搜不到。实测排查成本上限降级时被此坑到。
        """
        if not self._cfg.enabled:
            return
        _slog(self._log, 'info', event, **kwargs)

    # ---- Hook 5: 完整日志（康养 Demo 扩展：query/检索片段/prompt/输出/token 消耗）----

    def log_usage(self, request_id: str, role: str, usage) -> None:
        # 记账在 enabled 判断**之外**：成本核算是安全设施，不能跟着日志开关一起关掉，
        # 否则关掉网关日志就出现一条完全不记账的调用路径。
        self._cost_guard.record(usage)
        if self._cfg.enabled:
            self._log.log_usage(request_id, role, usage)

    def log_retrieval(self, request_id: str, query: str, docs: list) -> None:
        if self._cfg.enabled:
            self._log.log_retrieval(request_id, query, docs)

    def log_prompt(self, request_id: str, role: str, prompt: str) -> None:
        if self._cfg.enabled:
            self._log.log_prompt(request_id, role, prompt)

    def log_answer(self, request_id: str, answer: str, censor: dict | None = None) -> None:
        if self._cfg.enabled:
            self._log.log_answer(request_id, answer, censor)

    # ---- 公开属性（供 UI 使用） ----

    @property
    def is_degraded(self) -> bool:
        return self._memory_guard.is_degraded

    @property
    def last_budget_info(self) -> dict | None:
        return getattr(self, "_last_budget_info", None)

    def get_degraded_banner(self) -> str | None:
        return self._memory_guard.get_banner()
