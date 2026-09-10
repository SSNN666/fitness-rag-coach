"""
cost_guard.py —— 累计用量与成本上限
====================================
公开服务挂着你的 API Key，被刷就是**真金白银的损失**。本模块给用量加一个
进程级的累计账本和两级闸门：

    ok ──(累计 ≥ 软阈值)──> degraded ──(累计 ≥ 硬阈值)──> exhausted
                              │                              │
              强制走 cheapest 层，服务仍可用           直接拒绝，不再花钱

为什么两级而不是一级：一刀切拒绝会让「今天用得多」直接变成「服务不可用」，
而多数情况下我们希望**继续服务但变便宜**。硬阈值只在真正失控时兜底。

为什么按 token 不按金额：token 数是供应商真实返回的计量（UsageInfo），可核对；
换算成钱需要单价表，而单价随供应商/型号/时段变化，写死在代码里等于造数据。
要用金额就把预算换算成 token：
    预算 token = 预算金额 ÷ 单价(元/千token) × 1000
换算结果填进 config 的 COST_SOFT_LIMIT_TOKENS / COST_HARD_LIMIT_TOKENS。

统计口径：
  - **进程级累计**（不是按访客）——保护的是这个服务的钱包，不是某个用户
  - 按滑动窗口重置（默认 24h），避免长期运行后永久锁死
  - 落盘（可选）：否则重启即清零 = 一条绕过限额的捷径
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass

_log = logging.getLogger("cost_guard")

# 状态常量（对外字符串，避免调用方硬编码字面量）
OK = "ok"
DEGRADED = "degraded"
EXHAUSTED = "exhausted"


@dataclass
class CostState:
    """窗口内的累计用量。"""

    window_start: float = 0.0
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class CostGuard:
    """进程级用量账本 + 两级闸门。所有异常 fail-open（不让计费故障拖垮服务）。"""

    def __init__(self, soft_limit: int = 0, hard_limit: int = 0,
                 window_s: float = 86400.0, path: str | None = None,
                 enabled: bool = True):
        """
        soft_limit / hard_limit: 窗口内 token 上限；0 = 不启用该级
        window_s: 窗口秒数（默认 24h）；超过则清零重新累计
        path: 落盘路径；None = 仅内存（测试/本地调试用）
        """
        self._soft = max(0, int(soft_limit or 0))
        self._hard = max(0, int(hard_limit or 0))
        self._window_s = float(window_s or 86400.0)
        self._path = path
        self._enabled = enabled
        self._lock = threading.Lock()
        self._state = CostState()
        self._load()

    # ---------------- 记账 ----------------

    def record(self, usage) -> None:
        """记一次 LLM 调用（usage 为 UsageInfo；None/异常静默跳过）。

        由 Gateway.log_usage 在每个用量事件上调用——那是所有 LLM 调用的公共漏斗，
        漏挂一处就等于给了一条不记账的调用路径。
        """
        if not self._enabled or usage is None:
            return
        try:
            p = int(getattr(usage, "prompt_tokens", 0) or 0)
            c = int(getattr(usage, "completion_tokens", 0) or 0)
            t = int(getattr(usage, "total_tokens", 0) or 0)
            # 部分供应商的 total 不含缓存命中/思考 token，与 p+c 取大值更保守
            with self._lock:
                self._roll_window_locked()
                self._state.requests += 1
                self._state.prompt_tokens += p
                self._state.completion_tokens += c
                self._state.total_tokens += max(t, p + c)
            self._save()
        except Exception as e:
            _log.error("cost_guard record failed: %s", e)

    def _roll_window_locked(self) -> None:
        """窗口过期 → 清零重来。调用方持有锁。"""
        now = time.time()
        if self._state.window_start <= 0:
            self._state.window_start = now
        elif now - self._state.window_start >= self._window_s:
            self._state = CostState(window_start=now)

    # ---------------- 判定 ----------------

    @property
    def used(self) -> int:
        with self._lock:
            self._roll_window_locked()
            return self._state.total_tokens

    @property
    def state(self) -> str:
        used = self.used
        if self._hard > 0 and used >= self._hard:
            return EXHAUSTED
        if self._soft > 0 and used >= self._soft:
            return DEGRADED
        return OK

    def snapshot(self) -> dict:
        """当前账本快照（/healthz 与调试端点展示用）。"""
        with self._lock:
            self._roll_window_locked()
            st = self._state.as_dict()
        return {
            **st,
            "state": self.state,
            "soft_limit": self._soft,
            "hard_limit": self._hard,
            "window_s": self._window_s,
        }

    def reset(self) -> None:
        """手动清零（运维用；窗口到期会自动重置，正常不需要调）。"""
        with self._lock:
            self._state = CostState(window_start=time.time())
        self._save()

    # ---------------- 落盘 ----------------

    def _load(self) -> None:
        if not self._path:
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            self._state = CostState(**{k: data.get(k, 0) for k in CostState().as_dict()})
            _log.info("cost_guard loaded: used=%d", self._state.total_tokens)
        except FileNotFoundError:
            pass
        except Exception as e:
            # 坏文件按空账本启动（与 session_store 同策略：宁可少算也不阻断启动）
            _log.warning("cost_guard load failed (%s), starting empty", e)
            self._state = CostState()

    def _save(self) -> None:
        if not self._path:
            return
        try:
            with self._lock:
                payload = self._state.as_dict()
            # 原子写：tmp → os.replace，避免进程中断留下半截 JSON
            d = os.path.dirname(os.path.abspath(self._path)) or "."
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                os.replace(tmp, self._path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            # 落盘失败静默降级为内存态：计费持久化不该阻断服务
            _log.error("cost_guard save failed: %s", e)
