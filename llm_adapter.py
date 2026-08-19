"""
llm_adapter.py —— 统一大模型适配器 + 多供应商降级链
======================================================
主链路云端大模型（阿里云百炼 DashScope / DeepSeek / 百度千帆 ERNIE），本地 Ollama 兜底。

设计要点:
  - classify_error() 错误分类表：超时 / 429 限流 / 额度不足 / 上下文超长 / 鉴权 / 5xx / 网络
    → 驱动"重试退避 → 下一级供应商"的降级决策（额度不足/鉴权不盲目重试，直接降级）
  - LLMResponse.content 鸭子类型兼容旧代码（hyde.py / reranker.py / fact_checker.py
    只读 response.content，零改动）
  - usage 上报回调（on_usage），由 gateway 结构化日志消费
  - build_llm(role) 工厂：按角色取模型映射，未配 Key 的供应商自动跳过
  - 工具调用支持（OpenAI tools 协议）：invoke/stream_events 均可传 tools/tool_choice/stop，
    供 LangGraph 桥接层复用（详见 StreamChunk / LLMResponse.tool_calls）
  - 多模态 VL（invoke_vision）：仅云适配器，本地 Ollama 无 VL → 明确提示

用法:
    from llm_adapter import build_llm
    llm = build_llm("chat")                        # DashScope → DeepSeek → 千帆(可选) → Ollama
    resp = llm.invoke("腰突能深蹲吗")                # LLMResponse(content, usage, fallback)
    for chunk in llm.stream("..."): print(chunk)   # 流式生成（纯文本，兼容旧接口）
    for ev in llm.stream_events("..."): ...        # 流式事件（文本/工具调用/usage/降级标记）
    llm.invoke_vision(img_bytes, "解读这份报告")     # 多模态（仅云适配器，本地无 VL）
"""

from __future__ import annotations

import base64
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator

import httpx

_log = logging.getLogger("llm_adapter")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class UsageInfo:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    provider: str = ""
    latency_ms: int = 0


@dataclass
class LLMResponse:
    content: str
    usage: UsageInfo | None = None
    fallback: bool = False            # True = 来自降级供应商
    error_kind: str | None = None
    tool_calls: list[dict] | None = None   # [{name, args:dict, id}]，无工具调用时为 None


@dataclass
class StreamChunk:
    """流式事件协议：文本增量 / 完整工具调用（仅 finish 时） / usage / 降级切换标记。"""
    text: str = ""
    tool_calls: list[dict] | None = None    # 完整 tool_calls，仅在 finish 时给出
    finish_reason: str | None = None
    usage: UsageInfo | None = None
    provider: str = ""
    fallback_switch: bool = False           # 本块前发生了流中降级切换


class LLMErrorKind(str, Enum):
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    QUOTA = "quota"
    CONTEXT_OVERFLOW = "context_overflow"
    AUTH = "auth"
    SERVER = "server"
    NETWORK = "network"
    UNKNOWN = "unknown"


# 可重试的错误类型（额度不足/鉴权/上下文超长不做盲目重试）
_RETRYABLE = {LLMErrorKind.TIMEOUT, LLMErrorKind.RATE_LIMITED,
              LLMErrorKind.SERVER, LLMErrorKind.NETWORK}


def classify_error(exc: Exception) -> LLMErrorKind:
    """错误分类表：异常类型名 + HTTP 状态码 + 消息关键词 → LLMErrorKind。"""
    name = type(exc).__name__
    text = str(exc).lower()
    status = getattr(exc, "status_code", None)

    # 超时（openai SDK APITimeoutError / httpx ReadTimeout / 消息关键词）
    if name in ("APITimeoutError", "ReadTimeout", "ConnectTimeout",
                "WriteTimeout", "TimeoutException") or "timeout" in text or "timed out" in text:
        return LLMErrorKind.TIMEOUT
    # 429 限流
    if status == 429 or name == "RateLimitError" or "throttling" in text or "rate limit" in text:
        return LLMErrorKind.RATE_LIMITED
    # 鉴权失败
    if status == 401 or name == "AuthenticationError" or \
            "invalid api key" in text or "invalidapikey" in text.replace(" ", ""):
        return LLMErrorKind.AUTH
    # 额度不足（DashScope: 400 code=Arrearage；通用关键词）
    if "arrearage" in text or "欠费" in text or "insufficient_quota" in text or \
            "quota exceeded" in text or "额度不足" in text:
        return LLMErrorKind.QUOTA
    # 千帆 403 = Key 有效但模型未启用/无调用额度 → 直接降级
    if status == 403:
        return LLMErrorKind.QUOTA
    # 上下文超长
    if "maximum context" in text or "too many tokens" in text or "input length" in text or \
            ("context" in text and ("length" in text or "exceed" in text or "超长" in text or "太长" in text)):
        return LLMErrorKind.CONTEXT_OVERFLOW
    # 5xx 服务端
    if status and 500 <= status < 600:
        return LLMErrorKind.SERVER
    # 网络连接
    if name in ("APIConnectionError", "ConnectionError", "ConnectError",
                "NetworkError", "RemoteProtocolError") or "connection" in text:
        return LLMErrorKind.NETWORK
    return LLMErrorKind.UNKNOWN


def _extract_retry_after(exc: Exception) -> float | None:
    """从异常响应头解析 Retry-After（秒）。openai SDK APIStatusError / httpx 异常均带 .response。"""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    val = headers.get("retry-after")
    if val is None:
        return None
    try:
        return max(0.0, float(val))
    except (TypeError, ValueError):
        return None


# ============================================================
# 消息规范化（str / OpenAI dict / langchain Message → OpenAI 格式）
# ============================================================

def _normalize_messages(messages) -> list[dict]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out = []
    for m in messages:
        if isinstance(m, dict):
            out.append(m)
        else:
            role = {"human": "user", "ai": "assistant", "system": "system"}.get(
                getattr(m, "type", ""), "user")
            out.append({"role": role, "content": getattr(m, "content", "") or ""})
    return out


def _truncate_messages(messages) -> list:
    """上下文超长时兜底截断：保留 system 消息 + 最后一条 user 消息。"""
    msgs = _normalize_messages(messages)
    kept = [m for m in msgs if m.get("role") == "system"]
    tail_user = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
    if tail_user is not None:
        kept.append(tail_user)
    return kept or msgs


# ============================================================
# 适配器抽象与实现
# ============================================================

class BaseLLMAdapter(ABC):
    """供应商适配器抽象。invoke 抛异常由 FallbackChain 分类处理；stream 逐块 yield。

    工具参数（tools/tool_choice/stop）为可选扩展：FallbackChain 仅在显式传入时下发，
    旧实现（仅实现 stream）经 stream_events 默认包装自动兼容。
    """

    name: str = "base"

    @abstractmethod
    def invoke(self, messages, *, temperature: float = 0.7, max_tokens: int | None = None,
               timeout: float | None = None, request_id: str | None = None,
               tools: list[dict] | None = None, tool_choice: str | None = None,
               stop: list[str] | None = None) -> LLMResponse:
        ...

    @abstractmethod
    def stream(self, messages, *, temperature: float = 0.7, max_tokens: int | None = None,
               timeout: float | None = None, request_id: str | None = None) -> Iterator[str]:
        ...

    def stream_events(self, messages, *, temperature: float = 0.7, max_tokens: int | None = None,
                      timeout: float | None = None, request_id: str | None = None,
                      tools: list[dict] | None = None, tool_choice: str | None = None,
                      stop: list[str] | None = None) -> Iterator[StreamChunk]:
        """默认实现：包装 stream() → 纯文本事件块（旧适配器零改动）。"""
        for piece in self.stream(messages, temperature=temperature, max_tokens=max_tokens,
                                 timeout=timeout, request_id=request_id):
            yield StreamChunk(text=piece, provider=self.name)

    def invoke_vision(self, image_bytes: bytes, prompt: str,
                      mime: str = "image/jpeg", **kwargs) -> LLMResponse:
        raise NotImplementedError(f"{self.name} 不支持多模态")


class OpenAICompatAdapter(BaseLLMAdapter):
    """OpenAI 兼容协议适配器基类（DashScope / DeepSeek / 千帆共用 openai SDK，
    仅 base_url/key/模型名不同）。"""

    DEFAULT_BASE_URL = ""
    DEFAULT_MODEL = ""
    SUPPORTS_STREAM_USAGE = True        # 兼容端点是否支持 stream_options.include_usage
    SUPPORTS_THINKING = False           # enable_thinking 仅 DashScope 混合思考模型支持

    def __init__(self, model: str | None = None, api_key: str = "",
                 base_url: str | None = None, timeout: float = 60.0,
                 enable_thinking: bool = True, thinking_budget: int | None = None):
        from openai import OpenAI
        self._model = model or self.DEFAULT_MODEL
        self._enable_thinking = enable_thinking
        self._thinking_budget = thinking_budget
        self._client = OpenAI(
            api_key=api_key, base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout, max_retries=0,  # 重试统一由 FallbackChain 做
        )

    def _base_kwargs(self, messages, temperature, max_tokens, stream, request_id,
                     timeout=None, tools=None, tool_choice=None, stop=None) -> dict:
        kw = dict(model=self._model, messages=_normalize_messages(messages),
                  temperature=temperature, stream=stream)
        if max_tokens:
            kw["max_tokens"] = max_tokens
        if timeout is not None:
            kw["timeout"] = timeout          # per-call 超时（SDK 支持，A1.1 修复）
        if tools:
            kw["tools"] = tools
        if tool_choice:
            kw["tool_choice"] = tool_choice
        if stop:
            kw["stop"] = stop
        if stream and self.SUPPORTS_STREAM_USAGE:
            kw["stream_options"] = {"include_usage": True}   # A1.2：流式路径收集 usage
        if self.SUPPORTS_THINKING:
            # enable_thinking 是 DashScope 混合思考模型专属字段：DeepSeek/千帆等
            # OpenAI 兼容端点不认识该字段，原样下发可能被拒绝 → 按适配器能力分流
            if not self._enable_thinking:
                kw["extra_body"] = {"enable_thinking": False}  # 混合思考模型关思考，大幅提速
            elif self._thinking_budget:
                # 思考预算上限：max_tokens 不约束思考 token（MaaS 分开计数），无预算会失控（实测 3 分钟）
                kw["extra_body"] = {"enable_thinking": True,
                                    "thinking_budget": self._thinking_budget}
        if request_id:
            kw["extra_headers"] = {"X-Request-ID": request_id}
        return kw

    @staticmethod
    def _parse_tool_calls(msg) -> list[dict] | None:
        """openai SDK 响应的 message.tool_calls → [{name, args, id}]。args 解析失败保留原始串。"""
        raw = getattr(msg, "tool_calls", None)
        if not raw:
            return None
        out: list[dict] = []
        for i, tc in enumerate(raw):
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") if fn else ""
            args_raw = getattr(fn, "arguments", "") if fn else ""
            try:
                args = json.loads(args_raw) if args_raw else {}
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": args_raw}
            out.append({"name": name, "args": args,
                        "id": getattr(tc, "id", "") or f"{i}"})
        return out

    def invoke(self, messages, *, temperature=0.7, max_tokens=None,
               timeout=None, request_id=None,
               tools=None, tool_choice=None, stop=None) -> LLMResponse:
        t0 = time.time()
        kw = self._base_kwargs(messages, temperature, max_tokens, False, request_id,
                               timeout=timeout, tools=tools, tool_choice=tool_choice,
                               stop=stop)
        resp = self._client.chat.completions.create(**kw)
        msg = resp.choices[0].message if resp.choices else None
        content = (getattr(msg, "content", "") or "") if msg else ""
        u = getattr(resp, "usage", None)
        return LLMResponse(content=content, usage=UsageInfo(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
            model=resp.model, provider=self.name,
            latency_ms=int((time.time() - t0) * 1000),
        ), tool_calls=self._parse_tool_calls(msg))

    def stream_events(self, messages, *, temperature=0.7, max_tokens=None,
                      timeout=None, request_id=None,
                      tools=None, tool_choice=None, stop=None) -> Iterator[StreamChunk]:
        t0 = time.time()
        kw = self._base_kwargs(messages, temperature, max_tokens, True, request_id,
                               timeout=timeout, tools=tools, tool_choice=tool_choice,
                               stop=stop)
        tc_buf: dict[int, dict] = {}
        model_name = self._model
        for chunk in self._client.chat.completions.create(**kw):
            # include_usage 时部分供应商把 usage 放在无 choices 的末块
            if not chunk.choices:
                u = getattr(chunk, "usage", None)
                if u:
                    yield StreamChunk(usage=UsageInfo(
                        prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                        total_tokens=getattr(u, "total_tokens", 0) or 0,
                        model=model_name, provider=self.name,
                        latency_ms=int((time.time() - t0) * 1000),
                    ), provider=self.name)
                continue
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                if getattr(delta, "content", None):
                    yield StreamChunk(text=delta.content, provider=self.name)
                for tc in (getattr(delta, "tool_calls", None) or []):
                    buf = tc_buf.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        buf["id"] += tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if fn.name:
                            buf["name"] += fn.name
                        if fn.arguments:
                            buf["arguments"] += fn.arguments
            finish = getattr(choice, "finish_reason", None)
            u = getattr(chunk, "usage", None)
            if finish or u:
                tool_calls = None
                if tc_buf:
                    tool_calls = []
                    for i in sorted(tc_buf):
                        b = tc_buf[i]
                        try:
                            args = json.loads(b["arguments"]) if b["arguments"] else {}
                        except (json.JSONDecodeError, TypeError):
                            args = {"_raw": b["arguments"]}
                        tool_calls.append({"name": b["name"], "args": args,
                                           "id": b["id"] or f"{self.name}-{i}"})
                usage = None
                if u:
                    usage = UsageInfo(
                        prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                        total_tokens=getattr(u, "total_tokens", 0) or 0,
                        model=model_name, provider=self.name,
                        latency_ms=int((time.time() - t0) * 1000),
                    )
                yield StreamChunk(text="", tool_calls=tool_calls, finish_reason=finish,
                                  usage=usage, provider=self.name)

    def stream(self, messages, *, temperature=0.7, max_tokens=None,
               timeout=None, request_id=None) -> Iterator[str]:
        for ev in self.stream_events(messages, temperature=temperature,
                                     max_tokens=max_tokens, timeout=timeout,
                                     request_id=request_id):
            if ev.text:
                yield ev.text

    def invoke_vision(self, image_bytes, prompt, mime="image/jpeg", **kwargs) -> LLMResponse:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}]
        return self.invoke(msgs, **kwargs)


class DashScopeAdapter(OpenAICompatAdapter):
    """阿里云百炼 DashScope（OpenAI 兼容模式）。"""

    name = "dashscope"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "qwen-plus"
    SUPPORTS_THINKING = True   # qwen3.x 混合思考模型支持 enable_thinking/thinking_budget


class DeepSeekAdapter(OpenAICompatAdapter):
    """DeepSeek 开放平台（OpenAI 兼容，https://api.deepseek.com/v1）。无多模态。"""

    name = "deepseek"
    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-chat"


class QianfanAdapter(OpenAICompatAdapter):
    """百度千帆 ModelBuilder（OpenAI 兼容模式 v2，Bearer bce-v3/ALTAK-... Key）。

    兼容端点不支持 stream_options.include_usage → SUPPORTS_STREAM_USAGE=False。
    """

    name = "qianfan"
    DEFAULT_BASE_URL = "https://qianfan.baidubce.com/v2"
    DEFAULT_MODEL = "ernie-4.5-turbo-128k"
    SUPPORTS_STREAM_USAGE = False


class OllamaAdapter(BaseLLMAdapter):
    """httpx 直连 Ollama /api/chat（不依赖 langchain_ollama，错误可控）。

    工具调用：Ollama 原生 tools 协议；tool_choice 仅在 "auto"/None 时下发（部分版本不支持）。
    """

    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434",
                 num_ctx: int = 8192, num_predict: int | None = None, timeout: float = 300.0):
        self._model = model
        self._base = base_url.rstrip("/")
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._timeout = timeout

    def _payload(self, messages, temperature, max_tokens,
                 tools=None, tool_choice=None, stop=None) -> dict:
        options = {"num_ctx": self._num_ctx, "temperature": temperature}
        predict = max_tokens or self._num_predict
        if predict:
            options["num_predict"] = predict
        if stop:
            options["stop"] = stop
        payload = {"model": self._model, "messages": _normalize_messages(messages),
                   "options": options}
        if tools:
            payload["tools"] = tools
        if tool_choice in ("auto", None):
            # Ollama 部分版本不支持 tool_choice 字段 → 保持默认行为，不显式下发
            pass
        return payload

    @staticmethod
    def _parse_tool_calls(msg: dict, prefix: str = "ollama") -> list[dict] | None:
        """Ollama 响应的 message.tool_calls → [{name, args, id}]。"""
        raw = msg.get("tool_calls") or []
        if not raw:
            return None
        out: list[dict] = []
        for i, tc in enumerate(raw):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            out.append({"name": fn.get("name", ""), "args": args,
                        "id": tc.get("id") or f"{prefix}-{i}"})
        return out

    def invoke(self, messages, *, temperature=0.7, max_tokens=None,
               timeout=None, request_id=None,
               tools=None, tool_choice=None, stop=None) -> LLMResponse:
        t0 = time.time()
        payload = self._payload(messages, temperature, max_tokens,
                                tools=tools, tool_choice=tool_choice, stop=stop)
        payload["stream"] = False
        r = httpx.post(f"{self._base}/api/chat", json=payload,
                       timeout=timeout or self._timeout)
        r.raise_for_status()
        data = r.json()
        message = data.get("message") or {}
        content = message.get("content", "")
        prompt_tok = data.get("prompt_eval_count", 0) or 0
        eval_tok = data.get("eval_count", 0) or 0
        return LLMResponse(content=content, usage=UsageInfo(
            prompt_tokens=prompt_tok, completion_tokens=eval_tok,
            total_tokens=prompt_tok + eval_tok,
            model=self._model, provider=self.name,
            latency_ms=int((time.time() - t0) * 1000),
        ), tool_calls=self._parse_tool_calls(message))

    def stream_events(self, messages, *, temperature=0.7, max_tokens=None,
                      timeout=None, request_id=None,
                      tools=None, tool_choice=None, stop=None) -> Iterator[StreamChunk]:
        t0 = time.time()
        payload = self._payload(messages, temperature, max_tokens,
                                tools=tools, tool_choice=tool_choice, stop=stop)
        payload["stream"] = True
        done_msg: dict = {}
        with httpx.stream("POST", f"{self._base}/api/chat", json=payload,
                          timeout=timeout or self._timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    done_msg = data
                    break
                piece = (data.get("message") or {}).get("content", "")
                if piece:
                    yield StreamChunk(text=piece, provider=self.name)
        # done 帧：完整 tool_calls + usage
        message = done_msg.get("message") or {}
        prompt_tok = done_msg.get("prompt_eval_count", 0) or 0
        eval_tok = done_msg.get("eval_count", 0) or 0
        yield StreamChunk(
            text="",
            tool_calls=self._parse_tool_calls(message),
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=prompt_tok, completion_tokens=eval_tok,
                            total_tokens=prompt_tok + eval_tok,
                            model=self._model, provider=self.name,
                            latency_ms=int((time.time() - t0) * 1000)),
            provider=self.name,
        )

    def stream(self, messages, *, temperature=0.7, max_tokens=None,
               timeout=None, request_id=None) -> Iterator[str]:
        for ev in self.stream_events(messages, temperature=temperature,
                                     max_tokens=max_tokens, timeout=timeout,
                                     request_id=request_id):
            if ev.text:
                yield ev.text


# ============================================================
# 云端 Embedding 适配器
# ============================================================

class CloudEmbeddings:
    """云端 Embedding（OpenAI 兼容 /embeddings 接口）。

    注意：切换 Embedding 供应商必须重建索引——不同模型的向量空间不兼容，
    同维度（dimensions=768）只保证 schema 兼容，不保证相似度语义兼容。
    """

    def __init__(self, model: str = "qwen3.7-text-embedding", api_key: str = "",
                 base_url: str = "", dimensions: int = 768, timeout: float = 30.0):
        from openai import OpenAI
        self._model = model
        self._dimensions = dimensions
        self._client = OpenAI(api_key=api_key, base_url=base_url,
                              timeout=timeout, max_retries=2)

    def _call(self, texts: list[str]) -> list[list[float]]:
        kw: dict = dict(model=self._model, input=texts)
        if self._dimensions:
            kw["extra_body"] = {"dimensions": self._dimensions}
        resp = self._client.embeddings.create(**kw)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self._call([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts)


def build_embeddings(provider: str | None = None, settings=None):
    """按配置构建 Embedding：cloud（MaaS，无本地并发队列）| ollama（全本地）。

    未配置云 Key 时自动回退本地 Ollama + 告警日志。
    settings: 配置对象（pydantic-settings 实例或 config 模块）。None = 使用本仓库 config 模块。
    """
    def _cfg(key, default):
        if settings is not None:
            return getattr(settings, key, default)
        import config as cfg
        return getattr(cfg, key, default)

    provider = provider or _cfg("EMBEDDING_PROVIDER", "ollama")
    if provider == "cloud" and _cfg("DASHSCOPE_API_KEY", ""):
        return CloudEmbeddings(
            model=_cfg("EMBEDDING_CLOUD_MODEL", "qwen3.7-text-embedding"),
            api_key=_cfg("DASHSCOPE_API_KEY", ""),
            base_url=_cfg("DASHSCOPE_BASE_URL", ""),
            dimensions=_cfg("EMBEDDING_CLOUD_DIM", 768),
            timeout=_cfg("LLM_TIMEOUT", 60.0),
        )
    _log.warning("embedding_fallback_ollama provider=%s (云 Key 未配置)", provider)
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=_cfg("EMBEDDING_MODEL", "nomic-embed-text"))


# ============================================================
# 降级链（核心）
# ============================================================

class FallbackChain:
    """多供应商降级链：逐级尝试，每级内重试退避，全部失败返回兜底文案。

    错误分类驱动决策：
      - TIMEOUT / RATE_LIMITED / SERVER / NETWORK → 退避重试（429 有 Retry-After 则按其值），
        耗尽后降级下一级
      - CONTEXT_OVERFLOW → 截断历史重试一次，再失败降级
      - QUOTA / AUTH / UNKNOWN → 不重试，直接降级下一级

    工具参数（tools/tool_choice/stop）仅在显式传入时下发给适配器，
    保证旧适配器（签名不含这些参数）继续可用。
    """

    def __init__(self, adapters: list[BaseLLMAdapter | None], role: str = "chat",
                 max_retries: int = 2, backoff: tuple = (1.0, 2.0),
                 default_temperature: float = 0.7, default_max_tokens: int | None = None,
                 on_usage: Callable[[str, UsageInfo], None] | None = None):
        self._adapters = [a for a in adapters if a is not None]
        self._role = role
        self._max_retries = max(0, max_retries)
        self._backoff = backoff
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._on_usage = on_usage

    @staticmethod
    def _adapter_kwargs(temperature, max_tokens, timeout, request_id,
                        tools, tool_choice, stop) -> dict:
        kw = dict(temperature=temperature, max_tokens=max_tokens,
                  timeout=timeout, request_id=request_id)
        if tools is not None:
            kw["tools"] = tools
        if tool_choice is not None:
            kw["tool_choice"] = tool_choice
        if stop is not None:
            kw["stop"] = stop
        return kw

    def invoke(self, messages, *, temperature: float | None = None,
               max_tokens: int | None = None, timeout: float | None = None,
               request_id: str | None = None,
               tools: list[dict] | None = None, tool_choice: str | None = None,
               stop: list[str] | None = None) -> LLMResponse:
        temperature = temperature if temperature is not None else self._default_temperature
        max_tokens = max_tokens if max_tokens is not None else self._default_max_tokens
        last_kind = LLMErrorKind.UNKNOWN

        for level, adapter in enumerate(self._adapters):
            msgs = messages
            attempt = 0
            overflow_retried = False
            while True:
                try:
                    resp = adapter.invoke(msgs, **self._adapter_kwargs(
                        temperature, max_tokens, timeout, request_id,
                        tools, tool_choice, stop))
                    resp.fallback = level > 0
                    if self._on_usage and resp.usage:
                        self._on_usage(self._role, resp.usage)
                    return resp
                except Exception as e:
                    last_kind = classify_error(e)
                    _log.error("llm_error role=%s provider=%s attempt=%d kind=%s err=%s",
                               self._role, adapter.name, attempt, last_kind.value, str(e)[:200])
                    # 上下文超长：截断历史重试一次（独立于普通重试计数）
                    if last_kind == LLMErrorKind.CONTEXT_OVERFLOW and not overflow_retried:
                        overflow_retried = True
                        msgs = _truncate_messages(msgs)
                        continue
                    # 可重试错误：退避重试（429 有 Retry-After 头则按其值），耗尽后降级
                    if last_kind in _RETRYABLE and attempt < self._max_retries:
                        attempt += 1
                        delay = (_extract_retry_after(e)
                                 if last_kind == LLMErrorKind.RATE_LIMITED else None)
                        if delay is None:
                            delay = self._backoff[min(attempt - 1, len(self._backoff) - 1)]
                        time.sleep(delay)
                        continue
                    break  # 该供应商放弃 → 降级下一级

        return LLMResponse(content="[服务暂时不可用，请稍后重试]", error_kind=last_kind.value)

    def stream_events(self, messages, *, temperature: float | None = None,
                      max_tokens: int | None = None, timeout: float | None = None,
                      request_id: str | None = None,
                      tools: list[dict] | None = None, tool_choice: str | None = None,
                      stop: list[str] | None = None) -> Iterator[StreamChunk]:
        """流式事件降级：首块发出前失败 → 换下一级；流中途失败 → 插入标记后换下一级。

        注：已发出的内容无法撤回（流式固有权衡），标记通过 fallback_switch 字段透出。
        本级完整结束时上报 usage（若供应商流式路径提供）。
        """
        temperature = temperature if temperature is not None else self._default_temperature
        max_tokens = max_tokens if max_tokens is not None else self._default_max_tokens

        for level, adapter in enumerate(self._adapters):
            started = False
            last_ev: StreamChunk | None = None
            try:
                for ev in adapter.stream_events(messages, **self._adapter_kwargs(
                        temperature, max_tokens, timeout, request_id,
                        tools, tool_choice, stop)):
                    started = True
                    last_ev = ev
                    yield ev
                if last_ev is not None and last_ev.usage and self._on_usage:
                    self._on_usage(self._role, last_ev.usage)
                return  # 本级完整结束
            except Exception as e:
                kind = classify_error(e)
                _log.error("llm_stream_error role=%s provider=%s kind=%s err=%s",
                           self._role, adapter.name, kind.value, str(e)[:200])
                if started and level + 1 < len(self._adapters):
                    yield StreamChunk(text="\n\n[生成中断，已切换备用模型]\n\n",
                                      provider=adapter.name, fallback_switch=True)
        yield StreamChunk(text="[服务暂时不可用，请稍后重试]", provider="")

    def stream(self, messages, *, temperature: float | None = None,
               max_tokens: int | None = None, timeout: float | None = None,
               request_id: str | None = None) -> Iterator[str]:
        """流式降级（纯文本，兼容旧接口）：stream_events 的薄包装。"""
        for ev in self.stream_events(messages, temperature=temperature,
                                     max_tokens=max_tokens, timeout=timeout,
                                     request_id=request_id):
            if ev.text:
                yield ev.text

    def invoke_vision(self, image_bytes: bytes, prompt: str,
                      mime: str = "image/jpeg", **kwargs) -> LLMResponse:
        """多模态调用：仅尝试支持视觉的云适配器（本地无 VL → 明确提示）。"""
        for level, adapter in enumerate(self._adapters):
            try:
                resp = adapter.invoke_vision(image_bytes, prompt, mime, **kwargs)
                resp.fallback = level > 0
                if self._on_usage and resp.usage:
                    self._on_usage(self._role, resp.usage)
                return resp
            except NotImplementedError:
                continue
            except Exception as e:
                _log.error("llm_vision_error provider=%s kind=%s err=%s",
                           adapter.name, classify_error(e).value, str(e)[:200])
        return LLMResponse(content="[当前无可用多模态模型，无法解析图片]",
                           error_kind="no_vision_provider")

    @property
    def active_providers(self) -> list[str]:
        return [a.name for a in self._adapters]


# ============================================================
# 工厂
# ============================================================

def build_llm(role: str = "chat", *, provider: str | None = None,
              on_usage: Callable[[str, UsageInfo], None] | None = None,
              settings=None, **overrides) -> FallbackChain:
    """按角色构建降级链。

    provider:
      None                  → LLM_PROVIDER_PRIMARY 决定主链
      "ollama"              → 仅本地 Ollama（本地调试 / 评测防烧钱）
      "dashscope"/"deepseek"/"qianfan" → 该云供应商为主 + 链中后续供应商兜底
    未配置 API Key 的云供应商自动跳过。overrides 可覆盖角色默认参数（如 max_tokens）。
    settings: 配置对象（pydantic-settings 实例或 config 模块）。None = 使用本仓库 config 模块。
    """
    def _cfg(key, default=None):
        if settings is not None:
            return getattr(settings, key, default)
        import config as cfg
        return getattr(cfg, key, default)

    roles = _cfg("LLM_ROLES", {})
    role_cfg = dict(roles.get(role, roles.get("chat", {})))
    role_cfg.update(overrides)

    primary = provider or _cfg("LLM_PROVIDER_PRIMARY", "dashscope")
    if primary == "ollama":
        chain_names = ["ollama"]
    else:
        full_chain = _cfg("LLM_FALLBACK_CHAIN",
                          ["dashscope", "deepseek", "qianfan", "ollama"])
        chain_names = full_chain[full_chain.index(primary):] if primary in full_chain \
            else [primary] + full_chain

    cloud_timeout = _cfg("LLM_TIMEOUT", 60.0)
    adapters: list[BaseLLMAdapter | None] = []
    for name in chain_names:
        if name == "dashscope":
            key = _cfg("DASHSCOPE_API_KEY", "")
            adapters.append(DashScopeAdapter(
                model=role_cfg.get("dashscope"), api_key=key,
                base_url=_cfg("DASHSCOPE_BASE_URL", None),
                timeout=cloud_timeout,
                enable_thinking=role_cfg.get("thinking", True),
                thinking_budget=role_cfg.get("thinking_budget"),
            ) if (key and role_cfg.get("dashscope")) else None)
        elif name == "deepseek":
            key = _cfg("DEEPSEEK_API_KEY", "")
            adapters.append(DeepSeekAdapter(
                model=role_cfg.get("deepseek") or "deepseek-chat", api_key=key,
                base_url=_cfg("DEEPSEEK_BASE_URL", None),
                timeout=cloud_timeout,
            ) if (key and role_cfg.get("deepseek")) else None)
        elif name == "qianfan":
            key = _cfg("QIANFAN_API_KEY", "")
            adapters.append(QianfanAdapter(
                model=role_cfg.get("qianfan"), api_key=key,
                base_url=_cfg("QIANFAN_BASE_URL", None),
                timeout=cloud_timeout,
            ) if (key and role_cfg.get("qianfan")) else None)
        elif name == "ollama":
            model = role_cfg.get("ollama")
            adapters.append(OllamaAdapter(
                model=model, base_url=_cfg("OLLAMA_BASE_URL", "http://localhost:11434"),
                num_ctx=_cfg("OLLAMA_NUM_CTX", 8192),
                timeout=_cfg("LLM_LOCAL_TIMEOUT", 300.0),
            ) if (_cfg("LLM_LOCAL_FALLBACK_ENABLED", True) and model) else None)

    return FallbackChain(adapters, role=role,
                         max_retries=_cfg("LLM_MAX_RETRIES", 2),
                         backoff=_cfg("LLM_RETRY_BACKOFF", (1.0, 2.0)),
                         default_temperature=role_cfg.get("temperature", 0.7),
                         default_max_tokens=role_cfg.get("max_tokens"),
                         on_usage=on_usage)


def to_runnable(llm: FallbackChain):
    """包装成 LangChain Runnable，供 eval_testset 的 LCEL 链使用（prompt | to_runnable(llm) | parser）。"""
    from langchain_core.runnables import RunnableLambda

    def _call(prompt_value):
        msgs = prompt_value.to_messages() if hasattr(prompt_value, "to_messages") else prompt_value
        return llm.invoke(msgs).content

    return RunnableLambda(_call)
