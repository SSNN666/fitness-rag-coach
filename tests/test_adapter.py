"""llm_adapter 单测：错误分类表 + 降级链（mock 适配器，不发真实请求）。"""
import pytest

from llm_adapter import (
    BaseLLMAdapter, FallbackChain, LLMErrorKind, LLMResponse, UsageInfo,
    classify_error, build_llm,
)


class _HTTPErr(Exception):
    """模拟带 HTTP 状态码的异常。"""

    def __init__(self, status_code, message=""):
        self.status_code = status_code
        super().__init__(message)


class _MockAdapter(BaseLLMAdapter):
    name = "mock"

    def __init__(self, content="ok", fail_spec=None):
        self._fail_spec = list(fail_spec or [])  # 每次调用弹出一个异常
        self._content = content
        self.calls = []

    def invoke(self, messages, *, temperature=0.7, max_tokens=None,
               timeout=None, request_id=None):
        self.calls.append(messages)
        if self._fail_spec:
            raise self._fail_spec.pop(0)
        return LLMResponse(content=self._content,
                           usage=UsageInfo(provider=self.name, total_tokens=7))

    def stream(self, messages, *, temperature=0.7, max_tokens=None,
               timeout=None, request_id=None):
        yield self._content


class TestClassifyError:
    @pytest.mark.parametrize("exc,kind", [
        (_HTTPErr(429), LLMErrorKind.RATE_LIMITED),
        (_HTTPErr(401), LLMErrorKind.AUTH),
        (_HTTPErr(403, "model not enabled"), LLMErrorKind.QUOTA),
        (_HTTPErr(500), LLMErrorKind.SERVER),
        (_HTTPErr(400, "code: Arrearage"), LLMErrorKind.QUOTA),
        (_HTTPErr(400, "maximum context length exceeded"), LLMErrorKind.CONTEXT_OVERFLOW),
        (ConnectionError("connection refused"), LLMErrorKind.NETWORK),
        (TimeoutError("Request timed out"), LLMErrorKind.TIMEOUT),
        (ValueError("something else"), LLMErrorKind.UNKNOWN),
    ])
    def test_mapping(self, exc, kind):
        assert classify_error(exc) == kind


class TestFallbackChain:
    def test_fallback_on_network_error(self, monkeypatch):
        monkeypatch.setattr("llm_adapter.time.sleep", lambda s: None)
        primary = _MockAdapter(fail_spec=[ConnectionError("down")])
        backup = _MockAdapter(content="backup-answer")
        chain = FallbackChain([primary, backup], max_retries=0)
        r = chain.invoke("hi")
        assert r.content == "backup-answer"
        assert r.fallback is True

    def test_retry_then_success(self, monkeypatch):
        monkeypatch.setattr("llm_adapter.time.sleep", lambda s: None)
        primary = _MockAdapter(fail_spec=[_HTTPErr(429)])
        chain = FallbackChain([primary], max_retries=1)
        r = chain.invoke("hi")
        assert r.content == "ok"
        assert len(primary.calls) == 2      # 429 重试一次后成功
        assert r.fallback is False

    def test_quota_no_retry_goes_straight_to_fallback(self, monkeypatch):
        monkeypatch.setattr("llm_adapter.time.sleep", lambda s: None)
        primary = _MockAdapter(fail_spec=[_HTTPErr(400, "Arrearage 欠费")])
        backup = _MockAdapter(content="backup-answer")
        chain = FallbackChain([primary, backup], max_retries=3)
        r = chain.invoke("hi")
        assert len(primary.calls) == 1      # 额度不足不盲目重试
        assert r.content == "backup-answer"

    def test_context_overflow_truncates_history(self, monkeypatch):
        primary = _MockAdapter(fail_spec=[_HTTPErr(400, "maximum context length")])
        chain = FallbackChain([primary], max_retries=0)
        msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        r = chain.invoke(msgs)
        assert r.content == "ok"
        # 第二次调用应只剩 system + 最后一条 user
        assert primary.calls[1] == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Q2"},
        ]

    def test_all_fail_returns_fallback_text(self, monkeypatch):
        monkeypatch.setattr("llm_adapter.time.sleep", lambda s: None)
        primary = _MockAdapter(fail_spec=[ValueError("boom")])
        chain = FallbackChain([primary], max_retries=0)
        r = chain.invoke("hi")
        assert "服务暂时不可用" in r.content
        assert r.error_kind == "unknown"

    def test_stream_midway_failure_marker(self, monkeypatch):
        monkeypatch.setattr("llm_adapter.time.sleep", lambda s: None)

        class _FailingStream(_MockAdapter):
            def stream(self, messages, **kw):
                yield "部分内容"
                raise ConnectionError("cut")

        chain = FallbackChain([_FailingStream(), _MockAdapter(content="后续")], max_retries=0)
        pieces = list(chain.stream("hi"))
        assert "部分内容" in pieces[0]
        assert any("备用模型" in p for p in pieces)
        assert pieces[-1] == "后续"


class TestThinkingFieldIsolation:
    """enable_thinking 是 DashScope 混合思考模型专属字段：其他 OpenAI 兼容端点不得下发。"""

    def test_deepseek_qianfan_no_extra_body(self):
        from llm_adapter import DeepSeekAdapter, QianfanAdapter
        for cls in (DeepSeekAdapter, QianfanAdapter):
            a = cls(model="m", api_key="k", enable_thinking=False)
            kw = a._base_kwargs([{"role": "user", "content": "hi"}], 0.7, None, False, None)
            assert "extra_body" not in kw, f"{cls.__name__} 不应下发 enable_thinking"

    def test_dashscope_sends_thinking_controls(self):
        from llm_adapter import DashScopeAdapter
        off = DashScopeAdapter(model="m", api_key="k", enable_thinking=False)
        kw = off._base_kwargs([{"role": "user", "content": "hi"}], 0.7, None, False, None)
        assert kw["extra_body"] == {"enable_thinking": False}
        on = DashScopeAdapter(model="m", api_key="k", enable_thinking=True, thinking_budget=2048)
        kw2 = on._base_kwargs([{"role": "user", "content": "hi"}], 0.7, None, False, None)
        assert kw2["extra_body"] == {"enable_thinking": True, "thinking_budget": 2048}


class TestBuildLLM:
    def test_unconfigured_cloud_providers_skipped(self, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "DASHSCOPE_API_KEY", "")
        monkeypatch.setattr(cfg, "QIANFAN_API_KEY", "")
        monkeypatch.setattr(cfg, "LLM_PROVIDER_PRIMARY", "dashscope")
        monkeypatch.setattr(cfg, "LLM_LOCAL_FALLBACK_ENABLED", True)
        llm = build_llm("chat")
        assert llm.active_providers == ["ollama"]

    def test_local_fallback_disabled_gives_empty_chain(self, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "DASHSCOPE_API_KEY", "")
        monkeypatch.setattr(cfg, "QIANFAN_API_KEY", "")
        monkeypatch.setattr(cfg, "LLM_LOCAL_FALLBACK_ENABLED", False)
        llm = build_llm("chat")
        assert llm.active_providers == []

    def test_provider_override(self, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "DASHSCOPE_API_KEY", "")
        monkeypatch.setattr(cfg, "QIANFAN_API_KEY", "")
        llm = build_llm("chat", provider="ollama")
        assert llm.active_providers == ["ollama"]
