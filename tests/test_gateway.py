"""gateway 单测：降噪器模式标签语义 + 令牌预算的上下文窗口解析。"""
from gateway import (
    GatewayConfig,
    _NoiseReducer,
    _SlidingWindowLimiter,
    _TokenBudgetGuard,
)


def _make_reducer():
    return _NoiseReducer(GatewayConfig(noise_reduction_enabled=True), logger=None)


def test_first_query_not_duplicate():
    g = _make_reducer()
    assert g.is_duplicate("腰突能深蹲吗", "s1", {}, tag="fast") is False


def test_same_query_same_tag_blocked():
    g = _make_reducer()
    state = {}
    g.is_duplicate("腰突能深蹲吗", "s2", state, tag="fast")
    assert g.is_duplicate("腰突能深蹲吗", "s2", state, tag="fast") is True


def test_mode_switch_same_query_not_blocked():
    """快速模式问过 → 开深度思考重问同一问题：应放行（用户有意行为）。"""
    g = _make_reducer()
    state = {}
    g.is_duplicate("腰突能深蹲吗", "s3", state, tag="fast")
    assert g.is_duplicate("腰突能深蹲吗", "s3", state, tag="deep") is False


def test_deep_to_fast_switch_also_allowed():
    g = _make_reducer()
    state = {}
    g.is_duplicate("腰突能深蹲吗", "s4", state, tag="deep")
    assert g.is_duplicate("腰突能深蹲吗", "s4", state, tag="fast") is False


def test_different_query_not_blocked():
    g = _make_reducer()
    state = {}
    g.is_duplicate("腰突能深蹲吗", "s5", state, tag="fast")
    assert g.is_duplicate("平板支撑练什么肌群", "s5", state, tag="fast") is False


# ----------------------------------------------------------------
# 速率限制器
# ----------------------------------------------------------------

def test_rate_limiter_blocks_after_max():
    """窗口内第 11 个请求被拒；冷却期内持续拒绝；其他会话不受影响。"""
    limiter = _SlidingWindowLimiter(
        GatewayConfig(rate_limit_enabled=True, rate_limit_max=10,
                      rate_limit_window_s=60, rate_limit_cooldown_s=30),
        logger=None)
    state = {}
    for _ in range(10):
        ok, _ = limiter.check("s1", state)
        assert ok
    ok, reason = limiter.check("s1", state)
    assert not ok and "频繁" in reason
    assert not limiter.check("s1", state)[0]      # 冷却期内持续拒绝
    assert limiter.check("s2", {})[0]             # 其他会话不受影响


def test_rate_limiter_disabled_pass_through():
    limiter = _SlidingWindowLimiter(GatewayConfig(rate_limit_enabled=False), logger=None)
    state = {}
    for _ in range(50):
        assert limiter.check("s1", state)[0]


# ================================================================
# 令牌预算的上下文窗口按「实际激活的供应商」取值
# ================================================================

class TestContextWindowResolution:
    """回归护栏：预算不能退回到「恒按本地 OLLAMA_NUM_CTX(8192)」。

    原实现恒用本地窗口，云端主链（qwen3.7 128k）的上下文预算只有 4505 tokens，
    等于把可用窗口白扔 16 倍——检索到的内容被无谓截断。
    """

    def _guard(self):
        return _TokenBudgetGuard(GatewayConfig.from_module(), logger=None)

    def test_cloud_providers_use_their_own_window(self):
        from config import LLM_CONTEXT_WINDOWS
        g = self._guard()
        for p in ("dashscope", "deepseek", "qianfan"):
            assert g._resolve_ctx_window(p) == LLM_CONTEXT_WINDOWS[p], p

    def test_ollama_uses_local_config(self):
        g = self._guard()
        assert g._resolve_ctx_window("ollama") == GatewayConfig.from_module().ollama_num_ctx

    def test_none_and_unknown_fall_back_to_local(self):
        """取不到供应商时安全回退，绝不因此放大预算。"""
        g = self._guard()
        local = GatewayConfig.from_module().ollama_num_ctx
        assert g._resolve_ctx_window(None) == local
        assert g._resolve_ctx_window("brand-new-provider") == local

    def test_cloud_budget_is_substantially_larger(self):
        """核心断言：云端预算必须显著大于本地（防止无意中退回旧行为）。"""
        g = self._guard()
        local = g._resolve_ctx_window(None)
        assert g._resolve_ctx_window("dashscope") >= local * 8

    def test_guard_accepts_provider_and_stays_within_budget(self):
        """带 provider 调用不报错，且超长 context 被截到供应商预算内。"""
        g = self._guard()
        huge = "深蹲。 " * 40000          # 远超任何预算
        ctx, info = g.guard("sys", {}, "s1", huge, "问题", provider="dashscope")
        assert info is not None
        assert info["budget"] == int(g._resolve_ctx_window("dashscope")
                                     * GatewayConfig.from_module().token_budget_ratio)
        assert len(ctx) < len(huge)
