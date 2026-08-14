"""gateway 降噪器单测：模式标签（tag）语义——同 query 不同模式不判重。"""
from gateway import GatewayConfig, _NoiseReducer, _SlidingWindowLimiter


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
