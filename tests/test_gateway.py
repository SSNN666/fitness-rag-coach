"""gateway 单测：降噪器模式语义 + 令牌预算窗口解析 + 会话级状态回收。"""
from gateway import (
    Gateway,
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
# 会话级状态回收（会话隔离的前置条件）
# ================================================================
#
# 背景：限流/降噪状态按 session_id 建键，原实现 session_id 恒为 "default_user"
# → 字典永远只有一组键，无回收也看不出问题。会话隔离后每个访客一个键，
# 不回收即无上限增长——把一个隐私 bug 换成内存泄漏。

def _make_gateway(**over):
    over.setdefault("enabled", True)
    return Gateway(GatewayConfig(**over))


def _expired_state(n: int, now: float) -> dict:
    """构造 n 个「对限流与降噪都已过期」的访客状态键。

    两者 TTL 不同（限流=窗口 60s，降噪=NOISE_TTL 600s），取 max 再留余量，
    否则降噪键会因仍在 TTL 内而存活——测出来的「没回收」是假象。
    """
    old = now - max(600.0, 60.0) - 60
    state = {}
    for i in range(n):
        state[f"_gw_rate_visitor{i}"] = {"timestamps": [old], "blocked_until": 0}
        state[f"_gw_noise_visitor{i}"] = [{"q": "旧问题", "tag": "fast", "ts": old}]
    return state


def test_expired_state_reclaimed():
    """冷却/窗口都过期的键被回收（否则进程内永久驻留）。"""
    import time as _t
    g = _make_gateway(state_sweep_interval_s=0)   # 0 = 每个请求都可触发清扫
    state = _expired_state(50, _t.time())
    assert len(state) == 100
    g.check_rate_limit("fresh", state)
    # 50 个访客的键全部回收，只剩本次请求自己的键
    assert len(state) == 1
    assert "_gw_rate_fresh" in state


def test_active_rate_limit_key_survives():
    """冷却期内的封禁不可被回收——否则清扫等于提前解封。"""
    import time as _t
    g = _make_gateway(state_sweep_interval_s=0)
    now = _t.time()
    state = {"_gw_rate_banned": {"timestamps": [now - 120],
                                 "blocked_until": now + 300}}
    g.check_rate_limit("other", state)
    assert "_gw_rate_banned" in state
    assert not g.check_rate_limit("banned", state)[0]   # 仍然封禁


def test_recent_noise_key_survives():
    """窗口内的降噪记录必须保留，否则降噪功能失效。"""
    import time as _t
    g = _make_gateway(state_sweep_interval_s=0)
    state = {}
    assert g.is_duplicate("腰突能深蹲吗", "v1", state, tag="fast") is False
    g.check_rate_limit("v1", state)                      # 触发可能的清扫
    assert g.is_duplicate("腰突能深蹲吗", "v1", state, tag="fast") is True


def test_sweep_throttled_within_interval():
    """未超上限时，一个 interval 内只清扫一次（把 O(n) 摊薄）。"""
    import time as _t
    g = _make_gateway(state_sweep_interval_s=3600)
    state = _expired_state(30, _t.time())
    g.check_rate_limit("fresh", state)
    after_first = len(state)
    for i in range(20):                                  # 同一 interval 内的后续请求
        g.check_rate_limit(f"v{i}", state)
    # 首次清扫后新键不再被回收（间隔未到），但也没有继续膨胀
    assert len(state) > after_first
    assert len(state) <= after_first + 20


def test_hard_cap_evicts_oldest():
    """短时大量不同 session_id → 超过硬上限时按最久未活动驱逐。

    键的时间戳都落在限流窗口内（不会被过期清扫掉），逼出的是**上限**这条路径，
    否则测的其实是过期回收——两者都能让 len() 变小，断言分不出来。
    """
    import time as _t
    g = _make_gateway(state_max_keys=10, state_sweep_interval_s=3600)
    now = _t.time()
    state = {}
    for i in range(30):            # ts 递增：编号越大越「新」
        state[f"_gw_rate_v{i:02d}"] = {"timestamps": [now - 30 + i * 0.1],
                                       "blocked_until": 0}
    before_last_active = {k: g._entry_last_active(v) for k, v in state.items()}

    g.check_rate_limit("trigger", state)

    assert len(state) == 11        # 上限 10 + 本次请求自己的键
    survivors = [k for k in state if k.startswith("_gw_rate_v")]
    kept = sorted(survivors, key=lambda k: before_last_active[k])
    assert kept == [f"_gw_rate_v{i:02d}" for i in range(20, 30)]   # 留下最活跃的 10 个
    assert "_gw_rate_trigger" in state


def test_sweep_ignores_foreign_keys():
    """只回收自己的前缀——调用方放在同一 dict 里的其他键不受影响。"""
    import time as _t
    g = _make_gateway(state_sweep_interval_s=0)
    state = _expired_state(5, _t.time())
    state["unrelated_key"] = {"whatever": 1}
    g.check_rate_limit("fresh", state)
    assert state["unrelated_key"] == {"whatever": 1}


# ================================================================
# 成本上限：记账挂载点与档位判定
# ================================================================

def _usage(p=100, c=50, t=None):
    """用真实的 UsageInfo（不是 SimpleNamespace）：日志与记账都读它的字段，
    假对象字段不全就只能在测试里改来改去，迟早与生产口径脱节。"""
    from llm_adapter import UsageInfo
    return UsageInfo(prompt_tokens=p, completion_tokens=c,
                     total_tokens=t if t is not None else p + c,
                     model="m", provider="p", latency_ms=1)


def test_log_usage_is_the_accounting_funnel():
    """所有 LLM 调用都经 Gateway.log_usage → 它必须是成本的唯一记账入口。"""
    g = _make_gateway(cost_soft_limit_tokens=1000, cost_hard_limit_tokens=5000)
    assert g.cost_snapshot()["total_tokens"] == 0
    g.log_usage("rid", "chat", _usage(300, 200))
    g.log_usage("rid", "hyde", _usage(100, 50))
    snap = g.cost_snapshot()
    assert snap["total_tokens"] == 650
    assert snap["requests"] == 2


def test_cost_records_even_when_gateway_disabled():
    """关掉网关日志 ≠ 关掉计费：否则存在一条完全不记账的调用路径。"""
    g = _make_gateway(enabled=False, cost_soft_limit_tokens=1000,
                      cost_hard_limit_tokens=5000)
    g.log_usage("rid", "chat", _usage(300, 200))
    assert g.cost_snapshot()["total_tokens"] == 500


def test_check_cost_budget_transitions():
    g = _make_gateway(cost_soft_limit_tokens=1000, cost_hard_limit_tokens=2000)
    assert g.check_cost_budget() == (True, "")
    assert g.cost_state == "ok"

    g.log_usage("rid", "chat", _usage(0, 0, 1500))
    assert g.cost_state == "degraded"
    assert g.check_cost_budget()[0] is True        # 软阈值只降级，不拒绝

    g.log_usage("rid", "chat", _usage(0, 0, 600))
    assert g.cost_state == "exhausted"
    allowed, reason = g.check_cost_budget()
    assert allowed is False
    assert "上限" in reason


def test_check_cost_budget_disabled():
    g = _make_gateway(cost_guard_enabled=False, cost_hard_limit_tokens=1)
    g.log_usage("rid", "chat", _usage(0, 0, 999))
    assert g.check_cost_budget() == (True, "")


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


class TestContextTruncation:
    """级联截断的 context 尾部截断：保留头部（安全数据）+ 保留原标点。"""

    def _guard(self):
        return _TokenBudgetGuard(GatewayConfig.from_module(), logger=None)

    def test_keeps_head_and_original_punctuation(self):
        """回归：原实现用 "。".join() 重组，而 split 已吃掉分隔符 →
        ！？；与换行全部变成「。」，禁忌黑名单被压成一行跑文。"""
        g = self._guard()
        head = "【伤病禁忌黑名单】\n- 腰间盘突出禁忌: 硬拉\n- 半月板损伤禁忌: 深蹲跳\n"
        body = "深蹲要循序渐进！注意膝盖不要内扣？先做热身；再上重量。" * 300

        out = g._truncate_context(head + body, 200)

        # 头部安全数据完整保留（截断只删尾部）
        assert "硬拉" in out and "深蹲跳" in out
        # 原标点与换行未被统一替换
        assert "\n" in out
        for mark in ("！", "？", "；"):
            assert mark in out, f"{mark} 被吞掉了"
        assert out.endswith("[内容已截断]")

    def test_single_oversized_sentence_falls_back_to_char_cut(self):
        """连一个句子都放不下时退化为字符级截断，仍返回可用内容而非空串。"""
        g = self._guard()
        out = g._truncate_context("深蹲" * 5000, 10)
        assert out and len(out) > 10
        assert out.endswith("…")

    def test_short_context_returned_unchanged(self):
        g = self._guard()
        ctx = "深蹲主要锻炼股四头肌。"
        assert g._truncate_context(ctx, 10000) == ctx
