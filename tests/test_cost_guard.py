"""cost_guard 单测：累计、两级阈值、窗口重置、落盘与容错。"""
import json
import time
from types import SimpleNamespace

from cost_guard import DEGRADED, EXHAUSTED, OK, CostGuard


def _usage(p=100, c=50, t=None):
    return SimpleNamespace(prompt_tokens=p, completion_tokens=c,
                           total_tokens=t if t is not None else p + c)


def _guard(**over):
    kw = dict(soft_limit=1000, hard_limit=5000, window_s=86400, path=None)
    kw.update(over)
    return CostGuard(**kw)


# ---------------- 记账 ----------------

def test_record_accumulates():
    g = _guard()
    g.record(_usage(100, 50))
    g.record(_usage(200, 100))
    assert g.used == 450
    assert g.snapshot()["requests"] == 2
    assert g.snapshot()["prompt_tokens"] == 300


def test_total_takes_max_of_reported_and_sum():
    """部分供应商的 total 不含思考/缓存 token；与 prompt+completion 取大值更保守。"""
    g = _guard()
    g.record(_usage(p=100, c=50, t=10))    # 上报 total 偏小
    assert g.used == 150                   # 取 p+c，不采信偏小的 total
    g2 = _guard()
    g2.record(_usage(p=100, c=50, t=999))  # 上报 total 偏大（含缓存等）
    assert g2.used == 999


def test_record_none_usage_ignored():
    g = _guard()
    g.record(None)
    assert g.used == 0
    assert g.snapshot()["requests"] == 0


def test_disabled_guard_records_nothing():
    g = _guard(enabled=False)
    g.record(_usage())
    assert g.used == 0
    assert g.state == OK          # 关闭后永不拒绝


# ---------------- 两级阈值 ----------------

def test_state_transitions():
    # 用 p=0,c=0 才能精确控制步长（否则记账会取 max(total, p+c)，步长被 p+c 抬高）
    g = _guard(soft_limit=1000, hard_limit=2000)
    assert g.state == OK
    g.record(_usage(0, 0, 999))
    assert g.state == OK
    g.record(_usage(0, 0, 1))     # 累计 1000 = 软阈值
    assert g.state == DEGRADED
    g.record(_usage(0, 0, 999))   # 累计 1999
    assert g.state == DEGRADED
    g.record(_usage(0, 0, 1))     # 累计 2000 = 硬阈值
    assert g.state == EXHAUSTED


def test_zero_limit_disables_that_level():
    """0 = 不启用该级（只想设硬上限时不必编一个软阈值）。"""
    g = _guard(soft_limit=0, hard_limit=1000)
    g.record(_usage(t=999))
    assert g.state == OK          # 软阈值关掉 → 不降级
    g.record(_usage(t=1))
    assert g.state == EXHAUSTED

    g2 = _guard(soft_limit=1000, hard_limit=0)
    g2.record(_usage(t=5000))
    assert g2.state == DEGRADED   # 硬阈值关掉 → 只降级，不拒绝


# ---------------- 窗口重置 ----------------

def test_window_rollover_resets():
    """窗口到期清零——否则长期运行后服务会被永久锁死。"""
    g = _guard(soft_limit=1000, hard_limit=2000, window_s=0.2)
    g.record(_usage(t=1999))
    assert g.state == DEGRADED
    time.sleep(0.25)
    assert g.used == 0
    assert g.state == OK


def test_window_start_initialized_on_first_access():
    """读接口（used/state/snapshot）也会初始化窗口起点。

    这是有意的：长时间没记账的 guard 再次被查询时也要能正确滚动窗口，
    否则「窗口已过期」只有在下次 record 时才被发现，账本会滞留旧值。
    """
    g = _guard()
    assert g.snapshot()["window_start"] > 0    # 读取即初始化
    assert g.used == 0
    assert g.state == OK


# ---------------- 落盘 ----------------

def test_persist_and_reload(tmp_path):
    """重启不清零——否则重启就是一条绕过限额的捷径。"""
    p = tmp_path / "cost.json"
    g = CostGuard(soft_limit=1000, hard_limit=5000, path=str(p))
    g.record(_usage(300, 200))
    assert p.exists()

    g2 = CostGuard(soft_limit=1000, hard_limit=5000, path=str(p))
    assert g2.used == 500
    assert g2.snapshot()["requests"] == 1


def test_corrupt_file_starts_empty(tmp_path):
    """坏文件按空账本启动：计费故障不该阻断服务启动（宁可少算）。"""
    p = tmp_path / "cost.json"
    p.write_text("{ 这不是 JSON", encoding="utf-8")
    g = CostGuard(soft_limit=1000, hard_limit=5000, path=str(p))
    assert g.used == 0
    assert g.state == OK


def test_partial_record_filled_with_defaults(tmp_path):
    """旧版本文件缺字段时补默认值，不因 KeyError 崩掉。"""
    p = tmp_path / "cost.json"
    p.write_text(json.dumps({"total_tokens": 777}), encoding="utf-8")
    g = CostGuard(soft_limit=1000, hard_limit=5000, path=str(p))
    assert g.used == 777
    assert g.snapshot()["requests"] == 0


def test_unwritable_path_degrades_silently():
    """落盘失败静默降级为内存态：计费持久化不该让请求失败。"""
    g = CostGuard(soft_limit=1000, hard_limit=5000, path="Z:/nonexistent/cost.json")
    g.record(_usage(100, 100))     # 不应抛异常
    assert g.used == 200


def test_reset():
    g = _guard()
    g.record(_usage(t=1500))
    assert g.state == DEGRADED
    g.reset()
    assert g.used == 0
    assert g.state == OK
