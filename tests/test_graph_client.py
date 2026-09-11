"""graph_client 单测：熔断状态机 / 半开单探测 / drop-in 兼容 / 安全链路降级。

背景：图谱停机时裸 driver 单次 execute_query 实测 34.53s（驱动默认重试预算吃满），
而一次伤病问句含 2~3 次图谱调用。本模块把这些代价压到「一次连接尝试 + 之后零网络」。
"""
import threading

import pytest

from graph_client import (
    CLOSED, HALF_OPEN, OPEN,
    CircuitBreaker, GraphClient, GraphUnavailable,
)


class FakeClock:
    """可控时钟（熔断冷却不该让测试真的睡 30 秒）。"""
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeDriver:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple] = []
        self.verify_calls = 0
        self.closed = False

    def execute_query(self, query, parameters=None, **kwargs):
        self.calls.append((query, parameters, kwargs))
        if self.fail:
            raise RuntimeError("模拟图谱不可用")
        return (["rec"], None, None)

    def verify_connectivity(self, **kwargs):
        self.verify_calls += 1
        if self.fail:
            raise RuntimeError("模拟图谱不可用")

    def close(self):
        self.closed = True


def _client(fail=False, threshold=3, cooldown=30.0, clock=None):
    driver = FakeDriver(fail=fail)
    breaker = CircuitBreaker(fail_threshold=threshold, cooldown=cooldown,
                             clock=clock or FakeClock())
    return GraphClient(driver, breaker), driver, breaker


# ----------------------------------------------------------------
# 正常路径
# ----------------------------------------------------------------

def test_success_passes_through_unchanged():
    client, driver, breaker = _client()
    result = client.execute_query("MATCH (n) RETURN n", {"x": 1}, database_="neo4j")
    assert result == (["rec"], None, None)
    assert driver.calls == [("MATCH (n) RETURN n", {"x": 1}, {"database_": "neo4j"})]
    assert breaker.snapshot()["state"] == CLOSED


def test_params_and_kwargs_passthrough_is_drop_in():
    """调用方写法一字不改：位置参数 + database_ 关键字（retriever 四处均如此）。

    这条是「接线时 retriever/graph_view 一行都不用改」的前提保障——
    签名一旦漂移，失败会发生在**降级路径**上，测试更容易漏。
    """
    client, driver, _ = _client()
    client.execute_query("CYPHER", {"names": ["腰突"]}, database_="neo4j")
    assert driver.calls[0][1] == {"names": ["腰突"]}
    assert driver.calls[0][2] == {"database_": "neo4j"}


def test_success_resets_consecutive_failures():
    client, driver, breaker = _client(threshold=3)
    driver.fail = True
    for _ in range(2):
        with pytest.raises(RuntimeError):
            client.execute_query("Q")
    driver.fail = False
    client.execute_query("Q")            # 成功 → 计数清零
    driver.fail = True
    for _ in range(2):                   # 再失败 2 次（若未清零则第 2 次就该熔断）
        with pytest.raises(RuntimeError):
            client.execute_query("Q")
    assert breaker.snapshot()["state"] == CLOSED


# ----------------------------------------------------------------
# 熔断：打开 / 短路
# ----------------------------------------------------------------

def test_opens_after_threshold_consecutive_failures():
    client, driver, breaker = _client(fail=True, threshold=3)
    for i in range(3):
        with pytest.raises(RuntimeError):
            client.execute_query("Q")
        assert breaker.snapshot()["state"] == (OPEN if i == 2 else CLOSED)
    assert len(driver.calls) == 3


def test_open_short_circuits_without_touching_driver():
    """熔断打开后**零网络尝试**——这是本模块存在的全部意义。"""
    client, driver, breaker = _client(fail=True, threshold=3)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            client.execute_query("Q")
    calls_before = len(driver.calls)

    for _ in range(10):
        with pytest.raises(GraphUnavailable):
            client.execute_query("Q")
    assert len(driver.calls) == calls_before, "短路时不应触碰驱动"
    assert breaker.snapshot()["total_short_circuits"] == 10


def test_graph_unavailable_is_plain_exception_for_existing_handlers():
    """调用方一律 `except Exception` 后走本地降级——短路异常必须被它接住。

    若继承 BaseException（如 KeyboardInterrupt 那一系），
    现有降级分支会全部失效，图谱一挂就变成 500。
    """
    assert issubclass(GraphUnavailable, Exception)
    try:
        raise GraphUnavailable("熔断")
    except Exception:
        pass


# ----------------------------------------------------------------
# 半开：冷却 / 单探测 / 恢复与再失败
# ----------------------------------------------------------------

def test_half_open_after_cooldown_then_recovers():
    clock = FakeClock()
    client, driver, breaker = _client(fail=True, threshold=3, cooldown=30.0, clock=clock)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            client.execute_query("Q")
    assert breaker.snapshot()["state"] == OPEN

    clock.advance(29.9)
    with pytest.raises(GraphUnavailable):
        client.execute_query("Q")        # 冷却未到 → 仍短路
    assert breaker.snapshot()["state"] == OPEN

    clock.advance(0.2)                   # 冷却已到
    driver.fail = False
    client.execute_query("Q")            # 半开探测成功
    assert breaker.snapshot()["state"] == CLOSED
    assert breaker.snapshot()["consecutive_failures"] == 0


def test_half_open_probe_failure_reopens_and_restarts_cooldown():
    clock = FakeClock()
    client, driver, breaker = _client(fail=True, threshold=3, cooldown=30.0, clock=clock)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            client.execute_query("Q")
    clock.advance(30.1)
    with pytest.raises(RuntimeError):    # 探测失败 → 重新打开
        client.execute_query("Q")
    assert breaker.snapshot()["state"] == OPEN
    assert breaker.snapshot()["retry_in_seconds"] == pytest.approx(30.0, abs=0.2)


def test_half_open_admits_only_one_probe():
    """图谱刚恢复时若放整批并发进来，失败的那些各自再等一次完整超时。

    半开只放一个探测：其余请求立刻短路（宁可这一批走本地降级）。
    """
    clock = FakeClock()
    client, driver, breaker = _client(fail=True, threshold=1, cooldown=10.0, clock=clock)
    with pytest.raises(RuntimeError):
        client.execute_query("Q")        # threshold=1 → 立刻打开
    clock.advance(10.1)
    driver.fail = False                  # 图谱已恢复：探测应当成功

    # 用「阻塞在驱动里的探测」模拟并发：探测在途时其他线程必须被挡
    entered = threading.Event()
    release = threading.Event()
    real_execute = driver.execute_query

    def slow_execute(query, parameters=None, **kwargs):
        entered.set()
        release.wait(timeout=5)
        return real_execute(query, parameters, **kwargs)

    driver.execute_query = slow_execute
    probe_result = {}

    def probe():
        try:
            probe_result["ok"] = client.execute_query("Q")
        except Exception as e:               # noqa: BLE001 — 测试内记录即可
            probe_result["err"] = e

    t = threading.Thread(target=probe)
    t.start()
    assert entered.wait(timeout=5), "探测线程应已进入驱动"

    # 探测在途：后续调用必须短路，不得进入驱动
    for _ in range(3):
        with pytest.raises(GraphUnavailable):
            client.execute_query("Q")

    release.set()
    t.join(timeout=5)
    assert "ok" in probe_result
    assert breaker.snapshot()["state"] == CLOSED


# ----------------------------------------------------------------
# 观测
# ----------------------------------------------------------------

def test_snapshot_reports_state_for_healthz():
    client, driver, breaker = _client(threshold=2, cooldown=15.0)
    snap = breaker.snapshot()
    assert snap["state"] == CLOSED and snap["retry_in_seconds"] == 0.0

    driver.fail = True
    for _ in range(2):
        with pytest.raises(RuntimeError):
            client.execute_query("Q")
    snap = breaker.snapshot()
    assert snap["state"] == OPEN
    assert snap["consecutive_failures"] == 2
    assert snap["total_failures"] == 2
    assert 0 < snap["retry_in_seconds"] <= 15.0


def test_verify_connectivity_is_also_protected():
    client, driver, breaker = _client(threshold=2)
    driver.fail = True
    for _ in range(2):
        with pytest.raises(RuntimeError):
            client.verify_connectivity()
    assert breaker.snapshot()["state"] == OPEN
    with pytest.raises(GraphUnavailable):
        client.verify_connectivity()
    assert driver.verify_calls == 2, "短路后不应再检查连通性"


# ----------------------------------------------------------------
# 集成：安全链路不退化（判据④）
# ----------------------------------------------------------------

def test_retriever_contraindications_falls_back_when_breaker_open():
    """熔断短路 → get_contraindications 返回 {} → 调用方走本地副本。

    这是安全网的关键一环：禁忌是**不该答的绝不答**的最后一道闸，
    图谱不可用时它必须继续工作（bug#4 已修「运行时故障不降级」，此处守住它）。
    """
    from retriever import FitnessRAGRetriever

    client, driver, breaker = _client(fail=True, threshold=1)
    with pytest.raises(RuntimeError):
        client.execute_query("Q")        # 打开熔断

    r = FitnessRAGRetriever.__new__(FitnessRAGRetriever)   # 不加载 Milvus
    r._neo4j = client
    assert r.get_contraindications(["腰突"]) == {}
    assert r.graph_status()["state"] == OPEN


def test_pipeline_resolves_to_local_copy_when_graph_short_circuited():
    """判据④端到端：图谱熔断时，PipelineService 仍产出完整禁忌黑名单。"""
    from pipeline import PipelineService
    from retriever import FitnessRAGRetriever

    client, driver, breaker = _client(fail=True, threshold=1)
    with pytest.raises(RuntimeError):
        client.execute_query("Q")

    stub = PipelineService.__new__(PipelineService)        # 不加载服务
    stub._retriever = FitnessRAGRetriever.__new__(FitnessRAGRetriever)
    stub._retriever._neo4j = client

    contra_map, forbidden, text, hit = PipelineService._resolve_contraindications(
        stub, ["腰突"])
    assert hit is True, "图谱熔断不应让禁忌判定失效"
    assert "深蹲" in forbidden and "硬拉" in forbidden
    assert "禁忌黑名单" in text
