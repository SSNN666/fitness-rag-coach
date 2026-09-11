"""
graph_client.py —— Neo4j 客户端的延迟防护层（显式超时 + 熔断）
================================================================
背景（**实测数字**，2026-09-11）：
    图谱停机时，一次 `execute_query` 的代价是 **34.53 秒**——不是连接超时，
    而是驱动默认的 `max_transaction_retry_time=30s`：日志里 4 次重试
    退避 1.0 → 2.4 → 3.7 → 7.0s，退避预算耗尽才抛 `ServiceUnavailable`。
    而热路径（`pipeline.py` 建 driver）**从未传过任何超时参数**，
    一次伤病问句含 2 次图谱调用（禁忌名单 + 图谱检索），复合伤病问句
    因多跳失败回退 1-hop 更是 3 次 → 最坏 **~103 秒**纯等待。

    （ROADMAP 此前记的「connection_timeout=8」是 `graph_view.py` 的参数，
      属可视化页；热路径没这个参数——又一次「文档比代码乐观」。）

本模块做两件事，都不是新功能，是把已有的降级路径**从「慢」变成「快」**：
    1. **显式超时**：`connection_timeout` + `max_transaction_retry_time`
       —— 后者是 34.5s 的真正来源，设为 0（不重试）。
       不重试的理由：调用方已有**完整**的本地降级（`contra_data` 副本，
       与图谱同源），请求内再重试只增加延迟、不改变结果。
    2. **熔断**：连续失败 N 次后直接短路，冷却后半开探测一次。
       短路抛 `GraphUnavailable`——现有调用方一律 `except Exception`
       后走本地降级，**行为与「调用失败」完全一致，只是不再等网络**。

⚠️ 为什么超时不能压到很小：本机「连接被拒」的 OS 地板实测 **2.0s**
   （127.0.0.1 / ::1 / localhost 三者一致，稳定复现），`connection_timeout`
   压不到它下面。所以目标是「34.5s → 地板」，不是「→ 0」。
   同理 `connection_timeout` 取 3.0s 而非更小值：**误判为故障的代价
   （图谱路被静默摘掉）远大于多等 1 秒**，宁可保守。

接线范围：**服务路径**（pipeline / mcp_server / graph_view）。
建索引与评测（build_index / eval_graph / eval_testset）刻意保持裸驱动——
那是一次性工具，要的是「响亮地失败」而不是「安静地降级」，
且没有用户在 HTTP 请求后面等着。
"""

from __future__ import annotations

import logging
import threading
import time

_log = logging.getLogger("graph_client")


class GraphUnavailable(Exception):
    """图谱不可用（熔断打开 / 连接失败）。

    调用方无需专门捕获：现有代码一律 `except Exception` 后走本地降级
    （见 `pipeline._resolve_contraindications` 的安全网设计）。
    """


# 熔断状态
CLOSED = "closed"          # 正常：调用放行
OPEN = "open"              # 熔断：直接短路，不碰网络
HALF_OPEN = "half_open"    # 半开：只放行一个探测请求


class CircuitBreaker:
    """连续失败计数式熔断器（线程安全）。

    状态机：CLOSED --(连续失败 ≥ threshold)--> OPEN
            OPEN --(冷却 cooldown 秒)--> HALF_OPEN（放行**一个**探测）
            HALF_OPEN --成功--> CLOSED ／ --失败--> OPEN（冷却重新计时）

    半开只放一个探测（`_probing` 标志）：图谱刚恢复时若放整批并发进来，
    失败的那部分会各自再等一次完整超时——正是要避免的东西。
    """

    def __init__(self, fail_threshold: int = 3, cooldown: float = 30.0,
                 clock=time.monotonic):
        self.fail_threshold = fail_threshold
        self.cooldown = cooldown
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CLOSED
        self._failures = 0            # 连续失败数（成功即清零）
        self._opened_at = 0.0
        self._probing = False         # 半开探测是否在途
        self._total_failures = 0      # 累计（观测用，不因成功清零）
        self._total_short_circuits = 0

    # ---------------- 门禁 ----------------

    def allow(self) -> bool:
        """放行一次调用？OPEN 且未到冷却时间 → False（调用方应短路）。"""
        with self._lock:
            if self._state == CLOSED:
                return True
            if self._state == OPEN:
                if self._clock() - self._opened_at < self.cooldown:
                    self._total_short_circuits += 1
                    return False
                # 冷却结束 → 半开，放行本次作为探测
                self._state = HALF_OPEN
                self._probing = True
                _log.warning("graph_breaker_half_open cooldown=%.1fs", self.cooldown)
                return True
            # HALF_OPEN：只放一个探测，其余短路
            if self._probing:
                self._total_short_circuits += 1
                return False
            self._probing = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._probing = False
            self._failures = 0
            if self._state != CLOSED:
                _log.warning("graph_breaker_closed（图谱恢复）累计失败 %d 次",
                             self._total_failures)
            self._state = CLOSED

    def record_failure(self, exc: BaseException | None = None) -> None:
        with self._lock:
            self._probing = False
            self._total_failures += 1
            if self._state == HALF_OPEN:
                # 探测失败 → 回到打开，冷却重新计时（不做退避升级：够用且好解释）
                self._state = OPEN
                self._opened_at = self._clock()
                _log.warning("graph_breaker_reopened（半开探测失败）err=%s",
                             type(exc).__name__)
                return
            self._failures += 1
            if self._state == CLOSED and self._failures >= self.fail_threshold:
                self._state = OPEN
                self._opened_at = self._clock()
                _log.warning(
                    "graph_breaker_opened 连续失败 %d 次，短路 %.0fs；"
                    "安全链路不受影响（禁忌走 contra_data 本地副本）err=%s",
                    self._failures, self.cooldown, type(exc).__name__)

    # ---------------- 观测 ----------------

    def snapshot(self) -> dict:
        """当前状态（healthz 暴露用）。`retry_in` = 距离半开探测的秒数。"""
        with self._lock:
            retry_in = 0.0
            if self._state == OPEN:
                retry_in = max(0.0, self.cooldown - (self._clock() - self._opened_at))
            return {
                "state": self._state,
                "consecutive_failures": self._failures,
                "retry_in_seconds": round(retry_in, 1),
                "total_failures": self._total_failures,
                "total_short_circuits": self._total_short_circuits,
            }


class GraphClient:
    """Neo4j driver 的包装：`execute_query` 与 driver 同签名（drop-in）。

    调用方（`retriever` / `graph_view`）**一行都不用改**——
    它们本来就 `try: driver.execute_query(...) except Exception: 降级`，
    短路时抛 `GraphUnavailable` 被同一分支接住。
    """

    def __init__(self, driver, breaker: CircuitBreaker | None = None):
        self._driver = driver
        self.breaker = breaker or CircuitBreaker()

    def execute_query(self, query: str, parameters: dict | None = None, **kwargs):
        if not self.breaker.allow():
            raise GraphUnavailable(
                f"图谱熔断中（{self.breaker.snapshot()['state']}），"
                f"短路以免等待网络")
        try:
            result = self._driver.execute_query(query, parameters, **kwargs)
        except Exception as exc:
            self.breaker.record_failure(exc)
            raise
        self.breaker.record_success()
        return result

    def verify_connectivity(self, **kwargs):
        """同 driver 语义；同样受熔断保护。"""
        if not self.breaker.allow():
            raise GraphUnavailable("图谱熔断中，跳过连通性检查")
        try:
            self._driver.verify_connectivity(**kwargs)
        except Exception as exc:
            self.breaker.record_failure(exc)
            raise
        self.breaker.record_success()

    def close(self) -> None:
        self._driver.close()

    def snapshot(self) -> dict:
        return self.breaker.snapshot()


def make_graph_client(uri: str, auth, *, fail_threshold: int | None = None,
                      cooldown: float | None = None,
                      connection_timeout: float | None = None,
                      max_retry_time: float | None = None) -> GraphClient:
    """按 config 旋钮建带防护的图谱客户端（服务路径统一走这里）。

    超时/熔断参数默认取 config，便于测试与调参覆盖。
    """
    from neo4j import GraphDatabase
    import config as cfg

    if connection_timeout is None:
        connection_timeout = cfg.NEO4J_CONNECTION_TIMEOUT
    if max_retry_time is None:
        max_retry_time = cfg.NEO4J_MAX_RETRY_TIME

    driver = GraphDatabase.driver(
        uri, auth=auth,
        connection_timeout=connection_timeout,
        # 34.53s 的真正来源；调用方已有完整本地降级，请求内重试只加延迟
        max_transaction_retry_time=max_retry_time,
    )
    breaker = CircuitBreaker(
        fail_threshold=(cfg.NEO4J_BREAKER_FAIL_THRESHOLD
                        if fail_threshold is None else fail_threshold),
        cooldown=cfg.NEO4J_BREAKER_COOLDOWN if cooldown is None else cooldown,
    )
    return GraphClient(driver, breaker)
