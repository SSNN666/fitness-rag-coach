"""
session_store.py —— 会话记忆持久化
====================================
把多轮对话上下文从「进程内 dict」升级为「可跨重启恢复的存储」。

为什么需要
----------
原实现是进程内 `dict` + LRU 驱逐（`SESSION_MAX_COUNT=64`）：**进程一退上下文就没了**。
用户第二轮问「那硬拉呢」时，指代消解拿不到上一轮的「腰突」——多轮改写、会话记忆
这些能力全部依赖历史存在。重启一次就退化回单轮。

设计
----
- `SessionHistory`：单会话消息列表（接口与网关令牌预算守卫兼容：
  `messages` / `add_message` / `clear`）
- `SessionStore(dict)`：**dict 子类**，对 pipeline / 网关完全透明——
  取值、`.get()`、`pop()`、`len()` 行为不变，额外提供 `save()` / `touch()` / `evict_lru()`
- 落盘：JSON + 原子写（tmp → `os.replace`，与 fact_cache.py 同款）。
  崩溃不会留下半截文件
- 落盘失败**不阻断请求**：内存态照常工作，退化为原来的非持久行为

边界
----
- 单机、单进程适用。多实例要换 Redis/DB —— 本模块的 dict 接口正好是替换点。
- 会话内容含用户对话（可能含健康信息），落盘文件已加入 .gitignore。
"""

from __future__ import annotations

import json
import os
import threading
import time

from config import SESSION_MAX_COUNT, SESSION_PERSIST_ENABLED, SESSION_PERSIST_PATH


class SessionHistory:
    """单会话消息列表。

    接口保持与 langchain_community ChatMessageHistory 兼容（messages / add_message / clear），
    网关的令牌预算守卫直接可用。
    """

    def __init__(self, messages: list | None = None):
        self.messages: list = list(messages) if messages else []

    def add_message(self, msg) -> None:
        self.messages.append(msg)

    def clear(self) -> None:
        self.messages = []

    # ---- 序列化（落盘用）----
    def to_dict(self) -> dict:
        return {"messages": self.messages}

    @classmethod
    def from_dict(cls, d: dict) -> "SessionHistory":
        msgs = d.get("messages") if isinstance(d, dict) else None
        return cls(msgs if isinstance(msgs, list) else [])


class SessionStore(dict):
    """会话存储：dict 兼容 + 原子落盘。

    Args:
        path: 落盘文件路径；None 或 enabled=False 时退化为纯内存 dict
        max_sessions: LRU 上限（超出按最久未访问驱逐）
        enabled: 总开关

    用法与普通 dict 完全一致：
        store = SessionStore()
        store["s1"] = SessionHistory()
        store.get("s1").add_message({...})
        store.save()
    """

    def __init__(self, path: str | None = None, max_sessions: int | None = None,
                 enabled: bool | None = None):
        super().__init__()
        self._path = path if path is not None else SESSION_PERSIST_PATH
        self._max = max_sessions if max_sessions is not None else SESSION_MAX_COUNT
        _enabled = SESSION_PERSIST_ENABLED if enabled is None else enabled
        self._enabled = bool(_enabled and self._path)
        self._lock = threading.Lock()
        self._last_access: dict[str, float] = {}
        if self._enabled:
            self._load()

    # ---- 持久化 ----

    def _load(self) -> None:
        """启动时恢复。文件缺失/损坏 → 空存储启动（绝不因坏文件起不来）。"""
        try:
            if not os.path.isfile(self._path):
                return
            with open(self._path, encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return
            for sid, payload in raw.items():
                super().__setitem__(sid, SessionHistory.from_dict(payload))
                self._last_access[sid] = time.time()
        except Exception:
            # 坏文件不应阻断启动：清空内存态继续（下次 save 会覆盖）
            super().clear()
            self._last_access.clear()

    def save(self) -> None:
        """原子落盘。失败静默（内存态仍可用）。"""
        if not self._enabled:
            return
        with self._lock:
            try:
                payload = {sid: h.to_dict() for sid, h in self.items()}
                tmp = f"{self._path}.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self._path)   # 原子替换：不会读到半截文件
            except Exception:
                pass

    # ---- LRU ----

    def touch(self, session_id: str) -> None:
        """记录最近访问时间（读取历史也算访问）。"""
        self._last_access[session_id] = time.time()

    def evict_lru(self) -> None:
        """会话数超过上限时按最久未访问驱逐，并同步移除访问时间记录。

        ⚠️ 未 touch 过的会话（无时间戳）视为「最久未访问」优先驱逐——
        它们不在 _last_access 里，若只对 _last_access 排序会把**有**时间戳的
        会话当成最旧驱逐掉，与意图完全相反（本模块单测覆盖此边界）。
        """
        if len(self) <= self._max:
            return
        excess = len(self) - self._max
        untouched = [sid for sid in self if sid not in self._last_access]
        by_time = [sid for sid, _ in sorted(self._last_access.items(),
                                            key=lambda kv: kv[1])]
        for sid in (untouched + by_time)[:excess]:
            super().pop(sid, None)
            self._last_access.pop(sid, None)

    def get(self, session_id: str):
        """取值并刷新 LRU 时间戳（读取也是访问）。"""
        session = super().get(session_id)
        if session is not None:
            self.touch(session_id)
        return session


def build_session_store() -> SessionStore:
    """按配置构造会话存储（build_pipeline 调用）。"""
    return SessionStore()
