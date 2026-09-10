"""session_store 单测：持久化 / 跨重启恢复 / 坏文件容错 / LRU / 开关（纯本地，零依赖）。"""
import json
import os
import threading
import time

from session_store import SessionHistory, SessionStore


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "s.json")
        s = SessionStore(path=path)
        s["u1"] = SessionHistory()
        s["u1"].add_message({"type": "human", "content": "腰突能深蹲吗"})
        s["u1"].add_message({"type": "ai", "content": "不建议"})
        s.save()

        # 全新实例 = 模拟进程重启
        s2 = SessionStore(path=path)
        msgs = s2.get("u1").messages
        assert [m["content"] for m in msgs] == ["腰突能深蹲吗", "不建议"]
        assert [m["type"] for m in msgs] == ["human", "ai"]

    def test_no_file_written_when_disabled(self, tmp_path):
        path = str(tmp_path / "s.json")
        s = SessionStore(path=path, enabled=False)
        s["u1"] = SessionHistory()
        s["u1"].add_message({"type": "human", "content": "x"})
        s.save()
        assert not os.path.exists(path)

    def test_missing_file_starts_empty(self, tmp_path):
        s = SessionStore(path=str(tmp_path / "nope.json"))
        assert len(s) == 0

    def test_corrupt_file_does_not_crash(self, tmp_path):
        """坏文件必须能启动（否则一次写坏盘就再也起不来）。"""
        path = tmp_path / "bad.json"
        path.write_text("{ 这不是合法 JSON", encoding="utf-8")
        s = SessionStore(path=str(path))
        assert len(s) == 0
        # 且能自愈：再存一次即覆盖坏文件
        s["u1"] = SessionHistory()
        s["u1"].add_message({"type": "human", "content": "ok"})
        s.save()
        assert json.loads(path.read_text(encoding="utf-8"))["u1"]["messages"][0]["content"] == "ok"

    def test_non_dict_payload_starts_empty(self, tmp_path):
        """文件是合法 JSON 但不是对象（如数组）→ 同样安全降级。"""
        path = tmp_path / "arr.json"
        path.write_text("[1,2,3]", encoding="utf-8")
        assert len(SessionStore(path=str(path))) == 0

    def test_atomic_write_no_tmp_leftover(self, tmp_path):
        path = str(tmp_path / "s.json")
        s = SessionStore(path=path)
        s["u1"] = SessionHistory()
        s.save()
        assert os.path.isfile(path)
        assert not os.path.exists(f"{path}.tmp")


class TestLRU:
    def test_evicts_oldest_beyond_max(self, tmp_path):
        s = SessionStore(path=str(tmp_path / "s.json"), max_sessions=3)
        for i in range(3):
            s[f"u{i}"] = SessionHistory()
            s.touch(f"u{i}")
            time.sleep(0.01)          # 拉开时间戳，保证顺序确定
        s["u3"] = SessionHistory()
        s.touch("u3")
        s.evict_lru()
        assert len(s) == 3
        assert "u0" not in s            # 最久未访问的被驱逐
        assert "u3" in s

    def test_get_refreshes_lru(self, tmp_path):
        s = SessionStore(path=str(tmp_path / "s.json"), max_sessions=2)
        for i in range(2):
            s[f"u{i}"] = SessionHistory()
            s.touch(f"u{i}")
            time.sleep(0.01)
        s.get("u0")                     # 读取刷新 u0
        s["u2"] = SessionHistory()
        s.touch("u2")
        s.evict_lru()
        assert "u0" in s and "u2" in s
        assert "u1" not in s            # u1 成为最久未访问

    def test_eviction_is_persisted(self, tmp_path):
        """驱逐后落盘：重启不能把已驱逐的会话又载回来。"""
        path = str(tmp_path / "s.json")
        s = SessionStore(path=path, max_sessions=1)
        s["old"] = SessionHistory()
        s.touch("old")
        time.sleep(0.01)
        s["new"] = SessionHistory()
        s.touch("new")
        s.evict_lru()
        s.save()

        s2 = SessionStore(path=path, max_sessions=1)
        assert "new" in s2 and "old" not in s2

    def test_untouched_sessions_evicted_first(self, tmp_path):
        """没有时间戳的会话（从未 touch）优先被清理，不永久占位。"""
        s = SessionStore(path=str(tmp_path / "s.json"), max_sessions=1)
        s["no_ts"] = SessionHistory()   # 不 touch
        s["with_ts"] = SessionHistory()
        s.touch("with_ts")
        s.evict_lru()
        assert "with_ts" in s and len(s) == 1


class TestCompatibility:
    """SessionStore 必须对 pipeline / 网关完全透明（dict 语义不变）。"""

    def test_dict_semantics(self, tmp_path):
        s = SessionStore(path=str(tmp_path / "s.json"))
        s["u1"] = SessionHistory()
        assert "u1" in s
        assert len(s) == 1
        assert s.get("missing") is None
        s.pop("u1", None)
        assert len(s) == 0
        assert list(s.items()) == []

    def test_session_history_interface(self):
        """网关令牌预算守卫使用 messages / clear / add_message 三个接口。"""
        h = SessionHistory()
        h.add_message({"type": "human", "content": "a"})
        h.add_message({"type": "ai", "content": "b"})
        assert len(h.messages) == 2
        h.clear()
        assert h.messages == []
        h.add_message({"type": "human", "content": "c"})
        assert len(h.messages) == 1

    def test_concurrent_save_is_safe(self, tmp_path):
        """多线程并发 save 不产生损坏文件（原子写 + 锁）。"""
        path = str(tmp_path / "s.json")
        s = SessionStore(path=path)
        for i in range(5):
            s[f"u{i}"] = SessionHistory()
            s[f"u{i}"].add_message({"type": "human", "content": f"q{i}"})

        def _worker():
            for _ in range(20):
                s.save()

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 文件必须是完整合法 JSON
        data = json.loads(open(path, encoding="utf-8").read())
        assert len(data) == 5
        assert not os.path.exists(f"{path}.tmp")
