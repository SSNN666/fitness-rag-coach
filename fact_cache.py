"""
FactCache —— 简单 JSON 文件缓存
===============================
缓存校验通过的伤病问答结果，避免重复校验同一问题。
Key = MD5(question + sorted entities)，同一问题不同措辞命中同一缓存。

用法:
    from fact_cache import FactCache
    cache = FactCache("fact_cache.json", max_entries=200)
    cached = cache.get("腰突能深蹲吗")  # → str or None
    cache.set("腰突能深蹲吗", "建议避免负重深蹲...")
"""

import hashlib
import json
import os
import threading


class FactCache:
    """线程安全的 JSON 文件缓存。"""

    def __init__(self, path: str = "fact_cache.json", max_entries: int = 200):
        self._path = path
        self._max = max_entries
        self._lock = threading.Lock()
        self._data: dict[str, str] = self._load()

    # ----------------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------------

    def get(self, question: str, entities: list[str] | None = None) -> str | None:
        """返回缓存的校验通过回答；无缓存返回 None。

        Args:
            question: 原始查询
            entities: 抽取出的实体列表（用于语义标准化 key）
        """
        key = self._make_key(question, entities)
        with self._lock:
            return self._data.get(key)

    def set(self, question: str, answer: str, entities: list[str] | None = None):
        """存入校验通过的回答；超过最大条数自动淘汰旧条目。"""
        key = self._make_key(question, entities)
        with self._lock:
            self._data[key] = answer
            if len(self._data) > self._max:
                # LRU 淘汰：删除最旧的 20%
                evict_count = max(1, self._max // 5)
                old_keys = list(self._data.keys())[:evict_count]
                for k in old_keys:
                    del self._data[k]
            self._save()

    # ----------------------------------------------------------------
    # 内部
    # ----------------------------------------------------------------

    @staticmethod
    def _make_key(question: str, entities: list[str] | None = None) -> str:
        """MD5(question + sorted entities) 做语义标准化 key。"""
        raw = question.strip()
        if entities:
            raw += "|" + ",".join(sorted(entities))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _load(self) -> dict[str, str]:
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self):
        """原子写入（先写 .tmp 再替换，防中断损坏）。"""
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            pass
