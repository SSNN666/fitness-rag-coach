"""
ingest_cache.py —— 源文件去重与嵌入缓存
==========================================
解决两个重复劳动问题：

1. **同一份文件反复摄入**：每次重建索引都重新解析、切块、嵌入全部源文件。
   解答：`IngestManifest` 按 **SHA256** 记住每个源文件的指纹——
   内容没变就不必重新处理（整轮无变化时可直接跳过重建）。

2. **相同内容反复嵌入**：跨文件、跨重建都可能出现一模一样的文本块
   （文档改一个字，其余片段全部照旧），却每次都调一次嵌入 API。
   解答：`EmbeddingCache` 以**内容哈希**为键缓存向量，内容相同直接复用。

为什么用内容哈希而不是 mtime
----------------------------
mtime 会被 checkout / 复制 / 恢复备份改掉，内容却完全没变——用 mtime 判断会
产生无谓的重建。SHA256 只看内容，**同一份文件换个名字、换个位置、重下一次，
指纹都一样**，这才是「重复上传」的正确判据。

边界
----
- 哈希是整文件级的，**不是增量更新**：文件改一个字，整文件的块都会重算
  （但嵌入层会按块内容哈希命中缓存，实际只重算变化的那部分）。
- 清单与缓存都是本地产物，已加入 .gitignore（含源文件路径与内容指纹）。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

DEFAULT_MANIFEST_PATH = "ingest_manifest.json"
DEFAULT_EMBED_CACHE_PATH = "embed_cache.json"

_CHUNK = 1 << 20        # 1MB 分块读取，避免大文件一次性读入内存


def file_sha256(path: str) -> str:
    """整文件 SHA256（流式，大文件也不吃内存）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    """文本内容 SHA256（嵌入缓存的键）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================
# 源文件清单
# ============================================================

@dataclass
class ManifestDiff:
    """一次摄入前的变更比对结果。"""
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def needs_rebuild(self) -> bool:
        return bool(self.new or self.changed or self.removed)

    def describe(self) -> str:
        return (f"新增 {len(self.new)} / 变更 {len(self.changed)} / "
                f"未变 {len(self.unchanged)} / 移除 {len(self.removed)}")


class IngestManifest:
    """源文件指纹清单：`路径 -> {sha256, size, indexed_at}`。

    用途：判断「这次重建是否真的需要跑」。全部文件指纹未变 → 可以直接跳过。
    """

    def __init__(self, path: str | None = None):
        self.path = path or DEFAULT_MANIFEST_PATH
        self._entries: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._entries = data.get("files", {}) if "files" in data else data
        except Exception:
            self._entries = {}       # 缺失/损坏 → 空清单（触发一次全量重建）

    def save(self) -> None:
        # 原子写：与项目其它持久化一致，避免半截文件
        tmp = f"{self.path}.tmp"
        payload = {"version": 1, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "files": self._entries}
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def diff(self, paths: list[str]) -> ManifestDiff:
        """比对给定源文件与清单，返回新增/变更/未变/移除。"""
        d = ManifestDiff()
        seen = set()
        for p in paths:
            if not os.path.isfile(p):
                continue
            key = os.path.abspath(p)
            seen.add(key)
            prev = self._entries.get(key)
            if prev is None:
                d.new.append(p)
            elif prev.get("sha256") != file_sha256(p):
                d.changed.append(p)
            else:
                d.unchanged.append(p)
        d.removed = [e.get("path", k) for k, e in self._entries.items()
                     if k not in seen]
        return d

    def record(self, paths: list[str]) -> None:
        """把这批文件的当前指纹写入清单。"""
        for p in paths:
            if not os.path.isfile(p):
                continue
            self._entries[os.path.abspath(p)] = {
                "path": p,
                "sha256": file_sha256(p),
                "size": os.path.getsize(p),
                "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        self.save()


# ============================================================
# 嵌入缓存
# ============================================================

class EmbeddingCache:
    """内容哈希 → 向量。内容相同的块不会重复调用嵌入 API。

    命中率在重建索引时打印——这是「省了多少」的直接证据。
    """

    def __init__(self, path: str | None = None, enabled: bool = True):
        self.path = path or DEFAULT_EMBED_CACHE_PATH
        self.enabled = enabled
        self._cache: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0
        if enabled:
            self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._cache = data.get("vectors", {})
        except Exception:
            self._cache = {}

    def save(self) -> None:
        if not self.enabled:
            return
        tmp = f"{self.path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "vectors": self._cache},
                          f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception:
            pass      # 缓存写失败不影响索引构建

    def get(self, text: str) -> list[float] | None:
        if not self.enabled:
            return None
        v = self._cache.get(text_sha256(text))
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, text: str, vector: list[float]) -> None:
        if self.enabled:
            self._cache[text_sha256(text)] = list(vector)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def describe(self) -> str:
        return (f"嵌入缓存命中 {self.hits} / 未命中 {self.misses} "
                f"（命中率 {self.hit_rate:.1%}）")


def embed_with_cache(cache: EmbeddingCache, texts: list[str], embed_fn):
    """批量嵌入：优先命中缓存，只对未命中的文本调用 embed_fn。

    Args:
        embed_fn: 接受 list[str]，返回 list[list[float]]（与批量嵌入接口同形）
    Returns:
        list[list[float]]，与 texts 一一对应、顺序一致
    """
    result: list[list[float] | None] = [None] * len(texts)
    missing_idx: list[int] = []
    for i, t in enumerate(texts):
        cached = cache.get(t)
        if cached is not None:
            result[i] = cached
        else:
            missing_idx.append(i)

    if missing_idx:
        fresh = embed_fn([texts[i] for i in missing_idx])
        for slot, vec in zip(missing_idx, fresh):
            result[slot] = list(vec)
            cache.put(texts[slot], list(vec))

    return [r if r is not None else [] for r in result]
