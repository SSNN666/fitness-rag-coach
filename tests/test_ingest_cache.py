"""ingest_cache 单测：SHA256 指纹 / 清单比对 / 嵌入缓存命中（纯本地，零依赖）。"""
import json
import os

from ingest_cache import (
    EmbeddingCache,
    IngestManifest,
    embed_with_cache,
    file_sha256,
    text_sha256,
)


# ================================================================
# 哈希
# ================================================================

class TestHashing:
    def test_file_hash_is_content_based(self, tmp_path):
        """同一内容、不同文件名/位置 → 同一指纹（这才是「重复上传」的正确判据）。"""
        a = tmp_path / "a.txt"
        b = tmp_path / "sub" / "b.txt"
        b.parent.mkdir()
        a.write_text("同样的内容", encoding="utf-8")
        b.write_text("同样的内容", encoding="utf-8")
        assert file_sha256(str(a)) == file_sha256(str(b))

    def test_content_change_changes_hash(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("原文", encoding="utf-8")
        h1 = file_sha256(str(p))
        p.write_text("原文加一个字", encoding="utf-8")
        assert file_sha256(str(p)) != h1

    def test_text_hash_stable(self):
        assert text_sha256("深蹲") == text_sha256("深蹲")
        assert text_sha256("深蹲") != text_sha256("硬拉")


# ================================================================
# 源文件清单
# ================================================================

class TestIngestManifest:
    def _mk(self, tmp_path, name, content):
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return str(p)

    def test_first_run_all_new(self, tmp_path):
        m = IngestManifest(path=str(tmp_path / "m.json"))
        f = self._mk(tmp_path, "a.txt", "x")
        d = m.diff([f])
        assert d.new == [f] and d.needs_rebuild

    def test_unchanged_detected(self, tmp_path):
        m = IngestManifest(path=str(tmp_path / "m.json"))
        f = self._mk(tmp_path, "a.txt", "x")
        m.record([f])

        d = m.diff([f])
        assert d.unchanged == [f]
        assert not d.needs_rebuild          # 内容没变 → 无需重建

    def test_changed_detected(self, tmp_path):
        m = IngestManifest(path=str(tmp_path / "m.json"))
        f = self._mk(tmp_path, "a.txt", "x")
        m.record([f])
        self._mk(tmp_path, "a.txt", "x 改了")

        d = m.diff([f])
        assert d.changed == [f] and d.needs_rebuild

    def test_removed_detected(self, tmp_path):
        m = IngestManifest(path=str(tmp_path / "m.json"))
        f = self._mk(tmp_path, "a.txt", "x")
        m.record([f])
        os.remove(f)

        d = m.diff([])
        assert len(d.removed) == 1 and d.needs_rebuild

    def test_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "m.json")
        f = self._mk(tmp_path, "a.txt", "x")
        IngestManifest(path=path).record([f])

        m2 = IngestManifest(path=path)
        assert m2.diff([f]).unchanged == [f]

    def test_corrupt_manifest_starts_empty(self, tmp_path):
        """清单损坏 → 当作空清单（触发一次全量重建），不能阻断构建。"""
        path = tmp_path / "m.json"
        path.write_text("{ 不是合法 JSON", encoding="utf-8")
        m = IngestManifest(path=str(path))
        f = self._mk(tmp_path, "a.txt", "x")
        assert m.diff([f]).new == [f]

    def test_missing_file_ignored(self, tmp_path):
        m = IngestManifest(path=str(tmp_path / "m.json"))
        d = m.diff([str(tmp_path / "不存在.txt")])
        assert d.new == [] and not d.needs_rebuild

    def test_atomic_write_leaves_no_tmp(self, tmp_path):
        path = str(tmp_path / "m.json")
        m = IngestManifest(path=path)
        m.record([self._mk(tmp_path, "a.txt", "x")])
        assert os.path.isfile(path)
        assert not os.path.exists(f"{path}.tmp")


# ================================================================
# 嵌入缓存
# ================================================================

class TestEmbeddingCache:
    def test_hit_and_miss_accounting(self, tmp_path):
        c = EmbeddingCache(path=str(tmp_path / "c.json"))
        assert c.get("未见过") is None        # miss
        c.put("未见过", [0.1, 0.2])
        assert c.get("未见过") == [0.1, 0.2]   # hit
        assert (c.hits, c.misses) == (1, 1)
        assert c.hit_rate == 0.5

    def test_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "c.json")
        c = EmbeddingCache(path=path)
        c.put("深蹲", [1.0, 2.0])
        c.save()

        assert EmbeddingCache(path=path).get("深蹲") == [1.0, 2.0]

    def test_disabled_cache_never_hits(self, tmp_path):
        c = EmbeddingCache(path=str(tmp_path / "c.json"), enabled=False)
        c.put("x", [1.0])
        assert c.get("x") is None
        assert not os.path.exists(str(tmp_path / "c.json"))   # 禁用时不落盘


class TestEmbedWithCache:
    def _spy(self, calls):
        def fn(texts):
            calls.append(list(texts))
            return [[float(len(t))] for t in texts]
        return fn

    def test_only_misses_are_embedded(self, tmp_path):
        c = EmbeddingCache(path=str(tmp_path / "c.json"))
        c.put("已有", [9.9])

        calls = []
        out = embed_with_cache(c, ["已有", "新的"], self._spy(calls))

        assert calls == [["新的"]]                       # 只嵌入未命中的那条
        assert out[0] == [9.9]                           # 命中项用缓存值
        assert out[1] == [float(len("新的"))]

    def test_all_cached_skips_embed_fn_entirely(self, tmp_path):
        c = EmbeddingCache(path=str(tmp_path / "c.json"))
        for t in ("a", "b", "c"):
            c.put(t, [1.0])

        def _boom(texts):
            raise AssertionError("全部命中时不该调用嵌入函数")

        assert embed_with_cache(c, ["a", "b", "c"], _boom) == [[1.0]] * 3

    def test_order_preserved_with_mixed_hits(self, tmp_path):
        """顺序必须与输入一一对应——错位会把向量配到错误的块上。"""
        c = EmbeddingCache(path=str(tmp_path / "c.json"))
        c.put("b", [2.0])

        texts = ["a", "b", "c"]
        out = embed_with_cache(c, texts, lambda ts: [[1.0], [3.0]])
        assert out == [[1.0], [2.0], [3.0]]

    def test_empty_input(self, tmp_path):
        c = EmbeddingCache(path=str(tmp_path / "c.json"))
        assert embed_with_cache(c, [], lambda ts: []) == []
