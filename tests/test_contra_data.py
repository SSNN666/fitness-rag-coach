"""contra_data 单测：内容指纹（缓存失效）+ 「腰突」重复条目的不可删约束。

两个都是**踩过坑才加的**：
  · 指纹：改禁忌表后 fact_cache 仍吐旧答案（安全数据靠「记得手动 bump 版本号」）
  · 腰突：表里看起来是可以清理的重复条目，实际是图谱路径的承重墙
"""
import contra_data
from contra_data import INJURY_ACTION_MAP, data_fingerprint
from pipeline import PipelineService


# ----------------------------------------------------------------
# 内容指纹
# ----------------------------------------------------------------

def test_fingerprint_is_stable():
    assert data_fingerprint() == data_fingerprint()
    assert len(data_fingerprint()) == 8


def test_fingerprint_changes_when_contraindication_added(monkeypatch):
    """加一条禁忌 → 指纹必须变，否则缓存会继续吐按旧禁忌表的答案。"""
    before = data_fingerprint()
    orig = INJURY_ACTION_MAP["腰间盘突出"]
    monkeypatch.setitem(INJURY_ACTION_MAP, "腰间盘突出",
                        orig + [["引体向上", "禁忌动作", "悬垂牵拉加重腰椎负担"]])
    assert data_fingerprint() != before


def test_fingerprint_changes_when_action_removed(monkeypatch):
    before = data_fingerprint()
    orig = INJURY_ACTION_MAP["半月板损伤"]
    monkeypatch.setitem(INJURY_ACTION_MAP, "半月板损伤", orig[:-1])
    assert data_fingerprint() != before


def test_fingerprint_ignores_reason_wording(monkeypatch):
    """只改原因文案不该让缓存全废——但关系本身一条都不能漏。"""
    before = data_fingerprint()
    orig = INJURY_ACTION_MAP["半月板损伤"]
    monkeypatch.setitem(INJURY_ACTION_MAP, "半月板损伤",
                        [[a, rel, "改了错别字"] for a, rel, _ in orig])
    assert data_fingerprint() == before


def test_fingerprint_changes_when_relation_type_changes(monkeypatch):
    """康复动作 ←→ 禁忌动作 的翻转是安全相关的，必须换指纹。"""
    before = data_fingerprint()
    orig = INJURY_ACTION_MAP["半月板损伤"]
    flipped = [["深蹲", "康复动作" if rel == "禁忌动作" else "禁忌动作", d]
               for a, rel, d in orig if a == "深蹲"] or orig
    monkeypatch.setitem(INJURY_ACTION_MAP, "半月板损伤", flipped)
    assert data_fingerprint() != before


def test_cache_key_carries_data_fingerprint():
    """事实缓存的 key 必须带禁忌表指纹（pipeline._cache_key_entities）。"""
    names = PipelineService._cache_key_entities("腰突能深蹲吗")
    assert any(n == f"cd:{data_fingerprint()}" for n in names), names
    # 提示词版本仍然保留（两者管的是不同东西：pv 管提示词，cd 管禁忌数据）
    assert any(n.startswith("pv:") for n in names), names


# ----------------------------------------------------------------
# 「腰突」条目：看着像重复，其实是承重墙
# ----------------------------------------------------------------

def test_yaotu_entry_unreachable_in_local_path():
    """本地降级路径里「腰突」条目确实不可达（别名归一后不匹配任何比较式）。

    这是 ROADMAP 说它「重复」的依据——但这个依据**只对这一条路径成立**。
    """
    m = PipelineService._local_contraindications(["腰突"])
    assert list(m.keys()) == ["腰间盘突出"], "别名应归一，不应出现「腰突」这个键"


def test_yaotu_entry_must_stay_for_graph_path():
    """⚠️ 但它**不能删**：图谱按本表的键建 Entity 节点，而实体抽取不归一别名。

    `_extract_entities('腰突怎么康复')` → {'injury': ['腰突']}（原样，不改写），
    图谱里的 `(腰突:injury)` 节点正是这个问法的命中来源。
    删掉本条 → 重建图谱后「腰突」的图谱检索整条失效。
    真要做别名归一，得先改实体抽取，那是检索改动、必须重新评测。
    """
    from retriever import FitnessRAGRetriever

    assert "腰突" in INJURY_ACTION_MAP, "删掉它会让「腰突」问句的图谱路径失效"
    assert INJURY_ACTION_MAP["腰突"], "条目不应为空"
    assert INJURY_ACTION_MAP["腰突"] != INJURY_ACTION_MAP["腰间盘突出"], \
        "两条并不完全相同（腰突 只有禁忌动作，缺康复动作）"
    # 实体抽取确实原样输出「腰突」——这就是它必须留在表里的原因
    assert FitnessRAGRetriever._extract_entities("腰突怎么康复")["injury"] == ["腰突"]


def test_aliases_are_not_contra_map_keys_that_leak():
    """别名指向的全称键必须真实存在，否则归一后查不到数据。"""
    for alias, canon in contra_data.INJURY_ALIASES.items():
        assert canon in INJURY_ACTION_MAP, f"别名 {alias} 指向不存在的键 {canon}"
