"""pipeline 本地禁忌降级单测（仅静态方法，不加载服务）。"""
from pipeline import PipelineService


def test_local_contraindications_alias_resolution():
    """实体词典提取简称"腰突" → 别名映射到"腰间盘突出" → 命中禁忌动作。"""
    m = PipelineService._local_contraindications(["腰突"])
    assert m, "应命中本地禁忌数据"
    assert "深蹲" in m.get("腰间盘突出", [])
    assert "硬拉" in m.get("腰间盘突出", [])


def test_local_contraindications_exact_name():
    m = PipelineService._local_contraindications(["半月板损伤"])
    assert "深蹲" in m.get("半月板损伤", [])


def test_local_contraindications_unknown_injury_empty():
    assert PipelineService._local_contraindications(["量子力学"]) == {}


def test_local_contraindications_excludes_rehab_actions():
    """仅取「禁忌动作」关系：康复动作（如臀桥）不应出现在黑名单。"""
    m = PipelineService._local_contraindications(["腰突"])
    forbidden = m.get("腰间盘突出", [])
    assert "臀桥" not in forbidden
    assert "平板支撑" not in forbidden


# ----------------------------------------------------------------
# 禁忌拒绝（边界拒绝 + 具体原因）
# ----------------------------------------------------------------

def test_contraindication_reason_lookup():
    """半月板损伤→深蹲 的拒绝原因应来自禁忌数据（而非通用文案）。"""
    r = PipelineService._contraindication_reason(["半月板损伤"], "深蹲")
    assert r and "半月板" in r


def test_contraindication_reason_unknown_action():
    assert PipelineService._contraindication_reason(["半月板损伤"], "瑜伽") is None


def test_reject_message_uses_specific_reason():
    """拒绝文案优先用数据原因，查不到时回退通用文案。"""
    msg = PipelineService._reject_message("深蹲", ["半月板损伤"], "膝盖屈伸负重挤压半月板")
    assert "膝盖屈伸负重挤压半月板" in msg
    assert "明确禁忌动作" in msg
    fallback = PipelineService._reject_message("深蹲", ["半月板损伤"])
    assert "可能加重损伤" in fallback


# ----------------------------------------------------------------
# 阶段进度回调（SSE status 帧数据源）
# ----------------------------------------------------------------

def test_answer_refusal_emits_stage_progress():
    """边界拒绝路径：on_stage 至少收到安全分析阶段播报（不加载服务/Milvus）。"""

    class _DummyRetriever:
        def get_contraindications(self, names):
            return {}   # 空 → 走本地禁忌降级数据源

    stages: list[str] = []
    svc = PipelineService(retriever=_DummyRetriever(), llms={}, gateway=None,
                          fact_engine=None, fact_cache=None, store={})
    result = svc.answer("腰突能深蹲吗", on_stage=stages.append)
    assert result.refusal
    assert any("分析" in s for s in stages), stages   # 拒绝路径至少播报首阶段


# ----------------------------------------------------------------
# 硬性禁忌过滤（含否定语境行保留）
# ----------------------------------------------------------------

def test_hard_filter_removes_violating_lines():
    out, filtered = PipelineService._hard_filter(
        "可以试试平板支撑。\n深蹲可以负重训练。\n臀桥也不错。",
        ["深蹲"])
    assert filtered
    assert "深蹲可以负重训练" not in out
    assert "平板支撑" in out and "臀桥也不错" in out


def test_hard_filter_keeps_negation_lines():
    """「避免深蹲」是安全提醒，应保留而非误删；且未删行时不报「已剔除」警告。"""
    out, filtered = PipelineService._hard_filter(
        "建议避免深蹲。\n可以做臀桥强化臀肌。",
        ["深蹲"])
    assert not filtered
    assert "避免深蹲" in out


def test_hard_filter_mixed_lines():
    """混合场景：违规行删除、否定行保留，警告只列被删的动作。"""
    out, filtered = PipelineService._hard_filter(
        "深蹲训练三组。\n建议避免深蹲。\n可以尝试臀桥。",
        ["深蹲"])
    assert filtered
    assert "深蹲训练三组" not in out
    assert "建议避免深蹲" in out
    assert "深蹲" in out.split("[!]")[1] or "已被自动剔除" in out


def test_hard_filter_equipment_suffix_not_a_hit():
    """2 字动作名后接器械后缀不算命中：「划船」不误删「划船机」行。"""
    out, filtered = PipelineService._hard_filter(
        "可以使用划船机进行有氧训练。\n划船三组。",
        ["划船"])
    assert filtered
    assert "划船机" in out          # 器械行保留
    assert "划船三组" not in out    # 真命中行删除


# ----------------------------------------------------------------
# 会话 store LRU 驱逐
# ----------------------------------------------------------------

def test_session_store_lru_eviction():
    """会话数超过 SESSION_MAX_COUNT 时按最久未访问驱逐。"""
    svc = PipelineService(retriever=None, llms=None, gateway=None, store={})
    for i in range(70):
        svc._append_history(f"s{i}", f"q{i}", f"a{i}")
    assert len(svc._store) <= 64
    # 最早建立的会话应被驱逐
    assert "s0" not in svc._store
    # 最近访问的会话应保留：touch s1 后建立 10 个新会话，s1 仍应存活
    svc2 = PipelineService(retriever=None, llms=None, gateway=None, store={})
    for i in range(64):
        svc2._append_history(f"s{i}", "q", "a")
    svc2._history_messages("s0")    # 读取即访问
    for i in range(10):
        svc2._append_history(f"n{i}", "q", "a")
    assert "s0" in svc2._store      # 最近访问过，LRU 不应驱逐
    assert "s1" not in svc2._store  # 未被访问的最老会话被驱逐


# ----------------------------------------------------------------
# 引用片段清洗（_clean_snippet）
# ----------------------------------------------------------------

def test_clean_snippet_strips_tail_residue():
    """尾部孤立数字/标点残渣（OCR 常见 "2," 尾巴）应被剔除。"""
    out = PipelineService._clean_snippet("正常的训练建议内容 2,", 120)
    assert out == "正常的训练建议内容"


def test_clean_snippet_whitespace_normalized():
    """OCR 行内换行折叠为空格，多空格合并。"""
    out = PipelineService._clean_snippet("杠\n铃 施加于  肩部的压力", 120)
    assert out == "杠 铃 施加于 肩部的压力"


def test_clean_snippet_sentence_boundary_truncation():
    """超长片段停在完整句子边界，不从词中间切断。"""
    text = "深蹲锻炼股四头肌和臀大肌。深蹲时注意膝盖与脚尖方向一致。后面还有更多内容。"
    out = PipelineService._clean_snippet(text, 30)
    # 30 字符窗口内最后一句到「一致。」为止（后半句被截掉，不留半个词）
    assert out.endswith("一致。")
    assert "后面还有" not in out


def test_clean_snippet_no_boundary_falls_back_to_ellipsis():
    """窗口内无句子标点 → 硬截断 + 省略号。"""
    out = PipelineService._clean_snippet("没有句号结尾的长文本" * 8, 40)
    assert out.endswith("…") and len(out) == 41


def test_clean_snippet_empty_and_residue_only():
    assert PipelineService._clean_snippet("") == ""
    assert PipelineService._clean_snippet("  \n 2, ") == ""
