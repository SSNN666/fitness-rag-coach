"""graph_view 单测：图数据构建 / 别名合并 / 聚焦过滤 / HTML 渲染。"""
import graph_view


def test_load_graph_local_shape():
    g, note = graph_view.load_graph()
    assert note, "数据源说明非空"
    assert len(g["nodes"]) > 50
    assert len(g["links"]) > 50
    assert g["injuries"], "伤病列表非空"


def test_load_graph_alias_merged():
    """简称（腰突）应与全称（腰间盘突出）合并为同一节点。"""
    g, _ = graph_view.load_graph()
    names = {n["name"] for n in g["nodes"]}
    assert "腰突" not in names
    assert "腰间盘突出" in names


def test_load_graph_relations_all_styled():
    g, _ = graph_view.load_graph()
    rels = {l["relation"] for l in g["links"]}
    assert rels <= set(graph_view.RELATION_STYLE), f"缺色映射: {rels - set(graph_view.RELATION_STYLE)}"
    assert {"禁忌动作", "康复动作"} <= rels


def test_filter_graph_focus_subgraph():
    g, _ = graph_view.load_graph()
    sub = graph_view.filter_graph(g, ["腰间盘突出"])
    assert len(sub["nodes"]) < len(g["nodes"])
    assert len(sub["links"]) <= len(g["links"])
    assert any(n["name"] == "腰间盘突出" for n in sub["nodes"])
    # 子图边只连聚焦伤病
    focus_ids = {n["id"] for n in sub["nodes"] if n["name"] == "腰间盘突出"}
    assert all(l["source"] in focus_ids for l in sub["links"])


def test_filter_graph_empty_returns_full():
    g, _ = graph_view.load_graph()
    assert graph_view.filter_graph(g, []) is g


def test_build_html_contains_chart_bootstrap():
    g, _ = graph_view.load_graph()
    html = graph_view.build_html(g)
    assert 'id="kgraph"' in html
    assert "echarts.min.js" in html          # 本地外链
    assert "function(p)" in html             # tooltip formatter 内联为 JS 函数而非字符串
    assert '"__TOOLTIP_FN__"' not in html    # 占位符已被替换


def test_match_focus_alias_resolved():
    """简称（腰突）应归一为图谱全称节点（腰间盘突出）。"""
    g, _ = graph_view.load_graph()
    focus = graph_view.match_focus(g, "腰突能深蹲吗")
    assert "腰间盘突出" in focus
    assert "腰突" not in focus


def test_match_focus_no_hit_returns_empty():
    g, _ = graph_view.load_graph()
    assert graph_view.match_focus(g, "深蹲主要锻炼哪些肌群") == []


def test_match_focus_values_valid_for_multiselect():
    """命中结果必须全部是图谱节点名（multiselect options 约束，否则被 Streamlit 丢弃）。"""
    g, _ = graph_view.load_graph()
    names = set(g["injuries"])
    for q in ["腰突能深蹲吗", "膝盖疼能跑步吗", "肩袖损伤怎么恢复", "半月板损伤加扁平足"]:
        focus = graph_view.match_focus(g, q)
        assert set(focus) <= names, f"{q!r} → {focus}"
