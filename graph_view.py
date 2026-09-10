"""
graph_view.py —— 伤病禁忌知识图谱可视化（Streamlit 页签）
========================================================
数据源双模式：
  1. 本地模式（默认，NEO4J_ENABLED=False）：contra_data.INJURY_ACTION_MAP
     —— 图谱停机时零外部依赖可用，与 pipeline 的禁忌降级共用单一数据源
  2. Neo4j 模式（NEO4J_ENABLED=True）：直连 Cypher 查询 Entity/RELATES_TO
     —— 连接失败自动回退本地副本（fail-open，与检索层一致）

渲染：ECharts 力导向图（graph series + force layout），echarts.min.js 本地内置
（static/echarts.min.js），离线可用、无 CDN 依赖（国内网络友好）。
  - 节点：伤病（红） / 动作（蓝），别名（腰突→腰间盘突出）自动归一合并
  - 边：禁忌动作（红） / 谨慎动作（黄） / 康复动作（绿），带方向箭头
  - 交互：拖拽 / 滚轮缩放 / 悬停 tooltip / 点选高亮邻接节点

用法:
    from graph_view import load_graph, filter_graph, render
    graph, note = load_graph()
    html = render(graph)                      # st.components.html(html, height=680)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_ECHARTS_PATH = Path(__file__).parent / "static" / "echarts.min.js"

# 关系 → 边颜色（与语义一致：红=危险 黄=谨慎 绿=安全）
RELATION_STYLE = {
    "禁忌动作": "#d9534f",
    "谨慎动作": "#f0ad4e",
    "康复动作": "#5cb85c",
}
_RELATION_DEFAULT_COLOR = "#9aa5b1"

# 节点类别（ECharts categories：0=伤病 1=动作）
CATEGORIES = [
    {"name": "伤病", "itemStyle": {"color": "#d9534f"}},
    {"name": "动作", "itemStyle": {"color": "#4d7fd6"}},
]


# ============================================================
# 数据加载
# ============================================================

def load_graph() -> tuple[dict, str]:
    """返回 (graph, 数据源说明)。graph = {injuries, nodes, links}。

    Neo4j 启用时优先 Cypher，失败自动回退本地副本（不阻断页面）。
    """
    from config import NEO4J_ENABLED
    if NEO4J_ENABLED:
        g = _load_from_neo4j()
        if g is not None:
            return g, "Neo4j 实例（Cypher 实时查询）"
    return _load_from_local(), "contra_data.py 本地副本（28 类伤病）"


def _load_from_local() -> dict:
    """contra_data.INJURY_ACTION_MAP → 图数据；别名键归一合并到全称节点。"""
    from contra_data import INJURY_ACTION_MAP, INJURY_ALIASES

    # (全称伤病, 动作) -> 关系集合（去重，如 腰突/腰间盘突出 重复条目合并）
    edges: dict[tuple[str, str], set[str]] = {}
    for key, acts in INJURY_ACTION_MAP.items():
        canon = INJURY_ALIASES.get(key, key)
        for action, rel, _reason in acts:
            edges.setdefault((canon, action), set()).add(rel)

    return _assemble(edges)


def _load_from_neo4j() -> dict | None:
    """Cypher 直查图谱（与 build_index 写入的 schema 一致：RELATES_TO + r.relation）。"""
    import config as cfg
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            cfg.NEO4J_URI, auth=(cfg.NEO4J_USER, cfg.NEO4J_PASSWORD),
            connection_timeout=8,
        )
        driver.verify_connectivity()
        records, _, _ = driver.execute_query(
            "MATCH (i:Entity)-[r:RELATES_TO]->(a:Entity) "
            "RETURN i.name AS injury, a.name AS action, "
            "       coalesce(r.relation, type(r)) AS relation "
            "LIMIT 1000",
            database_=cfg.NEO4J_DATABASE,
        )
        driver.close()
    except Exception:
        return None  # 图谱不可用 → 调用方回退本地副本

    # 别名归一：与 _load_from_local 口径一致（图谱按 contra_data 原键建节点，
    # 「腰突」与「腰间盘突出」会同时存在 → 不归一会显示成两个重复伤病节点）
    from contra_data import INJURY_ALIASES

    edges: dict[tuple[str, str], set[str]] = {}
    for rec in records:
        if rec["injury"] and rec["action"]:
            canon = INJURY_ALIASES.get(rec["injury"], rec["injury"])
            edges.setdefault((canon, rec["action"]), set()).add(rec["relation"] or "关联")
    return _assemble(edges)


def _assemble(edges: dict[tuple[str, str], set[str]]) -> dict:
    """(injury, action)->rels → ECharts 图数据（节点/边/伤病列表）。"""
    nodes: list[dict] = []
    node_ids: set[str] = set()

    def nid(kind: str, name: str) -> str:
        key = f"{kind}:{name}"
        if key not in node_ids:
            node_ids.add(key)
            nodes.append({
                "id": key,
                "name": name,
                "category": 0 if kind == "inj" else 1,
                "symbolSize": 48 if kind == "inj" else 32,
            })
        return key

    links: list[dict] = []
    injuries: set[str] = set()
    for (injury, action), rels in sorted(edges.items()):
        src, tgt = nid("inj", injury), nid("act", action)
        for rel in rels:
            links.append({
                "source": src, "target": tgt, "relation": rel,
                "source_name": injury, "target_name": action,
            })
        injuries.add(injury)

    return {"injuries": sorted(injuries), "nodes": nodes, "links": links}


def filter_graph(graph: dict, focus: list[str]) -> dict:
    """聚焦伤病子图：保留所选伤病 + 其关联动作；空选择 = 全图。"""
    if not focus:
        return graph
    focus_set = set(focus)
    keep = {n["id"] for n in graph["nodes"] if n["name"] in focus_set}
    links = [l for l in graph["links"] if l["source"] in keep]
    targets = {l["target"] for l in links}
    nodes = [n for n in graph["nodes"] if n["id"] in keep | targets]
    return {"injuries": graph["injuries"], "nodes": nodes, "links": links}


def match_focus(graph: dict, question: str) -> list[str]:
    """从问题文本匹配图谱伤病节点（别名归一），返回可传给 filter_graph 的聚焦列表。

    用途：聊天页「查看禁忌图谱」按钮 —— 拒绝/拦截回答后一键跳到图谱视图并聚焦涉及伤病。
    纯字符串匹配（贪心最长优先防子串误命中），不引入检索/LLM 依赖（app.py 零索引依赖约束）。

    Returns:
        问题中命中的伤病全称列表（仅含图中存在的节点，可直接作 multiselect 值；未命中返回 []）
    """
    from contra_data import INJURY_ALIASES

    names = set(graph["injuries"])
    # 全称 → 全称；别名 → 全称（别名指向的伤病不在图中时忽略）
    alias_to_canon: dict[str, str] = {name: name for name in names}
    for alias, canon in INJURY_ALIASES.items():
        if canon in names:
            alias_to_canon.setdefault(alias, canon)

    hits: list[str] = []
    for alias in sorted(alias_to_canon, key=len, reverse=True):
        if alias in question:
            canon = alias_to_canon[alias]
            if canon not in hits:
                hits.append(canon)
    return hits


# ============================================================
# ECharts option 构建
# ============================================================

_TOOLTIP_FN = (
    "function(p){"
    " if(p.dataType==='edge'){"
    "  return '<b>' + p.data.relation + '</b><br/>'"
    "   + p.data.source_name + ' → ' + p.data.target_name;"
    " }"
    " return (p.data.category===0 ? '伤病' : '动作') + '：<b>' + p.data.name + '</b>';"
    "}"
)


def build_option(graph: dict) -> dict:
    """图数据 → ECharts option dict（force layout）。"""
    data = [
        {**n, "category": n["category"]}
        for n in graph["nodes"]
    ]
    links = [
        {
            "source": l["source"], "target": l["target"],
            "relation": l["relation"],
            "source_name": l["source_name"], "target_name": l["target_name"],
            "lineStyle": {
                "color": RELATION_STYLE.get(l["relation"], _RELATION_DEFAULT_COLOR),
                "curveness": 0.08,
            },
        }
        for l in graph["links"]
    ]
    return {
        "tooltip": {"formatter": "__TOOLTIP_FN__"},
        "legend": [{"data": ["伤病", "动作"], "top": 6, "left": "center"}],
        "series": [{
            "type": "graph",
            "layout": "force",
            "roam": True,
            "data": data,
            "links": links,
            "categories": CATEGORIES,
            "label": {"show": True, "position": "right", "fontSize": 11},
            "edgeSymbol": ["none", "arrow"],
            "edgeSymbolSize": 7,
            "force": {
                "repulsion": 320,
                "edgeLength": [50, 140],
                "gravity": 0.06,
                "friction": 0.25,
            },
            "emphasis": {
                "focus": "adjacency",           # 点选高亮邻接节点/边
                "lineStyle": {"width": 3},
                "itemStyle": {"shadowBlur": 12, "shadowColor": "rgba(0,0,0,0.3)"},
            },
        }],
    }


# ============================================================
# HTML 渲染
# ============================================================

@lru_cache(maxsize=1)
def _echarts_src() -> str:
    """echarts 脚本地址（standalone HTML 内使用，与页面同目录相对路径）。"""
    return "echarts.min.js"


def build_html(graph: dict, height: int = 660) -> str:
    """生成自包含图页面 HTML（写入 static/graph.html 后经 iframe 加载）。

    实测教训：st.components.html 的 srcdoc iframe 中 <script src> 不会被浏览器加载
    （resource timing 无任何请求记录），而真实静态页面无此问题——
    故走 static 文件 + st.iframe 路线，标准浏览器行为，稳定可靠。
    """
    option = build_option(graph)
    # formatter 是 JS 函数不是 JSON 值：先占位后替换
    opt_json = json.dumps(option, ensure_ascii=False).replace('"__TOOLTIP_FN__"', _TOOLTIP_FN)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>伤病禁忌知识图谱</title>
<style>html,body{{margin:0;padding:0;width:100%;height:100%;}}</style>
</head>
<body>
<div id="kgraph" style="width:100%;height:{height}px;"></div>
<script src="{_echarts_src()}"></script>
<script>
(function(){{
  var el = document.getElementById('kgraph');
  var chart = echarts.init(el);
  chart.setOption({opt_json});
  window.addEventListener('resize', function(){{ chart.resize(); }});
}})();
</script>
</body>
</html>
"""


def serve_page(graph: dict, height: int = 680) -> str:
    """写 static/graph.html 并返回 iframe URL（带 mtime 防浏览器缓存旧图）。

    Streamlit 把项目 static/ 挂到 /app/static/（需 server.enableStaticServing，
    已由 .streamlit/config.toml 默认开启）。
    """
    from pathlib import Path
    out = Path(__file__).parent / "static" / "graph.html"
    out.write_text(build_html(graph, height=height), encoding="utf-8")
    import time
    return f"/app/static/graph.html?v={int(out.stat().st_mtime)}"
