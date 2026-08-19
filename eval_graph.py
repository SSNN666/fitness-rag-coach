"""
eval_graph.py —— 图谱检索专项评测（multi-hop 多跳推理）
========================================================
Neo4j 停用（Demo 默认 NEO4J_ENABLED=False）时，用确定性 mock 图谱评测图谱检索逻辑：
单伤病 1-hop 差异化评分 / 复合伤病多跳展开 / 禁忌 boost / 矛盾标注。
`--neo4j` 参数切真实实例（需 NEO4J_ENABLED=True 且图谱已构建），跑同一评测集。

数据源：contra_data.INJURY_ACTION_MAP（与 build_index 图谱构建共用单一数据源），
mock 图谱由该数据自动构造 + 动作→肌群边补充多跳素材——评测可复现、零网络依赖。

用法:
  python -u eval_graph.py            # mock 图谱（默认，确定性）
  python -u eval_graph.py --neo4j    # 真实 Neo4j 实例（需图谱已构建）

指标（确定性，无 LLM）:
  - 单伤病 1-hop：禁忌召回@5 / Top-1 禁忌命中 / 差异化评分校验（禁忌>康复>谨慎）
  - 复合伤病 multi-hop：双伤病路径覆盖 / 平均路径深度 / 深度评分（hop 越浅越高）/ 禁忌 boost
  - 矛盾标注：同一动作被标为禁忌+康复时检出 [冲突]
"""

import sys
from collections import deque

from langchain_core.documents import Document

from retriever import FitnessRAGRetriever

USE_NEO4J = "--neo4j" in sys.argv


# 动作 → 目标肌群（构造多跳路径素材：伤病→禁忌动作→肌群，验证 hop>1 展开）
_MUSCLE_LINKS = {
    "深蹲": [("股四头肌", "TARGETS_MUSCLE", "蹲类动作主目标"),
             ("臀大肌", "TARGETS_MUSCLE", "髋伸主力")],
    "硬拉": [("腘绳肌", "TARGETS_MUSCLE", "髋铰链主目标"),
             ("竖脊肌", "TARGETS_MUSCLE", "脊柱稳定肌群")],
    "臀桥": [("臀大肌", "TARGETS_MUSCLE", "髋伸主力")],
    "推举": [("三角肌", "TARGETS_MUSCLE", "肩部主目标")],
    "面拉": [("菱形肌", "TARGETS_MUSCLE", "上背稳定肌群")],
    # 第二层扩展：肌肉 → 关节（补 3 跳路径素材：伤病→动作→肌群→关节）
    "股四头肌": [("髌骨", "ATTACHES_TO", "股四头肌经髌腱附着髌骨")],
    "腘绳肌": [("膝关节", "STABILIZES", "腘绳肌稳定膝关节后侧")],
    "三角肌": [("肩关节", "PRIMARY_MOVER", "肩外展主驱动")],
}


def build_mock_graph() -> dict:
    """从 contra_data 构造 mock 图谱：{node: {"type", "edges": [(target, rel, desc), ...]}}。"""
    from contra_data import INJURY_ACTION_MAP
    graph: dict = {}
    for injury, actions in INJURY_ACTION_MAP.items():
        graph.setdefault(injury, {"type": "injury", "edges": []})
        for action, rel, reason in actions:
            graph.setdefault(action, {"type": "exercise", "edges": []})
            graph[injury]["edges"].append((action, rel, reason))
    for action, links in _MUSCLE_LINKS.items():
        if action not in graph:
            continue
        for muscle, rel, desc in links:
            graph.setdefault(muscle, {"type": "muscle", "edges": []})
            graph[action]["edges"].append((muscle, rel, desc))
    return graph


class MockGraphDriver:
    """确定性 mock Neo4j driver：execute_query 按 Cypher 特征分发（禁忌名单 / 1-hop / multi-hop）。"""

    def __init__(self, graph: dict, max_depth: int = 3):
        self._g = graph
        self._max_depth = max_depth

    def execute_query(self, cypher, params, database_=None):
        names = params.get("names", [])
        limit = params.get("limit", 10)
        if "collect(DISTINCT a.name)" in cypher:
            return self._contraindications(names)
        if "relationships(path)" in cypher:
            return self._multi_hop(names, limit)
        return self._one_hop(names, limit)

    def _contraindications(self, names):
        out = []
        for name in names:
            node = self._g.get(name)
            if not node:
                continue
            forbidden = [t for t, rel, _d in node["edges"] if "禁忌" in rel]
            out.append({"injury": name, "forbidden": forbidden})
        return out, None, None

    def _one_hop(self, names, limit):
        records = []
        for name in names:
            node = self._g.get(name)
            if not node:
                continue
            for target, rel, desc in node["edges"]:
                if len(records) >= limit:
                    break
                records.append({
                    "src": name, "src_type": node["type"],
                    "relation": rel,          # 关系类型即标签（与真实图谱 type(r) 一致）
                    "rel_desc": desc,         # 描述为原因文本（不含标签，验证双源匹配）
                    "target": target,
                    "target_type": self._g.get(target, {}).get("type", "exercise"),
                })
        return records, None, None

    def _multi_hop(self, names, limit):
        # 每个 start 独立配额（真实 Cypher 全局 LIMIT 会截断后序实体的路径，
        # 模拟合理行为：多实体查询各实体都能展开）
        per_start = max(1, limit // max(len(names), 1))
        records = []
        for start in names:
            start_node = self._g.get(start)
            if not start_node:
                continue
            # BFS 1..max_depth 跳路径展开（防环：路径内不重复访问）
            # 配额按 start 独立计数（全局 records 会因前序 start 先占满而饿死后序实体）
            count = 0
            queue = deque([(start, [start], [start_node["type"]], [], [], 0)])
            while queue and count < per_start:
                node, path_nodes, node_types, rels, rel_descs, depth = queue.popleft()
                if depth > 0:
                    count += 1
                    records.append({
                        "src": start, "src_type": start_node["type"],
                        "relations": list(rels), "rel_descs": list(rel_descs),
                        "path_nodes": list(path_nodes), "node_types": list(node_types),
                        "hops": depth,
                    })
                if depth >= self._max_depth:
                    continue
                for target, rel, desc in self._g.get(node, {}).get("edges", []):
                    if target in path_nodes:
                        continue
                    ttype = self._g.get(target, {}).get("type", "")
                    queue.append((target, path_nodes + [target], node_types + [ttype],
                                  rels + [rel], rel_descs + [desc], depth + 1))
        return records, None, None


def build_retriever(driver) -> FitnessRAGRetriever:
    return FitnessRAGRetriever(
        milvus_client=None,
        bm25_index=(None, []),
        neo4j_driver=driver,
        embedding_fn=lambda t: [0.0],
        fusion_threshold=0.0,
        neo4j_depth=1,
        neo4j_max_depth=3,
    )


# ----------------------------------------------------------------
# 评测
# ----------------------------------------------------------------

def _graph_target(doc: Document) -> str:
    """从图谱 doc 文本提取目标节点名：'[图谱] 腰突(injury) --[RELATES_TO]--> 深蹲(exercise): 禁忌动作...'"""
    if "--> " in doc.page_content:
        return doc.page_content.split("--> ")[1].split("(")[0].strip()
    return ""


def evaluate(retriever, source_label: str) -> dict:
    """返回 (汇总 dict, 校验结果 list[(名称, PASS/FAIL, 说明)])。"""
    from contra_data import INJURY_ACTION_MAP

    checks: list[tuple[str, bool, str]] = []
    summary: dict = {}

    # ---- [1/3] 单伤病 1-hop：禁忌召回 + 差异化评分 ----
    n_injuries = n_recall = n_top1 = 0
    for injury, actions in INJURY_ACTION_MAP.items():
        forbidden = {a for a, rel, _d in actions if "禁忌" in rel}
        if not forbidden:
            continue
        docs = retriever._search_neo4j(f"我有{injury}，训练怎么安排", 5)
        docs = sorted(docs, key=lambda x: x[1], reverse=True)  # 与融合排序一致
        top_actions = {_graph_target(d) for d, _s in docs}
        n_injuries += 1
        if forbidden & top_actions:
            n_recall += 1
        if docs and "禁忌" in (docs[0][0].metadata.get("relation", "")):
            n_top1 += 1

    summary["single_n"] = n_injuries
    summary["contra_recall@5"] = n_recall / max(n_injuries, 1)
    summary["contra_top1"] = n_top1 / max(n_injuries, 1)

    # 差异化评分校验：禁忌(1.0) > 康复(0.9) > 谨慎(0.7)
    docs = retriever._neo4j_one_hop({"injury": ["髌骨软化"]}, 10)
    score_by_rel = {_graph_target(d): s for d, s in docs}
    ok = ("腿伸展" in score_by_rel and "深蹲" in score_by_rel
          and score_by_rel["腿伸展"] > score_by_rel["深蹲"] > 0.5)
    checks.append(("差异化评分（禁忌>谨慎>默认）", ok,
                   f"腿伸展(禁忌)={score_by_rel.get('腿伸展', 0):.1f} vs "
                   f"深蹲(谨慎)={score_by_rel.get('深蹲', 0):.1f}"))

    # ---- [2/3] 复合伤病 multi-hop：双伤病路径覆盖 + 深度评分 ----
    compound_queries = [
        ("腰突合并肩袖损伤怎么练", {"腰突", "肩袖损伤"}),
        ("半月板损伤加骨盆前倾怎么安排", {"半月板损伤", "骨盆前倾"}),
        ("骶髂关节炎和膝内扣能深蹲吗", {"骶髂关节炎", "膝内扣"}),
    ]
    n_compound = n_covered = 0
    hop_depths: list[int] = []
    for q, expected in compound_queries:
        docs = retriever._search_neo4j(q, 5)
        n_compound += 1
        # 双伤病路径覆盖：top-3 路径集合中两个伤病都被召回（各自路径分别命中也算）
        path_text = " ".join(d.page_content for d, _s in docs[:3])
        if all(n in path_text for n in expected):
            n_covered += 1
        hop_depths.extend(d.metadata.get("hops", 0) for d, _s in docs)
        # 禁忌 boost 校验：同一复合查询内，含禁忌路径分数不低于同 hop 的普通路径
        by_hop: dict[int, list[tuple[str, float]]] = {}
        for d, s in docs:
            h = d.metadata.get("hops", 0)
            by_hop.setdefault(h, []).append((d.page_content, s))
        for h, items in by_hop.items():
            contra = [s for t, s in items if "禁忌" in t]
            plain = [s for t, s in items if "禁忌" not in t]
            if contra and plain and max(contra) < max(plain):
                checks.append(("禁忌路径 boost", False, f"{q} hop{h} 禁忌未加权"))
                break

    summary["compound_n"] = n_compound
    summary["compound_cover"] = n_covered / max(n_compound, 1)
    summary["avg_hops"] = sum(hop_depths) / max(len(hop_depths), 1)

    # 深度评分校验：hop 越浅分越高（1-hop 禁忌 boost 后 ≤1.0 > 2-hop 0.5 > 3-hop 0.33）
    docs = retriever._neo4j_multi_hop({"injury": ["腰突", "肩袖损伤"]}, 15, 3)
    by_hop = {}
    for d, s in docs:
        by_hop.setdefault(d.metadata["hops"], []).append(s)
    ok = (by_hop.get(1) and by_hop.get(2) and by_hop.get(3)
          and max(by_hop[1]) > max(by_hop[2]) > max(by_hop[3]))
    checks.append(("深度评分（hop 越浅越高）", ok,
                   f"hop1={max(by_hop.get(1, [0])):.2f} hop2={max(by_hop.get(2, [0])):.2f} "
                   f"hop3={max(by_hop.get(3, [0])):.2f}"))

    # ---- [3/3] 矛盾标注（mock 数据源注入禁忌+康复矛盾边）----
    if retriever._neo4j is not None and not USE_NEO4J:
        g = retriever._neo4j._g
        g["腰突"]["edges"].append(("深蹲", "康复动作", "矛盾标注测试：深蹲标为康复"))
        docs = retriever._search_neo4j("腰突合并骶髂关节炎怎么练", 10)
        conflict = sum(1 for d, _s in docs if "[冲突]" in d.page_content)
        checks.append(("矛盾标注检出", conflict >= 1, f"检出 {conflict} 条 [冲突] 路径"))
    else:
        checks.append(("矛盾标注检出", True, "真实图谱模式跳过（数据源无矛盾边）"))

    return summary, checks


# ----------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------

def main():
    if USE_NEO4J:
        from config import (NEO4J_DATABASE, NEO4J_ENABLED, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER)
        if not NEO4J_ENABLED:
            print("[ERROR] --neo4j 需要 NEO4J_ENABLED=True 且实例已构建图谱")
            sys.exit(1)
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        source_label = "真实 Neo4j 实例"
        print(f"[INFO] 图谱数据源: {source_label}（database={NEO4J_DATABASE}）")
    else:
        driver = MockGraphDriver(build_mock_graph())
        source_label = "mock（contra_data 自动构造，确定性）"
        print(f"[INFO] 图谱数据源: {source_label}")

    retriever = build_retriever(driver)
    summary, checks = evaluate(retriever, source_label)

    print("\n" + "=" * 62)
    print(f"  [图谱检索专项评测] 数据源: {source_label}")
    print("=" * 62)
    print(f"\n[1/3] 单伤病 1-hop（{summary['single_n']} 个含禁忌伤病）")
    print(f"  禁忌召回@5 : {summary['contra_recall@5']:.1%}")
    print(f"  Top-1 禁忌 : {summary['contra_top1']:.1%}")
    print(f"\n[2/3] 复合伤病 multi-hop（{summary['compound_n']} 条双伤病查询）")
    print(f"  双伤病路径覆盖: {summary['compound_cover']:.1%}")
    print(f"  平均路径深度 : {summary['avg_hops']:.2f} hop")
    print(f"\n[3/3] 校验断言")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({detail})")

    n_fail = sum(1 for _n, ok, _d in checks if not ok)
    print(f"\n校验: {len(checks) - n_fail}/{len(checks)} PASS")
    if n_fail:
        print("[RESULT] 存在 FAIL 校验，请检查图谱检索逻辑")
        sys.exit(1)
    print("[RESULT] 图谱检索专项评测全部通过")


if __name__ == "__main__":
    main()
