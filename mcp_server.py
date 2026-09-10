"""
mcp_server.py —— 把康养 RAG 的能力暴露成 MCP 工具
====================================================
把项目从「一个问答应用」变成「其他 Agent 可消费的能力」：任意 MCP 客户端
（Claude Code / Claude Desktop 等）都能调用这里的检索、禁忌判定、图谱多跳
与确定性健康计算。

暴露的工具
----------
| 工具 | 依赖 | 说明 |
|---|---|---|
| `search_knowledge_base` | Milvus + BM25 + Neo4j | 三路检索，返回命中片段与得分 |
| `check_contraindication` | contra_data（+Neo4j 可选） | 某动作对某伤病是否禁忌，含原因 |
| `get_injury_graph` | Neo4j | 伤病关联动作的多跳关系 |
| `calculate_bmi` / `estimate_water_intake` / `heart_rate_zone` | 无 | 确定性健康计算 |

用法
----
    python mcp_server.py                    # stdio（默认；供 MCP 客户端接入）
    python mcp_server.py --transport sse    # SSE
    python mcp_server.py --list             # 只列出工具并自检，不起服务
    python mcp_server.py --selftest         # 本地跑一遍各工具，验证可用性

客户端配置示例（Claude Code 的 mcp 配置）:
    {
      "mcpServers": {
        "fitness-rag": {
          "command": "E:/projects/fitness_rag_coach/.venv/Scripts/python.exe",
          "args": ["E:/projects/fitness_rag_coach/mcp_server.py"]
        }
      }
    }

⚠️ 关键约束：Milvus Lite 是**单进程独占**的。API 服务（api.py）在跑时，
   本服务的检索类工具无法打开向量库；禁忌判定与健康计算不受影响
   （不碰 Milvus）。要在 API 运行时用检索工具，应改为调用 API 的 HTTP 接口。
"""

from __future__ import annotations

import sys

from mcp.server.mcpserver import MCPServer

from config import (
    FUSION_MODE,
    MILVUS_COLLECTION,
    MILVUS_GRPC_OPTIONS,
    MILVUS_URI,
    NEO4J_DATABASE,
    NEO4J_ENABLED,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)
from contra_data import INJURY_ACTION_MAP, INJURY_ALIASES
from health_tools import TOOL_REGISTRY

server = MCPServer(
    name="fitness-rag",
    instructions=(
        "康养知识库检索与伤病安全判定。"
        "涉及具体伤病与动作搭配时，建议先调用 check_contraindication 再给建议；"
        "需要精确健康数值时调用对应的计算工具，不要自行估算。"
    ),
)

# ============================================================
# 懒加载：检索器很重（Milvus + BM25 + 嵌入模型），且 Milvus Lite 单进程独占
# ============================================================

_retriever = None
_retriever_error: str | None = None


def _get_retriever():
    """首次调用时初始化；失败记录原因（不抛异常，让工具返回可读错误）。"""
    global _retriever, _retriever_error
    if _retriever is not None or _retriever_error is not None:
        return _retriever
    try:
        from pymilvus import MilvusClient
        from llm_adapter import build_embeddings
        from retriever import FitnessRAGRetriever, load_bm25_from_pickle
        from config import BM25_INDEX_PATH

        client = MilvusClient(uri=MILVUS_URI, grpc_options=MILVUS_GRPC_OPTIONS)
        client.load_collection(MILVUS_COLLECTION)
        bm25_idx, bm25_docs = load_bm25_from_pickle(BM25_INDEX_PATH)
        driver = None
        if NEO4J_ENABLED:
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(
                    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            except Exception:
                driver = None      # 图谱不可用不影响其余两路
        _retriever = FitnessRAGRetriever(
            milvus_client=client,
            bm25_index=(bm25_idx, bm25_docs),
            neo4j_driver=driver,
            embedding_fn=build_embeddings().embed_query,
        )
    except Exception as e:
        _retriever_error = (
            f"检索器初始化失败：{e}\n"
            f"（Milvus Lite 单进程独占——若 API 服务正在运行，请先停止它，"
            f"或改用 API 的 HTTP 接口）"
        )
    return _retriever


# ============================================================
# 工具 1：知识库检索
# ============================================================

@server.tool()
def search_knowledge_base(query: str, k: int = 5) -> str:
    """在康养知识库中检索（向量 + 关键词 + 图谱三路融合）。

    Args:
        query: 自然语言问题，例如「深蹲主要锻炼哪些肌群」
        k: 返回条数，默认 5

    Returns:
        命中的知识片段及来源与得分；检索不可用时返回错误说明。
    """
    r = _get_retriever()
    if r is None:
        return _retriever_error or "检索器不可用"

    docs = r.search_with_scores(query, k=k, use_rerank=False)
    if not docs:
        return f"未检索到与「{query}」相关的内容（知识库覆盖有限，可换种问法）。"

    lines = [f"共 {len(docs)} 条（融合模式：{FUSION_MODE}）\n"]
    for i, (doc, score) in enumerate(docs, 1):
        src = doc.metadata.get("source", "未知")
        text = " ".join((doc.page_content or "").split())[:300]
        lines.append(f"[{i}] 得分 {score:.3f} | 来源 {src}\n{text}\n")
    return "\n".join(lines)


# ============================================================
# 工具 2：禁忌判定（项目的独家能力，不依赖 Milvus）
# ============================================================

def _canon(name: str) -> str:
    return INJURY_ALIASES.get(name, name)


def _match_injury(name: str) -> tuple[str, list] | None:
    """按别名/全称匹配伤病条目（与 pipeline 同口径：相等或互为子串）。"""
    canon = _canon(name)
    for key, actions in INJURY_ACTION_MAP.items():
        if canon == key or canon in key or key in canon:
            return key, actions
    return None


@server.tool()
def check_contraindication(injury: str, action: str) -> str:
    """判断某个训练动作对某个伤病是否属于禁忌/谨慎/康复动作。

    涉及伤病与动作搭配时**优先调用本工具**，不要凭常识判断。

    Args:
        injury: 伤病名，如「腰间盘突出」「腰突」「半月板损伤」
        action: 动作名，如「硬拉」「深蹲」

    Returns:
        关系判定与原因；该伤病无记录时明确说明「暂无禁忌记录」。
    """
    hit = _match_injury(injury)
    if hit is None:
        return (f"知识库中暂无「{injury}」的禁忌记录。"
                f"（已收录 {len(INJURY_ACTION_MAP)} 类伤病）"
                f"——这不代表该动作安全，请咨询专业人士。")
    key, actions = hit

    matched = [(a, rel, reason) for a, rel, reason in actions if a == action]
    if not matched:
        # 退一步：动作名互为子串（如「深蹲」vs「负重深蹲」）
        matched = [(a, rel, reason) for a, rel, reason in actions
                   if action in a or a in action]
    if not matched:
        known = "、".join(a for a, _r, _d in actions) or "（无）"
        return (f"「{key}」的记录中没有与「{action}」直接相关的条目。\n"
                f"该伤病已记录的动作：{known}\n"
                f"未记录不等于安全，请结合专业意见判断。")

    lines = [f"伤病「{key}」× 动作「{action}」："]
    for a, rel, reason in matched:
        lines.append(f"  · {a} → 判定【{rel}】\n    原因：{reason}")
    return "\n".join(lines)


# ============================================================
# 工具 3：图谱多跳
# ============================================================

@server.tool()
def get_injury_graph(injury: str, max_depth: int = 2) -> str:
    """查询伤病在图谱中的关联路径（多跳），用于发现「间接」禁忌关系。

    例：腰突 → 硬拉（禁忌）、膝内扣 → 臀中肌 → ... 这类跨节点的关联，
    单看知识库文本看不出来。

    Args:
        injury: 伤病名，如「腰间盘突出」
        max_depth: 最大跳数，默认 2（1-3）

    Returns:
        关联路径与关系类型（禁忌动作/康复动作/谨慎动作）。
    """
    if not NEO4J_ENABLED:
        return ("图谱未启用（NEO4J_ENABLED=False），无法查询多跳关系；"
                "可改用 check_contraindication 查本地禁忌副本。")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        depth = max(1, min(int(max_depth), 3))
        cypher = f"""
            MATCH path = (s:Entity {{name: $name}})-[*1..{depth}]->(e:Entity)
            RETURN [n in nodes(path) | n.name] AS nodes,
                   [r in relationships(path) | coalesce(r.relation, type(r))] AS rels,
                   length(path) AS hops
            ORDER BY hops ASC LIMIT 30
        """
        with driver.session(database=NEO4J_DATABASE) as s:
            recs = list(s.run(cypher, name=_canon(injury)))
        driver.close()
    except Exception as e:
        return f"图谱查询失败：{e}（图谱不可用时可改用 check_contraindication）"

    if not recs:
        return f"图谱中没有「{injury}」的关联路径。"

    lines = [f"「{_canon(injury)}」关联路径共 {len(recs)} 条（最多 {depth} 跳）：\n"]
    for r in recs:
        path = " → ".join(r["nodes"])
        rel = "/".join(str(x) for x in r["rels"] if x)
        lines.append(f"  [{r['hops']}跳] {path}\n        关系：{rel}")
    return "\n".join(lines)


# ============================================================
# 工具 4-6：确定性健康计算（不依赖任何外部服务）
# ============================================================

def _make_calculator(tool: dict):
    """把 TOOL_REGISTRY 的 compute 包装成 MCP 工具函数。

    与 pipeline 内的调用不同：MCP 的调用方是 Agent，参数由它显式给出
    （pipeline 里参数走确定性抽取是为了不让模型编造用户体征）。
    """
    name = tool["name"]
    title = tool["title"]

    def _calc(height_cm: float | None = None, weight_kg: float | None = None,
              age: float | None = None) -> str:
        params = {}
        if height_cm:
            params["height"] = float(height_cm)
        if weight_kg:
            params["weight"] = float(weight_kg)
        if age:
            params["age"] = float(age)
        result = tool["compute"](params)
        if result is None:
            return (f"{title}：参数不足，无法计算。"
                    f"请提供所需参数（身高 cm / 体重 kg / 年龄 岁）。")
        return result.content

    _calc.__name__ = name
    _calc.__doc__ = (
        f"{title}。\n\n"
        f"Args:\n"
        f"    height_cm: 身高（厘米）\n"
        f"    weight_kg: 体重（公斤）\n"
        f"    age: 年龄（岁）\n\n"
        f"Returns:\n    计算结果与参考说明（确定性计算，非模型估算）。"
    )
    return _calc


for _tool in TOOL_REGISTRY:
    _fn = _make_calculator(_tool)
    server.tool(name=_tool["name"], description=_tool["title"])(_fn)
    # 绑定到模块命名空间：注册到 server 不等于模块内可见，
    # 自检与单元测试需要按名字直接调用
    globals()[_tool["name"]] = _fn


# ============================================================
# 入口
# ============================================================

def _selftest() -> None:
    """本地跑一遍各工具，验证可用性（不启动 MCP 服务）。"""
    print("=" * 60)
    print("  MCP 工具自检")
    print("=" * 60)

    print("\n[1] check_contraindication('腰突', '硬拉')")
    print(check_contraindication("腰突", "硬拉"))

    print("\n[2] check_contraindication('半月板损伤', '深蹲')")
    print(check_contraindication("半月板损伤", "深蹲"))

    print("\n[3] check_contraindication('不存在的伤病', '深蹲')")
    print(check_contraindication("不存在的伤病", "深蹲"))

    print("\n[4] calculate_bmi(height_cm=175, weight_kg=75)")
    print(calculate_bmi(height_cm=175, weight_kg=75))

    print("\n[5] estimate_water_intake(weight_kg=75)")
    print(estimate_water_intake(weight_kg=75))

    print("\n[6] heart_rate_zone(age=30)")
    print(heart_rate_zone(age=30))

    print("\n[7] search_knowledge_base('深蹲主要锻炼哪些肌群', k=3)")
    print(search_knowledge_base("深蹲主要锻炼哪些肌群", k=3))

    print("\n[8] get_injury_graph('腰突', max_depth=2)")
    print(get_injury_graph("腰突", max_depth=2))
    print()


def main() -> None:
    if "--list" in sys.argv:
        import asyncio
        tools = asyncio.run(server.list_tools())
        print(f"已注册 {len(tools)} 个 MCP 工具：")
        for t in tools:
            print(f"  - {t.name}: {(t.description or '')[:56]}")
        return
    if "--selftest" in sys.argv:
        _selftest()
        return

    transport = "stdio"
    if "--transport" in sys.argv:
        i = sys.argv.index("--transport")
        if i + 1 < len(sys.argv):
            transport = sys.argv[i + 1]
    server.run(transport=transport)


if __name__ == "__main__":
    main()
