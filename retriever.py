"""
FitnessRAGRetriever —— 三路并行检索 + 加权融合
================================================
① Milvus 语义向量检索（替代原 FAISS）
② BM25 关键词全文检索
③ Neo4j 知识图谱一跳关联检索

三路结果加权累加 → 全局阈值去噪 → MD5去重 → 按综合得分排序 → 输出最终上下文

用法:
    from retriever import FitnessRAGRetriever
    retriever = FitnessRAGRetriever(milvus_client, bm25_idx, neo4j_driver)
    docs = retriever.similarity_search("腰突能深蹲吗", k=5)
    scored = retriever.search_with_scores("腰突能深蹲吗", k=5)
"""

import hashlib
import json
import pickle
import re
from collections import defaultdict

import jieba
import numpy as np
from langchain_core.documents import Document
from pymilvus import MilvusClient
from config import MILVUS_COLLECTION, NEO4J_DATABASE


class FitnessRAGRetriever:
    """
    三路并行检索 + 加权融合降噪，替代原 ParentChunkRetriever。

    对外接口（兼容现有 hyde.py / eval_testset.py）:
        - similarity_search(query, k)       → list[Document]
        - invoke(query)                     → list[Document]
        - search_with_scores(query, k)      → list[(Document, float)]
        - search_milvus / search_bm25 / search_neo4j   → 各路独立调用
    """

    def __init__(
        self,
        milvus_client: MilvusClient,
        bm25_index,
        neo4j_driver,
        embedding_fn,
        w_milvus: float = 0.5,
        w_bm25: float = 0.3,
        w_neo4j: float = 0.2,
        fusion_threshold: float = 0.15,
        reranker=None,
        milvus_factor: int = 3,
        bm25_factor: int = 3,
        neo4j_factor: int = 2,
        neo4j_depth: int = 1,
        neo4j_max_depth: int = 3,
    ):
        """
        Args:
            milvus_client: pymilvus.MilvusClient（Milvus Lite 嵌入式连接）
            bm25_index: rank_bm25.BM25Okapi 实例 + docstore 元组
            neo4j_driver: neo4j.Driver（bolt 连接）
            embedding_fn: callable，输入 text 返回 768 维向量
            w_milvus / w_bm25 / w_neo4j: 三路权重，建议和为 1.0
            fusion_threshold: 全局最低分数阈值
            reranker: FitnessReranker 实例（可选），用于 LLM listwise 重排序
            milvus_factor / bm25_factor / neo4j_factor: 各路检索扩大倍数
            neo4j_depth / neo4j_max_depth: 单伤病 1-hop / 复合伤病最大 hop 数
        """
        self._milvus = milvus_client
        self._bm25_idx, self._bm25_docs = bm25_index  # 解包 (BM25Okapi, doc_list)
        self._neo4j = neo4j_driver
        self._embed = embedding_fn
        self.w_milvus = w_milvus
        self.w_bm25 = w_bm25
        self.w_neo4j = w_neo4j
        self.threshold = fusion_threshold
        self._reranker = reranker
        self.milvus_factor = milvus_factor
        self.bm25_factor = bm25_factor
        self.neo4j_factor = neo4j_factor
        self.neo4j_depth = neo4j_depth
        self.neo4j_max_depth = neo4j_max_depth

    # ================================================================
    # 对外接口
    # ================================================================

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        """三路检索 + 融合去噪 → 返回 top-k Document（兼容现有调用方）。"""
        scored = self.search_with_scores(query, k)
        return [doc for doc, _ in scored]

    def invoke(self, query: str) -> list[Document]:
        """LangChain BaseRetriever 兼容接口。"""
        return self.similarity_search(query)

    def get_contraindications(self, injury_names: list[str]) -> dict[str, list[str]]:
        """
        查询 Neo4j 获取指定伤病 → 禁忌动作/器械映射。

        Returns:
            {"腰突": ["深蹲","硬拉","罗马尼亚硬拉",...], ...}
        """
        if not injury_names:
            return {}

        cypher = """
            MATCH (i:Entity {type: 'injury'})-[r:RELATES_TO]->(a:Entity)
            WHERE i.name IN $names
              AND (r.relation = '禁忌动作' OR r.relation CONTAINS '禁忌')
            RETURN i.name AS injury, collect(DISTINCT a.name) AS forbidden
        """
        try:
            records, _, _ = self._neo4j.execute_query(
                cypher, {"names": injury_names},
                database_=NEO4J_DATABASE,
            )
        except Exception:
            return {}

        return {rec["injury"]: rec["forbidden"] for rec in records}

    def search_with_scores(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """
        三路独立检索 + 动态权重融合（网关路由） + 去噪去重 + (可选) LLM 重排序。

        Returns:
            [(Document, final_score), ...] 按得分降序
        """
        # 三路独立检索（使用可配置的扩大倍数，给 reranker 更多候选）
        milvus_results = self._search_milvus(query, k * self.milvus_factor)
        bm25_results = self._search_bm25(query, k * self.bm25_factor)
        neo4j_results = self._search_neo4j(query, k * self.neo4j_factor)

        # 网关路由：根据查询类型动态调整权重
        w_m, w_b, w_n = self._route_weights(query)

        # 加权融合：不截断，返回所有通过阈值的候选
        merged = self._weighted_fusion(
            [milvus_results, bm25_results, neo4j_results],
            [w_m, w_b, w_n],
            top_k=None,  # 不截断，交给 reranker
        )

        # 实体匹配 boost：匹配 query 实体的文档提权，不匹配的降权
        merged = self._entity_match_boost(query, merged)

        # LLM 重排序（如果启用）
        if self._reranker is not None and len(merged) > k:
            merged = self._reranker.rerank(query, merged, top_k=k)

        return merged[:k]

    # ----------------------------------------------------------------
    # 网关路由：查询 → 动态权重
    # ----------------------------------------------------------------

    @staticmethod
    def _route_weights(query: str) -> tuple[float, float, float]:
        """
        根据查询中是否涉及伤病/体态/疼痛 → 动态分配三路权重。

        simple:          普通问答，向量为主
        single_injury:   单伤病，适度提升图谱
        compound_injury: 复合伤病，图谱主导（多跳禁忌/康复关联优先）

        Returns:
            (w_milvus, w_bm25, w_neo4j)
        """
        # 复用已有实体抽取（纯字符串匹配，~0.01ms）
        entities = FitnessRAGRetriever._extract_entities(query)
        has_injury = bool(entities.get("injury"))
        has_body_part = bool(entities.get("body_part"))

        # 无伤病/无体态 → 普通问答
        if not has_injury and not has_body_part:
            from config import ROUTE_WEIGHTS_DEFAULT
            return ROUTE_WEIGHTS_DEFAULT

        # 复合伤病（≥2 伤病 或 伤病+身体部位）→ 图谱主导
        injury_count = len(entities.get("injury", []))
        if injury_count >= 2 or (has_injury and has_body_part):
            from config import ROUTE_WEIGHTS_COMPOUND
            return ROUTE_WEIGHTS_COMPOUND

        # 单伤病 → 图谱提权
        from config import ROUTE_WEIGHTS_SINGLE
        return ROUTE_WEIGHTS_SINGLE

    # ================================================================
    # 路1: Milvus 语义向量检索
    # ================================================================

    def _search_milvus(self, query: str, k: int) -> list[tuple[Document, float]]:
        """
        向量语义检索：query → embedding → Milvus COSINE search。

        Returns:
            [(Document, cosine_score), ...]  cosine_score ∈ [0, 1]
        """
        vec = self._embed(query)
        results = self._milvus.search(
            collection_name=MILVUS_COLLECTION,
            data=[vec],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=k,
            output_fields=["page_content", "metadata_json", "entity_labels"],
        )
        hits = results[0]  # 只有一个 query vector
        out = []
        for hit in hits:
            entity = hit.get("entity", {})
            doc = Document(
                page_content=entity.get("page_content", ""),
                metadata=_parse_json_safe(entity.get("metadata_json", "{}")),
            )
            out.append((doc, float(hit["distance"])))  # COSINE distance → similarity
        return out

    # ================================================================
    # 路2: BM25 关键词全文检索
    # ================================================================

    def _search_bm25(self, query: str, k: int) -> list[tuple[Document, float]]:
        """
        BM25 关键词检索：jieba 分词 → BM25 打分 → 取 top-k。

        Returns:
            [(Document, normalized_bm25_score), ...]  归一化到 [0, 1]
        """
        tokens = list(jieba.cut(query))
        scores = self._bm25_idx.get_scores(tokens)
        # 取 top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        # Min-max 归一化
        score_min = scores.min()
        score_max = scores.max()
        denom = score_max - score_min if score_max > score_min else 1.0

        out = []
        for idx in top_indices:
            raw = scores[idx]
            norm = (raw - score_min) / denom
            doc = self._bm25_docs[idx]
            out.append((doc, float(norm)))
        return out

    # ================================================================
    # 路3: Neo4j 知识图谱关联检索
    # ================================================================

    # 多类型实体词典（替代旧正则，贪心最长匹配防子串误匹配）
    _ENTITY_DICT: dict[str, list[str]] = {
        "injury": [
            "腰间盘突出", "腰突", "腰肌劳损", "腰痛", "腰椎间盘", "腰椎",
            "半月板损伤", "半月板撕裂", "半月板",
            "肩袖损伤", "肩袖撕裂", "肩峰撞击", "肩周炎",
            "颈椎病", "颈椎间盘突出", "颈椎曲度变直",
            "网球肘", "高尔夫球肘", "腕管综合征",
            "膝内扣", "膝超伸", "髌骨软化", "髌腱炎", "膝关节积液",
            "骨盆前倾", "骨盆后倾", "脊柱侧弯",
            "扁平足", "足底筋膜炎", "跟腱炎",
            "富贵包", "圆肩", "驼背", "头前伸", "高低肩", "翼状肩",
            "髂胫束综合征", "弹响髋", "弹响肩",
            "踝关节扭伤", "膝关节扭伤", "腕关节扭伤",
            "肌肉拉伤", "韧带撕裂", "关节脱位",
            "坐骨神经痛", "椎管狭窄", "骨质增生",
            "滑囊炎", "腱鞘炎", "关节炎", "筋膜炎",
            "臀肌失忆症", "腹直肌分离",
            "梨状肌综合征", "骶管狭窄", "梨状肌",
            "骶髂关节炎", "骶髂关节紊乱",
            "椎间盘突出", "椎间盘膨出", "椎间盘脱出",
            "腰椎管狭窄", "腰椎滑脱", "腰椎退行性变",
            "颈椎间盘突出", "胸椎间盘突出",
            "肩胛骨疼痛", "肩胛骨不稳",
            "髌骨脱位", "髌骨不稳",
            "腕关节疼痛", "腕关节不稳定",
            "髋关节撞击", "髋臼盂唇撕裂",
            "踝关节不稳", "踝关节撞击",
            "跟骨骨刺", "跖骨痛", "拇外翻",
        ],
        "exercise": [
            "深蹲", "卧推", "硬拉", "划船", "弯举", "推举", "飞鸟", "卷腹",
            "平板支撑", "臀桥", "引体向上", "箭步蹲", "腿举", "面拉",
            "高位下拉", "坐姿划船", "山羊挺身", "俄罗斯转体", "早安式",
            "猫牛式", "反向卷腹", "悬垂举腿", "哑铃耸肩", "杠铃耸肩",
            "立姿划船", "绳索下压", "蝴蝶机夹胸", "侧平举", "窄距卧推",
            "保加利亚分腿蹲", "罗马尼亚硬拉", "相扑硬拉", "架上硬拉",
            "上斜哑铃卧推", "下斜哑铃卧推", "俯身哑铃划船", "俯身飞鸟",
            "站姿杠铃推举", "哑铃集中弯举", "锤式弯举", "俯身臂屈伸",
            "过头三头肌伸展", "站姿提踵", "坐姿提踵", "高脚杯深蹲",
            "侧弓步蹲", "单腿臀桥", "侧平板支撑", "绳索交叉夹胸",
            "蝴蝶机反向飞鸟", "腿弯举", "腿伸展", "卷腹机卷腹",
            "坐姿髋外展", "坐姿髋内收", "辅助引体向上", "辅助双杠臂屈伸",
            "仰卧哑铃拉举", "直腿抬高", "收下巴训练", "双杠臂屈伸",
            "臀中肌激活", "足弓训练", "足底滚球", "髋屈肌拉伸",
        ],
        "muscle": [
            "股四头肌", "腘绳肌", "臀大肌", "臀中肌", "臀小肌",
            "背阔肌", "胸大肌", "三角肌", "肱二头肌", "肱三头肌",
            "竖脊肌", "腹直肌", "腹斜肌", "斜方肌", "核心肌群",
            "菱形肌", "腓肠肌", "比目鱼肌", "内收肌", "肱桡肌",
            "前锯肌", "腰方肌", "髂腰肌", "阔筋膜张肌",
        ],
        "equipment": [
            "杠铃", "哑铃", "弹力带", "龙门架", "徒手", "单杠", "双杠",
            "罗马椅", "蝴蝶机", "腿举机", "腿弯举机", "腿伸展机",
            "卷腹机", "髋外展机", "髋内收机", "坐姿提踵机",
            "辅助引体机", "辅助双杠机", "史密斯机", "壶铃", "药球",
            "TRX", "泡沫轴", "瑜伽垫", "瑞士球", "绳索", "杠铃片",
        ],
        "population": [
            "大体重", "肥胖", "超重", "新手", "初学者", "入门",
            "老年人", "中老年", "孕妇", "产后", "青少年",
            "久坐人群", "办公室人群", "产后恢复",
        ],
        "body_part": [
            "膝盖", "膝关节", "腰椎", "颈椎", "肩关节", "肩部",
            "手腕", "肘部", "脚踝", "踝关节", "髋关节", "髋部",
            "下背", "上背", "核心", "小腿", "大腿", "前臂",
            "肩胛骨", "足弓", "骨盆",
        ],
    }

    @classmethod
    def _extract_entities(cls, query: str) -> dict[str, list[str]]:
        """从查询中按类型抽取实体（贪心最长匹配，防子串误判）。

        Returns: {type: [names], ...}  例: {"injury":["腰突"], "exercise":["深蹲"]}
        """
        found: dict[str, list[str]] = {}
        for etype, keywords in cls._ENTITY_DICT.items():
            sorted_kw = sorted(keywords, key=len, reverse=True)
            matched = []
            remaining = query
            for kw in sorted_kw:
                if kw in remaining:
                    matched.append(kw)
                    remaining = remaining.replace(kw, "", 1)
            if matched:
                found[etype] = matched
        return found

    def _search_neo4j(self, query: str, k: int) -> list[tuple[Document, float]]:
        """
        图谱检索：抽取实体 → 单伤病 1-hop / 复合伤病多跳。

        Returns:
            [(Document(context_text), score), ...]  score 按路径深度和关系类型差异化
        """
        entities = self._extract_entities(query)
        if not entities:
            return []

        # 判断是否需要多跳：≥2 个伤病 或 伤病+身体部位 或 复合标记触发
        from hyde import classify_query
        has_compound = (
            len(entities.get("injury", [])) >= 2
            or (entities.get("injury") and entities.get("body_part"))
            or classify_query(query) == "compound_injury"
        )
        depth = self.neo4j_max_depth if has_compound else self.neo4j_depth

        if depth <= 1:
            return self._neo4j_one_hop(entities, k)
        else:
            return self._neo4j_multi_hop(entities, k, depth)

    # ----------------------------------------------------------------
    # 1-hop 查询（改进评分）
    # ----------------------------------------------------------------

    _RELATION_SCORE = {
        # 关系类型 → 分数（禁忌/康复信号 > 普通关联）
        "禁忌动作": 1.0,
        "康复动作": 0.9,
        "谨慎动作": 0.7,
    }

    def _neo4j_one_hop(self, entities: dict, k: int) -> list[tuple[Document, float]]:
        """MATCH (e)-[r]->(n) WHERE e.name IN $names，按关系类型差异化评分。"""
        all_names = list(set(n for names in entities.values() for n in names))
        if not all_names:
            return []

        cypher = """
            MATCH (e:Entity)-[r]->(n:Entity)
            WHERE e.name IN $names
            RETURN e.name AS src, e.type AS src_type,
                   type(r) AS relation, r.description AS rel_desc,
                   n.name AS target, n.type AS target_type
            LIMIT $limit
        """
        try:
            records, _, _ = self._neo4j.execute_query(
                cypher, {"names": all_names, "limit": k},
                database_=NEO4J_DATABASE,
            )
        except Exception:
            return []

        out = []
        for rec in records:
            text = (
                f"[图谱] {rec['src']}({rec['src_type']}) "
                f"--[{rec['relation']}]--> {rec['target']}({rec['target_type']})"
            )
            rel_desc = rec.get("rel_desc", "")
            if rel_desc:
                text += f": {rel_desc}"

            # 按关系类型评分（禁忌 > 康复 > 谨慎 > 其他默认 0.5）
            score = self._score_relation(rel_desc)
            labels = [f"{rec['src_type']}:{rec['src']}", f"{rec['target_type']}:{rec['target']}"]
            doc = Document(page_content=text, metadata={"source": "neo4j", "hops": 1, "entity_labels": labels})
            out.append((doc, score))
        return out

    @staticmethod
    def _score_relation(desc: str) -> float:
        """从关系描述中提取关系标签并映射到分数。"""
        for label, s in FitnessRAGRetriever._RELATION_SCORE.items():
            if label in (desc or ""):
                return s
        return 0.5  # 默认分数

    # ----------------------------------------------------------------
    # 多跳查询（核心新增）
    # ----------------------------------------------------------------

    def _neo4j_multi_hop(self, entities: dict, k: int, max_depth: int = 3) -> list[tuple[Document, float]]:
        """
        多跳路径展开 → 差异化评分 → 冲突合并。

        路径示例（腰突 + 腘绳肌拉伤）:
          hop1: 腰突 → RELATES_TO → 深蹲(禁忌)
          hop2: 深蹲 → TARGETS_MUSCLE → 股四头肌
          hop3: 腘绳肌拉伤 → RELATES_TO → 罗马尼亚硬拉(禁忌)
        """
        all_names = list(set(n for names in entities.values() for n in names))
        if not all_names:
            return []

        cypher = f"""
            MATCH path = (start:Entity)-[*1..{max_depth}]->(end:Entity)
            WHERE start.name IN $names
            RETURN start.name AS src, start.type AS src_type,
                   [rel in relationships(path) | type(rel)] AS relations,
                   [rel in relationships(path) | coalesce(rel.description, '')] AS rel_descs,
                   [node in nodes(path) | node.name] AS path_nodes,
                   [node in nodes(path) | node.type] AS node_types,
                   length(path) AS hops
            ORDER BY hops ASC
            LIMIT $limit
        """
        try:
            records, _, _ = self._neo4j.execute_query(
                cypher, {"names": all_names, "limit": k * 3},
                database_=NEO4J_DATABASE,
            )
        except Exception:
            # 多跳失败回退 1-hop
            return self._neo4j_one_hop(entities, k)

        from config import NEO4J_PATH_BOOST_CONTRAIND

        results = []
        for rec in records:
            path_nodes = rec["path_nodes"]
            hops = rec["hops"]
            rel_descs = rec.get("rel_descs", [])

            # 路径文本：start → mid → ... → end
            path_text = " → ".join(path_nodes)
            # 拼接关系描述
            desc_text = " | ".join(d for d in rel_descs if d)

            text = f"[图谱·{hops}跳] {path_text}"
            if desc_text:
                text += f": {desc_text}"

            # 评分：深度越浅越高；含禁忌关系 boost
            base_score = 1.0 / hops
            has_contraind = any("禁忌" in (d or "") for d in rel_descs)
            if has_contraind:
                base_score = min(1.0, base_score * NEO4J_PATH_BOOST_CONTRAIND)

            labels = [f"{rec['node_types'][0]}:{rec['path_nodes'][0]}" if rec.get("node_types") and rec.get("path_nodes") else f"injury:{rec['src']}"]
            doc = Document(page_content=text, metadata={"source": "neo4j", "hops": hops, "entity_labels": labels})
            results.append((doc, base_score))

        return self._merge_conflicts(results)[:k]

    @staticmethod
    def _merge_conflicts(results: list) -> list:
        """检测路径冲突：同一目标动作被标记为禁忌+康复时同时保留并标注。"""
        seen = {}
        merged = []
        for doc, score in results:
            # 提取路径末端动作名作为去重键
            path_parts = doc.page_content.split(" → ")
            key = path_parts[-1].split(":")[0].strip() if path_parts else ""
            if key and key in seen:
                prev_doc, _ = seen[key]
                prev_rel = prev_doc.metadata.get("relation", "")
                cur_rel = doc.metadata.get("relation", "")
                if prev_rel and cur_rel and prev_rel != cur_rel:
                    doc.page_content += "\n[冲突] 该动作在不同路径中有矛盾标注，请结合医嘱评估"
            if key:
                seen[key] = (doc, score)
            merged.append((doc, score))
        return merged

    def _semantic_dedup(
        self, merged: list[tuple[Document, float]]
    ) -> list[tuple[Document, float]]:
        """
        语义去重：对融合后的候选文档两两计算余弦相似度。
        相似度 > FUSION_SEMANTIC_DEDUP_THRESHOLD 的文档对中只保留得分更高者。
        至少保留 FUSION_MIN_DOCS 条文档（防御性：避免全部被误判为重复）。
        """
        import config as _cfg
        threshold = getattr(_cfg, "FUSION_SEMANTIC_DEDUP_THRESHOLD", 0.85)
        min_docs = getattr(_cfg, "FUSION_MIN_DOCS", 3)

        if len(merged) <= min_docs:
            return merged

        # 嵌入所有候选文档
        texts = [doc.page_content for doc, _ in merged]
        embeddings = [np.array(self._embed(t)) for t in texts]

        kept: list[tuple[Document, float]] = []
        for i, (doc, score) in enumerate(merged):
            is_dup = False
            for j in range(len(kept)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9
                )
                if sim >= threshold:
                    is_dup = True
                    break
            if not is_dup or len(kept) < min_docs:
                kept.append((doc, score))

        return kept

    def _entity_match_boost(
        self, query: str, merged: list[tuple[Document, float]]
    ) -> list[tuple[Document, float]]:
        """
        实体匹配 boost：query 抽取实体（exercise/muscle/equipment/injury/body_part），
        与每条文档的 entity_labels 取交集。有交集 → boost ×1.2，无交集 → penalize ×0.6。
        query 无实体时跳过（普通问答不分类型过滤）。
        """
        q_entities = set()
        for etype, names in self._extract_entities(query).items():
            for n in names:
                q_entities.add(f"{etype}:{n}")

        if not q_entities:
            return merged

        boosted = []
        for doc, score in merged:
            labels = set(doc.metadata.get("entity_labels", []))
            # PDF chunk 无预存标签 → 运行时从 page_content 抽取
            if not labels:
                doc_entities = self._extract_entities(doc.page_content)
                for etype, names in doc_entities.items():
                    for n in names:
                        labels.add(f"{etype}:{n}")
            if labels & q_entities:
                boosted.append((doc, score * 1.5))  # 匹配 → 强提权
            else:
                boosted.append((doc, score * 0.85))  # 不匹配 → 轻微降权
        return boosted

    # ================================================================
    # 加权融合 + 降噪去重
    # ================================================================

    def _weighted_fusion(
        self,
        path_results: list[list[tuple[Document, float]]],
        weights: list[float],
        top_k: int = 3,
    ) -> list[tuple[Document, float]]:
        """
        三路结果加权融合：
          1. 对每条唯一文档（MD5 去重），加权累加各路得分
          2. 过滤 total_score < threshold 的低质量噪声
          3. 按最终得分降序排序，取 top_k

        Args:
            path_results: [[(doc, score), ...], ...] 每路检索结果
            weights: [w_milvus, w_bm25, w_neo4j]
            top_k: 返回条数

        Returns:
            [(Document, final_score), ...] 按得分降序
        """
        # key = MD5 → (Document, accumulated_score)
        doc_map: dict[str, tuple[Document, float]] = {}

        for path_idx, results in enumerate(path_results):
            w = weights[path_idx]
            if w == 0.0:
                continue
            for doc, score in results:
                key = _md5(doc.page_content)
                if key in doc_map:
                    _, acc_score = doc_map[key]
                    doc_map[key] = (doc, acc_score + w * score)
                else:
                    doc_map[key] = (doc, w * score)

        # 过滤 & 排序
        merged = [
            (doc, score)
            for doc, score in doc_map.values()
            if score >= self.threshold
        ]

        # 语义去重（如启用）
        if len(merged) > 1:
            merged = self._semantic_dedup(merged)

        merged.sort(key=lambda x: x[1], reverse=True)

        if top_k is None:
            return merged
        return merged[:top_k]


# ================================================================
# 工具函数
# ================================================================

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _parse_json_safe(raw: str) -> dict:
    """安全解析 JSON 字符串，失败返回空 dict。"""
    import json
    try:
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def load_bm25_from_pickle(path: str):
    """从 pickle 文件加载 BM25 索引 + 文档列表。"""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_bm25_to_pickle(bm25_index, docs: list[Document], path: str) -> None:
    """保存 BM25 索引 + 文档列表到 pickle。"""
    with open(path, "wb") as f:
        pickle.dump((bm25_index, docs), f)
