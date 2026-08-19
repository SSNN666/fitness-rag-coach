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
        fusion_mode: str | None = None,
    ):
        """
        Args:
            milvus_client: pymilvus.MilvusClient（Milvus Lite 嵌入式连接）
            bm25_index: rank_bm25.BM25Okapi 实例 + docstore 元组
            neo4j_driver: neo4j.Driver（bolt 连接）；None = 图谱停用（NEO4J_ENABLED=False）
            embedding_fn: callable，输入 text 返回 768 维向量
            w_milvus / w_bm25 / w_neo4j: 三路权重，建议和为 1.0（仅 weighted 模式）
            fusion_threshold: 全局最低分数阈值（仅 weighted 模式）
            reranker: FitnessReranker 实例（可选），用于 LLM listwise 重排序
            milvus_factor / bm25_factor / neo4j_factor: 各路检索扩大倍数
            neo4j_depth / neo4j_max_depth: 单伤病 1-hop / 复合伤病最大 hop 数
            fusion_mode: "weighted"（加权融合）| "rrf"（Reciprocal Rank Fusion）；
                         默认读 config.FUSION_MODE
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
        if fusion_mode is None:
            import config as _cfg
            fusion_mode = getattr(_cfg, "FUSION_MODE", "weighted")
        self._fusion_mode = fusion_mode
        # 语义去重开关（默认关：开启会对全部候选两两余弦 + N 次 embedding，费时费钱）
        import config as _cfg
        self._semantic_dedup_enabled = getattr(_cfg, "FUSION_SEMANTIC_DEDUP_ENABLED", False)
        # 父块回取:子块命中后按 parent_id 聚合回父块(完整上下文),parents.json 由建库脚本写入
        self._parents: dict = self._load_parent_store()
        # 检索降级:可用内存过低时跳过 Milvus/图谱,仅走 BM25(检索侧 OOM 防线)
        self._retrieval_degraded = False
        # 最近一次 query 向量缓存（search_with_scores 内锁下写入；供 grounding 相关性判定复用，
        # 免每次请求重复 embedding。锁外读取方需自行保证时序——pipeline 在锁内读取）
        self._last_query_embedding: list | None = None

    # ================================================================
    # 父块回取 / 检索降级
    # ================================================================

    _PARENT_MAX_CHARS = 5000   # 父块注入上限(令牌预算守卫会再做句子级裁剪)

    def _load_parent_store(self) -> dict:
        """加载 parents.json({parent_id: {page_content, metadata}});缺失/损坏 → 空 dict(降级为子块直出)。"""
        import os as _os
        path = _os.path.join("milvus_data", "parents.json")
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _expand_to_parents(self, docs: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        """子块 → 父块聚合:命中子块按 parent_id 取父块全文,同父块多子块去重。"""
        if not self._parents or not docs:
            return docs
        out: list[tuple[Document, float]] = []
        seen: set = set()
        for doc, score in docs:
            pid = doc.metadata.get("parent_id", "")
            parent = self._parents.get(pid)
            if not parent:
                out.append((doc, score))
                continue
            if pid in seen:
                continue   # 同一父块的多个子块 → 只保留首个(取最高得分)
            seen.add(pid)
            meta = dict(doc.metadata)
            meta.update(parent.get("metadata") or {})
            meta["parent_id"] = pid
            meta["child_source"] = doc.metadata.get("source", "")
            parent_doc = Document(
                page_content=str(parent.get("page_content", ""))[:self._PARENT_MAX_CHARS],
                metadata=meta,
            )
            out.append((parent_doc, score))
        return out

    def set_degraded(self, degraded: bool) -> None:
        """检索降级开关:True = 仅 BM25 稀疏检索(Milvus/图谱跳过)。"""
        self._retrieval_degraded = degraded

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

    def rerank(self, query: str, candidates: list[tuple[Document, float]], top_k: int):
        """LLM 重排（pipeline 锁外调用——云端 LLM 无共享状态；融合检索仍在锁内）。"""
        if self._reranker is None or len(candidates) <= top_k:
            return candidates[:top_k]
        return self._reranker.rerank(query, candidates, top_k=top_k)

    def get_contraindications(self, injury_names: list[str]) -> dict[str, list[str]]:
        """
        查询 Neo4j 获取指定伤病 → 禁忌动作/器械映射。

        Returns:
            {"腰突": ["深蹲","硬拉","罗马尼亚硬拉",...], ...}
        """
        if not injury_names:
            return {}
        if self._neo4j is None:  # 图谱停用（NEO4J_ENABLED=False）
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

    def search_with_scores(self, query: str, k: int = 3,
                           use_rerank: bool | None = None) -> list[tuple[Document, float]]:
        """
        三路独立检索 + 动态权重融合（网关路由） + 去噪去重 + (可选) LLM 重排序。

        Args:
            use_rerank: None=按是否配置 reranker 自动；False=本查询跳过 LLM 重排
                        （分层生成策略：simple 查询跳过重排省时）

        Returns:
            [(Document, final_score), ...] 按得分降序
        """
        # 三路独立检索（使用可配置的扩大倍数，给 reranker 更多候选）
        # 检索降级:低内存时跳过 Milvus/图谱,仅 BM25(稀疏索引内存占用极小)
        if self._retrieval_degraded:
            milvus_results, neo4j_results = [], []
            self._last_query_embedding = None   # 稀疏模式不计算向量
        else:
            # query 向量只嵌入一次：检索 + grounding 相关性判定复用（见 _last_query_embedding 注释）
            query_vec = self._embed(query)
            self._last_query_embedding = query_vec
            milvus_results = self._search_milvus_with_vec(query_vec, k * self.milvus_factor)
            neo4j_results = self._search_neo4j(query, k * self.neo4j_factor)
        bm25_results = self._search_bm25(query, k * self.bm25_factor)

        path_results = [milvus_results, bm25_results, neo4j_results]

        # 融合模式分支：rrf（分数=Σ1/(k+rank)，无数值阈值语义）| weighted（动态权重+阈值去噪）
        if self._fusion_mode == "rrf":
            merged = self._rrf_fusion(path_results, top_k=None)
        else:
            # 网关路由：根据查询类型动态调整权重（仅 weighted 模式有意义）
            w_m, w_b, w_n = self._route_weights(query)
            merged = self._weighted_fusion(
                path_results, [w_m, w_b, w_n],
                top_k=None,  # 不截断，交给 reranker
            )

        # 实体匹配 boost：匹配 query 实体的文档提权，不匹配的降权
        merged = self._entity_match_boost(query, merged)

        # LLM 重排序（分层生成策略可跳过；默认按配置）
        if use_rerank is None:
            use_rerank = self._reranker is not None
        if use_rerank and self._reranker is not None and len(merged) > k:
            merged = self._reranker.rerank(query, merged, top_k=k)

        # 父块回取:子块命中 → 聚合为父块全文(同父块去重)
        merged = self._expand_to_parents(merged)
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
        return self._search_milvus_with_vec(vec, k)

    def _search_milvus_with_vec(self, vec: list[float], k: int) -> list[tuple[Document, float]]:
        """向量检索（复用已计算的 query 向量，search_with_scores 传参免重复 embedding）。"""
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
        if self._neo4j is None:  # 图谱停用（NEO4J_ENABLED=False）→ 空结果，融合不受影响
            return []
        entities = self._extract_entities(query)
        if not entities:
            return []

        # 判断是否需要多跳：≥2 个伤病 或 伤病+独立身体部位 或 复合标记触发
        # 部位名是伤病名的组成部分时不叠加（"膝关节积液"自带"膝关节"，单伤病 1-hop）
        from hyde import classify_query
        injuries = entities.get("injury", [])
        extra_parts = [p for p in entities.get("body_part", [])
                       if not any(p in i for i in injuries)]
        has_compound = (
            len(injuries) >= 2
            or (injuries and extra_parts)
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
        # 关系标签 → 分数（禁忌/康复信号 > 普通关联）
        "禁忌动作": 1.0,
        "康复动作": 0.9,
        "谨慎动作": 0.7,
    }

    @staticmethod
    def _relation_label(relation: str, desc: str) -> str:
        """从关系类型名或描述中提取完整标签（禁忌动作/康复动作/谨慎动作）。

        图谱数据源两种落点都兼容：标签在关系类型（type(r)）或描述（r.description）。
        查不到时返回原始描述（默认分数 0.5 语义）。
        """
        for lbl in ("禁忌动作", "康复动作", "谨慎动作"):
            if lbl in (relation or "") or lbl in (desc or ""):
                return lbl
        return desc or relation

    @staticmethod
    def _score_relation(label: str) -> float:
        """关系标签 → 分数（禁忌 1.0 > 康复 0.9 > 谨慎 0.7 > 默认 0.5）。"""
        for k, s in FitnessRAGRetriever._RELATION_SCORE.items():
            if k in (label or ""):
                return s
        return 0.5  # 默认分数

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
            rel_label = self._relation_label(rec.get("relation", ""), rec.get("rel_desc", ""))
            text = (
                f"[图谱] {rec['src']}({rec['src_type']}) "
                f"--[{rel_label}]--> {rec['target']}({rec['target_type']})"
            )
            rel_desc = rec.get("rel_desc", "")
            if rel_desc:
                text += f": {rel_desc}"

            # 按关系标签评分（禁忌 > 康复 > 谨慎 > 其他默认 0.5）
            score = self._score_relation(rel_label)
            labels = [f"{rec['src_type']}:{rec['src']}", f"{rec['target_type']}:{rec['target']}"]
            # relation 元数据供 _merge_conflicts 检测矛盾标注（禁忌 vs 康复）
            doc = Document(page_content=text, metadata={"source": "neo4j", "hops": 1,
                                                        "entity_labels": labels,
                                                        "relation": rel_label})
            out.append((doc, score))
        out.sort(key=lambda x: x[1], reverse=True)   # 先排序再截断（调用方 fusion 前就取 top-k）
        return out

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

            # 评分：关系标签基础分（禁忌1.0/康复0.9/谨慎0.7/默认0.5）× 深度衰减（hop 越浅越高）；
            # 含禁忌关系路径额外 boost（类型名/描述双源匹配）
            relations = rec.get("relations", []) or []
            # 末段关系即指向目标动作的关系（供 _merge_conflicts 冲突检测；标签归一）
            rel_label = self._relation_label(
                relations[-1] if relations else "", rel_descs[-1] if rel_descs else "")
            tag_score = self._RELATION_SCORE.get(rel_label, 0.5)
            base_score = tag_score / hops
            has_contraind = any("禁忌" in (d or "") for d in rel_descs) or \
                any("禁忌" in (r or "") for r in relations)
            if has_contraind:
                base_score = min(1.0, base_score * NEO4J_PATH_BOOST_CONTRAIND)

            labels = [f"{rec['node_types'][0]}:{rec['path_nodes'][0]}" if rec.get("node_types") and rec.get("path_nodes") else f"injury:{rec['src']}"]
            doc = Document(page_content=text, metadata={"source": "neo4j", "hops": hops,
                                                        "entity_labels": labels,
                                                        "relation": rel_label})
            results.append((doc, base_score))

        merged = self._merge_conflicts(results)
        merged.sort(key=lambda x: x[1], reverse=True)   # 先排序再截断（否则高分路径被挤出 top-k）
        return merged[:k]

    @staticmethod
    def _merge_conflicts(results: list) -> list:
        """检测路径冲突：同一目标动作被标记为禁忌+康复时同时保留并标注。

        关系标签归一后再比较（禁忌/康复/谨慎），避免同类别描述差异（如
        "禁忌动作: 深蹲压迫半月板" vs "禁忌动作: 深蹲挤压椎间盘"）误报冲突。
        """
        seen = {}
        merged = []
        for doc, score in results:
            # 提取路径末端动作名作为去重键
            path_parts = doc.page_content.split(" → ")
            key = path_parts[-1].split(":")[0].strip() if path_parts else ""
            rel = doc.metadata.get("relation", "")
            rel_label = next((lbl for lbl in ("禁忌", "康复", "谨慎") if lbl in rel), rel)
            if key and key in seen:
                prev_doc, _ = seen[key]
                prev_label = prev_doc.metadata.get("_rel_label", "")
                if prev_label and rel_label and prev_label != rel_label:
                    doc.page_content += "\n[冲突] 该动作在不同路径中有矛盾标注，请结合医嘱评估"
            if key:
                seen[key] = (doc, score)
                doc.metadata["_rel_label"] = rel_label
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
        与每条文档的 entity_labels 取交集。有交集 → ×1.5，无交集 → ×0.85。
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

        # 语义去重（FUSION_SEMANTIC_DEDUP_ENABLED 默认关——开启会对全部候选做
        # N 次 embedding + 两两余弦，候选池大时费时费钱；配置开关需重建索引语义）
        if self._semantic_dedup_enabled and len(merged) > 1:
            merged = self._semantic_dedup(merged)

        merged.sort(key=lambda x: x[1], reverse=True)

        if top_k is None:
            return merged
        return merged[:top_k]

    def _rrf_fusion(
        self,
        path_results: list[list[tuple[Document, float]]],
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Reciprocal Rank Fusion：score(d) = Σ_r 1/(k + rank_r(d))。

        各路先按自身得分降序排位，再对每篇文档跨路累加 RRF 分。
          - 天然跨路去重（同一文档多路命中 → 分数累加）
          - 分数是 [0, ~n/k] 的小数值，无明确"最低阈值"语义 → 不应用 self.threshold
          - 动态路由权重（_route_weights）仅对 weighted 模式有意义，本模式忽略
        """
        from config import RRF_K
        k = RRF_K
        doc_map: dict[str, tuple[Document, float]] = {}
        for results in path_results:
            ranked = sorted(results, key=lambda x: x[1], reverse=True)
            for rank, (doc, _score) in enumerate(ranked):
                key = _md5(doc.page_content)
                rrf_score = 1.0 / (k + rank + 1)
                if key in doc_map:
                    prev_doc, prev_score = doc_map[key]
                    doc_map[key] = (prev_doc, prev_score + rrf_score)
                else:
                    doc_map[key] = (doc, rrf_score)

        merged = list(doc_map.values())
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
