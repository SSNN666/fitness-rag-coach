"""
FitnessReranker —— 基于现有 Ollama LLM 的 listwise 重排序器
============================================================
复用已加载的 qwen2.5:7b，零额外内存开销。
输入: query + 候选文档池 → LLM 排名 → 返回 top_k

用法:
    from reranker import FitnessReranker
    reranker = FitnessReranker(llm)
    reranked = reranker.rerank(query, candidates, top_k=3)
"""

import re
from langchain_core.documents import Document


class FitnessReranker:
    """LLM listwise 重排序器，复用已有 Ollama LLM 实例（零额外内存）。"""

    def __init__(self, llm, max_candidates: int = 15, doc_max_chars: int = 200):
        """
        Args:
            llm: 统一适配器 FallbackChain 实例（build_llm("rerank")，temperature=0 确保确定性排序）
            max_candidates: 送入 LLM 的最大候选数
            doc_max_chars: 每个文档截断字符数（~130 中文字足够判断相关性）
        """
        self._llm = llm
        self.max_candidates = max_candidates
        self.doc_max_chars = doc_max_chars

    # ================================================================
    # 公开接口
    # ================================================================

    def rerank(
        self, query: str, candidates: list[tuple[Document, float]], top_k: int = 3
    ) -> list[tuple[Document, float]]:
        """
        对加权融合后的候选池进行 LLM 重排序。

        Args:
            query: 用户原始查询
            candidates: [(doc, fusion_score), ...] 已按融合分降序排列
            top_k: 返回数量

        Returns:
            重排后的 [(doc, fusion_score), ...]，LLM 判定最相关的在前
        """
        # 候选少时不需要重排
        if len(candidates) <= top_k:
            return candidates

        # 截断候选池控制 prompt 长度
        pool = candidates[:self.max_candidates]
        if len(pool) <= top_k:
            return candidates[:top_k]

        # 调用 LLM 排名
        try:
            ranked_indices = self._llm_listwise_rank(query, pool)
        except Exception:
            # LLM 调用失败 → 回退到融合分排序
            return candidates[:top_k]

        # 按 LLM 输出重排
        reranked = [pool[i] for i in ranked_indices if 0 <= i < len(pool)]

        # 补上未被 LLM 提及的文档（防御性：排在末尾）
        mentioned = set(ranked_indices)
        for i, item in enumerate(pool):
            if i not in mentioned:
                reranked.append(item)

        return reranked[:top_k]

    # ================================================================
    # LLM 排名调用
    # ================================================================

    _RANK_PROMPT = (
        "你是检索排序专家。根据用户查询，将候选文档按相关性从高到低排列。\n"
        "排序标准：\n"
        "- 能直接回答用户问题的文档 → 排最前面\n"
        "- 仅部分相关的文档 → 排中间\n"
        "- 完全无关的文档 → 排最后\n\n"
        "用户查询：{query}\n\n"
        "候选文档：\n{docs}\n\n"
        "请只输出排序后的文档编号，用逗号分隔（例如：3,0,5,1,2,4）。不要输出任何解释。"
    )

    def _llm_listwise_rank(
        self, query: str, pool: list[tuple[Document, float]]
    ) -> list[int]:
        """构造 prompt → LLM 推理 → 解析编号序列。"""
        # 格式化候选文档
        lines = []
        for idx, (doc, _score) in enumerate(pool):
            snippet = doc.page_content.replace("\n", " ")[:self.doc_max_chars]
            lines.append(f"[{idx}] {snippet}")

        prompt_text = self._RANK_PROMPT.format(
            query=query, docs="\n".join(lines)
        )

        response = self._llm.invoke(prompt_text)
        content = response.content if hasattr(response, "content") else str(response)

        return self._parse_rank_output(content, len(pool))

    # ================================================================
    # 输出解析
    # ================================================================

    @staticmethod
    def _parse_rank_output(raw: str, pool_size: int) -> list[int]:
        """
        从 LLM 输出中提取编号序列。
        支持: "3,0,5,1,2,4" / "3 0 5 1 2 4" / "[3,0,5,1,2,4]"

        Returns:
            解析后的索引列表；解析失败返回原始顺序 [0,1,2,...]
        """
        tokens = re.findall(r"\d+", raw)
        if not tokens:
            return list(range(pool_size))

        indices = []
        seen = set()
        for t in tokens:
            i = int(t)
            if 0 <= i < pool_size and i not in seen:
                indices.append(i)
                seen.add(i)

        if not indices:
            return list(range(pool_size))

        return indices
