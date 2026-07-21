"""
CRAG 联网检索 —— 博查 AI 搜索（国内可用，个人免费套餐）
========================================================
- 博查 Web Search API：`POST https://api.bochaai.com/v1/web-search`
- jieba 分词过滤噪音结果
- 异常降级：返回 None → 调用方用 LLM closed-book recall 兜底

用法:
    from crag_search import crag_retrieve, search_fitness, build_search_query
    context = crag_retrieve("梨状肌综合征 康复动作", failed=["动作","禁忌"])
"""

import sys
import requests
from config import CRAG_MAX_RESULTS, CRAG_TIMEOUT

BOCHA_SEARCH_URL = "https://api.bochaai.com/v1/web-search"


def _get_api_key() -> str:
    """读取 Bocha API Key。优先从 config 模块读，importlib.reload 清空时用硬编码兜底。"""
    cfg = sys.modules.get("config")
    if cfg is not None:
        key = getattr(cfg, "BOCHA_API_KEY", "")
        if key:
            return key
    # 兜底（importlib.reload 可能短暂清空 config 属性）
    return "sk-fd46e722a04749f6b6d758317fa535fc"


def build_search_query(question: str, failed_categories: list[str]) -> str:
    """根据失败类别构造精准搜索词。"""
    cat = failed_categories[0] if failed_categories else ""
    templates = {
        "动作": '"{q}" 禁忌动作 康复训练',
        "康复": '"{q}" 恢复周期 康复指南',
        "负重": '"{q}" 负重限制 训练安全',
        "禁忌": '"{q}" 禁忌注意事项',
    }
    template = templates.get(cat, '"{q}" 症状 治疗 康复')
    return template.format(q=question[:40])


def search_fitness(query: str, max_results: int = CRAG_MAX_RESULTS) -> list[dict]:
    """博查 Web Search API 搜索，返回 [{title, snippet, url}, ...]。(v4-final)"""
    try:
        import logging
        logging.getLogger("gateway").info("crag_v4", extra={"extra_fields": {"key_ok": bool(_get_api_key()), "q": query[:30]}})
        resp = requests.post(
            BOCHA_SEARCH_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_get_api_key()}",
            },
            json={"query": query, "count": max_results},
            timeout=CRAG_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("data", {}).get("webPages", {}).get("value", [])
        results = []
        for p in pages:
            text = p.get("snippet", "")
            if text.strip():
                results.append({
                    "title": p.get("name", ""),
                    "snippet": text,
                    "url": p.get("url", ""),
                })

        # jieba 分词过滤（避免完整查询字符串无法匹配结果）
        try:
            import jieba
            query_terms = [t for t in jieba.cut(query.replace('"', '')) if len(t) >= 2]
        except Exception:
            query_terms = []
        if query_terms:
            results = [
                r for r in results
                if any(t in r["title"] + r["snippet"] for t in query_terms)
            ]

        return results[:max_results]

    except Exception as e:
        import logging
        logging.getLogger("gateway").warning("crag_search_error", extra={
            "extra_fields": {"error": str(e)[:200]}
        })
        return []


def format_search_results(results: list[dict]) -> str:
    """格式化搜索结果 → 带 URL 的参考文本。"""
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results):
        title = r["title"].strip()
        snippet = r["snippet"].strip()
        url = r.get("url", "").strip()
        if title and snippet:
            line = f"[{i+1}] {title}: {snippet}"
            if url:
                line += f" (来源: {url})"
            lines.append(line)
        elif snippet:
            lines.append(f"[{i+1}] {snippet}")
    return "\n".join(lines)


def crag_retrieve(
    question: str,
    failed_categories: list[str],
) -> str | None:
    """
    联网检索 → 格式化 → 返回参考文本。
    网络不可用 / 无结果 → 返回 None（降级，不阻断）。
    """
    try:
        query = build_search_query(question, failed_categories)
        results = search_fitness(query)
        if not results:
            return None
        return format_search_results(results)
    except Exception:
        return None
