"""
text_quality.py —— OCR 乱码质检（摄入层）
=========================================
放在 build_index / ingest_pdf 的摄入层（不进 pdf_ocr.py：OCR 缓存键含图片 hash，
混入质检会污染缓存语义——质检属于摄入策略，不属于 OCR 引擎本身）。

降级流程（摄入脚本内）：
  OCR 输出 → filter_ocr_pages → 乱码页提分辨率重试一次 → 仍乱码则丢弃并打印统计
  （丢弃的页码数字可见，面试有料）

用法:
    from text_quality import filter_ocr_pages, is_garbled
    ok_pages, garbled_pages = filter_ocr_pages(results)  # [(fname, text, page_num), ...]
"""

import re
from collections import Counter

# 合法字符：汉字 + ASCII 字母数字 + 常用中英文标点
_VALID_CHAR = re.compile(
    r"[一-鿿A-Za-z0-9，。、；：？！“”‘’（）【】《》…—·,.;:?!()\[\]<>%+\-*/=@#&_\s]"
)


def char_valid_ratio(text: str) -> float:
    """合法字符占比（乱码常表现为大量不可识别符号/生僻字）。"""
    if not text:
        return 0.0
    return len(_VALID_CHAR.findall(text)) / len(text)


def dict_coverage(text: str, sample_limit: int = 200) -> float:
    """jieba 词典命中率（零外部词表依赖）。康养术语 OOV 多，阈值需放宽松。"""
    try:
        import jieba
    except ImportError:
        return 1.0
    tokens = [t for t in jieba.lcut(text[:sample_limit]) if t.strip()]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if jieba.dt.FREQ.get(t, 0) > 0 or len(t) == 1)
    return hits / len(tokens)


def repetition_ratio(text: str) -> float:
    """最高频单字符占比（乱码常表现为同字符大量重复，如"嗒嗒嗒嗒"、同一行重复）。"""
    if not text:
        return 0.0
    return Counter(text).most_common(1)[0][1] / len(text)


def is_garbled(text: str) -> bool:
    """硬规则判定乱码。阈值全部在 config（OCR_QUALITY_*）。"""
    import config as cfg

    text = (text or "").strip()
    if len(text) < getattr(cfg, "OCR_QUALITY_MIN_CHARS", 10):
        return True  # 空页/碎片页
    if char_valid_ratio(text) < getattr(cfg, "OCR_QUALITY_CHAR_RATIO", 0.6):
        return True
    if dict_coverage(text) < getattr(cfg, "OCR_QUALITY_DICT_COVERAGE", 0.35):
        return True
    if repetition_ratio(text) > getattr(cfg, "OCR_QUALITY_REPETITION", 0.3):
        return True
    return False


def filter_ocr_pages(results: list) -> tuple[list, list]:
    """results: [(filename, text, page_num), ...] → (ok_pages, garbled_pages)。"""
    import config as cfg

    if not getattr(cfg, "OCR_QUALITY_ENABLED", True):
        return results, []
    ok, bad = [], []
    for item in results:
        (bad if is_garbled(item[1]) else ok).append(item)
    return ok, bad
