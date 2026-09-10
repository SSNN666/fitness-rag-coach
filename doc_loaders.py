"""
doc_loaders.py —— 统一文档归一化层
====================================
多格式文档 → **统一 Markdown 中间表示**。

为什么需要
----------
摄入层原本是硬编码分支（CSV / txt / OCR 图片三分支写在 build_index 里），
加一种格式就要改索引构建代码。**格式差异不该泄漏到切块、嵌入、建图里**——
那些环节只应关心「统一的文本 + 元数据」。

设计
----
- **注册表**：扩展名 → loader 函数。加格式只需 `@register(".xxx")`，不动 build_index。
- **统一产出** `LoadedDoc(markdown, metadata)`：
  - markdown 是给切块/嵌入用的正文
  - metadata 保留结构化字段（如 CSV 的列），供图谱构建等下游使用
- **表格是原子块**：Markdown 表格用 `TABLE_BLOCK_SEP` 标记边界，
  切块时整块保留，不会被 `\n` 递归切分器打散（见 build_index 的 `_split_preserving_tables`）。

支持格式
--------
| 扩展名 | 解析 | 说明 |
|---|---|---|
| `.txt` / `.md` | 直读 | 已归一，原样返回 |
| `.csv` | csv 模块 | 每行一条；列名进 metadata（结构化字段供图谱用） |
| `.pdf` | pdfplumber | 文本 + 表格抽取 + **跨页表格合并** |
| `.docx` | python-docx | 段落层级 + 表格 |
| `.xlsx` | openpyxl | 每个 sheet → 一个 Markdown 表格 |
| 图片 | cnocr（可选） | 未安装时明确报错，不静默跳过 |

用法
----
    from doc_loaders import load_document, SUPPORTED_EXTENSIONS
    docs = load_document("kb_zhinan.pdf")     # -> list[LoadedDoc]
"""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# 表格块边界标记：切块器据此把整张表当一个单元，不按 \n 打散
TABLE_BLOCK_SEP = "\n<!--TABLE-->\n"

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


class UnsupportedFormat(ValueError):
    """没有注册对应扩展名的 loader。"""


@dataclass
class LoadedDoc:
    """归一化后的文档单元。"""
    markdown: str
    metadata: dict = field(default_factory=dict)


# ============================================================
# 注册表
# ============================================================

_LOADERS: dict[str, Callable[[str], list[LoadedDoc]]] = {}


def register(*exts: str):
    """把函数注册为若干扩展名的 loader。"""
    def deco(fn):
        for e in exts:
            _LOADERS[e.lower()] = fn
        return fn
    return deco


def supported_extensions() -> set[str]:
    return set(_LOADERS) | _IMAGE_EXTS


def load_document(path: str) -> list[LoadedDoc]:
    """按扩展名分发到对应 loader。未注册 → UnsupportedFormat（不静默跳过）。"""
    ext = Path(path).suffix.lower()
    if ext in _IMAGE_EXTS:
        return _load_image(path)
    fn = _LOADERS.get(ext)
    if fn is None:
        raise UnsupportedFormat(
            f"不支持的格式 {ext}（已支持：{', '.join(sorted(supported_extensions()))}）")
    return fn(path)


# ============================================================
# 纯文本 / Markdown
# ============================================================

@register(".txt", ".md")
def _load_text(path: str) -> list[LoadedDoc]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    return [LoadedDoc(markdown=text, metadata={"source": os.path.basename(path)})]


# ============================================================
# CSV
# ============================================================

def _md_table(rows: list[list[str]], header: list[str] | None = None) -> str:
    """二维数据 → Markdown 表格（保留列结构，模型与人都更易读）。"""
    if not rows:
        return ""
    out: list[str] = []
    if header:
        out.append("| " + " | ".join(str(h) for h in header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c).replace("|", "\\|")
                                     for c in r) + " |")
    return "\n".join(out)


# metadata 只收「短字段」。原因：Milvus 的 metadata_json 是 VARCHAR(1024)，
# 长文本列（如「步骤」「注意事项」）会把上限撑爆。长列保留在**正文**里
# （照样能进 BM25 / 向量索引被检索到），只是不进结构化 metadata。
_METADATA_MAX_FIELD_CHARS = 64


@register(".csv")
def _load_csv(path: str) -> list[LoadedDoc]:
    """每行一条 LoadedDoc：正文为「列名: 值」，短列同时进 metadata。

    ⚠️ 列名**必须显式**放进正文：langchain_community 的 CSVLoader 默认
    metadata_columns=() 且正文处理有坑，曾导致下游读 metadata 全部落空、
    图谱缺三种关系（见 CHANGELOG）。这里由本模块自己控制，行为确定。
    """
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    # 哪些列适合进 metadata：该列所有值都短
    short_cols = [
        c for c in header
        if all(len(str(r.get(c, "") or "")) <= _METADATA_MAX_FIELD_CHARS for r in rows)
    ]

    source = os.path.basename(path)
    out: list[LoadedDoc] = []
    for i, row in enumerate(rows):
        body = "\n".join(f"{k}: {row.get(k, '')}" for k in header)
        md = {c: row.get(c, "") for c in short_cols}    # 结构化字段供图谱/实体标签用
        md["source"] = source
        md["row"] = i
        out.append(LoadedDoc(markdown=body, metadata=md))
    return out


# ============================================================
# PDF（含跨页表格合并）
# ============================================================

# 跨页表格合并的判定阈值（页高比例）：
# 表尾距页底 < 15% 且续表距页顶 < 15% 时，认为是同一张表被分页截断
_BOTTOM_MARGIN_RATIO = 0.15
_TOP_MARGIN_RATIO = 0.15


def _rows_signature(rows: list[list[str]]) -> tuple:
    """用列数 + 首行内容刻画表结构，用于判断两张表是否同构。"""
    if not rows:
        return ()
    return (len(rows[0]), tuple(str(c).strip() for c in rows[0]))


def _is_real_table(rows: list[list[str]]) -> bool:
    """结构校验：过滤 pdfplumber 的表格误判。

    ⚠️ 实测教训：`extract_tables()` 会把**竖排文本段落**识别成「单列表格」。
    若不加校验直接采用，原本正常的正文会被重排成无意义的表格，**反而破坏
    已有的文本抽取质量**。判定标准（保守）：
      1. ≥2 行且 **≥2 列**（单列几乎必然是误判）
      2. 非空单元格占比 ≥ 30%（稀疏的框线噪声不算表）
      3. **≥2 行**含 ≥2 个非空单元格——排除「只有表头、数据行全空」的退化情况
    """
    if not rows or len(rows) < 2:
        return False
    n_cols = max(len(r) for r in rows)
    if n_cols < 2:
        return False
    cells = [c for r in rows for c in r]
    if not cells:
        return False
    non_empty = sum(1 for c in cells if c and c.strip())
    if non_empty / len(cells) < 0.30:
        return False
    substantial_rows = sum(
        1 for r in rows if sum(1 for c in r if c and c.strip()) >= 2)
    return substantial_rows >= 2


def _extract_pdf_tables(page) -> list[dict]:
    """抽取一页里的表格（含 bbox），坏表/误判跳过不阻断整页。"""
    out = []
    try:
        tables = page.extract_tables() or []
    except Exception:
        return out
    for t in tables:
        if not t:
            continue
        rows = [[("" if c is None else str(c).strip()) for c in r] for r in t]
        if not _is_real_table(rows):
            continue                     # 单列/稀疏 → 文本被误判成表格
        out.append({"rows": rows})
    try:
        found = page.find_tables() or []
    except Exception:
        found = []
    for i, ft in enumerate(found):
        if i < len(out):
            out[i]["bbox"] = tuple(ft.bbox)   # (x0, top, x1, bottom)
    return out


def _merge_cross_page_tables(pages_tables: list[list[dict]], heights: list[float]) -> list[dict]:
    """合并被分页截断的表格。

    判定（保守，避免误合并不同表）：
      1. 相邻两页（第 N 页的末表 + 第 N+1 页的首表）
      2. 第 N 页末表贴近页底，且第 N+1 页首表贴近页顶
      3. 两表列数相同
    合并时若续表首行与主表表头相同（PDF 常见重复表头），丢弃续表首行。
    """
    merged: list[dict] = []
    for p_idx, tables in enumerate(pages_tables):
        for t_idx, t in enumerate(tables):
            # ⚠️ 跨页合并的两端是：**上页的末表** + **本页的首表**。
            # 曾经错写成判断「当前表是末表」，导致首表永远合并不上（单测覆盖）。
            is_first_of_page = t_idx == 0
            is_last_of_page = t_idx == len(tables) - 1
            prev = merged[-1] if merged else None

            can_merge = bool(
                is_first_of_page
                and prev is not None
                and prev.get("_is_last_on_page")            # 上页末表
                and prev.get("_page") == p_idx - 1          # 相邻页
                and len(prev["rows"]) and len(t["rows"])
                and len(prev["rows"][0]) == len(t["rows"][0])   # 列数相同
            )
            if can_merge:
                page_h = heights[prev["_page"]] if prev["_page"] < len(heights) else 0
                bbox, nxt_bbox = prev.get("bbox"), t.get("bbox")
                if bbox and nxt_bbox and page_h:
                    can_merge = (bbox[3] / page_h >= 1 - _BOTTOM_MARGIN_RATIO
                                 and nxt_bbox[1] / page_h <= _TOP_MARGIN_RATIO)
                else:
                    can_merge = False          # 无坐标信息时不冒险合并

            if can_merge:
                rows = t["rows"]
                # 续表重复了表头 → 去掉（PDF 分页常见）
                if rows and _rows_signature(rows) == _rows_signature(prev["rows"][:1]):
                    rows = rows[1:]
                prev["rows"].extend(rows)
                prev["_merged_pages"] = prev.get("_merged_pages", 1) + 1
                prev["_is_last_on_page"] = is_last_of_page
                continue

            nt = dict(t)
            nt["_page"] = p_idx
            nt["_is_last_on_page"] = is_last_of_page
            merged.append(nt)
    return merged


@register(".pdf")
def _load_pdf(path: str) -> list[LoadedDoc]:
    """PDF → 逐页 Markdown；表格抽为 Markdown 表格并合并跨页表。

    注意：本 loader 走 **文本层直抽**（pdfplumber），适用于文本型 PDF。
    扫描件（无文本层）应走图片 OCR 路径（见 build_index 的 pdf_pages 分支）。
    """
    import pdfplumber

    out: list[LoadedDoc] = []
    name = os.path.basename(path)
    with pdfplumber.open(path) as pdf:
        heights = [float(p.height or 0) for p in pdf.pages]
        pages_tables = [_extract_pdf_tables(p) for p in pdf.pages]
        merged_tables = _merge_cross_page_tables(pages_tables, heights)

        # 页 → 该页的表格（按 _page 归位）
        by_page: dict[int, list[dict]] = {}
        for t in merged_tables:
            by_page.setdefault(t["_page"], []).append(t)

        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            tables = by_page.get(i, [])
            if not text and not tables:
                continue
            parts: list[str] = []
            if text:
                parts.append(text)
            for t in tables:
                md = _md_table(t["rows"])
                if md:
                    span = t.get("_merged_pages", 1)
                    note = f"（跨 {span} 页合并）" if span > 1 else ""
                    parts.append(f"[表格{note}]{TABLE_BLOCK_SEP}{md}")
            out.append(LoadedDoc(
                markdown="\n\n".join(parts),
                metadata={"source": name, "page": i + 1},
            ))
    return out


# ============================================================
# DOCX
# ============================================================

@register(".docx")
def _load_docx(path: str) -> list[LoadedDoc]:
    """段落 + 表格 → Markdown（标题按 style 转 #，表格转 Markdown 表格）。"""
    import docx  # python-docx

    d = docx.Document(path)
    name = os.path.basename(path)
    blocks: list[str] = []

    for p in d.paragraphs:
        text = (p.text or "").strip()
        if not text:
            continue
        style = (getattr(p.style, "name", "") or "").lower()
        if "heading" in style or style.startswith("标题"):
            level = 1
            m = re.search(r"(\d+)", style)
            if m:
                level = min(int(m.group(1)), 6)
            blocks.append("#" * level + " " + text)
        else:
            blocks.append(text)

    for t in d.tables:
        rows = [[("" if c.text is None else c.text.strip()) for c in r.cells] for r in t.rows]
        md = _md_table(rows)
        if md:
            blocks.append(f"[表格]{TABLE_BLOCK_SEP}{md}")

    if not blocks:
        return []
    return [LoadedDoc(markdown="\n\n".join(blocks), metadata={"source": name})]


# ============================================================
# XLSX
# ============================================================

@register(".xlsx", ".xls")
def _load_xlsx(path: str) -> list[LoadedDoc]:
    """每个 worksheet → 一个 Markdown 表格（一个 sheet 一个文档单元）。

    首行视为表头。空 sheet 跳过。
    """
    from openpyxl import load_workbook

    name = os.path.basename(path)
    wb = load_workbook(path, read_only=True, data_only=True)
    out: list[LoadedDoc] = []
    try:
        for ws in wb.worksheets:
            rows = []
            for row in ws.iter_rows(values_only=True):
                if row is None or all(c is None or str(c).strip() == "" for c in row):
                    continue
                rows.append(["" if c is None else str(c) for c in row])
            if len(rows) < 2:
                continue
            header, body = rows[0], rows[1:]
            md = _md_table(body, header=header)
            if md:
                out.append(LoadedDoc(
                    markdown=f"## {ws.title}\n\n[表格]{TABLE_BLOCK_SEP}{md}",
                    metadata={"source": name, "sheet": ws.title},
                ))
    finally:
        wb.close()
    return out


# ============================================================
# 图片（OCR，可选依赖）
# ============================================================

def _load_image(path: str) -> list[LoadedDoc]:
    """图片 → OCR 文本。cnocr 未安装时**明确报错**，不静默返回空。"""
    try:
        from pdf_ocr import ocr_single_page
    except Exception as e:
        raise UnsupportedFormat(
            f"图片 OCR 不可用（{e}）。扫描件路径需要 cnocr，"
            f"或先用 pdf_ocr.py 的缓存产出文本再入库。") from e
    text = (ocr_single_page(path) or "").strip()
    if not text:
        return []
    return [LoadedDoc(markdown=text, metadata={"source": os.path.basename(path)})]
