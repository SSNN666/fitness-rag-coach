"""doc_loaders 单测：注册表 / 各格式 loader / 表格误判过滤 / 跨页表格合并。"""
import csv

import pytest

from doc_loaders import (
    TABLE_BLOCK_SEP,
    LoadedDoc,
    UnsupportedFormat,
    _is_real_table,
    _merge_cross_page_tables,
    load_document,
    supported_extensions,
)


# ================================================================
# 注册表
# ================================================================

class TestRegistry:
    def test_expected_extensions_registered(self):
        exts = supported_extensions()
        for e in (".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx", ".png", ".jpg"):
            assert e in exts, e

    def test_unsupported_format_raises(self, tmp_path):
        """未注册的扩展名必须显式报错，不能静默返回空。"""
        p = tmp_path / "x.unknownext"
        p.write_text("hello", encoding="utf-8")
        with pytest.raises(UnsupportedFormat):
            load_document(str(p))


# ================================================================
# 文本 / CSV
# ================================================================

class TestTextLoader:
    def test_txt_and_md(self, tmp_path):
        for name in ("a.txt", "b.md"):
            p = tmp_path / name
            p.write_text("第一段。\n\n第二段。", encoding="utf-8")
            docs = load_document(str(p))
            assert len(docs) == 1
            assert "第一段" in docs[0].markdown
            assert docs[0].metadata["source"] == name

    def test_empty_file_returns_nothing(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("   \n  ", encoding="utf-8")
        assert load_document(str(p)) == []


class TestCsvLoader:
    def _write(self, tmp_path, rows, header):
        p = tmp_path / "t.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        return str(p)

    def test_one_doc_per_row_with_columns_in_body(self, tmp_path):
        """列名必须进正文——这是图谱与实体标签的数据来源。"""
        path = self._write(tmp_path, [["深蹲", "股四头肌"]], ["动作名称", "目标肌群"])
        docs = load_document(path)
        assert len(docs) == 1
        assert "动作名称: 深蹲" in docs[0].markdown
        assert "目标肌群: 股四头肌" in docs[0].markdown
        assert docs[0].metadata["动作名称"] == "深蹲"

    def test_long_columns_excluded_from_metadata(self, tmp_path):
        """metadata 只收短字段：Milvus 的 metadata_json 上限 1024 字节。"""
        long_text = "这是一段很长的说明文字。" * 20
        path = self._write(
            tmp_path,
            [["深蹲", long_text]],
            ["动作名称", "注意事项"],
        )
        doc = load_document(path)[0]
        assert "注意事项" not in doc.metadata          # 长列不进 metadata
        assert "注意事项" in doc.markdown              # 但仍在正文里（可被检索）
        assert "动作名称" in doc.metadata

    def test_row_index_recorded(self, tmp_path):
        path = self._write(tmp_path, [["a"], ["b"], ["c"]], ["动作名称"])
        rows = [d.metadata["row"] for d in load_document(path)]
        assert rows == [0, 1, 2]


# ================================================================
# 表格误判过滤（实测教训：pdfplumber 会把竖排文本识别成单列表格）
# ================================================================

class TestRealTableGuard:
    @pytest.mark.parametrize("rows,expected", [
        ([["单列文本"], ["第二行"], ["第三行"]], False),          # 单列 → 误判
        ([["动作", "组数"], ["深蹲", "4"]], True),                 # 正常两列
        ([["", "", ""], ["", "x", ""]], False),                   # 稀疏 → 噪声
        ([["只有一行", "两个字段"]], False),                       # 行数不足
        ([["a", "b"], ["", ""], ["", ""]], False),                # 非空占比过低
        ([], False),                                              # 空
    ])
    def test_guard(self, rows, expected):
        assert _is_real_table(rows) is expected


# ================================================================
# 跨页表格合并
# ================================================================

class TestCrossPageMerge:
    @staticmethod
    def _t(rows, bbox, page):
        return {"rows": rows, "bbox": bbox, "_page": page}

    def test_merges_continuation_and_dedupes_header(self):
        """上页末表 + 本页首表，列数相同 → 合并，并去掉重复表头。"""
        t1 = self._t([["动作", "组数"], ["深蹲", "4"]], (0, 100, 500, 790), 0)
        t2 = self._t([["动作", "组数"], ["硬拉", "3"]], (0, 60, 500, 300), 1)
        out = _merge_cross_page_tables([[t1], [t2]], [800.0, 800.0])
        assert len(out) == 1
        assert out[0]["_merged_pages"] == 2
        assert out[0]["rows"] == [["动作", "组数"], ["深蹲", "4"], ["硬拉", "3"]]

    def test_does_not_merge_mid_page_tables(self):
        """两张表都不在页边界 → 不是被分页截断，不合并。"""
        t1 = self._t([["a", "b"], ["1", "2"]], (0, 300, 500, 500), 0)
        t2 = self._t([["a", "b"], ["3", "4"]], (0, 300, 500, 500), 1)
        assert len(_merge_cross_page_tables([[t1], [t2]], [800.0, 800.0])) == 2

    def test_does_not_merge_different_column_count(self):
        t1 = self._t([["a", "b"], ["1", "2"]], (0, 100, 500, 790), 0)
        t2 = self._t([["a", "b", "c"], ["3", "4", "5"]], (0, 60, 500, 300), 1)
        assert len(_merge_cross_page_tables([[t1], [t2]], [800.0, 800.0])) == 2

    def test_conservative_without_bbox(self):
        """没有坐标信息就不冒险合并（宁可拆开也不要拼错表）。"""
        t1 = {"rows": [["a", "b"], ["1", "2"]], "_page": 0}
        t2 = {"rows": [["a", "b"], ["3", "4"]], "_page": 1}
        assert len(_merge_cross_page_tables([[t1], [t2]], [800.0, 800.0])) == 2

    def test_non_adjacent_pages_not_merged(self):
        t1 = self._t([["a", "b"], ["1", "2"]], (0, 100, 500, 790), 0)
        t2 = self._t([["a", "b"], ["3", "4"]], (0, 60, 500, 300), 2)
        assert len(_merge_cross_page_tables(
            [[t1], [], [t2]], [800.0, 800.0, 800.0])) == 2


# ================================================================
# DOCX / XLSX
# ================================================================

class TestDocxLoader:
    def test_paragraphs_and_headings(self, tmp_path):
        docx = pytest.importorskip("docx")
        p = tmp_path / "d.docx"
        d = docx.Document()
        d.add_heading("训练指南", level=1)
        d.add_paragraph("深蹲主要锻炼股四头肌。")
        d.save(str(p))

        docs = load_document(str(p))
        assert len(docs) == 1
        assert "# 训练指南" in docs[0].markdown      # 标题保留层级
        assert "股四头肌" in docs[0].markdown

    def test_table_becomes_markdown_block(self, tmp_path):
        docx = pytest.importorskip("docx")
        p = tmp_path / "t.docx"
        d = docx.Document()
        t = d.add_table(rows=2, cols=2)
        t.cell(0, 0).text = "动作"
        t.cell(0, 1).text = "组数"
        t.cell(1, 0).text = "深蹲"
        t.cell(1, 1).text = "4"
        d.save(str(p))

        md = load_document(str(p))[0].markdown
        assert TABLE_BLOCK_SEP in md                  # 表格被标记为原子块
        assert "| 动作 | 组数 |" in md


class TestXlsxLoader:
    def test_each_sheet_becomes_one_doc(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        p = tmp_path / "s.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "动作表"
        ws.append(["动作", "组数"])
        ws.append(["深蹲", 4])
        ws.append(["硬拉", 3])
        wb.create_sheet("空表")           # 少于 2 行 → 跳过
        wb.save(str(p))

        docs = load_document(str(p))
        assert len(docs) == 1
        assert docs[0].metadata["sheet"] == "动作表"
        assert "| 动作 | 组数 |" in docs[0].markdown
        assert "| 深蹲 | 4 |" in docs[0].markdown
