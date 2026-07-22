"""
PDF 摄入脚本 —— 肌肉力量训练彩色图谱
=====================================
从 pdf_pages/ 文件夹读取预提取的页面图片 → 并行 OCR 中文文本 → 父子 Chunk → Milvus + BM25

用法:
  python ingest_pdf.py                      # cnOCR 后端（默认，推荐中文），并行 4 进程
  python ingest_pdf.py --backend easyocr    # EasyOCR 后端（串行）
  python ingest_pdf.py --dry-run            # 仅 OCR 预览，不入库
  python ingest_pdf.py --force              # 强制重新 OCR（忽略缓存）
  python ingest_pdf.py --workers 8          # 指定并行进程数
"""
import json
import os
import sys
import time
import uuid

import jieba
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient, DataType
from rank_bm25 import BM25Okapi

from config import *
from retriever import save_bm25_to_pickle
from pdf_ocr import ocr_pages_parallel, ocr_single_page, _create_cnocr_engine

PAGES_DIR = "pdf_pages"
DRY_RUN = "--dry-run" in sys.argv
FORCE = "--force" in sys.argv
BACKEND = "cnocr"

# 解析 --workers N
WORKERS = 4
for i, arg in enumerate(sys.argv):
    if arg == "--workers" and i + 1 < len(sys.argv):
        WORKERS = int(sys.argv[i + 1])
    if arg == "--backend" and i + 1 < len(sys.argv):
        BACKEND = sys.argv[i + 1]


def _init_easyocr():
    """初始化 EasyOCR 引擎（串行回退用）。"""
    import easyocr
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    engine = easyocr.Reader(["ch_sim"], gpu=False)
    return lambda path: "\n".join(
        text for _, text, _ in sorted(
            engine.readtext(path),
            key=lambda b: (round(b[0][0][1], -1), b[0][0][0])
        )
    )


def build():
    page_files = sorted(
        [f for f in os.listdir(PAGES_DIR) if f.endswith(".png")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    if not page_files:
        raise SystemExit(f"未找到页面图片！请先将 PDF 提取到 {PAGES_DIR}/")

    print(f"[0/6] 找到 {len(page_files)} 页图片")

    # ================================================================
    # 1. OCR 全部页面（并行 + 缓存）
    # ================================================================
    t0 = time.time()

    if BACKEND == "cnocr":
        page_results = ocr_pages_parallel(
            PAGES_DIR, max_workers=WORKERS, use_cache=True, force=FORCE
        )
        # 转换为 {filename: text} 和 page_texts 列表
        ocr_map = {fname: text for fname, text, _ in page_results}
        page_texts = [text for _, text, _ in page_results if text.strip()]
    else:
        # EasyOCR 回退（串行）
        ocr_fn = _init_easyocr()
        page_texts = []
        for i, fname in enumerate(page_files):
            path = os.path.join(PAGES_DIR, fname)
            text = ocr_fn(path)
            if text.strip():
                page_texts.append(text)
            if (i + 1) % 10 == 0:
                print(f"  EasyOCR... {i+1}/{len(page_files)}")
        ocr_map = {}

    total_chars = sum(len(t) for t in page_texts)
    print(f"[1/6] OCR 完成：{len(page_texts)}/{len(page_files)} 页有文本"
          f"（{total_chars} 字符，{time.time()-t0:.0f}s）")

    if DRY_RUN:
        print("\n[Dry-run] 预览前 3 页 OCR 结果:")
        for i, text in enumerate(page_texts[:3]):
            print(f"\n--- Page {i+1} ---")
            print(text[:300])
        return

    # 转为 LangChain Documents
    raw_docs = []
    for fname in page_files:
        text = ocr_map.get(fname, "") if BACKEND == "cnocr" else ""
        if BACKEND != "cnocr":
            # EasyOCR path: page_texts 已过滤，需要重新对齐
            pass
        if text.strip():
            page_num = int(fname.split("_")[1].split(".")[0])
            raw_docs.append(Document(
                page_content=text,
                metadata={"source": "肌肉力量训练彩色图谱.pdf", "page": page_num}
            ))

    # EasyOCR 兼容
    if BACKEND != "cnocr" and not raw_docs:
        for i, text in enumerate(page_texts):
            raw_docs.append(Document(
                page_content=text,
                metadata={"source": "肌肉力量训练彩色图谱.pdf", "page": i + 1}
            ))

    # ================================================================
    # 2. 父子切片
    # ================================================================
    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_PARENT_TOKENS,
        chunk_overlap=int(CHUNK_PARENT_TOKENS * CHUNK_OVERLAP_RATIO),
        separators=["\n", "。", "！", "？", "；", "，", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_CHILD_TOKENS,
        chunk_overlap=int(CHUNK_CHILD_TOKENS * CHUNK_OVERLAP_RATIO),
        separators=["\n", "。", "！", "？", "；", "，", " ", ""],
    )

    parent_docs = parent_splitter.split_documents(raw_docs)
    parent_store = {}
    child_docs = []
    for parent in parent_docs:
        pid = str(uuid.uuid4())
        parent_store[pid] = {"page_content": parent.page_content, "metadata": parent.metadata}
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = pid
        child_docs.extend(children)

    print(f"[2/6] 父块 {len(parent_docs)} | 子块 {len(child_docs)}")

    # ================================================================
    # 3. BM25 索引
    # ================================================================
    bm25_corpus = [list(jieba.cut(doc.page_content)) for doc in child_docs]
    bm25_idx = BM25Okapi(bm25_corpus)
    save_bm25_to_pickle(bm25_idx, child_docs, BM25_INDEX_PATH)
    print(f"[3/6] BM25 索引已保存 ({len(child_docs)} 条)")

    # ================================================================
    # 4. Milvus Lite 写入（MilvusClient API）
    # ================================================================
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    client = MilvusClient(uri=MILVUS_URI)

    # 如果 collection 存在则先删除（全量重建）
    if client.has_collection(MILVUS_COLLECTION):
        client.drop_collection(MILVUS_COLLECTION)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_DIM)
    schema.add_field(field_name="page_content", datatype=DataType.VARCHAR, max_length=4096)
    schema.add_field(field_name="metadata_json", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 16},
    )

    client.create_collection(
        collection_name=MILVUS_COLLECTION,
        schema=schema,
        index_params=index_params,
    )

    batch_size = 50
    for i in range(0, len(child_docs), batch_size):
        batch = child_docs[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        vecs = embeddings.embed_documents(texts)
        data = [
            {
                "id": str(uuid.uuid4()),
                "embedding": vec,
                "page_content": doc.page_content,
                "metadata_json": json.dumps(doc.metadata, ensure_ascii=False),
                "parent_id": doc.metadata.get("parent_id", ""),
            }
            for doc, vec in zip(batch, vecs)
        ]
        client.insert(collection_name=MILVUS_COLLECTION, data=data)
        if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(child_docs):
            print(f"  Milvus... {min(i + batch_size, len(child_docs))}/{len(child_docs)}")

    print(f"[4/6] Milvus 写入完成 ({len(child_docs)} 条)")

    # 父块 JSON
    parent_dir = "milvus_data"
    os.makedirs(parent_dir, exist_ok=True)
    parent_path = os.path.join(parent_dir, "parents.json")
    with open(parent_path, "w", encoding="utf-8") as f:
        json.dump(parent_store, f, ensure_ascii=False)
    print(f"[5/6] 父块 JSON 已保存 ({len(parent_store)} 条)")

    client.close()
    print("[6/6] PDF 摄入完毕！")


if __name__ == "__main__":
    build()
