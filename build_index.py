"""
构建三路检索索引 —— Milvus + BM25 + Neo4j
===========================================
数据源: fitness_data.csv + pdf_pages/（如存在）
用法: python build_index.py              # 全量构建
      python build_index.py --fast        # 快速模式（降采样 + 小批次，防蓝屏）
      python build_index.py --skip-neo4j  # 跳过图谱
      python build_index.py --workers 2   # 指定 OCR 进程数
      python build_index.py --resize 1200 # 图片降采样尺寸（0=禁用）
"""
import gc
import json
import os
import re
import sys
import time
import uuid

import jieba
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient, DataType
from rank_bm25 import BM25Okapi

from config import *
from doc_loaders import TABLE_BLOCK_SEP, load_document
from ingest_cache import EmbeddingCache, IngestManifest, embed_with_cache
from llm_adapter import build_embeddings
from retriever import save_bm25_to_pickle

# 防 Windows OOM 蓝屏：限制 Ollama 并发
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "1")
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "2")

SKIP_NEO4J = "--skip-neo4j" in sys.argv
FAST_MODE = "--fast" in sys.argv
FORCE_REBUILD = "--force" in sys.argv          # 忽略指纹清单与嵌入缓存，强制全量重建

# 解析 --workers N（默认 2，--fast 模式默认 2）
OCR_WORKERS = 2
for i, arg in enumerate(sys.argv):
    if arg == "--workers" and i + 1 < len(sys.argv):
        OCR_WORKERS = int(sys.argv[i + 1])

# 解析 --resize N（默认 1600，--fast 模式默认 1200）
MAX_IMAGE_SIZE = 1200 if FAST_MODE else 1600
for i, arg in enumerate(sys.argv):
    if arg == "--resize" and i + 1 < len(sys.argv):
        MAX_IMAGE_SIZE = int(sys.argv[i + 1])

# 解析 --source-dir（PDF 页面图片目录，默认 pdf_pages）
PDF_PAGES_DIR = "pdf_pages"
for i, arg in enumerate(sys.argv):
    if arg == "--source-dir" and i + 1 < len(sys.argv):
        PDF_PAGES_DIR = sys.argv[i + 1]

# embedding 批次大小（--fast 模式用小批次）
EMBED_BATCH = 20 if FAST_MODE else 50


def _extract_entity_labels(metadata: dict) -> list[str]:
    """从 CSV 元数据抽取实体标签，用于检索时按实体类型过滤/加权。"""
    labels = []
    if "动作名称" in metadata and metadata["动作名称"].strip():
        labels.append(f"exercise:{metadata['动作名称'].strip()}")
    if "目标肌群" in metadata:
        for m in re.split(r"[/、，]", metadata["目标肌群"]):
            m = m.strip()
            if len(m) >= 2:
                labels.append(f"muscle:{m}")
    if "器械" in metadata and metadata["器械"].strip():
        labels.append(f"equipment:{metadata['器械'].strip()}")
    if "难度" in metadata and metadata["难度"].strip():
        labels.append(f"difficulty:{metadata['难度'].strip()}")
    if "source" in metadata:
        labels.append(f"source:{metadata['source']}")
    return labels


def _semantic_pre_chunk(raw_docs: list[Document]) -> list[Document]:
    """
    语义预分块：在 token 切分前，先按文档边界分组。
    - **表格**：整张表独立成块，不与正文混排（见下）
    - CSV 行（source='fitness_data.csv'）：每行 = 一个独立动作，保持完整
    - PDF 页：按双换行（段落边界）切分
    这样 token splitter 只会在单条目超出预算时才细分。

    ⚠️ 表格为什么必须独立成块：
    切分器的分隔符含 "\\n"，而 Markdown 表格**按行分隔**——混在正文里会被
    逐行打散成碎片，表格结构彻底丢失（LLM 拿到的是「| 动作 |」这种残行）。
    独立成块后：常规表格整张落在同一个父块里；即便某张超大表仍超预算，
    切分也会发生在**行边界**上（行本身完整），而不是切在行中间。
    """
    merged: list[Document] = []
    for doc in raw_docs:
        source = doc.metadata.get("source", "")

        # 含表格的文档 → 正文与每张表各自成块
        if TABLE_BLOCK_SEP in (doc.page_content or ""):
            for seg in doc.page_content.split(TABLE_BLOCK_SEP):
                seg = seg.strip()
                if seg:
                    md = doc.metadata.copy()
                    md["is_table"] = seg.startswith("[表格]")
                    merged.append(Document(page_content=seg, metadata=md))
            continue

        if source == CSV_FILE or doc.metadata.get("row") is not None:
            # CSV 每行已经是独立条目，保持原样
            merged.append(doc)
        else:
            # 按段落边界切分
            for para in re.split(r"\n\s*\n", doc.page_content):
                para = para.strip()
                if para:
                    merged.append(Document(
                        page_content=para,
                        metadata=doc.metadata.copy()
                    ))
    return merged


def build():
    global child_docs  # for GC

    mode_str = "FAST" if FAST_MODE else "NORMAL"
    print(f"[MODE] {mode_str} | OCR workers={OCR_WORKERS} | resize={MAX_IMAGE_SIZE}px | embed_batch={EMBED_BATCH}")
    print(f"[TIP]  加 --fast 使用快速防蓝屏模式，加 --skip-neo4j 跳过图谱，加 --force 忽略指纹强制重建\n")

    # ================================================================
    # 0. 源文件指纹比对（SHA256）—— 同一份文件重复摄入不必重跑
    # ================================================================
    source_files = [CSV_FILE] if os.path.isfile(CSV_FILE) else []
    if TEXT_KB_SOURCES:
        source_files += [e["file"] for e in TEXT_KB_SOURCES if os.path.isfile(e["file"])]
    elif os.path.isdir(PDF_PAGES_DIR):
        source_files += [os.path.join(PDF_PAGES_DIR, f)
                         for f in sorted(os.listdir(PDF_PAGES_DIR))]

    manifest = IngestManifest()
    diff = manifest.diff(source_files)
    print(f"[0/7] 源文件指纹比对：{diff.describe()}")
    for label, items in (("新增", diff.new), ("变更", diff.changed), ("移除", diff.removed)):
        for p in items[:5]:
            print(f"      {label}: {p}")

    if not diff.needs_rebuild and not FORCE_REBUILD:
        print("      → 所有源文件内容未变（SHA256 一致），跳过重建。")
        print("        如确需重建请加 --force")
        return
    if FORCE_REBUILD:
        print("      --force：忽略指纹与嵌入缓存，强制全量重建")

    # ================================================================
    # 1. 加载数据源
    # ================================================================
    # 1a. CSV 动作库 —— 统一走 doc_loaders 注册表（格式差异不再泄漏到这里）
    # 注：原用 langchain_community CSVLoader，其默认行为（metadata_columns=()）
    # 曾导致下游读 metadata 全部落空、图谱缺三种关系（见 CHANGELOG）。
    # 现由 doc_loaders 显式控制：列名进正文，短列进 metadata，行为可确定。
    raw_docs: list[Document] = []
    if os.path.isfile(CSV_FILE):
        raw_docs.extend(
            Document(page_content=d.markdown, metadata=d.metadata)
            for d in load_document(CSV_FILE)
        )
    csv_count = len(raw_docs)
    print(f"  CSV 动作库 {csv_count} 条")

    # 1b. 知识库文档摄入：
    #   优先 TEXT_KB_SOURCES 文本直抽（合规公开资料，零 OCR 错误）；
    #   未配置时回退 pdf_pages/ 图片 OCR（占位扫描件路径）
    if TEXT_KB_SOURCES:
        for entry in TEXT_KB_SOURCES:
            path = entry["file"]
            if not os.path.isfile(path):
                print(f"  [WARN] 文本源缺失: {path}，跳过")
                continue
            docs = load_document(path)   # 按扩展名分发：txt/md/pdf/docx/xlsx
            chars = 0
            for d in docs:
                md = dict(d.metadata)
                md["source"] = entry["name"]   # 用配置的引用显示名覆盖文件名
                raw_docs.append(Document(page_content=d.markdown, metadata=md))
                chars += len(d.markdown)
            print(f"  文档摄入 {entry['name']}（{len(docs)} 个单元 / {chars} 字符）")
    elif os.path.isdir(PDF_PAGES_DIR):
        pdf_pages_dir = PDF_PAGES_DIR
        from pdf_ocr import ocr_pages_parallel, ocr_single_page
        page_results = ocr_pages_parallel(
            pdf_pages_dir,
            max_workers=OCR_WORKERS,
            use_cache=True,
            max_image_size=MAX_IMAGE_SIZE,
        )
        if page_results:
            # [OCR质检] 乱码过滤：不合格页提分辨率重试一次，仍不合格丢弃并打印页码
            from text_quality import filter_ocr_pages, is_garbled
            ok_pages, garbled_pages = filter_ocr_pages(page_results)
            if garbled_pages:
                print(f"  [OCR质检] 发现 {len(garbled_pages)} 页疑似乱码，"
                      f"提分辨率重试（{OCR_QUALITY_RETRY_SIZE}px 原图）...")
                dropped = []
                for fname, text, page_num in garbled_pages:
                    path = os.path.join(pdf_pages_dir, fname)
                    text_retry = ocr_single_page(path)  # 无降采样 = 更高分辨率
                    if text_retry.strip() and not is_garbled(text_retry):
                        ok_pages.append((fname, text_retry, page_num))
                    else:
                        dropped.append(page_num)
                if dropped:
                    print(f"  [OCR质检] 丢弃 {len(dropped)} 页乱码：page {dropped}")
            ok_pages.sort(key=lambda x: x[2])

            pdf_count = 0
            for filename, text, page_num in ok_pages:
                if text.strip():
                    raw_docs.append(Document(
                        page_content=text,
                        metadata={"source": PDF_SOURCE_NAME, "page": page_num}
                    ))
                    pdf_count += 1
            print(f"  PDF 摄入完成（{pdf_count} 页有文本）")
        else:
            print(f"  {pdf_pages_dir}/ 为空，跳过 PDF")

        # OCR 后释放内存
        gc.collect()
        time.sleep(0.5)
    else:
        print(f"  未找到 pdf_pages/ 目录，仅加载 CSV")

    print(f"[1/7] 加载 {len(raw_docs)} 条文档（CSV {csv_count} + PDF {len(raw_docs)-csv_count}）")

    # 语义预分块：段落/动作边界优先
    raw_docs = _semantic_pre_chunk(raw_docs)
    print(f"[1a/7] 语义预分块后 {len(raw_docs)} 条（CSV 1行=1块, PDF 按段落切分）")

    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_PARENT_TOKENS,
        chunk_overlap=int(CHUNK_PARENT_TOKENS * CHUNK_OVERLAP_RATIO),
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_CHILD_TOKENS,
        chunk_overlap=int(CHUNK_CHILD_TOKENS * CHUNK_OVERLAP_RATIO),
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
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
            # 传递实体标签和原始元数据
            child.metadata["entity_labels"] = _extract_entity_labels(parent.metadata)
            for key in ("动作名称", "目标肌群", "器械", "难度", "source", "page"):
                if key in parent.metadata:
                    child.metadata[key] = parent.metadata[key]
        child_docs.extend(children)

    print(f"[2/7] 父块 {len(parent_docs)} | 子块 {len(child_docs)}")
    gc.collect()

    # ================================================================
    # 2. BM25 索引构建
    # ================================================================
    bm25_corpus = [list(jieba.cut(doc.page_content)) for doc in child_docs]
    bm25_idx = BM25Okapi(bm25_corpus)
    save_bm25_to_pickle(bm25_idx, child_docs, BM25_INDEX_PATH)
    print(f"[3/7] BM25 索引已保存至 {BM25_INDEX_PATH} ({len(child_docs)} 条)")
    gc.collect()

    # ================================================================
    # 3. Milvus Lite 向量库构建（嵌入式，无需 Docker）
    # ================================================================
    print(f"  正在初始化 embedding 模型（provider={EMBEDDING_PROVIDER}, model={EMBEDDING_CLOUD_MODEL if EMBEDDING_PROVIDER == 'cloud' else EMBEDDING_MODEL}）...")
    embeddings = build_embeddings()

    # 使用 Milvus Lite 嵌入式模式（uri 指向本地文件）
    client = MilvusClient(uri=MILVUS_URI, grpc_options=MILVUS_GRPC_OPTIONS)

    if client.has_collection(MILVUS_COLLECTION):
        try:
            client.drop_collection(MILVUS_COLLECTION)
        except Exception as e:
            # Windows 下 Milvus Lite 偶尔残留 .tmp 文件导致 drop 失败
            print(f"  [WARN] drop_collection 失败 ({e})，正在强制清理 milvus.db...")
            client.close()
            import shutil
            shutil.rmtree(MILVUS_URI, ignore_errors=True)
            client = MilvusClient(uri=MILVUS_URI, grpc_options=MILVUS_GRPC_OPTIONS)

    # 定义 schema
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_DIM)
    schema.add_field(field_name="page_content", datatype=DataType.VARCHAR, max_length=4096)
    schema.add_field(field_name="metadata_json", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="entity_labels", datatype=DataType.VARCHAR, max_length=1024)

    # 定义索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 16},
    )

    # 创建 collection
    client.create_collection(
        collection_name=MILVUS_COLLECTION,
        schema=schema,
        index_params=index_params,
    )

    # 批量写入（小批次 + 间隔，防 Ollama OOM）
    # 嵌入缓存：以**内容哈希**为键复用向量——文档改一处，其余块不必重复调 API
    embed_cache = EmbeddingCache(enabled=not FORCE_REBUILD)
    batch_size = EMBED_BATCH
    total_docs = len(child_docs)
    for i in range(0, total_docs, batch_size):
        batch = child_docs[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        vecs = embed_with_cache(embed_cache, texts, embeddings.embed_documents)
        data = [
            {
                "id": str(uuid.uuid4()),
                "embedding": vec,
                "page_content": doc.page_content,
                "metadata_json": json.dumps(doc.metadata, ensure_ascii=False),
                "parent_id": doc.metadata.get("parent_id", ""),
                "entity_labels": json.dumps(doc.metadata.get("entity_labels", []), ensure_ascii=False),
            }
            for doc, vec in zip(batch, vecs)
        ]
        client.insert(collection_name=MILVUS_COLLECTION, data=data)

        done = min(i + batch_size, total_docs)
        if done % 100 == 0 or done >= total_docs:
            print(f"  Milvus... {done}/{total_docs}")
        # 每 5 批释放一次内存，让 Ollama 喘气
        if (i // batch_size + 1) % 5 == 0:
            gc.collect()
            time.sleep(0.3)

    # 检查写入数量
    print(f"[4/7] Milvus 写入完成 ({len(child_docs)} 条, dim={MILVUS_DIM})")
    embed_cache.save()
    print(f"       {embed_cache.describe()}")

    # 保存父块 JSON
    parent_dir = "milvus_data"
    os.makedirs(parent_dir, exist_ok=True)
    with open(os.path.join(parent_dir, "parents.json"), "w", encoding="utf-8") as f:
        json.dump(parent_store, f, ensure_ascii=False)
    print(f"[5/7] 父块 JSON 已保存 ({len(parent_store)} 条)")

    # ================================================================
    # 4. Neo4j 知识图谱构建
    # ================================================================
    if SKIP_NEO4J or not NEO4J_ENABLED:
        reason = "--skip-neo4j" if SKIP_NEO4J else "NEO4J_ENABLED=False"
        print(f"[6/7] Neo4j 跳过 ({reason})")
    else:
        _build_neo4j(raw_docs)
        print(f"[6/7] Neo4j 图谱构建完成")

    client.close()

    # 记录本次摄入的源文件指纹：下次比对「内容是否变了」，未变可整轮跳过
    manifest.record(source_files)
    print(f"[7/7] 全部索引构建完毕！（已记录 {len(source_files)} 个源文件指纹）")


# ================================================================
# Neo4j 图谱：从 CSV 抽取 Entity + 关系
# ================================================================

def _build_neo4j(docs):
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # 清空旧数据
    driver.execute_query("MATCH (n) DETACH DELETE n", database_=NEO4J_DATABASE)
    # 约束（AuraDB 需要在对应 database 上创建约束）
    driver.execute_query(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        database_=NEO4J_DATABASE,
    )

    # 伤病 → 动作的禁忌/关联映射（单一数据源：contra_data.py，图谱与本地降级共用）
    from contra_data import INJURY_ACTION_MAP as injury_action_map

    with driver.session(database=NEO4J_DATABASE) as session:
        # 从 CSV 行抽取实体
        for doc in docs:
            name = doc.metadata.get("动作名称", "")
            muscles_raw = doc.metadata.get("目标肌群", "")
            equipment = doc.metadata.get("器械", "")

            if name:
                session.run(
                    "MERGE (e:Entity {name: $name}) SET e.type = 'exercise'",
                    name=name,
                )

            # 目标肌群 → 多个肌肉实体
            for muscle in re.split(r"[/、，,]", muscles_raw):
                muscle = muscle.strip()
                if muscle and len(muscle) >= 2:
                    session.run(
                        "MERGE (m:Entity {name: $name}) SET m.type = 'muscle'",
                        name=muscle,
                    )
                    session.run(
                        """MATCH (e:Entity {name: $exercise}), (m:Entity {name: $muscle})
                           MERGE (e)-[:TARGETS_MUSCLE {description: $desc}]->(m)""",
                        exercise=name, muscle=muscle,
                        desc=f"{name} 主要锻炼 {muscle}",
                    )

            # 器械
            if equipment and len(equipment) >= 2:
                session.run(
                    "MERGE (eq:Entity {name: $name}) SET eq.type = 'equipment'",
                    name=equipment,
                )
                session.run(
                    """MATCH (e:Entity {name: $exercise}), (eq:Entity {name: $equipment})
                       MERGE (e)-[:USES_EQUIPMENT {description: $desc}]->(eq)""",
                    exercise=name, equipment=equipment,
                    desc=f"{name} 使用 {equipment}",
                )

        # 伤病 → 动作映射
        for injury, actions in injury_action_map.items():
            session.run("MERGE (i:Entity {name: $name}) SET i.type = 'injury'", name=injury)
            for action_name, relation, desc in actions:
                session.run("MERGE (a:Entity {name: $name}) SET a.type = 'exercise'", name=action_name)
                session.run(
                    """MATCH (i:Entity {name: $injury}), (a:Entity {name: $action})
                       MERGE (i)-[:RELATES_TO {relation: $rel, description: $desc}]->(a)""",
                    injury=injury, action=action_name, rel=relation, desc=desc,
                )

    # ================================================================
    # 新增三元组: 肌群 → 损伤 (MUSCLE_INJURY)
    # ================================================================
    muscle_injury_map = {
        "腘绳肌": [("肌肉拉伤", "短跑/冲刺时腘绳肌易拉伤")],
        "股四头肌": [
            ("髌腱炎", "过度深蹲/跳跃导致股四头肌肌腱炎"),
            ("髌骨软化", "股四头肌力量不足导致髌骨轨迹异常"),
        ],
        "臀大肌": [("臀肌失忆症", "久坐导致臀肌无力，腰椎和腘绳肌代偿")],
        "竖脊肌": [("腰肌劳损", "久坐/错误弯腰发力导致竖脊肌劳损")],
        "三角肌": [
            ("肩袖损伤", "过度推举/侧平举导致三角肌肌腱炎"),
            ("肩峰撞击", "不正确推举角度导致肩峰下撞击"),
        ],
        "腹直肌": [("腹直肌分离", "产后/过度腹部训练导致腹白线分离")],
        "腓肠肌": [
            ("跟腱炎", "过度跑跳/提踵导致跟腱炎"),
            ("小腿拉伤", "突然加速或爆发力动作导致腓肠肌拉伤"),
        ],
        "斜方肌": [("颈椎病", "耸肩/头前伸导致上斜方肌过度紧张")],
        "胸大肌": [("圆肩", "胸肌过紧牵拉肩胛骨前倾形成圆肩")],
        "臀中肌": [("膝内扣", "臀中肌无力导致下蹲时膝盖内扣")],
        "阔筋膜张肌": [("髂胫束综合征", "阔筋膜张肌过紧摩擦股骨外髁")],
        "比目鱼肌": [("足底筋膜炎", "比目鱼肌紧张连锁引发足底筋膜张力异常")],
    }

    with driver.session(database=NEO4J_DATABASE) as session:
        for muscle, injuries in muscle_injury_map.items():
            session.run("MERGE (m:Entity {name: $name}) SET m.type = 'muscle'", name=muscle)
            for injury_name, desc in injuries:
                session.run("MERGE (i:Entity {name: $name}) SET i.type = 'injury'", name=injury_name)
                session.run(
                    """MATCH (m:Entity {name: $muscle, type: 'muscle'}),
                             (i:Entity {name: $injury, type: 'injury'})
                       MERGE (m)-[:MUSCLE_INJURY {description: $desc}]->(i)""",
                    muscle=muscle, injury=injury_name, desc=desc,
                )

    # ================================================================
    # 新增三元组: 器械 → 适用肌群 (EQUIPMENT_TARGETS)
    # （从 CSV 自动推导：某器械关联的动作 → 目标肌群 → 聚合去重）
    # ================================================================
    equipment_muscles: dict[str, set] = {}
    for doc in docs:
        eq = doc.metadata.get("器械", "")
        muscles_raw = doc.metadata.get("目标肌群", "")
        if eq and muscles_raw:
            for m in re.split(r"[/、，,]", muscles_raw):
                m = m.strip()
                if len(m) >= 2:
                    equipment_muscles.setdefault(eq, set()).add(m)

    with driver.session(database=NEO4J_DATABASE) as session:
        for eq_name, muscle_set in equipment_muscles.items():
            session.run("MERGE (eq:Entity {name: $name}) SET eq.type = 'equipment'", name=eq_name)
            for muscle_name in muscle_set:
                session.run("MERGE (m:Entity {name: $name}) SET m.type = 'muscle'", name=muscle_name)
                session.run(
                    """MATCH (eq:Entity {name: $eq, type: 'equipment'}),
                             (m:Entity {name: $muscle, type: 'muscle'})
                       MERGE (eq)-[:EQUIPMENT_TARGETS {description: $desc}]->(m)""",
                    eq=eq_name, muscle=muscle_name,
                    desc=f"{eq_name} 可用于训练 {muscle_name}",
                )

    # ================================================================
    # 新增三元组: 人群 → 训练方案 (POPULATION_PLAN)
    # ================================================================
    population_plan_map = {
        "大体重": [
            ("游泳", "零冲击全身运动适合大体重燃脂"),
            ("快走", "低冲击每天30分钟起步"),
            ("椭圆机", "无跑跳冲击保护膝盖"),
            ("避免跑步", "高冲击加重膝关节负担"),
            ("避免跳绳", "跳跃冲击不适合大体重"),
        ],
        "新手": [
            ("全身训练", "每周3次学习基础动作模式"),
            ("低强度起步", "从徒手动作开始逐步增加负重"),
            ("核心激活", "优先学会核心收紧和呼吸配合"),
            ("避免过度训练", "新手易过度训练导致关节损伤"),
        ],
        "老年人": [
            ("抗阻训练", "轻重量多次数维持肌肉量"),
            ("平衡训练", "单腿站立/太极预防跌倒"),
            ("柔韧性训练", "每日拉伸维持关节活动度"),
            ("避免大重量深蹲", "骨质疏松风险需避免脊柱负重"),
        ],
        "久坐人群": [
            ("髋屈肌拉伸", "久坐导致髋屈肌缩短需每日拉伸"),
            ("上背强化", "面拉/划船纠正圆肩驼背"),
            ("核心激活", "平板支撑/死虫式激活深层核心"),
        ],
        "产后": [
            ("腹直肌修复", "避免卷腹类动作优先腹横肌激活"),
            ("盆底肌训练", "凯格尔运动恢复盆底功能"),
            ("低冲击有氧", "快走/游泳逐步恢复体能"),
        ],
    }

    with driver.session(database=NEO4J_DATABASE) as session:
        for population, plans in population_plan_map.items():
            session.run("MERGE (p:Entity {name: $name}) SET p.type = 'population'", name=population)
            for plan_name, desc in plans:
                session.run("MERGE (pl:Entity {name: $name}) SET pl.type = 'training_plan'", name=plan_name)
                session.run(
                    """MATCH (p:Entity {name: $pop, type: 'population'}),
                             (pl:Entity {name: $plan, type: 'training_plan'})
                       MERGE (p)-[:POPULATION_PLAN {description: $desc}]->(pl)""",
                    pop=population, plan=plan_name, desc=desc,
                )

    driver.close()


if __name__ == "__main__":
    build()
