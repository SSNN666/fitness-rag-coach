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
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient, DataType
from rank_bm25 import BM25Okapi

from config import *
from retriever import save_bm25_to_pickle

# 防 Windows OOM 蓝屏：限制 Ollama 并发
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "1")
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "2")

SKIP_NEO4J = "--skip-neo4j" in sys.argv
FAST_MODE = "--fast" in sys.argv

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
    - CSV 行（source='fitness_data.csv'）：每行 = 一个独立动作，保持完整
    - PDF 页：按双换行（段落边界）切分
    这样 token splitter 只会在单条目超出预算时才细分。
    """
    merged = []
    for doc in raw_docs:
        source = doc.metadata.get("source", "")
        if source == "fitness_data.csv":
            # CSV 每行已经是独立动作，保持原样
            merged.append(doc)
        else:
            # PDF：按段落切分
            paragraphs = re.split(r"\n\s*\n", doc.page_content)
            for para in paragraphs:
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
    print(f"[TIP]  加 --fast 使用快速防蓝屏模式，加 --skip-neo4j 跳过图谱\n")

    # ================================================================
    # 1. 加载数据源
    # ================================================================
    # 1a. CSV 动作库
    loader = CSVLoader(file_path=CSV_FILE, encoding="utf-8")
    raw_docs = loader.load()
    csv_count = len(raw_docs)

    # 1b. PDF 图谱页面（并行 OCR + 缓存 + 降采样）
    pdf_pages_dir = "pdf_pages"
    if os.path.isdir(pdf_pages_dir):
        from pdf_ocr import ocr_pages_parallel
        page_results = ocr_pages_parallel(
            pdf_pages_dir,
            max_workers=OCR_WORKERS,
            use_cache=True,
            max_image_size=MAX_IMAGE_SIZE,
        )
        if page_results:
            pdf_count = 0
            for filename, text, page_num in page_results:
                if text.strip():
                    raw_docs.append(Document(
                        page_content=text,
                        metadata={"source": "肌肉力量训练彩色图谱.pdf", "page": page_num}
                    ))
                    pdf_count += 1
            print(f"  PDF 摄入完成（{pdf_count} 页有文本）")
        else:
            print(f"  pdf_pages/ 为空，跳过 PDF")

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
    print(f"  正在初始化 Ollama embedding 模型 ({EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 使用 Milvus Lite 嵌入式模式（uri 指向本地文件）
    client = MilvusClient(uri=MILVUS_URI)

    if client.has_collection(MILVUS_COLLECTION):
        try:
            client.drop_collection(MILVUS_COLLECTION)
        except Exception as e:
            # Windows 下 Milvus Lite 偶尔残留 .tmp 文件导致 drop 失败
            print(f"  [WARN] drop_collection 失败 ({e})，正在强制清理 milvus.db...")
            client.close()
            import shutil
            shutil.rmtree(MILVUS_URI, ignore_errors=True)
            client = MilvusClient(uri=MILVUS_URI)

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
    batch_size = EMBED_BATCH
    total_docs = len(child_docs)
    for i in range(0, total_docs, batch_size):
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

    # 保存父块 JSON
    parent_dir = "milvus_data"
    os.makedirs(parent_dir, exist_ok=True)
    with open(os.path.join(parent_dir, "parents.json"), "w", encoding="utf-8") as f:
        json.dump(parent_store, f, ensure_ascii=False)
    print(f"[5/7] 父块 JSON 已保存 ({len(parent_store)} 条)")

    # ================================================================
    # 4. Neo4j 知识图谱构建
    # ================================================================
    if SKIP_NEO4J:
        print("[6/7] Neo4j 跳过 (--skip-neo4j)")
    else:
        _build_neo4j(raw_docs)
        print(f"[6/7] Neo4j 图谱构建完成")

    client.close()
    print("[7/7] 全部索引构建完毕！")


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

    # 伤病 → 动作的禁忌/关联映射（手动定义，覆盖测试集中所有伤病类型）
    injury_action_map = {
        # --- 腰椎 ---
        "腰间盘突出": [
            ("深蹲", "禁忌动作", "负重深蹲挤压椎间盘，加重突出"),
            ("硬拉", "禁忌动作", "硬拉需弓背发力，直接压迫腰椎"),
            ("臀桥", "康复动作", "强化臀肌分担腰椎压力"),
            ("平板支撑", "康复动作", "静态核心训练，腰椎零压力"),
        ],
        "腰突": [
            ("深蹲", "禁忌动作", "负重深蹲挤压椎间盘，加重突出"),
            ("硬拉", "禁忌动作", "硬拉需弓背发力，直接压迫腰椎"),
        ],
        "腰肌劳损": [
            ("硬拉", "禁忌动作", "劳损期负重会加重炎症"),
            ("猫牛式", "康复动作", "脊柱灵活性训练，缓解劳损僵硬"),
        ],
        "坐骨神经痛": [
            ("深蹲", "禁忌动作", "脊柱轴向负重压迫坐骨神经"),
            ("硬拉", "禁忌动作", "椎间孔受压加重坐骨神经症状"),
        ],
        # --- 膝关节 ---
        "半月板损伤": [
            ("深蹲", "禁忌动作", "膝盖屈伸负重挤压半月板"),
            ("直腿抬高", "康复动作", "静态抬腿强化股四头肌，零关节压力"),
        ],
        "半月板撕裂": [
            ("深蹲", "禁忌动作", "膝盖深度屈伸直接磨损撕裂半月板"),
            ("箭步蹲", "禁忌动作", "单腿负重旋转力加重半月板撕裂"),
        ],
        "髌骨软化": [
            ("深蹲", "谨慎动作", "控制下蹲深度不超过90度"),
            ("腿伸展", "禁忌动作", "开链动作加重髌股关节压力"),
        ],
        "膝内扣": [
            ("深蹲", "谨慎动作", "需弹力带辅助外展激活臀中肌"),
            ("臀中肌激活", "康复动作", "蚌式开合/侧卧抬腿纠正膝内扣"),
        ],
        "膝超伸": [
            ("腿伸展", "禁忌动作", "膝超伸加重关节后侧压力"),
            ("腿弯举", "康复动作", "强化腘绳肌稳定膝关节后侧"),
        ],
        "膝关节积液": [
            ("深蹲", "禁忌动作", "积液期高负荷加重炎症"),
            ("直腿抬高", "康复动作", "零关节压力的股四头肌激活"),
        ],
        # --- 肩部 ---
        "肩袖损伤": [
            ("推举", "禁忌动作", "肩关节外展加重肩袖撕裂"),
            ("侧平举", "禁忌动作", "肩外展直接牵拉损伤肩袖"),
            ("面拉", "康复动作", "强化肩袖外旋肌群稳定性"),
        ],
        "肩峰撞击": [
            ("推举", "禁忌动作", "肩外展超过90度加重撞击"),
            ("侧平举", "禁忌动作", "掌心向下侧举直接撞击肩峰"),
            ("面拉", "康复动作", "强化肩袖外旋肌群纠正肱骨头位置"),
        ],
        "肩周炎": [
            ("推举", "禁忌动作", "肩关节僵硬时大重量推举风险高"),
            ("钟摆运动", "康复动作", "轻幅度摆动保持关节活动度"),
        ],
        # --- 颈椎 ---
        "颈椎病": [
            ("杠铃耸肩", "禁忌动作", "耸肩压迫颈椎神经"),
            ("收下巴训练", "康复动作", "拉伸颈后肌群缓解压迫"),
        ],
        # --- 肘部 ---
        "网球肘": [
            ("哑铃弯举", "谨慎动作", "握力负荷可能加重前臂伸肌炎症"),
            ("锤式弯举", "康复动作", "中立握法减轻伸肌群负荷"),
        ],
        # --- 骨盆/脊柱 ---
        "骨盆前倾": [
            ("深蹲", "谨慎动作", "需先纠正体态再负重"),
            ("臀桥", "康复动作", "强化臀肌改善骨盆前倾"),
            ("髋屈肌拉伸", "康复动作", "松解紧张髋屈肌"),
        ],
        "骨盆后倾": [
            ("硬拉", "谨慎动作", "骨盆活动度不足时不可负重硬拉"),
            ("猫牛式", "康复动作", "改善脊柱灵活性"),
        ],
        "脊柱侧弯": [
            ("杠铃深蹲", "禁忌动作", "轴向负重加重侧弯不对称"),
            ("单侧训练", "康复动作", "针对性强化弱侧肌群纠正不平衡"),
        ],
        # --- 足部 ---
        "扁平足": [
            ("跑步", "谨慎动作", "需足弓支撑鞋垫"),
            ("足弓训练", "康复动作", "抓毛巾/提踵强化足弓"),
        ],
        "足底筋膜炎": [
            ("跑步", "禁忌动作", "足底反复冲击加重炎症"),
            ("足底滚球", "康复动作", "用网球放松足底筋膜"),
        ],
        "跟腱炎": [
            ("跑步", "禁忌动作", "跑跳加重跟腱炎症"),
            ("站姿提踵", "谨慎动作", "离心阶段需缓慢控制"),
        ],
        # --- 体态 ---
        "圆肩": [
            ("面拉", "康复动作", "强化菱形肌和肩外旋纠正圆肩"),
            ("卧推", "谨慎动作", "胸肌过紧可能加重圆肩"),
        ],
        "驼背": [
            ("卧推", "谨慎动作", "胸大肌缩短加重驼背体态"),
            ("划船", "康复动作", "强化上背肌群改善驼背"),
        ],
        "富贵包": [
            ("收下巴训练", "康复动作", "改善颈后肌群紧张"),
        ],
        "高低肩": [
            ("杠铃深蹲", "谨慎动作", "杠铃负荷不均加重肩部不对称"),
            ("单侧哑铃推举", "康复动作", "针对性纠正弱侧力量"),
        ],
        "梨状肌综合征": [
            ("深蹲", "禁忌动作", "髋关节深度屈伸加重梨状肌压迫坐骨神经"),
            ("硬拉", "禁忌动作", "负重硬拉使梨状肌过度紧张痉挛"),
            ("臀桥", "康复动作", "温和激活臀肌，注意幅度不宜过大"),
            ("骨盆倾斜", "康复动作", "放松梨状肌及下背部紧张"),
        ],
        "骶管狭窄": [
            ("深蹲", "禁忌动作", "脊柱轴向负重使椎管空间进一步变窄"),
            ("硬拉", "禁忌动作", "腰椎屈伸加重骶管神经压迫"),
            ("臀桥", "康复动作", "小幅度激活臀肌，避免腰椎过度伸展"),
            ("单腿臀桥", "康复动作", "单侧训练减少腰椎负荷"),
        ],
        "坐骨神经痛": [
            ("深蹲", "禁忌动作", "脊柱轴向负重压迫坐骨神经"),
            ("硬拉", "禁忌动作", "椎间孔受压加重坐骨神经症状"),
            ("臀桥", "康复动作", "温和激活臀肌，减轻神经根压迫"),
            ("骨盆倾斜", "康复动作", "改善腰椎-骨盆位置减轻坐骨神经张力"),
        ],
        "骶髂关节炎": [
            ("深蹲", "禁忌动作", "骶髂关节负重加重炎症"),
            ("单腿臀桥", "康复动作", "单侧训练避免直接压迫骶髂关节"),
        ],
    }

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
