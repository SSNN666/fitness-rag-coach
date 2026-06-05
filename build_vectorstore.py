from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS    # 改用 FAISS
from config import *

def build_vectorstore():
    loader = CSVLoader(file_path=CSV_FILE, encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n", "。", "！", "？", "；", "，", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(splits, embeddings)

    # FAISS 持久化：保存到本地文件夹
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"✅ FAISS 向量库已保存至 {VECTORSTORE_DIR}，共 {len(splits)} 条切片")

if __name__ == "__main__":
    build_vectorstore()