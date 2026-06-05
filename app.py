import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from config import *

st.set_page_config(page_title="AI 健身教练", page_icon="🏋️")
st.title("🏋️ AI 健身教练（本地 Qwen + RAG）")

@st.cache_resource
def load_retriever():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})

retriever = load_retriever()
llm = ChatOllama(model=LLM_MODEL, temperature=0.7)

system_prompt = """你是一个专业健身教练AI，会根据用户的身体数据和健身目标，并参考检索到的健身动作知识，给出安全、个性化的训练建议。

用户画像：
{user_profile}

你可以参考以下动作信息：
{context}

如果用户的问题与健身无关，请礼貌地引导回健身话题。
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

if "user_profiles" not in st.session_state:
    st.session_state.user_profiles = {}

def get_user_profile(session_id):
    return st.session_state.user_profiles.get(session_id, "暂无身体数据")

def build_chain():
    # 检索函数
    def retrieve_context(input_dict):
        question = input_dict.get("question", "")
        docs = retriever.invoke(question)
        return format_docs(docs)

    # 使用 RunnablePassthrough.assign 自动构建输入字典
    rag_chain = (
        RunnablePassthrough.assign(context=retrieve_context)
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

chain_with_history = build_chain()

# --- 侧边栏 ---
st.sidebar.header("⚙️ 我的身体数据")
with st.sidebar.form("profile_form"):
    height = st.number_input("身高（cm）", min_value=100, max_value=250, value=170)
    weight = st.number_input("体重（kg）", min_value=30, max_value=300, value=70)
    goal = st.selectbox("健身目标", ["增肌", "减脂", "塑形", "力量提升"])
    submitted = st.form_submit_button("保存画像")
    if submitted:
        session_id = "default_user"
        profile_str = f"身高{height}cm，体重{weight}kg，目标：{goal}"
        st.session_state.user_profiles[session_id] = profile_str
        st.sidebar.success("✅ 身体数据已保存")
        st.cache_data.clear()
        st.rerun()#强制刷新页面，避免 DOM 冲突

st.sidebar.markdown("---")
st.sidebar.caption("数据保存后，教练将基于你的画像给出建议。")

# --- 聊天区域（稳定版） ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_input := st.chat_input("请输入你的健身问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    session_id = "default_user"
    input_data = {
        "question": prompt_input,
        "user_profile": get_user_profile(session_id),
    }

    with st.chat_message("assistant"):
        #  完全放弃流式，收集完整响应后一次性渲染
        full_response = ""
        for chunk in chain_with_history.stream(
            input_data,
            config={"configurable": {"session_id": session_id}}
        ):
            full_response += chunk
        # 在 for 循环外面一次性显示
        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})