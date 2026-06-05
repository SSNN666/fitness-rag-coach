# AI 健身教练 (RAG + Local LLM)
一个基于检索增强生成（RAG）技术的智能健身教练应用，完全本地运行，保护隐私。  
用户可以通过自然语言提问，AI 会从结构化知识库中检索相关健身动作，结合用户身体数据生成个性化训练建议。

## 功能特点
- **本地大模型**：基于 Ollama 运行 Qwen2.5 模型，无需联网
- **RAG 检索增强**：从结构化 CSV 知识库检索相关动作，提供精准回答
- **用户画像系统**：记录身高、体重、健身目标，回答个性化
- **对话记忆**：多轮对话上下文保持
- **完全离线运行**：所有数据都在本地，无隐私泄露风险

## 🛠️ 技术栈
- Python 3.12
- LangChain
- Streamlit
- Ollama (Qwen2.5:7b, Nomic Embed Text)
- FAISS (向量检索)
- UV (包管理)
  
## 安装与运行
推荐使用 uv
uv venv
uv pip install -r requirements.txt
安装 Ollama，然后拉取所需模型：
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
构建向量数据库
python build_vectorstore.py
启动应用
streamlit run app.py

使用说明
在侧边栏填写你的身体数据（身高、体重、健身目标），点击“保存画像”

在聊天框输入你的健身问题（例如：“怎么练胸肌？”、“增肌应该做什么动作？”）

AI 会结合你的身体数据和知识库给出个性化建议
