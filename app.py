"""
app.py —— Streamlit SSE 客户端（康养 Demo 重构）
================================================
- 零索引/LLM 依赖：全部能力由 FastAPI 后端（api.py）提供
- 消费 POST /v1/chat/stream 的 SSE 事件：meta → delta* → citations → done | error
- 保留：侧边栏身体画像表单、聊天渲染
- 新增：降级横幅（fallback_active）、结构化引用展示、token 消耗标注、
  伤病回答的图谱解释入口（一键跳到知识图谱视图并聚焦涉及伤病）、
  体检报告图文解读视图（上传图片 → /v1/vision 多模态解析）

启动（先 API 后 UI；Milvus Lite 单进程约束，后端禁用 --reload）：
  uvicorn api:app --host 127.0.0.1 --port 8000
  streamlit run app.py
"""

import json

import httpx
import streamlit as st

import config as _cfg
import graph_view

API_BASE = f"http://{_cfg.API_HOST}:{_cfg.API_PORT}"
API_KEY = _cfg.API_KEY_AUTH
SESSION_ID = "default_user"

st.set_page_config(page_title="AI 健身教练", page_icon="🏋️")
st.title("🏋️ AI 健身教练（康养知识库 RAG）")
st.caption(f"后端 API: {API_BASE} | 主链路: {_cfg.LLM_PROVIDER_PRIMARY}"
           " | 云端不可用时自动降级本地模型")

# ============================================================
# 侧边栏（保留原有画像表单）
# ============================================================
st.sidebar.header("⚙️ 我的身体数据")
with st.sidebar.form("profile_form"):
    height = st.number_input("身高（cm）", min_value=100, max_value=250, value=170)
    weight = st.number_input("体重（kg）", min_value=30, max_value=300, value=70)
    goal = st.selectbox("健身目标", ["增肌", "减脂", "塑形", "力量提升"])
    submitted = st.form_submit_button("保存画像")
    if submitted:
        profile_str = f"身高{height}cm，体重{weight}kg，目标：{goal}"
        st.session_state.user_profile = profile_str
        st.sidebar.success("✅ 身体数据已保存")
st.sidebar.markdown("---")
st.sidebar.header("🧠 生成选项")
if "deep_thinking" not in st.session_state:
    st.session_state.deep_thinking = False
st.session_state.deep_thinking = st.sidebar.checkbox(
    "深度思考模式",
    value=st.session_state.deep_thinking,
    help="伤病/计划类问题分析更深入，耗时更长（默认快速模式，回答后会有提示）")
st.sidebar.markdown("---")
st.sidebar.caption("数据保存后，教练将基于你的画像给出建议。")


# ============================================================
# SSE 客户端
# ============================================================

def _stream_events(question: str, session_id: str):
    """调用后端 SSE 接口，逐事件 yield (event, data)。"""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    payload = {
        "question": question,
        "session_id": session_id,
        "user_profile": st.session_state.get("user_profile"),
        "deep_thinking": st.session_state.get("deep_thinking", False),
    }
    with httpx.stream("POST", f"{API_BASE}/v1/chat/stream",
                      json=payload, headers=headers, timeout=None) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            yield data.get("event"), data


def _render_graph_button(focus: list[str], is_contra: bool, msg_idx: int) -> None:
    """渲染「查看图谱」按钮；点击 → 跳图谱视图 + 聚焦伤病（key 按消息序号唯一）。

    两个实测踩坑（最小 AppTest 复现确认）：
    1. 按钮必须渲染在 st.chat_input 块之外——点击 rerun 中 chat_input 返回 None、
       块整体跳过，块内按钮不被实例化 → 点击事件丢失；
    2. 点击送达后不能直接写 st.session_state.nav——该 run 中 segmented_control
       已实例化，写 widget key 抛 StreamlitAPIException → 经 query_params 中转。
    """
    kind = "禁忌关系" if is_contra else "伤病关系"
    if st.button(f"🕸️ 查看{kind}图谱（{'、'.join(focus)}）", key=f"graph_jump_{msg_idx}"):
        st.session_state.graph_focus = focus   # 图谱视图的 multiselect 本 run 未实例化，可安全写
        st.query_params["view"] = "graph"
        st.rerun()


# ============================================================
# 视图导航（状态驱动）
# st.tabs 无程序化选中 API，而聊天页「查看图谱」按钮需要跨视图跳转，
# 故用 segmented_control + session_state 驱动（观感与页签一致）。
# 跳转经 query_params 中转（widget 实例化后禁写其 session_state key）。
# ============================================================
if st.query_params.get("view") == "graph":
    st.session_state.nav = "🕸️ 知识图谱"
    st.query_params.clear()
if "nav" not in st.session_state:
    st.session_state.nav = "💬 智能问答"
nav = st.segmented_control(
    "视图",
    ["💬 智能问答", "🕸️ 知识图谱", "🖼️ 图文问答"],
    key="nav",
    label_visibility="collapsed",
)

# ============================================================
# 视图一：智能问答（聊天区域）
# ============================================================
if nav == "💬 智能问答":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 图谱入口按钮实例化在历史循环中（每次 run 都渲染），点击才不会被丢
            # （st.chat_input 块内的按钮在点击 rerun 中不实例化，见 _render_graph_button 注释）
            if (msg["role"] == "assistant" and i == len(st.session_state.messages) - 1
                    and msg.get("focus")):
                _render_graph_button(msg["focus"], msg.get("is_contra", False), i)

    if prompt_input := st.chat_input("请输入你的健身问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            # st.container(border=False) 隔离 st.empty()，规避前端 removeChild 竞态
            container = st.container(border=False)
            holder = container.empty()

            answer_parts: list[str] = []
            citations: list = []
            refusal = False
            grounded = True   # refusal=True 且 grounded=True → 禁忌拒绝；grounded=False → 无依据拒答
            done_info: dict = {}
            error_msg: str | None = None

            try:
                for event, data in _stream_events(prompt_input, SESSION_ID):
                    if event == "status":
                        # 流水线阶段进度（生成前的等待期反馈；delta 到来后覆盖）
                        holder.markdown(f"🔄 {data.get('stage', '处理中…')} ▌")
                    elif event == "delta":
                        answer_parts.append(data.get("text", ""))
                        holder.markdown("".join(answer_parts) + " ▌")
                    elif event == "citations":
                        citations = data.get("docs", [])
                        refusal = data.get("refusal", False)
                        grounded = data.get("grounded", True)
                    elif event == "done":
                        done_info = data
                    elif event == "error":
                        error_msg = data.get("message", "服务错误")
            except httpx.HTTPError:
                error_msg = f"无法连接后端服务（{API_BASE}），请先启动：uvicorn api:app --port {_cfg.API_PORT}"

            if error_msg:
                holder.markdown(f"⚠️ {error_msg}")
                final = f"⚠️ {error_msg}"
            else:
                final = "".join(answer_parts)
                holder.markdown(final)

            # 先写入历史（含 focus 元信息）：图谱按钮跨 run 渲染需要它，且 st.rerun() 会中断脚本
            _g, _ = graph_view.load_graph()
            focus = graph_view.match_focus(_g, prompt_input)
            is_contra = (refusal and grounded) or "已自动剔除禁忌动作" in final
            st.session_state.messages.append({
                "role": "assistant", "content": final,
                "focus": focus, "is_contra": is_contra,
            })

            if not error_msg:
                if done_info.get("fallback_active"):
                    st.info("⚡ 云端模型不可用，本次回答由本地模型生成（降级链已生效）。")

                if citations:
                    with st.expander("📚 参考来源"):
                        for c in citations:
                            if c.get("kind") == "web" and c.get("url"):
                                st.markdown(f"- 🌐 [{c.get('source', '来源')}]({c['url']})")
                            else:
                                snip = (c.get("snippet") or "")[:80]
                                st.markdown(f"- {c.get('source', '未知来源')} — {snip}")
                elif refusal:
                    if grounded:
                        st.caption("🛡️ 该动作属于伤病禁忌，已自动拒绝（安全拦截）。")
                    else:
                        st.caption("本次回答被拒答：知识库无相关依据。")

                # 伤病相关回答 → 图谱解释入口（本次 run 立即显示；后续 run 由历史循环实例化）
                # （禁忌拒绝/硬过滤必带伤病；普通伤病问答同样可查看该伤病的动作关系网，
                #   实测「腰突」类咨询型问题不触发拒绝/过滤，但回答含大量禁忌信息，同样需要入口）
                if focus:
                    _render_graph_button(focus, is_contra, len(st.session_state.messages) - 1)

                usage = done_info.get("usage") or []
                if usage:
                    total = sum(u.get("total_tokens", 0) for u in usage)
                    providers = ", ".join(u.get("provider", "") for u in usage)
                    mode = "🧠 深度思考" if st.session_state.get("deep_thinking") else "⚡ 快速"
                    st.caption(f"⚙️ tokens: {total} | provider: {providers} | 模式: {mode}")

# ============================================================
# 视图二：伤病禁忌知识图谱（ECharts 力导向图，零外部依赖）
# ============================================================
elif nav == "🕸️ 知识图谱":
    graph, source_note = graph_view.load_graph()
    st.markdown("### 🕸️ 伤病禁忌知识图谱")
    st.caption(
        f"数据源：{source_note}。与回答安全流水线（禁忌黑名单/硬过滤）共用同一数据源。"
        "拖拽节点 / 滚轮缩放 / 点选节点高亮关联。"
        "聊天页的「查看伤病关系图谱」按钮会跳转到本视图并自动聚焦涉及伤病。"
    )

    focus = st.multiselect(
        "聚焦伤病（可多选；留空 = 全图）",
        options=graph["injuries"],
        key="graph_focus",
        placeholder="例如：腰间盘突出",
    )
    sub = graph_view.filter_graph(graph, focus)
    st.caption(f"节点 {len(sub['nodes'])} · 关系 {len(sub['links'])}")
    st.iframe(graph_view.serve_page(sub), height=720)
    st.caption("图例：● 伤病（红） ● 动作（蓝） | 边：🔴 禁忌动作 · 🟡 谨慎动作 · 🟢 康复动作")

# ============================================================
# 视图三：体检报告图文解读（云端多模态 /v1/vision）
# ============================================================
else:
    st.markdown("### 🖼️ 体检报告图文解读")
    st.caption(
        "上传体检/健康报告图片，由云端多模态模型（qwen3-vl-plus）解析后回答问题。"
        "本地无 VL 模型——未配置 DASHSCOPE_API_KEY 时后端返回 503 明确降级提示。"
    )

    img = st.file_uploader(
        "上传图片（PNG/JPG，≤8MB）",
        type=["png", "jpg", "jpeg"],
        key="vision_upload",
    )
    vq = st.text_input(
        "关于这份报告的问题",
        placeholder="例如：这份报告有哪些异常指标？",
        key="vision_question",
    )

    if img and img.size > 8 * 1024 * 1024:
        st.warning("图片超过 8MB，请压缩后重试。")

    if st.button("🔍 开始解读", disabled=not (img and vq)):
        with st.spinner("正在解析图片并生成回答（多模态模型，约 10-30 秒）…"):
            try:
                headers = {}
                if API_KEY:
                    headers["X-API-Key"] = API_KEY
                r = httpx.post(
                    f"{API_BASE}/v1/vision",
                    files={"file": (img.name, img.getvalue(), img.type or "image/jpeg")},
                    data={"question": vq, "session_id": SESSION_ID},
                    headers=headers,
                    timeout=120.0,
                )
                if r.status_code == 200:
                    resp = r.json()
                    st.session_state.vision_result = {
                        "answer": resp.get("answer", ""),
                        "provider": resp.get("provider"),
                    }
                else:
                    try:
                        detail = r.json().get("detail", f"HTTP {r.status_code}")
                    except Exception:
                        detail = f"HTTP {r.status_code}"
                    msg = detail.get("message", detail) if isinstance(detail, dict) else str(detail)
                    st.session_state.vision_result = {"error": msg}
            except httpx.HTTPError:
                st.session_state.vision_result = {
                    "error": f"无法连接后端服务（{API_BASE}），请先启动：uvicorn api:app --port {_cfg.API_PORT}",
                }

    result = st.session_state.get("vision_result")
    if result:
        if result.get("error"):
            st.error(result["error"])
        else:
            st.image(img, caption="上传的图片", width=420)
            st.markdown(result["answer"])
            if result.get("provider"):
                st.caption(f"⚙️ 多模态模型: {result['provider']}")
