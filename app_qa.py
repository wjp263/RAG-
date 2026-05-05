"""Streamlit 智能客服前端"""
import os
os.environ["OTEL_SDK_DISABLED"] = "true"

from rag import RagService,AgentService
import streamlit as st
import config_data as config

# 标题
st.title("京东智能客服")
st.divider()            # 分隔符

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]

if "rag_service" not in st.session_state:
    st.session_state["rag_service"] = RagService()

if "agent_service" not in st.session_state:
    st.session_state["agent_service"] = AgentService()

# 侧边栏：选择服务模式
with st.sidebar:
    service_mode = st.radio(
        "选择服务模式",
        options=["标准 RAG 链", "ReAct Agent (支持工具调用)"],
        index=0,#默认选择RAG
        help="标准 RAG 链只检索知识库；Agent 可以自主调用知识库和计算器"
    )
    session_id = st.text_input("会话 ID", value="user_002")
    if st.button("清空历史"):
        from file_history_store import get_history
        get_history(session_id).clear()
        st.success("对话历史已清空")

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])#创建角色，显示历史记录

prompt = st.chat_input()# 在页面最下方提供用户输入栏

if prompt and prompt.strip():
    # 显示用户消息
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.spinner("思考中..."):
        if service_mode == "标准 RAG 链":
            # 调用标准 RAG 链
            res = st.session_state["rag_service"].chain.invoke(
                {"input": prompt},
                {"configurable": {"session_id": session_id}}
            )
            # 兼容不同的返回格式
            answer = res if isinstance(res, str) else res.get("output", str(res))
        else:
            # 调用 ReAct Agent
            answer = st.session_state["agent_service"].query(prompt, session_id)

    st.chat_message("assistant").write(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})