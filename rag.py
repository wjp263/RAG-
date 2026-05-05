from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from logger import setup_logger
logger = setup_logger(__name__)
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi


def print_prompt(prompt):
    print("="*20)#生成=====用来做分隔符
    print(prompt.to_string())#将自身转化为字符串
    print("="*20)
    return prompt

class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(embedding=config.get_embedding())
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个专业的客服助手。你必须始终遵循以下规则："
                           "1.如果【参考资料】中不包含回答用户问题所需的信息，你必须直接回答：‘抱歉，知识库中没有相关信息，无法回答您的问题。’ 绝不要使用你自己的知识来补充,不能凭空捏造。'"
                            "2. 回答必须简洁且专业：每轮回答不超过10句话"
                            "3. 如果问题与历史重复，不要复述完整的历史过程，直接给出结论。"
                            "4. 当用户问‘你记得吗’、‘我刚才说了什么’等与记忆相关的问题时，你必须诚实回答‘我记得’或‘你刚才说的是……’，并准确复述对话历史中出现的相关信息。"
                            "5. 不要主动提及‘我无法记住’、‘每次对话独立处理’等否认记忆的表述，记住：你有当前对话的记忆能力。"
                           "以下是参考资料：{context}。\n\n"
                   "下面是对话历史："
                 ),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name)

        self.chain = self.__get_chain()

    def __get_chain(self):
        retriever = self.vector_service.get_retriever()

        def format_docs(docs: list[Document]) -> str:
            if not docs:
                return "无相关参考资料"
            return "\n\n".join([
                f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}"
                for doc in docs
            ])

        # 核心 RAG 链：输入 -> 检索 -> 添加上下文 -> 生成 prompt -> 调用模型
        rag_chain = (
                RunnablePassthrough.assign(
                    context=lambda x: format_docs(retriever.invoke(x["input"]))
                )
                | self.prompt_template
                | self.chat_model
                | StrOutputParser()
        )

        # 包装历史消息支持
        conversation_chain = RunnableWithMessageHistory(
            rag_chain,
            get_history,
            input_messages_key="input",  # 输入字典中用户问题的键名
            history_messages_key="history",  # 历史消息将会以 "history" 键注入
        )
        return conversation_chain

"============================================================================"
"""Agent"""

class AgentService:
    def __init__(self):
        from langchain_community.chat_models.tongyi import ChatTongyi
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
        from tools import search_knowledge_base, calculator
        import config_data as config

        self.tools = [search_knowledge_base, calculator]
        self.llm = ChatTongyi(model=config.chat_model_name, temperature=0)
        self.checkpointer = MemorySaver()
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.checkpointer,
            prompt="你是一个智能助手，可以调用知识库和计算器。请用中文回答,要求："
                   "1. 必须使用中文回答。"
                    "2. 如果遇到需要查询知识库信息，务必调用 `search_knowledge_base` 工具。"
                    "3. 如果需要进行数学计算，务必调用 `calculator` 工具。"
                   " 4. 回答要简洁、专业，并记住对话历史。"
        )

    def query(self, user_input: str, session_id: str) -> str:
        # 防御：输入为空时直接返回提示
        if not user_input or not isinstance(user_input, str) or user_input.strip() == "":
            return "请提出有效问题。"

        config = {"configurable": {"thread_id": session_id}}
        try:
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},  # 直接用字典，不用 HumanMessage 类
                config=config
            )
            # 提取最后一条 AI 消息
            return response["messages"][-1]["content"] if isinstance(response["messages"][-1], dict) else response["messages"][-1].content
        except Exception as e:
            return f"处理出错：{str(e)}"
