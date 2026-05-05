"""
Agent 工具箱
"""
# tools.py
from langchain_core.tools import tool
from langchain_chroma import Chroma
import config_data as config
from logger import setup_logger
logger = setup_logger(__name__)

# 添加以下单例缓存
_CHROMA_INSTANCE = None

def _get_chroma():
    global _CHROMA_INSTANCE
    if _CHROMA_INSTANCE is None:
        _CHROMA_INSTANCE = Chroma(
            collection_name=config.collection_name,
            embedding_function=config.get_embedding(),
            persist_directory=config.persist_directory,
        )
    return _CHROMA_INSTANCE

@tool
def search_knowledge_base(query: str) -> str:
    """从知识库中检索相关信息..."""
    vector_store = _get_chroma()
    retriever = vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})
    docs = retriever.invoke(query)
    if not docs:
        return "在知识库中未找到相关信息。"
    return "\n\n".join([doc.page_content for doc in docs])


# 计算器工具
@tool
def calculator(expression: str) -> float:
    """
    计算，支持基本算术运算（+ - * / // % ** 等）。
    """
    from simpleeval import simple_eval
    try:
        result = simple_eval(expression)
        return float(result)
    except Exception as e:
        import logging
        logging.warning(f"Calculator failed for '{expression}': {e}")
        return 0.0