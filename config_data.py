
md5_path = "./md5.txt"


# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"


# spliter
chunk_size =200
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_split_char_number = 1000        # 文本分割的阈值

#
similarity_threshold =2            # 检索返回匹配的文档数量

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

session_config = {
        "configurable": {
            "session_id": "user_002",
        }
    }

# 单例 embedding
from langchain_community.embeddings import DashScopeEmbeddings
_embedding_instance = None

def get_embedding():
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = DashScopeEmbeddings(model=embedding_model_name)
    return _embedding_instance