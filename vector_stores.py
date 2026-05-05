from langchain_chroma import Chroma
import config_data as config


class VectorStoreService(object):#封装向量数据库的创建、配置和检索功能，便于调用
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,#集合名称
            embedding_function=self.embedding,#指定模型
            persist_directory=config.persist_directory,#持久化向量数据储存路径，即使重启也会保留它
        )

    def get_retriever(self):#检索器，使其可以加入lcel链使用invoke方法和|操作符
        """返回向量检索器，方便加入chain"""
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})#top=k


if __name__ == '__main__':#自测
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    res = retriever.invoke("我的体重100斤，尺码推荐")
    print(res)

