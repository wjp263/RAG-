"""
知识库
"""
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from logger import setup_logger
logger = setup_logger(__name__)


def check_md5(md5_str: str):
    """检查传入的md5字符串是否已经被处理过了
        return False(md5未处理过)  True(已经处理过，已有记录）
    """
    if not os.path.exists(config.md5_path):
        # if进入表示文件不存在，那肯定没有处理过这个md5了
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip()     # 处理字符串前后的空格和回车
            if line == md5_str:
                return True         # 已处理过

        return False


def save_md5(md5_str: str):
    """将传入的md5字符串，记录到文件内保存"""
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8'):
    """将传入的字符串转换为md5字符串"""

    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)

    # 创建md5对象
    md5_obj = hashlib.md5()     # 得到md5对象
    md5_obj.update(str_bytes)   # 更新内容（传入即将要转换的字节数组）
    md5_hex = md5_obj.hexdigest()       # 得到md5的十六进制字符串

    return md5_hex


class KnowledgeBaseService(object):
    def __init__(self):
        # 如果文件夹不存在则创建，如果存在则跳过
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.collection_name,     # 数据库的表名
            embedding_function=config.get_embedding(),
            persist_directory=config.persist_directory,     # 数据库本地存储文件夹
        )     # 向量存储的实例 Chroma向量库对象

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,       # 分割后的文本段最大长度
            chunk_overlap=config.chunk_overlap,     # 连续文本段之间的字符重叠数量
            separators=config.separators,       # 自然段落划分的符号
            length_function=len,                # 使用Python自带的len函数做长度统计的依据
        )     # 文本分割器的对象

        # ---- 新增：检查并加载初始知识库 ----
        self._check_and_load_initial_data()

    def _check_and_load_initial_data(self):
        """检查向量库是否为空，若为空则从 initial_knowledge.txt 加载初始数据"""
        try:
            # 获取集合中的文档数量（粗略判断是否为空）
            # 注意：Chroma 的 get() 可能返回很多数据，这里只取一个文档来快速判断
            existing_docs = self.chroma.get(limit=1)
            if existing_docs and existing_docs['ids']:
                logger.info("向量数据库已有数据，跳过初始知识库加载")
                return
        except Exception as e:
            logger.warning(f"检查向量库数据时出错: {e}，将尝试加载初始数据")

        # 尝试读取初始知识库文件
        initial_file_path = "initial_knowledge.txt"
        if not os.path.exists(initial_file_path):
            logger.info("未找到 initial_knowledge.txt 文件，跳过初始加载")
            return

        try:
            with open(initial_file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if not content:
                logger.info("initial_knowledge.txt 文件为空，跳过加载")
                return

            logger.info("检测到向量库为空，开始加载初始知识库...")
            # 调用 upload_by_str 方法载入，传入一个固定的虚拟文件名
            result = self.upload_by_str(content, "initial_knowledge.txt")
            logger.info(f"初始知识库加载完成: {result}")
        except Exception as e:
            logger.error(f"加载初始知识库失败: {e}")

    # 以下原有的 upload_by_str 方法保持不变...
    def upload_by_str(self, data: str, filename):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        # 先得到传入字符串的md5值
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            logger.info(f"Skip upload: {filename} already exists")
            return "[跳过]内容已经存在知识库中"

        if len(data) > config.max_split_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "system",   # 将默认操作员改为通用名称
        }

        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )
        save_md5(md5_hex)

        logger.info(f"Successfully uploaded {filename} with {len(knowledge_chunks)} chunks")
        return "[成功]内容已经成功载入向量库"


if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("周杰轮222", "testfile")
    print(r)
