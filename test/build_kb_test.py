import os

import torch
from dotenv import find_dotenv, load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from transformers import AutoTokenizer
from utils import MarkdownTreeParser, encode_document_for_milvus, decode_document_from_milvus


load_dotenv(find_dotenv())
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')
MILVUS_USER = os.getenv('MILVUS_USER')
MILVUS_PASSWORD = os.getenv('MILVUS_ROOT_PASSWORD')

RAW_DATA_DIR = "../raw_data"

def chunk_list(lst, chunk_size):
    """将列表按固定大小切分为多个子列表"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

qwen_path = "../models/Qwen/Qwen3-Embedding-0.6B"
bge_path = "D:/学习资料/毕业设计/KubAge/models/BAAI/bge-large-zh-v1___5"

# 1. 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name=qwen_path,
    model_kwargs={"device": "cuda", "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)

# 2. 初始化 MarkdownTreeParser
tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
parser = MarkdownTreeParser(
    embeddings=embeddings,
    tokenizer=tokenizer,
    min_chunk_size=256,
    core_chunk_size=1024,
    max_chunk_size=4096
)

# 3. 解析所有 Markdown 文件
docs = []
for file in os.listdir(RAW_DATA_DIR):
    print(f"正在处理：{file}, 已完成：{len(docs)}")
    if file.endswith(".md"):
        file_path = os.path.join(RAW_DATA_DIR, file)
        docs.extend(parser.parse_markdown_to_tree(file_path))
        torch.cuda.empty_cache()

# 4. 存入 Milvus
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={
        "host": MILVUS_HOST,
        "port": MILVUS_PORT,
        "user": MILVUS_USER,
        "password": MILVUS_PASSWORD,
    },
    collection_name="knowledge_base_v1",
    # 注意：index_params 在 add_texts 首次调用时才会生效（如果集合不存在）
    index_params={
        "index_type": "FLAT",
        "metric_type": "COSINE",
    },
    # 搜索参数
    search_params={
        "metric_type": "COSINE",
    }
)


max_batch_size = 1000  # 建议略低于 5461
for i, batch in enumerate(chunk_list(docs, max_batch_size)):
    # 编码文档以适应Milvus存储
    encoded_batch = [encode_document_for_milvus(doc) for doc in batch]
    print(f"[4.{i + 1}] 正在写入批次 {i + 1}，共 {len(batch)} 条...")
    vector_store.add_documents(documents=encoded_batch)
print("[完成] 文档构建成功！")