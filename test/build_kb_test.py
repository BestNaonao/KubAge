import json
import os
import traceback

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

def chunk_by_token(documents, max_tokens_per_batch=512):  # 例如 512 或 1024
    batches = []
    current_batch = []
    current_tokens = 0

    for doc in documents:
        tokens = doc.metadata.get("token_count", 0)
        if tokens > max_tokens_per_batch:   # 单个文档超限
            print(f"警告：文档 {doc.metadata.get('source')} 超过单批限制 ({tokens} tokens)")
            # 这里可选择跳过、切分或单独处理
            batches.append([doc])  # 保守处理：单独成批
            continue

        if current_tokens + tokens > max_tokens_per_batch:
            batches.append(current_batch)
            current_batch = [doc]
            current_tokens = tokens
        else:
            current_batch.append(doc)
            current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    return batches

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
    core_chunk_size=512,
    max_chunk_size=2048,
)

# 3. 解析所有 Markdown 文件
docs = []
for file in os.listdir(RAW_DATA_DIR):
    print(f"\r正在处理：{file}, 已完成：{len(docs)}", end="", flush=True)
    if file.endswith(".md"):
        file_path = os.path.join(RAW_DATA_DIR, file)
        docs.extend(parser.parse_markdown_to_tree(file_path))
        torch.cuda.empty_cache()
total = len(docs)

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


max_token = 0
doc_count = 0
for i, batch in enumerate(chunk_by_token(docs, 2048)):
    # 编码文档以适应Milvus存储
    encoded_batch = [encode_document_for_milvus(doc) for doc in batch]
    titles = [doc.metadata.get('title', '') for doc in encoded_batch]
    sources = [doc.metadata.get('source', '') for doc in encoded_batch]
    token_counts = [doc.metadata.get('token_count', 0) for doc in encoded_batch]
    max_token = max(max_token, max(token_counts))

    print(f"\r[4.{i + 1}] 正在写入批次 {i + 1}，共 {len(batch)} 条... 已处理: {doc_count} / {total} 条,"
          f"最大token数: {max_token}, tokens: {json.dumps(token_counts, ensure_ascii=False)}, ", end="", flush=True)
    try:
        vector_store.add_documents(documents=encoded_batch, ids=[doc.id for doc in encoded_batch])
    except Exception as e:
        traceback.print_exc()
        print(f"文档: {json.dumps(titles, ensure_ascii=False)[-30:]}, "
          f"来自: {json.dumps(sources, ensure_ascii=False)[-30:]}")
        raise e
    doc_count += len(batch)
    torch.cuda.empty_cache()

print("\n[完成] 文档构建成功！")