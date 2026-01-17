import json
import os
import traceback
from pathlib import Path

import torch
from dotenv import find_dotenv, load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from transformers import AutoTokenizer

from utils import MarkdownTreeParser, encode_document_for_milvus


def batch_by_token(documents, max_tokens_per_batch=512):  # 例如 512 或 1024
    """
    根据token数量对文档进行分批
    
    Args:
        documents: 文档列表
        max_tokens_per_batch: 每批的最大token数量
    Returns:
        批次列表
    """
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


def build_knowledge_base(
    embedding_model_path="../models/Qwen/Qwen3-Embedding-0.6B",
    markdown_folder_path="../raw_data",
    collection_name="knowledge_base_v1",
    max_tokens_per_batch=2048,
    min_chunk_size=256,
    core_chunk_size=512,
    max_chunk_size=2048,
    milvus_host=None,
    milvus_port=None,
    milvus_user=None,
    milvus_password=None,
    index_type="FLAT",
    metric_type="COSINE"
):
    """
    构建知识库到Milvus
    
    Args:
        embedding_model_path: 嵌入模型路径
        markdown_folder_path: Markdown文件夹路径
        collection_name: Milvus集合名
        max_tokens_per_batch: 批量存入数据库的批token数量上限
        min_chunk_size: 最小块大小
        core_chunk_size: 核心块大小
        max_chunk_size: 最大块大小
        milvus_host: Milvus主机地址
        milvus_port: Milvus端口
        milvus_user: Milvus用户名
        milvus_password: Milvus密码
        index_type: 索引类型
        metric_type: 度量类型
    """
    
    # 加载环境变量
    load_dotenv(find_dotenv())
    if milvus_host is None:
        milvus_host = os.getenv('MILVUS_HOST', 'localhost')
    if milvus_port is None:
        milvus_port = os.getenv('MILVUS_PORT', '19530')
    if milvus_user is None:
        milvus_user = os.getenv('MILVUS_USER', 'root')
    if milvus_password is None:
        milvus_password = os.getenv('MILVUS_ROOT_PASSWORD', 'Milvus')

    # 1. 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 2. 初始化 MarkdownTreeParser
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_path, trust_remote_code=True)
    parser = MarkdownTreeParser(
        embeddings=embeddings,
        tokenizer=tokenizer,
        min_chunk_size=min_chunk_size,
        core_chunk_size=core_chunk_size,
        max_chunk_size=max_chunk_size,
    )

    # 3. 解析所有 Markdown 文件
    docs = []
    for file in os.listdir(markdown_folder_path):
        print(f"\r正在处理：{file}, 已完成：{len(docs)}", end="", flush=True)
        if file.endswith(".md"):
            file_path = os.path.join(markdown_folder_path, file)
            docs.extend(parser.parse_markdown_to_tree(Path(file_path)))
            torch.cuda.empty_cache()
    total = len(docs)

    # 4. 存入 Milvus
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={
            "host": milvus_host,
            "port": milvus_port,
            "user": milvus_user,
            "password": milvus_password,
        },
        collection_name=collection_name,
        # 注意：index_params 在 add_texts 首次调用时才会生效（如果集合不存在）
        index_params={
            "index_type": index_type,
            "metric_type": metric_type,
        },
        # 搜索参数
        search_params={
            "metric_type": metric_type,
        }
    )

    max_token = 0
    doc_count = 0
    for i, batch in enumerate(batch_by_token(docs, max_tokens_per_batch)):
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


if __name__ == "__main__":
    # 使用默认参数运行知识库构建
    build_knowledge_base()