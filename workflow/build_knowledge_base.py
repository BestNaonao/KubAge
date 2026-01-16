import json
import os
import traceback
import time
from pathlib import Path

import torch
from dotenv import find_dotenv, load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from utils import MarkdownTreeParser, encode_document_for_milvus


# ==========================================
# Schema 定义
# ==========================================
def create_hybrid_schema(dense_dim):
    """
    定义支持混合检索的 Schema
    """
    # 1. 基础字段
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, max_length=256, is_primary=True, auto_id=False, description="主键"),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, description="文档文本内容"),
        # Metadata 字段
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024, description="源文件路径"),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024, description="节点标题"),
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="child_ids", dtype=DataType.VARCHAR, max_length=65535),  # JSON string
        FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="level", dtype=DataType.INT64),
        FieldSchema(name="token_count", dtype=DataType.INT64),
        FieldSchema(name="left_sibling", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="right_sibling", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="from_split", dtype=DataType.BOOL),
        FieldSchema(name="merged", dtype=DataType.BOOL),
        FieldSchema(name="nav_next_step", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="nav_see_also", dtype=DataType.VARCHAR, max_length=65535),

        # 2. 向量字段
        # (A) Text 的稠密向量 (Qwen 生成)
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim, description="Text Dense Vector"),
        # (B) Text 的稀疏向量 (BGE-M3 生成)
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR, description="Text Sparse Vector"),
        # (C) Title 的稀疏向量 (BGE-M3 生成)
        FieldSchema(name="title_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR, description="Title Sparse Vector")
    ]

    schema = CollectionSchema(fields, description="Hybrid Search Knowledge Base")
    return schema


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
        if tokens > max_tokens_per_batch:  # 单个文档超限
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

def csr_to_milvus_format(csr_matrix):
    """
    将 Scipy CSR 矩阵高效转换为 Milvus 接受的字典列表格式
    格式: [{token_id: weight, ...}, ...]
    """
    results = []
    # 使用 CSR 内部结构进行遍历，速度极快且不会报错
    for i in range(csr_matrix.shape[0]):
        start = csr_matrix.indptr[i]
        end = csr_matrix.indptr[i + 1]

        # 提取当前行的非零元素索引和值
        indices = csr_matrix.indices[start:end]
        data = csr_matrix.data[start:end]

        # 转换为字典 {int: float}
        row_dict = {int(k): float(v) for k, v in zip(indices, data)}
        results.append(row_dict)
    return results

def build_knowledge_base(
        embedding_model_path="../models/Qwen/Qwen3-Embedding-0.6B",
        sparse_model_path="BAAI/bge-m3",  # 新增：稀疏向量模型路径
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
    # 加载环境变量
    load_dotenv(find_dotenv())
    host = milvus_host or os.getenv('MILVUS_HOST', 'localhost')
    port = milvus_port or os.getenv('MILVUS_PORT', '19530')
    user = milvus_user or os.getenv('MILVUS_USER', 'root')
    password = milvus_password or os.getenv('MILVUS_ROOT_PASSWORD', 'Milvus')

    # 1. 连接 Milvus
    print(f"正在连接 Milvus ({host}:{port})...")
    connections.connect(alias="default", host=host, port=port, user=user, password=password)

    # 2. 初始化 Dense Embedding 模型 (Qwen)
    print(f"正在加载 Dense 模型: {embedding_model_path}...")
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 获取稠密向量维度 (用于 Schema)
    test_vec = dense_embeddings.embed_query("test")
    dense_dim = len(test_vec)
    print(f"Dense 向量维度: {dense_dim}")

    # 3. 初始化 Sparse Embedding 模型 (BGE-M3)
    # BGE-M3 是目前生成稀疏向量(SPLADE/BM25 style)的最佳模型之一
    print(f"正在加载 Sparse 模型: {sparse_model_path}...")
    sparse_ef = BGEM3EmbeddingFunction(
        model_name=sparse_model_path,
        use_fp16=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 4. 初始化 MarkdownTreeParser
    print("正在初始化解析器...")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_path, trust_remote_code=True)
    parser = MarkdownTreeParser(
        embeddings=dense_embeddings,  # 这里的 embeddings 仅用于 parser 内部计算相似度合并等
        tokenizer=tokenizer,
        min_chunk_size=min_chunk_size,
        core_chunk_size=core_chunk_size,
        max_chunk_size=max_chunk_size,
    )

    # 5. 创建或重建集合
    if utility.has_collection(collection_name):
        print(f"集合 {collection_name} 已存在，正在删除重建...")
        utility.drop_collection(collection_name)

    schema = create_hybrid_schema(dense_dim)
    collection = Collection(name=collection_name, schema=schema)
    print(f"集合 {collection_name} 创建成功。")

    # 6. 解析所有 Markdown 文件
    docs = []
    print("开始解析 Markdown 文件...")
    for file in os.listdir(markdown_folder_path):
        if file.endswith(".md"):
            print(f"\r处理文件: {file} | 已累计文档: {len(docs)}", end="", flush=True)
            file_path = Path(markdown_folder_path) / file
            # 注意：解析过程可能会很长
            docs.extend(parser.parse_markdown_to_tree(file_path))
            torch.cuda.empty_cache()
    print(f"\n解析完成，共 {len(docs)} 个文档片段。")

    # 7. 批量处理、生成向量并存入 Milvus
    total = len(docs)
    processed_count = 0

    for i, batch in enumerate(batch_by_token(docs, max_tokens_per_batch)):
        batch_start_time = time.time()

        # 7.1 准备基础数据 (Metadata 和 Text)
        # 假设 encode_document_for_milvus 返回的是整理好的 Document 对象
        encoded_batch = [encode_document_for_milvus(doc) for doc in batch]

        texts = [doc.page_content for doc in encoded_batch]
        titles = [doc.metadata.get('title', '') for doc in encoded_batch]

        # 7.2 生成 Dense Vectors (Qwen) - 用于 text
        text_dense_vectors = dense_embeddings.embed_documents(texts)

        # 7.3 生成 Sparse Vectors (BGE-M3)
        # compute_source_embedding=False 表示只生成 content 的 embedding
        # BGEM3EmbeddingFunction 的 encode_documents 返回结果通常包含 'dense', 'sparse', 'colbert_vecs'
        # 我们这里直接调用它的底层 sparse 编码逻辑或者通过 wrapper 调用
        # 注意：pymilvus 自带的 BGEM3EmbeddingFunction `encode_documents` 返回的是 generator 或 list

        # 获取原始 CSR 矩阵
        text_csr = sparse_ef.encode_documents(texts)["sparse"]
        title_csr = sparse_ef.encode_documents(titles)["sparse"]

        # [关键修改] 将 CSR 矩阵转换为 list of dicts
        text_sparse_list = csr_to_milvus_format(text_csr)
        title_sparse_list = csr_to_milvus_format(title_csr)

        # 7.4 组装数据用于 Collection.insert
        # Pymilvus insert 需要 list of list (按列) 或者 list of dict (按行)
        # 这里构造 list of dict 更清晰
        insert_data = []
        for j, doc in enumerate(encoded_batch):
            meta = doc.metadata

            entry = {
                "pk": doc.id,
                "text": texts[j],
                "vector": text_dense_vectors[j],  # 对应 text (Dense)
                "sparse_vector": text_sparse_list[j],  # 对应 text (Sparse)
                "title_sparse": title_sparse_list[j],  # 对应 title (Sparse)
                "source": meta.get("source"),
                "title": titles[j],
                "parent_id": meta.get("parent_id"),
                "child_ids": meta.get("child_ids"),
                "node_type": meta.get("node_type"),
                "level": meta.get("level"),
                "token_count": meta.get("token_count"),
                "left_sibling": meta.get("left_sibling"),
                "right_sibling": meta.get("right_sibling"),
                "from_split": meta.get("from_split"),
                "merged": meta.get("merged"),
                "nav_next_step": meta.get("nav_next_step"),
                "nav_see_also": meta.get("nav_see_also")
            }
            insert_data.append(entry)

        # 7.5 执行插入
        try:
            collection.insert(insert_data)
            processed_count += len(batch)
            batch_used_time = time.time() - batch_start_time
            print(
                f"\r[Batch {i + 1}] 写入 {len(batch)} 条 | 耗时 {batch_used_time:.2f}s | 进度: {processed_count}/{total}",
                end="", flush=True)
        except Exception as e:
            print(f"\n[Error] 批次写入失败: {e}")
            traceback.print_exc()
            print(f"文档标题: {json.dumps([entry['title'] for entry in insert_data], ensure_ascii=False)}")
            # 可以选择在这里 break 或者 continue

        torch.cuda.empty_cache()

    # 8. 创建索引 (在数据插入后创建索引通常更快)
    print("\n\n正在创建索引...")

    # 8.1 稠密向量索引
    index_params_dense = {
        "index_type": index_type,
        "metric_type": metric_type,
    }
    collection.create_index("vector", index_params_dense)
    print("稠密索引 (vector) 创建完成。")

    # 8.2 稀疏向量索引 (Text)
    index_params_sparse = {
        "metric_type": "IP",  # 稀疏向量通常使用 IP (Inner Product)
        "index_type": "SPARSE_INVERTED_INDEX",  # Milvus 稀疏倒排索引
        "params": {"drop_ratio_build": 0.2}  # 过滤掉贡献小的维度以加速
    }
    collection.create_index("sparse_vector", index_params_sparse)
    print("稀疏索引 (sparse_vector) 创建完成。")

    # 8.3 稀疏向量索引 (Title)
    collection.create_index("title_sparse", index_params_sparse)
    print("稀疏索引 (title_sparse) 创建完成。")

    # 9. 加载集合以供查询
    collection.load()
    print("[完成] 知识库构建完成，集合已加载！")


if __name__ == "__main__":
    build_knowledge_base()