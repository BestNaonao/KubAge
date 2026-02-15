import json
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List

import torch
from pymilvus import (
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
from transformers import AutoTokenizer

from utils import MarkdownTreeParser, encode_document_for_milvus, csr_to_milvus_format, get_dense_embed_model, \
    get_sparse_embed_model
from utils.milvus_adapter import connect_milvus_by_env


# 定义分区名称常量
STATIC_PARTITION_NAME = "static_knowledge"
DYNAMIC_PARTITION_NAME = "dynamic_events"

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
        FieldSchema(name="child_ids", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=256,
                    max_length=64, description="子节点列表"),  # JSON string
        FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="level", dtype=DataType.INT64, description="节点层级"),
        FieldSchema(name="token_count", dtype=DataType.INT64),
        FieldSchema(name="left_sibling", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="right_sibling", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="from_split", dtype=DataType.BOOL),
        FieldSchema(name="merged", dtype=DataType.BOOL),
        FieldSchema(name="nav_next_step", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="nav_see_also", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="entry_urls", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=200,
                    max_length=1024, description="超链接入口列表"),
        FieldSchema(name="related_links", dtype=DataType.JSON, description="解析后的关联链路"),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=65535, description="摘要"),

        # 2. 向量字段
        # (A) Text 的稠密向量 (Qwen 生成)
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim, description="文本稠密向量"),
        # (B) Summary 的稠密向量 (Qwen 生成)
        FieldSchema(name="summary_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim, description="摘要语义向量"),
        # (C) Text 的稀疏向量 (BGE-M3 生成)
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR, description="文本稀疏向量"),
        # (D) Title 的稀疏向量 (BGE-M3 生成)
        FieldSchema(name="title_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR, description="标题稀疏向量")
    ]

    schema = CollectionSchema(fields, description="Hybrid Search Knowledge Base v3 (Graph Enhanced)")
    return schema


def batch_by_token(documents, max_tokens_per_batch=1024):  # 例如 1024
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


def resolve_links(documents: list):
    """
    全局链接解析
    构建 URL -> Document PK 的映射，并构建 document.metadata['related_links'] (JSON)
    """
    # 1. 构建全局注册表 (Registry)
    # Key: 完整的 Entry URL (例如 https://...#section-1)
    # Value: Document ID (UUID)
    url_registry = {}

    # 统计数据
    total_anchors = 0
    max_anchors = 0

    for doc in documents:
        # 获取该文档所有的入口锚点
        anchors = doc.metadata.get("anchors", [])
        max_anchors = max(max_anchors, len(anchors))
        doc_id = doc.id  # 确保 Document 对象有 id 属性

        for anchor in anchors:
            # 注意：如果有重复的 anchor (例如不同文档声明了同一个 URL)，后遍历的会覆盖前面的
            # 在 k8s 文档中，URL+Hash 应该是唯一的
            url_registry[anchor] = doc_id
            total_anchors += 1

    print(f"全局注册表构建完成: 包含 {len(documents)} 个文档, {total_anchors} 个锚点入口，单个文档块最多 {max_anchors} 个锚点。")

    # 2. 解析出口链接 (Resolution)
    resolved_count = 0
    total_outlinks = 0
    max_outlinks = 0

    for doc in documents:
        # outlinks 现在是字典: {"Anchor Text": ["url1", "url2"]}
        outlinks_map: Dict[str, List[str]] = doc.metadata.get("outlinks", {})

        # 准备构建 related_links 列表 (JSON 对象列表)
        resolved_links_data = []

        # 确保 outlinks_map 是字典
        if not isinstance(outlinks_map, dict):
            doc.metadata["related_links"] = []
            continue

        for anchor_text, urls in outlinks_map.items():
            for link in urls:
                total_outlinks += 1

                # 尝试查找 PK
                target_pk = url_registry.get(link)

                # 模糊匹配尝试 (去除 fragment)
                if not target_pk and '#' in link:
                    base_url = link.split('#')[0]
                    target_pk = url_registry.get(base_url)

                # 构建连接对象
                link_obj = {
                    "text": anchor_text,  # 锚点文本 (语义)
                    "url": link,  # 原始 URL
                    "pk": target_pk,  # 目标文档 ID (如果存在)
                    "type": "internal" if target_pk else "external"  # 连接类型
                }

                # 可选：如果你只关心内部链接，可以在这里 filter
                # 但保留 external 链接对 Agent 也有用
                resolved_links_data.append(link_obj)

                if target_pk:
                    resolved_count += 1

        # 将结果存回 metadata，准备写入 JSON 字段
        doc.metadata["related_links"] = resolved_links_data
        max_outlinks = max(max_outlinks, len(resolved_links_data))

    print(f"链接解析完成: 扫描 {total_outlinks} 个外链, 成功关联 {resolved_count} 个内部引用，单个文档块最多 {max_outlinks} 个链接。")

def build_knowledge_base(
        embedding_model_path="../models/Qwen/Qwen3-Embedding-0.6B",
        sparse_model_path="BAAI/bge-m3",  # 稀疏向量模型路径
        markdown_folder_path="../raw_data2",
        collection_name="knowledge_base_v3",
        max_tokens_per_batch=2048,
        min_chunk_size=256,
        core_chunk_size=512,
        max_chunk_size=2048,
        index_type="FLAT",
        metric_type="COSINE"
):
    # 1. 连接 Milvus
    connect_milvus_by_env()

    # 2. 初始化 Dense Embedding 模型 (Qwen)
    print(f"正在加载 Dense 模型: {embedding_model_path}...")
    dense_embeddings = get_dense_embed_model(embedding_model_path)

    # 获取稠密向量维度 (用于 Schema)
    test_vec = dense_embeddings.embed_query("test")
    dense_dim = len(test_vec)
    print(f"Dense 向量维度: {dense_dim}")

    # 3. 初始化 Sparse Embedding 模型 (BGE-M3)
    # BGE-M3 是目前生成稀疏向量(SPLADE/BM25 style)的最佳模型之一
    print(f"正在加载 Sparse 模型: {sparse_model_path}...")
    sparse_ef = get_sparse_embed_model(sparse_model_path)

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

    # 5. 创建或重建集合与分区
    if utility.has_collection(collection_name):
        print(f"集合 {collection_name} 已存在，正在删除重建...")
        utility.drop_collection(collection_name)

    schema = create_hybrid_schema(dense_dim)
    collection = Collection(name=collection_name, schema=schema)
    print(f"集合 {collection_name} 创建成功。")

    # 创建静态知识分区
    if not collection.has_partition(STATIC_PARTITION_NAME):
        collection.create_partition(STATIC_PARTITION_NAME)
        print(f"分区 {STATIC_PARTITION_NAME} 创建成功。")

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

    # 执行全局链接解析 ===
    resolve_links(docs)

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
        dense_start_time = time.time()
        text_dense_vectors = dense_embeddings.embed_documents(texts)
        dense_used_time = time.time() - dense_start_time

        # 7.3 生成 Sparse Vectors (BGE-M3)
        # compute_source_embedding=False 表示只生成 content 的 embedding
        # BGEM3EmbeddingFunction 的 encode_documents 返回结果通常包含 'dense', 'sparse', 'colbert_vecs'
        # 我们这里直接调用它的底层 sparse 编码逻辑或者通过 wrapper 调用
        # 注意：pymilvus 自带的 BGEM3EmbeddingFunction `encode_documents` 返回的是 generator 或 list

        # 获取原始 CSR 矩阵
        sparse_start_time = time.time()
        text_csr = sparse_ef.encode_documents(texts)["sparse"]
        title_csr = sparse_ef.encode_documents(titles)["sparse"]
        sparse_used_time = time.time() - sparse_start_time

        # 将 CSR 矩阵转换为 list of dicts
        text_sparse_list = csr_to_milvus_format(text_csr)
        title_sparse_list = csr_to_milvus_format(title_csr)

        # 7.4 组装数据用于 Collection.insert
        # Pymilvus insert 需要 list of list (按列) 或者 list of dict (按行)
        # 这里构造 list of dict 更清晰
        insert_data = []
        for j, doc in enumerate(encoded_batch):
            meta = doc.metadata

            entry_urls_list = list(meta.get("anchors", []))
            related_links_json = meta.get("related_links", [])

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
                "nav_see_also": meta.get("nav_see_also"),
                # Graph-RAG 字段
                "entry_urls": entry_urls_list,
                "related_links": related_links_json,
                # 递归摘要字段
                "summary": "",
                "summary_vector": text_dense_vectors[j],    # 暂时使用 text vector 占位
            }
            insert_data.append(entry)

        # 7.5 执行插入
        try:
            insert_start_time = time.time()
            # 插入静态分区
            collection.insert(insert_data, partition_name=STATIC_PARTITION_NAME)
            processed_count += len(batch)
            insert_used_time = time.time() - insert_start_time
            batch_used_time = time.time() - batch_start_time
            time_usage_str = f"总耗时 {batch_used_time:.2f}s | 稠密耗时 {dense_used_time:.2f}s | 稀疏耗时 {sparse_used_time:.2f}s | 插入耗时 {insert_used_time:.2f}s "
            print(
                f"\r[Batch {i + 1}] 写入 {len(batch)} 条 | {time_usage_str} | 进度: {processed_count}/{total}",
                end="", flush=True)
        except Exception as e:
            print(f"\n[Error] 批次写入失败: {e}")
            traceback.print_exc()
            if insert_data:
                print(f"Data Keys: {insert_data[0].keys()}")
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
    collection.create_index("summary_vector", index_params_dense)
    print("稠密索引 (vector) 创建完成。")

    # 8.2 稀疏向量索引 (Text、Title)
    index_params_sparse = {
        "metric_type": "IP",  # 稀疏向量通常使用 IP (Inner Product)
        "index_type": "SPARSE_INVERTED_INDEX",  # Milvus 稀疏倒排索引
        "params": {"drop_ratio_build": 0.2}  # 过滤掉贡献小的维度以加速
    }
    collection.create_index("sparse_vector", index_params_sparse)
    collection.create_index("title_sparse", index_params_sparse)
    print("稀疏索引创建完成。")

    # 9. 加载集合以供查询
    collection.load()
    print("[完成] 知识库构建完成，集合已加载！")


if __name__ == "__main__":
    build_knowledge_base()