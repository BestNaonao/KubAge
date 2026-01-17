import os

import torch
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from retriever import MilvusHybridRetriever


def main():
    # 加载环境变量
    load_dotenv(find_dotenv())
    host = os.getenv('MILVUS_HOST', 'localhost')
    port = os.getenv('MILVUS_PORT', '19530')
    user = os.getenv('MILVUS_USER', 'root')
    password = os.getenv('MILVUS_ROOT_PASSWORD', 'Milvus')

    # 1. 连接 Milvus
    print(f"正在连接 Milvus ({host}:{port})...")
    connections.connect(alias="default", host=host, port=port, user=user, password=password)

    # 2. 加载模型 (这里需要加载和 Build 阶段一样的模型)
    # 注意显存控制，如果显存不够，可以把 dense 模型放到 CPU 或者按需加载
    print("Loading Dense Model...")
    dense_ef = HuggingFaceEmbeddings(
        model_name="../models/Qwen/Qwen3-Embedding-0.6B",  # 替换你的路径
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    print("Loading Sparse Model...")
    sparse_ef = BGEM3EmbeddingFunction(
        model_name="BAAI/bge-m3",  # 替换你的路径
        use_fp16=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 3. 初始化 Retriever
    retriever = MilvusHybridRetriever(
        collection_name="knowledge_base_v2",
        dense_embedding_func=dense_ef,
        sparse_embedding_func=sparse_ef,
        top_k=5
    )

    # 4. 测试查询
    query = "Pod 处于 Pending 状态怎么排查？"
    print(f"\nQuery: {query}")
    print("-" * 50)

    docs = retriever.invoke(query)
    print(docs[0].metadata.keys())

    for i, doc in enumerate(docs):
        print(f"[{i + 1}] Score: {doc.metadata['score']:.4f}")
        print(f"Title: {doc.metadata['title']}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content Preview: {doc.page_content[:100]}...")
        print("-" * 30)


if __name__ == "__main__":
    main()