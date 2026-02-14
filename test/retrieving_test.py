from retriever import MilvusHybridRetriever
from utils import get_dense_embed_model, get_sparse_embed_model
from utils.milvus_adapter import connect_milvus_by_env


def main():
    # 1. 连接 Milvus
    connect_milvus_by_env()

    # 2. 加载模型 (这里需要加载和 Build 阶段一样的模型)
    # 注意显存控制，如果显存不够，可以把 dense 模型放到 CPU 或者按需加载
    print("Loading Dense Model...")
    dense_ef = get_dense_embed_model("../models/Qwen/Qwen3-Embedding-0.6B")

    print("Loading Sparse Model...")
    sparse_ef = get_sparse_embed_model("BAAI/bge-m3")

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