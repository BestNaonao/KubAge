import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from dotenv import load_dotenv, find_dotenv
from pymilvus import Collection, connections
from transformers import AutoConfig
from utils import encode_document_for_milvus, decode_document_from_milvus
from langchain_core.documents import Document


# ==================== 环境变量加载 ====================
load_dotenv(find_dotenv())
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')
MILVUS_USER = os.getenv('MILVUS_USER')
MILVUS_PASSWORD = os.getenv('MILVUS_ROOT_PASSWORD')


def main_test(embedding_model_path, collection_name):
    # 初始化嵌入模型
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={
            "device": "cuda",               # 如果无 GPU，改为 "cpu"
            "trust_remote_code": True,      # Qwen 必须开启
        },
        encode_kwargs={
            "normalize_embeddings": True    # Qwen 推荐开启，用于 COSINE 相似度
        }
    )

    # 连接到 Milvus
    vector_store = Milvus(
        embedding_function=embedding_model,
        connection_args={
            "host": MILVUS_HOST,
            "port": MILVUS_PORT,
            "user": MILVUS_USER,
            "password": MILVUS_PASSWORD,
        },
        collection_name=collection_name,
        # 注意：index_params 在 add_texts 首次调用时才会生效（如果集合不存在）
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200}
        },
        # 搜索参数
        search_params={
            "metric_type": "COSINE",
            "params": {"ef": 64}  # 搜索时的 ef 值
        }
    )

    # 创建带复杂元数据的文档
    docs = [
        Document(
            page_content="我的姓名叫BestNaonao",
            metadata={
                "source": "test.md",
                "parent_id": None,
                "child_ids": ["child1", "child2"],
                "node_type": "leaf",  # 字符串格式
                "level": 1,
                "title": "test_title",
                "token_count": 5,
                "left_sibling": None,
                "right_sibling": None,
                "from_split": False,
                "merged": False
            }
        ),
        Document(
            page_content="我最喜欢摸鱼了！",
            metadata={
                "source": "test.md",
                "parent_id": None,
                "child_ids": ["child3"],
                "node_type": "section",  # 字符串格式
                "level": 2,
                "title": "hobby",
                "token_count": 6,
                "left_sibling": None,
                "right_sibling": None,
                "from_split": True,
                "merged": False
            }
        )
    ]

    # 编码文档以存入Milvus
    encoded_docs = [encode_document_for_milvus(doc) for doc in docs]
    
    print("编码后的文档元数据:")
    for i, doc in enumerate(encoded_docs):
        print(f"文档 {i+1}: {doc.metadata}")
    
    # 添加编码后的文档到Milvus
    vector_store.add_documents(documents=encoded_docs)

    # 查询
    results = vector_store.similarity_search("名字", k=2)
    print("\n从Milvus检索到的结果:")
    for i, doc in enumerate(results):
        print(f"原始文档 {i+1} 元数据: {doc.metadata}")

    # 解码检索到的文档
    decoded_results = [decode_document_from_milvus(doc) for doc in results]
    print("\n解码后的文档元数据:")
    for i, doc in enumerate(decoded_results):
        print(f"解码文档 {i+1} 元数据: {doc.metadata}")

    results2 = vector_store.similarity_search_with_score("姓名", k=2)

    print("\n=== 查询结果（带分数） ===")
    for i, (doc, score) in enumerate(results2, 1):
        decoded_doc = decode_document_from_milvus(doc)
        print(f"结果 {i}:")
        print(f"  内容: {decoded_doc.page_content}")
        print(f"  相似度: {score:.4f}")
        print(f"  元数据: {decoded_doc.metadata}")

    results3 = vector_store.similarity_search_with_score("爱好", k=2)

    print("=== 查询结果（带分数） ===")
    for i, (doc, score) in enumerate(results3, 1):
        decoded_doc = decode_document_from_milvus(doc)
        print(f"结果 {i}:")
        print(f"  内容: {decoded_doc.page_content}")
        print(f"  相似度: {score:.4f}")
        print(f"  元数据: {decoded_doc.metadata}")

    # 清空这个表的数据
    print("=== 清空数据库 ===")
    connections.connect(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        user=MILVUS_USER,
        password=MILVUS_PASSWORD,
    )
    collection = Collection(collection_name)
    collection.load()
    result = collection.query(
        expr="pk >= 0",  # 查询所有记录
        output_fields=["pk"]  # pk 是主键字段名
    )
    if result:
        # 提取所有 ID
        ids_to_delete = [item["pk"] for item in result]
        # 批量删除
        collection.delete(f"pk in {ids_to_delete}")
        print(f"✅ 已删除 {len(ids_to_delete)} 条记录")
    else:
        print("⚠️ Collection 为空")
    # 刷新
    collection.flush()


if __name__ == "__main__":
    embedding = "../models/Qwen/Qwen3-Embedding-0.6B"   # 相对路径：从当前脚本所在目录出发
    my_collection = "test_encoded_collection"
    main_test(embedding, my_collection)