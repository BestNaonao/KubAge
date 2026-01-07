import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from dotenv import load_dotenv, find_dotenv
from pymilvus import Collection, connections
from transformers import AutoConfig


# ==================== 环境变量加载 ====================
load_dotenv(find_dotenv())
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')
MILVUS_USER = os.getenv('MILVUS_USER')
MILVUS_PASSWORD = os.getenv('MILVUS_ROOT_PASSWORD')

def main_test(embedding_model_path, collection_name):
    # 测试加载QWen的配置
    config = AutoConfig.from_pretrained(embedding_model_path, trust_remote_code=True)
    print(config.model_type)  # 如输出 "qwen3"

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

    # 添加文本（会触发集合创建 + 索引构建）
    texts = ["我的姓名叫BestNaonao", "我最喜欢摸鱼了！"]
    vector_store.add_texts(texts)

    # 查询
    results = vector_store.similarity_search("名字", k=2)
    print(results)
    results2 = vector_store.similarity_search_with_score("姓名", k=2)

    print("\n=== 查询结果（带分数） ===")
    for i, (doc, score) in enumerate(results2, 1):
        print(f"结果 {i}:")
        print(f"  内容: {doc.page_content}")
        print(f"  相似度: {score:.4f}")      # 注意：Milvus COSINE 下 score 是距离（越小越相似）？
        print(f"  元数据: {doc.metadata}")

    results3 = vector_store.similarity_search_with_score("爱好", k=2)

    print("=== 查询结果（带分数） ===")
    for i, (doc, score) in enumerate(results3, 1):
        print(f"结果 {i}:")
        print(f"  内容: {doc.page_content}")
        print(f"  相似度: {score:.4f}")      # 注意：Milvus COSINE 下 score 是距离（越小越相似）？
        print(f"  元数据: {doc.metadata}")

    # 清空这个表的数据
    print("=== 清空数据库 ===")
    connections.connect(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        user=MILVUS_USER,
        password=MILVUS_PASSWORD,
        # 可选：指定 alias，但默认 "default" 即可
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
    my_collection = "my_collection"
    main_test(embedding, my_collection)