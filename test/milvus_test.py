import os
import random

from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import Collection, connections

from utils.rag_utils import get_full_node_content

# ==================== 环境变量加载 ====================
load_dotenv(find_dotenv())
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')
MILVUS_USER = os.getenv('MILVUS_USER')
MILVUS_PASSWORD = os.getenv('MILVUS_ROOT_PASSWORD')

def milvus_store(embedding_model, collection_name, index_params=None, search_params=None) -> Milvus:
    return Milvus(
        embedding_function=embedding_model,
        connection_args={
            "host": MILVUS_HOST,
            "port": MILVUS_PORT,
            "user": MILVUS_USER,
            "password": MILVUS_PASSWORD,
        },
        collection_name=collection_name,
        index_params=index_params,
        search_params=search_params,
    )

def basic_test(embedding_model, collection_name):
    # 连接到 Milvus
    vector_store = milvus_store(
        embedding_model,
        collection_name,
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

def kb_test(milvus: Milvus):

    def condition_query(expr: str):
        docs = milvus.search_by_metadata(
            expr=expr,
            limit=500
        )
        print(f"共 {len(docs)} 条")
        for doc in docs[:10]:
            print(doc.metadata['title'])
        return docs

    print("========= “接下来”和“另请参见”导航章节测试 =========")
    condition_query("text like '%# 接下来%'")
    condition_query("text like '%# 另请参见%'")

    print("========= API块测试 =========")

    condition_1 = "(text like '%# HTTP%')"
    condition_2 = "(text like '%# 参数%' or text like '%# Parameter%')"
    condition_3 = "(text like '%# 响应%' or text like '%# Response%')"

    condition_4 = "(text like '%**HTTP%')"
    condition_5 = "(text like '%**参数**%' or text like '%**Parameter%')"
    condition_6 = "(text like '%**响应**%' or text like '%**Response%')"

    condition_7 = "(title like '文档_参考_Kubernetes API%')"

    def satisfy_2_at_least(c1, c2, c3):
        return f"(({c1} and {c2}) or ({c2} and {c3}) or ({c3} and {c1}))"

    satisfy_1_2_3_al2 = satisfy_2_at_least(condition_1, condition_2, condition_3)
    satisfy_4_5_6_al2 = satisfy_2_at_least(condition_4, condition_5, condition_6)

    def satisfy_only_one(c1, c2, c3):
        return f"(({c1} and !{c2} and !{c3}) or (!{c1} and {c2} and !{c3}) or (!{c1} and !{c2} and {c3}))"

    print("---------至少满足条件1、2、3中的两个")
    condition_query(f"{satisfy_1_2_3_al2}")
    print("---------只满足条件1、2、3中的一个，并且满足条件7")
    condition_query(f"{satisfy_only_one(condition_1, condition_2, condition_3)} and {condition_7}")
    print("---------至少满足条件4、5、6中的两个")
    condition_query(f"{satisfy_4_5_6_al2}")
    print("---------至少满足条件4、5、6中的两个，并且不满足条件7")
    condition_query(f"{satisfy_4_5_6_al2} and !{condition_7}")
    print("---------只满足条件4、5、6中的一个，并且满足条件7")
    condition_query(f"{satisfy_only_one(condition_4, condition_5, condition_6)} and {condition_7}")
    print("---------只满足条件4、5、6中的一个，并且不满足条件7")
    condition_query(f"{satisfy_only_one(condition_4, condition_5, condition_6)} and !{condition_7}")

def title_test(milvus: Milvus):
    print("========= 标题字段测试 =========")
    root_docs = milvus.search_by_metadata(
        expr="node_type == 'root'",
        limit = 1000
    )
    invalid_root_titles = []
    for doc in root_docs:
        title_from_source = doc.metadata['source'].split('/')[-1].split('.md')[0]
        if title_from_source != doc.metadata['title']:
            invalid_root_titles.append(title_from_source)
    print(f"标题非法的根节点标题 共有 {len(invalid_root_titles)} 个")
    for invalid_title in invalid_root_titles:
        print(invalid_title)

def nav_test(milvus: Milvus):
    print("========= 导航章节测试 =========")
    invalid_nav_docs = milvus.search_by_metadata(
        expr="node_type != 'root' and (nav_next_step != '' or nav_see_also != '')",
        limit = 1000
    )
    print(f"是否包含非根节点的导航内容: {len(invalid_nav_docs) != 0}")

def rag_utils_test(milvus: Milvus):
    print("========= RAG工具测试 =========")
    print(get_full_node_content(milvus, "9a0c26a5-decf-5184-b5fc-9a2e7fce0cd6"))

def hlink_cleanliness_test(milvus: Milvus):
    print("========= 超链接残留测试 =========")
    # 检测字段
    fields_to_check = ["text", "title", "nav_next_step", "nav_see_also"]
    # 检测模式
    patterns = ["HLINK", "ANCHOR"]

    total_dirty_docs = 0

    # 也可以检查 related_links 字段是否解析正确（非字符串）

    for field in fields_to_check:
        for pattern in patterns:
            # 构造模糊匹配表达式 (注意: Milvus 的 like 匹配大小写敏感)
            expr = f'{field} like "%{pattern}%"'

            try:
                # 使用底层 pymilvus collection 进行查询
                res = milvus.search_by_metadata(
                    expr=expr,
                    fields=["pk", "title"],
                    limit=5  # 仅取样展示
                )

                if res:
                    print(f"⚠️  [失败] 字段 '{field}' 中发现残留 '{pattern}' (示例):")
                    for doc in res:
                        print(f"    - PK: {doc.id} | Title: {doc.metadata.get('title', 'Unknown')}")
                    total_dirty_docs += len(res)
            except Exception as e:
                # 某些字段可能因为长度问题无法执行 like 查询，视情况忽略
                print(f"    [跳过] 字段 '{field}' 查询出错: {e}")

    if total_dirty_docs == 0:
        print("✅ 内容测试通过：未发现残留的 HLINK 或 ANCHOR 标记。")
    else:
        print(f"❌ 内容测试失败：发现潜在残留标记。")

def graph_traversal_test(milvus: Milvus):
    print("========= 图谱跳转测试 (Random Walk) =========")

    # 1. 获取一批候选文档 (related_links 不为空)
    # 由于 Milvus 对 JSON 字段的空值查询支持有限，我们先拉取一批非根节点文档进行筛选
    try:
        candidates_res = milvus.search_by_metadata(
            expr='pk != ""',  # 获取所有文档（受限于 limit）
            fields=["pk", "title", "related_links"],
            limit=500
        )
    except Exception as e:
        print(f"❌ 查询文档失败: {e}")
        return

    # 2. 在内存中筛选出 related_links 有内容的文档
    valid_candidates = [
        doc for doc in candidates_res
        if doc.metadata.get("related_links") and len(doc.metadata.get("related_links")) > 0
    ]

    if len(valid_candidates) < 3:
        print(f"⚠️ 有效链接文档不足 3 个 (当前: {len(valid_candidates)})，无法执行测试。")
        return

    # 3. 随机选择 3 个起点
    start_docs = random.sample(valid_candidates, 3)

    for i, start_doc in enumerate(start_docs, 1):
        print(f"\n🔗 [路径 {i}]")
        current_doc = start_doc
        print(f"   🚩 起点: {current_doc.metadata.get('title')}")

        steps = 0
        max_steps = 5

        while steps < max_steps:
            # 获取当前文档的所有链接
            links = current_doc.metadata.get("related_links", [])

            # 筛选出内部链接 (type == 'internal' 且 pk 存在)
            internal_links = [
                l for l in links
                if l.get("type") == "internal" and l.get("pk")
            ]

            if not internal_links:
                print("      🛑 停止: 当前文档无内部链接")
                break

            # 随机选择一个链接进行跳跃
            chosen_link = random.choice(internal_links)
            target_pk = chosen_link['pk']
            anchor_text = chosen_link.get('text', 'unknown')

            # 查询目标文档
            next_docs = milvus.search_by_metadata(
                expr=f'pk == "{target_pk}"',
                fields=["pk", "title", "related_links"]
            )

            if not next_docs:
                print(f"      ⚠️ 错误: 链接指向的 PK {target_pk} 不存在 (死链)")
                break

            next_doc = next_docs[0]
            print(f"      ⬇️  (点击: '{anchor_text}')")
            print(f"   📍 跳跃至: {next_doc.metadata.get('title')}")

            current_doc = next_doc
            steps += 1

        if steps == max_steps:
            print("      🏁 达到最大跳跃次数")

def main():
    embedding_path = "../models/Qwen/Qwen3-Embedding-0.6B"   # 相对路径：从当前脚本所在目录出发
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_path,
        model_kwargs={
            "device": "cuda",  # 如果无 GPU，改为 "cpu"
            "trust_remote_code": True,  # Qwen 必须开启
        },
        encode_kwargs={
            "normalize_embeddings": True  # Qwen 推荐开启，用于 COSINE 相似度
        }
    )
    # basic_test(embedding_model, "my_collection")
    kb_store = milvus_store(embedding_model, "knowledge_base_v3")
    kb_test(kb_store)
    title_test(kb_store)
    nav_test(kb_store)
    rag_utils_test(kb_store)

    # 执行新测试
    hlink_cleanliness_test(kb_store)
    graph_traversal_test(kb_store)


if __name__ == '__main__':
    main()