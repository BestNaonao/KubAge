import os

from dotenv import load_dotenv, find_dotenv
from pymilvus import Collection, connections

# ==================== 环境变量加载 ====================
load_dotenv(find_dotenv())
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')
MILVUS_USER = os.getenv('MILVUS_USER')
MILVUS_PASSWORD = os.getenv('MILVUS_ROOT_PASSWORD')


def content_test(collection_name):
    connections.connect(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        user=MILVUS_USER,
        password=MILVUS_PASSWORD,
    )
    # 获取底层 Milvus collection 对象（LangChain 的 Milvus 封装可能不支持直接 query）
    # 所以我们直接使用 pymilvus 来查询
    collection = Collection(name=collection_name)

    # 确保加载了（如果未加载）
    collection.load()

    def condition_query(condition: str):
        results = collection.query(
            expr=condition,
            output_fields=["pk", "text", "right_sibling", "child_ids", "source", "token_count", "title"],
            consistency_level="Strong"
        )
        print(f"共 {len(results)} 条")
        for res in results[:10]:
            print(res)
        return results

    print("=========“接下来”和“另请参见”导航章节测试")
    condition_query("text like '%# 接下来%'")
    condition_query("text like '%# 另请参见%'")

    print("=========API块测试")

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


if __name__ == "__main__":
    content_test("knowledge_base_v1")