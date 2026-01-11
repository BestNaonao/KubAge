import os
import re

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

    # 分页查询所有数据（避免一次性拉取太多）
    page_size = 5000
    offset = 0
    all_entities = []

    while True:
        try:
            res = collection.query(
                expr="",  # 空 expr 表示返回所有
                output_fields=["pk", "text", "right_sibling", "child_ids", "source", "token_count"],
                offset=offset,
                limit=page_size,
                consistency_level="Strong"
            )
            if not res:
                break
            all_entities.extend(res)
            if len(res) < page_size:
                break
            offset += page_size
        except Exception as e:
            print(f"Query failed at offset {offset}: {e}")
            break

    def filter_by_reg(reg):
        pattern = re.compile(reg)
        return [ent for ent in all_entities if pattern.search(ent.get("text", ""))]

    # 存储结果
    matched_records = filter_by_reg(r'#{1,}\s+接下来')
    all_empty = True
    for entity in matched_records:
        right_sibling = entity.get("right_sibling")
        child_ids = entity.get("child_ids")

        rs_empty = (
                right_sibling is None or
                right_sibling == "" or
                (isinstance(right_sibling, str) and right_sibling.lower() == "null")
        )

        ci_empty = (
                child_ids is None or
                child_ids == [] or
                child_ids == "" or
                (isinstance(child_ids, str) and child_ids.strip() in ("", "[]", "null"))
        )

        print(f"ID: {entity['pk']}")
        print(f"Right sibling empty: {rs_empty} (raw: {right_sibling})")
        print(f"Child IDs empty: {ci_empty} (raw: {child_ids})")
        print("-" * 60)
        if not rs_empty or not ci_empty:
            all_empty = False

    # 输出结果
    print(f"Found {len(matched_records)} records matching the pattern.")
    print(f"All 'Next' records has no right sibling and no children: {all_empty}")
    print("-" * 60)
    # 检查接下来、另请参见是否出现在同一源文档中
    matched_next = filter_by_reg(r'#{1,}\s+接下来')
    matched_refer = filter_by_reg(r'#{1,}\s+另请参见')
    next_source_set = set()
    refer_source_set = set()
    for source in [entity.get("source") for entity in matched_next]:
        if not source in next_source_set:
            next_source_set.add(source)
        else:
            print(f"Duplicate Next entity found: {source}")
    print("-" * 60 + str(len(matched_next)))
    for source in [entity.get("source") for entity in matched_refer]:
        if not source in refer_source_set:
            refer_source_set.add(source)
        else:
            print(f"Duplicate Reference entity found: {source}")
    print("-" * 60 + str(len(matched_refer)))
    for next_source in next_source_set:
        if next_source in refer_source_set:
            print(f"Duplicate Next and Reference entity found: {next_source}")


if __name__ == "__main__":
    content_test("knowledge_base_v1")