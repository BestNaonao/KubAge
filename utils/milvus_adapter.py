"""
Milvus 适配器：处理 Document 对象与 Milvus 数据格式之间的双向转换
包括元数据序列化、反序列化、CSR 矩阵转换以及结果解析
"""
import os
from typing import Dict, Any

from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from pymilvus import Hit, connections

from .MarkdownTreeParser import NodeType

# 定义 Milvus 中存储的所有标量字段 (用于检索时 output_fields，避免拉取 vector 字段)
SCALAR_FIELDS = [
    "pk", "text", "source", "title",
    "parent_id", "child_ids", "node_type", "level",
    "token_count", "left_sibling", "right_sibling",
    "from_split", "merged", "nav_next_step", "nav_see_also",
    "entry_urls", "related_links", "summary",
]

HYBRID_SEARCH_FIELDS = SCALAR_FIELDS + ["summary_vector",]


def connect_milvus_by_env():
    # 加载环境变量
    load_dotenv(find_dotenv())
    host = os.getenv('MILVUS_HOST', 'localhost')
    port = os.getenv('MILVUS_PORT', '19530')
    user = os.getenv('MILVUS_USER', 'root')
    password = os.getenv('MILVUS_ROOT_PASSWORD', 'Milvus')

    # 连接 Milvus
    print(f"正在连接 Milvus ({host}:{port})...")
    connections.connect(alias="default", host=host, port=port, user=user, password=password)


def encode_metadata_for_milvus(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    将Document的元数据编码为Milvus支持的格式
    - 列表字段转换为JSON字符串
    - 枚举字段转换为字符串值
    """
    encoded_metadata = {}
    
    for key, value in metadata.items():
        if isinstance(value, NodeType):
            # 将枚举转换为字符串值
            encoded_metadata[key] = value.value
        else:
            # 其他类型保留
            encoded_metadata[key] = value
    
    return encoded_metadata


def decode_metadata_from_milvus(encoded_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    从Milvus读取后将元数据解码回原始格式
    - JSON字符串转换回列表
    - 字符串值转换回枚举
    """
    decoded_metadata = {}
    
    for key, value in encoded_metadata.items():
        if key == 'node_type' and isinstance(value, str):
            # 将node_type字符串转换回枚举
            try:
                decoded_metadata[key] = NodeType(value)
            except ValueError:
                decoded_metadata[key] = NodeType.LEAF  # 默认值
        else:
            # 其他字段保持不变
            decoded_metadata[key] = value
    
    return decoded_metadata


def encode_document_for_milvus(doc: Document) -> Document:
    """编码单个Document对象的元数据以存入Milvus"""
    encoded_doc = Document(
        page_content=doc.page_content,
        metadata=encode_metadata_for_milvus(doc.metadata)
    )
    # 保留自定义的id属性
    if hasattr(doc, 'id'):
        setattr(encoded_doc, 'id', doc.id)
    return encoded_doc


def decode_document_from_milvus(encoded_doc: Document) -> Document:
    """从Milvus读取后解码Document对象的元数据"""
    decoded_doc = Document(
        page_content=encoded_doc.page_content,
        metadata=decode_metadata_from_milvus(encoded_doc.metadata)
    )
    # 保留自定义的id属性
    if hasattr(encoded_doc, 'id'):
        setattr(decoded_doc, 'id', encoded_doc.id)
    return decoded_doc

def decode_hit_to_document(hit: Hit, content_field: str = "text") -> Document:
    """
    将 Milvus Search 返回的 Hit 对象直接转换为 LangChain Document

    Args:
        hit: pymilvus 的 Hit 对象 (SearchResult 的元素)
        content_field: 存储文本内容的字段名
    """
    # 1. 获取实体内容 (类似于字典)
    entity = hit['entity']

    # 2. 提取正文，如果未指定 output_fields，可能拿不到 text
    page_content = entity.get(content_field, "")

    # 3. 提取所有元数据字段
    raw_metadata = {}
    for field in entity.keys():
        # 只放入所需字段
        if field != content_field and field in HYBRID_SEARCH_FIELDS:
            raw_metadata[field] = entity.get(field)

    # 4. 注入 Milvus 特有的信息 (Primary Key 和 Score)
    raw_metadata["pk"] = hit.id
    raw_metadata["score"] = hit.score

    # 5. 解码特殊格式 (JSON -> List, Str -> Enum)
    final_metadata = decode_metadata_from_milvus(raw_metadata)

    # 6. 构建 Document
    doc = Document(page_content=page_content, metadata=final_metadata)

    # 将 pk 也赋值给 doc.id，方便 LangChain 后续使用
    setattr(doc, 'id', str(hit.id))

    return doc

def decode_query_result_to_document(
    row: dict,
    content_field: str = "text"
) -> Document:
    """
    将 Milvus collection.query 返回的 dict 转换为 LangChain Document
    """
    metadata = {}
    for k, v in row.items():
        if k != content_field and k in HYBRID_SEARCH_FIELDS:
            metadata[k] = v

    final_metadata = decode_metadata_from_milvus(metadata)
    doc = Document(page_content=row.get(content_field, ""), metadata=final_metadata)
    setattr(doc, 'id', str(row['pk']))

    return doc

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