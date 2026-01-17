import uuid
from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus

from . import decode_document_from_milvus
from .milvus_adapter import decode_metadata_from_milvus


def validate_node_id(node_id: str):
    try:
        if uuid.UUID(node_id).version != 5:
            raise ValueError("Invalid node id.")
    except ValueError as e:
        raise ValueError("Invalid node id.")

def get_full_node_content(milvus: Milvus, node_id: str) -> str:
    """
    获取指定节点的全部内容，按照先序遍历拼接所有节点的内容
    
    Args:
        milvus: Milvus向量数据库实例
        node_id: 起始节点Document对象的id(pk)
    
    Returns:
        str: 拼接后的完整内容，包含当前节点及其所有子节点的内容
    """
    # 使用similarity_search方法结合expr参数根据pk字段查询文档
    results = milvus.search_by_metadata(expr=f"pk == '{node_id}'", limit=1)
    if not results:
        return ""
    
    node = results[0]
    
    # 解码元数据以恢复child_ids列表等字段
    node.metadata = decode_metadata_from_milvus(node.metadata)
    
    # 先序遍历：先访问当前节点
    contents = [node.page_content.strip()]
    
    # 递归访问所有子节点
    child_ids = node.metadata.get("child_ids")
    if child_ids is not None and len(child_ids) > 0:
        # 递归获取每个子节点的内容
        for child_id in child_ids:
            # 递归获取子节点的完整内容
            child_content = get_full_node_content(milvus, child_id)
            contents.append(child_content)
    
    return "\n\n".join(contents)

def query_nodes_with_similarity(milvus: Milvus, query: str, k: int, expr: str) -> List[Document]:
    """
    使用相似度搜索查询节点

    Args:
        milvus: Milvus向量数据库实例
        query: 查询文本
        k: 返回的节点数量
        expr: 查询条件
    
    Returns:
        List[Document]: 查询结果
    """
    results = milvus.similarity_search(
        query, k=k,
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        expr=expr
    )
    nodes = [decode_document_from_milvus(result) for result in results]
    return nodes

def get_root_node(milvus: Milvus, node: Document) -> Document | None:
    if node.metadata["node_type"] == "ROOT":
        return node
    results = milvus.search_by_metadata(expr=f"source == '{node.metadata["source"]}' and node_type == 'root'", limit=1)
    if not results:
        return None

    return decode_document_from_milvus(results[0])