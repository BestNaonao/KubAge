from langchain_core.documents import Document
from langchain_milvus import Milvus
from .metadata_utils import decode_metadata_from_milvus


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
    decoded_metadata = decode_metadata_from_milvus(node.metadata)
    node.metadata = decoded_metadata
    
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
