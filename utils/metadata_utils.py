"""处理Document对象元数据的序列化和反序列化工具"""
import json
from typing import Dict, Any, List
from langchain_core.documents import Document
from .MarkdownTreeParser import NodeType


def encode_metadata_for_milvus(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    将Document的元数据编码为Milvus支持的格式
    - 列表字段转换为JSON字符串
    - 枚举字段转换为字符串值
    """
    encoded_metadata = {}
    
    for key, value in metadata.items():
        if isinstance(value, list):
            # 将列表转换为JSON字符串
            encoded_metadata[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, NodeType):
            # 将枚举转换为字符串值
            encoded_metadata[key] = value.value
        elif isinstance(value, bool) or isinstance(value, (int, float)) or isinstance(value, str):
            # 基本类型直接保留
            encoded_metadata[key] = value
        elif value is None:
            # None值直接保留
            encoded_metadata[key] = value
        else:
            # 其他类型也尝试转换为字符串
            encoded_metadata[key] = str(value)
    
    return encoded_metadata


def decode_metadata_from_milvus(encoded_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    从Milvus读取后将元数据解码回原始格式
    - JSON字符串转换回列表
    - 字符串值转换回枚举
    """
    decoded_metadata = {}
    
    for key, value in encoded_metadata.items():
        if key == 'child_ids' and isinstance(value, str):
            # 将child_ids的JSON字符串转换回列表
            try:
                decoded_metadata[key] = json.loads(value)
            except json.JSONDecodeError:
                decoded_metadata[key] = []
        elif key == 'node_type' and isinstance(value, str):
            # 将node_type字符串转换回枚举
            try:
                decoded_metadata[key] = NodeType(value)
            except ValueError:
                decoded_metadata[key] = NodeType.LEAF  # 默认值
        elif key in ['child_ids', 'from_split', 'merged'] and isinstance(value, str):
            # 尝试解析可能的JSON数组
            try:
                decoded_metadata[key] = json.loads(value)
            except json.JSONDecodeError:
                # 如果不是JSON格式，保持原值
                decoded_metadata[key] = value
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