import re
from typing import List, Any

from graphviz import Digraph
from langchain_core.documents import Document


HLINK_PATTERN = re.compile(r'\[HLINK:\s*(.*?)]')


def extract_blocks(content: str) -> tuple[str, list[Any]]:
    """提取提醒框和代码块，返回替换后的内容和块列表"""
    block_pattern = re.compile(
        r'''
        (?P<code_block>```.*?```)|             # 代码块
        (?P<alert_block>:::.*?:::)             # 提醒块
        ''',
        re.DOTALL | re.VERBOSE
    )
    blocks = []

    def replacer(match):
        block = match.group()
        blocks.append(block)
        return f"@@BLOCK_{len(blocks) - 1}@@"

    content = block_pattern.sub(replacer, content)
    return content, blocks


def restore_blocks(text: str, blocks: List[str]) -> str:
    """将占位符替换回原始块内容"""
    def replacer(match):
        idx = int(match.group(1))
        return blocks[idx]

    return re.sub(r"@@BLOCK_(\d+)@@", replacer, text)


def extract_hlink_from_text(text: str) -> tuple[str, list[str]]:
    """
    从文本中提取 [HLINK: url] 标记。
    返回: (清洗后的文本, 提取出的URL列表)
    """
    if not text:
        return "", []

    links = HLINK_PATTERN.findall(text)
    # 将 [HLINK: url] 替换为空字符串，避免粘连，或者直接删除
    # 这里使用 strip() 去除标题可能留下的多余空格
    clean_text = HLINK_PATTERN.sub('', text).strip()

    return clean_text, links


def get_node_label(doc: Document, length: int = 10) -> str:
    title = doc.metadata["title"]
    content = doc.page_content.strip().replace("\n", " ").replace("\"", "'")
    if len(content) > 3 * length:
        label = f"{content[:length]}...{content[-length:]}"
    else:
        label = content
    node_type = doc.metadata.get("node_type", "")
    level = doc.metadata.get("level", "")
    return f"{title}\n{label}\n[{node_type}][L{level}]"  # ✅ 展示层级


def visualize_document_tree(documents: list[Document], output_file="doc_tree", fontname="Microsoft YaHei", show_siblings=False):
    dot = Digraph(comment="Document Tree", format="png")
    dot.attr(fontname=fontname)
    dot.node_attr.update(fontname=fontname)
    dot.edge_attr.update(fontname=fontname)

    for doc in documents:
        node_id = doc.id
        label = get_node_label(doc, 20)
        dot.node(node_id, label)

    for doc in documents:
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            dot.edge(parent_id, doc.id, label="child")
        left_sibling_id = doc.metadata.get("left_sibling")
        if left_sibling_id and show_siblings:
            dot.edge(doc.id, left_sibling_id, label="left")
        right_sibling_id = doc.metadata.get("right_sibling")
        if right_sibling_id and show_siblings:
            dot.edge(doc.id, right_sibling_id, label="right")

    dot.render(output_file, view=True)
