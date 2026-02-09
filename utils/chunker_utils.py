import re
from typing import List, Any, Dict, Tuple

from graphviz import Digraph
from langchain_core.documents import Document


# 1. 简单的 HLINK 模式 (用于标题/入口)
# 匹配: [HLINK: https://...]
SIMPLE_HLINK_PATTERN = re.compile(r'\[HLINK:\s*(.*?)]')

# 2. 复杂的 ANCHOR+HLINK 模式 (用于正文/出口)
# 匹配: [ANCHOR: 文本, HLINK: https://...]
COMPLEX_HLINK_PATTERN =re.compile(r'\\?\[ANCHOR:\s*(.*?),\s*HLINK:\s*(.*?)\\?]')


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


def extract_entry_hlink(text: str) -> Tuple[str, List[str]]:
    """
    【入口提取】专门用于处理标题行。
    输入: "## Title [HLINK: url]"
    输出: ("## Title", ["url"])
    """
    if not text:
        return "", []

    links = SIMPLE_HLINK_PATTERN.findall(text)
    # 直接移除标记，保留标题文字
    clean_text = SIMPLE_HLINK_PATTERN.sub('', text).strip()

    return clean_text, links


def extract_exit_hlink(text: str) -> Tuple[str, Dict[str, List[str]]]:
    """
    【出口提取】专门用于处理正文内容的超链接。
    输入: "Click [ANCHOR: here, HLINK: url]..."
    输出: ("Click here...", {"here": ["url"]})
    """
    if not text:
        return "", {}

    outlinks_map = {}

    def replacer(match):
        # group(1): Anchor Text
        # group(2): URL
        anchor_text = match.group(1).strip()
        url = match.group(2).strip()

        # 将 URL 记录到字典列表
        if anchor_text not in outlinks_map:
            outlinks_map[anchor_text] = []
        # 避免同一个位置重复添加相同的 URL (虽然罕见)
        if url not in outlinks_map[anchor_text]:
            outlinks_map[anchor_text].append(url)

        # 还原正文：只保留 Anchor Text，去掉 HLINK 标记。前后加空格防止粘连
        return f" {anchor_text} "

    clean_text = COMPLEX_HLINK_PATTERN.sub(replacer, text)
    # 清理多余空格 (把 "  " 变 " ")
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text, outlinks_map


def merge_outlinks(
        target: Dict[str, List[str]],
        source: Dict[str, List[str]]
) -> None:
    """
    原地合并 outlinks 字典 (用于 TreeParser)
    target: 被更新的字典
    source: 新增数据的字典
    """
    for text, urls in source.items():
        if text not in target:
            target[text] = []
        # 合并 URL 列表并去重
        existing = set(target[text])
        for u in urls:
            if u not in existing:
                target[text].append(u)
                existing.add(u)


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
