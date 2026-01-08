import os
import re
import uuid
from enum import Enum
from typing import Dict, List, Any
from queue import PriorityQueue

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from transformers import PreTrainedTokenizerBase

from utils.chunker_utils import extract_blocks, restore_blocks


class NodeType(Enum):
    ROOT = "root"
    SECTION = "section"
    CONTAINER = "container"
    LEAF = "leaf"

class MarkdownTreeParser:
    """将Markdown文档解析为树状结构的文档块"""

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        tokenizer: PreTrainedTokenizerBase,
        min_chunk_size: int = 256,
        core_chunk_size: int = 1024,
        max_chunk_size: int = 4096
    ):
        """
        初始化Markdown树解析器

        参数:
            embeddings: HuggingFace嵌入模型
            tokenizer: 用于token计数的分词器
            min_chunk_size: 最小块大小（用于语义切分和合并）
            core_chunk_size: 核心块大小（理想块大小）
            max_chunk_size: 最大块大小（超过则进行语义切分）
        """
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.min_chunk_size = min_chunk_size
        self.core_chunk_size = core_chunk_size
        self.max_chunk_size = max_chunk_size
        self.semantic_splitter = SemanticChunker(
            embeddings,
            min_chunk_size=self.min_chunk_size,
            sentence_split_regex = r"(?<=[.?!。？！])\s+",
        )

    def count_tokens(self, content: str) -> int:
        """计算文本的token数量"""
        tokenized = self.tokenizer.tokenize(content)
        return len(tokenized)

    def parse_markdown_to_tree(self, file_path: str) -> List[Document]:
        """
        将Markdown文档解析为文档块树

        参数:
            file_path: Markdown文件的完整路径

        返回:
            文档块列表，每个文档块属性参考_create_document_node方法。
        """
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 分离文件头部和主体内容
        header_content = ''.join(lines[:4])
        body_content = ''.join(lines[4:])
        processed_content, extracted_blocks = extract_blocks(body_content)

        # 创建根节点
        file_name = os.path.basename(file_path).removesuffix('.md')
        root_doc = MarkdownTreeParser._create_document_node(
            content=processed_content.strip(),
            source=file_path,
            parent_id='',
            level=0,
            title=file_name,
            token_count=0,
            node_type=NodeType.ROOT
        )
        # 构建文档树
        id_map = {root_doc.id: root_doc}
        self._build_subtree(root_doc.id, id_map, extracted_blocks)
        self._optimize_tree_structure(id_map)   # 优化树结构（合并小节点）
        self._set_sibling_relations(id_map)  # 统一设置兄弟关系

        # 添加回文件头部
        if header_content:
            root_doc.page_content = header_content + root_doc.page_content
            root_doc.metadata["token_count"] = self.count_tokens(root_doc.page_content)

        return list(id_map.values())

    @staticmethod
    def _generate_node_id(title: str) -> str:
        """生成基于标题的唯一节点ID"""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, title))

    @staticmethod
    def _create_document_node(
            content: str,
            *,
            source: str,
            parent_id: str,
            level: int,
            title: str,
            token_count: int,
            node_type: NodeType = NodeType.LEAF,
            child_ids: List[str] = None,
            left_sibling: str = '',
            right_sibling: str = '',
            from_split: bool = False,
            merged: bool = False
    ) -> Document:
        """创建文档节点

        参数:
            content: 节点内容
            source: 源文件路径
            parent_id: 父节点ID
            level: 层级
            title: 标题
            token_count: token数量
            node_type: 节点类型（默认LEAF）
            child_ids: 子节点ID列表（默认空列表）
            left_sibling: 左兄弟节点ID（默认None）
            right_sibling: 右兄弟节点ID（默认None）
            from_split: 是否来自切分（默认False）
            merged: 是否合并节点（默认False）
        """
        metadata = {
            "source": source,
            "parent_id": parent_id,
            "child_ids": child_ids if child_ids else [],
            "node_type": node_type,
            "level": level,
            "title": title,
            "token_count": token_count,
            "left_sibling": left_sibling,
            "right_sibling": right_sibling,
            "from_split": from_split,
            "merged": merged
        }

        doc = Document(page_content=content, metadata=metadata)
        setattr(doc, 'id', MarkdownTreeParser._generate_node_id(title))
        return doc

    def _build_subtree(self, parent_id: str, id_map: Dict[str, Document], extracted_blocks: List[Any]) -> None:
        """
        递归构建文档子树

        参数:
            parent_id: 父节点ID
            id_map: 节点ID到文档节点的映射
            extracted_blocks: 提取的块内容（代码块、提醒框等）
        """
        parent_doc = id_map[parent_id]
        parent_level = parent_doc.metadata["level"]
        parent_title = parent_doc.metadata["title"]
        parent_content = parent_doc.page_content

        # 查找当前层级下的所有子标题
        heading_pattern = re.compile(
            rf"^({'#' * (parent_level + 1)})\s+(.*)",
            re.MULTILINE
        )
        matches = list(heading_pattern.finditer(parent_content))

        if len(matches) != 0:
            # 将父节点标记为section类型（如果是非根节点），截取父节点内容（第一个子标题之前）
            if parent_doc.metadata["node_type"] != NodeType.ROOT:
                parent_doc.metadata["node_type"] = NodeType.SECTION
            parent_doc.page_content = parent_content[:matches[0].start()].strip()

            title_counter: Dict[str, int] = {}  # 子标题的计数器
            # 处理每个子标题部分
            for idx, match in enumerate(matches):
                start_pos = match.start()
                end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(parent_content)
                section_title = match.group(2).strip()
                section_content = parent_content[start_pos:end_pos].strip()

                # 处理重复标题
                title_counter[section_title] = title_counter.get(section_title, 0) + 1
                count = title_counter[section_title]
                unique_title = f"{section_title}_{count}" if count > 1 else section_title
                full_title = f"{parent_title}_{unique_title}"

                # 创建子节点
                child_doc = MarkdownTreeParser._create_document_node(
                    content=section_content,
                    source=parent_doc.metadata["source"],
                    parent_id=parent_id,
                    level=parent_level + 1,
                    title=full_title,
                    token_count=0
                )
                node_id = child_doc.id

                # 更新节点映射和父节点关系
                id_map[node_id] = child_doc
                parent_doc.metadata["child_ids"].append(node_id)

                # 递归构建子树
                self._build_subtree(node_id, id_map, extracted_blocks)

        # 回溯阶段：统一恢复内容并计算token数量
        # 1. 恢复块内容
        restored_content = restore_blocks(parent_doc.page_content, extracted_blocks)
        # 2. 计算token数量
        token_count = self.count_tokens(restored_content)
        # 3. 如果是叶子节点且超过限制，则切分
        if parent_doc.metadata["node_type"] == NodeType.LEAF and token_count > self.max_chunk_size:
            # 使用原始占位符内容进行切分（避免重复提取块内容）
            self._split_large_leaf_node(parent_doc, id_map, extracted_blocks)
        else:
            parent_doc.page_content = restored_content
            parent_doc.metadata["token_count"] = token_count

    def _split_by_patterns(self, original_content: str, patterns: List[str], extracted_blocks: List[Any]) -> List[
        Dict[str, Any]]:
        """
        根据给定的正则表达式列表进行切分与合并

        参数:
            original_content: 原始内容
            patterns: 正则表达式列表，每个模式应该包含捕获组
            extracted_blocks: 提取的块内容

        返回:
            切分合并后的块列表，每个元素包含：
            - placeholder: 占位符内容（用于进一步处理）
            - content: 恢复后的实际内容
            - tokens: token数量
        """
        if not patterns:
            # 如果没有模式，直接返回整个内容
            restored_content = restore_blocks(original_content, extracted_blocks)
            token_count = self.count_tokens(restored_content)
            return [{
                'placeholder': original_content,
                'content': restored_content,
                'tokens': token_count
            }]

        # 使用第一个模式进行切分
        pattern = patterns[0]
        rough_chunks = re.split(pattern, original_content)

        # 合并分隔符到后续块
        combined_chunks = []
        skip_next = False
        for i, chunk in enumerate(rough_chunks):
            if skip_next or chunk.strip() == '':
                skip_next = False
                continue

            # 检查当前块是否是分隔符（匹配模式）
            is_separator = any(re.match(p, chunk) for p in patterns)

            if is_separator and i + 1 < len(rough_chunks):
                # 分隔符应合并到下一块
                combined_chunks.append(chunk + rough_chunks[i + 1])
                skip_next = True
            else:
                # 普通内容块
                combined_chunks.append(chunk)

        # 递归处理剩余的切分模式
        final_chunks = []
        for chunk in combined_chunks:
            # 如果块仍然很大，使用剩余的模式进一步切分
            restored_chunk = restore_blocks(chunk, extracted_blocks)
            token_count = self.count_tokens(restored_chunk)

            if token_count > self.max_chunk_size and len(patterns) > 1:
                # 递归使用剩余的模式切分
                sub_chunks = self._split_by_patterns(chunk, patterns[1:], extracted_blocks)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append({
                    'placeholder': chunk,
                    'content': restored_chunk,
                    'tokens': token_count
                })

        return final_chunks

    def _split_large_leaf_node(self, doc: Document, id_map: Dict[str, Document], extracted_blocks: List[Any]) -> None:
        """
        将大叶子节点转换为 CONTAINER 节点，并将其切分块作为子节点。
        """
        original_content = doc.page_content  # 占位符内容
        all_final_chunks = []

        # === 步骤1: 使用两种正则表达式切分 ===
        split_patterns = [
            r'(\n\*\*[^*].*?\n)',   # **标题
            r'(\n\\--\S.*?\n)',     # \--标题
            r'(\n)(?=`[^`\n]*`[^|\n]*?\|)',     # 用 | 分列的表格
            r'(\n\s{2,}\* \*\*[^*]+?\*\* .*?\n)',    # * **标题
        ]
        pattern_chunks = self._split_by_patterns(original_content, split_patterns, extracted_blocks)

        # === 步骤2: 对仍超限的块进行语义切分 ===
        for chunk in pattern_chunks:
            if chunk['tokens'] <= self.max_chunk_size:
                all_final_chunks.append(chunk)
            else:
                semantic_chunks = self.semantic_splitter.split_text(chunk['placeholder'])
                for sc in semantic_chunks:
                    restored = restore_blocks(sc, extracted_blocks)
                    tokens = self.count_tokens(restored)
                    all_final_chunks.append({
                        'placeholder': sc,
                        'content': restored,
                        'tokens': tokens
                    })

        if not all_final_chunks:
            return

        # === 步骤3: 将原叶子节点升级为 CONTAINER ===
        doc.metadata["node_type"] = NodeType.CONTAINER
        doc.metadata["child_ids"] = []  # 清空（原为叶子，无子）
        # 可选：保留标题行作为 page_content，或置空
        # 这里保留原始标题行（如 "## 标题"）以维持可读性
        # 若原始内容无标题行，可设为空
        doc.page_content = doc.page_content.split('\n')[0]  # 保守做法

        # === 步骤4: 创建子节点 ===
        for i, chunk in enumerate(all_final_chunks):
            child_title = f"{doc.metadata['title']}_part{i + 1}"
            child_doc = MarkdownTreeParser._create_document_node(
                content=chunk['content'],
                source=doc.metadata["source"],
                parent_id=doc.id,  # 关键：父为容器
                level=doc.metadata["level"] + 1,
                title=child_title,
                token_count=chunk['tokens'],
                node_type=NodeType.LEAF,
                from_split=True
            )
            id_map[child_doc.id] = child_doc
            doc.metadata["child_ids"].append(child_doc.id)

    def _optimize_tree_structure(self, id_map: Dict[str, Document]):
        """优化树结构：合并兄弟节点，跳过切分节点"""
        # 创建按层级降序的优先队列
        priority_queue = PriorityQueue()
        for doc in id_map.values():
            # 仅非叶子节点入队（跳过所有叶子节点）
            if doc.metadata["node_type"] != NodeType.LEAF:
                priority_queue.put((-doc.metadata["level"], doc.id, doc))

        nodes_to_remove = set()

        while not priority_queue.empty():
            _, _, doc = priority_queue.get()
            # 跳过已删除的节点，处理非叶子节点的子节点合并
            if doc.id in nodes_to_remove:
                continue
            self._handle_parent_node(doc, id_map, nodes_to_remove)

        # 移除标记的节点
        for node_id in nodes_to_remove:
            if node_id in id_map:
                del id_map[node_id]

    def _handle_parent_node(self, doc: Document, id_map: Dict[str, Document], nodes_to_remove: set):
        """处理非叶子节点的子节点合并"""
        child_ids = doc.metadata["child_ids"]
        if not child_ids:
            return
        children = [id_map[cid] for cid in child_ids]

        # 判断父节点类型以决定合并上限
        merge_target_size = self.max_chunk_size if doc.metadata["node_type"] == NodeType.CONTAINER else self.core_chunk_size

        # 第一阶段分组：按节点类型和token数量,仅合并可以合并的连续叶子
        groups = []
        current_group = []
        for child in children:
            if (child.metadata["node_type"] != NodeType.LEAF or
                child.metadata.get("from_split", False) or
                child.metadata["token_count"] >= merge_target_size):
                # 结束当前组并开始新组
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([child])
            else:
                current_group.append(child)
        # 添加最后的组
        if current_group:
            groups.append(current_group)

        # 第二阶段：组内合并
        merged_children = []
        for group in groups:
            # 单节点组直接保留
            if len(group) <= 1:
                merged_children.extend(group)
            else:
                # 只有连续叶子节点组才进行合并
                merged_children.extend(self._merge_sibling_group(group, merge_target_size))

        # 更新父节点的子节点
        if len(merged_children) != len(children):
            # 更新子节点ID列表
            new_child_ids = [child.id for child in merged_children]
            doc.metadata["child_ids"] = new_child_ids
            for child in merged_children:
                id_map[child.id] = child

            # 标记被合并的节点待删除
            original_ids = set(child.id for child in children)
            merged_ids = set(child.id for child in merged_children)
            nodes_to_remove.update(original_ids - merged_ids)

        # 2.4 检查父节点是否可以与唯一子节点合并
        if (len(merged_children) == 1 and doc.metadata["node_type"] != NodeType.CONTAINER and
                merged_children[0].metadata["node_type"] == NodeType.LEAF and
                not merged_children[0].metadata.get("from_split", False) and
                doc.metadata["token_count"] + merged_children[0].metadata["token_count"] <= self.max_chunk_size):
            # 更新父节点
            doc.page_content += "\n" + merged_children[0].page_content
            doc.metadata["token_count"] += merged_children[0].metadata["token_count"]
            doc.metadata["child_ids"] = []
            doc.metadata["node_type"] = NodeType.LEAF

            # 标记子节点待删除
            nodes_to_remove.add(merged_children[0].id)

    @staticmethod
    def _create_merged_node(nodes: List[Document]) -> Document:
        """创建合并节点"""
        merged_content = "\n".join(leaf.page_content for leaf in nodes)
        # 生成合并标题 (父标题 + 末级标题组合)
        parent_title = nodes[0].metadata["title"].rsplit('_', 1)[0]
        last_titles = [leaf.metadata["title"].split('_')[-1] for leaf in nodes]
        merged_title = f"{parent_title}_{'~'.join(last_titles)}"

        return MarkdownTreeParser._create_document_node(
            content=merged_content,
            source=nodes[0].metadata["source"],
            parent_id=nodes[0].metadata["parent_id"],
            level=nodes[0].metadata["level"],
            title=merged_title,
            token_count=sum(leaf.metadata["token_count"] for leaf in nodes),
            merged=True
        )

    def _merge_sibling_group(self, group: List[Document], target_size: int) -> List[Document]:
        """使用动态规划优化节点合并，最小化小token块数量，以target_size为上限"""
        if not group:
            return []

        # 提取token数量并计算前缀和
        tokens = [node.metadata["token_count"] for node in group]
        n = len(tokens)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + tokens[i]

        # 动态规划表初始化
        dp = [float('inf')] * (n + 1)  # dp[i] = 前i个节点的最小违规数
        path = [-1] * (n + 1)  # 记录最优分割点
        dp[0] = 0  # 基础情况：0个节点有0个违规

        # 填充DP表
        for i in range(1, n + 1):
            for j in range(i - 1, -1, -1):
                # 计算当前子序列的token总和
                total = prefix_sum[i] - prefix_sum[j]
                # 如果超过core_chunk_size，提前终止（因为token只会增加）
                if total > target_size:
                    break
                # 计算违规惩罚（小于min_chunk_size为1，否则为0）
                penalty = 1 if total < self.min_chunk_size else 0
                # 更新最优解
                if dp[j] + penalty < dp[i]:
                    dp[i] = dp[j] + penalty
                    path[i] = j

        # 回溯构建最优分割
        segments = []
        i = n
        while i > 0:
            j = path[i]
            segments.append((j, i))  # 左闭右开区间[j, i)
            i = j
        segments.reverse()

        # 根据分割结果创建合并节点
        merged_nodes = []
        for start, end in segments:
            # 获取当前分组的节点
            group_nodes = group[start:end]
            # 单个节点直接保留
            if len(group_nodes) == 1:
                merged_nodes.append(group_nodes[0])
            # 多个节点合并为新节点
            else:
                merged_nodes.append(MarkdownTreeParser._create_merged_node(group_nodes))

        return merged_nodes

    @staticmethod
    def _set_sibling_relations(id_map: Dict[str, Document]):
        """统一设置兄弟关系"""
        for node_id, node in id_map.items():
            if node.metadata["node_type"] in [NodeType.ROOT, NodeType.SECTION]:
                child_ids = node.metadata["child_ids"]
                for i, child_id in enumerate(child_ids):
                    child = id_map[child_id]
                    child.metadata["left_sibling"] = child_ids[i - 1] if i > 0 else ''
                    child.metadata["right_sibling"] = child_ids[i + 1] if i < len(child_ids) - 1 else ''