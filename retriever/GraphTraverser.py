import logging
from typing import List, Set, Dict

import numpy as np
from langchain_core.documents import Document
from pymilvus import Collection

from utils import generate_node_id


class GraphTraverser:
    """
    图拓扑扩展器
    负责基于初始锚点文档 (Anchors) 进行父级递归扩展和链接拓扑扩展
    """
    # 配置日志
    logger = logging.getLogger(__name__)
    def __init__(
            self,
            milvus_collection_name: str,
            milvus_connection_alias: str = "default",
            parent_decay_threshold: float = 0.75,
            absolute_min_similarity: float = 0.2,  # 防止相关性太低
            link_proportion: float = 0.75,
            max_link_top_k: int = 5
    ):
        self.collection_name = milvus_collection_name
        self.alias = milvus_connection_alias

        # 阈值配置
        self.decay_threshold = parent_decay_threshold
        self.min_sim = absolute_min_similarity
        self.link_proportion = link_proportion
        self.max_link_top_k = max_link_top_k

    def expand(self, anchors: List[Document], query_vec: List[float]) -> List[Document]:
        """
        执行图扩展的主入口
        Args:
            anchors: 初始检索到的锚点文档列表
            query_vec: 预计算好的 Query 稠密向量 (来自 RetrievalNode 的缓存)
        """
        if not anchors:
            return []

        # 建立已存在 ID 集合，用于去重
        existing_pks = {doc.metadata.get("pk") for doc in anchors if doc.metadata.get("pk")}

        # 1. 父级递归扩展
        parent_docs = self._expand_parents(anchors, query_vec, existing_pks)
        print(f"   ⬆️  Parent Expansion: Found {len(parent_docs)} docs")

        # 2. 链接拓扑扩展
        link_docs = self._expand_links(anchors, query_vec, existing_pks)
        print(f"   🔗 Link Expansion: Found {len(link_docs)} docs")

        # 3. 合并结果 (此时所有文档已去重且标记了 metadata)
        # 注意：这里只返回新增的扩展文档，还是返回全部？
        # 通常 Traverser 返回扩展部分，由调用方合并。但为了方便，这里返回 List[ExpandedDoc]
        return parent_docs + link_docs

    def _expand_parents(self, anchors: List[Document], query_vec: List[float], existing_pks: Set[str]) -> List[Document]:
        """
        基于 Title 面包屑结构一次性溯源，向上扩展父节点
        逻辑：Sim_j >= Sim_child * Threshold
        """
        # 1. 计算所有潜在的祖先节点 ID
        # Map: anchor_pk -> ancestor_ids (按 parent -> root 排序)
        lineage_map: Dict[str, List[str]] = {}
        all_ancestor_pks = set()

        for doc in anchors:
            pk = doc.metadata.get("pk")
            title = doc.metadata.get("title")
            parts = title.split('_')    # 分割标题

            if not pk or not title or len(parts) <= 1:
                continue  # 没有父节点（已经是根或无结构）

            ancestor_ids = []
            # 从长到短切分，确保顺序是：直接父节点 -> ... -> 根节点。indices: len-1, len-2, ... 1
            for i in range(len(parts) - 1, 0, -1):
                parent_title_str = "_".join(parts[:i])
                parent_id = generate_node_id(parent_title_str)
                ancestor_ids.append(parent_id)

            lineage_map[pk] = ancestor_ids
            all_ancestor_pks.update(ancestor_ids)

        if not all_ancestor_pks:
            return []

        # 2. 批量拉取所有祖先文档 (1次 IO)
        # 过滤掉已经是 Anchor 自身的文档 (理论上 generate_node_id 不会冲突，但为了安全)
        fetch_list = list(all_ancestor_pks - existing_pks)
        fetched_docs = self._batch_fetch(fetch_list)

        # 建立速查表: pk -> Document
        doc_lookup = {d.metadata.get("pk"): d for d in fetched_docs}

        # 3. 内存中执行语义衰减检查
        expanded_docs = []
        for anchor in anchors:
            anchor_pk = anchor.metadata.get("pk")
            ancestors = lineage_map.get(anchor_pk, [])  # 已按 parent -> root 排序

            # 获取 Anchor 自身相似度作为基准
            summary_vec = anchor.metadata.get("summary_vector")
            child_sim = self._cosine_sim(query_vec, summary_vec) if summary_vec else 0.5

            # 当前这一代的“子节点”分数，初始为 Anchor 的分数
            current_child_score = child_sim
            current_child_text = anchor.metadata.get("title", anchor.page_content[:50])

            for ancestor_pk in ancestors:
                parent_doc = doc_lookup.get(ancestor_pk)

                # 检查文档是否存在并且摘要向量是否存在，同时赋值给 summary_vec
                if not parent_doc or not (summary_vec := parent_doc.metadata.get("summary_vector")):
                    continue

                # 计算相似度与判断阈值: Parent 必须达到 Child * Threshold
                sim_j = self._cosine_sim(query_vec, summary_vec)
                required_score = current_child_score * self.decay_threshold

                if sim_j >= required_score and sim_j > self.min_sim:
                    if ancestor_pk not in existing_pks:     # 达标：加入结果
                        # 注入 Metadata
                        parent_doc.metadata["expansion_type"] = "parent"
                        parent_doc.metadata["expansion_source"] = f"Parent of: '{current_child_text}'"
                        parent_doc.metadata["expansion_score"] = float(sim_j)

                        existing_pks.add(ancestor_pk)
                        expanded_docs.append(parent_doc)

                    # 更新状态，准备判断下一级 (GrandParent)
                    current_child_score = sim_j
                    current_child_text = parent_doc.metadata.get("title", "parent_node")
                else:
                    # 衰减阻断：如果这一级父节点不相关，不再继续向上追溯根节点，避免把无关的全局 Root 拉进来
                    break

        return expanded_docs

    def _expand_links(self, anchors: List[Document], query_vec: List[float], existing_pks: Set[str]) -> List[Document]:
        """
        扩展文档内部的关联链接 (Related Links)
        逻辑：获取所有 Link -> 计算 Sim(Link, Query) -> Top-L 截断
        """

        # 1. 收集所有待选链接 ID
        # Map: link_pk -> (link_text, source_anchor_text)
        link_candidates = {}

        for doc in anchors:
            # related_links 是 list of dict: [{'pk':..., 'text':..., 'type':...}]
            links = doc.metadata.get("related_links", [])
            if not links:
                continue

            for link in links:
                target_pk = link.get("pk")
                l_type = link.get("type")
                l_text = link.get("text", "link")
                # 只处理内部链接且未被收录的
                if target_pk and l_type == "internal" and target_pk not in existing_pks:
                    # 如果同一个文档被多次引用，保留任意一个来源即可
                    link_candidates[target_pk] = (l_text, doc.page_content[:50])

        if not link_candidates:
            return []

        # 2. 批量拉取链接文档
        candidate_pks = list(link_candidates.keys())
        fetched_docs = self._batch_fetch(candidate_pks)

        # 3. 计算相似度并评分
        scored_candidates = []
        for doc in fetched_docs:
            pk = doc.metadata.get("pk")
            link_text, source_text = link_candidates.get(pk, ("unknown", "unknown"))

            summary_vec = doc.metadata.get("summary_vector")
            if not summary_vec:
                continue

            score = self._cosine_sim(query_vec, summary_vec)

            # 记录必要信息以便排序
            scored_candidates.append({
                "doc": doc,
                "score": score,
                "link_text": link_text,
                "source_text": source_text
            })

        # 4. 动态计算 Top-L
        # max(1, min(5, ceil(len * proportion)))
        import math
        total_candidates = len(scored_candidates)
        top_l = max(1, min(self.max_link_top_k, math.ceil(total_candidates * self.link_proportion)))

        # 5. 排序截断
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        selected_items = scored_candidates[:top_l]

        final_docs = []
        for item in selected_items:
            doc = item["doc"]
            # 注入 Metadata
            doc.metadata["expansion_type"] = "link"
            doc.metadata["expansion_source"] = f"Linked via '{item['link_text']}' from anchor '{item['source_text']}...'"
            doc.metadata["expansion_score"] = float(item["score"])

            existing_pks.add(doc.metadata.get("pk"))  # 更新去重集合
            final_docs.append(doc)

        return final_docs

    def _batch_fetch(self, pks: List[str]) -> List[Document]:
        """
        从 Milvus 批量获取文档
        """
        if not pks:
            return []

        try:
            col = Collection(self.collection_name, using=self.alias)
            # 构造表达式
            expr = f"pk in {str(pks)}"

            # 需要拉取的字段
            output_fields = [
                "pk", "text", "title", "parent_id", "summary_vector",
                "node_type", "related_links", "source"
            ]

            res = col.query(expr, output_fields=output_fields)

            # 转换为 Document 对象
            documents = []
            for hit in res:
                content = hit.get("text", "")
                # 移除 vector 字段以节省内存 (除非下一轮需要)
                # 这里我们需要 summary_vector 计算相似度，保留在 metadata 中
                meta = {k: v for k, v in hit.items() if k != "text"}
                doc = Document(page_content=content, metadata=meta)
                documents.append(doc)

            return documents

        except Exception as e:
            self.logger.error(f"Milvus batch fetch failed: {e}")
            return []

    @staticmethod
    def _cosine_sim(vec_a: List[float], vec_b: List[float]) -> float:
        """
        计算余弦相似度
        """
        # 转换为 numpy 数组
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))