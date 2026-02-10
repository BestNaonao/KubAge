import logging
from typing import List, Set, Any

import numpy as np
from langchain_core.documents import Document
from pymilvus import Collection

# 配置日志
logger = logging.getLogger(__name__)


class GraphTraverser:
    """
    图拓扑扩展器
    负责基于初始锚点文档 (Anchors) 进行父级递归扩展和链接拓扑扩展
    """

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
        向上递归扩展父节点
        逻辑：Sim_j >= Sim_child * Threshold
        """
        expanded_docs = []

        # 队列：存储 (doc_id, child_similarity_score, source_anchor_text)
        # 初始阶段，我们将 anchor 视为 "child"，其 similarity 设为 1.0 (或者基于检索分，这里简化为 1.0 作为基准)
        # 或者更严格：计算 Anchor 本身与 Query 的相似度作为基准

        next_batch_ids = []
        # 记录每个父ID对应的基准分数和来源锚点
        # Map: parent_id -> (child_score, anchor_text)
        candidates_map = {}

        # --- 初始化：从 Anchors 获取第一层 Parent ---
        for doc in anchors:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in existing_pks:
                # 计算当前 Anchor 的相似度作为基准 Sim_child
                # 如果没有向量，暂时用 1.0，但在严格模式下应该计算
                sim_child = 1.0
                if "summary_vector" in doc.metadata and doc.metadata["summary_vector"]:
                    sim_child = self._cosine_sim(query_vec, doc.metadata["vector"])

                # 如果多个子节点指向同一个父节点，取分数最高的那个路径
                if parent_id not in candidates_map or sim_child > candidates_map[parent_id][0]:
                    candidates_map[parent_id] = (sim_child, doc.page_content[:50])
                    next_batch_ids.append(parent_id)

        # --- 递归循环 ---
        # 设置最大深度防止死循环，例如 5 层
        depth = 0
        while next_batch_ids and depth < 5:
            depth += 1
            # 1. 批量拉取父文档 (包含 summary_vector)
            fetched_docs = self._batch_fetch(next_batch_ids)

            current_generation_ids = []

            for doc in fetched_docs:
                pk = doc.metadata.get("pk")

                # 获取该文档的 "子节点分数" 和 "来源"
                child_score, source_text = candidates_map.get(pk, (0.0, ""))

                # 2. 计算当前父节点的相似度 Sim_j (使用 summary_vector)
                summary_vec = doc.metadata.get("summary_vector")
                if not summary_vec:
                    continue  # 没有向量无法计算，跳过

                sim_j = self._cosine_sim(query_vec, summary_vec)

                # 3. 阈值判断逻辑
                # 规则: Sim_j >= Sim_child * Threshold
                # 同时也必须满足绝对底线 min_sim
                required_score = child_score * self.decay_threshold

                if sim_j >= required_score and sim_j > self.min_sim:
                    # --> 接受该父节点
                    doc.metadata["expansion_type"] = "parent"
                    doc.metadata["expansion_source"] = f"Parent of anchor: '{source_text}...'"
                    doc.metadata["expansion_score"] = float(sim_j)

                    # 加入结果集
                    if pk not in existing_pks:
                        existing_pks.add(pk)
                        expanded_docs.append(doc)

                        # 4. 准备下一轮递归：获取该节点的 parent
                        grand_parent_id = doc.metadata.get("parent_id")
                        if grand_parent_id and grand_parent_id not in existing_pks:
                            # 记录当前节点的分数，作为下一级的 "child_score"
                            if grand_parent_id not in candidates_map or sim_j > candidates_map[grand_parent_id][0]:
                                candidates_map[grand_parent_id] = (sim_j, source_text)
                                current_generation_ids.append(grand_parent_id)

            # 更新下一轮 ID
            next_batch_ids = current_generation_ids

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
            logger.error(f"Milvus batch fetch failed: {e}")
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