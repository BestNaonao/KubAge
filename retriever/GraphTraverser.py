import json
import logging
import math
from typing import List, Set, Dict, Optional

import numpy as np
from langchain_core.documents import Document
from pymilvus import Collection

from retriever import MilvusHybridRetriever
from utils import generate_node_id
from utils.milvus_adapter import HYBRID_SEARCH_FIELDS, decode_hit_to_document, decode_query_result_to_document


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
            partition_names: Optional[List[str]] = None,
            parent_decay_threshold: float = 0.75,
            absolute_min_similarity: float = 0.2,  # 防止相关性太低
            link_proportion: float = 0.75,
            max_link_top_k: int = 10
    ):
        self.collection = Collection(milvus_collection_name, using=milvus_connection_alias)
        self.partition_names = partition_names
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

        # 1. 父级递归扩展 (基于 Title 面包屑，优先执行以确立上下文 Scope)
        parent_docs = self._expand_parents(anchors, query_vec, existing_pks)
        print(f"   ⬆️  Parent Expansion: Found {len(parent_docs)} docs")

        # 2. 链接与兄弟扩展 (基于 Milvus Search，补充关联信息)
        link_docs = self._expand_links(anchors, query_vec, existing_pks)
        print(f"   🔗 Link Expansion: Found {len(link_docs)} docs")

        # 3. 合并结果 (此时所有文档已去重且标记了 metadata)
        # 通常 Traverser 返回扩展部分，由调用方合并。为了方便，这里返回 List[ExpandedDoc]
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
        fetched_docs = self.batch_fetch(fetch_list)
        doc_lookup = {d.metadata.get("pk"): d for d in fetched_docs}    # 建立倒查表: pk -> Document

        # 3. 内存中执行语义衰减检查
        expanded_docs = []
        for anchor in anchors:
            anchor_pk = anchor.metadata.get("pk")
            ancestors = lineage_map.get(anchor_pk, [])  # 已按 parent -> root 排序

            # 获取 Anchor 自身相似度作为基准
            anchor_summary = anchor.metadata.get("summary_vector")
            child_sim = self._cosine_sim(query_vec, anchor_summary) if anchor_summary else 0.5

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
                    if ancestor_pk not in existing_pks:     # 达标：加入结果，元数据增强
                        parent_doc.metadata["source_type"] = "parent"
                        parent_doc.metadata["source_desc"] = f"Parent of: '{current_child_text}'"
                        # 记录并扩展
                        existing_pks.add(ancestor_pk)
                        expanded_docs.append(parent_doc)

                    # 更新状态，准备判断下一级 (GrandParent)
                    current_child_score = sim_j
                    current_child_text = parent_doc.metadata.get("title", "parent_node")
                else:
                    break   # 衰减阻断：如果这一级父节点不相关，不再继续向上追溯根节点，避免拉入无关 Root

        return expanded_docs

    def _expand_links(self, anchors: List[Document], query_vec: List[float], existing_pks: Set[str]) -> List[Document]:
        """
        扩展文档内部的关联链接 (Related Links)和兄弟节点 (Siblings)
        逻辑：获取所有 Link -> 计算 Sim(Link, Query) -> Top-L 截断
        """
        # Map: candidate_pk -> (source_anchor_text, relationship_type)，用于后续给召回文档打标
        candidate_map = {}

        for doc in anchors:
            source_title = doc.metadata.get("title")

            # 1. 先处理兄弟节点 (优先级较低，作为 Base)
            prev_id: str = doc.metadata.get("left_sibling")
            next_id: str = doc.metadata.get("right_sibling")

            if prev_id and prev_id not in existing_pks and prev_id not in candidate_map:
                candidate_map[prev_id] = (f"Previous of '{source_title}'", "sibling")
            if next_id and next_id not in existing_pks and next_id not in candidate_map:
                candidate_map[next_id] = (f"Next of '{source_title}'", "sibling")

            # 2. 后处理引用链接 (优先级较高，覆写 or 融合)
            for link in doc.metadata.get("related_links", []):
                if isinstance(link, dict):
                    target_pk: str = link.get("pk")
                    l_type = link.get("type")
                    l_text = link.get("text", "link")

                    if target_pk and l_type == "internal" and target_pk not in existing_pks:
                        # 构造强语义描述
                        link_desc = f"Linked via '{l_text}' from '{source_title}'"

                        # 逻辑：无论之前是否作为兄弟节点添加过，这里都进行覆盖或增强
                        # 因为锚点文本 (l_text) 对 Rerank 的价值远大于 "Next step"
                        if target_pk in candidate_map:
                            # 【高阶策略】如果已经存在（说明既是兄弟又是引用），可以合并描述
                            old_desc, old_type = candidate_map[target_pk]
                            if "sibling" in old_type:
                                merged_desc = f"{link_desc} (also {old_desc})"
                                candidate_map[target_pk] = (merged_desc, "link")
                        else:
                            candidate_map[target_pk] = (link_desc, "link")  # 未检索，直接添加

        if not candidate_map:
            return []

        candidate_pks: list[str] = list(candidate_map.keys())

        # 3. 动态计算 Top-L
        # P: 数据驱动的目标数量 (候选总数的一定比例)
        # A. base_floor: 基础保底数量 (1 + 锚点数)，保证每个锚点至少有一个扩展机会
        # K. max_link_top_k: 系统硬性上限，防止 Context 爆炸

        limit_by_prop = math.ceil(len(candidate_pks) * self.link_proportion)
        base_floor = 1 + len(anchors)
        top_l = min(self.max_link_top_k, max(base_floor, limit_by_prop))

        # 4. 使用 Milvus 进行高效过滤和排序
        # 注意：这里我们使用 summary_vector 进行相似度计算（通常更轻量且代表性强）
        # 如果 collection 中 vector 是全文向量，summary_vector 是摘要向量，
        # 在做“链接推荐”时，用 Query 匹配 Link 的 Summary 可能比匹配 Full Text 更准。
        try:
            # 构造 expr: pk in ["a", "b", ...]。注意 Milvus expr 对 list 长度有限制 (通常 < 16384)，这里通常不会超
            expr = f"pk in {json.dumps(candidate_pks)}"
            search_params = MilvusHybridRetriever.dense_search_params

            # 执行 Search
            res = self.collection.search(
                data=[query_vec],
                anns_field="summary_vector",
                param=search_params,
                limit=top_l,
                expr=expr,
                partition_names=self.partition_names,
                output_fields=HYBRID_SEARCH_FIELDS,
            )

            final_docs = []
            # 5. 解析结果
            for hits in res:
                for hit in hits:
                    decoded_doc = decode_hit_to_document(hit, content_field="text")
                    pk = decoded_doc.metadata.get("pk")
                    # 恢复来源上下文，注入扩展元数据
                    source_desc, source_type = candidate_map.get(pk, ("Unknown link", "link"))
                    decoded_doc.metadata["source_type"] = source_type
                    decoded_doc.metadata["source_desc"] = source_desc
                    # 记录并扩展
                    existing_pks.add(pk)
                    final_docs.append(decoded_doc)

            return final_docs

        except Exception as e:
            self.logger.error(f"Milvus link expansion search failed: {e}")
            # Fallback (可选): 如果 search 失败，可以降级回 batch fetch，但通常 search 失败 fetch 也会失败
            return []

    def batch_fetch(self, pks: List[str]) -> List[Document]:
        """
        从 Milvus 批量获取文档
        """
        if not pks:
            return []

        try:
            expr = f"pk in {json.dumps(pks)}"   # 构造表达式
            res = self.collection.query(expr, output_fields=HYBRID_SEARCH_FIELDS, partition_names=self.partition_names)
            return [decode_query_result_to_document(row, content_field="text") for row in res]

        except Exception as e:
            self.logger.error(f"Milvus batch fetch failed: {e}")
            return []

    @staticmethod
    def _cosine_sim(vec_a: List[float], vec_b: List[float]) -> float:
        """计算余弦相似度"""
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))