import logging
from typing import List, Dict, Any, TypedDict, Optional

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from agent.nodes import RerankNode
from agent.schemas import ExecutionPlan, OperationType
from agent.state import AgentState
from retriever import MilvusHybridRetriever, GraphTraverser
from utils import csr_to_milvus_format
from utils.document_schema import SourceType
from workflow.build_knowledge_base import STATIC_PARTITION_NAME, DYNAMIC_PARTITION_NAME


class VectorSchema(TypedDict):
    dense: Optional[List[float]]
    sparse: Optional[Dict[int, float]]

class RetrievalNode:
    """
    检索节点，批处理向量嵌入，实现双轨制、三阶段检索：静态轨与动态轨，粗筛（Retrieval）、扩展（Expansion）和精筛（Rerank）
    """
    logger = logging.getLogger(__name__)
    priority_map = {
        SourceType.DYNAMIC: 4,
        SourceType.ANCHOR: 3,
        SourceType.PARENT: 2,
        SourceType.LINK: 1,
        SourceType.SIBLING: 1,
        SourceType.UNKNOWN: 0
    }       # 文档来源的优先级
    dynamic_track_ops = [OperationType.DIAGNOSIS, OperationType.RESOURCE_INQUIRY, OperationType.RESTART]

    def __init__(self, retriever: MilvusHybridRetriever, traverser: GraphTraverser, reranker: RerankNode):
        """
        初始化检索节点
        :param retriever: 已初始化的 MilvusHybridRetriever (持有 embedding models)
        :param traverser: 已初始化的 GraphTraverser (只负责拓扑计算)
        :param reranker: 已初始化的 RerankNode
        """
        self.retriever = retriever
        self.traverser = traverser
        self.reranker = reranker

    def _batch_embed_queries(self, queries: List[str]) -> Dict[str, VectorSchema]:
        """
        Batch embed all queries using models from the retriever.
        """
        if not queries:
            return {}

        # 1. Dense Embeddings (尝试使用 batch 接口)
        try:
            dense_vecs = self.retriever.dense_embedding_func.embed_documents(queries)
        except AttributeError:  # 回退到循环
            dense_vecs = [self.retriever.dense_embedding_func.embed_query(q) for q in queries]

        # 2. Sparse Embeddings (适配 BGE-M3)
        try:
            sparse_result = self.retriever.sparse_embedding_func.encode_queries(queries)["sparse"]
            sparse_vecs = csr_to_milvus_format(sparse_result)
        except Exception as e:
            self.logger.error(f"Batch sparse embedding failed: {e}")
            raise e

        # 3. Construct Cache
        cache: Dict[str, VectorSchema] = {}
        for i, query in enumerate(queries):
            cache[query]: VectorSchema = {"dense": dense_vecs[i], "sparse": sparse_vecs[i],}
        return cache

    def _get_source_priority(self, document: Document) -> int:
        return self.priority_map.get(document.metadata.get("source_type"), 0)

    def _upsert_doc(self, buffer: Dict[str, Document], doc: Document) -> None:
        """合并文档到 Buffer，保留高优先级版本"""
        if not (pk := doc.metadata.get("pk")):
            return

        if pk not in buffer:
            buffer[pk] = doc  # 新文档，直接加入
        else:
            # 已存在的文档，检查优先级，优先级更高，覆盖旧文档，保留更重要的 source_desc
            # 相同优先级默认保留第一个，通常第一个 Query 在 Planner 中更重要。
            old_prio = self._get_source_priority(buffer[pk])
            new_prio = self._get_source_priority(doc)
            if new_prio > old_prio:
                buffer[pk] = doc

    def _execute_static_retrieval(self, queries: List[str], embedding_cache: Dict) -> Dict[str, Document]:
        """
        轨道一：静态知识检索
        - 使用 Plan 生成的泛化 Query
        - 目标：static_knowledge 分区
        - 包含：Topology Expansion (父节点/兄弟节点/内部链接)
        """
        # 使用 Dict[pk, Document] 代替 List 进行去重和管理
        candidate_buffer: Dict[str, Document] = {}
        print(f"   📚 [Static Track]: Processing {len(queries)} queries on 'static_knowledge'...")
        # 遍历每个 Query (Retrieval + Expansion)
        for query in queries:
            if not (vectors := embedding_cache.get(query)):
                continue
            try:
                # 1. Hybrid Search (获取 Anchors)
                anchors = self.retriever.search_with_vectors(
                    dense_vec=vectors["dense"],
                    sparse_vec=vectors["sparse"],
                    partition_names=[STATIC_PARTITION_NAME]
                )
                # 标记来源
                for doc in anchors:
                    doc.metadata["source_type"] = SourceType.ANCHOR
                    doc.metadata["source_desc"] = f"Direct hit by query: '{query}'"

                # 2. 拓扑扩展 (Graph Topology Expansion)
                expanded_docs = self.traverser.expand(anchors, vectors['dense'])
                current_batch = anchors + expanded_docs

                # 3. 基于优先级的 Upsert (合并到 Buffer)
                for doc in current_batch:
                    self._upsert_doc(candidate_buffer, doc)

                print(f"   Query: '{query}' -> Found {len(current_batch)} docs "
                      f"(Anchors: {len(anchors)}, Expanded: {len(expanded_docs)})")

            except Exception as e:
                print(f"❌ Static retrieval error for query '{query}': {e}")
                continue  # 单个 query 失败不应阻断整个流程
        return candidate_buffer

    def _execute_dynamic_retrieval(self, technical_summary: str, embedding_cache: Dict) -> Dict[str, Document]:
        """
        轨道二：动态事件检索
        - 使用 Analysis 中的 Technical Summary (包含具体实体)
        - 目标：dynamic_events 分区
        - 包含：动静关联 (通过 related_links 拉取静态手册)
        """
        candidate_buffer: Dict[str, Document] = {}
        # 获取向量
        if not (vectors := embedding_cache.get(technical_summary)):
            return {}

        print(f"   🚨 [Dynamic Track]: Searching 'dynamic_events' with summary...")

        try:
            # 1. 动态检索 (Top-K 较小，例如 2)
            # 需要retriever支持动态传参
            dynamic_hits = self.retriever.search_with_vectors(
                dense_vec=vectors["dense"],
                sparse_vec=vectors["sparse"],
                limit=2,
                partition_names=[DYNAMIC_PARTITION_NAME]  # 显式指定动态分区
            )

            for doc in dynamic_hits:
                # 标记 Dynamic
                doc.metadata["source_type"] = SourceType.DYNAMIC
                doc.metadata["source_desc"] = "Runtime Event Match"
                self._upsert_doc(candidate_buffer, doc)

                # 2. 动静关联 (Reverse Instantiation / Alignment)
                # 检查动态节点是否通过 related_links 指向了静态锚点，这些链接是在 RuntimeBridge 入库时计算好的
                related_links = doc.metadata.get("related_links", [])

                static_anchor_pks = []
                for link in related_links:
                    if link.get("type") == "static_anchor":
                        static_anchor_pks.append(link.get("pk"))

                if static_anchor_pks:
                    print(f"      🔗 Linked to {len(static_anchor_pks)} static anchors.")
                    # 批量拉取这些静态文档 (复用 traverser 的 batch_fetch)
                    linked_static_docs = self.traverser.batch_fetch(static_anchor_pks)

                    for static_doc in linked_static_docs:
                        static_doc.metadata["source_type"] = SourceType.LINK    # 或者叫 alignment_anchor
                        static_doc.metadata["source_desc"] = f"Aligned from Event: {doc.metadata.get('title')}"
                        self._upsert_doc(candidate_buffer, static_doc)

        except Exception as e:
            print(f"❌ Dynamic retrieval error: {e}")

        return candidate_buffer

    def __call__(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        执行检索逻辑
        """
        # 1. 获取上一个节点的分析结果，并安全检查
        plan: ExecutionPlan = state.get("plan")
        analysis = state.get("analysis")
        if not plan or not plan.search_queries:
            print("❌ No search queries found in state.")
            return {"retrieved_docs": [], "error": "No queries in plan"}

        # 获取当前次数 (默认为0)
        current_attempts = state.get("retrieval_attempts", 0)
        print(f"   🔄 Retrieval Attempts: {current_attempts + 1}")

        # 1. 准备 Query 列表
        # 静态轨 Query
        static_queries = plan.search_queries
        # 动态轨 Query (仅当需要诊断/查询资源时，使用 technical_summary，因为它保留了实体信息)
        dynamic_queries = [analysis.technical_summary] if analysis and analysis.target_operation in self.dynamic_track_ops else []

        # 2. 统一 Embedding (Batch 处理提高效率)
        all_queries = static_queries + dynamic_queries
        print(f"🔍 Embedding {len(all_queries)} queries...")
        embedding_cache = self._batch_embed_queries(all_queries)

        # 3. 并行/串行执行双轨检索
        # A. 静态轨
        static_results = self._execute_static_retrieval(static_queries, embedding_cache)
        # B. 动态轨
        dynamic_results = self._execute_dynamic_retrieval(dynamic_queries[0], embedding_cache) if dynamic_queries else {}

        # 4. 合并结果
        # 由于 priority_map 中 dynamic_event 优先级最高，所以动态事件肯定会被保留。
        final_buffer = static_results.copy()
        for doc in dynamic_results.values():
            self._upsert_doc(final_buffer, doc)

        all_candidates = list(final_buffer.values())
        print(f"∑ Total unique candidates after merging: {len(all_candidates)}")

        # 5. Rerank 阶段
        # Rerank 需要知道 retrieved_docs、technical summary、last_message
        state_for_rerank = {
            "retrieved_docs": all_candidates,
            "analysis": state.get("analysis"),
            "message": state.get("messages")
        }

        # 调用 Rerank (假设 RerankNode 已经是一个 callable)
        # 这里为了简单，假设我们可以直接复用 reranker 实例的方法，或者在这里直接实例化 RerankNode 并调用
        reranked_result = self.reranker(state_for_rerank, config=config)
        final_docs = reranked_result.get("retrieved_docs", [])
        print(f"   Found {len(final_docs)} relevant docs.")

        return {
            "retrieved_docs": final_docs,
            "tool_output": None,
        }  # 清空之前的工具输出