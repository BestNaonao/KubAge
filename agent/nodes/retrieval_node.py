import logging
from typing import List, Dict, Any, TypedDict, Optional

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from agent.nodes import RerankNode
from agent.schemas import ExecutionPlan
from agent.state import AgentState
from retriever import MilvusHybridRetriever, GraphTraverser
from utils import csr_to_milvus_format


class VectorSchema(TypedDict):
    dense: Optional[List[float]]
    sparse: Optional[Dict[int, float]]

class RetrievalNode:
    """
    检索节点，批处理向量嵌入，分三阶段检索：粗筛（Retrieval）、扩展（Expansion）和精筛（Rerank）
    """
    logger = logging.getLogger(__name__)
    priority_map = {'anchor': 3, 'parent': 2, 'link': 1,  'sibling': 1}     # 文档来源的优先级

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
            # 假设 sparse_embedding_func 是 BGE-M3 wrapper，具有 encode_queries
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
        return self.priority_map.get(document.metadata.get("source_type", "unknown"), 0)

    def __call__(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        执行检索逻辑
        """
        # 1. 获取上一个节点的分析结果，并安全检查
        plan: ExecutionPlan = state.get("plan")
        if not plan or not plan.search_queries:
            print("❌ No search queries found in state.")
            return {"retrieved_docs": [], "error": "No queries in plan"}

        # 获取当前次数 (默认为0)
        current_attempts = state.get("retrieval_attempts", 0)
        print(f"   🔄 Retrieval Attempts: {current_attempts + 1}")

        # 批量 Embedding，生成上下文缓存
        queries = plan.search_queries
        print(f"🔍 Processing {len(queries)} queries...")
        embedding_cache = self._batch_embed_queries(queries)

        # 使用 Dict[pk, Document] 代替 List 进行去重和管理
        # 键是 PK，值是 Document 对象
        candidate_buffer: Dict[str, Document] = {}

        # 2. 遍历每个 Query (Retrieval + Expansion)
        for query in plan.search_queries:
            vectors = embedding_cache.get(query)
            try:
                # 调用 MilvusHybridRetriever
                # A. Hybrid Search (获取 Anchors)
                anchors = self.retriever.search_with_vectors(
                    dense_vec=vectors["dense"],
                    sparse_vec=vectors["sparse"],
                )
                # 标记来源
                for doc in anchors:
                    doc.metadata["source_type"] = "anchor"
                    doc.metadata["source_desc"] = f"Direct hit by query: '{query}'"

                # B. Graph Expansion (传入 Dense Vector 即可)
                expanded_docs = self.traverser.expand(anchors, vectors['dense'])
                current_batch = anchors + expanded_docs
                print(f"   Query: '{query}' -> Found {len(current_batch)} docs "
                      f"(Anchors: {len(anchors)}, Expanded: {len(expanded_docs)})")

                # C. 基于优先级的 Upsert (合并到 Buffer)
                for doc in current_batch:
                    if not (pk := doc.metadata.get("pk")):
                        continue

                    if pk not in candidate_buffer:
                        candidate_buffer[pk] = doc  # 新文档，直接加入
                    else:
                        # 已存在的文档，检查优先级
                        # 优先级更高，覆盖旧文档，保留更重要的 source_desc
                        # 相同优先级默认保留第一个，通常第一个 Query 在 Planner 中更重要。
                        old_prio = self._get_source_priority(candidate_buffer[pk])
                        new_prio = self._get_source_priority(doc)
                        if new_prio > old_prio:
                            candidate_buffer[pk] = doc

            except Exception as e:
                print(f"❌ Error retrieving for query '{query}': {e}")
                continue    # 单个 query 失败不应阻断整个流程

        # 将 Buffer 转回 List
        all_candidates = list(candidate_buffer.values())
        print(f"∑ Total unique candidates after merging: {len(all_candidates)}")

        # 3. Rerank 阶段
        # Rerank 需要知道 retrieved_docs、technical summary、last_message
        state_for_rerank = {
            "retrieved_docs": all_candidates,
            "analysis": state.get("analysis"),
            "message": state.get("messages")
        }

        # 调用 Rerank (假设 RerankNode 已经是一个 callable)
        # 这里为了简单，假设我们可以直接复用 reranker 实例的方法
        # 或者在这里直接实例化 RerankNode 并调用
        reranked_result = self.reranker(state_for_rerank, config=config)
        final_docs = reranked_result.get("retrieved_docs", [])
        print(f"   Found {len(final_docs)} relevant docs.")

        return {
            "retrieved_docs": final_docs,
            "tool_output": None,
        }  # 清空之前的工具输出