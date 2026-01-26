import uuid
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from agent.nodes import RerankNode
from agent.schemas import ExecutionPlan
from agent.state import AgentState
from retriever.MilvusHybridRetriever import MilvusHybridRetriever


class RetrievalNode:
    """检索节点，合并了RerankNode，包含了粗筛（Retrieve）和精筛（Rerank）"""
    def __init__(self, retriever: MilvusHybridRetriever, reranker: RerankNode):
        """
        初始化检索节点
        :param retriever: 已经初始化好的 MilvusHybridRetriever 实例
        """
        self.retriever = retriever
        self.reranker = reranker

    def __call__(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        执行检索逻辑
        """
        # 1. 获取上一个节点的分析结果
        plan: ExecutionPlan = state.get("plan")

        # 安全检查：如果没有分析结果或没有生成搜索查询，直接返回空
        if not plan or not plan.search_queries:
            print("❌ No search queries found in state.")
            return {"retrieved_docs": [], "error": "No queries in plan"}

        # 获取当前次数 (默认为0)
        current_attempts = state.get("retrieval_attempts", 0)
        print(f"   🔄 Retrieval Attempts: {current_attempts + 1}")

        all_retrieved_docs = []

        # 2. 遍历所有 Query 进行检索
        for query in plan.search_queries:
            try:
                # 调用 MilvusHybridRetriever
                # 注意：retriever.invoke 是 LangChain 标准接口，底层会调用 _get_relevant_documents
                docs = self.retriever.invoke(query)
                all_retrieved_docs.extend(docs)
                print(f"   Query: '{query}' -> Found {len(docs)} docs")
            except Exception as e:
                print(f"❌ Error retrieving for query '{query}': {e}")
                # 单个 query 失败不应阻断整个流程
                continue

        # 3. 文档去重 (Deduplication)
        # 不同的 query 可能会召回相同的文档片段，需要基于 pk 去重
        unique_docs = self._deduplicate(all_retrieved_docs)

        # 4. 更新状态用于 Rerank 子节点
        # Rerank 需要知道 retrieved_docs、technical summary、last_message
        state_for_rerank = {
            "retrieved_docs": unique_docs,
            "analysis": state.get("analysis"),
            "message": state.get("messages")
        }

        # 3. 调用 Rerank 逻辑 (假设 RerankNode 已经是一个 callable)
        # 这里为了简单，假设我们可以直接复用 reranker 实例的方法
        # 或者在这里直接实例化 RerankNode 并调用
        reranked_result = self.reranker(state_for_rerank, config=config)

        final_docs = reranked_result.get("retrieved_chunks", [])
        print(f"   Found {len(final_docs)} relevant docs.")

        return {
            "retrieved_docs": final_docs,
            "tool_output": None,
            "retrieval_attempts": current_attempts + 1
        }  # 清空之前的工具输出

    @staticmethod
    def _deduplicate(documents: List[Document]) -> List[Document]:
        """
        基于文档的 metadata['pk'] 进行去重
        如果 pk 不存在，则回退到使用 page_content 的哈希值
        """
        unique_docs = []
        seen_ids = set()

        for doc in documents:
            # 优先使用数据库主键 pk
            doc_id = doc.metadata.get("pk")

            # 如果 retrieve 的时候没有拉取 pk，则使用内容的哈希兜底
            if not doc_id:
                title = doc.metadata.get("title")
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, title))

            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

        return unique_docs