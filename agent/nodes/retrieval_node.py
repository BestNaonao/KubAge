import uuid
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from retriever.MilvusHybridRetriever import MilvusHybridRetriever
from agent.state import AgentState


class RetrievalNode:
    def __init__(self, retriever: MilvusHybridRetriever):
        """
        初始化检索节点
        :param retriever: 已经初始化好的 MilvusHybridRetriever 实例
        """
        self.retriever = retriever

    def __call__(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        执行检索逻辑
        """
        # 1. 获取上一个节点的分析结果
        analysis = state.get("analysis")

        # 安全检查：如果没有分析结果或没有生成搜索查询，直接返回空
        if not analysis or not analysis.search_queries:
            print("❌ No search queries found in state.")
            return {"retrieved_chunks": []}

        queries = analysis.search_queries

        all_retrieved_docs = []

        # 2. 遍历所有 Query 进行检索
        for query in queries:
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
        unique_docs = self._deduplicate_documents(all_retrieved_docs)

        # 4. 更新状态
        # 根据 state.py 的定义，我们返回字典，LangGraph 会将其合并到 State 中
        return {"retrieved_chunks": unique_docs}

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
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