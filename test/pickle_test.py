import os
import pickle

from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document

from agent.nodes import RetrievalNode, RerankNode
from agent.schemas import ProblemAnalysis, PlanAction, ExecutionPlan, OperationType, NamedEntity, RiskLevel
from agent.state import AgentState
from retriever import GraphTraverser, MilvusHybridRetriever
from utils import get_dense_embed_model, get_sparse_embed_model
from utils.milvus_adapter import connect_milvus_by_env
from workflow.build_knowledge_base import STATIC_PARTITION_NAME

# ==================== 环境变量加载 ====================
load_dotenv(find_dotenv())
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')
MILVUS_USER = os.getenv('MILVUS_USER')
MILVUS_PASSWORD = os.getenv('MILVUS_ROOT_PASSWORD')

def pickle_test():
    print("🚀 Starting Kubernetes Agent...")
    # --- A. 连接 Milvus
    connect_milvus_by_env()

    # --- B. 初始化资源 (一次性加载模型，避免重复加载) ---
    print("⏳ Initializing Embeddings and Retriever (this may take a while)...")

    # 请根据你的实际模型路径修改
    DENSE_MODEL_PATH = "../models/Qwen/Qwen3-Embedding-0.6B"
    SPARSE_MODEL_PATH = "BAAI/bge-m3"
    RERANKER_MODEL_PATH = "../models/Qwen/Qwen3-Reranker-0.6B"
    COLLECTION_NAME = "knowledge_base_v3"
    dense_embedding = get_dense_embed_model(DENSE_MODEL_PATH)
    sparse_embedding = get_sparse_embed_model(SPARSE_MODEL_PATH)


    retriever = MilvusHybridRetriever(
        collection_name=COLLECTION_NAME,
        dense_embedding_func=dense_embedding,
        sparse_embedding_func=sparse_embedding,
        top_k=5
    )

    traverser = GraphTraverser(COLLECTION_NAME, partition_names=[STATIC_PARTITION_NAME])

    reranker = RerankNode(RERANKER_MODEL_PATH, top_n=5)

    retrieval_node = RetrievalNode(retriever=retriever, traverser=traverser, reranker=reranker)

    analysis_diagnosis = ProblemAnalysis(
        reasoning="用户明确指出了具体的 Pod 名称 'redis-cart'...",
        technical_summary="用户报告在 default 命名空间下的 redis-cart Pod 昨天正常，今天突然无法连接，需要进行故障排查。",
        target_operation=OperationType.DIAGNOSIS,
        entities=[
            NamedEntity(name="redis-cart", type="Pod"),
            NamedEntity(name="default", type="Namespace")
        ],
        risk_level=RiskLevel.LOW,
        clarification_question=None
    )

    plan = ExecutionPlan(
        reasoning = "规划的理由，为什么选择这个动作",
        action = PlanAction.RETRIEVE,
        search_queries = ["OOMKilled"],
        tool_name = None,
        tool_args = None,
        final_answer = None,
    )

    state = AgentState(
        messages=[],
        reflections=[],
        analysis=analysis_diagnosis,
        plan=plan,
        evaluation=None,
        retrieved_docs=None,
        tool_output=None,
        retrieval_attempts=0,
        tool_use_attempts=0,
        error=None,
        metadata={}
    )

    new_state = retrieval_node(state, {})
    # new_state["retrieved_docs"] = []
    print(len(new_state["retrieved_docs"]))
    #
    # for dox in new_state["retrieved_docs"]:
    #     for idx in dox.metadata["child_ids"]:
    #         print(idx)
    #     for url in dox.metadata["entry_urls"]:
    #         print(url)
    #     print(dox.metadata["source_type"])
    #     dox.metadata["child_ids"] = list(dox.metadata["child_ids"])
    #     dox.metadata["entry_urls"] = list(dox.metadata["entry_urls"])

    try_pickle(new_state)
    print(pickle.dumps(new_state))

def try_pickle(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                pickle.dumps(v)
            except Exception as e:
                print("❌ Problem field:", k, type(v), e)
                try_pickle(v)

    if isinstance(obj, list):
        for i in range(len(obj)):
            try:
                pickle.dumps(obj[i])
            except Exception as e:
                print("❌ Problem index:", i)
                try_pickle(obj[i])

    if isinstance(obj, Document):
        try_pickle(obj.metadata)

if __name__ == '__main__':
    pickle_test()

