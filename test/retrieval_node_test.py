from typing import List, Dict, Any

from langchain_core.runnables import RunnableConfig
# 引入 Embedding 依赖
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agent.nodes import RerankNode, RetrievalNode
from agent.schemas import ProblemAnalysis, ExecutionPlan, PlanAction
# 引入项目模块 (请根据实际路径调整)
from agent.state import AgentState
from retriever import MilvusHybridRetriever
from test_dataset.retrieval_cases import ALL_RETRIEVAL_SCENARIOS, RetrievalTestScenario
from utils import get_dense_embed_model, get_sparse_embed_model
from utils.milvus_adapter import connect_milvus_by_env


# ==========================================
# 1. 定义 Dummy Analysis Node (虚拟节点)
# ==========================================
class DummyAnalysisNode:
    """
    一个伪造的分析节点，它不调用 LLM，
    而是直接从输入的 input['analysis'] 中读取预设好的 Analysis 对象，
    并更新到 State 中。
    """

    def __call__(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n--- [Dummy Analysis Node] Injecting Mock Data ---")
        # 这里的 state 在 invoke 时会传入我们构造的初始数据
        # 我们约定在 metadata 中传入预设的 analysis 对象
        metadata = state.get("metadata", {})
        mock_analysis: ProblemAnalysis = metadata.get("inject_analysis")
        queries = metadata.get("inject_plan_queries")
        exePlan = ExecutionPlan(
            reasoning="Dummy Reason",
            action=PlanAction.RETRIEVE,
            search_queries=queries,
        )

        if not mock_analysis:
            raise ValueError("Test Error: No mock analysis data found in metadata!")

        print(f"✅ Injected Analysis for operation: {mock_analysis.target_operation}")
        return {"analysis": mock_analysis, "plan": exePlan}


# ==========================================
# 2. 测试主流程
# ==========================================
def retrieval_workflow_test(scenarios: List[RetrievalTestScenario]):
    print("🚀 Starting Retrieval Node Workflow Test Batch...")
    # --- A. 连接 Milvus
    connect_milvus_by_env()

    # --- B. 初始化资源 (一次性加载模型，避免重复加载) ---
    print("⏳ Initializing Embeddings and Retriever (this may take a while)...")

    # 请根据你的实际模型路径修改
    DENSE_MODEL_PATH = "../models/Qwen/Qwen3-Embedding-0.6B"
    SPARSE_MODEL_PATH = "BAAI/bge-m3"
    RERANKER_MODEL_PATH = "../models/Qwen/Qwen3-Reranker-0.6B"
    COLLECTION_NAME = "knowledge_base_v2"

    # 1. Dense Embedding
    dense_embedding = get_dense_embed_model(DENSE_MODEL_PATH)

    # 2. Sparse Embedding
    sparse_embedding = get_sparse_embed_model(SPARSE_MODEL_PATH)

    # 3. Retriever
    retriever = MilvusHybridRetriever(
        collection_name=COLLECTION_NAME,
        dense_embedding_func=dense_embedding,
        sparse_embedding_func=sparse_embedding,
        top_k=5
    )

    # --- B. 构建图 (Build the Graph) ---
    workflow = StateGraph(AgentState)

    dummy_analysis_node = DummyAnalysisNode()
    reranker = RerankNode(
        model_path=RERANKER_MODEL_PATH,
        top_n=3,
    )
    retrieval_node = RetrievalNode(retriever, reranker)

    # 添加节点
    workflow.add_node("mock_analysis", dummy_analysis_node)
    workflow.add_node("retrieve_docs", retrieval_node)

    # 定义边
    workflow.add_edge(START, "mock_analysis")
    workflow.add_edge("mock_analysis", "retrieve_docs")
    workflow.add_edge("retrieve_docs", END)

    app = workflow.compile()

    # --- C. 循环运行测试用例 ---
    success_count = 0
    total_count = len(scenarios)

    for i, case in enumerate(scenarios, 1):
        print(f"\n{'=' * 20} Test Case {i}/{total_count}: {case.name} {'=' * 20}")
        if case.description:
            print(f"📝 Description: {case.description}")

        try:
            # 构造输入
            # 我们通过 metadata 把 mock_analysis 传递给 DummyNode
            inputs = {
                "messages": [],  # 检索节点其实不看 messages，只看 analysis
                "metadata": {
                    "inject_analysis": case.mock_analysis,
                    "inject_plan_queries": case.mock_plan_queries
                }
            }

            print(f"⏳ Invoking Workflow...")
            final_state = app.invoke(inputs)

            # 获取结果
            retrieved_docs = final_state.get("retrieved_docs", [])

            # 打印部分结果用于人工检查
            print(f"\n📄 Final Retrieved {len(retrieved_docs)} documents.")
            for idx, doc in enumerate(retrieved_docs):  # 只打印前3条避免刷屏
                score = doc.metadata.get('rerank_score', 'N/A')
                print(f"   [Doc {idx + 1}] Source: {doc.metadata.get('source', 'unknown')}")
                print(f"   Title: {doc.metadata.get('title')}")
                print(f"   Snippet: {doc.page_content[:50].replace('\n', ' ')}...")
                print(f"   Score: {score}")

            # 执行验证
            print("🔍 Verifying results...")
            case.verify_func(retrieved_docs)

            print(f"✅ Passed!")
            success_count += 1

        except Exception as e:
            print(f"❌ Test Failed: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"📊 Retrieval Test Summary: {success_count}/{total_count} passed.")
    print("=" * 60)


if __name__ == "__main__":
    retrieval_workflow_test(ALL_RETRIEVAL_SCENARIOS)