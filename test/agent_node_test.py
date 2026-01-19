from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agent.nodes.analysis_node import AnalysisNode
from agent.state import AgentState
from utils.llm_factory import get_chat_model


def analysis_test_inputs():
    # 1. 模拟 State

    # 场景：用户先聊到了一个特定的 Nginx Pod，然后突然说“把它删了”
    # 这测试了：历史上下文融合、歧义消除、风险识别

    inputs = {
        "messages": [
            HumanMessage(content="我的 nginx-frontend-7b8c9 这里的 Pod 状态一直是 CrashLoopBackOff，怎么办？"),
            AIMessage(content="CrashLoopBackOff 通常意味着容器启动后立即退出。您可以检查一下日志或配置。"),
            HumanMessage(content="太麻烦了，直接帮我把它删了，让 Deployment 重启一个新的。")
        ]
    }

    inputs = {
        "messages": [
            HumanMessage(content="你好，我发现 payment-service namespace 下的 redis-cache 节点好像挂了。"),
            AIMessage(content="收到，我会帮您排查 redis-cache 的问题。请问具体表现是什么？"),
            HumanMessage(content="它一直在重启，状态显示 CrashLoopBackOff。请帮我分析一下原因并给出修复建议。")
        ]
    }


def analysis_workflow_test():
    print("🚀 Starting Analysis Node Workflow Test...")

    # ==========================================
    # 1. 构建图 (Build the Graph)
    # ==========================================

    # 初始化大模型
    llm = get_chat_model(
        temperature=0.5,
        extra_body={
            "top_k": 50,
            "thinking_budget": 8192,
        }
    )

    # 初始化 StateGraph
    workflow = StateGraph(AgentState)

    # 实例化节点 (Class-based Node)
    # 这里可以在测试中传入特定的 model_name，例如 "gpt-3.5-turbo" 以节省成本
    analysis_node = AnalysisNode(llm)

    # 添加节点
    workflow.add_node("analyze_problem", analysis_node)

    # 设置边 (Edges)
    # 这是一个单节点测试： Start -> Analysis -> End
    workflow.add_edge(START, "analyze_problem")
    workflow.add_edge("analyze_problem", END)

    # 编译图 (Compile)
    app = workflow.compile()

    # (可选) 打印图的结构，确认连线正确
    # print(app.get_graph().draw_ascii())

    # ==========================================
    # 2. 准备测试数据 (Mock Data)
    # ==========================================

    # 模拟场景：用户想查看某个具体的 Pod 为何崩溃
    # 注意：这里我们故意制造一些指代不明 ("它")，看 LLM 能否结合历史识别

    inputs = {
        "messages": [
            HumanMessage(content="我的 Pod 昨天还能用，今天突然连不上了。"),
            AIMessage(content="请问能提供一下具体的 Pod 名称和 Namespace 吗？"),
            HumanMessage(content="是 default 命名空间下的 redis-cart。")
        ]
    }

    # ==========================================
    # 3. 运行工作流 (Invoke Workflow)
    # ==========================================
    print("\n⏳ Invoking Workflow...")

    # invoke 会返回最终的 State 快照
    final_state = app.invoke(inputs)

    # ==========================================
    # 4. 验证结果 (Verification)
    # ==========================================

    analysis_result = final_state.get("analysis")
    print(analysis_result)

    if analysis_result:
        print("\n✅ Workflow Execution Succeeded!")
        print("=" * 60)

        # 1. 验证思维链 (Reasoning)
        print(f"🧠 [Reasoning]:\n{analysis_result.reasoning}\n")

        # 2. 验证意图和实体
        print(f"🎯 [Target Operation]: {analysis_result.target_operation}")
        print(f"📦 [Entities]: {[f'{e.type}:{e.name}' for e in analysis_result.entities]}")

        # 3. 验证生成的检索词 (Queries)
        print(f"🔍 [Search Queries]:")
        for q in analysis_result.search_queries:
            print(f"  - {q}")

        # 4. 验证风险等级
        print(f"⚠️ [Risk Level]: {analysis_result.risk_level}")

        # 断言检查 (自动化测试用)
        assert "redis-cart" in str(analysis_result.entities), "Entity extraction failed"
        assert analysis_result.risk_level in ["Medium", "Low"], "Risk assessment seems off"

    else:
        print("\n❌ Workflow Execution Failed: Analysis result is empty.")


if __name__ == "__main__":
    analysis_workflow_test()