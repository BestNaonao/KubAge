from typing import List
from langgraph.constants import START, END
from langgraph.graph import StateGraph

# 假设这些是你项目中的实际模块
from agent.nodes.analysis_node import AnalysisNode
from agent.state import AgentState
from utils.model_factory import get_chat_model
from test_dataset.analysis_cases import ALL_SCENARIOS, AnalysisTestScenario


def analysis_workflow_test(scenarios: List[AnalysisTestScenario]):
    print("🚀 Starting Analysis Node Workflow Test Batch...")

    # ==========================================
    # 1. 构建图 (Build the Graph - 只需构建一次)
    # ==========================================

    # 初始化大模型
    llm = get_chat_model(
        temperature=0.1,  # 测试时建议降低温度以获得更稳定的结果
        extra_body={
            "top_k": 20,
            "thinking_budget": 8192,
        }
    )

    # 初始化 StateGraph
    workflow = StateGraph(AgentState)
    analysis_node = AnalysisNode(llm)

    workflow.add_node("analyze_problem", analysis_node)
    workflow.add_edge(START, "analyze_problem")
    workflow.add_edge("analyze_problem", END)

    app = workflow.compile()

    # ==========================================
    # 2. 循环运行测试用例
    # ==========================================

    success_count = 0
    total_count = len(scenarios)

    for i, case in enumerate(scenarios, 1):
        print(f"\n{'=' * 20} Test Case {i}/{total_count}: {case.name} {'=' * 20}")

        try:
            # 运行工作流
            print(f"⏳ Invoking Workflow for: {case.name}...")
            final_state = app.invoke(case.user_inputs)
            analysis_result = final_state.get("analysis")

            if not analysis_result:
                print(f"❌ Failed: Analysis result is empty.")
                continue

            print("\n✅ Workflow Execution Succeeded!")
            print("=" * 60)

            # 1. 验证思维链 (Reasoning)和技术摘要
            print(f"🧠 [Reasoning]:\n{analysis_result.reasoning}")
            print(f"🔧 [Technical Summary]:\n{analysis_result.technical_summary}\n")

            # 2. 验证意图和实体
            print(f"🎯 [Target Operation]: {analysis_result.target_operation}")
            print(f"📦 [Entities]: {[f'{e.type}:{e.name}' for e in analysis_result.entities]}")

            # 3. 验证风险等级
            print(f"⚠️ [Risk Level]: {analysis_result.risk_level}")

            # 4. 验证生成的检索词 (Queries)
            print(f"🔍 [Search Queries]:")
            for q in analysis_result.search_queries:
                print(f"  - {q}")

            # 5. 验证追问问题
            print(f"❓ [Clarification Question]: {analysis_result.clarification_question}")

            # 执行自定义断言验证
            print("🔍 Verifying results...")
            case.verify_func(analysis_result)

            print(f"✅ Passed!")
            success_count += 1

        except AssertionError as e:
            print(f"❌ Assertion Failed: {str(e)}")
        except Exception as e:
            print(f"❌ Runtime Error: {str(e)}")

    print("\n" + "=" * 60)
    print(f"📊 Test Summary: {success_count}/{total_count} passed.")
    print("=" * 60)


if __name__ == "__main__":
    analysis_workflow_test(ALL_SCENARIOS)