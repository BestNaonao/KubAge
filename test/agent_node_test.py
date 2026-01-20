from typing import Callable, List, Dict, Any
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph

# 假设这些是你项目中的实际模块
from agent.nodes.analysis_node import AnalysisNode
from agent.schemas import OperationType, RiskLevel
from agent.state import AgentState
from utils.llm_factory import get_chat_model


# ==========================================
# 1. 定义测试用例结构
# ==========================================
@dataclass
class TestScenario:
    name: str
    inputs: Dict[str, Any]
    # 验证函数接收 analysis_result，如果验证通过返回 None，失败抛出 AssertionError
    verify_func: Callable[[Any], None]


def analysis_workflow_test(scenarios: List[TestScenario]):
    print("🚀 Starting Analysis Node Workflow Test Batch...")

    # ==========================================
    # 2. 构建图 (Build the Graph - 只需构建一次)
    # ==========================================

    # 初始化大模型
    llm = get_chat_model(
        temperature=0.1,  # 测试时建议降低温度以获得更稳定的结果
        extra_body={
            "top_k": 50,
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
    # 3. 循环运行测试用例
    # ==========================================

    success_count = 0
    total_count = len(scenarios)

    for i, case in enumerate(scenarios, 1):
        print(f"\n{'=' * 20} Test Case {i}/{total_count}: {case.name} {'=' * 20}")

        try:
            # 运行工作流
            print(f"⏳ Invoking Workflow for: {case.name}...")
            final_state = app.invoke(case.inputs)
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


def main():
    # ==========================================
    # 4. 准备测试数据与验证逻辑
    # ==========================================

    # --- 验证逻辑函数定义 ---

    def verify_case_1(result):
        """
        场景: 昨天还能用，今天连不上 -> default/redis-cart
        预期: 诊断(Diagnosis) 或 资源查询(Resource_Inquiry)
        风险: Medium (因为涉及故障排查) 或 Low
        """
        # 1. 验证实体
        entities_str = str(result.entities)
        assert "redis-cart" in entities_str, f"Missing entity 'redis-cart', got: {entities_str}"

        # 2. 验证意图
        # 用户说"连不上了"，通常属于故障诊断，或者是查询状态
        valid_ops = [OperationType.DIAGNOSIS, OperationType.RESOURCE_INQUIRY]
        assert result.target_operation in valid_ops, \
            f"Expected Diagnosis/Inquiry, got: {result.target_operation}"

        # 3. 验证风险
        # 只是询问并没有修改，应该是 Low 或 Medium
        valid_risks = [RiskLevel.LOW, RiskLevel.MEDIUM]
        assert result.risk_level in valid_risks, \
            f"Expected Low/Medium risk, got: {result.risk_level}"

    def verify_case_2(result):
        """
        场景: 把它删了，让 Deployment 重启一个新的
        预期: 资源删除(Resource_Deletion) 或 重启(Restart)
        风险: High 或 Critical (绝对不能是 Low/Medium)
        """
        # 1. 验证指代消除 (它 -> nginx-frontend...)
        entities_str = str(result.entities)
        assert "nginx-frontend" in entities_str, \
            f"Failed to resolve pronoun 'it' to 'nginx-frontend', got: {entities_str}"

        # 2. 验证意图 (核心测试点)
        # 用户明确说了 "删了" (Deletion) 或者是为了 "重启" (Restart)
        # 根据你的新 Prompt，这应该被识别为特定操作，而不是笼统的 Dangerous
        valid_ops = [OperationType.RESOURCE_DELETION, OperationType.RESTART]
        assert result.target_operation in valid_ops, \
            f"Expected Resource_Deletion or Restart, got: {result.target_operation}"

        # 3. 验证风险 (核心测试点)
        # Prompt 中明确规定：删除/重启 = High/Critical
        valid_risks = [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert result.risk_level in valid_risks, \
            f"CRITICAL FAILURE: Deletion detected as {result.risk_level}, expected High/Critical"


    def verify_case_3(result):
        """
        场景: payment-service / redis-cache 挂了 -> 分析原因
        预期: 故障诊断 (Diagnosis)
        风险: Medium (涉及分析)
        """
        # 1. 验证完整实体提取
        entities_str = str(result.entities)
        assert "payment-service" in entities_str, "Namespace missing"
        assert "redis-cache" in entities_str, "Pod name missing"

        # 2. 验证意图
        # "帮我分析一下原因" -> 强烈的 Diagnosis 信号
        assert result.target_operation == OperationType.DIAGNOSIS, \
            f"Expected Diagnosis, got: {result.target_operation}"

    # --- 组装测试用例 ---

    scenarios = [
        TestScenario(
            name="Contextual Entity Extraction (Redis Connection)",
            inputs={
                "messages": [
                    HumanMessage(content="我的 Pod 昨天还能用，今天突然连不上了。"),
                    AIMessage(content="请问能提供一下具体的 Pod 名称和 Namespace 吗？"),
                    HumanMessage(content="是 default 命名空间下的 redis-cart。")
                ]
            },
            verify_func=verify_case_1
        ),
        TestScenario(
            name="Ambiguity & High Risk Operation (Delete Nginx)",
            inputs={
                "messages": [
                    HumanMessage(content="我的 nginx-frontend-7b8c9 这里的 Pod 状态一直是 CrashLoopBackOff，怎么办？"),
                    AIMessage(content="CrashLoopBackOff 通常意味着容器启动后立即退出。您可以检查一下日志或配置。"),
                    HumanMessage(content="太麻烦了，直接帮我把它删了，让 Deployment 重启一个新的。")
                ]
            },
            verify_func=verify_case_2
        ),
        TestScenario(
            name="Cross-Namespace Analysis (Payment Redis)",
            inputs={
                "messages": [
                    HumanMessage(content="你好，我发现 payment-service namespace 下的 redis-cache 节点好像挂了。"),
                    AIMessage(content="收到，我会帮您排查 redis-cache 的问题。请问具体表现是什么？"),
                    HumanMessage(content="它一直在重启，状态显示 CrashLoopBackOff。请帮我分析一下原因并给出修复建议。")
                ]
            },
            verify_func=verify_case_3
        )
    ]

    # 运行测试
    analysis_workflow_test(scenarios)

if __name__ == "__main__":
    main()