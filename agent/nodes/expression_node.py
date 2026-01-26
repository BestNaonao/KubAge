# agent/nodes/expression_node.py
from langchain_core.messages import AIMessage

from agent.schemas import ExecutionPlan, SelfEvaluation
from agent.state import AgentState


class ExpressionNode:
    def __call__(self, state: AgentState):
        print("\n💬 [Expression]: Generating Response...")
        plan: ExecutionPlan = state.get("plan")
        evaluation: SelfEvaluation = state.get("evaluation")

        # 如果是 Direct Answer，直接使用
        if plan and plan.final_answer:
            response = plan.final_answer
        elif evaluation and evaluation.next_step == "Expression":
            response = evaluation.feedback
        else:
            # 也可以在这里再次调用 LLM 综合 Docs 和 Tool Output 生成回答
            # 简单起见，这里假设 Analysis/Planning 阶段如果决定回答，内容已经生成
            response = "根据之前的步骤，任务已完成。"

        print(f"🤖 User Output: {response}")
        return {
            "generated_response": response,
            "messages": [AIMessage(content=response)]
        }