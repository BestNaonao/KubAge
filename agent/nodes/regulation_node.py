from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.schemas import SelfEvaluation, ExecutionPlan, PlanAction, EvaluatedStatus, ProblemAnalysis
from agent.state import AgentState

SYSTEM_EVALUATE_PROMPT = """你是一个 Kubernetes Agent 的执行监督者。评估最近的动作执行结果是否符合预期。
当前计划: {plan}
执行结果(Docs/ToolOutput/Error): {result}

判断标准：
1. 如果调用工具报错 -> Fail -> TO_PLANING (尝试修复参数) 或 TO_ANALYSIS (放弃工具)
2. 如果检索结果为空或不相关 -> Fail -> TO_PLANING (重写Query)
3. 如果执行成功且信息充足 -> Pass -> TO_EXPRESSION
4. 如果执行成功但还需要更多信息 -> Needs_Refinement -> TO_PLANING (下一步计划)

输出格式要求:
{format_instructions}
"""


class RegulationNode:
    """
    负责评估执行结果，并管理将工具结果写入历史记录
    TODO: 优化:检查文档是否合适且充足，足够回答或解决问题，如果充足，则反馈文档中的关键信息以及示例，如果不合适给出反馈意见
    """
    def __init__(self, llm):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=SelfEvaluation)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_EVALUATE_PROMPT),
            ("user", "Analysis: {analysis}\nHistory: {history}")
        ]).partial(
            format_instructions=self.parser.get_format_instructions(),
        )
        self.chain = prompt | self.llm | self.parser

    def __call__(self, state: AgentState):
        print("\n⚖️ [Self-Regulation]: Evaluating...")
        # 从 State 获取必要信息
        plan: ExecutionPlan = state.get("plan")
        analysis: ProblemAnalysis = state.get("analysis")
        messages = state.get("messages")

        # 1. 获取最近的执行产出
        if plan.action == PlanAction.RETRIEVE:
            result = f"Retrieved {len(state.get('retrieved_docs', []))} docs"
        elif plan.action == PlanAction.TOOL_USE:
            result = state.get("tool_output")
        else:
            # Direct Answer 不需要 evaluate，直接 pass
            return {"evaluation": SelfEvaluation(
                status="Pass", reasoning="Direct Answer", next_step="Expression", feedback=plan.final_answer
            )}

        # 2. 执行评估 Chain
        eval_result = self.chain.invoke({
            "plan": plan.model_dump(),
            "result": str(result),
            "analysis": analysis.model_dump() if analysis else None,
            "history": messages,
        })

        evaluation = SelfEvaluation(**eval_result)
        updates = {"evaluation": evaluation}

        # 检索成功则重置计数器
        if plan.action == PlanAction.RETRIEVE and evaluation.status == EvaluatedStatus.PASS:
            print("  Retrieval Passed.")
            updates["retrieval_attempts"] = 0

        print(f"   Status: {evaluation.status} -> Next: {evaluation.next_step}")
        return updates