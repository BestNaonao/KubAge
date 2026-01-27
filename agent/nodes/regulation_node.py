from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.schemas import SelfEvaluation, ExecutionPlan, PlanAction, EvaluatedStatus, ProblemAnalysis, OperationType, \
    NextStep
from agent.state import AgentState

SYSTEM_EVALUATE_PROMPT = """你是一个 Kubernetes Agent 的执行监督者。
你的核心职责是：评估最近的执行结果（检索文档或工具输出）是否可以有效地解决用户的问题，并决定下一步流向。

### 输入信息
1. **原始分析 (Analysis)**: 用户意图和关键实体。
2. **执行计划 (Plan)**: 刚刚执行的动作（Retrieve 或 Tool_Use）。
3. **执行结果 (Result)**: 检索到的文档内容摘要或工具的具体输出。

### 评估标准 (Evaluation Criteria)
{dynamic_criteria}

### 关键决策逻辑 (Decision Logic)
1. **判断下一步方向 (Next Step)**:
   - **TO_PLANING**: 
     - 场景 A (信息不足): 评估状态为 Fail 或 Needs_Refinement，需要重新检索或修复参数。
{to_planning_logic}
   - **TO_EXPRESSION**:
{to_expression_logic}
### 输出格式
严格遵守 JSON 格式:
{format_instructions}
"""

RETRIEVE_CRITERIA = """
#### 当前动作是 RETRIEVE (文档检索)
- **Pass (通过)**: 文档包含了解答问题或**指导后续操作**所需的具体事实（如具体的命令参数或示例、准确的配置 YAML、清晰的排错步骤）或概念知识（针对知识问答型问题）。
  - *Feedback 要求*: 提取文档中的关键信息摘要（例如："文档提供了 CrashLoopBackOff 的三种常见原因..."），必须包含:
     - 命中的核心概念 / 对象
     - 或至少一个可以直接引用的事实或步骤
- **Needs_Refinement (需改进)**: 文档相关，但信息不全或层级不对（例如：只查到了概念定义或原理，没查到具体命令，但用户需要的是操作步骤）。
  - *Feedback 要求*: 指出缺什么（例如："缺少具体的 kubectl 修复命令"）。
- **Fail (失败)**: 
  - 检索结果为空。
  - 文档内容与用户问题无关（例如：用户问 Pod，查出来的全是 Service）。
  - *Feedback 要求*: 建议更换关键词的方向(概念) 或 query表达。"""

TOOL_USE_CRITERIA = """
#### 当前动作是 TOOL_USE (工具调用)
- **Pass**: 工具成功执行，返回了预期的观察结果（即使结果是"资源不存在"，只要符合预期也算 Pass）。
- **Fail**: 工具报错（Error）、参数错误、或输出完全无法解析。"""

RETRIEVE_TO_PLANNING_LOGIC = "     - 场景 B (准备行动): 评估状态为 **Pass**，且用户的最终目标是 **具体操作**，目前执行检索步骤。**必须跳转回 Planning 以计划 Tool_Use。**\n"

TOOL_USE_TO_PLANNING_LOGIC = "     - 场景 C (继续行动): 评估状态为 Pass，但是仍然未满足用户的需求，需要按照步骤继续执行。\n"

RETRIEVE_TO_EXPRESS_LOGIC = "     - 场景 A (纯问答): 评估状态为 Pass，且用户的目标仅仅是 **知识咨询/QA**，不需要对集群或系统进行实质性修改。\n"

TOOL_USE_TO_EXPRESS_LOGIC = "     - 场景 B (任务完成): 所有的工具调用都已成功执行完毕，最后一次工具调用的结果可以证明已满足用户的需求。\n"


class RegulationNode:
    """
    负责评估执行结果并给出反馈
    """
    def __init__(self, llm):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=SelfEvaluation)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_EVALUATE_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("user",
             "Analysis:\n{analysis}\n\n"
             "Target Plan:\n{plan}\n\n"
             "Execution Result:\n{result}"
             )
        ]).partial(
            format_instructions=self.parser.get_format_instructions(),
        )
        self.chain = prompt | self.llm | self.parser

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """
        将文档列表格式化为字符串，供 LLM 审查。
        限制总字符数防止 Context 溢出。
        """
        if not docs:
            return "No documents retrieved."

        formatted = []
        current_chars = 0
        for i, doc in enumerate(docs):
            content = doc.page_content.replace("\n", " ")  # 单篇文档也做截断
            entry = f"[Doc {i + 1}] Title: {doc.metadata.get('title', 'Unknown')}\nContent: {content}..."
            # 内容暂时未做出截断
            formatted.append(entry)
            current_chars += len(entry)

        return "\n\n".join(formatted)

    @staticmethod
    def _generate_dynamic_criteria(action: PlanAction) -> str:
        if action == PlanAction.RETRIEVE:
            return RETRIEVE_CRITERIA
        else:
            return TOOL_USE_CRITERIA

    @staticmethod
    def _generate_dynamic_decision_logic(operation: OperationType, action: PlanAction) -> tuple[str, str]:
        if action == PlanAction.RETRIEVE and operation == OperationType.KNOWLEDGE_QA:
            return "", RETRIEVE_TO_EXPRESS_LOGIC
        elif action == PlanAction.RETRIEVE and operation != OperationType.KNOWLEDGE_QA:
            return RETRIEVE_TO_PLANNING_LOGIC, ""
        else:
            return TOOL_USE_TO_PLANNING_LOGIC, TOOL_USE_TO_EXPRESS_LOGIC

    @staticmethod
    def _evaluation_view(messages):
        # 只保留最近一轮用户问题 + 最近一次 assistant 决策
        return messages[-4:]

    def __call__(self, state: AgentState):
        print("\n⚖️ [Self-Regulation]: Evaluating...")
        # 1. 解包 State 获取必要信息
        plan: ExecutionPlan = state.get("plan")
        analysis: ProblemAnalysis = state.get("analysis")
        messages = state.get("messages")

        if not plan:
            return {"error": "Missing plan in regulation node"}

        # 2. 获取最近的执行产出，准备 Result 上下文
        if plan.action == PlanAction.RETRIEVE:
            docs = state.get("retrieved_docs", [])
            result_context = self._format_docs(docs)
            print(f"   Auditing {len(docs)} documents content...")
        elif plan.action == PlanAction.TOOL_USE:
            tool_output = state.get("tool_output", "")
            result_context = f"Tool Output: {str(tool_output)}"
        else:
            # Direct Answer 不需要 evaluate，直接 pass
            return {"evaluation": SelfEvaluation(
                status="Pass", reasoning="No execution required", next_step="Expression", feedback=plan.final_answer
            )}

        # 3. 执行评估 Chain
        dynamic_criteria = self._generate_dynamic_criteria(plan.action)
        to_planning_logic, to_expression_logic = self._generate_dynamic_decision_logic(analysis.target_operation, plan.action)
        eval_result = self.chain.invoke({
            "dynamic_criteria": dynamic_criteria,
            "to_planning_logic": to_planning_logic,
            "to_expression_logic": to_expression_logic,
            "history": self._evaluation_view(messages),
            "analysis": analysis.model_dump() if analysis else "{}",
            "plan": plan.model_dump(),
            "result": result_context,
        })

        evaluation = SelfEvaluation(**eval_result)
        updates = {"evaluation": evaluation}

        # 5. 状态更新逻辑: 检索成功则重置计数器
        if plan.action == PlanAction.RETRIEVE and evaluation.status == EvaluatedStatus.PASS:
            print("   ✅ Retrieval Passed (Quality Check OK). Resetting counter.")
            updates["retrieval_attempts"] = 0

        # 6. 硬规则约束: 非 QA 问题的检索，不允许直接表达
        if (
                plan.action == PlanAction.RETRIEVE
                and analysis.target_operation != OperationType.KNOWLEDGE_QA
                and evaluation.next_step == NextStep.TO_EXPRESSION
        ):
            evaluation.next_step = NextStep.TO_PLANNING
            evaluation.reasoning += "（强制修正：当前任务为操作型任务，检索完成后需继续规划执行）"

        print(f"   Decision: {evaluation.status} -> Next: {evaluation.next_step}")
        print(f"   Reason: {evaluation.reasoning}")
        print(f"   Feedback: {evaluation.feedback}")

        return updates