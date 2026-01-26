from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.schemas import SelfEvaluation, ExecutionPlan, PlanAction, EvaluatedStatus, ProblemAnalysis
from agent.state import AgentState

SYSTEM_EVALUATE_PROMPT = """你是一个 Kubernetes Agent 的执行监督者。
你的核心职责是：评估最近的执行结果（检索文档或工具输出）是否可以有效地解决用户的问题。

### 输入信息
1. **原始分析 (Analysis)**: 用户意图和关键实体。
2. **执行计划 (Plan)**: 刚刚执行的动作（Retrieve 或 Tool_Use）。
3. **执行结果 (Result)**: 检索到的文档内容摘要或工具的具体输出。

### 评估标准 (Evaluation Criteria)

#### A. 如果动作是 RETRIEVE (文档检索)
- **Pass (通过)**: 文档包含了解答问题所需的**具体事实**（如具体的命令参数、准确的配置 YAML、清晰的排错步骤）或概念知识（知识问答型问题）。
  - *Feedback 要求*: 提取文档中的关键信息摘要（例如："文档提供了 CrashLoopBackOff 的三种常见原因..."），包括示例。
- **Needs_Refinement (需改进)**: 文档相关但信息不全（例如：只查到了概念定义，没查到具体命令，但用户问的是操作步骤）。
  - *Feedback 要求*: 指出缺什么（例如："缺少具体的 kubectl 修复命令"）。
- **Fail (失败)**: 
  - 检索结果为空。
  - 文档内容与用户问题风马牛不相及（例如：用户问 Pod，查出来的全是 Service）。
  - *Feedback 要求*: 建议更换关键词的方向。

#### B. 如果动作是 TOOL_USE (工具调用)
- **Pass**: 工具成功执行，返回了预期的观察结果（即使结果是"资源不存在"，只要符合预期也算 Pass）。
- **Fail**: 工具报错（Error）、参数错误、或输出完全无法解析。

### 决策逻辑
- **Pass** -> Next Step: `Expression` (可以直接回答) 或 `Planning` (继续下一步动作)
- **Needs_Refinement** / **Fail** -> Next Step: `Planning` (需要重新规划 Query 或参数)

### 输出格式
严格遵守 JSON 格式:
{format_instructions}
"""


class RegulationNode:
    """
    负责评估执行结果并给出反馈
    TODO: 用MessagePlaceHolder代替history
    """
    def __init__(self, llm):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=SelfEvaluation)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_EVALUATE_PROMPT),
            ("user", "Analysis: {analysis}\nHistory: {history}\n\nTarget Plan: {plan}\nExecution Result: {result}")
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
                status="Pass", reasoning="Direct Answer", next_step="Expression", feedback=plan.final_answer
            )}

        # 3. 执行评估 Chain
        eval_result = self.chain.invoke({
            "plan": plan.model_dump(),
            "result": result_context,
            "analysis": analysis.model_dump() if analysis else "{}",
            "history": messages,
        })

        evaluation = SelfEvaluation(**eval_result)
        updates = {"evaluation": evaluation}

        # 5. 状态更新逻辑: 检索成功则重置计数器
        if plan.action == PlanAction.RETRIEVE and evaluation.status == EvaluatedStatus.PASS:
            print("   ✅ Retrieval Passed (Quality Check OK). Resetting counter.")
            updates["retrieval_attempts"] = 0

        print(f"   Decision: {evaluation.status} -> Next: {evaluation.next_step}")
        print(f"   Reason: {evaluation.reasoning}")
        return updates