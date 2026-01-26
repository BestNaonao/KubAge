from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.schemas import ExecutionPlan, SelfEvaluation, PlanAction, RiskLevel, ProblemAnalysis, OperationType, \
    EvaluatedStatus
from agent.state import AgentState

SYSTEM_PLANNING_PROMPT = """你是一个 Kubernetes Agent 的【规划大脑（Planning Node）】。
你的职责是：基于【问题分析 Analysis】、【历史对话 History】以及【上一步评估反馈 Feedback】，制定下一步最合理、安全、有效的行动计划。

### 一、 核心规划原则 (Core Planning Principles)
**请严格遵守以下三条原则：**

1. **反馈驱动修正 (Feedback-Driven Correction)**:
   - 如果 `Feedback` 表明存在 Fail / Needs_Refinement，必须分析失败原因。
   - 检索无结果或相关性低 -> 尝试优化 `search_queries` 或换一个方向检索。
   - 工具报错或参数错误 -> 检查 `tool_args` 是否符合 Schema，或检索文档寻找正确用法。

2. **知识检索优先原则 (Knowledge First)**:
   - 在未检索文档，并且你不确定具体命令参数、或 YAML 结构、或最佳实践时，**必须优先选择 `Retrieve`**。
   - **例外**: 只有当历史记录显示**已经进行过充分的检索**且获得了必要信息，才允许跳过检索直接使用工具。

3. **行动与回答 (Action & Answer)**:
   - 只有在信息充足、风险已知的情况下，选择 `Tool_Use` 执行操作。
   - 只有在任务已完成或无需操作即可回答时，选择 `Direct_Answer`。

### 二、 动态指导 (Contextual Guidance)
{dynamic_guidance}

### 三、 行动生成约束 (Generation Constraints)

#### 1. 当 Action = "Retrieve" (检索策略)
将用户的自然语言转化为专业的 K8s 术语，生成 `search_queries`：
- 你将要检索的知识库是《Kubernetes 官方中文文档》。
- **通用化**: 文档中没有用户的具体实体名称。**必须**将具体问题抽象为通用的 Kubernetes 概念或错误类型。**禁止**包含具体实体名（如 pod 名、IP 地址）。
- **语言要求**: 使用 **中文** 描述逻辑，保留 **英文** 专有名词。
- **混合模式**: 最佳结构是 “英文术语 + 中文描述”。
- **示例**:
  - ❌ 错误: ["redis-cart-7d8f 启动失败", "CrashLoopBackOff 怎么修"]
  - ✅ 正确: ["CrashLoopBackOff 排查思路", "Pod 状态 ImagePullBackOff 原因", "Deployment 滚动更新策略"]

#### 2. 当 Action = "Tool_Use" (工具调用)
- `tool_name` 和 `tool_args` 必须严格匹配【可用工具列表】中的定义的 Schema。
- 不允许通过猜测生成参数。如果不了解参数，请先 Retrieve。

#### 3. 当 Action = "Direct_Answer" (直接回答)
- 仅当任务已完成或无需外部信息即可给出最终结论时使用。
- `final_response` 必须包含完整的最终结论，总结之前的检索和操作结果。

### 四、 可用工具列表 (Tools Library)
{tool_descriptions}

### 五、 输入上下文
- **操作类型**: {op_type}
- **风险等级**: {risk_level}
- **上一步反馈**: {feedback}

### 六、输出格式
严格按照以下 JSON 格式输出:
{format_instructions}
"""


class PlanningNode:
    MAX_RETRIEVAL_ATTEMPTS = 3
    def __init__(self, llm, tool_descriptions: str):
        self.llm = llm
        self.tool_descriptions = tool_descriptions
        self.parser = JsonOutputParser(pydantic_object=ExecutionPlan)
        # 注入工具描述
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PLANNING_PROMPT),
            ("user", "Analysis: {analysis}\nHistory: {history}")
        ]).partial(
            format_instructions=self.parser.get_format_instructions(),
            tool_descriptions=self.tool_descriptions
        )
        self.chain = prompt | self.llm | self.parser

        self.retrieve_first_risks = [RiskLevel.HIGH, RiskLevel.CRITICAL]
        self.retrieve_first_ops = [OperationType.DIAGNOSIS, OperationType.KNOWLEDGE_QA, OperationType.CONFIGURE]

    def _generate_dynamic_guidance(self, state: AgentState) -> str:
        """
        根据当前状态动态生成指导语
        """
        analysis: ProblemAnalysis = state.get("analysis")
        plan: ExecutionPlan = state.get("plan")
        evaluation = state.get("evaluation")
        has_docs = bool(state.get("retrieved_docs"))
        retrieval_attempts = state.get("retrieval_attempts")
        guidance = []

        # --- 1. 检索次数熔断机制(最高优先级) ---
        if retrieval_attempts >= self.MAX_RETRIEVAL_ATTEMPTS:
            guidance.append(
                f"  **警告——检索异常**: 已经连续检索{retrieval_attempts}次均未能通过评估。\n"
                "   - **立即降低检索(`Retrieve`)的优先级**\n"
                "   - 策略 A: 如果问题含糊不清，请选择 `Direct_Answer` 向用户反问或澄清。\n"
                "   - 策略 B: 如果可以尝试通用排查命令 (如 `kubectl get events`)，请选择 `Tool_Use`。\n"
                "   - 策略 C: 基于现有信息给出“无法找到确切文档”的保守回答。"
            )
            return "\n".join(guidance)  # 避免被后续规则冲淡

        # --- 2. 常规指导 ---
        if not has_docs:
            guidance.append("  知识状态: 尚未检索任何文档。")
            if analysis and analysis.risk_level in self.retrieve_first_risks:
                guidance.append(f"  风险约束: 操作风险等级={analysis.risk_level.value}，建议优先检索官方文档！")
            if analysis and analysis.target_operation in self.retrieve_first_ops:
                guidance.append(f"  知识性操作: {analysis.target_operation.value}，建议优先检索官方文档。")
        else:
            guidance.append(f"  已获取{len(state.get("retrieved_docs"))}篇相关文档，可优先利用现有知识")
            if plan and plan.action == PlanAction.RETRIEVE and evaluation and evaluation.status != EvaluatedStatus.PASS:
                guidance.append(f"  查询文档有误: {plan.action.value}，建议根据反馈改写search_queries并重新检索。")
            if plan and plan.action == PlanAction.TOOL_USE and evaluation and evaluation.status != EvaluatedStatus.PASS:
                guidance.append(f"  工具调用错误: {plan.action.value}，建议根据反馈改写调用工具名或参数，并重新调用。")

        return "\n".join(guidance) if guidance else "  无特殊约束，按常规流程规划"

    def __call__(self, state: AgentState):
        print("\n🧠 [Planning]: Thinking...")
        messages = state.get("messages")
        analysis = state.get("analysis")

        # 获取上一步的评估反馈，如果有的话
        evaluation: SelfEvaluation = state.get("evaluation")
        feedback = evaluation.feedback if evaluation else "None"

        dynamic_guidance = self._generate_dynamic_guidance(state)

        # 调用链
        try:
            result = self.chain.invoke({
                "analysis": analysis.model_dump(),
                "history": messages,
                "dynamic_guidance": dynamic_guidance,
                "op_type": analysis.target_operation.value,
                "risk_level": analysis.risk_level.value,
                "feedback": feedback
            })

            plan = ExecutionPlan(**result)
            print(f"   Action: {plan.action.value}")
            if plan.action == PlanAction.TOOL_USE:
                print(f"   Target Tool: {plan.tool_name}")

            return {"plan": plan}

        except Exception as e:
            print(f"❌ Planning Error: {e}")
            # 简单的错误恢复：如果解析失败，返回需要人工干预或重试的计划（这里简化处理）
            return {"plan": None, "error": str(e)}