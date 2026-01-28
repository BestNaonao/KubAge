from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.prompts import format_docs
from agent.schemas import ExecutionPlan, SelfEvaluation, PlanAction, RiskLevel, ProblemAnalysis, OperationType, \
    EvaluatedStatus, analysis_view, evaluation_view, plan_view
from agent.state import AgentState


# =============================================================================
# 1. 模块化 Prompt 片段 (Modular Prompt Fragments)
# =============================================================================

RETRIEVE_DESCRIPTION = """- **目标**: 将用户的自然语言转化为专业的 Kubernetes 术语，生成 `search_queries`，这将用于在《Kubernetes 官方中文文档》中检索。
- **Query 规范**:
   - **通用化**: 必须将具体问题抽象为通用的 Kubernetes 概念或错误类型。**禁止**包含具体实体名（如 pod 名、IP 地址）。
   - **语言要求**: 必须使用 **中文** 描述逻辑，保留 **英文** 专有名词，最佳结构是 “英文术语 + 中文描述”。
   - **示例**:
     - ❌ 错误: ["redis-cart-7d8f 启动失败", "CrashLoopBackOff 怎么修"]
     - ✅ 正确: ["CrashLoopBackOff 排查思路", "Pod 状态 ImagePullBackOff 原因", "Deployment 滚动更新策略"]"""

# A. 基础 System Prompt (所有情况通用)
BASE_SYSTEM_PLAN_PROMPT = """你是 Kubernetes 智能运维系统中的【规划模块】。
你的职责是：基于【历史对话】和用户的【问题分析】、【动态指导】、【文档知识】、【上轮计划】、【步骤反馈】，制定下一步最合理、安全、有效的行动计划。

### 核心规划原则
1. **反馈驱动**: 必须基于 Evaluation 的反馈调整策略，严禁无视错误继续重复尝试。
2. **安全第一**: 高风险操作前必须确保信息充足。
3. **谨慎回答**: 只有历史消息中的工具消息足以证明任务已完成，或检索的文档知识可以回答用户的概念性问题时，才可以选择 `Direct_Answer`。
4. **环境意识**: 注意环境和上下文中的信息。
{dynamic_system_instructions}
### 输出格式
严格按照以下 JSON 格式要求输出:
{format_instructions}
"""

# B. 强制检索模式 (Force Retrieve Mode)
# 适用于：首次进入且需要查文档，或检索失败需要重试
FORCE_RETRIEVE_INSTRUCTIONS = """
### 当前阶段约束 (Current Phase Constraints)
**你正处于【强制检索阶段】**
由于任务涉及高风险操作、或属于知识密集型任务、或上一次检索失败，你**只能**选择 `Retrieve` 动作。

### 行动生成约束
**Action 必须为 "Retrieve"**
{retrieve_description}

### 可用工具
(当前阶段禁用所有工具，请专注于构建高质量的检索 Query)
"""

# C. 强制修复工具模式 (Force Tool Fix Mode)
# 适用于：工具调用失败，需要修复参数
FORCE_TOOL_FIX_INSTRUCTIONS = """
### 当前阶段约束 (Current Phase Constraints)
**你正处于【工具修复阶段】**
上轮工具调用失败。你必须分析 Feedback 中的错误信息，修正参数。

### 行动选择
1. **Tool_Use (优先)**: 如果你能根据报错信息修正参数（如修复 URL、更改参数格式），请再次调用工具。
- `tool_name` 和 `tool_args` 必须严格匹配【可用工具列表】中的定义的 Schema。
2. **Retrieve (备选)**: 如果报错信息表明你完全缺乏相关知识（如 "Command not found" 或 "Unknown field"），请转为检索。
{retrieve_description}

### 可用工具列表
{tool_descriptions}
"""

# D. 标准模式 (Standard Mode)
# 适用于：自由规划阶段
STANDARD_INSTRUCTIONS = """
### 行动选择规则
请根据当前信息充足度和风险等级选择：

1. **Retrieve**:
- **前提**: 发现现有知识不足以支持安全操作，或需要验证命令参数或 YAML 结构
{retrieve_description}

2. **Tool_Use**:
- **前提**: 信息充足、风险已知、Schema 匹配。
- `tool_name` 和 `tool_args` 必须严格匹配【可用工具列表】中的定义的 Schema。

3. **Direct_Answer**:
- 任务已完成，或无需操作即可回答。
- `final_response` 必须包含完整的最终结论，总结之前的检索和操作结果。

### 可用工具列表
{tool_descriptions}
"""

USER_PLANNING_PROMPT = """
### 问题分析:
{analysis}

### 动态指导:
{dynamic_guidance}

### 文档知识:
{documents}

### 上轮计划:
{former_plan}

### 步骤反馈:
{feedback}
"""


class PlanningNode:
    MAX_RETRIEVAL_ATTEMPTS = 3

    def __init__(self, llm, tool_descriptions: str):
        self.llm = llm
        self.tool_descriptions = tool_descriptions
        self.parser = JsonOutputParser(pydantic_object=ExecutionPlan)

        # 预定义的静态配置
        self.retrieve_first_risks = [RiskLevel.HIGH, RiskLevel.CRITICAL]
        self.retrieve_first_ops = [OperationType.DIAGNOSIS, OperationType.KNOWLEDGE_QA, OperationType.CONFIGURE]

        # 注入工具描述
        prompt = ChatPromptTemplate.from_messages([
            ("system", BASE_SYSTEM_PLAN_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("user", USER_PLANNING_PROMPT)
        ]).partial(
            format_instructions=self.parser.get_format_instructions()
        )
        self.chain = prompt | self.llm | self.parser

    def _generate_dynamic_system_instructions(self, state: AgentState) -> str:
        """
        核心逻辑：根据状态构建动态系统指令
        """
        analysis: ProblemAnalysis = state.get("analysis")
        plan: ExecutionPlan = state.get("plan")
        evaluation: SelfEvaluation = state.get("evaluation")
        has_docs = bool(state.get("retrieved_docs"))

        # 状态机逻辑判断
        mode = "STANDARD"  # 默认为标准模式

        # 1. 首次进入 (Plan/Eval 为空)
        if not plan or not evaluation:
            is_risky = analysis.risk_level in self.retrieve_first_risks
            is_knowledge_op = analysis.target_operation in self.retrieve_first_ops

            # 如果是知识型操作或高风险操作 -> 强制检索
            if is_knowledge_op or is_risky:
                mode = "FORCE_RETRIEVE"
            else:
                mode = "STANDARD"

        # 2. 已检索，已评估 (Feedback Loop)
        elif plan.action == PlanAction.RETRIEVE:
            if evaluation.status == EvaluatedStatus.PASS:
                # 检索成功 -> 进入标准模式 (可以开始思考用工具了)
                mode = "STANDARD"
            else:
                # 检索失败 -> 强制继续检索 (重写 Query)
                mode = "FORCE_RETRIEVE"

        # 3. 已调工具，已评估
        elif plan.action == PlanAction.TOOL_USE:
            if evaluation.status == EvaluatedStatus.PASS:
                # 工具成功 -> 标准模式
                mode = "STANDARD"
            else:
                # 工具失败 -> 修复模式
                mode = "FORCE_TOOL_FIX"

        # 4. 未检索，但调过工具 (Fail 情况)
        elif not has_docs and plan.action == PlanAction.TOOL_USE and evaluation.status != EvaluatedStatus.PASS:
            # 没查文档就瞎调工具还报错了 -> 强制检索！
            mode = "FORCE_RETRIEVE"

        # --- 组装 System Instruction ---
        if mode == "FORCE_RETRIEVE":
            return FORCE_RETRIEVE_INSTRUCTIONS.format(
                retrieve_description=RETRIEVE_DESCRIPTION
            )

        elif mode == "FORCE_TOOL_FIX":
            return FORCE_TOOL_FIX_INSTRUCTIONS.format(
                retrieve_description=RETRIEVE_DESCRIPTION,
                tool_descriptions=self.tool_descriptions
            )

        else:  # STANDARD
            return STANDARD_INSTRUCTIONS.format(
                retrieve_description=RETRIEVE_DESCRIPTION,
                tool_descriptions=self.tool_descriptions
            )

    def _generate_dynamic_guidance(self, state: AgentState) -> str:
        """
        生成简短的指导语，作为 USER_PROMPT 的一部分
        """
        retrieval_attempts = state.get("retrieval_attempts", 0)
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

        return "\n".join(guidance) if guidance else "  无特殊约束，按常规流程规划"

    def __call__(self, state: AgentState):
        print("\n🧠 [Planning]: Thinking...")
        messages = state.get("messages")
        analysis = state.get("analysis")
        documents = state.get("retrieved_docs")
        plan = state.get("plan")
        evaluation: SelfEvaluation = state.get("evaluation")

        dynamic_system_instruction = self._generate_dynamic_system_instructions(state)
        dynamic_guidance = self._generate_dynamic_guidance(state)

        for message in messages:
            message.pretty_print()  # 临时调试使用

        # 调用链
        try:
            result = self.chain.invoke({
                "dynamic_system_instructions": dynamic_system_instruction,
                "history": messages,
                "analysis": analysis_view(analysis),
                "dynamic_guidance": dynamic_guidance,
                "documents": format_docs(documents),
                "former_plan": plan_view(plan),
                "feedback": evaluation_view(evaluation),
            })

            new_plan = ExecutionPlan(**result)
            print(f"   Reasoning: {new_plan.reasoning}")
            print(f"   Action: {new_plan.action.value}")
            if new_plan.action == PlanAction.TOOL_USE:
                print(f"   Target Tool: {new_plan.tool_name}")

            return {"plan": new_plan}

        except Exception as e:
            print(f"❌ Planning Error: {e}")
            # 简单的错误恢复：如果解析失败，返回需要人工干预或重试的计划（这里简化处理）
            return {"plan": None, "error": str(e)}