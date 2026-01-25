from typing import Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from agent.schemas import ProblemAnalysis
from agent.state import AgentState


ANALYSIS_SYSTEM_PROMPT = """
你是一个 Kubernetes 运维专家 Agent。你的任务是基于 **对话历史** 和 **用户最新回复** 进行深度分析。

# 核心原则(Core Rules)
1. **指令优先级原则**
 * 最新指令优先: 用户的最新意图可以推翻之前的上下文。当用户明确表达新的指令时，应该以最新的指令为准。
2. **权限与责任原则**
 * 用户高于AI: AI作为辅助工具，最终决策权归于用户。
3. **风险评估原则**
 * 风险定级标准:
   - **Low**: 仅询问概念、查询状态、获取日志等只读操作 (ReadOnly)。
   - **Medium**: 检查配置、执行不改变状态的调试命令。
   - **High/Critical**: 如 修改、删除、重启、回滚 等可能影响系统状态的操作。

# 思维步骤 (Chain of Thought)
1. **上下文消歧**: 检查用户输入中是否存在代词（如“它”、“那个 pod”）。如果存在，请结合历史消息找到用户讨论的具体实体类型和名称。
2. **意图识别**: 确定用户想要做什么（如:查询、修改、删除、排错等）。
3. **关键信息提取**: 提取 K8s 资源名称、Namespace、错误代码等。
4. **风险评估**: 评估该操作对生产环境的潜在影响。

# 输出格式要求 (Output Format Instructions)
{format_instructions}
"""


class AnalysisNode:
    def __init__(self, llm):
        """
        在初始化阶段加载 LLM 和 Schema，只执行一次
        """
        self.llm = llm
        # 1. 定义解析器 (它会自动处理 Markdown 和不完整的 JSON)
        self.parser = JsonOutputParser(pydantic_object=ProblemAnalysis)

        # 2. 获取 Prompt 并注入 format_instructions
        prompt = ChatPromptTemplate.from_messages([
            ("system", ANALYSIS_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{current_input}")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        # 3. 组装 Chain
        self.chain = prompt | self.llm | self.parser

    def __call__(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        节点被调用时的逻辑
        """
        messages = state.get("messages", [])
        if not messages:
            return {"analysis": None}

        current_input = messages[-1].content
        history = messages[:-1]

        # 调用预定义好的 Chain
        # config 参数允许透传 trace_id, callbacks 等信息
        try:
            # invoke 的结果现在直接是字典 (Dict)，因为 parser 已经转好了
            analysis_dict = self.chain.invoke(
                {
                    "history": history,
                    "current_input": current_input,
                },
                config=config
            )

            # 4. 手动转为 Pydantic 对象 (进行二次校验)
            # 这一步是为了确保类型安全，如果模型漏字段，这里会报错
            analysis_result = ProblemAnalysis(**analysis_dict)

            return {"analysis": analysis_result}

        except Exception as e:
            print(f"❌ [Analysis Node Error]: {e}")
            return {"analysis": None}

    def prompt_preview(self, current_input, history):
        preview_prompt = self.chain.steps[0].format(
            sys_prompt_content=ANALYSIS_SYSTEM_PROMPT,  # 假设你用了变量注入
            history=history,
            current_input=current_input
        )
        print("\n" + "=" * 30 + " PROMPT PREVIEW " + "=" * 30)
        # 注意：History 在 preview 中可能是 list 对象，打印出来可能只显示对象地址
        # 但你可以检查 current_input 是否在最后
        print(preview_prompt)
        print("=" * 76 + "\n")