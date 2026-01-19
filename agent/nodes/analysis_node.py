from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig

from agent.prompts import get_analysis_prompt, ANALYSIS_SYSTEM_PROMPT
from agent.schemas import ProblemAnalysis


class AnalysisNode:
    def __init__(self, llm):
        """
        在初始化阶段加载 LLM 和 Schema，只执行一次
        """
        self.llm = llm
        # 1. 定义解析器 (它会自动处理 Markdown 和不完整的 JSON)
        self.parser = JsonOutputParser(pydantic_object=ProblemAnalysis)

        # 2. 获取 Prompt 并注入 format_instructions
        prompt = get_analysis_prompt().partial(format_instructions=self.parser.get_format_instructions())

        # 3. 组装 Chain
        self.chain = prompt | self.llm | self.parser

    def __call__(self, state: dict, config: RunnableConfig):
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