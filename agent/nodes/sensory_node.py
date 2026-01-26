from typing import Dict, Any

from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables import RunnableConfig

from agent.state import AgentState


class SensoryNode:
    def __init__(self):
        load_dotenv(find_dotenv())

    def __call__(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        # 在LangGraph中，如果是invoke传入的input，通常已经处理了
        # 这里可以做一些前置格式化或者日志记录
        print("\n👂 [Sensory]: Received Input")
        return {}   # Pass through
