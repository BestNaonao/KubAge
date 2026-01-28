import json
import os
import platform
from typing import Dict, Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.state import AgentState


class SensoryNode:
    def __init__(self, config_path: str):
        self.system_info_label = "【用户环境上下文】"
        self.system_info_str = self._get_static_system_info(config_path)


    def _get_static_system_info(self, config_path) -> str:
        """
        获取本地静态系统环境信息
        """
        os_type = platform.system()
        os_release = platform.release()
        os_arch = platform.machine()
        # 假设 workspace 位于项目根目录下的 workspace 文件夹
        # 你可以根据实际配置读取 config.json 或环境变量
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                workspace_path = config.get("workspace_path", os.path.abspath(os.path.join(os.getcwd(), "workspace")))
        except FileNotFoundError:
            workspace_path = os.path.abspath(os.path.join(os.getcwd(), "workspace"))

        # 构造环境上下文 Prompt
        info = (
            f"{self.system_info_label}\n"
            f"- 操作系统: {os_type} {os_release}\n"
            f"- 硬件架构: {os_arch}\n"
            f"- Workspace Root: {workspace_path}\n"
            f"- 当前工作目录: {os.getcwd()}\n"
            f"注意：所有文件操作和命令执行默认基于 Workspace Root 或当前目录。"
        )
        return info

    def __call__(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        感知节点：负责接收输入，并将环境上下文注入到历史记录中
        """
        print("\n👂 [Sensory]: Processing Input...")
        messages = state.get("messages", [])

        # 1. 检查是否已经注入过环境信息
        # 我们约定：环境信息作为 SystemMessage 存在，且包含特定的标记
        has_env_context = any(
            isinstance(m, SystemMessage) and self.system_info_label in m.content
            for m in messages
        )

        updates = {}

        # 2. 如果没有注入过，则插入到最前面 (或者追加到 System Prompt 之后)
        if not has_env_context:
            print("   ✨ Injecting System Environment Context to Memory...")
            env_message = SystemMessage(content=self.system_info_str)

            # 策略 A: 插入到消息列表的开头 (推荐，作为长期记忆的基础)
            # 注意：LangGraph 的 state["messages"] 通常是 append-only 的，
            # 如果使用 operator.add，这里返回 [env_message] 会追加到末尾。
            # 为了让它生效，我们需要确保 LLM 能看到它。
            # 如果你的 Graph state 定义是 Annotated[List, add_messages]，
            # 直接返回 messages=[env_message] 会追加。
            # 如果想"插队"到最前面，通常需要在 Graph 初始化时做，或者在这里处理。

            # 这里简单起见，我们追加它。对于 LLM 来说，位置在开头还是中间通常都能看见，
            # 但作为 SystemMessage，最好在 HumanMessage 之前。

            # 如果 state["messages"] 已经有用户输入，我们需要把这个 SystemMessage 放在用户输入之前吗？
            # 在 LangGraph 中，返回的 messages 会被追加。
            # 如果这是第一轮，用户的输入还在 input 阶段，可能还没进 messages (取决于你的 Graph 结构)。
            # 假设 Sensory 是第一个节点，state["messages"] 可能包含用户的 HumanMessage (如果通过 input 传入)。

            # 修正策略：直接返回包含环境信息的 SystemMessage。
            # 大模型通常能处理乱序的 SystemMessage，只要它在 Context window 里。
            state.get("messages").insert(0, env_message)

        return updates
