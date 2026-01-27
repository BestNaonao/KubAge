import uuid

from langchain_core.messages import AIMessage, ToolMessage

from agent.schemas import PlanAction
from agent.state import AgentState
from utils.mcp_manager import MCPToolManager


class ToolCallNode:
    def __init__(self):
        # 获取单例实例
        self.mcp_manager = MCPToolManager.get_instance()

    async def __call__(self, state: AgentState):
        """
        异步执行工具调用，并将交互过程记录到历史消息中
        """
        print("\n🛠️ [ToolCall]: Executing...")
        plan = state.get("plan")

        # 1. 安全检查
        if not plan or plan.action != PlanAction.TOOL_USE:
            return {"tool_output": "Error: Invalid plan for ToolCall node."}

        tool_name = plan.tool_name
        tool_args = plan.tool_args or {}

        # 2. 从 Manager 获取工具函数
        tool_func = self.mcp_manager.get_tool(tool_name)

        if not tool_func:
            error_msg = f"Error: Tool '{tool_name}' not found in MCP registry."
            print(f"   ❌ {error_msg}")
            return {"tool_output": error_msg}

        # 3. 生成唯一的 tool_call_id
        # 因为我们是手动执行 JSON 计划，而不是 LLM 原生生成的 tool_call，
        call_id = str(uuid.uuid4())

        # 4. 构造 AI 调用消息 (伪造的"思考"过程，让历史记录更连贯)
        # 这告诉未来的节点："我刚才决定调用这个工具"
        ai_msg_log = AIMessage(
            content=f"Executing tool: {tool_name}",  # 这里的文本内容可以帮助人类阅读
            tool_calls=[{
                "name": tool_name,
                "args": tool_args,
                "id": call_id,
                "type": "tool_call"
            }]
        )

        try:
            print(f"   Calling: {tool_name} with args: {tool_args}")

            # 5. 执行工具
            # 因为 tool_func 是异步闭包，必须 await
            # MCP 的 call_tool 返回的是 CallToolResult 对象，通常包含 content 列表
            mcp_result = await tool_func(**tool_args)

            # 6. 解析结果 (提取文本内容)
            # 根据 MCP 协议，result.content 是一个列表，通常包含 TextContent 或 ImageContent
            output_text_list = []
            if hasattr(mcp_result, 'content'):
                for item in mcp_result.content:
                    if hasattr(item, 'text'):
                        output_text_list.append(item.text)
                    else:
                        output_text_list.append(str(item))
                final_output = "\n".join(output_text_list)
            else:
                final_output = str(mcp_result)

            print(f"   ✅ Tool Output Length: {len(final_output)}")
            # 截断过长输出，防止撑爆上下文 (可选)
            # if len(final_output) > 5000:
            #     final_output = final_output[:5000] + "...(truncated)"

            # 7. 构造工具结果消息
            # 这告诉未来的节点："这是工具运行的实际结果"
            tool_msg_log = ToolMessage(
                content=final_output,
                tool_call_id=call_id,
                name=tool_name
            )

            # 8. 返回状态更新
            # 注意：这里返回的 messages 列表会被 LangGraph 追加到 state["messages"] 中
            return {
                "tool_output": final_output,  # 供 Regulation 节点即时检查
                "messages": [ai_msg_log, tool_msg_log]  # 供 Analysis/Planning 节点作为长期记忆
            }

        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            print(f"   ❌ {error_msg}")

            # 即使报错，也要记录到历史，防止 Agent 不知道自己已经失败过
            tool_msg_error = ToolMessage(
                content=error_msg,
                tool_call_id=call_id,
                name=tool_name,
                status="error"  # LangChain 新版支持 status 字段
            )

            return {
                "tool_output": error_msg,
                "messages": [ai_msg_log, tool_msg_error]
            }