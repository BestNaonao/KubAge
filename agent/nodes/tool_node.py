from agent.schemas import PlanAction
from utils.mcp_manager import MCPToolManager


class ToolCallNode:
    def __init__(self):
        # 获取单例实例
        self.mcp_manager = MCPToolManager.get_instance()

    async def __call__(self, state: dict):
        """
        异步执行工具调用
        """
        print("\n🛠️ [ToolCall]: Executing...")
        plan = state.get("plan")

        # 安全检查
        if not plan or plan.action != PlanAction.TOOL_USE:
            return {"tool_output": "Error: Invalid plan for ToolCall node."}

        tool_name = plan.tool_name
        tool_args = plan.tool_args or {}

        # 1. 从 Manager 获取工具函数
        tool_func = self.mcp_manager.get_tool(tool_name)

        if not tool_func:
            error_msg = f"Error: Tool '{tool_name}' not found in MCP registry."
            print(f"   ❌ {error_msg}")
            return {"tool_output": error_msg}

        # 2. 执行工具
        try:
            print(f"   Calling: {tool_name} with args: {tool_args}")

            # 因为 tool_func 是异步闭包，必须 await
            # MCP 的 call_tool 返回的是 CallToolResult 对象，通常包含 content 列表
            mcp_result = await tool_func(**tool_args)

            # 3. 解析结果 (提取文本内容)
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

            return {"tool_output": final_output}

        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            print(f"   ❌ {error_msg}")
            return {"tool_output": error_msg}