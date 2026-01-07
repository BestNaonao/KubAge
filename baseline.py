import asyncio
import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import Tool
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END, add_messages
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ==================== 环境变量加载 ====================
load_dotenv(find_dotenv())
API_KEY = os.getenv('OPENAI_API_KEY')
BASE_URL = os.getenv('OPENAI_BASE_URL')
MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')

for var in [API_KEY, BASE_URL, MODEL_NAME]:
    if var is None:
        raise ValueError(f"环境变量缺失：{['API_KEY', 'BASE_URL', 'MODEL_NAME'][[API_KEY, BASE_URL, MODEL_NAME].index(var)]}")
    print(f"{var} loaded (type: {type(var)})")

os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ==================== 自定义状态类型 ====================
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ==================== 异步加载 MCP 工具 ====================
async def get_k8s_tools():
    LOCAL_MCP_PATH = "../mcp-server-kubernetes"
    ENTRY_FILE = os.path.join(LOCAL_MCP_PATH, "dist", "index.js")

    if not os.path.exists(ENTRY_FILE):
        raise FileNotFoundError(f"找不到入口文件: {ENTRY_FILE}，请确认您是否执行了 npm run build")

    server_params = StdioServerParameters(
        command="node",
        args=[ENTRY_FILE],
        env={**os.environ}
    )

    print(f"正在连接本地 MCP 服务: {ENTRY_FILE} ...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            langchain_tools = []

            for tool in mcp_tools.tools:
                # 修复闭包：通过默认参数捕获当前 tool.name
                async def _call_mcp_tool(tool_name=tool.name, **kwargs):
                    return await session.call_tool(tool_name, arguments=kwargs)

                langchain_tools.append(
                    Tool(
                        name=tool.name,
                        description=tool.description,
                        func=None,
                        coroutine=_call_mcp_tool
                    )
                )

            print(f"✅ 成功加载 {len(langchain_tools)} 个工具: {[t.name for t in langchain_tools]}")
            return langchain_tools


# ==================== 构建 Agent ====================
def build_agent(tools_list):
    # 初始化 LLM
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0.6,
        max_tokens=4096,
        frequency_penalty=0,
        top_p=0.95,
        extra_body={
            "top_k": 50,
            "thinking_budget": 32768,
        }
    )

    tools_by_name = {tool.name: tool for tool in tools_list}
    llm_with_tools = llm.bind_tools(tools_list)

    # --- 节点函数 ---
    def llm_call(state: AgentState) -> dict:
        response = llm_with_tools.invoke(
            [SystemMessage(content="你是一个有用的助手，根据用户问题选择合适的工具调用，或者不用调用工具")]
            + state["messages"]
        )
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        result = []
        last_message = state["messages"][-1]
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            # 调用异步工具函数（在同步上下文中）
            observation = asyncio.run(tool.coroutine(**tool_call["args"]))
            result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        return {"messages": result}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        return "tools" if last_message.tool_calls else END

    # --- 构建图 ---
    workflow = StateGraph(AgentState)
    workflow.add_node("llm_call", llm_call)
    workflow.add_node("tool_node", tool_node)
    workflow.set_entry_point("llm_call")
    workflow.add_conditional_edges(
        "llm_call",
        should_continue,
        {"tools": "tool_node", END: END}
    )
    workflow.add_edge("tool_node", "llm_call")

    return workflow.compile()


# ==================== 主运行函数 ====================
async def run():
    print("🚀 开始初始化 MCP 工具和 LLM...")

    # 1. 加载工具
    k8s_tools = await get_k8s_tools()

    # 2. 构建 agent
    agent = build_agent(k8s_tools)

    print("\n✅ 初始化完成！开始交互...\n")

    # 3. 交互循环
    while True:
        try:
            user_input = input("❓ 今天想问点什么呢？（输入 'quit' 退出）: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                print("👋 再见！")
                break
            if not user_input:
                continue

            # 初始状态
            initial_state: AgentState = {
                "messages": [HumanMessage(content=user_input)]
            }

            # 执行 agent（同步）
            final_state = agent.invoke(initial_state)

            # 提取最终 AI 回复
            messages = final_state["messages"]
            # 从后往前找第一个非工具调用的 AI 消息
            for msg in reversed(messages):
                if msg.type == "ai" and not getattr(msg, 'tool_calls', None):
                    print(f"\n🤖 助手: {msg.content}\n")
                    break
            else:
                # fallback: 打印最后一条消息
                last_msg = messages[-1]
                print(f"\n🤖 助手: {getattr(last_msg, 'content', str(last_msg))}\n")

        except KeyboardInterrupt:
            print("\n👋 被用户中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}\n")
            import traceback
            traceback.print_exc()


# ==================== 入口 ====================
if __name__ == "__main__":
    asyncio.run(run())