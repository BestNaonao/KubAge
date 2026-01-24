import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_client():
    # 配置连接参数
    # 我们直接运行 python 脚本作为子进程
    server_params = StdioServerParameters(
        command=sys.executable,  # 使用当前的 python 解释器
        args=["../os_mcp/os_mcp_server.py"],  # 确保文件名匹配
        env=None
    )

    print("Connecting to MCP server...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. 初始化
            await session.initialize()
            print("Connected!")

            # 2. 获取可用工具列表
            print("\n--- Fetching Tools ---")
            tools_response = await session.list_tools()

            for tool in tools_response.tools:
                print(f"Name: {tool.name}")
                print(f"Description: {tool.description}")
                print(f"Args: {tool.inputSchema.get('properties', {}).keys()}")
                print("-" * 20)

            # 3. (可选) 测试调用一个工具
            print("\n--- Testing 'get_system_info' ---")
            result = await session.call_tool("get_system_info", arguments={})
            print("Result Content:")
            print(result.content[0].text)

            # 4. (可选) 测试写入文件
            print("\n--- Testing 'write_file' ---")
            write_res = await session.call_tool(
                "write_file",
                arguments={"path": "test_k8s_template.yaml", "content": "apiVersion: v1\nkind: Pod"}
            )
            print(write_res.content[0].text)


if __name__ == "__main__":
    asyncio.run(run_client())