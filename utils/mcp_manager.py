import json
import os
from contextlib import AsyncExitStack
from typing import Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolManager:
    _instance = None
    _is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MCPToolManager, cls).__new__(cls)
            cls._instance.exit_stack = AsyncExitStack()
            cls._instance.sessions = []
            cls._instance.tools_map = {}  # {tool_name: tool_callable}
            cls._instance.tools_meta = []  # List[Dict] for descriptions
            cls._instance._is_initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls()
        return cls._instance

    async def initialize(self, config_path: str = "config/mcp_config.json"):
        """
        读取配置并初始化所有 MCP 服务器连接
        """
        if self._is_initialized:
            print("⚠️ MCPToolManager already initialized.")
            return

        print(f"🔌 Loading MCP config from {config_path}...")

        # 1. 读取配置文件
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"❌ Config file not found: {config_path}")
            return

        mcp_servers = config.get("mcpServers", {})

        # 2. 遍历并连接每个服务器
        for server_name, server_config in mcp_servers.items():
            await self._load_single_mcp(server_name, server_config)

        self._is_initialized = True
        print(f"✅ All MCP servers loaded. Total tools: {len(self.tools_map)}")

    async def _load_single_mcp(self, name: str, config: Dict[str, Any]):
        """
        加载单个 MCP 服务器，参考 baseline 实现
        """
        command = config.get("command")
        args = config.get("args", [])
        env= config.get("env", None)  # 可选的环境变量配置

        # 处理环境变量，确保继承当前环境
        run_env = os.environ.copy()
        if isinstance(env, dict):
            run_env.update(env)

        print(f"   Connecting to [{name}] via {command} {args}...")

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=run_env
        )

        try:
            # 使用 ExitStack 保持连接上下文开启
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))

            await session.initialize()
            self.sessions.append(session)

            # 获取工具列表
            result = await session.list_tools()

            for tool in result.tools:
                tool_name = tool.name

                # 构造闭包函数以捕获当前 session 和 tool_name
                async def _call_mcp_tool(*inner_args, _session=session, _name=tool_name, **kwargs):
                    # 合并 args 和 kwargs，因为 call_tool 只接受 arguments 字典
                    # 这里做一个简单的假设：如果只有 kwargs，直接传；如果有 args，可能需要根据 schema 映射
                    # 为了简化，我们在 Agent 中约定生成 tool_args (dict)
                    arguments = kwargs if kwargs else {}
                    if inner_args and not kwargs:
                        # 如果传入的是位置参数，尝试作为第一个参数或者报错（视具体情况而定）
                        # MCP 协议通常要求 arguments 是字典
                        pass

                    return await _session.call_tool(_name, arguments=arguments)

                # 注册到 tools_map
                self.tools_map[tool_name] = _call_mcp_tool

                # 保存元数据用于生成描述
                self.tools_meta.append({
                    "name": tool_name,
                    "description": tool.description,
                    "schema": tool.inputSchema
                })

            print(f"   ✅ [{name}] connected. Loaded {len(result.tools)} tools.")

        except Exception as e:
            print(f"   ❌ Failed to load [{name}]: {e}")

    def get_tools_description(self) -> str:
        """
        生成格式化的工具描述字符串，用于注入 Prompt
        """
        lines = ["Available Tools:"]
        for meta in self.tools_meta:
            schema_str = json.dumps(meta['schema'], ensure_ascii=False)
            lines.append(f"- Name: {meta['name']}")
            lines.append(f"  Description: {meta['description']}")
            lines.append(f"  Args Schema: {schema_str}")
            lines.append("")
        return "\n".join(lines)

    def get_tool(self, name: str):
        return self.tools_map.get(name)

    async def close(self):
        """
        关闭所有连接
        """
        print("🔌 Closing MCP connections...")
        await self.exit_stack.aclose()