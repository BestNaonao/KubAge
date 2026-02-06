import asyncio
import os

import torch
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from agent.graph import build_react_agent
from agent.nodes import RerankNode
from retriever import MilvusHybridRetriever
from utils.llm_factory import get_chat_model
from utils.mcp_manager import MCPToolManager


async def main():
    print("🚀 Starting Kubernetes Agent...")
    # 加载环境变量
    load_dotenv(find_dotenv())
    host = os.getenv('MILVUS_HOST', 'localhost')
    port = os.getenv('MILVUS_PORT', '19530')
    user = os.getenv('MILVUS_USER', 'root')
    password = os.getenv('MILVUS_ROOT_PASSWORD', 'Milvus')

    # --- A. 连接 Milvus
    print(f"正在连接 Milvus ({host}:{port})...")
    connections.connect(alias="default", host=host, port=port, user=user, password=password)

    # --- B. 初始化资源 (一次性加载模型，避免重复加载) ---
    print("⏳ Initializing Embeddings and Retriever (this may take a while)...")

    # 请根据你的实际模型路径修改
    DENSE_MODEL_PATH = "models/Qwen/Qwen3-Embedding-0.6B"
    SPARSE_MODEL_PATH = "BAAI/bge-m3"
    RERANKER_MODEL_PATH = "models/Qwen/Qwen3-Reranker-0.6B"
    COLLECTION_NAME = "knowledge_base_v2"

    # 1. Dense Embedding
    dense_embedding = HuggingFaceEmbeddings(
        model_name=DENSE_MODEL_PATH,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    # 2. Sparse Embedding
    sparse_embedding = BGEM3EmbeddingFunction(
        model_name=SPARSE_MODEL_PATH,
        use_fp16=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 3. Retriever
    retriever = MilvusHybridRetriever(
        collection_name=COLLECTION_NAME,
        dense_embedding_func=dense_embedding,
        sparse_embedding_func=sparse_embedding,
        top_k=5
    )

    reranker = RerankNode(RERANKER_MODEL_PATH, 5)

    # 1. 初始化 MCP Manager
    mcp_manager = MCPToolManager.get_instance()
    # 确保 config/mcp_config.json 存在且路径正确
    await mcp_manager.initialize(config_path="config/mcp_config.json")

    count = 0
    for tool_name, tool in mcp_manager.tools_map.items():
        print(f" -Tool: {tool_name}")
        count += 1
    if count < 26:
        print("   ❌ 加载MCP工具失败:缺少部分工具！")
        await mcp_manager.close()
        return

    try:
        # 获取工具描述文本
        tool_str = mcp_manager.get_tools_description()

        # 2. 初始化其他组件 (LLM, Retriever 等)
        llm = get_chat_model(
            temperature=0.1,
            extra_body={
                "top_k": 50,
                "thinking_budget": 32768,
            }
        )

        # 3. 构建 Agent
        app = build_react_agent(llm, retriever, reranker, tool_descriptions=tool_str)

        print("\n🚀 Agent Initialized. Ready for queries.")

        # 4. 运行 Agent (示例)
        inputs = {"messages": [HumanMessage(content="请帮我在workspace/bin文件夹下用curl安装kubectl。")]}

        # 使用 ainvoke 因为 ToolNode 是异步的
        async for event in app.astream(inputs):
            for key, value in event.items():
                print(f"Completed Node: {key}")

    finally:
        # 5. 清理资源
        await mcp_manager.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass