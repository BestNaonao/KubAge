import asyncio
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from agent.graph import build_react_agent
from agent.nodes import RerankNode
from informer import RuntimeBridge
from retriever import MilvusHybridRetriever, GraphTraverser
from utils import get_chat_model, get_dense_embed_model, get_sparse_embed_model
from utils.mcp_manager import MCPToolManager
from utils.milvus_adapter import connect_milvus_by_env
from workflow.build_knowledge_base import STATIC_PARTITION_NAME

# 设置页面配置
st.set_page_config(page_title="KubAge 智能运维中枢", page_icon="☸️", layout="wide")
st.title("☸️ KubAge 智能运维智能体工作台")

# 🌟 1. 加载环境变量并拼接数据库 URI
load_dotenv()
DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"


@st.cache_resource
def init_agent_system():
    # 增加详细的控制台打印，方便在终端定位卡顿位置
    print("\n" + "=" * 50)
    print(">>> 🚀 开始初始化 Agent 核心系统 (由于加载大模型，可能需要几分钟)...")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    print(">>> [1/9] 正在连接 Milvus 向量数据库...")
    connect_milvus_by_env()

    # 🌟 将数据库和 MCP 等异步操作封装在一个 async 函数中，安全调度
    async def _async_init():
        print(">>> [2/9] 正在初始化 PostgreSQL 异步连接池...")
        # 注意：必须加 open=False，避免在同步上下文中强制开启导致死锁
        conn_pool = AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=20,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False
        )
        await conn_pool.open()  # 手动在异步上下文中打开连接

        print(">>> [3/9] 正在设置 PostgreSQL 检查点持久化表...")
        pg_checkpointer = AsyncPostgresSaver(conn_pool)
        await pg_checkpointer.setup()

        print(">>> [4/9] 正在初始化 MCP Manager 与 Kubernetes 沙箱...")
        mcp_manager = MCPToolManager.get_instance()
        await mcp_manager.initialize(config_path="config/mcp_config.json")
        tool_string = mcp_manager.get_tools_description()

        return conn_pool, pg_checkpointer, tool_string

    # 执行所有异步初始化任务
    pool, checkpointer, tool_str = loop.run_until_complete(_async_init())

    DENSE_MODEL_PATH = "models/Qwen/Qwen3-Embedding-0.6B"
    SPARSE_MODEL_PATH = "BAAI/bge-m3"
    RERANKER_MODEL_PATH = "models/Qwen/Qwen3-Reranker-0.6B"
    COLLECTION_NAME = "knowledge_base_v3"

    print(f">>> [5/9] 正在加载 Dense Embedding 模型 ({DENSE_MODEL_PATH})...")
    dense_embedding = get_dense_embed_model(DENSE_MODEL_PATH)

    print(f">>> [6/9] 正在加载 Sparse Embedding 模型 ({SPARSE_MODEL_PATH})...")
    sparse_embedding = get_sparse_embed_model(SPARSE_MODEL_PATH)

    print(f">>> [7/9] 正在加载 Reranker 重排模型 ({RERANKER_MODEL_PATH})...")
    reranker = RerankNode(RERANKER_MODEL_PATH, top_n=5)

    print(">>> [8/9] 正在加载 LLM 大语言模型...")
    llm = get_chat_model(temperature=0.1, extra_body={"top_k": 50, "thinking_budget": 32768})

    print(">>> [9/9] 正在组装 Agent Graph...")
    informer = RuntimeBridge(
        dense_embedding_func=dense_embedding,
        sparse_embedding_func=sparse_embedding,
        collection_name=COLLECTION_NAME,
    )

    retriever = MilvusHybridRetriever(
        collection_name=COLLECTION_NAME,
        dense_embedding_func=dense_embedding,
        sparse_embedding_func=sparse_embedding,
        top_k=5
    )
    traverser = GraphTraverser(COLLECTION_NAME, partition_names=[STATIC_PARTITION_NAME])

    built_app = build_react_agent(
        llm, informer, retriever, traverser, reranker, tool_str,
        checkpointer=checkpointer
    )

    print(">>> ✅ Agent 系统初始化完成！")
    print("=" * 50 + "\n")

    return built_app, loop

app, async_loop = init_agent_system()

# --- 会话状态管理 ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# 配置 thread_id，用于 Checkpointer 的断点恢复
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- 渲染历史对话 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- 核心交互逻辑 ---
def run_agent_step(user_input=None, feedback=None):
    """步进运行 Agent，处理正常输入或 HITL 反馈"""
    if user_input:
        inputs = {"messages": [HumanMessage(content=user_input)]}
    elif feedback:
        # 人工驳回并注入指导指令，强制路由回 Planning 节点
        inputs = {"messages": [HumanMessage(content=f"【人工干预】: {feedback}")]}
    else:
        # 直接审批通过，无新输入，继续挂起的图
        inputs = None

    status_container = st.status("🚀 KubAge 工作流执行中...", expanded=True)

    async def _stream():
        # 根据是否有新输入决定是调用 ainvoke 还是单纯继续 astream
        streamer = app.astream(inputs, config, stream_mode="values") if inputs else app.astream(None, config,
                                                                                                stream_mode="values")
        async for event in streamer:
            # 获取最新状态进行可视化渲染
            state = event

            # 推理流展示: Analysis
            if "analysis" in state and state["analysis"]:
                analysis = state["analysis"]
                with status_container:
                    st.markdown(
                        f"**🧠 意图分析 (Analysis)**\n* **技术摘要**: {analysis.technical_summary}\n* **风险等级**: `{analysis.risk_level}`\n* **推理过程**: {analysis.reasoning}")

            # 推理流展示: Planning
            if "plan" in state and state["plan"]:
                plan = state["plan"]
                with status_container:
                    st.markdown(
                        f"**📝 规划动作 (Planning)**\n* **Action**: `{plan.action}`\n* **推理**: {plan.reasoning}")

            # 反思流展示: Regulation
            if "evaluation" in state and state["evaluation"]:
                eval_obj = state["evaluation"]
                with status_container:
                    if eval_obj.status.value == "Fail" or eval_obj.status.value == "Needs Refinement":
                        st.error(
                            f"**⚖️ 自我反思 (Regulation)**\n* **状态**: {eval_obj.status.value}\n* **反馈**: {eval_obj.feedback}\n* **经验总结**: {eval_obj.reflection}")
                    else:
                        st.success(f"**⚖️ 评估通过**: {eval_obj.feedback}")

            # 生成最终回答: Expression
            if "generated_response" in state:
                return state["generated_response"]

    final_reply = async_loop.run_until_complete(_stream())
    status_container.update(label="✅ 工作流执行完毕", state="complete", expanded=False)

    # 检查是否触发了断点 (HITL)
    curr_state = app.get_state(config)
    if curr_state.next and "ToolCall" in curr_state.next:
        st.warning("⚠️ **执行面隔离拦截**: 即将调用破坏性工具，系统已自动挂起执行流。等待人工审批。")
        st.session_state.awaiting_approval = True
        st.rerun()

    if final_reply:
        st.session_state.messages.append({"role": "assistant", "content": final_reply})
        with st.chat_message("assistant"):
            st.markdown(final_reply)


# --- 接收用户输入 ---
if prompt := st.chat_input("描述您的 Kubernetes 运维需求..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    run_agent_step(user_input=prompt)

# --- 人在回路 (HITL) 审批面板 ---
if st.session_state.get("awaiting_approval", False):
    current_state = app.get_state(config)
    plan = current_state.values.get("plan")

    with st.form("hitl_form"):
        st.subheader("🛡️ 人工审批沙箱")
        st.info(f"**请求调用工具**: `{plan.tool_name}`\n\n**执行参数**: {plan.tool_args}")
        feedback_input = st.text_input("驳回附加指导 (可选，如: '只重启 Pod，不要删除')")

        col1, col2 = st.columns(2)
        approved = col1.form_submit_button("✅ 批准执行", type="primary")
        rejected = col2.form_submit_button("❌ 驳回并重新规划")

        if approved:
            st.session_state.awaiting_approval = False
            run_agent_step()  # 继续流转
        elif rejected:
            st.session_state.awaiting_approval = False
            run_agent_step(feedback=feedback_input)  # 注入指导并继续