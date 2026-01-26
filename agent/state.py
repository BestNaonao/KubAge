import operator
from typing import Optional, Dict, Any
from typing import TypedDict, Annotated, List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from agent.schemas import ProblemAnalysis, ExecutionPlan, SelfEvaluation


class AgentState(TypedDict):
    # --- 基础对话历史 ---
    messages: Annotated[List[BaseMessage], operator.add]

    # --- 节点中间产物 ---
    # 1. 分析结果
    analysis: Optional[ProblemAnalysis]

    # 2. 当前计划 (每次Planning覆盖)
    plan: Optional[ExecutionPlan]

    # 3. 执行结果
    # 检索到的文档 (Retrieval Node 更新)
    retrieved_docs: Optional[List[Document]]
    retrieval_attempts: int
    # 工具输出结果 (ToolCall Node 更新)
    tool_output: Optional[str]

    # 4. 评估结果 (Self-Regulation Node 更新)
    evaluation: Optional[SelfEvaluation]

    # --- 辅助信息 ---
    error: Optional[str]
    metadata: Optional[Dict[str, Any]]
