import operator
from typing import Optional, Dict, Any
from typing import TypedDict, Annotated, List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from agent.schemas import ProblemAnalysis, ExecutionPlan, SelfEvaluation


class AgentState(TypedDict):
    # --- 基础记忆 ---
    messages: Annotated[List[BaseMessage], operator.add]
    # 存储自然语言反思摘要(Self-Reflection)，作为"语义梯度"
    reflections: Annotated[List[str], operator.add]

    # --- 认知快照 (Cognitive Snapshot) ---
    analysis: Optional[ProblemAnalysis]
    plan: Optional[ExecutionPlan]
    evaluation: Optional[SelfEvaluation]

    # --- 执行上下文 (Execution Context) ---
    retrieved_docs: Optional[List[Document]]
    tool_output: Optional[str]

    # --- 控制元数据 (Control Metadata) ---
    retrieval_attempts: int
    tool_use_attempts: int

    # --- 辅助信息 ---
    error: Optional[str]
    metadata: Optional[Dict[str, Any]]
