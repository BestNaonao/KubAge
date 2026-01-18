import operator
from typing import Optional, Dict, Any
from typing import TypedDict, Annotated, List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from agent.schemas import ProblemAnalysis


class AgentState(TypedDict):
    # 消息历史，使用 operator.add 表示追加模式
    messages: Annotated[List[BaseMessage], operator.add]
    # 存储分析结果，覆盖模式（最新的分析覆盖旧的）
    analysis: ProblemAnalysis | None
    # 检索到的文档片段
    retrieved_chunks: Optional[List[Document]]
    # 生成的结果
    generated_response: Optional[str]
    # 元数据
    metadata: Optional[Dict[str, Any]]
    # 错误信息
    error: Optional[str]
