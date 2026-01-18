from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "Low"  # 信息查询
    MEDIUM = "Medium"  # 配置检查、非破坏性调试
    HIGH = "High"  # 配置变更、重启
    CRITICAL = "Critical"  # 删除资源、危险操作

class OperationType(str, Enum):
    FAULT_REPORT = "Fault_Report"  # 故障报告
    PERFORMANCE_TUNING = "Performance_Tuning"  # 性能调优
    CONFIG_CHANGE = "Config_Change"  # 配置变更
    INFO_QUERY = "Info_Query"  # 信息查询
    DANGEROUS_OP = "Dangerous_Op"  # 危险操作 (如删除)
    OTHER = "Other"

class NamedEntity(BaseModel):
    name: str = Field(description="实体的名称，如 'nginx-deployment', 'kube-system'")
    type: str = Field(description="实体的类型，如 'Pod', 'Namespace', 'Service', 'ErrorLog'")

class ProblemAnalysis(BaseModel):
    """
    对用户问题的深度分析与结构化提取
    """
    # 这一步是思维链的核心：强制模型先思考，再填空
    reasoning: str = Field(
        description="思维链推理过程：1.结合历史消除歧义 2.分析意图 3.判断风险。"
    )

    technical_summary: str = Field(
        description="用户问题的简短技术摘要，去除口语化表达，补充完整的上下文信息。"
    )

    target_operation: OperationType = Field(
        description="用户的目标操作类型。"
    )

    entities: List[NamedEntity] = Field(
        default_factory=list,
        description="提取到的关键命名实体列表。"
    )

    risk_level: RiskLevel = Field(
        description="该操作或问题的风险等级。"
    )

    search_queries: List[str] = Field(
        description="用于在 Kubernetes 官方文档知识库中检索的 Query 列表。应包含同义词或底层概念（例如用户说'挂了'，检索'CrashLoopBackOff'）。"
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description="如果信息缺失严重无法进行下一步，生成追问问题；否则为 None。"
    )