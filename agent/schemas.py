from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "Low"  # 信息查询
    MEDIUM = "Medium"  # 配置检查、非破坏性调试
    HIGH = "High"  # 配置变更、重启
    CRITICAL = "Critical"  # 删除资源、危险操作

class OperationType(str, Enum):
    KNOWLEDGE_QA = "Knowledge_QA"   # 知识问答

    RESOURCE_CREATION = "Resource_Creation"     # 资源创建
    RESOURCE_MUTATION = "Resource_Mutation"     # 资源变更
    RESOURCE_DELETION = "Resource_Deletion"     # 资源删除
    RESOURCE_INQUIRY = "Resource_Inquiry"       # 资源查询

    DIAGNOSIS = "Diagnosis"     # 故障诊断
    CONFIGURE = "Configure"     # 配置变更
    SCALING = "Scaling"     # 性能调优——水平伸缩
    RESTART = "Restart"     # 重启运行时
    ROLLOUT = "Rollout"     # 回滚
    PROXY = "Proxy"     # 代理
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
        description="思维链推理过程：1.结合历史消除歧义 2.分析意图 3.提取信息 4.判断风险。"
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
        description="非空。该操作或问题的风险等级。"
    )

    search_queries: List[str] = Field(
        description="非空。用于在 Kubernetes 官方文档知识库中检索的通用技术 Query 列表。要求: 1. 必须去除所有用户特定的实体名称 2. 必须转换为 Kubernetes 通用术语 3. 包含排错指南、命令参考、概念解释或其他类型的查询 4. 用【中文描述 + 英文术语】的混合形式表达。"
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description="如果信息缺失严重无法进行下一步，生成追问问题；否则为 None。"
    )