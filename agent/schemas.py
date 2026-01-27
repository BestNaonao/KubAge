from enum import Enum
from typing import List, Optional, Dict, Any

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
    INSTALL = "Install"     # 安装
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

    entities: List[NamedEntity] = Field(
        default_factory=list,
        description="提取到的关键命名实体列表。"
    )

    target_operation: OperationType = Field(
        description="用户的目标操作类型。"
    )

    technical_summary: str = Field(
        description="用户问题的简短技术摘要，去除口语化表达，补充完整的上下文信息。"
    )

    risk_level: RiskLevel = Field(
        description="该操作或问题的风险等级。"
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description="如果信息缺失严重无法进行下一步，生成追问问题；否则为 None。"
    )


class PlanAction(str, Enum):
    RETRIEVE = "Retrieve"   # 查询知识库
    TOOL_USE = "Tool_Use"   # 调用工具
    DIRECT_ANSWER = "Direct_Answer"     # 直接回答/任务结束


# --- 新增：执行计划模型 ---
class ExecutionPlan(BaseModel):
    """
    Planning节点的输出结构
    """
    reasoning: str = Field(description="规划的理由，为什么选择这个动作")
    action: PlanAction = Field(description="下一步的具体动作类型，或查询知识库，或调用工具，或直接回答")

    # 如果 action == RETRIEVE
    search_queries: Optional[List[str]] = Field(
        default=None,
        description="当 action == RETRIEVE 时必填。用于在 Kubernetes 官方文档知识库中检索的通用技术 Query 列表。要求: 1. 必须去除所有用户特定的实体名称 2. 必须转换为 Kubernetes 通用术语 3. 包含排错指南、命令参考、概念解释或其他类型的查询 4. 用【中文描述 + 英文术语】的混合形式表达"
    )
    # 如果 action == TOOL_USE
    tool_name: Optional[str] = Field(
        default=None,
        description="当 action == TOOL_USE 时必填。调用的工具名称"
    )
    tool_args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="当 action == TOOL_USE 时必填。工具调用的参数"
    )
    # 如果 action == DIRECT_ANSWER
    final_answer: Optional[str] = Field(
        default=None,
        description="当 action 为 DIRECT_ANSWER 时必填。直接对用户问题的回答"
    )


class EvaluatedStatus(Enum):
    PASS = "Pass"       # 执行成功
    FAIL = "Fail"       # 执行失败
    NEEDS_REFINEMENT = "Needs Refinement"   # 结果不理想，需要优化


class NextStep(str, Enum):
    TO_ANALYSIS = "Analysis"        # 重新分析(意图理解有误)
    TO_PLANNING = "Planning"        # 重新规划(更换工具或检索词)
    TO_RETRIEVAL = "Retrieval"      # 重新检索(极少直接用)
    TO_TOOL = "ToolCall"            # 重新调用(极少直接用)
    TO_EXPRESSION = "Expression"    # 回答用户


class SelfEvaluation(BaseModel):
    """
    Self-Regulation节点的输出结构
    """
    reasoning: str = Field(description="评估理由")
    status: EvaluatedStatus = Field(description="当前步骤执行结果的评估状态")
    next_step: NextStep = Field(description="决定回退到哪一步或继续前进")
    feedback: str = Field(description="反馈给下一步骤的改进建议或错误信息")