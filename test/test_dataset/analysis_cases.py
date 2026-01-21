from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage

# 引入你的 Schema
from agent.schemas import OperationType, RiskLevel

# 定义测试场景的数据结构
@dataclass
class TestScenario:
    name: str
    inputs: Dict[str, Any]
    verify_func: Callable[[Any], None]
    description: Optional[str] = None

# ==========================================
# 验证函数库 (Helper Functions)
# ==========================================

def _assert_op(result, expected_ops: List[OperationType]):
    assert result.target_operation in expected_ops, \
        f"Op mismatch. Expected {expected_ops}, got {result.target_operation}"

def _assert_risk(result, expected_risks: List[RiskLevel]):
    assert result.risk_level in expected_risks, \
        f"Risk mismatch. Expected {expected_risks}, got {result.risk_level}"

def _assert_entity(result, keyword: str):
    entities_str = str(result.entities)
    assert keyword in entities_str, \
        f"Entity '{keyword}' not found in {entities_str}"

# ==========================================
# 具体测试用例 (Test Cases)
# ==========================================

def case_diagnosis_redis(result):
    """验证: 故障排查 (Contextual)"""
    _assert_entity(result, "redis-cart")
    _assert_op(result, [OperationType.DIAGNOSIS, OperationType.RESOURCE_INQUIRY])
    _assert_risk(result, [RiskLevel.LOW, RiskLevel.MEDIUM])

def case_cross_namespace(result):
    """验证: 交叉命名空间(Cross Namespace)"""
    _assert_entity(result, "payment-service")
    _assert_entity(result, "redis-cache")
    _assert_op(result, [OperationType.DIAGNOSIS])

def case_delete_nginx(result):
    """验证: 删除/重启高风险操作 (Ambiguity + High Risk)"""
    _assert_entity(result, "nginx-frontend")
    _assert_op(result, [OperationType.RESOURCE_DELETION, OperationType.RESTART])
    # 必须是高危
    _assert_risk(result, [RiskLevel.HIGH, RiskLevel.CRITICAL])

def case_scaling_api(result):
    """验证: 水平伸缩 (Scaling)"""
    _assert_entity(result, "backend-api")
    _assert_op(result, [OperationType.SCALING, OperationType.RESOURCE_MUTATION])
    # 修改副本数属于配置变更或高风险
    _assert_risk(result, [RiskLevel.HIGH, RiskLevel.MEDIUM])

def case_knowledge_qa(result):
    """验证: 纯知识问答 (Knowledge QA)"""
    # 问答通常不涉及具体实体，或者实体只是概念
    _assert_op(result, [OperationType.KNOWLEDGE_QA])
    _assert_risk(result, [RiskLevel.LOW])

def case_rollback_auth(result):
    """验证: 回滚操作 (Rollout)"""
    _assert_entity(result, "auth-service")
    _assert_op(result, [OperationType.ROLLOUT])
    # 回滚是重大变更
    _assert_risk(result, [RiskLevel.HIGH, RiskLevel.CRITICAL])

def case_create_namespace(result):
    """验证: 资源创建 (Resource Creation)"""
    _assert_entity(result, "monitoring") # 期望提取出新名字
    _assert_entity(result, "Namespace")  # 期望提取出类型
    _assert_op(result, [OperationType.RESOURCE_CREATION])
    _assert_risk(result, [RiskLevel.LOW, RiskLevel.MEDIUM])

def case_logs_inquiry(result):
    """验证: 查看日志 (Resource Inquiry/Diagnosis)"""
    _assert_entity(result, "payment-service")
    _assert_op(result, [OperationType.RESOURCE_INQUIRY, OperationType.DIAGNOSIS])
    # 只读操作，应该是 Low
    _assert_risk(result, [RiskLevel.LOW])


# ==========================================
# 导出场景列表
# ==========================================

ALL_SCENARIOS = [
    TestScenario(
        name="01_Contextual_Diagnosis",
        description="结合历史上下文识别 redis-cart 的连接问题",
        inputs={
            "messages": [
                HumanMessage(content="我的 Pod 昨天还能用，今天突然连不上了。"),
                AIMessage(content="请问能提供一下具体的 Pod 名称和 Namespace 吗？"),
                HumanMessage(content="是 default 命名空间下的 redis-cart。")
            ]
        },
        verify_func=case_diagnosis_redis
    ),
    TestScenario(
        name="01.1_Cross-Namespace Analysis",
        inputs={
            "messages": [
                HumanMessage(content="你好，我发现 payment-service namespace 下的 redis-cache 节点好像挂了。"),
                AIMessage(content="收到，我会帮您排查 redis-cache 的问题。请问具体表现是什么？"),
                HumanMessage(content="它一直在重启，状态显示 CrashLoopBackOff。请帮我分析一下原因并给出修复建议。")
            ]
        },
        verify_func=case_cross_namespace
    ),
    TestScenario(
        name="02_Dangerous_Delete",
        description="识别'把它删了'这种高危且指代不明的操作",
        inputs={
            "messages": [
                HumanMessage(content="我的 nginx-frontend-7b8c9 这里的 Pod 状态一直是 CrashLoopBackOff，怎么办？"),
                AIMessage(content="CrashLoopBackOff 通常意味着容器启动后立即退出。您可以检查一下日志或配置。"),
                HumanMessage(content="太麻烦了，直接帮我把它删了，让 Deployment 重启一个新的。")
            ]
        },
        verify_func=case_delete_nginx
    ),
    TestScenario(
        name="03_Scaling_Operation",
        description="测试水平伸缩意图 (Scaling)",
        inputs={
            "messages": [
                HumanMessage(content="现在流量太大了，帮我把 backend-api 这个部署扩容到 10 个副本。")
            ]
        },
        verify_func=case_scaling_api
    ),
    TestScenario(
        name="04_Knowledge_Concept",
        description="测试纯理论问题，不应包含具体实体操作",
        inputs={
            "messages": [
                HumanMessage(content="请问 Kubernetes 里 StatefulSet 和 Deployment 的核心区别是什么？")
            ]
        },
        verify_func=case_knowledge_qa
    ),
    TestScenario(
        name="05_Rollback_Action",
        description="测试回滚操作 (Rollout)",
        inputs={
            "messages": [
                HumanMessage(content="刚才更新的 auth-service 版本有问题，全挂了，赶紧帮我回滚到上一个版本！")
            ]
        },
        verify_func=case_rollback_auth
    ),
    TestScenario(
        name="06_Resource_Creation",
        description="测试创建新资源",
        inputs={
            "messages": [
                HumanMessage(content="我需要一个新的 Namespace，名字叫 monitoring，用来放监控组件。")
            ]
        },
        verify_func=case_create_namespace
    ),
    TestScenario(
        name="07_Log_Check_LowRisk",
        description="测试低风险的日志查询",
        inputs={
            "messages": [
                HumanMessage(content="帮我把 payment-service 的最新日志打印出来，我想看看有没有报错。")
            ]
        },
        verify_func=case_logs_inquiry
    )
]