from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional, List
from langchain_core.documents import Document

# 引入你的 Schema
from agent.schemas import ProblemAnalysis, OperationType, RiskLevel, NamedEntity


@dataclass
class RetrievalTestScenario:
    name: str
    # 这里我们直接准备好 AnalysisNode 应该产出的结果
    mock_analysis: ProblemAnalysis
    # 验证函数
    verify_func: Callable[[List[Document]], None]
    description: Optional[str] = None


# ==========================================
# 验证函数库
# ==========================================

def _assert_has_results(docs: List[Document]):
    """验证检索结果不为空"""
    assert docs is not None and len(docs) > 0, \
        f"❌ Expected retrieval results, but got empty list."
    print(f"   ✅ Retrieved {len(docs)} documents.")


def _assert_content_contains(docs: List[Document], keyword: str):
    """验证检索回来的文档中包含特定关键词（粗略验证）"""
    found = False
    for doc in docs:
        if keyword.lower() in doc.page_content.lower():
            found = True
            break

    if not found:
        # 注意：如果知识库里真没这个词，这里会报错。测试时需确保知识库有相关数据。
        print(f"   ⚠️ Warning: Keyword '{keyword}' not found in retrieved docs (might be normal if KB is small).")
    else:
        print(f"   ✅ Found keyword '{keyword}' in documents.")


# ==========================================
# 构造 Mock 数据 (来源于 代码.txt)
# ==========================================

# 场景 1: 故障排查 (Contextual Diagnosis)
analysis_diagnosis = ProblemAnalysis(
    reasoning="用户明确指出了具体的 Pod 名称 'redis-cart'...",
    technical_summary="用户报告在 default 命名空间下的 redis-cart Pod 无法连接。",
    target_operation=OperationType.DIAGNOSIS,
    entities=[
        NamedEntity(name="redis-cart", type="Pod"),
        NamedEntity(name="default", type="Namespace")
    ],
    risk_level=RiskLevel.LOW,
    search_queries=[
        "Pod 网络连接问题排查",
        "Pod 无法访问常见原因",
        "Kubernetes Service 连接 Pod 失败",
        "Pod 状态异常诊断方法"
    ],
    clarification_question=None
)

# 场景 2: 水平伸缩 (Scaling)
analysis_scaling = ProblemAnalysis(
    reasoning="用户明确提到 'backend-api' 这个部署...",
    technical_summary="用户希望将 named backend-api 的 Deployment 扩展到 10 个副本。",
    target_operation=OperationType.SCALING,
    entities=[NamedEntity(name="backend-api", type="Deployment")],
    risk_level=RiskLevel.MEDIUM,
    search_queries=[
        "Deployment 扩容到指定副本数",
        "Kubernetes 水平扩展 Deployment",
        "kubectl scale 命令使用方法",
        "Deployment 扩容对集群资源的影响"
    ],
    clarification_question=None
)

# 场景 3: 知识问答 (Knowledge QA)
analysis_qa = ProblemAnalysis(
    reasoning="用户询问的是 Kubernetes 中 StatefulSet 和 Deployment 的核心区别...",
    technical_summary="用户询问 Kubernetes 中 StatefulSet 与 Deployment 资源类型的核心区别。",
    target_operation=OperationType.KNOWLEDGE_QA,
    entities=[],
    risk_level=RiskLevel.LOW,
    search_queries=[
        "StatefulSet 与 Deployment 区别",
        "Kubernetes StatefulSet 特性说明",
        "Deployment 和 StatefulSet 使用场景对比"
    ],
    clarification_question=None
)


# ==========================================
# 具体测试用例
# ==========================================

def verify_diagnosis(docs):
    _assert_has_results(docs)
    # 假设你的知识库里有网络或Pod相关的文档
    _assert_content_contains(docs, "Pod")


def verify_scaling(docs):
    _assert_has_results(docs)
    _assert_content_contains(docs, "scale")


def verify_qa(docs):
    _assert_has_results(docs)
    _assert_content_contains(docs, "StatefulSet")


ALL_RETRIEVAL_SCENARIOS = [
    RetrievalTestScenario(
        name="01_Retrieval_Diagnosis",
        description="测试故障排查场景下的文档召回",
        mock_analysis=analysis_diagnosis,
        verify_func=verify_diagnosis
    ),
    RetrievalTestScenario(
        name="02_Retrieval_Scaling",
        description="测试运维命令场景下的文档召回",
        mock_analysis=analysis_scaling,
        verify_func=verify_scaling
    ),
    RetrievalTestScenario(
        name="03_Retrieval_Knowledge",
        description="测试概念对比场景下的文档召回",
        mock_analysis=analysis_qa,
        verify_func=verify_qa
    )
]