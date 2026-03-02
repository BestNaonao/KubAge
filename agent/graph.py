from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END

# 引入节点
from agent.nodes import (
    AnalysisNode, ExpressionNode, PlanningNode,
    RegulationNode, RetrievalNode, SensoryNode,
    ToolCallNode,
)
from agent.schemas import NextStep, PlanAction
from agent.state import AgentState


Sensory = "Sensory"
Analysis = "Analysis"
Planning = "Planning"
Retrieval = "Retrieval"
ToolCall = "ToolCall"
Self_Regulation = "Self-Regulation"
Expression =  "Expression"


def build_react_agent(
        llm,
        informer,       # 感知器
        retriever,      # 检索器
        traverser,      # 拓扑扩展器
        reranker,       # 重排器
        tool_descriptions: str,
        config_path="config/mcp_config.json",
        checkpointer: BaseCheckpointSaver=None
):
    """
    构建图，传入 tool_descriptions 供 Planning 节点使用
    """
    workflow = StateGraph(AgentState)

    # 1. 初始化节点
    sensory_node = SensoryNode(config_path=config_path, informer=informer)
    analysis_node = AnalysisNode(llm)
    planning_node = PlanningNode(llm, tool_descriptions)
    retrieval_node = RetrievalNode(retriever, traverser, reranker)
    tool_node = ToolCallNode()  # ToolNode 内部自行获取单例，无需传入
    regulation_node = RegulationNode(llm)
    expression_node = ExpressionNode()

    # 2. 添加节点
    workflow.add_node(Sensory, sensory_node)
    workflow.add_node(Analysis, analysis_node)
    workflow.add_node(Planning, planning_node)
    workflow.add_node(Retrieval, retrieval_node)
    workflow.add_node(ToolCall, tool_node)
    workflow.add_node(Self_Regulation, regulation_node)
    workflow.add_node(Expression, expression_node)

    # 3. 定义边 (保持之前的逻辑不变)
    workflow.add_edge(START, Sensory)
    workflow.add_edge(Sensory, Analysis)

    def route_analysis(state: AgentState):
        analysis = state.get("analysis")
        if analysis and analysis.clarification_question:
            return Expression
        return Planning

    workflow.add_conditional_edges(Analysis, route_analysis)

    def route_planning(state: AgentState):
        plan = state.get("plan")
        if not plan: return Expression  # Fallback

        if plan.action == PlanAction.RETRIEVE:
            return Retrieval
        elif plan.action == PlanAction.TOOL_USE:
            return ToolCall
        else:
            return Expression

    workflow.add_conditional_edges(Planning, route_planning)

    workflow.add_edge(Retrieval, Self_Regulation)
    workflow.add_edge(ToolCall, Self_Regulation)

    def route_regulation(state: AgentState):
        retrieval_attempts = state.get("retrieval_attempts", 0)
        tool_use_attempts = state.get("tool_use_attempts", 0)

        # 如果超过最大重试次数，强制结束
        if retrieval_attempts >= 3 and tool_use_attempts >= 3:
            print("   🚫 Circuit Breaker Triggered -> To Expression")
            return Expression

        step = state.get("evaluation").next_step
        if step == NextStep.TO_EXPRESSION: return Expression
        if step == NextStep.TO_ANALYSIS: return Analysis
        if step == NextStep.TO_PLANNING: return Planning
        return Planning

    workflow.add_conditional_edges(Self_Regulation, route_regulation)

    workflow.add_edge(Expression, END)

    return workflow.compile(checkpointer=checkpointer, interrupt_before=["ToolCall"])