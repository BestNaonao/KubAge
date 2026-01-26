from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.schemas import NextStep, PlanAction

# 引入节点
from agent.nodes.analysis_node import AnalysisNode
from agent.nodes.planning_node import PlanningNode
from agent.nodes.retrieval_node import RetrievalNode
from agent.nodes.tool_node import ToolCallNode
from agent.nodes.regulation_node import RegulationNode
from agent.nodes.expression_node import ExpressionNode
from agent.nodes.sensory_node import SensoryNode  # 假设你有这个节点


Sensory = "Sensory"
Analysis = "Analysis"
Planning = "Planning"
Retrieval = "Retrieval"
ToolCall = "ToolCall"
Self_Regulation = "Self-Regulation"
Expression =  "Expression"


def build_react_agent(llm, retriever, reranker, tool_descriptions: str):
    """
    构建图，传入 tool_descriptions 供 Planning 节点使用
    """
    workflow = StateGraph(AgentState)

    # 1. 初始化节点
    sensory_node = SensoryNode()
    analysis_node = AnalysisNode(llm)
    # 注入工具描述
    planning_node = PlanningNode(llm, tool_descriptions)
    retrieval_node = RetrievalNode(retriever, reranker)
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
        step = state.get("evaluation").next_step
        if step == NextStep.TO_ANALYSIS: return Analysis
        if step == NextStep.TO_PLANNING: return Planning
        if step == NextStep.TO_RETRIEVAL: return Retrieval
        if step == NextStep.TO_TOOL: return ToolCall
        return Expression

    workflow.add_conditional_edges(Self_Regulation, route_regulation)

    workflow.add_edge(Expression, END)

    return workflow.compile()