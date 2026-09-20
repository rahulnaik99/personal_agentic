from langgraph.graph import END, START, StateGraph

from orchestrator_service.app.agent import (
    call_rag_node,
    call_tool_node,
    classify_intent,
    direct_answer,
    finalize,
    guardrail_input_node,
    increment_loop,
    route_after_classification,
    route_after_rag,
    route_from_start,
)
from orchestrator_service.app.schemas import GraphState


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("guardrail_input", guardrail_input_node)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("rag_agent", call_rag_node)
    graph.add_node("tool_agent", call_tool_node)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("increment_loop", increment_loop)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "guardrail_input")

    graph.add_conditional_edges(
        "guardrail_input", route_from_start,
        {"finalize": "finalize", "classify_intent": "classify_intent"},
    )

    graph.add_conditional_edges(
        "classify_intent", route_after_classification,
        {"rag": "rag_agent", "both": "rag_agent", "tool": "tool_agent", "direct": "direct_answer"},
    )

    graph.add_conditional_edges(
        "rag_agent", route_after_rag,
        {"finalize": "finalize", "tool_agent": "increment_loop"},
    )

    graph.add_edge("increment_loop", "tool_agent")
    graph.add_edge("tool_agent", "finalize")
    graph.add_edge("direct_answer", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


compiled_graph = build_graph()
