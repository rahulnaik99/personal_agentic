"""
Orchestrator node functions, used by graph.py to build the StateGraph.
Routing/RAG/tool nodes dispatch over gRPC (grpc_clients.py) rather than
calling agent code in-process — the orchestrator now only knows about
the AgentService contract, not agent internals.
"""
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator_service.app.grpc_clients import call_rag_agent, call_tool_agent
from orchestrator_service.app.guardrails import guard_input, guard_output
from orchestrator_service.app.schemas import GraphState, RouteDecision
from shared.core.logging import log_step
from shared.core.prompts import load_prompt
from shared.services.chat_llm import get_chat_llm, invoke_and_track

PROMPTS_DIR = Path(__file__).parent / "prompts"
ROUTING_PROMPT_PATH = PROMPTS_DIR / "routing_prompt.md"
DIRECT_ANSWER_PROMPT_PATH = PROMPTS_DIR / "direct_answer_prompt.md"

MAX_LOOPS = 3


def guardrail_input_node(state: GraphState) -> dict:
    result = guard_input(state["query"])
    if result.allowed:
        return {"blocked": False}
    return {
        "blocked": True,
        "block_reason": result.reason,
        "final_answer": f"I can't process this request: {result.reason}",
    }


def classify_intent(state: GraphState) -> dict:
    query = state["query"]
    override = state.get("model_override") or {}
    with log_step("classify_intent", agent="orchestrator", query=query) as ctx:
        system_prompt = load_prompt(str(ROUTING_PROMPT_PATH))
        llm = get_chat_llm(
            temperature=0.0,
            provider_override=override.get("provider"),
            model_override=override.get("model"),
        )
        response = invoke_and_track(
            llm, [SystemMessage(content=system_prompt), HumanMessage(content=query)],
            node="classify_intent", agent="orchestrator",
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)

        try:
            cleaned = raw.strip().strip("```").replace("json", "", 1).strip() if raw.strip().startswith("```") else raw.strip()
            decision = RouteDecision.model_validate_json(cleaned)
        except Exception:
            decision = RouteDecision(route="direct", reason="fallback: routing output was not valid JSON")

        ctx["output"] = {"route": decision.route}

    return {"route": decision.route, "routing_reason": decision.reason, "loop_count": state.get("loop_count", 0)}


def call_rag_node(state: GraphState) -> dict:
    result = call_rag_agent(state["query"], state["trace_id"], model_override=state.get("model_override"))
    return {"rag_result": result}


def call_tool_node(state: GraphState) -> dict:
    result = call_tool_agent(state["query"], state["trace_id"], model_override=state.get("model_override"))
    return {"tool_result": result}


def increment_loop(state: GraphState) -> dict:
    return {"loop_count": state.get("loop_count", 0) + 1}


def direct_answer(state: GraphState) -> dict:
    query = state["query"]
    override = state.get("model_override") or {}
    with log_step("direct_answer", agent="orchestrator", query=query) as ctx:
        system_prompt = load_prompt(str(DIRECT_ANSWER_PROMPT_PATH))
        llm = get_chat_llm(
            provider_override=override.get("provider"),
            model_override=override.get("model"),
        )
        response = invoke_and_track(
            llm, [SystemMessage(content=system_prompt), HumanMessage(content=query)],
            node="direct_answer", agent="orchestrator",
        )
        ctx["output"] = {"len": len(response.content or "")}
    return {"final_answer": response.content}


def finalize(state: GraphState) -> dict:
    if state.get("blocked"):
        return {
            "final_answer": state.get("final_answer", "Request blocked."),
            "output_guardrail_passed": False,
            "output_guardrail_flags": ["input_blocked"],
        }

    with log_step("finalize", agent="orchestrator") as ctx:
        if state.get("final_answer"):
            raw_answer = state["final_answer"]
        else:
            rag_result = state.get("rag_result") or {}
            tool_result = state.get("tool_result") or {}
            rag_answer = rag_result.get("answer")
            tool_answer = tool_result.get("answer")

            if rag_answer and tool_answer:
                raw_answer = f"{rag_answer}\n\nAdditional context from live sources:\n{tool_answer}"
            elif rag_answer:
                raw_answer = rag_answer
            elif tool_answer:
                raw_answer = tool_answer
            else:
                raw_answer = "I wasn't able to produce an answer."

        output_guard = guard_output(raw_answer)
        final_answer = output_guard.redacted_text if output_guard.allowed else "I can't share that response as generated."
        ctx["output"] = {"final_len": len(final_answer)}

    return {
        "final_answer": final_answer,
        "output_guardrail_passed": output_guard.allowed,
        "output_guardrail_flags": output_guard.flags,
    }


# ---- Conditional edge functions ----

def route_from_start(state: GraphState) -> str:
    return "finalize" if state.get("blocked") else "classify_intent"


def route_after_classification(state: GraphState) -> str:
    return state.get("route", "direct")


def route_after_rag(state: GraphState) -> str:
    rag_result = state.get("rag_result") or {}
    sufficient = rag_result.get("sufficient", True)
    loop_count = state.get("loop_count", 0)
    route = state.get("route")

    if route == "both":
        return "tool_agent"
    if sufficient or loop_count >= MAX_LOOPS:
        return "finalize"
    return "tool_agent"
