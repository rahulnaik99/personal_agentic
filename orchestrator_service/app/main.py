import json
import uuid

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from orchestrator_service.app.graph import compiled_graph
from orchestrator_service.app.grpc_clients import get_agent_card
from orchestrator_service.app.schemas import ChatRequest, ChatResponse
from orchestrator_service.app.status import get_system_status
from shared.core.config import get_settings
from shared.core.logging import configure_logging, get_trace_collector, set_trace_id, start_trace_collector

settings = get_settings()
configure_logging()

app = FastAPI(title="orchestrator-gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

router = APIRouter(prefix="/chat", tags=["chat"])

NODE_LABELS = {
    "guardrail_input": "Checking request...",
    "classify_intent": "Routing your request...",
    "rag_agent": "Searching the knowledge base...",
    "tool_agent": "Searching the web...",
    "direct_answer": "Thinking...",
    "increment_loop": "Gathering more context...",
    "finalize": "Composing final answer...",
}


def _build_turn_log(query: str, final_state: dict, collector) -> dict:
    """
    Reshapes everything we already track into the structured per-turn
    log the Streamlit UI's "Logs" panel renders (request/response,
    model, which agent(s) ran, tools used, token counts, raw RAG/tool
    data, RAGAS scores, and aggregate guardrail status).
    """
    rag_result = final_state.get("rag_result") or {}
    tool_result = final_state.get("tool_result") or {}
    route = final_state.get("route", "direct")

    agent_invoked = []
    if route in ("rag", "both"):
        agent_invoked.append("rag_agent")
    if route in ("tool", "both"):
        agent_invoked.append("tool_agent")
    if route == "direct":
        agent_invoked.append("orchestrator_direct")

    tools_used = [c["name"] for c in (tool_result.get("tool_calls_made") or [])]

    trace_list = collector.to_list() if collector else []
    # Last step that actually reports a model name wins — that's the
    # generation call (rag_generate / tool_agent_step / direct_answer),
    # which is more useful here than an earlier routing-only call.
    model_name = next((s["model"] for s in reversed(trace_list) if s.get("model")), None)

    guardrail_flags: list[str] = []
    guardrail_flags += rag_result.get("guardrail_flags") or []
    guardrail_flags += tool_result.get("guardrail_flags") or []
    guardrail_flags += final_state.get("output_guardrail_flags") or []
    guardrail_passed = bool(final_state.get("output_guardrail_passed", True)) and not final_state.get("blocked")

    tools_rag_raw_data = rag_result.get("contexts") or tool_result.get("tool_calls_made") or []

    total_tokens = collector.total_tokens() if collector else {"input_tokens": 0, "output_tokens": 0}

    return {
        "request": query,
        "response": final_state.get("final_answer", ""),
        "model_name": model_name,
        "agent_invoked": agent_invoked,
        "tools_used": tools_used,
        "input_token": total_tokens.get("input_tokens", 0),
        "output_token": total_tokens.get("output_tokens", 0),
        "tools_rag_raw_data": tools_rag_raw_data,
        "rags_evaluation": rag_result.get("ragas_scores") or {},
        "guard_rail": {"passed": guardrail_passed, "flags": guardrail_flags},
    }


def _build_response_payload(query: str, final_state: dict, trace_id: str) -> dict:
    collector = get_trace_collector()
    rag_result = final_state.get("rag_result") or {}
    tool_result = final_state.get("tool_result") or {}
    return {
        "answer": final_state.get("final_answer", ""),
        "route": final_state.get("route", "unknown"),
        "trace_id": trace_id,
        "ragas_scores": rag_result.get("ragas_scores"),
        "tool_calls_made": tool_result.get("tool_calls_made"),
        "trace": collector.to_list() if collector else [],
        "total_cost_usd": collector.total_cost_usd() if collector else 0.0,
        "total_tokens": collector.total_tokens() if collector else {},
        "turn_log": _build_turn_log(query, final_state, collector),
    }


async def _event_stream(query: str, trace_id: str, model_override: dict | None = None):
    set_trace_id(trace_id)
    start_trace_collector()
    initial_state = {"query": query, "trace_id": trace_id, "loop_count": 0, "model_override": model_override}

    final_state: dict = {}
    async for event in compiled_graph.astream_events(initial_state, version="v2"):
        kind = event.get("event")
        node_name = event.get("name")

        if kind == "on_chain_start" and node_name in NODE_LABELS:
            yield {"event": "progress", "data": json.dumps({"node": node_name, "label": NODE_LABELS[node_name]})}

        if kind == "on_chain_end" and node_name == "finalize":
            output = event.get("data", {}).get("output", {})
            if isinstance(output, dict):
                final_state.update(output)

    if "final_answer" not in final_state:
        final_state = await compiled_graph.ainvoke(initial_state)

    yield {"event": "done", "data": json.dumps(_build_response_payload(query, final_state, trace_id))}


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    trace_id = request.session_id or str(uuid.uuid4())
    override = request.model_override.model_dump() if request.model_override else None
    return EventSourceResponse(_event_stream(request.query, trace_id, override))


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    trace_id = request.session_id or str(uuid.uuid4())
    set_trace_id(trace_id)
    start_trace_collector()
    override = request.model_override.model_dump() if request.model_override else None
    result = await compiled_graph.ainvoke({
        "query": request.query, "trace_id": trace_id, "loop_count": 0, "model_override": override,
    })
    return _build_response_payload(request.query, result, trace_id)


app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok", "app": "orchestrator-gateway", "env": settings.ENV}


@app.get("/status")
async def status():
    """
    Aggregated health of all four services — powers the Streamlit UI's
    top status bar (Weaviate / Orchestrator / RAG agent / Tool agent).
    """
    return get_system_status()


@app.get("/agents")
async def list_agents():
    """Discovery endpoint — fetches each agent's card over gRPC."""
    agents = {}
    for name, host, port in [
        ("rag_agent", settings.RAG_AGENT_GRPC_HOST, settings.RAG_AGENT_GRPC_PORT),
        ("tool_agent", settings.TOOL_AGENT_GRPC_HOST, settings.TOOL_AGENT_GRPC_PORT),
    ]:
        try:
            agents[name] = get_agent_card(host, port)
        except Exception as exc:
            agents[name] = {"error": str(exc)}
    return agents


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator_service.app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
