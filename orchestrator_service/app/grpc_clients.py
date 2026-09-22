"""
gRPC clients the orchestrator uses to dispatch tasks to the RAG and tool
agent services. Each call also records a StepRecord into the
orchestrator's own TraceCollector — using the token usage the remote
agent reported back in its response — so the Streamlit "thinking" trace
shows a unified view across all services for one request, even though
the actual LLM calls happened in a different process.
"""
import json
import time
import uuid

import grpc

from shared.core.config import get_settings
from shared.core.logging import StepRecord, get_trace_collector, log_step
from shared.core.pricing import estimate_cost_usd
from shared.generated import agent_pb2, agent_pb2_grpc


def _record_remote_steps(node: str, agent: str, latency_ms: float, usage_list) -> None:
    collector = get_trace_collector()
    if collector is None:
        return
    if not usage_list:
        collector.add(StepRecord(node=node, agent=agent, latency_ms=latency_ms))
        return
    for u in usage_list:
        cost = estimate_cost_usd(u.model, u.input_tokens, u.output_tokens)
        collector.add(StepRecord(
            node=node, agent=agent, latency_ms=latency_ms, model=u.model,
            input_tokens=u.input_tokens, output_tokens=u.output_tokens, cost_usd=cost,
        ))


def _build_context(category: str | None = None, model_override: dict | None = None, conversation_history: list[dict[str, str]] | None = None) -> dict[str, str]:
    """
    Builds the gRPC SendTaskRequest.context map (string->string only, per
    the proto). model_override is {"provider": ..., "model": ...} from
    ChatRequest.model_override — split into two flat keys here since
    proto maps can't carry nested structures.
    """
    context: dict[str, str] = {}
    if category:
        context["category"] = category
    if conversation_history:
        context["conversation_history"] = json.dumps(conversation_history, ensure_ascii=False)
    if model_override:
        if model_override.get("provider"):
            context["model_provider"] = model_override["provider"]
        if model_override.get("model"):
            context["model_name"] = model_override["model"]
    return context


def call_rag_agent(
    query: str, trace_id: str, category: str | None = None,
    model_override: dict | None = None, timeout: int | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict:
    settings = get_settings()
    address = f"{settings.RAG_AGENT_GRPC_HOST}:{settings.RAG_AGENT_GRPC_PORT}"
    task_id = str(uuid.uuid4())
    request_context = _build_context(category, model_override, conversation_history)

    with log_step("dispatch_rag_agent", agent="orchestrator", query=query, address=address) as ctx:
        start = time.perf_counter()
        with grpc.insecure_channel(address) as channel:
            stub = agent_pb2_grpc.AgentServiceStub(channel)
            request = agent_pb2.SendTaskRequest(
                task_id=task_id, trace_id=trace_id, query=query, context=request_context,
            )
            response = stub.SendTask(request, timeout=timeout or settings.GRPC_CALL_TIMEOUT_SECONDS)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        ctx["output"] = {"state": response.state, "error": response.error or None}

    _record_remote_steps("rag_agent_call", "rag_agent", latency_ms, response.usage)

    if response.state != agent_pb2.TASK_STATE_COMPLETED:
        return {"answer": "", "sufficient": False, "ragas_scores": {}, "contexts": [],
                "guardrail_flags": [], "category_used": None, "error": response.error}

    return {
        "answer": response.result_text,
        "sufficient": response.sufficient,
        "ragas_scores": json.loads(response.metadata.get("ragas_scores", "{}")),
        "contexts": json.loads(response.metadata.get("contexts", "[]")),
        "guardrail_flags": json.loads(response.metadata.get("guardrail_flags", "[]")),
        "category_used": response.metadata.get("category_used") or None,
        "error": None,
    }


def call_tool_agent(query: str, trace_id: str, model_override: dict | None = None, timeout: int | None = None, conversation_history: list[dict[str, str]] | None = None) -> dict:
    settings = get_settings()
    address = f"{settings.TOOL_AGENT_GRPC_HOST}:{settings.TOOL_AGENT_GRPC_PORT}"
    task_id = str(uuid.uuid4())
    request_context = _build_context(model_override=model_override, conversation_history=conversation_history)

    with log_step("dispatch_rag_agent", agent="orchestrator", query=query, address=address) as ctx:
        start = time.perf_counter()
        with grpc.insecure_channel(address) as channel:
            stub = agent_pb2_grpc.AgentServiceStub(channel)
            request = agent_pb2.SendTaskRequest(
                task_id=task_id, trace_id=trace_id, query=query, context=request_context,
            )
            response = stub.SendTask(request, timeout=timeout or settings.GRPC_CALL_TIMEOUT_SECONDS)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        ctx["output"] = {"state": response.state, "error": response.error or None}

    _record_remote_steps("tool_agent_call", "tool_agent", latency_ms, response.usage)

    if response.state != agent_pb2.TASK_STATE_COMPLETED:
        return {"answer": "", "tool_calls_made": [], "guardrail_flags": [], "error": response.error}

    return {
        "answer": response.result_text,
        "tool_calls_made": json.loads(response.metadata.get("tool_calls_made", "[]")),
        "guardrail_flags": json.loads(response.metadata.get("guardrail_flags", "[]")),
        "error": None,
    }


def get_agent_card(host: str, port: int) -> dict:
    """Fetches an agent's card for discovery/health purposes."""
    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = agent_pb2_grpc.AgentServiceStub(channel)
        card = stub.GetAgentCard(agent_pb2.GetAgentCardRequest(), timeout=5)
    return {
        "name": card.name, "version": card.version, "description": card.description,
        "skills": [{"id": s.id, "name": s.name, "description": s.description} for s in card.skills],
        "input_modes": list(card.input_modes), "output_modes": list(card.output_modes),
    }
