"""
Aggregated system health, used by the Streamlit UI's top status bar
(Weaviate / Orchestrator / RAG agent / Tool agent pills).

Each check has a short timeout and never raises — a down dependency
should degrade the status response, not break the endpoint itself.
"""
import httpx

from orchestrator_service.app.grpc_clients import get_agent_card
from shared.core.config import get_settings

HEALTH_CHECK_TIMEOUT_SECONDS = 5


def _check_agent(host: str, port: int) -> str:
    try:
        get_agent_card(host, port)
        return "ok"
    except Exception:
        return "down"


def _check_weaviate_via_rag_agent() -> str:
    """
    The orchestrator has no direct Weaviate connection — only the RAG
    agent does. So Weaviate health is read from the RAG agent's own
    /health, which pings Weaviate itself (see rag_agent_service/app/ingest_api.py).
    """
    settings = get_settings()
    try:
        resp = httpx.get(f"{settings.RAG_AGENT_INGEST_URL}/health", timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        return "ok" if data.get("weaviate_ready") else "down"
    except Exception:
        return "down"


def get_system_status() -> dict:
    settings = get_settings()
    return {
        "weaviate": _check_weaviate_via_rag_agent(),
        "orchestrator": "ok",  # answering this request at all proves it's up
        "rag_agent": _check_agent(settings.RAG_AGENT_GRPC_HOST, settings.RAG_AGENT_GRPC_PORT),
        "tool_agent": _check_agent(settings.TOOL_AGENT_GRPC_HOST, settings.TOOL_AGENT_GRPC_PORT),
    }
