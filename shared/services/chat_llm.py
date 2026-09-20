"""
Multi-provider chat LLM factory.

Agents never instantiate a provider SDK directly — they call
`get_chat_llm()` and get back a LangChain-compatible chat model, chosen
by `LLM_PROVIDER` in .env. This keeps provider swapping to a single
config change.
"""
import time
from functools import lru_cache

from shared.core.config import get_settings


@lru_cache
def get_chat_llm(
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    model_override: str | None = None,
    provider_override: str | None = None,
):
    """
    Returns a LangChain chat model instance for the configured provider.
    Cached per unique parameter combo so nodes can request slightly
    different sampling params (e.g. orchestrator wants low temp for
    routing, generation wants higher) without re-instantiating clients.

    provider_override / model_override let a caller switch provider AND
    model per-request (e.g. the Streamlit UI's model selector) without
    touching .env — see orchestrator_service/app/schemas.py's
    ChatRequest.model_override, threaded through via GraphState and the
    gRPC context map to the agent services.
    """
    settings = get_settings()
    temperature = settings.TEMPERATURE if temperature is None else temperature
    top_p = settings.TOP_P if top_p is None else top_p
    max_tokens = settings.MAX_TOKENS if max_tokens is None else max_tokens
    provider = provider_override or settings.LLM_PROVIDER

    # settings.LLM_MODEL is only a sensible default when we're actually
    # using the .env-configured provider — if provider_override switches
    # to a different provider (e.g. "local") with no explicit model_override,
    # settings.LLM_MODEL (e.g. an Anthropic model name) would be wrong for
    # it, so each branch below falls back to its OWN default instead.
    model = model_override or (settings.LLM_MODEL if not provider_override else None)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or settings.OPENAI_MODEL_FALLBACK,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or settings.ANTHROPIC_MODEL_FALLBACK,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    if provider == "local":
        # Any OpenAI-compatible local server (Ollama, vLLM, LM Studio, etc.)
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or settings.LOCAL_MODEL_FALLBACK,
            base_url=settings.LOCAL_LLM_BASE_URL,
            api_key=settings.LOCAL_LLM_API_KEY,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def invoke_and_track(llm, messages, node: str, agent: str, model_name: str | None = None):
    """
    Invokes a chat model and records token usage + latency into the
    current request's TraceCollector (if one is active) — this is the
    single place every agent should call through so the Streamlit
    "thinking" trace gets populated consistently.

    Returns the raw AIMessage response, same as llm.invoke() would.
    """
    from shared.core.logging import record_llm_usage  # local import avoids a cycle

    start = time.perf_counter()
    response = llm.invoke(messages)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
    output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0

    resolved_model = model_name or getattr(llm, "model", None) or getattr(llm, "model_name", None)
    record_llm_usage(
        node=node, agent=agent, latency_ms=latency_ms, model=resolved_model,
        input_tokens=input_tokens, output_tokens=output_tokens,
    )
    return response
