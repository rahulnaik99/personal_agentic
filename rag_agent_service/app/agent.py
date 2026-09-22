"""
RAG agent core logic — called by server.py (the gRPC handler). Kept
separate from the gRPC plumbing so it's independently testable.
"""
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from rag_agent_service.app.evaluation import evaluate_rag_response
from rag_agent_service.app.guardrails import guard_input, guard_output
from rag_agent_service.app.retriever import hybrid_retrieve
from shared.core.logging import log_step
from shared.core.prompts import load_prompt
from shared.services.chat_llm import get_chat_llm, invoke_and_track
from shared.vectorstore.weaviate_client import get_active_categories

PROMPT_PATH = Path(__file__).parent / "prompts" / "rag_system_prompt.md"
CATEGORY_PROMPT_PATH = Path(__file__).parent / "prompts" / "category_classifier_prompt.md"


class CategoryDecision(BaseModel):
    category: str | None = None
    confident: bool = False


def classify_category(query: str, model_override: dict | None = None) -> str | None:
    """
    Picks the best-matching ingestion category for a query, or returns
    None (meaning: search unfiltered across every category) when the
    classifier isn't confident or there's only one/zero categories to
    choose from — a false "wrong category" filter is worse than a
    slightly noisier unfiltered search.
    """
    categories = get_active_categories()
    if len(categories) <= 1:
        return categories[0] if categories else None

    override = model_override or {}
    with log_step("classify_category", agent="rag_agent", query=query, categories=categories) as ctx:
        system_prompt = load_prompt(str(CATEGORY_PROMPT_PATH))
        llm = get_chat_llm(
            temperature=0.0,
            provider_override=override.get("provider"),
            model_override=override.get("model"),
        )
        human = f"Available categories: {categories}\nQuestion: {query}"
        response = invoke_and_track(
            llm, [SystemMessage(content=system_prompt), HumanMessage(content=human)],
            node="classify_category", agent="rag_agent",
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)

        try:
            cleaned = raw.strip().strip("```").replace("json", "", 1).strip() if raw.strip().startswith("```") else raw.strip()
            decision = CategoryDecision.model_validate_json(cleaned)
        except Exception:
            decision = CategoryDecision(category=None, confident=False)

        ctx["output"] = decision.model_dump()

    return decision.category if decision.confident else None


def run_rag_agent(
    query: str, trace_id: str, forced_category: str | None = None, model_override: dict | None = None, conversation_history: list[dict[str, str]] | None = None,
) -> dict:
    """
    forced_category: bypasses classify_category() when the caller already
    knows the right category (e.g. passed via the gRPC SendTaskRequest's
    context map) — see orchestrator_service/app/grpc_clients.py.

    model_override: {"provider": ..., "model": ...} — lets a caller (the
    Streamlit UI's model selector, via the orchestrator) switch LLM
    provider/model per-request instead of using this service's .env
    default. Applied to both the category classifier and generation calls.

    Returns:
      answer: str
      contexts: list[str]
      sufficient: bool
      ragas_scores: dict
      guardrail_flags: list[str]
      blocked: bool
      block_reason: str | None
      category_used: str | None
    """
    input_guard = guard_input(query)
    if not input_guard.allowed:
        return {
            "answer": "", "contexts": [], "sufficient": False, "ragas_scores": {},
            "guardrail_flags": input_guard.flags, "blocked": True, "block_reason": input_guard.reason,
            "category_used": None,
        }

    override = model_override or {}
    with log_step("run_rag_agent", agent="rag_agent", query=query) as ctx:
        category = forced_category if forced_category else classify_category(query, model_override)
        contexts = hybrid_retrieve(query, category=category)
        context_texts = [
            f"[{c.element_type.upper()}] {c.text}" if c.element_type != "text" else c.text
            for c in contexts
        ]

        system_prompt = load_prompt(str(PROMPT_PATH))
        llm = get_chat_llm(
            temperature=0.1,
            provider_override=override.get("provider"),
            model_override=override.get("model"),
        )
        context_block = "\n\n---\n\n".join(context_texts) if context_texts else "(no context retrieved)"
        history = conversation_history or []
        history_block = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in history[-8:]
            if m.get("content")
        )
        user_content = (
            f"Conversation history:\n{history_block}\n\n" if history_block else ""
        ) + f"Context:\n{context_block}\n\nQuestion: {query}"
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        response = invoke_and_track(llm, messages, node="rag_generate", agent="rag_agent")
        answer = response.content if isinstance(response.content, str) else str(response.content)

        sufficient = not answer.strip().upper().startswith("INSUFFICIENT_CONTEXT")
        ragas_scores = evaluate_rag_response(
            question=query, answer=answer, contexts=context_texts
        ) if context_texts else {}

        output_guard = guard_output(answer, ragas_scores)
        final_answer = output_guard.redacted_text if output_guard.allowed else ""

        ctx["output"] = {"sufficient": sufficient, "num_contexts": len(context_texts), "category": category}

    return {
        "answer": final_answer,
        "contexts": context_texts,
        "sufficient": sufficient,
        "ragas_scores": ragas_scores,
        "guardrail_flags": output_guard.flags,
        "blocked": not output_guard.allowed,
        "block_reason": output_guard.reason,
        "category_used": category,
    }
