"""Professional Streamlit UI for the agentic assistant.

Features:
- ChatGPT-style sidebar with independent chat sessions.
- Persistent provider/model selection for the lifetime of the Streamlit session.
- MLX local model provider support via mlx-lm's OpenAI-compatible server.
- Live orchestrator/agent status.
- Streaming agent progress and per-turn diagnostics.
- Knowledge-base ingestion and dashboard.
"""

import json
import os
import uuid
from datetime import datetime

import httpx
import streamlit as st
from httpx_sse import connect_sse


ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")
RAG_INGEST_URL = os.environ.get("RAG_INGEST_URL", "http://localhost:8001")

INGEST_STAGES = [
    "loading_file",
    "checking_duplicate",
    "chunking",
    "embedding_and_storing",
    "completed",
]

STAGE_LABELS = {
    "loading_file": "Loading file",
    "checking_duplicate": "Checking duplicate",
    "chunking": "Chunking",
    "embedding_and_storing": "Embedding and storing",
    "completed": "Completed",
}

CATEGORY_OPTIONS = [
    "personal", "professional", "financial", "goals", "research",
    "future_planning", "health", "family", "academic", "work", "career",
    "education", "legal", "tax", "insurance", "investments", "business",
    "projects", "technology", "travel", "relationships", "fitness", "nutrition",
    "housing", "vehicles", "shopping", "subscriptions", "documents",
    "reference", "notes", "other",
]

MODEL_PROVIDERS = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "local": "Ollama / OpenAI-compatible",
    "mlx": "MLX — Apple Silicon",
}

MODEL_CATALOG = {
    "anthropic": {
        "claude-sonnet-4-6": {"label": "Claude Sonnet 4.6", "input": 3.00, "output": 15.00},
        "claude-haiku-4-5": {"label": "Claude Haiku 4.5", "input": 1.00, "output": 5.00},
    },
    "openai": {
        "gpt-4o": {"label": "GPT-4o", "input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"label": "GPT-4o mini", "input": 0.15, "output": 0.60},
    },
    "local": {
        "qwen3:8b": {"label": "Qwen3 8B", "input": 0.00, "output": 0.00},
        "qwen3:4b": {"label": "Qwen3 4B", "input": 0.00, "output": 0.00},
        "qwen3.5:4b": {"label": "Qwen3.5 4B", "input": 0.00, "output": 0.00},
        "gemma:latest": {"label": "Gemma", "input": 0.00, "output": 0.00},
    },
    "mlx": {
        "mlx-community/DeepSeek-R1-Distill-Qwen-14B-MLX": {
            "label": "DeepSeek R1 Distill Qwen 14B (MLX)",
            "input": 0.00,
            "output": 0.00,
        },
    },
}

MODEL_PLACEHOLDERS = {
    provider: next(iter(models))
    for provider, models in MODEL_CATALOG.items()
}

st.set_page_config(
    page_title="Agentic Assistant",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Styling
# ============================================================================

st.markdown(
    """
<style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    .block-container {
        max-width: 1500px;
        padding-top: 1.25rem;
        padding-bottom: 6rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,.18);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 4px 0 18px 4px;
    }
    .brand-icon {
        width: 36px;
        height: 36px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #111827;
        color: white;
        font-size: 19px;
    }
    .brand-title { font-weight: 700; font-size: 17px; line-height: 1.1; }
    .brand-subtitle { color: #8b8f98; font-size: 11px; margin-top: 3px; }

    .session-label {
        color: #8b8f98;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .07em;
        text-transform: uppercase;
        margin: 20px 0 8px 5px;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(128,128,128,.18);
        padding: 0 0 14px 0;
        margin-bottom: 16px;
    }
    .topbar-title { font-size: 22px; font-weight: 700; }
    .topbar-subtitle { color: #8b8f98; font-size: 12px; margin-top: 3px; }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 999px;
        padding: 5px 9px;
        margin-left: 5px;
        font-size: 11px;
    }
    .status-dot { font-size: 8px; }

    .empty-chat {
        min-height: 48vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: #8b8f98;
    }
    .empty-chat .emoji { font-size: 38px; margin-bottom: 8px; }
    .empty-chat h2 { color: inherit; margin: 0; font-size: 24px; }
    .empty-chat p { max-width: 520px; font-size: 13px; }

    .session-card {
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }

    .composer-hint {
        color: #8b8f98;
        font-size: 11px;
        text-align: center;
        margin-top: 5px;
    }

    .metric-strip {
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 14px;
        padding: 10px 12px;
        margin-bottom: 12px;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
    }

    .agent-chip {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 999px;
        background: rgba(99,102,241,.10);
        font-size: 11px;
        margin-right: 4px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# Session state
# ============================================================================


def _new_session() -> dict:
    session_id = str(uuid.uuid4())
    return {
        "id": session_id,
        "title": "New chat",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],
        "cumulative_cost_usd": 0.0,
        "cumulative_tokens": {"input_tokens": 0, "output_tokens": 0},
    }


if "sessions" not in st.session_state:
    first = _new_session()
    st.session_state.sessions = {first["id"]: first}
    st.session_state.active_session_id = first["id"]

# Migrate sessions created by older versions of the UI. Older session
# dictionaries may not have ``created_at`` (or other newer fields).
for _sid, _session in st.session_state.sessions.items():
    _session.setdefault("id", _sid)
    _session.setdefault("title", "New chat")
    _session.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    _session.setdefault("messages", [])
    _session.setdefault("cumulative_cost_usd", 0.0)
    _session.setdefault(
        "cumulative_tokens",
        {"input_tokens": 0, "output_tokens": 0},
    )
    _session["cumulative_tokens"].setdefault("input_tokens", 0)
    _session["cumulative_tokens"].setdefault("output_tokens", 0)

# These settings intentionally live outside individual chats: changing the
# model is a user preference and therefore applies to the next turn.
st.session_state.setdefault("model_provider", "anthropic")
st.session_state.setdefault("model_name", "")
st.session_state.setdefault("show_logs", False)


def current_session() -> dict:
    sid = st.session_state.active_session_id
    session = st.session_state.sessions[sid]

    # Backward-compatible migration for sessions created by older UI builds.
    # The request ID is always the dictionary key, so never depend on a
    # potentially missing ``session["id"]`` field.
    session.setdefault("id", sid)
    session.setdefault("title", "New chat")
    session.setdefault("messages", [])
    session.setdefault("cumulative_cost_usd", 0.0)
    session.setdefault(
        "cumulative_tokens",
        {"input_tokens": 0, "output_tokens": 0},
    )
    if "output_tokens" not in session["cumulative_tokens"]:
        session["cumulative_tokens"]["output_tokens"] = 0
    if "input_tokens" not in session["cumulative_tokens"]:
        session["cumulative_tokens"]["input_tokens"] = 0

    return session


def create_new_session() -> None:
    session = _new_session()
    st.session_state.sessions[session["id"]] = session
    st.session_state.active_session_id = session["id"]


def activate_session(session_id: str) -> None:
    if session_id in st.session_state.sessions:
        st.session_state.active_session_id = session_id


def rename_from_query(query: str) -> str:
    clean = " ".join(query.strip().split())
    return clean[:42] + ("…" if len(clean) > 42 else "") or "New chat"


# ============================================================================
# Sidebar
# ============================================================================


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-icon">🧭</div>'
            '<div><div class="brand-title">Agentic Assistant</div>'
            '<div class="brand-subtitle">RAG · Tools · LangGraph</div></div></div>',
            unsafe_allow_html=True,
        )

        if st.button("＋  New chat", use_container_width=True, type="primary"):
            create_new_session()
            st.rerun()

        st.markdown('<div class="session-label">Sessions</div>', unsafe_allow_html=True)

        sessions = list(st.session_state.sessions.values())
        # Defensive fallback so a legacy session can never crash the sidebar.
        sessions.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )

        for session in sessions:
            active = session["id"] == st.session_state.active_session_id
            label = f"💬  {session['title']}"
            if active:
                label = "● " + label
            if st.button(label, key=f"session_{session['id']}", use_container_width=True):
                activate_session(session["id"])
                st.rerun()

        st.markdown('<div class="session-label">Model</div>', unsafe_allow_html=True)

        provider = st.selectbox(
            "Provider",
            options=list(MODEL_PROVIDERS),
            format_func=lambda value: MODEL_PROVIDERS[value],
            key="model_provider",
        )

        available_models = MODEL_CATALOG[provider]
        model_keys = list(available_models)

        # Preserve a manually selected model when switching providers, but
        # automatically choose the first valid model for the new provider.
        if st.session_state.get("model_name") not in model_keys:
            st.session_state.model_name = model_keys[0]

        selected_model = st.selectbox(
            "Model",
            options=model_keys,
            format_func=lambda value: available_models[value]["label"],
            key="model_name",
        )

        model_meta = available_models[selected_model]
        input_price = model_meta["input"]
        output_price = model_meta["output"]

        if input_price == 0 and output_price == 0:
            st.caption("💰 Cost: $0 API cost · runs locally")
        else:
            blended = (input_price * 0.8) + (output_price * 0.2)
            st.caption(
                f"💰 ${blended:.2f}/1M tokens typical · "
                f"input ${input_price:.2f} · output ${output_price:.2f}"
            )

        st.caption(
            "Model selection persists for the next turn."
        )

        if provider == "mlx":
            st.info("MLX endpoint: http://127.0.0.1:8080/v1", icon="🍎")

        session = current_session()
        st.caption(
            f"Session cost: ${session['cumulative_cost_usd']:.5f} · "
            f"{session['cumulative_tokens']['input_tokens'] + session['cumulative_tokens']['output_tokens']:,} tokens"
        )

        st.markdown('<div class="session-label">Conversation</div>', unsafe_allow_html=True)
        st.toggle("Show diagnostics", key="show_logs")

        session = current_session()
        if st.button("🗑  Clear current chat", use_container_width=True):
            session["messages"] = []
            session["title"] = "New chat"
            session["cumulative_cost_usd"] = 0.0
            session["cumulative_tokens"] = {"input_tokens": 0, "output_tokens": 0}
            st.rerun()


render_sidebar()


# ============================================================================
# API / status helpers
# ============================================================================


def fetch_status() -> dict:
    try:
        response = httpx.get(f"{ORCHESTRATOR_URL}/status", timeout=6)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {"weaviate": "down", "orchestrator": "down", "rag_agent": "down", "tool_agent": "down"}


def render_topbar() -> None:
    status = fetch_status()
    labels = [
        ("weaviate", "Weaviate"),
        ("orchestrator", "Orchestrator"),
        ("rag_agent", "RAG Agent"),
        ("tool_agent", "Tool Agent"),
    ]
    pills = []
    for key, label in labels:
        ok = status.get(key) == "ok"
        pills.append(
            f'<span class="status-pill"><span class="status-dot">{"🟢" if ok else "🔴"}</span>{label}</span>'
        )
    session = current_session()
    st.markdown(
        f'<div class="topbar"><div><div class="topbar-title">{session["title"]}</div>'
        f'<div class="topbar-subtitle">Session {session["id"][:8]} · {MODEL_PROVIDERS[st.session_state.model_provider]}</div></div>'
        f'<div>{"".join(pills)}</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# Model / chat
# ============================================================================


def get_model_override() -> dict | None:
    provider = (st.session_state.get("model_provider") or "").strip()
    if not provider:
        return None
    override = {"provider": provider}
    model = (st.session_state.get("model_name") or "").strip()
    if model:
        override["model"] = model
    return override


def stream_chat(query: str, status_box) -> dict:
    session = current_session()
    active_session_id = st.session_state.active_session_id

    # The current user turn is already stored in the UI session before this
    # function is called. The API receives it separately as ``query``.
    history = [
        {
            "role": m["role"],
            "content": m["content"],
        }
        for m in session["messages"][-12:]
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]

    if history and history[-1]["role"] == "user" and history[-1]["content"] == query:
        history.pop()

    payload_body = {
        "query": query,
        "session_id": active_session_id,
        "model_override": get_model_override(),
        "conversation_history": history,
    }
    final_payload: dict = {}

    with httpx.Client(timeout=300) as client:
        with connect_sse(
            client,
            "POST",
            f"{ORCHESTRATOR_URL}/chat/stream",
            json=payload_body,
        ) as event_source:
            for sse in event_source.iter_sse():
                if sse.event == "progress":
                    data = json.loads(sse.data)
                    status_box.update(label=data.get("label", "Working…"))
                    status_box.write(f"→ {data.get('label', 'Working…')}")
                elif sse.event == "done":
                    final_payload = json.loads(sse.data)
    return final_payload


def render_trace_summary(payload: dict) -> None:
    trace = payload.get("trace") or []
    total_cost = payload.get("total_cost_usd") or 0.0
    tokens = payload.get("total_tokens") or {}

    with st.expander("Turn diagnostics", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route", payload.get("route", "unknown"))
        c2.metric("Cost", f"${total_cost:.5f}")
        c3.metric("Input", tokens.get("input_tokens", 0))
        c4.metric("Output", tokens.get("output_tokens", 0))

        turn_log = payload.get("turn_log") or {}
        agents = turn_log.get("agent_invoked") or []
        if agents:
            st.markdown(" ".join(f'<span class="agent-chip">{a}</span>' for a in agents), unsafe_allow_html=True)

        if trace:
            st.dataframe(
                [
                    {
                        "Node": step.get("node"),
                        "Agent": step.get("agent"),
                        "Latency (ms)": step.get("latency_ms"),
                        "Model": step.get("model") or "—",
                        "Cost ($)": round(step.get("cost_usd") or 0.0, 6),
                    }
                    for step in trace
                ],
                use_container_width=True,
                hide_index=True,
            )

        # Streamlit does not allow an expander inside another expander.
        # Keep the raw JSON inside the existing "Turn diagnostics" expander.
        st.markdown("**Raw turn JSON**")
        st.code(
            json.dumps(turn_log or payload, indent=2, ensure_ascii=False, default=str),
            language="json",
        )


def render_chat() -> None:
    session = current_session()
    messages = session["messages"]

    if not messages:
        st.markdown(
            '<div class="empty-chat"><div class="emoji">🧭</div>'
            '<h2>How can I help?</h2>'
            '<p>Ask about your ingested knowledge, request live web research, or use me for general reasoning and coding.</p></div>',
            unsafe_allow_html=True,
        )

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("payload"):
                render_trace_summary(message["payload"])

    query = st.chat_input("Message your assistant…")
    if not query:
        st.markdown('<div class="composer-hint">Enter to send · Your selected model stays active for the next turn</div>', unsafe_allow_html=True)
        return

    if session["title"] == "New chat":
        session["title"] = rename_from_query(query)

    session["messages"].append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        status_box = st.status("Thinking…", expanded=True)
        try:
            payload = stream_chat(query, status_box)
            if not payload:
                raise RuntimeError("The orchestrator returned an empty response.")

            status_box.update(label="Done", state="complete", expanded=False)
            answer = payload.get("answer") or "I wasn't able to produce an answer."
            st.markdown(answer)
            render_trace_summary(payload)

            session["messages"].append({
                "role": "assistant",
                "content": answer,
                "payload": payload,
            })

            session["cumulative_cost_usd"] += payload.get("total_cost_usd") or 0.0
            turn_tokens = payload.get("total_tokens") or {}
            session["cumulative_tokens"]["input_tokens"] += turn_tokens.get("input_tokens", 0)
            session["cumulative_tokens"]["output_tokens"] += turn_tokens.get("output_tokens", 0)
            st.rerun()

        except Exception as exc:
            status_box.update(label="Request failed", state="error", expanded=False)
            error_text = f"I couldn't complete the request.\n\n`{exc}`"
            st.error(error_text)
            if st.session_state.show_logs:
                st.caption(
                    f"Request ID: `{st.session_state.active_session_id}`"
                )
            session["messages"].append({"role": "assistant", "content": error_text})
            st.rerun()


# ============================================================================
# Ingestion
# ============================================================================


def fetch_stats() -> dict:
    try:
        response = httpx.get(f"{RAG_INGEST_URL}/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def render_dashboard() -> None:
    stats = fetch_stats()
    if "error" in stats:
        st.warning(f"Knowledge-base dashboard unavailable: {stats['error']}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objects", stats.get("total_objects", 0))
    c2.metric("Active chunks", stats.get("total_chunks_active", 0))
    c3.metric("Active embeddings", stats.get("total_embeddings_active", 0))
    c4.metric("Inactive chunks", stats.get("total_chunks_inactive", 0))

    categories = stats.get("total_files_by_category") or {}
    if categories:
        st.dataframe(
            [{"Category": k, "Files": v} for k, v in categories.items()],
            use_container_width=True,
            hide_index=True,
        )

    if st.button("Refresh dashboard"):
        st.rerun()


def stream_ingest(file_bytes: bytes, filename: str, category: str, placeholders: dict, progress_bar):
    files = {"file": (filename, file_bytes)}
    data = {"category": category}
    final_result = None

    with httpx.Client(timeout=300) as client:
        with client.stream("POST", f"{RAG_INGEST_URL}/ingest/file/stream", files=files, data=data) as response:
            response.raise_for_status()
            event_name = None
            event_data = None

            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    event_data = line.split(":", 1)[1].strip()
                elif line == "" and event_name and event_data:
                    payload = json.loads(event_data)
                    stage = payload.get("stage")

                    if event_name == "error":
                        st.error(f"Ingestion error: {payload.get('error')}")
                        return None

                    if stage in placeholders:
                        idx = INGEST_STAGES.index(stage)
                        progress_bar.progress(int((idx + 1) / len(INGEST_STAGES) * 100))
                        icon = "✅" if payload.get("status") == "done" else "⏳"
                        placeholders[stage].markdown(f"{icon} {STAGE_LABELS[stage]}")

                    if stage == "completed":
                        final_result = payload.get("result")
                        progress_bar.progress(100)
                        if (payload.get("result") or {}).get("status") == "unchanged":
                            for skipped_stage in ("chunking", "embedding_and_storing"):
                                placeholders[skipped_stage].markdown(
                                    f"⏭️ {STAGE_LABELS[skipped_stage]} (skipped — duplicate)"
                                )

                    event_name = None
                    event_data = None

    return final_result


def render_ingestion() -> None:
    st.subheader("Knowledge base")
    st.caption("Upload documents into the versioned parent/child RAG store.")

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        uploaded_file = st.file_uploader("Document", type=["pdf", "txt"])
        selection = st.selectbox(
            "Category",
            CATEGORY_OPTIONS,
            format_func=lambda value: "Other / Custom" if value == "other" else value.replace("_", " ").title(),
        )

        if selection == "other":
            custom = st.text_input("Custom category", placeholder="e.g. hobbies, volunteering")
            category = custom.strip().lower().replace(" ", "_")
        else:
            category = selection

        if uploaded_file:
            if st.button("Ingest document", type="primary", use_container_width=True) and category:
                progress = st.progress(0)
                placeholders = {stage: st.empty() for stage in INGEST_STAGES}
                for stage in INGEST_STAGES:
                    placeholders[stage].markdown(f"⬜ {STAGE_LABELS[stage]}")

                result = stream_ingest(
                    uploaded_file.getvalue(), uploaded_file.name, category, placeholders, progress
                )
                if result:
                    status = result.get("status", "unknown")
                    if status == "unchanged":
                        st.info("This document is unchanged from the active version; no new chunks were created.")
                    else:
                        st.success(
                            f"Ingestion {status}: {result.get('parents', 0)} parent chunks · "
                            f"{result.get('children', 0)} child chunks."
                        )
        else:
            st.info("Choose a PDF/TXT file to begin.")

    with right:
        st.markdown("### Store health")
        render_dashboard()


# ============================================================================
# Main
# ============================================================================

render_topbar()
chat_tab, knowledge_tab = st.tabs(["Chat", "Knowledge Base"])

with chat_tab:
    render_chat()

with knowledge_tab:
    render_ingestion()
