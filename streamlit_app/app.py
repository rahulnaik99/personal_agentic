"""
Streamlit UI — redesigned per wireframe:
  - Top status bar: Weaviate / Orchestrator / RAG Agent / Tool Agent health
  - Chat tab: request/response bubbles, model selector, collapsible raw
    per-turn JSON logs panel
  - Ingestion tab: file + category upload with a live staged progress
    checklist (SSE), plus a live dashboard (Weaviate size, categories,
    active/inactive chunk counts)
"""
import json
import os
import uuid

import httpx
import streamlit as st
from httpx_sse import connect_sse

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000")
RAG_INGEST_URL = os.environ.get("RAG_INGEST_URL", "http://localhost:8001")

INGEST_STAGES = ["loading_file", "checking_duplicate", "chunking", "embedding_and_storing", "completed"]
STAGE_LABELS = {
    "loading_file": "Loading file",
    "checking_duplicate": "Checking duplicate",
    "chunking": "Chunking",
    "embedding_and_storing": "Embedding and storing",
    "completed": "Completed",
}

st.set_page_config(page_title="Agentic Assistant", page_icon="🧭", layout="wide")

# ---- Session state ----
defaults = {
    "messages": [],
    "session_id": str(uuid.uuid4()),
    "cumulative_cost_usd": 0.0,
    "cumulative_tokens": {"input_tokens": 0, "output_tokens": 0},
    "show_logs": True,
    "model_provider": "anthropic",
    "model_name": "",
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


# ---- Status bar ----
def fetch_status() -> dict:
    try:
        resp = httpx.get(f"{ORCHESTRATOR_URL}/status", timeout=6)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"weaviate": "down", "orchestrator": "down", "rag_agent": "down", "tool_agent": "down"}


def render_status_bar():
    status = fetch_status()
    labels = [("weaviate", "Weaviate"), ("orchestrator", "Orch agent"), ("rag_agent", "Agent1 (RAG)"), ("tool_agent", "Agent (Tool)")]
    cols = st.columns(len(labels))
    for col, (key, label) in zip(cols, labels, strict=True):
        ok = status.get(key) == "ok"
        col.markdown(f"{'🟢' if ok else '🔴'} **{label}**")


# ---- Model override helper ----
def get_model_override() -> dict | None:
    if not st.session_state.model_provider:
        return None
    override = {"provider": st.session_state.model_provider}
    if st.session_state.model_name.strip():
        override["model"] = st.session_state.model_name.strip()
    return override


# ==================== CHAT TAB ====================

def render_trace_summary(payload: dict) -> None:
    trace = payload.get("trace") or []
    total_cost = payload.get("total_cost_usd") or 0.0
    total_tokens = payload.get("total_tokens") or {}
    with st.expander("Trace summary", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Cost", f"${total_cost:.5f}")
        c2.metric("Input tokens", total_tokens.get("input_tokens", 0))
        c3.metric("Output tokens", total_tokens.get("output_tokens", 0))
        if trace:
            st.dataframe(
                [{"Node": s.get("node"), "Agent": s.get("agent"), "Latency (ms)": s.get("latency_ms"),
                  "Model": s.get("model") or "—", "Cost ($)": round(s.get("cost_usd") or 0.0, 6)} for s in trace],
                use_container_width=True, hide_index=True,
            )


def stream_chat(query: str, status_box) -> dict:
    final_payload: dict = {}
    payload_body = {
        "query": query, "session_id": st.session_state.session_id,
        "model_override": get_model_override(),
    }
    with httpx.Client(timeout=120) as client:
        with connect_sse(client, "POST", f"{ORCHESTRATOR_URL}/chat/stream", json=payload_body) as event_source:
            for sse in event_source.iter_sse():
                if sse.event == "progress":
                    data = json.loads(sse.data)
                    status_box.update(label=data.get("label", "Working..."))
                    status_box.write(f"→ {data.get('label')}")
                elif sse.event == "done":
                    final_payload = json.loads(sse.data)
    return final_payload


def render_chat_tab():
    chat_col, logs_col = st.columns([2, 1]) if st.session_state.show_logs else (st.container(), None)

    with chat_col:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query := st.chat_input("Type here..."):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                status_box = st.status("Thinking...", expanded=True)
                try:
                    payload = stream_chat(query, status_box)
                except Exception as exc:
                    status_box.update(label="Error", state="error")
                    st.error(f"Could not reach the orchestrator at {ORCHESTRATOR_URL}: {exc}")
                    payload = None

                if payload:
                    status_box.update(label="Done", state="complete", expanded=False)
                    answer = payload.get("answer", "(no answer)")
                    st.markdown(answer)
                    render_trace_summary(payload)

                    st.session_state.cumulative_cost_usd += payload.get("total_cost_usd") or 0.0
                    tokens = payload.get("total_tokens") or {}
                    st.session_state.cumulative_tokens["input_tokens"] += tokens.get("input_tokens", 0)
                    st.session_state.cumulative_tokens["output_tokens"] += tokens.get("output_tokens", 0)

                    st.session_state.messages.append({"role": "assistant", "content": answer, "turn_log": payload.get("turn_log")})
                    st.rerun()

    if logs_col is not None:
        with logs_col:
            st.markdown("**Logs** _(can be hidden/unhidden)_")
            last_assistant = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
            if last_assistant and last_assistant.get("turn_log"):
                st.json(last_assistant["turn_log"])
            else:
                st.caption("No turns yet.")

    # ---- Bottom controls: model selector + logs toggle ----
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
    with ctrl1:
        st.selectbox("Select model", ["local", "openai", "anthropic"], key="model_provider")
    with ctrl2:
        st.text_input("Model name (optional)", key="model_name", placeholder="e.g. llama3.1, gpt-4o-mini")
    with ctrl3:
        st.toggle("Show logs", key="show_logs")


# ==================== INGESTION TAB ====================

def fetch_stats() -> dict:
    try:
        resp = httpx.get(f"{RAG_INGEST_URL}/stats", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


def render_dashboard():
    st.markdown("**Dashboard**")
    stats = fetch_stats()
    if "error" in stats:
        st.caption(f"Dashboard unavailable: {stats['error']}")
        return
    st.write(f"Current size of Weaviate (total objects): **{stats.get('total_objects', 0)}**")
    st.write("Types of category — no. of files:")
    categories = stats.get("total_files_by_category") or {}
    if categories:
        for category, count in categories.items():
            st.write(f"&nbsp;&nbsp;• {category}: **{count}**")
    else:
        st.write("&nbsp;&nbsp;_(none ingested yet)_")
    st.write(f"Total active chunks: **{stats.get('total_chunks_active', 0)}**")
    st.write(f"Total active embeddings: **{stats.get('total_embeddings_active', 0)}**")
    st.write(f"Total inactive chunks: **{stats.get('total_chunks_inactive', 0)}**")
    if st.button("Refresh dashboard"):
        st.rerun()


def stream_ingest(file_bytes: bytes, filename: str, category: str, checklist_placeholders: dict, progress_bar):
    files = {"file": (filename, file_bytes)}
    data = {"category": category}
    final_result = None

    with httpx.Client(timeout=300) as client:
        with client.stream("POST", f"{RAG_INGEST_URL}/ingest/file/stream", files=files, data=data) as response:
            event_name, event_data = None, None
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

                    if stage in checklist_placeholders:
                        idx = INGEST_STAGES.index(stage)
                        progress_bar.progress(int((idx + 1) / len(INGEST_STAGES) * 100))
                        if payload.get("status") == "done":
                            checklist_placeholders[stage].markdown(f"✅ {STAGE_LABELS[stage]}")
                        else:
                            checklist_placeholders[stage].markdown(f"⏳ {STAGE_LABELS[stage]}")

                    if stage == "completed":
                        final_result = payload.get("result")
                        progress_bar.progress(100)
                        # If the file was an unchanged duplicate, stages after
                        # checking_duplicate never ran — mark them skipped.
                        if payload.get("result", {}).get("status") == "unchanged":
                            for skipped_stage in ("chunking", "embedding_and_storing"):
                                checklist_placeholders[skipped_stage].markdown(f"⏭️ {STAGE_LABELS[skipped_stage]} (skipped — duplicate)")

                    event_name, event_data = None, None
    return final_result


def render_ingestion_tab():
    left, right = st.columns([2, 1])

    with left:
        uploaded_file = st.file_uploader("Select file to ingest", type=["pdf", "txt"])
        category = st.text_input("Category", placeholder="e.g. profession_doc, financial_doc", value="uncategorized")

        if uploaded_file and st.button("Ingest", type="primary"):
            progress_bar = st.progress(0)
            checklist_placeholders = {stage: st.empty() for stage in INGEST_STAGES}
            for stage in INGEST_STAGES:
                checklist_placeholders[stage].markdown(f"⬜ {STAGE_LABELS[stage]}")

            result = stream_ingest(uploaded_file.getvalue(), uploaded_file.name, category, checklist_placeholders, progress_bar)

            if result:
                st.success(f"Ingestion {result['status']}: {result['parents']} parent chunks, {result['children']} child chunks.")

    with right:
        render_dashboard()


# ==================== MAIN ====================

st.title("🧭 Agentic Assistant")
render_status_bar()

chat_tab, ingestion_tab = st.tabs(["Chat", "Ingestion"])
with chat_tab:
    render_chat_tab()
with ingestion_tab:
    render_ingestion_tab()
