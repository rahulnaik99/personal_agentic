# Agentic System — Full A2A/gRPC Multi-Service Build

Three independently deployable services + a Streamlit UI, orchestrated
via LangGraph and communicating over gRPC using an A2A-inspired task
protocol.

```
streamlit-ui  →  orchestrator-service (FastAPI/SSE gateway + LangGraph)
                        │  gRPC (AgentService: GetAgentCard / SendTask / StreamTask)
                        ├──→ rag-agent-service   (+ separate HTTP ingest API)
                        └──→ tool-agent-service
                        │
                    Weaviate (rag-agent-service only)
```

## Services

- **`orchestrator_service/`** — classifies each query (`rag`/`tool`/`both`/`direct`),
  dispatches to agents over gRPC, loops RAG → tool agent when retrieved
  context is insufficient, applies input/output guardrails, and exposes
  `/chat`, `/chat/stream` (SSE), `/agents` (discovery), `/health`.
- **`rag_agent_service/`** — hybrid retrieval (dense + sparse → manual RRF
  → MMR → cross-encoder rerank → parent-chunk expansion) over Weaviate,
  RAGAS evaluation per response, PDF/TXT ingestion with table/image-aware
  parent-child chunking (tables kept whole, images captioned via a vision
  LLM call). Exposes an `AgentService` gRPC server plus a small HTTP API
  (`/ingest/text`, `/ingest/file`) for bulk ingestion.
- **`tool_agent_service/`** — free web search (DuckDuckGo) + URL fetch,
  every proposed tool call validated (schema + domain allow-list) before
  execution. Exposes an `AgentService` gRPC server.
- **`streamlit_app/`** — chat UI with a live "Thinking..." status that
  shows which agent/tool is running, then a trace table (tokens, cost,
  latency per step), RAGAS scores, and tool call details per turn.

## Shared foundation (`shared/`)

Config (all tunables via `.env`), structured JSON logging + a per-request
trace collector (feeds the Streamlit UI), pricing table for cost
estimation, `.md` prompt loader, guardrail primitives (PII/injection
detection), multi-provider LLM/embedding clients, Weaviate client, and
the generated gRPC stubs (`shared/generated/`, from `proto/agent.proto`).

## Setup

```bash
cp .env.example .env   # fill in API keys
docker compose up --build
```

This starts Weaviate, both agent services, the orchestrator, and the
Streamlit UI — visit **http://localhost:8501**.

To run services individually (e.g. for development), see each service's
Dockerfile `CMD` for the exact entrypoint command, and set `PYTHONPATH`
to the repo root.

## Ingesting documents

```bash
curl -X POST http://localhost:8001/ingest/file -F "file=@/path/to/doc.pdf"
```

## Regenerating gRPC stubs after editing `proto/agent.proto`

```bash
pip install grpcio-tools
bash scripts/gen_proto.sh
```

## Tests & CI

- `pytest tests/unit` — pure-logic tests (RRF/MMR math, tool validation,
  guardrails, chunking, pricing, prompt loading) — no live services
  needed, runs on every push via `.github/workflows/ci.yml`.
- `pytest tests/benchmark/test_ragas_benchmark.py` — RAGAS regression
  check against `tests/benchmark/ragas_baseline.json`; needs a live
  Weaviate + LLM API key, so it's a separate `workflow_dispatch` CI job,
  not part of the default pipeline.
- `python tests/benchmark/update_baseline.py --confirm` — the *only*
  way to move the baseline; deliberately manual.
- Lint: `ruff check .` (config in `ruff.toml`).

## Notes

- Guardrails (`shared/core/guardrails.py`) are heuristic (regex/keyword),
  not a trained moderation model — solid for obvious cases, not a
  substitute for a real moderation API if you need stronger coverage.
- `tests/benchmark/ragas_baseline.json` ships with placeholder numbers —
  run `update_baseline.py --confirm` against your real setup before
  relying on the regression check.
- PDF ingestion (`unstructured[pdf]`) needs `poppler-utils` and
  `tesseract-ocr` system packages (already included in
  `rag_agent_service/Dockerfile`).
