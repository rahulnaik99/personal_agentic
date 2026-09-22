# Agentic System — LangGraph + A2A-style gRPC Multi-Service Build

Four services, orchestrated via LangGraph, communicating over gRPC using
an A2A-inspired task protocol, with a Streamlit UI on top.

## Architecture

```
                              ┌───────────────────────────┐
   Streamlit UI  ── HTTP/SSE ▶│  orchestrator-service      │
   (:8501)                   │  FastAPI gateway + LangGraph│
                              │  /chat /chat/stream /status│
                              │  /agents /health           │
                              └────────────┬───────────────┘
                                           │ gRPC (AgentService:
                                           │  GetAgentCard / SendTask / StreamTask)
                          ┌────────────────┼────────────────┐
                          ▼                                 ▼
              ┌───────────────────────┐         ┌───────────────────────┐
              │  rag-agent-service    │         │  tool-agent-service    │
              │  gRPC :50061          │         │  gRPC :50062           │
              │  + HTTP ingest :8001  │         │  (web_search, fetch_url)│
              │  /ingest/file(/stream)│         └───────────────────────┘
              │  /stats /health       │
              └───────────┬───────────┘
                          ▼
                   Weaviate (:8080, :50051)
```

**Request flow**: the orchestrator classifies each query (`rag` / `tool`
/ `both` / `direct`) via an LLM routing prompt, dispatches to the
relevant agent(s) over gRPC, loops RAG → tool agent when retrieved
context is insufficient, applies input/output guardrails, and streams
back progress + a structured per-turn log (tokens, cost, model used,
RAGAS scores, guardrail status) over SSE.

**RAG pipeline**: hybrid retrieval (dense + sparse → manual RRF fusion →
MMR → cross-encoder rerank → parent-chunk expansion), element-aware
parent-child chunking (tables/images kept whole, images captioned via a
vision LLM call), category-tagged + hash-versioned ingestion (re-ingesting
unchanged content is a no-op; changed content deactivates the old chunks
and inserts new ones as active), RAGAS evaluation per response.

**Model selection**: every LLM call (routing, RAG generation, tool
calling) can be overridden per-request to a different provider/model
(local / MLX / OpenAI / Anthropic) via `ChatRequest.model_override`, threaded
through the gRPC context map to whichever agent(s) handle the request.

See `ARCHITECTURE.md` for the full per-service breakdown and `TECH_STACK.md`
for every component/library used and why.

## Services & ports

| Service | Ports | What it does |
|---|---|---|
| `weaviate` | 8080 (HTTP), 50051 (gRPC) | Vector DB — only `rag_agent_service` talks to it directly |
| `orchestrator_service` | 8000 (HTTP/SSE) | Gateway, routing, LangGraph state machine |
| `rag_agent_service` | 50061 (gRPC), 8001 (HTTP ingest) | Retrieval, generation, ingestion, RAGAS |
| `tool_agent_service` | 50062 (gRPC) | Web search + URL fetch, tool-call validation |
| `streamlit_app` | 8501 | Chat + Ingestion UI |

---

## Option A — Docker (simplest, but slow to rebuild)

```bash
cp .env.example .env   # then edit .env: set a real ANTHROPIC_API_KEY or OPENAI_API_KEY
./start.sh
```

Rebuilds all 4 custom images (torch/sentence-transformers/unstructured
make this slow, especially on the first run or after a `requirements.txt`
change) and starts everything, including Weaviate. Visit `http://localhost:8501`.

To stop: `docker compose down` (add `-v` to also wipe Weaviate's data volume).

---

## Option B — Native / venvs (fast iteration, recommended for development)

Each service gets its own isolated venv. Code edits take effect
immediately on restart — no image rebuild, ever. Only Weaviate stays in
Docker (it's a compiled binary, no pip equivalent).

### Quick path — one script does everything

```bash
cp .env.example .env   # then edit .env: set a real API key
chmod +x local_start.sh local_stop.sh
./local_start.sh              # first run: creates venvs + installs deps (slow, once)
./local_start.sh              # every run after: venv exists -> starts immediately
./local_start.sh --reinstall  # force-recreate every venv (after changing a requirements.txt)
```

Stop everything: `./local_stop.sh` (add `--all` to also stop Weaviate).
Logs land in `./logs/<service>.log`.

### Manual path — one venv per service, explicit commands

**Important**: activating a venv with `source .venv/bin/activate` can
silently fail to actually change what `python`/`pip`/`uvicorn` resolve to,
if your shell config (`.zshrc` — pyenv/conda/oh-my-zsh hooks are common
culprits) re-touches `$PATH` afterward. The commands below sidestep that
entirely by calling each venv's binaries with their **full path** — this
works regardless of shell config, every time.

Run everything below from the **project root** (the folder containing
`orchestrator_service/`, `rag_agent_service/`, etc. as direct
subfolders) — never `cd` into a service folder first.

**1. Weaviate** (still via Docker — no native alternative):
```bash
docker compose up -d weaviate
curl http://localhost:8080/v1/.well-known/ready   # confirm it's up
```

**2. Tool agent**:
```bash
python3 -m venv tool_agent_service/.venv
tool_agent_service/.venv/bin/python -m pip install -r requirements-common.txt -r tool_agent_service/requirements.txt
PYTHONPATH=. tool_agent_service/.venv/bin/python -m tool_agent_service.app.server
```

**3. RAG agent** — two separate processes, same venv (run in two terminals):
```bash
python3 -m venv rag_agent_service/.venv
rag_agent_service/.venv/bin/python -m pip install -r requirements-common.txt -r rag_agent_service/requirements.txt
```
```bash
# terminal A — gRPC server (AgentService)
PYTHONPATH=. rag_agent_service/.venv/bin/python -m rag_agent_service.app.server
```
```bash
# terminal B — HTTP ingest API (/ingest/file, /ingest/file/stream, /stats, /health)
PYTHONPATH=. rag_agent_service/.venv/bin/python -m uvicorn rag_agent_service.app.ingest_api:app --port 8001
```

**4. Orchestrator** (needs both agents above already running):
```bash
python3 -m venv orchestrator_service/.venv
orchestrator_service/.venv/bin/python -m pip install -r requirements-common.txt -r orchestrator_service/requirements.txt
PYTHONPATH=. orchestrator_service/.venv/bin/python -m uvicorn orchestrator_service.app.main:app --host 0.0.0.0 --port 8000 --reload
```

**5. Streamlit UI**:
```bash
python3 -m venv streamlit_app/.venv
streamlit_app/.venv/bin/python -m pip install -r streamlit_app/requirements.txt
ORCHESTRATOR_URL=http://localhost:8000 RAG_INGEST_URL=http://localhost:8001 streamlit_app/.venv/bin/python -m streamlit run streamlit_app/app.py
```

**Verify everything is wired up**:
```bash
curl http://localhost:8000/status   # {"weaviate":"ok","orchestrator":"ok","rag_agent":"ok","tool_agent":"ok"}
curl http://localhost:8000/agents   # both agent cards, no errors
```

**One `requirements.txt` changed?** Only reinstall that one service — no
need to touch the others:
```bash
rag_agent_service/.venv/bin/python -m pip install -r requirements-common.txt -r rag_agent_service/requirements.txt
```

---

## Local model backends

The chat model can be selected per request from the Streamlit sidebar. The selection is kept in the current Streamlit session and automatically applies to subsequent turns. Supported providers are Anthropic, OpenAI, Ollama/OpenAI-compatible local servers, and MLX.

### MLX on Apple Silicon

`mlx-lm` provides an OpenAI-compatible HTTP server on port `8080`. The project uses the `mlx` provider to connect to it.

Install and start an MLX model, for example:

```bash
pip install -U mlx-lm
mlx_lm.server --model mlx-community/DeepSeek-R1-Distill-Qwen-14B-MLX --port 8080
```

The project defaults to:

```text
MLX_LLM_BASE_URL=http://127.0.0.1:8080/v1
MLX_MODEL_FALLBACK=mlx-community/DeepSeek-R1-Distill-Qwen-14B-MLX
```

In the UI select **MLX — Apple Silicon** and optionally enter the exact model id served by `mlx-lm`. `mlx-lm` exposes `/v1/chat/completions`, which is the interface used by the project's LangChain/OpenAI-compatible client.

MLX tool calling depends on the model/server combination. Use a tool-capable MLX model when selecting MLX for web-tool routes; normal chat/RAG/direct generation does not require tool calling.

## Ingesting documents

```bash
curl -X POST http://localhost:8001/ingest/file \
  -F "file=@/path/to/resume.pdf" \
  -F "category=profession_doc"
```

`category` lets retrieval later filter to the right document set — ask
"what's my work experience?" and it'll search `profession_doc` chunks
specifically rather than everything ingested. Re-uploading unchanged
content is a no-op; re-uploading changed content under the same filename
deactivates the old chunks and activates the new ones (nothing is ever
silently duplicated).

For live staged progress (loading → dedup check → chunking → embedding →
completed) instead of a single blocking response, use
`/ingest/file/stream` (SSE) — this is what the Streamlit Ingestion tab
uses under the hood.

## Tests & CI

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest tests/unit -v      # pure-logic tests, no live services needed
ruff check .                            # lint
```

`tests/benchmark/test_ragas_benchmark.py` needs a live Weaviate + LLM API
key — it's a separate `workflow_dispatch` CI job (`.github/workflows/ci.yml`),
not part of the default push/PR pipeline. Update the accepted baseline
deliberately with `python tests/benchmark/update_baseline.py --confirm`.

## Notes

- Guardrails (`shared/core/guardrails.py`) are heuristic regex/keyword
  checks, not a trained moderation model.
- `tests/benchmark/ragas_baseline.json` ships with placeholder numbers —
  regenerate it against your real setup before trusting the regression check.
- PDF ingestion (`unstructured[pdf]`) needs `poppler-utils` and
  `tesseract-ocr` system packages — already in `rag_agent_service/Dockerfile`;
  install them yourself (`brew install poppler tesseract` on macOS) if
  running Option B without Docker.
- `.docx` isn't supported yet — only `.pdf` and `.txt`.
