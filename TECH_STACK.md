# Tech Stack & Components

Every technology, framework, protocol, and technique used across the
system, what it does, and exactly where/why it's used in this codebase.

## Languages & Runtime

| Component | Description | Where and why it's used |
|---|---|---|
| Python 3.11 | Primary language for all four services | All of `orchestrator_service/`, `rag_agent_service/`, `tool_agent_service/`, `streamlit_app/`, `shared/` |
| Bash | Shell scripting | `start.sh` (one-command bootstrap), `scripts/gen_proto.sh` (proto codegen) |

## Web / API Frameworks

| Component | Description | Where and why it's used |
|---|---|---|
| FastAPI | Async Python web framework | Orchestrator's external gateway (`orchestrator_service/app/main.py`: `/chat`, `/chat/stream`, `/agents`, `/health`) and the RAG agent's separate ingestion API (`rag_agent_service/app/ingest_api.py`) |
| Uvicorn | ASGI server that runs FastAPI apps | Runs both FastAPI apps above, in each service's Dockerfile `CMD`/entrypoint |
| SSE-Starlette | Server-Sent Events support for Starlette/FastAPI | Powers `/chat/stream`'s live `progress`/`done` event stream from orchestrator to the Streamlit UI |
| Streamlit | Python UI framework for data/chat apps | The entire `streamlit_app/app.py` — chat interface, live "Thinking" status, trace/cost visualization |
| httpx | Async/sync HTTP client | Tool agent's `fetch_url` tool; Streamlit's SSE connection to the orchestrator |
| httpx-sse | SSE client helper on top of httpx | Streamlit's `connect_sse()` call to consume the orchestrator's `/chat/stream` |

## Agent Orchestration

| Component | Description | Where and why it's used |
|---|---|---|
| LangGraph | Graph-based agent orchestration framework (StateGraph, conditional edges) | The orchestrator's entire routing/looping logic (`orchestrator_service/app/graph.py`) — classify → route to RAG/tool/direct → conditional loop back on insufficient RAG context → finalize |
| LangChain Core | Message types (`HumanMessage`, `SystemMessage`, `ToolMessage`), `@tool` decorator, `bind_tools()` | Every LLM call across all three services; the tool agent's function-calling setup |
| LangChain (OpenAI / Anthropic integrations) | Provider-specific chat model wrappers (`ChatOpenAI`, `ChatAnthropic`) | `shared/services/chat_llm.py`'s multi-provider factory — swaps model backend based on `LLM_PROVIDER` in `.env` |

## Agent-to-Agent Communication

| Component | Description | Where and why it's used |
|---|---|---|
| gRPC (grpcio, grpcio-tools) | High-performance RPC framework over HTTP/2 + Protocol Buffers | Transport between the orchestrator and both agent services — `AgentService.SendTask`/`StreamTask`/`GetAgentCard` |
| Protocol Buffers (protobuf) | Interface definition language + binary wire format | `proto/agent.proto` defines the shared contract; `shared/generated/agent_pb2*.py` are the generated Python stubs |
| A2A Protocol (custom, A2A-inspired) | Our hand-rolled task-lifecycle contract modeled on Google's Agent2Agent pattern (Agent Cards, `SendTask`, task states) | Every agent publishes an `agent_card.json`; the orchestrator is the only gRPC *client*, agents are gRPC *servers* — never call each other directly. **Currently in use; planned to be replaced below.** |
| `a2a-sdk` (official A2A Python SDK) | Official implementation of the A2A Protocol Specification 1.0 (PyPI, Apache-2.0), with a 0.3 compatibility mode | **Not yet integrated — planned upgrade.** Would replace the custom contract above: `AgentCard`/`AgentSkill`/`AgentInterface` types, `/.well-known/agent-card.json` discovery, `DefaultRequestHandler`/`AgentExecutor`/`EventQueue` for agent servers, `A2ACardResolver`/`create_client()` for the orchestrator's client. Full plan in `ARCHITECTURE.md` §5 — not in `requirements-common.txt` yet, pending go-ahead to implement |

## Retrieval-Augmented Generation (RAG)

| Component | Description | Where and why it's used |
|---|---|---|
| Weaviate | Open-source vector database | Stores `ParentChunk` and `ChildChunk` collections; dense (`near_vector`) and sparse (`bm25`) search both run against it — `shared/vectorstore/weaviate_client.py`, `rag_agent_service/app/retriever.py` |
| Sentence-Transformers (bi-encoder) | Embedding model library | `shared/services/encoder_llm.py`'s `get_encoder()` — embeds documents/queries for dense retrieval and MMR similarity scoring |
| Sentence-Transformers (cross-encoder) | Reranking model library | `get_reranker()` in the same file — final precision filter on the MMR shortlist before parent expansion |
| RRF (Reciprocal Rank Fusion) | Rank-fusion algorithm (not a library — a technique we implement directly) | `rag_agent_service/app/retriever.py`'s `_reciprocal_rank_fusion()` — merges dense + sparse ranked lists using only rank position, robust to differing score scales |
| MMR (Maximal Marginal Relevance) | Diversity-aware re-ranking technique (implemented directly, not a library) | `_mmr()` in the same file — reduces redundancy among fused candidates before the (expensive) cross-encoder step |
| RAGAS | RAG evaluation framework (faithfulness, answer relevancy, context precision/recall) | `rag_agent_service/app/evaluation.py` — scores every RAG response; also powers the CI regression benchmark in `tests/benchmark/` |
| Hugging Face `datasets` | Dataset library | Required by RAGAS internally to structure evaluation inputs |
| `unstructured[pdf]` | Document partitioning library (text/table/image extraction from PDFs) | `rag_agent_service/app/loaders/pdf_loader.py` — splits a PDF into typed elements for element-aware chunking |
| Content-hash-based dedup/versioning (custom) | SHA-256 of ingested content compared against the active version for the same `source`; unchanged content is skipped, changed content deactivates the old chunks and inserts new ones as active (never hard-deleted) | `rag_agent_service/app/ingestion.py`, `shared/vectorstore/weaviate_client.py` (`get_existing_chunk_hash`, `deactivate_source`) |
| Category-aware retrieval (custom) | Every chunk tagged with a caller-supplied category at ingest time; an LLM classifier picks the best-matching category per query (or leaves retrieval unfiltered if unsure) so answers only draw from the relevant document set | `rag_agent_service/app/agent.py`'s `classify_category()`, `category_classifier_prompt.md`, `retriever.py`'s category filter |
| Poppler, Tesseract OCR | System-level PDF rendering / OCR engines | System dependencies required by `unstructured`'s `hi_res` PDF strategy (installed via `apt-get` in `rag_agent_service/Dockerfile`) |
| Pillow | Image processing library | Transitive dependency for image handling during PDF partitioning |

## Web Search & Content Fetching

| Component | Description | Where and why it's used |
|---|---|---|
| DuckDuckGo Search (`ddgs`) | Free, no-API-key web search | `tool_agent_service/app/tools.py`'s `web_search()` |
| trafilatura | Readable-content extraction from raw HTML | Primary extraction method in `tools.py`'s `fetch_url()` |
| BeautifulSoup4 | HTML parsing library | Fallback extraction path in `fetch_url()` if trafilatura fails |

## Validation, Config & Cross-Cutting Concerns

| Component | Description | Where and why it's used |
|---|---|---|
| Pydantic / pydantic-settings | Data validation + settings management | `shared/core/config.py`'s `Settings` (reads `.env`); every request/response schema (`ChatRequest`, `RouteDecision`, tool argument schemas in `validation.py`) |
| Custom guardrails (regex-based) | Heuristic PII detection/redaction, prompt-injection phrase matching, length limits | `shared/core/guardrails.py`, wrapped per-service; applied to every agent's input and output |
| `.md` prompt files + YAML frontmatter | Prompts stored as version-controllable markdown (Role/Instructions/Output Format/Few-Shot Examples), loaded via a small parser | `shared/core/prompts.py`'s `load_prompt()`; every `*/app/prompts/*.md` file |
| Structured JSON logging | Every node/tool call logs a `start`/`end`/`error` JSON record tagged with a `trace_id` | `shared/core/logging.py`'s `log_step()` — used by every agent and orchestrator node |
| Per-request trace collector | Aggregates token usage, latency, and estimated cost per LLM call across all services for one request | `shared/core/logging.py`'s `TraceCollector`; fed by `invoke_and_track()` (local calls) and `grpc_clients.py` (remote agent calls); rendered in the Streamlit "🧠 Thinking" panel |
| Cost/pricing table | Static `$/1M tokens` lookup per model | `shared/core/pricing.py`'s `estimate_cost_usd()` |

## Infrastructure & DevOps

| Component | Description | Where and why it's used |
|---|---|---|
| Docker | Containerization | One image per service (`*/Dockerfile`) |
| Docker Compose | Multi-container orchestration | `docker-compose.yml` wires Weaviate + both agents + orchestrator + Streamlit together; `start.sh` wraps it into one command |
| GitHub Actions | CI/CD | `.github/workflows/ci.yml` — lint + unit tests on every push; RAGAS regression benchmark as a separate gated `workflow_dispatch` job |
| Ruff | Python linter | `ruff.toml` config; run in CI and locally (`ruff check .`) |
| Pytest / pytest-mock | Testing framework | `tests/unit/` (43 tests, fully mocked) and `tests/benchmark/` (live-dependency RAGAS regression check) |

## LLM Providers (pluggable, selected via `.env`)

| Component | Description | Where and why it's used |
|---|---|---|
| Anthropic Claude | LLM provider (default) | `LLM_PROVIDER=anthropic` in `.env`; used for routing, generation, tool-calling, and vision captioning (image chunk descriptions during PDF ingestion) |
| OpenAI GPT | LLM provider (alternate) | `LLM_PROVIDER=openai`; same call sites, swapped via config with no code changes |
| Local model server (e.g. Ollama) | Self-hosted LLM via an OpenAI-compatible endpoint | `LLM_PROVIDER=local`; same call sites, points at `LOCAL_LLM_BASE_URL` |
