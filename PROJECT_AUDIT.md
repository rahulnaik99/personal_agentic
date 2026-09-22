# Project Audit & Upgrade Report

## 1. Architecture reviewed

The project is a four-service LangGraph system:

- Streamlit UI
- Orchestrator / FastAPI / LangGraph
- RAG agent over Weaviate
- Tool agent over web search + URL fetch

The orchestrator dispatches to the agents over the generated gRPC `AgentService` contract.

## 2. Root cause of the apparent "orchestrator not invoking agents" issue

The LangGraph routing graph itself was wired correctly:

`guardrail -> classify -> rag/tool/direct -> finalize`

and `both` correctly executes RAG followed by the tool agent.

The important bug was in the SSE event aggregation in `orchestrator_service/app/main.py`.
The previous implementation retained only the output of the `finalize` node from
`astream_events()`. LangGraph node outputs such as `route`, `rag_result`, and
`tool_result` were therefore discarded before the response was sent to Streamlit.

That caused the UI turn log to appear as if the agents had not run even when the gRPC
calls had executed.

The SSE implementation now merges every relevant node's `on_chain_end` output into
`final_state`, so the route, agent results, RAG contexts, tool calls, and final answer
survive into the response payload.

A second improvement is explicit orchestration-error handling so a failed graph does
not produce an empty SSE response.

## 3. Prompt upgrades

All five project prompts were reviewed and upgraded:

- `orchestrator_service/app/prompts/routing_prompt.md`
- `orchestrator_service/app/prompts/direct_answer_prompt.md`
- `rag_agent_service/app/prompts/category_classifier_prompt.md`
- `rag_agent_service/app/prompts/rag_system_prompt.md`
- `tool_agent_service/app/prompts/tool_system_prompt.md`

The routing prompt now uses an explicit decision matrix and rules for internal,
external/current, mixed, URL, user-provided-text, and ambiguous requests.

The RAG prompt now explicitly treats retrieved content as evidence rather than
instructions and enforces an `INSUFFICIENT_CONTEXT:` sentinel.

The tool prompt now has a clearer search/fetch policy, source-quality guidance,
prompt-injection handling for fetched pages, and a bounded tool loop.

## 4. Session architecture

The old UI had one global `messages` list and one `session_id`. It did not provide
real independent chats.

The upgraded UI maintains:

- Multiple independent chat sessions
- A persistent active-session id
- A session sidebar
- New-chat creation
- Per-session messages
- Per-session token/cost totals
- Automatic session titles based on the first user message
- Conversation history sent to the orchestrator and downstream agents

The last point is important: a session is now more than a UI label. The selected
conversation history is passed through the orchestrator and gRPC context to RAG/tool
agents, so follow-up questions can use earlier turns.

## 5. Model persistence

Provider and model are stored directly in Streamlit session state and are not reset
when the assistant reruns the application after a response.

The selected model therefore remains active for subsequent turns.

The selected model is a user-level preference for the current Streamlit browser
session rather than a property of one chat.

## 6. MLX support

A dedicated `mlx` provider was added to the shared LLM factory.

Configuration:

```text
MLX_LLM_BASE_URL=http://127.0.0.1:8080/v1
MLX_LLM_API_KEY=not-needed
MLX_MODEL_FALLBACK=mlx-community/DeepSeek-R1-Distill-Qwen-14B-MLX
```

The UI exposes **MLX — Apple Silicon** as a provider.

MLX-LM's server exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so the
existing LangChain `ChatOpenAI` adapter can be reused without introducing a separate
chat stack.

## 7. Important remaining gaps

### Tool-call compatibility

Not every MLX model supports structured tool calling. MLX should therefore be used
with a tool-capable model when the selected route requires the Tool Agent.

### Authentication / deployment security

The current gRPC services use insecure local gRPC channels. This is appropriate for
local development but should be replaced with TLS/authentication for a distributed
production deployment.

### Tool URL security

`TOOL_ALLOWED_DOMAINS` is optional and currently allows unrestricted domains when it
is empty. For production, enable a domain policy and add SSRF protection for private,
loopback, link-local, and reserved addresses.

### Persistent server-side sessions

Streamlit sessions are browser-session state. They are not a durable database-backed
conversation store. If chats must survive browser restart, multiple devices, or
server restarts, add a session store such as PostgreSQL/Redis.

### RAGAS latency

RAGAS evaluation runs synchronously after RAG generation. This can materially increase
latency. Production deployments should consider asynchronous evaluation or a sampling
policy.

### Model capability metadata

The UI currently lets the user select a provider/model by name. A future improvement
would be a model registry that records capabilities such as tool calling, vision,
context length, and structured-output support and prevents incompatible combinations.

### File types

The ingestion UI currently exposes PDF and TXT. DOCX and other enterprise document
formats remain unsupported by the current ingestion path.

## 8. Validation performed

Python bytecode compilation was run across the modified Python packages successfully:

```text
python3 -m compileall -q orchestrator_service rag_agent_service tool_agent_service shared streamlit_app
```

The repository's unit-test command was also attempted. Test collection could not start
in the current execution environment because the project dependencies are not installed
there (`langgraph`, `weaviate`, `langchain_text_splitters`, `ddgs`, `sse_starlette`, etc.).
No test result was therefore fabricated.

## 9. MLX startup example

Install MLX-LM and start the server:

```bash
pip install -U mlx-lm
mlx_lm.server --model mlx-community/DeepSeek-R1-Distill-Qwen-14B-MLX --port 8080
```

Then start the project and select **MLX — Apple Silicon** in the Streamlit sidebar.
