#!/usr/bin/env bash
# local_start.sh — runs everything WITHOUT rebuilding Docker images every
# time. Only Weaviate (a compiled Go binary — no simple pip alternative)
# runs via Docker; every Python service gets its own venv and runs
# natively in the background. Code edits take effect on the next
# `local_stop.sh && local_start.sh` with NO rebuild step at all.
#
# First run: creates each venv + installs deps (slow, once).
# Every run after: if a service's .venv folder already exists, it's used
# as-is — no reinstall, no check, just starts. If you change a
# requirements.txt, either delete that service's .venv folder or run
# with --reinstall to recreate every venv from scratch.
#
# Usage:
#   ./local_start.sh              # normal start
#   ./local_start.sh --reinstall  # wipe and recreate all venvs first
set -euo pipefail
cd "$(dirname "$0")"

REINSTALL="false"
if [[ "${1:-}" == "--reinstall" ]]; then
  REINSTALL="true"
fi

LOG_DIR="./logs"
PID_DIR="./.pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found. Run: cp .env.example .env   (then fill in real API keys)" >&2
  exit 1
fi
set -a
source .env
set +a

# ---- Weaviate stays in Docker — it's a compiled binary, not pip-installable ----
if ! command -v docker &> /dev/null; then
  echo "ERROR: docker is required for Weaviate. Install Docker Desktop first." >&2
  exit 1
fi
echo "Starting Weaviate..."
docker compose up -d weaviate

echo -n "Waiting for Weaviate"
for _ in $(seq 1 30); do
  if curl -sf http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
    echo " — up!"
    break
  fi
  echo -n "."
  sleep 1
done

# ---- Helper: create venv + install deps ONLY if the venv doesn't exist yet.
# If you change a requirements.txt later, run: ./local_start.sh --reinstall
# (or just delete that service's .venv folder and re-run normally).
ensure_venv() {
  local service_dir="$1"
  local venv_dir="$service_dir/.venv"
  local reqs_file="$service_dir/requirements.txt"

  if [[ "$REINSTALL" == "true" && -d "$venv_dir" ]]; then
    echo "[$service_dir] --reinstall passed — removing existing venv..."
    rm -rf "$venv_dir"
  fi

  if [[ -d "$venv_dir" ]]; then
    echo "[$service_dir] venv exists — skipping install."
    return
  fi

  echo "[$service_dir] No venv found — creating and installing dependencies..."
  python3 -m venv "$venv_dir"
  "$venv_dir/bin/pip" install --quiet --upgrade pip
  "$venv_dir/bin/pip" install --quiet -r "$reqs_file"
}

# ---- Helper: start a background process, log it, save its PID ----
start_bg() {
  local name="$1"
  local venv_python="$2"
  shift 2
  echo "[$name] Starting..."
  PYTHONPATH=. "$venv_python" "$@" > "$LOG_DIR/$name.log" 2>&1 &
  echo $! > "$PID_DIR/$name.pid"
}

# ---- Tool agent ----
ensure_venv tool_agent_service
start_bg tool-agent tool_agent_service/.venv/bin/python -m tool_agent_service.app.server

# ---- RAG agent (gRPC server + separate ingest HTTP API) ----
ensure_venv rag_agent_service
start_bg rag-agent-grpc rag_agent_service/.venv/bin/python -m rag_agent_service.app.server
start_bg rag-agent-ingest rag_agent_service/.venv/bin/python -m uvicorn rag_agent_service.app.ingest_api:app --port 8001

# ---- Orchestrator ----
ensure_venv orchestrator_service
start_bg orchestrator orchestrator_service/.venv/bin/python -m uvicorn orchestrator_service.app.main:app --host 0.0.0.0 --port 8000

# ---- Streamlit UI ----
ensure_venv streamlit_app
start_bg streamlit-ui streamlit_app/.venv/bin/python -m streamlit run streamlit_app/app.py --server.headless true

echo
echo "Waiting for orchestrator to become healthy..."
for _ in $(seq 1 20); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

cat <<'EOF'

Everything is running natively (no Docker rebuilds):
  - Streamlit UI:         http://localhost:8501
  - Orchestrator API:     http://localhost:8000  (docs: /docs, health: /health, status: /status)
  - RAG agent ingest API: http://localhost:8001  (docs: /docs)
  - Weaviate:             http://localhost:8080  (via Docker)

Logs:    tail -f logs/<service>.log
Stop:    ./local_stop.sh

EOF
