#!/usr/bin/env bash
# One-shot bootstrap: sets up .env if missing, builds all images, starts
# every service (Weaviate, RAG agent, tool agent, orchestrator, Streamlit
# UI), and waits until the orchestrator's health endpoint responds before
# printing where to go.
#
# Usage:
#   ./start.sh            # build + start everything, follow logs
#   ./start.sh --detach    # build + start in the background, don't follow logs
set -euo pipefail
cd "$(dirname "$0")"

DETACH=false
if [[ "${1:-}" == "--detach" || "${1:-}" == "-d" ]]; then
  DETACH=true
fi

# ---- Preflight checks ----
if ! command -v docker &> /dev/null; then
  echo "ERROR: docker is not installed or not on PATH." >&2
  exit 1
fi

if ! docker compose version &> /dev/null; then
  echo "ERROR: 'docker compose' (v2 plugin) is required. Update Docker Desktop / the docker-compose-plugin package." >&2
  exit 1
fi

# ---- .env setup ----
if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit it to add your real API keys, then re-run this script."
    exit 1
  else
    echo "ERROR: no .env or .env.example found in $(pwd)." >&2
    exit 1
  fi
fi

if grep -qE "REPLACE_ME" .env; then
  echo "WARNING: .env still contains placeholder API key values (REPLACE_ME)."
  echo "         LLM calls will fail until you set real keys for your chosen LLM_PROVIDER."
  echo
fi

# ---- Build + start ----
echo "Building images (this can take a while the first time — PDF/vision deps are heavy)..."
docker compose build

echo "Starting services..."
if $DETACH; then
  docker compose up -d
else
  docker compose up -d
fi

# ---- Wait for orchestrator to become healthy ----
echo -n "Waiting for orchestrator-service to become healthy"
ORCH_URL="http://localhost:8000/health"
for _ in $(seq 1 60); do
  if curl -sf "$ORCH_URL" > /dev/null 2>&1; then
    echo " — up!"
    break
  fi
  echo -n "."
  sleep 2
done

if ! curl -sf "$ORCH_URL" > /dev/null 2>&1; then
  echo
  echo "WARNING: orchestrator-service did not become healthy within the timeout."
  echo "         Check logs with: docker compose logs -f orchestrator-service"
fi

cat <<'EOF'

Everything is up:
  - Streamlit UI:        http://localhost:8501
  - Orchestrator API:    http://localhost:8000  (docs: /docs, health: /health, agents: /agents)
  - RAG agent ingest API: http://localhost:8001  (POST /ingest/file, /ingest/text)
  - Weaviate:            http://localhost:8080

Ingest a document:
  curl -X POST http://localhost:8001/ingest/file -F "file=@/path/to/doc.pdf"

Stop everything:
  docker compose down

EOF

if ! $DETACH; then
  echo "Following logs (Ctrl+C to stop watching — services keep running in the background)..."
  docker compose logs -f
fi
