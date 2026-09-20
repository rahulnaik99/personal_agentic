#!/usr/bin/env bash
set -e
python -m rag_agent_service.app.server &
uvicorn rag_agent_service.app.ingest_api:app --host 0.0.0.0 --port 8001 &
wait -n
