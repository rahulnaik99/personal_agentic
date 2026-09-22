#!/usr/bin/env bash
# local_stop.sh — stops everything local_start.sh started.
# Weaviate keeps running by default (it holds your ingested data and is
# slow-ish to warm back up) — pass --all to stop it too.
set -uo pipefail
cd "$(dirname "$0")"

PID_DIR="./.pids"

if [[ -d "$PID_DIR" ]]; then
  for pid_file in "$PID_DIR"/*.pid; do
    [[ -e "$pid_file" ]] || continue
    name="$(basename "$pid_file" .pid)"
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2> /dev/null; then
      echo "Stopping $name (pid $pid)..."
      kill "$pid" 2> /dev/null || true
    else
      echo "$name (pid $pid) was not running."
    fi
    rm -f "$pid_file"
  done
else
  echo "No .pids directory found — nothing to stop."
fi

if [[ "${1:-}" == "--all" ]]; then
  echo "Stopping Weaviate too..."
  docker compose stop weaviate
fi

echo "Done."
