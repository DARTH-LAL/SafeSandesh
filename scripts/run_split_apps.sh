#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for port in 8501 8502 8503; do
  pids="$(lsof -ti "tcp:${port}" || true)"
  if [[ -n "$pids" ]]; then
    kill $pids || true
    sleep 0.5
  fi
done

"$ROOT/.venv/bin/streamlit" run apps/consumer_app.py \
  --server.port 8502 \
  --server.fileWatcherType none &
consumer_pid=$!

"$ROOT/.venv/bin/streamlit" run apps/technical_app.py \
  --server.port 8503 \
  --server.fileWatcherType none &
technical_pid=$!

sleep 3
open "http://localhost:8502"
open "http://localhost:8503"

wait "$consumer_pid" "$technical_pid"
