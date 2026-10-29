#!/usr/bin/env bash
# Start the transcript viewer on loopback: http://127.0.0.1:${PORT:-8501}. Build the cache first (run_build_cache.py).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
exec uv run --group viewer streamlit run src/probe_shift/viewer/app.py \
  --server.address 127.0.0.1 --server.port "${PORT:-8501}" --server.headless true \
  --browser.gatherUsageStats false --client.toolbarMode minimal -- "$@"  # minimal: no Deploy button or dev menu
