#!/usr/bin/env bash
# Serve the viewer from a devbox, privately over Tailscale (https://<devbox>.<tailnet>.ts.net/). Run on the box,
# in tmux, after sync_devbox.sh. The box has neither the sibling simulated-control-evals checkout nor GitHub
# credentials, so this runs only the app's runtime dependencies, from requirements-devbox.txt (every package pinned
# to its uv.lock version); the cache is built on a laptop.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
exec redwood-app run 8501 -- uv run --no-project --python 3.12 \
  --with-requirements experiments/2026-10-06_transcript_viewer/requirements-devbox.txt \
  streamlit run src/probe_shift/viewer/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true \
  --browser.gatherUsageStats false --client.toolbarMode minimal -- "$@"
