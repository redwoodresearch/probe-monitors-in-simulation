#!/usr/bin/env bash
# Serve the pair viewer on the devbox that holds ~/phase2/runs (sims-astra), privately on the tailnet at
# https://<devbox>.<tailnet>.ts.net:8443/. Run after sync_devbox.sh, in its own tmux socket (never the phase2 one):
#   tmux -L viewer new -d -s viewer-pairs pair-viewer/experiments/2026-10-08_pilot_pair_viewer/serve_devbox.sh
# Read-only on ~/phase2, loopback only, niced so the pilot's environments keep the CPU; starts no containers.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
exec nice -n 10 redwood-app run 8502 8443 -- uv run --no-project --python 3.12 \
  --with-requirements experiments/2026-10-06_transcript_viewer/requirements-devbox.txt --with zstandard==0.25.0 \
  streamlit run src/probe_shift/viewer/pairs_app.py --server.address 127.0.0.1 --server.port 8502 --server.headless true \
  --browser.gatherUsageStats false --client.toolbarMode minimal -- "$@"
