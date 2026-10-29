#!/usr/bin/env bash
# Copy the viewer's code, the splits and the built cache (not the raw .eval downloads) to a devbox, over the same
# SSM-tunnelled ssh `rw devbox ssh` uses. Get <user>@<instance-id> from `rw devbox ssh --name <box> --dry-run`.
#   experiments/2026-10-06_transcript_viewer/sync_devbox.sh aswin@i-0123456789abcdef0
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
target="${1:?usage: sync_devbox.sh <user>@<instance-id>}"
mkdir -p .run_cache/viewer/scores  # probe score files; the app picks up new ones without a restart
proxy="aws ssm start-session --region us-west-2 --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p"
AWS_PROFILE="${AWS_PROFILE:-redwood-sandbox}" rsync -az --delete --relative --exclude __pycache__ \
  -e "ssh -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519 -o ForwardAgent=no -o StrictHostKeyChecking=accept-new -o ProxyCommand='$proxy'" \
  src/probe_shift splits experiments/2026-10-06_transcript_viewer/config experiments/2026-10-06_transcript_viewer/serve_devbox.sh \
  experiments/2026-10-06_transcript_viewer/requirements-devbox.txt \
  .run_cache/viewer/index.jsonl .run_cache/viewer/version.json .run_cache/viewer/samples .run_cache/viewer/scores \
  "$target:probe-shift-viewer/"
