#!/usr/bin/env bash
# Copy the pair viewer's code and config to a devbox (into ~/pair-viewer), over the SSM-tunnelled ssh `rw devbox ssh`
# uses. Get <user>@<instance-id> from `rw devbox ssh --project probe-shift-sims --name sims-astra --dry-run`.
#   experiments/2026-10-08_pilot_pair_viewer/sync_devbox.sh aswin@i-0123456789abcdef0
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
target="${1:?usage: sync_devbox.sh <user>@<instance-id>}"
proxy="aws ssm start-session --region us-west-2 --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p"
AWS_PROFILE="${AWS_PROFILE:-redwood-sandbox}" rsync -az --relative --exclude __pycache__ \
  -e "ssh -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519 -o ForwardAgent=no -o StrictHostKeyChecking=accept-new -o ProxyCommand='$proxy'" \
  src/probe_shift/__init__.py src/probe_shift/viewer/__init__.py src/probe_shift/viewer/pairs.py src/probe_shift/viewer/pairs_app.py \
  experiments/2026-10-08_pilot_pair_viewer experiments/2026-10-06_transcript_viewer/requirements-devbox.txt \
  "$target:pair-viewer/"
