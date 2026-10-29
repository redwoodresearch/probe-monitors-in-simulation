#!/usr/bin/env bash
# Run a command on the pilot devbox over the SSM-tunnelled ssh `rw devbox ssh` uses (no agent forwarding), or copy
# files with `devbox.sh rsync <args>` where the remote side is written `box:<path>`.
# DEVBOX is `<user>@<instance-id>` from `rw devbox ssh --name phase2 --dry-run`.
#   DEVBOX=aswin@i-0123 experiments/2026-10-07_phase2_attack_check/devbox.sh 'docker ps'
#   DEVBOX=aswin@i-0123 experiments/2026-10-07_phase2_attack_check/devbox.sh rsync -az src/ box:dest/
set -euo pipefail
target="${DEVBOX:?set DEVBOX=<user>@<instance-id>}"
proxy="aws ssm start-session --region us-west-2 --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p"
ssh_opts=(-o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" -o ForwardAgent=no -o StrictHostKeyChecking=accept-new
          -o ServerAliveInterval=30 -o "ProxyCommand=$proxy")
# DEVBOX_SEND_ENV=NAME forwards one variable over ssh's encrypted env channel (never argv), as `rw devbox ssh --env`.
[[ -n "${DEVBOX_SEND_ENV:-}" ]] && ssh_opts+=(-o "SendEnv=$DEVBOX_SEND_ENV")
export AWS_PROFILE="${AWS_PROFILE:-redwood-sandbox}"
case "${1:-}" in
  __ssh) shift; exec ssh "${ssh_opts[@]}" "$@" ;;  # rsync's transport: this script, called back
  rsync) shift; exec rsync -e "$0 __ssh" "${@/#box:/$target:}" ;;
  *) exec ssh "${ssh_opts[@]}" "$target" "$@" ;;
esac
