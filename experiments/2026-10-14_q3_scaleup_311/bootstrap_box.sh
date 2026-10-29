#!/usr/bin/env bash
# Laptop side: prepare one fresh scale-up devbox (RULE.md, boxes), as the real-honest fork prepared q3-realhonest
# (its LAUNCH.md and message to this fork). Only moves files and runs setup on the box; reads no transcript or approach
# text (approach files are copied opaque). Every devbox call is bounded.
#   DEVBOX=aswin@<id> HB=<local harness dir> bootstrap_box.sh
# Steps: sims trees (September: rsync without .venv/.git, then .settings WITH .git, then `uv sync --frozen`; today: src
# and its top-level files, setup_devbox.sh hashes every simulator tree); Docker address pools (60 environments); the
# harness at ~/phase2/attack_check_311; the 196's approach files (opaque, for the preflight's renderer check); the
# pinned MinIO image streamed READ-ONLY from sims-astra with its pinned images.json (d9's approval: that image and that
# file only); the loaded image ID checked against the pinned manifest; then setup_devbox.sh in tmux -L setup.
set -euo pipefail
: "${DEVBOX:?}" "${HB:?}"
E=$HB/experiments/2026-10-07_phase2_attack_check
ASTRA=aswin@i-05b1d910fd70016e6
MINIO=quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z
PIN_JSON=phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
PIN_SHA=20d4df7e
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [$DEVBOX] $*"; }
box() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$E/devbox.sh" "$2"; }
log "dirs"; box 120 'mkdir -p ~/sims/simulated-control-evals-sept ~/sims/simulated-control-evals-today ~/phase2/approaches ~/phase2/runs ~/'"$(dirname $PIN_JSON)"
log "september tree"
DEVBOX=$DEVBOX perl -e 'alarm 1800; exec @ARGV' bash "$E/devbox.sh" rsync -az --exclude .venv --exclude __pycache__ --exclude .git \
  ~/simulated-control-evals-sept/ box:sims/simulated-control-evals-sept/
DEVBOX=$DEVBOX perl -e 'alarm 1800; exec @ARGV' bash "$E/devbox.sh" rsync -az ~/simulated-control-evals-sept/.settings/ \
  box:sims/simulated-control-evals-sept/.settings/
log "today tree"
DEVBOX=$DEVBOX perl -e 'alarm 900; exec @ARGV' bash "$E/devbox.sh" rsync -az --exclude __pycache__ \
  ~/simulated-control-evals-today/src/ box:sims/simulated-control-evals-today/src/
DEVBOX=$DEVBOX perl -e 'alarm 300; exec @ARGV' bash "$E/devbox.sh" rsync -az ~/simulated-control-evals-today/pyproject.toml \
  ~/simulated-control-evals-today/uv.lock box:sims/simulated-control-evals-today/
log "harness"
DEVBOX=$DEVBOX perl -e 'alarm 600; exec @ARGV' bash "$E/devbox.sh" rsync -a --checksum "$E/" box:phase2/attack_check_311/
log "approach files (opaque)"
DEVBOX=$DEVBOX perl -e 'alarm 600; exec @ARGV' bash "$E/devbox.sh" rsync -a ~/probe-shift-q2-tasks/data/phase2_attack_check/approaches/ \
  box:phase2/approaches/
log "uv sync and docker pools"
box 1200 'cd ~/sims/simulated-control-evals-sept && uv sync --frozen >/tmp/uv_sync.log 2>&1 && tail -1 /tmp/uv_sync.log
  echo "{\"default-address-pools\":[{\"base\":\"10.210.0.0/16\",\"size\":24}]}" | sudo tee /etc/docker/daemon.json >/dev/null
  sudo systemctl restart docker && docker info --format "{{.DefaultAddressPools}}"
  echo "fs.inotify.max_user_instances = 8192" | sudo tee /etc/sysctl.d/90-scaleup311-inotify.conf >/dev/null && sudo sysctl -q --system
  echo "inotify max_user_instances $(cat /proc/sys/fs/inotify/max_user_instances)"
  TOKEN=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
  curl -s -m 2 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type > ~/phase2/HOST_INSTANCE_TYPE
  echo "host $(cat ~/phase2/HOST_INSTANCE_TYPE)"'
log "MinIO from sims-astra (read-only docker save of $MINIO only) and its pinned images.json"
DEVBOX=$ASTRA perl -e 'alarm 900; exec @ARGV' bash "$E/devbox.sh" "docker save $MINIO" |
  DEVBOX=$DEVBOX perl -e 'alarm 900; exec @ARGV' bash "$E/devbox.sh" 'docker load'
DEVBOX=$ASTRA perl -e 'alarm 120; exec @ARGV' bash "$E/devbox.sh" "cat ~/$PIN_JSON" |
  DEVBOX=$DEVBOX perl -e 'alarm 120; exec @ARGV' bash "$E/devbox.sh" "cat > ~/$PIN_JSON"
box 120 "sha256sum ~/$PIN_JSON | cut -c1-8 | grep -qx $PIN_SHA && echo 'images.json pinned $PIN_SHA' &&
  want=\$(jq -r '.images[\"$MINIO\"].id' ~/$PIN_JSON) && have=\$(docker image inspect $MINIO --format '{{.Id}}') &&
  [ \"\$want\" = \"\$have\" ] && echo \"MinIO id matches the pinned manifest (\${have:7:12})\""
box 30 "printf '%s\t%s\n' $MINIO \$(docker image inspect $MINIO --format '{{.Id}}') >> ~/phase2/verified_images.tsv"
SCRATCH=control-tower-sim-scratch:latest  # the simulator's scratch image: only on sims-astra, never built by setup (d9, 15:28)
log "simulator scratch image from sims-astra (read-only docker save of $SCRATCH only)"
DEVBOX=$ASTRA perl -e 'alarm 1200; exec @ARGV' bash "$E/devbox.sh" "docker save $SCRATCH" |
  DEVBOX=$DEVBOX perl -e 'alarm 1200; exec @ARGV' bash "$E/devbox.sh" 'docker load'
box 60 "docker image inspect $SCRATCH --format '{{.Id}}' | grep -q '^sha256:ba1bed0c' && echo 'scratch image ID ba1bed0c'"
[[ -n ${SKIP_SETUP:-} ]] && { log "SKIP_SETUP: no setup_devbox.sh, so no Docker pull (the shared NAT-IP quota is reserved for the box before)"; exit 0; }
log "setup_devbox.sh in tmux -L setup (session setup311)"
box 60 "tmux -L setup new -d -s setup311 'cd ~/phase2/attack_check_311 && bash setup_devbox.sh > ~/phase2/setup311.log 2>&1; echo EXIT \$? >> ~/phase2/setup311.log'"
log "bootstrap done; setup running"
