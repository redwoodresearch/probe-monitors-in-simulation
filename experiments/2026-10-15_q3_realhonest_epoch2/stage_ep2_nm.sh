#!/usr/bin/env bash
# Laptop side: stage the nm360 epoch-2 launch on q3-ep2-nm (RULE.md amendment 2): stage_ep2_311.sh for nm, plus the frozen nm harness
# itself (this box has none yet). Configs, code and hashes only: no approach or attack file is moved here (the 14 missing approach
# files went through relay_approaches.sh, results/relay_approaches_ep2_nm_box1_to_q3-ep2-nm.json).
#   1. the harness: the laptop's nm harness copy (scratchpad nm_launch/harness, the one box_setup_nm.sh sent to the nm boxes) minus
#      the 3 server configs added after the freeze, rsync'd to ~/phase2/nm/attack_check_nm (refused if that exists), then the freeze
#      diff on the box against the committed manifest (harness_manifest_nm.tsv @ 7fce6fb25, 178 files): must be empty;
#   2. ~/phase2/ep2_nm (refused if it exists), from commits: this folder @ HEAD (pilot_honest_ep2_nm.yaml, tasks_ep2_nm_351.json,
#      make_tasks_ep2_nm.py, gate_ep2_arms.py, launch_ep2_nm.sh), the vLLM-only env template @ 28fb1cf6a, the manifest @ HEAD;
#   3. the 351-cell tasks file built on the box from the harness's S1-S5 launch files; every staged file's sha256 printed.
#   bash stage_ep2_nm.sh
set -euo pipefail
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
export DEVBOX=aswin@i-07f91f4dbd4192faa
HERE=$(cd "$(dirname "$0")" && pwd); W=$(git -C "$HERE" rev-parse --show-toplevel); F=experiments/2026-10-15_q3_realhonest_epoch2
MAN=experiments/2026-10-14_q3_new_mains_launch/config/harness_manifest_nm.tsv
HN=$SP/nm_launch/harness/experiments/2026-10-07_phase2_attack_check
box() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$SP/nm_launch/devbox.sh" "$2" < /dev/null; }
L=$SP/ep2/stage_nm; [[ ! -e $L ]] || { echo "$L exists"; exit 1; }
[[ -z $(git -C "$W" status --porcelain -- "$F/config/pilot_honest_ep2_nm.yaml" "$F/config/tasks_ep2_nm_351.json" "$F/make_tasks_ep2_nm.py" \
  "$F/gate_ep2_arms.py" "$F/launch_ep2_nm.sh" "$MAN") ]] || { echo "commit the staged files first"; exit 1; }
mkdir -p "$L"
for f in config/pilot_honest_ep2_nm.yaml config/tasks_ep2_nm_351.json make_tasks_ep2_nm.py gate_ep2_arms.py launch_ep2_nm.sh; do
  git -C "$W" show "HEAD:$F/$f" > "$L/$(basename "$f")"; done
git -C "$W" show 28fb1cf6a:experiments/2026-10-14_q3_q35_realhonest_1e4/config/env.vllm-only.tpl > "$L/env.vllm-only.tpl"
git -C "$W" show "HEAD:$MAN" > "$L/harness_manifest_nm.tsv"
(cd "$L" && shasum -a 256 -c <<< "59bccf88aab48df1c2d562634d36137363e88c4f05667a44d431486cd4443a51  harness_manifest_nm.tsv")
# 1. the harness
box 60 "! test -e ~/phase2/nm/attack_check_nm && mkdir -p ~/phase2/nm/attack_check_nm/results" || { echo "~/phase2/nm/attack_check_nm exists"; exit 1; }
perl -e 'alarm 900; exec @ARGV' bash "$SP/nm_launch/devbox.sh" rsync -a --exclude __pycache__ --exclude 'results/*' \
  --exclude config/servers_chain.yaml --exclude config/servers_nm_fu_long.yaml --exclude config/servers_nm_s5.yaml "$HN/" box:phase2/nm/attack_check_nm/
perl -e 'alarm 120; exec @ARGV' bash "$SP/nm_launch/devbox.sh" rsync -a "$L/harness_manifest_nm.tsv" box:/tmp/ep2_nm_manifest.tsv
box 120 'cd ~/phase2/nm/attack_check_nm && diff <(find . -type f ! -path "./results/*" ! -path "*/__pycache__/*" ! -path "./config/harness_manifest.tsv" | LC_ALL=C sort | xargs -d "\n" sha256sum) /tmp/ep2_nm_manifest.tsv > /tmp/ep2_nm_freeze.txt; rc=$?; echo "freeze diff rc $rc, $(wc -l < /tmp/ep2_nm_freeze.txt) lines, $(wc -l < /tmp/ep2_nm_manifest.tsv) manifest files"; exit $rc' ||
  { echo "the nm harness on the box differs from the frozen manifest"; exit 1; }
# 2. the staged folder
box 60 "! test -e ~/phase2/ep2_nm && mkdir -p ~/phase2/ep2_nm" || { echo "~/phase2/ep2_nm exists on the box (or mkdir failed)"; exit 1; }
perl -e 'alarm 300; exec @ARGV' bash "$SP/nm_launch/devbox.sh" rsync -a --ignore-existing "$L/" box:phase2/ep2_nm/
# 3. the tasks file
box 300 "cd ~/phase2/nm/attack_check_nm && ~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/ep2_nm/make_tasks_ep2_nm.py \
  ~/phase2/ep2_nm/tasks_ep2_nm_351.json ~/phase2/ep2_nm/pilot_tasks_ep2_nm_351.json config/tasks_nm_s1_launch.json \
  config/tasks_nm_s2_launch.json config/tasks_nm_s3_launch.json config/tasks_nm_s4_launch.json config/tasks_nm_s5_launch.json"
echo "laptop staging:"; (cd "$L" && shasum -a 256 *)
echo "box:"; box 60 "cd ~/phase2/ep2_nm && sha256sum * && chmod a-w *"
rm -rf "$L"; echo "laptop staging removed: $(test -e "$L" && echo NO || echo yes)"
