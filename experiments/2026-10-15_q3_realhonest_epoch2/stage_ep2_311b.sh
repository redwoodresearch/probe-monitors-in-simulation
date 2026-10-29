#!/usr/bin/env bash
# Laptop side: stage the +311 tail launch ep2_311b on xbox2 (RULE.md amendment 3, box changed to xbox2 ~14:31; was q3-realhonest), after its relay of the 3 missing approach
# files (relay_approaches.sh; results/relay_approaches_ep2_311b_box1_to_q3-realhonest.json). Configs, code and hashes only; no
# approach or attack file is moved here. Refuses if any target folder exists. In order, each must succeed:
#   1. the frozen +311 harness at ~/phase2/attack_check_311: the laptop's scale-up harness copy minus its follow-up files (names with
#      "_fu", fu_launch.sh, LAUNCH_COMMIT_FU; added after the freeze) and results/, then the freeze diff on the box against the
#      committed manifest (@ 9d75353d5, 125 files): must be empty;
#   2. ~/phase2/ep2_311: launch 1's staged inputs, from the same commits as stage_ep2_311.sh, and launch 1's tasks file rebuilt by
#      the same command, which must hash to 3b645674... (the file the split record names);
#   3. ~/phase2/ep2_311b (from this folder @ HEAD: pilot_honest_ep2_311b.yaml, make_tasks_ep2_311b.py, gate_ep2_arms.py,
#      launch_ep2_311b.sh, results/split_311_P265_2026-10-15.json; the env template @ 28fb1cf6a; the manifest @ 9d75353d5) and
#      the 44-cell tasks file (launch 1's cells[265:]); every staged file's sha256 printed, then made read-only.
#   bash stage_ep2_311b.sh
set -euo pipefail
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
export DEVBOX=aswin@i-00db08777bb29d430  # xbox2 (09's amendment @ 81f75b763, 0d via d9 ~14:31): was q3-realhonest i-0abea93b8ad99ee41
HERE=$(cd "$(dirname "$0")" && pwd); W=$(git -C "$HERE" rev-parse --show-toplevel); F=experiments/2026-10-15_q3_realhonest_epoch2
HB=$SP/scaleup311/harness/experiments/2026-10-07_phase2_attack_check
box() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$SP/nm_launch/devbox.sh" "$2" < /dev/null; }
put() { perl -e 'alarm 600; exec @ARGV' bash "$SP/nm_launch/devbox.sh" rsync "$@"; }
L1=$SP/ep2/stage_311b_l1 L2=$SP/ep2/stage_311b_l2
[[ ! -e $L1 && ! -e $L2 ]] || { echo "$L1 or $L2 exists"; exit 1; }
[[ -z $(git -C "$W" status --porcelain -- "$F/config/pilot_honest_ep2_311b.yaml" "$F/make_tasks_ep2_311b.py" "$F/launch_ep2_311b.sh" \
  "$F/results/split_311_P265_2026-10-15.json" "$F/gate_ep2_arms.py" "$F/config/tasks_ep2_311_309.json" "$F/make_tasks_ep2_311.py") ]] ||
  { echo "commit the staged files first"; exit 1; }
box 60 "! test -e ~/phase2/attack_check_311 && ! test -e ~/phase2/ep2_311 && ! test -e ~/phase2/ep2_311b" || { echo "a target folder exists on the box"; exit 1; }
mkdir -p "$L1" "$L2"
for f in config/tasks_ep2_311_309.json make_tasks_ep2_311.py; do git -C "$W" show "HEAD:$F/$f" > "$L1/$(basename "$f")"; done
git -C "$W" show 52350b585:experiments/2026-10-14_q3_scaleup_311/config/tasks_311_q4_launch.json > "$L1/tasks_311_q4_launch_85.json"
for f in config/pilot_honest_ep2_311b.yaml make_tasks_ep2_311b.py gate_ep2_arms.py launch_ep2_311b.sh results/split_311_P265_2026-10-15.json; do
  git -C "$W" show "HEAD:$F/$f" > "$L2/$(basename "$f")"; done
git -C "$W" show 28fb1cf6a:experiments/2026-10-14_q3_q35_realhonest_1e4/config/env.vllm-only.tpl > "$L2/env.vllm-only.tpl"
git -C "$W" show 9d75353d5:experiments/2026-10-14_q3_scaleup_311/config/harness_manifest.tsv > "$L2/harness_manifest_311.tsv"
(cd "$L1" && shasum -a 256 -c <<< "033d4bebf7bb93c78cada607f698e9ce32e157a6d808adb735d24574d6fc8a2e  tasks_311_q4_launch_85.json")
(cd "$L2" && shasum -a 256 -c <<< "5d17a3966386446e41450a6c85e9948c4d7ee9e98ae7e424cb1c6d82ef874591  harness_manifest_311.tsv")
# 1. the harness
box 60 "mkdir -p ~/phase2/attack_check_311/results"
put -a --exclude __pycache__ --exclude 'results/*' --exclude '*_fu*' --exclude fu_launch.sh --exclude LAUNCH_COMMIT_FU "$HB/" box:phase2/attack_check_311/
put -a "$L2/harness_manifest_311.tsv" box:/tmp/ep2_311b_manifest.tsv
box 120 'cd ~/phase2/attack_check_311 && diff <(find . -type f ! -path "./results/*" ! -path "*/__pycache__/*" ! -path "./config/harness_manifest.tsv" | LC_ALL=C sort | xargs -d "\n" sha256sum) /tmp/ep2_311b_manifest.tsv > /tmp/ep2_311b_freeze.txt; rc=$?; echo "freeze diff rc $rc, $(wc -l < /tmp/ep2_311b_freeze.txt) lines, $(wc -l < /tmp/ep2_311b_manifest.tsv) manifest files"; exit $rc' ||
  { echo "the +311 harness on the box differs from the frozen manifest"; exit 1; }
# 2. launch 1's tasks file, rebuilt
box 60 "mkdir -p ~/phase2/ep2_311"
put -a --ignore-existing "$L1/" box:phase2/ep2_311/
box 300 "cd ~/phase2/attack_check_311 && ~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/ep2_311/make_tasks_ep2_311.py \
  ~/phase2/ep2_311/tasks_ep2_311_309.json ~/phase2/ep2_311/pilot_tasks_ep2_311_309.json config/tasks_311_q1_launch.json \
  config/tasks_311_q2_launch.json config/tasks_311_q3_launch.json ~/phase2/ep2_311/tasks_311_q4_launch_85.json | tail -1"
box 60 "sha256sum ~/phase2/ep2_311/pilot_tasks_ep2_311_309.json | grep -q '^3b645674937459b697a9fec427aa703b322082260b34f0485a22eb6ac37a8a84 ' && echo 'launch 1 tasks file reproduced: 3b645674' && chmod a-w ~/phase2/ep2_311/*" ||
  { echo "launch 1's tasks file did not reproduce"; exit 1; }
# 3. ep2_311b
box 60 "mkdir -p ~/phase2/ep2_311b"
put -a --ignore-existing "$L2/" box:phase2/ep2_311b/
box 300 "cd ~/phase2/attack_check_311 && ~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/ep2_311b/make_tasks_ep2_311b.py \
  ~/phase2/ep2_311/pilot_tasks_ep2_311_309.json ~/phase2/ep2_311b/split_311_P265_2026-10-15.json ~/phase2/ep2_311b/pilot_tasks_ep2_311b_44.json"
echo "laptop staging:"; (cd "$L2" && shasum -a 256 *)
echo "box:"; box 60 "cd ~/phase2/ep2_311b && sha256sum * && chmod a-w *"
rm -rf "$L1" "$L2"; echo "laptop staging removed: $(test -e "$L1" -o -e "$L2" && echo NO || echo yes)"
