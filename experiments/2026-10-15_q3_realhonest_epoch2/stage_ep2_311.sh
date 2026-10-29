#!/usr/bin/env bash
# Laptop side: stage the +311 epoch-2 launch on q3-scaleup4 (RULE.md, amendment 1) in ~/phase2/ep2_311/, OUTSIDE the frozen harness
# (~/phase2/attack_check_311 stays byte-identical to its manifest), then build the 309-cell tasks file there. Configs and hashes only:
# no approach or attack file is moved here (those go through relay_approaches.sh only). Every file comes from a commit:
#   - this folder @ HEAD (committed first): pilot_honest_ep2_311.yaml, tasks_ep2_311_309.json, make_tasks_ep2_311.py, gate_ep2_arms.py,
#     launch_ep2_311.sh;
#   - the vLLM-only env template (references only, no value) @ 28fb1cf6a (the 1e-4 job's, also on q3-realhonest's T196 harness);
#   - the frozen +311 manifest @ 9d75353d5 (5d17a396...), and Q4's 85-cell launch file @ 52350b585 (033d4beb..., Amendment 5).
# Refuses if ~/phase2/ep2_311 exists on the box. Prints every staged file's sha256 and the builder's output.
#   bash stage_ep2_311.sh
set -euo pipefail
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
export DEVBOX=aswin@i-0ad4140e97b11d30f
HERE=$(cd "$(dirname "$0")" && pwd); W=$(git -C "$HERE" rev-parse --show-toplevel); F=experiments/2026-10-15_q3_realhonest_epoch2
box() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$SP/nm_launch/devbox.sh" "$2" < /dev/null; }
L=$SP/ep2/stage_311; [[ ! -e $L ]] || { echo "$L exists"; exit 1; }
[[ -z $(git -C "$W" status --porcelain -- "$F/config/pilot_honest_ep2_311.yaml" "$F/config/tasks_ep2_311_309.json" "$F/make_tasks_ep2_311.py" \
  "$F/gate_ep2_arms.py" "$F/launch_ep2_311.sh") ]] || { echo "commit the staged files first"; exit 1; }
mkdir -p "$L"
for f in config/pilot_honest_ep2_311.yaml config/tasks_ep2_311_309.json make_tasks_ep2_311.py gate_ep2_arms.py launch_ep2_311.sh; do
  git -C "$W" show "HEAD:$F/$f" > "$L/$(basename "$f")"; done
git -C "$W" show 28fb1cf6a:experiments/2026-10-14_q3_q35_realhonest_1e4/config/env.vllm-only.tpl > "$L/env.vllm-only.tpl"
git -C "$W" show 9d75353d5:experiments/2026-10-14_q3_scaleup_311/config/harness_manifest.tsv > "$L/harness_manifest_311.tsv"
git -C "$W" show 52350b585:experiments/2026-10-14_q3_scaleup_311/config/tasks_311_q4_launch.json > "$L/tasks_311_q4_launch_85.json"
(cd "$L" && shasum -a 256 -c <<'EOF'
5d17a3966386446e41450a6c85e9948c4d7ee9e98ae7e424cb1c6d82ef874591  harness_manifest_311.tsv
033d4bebf7bb93c78cada607f698e9ce32e157a6d808adb735d24574d6fc8a2e  tasks_311_q4_launch_85.json
EOF
)
box 60 "! test -e ~/phase2/ep2_311 && mkdir -p ~/phase2/ep2_311" || { echo "~/phase2/ep2_311 exists on the box (or mkdir failed)"; exit 1; }
perl -e 'alarm 300; exec @ARGV' bash "$SP/nm_launch/devbox.sh" rsync -a --ignore-existing "$L/" box:phase2/ep2_311/
box 300 "cd ~/phase2/attack_check_311 && ~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/ep2_311/make_tasks_ep2_311.py \
  ~/phase2/ep2_311/tasks_ep2_311_309.json ~/phase2/ep2_311/pilot_tasks_ep2_311_309.json config/tasks_311_q1_launch.json \
  config/tasks_311_q2_launch.json config/tasks_311_q3_launch.json ~/phase2/ep2_311/tasks_311_q4_launch_85.json"
echo "laptop staging:"; (cd "$L" && shasum -a 256 *)
echo "box:"; box 60 "cd ~/phase2/ep2_311 && sha256sum * && chmod a-w *"
rm -rf "$L"; echo "laptop staging removed: $(test -e "$L" && echo NO || echo yes)"
