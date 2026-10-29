#!/usr/bin/env bash
# Laptop side: stage S_paper493's real arms on one box in ~/phase2/sp493/, OUTSIDE the frozen harness (~/phase2/attack_check_311
# stays byte-identical to its manifest). Configs, scripts and key-only task files, each from a commit; NO approach or attack file
# is moved (approach texts are created on the box by record_fleet_approaches.py in prep_sp493.sh). Read-only checks first:
# the harness hashes as the frozen manifest (5d17a396...), no launcher runs, ~/phase2/sp493 doesn't exist.
#   bash stage_sp493.sh <A|B|C> <instance id>
set -euo pipefail
X=$1 I=$2; [[ $X == A || $X == B || $X == C ]]
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
export DEVBOX=aswin@$I
HERE=$(cd "$(dirname "$0")" && pwd); W=$(git -C "$HERE" rev-parse --show-toplevel); F=experiments/2026-10-15_q3_sp493
box() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$SP/nm_launch/devbox.sh" "$2" < /dev/null; }
FILES="config/pilot_sp493_real_box$X.yaml config/tasks_sp493_box$X.json config/honest_keys_sp493.json prep_sp493.sh
  launch_sp493.sh gate_sp493.py add_sp493_arms.py"
[[ -z $(git -C "$W" status --porcelain -- $(for f in $FILES; do echo "$F/$f"; done)) ]] || { echo "commit the staged files first"; exit 1; }
L=$SP/bfork/stage_$X; [[ ! -e $L ]] || { echo "$L exists"; exit 1; }
mkdir -p "$L"
for f in $FILES; do git -C "$W" show "HEAD:$F/$f" > "$L/$(basename "$f")"; done
(cd "$L" && sha256sum < "pilot_sp493_real_box$X.yaml" | cut -d' ' -f1 > "pilot_sp493_real_box$X.yaml.sha256")
git -C "$W" show 28fb1cf6a:experiments/2026-10-14_q3_q35_realhonest_1e4/config/env.vllm-only.tpl > "$L/env.vllm-only.tpl"
git -C "$W" show 9d75353d5:experiments/2026-10-14_q3_scaleup_311/config/harness_manifest.tsv > "$L/harness_manifest_311.tsv"
git -C "$W" show 9d75353d5:experiments/2026-10-14_q3_scaleup_311/add_r2.py > "$L/add_r2.py"
(cd "$L" && shasum -a 256 -c <<'EOF'
5d17a3966386446e41450a6c85e9948c4d7ee9e98ae7e424cb1c6d82ef874591  harness_manifest_311.tsv
EOF
)
EXR=/dev/null; if git -C "$W" cat-file -e "HEAD:$F/config/freeze_exempt_box$X.tsv" 2>/dev/null; then
  git -C "$W" show "HEAD:$F/config/freeze_exempt_box$X.tsv" > "$L/freeze_exempt_box$X.tsv"
  (cd "$L" && sha256sum < "freeze_exempt_box$X.tsv" | cut -d' ' -f1 > "freeze_exempt_box$X.tsv.sha256"); EXR="$L/freeze_exempt_box$X.tsv"; fi
# the box's file list minus the exact exempt lines must equal the frozen manifest (exemption list sent first, to /tmp)
perl -e 'alarm 60; exec @ARGV' bash "$SP/nm_launch/devbox.sh" "cat > /tmp/sp493_exempt_$X.tsv" < "$EXR" || { echo "exemption copy failed"; rm -rf "$L"; exit 1; }
perl -e 'alarm 120; exec @ARGV' bash "$SP/nm_launch/devbox.sh" "set -o pipefail; cd ~/phase2/attack_check_311 && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './config/harness_manifest.tsv' | LC_ALL=C sort | xargs -d '\n' sha256sum > /tmp/sp493_list_$X.txt && grep -vxF -f /tmp/sp493_exempt_$X.tsv /tmp/sp493_list_$X.txt > /tmp/sp493_kept_$X.txt && diff /tmp/sp493_kept_$X.txt - > /dev/null" < "$L/harness_manifest_311.tsv" \
  && echo "box harness: matches the frozen manifest" || { echo "box harness differs from the frozen manifest (or is missing)"; rm -rf "$L"; exit 1; }
box 60 "! pgrep -f '[r]un_pilot.py --pilot-config' > /dev/null" || { echo "a launcher runs on the box"; rm -rf "$L"; exit 1; }
box 60 "! test -e ~/phase2/sp493 && mkdir -p ~/phase2/sp493" || { echo "~/phase2/sp493 exists on the box"; rm -rf "$L"; exit 1; }
perl -e 'alarm 300; exec @ARGV' bash "$SP/nm_launch/devbox.sh" rsync -a --ignore-existing "$L/" box:phase2/sp493/
echo "laptop staging:"; (cd "$L" && shasum -a 256 *)
echo "box:"; box 60 "cd ~/phase2/sp493 && sha256sum *"
rm -rf "$L"; echo "laptop staging removed: $(test -e "$L" && echo NO || echo yes)"
