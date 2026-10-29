#!/usr/bin/env bash
# stage_nm.sh (nm360 sim-honest): stage the start path's small files on a host in /home/aswin/phase2/simh_nm_stage, by
# sha256: simhnm_box.sh and simh_nm_manifest.py (this branch's commit), the groups' configs and task files (config_nm/),
# the three committed frozen nm harness manifests and the nm mains' own task files (origin/aswin/q3-new-mains-launch
# config/), and gate_pause_311_rl.py (@ e46ba05d0). A file already on the host must hash the same (else abort before a byte
# moves); rsync --ignore-existing; every file re-verified on the host. Code, configs and keys only; no transcript.
#   stage_nm.sh <host instance> <record json>
set -euo pipefail
INST=$1 REC=$2
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims; WT=$SP/wt_simh311; E=$WT/experiments/2026-10-15_q3_simhonest_311
H=$SP/simh_launch_X/h; DST=/home/aswin/phase2/simh_nm_stage
C=$(mktemp -d "$SP/simh_drive/stage_nm_XXXXXX")
NM=origin/aswin/q3-new-mains-launch; NMC=experiments/2026-10-14_q3_new_mains_launch/config
for f in harness_manifest_fu_nm.tsv harness_manifest_fu_box4.tsv harness_manifest_fu_simsastra.tsv \
         tasks_nm_s1_launch.json tasks_nm_s2_launch.json tasks_nm_s3_launch.json tasks_nm_s4_launch.json tasks_nm_s4r_launch.json tasks_nm_s5_launch.json; do
  git -C "$REPO" show "${NM}:$NMC/$f" > "$C/$f"
done
git -C "$REPO" show "e46ba05d0:experiments/2026-10-15_q3_interleaved_run3_sim2/gate_pause_311_rl.py" > "$C/gate_pause_311_rl.py"
COMMIT=$(git -C "$WT" rev-parse HEAD)
[[ -z $(git -C "$WT" status --porcelain -- experiments/2026-10-15_q3_simhonest_311/simhnm_box.sh experiments/2026-10-15_q3_simhonest_311/simh_nm_manifest.py experiments/2026-10-15_q3_simhonest_311/config_nm) ]] \
  || { echo "commit simhnm_box.sh, simh_nm_manifest.py and config_nm first"; exit 1; }
cp "$E/simhnm_box.sh" "$E/simh_nm_manifest.py" "$E"/config_nm/pilot_simh_nm_*.yaml "$E"/config_nm/tasks_simh_nm_*.json "$E/config_nm/harness_manifest_fu_box4_s4r.tsv" "$C/"
MAN=$C.manifest.sha  # Codex r12: private to this invocation (outside the rsynced cache), sha-pinned on the host
(cd "$C" && shasum -a 256 * | sed 's#  #  ./#') > "$MAN"
N=$(wc -l < "$MAN" | tr -d ' '); MSHA=$(shasum -a 256 < "$MAN" | cut -c1-64)
RECV="T=\$(mktemp); cat > \$T; [[ \$(sha256sum < \$T | cut -c1-64) == $MSHA ]] || { echo MANIFEST_BAD; exit 1; }"
box() { (cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 300; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh "$@"); }
PRE=$(box "$RECV; mkdir -p $DST && cd $DST && e=0; d=0; while read -r h p; do if [[ -e \$p ]]; then e=\$((e+1)); [[ \$(sha256sum < \$p | cut -c1-64) == \$h ]] || d=\$((d+1)); fi; done < \$T; echo EXISTING=\$e DIFFER=\$d" < "$MAN" 2>/dev/null | grep -E '^EXISTING=' || true)
[[ $PRE =~ ^EXISTING=([0-9]+)\ DIFFER=0$ ]] || { echo "host stage holds differing files or check failed: '$PRE'"; exit 1; }
(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 300; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  rsync -a --ignore-existing "$C/" "box:$DST/") > /dev/null 2>&1 || true  # verified below either way
V=$(box "$RECV; cd $DST && sha256sum -c \$T 2>/dev/null | grep -c ': OK\$'" < "$MAN" 2>/dev/null | tail -1)
[[ $V == "$N" ]] || { echo "host verify $V of $N"; exit 1; }
python3 -I - "$MAN" "$REC" "$INST" "$COMMIT" "${BASH_REMATCH[1]}" <<'PY'
import json, sys, datetime
from zoneinfo import ZoneInfo
man, rec, inst, commit, pre = sys.argv[1:]
json.dump({"host": inst, "stage": "/home/aswin/phase2/simh_nm_stage", "branch_commit": commit, "already_present_identical": int(pre),
           "finished_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T"),
           "files": [{"path": l.split()[1], "sha256": l.split()[0]} for l in open(man).read().splitlines()]}, open(rec, "w"), indent=1)
PY
echo "staged $N files on $INST ($DST), all verified; record $REC"
