#!/usr/bin/env bash
# stage_rlr.sh (sim-honest replacement launches; derived from stage_nm.sh, Codex r12c AGREE): stage the replacement start
# path's small files on a host in /home/aswin/phase2/simh_rlr_stage, by sha256: simh_rlr_box.sh and simh_rlr_manifest.py
# (this branch's commit), the replacement configs and task files (config_rlr/), the parents' task files (+311: the frozen
# follow-up files; nm360: config_nm's simh task files), the mains' own task files and the families' committed frozen harness
# manifests (+311: origin/aswin/q3-scaleup-311 config/; nm360: origin/aswin/q3-new-mains-launch config/ and config_nm's
# box4_s4r copy), and gate_pause_311_rl.py (@ e46ba05d0). Same guarantees as stage_nm.sh.
#   stage_rlr.sh <host instance> <record json>
set -euo pipefail
INST=$1 REC=$2
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims; WT=$SP/wt_simh311; E=$WT/experiments/2026-10-15_q3_simhonest_311
H=$SP/simh_launch_X/h; DST=/home/aswin/phase2/simh_rlr_stage
C=$(mktemp -d "$SP/simh_drive/stage_rlr_XXXXXX")
NM=origin/aswin/q3-new-mains-launch; NMC=experiments/2026-10-14_q3_new_mains_launch/config
for f in harness_manifest_fu_nm.tsv harness_manifest_fu_box4.tsv harness_manifest_fu_simsastra.tsv tasks_nm_s3_launch.json tasks_nm_s4_launch.json tasks_nm_s2_launch.json tasks_nm_s4r_launch.json; do
  git -C "$REPO" show "${NM}:$NMC/$f" > "$C/$f"
done
SU=origin/aswin/q3-scaleup-311; SUC=experiments/2026-10-14_q3_scaleup_311/config
for f in harness_manifest_fu_q123.tsv harness_manifest_fu_q2r.tsv harness_manifest_fu_q4.tsv \
         tasks_311_q1_launch.json tasks_311_q2_launch.json tasks_311_q3_launch.json tasks_311_q4_launch.json \
         tasks_311_q1_fu.json tasks_311_q2_fu.json tasks_311_q3_fu.json tasks_311_q4_fu.json; do
  git -C "$REPO" show "${SU}:$SUC/$f" > "$C/$f"
done
git -C "$REPO" show "e46ba05d0:experiments/2026-10-15_q3_interleaved_run3_sim2/gate_pause_311_rl.py" > "$C/gate_pause_311_rl.py"
# S_paper493 (d9 ~17:40): the T196 cells' source task file and primary_v3's merged refs index; each sha-pinned by its writer
git -C "$REPO" show "origin/aswin/q3-interleaved-run3-sim2:experiments/2026-10-15_q3_interleaved_run3_sim2/config/tasks_il_old_all.json" > "$C/tasks_il_old_all.json"
git -C "$REPO" show "origin/aswin/q3-q35-simhonest-purchase:experiments/2026-10-14_q3_q35_simhonest_purchase/results/merged_refs_launch_2026-10-14.json" > "$C/merged_refs_launch_2026-10-14.json"
cp "$E/simh_sp493_box.sh" "$E/simh_sp493_manifest.py" "$E"/config_sp493/pilot_simh_*.yaml "$E"/config_sp493/tasks_simh_*.json "$E"/config_sp493/sp493_v2_r1_entries.json "$C/"
COMMIT=$(git -C "$WT" rev-parse HEAD)
[[ -z $(git -C "$WT" status --porcelain -- experiments/2026-10-15_q3_simhonest_311/simh_rlr_box.sh experiments/2026-10-15_q3_simhonest_311/simh_rlr_manifest.py experiments/2026-10-15_q3_simhonest_311/config_rlr experiments/2026-10-15_q3_simhonest_311/config_nm experiments/2026-10-15_q3_simhonest_311/simh_sp493_box.sh experiments/2026-10-15_q3_simhonest_311/simh_sp493_manifest.py experiments/2026-10-15_q3_simhonest_311/config_sp493) ]] \
  || { echo "commit simh_rlr_box.sh, simh_rlr_manifest.py, config_rlr and config_nm first"; exit 1; }
cp "$E/simh_rlr_box.sh" "$E/simh_rlr_manifest.py" "$E"/config_rlr/pilot_simh_*_rlr.yaml "$E"/config_rlr/tasks_simh_*_rlr.json "$E"/config_rlr/pilot_simh_*_rlr2.yaml "$E"/config_rlr/tasks_simh_*_rlr2.json \
   "$E/config_nm/tasks_simh_nm_s3.json" "$E/config_nm/tasks_simh_nm_s4d.json" "$E/config_nm/tasks_simh_nm_s2.json" "$E/config_nm/tasks_simh_nm_s4r.json" "$E/config_nm/harness_manifest_fu_box4_s4r.tsv" "$C/"
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
json.dump({"host": inst, "stage": "/home/aswin/phase2/simh_rlr_stage", "branch_commit": commit, "already_present_identical": int(pre),
           "finished_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T"),
           "files": [{"path": l.split()[1], "sha256": l.split()[0]} for l in open(man).read().splitlines()]}, open(rec, "w"), indent=1)
PY
echo "staged $N files on $INST ($DST), all verified; record $REC"
