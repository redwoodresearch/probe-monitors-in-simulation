#!/usr/bin/env bash
# Laptop side, RULE.md Amendment 10: the reviewed 5r command list (Codex r19/r20), run once on d9's GO. Each step must
# pass; the first failure stops with nothing further done:
#  0. S5's cleanup watcher on sims-astra has logged CLEAN (S5's follow-up ended; timers stopped; PAUSE_SIMS absent);
#  1. the 6 q8 servers (il_servers/servers_q8.log) are listed live by `modal app list`;
#  2. the 3 config files and the sims-astra manifests copied into the nm harness (manifests as harness_manifest.tsv and
#     harness_manifest_fu.tsv), then both freezes verified on the box with the gates' own find commands;
#  3. the 6 q8 URLs written to ~/phase2/nm/checks/fu_bases_s5r.txt;
#  4. s5r_handrun.sh <END_FU> on the box (its own checks refuse unless everything is clean; it logs HAND_RUN_s5r.txt).
#   bash s5r_go.sh   (from this folder; scratchpad paths below)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
DBX=$SP/nm_launch/devbox.sh; MODAL=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal; export DEVBOX=aswin@i-05b1d910fd70016e6
END_FU=$(TZ=America/Los_Angeles date -j -f '%Y-%m-%d %H:%M:%S' '2026-10-15 10:24:00' +%s)  # q8 start 00:24:58 + 10 h, conservative
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [5r go] $*"; }
die() { log "STOPPED: $*"; exit 1; }
box() { perl -e 'alarm shift; exec @ARGV' "$1" bash "$DBX" "$2" < /dev/null; }
# 0
c=$(box 60 "tail -1 /home/aswin/phase2/nm/checks/s5_cleanup.log") || die "could not read s5_cleanup.log"
grep -q "CLEAN: sims-astra ready" <<< "$c" || die "S5 cleanup not CLEAN: $c"
log "S5 cleanup CLEAN"
# 1
ids=$(grep -E "^app IDs" $SP/il_servers/servers_q8.log | sed 's/.*: //'); urls=$(grep -E "^base URLs" $SP/il_servers/servers_q8.log | sed 's/.*: //')
[[ $(tr ',' '\n' <<< "$ids" | grep -c '^ap-') == 6 && $(tr ',' '\n' <<< "$urls" | grep -c https) == 6 ]] || die "expected 6 q8 apps and URLs"
perl -e 'alarm 120; exec @ARGV' "$MODAL" app list --json > /tmp/claude-501/s5r_applist.json 2>/dev/null || die "modal app list failed"
python3 - "$ids" /tmp/claude-501/s5r_applist.json <<'PY' || die "a q8 server is not live"
import json, sys
live = {a["app_id"]: a["state"] for a in json.load(open(sys.argv[2]))}
bad = [(i, live.get(i, "GONE")) for i in sys.argv[1].split(",") if not live.get(i, "").startswith(("deployed", "ephemeral"))]
print("q8:", {i: live.get(i, "GONE") for i in sys.argv[1].split(",")})
sys.exit(1 if bad else 0)
PY
log "q8 servers live"
# 2
perl -e 'alarm 120; exec @ARGV' bash "$DBX" rsync -a "$HERE/config/tasks_nm_s5r_launch.json" "$HERE/config/tasks_nm_s5r_fu.json" "$HERE/config/pilot_nm_fu_s5r.yaml" box:phase2/nm/attack_check_nm/config/ < /dev/null || die "config copy"
perl -e 'alarm 120; exec @ARGV' bash "$DBX" rsync -a "$HERE/s5r_handrun.sh" box:phase2/nm/checks/ < /dev/null || die "hand-run copy"
perl -e 'alarm 120; exec @ARGV' bash "$DBX" rsync -a "$HERE/config/harness_manifest_simsastra.tsv" box:phase2/nm/attack_check_nm/config/harness_manifest.tsv < /dev/null || die "main manifest copy"
perl -e 'alarm 120; exec @ARGV' bash "$DBX" rsync -a "$HERE/config/harness_manifest_fu_simsastra.tsv" box:phase2/nm/attack_check_nm/config/harness_manifest_fu.tsv < /dev/null || die "fu manifest copy"
f=$(box 180 "set -o pipefail; cd /home/aswin/phase2/nm/attack_check_nm && (find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' ! -path './config/harness_manifest.tsv' ! -path './config/harness_manifest_fu.tsv' | LC_ALL=C sort | xargs -d '\n' sha256sum) > /tmp/s5r_ff.tsv && cmp -s /tmp/s5r_ff.tsv config/harness_manifest_fu.tsv && echo FU_OK; (find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './config/harness_manifest.tsv' | LC_ALL=C sort | xargs -d '\n' sha256sum) > /tmp/s5r_fm.tsv && cmp -s /tmp/s5r_fm.tsv config/harness_manifest.tsv && echo MAIN_OK")
rc=$?; [[ $rc == 0 && $f == *FU_OK* && $f == *MAIN_OK* ]] || die "freeze check failed (exit $rc): $f"  # Codex r21
log "files installed; both freezes pass"
# 3
box 60 "echo '$urls' > /home/aswin/phase2/nm/checks/fu_bases_s5r.txt && [ \$(tr ',' '\n' < /home/aswin/phase2/nm/checks/fu_bases_s5r.txt | grep -c https) = 6 ]" || die "fu_bases_s5r.txt"
log "fu_bases_s5r.txt written (6 q8 URLs)"
# 4
out=$(box 900 "bash /home/aswin/phase2/nm/checks/s5r_handrun.sh $END_FU"); rc=$?
echo "$out" | tail -8
[[ $rc == 0 ]] && grep -q "FOLLOW-UP STARTED" <<< "$out" || die "s5r_handrun.sh exit $rc"
log "5r STARTED"
