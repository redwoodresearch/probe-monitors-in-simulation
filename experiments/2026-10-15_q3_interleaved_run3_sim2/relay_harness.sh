#!/usr/bin/env bash
# relay_harness.sh (d9/0d 04:53, oldc -> sims-astra): copy the FROZEN +311 harness (code and config only: exactly the files
# listed in config/harness_manifest_fu.tsv, plus that manifest) from a box whose harness passes the freeze check to a box
# that has none, at the same path, and prove the copy equals the manifest line for line. No transcripts, no results, no
# compose files, no credentials (config/env.aws.tpl holds secret REFERENCES for rr-env, no values). Never overwrites: the
# destination harness directory must be absent. The laptop cache is verified and deleted.
#   relay_harness.sh <src instance> <dst instance> <record json>
set -euo pipefail
SRCI=$1 DSTI=$2 REC=$3
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
H=$SP/simh_launch_X/h; C=$SP/relay_cache_harness311; HS=/home/aswin/phase2/attack_check_311; MAN=config/harness_manifest_fu.tsv
box() { local i=$1; shift; (cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$i perl -e 'alarm 300; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh "$@"); }
FREEZE="cd $HS && set -o pipefail && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' ! -path './config/harness_manifest*.tsv' ! -path './config/pilot_il_*' ! -path './config/tasks_il_*' ! -path './config/pilot_ilrr_*' ! -path './config/tasks_ilrr_*' ! -path './config/ilrr_*' ! -path './config/pilot_ilrr2_*' ! -path './config/pilot_ilrr3_*' ! -path './config/tasks_ilrr3_*' ! -path './config/ilrr3_*' ! -path './config/pilot_ilc_*' ! -path './config/tasks_ilc_*' ! -path './config/ilc_*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > /tmp/relay_h_now.tsv && grep -v '  ./config/harness_manifest' $MAN > /tmp/relay_h_frozen.tsv && cmp -s /tmp/relay_h_now.tsv /tmp/relay_h_frozen.tsv && echo FREEZE_OK \$(wc -l < /tmp/relay_h_now.tsv)"
[[ ! -e $C ]] || { echo "cache $C exists"; exit 1; }
mkdir -p "$C/root"
# 1. ONE source call: the freeze check, and in the same call the exact frozen list it compared (the manifest minus its own
#    line) plus the manifest's own hash line (Codex r25: the exported list is the snapshot the freeze used)
box "$SRCI" "$FREEZE && cat /tmp/relay_h_frozen.tsv && cd $HS && sha256sum ./$MAN" 2> /dev/null > "$C/source.out" || { echo "source call failed"; exit 1; }
S=$(grep '^FREEZE_OK' "$C/source.out" || true)
[[ $S == FREEZE_OK\ * ]] || { echo "source harness fails its freeze check: '$S'"; exit 1; }
grep -E '^[0-9a-f]{64}  ' "$C/source.out" > "$C/manifest.sha"
N=$(wc -l < "$C/manifest.sha" | tr -d ' ')
(( N == ${S#FREEZE_OK } + 1 )) || { echo "manifest $N lines vs freeze ${S#FREEZE_OK } + 1"; exit 1; }
# every line exact (no whitespace in names) and every file on the permitted list; anything else aborts before a byte moves
python3 - "$C/manifest.sha" "$C/files.txt" <<'PY' || { echo "the file list failed validation"; exit 1; }
import re, sys
lines = open(sys.argv[1]).read().splitlines()
ok_line = re.compile(r"[0-9a-f]{64}  \./([A-Za-z0-9._+/-]+)")
ok_name = re.compile(r".*\.(py|sh|md|txt|json|yaml|yml|tsv|tpl|toml|cfg|ini|patch)|fixed_digest_lookup/docker|no_digest_lookup/docker|minio/Dockerfile|LAUNCH_COMMIT[A-Z_]*")
bad_name = re.compile(r"(^|.*/)(\.env[^/]*|id_[^/]*|[^/]*\.(pem|key|eval|p12|crt)|credentials[^/]*|[^/]*compose[^/]*)$|(^|/)results/")
paths = []
for l in lines:
    m = ok_line.fullmatch(l)
    if not m: sys.exit(f"bad manifest line: {l[:90]!r}")
    p = m.group(1)
    if bad_name.search(p) or not ok_name.fullmatch(p): sys.exit(f"not permitted: {p}")
    paths.append(p)
assert len(set(paths)) == len(paths)
open(sys.argv[2], "w").write("\n".join(paths) + "\n")
PY
# 2. source -> laptop cache, verified
(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$SRCI perl -e 'alarm 600; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  rsync -a --files-from="$C/files.txt" "box:$HS/" "$C/root/") > /dev/null 2>&1 || { echo "rsync from source failed"; exit 1; }
OKC=$(cd "$C/root" && shasum -a 256 -c ../manifest.sha 2>/dev/null | grep -c ': OK$' || true)
[[ $OKC == "$N" ]] || { echo "cache verify $OKC of $N"; exit 1; }
# 3. destination: the harness directory must be absent; copy; then the destination passes the same freeze check
A=$(box "$DSTI" "[[ -e $HS || -L $HS ]] && echo PRESENT || echo ABSENT" 2>/dev/null | grep -E '^(PRESENT|ABSENT)$' || true)
[[ $A == ABSENT ]] || { echo "destination $HS: '$A'"; exit 1; }
box "$DSTI" "mkdir $HS" > /dev/null 2>&1 || { echo "mkdir failed"; exit 1; }
(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$DSTI perl -e 'alarm 600; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  rsync -a --ignore-existing --files-from="$C/files.txt" "$C/root/" "box:$HS/") > /dev/null 2>&1 || true  # verified below either way
D=$(box "$DSTI" "$FREEZE" 2>/dev/null | grep '^FREEZE_OK' || true)
[[ $D == "$S" ]] || { echo "destination freeze '$D' != source '$S' (cache kept at $C)"; exit 1; }
MD=$(box "$DSTI" "sha256sum $HS/$MAN | cut -c1-64" 2>/dev/null | grep -E '^[0-9a-f]{64}$' || true)
[[ $MD == $(grep "  ./$MAN\$" "$C/manifest.sha" | cut -c1-64) ]] || { echo "manifest copy differs"; exit 1; }
# 4. delete the laptop cache; record
rm -rf "$C/root"
python3 - "$C/manifest.sha" "$REC" "$SRCI" "$DSTI" "$S" "$D" <<'PY'
import json, sys, datetime
from zoneinfo import ZoneInfo
man, rec, src, dst, s, d = sys.argv[1:]
rows = [l.split(None, 1) for l in open(man).read().splitlines()]
json.dump({"relay": "frozen +311 harness (code and config only)", "from": src, "to": dst, "path": "/home/aswin/phase2/attack_check_311",
           "finished_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T"),
           "source_freeze": s, "destination_freeze": d, "n_files": len(rows),
           "files": [{"path": p, "sha256": h} for h, p in rows]}, open(rec, "w"), indent=1)
PY
echo "harness relay: $N files; source $S; destination $D; manifest identical; laptop cache deleted; record $REC"
