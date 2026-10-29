#!/usr/bin/env bash
# relay_files_box_to_box.sh (0d GO 09:0x, q4c on xbox3): copy CONTENT-ADDRESSED files (name = sha256 of content, e.g.
# ~/phase2/approaches/<sha>.txt) from one box to another at the SAME absolute paths, with the standard guarantees and
# without ever opening, printing or parsing a file's content (only sha256 digests are computed and compared):
# 1. source: each path exists and its sha256 equals the sha in its name -> manifest;  2. source -> laptop cache (rsync over
# SSM ssh), shasum -c on the cache;  3. destination: every path absent; rsync --ignore-existing; sha256sum -c there;
# 4. delete the laptop cache, list what is left (must be 0 files); write the record JSON.
#   relay_files_box_to_box.sh <label> <src instance> <dst instance> <record json> <abs path>...
set -euo pipefail
LABEL=$1 SRC=$2 DST=$3 REC=$4; shift 4
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
H=$SP/simh_launch_X/h; C=$SP/relay_cache_$LABEL
box() { local i=$1; shift; (cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$i perl -e 'alarm 300; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh "$@"); }
[[ ! -e $C && ! -e $REC ]] || { echo "cache $C or record exists"; exit 1; }
for p in "$@"; do [[ $p =~ ^/home/aswin/phase2/[A-Za-z0-9_./-]+/([0-9a-f]{64})\.[a-z]+$ ]] || { echo "not a content-addressed path: $p"; exit 1; }; done
mkdir -p "$C/root"; trap 'rm -rf "$C/root"' EXIT  # Codex r31: the cache goes on every exit path
printf '%s\n' "$@" > "$C/paths.txt"
# 1. manifest on the source (digests only), each name = its content's sha256
box "$SRC" "xargs -d '\n' sha256sum" < "$C/paths.txt" 2> /dev/null | grep -E '^[0-9a-f]{64}  /home/aswin/phase2/' > "$C/manifest.sha" || { echo "manifest failed"; exit 1; }
N=$(wc -l < "$C/manifest.sha" | tr -d ' '); [[ $N == $# ]] || { echo "manifest $N of $#"; exit 1; }
while read -r h p; do b=$(basename "$p"); [[ ${b%.*} == "$h" ]] || { echo "name/content sha mismatch: $p"; exit 1; }; done < "$C/manifest.sha"
awk '{print substr($2, 2)}' "$C/manifest.sha" > "$C/files.txt"
# 2. source -> laptop cache, verified
(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$SRC perl -e 'alarm 300; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  rsync -a --files-from="$C/files.txt" box:/ "$C/root/") > /dev/null 2>&1 || { echo "rsync from source failed"; exit 1; }
OKC=$(cd "$C/root" && awk '{print $1"  ."$2}' ../manifest.sha | shasum -a 256 -c 2>/dev/null | grep -c ': OK$' || true)
[[ $OKC == "$N" ]] || { echo "cache verify $OKC of $N"; rm -rf "$C/root"; exit 1; }
# 3. destination: absent paths, copy without overwriting, verify
ABS=$(box "$DST" "c=0; n=0; while read -r p; do c=\$((c+1)); [[ -e \$p || -L \$p ]] && n=\$((n+1)); done; echo CHECKED=\$c EXISTING=\$n" < "$C/paths.txt" 2> /dev/null | grep '^CHECKED' || true)
[[ $ABS == "CHECKED=$N EXISTING=0" ]] || { echo "destination not empty: '$ABS'"; rm -rf "$C/root"; exit 1; }
(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$DST perl -e 'alarm 300; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  rsync -a --ignore-existing --files-from="$C/files.txt" "$C/root/" box:/) > /dev/null 2>&1 || true  # verified below either way
OKD=$(box "$DST" "sha256sum -c 2>/dev/null | grep -c ': OK\$' || true" < "$C/manifest.sha" 2> /dev/null | tail -1)
# 4. delete the laptop cache (whatever the outcome), list what is left
rm -rf "$C/root"; LEFT=$(find "$C" -type f ! -name manifest.sha ! -name files.txt ! -name paths.txt | wc -l | tr -d ' ')
[[ $OKD == "$N" ]] || { echo "destination verify '$OKD' of $N; laptop files left $LEFT"; exit 1; }
python3 - "$C/manifest.sha" "$REC" "$LABEL" "$SRC" "$DST" "$N" "$OKC" "$OKD" "$LEFT" <<'PY'
import json, sys, datetime
from zoneinfo import ZoneInfo
man, rec, label, src, dst, n, okc, okd, left = sys.argv[1:]
rows = [l.split(None, 1) for l in open(man).read().splitlines()]
json.dump({"relay": label, "from": src, "to": dst, "finished_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T"),
           "guarantees": "content-addressed files only (name = content sha256); digests computed, content never opened, printed or parsed; laptop cache sha256-verified; destination paths absent before; destination sha256-verified; laptop cache deleted",
           "n_files": int(n), "verified_cache": int(okc), "verified_destination": int(okd), "laptop_files_left": int(left),
           "files": [{"path": p, "sha256": h} for h, p in rows]}, open(rec, "w"), indent=1)
PY
echo "relay $LABEL: $N files, cache $OKC OK, destination $OKD OK, laptop files left $LEFT; record $REC"
