#!/usr/bin/env bash
# relay_allowlist.sh (d9 ~17:40: ONE reviewed tool for allowlist relays, S_paper493): copy EXACTLY the files of a committed
# allowlist JSON ({"files": [{"path", "sha256"}, ...]}) from one box to others at the SAME absolute paths. DERIVED from
# relay_simh.sh @ 17435cf2e (itself from the interleaved launch's reviewed relay_box_to_box.sh): same guarantees (no file
# opened or parsed; a hash manifest built on the source; laptop cache sha256-verified; destination paths absent or already
# identical before, sha256-verified after; laptop cache deleted; a record JSON). Changed only where marked "allowlist:": the
# file list is the allowlist's, and every source file's sha256 must equal its allowlist sha256 or nothing moves; no
# name may be cells.json, contain "approach" or sit under approaches/ (d9's rule), and every path is under /home/aswin/phase2/.
#   relay_allowlist.sh <label> <src instance> <dst instance> <allowlist json> <record json>
set -euo pipefail
LABEL=$1 SRC=$2 DST=$3 AL=$4 REC=$5  # allowlist: the committed allowlist replaces the main
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
H=$SP/simh_launch_X/h; C=$SP/relay_cache_$LABEL
box() { local i=$1; shift; (cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$i perl -e 'alarm 300; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh "$@"); }
[[ ! -e $C ]] || { echo "cache $C exists"; exit 1; }
mkdir -p "$C/root"
# 1. the manifest, built on the source box (structure only; honest_a references only)
# allowlist (Codex r46): the allowlist must be a COMMITTED, unmodified file of this repo (its commit and sha256 are recorded)
ALDIR=$(cd "$(dirname "$AL")" && pwd) || { echo "allowlist dir"; exit 1; }
AL=$ALDIR/$(basename "$AL")  # Codex r46b: absolute, so git -C resolves it
git -C "$ALDIR" ls-files --error-unmatch "$AL" > /dev/null 2>&1 || { echo "allowlist is not tracked by git"; exit 1; }
git -C "$ALDIR" diff --quiet HEAD -- "$AL" || { echo "allowlist differs from HEAD (commit it first)"; exit 1; }
ALCOMMIT=$(git -C "$ALDIR" log -1 --format=%H -- "$AL"); ALSHA=$(shasum -a 256 < "$AL" | cut -c1-64)
[[ $ALCOMMIT =~ ^[0-9a-f]{40}$ ]] || { echo "allowlist commit unknown"; exit 1; }
python3 - "$AL" > "$C/allow.sha" <<'PY' || { echo "allowlist rejected"; exit 1; }  # explicit checks (no assert: -O safe)
import json, re, sys
def die(msg):
    sys.exit("allowlist: " + msg)
fs = json.load(open(sys.argv[1])).get("files")
if not isinstance(fs, list) or not fs: die("no files")
paths = [f.get("path") for f in fs]
if len(set(paths)) != len(paths): die("duplicate paths")
for f in fs:
    p, h = f.get("path"), f.get("sha256")
    if not (isinstance(p, str) and re.fullmatch(r"/home/aswin/phase2/[A-Za-z0-9_@./:+=-]+", p) and ".." not in p): die(f"bad path {p!r}")
    if p.rsplit("/", 1)[-1] == "cells.json" or "approach" in p.lower(): die(f"forbidden path {p}")
    if not (isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)): die(f"bad sha256 for {p}")
    print(f"{h}  {p}")
PY
awk '{print $2}' "$C/allow.sha" > "$C/allow.list"
box "$SRC" "xargs -d '\n' sha256sum" < "$C/allow.list" 2> /dev/null | grep -E '^[0-9a-f]{64}  /home/aswin/phase2/' > "$C/manifest.sha" || { echo "manifest failed"; exit 1; }
cmp -s <(sort -k2 "$C/manifest.sha") <(sort -k2 "$C/allow.sha") || { echo "source files differ from the allowlist (missing or other sha256): nothing moved"; exit 1; }  # allowlist
N=$(wc -l < "$C/manifest.sha" | tr -d ' ')
(( N >= 1 )) || { echo "empty manifest"; exit 1; }
awk '{print substr($2, 2)}' "$C/manifest.sha" > "$C/files.txt"
# 2. source -> laptop cache, verified
(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$SRC perl -e 'alarm 600; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  rsync -a --files-from="$C/files.txt" box:/ "$C/root/") > /dev/null 2>&1 || { echo "rsync from source failed"; exit 1; }
OKC=$(cd "$C/root" && awk '{print $1"  ."$2}' ../manifest.sha | shasum -a 256 -c 2>/dev/null | grep -c ': OK$' || true)
[[ $OKC == "$N" ]] || { echo "cache verify $OKC of $N"; exit 1; }
# 3. destination: absent paths, copy without overwriting, verify
MSHA=$(shasum -a 256 < "$C/manifest.sha" | cut -c1-64)
# the destination receives the manifest into a fresh temp file and proves it by sha256 before using it (Codex r19b)
RECV="set -euo pipefail; T=\$(mktemp /tmp/relay_${LABEL}_XXXXXX); cat > \$T; [[ \$(sha256sum < \$T | cut -c1-64) == $MSHA ]] || { echo MANIFEST_BAD; exit 1; }"
# simh: an existing destination path is allowed ONLY if it is a regular file byte-identical (sha256) to the manifest entry
# (e.g. an earlier honest-only relay of the same main); rsync --ignore-existing never overwrites it, and step 3's verify
# re-checks every path. Any existing path that differs (or is a symlink/non-file) aborts before a byte moves.
ABS=$(box "$DST" "$RECV; c=0; n=0; m=0; while read -r h p; do c=\$((c+1)); if [[ -e \$p || -L \$p ]]; then n=\$((n+1)); if [[ -f \$p && ! -L \$p && \$(sha256sum < \$p | cut -c1-64) == \$h ]]; then m=\$((m+1)); fi; fi; done < \$T; rm -f \$T; echo CHECKED=\$c EXISTING=\$n IDENTICAL=\$m" < "$C/manifest.sha" 2> /dev/null | grep -E '^(CHECKED|MANIFEST_BAD)' || true)
[[ $ABS =~ ^CHECKED=$N\ EXISTING=([0-9]+)\ IDENTICAL=([0-9]+)$ && ${BASH_REMATCH[1]} == "${BASH_REMATCH[2]}" ]] \
  || { echo "destination check failed or holds differing files: '$ABS' (expected CHECKED=$N and EXISTING = IDENTICAL)"; exit 1; }
PRE=${BASH_REMATCH[1]}  # simh: identical files already there (recorded)
(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$DST perl -e 'alarm 600; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  rsync -a --ignore-existing --files-from="$C/files.txt" "$C/root/" box:/) > /dev/null 2>&1 || true  # verified below either way
OKD=$(box "$DST" "$RECV; ok=\$(sha256sum -c \$T 2>/dev/null | grep -c ': OK\$' || true); rm -f \$T; echo VERIFIED=\$ok" < "$C/manifest.sha" 2> /dev/null | grep -E '^(VERIFIED|MANIFEST_BAD)' || true)
[[ $OKD == "VERIFIED=$N" ]] || { echo "destination verify '$OKD' of $N (cache kept at $C)"; exit 1; }
OKD=$N
# 4. delete the laptop cache, then the record
rm -rf "$C/root"
LEFT=$(find "$C" -type f -name '*.eval' | wc -l | tr -d ' ')
python3 - "$C/manifest.sha" "$REC" "$LABEL" "$SRC" "$DST" "$N" "$OKC" "$OKD" "$LEFT" "$PRE" "$AL" "$ALCOMMIT" "$ALSHA" <<'PY'
import json, sys, datetime
from zoneinfo import ZoneInfo
man, rec, label, src, dst, n, okc, okd, left, pre = sys.argv[1:11]
rows = [l.split(None, 1) for l in open(man).read().splitlines()]
json.dump({"relay": label, "from": src, "to": dst, "allowlist": {"path": sys.argv[11], "commit": sys.argv[12], "sha256": sys.argv[13]}, "finished_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T"),
           "guarantees": "allowlist relay (relay_allowlist.sh): exactly the committed allowlist; source sha256 == allowlist; no transcript parsed; laptop cache sha256-verified; destination paths absent before, or byte-identical (identical_already_present); destination sha256-verified; laptop cache deleted",
           "n_files": int(n), "verified_cache": int(okc), "verified_destination": int(okd), "laptop_eval_files_left": int(left), "identical_already_present": int(pre),
           "files": [{"path": p, "sha256": h} for h, p in rows]}, open(rec, "w"), indent=1)
PY
echo "relay $LABEL: $N files ($PRE identical already present), cache $OKC OK, destination $OKD OK, laptop .eval left $LEFT; record $REC"
