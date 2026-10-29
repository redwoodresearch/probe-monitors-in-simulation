#!/usr/bin/env bash
# relay_approaches.sh: copy content-addressed approach files ~/phase2/approaches/<sha256>.txt (the texts the frozen
# launcher restores per cell, approaches.py materialize(), which asserts each one's sha256) from any box that holds them
# to the box that runs a +311 simulated-honest quarter. BY HASH ONLY: no file is opened, printed or parsed on the laptop;
# the only content operation anywhere is sha256. Pattern of the reviewed relay_simh.sh: the manifest (sha256 + path) is
# built from the sha list alone; the receiving shell proves the manifest by its own sha256 before using it; every source
# file is a regular non-symlink file hashing to its name; the laptop cache is sha256-verified and holds no symlinks; a
# destination file that already exists must be a regular non-symlink file hashing to its name, else abort before a byte
# moves; rsync --ignore-existing never overwrites; every listed file is re-verified on the destination; the laptop cache
# is deleted (0 files left, listed) and a sha-only record JSON is written.
#   relay_approaches.sh <label> <src instance> <dst instance> <file of sha256s, one per line> <record json>
set -euo pipefail
LABEL=$1 SRC=$2 DST=$3 LIST=$4 REC=$5
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
H=$SP/simh_launch_X/h; C=$SP/appr_cache_$LABEL; A=/home/aswin/phase2/approaches
box() { local i=$1; shift; (cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$i perl -e 'alarm 300; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh "$@"); }
[[ ! -e $C ]] || { echo "cache $C exists"; exit 1; }
[[ -s $LIST ]] && ! grep -Evq '^[0-9a-f]{64}$' "$LIST" || { echo "list is empty or not sha256s only"; exit 1; }
mkdir -p "$C/root"
sort -u "$LIST" | awk -v a="$A" '{print $1"  "a"/"$1".txt"}' > "$C/manifest.sha"
N=$(wc -l < "$C/manifest.sha" | tr -d ' ')
awk '{print substr($2, 2)}' "$C/manifest.sha" > "$C/files.txt"
MSHA=$(shasum -a 256 < "$C/manifest.sha" | cut -c1-64)
RECV="set -euo pipefail; T=\$(mktemp /tmp/appr_${LABEL}_XXXXXX); cat > \$T; [[ \$(sha256sum < \$T | cut -c1-64) == $MSHA ]] || { echo MANIFEST_BAD; exit 1; }"
# counts, per manifest row: regular non-symlink files hashing to their name (OK), and paths that exist at all (EXISTING)
COUNT="c=0; n=0; m=0; while read -r h p; do c=\$((c+1)); if [[ -e \$p || -L \$p ]]; then n=\$((n+1)); if [[ -f \$p && ! -L \$p && \$(sha256sum < \$p | cut -c1-64) == \$h ]]; then m=\$((m+1)); fi; fi; done < \$T; rm -f \$T; echo CHECKED=\$c EXISTING=\$n OK=\$m"
# 1. source: every listed file present, regular, hashing to its name
S=$(box "$SRC" "$RECV; $COUNT" < "$C/manifest.sha" 2> /dev/null | grep -E '^(CHECKED|MANIFEST_BAD)' || true)
[[ $S == "CHECKED=$N EXISTING=$N OK=$N" ]] || { echo "source check '$S' (expected all $N)"; exit 1; }
# 2. source -> laptop cache, verified (hash only; no symlinks)
(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$SRC perl -e 'alarm 600; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  rsync -a --files-from="$C/files.txt" box:/ "$C/root/") > /dev/null 2>&1 || { echo "rsync from source failed"; exit 1; }
OKC=$(cd "$C/root" && awk '{print $1"  ."$2}' ../manifest.sha | shasum -a 256 -c 2>/dev/null | grep -c ': OK$' || true)
NL=$(find "$C/root" -type l | wc -l | tr -d ' '); NF=$(find "$C/root" -type f | wc -l | tr -d ' ')
[[ $OKC == "$N" && $NL == 0 && $NF == "$N" ]] || { echo "cache verify $OKC of $N, files $NF, symlinks $NL"; exit 1; }
# 3. destination: any existing path must be a regular file hashing to its name; copy without overwriting; verify all
D=$(box "$DST" "$RECV; $COUNT" < "$C/manifest.sha" 2> /dev/null | grep -E '^(CHECKED|MANIFEST_BAD)' || true)
[[ $D =~ ^CHECKED=$N\ EXISTING=([0-9]+)\ OK=([0-9]+)$ && ${BASH_REMATCH[1]} == "${BASH_REMATCH[2]}" ]] \
  || { echo "destination check failed or holds differing files: '$D' (expected CHECKED=$N and EXISTING = OK)"; exit 1; }
PRE=${BASH_REMATCH[1]}
box "$DST" "mkdir -p $A" > /dev/null 2>&1 || { echo "mkdir on destination failed"; exit 1; }
(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$DST perl -e 'alarm 600; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  rsync -a --ignore-existing --files-from="$C/files.txt" "$C/root/" box:/) > /dev/null 2>&1 || true  # verified below either way
V=$(box "$DST" "$RECV; $COUNT" < "$C/manifest.sha" 2> /dev/null | grep -E '^(CHECKED|MANIFEST_BAD)' || true)
[[ $V == "CHECKED=$N EXISTING=$N OK=$N" ]] || { echo "destination verify '$V' of $N (cache kept at $C)"; exit 1; }
# 4. delete the laptop cache (listing what is left), then the record
rm -rf "$C/root"
LEFT=$(find "$C" -mindepth 1 ! -name manifest.sha ! -name files.txt | wc -l | tr -d ' ')
echo "laptop cache after delete: $(find "$C" -mindepth 1 | sed "s#^$C/##" | tr '\n' ' ')"
python3 - "$C/manifest.sha" "$REC" "$LABEL" "$SRC" "$DST" "$N" "$OKC" "$PRE" "$LEFT" <<'PY'
import json, sys, datetime
from zoneinfo import ZoneInfo
man, rec, label, src, dst, n, okc, pre, left = sys.argv[1:]
json.dump({"relay": label, "from": src, "to": dst, "finished_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T"),
           "guarantees": "approach files by sha256 only (relay_approaches.sh): never opened, printed or parsed; source regular files hashing to their names; laptop cache sha256-verified, no symlinks; destination paths absent or byte-identical, rsync --ignore-existing; destination sha256-verified; laptop cache deleted",
           "n_files": int(n), "verified_cache": int(okc), "verified_destination": int(n), "identical_already_present": int(pre),
           "laptop_approach_files_left": int(left), "sha256s": [l.split()[0] for l in open(man).read().splitlines()]}, open(rec, "w"), indent=1)
PY
[[ $LEFT == 0 ]] || { echo "laptop cache still holds $LEFT entries ($C)"; exit 1; }  # Codex r7 hardening
echo "approach relay $LABEL: $N files ($PRE identical already present), cache $OKC OK, destination $N OK, laptop files left $LEFT; record $REC"
