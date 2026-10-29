#!/usr/bin/env bash
# relay_box_to_box.sh (0d/d9 03:30: ad-hoc relays only with the standard guarantees - a hash manifest, no parsing of
# transcript content, delete after verifying the destination copy): copy a completion launch's honest-only inputs from
# its original box to its placed box, to the SAME absolute paths:
#   the main's launch.json, every references[*].copy of it (each reference's recorded source must be an honest_a run;
#   anything else aborts before a byte moves), and the stopped original's launch.json.
# 1. ON the source box: the path list from launch.json STRUCTURE only (paths and source fields), sha256 of each -> manifest.
# 2. source -> laptop cache (rsync over SSM ssh); sha256 -c on the cache against the manifest.
# 3. destination: every target path must be absent; rsync --ignore-existing; sha256sum -c on the destination: all OK.
# 4. delete the laptop cache, count 0 files left; write the record JSON (paths, sha256s, counts) next to this script's run.
#   relay_box_to_box.sh <label> <src instance> <dst instance> <main dir> <original run dir> <record json>
set -euo pipefail
LABEL=$1 SRC=$2 DST=$3 MAIN=$4 ORIG=$5 REC=$6
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
H=$SP/simh_launch_X/h; C=$SP/relay_cache_$LABEL
box() { local i=$1; shift; (cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$i perl -e 'alarm 300; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh "$@"); }
[[ ! -e $C ]] || { echo "cache $C exists"; exit 1; }
mkdir -p "$C/root"
# 1. the manifest, built on the source box (structure only; honest_a references only)
LIST='import json, sys
main, orig = sys.argv[1], sys.argv[2]
d = json.load(open(main + "/launch.json"))
refs = d["references"]; refs = list(refs.values()) if isinstance(refs, dict) else refs
bad = [e.get("source") for e in refs if "/honest_a/" not in str(e.get("source", ""))]
assert not bad, f"{len(bad)} references not from honest_a runs"
out = [main + "/launch.json"] + [e["copy"] for e in refs] + [orig + "/launch.json"]
assert all(p.startswith("/home/aswin/phase2/") and "\n" not in p for p in out)
print("\n".join(dict.fromkeys(out)))'
box "$SRC" "python3 -c '$LIST' $MAIN $ORIG > /tmp/relay_$LABEL.list && xargs -d '\n' sha256sum < /tmp/relay_$LABEL.list" 2> /dev/null \
  | grep -E '^[0-9a-f]{64}  /home/aswin/phase2/' > "$C/manifest.sha" || { echo "manifest failed"; exit 1; }
N=$(wc -l < "$C/manifest.sha" | tr -d ' ')
(( N > 2 )) || { echo "manifest too short ($N)"; exit 1; }
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
ABS=$(box "$DST" "$RECV; c=0; n=0; while read -r h p; do c=\$((c+1)); if [[ -e \$p || -L \$p ]]; then n=\$((n+1)); fi; done < \$T; rm -f \$T; echo CHECKED=\$c EXISTING=\$n" < "$C/manifest.sha" 2> /dev/null | grep -E '^(CHECKED|MANIFEST_BAD)' || true)
[[ $ABS == "CHECKED=$N EXISTING=0" ]] || { echo "destination check failed or not empty: '$ABS' (expected CHECKED=$N EXISTING=0)"; exit 1; }
(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$DST perl -e 'alarm 600; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  rsync -a --ignore-existing --files-from="$C/files.txt" "$C/root/" box:/) > /dev/null 2>&1 || true  # verified below either way
OKD=$(box "$DST" "$RECV; ok=\$(sha256sum -c \$T 2>/dev/null | grep -c ': OK\$' || true); rm -f \$T; echo VERIFIED=\$ok" < "$C/manifest.sha" 2> /dev/null | grep -E '^(VERIFIED|MANIFEST_BAD)' || true)
[[ $OKD == "VERIFIED=$N" ]] || { echo "destination verify '$OKD' of $N (cache kept at $C)"; exit 1; }
OKD=$N
# 4. delete the laptop cache, then the record
rm -rf "$C/root"
LEFT=$(find "$C" -type f -name '*.eval' | wc -l | tr -d ' ')
python3 - "$C/manifest.sha" "$REC" "$LABEL" "$SRC" "$DST" "$N" "$OKC" "$OKD" "$LEFT" <<'PY'
import json, sys, datetime
from zoneinfo import ZoneInfo
man, rec, label, src, dst, n, okc, okd, left = sys.argv[1:]
rows = [l.split(None, 1) for l in open(man).read().splitlines()]
json.dump({"relay": label, "from": src, "to": dst, "finished_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T"),
           "guarantees": "manifest built on the source from launch.json structure (honest_a references only); no transcript parsed; laptop cache sha256-verified; destination paths absent before; destination sha256-verified; laptop cache deleted",
           "n_files": int(n), "verified_cache": int(okc), "verified_destination": int(okd), "laptop_eval_files_left": int(left),
           "files": [{"path": p, "sha256": h} for h, p in rows]}, open(rec, "w"), indent=1)
PY
echo "relay $LABEL: $N files, cache $OKC OK, destination $OKD OK, laptop .eval left $LEFT; record $REC"
