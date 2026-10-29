#!/usr/bin/env bash
# relay_simh.sh: copy a +311 main's honest-only simulated-honest inputs from its own box to the box that will run its
# simulated-honest launch, to the SAME absolute paths. DERIVED from the interleaved launch's reviewed relay_box_to_box.sh
# @ d35284ffc (experiments/2026-10-15_q3_interleaved_run3_sim2/): same guarantees (a hash manifest built on the source from
# launch.json STRUCTURE only, no transcript parsed, laptop cache sha256-verified, destination paths absent before and
# sha256-verified after, laptop cache deleted, a record JSON). Changed only where marked "simh:": the file list is the
# main's launch.json, every references[*].copy AND every references[*].source (the selected honest_a .eval the copy was
# made from, so the generation manifest can verify source == copy on the host); every copy must be under
# <main>/references/ and every source under <main>/honest_a/, or nothing moves.
#   relay_simh.sh <label> <src instance> <dst instance> <main dir> <record json>
set -euo pipefail
LABEL=$1 SRC=$2 DST=$3 MAIN=$4 REC=$5  # simh: no original run
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
H=$SP/simh_launch_X/h; C=$SP/relay_cache_$LABEL
box() { local i=$1; shift; (cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$i perl -e 'alarm 300; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh "$@"); }
[[ ! -e $C ]] || { echo "cache $C exists"; exit 1; }
mkdir -p "$C/root"
# 1. the manifest, built on the source box (structure only; honest_a references only)
LIST='import json, sys
main = sys.argv[1]  # simh: the main only
d = json.load(open(main + "/launch.json"))
refs = d["references"]; refs = list(refs.values()) if isinstance(refs, dict) else refs
bad = [e.get("source") for e in refs if not str(e.get("source", "")).startswith(main + "/honest_a/")]
assert not bad, f"{len(bad)} references not from this main honest_a runs"
badc = [e.get("copy") for e in refs if not str(e.get("copy", "")).startswith(main + "/references/")]
assert not badc, f"{len(badc)} reference copies outside {main}/references/"
out = [main + "/launch.json"] + [e["copy"] for e in refs] + [e["source"] for e in refs]  # simh: + sources
assert all(p.startswith("/home/aswin/phase2/") and "\n" not in p for p in out)
print("\n".join(dict.fromkeys(out)))'
box "$SRC" "python3 -c '$LIST' $MAIN > /tmp/relay_$LABEL.list && xargs -d '\n' sha256sum < /tmp/relay_$LABEL.list" 2> /dev/null \
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
python3 - "$C/manifest.sha" "$REC" "$LABEL" "$SRC" "$DST" "$N" "$OKC" "$OKD" "$LEFT" "$PRE" <<'PY'
import json, sys, datetime
from zoneinfo import ZoneInfo
man, rec, label, src, dst, n, okc, okd, left, pre = sys.argv[1:]
rows = [l.split(None, 1) for l in open(man).read().splitlines()]
json.dump({"relay": label, "from": src, "to": dst, "finished_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T"),
           "guarantees": "simh relay (relay_simh.sh): manifest built on the source from launch.json structure (honest_a references: copies under <main>/references/, sources under <main>/honest_a/); no transcript parsed; laptop cache sha256-verified; destination paths absent before, or byte-identical (identical_already_present); destination sha256-verified; laptop cache deleted",
           "n_files": int(n), "verified_cache": int(okc), "verified_destination": int(okd), "laptop_eval_files_left": int(left), "identical_already_present": int(pre),
           "files": [{"path": p, "sha256": h} for h, p in rows]}, open(rec, "w"), indent=1)
PY
echo "relay $LABEL: $N files ($PRE identical already present), cache $OKC OK, destination $OKD OK, laptop .eval left $LEFT; record $REC"
