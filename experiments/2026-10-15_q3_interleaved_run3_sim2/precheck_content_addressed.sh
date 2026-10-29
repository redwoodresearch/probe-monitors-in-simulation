#!/usr/bin/env bash
# precheck_content_addressed.sh (d9 09:0x, after q4c's 08:56 abort): before any placement start, EVERY content-addressed
# file a task file references must exist on the target box and hash to its name. Today that is the approach texts
# (cells[].approach_sha256 and q2_approach_sha256 -> ~/phase2/approaches/<sha>.txt, read by approaches.materialize);
# template_sha256 is a harness template hash, covered by the harness freeze. Digests only; no file is opened or printed.
#   precheck_content_addressed.sh <task file (committed)> <instance> <out json>
set -euo pipefail
TF=$1 INST=$2 OUT=$3
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad; H=$SP/simh_launch_X/h
[[ ! -e $OUT ]] || { echo "$OUT exists"; exit 1; }
L=$(python3 -c 'import json,re,sys; t=json.load(open(sys.argv[1])); v={c[k] for c in t["cells"] for k in ("approach_sha256","q2_approach_sha256") if c.get(k)}; bad=[x for x in v if not (isinstance(x,str) and re.fullmatch("[0-9a-f]{64}",x))]; sys.exit("not a sha256 value: refusing") if bad else print(" ".join(sorted(v)))' "$TF")  # Codex r31b: each JSON value validated before joining
[[ -n $L ]] || { echo "no approach shas in $TF"; exit 1; }
for s in $L; do [[ $s =~ ^[0-9a-f]{64}$ ]] || { echo "not a sha256: refusing"; exit 1; }; done  # Codex r31: validated before the remote command
R=$(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 90; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  "for s in $L; do f=/home/aswin/phase2/approaches/\$s.txt; if [[ -f \$f ]]; then h=\$(sha256sum < \$f | cut -c1-64); [[ \$h == \$s ]] && echo \"OK \$s\" || echo \"BADHASH \$s\"; else echo \"MISSING \$s\"; fi; done" 2>/dev/null | grep -E '^(OK|BADHASH|MISSING) ')
python3 - "$TF" "$INST" "$OUT" "$R" <<'PY'
import json, sys, datetime, hashlib
from zoneinfo import ZoneInfo
tf, inst, out, r = sys.argv[1:]
rows = [l.split() for l in r.splitlines()]
t = json.load(open(tf)); want = {c[k] for c in t["cells"] for k in ("approach_sha256", "q2_approach_sha256") if c.get(k)}
got = {s: st for st, s in rows}
res = {"check": "every content-addressed approach file the task file references exists on the target and hashes to its name (digests only)",
       "task_file": tf.split("experiments/")[-1], "task_file_sha256": hashlib.sha256(open(tf, "rb").read()).hexdigest(), "instance": inst,
       "read_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T"), "n_required": len(want),
       "results": {s: got.get(s, "NOT REPORTED") for s in sorted(want)},
       "template_sha256": "harness template hashes, covered by the harness freeze (not separate files)"}
res["pass"] = all(v == "OK" for v in res["results"].values()) and len(res["results"]) == len(want)
json.dump(res, open(out, "w"), indent=1); open(out, "a").write("\n")
print("PASS" if res["pass"] else "FAIL", {k[:12]: v for k, v in res["results"].items()})
sys.exit(0 if res["pass"] else 1)  # Codex r31b: a FAIL exits nonzero
PY
