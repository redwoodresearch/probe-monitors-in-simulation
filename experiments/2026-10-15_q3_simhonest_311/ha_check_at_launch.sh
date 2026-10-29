#!/usr/bin/env bash
# ha_check_at_launch.sh (09's per-cell H_A check, committed at launch): for every PINNED cell of the quarter's committed
# generation manifest, the running launch's launch.json references[<key>].sha256 equals the pin and the copy on disk
# hashes to it; the 25f-listed cells (t389, t325, t309) are printed by name when in the quarter. Hashes and keys only.
#   ha_check_at_launch.sh <k> <host instance> <generation manifest json> <out json>
set -euo pipefail
K=$1 INST=$2 MAN=$3 OUT=$4
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad; H=$SP/simh_launch_X/h
RUN=/home/aswin/phase2/runs/2026-10-15_simh_q$K
python3 -I -c 'import json, sys; m = json.load(open(sys.argv[1])); assert m["quarter"] in (int(sys.argv[2]), "q" + sys.argv[2], sys.argv[2]), m["quarter"]
[print(c["task"], c["eval_sha256"]) for c in m["cells"] if c["status"] == "pinned"]' "$MAN" "$K" > "$SP/simh_drive/pins_q$K.txt"
N=$(wc -l < "$SP/simh_drive/pins_q$K.txt" | tr -d ' '); (( N > 0 )) || { echo "no pins"; exit 1; }
(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$INST perl -e 'alarm 300; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh \
  "python3 -I -c 'import json, sys, hashlib
pins = dict(l.split() for l in sys.stdin.read().splitlines())
d = json.load(open(sys.argv[1] + \"/launch.json\")); r = d[\"references\"]
rows = {}
for k, pin in pins.items():
    e = r.get(k) or {}
    disk = hashlib.sha256(open(e[\"copy\"], \"rb\").read()).hexdigest() if e.get(\"copy\") else None
    rows[k] = {\"pin\": pin, \"launch_json_sha256\": e.get(\"sha256\"), \"copy_on_disk_sha256\": disk, \"ok\": e.get(\"sha256\") == pin == disk}
print(\"HA_JSON \" + json.dumps({\"run\": sys.argv[1], \"started\": d.get(\"started\"), \"n_pinned\": len(pins), \"n_references\": len(r), \"n_ok\": sum(v[\"ok\"] for v in rows.values()), \"named_25f\": {k: v for k, v in rows.items() if k.split(\"_\")[0] in (\"t389\", \"t325\", \"t309\")}, \"cells\": rows}))
' $RUN" < "$SP/simh_drive/pins_q$K.txt") 2> /dev/null | grep '^HA_JSON ' | sed 's/^HA_JSON //' > "$OUT.tmp"
rc=0; python3 -I - "$OUT.tmp" "$OUT" "$K" <<'PY' || rc=$?
import json, sys
d = json.load(open(sys.argv[1])); json.dump(d, open(sys.argv[2], "w"), indent=1)
named = {k: (v["ok"], v["pin"][:8]) for k, v in d["named_25f"].items()}
print(f"q{sys.argv[3]} H_A at launch: {d['n_ok']} of {d['n_pinned']} pinned cells ok (references {d['n_references']}, started {d['started']}); 25f-named: {named}")
sys.exit(0 if d["n_ok"] == d["n_pinned"] else 1)
PY
rm -f "$OUT.tmp"; exit $rc
