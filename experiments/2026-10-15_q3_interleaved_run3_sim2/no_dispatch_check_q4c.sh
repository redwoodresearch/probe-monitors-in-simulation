#!/usr/bin/env bash
# no_dispatch_check_q4c.sh (09's reversion @ 637b243df, d9 09:1x): read-only evidence that q4c has dispatched NOTHING anywhere
# before box1's start: box1 and xbox3 (run dir, launcher session, run_pilot.py, q4c processes), the laptop (starter processes,
# locks, starter logs' dispatch lines) and Modal (box1's set 6e5f71 load lines; xbox3's be8a1d state). Unreadable evidence
# is reported as UNREADABLE, which BLOCKS the start (pass = false). Writes <out json>.
#   no_dispatch_check_q4c.sh <out json>
set -uo pipefail
OUT=$1; SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
H=$SP/simh_launch_X/h; W=$SP/il_watch; M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal; T=$(mktemp -d)
[[ ! -e $OUT ]] || { echo "$OUT exists"; exit 1; }
BOXCMD='R=/home/aswin/phase2/runs/2026-10-15_il_q''4c; echo "RUNDIR $([[ -e $R ]] && echo PRESENT || echo absent)"; echo "LAUNCHJSON $([[ -e $R/launch.json ]] && echo PRESENT || echo absent)"
tm() { local o rc; o=$(tmux "$@" ls 2>&1); rc=$?; if (( rc == 0 )); then grep -c ilcq4 <<< "$o"; elif grep -qE "no server running|error connecting" <<< "$o"; then echo 0; else echo UNREADABLE; fi; }
echo "SESSION $(tm)"; echo "GUARDSESSION $(tm -L guards)"
P=$(ps -eo args= 2>/dev/null); n=$(wc -l <<< "$P"); if (( n < 20 )); then echo "PSOK UNREADABLE"; else echo "PSOK $n"; fi
echo "RUNPILOT $(grep -c "[r]un_pilot.py" <<< "$P")"; echo "Q4CPROC $(grep -F "il_q""4c" <<< "$P" | grep -vc -F "grep -F")"; echo "END"'
for hm in box1:i-002bbb22c185c0f07 xbox3:i-0f2299f29dd296259; do h=${hm%%:*}; i=${hm##*:}
  (cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$i perl -e 'alarm 60; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh "$BOXCMD" 2>/dev/null) > "$T/$h.txt"; done
{ echo "STARTERS $(pgrep -f 'ilrr_gated_start' | wc -l | tr -d ' ')"; echo "LOCKS $(ls -d $W/*.lock 2>/dev/null | wc -l | tr -d ' ')"
  for k in cq4 cq4x3; do echo "DISPATCH_$k $(grep -cE 'starting il|dispatch returned|] STARTED:' $W/rr_$k.log 2>/dev/null || echo 0)"; done; } > "$T/laptop.txt"
J=$SP/il_servers/h/results/servers_2026-10-15_025316_6e5f71.json
[[ $(jq -r '.servers[].app_id' "$J" 2>/dev/null | grep -c '^ap-') == 6 ]] || echo "SERVERS_JSON UNREADABLE" > "$T/m6e5_bad.txt"  # Codex r32: 6 app IDs or block
for a in $(jq -r '.servers[].app_id' "$J"); do l=$(perl -e 'alarm 25; exec @ARGV' $M app logs $a 2>/dev/null | grep -E 'running=' | tail -3 | tr '\n' '|'); echo "$a ${l:-UNREADABLE}"; done > "$T/m6e5.txt"
perl -e 'alarm 60; exec @ARGV' $M app list --json > "$T/applist.json" 2>/dev/null
python3 - "$T" "$OUT" "$SP/il_watch/apps_be8a1d.txt" <<'PY'
import json, sys, re, datetime
from zoneinfo import ZoneInfo
T, out, be = sys.argv[1:]
def kv(p):
    d = {}
    for l in open(p).read().splitlines():
        if " " in l: k, v = l.split(" ", 1); d[k] = v.strip()
        elif l.strip(): d[l.strip()] = ""  # a bare marker such as END
    return d
res = {"check": "q4c dispatched nothing anywhere (read-only)", "read_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%F %T")}
ok = True
for h in ("box1", "xbox3"):
    d = kv(f"{T}/{h}.txt"); res[h] = d
    good = d.get("END") is not None or "RUNPILOT" in d
    if not all(k in d for k in ("RUNDIR", "LAUNCHJSON", "SESSION", "GUARDSESSION", "PSOK", "RUNPILOT", "Q4CPROC", "END")) or "UNREADABLE" in d.values():
        res[h] = {"UNREADABLE": True, "raw": d}; ok = False; continue  # Codex r32: any failed probe blocks
    ok &= d["RUNDIR"] == "absent" and d["LAUNCHJSON"] == "absent" and d["SESSION"] == "0" and d["GUARDSESSION"] == "0" and d["RUNPILOT"] == "0" and d["Q4CPROC"] == "0"
lp = kv(f"{T}/laptop.txt"); res["laptop"] = lp
ok &= lp.get("STARTERS") == "0" and lp.get("LOCKS") == "0" and lp.get("DISPATCH_cq4") == "0"
res["laptop_note"] = "DISPATCH_cq4x3 counts the 08:55 xbox3 attempt (aborted before launch.json; run dir now ...aborted_0855); it is not a box1 dispatch"
m = {}
for l in open(f"{T}/m6e5.txt").read().splitlines():
    a, rest = l.split(" ", 1); m[a] = rest
    if rest == "UNREADABLE": ok = False
    else:
        gen = re.findall(r"generated=(\d+)", rest); run = re.findall(r"running=(\d+)", rest)
        ok &= len(run) >= 1 and len(gen) >= 1 and all(x == "0" for x in run) and all(x == "0" for x in gen)  # Codex r32: no parseable line blocks
res["modal_6e5f71_last3_load_lines"] = m
import os
if os.path.exists(f"{T}/m6e5_bad.txt") or len(m) != 6: ok = False; res["modal_6e5f71_coverage"] = f"{len(m)} of 6 apps read (blocks unless 6)"
apps = json.load(open(f"{T}/applist.json")); st = {a.get("App ID", a.get("app_id")): a.get("State", a.get("state")) for a in apps}
res["modal_be8a1d_states"] = {a: st.get(a, "absent from list") for a in open(be).read().split()}
ok &= all(s in ("stopped", "absent from list") for s in res["modal_be8a1d_states"].values())
res["pass"] = bool(ok)
json.dump(res, open(out, "w"), indent=1); open(out, "a").write("\n")
print("PASS" if ok else "FAIL (blocks the start)")
sys.exit(0 if ok else 1)
PY
