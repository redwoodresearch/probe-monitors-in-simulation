"""census_rl_v3.py (new file, d9 ~11:45 relaying 0d: sel97 counted; v2 stays as reviewed): the +311 and nm360
simulated-honest start gate and rate-limit pause test, read from one committed sim-error census
(origin/aswin/q3-new-mains-launch experiments/2026-10-14_q3_new_mains_launch/results/sim_error_census_*.json) on stdin.
Rule (d9 09:27 PT 10-15, 0d raised the start cap 60 -> 90 for speed):
  gate:  census <= 30 min old and complete (no box without data unless recorded powered off, no read errors, no launch
         without the exact count), < 90 sims running (< 115 once il_q4c has ended, see IL_Q4C below), and NO
         RateLimitError in the latest 10-min bin. The bin labels' edge convention is not relied on: every bin labelled within the 20 min before the census time (or later) must show 0 RateLimitError, which
         covers the latest complete bin and the one in progress under either convention.
  pause: any 10-min bin labelled at or after --since (rounded down to its 10-min bin) with ended > 0 and RateLimitError / ended >= 2%  -> exit 3.
Bins come from the census's own "bins (end time PT): HH:MM rl/ended, ..." line; a label later than the census time
+ 10 min is from the day before (the list wraps midnight).
  python3 census_rl_v3.py gate --start-conc <n> --sel97-dir <dir> < census.json
v3 gate (d9 ~11:45, 0d's numbers): PROJECTED = census simulated running + the starting launch's own concurrency (--start-conc,
its cap / $106 rounded up) + sel97's reserve gap (from <dir>/RESERVED.json until <dir>/DONE.json: max(0, reserve - sel97's
census sim_running); sel97 absent from the census while RESERVED -> the full reserve; a malformed flag file fails the gate)
+, while il_q4c runs, ug's missing concurrency (UG_NORMAL - ug running). Limits: projected < 90 while il_q4c runs (0d 10:27,
kept as the stricter bound), projected <= 110 after (0d ~11:45). Plus 0 RateLimitError in the latest bin, complete data,
the powered-off rule.            exit 0 pass, 1 fail
  python3 census_rl_v3.py pause --since 'YYYY-mm-dd HH:MM' < census.json   exit 0 none, 3 pause
"""
import ast
import datetime as dt
import json
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

POWERED_OFF_KEY = "powered_off_boxes"  # field name agreed with the new-mains fork
# 0d 10:17 (via d9): the start threshold is < 90 while il_q4c runs and < 115 once il_q4c has ended, i.e. its launch.json
# "finished" is seen in a census: a non-empty string "finished" on its per_box entry (the new-mains fork's v2 census
# copies launch.json "finished" verbatim, null when unset), or its box recorded powered
# off with il_q4c among launch_end_times (listed only when every launch there finished). Anything else: < 90.
IL_Q4C, IL_Q4C_BOX = "2026-10-15_il_q4c", "box1"
# 0d 10:27 (via d9): while il_q4c runs, the PROJECTED total must stay < 90: census running + ug's missing concurrency
# (UG_NORMAL - ug's running, counted whenever ug runs below normal, e.g. paused or resuming; all of it if ug is not in the
# census) + the starting quarter's concurrency (START_CONC). After il_q4c has ended: running < 115, no projection.
UG, UG_BOX, UG_NORMAL = "2026-10-15_ug", "xbox3", 36
SEL97 = "2026-10-15_il_sel97_addon"  # d9 ~11:45: on xbox1, /home/aswin/phase2/runs/2026-10-15_il_sel97_addon; never paused
d = json.load(sys.stdin)
mode = sys.argv[1]
t = d["census_time_pt"]
ct = dt.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
lines = d["rates_text"].split("\n")
head = lines[0]
(bins_line,) = [line for line in lines if line.startswith("bins (end time PT):")]
entries = bins_line.removeprefix("bins (end time PT):").strip().split(", ")
bad = [e for e in entries if not re.fullmatch(r"\d\d:\d\d \d+/\d+", e)]
assert not bad, f"malformed bin entries {bad[:3]}"  # Codex r8: never skip an entry
bins = []
for hhmm, rl, ended in (re.fullmatch(r"(\d\d:\d\d) (\d+)/(\d+)", e).groups() for e in entries):
    b = dt.datetime.combine(ct.date(), dt.time.fromisoformat(hhmm))
    if b > ct + dt.timedelta(minutes=10):
        b -= dt.timedelta(days=1)
    bins.append((b, int(rl), int(ended)))
assert bins, "no bins parsed"

if mode == "gate":
    now = dt.datetime.now(ZoneInfo("America/Los_Angeles")).replace(tzinfo=None)
    age = (now - ct).total_seconds() / 60
    running = int(re.search(r"simulated running (\d+)", head).group(1))
    lists = {k: re.search(rf"{k} (\[[^]]*\])", head).group(1) for k in ("boxes without data", "launch read errors", "launches without the exact count")}
    recent = [(b.strftime("%H:%M"), rl, n) for b, rl, n in bins if b >= ct - dt.timedelta(minutes=20)]
    rl_recent = sum(rl for _, rl, _ in recent)
    # d9 10:1x: a box without data counts as complete only if this census records it as powered off with 0 launches
    # (POWERED_OFF_KEY, the new-mains fork's fresh EC2 read at census time): state stopped/terminated, 0 running launches,
    # read within 30 min of the census time. Any other no-data box fails the gate, as before.
    nodata = ast.literal_eval(lists["boxes without data"])
    off = d.get(POWERED_OFF_KEY) or {}
    def off_ok(box):
        e = off.get(box)
        if not isinstance(e, dict) or e.get("state") not in ("stopped", "terminated") or type(e.get("running_launches")) is not int or e["running_launches"] != 0:
            return False
        read = dt.datetime.strptime(e["read_pt"], "%Y-%m-%d %H:%M:%S")
        return abs((ct - read).total_seconds()) <= 1800
    nodata_ok = isinstance(nodata, list) and all(isinstance(b, str) and off_ok(b) for b in nodata)
    q4c_entry = (d["per_box"].get(IL_Q4C_BOX) or {}).get(IL_Q4C)
    q4c_ends = (off.get(IL_Q4C_BOX) or {}).get("launch_end_times") if isinstance(off.get(IL_Q4C_BOX), dict) else None
    q4c_off = isinstance(q4c_ends, dict) and IL_Q4C in q4c_ends and off_ok(IL_Q4C_BOX)
    q4c_fin = q4c_entry.get("finished") if isinstance(q4c_entry, dict) else None
    q4c_ended = (isinstance(q4c_fin, str) and q4c_fin.strip() != "") or q4c_off
    start_conc = int(sys.argv[sys.argv.index("--start-conc") + 1])
    assert 1 <= start_conc <= 60, start_conc
    flags = Path(sys.argv[sys.argv.index("--sel97-dir") + 1])
    sel97_term, sel97_note = 0, "not reserved"
    # Codex r14: every present flag is validated before it can release anything; DONE without RESERVED is inconsistent
    has_res, has_done = (flags / "RESERVED.json").exists(), (flags / "DONE.json").exists()
    if has_res:
        res = json.loads((flags / "RESERVED.json").read_text())
        assert isinstance(res, dict) and type(res.get("reserve")) is int and res["reserve"] > 0, res
    if has_done:
        assert has_res, "DONE.json without RESERVED.json"
        done = json.loads((flags / "DONE.json").read_text())
        assert isinstance(done, dict) and done, "DONE.json is not a non-empty JSON object"
    if has_res and not has_done:
        reserve = res["reserve"]
        hits = [runs[SEL97] for runs in d["per_box"].values() if isinstance(runs, dict) and SEL97 in runs]
        s_run = hits[0].get("sim_running") if len(hits) == 1 and isinstance(hits[0], dict) else None
        if type(s_run) is int:
            sel97_term, sel97_note = max(0, reserve - s_run), f"reserved {reserve}, running {s_run}"
        else:
            sel97_term, sel97_note = reserve, f"reserved {reserve}, not in census: full reserve counted"
    elif has_done:
        sel97_note = "done"
    limit_ok = (lambda x: x < 90) if not q4c_ended else (lambda x: x <= 110)
    limit_txt = "< 90 (il_q4c running)" if not q4c_ended else "<= 110"
    ug_entry = (d["per_box"].get(UG_BOX) or {}).get(UG)
    ug_running = ug_entry["sim_running"] if isinstance(ug_entry, dict) and type(ug_entry.get("sim_running")) is int else 0  # r11: not bool
    ug_term = 0 if q4c_ended else max(0, UG_NORMAL - ug_running)
    projected = running + ug_term + sel97_term + start_conc
    proj_note = f" + ug back {ug_term} + sel97 {sel97_term} ({sel97_note}) + start {start_conc}"
    offrec = {b: {k: off.get(b, {}).get(k) for k in ("instance", "state", "read_pt", "running_launches")} for b in nodata}
    q4c_note = f"ended: finished {q4c_fin}" if q4c_ended and not q4c_off else "ended: box powered off" if q4c_off else "not seen ended"
    print(f"census {t} age {age:.1f} min | running {running} (il_q4c {q4c_note}; projected {projected} = running{proj_note}; limit {limit_txt}) | recent bins (label, RL, ended) {recent} | {lists} | powered off: {offrec}")
    # Codex r8: no future-dated census (2 min clock skew); r8b: the RL and completeness predicates stay in force
    ok = (-2 <= age <= 30 and limit_ok(projected) and rl_recent == 0 and nodata_ok
          and lists["launch read errors"] == "[]" and lists["launches without the exact count"] == "[]")
    sys.exit(0 if ok else 1)
elif mode == "pause":
    since = dt.datetime.strptime(sys.argv[sys.argv.index("--since") + 1], "%Y-%m-%d %H:%M")
    since = since.replace(minute=since.minute - since.minute % 10)  # Codex r8: include the bin holding --since
    hot = [(b.strftime("%F %H:%M"), rl, n) for b, rl, n in bins if b >= since and n > 0 and rl / n >= 0.02]
    print(f"census {t} | bins since {since:%H:%M} at >= 2% RateLimitError: {hot}")
    sys.exit(3 if hot else 0)
sys.exit(f"unknown mode {mode}")
