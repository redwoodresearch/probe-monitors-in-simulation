"""Read-only census of simulated-run failures (d9, 10-15 ~01:05 PT): for every simulated run of the given launches, its
last attempt's sample errors, classified by the exception the RetryError wraps (RateLimitError, APIConnectionError,
Timeout, ...) or the error class; with attempt count, start/end times, server index; counts per launch, per class and per
10-minute bin of the attempt end (PT). Metadata only (error class names, keys, times).
  python3 sim_error_census.py <launch dir> [<launch dir> ...]"""
import collections, json, re, sys
from datetime import datetime
from zoneinfo import ZoneInfo
PT = ZoneInfo("America/Los_Angeles")
NOW = datetime.now(PT)
out = {}
for run in sys.argv[1:]:
    try:
        d = json.load(open(run + "/launch.json"))
    except (OSError, ValueError) as e:
        out[run.split("/")[-1]] = {"error": type(e).__name__}; continue
    servers = d.get("servers") or []
    per_class, bins, rows, ended = collections.Counter(), collections.Counter(), [], 0
    ended_bins = collections.Counter()  # every ended simulated run by the 10-minute bin of its attempt end (the rate denominator)
    e30 = rl30 = 0  # Codex r26 item 3: exact window, runs whose last attempt ended within 1,800 s of NOW, and those with a RateLimitError
    running = sum(1 for r in d.get("runs") or [] if not r.get("real") and r.get("state") == "running")
    for r in d.get("runs") or []:
        if r.get("real") or not r.get("attempts"):
            continue
        a = r["attempts"][-1]
        if not a.get("end"):
            continue
        ended += 1
        e_end = datetime.fromisoformat(a["end"]).astimezone(PT)
        ended_bins[f"{e_end:%H}:{e_end.minute // 10}0"] += 1
        in30 = (NOW - e_end).total_seconds() <= 1800
        e30 += in30
        errs = [str(e) for e in ((a.get("outcome") or {}).get("sample_errors") or [])]
        if not errs:
            continue
        for e in errs:
            m = re.search(r"raised (\w+)", e)
            cls = (m.group(1) if m else e.split("(")[0])[:40]
            per_class[cls] += 1
            end = datetime.fromisoformat(a["end"]).astimezone(PT)
            bins[f"{cls} {end:%H}:{end.minute // 10}0"] += 1
            rows.append({"key": r["key"], "class": cls, "attempts": len(r["attempts"]), "start": a.get("start"), "end": a["end"],
                         "server": servers.index(r["server"]) if r.get("server") in servers else None})
        rl30 += in30 and any(x["key"] == r["key"] and x["class"] == "RateLimitError" for x in rows)
    rl = sorted(x["end"] for x in rows if x["class"] == "RateLimitError")
    # stall alarm (d9 / 0d, 10-15 ~02:50): running == 0 while runs are ready/waiting, with no attempt activity (latest
    # attempt start or end, any arm) for more than 15 min; a launch with "finished" set is "stopped", never an alarm
    states = collections.Counter(r.get("state") for r in d.get("runs") or [])
    stamps = [x for r in d.get("runs") or [] for a in r.get("attempts") or [] for x in (a.get("start"), a.get("end")) if x]
    # Codex r24: a launch with no attempt yet is timed from its own start; compare unrounded, round only for display
    last = max(datetime.fromisoformat(x) for x in stamps) if stamps else (datetime.fromisoformat(d["started"]) if d.get("started") else None)
    idle_s = (datetime.now(PT) - last.astimezone(PT)).total_seconds() if last else None
    idle_min = round(idle_s / 60, 1) if idle_s is not None else None
    pending = states.get("ready", 0) + states.get("waiting", 0)
    running_all = states.get("running", 0)
    snap = (d.get("spend") or [{}])[-1]
    markers = sorted(m for m in ("GATE_PAUSE.txt", "PAUSE_SIMS", "SIM_CAP_PAUSE.txt", "SIM_BLIND_PAUSE.txt", "SIM_HARD_STOP.txt", "GATE1_ALERT.txt", "DEADLINE_STOP.txt")
                     if __import__("os").path.exists(run + "/" + m)) + [f"PAUSED:{p}" for p in sorted(__import__("glob").glob(run + "/PAUSED_*.txt"))]
    stall = {"state": "stopped" if d.get("finished") else "running", "running_all_arms": running_all, "pending_all_arms": pending,
             "last_attempt_activity": str(last) if last else None, "idle_minutes": idle_min, "markers": markers,
             "next_task_fits": snap.get("next_task_fits"),
             "ALARM": (not d.get("finished")) and running_all == 0 and pending > 0 and idle_s is not None and idle_s > 900}
    out[run.split("/")[-1]] = {"sim_ended": ended, "sim_running": running, "errors_by_class": dict(per_class),
                               "rate_limit_first_last": [rl[0], rl[-1]] if rl else None, "ended_by_10min": dict(sorted(ended_bins.items())), "by_class_and_10min": dict(sorted(bins.items())), "stall": stall, "rows": rows,
                               "last30": {"now": NOW.isoformat(), "ended": e30, "rl_runs": rl30}}
print(json.dumps(out))
