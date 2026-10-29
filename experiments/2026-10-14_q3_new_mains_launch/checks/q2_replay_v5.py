"""v5 (10-15 ~17:50, d9): q2 CONFIRMED the replay main pass finished (final live.json written 17:33:46 PT: runs_done 135, in flight 0,
pending 0; record experiments/2026-10-15_q2_replay/results/pass_record_t196_v1_2026-10-15_173829.json @ 89da64af on
aswin/q2-onpolicy-sim, pass_end 17:33:47 PT). A status older than 120 s is no longer a read error ONLY while it is that exact
final status (written 17:33:46 PT, 135 done, 0 in flight, 0 pending): it is then listed "ended (q2 confirmed @ 89da64af)" with
0 running. Any newer write (q2's extension) takes the normal fresh/stale path.
Laptop, read-only (d9 10-15 ~10:20; v4 10-15 ~11:12: the record carries every key a box launch record carries, "rows" (always a
list, empty: the replay status has no per-run rows) and "rate_limit_first_last" (null, or the first and last bin ends with a
RateLimitError), so consumers that require them, such as ug's checker, read it): df/q2's Modal replay as a census source "q2_replay". Reads /replay/status/live.json from the
q2 volume (`modal volume get`, 60 s bound; the file is rewritten every 15 s) and writes <census dir>/q2_replay.json in the per-launch
census format, so the existing rates/stall code counts it like any launch:
  - absent file (the replay hasn't started): {"q2_replay": {... zeros, "stall": {"state": "not started"}}}; never fails completeness;
  - a failed read, unparseable JSON, a missing field, or status_written_pt older than 120 s at read time: {"q2_replay": {"error": ...}}
    (a census read error, so start gates see an incomplete census; nothing is paused);
  - otherwise sim_running = runs_in_flight, sim_ended = runs_done, its 10-minute bins (keyed by bin END, PT) mapped to the census's
    bin labels (bin START, "HH:M0"), errors_by_class {"RateLimitError": total}, last30 from the bins ending within 30 min (approximate,
    flagged), stall from runs_in_flight / runs_pending / paused, plus status_written_pt, paused and rule_sha verbatim.
Ignores /replay/<label>/status.json (the stub dry run's).
  python3 q2_replay_v4.py <census dir> [<local status file instead of the volume, for tests>]"""

import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
VOL, PATH = "probe-shift_aswin-surya_sub-q2-onpolicysim_data_kill-2026-11-14", "/replay/status/live.json"
MODAL = "/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal"
FIELDS = ("status_written_pt", "paused", "runs_in_flight", "runs_done", "runs_pending", "rate_limit_errors_by_10min", "rule_sha")
cdir = Path(sys.argv[1])
now = datetime.now(PT)


def parse_pt(s: str) -> datetime:
    s = s.strip().replace("T", " ")
    s = re.sub(r"\s*(PT|PDT|PST)$", "", s)
    t = datetime.fromisoformat(s)
    return t.astimezone(PT) if t.tzinfo else t.replace(tzinfo=PT)


CONFIRMED_LABEL = "ended (q2 confirmed @ 89da64af; pass_end 2026-10-15 17:33:47 PT)"
CONFIRMED_WRITTEN = datetime(2026, 10, 15, 17, 33, 46, tzinfo=PT)  # the final live.json write q2 confirmed (d9 ~17:50)


def entry() -> dict:
    global now
    if len(sys.argv) > 2:
        raw = Path(sys.argv[2]).read_text()
    else:
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "live.json"
            try:
                p = subprocess.run([MODAL, "volume", "get", "--force", VOL, PATH, str(dst)], capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                return {"error": "read timed out (60 s)"}
            if p.returncode != 0:
                if re.search(r'path "[^"]*" does not exist', p.stdout + p.stderr):
                    return {"sim_ended": 0, "sim_running": 0, "errors_by_class": {}, "rate_limit_first_last": None, "rows": [],
                            "ended_by_10min": {}, "by_class_and_10min": {},
                            "last30": {"now": now.isoformat(), "ended": 0, "rl_runs": 0}, "finished": None, "source": f"modal volume {VOL}{PATH}",
                            "stall": {"state": "not started", "running_all_arms": 0, "pending_all_arms": 0, "last_attempt_activity": None,
                                      "idle_minutes": None, "markers": [], "next_task_fits": None, "ALARM": False}}
                return {"error": f"read failed (exit {p.returncode})"}
            raw = dst.read_text()
    now = datetime.now(PT)  # Codex r51: staleness and windows from the time the file was actually read
    try:
        s = json.loads(raw)
        missing = [f for f in FIELDS if f not in s]
        if missing:
            return {"error": f"status missing fields {missing}"}
        written = parse_pt(str(s["status_written_pt"]))
        age = (now - written).total_seconds()
        confirmed = (written == CONFIRMED_WRITTEN and int(s["runs_done"]) == 135 and int(s["runs_in_flight"]) == 0
                     and int(s["runs_pending"]) == 0)
        if age > 120 and not confirmed:
            return {"error": f"stale status: written {written:%T} PT, {age:.0f} s before the read"}
        ended_bins, rl_bins, e30, r30, rl_ends = {}, {}, 0, 0, []
        for end, v in s["rate_limit_errors_by_10min"].items():
            te = parse_pt(end) if re.search(r"\d{4}-\d\d-\d\d", end) else datetime.combine(now.date(), datetime.strptime(end.strip()[-5:], "%H:%M").time(), PT)
            start = te - timedelta(minutes=10)
            label = f"{start:%H}:{start.minute // 10}0"
            n_end, n_rl = int(v["runs_ended"]), int(v["rate_limit_errors"])
            ended_bins[label] = ended_bins.get(label, 0) + n_end
            if n_rl:
                rl_bins[f"RateLimitError {label}"] = rl_bins.get(f"RateLimitError {label}", 0) + n_rl
                rl_ends.append(te)
            if (now - te).total_seconds() < 1800:
                e30, r30 = e30 + n_end, r30 + n_rl
        inflight, pending, paused = int(s["runs_in_flight"]), int(s["runs_pending"]), bool(s["paused"])
        return {"sim_ended": int(s["runs_done"]), "sim_running": inflight,
                "errors_by_class": {k: n for k, n in {"RateLimitError": sum(int(v["rate_limit_errors"]) for v in s["rate_limit_errors_by_10min"].values())}.items() if n},
                "rate_limit_first_last": [min(rl_ends).isoformat(), max(rl_ends).isoformat()] if rl_ends else None, "rows": [],
                "ended_by_10min": dict(sorted(ended_bins.items())), "by_class_and_10min": dict(sorted(rl_bins.items())),
                "last30": {"now": now.isoformat(), "ended": e30, "rl_runs": r30, "approx_from_10min_bins": True},
                "finished": (CONFIRMED_LABEL if confirmed and age > 120 else None), "source": f"modal volume {VOL}{PATH}",
                "status_written_pt": s["status_written_pt"], "status_age_s": round(age),
                "paused": paused, "rule_sha": s["rule_sha"], "runs_pending": pending,
                "stall": {"state": "stopped" if inflight == 0 and pending == 0 else "running", "running_all_arms": inflight,
                          "pending_all_arms": pending, "last_attempt_activity": s["status_written_pt"], "idle_minutes": None,
                          "markers": ["PAUSED"] if paused else [], "next_task_fits": None, "ALARM": False}}
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        return {"error": f"unreadable status ({type(e).__name__})"}


(cdir / "q2_replay.json").write_text(json.dumps({"q2_replay": entry()}) + "\n")
print((cdir / "q2_replay.json").read_text().strip()[:400])
