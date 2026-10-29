"""Group and total the per-host simulator-spend reads (sim_spend_read.py output, one <host>.jsonl per box).

  python analyze_sim_spend.py <dir of <host>.jsonl> <out.json>

Launches = launch.json files with a `started` field. A launch relayed to several boxes is counted once: copies are
matched by run-dir name and start time; if their launch.json differs, the copy with the most recent mtime is used and
the difference is flagged. Groups (by run-dir name): scale-up follow-ups (scaleup311_*_fu), IL (_il_), nm360
(newmains_*), other. Totals cover launches that started at or after the cutoff; launches that started before it but
finished after it (or never finished) are listed separately and not totalled. Spend source: launch.json priced by the
harness's run_spend, not OpenAI-billed. Optional guard_lines.tsv in the same dir (host, run dir, the cap guard's
latest "simulator $X" line from spend_now.py, which prices running attempts itself and runs ahead of launch.json for
running launches): reported per launch, and a second total uses max(launch.json figure, guard reading).
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
CUTOFF = datetime.fromisoformat("2026-10-14T17:00:00-07:00")
src, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(line) for f in sorted(src.glob("*.jsonl")) for line in f.read_text().splitlines() if line.strip()]
not_launches = [r for r in rows if "error" in r]
guard = {}
gl = src / "guard_lines.tsv"
for line in (gl.read_text().splitlines() if gl.exists() else []):
    host, run_dir, txt = line.split("\t")
    m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) PT simulator \$([0-9,.]+)", txt)
    assert m, line
    guard[run_dir] = {"time_pt": m.group(1), "usd": float(m.group(2).replace(",", "")), "host": host}
launches = {}
for r in (r for r in rows if "error" not in r):
    k = (Path(r["run_dir"]).name, r["started"])
    launches.setdefault(k, []).append(r)


def pt(s):
    return datetime.fromisoformat(s).astimezone(PT).strftime("%m-%d %H:%M") if s else None


def group(name):
    if "scaleup311" in name:
        return "scale-up follow-ups" if name.endswith("_fu") else "scale-up mains"
    if "_il_" in name:
        return "IL"
    if "newmains" in name:
        return "nm360"
    return "other"


table = []
for (name, started), copies in sorted(launches.items(), key=lambda kv: kv[0][1]):
    best = max(copies, key=lambda c: c["launch_json_mtime"])
    fin = best["finished"]
    t0 = datetime.fromisoformat(started)
    in_window = t0 >= CUTOFF
    if not in_window and fin and datetime.fromisoformat(fin) < CUTOFF:
        continue  # started and finished before the cutoff
    table.append({"group": group(name), "launch": name, "run_dir": best["run_dir"], "host_read": best["host"],
                  "copies_on": sorted(c["host"].split()[0] for c in copies),
                  "copies_differ": len({c["launch_json_sha256"] for c in copies}) > 1,
                  "launch_json_sha256": best["launch_json_sha256"], "started_pt": pt(started),
                  "finished_pt": pt(fin) if fin else "running", "started_after_cutoff": in_window,
                  "n_sim_runs": best["n_sim_runs"], "n_sim_started": best["n_sim_started"],
                  "n_sim_done": best["n_sim_done"], "n_sim_running": best["n_sim_running"],
                  "snapshot_actual_usd": best["snapshot_actual_usd"], "last_snapshot_time_pt": pt(best["last_snapshot_time"]),
                  "attempts_all_usd": best["attempts_all_usd"], "spend_usd": best["spend_usd"],
                  "read_time_pt": pt(best["read_time"]), "cap_guard_reading": guard.get(best["run_dir"]),
                  "spend_usd_with_guard": round(max(best["spend_usd"], guard.get(best["run_dir"], {}).get("usd", 0.0)), 2)})
groups = ["scale-up follow-ups", "IL", "nm360", "other"]
totals = {g: round(sum(t["spend_usd"] for t in table if t["group"] == g and t["started_after_cutoff"]), 2) for g in groups}
counts = {g: sum(t["group"] == g and t["started_after_cutoff"] for t in table) for g in groups}
totals_g = {g: round(sum(t["spend_usd_with_guard"] for t in table if t["group"] == g and t["started_after_cutoff"]), 2) for g in groups}
res = {"cutoff_started_at_or_after": CUTOFF.isoformat(), "spend_source": "launch.json priced (harness run_spend), not OpenAI-billed",
       "launches": table, "group_totals_usd": totals, "group_launch_counts": counts, "grand_total_usd": round(sum(totals.values()), 2),
       "group_totals_with_guard_usd": totals_g, "grand_total_with_guard_usd": round(sum(totals_g.values()), 2),
       "before_cutoff_still_running_or_finished_after": [t for t in table if not t["started_after_cutoff"]],
       "not_launches_skipped": {"count": len(not_launches), "folders": sorted({str(Path(r["run_dir"]).parent) for r in not_launches})}}
assert all(t["group"] in groups or not t["started_after_cutoff"] for t in table)
out.write_text(json.dumps(res, indent=2) + "\n")
for g in groups:
    print(f"== {g}: {counts[g]} launches, ${totals[g]:,.2f} (with guard readings ${totals_g[g]:,.2f})")
    for t in table:
        if t["group"] == g and t["started_after_cutoff"]:
            print(f"  {t['launch']:<40} {t['host_read'].split()[0]:<10} {t['started_pt']} -> {t['finished_pt']:<11} "
                  f"sims {t['n_sim_started']:>2}/{t['n_sim_runs']:<3} ${t['spend_usd']:>8,.2f}"
                  f"{'  guard ' + t['cap_guard_reading']['time_pt'][11:16] + ' $' + format(t['cap_guard_reading']['usd'], ',.2f') if t['cap_guard_reading'] else ''}"
                  f"{'  COPIES DIFFER' if t['copies_differ'] else ''}")
print(f"GRAND TOTAL (started >= 10-14 17:00 PT): ${res['grand_total_usd']:,.2f}; with guard readings ${res['grand_total_with_guard_usd']:,.2f}")
print("Started before 17:00, ran past it (not totalled):")
for t in res["before_cutoff_still_running_or_finished_after"]:
    print(f"  {t['group']:<12} {t['launch']:<40} {t['started_pt']} -> {t['finished_pt']:<11} sims {t['n_sim_started']}/{t['n_sim_runs']} ${t['spend_usd']:,.2f}")
print(f"skipped non-launch launch.json files: {len(not_launches)} (test fixtures / reference indexes)")
