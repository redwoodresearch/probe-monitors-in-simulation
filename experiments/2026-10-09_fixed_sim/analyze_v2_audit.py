"""Plan amendment 1, v2 audit criteria 1 and 2, by script, on local copies of the v1 and v2 runs:
  1. every shifted simulated run's "now" is dated September 14-21 (2025 or 2026), every donor's "now" is dated
     2026-09-14 to 2026-09-21, and every offset is at most 3 days or between 360 and 370 days;
  2. every simulated output R6 touched in neither run is byte-identical between v1 and v2, and every real row is.
Writes results/v2_audit_checks_<stamp>.json (counts, and up to 5 examples of each violation).

  uv run python experiments/2026-10-09_fixed_sim/analyze_v2_audit.py <v1 dir> <v2 dir>
"""

import gzip
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent


def rows_of(run: Path) -> dict:
    out = {}
    for f in sorted((run / "inputs").glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            out[r["key"]] = (line, r)
    return out


def r6_msgs(run: Path) -> set:
    return {(x["key"], x["msg"]) for f in sorted((run / "records").glob("records_*.jsonl.gz"))
            for x in map(json.loads, gzip.decompress(f.read_bytes()).decode().splitlines()) if x["rule"] == "R6" and x["changed"]}


def run_week(d: datetime, years) -> bool:
    return d.year in years and d.month == 9 and 14 <= d.day <= 21


def main(v1_dir: str, v2_dir: str):
    v1, v2 = Path(v1_dir), Path(v2_dir)
    a, b = rows_of(v1), rows_of(v2)
    assert set(a) == set(b), "the two runs hold different keys"
    touched = r6_msgs(v1) | r6_msgs(v2)
    c, bad = Counter(), {}

    def flag(kind, ex):
        c[kind] += 1
        bad.setdefault(kind, [])
        if len(bad[kind]) < 5:
            bad[kind].append(ex)

    for key, (line2, r2) in sorted(b.items()):
        line1, r1 = a[key]
        if r2["arm"] != "sim-attack":
            c["real_rows"] += 1
            if line1 != line2:
                flag("real_row_differs", key)
            continue
        c["sim_runs"] += 1
        k = r2["fixed_sim"]["clock"]
        if k["now"] is None:
            c["sim_runs_unshifted"] += 1
        else:
            c["sim_runs_shifted"] += 1
            now, donor = datetime.fromisoformat(k["now"]), datetime.fromisoformat(k["donor_now"])
            days = abs(k["offset_s"]) / 86400
            if not run_week(now, (2025, 2026)):
                flag("now_outside_run_week", {"key": key, "now": k["now"]})
            if not run_week(donor, (2026,)):
                flag("donor_now_outside_run_week", {"key": key, "donor_now": k["donor_now"]})
            if not (days <= 3 or 360 <= days <= 370):
                flag("offset_not_allowed", {"key": key, "offset_days": round(days, 2)})
            c["offset_at_most_3_days"] += days <= 3
            c["offset_about_1_year"] += 360 <= days <= 370
        m1, m2 = r1["messages"], r2["messages"]
        if len(m1) != len(m2):
            flag("message_count_differs", key)
            continue
        for i, (x, y) in enumerate(zip(m1, m2)):
            if (key, i) in touched:
                c["sim_messages_touched_by_r6_in_either"] += 1
            else:
                c["sim_messages_compared"] += 1
                if x != y:
                    flag("untouched_message_differs", {"key": key, "msg": i})
    violations = {k: v for k, v in c.items() if k in bad}
    out = {"v1": v1.name, "v2": v2.name, "counts": dict(sorted(c.items())), "violations": violations, "examples": bad,
           "criteria_1_and_2_pass": not violations}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"v2_audit_checks_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
