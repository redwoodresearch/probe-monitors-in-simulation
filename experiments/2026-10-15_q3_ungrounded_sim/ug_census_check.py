"""ug_census_check.py (RULE.md; d9 10-15 ~06:10-06:15): read ONE committed sim error census (the new-mains fork's
results/sim_error_census_2026-10-15_<HHMM>.json on aswin/q3-new-mains-launch) and print the numbers d9's rules use,
recomputed from the per-launch counts. Laptop, read-only, no box access.

  --prestart: d9's start condition 3. PASS iff the census is <= 30 min old, RateLimitError < 5% of the runs that ended
    in its exact last-30-min window (sum of every launch's last30 rl_runs / ended), and < 60 sims running (sum of every
    launch's sim_running, each box's launches counted separately: a name listed on two boxes counts twice, the larger
    figure), and the census is COMPLETE (below). Exit 0 PASS, 1 FAIL.
  --since <ISO start>: d9's pause rule after the ug start. TRIGGER iff any launch (any box, ug's own included) has a
    RateLimitError row whose last attempt ended at or after the start (d9: "RL counts rise"); each such row's 10-min bin
    rate (RateLimitError rows in that bin / that launch's ended runs in that bin) is printed, and >= 5% is the per-bin rule
    (implied by the first: a bin with a RateLimitError and <= 20 ended runs is >= 5%). A trigger is acted on even in an
    incomplete census. Exit 10 TRIGGER, 0 CLEAR, 11 census older than 45 min (cannot see), 13 census incomplete (cannot
    clear), 12 ug's run missing from a census taken >= 20 min after the start.
COMPLETE (Codex r12): all 8 boxes the census polls are present and not null, every launch record is a dict, its counts are
non-negative ints with last30 rl_runs <= ended, the census time is not in the future, and every time is offset-aware.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
UG = "2026-10-15_ug"
BOXES = ("box1", "box2", "box3", "box4", "xbox1", "xbox2", "xbox3", "simsastra")  # sim_error_census_all.sh's list


def aware(t: str) -> datetime:
    d = datetime.fromisoformat(t)
    assert d.tzinfo is not None, f"time without an offset: {t!r}"
    return d


def row_ok(row) -> bool:  # an error row with a class and an offset-aware end time
    try:
        return isinstance(row, dict) and isinstance(row.get("class"), str) and aware(row["end"]) is not None
    except (AssertionError, KeyError, TypeError, ValueError):
        return False


def problems(d: dict, age_min: float) -> list[str]:
    out = [f"box {b} missing or null" for b in BOXES if not isinstance(d["per_box"].get(b), dict)]
    if age_min < -2:
        out.append(f"census time is {-age_min:.0f} min in the future")
    for box, v in d["per_box"].items():
        if not isinstance(v, dict):
            if box not in BOXES:
                out.append(f"box {box} is {type(v).__name__}")
            continue
        for name, r in v.items():
            if not isinstance(r, dict):
                out.append(f"{box}/{name}: record is {type(r).__name__}")
                continue
            n = [r.get("sim_ended"), r.get("sim_running"), (r.get("last30") or {}).get("ended"), (r.get("last30") or {}).get("rl_runs")]
            if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in n) or n[3] > n[2]:
                out.append(f"{box}/{name}: invalid counts {n}")
            rows = r.get("rows")
            if not isinstance(rows, list) or not all(row_ok(x) for x in rows):
                out.append(f"{box}/{name}: an error row without a class or an offset-aware end time")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("census", help="the census JSON (a file; git show it out first)")
    ap.add_argument("--now", default=None, help="ISO time to measure age against (default: now)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prestart", action="store_true")
    g.add_argument("--since", default=None, help="the ug start, ISO with offset")
    a = ap.parse_args()
    d = json.load(open(a.census))
    taken = datetime.strptime(d["census_time_pt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=PT)
    now = aware(a.now) if a.now else datetime.now(PT)
    age_min = (now - taken).total_seconds() / 60
    bad = problems(d, age_min)
    launches = [(box, name, r) for box, v in d["per_box"].items() if isinstance(v, dict) for name, r in v.items()
                if isinstance(r, dict)]
    print(f"census {d['census_time_pt']} PT, {age_min:.1f} min old, {len(launches)} (box, launch) entries")
    print("census's own line:", next((ln for ln in d["rates_text"].splitlines() if ln.startswith("exact last 30")), "?"))
    if a.prestart:
        # sums over the launches whose counts are valid; any other record is in `bad`, so the verdict is FAIL anyway
        # (a launch read error such as {"error": "FileNotFoundError"} has no counts)
        counted = [(box, name, r) for box, name, r in launches if not any(x.startswith(f"{box}/{name}:") for x in bad)]
        ended = sum(r["last30"]["ended"] for _, _, r in counted)
        rl = sum(r["last30"]["rl_runs"] for _, _, r in counted)
        running = sum(r["sim_running"] for _, _, r in counted)
        by_name: dict[str, int] = {}
        for _, name, r in counted:
            by_name[name] = max(by_name.get(name, 0), r["sim_running"])
        pct = 100 * rl / ended if ended else 0.0
        ok = age_min <= 30 and pct < 5 and running < 60 and not bad
        for x in bad:
            print("INCOMPLETE:", x)
        print(f"recomputed: RateLimitError {rl} of {ended} ended in the last 30 min = {pct:.1f}%; sims running {running} "
              f"(per (box, launch); {sum(by_name.values())} if a name on two boxes counts once)")
        print("PRESTART", "PASS" if ok else "FAIL", f"(age <= 30: {age_min <= 30}, RL < 5%: {pct < 5}, running < 60: "
              f"{running < 60}, complete: {not bad})")
        sys.exit(0 if ok else 1)
    start = aware(a.since)
    hits = []
    for box, name, r in launches:
        for row in r.get("rows") if isinstance(r.get("rows"), list) else []:
            if not row_ok(row):
                continue  # counted in `bad`: the census cannot clear
            end = aware(row["end"]).astimezone(PT)
            if row["class"] == "RateLimitError" and end >= start:
                b = f"{end:%H}:{end.minute // 10}0"
                n_rl = (r.get("by_class_and_10min") or {}).get(f"RateLimitError {b}", 0)  # informational
                n_end = (r.get("ended_by_10min") or {}).get(b, 0)
                hits.append((box, name, b, n_rl, n_end))
    seen_ug = any(name == UG for _, name, _ in launches)
    print(f"since {start.astimezone(PT):%H:%M:%S} PT: {len(hits)} RateLimitError runs ended; ug in census: {seen_ug}")
    for box, name, b, n_rl, n_end in sorted(set(hits)):
        print(f"  {box} {name} bin {b}: {n_rl} RateLimitError of {n_end} ended = {100 * n_rl / n_end if n_end else 100:.0f}%")
    if hits:
        print("TRIGGER")
        sys.exit(10)
    if age_min > 45:
        print(f"STALE: newest census is {age_min:.0f} min old")
        sys.exit(11)
    if bad:
        print("INCOMPLETE: " + "; ".join(bad[:5]))
        sys.exit(13)
    if not seen_ug and (taken - start).total_seconds() >= 1200:
        print("UG MISSING from a census taken >= 20 min after the start")
        sys.exit(12)
    print("CLEAR")


if __name__ == "__main__":
    main()
