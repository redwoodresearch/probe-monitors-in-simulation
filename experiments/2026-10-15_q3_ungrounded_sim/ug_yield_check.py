"""ug_yield_check.py (RULE.md; d9 10-15 ~06:30, 0d's choice (d): ug YIELDS to 4r's follow-up): read the new-mains fork's
4r hold status and say whether ug must pause (phase run) or may resume (phase yield). Laptop, read-only.

Status: the LAST line of results/s4r_hold_status_2026-10-15.jsonl (committed on aswin/q3-new-mains-launch; git show it out
first) and the publisher's laptop copy s4r_hold_status_latest.json (same record, overwritten each poll); the newer by
time_pt is used. Keys (the new-mains fork, 06:2x): time_pt, readable, open_real_4r_main, main_real_not_started,
released, fu_live, fu_sims_started, fu_sims_total.

  run:   PAUSE_4R (exit 20) at the first of open_real_4r_main <= 40, main_real_not_started == 0 (d9's ruling: ~45 min
         before 4r's main ends, so ug's in-flight sims, p90 ~44 min, drain before 4r's release census), released, or
         fu_live. PAUSE_FAILCLOSED (exit 24) if the status is missing, unparseable, readable false, > 15 min old, dated
         > 2 min in the future, or INVALID (d9: fail closed). Else OK (exit 0).
  INVALID (Codex r13): open_real_4r_main and main_real_not_started not non-negative ints (bools rejected); released or
         fu_live not bools; fu_sims_started / fu_sims_total each null or a non-negative int, and when both are ints,
         total > 0 and started <= total.
  yield: RESUME_ALL (exit 21) if fu_sims_total is an int > 0 and fu_sims_started == fu_sims_total; WAVE_IN (exit 26) if
         fu_sims_started >= ceil(0.75 x fu_sims_total) (the caller then requires a census <= 30 min old with < 60
         running and < 5% RateLimitError); else HOLD (exit 0). A missing, stale, future, unreadable or INVALID status is
         HOLD (stay paused).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
OPEN_MAX, WAVE_FRAC, STALE_MIN = 40, 0.75, 15


def when(r) -> datetime | None:
    try:
        return datetime.strptime(r["time_pt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=PT)
    except (KeyError, TypeError, ValueError):
        return None


def nonneg_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def invalid(r: dict) -> str:
    if not (nonneg_int(r.get("open_real_4r_main")) and nonneg_int(r.get("main_real_not_started"))):
        return "open_real_4r_main / main_real_not_started not non-negative ints"
    if not (isinstance(r.get("released"), bool) and isinstance(r.get("fu_live"), bool)):
        return "released / fu_live not bools"
    st, tot = r.get("fu_sims_started"), r.get("fu_sims_total")
    if not all(x is None or nonneg_int(x) for x in (st, tot)):
        return "fu_sims_started / fu_sims_total not null or non-negative ints"
    if st is not None and tot is not None and (tot == 0 or st > tot):
        return f"fu_sims_started {st} / fu_sims_total {tot} out of bounds"
    return ""


def load(jsonl: Path, latest: Path, now: datetime) -> dict | None:
    recs = []
    for p, last_line in ((jsonl, True), (latest, False)):
        try:
            text = p.read_text().strip()
            recs.append(json.loads(text.splitlines()[-1] if last_line else text))
        except (OSError, ValueError, IndexError) as e:
            print(f"status source {p.name}: unreadable ({type(e).__name__})")
    recs = [r for r in recs if isinstance(r, dict) and when(r) is not None]
    future = [r for r in recs if (when(r) - now).total_seconds() > 120]
    for r in future:
        print(f"status record dated {r['time_pt']} is in the future: ignored")
    recs = [r for r in recs if r not in future]
    return max(recs, key=when) if recs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "yield"])
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("latest", type=Path)
    ap.add_argument("--now", default=None)
    a = ap.parse_args()
    now = datetime.fromisoformat(a.now) if a.now else datetime.now(PT)
    assert now.tzinfo is not None, "--now needs an offset"
    r = load(a.jsonl, a.latest, now)
    if r is None:
        why = "no status record (or only future-dated ones)"
    else:
        age = (now - when(r)).total_seconds() / 60
        why = (f"status {r['time_pt']} is {age:.0f} min old" if age > STALE_MIN
               else "status readable: false" if r.get("readable") is not True else
               f"status INVALID: {invalid(r)}" if invalid(r) else "")
    if why:
        print(f"{'PAUSE_FAILCLOSED' if a.phase == 'run' else 'HOLD'}: {why}")
        sys.exit(24 if a.phase == "run" else 0)
    keys = ("open_real_4r_main", "main_real_not_started", "released", "fu_live", "fu_sims_started", "fu_sims_total")
    s = f"{r['time_pt']}: " + ", ".join(f"{k} {r.get(k)}" for k in keys)
    if a.phase == "run":
        o, ns = r["open_real_4r_main"], r["main_real_not_started"]
        hit = [w for w, c in ((f"open_real_4r_main {o} <= {OPEN_MAX}", o <= OPEN_MAX),
                              ("main_real_not_started == 0", ns == 0), ("released", r["released"]),
                              ("fu_live", r["fu_live"])) if c]
        print(f"PAUSE_4R: {'; '.join(hit)} | {s}" if hit else f"OK | {s}")
        sys.exit(20 if hit else 0)
    st, tot = r.get("fu_sims_started"), r.get("fu_sims_total")
    if st is not None and tot is not None:  # validated: non-negative ints, 0 < total, started <= total
        if st == tot:
            print(f"RESUME_ALL: all {tot} follow-up sims started | {s}")
            sys.exit(21)
        if st >= math.ceil(WAVE_FRAC * tot):
            print(f"WAVE_IN: {st} of {tot} started (>= {math.ceil(WAVE_FRAC * tot)}) | {s}")
            sys.exit(26)
    print(f"HOLD | {s}")


if __name__ == "__main__":
    main()
