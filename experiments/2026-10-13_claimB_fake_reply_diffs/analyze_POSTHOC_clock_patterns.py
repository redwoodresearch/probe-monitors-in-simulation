"""POST HOC, descriptive (RULE.md Amendment 1 @ 266ecf07, item 4): how the simulator's timestamps differ from real-vs-real
differences at the 91 strict honest sites: format, year and timezone, advancing within a run, repeated values, and offsets
from each reply's own run. Honest runs only, read in full from the local eval caches by build_sites.py's sample() and
actions() (imported unchanged). Counts and patterns only: no timestamp or reply text is written or printed (year and zone
values are). Local, $0, parallel over runs. Writes results/clock_patterns_POSTHOC_<PT stamp>.json.

  PYTHONPATH=src uv run --no-project --python 3.14 --with 'modal>=1.4.3' --with pyyaml python \\
      experiments/2026-10-13_claimB_fake_reply_diffs/analyze_POSTHOC_clock_patterns.py --data <meaning judge data dir>
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "experiments/2026-10-12_claimB_meaning_judge"), str(REPO / "experiments/2026-10-09_fixed_sim")]
import build_sites as B  # noqa: E402  (sample(), actions(): the meaning judge's run reader, unchanged)
import clock  # noqa: E402  (R6's matcher: stamps, now_of, WINDOW, LS_YEAR)
from probe_shift.scrub import scrub  # noqa: E402

PM = REPO / "experiments/2026-10-12_claimB_meaning_judge/results/prefix_matched_2026-10-12_145915.json"
FULL = ("DATE_CMD", "ISO", "HTTP")
DATE_CMD_ONLY = re.compile(r"^date(\s+(-u|--utc|-R|--rfc-email|-I\w*|--iso-8601(=\w+)?))*$")
BUCKETS = ((10, "le_10min"), (120, "10min_2h"), (2880, "2h_48h"), (float("inf"), "gt_48h"))


def run_record(row: dict) -> dict:
    """One run: its outputs and calls in order (actions()), and its started_at in naive UTC (clock's convention)."""
    s = B.sample(row)
    acts = B.actions(s)
    start = datetime.fromisoformat(s["started_at"])
    if start.tzinfo is None:  # a zone-less stamp is UTC (Codex review), never machine-local time
        start = start.replace(tzinfo=timezone.utc)
    start = start.astimezone(timezone.utc).replace(tzinfo=None)
    return {"outputs": [a["output"] for a in acts], "calls": [(a["function"], a["arguments"]) for a in acts], "started": start}


def full(text: str) -> list[dict]:
    return [s for s in clock.stamps(text, clock.LS_YEAR) if s["kind"] in FULL]


def is_date_call(fn: str, args: dict) -> bool:
    cmd = args.get("cmd", args.get("command")) if fn != "text_editor" else None
    return isinstance(cmd, str) and DATE_CMD_ONLY.match(cmd.strip()) is not None


def bucket(minutes: float) -> str:
    return next(name for lim, name in BUCKETS if abs(minutes) <= lim)


def sequences(run: dict) -> dict:
    """(1) clock readings (outputs of a bare `date` call) and (2) each output's latest full-dated stamp within 48 h of the run's now."""
    now = clock.now_of(run["outputs"])
    readings = [s["utc"] for (fn, a), o in zip(run["calls"], run["outputs"]) if is_date_call(fn, a) for s in full(o)]
    outs = []
    if now is not None:
        for o in run["outputs"]:
            near = [s["utc"] for s in full(o) if abs(s["utc"] - now) <= clock.WINDOW]
            if near:
                outs.append(max(near))
    return {"clock_readings": readings, "output_timestamps": outs}


def seq_summary(seqs: list[list]) -> dict:
    two = [q for q in seqs if len(q) >= 2]
    d = lambda q: [(b - a).total_seconds() for a, b in zip(q, q[1:])]  # noqa: E731
    return {"runs": len(seqs), "runs_with_2plus": len(two), "never_step_back": sum(all(x >= 0 for x in d(q)) for q in two),
            "backward_step_over_60s": sum(any(x < -60 for x in d(q)) for q in two), "all_identical": sum(len(set(q)) == 1 for q in two),
            "median_span_min": round(statistics.median([(q[-1] - q[0]).total_seconds() / 60 for q in two]), 1) if two else None,
            "too_few": len(two) < 10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="the meaning judge's data dir (sites.jsonl, sim_new.jsonl)")
    args = ap.parse_args()
    me = Path(__file__).resolve()
    if subprocess.run(["git", "status", "--porcelain", str(me)], capture_output=True, text=True, check=True, cwd=REPO).stdout:
        raise SystemExit(f"commit {me.name} first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    strict = [s for s in map(json.loads, (args.data / "sites.jsonl").read_text().splitlines()) if s["set"] == "strict"]
    new = {r["pair"]: r["result"]["reader_output"] for r in map(json.loads, (args.data / "sim_new.jsonl").read_text().splitlines()) if r["status"] == "ok"}
    pm = json.loads(PM.read_text())["strict_sites"]
    assert len(strict) == len(pm) == 91 and [s["site"] for s in strict] == list(range(91))
    rows = {r["key"]: r for r in json.loads((REPO / "splits/exclusions_v4.json").read_text())["samples"]}
    keys = sorted({p[w] for p in pm for w in ("e1", "e2", "sim")})
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        runs = dict(zip(keys, ex.map(run_record, [rows[k] for k in keys])))

    # the site replies by world, with each reply's own reference run, and integrity checks against the runs
    replies = defaultdict(list)  # world -> [(site, text, identity, internal now, started)]
    for s, p in zip(strict, pm):
        k = s["k"]
        e1, e2, sim = runs[p["e1"]], runs[p["e2"]], runs[p["sim"]]
        assert e1["outputs"][k] == s["e1"] and e2["outputs"][k] == s["e2"], s["site"]
        for world, run, key, text in (("first_real", e1, p["e1"], s["e1"]), ("second_real", e2, p["e2"], s["e2"])):
            replies[world].append((s["site"], text, key, clock.now_of(run["outputs"][:k] + run["outputs"][k + 1:]), run["started"]))
        if s["sim_source"] == "natural":
            assert sim["outputs"][k] == s["sim"], s["site"]
            replies["fake_natural"].append((s["site"], s["sim"], p["sim"], clock.now_of(sim["outputs"][:k] + sim["outputs"][k + 1:]), sim["started"]))
        else:
            replies["fake_new"].append((s["site"], new[s["site"]], f"new:{s['site']}", clock.now_of(e1["outputs"][:k]), e1["started"]))

    out = {"label": "POST HOC, descriptive, preliminary until reviewed", "rule": "RULE.md Amendment 1 @ 266ecf07, item 4", "git": git,
           "input_digest": {f: hashlib.sha256((args.data / f).read_bytes()).hexdigest() for f in ("sites.jsonl", "sim_new.jsonl")},
           "runs_read": len(keys), "worlds": {}, "advance": {}}
    for world, rs in replies.items():
        kinds, iso_forms, years, zones, ph = Counter(), Counter(), Counter(), Counter(), Counter()
        off = {"internal": Counter(), "external": Counter(), "internal_ls": Counter(), "external_ls": Counter()}
        signed = defaultdict(list)
        by_id, sec_shown, sec_zero = defaultdict(set), 0, 0
        site_flag = Counter({k: 0 for k in ("replies", "with_full_dated_stamp", "with_internal_reference", "eligible", "all_within_10min_of_internal")})
        for site, text, ident, now, started in rs:
            st = clock.stamps(text, clock.LS_YEAR)
            kinds.update(s["kind"] for s in st)
            p1 = scrub(text, 1)
            ph.update({x: p1.count(x) for x in ("<date>", "<time>", "<epoch>")})
            fs = [s for s in st if s["kind"] in FULL]
            for s in fs:
                g = s["m"].groups()
                if s["kind"] == "ISO":
                    iso_forms.update([f"sep_{'T' if g[3] == 'T' else 'space'}", f"seconds_{g[6] is not None}", f"fraction_{g[7] is not None}",
                                      f"zone_{'none' if not g[8] else 'Z' if g[8] == 'Z' else 'offset'}"])
                years[s["dt"].year] += 1
                zones[g[6] if s["kind"] == "DATE_CMD" else (g[8] or "none") if s["kind"] == "ISO" else "GMT"] += 1
                if s["kind"] != "ISO" or g[6] is not None:
                    sec_shown += 1
                    sec_zero += s["dt"].second == 0
                by_id[ident].add(s["utc"].replace(second=0, microsecond=0))
            for s in st:
                if s["kind"] == "FAILED":
                    continue
                tag = "" if s["kind"] in FULL else "_ls"
                if now is not None:
                    m = (s["utc"] - now).total_seconds() / 60
                    off["internal" + tag][bucket(m)] += 1
                    signed["internal" + tag].append(m / 60)
                m = (s["utc"] - started).total_seconds() / 60
                off["external" + tag][bucket(m)] += 1
                signed["external" + tag].append(m / 60)
            site_flag["replies"] += 1
            site_flag["with_full_dated_stamp"] += bool(fs)
            site_flag["with_internal_reference"] += now is not None
            if fs and now is not None:
                site_flag["eligible"] += 1
                site_flag["all_within_10min_of_internal"] += all(abs((s["utc"] - now).total_seconds()) <= 600 for s in fs)
        holders = Counter(v for vals in by_id.values() for v in vals)
        out["worlds"][world] = {
            "sites": len(rs), "stamp_kinds": dict(kinds), "iso_forms": dict(iso_forms), "level1_placeholders": dict(ph),
            "years": {str(y): n for y, n in sorted(years.items())}, "zones": dict(zones),
            "offset_buckets": {k: dict(v) for k, v in off.items()},
            "offset_signed_median_hours": {k: round(statistics.median(v), 2) for k, v in signed.items() if v},
            "repeats": {"identities_with_stamp": sum(bool(v) for v in by_id.values()), "distinct_minute_values": len(holders),
                        "values_held_by_2plus_identities": sum(n >= 2 for n in holders.values()),
                        "max_identities_sharing_one_value": max(holders.values(), default=0)},
            "seconds_shown": sec_shown, "seconds_zero": sec_zero, "per_site": dict(site_flag)}

    groups = {"first_real": {p["e1"] for p in pm}, "second_real": {p["e2"] for p in pm},
              "n437_natural_sites": {p["sim"] for s, p in zip(strict, pm) if s["sim_source"] == "natural"}, "n437_all_sites": {p["sim"] for p in pm}}
    seqs = {k: sequences(runs[k]) for k in keys}
    for g, ks in groups.items():
        out["advance"][g] = {kind: seq_summary([seqs[k][kind] for k in sorted(ks)]) for kind in ("clock_readings", "output_timestamps")}
    for w, r in out["worlds"].items():
        print(w, json.dumps({k: r[k] for k in ("sites", "stamp_kinds", "years", "zones", "per_site", "repeats")}))
        print("   offsets", json.dumps(r["offset_buckets"]), "medians h", json.dumps(r["offset_signed_median_hours"]))
    for g, r in out["advance"].items():
        print("advance", g, json.dumps(r))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"clock_patterns_POSTHOC_{stamp}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
