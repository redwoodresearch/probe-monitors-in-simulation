"""Row 11 pilot, step 4 (PLAN.md: the pilot is held out of every reading and only sizes the full stage). It reports:
- usable yield and the named share;
- runs with both groups;
- the pooled within-run, within-group SD of the probe readout;
- the judges' exclusion rates;
- the agreement of the matcher label with `discloses_section`.
No hid-minus-named mean is written.

It then simulates reading A's complete decision rule for the full stage (the other 176 runs, each keeping its real
task) at a true +0.3 SD:
- each run's group counts come from a pilot run's per-sample rates, drawn by multinomial over k samples;
- per-sample readouts = run effect + 0.3 for hid + noise at the pilot's within-run SD;
- A = the mean over runs with both groups of (hid mean − named mean);
- detection = a task-bootstrap 95% interval above 0 (10,000 draws) and a within-run label-shuffle p below 0.05.
The shuffle uses 2,000 draws per simulated dataset (the full stage uses 10,000), 300 datasets, seed 20261011. The full
stage launches only if power ≥ 0.80 at k = 8. Otherwise it reports power at k = 12, 16 and 24. Local, $0.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_hide_same_instruction/analyze_pilot.py <pilot stage folder> \\
      <readout json> <judgments json> <confirm index jsonl>
"""
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
SD0, TRUE_D, SIMS, BOOT, SHUF, SEED, GOAL, MIN_BOTH = 0.5396148830137296, 0.3, 300, 10_000, 2_000, 20261011, 0.80, 30
CONFIRM_SHA = "a4217da0379df98b118584e6015f5fe7eb49d7739d5ccd5f5127937304d2a127"


def power(rates, tasks, sd_w, k, g):
    """rates: per pilot run (p_hid, p_named) per sample; tasks: the full stage's task per run."""
    T = sorted(set(tasks)); ti = np.array([T.index(t) for t in tasks]); hits = 0
    for _ in range(SIMS):
        idx = g.integers(0, len(rates), len(tasks))
        counts = [g.multinomial(k, [rates[i][0], rates[i][1], max(0.0, 1 - rates[i][0] - rates[i][1])])[:2] for i in idx]
        both = np.array([a > 0 and b > 0 for a, b in counts])
        if both.sum() < MIN_BOTH:
            continue
        runs = np.flatnonzero(both)
        vals, labs, rid = [], [], []
        for j, r in enumerate(runs):
            a, b = counts[r]; mu = g.normal()
            vals += list(mu + TRUE_D + g.normal(0, sd_w, a)) + list(mu + g.normal(0, sd_w, b)); labs += [1] * a + [0] * b; rid += [j] * (a + b)
        vals, labs, rid = np.array(vals), np.array(labs), np.array(rid)
        def stat(lab):
            sh = np.bincount(rid, weights=vals * lab) / np.bincount(rid, weights=lab)
            sn = np.bincount(rid, weights=vals * (1 - lab)) / np.bincount(rid, weights=1 - lab)
            return sh - sn
        d = stat(labs); obs = d.mean()
        tr = ti[runs]; Tn = len(T)
        W = np.stack([np.bincount(g.integers(0, Tn, Tn), minlength=Tn) for _ in range(BOOT)]).astype(float)
        num = W @ np.bincount(tr, weights=d, minlength=Tn); den = W @ np.bincount(tr, minlength=Tn)
        boot = num[den > 0] / den[den > 0]
        null = []
        for _ in range(SHUF):  # labels permuted within each run, counts kept
            key = rid * 10.0 + g.random(len(rid)); order = np.argsort(key, kind="stable")
            lab_p = np.empty_like(labs); lab_p[order] = labs[np.argsort(rid * 10.0, kind="stable")]
            null.append(stat(lab_p).mean())
        p = (1 + (np.abs(null) >= abs(obs)).sum()) / (1 + SHUF)
        hits += np.percentile(boot, 2.5) > 0 and p < 0.05
    return hits / SIMS


def main(folder: Path, readout_p: Path, judg_p: Path, confirm_p: Path):
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    ro, ju = json.loads(readout_p.read_text()), json.loads(judg_p.read_text())
    if ro["staged.json sha256"] != hashlib.sha256((folder / "staged.json").read_bytes()).hexdigest():
        raise SystemExit("the readout was made from another stage folder")
    if ju["inputs sha256"] != ro["judge inputs sha256"]:
        raise SystemExit("the judgments were made from other judge inputs")
    if hashlib.sha256(confirm_p.read_bytes()).hexdigest() != CONFIRM_SHA:
        raise SystemExit("confirm index differs")
    staged = {(s["key"], s["seq"]): s for s in json.loads((folder / "staged.json").read_text())}
    scores = {}
    for p_ in (folder / "scores").glob("*.json"):
        r = json.loads(p_.read_text())
        if r["ids_sha256"] != staged[(r["key"], r["seq"])]["ids_sha256"]:
            raise SystemExit(f"{p_.name}: score doesn't match its staged sequence")
        scores[(r["key"], r["seq"])] = r
    rows = ro["rows"]; runs = sorted({r["key"] for r in rows})
    per = defaultdict(lambda: {"hid": [], "named": []}); oom = 0
    for r in rows:
        if not r["usable"]:
            continue
        s = scores.get((r["key"], r["seq"]))
        if s is None:
            raise SystemExit(f"{r['key']} {r['seq']}: a usable sample has no score file (incomplete download?)")
        if s.get("oom"):
            oom += 1; continue
        per[r["key"]][r["label"]].append(float(np.asarray(s["per_token"], float)[r["readout"]].mean()) / SD0)
    resid = [v - np.mean(vs) for k in runs for grp in ("hid", "named") if len(vs := per[k][grp]) >= 2 for v in vs]
    dfree = sum(len(per[k][grp]) - 1 for k in runs for grp in ("hid", "named") if len(per[k][grp]) >= 2)
    sd_w = float(np.sqrt(np.sum(np.square(resid)) / dfree)) if dfree > 0 else None
    n_h = {k: len(per[k]["hid"]) for k in runs}; n_n = {k: len(per[k]["named"]) for k in runs}
    usable = sum(r["usable"] for r in rows); nonempty = [r for r in rows if r["label"] is not None]
    out = {"plan": "PLAN.md (this folder)", "git": head, "readout": readout_p.name, "judgments": judg_p.name,
           "samples": len(rows), "usable samples": usable, "usable yield": usable / len(rows),
           "usable named / hid": [sum(n_n.values()), sum(n_h.values())], "named share among usable": sum(n_n.values()) / max(1, usable),
           "scored usable samples out of memory": oom, "runs": len(runs), "runs with both groups": sum(1 for k in runs if n_h[k] and n_n[k]),
           "within-run, within-group SD of the readout (frozen honest SD units)": sd_w,
           "label agreement, matcher vs discloses_section (non-empty reports)": sum((r["label"] == "named") == r["discloses_section"] for r in nonempty) / max(1, len(nonempty)),
           "judges": {"summary": ju["summary"], "excluded or skipped": ju["excluded or skipped"],
                      "exclusion rate": {j: ju["excluded or skipped"][j] / max(1, sum(1 for r in ju["rows"] if r["judge"] == j)) for j in ("J1", "J2")}}}
    pilot = set(runs)
    full_tasks = [r["main_task"] for r in map(json.loads, confirm_p.open()) if r["key"] not in pilot]
    rates = [(n_h[k] / 8, n_n[k] / 8) for k in runs]
    if sd_w is None or out["runs with both groups"] < 3:
        out["decision"] = "too few to size (pilot has too few runs with both groups)"
    else:
        g = np.random.default_rng(SEED)
        pw = {8: power(rates, full_tasks, sd_w, 8, g)}
        if pw[8] < GOAL:
            for k in (12, 16, 24):
                pw[k] = power(rates, full_tasks, sd_w, k, g)
        out["power"] = {"full-stage runs": len(full_tasks), "full-stage tasks": len(set(full_tasks)), "true D": TRUE_D,
                        "P(A detects at 0.3 SD) by k": pw, "simulations": SIMS, "shuffle draws (approximation)": SHUF}
        need = next((k for k in sorted(pw) if pw[k] >= GOAL), None)
        out["decision"] = ("full stage may launch at k = 8 (power >= 0.80; the orchestrator confirms)" if pw[8] >= GOAL
                           else f"too few at k = 8; k = {need} reaches 0.80 (price in the README)" if need else "too few even at k = 24")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"pilot_sizing_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
