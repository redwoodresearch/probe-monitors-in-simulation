"""Part 2's comparison (PLAN.md section 3, "Comparison"): each probe scores the three original runs of the other fold's
cells at the question's full stop (part 1's residuals at its layer); on Modal CPU where the rows are; numbers only.

Runs only if both probes passed G1-G3 (results/part2_gates_*.json) and G4 (results/part2_g4_*.json). Cells need all
three originals kept. As part 1's comparison (analyze_part1.py): the yardstick P(e1 > e2) - 0.5 first; Δc over all
cells (each scored by the probe that never trained on its fold), with an arm-permutation null (one permutation per main
task); one R per probe over its scored cells, each with its null; readings: Δc beyond ±m with the point beyond the null,
either R's lower end above 1.2 with its point above that R's null 97.5th, or "within" (Δc within ±m and both R's within
[1/1.2, 1.2]); else "too few to tell". Ties count one half. 2,000 task bootstrap draws (seed 20261011); nulls 2,000
draws (seed 20261012); m = 0.06.

  PYTHONPATH=src uv run modal deploy experiments/2026-10-11_is_this_a_test/analyze_part2_compare.py
  PYTHONPATH=src uv run python experiments/2026-10-11_is_this_a_test/analyze_part2_compare.py results/part2_gates_<stamp>.json results/part2_g4_<stamp>.json results/ask_ask_<stamp>.json
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-itat_compare_kill-2026-11-10"
VOL = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09"
M, DRAWS, SEED, NULL_SEED, R_BAND = 0.06, 2000, 20261011, 20261012, 1.2

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
fold_of = lambda t: "B" if int(hashlib.sha256(f"20261011|{t}".encode()).hexdigest(), 16) % 2 else "A"


@app.function(image=image, volumes={"/acts": vol}, cpu=4, memory=16384, timeout=1800, retries=0)
def compare(fits: dict, part1: dict, own_sha: str) -> dict:
    import numpy as np

    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != own_sha:
        raise ValueError("the deployed analysis is not the launching commit's: redeploy")
    vol.reload()
    c = lambda a, b: (a > b) + 0.5 * (a == b)
    ci = lambda b: [float(x) for x in np.percentile(b, [2.5, 97.5])]
    p1 = Path("/acts") / "itat" / f"part1_{part1['stamp']}" / "assembled"
    blob = (p1 / "rows.json").read_bytes()
    if hashlib.sha256(blob).hexdigest() != part1["rows_sha256"]:
        raise ValueError("part 1's assembled rows are not the collected ones")
    rows = [r for r in json.loads(blob) if r.get("row") is not None and r["version"] == "original"]
    Y, tasks, probe_of = [], [], []
    for name, fit in sorted(fits.items()):
        pblob = (Path("/acts") / fit["run"] / "fits" / fit["fit_id"] / "probes.json").read_bytes()
        if hashlib.sha256(pblob).hexdigest() != fit["probes_sha256"]:
            raise ValueError(f"{name}: probes.json does not match its fit result")
        pr = json.loads(pblob)["probes"]["last_token"]
        X = np.load(p1 / f"L{pr['layer']}.npy")
        sc = lambda i: float(((X[i].astype(np.float64) - np.array(pr["mu"])) / np.array(pr["sd"])) @ np.array(pr["w"]))
        V = {(r["cell"], r["arm"]): (sc(r["row"]), r["task"]) for r in rows if fold_of(r["task"]) != name}
        for x in sorted({k[0] for k in V}):
            if all((x, a) in V for a in ("e1", "e2", "sim")):
                Y.append([V[(x, a)][0] for a in ("e1", "e2", "sim")])
                tasks.append(V[(x, "e1")][1])
                probe_of.append(name)
    Y, probe_of = np.array(Y), np.array(probe_of)
    ut = sorted(set(tasks))
    ti = np.array([ut.index(t) for t in tasks])
    W = np.random.default_rng(SEED).multinomial(len(ut), np.full(len(ut), 1 / len(ut)), size=DRAWS).astype(float)
    cnt = np.bincount(ti, minlength=len(ut)).astype(float)
    bsum = lambda v: W @ np.bincount(ti, v, len(ut))

    def stats(Z):
        dc = (c(Z[:, 2], Z[:, 0]) + c(Z[:, 2], Z[:, 1])) / 2 - 0.5
        num, den = (np.abs(Z[:, 2] - Z[:, 0]) + np.abs(Z[:, 2] - Z[:, 1])) / 2, np.abs(Z[:, 1] - Z[:, 0])
        return dc, num, den

    dc, num, den = stats(Y)
    rng, nulls = np.random.default_rng(NULL_SEED), []
    for _ in range(DRAWS):  # one arm permutation per main task, applied to all its cells
        Z = np.take_along_axis(Y, rng.permuted(np.tile([0, 1, 2], (len(ut), 1)), axis=1)[ti], 1)
        z = stats(Z)
        nulls.append([z[0].mean()] + [z[1][probe_of == p].sum() / z[2][probe_of == p].sum() for p in sorted(fits)])
    nulls = np.array(nulls)
    if not np.isfinite(nulls).all():
        raise ValueError("an arm-permutation null draw has a zero R denominator")
    dci, yi = ci(bsum(dc) / (W @ cnt)), ci(bsum(c(Y[:, 0], Y[:, 1]) - 0.5) / (W @ cnt))
    Rs = {}
    for k, p in enumerate(sorted(fits)):
        mk = (probe_of == p).astype(float)
        if (den * mk).sum() == 0 or (bsum(den * mk) == 0).any():
            raise ValueError(f"probe {p}: R's denominator is zero")
        Rs[p] = {"cells": int(mk.sum()), "point": float((num * mk).sum() / (den * mk).sum()), "ci": ci(bsum(num * mk) / bsum(den * mk)),
                 "null 97.5th": float(np.percentile(nulls[:, 1 + k], 97.5))}
    readings = []
    if not yi[0] <= 0 <= yi[1]:
        readings = ["two real runs don't agree for this readout; no real-vs-simulated reading"]
    else:
        if dci[0] > M and dc.mean() > np.percentile(nulls[:, 0], 97.5):
            readings.append("the probe scores simulated runs higher than real runs of the same task, beyond the margin")
        if dci[1] < -M and dc.mean() < np.percentile(nulls[:, 0], 2.5):
            readings.append("the probe scores simulated runs lower than real runs of the same task, beyond the margin")
        for p, r in Rs.items():
            if r["ci"][0] > R_BAND and r["point"] > r["null 97.5th"]:
                readings.append(f"probe {p}: simulated runs sit farther from real runs than two real runs from each other, beyond 20%")
        if not readings and -M <= dci[0] and dci[1] <= M and all(1 / R_BAND <= r["ci"][0] and r["ci"][1] <= R_BAND for r in Rs.values()):
            readings.append("within the margins of two real runs (average rank advantage within ±0.06, distances within 20%)")
    return {"cells": len(Y), "tasks": len(ut), "yardstick P(e1>e2)-0.5": {"point": float((c(Y[:, 0], Y[:, 1]) - 0.5).mean()), "ci": yi},
            "dc": {"point": float(dc.mean()), "ci": dci, "null 2.5th/97.5th": ci(nulls[:, 0])}, "R": Rs, "reading": readings or ["too few to tell"]}


if __name__ == "__main__":
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    st = git("status", "--porcelain", "--untracked-files=no")
    if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
        raise SystemExit("commit first: the result must record the commit of the code that made it")
    g, g4, p1 = (json.loads((HERE / f).read_text()) for f in sys.argv[1:4])
    if not (g["both probes pass G1-G3"] and g4["both probes pass G4"]):
        raise SystemExit("part 2's gates did not all pass: no comparison")
    part1 = {"stamp": p1["provenance"]["stamp"], "rows_sha256": p1["assembled"]["rows_sha256"]}
    sp = g4["steering provenance"]  # G4 must be the steering of these probes, on this read, with the gates' directions
    if part1 != g["part1"] or sp["part1_read"] != part1 or sp["fits"] != g["fits"] or \
            sp["directions"] != {k: {"file": v["G4 directions"]["file"], "sha256": v["G4 directions"]["sha256"]} for k, v in g["probes"].items()}:
        raise ValueError("the gates, G4 and part 1 read do not chain")
    with modal.enable_output(), app.run():
        res = compare.remote(g["fits"], part1, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out = {"git": git("rev-parse", "HEAD").stdout.strip(), "inputs": sys.argv[1:4], "m": M, "comparison": res}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"part2_compare_{stamp}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(out, indent=1, default=lambda x: x.item()) + "\n")
    print(json.dumps(out, indent=1, default=lambda x: x.item()), f"\n-> {path}")
