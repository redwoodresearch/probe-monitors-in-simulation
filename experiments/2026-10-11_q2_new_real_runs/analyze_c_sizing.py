"""Part B (zoo C): how many new tasks, two new real runs each, the committed positive-count gate needs for an 80% chance
of a clear answer (PREREG.md @ a12f23b3, committed before this script). DE = max(1.25, 1.2 x the largest Kish over the
four counts), as Amendment 1 applied it; first-two pairing; new runs add matched real positives. For each k, 1,000
simulated batches (default_rng([20261011, k])) pick k of the 274 unpaired simulated-cohort tasks; new runs' counts come
from the task's own overlap800 runs (pessimistic) or from one of today's 163 same-batch pairs (pair donor). Reads only
run_id and credential from Aswin's labels; counts only; $0.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_new_real_runs/analyze_c_sizing.py \\
      <screening_labels_v1.csv> <runs_phase1_honest.csv>
"""
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
FROZEN = {"screening_labels_v1.csv": "edfeb7d43a104113453d5a768964b87b0ee8fc1bd9a6e286276e9b8fd58ae224",
          "runs_phase1_honest.csv": "75381ffed70ddc96ddf59b55eff3c7cf9df77db82063cea009a49bd5d37a83ce"}
UNPAIRED_SHA = "b735a5e06016a59df327b1ad6b7b373209a8a1dfb891853d943cf9eb87414c3f"
GATE_V3 = {"simulated": 1385, "matched real": 2859, "run 1": 546, "run 2": 578}
SEED, BATCHES, GOAL = 20261011, 1000, 0.80
RULES = ("largest Kish (committed, as Amendment 1)", "per-comparison Kish (beside)")
MODELS = ("pessimistic (own task's runs)", "pair donor (today's 163 pairs)")
RUN_RAW, BUF, BATCH, FIT = 1.00, 1.30, 1.11, 3.00  # df's per run; once per batch; C's fits and nulls
erf = np.vectorize(math.erf)


def pass_prob(n_a, n_b, z, de):
    se = np.sqrt(0.25 * de * (1 / n_a + 1 / n_b))
    return np.maximum(0.0, erf((0.10 - z * se) / se / math.sqrt(2)))


def kish(m):
    """Per row of m (batches x tasks): T sum m^2 / (sum m)^2 over the tasks with positives."""
    m = np.atleast_2d(m).astype(float)
    return (m > 0).sum(1) * (m ** 2).sum(1) / m.sum(1) ** 2


def gate(groups, rule):
    k = {g: kish(m) for g, m in groups.items()}
    n = {g: m.sum(1) for g, m in groups.items()}
    if rule == RULES[0]:
        de_rs = de_rr = np.maximum(1.25, 1.2 * np.max(np.stack(list(k.values())), 0))
    else:
        de_rs = np.maximum(1.25, 1.2 * np.maximum(k["simulated"], k["matched real"]))
        de_rr = np.maximum(1.25, 1.2 * np.maximum(k["run 1"], k["run 2"]))
    return pass_prob(n["matched real"], n["simulated"], 2.24, de_rs) * pass_prob(n["run 1"], n["run 2"], 1.96, de_rr), de_rs, de_rr, k


def main(labels_p, runs_p):
    sha = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (labels_p, runs_p)}
    if sha != FROZEN:
        raise SystemExit(f"inputs differ from PREREG's: {sha}")
    me = Path(__file__).resolve()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    pos = defaultdict(int)
    with open(labels_p) as fh:
        for r in csv.DictReader(fh):
            pos[r["run_id"]] += int(r["credential"])
    with open(runs_p) as fh:
        runs = list(csv.DictReader(fh))
    order = next(c for c in runs[0] if c.startswith("real_run_order"))
    batch = lambda rid: "overlap800" if "|overlap800_" in rid else "base200" if "|base200_" in rid else "other"
    sim_tasks = sorted({r["main_task"] for r in runs if r["arm"] != "real-honest"})
    real = [r for r in runs if r["arm"] == "real-honest"]
    first2 = defaultdict(dict)
    for r in real:
        if r[order] in ("1", "2"):
            first2[r["main_task"]][r[order]] = r
    same = sorted(t for t, d in first2.items() if len(d) == 2 and all(batch(x["run_id"]) == "overlap800" for x in d.values()))
    unpaired = sorted(set(sim_tasks) - set(same))
    ti = {t: i for i, t in enumerate(sim_tasks)}; T = len(sim_tasks)
    groups = {g: np.zeros(T) for g in GATE_V3}
    for r in runs:
        if r["arm"] != "real-honest":
            groups["simulated"][ti[r["main_task"]]] += pos[r["run_id"]]
    matched_runs = [r for r in real if batch(r["run_id"]) == "overlap800" and r["main_task"] in ti]
    own = defaultdict(list)
    for r in matched_runs:
        groups["matched real"][ti[r["main_task"]]] += pos[r["run_id"]]; own[r["main_task"]].append(pos[r["run_id"]])
    for t in same:
        groups["run 1"][ti[t]] = pos[first2[t]["1"]["run_id"]]; groups["run 2"][ti[t]] = pos[first2[t]["2"]["run_id"]]
    got = {g: int(m.sum()) for g, m in groups.items()}
    if (got != GATE_V3 or len(same) != 163 or len(matched_runs) != 800 or T != 437
            or hashlib.sha256("\n".join(unpaired).encode()).hexdigest() != UNPAIRED_SHA):
        raise SystemExit(f"does not reproduce the gate's cohort: {got}, {len(same)} pairs, {len(unpaired)} unpaired, {len(matched_runs)} runs, {T} tasks")
    up = np.array([ti[t] for t in unpaired])
    lens = np.array([len(own[t]) for t in unpaired])
    if (lens == 0).any():
        raise SystemExit("an unpaired task has no overlap800 run to resample")
    pad = np.zeros((len(unpaired), lens.max()))
    for i, t in enumerate(unpaired):
        pad[i, :lens[i]] = own[t]
    pairs = np.array([[groups["run 1"][ti[t]], groups["run 2"][ti[t]]] for t in same])
    eff = {g: float(groups[g].sum() ** 2 / (groups[g] ** 2).sum()) for g in ("simulated", "matched real")}
    print(f"frozen unpaired list: {len(unpaired)} tasks, sha256 {UNPAIRED_SHA}")
    out = {"inputs sha256": sha, "git": head, "prereg": "PREREG.md @ a12f23b3", "unpaired sha256": UNPAIRED_SHA, "gate counts today": got,
           "effective tasks today (>= 20 required)": eff, "matched positives per run": float(groups["matched real"].sum() / 800), "today": {}, "sizing": {}}
    for rule in RULES:
        p, de_rs, de_rr, k = gate({g: m[None, :] for g, m in groups.items()}, rule)
        out["today"][rule] = {"P(agree)": float(p[0]), "DE (real-sim, real-real)": [float(de_rs[0]), float(de_rr[0])], "Kish": {g: float(v[0]) for g, v in k.items()}}
    curve = {(mo, ru): [] for mo in MODELS for ru in RULES}
    for kk in range(1, len(unpaired) + 1):
        rng = np.random.default_rng([SEED, kk])
        pick = np.argsort(rng.random((BATCHES, len(unpaired))), axis=1)[:, :kk]  # positions in the 274, without replacement
        u = rng.random((BATCHES, kk, 2)); donor = rng.integers(0, len(same), (BATCHES, kk))
        new = {MODELS[0]: pad[pick[:, :, None], np.floor(u * lens[pick][:, :, None]).astype(int)],
               MODELS[1]: pairs[donor]}
        rows = np.repeat(np.arange(BATCHES)[:, None], kk, 1); cols = up[pick]
        for mo in MODELS:
            g = {name: np.repeat(m[None, :], BATCHES, 0) for name, m in groups.items()}
            g["run 1"][rows, cols] = new[mo][:, :, 0]; g["run 2"][rows, cols] = new[mo][:, :, 1]
            np.add.at(g["matched real"], (rows, cols), new[mo].sum(2))
            for ru in RULES:
                p, de_rs, de_rr, _ = gate(g, ru)
                eff_ok = np.all([m.sum(1) ** 2 / (m ** 2).sum(1) >= 20 for m in (g["simulated"], g["matched real"])], 0)
                v = np.where((p >= GOAL) & eff_ok, p, 0.0)  # the gate and the size rule (>= 20 effective tasks per world)
                curve[(mo, ru)].append({"k": kk, "P(clear)": float(v.mean()), "MC SE": float(v.std() / math.sqrt(BATCHES)),
                                        "P(gate clears 0.80)": float((p >= GOAL).mean()), "median P(agree)": float(np.median(p)),
                                        "median DE (real-sim, real-real)": [float(np.median(de_rs)), float(np.median(de_rr))]})
    for ru in RULES:
        first = {mo: next((c for c in curve[(mo, ru)] if c["P(clear)"] >= GOAL), None) for mo in MODELS}
        common = next((i + 1 for i in range(len(unpaired)) if all(curve[(mo, ru)][i]["P(clear)"] >= GOAL for mo in MODELS)), None)
        if common:  # the smallest k at which both models reach 0.80 at that same k
            kk = common
            size = {"new tasks": kk, "new runs": 2 * kk, "per model at that k": {mo: curve[(mo, ru)][kk - 1] for mo in MODELS},
                    "price raw": 2 * kk * RUN_RAW + BATCH + FIT, "price with 30%": (2 * kk * RUN_RAW + BATCH + FIT) * BUF}
        else:
            size = "too few even with spending: no k up to 274 has both models at 0.80"
        out["sizing"][ru] = {"size": size, "first k reaching 0.80 per model": {mo: (c["k"] if c else None) for mo, c in first.items()},
                             "P(clear) per k (both models)": {mo: [round(c["P(clear)"], 4) for c in curve[(mo, ru)]] for mo in MODELS},
                             "at k = 274": {mo: curve[(mo, ru)][-1] for mo in MODELS},
                             "curve (every 10th k)": {mo: [c for c in curve[(mo, ru)] if c["k"] % 10 == 0] for mo in MODELS}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"c_sizing_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    print("today", json.dumps(out["today"]), "| effective tasks", eff)
    for ru in RULES:
        s = out["sizing"][ru]
        print(ru, "->", s["size"] if isinstance(s["size"], str) else {k: s["size"][k] for k in ("new tasks", "new runs", "price raw", "price with 30%")},
              "| first k per model", s["first k reaching 0.80 per model"])
        for mo in MODELS:
            c = s["at k = 274"][mo]; print(f"   at k=274, {mo}: P(clear) {c['P(clear)']:.3f}, P(gate) {c['P(gate clears 0.80)']:.3f}, median DE {c['median DE (real-sim, real-real)']}")
    print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:3])
