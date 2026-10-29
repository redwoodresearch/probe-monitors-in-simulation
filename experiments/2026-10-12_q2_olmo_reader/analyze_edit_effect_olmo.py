"""Olmo part 2's reading (PLAN.md, part 2; committed before any part-2 number is read). Local, $0.

Part 1's rules unchanged (analyze_reply_break_olmo.py: each test fold's layer, #36's thresholds, the catch, 2,000
main-task resamples with thresholds redrawn, seed 20261011), with one more simulated arm: "fixed", the 823 re-rendered
runs' scores from run_score_fix_olmo.py and the other 116 runs' as-simulated scores. Reported as Aswin asked, real -
simulated: real e2 - fixed, real e2 - as simulated, real e2 - e1, and the edit effect fixed - as simulated (paired),
each with its task-clustered 95% interval; the swap null for fixed vs e2 (per main task, 10,000, seed 20261012);
plants of +-10 on the fixed arm; fix939's bar on fixed - e2; Qwen3-32B #36's same rows from fix939, read by key.

Gates: part 1's as-simulated numbers (sim - e2 and e2 - e1, both rates, and the layer per fold) are reproduced from
reply_break_olmo_2026-10-12_113245.json; the scores file matches its record's sha256 and holds 823 re-rendered runs.

  uv run --no-project --with numpy python experiments/2026-10-12_q2_olmo_reader/analyze_edit_effect_olmo.py \\
      <results/run_scores_2026-10-12_110549> <results/fix_scores_olmo_<stamp>.json>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze_reply_break_olmo import B, BAR, LAYERS, N_NULL, PLANT, RATES, SEED, SEED_NULL, TOL, auc, thresh  # noqa: E402

PART1 = HERE / "results" / "reply_break_olmo_2026-10-12_113245.json"
FIX939 = HERE.parent / "2026-10-08_q3_swap_cue" / "results" / "fix939_POSTHOC_2026-10-11_020519.json"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def main(d: Path, rec_p: Path):
    me = Path(__file__).resolve()
    for f in (me, HERE / "PLAN.md", HERE / "analyze_reply_break_olmo.py"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    for f in (PART1, FIX939):  # both committed: their bytes at HEAD are the pins (sha256 recorded below)
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} is not committed as is")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    rec = json.loads(rec_p.read_text()); npz = rec_p.parent / rec["scores file"]
    if sha(npz) != rec["scores sha256"] or rec["with re-rendered scores"] != 823 or rec["run"] != "2026-10-12_110549":
        raise SystemExit("not the recorded fixed scores of run 2026-10-12_110549, with 823 re-rendered runs")
    part1 = json.loads(PART1.read_text())
    # the fixed scores' run order is runs.json's: the scorer, part 1 and this reading must have read the same files
    here = {"runs.json": sha(d / "runs.json")} | {f"scores_L{l}.npz": sha(d / f"scores_L{l}.npz") for l in LAYERS}
    if any(rec["inputs sha256"].get(k) != v for k, v in here.items()) or any(part1["inputs sha256"].get(k) != v for k, v in here.items()):
        raise SystemExit("the scores dir differs from the one the scorer or part 1 read")
    runs = json.loads((d / "runs.json").read_text())
    arm = np.array([r["arm"] for r in runs]); fold = np.array([r["fold"] for r in runs]); task = np.array([r["main_task"] for r in runs])
    cell = np.array([r["cell"] for r in runs])
    N = {l: np.load(d / f"scores_L{l}.npz")["output_end__mean__task_nested"] for l in LAYERS}
    hon, e1 = arm == "real-honest", arm == "real-attack-e1"
    layer = {}
    for f in range(5):  # part 1's rule
        fin = lambda x: x[np.isfinite(x)]  # noqa: E731
        layer[f] = max(LAYERS, key=lambda l: np.mean([auc(fin(N[l][f, v][e1 & (fold == v)]), fin(N[l][f, v][hon & (fold == v)])) for v in range(5) if v != f]))
    if {str(k): v for k, v in layer.items()} != {str(k): v for k, v in part1["layer per test fold"].items()}:
        raise SystemExit(f"layers {layer} differ from part 1's")
    S_all = np.stack([N[layer[fold[i]]][fold[i], :, i] for i in range(len(runs))])
    Z = np.load(npz); ridx, has = Z["run_index"], Z["has_fix"]
    S_fix = S_all.copy()
    for j, i in enumerate(ridx):
        if has[j]:
            S_fix[i] = Z[f"L{layer[fold[i]]}"][fold[i], :, j]
    sim_i = {c: i for i, c in enumerate(cell) if arm[i] == "sim-attack"}; e2_i = {c: i for i, c in enumerate(cell) if arm[i] == "real-attack-e2"}
    cells = sorted(set(sim_i) & set(e2_i))
    ok = lambda S, i: np.isfinite(np.delete(S[i], fold[i])).all()  # noqa: E731
    if len(cells) != 939 or not all(ok(S_all, sim_i[c]) and ok(S_all, e2_i[c]) and ok(S_fix, sim_i[c]) for c in cells):
        raise SystemExit("expected 939 cells with every nested score")
    si, ei = np.array([sim_i[c] for c in cells]), np.array([e2_i[c] for c in cells])
    e1i = np.array([i for i in range(len(runs)) if e1[i] and ok(S_all, i)])
    tasks = np.unique(task); tix = np.searchsorted(tasks, task); ctix = tix[si]
    changed = np.isin(si, ridx[has])

    def thresholds(alpha, w):
        T = np.full((5, 5), np.nan)
        for f in range(5):
            for v in range(5):
                if v != f:
                    m = hon & (fold == v) & np.isfinite(N[layer[f]][f, v])
                    T[f, v] = thresh(N[layer[f]][f, v][m], w[tix[m]].astype(float), alpha)
        return T

    def catch(idx, T, S):
        F = fold[idx]; valid = np.arange(5)[None, :] != F[:, None]
        return np.where(valid, S[idx] > T[F], False).sum(1) / valid.sum(1)
    ones = np.ones(len(tasks))
    g = np.random.default_rng(SEED); draws = [np.bincount(g.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)]
    gn = np.random.default_rng(SEED_NULL)
    q = json.loads(FIX939.read_text())["readouts"]["output_end"]
    out = {"plan": "PLAN.md part 2 (this folder)", "git": head, "scores dir": d.name, "fixed scores": rec_p.name,
           "inputs sha256": {rec_p.name: sha(rec_p), npz.name: sha(npz), PART1.name: sha(PART1), FIX939.name: sha(FIX939), "runs.json": sha(d / "runs.json")},
           "layer per test fold": layer, "cells": len(cells), "tasks": int(len(np.unique(task[si]))),
           "simulated runs with re-rendered scores": int(changed.sum()), "unchanged (as-simulated scores)": int((~changed).sum()),
           "orientation": "real - simulated, as Aswin asked (part 1 reported simulated - real)", "rates": {}}
    for alpha in RATES:
        T0 = thresholds(alpha, ones)
        cf, cs, ce, c1 = catch(si, T0, S_fix), catch(si, T0, S_all), catch(ei, T0, S_all), catch(e1i, T0, S_all)
        pt = {"real e2 - fixed": ce.mean() - cf.mean(), "real e2 - as simulated": ce.mean() - cs.mean(),
              "real e2 - e1 (two real runs; e1 trained the detector)": ce.mean() - c1.mean(), "edit effect: fixed - as simulated (paired)": cf.mean() - cs.mean()}
        p1 = part1["rates"][str(alpha)]
        if abs(100 * (cs.mean() - ce.mean()) - p1["sim - e2 (points)"]) > 1e-9 or abs(100 * pt["real e2 - e1 (two real runs; e1 trained the detector)"] - p1["beside: real e2 - e1 (points; e1 trained the detector)"][0]) > 1e-9:
            raise SystemExit(f"{alpha}: part 1's numbers are not reproduced")
        bs = {k: [] for k in pt}
        for w in draws:
            T = thresholds(alpha, w); ww = w[ctix].astype(float); wv = w[tix[e1i]].astype(float)
            mf, ms, me = [(ww * catch(si, T, S)).sum() / ww.sum() for S in (S_fix, S_all)] + [(ww * catch(ei, T, S_all)).sum() / ww.sum()]
            m1 = (wv * catch(e1i, T, S_all)).sum() / wv.sum()
            for k, v in zip(pt, (me - mf, me - ms, me - m1, mf - ms)):
                bs[k].append(v)
        rows = {k: [100 * float(v), [100 * float(x) for x in np.percentile(bs[k], [2.5, 97.5])]] for k, v in pt.items()}
        gap = 100 * (cf.mean() - ce.mean())  # fixed - e2, the bar's orientation
        null = []
        for _ in range(N_NULL):
            flip = (gn.random(len(tasks)) < 0.5)[ctix]
            null.append(100 * (np.where(flip, ce, cf).mean() - np.where(flip, cf, ce).mean()))
        p = float((1 + np.sum(np.abs(null) >= abs(gap))) / (1 + N_NULL))
        n = len(si); k = round(PLANT * n); plants = {}
        gp = np.random.default_rng([SEED, int(alpha * 100)])
        for sign, want, new in (("-10", 1.0, -np.inf), ("+10", 0.0, np.inf)):
            elig = np.flatnonzero(cf == want)
            if len(elig) < k:
                plants[sign] = {"constructible": False, "eligible": int(len(elig)), "needed": k}; continue
            pick = gp.choice(elig, k, replace=False); Sp = S_fix.copy(); Sp[si[pick]] = new
            gpl = 100 * (catch(si, T0, Sp).mean() - ce.mean()); shift = gpl - gap
            pb = []
            for w in draws:
                T = thresholds(alpha, w); ww = w[ctix].astype(float)
                pb.append(100 * ((ww * catch(si, T, Sp)).sum() - (ww * catch(ei, T, S_all)).sum()) / ww.sum())
            plo, phi = np.percentile(pb, [2.5, 97.5])
            plants[sign] = {"constructible": True, "k": k, "planted fixed - e2": gpl, "shift": shift, "planted 95%": [float(plo), float(phi)],
                            "detected": bool(abs(shift / 100 - np.sign(new) * PLANT) <= TOL and not (plo <= gap <= phi))}
        lo, hi = -rows["real e2 - fixed"][1][1], -rows["real e2 - fixed"][1][0]  # fixed - e2's 95% range
        asis = float(100 * (cs.mean() - ce.mean())); gap = float(gap); outside = bool(not (lo <= asis <= hi))
        if abs(gap) <= BAR and outside:
            reading = "explains it"
        elif outside and abs(gap) < abs(asis):
            reading = "partial"
        else:
            reading = "doesn't explain it" + (" (the gap grew)" if abs(gap) > abs(asis) else "")
        qr = q[str(alpha)]
        flip_ci = lambda o: [-100 * o["observed"], [-100 * o["ci95"][1], -100 * o["ci95"][0]]]  # noqa: E731
        same = lambda o: [100 * o["observed"], [100 * x for x in o["ci95"]]]  # noqa: E731
        out["rates"][str(alpha)] = {
            "catch": {"fixed": float(cf.mean()), "as simulated": float(cs.mean()), "e2": float(ce.mean()), "e1": float(c1.mean())},
            "rows (points; 95% task bootstrap, thresholds redrawn)": rows,
            "swap null, fixed vs e2 (two-sided p)": p, "plants on the fixed arm": plants,
            "fix939's bar (on fixed - e2)": {"fixed - e2": gap, "95%": [lo, hi], "as simulated - e2": asis, "point within +-5": bool(abs(gap) <= BAR),
                                            "as-simulated gap outside the fixed range": bool(outside), "whole range within +-5": bool(-BAR <= lo and hi <= BAR)},
            "reading": reading if alpha == 0.05 else reading + " (1%, reported)",
            "beside: Qwen3-32B #36, fix939 (points, real - simulated; its sim - e2 rows sign-flipped)": {
                "real e2 - fixed": flip_ci(qr["fixed_sim_minus_e2 (primary)"]), "real e2 - as simulated": flip_ci(qr["as_is_sim_minus_e2"]),
                "real e2 - e1": same(qr["real_e2_minus_e1 (e1 was in training)"]), "edit effect: fixed - as simulated": same(qr["shift_fixed_minus_as_is"])}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"edit_effect_olmo_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
