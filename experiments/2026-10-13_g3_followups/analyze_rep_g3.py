"""G3 follow-up (2), POST HOC (PLAN.md here; preliminary until Codex): how much of Seed-OSS-36B's gap fix939's repairs close.
Local, $0. 1% leads; 5% beside. The frozen detector (forecast.json: layers by fold, thresholds at 1%; 5% thresholds by the same
rule from the fit's stored honest scores), G3's draws (2,000, seed 20261012, the 939 cells' sorted main tasks, thresholds fixed).
Per cell: R (second real), S (as simulated), Rep (repaired: fix939's 823 FIX runs scored by g3_score_rep.py; the other 116
cells are the as-simulated runs fix939 left unchanged, so their simulated scores are reused).
  - G_rep = catch(second real) - catch(repaired), points; the paired change against as simulated, y - G_rep =
    catch(repaired) - catch(as simulated); the share closed (y - G_rep) / y (in draws only where y* >= 0.5 points; counted);
  - the same over the 823 repaired-only cells beside it.
Gate first: y and its range at 1% equal g3_y's within 1e-9.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-13_g3_followups/analyze_rep_g3.py --rep-scores results/g3_scores_rep_<stamp>.json
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
G3R = HERE.parent / "2026-10-13_g3_new_reader" / "results"
sys.path.insert(0, str(REPO / "src"))
from probe_shift.attack_eval import threshold  # noqa: E402

B, SEED, LAYER_BY_FOLD, FIT, MIN_DEN = 2000, 20261012, (40, 63, 63, 63, 63), "2026-10-13_111525", 0.5
PINS = {"g3_y_2026-10-13_124203.json": "fcc04de55dbb9bc5df702c3c420814fe70ec736b4cabb8938cbc74e0631b2a69",
        "g3_x_2026-10-13_112514.json": "6fb7f787ebd4853a0f8b9f4e31c4585e602c88cf305e2fef11562420fe5baee2",
        "g3_scores_sim_2026-10-13_124117.json": "d828cc27bb87f07a2763948ec8371fe0ce6b95e3dec7895447a9485184b88798",
        "forecast.json": "4243f2cd38d93d33e5b63ef08bc7e3fb3b387198af43e667c8821966504fff77"}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def caught(S, F, T):
    valid = np.arange(5)[None, :] != F[:, None]
    return np.where(valid, S > T[F], False).sum(1) / valid.sum(1)


def rng(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rep-scores", required=True, type=Path); a = ap.parse_args()
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    for n, p in ((k, G3R / k) for k in PINS if k != "forecast.json"):
        if sha(p) != PINS[n]:
            raise SystemExit(f"{n} is not the pinned file")
    if sha(G3R.parent / "forecast.json") != PINS["forecast.json"]:
        raise SystemExit("not the acknowledged forecast.json")
    fc = json.loads((G3R.parent / "forecast.json").read_text())["frozen"]
    x, y = json.loads((G3R / "g3_x_2026-10-13_112514.json").read_text()), json.loads((G3R / "g3_y_2026-10-13_124203.json").read_text())
    sd = G3R / f"run_scores_{FIT}"
    if sha(sd / "runs.json") != fc["inputs of x sha256"]["runs.json"] or any(sha(sd / f"scores_L{l}.npz") != fc["inputs of x sha256"][f"scores_L{l}.npz"] for l in set(LAYER_BY_FOLD)):
        raise SystemExit("the fit's stored scores differ from the frozen ones")
    runs = json.loads((sd / "runs.json").read_text()); arm = np.array([r["arm"] for r in runs]); fold = np.array([int(r["fold"]) for r in runs])
    N = {l: np.load(sd / f"scores_L{l}.npz")["output_end__mean__task_nested"] for l in set(LAYER_BY_FOLD)}
    e2 = {r["cell"]: i for i, r in enumerate(runs) if r["arm"] == "real-attack-e2"}
    cells = sorted(x["caught, real e2 by cell (R)"]); ei = np.array([e2[c] for c in cells]); F = fold[ei]
    S_e2 = np.stack([N[LAYER_BY_FOLD[fold[i]]][fold[i], :, i] for i in ei])

    def load(rec_p, mode, n):
        rec = json.loads(Path(rec_p).read_text()); z = np.load(Path(rec_p).parent / rec["scores file"])
        if sha(Path(rec_p).parent / rec["scores file"]) != rec["scores sha256"] or rec.get("mode") != mode or rec.get("status") != "scored" or rec["runs"] != n \
                or rec["frozen run"] != FIT or any(rec["probes sha256"][str(l)] != fc["probes sha256 by layer"][str(l)] for l in fc["probes sha256 by layer"]):
            raise SystemExit(f"{Path(rec_p).name}: not a scored {mode} record of the frozen detector")
        kc = {k: c for k, c in rec["keys and cells"]}
        return {kc[k]: (j, z) for j, k in enumerate(z["keys"])}, rec
    sim_by, _ = load(G3R / "g3_scores_sim_2026-10-13_124117.json", "sim", 939)
    rep_by, rrec = load(a.rep_scores, "rep", 823)
    if set(sim_by) != set(cells) or not set(rep_by) <= set(cells):
        raise SystemExit("the scores do not cover the frozen cells")
    pick = lambda by, t, c: by[c][1][f"L{LAYER_BY_FOLD[F[t]]}"][F[t], :, by[c][0]]  # noqa: E731
    S_sim = np.stack([pick(sim_by, t, c) for t, c in enumerate(cells)])
    S_rep = np.stack([pick(rep_by, t, c) if c in rep_by else pick(sim_by, t, c) for t, c in enumerate(cells)])
    in_rep = np.array([c in rep_by for c in cells])
    vs = np.arange(5)[None, :] != F[:, None]
    if not (np.isfinite(S_rep[vs]).all() and np.isfinite(S_sim[vs]).all()):
        raise SystemExit("a non-finite nested score")
    T1 = np.array([[np.nan if v is None else v for v in row] for row in fc["thresholds at 1% [f][v]"]], float)
    T5 = np.full((5, 5), np.nan); hon = arm == "real-honest"
    for f in range(5):
        for v in range(5):
            if v != f:
                T5[f, v] = threshold(N[LAYER_BY_FOLD[f]][f, v][hon & (fold == v)], 0.05)
    e2r = {r["cell"]: r for r in runs if r["arm"] == "real-attack-e2"}
    tasks = sorted({e2r[c]["main_task"] for c in cells}); tix = np.array([tasks.index(e2r[c]["main_task"]) for c in cells])
    g = np.random.default_rng(SEED); W = np.stack([np.bincount(g.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)]).astype(float)
    out = {"POST HOC": True, "status": "preliminary until Codex", "plan": "PLAN.md (2)", "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
           "inputs sha256": PINS | {a.rep_scores.name: sha(a.rep_scores), rrec["scores file"]: rrec["scores sha256"]},
           "the 116 unchanged cells": "fix939 left their as-simulated runs unchanged; their G3 simulated scores are reused (not re-extracted)",
           "cells repaired / reused": [int(in_rep.sum()), int((~in_rep).sum())], "rates": {}}
    for r_, T in ((0.01, T1), (0.05, T5)):
        cR, cS, cP = caught(S_e2, F, T), caught(S_sim, F, T), caught(S_rep, F, T)
        if r_ == 0.01:
            if any(cR[t] != x["caught, real e2 by cell (R)"][c] or cS[t] != y["caught, simulated by cell (S)"][c] for t, c in enumerate(cells)):
                raise SystemExit("gate: a per-cell catch differs from g3_x's R or g3_y's S")
            yy = 100 * (cR - cS).mean(); yb = [100 * (w[tix] @ (cR - cS)) / w[tix].sum() for w in W]
            if abs(yy - y["y = R - S (points)"]) > 1e-9 or max(abs(rng(yb)[0] - y["y 95% (points)"][0]), abs(rng(yb)[1] - y["y 95% (points)"][1])) > 1e-9:
                raise SystemExit("gate: y or its range at 1% differs from g3_y's")
        res = {}
        for name, m in (("all 939 cells", np.ones(939, bool)), ("the 823 repaired-only cells", in_rep)):
            wcs = [w[tix][m] for w in W]
            yv, Gv, dv = (100 * (cR[m] - cS[m]).mean(), 100 * (cR[m] - cP[m]).mean(), 100 * (cP[m] - cS[m]).mean())
            yb = [100 * (w @ (cR[m] - cS[m])) / w.sum() for w in wcs]; Gb = [100 * (w @ (cR[m] - cP[m])) / w.sum() for w in wcs]
            db = [yb_ - Gb_ for yb_, Gb_ in zip(yb, Gb)]
            sh = [d_ / y_ for d_, y_ in zip(db, yb) if y_ >= MIN_DEN]
            res[name] = {"cells": int(m.sum()), "catch: second real, as simulated, repaired (points)": [100 * cR[m].mean(), 100 * cS[m].mean(), 100 * cP[m].mean()],
                         "y = second real - as simulated (points)": [yv, rng(yb)], "G_rep = second real - repaired (points)": [Gv, rng(Gb)],
                         "paired change y - G_rep = repaired - as simulated (points)": [dv, rng(db)],
                         "share of y closed (y - G_rep) / y": [dv / yv if yv else None, rng(sh) if len(sh) >= 1800 else f"not reported: {len(sh)} valid draws"],
                         "draws left out of the share (y* < 0.5 points)": B - len(sh)}
        out["rates"][str(r_)] = res
    p = out["rates"]["0.01"]
    out["prediction 3 (PLAN.md, at 1%)"] = {"G_rep in [+4.5, +11.5]": 4.5 <= p["all 939 cells"]["G_rep = second real - repaired (points)"][0] <= 11.5,
                                            "share closed in [0.15, 0.65]": 0.15 <= (p["all 939 cells"]["share of y closed (y - G_rep) / y"][0] or 0) <= 0.65,
                                            "823-cell G_rep within 1.5 points of the 939's": abs(p["the 823 repaired-only cells"]["G_rep = second real - repaired (points)"][0]
                                                                                                  - p["all 939 cells"]["G_rep = second real - repaired (points)"][0]) <= 1.5}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    txt = json.dumps(out, indent=1, default=lambda o: o.item()) + "\n"  # numpy scalars (the prediction checks) as Python values; serialized before the file opens
    with open(dest := HERE / "results" / f"rep_reading_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(txt)
    print(txt); print(dest)


if __name__ == "__main__":
    main()
