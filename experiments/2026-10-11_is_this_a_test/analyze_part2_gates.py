"""Part 2's gates G1-G3 (PLAN.md section 3, amendments 2-3) for both probes, and the steering directions G4 uses; on Modal
CPU where the rows are. Only numbers and identifiers return.

Each probe was fit by run_fit.py on its own fold's pairs (spawn_fit_itat.py); score = ((x - mu) / sd) . w at its layer.
- G1, held-out wordings: pair accuracy (the simulated side scores higher) at least 0.80, its lower end above 0.70, and
  its point above the 95th percentile of a task-grouped flip null (one coin per task flips its pairs' correctness).
- G2, neutral lines: with d_h the held-out pairs' mean difference (simulated minus production) and d_n the neutral
  pairs' (UTC minus GMT), on the same quarter-3 tasks: d_n - d_h / 3 has its upper end below 0 and d_n + d_h / 3 its
  lower end above 0 (joint draws), i.e. d_n lies within ±1/3 of d_h.
- G3, at the question's full stop, on part 1's planted units in the other fold's cells (part 1's residuals at the
  probe's layer; a unit is a cell's e1 or e2 with its original and all three lines kept, as in analyze_part1.py): pair
  accuracy (sim line above production line) at least 0.70; the symmetric rank change's lower end above m (cells with
  both units); and on e1 and on e2, the neutral line's shift against the original within ±1/3 of the sim line's (joint).
  If part 1's eligibility rule stops its reading, these gates are not computed.
- Steering directions for G4, raw-space unit vectors: the probe's w / sd; standardized logistic regressions (C = 0.1,
  no intercept, scaler sd + 1e-6 as run_fit.py, converged or refused) on its "control" pairs and on its "neutral_fit"
  pairs, at its layer; a seeded random direction. Frozen step rule: every direction is applied at the probe's step,
  ±2σ or ±4σ with σ the SD of the probe's "train" rows projected on the probe's direction, so every perturbation has
  the same norm (the random floor's "same norm"); each direction's own SD on those rows is reported beside. Saved to
  the Volume under this script's hash, never overwritten.
Intervals: 2,000 main-task bootstrap draws (seed 20261011); nulls: 2,000 draws (seed 20261012); m = 0.06. Ties count
one half.

  PYTHONPATH=src uv run modal deploy experiments/2026-10-11_is_this_a_test/analyze_part2_gates.py
  PYTHONPATH=src uv run python experiments/2026-10-11_is_this_a_test/analyze_part2_gates.py results/fit_<A>.json results/fit_<B>.json results/ask_ask_<stamp>.json
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-itat_gates_kill-2026-11-10"
VOL = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09"
M, DRAWS, SEED, NULL_SEED, RANDOM_SEED, C = 0.06, 2000, 20261011, 20261012, 20261014, 0.1

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2")
fold_of = lambda t: "B" if int(hashlib.sha256(f"20261011|{t}".encode()).hexdigest(), 16) % 2 else "A"


@app.function(image=image, volumes={"/acts": vol}, cpu=4, memory=16384, timeout=1800, retries=0)
def gates(fits: dict, part1: dict, own_sha: str) -> dict:
    import warnings

    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    warnings.simplefilter("error", ConvergenceWarning)  # an unconverged control fit fails loudly

    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != own_sha:
        raise ValueError("the deployed analysis is not the launching commit's: redeploy")
    vol.reload()
    c = lambda a, b: (a > b) + 0.5 * (a == b)
    ci = lambda b: [float(x) for x in np.percentile(b, [2.5, 97.5])]

    def boot(tasks, vals):  # main-task bootstrap means of per-unit values (units x k): draws x k
        ut = sorted(set(tasks))
        ti = np.array([ut.index(t) for t in tasks])
        vals = np.asarray(vals, float).reshape(len(tasks), -1)
        W = np.random.default_rng(SEED).multinomial(len(ut), np.full(len(ut), 1 / len(ut)), size=DRAWS).astype(float)
        sums = np.stack([np.bincount(ti, vals[:, k], len(ut)) for k in range(vals.shape[1])], 1)
        return (W @ sums) / (W @ np.bincount(ti, minlength=len(ut)).astype(float))[:, None]

    p1 = Path("/acts") / "itat" / f"part1_{part1['stamp']}" / "assembled"
    blob = (p1 / "rows.json").read_bytes()
    if hashlib.sha256(blob).hexdigest() != part1["rows_sha256"]:
        raise ValueError("part 1's assembled rows are not the collected ones")
    all1 = json.loads(blob)
    rows1 = [r for r in all1 if r.get("row") is not None]
    cells_all = sorted({r["cell"] for r in all1})  # part 1's eligibility rule (analyze_part1.py), in integers
    kept = {(r["cell"], r["arm"]) for r in rows1 if r["version"] == "original"}
    ok = [x for x in cells_all if all((x, a) in kept for a in ("e1", "e2", "sim"))]
    drops = [sum((x, a) not in kept for x in cells_all) for a in ("e1", "e2", "sim")]
    if 100 * (len(cells_all) - len(ok)) > 15 * len(cells_all) or 100 * (max(drops) - min(drops)) > 5 * len(cells_all):
        return {"stopped": "part 1's eligibility rule stops the reading; no gate computed"}
    out = {}
    for name, fit in fits.items():
        d = Path("/acts") / fit["run"]
        pblob = (d / "fits" / fit["fit_id"] / "probes.json").read_bytes()
        if hashlib.sha256(pblob).hexdigest() != fit["probes_sha256"]:
            raise ValueError(f"{name}: probes.json does not match its fit result")
        pr = json.loads(pblob)["probes"]["last_token"]
        L, w, mu, sd = pr["layer"], np.array(pr["w"]), np.array(pr["mu"]), np.array(pr["sd"])
        score = lambda X: ((X.astype(np.float64) - mu) / sd) @ w
        meta = json.loads((d / "meta.json").read_text())
        X = np.load(d / f"L{L}.npy")
        s = score(X)
        by = lambda role: {(m["pair"], m["label"]): i for i, m in enumerate(meta) if m["role"] == role}
        res = {"layer": L}
        # G1: held-out pairs
        h = by("heldout")
        hp = sorted({p for p, _ in h})
        diff_h = np.array([s[h[(p, 1)]] - s[h[(p, 0)]] for p in hp])
        task_h = [p.split("|")[1] for p in hp]
        corr = c(diff_h, 0.0)
        acc_b = boot(task_h, corr)[:, 0]
        ut = sorted(set(task_h))
        ti = np.array([ut.index(t) for t in task_h])
        flips = np.random.default_rng(NULL_SEED).integers(0, 2, size=(DRAWS, len(ut)))[:, ti]  # one coin per task
        null = np.where(flips == 1, 1 - corr, corr).mean(1)
        res["G1"] = {"pairs": len(hp), "accuracy": float(corr.mean()), "ci": ci(acc_b), "null 95th": float(np.percentile(null, 95)),
                     "by mechanism": {mm: float(corr[[p.split("|")[4] == mm for p in hp]].mean()) for mm in sorted({p.split("|")[4] for p in hp})}}
        res["G1"]["pass"] = corr.mean() >= 0.80 and ci(acc_b)[0] > 0.70 and corr.mean() > np.percentile(null, 95)
        # G2: neutral pairs on the quarter-3 tasks, jointly with the held-out difference
        n = by("neutral")
        n_by_task = {p.split("|")[1]: s[n[(p, 1)]] - s[n[(p, 0)]] for p, _ in n}
        tasks = sorted(set(task_h) & set(n_by_task))
        dh = np.array([diff_h[[t == u for u in task_h]].mean() for t in tasks])
        dn = np.array([n_by_task[t] for t in tasks])
        b = boot(tasks, np.stack([dn - dh / 3, dn + dh / 3], 1))
        res["G2"] = {"tasks": len(tasks), "d_h": float(dh.mean()), "d_n": float(dn.mean()), "d_n - d_h/3": ci(b[:, 0]), "d_n + d_h/3": ci(b[:, 1])}
        res["G2"]["pass"] = ci(b[:, 0])[1] < 0 and ci(b[:, 1])[0] > 0
        # G3: part 1's planted versions in the other fold's cells, at the question's full stop
        X1 = np.load(p1 / f"L{L}.npy")
        V = {(r["cell"], r["arm"], r["version"]): float(score(X1[r["row"]][None])[0]) for r in rows1 if fold_of(r["task"]) != name}
        task1 = {r["cell"]: r["task"] for r in rows1}
        cells = sorted({k[0] for k in V if k[2] == "sim_line"})
        units = [(x, a) for x in cells for a in ("e1", "e2") if all((x, a, v) in V for v in ("original", "sim_line", "prod_line", "neutral_line"))]
        if not units:
            raise ValueError(f"{name}: no complete planted unit in the other fold")
        pa = units
        acc3 = float(np.mean([c(V[(x, a, "sim_line")], V[(x, a, "prod_line")]) for x, a in pa]))
        rc_cells = [x for x in cells if (x, "e1") in units and (x, "e2") in units]
        o = lambda x, a, v="original": V[(x, a, v)]
        rc = np.array([(c(o(x, "e1", "sim_line"), o(x, "e2")) - c(o(x, "e1"), o(x, "e2")) + c(o(x, "e2", "sim_line"), o(x, "e1")) - c(o(x, "e2"), o(x, "e1"))) / 2
                       for x in rc_cells])
        rcb = boot([task1[x] for x in rc_cells], rc)[:, 0]
        g3 = {"pairs": len(pa), "pair accuracy": acc3, "rank change": {"cells": len(rc), "point": float(rc.mean()), "ci": ci(rcb)}}
        ok_n = True
        for arm in ("e1", "e2"):
            u = [x for x, a in units if a == arm]
            ds = np.array([o(x, arm, "sim_line") - o(x, arm) for x in u])
            dn3 = np.array([o(x, arm, "neutral_line") - o(x, arm) for x in u])
            b3 = boot([task1[x] for x in u], np.stack([dn3 - ds / 3, dn3 + ds / 3], 1))
            g3[f"neutral, {arm}"] = {"units": len(u), "sim shift": float(ds.mean()), "neutral shift": float(dn3.mean()),
                                     "n - s/3": ci(b3[:, 0]), "n + s/3": ci(b3[:, 1])}
            ok_n = ok_n and ci(b3[:, 0])[1] < 0 and ci(b3[:, 1])[0] > 0
        g3["pass"] = acc3 >= 0.70 and ci(rcb)[0] > M and ok_n
        res["G3"] = g3
        # G4's directions
        tr = np.array([m["role"] == "train" for m in meta])
        dirs = {"probe": w / sd}
        for role in ("control", "neutral_fit"):
            k = np.array([m["role"] == role for m in meta])
            Xk = X[k].astype(np.float64)
            m_, s_ = Xk.mean(0), Xk.std(0) + 1e-6
            lr = LogisticRegression(C=C, fit_intercept=False, max_iter=10000, tol=1e-6).fit((Xk - m_) / s_, np.array([m["label"] for m in meta])[k])
            dirs[role] = lr.coef_[0] / s_
        dirs["random"] = np.random.default_rng(RANDOM_SEED).standard_normal(len(w))
        if not all(np.isfinite(v).all() and np.linalg.norm(v) > 0 for v in dirs.values()):
            raise ValueError(f"{name}: a direction is zero or not finite")
        dirs = {k: v / np.linalg.norm(v) for k, v in dirs.items()}
        own_sd = {k: float((X[tr].astype(np.float64) @ v).std()) for k, v in dirs.items()}
        path = Path("/acts") / "itat" / f"steer_{fit['fit_id']}_{name}_{own_sha[:12]}.npz"
        tmp = path.parent / f".{path.stem}.{os.getpid()}.tmp.npz"
        np.savez(tmp, layer=L, sigma=own_sd["probe"], **{f"d_{k}": v.astype(np.float32) for k, v in dirs.items()})
        try:
            os.link(tmp, path)  # exclusive: never overwrites
        except FileExistsError:  # a rerun of the same code must reproduce it exactly
            old, new = np.load(path), np.load(tmp)
            if any(not np.array_equal(old[k], new[k]) for k in new.files) or set(old.files) != set(new.files):
                raise ValueError(f"{path.name} exists with other contents") from None
        tmp.unlink()
        res["G4 directions"] = {"file": str(path.relative_to("/acts")), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "step sigma (the probe's, every direction)": own_sd["probe"], "each direction's own SD (beside)": own_sd,
                                "cosine to the probe": {k: float(v @ dirs["probe"]) for k, v in dirs.items()}}
        res["G1-G3 pass"] = bool(res["G1"]["pass"] and res["G2"]["pass"] and g3["pass"])
        out[name] = res
    vol.commit()
    return out


if __name__ == "__main__":
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    st = git("status", "--porcelain", "--untracked-files=no")
    if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
        raise SystemExit("commit first: the result must record the commit of the code that made it")
    fits = {}
    for name, f in zip("AB", sys.argv[1:3]):
        r = json.loads((HERE / f).read_text())
        if not r["run"].startswith(f"itat{name}_"):
            raise ValueError(f"{f} is not probe {name}'s fit")
        fits[name] = {"run": r["run"], "fit_id": r["fit_id"], "probes_sha256": r["probes_sha256"]}
    p1 = json.loads((HERE / sys.argv[3]).read_text())
    part1 = {"stamp": p1["provenance"]["stamp"], "rows_sha256": p1["assembled"]["rows_sha256"]}
    if p1["provenance"].get("pilot"):
        raise ValueError("that is the pilot")
    with modal.enable_output(), app.run():
        res = gates.remote(fits, part1, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out = {"git": git("rev-parse", "HEAD").stdout.strip(), "fits": fits, "part1": part1, "m": M, "probes": res,
           "both probes pass G1-G3": False if "stopped" in res else all(r["G1-G3 pass"] for r in res.values())}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"part2_gates_{stamp}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(out, indent=1, default=lambda o: o.item()) + "\n")
    print(json.dumps(out, indent=1, default=lambda o: o.item())[:5000], f"\n-> {path}")
