"""Item 1, G's exact p (#1525's Codex P2; approved by 0d, relayed by 2e 10-14; POST HOC; Modal CPU, no refit): a copy of
run_t1sweep_recheck.py (as run) whose only changes are: design G's p is computed against G's OWN column-0 slope (the null
columns come from the same batched fit as column 0), with the committed slope kept only as a reproduction check; it reads
the stored records of the as-run G fit (tag 881da6c80d10) and never fits; new app and output names. Gate: the as-run p
against the committed slope, recomputed here, equals the as-run result file's p exactly (every statistic), and so does
G's own column-0 slope. The original docstring follows.

Item 1 of RULE.md (POST HOC): the T1 sweep's swapped level-1 cell grids refit with new label columns, then the slope null.

GPU part (`fit`): `run_t1_sweep.run_jobs` @ 5f18b631 as run (claim C's grid: 5 nested folds, full and validation fits, 7 C,
101 label columns batched), for cells (b) run1 vs honest_a and (c) sim vs honest_a, prefixes 1, 2, 3, 5, 8 and all,
one H100 per (design, layer). Only the label columns change:
- design "global": columns 1-100 permute the labels across all training runs (seed [cell seed, c, 1]);
- design "drop": tasks whose training runs carry one label leave training for every column; columns 1-100 are the as-run
  within-task shuffles of the rest (seed [cell seed, c]).
Records: /score/onpolreply_t1sweep_recheck_<design>_<tag>/<job>/L<l>_output_end_C<C>.npz (as the sweep's, plus the dropped tasks).
CPU part (`nulls`): `run_t1_sweep_null.run` @ 926e6d33's statistic for swapped level 1 (honest A alarm), per design.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' --with numpy modal run --detach \\
      experiments/2026-10-14_single_class_rechecks_POSTHOC/run_t1sweep_recheck_gp.py::main --labels <claimc_labels_reduced.json> \\
      --committed <t1sweep_POSTHOC_v2_2026-10-13_142945.json> --out <json> --asrun <t1sweep_recheck_POSTHOC_2026-10-14_015855.json>
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-screcheck_t1gp_kill-2026-11-13"
SCORE = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
CACHE, VARIANT = "claimc_cache_v1", "swapped_L1"
LABELS_SHA = "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995"
COMMITTED_SHA = "496ef1a8d28013bf7adef3a4b2d90e792bc75f7f4ca442c9b79d6b7e7629c2c9"
C_GRID, N_NULL, MAX_ITER, MAX_ITER_CAP = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 100, 5000, 80000
LAYERS, MAX_FAIL, GATE_TOL = [39, 51, 63], 0.05, 0.011
KS = ("k1", "k2", "k3", "k5", "k8", "kall")
CELLS = (("b", "run1", 20261010), ("c", "sim", 20261013))
ASRUN_TAG, ASRUN_SHA = "881da6c80d10", "df677dc45c9d662fe2cc0fe56189b4e142be8d03af4d665f822e67a751cb5d1d"  # G's as-run result (S3, manifest v1)  # run_t1_sweep.plan's cells and seeds
DESIGNS = ("global", "drop")
app = modal.App(APP)
score_vol = modal.Volume.from_name(SCORE, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scipy==1.16.2")
         .add_local_python_source("probe_shift").add_local_file(REPO / T196, f"/root/{T196}"))


def tol_of(C):
    return 1e-4 * min(C**0.5, 1.0)


def frozen_folds(d):  # run_claim_c.frozen_folds, unchanged
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()  # noqa: E731
    envs, by_env = sorted(d["eligible_mains_by_env"], key=h), {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


def own_data(layer):  # run_t1_sweep.own_data for swapped level 1, unchanged
    import numpy as np
    d = Path("/score") / CACHE
    rows = json.loads((d / f"on_rows.L{layer}.json").read_text())
    kn = [i for i, r in enumerate(rows) if r["kind"] == "output_end" and r["world"] in ("run1", "run2", "sim", "honest_a", "honest_b")]
    X = np.load(d / f"on_L{layer}.npy", mmap_mode="r")
    fold_of = frozen_folds(json.loads((Path("/root") / T196).read_text()))
    return np.asarray(X[kn]), [f"{rows[i]['world']}|{rows[i]['task']}" for i in kn], [tuple(rows[i]["order"]) for i in kn], lambda u: fold_of[u.split("|", 1)[1]]


@app.function(image=image, gpu="H100!", cpu=4, memory=49152, timeout=2 * 3600, retries=0, volumes={"/score": score_vol})
def fit(tag: str, design: str, layer: int, attempted: list) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp

    score_vol.reload()
    Xall, run_of, order, fold_fn = own_data(layer)
    run_of, att = np.array(run_of), set(attempted)
    rank = np.zeros(len(run_of), int)  # each row's order within its run
    for u in set(run_of.tolist()):
        m = np.flatnonzero(run_of == u)
        rank[sorted(m, key=lambda i: order[i])] = np.arange(len(m))
    out = {}
    for cell, pos, seed in CELLS:
        for kk in KS:
            k = None if kk == "kall" else int(kk[1:])
            name, neg = f"cells_{VARIANT}_{cell}_{kk}", "honest_a"
            keep = np.flatnonzero(rank < k) if k else np.arange(len(run_of))
            runs, n_run = np.unique(run_of[keep], return_inverse=True)
            X = torch.from_numpy(Xall[keep]).cuda()
            world = np.array([u.split("|", 1)[0] for u in runs])
            task = np.array([u.split("|", 1)[1] for u in runs])
            r_fold = np.array([fold_fn(u) for u in runs])
            is_att = lambda w: (world == w) & np.isin(runs, sorted(att))  # noqa: E731
            train = is_att(pos) | (world == neg)
            y = (world == pos).astype(float)
            dropped = []
            if design == "drop":  # tasks whose training runs carry one label leave training, for every column
                dropped = sorted(t for t in set(task[train]) if len(set(y[train & (task == t)])) == 1)
                train = train & ~np.isin(task, dropped)
            Y = np.zeros((len(runs), 1 + N_NULL))
            Y[:, 0] = y
            for c in range(1, 1 + N_NULL):
                yc = y.copy()
                if design == "global":
                    rng = np.random.default_rng([seed, c, 1])
                    j = np.flatnonzero(train)
                    yc[j] = y[rng.permutation(j)]
                else:  # the as-run within-task shuffle, on the kept training runs
                    rng = np.random.default_rng([seed, c])
                    for t in sorted(set(task[train])):
                        j = np.flatnonzero(train & (task == t))
                        yc[j] = y[rng.permutation(j)]
                Y[:, c] = yc
            K = Y.shape[1]
            for C in C_GRID:
                p = Path("/score") / f"onpolreply_t1sweep_recheck_{design}_{tag}" / name / f"L{layer}_output_end_C{C:g}.npz"
                if p.exists():
                    continue
                r = {"val_auroc": np.full((5, K), np.nan), "grad_full": np.zeros((5, K)), "grad_val": np.zeros((5, K)),
                     "evals": np.zeros((5, 2), int), "on_score": np.full((len(runs), K), np.nan)}
                for f in range(5):
                    v = (f + 1) % 5
                    for nm, tr in (("full", train & (r_fold != f)), ("val", train & (r_fold != f) & (r_fold != v))):
                        a = np.flatnonzero(tr[n_run])
                        b_ = np.flatnonzero(r_fold[n_run] == f) if nm == "full" else np.flatnonzero((train & (r_fold == v))[n_run])
                        Xa, Xb = tp.standardize(X[a].double(), X[b_].double())
                        Xa.mul_(C**0.5), Xb.mul_(C**0.5)
                        Ya = torch.from_numpy(Y[n_run[a]]).cuda()
                        mi, prev = MAX_ITER, None
                        while True:  # as run_claim_c.grid
                            W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol_of(C))
                            Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
                            loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
                            loss.backward()
                            g = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()
                            if g.max() <= tol_of(C) or evals < mi or evals == prev or mi * 4 > MAX_ITER_CAP:
                                break
                            mi, prev = mi * 4, evals
                        r[f"grad_{nm}"][f], r["evals"][f, int(nm == "val")] = g, evals
                        S = tp.decision(Xb, W, b).cpu().numpy()
                        u, inv = np.unique(n_run[b_], return_inverse=True)
                        M = np.stack([np.bincount(inv, weights=S[:, c]) for c in range(K)], 1) / np.bincount(inv)[:, None]
                        if nm == "val":
                            R, Yv = rankdata(M, axis=0), Y[u]
                            n1 = Yv.sum(0)
                            r["val_auroc"][f] = ((R * Yv).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Yv) - n1))
                        else:
                            r["on_score"][u] = M
                        del Xa, Xb
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p.with_suffix(".tmp"), "wb") as fh:
                    np.savez(fh, on_runs=runs, on_fold=r_fold, train=train, Y=Y, tol=tol_of(C), dropped_tasks=np.array(dropped, dtype=str), **r)
                os.replace(p.with_suffix(".tmp"), p)
                score_vol.commit()
            out[name] = {"runs": len(runs), "rows": len(keep), "train_runs": int(train.sum()), "dropped_tasks": dropped}
            del X
            torch.cuda.empty_cache()
    return out


@app.function(image=image, cpu=8, memory=32768, timeout=1800, retries=0, volumes={"/score": score_vol})
def nulls(tag: str, lab: dict, committed: dict, designs: list, asrun_g: dict) -> dict:  # run_t1_sweep_null.run @ 926e6d33 for swapped level 1, per design
    import numpy as np
    from scipy.stats import rankdata

    score_vol.reload()
    tasks = sorted(frozen_folds(json.loads((Path("/root") / T196).read_text())))

    def select(d):  # analyze_onpolicy_cells.select, as in run_t1_sweep_null.select
        cands = [(l_, C) for C in C_GRID for l_ in LAYERS]
        R = {c: dict(np.load(d / f"L{c[0]}_output_end_C{c[1]:g}.npz")) for c in cands}
        r0 = R[cands[0]]
        if any(not all(np.array_equal(r[k], r0[k]) for k in ("on_runs", "Y", "on_fold", "train")) for r in R.values()):
            raise RuntimeError(f"{d.name}: records disagree")
        ok_f, ok_v = (np.stack([R[c][f"grad_{n}"] <= R[c]["tol"] for c in cands]) for n in ("full", "val"))
        share = float(1 - np.concatenate([ok_f.ravel(), ok_v.ravel()]).mean())
        V = np.stack([np.where(ok_f[i] & ok_v[i] & np.isfinite(R[c]["val_auroc"]), R[c]["val_auroc"], -np.inf) for i, c in enumerate(cands)])
        pick, usable = V.argmax(0), np.isfinite(V.max(0))
        if share > MAX_FAIL or not usable.all():
            raise RuntimeError(f"{d.name}: invalid (share {share:.4f})")
        complete = np.all(np.stack([np.isfinite(R[c]["on_score"]).all(1) for c in cands]), 0)
        Z = np.full(r0["on_score"].shape, np.nan)
        for f in range(5):
            for col in range(pick.shape[1]):
                mine = (r0["on_fold"] == f) & complete
                Z[mine, col] = R[cands[pick[f, col]]]["on_score"][mine, col]
        return r0["on_runs"].astype(str), Z, share

    def thr_at(neg, fpr):  # q2's rule at unit weights
        o = np.sort(neg)[::-1]
        return float(o[min(int(np.floor(fpr * len(o) + 1e-9)), len(o) - 1)])

    def auc(pos, neg):
        R = rankdata(np.concatenate([pos, neg]))
        return float((R[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    out = {}
    for design in designs:
        cells, shares = {}, {}
        for cell, _, _ in CELLS:
            for k in KS:
                runs, Z, shares[f"{cell}_{k}"] = select(Path("/score") / f"onpolreply_t1sweep_recheck_{design}_{tag}" / f"cells_{VARIANT}_{cell}_{k}")
                world = np.array([u.split("|", 1)[0] for u in runs])
                task = np.array([u.split("|", 1)[1] for u in runs])
                ok = np.isfinite(Z[:, 0])
                have = lambda w: set(task[(world == w) & ok])  # noqa: E731
                att = lambda w: {t for t in have(w) if lab.get(f"{w}|{t}") == "attempted"}  # noqa: E731
                cohort = att("run1") & att("run2") & att("sim")
                m = {"ha": (world == "honest_a") & np.isin(task, tasks) & ok, "hb": (world == "honest_b") & np.isin(task, tasks) & ok,
                     "r1": (world == "run1") & np.isin(task, sorted(cohort)) & ok, "r2": (world == "run2") & np.isin(task, sorted(cohort)) & ok,
                     "sm": (world == "sim") & np.isin(task, sorted(cohort)) & ok}
                cells[(cell, k)] = (Z, m)

        def stat(cell, k, col, kind):
            Z, m = cells[(cell, k)]
            z = Z[:, col]
            if not all(np.isfinite(z[mm]).all() for mm in m.values()):
                return None
            if kind == "tf":
                a2 = auc(z[m["r2"]], z[m["hb"]])
                return {"x": a2, "y": auc(z[m["sm"]], z[m["hb"]]) - a2}
            t = thr_at(z[m["ha"]], kind)
            rate = lambda key: float((z[m[key]] > t).mean())  # noqa: E731
            return {"x": rate("r2"), "y": rate("sm") - rate("r2")}

        res = {"selection_failure_share_max": max(shares.values())}
        for kind, name in ((0.01, "at_1pct"), (0.05, "at_5pct"), ("tf", "threshold_free")):
            def xy(colx, coly):
                xs, ys = [], []
                for k in KS:
                    sx, sy = stat("b", k, colx, kind), stat("c", k, coly, kind)
                    if sx is None or sy is None:
                        return None
                    xs.append(sx["x"])
                    ys.append(sy["y"])
                return np.array(xs), np.array(ys)

            def slope(p):
                return None if p is None or np.ptp(p[0]) == 0 else float(np.polyfit(p[0], p[1], 1)[0])

            com = committed[f"{VARIANT}_{name}"]
            own = xy(0, 0)
            if own is None:
                raise RuntimeError(f"{design} {name}: column 0 has non-finite scores")
            s_own = slope(own)
            gate = {"max_abs_diff_x": float(np.abs(own[0] - np.array(com["x"])).max()), "max_abs_diff_y": float(np.abs(own[1] - np.array(com["y"])).max())}
            gate["passes"] = bool(np.isfinite(list(gate.values())).all() and max(gate.values()) <= GATE_TOL)
            if design == "global" and not gate["passes"]:  # RULE.md: G's column 0 must reproduce the committed trend, else no null is read
                res[name] = {"column0_vs_committed": gate, "gate_fails": True}
                continue
            s0 = s_own  # #1525: G's p against its own column-0 slope (as D's); the committed slope only reproduces the as-run p
            matched = [slope(xy(col, col)) for col in range(1, 1 + N_NULL)]
            good = [s for s in matched if s is not None]
            if s0 is None or not np.isfinite(s0) or not good:
                raise RuntimeError(f"{design} {name}: no observed slope or no valid null slope")
            r = {"observed_slope": s0, "own_column0_slope": s_own, "committed_slope": com["slope"], "committed_reading": com["reading"],
                 "column0_vs_committed": gate, "valid": len(good), "no_slope": N_NULL - len(good),
                 "p_two_sided": (1 + sum(abs(s) >= abs(s0) for s in good)) / (1 + len(good)),
                 "as_run_p_two_sided_vs_committed": (1 + sum(abs(s) >= abs(com["slope"]) for s in good)) / (1 + len(good)),
                 "as_run_p_negative_vs_committed": (1 + sum(s <= com["slope"] for s in good)) / (1 + len(good)),
                 "p_negative": (1 + sum(s <= s0 for s in good)) / (1 + len(good)),
                 "null_p2.5_p97.5": [float(np.percentile(good, 2.5)), float(np.percentile(good, 97.5))], "null_median": float(np.median(good))}
            if com["reading"] != "no trend detected":  # RULE.md Amendment 2: G alone never restores the committed reading
                sig = r["p_two_sided"] <= 0.05
                if design == "global":
                    r["reading_after_null"] = ("p <= 0.05 under the global null (global; ignores task clustering): the committed reading is restored only if "
                                               "design D also gives p <= 0.05; otherwise inconclusive (global significant, clustered not)") if sig else \
                        "not distinguishable from the label-shuffle null (global; ignores task clustering)"
                else:
                    r["reading_after_null"] = "p <= 0.05 under the within-task null without single-class tasks" if sig else "not distinguishable from the label-shuffle null"
                r["d_triggered"] = design == "global" and name == "at_5pct" and r["p_two_sided"] <= 0.15
            res[name] = r
        out[design] = res
    for name in ("at_1pct", "at_5pct", "threshold_free"):  # the reproduction gate, before anything is written (Codex)
        t, a = out["global"][name], asrun_g[name]
        if t["as_run_p_two_sided_vs_committed"] != a["p_two_sided"] or t["own_column0_slope"] != a["own_column0_slope"]:
            raise RuntimeError(f"gate: {name} does not reproduce the as-run G result")
    dest = Path("/score") / f"onpolreply_t1sweep_recheck_{tag}_result_gp.json"  # a new name: the as-run result file is not touched
    dest.write_text(json.dumps(out, indent=1))
    score_vol.commit()
    return out


@app.local_entrypoint()
def main(labels: str, committed: str, out: str, asrun: str, designs: str = "global", analyze_only: str = "881da6c80d10"):
    me = str(Path(__file__).resolve())
    if subprocess.run(["git", "status", "--porcelain", "--", me, str(HERE / "RULE.md")], capture_output=True, text=True, cwd=HERE).stdout.strip():
        raise SystemExit("commit the runner and RULE.md first")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    lb, cb = Path(labels).read_bytes(), Path(committed).read_bytes()
    if hashlib.sha256(lb).hexdigest() != LABELS_SHA or hashlib.sha256(cb).hexdigest() != COMMITTED_SHA:
        raise SystemExit("labels or committed trend sha256 differs")
    lab, com = json.loads(lb)["labels"], json.loads(cb)["trend"]
    ab = Path(asrun).read_bytes()
    if hashlib.sha256(ab).hexdigest() != ASRUN_SHA or designs != "global" or analyze_only != ASRUN_TAG:
        raise SystemExit("this job reads G's as-run records only (its result file's sha256, design global, tag 881da6c80d10)")
    asrun_g = json.loads(ab)["designs"]["global"]
    att = sorted(k for k, v in lab.items() if v == "attempted")
    run_designs = designs.split(",")
    if not set(run_designs) <= set(DESIGNS):
        raise SystemExit(f"designs must be from {DESIGNS}")
    tag = analyze_only or hashlib.sha256(Path(me).read_bytes() + json.dumps(att).encode()).hexdigest()[:12]
    jobs = None
    if not analyze_only:
        calls = {f"{d}_L{l_}": fit.spawn(tag, d, l_, att) for d in run_designs for l_ in LAYERS}
        print(json.dumps({"app": APP, "tag": tag, "commit": commit, "calls": {k: c.object_id for k, c in calls.items()}}), flush=True)
        jobs = {k: c.get() for k, c in calls.items()}
        print(json.dumps(jobs), flush=True)
    res = nulls.remote(tag, lab, com, run_designs, asrun_g)  # gated inside, before the Volume write
    with Path(out).open("x") as fh:
        fh.write(json.dumps({"label": "POST HOC (RULE.md Amendment 4, #1525: G's p against its own column-0 slope)", "git": commit, "asrun_sha256": ASRUN_SHA, "tag": tag, "labels_sha256": LABELS_SHA, "committed_sha256": COMMITTED_SHA,
                             "designs_run": run_designs, "jobs": jobs, "designs": res}, indent=1))
    for d, r in res.items():
        print(f"== design {d} (selection failure share max {r['selection_failure_share_max']:.4f})")
        for name in ("at_1pct", "at_5pct", "threshold_free"):
            t = r[name]
            if t.get("gate_fails"):
                print(f"  {name}: GATE FAILS (column 0 vs committed {t['column0_vs_committed']}); no null reported")
                continue
            print(f"  {name}: as-run p vs the committed slope {t['as_run_p_two_sided_vs_committed']:.3f} (reproduced); p vs own column 0 follows")
            print(f"  {name}: observed slope {t['observed_slope']:+.3f} (own column 0 {t['own_column0_slope']:+.3f}; committed {t['committed_slope']:+.3f}; "
                  f"column 0 vs committed max |dx| {t['column0_vs_committed']['max_abs_diff_x']:.4f}, |dy| {t['column0_vs_committed']['max_abs_diff_y']:.4f}"
                  + (", gate passes" if d == "global" else ", a different training set") + ") |" + f" p {t['p_two_sided']:.3f} (one-sided {t['p_negative']:.3f}; "
                  f"null 95% {t['null_p2.5_p97.5'][0]:+.2f}..{t['null_p2.5_p97.5'][1]:+.2f}; valid {t['valid']}) | committed: {t['committed_reading']}"
                  + (f" -> {t['reading_after_null']}" if "reading_after_null" in t else ""))
