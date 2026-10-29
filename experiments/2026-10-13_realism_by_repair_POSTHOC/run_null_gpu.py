"""Label-shuffle nulls on H100 with per-unit checkpoints (realism RULE.md Amendments 1-2; world-direction RULE.md Amendment 1).

One container per layer loads d9's fix-full packed features once and runs its units, each writing a JSON to the checkpoint
Volume as it finishes (a stop never loses finished units; a rerun skips them):
- "rr:<job>": realism by repair level, Amendment 1's statistic (summary (ii) per column, 101 columns: real + 100 within-task
  shuffles, seed [20261016, job index, layer]); gate: column 0 equals the as-run value within 0.005.
- "wd": the world-direction realism-left null for k in (0, 1, 2, 3, 5, 10) and the mean difference: the as-run directions
  (from real labels), then each model's held-out realism probe on 101 columns (seed [20261017, layer]); statistic = mean
  over the 20 models of the held-out run AUROC (finite scores only); gate: column 0 within 0.005 of the as-run value;
  plus the count of honest runs with no output rows (K6c2).
- "k17p": the scaling diagnostic: directions fit directly on the honest-z-scored projected features, b″ and c″ refit for
  k in (1, 2, 3, 5, 10), point D (d9's metric rule, no bootstrap) for these and for the as-run refits.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' --with numpy modal run --detach \\
      experiments/2026-10-13_realism_by_repair_POSTHOC/run_null_gpu.py --units <comma list or 'all'> --layers 48,63 --tag <tag>
"""
import json
import os
import subprocess
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-realism-null_gpu_kill-2026-11-13"
WORK, STAMP = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", "2026-10-13_152618"
CKPT = "probe-shift_aswin-surya_sub-realism-null_ckpt_kill-2026-11-13"
IN = Path(os.environ.get("NULL_IN", "/nonexistent"))
SHIP = {"d9.npz": "sce_fixfull_scores_2026-10-13_152618.npz", "wd48.npz": "worlddir_scores_2026-10-13_171314.npz", "wd63.npz": "worlddir_scores_L63_2026-10-13_171543.npz"}
C, MAX_ITER, N_NULL, N_DIR = 0.01, 1000, 100, 10
H, E1, E2, SA, SAF, SAX, SAN, SAF9, SHX = ("real-honest", "real-attack-e1", "real-attack-e2", "sim-attack", "sim-attack-fixed", "sim-attack-fullfix",
                                          "sim-attack-newline", "sim-attack-f9", "sim-honest-fullfix")
LEVELS = {"as-is": "sim-honest", "fix939 (SH′)": "sim-honest-edited", "fix-full (SH″)": "sim-honest-fullfix", "newline-only": "sim-honest-newline",
          "full + F9": "sim-honest-f9"}
RR_JOBS = [*LEVELS, "control"]
AS_RUN = {48: {"as-is": 0.941757, "fix939 (SH′)": 0.814501, "fix-full (SH″)": 0.807814, "newline-only": 0.936638, "full + F9": 0.80801, "control": 0.492752},
          63: {"as-is": 0.971457, "fix939 (SH′)": 0.889374, "fix-full (SH″)": 0.858328, "newline-only": 0.950738, "full + F9": 0.855199, "control": 0.540604}}
WD_AS_RUN = {48: {"k0": 0.7841, "k1": 0.7884, "k2": 0.7736, "k3": 0.7368, "k5": 0.6695, "k10": 0.5825, "md1": 0.4885},
             63: {"k0": 0.8419, "k1": 0.8141, "k2": 0.7600, "k3": 0.7148, "k5": 0.6591, "k10": 0.5654, "md1": 0.5048}}
KS = (0, 1, 2, 3, 5, 10)
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
app = modal.App(APP)
work = modal.Volume.from_name(WORK, version=2).read_only()
ckpt = modal.Volume.from_name(CKPT, version=2, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3").add_local_python_source("probe_shift")
for k, f in SHIP.items():
    if IN.exists():
        image = image.add_local_file(IN / f, f"/root/in/{k}")


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, timeout=1800, retries=0, volumes={"/work": work, "/ckpt": ckpt})
def run(layer: int, units: list, tag: str) -> list:
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import auc, pool, shuffle_within_task, threshold

    t0 = time.time()
    R = np.load("/root/in/d9.npz")
    arm, fold, task, key, cell = (R[f"runs__{k}"] for k in ("arm", "fold", "main_task", "key", "cell"))
    fold = fold.astype(int)
    n = len(arm)
    d = Path("/work/sce_fixfull") / STAMP
    r_run = np.load(d / f"rows_L{layer}.npz")["run"].astype(np.int64)
    X = torch.from_numpy(np.load(d / f"L{layer}.npy")).cuda().float()
    out_dir = Path("/ckpt") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []

    def fit(Z, tr_runs, te_runs, Y, restd=True):
        """Batched fit on rows of tr_runs; returns run scores [n, cols] for te_runs (NaN elsewhere) and W in Z coordinates."""
        tr, te = np.flatnonzero(tr_runs[r_run]), np.flatnonzero(te_runs[r_run])
        Ztr = Z[torch.from_numpy(tr).cuda()]
        if restd:
            mean, std = Ztr.mean(0), Ztr.std(0, unbiased=False)
            std = std.where(std > 0, std.new_ones(()))
        else:
            mean, std = torch.zeros_like(Ztr[0]), torch.ones_like(Ztr[0])
        Yt = torch.from_numpy(np.asarray(Y)[r_run[tr]]).cuda().float()
        W, b, _ = tp.fit((Ztr - mean) / std, Yt, C, MAX_ITER)
        del Ztr
        s = tp.decision((Z[torch.from_numpy(te).cuda()] - mean) / std, W, b).double().cpu().numpy()
        S = pool(r_run[te], s, n, "mean")
        S[~te_runs] = np.nan
        return S, (W / std[:, None])

    def col_auc(S, Y, m):
        return np.array([float(auc(S[m, c], Y[m, c], np.ones((1, m.sum())))[0]) for c in range(Y.shape[1])])

    def save(name, obj):
        p = out_dir / f"L{layer}__{name}.json"
        p.write_text(json.dumps(obj, indent=1, default=float))
        ckpt.commit()
        done.append(name)

    def exists(name):
        return (out_dir / f"L{layer}__{name}.json").exists()

    honest = np.isin(arm, [H, SHX])
    for unit in units:
        if exists(unit.replace(":", "_").replace(" ", "_")):
            continue
        tu = time.time()
        if unit.startswith("rr:"):
            job = unit[3:]
            if job == "control":
                y = np.full(n, np.nan)
                for t in np.unique(task[arm == H]):
                    idx = np.flatnonzero((arm == H) & (task == t))
                    if len(idx) >= 2:
                        o = idx[np.argsort(key[idx], kind="mergesort")]
                        y[o[0]], y[o[1]] = 0.0, 1.0
            else:
                sim_tasks = sorted(set(task[np.isin(arm, list(LEVELS.values()))]))
                y = np.where(arm == LEVELS[job], 1.0, np.where((arm == H) & np.isin(task, sim_tasks), 0.0, np.nan))
            use = ~np.isnan(y)
            rng = np.random.default_rng([20261016, RR_JOBS.index(job), layer])
            Y = np.stack([y] + [shuffle_within_task(y, task, use, rng) for _ in range(N_NULL)], 1)
            acc, cnt = np.zeros((n, N_NULL + 1)), np.zeros(n)
            for f, v in PAIRS:
                te = (fold == f) | (fold == v)
                S, _ = fit(X, use & ~te, use & te, Y)
                m = use & te & np.isfinite(S[:, 0])
                acc[m] += S[m]
                cnt[m] += 1
            m = use & (cnt > 0)
            a = col_auc(acc / np.maximum(cnt, 1)[:, None], Y, m)
            gate = abs(a[0] - AS_RUN[layer][job]) <= 0.005
            null = a[1:]
            save(unit.replace(":", "_").replace(" ", "_"), {"unit": unit, "layer": layer, "gate_passes": bool(gate), "observed": a[0], "as_run": AS_RUN[layer][job],
                                                            "null": null.tolist() if gate else None, "null_mean": float(null.mean()) if gate else None,
                                                            "null_p95": float(np.percentile(null, 95)) if gate else None,
                                                            "p": float((1 + (null >= a[0]).sum()) / (1 + N_NULL)) if gate else None,
                                                            "runs": [int((y[m] == 1).sum()), int((y[m] == 0).sum())], "seconds": round(time.time() - tu)})
        elif unit == "wd":
            y_world = np.where(arm == SHX, 1.0, np.where(arm == H, 0.0, np.nan))
            use = ~np.isnan(y_world)
            rng = np.random.default_rng([20261017, layer])
            Y = np.stack([y_world] + [shuffle_within_task(y_world, task, use, rng) for _ in range(N_NULL)], 1)
            per = {v: [] for v in [f"k{k}" for k in KS] + ["md1"]}
            norows = int(np.sum(use & ~np.isin(np.arange(n), np.unique(r_run))))
            for f, v in PAIRS:
                te = (fold == f) | (fold == v)
                htr, hte = honest & ~te, honest & te
                rows_h = torch.from_numpy(np.flatnonzero(htr[r_run])).cuda()
                mu, sd = X[rows_h].mean(0), X[rows_h].std(0, unbiased=False)
                sd = sd.where(sd > 0, sd.new_ones(()))
                Z = (X - mu) / sd
                U = torch.zeros(X.shape[1], 0, device="cuda")
                for j in range(N_DIR):  # the as-run directions, from the real labels only
                    Zp = Z - (Z @ U) @ U.T if U.shape[1] else Z
                    _, Wd = fit(Zp, htr, hte, y_world[:, None])
                    w = Wd[:, 0]
                    w = w - U @ (U.T @ w) if U.shape[1] else w
                    U = torch.cat([U, (w / w.norm())[:, None]], 1)
                    del Zp
                md = Z[torch.from_numpy(np.flatnonzero((htr & (arm == SHX))[r_run])).cuda()].mean(0) - Z[torch.from_numpy(np.flatnonzero((htr & (arm == H))[r_run])).cuda()].mean(0)
                md = (md / md.norm())[:, None]
                for name, Uk in ({f"k{k}": U[:, :k] for k in KS} | {"md1": md}).items():
                    Zp = Z - (Z @ Uk) @ Uk.T if Uk.shape[1] else Z
                    S, _ = fit(Zp, htr, hte, Y)
                    m = hte & np.isfinite(S[:, 0])
                    per[name].append(col_auc(S, Y, m))
                    del Zp
                del Z
            res = {}
            for name, rows in per.items():
                means = np.mean(np.stack(rows), 0)
                obs, null = means[0], means[1:]
                gate = abs(obs - WD_AS_RUN[layer][name]) <= 0.005
                res[name] = {"gate_passes": bool(gate), "observed": float(obs), "as_run": WD_AS_RUN[layer][name], "null_mean": float(null.mean()),
                             "null_p5": float(np.percentile(null, 5)), "null_p95": float(np.percentile(null, 95)),
                             "p_above": float((1 + (null >= obs).sum()) / (1 + N_NULL)), "p_below": float((1 + (null <= obs).sum()) / (1 + N_NULL))}
            save("wd", {"unit": "wd", "layer": layer, "honest_runs_without_rows": norows, "variants": res, "seconds": round(time.time() - tu)})
        elif unit == "k17p":
            cells = {"b_full": (SAX, "sim-honest-fullfix"), "c_full": (SAX, H)}
            y_world = np.where(arm == SHX, 1.0, np.where(arm == H, 0.0, np.nan))
            alt = {f"k{k}__{c}": np.full((5, 5, n), np.nan) for k in KS if k for c in cells}
            for f, v in PAIRS:
                te = (fold == f) | (fold == v)
                htr, hte = honest & ~te, honest & te
                rows_h = torch.from_numpy(np.flatnonzero(htr[r_run])).cuda()
                mu, sd = X[rows_h].mean(0), X[rows_h].std(0, unbiased=False)
                sd = sd.where(sd > 0, sd.new_ones(()))
                Z = (X - mu) / sd
                U = torch.zeros(X.shape[1], 0, device="cuda")
                for j in range(N_DIR):  # the alternative: no re-standardization inside the nullspace iteration
                    Zp = Z - (Z @ U) @ U.T if U.shape[1] else Z
                    _, Wd = fit(Zp, htr, hte, y_world[:, None], restd=False)
                    w = Wd[:, 0]
                    w = w - U @ (U.T @ w) if U.shape[1] else w
                    U = torch.cat([U, (w / w.norm())[:, None]], 1)
                    del Zp
                for k in KS:
                    if not k:
                        continue
                    Zp = Z - (Z @ U[:, :k]) @ U[:, :k].T
                    for c, (pos, neg) in cells.items():
                        y = np.where(arm == pos, 1.0, np.where(arm == neg, 0.0, np.nan))
                        S, _ = fit(Zp, ~np.isnan(y) & ~te, te, y[:, None])
                        alt[f"k{k}__{c}"][f, v] = S[:, 0]
                    del Zp
                del Z
            wd = np.load(f"/root/in/wd{layer}.npz")
            pairing = [a for a in (E1, E2, SA, SAF, SAX, SAN, SAF9) if (arm == a).any()]

            def caught(nested, alpha):
                ok = np.ones(n, bool)
                for f, v in PAIRS:
                    ok &= ~((fold == f) & ~np.isfinite(nested[f, v]))
                bad = set(cell[np.isin(arm, pairing) & ~ok])
                keep = np.where(np.isin(arm, pairing), ~np.isin(cell, list(bad)), ok)
                hits = np.zeros(n)
                for f, v in PAIRS:
                    s = nested[f, v]
                    hv = np.flatnonzero((arm == H) & (fold == v) & np.isfinite(s))
                    hits[fold == f] += s[fold == f] > threshold(s[hv], alpha)
                return keep, hits / 4

            res = {}
            for alpha in (0.01, 0.05):
                ka, ca = caught(R[f"L{layer}__a__nested_real"], alpha)
                for k in KS:
                    if not k:
                        continue
                    for c in cells:
                        row = {}
                        for src, nested in (("as_run", wd[f"L{layer}__k{k}__{c}"]), ("alt", alt[f"k{k}__{c}"])):
                            kc, cc = caught(nested, alpha)
                            both = kc & ka & (arm == E2)
                            row[src] = {"real_catch": float(cc[kc & (arm == E2)].mean()), "fake_catch": float(cc[kc & (arm == SAX)].mean()),
                                        "D": float(cc[both].mean() - ca[both].mean())}
                        row["alt_minus_as_run_D"] = row["alt"]["D"] - row["as_run"]["D"]
                        res[f"k{k}__{c}__fpr_{alpha:g}"] = row
            save("k17p", {"unit": "k17p", "layer": layer, "points": res, "seconds": round(time.time() - tu)})
    return [{"layer": layer, "done": done, "seconds": round(time.time() - t0)}]


@app.local_entrypoint()
def main(units: str = "all", layers: str = "48,63", tag: str = ""):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout:
        raise SystemExit("refusing to run from a dirty tree")
    if not tag:
        raise SystemExit("--tag is required (the checkpoint folder)")
    us = [f"rr:{j}" for j in RR_JOBS] + ["wd", "k17p"] if units == "all" else units.split(",")
    for r in run.starmap([(int(L), us, tag) for L in layers.split(",")]):
        print(json.dumps(r))
