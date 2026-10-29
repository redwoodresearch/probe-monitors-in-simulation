"""POST-HOC claim C test (plan Amendment 8): a detector trained on honest-instruction readings. Deployed Modal app:
- cache (CPU): every readout (output_end, call_end) at layers 39, 51, 63 of the swapped run 1, swapped run 2 and
  swapped simulated attack extractions (each must be complete), plus honest_a and honest_b from the retrain's level-1
  cache, into /score/claimc_cache_v1/.
- grid (H100, one per layer and readout): as run_retrain_qwen.grid, one cell (level 1, attempted): swapped run 1
  (attempted) vs honest_a, the 196-task frozen folds, the inner validation fold, 7 C, 1 + 100 label columns (shuffles
  within each task, seed [20261010, d]); records under /score/claimc_<tag>/ with Y, the training mask and out-of-fold run
  scores. `timing`: Aswin's 10:51 PT rule: L51 output_end, C 1e-6, 1e-2 and 1, every column a shuffle, never analyzed.

  modal deploy run_claim_c.py
  uv run ... python run_claim_c.py cache | timing <labels> | grid <labels>     (spawns on the deployed app)
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q3onpol_claimc_kill-2026-11-10"
SCORE_VOL, ACTS_VOL = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
CACHE, SRC_CACHE = "claimc_cache_v1", "retrain_cache_v1/level1"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
BASES = ["2026-10-09_rerun_strict_segends_v1_scrub1", "2026-10-10_scaleup_canary_segends_v1_scrub1",
         "2026-10-10_scaleup_l2_segends_v1_scrub1", "2026-10-10_scaleup_topup_segends_v1_scrub1"]
SWAPPED = [(f"{b}_swapsys", "attack", "run1") for b in BASES] + [("2026-10-10_second_attack_segends_v1_scrub1_swapsys", "attack", "run2")] \
          + [(f"{b}_swapsys2", "sim_attack@september", "sim") for b in BASES]
LAYERS, KINDS, C_GRID = [39, 51, 63], ["output_end", "call_end"], [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_NULL, MAX_ITER, MAX_ITER_CAP = 100, 5000, 80000

app = modal.App(APP)
score_vol, acts_vol = modal.Volume.from_name(SCORE_VOL, version=2), modal.Volume.from_name(ACTS_VOL, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy")
         .add_local_python_source("probe_shift").add_local_file(REPO / T196, f"/root/{T196}"))


def tol_of(C: float) -> float:
    return 1e-4 * min(C**0.5, 1.0)


def frozen_folds(d: dict) -> dict:  # run_score_onpolicy.frozen_folds (amendment 25), on a parsed task file
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()
    envs, by_env = sorted(d["eligible_mains_by_env"], key=h), {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


@app.function(image=image, cpu=8, memory=65536, timeout=3600, retries=0, volumes={"/score": score_vol, "/acts": acts_vol.read_only()})
def cache(layer: int) -> dict:
    import numpy as np

    score_vol.reload()
    acts_vol.reload()
    d = Path("/score") / CACHE
    d.mkdir(exist_ok=True)
    rows, parts, counted = [], [], {}
    for run, arm, world in SWAPPED:
        root = Path("/acts") / run
        done = json.loads((root / "complete.json").read_text())
        metas = [json.loads(f.read_text()) for f in sorted((root / "meta" / arm).glob("*.json"))]
        if len(metas) != done["epochs"]:
            raise ValueError(f"{run}: {len(metas)} epochs extracted, {done['epochs']} staged")
        picks = []
        for m in metas:
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            counted.setdefault((world, m["task"]), set()).add(m["run"])
            for c in m["calls"]:
                if c.get("kind") in KINDS and c.get("row") is not None:
                    rows.append({"world": world, "task": m["task"], "kind": c["kind"], "order": [c["turn"], c.get("call_index", 0)]})
                    picks.append((m["chunk"], m["row_start"] + c["row"]))
        X = np.empty((len(picks), 5120), np.float16)
        for chunk in sorted({p[0] for p in picks}):
            a = np.load(root / "by_layer" / f"L{layer}" / f"{chunk}.npy")
            idx = [i for i, p in enumerate(picks) if p[0] == chunk]
            X[idx] = a[[picks[i][1] for i in idx]]
        parts.append(X)
    if any(len(v) > 1 for v in counted.values()):
        raise ValueError("several counted runs for one world and task")
    src = json.loads((Path("/score") / SRC_CACHE / f"on_rows.L{layer}.json").read_text())
    keep = [i for i, r in enumerate(src) if r["world"] in ("honest_a", "honest_b")]
    parts.append(np.load(Path("/score") / SRC_CACHE / f"on_L{layer}.npy")[keep])
    rows += [src[i] for i in keep]
    X = np.concatenate(parts)
    if len(X) != len(rows) or not np.isfinite(X.astype(np.float32)).all():
        raise ValueError("rows and features disagree, or non-finite activations")
    np.save(d / f"on_L{layer}.npy", X)
    (d / f"on_rows.L{layer}.json").write_text(json.dumps(rows))
    score_vol.commit()
    return {"layer": layer, "rows": len(rows), "by_world": {w: sum(r["world"] == w for r in rows) for w in ("run1", "run2", "sim", "honest_a", "honest_b")}}


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, timeout=2 * 3600, retries=0, volumes={"/score": score_vol}, scaledown_window=5)
def grid(tag: str, layer: int, kind: str, attempted: list, timing: bool = False) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp

    score_vol.reload()
    fold_of = frozen_folds(json.loads((Path("/root") / T196).read_text()))
    att, t0, out = set(attempted), time.time(), {}
    d = Path("/score") / CACHE
    rows = json.loads((d / f"on_rows.L{layer}.json").read_text())
    kn = np.flatnonzero([r["kind"] == kind for r in rows])
    X = torch.from_numpy(np.load(d / f"on_L{layer}.npy")[kn]).cuda()
    runs, n_run = np.unique([f"{r['world']}|{r['task']}" for r in rows], return_inverse=True)
    n_run = n_run[kn]
    world = np.array([u.split("|", 1)[0] for u in runs])
    task = np.array([u.split("|", 1)[1] for u in runs])
    r_fold = np.array([fold_of[t] for t in task])
    train = (world == "honest_a") | ((world == "run1") & np.isin(runs, sorted(att)))
    y = (world == "run1").astype(float)
    Y = np.zeros((len(runs), 1 + N_NULL))
    Y[:, 0] = y
    for c in range(0 if timing else 1, 1 + N_NULL):  # timing: column 0 is a shuffle too (no real labels)
        rng = np.random.default_rng([20261010, c + (1000 if timing and c == 0 else 0)])
        yc = y.copy()
        for t in sorted(set(task[train])):
            j = np.flatnonzero(train & (task == t))
            yc[j] = y[rng.permutation(j)]
        Y[:, c] = yc
    K = Y.shape[1]
    for C in ([1e-6, 1e-2, 1.0] if timing else C_GRID):
        p = Path("/score") / f"claimc_{tag}{'_TIMING' if timing else ''}" / f"L{layer}_{kind}_C{C:g}.npz"
        if p.exists():
            continue
        r = {"val_auroc": np.full((5, K), np.nan), "grad_full": np.zeros((5, K)), "grad_val": np.zeros((5, K)),
             "evals": np.zeros((5, 2), int), "on_score": np.full((len(runs), K), np.nan)}
        for f in range(5):
            v = (f + 1) % 5
            for name, tr in (("full", train & (r_fold != f)), ("val", train & (r_fold != f) & (r_fold != v))):
                a = np.flatnonzero(tr[n_run])
                b_ = np.flatnonzero(r_fold[n_run] == f) if name == "full" else np.flatnonzero((train & (r_fold == v))[n_run])
                Xa, Xb = tp.standardize(X[a].double(), X[b_].double())
                Xa.mul_(C**0.5), Xb.mul_(C**0.5)
                Ya = torch.from_numpy(Y[n_run[a]]).cuda()
                mi, prev = MAX_ITER, None
                while True:  # refit only fits that used all their iterations; stop when a refit repeats
                    W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol_of(C))
                    Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
                    loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
                    loss.backward()
                    g = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()
                    if g.max() <= tol_of(C) or evals < mi or evals == prev or mi * 4 > MAX_ITER_CAP:
                        break
                    mi, prev = mi * 4, evals
                r[f"grad_{name}"][f], r["evals"][f, int(name == "val")] = g, evals
                S = tp.decision(Xb, W, b).cpu().numpy()
                u, inv = np.unique(n_run[b_], return_inverse=True)
                M = np.stack([np.bincount(inv, weights=S[:, c]) for c in range(K)], 1) / np.bincount(inv)[:, None]
                if name == "val":
                    R, Yv = rankdata(M, axis=0), Y[u]
                    n1 = Yv.sum(0)
                    r["val_auroc"][f] = ((R * Yv).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Yv) - n1))
                else:
                    r["on_score"][u] = M
                del Xa, Xb
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p.with_suffix(".tmp"), "wb") as fh:
            np.savez(fh, on_runs=runs, on_fold=r_fold, train=train, Y=Y, tol=tol_of(C), **r)
        os.replace(p.with_suffix(".tmp"), p)
        score_vol.commit()
        out[f"{C:g}"] = round(time.time() - t0)
        print(f"L{layer} {kind} C {C:g} done at {out[f'{C:g}']} s; evals max {r['evals'].max()}", flush=True)
    return out


def attempted_run1(labels: str) -> list:
    att = []
    for f in labels.split(","):  # run 1 is every label folder's attack arm except the second batch's
        if "second_attack" in Path(f).name:
            continue
        att += [f"run1|{r['task']}" for r in map(json.loads, Path(f).read_text().splitlines()) if r["arm"] == "attack" and r.get("label") == "attempted"]
    if len(att) != len(set(att)):
        raise ValueError("a run-1 task is labelled twice")
    return sorted(att)


if __name__ == "__main__":  # spawns on the deployed app
    cmd = sys.argv[1]
    if cmd not in ("cache", "timing", "grid") or (cmd != "cache" and len(sys.argv) != 3):
        raise SystemExit("usage: run_claim_c.py cache | timing <labels> | grid <labels>")
    if cmd == "cache":
        print(json.dumps({l: modal.Function.from_name(APP, "cache").spawn(l).object_id for l in LAYERS}))
    else:
        att = attempted_run1(sys.argv[2])
        tag = hashlib.sha256((HERE / "run_claim_c.py").read_bytes() + json.dumps(att).encode()).hexdigest()[:12]
        fn = modal.Function.from_name(APP, "grid")
        jobs = [(51, "output_end", True)] if cmd == "timing" else [(l, k, False) for l in LAYERS for k in KINDS]
        print(json.dumps({"tag": tag, "attempted_run1": len(att), "calls": {f"L{l}_{k}": fn.spawn(tag, l, k, att, t).object_id for l, k, t in jobs}}))
