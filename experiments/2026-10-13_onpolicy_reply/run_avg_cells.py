"""Cells (b) and (c) on the averaged-token readout (RULE.md Amendment 4). Deployed Modal app.

- build (CPU, per variant and layer): a claim C-format cache (`on_L<l>.npy` fp16 averaged rows, `on_rows.L<l>.json` with
  world, task, kind "output_end", order [turn, call_index]) from the pooled extraction runs, counted attempts only
  (redrawn and rerun attempts out, as claim C's cache). Worlds: swapped_L1 = run 1 and run 2 from `*_scrub1_swapsys`,
  simulated from `*_scrub1_swapsys2`, honest A and B from the level-1 stagings (claim C shares them the same way);
  unswapped_L1 / unswapped_L0 = every world from the level-1 / level-0 stagings. Gate (row identity): each row's
  LAST-token vector from the pooled run equals the existing last-token cache's row for the same (world, task, turn,
  call_index) (claimc_cache_v1 / retrain_cache_v1/level1 / level0), max |diff| <= 1e-2, and the row sets are equal; else
  nothing is written for that variant and layer.
- grid (H100, per variant, training arm and layer): run_cell_c.grid unchanged except the cache (onpolreply_avg_cache/
  <variant>), the training attack world ("run1" for cell (b), "sim" for (c)), the label y = world == that arm, and the
  shuffle seed [20261013, c] for (c) / [20261010, c] for (b) (claim C's and the retrain's own seeds). Records under
  onpolreply_avg_<tag>/<variant>_<cell>/, read by analyze_onpolicy_cells.py.

  modal deploy experiments/2026-10-13_onpolicy_reply/run_avg_cells.py
  uv run --no-project --python 3.12 --with "modal==1.6.1" python experiments/2026-10-13_onpolicy_reply/run_avg_cells.py build | grid <labels json>
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
APP = "probe-shift_aswin-surya_sub-onpol-reply_avgcells_kill-2026-11-13"
ACTS, SCORE = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
LAYERS, C_GRID, N_NULL, MAX_ITER, MAX_ITER_CAP = [39, 51, 63], [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 100, 5000, 80000
P = "2026-10-13_onpolreply_pooled_"
L0 = ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1", "2026-10-10_scaleup_topup_segends_v1"]
L1 = [f"{b}_scrub1" for b in L0]
ARMW = {"attack": "run1", "honest_a": "honest_a", "honest_b": "honest_b", "sim_attack@september": "sim"}
SOURCES = {  # variant -> [(pooled run, arm, world)]
    "swapped_L1": [(P + f"{b}_swapsys", "attack", "run1") for b in L1] + [(P + "2026-10-10_second_attack_segends_v1_scrub1_swapsys", "attack", "run2")]
                  + [(P + f"{b}_swapsys2", "sim_attack@september", "sim") for b in L1] + [(P + b, a, ARMW[a]) for b in L1 for a in ("honest_a", "honest_b")],
    "unswapped_L1": [(P + b, a, ARMW[a]) for b in L1 for a in ARMW] + [(P + "2026-10-10_second_attack_segends_v1_scrub1", "attack", "run2")],
    "unswapped_L0": [(P + b, a, ARMW[a]) for b in L0 for a in ARMW] + [(P + "2026-10-10_second_attack_segends_v1", "attack", "run2")]}
REF = {"swapped_L1": "claimc_cache_v1", "unswapped_L1": "retrain_cache_v1/level1", "unswapped_L0": "retrain_cache_v1/level0"}
OUT = "onpolreply_avg_cache"

app = modal.App(APP)
acts_vol, score_vol = modal.Volume.from_name(ACTS, version=2), modal.Volume.from_name(SCORE, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy")
         .add_local_python_source("probe_shift").add_local_file(REPO / T196, f"/root/{T196}"))


def tol_of(C):
    return 1e-4 * min(C**0.5, 1.0)


def frozen_folds(d):  # run_claim_c.frozen_folds, unchanged
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()  # noqa: E731
    envs, by_env = sorted(d["eligible_mains_by_env"], key=h), {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


@app.function(image=image, cpu=8, memory=98304, timeout=3600, retries=0, volumes={"/acts": acts_vol.read_only(), "/score": score_vol})
def build(variant: str, layer: int) -> dict:
    import numpy as np

    for run in {s[0] for s in SOURCES[variant]}:
        if not (Path("/acts") / run / "complete.json").exists():
            raise ValueError(f"{run}: extraction not complete")
    rows, X, L = [], [], []
    for run, arm, world in SOURCES[variant]:
        root = Path("/acts") / run
        for f in sorted((root / "meta" / arm).glob("*.json")):
            m = json.loads(f.read_text())
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            if not m["calls"]:
                continue
            A = np.load(root / "by_layer" / f"L{layer}" / f"{m['chunk']}.npy", mmap_mode="r")
            Bl = np.load(root / "last" / f"L{layer}" / f"{m['chunk']}.npy", mmap_mode="r")
            idx = [m["row_start"] + c["row"] for c in m["calls"]]
            X.append(np.asarray(A[idx]))
            L.append(np.asarray(Bl[idx]))
            rows += [{"world": world, "task": m["task"], "kind": "output_end", "order": [c["turn"], c["call_index"]]} for c in m["calls"]]
    X, L = np.concatenate(X), np.concatenate(L)
    key = lambda r: (r["world"], r["task"], r["order"][0], r["order"][1])  # noqa: E731
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError(f"{variant} L{layer}: duplicate rows")
    ref_rows = json.loads((Path("/score") / REF[variant] / f"on_rows.L{layer}.json").read_text())
    R = np.load(Path("/score") / REF[variant] / f"on_L{layer}.npy", mmap_mode="r")
    ref = {key(r): i for i, r in enumerate(ref_rows) if r["kind"] == "output_end" and r["world"] in ("run1", "run2", "sim", "honest_a", "honest_b")}
    mine = {key(r): i for i, r in enumerate(rows)}
    same_set = set(mine) == set(ref)
    common = sorted(set(mine) & set(ref))
    d = max(float(np.abs(L[mine[k]].astype(np.float32) - np.asarray(R[ref[k]], np.float32)).max()) for k in common) if common else None
    gate = {"rows": len(rows), "ref_rows": len(ref), "same_row_set": same_set, "only_new": len(set(mine) - set(ref)), "only_ref": len(set(ref) - set(mine)),
            "max_abs_diff_last_token": d, "passes": bool(same_set and d is not None and d <= 1e-2)}
    if gate["passes"]:
        o = Path("/score") / OUT / variant
        o.mkdir(parents=True, exist_ok=True)
        with open(o / f".on_L{layer}.npy.tmp", "wb") as fh:
            np.save(fh, X.astype(np.float16))
        os.replace(o / f".on_L{layer}.npy.tmp", o / f"on_L{layer}.npy")
        (o / f"on_rows.L{layer}.json").write_text(json.dumps(rows))
        score_vol.commit()
    return {"variant": variant, "layer": layer, "gate": gate, "by_world": {w: sum(r["world"] == w for r in rows) for w in ("run1", "run2", "sim", "honest_a", "honest_b")}}


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, timeout=2 * 3600, retries=0, volumes={"/score": score_vol}, scaledown_window=5)
def grid(tag: str, variant: str, cell: str, layer: int, attempted: list) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp

    score_vol.reload()
    tw, seed = {"b": ("run1", 20261010), "c": ("sim", 20261013)}[cell]
    fold_of = frozen_folds(json.loads((Path("/root") / T196).read_text()))
    att, t0, out = set(attempted), time.time(), {}
    d = Path("/score") / OUT / variant
    rows = json.loads((d / f"on_rows.L{layer}.json").read_text())
    X = torch.from_numpy(np.load(d / f"on_L{layer}.npy")).cuda()
    runs, n_run = np.unique([f"{r['world']}|{r['task']}" for r in rows], return_inverse=True)
    world = np.array([u.split("|", 1)[0] for u in runs])
    task = np.array([u.split("|", 1)[1] for u in runs])
    r_fold = np.array([fold_of[t] for t in task])
    train = (world == "honest_a") | ((world == tw) & np.isin(runs, sorted(att)))
    y = (world == tw).astype(float)
    Y = np.zeros((len(runs), 1 + N_NULL))
    Y[:, 0] = y
    for c in range(1, 1 + N_NULL):
        rng = np.random.default_rng([seed, c])
        yc = y.copy()
        for t in sorted(set(task[train])):
            j = np.flatnonzero(train & (task == t))
            yc[j] = y[rng.permutation(j)]
        Y[:, c] = yc
    K = Y.shape[1]
    for C in C_GRID:
        p = Path("/score") / f"onpolreply_avg_{tag}" / f"{variant}_{cell}" / f"L{layer}_output_end_C{C:g}.npz"
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
                while True:  # as run_claim_c.grid
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
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        f = modal.Function.from_name(APP, "build")
        calls = [f.spawn(v, l) for v in SOURCES for l in LAYERS]
        print(json.dumps([c.get() for c in calls]))
    elif cmd == "grid":
        lab = json.loads(Path(sys.argv[2]).read_text())["labels"]
        att = {cell: sorted(k for k, v in lab.items() if k.startswith(f"{w}|") and v == "attempted") for cell, w in (("b", "run1"), ("c", "sim"))}
        tag = hashlib.sha256((HERE / "run_avg_cells.py").read_bytes() + json.dumps(att).encode()).hexdigest()[:12]
        f = modal.Function.from_name(APP, "grid")
        print(json.dumps({"tag": tag, "calls": {f"{v}_{c}_L{l}": f.spawn(tag, v, c, l, att[c]).object_id for v in SOURCES for c in ("b", "c") for l in LAYERS}}))
