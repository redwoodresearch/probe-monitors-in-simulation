"""PLAN.md section 9, Modal CPU: the frozen generic direction on claim C's swapped call_end readings, and the claim C
detector's stored out-of-fold run scores, per run. Keys and numbers only; no stage-1 label, no stage-2 task, no zoo C.

Readings at the chosen layer L: claim C's cache (`/score/claimc_cache_v1/on_L<L>.npy`, call_end rows) if L is 39, 51 or
63. Otherwise built the same way as run_claim_c.cache (its SWAPPED extractions; counted runs only: not redrawn, never
a rerun), with honest A and B from the four scrub-1 base extractions' honest arms, after the same build at layer 39 is
checked row for row against the cache. Per run (world|task): the mean of its call_end projections (claim C's
aggregation) and its first command's projection (lowest turn, then call index), onto v, v_res, r and 100 random unit
directions (seed 20261013). The detector: analyze_claim_c.select's command-end selection on claim C's records (tag
fb82f67348ac), column 0 (the real labels).

  modal deploy experiments/2026-10-11_generic_contrast_probe/score_part2.py
  (spawn `score` with the layer and the direction file's name on the score Volume)
"""

import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-gcp_part2_kill-2026-11-10"
SCORE_VOL, ACTS_VOL = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
CACHE, OUT, TAG = "claimc_cache_v1", "generic_contrast_v1", "fb82f67348ac"
BASES = ["2026-10-09_rerun_strict_segends_v1_scrub1", "2026-10-10_scaleup_canary_segends_v1_scrub1",
         "2026-10-10_scaleup_l2_segends_v1_scrub1", "2026-10-10_scaleup_topup_segends_v1_scrub1"]
SWAPPED = [(f"{b}_swapsys", "attack", "run1") for b in BASES] + [("2026-10-10_second_attack_segends_v1_scrub1_swapsys", "attack", "run2")] \
          + [(f"{b}_swapsys2", "sim_attack@september", "sim") for b in BASES]
HONEST = [(b, arm, arm) for b in BASES for arm in ("honest_a", "honest_b")]
CACHED, C_GRID, RANDOM_SEED, N_RANDOM = [39, 51, 63], [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 20261013, 100

app = modal.App(APP)
score_vol, acts_vol = modal.Volume.from_name(SCORE_VOL, version=2), modal.Volume.from_name(ACTS_VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


def build(layer: int):
    """call_end rows (world, task, order) and their readings at `layer`, from the extractions."""
    import numpy as np

    rows, parts, counted_all = [], [], {}
    for run, arm, world in SWAPPED + HONEST:
        root = Path("/acts") / run
        done = json.loads((root / "complete.json").read_text())
        files = sorted((root / "meta" / arm).glob("*.json"))
        metas = [json.loads(f.read_text()) for f in files]
        if not metas or (arm in ("attack", "sim_attack@september") and len(list((root / "meta").glob("*/*.json"))) != done["epochs"]):
            raise ValueError(f"{run} {arm}: metas incomplete")
        picks, counted = [], {}
        for m in metas:
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            counted.setdefault(m["task"], set()).add(m["run"])
            for c in m["calls"]:
                if c.get("kind") == "call_end" and c.get("row") is not None:
                    rows.append({"world": world, "task": m["task"], "order": [c["turn"], c.get("call_index", 0)]})
                    picks.append((m["chunk"], m["row_start"] + c["row"]))
        for t, v in counted.items():
            counted_all.setdefault((world, t), set()).update(v)
        if any(len(v) > 1 for v in counted_all.values()):
            raise ValueError(f"{run} {arm}: several counted runs for one world and task")
        X = np.empty((len(picks), 5120), np.float32)
        for chunk in sorted({p[0] for p in picks}):
            a = np.load(root / "by_layer" / f"L{layer}" / f"{chunk}.npy")  # whole file, then index (row mmap reads are slow)
            idx = [i for i, p in enumerate(picks) if p[0] == chunk]
            X[idx] = a[[picks[i][1] for i in idx]]
        parts.append(X)
    return rows, np.concatenate(parts)


def cached(layer: int):
    import numpy as np

    d = Path("/score") / CACHE
    rows = json.loads((d / f"on_rows.L{layer}.json").read_text())
    keep = [i for i, r in enumerate(rows) if r["kind"] == "call_end"]
    return [{k: rows[i][k] for k in ("world", "task", "order")} for i in keep], np.load(d / f"on_L{layer}.npy")[keep].astype(np.float32)


def detector():
    """analyze_claim_c.select, command-end part, column 0: per run the out-of-fold score of the selected candidate."""
    import numpy as np

    cands = [(l, "call_end", C) for C in C_GRID for l in CACHED]
    R = {c: dict(np.load(Path("/score") / f"claimc_{TAG}" / f"L{c[0]}_{c[1]}_C{c[2]:g}.npz")) for c in cands}
    runs, r0 = R[cands[0]]["on_runs"], R[cands[0]]
    if any(not all(np.array_equal(r[k], r0[k]) for k in ("on_runs", "Y", "on_fold", "train")) for r in R.values()):
        raise ValueError("records disagree")
    ok_f, ok_v = (np.stack([R[c][f"grad_{n}"] <= R[c]["tol"] for c in cands]) for n in ("full", "val"))
    share = float(1 - np.concatenate([ok_f.ravel(), ok_v.ravel()]).mean())
    V = np.stack([np.where(ok_f[i] & ok_v[i] & np.isfinite(R[c]["val_auroc"]), R[c]["val_auroc"], -np.inf) for i, c in enumerate(cands)])
    pick, usable = V.argmax(0), np.isfinite(V.max(0))
    if share > 0.05 or not usable.all():  # select's validity gates (MAX_FAIL 0.05)
        raise ValueError(f"detector selection invalid: unconverged share {share:.4f}, unusable fold-columns {int((~usable).sum())}")
    complete = np.all(np.stack([np.isfinite(R[c]["on_score"]).all(1) for c in cands]), 0)
    z = np.full(len(runs), np.nan)
    for f in range(5):
        r = R[cands[pick[f, 0]]]
        mine = (r["on_fold"] == f) & complete
        z[mine] = r["on_score"][mine, 0]
    return runs.astype(str).tolist(), z


@app.function(image=image, cpu=16, memory=65536, timeout=3600, retries=0, volumes={"/score": score_vol.read_only(), "/acts": acts_vol.read_only()})
def score(layer: int) -> dict:
    import numpy as np

    score_vol.reload()
    acts_vol.reload()
    out = Path("/score") / OUT
    v, r = np.load(out / "v.npy")[layer].astype(np.float32), np.load(out / "r.npy")[layer].astype(np.float32)
    r_hat = r / np.linalg.norm(r)
    v_res = v - float(v @ r_hat) * r_hat
    rnd = np.random.default_rng(RANDOM_SEED).normal(size=(N_RANDOM, v.size)).astype(np.float32)
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    D = np.stack([v, v_res, r, *rnd], 1)  # [d, 3 + 100]
    check = None
    if layer in CACHED:
        rows, X = cached(layer)
    else:  # the layer-39 build must reproduce the cache before another layer is trusted
        b_rows, b_X = build(39)
        c_rows, c_X = cached(39)
        key = lambda rr: [(q["world"], q["task"], tuple(q["order"])) for q in rr]  # noqa: E731
        bk, ck = key(b_rows), key(c_rows)
        if sorted(bk) != sorted(ck):
            raise ValueError(f"layer-39 build rows differ from the cache: {len(bk)} vs {len(ck)}")
        bi, ci = np.argsort(np.array([str(k) for k in bk])), np.argsort(np.array([str(k) for k in ck]))
        check = {"rows": len(bk), "max_abs_diff": float(np.abs(b_X[bi] - c_X[ci]).max())}
        if check["max_abs_diff"] > 1e-3:
            raise ValueError(f"layer-39 build differs from the cache: {check}")
        rows, X = build(layer)
    if not np.isfinite(X).all():
        raise ValueError("non-finite readings")
    P = X @ D
    runs, inv = np.unique([f"{q['world']}|{q['task']}" for q in rows], return_inverse=True)
    mean = np.zeros((len(runs), D.shape[1]))
    np.add.at(mean, inv, P)
    mean /= np.bincount(inv)[:, None]
    first = np.zeros((len(runs), D.shape[1]))
    order = np.lexsort((np.array([q["order"][1] for q in rows]), np.array([q["order"][0] for q in rows]), inv))
    seen = set()
    for i in order:
        if inv[i] not in seen:
            seen.add(inv[i])
            first[inv[i]] = P[i]
    d_runs, d_z = detector()
    return {"layer": layer, "rows": len(rows), "runs": runs.tolist(), "columns": ["v", "v_res", "r"] + [f"random_{i}" for i in range(N_RANDOM)],
            "mean": mean.tolist(), "first": first.tolist(), "calls_per_run": np.bincount(inv).tolist(),
            "detector_runs": d_runs, "detector_score": [None if not np.isfinite(x) else float(x) for x in d_z], "layer39_check": check}
