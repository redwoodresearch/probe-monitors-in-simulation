"""PLAN.md section 3, Modal CPU: the frozen white-box-dataset probe's scores on stored call_end readings, plus 100 random
unit directions (seed 20261013, as the family) at the same layer. Keys and numbers only: no stage-1 label is read here,
and no stage-2 task or sealed run is touched. Run only after the probe's freeze (sha256 on S3 and the manifest
committed).

The probe (run_whitebox.fit, saved only when the validity gate passed): layer L, score = ((h - mean) / std) . coef +
intercept, written as h . (coef / std) + (intercept - (mean / std) . coef). Our layer L is the output of decoder block L
(run_extract_replay.py: base.layers[l] forward hook), the same convention as the probe's training activations.

  part 1: the rows in rows_part1.json (the family's build_rows.py, sha256 db2ba665...): stage 1's listed real calls,
          honest A calibration calls, honest B calls.
  part 2: claim C's swapped cohort via score_part2_generic (byte-identical to the family's, blob 469349bc), after its
          layer-39 build matches claim C's cache. Per run: the mean over call_ends and the first command.

  modal deploy experiments/2026-10-11_whitebox_dataset_probe/score_whitebox.py   (then spawn `score` with the rows and the probe's sha256)
"""

import hashlib
import io
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-wbprobe_score_kill-2026-11-10"
SCORE_VOL, ACTS_VOL = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
PROBE, OUT, RANDOM_SEED, N_RANDOM, D_MODEL = "whitebox_probe_v1/probe.npz", "whitebox_score_v1", 20261013, 100, 5120

app = modal.App(APP)
score_vol, acts_vol = (modal.Volume.from_name(n, version=2) for n in (SCORE_VOL, ACTS_VOL))
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").add_local_file(HERE / "score_part2_generic.py", "/root/score_part2_generic.py")


def load_probe(sha: str):
    import numpy as np

    b = (Path("/score") / PROBE).read_bytes()
    if hashlib.sha256(b).hexdigest() != sha:
        raise ValueError(f"probe sha256 {hashlib.sha256(b).hexdigest()} != frozen {sha}")
    z = np.load(io.BytesIO(b), allow_pickle=False)
    coef, m, s = (z[k].astype(np.float64) for k in ("coef", "mean", "std"))
    if any(a.shape != (D_MODEL,) for a in (coef, m, s)) or not (np.isfinite(coef).all() and np.isfinite(m).all() and (s > 0).all()):
        raise ValueError("probe arrays")
    w, c0 = coef / s, float(z["intercept"]) - float((m / s) @ coef)
    x = np.random.default_rng(0).normal(size=D_MODEL)
    if abs(((x - m) / s) @ coef + float(z["intercept"]) - (x @ w + c0)) > 1e-6 * max(1.0, abs(x @ w)):
        raise ValueError("probe score rewrite")
    return int(z["layer"]), w.astype(np.float32), c0


@app.function(image=image, cpu=16, memory=65536, timeout=1800, retries=0, scaledown_window=2,
              volumes={"/score": score_vol, "/acts": acts_vol.read_only()})
def score(rows: list, probe_sha256: str) -> dict:
    import sys

    import numpy as np

    sys.path.insert(0, "/root")
    import score_part2_generic as G

    for v in (score_vol, acts_vol):
        v.reload()
    out = Path("/score") / OUT
    if out.exists():
        raise FileExistsError(out)
    L, w, c0 = load_probe(probe_sha256)
    rnd = np.random.default_rng(RANDOM_SEED).normal(size=(N_RANDOM, D_MODEL)).astype(np.float32)
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    names, M = ["whitebox"] + [f"random_{i}" for i in range(N_RANDOM)], np.stack([w, *rnd], 1)
    out.mkdir(parents=True)
    res = {"layer": L, "probe_sha256": probe_sha256, "constant": c0}
    # part 1
    X = np.empty((len(rows), D_MODEL), np.float32)
    groups = {}
    for i, r in enumerate(rows):
        x, _arm, chunk, g = r["loc"]
        groups.setdefault((x, chunk), []).append((i, g))
    for (x, chunk), items in sorted(groups.items()):
        a = np.load(Path("/acts") / x / "by_layer" / f"L{L}" / f"{chunk}.npy")  # whole file, then index
        X[[i for i, _ in items]] = a[[g for _, g in items]]
    if not np.isfinite(X).all():
        raise ValueError(f"non-finite readings at L{L}")
    P = X @ M
    P[:, 0] += c0
    np.savez(out / f"part1_L{L}.npz", scores=P.astype(np.float32), ids=np.array([r["id"] for r in rows]), columns=np.array(names))
    res["part1"] = {"rows": len(rows)}
    # part 2: claim C's swapped cohort, after the layer-39 build reproduces the cache
    b_rows, b_X = G.build(39)
    c_rows, c_X = G.cached(39)
    key = lambda rr: [(q["world"], q["task"], tuple(q["order"])) for q in rr]  # noqa: E731
    bk, ck = key(b_rows), key(c_rows)
    if sorted(bk) != sorted(ck):
        raise ValueError(f"layer-39 build rows differ from the cache: {len(bk)} vs {len(ck)}")
    bi, ci = np.argsort(np.array([str(k) for k in bk])), np.argsort(np.array([str(k) for k in ck]))
    diff = np.abs(b_X[bi] - c_X[ci])
    res["layer39_check"] = {"rows": len(bk), "max_abs_diff": float(diff.max()), "finite": bool(np.isfinite(diff).all())}
    if not res["layer39_check"]["finite"] or res["layer39_check"]["max_abs_diff"] > 1e-3:
        raise ValueError(f"layer-39 build differs from the cache: {res['layer39_check']}")
    d_runs, d_z = G.detector()
    prow, X2 = G.build(L)
    if not np.isfinite(X2).all():
        raise ValueError(f"non-finite part-2 readings at L{L}")
    P2 = X2 @ M
    P2[:, 0] += c0
    runs, inv = np.unique([f"{q['world']}|{q['task']}" for q in prow], return_inverse=True)
    mean = np.zeros((len(runs), M.shape[1]))
    np.add.at(mean, inv, P2)
    mean /= np.bincount(inv)[:, None]
    first = np.zeros((len(runs), M.shape[1]))
    order = np.lexsort((np.array([q["order"][1] for q in prow]), np.array([q["order"][0] for q in prow]), inv))
    seen = set()
    for i in order:
        if inv[i] not in seen:
            seen.add(inv[i])
            first[inv[i]] = P2[i]
    part2 = {"layer": L, "rows": len(prow), "runs": runs.tolist(), "columns": names, "mean": mean.tolist(), "first": first.tolist(),
             "calls_per_run": np.bincount(inv).tolist(), "detector_runs": d_runs,
             "detector_score": [None if not np.isfinite(x) else float(x) for x in d_z], "layer39_check": res["layer39_check"]}
    (out / f"part2_L{L}.json").write_text(json.dumps(part2))
    res["part2"] = {"rows": len(prow), "runs": len(runs)}
    (out / "summary.json").write_text(json.dumps(res))
    score_vol.commit()
    return res


@app.function(image=image, timeout=300, retries=0, scaledown_window=2, volumes={"/score": score_vol})
def fetch(name: str) -> bytes:
    """A score file (numbers only) for the local analysis."""
    score_vol.reload()
    return (Path("/score") / OUT / name).read_bytes()
