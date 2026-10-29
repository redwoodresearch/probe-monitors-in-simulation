"""B3 (RULE.md here): fit the training grid (S, R44 x5, M x5, D x5) for one reader at its committed layers and both readouts
(`last`, `mean`), and score every read eligible attack call and every read honest row. Numbers only; no text.

One Modal CPU container reads, read-only, the reader's extraction (run_extract_readers.py: meta/shard*_a*.json, last|mean/L<l>/).
Every shard has exactly one meta file; every staged run is in exactly one; every key the spec reads maps to one row. Fits:
Amendment 3's recipe (unit weights, standardization on the draw's training rows, C 1e-4, fit_dual copied from run_grid.py).
Checks before anything is written: R44_0 fitted twice agrees within 1e-6; every score is finite; the spec's sha256 is the
one passed. Writes results/grid_scores_<reader>_<stamp>.json.

  uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run run_grid_readers.py --reader seed --spec <grid_spec json>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q3sabr_grid_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3sabr_acts_kill-2026-11-13"
HERE = Path(__file__).parent
LAYERS = {"seed": [40, 63], "olmo": [32, 40, 56], "llama": [60, 70], "qwen3": [39, 51, 63]}  # committed layers; primary first (RULE.md, Amendment 2)
C = 1e-4
ARM_WORLD = {"real1": "real", "real2": "run2", "sim": "sim", "honest_b": "honest_b"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy")


def fit_dual(K, y, w, Cs, iters=100, tol=1e-9):
    """Copied from run_grid.py (aswin/q3-sabotage-scaleup @ 7da16618), itself from run_persist_gate.py."""
    import torch
    import torch.nn.functional as F

    L, n, _ = K.shape
    C = torch.tensor(Cs, dtype=K.dtype)[None, :, None]
    Kb, eye = K[:, None], torch.eye(n, dtype=K.dtype)
    a = torch.zeros(L, len(Cs), n, dtype=K.dtype)

    def obj(a):
        f = (Kb @ a[..., None])[..., 0]
        return (C * w * (F.softplus(f) - y * f)).sum(-1) + 0.5 * (a * f).sum(-1)

    for _ in range(iters):
        f = (Kb @ a[..., None])[..., 0]
        p = torch.sigmoid(f)
        r = a - C * w * (y - p)
        res = r.abs().amax(-1)
        if float(res.max()) < tol:
            return a
        delta = torch.linalg.solve(eye + (C * w * p * (1 - p))[..., :, None] * Kb, r[..., None])[..., 0]
        step, j0 = torch.ones(L, len(Cs), 1, dtype=K.dtype), obj(a)
        for _ in range(40):
            worse = obj(a - step * delta) > j0 + 1e-12 * j0.abs()
            if not worse.any():
                break
            step = torch.where(worse[..., None], step / 2, step)
        a = a - step * delta
    raise RuntimeError(f"dual Newton did not converge: max residual {float(res.max()):.3g}")


def load_metas(run: Path, n_shards: int, worlds=None) -> tuple:
    """key -> (run, shard stem, row) over a run's metas (one per shard), and the meta file names."""
    metas = sorted((run / "meta").glob("shard*_a*.json"))
    by_shard = {}
    for f in metas:
        by_shard.setdefault(f.name[:8], []).append(f)
    if len(by_shard) != n_shards or any(len(v) != 1 for v in by_shard.values()):
        raise SystemExit(f"{run.name}: expected one meta per shard for {n_shards} shards, found {sorted((k, len(v)) for k, v in by_shard.items())}")
    where, runs = {}, set()
    for f in metas:
        for r in json.loads(f.read_text())["runs"]:
            if r["key"] in runs:
                raise SystemExit("a run is in two shards")
            runs.add(r["key"])
            for c in r["rows"]:
                k = (ARM_WORLD[r["arm"]], r["run_id"], c["irid"], int(c["call_index"]))
                if worlds is not None and k[0] not in worlds:
                    raise SystemExit(f"{run.name}: an unexpected world {k[0]}")
                if k in where:
                    raise SystemExit("a key maps to two rows")
                where[k] = (run, f.stem, c["row"])
    return where, [f"{run.name}/{f.name}" for f in metas]


@app.function(image=image, cpu=8, memory=65536, timeout=3600, retries=0, volumes={"/acts": acts})
def grid_job(reader: str, spec_sha: str, spec_bytes: bytes, shards: int, repair_run: str = "", repair_shards: int = 0, head: dict = None) -> dict:
    """Without repair_run: one score set per layer and readout. With it (Amendment 2): an "asis" and a "repaired" set on the
    same spec, the second reading every simulated key from the repaired extraction (training and exam alike)."""
    import numpy as np
    import torch

    if hashlib.sha256(spec_bytes).hexdigest() != spec_sha:
        raise SystemExit("spec bytes differ from the sha256 passed")
    spec = json.loads(spec_bytes)
    where, metas = load_metas(Path("/acts") / f"2026-10-13_sabreaders_{reader}_v1", shards)
    variants = {"asis": where}
    if repair_run:
        wr, mr = load_metas(Path("/acts") / repair_run, repair_shards, {"sim"})
        variants["repaired"] = {k: (wr[k] if k[0] == "sim" and k in wr else v) for k, v in where.items()} | {k: v for k, v in wr.items() if k not in where}
        metas += mr
        if spec.get("repair") is not True:
            raise SystemExit("a paired grid needs make_grid_spec's paired spec (simulated calls read in both renderings)")
    ev = [c for c in spec["calls"] if c["read"]]
    ho = [h for h in spec["honest"] if h["read"]]
    by_id = {c["call_id"]: c for c in spec["calls"]}
    need = [tuple(c["key"]) for c in ev] + [tuple(h["key"]) for h in ho]
    fit_keys = [tuple(by_id[i]["key"]) for f in spec["fits"].values() for i in f["pos"] + f["neg"]]
    for vn, w in variants.items():
        if missing := [k for k in need + fit_keys if k not in w or (vn == "repaired" and k[0] == "sim" and w[k][0].name != repair_run)]:
            raise SystemExit(f"{vn}: {len(missing)} read keys have no extracted row")
    names = sorted(spec["fits"], key=lambda n: (n.split("_")[0], n))
    out = {"probes": names, "eval_call_ids": [c["call_id"] for c in ev], "honest_roles": [h["role"] for h in ho],
           "honest_tasks": [h["task"] for h in ho], "scores": {}, "checks": {}}
    for (vn, where), layer, ro in [(v, layer, ro) for v in variants.items() for layer in LAYERS[reader] for ro in ("last", "mean")]:
        if True:
            arrays = {}

            def rows(keys):
                idx = [where[k] for k in keys]
                for run, stem in {(r_, s_) for r_, s_, _ in idx}:
                    if (run, stem) not in arrays:
                        arrays[(run, stem)] = np.load(run / ro / f"L{layer}" / f"{stem}.npy", mmap_mode="r")
                return np.stack([np.asarray(arrays[(r_, s_)][i], np.float64) for r_, s_, i in idx])

            E, H = rows([tuple(c["key"]) for c in ev]), rows([tuple(h["key"]) for h in ho])
            sc, rep = {}, None

            def fit(fd):
                keys = [tuple(by_id[i]["key"]) for i in fd["pos"] + fd["neg"]]
                Xt = rows(keys)
                y = torch.tensor([1.0] * len(fd["pos"]) + [0.0] * len(fd["neg"]), dtype=torch.float64)
                mu, sd = Xt.mean(0), Xt.std(0)
                sd = np.where(sd < 1e-6, 1.0, sd)
                Zt = torch.from_numpy((Xt - mu) / sd)
                a = fit_dual((Zt @ Zt.T)[None], y[None, None, :], torch.ones(len(keys), dtype=torch.float64), [C])
                return (a[0, 0] @ Zt).numpy(), mu, sd

            for n in names:
                beta, mu, sd = fit(spec["fits"][n])
                se, sh = ((E - mu) / sd) @ beta, ((H - mu) / sd) @ beta
                if not (np.isfinite(se).all() and np.isfinite(sh).all()):
                    raise SystemExit(f"{n} L{layer} {ro}: non-finite scores")
                sc[n] = {"eval": se.tolist(), "honest": sh.tolist()}
                if n == "R44_0":
                    b2, mu2, sd2 = fit(spec["fits"][n])
                    rep = float(np.abs(((E - mu2) / sd2) @ b2 - se).max())
            if rep is None or rep > 1e-6:
                raise SystemExit(f"L{layer} {ro}: R44_0 refit differs by {rep}")
            lr = f"L{layer}/{ro}" if not repair_run else f"{vn}:L{layer}/{ro}"
            out["scores"][lr] = sc
            out["checks"][lr] = {"r44_0_refit_max_abs_diff": rep, "eval_rows": len(ev), "honest_rows": len(ho)}
    out["checks"]["metas"] = metas
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = (head or {}) | {"written": stamp, "reader": reader, "spec_sha256": spec_sha, "layers": LAYERS[reader], "primary_layer": LAYERS[reader][0],
                          "C": C, "repair_run": repair_run or None} | out
    name = f"grid_scores_{reader}{'_paired' if repair_run else ''}_{stamp}.json"
    (Path("/acts") / "grid_scores_v1").mkdir(exist_ok=True)
    (Path("/acts") / "grid_scores_v1" / name).write_text(json.dumps(out))  # on the Volume first (d9: a lost client loses nothing)
    acts.commit()
    return out | {"_name": name}


@app.local_entrypoint()
def main(reader: str, spec: str, shards: int, repair_run: str = "", repair_shards: int = 0):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--", Path(__file__).name):
        raise SystemExit("commit this script before it runs")
    b = Path(spec).read_bytes()
    sp = json.loads(b)
    if sp["reader"] != reader:
        raise SystemExit("spec is another reader's")
    res = grid_job.remote(reader, hashlib.sha256(b).hexdigest(), b, shards, repair_run, repair_shards, {"git": git("rev-parse", "HEAD"), "spec": Path(spec).name})
    dst = HERE / "results" / res.pop("_name")
    with dst.open("x") as fh:
        fh.write(json.dumps(res))
    print(json.dumps(res["checks"], indent=1), "\n->", dst)
