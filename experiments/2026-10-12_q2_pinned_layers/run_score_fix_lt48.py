"""Addendum 2 (PLAN.md here, committed before this ran): #36's frozen nested probes (run 2026-10-06_181548) score fix939's 823
repaired runs at layer 48, from fix939's dump (layers 48 and 63). Gate: the same code at layer 63 reproduces fix939's stored
repaired nested scores within 1e-3, or nothing is written. One Modal CPU container; saved activations and probes only.
The scoring is ../2026-10-12_q2_meanpool_36/run_score_fix_meanpool.py's (at 08236357) with the readout (output_end), the
dump, the work Volume and the layers changed.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-12_q2_pinned_layers/run_score_fix_lt48.py --lt-dir <#36 run_scores_2026-10-06_181548> --fix <fix939 per-run jsonl>
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-pinned_lt48score_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
RUN_ID, FIXRUN, FIX_ARM, KIND = "2026-10-06_181548", "2026-10-11_q3_fix939_qwen3-32b_l48-63_v1", "sim-attack-note-sr-both", "output_end"
LAYERS, TOL = (48, 63), 1e-3
PIN = {"fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd",
       "runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
work = modal.Volume.from_name(WORK, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, volumes={"/acts": acts, "/work": work}, cpu=8, memory=32768, timeout=3600, retries=0)
def score(sim: list[dict]) -> dict:
    """sim: the as-simulated runs, {filename, fold}. Per layer, nested scores [5, 5, len(sim)] for the repaired runs (NaN
    where a run has no repaired version). Failures re-raise by type name only."""
    try:
        return _score(sim)
    except Exception as e:  # noqa: BLE001 — sanitized and re-raised, never swallowed
        raise RuntimeError(f"score failed: {type(e).__name__}") from None


def _score(sim):
    import numpy as np
    d = Path("/acts") / FIXRUN
    metas = {}
    for j, r in enumerate(sim):
        p = d / "meta" / FIX_ARM / f"{r['filename']}.json"
        if p.exists():
            m = json.loads(p.read_text())
            rows = [m["row_start"] + q["row"] for q in m["positions"] if q["kind"] == KIND and q["kept"]]
            metas[j] = (m["chunk"], rows, m["key"])
    out = {}
    for layer in LAYERS:
        with np.load(Path("/work") / "runs" / RUN_ID / f"probes_L{layer}.npz") as zf:
            z = {k: zf[k] for k in zf.files}
        S = np.full((5, 5, len(sim)), np.nan); by_chunk = {}
        for j, (c, rows, _) in metas.items():
            by_chunk.setdefault(c, []).append((j, rows))
        for c, items in by_chunk.items():
            X = np.load(d / "by_layer" / f"L{layer}" / f"{c}.npy")  # whole file: row-indexed mmap is slow on a Volume
            for j, rows in items:
                if not rows:
                    continue
                x, f = X[rows].astype(np.float32), sim[j]["fold"]
                for v in range(5):
                    if v != f:
                        k = f"{KIND}__task_nested_{f}_{v}__"
                        S[f, v, j] = float(np.mean((((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])).astype(np.float64)))
        out[str(layer)] = S.tolist()
    return {"fix keys": sorted(m[2] for m in metas.values()), "scores": out}


@app.local_entrypoint()
def main(lt_dir: str, fix: str):
    import numpy as np
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    d, fix_p = Path(lt_dir), Path(fix)
    if sha(fix_p) != PIN[fix_p.name] or sha(d / "runs.json") != PIN["runs.json"]:
        raise SystemExit("not the pinned fix939 per-run file or #36's runs.json")
    runs = json.loads((d / "runs.json").read_text())
    sim = [{"i": i, "filename": r["filename"], "fold": int(r["fold"]), "key": r["key"]} for i, r in enumerate(runs) if r["arm"] == "sim-attack"]
    stored = {}
    for line in fix_p.open():
        r = json.loads(line)
        if r["label"] == "FIX":
            stored[r["key"]] = {int(v): s for v, s in r["nested_scores"]["output_end"].items()}
    if len(sim) != 939 or len(stored) != 823 or not set(stored) <= {r["key"] for r in sim}:
        raise SystemExit("not the 939 as-simulated runs and fix939's 823 repaired keys")
    res = score.remote([{k: r[k] for k in ("filename", "fold")} for r in sim])
    if res["fix keys"] != sorted(stored):
        raise SystemExit(f"the dump's repaired runs are not fix939's 823: {len(res['fix keys'])} found")
    S = {l: np.array(res["scores"][str(l)]) for l in LAYERS}
    diffs = [abs(S[63][r["fold"], v, j] - s) for j, r in enumerate(sim) if r["key"] in stored for v, s in stored[r["key"]].items() if s is not None]
    if len(diffs) != 823 * 4 or not np.isfinite(diffs).all() or max(diffs) > TOL:
        raise SystemExit(f"gate failed: {len(diffs)} comparisons, max {max(diffs) if diffs else None}")
    idx = np.array([r["i"] for r in sim]); has_fix = np.array([r["key"] in stored for r in sim])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    npz = HERE / "results" / f"fix_scores_lt48_{stamp}.npz"
    np.savez(npz, run_index=idx, has_fix=has_fix, L48=S[48], L63=S[63])
    rec = {"plan": "PLAN.md addendum 2 (this folder)", "git": head, "run": RUN_ID, "extraction": FIXRUN, "arm": FIX_ARM,
           "inputs sha256": {fix_p.name: sha(fix_p), "runs.json": sha(d / "runs.json")},
           "gate: L63 re-scored vs fix939's stored repaired nested scores": {"comparisons": len(diffs), "max abs diff": float(max(diffs)), "tolerance": TOL},
           "repaired runs scored": int(has_fix.sum()), "scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"fix_scores_lt48_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
