"""Row 2, steps 2-3 (RULE.md here, Amendment 1, item 2): the frozen-probe scorer for the Qwen3-32B configurations. The real-only
fit's per-fold nested probes (q2_run_q3_attack_heldout.py's run, `runs/<run>/probes_L<l>.npz` on its work Volume) score
staged real e2 transcripts from q2_run_extract_heldout.py's runs: each run's kept output_end rows, ((x - mean) / std) . w + b,
averaged over its rows, as the runner pools them (the scoring of ../2026-10-12_q2_pinned_layers/run_score_fix_lt48.py).
  --mode gate     the 20 controls; every nested score must match the fit's stored nested score for the same real e2 run
                  within 1e-3 logits, at every layer and nested model (Amendment 1's score-level gate), or nothing is written.
  --mode planted  the 845 planted e2 runs; refuses unless a passing gate record for the same fit and probes is given.
One Modal CPU container; real runs only (the extraction runs hold no simulated transcript). Failures re-raise by type name.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-12_plantslip_heldout/q2_score_frozen_qwen3.py --mode gate|planted --fit-run <run id> --stage <staging record> [--gate <gate json>]
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-heldout_score_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q2-heldout_work_kill-2026-11-12"
RUNS = {"gate": ("2026-10-12_q2_heldout_controls-e2_qwen3-32b_l20-60_v1", "real-attack-e2-control", 20),
        "planted": ("2026-10-12_q2_heldout_planted-e2_qwen3-32b_l20-60_v1", "real-attack-e2-planted", 845)}
KIND, LAYERS, TOL, STAGE_SHA = "output_end", (20, 28, 36, 44, 52, 60), 1e-3, "f1a68c90a5d2"

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
work = modal.Volume.from_name(WORK, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/acts": acts, "/work": work}, cpu=8, memory=32768, timeout=3600, retries=0)
def score(mode: str, fit_run: str, items: list[dict], layers: list[int]) -> dict:
    try:
        return _score(mode, fit_run, items, layers)
    except Exception as e:  # noqa: BLE001 — sanitized and re-raised, never swallowed
        raise RuntimeError(f"score failed: {type(e).__name__}") from None


def _score(mode, fit_run, items, layers):
    import numpy as np
    run, arm, _ = RUNS[mode]
    d = Path("/acts") / run
    metas = {}
    for j, it in enumerate(items):
        mp = d / "meta" / arm / f"{filename(it['key'])}.json"
        if not mp.exists():  # a failed extraction: its scores stay NaN and it is counted, for the masks
            continue
        m = json.loads(mp.read_text())
        if m["key"] != it["key"]:
            raise ValueError("meta key mismatch")
        metas[j] = (m["chunk"], [m["row_start"] + q["row"] for q in m["positions"] if q["kind"] == KIND and q["kept"]])
    out, probe_sha = {}, {}
    for layer in layers:
        p = Path("/work") / "runs" / fit_run / f"probes_L{layer}.npz"
        probe_sha[str(layer)] = hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p) as zf:
            z = {k: zf[k] for k in zf.files}
        S = np.full((5, 5, len(items)), np.nan); by_chunk = {}
        for j, (c, rows) in metas.items():
            by_chunk.setdefault(c, []).append((j, rows))
        for c, js in by_chunk.items():
            X = np.load(d / "by_layer" / f"L{layer}" / f"{c}.npy")  # this run's chunks hold only its own real e2 runs
            for j, rows in js:
                if not rows:
                    continue
                x, f = X[rows].astype(np.float32), items[j]["fold"]
                for v in range(5):
                    if v != f:
                        k = f"{KIND}__task_nested_{f}_{v}__"
                        S[f, v, j] = float(np.mean((((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])).astype(np.float64)))
        out[str(layer)] = S.tolist()
    return {"scores": out, "probes sha256": probe_sha, "runs with no kept row": sum(not r for _, r in metas.values())}


@app.local_entrypoint()
def main(mode: str, fit_run: str, stage: str, gate: str = ""):
    import numpy as np
    for f in (Path(__file__).resolve(), HERE / "RULE.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    if not sha(stage).startswith(STAGE_SHA):
        raise SystemExit("not the pinned staging record")
    st = json.loads(Path(stage).read_text()); run, arm, expect = RUNS[mode]
    keys = sorted(x["key"] for x in (st["controls"] if mode == "gate" else st["planted"]) if x["arm"] == arm)
    sdir = HERE / "results" / f"run_scores_{fit_run}"  # the fit's collected scores (real arms only)
    runs = json.loads((sdir / "runs.json").read_text())
    if any(r["arm"] == "sim-attack" for r in runs):
        raise SystemExit("the fit's runs hold a sim-attack run")
    e2 = {r["key"]: (i, int(r["fold"])) for i, r in enumerate(runs) if r["arm"] == "real-attack-e2"}
    if len(keys) != expect or not set(keys) <= set(e2):
        raise SystemExit(f"not {expect} staged keys of the fit's real e2 runs")
    if mode == "planted":
        g = json.loads(Path(gate).read_text())
        if g.get("mode") != "gate" or g.get("fit run") != fit_run or not g.get("layers passing") or g.get("probes sha256") is None:
            raise SystemExit("planted scoring needs a gate record for this fit with at least one passing layer")
        if sha(HERE / "results" / g["scores file"]) != g["scores sha256"]:
            raise SystemExit("the gate record's scores file differs from its sha256")
    items = [{"key": k, "fold": e2[k][1]} for k in keys]
    layers = list(LAYERS) if mode == "gate" else sorted(int(l) for l in g["layers passing"])  # Amendment 1: a failing layer is dropped
    res = score.remote(mode, fit_run, items, layers)
    if mode == "planted" and any(res["probes sha256"][str(l)] != g["probes sha256"][str(l)] for l in layers):
        raise SystemExit("the probes differ from the gate's")
    S = {l: np.array(res["scores"][str(l)]) for l in layers}
    rec = {"rule": "RULE.md step 2 / Amendment 1 (this folder)", "git": head, "mode": mode, "fit run": fit_run, "extraction run": run, "arm": arm, "layers scored": layers,
           "runs": len(keys), "runs with no kept row": res["runs with no kept row"], "probes sha256": res["probes sha256"], "staging sha256": sha(stage)}
    if mode == "gate":
        diffs = {}
        for l in LAYERS:
            stored = np.load(sdir / f"scores_L{l}.npz")[f"{KIND}__mean__task_nested"]
            dl = [abs(S[l][it["fold"], v, j] - stored[it["fold"], v, e2[it["key"]][0]]) for j, it in enumerate(items) for v in range(5) if v != it["fold"]]
            if not np.isfinite(dl).all():
                raise SystemExit(f"L{l}: a non-finite score")
            diffs[str(l)] = float(max(dl))
        rec |= {"gate: max |re-scored - stored| nested score, by layer (tolerance 1e-3)": diffs, "comparisons per layer": len(items) * 4,
                "layers passing": [l for l in diffs if diffs[l] <= TOL], "layers failing": [l for l in diffs if diffs[l] > TOL],
                "gate passes": all(v <= TOL for v in diffs.values())}
    # availability, for the retained-cell masks: a run whose own-fold nested scores are not all finite at a layer
    unavailable = {str(l): [it["key"] for j, it in enumerate(items) if not np.isfinite(np.delete(S[l][it["fold"], :, j], it["fold"])).all()] for l in layers}
    rec["runs without four finite nested scores, by layer"] = {l: len(v) for l, v in unavailable.items()}
    rec["unavailable keys, by layer"] = unavailable
    if mode == "gate" and any(unavailable.values()):
        raise SystemExit("a control lacks finite nested scores")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    npz = HERE / "results" / f"q2_frozen_scores_{mode}_{stamp}.npz"
    np.savez(npz, keys=np.array(keys), folds=np.array([it["fold"] for it in items]), **{f"L{l}": S[l] for l in layers})
    rec |= {"scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"q2_frozen_scores_{mode}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "probes sha256"}, indent=1)); print(dest)
