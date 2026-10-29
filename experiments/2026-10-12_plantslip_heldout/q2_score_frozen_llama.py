"""Row 2, step 3 (RULE.md here): Llama-3.3-70B's frozen probes (run 2026-10-12_143129) score row 3's planted activations dump
(aswin/q2-llama-plant-row3, option B: run_extract_llama_plant.py's controls and planted-e2 runs, all 80 layers at R2J) at the
rule's Llama layers 20, 30, 40, 50, 79. A copy of q2_score_frozen_olmo.py (this folder) with only Llama's names, Volumes and
layers. Its gate reads the real-only files of step 2(c), never the mixed score arrays, and it reads no score file that row 3's
scorer (run_score_plant_llama.py) writes: only the dump's metas and activations, and the frozen probes.
  --mode gate     the 20 controls (real e2, extraction 2026-10-12_q2_plantslip_reply_controls_llama33-70b_v1) must reproduce
                  their stored nested scores in the real-only files within 1e-3 logits at each layer; a failing layer is dropped.
  --mode planted  the 845 planted e2 runs (2026-10-12_q2_plantslip_reply_planted-e2_llama33-70b_v1) at the gate's passing
                  layers; needs a gate record with the same probe hashes. Availability is recorded.
One Modal CPU container; real runs only. Failures re-raise by type name only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-12_plantslip_heldout/q2_score_frozen_llama.py --mode gate|planted --stage <staging record> --real-only <real_only dir> [--gate <gate json>]
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-heldout_llamascore_kill-2026-11-12"
WORK, RUN = "probe-shift_aswin-surya_sub-q2-llama_work_kill-2026-11-12", "2026-10-12_143129"
ACTS = "probe-shift_aswin-surya_sub-q2-llama_acts_kill-2026-11-12"
EXTRACT = {"gate": ("2026-10-12_q2_plantslip_reply_controls_llama33-70b_v1", "real-attack-e2-control", 20),
           "planted": ("2026-10-12_q2_plantslip_reply_planted-e2_llama33-70b_v1", "real-attack-e2-planted", 845)}
LAYERS, TOL, STAGE_SHA = (20, 30, 40, 50, 79), 1e-3, "f1a68c90a5d2"

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
work = modal.Volume.from_name(WORK, version=2).read_only()
acts = modal.Volume.from_name(ACTS, version=2).read_only()


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/work": work, "/acts": acts}, cpu=8, memory=32768, timeout=3600, retries=0)
def score(mode: str, items: list[dict], layers: list[int]) -> dict:
    try:
        return _score(mode, items, layers)
    except Exception as e:  # noqa: BLE001 — sanitized and re-raised, never swallowed
        raise RuntimeError(f"score failed: {type(e).__name__}") from None


def _score(mode, items, layers):
    import numpy as np
    run, arm, _ = EXTRACT[mode]
    root = Path("/acts") / run
    metas = {}
    for j, it in enumerate(items):
        mp = root / "meta" / arm / f"{filename(it['key'])}.json"
        if not mp.exists():  # a failed extraction: its scores stay NaN and it is counted, for the masks
            continue
        m = json.loads(mp.read_text())
        if m["key"] != it["key"] or m["arm"] != arm:
            raise ValueError("meta mismatch")
        metas[j] = (m["chunk"], [m["row_start"] + q["row"] for q in m["positions"] if q["kind"] == "output_end" and q["kept"]])
    out, shas = {}, {}
    for l in layers:
        f = Path("/work") / "runs" / RUN / f"probes_L{l}.npz"
        shas[str(l)] = hashlib.sha256(f.read_bytes()).hexdigest()
        with np.load(f) as zf:
            z = {k: zf[k] for k in zf.files}
        S = np.full((5, 5, len(items)), np.nan); by_chunk = {}
        for j, (c, rows) in metas.items():
            by_chunk.setdefault(c, []).append((j, rows))
        for c, js in by_chunk.items():
            X = np.load(root / "by_layer" / f"L{l}" / f"{c}.npy")  # row 3's dumps hold only real runs (staged planted e2, or e2 controls)
            for j, rows in js:
                if not rows:
                    continue
                x, f_ = X[rows].astype(np.float32), items[j]["fold"]
                for v in range(5):
                    if v != f_:
                        k = f"output_end__task_nested_{f_}_{v}__"
                        S[f_, v, j] = float(np.mean((((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])).astype(np.float64)))
        out[str(l)] = S.tolist()
    return {"scores": out, "probes sha256": shas}


@app.local_entrypoint()
def main(mode: str, stage: str, real_only: str, gate: str = ""):
    import numpy as np
    for f in (Path(__file__).resolve(), HERE / "RULE.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    if not sha(stage).startswith(STAGE_SHA):
        raise SystemExit("not the pinned staging record")
    row3 = (HERE.parent / "2026-10-12_q2_llama_plant").resolve()  # row 3's folder: none of its score files may be read here
    for p_ in [stage, real_only] + ([gate] if gate else []):
        rp = Path(p_).resolve()
        if row3 in rp.parents or "llama_plant_scores" in rp.name or "scores_L" in rp.name and "real_only" not in rp.name:
            raise SystemExit(f"{rp.name}: a row-3 score file or a mixed array is not an input of this scorer")
    if gate and Path(gate).resolve().parent != (HERE / "results").resolve():
        raise SystemExit("the gate record must be this rule's own (results/q2_llama_scores_gate_*.json)")
    ro = Path(real_only); rrec = json.loads((ro / "real_only_record.json").read_text())["readers"]["llama"]
    if sha(ro / "llama_runs.json") != rrec["runs file"]["sha256"] or any(sha(ro / v["file"]) != v["sha256"] for v in rrec["layers"].values()):
        raise SystemExit("the real-only files differ from their record")
    runs = json.loads((ro / "llama_runs.json").read_text())
    if any(r["arm"] == "sim-attack" for r in runs):
        raise SystemExit("a sim-attack run in the real-only runs")
    e2 = {r["key"]: (i, int(r["fold"])) for i, r in enumerate(runs) if r["arm"] == "real-attack-e2"}
    st = json.loads(Path(stage).read_text()); run, arm, expect = EXTRACT[mode]
    keys = sorted(x["key"] for x in (st["controls"] if mode == "gate" else st["planted"]) if x["arm"] == arm)
    if len(keys) != expect or not set(keys) <= set(e2):
        raise SystemExit(f"not {expect} staged keys of the real e2 runs")
    if mode == "planted":
        g = json.loads(Path(gate).read_text())
        if g.get("mode") != "gate" or not g.get("layers passing") or sha(HERE / "results" / g["scores file"]) != g["scores sha256"]:
            raise SystemExit("planted scoring needs a Llama gate record with at least one passing layer")
    items = [{"key": k, "fold": e2[k][1]} for k in keys]
    layers = list(LAYERS) if mode == "gate" else sorted(int(l) for l in g["layers passing"])  # a failing layer is dropped
    res = score.remote(mode, items, layers)
    if mode == "planted" and any(res["probes sha256"][str(l)] != g["probes sha256"][str(l)] for l in layers):
        raise SystemExit("the probes differ from the gate's")
    S = {l: np.array(res["scores"][str(l)]) for l in layers}
    rec = {"rule": "RULE.md step 3 (this folder)", "git": head, "mode": mode, "reader": "Llama-3.3-70B", "frozen run": RUN, "extraction run": run, "arm": arm,
           "layers scored": layers, "runs": len(keys), "probes sha256": res["probes sha256"], "staging sha256": sha(stage), "real-only record sha256": sha(ro / "real_only_record.json")}
    unavailable = {str(l): [it["key"] for j, it in enumerate(items) if not np.isfinite(np.delete(S[l][it["fold"], :, j], it["fold"])).all()] for l in layers}
    rec |= {"runs without four finite nested scores, by layer": {l: len(v) for l, v in unavailable.items()}, "unavailable keys, by layer": unavailable}
    if mode == "gate":
        diffs = {}
        for l in layers:
            stored = np.load(ro / f"llama_scores_L{l}_real_only.npz")["output_end__mean__task_nested"]
            dl = [abs(S[l][it["fold"], v, j] - stored[it["fold"], v, e2[it["key"]][0]]) for j, it in enumerate(items) for v in range(5) if v != it["fold"]]
            diffs[str(l)] = float(max(dl)) if np.isfinite(dl).all() else float("inf")
        rec |= {"gate: max |re-scored - stored| nested score, by layer (tolerance 1e-3)": diffs, "comparisons per layer": len(items) * 4,
                "layers passing": [l for l in diffs if diffs[l] <= TOL], "layers failing": [l for l in diffs if diffs[l] > TOL]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    npz = HERE / "results" / f"q2_llama_scores_{mode}_{stamp}.npz"
    np.savez(npz, keys=np.array(keys), folds=np.array([it["fold"] for it in items]), **{f"L{l}": S[l] for l in layers})
    rec |= {"scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"q2_llama_scores_{mode}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("probes sha256", "unavailable keys, by layer")}, indent=1)); print(dest)
