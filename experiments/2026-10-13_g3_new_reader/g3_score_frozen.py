"""G3 step 4's score-reproduction gate and step 5's planted scoring (RULE.md here): a copy of
../2026-10-12_plantslip_heldout/q2_score_frozen_olmo.py (at 62edb8e8) for Seed-OSS-36B. Its scoring is unchanged (each
nested model (f, v), v != f, on the run's kept output_end rows, ((x - mean) / std) . w + b, averaged). Changed: G3's fit
(--fit-run, its run record's archived probe hashes checked against the probes read), dumps and Volumes, all 7 layers, and the
gate's reference: the fit's own stored nested scores (results/run_scores_<fit>/scores_L<l>.npz and runs.json, checked against
the run record's archive; G3's fit holds real arms only, so no real-only preparation is needed).
  --mode gate     the 20 controls (real e2, 2026-10-13_g3_controls_seed-oss-36b_v1) must reproduce the fit's stored nested
                  scores within 1e-3 logits at every layer (RULE.md step 4; G3 stops if any layer fails).
  --mode planted  the 845 planted e2 runs (2026-10-13_g3_planted_seed-oss-36b_v1) at every layer; needs a passing gate record
                  with the same probe hashes. Availability (runs without four finite nested scores) is recorded.
Amendment 2's row drops: the step-2 render gate (--render-gate) must record no left-out row (G3's render had none), else this
script stops (the drop path is not implemented here). One Modal CPU container; real runs only. Failures re-raise by type only.
Pre-run fixes: the fit must be a full fit of the committed config/q3_attack_g3.yaml (its recorded sha256; r3ocj); step 3's
activation gate must have passed at every layer (--act-gate, r38mf); the planted record names its gate record (r38mp); an
"unevaluable" status is kept beside the reproduction result (r4PYs).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_g3_new_reader/g3_score_frozen.py --mode gate|planted --stage <staging record> --fit-run <id> \\
      --render-gate results/render_g3_gate_<stamp>.json --act-gate results/g3_gate_activations_<stamp>.json [--gate <gate json>]
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-g3_score_kill-2026-11-12"
WORK = "probe-shift_aswin-surya_sub-q2-g3_work_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"
EXTRACT = {"gate": ("2026-10-13_g3_controls_seed-oss-36b_v1", "real-attack-e2-control", 20),
           "planted": ("2026-10-13_g3_planted_seed-oss-36b_v1", "real-attack-e2-planted", 845)}
LAYERS, TOL, STAGE_SHA = (16, 24, 32, 40, 48, 56, 63), 1e-3, "f1a68c90a5d2"

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
work = modal.Volume.from_name(WORK, version=2).read_only()
acts = modal.Volume.from_name(ACTS, version=2).read_only()


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/work": work, "/acts": acts}, cpu=8, memory=32768, timeout=3600, retries=0)
def score(mode: str, items: list[dict], layers: list[int], run_id: str) -> dict:
    try:
        return _score(mode, items, layers, run_id)
    except Exception as e:  # noqa: BLE001 — sanitized and re-raised, never swallowed
        raise RuntimeError(f"score failed: {type(e).__name__}") from None


def _score(mode, items, layers, RUN):
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
        idx = json.loads((root / "chunks" / f"{m['chunk']}.json").read_text())  # G3: the chunk holds only this job's arm,
        if any(t["arm"] != arm for t in idx["transcripts"]):                   # checked before any array is read
            raise ValueError("a chunk holds another arm")
        blk = [t for t in idx["transcripts"] if t["key"] == it["key"]]
        rows_ = [m["row_start"] + q["row"] for q in m["positions"] if q["kind"] == "output_end" and q["kept"]]
        if len(blk) != 1 or blk[0]["row_start"] != m["row_start"] or any(not (m["row_start"] <= r < m["row_start"] + blk[0]["n_rows"]) for r in rows_):
            raise ValueError("a run's rows lie outside its own block")
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
            X = np.load(root / "by_layer" / f"L{l}" / f"{c}.npy")  # G3's planted and controls dumps hold only real runs
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
def main(mode: str, stage: str, fit_run: str, render_gate: str, act_gate: str, gate: str = ""):
    import numpy as np
    for f in (Path(__file__).resolve(), HERE / "RULE.md", HERE / "config" / "q3_attack_g3.yaml"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    if not sha(stage).startswith(STAGE_SHA):
        raise SystemExit("not the pinned staging record")
    rg = json.loads(Path(render_gate).read_text())
    if rg.get("scope") != "step 2" or rg.get("gate") != "pass" or rg["matched (real e2 and planted e2)"]["rows left out of both (union)"] \
            or any(rg["frozen row drops (for step 6)"].values()) or rg["dropped cells, count"]:
        raise SystemExit("the render gate is not a passing step-2 gate with no left-out row (the drop path is not implemented)")
    ag = json.loads(Path(act_gate).read_text())  # r38mf: step 3's activation gate, passed at every layer on G3's real dump
    if ag.get("gate") != "pass" or sorted(map(int, ag["layers passing"])) != list(LAYERS) or ag.get("dump") != "2026-10-13_g3_real_seed-oss-36b_v1":
        raise SystemExit("the activation gate did not pass at every layer of G3's real dump")
    rec_fit = json.loads((HERE / "results" / f"run_{fit_run}.json").read_text()); cf = rec_fit["config"]
    if rec_fit.get("test") is not False or cf.get("config file") != "q3_attack_g3.yaml" or cf.get("config sha256 (read)") != sha(HERE / "config" / "q3_attack_g3.yaml"):
        raise SystemExit("the fit is not a full fit of the committed config/q3_attack_g3.yaml (r3ocj)")
    objs = rec_fit["archive"]["objects"]; sd = HERE / "results" / f"run_scores_{fit_run}"
    if rec_fit["config"]["extraction_run"] != "2026-10-13_g3_real_seed-oss-36b_v1" or sha(sd / "runs.json") != objs["data/runs.json"]["sha256"] \
            or any(sha(sd / f"scores_L{l}.npz") != objs[f"scores_L{l}.npz"]["sha256"] for l in LAYERS):
        raise SystemExit("the fit's local runs.json or scores differ from its run record's archive")
    runs = json.loads((sd / "runs.json").read_text())
    if any(r["arm"] == "sim-attack" for r in runs):
        raise SystemExit("a sim-attack run in the real-only runs")
    e2 = {r["key"]: (i, int(r["fold"])) for i, r in enumerate(runs) if r["arm"] == "real-attack-e2"}
    st = json.loads(Path(stage).read_text()); run, arm, expect = EXTRACT[mode]
    keys = sorted(x["key"] for x in (st["controls"] if mode == "gate" else st["planted"]) if x["arm"] == arm)
    if len(keys) != expect or not set(keys) <= set(e2):
        raise SystemExit(f"not {expect} staged keys of the real e2 runs")
    if mode == "planted":
        g = json.loads(Path(gate).read_text())
        if g.get("mode") != "gate" or g.get("gate") != "pass" or g.get("frozen run") != fit_run or sha(HERE / "results" / g["scores file"]) != g["scores sha256"]:
            raise SystemExit("planted scoring needs a passing G3 gate record of this fit")
    items = [{"key": k, "fold": e2[k][1]} for k in keys]
    layers = list(LAYERS) if mode == "gate" else sorted(int(l) for l in g["layers passing"])  # a failing layer is dropped
    res = score.remote(mode, items, layers, fit_run)
    if any(res["probes sha256"][str(l)] != objs[f"probes_L{l}.npz"]["sha256"] for l in layers):
        raise SystemExit("the probes read differ from the fit's archived ones")
    if mode == "planted" and any(res["probes sha256"][str(l)] != g["probes sha256"][str(l)] for l in layers):
        raise SystemExit("the probes differ from the gate's")
    S = {l: np.array(res["scores"][str(l)]) for l in layers}
    rec = {"rule": "RULE.md steps 4-5 (this folder)", "git": head, "mode": mode, "reader": "Seed-OSS-36B-Instruct @ 497f1dca", "frozen run": fit_run,
           "extraction run": run, "arm": arm, "layers scored": layers, "runs": len(keys), "probes sha256": res["probes sha256"], "staging sha256": sha(stage),
           "fit run record sha256": sha(HERE / "results" / f"run_{fit_run}.json"), "render gate sha256": sha(render_gate),
           "activation gate": {"file": Path(act_gate).name, "sha256": sha(act_gate)}} | ({"score gate": {"file": Path(gate).name, "sha256": sha(gate)}} if mode == "planted" else {})
    unavailable = {str(l): [it["key"] for j, it in enumerate(items) if not np.isfinite(np.delete(S[l][it["fold"], :, j], it["fold"])).all()] for l in layers}
    rec |= {"runs without four finite nested scores, by layer": {l: len(v) for l, v in unavailable.items()}, "unavailable keys, by layer": unavailable}
    if any(unavailable.values()):  # RULE.md step 3: a missing or non-finite row stops G3 as unevaluable
        rec["gate" if mode == "gate" else "status"] = "unevaluable: runs without four finite nested scores"
    if mode == "gate":
        diffs = {}
        for l in layers:
            stored = np.load(sd / f"scores_L{l}.npz")["output_end__mean__task_nested"]
            dl = [abs(S[l][it["fold"], v, j] - stored[it["fold"], v, e2[it["key"]][0]]) for j, it in enumerate(items) for v in range(5) if v != it["fold"]]
            diffs[str(l)] = float(max(dl)) if np.isfinite(dl).all() else float("inf")
        rec |= {"gate: max |re-scored - stored| nested score, by layer (tolerance 1e-3)": diffs, "comparisons per layer": len(items) * 4,
                "layers passing": [l for l in diffs if diffs[l] <= TOL], "layers failing": [l for l in diffs if diffs[l] > TOL]}
        repro = "pass" if not rec["layers failing"] else "fail (G3 stops as unevaluable: RULE.md step 4)"
        rec |= {"reproduction": repro, "gate": rec.get("gate", repro)}  # r4PYs: an "unevaluable" status set above is kept
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    npz = HERE / "results" / f"g3_scores_{mode}_{stamp}.npz"
    np.savez(npz, keys=np.array(keys), folds=np.array([it["fold"] for it in items]), **{f"L{l}": S[l] for l in layers})
    rec |= {"scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"g3_scores_{mode}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("probes sha256", "unavailable keys, by layer")}, indent=1)); print(dest)
    if (mode == "gate" and rec["gate"] != "pass") or rec.get("status", "").startswith("unevaluable"):
        raise SystemExit(1)
