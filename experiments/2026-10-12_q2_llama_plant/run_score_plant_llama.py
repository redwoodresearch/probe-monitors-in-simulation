"""Gate row 3 (PLAN.md here): frozen-probe scoring of the Llama planted-slip extraction. A copy of
../2026-10-12_plantslip_transfer/q2_reply_score.py (P4-P6's scorer, at b2c2ec04's tree) with one reader, Llama-3.3-70B's
part-1 fit (run 2026-10-12_143129), scored at all 7 fitted layers, and the gate run at every layer. One Modal CPU
container; saved activations and frozen probes only, no fit, no model, no text.

Per run in test fold f: each nested model (f, v), v != f, scores the run's kept output_end rows as
((x - mean) / std) . w + b in float32 from the float16 rows; the run score is the mean over those rows.

  --which controls: the 20 controls, and the gate: every control's re-scored nested scores against the stored ones
                    (run_scores_2026-10-12_143129/scores_L<l>.npz) at all 7 layers, tolerance 1e-3 (560 comparisons).
  --which planted:  needs --gate, a passing controls record from the same staging record; scores the planted arms that
                    were extracted (--arms e2 or e1e2).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-12_q2_llama_plant/run_score_plant_llama.py --which <controls|planted> --stage <stage json> \\
      --scores-dir <run_scores_2026-10-12_143129> [--arms e2|e1e2] [--gate <controls record json>]
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-llamaplant_score_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-q2-llama_acts_kill-2026-11-12"
WORK = "probe-shift_aswin-surya_sub-q2-llama_work_kill-2026-11-12"
RUN_ID, LAYERS, FOLD_LAYERS, TOL = "2026-10-12_143129", (20, 30, 40, 50, 60, 70, 79), (60, 70, 60, 60, 70), 1e-3
EXTRACT = "2026-10-12_q2_plantslip_reply_{which}_llama33-70b_v1"  # as run_extract_llama_plant.py names its runs
ARMS = {"controls": {"e2": ["real-attack-e2-control"]}, "planted": {"e2": ["real-attack-e2-planted"], "e1e2": ["real-attack-e1-planted", "real-attack-e2-planted"]}}
STAGE_PIN = "f1a68c90a5d26799925bda1c0d574728a547cde6c2ec3b76bfa18c3691c75157"
RUNS_JSON = "ad10475afccc285bde39a389a17cb8b278a15d670c5d673933a094c6ffbc5254"  # run 2026-10-12_143129's runs.json (fix_scores record)

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
acts = modal.Volume.from_name(ACTS, version=2).read_only()
work = modal.Volume.from_name(WORK, version=2).read_only()


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/acts": acts, "/work": work}, cpu=8, memory=32768, timeout=3600, retries=0)
def score(extract_run: str, runs: list[dict]) -> dict:
    """runs: {key, fold, arm}. Per layer, nested scores [len(runs), 5] (NaN off the run's fold and where a run has no
    meta or no kept row), the sha256 of the probe files read, and the runs with no meta."""
    import numpy as np
    root = Path("/acts") / extract_run
    metas, missing = {}, []
    for i, r in enumerate(runs):
        p = root / "meta" / r["arm"] / f"{filename(r['key'])}.json"
        if not p.exists():
            missing.append(r["key"]); continue
        m = json.loads(p.read_text())
        rows = sorted(q["row"] for q in m["positions"] if q["kind"] == "output_end" and q["kept"])
        metas[i] = (m["chunk"], m["row_start"], rows)
    out, shas = {}, {}
    for l in LAYERS:
        f = Path("/work") / "runs" / RUN_ID / f"probes_L{l}.npz"
        shas[str(l)] = hashlib.sha256(f.read_bytes()).hexdigest()
        with np.load(f) as zf:
            z = {k: zf[k] for k in zf.files}
        S = np.full((len(runs), 5), np.nan)
        by_chunk = {}
        for i, (c, _, _) in metas.items():
            by_chunk.setdefault(c, []).append(i)
        for c, idx in by_chunk.items():
            X = np.load(root / "by_layer" / f"L{l}" / f"{c}.npy")  # whole file: row-indexed mmap is slow on a Volume
            for i in idx:
                _, s0, rows = metas[i]
                if not rows:
                    continue
                x = X[[s0 + q for q in rows]].astype(np.float32); f_ = runs[i]["fold"]
                for v in range(5):
                    if v != f_:
                        k = f"output_end__task_nested_{f_}_{v}__"
                        S[i, v] = float(np.mean((((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])).astype(np.float64)))
        out[str(l)] = S.tolist()
    return {"scores": out, "probes sha256": shas, "runs with no meta": missing}


@app.local_entrypoint()
def main(which: str, stage: str, scores_dir: str, arms: str = "e2", gate: str = ""):
    import numpy as np
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    if which not in ARMS or arms not in ARMS[which]:
        raise SystemExit("--which controls|planted [--arms e2|e1e2]")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    d = Path(scores_dir)
    if sha(stage) != STAGE_PIN or sha(d / "runs.json") != RUNS_JSON:
        raise SystemExit("not the pinned staging record or run 2026-10-12_143129's runs.json")
    st = json.loads(Path(stage).read_text())
    runs_all = json.loads((d / "runs.json").read_text())
    j_of = {(r["arm"], r["key"]): j for j, r in enumerate(runs_all)}
    if which == "planted":  # the gate must have passed, from the same staging record, before any planted score
        g = json.loads(Path(gate).read_text()) if gate else {}
        if g.get("gate") != "pass" or g.get("stage sha256") != STAGE_PIN or g.get("frozen run") != RUN_ID or g.get("which") != "controls":
            raise SystemExit("planted scoring needs a passing controls record (--gate) from the same staging record")
        if g.get("stored scores sha256") != {f: sha(d / f) for f in ["runs.json"] + [f"scores_L{l}.npz" for l in LAYERS]}:
            raise SystemExit("the stored scores differ from the ones the gate checked")
    arm_list = ARMS[which][arms]
    todo = [{"j": j_of[(x["orig_arm"], x["key"])], "key": x["key"], "fold": int(runs_all[j_of[(x["orig_arm"], x["key"])]]["fold"]), "arm": x["arm"]}
            for x in (st["controls"] if which == "controls" else st["planted"]) if x["arm"] in arm_list]
    extract_run = EXTRACT.format(which=which if which == "controls" else f"planted-{arms}")
    res = score.remote(extract_run, todo)
    if which == "planted" and res["probes sha256"] != g["probes sha256"]:
        raise SystemExit("the frozen probes differ from the ones the gate checked")
    S = {l: np.array(v) for l, v in res["scores"].items()}
    rec = {"plan": "PLAN.md (this folder)", "git": head, "reader": "meta-llama/Llama-3.3-70B-Instruct", "frozen run": RUN_ID, "extract run": extract_run,
           "which": which, "arms list": arm_list, "stage sha256": STAGE_PIN, "probes sha256": res["probes sha256"],
           "stored scores sha256": {f: sha(d / f) for f in ["runs.json"] + [f"scores_L{l}.npz" for l in LAYERS]},
           "runs": len(todo), "runs with no meta": res["runs with no meta"]}
    if which == "controls":  # the gate: every control, every layer, every nested model v != f
        diffs = []
        for l in LAYERS:
            stored = np.load(d / f"scores_L{l}.npz")["output_end__mean__task_nested"]
            for i, t in enumerate(todo):
                for v in range(5):
                    if v != t["fold"]:
                        diffs.append(abs(S[str(l)][i, v] - stored[t["fold"], v, t["j"]]))
        distinct = len({t["key"] for t in todo})
        ok = bool(distinct == 20 == len(todo) and not res["runs with no meta"] and len(diffs) == 560 and np.isfinite(diffs).all() and max(diffs) <= TOL)
        rec |= {"gate: max |re-scored - stored| nested score, 7 layers (tolerance 1e-3)": float(np.nanmax(diffs)) if diffs else None,
                "gate: comparisons (20 controls x 4 nested models x 7 layers)": len(diffs), "gate: distinct controls": distinct, "gate": "pass" if ok else "fail"}
    npz = HERE / "results" / f"llama_plant_scores_{which}-{arms}_{stamp}.npz"
    np.savez(npz, j=np.array([t["j"] for t in todo]), arm=np.array([t["arm"] for t in todo]), **{f"L{l}": v for l, v in S.items()})
    rec |= {"scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"llama_plant_scores_{which}-{arms}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
    if which == "controls" and rec["gate"] != "pass":
        raise SystemExit("the gate failed: no planted scoring")
