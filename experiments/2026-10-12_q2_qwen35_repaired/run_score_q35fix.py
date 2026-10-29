"""Qwen3.5 repaired arm (post hoc; PLAN.md here): frozen-probe scoring. One Modal CPU container per call; saved activations
and the frozen probes of run 2026-10-12_011944 only, no fit, no model, no text. The scoring core is
../2026-10-12_plantslip_transfer/q2_reply_score.py's (P6, as run at ee60bf09), for this one reader at layer 63.

Each nested model (f, v), v != f, scores a run's kept output_end rows as ((x - mean) / std) . w + b in float32 from the
float16 rows, and the run score is the mean over those rows, as the fit pools them.

  --which controls  scores the 20 re-extracted controls (config/q35fix_controls.json) and runs the gate. All 80
                    comparisons (20 runs x 4 nested models off each run's fold) must match the stored nested scores within
                    1e-3 and be finite.
  --which repaired  needs --gate (a passing controls record from the same stored scores and probes). It scores fix939's 823
                    repaired runs and writes them in the layout analyze_edit_effect_q35.py reads: run_index (the 939
                    simulated runs, runs.json order), has_fix, and L63 [5, 5, 939], NaN where a run has no repaired version.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-12_q2_qwen35_repaired/run_score_q35fix.py --which controls|repaired --extract-run <run> \\
      --scores-dir <run_scores_2026-10-12_011944> --fix <fix939 per-run jsonl> [--gate <controls record>]
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-q35fix_score_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-q2-qwen35-reply_work_kill-2026-11-12"  # run 2026-10-12_011944's probes
ACTS = "probe-shift_aswin-surya_sub-q2-q35fix_acts_kill-2026-11-13"        # this folder's extractions
RUN, LAYER, LAYERS, FIX_ARM, TOL = "2026-10-12_011944", 63, (16, 24, 32, 40, 48, 56, 63), "sim-attack-note-sr-both", 1e-3
PIN = {"fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
work = modal.Volume.from_name(WORK, version=2).read_only()
acts = modal.Volume.from_name(ACTS, version=2, create_if_missing=True).read_only()


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/work": work, "/acts": acts}, cpu=8, memory=32768, timeout=3600, retries=0)
def score(extract_run: str, runs: list[dict]) -> dict:
    """runs: {key, fold, arm}. Returns the nested scores [len(runs), 5] at LAYER (NaN on the run's own fold and where a run
    has no meta or no kept row), the probe file's sha256 and the count of runs with no meta. Failures re-raise by type only."""
    try:
        return _score(extract_run, runs)
    except Exception as e:  # noqa: BLE001 — sanitized and re-raised, never swallowed
        raise RuntimeError(f"score failed: {type(e).__name__}") from None


def _score(extract_run, runs):
    import numpy as np
    root = Path("/acts") / extract_run
    metas, missing = {}, 0
    for i, r in enumerate(runs):
        p = root / "meta" / r["arm"] / f"{filename(r['key'])}.json"
        if not p.exists():
            missing += 1
            continue
        m = json.loads(p.read_text())
        if m["key"] != r["key"]:
            raise ValueError("meta key mismatch")
        rows = sorted(q["row"] for q in m["positions"] if q["kind"] == "output_end" and q["kept"])
        metas[i] = (m["chunk"], m["row_start"], rows)
    f = Path("/work") / "runs" / RUN / f"probes_L{LAYER}.npz"
    probe_sha = hashlib.sha256(f.read_bytes()).hexdigest()
    with np.load(f) as zf:
        z = {k: zf[k] for k in zf.files}
    S = np.full((len(runs), 5), np.nan)
    by_chunk = {}
    for i, (c, s0, rows) in metas.items():
        by_chunk.setdefault(c, []).append(i)
    for c, idx in by_chunk.items():
        X = np.load(root / "by_layer" / f"L{LAYER}" / f"{c}.npy")  # whole file: row-indexed mmap is slow on a Volume
        for i in idx:
            _, s0, rows = metas[i]
            if not rows:
                continue
            x = X[[s0 + q for q in rows]].astype(np.float32); f_ = runs[i]["fold"]
            for v in range(5):
                if v != f_:
                    k = f"output_end__task_nested_{f_}_{v}__"
                    S[i, v] = float(np.mean((((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])).astype(np.float64)))
    return {"scores": S.tolist(), "probe sha256": probe_sha, "runs with no meta": missing}


@app.local_entrypoint()
def main(which: str, extract_run: str, scores_dir: str, fix: str, gate: str = ""):
    import numpy as np
    for f in (Path(__file__).resolve(), HERE / "PLAN.md", HERE / "config" / "q35fix_controls.json"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    d, fix_p = Path(scores_dir), Path(fix)
    if not d.name.endswith(RUN) or sha(fix_p) != PIN[fix_p.name]:
        raise SystemExit("not run 2026-10-12_011944's scores dir, or not the pinned fix939 per-run file")
    stored_sha = {"runs.json": sha(d / "runs.json")} | {f"scores_L{l}.npz": sha(d / f"scores_L{l}.npz") for l in LAYERS}
    runs = json.loads((d / "runs.json").read_text())
    j_of = {r["key"]: j for j, r in enumerate(runs) if r["arm"] == "sim-attack"}
    fixk = sorted({json.loads(l)["key"] for l in fix_p.open() if json.loads(l)["label"] == "FIX"})
    if len(j_of) != 939 or len(fixk) != 823 or not set(fixk) <= set(j_of):
        raise SystemExit("not the 939 simulated runs and fix939's 823 repaired keys")
    stored = np.load(d / f"scores_L{LAYER}.npz")["output_end__mean__task_nested"]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"plan": "PLAN.md (this folder), POST HOC", "git": head, "run": RUN, "layer": LAYER, "extraction": extract_run, "which": which,
           "inputs sha256": {fix_p.name: sha(fix_p)} | stored_sha}
    if which == "controls":
        ctl = json.loads((HERE / "config" / "q35fix_controls.json").read_text())["controls"]
        if len(ctl) != 20 or len({k for _, k in ctl}) != 20 or any(a != "sim-attack" or k in fixk for a, k in ctl):
            raise SystemExit("the controls must be 20 distinct simulated runs that fix939 left unchanged")
        todo = [{"key": k, "fold": int(runs[j_of[k]]["fold"]), "arm": "sim-attack"} for _, k in ctl]
        res = score.remote(extract_run, todo); S = np.array(res["scores"])
        diffs = [abs(S[i, v] - stored[t["fold"], v, j_of[t["key"]]]) for i, t in enumerate(todo) for v in range(5) if v != t["fold"]]
        ok = bool(res["runs with no meta"] == 0 and len(diffs) == 80 and np.isfinite(diffs).all() and max(diffs) <= TOL)
        rec |= {"probe sha256": res["probe sha256"], "runs with no meta": res["runs with no meta"],
                "folds covered": sorted({t["fold"] for t in todo}),
                "gate: max |re-scored - stored| nested score (tolerance 1e-3)": float(np.nanmax(diffs)) if diffs else None,
                "gate: comparisons (20 controls x 4 nested models)": len(diffs), "gate": "pass" if ok else "fail"}
        with open(dest := HERE / "results" / f"q35fix_scores_controls_{stamp}.json", "x") as fh:
            fh.write(json.dumps(rec, indent=1) + "\n")
        print(json.dumps(rec, indent=1)); print(dest)
        if not ok:
            raise SystemExit("gate failed: nothing else may be scored")
        return
    if which != "repaired":
        raise SystemExit("--which controls|repaired")
    g = json.loads(Path(gate).read_text()) if gate else {}
    if g.get("gate") != "pass" or g.get("inputs sha256") != rec["inputs sha256"]:
        raise SystemExit("repaired scoring needs a passing controls record (--gate) from the same stored scores and fix939 file")
    todo = [{"key": k, "fold": int(runs[j_of[k]]["fold"]), "arm": FIX_ARM} for k in fixk]
    res = score.remote(extract_run, todo); S = np.array(res["scores"])
    if res["probe sha256"] != g["probe sha256"]:
        raise SystemExit("the probe file differs from the one the gate used")
    finite = all(np.isfinite(np.delete(S[i], t["fold"])).all() for i, t in enumerate(todo))
    if res["runs with no meta"] or not finite:
        raise SystemExit(f"incomplete: {res['runs with no meta']} repaired runs without a meta, or a missing nested score")
    sim_idx = np.array(sorted(j_of.values())); col = {j: c for c, j in enumerate(sim_idx)}
    L = np.full((5, 5, len(sim_idx)), np.nan); has_fix = np.zeros(len(sim_idx), bool)
    for i, t in enumerate(todo):
        c = col[j_of[t["key"]]]; has_fix[c] = True; L[t["fold"], :, c] = S[i]
    npz = HERE / "results" / f"q35fix_scores_repaired_{stamp}.npz"
    if npz.exists() or npz.with_suffix(".json").exists():  # reserve the names before writing (#92, from #1100)
        raise SystemExit(f"{npz.name} or its record already exists")
    np.savez(npz, run_index=sim_idx, has_fix=has_fix, **{f"L{LAYER}": L})
    rec |= {"probe sha256": res["probe sha256"], "gate record": Path(gate).name, "gate record sha256": sha(gate),
            "as-simulated runs": len(sim_idx), "with re-rendered scores": int(has_fix.sum()), "unchanged (as-simulated scores stand)": int((~has_fix).sum()),
            "scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"q35fix_scores_repaired_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
