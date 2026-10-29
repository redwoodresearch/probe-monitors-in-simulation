"""Planted-slip transfer, q2's reply pairs P4-P6 (RULE.md, Amendment 1): frozen-probe scoring. One Modal CPU container per
call; saved activations and frozen probes only, no fit, no model, no text.

Per run in test fold f, at the reader's layer for that fold: each nested model (f, v), v != f, scores the run's kept
output_end rows as ((x - mean) / std) . w + b in float32 from the float16 rows, and the run score is the mean over those
rows (the fit's "mean" pooling), as in ../2026-10-12_q2_olmo_reader/run_score_fix_olmo.py.

  --mode pins      sha256 of each reader's frozen probe files (its work Volume) and of the stored score files (local);
                   writes results/q2_reply_pins_<stamp>.json. Committed before any staging.
  --mode score     scores the runs of one extraction run (`--extract-run`) for one pair (`--pair`), writes their nested
                   scores (npz) and a record (json). For the controls run, the record also holds the reproduction gate:
                   each control's re-scored nested scores against the stored ones, tolerance 1e-3 logit.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-12_plantslip_transfer/q2_reply_score.py --mode <pins|score> [--pair P4|P5|P6] [--extract-run <run>] [--stage <stage json>]
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-plantq2_score_kill-2026-11-12"
SCRATCH = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-shift-q2/802b982d-f788-49b8-9390-11c29e3725f3/scratchpad")
READERS = {  # pair -> the frozen detector behind its forest row
    "P4": {"reader": "Qwen/Qwen3-32B (#36)", "work": "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", "run": "2026-10-06_181548",
           "acts": "probe-shift_aswin-surya_sub-la-activations", "layers": [63], "fold_layers": [63, 63, 63, 63, 63],
           "scores": "/Users/aswinsurya/probe-distribution-shift-from-sims/.claude/worktrees/agent-a41654b9677ba3292/experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548"},
    "P5": {"reader": "allenai/Olmo-3.1-32B-Instruct", "work": "probe-shift_aswin-surya_sub-q2-olmo_work_kill-2026-11-12", "run": "2026-10-12_110549",
           "acts": "probe-shift_aswin-surya_sub-q2-olmo_acts_kill-2026-11-12", "layers": [32, 40, 56], "fold_layers": [32, 56, 56, 56, 40],
           "scores": str(SCRATCH / "wt_olmo/experiments/2026-10-12_q2_olmo_reader/results/run_scores_2026-10-12_110549")},
    "P6": {"reader": "Qwen/Qwen3.5-27B", "work": "probe-shift_aswin-surya_sub-q2-qwen35-reply_work_kill-2026-11-12", "run": "2026-10-12_011944",
           "acts": "probe-shift_aswin-surya_sub-q2-plantq2_acts_kill-2026-11-12", "layers": [63], "fold_layers": [63, 63, 63, 63, 63],
           "scores": str(SCRATCH / "wt_q35/experiments/2026-10-12_q2_qwen35_reply_readout/results/run_scores_2026-10-12_011944")},
}
TOL = 1e-3

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
VOLS = {f"/work_{p}": modal.Volume.from_name(r["work"], version=2).read_only() for p, r in READERS.items()}
VOLS |= {f"/acts_{p}": modal.Volume.from_name(r["acts"], version=2, create_if_missing=True).read_only() for p, r in READERS.items()}


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes=VOLS, cpu=2, memory=4096, timeout=600, retries=0)
def pins() -> dict:
    out = {}
    for p, r in READERS.items():
        for l in r["layers"]:
            f = Path(f"/work_{p}") / "runs" / r["run"] / f"probes_L{l}.npz"
            out[f"{p} {r['work']}:runs/{r['run']}/probes_L{l}.npz"] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


@app.function(image=image, volumes=VOLS, cpu=8, memory=32768, timeout=3600, retries=0)
def score(pair: str, extract_run: str, runs: list[dict]) -> dict:
    """runs: {j, key, fold, arm}. Returns per layer the nested scores [len(runs), 5] (NaN off the run's fold and where
    a run has no meta or no kept row), the sha256 of the probe files read, and the count of runs with no meta."""
    import numpy as np
    r_ = READERS[pair]
    root = Path(f"/acts_{pair}") / extract_run
    metas, missing = {}, 0
    for i, r in enumerate(runs):
        p = root / "meta" / r["arm"] / f"{filename(r['key'])}.json"
        if not p.exists():
            missing += 1
            continue
        m = json.loads(p.read_text())
        rows = sorted(q["row"] for q in m["positions"] if q["kind"] == "output_end" and q["kept"])
        metas[i] = (m["chunk"], m["row_start"], rows)
    out, shas = {}, {}
    for l in r_["layers"]:
        f = Path(f"/work_{pair}") / "runs" / r_["run"] / f"probes_L{l}.npz"
        shas[str(l)] = hashlib.sha256(f.read_bytes()).hexdigest()
        with np.load(f) as zf:
            z = {k: zf[k] for k in zf.files}
        S = np.full((len(runs), 5), np.nan)
        by_chunk = {}
        for i, (c, s0, rows) in metas.items():
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
def main(mode: str, pair: str = "", extract_run: str = "", stage: str = "", which: str = "", gate: str = ""):
    """--which controls|planted. Planted scoring needs --gate: this pair's controls record, with a passing gate."""
    import numpy as np
    me = Path(__file__).resolve()
    for f in (me, HERE / "RULE.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    if mode == "pins":
        rec = {"git": head, "probes (work Volume)": pins.remote(),
               "stored scores (local)": {f"{p} {Path(r['scores']).name}/{f}": sha(Path(r["scores"]) / f)
                                         for p, r in READERS.items() for f in ["runs.json"] + [f"scores_L{l}.npz" for l in r["layers"]]},
               "readers": {p: {k: r[k] for k in ("reader", "work", "run", "layers", "fold_layers")} for p, r in READERS.items()}}
        with open(dest := HERE / "results" / f"q2_reply_pins_{stamp}.json", "x") as fh:
            fh.write(json.dumps(rec, indent=1) + "\n")
        print(json.dumps(rec, indent=1)); print(dest)
        return
    if mode != "score" or pair not in READERS or not extract_run or not stage or which not in ("controls", "planted"):
        raise SystemExit("--mode score needs --pair, --extract-run, --stage and --which controls|planted")
    r_ = READERS[pair]; d = Path(r_["scores"])
    st = json.loads(Path(stage).read_text())
    runs_all = json.loads((d / "runs.json").read_text())
    j_of = {(r["arm"], r["key"]): j for j, r in enumerate(runs_all)}
    is_controls = which == "controls"
    if not is_controls:  # the reproduction gate must have passed for this pair, from the same staging, before any planted score
        g = json.loads(Path(gate).read_text()) if gate else {}
        if g.get("pair") != pair or g.get("gate") != "pass" or g.get("stage sha256") != sha(stage):
            raise SystemExit("planted scoring needs this pair's passing controls record (--gate), from the same staging")
    todo = [{"j": j_of[(x["orig_arm"], x["key"])], "key": x["key"], "fold": int(runs_all[j_of[(x["orig_arm"], x["key"])]]["fold"]), "arm": x["arm"]}
            for x in (st["controls"] if is_controls else st["planted"])]
    res = score.remote(pair, extract_run, todo)
    S = {l: np.array(v) for l, v in res["scores"].items()}
    rec = {"plan": "RULE.md, Amendment 1 (this folder)", "git": head, "pair": pair, "reader": r_["reader"], "frozen run": r_["run"], "extract run": extract_run,
           "stage file": Path(stage).name, "stage sha256": sha(stage), "probes sha256": res["probes sha256"],
           "stored scores sha256": {f: sha(d / f) for f in ["runs.json"] + [f"scores_L{l}.npz" for l in r_["layers"]]},
           "runs": len(todo), "runs with no meta": res["runs with no meta"]}
    if is_controls:  # the reproduction gate: each control at its fold's layer, every nested model v != f
        diffs = []
        for i, t in enumerate(todo):
            l = str(r_["fold_layers"][t["fold"]]); stored = np.load(d / f"scores_L{l}.npz")["output_end__mean__task_nested"]
            for v in range(5):
                if v != t["fold"]:
                    diffs.append(abs(S[l][i, v] - stored[t["fold"], v, t["j"]]))
        distinct = len({t["key"] for t in todo})
        ok = bool(distinct == 20 == len(todo) and len(diffs) == 80 and np.isfinite(diffs).all() and max(diffs) <= TOL)
        rec |= {"gate: max |re-scored - stored| nested score (tolerance 1e-3)": float(np.nanmax(diffs)) if diffs else None,
                "gate: comparisons (20 controls x 4 nested models)": len(diffs), "gate: distinct controls": distinct, "gate": "pass" if ok else "fail"}
    npz = HERE / "results" / f"q2_reply_scores_{pair}_{which}_{stamp}.npz"
    np.savez(npz, j=np.array([t["j"] for t in todo]), arm=np.array([t["arm"] for t in todo]), **{f"L{l}": v for l, v in S.items()})
    rec |= {"scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"q2_reply_scores_{pair}_{which}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
