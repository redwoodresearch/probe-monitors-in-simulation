"""F10 (PLAN_F10.md, POST HOC): g3_score_f9.py with the F10 run (run_extract_seedcall_f10.py's, whose output_end is G3's readout),
arms and app name, and main's two record checks for that extractor's formats (its in-job render gate, record
f10seed_render_*; its check, extract_check_*: "render gate" names the render record and its sha256). filename and _score
are unchanged (AST-checked against the frozen scorer). Its docstring follows.

Fix-full F9 (PLAN.md Amendment 3): g3_score_fx.py with the F9 run and arms; nothing else changed. Its docstring follows.

Fix-full (PLAN.md here), POST HOC: the frozen G3 detector's nested scores on every fx arm (run_extract_g3_fx.py --which fx).
A copy of ../2026-10-13_g3_followups/g3_score_rep.py (1c277bfc): `filename` and `_score` stay the frozen scorer's (AST-checked
against ../2026-10-13_g3_new_reader/g3_score_frozen.py, whose bytes forecast.json pins), as do the frozen fit and probe
checks; EXTRACT holds the fx arms, scored one .remote() call per arm in one container. forecast.json is read from its
acknowledged commit (6d7b1f50). Folds: an attack run takes its cell's real e2 fold in the frozen fit (as G3); a simulated
honest run takes its main task's fold in the frozen fit (task-grouped folds), and a main task the fit never saw is counted
and left out. --check must be a clean full check of the fx dump against --gate (a passing fx render gate).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_q2_fixfull/g3_score_fx.py --runs <#36 runs.json> --stage results/stage_<stamp>.json \\
      --gate results/render_fx_gate_<stamp>.json --check results/extract_check_2026-10-13_fixfull_seed-oss-36b_v1_<stamp>.json \\
      --fit-dir <G3 results dir holding run_2026-10-13_111525.json and run_scores_2026-10-13_111525/>
"""
import ast
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q2-f10_seedscore_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-q2-g3_work_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"
FX_RUN = "2026-10-13_f10_seedcall_seed-oss-36b_v1"
FX_ARMS = ("sim-attack-fx-fullF9F10", "sim-honest-fx-fullF9F10")
EXTRACT = {a: (FX_RUN, a, None) for a in FX_ARMS}
G3 = HERE.parent / "2026-10-13_g3_new_reader"
FORECAST = ("6d7b1f50c2c08a5887962b5066681684d6e1ae63", "experiments/2026-10-13_g3_new_reader/forecast.json")
LAYERS = (16, 24, 32, 40, 48, 56, 63)
FORECAST_SHA = "4243f2cd38d93d33e5b63ef08bc7e3fb3b387198af43e667c8821966504fff77"
RUNS_PIN = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"

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


def git(*a, check=True):
    return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, check=check)


def same_functions(frozen_src: str):
    """filename and _score here must be the frozen scorer's, source for source."""
    mine = Path(__file__).read_text()
    def fn(src, name):
        return next(ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    for name in ("filename", "_score"):
        if fn(mine, name) != fn(frozen_src, name):
            raise SystemExit(f"{name} differs from the frozen scorer's")


@app.local_entrypoint()
def main(runs: str, stage: str, gate: str, check: str, fit_dir: str):
    import numpy as np
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    for f in (Path(__file__).resolve(), HERE / "PLAN.md", G3 / "g3_score_frozen.py"):
        if git("ls-files", "--error-unmatch", str(f), check=False).returncode or git("diff", "--quiet", "HEAD", "--", str(f), check=False).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    blob = subprocess.run(["git", "show", f"{FORECAST[0]}:{FORECAST[1]}"], cwd=HERE, capture_output=True, check=True).stdout
    if hashlib.sha256(blob).hexdigest() != FORECAST_SHA:
        raise SystemExit("forecast.json at its acknowledged commit does not match the recorded sha256")
    fc = json.loads(blob)["frozen"]
    if sha(G3 / "g3_score_frozen.py") != fc["files sha256"]["scorer"]:
        raise SystemExit("g3_score_frozen.py differs from the frozen scorer")
    same_functions((G3 / "g3_score_frozen.py").read_text())
    rg = json.loads(Path(gate).read_text())
    if not (rg.get("gate") or {}).get("pass") or not Path(gate).name.startswith("f10seed_render_"):
        raise SystemExit("--gate is not a passing F10 render gate")
    st = json.loads(Path(stage).read_text())
    expect = {a: len(ks) for a, ks in st["runs by arm"].items()}
    ck = json.loads(Path(check).read_text())
    if ck.get("run") != FX_RUN or ck["render gate"]["render record"] != Path(gate).name or ck["render gate"]["render record sha256"] != sha(gate) \
            or ck["meta files by arm"] != expect or ck["kept readouts (meta)"] != ck["rows (chunks)"] or any(ck["problems"].values()) \
            or ck["non-finite values: every layer (chunk indexes)"] or ck["non-finite values: first and last layer recounted"]:
        raise SystemExit("--check is not a clean full check of the fx dump against --gate")
    if set(expect) - set(FX_ARMS):
        raise SystemExit(f"staged arms outside this scorer's: {sorted(set(expect) - set(FX_ARMS))}")
    fit_run = fc["fit run"]
    fd = Path(fit_dir)
    if sha(fd / f"run_{fit_run}.json") != fc["fit run record sha256"]:
        raise SystemExit("the fit run record differs from the frozen one")
    sd = fd / f"run_scores_{fit_run}"
    if sha(sd / "runs.json") != fc["inputs of x sha256"]["runs.json"]:
        raise SystemExit("the fit's runs.json differs from the frozen one")
    if sha(runs) != RUNS_PIN:
        raise SystemExit("not #36's pinned runs.json")
    fit_runs = json.loads((sd / "runs.json").read_text())
    e2_fold = {r["cell"]: int(r["fold"]) for r in fit_runs if r["arm"] == "real-attack-e2"}
    task_fold = {}
    for r in fit_runs:
        if task_fold.setdefault(r["main_task"], int(r["fold"])) != int(r["fold"]):
            raise SystemExit("a main task in two folds of the frozen fit")
    sims36 = {r["key"]: r for r in json.loads(Path(runs).read_text()) if r["arm"] == "sim-attack"}
    hon = {r["key"]: r for r in json.loads((HERE.parent / "2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json").read_text())
           if r["arm"] == "sim-honest"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"rule": "PLAN_F10.md, POST HOC (this folder)", "git": git("rev-parse", "HEAD").stdout.strip(), "reader": "Seed-OSS-36B-Instruct @ 497f1dca",
           "frozen run": fit_run, "extraction run": FX_RUN, "layers scored": list(LAYERS), "forecast.json": {"commit": FORECAST[0], "sha256": FORECAST_SHA},
           "runs.json sha256": RUNS_PIN, "stage record": {"file": Path(stage).name, "sha256": sha(stage)},
           "fx render gate": {"file": Path(gate).name, "sha256": sha(gate)}, "extraction check": {"file": Path(check).name, "sha256": sha(check)}, "arms": {}}
    for arm in sorted(expect):
        keys = sorted(st["runs by arm"][arm])
        if arm.startswith("sim-attack-fx-"):
            items = [{"key": k, "fold": e2_fold[sims36[k]["cell"]], "cell": sims36[k]["cell"]} for k in keys]
            left_out = []
        else:
            items = [{"key": k, "fold": task_fold[hon[k]["main_task"]], "cell": hon[k]["cell"]} for k in keys if hon[k]["main_task"] in task_fold]
            left_out = [k for k in keys if hon[k]["main_task"] not in task_fold]
        res = score.remote(arm, [{"key": it["key"], "fold": it["fold"]} for it in items], list(LAYERS), fit_run)
        if any(res["probes sha256"][str(l)] != fc["probes sha256 by layer"][str(l)] for l in LAYERS):
            raise SystemExit("the probes read differ from the frozen ones")
        S = {l: np.array(res["scores"][str(l)]) for l in LAYERS}
        unavailable = {str(l): [it["key"] for j, it in enumerate(items) if not np.isfinite(np.delete(S[l][it["fold"], :, j], it["fold"])).all()] for l in LAYERS}
        npz = HERE / "results" / f"g3_scores_f10_{arm}_{stamp}.npz"
        np.savez(npz, keys=np.array([it["key"] for it in items]), folds=np.array([it["fold"] for it in items]), **{f"L{l}": S[l] for l in LAYERS})
        rec["arms"][arm] = {"runs": len(items), "keys and cells": [[it["key"], it["cell"]] for it in items], "left out (main task not in the fit)": left_out,
                            "probes sha256": res["probes sha256"], "runs without four finite nested scores, by layer": {l: len(v) for l, v in unavailable.items()},
                            "unavailable keys, by layer": unavailable, "scores file": npz.name, "scores sha256": sha(npz),
                            "status": "unevaluable: runs without four finite nested scores" if any(unavailable.values()) else "scored"}
        print(arm, rec["arms"][arm]["runs"], rec["arms"][arm]["status"], flush=True)
    with open(dest := HERE / "results" / f"g3_scores_f10_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(dest)
    if any(v["status"] != "scored" for v in rec["arms"].values()):
        raise SystemExit(1)
