"""Fix-full (PLAN.md Amendment 6), POST HOC: the frozen G3 detector's FULL nested score matrix [f, v] (v != f) on the simulated
honest runs, as simulated (d9's dump, arm sim-honest) and under full and N (our fx dump, sim-honest-fx-full / -NL), for the
matched-false-alarm calibration: a run in fold v calibrates model (f, v) for every f != v, which the frozen scorer (it
scores a run's own fold row only) never computes. The frozen `_score` (with g3_score_simh.py's one documented line) is
called once per pretend fold f in 0..4, which fills row f; layers 40 and 63 (the detector's layer by fold). Gate: each run's
own-fold row equals the stored scores (g3_scores_simh_2026-10-13_163239 / g3_scores_fx_2026-10-13_161130) exactly.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_q2_fixfull/g3_score_honest_matrix.py --runs <#36 runs.json> --fit-dir <G3 results dir> \\
      --simh-scores results/g3_scores_simh_2026-10-13_163239.json --fx-scores <dir>/g3_scores_fx_2026-10-13_161130.json
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
APP = "probe-shift_aswin-surya_sub-q2-fixfull_seedhmatrix_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-q2-g3_work_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"
SIMH_RUN = "2026-10-13_g3_simhonest_seed-oss-36b_v1"
EXTRACT = {"sim-honest": (SIMH_RUN, "sim-honest", None), "sim-honest-fx-full": ("2026-10-13_fixfull_seed-oss-36b_v1", "sim-honest-fx-full", None),
           "sim-honest-fx-NL": ("2026-10-13_fixfull_seed-oss-36b_v1", "sim-honest-fx-NL", None)}
MLAYERS = (40, 63)
CHECK = ("3c4ea6ef", "experiments/2026-10-13_q3_sce_reply_probe/results/extract_check_2026-10-13_g3_simhonest_seed-oss-36b_v1_2026-10-13_143315.json",
         "43ee1eace0ee80e9629b24297a1df3cedb643462eafb7a0f8249fa4c1053956e")
OLD_LINE = 'if any(t["arm"] != arm for t in idx["transcripts"]):'
NEW_LINE = 'if any(t["arm"] != arm for t in idx["transcripts"] if t["key"] == it["key"]):'
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
        if any(t["arm"] != arm for t in idx["transcripts"] if t["key"] == it["key"]):                   # checked before any array is read
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
    if fn(mine, "filename") != fn(frozen_src, "filename"):
        raise SystemExit("filename differs from the frozen scorer's")
    fz = fn(frozen_src, "_score")
    if fz.count(OLD_LINE) != 1 or fn(mine, "_score") != fz.replace(OLD_LINE, NEW_LINE):
        raise SystemExit("_score differs from the frozen scorer's beyond the one documented line")


@app.local_entrypoint()
def main(runs: str, fit_dir: str, simh_scores: str, fx_scores: str):
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
    fit_run = fc["fit run"]
    fd = Path(fit_dir)
    if sha(fd / f"run_{fit_run}.json") != fc["fit run record sha256"] or sha(fd / f"run_scores_{fit_run}" / "runs.json") != fc["inputs of x sha256"]["runs.json"]:
        raise SystemExit("the fit's records differ from the frozen ones")
    if sha(runs) != RUNS_PIN:
        raise SystemExit("not #36's pinned runs.json")
    stored = {}
    for p, arms in ((simh_scores, ("sim-honest",)), (fx_scores, ("sim-honest-fx-full", "sim-honest-fx-NL"))):
        committed = subprocess.run(["git", "show", f"HEAD:experiments/2026-10-13_q2_fixfull/results/{Path(p).name}"], cwd=HERE, capture_output=True, check=True).stdout
        if hashlib.sha256(committed).hexdigest() != sha(p):
            raise SystemExit(f"{Path(p).name} is not the committed record")
        rec_ = json.loads(Path(p).read_text())
        for a_ in arms:
            r_ = rec_["arms"][a_]
            if sha(Path(p).parent / r_["scores file"]) != r_["scores sha256"]:
                raise SystemExit(f"{a_}: the stored scores file differs from its record")
            z = np.load(Path(p).parent / r_["scores file"])
            stored[a_] = ([str(k) for k in z["keys"]], z["folds"].astype(int), {l: z[f"L{l}"] for l in MLAYERS})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"rule": "PLAN.md Amendment 6, POST HOC (this folder)", "git": git("rev-parse", "HEAD").stdout.strip(), "frozen run": fit_run,
           "layers scored": list(MLAYERS), "scorer deviation": {"frozen line": OLD_LINE, "this scorer": NEW_LINE}, "arms": {}}
    for arm, (keys, folds, old) in stored.items():
        M = {l: np.full((5, 5, len(keys)), np.nan) for l in MLAYERS}
        probes = None
        for f_ in range(5):
            res = score.remote(arm, [{"key": k, "fold": f_} for k in keys], list(MLAYERS), fit_run)
            if any(res["probes sha256"][str(l)] != fc["probes sha256 by layer"][str(l)] for l in MLAYERS):
                raise SystemExit("the probes read differ from the frozen ones")
            probes = res["probes sha256"]
            for l in MLAYERS:
                M[l][f_] = np.array(res["scores"][str(l)])[f_]
        own = {l: max(float(np.nanmax(np.abs(M[l][fo, :, j] - old[l][fo, :, j]))) for j, fo in enumerate(folds)) for l in MLAYERS}
        if any(v != 0.0 for v in own.values()):
            raise SystemExit(f"{arm}: the own-fold rows differ from the stored scores: {own}")
        off = ~np.eye(5, dtype=bool)
        nonfinite = {str(l): int((~np.isfinite(M[l][off])).sum()) for l in MLAYERS}
        npz = HERE / "results" / f"g3_scores_hmatrix_{arm}_{stamp}.npz"
        np.savez(npz, keys=np.array(keys), folds=folds, **{f"L{l}": M[l] for l in MLAYERS})
        rec["arms"][arm] = {"runs": len(keys), "own-fold rows vs stored, max abs diff": own, "non-finite off-diagonal entries": nonfinite,
                            "probes sha256": probes, "scores file": npz.name, "scores sha256": sha(npz),
                            "status": "scored" if not any(nonfinite.values()) else "unevaluable: non-finite off-diagonal entries"}
        print(arm, len(keys), rec["arms"][arm]["status"], flush=True)
    with open(dest := HERE / "results" / f"g3_scores_hmatrix_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(dest)
    if any(v["status"] != "scored" for v in rec["arms"].values()):
        raise SystemExit(1)
