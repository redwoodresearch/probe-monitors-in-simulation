"""Run TF-IDF baselines in parallel Modal CPU containers: the baselines a config lists (one container each) and
the learning curve (one container per variant and fold). The analysis itself is analyze_tfidf.py, unchanged; the staged
examples are uploaded to a Modal Volume, and each container's output files come back to results/ as soon as it finishes.

  uv run modal run experiments/2026-10-06_q1_text_baselines/run_tfidf_modal.py::baselines                                    # main
  (add --stamp <yyyy-mm-dd_HHMMSS> to resume an interrupted run; run with `modal run --detach` so a dropped connection
  does not stop the app: finished files are also copied to the Volume's /results/)
  uv run modal run experiments/2026-10-06_q1_text_baselines/run_tfidf_modal.py::baselines --config config/tfidf_ladder.yaml  # ladder
  uv run modal run experiments/2026-10-06_q1_text_baselines/run_tfidf_modal.py::curve --subsets <task_subsets file in results/>
A detached run's files can also be fetched later: run_tfidf_modal.py::fetch_results --stamp <stamp> (the stamp is in
results/<config name>_launch_<stamp>.json, written at launch).
"""

import gzip
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
NAME = "probe-shift_aswin-surya_sub-baselines_tfidf_kill-2026-10-20"
TIMEOUT_S = 3600
app = modal.App(NAME)
vol = modal.Volume.from_name(NAME, create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.13")
         .uv_pip_install("numpy", "scipy", "scikit-learn==1.9.1", "joblib==1.6.0", "pyyaml")
         .env({"JOBLIB_TEMP_FOLDER": "/tmp"})  # joblib's shared-memory default (/dev/shm) is small in containers
         .add_local_python_source("probe_shift")
         .add_local_file(HERE / "analyze_tfidf.py", "/root/analyze_tfidf.py")
         .add_local_file(HERE / "analyze_tfidf_context.py", "/root/analyze_tfidf_context.py"))


def _analysis():
    sys.path.insert(0, "/root" if Path("/root/analyze_tfidf.py").exists() else str(HERE))
    import analyze_tfidf

    return analyze_tfidf


@app.function(image=image, volumes={"/data": vol}, cpu=16, memory=65536, timeout=TIMEOUT_S)
def baseline_job(cfg: dict, fold_of: dict, baseline: str, spec: dict, stamp: str, fold_key: str = "main_task") -> dict:
    A = _analysis()
    d = A.load(Path("/data") / cfg["examples"], fold_of, spec.get("segments", "outputs"), fold_key, cfg.get("cells"),
               cfg.get("holdout_arm"))
    out = Path(tempfile.mkdtemp())  # per call: a reused container must not return an earlier job's files
    # Checkpoints on the Volume, keyed by run stamp: a preempted job restarts with the same input and resumes.
    A.run_baseline(cfg, d, baseline, spec, out, stamp, Path("/data/checkpoints") / f"{cfg['name']}_{stamp}", vol.commit)
    keep = Path("/data/results")  # the finished files also stay on the Volume, not only on the laptop
    keep.mkdir(exist_ok=True)
    for p in out.iterdir():
        shutil.copy(p, keep / p.name)
    vol.commit()
    return {p.name: p.read_bytes() for p in out.iterdir()}


@app.function(image=image, volumes={"/data": vol}, cpu=16, memory=65536, timeout=TIMEOUT_S)
def curve_job(cfg: dict, fold_of: dict, variant: str, fold: int, fold_subsets: dict) -> dict:
    A = _analysis()
    return A.curve_fold(cfg, A.load(Path("/data") / cfg["examples"], fold_of), variant, fold, fold_subsets)


def _setup(config: str, stamp: str = ""):
    import yaml

    A = _analysis()
    # A new run's stamp is taken at launch, before the upload, and must be unused: two runs sharing a stamp would write
    # the same <baseline>_scores_<stamp> files (it happened once; see README, whole-transcript level 7).
    new = not stamp
    stamp = stamp or datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    assert not new or not list((HERE / "results").glob(f"*_{stamp}.*")), f"stamp {stamp} is already used in results/"
    cfg = A.with_cells(yaml.safe_load((HERE / config).read_text()))
    fold_key, fold_of = A.fold_map(cfg, json.loads((A.SPLITS / cfg["folds_file"]).read_text()))
    with vol.batch_upload(force=True) as batch:  # the staged examples, as read locally
        batch.put_directory(str(HERE / ".run_cache" / cfg["examples"]), f"/{cfg['examples']}")
    # A given stamp resumes an interrupted run of this config: its checkpoints on the Volume are keyed by it.
    if not new:
        started = {e.path.rsplit("/", 1)[-1] for e in vol.listdir("/checkpoints")}
        assert f"{cfg['name']}_{stamp}" in started, f"no run of {cfg['name']} with stamp {stamp} to resume"
    (HERE / "results").mkdir(exist_ok=True)
    return A, cfg, fold_key, fold_of, stamp


@app.function(image=image, volumes={"/data": vol}, timeout=4 * TIMEOUT_S)
def run_config(cfg: dict, fold_of: dict, fold_key: str, stamp: str, summary: dict) -> dict:
    """The fan-out, run remotely so that a detached run survives its local client closing (only the last function a
    local entrypoint calls outlives it): every baseline's job, then the run's summary in the Volume's /results/."""
    jobs = [(cfg, fold_of, b, spec, stamp, fold_key) for b, spec in cfg["baselines"].items()]
    for files in baseline_job.starmap(jobs, order_outputs=False):  # each job also copies its files to /data/results
        s = json.loads(next(v for k, v in files.items() if k.endswith(".json")))
        summary["baselines"][s["baseline"]] = s
    keep = Path("/data/results")
    keep.mkdir(exist_ok=True)
    (keep / f"{cfg['name']}_summary_{stamp}.json").write_text(json.dumps(summary, indent=1))
    vol.commit()
    return summary


def _fetch(stamp: str) -> None:
    """Copy a run's finished files (stamp `stamp`) from the Volume's /results/ to results/, never overwriting."""
    for entry in vol.listdir("/results"):
        name = entry.path.rsplit("/", 1)[-1]
        if f"_{stamp}." in name and not (HERE / "results" / name).exists():
            (HERE / "results" / name).write_bytes(b"".join(vol.read_file(f"/results/{name}")))
            print(f"-> results/{name}", flush=True)


@app.local_entrypoint()
def baselines(config: str = "config/tfidf.yaml", stamp: str = ""):
    """The baselines a config lists, one container each (config/tfidf.yaml: main; config/tfidf_ladder.yaml: scrub ladder).
    `stamp`: resume an interrupted run under its own stamp, reusing its checkpoints. Launch with `modal run --detach`;
    the local client may be closed once the jobs start, and fetch_results brings the files back later."""
    A, cfg, fold_key, fold_of, stamp = _setup(config, stamp)
    d = A.load(HERE / ".run_cache" / cfg["examples"], fold_of, fold_key=fold_key, cells=cfg.get("cells"),
               holdout_arm=cfg.get("holdout_arm"))
    summary = A.provenance(cfg, d["counts"]) | {"modal": {"app": NAME, "volume": NAME}, "baselines": {}}
    # Provenance at launch, so a run whose local client is closed can be fetched later with its config and git hash.
    launch = HERE / "results" / f"{cfg['name']}_launch_{stamp}.json"
    if launch.exists():  # a resume keeps the original launch record and records itself next to it
        now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        launch = HERE / "results" / f"{cfg['name']}_resume_{stamp}_at_{now}.json"
    launch.write_text(json.dumps(summary | {"stamp": stamp}, indent=1))
    print(f"stamp {stamp}", flush=True)
    summary = run_config.remote(cfg, fold_of, fold_key, stamp, summary)
    _fetch(stamp)
    for s in summary["baselines"].values():
        for key in [k for k in s["results"] if k.startswith(("real_vs_sim/", "null_e1_vs_e2/"))]:  # the variants this config ran
            a = s["results"][key]["all_cells"]
            print(f"{s['baseline']} {key:26} outputs {a['outputs']['auroc']:.3f} {a['outputs']['ci95']}  "
                  f"runs {a['runs']['auroc']:.3f} {a['runs']['ci95']}  pairs {a['pairs']['accuracy']:.3f}", flush=True)


@app.function(image=image, volumes={"/data": vol}, cpu=1, memory=16384, timeout=TIMEOUT_S)  # single-threaded: one core
def context_variant_job(cfg: dict, fold_of: dict, stamp: str, variant: str) -> dict:
    """One variant of analyze_tfidf_context.py on the Volume's stagings and score files: its part score file and summary."""
    _analysis()
    import analyze_tfidf_context as C

    vol.reload()
    out = Path(tempfile.mkdtemp())
    summary = C.run(cfg | {"variants": {variant: cfg["variants"][variant]}}, fold_of, Path("/data"), Path("/data/results"), out, stamp, variant)
    return {"summary": summary, "scores": (out / summary["score_file"]).read_bytes()}


@app.function(image=image, volumes={"/data": vol}, timeout=4 * TIMEOUT_S)
def run_context(cfg: dict, fold_of: dict, stamp: str, launch: dict) -> dict:
    """Every variant in its own container, then one score file (gzip parts concatenated, still one gzip stream of
    rows) and one summary in the Volume's /results/."""
    parts = list(context_variant_job.starmap([(cfg, fold_of, stamp, v) for v in cfg["variants"]]))
    name = f"{cfg['name']}_scores_{stamp}.jsonl.gz"
    keep = Path("/data/results")
    keep.mkdir(exist_ok=True)
    (keep / name).write_bytes(b"".join(p["scores"] for p in parts))
    summary = parts[0]["summary"] | launch | {"score_file": name, "results": {}, "segments_used": {}}  # config and git from launch
    for p in parts:
        summary["results"] |= p["summary"]["results"]
        summary["segments_used"] |= p["summary"]["segments_used"]
    (keep / f"{cfg['name']}_summary_{stamp}.json").write_text(json.dumps(summary, indent=1))
    vol.commit()
    return summary


@app.local_entrypoint()
def context(config: str):
    """analyze_tfidf_context.py for a config, one container per variant (launch with `modal run --detach`; the local
    client may be closed once it prints the stamp, and fetch_results brings the files back)."""
    import yaml

    A = _analysis()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    assert not list((HERE / "results").glob(f"*_{stamp}.*")), f"stamp {stamp} is already used in results/"
    cfg = A.with_cells(yaml.safe_load((HERE / config).read_text()))
    fold_of = json.loads((A.SPLITS / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    files = {f for v in cfg["variants"].values() for k, f in v.items() if k != "running" and f}
    on_volume = {e.path.rsplit("/", 1)[-1] for e in vol.listdir("/results")}
    with vol.batch_upload(force=True) as batch:  # the staging, and any score file the Volume lacks
        batch.put_directory(str(HERE / ".run_cache" / cfg["examples"]), f"/{cfg['examples']}")
        for f in files - on_volume:
            batch.put_file(str(HERE / "results" / f), f"/results/{f}")
    launch = {k: v for k, v in A.provenance(cfg, {}).items() if k != "counts"}
    (HERE / "results" / f"{cfg['name']}_launch_{stamp}.json").write_text(json.dumps(launch | {"stamp": stamp}, indent=1))
    print(f"stamp {stamp}", flush=True)
    run_context.remote(cfg, fold_of, stamp, launch)
    _fetch(stamp)


@app.local_entrypoint()
def fetch_results(stamp: str):
    """Copy a detached run's finished files from the Volume's /results/ to results/, never overwriting."""
    _fetch(stamp)


@app.local_entrypoint()
def curve(config: str = "config/tfidf_learning_curve.yaml", subsets: str = ""):
    """`subsets`: an existing task-subsets file in results/ (shared with the probe's curve); it must match the rule."""
    A, cfg, fold_key, fold_of, stamp = _setup(config)
    assert fold_key == "main_task", "the learning curve subsets main tasks"
    d = A.load(HERE / ".run_cache" / cfg["examples"], fold_of)
    computed = A.curve_subsets(cfg, d, fold_of)
    if subsets:
        path = HERE / "results" / subsets
        # Compare the task selection (per fold: training and test main tasks, and each subset), not metadata such as
        # the staging name: subsets depend only on the paired cells, the folds and the seeds.
        assert json.loads(path.read_text())["folds"] == computed["folds"], f"{subsets} does not match the subsets this config produces"
    else:
        path = HERE / "results" / f"{cfg['name']}_task_subsets_{stamp}.json"
        path.write_text(json.dumps(computed, indent=1))
    subsets = computed
    print(f"task subsets: {path}", flush=True)
    jobs = [(cfg, fold_of, v, int(f), subsets["folds"][f]) for v in cfg["variants"] for f in subsets["folds"]]
    results = []
    fold_scores = HERE / "results" / f"{cfg['name']}_fold_scores_{stamp}.jsonl.gz"
    with gzip.open(fold_scores, "wt") as fh:  # each fold's held-out scores, saved as it arrives
        for r in curve_job.starmap(jobs, order_outputs=False):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            results.append(r)
            print(f"  {r['variant']} fold {r['fold']} done", flush=True)
    summary = A.provenance(cfg, d["counts"]) | {"modal": {"app": NAME, "volume": NAME}, "task_subsets": path.name, "fold_scores": fold_scores.name,
                                                 "curve": A.curve_summary(cfg, d, results)}
    out = HERE / "results" / f"{cfg['name']}_{stamp}.json"
    out.write_text(json.dumps(summary, indent=1))
    for v, res in summary["curve"].items():
        for n, r in res.items():
            print(f"{v:18} train tasks {n:>4}: outputs {r['outputs_mean']:.3f} ± {r['outputs_sd']:.3f}  runs {r['runs_mean']:.3f} ± {r['runs_sd']:.3f}")
    print(f"-> {out}")
