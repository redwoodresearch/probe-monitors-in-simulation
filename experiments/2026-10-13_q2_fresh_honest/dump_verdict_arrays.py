"""Fresh-task honest runs: dump the verdict's per-run inputs for an independent recompute (q2 via 0d, 10-14). The verdict
run's own scoring path, score_fresh._analyze(keep=True), with the verdict's inputs (all seven stages): for every row, the
fresh runs' nested scores [5, 5, n_fresh], their eligibility, and the cohort runs' nested scores, folds, tasks and
eligibility. Also the fresh runs' main tasks, the 327-task fresh universe and the cohort task lists per fold. Saved as one
npz plus a manifest on the fresh work Volume (verdict_arrays/<stamp>/), with sha256s. Thresholds are NOT saved: the
recompute derives them itself. Numbers and task names only, no text. Modal CPU.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' --with numpy modal run \\
      experiments/2026-10-13_q2_fresh_honest/dump_verdict_arrays.py
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
APP = "probe-shift_aswin-surya_sub-q2-fresh_dump_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-q2-fresh_work_kill-2026-11-13"

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_python_source("probe_shift")
         .add_local_file(HERE / "score_fresh.py", "/root/score_fresh.py"))
VOLS = {"/acts": modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only(),  # score_fresh.VOLS
        "/seed": modal.Volume.from_name("probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12", version=2).read_only(),
        "/q3w": modal.Volume.from_name("probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", version=2).read_only(),
        "/g3w": modal.Volume.from_name("probe-shift_aswin-surya_sub-q2-g3_work_kill-2026-11-12", version=2).read_only(),
        "/df": modal.Volume.from_name("probe-shift_aswin-surya_sub-whysteps_work_kill-2026-11-13", version=2).read_only(),
        "/out": modal.Volume.from_name(WORK, version=2)}


@app.function(image=image, cpu=16, memory=131072, timeout=3600, retries=0, volumes=VOLS)
def dump(fresh: list[dict], base: list[dict], universe: list[str], planned: int, pre_left: int, stamp: str, meta: dict) -> dict:
    try:
        import numpy as np

        import score_fresh as sf
        gates, kept = sf._analyze(fresh, base, universe, planned, False, pre_left, keep=True)
        arrays = {"fresh__main_task": np.array([r["main_task"] for r in fresh]), "fresh__run_id": np.array([r["run_id"] for r in fresh]),
                  "fresh__transcript": np.array([r["transcript"] for r in fresh]), "fresh__stage": np.array([r["stage"] for r in fresh]),
                  "universe": np.array(universe), "rows": np.array(list(kept))}
        for i, (name, k) in enumerate(kept.items()):
            for key in ("SF", "elig", "S_c", "fold_c", "task_c", "c_elig"):
                arrays[f"row{i}__{key}"] = np.asarray(k[key])
        d = Path("/out/verdict_arrays") / stamp
        d.mkdir(parents=True)
        with open(d / "arrays.npz", "wb") as fh:
            np.savez(fh, **arrays)
        sha = hashlib.sha256((d / "arrays.npz").read_bytes()).hexdigest()
        man = meta | {"rows (index -> name)": list(kept), "arrays sha256": sha, "rescore gates": gates["gates"],
                      "layout": "row<i>__SF [5,5,n_fresh] fresh nested scores (NaN on the diagonal); row<i>__elig [n_fresh] bool; "
                                "row<i>__S_c [5,5,n_cohort] cohort nested scores (NaN outside folds f and v); row<i>__fold_c, "
                                "row<i>__task_c, row<i>__c_elig [n_cohort]; fresh__main_task, fresh__run_id, fresh__transcript (the dump key), fresh__stage [n_fresh]; universe [327]"}
        (d / "manifest.json").write_text(json.dumps(man, indent=1))
        VOLS["/out"].commit()
        return man
    except Exception as e:  # noqa: BLE001 — type and location only
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"dump failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}") from None


@app.local_entrypoint()
def main():
    import score_fresh as sf
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    runs36 = json.loads(sf.BASE_RUNS.read_text())
    base = [{"key": r["key"], "main_task": r["main_task"], "fold": r["fold"], "i36": i} for i, r in enumerate(runs36) if r["arm"] == sf.H]
    cells = json.loads(sf.TASKS.read_text())["cells"]
    labels = [str(k) for k in range(1, 8)]
    fresh, pre_left, inputs = [], 0, {}
    for lab in labels:  # score_fresh.main's inputs, unchanged
        b = f"s{lab}"
        st_p, qc_p, rl_p = (sf.one(f"stage_2026-10-13_q2_fresh_{b}_inputs_v1_*.json"), sf.one(f"check_2026-10-13_q2_fresh_{b}_qwen3-32b_l7_v1_*.json"),
                            sf.one(f"fresh_runs_fresh1013_stage{lab}_*.json"))
        for p in (st_p, qc_p, rl_p):
            inputs[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        st, qc, rl = (json.loads(p.read_text()) for p in (st_p, qc_p, rl_p))
        bad = sorted({str(s) for s in qc["failing shards (their rows are not used)"]} | {str(s) for s in qc["shards without a reproduction run"]})
        errored = set(rl["head"]["errored_keys"])
        pre_left += sum(st["excluded"].values()) + st["rows without a log"] + st.get("errored rows, not staged", 0)
        for x in st["per run"]:
            fresh.append({"run_id": x["run_id"], "main_task": x["main_task"], "environment": x["environment"], "transcript": x["transcript"],
                          "arm": "real-honest-fresh", "stage": lab, "qwen_run": f"2026-10-13_q2_fresh_{b}_qwen3-32b_l7_v1",
                          "seed_run": f"2026-10-13_q2_fresh_{b}_seedcall_seed-oss-36b_v1", "eligible before scoring": x["run_id"] not in errored,
                          "qwen bad shards": bad})
    planned = sum(c["stage"] in labels for c in cells)
    universe = sorted({c["main_task"] for c in cells if c["stage"] in labels})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    meta = {"label": "per-run inputs of the fresh-task verdict (all seven stages), for an independent recompute",
            "git": git("rev-parse", "HEAD"), "verdict record": "results/score_fresh_s1234567_2026-10-14_080417.json",
            "inputs sha256": inputs, "planned": planned, "left out before scoring (not in the arrays)": pre_left, "stamp": stamp}
    man = dump.remote(fresh, base, universe, planned, pre_left, stamp, meta)
    with open(p := HERE / "results" / f"verdict_arrays_manifest_{stamp}.json", "x") as fh:
        fh.write(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k not in ("rescore gates", "inputs sha256")}, indent=1))
    print(p)
