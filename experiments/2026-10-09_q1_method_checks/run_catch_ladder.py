"""POST-HOC metric addition (Aswin, 2026-10-10 22:30:17 and 22:30:41 PT, transcript times; the registered metric was AUROC): the share caught
targeting 1% and 5% false alarms beside the refreshed ladder (levels 0-7, real vs simulated and real e1 vs e2, the
earliest-good selection), from the C search's stored records. One Modal CPU function rebuilds each (level, comparison)'s
earliest-good held-out real-column run scores as select_level does, from the summary's per-fold (layer, C), and checks
they reproduce the summary's AUROC; the statistics are local (analyze_catch_plants.halves: main tasks split in halves,
threshold on the calibration half's reference arm, catch and achieved false alarms on the test half). Reference arm:
real (real vs simulated) and e1 (e1 vs e2). A separate file, so run_q1_cgrid.py's cache tag is unchanged.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml --with numpy --with scipy modal run \
      experiments/2026-10-09_q1_method_checks/run_catch_ladder.py
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
SUMMARIES = {"results/partial/cgrid_2026-10-10_111249_REPRO_FAILED.json": "86829851fe41",  # levels 0 and 7
             "results/partial/cgrid_ladder_2026-10-10_145944_REPRO_FAILED.json": "381ebc7001fb"}  # levels 1-6
SEL = "earliest_good_layer_and_C"

app = modal.App("probe-shift_aswin-surya_sub-q1-method-checks_catch_kill-2026-11-10")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scikit-learn==1.7.2", "scipy", "pyyaml")
         .add_local_python_source("probe_shift").add_local_file(HERE / "run_q1_cgrid.py", "/root/run_q1_cgrid.py"))
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)


@app.function(image=image, cpu=2, memory=8192, volumes={"/acts": vol}, timeout=1800, retries=0)
def scores(jobs: list) -> dict:
    """jobs: (level, cfg, tag, comparison, {fold: [layer, C]}). Per job, the compared runs' arm, main task, label and
    the earliest-good held-out real-column score, with its AUROC."""
    import numpy as np

    from run_q1_cgrid import LAYERS, col_auc, done_path, setup

    out = {}
    for level, cfg, tag, comp, choice in jobs:
        _, runs, _, sc, _, env = setup(cfg, LAYERS[0], comp)  # select_level's call
        s = np.full(len(runs), np.nan)
        for f, (layer, C) in choice.items():
            m = env == int(f)
            s[m] = np.load(done_path(cfg, tag, level, layer, C, "scaled", comp))["test_run_score"][m, 0]
        r_in = np.flatnonzero(sc.runs_in)
        if not np.isfinite(s[r_in]).all():
            raise ValueError(f"{level} {comp}: a compared run has no selected score")
        out[f"{level}|{comp}"] = {"arm": [runs[i]["arm"] for i in r_in], "main": [str(sc.main[i]) for i in r_in],
                                  "y": sc.y_run[r_in].tolist(), "score": s[r_in].tolist(),
                                  "auroc": float(col_auc(sc.y_run[r_in][:, None], s[r_in][:, None])[0])}
    return out


@app.local_entrypoint()
def main():
    import numpy as np

    from analyze_catch_plants import halves, line
    from run_q1_cgrid import COMPARISONS, level_cfg

    jobs, reported = [], {}
    for f, tag in SUMMARIES.items():
        d = json.loads((HERE / f).read_text())
        if d["meta"]["volume_results"] != f"cgrid_v1_{tag}":
            raise SystemExit(f"{f}: not tag {tag}")
        for lv, by in d["levels"].items():
            cfg = level_cfg(lv)
            for comp in COMPARISONS:
                e = by[comp][SEL]
                jobs.append((lv, cfg, tag, comp, e["choice_by_fold"]))
                reported[f"{lv}|{comp}"] = e["test"]
    got = scores.remote(jobs)
    res = {"metric": "POST HOC addition (Aswin 22:30:17 and 22:30:41 PT): catch targeting 1% and 5% false alarms; registered metric AUROC",
           "selection": SEL, "summaries": SUMMARIES,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(), "rows": {}}
    lines = []
    for key in sorted(got, key=lambda k: (int(k.split("|")[0][5:]), k)):
        g, rep = got[key], reported[key]
        if abs(g["auroc"] - rep["auroc"]) > 5e-5 + 1e-12 or len(g["y"]) != rep["n"]:
            raise SystemExit(f"{key}: rebuilt scores give AUROC {g['auroc']:.6f} on {len(g['y'])} runs, the summary {rep}")
        y, arm = np.array(g["y"]), np.array(g["arm"], object)
        want = ("real", "sim") if key.endswith("real_vs_sim") else ("e1", "e2")  # reference arm, other arm
        if not (all(want[0] in a and want[1] not in a for a in arm[y == 0]) and all(want[1] in a and want[0] not in a for a in arm[y == 1])):
            raise SystemExit(f"{key}: arms {sorted(set(arm[y == 0]))} vs {sorted(set(arm[y == 1]))}, expected {want}")
        row = halves(g["main"], g["score"], y == 0, y == 1) | {
            "auroc_as_reported": rep["auroc"], "auroc_ci95_as_reported": rep["ci95"], "auroc_rebuilt": g["auroc"],
            "reference_arms": sorted(set(arm[y == 0])), "other_arms": sorted(set(arm[y == 1]))}
        res["rows"][key] = row
        lines.append(line(f"{key.replace('|', ' ')} (ref {','.join(row['reference_arms'])})", row, rep["auroc"]))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"catch_ladder_POSTHOC_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print("\n".join(lines))
    print(path.name)
