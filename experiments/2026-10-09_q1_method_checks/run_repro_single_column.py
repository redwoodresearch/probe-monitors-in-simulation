"""The C search's reproduction gate, as Q1 published it: the published optimizer (float32, C = 0.01, tol 1e-5, max_iter
2000) fit on the real labels ALONE, one label column, as analyze_q1_gpu.py did (its real probe is a single column; the
shuffle null was a separate fit). run_q1_cgrid.py's gate refit fits the real column jointly with the 100 shuffles under
one stopping test, and failed by up to 0.0036 (results/partial/cgrid_2026-10-10_111249_REPRO_FAILED.json). Modal H100.

Only the six failing (level, comparison, layer) pairs are refit, and only the held-out-environment run AUROC at
C = 0.01 is computed and printed: no other C, no search outcome. A separate file, so run_q1_cgrid.py's cache tag (it
hashes that script) is unchanged. The reading, committed before this runs (predictions_cgrid_2026-10-09.md): all six
within 0.002 of the published values -> the gate passes as designed and the levels 1-6 rule is applied unchanged;
otherwise the C search failed its reproduction, and nothing more is read.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml modal run \
      experiments/2026-10-09_q1_method_checks/run_repro_single_column.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
PAIRS = [("level7", "real_vs_sim", 32), ("level7", "real_vs_sim", 41), ("level7", "real_vs_sim", 52),
         ("level7", "null_e1_vs_e2", 44), ("level0", "null_e1_vs_e2", 47), ("level0", "null_e1_vs_e2", 58)]
GATE = 0.002

app = modal.App("probe-shift_aswin-surya_sub-q1-method-checks_repro_kill-2026-11-09")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scikit-learn==1.7.2", "scipy", "pyyaml")
         .add_local_python_source("probe_shift").add_local_file(HERE / "run_q1_cgrid.py", "/root/run_q1_cgrid.py"))
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)


@app.function(image=image, gpu="H100!", cpu=4, memory=16384, volumes={"/acts": vol}, timeout=1200, retries=0)
def refit(cfg: dict, layer: int, comparison: str) -> float:
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp
    from run_q1_cgrid import POS, SETTINGS, col_auc, labels, setup

    st = SETTINGS["published"]
    src, runs, t, sc, _, env = setup(cfg, layer, comparison)
    Y = labels(cfg, runs, sc)[:, :1]  # the real column only
    X = torch.from_numpy(np.load(src / f"L{layer}_{POS}.npy")).cuda().float()
    use = sc.runs_in[t["run"]]
    efold = env[t["run"]]
    idx = lambda m: torch.from_numpy(np.flatnonzero(m)).cuda()
    score = np.full(len(runs), np.nan)
    for f in sorted(set(efold[use])):
        tr, te = use & (efold != f), use & (efold == f)
        Xa, Xb = tp.standardize(X[idx(tr)], X[idx(te)])
        W, b, _ = tp.fit(Xa, torch.from_numpy(Y[t["run"][tr]]).cuda().float(), 0.01, st["max_iter"], tol=st["tol"])
        s = tp.decision(Xb, W, b)[:, 0].double().cpu().numpy()
        k, inv = np.unique(t["run"][te], return_inverse=True)
        score[k] = np.bincount(inv, weights=s) / np.bincount(inv)
    r_in = np.flatnonzero(sc.runs_in)
    if not np.isfinite(score[r_in]).all():
        raise ValueError("a compared run got no test score")
    return float(col_auc(Y[r_in], score[r_in, None])[0])


@app.local_entrypoint()
def main(pairs: str = ""):
    """`--pairs level3:real_vs_sim:41,...`: other failing pairs (the levels 1-6 run); default: the six of levels 0 and 7."""
    import subprocess

    from run_q1_cgrid import ALL_LEVELS, level_cfg, published

    todo = [(lv, c, int(l)) for lv, c, l in (x.split(":") for x in pairs.split(","))] if pairs else PAIRS
    cfgs = {lv: level_cfg(lv) for lv in sorted({p[0] for p in todo})}
    calls = [(p, refit.spawn(cfgs[p[0]], p[2], p[1])) for p in todo]
    rows = []
    for (lv, comp, layer), c in calls:
        ref = published(lv, cfgs[lv]["extraction_run"], comp)[layer]
        here = c.get()
        rows.append({"level": lv, "comparison": comp, "layer": layer, "published": ref, "single_column_refit": here,
                     "diff": here - ref, "within_gate": abs(here - ref) <= GATE})  # unrounded: the gate reads these
    out = {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
           "configs": {lv: ALL_LEVELS[lv] for lv in cfgs}, "gate": GATE, "pairs": rows, "all_within_gate": all(r["within_gate"] for r in rows)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with (HERE / "results" / f"repro_single_column_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
