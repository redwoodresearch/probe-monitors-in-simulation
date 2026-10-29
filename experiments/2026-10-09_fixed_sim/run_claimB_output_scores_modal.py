"""Check 2 of PLAN.md amendment 3, part 1 (Modal H100, one container): per-output scores of probe step A's selected
probes. For fixed sim vs real e1 at both levels, each outer fold's chosen layer and C (A's nested headline,
`best_layer_and_C`) is refit once on the real labels with d9's fitter (run_claimB_grid.py at a1821be0, its refit loop
unchanged) and every test-fold output is scored. Check before use: the refit's run means must reproduce A's saved
test run scores for that (layer, C) (correlation >= 0.999). Writes per-output scores (run key, arm, step, action, tool,
fold, layer, C, score; no text) to the Volume and to results/partial-free JSON on the laptop for the split
(analyze_claimB_output_types.py).

  cd ~/probe-shift-claimb && PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml \
      modal run ~/probe-shift-fixed-sim/experiments/2026-10-09_fixed_sim/run_claimB_output_scores_modal.py
"""

import gzip
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

CLAIMB = Path.home() / "probe-shift-claimb"
GRID_FILE = CLAIMB / "experiments/2026-10-10_claimB_probe/run_claimB_grid.py"
HERE = Path(__file__).parent
SUMMARY = HERE / "results" / "claimB_probe_summary_2026-10-10_021700.json"
PINNED = "a1821be0"
sys.path.insert(0, str(GRID_FILE.parent) if modal.is_local() else "/root")
import run_claimB_grid as G  # noqa: E402  (d9's module, unchanged)

PACKS = {"level0": "10ac44f21b93", "level1": "39f842231a46"}
COMP = "fixed_vs_e1"  # default; --comp e2_vs_e1 for the real-vs-real control
app = modal.App("probe-shift_aswin-surya_sub-claimb-probe_outscores_kill-2026-11-10")
image = G.image.add_local_file(GRID_FILE, "/root/run_claimB_grid.py")


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, volumes={"/acts": G.vol}, timeout=2400, retries=0)
def scores(cfg: dict, choices: dict, comp: str = COMP) -> dict:
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

    out = {}
    for level, by_fold in choices.items():
        rows, check = [], {}
        for layer in sorted({v[0] for v in by_fold.values()}):
            src, runs, t, sc, Y, env = G.setup(cfg, level, comp, layer)  # per layer, as the grid does
            use, efold = sc.runs_in[t["run"]], env[t["run"]]
            y = torch.from_numpy(Y[:, :1]).cuda().double()
            X = torch.from_numpy(np.load(src / f"L{layer}_{G.POS}.npy")).cuda().float()
            assert len(X) == len(t["run"]), (len(X), len(t["run"]))
            saved = {C: np.load(G.done_path(cfg, level, comp, layer, C))["test_run_score"][:, 0]
                     for C in {v[1] for v in by_fold.values() if v[0] == layer}}
            for f, (l, C) in by_fold.items():
                if l != layer:
                    continue
                f = int(f)
                tr, te = use & (efold != f), use & (efold == f)
                Xa, Xb = tp.standardize(X[torch.from_numpy(np.flatnonzero(tr)).cuda()].double(), X[torch.from_numpy(np.flatnonzero(te)).cuda()].double())
                Xa, Xb, Ya = Xa * C**0.5, Xb * C**0.5, y[torch.from_numpy(t["run"][tr]).cuda()]
                mi, prev = G.MAX_ITER, None
                while True:  # d9's refit loop
                    W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=G.tol_of(C))
                    if evals < mi or evals == prev or mi * 4 > G.MAX_ITER_CAP:
                        break
                    mi, prev = mi * 4, evals
                s = tp.decision(Xb, W, b)[:, 0].cpu().numpy()
                ri = t["run"][te]
                k, inv = np.unique(ri, return_inverse=True)
                means = np.bincount(inv, s) / np.bincount(inv)
                check[f] = {"layer": layer, "C": C, "evals": int(evals), "corr_with_saved": float(np.corrcoef(means, saved[C][k])[0, 1]),
                            "max_abs_diff": float(np.abs(means - saved[C][k]).max())}
                for j, i in enumerate(np.flatnonzero(te)):
                    r = runs[t["run"][i]]
                    rows.append({"key": r["key"], "arm": r["arm"], "environment": r["environment"], "main_task": r["main_task"],
                                 "step": int(t["step"][i]), "action": int(t["action"][i]), "tool": str(t["tool"][i]),
                                 "fold": f, "layer": layer, "C": C, "score": float(s[j])})
                del Xa, Xb
            del X
        p = Path("/acts/claimB") / level / f"output_scores_A_{comp}_{cfg['tag']}.jsonl.gz"
        with open(p.with_suffix(".tmp"), "wb") as fh:
            fh.write(gzip.compress("".join(json.dumps(x) + "\n" for x in rows).encode(), mtime=0))
        p.with_suffix(".tmp").replace(p)
        G.vol.commit()
        out[level] = {"rows": rows, "check": check}
    return out


@app.local_entrypoint()
def main(comp: str = COMP):
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CLAIMB, capture_output=True, text=True, check=True).stdout.strip()
    assert head.startswith(PINNED), head
    folds = json.loads((CLAIMB / "splits" / "folds_v2.json").read_text())
    tag = hashlib.sha256(GRID_FILE.read_bytes() + (CLAIMB / "src/probe_shift/torch_probe.py").read_bytes()
                         + (CLAIMB / "src/probe_shift/q1_eval.py").read_bytes()).hexdigest()[:12]
    cfg = {"packs": PACKS, "seed": G.SEED, "bootstrap": G.BOOT, "tag": tag,
           "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
    summ = json.loads(SUMMARY.read_text())["levels"]
    choices = {lv: summ[lv][comp]["choices"]["best_layer_and_C"] for lv in PACKS if comp in summ[lv]}  # planted checks: level 1 only
    res = scores.remote(cfg, choices, comp)
    for lv, r in res.items():
        bad = {f: c for f, c in r["check"].items() if c["corr_with_saved"] < 0.999}
        print(lv, len(r["rows"]), "outputs scored; refit vs saved run scores:",
              {f: round(c["corr_with_saved"], 5) for f, c in r["check"].items()}, "| below 0.999:", bad)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    local = HERE / "data" / f"claimB_output_scores_A_{comp}_{stamp}.json.gz"  # scores and run keys, no text; data/ is git-ignored
    local.parent.mkdir(exist_ok=True)
    with open(local, "xb") as fh:
        fh.write(gzip.compress(json.dumps({"code": head, "tag": tag, "packs": PACKS, "comparison": comp, "choices": choices, "levels": res}).encode()))
    print(local)
