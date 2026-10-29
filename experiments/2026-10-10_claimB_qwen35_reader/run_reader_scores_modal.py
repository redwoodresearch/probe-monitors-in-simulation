"""Follow-up (b), per-output scores of the chosen probes: a copy of experiments/2026-10-10_claimB_onpolicy/run_onpolicy_scores_modal.py
(follow-up (c)) that imports this folder's grid copy (run_claimB_grid.py, fitting identical to d9's 857cb3cc) and writes
reader_output_scores_*; for the per-environment readouts (secondary, exploratory). That docstring follows. Each outer
fold's choice from the grid (`--readouts output_end`, run on run_reader_pack.py's packs; its result file's `by_readout.output_end.best.choices`) is
refit once on the real labels with the grid's own fitter, refit loop and gradient check; a refit must reproduce the grid's
saved test run scores (correlation >= 0.999, finite). Every test output is scored. Writes the scores (run key, own arm,
step, action, environment, main task, cell, fold; no text) to the Volume first, then data/reader_output_scores_<stamp>.json.gz.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with pyyaml modal run --detach \
      experiments/2026-10-10_claimB_qwen35_reader/run_reader_scores_modal.py --packs level1=<digest> --grid-result <claimB_b_grid_*.json>
"""

import gzip
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
GRID_FILE = HERE / "run_claimB_grid.py"
REPO = HERE.parents[1] if modal.is_local() else HERE
READOUT = "output_end"
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_claimB_grid as G  # noqa: E402  (the grid copy in this folder)

app = modal.App("probe-shift_aswin-surya_sub-claimb_reader-scores_kill-2026-11-10")
image = G.image.add_local_file(GRID_FILE, "/root/run_claimB_grid.py")


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, volumes={"/acts": G.vol}, timeout=1800, retries=0)
def scores(cfg: dict, choices: dict, tag: str) -> dict:
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

    out = {}
    for level, by_comp in choices.items():
        for comp, by_fold in by_comp.items():
            rows, check = [], {}
            for layer in sorted({v[1] for v in by_fold.values()}):
                src, runs, t, sc, Y, env = G.setup(cfg, level, comp, layer)
                X = torch.from_numpy(np.load(src / f"L{layer}_{READOUT}.npy")).cuda().float()
                use, efold = sc.runs_in[t["run"]], env[t["run"]]
                for f, (rd, l, C) in by_fold.items():
                    if l != layer or rd != READOUT:
                        continue
                    f = int(f)
                    tr, te = use & (efold != f), use & (efold == f)
                    Xa, Xb = tp.standardize(X[torch.from_numpy(np.flatnonzero(tr)).cuda()].double(), X[torch.from_numpy(np.flatnonzero(te)).cuda()].double())
                    Xa, Xb, Ya = Xa * C**0.5, Xb * C**0.5, torch.from_numpy(Y[t["run"][tr], :1]).cuda().double()
                    mi, prev = G.MAX_ITER, None
                    while True:  # the grid's refit loop
                        W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=G.tol_of(C))
                        if evals < mi or evals == prev or mi * 4 > G.MAX_ITER_CAP:
                            break
                        mi, prev = mi * 4, evals
                    Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)  # the grid's gradient check
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
                    loss.backward()
                    grad = float(max(Wg.grad.abs().max(), bg.grad.abs().max()))
                    s = tp.decision(Xb, W, b)[:, 0].cpu().numpy()
                    saved = np.load(G.done_path(cfg, level, comp, READOUT, layer, C))["test_run_score"][:, 0]
                    k, inv = np.unique(t["run"][te], return_inverse=True)
                    means = np.bincount(inv, s) / np.bincount(inv)
                    check[f] = {"layer": layer, "C": C, "evals": int(evals), "grad_max": grad, "converged": grad <= G.tol_of(C),
                                "corr_with_saved": float(np.corrcoef(means, saved[k])[0, 1])}
                    for j, i in enumerate(np.flatnonzero(te)):
                        r = runs[t["run"][i]]
                        rows.append({"key": r["key"], "arm": r["arm"], "arm_own": r["arm_own"], "environment": r["environment"],
                                     "main_task": r["main_task"], "cell": r["cell"], "step": int(t["step"][i]), "action": int(t["action"][i]),
                                     "fold": f, "score": float(s[j])})
                    del Xa, Xb
                del X
            out[f"{level}|{comp}"] = {"rows": rows, "check": check}
    bad = {k: f for k, v in out.items() for f, c in v["check"].items()
           if not (np.isfinite(c["corr_with_saved"]) and c["corr_with_saved"] >= 0.999 and c["converged"])}
    if bad or not all(np.isfinite(r["score"]) for v in out.values() for r in v["rows"]):
        raise ValueError(f"refits failed the reproduction, convergence or finiteness check: {bad}")  # nothing is published
    p = Path("/acts/claimB") / f"reader_output_scores_{tag}.json.gz"  # a copy in case the client dies; never overwritten
    with open(p, "xb") as fh:
        fh.write(gzip.compress(json.dumps(out).encode(), mtime=0))
    G.vol.commit()
    return out


@app.local_entrypoint()
def main(packs: str, grid_result: str):
    import subprocess

    import numpy as np  # noqa: F401  (the client must be able to read what the worker returns)

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    grid = json.loads(Path(grid_result).read_text())
    folds = json.loads((REPO / "splits" / "folds_v2.json").read_text())
    cfg = {"packs": dict(x.split("=") for x in packs.split(",")), "seed": G.SEED, "bootstrap": G.BOOT, "tag": grid["meta"]["code_tag"],
           "main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "environment_to_fold": folds["environment_folds"]["environment_to_fold"]}
    if grid["meta"]["packs"] != cfg["packs"]:  # exactly the grid's packs, levels included
        raise SystemExit(f"the grid result was run on packs {grid['meta']['packs']}, not {cfg['packs']}")
    choices = {lv: {c: v["by_readout"][READOUT]["best"]["choices"] for c, v in by.items()} for lv, by in grid["levels"].items()}
    import hashlib

    tag = hashlib.sha1(json.dumps({"packs": cfg["packs"], "grid": Path(grid_result).name, "choices": choices, "git": git}, sort_keys=True).encode()).hexdigest()[:12]
    res = scores.remote(cfg, choices, tag)
    for k, r in res.items():
        bad = {f: c for f, c in r["check"].items() if not np.isfinite(c["corr_with_saved"]) or c["corr_with_saved"] < 0.999}
        print(k, len(r["rows"]), "outputs; converged", all(c["converged"] for c in r["check"].values()), "| reproduction failures:", bad)
        if bad:
            raise SystemExit("a refit did not reproduce the grid's run scores")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "data").mkdir(exist_ok=True)
    with open(HERE / "data" / f"reader_output_scores_{stamp}.json.gz", "xb") as fh:
        fh.write(gzip.compress(json.dumps({"git": git, "tag": tag, "grid_result": Path(grid_result).name, "packs": cfg["packs"], "scores": res}).encode()))
    print("->", HERE / "data" / f"reader_output_scores_{stamp}.json.gz")
