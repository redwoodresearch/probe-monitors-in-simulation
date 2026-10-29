"""PLAN.md amendment 7, version (a), last part: per-output scores of the max-pooled readout's chosen probes, so runs can
also be scored by their highest output, plus the length baseline (Modal H100, one container, about $1).

Each outer fold's choice from the grid (d9's run_claimB_b_grid.py at 857cb3cc, `--readouts output_end_max`; its result
file's `by_readout.output_end_max.best.choices`) is refit once on the real labels with the grid's own fitter and refit
loop; a refit must reproduce the grid's saved test run scores (correlation >= 0.999). Every test output is scored.
Locally: run-level AUROC with the run's mean (B's score, reproduced) and max output score, 95% main-task interval
(unrounded), a fixed-choice null (1,000 draws, one coin per main task for the real comparisons, per pair for the planted
checks), the detection rule for the planted checks (amendment 7), and the length baseline (run mean of log token count).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with pyyaml \
      modal run experiments/2026-10-09_fixed_sim/run_claimB_max_scores_modal.py --pack <max pack digest> --grid-result <claimB_b_grid_*.json>
"""

import gzip
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

CLAIMB = Path.home() / "probe-shift-claimb-b"
GRID_FILE = CLAIMB / "experiments/2026-10-10_claimB_probe/run_claimB_b_grid.py"
HERE = Path(__file__).parent
PINNED, READOUT, LEVEL = "857cb3cc", "output_end_max", "level1"
BARS = {"plant1": 0.95, "plant2": 0.90}
sys.path.insert(0, str(GRID_FILE.parent) if modal.is_local() else "/root")
import run_claimB_b_grid as G  # noqa: E402  (d9's module, unchanged)

app = modal.App("probe-shift_aswin-surya_sub-claimb_maxscores_kill-2026-11-10")
image = G.image.add_local_file(GRID_FILE, "/root/run_claimB_b_grid.py")


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, volumes={"/acts": G.vol}, timeout=1800, retries=0)
def scores(cfg: dict, choices: dict) -> dict:
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

    out = {}
    for comp, by_fold in choices.items():
        rows, check = [], {}
        for layer in sorted({v[1] for v in by_fold.values()}):
            src, runs, t, sc, Y, env = G.setup(cfg, LEVEL, comp, layer)
            X = torch.from_numpy(np.load(src / f"L{layer}_{READOUT}.npy")).cuda().float()
            use, efold = sc.runs_in[t["run"]], env[t["run"]]
            n_tok = {(i, p["step"], p["action"]): p["n_tokens"] for i, m in enumerate(runs) for p in m["positions"] if p["kind"] == "output_end"}
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
                Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)  # the grid's gradient check, at C = 1 rescaled
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
                loss.backward()
                grad = float(max(Wg.grad.abs().max(), bg.grad.abs().max()))
                s = tp.decision(Xb, W, b)[:, 0].cpu().numpy()
                saved = np.load(G.done_path(cfg, LEVEL, comp, READOUT, layer, C))["test_run_score"][:, 0]
                k, inv = np.unique(t["run"][te], return_inverse=True)
                means = np.bincount(inv, s) / np.bincount(inv)
                check[f] = {"layer": layer, "C": C, "evals": int(evals), "corr_with_saved": float(np.corrcoef(means, saved[k])[0, 1]),
                            "grad_max": grad, "converged": grad <= G.tol_of(C)}
                for j, i in enumerate(np.flatnonzero(te)):
                    r = runs[t["run"][i]]
                    rows.append({"key": r["key"], "arm": r["arm"], "environment": r["environment"], "main_task": r["main_task"], "cell": r["cell"],
                                 "step": int(t["step"][i]), "action": int(t["action"][i]), "fold": f, "score": float(s[j]),
                                 "n_tokens": int(n_tok[(int(t["run"][i]), int(t["step"][i]), int(t["action"][i]))])})
                del Xa, Xb
            del X
        out[comp] = {"rows": rows, "check": check}
    p = Path("/acts/claimB") / LEVEL / f"output_scores_max_{cfg['packs'][LEVEL]}.json.gz"  # a copy in case the client dies
    p.write_bytes(gzip.compress(json.dumps(out).encode(), mtime=0))
    G.vol.commit()
    return out


def analyze(rows: list[dict], pos: str, unit: str, rng) -> dict:
    """Run-level AUROC of the mean and the max output score, the length baseline, intervals, null and detection."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    by = {}
    for r in rows:
        by.setdefault(r["key"], []).append(r)
    keys = sorted(by)
    y = np.array([by[k][0]["arm"] == pos for k in keys])
    task = np.array([by[k][0]["main_task"] for k in keys])
    grp = np.array([by[k][0]["key"].removesuffix("__plant1").removesuffix("__plant2") if unit == "pair" else by[k][0]["cell"] for k in keys])
    pairs = [np.flatnonzero(grp == g) for g in np.unique(grp)]
    pairs = [p for p in pairs if len(p) == 2 and y[p].sum() == 1]
    coin = {p[0]: (task[p[0]] if unit == "task" else grp[p[0]]) for p in pairs}
    out = {}
    for name, agg in (("mean", lambda v: float(np.mean([x["score"] for x in v]))), ("max", lambda v: float(np.max([x["score"] for x in v]))),
                      ("length_mean_log_tokens", lambda v: float(np.mean([np.log(max(x["n_tokens"], 1)) for x in v])))):
        s = np.array([agg(by[k]) for k in keys])
        auc, tasks, boot, null = roc_auc_score(y, s), np.unique(task), [], []
        ix = {t_: np.flatnonzero(task == t_) for t_ in tasks}
        for _ in range(1000):
            i = np.concatenate([ix[t_] for t_ in rng.choice(tasks, len(tasks))])
            if len(set(y[i])) == 2:
                boot.append(roc_auc_score(y[i], s[i]))
            flips = {c: rng.random() < 0.5 for c in set(coin.values())}
            yy = y.copy()
            for p in pairs:
                if flips[coin[p[0]]]:
                    yy[p] = yy[p[::-1]]
            null.append(roc_auc_score(yy, s))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = {"auroc": auc, "ci95": [float(lo), float(hi)], "null_p95": float(np.percentile(null, 95)), "runs": len(keys)}
    return out


@app.local_entrypoint()
def main(pack: str, grid_result: str):
    import subprocess

    import numpy as np

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CLAIMB, capture_output=True, text=True, check=True).stdout.strip()
    assert head.startswith(PINNED), head
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    grid = json.loads(Path(grid_result).read_text())
    cfg = {"packs": {LEVEL: pack}, "seed": G.SEED, "bootstrap": G.BOOT, "tag": grid["meta"]["code_tag"],
           "main_task_to_fold": json.loads((CLAIMB / "splits" / "folds_v2.json").read_text())["task_folds"]["main_task_to_fold"],
           "environment_to_fold": json.loads((CLAIMB / "splits" / "folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]}
    choices = {c: v["by_readout"][READOUT]["best"]["choices"] for c, v in grid["levels"][LEVEL].items()}
    res = scores.remote(cfg, choices)
    rng, out = np.random.default_rng(0), {"git": git, "grid_result": Path(grid_result).name, "pack": pack, "rows": {}}
    for comp, r in res.items():
        bad = {f: c for f, c in r["check"].items() if not np.isfinite(c["corr_with_saved"]) or c["corr_with_saved"] < 0.999}
        if bad:
            raise ValueError(f"{comp}: refits do not reproduce the grid's run scores: {bad}")
        pos = G.COMPS[comp][1][0]
        test = [x for x in r["rows"] if x["environment"] in G.TEST_ENVS]
        a = analyze(test, pos, "pair" if comp.startswith("plant") else "task", rng)
        sel = grid["levels"][LEVEL][comp]["by_readout"][READOUT]["best"]
        a["mean"]["selection_aware_null"] = sel["test_envs"]["null"] | {"null_invalid": sel["null_invalid"]}
        converged = all(c["converged"] for c in r["check"].values())
        if comp in BARS:  # the mean uses the grid's selection-aware null; the post-hoc max its fixed-choice null
            for name, p95 in (("mean", sel["test_envs"]["null"].get("null_p95")), ("max", a["max"]["null_p95"])):
                v = a[name]
                ok = converged and not (name == "mean" and sel["null_invalid"]) and p95 is not None
                v["detected"] = bool(v["auroc"] >= BARS[comp] and v["ci95"][0] > 0.5 and v["auroc"] > p95) if ok else "no verdict: unconverged fit or invalid null"
        out["rows"][comp] = a | {"refit_check": r["check"]}
        print(comp, {k: (round(v["auroc"], 4), [round(x, 4) for x in v["ci95"]], round(v["null_p95"], 4), v.get("detected")) for k, v in a.items()})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "data" / f"claimB_max_output_scores_{stamp}.json.gz", "xb") as fh:  # keys and scores, no text
        fh.write(gzip.compress(json.dumps(res).encode(), mtime=0))
    with open(HERE / "results" / f"claimB_max_scores_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", HERE / "results" / f"claimB_max_scores_{stamp}.json")
