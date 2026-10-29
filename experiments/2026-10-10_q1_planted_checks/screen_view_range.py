"""PLAN.md Amendment 3: the frozen level-7 probe's sensitivity screen on existing data. One 16 GB H100 container on a deployed
app.

- **Refit and gate.** The frozen probe is refit exactly as score_frozen.py does: the C search's earliest-good model per
  held-out environment fold, on Q1's level-7 pack d0c46e0e5516. The gate (pooled held-out AUROC 0.5703 ± 0.002, every
  fold converged) is checked first; if it fails, nothing is scored.
- **Error outputs.** The tool outputs of each level-7 simulated (and, as a diagnostic, real e1) run whose staged text
  (`2026-10-06_la_segment-end-scrub7_qwen3-32b_v1`) matches ERR, keyed by (step, action). The text is read only in the
  container; only keys and counts are returned.
- **Per run.** Every run is scored by its own fold's model (its environment held out), per output_end row. For a run
  with at least one error output and one other output in the pack, d_run = mean logit over its error outputs − mean
  over its other outputs.
- The interval and verdict come from analyze_screen.py.

  modal deploy experiments/2026-10-10_q1_planted_checks/screen_view_range.py
  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with pyyaml python \
      experiments/2026-10-10_q1_planted_checks/screen_view_range.py spawn       # prints the call id
  ... screen_view_range.py get <call id>                                         # writes results/screen_view_range_<stamp>.json
"""

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in the container the script sits at /root
CGRID = ROOT / "experiments" / "2026-10-09_q1_method_checks"
APP = "probe-shift_aswin-surya_sub-q1plants_screen_kill-2026-11-10"
STAGED = "2026-10-06_la_segment-end-scrub7_qwen3-32b_v1"
ERR = re.compile(r"Invalid `?view_range")  # Amendment 3, fixed before scoring
FOLD_CHOICES = {0: (19, 1e-3), 1: (41, 1e-3), 2: (47, 1e-4), 3: (44, 1e-5), 4: (41, 1e-4), 5: (19, 1e-3), 6: (41, 1e-3), 7: (52, 1e-4),
                8: (44, 1e-4), 9: (47, 1e-3)}  # as score_frozen.py (22db8abd's earliest-good choices)
EXPECTED_AUROC, GATE, MAX_ITER, MAX_ITER_CAP = 0.5703, 0.002, 5000, 80000

app = modal.App(APP)
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scikit-learn==1.7.2", "scipy", "pyyaml")
         .add_local_python_source("probe_shift").add_local_file(CGRID / "run_q1_cgrid.py", "/root/run_q1_cgrid.py"))


def output_index(messages: list[dict]) -> dict:
    """(step, action) of each tool output -> its message index, counted as probe_shift.reader._layout counts them."""
    out, step, action = {}, -1, 0
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            step, action = step + 1, 0
        elif m["role"] == "tool":
            out[(step, action)] = i
            action += 1
    return out


@app.function(image=image, gpu="H100!", cpu=4, memory=16384, volumes={"/acts": vol}, timeout=3600, retries=0)
def screen(cfg: dict) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from probe_shift import torch_probe as tp
    from run_q1_cgrid import PACKED, POS, col_auc, setup

    # error outputs, from the staged level-7 texts (text stays here)
    err, counts = {}, Counter()
    for p in sorted((Path("/acts") / STAGED / "inputs").glob("shard_*.jsonl")):
        with p.open() as fh:
            for line in fh:
                it = json.loads(line)
                if it["arm"] not in ("sim-attack", "real-attack-e1"):
                    continue
                where = output_index(it["messages"])
                hits = {sa for sa, i in where.items() if isinstance(it["messages"][i]["content"], str) and ERR.search(it["messages"][i]["content"])}
                counts[f"{it['arm']}: runs read"] += 1
                if (it["arm"], it["key"]) in err:
                    raise ValueError(f"{it['arm']} {it['key']}: two staged records")
                err[(it["arm"], it["key"])] = hits  # every staged run, zero hits included
                if hits:
                    counts[f"{it['arm']}: runs with an error output"] += 1
                    counts[f"{it['arm']}: error outputs in the staged text"] += len(hits)

    by_layer, X_of, fits, models, test_score = {}, {}, {}, {}, None

    def load(layer):
        nonlocal X_of
        if layer not in by_layer:
            by_layer[layer] = setup(cfg, layer, "real_vs_sim")
        if layer not in X_of:
            X_of = {layer: torch.from_numpy(np.load(by_layer[layer][0] / f"L{layer}_{POS}.npy")).cuda().double()}
        return by_layer[layer], X_of[layer]

    # pass 1: the frozen probe's refit and its gate (as score_frozen.py)
    for f, (layer, C) in sorted(FOLD_CHOICES.items()):
        (src, runs, t, sc, _, env), X = load(layer)
        if test_score is None:
            test_score, run_keys = np.full(len(runs), np.nan), [(m["arm"], m["key"]) for m in runs]
        if [(m["arm"], m["key"]) for m in runs] != run_keys:
            raise ValueError(f"L{layer}: the pack's runs differ from the first fold's")
        use, efold = sc.runs_in[t["run"]], env[t["run"]]
        tr, te = use & (efold != f), use & (efold == f)
        Xa = X[torch.from_numpy(np.flatnonzero(tr)).cuda()]
        mean, std = Xa.mean(0), Xa.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        Xa = (Xa - mean) / std * C**0.5
        Ya = torch.from_numpy(sc.y_run[t["run"][tr]].reshape(-1, 1)).cuda().double()
        tol, mi, prev = 1e-4 * min(C**0.5, 1.0), MAX_ITER, None
        while True:
            W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol)
            Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
            loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
            loss.backward()
            g = float(torch.maximum(Wg.grad.abs().amax(), bg.grad.abs().amax()))
            if g <= tol or evals < mi or evals == prev or mi * 4 > MAX_ITER_CAP:
                break
            mi, prev = mi * 4, evals
        del Xa
        fits[f] = {"layer": layer, "C": C, "evals": int(evals), "grad_max": g, "tol": tol, "converged": g <= tol}
        models[f] = (layer, C, mean, std, W, b)
        rows_te = np.flatnonzero(te)
        s = tp.decision((X[torch.from_numpy(rows_te).cuda()] - mean) / std * C**0.5, W, b)[:, 0].cpu().numpy()
        k, inv = np.unique(t["run"][rows_te], return_inverse=True)
        test_score[k] = np.bincount(inv, weights=s) / np.bincount(inv)
    sc = by_layer[FOLD_CHOICES[0][0]][3]
    r_in = np.flatnonzero(sc.runs_in)
    auroc = float(col_auc(sc.y_run[r_in].reshape(-1, 1), test_score[r_in].reshape(-1, 1))[0])
    converged = all(v["converged"] for v in fits.values())
    gate = {"pooled_held_out_auroc": auroc, "expected": EXPECTED_AUROC, "tolerance": GATE, "all_fits_converged": converged,
            "passed": abs(auroc - EXPECTED_AUROC) <= GATE and converged, "packed": PACKED}
    if not gate["passed"]:
        return {"gate": gate, "fits": fits, "counts": dict(counts), "runs": None}

    # pass 2: every run with error outputs, scored by its own fold's model, error outputs against its other outputs
    out = []
    for f, (layer, C, mean, std, W, b) in sorted(models.items()):
        (src, runs, t, sc, _, env), X = load(layer)
        use, efold = sc.runs_in[t["run"]], env[t["run"]]
        for i in sorted({int(r) for r in t["run"][use & (efold == f)]}):
            if (runs[i]["arm"], runs[i]["key"]) not in err:  # every screened run must have its staged text
                raise ValueError(f"{runs[i]['arm']} {runs[i]['key']}: no staged record")
            hits = err[(runs[i]["arm"], runs[i]["key"])]
            if not hits:
                continue
            rows = np.flatnonzero(t["run"] == i)
            s = tp.decision((X[torch.from_numpy(rows).cuda()] - mean) / std * C**0.5, W, b)[:, 0].cpu().numpy()
            is_err = np.array([(int(a), int(c)) in hits for a, c in zip(t["step"][rows], t["action"][rows])])
            rec = {"arm": runs[i]["arm"], "cell": runs[i]["cell"], "main_task": runs[i]["main_task"], "environment": runs[i]["environment"],
                   "fold": f, "layer": layer, "C": C, "error_outputs_staged": len(hits), "error_outputs": int(is_err.sum()),
                   "other_outputs": int((~is_err).sum())}
            if is_err.any() and (~is_err).any():
                rec |= {"mean_error": float(s[is_err].mean()), "mean_other": float(s[~is_err].mean()), "d": float(s[is_err].mean() - s[~is_err].mean())}
            out.append(rec)
    return {"gate": gate, "fits": fits, "counts": dict(counts), "runs": out}


def main(argv: list[str]):
    import subprocess

    sys.path.insert(0, str(CGRID))
    from run_q1_cgrid import level_cfg

    if argv[0] == "spawn":
        call = modal.Function.from_name(APP, "screen").spawn(level_cfg("level7"))
        print(json.dumps({"call_id": call.object_id}))
        return
    r = modal.FunctionCall.from_id(argv[1]).get()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with (HERE / "results" / f"screen_view_range_{stamp}.json").open("x") as fh:
        fh.write(json.dumps({"git": git, "call_id": argv[1], "regex": ERR.pattern, "fold_choices": FOLD_CHOICES} | r, indent=1))
    print(json.dumps({"gate": r["gate"], "counts": r["counts"], "runs": None if r["runs"] is None else len(r["runs"]),
                      "file": f"screen_view_range_{stamp}.json"}))


if __name__ == "__main__":
    main(sys.argv[1:])
