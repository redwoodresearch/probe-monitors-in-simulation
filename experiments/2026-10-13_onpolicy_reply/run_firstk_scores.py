"""Amendment 3's per-output scores for cell (b) (RULE.md): one Modal CPU container returning, per variant, every output's
out-of-fold score at its fold's selected candidate, with its run and turn order. Gate: per run, the mean over all
outputs equals the stored out-of-fold run score (column 0) within 1e-6, else that variant is returned as failed.

- swapped (level 1): claim C's selection; the pinned refit weights `plantslip_2e/claimc_refit_reply_end.npz` on claim C's
  cache `claimc_cache_v1` (score = ((x - mean) / std) . w + b; w already carries sqrt(C), as in score_p78); stored run scores from `claimc_fb82f67348ac`.
- unswapped (levels 1 and 0): the retrain records' `*.outscores.npz` (column 0) at the selected candidates of stage 1
  (results/stage1_cell_b_POSTHOC_2026-10-13_120555.json, choices_real_column).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run experiments/2026-10-13_onpolicy_reply/run_firstk_scores.py
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-onpol-reply_firstk_kill-2026-11-13"
SCORE = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
STAGE1 = "stage1_cell_b_POSTHOC_2026-10-13_120555.json"
VARIANTS = {"unswapped_L1": ("retrain_qwen_5eb808288f50/level1/attempted", "b_unswapped_L1"),
            "unswapped_L0": ("retrain_qwen_5eb808288f50/level0/attempted", "b_unswapped_L0")}
app = modal.App(APP)
vol = modal.Volume.from_name(SCORE, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, cpu=8, memory=65536, timeout=1800, retries=0, volumes={"/score": vol})
def per_output(choices: dict) -> dict:
    import numpy as np

    root, out = Path("/score"), {}
    # swapped: refit weights on claim C's cache
    W = np.load(root / "plantslip_2e/claimc_refit_reply_end.npz")
    meta = json.loads(str(W["meta"]))
    ref = np.load(root / "claimc_fb82f67348ac/L63_output_end_C0.01.npz")
    runs, folds = ref["on_runs"].astype(str), ref["on_fold"]
    fold_of = dict(zip(runs, folds))
    stored = {}
    for f in range(5):
        c = meta[str(f)]
        rec = np.load(root / f"claimc_fb82f67348ac/L{c['layer']}_output_end_C{c['C']:g}.npz")
        for u, fo, s in zip(rec["on_runs"].astype(str), rec["on_fold"], rec["on_score"][:, 0]):
            if fo == f:
                stored[u] = float(s)
    rows_s, sc_s = [], []
    for layer in sorted({meta[str(f)]["layer"] for f in range(5)}):
        rows = json.loads((root / "claimc_cache_v1" / f"on_rows.L{layer}.json").read_text())
        X = np.load(root / "claimc_cache_v1" / f"on_L{layer}.npy", mmap_mode="r")
        for f in range(5):
            c = meta[str(f)]
            if c["layer"] != layer:
                continue
            idx = [i for i, r in enumerate(rows) if r["kind"] == "output_end" and fold_of.get(f"{r['world']}|{r['task']}") == f]
            x = (np.asarray(X[idx], np.float64) - W[f"{f}|mean"]) / W[f"{f}|std"]
            s = x @ W[f"{f}|w"] + c["b"]  # the saved w already carries sqrt(C), as score_p78 applies it
            rows_s += [[f"{rows[i]['world']}|{rows[i]['task']}", *rows[i]["order"]] for i in idx]
            sc_s += s.tolist()
    out["swapped_L1"] = {"rows": rows_s, "scores": sc_s, "stored_run_scores": stored}
    # unswapped: stored per-output scores
    for name, (d, _) in VARIANTS.items():
        rows_u, sc_u, st = [], [], {}
        for f in range(5):
            layer, C = choices[name][f]
            rec = np.load(root / d / f"L{layer}_output_end_C{C:g}.npz")
            o = np.load(root / d / f"L{layer}_output_end_C{C:g}.outscores.npz")
            if not np.array_equal(rec["on_runs"], o["on_runs"]):
                raise ValueError(f"{name}: record and outscores runs differ")
            ru = rec["on_runs"].astype(str)
            for u, fo, s in zip(ru, rec["on_fold"], rec["on_score"][:, 0]):
                if fo == f:
                    st[u] = float(s)
            m = rec["on_fold"][o["row_run"]] == f
            rows_u += [[ru[k], int(a), int(b)] for k, (a, b) in zip(o["row_run"][m], o["row_order"][m])]
            sc_u += o["out_scores"][m, 0].astype(float).tolist()
        out[name] = {"rows": rows_u, "scores": sc_u, "stored_run_scores": st}
    for name, v in out.items():  # the gate: run means over all outputs equal the stored out-of-fold run scores
        tot, cnt = {}, {}
        for (u, *_), s in zip(v["rows"], v["scores"]):
            tot[u] = tot.get(u, 0.0) + s
            cnt[u] = cnt.get(u, 0) + 1
        common = [u for u in tot if u in v["stored_run_scores"] and np.isfinite(v["stored_run_scores"][u])]
        dmax = max(abs(tot[u] / cnt[u] - v["stored_run_scores"][u]) for u in common)
        v["gate"] = {"runs_compared": len(common), "max_abs_diff": float(dmax), "passes": bool(dmax <= 1e-6)}
    return out


@app.local_entrypoint()
def main():
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE).stdout:
        raise SystemExit("uncommitted tracked changes")
    s1 = json.loads((HERE / "results" / STAGE1).read_text())["cells"]
    choices = {name: s1[key]["choices_real_column"] for name, (_, key) in VARIANTS.items()}
    res = per_output.remote(choices)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"firstk_per_output_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 3); per-output out-of-fold scores", "git": git, "variants": res}))
    print({k: v["gate"] for k, v in res.items()}, dest.name)
