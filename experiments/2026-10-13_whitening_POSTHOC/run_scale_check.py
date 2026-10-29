"""Q1(a) whitening, Amendment 1 (RULE.md): was the manipulation strong enough? One detached Modal CPU function on df's full +
F9 averaged pack (L48). Per nested model (f, v), on the rows run_whitening.py fit on, the per-feature SD ratio (real honest
training-fold rows / fake-fake training rows) and the mean shift (real honest mean - training mean) in training-SD units:
5th, 50th and 95th percentiles over features, per model and pooled. Nothing computes on the laptop.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_whitening_POSTHOC/run_scale_check.py --out <results json>
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-whitening_scale-check_kill-2026-11-13"
DFV = "probe-shift_aswin-surya_sub-whysteps_work_kill-2026-11-13"
STAMP, KIND, LAYER = "2026-10-13_185308", "ffavg_output_end_mean", 48
SHA_FEATURES = "7b7b1d579312b3ccdf002a243dc8fdf7610eb3531969fd176f14dce124fc5ae2"
R36 = REPO / "experiments/2026-10-06_q3_attack_detector/results"
H, SA, SH = "real-honest", "sim-attack", "sim-honest"
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
Q = (5, 50, 95)
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=image, cpu=(2, 2), memory=(16384, 16384), timeout=900, retries=0,
              volumes={"/df": modal.Volume.from_name(DFV, version=2).read_only()})
def run(runs: list) -> dict:
    import numpy as np

    arm, fold = np.array([r["arm"] for r in runs]), np.array([r["fold"] for r in runs], int)
    X = np.load(f"/df/packs/{STAMP}/{KIND}_L{LAYER}.npy")
    if hashlib.sha256(X.tobytes()).hexdigest() != SHA_FEATURES:
        raise RuntimeError("features differ from df's record")
    r_run = np.load(f"/df/packs/{STAMP}/rows_{KIND}_L{LAYER}.npz")["run"].astype(np.int64)
    labelled = np.isin(arm, [SA, SH])
    pct = lambda x: [float(q) for q in np.percentile(x, Q)]  # noqa: E731
    per, ratios, shifts = [], [], []
    for f, v in PAIRS:
        te = (fold == f) | (fold == v)
        tr, rh = np.flatnonzero((labelled & ~te)[r_run]), np.flatnonzero(((arm == H) & ~te)[r_run])  # as run_whitening.py
        stats = []
        for rows in (tr, rh):
            Xr = X[rows]
            stats.append((Xr.mean(0, dtype=np.float64), Xr.std(0, dtype=np.float64)))
            del Xr
        (m_tr, s_tr), (m_rh, s_rh) = stats
        zero = {"train": int((s_tr == 0).sum()), "real_honest": int((s_rh == 0).sum())}
        ok = (s_tr > 0) & (s_rh > 0)
        ratio, shift = s_rh[ok] / s_tr[ok], (m_rh[ok] - m_tr[ok]) / s_tr[ok]
        ratios.append(ratio)
        shifts.append(shift)
        per.append({"f": f, "v": v, "train_rows": len(tr), "real_honest_rows": len(rh), "zero_sd": zero,
                    "sd_ratio_p5_p50_p95": pct(ratio), "share_ratio_in_0.9_1.1": float(((ratio >= 0.9) & (ratio <= 1.1)).mean()),
                    "mean_shift_sd_p5_p50_p95": pct(shift), "abs_mean_shift_sd_median": float(np.median(np.abs(shift)))})
    R, S = np.concatenate(ratios), np.concatenate(shifts)
    pooled = {"n_values": len(R), "sd_ratio_p5_p50_p95": pct(R), "share_ratio_in_0.9_1.1": float(((R >= 0.9) & (R <= 1.1)).mean()),
              "mean_shift_sd_p5_p50_p95": pct(S), "abs_mean_shift_sd_median": float(np.median(np.abs(S)))}
    return {"features": int(X.shape[1]), "rows": int(X.shape[0]), "pooled": pooled, "per_model": per}


@app.local_entrypoint()
def main(out: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    keep = ("arm", "fold")
    base = [{k: r[k] for k in keep} for r in json.loads((R36 / "run_scores_2026-10-06_181548/runs.json").read_text())]
    sh = [{k: r[k] for k in keep} for r in json.loads((R36 / "sim_honest_scores_2026-10-07_124717/runs.json").read_text()) if r["arm"] == SH]
    runs = base + sh
    if len(runs) != 4244:
        raise SystemExit("unexpected runs")
    r = run.remote(runs)
    rec = {"label": "POST HOC (RULE.md Amendment 1), preliminary until Codex agrees", "git": git("rev-parse", "HEAD"), "app": APP,
           "df_stamp": STAMP, "sha256_features": SHA_FEATURES} | r
    Path(out).open("x").write(json.dumps(rec, indent=1))
    f3 = lambda x: " / ".join(f"{q:+.3f}" for q in x)  # noqa: E731
    p = r["pooled"]
    lines = [f"L48 averaged pack, {r['rows']} rows x {r['features']} features; percentiles 5 / 50 / 95 over features",
             f"POOLED ({p['n_values']} values): SD ratio real honest / train {f3(p['sd_ratio_p5_p50_p95'])} | share in 0.9-1.1 "
             f"{p['share_ratio_in_0.9_1.1']:.3f} | mean shift in train SDs {f3(p['mean_shift_sd_p5_p50_p95'])} | median |shift| {p['abs_mean_shift_sd_median']:.3f}"]
    for m in r["per_model"]:
        lines.append(f"  f{m['f']} v{m['v']}: rows train {m['train_rows']} real honest {m['real_honest_rows']} | zero SD {m['zero_sd']} | "
                     f"ratio {f3(m['sd_ratio_p5_p50_p95'])} (in 0.9-1.1 {m['share_ratio_in_0.9_1.1']:.3f}) | shift {f3(m['mean_shift_sd_p5_p50_p95'])}")
    txt = "\n".join(lines)
    print(txt)
    Path(out[:-5] + ".printout.txt").open("x").write(txt + "\n")
