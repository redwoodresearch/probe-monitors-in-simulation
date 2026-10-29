"""POSTHOC (after 0d's Codex review of item (A)'s result): the direct yardstick. Held-out real e2 runs are flagged by
the SIM-TRAINED fixed-vs-e1 detector (as selected per direction in item (A)) at the SAME 1% threshold set on all
held-out e1 runs. Gap = sim catch - e2 flagged by that detector, 95% interval from 2,000 main-task draws within each half
with the threshold recomputed per draw (the same seeded draws as analyze_halves.py), per direction and pooled; whether
each gap's upper bound is below 10 pts. Item (A)'s separately trained e2-vs-e1 yardstick is shown beside it, from the
committed result file. Reading (fixed before running): if the direct gap's lower bound is above 0 (with item (A)'s sim
null p < 0.05), "raises alarms beyond real-run noise under both yardsticks"; otherwise "exceeds this separately trained
real-versus-real benchmark" (the original reading) with the direct comparison stated as not beyond. Numbers only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy --with matplotlib \\
      python experiments/2026-10-11_claimB_halves/analyze_POSTHOC_halves_e2.py results/halves_2026-10-11_164208.json
"""

import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent / "2026-10-11_claimB_h3_fix_to_plateau"), str(HERE.parent / "2026-10-11_claimB_slips_evened_probe")]
import analyze_halves as A  # noqa: E402  (thr, share_above, DRAWS, SEED, ALARM_MARGIN)
import run_POSTHOC_halves_e2_modal as PH  # noqa: E402
import run_halves_modal as R  # noqa: E402


def main(result_path):
    orig = json.loads((HERE / result_path).read_text())
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
    runs = json.loads(read(f"{R.OUT}/runs.json"))["runs"]
    env = np.array([r["environment"] for r in runs])
    arm = np.array([r["arm"] for r in runs])
    task = np.array([r["main_task"] for r in runs])
    out = {"label": "POSTHOC: the direct yardstick for item (A) (e2 flagged by the sim-trained detector at the same 1% threshold)",
           "result_file": result_path, "inputs": {}}
    held = {d: np.isin(env, R.DIRS[d][1]) for d in R.DIRS}
    score = {}
    for d in R.DIRS:
        ro, rp = read(f"{R.OUT}/{d}/fixed_vs_e1.npz"), read(f"{PH.OUT}/{d}/fixed_vs_e1.npz")
        out["inputs"][f"{d}/fixed_vs_e1.npz"], out["inputs"][f"POSTHOC_e2/{d}/fixed_vs_e1.npz"] = hashlib.sha256(ro).hexdigest(), hashlib.sha256(rp).hexdigest()
        zo, zp = np.load(io.BytesIO(ro)), np.load(io.BytesIO(rp))
        if str(zo["script_sha256"]) != R.SCRIPT_SHA or str(zp["script_sha256"]) != PH.SCRIPT_SHA:
            raise SystemExit(f"{d}: arrays from another runner version")
        b = int(zo["val_auroc"][:, 0].argmax())  # item (A)'s selection for the real-label column
        if int(zp["val_auroc"][:, 0].argmax()) != b:
            raise SystemExit(f"{d}: the rerun picks another sub-configuration")
        score[d] = zp["test_run_score"][b, :, 0]
    rng = np.random.default_rng(A.SEED)  # the same draws as analyze_halves.py
    halves = {d: sorted(set(task[held[d] & np.isin(arm, [R.SIM, R.E1, R.E2])])) for d in R.DIRS}
    draws = {d: [rng.integers(0, len(halves[d]), len(halves[d])) for _ in range(A.DRAWS)] for d in R.DIRS}

    def shares(d, dr=None):
        m = held[d]
        s, a, t = score[d][m], arm[m], task[m]
        if dr is not None:
            pos = {x: j for j, x in enumerate(halves[d])}
            w = np.bincount(dr, minlength=len(halves[d]))[[pos[x] for x in t]]
            s, a = np.repeat(s, w), np.repeat(a, w)
        th = A.thr(s[a == R.E1])
        return {"sim": A.share_above(s[a == R.SIM], th), "e2": A.share_above(s[a == R.E2], th), "e1_fpr": A.share_above(s[a == R.E1], th),
                "n_sim": int((a == R.SIM).sum()), "n_e2": int((a == R.E2).sum())}
    res = {}
    for d in R.DIRS:
        o = orig["results"][d]
        if "catch_1pct" not in o:
            res[d] = {"reading": "not read (planted check failed in item (A))"}
            continue
        s0 = shares(d)
        g = np.array([(lambda x: x["sim"] - x["e2"])(shares(d, dr)) for dr in draws[d]])
        lo, hi = (float(x) for x in np.percentile(g, [2.5, 97.5]))
        beyond = o["catch_1pct"]["sim_null_p"] < 0.05 and lo > 0
        res[d] = {"sim_catch": s0["sim"], "sim_catch_item_A": o["catch_1pct"]["sim_catch"], "e1_fpr": s0["e1_fpr"],
                  "direct": {"e2_flagged_by_sim_detector": s0["e2"], "gap": s0["sim"] - s0["e2"], "gap_ci95": [lo, hi], "upper_below_10pts": hi < A.ALARM_MARGIN},
                  "separate": {"e2_flagged_by_e2_detector": o["catch_1pct"]["e2_flagged"], "gap": o["catch_1pct"]["gap"], "gap_ci95": o["catch_1pct"]["gap_ci95"],
                               "upper_below_10pts": o["catch_1pct"]["gap_ci95"][1] < A.ALARM_MARGIN},
                  "reading": "raises alarms beyond real-run noise under both yardsticks" if beyond
                  else "exceeds this separately trained real-versus-real benchmark (the direct comparison is not beyond real-run noise)"}
    ds = [d for d in R.DIRS if "direct" in res[d]]
    if len(ds) == len(R.DIRS):
        def pooled(drs=None):
            xs = [shares(d, drs[j] if drs else None) for j, d in enumerate(ds)]
            ns, ne = sum(x["n_sim"] for x in xs), sum(x["n_e2"] for x in xs)
            return sum(x["sim"] * x["n_sim"] for x in xs) / ns, sum(x["e2"] * x["n_e2"] for x in xs) / ne
        ps, pe = pooled()
        pg = np.array([(lambda x: x[0] - x[1])(pooled([draws[d][b] for d in ds])) for b in range(A.DRAWS)])
        lo, hi = (float(x) for x in np.percentile(pg, [2.5, 97.5]))
        op = orig["results"]["pooled"]["catch_1pct"]
        beyond = op["sim_null_p"] < 0.05 and lo > 0
        res["pooled"] = {"sim_catch": ps, "direct": {"e2_flagged_by_sim_detector": pe, "gap": ps - pe, "gap_ci95": [lo, hi], "upper_below_10pts": hi < A.ALARM_MARGIN},
                         "separate": {"e2_flagged_by_e2_detector": op["e2_flagged"], "gap": op["gap"], "gap_ci95": op["gap_ci95"], "upper_below_10pts": op["gap_ci95"][1] < A.ALARM_MARGIN},
                         "reading": "raises alarms beyond real-run noise under both yardsticks" if beyond
                         else "exceeds this separately trained real-versus-real benchmark (the direct comparison is not beyond real-run noise)"}
    out["results"] = res
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "posthoc_runner_sha256": PH.SCRIPT_SHA}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"halves_e2_direct_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    plot(res, dst.with_suffix(".png"))
    print(json.dumps(res, indent=1))
    print("->", dst)


def plot(res, dst):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [d for d in ("dev_to_test", "test_to_dev", "pooled") if "direct" in res.get(d, {})]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for j, d in enumerate(names):
        r = res[d]
        vals = [("fixed sim", r["sim_catch"], "#dd8452"), ("e2, sim-trained detector (direct)", r["direct"]["e2_flagged_by_sim_detector"], "#4c72b0"),
                ("e2, e2-trained detector (separate)", r["separate"]["e2_flagged_by_e2_detector"], "#9aa9c9")]
        for o, (lab, v, col) in enumerate(vals):
            x = j + (o - 1) * 0.27
            ax.bar(x, v, 0.25, color=col, label=lab if j == 0 else None)
            ax.text(x, v + 0.004, f"{v:.1%}", ha="center", fontsize=9)
        gd = r["direct"]
        ax.text(j, -0.04, f"direct gap {gd['gap']:+.1%}\n[{gd['gap_ci95'][0]:+.1%}, {gd['gap_ci95'][1]:+.1%}]", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)), [n.replace("_", " ") for n in names])
    ax.set_ylabel("share flagged at the 1% alarm\n(higher = more alarms)")
    ax.set_ylim(-0.07, max(0.15, max(res[n]["sim_catch"] for n in names) + 0.05))
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)


if __name__ == "__main__":
    main(sys.argv[1])
