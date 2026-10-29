"""Item 1, the cheaper design (Aswin just before 15:07 PT via 0d: "First size a cheaper design ($0)"): the 1% alarm
threshold is FIXED from old real runs, and new runs only check the catch and the achieved false-alarm rate. Written
before it runs; $0; reads saved out-of-fold scores (numbers only).

What a fixed threshold assumes: the real runs' score distribution transfers from the old tasks to the new ones. So the
achieved false-alarm rate on the new reals is part of the rule.

Rule (frozen in PLAN.md):
- Transfer check: each model's achieved false-alarm rate on the new reals is <= 3%, at most 3x the 1% target. Otherwise
  the result is "too few: the threshold did not transfer", and no gap is read.
- Pass: the paired gap (M2 - A0, share of new sims caught) has a one-sided 95% lower bound > 0.
- Fail: the gap's two-sided 95% interval lies inside +-0.0845 (half the test-environment lead).
- Too few: otherwise.
Intervals for sizing: normal, with the gap's spread across this simulation's draws, which includes the threshold's own
noise from the old-real set (the planned analysis would bound the gap by main-task draws that redo the threshold from
the old reals in each draw).

Emulation per scenario (test: H2's 473 test cells; dev: the dev check's 466 cells), per draw: split the MAIN TASKS at
random into an "old" share (its cells' reals set each model's threshold by the 1% table's rule) and a "new" pool, so
no task crosses; draw n new cells with replacement from the pool's cells; flag the new sims and new reals against the
fixed thresholds. Old shares: 1/2 and 2/3 of the tasks (about 235 and 315 old reals). The real design would calibrate
on more old reals (946 test-environment e1 and e2 runs scored by dev-trained models, or old reals plus new real-only
negatives), so its threshold noise is lower: these powers are conservative on that count. The split draws old and new
tasks from one population, so it cannot show an old-to-new environment shift; that is what the transfer check is for.
The interval is the same normal approximation as v2 (one across-draw spread), not the planned bootstrap's own.
1,000 draws per (scenario, variant, n), default_rng(20261011).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy \\
      --with pyyaml python experiments/2026-10-11_claimB_m2_new_runs/analyze_power_fixed_threshold.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_power_new_runs_v2 as V  # noqa: E402  (scenario loading, as sized there)

A = V.A
NS, OLDS, DRAWS, M, FPR_MAX = (55, 110, 165, 220, 330, 440), (1 / 2, 2 / 3), 1000, 0.0845, 0.03


def load():
    """The two scenarios' per-run scores (real labels), as analyze_power_new_runs_v2.main builds them."""
    import io

    import modal

    summary = json.loads(V.SUMMARY.read_text())
    vol = modal.Volume.from_name(A.H.VOL, version=2)
    read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
    folds_file = json.loads((A.CLAIMB / "splits/folds_v2.json").read_text())
    cfg = {"packs": {"level1": "39f842231a46"}, "seed": 0, "bootstrap": 1000, "tag": "h2",
           "main_task_to_fold": folds_file["task_folds"]["main_task_to_fold"], "environment_to_fold": folds_file["environment_folds"]["environment_to_fold"]}
    runs = json.loads(read(f"{A.H.PACK_A}/runs.L63.json"))
    env = np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])
    sc, Y = A.labels(cfg, runs, "fixed_vs_e1")
    data = {}
    for scen, envs, src, names, shas in (("test", A.H.TEST_ENVS, A.H.OUT, summary["eligible"], {n: summary["inputs"][f"fixed_vs_e1/{n}.npz"] for n in summary["eligible"]}),
                                         ("dev", V.DEV_ENVS, V.DEV_OUT, ["M2", "A0"], V.DEV_SHA)):
        folds = sorted({cfg["environment_to_fold"][e] for e in envs})
        per = {}
        for name in names:
            raw = read(f"{src}/fixed_vs_e1/{name}.npz")
            if hashlib.sha256(raw).hexdigest() != shas[name]:
                raise SystemExit(f"{scen} {name}.npz is not the file the result used")
            z = np.load(io.BytesIO(raw))
            per[name] = (z["val_auroc"], z["test_run_score"])
        sel = lambda cfgs: A.select(np.concatenate([per[n][0] for n in cfgs]), np.concatenate([per[n][1] for n in cfgs]), env, folds)[:, 0]  # noqa: E731
        test = sc.runs_in & np.isin(env, folds)
        data[scen] = {"m2": sel(summary["eligible"] if scen == "test" else ["M2"]), "a0": sel(["A0"]), "y": Y[:, 0],
                      "cell": np.array([r["cell"] for r in runs]), "test": test}
    return data


def main():
    data = load()
    out = {"label": "item 1, fixed threshold from old reals: power (pricing only)", "rule": {"fpr_max": FPR_MAX, "margin": M}, "power": {}, "spread": {}}
    rng = np.random.default_rng(20261011)
    for scen, d in data.items():
        idx = np.flatnonzero(d["test"])
        sim_of = {d["cell"][i]: i for i in idx if d["y"][i]}
        real_of = {d["cell"][i]: i for i in idx if not d["y"][i]}
        cells = sorted(set(sim_of) & set(real_of))
        tasks = sorted({c.split(":")[0] for c in cells})
        by_task = {t: [j for j, c in enumerate(cells) if c.split(":")[0] == t] for t in tasks}
        for old_n in OLDS:
            for n in NS:
                gaps, ok = [], []
                for _ in range(DRAWS):
                    perm = rng.permutation(len(tasks))
                    cut = int(round(old_n * len(tasks)))
                    old = np.array([j for t in perm[:cut] for j in by_task[tasks[t]]])
                    pool = np.array([j for t in perm[cut:] for j in by_task[tasks[t]]])
                    new = rng.choice(pool, n)
                    flags, fpr = {}, {}
                    for nm in ("m2", "a0"):
                        t = A.T.threshold(d[nm][[real_of[cells[j]] for j in old]], 1)[0]
                        flags[nm] = d[nm][[sim_of[cells[j]] for j in new]] > t
                        fpr[nm] = float((d[nm][[real_of[cells[j]] for j in new]] > t).mean())
                    gaps.append(float(flags["m2"].mean() - flags["a0"].mean()))
                    ok.append(max(fpr.values()) <= FPR_MAX)
                g, ok = np.array(gaps), np.array(ok)
                sd = float(g.std())
                p = g - 1.645 * sd > 0
                f = (g - 1.96 * sd > -M) & (g + 1.96 * sd < M) & ~p
                key = f"{scen}_old{old_n:.2f}_n{n}"
                out["power"][key] = {"pass": float((p & ok).mean()), "fail": float((f & ok).mean()), "too few": float(1 - ((p | f) & ok).mean()),
                                     "transfer_check_passes": float(ok.mean())}
                out["spread"][key] = {"gap_mean": float(g.mean()), "gap_sd": sd}
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "v2_script_sha256": hashlib.sha256(Path(V.__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"power_fixed_threshold_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    for k, v in out["power"].items():
        print(k, {a: round(b, 3) for a, b in v.items()}, "gap", round(out["spread"][k]["gap_mean"], 4), "sd", round(out["spread"][k]["gap_sd"], 4))
    print("->", dst)


if __name__ == "__main__":
    main()
