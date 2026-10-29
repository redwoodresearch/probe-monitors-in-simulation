"""H2 dev-environment robustness check, summary (PLAN.md "Dev-environment robustness check (Aswin ~13:56 PT)"; written
before any run). Reads run_h2_devcheck_modal.py's saved score arrays from the Volume (numbers only) and applies the
committed rule. A robustness check, not an untouched confirmation; H2's catch-at-1% lead stays "unconfirmed".

- Scores: A0 with step A's selection (best layer and C per dev fold and column on validation AUROC); M2 frozen (one
  sub-configuration). Dev cells: fixed sim vs real e1 runs in the 5 dev environments.
- Catch at 1% and 5%: the 1% table's two-way cross-fit (analyze_tpr_at_fpr.crossfit, as analyze_h2_search.catch), with
  achieved false-alarm rates; AUROC beside.
- M2's (and A0's) 100-shuffle null for catch: median, p95, p = (1 + shuffles >= observed) / (1 + shuffles).
- The paired gap M2 - A0 at 1%: 2,000 main-task draws that redo the cross-fit calibration in every draw
  (analyze_h2_search.catch_diff), and the gap with each dev environment left out.
- Rule: "supports the lead" only if (1) the gap's 95% lower bound > 0, (2) M2's p <= 0.05, (3) every leave-one-out
  gap > 0. If the gap's interval crosses 0: "inconclusive". Otherwise "not supported".

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy \\
      --with pyyaml --with matplotlib python experiments/2026-10-11_claimB_h2_probe_best_found/analyze_h2_devcheck.py
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
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_h2_search as A  # noqa: E402  (labels, select, catch, catch_diff, T)
import run_h2_devcheck_modal as D  # noqa: E402  (the dev-check runner: folds, OUT, identity)

COMP, PCTS = "fixed_vs_e1", A.T.TARGETS


def main():
    vol = modal.Volume.from_name(D.VOL, version=2)
    read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
    folds_file = json.loads((A.CLAIMB / "splits/folds_v2.json").read_text())
    cfg = {"packs": {"level1": "39f842231a46"}, "seed": 0, "bootstrap": 1000, "tag": "h2",
           "main_task_to_fold": folds_file["task_folds"]["main_task_to_fold"], "environment_to_fold": folds_file["environment_folds"]["environment_to_fold"]}
    runs = json.loads(read(f"{D.PACK_A}/runs.L63.json"))
    env = np.array([cfg["environment_to_fold"][m["environment"]] for m in runs])
    envname = np.array([m["environment"] for m in runs])
    folds = sorted({cfg["environment_to_fold"][e] for e in D.TEST_ENVS})
    ident = D.ident_of(cfg, D.code_sha())
    sc, Y = A.labels(cfg, runs, COMP)  # the same 101 columns as the runner (H2's seed rule)
    test = sc.runs_in & np.isin(env, folds)
    y, task, names_e = Y[test], sc.main[test], envname[test]
    out = {"label": "H2 dev-environment robustness check (not an untouched confirmation; the 1% lead stays unconfirmed)", "inputs": {},
           "dev_environments": D.TEST_ENVS, "sim_runs": int(y[:, 0].sum()), "real_runs": int((~y[:, 0]).sum()), "main_tasks": int(len(set(task)))}
    S = {}
    for name in ("A0", "M2"):
        raw = read(f"{D.OUT}/{COMP}/{name}.npz")
        z = np.load(io.BytesIO(raw))
        if str(z["ident"]) != ident or json.loads(str(z["subs"])) != D.subconfigs(name) or list(z["folds"]) != folds:
            raise SystemExit(f"{name}.npz does not match this runner's code, inputs, folds or sub-configurations")
        out["inputs"][f"{COMP}/{name}.npz"] = hashlib.sha256(raw).hexdigest()
        S[name] = A.select(z["val_auroc"], z["test_run_score"], env, folds)[test]
        if not np.isfinite(S[name]).all():
            raise SystemExit(f"{name}: a dev run has no score")
        out[f"{name}_unconverged_fits"] = int((~z["converged"]).sum())
        out[f"{name}_auroc"] = float(roc_auc_score(y[:, 0], S[name][:, 0]))

    def share(c, pct):
        return c[pct]["caught"] / c[pct]["positives"]

    for pct in PCTS:
        row = {}
        for name in ("M2", "A0"):
            c = A.catch(y[:, 0], S[name][:, 0], task)
            nl = np.array([share(x, pct) if (x := A.catch(y[:, k], S[name][:, k], task)) else np.nan for k in range(1, D.K)])
            ok = np.isfinite(nl)
            row[name] = {"caught": share(c, pct), "caught_n": c[pct]["caught"], "positives": c[pct]["positives"],
                         "achieved_fpr": c[pct]["false_alarms"] / c[pct]["negatives"], "false_alarms": c[pct]["false_alarms"], "negatives": c[pct]["negatives"],
                         "null_median": float(np.median(nl[ok])), "null_p95": float(np.percentile(nl[ok], 95)), "null_n": int(ok.sum()),
                         "p": float((1 + (nl[ok] >= share(c, pct)).sum()) / (1 + ok.sum()))}
        ci, n = A.catch_diff(y[:, 0], S["M2"][:, 0], S["A0"][:, 0], task, pct, np.random.default_rng(A.SEED))
        row["gap"] = {"value": row["M2"]["caught"] - row["A0"]["caught"], "ci95": ci, "draws": n}
        loo = {}
        for e in D.TEST_ENVS:
            m = names_e != e
            cm, ca = A.catch(y[m, 0], S["M2"][m, 0], task[m]), A.catch(y[m, 0], S["A0"][m, 0], task[m])
            loo[e] = {"M2": share(cm, pct), "A0": share(ca, pct), "gap": share(cm, pct) - share(ca, pct),
                      "M2_fpr": cm[pct]["false_alarms"] / cm[pct]["negatives"], "A0_fpr": ca[pct]["false_alarms"] / ca[pct]["negatives"]}
        row["leave_one_out"] = loo
        out[f"catch_{pct}pct"] = row
    r = out["catch_1pct"]
    rule = {"gap_lower_bound_above_0": r["gap"]["ci95"][0] > 0, "M2_beats_null": r["M2"]["p"] <= 0.05,
            "every_leave_one_out_gap_positive": all(v["gap"] > 0 for v in r["leave_one_out"].values())}
    crosses = r["gap"]["ci95"][0] <= 0 <= r["gap"]["ci95"][1]
    out["rule"] = rule
    out["reading"] = "supports the lead" if all(rule.values()) else "inconclusive" if crosses else "not supported"
    out["reading_label"] = "robustness check, not an untouched confirmation; the 1% lead stays unconfirmed"
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "runner_ident": ident,
            "tpr_script_sha256": hashlib.sha256((Path.home() / "probe-shift-sims/experiments/2026-10-09_fixed_sim/analyze_tpr_at_fpr.py").read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"h2_devcheck_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    plot(out, dst.with_suffix(".png"))
    print(json.dumps({k: out[k] for k in ("sim_runs", "real_runs", "main_tasks", "A0_auroc", "M2_auroc", "rule", "reading")}, indent=1))
    for pct in PCTS:
        row = out[f"catch_{pct}pct"]
        print(f"catch at {pct}%: M2 {row['M2']['caught']:.4f} ({row['M2']['caught_n']}/{row['M2']['positives']}, FPR {row['M2']['achieved_fpr']:.4f}; "
              f"null median {row['M2']['null_median']:.4f} p95 {row['M2']['null_p95']:.4f} p {row['M2']['p']:.4f}) | "
              f"A0 {row['A0']['caught']:.4f} ({row['A0']['caught_n']}/{row['A0']['positives']}, FPR {row['A0']['achieved_fpr']:.4f}; "
              f"null p95 {row['A0']['null_p95']:.4f} p {row['A0']['p']:.4f}) | gap {row['gap']['value']:+.4f} "
              f"[{row['gap']['ci95'][0]:+.4f}, {row['gap']['ci95'][1]:+.4f}]")
        print("  leave one out:", {e: round(v["gap"], 4) for e, v in row["leave_one_out"].items()})
    print("->", dst)


def plot(out, dst):
    """One bar chart: catch at 1% on the dev cells, M2 vs A0, with each one's shuffle-null p95; the gap's interval in the
    title."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = out["catch_1pct"]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for i, (name, label, color) in enumerate((("A0", "step A's probe", "#4c72b0"), ("M2", "M2 (frozen)", "#dd8452"))):
        v = r[name]["caught"]
        ax.bar(i, v, 0.6, color=color)
        ax.text(i, v + 0.005, f"{100 * v:.1f}%\n(FPR {100 * r[name]['achieved_fpr']:.1f}%)", ha="center", va="bottom", fontsize=10)
        ax.plot([i - 0.3, i + 0.3], [r[name]["null_p95"]] * 2, color="k", ls="--", lw=1, label="shuffle null p95" if i == 0 else None)
    ax.set_xticks([0, 1], ["step A's probe", "M2 (frozen)"])
    ax.set_ylabel("sim runs caught at a 1% false-alarm target\n(higher = more caught)")
    ax.set_ylim(0, max(0.1, 1.35 * max(r["M2"]["caught"], r["A0"]["caught"])))
    g = r["gap"]
    ax.set_title(f"dev environments: gap {100 * g['value']:+.1f} pts [{100 * g['ci95'][0]:+.1f}, {100 * g['ci95'][1]:+.1f}]; {out['reading']}", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(dst, dpi=150)


if __name__ == "__main__":
    main()
