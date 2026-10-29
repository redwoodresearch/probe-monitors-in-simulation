"""Date decodability, the reading (PLAN.md bars, fixed before the run): the positive control, T1 at level 0 and level 1
against the 40-draw label-swap null, and T2 against step B's null p95. Local and free; writes one result file.

  uv run --no-project --with numpy --with scikit-learn python experiments/2026-10-11_q2_date_decodability/analyze_date_decode.py <date_decode output>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
TEST_ENVS = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]
B_NULL_P95 = (0.514, 0.525)  # step B's null p95 range for raw real e2 vs e1 (claimB_table, the pooled readouts)


def main(path: Path):
    d = json.loads(path.read_text()); u = d["units"]
    nulls = sorted(u[f"null {i}"]["AUROC"] for i in range(1, 41) if f"null {i}" in u)
    if len(nulls) < 20:
        raise ValueError(f"only {len(nulls)} null draws; the null bars need at least 20")
    lo95, p95 = float(np.percentile(nulls, 2.5)), float(np.percentile(nulls, 95))
    t0, t1, pc = u["T1 level0"]["AUROC"], u["T1 level1"]["AUROC"], u["plant"]["AUROC"]
    if pc < 0.80:
        verdict = "not interpretable (positive control failed)"
    elif t0 >= 0.75 and t0 > p95 and t0 - t1 >= 0.15:
        verdict = "decodable"
    elif t0 >= 0.75 and t0 > p95:
        verdict = "separable at both levels (not date-specific)"
    elif t0 <= 0.60 or lo95 <= t0 <= float(np.percentile(nulls, 97.5)):
        verdict = "not detected by this pipeline"
    else:
        verdict = "partial"
    scores, labels, env_of = {}, {}, {}
    for k, v in u.items():
        if k.startswith("T2 "):
            scores |= v["run scores"]; labels |= v["labels"]; env_of |= {r: v["environment"] for r in v["run scores"]}
    def auc(envs):
        ks = [k for k in scores if env_of[k] in envs]
        return float(roc_auc_score([labels[k] for k in ks], [scores[k] for k in ks]))
    present = sorted(set(env_of.values())); test_present = [e for e in TEST_ENVS if e in present]
    if not test_present:
        raise ValueError("no T2 test-environment fold finished")
    t2_test, t2_all = auc(test_present), auc(present)
    t2_env = {e: auc([e]) for e in test_present}
    out = {"input": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "units done": f"{d.get('units done')} of {d.get('units planned')}",
           "clock outputs (e2)": [d["clock outputs"], d["clock outputs e2"]], "plant pairs": d["plant pairs"],
           "positive control (plant, level 1)": {k: u["plant"][k] for k in ("AUROC", "AUROC per fold", "AUROC pooled (mixes fold scales)", "planted above twin", "picks", "fits not converged")},
           "T1 level 0": {k: u["T1 level0"][k] for k in ("AUROC", "AUROC per fold", "AUROC pooled (mixes fold scales)", "picks", "fits not converged")},
           "T1 level 1 (times masked)": {k: u["T1 level1"][k] for k in ("AUROC", "AUROC per fold", "AUROC pooled (mixes fold scales)", "picks", "fits not converged")},
           "T1 null (40 within-main-task swaps, selection repeated)": {"values": nulls, "2.5%": lo95, "95%": p95, "max": nulls[-1],
                                                                       "draws": len(nulls), "p (level 0)": (1 + sum(n >= t0 for n in nulls)) / (1 + len(nulls))},
           "T1 verdict (PLAN bars)": verdict,
           "T2 run-label probe, C = 100, run-mean features": {"test environments, pooled": t2_test, "per test environment": t2_env,
                                                              "mean of per-environment": float(np.mean(list(t2_env.values()))), "all environments present, pooled": t2_all, "environments present": present,
                                                              "complete": len(present) == 10,
                                                              "picks": {v["environment"]: [v["readout"], v["layer"]] for k, v in u.items() if k.startswith("T2 ")},
                                                              "step B null p95 (fixed reference, not selection-aware)": B_NULL_P95,
                                                              "verdict": "weak regularization finds it" if t2_test > 0.55 and t2_test > max(B_NULL_P95) else "not found at C = 100"}}
    dest = HERE / "results" / f"date_decode_{d['stamp']}.json"; dest.parent.mkdir(exist_ok=True)
    with open(dest, "x") as fh:
        fh.write("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
    print(f"positive control {pc:.3f} ({u['plant']['planted above twin']} planted above twin)")
    print(f"T1 level 0 {t0:.3f}, level 1 {t1:.3f}; null 95th {p95:.3f}, max {nulls[-1]:.3f}, p {out['T1 null (40 within-main-task swaps, selection repeated)']['p (level 0)']:.3f}; verdict: {verdict}")
    print(f"T2 test environments {t2_test:.3f} (per-env mean {out['T2 run-label probe, C = 100, run-mean features']['mean of per-environment']:.3f}), all {len(present)} present {t2_all:.3f}; "
          f"{out['T2 run-label probe, C = 100, run-mean features']['verdict']}")
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
