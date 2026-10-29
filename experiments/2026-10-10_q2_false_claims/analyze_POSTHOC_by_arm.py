"""False success claims, post hoc re-cut by real arm (the reference audit; orchestrator, 2026-10-11). It was written after
the result was read, so it carries no verdict. Phase 2's simulated runs were built from honest_a runs, and the real side
of "real vs simulated" pools honest_a and honest_b. This recomputes that comparison three ways:
- honest_a only;
- honest_b only (equivalently, honest_a excluded);
- both, which must reproduce the committed v2 numbers exactly.
It uses analyze_false_claims.py's procedure: shared tasks, 10,000 within-task world permutations, one default_rng(0)
across the two simulators in order. It reads the committed result's per-report rows (scores, worlds, arms), not the index.
Local, $0.

  uv run --no-project --with numpy --with scikit-learn python experiments/2026-10-10_q2_false_claims/analyze_POSTHOC_by_arm.py
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
RESULT = HERE / "results/false_claims_analysis_2026-10-10_160736_v2.json"
RESULT_SHA = "661560be983a9376e0ad56927577e0418ba98d667e95b7fa75ff4bba42479a33"
SIMS = ("qwen-sim@september", "qwen-sim@today")


def mc_two_sided(null, obs):  # analyze_false_claims.mc_two_sided
    null = np.asarray(null)
    hi, lo = (1 + np.sum(null >= obs - 1e-12)) / (len(null) + 1), (1 + np.sum(null <= obs + 1e-12)) / (len(null) + 1)
    return float(min(1.0, 2 * min(hi, lo)))


def real_vs_sim(real, claims):
    """analyze_false_claims.py's real-vs-simulated block, for a given real set."""
    rng, out = np.random.default_rng(0), {}
    for w in SIMS:
        sim = [r for r in claims if r["world"] == w]
        allr = real + sim
        yy = np.array([r["world"] == w for r in allr], int); ss = np.array([r["score"] for r in allr])
        shared = {r["task"] for r in sim} & {r["task"] for r in real}
        sh = [r for r in allr if r["task"] in shared]
        ys = np.array([r["world"] == w for r in sh], int); s2 = np.array([r["score"] for r in sh]); ts = np.array([r["task"] for r in sh])
        obs_s = float(roc_auc_score(ys, s2)) if len(set(ys)) == 2 else None
        perm = []
        for _ in range(10000):
            yp = ys.copy()
            for k in sorted(shared):
                i = np.flatnonzero(ts == k)
                yp[i] = rng.permutation(yp[i])
            perm.append(roc_auc_score(yp, s2) if obs_s is not None else np.nan)
        out[f"real vs {w}"] = {"claims (real / simulated)": [len(real), len(sim)], "AUROC (simulated higher), all claims": float(roc_auc_score(yy, ss)),
                               "shared tasks": len(shared), "AUROC, shared tasks": obs_s,
                               "within-task permutation p (two-sided)": mc_two_sided(perm, obs_s) if obs_s is not None else None,
                               "null mean": float(np.mean(perm)) if obs_s is not None else None,
                               "mean score real / simulated": [float(ss[yy == 0].mean()), float(ss[yy == 1].mean())]}
    return out


def main():
    if hashlib.sha256(RESULT.read_bytes()).hexdigest() != RESULT_SHA:
        raise SystemExit("the committed result differs")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__).resolve())], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    res = json.loads(RESULT.read_text())
    claims = [r for r in res["per report"] if r["class"] == "claims success"]
    real = [r for r in claims if r["world"] == "qwen-real"]
    out = {"label": "POST HOC (reference audit; no verdict)", "git": head, "result sha256": RESULT_SHA,
           "simulated runs built from": "honest_a (Phase 2 launch.json references; reference_audit_2026-10-11_194008.json)"}
    both = real_vs_sim(real, claims)
    for w in SIMS:
        a, b = both[f"real vs {w}"], res[f"real vs {w}"]
        if any(abs(a[k] - b[k]) > 1e-12 for k in ("AUROC, shared tasks", "within-task permutation p (two-sided)", "AUROC (simulated higher), all claims")):
            raise SystemExit(f"'both' doesn't reproduce the committed {w} numbers")
    out["both arms (reproduces the committed v2)"] = both
    out["honest_a only"] = real_vs_sim([r for r in real if r["arm"] == "honest_a"], claims)
    out["honest_b only (honest_a excluded)"] = real_vs_sim([r for r in real if r["arm"] == "honest_b"], claims)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"false_claims_by_arm_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main()
