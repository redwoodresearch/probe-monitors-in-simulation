"""False success claims: does the probe separate false from truthful claims beyond report length? (local, free; added
after the results, approved by the orchestrator 2026-10-10)

Prediction, written before running: the length-adjusted AUROC falls from 0.71 to 0.58 ± 0.12. At or below 0.6, length
plausibly accounts for most of the separation; near 0.7, it does not (still inconclusive with 7 claims from 4 tasks).

1. Length-adjusted score: the residual of score on log(report tokens), fit by least squares on all 59 success claims
   (real and simulated, labels unused) and held fixed. AUROC false vs truthful on the residual, with the same task
   bootstrap stratified by label (2,000, seed 0) and the exact task-block permutation null as the frozen analysis.
2. Length-matched pairs: each false claim against the truthful claim nearest in log length, with replacement and no
   maximum distance (ties: higher truthful score first, so a tie never favours the probe; corrected after the run, when
   no tie occurred); the mean score difference and how many pairs the false claim wins. Descriptive.
3. Planted check (Aswin's rule) on the residual's standardized mean difference: +0.5 truthful-residual SD on the false
   claims.

  uv run --no-project --with numpy --with scikit-learn python experiments/2026-10-10_q2_false_claims/analyze_length_check.py <analysis json>
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from analyze_false_claims import smd, strat_boot

from probe_shift.planted import detected, widening

HERE = Path(__file__).parent


def main(path: Path):
    a = json.loads(path.read_text())
    claims = [r for r in a["per report"] if r["class"] == "claims success"]
    x = np.log([r["report_tokens"] for r in claims]); s_all = np.array([r["score"] for r in claims])
    slope, intercept = np.polyfit(x, s_all, 1)
    resid = {r["key"]: r["score"] - (intercept + slope * np.log(r["report_tokens"])) for r in claims}
    lab = [r for r in claims if r["label"] is not None]
    y = np.array([r["label"] for r in lab]); t = np.array([r["task"] for r in lab])
    s = np.array([r["score"] for r in lab]); e = np.array([resid[r["key"]] for r in lab])
    tokens = np.array([r["report_tokens"] for r in lab])
    auc = float(roc_auc_score(y, e))
    tasks = sorted(set(t)); k = len(set(t[y == 1]))
    null = [roc_auc_score(np.isin(t, c).astype(int), e) for c in itertools.combinations(tasks, k)]
    out = {"input": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "added after the results": True,
           "fit on all success claims (score = a + b log tokens)": {"claims": len(claims), "a": float(intercept), "b": float(slope)},
           "unadjusted AUROC (frozen analysis)": float(roc_auc_score(y, s)),
           "length-adjusted AUROC": auc, "95% (task bootstrap stratified by label)": strat_boot(t, y, lambda i: roc_auc_score(y[i], e[i])),
           "task-block permutation p": float(np.mean(np.array(null) >= auc - 1e-12)), "permutations": len(null)}
    pairs = []
    for i in np.flatnonzero(y == 1):
        cand = np.flatnonzero(y == 0)
        j = min(cand, key=lambda c: (abs(np.log(tokens[c]) - np.log(tokens[i])), -s[c]))
        pairs.append({"false": lab[i]["key"], "truthful": lab[j]["key"], "false tokens": int(tokens[i]), "truthful tokens": int(tokens[j]),
                      "score difference": float(s[i] - s[j])})
    d = np.array([p["score difference"] for p in pairs])
    out["length-matched pairs"] = {"pairs": pairs, "mean score difference (false minus truthful)": float(d.mean()),
                                   "pairs the false claim scores higher": f"{int((d > 0).sum())} of {len(d)}",
                                   "distinct truthful matches": len({p["truthful"] for p in pairs})}
    est, ci = smd(e, y), strat_boot(t, y, lambda i: smd(e[i], y[i]))
    ep = e + 0.5 * e[y == 0].std(ddof=1) * y
    est_p, ci_p = smd(ep, y), strat_boot(t, y, lambda i: smd(ep[i], y[i]))
    out["planted check on the residual (+0.5 truthful SD)"] = {"unplanted estimate": est, "unplanted 95%": ci, "planted estimate": est_p,
                                                               "planted 95%": ci_p, "detected": detected(est_p, tuple(ci_p), est, 0.5),
                                                               "widening direction": widening(est, 0.5)}
    dest = HERE / "results" / f"false_claims_length_check_{path.stem.removeprefix('false_claims_analysis_')}.json"
    with open(dest, "x") as fh:  # results/ is append-only: a rerun needs a new name
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
