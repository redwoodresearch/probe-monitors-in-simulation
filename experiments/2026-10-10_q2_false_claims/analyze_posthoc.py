"""False success claims: two checks added after the plan, once the frozen analysis was read (local, free).

1. Length: false claims are shorter (length-baseline AUROC 0.27). Does the probe score track report length? Spearman
   correlation of score and report tokens, among labelled claims and among all claims.
2. Planted check for real vs simulated (Aswin's rule, probe_shift.planted): the estimate is the mean score of simulated
   minus real success claims, in units of the real claims' SD, with a task bootstrap (2,000, seed 0). Plant +0.5 SD on
   the simulated claims' scores: is it detected?

  uv run --no-project --with numpy --with scipy python experiments/2026-10-10_q2_false_claims/analyze_posthoc.py <analysis json>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from probe_shift.planted import detected, widening

HERE = Path(__file__).parent


def main(path: Path):
    a = json.loads(path.read_text())
    rows = [r for r in a["per report"] if r["class"] == "claims success"]
    lab = [r for r in rows if r["label"] is not None]
    out = {"input": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "added after the plan": True}
    for name, rs in (("labelled claims", lab), ("all success claims", rows)):
        rho, p = spearmanr([r["score"] for r in rs], [r["report_tokens"] for r in rs])
        out[f"Spearman score vs report tokens, {name}"] = {"n": len(rs), "rho": float(rho), "p": float(p)}
    real = [r for r in rows if r["world"] == "qwen-real"]
    sd = float(np.std([r["score"] for r in real], ddof=1))
    rng_seed = 0
    for w in ("qwen-sim@september", "qwen-sim@today"):
        sim = [r for r in rows if r["world"] == w]
        allr = real + sim
        tasks = np.array([r["task"] for r in allr]); is_sim = np.array([r["world"] == w for r in allr])
        s = np.array([r["score"] for r in allr])
        names = sorted(set(tasks)); rows_of = {t: np.flatnonzero(tasks == t) for t in names}
        est = lambda sc, i, m=is_sim: float((sc[i][m[i]].mean() - sc[i][~m[i]].mean()) / sd)
        rng = np.random.default_rng(rng_seed)
        draws = [np.concatenate([rows_of[t] for t in rng.choice(names, len(names))]) for _ in range(2000)]
        draws = [i for i in draws if is_sim[i].any() and (~is_sim[i]).any()]
        sp = s + 0.5 * sd * is_sim
        full = np.arange(len(s))
        u, p_ = est(s, full), est(sp, full)
        ci_u = np.percentile([est(s, i) for i in draws], [2.5, 97.5]).tolist()
        ci_p = np.percentile([est(sp, i) for i in draws], [2.5, 97.5]).tolist()
        out[f"planted check, real vs {w} (+0.5 real-claim SD on simulated)"] = {
            "unplanted estimate": u, "unplanted 95%": ci_u, "planted estimate": p_, "planted 95%": ci_p,
            "detected": detected(p_, tuple(ci_p), u, 0.5), "widening direction": widening(u, 0.5), "bootstrap draws used": len(draws)}
    dest = HERE / "results" / f"false_claims_posthoc_{path.stem.removeprefix('false_claims_analysis_')}.json"
    with open(dest, "x") as fh:  # results/ is append-only: a rerun needs a new name
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
