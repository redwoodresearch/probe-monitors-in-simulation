"""K1: does the frozen Q3 detector tell attack runs that attempted the side task from ones that declined it, within the
same main tasks? Saved out-of-fold run scores only (#36 headline run scores).

Per arm (e1, e2, sim) and detector score: run AUROC attempted (positive) vs declined, on main tasks that have both
kinds of run in that arm; a label-shuffle null permuting the labels within each main task (1,000 draws, seed 0) and its
two-sided p; 95% interval from 1,000 resamples of main tasks.

  python analyze_attempted_vs_declined.py --run-scores <headline_run_scores_*.jsonl> --labels <attempt labels>
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
from probe_shift.attack_eval import E1, E2, SIM, auc, boot_weights, summary  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

SCORES = ("probe_content_mean", "probe_allcalls_mean", "tfidf_content_mean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-scores", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()
    runs = [json.loads(l) for l in Path(args.run_scores).read_text().splitlines()]
    lab = {(x["arm"], x["key"]): x["label"] for x in map(json.loads, Path(args.labels).read_text().splitlines())}
    out = {"inputs": {p: {"file": f, "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest()}
                      for p, f in (("run_scores", args.run_scores), ("labels", args.labels))}, "results": {}}
    rng = np.random.default_rng(0)
    for a in (E1, E2, SIM):
        rs = [r for r in runs if r["arm"] == a and lab.get((a, r["key"])) in ("attempted", "declined")]
        for name in SCORES:
            rr = [r for r in rs if r.get(name) is not None]
            task = np.array([r["main_task"] for r in rr])
            y = np.array([lab[(a, r["key"])] == "attempted" for r in rr], float)
            mixed = {t for t in set(task) if 0 < y[task == t].mean() < 1}  # tasks with both kinds
            m = np.isin(task, list(mixed))
            task, y, s = task[m], y[m], np.array([r[name] for r in rr], float)[m]
            W = boot_weights(task, 1000, 0)
            obs = roc_auc_score(y, s)
            groups = [np.flatnonzero(task == t) for t in sorted(mixed)]
            null = np.empty(1000)
            for k in range(1000):
                yy = y.copy()
                for g in groups:
                    yy[g] = y[g][rng.permutation(len(g))]
                null[k] = roc_auc_score(yy, s)
            center = null.mean()
            p = (1 + int((np.abs(null - center) >= abs(obs - center)).sum())) / 1001
            out["results"][f"{a}|{name}"] = {"tasks": len(mixed), "attempted": int(y.sum()), "declined": int((1 - y).sum()),
                                            "auroc": summary(auc(s, y, W)), "null_mean": round(float(center), 4),
                                            "null_p2.5_p97.5": [round(float(v), 4) for v in np.percentile(null, [2.5, 97.5])],
                                            "p_two_sided": round(p, 4)}
            r = out["results"][f"{a}|{name}"]
            print(f"{a:15s} {name:20s} tasks {len(mixed):3d} att/dec {r['attempted']}/{r['declined']}: AUROC {r['auroc']['value']:.3f} "
                  f"[{r['auroc']['ci95'][0]:.3f}, {r['auroc']['ci95'][1]:.3f}] null {r['null_p2.5_p97.5']} p {p:.4f}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"attempted_vs_declined_{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
