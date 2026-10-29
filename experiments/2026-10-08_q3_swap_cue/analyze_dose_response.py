"""Quick check: does more fixing mean a bigger shift? $0, local, saved scores only.

Per run, the tool-output score change (both fixes − as simulated, outer model, logits) against the number of outputs
the fixes changed, at L63 and L48, over the 100 swap runs and the 117-run extension. Both are taken over the outputs
both arms kept (the fixed arm's longer transcripts can lose late outputs to the 32k cut): the change is the mean
per-output change over those outputs, and the count is the changed outputs among them (q2's stage reports: rendered
inserts and changed str_replaces, as analyze_locality.py). Spearman ρ with a 95% interval from 2,000 resamples of main
tasks, and the mean change per tercile of the count.

  python analyze_dose_response.py --calls <swap_frozen_calls ... with output_end@48> ... --runs <matching runs files> ... \
      --stage-reports <stage_report_swap_combined_*.json> <stage_report_swap_ext_*.json>
"""

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
RENDERED = {"exact", "approx_window", "approx_numbering"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--stage-reports", nargs="+", required=True)
    args = ap.parse_args()
    changed = {}
    for f in args.stage_reports:
        for k, v in json.loads(Path(f).read_text())["report"].items():
            sr = [x for kk, x in v.items() if kk.startswith("str_replaces")]
            changed[k] = ({(x["step"], x["action"]) for x in v.get("inserts", []) if x["category"] in RENDERED}
                          | {(x["step"], x["action"]) for x in (sr[0] if sr else []) if x["changed"]})
    task_of = {json.loads(line)["key"]: json.loads(line)["main_task"] for f in args.runs for line in Path(f).read_text().splitlines()}
    per = {}  # (readout, key) -> label -> (step, action) -> outer score
    for f in args.calls:
        with gzip.open(f, "rt") as fh:
            for line in fh:
                c = json.loads(line)
                if c["readout"] in ("output_end", "output_end@48") and c["label"] in ("V1", "COMB"):
                    per.setdefault((c["readout"], c["key"]), {}).setdefault(c["label"], {})[(c["step"], c["action"])] = c["score"]
    rng = np.random.default_rng(0)
    out = {"inputs": {"calls": [Path(f).name for f in args.calls], "stage_reports": [Path(f).name for f in args.stage_reports]}, "readouts": {}}
    for ro in ("output_end", "output_end@48"):
        keys = sorted(k for (r_, k), a in per.items() if r_ == ro and set(a) == {"V1", "COMB"} and k in changed)
        common = {k: sorted(set(per[(ro, k)]["V1"]) & set(per[(ro, k)]["COMB"])) for k in keys}
        x = np.array([len(changed[k] & set(common[k])) for k in keys], float)
        d = np.array([np.mean([per[(ro, k)]["COMB"][sa] - per[(ro, k)]["V1"][sa] for sa in common[k]]) for k in keys])
        task = np.array([task_of[k] for k in keys])
        tasks = np.unique(task)
        tix = np.searchsorted(tasks, task)
        boots = []
        for _ in range(2000):
            idx = np.concatenate([np.flatnonzero(tix == t) for t in rng.integers(0, len(tasks), len(tasks))])
            boots.append(spearmanr(x[idx], d[idx]).statistic)
        cut = np.quantile(x, [1 / 3, 2 / 3])
        bins = np.digitize(x, cut, right=True)  # 0: <= first cut, 1: <= second, 2: above
        out["readouts"][ro] = {"runs": len(keys), "changed_outputs_outside_the_common_set": int(sum(len(changed[k] - set(common[k])) for k in keys)),
                               "spearman_rho": float(spearmanr(x, d).statistic),
                               "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                               "mean_change_logits": float(d.mean()), "n_changed_median": float(np.median(x)),
                               "terciles": [{"n_changed": [float(x[bins == b].min()), float(x[bins == b].max())], "runs": int((bins == b).sum()),
                                             "mean_change_logits": float(d[bins == b].mean())} for b in range(3)]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"dose_response_{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))
    for ro, r in out["readouts"].items():
        print(ro, f"rho {r['spearman_rho']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]", [round(t["mean_change_logits"], 2) for t in r["terciles"]])
    print("->", path)


if __name__ == "__main__":
    main()
