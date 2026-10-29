"""K3: does the frozen real-vs-sim probe move on simulated outputs that only change presentation, or only where the
facts change? Matched-step pairs (2026-10-06_matched_step_pairs; real history fixed, step k's real vs simulated
output), blind fact labels (matched_pair_fact_labels_2026-10-07.jsonl), frozen Q1-recipe probe scores at the output
(output_end, layer 61) and at the next call (call_end, layer 2), the layers that experiment chose by cohort CV.

Per group (identical, presentation, changed facts): the share of pairs where the simulated version scores higher
(ties count half; 0.5 = no shift) and the mean score shift (sim - real), with 2,000 pair resamples.

  python analyze_k3_fact_groups.py --pairs-dir <matched_step_pairs experiment> --labels <fact labels>
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
READS = {"output_end": "61", "call_end": "2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()
    pd_ = Path(args.pairs_dir)
    sc = json.loads((pd_ / "data/probe_scores/probe_scores_2026-10-06_191958.json").read_text())
    items = [tuple(i) for i in sc["frozen"]["items"]]
    idx = {it: j for j, it in enumerate(items)}
    lab = {x["pair"]: x["label"] for x in map(json.loads, Path(args.labels).read_text().splitlines())}
    pairs = sorted({p for p, _ in items})
    group = {p: lab.get(p, "identical") for p in pairs}
    rng = np.random.default_rng(0)
    out = {"inputs": {"probe_scores": "data/probe_scores/probe_scores_2026-10-06_191958.json",
                      "labels": {"file": args.labels, "sha256": hashlib.sha256(Path(args.labels).read_bytes()).hexdigest()}},
           "results": {}}
    for ro, layer in READS.items():
        s = sc["frozen"]["layers"][layer][ro]["scores"]
        for g in ("identical", "presentation", "changed_facts"):
            ps = [p for p in pairs if group[p] == g and s[idx[(p, "real")]] is not None and s[idx[(p, "sim")]] is not None]
            d = np.array([s[idx[(p, "sim")]] - s[idx[(p, "real")]] for p in ps])
            win = np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))
            bw = [rng.integers(0, len(d), len(d)) for _ in range(2000)]
            ci = lambda x: [round(float(v), 4) for v in np.percentile([x[b].mean() for b in bw], [2.5, 97.5])]
            out["results"][f"{ro}|{g}"] = {"pairs": len(d), "sim_higher_share": round(float(win.mean()), 4), "sim_higher_ci95": ci(win),
                                          "mean_shift": round(float(d.mean()), 4), "mean_shift_ci95": ci(d),
                                          "mean_abs_shift": round(float(np.abs(d).mean()), 4)}
            r = out["results"][f"{ro}|{g}"]
            print(f"{ro} L{layer} {g:14s} n={len(d):3d}: sim higher {r['sim_higher_share']:.3f} {r['sim_higher_ci95']} | shift {r['mean_shift']:+.3f} {r['mean_shift_ci95']} | |shift| {r['mean_abs_shift']:.3f}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"k3_fact_groups_{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
