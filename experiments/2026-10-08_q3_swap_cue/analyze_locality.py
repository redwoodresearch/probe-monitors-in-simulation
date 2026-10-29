"""Quick check: is the formatting effect local to the edited outputs, or carried over to later ones? $0, local.

Per tool output, the both fixes − as simulated change in its outer score (per-call files of the layer-48 scoring, which
hold L63 and L48), on outputs both arms kept. Each output is one of:
- changed: an insert the fix re-rendered (category exact / approx_window / approx_numbering) or a str_replace whose
  text changed (q2's stage reports; step = assistant turn, action = call index, as the extraction's positions);
- after: an unchanged output after the run's first changed output;
- before: an unchanged output before it (must be 0: the prefix check).
Reports the mean change per output in each class, and each class's share of the common-output score change (the run
mean over outputs both arms kept), both with intervals from 2,000 resamples of main tasks. As-simulated outputs lost to
the 32k cut in the fixed arm are outside this decomposition; their count and the gap to the saved run change are reported.

  python analyze_locality.py --calls <swap_frozen_calls ... with output_end@48> ... --runs <matching runs files> ... \
      --stage-reports <stage_report_swap_combined_*.json> <stage_report_swap_ext_*.json>
"""

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

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
    task, first = {}, {}
    for f in args.runs:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            if r["label"] == "COMB":
                task[r["key"]] = r["main_task"]
                first[r["key"]] = tuple(r["first_changed_output"]) if r["first_changed_output"] else None
    per = {}
    for f in args.calls:
        with gzip.open(f, "rt") as fh:
            for line in fh:
                c = json.loads(line)
                if c["readout"] in ("output_end", "output_end@48") and c["label"] in ("V1", "COMB"):
                    per.setdefault((c["readout"], c["key"]), {}).setdefault(c["label"], {})[(c["step"], c["action"])] = c["score"]
    rng = np.random.default_rng(0)
    out = {"inputs": {"calls": [Path(f).name for f in args.calls], "stage_reports": [Path(f).name for f in args.stage_reports]}, "readouts": {}}
    for ro in ("output_end", "output_end@48"):
        rows = []  # (task, class, delta, n outputs of the run)
        for (r_, k), arms in per.items():
            if r_ != ro or set(arms) != {"V1", "COMB"} or k not in changed:
                continue
            common = sorted(set(arms["V1"]) & set(arms["COMB"]))
            fc = first[k]
            for sa in common:
                cls = "changed" if sa in changed[k] else ("before" if fc is None or sa < fc else "after")
                rows.append((task[k], k, cls, arms["COMB"][sa] - arms["V1"][sa], len(common)))
        tk = np.array([r[0] for r in rows]); key = np.array([r[1] for r in rows]); cls = np.array([r[2] for r in rows])
        dl = np.array([r[3] for r in rows]); nrun = np.array([r[4] for r in rows], float)
        tasks = np.unique(tk); tix = np.searchsorted(tasks, tk)
        draws = [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))[tix] for _ in range(2000)]
        res = {"outputs": {c: int((cls == c).sum()) for c in ("changed", "after", "before")}}
        for c in ("changed", "after", "before"):
            m = cls == c
            b = [(dl[m] * w[m]).sum() / w[m].sum() for w in draws]
            res[f"mean_change_per_output ({c})"] = {"logits": float(dl[m].mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        contrib = dl / nrun  # each output's contribution to its run's mean-score change
        total = [(contrib * w).sum() for w in draws]
        for c in ("changed", "after"):
            m = cls == c
            sh = [(contrib[m] * w[m]).sum() / t_ for w, t_ in zip(draws, total)]
            res[f"share_of_common_output_change ({c})"] = {"share": float(contrib[m].sum() / contrib.sum()), "ci95": [float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))]}
        res["runs"] = int(len(np.unique(key)))
        n_v1_only = sum(len(set(a_["V1"]) - set(a_["COMB"])) for (r_, k), a_ in per.items() if r_ == ro and set(a_) == {"V1", "COMB"} and k in changed)
        res["as_simulated_only_outputs (outside the decomposition)"] = int(n_v1_only)
        res["common_output_mean_run_change_logits"] = float(np.mean([np.mean([r[3] for r in rows if r[1] == k]) for k in np.unique(key)]))
        out["readouts"][ro] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"locality_{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))
    print(json.dumps(out["readouts"], indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
