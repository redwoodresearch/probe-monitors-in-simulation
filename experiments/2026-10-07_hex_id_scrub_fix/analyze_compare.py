"""TF-IDF before and after the underscore fix of the ID rule: per-output and per-run AUROC with 95% intervals.

Reads per-output out-of-fold score files (held-out environment, all outputs): the saved run made with the rule before
the fix, and the rerun with the fixed rule (`--new-stamp`). Also a $0 proxy on the saved scores: the same metrics with
the outputs whose text the fix changes left out (from analyze_mangling_counts.py). Comparisons: real e1 vs simulated,
and the control, real e1 vs real e2 (each scored by its own model). Level 1 is a reproduction check: the ID rule enters
at level 2, so its rerun scores must equal the saved ones.

  uv run python experiments/2026-10-07_hex_id_scrub_fix/analyze_compare.py --counts results/mangling_counts_<stamp>.json [--new-stamp <stamp>]
Writes results/compare_<stamp>.json and .md.
"""

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from probe_shift.provenance import capture
from probe_shift.scoring import auroc_ci

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
POS = {"real_vs_sim": ("real-attack-e1", "sim-attack"), "null_e1_vs_e2": ("real-attack-e1", "real-attack-e2")}


def load(path: Path) -> dict:
    """{comparison: {key|step|action: (arm, main_task, score)}} for all-outputs rows."""
    out = {c: {} for c in POS}
    for r in map(json.loads, gzip.open(path, "rt")):
        if r["outputs"] == "all":
            out[r["comparison"]][f"{r['key']}|{r['step']}|{r['action']}"] = (r["arm"], r["main_task"], r["score"])
    return out


def metrics(rows: dict, comparison: str, drop: set) -> dict:
    neg, pos = POS[comparison]
    items = [(k, *v) for k, v in rows.items() if k not in drop and v[0] in (neg, pos)]
    keys = np.array([k.rsplit("|", 2)[0] for k, *_ in items])
    arm = np.array([a for _, a, _, _ in items])
    task = np.array([t for _, _, t, _ in items])
    score = np.array([s for *_, s in items], float)
    runs, inv = np.unique(keys, return_inverse=True)
    mean = np.bincount(inv, score) / np.bincount(inv)
    first = np.unique(inv, return_index=True)[1]
    return {"per_output": auroc_ci(arm == pos, score, task), "per_run": auroc_ci(arm[first] == pos, mean, task[first])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--old-dir", default="~/probe-shift-baselines/experiments/2026-10-06_q1_text_baselines/results")
    ap.add_argument("--old-stamp", default="2026-10-06_190619")  # held-out environment, every level, rule before the fix
    ap.add_argument("--new-dir", default=str(ROOT / "experiments/2026-10-06_q1_text_baselines/results"))
    ap.add_argument("--new-stamp", default="")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 7])
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    counts = json.loads((HERE / args.counts if not Path(args.counts).is_absolute() else Path(args.counts)).read_text())
    out = {"stamp": stamp, "provenance": capture(ROOT, [Path(__file__), ROOT / "src/probe_shift/scrub.py"]), "args": vars(args), "levels": {}}
    md = [f"# TF-IDF before and after the underscore fix ({stamp}): held-out environment, all outputs", "",
          "| level | comparison | version | per output | per run |", "|---|---|---|---|---|"]
    fmt = lambda m: f"{m['auroc']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]"
    for lv in args.levels:
        old = load(Path(args.old_dir).expanduser() / f"level{lv}_scores_{args.old_stamp}.jsonl.gz")
        new = load(Path(args.new_dir).expanduser() / f"level{lv}_scores_{args.new_stamp}.jsonl.gz") if args.new_stamp else None
        changed = set(counts["changed_rows"].get(f"L{lv}", []))
        res = {"changed_outputs": len(changed)}
        for c in POS:
            res[c] = {"before_fix": metrics(old[c], c, set())}
            if changed:
                res[c]["before_fix_changed_outputs_dropped"] = metrics(old[c], c, changed)
            if new:
                assert set(new[c]) == set(old[c]), f"level {lv} {c}: the rerun scored other outputs"
                res[c]["after_fix"] = metrics(new[c], c, set())
                diff = np.array([abs(new[c][k][2] - old[c][k][2]) for k in old[c]])
                res[c]["max_abs_score_change"] = float(diff.max())
                res[c]["outputs_whose_score_changed"] = int((diff > 1e-4).sum())
            for version, m in res[c].items():
                if isinstance(m, dict) and "per_run" in m:
                    md.append(f"| {lv} | {c} | {version.replace('_', ' ')} | {fmt(m['per_output'])} | {fmt(m['per_run'])} |")
        out["levels"][f"L{lv}"] = res
        if new and lv == 1:  # reproduction: the ID rule does not act at level 1
            for c in POS:
                assert res[c]["max_abs_score_change"] < 1e-4, f"level 1 {c}: rerun differs from the saved run"
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"compare_{stamp}.json").write_text(json.dumps(out, indent=1))
    (HERE / "results" / f"compare_{stamp}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
