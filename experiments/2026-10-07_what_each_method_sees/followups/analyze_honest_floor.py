"""Why does TF-IDF separate a task's two real honest runs at level 7 (no editor outputs, held-out environments)?

Reads baselines-q3's saved out-of-fold scores (`level7_scores_<stamp>`, comparison null_e1_vs_e2 = first real honest
run by key (0) vs second (1)) and the honest staging. No fits. Reports the run AUROC with a 95% main-task bootstrap:
- overall, and split by the pair's batch pattern: the first run by key can come from the base200 batch (keys sort
  `base200_…` < `overlap800_…`) while the second is always overlap800, so a pair is base200/overlap800 or
  overlap800/overlap800;
- per environment;
and the 10 outputs the model is surest of (5 each way, correct side, one per run), as level-7 text.

  uv run python experiments/2026-10-07_what_each_method_sees/followups/analyze_honest_floor.py
"""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from probe_shift.provenance import capture
from probe_shift.scoring import auroc_ci
from probe_shift.scrub import scrub

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
R1, R2 = "real-honest", "real-honest-2"


def run_auc(rows, keep_task=lambda t: True):
    by_run = defaultdict(list)
    meta = {}
    for r in rows:
        if keep_task(r["main_task"]):
            by_run[r["key"]].append(r["score"])
            meta[r["key"]] = (r["arm"], r["main_task"])
    keys = sorted(by_run)
    y = np.array([meta[k][0] == R2 for k in keys])
    s = np.array([np.mean(by_run[k]) for k in keys])
    return auroc_ci(y, s, np.array([meta[k][1] for k in keys])) if len(set(y)) == 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="~/probe-shift-baselines/experiments/2026-10-06_q1_text_baselines")
    ap.add_argument("--stamp", default="2026-10-06_213857")  # honest design, held-out environments, every level
    ap.add_argument("--staging", default="examples_2026-10-06_213745")
    ap.add_argument("--outputs", default="no_editor")
    ap.add_argument("--n-examples", type=int, default=5)
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    base = Path(args.baselines).expanduser()
    rows = [r for r in map(json.loads, gzip.open(base / f"results/level7_scores_{args.stamp}.jsonl.gz", "rt"))
            if r["comparison"] == "null_e1_vs_e2" and r["outputs"] == args.outputs]
    runs = [json.loads(l) for l in (base / ".run_cache" / args.staging / "runs.jsonl").read_text().splitlines()]
    pair = defaultdict(dict)
    for r in runs:
        if r["arm"] in (R1, R2):
            pair[r["main_task"]][r["arm"]] = r["key"].split("_")[0]
    pattern = {t: f"{d[R1]}/{d[R2]}" for t, d in pair.items() if len(d) == 2}
    env = {}
    text = {}
    for r in map(json.loads, gzip.open(base / ".run_cache" / args.staging / "outputs.jsonl.gz", "rt")):
        if r["arm"] in (R1, R2) and r["kept"]:
            env[r["main_task"]] = r["environment"]
            text[(r["key"], r["step"], r["action"])] = r["readout_text"]
    out = {"stamp": stamp, "provenance": capture(ROOT, [Path(__file__)]), "args": vars(args), "outputs": len(rows),
           "pairs_by_pattern": dict(Counter(pattern.values())), "run_auroc": {"all pairs": run_auc(rows)}}
    for p in sorted(set(pattern.values())):
        out["run_auroc"][f"pairs {p}"] = run_auc(rows, lambda t, p=p: pattern.get(t) == p)
    out["run_auroc_by_environment"] = {e: run_auc(rows, lambda t, e=e: env.get(t) == e) for e in sorted(set(env.values()))}
    # The outputs the model is surest of, on the correct side, one per run.
    ex = []
    for side, arm, sign in (("second run", R2, -1), ("first run", R1, 1)):
        seen = set()
        for r in sorted((r for r in rows if r["arm"] == arm), key=lambda r: sign * r["score"]):
            if r["key"] in seen:
                continue
            seen.add(r["key"])
            ex.append({"side": side, "key": r["key"], "step": r["step"], "action": r["action"], "tool": r["tool"],
                       "environment": env[r["main_task"]], "pattern": pattern.get(r["main_task"]), "score": r["score"],
                       "level7_text": scrub(text[(r["key"], r["step"], r["action"])], 7)[:700]})
            if len(seen) == args.n_examples:
                break
    out["examples"] = ex
    fmt = lambda m: "n/a" if m is None else f"{m['auroc']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] (n {m['n']})"
    md = [f"# Honest real-vs-real floor, level 7, {args.outputs}, held-out environments ({stamp})", "",
          f"Pairs by batch pattern (first run / second run): {out['pairs_by_pattern']}", "", "| runs | run AUROC, second vs first |", "|---|---|"]
    md += [f"| {k} | {fmt(v)} |" for k, v in out["run_auroc"].items()]
    md += ["", "| environment | run AUROC |", "|---|---|"] + [f"| {k} | {fmt(v)} |" for k, v in out["run_auroc_by_environment"].items()]
    md += ["", "## The outputs the model is surest of", ""]
    for e in ex:
        md += [f"**{e['side']}** ({e['pattern']}), {e['environment']}, {e['tool']}, step {e['step']}, log-odds {e['score']:+.2f}, `{e['key'][:60]}`",
               "", "```", e["level7_text"].replace("```", "'''"), "```", ""]
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"honest_floor_{stamp}.json").write_text(json.dumps(out, indent=1))
    (HERE / "results" / f"honest_floor_{stamp}.md").write_text("\n".join(md))
    print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
