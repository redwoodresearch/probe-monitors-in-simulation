"""Check 3 of probe step A (PLAN.md amendment 3), local: the prefix test. In every run, only the outputs before its
first rule-type output (text_editor insert or str_replace, dpkg, or a parseable clock stamp: the only outputs any
rule can change), the same rule in both arms, so no edited text is in the probe's context. Uses the per-output scores
of A's chosen probes (run_claimB_output_scores_modal.py) for fixed sim vs real e1 and, as the control, real e2 vs e1.
Per level and environment subset: run-level AUROC (each run's mean over its prefix outputs), output-level AUROC, and
how many runs and outputs qualify per side. Writes results/claimB_prefix_<stamp>.json.

  uv run python experiments/2026-10-09_fixed_sim/analyze_claimB_prefix.py <fixed scores .json.gz> <e2 scores .json.gz> <local copy of the v2 run>
"""

import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import clock  # noqa: E402
import fixes  # noqa: E402

from probe_shift.scoring import auroc_ci  # noqa: E402

ARMS = {"real-attack-e1": "real-attack-e1", "real-attack-e2": "real-attack-e2", "sim-attack": "sim-attack-fixed"}


def rule_type(tool: str, args: dict, text: str) -> bool:
    if tool == "text_editor":
        return args.get("command") in ("insert", "str_replace")
    return fixes.DPKG_MARKER in text or any(s["kind"] != "FAILED" for s in clock.stamps(text, clock.LS_YEAR))


def first_rule_type(run_dir: Path) -> dict:
    """{(arm, key): (step, action) of the run's first rule-type output, or None}; arms named as in the score files."""
    out = {}
    for f in sorted((run_dir / "inputs").glob("shard_*.jsonl")):
        for r in map(json.loads, f.read_text().splitlines()):
            first = None
            for c in fixes.calls_of(r["messages"]):
                text = r["messages"][c["msg"]]["content"]
                if rule_type(c["tool"], c["args"], text if isinstance(text, str) else ""):
                    first = (c["step"], c["action"])
                    break
            out[(ARMS[r["arm"]], r["key"])] = first
    return out


def metrics(rows, pos_arm):
    y_o = np.array([r["arm"] == pos_arm for r in rows])
    s_o = np.array([r["score"] for r in rows])
    out = {"outputs": {"positive": int(y_o.sum()), "negative": int((~y_o).sum())}}
    if len(set(y_o)) == 2:
        out["output_level"] = auroc_ci(y_o, s_o, np.array([r["main_task"] for r in rows]), 1000, 0)
    by = defaultdict(list)
    for r in rows:
        by[(r["key"], r["arm"], r["main_task"])].append(r["score"])
    keys = sorted(by)
    y = np.array([k[1] == pos_arm for k in keys])
    out["runs"] = {"positive": int(y.sum()), "negative": int((~y).sum())}
    if len(set(y)) == 2:
        out["run_level"] = auroc_ci(y, np.array([np.mean(by[k]) for k in keys]), np.array([k[2] for k in keys]), 1000, 0)
    return out


def main(fixed_file: str, e2_file: str, run_dir: str):
    first = first_rule_type(Path(run_dir))
    test_envs = set(json.loads((HERE / "split.json").read_text())["test"])
    res = {"files": [Path(fixed_file).name, Path(e2_file).name], "levels": {}}
    for name, path, pos in (("fixed_vs_e1", fixed_file, "sim-attack-fixed"), ("e2_vs_e1", e2_file, "real-attack-e2")):
        data = json.loads(gzip.decompress(Path(path).read_bytes()))
        for level, d in data["levels"].items():
            prefix = []
            for r in d["rows"]:
                f = first[(r["arm"], r["key"])]
                if f is None or (r["step"], r["action"]) < f:
                    prefix.append(r)
            for subset, keep in (("test_envs", lambda r: r["environment"] in test_envs), ("all_envs", lambda r: True)):
                res["levels"].setdefault(level, {}).setdefault(subset, {})[name] = {
                    "prefix": metrics([r for r in prefix if keep(r)], pos), "all_outputs": metrics([r for r in d["rows"] if keep(r)], pos)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"claimB_prefix_{stamp}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    for level in sorted(res["levels"], reverse=True):
        for subset, by in res["levels"][level].items():
            for name, m in by.items():
                p = m["prefix"]
                print(f"{level} {subset} {name}: prefix run-level {p.get('run_level', {}).get('auroc')} {p.get('run_level', {}).get('ci95')} "
                      f"runs {p['runs']}; output-level {p.get('output_level', {}).get('auroc')} outputs {p['outputs']} | all outputs run-level "
                      f"{m['all_outputs']['run_level']['auroc']}")
    print(path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
