"""Check 2 of PLAN.md amendment 3, part 2 (local; scores and run keys, no activations): do the fix rules leave traces
the probe reads? Per-output scores of A's selected probes (run_claimB_output_scores_modal.py) joined with each output's
type from the fixed run's level-0 text and the rules' records:
  rule-type: text_editor insert and str_replace outputs, dpkg outputs (`Preparing to unpack`), and outputs holding a
             parseable clock stamp (clock.py's formats); on the simulated side also whether a rule changed the output;
  other:     everything else (bash without dpkg or stamps, view, create, ...).
Reports per level: output-level AUROC (sim over real) within each type; run-level AUROC with each run scored on its
other outputs only, and on its rule-type outputs only (95% intervals from main-task resamples); the simulated split by
changed / unchanged; the 10 highest-scoring simulated and 10 lowest-scoring real outputs (run, step, action, type).
Writes results/claimB_output_types_<stamp>.json.

  uv run python experiments/2026-10-09_fixed_sim/analyze_claimB_output_types.py <scores .json.gz> <local copy of the v2 run>
"""

import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import clock  # noqa: E402
import fixes  # noqa: E402

from probe_shift.scoring import auroc_ci  # noqa: E402


def output_types(run_dir: Path) -> dict:
    """{(key, step, action): (tool, type, changed)} for the real e1 and simulated runs, from the level-0 text."""
    changed = set()
    for f in sorted((run_dir / "records").glob("records_*.jsonl.gz")):
        for x in map(json.loads, gzip.decompress(f.read_bytes()).decode().splitlines()):
            if x["changed"]:
                changed.add((x["key"], x["msg"]))
    out = {}
    for f in sorted((run_dir / "inputs").glob("shard_*.jsonl")):
        for r in map(json.loads, f.read_text().splitlines()):
            if r["arm"] not in ("real-attack-e1", "sim-attack"):
                continue
            for c in fixes.calls_of(r["messages"]):
                text = r["messages"][c["msg"]]["content"]
                text = text if isinstance(text, str) else ""
                cmd = c["args"].get("command") if c["tool"] == "text_editor" else None
                if cmd in ("insert", "str_replace"):
                    kind = f"editor_{cmd}"
                elif c["tool"] != "text_editor" and fixes.DPKG_MARKER in text:
                    kind = "dpkg"
                elif any(s["kind"] != "FAILED" for s in clock.stamps(text, clock.LS_YEAR)):
                    kind = "clock_stamp"
                else:
                    kind = "other"
                out[(r["key"], c["step"], c["action"])] = (c["tool"], kind, (r["key"], c["msg"]) in changed)
    return out


def run_auc(rows, keep):
    """Run-level AUROC (sim over real) with each run's score the mean of its kept outputs; runs without one dropped."""
    by = defaultdict(list)
    for r in rows:
        if keep(r):
            by[(r["key"], r["arm"], r["main_task"])].append(r["score"])
    keys = sorted(by)
    y = np.array([k[1] == "sim-attack-fixed" for k in keys])
    s = np.array([np.mean(by[k]) for k in keys])
    return auroc_ci(y, s, np.array([k[2] for k in keys]), 1000, 0) | {"runs": {"sim": int(y.sum()), "real": int((~y).sum())}}


def out_auc(rows, keep):
    sel = [r for r in rows if keep(r)]
    y = np.array([r["arm"] == "sim-attack-fixed" for r in sel])
    if len(set(y)) < 2:
        return {"outputs": len(sel), "note": "one arm only"}
    return auroc_ci(y, np.array([r["score"] for r in sel]), np.array([r["main_task"] for r in sel]), 1000, 0) | {
        "outputs": {"sim": int(y.sum()), "real": int((~y).sum())}}


def main(scores_file: str, run_dir: str):
    data = json.loads(gzip.decompress(Path(scores_file).read_bytes()))
    types = output_types(Path(run_dir))
    res = {"scores_file": Path(scores_file).name, "code": data["code"], "levels": {}}
    for level, d in data["levels"].items():
        rows, miss, tool_mismatch = [], 0, 0
        for r in d["rows"]:
            t = types.get((r["key"], r["step"], r["action"]))
            if t is None:
                miss += 1
                continue
            tool_mismatch += t[0] != r["tool"]
            rows.append(r | {"type": t[1], "changed": t[2], "rule_type": t[1] != "other"})
        assert miss == 0 and tool_mismatch == 0, (miss, tool_mismatch)
        sim = lambda r: r["arm"] == "sim-attack-fixed"
        res["levels"][level] = {}
        test_envs = set(json.loads((HERE / "split.json").read_text())["test"])
        for subset, rows_s in (("all_envs", rows), ("test_envs", [r for r in rows if r["environment"] in test_envs])):
            res["levels"][level][subset] = summarize(rows_s, sim)
    write(res)


def summarize(rows, sim):
    if True:
        out = {"outputs_by_type": {f"{a}|{k}": v for (a, k), v in sorted(Counter((r["arm"], r["type"]) for r in rows).items())},
               "run_level": {"all_outputs": run_auc(rows, lambda r: True),
                             "other_outputs_only": run_auc(rows, lambda r: not r["rule_type"]),
                             "rule_type_outputs_only": run_auc(rows, lambda r: r["rule_type"])},
               "output_level": {"all": out_auc(rows, lambda r: True),
                                "other": out_auc(rows, lambda r: not r["rule_type"]),
                                "rule_type": out_auc(rows, lambda r: r["rule_type"]),
                                **{f"type={k}": out_auc(rows, lambda r, k=k: r["type"] == k) for k in ("editor_insert", "editor_str_replace", "dpkg", "clock_stamp", "other")},
                                "rule_type: sim changed by a rule vs real": out_auc(rows, lambda r: r["rule_type"] and (not sim(r) or r["changed"])),
                                "rule_type: sim not changed vs real": out_auc(rows, lambda r: r["rule_type"] and (not sim(r) or not r["changed"]))}}
        ranked = sorted(rows, key=lambda r: r["score"])
        pick = lambda r: {k: r[k] for k in ("key", "step", "action", "tool", "type", "changed", "environment")} | {"score": round(r["score"], 3)}
        out["top10_sim"] = [pick(r) for r in reversed([r for r in ranked if sim(r)][-10:])]
        out["bottom10_real"] = [pick(r) for r in [r for r in ranked if not sim(r)][:10]]
        return out


def write(res):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"claimB_output_types_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(res, indent=1))
    for level, subs in res["levels"].items():
      for subset, out in subs.items():
        print(f"== {level} {subset}")
        for k, v in out["run_level"].items():
            print(f"  run level, {k:24s} {v['auroc']:.3f} {v['ci95']}  runs {v['runs']}")
        for k, v in out["output_level"].items():
            print(f"  output level, {k:44s} " + (f"{v['auroc']:.3f} {v['ci95']}  {v['outputs']}" if "auroc" in v else str(v)))
    print(path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
