"""The Llama-3.3-70B honest check's 40 runs in one table, by key, against PREDICTIONS.md and the Qwen3-32B check.
Read-only, $0.

  uv run --no-project --python 3.14 python combine_40.py --reports <report_*.json from report_qwen3_32b_native.py,
      one per launch> --runs <the matching run dir copies, same order>

Rows come from report_qwen3_32b_native.py (the Qwen check's own report: steps, ending, main-task score, tool errors by
PR #67's rule, which skips Inspect "parsing" errors, cut replies). Added per run from its .eval: tool messages whose
Inspect error type is "parsing", for the second count (PR #67's rule plus parsing errors); assistant messages with
more than one tool call. Fails unless the runs are exactly the 40 (task, arm) pairs of the design, each once.
Writes results/combined_40_<stamp>.json.
"""

import argparse
import json
import statistics
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
TASKS = HERE.parents[1] / "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9.json"


def sample(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
        return json.loads(z.read(n))


def eval_extras(run_dir: Path, arm: str, task: str) -> dict:
    (ev,) = sorted((run_dir / arm / task).glob("*.eval"))
    s = sample(ev)
    calls = {t["id"]: t["function"] for m in s["messages"] if m["role"] == "assistant" for t in m.get("tool_calls") or []}
    tools = [m for m in s["messages"] if m["role"] == "tool"]
    return {"parsing_errors": sum(bool(m.get("error")) and m["error"].get("type") == "parsing"
                                  and calls.get(m.get("tool_call_id")) != "submit" for m in tools),
            "multi_tool_call_messages": sum(len(m.get("tool_calls") or []) > 1 for m in s["messages"] if m["role"] == "assistant"),
            "messages": len(s["messages"]), "eval": ev.name}


def main(reports: list[str], runs: list[str]):
    assert len(reports) == len(runs), "one run dir per report"
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    rows = []
    for rep, run in zip(reports, runs):
        for r in json.loads(Path(rep).read_text())["rows"]:
            assert "steps" in r, f"{rep}: {r['arm']}/{r['task']} was not read: {r.get('ending')}"
            rows.append(r | eval_extras(Path(run), r["arm"], r["task"]) | {"launch": Path(run).name, "report": Path(rep).name})
    keys = Counter((r["task"], r["arm"]) for r in rows)
    spec = json.loads(TASKS.read_text())
    tasks = [t["key"] for t in spec["cells"]]  # the 20 tasks (pilot_tasks_amendment9.json's cells)
    want = {(t, a) for t in tasks for a in ("honest_a", "honest_b")}
    assert len(want) == 40 and set(keys) == want and set(keys.values()) == {1}, (sorted(want - set(keys)), [k for k, v in keys.items() if v > 1])
    calls = sum(r["calls"] for r in rows)
    err_a = sum(r["tool_errors"] or 0 for r in rows)
    err_b = err_a + sum(r["parsing_errors"] for r in rows)
    out = {"git": git, "n_runs": len(rows),
           "endings": dict(Counter(r["ending"] for r in rows)),
           "main_task_raw": dict(Counter(str(r["main_task_raw"]) for r in rows)),
           "success_counted": sum(bool(r["success_counted"]) for r in rows),
           "median_steps": statistics.median(r["steps"] for r in rows),
           "steps_min_max": [min(r["steps"] for r in rows), max(r["steps"] for r in rows)],
           "out_of_context": sum(r["ending"] == "out of context" for r in rows),
           "step_limit": sum("step" in str(r["ending"]) and "limit" in str(r["ending"]) for r in rows),
           "runs_with_cut_reply": sum(bool(r["cap_hits"]) for r in rows),
           "tool_calls": calls, "tool_errors_pr67_rule": err_a, "tool_errors_pr67_plus_parsing": err_b,
           "tool_error_share_pr67_rule": round(err_a / calls, 4) if calls else None,
           "tool_error_share_plus_parsing": round(err_b / calls, 4) if calls else None,
           "str_replace": sum(r["str_replace"] for r in rows), "str_replace_failed": sum(r["str_replace_failed"] for r in rows),
           "multi_tool_call_messages": sum(r["multi_tool_call_messages"] for r in rows),
           "rows": sorted(rows, key=lambda r: (r["task"], r["arm"]))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"combined_40_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in out["rows"]:
        print(f"  {r['task']:22s} {r['arm']:8s} steps {r['steps']:3d} | {r['ending']:16s} | {r['main_task_raw']} | "
              f"errors {r['tool_errors']} (+{r['parsing_errors']} parsing) of {r['calls']} | {r['launch']}")
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    main(**vars(ap.parse_args()))
