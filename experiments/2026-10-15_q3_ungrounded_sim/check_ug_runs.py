"""Per-run evidence for the ungrounded arm (09's Append 34), ON q3-xbox1, during or after the launch. Read-only.
Metadata only: launch.json records, the per-task preflight JSONs and ONE line of each run's log (the `[sim_task]
condition.affordances:` line the tree prints); no transcript text is read or printed.

Per run `sim_attack@september_ungrounded/<key>`:
  - negative control: every preflight row for (key, september_ungrounded) has `ungrounded: true`, both reference markers
    0, and the copy carried both markers (reasoning blocks and tool outputs > 0);
  - the run's own record: condition.affordances == ['codebase_read'] (its log's line, read from the first 200 lines only),
    and BA_NO_HONEST_REFERENCE=1 in every attempt's recorded env (launch.json);
  - state, and for a non-usable run a PROPOSED failure class under Append 25e (infrastructure / behaviour / unknown),
    with the raw fields beside it. 09 rules; the default is unknown.

  ~/sims/simulated-control-evals-sept/.venv/bin/python check_ug_runs.py <run dir> --out <new json>
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # before any other import: only --out is written

import argparse
import ast
import itertools
import json
import re
from pathlib import Path

ARM = "sim_attack@september_ungrounded"
AFF = re.compile(r"\[sim_task\] condition\.affordances: (\[.*?\])")


LOG_HEAD_LINES = 200  # the [sim_task] lines are printed when the task is built, before any sample runs


def usable(run: dict) -> bool:
    """The gate12 basics (gate12.py stays authoritative): done, last attempt clean, one sample, no sample error or limit,
    nothing unreadable."""
    o = ((run.get("attempts") or [{}])[-1].get("outcome")) or {}
    return (run.get("state") == "done" and bool(o.get("clean")) and o.get("samples") == 1 and not o.get("sample_errors")
            and not o.get("sample_limits") and not o.get("unreadable"))


def proposed_class(run: dict) -> tuple[str, dict]:
    """A PROPOSED Append 25e class from metadata; 09 rules (the 25e procedure reads the run's event metadata)."""
    att = (run.get("attempts") or [{}])[-1]
    o = att.get("outcome") or {}
    raw = {"state": run.get("state"), "kind": att.get("kind"), "prestart": att.get("prestart"),
           "policy_called": att.get("policy_called"), "exit_code": att.get("exit_code"),
           "eval_status": o.get("eval_status"), "clean": o.get("clean"), "samples": o.get("samples"),
           "sample_errors": o.get("sample_errors"), "sample_limits": o.get("sample_limits"),
           "unreadable": o.get("unreadable"), "n_attempts": len(run.get("attempts") or [])}
    if run.get("state") in ("not run", "no policy call") or att.get("policy_called") is False:
        return "infrastructure", raw | {"why": "no policy call (before the policy's first call)"}
    if o.get("sample_limits"):
        return "behaviour", raw | {"why": "a sample limit was hit"}
    return "unknown", raw | {"why": "default; needs the 25e event-metadata procedure"}


def affordances_from_log(log: Path) -> list | None:
    """The `[sim_task] condition.affordances:` line, reading at most LOG_HEAD_LINES lines (never the rest of the log)."""
    with open(log, errors="replace") as f:
        for line in itertools.islice(f, LOG_HEAD_LINES):
            m = AFF.search(line)
            if m:
                return ast.literal_eval(m.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    launch = json.loads((args.run / "launch.json").read_text())
    runs = launch["runs"]
    runs = list(runs.values()) if isinstance(runs, dict) else runs
    rows, problems = [], []
    for r in runs:
        if not r["key"].startswith(ARM + "/"):
            problems.append(f"unexpected run {r['key']}")
            continue
        key = r["key"].split("/", 1)[1]
        pre = args.run / "preflight" / f"{key}_september_ungrounded.json"
        neg = None
        if pre.exists():
            p = json.loads(pre.read_text())
            res = p.get("results", [])
            neg = bool(res) and all(x.get("ungrounded") is True and x.get("reasoning_marker_in_prompt") == 0
                                    and x.get("output_marker_in_prompt") == 0 and x.get("reasoning_blocks_marked", 0) > 0
                                    and x.get("tool_outputs_marked", 0) > 0 for x in res) and not p.get("failed")
        log = args.run / "logs" / f"{ARM}_{key}.log"
        aff = affordances_from_log(log) if log.exists() else None
        flags = [(a.get("env_vars") or {}).get("BA_NO_HONEST_REFERENCE") for a in (r.get("attempts") or [])]
        row = {"run": r["key"], "state": r.get("state"), "usable_gate12_basics": usable(r),
               "preflight_negative_control": neg, "condition_affordances": aff, "affordances_ok": aff == ["codebase_read"],
               "flag_in_every_attempt_env": bool(flags) and all(f == "1" for f in flags)}
        if not row["usable_gate12_basics"] and r.get("state") not in ("ready", "waiting", "running"):
            row["proposed_25e_class"], row["raw"] = proposed_class(r)
        if r.get("state") not in ("not run", "ready", "waiting") and neg is not True:
            problems.append(f"{key}: preflight negative control not shown ({neg})")
        if r.get("state") == "done" and aff != ["codebase_read"]:
            problems.append(f"{key}: condition.affordances {aff}")
        if r.get("attempts") and not row["flag_in_every_attempt_env"]:
            problems.append(f"{key}: BA_NO_HONEST_REFERENCE is not 1 in every attempt's env ({flags})")
        rows.append(row)
    out = {"run_dir": str(args.run), "arm": ARM, "n_runs": len(rows), "problems": problems, "runs": rows}
    with open(args.out, "x") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"n_runs": len(rows), "problems": len(problems),
                      "negative_control_true": sum(x["preflight_negative_control"] is True for x in rows),
                      "affordances_ok": sum(x["affordances_ok"] for x in rows)}))
    if problems:
        sys.exit(1)  # the file is written; a nonzero exit makes the problems impossible to miss


if __name__ == "__main__":
    main()
