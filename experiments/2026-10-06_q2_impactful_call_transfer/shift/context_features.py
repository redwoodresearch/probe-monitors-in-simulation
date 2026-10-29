"""Per call (attack view, Qwen3-32B and Qwen3.5 tables): features of what the reader saw before the pre_call readout.

  prev_*: the call whose output the reader saw last (the last call of an earlier turn; calls of one turn are all
    written before any of their outputs) and that output (tool, editor command, format);
  n_editor_before / n_editor_invented_before / n_simfmt_before: earlier text_editor outputs in the run, those flagged
    `none` against the reference, and those in the simulator's edit format (a str_replace or insert diff with no
    line-number gutter, or an edit/create output ending in a newline, which real ones almost never do);
  next_kind: this call's own kind (view / edit / create / bash-like / other);
  agent_chars, agent_text: the assistant text in this call's turn (reasoning is not visible to the reader).
Writes context_features.parquet keyed by (arm, key, step, action).
"""
import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

import examples as ex  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402

PIPE, TAB = re.compile(r"^\s*\d+\|", re.M), re.compile(r"^\s*\d+\t", re.M)


def sim_format(cmd, out):
    """True when an editor edit output has the simulator's format, False when it has the real one, None otherwise. Any of
    the simulator's tells counts: a trailing newline (any edit), a str_replace diff with no line-number gutter, an
    insert answered with a diff instead of a `cat -n` snippet."""
    if cmd not in ("create", "str_replace", "insert") or not out:
        return None
    tells = [out.endswith("\n")]
    if cmd == "str_replace" and out.startswith("The file") and "diff" in out[:200]:
        tells.append(not PIPE.search(out))
    if cmd == "insert" and out.startswith("The file"):
        tells.append("Here's a diff" in out[:200] or not TAB.search(out))
    return any(tells)


def kind(c):
    if c["tool"] != "text_editor":
        return "bash-like" if "bash" in c["tool"] or c["tool"].endswith("_restart") else "other"
    cmd = ex.editor_args(c)[0]
    return {"view": "view", "create": "create", "str_replace": "edit", "insert": "edit"}.get(cmd, "other editor")


def main():
    q = ex.table("qwen3-32b")
    runs = ex.load_runs(set(q.key))
    arm_of = dict(zip(q.key, q.arm))
    flags = {(r["arm"], r["key"], r["step"], r["action"]): r["out_match"]
             for r in map(json.loads, gzip.open(ex.FLAGS, "rt")) if r["arm"] in ex.COHORT}
    rows = []
    for key, calls in runs.items():
        arm = arm_of[key]
        n_ed = n_inv = n_sf = n_edit = 0
        order, pk = sorted(calls), None  # pk: the call whose output the reader saw last (the last call of an earlier step)
        for s in sorted({o[0] for o in order}):
            step_calls = [o for o in order if o[0] == s]  # one assistant turn: every call is written before any output
            prev = calls[pk] if pk else None
            pcmd = ex.editor_args(prev)[0] if prev and prev["tool"] == "text_editor" else None
            pout = (prev or {}).get("output") or ""
            for sa in step_calls:
                c = calls[sa]
                rows.append({"arm": arm, "key": key, "step": sa[0], "action": sa[1], "next_kind": kind(c),
                             "prev_kind": kind(prev) if prev else "first", "prev_cmd": pcmd,
                             "prev_simfmt": sim_format(pcmd, pout) if pcmd else None,
                             "prev_flag": flags.get((arm, key) + pk) if pk else "first",
                             "n_editor_before": n_ed, "n_editor_invented_before": n_inv, "n_simfmt_before": n_sf, "n_edit_before": n_edit,
                             "agent_chars": len(c["text"].strip()), "agent_text": c["text"].strip()[:2000]})
            for sa in step_calls:  # this turn's outputs are visible from the next turn on
                c = calls[sa]
                if c["tool"] == "text_editor":
                    n_ed += 1
                    n_inv += flags.get((arm, key) + sa) == "none"
                    n_sf += bool(sim_format(ex.editor_args(c)[0], c["output"] or ""))
                    n_edit += ex.editor_args(c)[0] in ("create", "str_replace", "insert")
            pk = step_calls[-1]
    out = pd.DataFrame(rows)
    out.to_parquet(DATA / "context_features.parquet")
    print(out.groupby(["arm", "prev_cmd", "prev_simfmt"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
