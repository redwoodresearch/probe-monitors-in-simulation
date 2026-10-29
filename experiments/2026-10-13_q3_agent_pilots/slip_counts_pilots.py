"""Tool-call slips per agent-pilot model, keys only (d9 via 0d, 10-13: "report the argument-name slip rate beside the
selection numbers"). Read-only, $0. No argument values, outputs or model text are read into the output.

  uv run --no-project --python 3.14 python slip_counts_pilots.py --model {gptoss,qcoder} --runs <run dir> [<run dir> ...]

From every honest .eval's tool events (the calls as Inspect handled them):
- text_editor edit calls (str_replace, insert, create) by argument names: "schema names" when every key is one of the
  tool schema's parameters (as sent to the model in the runs: command, path, file_text, insert_line, insert_text,
  new_str, old_str, view_range) and the command's required ones are there (str_replace: old_str; insert: insert_line,
  insert_text; create: file_text). Otherwise "outside names" when a key is not a schema parameter (e.g. gpt-oss's
  old/new), with the sorted key sets, or else "missing required" when only schema names are used but a required one is
  missing (e.g. insert with new_str and no insert_text), with the missing names (Codex, 10-13: keep the two apart);
  plus str_replace calls with schema names but no new_str;
- every call Inspect rejected as a "parsing" error, by kind, from the error message's fixed prefix: unknown tool name,
  unexpected argument names, missing required argument, value not valid for the schema, unparseable JSON, other.
Per run and in total. A cancelled run (status not "success") is counted separately. Writes results/slips_<model>_<stamp>.json.
"""

import argparse
import glob
import json
import re
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
PROPS = {"command", "path", "file_text", "insert_line", "insert_text", "new_str", "old_str", "view_range"}
REQUIRED = {"str_replace": {"old_str"}, "insert": {"insert_line", "insert_text"}, "create": {"file_text"}}
KINDS = [("unknown tool", r"^Tool .+ not found"), ("unexpected argument names", r"Additional properties are not allowed"),
         ("missing required argument", r"is a required property"), ("value not valid for the schema", r"is not valid under any|is not of type"),
         ("unparseable JSON", r"^Error parsing the following tool call arguments")]


def kind(msg: str) -> str:
    return next((k for k, pat in KINDS if re.search(pat, msg)), "other")


def one(path: str) -> dict:
    with zipfile.ZipFile(path) as z:
        status = json.loads(z.read("header.json")).get("status") if "header.json" in z.namelist() else None
        names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
        s = json.loads(z.read(names[0])) if names else {"events": []}
    edits, off_keys, missing, parsing = Counter(), Counter(), Counter(), Counter()
    calls = 0
    for e in s.get("events", []):
        if e.get("event") != "tool":
            continue
        calls += 1
        args, err = e.get("arguments") or {}, e.get("error") or {}
        if err.get("type") == "parsing":
            parsing[kind(str(err.get("message") or ""))] += 1
        cmd = args.get("command")
        if e.get("function") == "text_editor" and cmd in REQUIRED:
            if not set(args) <= PROPS:
                edits[f"{cmd}: outside names"] += 1
                off_keys[",".join(sorted(args))] += 1
            elif not REQUIRED[cmd] <= set(args):
                edits[f"{cmd}: missing required"] += 1
                missing[f"{cmd}: " + ",".join(sorted(REQUIRED[cmd] - set(args)))] += 1
            else:
                edits[f"{cmd}: schema names"] += 1
                if cmd == "str_replace" and "new_str" not in args:
                    edits["str_replace: schema names, no new_str"] += 1
    return {"status": status, "tool_events": calls, "edits": dict(edits), "off_schema_key_sets": dict(off_keys),
            "missing_required": dict(missing),
            "parsing_errors_by_kind": dict(parsing)}


def main(model: str, runs: list[str], git: str | None = None):
    git = git or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    rows = []
    for run in runs:
        for f in sorted(glob.glob(f"{run}/honest_*/*/*.eval")):
            p = Path(f)
            rows.append({"launch": Path(run).name, "arm": p.parts[-3], "task": p.parts[-2], "eval": p.name} | one(f))
    out = {"git": git, "model": model, "runs": len(rows)}
    for label, sel in (("finished", [r for r in rows if r["status"] == "success"]), ("not_finished", [r for r in rows if r["status"] != "success"])):
        tot = lambda k: dict(sum((Counter(r[k]) for r in sel), Counter()))  # noqa: E731
        out[label] = {"runs": len(sel), "tool_events": sum(r["tool_events"] for r in sel), "edits": tot("edits"),
                      "off_schema_key_sets": tot("off_schema_key_sets"), "missing_required": tot("missing_required"),
                      "parsing_errors_by_kind": tot("parsing_errors_by_kind"),
                      "runs_with_an_off_schema_edit": sum(bool(r["off_schema_key_sets"]) for r in sel)}
    out["rows"] = rows
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"slips_{model}_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gptoss", "qcoder"], required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--git", help="the commit, where git isn't available (the Modal rerun)")
    main(**vars(ap.parse_args()))
