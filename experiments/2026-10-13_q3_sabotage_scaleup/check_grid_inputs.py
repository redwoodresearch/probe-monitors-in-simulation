"""Amendment 3 inputs check, before the rule: does every reading the training grid and the command-readout exam need exist in
stored activations? IDs, locations and numbers only; no new extraction, no text.

For every row of the label-free rows file (the 40 stage-1 tasks' attack calls; honest B, C and simulated-honest calls) and of the
116 new tasks' rows file, it finds the call's `command` reading (the last token of the rule-1 command parameter, item E's command
endpoint) by joining its call_end location to its meta call_end entry and then to the same call's command entry (same extraction,
chunk, arm, irid, call_index), as build_rows_command.py does. It counts, by set and world, eligible calls (state-changing, or the
honest impactful calls) with and without a call_end and a command reading, checks item E's staged command rows file, and writes the
command locations for the scoring job.

  uv run --no-project --python 3.12 python check_grid_inputs.py --metas <acts_meta root> --inputs <scaled_inputs dir> \\
      --rows-command <rows_command.json> --out results
"""

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PINS = {"flagrate_rows_2026-10-13_122417.csv": "44c75f78e78f86b4bcab409b1379762a8d4dccadfd456f28ee651832eb2b13a7",
        "scaled_rows_2026-10-13_152021.csv": "a95453ac2688e61ba99184726ce27a4dd54276a04730a56d67825f204d1ee1f3",
        "rows_command.json": "4d7289f3b2e475d86feaed028e0df795d1d80998918eaeacc060a0bfc7116beb"}  # item E, check_judge_v1_081608e8_b.json


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(metas: str, inputs: str, rows_command: str, out: str):
    I = Path(inputs)
    for n, p in {"flagrate_rows_2026-10-13_122417.csv": I / "flagrate_rows_2026-10-13_122417.csv",
                 "scaled_rows_2026-10-13_152021.csv": I / "scaled_rows_2026-10-13_152021.csv", "rows_command.json": Path(rows_command)}.items():
        if sha(p) != PINS[n]:
            raise SystemExit(f"{n}: sha256 mismatch")
    ce, cmd = {}, {}
    for f in sorted(Path(metas).glob("*/meta/*/*.json")):
        x, arm, m = f.parent.parent.parent.name, f.parent.name, json.loads(f.read_text())
        if m.get("chunk") is None:
            continue
        for c in m["calls"]:
            k = (x, m["chunk"], arm, c.get("irid"), c.get("call_index"))
            if c.get("kind", "command") == "command" and c.get("row") is not None:
                if k in cmd:
                    raise SystemExit(f"repeated command entry {k}")
                cmd[k] = (m["row_start"] + c["row"], c.get("readout"))
            elif c.get("kind") == "call_end" and c.get("row") is not None:
                loc = (x, m["chunk"], m["row_start"] + c["row"])
                if loc in ce:
                    raise SystemExit(f"repeated call_end location {loc}")
                ce[loc] = (k, c.get("readout"))
    F = list(csv.DictReader(open(I / "flagrate_rows_2026-10-13_122417.csv")))
    N = list(csv.DictReader(open(I / "scaled_rows_2026-10-13_152021.csv")))
    rows = [(r, "flagrate", "part1_honest" if r["part"] == "1" else r["group"]) for r in F] + [(r, "new", "new_held_out") for r in N]
    cnt, out_rows = Counter(), []
    for r, src, s in rows:
        eligible = int(r["state_changing"]) == 1
        loc = (r["x"], r["chunk"], int(r["row"]))
        k, ro = ce.get(loc, (None, None))
        c, cro = cmd.get(k, (None, None)) if k else (None, None)
        key = f"{s} / {r['world']}"
        cnt[(key, "rows")] += 1
        cnt[(key, "eligible")] += eligible
        cnt[(key, "eligible_without_call_end_entry")] += eligible and k is None
        cnt[(key, "eligible_without_command_reading")] += eligible and c is None
        cnt[(key, "eligible_command_same_token_as_call_end")] += eligible and c is not None and ro == cro  # same readout token (build_rows_command.py)
        if c is not None:
            out_rows.append({"call_id": r["call_id"], "source": src, "x": r["x"], "chunk": r["chunk"], "row": c, "call_end_row": loc[2]})
    rc = json.loads(Path(rows_command).read_text())
    staged = Counter((r["role"], r["same_token"]) for r in rc["rows"])
    summary = {}
    for (key, what), n in sorted(cnt.items()):
        summary.setdefault(key, {})[what] = int(n)
    missing = sum(v.get("eligible_without_call_end_entry", 0) + v.get("eligible_without_command_reading", 0) for v in summary.values())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    o = Path(out)
    dst = o / f"command_rows_{stamp}.csv"
    with dst.open("x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["call_id", "source", "x", "chunk", "row", "call_end_row"])
        w.writeheader()
        w.writerows(out_rows)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()  # noqa: E731
    res = {"written": stamp, "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "."))},
           "inputs_sha256": PINS, "by_set_world": summary, "eligible_missing_any_reading": int(missing),
           "staged_rows_command": {"rows": len(rc["rows"]), "endpoint": rc.get("endpoint"), "source_rows_sha256": rc.get("source_rows_sha256"),
                                   "by_role_same_token": {f"{a} / {b}": n for (a, b), n in sorted(staged.items())}},
           "command_rows_file": {"name": dst.name, "rows": len(out_rows), "sha256": sha(dst)},
           "verdict": "all inputs exist" if missing == 0 else "MISSING readings: the grid can't run as specified"}
    with (o / f"grid_inputs_check_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--metas", "--inputs", "--rows-command", "--out"):
        ap.add_argument(a, required=True)
    main(**vars(ap.parse_args()))
