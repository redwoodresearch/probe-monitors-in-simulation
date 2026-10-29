"""Blinded sheet v3: v2 without the task column (#1007 review, 12:32 PT; Aswin asked for "tool type and the impactful
flag", not task). Same blind_ids, same order, so the held v2 key and the command bundle stay valid. $0, no text.

Leak audit (aggregates only, printed): with task gone, the visible columns are tool and existing_label. Reports the rows
in (tool, existing_label) cells that hold only one arm, only real or only simulated rows, or only honest or only attack
rows. The key is read for the audit only, and its path is not printed.

  uv run --no-project python make_blind_sheet_v3.py <v2 sheet> <held v2 key> --out-dir <sheet dir>
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
V2_SHA = "707188aedd55fc7091c090e6a555bb8aa61b4e83f10897d06fc52eb94dda8385"
KEY_SHA = "8af4a987"
SIM = {"sim_honest", "sim_attack"}
HONEST = {"real_honest_A", "real_honest_B", "sim_honest"}


class Parser(argparse.ArgumentParser):
    def error(self, message):  # no echo of the arguments: they include the held key's path
        sys.stderr.write("failed: bad arguments\n")
        sys.exit(2)


def main(v2: str, key: str, out_dir: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    if hashlib.sha256(Path(v2).read_bytes()).hexdigest() != V2_SHA or not hashlib.sha256(Path(key).read_bytes()).hexdigest().startswith(KEY_SHA):
        raise SystemExit("input sha256 mismatch")
    rows = list(csv.DictReader(open(v2, newline="")))
    arm = {r["blind_id"]: r["arm"] for r in csv.DictReader(open(key, newline=""))}
    if len(arm) != len(rows) or any(r["blind_id"] not in arm for r in rows):
        raise SystemExit("sheet and key differ")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = Path(out_dir) / f"dt_label_sheet_blind_v3_{stamp}.csv"
    with out.open("x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["blind_id", "tool", "existing_label", "destructive", "transfer"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ("blind_id", "tool", "existing_label", "destructive", "transfer")})
    cells = defaultdict(set)
    for r in rows:
        cells[(r["tool"], r["existing_label"])].add(arm[r["blind_id"]])
    n = len(rows)
    one = sum(len(cells[(r["tool"], r["existing_label"])]) == 1 for r in rows)
    rs = sum(len({a in SIM for a in cells[(r["tool"], r["existing_label"])]}) == 1 for r in rows)
    ha = sum(len({a in HONEST for a in cells[(r["tool"], r["existing_label"])]}) == 1 for r in rows)
    print(json.dumps({"sheet": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "rows": n, "cells": len(cells),
                      "rows_in_one_arm_cells": one, "rows_in_only_real_or_only_sim_cells": rs, "rows_in_only_honest_or_only_attack_cells": ha}))


if __name__ == "__main__":
    ap = Parser()
    ap.add_argument("v2")
    ap.add_argument("key")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    main(a.v2, a.key, a.out_dir)
