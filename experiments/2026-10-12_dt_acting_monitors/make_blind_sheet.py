"""Blinded labelling sheet and its separate key, from the ID-only sheet (2026-10-12; $0, no text, no labelling).

Aswin, relayed by the orchestrator at about 12:05 PT: "Labelling: I'll label myself. Send me … (b) an ID-only sheet with
real vs simulated hidden (arm column removed, rows shuffled, key held separately), showing tool type and the impactful
flag; (c) counts of impactful calls per tool type. No scripted labelling on your side."

- Sheet: blind_id, task, tool, existing_label, destructive (empty), transfer (empty); rows shuffled. arm, run_key and
  irid are removed (run_key's file name shows real / honest).
- Key, written to a different folder: blind_id -> arm, run_key, irid, call, call_index.
- blind_id is 12 hex characters and the shuffle is random. Both come from a generator seeded with SECRET randomness
  (`secrets.randbits(128)`, per #1007's review). The seed is written only to the key-side summary in the key folder,
  never to stdout, git or the sheet side. So the mapping can't be rebuilt from the repo and the source sheet.
- Sheet-side summary: impactful calls per tool type overall. Key-side summary: per arm and per tool (key folder only).
- Leak audit, in the key-side summary and printed as aggregates only: the share of rows whose visible columns
  (task, tool, existing_label) occur in only one arm, only real or only simulated rows, or only honest or only attack rows.

  uv run --no-project --with numpy python make_blind_sheet.py <sheet.csv> --sheet-dir <dir> --key-dir <dir>
"""

import argparse
import secrets
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
SHEET_SHA = "9d43c790e5dc7415d2e6e23f67673dba5e4946de11e7b61c7f1d253d3d9bf87b"
SIM = {"sim_honest", "sim_attack"}
HONEST = {"real_honest_A", "real_honest_B", "sim_honest"}


def main(sheet: str, sheet_dir: str, key_dir: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if Path(sheet_dir).resolve() == Path(key_dir).resolve():
        raise SystemExit("the key must live apart from the sheet")
    raw = Path(sheet).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHEET_SHA:
        raise SystemExit("input sheet sha256 mismatch")
    rows = list(csv.DictReader(raw.decode().splitlines()))
    seed = secrets.randbits(128)  # secret: recorded only in the key-side summary
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    ids = set()
    while len(ids) < len(rows):
        ids |= {f"{x:012x}" for x in rng.integers(0, 16 ** 12, len(rows) - len(ids) + 100)}
    ids = sorted(ids)[: len(rows)]
    ids = [ids[i] for i in rng.permutation(len(ids))]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    sd, kd = Path(sheet_dir), Path(key_dir)
    sd.mkdir(parents=True, exist_ok=True)
    kd.mkdir(parents=True, exist_ok=True)
    s_path, k_path = sd / f"dt_label_sheet_blind_{stamp}.csv", kd / f"dt_label_key_{stamp}.csv"
    with s_path.open("x", newline="") as fs, k_path.open("x", newline="") as fk:
        ws = csv.DictWriter(fs, fieldnames=["blind_id", "task", "tool", "existing_label", "destructive", "transfer"])
        wk = csv.DictWriter(fk, fieldnames=["blind_id", "arm", "run_key", "irid", "call", "call_index"])
        ws.writeheader()
        wk.writeheader()
        for bid, i in zip(ids, order):
            r = rows[i]
            ws.writerow({"blind_id": bid, "task": r["task"], "tool": r["tool"], "existing_label": r["existing_label"], "destructive": "", "transfer": ""})
            wk.writerow({"blind_id": bid, "arm": r["arm"], "run_key": r["run_key"], "irid": r["irid"], "call": r["call"], "call_index": r["call_index"]})
    # leak audit over the visible columns
    cell_arms = defaultdict(set)
    for r in rows:
        cell_arms[(r["task"], r["tool"], r["existing_label"])].add(r["arm"])
    n = len(rows)
    one_arm = sum(len(cell_arms[(r["task"], r["tool"], r["existing_label"])]) == 1 for r in rows)
    only_real_or_sim = sum(len({a in SIM for a in cell_arms[(r["task"], r["tool"], r["existing_label"])]}) == 1 for r in rows)
    only_hon_or_att = sum(len({a in HONEST for a in cell_arms[(r["task"], r["tool"], r["existing_label"])]}) == 1 for r in rows)
    sim_possible_tasks = {r["task"] for r in rows if r["arm"] == "sim_honest"}
    honest_rows_t21 = sum(r["arm"] in HONEST and r["task"] not in sim_possible_tasks for r in rows)
    tools_by_arm = defaultdict(Counter)
    for r in rows:
        tools_by_arm[r["tool"]][r["arm"]] += 1
    tool_only_in = {t: sorted(c) for t, c in tools_by_arm.items() if len({a in SIM for a in c}) == 1 or len(c) == 1}
    audit = {"rows": n, "visible_columns": ["task", "tool", "existing_label"],
             "rows_in_cells_with_one_arm": one_arm, "rows_in_cells_only_real_or_only_sim": only_real_or_sim,
             "rows_in_cells_only_honest_or_only_attack": only_hon_or_att,
             "honest_rows_in_tasks_without_sim_honest": honest_rows_t21, "tools_seen_only_in_real_or_only_in_sim_or_one_arm": tool_only_in}
    sheet_summary = {"sheet": s_path.name, "sheet_sha256": hashlib.sha256(s_path.read_bytes()).hexdigest(), "rows": n, "seed": "secret (key side only)", "git": git,
                     "impactful_per_tool": dict(Counter(r["tool"] for r in rows if r["existing_label"] == "impactful").most_common()),
                     "read_only_per_tool": dict(Counter(r["tool"] for r in rows if r["existing_label"] == "read_only").most_common()),
                     "source_sheet_sha256": SHEET_SHA}
    key_summary = {"key": k_path.name, "key_sha256": hashlib.sha256(k_path.read_bytes()).hexdigest(), "seed": str(seed), "git": git,
                   "per_arm_per_tool_impactful": {a: dict(Counter(r["tool"] for r in rows if r["arm"] == a and r["existing_label"] == "impactful"))
                                                  for a in sorted({r["arm"] for r in rows})}, "leak_audit": audit}
    with (sd / f"dt_label_sheet_blind_summary_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(sheet_summary, indent=1))
    with (kd / f"dt_label_key_summary_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(key_summary, indent=1))
    print("sheet", s_path, sheet_summary["sheet_sha256"])
    print("key", k_path, key_summary["key_sha256"])
    print("impactful per tool (overall):", json.dumps(sheet_summary["impactful_per_tool"]))
    print(f"leak audit (aggregates): rows {n}; in a visible cell with one arm only {one_arm} ({100 * one_arm / n:.1f}%); "
          f"only real or only sim {only_real_or_sim} ({100 * only_real_or_sim / n:.1f}%); only honest or only attack {only_hon_or_att} "
          f"({100 * only_hon_or_att / n:.1f}%); honest rows in tasks without sim honest {honest_rows_t21} ({100 * honest_rows_t21 / n:.1f}%); "
          f"tools restricted by world: {len(tool_only_in)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--sheet-dir", required=True)
    ap.add_argument("--key-dir", required=True)
    a = ap.parse_args()
    main(a.sheet, a.sheet_dir, a.key_dir)
