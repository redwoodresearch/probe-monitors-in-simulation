"""Scoring rows for the 116 new held-out tasks (RULE_scaled.md + Amendments 1–2). IDs, reading locations and numbers only.

Every call of the 348 runs (task-list ranks 1–116, runs_2026-10-13_125944.csv) in the committed ID list (calls_all_…csv, call_ids as
in Aswin's K = 116 votes), with its layer-30 call_end location from the extraction metas (check_joins.entries, the same locations
every earlier scoring job used). It checks by script, before any scoring, that every state-changing call has a stored reading
(the rule stops otherwise) and reports any other call without one.

  uv run --no-project --python 3.12 --with pandas --with numpy python make_scaled_rows.py --inputs <stage-1 ID inputs dir> \\
      --metas <acts_meta root> --calls <calls_all csv> --runs <runs csv> --out results
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coverage as COV  # noqa: E402  (check_metas, J = check_joins)

PINS = {"calls": "6dc132affb389e3dd66737d2d5db199c2bb5fe70e13a723a3c0af319e95d7595", "runs": "52a2e288b1cc684d8b17016adda88778fb5077388961ccd6aa3bdcf4c915accf"}
COLS = ["call_id", "world", "task", "run_id", "epoch", "turn", "call_index", "irid", "state_changing", "x", "chunk", "row"]


def main(inputs: str, metas: str, calls: str, runs: str, out: str):
    for n, p in {"calls": calls, "runs": runs}.items():
        if COV.sha256(p) != PINS[n]:
            raise SystemExit(f"{n}: sha256 mismatch")
    digests = COV.check_metas(Path(metas))
    _, call_end = COV.J.entries(metas)
    U = pd.read_csv(runs)
    want = set(U.run_id[(U.side == "held_out") & (U["rank"] <= 116)])
    C = pd.read_csv(calls)
    C = C[C.run_id.isin(want)].copy()
    if C.run_id.nunique() != 348 or len(want) != 348:
        raise SystemExit("expected the 348 runs of ranks 1-116")
    loc = [call_end.get((w, i, c)) for w, i, c in zip(C.world, C.irid, C.call_index)]
    C["has_reading"] = [v is not None for v in loc]
    if not (C.has_reading == C.has_call_end_reading).all():
        raise SystemExit("reading locations differ from the ID list's has_call_end_reading")
    miss_sc = C[(C.state_changing == 1) & ~C.has_reading]
    if len(miss_sc):
        raise SystemExit(f"{len(miss_sc)} state-changing calls have no stored reading: the rule stops here")
    C["x"], C["chunk"], C["row"] = ([v[k] if v else None for v in loc] for k in range(3))
    R = C[C.has_reading][COLS].copy()
    R["row"] = R.row.astype(int)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    o = Path(out)
    dst = o / f"scaled_rows_{stamp}.csv"
    with dst.open("x") as fh:
        R.to_csv(fh, index=False)
    cnt = lambda df: {w: int((df.world == w).sum()) for w in ("real", "run2", "sim")}  # noqa: E731
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    res = {"written": stamp, "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", str(HERE)))},
           "inputs_sha256": PINS | {n: COV.sha256(Path(inputs) / n) for n in COV.P.SHA}, "metas_digests": digests,
           "rows_file": {"name": dst.name, "rows": len(R), "sha256": COV.sha256(dst)},
           "calls_by_world": cnt(C), "rows_by_world": cnt(R), "state_changing_by_world": cnt(C[C.state_changing == 1]),
           "state_changing_without_reading": 0, "other_calls_without_reading_by_world": cnt(C[~C.has_reading]),
           "chunks": int(R[["x", "chunk"]].drop_duplicates().shape[0])}
    with (o / f"scaled_rows_summary_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), "\n->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--inputs", "--metas", "--calls", "--runs", "--out"):
        ap.add_argument(a, required=True)
    main(**vars(ap.parse_args()))
