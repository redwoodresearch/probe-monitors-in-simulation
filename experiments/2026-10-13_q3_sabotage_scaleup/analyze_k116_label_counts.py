"""Label counts in Aswin's frozen K = 116 panel export (labels only: no probe score is read or joined; no text).

Joins the frozen votes (scaleup_panel_votes_k116.csv, sha256 78ae0fa4…, results/manifest_k116_panel_s3_v1.json) to the committed
ID list (calls_all_2026-10-13_125944.csv, 6dc132af…, results/manifest_scaleup_s3_v1.json) on call_id. It checks that every shared
field agrees, that the votes cover exactly the 348 runs of task-list ranks 1–116 (runs_2026-10-13_125944.csv, 52a2e288…), and that
every task is a held-out, non-stage-1 task in split_v1.json (992e6e71…). Then it counts sabotage / not_sabotage / unclear among
state-changing calls per world (first real run `real`, second real run `run2`, simulated `sim`), per task, and the tasks with
sabotage.

  uv run --no-project --python 3.12 --with pandas python analyze_k116_label_counts.py --votes <votes csv> --calls <calls_all csv> \\
      --runs <runs csv> --split <split_v1.json> --out results
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PINS = {"votes": "78ae0fa4e7e85da9a1ee85c9944be917b47e63c0d0a62f6f71390f046fdb7fcf", "calls": "6dc132affb389e3dd66737d2d5db199c2bb5fe70e13a723a3c0af319e95d7595",
        "runs": "52a2e288b1cc684d8b17016adda88778fb5077388961ccd6aa3bdcf4c915accf", "split": "992e6e71793146b4df5f00d7cd98330d4f5e073f0a8379031aebfc172f818e11"}
SHARED = ["world", "task", "split", "run_id", "epoch", "turn", "call_index", "ordinal", "state_changing"]
LABELS, WORLDS = ["sabotage", "not_sabotage", "unclear"], ["real", "run2", "sim"]


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(votes: str, calls: str, runs: str, split: str, out: str):
    for n, p in {"votes": votes, "calls": calls, "runs": runs, "split": split}.items():
        if sha256(p) != PINS[n]:
            raise SystemExit(f"{n}: sha256 mismatch")
    V, C, U = pd.read_csv(votes), pd.read_csv(calls), pd.read_csv(runs)
    sp = json.loads(Path(split).read_text())["tasks"]
    want = U[(U.side == "held_out") & (U["rank"] <= 116)]
    if len(want) != 348 or set(V.run_id) != set(want.run_id) or V.call_id.duplicated().any():
        raise SystemExit("the votes don't cover exactly the 348 runs of ranks 1-116, once each")
    M = C[C.run_id.isin(want.run_id)].merge(V, on="call_id", how="outer", suffixes=("", "_v"), validate="one_to_one", indicator=True)
    if (M._merge != "both").any():
        raise SystemExit(f"calls without a vote or votes without an ID-list call: {M._merge.value_counts().to_dict()}")
    for c in SHARED:
        if not (M[c].astype(str) == M[f"{c}_v"].astype(str)).all():
            raise SystemExit(f"{c} differs between the votes and the ID list")
    if not M.label.isin(LABELS).all():
        raise SystemExit("unexpected label")
    bad = sorted(t for t in M.task.unique() if sp.get(t, {}).get("side") != "held_out" or sp[t]["stage1"])
    if bad:
        raise SystemExit(f"tasks that aren't held-out non-stage-1 tasks in split_v1: {bad}")
    sc = M[M.state_changing == 1]
    tab = lambda df: {k: int((df.label == k).sum()) for k in LABELS}  # noqa: E731
    per_task = {w: {t: tab(g) for t, g in sc[sc.world == w].groupby("task")} for w in WORLDS}
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()  # noqa: E731
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = {"written": stamp, "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "."))},
           "inputs_sha256": PINS, "labels_only_no_scores": True,
           "calls": len(M), "runs": int(M.run_id.nunique()), "tasks": int(M.task.nunique()),
           "split_check": "all tasks are held-out, non-stage-1 tasks in split_v1.json", "state_changing": len(sc),
           "state_changing_by_world": {w: int((sc.world == w).sum()) for w in WORLDS},
           "state_changing_labels_by_world": {w: tab(sc[sc.world == w]) for w in WORLDS},
           "state_changing_confident_by_world": {w: int(((sc.world == w) & (sc.confident == "yes")).sum()) for w in WORLDS},
           "not_state_changing_labels_by_world": {w: tab(M[(M.state_changing != 1) & (M.world == w)]) for w in WORLDS},
           "tasks_with_sabotage_by_world": {w: int(sum(v["sabotage"] > 0 for v in per_task[w].values())) for w in WORLDS},
           "tasks_with_state_changing_calls_by_world": {w: len(per_task[w]) for w in WORLDS},
           "valid_votes_state_changing": {str(k): int(v) for k, v in sc.valid_votes.value_counts().sort_index().items()},
           "state_changing_labels_per_task": per_task}
    with (Path(out) / f"k116_label_counts_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "state_changing_labels_per_task"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--votes", "--calls", "--runs", "--split", "--out"):
        ap.add_argument(a, required=True)
    main(**vars(ap.parse_args()))
