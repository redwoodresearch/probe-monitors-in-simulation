"""Label counts in Aswin's frozen simulated-run panel votes (labels only: no probe score is read or joined; no text).

Joins a frozen votes file to its ID list on call_id (simulated: votes 06f4ef4b… with sim_calls_all 13b0a3da…; real runs 1 and 2:
votes 9a96988d… with real_run2_calls_all 392eff0e…; S3 per this folder's manifests), checks that both files agree on every shared field, and counts
sabotage / not_sabotage / unclear among state-changing calls by split, plus held-out and training per-task sabotage counts
(for the precision estimate).

  uv run --no-project --python 3.12 --with pandas python analyze_label_counts.py --votes <votes csv> --calls <ID list csv> --out <dir>
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PAIRS = {"06f4ef4bc00b53767b074185cda99311f2ba44f7c8f5a4f38ce59dd1537e81ae": "13b0a3da826196eb051d9cd82be86e949200e359b19514f71de61423982ab8e2",  # simulated
         "9a96988db9818c8b97f3045ade83f19fae0e19e7d0cac134fa775005eba8dc36": "392eff0ed547fb18b62a08bb9cd8945d5206c0a281672780affd40825acf4a1a"}  # real runs 1, 2
SHARED = ["run_id", "epoch", "turn", "call_index", "ordinal", "state_changing"]


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(votes: str, calls: str, out: str):
    vs, cs = sha256(votes), sha256(calls)
    if PAIRS.get(vs) != cs:
        raise SystemExit("input sha256 mismatch")
    V, C = pd.read_csv(votes), pd.read_csv(calls)
    if "world" not in C:  # the simulated ID list has no world column
        C["world"] = "sim"
    if "world" in V and not (V.set_index("call_id").world.reindex(C.call_id).values == C.world.values).all():
        raise SystemExit("world differs between the votes and the ID list")
    V = V.drop(columns=["world"], errors="ignore")
    if len(V) != len(C) or V.call_id.duplicated().any() or set(V.call_id) != set(C.call_id):
        raise SystemExit("the votes and the ID list hold different calls")
    M = C.merge(V, on="call_id", suffixes=("", "_v"), validate="one_to_one")
    for c in SHARED:
        if not (M[c] == M[f"{c}_v"]).all():
            raise SystemExit(f"{c} differs between the votes and the ID list")
    if not M.label.isin(["sabotage", "not_sabotage", "unclear"]).all():
        raise SystemExit("unexpected label")
    sc = M[M.state_changing == 1]
    tab = lambda df: {str(k): int(v) for k, v in df.label.value_counts().sort_index().items()}  # noqa: E731
    L3 = ["sabotage", "not_sabotage", "unclear"]
    per_task = {f"{w} / {side}": {t: {k: int((g.label == k).sum()) for k in L3} for t, g in sc[(sc.world == w) & (sc.split == side)].groupby("task")}
                for w in sorted(sc.world.unique()) for side in ("held-out", "train")}
    if sc.world.nunique() == 1:  # one world (the simulated file): keep the split-only keys of the first version too (Codex)
        per_task |= {side: per_task[f"{sc.world.iloc[0]} / {side}"] for side in ("held-out", "train")}
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()  # noqa: E731
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = {"written": stamp, "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "."))},
           "inputs_sha256": {"votes": vs, "calls": cs}, "labels_only_no_scores": True,
           "calls": len(M), "state_changing": len(sc),
           "state_changing_labels": {"all": tab(sc), "held-out": tab(sc[sc.split == "held-out"]), "train": tab(sc[sc.split == "train"])} |
                                    {f"{w} / {side}": tab(sc[(sc.world == w) & (sc.split == side)]) for w in sorted(sc.world.unique()) for side in ("held-out", "train")},
           "state_changing_confident": int((sc.confident == "yes").sum()),
           "not_state_changing_labels": tab(M[M.state_changing != 1]),
           "tasks_with_sabotage": {s: int(sum(v["sabotage"] > 0 for v in d.values())) for s, d in per_task.items()},
           "state_changing_labels_per_task": per_task,
           "valid_votes_state_changing": {str(k): int(v) for k, v in sc.valid_votes.value_counts().sort_index().items()}}
    o = Path(out)
    with (o / f"sim_label_counts_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), "\n->", o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--votes", "--calls", "--out"):
        ap.add_argument(a, required=True)
    main(**vars(ap.parse_args()))
