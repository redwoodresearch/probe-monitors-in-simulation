"""K4: within a paired cell where one run attempted the side task and the other declined, does the off-policy
eval-awareness probe score the declining run higher? Predictive evidence only, not a necessity test.

Run score = mean of the probe's call_end scores (Q3 off-policy probes, scores_eval_awareness_segment_end). For e1 vs
sim and, as the yardstick, e1 vs e2: over cells where the two runs disagree (attempted vs declined), the share where
the declining run has the higher awareness score (0.5 = no relation), with 1,000 main-task resamples.

  python analyze_awareness_vs_attempt.py --scores <scores_eval_awareness_segment_end_*.parquet> --labels <attempt labels>
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
from probe_shift.attack_eval import E1, E2, SIM, boot_weights, summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()
    s = pd.read_parquet(args.scores)
    s = s[(s.kind == "call_end") & s.arm.isin([E1, E2, SIM])]
    layers = [c for c in s.columns if c.startswith("L")]
    runs = s.groupby(["arm", "key", "cell", "main_task"], as_index=False)[layers].mean()
    lab = {(x["arm"], x["key"]): x["label"] for x in map(json.loads, Path(args.labels).read_text().splitlines())}
    runs["label"] = [lab.get((a, k)) for a, k in zip(runs.arm, runs.key)]
    runs = runs[runs.label.isin(["attempted", "declined"])]
    out = {"inputs": {p: {"file": f, "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest()}
                      for p, f in (("scores", args.scores), ("labels", args.labels))}, "results": {}}
    for other in (SIM, E2):
        a = runs[runs.arm == E1].set_index("cell"); b = runs[runs.arm == other].set_index("cell")
        cells = a.index.intersection(b.index)
        disc = [c for c in cells if a.loc[c, "label"] != b.loc[c, "label"]]
        task = np.array([a.loc[c, "main_task"] for c in disc])
        W = boot_weights(task, 1000, 0)
        for L in layers:
            # 1 if the run that declined has the higher awareness score
            dec_higher = np.array([float((a.loc[c, L] > b.loc[c, L]) == (a.loc[c, "label"] == "declined")) for c in disc])
            v = (W * dec_higher).sum(1) / W.sum(1)
            out["results"][f"e1_vs_{other}|{L}"] = {"discordant_cells": len(disc), "declining_run_higher": summary(v)}
            r = out["results"][f"e1_vs_{other}|{L}"]["declining_run_higher"]
            print(f"e1 vs {other:15s} {L}: {len(disc)} discordant cells; declining run scores higher in {r['value']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"awareness_vs_attempt_{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
