"""Post-hoc, after the gate ($0, saved rows only): how much the inserted line moves each call's score, by call step.

Bar (b) failed. This asks whether the line's effect fades with distance from the system prompt (big on early calls,
small later) or is small everywhere. Pairs each scored call's original and inserted per-token scores (same run, step
and action), and reports per step bin the median original score and the inserted-minus-original change.

  uv run --no-project --with pandas python experiments/2026-10-10_eval_awareness_probe/analyze_inserted_by_step.py --la results/la_scores_<stamp>.json
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--la", required=True)
a = ap.parse_args()
la = json.loads((HERE / a.la).read_text())
raw = (ROOT / ".run_cache" / "eval_awareness_probe" / la["rows"]).read_bytes()
if hashlib.sha256(raw).hexdigest() != la["rows_sha256"]:
    raise ValueError("the rows file does not match its recorded sha256")
df = pd.DataFrame(json.loads(raw))
out = {"la": a.la, "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}
for col in ("score_per_token", "score_last_token"):
    w = df.pivot(index=["key", "step", "action"], columns="version", values=col)  # raises on a duplicate call row
    unpaired = int(w.isna().any(axis=1).sum())
    w = w.dropna().reset_index()
    w["change"] = w["inserted"] - w["original"]
    w["step bin"] = pd.cut(w["step"], [-1, 0, 2, 5, 10, 20, 10_000], labels=["0", "1-2", "3-5", "6-10", "11-20", "21+"])
    g = w.groupby("step bin", observed=True).agg(calls=("change", "size"), original_median=("original", "median"),
                                                  change_median=("change", "median"), change_p90=("change", lambda s: s.quantile(0.9)))
    out[col] = {"paired calls": len(w), "unpaired calls": unpaired, "original score sd": round(float(w["original"].std()), 3),
                "change median": round(float(w["change"].median()), 3), "by step bin": g.round(3).to_dict(orient="index")}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"inserted_by_step_{stamp}.json"
path.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
