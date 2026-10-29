"""Post-hoc ($0, saved rows only): the near retry's per-call change by how much context precedes the call.

If the probe's weak response on real runs came from long preceding context, calls early in a transcript (little
context before them) should respond more. Uses the original version's span start as the context length before each
call (the near version's adds the inserted lines), and reports, per bin, the paired per-token and last-token changes.

  uv run --no-project --with pandas python experiments/2026-10-10_eval_awareness_probe/analyze_near_by_position.py --near results/la_scores_near_<stamp>.json
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
BINS = ([-1, 300, 1000, 3000, 10000, 10**9], ["0-300", "301-1000", "1001-3000", "3001-10000", "10001+"])

ap = argparse.ArgumentParser()
ap.add_argument("--near", required=True)
a = ap.parse_args()
s = json.loads((HERE / a.near).read_text())
if s.get("placement") != "near":
    raise ValueError(f"{a.near} is not a near-placement run")
raw = (ROOT / ".run_cache" / "eval_awareness_probe" / s["rows"]).read_bytes()
if hashlib.sha256(raw).hexdigest() != s["rows_sha256"]:
    raise ValueError("the rows file does not match its recorded sha256")
df = pd.DataFrame(json.loads(raw))
key = ["key", "step", "action"]
orig = df[df["version"] == "original"].set_index(key)
ins = df[df["version"] == "inserted"].set_index(key)
if not (orig.index.is_unique and ins.index.is_unique and set(orig.index) == set(ins.index)):
    raise ValueError("the scored calls differ between versions or repeat")
ins = ins.loc[orig.index]
w = pd.DataFrame({"start": orig["span_start"], "cue": ins["cue_distance"],
                  "per_token": ins["score_per_token"] - orig["score_per_token"],
                  "last_token": ins["score_last_token"] - orig["score_last_token"]}).reset_index()
w["bin"] = pd.cut(w["start"], BINS[0], labels=BINS[1])
out = {"near": a.near, "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip(),
       "context before the call (original span start, tokens) quantiles 0/10/50/90": [int(q) for q in w["start"].quantile([0, 0.1, 0.5, 0.9])],
       "by context before the call": {b: {"calls": len(g), "runs": int(g["key"].nunique()), "median cue distance": float(g["cue"].median()),
                                          "per-token change median": round(float(g["per_token"].median()), 3),
                                          "last-token change median": round(float(g["last_token"].median()), 3)}
                                      for b, g in w.groupby("bin", observed=True)}}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"near_by_position_{stamp}.json"
path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
