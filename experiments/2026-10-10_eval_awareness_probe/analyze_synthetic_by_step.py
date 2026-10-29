"""Post-hoc, after the gate ($0, saved outputs only): where the scored call sits in the synthetic dialogues, and how the
chosen probe scores the held-out pairs by that position.

On LinuxArena the probe's call scores fall steeply with call step (inserted_by_step_*.json). This reports:
- the final (scored) call's step in every split, i.e. which positions training ever saw;
- for the held-out pairs at the chosen readout and layer, the score by step and label, the AUROC within each step, and
  the paired change (implied-test side minus ordinary side), to set beside the inserted line's change on LinuxArena.
Only held-out dialogue scores are saved by run_fit.py, so train and dev get positions only.

  uv run --no-project --with pandas --with scikit-learn python experiments/2026-10-10_eval_awareness_probe/analyze_synthetic_by_step.py --fit results/fit_<stamp>.json --sets .run_cache/eval_awareness_probe/sets_<stamp>.jsonl
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--fit", required=True)
ap.add_argument("--sets", required=True)
a = ap.parse_args()
fit = json.loads((HERE / a.fit).read_text())
raw = (ROOT / a.sets).read_bytes()
if hashlib.sha256(raw).hexdigest() != fit["sets_sha256"]:
    raise ValueError(f"{a.sets} is not the pair file the probes were fit on")
items = {it["id"]: it for it in map(json.loads, raw.decode().splitlines())}
pos = pd.DataFrame([{"id": i, "split": it["split"], "pair": it["pair"], "label": it["label"], "mechanism": it["mechanism"],
                     "step": sum(m["role"] == "assistant" for m in it["messages"]) - 1,  # 0-based, as LinuxArena's steps
                     "chars": len(json.dumps(it["messages"]))} for i, it in items.items()])
r = fit["chosen_readout"]
sc = pd.DataFrame(fit["heldout_scores_at_chosen_layer"][r])[["id", "score"]]
h = pos.merge(sc, on="id", how="inner", validate="one_to_one")
if len(h) != (pos["split"] == "heldout").sum():
    raise ValueError("held-out scores do not cover every held-out dialogue")
pairs = h.pivot(index="pair", columns="label", values="score")
pairs["change"] = pairs[1] - pairs[0]
pairs = pairs.join(h[h["label"] == 1].set_index("pair")[["step", "mechanism"]])
out = {"fit": a.fit, "readout": r, "layer": fit["summary"][r]["layer"],
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip(),
       "final-call step counts by split": {s: {int(k): int(v) for k, v in g["step"].value_counts().sort_index().items()}
                                           for s, g in pos.groupby("split")},
       "dialogue characters by split (min/median/max)": {s: [int(g["chars"].min()), int(g["chars"].median()), int(g["chars"].max())]
                                                         for s, g in pos.groupby("split")},
       "held-out score median by step and label": {int(s): {int(lab): round(float(g2["score"].median()), 3) for lab, g2 in g.groupby("label")}
                                                   for s, g in h.groupby("step")},
       "held-out AUROC within step": {int(s): round(float(roc_auc_score(g["label"], g["score"])), 4) for s, g in h.groupby("step")},
       "held-out mechanisms by step": {int(s): sorted(g["mechanism"].unique()) for s, g in h.groupby("step")},
       "held-out paired change (implied-test minus ordinary)": {
           "pairs": len(pairs), "median": round(float(pairs["change"].median()), 3), "p10": round(float(pairs["change"].quantile(0.1)), 3),
           "fraction positive": round(float((pairs["change"] > 0).mean()), 4),
           "median by mechanism": {m: round(float(g["change"].median()), 3) for m, g in pairs.groupby("mechanism")}}}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"synthetic_by_step_{stamp}.json"
path.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
