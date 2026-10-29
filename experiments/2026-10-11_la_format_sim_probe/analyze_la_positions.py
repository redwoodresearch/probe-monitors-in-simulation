"""N1 Part 1, post hoc ($0, stored scores): the comparison's AUROCs by where the cue sits (README section 1).

Added after the result: each held-out, phrasing and attack-panel set split by cue position (system prompt, task
message, last tool output before the read call), for N1's and the concept probe's content and turn-start probes. Each
slice's chance level comes from a task-grouped label-flip null on the stored scores (the probes are fixed): every pair
of a task swaps its two labels together, 2,000 draws, seed 20261012; the null's 95th percentile and a one-sided p.

  uv run --no-project --with scikit-learn python experiments/2026-10-11_la_format_sim_probe/analyze_la_positions.py results/compare_scores_<stamp>.json
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
st = git("status", "--porcelain", "--untracked-files=no")
if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
    raise SystemExit("commit first: the result must record the commit of the code that made it")
raw = (HERE / sys.argv[1]).read_bytes()
rows = json.loads(raw)["rows"]
out = {"post hoc": "added after the stage 4 result", "scores": sys.argv[1], "scores_sha256": hashlib.sha256(raw).hexdigest(),
       "git": git("rev-parse", "HEAD").stdout.strip(), "null": "task-grouped label flip, 2000 draws, seed 20261012"}
rng = np.random.default_rng(20261012)
auc = lambda s, lab: ((lab * rankdata(s)).sum(-1) - lab.sum(-1) * (lab.sum(-1) + 1) / 2) / (lab.sum(-1) * (lab.shape[-1] - lab.sum(-1)))
for split in ("heldout", "heldout_phrasing", "panel", "panel_phrasing"):
    for pos in ("system", "task", "tool"):
        sel = [r for r in rows if r["split"] == split and r["position"] == pos]
        if sel:
            y, tasks = np.array([r["label"] for r in sel]), sorted({r["task"] for r in sel})
            yb = y ^ rng.integers(0, 2, (2000, len(tasks)))[:, [tasks.index(r["task"]) for r in sel]]  # a task's pairs flip together
            res = {"pairs": len(sel) // 2, "tasks": len(tasks)}
            for k in ("N1_C", "concept_C", "N1_T", "concept_T"):
                s = np.array([r[k] for r in sel])
                a, null = auc(s, y), auc(s, yb)  # the same formula on both sides, so tied draws count as ties
                assert abs(roc_auc_score(y, s) - a) < 1e-9
                res |= {f"AUROC {k}": round(a, 4), f"null p95 {k}": round(float(np.quantile(null, 0.95)), 4),
                        f"p {k}": round(float((1 + (null >= a).sum()) / 2001), 4)}
            out[f"{split} / {pos}"] = res
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"positions_{stamp}.json"
if path.exists():
    raise FileExistsError(path)
path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
print(json.dumps(out, indent=1), f"\n-> {path}")
