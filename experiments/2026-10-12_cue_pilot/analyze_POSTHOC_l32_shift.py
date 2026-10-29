"""POST HOC, descriptive only (not part of PLAN.md's rule; written after the screen): how far does the frozen L32 run
score move with the cue? Per test task and placement, cue minus neutral L32 run score; beside, the SD of the calibration
tasks' untouched L32 run scores. Numbers only, on the laptop ($0).

  uv run --no-project --with numpy python experiments/2026-10-12_cue_pilot/analyze_POSTHOC_l32_shift.py results/full_<stamp>.json
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
st = git("status", "--porcelain", "--untracked-files=no")
if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
    raise SystemExit("commit first: the result must record the commit of the code that made it")
res = json.loads((HERE / sys.argv[1]).read_text())
raw = (ROOT / ".run_cache" / "cue_pilot" / res["rows"]["file"]).read_bytes()
if hashlib.sha256(raw).hexdigest() != res["rows"]["sha256"]:
    raise ValueError("the rows file is not the collected one")
S = {(r["split"], r["task"], r["version"]): r for r in json.loads(raw)}
test = sorted({t for s, t, _ in S if s == "test"})
cal = sorted({t for s, t, _ in S if s == "cal"})
out = {"git": git("rev-parse", "HEAD").stdout.strip(), "input": sys.argv[1], "posthoc": True,
       "calibration untouched L32 SD": float(np.std([S[("cal", t, "untouched")]["l32"] for t in cal]))}
for place in ("far", "near"):
    d = np.array([S[("test", t, f"cue_{place}")]["l32"] - S[("test", t, f"neutral_{place}")]["l32"] for t in test])
    out[place] = {"cue minus neutral, mean": float(d.mean()), "median": float(np.median(d)), "cue higher": int((d > 0).sum()), "tasks": len(d)}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"l32_shift_POSTHOC_{stamp}.json"
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1), f"\n-> {path}")
