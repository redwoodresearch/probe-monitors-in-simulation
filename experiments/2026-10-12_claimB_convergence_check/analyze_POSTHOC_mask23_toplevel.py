"""POST HOC (RULE.md in this folder): mask23's top-level "primary" and "attenuation" (and "results"), unchanged re-run vs
the committed result (the gate) and masked vs unchanged. run_POSTHOC_mask23_checks.py's own top-level step stopped
because the archive's results folder already held the committed file; the two re-run outputs it left are read here.

  uv run --no-project --python 3.12 python experiments/2026-10-12_claimB_convergence_check/analyze_POSTHOC_mask23_toplevel.py \\
      <unchanged re-run output json> <masked re-run output json>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE)]
import rerun_POSTHOC_halves_masked as M  # noqa: E402  (diff)

SPEC = "c3793472:experiments/2026-10-12_claimB_halves_mask23/results/halves_mask23_2026-10-12_120522.json"
unm, msk = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
stored = json.loads(subprocess.run(["git", "show", SPEC], capture_output=True, check=True, cwd=HERE).stdout)
keys = ("primary", "attenuation", "results")
gate = {k: unm[k] == stored[k] for k in keys}
if not all(gate.values()):
    raise SystemExit(f"the unchanged re-run does not reproduce the committed result: {gate}")
out = {"label": "POST HOC: mask23 top-level fields, masked vs unchanged (RULE.md)", "source": SPEC, "gate_reproduced": gate,
       "differences_masked_vs_unmasked": {k: M.diff(unm[k], msk[k]) for k in keys},
       "inputs": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sys.argv[1:3]},
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
with open(HERE / "results" / f"mask23_toplevel_POSTHOC_{stamp}.json", "x") as fh:
    fh.write(json.dumps(out, indent=1) + "\n")
for k, v in out["differences_masked_vs_unmasked"].items():
    print(k, len(v), v)
