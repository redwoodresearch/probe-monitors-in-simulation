"""Run one pass of run_evalaware.py on its DEPLOYED app, for when the workspace is at Modal's 100-ephemeral-app limit
(10-14 ~21:30 PT). Same code, inputs, bounds and result file as `modal run`:
  uv run --no-project --python 3.12 --with modal==1.6.1 modal deploy experiments/2026-10-14_q2_evalaware_step1/run_evalaware.py
  uv run --no-project --python 3.12 --with modal==1.6.1 python experiments/2026-10-14_q2_evalaware_step1/launch_deployed.py \\
      --which first --inputs <dir holding the pinned files>
Stop the deployed app by ID when both passes are done."""
import argparse
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_evalaware as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--which", required=True, choices=["first", "simh"])
ap.add_argument("--inputs", required=True)
a = ap.parse_args()
blob, meta = R.prepare(a.which, a.inputs)
call = modal.Function.from_name(R.APP, "orchestrate").spawn(a.which, blob)
print("function call", call.object_id, flush=True)
R.write_result(a.which, blob, meta, call.get(timeout=3000), f"deployed app, function call {call.object_id}")
