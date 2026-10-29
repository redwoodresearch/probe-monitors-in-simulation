"""N1 Part 1, stage 3: spawn run_fit.py's fit on its deployed app (a dropped laptop does not stop it) and collect it.

Fixed layer 24 and the task-level null (README amendments). The result file is written as run_fit.py's entrypoint
writes it.

  PYTHONPATH=src uv run modal deploy experiments/2026-10-10_eval_awareness_probe/run_fit.py
  PYTHONPATH=src uv run python experiments/2026-10-11_la_format_sim_probe/spawn_fit_la.py launch n1C_<stamp>
  PYTHONPATH=src uv run python experiments/2026-10-11_la_format_sim_probe/spawn_fit_la.py collect <call id>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
FIT = HERE.parent / "2026-10-10_eval_awareness_probe" / "run_fit.py"
sys.path.insert(0, str(FIT.parent))
import run_fit  # noqa: E402  (its constants, for the result file)

LAYER, NULL_UNIT = 24, "task"

if __name__ == "__main__":
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    launches = ROOT / ".run_cache" / "la_format_sim_probe"
    launches.mkdir(parents=True, exist_ok=True)  # before any spawn, so a failed local write can't orphan a detached job
    if sys.argv[1] == "launch":
        st = git("status", "--porcelain", "--untracked-files=no")
        if st.returncode or st.stdout.strip():
            raise SystemExit("commit first: the fit must come from committed code")
        run = sys.argv[2]
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        sha = hashlib.sha256(FIT.read_bytes()).hexdigest()
        call = modal.Function.from_name(run_fit.APP, "fit").spawn(run, f"fit_{stamp}", False, NULL_UNIT, LAYER, sha)
        rec = {"call": call.object_id, "run": run, "stamp": stamp, "git": git("rev-parse", "HEAD").stdout.strip(), "run_fit_sha256": sha,
               "volume": run_fit.VOL}
        (launches / f"fit_launch_{call.object_id}.json").write_text(json.dumps(rec))
        print(json.dumps(rec, indent=1))
    elif sys.argv[1] == "collect":
        rec = json.loads((launches / f"fit_launch_{sys.argv[2]}.json").read_text())
        if hashlib.sha256(FIT.read_bytes()).hexdigest() != rec["run_fit_sha256"] or rec.get("volume", run_fit.VOL) != run_fit.VOL:
            raise ValueError("run_fit.py changed since the launch: its constants (the volume among them) would misrecord the fit")
        out = modal.FunctionCall.from_id(sys.argv[2]).get()
        st = out["settings"]  # as the worker ran them, not as declared here
        if (st["layer"], st["null_unit"], st["run_fit_sha256"]) != (LAYER, NULL_UNIT, rec["run_fit_sha256"]):
            raise ValueError(f"the fit ran other settings or code: {st}")
        out = {"run": rec["run"], "volume": rec.get("volume", run_fit.VOL), "shuffles": st["shuffles"], "seed": st["seed"], "C": st["C"],
               "max_iter": st["max_iter"], "grad_tol": st["grad_tol"], "null_unit": st["null_unit"], "fixed_layer": st["layer"],
               "launch": rec, **out}
        path = FIT.parent / "results" / f"fit_{rec['stamp']}_{rec['run']}.json"  # the run in the name: two same-second fits can't collide
        if path.exists():
            raise FileExistsError(f"{path} exists")
        tmp = path.with_name("." + path.name + ".tmp")
        tmp.write_text(json.dumps(out, indent=1) + "\n")
        tmp.rename(path)
        print(json.dumps({k: out[k] for k in ("summary", "max_grad (all fits)")}, indent=1), f"\n-> {path}")
