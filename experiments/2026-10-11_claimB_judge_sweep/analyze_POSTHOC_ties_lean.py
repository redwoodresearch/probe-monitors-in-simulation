"""POST HOC, descriptive ($0, numbers only; 0d's request, written before the test result): two extras beside plan B's
registered AUROC verdict, on the test runs' main scores for the picked set-up (analyze_sol_b.scores, unchanged).

1. Tied pairs: of all (positive run, negative run) pairs the AUROC compares, the share with equal scores, for fixed sim
   vs real e1 and for real e2 vs real e1.
2. The "real" lean for S1: the share of runs whose three answers all say "real" (score 0), and all say "simulated"
   (score 1), by arm.

  uv run --no-project --python 3.12 --with numpy --with scipy python \\
      experiments/2026-10-11_claimB_judge_sweep/analyze_POSTHOC_ties_lean.py <pick.json> <test calls.jsonl>
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
sys.path.insert(0, str(HERE))
import analyze_sol_b as A  # noqa: E402  (scores, arm names; the registered analysis, unchanged)


def tied(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else None


def main(pick_path, calls_path):
    setup = json.loads(Path(pick_path).read_text())["pick"]
    sel = json.loads(subprocess.run(["git", "show", "58ce4715:experiments/2026-10-11_claimB_judge/results/judge_run_selection.json"],
                                    capture_output=True, text=True, check=True, cwd=HERE).stdout)
    recs = [json.loads(line) for line in Path(calls_path).read_text().splitlines()]
    s = A.scores(recs, set(sel["cells"]), setup, "main")
    arm = {a: [s[(c, a)] for c in sel["cells"] if (c, a) in s] for a in (A.SIM, A.E1, A.E2)}
    out = {"label": "POST HOC descriptive: tied pairs and the 'real' lean on the test runs", "pick": setup,
           "runs": {a: len(v) for a, v in arm.items()},
           "tied_pair_share": {"sim_vs_e1": tied(arm[A.SIM], arm[A.E1]), "e2_vs_e1": tied(arm[A.E2], arm[A.E1])},
           "all_real_share": {a: float(np.mean(np.array(v) == 0.0)) for a, v in arm.items() if v},
           "all_simulated_share": {a: float(np.mean(np.array(v) == 1.0)) for a, v in arm.items() if v},
           "inputs": {Path(x).name: hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in (pick_path, calls_path)},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"ties_lean_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs", "git", "script_sha256")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
