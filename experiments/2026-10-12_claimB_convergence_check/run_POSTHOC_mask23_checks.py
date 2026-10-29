"""POST HOC (RULE.md in this folder): the same two checks for the mask23 result, after it was committed
(experiments/2026-10-12_claimB_halves_mask23/results/halves_mask23_2026-10-12_120522.json @ c3793472).
(1) analyze_POSTHOC_convergence.check on mask23's six refit arrays (local copies, sha256 against that result's inputs);
(2) rerun_POSTHOC_halves_masked.main with SUMMARIES set to mask23's analyzer and result at c3793472 (unchanged re-run as
the gate, then the masked selection line). Both modules are imported unchanged from this folder.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn --with scipy \\
      --with matplotlib python experiments/2026-10-12_claimB_convergence_check/run_POSTHOC_mask23_checks.py \\
      --arrays <dir with {dev_to_test,test_to_dev}/{fixed_vs_e1,e2_vs_e1,plant2}.npz> --tree <scratch dir>
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE)]
import analyze_POSTHOC_convergence as C  # noqa: E402
import rerun_POSTHOC_halves_masked as M  # noqa: E402

COMMIT = "c3793472"
RESULT = "experiments/2026-10-12_claimB_halves_mask23/results/halves_mask23_2026-10-12_120522.json"
ANALYZER = "experiments/2026-10-12_claimB_halves_mask23/analyze_mask23.py"
PREFIX = "claimB/halves_mask23_v1/refit_e079fda0_ac58acddbb9f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--tree", type=Path, required=True)
    a = ap.parse_args()
    inputs = C.git_json(f"{COMMIT}:{RESULT}")["inputs"]
    conv = {d: {c: C.check(a.arrays / d / f"{c}.npz", inputs[f"{PREFIX}/{d}/{c}.npz"]) for c in ("fixed_vs_e1", "e2_vs_e1", "plant2")}
            for d in C.DIRS}
    out = {"label": "POST HOC convergence check, mask23 (RULE.md)", "source": f"{RESULT}@{COMMIT}", "convergence": conv,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"convergence_mask23_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    for d, comps in conv.items():
        for c, r in comps.items():
            c0, sh = r["col0"], r["shuffles"]
            print(f"mask23/{d} {c:12s} col0 {c0['selected']} conv={c0['converged']} ratio={c0['ratio_to_tol']:.3g} masked_differs={c0['masked_differs']}"
                  f" | shuffles sel_unconv {sh['selected_unconverged']}/{sh['of']} masked_differs {sh['masked_selection_differs']} none_conv {sh['no_converged_entry']}")
    M.SUMMARIES = {"halves_mask23": (COMMIT, ANALYZER, RESULT)}
    res_dir = a.tree.resolve() / COMMIT / Path(ANALYZER).parent / "results"
    before = set(res_dir.glob("*.json")) if res_dir.exists() else set()
    sys.argv = [sys.argv[0], "--tree", str(a.tree)]
    M.main()
    # mask23's result also has top-level "primary" and "attenuation": gate and diff them here (M compares "results" only)
    new = sorted(set(res_dir.glob("*.json")) - before)  # the unchanged run's file, then the masked run's (stamped in order)
    if len(new) != 2:
        raise SystemExit(f"expected two new analyzer outputs, found {len(new)}")
    unm, msk = (json.loads(f.read_text()) for f in new)
    stored = C.git_json(f"{COMMIT}:{RESULT}")
    keys = ("primary", "attenuation", "results")
    if any(unm[k] != stored[k] for k in keys):
        raise SystemExit("the unchanged re-run does not reproduce primary/attenuation/results")
    extra = {"gate_reproduced_primary_attenuation_results": True,
             "differences_masked_vs_unmasked": {k: M.diff(unm[k], msk[k]) for k in keys}, "outputs": [f.name for f in new]}
    with open(HERE / "results" / f"mask23_masked_toplevel_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(extra, indent=1) + "\n")
    print("top-level differences:", {k: len(v) for k, v in extra["differences_masked_vs_unmasked"].items()})


if __name__ == "__main__":
    main()
