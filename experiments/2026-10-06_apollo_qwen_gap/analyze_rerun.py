"""Does a rerun of a model's sweep (run_qwen_gap.py::sweep --run-suffix rerun) reproduce the original?

  uv run python experiments/2026-10-06_apollo_qwen_gap/analyze_rerun.py --model qwen3-8b

Compares the latest results/sweep_<model>_rerun_*.json with the latest original results/sweep_<model>_2*.json: Roleplaying
AUROC under Apollo's labels for every probe, the held-out instructed-pair AUROC, and every per-dialogue score.
Writes results/rerun_check_<model>_<stamp>.json.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"
SET = "roleplaying_llama-70b-3.3"


def main(model: str):
    orig_path = sorted(p for p in RESULTS.glob(f"sweep_{model}_2*.json"))[-1]
    rerun_path = sorted(RESULTS.glob(f"sweep_{model}_rerun_*.json"))[-1]
    orig, rerun = (json.loads(p.read_text()) for p in (orig_path, rerun_path))
    assert orig["eval_sets"][SET]["ids"] == rerun["eval_sets"][SET]["ids"], "different Roleplaying dialogues"
    assert orig["val_ids"] == rerun["val_ids"], "different held-out instructed pairs"
    a = {(r["readout"], r["layer"]): r for r in orig["by_probe"]}
    b = {(r["readout"], r["layer"]): r for r in rerun["by_probe"]}
    assert a.keys() == b.keys()
    auroc_diff = max(abs(a[k][f"{SET}_auroc"] - b[k][f"{SET}_auroc"]) for k in a)
    val_diff = max(abs(a[k]["instructed_val_auroc"] - b[k]["instructed_val_auroc"]) for k in a)
    # A dialogue that loses (or gains) its detected tokens gets a null score; count those separately rather than letting
    # them drop out of the differences.
    score_diff, missing_mismatch = {}, {}
    for name in orig["scores"]:
        worst, mismatched = 0.0, 0
        for k in orig["scores"][name]:
            x, y = (np.array([np.nan if v is None else v for v in s[name][k]], float) for s in (orig["scores"], rerun["scores"]))
            mismatched += int((np.isnan(x) != np.isnan(y)).sum())
            ok = ~np.isnan(x) & ~np.isnan(y)
            worst = max(worst, float(np.abs(x[ok] - y[ok]).max()) if ok.any() else 0.0)
        score_diff[name], missing_mismatch[name] = worst, mismatched
    same_tokens = all(orig["eval_sets"][n]["n_detect"] == rerun["eval_sets"][n]["n_detect"] for n in orig["eval_sets"])
    best = max(a, key=lambda k: a[k][f"{SET}_auroc"] if k[0] == "apollo" else -1)
    out = {"model": model, "original": str(orig_path.relative_to(HERE)), "rerun": str(rerun_path.relative_to(HERE)),
           "original_commit": orig["config"]["git_commit"], "rerun_commit": rerun["config"]["git_commit"],
           "max_abs_roleplaying_auroc_diff": auroc_diff, "max_abs_instructed_val_auroc_diff": val_diff,
           "max_abs_score_diff": score_diff, "missing_score_mismatches": missing_mismatch, "same_detected_token_counts": same_tokens,
           "identical_aurocs": auroc_diff == 0 and val_diff == 0 and same_tokens and not any(missing_mismatch.values()),
           "best_final_norm": {"layer": best[1], "original": a[best][f"{SET}_auroc"], "rerun": b[best][f"{SET}_auroc"]}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(RESULTS / f"rerun_check_{model}_{stamp}.json", "x") as f:  # results/ is append-only
        f.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    main(p.parse_args().model)
