"""POST HOC, RULE.md amendment 1 (after the method gate failed): does the plant-2 direction transfer to honest replies at
its training position? On the 30 gutter pairs: the plant-2 score (direction of the other half) at output_end, planted
minus twin; counts (>, <, ties), a one-sided sign test on the non-tied pairs, the mean and median difference.
"transfers" iff planted > twin in a majority of the non-tied pairs and p <= 0.05; else "no transfer". $0, local, from the
job's saved scores (claimB/token_attr_v1/145c32c05f05); nothing else is read.

  uv run --no-project --python 3.12 --with numpy --with scipy --with 'modal==1.6.1' python \\
      experiments/2026-10-13_claimB_token_attribution/analyze_POSTHOC_transfer.py
"""

import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
OUT = "claimB/token_attr_v1/145c32c05f05"


def main():
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    raw = b"".join(vol.read_file(f"{OUT}/scores.npz"))
    z = np.load(io.BytesIO(raw))
    staged = HERE / "data/staged.jsonl"
    items = [json.loads(x) for x in staged.read_text().splitlines()]
    metas = json.loads(str(z["meta"]))
    meta = {m["item"]: m for m in metas}
    c = metas[0]["columns"].index("plant2|v")
    S = {m["item"]: z["scores"][st: st + ln, c].astype(np.float64) for m, st, ln in zip(metas, z["starts"], z["lens"])}
    end = lambda i: S[i][meta[i]["reply"][1] - 1 - meta[i]["w0"]]  # noqa: E731  (output_end: the reply's last content token)
    twin = {(it["run"], it["msg"]): it["item"] for it in items if it["kind"] == "twin" and it["slip"] == "gutter"}
    d = np.array([end(it["item"]) - end(twin[(it["run"], it["msg"])]) for it in items if it["kind"] == "plant" and it["slip"] == "gutter"])
    gt, lt, ties = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    p = float(binomtest(gt, gt + lt, alternative="greater").pvalue) if gt + lt else 1.0  # all tied: no transfer
    verdict = "transfers" if gt > lt and p <= 0.05 else "no transfer"
    out = {"label": "POST HOC amendment 1: plant-2 transfer at output_end on the 30 gutter pairs (preliminary until reviewed)",
           "pairs": len(d), "planted_gt_twin": gt, "planted_lt_twin": lt, "ties": ties, "sign_test_p_one_sided": p,
           "mean_diff": float(d.mean()), "median_diff": float(np.median(d)), "verdict": verdict,
           "closing": "per-token push doesn't localize tells" if verdict == "transfers" else "the gate couldn't test the method (no transfer to honest replies)",
           "inputs": {"scores_npz_sha256": hashlib.sha256(raw).hexdigest(), "staged_sha256": hashlib.sha256(staged.read_bytes()).hexdigest()},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"POSTHOC_transfer_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs",)}, indent=1))


if __name__ == "__main__":
    main()
