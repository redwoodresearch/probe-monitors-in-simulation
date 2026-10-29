"""I2 analysis (PRICES.md item 2's rule, fixed before the result; EXPLORATORY: reads the test environments). Local and
free: stored per-run scores only. Inputs: the job's fetched result (data/skeleton_<stamp>.json) and step A's stored
per-output probe scores (the files Q1.1's probe-vs-TF-IDF check used: experiments/2026-10-09_fixed_sim/data/
claimB_output_scores_A_2026-10-10_022646.json.gz and claimB_output_scores_A_e2_vs_e1_2026-10-10_024204.json.gz).

Gate: the probe's run-level AUROC reproduces step A's published value (tpr_at_fpr_2026-10-10_235259.json) to 1e-9, and
the probe and skeleton score the same runs with the same arms and main tasks, or the script stops.
Rule (primary: fixed vs e1, level 1, char_wb): S = AUROC(probe) - AUROC(skeleton), paired main-task bootstrap (2,000
draws, default_rng(20261011)), 95% interval: "probe exceeds this skeleton baseline" if the lower end > 0.05; "this
baseline reaches the probe" if the upper end < 0.05; else "too few". e2 vs e1 is reported the same way as the
yardstick. Null tests: the job's one-sided p for the skeleton's own AUROC (fixed vs e1, e2 vs e1), Holm at 0.05.

  uv run --no-project --python 3.13 --with numpy --with scikit-learn python \\
      experiments/2026-10-11_claimB_yardsticks/analyze_skeleton.py <skeleton json> <fixed_sim dir with data/ and results/>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
SKEL, FIXED = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(FIXED))
import analyze_tpr_at_fpr as T  # noqa: E402  (load_gz, runs_from_outputs)

PROBE = {"fixed_vs_e1": FIXED / "data/claimB_output_scores_A_2026-10-10_022646.json.gz",
         "e2_vs_e1": FIXED / "data/claimB_output_scores_A_e2_vs_e1_2026-10-10_024204.json.gz"}
PUBLISHED = FIXED / "results/tpr_at_fpr_2026-10-10_235259.json"
POS = {"fixed_vs_e1": "sim-attack-fixed", "e2_vs_e1": "real-attack-e2"}
ALIAS = {"sim-attack": "sim-attack-fixed"}
TEST = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}
SEED, DRAWS, MARGIN = 20261011, 2000, 0.05


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sk = json.loads(SKEL.read_text())
    pub = {r["id"]: r for r in json.loads(PUBLISHED.read_text())["rows"]}
    out = {"label": "I2 skeleton baseline vs step A's probe (EXPLORATORY: test environments read many times); paired task bootstrap",
           "inputs": {str(p): sha(p) for p in [SKEL, *PROBE.values(), PUBLISHED]}, "seed": SEED, "draws": DRAWS,
           "margin": MARGIN, "rows": {}}
    for comp, path in PROBE.items():
        probe = {k: v for k, v in T.runs_from_outputs(T.load_gz(path)["levels"]["level1"]["rows"], np.mean).items()
                 if v["env"] in TEST}
        skel = {r["key"]: r for r in sk["rows"][f"{comp}|level1|char_wb"]["run_scores"]}
        if set(probe) != set(skel):
            raise SystemExit(f"{comp}: run sets differ ({len(set(probe) - set(skel))} probe-only, {len(set(skel) - set(probe))} skeleton-only)")
        keys = sorted(probe)
        if any(probe[k]["arm"] != ALIAS.get(skel[k]["arm"], skel[k]["arm"]) or probe[k]["task"] != skel[k]["main_task"] for k in keys):
            raise SystemExit(f"{comp}: arm or main task disagrees")
        y = np.array([probe[k]["arm"] == POS[comp] for k in keys])
        sp, ss = np.array([probe[k]["score"] for k in keys]), np.array([skel[k]["score"] for k in keys])
        a_p, a_s = roc_auc_score(y, sp), roc_auc_score(y, ss)
        ref = pub[f"step A|level1 · run mean|{comp}|test 5"]["auroc"]
        if abs(a_p - ref) > 1e-9:
            raise SystemExit(f"{comp}: probe AUROC {a_p} does not reproduce {ref}")
        if abs(a_s - sk["rows"][f"{comp}|level1|char_wb"]["auroc"]) > 1e-9:
            raise SystemExit(f"{comp}: skeleton AUROC does not match the job's")
        tasks, inv = np.unique([probe[k]["task"] for k in keys], return_inverse=True)
        members = [np.flatnonzero(inv == j) for j in range(len(tasks))]
        rng, diffs, skipped = np.random.default_rng(SEED), [], 0
        for _ in range(DRAWS):
            idx = np.concatenate([members[j] for j in rng.choice(len(tasks), len(tasks))])
            if y[idx].all() or not y[idx].any():
                skipped += 1
                continue
            diffs.append(roc_auc_score(y[idx], sp[idx]) - roc_auc_score(y[idx], ss[idx]))
        lo, hi = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))
        verdict = ("probe exceeds this skeleton baseline" if lo > MARGIN else
                   "this baseline reaches the probe" if hi < MARGIN else "too few")
        out["rows"][comp] = {"probe_auroc": float(a_p), "skeleton_auroc": float(a_s), "S": float(a_p - a_s),
                             "ci95": [lo, hi], "skipped_draws": skipped, "runs": len(keys), "main_tasks": len(tasks),
                             "reading": verdict if comp == "fixed_vs_e1" else f"(yardstick) {verdict}"}
    ps = {c: sk["nulls"][c]["p_upper"] for c in ("fixed_vs_e1", "e2_vs_e1")}
    order = sorted(ps, key=ps.get)
    holm, stop = {}, False
    for i, c in enumerate(order):
        rej = (not stop) and ps[c] <= 0.05 / (len(order) - i)
        stop = stop or not rej
        holm[c] = {"p": ps[c], "threshold": 0.05 / (len(order) - i), "above_null": bool(rej)}
    out |= {"holm_null_tests": holm, "planted": sk["planted"],
            "secondary_aurocs": {k: v["auroc"] for k, v in sk["rows"].items()},
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": sha(Path(__file__))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"skeleton_vs_probe_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("rows", "holm_null_tests", "planted", "secondary_aurocs")}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
