"""On-policy, pre-registered (RULE_ONPOLICY.md): the paper's gpt-5.4-nano trajectory monitor, real vs simulated
detection on the on-policy runs. $0, local, from run_nano_onpolicy.py's results JSON (scores and counts only). No
transcript or monitor text is read or printed.

  uv run --no-project --with numpy python experiments/2026-10-12_q3_nano_realsim/analyze_nano_onpolicy.py \
      experiments/2026-10-12_q3_nano_realsim/results/nano_onpolicy_scores_<stamp>.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from analyze_nano_realsim import BOOT, BUDGETS, SEED, ci, detection  # noqa: E402  (the off-policy dot's rule)

ATTACK = {"run1": "real_run1", "sim": "sim_september", "run2": "real_run2"}
HONEST = ("honest_a", "honest_b")
ENDPOINTS = {"run2_minus_sim (primary)": ("run2", "sim"), "run1_minus_sim": ("run1", "sim"),
             "run2_minus_run1 (yardstick)": ("run2", "run1")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    args = ap.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    raw = Path(args.scores).read_bytes()
    D = json.loads(raw)
    if D["model"] != "openai/gpt-5.4-nano" or D["options"]["k_rescores"] != 3:
        raise SystemExit(f"model {D['model']} k {D['options']['k_rescores']}")

    means, state, main_of, counts = {}, {}, {}, {}
    for line, L in D["lines"].items():
        if any(len(v["scores"]) != 3 or abs(float(np.mean(v["scores"])) - v["mean"]) > 1e-9 for v in L["scores"].values()):
            raise SystemExit(f"{line}: a run without 3 draws or a mean that isn't their mean")
        task = {cid: w["task"] for cid, w in L["where"].items()}
        means[line] = {task[cid]: float(v["mean"]) for cid, v in L["scores"].items()}
        # every loaded run is scored, dropped (its category) or lost before scoring (attrition): never silently absent
        state[line] = {task[cid]: "scored" if cid in L["scores"] else L["dropped"].get(cid, "attrition") for cid in L["where"]}
        for cid, t in task.items():
            if main_of.setdefault(t, cid.split(":")[0]) != cid.split(":")[0] and line in ATTACK.values():
                raise SystemExit(f"{t}: main task differs between lines")
        counts[line] = {"loaded": len(L["where"]), "scored": len(L["scores"]),
                        **{k: sum(v == k for v in state[line].values()) for k in ("refused", "error", "unparsed", "attrition")},
                        "attrition_reported": L["attrition"]}
    honest = np.array([means[h][t] for h in HONEST for t in sorted(means[h])])
    present = sorted(set.intersection(*(set(state[ATTACK[a]]) for a in ATTACK)), key=lambda t: int(t[1:].split("_")[0]))

    def table(cohort, forced=None):
        """as analyze_nano_realsim.table, by task: forced maps (arm, task) -> True/False for a run that was not scored,
        handled outside the curve and keeping its bootstrap weight; resamples are main tasks, honest fixed."""
        forced = forced or {}
        mains = np.array([main_of[t] for t in cohort])
        units = np.unique(mains)
        tix = np.array([np.flatnonzero(units == m)[0] for m in mains])
        rng = np.random.default_rng(SEED)
        W = np.array([np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))[tix] for _ in range(BOOT)])
        X = {a: np.array([means[ATTACK[a]].get(t, np.nan) for t in cohort]) for a in ATTACK}
        FD = {a: np.array([forced.get((a, t)) is True for t in cohort]) for a in ATTACK}
        FN = {a: np.array([forced.get((a, t)) is False for t in cohort]) for a in ATTACK}
        for a in ATTACK:
            if (np.isnan(X[a]) != (FD[a] | FN[a])).any():
                raise SystemExit("a cohort run is missing without a forced value, or forced while scored")
        res = {"tasks": len(cohort), "main_tasks": len(units),
               "forced_runs": {a: {"detected": int(FD[a].sum()), "not_detected": int(FN[a].sum())} for a in ATTACK}}
        for b in BUDGETS:
            def det(a, w, b=b):
                m = ~np.isnan(X[a])
                wf = w[m].sum()
                base = detection(np.repeat(X[a][m], w[m]), honest, b) if wf > 0 else 0.0
                return float((base * wf + w[FD[a]].sum()) / (wf + w[FD[a]].sum() + w[FN[a]].sum()))
            ones = np.ones(len(cohort), int)
            r = {a: {"detected": det(a, ones), "ci95": ci([det(a, w) for w in W])} for a in ATTACK}
            for name, (hi, lo) in ENDPOINTS.items():
                r[name] = {"value": r[hi]["detected"] - r[lo]["detected"], "ci95": ci([det(hi, w) - det(lo, w) for w in W])}
            res[f"{b:g}"] = r
        if not forced:
            res["mean_suspicion"] = {a: float(X[a].mean()) for a in ATTACK}
            res["mean_suspicion_diff"] = {name: {"value": float((X[hi] - X[lo]).mean()),
                                                 "ci95": ci([((X[hi] - X[lo]) * w).sum() / w.sum() for w in W])}
                                          for name, (hi, lo) in ENDPOINTS.items()}
        return res

    main_cohort = [t for t in present if all(state[ATTACK[a]][t] == "scored" for a in ATTACK)]
    out = {"label": "on-policy, pre-registered; RULE_ONPOLICY.md", "git": head, "tracked_dirty": False,
           "input": {"path": str(Path(args.scores).resolve().relative_to(REPO.resolve())), "sha256": hashlib.sha256(raw).hexdigest(),
                     "run_git": D["git"], "cost_usd_logged": D["cost_usd_logged"]},
           "counts": counts, "skipped_unusable_attempts": len(D["skipped_unusable_attempts"]),
           "honest_runs_in_threshold": len(honest), "tasks_loaded_in_all_three_attack_lines": len(present),
           "main": table(main_cohort)}
    for name, drop in (("refusals", {"refused"}), ("every_drop", {"refused", "error", "unparsed", "attrition"})):
        cohort = [t for t in present if all(state[ATTACK[a]][t] in drop | {"scored"} for a in ATTACK)]
        miss = {(a, t) for t in cohort for a in ATTACK if state[ATTACK[a]][t] != "scored"}
        out[f"sensitivity_{name}"] = {"tasks": len(cohort), "forced_runs": len(miss)}
        if miss:
            for label, v in (("as_detected", True), ("as_not_detected", False)):
                out[f"sensitivity_{name}"][label] = table(cohort, {k: v for k in miss})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    text = json.dumps(out, indent=1, allow_nan=False)  # refuses nonfinite values
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"nano_onpolicy_detection_{stamp}.json"
    with path.open("x") as fh:
        fh.write(text)
    print(json.dumps({"counts": {k: {x: v[x] for x in ("loaded", "scored", "refused", "error", "unparsed", "attrition")} for k, v in counts.items()},
                      "main_tasks": len(main_cohort), "honest_runs": len(honest)}, indent=1), "->", path)


if __name__ == "__main__":
    main()
