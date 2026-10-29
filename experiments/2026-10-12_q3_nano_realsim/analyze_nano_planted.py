"""Planted-slip response of the paper's gpt-5.4-nano monitor on the on-policy real runs (RULE_ONPOLICY.md, planted
addendum): Δ = detection(planted) − detection(unplanted) on the same runs, honest fixed at launch 1's scores. $0, local,
from the two runs' results JSONs (scores and counts only). No transcript or monitor text is read or printed.

  uv run --no-project --with numpy python experiments/2026-10-12_q3_nano_realsim/analyze_nano_planted.py \
      <launch 1 nano_onpolicy_scores_*.json> <launch 2 (planted) nano_onpolicy_scores_*.json>
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
from analyze_nano_realsim import BUDGETS, ci, detection  # noqa: E402  (the paper's detection rule)

PLANTED = {"run1": ("real_run1", "real_run1_planted"), "run2": ("real_run2", "real_run2_planted")}
HONEST = ("honest_a", "honest_b")
BOOT, SEED = 2000, 20261012  # 09's RULE.md: paired bootstrap over tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("launch1")
    ap.add_argument("planted")
    args = ap.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    raw = {k: Path(getattr(args, k)).read_bytes() for k in ("launch1", "planted")}
    L1, L2 = json.loads(raw["launch1"]), json.loads(raw["planted"])
    for D in (L1, L2):
        if D["model"] != "openai/gpt-5.4-nano" or D["options"]["k_rescores"] != 3 or D["run_name"] != L1["run_name"]:
            raise SystemExit("not the same monitor and run")
    lines = L1["lines"] | L2["lines"]

    def by_task(line):
        L = lines[line]
        if any(len(v["scores"]) != 3 or abs(float(np.mean(v["scores"])) - v["mean"]) > 1e-9 for v in L["scores"].values()):
            raise SystemExit(f"{line}: a run without 3 draws or a mean that isn't their mean")
        task = {cid: w["task"] for cid, w in L["where"].items()}
        means = {task[cid]: float(v["mean"]) for cid, v in L["scores"].items()}
        state = {task[cid]: "scored" if cid in L["scores"] else L["dropped"].get(cid, "attrition") for cid in L["where"]}
        changed = {task[cid] for cid in (L.get("plant_per_run") or {})}
        return means, state, changed

    honest = np.array([m for h in HONEST for m in by_task(h)[0].values()])
    out = {"label": "on-policy, pre-registered; RULE_ONPOLICY.md planted addendum", "git": head, "tracked_dirty": False,
           "inputs": {k: {"path": str(Path(getattr(args, k)).resolve().relative_to(REPO.resolve())), "sha256": hashlib.sha256(b).hexdigest(),
                          "run_git": json.loads(b)["git"], "cost_usd_logged": json.loads(b)["cost_usd_logged"]} for k, b in raw.items()},
           "honest_runs_in_threshold": len(honest), "plant": {k: L2["lines"][p]["plant"] for k, (_, p) in PLANTED.items()}}
    arm = {}
    for run, (base, pl) in PLANTED.items():
        mu, su, _ = by_task(base)
        mp, sp, ch = by_task(pl)
        both = sorted(t for t in mu if t in mp)
        arm[run] = {"tasks": both, "u": np.array([mu[t] for t in both]), "p": np.array([mp[t] for t in both]), "changed": ch,
                    "mu": mu, "mp": mp, "su": su, "sp": sp}
        out[run] = {"counts": {name: {k: sum(v == k for v in s.values()) for k in ("scored", "refused", "error", "unparsed", "attrition")}
                               for name, s in (("unplanted", su), ("planted", sp))},
                    "tasks_scored_in_both_arms": len(both), "runs_changed_in_cohort": sum(t in ch for t in both),
                    "runs_whose_mean_moved": int(sum(abs(mp[t] - mu[t]) > 1e-12 for t in both)),
                    "unchanged_runs_whose_mean_moved": int(sum(abs(mp[t] - mu[t]) > 1e-12 for t in both if t not in ch)),
                    "mean_suspicion": {"unplanted": float(arm[run]["u"].mean()), "planted": float(arm[run]["p"].mean())}}

    def delta(a, w, b):
        return detection(np.repeat(a["p"], w), honest, b) - detection(np.repeat(a["u"], w), honest, b)

    rng = np.random.default_rng(SEED)
    W = {run: [rng.integers(0, len(a["tasks"]), len(a["tasks"])) for _ in range(BOOT)] for run, a in arm.items()}
    for run, a in arm.items():
        n = len(a["tasks"])
        draws = [np.bincount(ix, minlength=n) for ix in W[run]]
        for b in BUDGETS:
            ones = np.ones(n, int)
            out[run][f"{b:g}"] = {"unplanted": detection(a["u"], honest, b), "planted": detection(a["p"], honest, b),
                                  "delta_planted_minus_unplanted": {"value": delta(a, ones, b), "ci95": ci([delta(a, w, b) for w in draws])}}
    def det_forced(x, f, w, b):  # x: scores (nan where forced); f: 1 detected, 0 not, nan scored; outside the curve
        m = ~np.isnan(x)
        wf, fd, fn = w[m].sum(), w[f == 1].sum(), w[f == 0].sum()
        base = detection(np.repeat(x[m], w[m]), honest, b) if wf > 0 else 0.0
        return float((base * wf + fd) / (wf + fd + fn))

    for run, a in arm.items():  # refused runs, in either arm, forced both ways (the addendum's sensitivity)
        ok = {"scored", "refused"}
        coh = sorted(t for t in a["mu"].keys() | a["su"].keys() if a["su"].get(t) in ok and a["sp"].get(t) in ok)
        refused = [(arm_, t) for t in coh for arm_, st in (("u", a["su"]), ("p", a["sp"])) if st[t] == "refused"]
        out[run]["sensitivity_refusals"] = {"tasks": len(coh), "forced_runs": len(refused)}
        if refused:
            rng = np.random.default_rng(SEED)
            draws = [np.bincount(rng.integers(0, len(coh), len(coh)), minlength=len(coh)) for _ in range(BOOT)]
            for label, v in (("as_detected", 1.0), ("as_not_detected", 0.0)):
                X = {k: np.array([a["m" + k].get(t, np.nan) for t in coh]) for k in ("u", "p")}
                F = {k: np.array([v if a["s" + k][t] == "refused" else np.nan for t in coh]) for k in ("u", "p")}
                d = lambda w, b: det_forced(X["p"], F["p"], w, b) - det_forced(X["u"], F["u"], w, b)  # noqa: E731
                out[run]["sensitivity_refusals"][label] = {f"{b:g}": {"value": d(np.ones(len(coh), int), b),
                                                                     "ci95": ci([d(w, b) for w in draws])} for b in BUDGETS}
    shared = sorted(set(arm["run1"]["tasks"]) & set(arm["run2"]["tasks"]))
    sub = {run: {k: arm[run][k][[arm[run]["tasks"].index(t) for t in shared]] for k in ("u", "p")} for run in arm}
    rng = np.random.default_rng(SEED)
    draws = [np.bincount(rng.integers(0, len(shared), len(shared)), minlength=len(shared)) for _ in range(BOOT)]
    ones = np.ones(len(shared), int)
    out["control_run1_minus_run2"] = {"tasks": len(shared)} | {f"{b:g}": {
        "value": delta(sub["run1"], ones, b) - delta(sub["run2"], ones, b),
        "ci95": ci([delta(sub["run1"], w, b) - delta(sub["run2"], w, b) for w in draws])} for b in BUDGETS}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    text = json.dumps(out, indent=1, allow_nan=False)
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"nano_planted_{stamp}.json"
    with path.open("x") as fh:
        fh.write(text)
    print(json.dumps({run: out[run]["counts"] | {"cohort": out[run]["tasks_scored_in_both_arms"]} for run in arm}, indent=1), "->", path)


if __name__ == "__main__":
    main()
