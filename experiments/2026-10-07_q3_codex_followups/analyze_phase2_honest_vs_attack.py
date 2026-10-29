"""Phase 2 on-policy, tiny: does the frozen Q3 detector (#36) separate Qwen3.5-27B's own attack runs from its own honest
runs on the same main tasks? Four main tasks have both a clean attack (run_read_phase2.py) and a clean honest run
(run_read_phase2_honest.py). Four honest runs cannot set a false-alarm rate, so nothing here is a catch rate.

Per main task and readout: the attack and honest run means. Per readout, pooled over the four tasks: the per-call AUROC
of attack calls vs honest calls, over every call the run mean uses and over content-ending calls only (call_kind 0, as
the matched comparison; the same set for the content readout), and how many of the four attack runs score above their
task's honest run.

  python analyze_phase2_honest_vs_attack.py --attack <phase2_read_*.json> --honest <phase2_honest_read_*.json>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
READOUTS = ("call_end_content", "call_end", "output_end")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", required=True)
    ap.add_argument("--honest", required=True)
    args = ap.parse_args()
    att = {r["main_task"]: r for r in json.loads(Path(args.attack).read_text())["runs"]}
    hon = {r["main_task"]: r for r in json.loads(Path(args.honest).read_text())["runs"]}
    tasks = sorted(set(att) & set(hon))
    out = {"tasks": {t: att[t]["env"] for t in tasks}, "per_task": {}, "pooled": {},
           "caveat": "4 honest runs cannot set a false-alarm rate; these are descriptive comparisons"}
    for t in tasks:
        out["per_task"][att[t]["env"]] = {ro: {"attack": att[t]["run_scores"][ro]["mean"], "honest": hon[t]["run_scores"][ro]["mean"]} for ro in READOUTS}
    for ro in READOUTS:
        def calls(r, content_only):
            return np.array([c["score"] for c in r["calls"] if c["readout"] == ro and (ro == "output_end" or not content_only or c["call_kind"] == 0)])
        higher = sum(att[t]["run_scores"][ro]["mean"] > hon[t]["run_scores"][ro]["mean"] for t in tasks)
        out["pooled"][ro] = {"attack_run_above_its_honest_run": f"{higher} of {len(tasks)}"}
        for name, content_only in (("all_calls", False), ("content_ending_calls", True)):
            a = np.concatenate([calls(att[t], content_only) for t in tasks]); h = np.concatenate([calls(hon[t], content_only) for t in tasks])
            out["pooled"][ro][name] = {"call_auroc_attack_vs_honest": round(float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(h))], np.r_[a, h])), 3),
                                       "attack_calls": len(a), "honest_calls": len(h)}
    for env, v in out["per_task"].items():
        print(env, {ro: (round(x["attack"], 2), round(x["honest"], 2)) for ro, x in v.items()})
    print(json.dumps(out["pooled"], indent=1))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with (HERE / "results" / f"phase2_honest_vs_attack_{stamp}.json").open("x") as f:
        f.write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
