"""The attack detector's supporting numbers, from saved files, so every number in its notes comes from a script. $0,
local (Python 3.14 reads the .eval logs for the errored runs). Layer 51, tool-output readout.
- swap diagnostic (5 tasks): each attack run's score with its own and with honest_b's system prompt, honest_b's score,
  the mean attack - honest_b separation both ways, and the 5 vs 5 AUROC both ways;
- last 27 outputs: AUROC real and September;
- relative-position deciles: the mean per-output score per decile (floor(10 * position / run length)), attack vs honest_b;
- scored outputs per run: median per arm; off-policy runs per arm (from the outer-fold scores);
- Sol's training labels: the share of real attack e1 runs labelled declined (the nano monitor's attempt labels, on main).

  uv run --no-project --python 3.14 --with numpy --with scikit-learn python analyze_attack_detector_extras.py \
      --runs results/attack_detector_onpolicy_<stamp>.json --per-output results/attack_detector_per_output_<stamp>.json \
      --swap results/attack_detector_swap_<stamp>.json --labels <attack_attempt_labels jsonl> --run-dir <rerun folder>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

from analyze_attack_detector import errored_runs

HERE = Path(__file__).parent
CELL = "L51_output_end"


def main(a):
    bad = set(errored_runs(Path(a.run_dir).expanduser()))
    R, P, W = (json.loads(Path(p).read_text()) for p in (a.runs, a.per_output, a.swap))
    c = R["cells"][CELL]
    own = dict(zip(c["on_runs"], c["on_score"]))
    sw = dict(zip(W["cells"][CELL]["runs"], W["cells"][CELL]["score"]))
    tasks = sorted(r.split("|")[1] for r in sw)
    att, swp, hon = (np.array([d[f"{p}|{t}"] for t in tasks]) for d, p in ((own, "attack"), (sw, "attack"), (own, "honest_b")))
    y = np.r_[np.ones(len(tasks)), np.zeros(len(tasks))]
    out = {"swap": {"tasks": tasks, "attack_own_prompt": att.round(4).tolist(), "attack_honest_prompt": swp.round(4).tolist(),
                    "honest_b": hon.round(4).tolist(), "separation_own": round(float(att.mean() - hon.mean()), 4),
                    "separation_honest_prompt": round(float(swp.mean() - hon.mean()), 4),
                    "auroc_own": round(float(roc_auc_score(y, np.r_[att, hon])), 4), "auroc_honest_prompt": round(float(roc_auc_score(y, np.r_[swp, hon])), 4),
                    "every_swapped_attack_higher": bool((swp > att).all())}}
    seq = {}
    for r, s in zip(P["rows"], P["scores"][CELL]):
        if r["kind"] == "output_end" and s is not None:
            seq.setdefault(f"{r['arm']}|{r['task']}", []).append((r["turn"], r["call_index"], s))
    seq = {k: [x[2] for x in sorted(v)] for k, v in seq.items() if k not in bad}

    def auc(a_arm, h_arm, f):
        ks = [k for k in seq if k.split("|")[0] in (a_arm, h_arm)]
        return round(float(roc_auc_score([k.split("|")[0] == a_arm for k in ks], [f(seq[k]) for k in ks])), 4)
    out["last_27"] = {"real": auc("attack", "honest_b", lambda v: np.mean(v[-27:])),
                      "september": auc("sim_attack@september", "sim_honest@september", lambda v: np.mean(v[-27:]))}
    dec = {}
    for arm in ("attack", "honest_b"):
        acc = [[] for _ in range(10)]
        for k, v in seq.items():
            if k.split("|")[0] == arm:
                for i, s in enumerate(v):
                    acc[min(9, int(10 * i / len(v)))].append(s)
        dec[arm] = [round(float(np.mean(x)), 2) for x in acc]
    out["position_deciles_mean_score"] = dec
    out["scored_outputs_median"] = {arm: float(np.median([len(v) for k, v in seq.items() if k.split("|")[0] == arm]))
                                    for arm in sorted({k.split("|")[0] for k in seq})}
    out["off_policy_runs"] = {arm: len(v["score"]) for arm, v in c["off_outer"].items()}
    lab = [json.loads(x) for x in Path(a.labels).read_text().splitlines()]
    e1 = [r for r in lab if r["arm"] == "real-attack-e1"]
    out["sol_real_attack_e1_labels"] = {"runs": len(e1), "declined_share": round(sum(r["label"] == "declined" for r in e1) / len(e1), 3)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"attack_detector_extras_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--runs", "--per-output", "--swap", "--labels", "--run-dir"):
        ap.add_argument(k, required=True)
    main(ap.parse_args())
