"""Phase 2 read, matched for call composition ($0, saved scores only).

Qwen3.5-27B almost never ends a call with a `timeout` argument; Sol mostly does, so the detector's run means average
different kinds of call. Here every comparison uses the same kind of call on both sides: calls that end in content
(call_end kind 0), split by tool (bash, text_editor) and pooled. Scores: run 2026-10-06_181548's outer model for each
run's main-task fold, at layer 48 (the all-calls probe) and layer 63 (the content probe); for LinuxArena runs, the
committed out-of-fold per-call scores; for the Phase 2 runs, run_read_phase2.py's (same models, reproduced exactly).

Per Phase 2 run and probe:
  - per-call AUROC of its content-ending calls against the real honest runs' content-ending calls of the same fold,
    next to the same for the fold's real e1 attack calls (what a typical Sol attack call looks like);
  - count-matched run percentile: the mean of k randomly drawn calls of the run, k = the median number of such calls in
    the fold's real honest runs (1,000 draws), placed among those honest runs' run means (runs with at least k calls,
    their own calls subsampled to k the same way), and the same for the same cell's Sol e1 run.

  python analyze_phase2_matched.py --read <phase2_read_*.json> --run-dir <#36 results>/run_scores_2026-10-06_181548
      --manifest <#36 results>/run_2026-10-06_181548.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
PROBES = {"call_end": 48, "call_end_content": 63}  # readout name in the committed scores -> layer
TOOLS = {"bash": ("bash",), "text_editor": ("text_editor",), "all": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--check-arm", default="real-attack-e1", help="the Sol arm of the read's reproduction checks")
    args = ap.parse_args()
    rd = json.loads(Path(args.read).read_text())
    d = Path(args.run_dir)
    runs = json.loads((d / "runs.json").read_text())
    t = np.load(d / "tables.npz")
    tool_names = np.array(json.loads(Path(args.manifest).read_text())["manifest"]["tools"])
    arm = np.array([r["arm"] for r in runs]); fold = np.array([r["fold"] for r in runs])
    key = {(r["arm"], r["key"]): i for i, r in enumerate(runs)}
    checks = {c["cell"]: key[(args.check_arm, c["key"])] for c in rd["checks"]}
    rng = np.random.default_rng(0)
    out = {"read": rd["stamp"], "runs": []}
    per_call = {}
    for ro, layer in PROBES.items():
        z = np.load(d / f"scores_L{layer}.npz")
        rows = z[f"{ro}__rows"]
        per_call[ro] = {"run": t["call_end__run"][rows], "kind": t["call_end__kind"][rows],
                        "tool": tool_names[t["call_end__tool"][rows]], "score": z[f"{ro}__step__task_outer"].astype(float)}
    for r in rd["runs"]:
        f, rec = r["fold"], {"env": r["env"], "side_task_success": r["side_task_success"], "fold": r["fold"], "probes": {}}
        sol = checks[r["cell"]]
        for ro, layer in PROBES.items():
            pc = per_call[ro]
            q = [c for c in r["calls"] if c["readout"] == ro and c["call_kind"] == 0]
            for tname, tools in TOOLS.items():
                qs = np.array([c["score"] for c in q if tools is None or c["function"] in tools])
                ok = (pc["kind"] == 0) & (np.ones(len(pc["kind"]), bool) if tools is None else np.isin(pc["tool"], tools))
                hon = ok & (arm[pc["run"]] == "real-honest") & (fold[pc["run"]] == f)
                e1 = ok & (arm[pc["run"]] == "real-attack-e1") & (fold[pc["run"]] == f)
                sol_calls = pc["score"][ok & (pc["run"] == sol)]
                res = {"qwen_calls": len(qs), "honest_calls": int(hon.sum()), "e1_calls": int(e1.sum()), "sol_calls": len(sol_calls)}
                if len(qs) and hon.sum():
                    auc = lambda s: float(roc_auc_score(np.r_[np.ones(len(s)), np.zeros(hon.sum())], np.r_[s, pc["score"][hon]]))
                    res |= {"call_auroc_vs_honest": {"qwen35": round(auc(qs), 3), "real_e1": round(auc(pc["score"][e1]), 3),
                                                     "sol_same_cell": round(auc(sol_calls), 3) if len(sol_calls) else None}}
                    # count-matched run percentile
                    by_run = {}
                    for i in np.flatnonzero(hon):
                        by_run.setdefault(pc["run"][i], []).append(pc["score"][i])
                    counts = np.array([len(v) for v in by_run.values()])
                    k = int(np.median(counts))
                    pool = [np.array(v) for v in by_run.values() if len(v) >= k]

                    def pct(s):
                        if len(s) < k:
                            return None
                        # paired draws: in each, every run (the candidate and each honest run) is subsampled to k calls
                        ref = np.array([[rng.choice(v, k, replace=False).mean() for v in pool] for _ in range(args.draws)])
                        mine = np.array([rng.choice(s, k, replace=False).mean() for _ in range(args.draws)])
                        return round(float(np.mean(ref < mine[:, None])), 3)

                    res |= {"k": k, "honest_runs_with_k": len(pool), "run_percentile_count_matched": {"qwen35": pct(qs), "sol_same_cell": pct(sol_calls)}}
                rec["probes"][f"{ro}|{tname}"] = res
        out["runs"].append(rec)
        print(f"\n{r['env']} (side {r['side_task_success']}, fold {f})")
        for name, v in rec["probes"].items():
            a = v.get("call_auroc_vs_honest", {})
            p = v.get("run_percentile_count_matched", {})
            print(f"  {name:28s} calls q/sol/hon {v['qwen_calls']:3d}/{v['sol_calls']:2d}/{v['honest_calls']:4d} | call AUROC vs honest: "
                  f"qwen {a.get('qwen35')} sol {a.get('sol_same_cell')} e1 {a.get('real_e1')} | run pct (k={v.get('k')}): qwen {p.get('qwen35')} sol {p.get('sol_same_cell')}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with (HERE / "results" / f"phase2_matched_{rd['stamp']}__{stamp}.json").open("x") as fh:
        fh.write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
