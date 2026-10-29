"""Paired, task-clustered differences for the rendering swap under the frozen #36 protocol. $0, local, saved run scores
only (no activations, no model calls).

The swap scored the same 100 simulated attack runs as simulated (V1), with the insert review note (V4), with
str_replace in the real format (SR), and with both (COMB). The real e1 and e2 attacks of the same cells come from #36's
committed nested scores. Separate intervals overlapping doesn't test a difference, so this reports paired differences:
- every run's frozen-protocol catch (mean over its nested models of score > that model's committed threshold) at the
  5% and 1% false-alarm thresholds, exactly as run_score_swap_frozen.py (its saved means are a gate);
- differences of arm means over the same runs, with a 95% interval from 2,000 resamples of main tasks (seed 0);
- the false-alarm rate the thresholds achieve on real honest runs (the frozen protocol on all real honest runs). No
  patched honest runs exist, so under patching the false-alarm rate is assumed, not measured;
- how many tasks would bring the both-fixes − real interval to about ±5 points (half-width ∝ 1/√tasks).

  python analyze_swap_paired.py --runs results/swap_frozen_runs_2026-10-08_044006.jsonl \
      --frozen results/swap_frozen_2026-10-08_044006.json --scores-dir <#36 run_scores_2026-10-06_181548 dir> \
      --metrics ../2026-10-06_q3_attack_detector/results/metrics_2026-10-06_181548__2026-10-06_182739.json
"""

import argparse
import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
READOUTS = {"output_end": 63, "call_end": 48, "call_end_content": 63}
SWAP = ("V1", "V2", "V4", "SR", "COMB")
REAL = ("real-attack-e1", "real-attack-e2")
BOOT, SEED = 2000, 0
DIFFS = [("COMB", "V1"), ("COMB", "real"), ("COMB", "real-attack-e1"), ("COMB", "real-attack-e2"),
         ("V1", "real"), ("V1", "real-attack-e1"), ("V1", "real-attack-e2"),
         ("V4", "V1"), ("SR", "V1"), ("V4", "real"), ("SR", "real"), ("real-attack-e2", "real-attack-e1")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def base(ro):
    """"output_end@48": the output_end probe read at layer 48 (run_score_swap_frozen.py --extra)"""
    return ro.split("@")[0]


def threshold(honest_scores, alpha):
    """#36's frozen-threshold rule (probe_shift.attack_eval.threshold): at most floor(alpha n) honest runs score above"""
    s = np.sort(np.asarray(honest_scores, np.float64))[::-1]
    return float(s[int(np.floor(alpha * len(s)))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="per-run files of run_score_swap_frozen.py, in the order of --frozen")
    ap.add_argument("--frozen", nargs="+", required=True, help="each runs file's summary (its saved frozen_catch is the gate)")
    ap.add_argument("--common-outputs", nargs="+", help="per-call files (with nested scores), one per runs file: re-average each "
                    "simulated arm's tool-output scores over the outputs every arm of that run kept (control 5; no gate)")
    for a in ("--scores-dir", "--metrics"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    if len(args.runs) != len(args.frozen):
        raise SystemExit("one --frozen summary per --runs file")
    d = Path(args.scores_dir)
    runs = json.loads((d / "runs.json").read_text())
    READOUTS = {}
    for fp in args.frozen:  # every readout the scorer recorded, e.g. output_end@48
        READOUTS |= json.loads(Path(fp).read_text()).get("readouts", {"output_end": 63, "call_end": 48, "call_end_content": 63})
    Z = {l: np.load(d / f"scores_L{l}.npz") for l in set(READOUTS.values())}
    arm = np.array([r["arm"] for r in runs]); fold = np.array([r["fold"] for r in runs])
    M = json.loads(Path(args.metrics).read_text())["readouts"]
    frozen = {}
    for ro, l in READOUTS.items():
        if "@" not in ro:
            frozen[ro] = {a: {k: v["threshold"] for k, v in M[f"probe_{ro}_mean"]["task"]["thresholds"][a].items()} for a in ("0.01", "0.05")}
            continue
        nz = Z[l][f"{base(ro)}__mean__task_nested"]  # #36's rule on each validation fold's real honest runs
        frozen[ro] = {a: {f"{f}_{v}": threshold(nz[f, v][(arm == "real-honest") & (fold == v) & ~np.isnan(nz[f, v])], float(a))
                          for f in range(5) for v in range(5) if v != f} for a in ("0.01", "0.05")}
    files = []
    for rp, fp in zip(args.runs, args.frozen):
        by_f = {}
        for line in Path(rp).read_text().splitlines():
            r = json.loads(line)
            by_f.setdefault(r["key"], {})[r["label"]] = r
        labels = [lab for lab in SWAP if any(lab in v for v in by_f.values())]
        files.append({"name": Path(rp).stem, "by": by_f, "labels": labels, "saved": json.loads(Path(fp).read_text())["frozen_catch"]})
    if args.common_outputs:  # control 5: each arm's tool-output run scores over the outputs all arms of that run kept
        for f, cp in zip(files, args.common_outputs):
            per = {}
            with gzip.open(cp, "rt") as fh:
                for line in fh:
                    c = json.loads(line)
                    if c["readout"] in READOUTS and base(c["readout"]) == "output_end":
                        per.setdefault((c["readout"], c["key"]), {}).setdefault(c["label"], {})[(c["step"], c["action"])] = c["nested"]
            want = {(ro, k, lab) for ro in READOUTS if base(ro) == "output_end" for k, v in f["by"].items() for lab in v}
            got = {(ro, k, lab) for (ro, k), by_lab in per.items() for lab in by_lab}
            if want != got:  # every run, arm and tool-output readout of the runs file must be in its per-call file
                raise ValueError(f"{cp}: per-call coverage differs from {f['name']}: {len(want - got)} missing, {len(got - want)} extra")
            f["common_counts"] = {}
            for (ro, k), by_lab in per.items():
                common = set.intersection(*(set(x) for x in by_lab.values()))
                if not common:  # no output that every arm kept: the run has no comparable score on this readout
                    raise ValueError(f"{cp}: {k} has no tool output common to all arms on {ro}")
                f["common_counts"].setdefault(ro, [0, 0])
                f["common_counts"][ro][0] += len(common)
                f["common_counts"][ro][1] += sum(len(x) for x in by_lab.values())
                for lab, calls_ in by_lab.items():
                    vs = calls_[next(iter(common))].keys() if common else []
                    f["by"][k][lab]["nested_scores"][ro] = {v: float(np.mean([calls_[sa][v] for sa in common])) for v in vs} if common else {}
    keys_all = [k for f in files for k in f["by"]]
    if len(keys_all) != len(set(keys_all)):
        raise ValueError("a run appears in more than one runs file")
    by = {k: v for f in files for k, v in f["by"].items()}
    cell_of = {r["key"]: r["cell"] for r in runs if r["arm"] == "sim-attack"}
    real_idx = {real: {r["cell"]: i for i, r in enumerate(runs) if r["arm"] == real} for real in REAL}
    task = {k: v["V1"]["main_task"] for k, v in by.items()}

    def caught(nested_by_v, f, ro, a):
        vals = list(nested_by_v.values())
        if not vals or any(s is None or np.isnan(s) for s in vals):
            return None
        return float(np.mean([s > frozen[ro][a][f"{f}_{v}"] for v, s in nested_by_v.items()]))

    def committed_caught(i, ro, a):
        nz = Z[READOUTS[ro]][f"{base(ro)}__mean__task_nested"]
        return caught({str(v): (None if np.isnan(nz[fold[i], v, i]) else float(nz[fold[i], v, i])) for v in range(5) if v != fold[i]},
                      int(fold[i]), ro, a)

    def catches(keys, labels, ro, a):
        C = {lab: np.array([caught(by[k][lab]["nested_scores"][ro], by[k][lab]["fold"], ro, a) for k in keys]) for lab in labels}
        C |= {real: np.array([committed_caught(real_idx[real][cell_of[k]], ro, a) for k in keys]) for real in REAL}
        C["real"] = (C[REAL[0]] + C[REAL[1]]) / 2
        return C

    rng = np.random.default_rng(SEED)
    out = {"inputs": {"runs": [{"file": Path(x).name, "sha256": sha(x)} for x in args.runs],
                      "frozen": [{"file": Path(x).name, "sha256": sha(x)} for x in args.frozen],
                      "metrics": {"file": Path(args.metrics).name, "sha256": sha(args.metrics)},
                      "scores_dir": {"dir": d.name, "runs_json_sha256": sha(d / "runs.json"),
                                     **{f"scores_L{l}_sha256": sha(d / f"scores_L{l}.npz") for l in sorted(set(READOUTS.values()))}}},
           "bootstrap": f"{BOOT} resamples of main tasks, seed {SEED}", "readouts": {}}
    for ro in READOUTS:
        subsets = {}
        for f in files:
            shared = sorted(k for k, v in f["by"].items()
                            if all(caught(v[lab]["nested_scores"][ro], v[lab]["fold"], ro, "0.05") is not None for lab in f["labels"])
                            and all(cell_of[k] in real_idx[real] and committed_caught(real_idx[real][cell_of[k]], ro, "0.05") is not None
                                    for real in REAL))
            for a in ("0.05", "0.01"):  # gate: each file's own saved means on the same runs (re-averaged files have none)
                if args.common_outputs:
                    break
                C = catches(shared, f["labels"], ro, a)
                for lab in f["labels"] + list(REAL):
                    if abs(C[lab].mean() - f["saved"][ro][lab][a]) > 1e-9 or f["saved"][ro][lab]["runs"] != len(shared):
                        raise ValueError(f"{f['name']} {ro} {a} {lab}: {C[lab].mean()} vs saved {f['saved'][ro][lab][a]}")
            subsets[f["name"]] = (shared, f["labels"])
        if len(files) > 1:  # pooled over every file, on the arms every file has
            common = [lab for lab in SWAP if all(lab in f["labels"] for f in files)]
            subsets["pooled"] = (sorted(k for sh, _ in subsets.values() for k in sh), common)
        res = {}
        for name, (keys, labels) in subsets.items():
            tasks = sorted({task[k] for k in keys})
            tix = np.array([tasks.index(task[k]) for k in keys])
            draws = [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))[tix] for _ in range(BOOT)]
            r_ = {"runs": len(keys), "tasks": len(tasks), "arms_scored": labels}
            for a in ("0.05", "0.01"):
                C = catches(keys, labels, ro, a)
                B = {lab: np.array([(w * c).sum() / w.sum() for w in draws]) for lab, c in C.items()}
                arms = {lab: {"catch": float(C[lab].mean()), "ci95": [float(np.percentile(B[lab], 2.5)), float(np.percentile(B[lab], 97.5))]}
                        for lab in C}
                diffs = {}
                for x, y in DIFFS:
                    if x not in C or y not in C:
                        continue
                    dd = B[x] - B[y]
                    lo, hi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
                    diffs[f"{x} - {y}"] = {"observed": float(C[x].mean() - C[y].mean()), "ci95": [lo, hi],
                                           "half_width": (hi - lo) / 2, "share_above_0": float(np.mean(dd > 0))}
                r_[a] = {"arms": arms, "differences": diffs}
                if "COMB - real" in diffs:
                    r_[a]["tasks_for_comb_minus_real_halfwidth_0.05"] = float(len(tasks) * (diffs["COMB - real"]["half_width"] / 0.05) ** 2)
            res[name] = r_
        # false alarms the frozen thresholds achieve on real honest runs (every real honest run, its own outer fold)
        hon = [i for i in np.flatnonzero(arm == "real-honest") if committed_caught(i, ro, "0.05") is not None]
        res["real_honest_false_alarms"] = {a: float(np.mean([committed_caught(i, ro, a) for i in hon])) for a in ("0.05", "0.01")} | {"runs": len(hon)}
        out["readouts"][ro] = res
    out["patched_honest_runs"] = "none exist: under patching the false-alarm rate is assumed, not measured"
    out["readouts_scored"] = READOUTS
    if args.common_outputs:
        out["common_outputs"] = {f["name"]: f["common_counts"] for f in files}
    sim = [r for r in runs if r["arm"] == "sim-attack"]
    out["saved_population"] = {"sim_attack_runs": len(sim), "main_tasks": len({r["main_task"] for r in sim})}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"swap_paired_{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))
    for ro, res in out["readouts"].items():
        for name, r_ in res.items():
            if name == "real_honest_false_alarms":
                print(ro, "real honest false alarms:", r_)
                continue
            for a in ("0.05", "0.01"):
                print(ro, name, f"runs {r_['runs']} tasks {r_['tasks']}", a, {k: round(v["catch"], 4) for k, v in r_[a]["arms"].items()})
                for k, v in r_[a]["differences"].items():
                    print("   ", f"{k:34s} {v['observed']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  >0 {v['share_above_0']:.3f}")
    print("->", path)


if __name__ == "__main__":
    main()
