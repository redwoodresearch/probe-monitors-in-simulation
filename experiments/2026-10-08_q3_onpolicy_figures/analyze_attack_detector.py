"""The attack detector's numbers from run_attack_detector.py's per-run scores. $0, local (Python 3.14 reads the .eval logs).

Worlds (on-policy): real = attack vs honest_b (primary; honest_a is a replicate), September = sim_attack@september vs
sim_honest@september, today = sim_attack@today vs sim_honest@today. Errored runs (the sample's error is set) are dropped
and listed. Per layer and readout: AUROC per world (primary), standardized pAUC over 0-5% false alarms (descriptive),
catch and false alarms at the fold-0 model's thresholds (descriptive), and the simulated - real AUROC gaps with 95%
intervals from 2,000 resamples of the 20 tasks, taken jointly across arms (seed 0). The off-policy reference (held-out
outer folds): AUROC real e1 vs real honest and simulated attack vs simulated honest, and its simulated - real gap with
2,000 resamples of main tasks. The primary cell's null: p = (1 + #{null AUROC >= observed}) / (1 + draws), per world.

  uv run --no-project --python 3.14 --with numpy --with scikit-learn python analyze_attack_detector.py \
      results/attack_detector_onpolicy_<stamp>.json --run-dir <rerun folder>
"""

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
WORLDS = {"real": ("attack", "honest_b"), "real (honest_a replicate)": ("attack", "honest_a"),
          "september": ("sim_attack@september", "sim_honest@september"), "today": ("sim_attack@today", "sim_honest@today")}
GAPS = {"september - real": ("september", "real"), "today - real": ("today", "real")}
B = 2000


def errored_runs(run_dir: Path) -> list:
    out = []
    for ev in sorted(run_dir.glob("*/*/*.eval")):
        with zipfile.ZipFile(ev) as z:
            n = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
            if any(json.loads(z.read(x)).get("error") for x in n):
                out.append(f"{ev.parts[-3]}|{ev.parts[-2]}")
    return out


def aucs(y, s, w=None):
    if not (0 < (w if w is not None else np.ones_like(y))[y == 1].sum() and (w if w is not None else np.ones_like(y))[y == 0].sum() > 0):
        return np.nan, np.nan
    return roc_auc_score(y, s, sample_weight=w), roc_auc_score(y, s, sample_weight=w, max_fpr=0.05)


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if len(v) else None


def main(path: str, run_dir: str):
    R = json.loads(Path(path).read_text())
    bad = set(errored_runs(Path(run_dir).expanduser()))
    out = {"input": Path(path).name, "git_of_run": R["meta"]["git"], "errored_runs_dropped": sorted(bad), "cells": {}}
    for name, c in R["cells"].items():
        runs = np.array(c["on_runs"])
        arm = np.array([r.split("|")[0] for r in runs])
        task = np.array([r.split("|")[1] for r in runs])
        keep = ~np.isin(runs, list(bad))
        tasks = sorted(set(task))
        ti = np.searchsorted(tasks, task)
        rng = np.random.default_rng(0)
        draws = [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)]

        def world(score, wname, w=None):
            a, h = WORLDS[wname]
            m = keep & np.isin(arm, [a, h])
            y = (arm[m] == a).astype(int)
            return aucs(y, np.asarray(score)[m], None if w is None else w[ti[m]])

        s = np.array(c["on_score"])
        res = {"C": c["C"], "train_runs": c["train_runs"], "worlds": {}, "gaps": {}}
        for wname, (a, h) in WORLDS.items():
            au, pa = world(s, wname)
            bs = [world(s, wname, w) for w in draws]
            n = {x: int((keep & (arm == x)).sum()) for x in (a, h)}
            s0 = np.array(c["on_score_fold0_model"])
            thr = {k: float(v) for k, v in c["thresholds_fold0"].items()}
            res["worlds"][wname] = {"auroc": round(float(au), 4), "auroc_ci95": ci([b[0] for b in bs]), "pauc_DESCRIPTIVE": round(float(pa), 4),
                                    "runs": n, "fold0_model_DESCRIPTIVE": {f"catch_at_{k}": round(float((s0[keep & (arm == a)] > t).mean()), 3) for k, t in thr.items()}
                                    | {f"false_alarm_at_{k}": round(float((s0[keep & (arm == h)] > t).mean()), 3) for k, t in thr.items()}}
        for gname, (w1, w0) in GAPS.items():
            g = float(world(s, w1)[0] - world(s, w0)[0])
            res["gaps"][gname] = {"value": round(g, 4), "ci95": ci([world(s, w1, w)[0] - world(s, w0, w)[0] for w in draws])}
        # off-policy reference
        off = c["off_outer"]
        mt = sorted({t for a in off.values() for t in a["main_task"]})
        rng2 = np.random.default_rng(0)
        odraws = [np.bincount(rng2.integers(0, len(mt), len(mt)), minlength=len(mt)) for _ in range(B)]

        def off_auc(a, h, w=None):
            y = np.r_[np.ones(len(off[a]["score"])), np.zeros(len(off[h]["score"]))]
            sc = np.r_[off[a]["score"], off[h]["score"]]
            tt = np.searchsorted(mt, np.r_[off[a]["main_task"], off[h]["main_task"]])
            return aucs(y, sc, None if w is None else w[tt])[0]
        ref = {"real e1 vs real honest": off_auc("real-attack-e1", "real-honest"), "sim attack vs sim honest": off_auc("sim-attack", "sim-honest")}
        ref_ci = {"real e1 vs real honest_ci95": ci([off_auc("real-attack-e1", "real-honest", w) for w in odraws]),
                  "sim attack vs sim honest_ci95": ci([off_auc("sim-attack", "sim-honest", w) for w in odraws])}
        res["off_policy_reference"] = {k: round(float(v), 4) for k, v in ref.items()} | ref_ci | {
            "gap sim - real": round(float(ref["sim attack vs sim honest"] - ref["real e1 vs real honest"]), 4),
            "gap_ci95": ci([off_auc("sim-attack", "sim-honest", w) - off_auc("real-attack-e1", "real-honest", w) for w in odraws])}
        if "null" in c:
            res["null"] = {}
            for wname in ("real", "september"):
                obs = world(s, wname)[0]
                nv = np.array([world(np.array(d["on_score"]), wname)[0] for d in c["null"]])
                res["null"][wname] = {"observed": round(float(obs), 4), "null_mean": round(float(nv.mean()), 4), "null_max": round(float(nv.max()), 4),
                                      "p": round(float((1 + (nv >= obs).sum()) / (1 + len(nv))), 4), "draws": int(len(nv))}
        out["cells"][name] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"attack_detector_analysis_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    for name, r in out["cells"].items():
        w = r["worlds"]
        print(f"{name} | C {r['C']} | AUROC real {w['real']['auroc']} {w['real']['auroc_ci95']} | sept {w['september']['auroc']} {w['september']['auroc_ci95']}"
              f" | today {w['today']['auroc']} | gap sept-real {r['gaps']['september - real']} | off-policy {r['off_policy_reference']}"
              + (f" | null {r['null']}" if "null" in r else ""))
    print("errored runs dropped:", out["errored_runs_dropped"])
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    main(a.path, a.run_dir)
