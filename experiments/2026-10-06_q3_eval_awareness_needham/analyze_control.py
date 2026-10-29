"""Positive control: do the Needham probe's call_end scores move when one system-prompt line says "evaluation" or
"deployment"? Reads run_control.py's table (fast, local).

Per probe column: per transcript, the mean over its calls of (eval_new - deploy_new), (eval_new - original) and
(deploy_new - original); then the mean over transcripts with a 95% interval resampling main tasks (paired e1, e2 and
sim transcripts share a task); the share of transcripts that move the expected way; eval_new - deploy_new by the
call's distance in tokens from the appended line (the end of the system turn, `system_end_token`, recorded by
run_control.py since 2026-10-07); tables without that column (the 2026-10-06 run) are banded by the call's position from
the transcript start instead, which is only a proxy because system prompts differ in length. For scale, the LinuxArena sim - e1 run shift (analyze_la.py) next to
the eval - deploy effect, and both in units of the e1 run-score sd.

  uv run --with pandas --with pyarrow python experiments/2026-10-06_q3_eval_awareness_needham/analyze_control.py \
      --control data/control_....parquet --la results/la_shift_....json
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CONTRASTS = {"eval_new - deploy_new": ("eval_new", "deploy_new"), "eval_new - original": ("eval_new", "original"),
             "deploy_new - original": ("deploy_new", "original")}
BANDS = [(0, 4000), (4000, 8000), (8000, 16000)]
SHOW = ("asst_last", "asst_mean", "templated")


def boot_mean(per_t: pd.DataFrame, n=1000, seed=0) -> dict:
    tasks = sorted(per_t.main_task.unique())
    t = per_t.main_task.map({k: i for i, k in enumerate(tasks)}).to_numpy()
    s, c = np.bincount(t, weights=per_t.d.to_numpy(), minlength=len(tasks)), np.bincount(t, minlength=len(tasks))
    W = np.stack([np.bincount(d, minlength=len(tasks)) for d in np.random.default_rng(seed).integers(0, len(tasks), (n, len(tasks)))])
    lo, hi = np.percentile((W @ s) / np.maximum(W @ c, 1), [2.5, 97.5])
    return {"est": round(float(per_t.d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": int(len(per_t))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--la", required=True)
    ap.add_argument("--layers", default="16,24,32,40,48,56,63")
    args = ap.parse_args()
    df = pd.read_parquet(HERE / args.control)
    la = json.loads((HERE / args.la).read_text())
    layers = [int(x) for x in args.layers.split(",")] + [la["primary"]["layer"]]
    key = ["key", "step", "action"]
    cols = [c for c in df.columns if c.startswith(SHOW) and int(c.rsplit("_L", 1)[1]) in layers]
    wide = {cond: g.set_index(key) for cond, g in df.groupby("cond")}
    if any(not w.index.is_unique for w in wide.values()) or len({frozenset(w.index) for w in wide.values()}) != 1:
        raise ValueError("conditions disagree on the calls read, or duplicate calls")
    base = wide["original"][["arm", "main_task", "token"]].copy()
    if "system_end_token" in wide["original"]:  # distance from the appended line (it ends with the system turn)
        base["token"] = base["token"] - wide["original"]["system_end_token"]
        band_basis = "distance from the appended line (call token - system_end_token)"
    else:
        band_basis = "call position from the transcript start (system_end_token not recorded; a proxy for distance)"
    out = {}
    for col in cols:
        res = {}
        for name, (a, b) in CONTRASTS.items():
            d = (wide[a][col] - wide[b][col]).rename("d").to_frame().join(base)
            per_t = d.reset_index().groupby(["key", "arm", "main_task"]).d.mean().reset_index()
            r = {"all": boot_mean(per_t), "share_positive": round(float((per_t.d > 0).mean()), 3),
                 "by_arm": {arm: round(float(g.d.mean()), 4) for arm, g in per_t.groupby("arm")}}
            if name == "eval_new - deploy_new":
                r["by_token_band"] = {f"{lo}-{hi}": boot_mean(d[(d.token >= lo) & (d.token < hi)].reset_index()
                                                             .groupby(["key", "main_task"]).d.mean().reset_index()) for lo, hi in BANDS}
                r["band_basis"] = band_basis
            res[name] = r
        src = la["templated"] if col.startswith("templated") else la["needham"]
        lcol = col.replace("templated_", "") if col.startswith("templated") else col
        if lcol in src:
            shift, effect = src[lcol]["sim_minus_e1"]["est"], res["eval_new - deploy_new"]["all"]["est"]
            res["la"] = {"sim_minus_e1": src[lcol]["sim_minus_e1"], "e2_minus_e1": src[lcol]["e2_minus_e1"], "sd_e1": src[lcol]["sd_e1"],
                         "eval_minus_deploy_in_sd_e1": round(effect / src[lcol]["sd_e1"], 4),
                         "sim_shift_over_eval_minus_deploy": round(shift / effect, 4) if effect else None}
        out[col] = res

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    path = HERE / "results" / f"control_{stamp}.json"
    path.write_text(json.dumps({"inputs": vars(args), "git_commit": commit, "transcripts": int(base.reset_index().key.nunique()),
                                "calls": len(base), "by_probe": out}, indent=1))
    f = lambda x: f"{x['est']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
    print(f"{base.reset_index().key.nunique()} transcripts, {len(base)} calls")
    for col, res in out.items():
        line = "  ".join(f"{n}: {f(r['all'])} ({r['share_positive']:.2f}>0)" for n, r in res.items() if n in CONTRASTS)
        ratio = (res.get("la") or {}).get("sim_shift_over_eval_minus_deploy")
        extra = (f" | LA sim-e1 {f(res['la']['sim_minus_e1'])} = " + (f"{ratio:+.2f} x eval-deploy" if ratio is not None else "n/a (no line effect)")) if "la" in res else ""
        print(f"{col}: {line}{extra}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
