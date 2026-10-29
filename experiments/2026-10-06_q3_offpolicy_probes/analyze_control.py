"""Positive control for the eval-awareness probe: do call_end scores move when one line in the system prompt says
"evaluation" or "deployment"? Reads run_eval_awareness_control.py's table (fast, local).

Per probe and layer: per transcript, the mean over its calls of (condition - original), then the mean over
transcripts with a 95% bootstrap interval over transcripts; eval - deploy for the training and the new wording;
the share of transcripts where eval > deploy; eval - deploy by the call's token position (distance from the line);
and the LinuxArena sim - e1 run shift from analyze_shift.py next to it, for scale. Pipeline check: the original
condition against the stored segment-end scores (same calls, same model).

  uv run --no-project --python 3.13 --with pandas --with pyarrow --with pyyaml python \
      experiments/2026-10-06_q3_offpolicy_probes/analyze_control.py --control data/control_....parquet \
      --stored-ea data/scores_eval_awareness_segment_end_....parquet --stored-apollo data/scores_apollo_deception_segment_end_....parquet \
      --shift-ea results/shift_eval_awareness_....json
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CONTRASTS = {"eval_train - original": ("eval_train", "original"), "deploy_train - original": ("deploy_train", "original"),
             "eval_new - original": ("eval_new", "original"), "deploy_new - original": ("deploy_new", "original"),
             "eval_train - deploy_train": ("eval_train", "deploy_train"), "eval_new - deploy_new": ("eval_new", "deploy_new")}
BANDS = [(0, 4000), (4000, 8000), (8000, 16000)]
BOOTSTRAP, SEED = 1000, 0


def boot_mean(per_t: pd.DataFrame, n=BOOTSTRAP, seed=SEED):
    """Mean of per-transcript values `d`, with a 95% interval resampling main tasks (paired e1/e2/sim transcripts
    share a task, so they are resampled together). An empty input (e.g. no calls in a token band) gives no estimate."""
    if per_t.empty:
        return {"est": None, "ci95": [None, None], "n": 0, "tasks": 0}
    tasks = sorted(per_t.main_task.unique())
    t = per_t.main_task.map({k: i for i, k in enumerate(tasks)}).to_numpy()
    s, c = np.bincount(t, weights=per_t.d.to_numpy(), minlength=len(tasks)), np.bincount(t, minlength=len(tasks))
    W = np.stack([np.bincount(d, minlength=len(tasks)) for d in np.random.default_rng(seed).integers(0, len(tasks), (n, len(tasks)))])
    draws = (W @ s) / np.maximum(W @ c, 1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"est": round(float(per_t.d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n": int(len(per_t)), "tasks": len(tasks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--stored-ea", required=True)
    ap.add_argument("--stored-apollo", required=True)
    ap.add_argument("--shift-ea", required=True)
    args = ap.parse_args()
    df = pd.read_parquet(HERE / args.control)
    key = ["key", "step", "action"]
    wide = {}
    cols = [c for c in df.columns if c.startswith(("eval_awareness_L", "apollo_deception_L"))]
    for cond, g in df.groupby("cond"):
        w = g.set_index(key)
        if not w.index.is_unique:  # sample keys embed the cohort's file name, so (key, step, action) is unique across arms
            raise ValueError(f"{cond}: duplicate (key, step, action) rows")
        wide[cond] = w[cols + ["arm", "main_task", "token"] if cond == "original" else cols]
    calls = set(wide["original"].index)
    if any(set(w.index) != calls for w in wide.values()):
        raise ValueError("conditions disagree on the calls read")
    base = wide["original"]

    # Pipeline check: original condition vs the stored segment-end scores.
    check = {}
    for name, path in (("eval_awareness", args.stored_ea), ("apollo_deception", args.stored_apollo)):
        st = pd.read_parquet(HERE / path)
        st = st[st.kind == "call_end"].set_index(key)
        if not st.index.is_unique:
            raise ValueError(f"duplicate (key, step, action) rows in the stored {name} scores")
        stored = base[[]].join(st[[c for c in st.columns if c.startswith("L")]], how="left")
        if stored["L48"].isna().any():
            raise ValueError(f"{int(stored['L48'].isna().sum())} control calls missing from the stored {name} scores")
        check[name] = {l: {"max_abs_diff": round(float((base[f"{name}_{l}"] - stored[l]).abs().max()), 4),
                           "corr": round(float(np.corrcoef(base[f"{name}_{l}"], stored[l])[0, 1]), 5)}
                       for l in ("L16", "L24", "L32", "L40", "L48")}

    shift = json.loads((HERE / args.shift_ea).read_text())["call_end"]["all"]
    out = {"pipeline_check": check, "transcripts": int(base.reset_index().key.nunique()),
           "transcripts_per_arm": base.reset_index().groupby("arm").key.nunique().to_dict(), "calls": len(base), "by_probe": {}}
    for col in cols:
        res = {}
        for name, (a, b) in CONTRASTS.items():
            d = (wide[a][col] - wide[b][col]).rename("d").to_frame().join(base[["arm", "main_task", "token"]])
            per_t = d.reset_index().groupby(["key", "arm", "main_task"]).d.mean().reset_index()
            r = {"all": boot_mean(per_t), "share_positive": round(float((per_t.d > 0).mean()), 3),
                 "by_arm": {arm: boot_mean(g) for arm, g in per_t.groupby("arm")}}
            if name in ("eval_train - deploy_train", "eval_new - deploy_new"):
                r["by_token_band"] = {f"{lo}-{hi}": boot_mean(d[(d.token >= lo) & (d.token < hi)].reset_index()
                                                             .groupby(["key", "main_task"]).d.mean().reset_index()) for lo, hi in BANDS}
            res[name] = r
        if col.startswith("eval_awareness_"):
            l = col.removeprefix("eval_awareness_")
            res["la_sim_minus_e1_run_shift"] = shift[l]["sim_minus_e1"]["run_shift"]
        out["by_probe"][col] = res

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
    # The code itself, not only the commit: a branch commit can be rewritten by a rebase before it is merged.
    meta = {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "settings": {"bootstrap": BOOTSTRAP, "seed": SEED, "token_bands": BANDS, "contrasts": CONTRASTS},
            "python": sys.version.split()[0], "packages": {p: version(p) for p in ("numpy", "pandas", "pyarrow")},
            "input_sha256": {k: hashlib.sha256((HERE / v).read_bytes()).hexdigest() for k, v in vars(args).items()}}
    path = HERE / "results" / f"control_{stamp}.json"
    with path.open("x") as fh:  # append-only
        fh.write(json.dumps({"inputs": vars(args)} | meta | out, indent=1))
    f = lambda x: f"{x['est']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
    print(f"{out['transcripts']} transcripts {out['transcripts_per_arm']}, {out['calls']} calls; pipeline check:",
          json.dumps({k: {l: v[l] for l in ("L24", "L48")} for k, v in check.items()}))
    for col, res in out["by_probe"].items():
        line = "  ".join(f"{n}: {f(r['all'])} ({r['share_positive']:.2f}>0)" for n, r in res.items() if isinstance(r, dict) and "all" in r)
        print(f"{col}: {line}")
        if "la_sim_minus_e1_run_shift" in res:
            bands = res["eval_new - deploy_new"]["by_token_band"]
            print(f"    LA sim - e1 run shift {f(res['la_sim_minus_e1_run_shift'])}; eval_new - deploy_new by token band: "
                  + "  ".join(f"{b} {f(v)} n={v['n']}" if v["n"] else f"{b} (no calls)" for b, v in bands.items()))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
