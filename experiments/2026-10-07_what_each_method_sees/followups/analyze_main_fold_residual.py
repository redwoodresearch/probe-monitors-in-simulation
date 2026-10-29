"""What carries Q1's level-7 residual on main-task folds (0.666 TF-IDF) when held-out environments are near chance?

Saved scores only (../.run_cache/joined.npz) plus the level-7 top features of the full TF-IDF fit
(../results/top_features_L7_2026-10-07_124121.json). Real e1 vs simulated, outputs both methods scored.
1. Environment identity vs environment-specific tells: pooled run AUROC; the same after centering run scores on their
   environment's mean (an environment offset cannot help in this paired design, so the two should match); the mean of
   per-environment AUROCs; and per environment, main-task folds vs held-out environments, with 95% main-task bootstraps.
2. The top 25 features each way: how many outputs contain each n-gram, per arm, and the share in its most common
   environment.
3. 10 examples: outputs where the main-fold model is surest and right relative to the held-out-environment model
   (largest z_main - z_heldout, signed by the label), one per run, two per main task.

  uv run python experiments/2026-10-07_what_each_method_sees/followups/analyze_main_fold_residual.py
"""

import gzip
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from probe_shift.provenance import capture
from probe_shift.scoring import auroc_ci
from probe_shift.scrub import scrub

HERE = Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent.parent
E1, SIM = "real-attack-e1", "sim-attack"
TOKEN = re.compile(r"(?u)\b\w+\b|[^\w\s]")


def contains(ngram: str, kind: str, text: str, tokens: set) -> bool:
    return ngram in text if kind == "char" else ngram in tokens


def main():
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    cfg = yaml.safe_load((EXP / "config/what_sees.yaml").read_text())
    T = dict(np.load(EXP / ".run_cache/joined.npz"))
    key, arm, env, task = T["key"], T["arm"], T["environment"], T["main_task"]
    raw = {(r["key"], r["step"], r["action"]): r["readout_text"]
           for r in map(json.loads, gzip.open(Path(cfg["staging_dir"]).expanduser() / "outputs.jsonl.gz", "rt")) if r["kept"]}
    s = {(m, sp): T[f"{m}_L7_{sp}_real_vs_sim"] for m in ("tfidf", "probe") for sp in ("task", "env")}
    rows = np.isin(arm, [E1, SIM]) & np.logical_and.reduce([~np.isnan(v) for v in s.values()])
    idx = np.flatnonzero(rows)
    out = {"stamp": stamp, "provenance": capture(ROOT, [Path(__file__)]), "outputs": int(len(idx))}

    # 1. Run scores, pooled vs environment-centered vs per environment.
    keys, inv = np.unique(key[idx], return_inverse=True)
    first = idx[np.unique(inv, return_index=True)[1]]
    y = arm[first] == SIM
    run_env, run_task = env[first], task[first]
    res = {}
    for (m, sp), v in s.items():
        rs = np.bincount(inv, v[idx]) / np.bincount(inv)
        centered = rs - np.array([rs[run_env == e].mean() for e in run_env])
        per_env = {e: auroc_ci(y[run_env == e], rs[run_env == e], run_task[run_env == e]) for e in sorted(set(run_env))}
        res[f"{m} {sp}"] = {"pooled": auroc_ci(y, rs, run_task), "environment_centered": auroc_ci(y, centered, run_task),
                            "mean_of_per_environment": round(float(np.mean([p["auroc"] for p in per_env.values()])), 4),
                            "per_environment": per_env}
    out["run_auroc"] = res

    # 2. Top features: presence per output (level-7 text), per arm, and environment concentration.
    texts = {i: scrub(raw[(key[i], int(T["step"][i]), int(T["action"][i]))], 7) for i in idx}
    toks = {i: set(TOKEN.findall(t)) | {" ".join(p) for p in zip(TOKEN.findall(t), TOKEN.findall(t)[1:])} for i, t in texts.items()}
    top = json.loads((EXP / "results/top_features_L7_2026-10-07_124121.json").read_text())["levels"]["L7"]["tfidf"]
    feats = []
    for side, fs in top.items():
        for rank, f in enumerate(fs, 1):
            g = f["ngrams"][0]
            hit = [i for i in idx if contains(g, f["kind"], texts[i], toks[i])]
            by_env = Counter(env[i] for i in hit)
            feats.append({"side": side, "rank": rank, "ngram": g, "kind": f["kind"], "weight": f["weight"], "outputs": len(hit),
                          "real": sum(arm[i] == E1 for i in hit), "sim": sum(arm[i] == SIM for i in hit),
                          "top_environment": by_env.most_common(1)[0][0] if hit else None,
                          "top_environment_share": round(by_env.most_common(1)[0][1] / len(hit), 3) if hit else None})
    out["top_features"] = feats

    # 3. Examples: what the main-fold model knows that the held-out-environment model doesn't.
    z = {sp: (s[("tfidf", sp)] - s[("tfidf", sp)][idx].mean()) / s[("tfidf", sp)][idx].std() for sp in ("task", "env")}
    sign = np.where(arm == SIM, 1.0, -1.0)
    gap = np.full(len(key), -np.inf)
    gap[idx] = sign[idx] * (z["task"][idx] - z["env"][idx])
    ex, runs, tasks = [], set(), Counter()
    for want in (SIM, E1):
        n = 0
        for i in np.argsort(-gap):
            if arm[i] != want or key[i] in runs or tasks[task[i]] >= 2 or n == 5:
                continue
            runs.add(key[i]); tasks[task[i]] += 1; n += 1
            ex.append({"arm": arm[i], "environment": env[i], "tool": T["tool"][i], "key": key[i], "step": int(T["step"][i]),
                       "z_main_folds": round(float(z["task"][i]), 2), "z_heldout_env": round(float(z["env"][i]), 2),
                       "level7_text": texts[i][:600]})
    out["examples"] = ex

    fmt = lambda v: f"{v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
    md = [f"# What carries the level-7 residual on main-task folds ({stamp})", "", "## Run AUROC, real e1 vs simulated", "",
          "| method, split | pooled | environment-centered | mean of per-environment |", "|---|---|---|---|"]
    md += [f"| {k} | {fmt(v['pooled'])} | {fmt(v['environment_centered'])} | {v['mean_of_per_environment']:.3f} |" for k, v in res.items()]
    md += ["", "Per environment (TF-IDF main folds / held-out env; probe main folds / held-out env):", "",
           "| environment | runs | TF-IDF main | TF-IDF held-out | probe main | probe held-out |", "|---|---|---|---|---|---|"]
    for e in sorted(set(run_env)):
        md.append(f"| {e} | {int((run_env == e).sum())} | " + " | ".join(fmt(res[k]["per_environment"][e]) for k in
                                                                    ("tfidf task", "tfidf env", "probe task", "probe env")) + " |")
    md += ["", "## Top 25 features each way (full fit): outputs containing the n-gram, and its most common environment", "",
           "| side | # | n-gram | weight | outputs (real / sim) | top environment (share) |", "|---|---|---|---|---|---|"]
    md += [f"| {f['side'].replace('toward_', '')} | {f['rank']} | `{f['ngram'].replace(chr(10), chr(92) + 'n').replace('|', chr(92) + '|')}` | "
           f"{f['weight']:+.3f} | {f['outputs']} ({f['real']} / {f['sim']}) | {f['top_environment']} ({f['top_environment_share']}) |" for f in feats]
    md += ["", "## Where the main-fold model is surer than the held-out-environment model (right side)", ""]
    for e in ex:
        md += [f"**{e['arm']}**, {e['environment']}, {e['tool']}, step {e['step']}: z main folds {e['z_main_folds']:+.2f}, "
               f"held-out env {e['z_heldout_env']:+.2f}", "", "```", e["level7_text"].replace("```", "'''"), "```", ""]
    (HERE / "results" / f"main_fold_residual_{stamp}.json").write_text(json.dumps(out, indent=1, default=str))
    (HERE / "results" / f"main_fold_residual_{stamp}.md").write_text("\n".join(md))
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
