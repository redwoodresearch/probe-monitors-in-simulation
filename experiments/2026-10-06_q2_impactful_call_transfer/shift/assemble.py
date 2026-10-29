"""Assemble per-call Qwen3-32B headline scores for the three-way views and the TF-IDF baseline, and check that they
reproduce the saved headline numbers. Reads only local copies; writes to scratchpad/shift/tables/."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from probe_shift.transfer import two_by_two
import sys
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402

S = Path(__file__).parent
VOL, OUT = DATA / "vol", DATA / "tables"
OUT.mkdir(exist_ok=True)
REPO = ROOT
E = REPO / "experiments/2026-10-06_q2_impactful_call_transfer"

res = json.loads((VOL / "tw_result.json").read_text())
cfg = res["config"]
manifest = json.loads((VOL / "tw_manifest.json").read_text())
labels = {(r["arm"], r["key"], r["step"], r["action"]): r for r in
          map(json.loads, (REPO / ".run_cache/q2_call_labels/2026-10-06_212047.jsonl").read_text().splitlines())}
POS = ["call_end", "pre_call", "prev_content"]
VIEWS = ["attack_only", "honest_only", "pooled"]


def view_domain(view):  # as run_q2_alllayers.view_domain
    v = cfg["views"][view]
    keep = lambda m: m.get("in_honest_tasks", True) and (m.get("primary_run", True) or not v.get("one_run"))
    return np.array([v["domains"].get(m["arm"], "-") if keep(m) else "-" for m in manifest])


def extra(m):
    lab = labels.get((m["arm"], m["key"], m["step"], m["action"]))
    return {"env": lab["environment"] if lab else None, "side_task": lab["side_task"] if lab else None,
            "sim_reference": lab["sim_reference"] if lab else None}


def compare(mine, saved):
    """Max absolute difference over every value and CI bound of two two_by_two reports (both rounded to 4 dp)."""
    diffs = []
    for k, cell in saved.items():
        for metric, d in cell.items():
            if metric == "n":
                diffs.append(0.0 if d == mine[k][metric] else np.inf)
                continue
            a = [d["value"], *d["ci95"]]
            b = [mine[k][metric]["value"], *mine[k][metric]["ci95"]]
            diffs += [0.0 if (np.isnan(x) and np.isnan(z)) else abs(x - z) for x, z in zip(a, b)]
    return max(diffs)


checks, counts = {}, {}


def one(view, pos):
    dom = view_domain(view)
    test = np.isin(dom, cfg["test_domains"])
    rows = [m for m, t in zip(manifest, test) if t]
    y = np.array([m["y"] for m in rows])
    domain, fold = dom[test], np.array([m["fold"] for m in rows])
    groups = np.array([m["main_task"] for m in rows])
    heads, frames = {}, []
    for t, picks in res["views"][view]["positions"][pos]["headline_layers"].items():
        score, flag, thr, layer = np.zeros(len(y)), np.zeros(len(y), bool), np.zeros(len(y)), np.zeros(len(y), int)
        for p in picks:
            a = np.load(VOL / "fits" / f"{view}__{pos}__{t}__L{p['layer']}__real.npz")
            if len(a["score"]) != len(y):
                raise ValueError(f"{view} {pos} {t} L{p['layer']}: {len(a['score'])} scores for {len(y)} rows")
            m = fold == p["fold"]
            score[m], flag[m], thr[m], layer[m] = a["score"][m], a["flag"][m], a["threshold"][m], p["layer"]
        heads[t] = {"score": score, "flag": flag, "threshold": thr}
        frames.append(pd.DataFrame([{k: r[k] for k in ("arm", "key", "cell", "step", "action", "tool", "category",
                                                         "main_task", "fold", "y")} | extra(r) for r in rows])
                      .assign(domain=domain, train_set=t, score=score.astype(np.float32), flag=flag, threshold=thr,
                              layer=layer, reader="Qwen3-32B", readout=pos, view=view))
    mine = two_by_two(heads, {t: cfg["home"][t] for t in cfg["train_sets"]}, y, domain, groups, cfg["test_domains"],
                      cfg["n_boot"], cfg["seed"])
    df = pd.concat(frames)
    df.to_json(OUT / f"qwen3-32b__{pos}__{view}.jsonl.gz", orient="records", lines=True, compression="gzip")
    return (view, pos), compare(mine, res["views"][view]["positions"][pos]["headline"]), \
        df.groupby(["train_set", "domain"]).size().to_dict()


for k, diff, n in Parallel(n_jobs=-1)(delayed(one)(v, p) for v in VIEWS for p in POS):
    checks[f"{k[0]} {k[1]}"], counts[f"{k[0]} {k[1]}"] = diff, {f"{a}/{b}": c for (a, b), c in n.items()}

# Command-text TF-IDF: attack-only predictions (the baselines ran on the attack arms only).
tf = pd.read_json(REPO / ".run_cache/q2_predictions/text_baseline_predictions_2026-10-06_214947.jsonl.gz", lines=True)
tf = tf[tf.baseline == "command_text"]
mrow = {(m["arm"], m["key"], m["step"], m["action"]): m for m in manifest}
tf["main_task"] = [mrow[(a, k, s, c)]["main_task"] for a, k, s, c in zip(tf.arm, tf.key, tf.step, tf.action)]
ex = pd.DataFrame([extra({"arm": a, "key": k, "step": s, "action": c}) for a, k, s, c in
                   zip(tf.arm, tf.key, tf.step, tf.action)], index=tf.index)
tf = pd.concat([tf, ex], axis=1)
saved = json.loads((E / "results/text_baselines_2026-10-06_214947.json").read_text())
bcfg = saved["config"]
first = tf[tf.train_set == tf.train_set.iloc[0]]
y, domain, groups = first.y.to_numpy(), first.arm.to_numpy(), first.main_task.to_numpy()
heads = {}
for t, g in tf.groupby("train_set", sort=False):
    if not ((g.arm.to_numpy() == domain).all() and (g.key.to_numpy() == first.key.to_numpy()).all()):
        raise ValueError(f"{t}: rows not aligned with the first training set")
    heads[t] = {"score": g.score.to_numpy(), "flag": g.flag.to_numpy(), "threshold": g.threshold.to_numpy()}
mine = two_by_two(heads, bcfg["home"], y, domain, groups, bcfg["test_domains"], bcfg["n_boot"], 0)
checks["command_text attack (all_calls)"] = compare(mine, saved["baselines"]["command_text"]["all_calls"])
tf = tf.assign(domain=tf.arm.map({"real-attack-e1": "real", "sim-attack": "sim"}), reader="Qwen3-32B-tokens-n/a",
               readout="command_text", view="attack_only")
tf.to_json(OUT / "command_text__attack_only.jsonl.gz", orient="records", lines=True, compression="gzip")
counts["command_text attack_only"] = {f"{a}/{b}": c for (a, b), c in tf.groupby(["train_set", "arm"]).size().items()}
report = {"max_abs_diff_vs_saved": checks, "rows": counts}
(RESULTS / f"shift_assemble_qwen3-32b_{now()}.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
