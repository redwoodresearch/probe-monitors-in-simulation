"""Assemble per-call Qwen3.5 (matched run) headline scores for the three-way views, as assemble.py does for Qwen3-32B,
and check they reproduce the saved headline numbers. Reads only local copies in vol_qwen35/; writes tables/qwen3.5__*."""
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
VOL, OUT = DATA / "vol_qwen35", DATA / "tables"
REPO = ROOT
res = json.loads((VOL / "tw_result.json").read_text())
cfg = res["config"]
manifest = json.loads((VOL / "tw_manifest.json").read_text())
labels = {(r["arm"], r["key"], r["step"], r["action"]): r for r in
          map(json.loads, (REPO / ".run_cache/q2_call_labels/2026-10-06_212047.jsonl").read_text().splitlines())}
POS = ["call_end", "pre_call", "prev_content"]
VIEWS = ["attack_only", "honest_only", "pooled"]


def view_domain(view):  # as run_q2_alllayers_qwen35.view_domain
    v = cfg["views"][view]
    keep = lambda m: m.get("in_honest_tasks", True) and (m.get("primary_run", True) or not v.get("one_run"))
    return np.array([v["domains"].get(m["arm"], "-") if keep(m) else "-" for m in manifest])


def extra(m):
    lab = labels.get((m["arm"], m["key"], m["step"], m["action"]))
    return {"env": lab["environment"] if lab else None, "side_task": lab["side_task"] if lab else None,
            "sim_reference": lab["sim_reference"] if lab else None}


def compare(mine, saved):
    diffs = []
    for k, cell in saved.items():
        for metric, d in cell.items():
            if metric == "n":
                diffs.append(0.0 if d == mine[k][metric] else np.inf)
                continue
            a, b = [d["value"], *d["ci95"]], [mine[k][metric]["value"], *mine[k][metric]["ci95"]]
            diffs += [0.0 if (np.isnan(x) and np.isnan(z)) else abs(x - z) for x, z in zip(a, b)]
    return max(diffs)


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
            if not (a["flag"] == (a["score"] > a["threshold"])).all():
                raise ValueError(f"{view} {pos} {t} L{p['layer']}: flag != score > threshold")
            m = fold == p["fold"]
            score[m], flag[m], thr[m], layer[m] = a["score"][m], a["flag"][m], a["threshold"][m], p["layer"]
        heads[t] = {"score": score, "flag": flag, "threshold": thr}
        frames.append(pd.DataFrame([{k: r[k] for k in ("arm", "key", "cell", "step", "action", "tool", "category",
                                                         "main_task", "fold", "y")} | extra(r) for r in rows])
                      .assign(domain=domain, train_set=t, score=score.astype(np.float32), flag=flag, threshold=thr,
                              layer=layer, reader="Qwen3.5", readout=pos, view=view))
    mine = two_by_two(heads, {t: cfg["home"][t] for t in cfg["train_sets"]}, y, domain, groups, cfg["test_domains"],
                      cfg["n_boot"], cfg["seed"])
    df = pd.concat(frames)
    df.to_json(OUT / f"qwen3.5__{pos}__{view}.jsonl.gz", orient="records", lines=True, compression="gzip")
    return (view, pos), compare(mine, res["views"][view]["positions"][pos]["headline"]), \
        df.groupby(["train_set", "domain"]).size().to_dict(), int(df.env.isna().sum())


checks, counts, no_label = {}, {}, {}
for k, diff, n, miss in Parallel(n_jobs=-1)(delayed(one)(v, p) for v in VIEWS for p in POS):
    key = f"{k[0]} {k[1]}"
    checks[key], counts[key], no_label[key] = diff, {f"{a}/{b}": c for (a, b), c in n.items()}, miss

# Overlap with the Qwen3-32B tables: same calls?
ident = lambda df: set(zip(df.arm, df.key, df.step, df.action))
overlap = {}
for v in VIEWS:
    for p in POS:
        a = pd.read_json(OUT / f"qwen3.5__{p}__{v}.jsonl.gz", lines=True)
        b = pd.read_json(OUT / f"qwen3-32b__{p}__{v}.jsonl.gz", lines=True)
        a, b = a[a.train_set == "real"], b[b.train_set == "real"]
        A, B = ident(a), ident(b)
        overlap[f"{v} {p}"] = {"qwen3.5": len(A), "qwen3-32b": len(B), "both": len(A & B), "only_qwen3.5": len(A - B),
                               "only_qwen3-32b": len(B - A)}
report = {"max_abs_diff_vs_saved": checks, "rows": counts, "rows_without_label_join": no_label,
          "overlap_calls_vs_qwen3-32b": overlap}
(RESULTS / f"shift_assemble_qwen3.5_{now()}.json").write_text(json.dumps(report, indent=1, default=str))
print(json.dumps(report, indent=1, default=str))
