"""The real->sim after-insert AUROC gap after standardizing measured trajectory characteristics within each label.

Descriptive, saved scores only. Within each label, strata are: next-call kind (tool and labeler category), step quartile
(edges from the pooled pre_call calls), earlier edits (0-3, 4-9, 10+), turns since the last insert (1, 2-3, 4+), and
the last output's kind. Real calls keep weight 1; sim calls are reweighted to real's stratum distribution within each
label; both arms are restricted to common support (strata with calls from both). Gaps are real minus sim AUROC:
  all calls (no restriction) | common support, unweighted | common support, sim standardized
Share removed = (all-calls gap - standardized gap) / all-calls gap. 1000 main-task draws (seed 0); every draw recomputes
the stratum counts, the support and the weights from its own multiplicities. Diagnostics: effective sample size and the
largest normalized weights per label; per draw, the share of calls outside common support.
Matching is on downstream variables, so a removed share does not show that behavior causes the gap, and a remaining gap
does not show that text does.

Primary (Codex's design note): the rendering swap's 100 runs, after their first re-rendered insert, as V2 (insert
outputs re-rendered with the review note) and V1 (as simulated), against the same cells' real e1 runs after their first
insert output. The question is what the rendering fix leaves.
Secondary: all 939 run pairs, raw sim against real, calls after the first insert output (the first scored version).
Readouts: pre_call and its content-token control prev_content (frozen real-trained Qwen3-32B probes). The two analyses
use different probe layers for prev_content (primary: the swap's 9/16/17 rule; secondary: the 64-layer headline
picks); the results file records the layers per fold and every input's sha256.
Predictions, written before scoring: predictions/behavior_matched_predictions.md (copied into the results file).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import examples as ex  # noqa: E402
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402

HERE = Path(__file__).parent
SWAP = ROOT / "experiments" / "2026-10-07_q2_rendering_swap"
N_BOOT, SEED = 1000, 0
K = ["arm", "key", "step", "action"]
COV = ["y", "kind", "stepbin", "editbin", "sincebin", "last"]
RENDERED = ("exact", "approx_window", "approx_numbering")


def insert_history(keys):
    """Per call: inserts shown in earlier turns, and turns since the last of them (a turn's calls see only earlier turns)."""
    rows = []
    for key, calls in ex.load_runs(keys).items():
        n_ins, last = 0, None
        for s in sorted({o[0] for o in calls}):
            turn = sorted(o for o in calls if o[0] == s)
            rows += [{"key": key, "step": sa[0], "action": sa[1], "n_ins": n_ins, "since_ins": None if last is None else s - last}
                     for sa in turn]
            if any(calls[sa]["tool"] == "text_editor" and ex.editor_args(calls[sa])[0] == "insert" for sa in turn):
                n_ins, last = n_ins + 1, s
    return pd.DataFrame(rows)


def prepare(df, edges):
    df = df.reset_index(drop=True).copy()
    df["kind"] = df.tool.astype(str) + ":" + df.category.astype(str)
    df["stepbin"] = np.clip(np.searchsorted(edges, df.step, side="right") - 1, 0, len(edges) - 2)
    df["editbin"] = pd.cut(df.n_edit_before, [-1, 3, 9, 10 ** 6], labels=False)
    df["sincebin"] = pd.cut(df.since_ins, [0, 1, 3, 10 ** 6], labels=False)
    df["last"] = df.prev_cmd.fillna(df.prev_kind)
    df["stratum"] = df.groupby(COV, dropna=False).ngroup()
    tasks, df["task_idx"] = np.unique(df.main_task, return_inverse=True)
    return df, len(tasks)


def gaps(df, mult, detail=False):
    base = mult[df.task_idx.to_numpy()].astype(float)
    real, y, s, strata = (df.domain == "real").to_numpy(), df.y.to_numpy(), df.score.to_numpy(), df.stratum.to_numpy()
    n = df.stratum.max() + 1
    cr, cs = np.bincount(strata, weights=base * real, minlength=n), np.bincount(strata, weights=base * ~real, minlength=n)
    support = (cr > 0) & (cs > 0)
    on = support[strata]
    w = np.zeros(len(df))
    for lab in (0, 1):
        sel = (y == lab) & ~real & on
        lab_strata = np.unique(strata[(y == lab) & on])
        tr, ts = cr[lab_strata].sum(), cs[lab_strata].sum()
        w[sel] = (cr[strata[sel]] / tr) / (cs[strata[sel]] / ts)

    def auc(mask, wt):
        k = mask & (wt > 0)
        return roc_auc_score(y[k], s[k], sample_weight=wt[k]) if len(set(y[k])) == 2 else np.nan

    g_all = auc(real, base) - auc(~real, base)
    g_sup = auc(real & on, base) - auc(~real & on, base)
    g_std = auc(real & on, base) - auc(~real & on, base * w)
    out = {"gap, all calls": g_all, "gap, common support (unweighted)": g_sup, "gap, common support (sim standardized)": g_std,
           "share removed by standardizing": (g_all - g_std) / g_all if g_all else np.nan,
           "auroc real (support)": auc(real & on, base), "auroc sim (support, standardized)": auc(~real & on, base * w),
           "calls outside support, real": float((base * (real & ~on)).sum() / (base * real).sum()),
           "calls outside support, sim": float((base * (~real & ~on)).sum() / (base * ~real).sum())}
    if not detail:
        return out
    diag = {}
    for lab in (0, 1):
        sel = (y == lab) & ~real & on
        ww = (base * w)[sel]
        norm = ww / ww.mean()
        diag[f"label {lab}"] = {"sim calls in support": int(sel.sum()), "effective sample size": round(float(ww.sum() ** 2 / (ww ** 2).sum()), 1),
                                "largest normalized weights": [round(float(x), 2) for x in np.sort(norm)[::-1][:5]],
                                "real calls in support": int(((y == lab) & real & on).sum())}
    return out, diag


def analyze(df, n_tasks):
    point, diag = gaps(df, np.ones(n_tasks), detail=True)
    attribution = {}
    for c in COV[1:]:
        for name, cols in ((f"only {c}", ["y", c]), (f"all but {c}", [x for x in COV if x != c])):
            attribution[name] = round(float(gaps(df.assign(stratum=df.groupby(cols, dropna=False).ngroup()), np.ones(n_tasks))[
                "share removed by standardizing"]), 4)
    rng = np.random.default_rng(SEED)
    draws = [gaps(df, np.bincount(rng.integers(0, n_tasks, n_tasks), minlength=n_tasks)) for _ in range(N_BOOT)]
    res = {k: {"value": round(float(v), 4), "ci95": [round(float(x), 4) for x in np.nanpercentile([d[k] for d in draws], [2.5, 97.5])]}
           for k, v in point.items()}
    res["bootstrap support"] = {f"calls outside support, {d} (median, max over draws)": [
        round(float(np.median([x[f"calls outside support, {d}"] for x in draws])), 4), round(float(max(x[f"calls outside support, {d}"] for x in draws)), 4)]
        for d in ("real", "sim")} | {"draws with an undefined gap": int(sum(np.isnan(x["gap, common support (sim standardized)"]) for x in draws))}
    return {"calls": {"real": int((df.domain == "real").sum()), "sim": int((df.domain == "sim").sum()),
                      "read-only real / sim": [int(((df.domain == "real") & (df.y == 0)).sum()), int(((df.domain == "sim") & (df.y == 0)).sum())],
                      "strata": int(df.stratum.nunique()), "main_tasks": n_tasks},
            "weights": diag, "share removed, one covariate or all but one (point)": attribution, **res}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    q = ex.table("qwen3-32b")
    hist = insert_history(set(q.key))
    ctx = pd.read_parquet(DATA / "context_features.parquet")[K + ["prev_cmd", "prev_kind", "n_edit_before"]]
    meta = q[["arm", "key", "step", "action", "tool", "category", "main_task", "cell"]].drop_duplicates(K)
    pre = pd.read_json(DATA / "tables" / "qwen3-32b__pre_call__attack_only.jsonl.gz", lines=True).query("train_set == 'real'")
    pre = pre.merge(hist, on=["key", "step", "action"], how="left", validate="one_to_one")
    edges = np.unique(np.quantile(pre[pre.n_ins > 0].step, [0, .25, .5, .75, 1]))  # step quartiles, all after-insert calls
    results = {"prediction (written before scoring)": (HERE / "predictions" / "behavior_matched_predictions.md").read_text(),
               "note": "Descriptive: the gap after standardizing measured trajectory characteristics within each label. "
                       "A removed share does not show that behavior causes the gap; a remaining gap does not show that text does.",
               "primary: rendering swap runs": {}, "secondary: all runs, raw sim": {}}
    out_lines = []

    # Primary: the swap's 100 runs (V1, V2) against the same cells' real runs
    base_file = sorted((ROOT / ".run_cache" / "q2_swap").glob("scores_2*.json"))[-1]
    base = json.loads(base_file.read_text())
    tables = {pos: DATA / "tables" / f"qwen3-32b__{pos}__attack_only.jsonl.gz" for pos in ("pre_call", "prev_content")}
    results["inputs"] = {"swap scores (primary)": [str(base_file.relative_to(ROOT)), sha(base_file)],
                         **{f"table {pos} (secondary)": [str(t.relative_to(ROOT)), sha(t)] for pos, t in tables.items()},
                         "context features": [str((DATA / "context_features.parquet").relative_to(ROOT)), sha(DATA / "context_features.parquet")],
                         "swap runs config": [str((SWAP / "config" / "swap_runs.json").relative_to(ROOT)), sha(SWAP / "config" / "swap_runs.json")]}
    results["probe layers per fold"] = {}
    info = json.loads((SWAP / "config" / "swap_runs.json").read_text())["runs"]
    first = {k: min((r["step"] for r in v["inserts"] if r["category"] in RENDERED), default=None) for k, v in info.items()}
    sw = pd.DataFrame(base["swap"])
    sw = sw[[first[k] is not None and s > first[k] for k, s in zip(sw.key, sw.step)]].assign(arm="sim-attack")
    cells = set(q[q.key.isin(set(sw.key))].cell)
    saved = pd.DataFrame(base["saved"])
    real = saved[(saved.arm == "real-attack-e1") & saved.key.isin(set(q[(q.domain == "real") & q.cell.isin(cells)].key))]
    for pos in ("pre_call", "prev_content"):
        x = sw[sw.position == pos]
        results["probe layers per fold"][f"primary {pos} (frozen swap probes; real side: the same refit's saved scores)"] = \
            {int(f): sorted(set(map(int, g.layer))) for f, g in x.groupby("fold")}
        r = real.assign(score=real[f"{pos}_score"], domain="real")[K + ["y", "score", "domain"]]
        for arm, col in (("V2 (insert outputs re-rendered with the note)", "rerendered"), ("V1 (as simulated)", "original")):
            s = sw[sw.position == pos].assign(score=lambda d: d[col], domain="sim")[K + ["y", "score", "domain"]]
            df = pd.concat([r, s]).merge(meta, on=K, how="left", validate="one_to_one").merge(ctx, on=K, how="left", validate="one_to_one") \
                .merge(hist, on=["key", "step", "action"], how="left", validate="one_to_one")
            df = df[df.n_ins > 0]
            df, n_tasks = prepare(df, edges)
            res = analyze(df, n_tasks)
            results["primary: rendering swap runs"][f"{pos} | {arm}"] = res
            out_lines.append((f"swap runs {pos} {arm}", res))

    # Secondary: all runs, raw sim
    for pos in ("pre_call", "prev_content"):
        t = pd.read_json(tables[pos], lines=True).query("train_set == 'real'")
        results["probe layers per fold"][f"secondary {pos} (64-layer headline picks)"] = {int(f): sorted(set(map(int, g.layer))) for f, g in t.groupby("fold")}
        df = t.merge(ctx, on=K, how="left", validate="one_to_one").merge(hist, on=["key", "step", "action"], how="left", validate="one_to_one")
        if df.n_ins.isna().any():
            raise ValueError(f"{pos}: {int(df.n_ins.isna().sum())} calls without insert history")
        df, n_tasks = prepare(df[df.n_ins > 0], edges)
        res = analyze(df, n_tasks)
        results["secondary: all runs, raw sim"][pos] = res
        out_lines.append((f"all runs {pos} raw sim", res))

    path = RESULTS / f"shift_behavior_matched_{now()}.json"
    path.write_text(json.dumps(results, indent=1))
    f = lambda v: f"{v['value']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
    for name, res in out_lines:
        print(f"{name}: calls {res['calls']} | gap all {f(res['gap, all calls'])}, support {f(res['gap, common support (unweighted)'])}, "
              f"standardized {f(res['gap, common support (sim standardized)'])}; removed {f(res['share removed by standardizing'])} | "
              f"ESS {[(d['effective sample size'], d['sim calls in support'], d['largest normalized weights'][0]) for d in res['weights'].values()]} | "
              f"{res['bootstrap support']}")
    print(path)


if __name__ == "__main__":
    main()
