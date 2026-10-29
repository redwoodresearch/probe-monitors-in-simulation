"""Stage 2 of the real->sim shift analysis: where does the gap sit?

Gap = real minus sim, in recall at the frozen 1% threshold and in AUROC (also FPR), for subsets of calls:
  (a) step quintile (edges from the view's pooled real and sim steps), and relative position in the run
      (step / the run's last step; attack arms only, where every call's step is known);
  (b) whether the previous tool output was a tool error (attack arms only: the Q1 tool-error table); first-turn calls,
      which have no previous output, are their own bucket;
  (d) tool, and environment.
95% intervals resample main tasks (1000 draws, seed 0), the same draws for every subset of one table.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from probe_shift.transfer import _weighted_metrics  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402

HERE = Path(__file__).parent
N_BOOT, SEED, CHUNK = 1000, 0, 100
ERRORS = ROOT / "experiments/2026-10-06_q1_segment_end/results/tool_errors_2026-10-06_134908.jsonl"
TF_TRAIN = {"real-attack-e1": "real", "sim-attack": "sim"}


def call_context():
    """Per attack call (arm, key, step, action): whether the output the reader saw last was a tool error (the last output
    of the previous turn; a turn's calls are all written before any of their outputs), whether the call is in the run's
    first turn (no previous output), and the run's last step."""
    e = pd.read_json(ERRORS, lines=True).sort_values(["arm", "key", "step", "action"])
    e["is_error"] = e.error.notna()
    last = e.groupby(["arm", "key", "step"]).is_error.last()  # each turn's last output
    prev = last.groupby(level=["arm", "key"]).shift(1).rename("prev_error")  # NaN in the first turn: nothing seen yet
    e = e.join(prev, on=["arm", "key", "step"])
    e["first_turn"] = e.prev_error.isna()
    e["prev_error"] = e.prev_error.fillna(False).astype(bool)
    e["last_step"] = e.groupby(["arm", "key"]).step.transform("max")
    return e[["arm", "key", "step", "action", "prev_error", "first_turn", "last_step"]]


def subsets(df, has_context):
    """name -> boolean mask over df's rows."""
    out = {"all": np.ones(len(df), bool)}
    q = pd.qcut(df.step, 5, labels=False, duplicates="drop")
    edges = pd.qcut(df.step, 5, duplicates="drop", retbins=True)[1]
    for b in sorted(q.unique()):
        out[f"a step Q{b + 1} [{edges[b]:.0f}, {edges[b + 1]:.0f}]"] = (q == b).to_numpy()
    if has_context:
        rel = df.step / df.last_step.clip(lower=1)
        for b, (lo, hi) in enumerate(zip(np.arange(0, 1, 0.2), np.arange(0.2, 1.2, 0.2))):
            out[f"a' relative position {lo:.1f}-{hi:.1f}"] = ((rel >= lo) & ((rel < hi) | (b == 4))).to_numpy()
        first = df.first_turn.to_numpy(bool)
        out["b previous output not an error"] = ~df.prev_error.to_numpy(bool) & ~first
        out["b previous output an error"] = df.prev_error.to_numpy(bool)
        out["b first turn (no previous output)"] = first
    for t in sorted(df.tool.unique()):
        out[f"d tool {t}"] = (df.tool == t).to_numpy()
    for e in sorted(df.env.unique()):
        out[f"d env {e}"] = (df.env == e).to_numpy()
    for c in sorted(df.category.unique()):
        out[f"d category {c}"] = (df.category == c).to_numpy()
    for t in ("bash", "text_editor"):  # is the step pattern the tool mix changing over the run?
        for b in sorted(q.unique()):
            out[f"x {t} step Q{b + 1}"] = ((df.tool == t) & (q == b)).to_numpy()
    return out


def analyze(table: Path, train: str, ctx: pd.DataFrame):
    df = pd.read_json(table, lines=True)
    if "command_text" in table.name:
        df = df[df.train_set.isin(TF_TRAIN)].assign(train_set=lambda x: x.train_set.map(TF_TRAIN))
    df = df[df.train_set == train].reset_index(drop=True)
    has_context = "attack_only" in table.name
    if has_context:
        n = len(df)
        df = df.merge(ctx, on=["arm", "key", "step", "action"], how="left", validate="one_to_one")
        if df.first_turn.isna().any() or len(df) != n:
            raise ValueError(f"{table.name}: {int(df.first_turn.isna().sum())} calls missing from the tool-error table")
    subs = subsets(df, has_context)
    uniq, inv = np.unique(df.main_task, return_inverse=True)
    rng = np.random.default_rng(SEED)
    counts = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(N_BOOT)])
    W_all = np.vstack([np.ones((1, len(uniq))), counts])
    y, s, flag = df.y.to_numpy(), df.score.to_numpy(np.float64), df.flag.to_numpy(float)
    dom = {d: (df.domain == d).to_numpy() for d in ("real", "sim")}
    acc = defaultdict(lambda: defaultdict(list))
    for k in range(0, len(W_all), CHUNK):
        Wc = W_all[k:k + CHUNK][:, inv].astype(float)
        for name, m in subs.items():
            for d, md in dom.items():
                i = np.flatnonzero(m & md)
                acc[name][d].append(_weighted_metrics(y[i], s[i], flag[i], Wc[:, i]))
    rep = {}
    for name, per in acc.items():
        res = {d: {met: np.concatenate([c[met] for c in chunks]) for met in chunks[0]} for d, chunks in per.items()}
        res["real minus sim"] = {met: res["real"][met] - res["sim"][met] for met in res["real"]}
        with np.errstate(all="ignore"):
            rep[name] = {d: {met: {"value": round(float(v[0]), 4),
                                   "ci95": [round(float(x), 4) for x in np.nanpercentile(v[1:], [2.5, 97.5])] if np.isfinite(v[1:]).any() else None}
                             for met, v in r.items()} for d, r in res.items()}
        for d, md in dom.items():
            i = subs[name] & md
            rep[name][d]["n"] = {"impactful": int(y[i].sum()), "read_only": int((y[i] == 0).sum()), "main_tasks": int(df.main_task[i].nunique())}
    reader, readout, view = (["Qwen3-32B", "command_text", "attack_only"] if "command_text" in table.name
                             else table.name.removesuffix(".jsonl.gz").split("__"))
    return {"reader": {"qwen3-32b": "Qwen3-32B", "qwen3.5": "Qwen3.5"}.get(reader, reader), "readout": readout, "view": view,
            "train": train, "subsets": rep}


if __name__ == "__main__":
    ctx = call_context()
    tables = sorted((DATA / "tables").glob("*.jsonl.gz"))
    tables = [t for t in tables if len(sys.argv) < 2 or any(a in t.name for a in sys.argv[1:])]
    rows = Parallel(n_jobs=-1)(delayed(analyze)(t, tr, ctx) for t in tables for tr in ("real", "sim"))
    stamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d_%H%M%S")
    out = RESULTS / f"shift_stage2_{stamp}.json"
    out.write_text(json.dumps({"n_boot": N_BOOT, "seed": SEED, "errors_table": str(ERRORS), "rows": rows}, indent=1))
    print(out)
