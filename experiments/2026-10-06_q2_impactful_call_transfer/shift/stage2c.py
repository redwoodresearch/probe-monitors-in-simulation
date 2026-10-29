"""Stage 2(c): the real->sim gap split by whether the previous tool output matched the simulator's reference.

Each call's previous tool output (the output of the call just before it in the run) carries Q1's flag against the
cell's reference trajectory: exact (copied), near (>= 0.9 similar) or none (invented). Real outputs are flagged against
the same reference, which gives the rate at which a real environment reproduces it. Gap = real minus sim, in recall at
the frozen 1% threshold, FPR and AUROC. Two comparisons per bin: sim calls in the bin vs real calls in the same bin, and
sim calls in the bin vs all real calls. Attack view, trained on real. 95% intervals resample main tasks (1000 draws,
seed 0).
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
FLAGS = ROOT / "experiments/2026-10-06_q1_copied_vs_invented/.run_cache/flags_2026-10-07_121021.jsonl.gz"  # also on S3
ERRORS = ROOT / "experiments/2026-10-06_q1_segment_end/results/tool_errors_2026-10-06_134908.jsonl"
K = ["arm", "key", "step", "action"]


def previous_flags():
    """Per call: the tool and flag of the output the reader saw last, the previous turn's last output (`first` if the call
    is in the run's first turn, `unflagged` if that output is not among Q1's kept outputs)."""
    order = pd.read_json(ERRORS, lines=True)[K + ["tool"]].sort_values(K)
    turns = order.groupby(["arm", "key", "step"]).agg(last_action=("action", "last"), last_tool=("tool", "last")).reset_index()
    g = turns.groupby(["arm", "key"])  # the output seen last is the previous turn's last (a turn's calls precede its outputs)
    turns["prev_step"], turns["prev_action"], turns["prev_tool"] = g.step.shift(1), g.last_action.shift(1), g.last_tool.shift(1)
    order = order.merge(turns[["arm", "key", "step", "prev_step", "prev_action", "prev_tool"]], on=["arm", "key", "step"], validate="many_to_one")
    flags = pd.read_json(FLAGS, lines=True)[K + ["out_match"]].rename(columns={"step": "prev_step", "action": "prev_action"})
    out = order.merge(flags, on=["arm", "key", "prev_step", "prev_action"], how="left", validate="many_to_one")  # a turn's calls share one previous output
    out["prev_flag"] = np.where(out.prev_step.isna(), "first", out.out_match.fillna("unflagged"))
    return out[K + ["prev_tool", "prev_flag"]]


def analyze(table: Path, prev: pd.DataFrame):
    df = pd.read_json(table, lines=True).query("train_set == 'real'").reset_index(drop=True)
    n = len(df)
    df = df.merge(prev, on=K, how="left", validate="one_to_one")
    if len(df) != n or df.prev_flag.isna().any():
        raise ValueError(f"{table.name}: {int(df.prev_flag.isna().sum())} calls without a previous-output flag")
    real, sim = (df.domain == "real").to_numpy(), (df.domain == "sim").to_numpy()
    flag, editor = df.prev_flag.to_numpy(), (df.prev_tool == "text_editor").to_numpy()
    subs = {"all": (real, sim)}
    for b in ("first", "exact", "near", "none", "unflagged"):
        subs[f"previous output {b}: same bin"] = (real & (flag == b), sim & (flag == b))
        subs[f"previous output {b}: sim bin vs all real"] = (real, sim & (flag == b))
    for b in ("exact", "near", "none"):
        subs[f"previous text_editor output {b}: same bin"] = (real & editor & (flag == b), sim & editor & (flag == b))
        subs[f"previous bash-like output {b}: same bin"] = (real & ~editor & (flag == b), sim & ~editor & (flag == b))
    uniq, inv = np.unique(df.main_task, return_inverse=True)
    rng = np.random.default_rng(SEED)
    counts = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(N_BOOT)])
    W_all = np.vstack([np.ones((1, len(uniq))), counts])
    y, s, fl = df.y.to_numpy(), df.score.to_numpy(np.float64), df.flag.to_numpy(float)
    acc = defaultdict(lambda: defaultdict(list))
    for k in range(0, len(W_all), CHUNK):
        Wc = W_all[k:k + CHUNK][:, inv].astype(float)
        for name, (mr, ms) in subs.items():
            for d, m in (("real", mr), ("sim", ms)):
                i = np.flatnonzero(m)
                acc[name][d].append(_weighted_metrics(y[i], s[i], fl[i], Wc[:, i]))
    rep = {}
    for name, per in acc.items():
        res = {d: {met: np.concatenate([c[met] for c in chunks]) for met in chunks[0]} for d, chunks in per.items()}
        res["real minus sim"] = {met: res["real"][met] - res["sim"][met] for met in res["real"]}
        with np.errstate(all="ignore"):
            rep[name] = {d: {met: {"value": round(float(v[0]), 4),
                                   "ci95": [round(float(x), 4) for x in np.nanpercentile(v[1:], [2.5, 97.5])] if np.isfinite(v[1:]).any() else None}
                             for met, v in r.items()} for d, r in res.items()}
        for d, m in zip(("real", "sim"), subs[name]):
            rep[name][d]["n"] = {"impactful": int(y[m].sum()), "read_only": int((y[m] == 0).sum())}
    reader, readout, view = table.name.removesuffix(".jsonl.gz").split("__")
    return {"reader": {"qwen3-32b": "Qwen3-32B", "qwen3.5": "Qwen3.5"}[reader], "readout": readout, "view": view,
            "prev_flag_counts": {f"{a}|{b}": int(v) for (a, b), v in df.groupby(["domain", "prev_flag"]).size().items()},
            "subsets": rep}


if __name__ == "__main__":
    prev = previous_flags()
    tables = [t for t in sorted((DATA / "tables").glob("*__attack_only.jsonl.gz")) if "command_text" not in t.name]
    rows = Parallel(n_jobs=-1)(delayed(analyze)(t, prev) for t in tables)
    stamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d_%H%M%S")
    out = RESULTS / f"shift_stage2c_{stamp}.json"
    out.write_text(json.dumps({"n_boot": N_BOOT, "seed": SEED, "flags": str(FLAGS), "rows": rows}, indent=1, default=str))
    print(out)
