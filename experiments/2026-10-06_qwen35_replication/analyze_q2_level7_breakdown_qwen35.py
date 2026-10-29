"""Where does Qwen3.5's level-7 pre_call residual sit (2026-10-07)?

At text level 7 the recalibrated recall gap (real minus sim) of pre_call trained on real is +0.046 (q2's cleaned-text
test, `analyze_q2_cleaned_text_qwen35.py`). Here it is broken down, on the same calls (common to levels 0, 4 and 7),
headline layers and 1000 main-task draws, by
  (a) whether the run has shown an insert yet: a text_editor insert in an earlier step (none / at least one);
  (b) step quintile (edges from the pooled real and sim steps, as q2's stage2.py);
  (c) the call's tool, and the kind of call whose output the reader saw last (the previous step's last call);
  (d) environment.
Each subset's recall is taken at its domain's own per-fold 1% threshold (recalibrated on that domain's read-only test
calls, recomputed in every draw), so subsets are compared at one operating point; its AUROC gap is threshold-free.
For each partition, a subset's contribution to the overall gap is (its share of real's impactful calls x its real
recall) minus (the same for sim); the contributions add up to the overall gap. Saved scores only, no compute on Modal.

  uv run --with pandas python experiments/2026-10-06_qwen35_replication/analyze_q2_level7_breakdown_qwen35.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_q2_cleaned_text_qwen35 as ct  # noqa: E402

LEVEL, POSITIONS, TRAIN = 7, ["pre_call", "prev_content"], "real"  # pre_call reads the <tool_call> template token: reported
# only next to prev_content, the content-token readout before it (the project rule for template-token readouts)


def context(labels: Path) -> pd.DataFrame:
    """Per attack call: tool, environment, earlier-step inserts in the run, and the kind of the previous step's last call."""
    rows = sorted((r for r in map(json.loads, labels.read_text().splitlines()) if r["arm"] in ct.TRAIN),
                  key=lambda r: (r["arm"], r["key"], r["step"], r["action"]))
    cmd = lambda r: (r["arguments"] if isinstance(r["arguments"], dict) else {}).get("command")

    def kind(r):
        if r is None:
            return "first"
        if r["tool"] != "text_editor":
            return "bash-like" if "bash" in r["tool"] or r["tool"].endswith("_restart") else "other"
        return {"view": "view", "create": "create", "str_replace": "edit", "insert": "insert"}.get(cmd(r), "other editor")

    out, run, last_of_step, inserts_by_step = [], None, {}, {}
    for r in rows:
        if (r["arm"], r["key"]) != run:
            run, last_of_step, inserts_by_step = (r["arm"], r["key"]), {}, {}
        earlier = [st for st in last_of_step if st < r["step"]]  # a text-only turn has no calls: the output seen last is
        seen = last_of_step[max(earlier)] if earlier else None  # the last call of the latest earlier step that has one
        out.append({"arm": r["arm"], "key": r["key"], "step": r["step"], "action": r["action"], "tool": r["tool"],
                    "env": r["environment"], "out_kind": kind(seen),
                    "inserts_before": sum(n for s, n in inserts_by_step.items() if s < r["step"])})
        last_of_step[r["step"]] = r
        inserts_by_step[r["step"]] = inserts_by_step.get(r["step"], 0) + (r["tool"] == "text_editor" and cmd(r) == "insert")
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ct.ROOT / ".run_cache" / "q2_call_labels" / "2026-10-06_212047.jsonl"))
    a = ap.parse_args()
    vol = modal.Volume.from_name(ct.FEATS, version=2)
    ctx = context(Path(a.labels))
    results = {}
    for position in POSITIONS:
        tabs = {lv: ct.level_table(vol, lv, position) for lv in ct.LEVELS}
        common = set.intersection(*[set(map(tuple, t[ct.K].drop_duplicates().to_numpy())) for t in tabs.values()])
        t = tabs[LEVEL]
        t = t[(t.train_set == TRAIN) & pd.Series(list(map(tuple, t[ct.K].to_numpy())), index=t.index).isin(common)]
        t = t.sort_values(ct.K).reset_index(drop=True).merge(ctx, on=ct.K, how="left", validate="one_to_one")
        if t.env.isna().any():
            raise ValueError(f"{int(t.env.isna().sum())} calls missing from the label table")
        q, edges = pd.qcut(t.step, 5, labels=False, duplicates="drop", retbins=True)
        parts = {
            "(a) inserts shown before this call": {"none": t.inserts_before == 0, "at least one": t.inserts_before > 0},
            "(b) step quintile": {f"Q{b + 1} [{edges[b]:.0f}, {edges[b + 1]:.0f}]": q == b for b in sorted(q.unique())},
            "(c) tool of the call": {k: t.tool == k for k in sorted(t.tool.unique())},
            "(c) kind of the previous output": {k: t.out_kind == k for k in sorted(t.out_kind.unique())},
            "(d) environment": {k: t.env == k for k in sorted(t.env.unique())},
        }
        y, s, fold = t.y.to_numpy(), t.score.to_numpy(np.float64), t.fold.to_numpy()
        real, sim = (t.domain == "real").to_numpy(), (t.domain == "sim").to_numpy()
        counts, inv = ct.task_counts(t.main_task.to_numpy())
        W_all = np.vstack([np.ones((1, counts.shape[1])), counts])  # row 0: the point estimate
        hit = np.zeros((len(W_all), len(t)), bool)  # per draw: above its domain's per-fold recalibrated threshold
        for k in range(0, len(W_all), ct.CHUNK):
            Wc = W_all[k:k + ct.CHUNK][:, inv].astype(float)
            for dom in (real, sim):
                for f in np.unique(fold):
                    nf, rows = dom & (y == 0) & (fold == f), np.flatnonzero(dom & (fold == f))
                    thr = ct.wthreshold(s[nf], Wc[:, nf])
                    hit[k:k + ct.CHUNK][:, rows] = s[rows][None, :] > thr[:, None]
        W = W_all[:, inv].astype(float)
        pos = y == 1
        rec = lambda m: (W[:, m & pos] * hit[:, m & pos]).sum(1) / W[:, m & pos].sum(1)
        part = lambda m, dom: (W[:, m & pos & dom] * hit[:, m & pos & dom]).sum(1) / W[:, pos & dom].sum(1)  # defined when m is empty
        S = lambda x: {"value": round(float(x[0]), 4), "ci95": [round(float(v), 4) for v in np.nanpercentile(x[1:], [2.5, 97.5])]}
        overall = rec(real) - rec(sim)
        out = {"overall recalibrated recall gap": S(overall)}
        for pname, subs in parts.items():
            res = {}
            for name, m in subs.items():
                m = m.to_numpy()
                with np.errstate(invalid="ignore", divide="ignore"):
                    rr, rs = rec(m & real), rec(m & sim)
                    contrib = part(m, real) - part(m, sim)  # hits in the subset over all of the domain's impactful calls
                    au = {d: ct._weighted_metrics(y[m & dm], s[m & dm], hit[0, m & dm].astype(float), W[:, m & dm])["auroc"]
                          for d, dm in (("real", real), ("sim", sim))}
                res[name] = {"n": {"real impactful": int((m & real & pos).sum()), "sim impactful": int((m & sim & pos).sum()),
                                   "real read-only": int((m & real & ~pos).sum()), "sim read-only": int((m & sim & ~pos).sum())},
                             "recall real": S(rr), "recall sim": S(rs), "gap": S(rr - rs), "contribution to overall gap": S(contrib),
                             "auroc gap": S(au["real"] - au["sim"])}
            out[pname] = res

        results[position] = out
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q2_level7_breakdown_qwen35_{stamp}.json"
    path.write_text(json.dumps({"level": LEVEL, "positions": POSITIONS, "train": TRAIN, "calls": len(t), "labels": a.labels,
                                "volume": ct.FEATS, "features_run": ct.FR, "n_boot": ct.N_BOOT, "seed": ct.SEED, "fpr": ct.FPR,
                                "layer_tolerance": ct.TOL, "layers": ct.LAYERS, "levels_common": ct.LEVELS, **results}, indent=1))
    for position, out in results.items():
        g = out["overall recalibrated recall gap"]
        print(f"== {position}: overall {g['value']:+.3f} {g['ci95']}")
        _show(out)
    print(f"-> {path}")


def _show(out):
    for pname, res in out.items():
        if pname.startswith("("):
            print(pname)
            for name, r in res.items():
                print(f"  {name[:34]:34} n_imp {r['n']['real impactful']:5}/{r['n']['sim impactful']:5}  gap {r['gap']['value']:+.3f} "
                      f"[{r['gap']['ci95'][0]:+.3f},{r['gap']['ci95'][1]:+.3f}]  contrib {r['contribution to overall gap']['value']:+.3f}  "
                      f"auroc gap {r['auroc gap']['value']:+.3f}")


if __name__ == "__main__":
    main()
