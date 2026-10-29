"""Review-note test: closure of the after-insert pre_call AUROC gap under each insert-output variant, against V1 (as
simulated), with the same real reference, subsets and bootstrap as analyze_swap.py.
  V2 full re-render (snippet + note)   V3 snippet without the note   V4 diff with the note appended
  V5 diff without its trailing newline (V4's control: V4 - V5 is the note in diff format; V2 - V3 the note in snippets)
Same main-task draws for every variant (1000, seed 0), so differences between variants get paired intervals.
Checks:
- V1 here equals the first run's V1: the same activations.
- Calls before the first changed output score the same in V3/V4 (this run: early stop after layer 17) as in V1 (the
  first run: all 64 layers). This checks the layer-subset extraction.
Matched-FPR recall per variant at 1%, 2% and 5%.

  uv run --with pandas --with scikit-learn python experiments/2026-10-07_q2_rendering_swap/analyze_note.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2026-10-06_q2_impactful_call_transfer" / "shift"))
import examples as ex  # noqa: E402
import rerender as rr  # noqa: E402

N_BOOT, SEED = 1000, 0
cfg = yaml.safe_load((HERE / "config" / "swap_note.yaml").read_text())
cache = ROOT / ".run_cache" / "q2_swap"
base = json.loads(sorted(cache.glob("scores_2*.json"))[-1].read_text())
note = json.loads(sorted(cache.glob("scores_swap_note_2*.json"))[-1].read_text())  # dated: not the V5 run's files
info = json.loads((HERE / "config" / "swap_runs.json").read_text())["runs"]
K = ["key", "step", "action", "position"]
arm3, arm4 = [a for a, v in cfg["variants"].items() if v == "snippet_no_note"][0], [a for a, v in cfg["variants"].items() if v == "diff_with_note"][0]
b = pd.DataFrame(base["swap"])[K + ["y", "fold", "threshold", "original", "rerendered"]].rename(columns={"original": "V1", "rerendered": "V2"})
n = pd.DataFrame([{**{k: r[k] for k in K}, "V1_note": r["scores"]["sim-attack"], "V3": r["scores"][arm3], "V4": r["scores"][arm4]} for r in note["swap"]])
df = b.merge(n, on=K, how="inner", validate="one_to_one")
# V5 (the diff without its trailing newline): the control for V4, from its own run
cfg5 = yaml.safe_load((HERE / "config" / "swap_note_v5.yaml").read_text())
arm5 = list(cfg5["variants"])[0]
v5 = json.loads(sorted(cache.glob("scores_swap_note_v5_2*.json"))[-1].read_text())
n5 = pd.DataFrame([{**{k: r[k] for k in K}, "V1_v5": r["scores"]["sim-attack"], "V5": r["scores"][arm5]} for r in v5["swap"]])
df = df.merge(n5, on=K, how="inner", validate="one_to_one")
first = {k: min((r["step"] for r in v["inserts"] if r["category"] in rr.RENDERED), default=None) for k, v in info.items()}
df["after"] = [first[k] is not None and s > first[k] for k, s in zip(df.key, df.step)]
pre = df[~df.after]
checks = {"V1 identical to the first run's V1": float((df.V1_note - df.V1).abs().max()),
          "calls before the first change": int(len(pre)),
          "V3 vs V1 before the first change (max abs, cross-run, layer-subset extraction)": float((pre.V3 - pre.V1).abs().max()),
          "V4 vs V1 before the first change": float((pre.V4 - pre.V1).abs().max()),
          "V5 vs V1 before the first change": float((pre.V5 - pre.V1).abs().max()),
          "V1 in the V5 run identical to the first run's V1": float((df.V1_v5 - df.V1).abs().max()),
          "merged rows": int(len(df)), "base rows": int(len(b)), "note rows": int(len(n)), "skipped in note run": len(note["skipped"])}

q = ex.table("qwen3-32b")
cell_of, task_of = dict(zip(q.key, q.cell)), dict(zip(q.key, q.main_task))
df = df[df.after].copy()
df["cell"], df["main_task"] = df.key.map(cell_of), df.key.map(task_of)
df["any_failed"] = df.key.map(lambda k: info[k]["any_failed"])
df["all_exact"] = df.key.map(lambda k: info[k]["all_exact"])
df["numbering_ok"] = [all(r["category"] in ("exact", "approx_window") for r in info[k]["inserts"] if r["step"] < s and r["category"] != "not_rendered")
                      for k, s in zip(df.key, df.step)]
saved = pd.DataFrame(base["saved"])
real_keys = sorted(set(q[(q.domain == "real") & q.cell.isin(set(df.cell))].key))
rruns = ex.load_runs(set(real_keys))
rfirst = {k: min((sa[0] for sa, c in calls.items() if c["tool"] == "text_editor" and rr.args_of(c).get("command") == "insert"), default=None)
          for k, calls in rruns.items()}
real = saved[(saved.arm == "real-attack-e1") & saved.key.isin(real_keys)].copy()
real = real[[rfirst[k] is not None and s > rfirst[k] for k, s in zip(real.key, real.step)]]
real["cell"], real["main_task"] = real.key.map(cell_of), real.key.map(task_of)
auc = lambda y, s: roc_auc_score(y, s) if len(set(y)) == 2 else np.nan
VARIANTS = ("V2", "V3", "V4", "V5")


def recall_at(y, s, fpr):
    neg = np.sort(s[y == 0])[::-1]
    t = neg[min(int(np.floor(fpr * len(neg) + 1e-9)), len(neg) - 1)]
    return float((s[y == 1] > t).mean())


def summarize(sub, pos):
    s = sub[sub.position == pos]
    r = real[real.cell.isin(set(s.cell))]
    tasks = sorted(set(s.main_task))
    s_by, r_by = {t: g for t, g in s.groupby("main_task")}, {t: g for t, g in r.groupby("main_task")}
    rcol = f"{pos}_score"

    def stats(ts):
        ss = pd.concat([s_by[t] for t in ts])
        rs = pd.concat([r_by[t] for t in ts if t in r_by])
        a_real, a1 = auc(rs.y, rs[rcol]), auc(ss.y, ss.V1)
        out = {"auroc_real": a_real, "auroc_V1": a1}
        for v in VARIANTS:
            a = auc(ss.y, ss[v])
            d = ss[v] - ss.V1
            out |= {f"auroc_{v}": a, f"closure_{v}": (a - a1) / (a_real - a1) if a_real != a1 else np.nan, f"auroc_change_{v}": a - a1,
                    f"read_only_change_{v}": float(d[ss.y == 0].mean()), f"impactful_change_{v}": float(d[ss.y == 1].mean())}
        # The four contrasts (closure points; AUROC differences are the same contrasts without the ratio's noise)
        out["note in snippet format (V2 - V3)"] = out["closure_V2"] - out["closure_V3"]
        out["note in diff format (V4 - V5)"] = out["closure_V4"] - out["closure_V5"]
        out["trailing newline alone (V5 - V1)"] = out["closure_V5"]
        out["snippet format without note (V3 - V1)"] = out["closure_V3"]
        out["auroc note in snippet format (V2 - V3)"] = out["auroc_V2"] - out["auroc_V3"]
        out["auroc note in diff format (V4 - V5)"] = out["auroc_V4"] - out["auroc_V5"]
        return out

    point = stats(tasks)
    rng = np.random.default_rng(SEED)
    draws = [stats(rng.choice(tasks, len(tasks), replace=True)) for _ in range(N_BOOT)]
    res = {k: {"value": round(float(v), 4), "ci95": [round(float(x), 4) for x in np.nanpercentile([d[k] for d in draws], [2.5, 97.5])]}
           for k, v in point.items()}
    res["matched_fpr_recall"] = {f"{f:.0%}": {"real": round(recall_at(r.y.to_numpy(), r[rcol].to_numpy(), f), 4),
                                              **{v: round(recall_at(s.y.to_numpy(), s[v].to_numpy(), f), 4) for v in ("V1",) + VARIANTS},
                                              "negatives_above (sim)": int(np.floor(f * (s.y == 0).sum() + 1e-9))}
                                 for f in (0.01, 0.02, 0.05)}
    res["n"] = {"sim impactful": int((s.y == 1).sum()), "sim read-only": int((s.y == 0).sum()), "runs": int(s.key.nunique()), "main_tasks": len(tasks)}
    return res


subsets = {"all": df, "runs with no failed insert": df[~df.any_failed], "only exact or approx_window inserts before the call": df[df.numbering_ok],
           "after an approx_numbering or failed insert": df[~df.numbering_ok], "all-exact runs (too few to carry the result)": df[df.all_exact]}
out = {"checks": checks, "results": {name: {pos: summarize(sub, pos) for pos in cfg["positions"]} for name, sub in subsets.items() if len(sub)}}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
(HERE / "results" / f"note_{stamp}.json").write_text(json.dumps(out, indent=1, default=float))
f = lambda v: f"{v['value']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
print("checks:", json.dumps(checks))
for name, by_pos in out["results"].items():
    for pos, r in by_pos.items():
        print(f"{name[:40]:40} {pos:12} n {r['n']} | closure V2 {f(r['closure_V2'])} V3 {f(r['closure_V3'])} V4 {f(r['closure_V4'])} V5 {f(r['closure_V5'])} | "
              f"note/snippet {f(r['note in snippet format (V2 - V3)'])} note/diff {f(r['note in diff format (V4 - V5)'])} | "
              f"read-only change V2 {r['read_only_change_V2']['value']:+.2f} V3 {r['read_only_change_V3']['value']:+.2f} V4 {r['read_only_change_V4']['value']:+.2f} V5 {r['read_only_change_V5']['value']:+.2f}")
print(f"wrote results/note_{stamp}.json")
