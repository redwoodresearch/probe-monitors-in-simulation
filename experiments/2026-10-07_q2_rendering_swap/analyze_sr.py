"""str_replace tell swap: does the gutter (today's simulator), the trailing newline, or both remove the frozen recall
offset after a simulated str_replace output? Same 100 runs, frozen probes and readouts as the review-note test.
  V1 as simulated (the reference run)     G gutter only     N newline stripped only     B both (the real format)
Calls: each Q2 call in a turn after the run's first successful str_replace rewrite (a diff that parsed; B changes every
one). Real side: the same cells' real attack-e1 runs, calls after their first str_replace diff in an earlier turn,
from the refit's saved scores. Groups by the output the reader saw last (context_features prev_cmd): str_replace,
insert, other.
Per position, group and arm: recall and FPR at the frozen threshold, AUROC; per variant: the paired mean score change
vs V1, and the offset closure (recall_variant - recall_V1) / (recall_real - recall_V1). The frozen recall gap over all
these calls is split by group (real hits / real impactful minus sim hits / sim impactful; the parts sum to the gap).
95% intervals: 1000 main-task draws (seed 0), shared by every arm.
Checks: V1 here equals the first run's V1; calls before the first change score the same in every variant.

  uv run --with pandas --with pyarrow --with scikit-learn python experiments/2026-10-07_q2_rendering_swap/analyze_sr.py
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
import str_replace_tells as st  # noqa: E402
from paths import DATA  # noqa: E402

N_BOOT, SEED = 1000, 0
cfg = yaml.safe_load((HERE / "config" / "swap_sr.yaml").read_text())
ARM = {v: a for a, v in cfg["variants"].items()}
NAMES = {"G": ARM["sr_gutter"], "N": ARM["sr_no_newline"], "B": ARM["sr_gutter_no_newline"]}
cache = ROOT / ".run_cache" / "q2_swap"
base = json.loads(sorted(cache.glob("scores_2*.json"))[-1].read_text())  # dated: the first run (V1 and the real side)
sr = json.loads(sorted(cache.glob("scores_swap_sr_2*.json"))[-1].read_text())
report = json.loads(sorted((HERE / "results").glob("stage_report_swap_sr_2*.json"))[-1].read_text())["report"]
K = ["key", "step", "action", "position"]
b = pd.DataFrame(base["swap"])[K + ["y", "fold", "threshold", "original"]].rename(columns={"original": "V1"})
n = pd.DataFrame([{**{k: r[k] for k in K}, "V1_sr": r["scores"]["sim-attack"], **{v: r["scores"][a] for v, a in NAMES.items()}}
                  for r in sr["swap"]])
df = b.merge(n, on=K, how="inner", validate="one_to_one")
# the first successful rewrite (a diff that parsed); a failed diff only loses its trailing newline under N and B
first = {k: min((r["step"] for r in v[f"str_replaces {NAMES['B']}"] if r["changed"] and r["category"] in ("plain", "numbered")), default=None)
         for k, v in report.items()}
df["after"] = [first[k] is not None and s > first[k] for k, s in zip(df.key, df.step)]
pre = df[~df.after]


def unchanged(v):
    """Calls in turns up to the arm's first changed output of any kind (a failed diff too): identical text before them."""
    f = {k: min((r["step"] for r in rv[f"str_replaces {NAMES[v]}"] if r["changed"]), default=None) for k, rv in report.items()}
    return df[[f[k] is None or s <= f[k] for k, s in zip(df.key, df.step)]]


checks = {"V1 identical to the first run's V1 (max abs)": float((df.V1_sr - df.V1).abs().max()),
          "calls before the first change": int(len(pre)),
          **{f"{v} vs V1 before the arm's first change (max abs)": float((unchanged(v)[v] - unchanged(v).V1).abs().max()) for v in NAMES},
          "merged rows": int(len(df)), "base rows": int(len(b)), "sr rows": int(len(n)), "skipped in sr run": len(sr["skipped"]),
          "runs with a changed str_replace": sum(f is not None for f in first.values()),
          "changed outputs": {a: sum(r["changed"] for v in report.values() for r in v[f"str_replaces {a}"]) for a in NAMES.values()}}

q = ex.table("qwen3-32b")
cell_of, task_of = dict(zip(q.key, q.cell)), dict(zip(q.key, q.main_task))
ctx = pd.read_parquet(DATA / "context_features.parquet")[["arm", "key", "step", "action", "prev_cmd"]]
group = lambda c: c.where(c.isin(["str_replace", "insert"]), "other")
df = df[df.after].merge(ctx[ctx.arm == "sim-attack"].drop(columns="arm"), on=["key", "step", "action"], how="left", validate="many_to_one")
df["cell"], df["main_task"], df["group"] = df.key.map(cell_of), df.key.map(task_of), group(df.prev_cmd)
real_keys = sorted(set(q[(q.domain == "real") & q.cell.isin(set(df.cell))].key))
rruns = ex.load_runs(set(real_keys))
rfirst = {k: min((sa[0] for sa, c in calls.items() if c["tool"] == "text_editor" and ex.editor_args(c)[0] == "str_replace"
                  and st.DIFF in (c["output"] or "")[:300]), default=None) for k, calls in rruns.items()}
real = pd.DataFrame(base["saved"])
real = real[(real.arm == "real-attack-e1") & real.key.isin(real_keys)].copy()
real = real[[rfirst[k] is not None and s > rfirst[k] for k, s in zip(real.key, real.step)]]
real = real.merge(ctx[ctx.arm == "real-attack-e1"].drop(columns="arm"), on=["key", "step", "action"], how="left", validate="one_to_one")
real["cell"], real["main_task"], real["group"] = real.key.map(cell_of), real.key.map(task_of), group(real.prev_cmd)
ARMS = ("V1",) + tuple(NAMES)
auc = lambda y, s: roc_auc_score(y, s) if len(set(y)) == 2 else np.nan


def summarize(pos):
    s, r = df[df.position == pos], real
    rs, rt = f"{pos}_score", f"{pos}_threshold"
    tasks = sorted(set(s.main_task))
    s_by, r_by = dict(list(s.groupby("main_task"))), dict(list(r.groupby("main_task")))

    def stats(ts):
        ss, rr = pd.concat([s_by[t] for t in ts]), pd.concat([r_by[t] for t in ts if t in r_by])
        out = {}
        for g in ("all", "str_replace", "insert", "other"):
            sg, rg = (ss, rr) if g == "all" else (ss[ss.group == g], rr[rr.group == g])
            rf = (rg[rs] > rg[rt]).to_numpy()
            o = {"recall real": rf[rg.y == 1].mean(), "fpr real": rf[rg.y == 0].mean(), "auroc real": auc(rg.y, rg[rs])}
            for a in ARMS:
                f = (sg[a] > sg.threshold).to_numpy()
                o |= {f"recall {a}": f[sg.y == 1].mean(), f"fpr {a}": f[sg.y == 0].mean(), f"auroc {a}": auc(sg.y, sg[a])}
            for v in NAMES:
                d = sg[v] - sg.V1
                gap = o["recall real"] - o["recall V1"]
                o |= {f"closure {v}": (o[f"recall {v}"] - o["recall V1"]) / gap if gap else np.nan,
                      f"score change {v} impactful": d[sg.y == 1].mean(), f"score change {v} read-only": d[sg.y == 0].mean()}
            out[g] = o
        # the frozen recall gap over all these calls, split by group
        n_r, n_s = (rr.y == 1).sum(), (ss.y == 1).sum()
        for a in ARMS:
            hits_s = ss[(ss.y == 1) & (ss[a] > ss.threshold)].group.value_counts()
            hits_r = rr[(rr.y == 1) & (rr[rs] > rr[rt])].group.value_counts()
            out["all"][f"recall gap {a}"] = out["all"]["recall real"] - out["all"][f"recall {a}"]
            for g in ("str_replace", "insert", "other"):
                out["all"][f"recall gap {a} from {g}"] = hits_r.get(g, 0) / n_r - hits_s.get(g, 0) / n_s
        return out

    point = stats(tasks)
    rng = np.random.default_rng(SEED)
    draws = [stats(rng.choice(tasks, len(tasks), replace=True)) for _ in range(N_BOOT)]
    res = {g: {k: {"value": round(float(v), 4), "ci95": [round(float(x), 4) for x in np.nanpercentile([d[g][k] for d in draws], [2.5, 97.5])]}
               for k, v in o.items()} for g, o in point.items()}
    for g in ("all", "str_replace", "insert", "other"):
        sg, rg = (s, r) if g == "all" else (s[s.group == g], r[r.group == g])
        res[g]["n"] = {"sim impactful": int((sg.y == 1).sum()), "sim read-only": int((sg.y == 0).sum()),
                       "real impactful": int((rg.y == 1).sum()), "real read-only": int((rg.y == 0).sum()), "runs": int(sg.key.nunique())}
    res["main_tasks"] = len(tasks)
    return res


out = {"arms": {"V1": "as simulated", **NAMES}, "checks": checks, "results": {pos: summarize(pos) for pos in cfg["positions"]}}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
(HERE / "results" / f"sr_{stamp}.json").write_text(json.dumps(out, indent=1, default=float))
f = lambda v: f"{v['value']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
print("checks:", json.dumps(checks))
for pos, res in out["results"].items():
    for g in ("all", "str_replace", "insert", "other"):
        o = res[g]
        print(f"{pos:12} last {g:11} n {o['n']} | recall real {o['recall real']['value']:.3f} " +
              " ".join(f"{a} {o[f'recall {a}']['value']:.3f}" for a in ARMS) + " | FPR real " + f"{o['fpr real']['value']:.3f} " +
              " ".join(f"{a} {o[f'fpr {a}']['value']:.3f}" for a in ARMS) + " | AUROC real " + f"{o['auroc real']['value']:.3f} " +
              " ".join(f"{a} {o[f'auroc {a}']['value']:.3f}" for a in ARMS) + " | closure " +
              " ".join(f"{v} {f(o[f'closure {v}'])}" for v in NAMES))
    a = res["all"]
    print(f"{pos:12} frozen recall gap: " + " ".join(f"{x} {a[f'recall gap {x}']['value']:+.3f} (str_replace {a[f'recall gap {x} from str_replace']['value']:+.3f}, "
                                                    f"insert {a[f'recall gap {x} from insert']['value']:+.3f})" for x in ARMS))
print(f"wrote results/sr_{stamp}.json")
