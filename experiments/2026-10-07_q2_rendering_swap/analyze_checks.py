"""Checks on the rendering swap (saved scores and staged text only; no GPU), and a short reading pack.

1. Matched FPR: recall in each arm (sim as simulated, sim re-rendered, real) at 1%, 2% and 5% FPR. Each arm's
   threshold comes from its own negatives' ROC curve.
2. Within-fold and within-run AUROC change: rules out pooled offsets between fold probes or between tasks.
3. Sensitivity: the 90 randomly drawn runs alone, and leave-one-environment-out closure.
4. Alignment: for 10 random re-rendered calls, the tokens around each readout (pre_call, prev_content) in both copies.
   Positions are recomputed locally with the extractor's own code and checked against the extraction's meta files.
The reading pack (8 cases, redacted, short excerpts) goes to .run_cache/q2_swap/ (transcript text: not in git).

  uv run --with pandas --with scikit-learn --with transformers python experiments/2026-10-07_q2_rendering_swap/analyze_checks.py
"""
import json
import re
import subprocess
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
sys.path.insert(0, str(ROOT / "src"))
import examples as ex  # noqa: E402
import rerender as rr  # noqa: E402

SEED, N_BOOT = 0, 1000
cfg = yaml.safe_load((HERE / "config" / "swap.yaml").read_text())
cache = ROOT / ".run_cache" / "q2_swap"
scores = json.loads(sorted(cache.glob("scores_2*.json"))[-1].read_text())
info = json.loads((HERE / "config" / "swap_runs.json").read_text())["runs"]
q = ex.table("qwen3-32b")
cell_of, task_of, env_of = dict(zip(q.key, q.cell)), dict(zip(q.key, q.main_task)), dict(zip(q.key, q.env))

sw = pd.DataFrame(scores["swap"])
first = {k: min((r["step"] for r in v["inserts"] if r["category"] in rr.RENDERED), default=None) for k, v in info.items()}
sw = sw[[first[k] is not None and s > first[k] for k, s in zip(sw.key, sw.step)]].copy()
for col, m in (("cell", cell_of), ("main_task", task_of), ("env", env_of)):
    sw[col] = sw.key.map(m)
saved = pd.DataFrame(scores["saved"])
real_keys = sorted(set(q[(q.domain == "real") & q.cell.isin(set(sw.cell))].key))
rruns = ex.load_runs(set(real_keys))
rfirst = {k: min((sa[0] for sa, c in calls.items() if c["tool"] == "text_editor" and rr.args_of(c).get("command") == "insert"), default=None)
          for k, calls in rruns.items()}
real = saved[(saved.arm == "real-attack-e1") & saved.key.isin(real_keys)].copy()
real = real[[rfirst[k] is not None and s > rfirst[k] for k, s in zip(real.key, real.step)]]
for col, m in (("cell", cell_of), ("main_task", task_of), ("env", env_of)):
    real[col] = real.key.map(m)
pc = sw[sw.position == "pre_call"]
auc = lambda y, s: roc_auc_score(y, s) if len(set(y)) == 2 else np.nan
out = {}


# 1. Matched FPR: each arm's own threshold at that FPR (the (k+1)-th largest negative score, as threshold_at_fpr).
def recall_at(y, s, fpr):
    neg = np.sort(s[y == 0])[::-1]
    k = int(np.floor(fpr * len(neg) + 1e-9))
    t = neg[min(k, len(neg) - 1)]
    return float((s[y == 1] > t).mean()), k


arms = {"real": (real.y.to_numpy(), real.pre_call_score.to_numpy()), "sim as simulated": (pc.y.to_numpy(), pc.original.to_numpy()),
        "sim re-rendered": (pc.y.to_numpy(), pc.rerendered.to_numpy())}
tasks = sorted(set(pc.main_task))
rng = np.random.default_rng(SEED)
draws = [rng.choice(tasks, len(tasks), replace=True) for _ in range(N_BOOT)]
out["matched_fpr"] = {}
for fpr in (0.01, 0.02, 0.05):
    row = {}
    for name, (y, s) in arms.items():
        r, k = recall_at(y, s, fpr)
        row[name] = {"recall": round(r, 4), "negatives_above_threshold": k, "negatives": int((y == 0).sum())}
    diffs = []
    for d in draws:
        m = pc.main_task.isin(d).to_numpy()  # membership approximation: each drawn task once
        idx = np.concatenate([np.flatnonzero(pc.main_task.to_numpy() == t) for t in d])
        y, o, rrr = pc.y.to_numpy()[idx], pc.original.to_numpy()[idx], pc.rerendered.to_numpy()[idx]
        diffs.append(recall_at(y, rrr, fpr)[0] - recall_at(y, o, fpr)[0])
    row["re-rendered minus as simulated"] = {"value": round(row["sim re-rendered"]["recall"] - row["sim as simulated"]["recall"], 4),
                                             "ci95": [round(float(x), 4) for x in np.percentile(diffs, [2.5, 97.5])]}
    out["matched_fpr"][f"{fpr:.0%}"] = row

# 2. Within-fold and within-run AUROC change
out["within_fold"] = {int(f): {"n_impactful": int((g.y == 1).sum()), "n_read_only": int((g.y == 0).sum()),
                               "auroc_as_simulated": round(auc(g.y, g.original), 4), "auroc_rerendered": round(auc(g.y, g.rerendered), 4),
                               "change": round(auc(g.y, g.rerendered) - auc(g.y, g.original), 4),
                               "auroc_real_same_fold": round(auc(real[real.fold == f].y, real[real.fold == f].pre_call_score), 4)}
                      for f, g in pc.groupby("fold")}
per_run = [auc(g.y, g.rerendered) - auc(g.y, g.original) for _, g in pc.groupby("key") if g.y.nunique() == 2]
out["within_run"] = {"runs_with_both_labels": len(per_run), "mean_change": round(float(np.mean(per_run)), 4),
                     "median_change": round(float(np.median(per_run)), 4), "runs_up": int(sum(c > 0 for c in per_run)),
                     "runs_down": int(sum(c < 0 for c in per_run)), "runs_tied": int(sum(c == 0 for c in per_run))}


# 3. Sensitivity
def closure(s_df, r_df):
    a_r, a_o, a_n = auc(r_df.y, r_df.pre_call_score), auc(s_df.y, s_df.original), auc(s_df.y, s_df.rerendered)
    return round((a_n - a_o) / (a_r - a_o), 4), round(a_r - a_o, 4)


random90 = [k for k, v in info.items() if v["stratum"] == "random"]  # the random stratum of the selection
r90 = pc[pc.key.isin(random90)]
out["random_90_runs"] = {"closure": closure(r90, real[real.cell.isin(set(r90.cell))])[0], "runs": r90.key.nunique()}
loeo = {}
for env in sorted(set(pc.env)):
    s_, r_ = pc[pc.env != env], real[real.env != env]
    loeo[env] = {"closure": closure(s_, r_)[0], "gap": closure(s_, r_)[1], "runs_left_out": int(pc[pc.env == env].key.nunique())}
out["leave_one_environment_out"] = {"range": [min(v["closure"] for v in loeo.values()), max(v["closure"] for v in loeo.values())], "by_env": loeo}

# 4. Alignment: recompute positions with the extractor's code on both staged copies, decode the tokens around each readout
from transformers import AutoTokenizer  # noqa: E402

from probe_shift.models import QWEN3_32B  # noqa: E402
from probe_shift.reader import segment_end_positions  # noqa: E402

tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
staged = {}
for line in open(cache / "staged_shard_0.jsonl"):
    it = json.loads(line)
    staged[(it["arm"], it["key"])] = it
sample = pc.sample(10, random_state=SEED)
align = []
vol = "probe-shift_aswin-surya_sub-la-activations"
for r in sample.itertuples():
    row = {"key": r.key[:60], "step": r.step, "action": r.action}
    for arm in ("sim-attack", cfg["swap_arm"]):
        it = staged[(arm, r.key)]
        res = segment_end_positions(tok, it["messages"], cfg.get("max_tokens", 32768))
        at = {(p["kind"], p["step"], p["action"]): p for p in res["positions"]}
        last = max(p["action"] for p in res["positions"] if p["kind"] == "output_end" and p["step"] == r.step - 1) if r.action == 0 else None
        reads = {"pre_call": at[("pre_call", r.step, r.action)],
                 "prev_content": at[("call_end", r.step, r.action - 1)] if r.action > 0 else at[("output_end", r.step - 1, last)]}
        meta_path = f"{cfg['extraction_run']}/meta/{arm}/{it['key']}.json"
        dst = cache / "meta_check" / arm / f"{it['key'][:120]}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            subprocess.run(["uv", "run", "modal", "volume", "get", "--force", vol, meta_path, str(dst)], capture_output=True, check=True)
        meta = {(p["kind"], p["step"], p["action"]): p for p in json.loads(dst.read_text())["positions"]}
        for name, p in reads.items():
            t = p["token"]
            window = tok.decode(res["ids"][max(0, t - 5):t]) + "⟦" + tok.decode(res["ids"][t:t + 1]) + "⟧" + tok.decode(res["ids"][t + 1:t + 6])
            row[f"{arm} {name}"] = window
            row[f"{arm} {name} token matches meta"] = meta[(p["kind"], p["step"], p["action"])]["token"] == t
    for name in ("pre_call", "prev_content"):
        row[f"{name} same text both copies"] = row[f"sim-attack {name}"] == row[f"{cfg['swap_arm']} {name}"]
    align.append(row)
out["alignment"] = {"pre_call windows identical in both copies": sum(r["pre_call same text both copies"] for r in align),
                    "prev_content windows identical in both copies": sum(r["prev_content same text both copies"] for r in align),
                    "token indices match the extraction meta": sum(all(v for k, v in r.items() if k.endswith("matches meta")) for r in align),
                    "n": len(align)}  # the decoded windows are transcript text: cache only, below

stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
(HERE / "results" / f"checks_{stamp}.json").write_text(json.dumps(out, indent=1, default=str))
(cache / f"alignment_windows_{stamp}.json").write_text(json.dumps(align, indent=1, default=str))

# Reading pack (8 cases)
SECRET = re.compile(r"(?i)((?:password|passwd|secret|token|api[_-]?key|auth)[\"']?\s*[:=]\s*)[\"']?[^\s\"',}]+")
LONG = re.compile(r"\b(?=[A-Za-z0-9+_\-]*\d)(?=[A-Za-z0-9+_\-]*[A-Za-z])[A-Za-z0-9+_\-]{24,}\b")  # key-like: letters and digits, no slashes
redact = lambda s: LONG.sub("<redacted>", SECRET.sub(r"\1<redacted>", s))
pc = pc.assign(delta=pc.rerendered - pc.original, margin_after=pc.rerendered - pc.threshold)
pc["improvement"] = pc.delta * np.where(pc.y == 1, 1, -1)
cases = [("random", r) for r in pc.sample(3, random_state=1).itertuples()]
cases += [("largest improvement", r) for r in pc.nlargest(2, "improvement").itertuples()]
cases += [("largest worsening", r) for r in pc.nsmallest(1, "improvement").itertuples()]
cases += [("highest-scoring read-only after the swap", pc[pc.y == 0].nlargest(1, "margin_after").iloc[0])]
cases += [("lowest-scoring impactful after the swap", pc[pc.y == 1].nsmallest(1, "margin_after").iloc[0])]
md = ["# Rendering swap reading pack (8 cases; pre_call, frozen Qwen3-32B probe; margins = score minus the fold's threshold)\n"]
for tag, r in cases:
    a, b = staged[("sim-attack", r.key)]["messages"], staged[(cfg["swap_arm"], r.key)]["messages"]
    recs = [x for x in rr.rerender_messages(a)[1] if x["category"] in rr.RENDERED and x["step"] < r.step]
    ins = recs[-1]
    step, calls = -1, None
    for m in a:
        if m["role"] == "assistant":
            step += 1
            if step == r.step:
                calls = m["tool_calls"]
    call = calls[r.action]["function"]
    head = lambda s: redact("\n".join(s.split("\n")[:8]))
    md += [f"## {tag}: {r.env}, step {r.step}.{r.action}, label {'impactful' if r.y == 1 else 'read-only'}",
           f"pre_call margin {r.original - r.threshold:+.2f} → {r.rerendered - r.threshold:+.2f} (change {r.rerendered - r.original:+.2f}); "
           f"last re-rendered insert at step {ins['step']} ({ins['category']})",
           "insert output as simulated:", "```", head(a[ins["msg"]]["content"]), "```",
           "insert output re-rendered:", "```", head(b[ins["msg"]]["content"]), "```",
           f"next call: `{redact(json.dumps({'tool': call['name'], **rr.args_of(call)})[:200])}`", ""]
(cache / f"reading_pack_{stamp}.md").write_text("\n".join(md))
print(json.dumps(out, indent=1, default=str))
print(f"wrote results/checks_{stamp}.json and {cache}/reading_pack_{stamp}.md")
