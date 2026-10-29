"""Leakage check on the retry's stage-3 pass (review thread on #574; $0, stored files only).

A held-out cue could share a label-aligned word with the training cues (the simulated card cue starts "Every card
number…", and a simulated training cue says "state is reset after every session"). For each readout's chosen layer:
the held-out AUROC with all four families, per family, and without the card family; beside each, the stored
whole-pipeline null p95 (all four families only: the fit files keep no per-draw held-out scores) and a free null on
the stored scores, flipping the labels of each held-out task's pairs together (2,000 draws, seed 20261011).
Word scan (first): per pair, the words only one side's differing message has; a held-out word is "aligned" if training pairs
(train and dev) show it on the same label's side, and "opposed" if on the other label's side. The AUROCs are also given on only the
families with no aligned word.

  uv run --no-project --with scikit-learn python experiments/2026-10-11_fake_env_concept_probe/analyze_concept_leakage.py --fit-c ../2026-10-10_eval_awareness_probe/results/fit_<C>.json --fit-t ../2026-10-10_eval_awareness_probe/results/fit_<T>.json --build results/concept_sets_v2_<stamp>.json
"""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
B, SEED, CARD = 2000, 20261011, "test_card_numbers"
ap = argparse.ArgumentParser()
ap.add_argument("--fit-c", required=True)
ap.add_argument("--fit-t", required=True)
ap.add_argument("--build", required=True)
a = ap.parse_args()
build = json.loads((HERE / a.build).read_text())
raw = (ROOT / ".run_cache" / "fake_env_concept_probe" / build["file"]).read_bytes()
fits = {"C": json.loads((HERE / a.fit_c).read_text()), "T": json.loads((HERE / a.fit_t).read_text())}
if hashlib.sha256(raw).hexdigest() != build["sha256"] or any(f["sets_sha256"] != build["sha256"] for f in fits.values()):
    raise ValueError("the set file or the fits are not this build")
items = [json.loads(x) for x in raw.decode().splitlines()]
by_id = {it["id"]: it for it in items}
out = {"reading": "post hoc check, review thread on #574", "fits": {"C": a.fit_c, "T": a.fit_t}, "build": a.build,
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()}
words = lambda t: set(re.findall(r"[a-z0-9]+", t.lower()))
pairs = {}
for it in items:
    pairs.setdefault(it["pair"], {})[it["label"]] = it
train = {0: Counter(), 1: Counter()}
held = {}
for pid, p in pairs.items():
    a1, a0 = p[1]["messages"], p[0]["messages"]
    j = [i for i, (u, v) in enumerate(zip(a1, a0)) if u != v][0]
    w1, w0 = words(json.dumps(a1[j])), words(json.dumps(a0[j]))
    only = {1: w1 - w0, 0: w0 - w1}
    if p[1]["split"] == "heldout":
        held[pid] = (p[1]["mechanism"], only)
    else:
        for lab in (0, 1):
            train[lab].update(only[lab])
scan = {}
for pid, (mm, only) in held.items():
    d = scan.setdefault(mm, {"aligned": Counter(), "opposed": Counter()})
    for lab in (0, 1):
        for w in only[lab]:
            if train[lab][w]:
                d["aligned"][f"{w} (label {lab}: {train[lab][w]} train+dev pairs on this side, {train[1 - lab][w]} on the other)"] += 1
            if train[1 - lab][w]:
                d["opposed"][f"{w} (label {lab}: {train[1 - lab][w]} train+dev pairs on the other side, {train[lab][w]} on this)"] += 1
out["word scan (held-out pairs with each word)"] = {mm: {kk: dict(v.most_common()) for kk, v in d.items()} for mm, d in sorted(scan.items())}
clean = sorted(mm for mm, d in scan.items() if not d["aligned"])  # families with no label-aligned shared word
out["families with no aligned word"] = clean
for k, fit in fits.items():
    s = fit["heldout_scores_at_chosen_layer"]["per_token"]
    y, sc = np.array([x["label"] for x in s]), np.array([x["score"] for x in s])
    mech = np.array([x["mechanism"] for x in s])
    task = np.array([by_id[x["id"]]["task"] for x in s])
    tasks = sorted(set(task))
    coins = np.random.default_rng(SEED).integers(0, 2, (B, len(tasks))).astype(bool)[:, np.searchsorted(tasks, task)]

    def summ(m):
        null = np.array([roc_auc_score(np.where(c[m], 1 - y[m], y[m]), sc[m]) for c in coins])
        return {"AUROC": round(float(roc_auc_score(y[m], sc[m])), 4), "pairs": int(m.sum() // 2),
                "task-flip null p95 (on the stored scores)": round(float(np.percentile(null, 95)), 4)}
    summary = fit["summary"]["per_token"]
    res = {"layer": summary["layer"], "stored whole-pipeline null p95 (all four families)": round(fit["null_heldout_p95 (selected pipeline)"], 4),
           "all four families": summ(np.ones(len(y), bool)), f"without {CARD}": summ(mech != CARD),
           "only the families with no aligned word": summ(np.isin(mech, clean))}
    res["by family"] = {mm: round(float(roc_auc_score(y[mech == mm], sc[mech == mm])), 4) for mm in sorted(set(mech))}
    if abs(roc_auc_score(y, sc) - summary["heldout_auroc"]) > 1e-9:  # the stored scores reproduce the fit's AUROC
        raise ValueError(f"{k}: the recomputed AUROC does not match the fit's")
    out[f"stage 3 ({k})"] = res
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
path = HERE / "results" / f"concept_leakage_{stamp}.json"
path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
print(json.dumps(out, indent=1))
print(f"-> {path}")
