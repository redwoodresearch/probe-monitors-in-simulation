"""Do Q1's real-vs-simulated numbers lean on the machine batch? Saved scores only (../.run_cache/joined.npz).

Batches: a real attack run is `random` if its key's batch prefix is `random_*` (one machine; group-writable `ls -l`
modes, as base200's honest runs), else `fleet` (retained_overlap, guidance_correction, missing_completion, invalid_*).
A simulated run's reference batch is its key prefix: base200 (the random-machine honest run) or overlap800 (fleet).
Reports, on outputs both methods scored, run AUROC with a 95% main-task bootstrap:
1. real e1 vs simulated, per (e1 batch, sim reference) stratum, levels 0 and 7, TF-IDF and probe;
2. the e1-vs-e2 floor per (e1 batch, e2 batch) pairing;
3. level 7 with outputs showing a group-writable `ls -l` mode dropped (no refit), and how common those outputs are.

  uv run python experiments/2026-10-07_what_each_method_sees/followups/analyze_attack_batch.py
"""

import gzip
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from probe_shift.provenance import capture
from probe_shift.scoring import auroc_ci

HERE = Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent.parent
E1, E2, SIM = "real-attack-e1", "real-attack-e2", "sim-attack"
GROUP_WRITABLE = re.compile(r"(?<![\w-])[-dlcbps][r-][w-][xsStT-][r-]w[xsS-][r-][w-][xtT-]")  # e.g. -rw-rw-r--, drwxrwxr-x


def batch(key: str, arm: str) -> str:
    prefix = re.sub(r"^\d+_", "", key).split("_")[0]
    if arm == SIM:
        return prefix  # base200 or overlap800
    return "random" if prefix == "random" else "fleet"


def main():
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    cfg = yaml.safe_load((EXP / "config/what_sees.yaml").read_text())
    T = dict(np.load(EXP / ".run_cache/joined.npz"))
    key, arm, cell, task = T["key"], T["arm"], T["cell"], T["main_task"]
    text = {(r["key"], r["step"], r["action"]): r["readout_text"]
            for r in map(json.loads, gzip.open(Path(cfg["staging_dir"]).expanduser() / "outputs.jsonl.gz", "rt")) if r["kept"]}
    gw = np.array([bool(GROUP_WRITABLE.search(text[(k, int(s), int(a))])) for k, s, a in zip(key, T["step"], T["action"])])
    run_batch = {}
    for k, a, c in zip(key, arm, cell):
        run_batch[(c, a)] = batch(k, a)
    b_of = lambda a: np.array([run_batch.get((c, a), "") for c in cell])
    b_e1, b_e2, b_sim = b_of(E1), b_of(E2), b_of(SIM)

    def run_auc(score, rows, neg, pos):
        m = rows & np.isin(arm, [neg, pos]) & ~np.isnan(score)
        keys, inv = np.unique(key[m], return_inverse=True)
        if len(keys) < 10:
            return None
        mean = np.bincount(inv, score[m]) / np.bincount(inv)
        first = np.unique(inv, return_index=True)[1]
        y = arm[m][first] == pos
        return auroc_ci(y, mean, task[m][first]) if 0 < y.sum() < len(y) else None

    out = {"stamp": stamp, "provenance": capture(ROOT, [Path(__file__)]), "cells": {}, "real_vs_sim": {}, "floor": {}, "drop_group_writable": {}}
    out["cells"]["e1 batch x sim reference"] = dict(Counter(f"{a}/{b}" for a, b in set(zip(cell, zip(b_e1, b_sim)))
                                                          for a, b in [b]))
    pairs = Counter((b_e1[i], b_sim[i]) for i in np.unique(cell, return_index=True)[1])
    out["cells"]["e1 batch x sim reference"] = {f"{a}/{b}": n for (a, b), n in pairs.items()}
    out["cells"]["e1 batch x e2 batch"] = {f"{a}/{b}": n for (a, b), n in
                                           Counter((b_e1[i], b_e2[i]) for i in np.unique(cell, return_index=True)[1]).items()}
    # How common group-writable modes are: share of runs with at least one such output, per arm and batch.
    has = defaultdict(set)
    for k, a, g in zip(key, arm, gw):
        if g:
            has[a].add(k)
    runs = defaultdict(set)
    for k, a in zip(key, arm):
        runs[(a, batch(k, a))].add(k)
    out["group_writable_share_of_runs"] = {f"{a} {b}": round(len(ks & has[a]) / len(ks), 3) for (a, b), ks in sorted(runs.items())}
    out["group_writable_share_of_outputs"] = {a: round(float(gw[arm == a].mean()), 4) for a in (E1, E2, SIM)}
    everything = np.ones(len(key), bool)
    for level in (0, 7):
        for split in ("env", "task"):
            for m in ("tfidf", "probe"):
                s = T[f"{m}_L{level}_{split}_real_vs_sim"]
                both = ~np.isnan(T[f"tfidf_L{level}_{split}_real_vs_sim"]) & ~np.isnan(T[f"probe_L{level}_{split}_real_vs_sim"])
                r = {"all": run_auc(s, both, E1, SIM)}
                for (a, b) in sorted(pairs):
                    r[f"e1 {a} / sim {b}"] = run_auc(s, both & (b_e1 == a) & (b_sim == b), E1, SIM)
                out["real_vs_sim"][f"L{level} {split} {m}"] = r
                n = T[f"{m}_L{level}_{split}_null_e1_vs_e2"]
                bn = ~np.isnan(T[f"tfidf_L{level}_{split}_null_e1_vs_e2"]) & ~np.isnan(T[f"probe_L{level}_{split}_null_e1_vs_e2"])
                f = {"all": run_auc(n, bn, E1, E2)}
                for a, b in sorted({(x, y) for x, y in zip(b_e1, b_e2)}):
                    f[f"e1 {a} / e2 {b}"] = run_auc(n, bn & (b_e1 == a) & (b_e2 == b), E1, E2)
                out["floor"][f"L{level} {split} {m}"] = f
                if level == 7:
                    out["drop_group_writable"][f"L7 {split} {m}"] = {"none dropped": r["all"], "dropped": run_auc(s, both & ~gw, E1, SIM)}
    fmt = lambda v: "n/a" if v is None else f"{v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] (n {v['n']})"
    md = [f"# Machine batch and Q1 ({stamp}): saved scores, outputs both methods scored", "",
          f"Cells by e1 batch / sim reference: {out['cells']['e1 batch x sim reference']}; by e1 / e2 batch: {out['cells']['e1 batch x e2 batch']}.",
          f"Runs with a group-writable `ls -l` mode: {out['group_writable_share_of_runs']}; outputs: {out['group_writable_share_of_outputs']}.", ""]
    for title, block in (("Real e1 vs simulated", out["real_vs_sim"]), ("Floor: e1 vs e2", out["floor"]), ("Level 7, group-writable outputs dropped", out["drop_group_writable"])):
        cols = sorted({c for v in block.values() for c in v}, key=lambda c: (c != "all" and c != "none dropped", c))
        md += [f"## {title}", "", "| level, split, method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        md += [f"| {k} | " + " | ".join(fmt(v.get(c)) for c in cols) + " |" for k, v in block.items()]
        md.append("")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"attack_batch_{stamp}.json").write_text(json.dumps(out, indent=1))
    (HERE / "results" / f"attack_batch_{stamp}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
