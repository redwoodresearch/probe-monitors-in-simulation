"""Fix-full (PLAN.md Amendment 2), POST HOC, preliminary until Codex: N's mechanism for Qwen3-32B. Per output of the 939
simulated attack runs, the outer model's output_end score under N (every trailing-newline fix, NL arm) minus as simulated
(V1), from score_qwen3_fx.py's per-call file, split by position relative to the run's first edited output (before; the
first edited output; later edited outputs; later unedited outputs) and by tool family (bash, view, the editor's edits,
other). Each readout comes before its own output's trailing newline, so a shift at the first edited output or before it
should be about 0; a newline effect through the history shows on later outputs. Mean shift in logits; 95% ranges from
2,000 multinomial draws over the runs' main tasks (seed 20261012), runs weighted by their task's draw count. Local, $0.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-13_q2_fixfull/analyze_fx_mech_qwen3.py \\
      --calls results/fx_frozen_calls_POSTHOC_<NL stamp>.jsonl.gz --stage results/stage_2026-10-13_142154.json
"""
import argparse
import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
B, SEED = 2000, 20261012
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def fam_group(f):
    return "bash" if f.endswith("bash") else "view" if f == "text_editor:view" else "edit" if f in ("text_editor:create", "text_editor:str_replace", "text_editor:insert") else "other"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--calls", required=True, type=Path); ap.add_argument("--stage", required=True, type=Path)
    a = ap.parse_args()
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    st = json.loads(a.stage.read_text())
    outs = {k.split("|", 1)[1]: v for k, v in st["NL outputs (step, action, family, changed)"].items() if k.startswith("attack|")}
    runs36 = {r["key"]: r for r in json.loads((HERE.parent / "2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json").read_text()) if r["arm"] == "sim-attack"}
    sc = defaultdict(dict)
    with gzip.open(a.calls, "rt") as fh:
        for line in fh:
            c = json.loads(line)
            if c["readout"] in ("output_end", "output_end@48"):
                sc[(c["key"], c["readout"], c["step"], c["action"])][c["label"]] = c["score"]
    if not sc:
        raise SystemExit("no output_end calls in the per-call file")
    keys = sorted({k for k, *_ in sc})
    if not set(keys) <= set(outs):
        raise SystemExit("per-call keys outside the NL arm's staged runs")
    tasks = sorted({runs36[k]["main_task"] for k in keys}); tix = {k: tasks.index(runs36[k]["main_task"]) for k in keys}
    g = np.random.default_rng(SEED); W = np.stack([np.bincount(g.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)]).astype(float)
    out = {"POST HOC": True, "status": "preliminary until Codex", "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
           "inputs sha256": {a.calls.name: sha(a.calls), a.stage.name: sha(a.stage)}, "runs": len(keys), "readouts": {}}
    for ro in ("output_end", "output_end@48"):
        cells = defaultdict(list)  # (position, family) -> [(task index, shift)]
        missing = 0
        for k in keys:
            o = outs[k]
            changed = [(s, a_) for s, a_, f, ch in o if ch]
            first = min(changed) if changed else None
            fam = {(s, a_): f for s, a_, f, _ in o}
            chs = {(s, a_) for s, a_, _, ch in o if ch}
            for (s, a_) in fam:
                v = sc.get((k, ro, s, a_))
                if not v:
                    continue
                if "V1" not in v or "FX" not in v:
                    missing += 1
                    continue
                if first is None or (s, a_) < first:
                    pos = "before the first edited output"
                elif (s, a_) == first:
                    pos = "the first edited output"
                elif (s, a_) in chs:
                    pos = "later edited outputs"
                else:
                    pos = "later unedited outputs"
                cells[(pos, fam_group(fam[(s, a_)]))].append((tix[k], v["FX"] - v["V1"]))
        res = {}
        for (pos, fg), xs in sorted(cells.items()):
            t = np.array([x[0] for x in xs]); d = np.array([x[1] for x in xs])
            wt = W[:, t]; boot = (wt @ d) / np.maximum(wt.sum(1), 1e-12)
            res[f"{pos} | {fg}"] = {"outputs": int(len(d)), "mean shift N - as simulated (logits)": float(d.mean()),
                                    "95%": [float(x) for x in np.percentile(boot[wt.sum(1) > 0], [2.5, 97.5])], "max |shift|": float(np.abs(d).max())}
        out["readouts"][ro] = {"calls with only one of V1, FX": missing, "cells": res}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"fx_mech_qwen3_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    for ro, r in out["readouts"].items():
        print(f"== Qwen3-32B {ro}: N - as simulated, per output (logits)")
        for k, v in r["cells"].items():
            print(f"  {k:48s} n {v['outputs']:6d}  {v['mean shift N - as simulated (logits)']:+.4f} {[round(x, 4) for x in v['95%']]}")
    print(dest)


if __name__ == "__main__":
    main()
