"""Pre_call real->sim gap within each batch of the real attack runs (file-name batches: random_* on one machine,
guidance_correction and retained_overlap on the fleet), each against the sim runs of the same cells; levels 0 and 7."""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).parent
import compare_levels as cl
import sys
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402
t = pd.read_json(DATA / "tables" / "qwen3-32b__pre_call__attack_only.jsonl.gz", lines=True).query("train_set == 'real'")
runs = t.drop_duplicates(["arm", "key"])[["arm", "key", "cell"]]
real = runs[runs.arm == "real-attack-e1"].copy()
real["batch"] = real.key.str.split("__").str[0].str.replace(r"^\d{4}_", "", regex=True).str.extract(r"^(random|guidance_correction|retained_overlap)")[0]
assert real.batch.notna().all()
sim = runs[runs.arm == "sim-attack"]
groups = {"random_* (one machine)": ["random"], "fleet (guidance_correction + retained_overlap)": ["guidance_correction", "retained_overlap"],
          "guidance_correction": ["guidance_correction"], "retained_overlap": ["retained_overlap"]}
cl.LEVELS, cl.POSITIONS = [0, 7], ["pre_call"]
out = {}
for name, bs in groups.items():
    r = real[real.batch.isin(bs)]
    s = sim[sim.cell.isin(set(r.cell))]
    cl.KEEP = set(zip(r.arm, r.key)) | set(zip(s.arm, s.key))
    newest = cl.main("_batch_" + bs[0] if len(bs) == 1 else "_batch_fleet")
    out[name] = {"real_runs": len(r), "sim_runs": len(s), "cells": r.cell.nunique()}
    out[name]["result"] = json.loads(newest.read_text())
stamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d_%H%M%S")
(RESULTS / f"shift_batch_split_{stamp}.json").write_text(json.dumps(out, indent=1))
f = lambda v: f"{v['value']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]"
for name, o in out.items():
    for lv in ("level 0", "level 7"):
        e = o["result"]["pre_call"]["train real"][lv]; g = e["gap real minus sim"]
        print(f"{name:48} {lv}: runs real/sim {o['real_runs']}/{o['sim_runs']} | AUROC {e['real']['auroc']['value']:.3f}/{e['sim']['auroc']['value']:.3f} gap {f(g['auroc'])} | recal {e['real']['recall_recal']['value']:.3f}/{e['sim']['recall_recal']['value']:.3f} gap {f(g['recall_recal'])} | frozen gap {f(g['recall'])} (FPR {e['real']['fpr']['value']:.3f}/{e['sim']['fpr']['value']:.3f})")
