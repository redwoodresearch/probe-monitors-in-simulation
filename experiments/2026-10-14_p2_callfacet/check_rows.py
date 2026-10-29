"""$0 checks before the call-level rule (counts only; no score is computed, no text is read): (a) the stored y of the 196-task
impactful monitor is the meta label (`label_call`, as_labeled), by reloading the main config's rows and comparing them with the
per-call file; (b) honest A, B (simhon config) and C (honest_scores_C.json's extraction) hold impactful rows at the same rule-1
command readout, with per-task coverage; (c) the DT monitor folders' layout. Modal CPU, read-only Volumes.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with pyyaml modal run experiments/2026-10-14_p2_callfacet/check_rows.py
"""

import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
Q3 = HERE.parent / "2026-10-08_q3_onpolicy_figures" if modal.is_local() else Path("/root")
sys.path.insert(0, str(Q3))
import run_score_onpolicy as S  # (unchanged)

APP = "probe-shift_aswin-surya_sub-p2probes_callcheck_kill-2026-11-13"
KEY = "scaleup196_probes_14b33da3d97a_percall"
DT = "dt_attack_fits_00f4994d6c1e"
app = modal.App(APP)
image = S.image.uv_pip_install("pyyaml").add_local_file(Q3 / "run_score_onpolicy.py", "/root/run_score_onpolicy.py")


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes={"/acts": S.acts_vol.read_only(), "/score": S.score_vol.read_only()})
def check(cfg: dict, ext: dict, pilot: str) -> dict:
    from collections import Counter

    tmp = Path("/tmp/pilot.json")
    tmp.write_text(pilot)
    folds = S.frozen_folds(tmp)
    d = Path("/score") / KEY
    draw = json.loads((d / "draw_00000.json").read_text())
    hc = json.loads((d / "honest_scores_C.json").read_text())
    cx = hc["c_extraction"]
    out = {"draw_keys": sorted(draw), "label_rule_main": cfg.get("label_rule", "as_labeled"), "c_extraction": cx}
    main = S.load(cfg | {"_rows_only": True}, folds)["rows"]
    out["xcache_rows_sha_equal"] = json.loads((Path("/score") / cfg["x_cache"] / "rows.json").read_text())["rows_sha256"] == S.rows_sha(main)
    meta = draw.get("meta")
    if meta is not None:
        out["stored_meta_rows"] = len(meta)
        out["y_equal_meta_label"] = [[r[0], r[1], r[5]] for r in main] == [[m["world"], m["task"], m["y"]] for m in meta]
    out["main_rows"] = {f"{w}|y{y}": n for (w, y), n in sorted(Counter((r[0], r[5]) for r in main).items())}
    hon = S.load(ext | {"_rows_only": True}, folds)["rows"]
    out["simhon_rows"] = {f"{w}|y{y}": n for (w, y), n in sorted(Counter((r[0], r[5]) for r in hon).items())}
    worlds_c = {w: v for w, v in ext["worlds"].items() if w not in ("honA", "honB", "simH")} | {"honB": f"honest_b@{cx}"}
    cfg_c = ext | {"label": "scaleup196_honc", "extractions": ext["extractions"] + [cx], "worlds": worlds_c}
    crow = [r for r in S.load(cfg_c | {"_rows_only": True}, folds)["rows"] if r[0] == "honB" and r[2][1] == cx]
    out["c_rows"] = {f"honC|y{y}": n for y, n in sorted(Counter(r[5] for r in crow).items())}
    cov = {}
    for name, rows, w in (("honA", hon, "honA"), ("honB", hon, "honB"), ("simH", hon, "simH"), ("honC", crow, "honB")):
        t_all = {r[1] for r in rows if r[0] == w}
        t_imp = {r[1] for r in rows if r[0] == w and r[5] == 1}
        cov[name] = {"tasks": len(t_all), "tasks_with_impactful": len(t_imp)}
    out["coverage"] = cov
    out["honest_scores_files"] = {"AB_rows_by_world_y": dict(Counter(f"{x['world']}|y{x['y']}" for x in json.loads((d / 'honest_scores.json').read_text())["rows"])),
                                  "C_rows_by_y": dict(Counter(f"y{x['y']}" for x in hc["rows"]))}
    pz = __import__("numpy").load(d / "probes_draw0.npz")
    out["probes_meta"] = json.loads(str(pz["meta"]))
    dt = {}
    for f in sorted((Path("/score") / DT).iterdir()):
        if f.is_dir():
            rows = json.loads((f / "rows.json").read_text())
            mj = json.loads((f / "main.json").read_text())
            dt[f.name] = {"files": sorted(p.name for p in f.iterdir()), "rows": len(rows["y"]), "counts": rows["counts"],
                          "main_keys": sorted(mj), "probes": sorted(mj.get("probes", {})), "cells": sorted(mj.get("cells", {})),
                          "rows_equal_main_cache": rows["task"] == [] or len(rows["task"]) == len(main)}
    out["dt"] = dt
    return out


@app.local_entrypoint()
def main():
    import yaml

    cfg = yaml.safe_load((Q3 / "config" / "score_scaleup196.yaml").read_text())
    ext = yaml.safe_load((Q3 / "config" / "score_scaleup196_simhon.yaml").read_text())
    pilot = (HERE.parents[1] / cfg["pilot_tasks"]).read_text()
    print(json.dumps(check.remote(cfg, ext, pilot), indent=1))
