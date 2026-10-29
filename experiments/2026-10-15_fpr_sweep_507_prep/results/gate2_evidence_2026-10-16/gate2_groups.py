"""Keys-and-counts check of 09's gate 2 groups on a pooled catalogue run's export files (no score value leaves the container).
For every label x combo: the groups of run keys whose units are byte-identical under pw_canon (zero-unit keys excluded, as the build does),
each key's unit count per role, and whether the group's world tokens differ. Usage:
  modal run gate2_groups.py --export-dir xfer/fpr507prep/pooledworld/<stem>/export --out <local json>"""
import json

import modal

VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
app = modal.App("probe-shift_aswin-surya_sub-pooledworld_gate2groups_kill-2026-11-16")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
vol = modal.Volume.from_name(VOL, version=2)


@app.function(image=image, cpu=4, memory=8192, timeout=1800, retries=0, volumes={"/sol": vol})
def groups(export_dir: str) -> dict:
    import hashlib
    from pathlib import Path

    import numpy as np

    def pw_canon(u_):  # verbatim logic of run_peraction311.pw_canon (gates 1 and 2)
        parts = []
        for a_ in u_:
            a_ = np.asarray(a_)
            if a_.dtype.kind == "f":
                a_ = a_.astype(np.float64)
                a_ = np.where(np.isnan(a_), np.nan, a_ + 0.0)
            elif a_.dtype.kind in "iub":
                a_ = a_.astype(np.int64)
            else:
                raise SystemExit(f"unexpected dtype {a_.dtype}")
            parts.append(repr(a_.shape).encode() + b":" + np.ascontiguousarray(a_).tobytes())
        return b"|".join(parts)

    by = {}
    files = sorted(Path("/sol", export_dir).glob("*.npz"))
    for f_ in files:
        lab_, org_ = f_.name.split("__", 1)
        combo_ = org_.split("[pooled:_", 1)[1].split("]", 1)[0] if "[pooled:_" in org_ else "?"
        by.setdefault((lab_, combo_), []).append(f_)
    out = {"export_dir": export_dir, "files": len(files), "label_combos": len(by), "groups": {}}
    for (lab_, combo_), fs_ in sorted(by.items()):
        ident_, nun_ = {}, {}
        for f_ in fs_:
            with np.load(f_) as z_:
                for nm_ in ("B", "A", "S"):
                    run = np.asarray(z_[f"{nm_}_run"]).astype(str)
                    arr_ = [np.asarray(z_[f"{nm_}_{k_}"]) for k_ in ("s", "f", "sbm")]
                    for r_ in sorted(set(run.tolist())):
                        ix_ = np.flatnonzero(run == r_)
                        ident_.setdefault(r_, pw_canon(tuple(a_[ix_] for a_ in arr_)))
                        nun_.setdefault(r_, {})[nm_] = int(len(ix_))
        ch_ = {}
        for r_, c_ in ident_.items():
            ch_.setdefault(hashlib.sha256(c_).hexdigest(), []).append(r_)
        gs_ = [sorted(v_) for v_ in ch_.values() if len(v_) > 1]
        if gs_:
            out["groups"][f"{lab_} | {combo_}"] = [{"keys": g_, "units_per_role": {r_: nun_[r_] for r_ in g_},
                                                    "worlds": sorted({r_.split("|", 1)[0] for r_ in g_}),
                                                    "cells": sorted({r_.split("|", 1)[1] for r_ in g_})} for g_ in sorted(gs_)]
    out["n_label_combos_with_groups"] = len(out["groups"])
    out["n_groups"] = sum(len(v_) for v_ in out["groups"].values())
    return out


@app.local_entrypoint()
def main(export_dir: str, out: str):
    r = groups.remote(export_dir)
    with open(out, "x") as fh:
        json.dump(r, fh, indent=1)
    print(json.dumps({"files": r["files"], "label_combos": r["label_combos"], "n_label_combos_with_groups": r["n_label_combos_with_groups"],
                      "n_groups": r["n_groups"]}))
