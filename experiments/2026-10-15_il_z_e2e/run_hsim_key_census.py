"""2e GO (17:4x): key-only census of the S_newmains H_sim roots under 09's option (a). Per root: the run keys and cell ids in the Sol file
(a_tight__avg_top2.npz) and the p2 file (fibs__reply_mean.npz), each read under its pin; only the "key" array is read, no scores. Reports any unit
of the 80 S4 cells (d9's 4r and s4d maps) by key and patches nothing. PRELIMINARY.

  PYTHONPATH=src modal run experiments/2026-10-15_il_z_e2e/run_hsim_key_census.py --config config/inputs_rms_S_newmains_dryrun3a.json
"""

import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-ilz_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
conv = modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14")
FILES = (("sol", "a_tight__avg_top2.npz"), ("p2", "fibs__reply_mean.npz"))


@app.function(image=image, cpu=1, memory=4096, timeout=900, retries=0, volumes={"/conv": conv.read_only()})
def keys_of(roots: list) -> dict:
    import re

    import numpy as np

    def load(path, pin):
        b = (Path("/conv") / path.lstrip("/").removeprefix("work/")).read_bytes()
        if not re.fullmatch("[0-9a-f]{64}", str(pin)) or hashlib.sha256(b).hexdigest() != pin:
            raise SystemExit(f"{path}: sha256 differs from the pin {pin}")
        return b

    def root_bytes(ent, fname):  # as run_rms_hits.root_bytes
        d = ent["path"].strip("/")
        if "files" in ent:
            return load(f"{d}/{fname}", ent["files"][fname])
        man = json.loads(load(f"{d}/{ent['manifest']['name']}", ent["manifest"]["sha256"]))["files"]
        e = man[fname]
        return load(f"{d}/{fname}", e if isinstance(e, str) else e["sha256"])

    out = {}
    for rt in roots:
        out[rt["root"]] = {}
        for sub, fn in FILES:
            with np.load(io.BytesIO(root_bytes(rt[sub], fn)), allow_pickle=False) as z:
                k = z["key"].astype(str)  # the key array only
            out[rt["root"]][sub] = {"n_units": int(len(k)), "runs": sorted(set(k.tolist()))}
    return out


@app.local_entrypoint()
def main(config: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    cb = (HERE / config).read_bytes()
    cfg = json.loads(cb)

    def pinned(ent):
        b = git("show", f"{ent['commit']}:{ent['path']}")
        if sha(b) != ent["sha256"]:
            raise SystemExit(f"{ent['path']}: sha256 differs from the pin")
        return json.loads(b)

    sm = pinned(cfg["s4_orphan_check"]["source_map"])
    s4 = {c["cell"] for c in sm["cells"]} | {r["cell"] for r in pinned(sm["s4d_part"])["rows"]}
    assert len(s4) == 80, len(s4)
    res = keys_of.remote(cfg["H_sim"]["roots"])
    roots, all_runs = {}, []
    for rt in cfg["H_sim"]["roots"]:
        r = res[rt["root"]]
        runs = r["sol"]["runs"]
        roots[rt["root"]] = {"batch": rt["batch"], "n_units_sol": r["sol"]["n_units"], "n_units_p2": r["p2"]["n_units"], "n_runs": len(runs),
                             "sol_and_p2_runs_equal": runs == r["p2"]["runs"], "runs_of_S4_cells": sorted(x for x in set(runs) | set(r["p2"]["runs"]) if x.split("|", 1)[1] in s4),
                             "run_list_sha256": sha("\n".join(runs).encode()), "runs": runs}
        all_runs += runs
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "H_sim key-only census (S_newmains, option (a)); keys and cell ids only, no scores; PRELIMINARY",
           "config": config, "config_sha256": sha(cb), "s4_cells": len(s4), "source_map": cfg["s4_orphan_check"]["source_map"],
           "totals": {"roots": len(roots), "runs": len(all_runs), "distinct_runs": len(set(all_runs)), "cells": len({x.split("|", 1)[1] for x in all_runs}),
                      "runs_of_S4_cells": sum(len(v["runs_of_S4_cells"]) for v in roots.values()),
                      "roots_with_sol_p2_mismatch": [k for k, v in roots.items() if not v["sol_and_p2_runs_equal"]]},
           "roots": roots,
           "meta": {"git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": sha(Path(__file__).read_bytes()), "stamp_pt": stamp}}
    b = json.dumps(rec, indent=1).encode()
    fn = f"hsim_key_census_S_newmains_{stamp}.json"
    (HERE / "results" / fn).open("xb").write(b)
    print(fn, sha(b), json.dumps(rec["totals"]))
