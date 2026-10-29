"""Pre-freeze check of a Stage A per-draw npz in the v2 <combo> key format (all-tasks, sweeps, union-Sol) for AMENDMENT 5: re-hash on the
Volume, NPY-header shapes, key structure, the LORO-required keys per origin, and the draw identity from the per-draw cohort main weights D
(S_old / S_new vs the v3 npzs; S_newmains with vs without S4 on one stream: D_with − D_without >= 0 on every draw; D identical across labels
and across combos for the same origin). Reads only key names, NPY headers, shas and the D weight arrays; never a gap or catch value.

  modal run --detach experiments/2026-10-15_loro_sep_09/check_stagea_combo.py --batch alltasks
"""

import hashlib
import io
import json
import re
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-loro_stageacheck_kill-2026-10-22"
IN_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
RD = "experiments/2026-10-15_fpr_sweep_507_prep/results"
BATCHES = {  # name: (npz path, sha256, record commit, record path, producer git) — relayed by 09 (keys and shas only)
    "alltasks": ("xfer/fpr507prep/alltasks_CRD_2026-10-15_230525_perdraw.npz", "fa193bdfb686b373347074722412db28c1ea85c8c1945136ea518faacc1e4cf8",
                 "8d1338faa", f"{RD}/alltasks_CRD_2026-10-15_230525.json", "ae010b45a"),
}
V3 = {"C": ("xfer/fpr507prep/pooled455v2_C_2026-10-15_160734_perdraw.npz", "89b0fb5d1b25e6ab0e60b35ea31600f6943619ced99b0085d6ba56dffef6dd2e"),
      "R": ("xfer/fpr507prep/pooled455v2_R_2026-10-15_160736_perdraw.npz", "4589a3e9b215a7d9c882573cebdc5b809d1530a328f6c6344673fb796beb0d3a")}
LORO_LABELS = ["Sol_a", "FIBS", "harmfulness", "refusal", "high-stakes", "c_asis", "own_frac10_L48", "own_frac25_L48", "own_frac100_L48"]
TAILS = {"catch_sim_at_S_primary", "catch_real_mean_at_B", "catch_run2_at_B", "catch_run2_at_A", "D", "gap_delta_full", "gap_delta_rr", "gap_sim_minus_run2"}
ORIGIN_COMBOS = {"S_old": ("pooled_455", "pooled_all", "pooled_all_without_S4"), "S_new": ("pooled_455", "pooled_all", "pooled_all_without_S4"),
                 "S_newmains": ("pooled_all",), "S_newmains_without_S4": ("pooled_all_without_S4",)}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.1.3")
in_vol = modal.Volume.from_name(IN_VOL).read_only()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@app.function(image=image, volumes={"/in": in_vol}, cpu=2, memory=16384, timeout=900, retries=0)
def run(batch: str, git: str, stamp: str, local_meta: dict) -> dict:
    import numpy as np

    p_, s_ = BATCHES[batch][:2]
    raw = (Path("/in") / p_).read_bytes()
    got = sha(raw)
    if got != s_:
        raise SystemExit(f"STOP: npz sha256 differs on the Volume ({got})")
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    keys = list(z.files)
    parts = [k.split("__") for k in keys]
    shapes = set()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:  # NPY headers only
        for name in zf.namelist():
            with zf.open(name) as fh:
                ver = np.lib.format.read_magic(fh)
                hdr = np.lib.format.read_array_header_1_0(fh) if ver == (1, 0) else np.lib.format.read_array_header_2_0(fh)
                shapes.add(str(tuple(hdr[0])))
    r = {"batch": batch, "git": git, "stamp": stamp, "local_metadata": local_meta, "npz": p_, "sha256_on_volume": got, "rehash_ok": True, "n_arrays": len(keys),
         "shapes": sorted(shapes), "part_counts": sorted({len(q) for q in parts}), "labels": sorted({q[0] for q in parts}), "combos": sorted({q[1] for q in parts if len(q) == 6}),
         "origins": sorted({q[4] for q in parts if len(q) == 6}), "tails": sorted({q[5] for q in parts if len(q) == 6 and q[4] != "pooled"}),
         "pooled_gaps": sorted({q[5] for q in parts if len(q) == 6 and q[4] == "pooled"}), "modes": sorted({q[2] for q in parts if len(q) == 6}), "points": sorted({q[3] for q in parts if len(q) == 6})}
    missing = []
    for ro in ("C", "R"):
        for fam in LORO_LABELS:
            lab = f"{fam}_{ro}"
            for o, combos in ORIGIN_COMBOS.items():
                for cb in combos:
                    for m in ("joint", "fixed"):
                        for f in ("0.01", "0.001", "0.05"):
                            for t in TAILS | ({"catch_run1_at_B2"} if o == "S_old" else {"catch_run1_at_B"}):
                                if f"{lab}__{cb}__{m}__{f}__{o}__{t}" not in z.files:
                                    missing.append(f"{lab}__{cb}__{m}__{f}__{o}__{t}")
    r["loro_required_missing_count"], r["loro_required_missing_sample"] = len(missing), missing[:30]

    def D(ro, fam, cb, m, f, o):
        k = f"{fam}_{ro}__{cb}__{m}__{f}__{o}__D"
        return np.asarray(z[k], dtype=float) if k in z.files else None

    res = {}
    for ro in ("C", "R"):
        raw3 = (Path("/in") / V3[ro][0]).read_bytes()
        if sha(raw3) != V3[ro][1]:
            raise SystemExit(f"STOP: v3 npz {ro} sha256 differs on the Volume")
        z3 = np.load(io.BytesIO(raw3), allow_pickle=False)
        rr = {"vs_v3": {"S_old": [0, 0, 0], "S_new": [0, 0, 0]}, "across_combos_equal": {}, "within_origin_identical": {}, "D0": {}, "newmains_with_minus_without": {}}
        for o in ("S_old", "S_new"):
            for fam in LORO_LABELS:
                for m in ("joint", "fixed"):
                    for f in ("0.01", "0.001", "0.05"):
                        b = f"{fam}_{ro}__{m}__{f}__{o}__D"
                        for cb in ORIGIN_COMBOS[o]:
                            a = D(ro, fam, cb, m, f, o)
                            if a is None or b not in z3.files:
                                rr["vs_v3"][o][2] += 1
                                continue
                            rr["vs_v3"][o][0 if np.array_equal(a, np.asarray(z3[b], dtype=float), equal_nan=True) else 1] += 1
        for o, combos in ORIGIN_COMBOS.items():
            arrs = [a for fam in LORO_LABELS for cb in combos for m in ("joint", "fixed") for f in ("0.01", "0.001", "0.05") if (a := D(ro, fam, cb, m, f, o)) is not None]
            rr["within_origin_identical"][o] = bool(arrs) and all(np.array_equal(arrs[0], x, equal_nan=True) for x in arrs)
            rr["D0"][o] = sorted({float(x[0]) for x in arrs})
            if len(combos) > 1:
                rr["across_combos_equal"][o] = all(np.array_equal(D(ro, fam, combos[0], m, f, o), D(ro, fam, cb, m, f, o), equal_nan=True)
                                                   for fam in LORO_LABELS for cb in combos[1:] for m in ("joint", "fixed") for f in ("0.01", "0.001", "0.05")
                                                   if D(ro, fam, combos[0], m, f, o) is not None and D(ro, fam, cb, m, f, o) is not None)
        a_, b_ = D(ro, "FIBS", "pooled_all", "joint", "0.01", "S_newmains"), D(ro, "FIBS", "pooled_all_without_S4", "joint", "0.01", "S_newmains_without_S4")
        if a_ is not None and b_ is not None:
            dif = a_ - b_
            rr["newmains_with_minus_without"] = {"draw0": float(dif[0]), "min_over_draws": float(np.nanmin(dif)), "all_nonnegative": bool(np.all(dif >= -1e-12))}
        rr["vs_v3"] = {o: {"equal": v[0], "differ": v[1], "missing": v[2]} for o, v in rr["vs_v3"].items()}
        res[ro] = rr
    r["D_checks"] = res
    r["D_reading"] = ("D equality (S_old / S_new vs v3; across labels and combos) supports one stream and the same cohort mains; for S_newmains, the extended stream is "
                      "evidenced by the producer code (local_metadata.draw_code) and by D_with − D_without >= 0 on every draw (estimation-only S4 removal on one stream); consistency, not definitive identity")
    return r


@app.local_entrypoint()
def main(batch: str = "alltasks"):
    def g(*a):
        return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout

    if g("status", "--porcelain", "--", str(HERE)).strip():
        raise SystemExit("refusing to run: commit this folder first")
    head = g("rev-parse", "HEAD").decode().strip()
    p_, s_, rc, rp, pg = BATCHES[batch]
    rec_b = g("show", f"{rc}:{rp}")
    meta = {"record": {"commit": rc, "path": rp, "sha256": hashlib.sha256(rec_b).hexdigest()}, "draw_code": {}}
    rec = json.loads(rec_b)
    meta["record_npz_pointer"] = {k: rec["npz"].get(k) for k in ("path", "sha256", "n_arrays")}
    meta["record_git"] = rec["meta"]["git"]
    for tag, c_ in (("stage_a", pg), ("v3", "9e18a3971")):
        src = g("show", f"{c_}:experiments/2026-10-15_fpr_sweep_507_prep/run_fprsweep_507.py").decode()
        fns = src[src.index("def draws_a24"):src.index("def summ")]
        pa = g("show", f"{c_}:experiments/2026-10-15_fpr_sweep_507_prep/run_peraction311.py").decode()
        meta["draw_code"][tag] = {"draw_functions_sha256": hashlib.sha256(fns.encode()).hexdigest(), "consts": re.findall(r"^D, SEED = .*$", src, re.M),
                                  "call_sites": [ln.strip() for ln in pa.splitlines() if re.search(r"draws_(main|a24)\(", ln)]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = run.remote(batch, head, stamp, meta)
    dest = HERE / "results" / f"stagea_npzcheck_{batch}_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("rehash_ok", "n_arrays", "shapes", "combos", "origins", "loro_required_missing_count", "D_checks")}, indent=1)[:5000])
    print("wrote", dest.name)
