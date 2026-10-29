"""Pre-freeze check of Stage A per-draw npzs for AMENDMENT 5 (09's request): re-hash on the Volume, list key structure, check the keys the
LORO needs are present, and assert the draw identity per origin by comparing the per-draw cohort main weights D with the v3 npzs (whose draws are
the registered streams: S_old A24's task draws, S_new run455's main draws). Reads only key names, shapes and the D weight arrays; never prints a
gap or catch value. One Modal CPU container.

  modal run --detach experiments/2026-10-15_loro_sep_09/check_stagea_npz.py --batch first2
"""

import hashlib
import io
import json
import re
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-loro_stageacheck_kill-2026-10-22"
IN_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
BATCHES = {  # name: readout: (npz path, sha256) — pointers relayed by 09 (keys and shas only)
    "first2": {"C": ("xfer/fpr507prep/pooled455v2_C_2026-10-15_221159_perdraw.npz", "e2128dcae019b3e48ac1428c2e9b6d3d52a3cbc4389bc4ecb65066db985528d7"),
               "R": ("xfer/fpr507prep/pooled455v2_R_2026-10-15_221200_perdraw.npz", "2c24ae59f8fd91fa2cdb68631515c2a0f17f63cbf24fc7eac36188d5a81e9244")},
}
V3 = {"C": ("xfer/fpr507prep/pooled455v2_C_2026-10-15_160734_perdraw.npz", "89b0fb5d1b25e6ab0e60b35ea31600f6943619ced99b0085d6ba56dffef6dd2e"),
      "R": ("xfer/fpr507prep/pooled455v2_R_2026-10-15_160736_perdraw.npz", "4589a3e9b215a7d9c882573cebdc5b809d1530a328f6c6344673fb796beb0d3a")}
LORO_LABELS = ["Sol_a", "FIBS", "harmfulness", "refusal", "high-stakes", "c_asis", "own_frac10_L48", "own_frac25_L48", "own_frac100_L48"]
NEED_ORIGIN = {"catch_sim_at_S_primary", "catch_real_mean_at_B", "catch_run2_at_B", "catch_run2_at_A", "D", "gap_delta_full", "gap_delta_rr", "gap_sim_minus_run2"}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.1.3")
in_vol = modal.Volume.from_name(IN_VOL).read_only()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@app.function(image=image, volumes={"/in": in_vol}, cpu=2, memory=8192, timeout=900, retries=0)
def run(batch: str, git: str, stamp: str, local_meta: dict) -> dict:
    import numpy as np

    out = {"batch": batch, "git": git, "stamp": stamp, "local_metadata": local_meta, "readouts": {}}
    for ro, (p_, s_) in BATCHES[batch].items():
        raw = (Path("/in") / p_).read_bytes()
        got = sha(raw)
        r = {"npz": p_, "sha256_expected": s_, "sha256_on_volume": got, "rehash_ok": got == s_}
        if got != s_:
            raise SystemExit(f"STOP: npz {ro} sha256 differs on the Volume ({got})")
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        keys = list(z.files)
        parts = [k.split("__") for k in keys]
        r["n_arrays"] = len(keys)
        shapes = set()  # NPY headers only: no payload of any gap or catch array is read
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for name in zf.namelist():
                with zf.open(name) as fh:
                    ver = np.lib.format.read_magic(fh)
                    hdr = np.lib.format.read_array_header_1_0(fh) if ver == (1, 0) else np.lib.format.read_array_header_2_0(fh)
                    shapes.add(str(tuple(hdr[0])))
        r["shapes"] = sorted(shapes)
        labels = sorted({q[0] for q in parts})
        origins = sorted({q[3] for q in parts if len(q) == 5})
        tails = sorted({q[4] for q in parts if len(q) == 5})
        r |= {"labels": labels, "origins": origins, "per_origin_tails": tails, "modes": sorted({q[1] for q in parts if len(q) >= 4}), "points": sorted({q[2] for q in parts if len(q) >= 4})}
        # keys the LORO needs, for the 9 inventory families on this readout
        missing = []
        for fam in LORO_LABELS:
            lab = f"{fam}_{ro}"
            for m in ("joint", "fixed"):
                for f in ("0.01", "0.001", "0.05"):
                    for o in ("S_old", "S_new"):
                        for t in NEED_ORIGIN | ({"catch_run1_at_B2"} if o == "S_old" else {"catch_run1_at_B"}):
                            if f"{lab}__{m}__{f}__{o}__{t}" not in z.files:
                                missing.append(f"{lab}__{m}__{f}__{o}__{t}")
        r["loro_required_missing"] = missing[:40]
        r["loro_required_missing_count"] = len(missing)
        # draw identity: D (per-draw cohort main weight) vs the v3 npz, same label / mode / f / origin
        raw3 = (Path("/in") / V3[ro][0]).read_bytes()
        if sha(raw3) != V3[ro][1]:
            raise SystemExit(f"STOP: v3 npz {ro} sha256 differs on the Volume")
        z3 = np.load(io.BytesIO(raw3), allow_pickle=False)
        cmp_ = {"S_old": {"equal": 0, "differ": 0, "missing": 0}, "S_new": {"equal": 0, "differ": 0, "missing": 0}}
        d0 = {}
        within = {}
        for fam in LORO_LABELS:
            lab = f"{fam}_{ro}"
            for m in ("joint", "fixed"):
                for f in ("0.01", "0.001", "0.05"):
                    for o in ("S_old", "S_new"):
                        kn, ko = f"{lab}__{m}__{f}__{o}__D", f"{lab}__{m}__{f}__{o}__D"
                        if kn not in z.files or ko not in z3.files:
                            cmp_[o]["missing"] += 1
                            continue
                        a, b = np.asarray(z[kn], dtype=float), np.asarray(z3[ko], dtype=float)
                        eq = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
                        cmp_[o]["equal" if eq else "differ"] += 1
                        d0.setdefault(o, set()).add((float(a[0]), float(b[0])))
                        within.setdefault(o, []).append(a)
        r["D_vs_v3"] = cmp_
        r["D_draw0_new_vs_v3"] = {o: sorted(v) for o, v in d0.items()}
        r["D_identical_within_origin"] = {o: bool(all(np.array_equal(v[0], x, equal_nan=True) for x in v)) for o, v in within.items()}
        r["D_reading"] = ("D equality with v3 supports the same draw stream AND the same cohort mains (consistency, not definitive identity); "
                          "a difference is inconclusive unless the cohort fingerprints (local metadata) are equal")
        r["status"] = "ok"
        out["readouts"][ro] = r
    return out


@app.local_entrypoint()
def main(batch: str = "first2"):
    def g(*a):
        return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout

    if g("status", "--porcelain", "--", str(HERE)).strip():
        raise SystemExit("refusing to run: commit this folder first")
    head = g("rev-parse", "HEAD").decode().strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    # local metadata only: cohort fingerprints from the records' coverage blocks and the draw code identity of both producers
    meta = {"cohorts": {}, "draw_code": {}}
    for tag, (c_, recs) in {"stage_a": ("106262746", {"C": "pooled455v2_C_2026-10-15_221159.json", "R": "pooled455v2_R_2026-10-15_221200.json"}),
                            "v3": ("672b1dd1b", {"C": "pooled455v2_C_2026-10-15_160734.json", "R": "pooled455v2_R_2026-10-15_160736.json"})}.items():
        for ro, fn in recs.items():
            rec = json.loads(g("show", f"{c_}:experiments/2026-10-15_fpr_sweep_507_prep/results/{fn}"))
            meta["cohorts"].setdefault(ro, {})[tag] = {f"{lab}|{o}": {k: rec["rows"][lab][o]["coverage"].get(k) for k in ("cohort_cells", "of", "H_B_runs", "H_A_runs", "H_sim_runs")}
                                                       for lab in rec["rows"] for o in ("S_old", "S_new") if o in rec["rows"][lab]}
            meta["cohorts"][ro].setdefault("producer_git", {})[tag] = rec["meta"]["git"]
    for tag, c_ in (("stage_a", "214f74de4"), ("v3", "9e18a3971")):
        src = g("show", f"{c_}:experiments/2026-10-15_fpr_sweep_507_prep/run_fprsweep_507.py").decode()
        fns = src[src.index("def draws_a24"):src.index("def summ")]
        pa = g("show", f"{c_}:experiments/2026-10-15_fpr_sweep_507_prep/run_peraction311.py").decode()
        meta["draw_code"][tag] = {"draw_functions_sha256": hashlib.sha256(fns.encode()).hexdigest(), "consts": re.findall(r"^D, SEED = .*$", src, re.M),
                                  "call_sites": [ln.strip() for ln in pa.splitlines() if re.search(r"draws_(main|a24)\(", ln)]}
    for ro in ("C", "R"):
        a_, b_ = meta["cohorts"][ro]["stage_a"], meta["cohorts"][ro]["v3"]
        meta["cohorts"][ro]["equal_keys"] = sum(1 for k in a_ if k != "producer_git" and k in b_ and a_[k] == b_[k])
        meta["cohorts"][ro]["differ_keys"] = sorted(k for k in a_ if k != "producer_git" and k in b_ and a_[k] != b_[k])[:20]
    res = run.remote(batch, head, stamp, meta)
    dest = HERE / "results" / f"stagea_npzcheck_{batch}_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:6000])
    print("wrote", dest.name)
