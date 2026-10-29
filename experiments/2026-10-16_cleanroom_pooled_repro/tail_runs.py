"""Which simulated-honest runs supply the units above τ_sim at the R37 levels (0d's question, 10-17; a1)? Keys and counts only.

For own_all_rs_uncut_L48 C and sol_a_uncut C (comparator), combo pooled_all_origins, levels 1e-4 / 2e-4 / 5e-4, from the R37 record
(@ dc38b77e3, decompressed sha256 pinned below):
  - T_S read from rows[label][<origin>].pools_at_draw0[level].S.T (must agree across the label's pooled_all_origins origin rows), and
    recomputed from the pool's draw-0 z with the registered rule (descending sort, k = floor(f n + 1e-9), T = z[min(k, n-1)]); the two
    must be bitwise equal, and the count of units strictly above T must equal the record's units_above;
  - the run keys (S_run) of the units above T, with the unit count per run and the origin tag (S_org).
No score value is returned: only T (already in the record), counts, run keys and origin tags. One Modal CPU container.

  CR_CFG=<folder holding the R37 .json.gz> uv run --no-project --python 3.12 --with modal==1.6.1 modal run tail_runs.py
"""

import gzip
import hashlib
import io
import json
import os
import time
from pathlib import Path

import modal

POOLED = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
R37 = "alltasks_CRD_FULL_r2_allorigins_rule37_pooledworld_scored_e12_2026-10-16_231356.json.gz"
R37_FULL = "71ad0868e6c2db42e91eadda0a9069a1d737529db42a66eb4bf19c4cfb5d02fe"  # decompressed sha256 (commit message: 71ad0868…)
LABELS = ["own_all_rs_uncut_L48 C", "sol_a_uncut C"]
COMBO = "pooled_all_origins"
LEVELS = ["0.0001", "0.0002", "0.0005"]

app = modal.App("probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if modal.is_local():
    image = image.add_local_file(str(Path(os.environ["CR_CFG"]) / R37), f"/cfg/{R37}")


def sha(b):
    return hashlib.sha256(b).hexdigest()


@app.function(image=image, cpu=1, memory=8192, timeout=900, retries=0,
              volumes={"/p": modal.Volume.from_name(POOLED, version=2).read_only()})
def tail() -> dict:
    import numpy as np

    raw = gzip.decompress(Path("/cfg", R37).read_bytes())
    assert sha(raw) == R37_FULL, "R37 record sha"
    rec = json.loads(raw)
    pw = rec["meta"]["pooled_world"]
    out = {"record_sha256": R37_FULL, "labels": {}}
    for lab in LABELS:
        lc = f"{lab} | {COMBO}"
        cal, pool = pw["calibrations"]["by_label_combo"][lc], pw["manifest_per_label"][lc]
        arrs = {}
        for name, e in (("calib", cal), ("pooled", pool)):
            b = Path("/p", e["path"]).read_bytes()
            assert sha(b) == e["sha256"], f"{lab} {name} sha"
            z = np.load(io.BytesIO(b), allow_pickle=False)
            arrs[name] = {k: z[k] for k in z.files}
        pick = lambda k: arrs["calib"][k] if k in arrs["calib"] else arrs["pooled"][k]  # noqa: E731
        zS, run, org = pick("draw0_z__S"), pick("S_run"), pick("S_org")
        assert zS.shape == run.shape == org.shape and zS.ndim == 1, f"{lab}: unit-order arrays differ in shape"
        assert np.all(np.isfinite(zS)), f"{lab}: non-finite draw-0 z"
        rows = {o: r for o, r in rec["rows"][lab].items() if f"[pooled: {COMBO}]" in o}
        lo = {}
        for lvl in LEVELS:
            Ts = {r["pools_at_draw0"][lvl]["S"]["T"] for r in rows.values()}
            ab = {r["pools_at_draw0"][lvl]["S"]["units_above"] for r in rows.values()}
            assert len(Ts) == 1 and len(ab) == 1, f"{lab} {lvl}: origin rows disagree on T_S or units_above"
            T_rec, above_rec = Ts.pop(), ab.pop()
            zs = -np.sort(-zS, kind="stable")
            n = len(zs)
            k = int(np.floor(float(lvl) * n + 1e-9))
            T = float(zs[min(k, n - 1)])
            hit = zS > T
            assert T == T_rec, f"{lab} {lvl}: recomputed T_S differs from the record"
            assert int(hit.sum()) == above_rec, f"{lab} {lvl}: units above T differ from the record"
            per = {}
            for r_, o_ in zip(run[hit].tolist(), org[hit].tolist()):
                key = (str(r_), str(o_))
                per[key] = per.get(key, 0) + 1
            lo[lvl] = {"T_S": T, "n_units": n, "k": k, "units_above": int(hit.sum()), "n_runs_above": len(per),
                       "runs": [{"run": r_, "origin": o_, "units_above": c} for (r_, o_), c in
                                sorted(per.items(), key=lambda kv: (-kv[1], kv[0]))]}
        out["labels"][lab] = {"n_units_S": int(len(zS)), "n_runs_S": int(len(set(run.tolist()))), "levels": lo,
                              "files": {"calib": cal["path"], "pooled": pool["path"]}}
    return out


@app.local_entrypoint()
def main():
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    res = tail.remote()
    path = Path(__file__).parent / "results" / f"tail_runs_{stamp}.json"
    path.open("x").write(json.dumps(res, indent=1))
    print("wrote", path)
