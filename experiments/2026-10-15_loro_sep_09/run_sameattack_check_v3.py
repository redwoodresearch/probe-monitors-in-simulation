"""v3: the same computation on 2e's Q2-FINAL pooled v2 inputs @ 672b1dd1b per AMENDMENT 3 @ e3a4b46a9 (H_A 292 incl. t309 / t325; no t325 caveat).
Every row is gated at draw 0 against the v3 pooled records' per-set S_new rows; no previous-record gate (every S_new row changed). The v2 records stay.

POST HOC same-attack add-on to the +311 launch check (RULE.md in this folder; the add-on line committed before computing @ 01aeb72b3).
Per probe and per LORO group (manifest loro_sep_manifest_311.json @ cd1afb33e), at per-action 1e-2 and 5e-2, both readouts:
(v) run1@A - run1@B and (vi) run2@A - run2@B: the calibration-pool swap (tau_A vs tau_B) with the attack run held fixed.
Inputs: the S_new per-origin catches of 2e's pooled-455 npzs (joint draws, equal-main as stored). Draw-0 gates against the hot-deck
records @ c838127d3 (gap.same_attack; catch run2@A - run2@B). POST HOC, PRELIMINARY, "+311: Q4 FINAL; Q1, Q3 interim".
One Modal CPU container.

  modal run --detach experiments/2026-10-15_loro_sep_09/run_sameattack_check_v3.py
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
APP = "probe-shift_aswin-surya_sub-loro_sameattack311v3_kill-2026-10-22"
IN_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
OUT_VOL = "probe-shift_aswin-surya_sub-loro_results_kill-2026-11-14"  # existing; not created here
NPZ = {"C": ("xfer/fpr507prep/pooled455v2_C_2026-10-15_160734_perdraw.npz", "89b0fb5d1b25e6ab0e60b35ea31600f6943619ced99b0085d6ba56dffef6dd2e"),
       "R": ("xfer/fpr507prep/pooled455v2_R_2026-10-15_160736_perdraw.npz", "4589a3e9b215a7d9c882573cebdc5b809d1530a328f6c6344673fb796beb0d3a")}
RD = "experiments/2026-10-15_fpr_sweep_507_prep/results"
RECS = {  # name: (commit, path, sha256)
    "pooled_C": ("672b1dd1b", f"{RD}/pooled455v2_C_2026-10-15_160734.json", "63871423fd878f62c3fc1275b1ff1f87e3e7135d16cbb391475c24666d102ad0"),
    "pooled_R": ("672b1dd1b", f"{RD}/pooled455v2_R_2026-10-15_160736.json", "3f36aab2c8820fd2812f9e8031dcd2c1627aeaf897e93b27aa16c6f5d8fdcc18"),
    "amend3": ("e3a4b46a9", "experiments/2026-10-14_priority1_spec_09/loro_sep/loro_sep_manifest_amend3.json", "95b3f4522af1ecd8a2e054b61bef363fc06e5c4ad94f717b37637e2d0461d23f"),
    "manifest_311": ("cd1afb33e", "experiments/2026-10-14_priority1_spec_09/loro_sep/loro_sep_manifest_311.json", "2dd2d5397b8ccb9f070e4301a41112740259560f9387692c1f6335ad7c1cfa73"),
}
ROW_LABEL = ("Q1 FINAL; replacements pending: terminal error t237, t300, t389 (@ c10a14948) and t320 (standing rule); RL t242, t462 (@ c5420f2f5) | Q2 interim (s1-s4) | "
             "Q3 FINAL; 2 RL replacements pending (t298, t310), shown with the RL sensitivities | Q4 FINAL; INTERIM v2 (the producer's row label); H_A 292 incl. t309 / t325")
PTS = ("0.01", "0.05")
B1, MIN_DRAWS = 10001, 9500
CONTRASTS = {"v_run1_A_minus_B": "run1@A - run1@B", "vi_run2_A_minus_B": "run2@A - run2@B"}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.1.3")
in_vol = modal.Volume.from_name(IN_VOL).read_only()
out_vol = modal.Volume.from_name(OUT_VOL)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@app.function(image=image, volumes={"/in": in_vol, "/out": out_vol}, cpu=2, memory=8192, timeout=900, retries=0)
def run(rec_bytes: dict, git: str, stamp: str, readouts: str) -> dict:
    import numpy as np

    for nm, (_, _, s_) in RECS.items():
        if sha(rec_bytes[nm]) != s_:
            raise SystemExit(f"{nm} sha256 differs")
    rec = {nm: json.loads(b) for nm, b in rec_bytes.items()}
    man = rec["manifest_311"]
    Z = {}
    RO = tuple(readouts)  # readouts run here; the others are pending (e.g. the overwritten reply npz)
    if not RO or any(ro not in NPZ for ro in RO) or len(set(RO)) != len(RO):
        raise SystemExit(f"bad readouts {readouts!r}")
    for ro in RO:
        p_, s_ = NPZ[ro]
        raw = (Path("/in") / p_).read_bytes()
        if sha(raw) != s_:
            raise SystemExit(f"npz {ro} sha256 differs")
        Z[ro] = np.load(io.BytesIO(raw), allow_pickle=False)

    def arr(ro, key):
        if key not in Z[ro].files:
            raise SystemExit(f"npz {ro} lacks {key}")
        a = np.asarray(Z[ro][key], dtype=float)
        if a.shape != (B1,) or np.isinf(a).any():
            raise SystemExit(f"{key}: shape {a.shape} or infinite values")
        return a

    def summ(v):
        """Draw 0, and the 95% percentile interval over the complete bootstrap draws (withheld below MIN_DRAWS)."""
        d = v[1:][np.isfinite(v[1:])]
        return {"draw0": (float(v[0]) if np.isfinite(v[0]) else None), "complete_draws": int(d.size),
                "ci95": ([float(x) for x in np.percentile(d, [2.5, 97.5])] if d.size >= MIN_DRAWS else f"withheld: {d.size} < {MIN_DRAWS} complete draws")}

    mp = rec["amend3"]["input_provenance"]["npz"]
    if any((mp[ro]["path"], mp[ro]["sha256"]) != NPZ[ro] for ro in RO):
        raise SystemExit("npz pins differ from AMENDMENT 3")
    pins = rec["amend3"]["input_provenance"]["records"]["pooled_455"]
    if any((pins[ro]["path"], pins[ro]["sha256"]) != RECS[f"pooled_{ro}"][1:] for ro in ("C", "R")):
        raise SystemExit("record pins differ from AMENDMENT 3")

    def src(lab):
        """Draw-0 reference row: the v3 pooled record's per-set S_new row (every row)."""
        return rec[f"pooled_{lab[-1]}"]["rows"][lab]["S_new"]

    worst, n_gate = 0.0, 0

    def g0(x, ref, what):
        nonlocal worst, n_gate
        x, ref = float(x), (None if ref is None else float(ref))
        if ref is None or not np.isfinite(ref) or not np.isfinite(x):
            raise SystemExit(f"draw-0 gate: non-finite value at {what}")
        worst = max(worst, abs(x - ref))
        n_gate += 1

    out = {}
    for ro in (x for x in ("R", "C") if x in RO):  # reply leads
        rws = [r for r in man["rows"] if r["readout"] == ro]
        for f in PTS:
            per_probe, vals = {}, {}
            for r in rws:
                lab, k = r["inventory_key"], r["npz_label"]
                r1A, r1B, r2A, r2B = (arr(ro, f"{k}__joint__{f}__S_new__{c}") for c in ("run1_at_A", "run1_at_B", "run2_at_A", "run2_at_B"))
                v = {"v_run1_A_minus_B": r1A - r1B, "vi_run2_A_minus_B": r2A - r2B}
                pt = src(lab)["points"]["joint"][f]
                g0(v["v_run1_A_minus_B"][0], pt["gap"]["same_attack"]["point"], f"{lab} {f} (v)")
                g0(v["vi_run2_A_minus_B"][0], pt["catch"]["run2@A"]["point"] - pt["catch"]["run2@B"]["point"], f"{lab} {f} (vi)")
                vals[lab] = {c: x * 100.0 for c, x in v.items()}
                per_probe[lab] = {"group": r["loro_group"]} | {c: summ(x) for c, x in vals[lab].items()}
            groups = sorted({r["loro_group"] for r in rws} - {"STRESS"})
            per_group, G = {}, {}
            for gname in groups:
                mem = [r["inventory_key"] for r in rws if r["loro_group"] == gname]
                G[gname] = {c: np.mean(np.stack([vals[m][c] for m in mem]), axis=0) for c in CONTRASTS}  # NaN if any member NaN
                per_group[gname] = {"members": mem} | {c: summ(x) for c, x in G[gname].items()}
            if len(groups) != 6:
                raise SystemExit(f"{ro} {f}: {len(groups)} groups, not 6")
            mean6 = {c: summ(np.mean(np.stack([G[gname][c] for gname in groups]), axis=0)) for c in CONTRASTS}
            out[f"{ro}|{f}"] = {"per_group": per_group, "mean_over_6_groups": mean6,
                                "per_probe": per_probe, "stress": {lab: per_probe[lab] for lab in per_probe if per_probe[lab]["group"] == "STRESS"}}
    if worst > 1e-9:
        raise SystemExit(f"draw-0 gate failed: max |diff| {worst}")
    res = {"label": "v3 (Q2-FINAL pooled v2 inputs; H_A 292 incl. t309 / t325). POST HOC same-attack add-on (calibration-pool swap tau_A vs tau_B, attack run fixed) to the +311 launch check; PRELIMINARY; +311: " + ROW_LABEL + "; POST HOC: values of Delta_full, Delta_rr and the launch check were read",
           "rule": "experiments/2026-10-15_loro_sep_09/RULE.md @ 01aeb72b3", "git": git, "stamp": stamp, "versions": {"numpy": np.__version__},
           "contrasts": CONTRASTS, "units": "percentage points (fractions x 100)", "draws": "joint draws only (run455's main draws over the 506-main universe), equal-main as stored",
           "readouts_run": list(RO), "readouts_pending": [ro for ro in ("R", "C") if ro not in RO],
           "inputs": {"npz": {ro: {"volume": IN_VOL, "path": NPZ[ro][0], "sha256": NPZ[ro][1]} for ro in RO},
                      "records": {nm: {"commit": c_, "path": p_, "sha256": s_} for nm, (c_, p_, s_) in RECS.items()}},
           "gates": {"shas": True, "draw0_checked": n_gate, "draw0_max_abs_diff": worst},
           "version": "v3 (Q2-FINAL pooled v2 inputs per AMENDMENT 3 @ e3a4b46a9); the v2 record stays; 0d / Aswin decide the figure source", "row_label": ROW_LABEL, "inputs_amendment": "AMENDMENT 3 @ e3a4b46a9 (272 cells; H_B / H_A / H_sim 306 / 292 / 292)", "results": out}
    dest = Path("/out") / "loro_sep_09" / f"sameattack311v3_{readouts}_{stamp}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    out_vol.commit()
    return res


@app.local_entrypoint()
def main(readouts: str = "CR"):
    def g(*a):
        return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout

    if g("status", "--porcelain", "--", str(HERE)).strip():
        raise SystemExit("refusing to run: commit this folder first")
    head = g("rev-parse", "HEAD").decode().strip()
    rec_bytes = {nm: g("show", f"{c_}:{p_}") for nm, (c_, p_, _) in RECS.items()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = run.remote(rec_bytes, head, stamp, readouts)
    dest = HERE / "results" / f"sameattack311v3_{readouts}_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    print(json.dumps(res["gates"]), "\nwrote", dest.name)
