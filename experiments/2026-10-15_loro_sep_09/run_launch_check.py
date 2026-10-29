"""POST HOC launch check on the +311 reply offset (RULE.md in this folder, committed before computing @ d13ad9a4b).
Per probe and per LORO group (manifest loro_sep_manifest_311.json @ cd1afb33e), at per-action 1e-2 and 5e-2, both readouts:
(i) sim@S - run1@B, (ii) sim@S - run2@B, (iii) run2@B - run1@B, (iv) sim@S - real_mean@B (= Delta_full).
Inputs: the S_new per-origin catches of 2e's pooled-455 npzs (joint draws, equal-main as stored). Draw-0 gates against the hot-deck
records @ c838127d3 and against the LORO +311 record @ 2351051cc. POST HOC, PRELIMINARY, "+311: Q4 FINAL; Q1, Q3 interim".
One Modal CPU container.

  modal run --detach experiments/2026-10-15_loro_sep_09/run_launch_check.py
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
APP = "probe-shift_aswin-surya_sub-loro_launchcheck311_kill-2026-10-22"
IN_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
OUT_VOL = "probe-shift_aswin-surya_sub-loro_results_kill-2026-11-14"  # existing; not created here
NPZ = {"C": ("xfer/fpr507prep/pooled455_2026-10-15_141921_perdraw.npz", "84112f5061455f9b1970ac90ff963e31d38f8c75b26b3cf28510e2c06476dbc8"),
       "R": ("xfer/fpr507prep/pooled455_2026-10-15_142548_perdraw.npz", "bd5875fbd09fc9177783f7bd84f6b862dae1d2eb8818f2d25b1461b1bd165a11")}
RD = "experiments/2026-10-15_fpr_sweep_507_prep/results"
RECS = {  # name: (commit, path, sha256)
    "hd_C": ("c838127d3", f"{RD}/peraction311_all_hotdeck_C_2026-10-15_140206.json", "8a00f2060ef6aec70822a6e6af3f8169540b2e513641a82a51749ff6bdc69c41"),
    "hd_R": ("c838127d3", f"{RD}/peraction311_all_hotdeck_R_2026-10-15_140206.json", "a1a150fb20c54c519d5ab127ca675d610efea41e54ce824d3eb6cb577fcfce78"),
    "hd_O": ("c838127d3", f"{RD}/peraction311_all_hotdeck_O_2026-10-15_140209.json", "bcd97eb54097bedb94e43a7672ab190cdb4fa24bc3db89307f6ac98b60a4cf0a"),
    "hd_X": ("c838127d3", f"{RD}/peraction311_all_hotdeck_X_2026-10-15_140208.json", "c232bbb6d7d3692aa8d5e9239bde4c9f0be96e0487228583c138eed7d0f8a6d5"),
    "manifest_311": ("cd1afb33e", "experiments/2026-10-14_priority1_spec_09/loro_sep/loro_sep_manifest_311.json", "2dd2d5397b8ccb9f070e4301a41112740259560f9387692c1f6335ad7c1cfa73"),
    "loro_311": ("2351051cc", "experiments/2026-10-15_loro_sep_09/results/loro_sep311_2026-10-15_145541.json", "b7fcb2e89d6cf63049ccc522624ce942f4216e0a8001d456dfc1bd2779511a80"),
}
PTS = ("0.01", "0.05")
B1, MIN_DRAWS = 10001, 9500
CONTRASTS = {"i_sim_minus_run1": "sim@S - run1@B", "ii_sim_minus_run2": "sim@S - run2@B", "iii_run2_minus_run1": "run2@B - run1@B", "iv_delta_full": "sim@S - real_mean@B"}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.1.3")
in_vol = modal.Volume.from_name(IN_VOL).read_only()
out_vol = modal.Volume.from_name(OUT_VOL)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@app.function(image=image, volumes={"/in": in_vol, "/out": out_vol}, cpu=2, memory=8192, timeout=900, retries=0)
def run(rec_bytes: dict, git: str, stamp: str) -> dict:
    import numpy as np

    for nm, (_, _, s_) in RECS.items():
        if sha(rec_bytes[nm]) != s_:
            raise SystemExit(f"{nm} sha256 differs")
    rec = {nm: json.loads(b) for nm, b in rec_bytes.items()}
    man, loro = rec["manifest_311"], rec["loro_311"]
    Z = {}
    for ro, (p_, s_) in NPZ.items():
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

    def hd(lab):
        fam, ro = lab[:-2], lab[-1]
        return rec["hd_X" if fam == "c_asis" else ("hd_O" if fam.startswith("own_") else f"hd_{ro}")]["rows"][lab]

    worst, n_gate = 0.0, 0

    def g0(x, ref, what):
        nonlocal worst, n_gate
        x, ref = float(x), (None if ref is None else float(ref))
        if ref is None or not np.isfinite(ref) or not np.isfinite(x):
            raise SystemExit(f"draw-0 gate: non-finite value at {what}")
        worst = max(worst, abs(x - ref))
        n_gate += 1

    out = {}
    for ro in ("R", "C"):  # reply leads
        rws = [r for r in man["rows"] if r["readout"] == ro]
        for f in PTS:
            per_probe, vals = {}, {}
            for r in rws:
                lab, k = r["inventory_key"], r["npz_label"]
                sim, r1, r2, rm = (arr(ro, f"{k}__joint__{f}__S_new__{c}") for c in ("sim_at_S__primary", "run1_at_B", "run2_at_B", "real_mean_at_B"))
                v = {"i_sim_minus_run1": sim - r1, "ii_sim_minus_run2": sim - r2, "iii_run2_minus_run1": r2 - r1, "iv_delta_full": sim - rm}
                pt = hd(lab)["points"]["joint"][f]
                g0(v["i_sim_minus_run1"][0], pt["gap"]["sim_minus_run1"]["point"], f"{lab} {f} (i)")
                g0(v["ii_sim_minus_run2"][0], pt["gap"]["sim_minus_run2"]["point"], f"{lab} {f} (ii)")
                g0(v["iii_run2_minus_run1"][0], pt["catch"]["run2@B"]["point"] - pt["catch"]["run1@B"]["point"], f"{lab} {f} (iii)")
                g0(v["iv_delta_full"][0], pt["gap"]["delta_full"]["point"], f"{lab} {f} (iv)")
                vals[lab] = {c: x * 100.0 for c, x in v.items()}
                per_probe[lab] = {"group": r["loro_group"]} | {c: summ(x) for c, x in vals[lab].items()}
            groups = sorted({r["loro_group"] for r in rws} - {"STRESS"})
            per_group, G = {}, {}
            for gname in groups:
                mem = [r["inventory_key"] for r in rws if r["loro_group"] == gname]
                G[gname] = {c: np.mean(np.stack([vals[m][c] for m in mem]), axis=0) for c in CONTRASTS}  # NaN if any member NaN
                per_group[gname] = {"members": mem} | {c: summ(x) for c, x in G[gname].items()}
                lr = loro["cells"][f"{ro}|{f}|primary"]["group_gaps_draw0"][gname]
                g0(G[gname]["iv_delta_full"][0], lr, f"group {gname} {ro} {f} (iv) vs LORO record")
            if len(groups) != 6:
                raise SystemExit(f"{ro} {f}: {len(groups)} groups, not 6")
            mean6 = {c: summ(np.mean(np.stack([G[gname][c] for gname in groups]), axis=0)) for c in CONTRASTS}
            out[f"{ro}|{f}"] = {"per_group": per_group, "mean_over_6_groups": mean6,
                                "per_probe": per_probe, "stress": {lab: per_probe[lab] for lab in per_probe if per_probe[lab]["group"] == "STRESS"}}
    if worst > 1e-9:
        raise SystemExit(f"draw-0 gate failed: max |diff| {worst}")
    res = {"label": "POST HOC launch check on the +311 reply offset; PRELIMINARY; '+311: Q4 FINAL; Q1, Q3 interim'; POST HOC: values of Delta_full were read",
           "rule": "experiments/2026-10-15_loro_sep_09/RULE.md @ d13ad9a4b", "git": git, "stamp": stamp, "versions": {"numpy": np.__version__},
           "contrasts": CONTRASTS, "units": "percentage points (fractions x 100)", "draws": "joint draws only (run455's main draws over the 506-main universe), equal-main as stored",
           "inputs": {"npz": {ro: {"volume": IN_VOL, "path": p_, "sha256": s_} for ro, (p_, s_) in NPZ.items()},
                      "records": {nm: {"commit": c_, "path": p_, "sha256": s_} for nm, (c_, p_, s_) in RECS.items()}},
           "gates": {"shas": True, "draw0_checked": n_gate, "draw0_max_abs_diff": worst}, "results": out}
    dest = Path("/out") / "loro_sep_09" / f"launch_check311_{stamp}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    out_vol.commit()
    return res


@app.local_entrypoint()
def main():
    def g(*a):
        return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout

    if g("status", "--porcelain", "--", str(HERE)).strip():
        raise SystemExit("refusing to run: commit this folder first")
    head = g("rev-parse", "HEAD").decode().strip()
    rec_bytes = {nm: g("show", f"{c_}:{p_}") for nm, (c_, p_, _) in RECS.items()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = run.remote(rec_bytes, head, stamp)
    dest = HERE / "results" / f"launch_check311_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    print(json.dumps(res["gates"]), "\nwrote", dest.name)
