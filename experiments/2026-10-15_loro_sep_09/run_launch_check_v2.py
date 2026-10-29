"""v2: SUPERSEDES launch_check311 (fold -1-dropped inputs; kept with its NOTE sidecar @ 3302b84df). Same computation on the CORRECTED pooled-455 inputs
@ 2892fa260 per AMENDMENT 1 @ 22c077ea2. Sol (a) / c_asis rows are gated against the corrected per-set rows (their hot-deck rows are fold -1-dropped);
every other row and every group other than other-model-real must equal the superseded record at draw 0.

POST HOC launch check on the +311 reply offset (RULE.md in this folder, committed before computing @ d13ad9a4b).
Per probe and per LORO group (manifest loro_sep_manifest_311.json @ cd1afb33e), at per-action 1e-2 and 5e-2, both readouts:
(i) sim@S - run1@B, (ii) sim@S - run2@B, (iii) run2@B - run1@B, (iv) sim@S - real_mean@B (= Delta_full).
Inputs: the S_new per-origin catches of 2e's pooled-455 npzs (joint draws, equal-main as stored). Draw-0 gates against the hot-deck
records @ c838127d3 and against the LORO +311 record @ 2351051cc. POST HOC, PRELIMINARY, "+311: Q4 FINAL; Q1, Q3 interim".
One Modal CPU container.

  modal run --detach experiments/2026-10-15_loro_sep_09/run_launch_check_v2.py
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
APP = "probe-shift_aswin-surya_sub-loro_launchcheck311v2_kill-2026-10-22"
IN_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
OUT_VOL = "probe-shift_aswin-surya_sub-loro_results_kill-2026-11-14"  # existing; not created here
NPZ = {"C": ("xfer/fpr507prep/pooled455_2026-10-15_151852_perdraw.npz", "219681d83278c349da16cf89ba910814852b053e4b5a6f39394659b821f9f003"),
       "R": ("xfer/fpr507prep/pooled455_R_2026-10-15_155043_perdraw.npz", "6470f012fda86ff9dc1d50b32a2aac9e2e57fcdd925bccbeb6822893d777a95f")}  # AMENDMENT 2: reissued path, sha unchanged
RD = "experiments/2026-10-15_fpr_sweep_507_prep/results"
RECS = {  # name: (commit, path, sha256)
    "hd_C": ("c838127d3", f"{RD}/peraction311_all_hotdeck_C_2026-10-15_140206.json", "8a00f2060ef6aec70822a6e6af3f8169540b2e513641a82a51749ff6bdc69c41"),
    "hd_R": ("c838127d3", f"{RD}/peraction311_all_hotdeck_R_2026-10-15_140206.json", "a1a150fb20c54c519d5ab127ca675d610efea41e54ce824d3eb6cb577fcfce78"),
    "hd_O": ("c838127d3", f"{RD}/peraction311_all_hotdeck_O_2026-10-15_140209.json", "bcd97eb54097bedb94e43a7672ab190cdb4fa24bc3db89307f6ac98b60a4cf0a"),
    "hd_X": ("c838127d3", f"{RD}/peraction311_all_hotdeck_X_2026-10-15_140208.json", "c232bbb6d7d3692aa8d5e9239bde4c9f0be96e0487228583c138eed7d0f8a6d5"),
    "pooled_C": ("2892fa260", f"{RD}/pooled455_C_2026-10-15_151852.json", "edfef938d3520cb736ae7bed970e82b6e1371dfbf37f5621afdf2156fa317146"),
    "pooled_R": ("2892fa260", f"{RD}/pooled455_R_2026-10-15_151846.json", "54ffd0778514bfd0c0b7ef0d266889a6f951dcfd4297c6dcc384baf7eed1313b"),
    "amend2": ("2a881b55d", "experiments/2026-10-14_priority1_spec_09/loro_sep/loro_sep_manifest_amend2.json", "ad8e997ec6e27c83d60033e6f0c47d6063fe2155b576fc4ec99d163354ba3ef4"),
    "amend_311": ("22c077ea2", "experiments/2026-10-14_priority1_spec_09/loro_sep/loro_sep_manifest_311_amend1.json", "0c89f34aa63257480392d998a90ebef3adea0a4efe52a07049ddbbaee8352f63"),
    "manifest_311": ("cd1afb33e", "experiments/2026-10-14_priority1_spec_09/loro_sep/loro_sep_manifest_311.json", "2dd2d5397b8ccb9f070e4301a41112740259560f9387692c1f6335ad7c1cfa73"),
}
PREV = ("dc34efd81", "experiments/2026-10-15_loro_sep_09/results/launch_check311_2026-10-15_150414.json", "ac040debf80ff11dc152b9f72fb1d6e11774a75b3877fbba2f489f4ef399dd27")  # the superseded record (previous-record gate)
FM1_ROWS = {"Sol (a) C", "Sol (a) R", "c_asis C", "c_asis R"}  # recomputed under 25h §1 in the corrected inputs
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
def run(rec_bytes: dict, prev_bytes: bytes, loro_bytes: bytes, loro_path: str, git: str, stamp: str, readouts: str) -> dict:
    import numpy as np

    for nm, (_, _, s_) in RECS.items():
        if sha(rec_bytes[nm]) != s_:
            raise SystemExit(f"{nm} sha256 differs")
    rec = {nm: json.loads(b) for nm, b in rec_bytes.items()}
    man, loro = rec["manifest_311"], json.loads(loro_bytes)
    RO = tuple(readouts)  # readouts run here; the others are pending (e.g. the overwritten reply npz)
    if not RO or any(ro not in NPZ for ro in RO) or len(set(RO)) != len(RO):
        raise SystemExit(f"bad readouts {readouts!r}")
    if "v2" not in loro.get("label", "") or any(loro["inputs"]["npz"].get(ro, {}).get("sha256") != NPZ[ro][1] for ro in RO):
        raise SystemExit("the LORO record for the (iv) gate is not the v2 record on these npzs")
    Z = {}
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

    if "R" in RO and (rec["amend2"]["reply_npz"]["path"], rec["amend2"]["reply_npz"]["sha256"]) != NPZ["R"]:
        raise SystemExit("reply npz pin differs from AMENDMENT 2")
    if any(rec["amend_311"]["input_provenance"]["npz"][ro]["sha256"] != NPZ[ro][1] for ro in RO):
        raise SystemExit("npz pins differ from AMENDMENT 1")
    if sha(prev_bytes) != PREV[2]:
        raise SystemExit("superseded record sha256 differs")
    prev = json.loads(prev_bytes)

    def src(lab):
        """Draw-0 reference row: the corrected pooled record's per-set S_new row for Sol (a) / c_asis, else the hot-deck row."""
        return rec[f"pooled_{lab[-1]}"]["rows"][lab]["S_new"] if lab in FM1_ROWS else hd(lab)

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
    for ro in (x for x in ("R", "C") if x in RO):  # reply leads
        rws = [r for r in man["rows"] if r["readout"] == ro]
        for f in PTS:
            per_probe, vals = {}, {}
            for r in rws:
                lab, k = r["inventory_key"], r["npz_label"]
                sim, r1, r2, rm = (arr(ro, f"{k}__joint__{f}__S_new__{c}") for c in ("sim_at_S__primary", "run1_at_B", "run2_at_B", "real_mean_at_B"))
                v = {"i_sim_minus_run1": sim - r1, "ii_sim_minus_run2": sim - r2, "iii_run2_minus_run1": r2 - r1, "iv_delta_full": sim - rm}
                pt = src(lab)["points"]["joint"][f]
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
    # previous-record gate: every row other than Sol (a) / c_asis, and every group other than other-model-real, equals the superseded record at draw 0
    pworst, pn = 0.0, 0
    for key, x in out.items():
        px = prev["results"][key]
        for lab, row in x["per_probe"].items():
            if lab in FM1_ROWS:
                continue
            for c in CONTRASTS:
                pworst, pn = max(pworst, abs(row[c]["draw0"] - px["per_probe"][lab][c]["draw0"])), pn + 1
        for gname, gr in x["per_group"].items():
            if gname == "other-model-real":
                continue
            for c in CONTRASTS:
                pworst, pn = max(pworst, abs(gr[c]["draw0"] - px["per_group"][gname][c]["draw0"])), pn + 1
    if pworst > 1e-9:
        raise SystemExit(f"previous-record gate failed: max |diff| {pworst}")
    res = {"label": "v2 (SUPERSEDES the fold -1-dropped launch_check311). POST HOC launch check on the +311 reply offset; PRELIMINARY; '+311: Q4 FINAL; Q1, Q3 interim'; POST HOC: values of Delta_full were read",
           "rule": "experiments/2026-10-15_loro_sep_09/RULE.md @ d13ad9a4b", "git": git, "stamp": stamp, "versions": {"numpy": np.__version__},
           "contrasts": CONTRASTS, "units": "percentage points (fractions x 100)", "draws": "joint draws only (run455's main draws over the 506-main universe), equal-main as stored",
           "readouts_run": list(RO), "readouts_pending": [ro for ro in ("R", "C") if ro not in RO],
           "inputs": {"npz": {ro: {"volume": IN_VOL, "path": NPZ[ro][0], "sha256": NPZ[ro][1]} for ro in RO},
                      "records": {nm: {"commit": c_, "path": p_, "sha256": s_} for nm, (c_, p_, s_) in RECS.items()}},
           "gates": {"shas": True, "draw0_checked": n_gate, "draw0_max_abs_diff": worst, "prev_values_checked": pn, "prev_max_abs_diff": pworst},
           "supersedes": {"commit": PREV[0], "path": PREV[1], "sha256": PREV[2]}, "loro_record_for_iv_gate": {"path": loro_path, "sha256": sha(loro_bytes), "commit": git}, "inputs_amendment": "AMENDMENT 1 @ 22c077ea2 (Sol (a) / c_asis S_new under 25h §1: 272 cells, 306 / 194 / 195)", "results": out}
    dest = Path("/out") / "loro_sep_09" / f"launch_check311v2_{readouts}_{stamp}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    out_vol.commit()
    return res


@app.local_entrypoint()
def main(readouts: str = "C"):
    def g(*a):
        return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout

    if g("status", "--porcelain", "--", str(HERE)).strip():
        raise SystemExit("refusing to run: commit this folder first")
    head = g("rev-parse", "HEAD").decode().strip()
    rec_bytes = {nm: g("show", f"{c_}:{p_}") for nm, (c_, p_, _) in RECS.items()}
    prev_bytes = g("show", f"{PREV[0]}:{PREV[1]}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    hits = [x for x in g("ls-files", "results/").decode().split() if x.startswith(f"results/loro_sep311v2_{readouts}_") and x.endswith(".json") and "__NOTE" not in x]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one committed loro_sep311v2 record, found {hits}")
    loro_path = f"experiments/2026-10-15_loro_sep_09/{hits[0]}"
    loro_bytes = g("show", f"HEAD:{loro_path}")
    res = run.remote(rec_bytes, prev_bytes, loro_bytes, loro_path, head, stamp, readouts)
    dest = HERE / "results" / f"launch_check311v2_{readouts}_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    print(json.dumps(res["gates"]), "\nwrote", dest.name)
