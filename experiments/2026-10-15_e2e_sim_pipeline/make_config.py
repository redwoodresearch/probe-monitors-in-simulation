"""Build config/inputs_e2e.json from committed sources only (no retyped shas): the common table's config and pins, 2e's Sol
sim-honest records (aswin/cmd-readout-curve), 4f's p2 sim-honest manifests (aswin/p2-simh-scores) and 09's A25 record.

  python3 experiments/2026-10-15_e2e_sim_pipeline/make_config.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FPR = HERE.parent / "2026-10-14_fpr_sweep_unit"
git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731

SOL_REC = {"b1": "c108cef81:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/simh_score_2026-10-14_230251.json",
           "b2": "d13079e99:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/simh_score_2026-10-14_232103.json"}
P2_REC = {"b1": "eb372c6f7:experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-14_212541.json",
          "b2": "3f8d3aa59:experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-14_231508.json"}
A25 = "63787479b:experiments/2026-10-14_simhonest_a25_09/results/a25_full_2026-10-14_232415.json"
LABELS = ["Sol (a) R", "Sol (a) C", "FIBS R", "FIBS C", "harmfulness R", "harmfulness C", "refusal R", "refusal C", "high-stakes R", "high-stakes C",
          "c_asis R", "c_asis C"]
SOL_MODEL = {"Sol (a)": "a_tight", "c_asis": "c_asis_tight"}
P2_NAME = {"FIBS": "fibs", "harmfulness": "harmfulness", "refusal": "refusal", "high-stakes": "highstakes"}
A25_NAME = {"Sol (a)": "sol_a", "FIBS": "fibs", "harmfulness": "harmfulness", "refusal": "refusal", "high-stakes": "highstakes", "c_asis": "c_asis"}
# own family (2e's resume, 11:4x): rows from the common table's config @ 0a7244d86 (record crossrecipe_2026-10-15_110033), sim-honest from famexp's sealed pack
OWN_LABELS = ["own_a R", "own_a C", "own_frac10_L48 R", "own_frac10_L48 C", "own_frac25_L48 R", "own_frac25_L48 C", "own_frac100_L48 R", "own_frac100_L48 C"]
CROSS2 = "0a7244d86:experiments/2026-10-14_fpr_sweep_unit/config/inputs_crossrecipe.json"
CROSS2_REC = "0a7244d86:experiments/2026-10-14_fpr_sweep_unit/results/crossrecipe_2026-10-15_110033.json"
FAM_REC = "a02df5169:experiments/2026-10-14_p2_family_expansion/results/simh_ownfam_2026-10-15_114221.json"
PREV = "results/e2e_2026-10-15_134139_reduce_2026-10-15_135347.json"  # v5b must reproduce v5 at every existing key (impactful-only DT rows are added)
SOL_REPLY = "35f648b53:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/simh_reply_rows_2026-10-15_115936.json"
P2_REPLY = {"b1": "bcd281ef1:experiments/2026-10-14_p2_simh_scores/results/simh_reply_rows_manifest_2026-10-15_115736.json",
            "b2": "bcd281ef1:experiments/2026-10-14_p2_simh_scores/results/simh_reply_rows_manifest_2026-10-15_115824.json"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


cross_cfg_b = (FPR / "config" / "inputs_crossrecipe.json").read_bytes()
cross = json.loads(cross_cfg_b)
loc = {}
for k in ("labels", "t196", "r36", "v6", "asis", "rs::fibs|reply_mean|corrected|none", "rs::harmfulness|reply_mean|corrected|none",
          "rs::refusal|reply_mean|corrected|none", "rs::highstakes|reply_mean|corrected|none"):
    loc[k] = cross["local"][k]
loc["rec_b"] = {"path": "../2026-10-14_fpr_sweep_unit/results/fprsweep_unit_2026-10-15_003643.json", "sha256": cross["local"]["rec_b"]["sha256"]}
rc = (FPR / "results" / "crossrecipe_2026-10-15_100733.json").read_bytes()
loc["rec_cross"] = {"path": "../2026-10-14_fpr_sweep_unit/results/crossrecipe_2026-10-15_100733.json", "sha256": sha(rc)}
loc["a25"] = {"git": A25, "sha256": sha(git("show", A25))}
sol = {b: json.loads(git("show", g)) for b, g in SOL_REC.items()}
p2 = {b: json.loads(git("show", g)) for b, g in P2_REC.items()}
rows = []
for r in cross["rows"]:
    if r["label"] not in LABELS:
        continue
    fam, ro = r["family"], r["readout"]
    batches = []
    for b in ("b1", "b2"):
        if fam in SOL_MODEL:
            rec = sol[b]
            od = rec["out_dir"]
            fn = f"{SOL_MODEL[fam]}__avg_top2.json" if ro == "R" else f"{SOL_MODEL[fam]}__cmd_A.npz"
            f = rec["outputs"]["files"][fn]
            batches.append({"batch": b, "volume": "convaudit", "path": f"{od}/{fn}", "sha256": f["sha256"], "n_runs": f["n_runs"], "n_rows": f.get("n_rows"),
                            "manifest": {"volume": "convaudit", "path": f"{od}/manifest_sol.json", "sha256": rec["outputs"]["manifest_sha256"]},
                            "record": SOL_REC[b]})
        else:
            rec = p2[b]
            (od, fo), = rec["folders"].items()
            nm = P2_NAME[fam]
            files = fo["files"]
            cnt = fo["counts"][nm]["command"]
            ent = {"batch": b, "volume": "convaudit", "n_runs": cnt["runs"], "record": P2_REC[b],
                   "command": {"path": f"{od.lstrip('/')}/{nm}__command.npz", "sha256": files[f"{nm}__command.npz"], "n_rows": cnt["rows"]},
                   "run_json": {"path": f"{od.lstrip('/')}/{nm}__reply_mean.json", "sha256": files[f"{nm}__reply_mean.json"]}}
            batches.append(ent)
    if ro == "R":  # per-row reply sim-honest rows (Sol @ 35f648b53; p2 @ bcd281ef1), for the reply unit level
        if fam in SOL_MODEL:
            srr = json.loads(git("show", SOL_REPLY))
            for bt in batches:
                ent = srr["batches"][f"simh/full_{bt['batch']}/sol"]
                fn = f"{SOL_MODEL[fam]}__avg_top2.npz"
                bt["reply_rows"] = {"volume": "convaudit", "path": ent["path"].split(":", 1)[1].strip("/") + "/" + fn, "sha256": ent["files"][fn]["sha256"],
                                    "n_rows": ent["files"][fn]["n_rows"], "record": SOL_REPLY}
        else:
            for bt in batches:
                rr_ = json.loads(git("show", P2_REPLY[bt["batch"]]))
                (fo_, v_), = rr_["folders"].items()
                fn = f"{P2_NAME[fam]}__reply_mean.npz"
                bt["reply_rows"] = {"volume": "convaudit", "path": f"{fo_.strip('/')}/{fn}", "sha256": v_["files"][fn], "n_rows": v_["counts"][P2_NAME[fam]]["reply_mean"]["rows"],
                                    "record": P2_REPLY[bt["batch"]]}
    rows.append(r | {"simh": {"kind": ("sol_" if fam in SOL_MODEL else "p2_") + ro, "batches": batches}, "a25_name": A25_NAME[fam]})
assert [r["label"] for r in rows] == [x for x in [r_["label"] for r_ in cross["rows"]] if x in LABELS] and len(rows) == len(LABELS)
cross2_b = git("show", CROSS2)
cross2 = json.loads(cross2_b)
fam = json.loads(git("show", FAM_REC))
for lab in OWN_LABELS:
    r = next(r_ for r_ in cross2["rows"] if r_["label"] == lab)
    old = next((r_ for r_ in cross["rows"] if r_["label"] == lab), None)
    assert old is not None and json.dumps(old, sort_keys=True) == json.dumps(r, sort_keys=True), f"{lab}: row spec changed between the two configs"
    if r["family"] == "own_a":
        fn = "own_a_tight__avg_top2" if r["readout"] == "R" else "own_a_tight__cmd_A"
    else:
        fn = f"{lab.split()[0]}__{r['readout']}"
    bt = {"batch": "primary_v3", "volume": "convaudit", "path": f"simh/primary_v3/famexp/{fn}.npz", "sha256": fam["files"][f"{fn}.npz"], "n_runs": fam["counts"][fn]["n_runs"],
          "n_rows": fam["counts"][fn]["n_units"], "manifest": {"volume": "convaudit", "path": "simh/primary_v3/famexp/manifest_famexp.json", "sha256": fam["manifest_sha256"]},
          "record": FAM_REC}
    rows.append(r | {"simh": {"kind": "fam_" + r["readout"], "batches": [bt]}, "a25_name": None, "gx_ref": "rec_cross2"})
# DT (Aswin-approved sim-honest scoring, 4f; 2e's relay 13:3x): the common table's DT rows @ 0a7244d86, sim-honest per-row files on the Volume,
# gated on the manifests' sha prefixes (the DT record commits after 4f's chain) and on per-file shas listed in them
DT_MAN = {"b1": ("simh_full_b1", "ef4e5eb9aeafc94a", 145, 5478), "b2": ("simh_full_b2", "1e40225670694b79", 4, 162)}
for r in cross2["rows"]:
    if r["recipe"] != "DT":
        continue
    design = r["label"].split()[0]
    bts = [{"batch": b_, "volume": "convaudit", "path": f"simh/primary_v3/dt/{d_}/{design}.npz", "n_runs": n_, "n_rows": nr_,
            "manifest": {"volume": "convaudit", "path": f"simh/primary_v3/dt/{d_}/manifest_dt.json", "sha256_prefix": pre_}} for b_, (d_, pre_, n_, nr_) in DT_MAN.items()]
    rows.append(r | {"simh": {"kind": "dt_C", "batches": bts}, "a25_name": None, "gx_ref": "rec_cross2"})
# v5b (2e: James/0d's DT impactful-only variant): the same files, impactful == 1 rows only on every side; a run with no impactful call never alarms
# and contributes no unit to any moment fit; per action only (run level skipped); no common-table (GX) reference exists for these rows
for r in [r_ for r_ in rows if r_.get("source") == "dt"]:
    design = r["label"].split()[0]
    rows.append(r | {"label": f"{design}_imp C", "family": r["family"] + " (impactful calls only)", "impactful_only": True, "gx_ref": None})
CK = "572f67dbb:experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_primary_v3.json"
loc["counted_keys"] = {"git": CK, "sha256": sha(git("show", CK))}
loc["own_a_record"] = cross["local"]["own_a_record"]
loc["rec_cross2"] = {"git": CROSS2_REC, "sha256": sha(git("show", CROSS2_REC))}
loc["rec_prev"] = {"path": PREV, "sha256": sha((HERE / PREV).read_bytes())}
cfg = {"label": "End-to-end simulated pipeline (POST HOC extension of 09's Append 25); every input sha256-pinned from committed sources by make_config.py",
       "crossrecipe_config": {"path": "../2026-10-14_fpr_sweep_unit/config/inputs_crossrecipe.json", "sha256": sha(cross_cfg_b)},
       "crossrecipe_config_own_rows": {"git": CROSS2, "sha256": sha(cross2_b)},
       "local": loc, "rows": rows, "recipe_order": ["other-model-real", "public direction", "stress test"],
       "run_alphas": [0.01, 0.02, 0.05, 0.10], "a25_alphas": [0.01, 0.05], "expected": {"cohort": 183, "estimation_half": 90, "simh_runs": 149},
       "curve_probes": ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes", "own_a", "own_frac100_L48"], "not_available": {"own_frac*_L16/L32": "not scored on sim-honest (09's registration covers L48 and own_a only)"}}
(HERE / "config").mkdir(exist_ok=True)
(HERE / "config" / "inputs_e2e.json").write_text(json.dumps(cfg, indent=1) + "\n")
print(json.dumps({"rows": len(rows), "local": sorted(loc), "sha256": sha((HERE / "config" / "inputs_e2e.json").read_bytes())}, indent=1))
