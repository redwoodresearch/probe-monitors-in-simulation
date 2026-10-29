"""Build config/inputs_rms_<set>.json for run_rms_hits.py from committed sources only:
- 09's five-arm cells v2 @ 668a3eb91 (sha256 pinned): the set's cells and, per cell and arm, the resolved scorer folder and row key;
- folder manifest pins: the IL FINAL record @ d851974f5 (IL folders) and 09's FINAL spec @ 3ce6ea55e (scale-up arms), as config/inputs_znew_final.json carries them;
- H_B: 09's FINAL S_new pool keys (306) on the honest-B arms; H_sim: the third pass's roots and pins (config/inputs_znew_final.json).

  python3 experiments/2026-10-15_il_z_e2e/make_config_rms.py S_new
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
V2 = ("668a3eb91:experiments/2026-10-15_il_analysis_09/results/il_five_arm_cells_2026-10-15_v2.json", "5ca4574c8d627125f0a02f87e3a1e8f91d94dd307793c9b6f72e5fb629c6db14")
SPEC = ("3ce6ea55e:experiments/2026-10-14_xerr_scaleup_09/specs/final_spec_3_launch.json", "6bd2ab192337e1023a5454c17217eb2b44f0f7c94165cd8671639443d5f90a5a")
ARM_OF = {"sim": "sim1", "sim2": "sim2", "run1": "run1", "run2": "run2", "run3": "run3"}
SET = sys.argv[1]
S1 = "--s1" in sys.argv  # S_new under 25h §1 on the fifth pass's roots
assert SET in ("S_new", "S_newmains")
IL_EXT = ("4d92cc2ca:experiments/2026-10-15_il_analysis_09/results/il_ext_final_result_il_2026-10-15_134705.json", None)
XERR_EXT = ("1a7e95259:experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_113025.json", None)
MAPB = "83cbff1f9"  # aswin/cmd-readout-curve head at build (verified Sol mappings)
NM_MAPS = {"s1": "simh_nm360_s1__2026-10-15_144824.json", "s2": "simh_nm360_s2__2026-10-15_144826.json", "s3": "simh_nm360_s3__2026-10-15_140926.json",
           "s4d": "simh_nm360_s4d__2026-10-15_145133.json", "s5": "simh_nm360_s5__2026-10-15_144834.json"}
# 2e's relay of 09's read status (~14:0x and the RMS go): S1 s1-s5, S2 (try2, s2), S3 (s2, s3, final), S4d (all); p2 pins from 4f's records
NM_ROOTS = {"simh_nm360_s1_sim_s1_v1__sim_honest": "ddc8e19da:simh_manifest_2026-10-15_122533.json", "simh_nm360_s1_sim_s2_v1__sim_honest": "2b1060531:simh_manifest_2026-10-15_125458.json",
            "simh_nm360_s1_sim_s3_v1__sim_honest": "33d0e0a95:simh_manifest_2026-10-15_132943.json", "simh_nm360_s1_sim_s4_v1__sim_honest": "5144587e6:simh_manifest_2026-10-15_133955.json",
            "simh_nm360_s1_sim_s5_v1__sim_honest": "d5396f4ee:simh_manifest_2026-10-15_135108.json",
            "simh_nm360_s2_sim_s1_v1_try2__sim_honest": "5144587e6:simh_manifest_2026-10-15_134516.json", "simh_nm360_s2_sim_s2_v1__sim_honest": "d1f237fdd:simh_manifest_2026-10-15_135921.json",
            "simh_nm360_s3_sim_s2_v1__sim_honest": "33d0e0a95:simh_manifest_2026-10-15_132944.json", "simh_nm360_s3_sim_s3_v1__sim_honest": "5144587e6:simh_manifest_2026-10-15_133954.json",
            "simh_nm360_s3_sim_v1_try2__sim_honest": "d1f237fdd:simh_manifest_2026-10-15_135922.json",
            "simh_nm360_s4d_sim_s1_v1__sim_honest": "f155abeaa:simh_manifest_2026-10-15_122812.json", "simh_nm360_s4d_sim_s2_v1__sim_honest": "2b1060531:simh_manifest_2026-10-15_125502.json",
            "simh_nm360_s4d_sim_v1__sim_honest": "ddaf4856d:simh_manifest_2026-10-15_131231.from_volume.json",
            # 09's FINAL GO (relayed 14:5x): S1 s6, s7 and the S1 remainder; S2 s3 and the S2 remainder _try2; S5 s1 (S4r out)
            "simh_nm360_s1_sim_s6_v1__sim_honest": "123fb303f:simh_manifest_2026-10-15_140934.json", "simh_nm360_s1_sim_s7_v1__sim_honest": "b74819a71:simh_manifest_2026-10-15_143057.json",
            "simh_nm360_s1_sim_v1__sim_honest": "08b95b38f:simh_manifest_2026-10-15_143842.json", "simh_nm360_s2_sim_s3_v1__sim_honest": "123fb303f:simh_manifest_2026-10-15_141825.json",
            "simh_nm360_s2_sim_v1_try2__sim_honest": "08b95b38f:simh_manifest_2026-10-15_143844.json", "simh_nm360_s5_sim_s1_v1__sim_honest": "dfc192597:simh_manifest_2026-10-15_143540.json"}
BATCH_LABELS = {"simh_nm360_s1": "nm360 S1 FINAL (60 authoritative keys)", "simh_nm360_s2": "nm360 S2 FINAL; replacement pending: nm150 (terminal error, standing rule @ 9843c6d7a / 87cbaca6a)",
                "simh_nm360_s3": "nm360 S3 FINAL", "simh_nm360_s4d": "nm360 S4d FINAL", "simh_nm360_s5": "nm360 S5 s1 (interim)"}

if SET == "S_newmains":
    v2b = git("show", V2[0])
    assert sha(v2b) == V2[1]
    v2 = json.loads(v2b)
    cells = v2["S_newmains_five_arm"]["cells"]
    rs = v2["resolved_sources"]["cells"]
    assert len(cells) == 239 and all(rs[c]["set"] == "S_newmains_five_arm" for c in cells)
    ilxb, xxb = git("show", IL_EXT[0]), git("show", XERR_EXT[0])
    ilx = json.loads(ilxb)["result_il"]["inputs"]
    xx = json.loads(xxb)["result"]
    sui = xx["scale_up_inputs"]
    folders, arm_rows = {}, {a: {} for a in ARM_OF.values()}
    for c in cells:
        for w, a in ARM_OF.items():
            e = rs[c][w]
            assert "scorer_folder" in e and e["row_key"] == f"{w}|{c}", (c, w)
            fol = e["scorer_folder"]
            arm_rows[a][c] = [fol, e["row_key"]]
            src_ = ilx.get(fol) or sui.get(fol)
            if src_ is None:
                raise SystemExit(f"no manifest pin for {fol}")
            folders[fol] = {"sol": src_["sol_manifest"], "p2": src_["p2_manifest"]}
    ext = xx["calibration_records"]["extended"]
    pools_ = {tuple(sorted(v["schemes"][v["primary"]]["pool_keys_newmains"])) for v in ext.values()}
    assert len(pools_) == 1
    hb_keys = sorted(next(iter(pools_)))
    hb = sorted(a for a in sui if a.startswith("nm360_") and a.endswith("__honest_b"))
    for a in hb:
        folders[a] = {"sol": sui[a]["sol_manifest"], "p2": sui[a]["p2_manifest"]}
    roots = []
    for rt, rec in NM_ROOTS.items():
        q = rt.split("_")[2]
        mb = git("show", f"{MAPB}:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/simh_mapping/{NM_MAPS[q]}")
        ent = [r_ for r_ in json.loads(mb)["roots"] if r_["routed_folder"] == rt]
        assert len(ent) == 1, rt
        cm_, fn_ = rec.split(":", 1)
        pb_ = git("show", f"{cm_}:experiments/2026-10-14_p2_simh_scores/results/{fn_}")
        pj = json.loads(pb_)
        fk = [k_ for k_ in (pj.get("folders") or {}) if k_.rstrip("/").endswith(rt)]
        pf = pj["folders"][fk[0]]["files"] if fk else pj["files"]
        ppath = (fk[0].strip("/") if fk else pj["folder"].strip("/"))
        assert ppath.endswith(rt), (rt, ppath)
        roots.append({"root": rt, "batch": f"simh_nm360_{q}", "sol": {"path": ent[0]["path_written"], "files": ent[0]["files"], "mapping": [f"{MAPB}:{NM_MAPS[q]}", sha(mb)]},
                      "p2": {"path": ppath, "files": pf, "record": [rec, sha(pb_)]}})
    zc = json.loads((HERE / "config/inputs_znew_final.json").read_text())
    for q_ in ("s1", "s2", "s3", "s4d", "s5"):  # one consistent root set across Sol and p2: the Sol mapping's roots filtered to NM_ROOTS, each with a p2 record
        mp_ = {r_["routed_folder"] for r_ in json.loads(git("show", f"{MAPB}:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/simh_mapping/{NM_MAPS[q_]}"))["roots"]}
        want_ = {r_ for r_ in NM_ROOTS if r_.split("_")[2] == q_}
        assert want_ <= mp_, (q_, want_ - mp_)
    assert {r_["root"] for r_ in roots} == set(NM_ROOTS)
    out = {"set": "S_newmains", "set_label": "S_newmains five-arm (239), from 09's five-arm v2; POST HOC; H_sim = nm360 FINAL set (S1, S2, S3, S4d FINAL; S5 s1)",
           "batch_labels": BATCH_LABELS, "withhold_5fold": True,
           "points": [0.001, 0.01], "cells": cells, "cell_fold": None, "arm_rows": arm_rows, "folders": folders, "H_B": {"folders": hb, "keys": hb_keys},
           "H_sim": {"label": "nm360 interim: " + ", ".join(NM_ROOTS), "roots": roots, "exclude_keys": []},
           "sources": {"five_arm_v2": list(V2), "il_ext_record": [IL_EXT[0], sha(ilxb)], "xerr_ext_record": [XERR_EXT[0], sha(xxb)], "sol_mappings_commit": MAPB},
           "fold_rule": "DEVIATION SENSITIVITY ONLY (09): 5-fold rows use the scorer's designated fold per row and drop fold -1 / non-finite rows; 25h §1 requires the five-model mean for fold -1 units. Sol (a)/c_asis rows are WITHHELD from the primary until 09 rules",
           "rows": [{k: r_[k] for k in ("label", "sub", "file", "folds")} for r_ in zc["rows"]]}
    (HERE / "config" / f"inputs_rms_{SET}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(SET, "cells", len(cells), "folders", len(folders), "H_B keys", len(hb_keys), "H_B folders", hb, "roots", len(roots))
    raise SystemExit(0)
v2b, specb = git("show", V2[0]), git("show", SPEC[0])
assert sha(v2b) == V2[1] and sha(specb) == SPEC[1]
v2 = json.loads(v2b)
cells = v2["S_new_on_M"]["cells"]
rs = v2["resolved_sources"]["cells"]
assert len(cells) == 257 and all(rs[c]["set"] == "S_new_on_M" for c in cells)
ZCFG = "config/inputs_znew_pass5.json" if S1 else "config/inputs_znew_final.json"
zc = json.loads((HERE / ZCFG).read_text())
spec_arms = json.loads(specb)["arms"]
folders, arm_rows = {}, {a: {} for a in ARM_OF.values()}
for c in cells:
    for w, a in ARM_OF.items():
        e = rs[c][w]
        assert "scorer_folder" in e and e["row_key"] == f"{w}|{c}", (c, w)
        fol = e["scorer_folder"]
        arm_rows[a][c] = [fol, e["row_key"]]
        if fol in zc["il_manifests"]:
            folders[fol] = zc["il_manifests"][fol]
        elif fol in spec_arms:
            folders[fol] = {"sol": spec_arms[fol]["sol_manifest"], "p2": spec_arms[fol]["p2_manifest"]}
        else:
            raise SystemExit(f"no manifest pin for {fol}")
hb = sorted(a for a in spec_arms if a.endswith("_main_v1__honest_b"))
for a in hb:
    folders[a] = {"sol": spec_arms[a]["sol_manifest"], "p2": spec_arms[a]["p2_manifest"]}
fom = zc["fold_of_main"]
out = {"set": "S_new", "set_label": ("S_new on Append 36's FINAL M (257), from 09's five-arm v2; POST HOC; H_sim = the fifth pass's 24 FINAL roots; Sol (a)/c_asis "
                                    "under 25h §1 fold -1 handling" if S1 else "S_new on Append 36's FINAL M (257), from 09's five-arm v2; POST HOC; H_sim interim (third-pass roots)"),
       "points": [0.001, 0.01], "cells": cells, "cell_fold": {c: fom.get(m_, -1) for c, m_ in zc["cell_main"].items()},  # every +311 cell (attack, H_B and H_sim cells)
       "arm_rows": arm_rows, "folders": folders, "H_B": {"folders": hb, "keys": zc["H_B_pool_keys"]},
       "H_sim": {"label": zc["simh"]["label"] + "; batches: " + json.dumps(zc["batch_labels"]), "roots": zc["simh_roots"], "exclude_keys": []},
       "sources": {"five_arm_v2": list(V2), "final_spec": list(SPEC), "znew_config": ZCFG, "znew_config_sha256": sha((HERE / ZCFG).read_bytes())},
       **({"fold_minus1_rule": "s1"} if S1 else {}),
       "rows": [{k: r_[k] for k in ("label", "sub", "file", "folds")} for r_ in zc["rows"]]}
(HERE / "config" / f"inputs_rms_{SET}{'_s1' if S1 else ''}.json").write_text(json.dumps(out, indent=1) + "\n")
print(SET, "cells", len(cells), "folders", len(folders), "H_B keys", len(zc["H_B_pool_keys"]), "roots", len(zc["simh_roots"]))
