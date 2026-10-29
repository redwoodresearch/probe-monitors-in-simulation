"""Build an IL spec: FINAL spec 3 unchanged + spec.il from the chain's resolution index (keys only; reads no score). Every input
is a {commit, path, sha256} entry, sha256-checked here with git show and again by the job. The counted manifests, label files,
run tables and supersession records are exactly the ones the index names (its `sources` sha256s must match).

  python3 -I experiments/2026-10-15_il_analysis_09/make_il_spec.py --index <path>@<commit> --out <spec json> \\
      [--parse-only] [--declaration <path>@<commit>] [--arms <arms json path>@<commit>]
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FINAL_SPEC = REPO / "experiments/2026-10-14_xerr_scaleup_09/specs/final_spec_3_launch.json"
CHAIN = "experiments/2026-10-14_q3_overnight_chain"
KEYLIST = {
    "commit": "2e75fa8c5",
    "path": "experiments/2026-10-14_il_launch_09/results/il_keylists_2026-10-14.json",
    "sha256": "9e73c6977e89f77751e7dc86a4aeaaa6d435be4875b41a6e411ca217d61f5879",
}
STAGE_MAPS = {
    "commit": "0a52fa131",
    "path": f"{CHAIN}/config/il_stage_maps.json",
    "sha256": "4479636e6d74058673b5acef1fe2cd8fc006f16d64bfe9c59c18b216b36aca50",
}
REGISTERED = {  # Append 32 registrations (09): tier-3 completion run3 originals and the rate-limit replacement-cell set
    "expected_superseded_run3_by_family": {  # q4c: 21 (supersession @ 912a9c102, confirmed by key by 09)
        "il311_q3c": 16,
        "ilold_oldc": 52,
        "il311_q1c": 17,
        "il311_q2c": 17,
        "il311_q4c": 21,
    },
    "expected_rate_limit_cells": 70,
    "registered_missing_cells": ["t278_model_registry"],
}
PARENT_SAME_LAUNCH = {  # selftest: FINAL's own real-vs-real same-launch cell counts, cited beside the IL extended output
    "cells_same_launch": {"S_new": 272, "S_newmains": 244},
    "source": "experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_113025.json @ 1a7e95259 "
    "(sha256 43ea698a22418168...): results[K].extended.<stratum>_same_launch_run2_minus_run1.cells_same_launch, all four K",
}
HITS_PENDING = {  # "<set>|<export commit>": probes whose rows in that export are NOT read (09's rulings)
    # 2e's S_new-on-M export @ a69d6ba67 dropped fold -1 for Sol (a)/c_asis (H_B 300, H_sim 225, M 254), against 25h section 1
    "S_new|a69d6ba67": {
        p_: {
            "label": "pending 25h §1 fold −1 rerun (current export dropped fold −1; values withheld)",
            "source": "09 to the IL fork, 10-15 ~15:10",
        }
        for p_ in ("Sol (a)", "c_asis")
    },
}
HITS_LABELS = {  # "<set>|<export commit>": labels carried into the per-action block (09's wording)
    "S_new|a69d6ba67": {"H_sim": "H_sim = interim (230 runs)"},
    "S_new|ac8818f71": {
        "H_sim": "H_sim = FINAL roots (292 runs; the fifth pass's 24 FINAL roots)",
        "supersedes": "run 5's S_new-on-M rows (export @ a69d6ba67: H_sim interim, 230 runs; Sol (a)/c_asis fold -1 dropped)",
        "fold_handling": "Sol (a)/c_asis under 25h §1 fold -1 handling (all 257 M cells for every probe)",
    },
}
STATUS_NOTES = {  # cites for uncited non-scored index rows (keys only; 09's request 10-15 ~13:30)
    "sim2|nm105_media_processing": {
        "status": "never_started",
        "cite": "no pinned simulator-grounding reference: il_no_reference_cells_2026-10-15.json @ b7dbecb18 (sha256 "
        "a2149da9c8a8dbd47d3d79677560300fcbaa7744e65fca13bb2034817e7df2be): sim_attack@september/nm105_media_processing "
        "'not run', 0 attempts, 'no pinned reference in .../2026-10-14_newmains_s1'; listed in RULE.md's grounding-"
        "reference pin (09, 2026-10-15 02:07 PDT)",
    },
    "sim2|nm214_slack_clone": {
        "status": "never_started",
        "cite": "never started: cause unrecorded at the r2 record (no committed record named a reason); recorded since "
        "@ cfa873de6: experiments/2026-10-15_q3_interleaved_run3_sim2/results/il_s5_nm214_sim2_cause_2026-10-15.json "
        "(sha256 42bd74e84aaf1d3e9d9e318c4a514a9a5bb72eeb8df2734445ecb03e4c583ddb): sim_attack@september/nm214_slack_clone "
        "'not run', 0 attempts, why 'no pinned reference in .../2026-10-14_newmains_s5' (launch.json sha256 6403974e...)",
    },
}
DISCLOSURES = [  # p2's disclosed 'near 0' exposure and Aswin's acceptance (relayed by 09 / d9)
    {"what": "exposure row", "commit": "8f769dac9"},
    {"what": "name correction row", "commit": "f1281ba0c"},
    {"what": "Aswin's acceptance row", "commit": "bfd18daf5"},
    {"what": "Aswin's acceptance, spec", "commit": "23d3fcfd4"},
]


def show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=REPO).stdout


def pin(commit, path, want=None):
    sha = hashlib.sha256(show(commit, path)).hexdigest()
    if want is not None and sha != want:
        raise SystemExit(f"{path} @ {commit}: sha256 differs from the index's source record")
    return {"commit": commit, "path": path, "sha256": sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--declaration")
    ap.add_argument("--arms")
    ap.add_argument("--il-mode", choices=("final", "interim"), default="final")
    ap.add_argument("--snapshot-index", type=int)
    ap.add_argument("--scope", choices=("primary", "extended"), default="primary")
    ap.add_argument("--parent-spec", help="<path>@<commit> of the FINAL parent spec (default: FINAL spec 3, primary)")
    ap.add_argument("--arms-extended", help="<path>@<commit> of the nm360 arms fragment (key arms_extended)")
    ap.add_argument("--cutoff-time")
    ap.add_argument("--posthoc-cells", help="<path>@<commit> of il_five_arm_cells (POST HOC RMS bias and S_newmains yardstick)")
    ap.add_argument("--posthoc-hits", action="append", default=[], help="<path>@<commit> of a 2e rmshits manifest (repeatable)")
    a = ap.parse_args()
    ip, ic = a.index.rsplit("@", 1)
    idx = json.loads(show(ic, ip))
    if idx.get("pass") is not True or idx.get("errors"):
        raise SystemExit("the index did not pass its own checks")
    chain, src = idx["chain_commit"], idx["sources"]
    rows = idx["rows"]
    scored = [r for r in rows if r["status"] == "scored"]
    counted = {"run3": [], "sim2": []}
    for w in counted:
        for m in sorted({r["counted_manifest"].rsplit("@", 1)[0] for r in scored if r["world"] == w}):
            counted[w].append(pin(chain, m, src.get(m)) | {"keys_path": ["files"]})
    labels = [
        pin(chain, f"{CHAIN}/results/hygiene/{f}", src.get(f"{CHAIN}/results/hygiene/{f}"))
        for f in sorted({r["label_file"] for r in scored})
    ]
    run_tables = [
        pin(chain, f"{CHAIN}/results/hygiene/{f}", src.get(f"{CHAIN}/results/hygiene/{f}"))
        for f in sorted(set(idx["run_tables"].values()))
    ]
    records = [pin(chain, p_, h_) for p_, h_ in sorted(src.items()) if "/results/supersession/supersession_" in p_]
    il = {
        "mode": "real",
        "parse_only": bool(a.parse_only),
        "il_mode": a.il_mode,
        **({"snapshot_index": a.snapshot_index, "cutoff_time": a.cutoff_time} if a.il_mode == "interim" else {}),
        "keylist": KEYLIST,
        "stage_maps": STAGE_MAPS,
        "index": pin(ic, ip),
        "index_rows_path": ["rows"],
        "index_field_map": {"scorer_folder": "folder", "attempt_label": "label"},
        "counted": counted,
        "labels": labels,
        "run_tables": run_tables,
        "supersession_records": records,
        "disclosures": DISCLOSURES,
        "status_notes": STATUS_NOTES,
        **REGISTERED,
    }
    if a.declaration:
        dp, dc = a.declaration.rsplit("@", 1)
        il["declaration"] = pin(dc, dp)
    if a.arms:
        ap_, ac_ = a.arms.rsplit("@", 1)
        frag = json.loads(show(ac_, ap_))
        arms = frag.get("arms", frag) if isinstance(frag, dict) else None  # a fragment with an `arms` key, or the arms dict
        if not isinstance(arms, dict) or not all(isinstance(v, dict) and "p2_manifest" in v for v in arms.values()):
            raise SystemExit("the arms source is not {scorer folder: {hygiene, p2_manifest, sol_manifest}}")
        il["arms"] = arms
        il["arms_source"] = pin(ac_, ap_)
        for k_ in ("p2_cells_list_sha256_allowed",):
            if k_ in frag:
                il[k_] = frag[k_]
    if a.parent_spec:
        pp_, pc_ = a.parent_spec.rsplit("@", 1)
        spec = json.loads(show(pc_, pp_))
        parent_pin = pin(pc_, pp_)
    else:
        spec = json.loads(FINAL_SPEC.read_text())
        parent_pin = {"path": str(FINAL_SPEC.relative_to(REPO)), "note": "working-tree FINAL spec 3"}
    if a.scope == "extended":
        if not a.arms_extended or not a.parent_spec:
            raise SystemExit("--scope extended needs --parent-spec (the FINAL extended spec) and --arms-extended")
        xp_, xc_ = a.arms_extended.rsplit("@", 1)
        fx_ = json.loads(show(xc_, xp_))
        il["scope"] = "extended"
        il["arms_extended"] = fx_["arms_extended"]
        il["arms_extended_source"] = pin(xc_, xp_)
        il["parent_same_launch_expected"] = PARENT_SAME_LAUNCH
    il["parent_spec"] = parent_pin
    if a.posthoc_cells:
        hp_, hc_ = a.posthoc_cells.rsplit("@", 1)
        fc_ = json.loads(show(hc_, hp_))
        il["posthoc_cells"] = {
            "S_old": fc_["S_old_five_arm"]["cells"],
            "S_new_on_M": fc_["S_new_on_M"]["cells"],
            "S_newmains": fc_["S_newmains_five_arm"]["cells"],
            "expected_counts": {"S_old": 88, "S_new_on_M": 257, "S_newmains": 239},  # 09's note @ c10a14948
            "source": pin(hc_, hp_),
        }
    hits_ = []
    for h_ in a.posthoc_hits:  # 2e's per-action hit exports (POST HOC RMS bias, 09 10-15 ~14:40); gate fields only
        mp_, mc_ = h_.rsplit("@", 1)
        m_ = json.loads(show(mc_, mp_))
        hits_.append(
            {
                "set": m_["set"],
                "manifest": pin(mc_, mp_),
                "cells": m_["cells"],
                "arms": m_["arms"],
                "probes": m_["probes"],
                "points": m_["points"],
                "lines": m_["lines"],
                "pools": {  # every field except the thresholds T; a withheld probe keeps its status string
                    pl: {
                        sd: (
                            {k_: ({f_: x_ for f_, x_ in v_.items() if f_ != "T"} if isinstance(v_, dict) else v_) for k_, v_ in sv_.items()}
                            if isinstance(sv_, dict)
                            else sv_
                        )
                        for sd, sv_ in pv_.items()
                    }
                    for pl, pv_ in m_["pools"].items()
                },
                "npz": {"path": f"/sol/xfer/il_z/rms_hits/{m_['npz']}", "sha256": m_["npz_sha256"]},
            }
            | ({"pending_probes": HITS_PENDING[f"{m_['set']}|{mc_}"]} if f"{m_['set']}|{mc_}" in HITS_PENDING else {})
            | ({"labels": HITS_LABELS[f"{m_['set']}|{mc_}"] | {"set_label_verbatim": m_.get("set_label")}} if f"{m_['set']}|{mc_}" in HITS_LABELS else {})
        )
    if hits_:
        il["posthoc_hits"] = hits_
    spec["note"] = f"IL {'PARSE-ONLY' if a.parse_only else 'REAL'} {a.il_mode.upper()} {a.scope.upper()} (09 IL fork): parent FINAL spec unchanged + spec.il. Base: " + spec["note"]
    spec["il"] = il
    Path(a.out).open("x").write(json.dumps(spec, indent=1))
    print(
        json.dumps(
            {
                "out": a.out,
                "counted_manifests": {w: len(v) for w, v in counted.items()},
                "labels": len(labels),
                "run_tables": len(run_tables),
                "supersession_records": len(records),
                "scored_rows": len(scored),
                "arms": len(il.get("arms") or {}),
            }
        )
    )


if __name__ == "__main__":
    main()
