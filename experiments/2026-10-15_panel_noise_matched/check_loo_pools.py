"""09's PRE-READ GATE for the LOO3 record (2e relay 10-16 ~01:00), before any LOO value is read: BY KEY for pool_old vs the pinned FINAL keys and own_all_L48's drops; LOO's own membership INFERRED from the record's statements and counts (no per-row key lists). Reads keys and counts only (git + JSON; no score,
catch or gap is read). LOO's S_old pools must equal pool_old (09's key export @ f09c4c0b4, 386 H_B keys) and its 192 honest_c (H_A = honest_c runs of H_B);
the 9 + 3 keys own_all_L48 lacks are listed with the record's reason. Any LOO key outside pool_old -> STOP. Writes results/NOTE_loo_pools_<stamp>.json.

  python3 experiments/2026-10-15_panel_noise_matched/check_loo_pools.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
git = lambda spec: subprocess.run(["git", "show", spec], capture_output=True, check=True, cwd=HERE).stdout


def pin(spec, sha):
    b = git(spec)
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"STOP: {spec} sha256 != {sha}")
    return b


LOO = ("0a9c2bdd3:experiments/2026-10-15_fpr_sweep_507_prep/results/alltasks_CRD_ownallLOO_2026-10-16_002812.json", "b8520a2ab981a5bdd5261d2daa19a3418029f197f62cf06f06d81bb34843b7f1")
EXPORT = ("f09c4c0b4:experiments/2026-10-14_priority1_spec_09/keys/final_HB_pool_keys_from_3ce6ea55e.json", None)
INPUTS = ("32abe4120:experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FINAL_2026-10-15_203213.json", "d8be69e672acaca11be3f76807929660c61db30a3cf50b083f8fef8c4d6dee9a")

rec = json.loads(pin(*LOO))
exp_b = git(EXPORT[0])
pool_old = json.loads(exp_b)["pool_keys_old_S_old_H_B"]
inp = json.loads(pin(*INPUTS))
hb = inp["pools"]["stage_a"]["S_old"]["H_B"]
fin = json.loads(pin(f"{hb['commit']}:{hb['path']}", hb["sha256"]))
ent = fin["result"]["calibration_records"]["primary"]["sol_a|R|0.01|primary|BuC|B"]["schemes"]
honest_c = sorted(k for k in pool_old if k.startswith("honest_c|"))
out = {"label": ("PRE-READ GATE (09 via 2e): LOO3 S_old pools; keys and counts only, no LOO value read. BY KEY: pool_old (09's export) = the FINAL record's pool_keys_old that "
                 "INPUTS_FINAL pins; own_all_L48's missing keys = its listed fold -1 drops. INFERRED (not by key): LOO's own pool membership, from the Stage A record's statements "
                 "(admitted = 'the pool's key list', no fold -1 drops, its by-key pools_check against INPUTS_FINAL) and counts; the record lists no per-row pool keys"), "inputs": {"loo_record": LOO, "pool_old_export": EXPORT[0],
       "pool_old_export_sha256": hashlib.sha256(exp_b).hexdigest(), "inputs_final": INPUTS, "final_record": [f"{hb['commit']}:{hb['path']}", hb["sha256"]]},
       "pool_old": {"n": len(pool_old), "distinct": len(set(pool_old)), "honest_c": len(honest_c), "honest_b": sum(k.startswith("honest_b|") for k in pool_old),
                    "other_prefix": sorted({k.split("|")[0] for k in pool_old} - {"honest_b", "honest_c"})},
       "inputs_final_pin": {"S_old_H_B_n": hb["n"], "S_old_H_A": inp["pools"]["stage_a"]["S_old"]["H_A"],
                            "final_pool_keys_old_equal_export_by_key": {s: sorted(v["pool_keys_old"]) == sorted(pool_old) for s, v in ent.items()}},
       "loo_record_inputs_final_sha256_matches": (rec["meta"].get("inputs_final") or {}).get("sha256") == INPUTS[1],
       "stage_a_pools_check": rec["meta"].get("pools_check_own_all_loo"), "rows": {}}
stop = []
for lab in ("own_all_LOO_L48 C", "own_all_LOO_L48 R", "own_all_L48 C", "own_all_L48 R"):
    o = rec["rows"][lab]["S_old"]
    dr = o.get("fold_minus1_dropped") or {}
    dB, dA = sorted((dr.get("H_B_runs") or {}).get("keys") or []), sorted((dr.get("H_A_runs") or {}).get("keys") or [])
    pr = o["pool_runs"]
    row = {"H_B_admitted": pr["H_B"]["admitted_runs"], "H_B_admitted_is": pr["H_B"].get("admitted"), "H_A_admitted": pr["H_A"]["admitted_runs"], "H_A_admitted_is": pr["H_A"].get("admitted"),
           "coverage_H_B_runs": o["coverage"]["H_B_runs"], "coverage_H_A_runs": o["coverage"]["H_A_runs"], "fold_minus1_rule": o.get("fold_minus1_rule"),
           "dropped_H_B_keys": dB, "dropped_H_A_keys": dA, "dropped_cohort_cells": sorted((dr.get("cohort_cells") or {}).get("keys") or []),
           "dropped_outside_pool_old": sorted(set(dB) - set(pool_old)), "dropped_H_A_outside_honest_c": sorted(set(dA) - set(honest_c))}
    row["H_B_used_by_key"] = len(set(pool_old) - set(dB))
    row["H_A_used_by_key"] = len(set(honest_c) - set(dA))
    row["counts_consistent"] = (row["H_B_used_by_key"] == row["coverage_H_B_runs"] and row["H_A_used_by_key"] == row["coverage_H_A_runs"]
                                and pr["H_B"]["admitted_runs"] == len(pool_old) and pr["H_A"]["admitted_runs"] == len(honest_c))
    out["rows"][lab] = row
    if row["dropped_outside_pool_old"] or row["dropped_H_A_outside_honest_c"] or not row["counts_consistent"]:
        stop.append(f"{lab}: keys outside pool_old / honest_c or inconsistent counts")
    if "LOO" in lab and (dB or dA):
        stop.append(f"{lab}: LOO drops pool keys {dB + dA} (expected none: no fold -1 units)")
loo_keys, oa_keys = set(pool_old) - set(out["rows"]["own_all_LOO_L48 C"]["dropped_H_B_keys"]), set(pool_old) - set(out["rows"]["own_all_L48 C"]["dropped_H_B_keys"])
out["loo_minus_own_all_L48"] = {"H_B": sorted(loo_keys - oa_keys), "H_A": sorted(set(out["rows"]["own_all_L48 C"]["dropped_H_A_keys"]) - set(out["rows"]["own_all_LOO_L48 C"]["dropped_H_A_keys"]))}
out["loo_membership_inferred"] = {"keys_outside_pool_old": sorted(loo_keys - set(pool_old)),
                                  "note": "empty by construction: LOO's pool is INFERRED as pool_old minus its (empty) fold -1 drops; substituted keys are not detectable from this record"}
pcheck = rec["meta"].get("pools_check_own_all_loo") or {}
out["stage_a_pools_check_valid"] = pcheck.get("rows_checked") == 6 and "INPUTS_FINAL pools.stage_a" in str(pcheck.get("against"))  # 2 readouts x 3 origins, by key (Stage A)
out["extras_are_exactly_own_all_L48_fold_minus1_drops"] = (out["loo_minus_own_all_L48"]["H_B"] == out["rows"]["own_all_L48 C"]["dropped_H_B_keys"]
                                                           and out["loo_minus_own_all_L48"]["H_A"] == out["rows"]["own_all_L48 C"]["dropped_H_A_keys"]
                                                           and out["rows"]["own_all_L48 C"]["dropped_H_B_keys"] == out["rows"]["own_all_L48 R"]["dropped_H_B_keys"])
out["basis"] = ("BY KEY: pool_old = 09's key export, equal BY KEY to the FINAL record's pool_keys_old that INPUTS_FINAL pins as Stage A's S_old H_B pool; the Stage A record lists no per-row "
                "pool keys, so LOO's pool = that pinned list minus its fold -1 drops (none; the record's admitted list is 'the pool's key list'), with Stage A's own by-key "
                "pools_check of the 6 LOO rows against INPUTS_FINAL pools.stage_a and count equality (386 / 192) as the record's evidence")
ok = (not stop and all(out["inputs_final_pin"]["final_pool_keys_old_equal_export_by_key"].values()) and out["pool_old"]["n"] == 386 == out["pool_old"]["distinct"]
      and out["pool_old"]["honest_c"] == 192 and not out["pool_old"]["other_prefix"] and out["loo_record_inputs_final_sha256_matches"] and out["stage_a_pools_check_valid"]
      and out["extras_are_exactly_own_all_L48_fold_minus1_drops"])
out["stop"], out["pass"] = stop, ok
p = HERE / "results" / f"NOTE_loo_pools_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
p.open("x").write(json.dumps(out, indent=1))
print(json.dumps({"file": p.name, "pass": ok, "stop": stop, "pool_old": out["pool_old"], "final_equal": out["inputs_final_pin"]["final_pool_keys_old_equal_export_by_key"],
                  "loo_minus_own_all_L48": out["loo_minus_own_all_L48"], "extras_exact": out["extras_are_exactly_own_all_L48_fold_minus1_drops"]}))
if not ok:
    raise SystemExit("STOP: pre-read gate failed")
