"""09 @ 2d61af835 sec. 4-5: the FIXED 32k vs full-length comparison and the descriptive disagreement flag (JSON reads only). Both records' comparator is (A).
Per probe Delta_full (A) at C and R, the six-probe S and the excess, side by side, each labelled with its build; NO difference test. flag = true iff the two
builds' six-probe C S points (comparator (A)) have opposite signs, or exactly one of the two intervals excludes 0. Writes results/compare_builds_<stamp>.json.

  PYTHONPATH=src python3 experiments/2026-10-15_panel_noise_matched/compare_builds.py <32k record> <full-length record>
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.fmtpts import raw_and_printed

HERE = Path(__file__).resolve().parent
P = [Path(x) for x in sys.argv[1:3]]
pt = "0.01"  # the registered level for the flag (09 sec. 5); not an argument
recs = [json.loads(p.read_bytes()) for p in P]
names = ["32k", "full_length"]
shas = [hashlib.sha256(p.read_bytes()).hexdigest() for p in P]
# provenance (Codex final r10): two different records, each stating its build in its config; same cohort rule; C leads in both
cfgs = [r["meta"]["config"] for r in recs]
if shas[0] == shas[1] or [c_.get("build") for c_ in cfgs] != names:
    raise SystemExit(f"STOP: the two records must be different and state build = {names} in their configs (got {[c_.get('build') for c_ in cfgs]})")
if [c_.get("headline_combo") for c_ in cfgs] != ["pooled_all_origins", "pooled_all_origins"]:  # 09 (via 2e): the disagreement flag uses pooled_all_origins in BOTH builds
    raise SystemExit(f"STOP: both builds must lead with pooled_all_origins (got {[c_.get('headline_combo') for c_ in cfgs]})")
crs = [[x.get("cohort_rule") for x in (c_.get("final_checks_resolved") or [])] for c_ in cfgs]
if not all(crs) or any(not isinstance(x, str) or not x for cr_ in crs for x in cr_) or crs[0] != crs[1] or [c_.get("lead_readout") for c_ in cfgs] != ["C", "C"]:
    raise SystemExit(f"STOP: the builds' cohort rules {crs} must be stated strings and equal, and both must lead with C")
# the registered probe sets per build (09 @ 002c1824b / 2d61af835) and labelled calibrations; the same draw count
PROBES = {"32k": ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes", "own_all_rs_L48"],
          "full_length": ["sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48"]}
CALIB = {"32k": "registered epoch-1 calibration (32k)", "full_length": "the full-length lead's calibration (uncut honest runs)"}
for n, r in zip(names, recs):
    if list(r["coheadline"]["probes"]) != PROBES[n] or r["meta"]["config"].get("calibration_label") != CALIB[n] or not r["meta"]["config"].get("inputs_final"):
        raise SystemExit(f"STOP: the {n} record's probes / calibration_label / inputs_final pin do not match the registered build ({PROBES[n]}; {CALIB[n]!r})")
ckm = [r.get("cohort_keys") or {} for r in recs]  # the shared cohort membership BY KEY: every origin's sorted-key sha256 (incl. S_paper493), equal across builds (Codex final r13)
ksh = [k_.get("key_sha256") or {} for k_ in ckm]
base_req = {"S_old", "S_new", "S_newmains", "S_paper493"}
if not ksh[0] or ksh[0] != ksh[1] or not base_req <= set(ksh[0]):
    raise SystemExit(f"STOP: the builds' cohort key sha256s differ or miss a base origin ({sorted(set(ksh[0]) ^ set(ksh[1]))[:5]}): not the same cohort")
# the registered calibration evidence per build (Stage A's meta.window), fixed here; unset values STOP (fail closed; 507-prep supplies them; Codex final r14)
EXPECTED_WINDOW = {"32k": "cut32768", "full_length": "full"}  # 507-prep's meta.window values (relayed 10-16 ~03:05)
cev = [r["meta"]["config"].get("calibration_evidence_resolved") for r in recs]  # resolved by run_pooled from Stage A against the config's expectation
if any(EXPECTED_WINDOW[n] is None for n in names) or [cev[i] for i in range(2)] != [EXPECTED_WINDOW[n] for n in names]:
    raise SystemExit(f"STOP: calibration evidence {cev} != the registered per-build windows {EXPECTED_WINDOW} (set them from 507-prep's records)")
for n, r in zip(names, recs):
    for sel in ("C", "R"):
        if r["coheadline"][pt][sel]["six"].get("S_estimator", "A") not in ("A", "B"):
            raise SystemExit(f"STOP: the {n} record's {sel} S_estimator label is not A or B")
dr0 = [r["coheadline"][pt]["C"]["six"].get("S", {}).get("draws") for r in recs]
if None in dr0 or dr0[0] != dr0[1]:
    raise SystemExit(f"STOP: the builds' draw counts {dr0} must be stated and equal (the same draw streams)")


def s_A(r, sel):  # the six-probe S on comparator (A): S itself, or S_A_beside when the record's lead used (B)
    six = r["coheadline"][pt][sel]["six"]
    return six.get("S_A_beside") if six.get("S_estimator") == "B" else six.get("S")


def s_sel(r, sel):  # the selected estimator's S (09 sec. 4b: printed beside, labelled, when (B) leads)
    six = r["coheadline"][pt][sel]["six"]
    return {"estimator": six.get("S_estimator", "A"), "S": raw_and_printed(six.get("S"))}


def complete(z):
    import math

    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    return (bool(z) and num(z.get("point")) and isinstance(z.get("ci95"), list) and len(z["ci95"]) == 2 and all(num(x) for x in z["ci95"]) and z["ci95"][0] <= z["ci95"][1])


out = {"label": "FIXED comparison (09 @ 2d61af835 sec. 4): 32k (registered epoch-1 calibration) vs full length (the lead's calibration and probes); comparator (A) in both; no difference test",
       "records": [{"build": n, "path": str(p), "sha256": h, "git": r["meta"]["git"], "headline_group": r["headline_group"], "cohort_rule": cr[0] if cr else None,
                    "calibration_label": r["meta"]["config"].get("calibration_label"), "probes": r["coheadline"]["probes"]}
                   for n, p, r, h, cr in zip(names, P, recs, shas, crs)],
       "level": pt, "six_S_A": {}, "excess_A": {}, "per_probe_delta_full_A": {}}
out["six_S_selected_beside"] = {}
for sel in ("C", "R"):
    out["six_S_A"][sel] = {n: raw_and_printed(s_A(r, sel)) for n, r in zip(names, recs)}
    out["six_S_selected_beside"][sel] = {n: s_sel(r, sel) for n, r in zip(names, recs)}
    out["excess_A"][sel] = {n: raw_and_printed((r["results"]["pooled"][pt]["groups"].get(f"{r['headline_group']}|{sel}") or {}).get("primary", {}).get("mean_G_minus_mean_Y"))
                            for n, r in zip(names, recs)}
    for n, r in zip(names, recs):
        for p_ in r["coheadline"]["probes"]:
            row = r["results"]["pooled"][pt]["rows"].get(f"{p_} {sel}") or {}
            out["per_probe_delta_full_A"].setdefault(sel, {}).setdefault(n, {})[p_] = raw_and_printed(row.get("delta_full_A"))
miss_ = [f"{k}/{sel}/{n}" for k in ("six_S_A", "excess_A") for sel in ("C", "R") for n in names if not complete(out[k][sel][n])]
miss_ += [f"delta_full_A/{sel}/{n}/{p_}" for sel, by in out["per_probe_delta_full_A"].items() for n, d_ in by.items() for p_, z in d_.items() if not complete(z)]
miss_ += [f"selected_S/{sel}/{n}" for sel in ("C", "R") for n in names if out["six_S_selected_beside"][sel][n]["estimator"] == "B" and not complete(out["six_S_selected_beside"][sel][n]["S"])]
miss_ += [f"selected_S_estimator/{n}" for n in names if out["six_S_selected_beside"]["C"][n]["estimator"] not in ("A", "B")]
if miss_:  # fail closed: every comparison entry needs a finite point and a two-ended interval (Codex final r10)
    raise SystemExit(f"STOP: incomplete comparison statistics: {miss_[:10]}")
a, b = out["six_S_A"]["C"]["32k"], out["six_S_A"]["C"]["full_length"]
excl = lambda z: bool(z and z.get("ci95") and not (z["ci95"][0] <= 0 <= z["ci95"][1]))
opp = bool(a and b and a["point"] * b["point"] < 0)
out["disagreement_flag"] = {"flag": opp or (excl(a) != excl(b)), "opposite_signs": opp, "excludes_0": {"32k": excl(a), "full_length": excl(b)},
                            "label": "descriptive: one interval excluding 0 while the other does not is not a test of a difference between the builds (09 @ 2d61af835 sec. 5)"}
q = HERE / "results" / f"compare_builds_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
q.open("x").write(json.dumps(out, indent=1))
print(json.dumps({"file": q.name, "disagreement_flag": out["disagreement_flag"]}))
