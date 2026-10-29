"""Lean fixtures for the route C numbers verifier, on SYNTHETIC records only (no real value is read).

  (1) Slot readouts, through a mini slot table with slot_table.py's anchors (instrumented the same way): the positive control PASSes with
      a value whose correct rounding is the binary float's (0.12345 -> "12.3%", not a naive half-up "12.4%"); R4: one row per printed
      number (a three-number slot is three rows) plus a slot summary row; a planted mismatch STOPs on that number only; an unsigned record
      STOPs (orphan); a printed count below the consumer's STOPs. The real slot_table.py blob (a8b4a156) takes the instrumentation.
  (2) Tables 3/4 on a synthetic A, D1 npz z and pools: the positive control PASSes with T recomputed bitwise by the real wthr (rp blob
      71633875); 09's condition (2) (A's T one ulp off) and condition (1) (an npz sha that is not A's pin) make orphans; R1: the
      not-printed list is generated (record numeric paths minus mapped paths), STOPs until approved, and a new numeric field changes it
      (STOP); R2: a (probe, pool) with dropped duplicates is an orphan; R3: a drawn cell outside A's cohort, A's bits disagreeing with
      the outcome, or a run_key whose file keys are another cell's, each STOP; five draws STOP on the count; a z one ulp off STOPs.
  (2b) Table 4's attack-side z (09's ruling (b)): synthetic score file and D1 moments through the real unit_z (run_tables_v81 blob
      88589b50) and argmax_command (run_examples_v81 blob f191a247): PASS; a wrong argmax STOPs; FROZEN unsigned makes an orphan; the
      FROZEN lookup (score_pins) STOPs on a missing override and on an override whose bytes are not FROZEN's pin.
  (2c) Figures: committed = re-render = record by key PASSes; a value off the record, a caption number off the re-render, one more
      re-rendered number, and an unsigned input each STOP.
  (2d) Delta RQ1-RQ3 and O2: a PNG must be byte-identical to the re-render (a pixel or a metadata difference STOPs, reported by chunk
      name and count only); token, unit/row_unit and unit/token (the latter only without a call_index) and the dedup count are verified
      rows, so the not-printed list holds only the remaining locators; a tables34 PASS says record-verified only; score-file keys of
      the other arm STOP.
  (2e) RN1: an accepted deferred row yields PARTIAL, never PASS; an unaccepted reason or a non-result row without a 09-approved
      list match is an orphan. O-a: the false-alarm token also needs the five-model scores to match the pool's.
  (3) R5: guarded() returns the error class and phase only, never the message. The recheck selection is fixed by the seed.

  uv run --no-project --python 3.12 --with numpy==2.3.3 python -I test_numbers_verify.py   (from this folder)
"""

import copy
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_core as VC  # noqa: E402

RP_BLOB = "71633875188c57ca1ceb8271a3f3ca0ed0989452"
ST_BLOB = "a8b4a1565321bc4c5aeb8d61ee616eeca162118b"
FAILS = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        FAILS.append(name)


def blob(b):
    return subprocess.run(["git", "cat-file", "blob", b], check=True, capture_output=True).stdout


# ---------- (1) slot readouts ----------
MINI = b'''import math
MINUS = "\\u2212"


class Missing(Exception):
    pass


def get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            raise Missing("/".join(map(str, path)))
        d = d[p]
    return d


def pct(x):
    return f"{100 * x:.1f}".replace("-", MINUS) + "%"


def build(recs, shas, paths, control=False):
    out, full, missing, nc = {}, {}, {}, {}
    out["mode"] = full["mode"] = "writing"

    def put(key, fn):
        try:
            out[key], full[key] = fn()
        except Missing as e:
            missing[key] = str(e)

    B = recs["B"]
    put("20.catch.joint.x C@0.0001.sim", lambda: (lambda n: (f"{pct(n['point'])} ({pct(n['ci95'][0])} to {pct(n['ci95'][1])})", "f"))(
        get(B, "pooled", "x C", "catch", "sim")))
    _ = get(B, "pooled")  # a read outside any put() (as slot_table's levels(get(...))): must not be logged under the slot before it
    put("19.units_above_tau_sim.x C@0.001", lambda: (f"{get(B, 'n'):,}", str(get(B, "n"))))
    return {"slots": out, "full": full, "missing": missing, "not_carried": nc}
'''
B = {"pooled": {"x C": {"catch": {"sim": {"point": 0.12345, "ci95": [0.10049, 0.15]}}}}, "n": 1906}
SIGNED = {"shaB"}
CATCH = "20.catch.joint.x C@0.0001.sim"
PRINTED = {"mode": "writing", CATCH: "12.3% (10.0% to 15.0%)", "19.units_above_tau_sim.x C@0.001": "1,906"}


def run_slots(printed, signed=SIGNED, rec=B, mini=MINI):
    ST = VC.instrumented_slot_table(mini)
    recs = {"B": copy.deepcopy(rec)}
    ST._NV_ABS.update({id(recs["B"]): ("B",)})
    reb = ST.build(recs, {"B": "shaB"}, {"B": "b.json"})
    rows, plans, stops = VC.slots_rows("rcX", printed, reb, ST._NV_LOG, recs, {"B": "c:b.json"}, {"B": "shaB"}, signed)
    return VC.summarize(rows, plans, stops), rows


s, rows = run_slots(PRINTED)
check("slots: positive control PASS (binary-float rounding 12.345 -> 12.3)", s["verdict"] == "PASS" and s["n_equal"] == 4)
check("R4: one row per printed number (3 + 1) plus 3 slot summary rows", s["n_printed"] == 4 and s["n_slot_rows"] == 3
      and [r["slot"] for r in rows if r["unit"] == "number" and r["slot"].startswith("20.")] == [f"{CATCH}#{i}" for i in range(3)])
check("R4: each number row has its own key path", [r["key_path"] for r in rows if r["slot"].startswith(f"{CATCH}#")]
      == [f"B/pooled/x C/catch/sim/{c}" for c in ("point", "ci95[0]", "ci95[1]")])
s, _ = run_slots(PRINTED | {CATCH: "12.4% (10.0% to 15.0%)"})
check("slots: planted mismatch (naive half-up 12.4) STOPs on that number and its slot only",
      set(s["stop"]["unmatched"]) == {f"rcX::{CATCH}#0", f"rcX/slots::{CATCH}"})
Bbad = copy.deepcopy(B)
Bbad["pooled"]["x C"]["catch"]["sim"]["ci95"][1] = 0.1561  # the record's hi end prints 15.6, the released string says 15.0
s, _ = run_slots(PRINTED, rec=Bbad)
check("R4: a wrong interval end STOPs on that number row", f"rcX::{CATCH}#2" in s["stop"]["unmatched"] and f"rcX::{CATCH}#0" not in s["stop"]["unmatched"])
s, _ = run_slots(PRINTED, signed=set())
check("slots: unsigned source STOPs as orphan", s["verdict"] == "STOP" and len(s["stop"]["orphan"]) == 4 + 2 and not s["stop"]["unmatched"])
s, _ = run_slots({k: v for k, v in PRINTED.items() if not k.startswith("19.")})
check("slots: printed count below the consumer's count STOPs", s["stop"]["count_mismatch"] == ["rcX", "rcX/slots"])
try:
    VC.instrumented_slot_table(blob(ST_BLOB))
    ok_ = True
except SystemExit:
    ok_ = False
check("slots: the real slot_table.py blob takes the instrumentation", ok_)

# ---------- (2) Tables 3/4 ----------
wthr = VC.extract_function(blob(RP_BLOB), "wthr")
rng = np.random.default_rng(7)
NRUN, PER = 40, 5
runs = np.repeat([f"r{i}" for i in range(NRUN)], PER)
TOKS = np.arange(PER) * 10 + 3  # every synthetic run's score-file tokens
A = {"meta": {"pooled_world": {"calibrations": {"by_label_combo": {}}}}, "rows": {}}
d1, pools, d1sha, events, ev_units = {}, {}, {}, [], []
for lb in VC.FA_PROBES:
    zs = {nm: rng.normal(size=NRUN * PER) for nm in ("B", "S")}
    ss = {nm: rng.normal(size=NRUN * PER) for nm in ("B", "S")}  # the pools' raw scores (the score-file identity check)
    sb = {nm: rng.normal(size=(NRUN * PER, 5)) for nm in ("B", "S")}  # and their five-model scores (O-a)
    sha_ = f"d1-{lb}"
    A["meta"]["pooled_world"]["calibrations"]["by_label_combo"][f"{lb} | {VC.LEAD}"] = {"path": f"calib/{lb}.npz", "sha256": sha_}
    d1[lb] = {"sha256_read": sha_, "path": f"calib/{lb}.npz", "z": zs}
    pools[lb] = {"B": runs, "S": runs, "B_s": ss["B"], "S_s": ss["S"], "B_sbm": sb["B"], "S_sbm": sb["S"]}
    d1sha[lb] = sha_
    p0 = {}
    for nm, z in zs.items():
        T = float(wthr(np.sort(z)[::-1][None, :], np.ones((1, len(z))), 0.01)[0][0])
        ra = sorted(set(runs[z > T].tolist()))
        p0[nm] = {"T": T, "n_units": len(z), "units_above": int((z > T).sum()), "k": int(np.floor(0.01 * len(z) + 1e-9)),
                  "per_run_alarm_share": len(ra) / NRUN, "runs": NRUN}
        for r in ra:
            ix = np.flatnonzero(runs == r)
            k = int(np.flatnonzero(z[ix] > T)[0])
            zf = float(z[ix][k])
            events.append({"probe": lb, "pool": "pooled H_B" if nm == "B" else "pooled H_sim", "alias": r, "z": zf, "T": T, "z_minus_T": zf - T,
                           "unit": {"row_unit": k, "step": 7, "token": int(TOKS[k])}, "token": int(TOKS[k]), "length": 42})
            ev_units.append({"frozen_ok": True, "score_ref": "vol:conv:h.npz@s", "units": {"s": ss[nm][ix], "sbm": sb[nm][ix], "token": TOKS}, "why": ""})
    A["rows"][lb] = {VC.MEMBER_ROW.format(o=o): {"pools_at_draw0": {"0.01": copy.deepcopy(p0)}} for o in VC.ORIGINS}
for o in VC.ORIGINS:
    A["rows"][VC.OUTCOME_LABEL][VC.MEMBER_ROW.format(o=o)] |= {"cohort_cell_keys": ["c0", "c1", "c2", "c3"],
                                                               "cell_hits_draw0": {"levels": {"0.01": {"run1": "1100", "sim": "1010"}}}}
FA_U = {i + 1: ev_units[i] for i in range(6)}
CELL = dict(zip(VC.OUTCOMES, ("S_old|c0", "S_old|c1", "S_old|c2", "S_old|c3"), strict=True))  # each outcome's cell by A's bits
T_REC = {"calibration": "pooled", "lead": {"record": ["a.json.gz", "c", "blob", "shaA"]},
         "outcome_table": {"universe_cells": 16, "outcomes": {o: {"n": 4, "drawn": {"cell": CELL[o], "real (run 1)": "no command",
                                                                                    "sim (sim 1)": "no command"}} for o in VC.OUTCOMES}},
         "false_alarms": {"draws": events[:6], "duplicates_dropped": [], "universe_events": len(events), "duplicates_dropped_count": 0,
                          "events_per_probe_pool": {f"{lb}|{p}": sum(e["probe"] == lb and e["pool"] == p for e in events)
                                                    for lb in VC.FA_PROBES for p in VC.POOL_NM}}}
REFS = {"A": "c:a.json.gz", "A_sha": "shaA", "d1": {lb: f"vol:sol:calib/{lb}.npz@{d1sha[lb]}" for lb in VC.FA_PROBES}, "d1_sha": d1sha,
        "frozen_sha": "shaF"}
SIG_T = {"shaA", "shaF"} | set(d1sha.values())


def run_tables(T=T_REC, A_=A, d1_=d1, approve=True, attack=None, signed=SIG_T, approve_from=None, fa_units=FA_U):
    refs = dict(REFS)
    if approve:  # approve the list this record generates (as 09 would, by count + sha), unless the test withholds approval
        _, _, _, dg = VC.tables34_rows(approve_from or T, A, d1, pools, wthr, refs | {"not_printed_approved": None}, signed, attack, fa_units)
        refs["not_printed_approved"] = {"sha256": dg["not_printed"]["sha256"], "count": dg["not_printed"]["count"]}
    rows, plan, stops, dg = VC.tables34_rows(T, A_, d1_, pools, wthr, refs, signed, attack, fa_units)
    return VC.summarize(rows, {"tables34": plan}, stops), dg


s, dg = run_tables()
check("tables: positive control PASS (T bitwise by the real wthr; z, row_unit and token; event counts; dedup count; universe cells)",
      s["verdict"] == "PASS" and s["n_printed"] == 1 + 4 + 6 * 5 + 12 + 1 + 1 and not any(dg["conditions"].values()))
check("RQ2: the not-printed list now holds only the locators no row verifies (here: length, unit/step, unit/token)",
      dg["not_printed"]["count"] == 18 and all(p.endswith(("/length", "/unit/step", "/unit/token")) for p in dg["not_printed"]["paths"]))
check("RQ3: a tables34 PASS says record-verified only (printed_numbers_verified false)",
      dg["printed_numbers_verified"] is False and dg["scope"].startswith("record verified against sources"))
_k = {"tables34": "tables34", "rc2": "slots", "fig4": "figrender", "rc1a": "slots"}
_s, _p = VC.result_scope(_k, {"tables34": dg, "rc2": {"rebuilt_missing": 0}, "fig4": {"render_exit_codes": [0]},
                              "rc1a": {"error": {"class": "SystemExit", "where": "rc1a"}}}, [{"class": "SystemExit", "where": "rc1a"}])
check("RT1: one top-level scope per artifact: tables34 record-only, a slot readout, a figure, an errored artifact (NOT VERIFIED, null)",
      set(_s) == set(_k) and _s["tables34"] == dg["scope"] and _p["tables34"] is False and _s["rc2"].startswith("slot readout strings verified")
      and _p["rc2"] is False and _s["fig4"].startswith("committed figure files") and _p["fig4"] is False
      and _s["rc1a"] == "NOT VERIFIED (error SystemExit in rc1a)" and _p["rc1a"] is None)
_s, _p = VC.result_scope(_k, None, [{"class": "Stop", "where": "verify"}], {"class": "Stop", "where": "verify"})
check("RT1: a whole-run error labels every artifact NOT VERIFIED with a null flag",
      all(v == "NOT VERIFIED (error Stop in verify)" for v in _s.values()) and set(_s) == set(_k) and all(v is None for v in _p.values()))
s, _ = run_tables(approve=False)
check("R1: an unapproved not-printed list STOPs", any("awaits 09's approval" in x for x in s["stop"]["other"]))
T1 = copy.deepcopy(T_REC)
T1["false_alarms"]["new_count"] = 3
s, _ = run_tables(T=T1, approve_from=T_REC)
check("R1: a new numeric field changes the list and STOPs", any("is not the approved one" in x for x in s["stop"]["other"]))
lb0, nm0 = events[0]["probe"], VC.POOL_NM[events[0]["pool"]]
A2 = copy.deepcopy(A)
for r in A2["rows"][lb0].values():
    r["pools_at_draw0"]["0.01"][nm0]["T"] = float(np.nextafter(r["pools_at_draw0"]["0.01"][nm0]["T"], np.inf))
s, dg = run_tables(A_=A2)
check("tables: condition (2) failing (A's T one ulp off) makes that label/pool's z orphans: STOP",
      "2:T" in dg["conditions"][f"{lb0}|{nm0}"] and "tables34::T3|fa1|z" in s["stop"]["orphan"])
d2 = copy.deepcopy(d1)
d2[lb0]["sha256_read"] = "not-the-pin"
s, dg = run_tables(d1_=d2)
check("tables: condition (1) failing (npz sha not A's pin) STOPs with orphans",
      dg["conditions"][f"{lb0}|{nm0}"] == ["1:npz-sha-not-A's-pin"] and "tables34::T3|fa1|z" in s["stop"]["orphan"])
T2 = copy.deepcopy(T_REC)
k2 = f"{lb0}|{events[0]['pool']}"
T2["false_alarms"]["duplicates_dropped"] = [{"probe": lb0, "pool": events[0]["pool"], "run": "rX", "unit": 0}]
s, _ = run_tables(T=T2)
check("R2: a (probe, pool) with dropped duplicates is an orphan (the record's dedup is not used)",
      f"tables34::T3|events|{k2}" in s["stop"]["orphan"] and f"tables34::T3|events|{k2}" in s["stop"]["unmatched"])
check("RQ2: with a drop, the dedup count has no recomputed value (orphan)", "tables34::T3|duplicates_dropped_count" in s["stop"]["orphan"])
T3 = copy.deepcopy(T_REC)
T3["false_alarms"]["draws"] = events[:5]
s, _ = run_tables(T=T3)
check("tables: five draws instead of six STOPs", any("5 false-alarm draws, not 6" in x for x in s["stop"]["other"]))
T4 = copy.deepcopy(T_REC)
T4["false_alarms"]["draws"][2]["z"] = float(np.nextafter(T4["false_alarms"]["draws"][2]["z"], np.inf))
s, _ = run_tables(T=T4)
check("tables: a z one ulp off STOPs as unmatched", s["stop"]["unmatched"] == ["tables34::T3|fa3|z"])
T7 = copy.deepcopy(T_REC)
T7["false_alarms"]["draws"][1]["token"] += 10
T7["false_alarms"]["draws"][1]["unit"]["row_unit"] += 1
s, _ = run_tables(T=T7)
check("RQ2: a wrong false-alarm token and row_unit each STOP", {"tables34::T3|fa2|token", "tables34::T3|fa2|unit/row_unit"} <= set(s["stop"]["unmatched"]))
FA_X = copy.deepcopy(FA_U)
FA_X[1]["units"]["s"] = FA_X[1]["units"]["s"] + 1.0
s, _ = run_tables(fa_units=FA_X)
check("RQ2: score rows that do not reproduce the pool's scores leave the token unverified (orphan)", "tables34::T3|fa1|token" in s["stop"]["orphan"])
check("09 option (i): unit/token is on the not-printed list again (no row checks it)", not any(r.endswith("|unit/token") for r in s["stop"]["unmatched"]) and any(p.endswith("/unit/token") for p in run_tables()[1]["not_printed"]["paths"]))
FA_B = copy.deepcopy(FA_U)
FA_B[1]["units"]["sbm"] = FA_B[1]["units"]["sbm"] + 1.0
s, _ = run_tables(fa_units=FA_B)
check("O-a: score rows whose five-model scores differ from the pool's leave the token unverified (orphan)", "tables34::T3|fa1|token" in s["stop"]["orphan"])
T5 = copy.deepcopy(T_REC)
T5["outcome_table"]["outcomes"]["both caught"]["drawn"]["cell"] = "S_old|c9"
s, _ = run_tables(T=T5)
check("R3: a drawn cell outside A's cohort STOPs", any("not in A's cohort" in x for x in s["stop"]["other"]))
T6 = copy.deepcopy(T_REC)
T6["outcome_table"]["outcomes"]["both caught"]["drawn"]["cell"] = "S_old|c3"
s, _ = run_tables(T=T6)
check("R3: A's hit bits not giving the outcome STOPs", any("do not give this outcome" in x for x in s["stop"]["other"]))

# ---------- (2b) Table 4's attack-side z (09's ruling (b)) ----------
unit_z = VC.extract_function(blob("88589b507a8439d4abb85be40a1d60f4590fd34d"), "unit_z")  # run_tables_v81.py L260-281
argmax_command = VC.extract_function(blob("f191a2477226f868633ec2c5f6092ebb85d48fea"), "argmax_command")  # run_examples_v81.py


def npz(**arrs):
    b = io.BytesIO()
    np.savez(b, **arrs)
    return np.load(io.BytesIO(b.getvalue()), allow_pickle=False)


D1O = npz(meta_json=np.array('{"folds": [0, 1, 2, 3, 4], "fold_model": true, "mom_none": false}'),
          **{f"joint__{nm}__mu": rng.normal(size=(3, 5)) for nm in ("B", "S")},
          **{f"joint__{nm}__sd": np.abs(rng.normal(size=(3, 5))) + 1.0 for nm in ("B", "S")},
          **{f"joint__{nm}__mu0": rng.normal(size=3) for nm in ("B", "S")}, **{f"joint__{nm}__sd0": np.abs(rng.normal(size=3)) + 1.0 for nm in ("B", "S")})
SC = npz(key=np.array(["attack|c0"] * 4 + ["attack|c1"] * 3), run_key=np.array(["runA"] * 4 + ["runB"] * 3),
         token=np.array([3, 9, 20, 41, 1, 2, 3]), fold=np.array([0, 2, -1, 4, 1, 1, 1]), score_by_model=rng.normal(size=(7, 5)) + 40,
         score=rng.normal(size=7) + 40)  # scores well above any synthetic T, so the side is caught as A's bit says
mom, fm, fl = VC.d1_moments(D1O, "B")
units = VC.attack_units(SC, "runA", 5)
res = VC.attack_argmax(units, mom, fm, fl, unit_z, argmax_command)
TB = A["rows"][VC.OUTCOME_LABEL][VC.MEMBER_ROW.format(o="S_old")]["pools_at_draw0"]["0.01"]["B"]["T"]
side = {"z": res["z"], "T": TB, "z_minus_T": res["z"] - TB, "unit_index": res["unit"], "n_units": res["n_units"], "caught": True,
        "token": res["token"], "unit": {"row_unit": res["unit"], "token": res["token"]}, "run_key": "runA"}
TA = copy.deepcopy(T_REC)
TA["outcome_table"]["outcomes"]["both caught"]["drawn"]["real (run 1)"] = side
ATT = {"both caught|real (run 1)": {"result": res, "frozen_ok": True, "score_ref": "vol:conv:x.npz@s", "why": "", "file_cells_ok": True, "file_arm_ok": True}}
s, _ = run_tables(T=TA, attack=ATT)
check("attack z: positive control PASS (FROZEN-pinned scores, D1 moments, the real unit_z and argmax_command)",
      s["verdict"] == "PASS" and s["n_printed"] == 1 + 4 + 7 + 6 * 5 + 12 + 1 + 1 and units["file_keys"] == ["attack|c0"])
other = (res["unit"] + 1) % res["n_units"]
uz_all = unit_z(units, mom, fm, fl)
TW = copy.deepcopy(TA)
TW["outcome_table"]["outcomes"]["both caught"]["drawn"]["real (run 1)"] |= {"unit_index": other, "z": float(uz_all[other]),
                                                                            "z_minus_T": float(uz_all[other]) - TB}
s, _ = run_tables(T=TW, attack=ATT)
check("attack z: a wrong argmax (another unit's z and id) STOPs as unmatched",
      {"tables34::T4|both caught|real (run 1)|z", "tables34::T4|both caught|real (run 1)|unit_index"} <= set(s["stop"]["unmatched"]))
s, _ = run_tables(T=TA, attack=ATT, signed=SIG_T - {"shaF"})
check("attack z: FROZEN not in the signed set makes the side's z an orphan", "tables34::T4|both caught|real (run 1)|z" in s["stop"]["orphan"])
ATT_X = {k: v | {"file_cells_ok": False} for k, v in ATT.items()}
s, _ = run_tables(T=TA, attack=ATT_X)
check("R3: a run_key whose score-file keys are another cell's STOPs", any("not the drawn cell's" in x for x in s["stop"]["other"]))
ATT_Y = {k: v | {"file_arm_ok": False} for k, v in ATT.items()}
s, _ = run_tables(T=TA, attack=ATT_Y)
check("O2: score-file keys of the other arm STOP", any("not this side's arm" in x for x in s["stop"]["other"]))
FZ = {"overrides": {"S_old|C": {"commit": "c1", "path": "ov.json", "sha256": VC.sha(b'{"arms": {"a": {"x.npz": {"path": "p/x.npz", "sha256": "s1"}}}}')},
                    "S_old|R": {"commit": "c2", "path": "ovr.json", "sha256": "r"}}}
OV = {"S_old|C": {"commit": "c1", "path": "ov.json", "sha256": FZ["overrides"]["S_old|C"]["sha256"],
                  "bytes": b'{"arms": {"a": {"x.npz": {"path": "p/x.npz", "sha256": "s1"}}}}'}}
check("FROZEN lookup: the score file's pin is found through FROZEN's C override", VC.score_pins(FZ, OV) == {"p/x.npz": {"s1"}})
for nm_, ov_ in (("a missing C override", {}), ("override bytes that are not FROZEN's pin", {"S_old|C": OV["S_old|C"] | {"bytes": b"{}"}})):
    try:
        VC.score_pins(FZ, ov_)
        ok_ = False
    except SystemExit:
        ok_ = True
    check(f"FROZEN lookup: {nm_} STOPs", ok_)

# ---------- (2c) figures: committed outputs vs the re-render and the record by key ----------
REC = {"pooled": {"x C": {"joint": {"0.01": {"gap": {"delta_full": {"point": 0.0123, "ci95": [-0.01, 0.031]}}}}}}, "n": 1906}
SRC = {"inputs": {"rec": {"file": "b.json", "sha256": "shaB"}}, "git": "c",
       "values": [{"input": "rec", "key": ["pooled", "x C", "joint", "0.01", "gap", "delta_full"], "point": 0.0123, "ci95": [-0.01, 0.031]},
                  {"input": "rec", "key": ["n"], "value": 1906}],
       "n_tasks": {"x C": 1906}, "gate": {"not_readable": {"total": 0}}}
TXT = "# Figure 4: gap\n\n1,906 tasks; not-readable pool flags at these levels: 0. Values printed at 1 in 100.\n"


def run_fig(ps=SRC, rs=SRC, pt=TXT, rt=TXT, signed=frozenset({"shaB"})):
    r, p, st = VC.fig_rows("figX", ps, rs, pt, rt, {"b.json": REC}, {"b.json": "shaB"}, set(signed))
    return VC.summarize(r, {"figX": p}, st)


s = run_fig()
check("figure: committed = re-render = record by key: PASS", s["verdict"] == "PASS" and s["n_printed"] == 4 + 2 + 5)
bad = copy.deepcopy(SRC)
bad["values"][0]["ci95"][1] = 0.0311
s = run_fig(ps=bad, rs=bad)
check("figure: a plotted value off the record (even when the re-render agrees) STOPs", s["stop"]["unmatched"] == ["figX::values[0]|ci95[1]"])
s = run_fig(rt=TXT.replace("1,906", "1,907"))
check("figure: a caption number differing from the re-render STOPs", s["stop"]["unmatched"] == ["figX::text#1"])
s = run_fig(rt=TXT + "Dropped: 3.\n")
check("figure: the re-render printing one more number STOPs on the count", s["stop"]["count_mismatch"] == ["figX"])
s = run_fig(signed=frozenset())
check("figure: an unsigned input makes its numbers orphans", s["verdict"] == "STOP" and len(s["stop"]["orphan"]) == s["n_printed"])



def png(idat, text=b""):
    def ch(t, d):
        return len(d).to_bytes(4, "big") + t + d + b"\0\0\0\0"
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", b"\0" * 13) + (ch(b"tEXt", text) if text else b"") + ch(b"IDAT", idat) + ch(b"IEND", b"")


st_, dg_ = VC.png_gate("figX", png(b"abc"), png(b"abc"))
check("RQ1: a byte-identical re-rendered PNG passes the gate", st_ == [] and dg_["png_bytes_identical"] is True)
st_, dg_ = VC.png_gate("figX", png(b"abc"), png(b"abd"))
check("RQ1: a PNG that differs from the re-render STOPs, naming the differing chunk (IDAT) and counts only",
      bool(st_) and dg_["chunk_names_differing"] == ["IDAT"] and dg_["chunk_count_committed"] == 3 and "abc" not in json.dumps(dg_))
st_, dg_ = VC.png_gate("figX", png(b"abc", b"Software\0mpl 3.10.6"), png(b"abc", b"Software\0mpl 3.10.7"))
check("RQ1: a metadata-only difference still STOPs (reported as tEXt, never loosened silently)", bool(st_) and dg_["chunk_names_differing"] == ["tEXt"])

# ---------- (3) R5 and the recheck selection ----------


ok_row = VC.row("p", "n1", "-", "-", "1", "1", "count", True, ["shaB"], {"shaB"})
REASON = "not route C; behaviour record"
PINS = {"accepted_reasons": [REASON]}
dfr = VC.row("p", "d1", "-", "-", "2", None, f"NOT VERIFIED ({REASON})", True, ["deferred"], {"deferred"}, unit="deferred", reason=REASON,
             record_ok=True)
s = VC.summarize([ok_row, dfr], {"p": 2}, pins=PINS)
check("RN1: an accepted deferred row can never yield PASS (PARTIAL, counted apart, not verified)",
      s["verdict"] == "PARTIAL" and s["n_equal"] == 1 and s["deferred_by_reason"] == {f"NOT VERIFIED ({REASON})": 1})
s = VC.summarize([ok_row, dfr | {"reason": "pending", "accepted": True}], {"p": 2}, pins=PINS)
check("C1: a deferred row whose reason is not in 09's pinned list is an orphan, whatever its own flags say",
      s["verdict"] == "STOP" and s["stop"]["orphan"] == ["p::d1"])
s = VC.summarize([ok_row, dfr], {"p": 2})
check("C1: with no 09 pin, no deferred row counts (orphan)", s["verdict"] == "STOP" and s["stop"]["orphan"] == ["p::d1"])
s = VC.summarize([ok_row, dfr | {"record_ok": False}], {"p": 2}, pins=PINS)
check("09 (10-20): a null record (record_ok False) makes a deferred row an orphan, even with an accepted reason",
      s["verdict"] == "STOP" and s["stop"]["orphan"] == ["p::d1"])
nr = VC.row("p", "c1", "-", "Table 3", "3", None, "reference", True, ["nonresult"], {"nonresult"}, unit="nonresult", nr_class="reference",
            matched="Table 3", checks_ok=True)
s = VC.summarize([ok_row, nr | {"approved": True}], {"p": 2})
check("C1: a non-result row without 09's approved list STOPs (orphan), even flagged approved by itself",
      s["verdict"] == "STOP" and s["stop"]["orphan"] == ["p::c1"])
_, g_sha, g_n = VC.nonresult_list([nr])
s = VC.summarize([ok_row, nr], {"p": 2}, pins={"nonresult_approved": {"sha256": g_sha, "count": g_n}})
check("C1: a non-result row on 09's approved list (recomputed sha, count) is counted apart and PASSes", s["verdict"] == "PASS" and s["n_nonresult"] == 1)
s = VC.summarize([ok_row, nr | {"checks_ok": False}], {"p": 2}, pins={"nonresult_approved": {"sha256": g_sha, "count": g_n}})
check("C1: a non-result row whose own checks fail (hypothetical) is an orphan", s["verdict"] == "STOP" and s["stop"]["orphan"] == ["p::c1"])
nrc = VC.row("p", "u1", "-", "-", "4", None, "eq", True, ["not-route-C"], {"not-route-C"}, unit="not_route_C")
s = VC.summarize([ok_row, nrc], {"p": 2})
check("not route C: a checked unsigned-record value is counted apart (not verified, not an orphan); R7: the verdict line names the bucket",
      s["verdict"] == "PASS; 1 not route C, checked against unsigned records by key" and s["verdict_base"] == "PASS"
      and s["n_not_route_C_checked"] == 1 and s["n_equal"] == 1)
check("09 words: the verdict line names the ungated number words", VC.verdict_line("PASS", 2, {"total": 9, "covered": 4, "uncovered": 5}) ==
      "PASS (digits); 2 not route C, checked against unsigned records by key; 9 number words not gated: 4 covered by a row, 5 uncovered")
s = VC.summarize([ok_row, nrc | {"equal": False}], {"p": 2})
check("not route C: an unequal unsigned-record value STOPs", s["verdict"] == "STOP" and s["stop"]["unmatched"] == ["p::u1"])


def boom():
    raise SystemExit("own_all C: 3 member rows for pooled_all_origins, expected 4 (STOP)")


out_, err_, tb_ = VC.guarded(boom, "rc2")
check("R5: an exception reaches the reply as its class and phase only (no message)",
      out_ is None and err_ == {"class": "SystemExit", "where": "rc2"} and "3 member rows" not in json.dumps(err_) and "3 member rows" in tb_)
s, rows = run_slots(PRINTED)
check("recheck: seeded selection is reproducible and draws number rows only",
      VC.select_recheck(rows, 20261020, 2) == VC.select_recheck(rows, 20261020, 2) and all("#" in x[1] for x in VC.select_recheck(rows, 20261020, 4)))

# numv7 (10-20): the reply guard STOPped on an over-long verdict line; 2e / 09: the guard stays at 300 and the line is neutral, with the
# full PRELIMINARY labels in the result's scope
LB_ = "route C, out of v1 scope (PRELIMINARY); checked against a signed record by key (09 S2 marker S2_2026-10-20_015417 @ 114ca1583)"
VL_ = VC.verdict_line("STOP", 15, {"total": 234, "covered": 43, "uncovered": 191}, n_pre=4, pre_labels=[LB_])
pre_clause = next(c for c in VL_.split("; ") if "PRELIMINARY" in c)
check("R7-1 (numv7 shape: 4 PRELIMINARY, 15 not route C, number words): the verdict line is at most 300 characters, its PRELIMINARY clause "
      "never says 'unsigned', and it points to the scope instead of carrying the label",
      len(VL_) <= 300 and "unsigned" not in pre_clause and "labels in scope" in VL_ and LB_ not in VL_)
_sP, _ = VC.result_scope({"post": "post"}, {"post": {"preliminary_labels_used": [LB_]}})
check("R7-1: the post's scope entry carries 09's full PRELIMINARY label verbatim", LB_ in _sP["post"])
import importlib.util  # noqa: E402
import unittest.mock  # noqa: E402

if importlib.util.find_spec("modal") is None:  # the reply guard lives in the runner; a stub stands in for modal when it is absent
    sys.modules["modal"] = unittest.mock.MagicMock()
import run_numbers_verify as RN  # noqa: E402


def reply_stops(v):
    try:
        RN.reply({"verdict": v, "stop": {"unmatched": [], "orphan": [], "duplicate_slots": []}}, {}, {}, {})
        return False
    except VC.Stop:
        return True


check("reply guard unchanged (2e / 09): a verdict line over 300 characters still STOPs; the numv7-shaped neutral line passes",
      reply_stops("x" * 301) and not reply_stops(VL_))

print(f"{len(FAILS)} failed")
sys.exit(1 if FAILS else 0)
