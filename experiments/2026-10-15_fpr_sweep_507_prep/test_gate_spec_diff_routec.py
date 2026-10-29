"""Synthetic tests of gate_spec_diff_routec.gate (no git, no Modal): the paid route C spec may differ from the pre-flight's only by the pinned
zero-unit list.

  uv run python experiments/2026-10-15_fpr_sweep_507_prep/test_gate_spec_diff_routec.py
"""

import gate_spec_diff_routec as G

PIN = "meta/full_length/s_old_simh_extra_roots/simh_q4f_v1_pilot__sim_honest/zero_unit_list"
ZU = ["jobs/3/1/simh_old/Sol (a) C/2/zero_unit_ok/0", "jobs/9/1/simh_old/Sol (a) C/2/zero_unit_ok/0"]
PF = {"spec": {"sha256": "a" * 64}, "zero_unit_by_file": {"f" * 64: ["simh_pilot|t21_task"]}, "expected_zero_unit_ok_paths": 2}
PF0 = {"spec": {"sha256": "a" * 64}, "zero_unit_by_file": {"f" * 64: []}, "expected_zero_unit_ok_paths": 0}
PIN_ = {"path": "results/routec_preflight/p.json", "sha256": "9" * 64}
ZUOK = ["simh_pilot|t21_task"]


def d(only_b=None, only_a=(), changed=(), a="a" * 64, b="b" * 64, **kw):
    ob = list(only_b if only_b is not None else ZU + [PIN])
    return {"a": {"sha256": a}, "b": {"sha256": b}, "only_a": list(only_a), "only_b": ob, "changed": list(changed),
            "n_differences": len(only_a) + len(ob) + len(changed), "ignored_differing": ["meta/git"], "normalized_differing": ["dest"],
            "a_pins": [], "b_pins": [{"at": PIN} | PIN_],
            "a_routec_entries": [{"at": f"jobs/{j}/1/simh_old/Sol (a) C/2", "file_sha256": "f" * 64, "zero_unit_ok": []} for j in (3, 9)],
            "b_routec_entries": [{"at": f"jobs/{j}/1/simh_old/Sol (a) C/2", "file_sha256": "f" * 64, "zero_unit_ok": ZUOK} for j in (3, 9)],
            "b_meta": {"git": "h" * 40, "dest": "xfer/fpr507prep/records/results/x.json", "pooled": True, "shipped_inputs": {}}} | kw


cases = {
    "PASS: pinned zero-unit keys on tagged entries + the meta pin": G.gate(d(), PF, PIN_)["pass"],
    "PASS: no zero-unit keys: the meta pin only": G.gate(d(only_b=[PIN], b_routec_entries=[{"at": f"jobs/{j}/1/simh_old/Sol (a) C/2", "file_sha256": "f" * 64,
                                                                                           "zero_unit_ok": []} for j in (3, 9)]), PF0, PIN_)["pass"],
    "FAIL: side a is not the pre-flight's spec": not G.gate(d(a="c" * 64), PF, PIN_)["pass"],
    "FAIL: side b is the same spec": not G.gate(d(b="a" * 64), PF, PIN_)["pass"],
    "FAIL: any other difference (a changed cfg field)": not G.gate(d(changed=["jobs/3/1/e12/fill_S_old/0/admitted/2"]), PF, PIN_)["pass"],
    "FAIL: a zero_unit_ok path on side a": not G.gate(d(only_a=ZU[:1]), PF, PIN_)["pass"],
    "FAIL: a zero_unit_ok path off the pattern (an S_paper493-like job field)": not G.gate(d(only_b=ZU + [PIN, "jobs/3/1/sp493/zero_unit_ok/0"]), PF, PIN_)["pass"],
    "FAIL: the meta pin missing": not G.gate(d(only_b=ZU), PF, PIN_)["pass"],
    "FAIL: the meta pin twice": not G.gate(d(only_b=ZU + [PIN, PIN.replace("pilot__", "pilot2__")]), PF, PIN_)["pass"],
    "FAIL: zero_unit_ok paths while the pre-flight lists none": not G.gate(d(), PF0, PIN_)["pass"],
    "FAIL: zero-unit keys listed but no zero_unit_ok path": not G.gate(d(only_b=[PIN]), PF, PIN_)["pass"],
    "FAIL: a planted diff": not G.gate(d(plant_b=True), PF, PIN_)["pass"],
    "FAIL: n_differences inconsistent": not G.gate(d(n_differences=99), PF, PIN_)["pass"],
    "FAIL (i): the paid spec pins another pre-flight record": not G.gate(d(b_pins=[{"at": PIN, "path": PIN_["path"], "sha256": "8" * 64}]), PF, PIN_)["pass"],
    "FAIL (i): no pin on side b": not G.gate(d(b_pins=[]), PF, PIN_)["pass"],
    "FAIL (i): the pre-flight's own spec carries a pin": not G.gate(d(a_pins=[{"at": PIN} | PIN_]), PF, PIN_)["pass"],
    "FAIL (iii): one zero_unit_ok path fewer than expected": not G.gate(d(only_b=ZU[:1] + [PIN]), PF, PIN_)["pass"],
    "FAIL (iii): one more than expected": not G.gate(d(only_b=ZU + ["jobs/12/1/simh_old/Sol (a) C/2/zero_unit_ok/0", PIN]), PF, PIN_)["pass"],
}
# Astra r2 R1, with REAL spec_diff output: same count, same paths, the right pin, but another admitted key on both entries of the one file
import spec_diff as SD  # noqa: E402

K0, K1 = "simh_pilot|t21_task", "simh_pilot|t22_task"


def spec_(zu, pin=None):
    ent = {"batch": "p", "root": "simh_q4f_v1_pilot__sim_honest", "path": "p.npz", "sha256": "f" * 64, SD.ROUTEC_TAG: True, "admitted": [K0, K1], "zero_unit_ok": zu}
    so = {"Sol (a) C": [{"root": "b1", "sha256": "b" * 64}, ent]}
    m = {"git": "h" * 40, "stamp": "s", "full_length": {"s_old_simh_extra_roots": {"simh_q4f_v1_pilot__sim_honest": {"n_keys": 20} | (
        {"zero_unit_list": pin} if pin else {})}}}
    return {"mode": "alltasks", "jobs": [("Sol (a) C", {"origin": "S_old", "simh_old": so}, None), ("Sol (a) R", {"origin": "S_old", "simh_old": so}, None)],
            "meta": m, "IN": {}, "cfg": {}, "label": "x", "dest": "r_s.json"}


def real(b_zu, reverse_b=False):
    a_, b_ = spec_([]), spec_(b_zu, dict(PIN_, n_keys=1))
    if reverse_b:  # Astra launch r1: the same simh_old map with another insertion order on side b
        b_["jobs"] = [(lb, c | {"simh_old": dict(reversed(list((c["simh_old"] | {"Sol (a) R": c["simh_old"]["Sol (a) C"]}).items())))}, x)
                      for lb, c, x in b_["jobs"]]
        a_["jobs"] = [(lb, c | {"simh_old": c["simh_old"] | {"Sol (a) R": c["simh_old"]["Sol (a) C"]}}, x) for lb, c, x in a_["jobs"]]
    rec = SD.diff(a_, b_) | {"a_pins": SD.pins_at(a_), "b_pins": SD.pins_at(b_), "a_routec_entries": SD.routec_entries(a_), "b_routec_entries": SD.routec_entries(b_),
                             "a_meta": SD.side_meta(a_), "b_meta": SD.side_meta(b_)}
    return rec | {"a": {"sha256": "a" * 64}, "b": {"sha256": "b" * 64}, "plant_b": False}


PFR = {"spec": {"sha256": "a" * 64}, "zero_unit_by_file": {"f" * 64: [K0]}, "expected_zero_unit_ok_paths": 2}
cases |= {
    "real spec_diff output, the pinned key on both entries: PASS": G.gate(real([K0]), PFR, PIN_)["pass"],
    "real spec_diff output, ANOTHER admitted key on both entries (same count, same paths, right pin): FAIL": not G.gate(real([K1]), PFR, PIN_)["pass"],
    "the verdict carries side b's launch metadata (no value)": G.gate(real([K0]), PFR, PIN_)["b_meta"] == {"git": "h" * 40, "dest": "r_s.json", "pooled": False,
                                                                                                     "shipped_inputs": {}, "keys_only": False},
    "FAIL: side b's launch metadata missing": not G.gate(d(b_meta=None), PF, PIN_)["pass"],
    "real spec_diff output, side b's simh_old in REVERSED insertion order (same entries): PASS, no false STOP": G.gate(
        real([K0], reverse_b=True), PFR | {"expected_zero_unit_ok_paths": 4}, PIN_)["pass"],
    "ROUTEC_TAG is full_override's S_OLD_ONLY": SD.ROUTEC_TAG == __import__("full_override").S_OLD_ONLY,
}
# gate_keys_only (the census launch, Aswin A1, 10-17): side a is a second --spec-only build of the same flags at the same HEAD; no pre-flight pin
KD = "xfer/fpr507prep/records/results/nofilter_keys_FULL_TEST_2026-10-17_120000.json"
KM = {"git": "h" * 40, "dest": KD, "pooled": False, "shipped_inputs": {}, "keys_only": True}


def dk(**kw):
    return {"a": {"sha256": "a" * 64}, "b": {"sha256": "b" * 64}, "only_a": [], "only_b": [], "changed": [], "n_differences": 0, "plant_b": False,
            "ignored_differing": ["meta/stamp"], "normalized_differing": ["dest"], "order_differing": [], "a_pins": [], "b_pins": [],
            "a_routec_entries": [], "b_routec_entries": [], "a_meta": KM | {"dest": KD.replace("120000", "115900")}, "b_meta": KM} | kw


def census_(stamp, ko=True):  # a census spec as dispatch("keys", ...) builds it: every job keys_only
    jobs = [(lb, {"origin": "S_newmains"} | ({"keys_only": True} if ko or lb.endswith("C") else {}), None) for lb in ("Sol (a) C", "Sol (a) R")]
    return {"mode": "keys", "jobs": jobs, "meta": {"git": "h" * 40, "stamp": stamp}, "IN": {}, "cfg": {}, "label": "x",
            "dest": f"xfer/fpr507prep/records/results/nofilter_keys_FULL_TEST_{stamp}.json"}


def real_keys(b_ko=True):
    a_, b_ = census_("2026-10-17_115900"), census_("2026-10-17_120000", b_ko)
    return SD.diff(a_, b_) | {"a_pins": SD.pins_at(a_), "b_pins": SD.pins_at(b_), "a_routec_entries": SD.routec_entries(a_), "b_routec_entries": SD.routec_entries(b_),
                              "a_meta": SD.side_meta(a_), "b_meta": SD.side_meta(b_), "a": {"sha256": "a" * 64}, "b": {"sha256": "b" * 64}, "plant_b": False}


GK = G.gate_keys_only
cases |= {
    "keys-only PASS: two census builds, 0 counted differences, no route C entry or pin": GK(dk())["pass"],
    "keys-only PASS on REAL spec_diff output (side_meta relays keys_only)": GK(real_keys())["pass"] and real_keys()["b_meta"]["keys_only"] is True,
    "keys-only FAIL: a tagged route C entry present": not GK(dk(b_routec_entries=[{"at": "jobs/0/1/simh_old/Sol (a) C/2", "file_sha256": "f" * 64,
                                                                                  "zero_unit_ok": []}]))["pass"],
    "keys-only FAIL: the meta pin present": not GK(dk(b_pins=[{"at": PIN} | PIN_]))["pass"],
    "keys-only FAIL: a differing path": not GK(dk(changed=["jobs/0/1/e12/fill_S_old/0/admitted/2"], n_differences=1))["pass"],
    "keys-only FAIL: side b's keys_only false": not GK(dk(b_meta=KM | {"keys_only": False}))["pass"],
    "keys-only FAIL on REAL spec_diff output: one job of side b not keys-only": not GK(real_keys(b_ko=False))["pass"],
    "keys-only FAIL: a wrong dest (not the FULL TEST census record)": not GK(dk(b_meta=KM | {"dest": KD.replace("_FULL_TEST_", "_FULL_")}))["pass"],
    "keys-only FAIL: a dest not under xfer": not GK(dk(b_meta=KM | {"dest": KD.removeprefix("xfer/fpr507prep/records/")}))["pass"],
    "keys-only FAIL: side b's build commit missing": not GK(dk(b_meta=KM | {"git": None}, a_meta=KM | {"git": None}))["pass"],
    "keys-only FAIL: side a built at another commit": not GK(dk(a_meta=KM | {"git": "g" * 40}))["pass"],
    "keys-only FAIL: side a is the same spec as side b": not GK(dk(b={"sha256": "a" * 64}))["pass"],
    "keys-only FAIL: no b_meta at all": not GK(dk(b_meta=None))["pass"],
    # RQ-A1 (a63478's first review of ca25a1b44): each condition on its own, everything else clean
    "keys-only FAIL: a planted diff alone": not GK(dk(plant_b=True))["pass"],
    "keys-only FAIL: side a's keys_only false": not GK(dk(a_meta=KM | {"keys_only": False}))["pass"],
    "keys-only FAIL: a zero-unit path in ignored_differing (lists and pins empty)": not GK(dk(ignored_differing=["jobs/0/1/simh_old/X/2/zero_unit_ok/0"]))["pass"],
    "keys-only FAIL: side a's sha256 missing": not GK(dk(a={}))["pass"],
    "keys-only FAIL: n_differences 1 with every list empty": not GK(dk(n_differences=1))["pass"],
    "keys-only FAIL: a path in only_b with n_differences 0": not GK(dk(only_b=["x"]))["pass"],
}
for name, ok in cases.items():
    print(("PASS " if ok else "FAIL ") + name)
assert all(cases.values()), "gate tests failed"
print(f"{len(cases)} checks pass")
