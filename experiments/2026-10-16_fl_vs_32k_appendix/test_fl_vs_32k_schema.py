"""Schema test on SYNTHETIC records (fake numbers shaped like the Stage A all-tasks records, their per-draw npz and the PANEL records; no real
value is read). Runs the table generator in a temp folder, then the gates' negative cases. The figure smoke test runs only with
FL_TEST_FIGURE=1 (the figure is rendered only if the author asks).

  uv run --no-project --python 3.12 --with numpy==2.3.3 --with matplotlib==3.10.6 python experiments/2026-10-16_fl_vs_32k_appendix/test_fl_vs_32k_schema.py
"""

import copy
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_fl_vs_32k_table import npzkey

SETS = tuple(f"{o} (all-origins draws)" for o in ("S_old", "S_new", "S_newmains", "S_paper493"))  # pooled_all_origins' members (RULE item 33)
NCELL = dict(zip(SETS, (19, 30, 35, 24)))
CB = "pooled_all_origins"
DRAWS = 50
PUB = ["FIBS", "harmfulness", "refusal", "high-stakes"]
PUB_F = [f"{p}|uncut" for p in PUB]
SIX = {"cut32k": ["Sol (a)", *PUB, "own_all_rs_L48"], "full": ["sol_a_uncut", *PUB_F, "own_all_rs_uncut_L48"]}
OFF5 = {"cut32k": SIX["cut32k"][:5], "full": SIX["full"][:5]}
LABELS = {"cut32k": SIX["cut32k"], "full": SIX["full"] + ["Sol (a)|uncut", "own_all_rs_L48|uncut"]}
SIXKEY = {"cut32k": "headline_six", "full": "headline_six_full"}
EXCKEY = {"cut32k": "off_policy", "full": "off_policy_five_full"}
WIN = {"cut32k": "cut32768", "full": "full"}
_SHA = {}  # each window's synthetic Stage A record sha256, as written (for the PANEL binding test)


def recs_sha(w):
    return _SHA[w]


def fake_node(rng, withheld=False):
    p = rng.uniform(-0.15, 0.15)
    n = {"point": p, "finite_draws": DRAWS, "draws": DRAWS}
    return n | ({"ci95": None, "withheld": "fewer than 95% complete draws"} if withheld else {"ci95": [p - 0.05, p + 0.05], "degenerate": False})


def fake_record(w, seed, withheld_label=None, lead="A"):
    rng = random.Random(seed)
    W = np.random.default_rng(7).integers(0, 3, (DRAWS + 1, len(SETS))).astype(float)  # draw 0 + DRAWS draws, the same in both windows
    meta = {"lead_readout": "C", "window": WIN[w], "cohort_list": {"ref": "SYNTHETIC cohort list", "headline_cells": NCELL}}
    if w == "full":
        meta |= {"estimator": {"lead": lead}, "calibration_label": "full length (lead) SYNTHETIC calibration"}
    rec = {"meta": meta, "pooled": {}, "coverage": {}, "rows": {}, "coheadline_signed": {CB: {"joint": {"0.01": {}}}},
           "panel_groups": {CB: {"pooled": {"joint": {"0.01": {}}}}}, "npz": {"path": f"npz_{w}.npz", "volume": "synthetic"}}
    arrs = {}
    for ro in ("C", "R"):
        for p in LABELS[w]:
            lab = f"{p} {ro}"
            rec["pooled"][lab] = {CB: {"members": list(SETS), "withheld_members": {}, "role": "lead", "joint": {"0.01": {
                "gap": {k: fake_node(rng, withheld=(lab == withheld_label and k == "delta_full")) for k in ("delta_full", "delta_rr", "sim_minus_run1", "sim_minus_run2")},
                "readable": {"all": True}, "D_draw0": {s: float(W[0, i]) for i, s in enumerate(SETS)}}}}}
            rec["coverage"][lab] = {s: {"coverage": {"cohort_cells": NCELL[s]}} for s in SETS}
            rec["rows"][lab] = {s: {"cohort_cell_keys": sorted(f"t{i:03d}_{s}" for i in range(NCELL[s]))} for s in SETS}
            for i, s in enumerate(SETS):
                arrs[npzkey(lab, CB, "joint", "0.01", s, "D")] = W[:, i]
        rec["coheadline_signed"][CB]["joint"]["0.01"][f"{SIXKEY[w]}|{ro}"] = {
            "S": fake_node(rng), "Y": fake_node(rng), "abs_Y": fake_node(rng), "rows": [f"{p} {ro}" for p in SIX[w]]}
        rec["panel_groups"][CB]["pooled"]["joint"]["0.01"][f"{EXCKEY[w]}|{ro}"] = {
            "primary": {"rows": [f"{p} {ro}" for p in OFF5[w]], "n": 5, "mean_G_minus_mean_Y": fake_node(rng)}}
    return rec, arrs


def pts(n):
    return {**n, "point": 100 * n["point"], **({"ci95": [100 * x for x in n["ci95"]]} if n.get("ci95") else {})}


def fake_panel(w, rec, stage_sha, hc=CB):
    """A PANEL FINAL record (points) made from the Stage A record's own groups; its npz source names that record and npz. PANEL keeps its
    headline combo's results under "pooled" and other combos by name; its coheadline is the headline combo's, others under other_combos."""
    g = rec["panel_groups"][CB]["pooled"]["joint"]["0.01"]
    src = {"path": rec["npz"]["path"], "sha256": rec["npz"]["sha256"], "record": {"git": "synthetic", "sha256": stage_sha}, "labels": sorted(rec["pooled"])}
    set_ = "pooled" if hc == CB else CB
    pan = {"meta": {"git": "deadbeef0123", "config": {"headline_combo": hc, "npz": [src]}},
           "results": {set_: {"0.01": {"groups": {k: {"primary": {"rows": v["primary"]["rows"], "mean_G_minus_mean_Y": pts(v["primary"]["mean_G_minus_mean_Y"])}}
                                                  for k, v in g.items()}}}}}
    if w == "full":
        co = rec["coheadline_signed"][CB]["joint"]["0.01"]
        blk = {"0.01": {ro: {"six": {"S": pts(fake_node(random.Random(9))), "S_estimator": "B", "S_A_beside": pts(co[f"{SIXKEY[w]}|{ro}"]["S"]),
                                     "rows": co[f"{SIXKEY[w]}|{ro}"]["rows"]}} for ro in ("C", "R")}}
        pan["coheadline"] = blk if hc == CB else {"other_combos": {CB: blk}}
    return pan


def setup(lead="A", withheld="refusal|uncut C"):
    plan = json.loads((HERE / "config" / "plan_TEMPLATE.json").read_text())
    for g in plan["groups"]:
        if g["stat"] != "excess":
            g["key"] = dict(SIXKEY)
        g["expect_members"] = copy.deepcopy(OFF5 if g["stat"] == "excess" else SIX)
    plan["full_calibration_label_path"] = ["meta", "calibration_label"]
    plan["excess_authority"] = {}  # filled by write() when a test asks for PANEL records
    recs = {"cut32k": fake_record("cut32k", 1), "full": fake_record("full", 2, withheld_label=withheld, lead=lead)}
    ln = {"probes": {}}
    for n, v in (("sol_a_registered", 32768), ("own_all_rs_registered", 32768), ("sol_a_uncut", None), ("own_all_rs_uncut", None)):
        ln["probes"][n] = {"status": "resolved", "probe": n, "by_readout": {ro: {"text": "SYNTHETIC", "field": "spec.cut", "value": v} for ro in "CR"}}
    for n in ("fibs", "harmfulness", "refusal", "high-stakes"):
        ln["probes"][n] = {"status": "resolved", "probe": n, "by_readout": {ro: {"text": "SYNTHETIC", "class_key": "iii"} for ro in "CR"}}
    return plan, recs, ln


def put(tmp, name, obj):
    b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    (tmp / name).write_bytes(b)
    return {"file": name, "sha256": hashlib.sha256(b).hexdigest(), "commit": "synthetic", "source": "sol:synthetic"}


def write(tmp, plan, recs, ln, panels=(), panel_edit=None, selection=None, hc=CB):
    lead = recs["full"][0]["meta"]["estimator"]["lead"]  # the selection record exists before the full record, which points to it
    plan["estimator_selection"] = put(tmp, "syn_selection.json", {"lead": selection or lead}) | {"lead_path": ["lead"]}
    recs["full"][0]["meta"]["estimator"].setdefault("selection_record", {"commit": "synthetic", "path": "results/syn_selection.json",
                                                                        "sha256": plan["estimator_selection"]["sha256"]})
    for w, (rec, arrs) in recs.items():
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrs)
        rec["npz"]["sha256"] = put(tmp, rec["npz"]["path"], buf.getvalue())["sha256"]
        plan["records"][w] = put(tmp, f"syn_{w}.json", rec)
    plan["lengths"] = put(tmp, "syn_lengths.json", ln)
    att = plan.pop("_att", None) or {"records": [plan["records"][w]["sha256"] for w in ("cut32k", "full")], "same_draw_streams": True}
    plan["draws_attestation"] = put(tmp, "syn_attestation.json", att) | {"pass_path": ["same_draw_streams"]}
    _SHA.update({w: plan["records"][w]["sha256"] for w in plan["records"]})
    code = plan.pop("_code", "deadbeef")  # the PANEL record's cited code commit (a prefix of its meta.git)
    for w in panels:
        pan = fake_panel(w, recs[w][0], plan["records"][w]["sha256"], hc=hc)
        if panel_edit:
            panel_edit(w, pan)
        plan["excess_authority"][w] = put(tmp, f"syn_panel_{w}.json", pan) | {"code": code}
    (tmp / "plan.json").write_text(json.dumps(plan))


def run(tmp, script, *args):
    return subprocess.run([sys.executable, str(HERE / script), *args, "--res-dir", str(tmp), "--git", "0" * 40],
                          capture_output=True, text=True, cwd=HERE, check=False)  # rc is asserted by each test


def table(tmp, stamp):
    return run(tmp, "make_fl_vs_32k_table.py", "--plan", str(tmp / "plan.json"), "--stamp", stamp, "--npz-root", str(tmp))


def build(mutate=None, panels=(), panel_edit=None, selection=None, lead="A", stamp="_syn", hc=CB):
    """Run the generator on a fresh synthetic set; returns (completed process, table or None, temp dir handle)."""
    d = tempfile.TemporaryDirectory()
    tmp = Path(d.name)
    plan, recs, ln = setup(lead=lead)
    if mutate:
        mutate(plan, recs, ln)
    write(tmp, plan, recs, ln, panels=panels, panel_edit=panel_edit, selection=selection, hc=hc)
    r = table(tmp, stamp)
    outs = list(tmp.glob(f"fl_vs_32k_table{stamp}_*.json"))
    return r, (json.loads(outs[0].read_text()) if outs else None), d


def expect_stop(mutate=None, *words, **kw):
    r, t, d = build(mutate, stamp="_neg", **kw)
    d.cleanup()
    assert r.returncode != 0 and t is None and all(w in r.stderr for w in words), r.stderr


def test_positive():
    def m(plan, recs, ln):
        q = recs["full"][0]["pooled"]["harmfulness|uncut R"][CB]["joint"]["0.01"]
        q["readable"]["all"] = False  # Stage A: a non-estimable threshold can leave a NaN point; the row must read "not readable", not stop
        q["gap"]["delta_full"]["point"] = float("nan")
        q["reading"] = "not readable (some pool at draw 0 has k < 1)"
        recs["full"][0]["coheadline_signed"][CB]["joint"]["0.01"]["headline_six_full|R"] = {"unavailable": "rows not readable ['harmfulness|uncut R']"}
        n2 = recs["full"][0]["pooled"]["high-stakes|uncut C"][CB]["joint"]["0.01"]["gap"]["delta_full"]
        n2["ci95"] = [n2["point"] + 0.01, n2["point"] + 0.05]  # a percentile interval that does not bracket its point (the figure draws it)
    r, t, d = build(m)
    assert r.returncode == 0, r.stderr
    assert len(t["probe_rows"]) == 12 and len(t["group_rows"]) == 6 and t["probe_rows"][0]["readout"] == "C" and t["probe_rows"][0]["lead"]
    assert list(t["gates"]["cohort_by_key"]) == ["all"] and len(t["gates"]["group_members"]) == 10
    nr = next(b for r_ in t["probe_rows"] for b in r_["bars"] if b["record_label"] == "harmfulness|uncut R")
    assert set(nr["value"]) == {"key", "not_readable"}  # no number survives for an unreadable row
    assert not any("diff" in k for r_ in t["probe_rows"] for b in r_["bars"] for k in b["value"])  # no difference field anywhere
    assert t["group_keys"]["excess"] == {"cut32k": "off_policy", "full": "off_policy_five_full"}
    assert t["build_labels"]["full"]["calibration"].startswith("full length (lead)") and t["lead_comparator"]["lead"] == "A"
    fl = t["disagreement_flag"]
    assert isinstance(fl["value"], bool) and set(fl["keys"]) == {"cut32k", "full"} and "not a test" in fl["label"]
    if os.environ.get("FL_TEST_FIGURE") == "1":
        (out,) = Path(d.name).glob("fl_vs_32k_table_syn_*.json")
        r = run(Path(d.name), "plot_fl_vs_32k.py", "--table", out.name, hashlib.sha256(out.read_bytes()).hexdigest(), "--stamp", "_syn")
        assert r.returncode == 0, r.stderr
    d.cleanup()


def test_panel_authority_and_lead_B():
    r, t, d = build(panels=("cut32k", "full"))
    d.cleanup()
    assert r.returncode == 0, r.stderr
    exc = [wv for g in t["group_rows"] if g["stat"] == "excess" for wv in g["windows"]]
    assert all(isinstance(wv["authority"], dict) and wv["authority"]["equals_stage_a_copy"] for wv in exc)

    def off(w, pan):  # PANEL's value no longer equals Stage A's copy x100
        pan["results"]["pooled"]["0.01"]["groups"]["off_policy|C"]["primary"]["mean_G_minus_mean_Y"]["point"] += 0.5
    expect_stop(None, "differs from the PANEL record", panels=("cut32k",), panel_edit=off)
    expect_stop(lambda plan, recs, ln: plan.update(_code="ed5be042e"), "not the cited code commit", panels=("cut32k",))
    expect_stop(None, "no key", panels=("cut32k",), panel_edit=lambda w, pan: pan["meta"].pop("git"))  # a cited code with no meta.git
    r, _t, d = build(lambda plan, recs, ln: plan.update(_code=""), panels=("cut32k",), panel_edit=lambda w, pan: pan["meta"].pop("git"))
    d.cleanup()
    assert r.returncode == 0, r.stderr  # no code cited: meta.git is not required
    for hc in (CB, "pooled_all"):  # PANEL's headline-combo ("pooled") branch, then its other_combos branch
        r, t, d = build(lead="B", panels=("cut32k", "full"), hc=hc)
        assert r.returncode == 0, (hc, r.stderr)
        sb = [g for g in t["group_rows"] if g["stat"] == "S (B)"]
        assert len(sb) == 2 and t["lead_comparator"]["lead"] == "B" and t["disagreement_flag"]["keys"]["full"].endswith('["S"]')
        v = sb[0]["windows"][0]["value"]
        want = fake_node(random.Random(9))  # the PANEL fixture's (B) S, in rate units before pts()
        assert abs(v["point"] - want["point"]) < 1e-12 and all(abs(x - y) < 1e-12 for x, y in zip(v["ci95"], want["ci95"]))
        assert v["units_source"] == "PANEL points / 100"  # the table keeps rate units; the figure multiplies by 100 once
        if os.environ.get("FL_TEST_FIGURE") == "1":
            (out,) = Path(d.name).glob("fl_vs_32k_table_syn_*.json")
            r = run(Path(d.name), "plot_fl_vs_32k.py", "--table", out.name, hashlib.sha256(out.read_bytes()).hexdigest(), "--stamp", "_synB")
            assert r.returncode == 0, r.stderr
        d.cleanup()

    def other_source(w, pan):  # the Stage A sha is still mentioned in the record, but the rows' npz source is another record
        src = pan["meta"]["config"]["npz"][0]
        pan["meta"]["config"]["note"] = f"computed beside Stage A record {src['record']['sha256']}"  # the true hash, kept elsewhere
        src["record"]["sha256"] = "0" * 64
        assert recs_sha(w) in json.dumps(pan)
    expect_stop(None, "is not read from this window's Stage A record", panels=("cut32k",), panel_edit=other_source)
    expect_stop(None, "excess_authority.full", lead="B")

    def a_off(w, pan):
        pan["coheadline"]["0.01"]["C"]["six"]["S_A_beside"]["point"] += 1.0
    expect_stop(None, "S_A_beside", lead="B", panels=("full",), panel_edit=a_off)


def test_record_read_settings_stop():
    expect_stop(None, "differs from the selection record", selection="B")
    expect_stop(lambda plan, recs, ln: recs["full"][0]["meta"].pop("calibration_label"), "no key")
    expect_stop(lambda plan, recs, ln: recs["full"][0]["meta"].update(calibration_label="registered epoch-1"), "pinned label")
    expect_stop(lambda plan, recs, ln: plan.update(lead_comparator="A"), "never set in the plan")
    sel_sha = hashlib.sha256(json.dumps({"lead": "A"}).encode()).hexdigest()  # the fixture's selection record, as write() stores it

    def ptr(**kw):  # a pointer that differs from the true one only in the given fields
        return lambda plan, recs, ln: recs["full"][0]["meta"]["estimator"].update(selection_record={"commit": "synthetic", "path": "results/syn_selection.json",
                                                                                                    "sha256": sel_sha} | kw)
    for kw in ({"sha256": "0" * 64}, {"path": "results/other.json"}, {"commit": "other"}, {"commit": ""}):
        expect_stop(ptr(**kw), "is not the record's meta.estimator.selection_record")
    for c in ("synth", "synthetic-and-more"):  # a short sha on either side matches
        r, _t, d = build(ptr(commit=c))
        d.cleanup()
        assert r.returncode == 0, (c, r.stderr)


def test_lead_combo_is_the_four_set_union():  # RULE item 33: pooled_all_origins by name, members exactly the four sets
    expect_stop(lambda plan, recs, ln: plan.update(combo="pooled_all"), "fixed settings")

    def three_sets(plan, recs, ln):  # a pooled row that silently dropped S_paper493 (not even withheld)
        for x in recs["full"][0]["pooled"].values():
            x[CB]["members"] = x[CB]["members"][:3]
    expect_stop(three_sets, "not exactly the four sets")


def test_empty_excess_group_stops():
    def m(plan, recs, ln):
        recs["full"][0]["panel_groups"][CB]["pooled"]["joint"]["0.01"]["off_policy_five_full|R"] = {"primary": {"rows": [], "unavailable": "no readable rows"}}
    expect_stop(m, "empty or unavailable")


def flag_for(s32, sfull):
    """The table's disagreement flag for given synthetic S C nodes (point, lo, hi) in the 32k and full-length records."""
    def m(plan, recs, ln):
        for w, (p, lo, hi) in (("cut32k", s32), ("full", sfull)):
            recs[w][0]["coheadline_signed"][CB]["joint"]["0.01"][f"{SIXKEY[w]}|C"]["S"].update(point=p, ci95=[lo, hi])
    r, t, d = build(m)
    d.cleanup()
    assert r.returncode == 0, r.stderr
    return t["disagreement_flag"]


def test_disagreement_flag_rule():  # 09 @ 2d61af835 §5: opposite signs, or exactly one interval excludes 0
    assert flag_for((0.02, -0.01, 0.05), (0.03, -0.01, 0.06))["value"] is False  # same sign, both include 0
    assert flag_for((0.02, -0.01, 0.05), (0.03, 0.01, 0.06))["value"] is True  # exactly one excludes 0
    f = flag_for((0.02, -0.01, 0.05), (-0.01, -0.04, 0.02))
    assert f["value"] is True and f["opposite_signs"] and not f["exactly_one_interval_excludes_0"]  # opposite signs alone
    assert flag_for((0.04, 0.01, 0.07), (0.05, 0.02, 0.08))["value"] is False  # both exclude 0, same sign
    assert flag_for((0.0, -0.02, 0.03), (-0.01, -0.03, 0.02))["value"] is False  # a zero point has no sign; both include 0
    assert flag_for((0.02, 0.0, 0.05), (0.03, -0.01, 0.06))["value"] is False  # an interval touching 0 includes it


def test_cohort_and_draw_gates_stop():
    def key_off(plan, recs, ln):
        ck = recs["full"][0]["rows"]["sol_a_uncut C"][SETS[1]]["cohort_cell_keys"]
        ck[3] = "t999_S_new"
        ck.sort()
    expect_stop(key_off, "same cohort and draws", "cohort_keys")

    def one_probe_both_windows(plan, recs, ln):  # only the cross-probe comparison can catch this
        for w, p in (("cut32k", "FIBS"), ("full", "FIBS|uncut")):
            for ro in "CR":
                ck = recs[w][0]["rows"][f"{p} {ro}"][SETS[0]]["cohort_cell_keys"]
                ck[0] = "t000_other"
                ck.sort()
    expect_stop(one_probe_both_windows, "same cohort and draws", "cohort_keys")

    def draw_off(plan, recs, ln):
        k = npzkey("FIBS|uncut R", CB, "joint", "0.01", SETS[0], "D")
        recs["full"][1][k] = recs["full"][1][k].copy()
        recs["full"][1][k][5] += 1.0  # a later draw differs; draw 0 and the shape still match
    expect_stop(draw_off, "same cohort and draws", "D[S_old (all-origins draws)]")

    def short_D(plan, recs, ln):
        k = npzkey("refusal C", CB, "joint", "0.01", SETS[1], "D")
        recs["cut32k"][1][k] = recs["cut32k"][1][k][:-1]
    expect_stop(short_D, "want draw 0 +")
    expect_stop(lambda plan, recs, ln: plan.update(_att={"records": ["only-one-sha"], "same_draw_streams": True}), "does not name")
    expect_stop(lambda plan, recs, ln: plan.update(_att={"records": [], "same_draw_streams": False}), "is not true")


def test_group_member_gates_stop():
    expect_stop(lambda plan, recs, ln: recs["full"][0]["panel_groups"][CB]["pooled"]["joint"]["0.01"]["off_policy_five_full|C"]["primary"]["rows"].pop(),
                "no silent subset")

    def unverified(plan, recs, ln):  # a group member that is not a gated bar (c_asis is no plan probe)
        recs["full"][0]["coheadline_signed"][CB]["joint"]["0.01"]["headline_six_full|C"]["rows"].append("c_asis C")
        next(g for g in plan["groups"] if g["stat"] == "S")["expect_members"]["full"].append("c_asis")
    expect_stop(unverified, "not bars that passed")


def test_schema_contract_length_placeholder_stop():
    def bad_ci(plan, recs, ln):
        n = recs["cut32k"][0]["pooled"]["FIBS C"][CB]["joint"]["0.01"]["gap"]["delta_full"]
        n["ci95"] = [n["ci95"][1], n["ci95"][0]]
    expect_stop(bad_ci, "summary-node schema")
    expect_stop(lambda plan, recs, ln: plan.update(mode="fixed"), "fixed settings")
    expect_stop(lambda plan, recs, ln: plan.update(paths={}), "pinned in code")
    expect_stop(lambda plan, recs, ln: plan.update(one_cohort_for_all_probes=False), "not configurable")
    expect_stop(lambda plan, recs, ln: next(g for g in plan["groups"] if g["stat"] == "excess").update(key={"cut32k": "x", "full": "y"}), "meta.window")
    expect_stop(lambda plan, recs, ln: ln["probes"].update(own_all_rs_registered={"status": "pending", "probe": "x", "why": "no cut field"}), "pending")
    expect_stop(lambda plan, recs, ln: plan.update(combo="<combo>"), "placeholders")


def test_resolver_row_checks():  # fit-qualified names pass; borrowing another readout's row or an unsigned table stops
    from resolve_training_lengths import table_row_checks

    table_row_checks("sol", {"rows": {"C": "Sol (a) (C; a_tight)", "R": "Sol (a) (R; a_tight)"}, "signed": {"commit": "abc"}})
    for bad in ({"rows": {"C": "FIBS (R)", "R": "FIBS (R)"}}, {"rows": {"C": "FIBS (C)", "R": "FIBS (R)"}, "signed": "<09's signature>"}):
        try:
            table_row_checks("x", bad)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {bad}")


def test_resolver_signature_chain():  # the r8 row must equal the row 09 signed, and each signature must name its own table (git reads patched)
    import resolve_training_lengths as RT

    rows = [{"probe": f"p ({ro})", "class_key": "i", "class": "(i) whole transcripts"} for ro in "CR"]
    tables = {"c8": {"rows": rows}, "c6": {"rows": copy.deepcopy(rows)}}
    texts = {"s8": "09 SIGNS r8 (@ c8, sha256 888888888888...)", "s6": "09 SIGNS r6 (@ c6, sha256 666666666666...)"}
    e = {"kind": "table_row", "probe": "p", "path": "t8.json", "commit": "c8", "sha256": "8" * 64, "rows": {"C": "p (C)", "R": "p (R)"},
         "expect_class_key": "i", "signed": {"commit": "s8", "file": "READ_LOG.md"},
         "row_signed_in": {"path": "t6.json", "commit": "c6", "sha256": "6" * 64, "signed": {"commit": "s6", "file": "READ_LOG.md"}}}
    old = RT.blob, RT.git
    RT.blob = lambda e_, name: tables[e_["commit"]]
    RT.git = lambda *a: texts[a[1]].encode()
    try:
        assert RT.resolve("p", e)["status"] == "resolved"
        for bad in ("row", "sig"):
            if bad == "row":
                tables["c6"]["rows"][0]["class"] = "changed after signing"
            else:
                tables["c6"]["rows"][0]["class"] = rows[0]["class"]
                texts["s6"] = "09 SIGNS something else"
            try:
                RT.resolve("p", e)
            except SystemExit:
                continue
            raise AssertionError(f"accepted a {bad} mismatch")
    finally:
        RT.blob, RT.git = old


def test_sheets_and_template():  # the committed sheets cover every length key the template uses
    plan = json.loads((HERE / "config" / "plan_TEMPLATE.json").read_text())
    used = {b["length"] for pr in plan["probes"] for b in pr["bars"]}
    for f in ("training_length_sheet.json", "training_length_sheet_r4.json", "training_length_sheet_r8.json"):
        sheet = json.loads((HERE / "config" / f).read_text())
        assert used <= set(sheet["probes"]), (f, used - set(sheet["probes"]))


if __name__ == "__main__":
    for f in (test_sheets_and_template, test_resolver_row_checks, test_resolver_signature_chain, test_positive, test_lead_combo_is_the_four_set_union, test_panel_authority_and_lead_B, test_record_read_settings_stop,
              test_empty_excess_group_stops, test_disagreement_flag_rule, test_cohort_and_draw_gates_stop, test_group_member_gates_stop,
              test_schema_contract_length_placeholder_stop):
        f()
        print("ok", f.__name__)
