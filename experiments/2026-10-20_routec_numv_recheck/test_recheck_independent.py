"""Synthetic checks for run_recheck_independent.py (no real values, no Modal calls).

uv run --no-project --with modal==1.6.1 --with pytest==9.1.1 pytest -q <this file>
"""

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "run_recheck_independent.py"
spec = importlib.util.spec_from_file_location("rri", SCRIPT)
rri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rri)

ROWS = {  # the two batches' selected rows, ids only (from recheck.rows of each result file)
    "rc2": ["18.Y.fixed.harmfulness|uncut C@0.0133352#2", "20.catch.joint.sol_a_uncut R@0.0002.sim#2",
            "20.catch.joint.high-stakes|uncut C@0.0002.sim#0", "18.delta_full.fixed.sol_a_uncut R@0.00237137#1",
            "19.real_units_above_tau_B.sol_a_uncut C@0.001#0", "18.Y.joint.harmfulness|uncut C@0.00421697#1",
            "18.Y.fixed.high-stakes|uncut R@0.00562341#1", "20.catch.fixed.own_all_rs_uncut_L48 C@0.0002.real_mean#0",
            "20.delta_full.joint.own_all_rs_uncut_L48 C@0.0005#1", "20.catch.fixed.harmfulness|uncut C@0.0005.sim#2",
            "18.Y.fixed.sol_a_uncut C@0.0177828#1", "18.delta_full.fixed.harmfulness|uncut R@0.0316228#2",
            "18.delta_full.joint.FIBS|uncut C@0.00749894#1", "18.delta_full.joint.FIBS|uncut R@0.00237137#0",
            "18.delta_full.joint.refusal|uncut R@0.00749894#0"],
    "rc1b": ["25.plant.aggregate.0.01.C.-8.S#1", "24.r2.lead.high-stakes|uncut C.main#1",
             "21.kappa.own_all_rs_uncut_L48 R.joint.kappa_sim_vs_run1#0",
             "22.kappa_five.C.fixed.mean_kappa_sim_vs_run1#2", "25.plant.count.0.01.-3.C#3",
             "21.kappa.sol_a_uncut R.fixed.kappa_sim_minus_run2#1", "14.after_runs.refusal|uncut R.S#0"],
    "fig": ["values[80]|ci95[1]", "values[150]|point", "values[21]|ci95[1]", "values[124]|ci95[1]",
            "values[180]|ci95[1]", "values[68]|ci95[1]", "values[16]|value"],
    "scatter": [f"src|scatter_points/own_all_rs_uncut_L48 C/points/{n}/{a}" for n, a in [
        (267, "run2"), (932, "run1"), (1164, "run1"), (1038, "run1"), (505, "run2"), (323, "run1"), (430, "sim"),
        (780, "sim"), (958, "sim"), (971, "run2")]],
}


def test_imports_as_root():
    """Modal runs the entry file at /root: module-level code must not depend on the repo path."""
    ns = {"__file__": "/root/run_recheck_independent.py", "__name__": "rri_root"}
    exec(compile(SCRIPT.read_text(), "/root/run_recheck_independent.py", "exec"), ns)
    assert len(ns["APP_NAME"]) <= 63


def test_every_row_id_routes():
    for sid in ROWS["rc2"] + ROWS["rc1b"]:
        key = sid.rsplit("#", 1)[0]
        assert rri.SLOT_ID.match(sid)
        ints = [rx.match(key) for rx in (rri.UNITS19_ID, rri.PLANT_COUNT_ID, rri.AFTER_RUNS_ID)]
        assert sum(bool(m) for m in ints) <= 1
    assert rri.UNITS19_ID.match("19.real_units_above_tau_B.sol_a_uncut C@0.001")["pp"] == "sol_a_uncut C"
    assert rri.PLANT_COUNT_ID.match("25.plant.count.0.01.-3.C").groupdict() == {"level": "0.01", "offset": "-3",
                                                                                 "pool": "C"}
    assert rri.AFTER_RUNS_ID.match("14.after_runs.refusal|uncut R.S")["pp"] == "refusal|uncut R"
    assert all(rri.FIG_VALUE_ID.match(s) for s in ROWS["fig"])
    assert all(rri.SCATTER_ID.match(s) for s in ROWS["scatter"])


@pytest.mark.parametrize("token,x,want", [
    ("+1.2", 0.012, (True, "slot.summ_x100")), ("0.45", 0.4512, (True, "slot.summ_x1")),
    ("12.3%", 0.1234, (True, "slot.summ_pct_x100")), ("−0.5", -0.005, (True, "slot.summ_x100")),
    ("+1.3", 0.012, (False, "slot.summ_x1_or_x100")), ("12.4%", 0.1234, (False, "slot.summ_pct_x100")),
])
def test_summ_compare(token, x, want):
    assert rri.summ_compare(token, x) == want


def test_every_summ_id_has_one_family():
    for art in ("rc2", "rc1b"):
        for sid in ROWS[art]:
            key = sid.rsplit("#", 1)[0]
            n = sum(bool(rx.match(key)) for a, rx, _, _ in rri.SUMM_FAMILIES if a == art)
            ints = any(rx.match(key) for rx in (rri.UNITS19_ID, rri.PLANT_COUNT_ID, rri.AFTER_RUNS_ID))
            assert n + ints == 1, sid


LEAF = {"point": 0.0123, "ci95": [-0.0021, 0.0333]}


def synthetic():
    node = lambda gap: {"gap": {"delta_full": gap}, "panel": {"pooled": {"Y": dict(LEAF)}}}  # noqa: E731
    other = {"point": 0.5, "ci95": [0.4, 0.6]}
    recs = {n: {} for n in ("B", "R37", "A", "A_group_excess", "KAPPA", "KAPPA_EXTRA", "ANCHOR", "PLANT", "PERTASK")}
    # the slot's own leaf, plus the SAME triple under the other pool (C vs R) and under another level
    recs["B"] = {"pooled": {"p C": {"pooled_all_origins": {"fixed": {"0.01": node(dict(LEAF)), "0.05": node(LEAF)}}},
                            "p R": {"pooled_all_origins": {"fixed": {"0.01": node(other)}}}}}
    recs["PLANT"] = {"counts": {"0.01": {"-3": {"C": {"detected": 1, "cases": 6, "not_detected": 5, "unavailable": 0}}}}}
    arts = {"rc2": {"slots": {"18.delta_full.fixed.p C@0.01": "+1.2 (−0.2 to +3.3)",
                              "18.delta_full.fixed.p R@0.01": "+1.2 (−0.2 to +3.3)"},
                    "full": {"18.delta_full.fixed.p C@0.01": "0.0123 [-0.0021, 0.0333]",
                             "18.delta_full.fixed.p R@0.01": "0.0123 [-0.0021, 0.0333]"}},
            "rc1b": {"slots": {"25.plant.count.0.01.-3.C": "1/6 detected, 5 missed, 0 unavailable (missed: a C)"},
                     "full": {"25.plant.count.0.01.-3.C": "1/6; missed 5; unavailable 0"}}}
    return arts, recs


def test_slot_rows_and_controls():
    arts, recs = synthetic()
    details = []
    for i, sid in enumerate(["18.delta_full.fixed.p C@0.01#1", "25.plant.count.0.01.-3.C#3"]):
        det = {"i": i, "rule": "none", "equal": False, "error": None}
        rri.check_slot(det, "rc2" if sid.startswith("18") else "rc1b", sid, arts, recs)
        details.append(det)
    assert [d["equal"] for d in details] == [True, True]
    assert details[0]["rule"] == "slot.summ_x100"
    assert (details[0]["n_candidates"], details[0]["n_matches"]) == (1, 1)
    assert details[1]["rule"] == "slot.int_exact_def"
    pc, nc = rri.controls(details)
    assert pc[0]["unequal"] is True and pc[1]["error"] == "no_row"
    assert nc == {"n": 2, "n_equal": 2}


def test_value_elsewhere_in_record_does_not_pass():
    """2e's risk: the slot shows a triple that exists in the record, but under another pool / level -> no_match."""
    arts, recs = synthetic()
    with pytest.raises(rri.RowError, match="no_match"):
        rri.check_slot({"i": 0}, "rc2", "18.delta_full.fixed.p R@0.01#1", arts, recs)
    arts["rc2"]["full"]["18.delta_full.fixed.p C@0.01"] = "0.0124 [-0.0021, 0.0333]"  # full not at the slot's path
    with pytest.raises(rri.RowError, match="no_match"):
        rri.check_slot({"i": 0}, "rc2", "18.delta_full.fixed.p C@0.01#1", arts, recs)


def test_star_template_multi_match_and_pooling_mismatch(monkeypatch):
    """No live template has a '*' now, but candidates() supports one: a test-only '*' family must still give
    multi_match on two matching fillings and pooling_mismatch on a unique match outside pooled_all_origins."""
    import re

    star = ("rc1b", re.compile(r"^99\.star\.(?P<pp>.+ [CR])$"), "KAPPA", ["kappa", "{pp}", "combos", "*", "joint", "q"])
    monkeypatch.setattr(rri, "SUMM_FAMILIES", rri.SUMM_FAMILIES + [star])
    arts, recs = synthetic()
    other = {"point": 0.5, "ci95": [0.4, 0.6]}
    combos = {"paper_cells": {"joint": {"q": dict(LEAF)}}, "pooled_all_origins": {"joint": {"q": dict(LEAF)}}}
    recs["KAPPA"] = {"kappa": {"p R": {"combos": combos}}}
    arts["rc1b"]["slots"]["99.star.p R"] = "0.01 (−0.00 to 0.03)"
    arts["rc1b"]["full"]["99.star.p R"] = "0.0123 [-0.0021, 0.0333]"
    det = {"i": 0}
    with pytest.raises(rri.RowError, match="multi_match"):
        rri.check_slot(det, "rc1b", "99.star.p R#0", arts, recs)
    assert (det["n_candidates"], det["n_matches"]) == (2, 2)
    combos["pooled_all_origins"]["joint"]["q"] = other
    det = {"i": 0}
    with pytest.raises(rri.RowError, match="pooling_mismatch"):
        rri.check_slot(det, "rc1b", "99.star.p R#0", arts, recs)
    assert det["match_pooled_all_origins"] is False


def test_definition_templates_y_r2lead_afterruns():
    """The three families fixed from 09's definition lines: Y = gap.run2_minus_run1_at_tauB (rc2 L45), r2 lead in
    union_stream (rc1b L170), after_runs at the pooled_all_origins label in pooled A (rc1b L40, thousands separators)."""
    arts, recs = synthetic()
    node = recs["B"]["pooled"]["p C"]["pooled_all_origins"]["fixed"]["0.01"]
    node["gap"]["run2_minus_run1_at_tauB"] = dict(LEAF)
    arts["rc2"]["slots"]["18.Y.fixed.p C@0.01"] = "+1.2 (−0.2 to +3.3)"
    arts["rc2"]["full"]["18.Y.fixed.p C@0.01"] = "0.0123 [-0.0021, 0.0333]"
    det = {"i": 0}
    rri.check_slot(det, "rc2", "18.Y.fixed.p C@0.01#0", arts, recs)
    assert det["equal"] is True and det["template"][-2:] == ["gap", "run2_minus_run1_at_tauB"]
    recs["ANCHOR"] = {"union_stream": {"lead": {"p C": {"sets": {
        "pooled_all_origins (union stream)": {"r2": {"main": {"point": 0.81, "ci95": [0.7, 0.9]}}},
        "paper_cells (union stream)": {"r2": {"main": dict(LEAF)}}}}}}}
    arts["rc1b"]["slots"]["24.r2.lead.p C.main"] = "0.81 (0.70 to 0.90)"
    arts["rc1b"]["full"]["24.r2.lead.p C.main"] = "0.81 [0.7, 0.9]"
    det = {"i": 0}
    rri.check_slot(det, "rc1b", "24.r2.lead.p C.main#1", arts, recs)
    assert det["equal"] is True and det["n_candidates"] == 1 and det["match_pooled_all_origins"] is True
    recs["A"] = {"meta": {"pooled_world": {"per_label": {
        "p R | pooled_all_origins": {"sizes": {"S": {"after": {"runs": 1234}}}},
        "p R | paper_cells": {"sizes": {"S": {"after": {"runs": 99}}}}}}}}
    arts["rc1b"]["slots"]["14.after_runs.p R.S"] = "1,234"
    arts["rc1b"]["full"]["14.after_runs.p R.S"] = "1234"
    det = {"i": 0}
    rri.check_slot(det, "rc1b", "14.after_runs.p R.S#0", arts, recs)
    assert det["equal"] is True and det["rule"] == "slot.int_exact_def"


def test_definition_guard_reads_rows_by_key():
    """The pinned slot-key NOTEs hold every definition substring (rows found by key); an edited row fails."""
    import subprocess

    data = {n: subprocess.run(["git", "show", f"{rri.PINS[n][0]}:{rri.PINS[n][1]}"], cwd=HERE, check=True,
                              capture_output=True).stdout for n in ("slotnote_rc2", "slotnote_rc1b")}
    assert rri.definitions_ok(data) is True
    bad = dict(data)
    bad["slotnote_rc2"] = data["slotnote_rc2"].replace(b"run2_minus_run1_at_tauB}", b"panel.Y}")
    assert rri.definitions_ok(bad) is False


def test_fig_row_exact_and_positive_control():
    _, recs = synthetic()
    src = {"inputs": {"rec": {"sha256": rri.PINS["B"][2]}},
           "values": [{"input": "rec", "key": ["pooled", "p C", "pooled_all_origins", "fixed", "0.01", "gap",
                                               "delta_full"], "point": 0.0123, "ci95": [-0.0021, 0.0333]}]}
    det = {"i": 0, "rule": "none", "equal": False, "error": None}
    rri.check_fig(det, "fig4:fig4_pooled_fig4_routecB", "values[0]|ci95[1]", {"fig4_src": src}, recs)
    assert det["equal"] is True
    pc, nc = rri.controls([det])
    assert pc[1]["unequal"] is True and nc == {"n": 1, "n_equal": 1}
    src["values"][0]["ci95"][1] = 0.03330000000000001
    det = {"i": 0, "rule": "none", "equal": False, "error": None}
    rri.check_fig(det, "fig4:fig4_pooled_fig4_routecB", "values[0]|ci95[1]", {"fig4_src": src}, recs)
    assert det["equal"] is False


def test_pins_resolve_by_sha():
    """Dry-resolve every git pin the local entrypoint reads: bytes hashed only, nothing parsed."""
    import gzip
    import hashlib
    import subprocess

    for name, (commit, path, want) in rri.PINS.items():
        raw = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=HERE, check=True, capture_output=True).stdout
        plain = gzip.decompress(raw) if path.endswith(".gz") else raw
        assert hashlib.sha256(plain).hexdigest() == want, name


@pytest.mark.parametrize("fixture,want", [
    ('[{"app_id":"ap-1","description":"APP","state":"ephemeral (detached)","tasks":"1"}]', "1"),  # live row
    ('[{"app_id":"ap-1","description":"APP","state":"stopped","tasks":"0"},'
     ' {"app_id":"ap-2","description":"other","state":"deployed","tasks":"1"}]', "0"),  # stopped + other app
    ('[{"App ID":"ap-1","Description":"APP","State":"ephemeral"}]', '"SCHEMA"'),  # upper-case keys
    ('[{"app_id":"ap-1","description":"APP"}]', '"SCHEMA"'),  # missing state
    ("[]", '"SCHEMA"'),  # empty
    ('{"description":"APP","state":"ephemeral"}', '"SCHEMA"'),  # not an array
])
def test_live_app_gate(fixture, want):
    import subprocess

    out = subprocess.run(["jq", "--arg", "a", "APP", "-f", str(HERE / "live_app_gate.jq")], input=fixture.encode(),
                         capture_output=True, check=True).stdout.decode().strip()
    assert out == want


def test_live_app_gate_garbage_fails():
    import subprocess

    r = subprocess.run(["jq", "--arg", "a", "APP", "-f", str(HERE / "live_app_gate.jq")], input=b"not json",
                       capture_output=True)
    assert r.returncode != 0


def test_kappa_per_s_definition_value_fails():
    """REQUIRED-K: row 21 defines kappa as combos.pooled_all_origins; an artifact showing a per-S-definition
    (origins) kappa, or the paper_cells combo, must not pass."""
    arts, recs = synthetic()
    other = {"point": 0.5, "ci95": [0.4, 0.6]}
    s_old = "S_old (all-origins draws) [pooled: pooled_all_origins]"
    recs["KAPPA"] = {"kappa": {"p R": {
        "combos": {"pooled_all_origins": {"joint": {"kappa_sim_vs_run1": other}},
                   "paper_cells": {"joint": {"kappa_sim_vs_run1": dict(LEAF)}}},
        "origins": {s_old: {"joint": {"kappa_sim_vs_run1": dict(LEAF)}}}}}}
    key = "21.kappa.p R.joint.kappa_sim_vs_run1"
    arts["rc1b"]["slots"][key] = "0.01 (−0.00 to 0.03)"
    arts["rc1b"]["full"][key] = "0.0123 [-0.0021, 0.0333]"
    det = {"i": 0}
    with pytest.raises(rri.RowError, match="no_match"):
        rri.check_slot(det, "rc1b", key + "#0", arts, recs)
    assert (det["n_candidates"], det["n_matches"]) == (1, 0)
    recs["KAPPA"]["kappa"]["p R"]["combos"]["pooled_all_origins"]["joint"]["kappa_sim_vs_run1"] = dict(LEAF)
    det = {"i": 0}
    rri.check_slot(det, "rc1b", key + "#0", arts, recs)
    assert det["equal"] is True and det["match_pooled_all_origins"] is True


def test_pc_invalid_at_max_float():
    """O-e: at x = max float the bump overflows or no-ops and nextafter gives inf -> pc_invalid, never a pass."""
    import sys

    x = sys.float_info.max
    det = {"i": 0, "rule": "fig.exact_float", "equal": True, "_x": x, "_printed": x}
    pc, _ = rri.controls([det])
    assert pc[1] == {"kind": "fig.", "i": 0, "unequal": False, "error": "pc_invalid"}


def test_fig_positive_control_on_17_digit_values():
    """Run 023801 bug: +1 in the last digit of a 17-digit repr can round back to the same double. The control must
    fall back to the next double and never report a void perturbation as a passing control."""
    import math
    from decimal import Decimal

    xs = [0.012345678901234567 + i * 1.7e-18 for i in range(2000)]
    void = [x for x in xs if float(Decimal(repr(x)) + rri.bump_last_digit_of(repr(x))) == x]
    assert void, "fixture must contain a 17-digit value where the decimal bump alone is void"
    x = void[0]
    assert len(repr(x).lstrip("0.").rstrip("0")) >= 16
    det = {"i": 0, "rule": "fig.exact_float", "equal": True, "_x": x, "_printed": x}
    pc, nc = rri.controls([det])
    assert pc[1] == {"kind": "fig.", "i": 0, "unequal": True, "error": None}
    assert float(det["positive_control"]["perturbed_source"]) == math.nextafter(x, math.inf)
    assert float(det["positive_control"]["perturbed_source"]) != x  # the planted value differs as a double (09)
    assert nc == {"n": 1, "n_equal": 1}


def test_cross_check_reports_paths_but_decides_nothing():
    arts, recs = synthetic()
    xindex = {n: rri.triple_index(r) for n, r in recs.items()}
    det = {"i": 0}
    rri.check_slot(det, "rc2", "18.delta_full.fixed.p C@0.01#1", arts, recs, xindex)
    assert det["equal"] is True and det["template_match_in_cross_check"] is True
    assert len(det["record_match_paths"]["B"]) == 5  # gap + panel.pooled.Y at p C 0.01 and 0.05, panel.pooled.Y at p R
    assert det["record_match_counts"] == {"B": 5}  # the reply carries counts only (O-d)
    det = {"i": 0}
    with pytest.raises(rri.RowError, match="no_match"):  # triple elsewhere in B: reported, still no_match
        rri.check_slot(det, "rc2", "18.delta_full.fixed.p R@0.01#1", arts, recs, xindex)
    assert len(det["record_match_paths"]["B"]) == 5
