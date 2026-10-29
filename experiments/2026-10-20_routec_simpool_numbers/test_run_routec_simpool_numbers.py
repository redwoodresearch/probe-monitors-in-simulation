"""Fixtures for run_routec_simpool_numbers.py: synthetic records only (no real record value is read), the pinned scripts imported from
their real blobs. Run: uv run pytest experiments/2026-10-20_routec_simpool_numbers/test_run_routec_simpool_numbers.py -q"""

import copy
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rsn", HERE / "run_routec_simpool_numbers.py")
rsn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rsn)
REPO = rsn.REPO


def pinned(name):
    c, p, want, _ = rsn.PINS[name]
    b = subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout
    assert rsn.sha(b) == want
    return rsn.load_pinned(name, b)


SF, RT, HD = pinned("solfree_script"), pinned("rates_script"), pinned("headers_script")
SOL, ASTRA, QWEN = SF.SOL, *SF.EXPECTED
CELLS = ("command|L48", "reply|L48")
BASE = [f"c{i:02d}" for i in range(10)]
PILOT = [f"p{i:02d}" for i in range(20)]
REAL = [f"r{i:02d}" for i in range(12)]


def path(cell):
    launch = "2026-10-14_140858_simh_pilot" if cell.startswith("p") else "2026-10-15_simh_q1"
    return f"{launch}/sim_honest@september/{cell}/run.eval"


def honest_record(sim_cells, seed):
    rng = np.random.default_rng(seed)
    rk, folds, mains = {}, {}, {}
    sim = {c: f"sim|{c}|e1|h" for c in sim_cells}
    a = {c: f"A|{c}|e1|h" for c in REAL}
    b = {c: f"B|{c}|e1|h" for c in REAL}
    zero = "sim|z00|e1|h"
    for c, k in [*sim.items(), *a.items(), *b.items(), ("z00", zero)]:
        rk[k] = path(c) + "#e0" if k.startswith("sim|") else f"real/{k.split('|')[0]}/{c}/run.eval"
        folds[k], mains[k] = int(c[1:]) % 5, c
    rec = {"run_keys": rk, "run_folds": folds, "run_main": mains, "cells": {},
           "zero_row_runs": {cell: {"n": 1, "runs": [{"key": rk[zero], "name": zero, "set": "S_newmains"}]} for cell in CELLS}}
    for cell in CELLS:
        probs = {}
        for name, pos, neg in (("simh_vs_b", sim, b), ("a_vs_b", a, b)):
            ks = sorted([*pos.values(), *neg.values()])
            s = {k: float(rng.normal()) for k in ks}
            y = np.array([0 if k.startswith("B|") else 1 for k in ks])
            au = SF.wauc(np.array([s[k] for k in ks]), y, np.ones(len(ks)))
            probs[name] = {"run_scores": s, "run_level": {"auroc": au}}
        rec["cells"][cell] = {"problems": probs}
    return rec


def header_pins():
    return {k: copy.deepcopy(v) for k, v in SF.HEADER_PINS.items()} | {"rows": [
        {"path": f"2026-10-14_160932_simh_cont/sim_honest@september/k{i:03d}/run.eval", "has_header": True,
         "models_any_tokens": {ASTRA: True, QWEN: True}} for i in range(SF.HEADER_PINS["n_files"])]}


def fixture():
    """10 base sim cells (c00, c01 Sol-using), 20 pilot cells (p00 Sol-using, p01 header unread), one zero-row sim run (z00)."""
    h171, hrc = honest_record(BASE, 1), honest_record(BASE + PILOT, 1)
    for c in BASE:  # the 10-16 record's shared runs are byte-identical in keys and paths
        assert h171["run_keys"][f"sim|{c}|e1|h"] == hrc["run_keys"][f"sim|{c}|e1|h"]
    hrc["cells"] = copy.deepcopy(hrc["cells"])
    for cell in CELLS:  # the a_vs_b control: same cohort and scores in both records
        hrc["cells"][cell]["problems"]["a_vs_b"] = copy.deepcopy(h171["cells"][cell]["problems"]["a_vs_b"])
    runs = {f"simh_311|{c}": {"roles": ["S"], "sets": ["S_new"], "world": "simh_311"} for c in BASE}
    runs |= {f"simh_pilot|{c}": {"roles": ["S"], "sets": ["S_old"], "world": "simh_pilot"} for c in PILOT}
    runs |= {f"honest_b|{c}": {"roles": ["B"], "sets": ["S_new"], "world": "honest_b"} for c in REAL}
    scan = {"rows": [{"parts": path(c).split("/"), "model_usage": {ASTRA: {"output_tokens": 5}, QWEN: {"output_tokens": 5}}
                      | ({SOL: {"output_tokens": 3}} if c in ("c00", "c01") else {})} for c in BASE]}
    x = {"catalogue": {"runs": runs},
         "record_A": {"meta": {"pooled_world": {"catalogue": {"sha256": rsn.PINS["catalogue"][3]},
                                                "per_label": {"lab C | pooled_all_origins": {"sizes": {"S": {"after": {"runs": 30}}}},
                                                              "lab C | paper_cells": {"sizes": {"S": {"after": {"runs": 7}}}}}}}},
         "honest_171": h171, "honest_routec": hrc, "scan_rows": scan, "usage": {"runs": []}, "cont_headers": header_pins(),
         "pilot_manifest": {"files": [{"path": path(c), "sha256": "0" * 64} for c in PILOT], "counts": {"files": 20, "usable": 20},
                            "counted_attempts_without_eval": {"simh": []}},  # the real manifest's shape @ 4af62db74 (review R2)
         "rates_171": {"headline_common_sets": {"sets": list(RT.COMMON), "Y_sim": {"failed": 3, "launched": 100, "rate": 3 / 100,
                                                                                   "by_category": {rsn.RATE_LIMIT: 1}}},
                       "sim": {"launch_source": {"manifests_rows": {"manifest_simh311_a.json": 5}}}},
         "chain_pilot_files": []}
    rows = [{"path": path(c), "has_header": True, "models_any_tokens": {ASTRA: True, QWEN: True} | ({SOL: True} if c == "p00" else {})}
            for c in PILOT if c != "p01"] + [{"path": path("p01"), "read_error": "FileNotFoundError"}]
    return x, rows


def test_load_pinned_gives_plain_functions_and_restores_modal():
    import modal

    assert sys.modules["modal"] is modal
    assert callable(HD.headers) and not hasattr(HD.headers, "remote") and callable(SF.sol_flags) and "S_old" not in RT.COMMON
    assert HD.VOL == rsn.VOL and HD.ROOT == rsn.HDR_ROOT


def test_build_on_fixture():
    x, rows = fixture()
    r = rsn.build(x, SF, RT, rows)
    assert set(r["gates"]) == {"G_TIE", "G_PILOT_SETS", "G_DELTA", "G_PILOT_PATHS", "G_POSITIVE_CONTROL", "G_SOL_MODEL_NAME", "G_RATES_SCOPE",
                               "G_PILOT_HEADERS", "G_POOL_V83_EQUALS_DETECTOR_171", "G_V83_PRINTED"}
    assert r["gates"]["G_POSITIVE_CONTROL"] == "PASS 8 of 8"  # 2 records x 2 cells x (simh_vs_b, a_vs_b)
    assert r["gates"]["G_PILOT_HEADERS"] == "read 19 of 20 (INCOMPLETE: the unread files are UNKNOWN, kept in n)"
    assert r["gates"]["G_POOL_V83_EQUALS_DETECTOR_171"] == "PASS"
    assert r["gates"]["G_V83_PRINTED"].startswith("FAIL 0 of 10; not reproduced: ")  # the fixture's v83 values are not the post's
    i1, i2, i4, i5 = (r["items"][k] for k in sorted(r["items"]))
    assert i2["pilot_attempts_S_old_apart"]["counted_attempts_without_eval"] == {"simh": 0}
    pool = i1["pool"]["routec"]
    assert (pool["n"], pool["sol_using"], pool["unknown"], pool["unknown_keys"]) == (30, 2 + 1, 1, ["simh_pilot|p01"])
    assert pool["shares"] == {"all_runs_denominator": 3 / 30, "unknown_counted_sol_upper": 4 / 30, "unknown_excluded": 3 / 29}
    assert (i1["pool"]["v83"]["n"], i1["pool"]["v83"]["sol_using"], i1["pool"]["v83"]["unknown"]) == (10, 2, 0)
    assert i1["pilot_header_scan"]["n_unknown"] == 1 and i1["detector_cells"]["routec"]["reply|L48"]["n"] == 30
    assert i1["flags"]["pool_v83_vs_routec"] == "changed" and i1["routec_pool_bounds_print_the_same"] is False  # 20% -> 10%; 10% vs 13%
    assert i2["flags"] == {"with_rate_limit": "unchanged", "without_rate_limit": "unchanged"} and i2["routec"]["launched"] == 100
    assert i4["routec"]["run_keys (all sim runs of the record)"]["detector_only_keys"] == ["sim|z00|e1|h"]
    assert i4["routec"]["run_keys (all sim runs of the record)"]["zero_row_cells_of_detector_only"] == {"sim|z00|e1|h": list(CELLS)}
    assert all(i4["routec"][c]["equal"] for c in CELLS) and i4["record_A_sizes_equal_pool"] and i4["flags"]["pool_n"] == "changed"
    assert i5["flags"] == {c: "unchanged" for c in CELLS} and all(i5["cohort_identical_by_key_path_fold_main"].values())
    assert sum(r["flag_counts"].values()) == len([v for it in r["items"].values() for v in it["flags"].values()])
    assert json.loads(json.dumps(r, sort_keys=True)) == r  # the container's serialisation round-trips (string keys, plain types)


def test_summary_carries_no_number_values():
    x, rows = fixture()
    s = rsn.summary(rsn.build(x, SF, RT, rows))

    def walk(o):
        if isinstance(o, float):
            raise AssertionError("a float in the laptop summary")
        for v in (o.values() if isinstance(o, dict) else o if isinstance(o, list) else []):
            walk(v)

    walk(s)
    assert set(s) == {"gates", "flag_counts", "items", "flag_keys"}


@pytest.mark.parametrize("sets", [["S_old", "S_new"], ["S_new"], []])
def test_pilot_not_s_old_only_stops(sets):
    x, rows = fixture()
    x["catalogue"]["runs"]["simh_pilot|p05"]["sets"] = sets
    with pytest.raises(rsn.Stop, match="G_PILOT_SETS"):
        rsn.build(x, SF, RT, rows)


def test_pilot_row_without_sets_stops():
    x, rows = fixture()
    del x["catalogue"]["runs"]["simh_pilot|p05"]["sets"]
    with pytest.raises(rsn.Stop, match="G_PILOT_SETS raised KeyError"):
        rsn.build(x, SF, RT, rows)


def test_pilot_count_not_20_stops():
    x, rows = fixture()
    del x["catalogue"]["runs"]["simh_pilot|p05"]
    with pytest.raises(rsn.Stop, match="G_PILOT_SETS"):
        rsn.build(x, SF, RT, rows)


def test_extra_non_pilot_sim_run_stops():
    x, rows = fixture()
    hrc = x["honest_routec"]
    hrc["run_keys"]["sim|c99|e1|h"], hrc["run_folds"]["sim|c99|e1|h"], hrc["run_main"]["sim|c99|e1|h"] = path("c99"), 0, "c99"
    with pytest.raises(rsn.Stop, match="G_DELTA"):
        rsn.build(x, SF, RT, rows)


def test_repathed_shared_run_stops():
    x, rows = fixture()
    x["honest_routec"]["run_keys"]["sim|c03|e1|h"] = "elsewhere/run.eval"
    with pytest.raises(rsn.Stop, match="G_DELTA"):
        rsn.build(x, SF, RT, rows)


def test_header_rows_not_the_manifest_stop():
    x, rows = fixture()
    with pytest.raises(rsn.Stop, match="G_PILOT_PATHS"):
        rsn.build(x, SF, RT, rows[:-1])


def test_positive_control_mismatch_stops():
    x, rows = fixture()
    x["honest_routec"]["cells"]["reply|L48"]["problems"]["simh_vs_b"]["run_level"]["auroc"] += 0.01
    with pytest.raises(rsn.Stop, match="positive control"):
        rsn.build(x, SF, RT, rows)


def test_catalogue_not_record_As_stops():
    x, rows = fixture()
    x["record_A"]["meta"]["pooled_world"]["catalogue"]["sha256"] = "0" * 64
    with pytest.raises(rsn.Stop, match="G_TIE"):
        rsn.build(x, SF, RT, rows)


def test_header_record_not_item_c_pins_stops():
    x, rows = fixture()
    x["cont_headers"]["rows"].pop()
    with pytest.raises(rsn.Stop, match="149-file header record"):
        rsn.build(x, SF, RT, rows)


def test_pilot_named_in_chain_manifests_stops():
    x, rows = fixture()
    x["chain_pilot_files"] = ["1df288e61:experiments/2026-10-14_q3_overnight_chain/config/manifest_x.json"]
    with pytest.raises(rsn.Stop, match="item 2"):
        rsn.build(x, SF, RT, rows)


def test_row5_refolded_control_is_not_confirmed():
    x, rows = fixture()
    x["honest_routec"]["run_folds"]["A|r03|e1|h"] += 1
    r = rsn.build(x, SF, RT, rows)["items"]["5_table7_row5_control"]
    assert r["flags"] == {c: "NOT CONFIRMED (cohort differs)" for c in CELLS}


def test_row5_missing_fold_stops():
    x, rows = fixture()
    del x["honest_routec"]["run_folds"]["B|r03|e1|h"]
    with pytest.raises(rsn.Stop, match="item 5 cohort check raised KeyError"):
        rsn.build(x, SF, RT, rows)


def test_decode_rejects_a_changed_input():
    blobs = {n: b"" for n in rsn.PINS}
    with pytest.raises(rsn.Stop, match="differs from its pin"):
        rsn.decode(blobs)
    assert gzip.decompress(gzip.compress(b"{}")) == b"{}" and json.loads(b"{}") == {}


def test_half_up_rounding():
    assert rsn.rnd(0.525, 2) == "0.53" and rsn.rnd(5.5, 0) == "6" and rsn.rnd(4.25, 1) == "4.3" and rsn.rnd(None, 2) is None
    assert rsn.flag(5.9, 5.76, 0) == "unchanged" and rsn.flag(5.4, 5.6, 0) == "changed" and rsn.flag(None, 1.0, 0) == "not comparable"


def test_container_import_at_root_and_stdlib_only_imports():
    """Modal mounts only the entrypoint at /root (modal 1.6.1): the script must import there, with no repo-relative module."""
    import ast

    src = (HERE / "run_routec_simpool_numbers.py").read_text()
    ns = {"__file__": "/root/run_routec_simpool_numbers.py", "__name__": "rsn_root"}
    exec(compile(src, "/root/run_routec_simpool_numbers.py", "exec"), ns)
    assert ns["REPO"] == Path("/root") and ns["HERE"] == Path("/root")
    top = {a.name.split(".")[0] for n in ast.parse(src).body if isinstance(n, ast.Import) for a in n.names}
    top |= {n.module.split(".")[0] for n in ast.parse(src).body if isinstance(n, ast.ImportFrom)}
    assert top <= {"gzip", "hashlib", "json", "re", "subprocess", "sys", "types", "decimal", "pathlib", "modal"}


def test_v83_printed_gate_passes_when_the_v83_values_match(monkeypatch):
    """Review O3, positive path: with the fixture's own v83 values as the 'printed' ones, every check passes."""
    x, rows = fixture()
    r0 = rsn.build(x, SF, RT, rows)["items"]["5_table7_row5_control"]["rounded"]["v83"]
    if r0["command|L48"] != r0["reply|L48"]:  # make both cells print alike, as in the post ("0.52 on either")
        x["honest_171"]["cells"]["reply|L48"]["problems"]["a_vs_b"] = copy.deepcopy(x["honest_171"]["cells"]["command|L48"]["problems"]["a_vs_b"])
        x["honest_routec"]["cells"]["reply|L48"]["problems"]["a_vs_b"] = copy.deepcopy(x["honest_171"]["cells"]["command|L48"]["problems"]["a_vs_b"])
    monkeypatch.setattr(rsn, "PRINTED_V83", {"item1_percent": "20", "item1_counts_sol_n": (2, 10), "item2_percent": "3.0",
                                             "item2_percent_without_rate_limit": "2.0", "item5_auroc": r0["command|L48"]})
    assert rsn.build(x, SF, RT, rows)["gates"]["G_V83_PRINTED"] == "PASS 10 of 10"


def test_no_sol_using_run_stops():
    x, rows = fixture()
    for r in x["scan_rows"]["rows"]:
        r["model_usage"].pop(SOL, None)
    with pytest.raises(rsn.Stop, match="no Sol-using"):
        rsn.build(x, SF, RT, rows)


def test_rates_manifest_naming_pilot_stops():
    x, rows = fixture()
    x["rates_171"]["sim"]["launch_source"]["manifests_rows"]["manifest_simh_pilot_x.json"] = 1
    with pytest.raises(rsn.Stop, match="item 2: a pilot launch row"):
        rsn.build(x, SF, RT, rows)


def test_rates_headline_sets_not_common_stops():
    x, rows = fixture()
    x["rates_171"]["headline_common_sets"]["sets"] = ["S_new", "S_newmains"]
    with pytest.raises(rsn.Stop, match="headline sets are not"):
        rsn.build(x, SF, RT, rows)


def test_rates_rate_not_failed_over_launched_stops():
    x, rows = fixture()
    x["rates_171"]["headline_common_sets"]["Y_sim"]["rate"] = 0.031
    with pytest.raises(rsn.Stop, match="rate != failed / launched"):
        rsn.build(x, SF, RT, rows)


@pytest.mark.parametrize("cawe", [None, [], {"simh": None}])
def test_pilot_attempts_field_not_an_object_of_lists_stops(cawe):
    x, rows = fixture()
    x["pilot_manifest"]["counted_attempts_without_eval"] = cawe
    with pytest.raises(rsn.Stop, match="not an object of lists"):
        rsn.build(x, SF, RT, rows)


def test_header_module_on_another_volume_or_root_stops():
    import types as t

    rsn.check_header_module(t.SimpleNamespace(VOL=rsn.VOL, ROOT=rsn.HDR_ROOT))
    for bad in (t.SimpleNamespace(VOL="other", ROOT=rsn.HDR_ROOT), t.SimpleNamespace(VOL=rsn.VOL, ROOT="/a25/elsewhere")):
        with pytest.raises(rsn.Stop, match="Volume or root"):
            rsn.check_header_module(bad)


def test_row5_same_cohort_new_scores_is_changed():
    """Same cohort by key, path, fold and main, but route C's A scores move up: the 2 dp AUROC differs -> 'changed'."""
    x, rows = fixture()
    for cell in CELLS:
        p = x["honest_routec"]["cells"][cell]["problems"]["a_vs_b"]
        p["run_scores"] = {k: v + (3.0 if k.startswith("A|") else 0.0) for k, v in p["run_scores"].items()}
        ks = sorted(p["run_scores"])
        p["run_level"]["auroc"] = SF.wauc(np.array([p["run_scores"][k] for k in ks]), np.array([int(k.startswith("A|")) for k in ks]),
                                          np.ones(len(ks)))
    r = rsn.build(x, SF, RT, rows)["items"]["5_table7_row5_control"]
    assert all(r["cohort_identical_by_key_path_fold_main"].values()) and r["flags"] == {c: "changed" for c in CELLS}


def test_row5_auroc_not_reproduced_stops():
    x, rows = fixture()
    x["honest_routec"]["cells"]["reply|L48"]["problems"]["a_vs_b"]["run_level"]["auroc"] += 0.01
    with pytest.raises(rsn.Stop, match="positive control failed in \\['routec \\| reply\\|L48 \\| a_vs_b'\\]"):
        rsn.build(x, SF, RT, rows)


def test_pool_v83_not_the_10_16_detector_class_is_stated():
    x, rows = fixture()
    x["catalogue"]["runs"]["simh_311|c50"] = {"roles": ["S"], "sets": ["S_new"], "world": "simh_311"}
    r = rsn.build(x, SF, RT, rows)
    assert r["gates"]["G_POOL_V83_EQUALS_DETECTOR_171"].startswith("FAIL in cells ")
    assert r["items"]["1_sim_footnote"]["pool"]["v83"]["unknown_keys"] == ["simh_311|c50"]


def test_out_name_pattern():
    assert rsn.OUT_NAME.fullmatch("routec_simpool_numbers_2026-10-20_021500.json")
    assert not rsn.OUT_NAME.fullmatch("../routec_simpool_numbers_2026-10-20_021500.json")
    assert not rsn.OUT_NAME.fullmatch("routec_simpool_numbers_2026-10-20_021500.json.tmp")
