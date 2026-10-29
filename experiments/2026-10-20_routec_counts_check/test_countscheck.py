"""COUNTSCHECK tests: synthetic fixtures only (no real record is read). Positive world, negatives that really differ, STOPs,
the jq filters' value stripping, the container share compare, and the Modal entry's import at /root."""

import copy
import hashlib
import importlib.util
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
import countscheck as cc  # noqa: E402

SENT = 0.987654321  # a "value" leaf the filters must drop
ENTRY = HERE / "run_routec_counts_check.py"


def load_entry():
    spec = importlib.util.spec_from_file_location("run_routec_counts_check", ENTRY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- synthetic world

CELL_SET = {"t1": "S_old", "t2": "S_old", "t6": "S_old", "t7": "S_old", "t3": "S_new", "t4": "S_newmains", "t5": "S_paper493"}
PRIOR = ["A|t3|e1|a1", "A|t3|e2|a2", "A|t1|e1|a3", "B|t3|e1|b1", "B|t4|e1|b2", "B|t1|e2|b3", "B|t6|e1|b6",
         "sim|t1|e1|s1", "sim|t2|e1|s2", "sim|t7|e1|s7", "sim|t3|e1|s3", "sim|t4|e1|s4", "sim|t5|e1|s5"]
PILOT, ZERO = "sim|t6|e1|s6", "sim|t4|e1|s4"
FOLD = {"t1": 0, "t2": 1, "t3": 2, "t4": 3, "t5": 4, "t6": 0, "t7": 1}


def path(k):
    c, cell, _, h = k.split("|")
    return f"L_{c}/arm/{cell}/{h}.eval"


def ext(keys, routec):
    run_keys = {k: path(k) + "#e0" for k in keys}
    rpr = {k: 3 for k in keys if k != ZERO}
    cells = {}
    for cell in cc.CELLS:
        probs = {}
        for p, (c1, c0) in cc.PROBLEMS.items():
            sk = sorted(k for k in rpr if k.split("|")[0] in (c1, c0))
            probs[p] = {"n_runs": {"class1": sum(k.startswith(c1 + "|") for k in sk), "class0": sum(k.startswith(c0 + "|") for k in sk)},
                        "observed_complete": True, "null_complete": True, "observed_certified": True, "null_certified": True,
                        "run_score_keys": sk}
        cells[cell] = {"rows": 3 * len(rpr), "rows_per_run": dict(rpr), "runs_without_rows": [run_keys[ZERO]],
                       "certified_per_column": [[True, True]], "problems": probs}
    pcse = {c: {s: {"e1": 0, "e2": 0} for s in cc.SETS} for c in cc.CLASSES}
    for k in keys:
        c, cell, e, _ = k.split("|")
        pcse[c][CELL_SET[cell]][e] += 1
    zr = {"n": 1, "by_class": {"A": 0, "B": 0, "sim": 1}, "by_set": {"S_old": 0, "S_new": 0, "S_newmains": 1, "S_paper493": 0},
          "runs": [{"key": run_keys[ZERO], "name": ZERO, "set": "S_newmains"}]}
    return {
        "run_keys": run_keys, "run_set": {k: CELL_SET[k.split("|")[1]] for k in keys}, "run_epoch": {k: int(k.split("|")[2][1:]) for k in keys},
        "run_cell": {k: k.split("|")[1] for k in keys}, "run_main": {k: "m_" + k.split("|")[1] for k in keys},
        "run_folds": {k: FOLD[k.split("|")[1]] for k in keys}, "run_source": {k: "src" for k in keys},
        "epochs_per_class": {c: sum(k.startswith(c + "|") for k in keys) for c in cc.CLASSES},
        "per_class_set_epoch": pcse,
        "pool_sizes": {c: {s: dict(pcse[c][s]) for s in ("S_old", "S_new", "S_newmains")} for c in ("A", "B")},
        "routec_sold_sim": {"sold_sim": 4, "registered": 3, "pilot": 1} if routec else None,
        "routec_prior_match": {"equal": True, "n_prior": len(PRIOR), "n_pilot": 1, "n_mains": 7} if routec else None,
        "drops": [{"world": "honest_a", "source": "S_old", "reason": "epoch 2 not in a pool", "n": 3}],
        "meta_checked": len(keys), "zero_row_runs": {c: copy.deepcopy(zr) for c in cc.CELLS},
        "uncut_zero_row_runs": {c: [ZERO] for c in cc.CELLS}, "n_uncut_failures": 0, "n_failed": 0, "n_required_gaps": 0,
        "rows_per_run_equal_across_layers": {"command": True, "reply": True}, "cells": cells,
    }


A3, Q = {cc.ASTRA: True, cc.QWEN: True}, {cc.ASTRA: True, cc.QWEN: True, cc.SOL: True}
POOL_RC = {"n": 6, "sol_using": 1, "sol_free": 3, "unknown": 2}
POOL_V83 = {"n": 5, "sol_using": 1, "sol_free": 2, "unknown": 2}
RK = "run_keys (all sim runs of the record)"


def world():
    _, ppath, psha = cc.PINS["pilot_v3"]
    item4 = {}
    for scope, n, pool in (("routec", 7, 6), ("v83", 6, 5)):
        item4[scope] = {c: {"n_detector_sim": pool, "n_pool": pool, "n_both": pool, "equal": True} for c in cc.CELLS}
        item4[scope][RK] = {"n": n, "detector_only_keys": [ZERO], "zero_row_cells_of_detector_only": {ZERO: list(cc.CELLS)}}
    item4["pool_n"] = {"routec": 6, "v83": 5}
    item4["flags"] = {**{f"cell {c} n_detector_sim": "changed (5 -> 6)" for c in cc.CELLS}, "pool_n": "changed"}
    ctx = {
        "ext_rc": ext(PRIOR + [PILOT], True), "ext_prior": ext(PRIOR, False),
        "folds": {"cell_set": CELL_SET},
        "pilot_v3": {"keys": [{"volume_path": "simh_evals_v1/" + path(PILOT), "task": "t6"}]},
        "pilot_manifest": {"files": [{"path": path(PILOT), "sha256": "0" * 64}], "counts": {"files": 1, "usable": 1},
                           "counted_attempts_without_eval": {"simh": []}},
        "inputs_full": {"freeze_checks": {"tau_sim": {"S_old": {"fill_in_admitted": {
            "simh_t196fill_v1__sim_honest": {"admitted": ["t196fill|t2"]},
            "sp493_g4_simh_v1_try2__sim_honest": {"admitted": ["sp493_t196|t7"]}}}}},
            "s_old_simh_extra_roots": {"simh_q4f_v1_pilot__sim_honest": {"commit": "344829af6", "path": ppath, "sha256": psha}}},
        "cont_headers": {"rows": [{"path": path("sim|t1|e1|s1"), "has_header": True, "models_any_tokens": Q}]},
        "scan_bits": [{"path": path("sim|t3|e1|s3"), "models": A3}, {"path": path("sim|t5|e1|s5"), "models": {}},
                      {"path": path("sim|t2|e1|s2"), "models": A3}],
        "usage_bits": [{"path": path("sim|t3|e1|s3"), "models": A3}],
        "catalogue": {"runs": {
            "simh_full|t1": {"roles": ["S"], "world": "simh_full", "epoch": "e1", "sets": ["S_old"]},
            "t196fill|t2": {"roles": ["S"], "world": "t196fill", "epoch": "e1", "sets": ["S_old"]},
            "sp493_t196|t7": {"roles": ["S"], "world": "sp493_t196", "epoch": "e1", "sets": ["S_old"]},
            "simh_311|t3": {"roles": ["S"], "world": "simh_311", "epoch": "e1", "sets": ["S_new"]},
            "simh_pilot|t6": {"roles": ["S"], "world": "simh_pilot", "epoch": "e1", "sets": ["S_old"]},
            "sp493_new|t5": {"roles": ["S"], "world": "sp493_new", "epoch": "e1", "sets": ["S_paper493"]},
            "honest_b|t3": {"roles": ["B"], "world": "honest_b", "epoch": "e1", "sets": ["S_new"]}}},
        "rates_171": {"headline_common_sets": {"sets": ["S_new", "S_newmains", "S_paper493"],
                                               "Y_sim": {"failed": 3, "launched": 40, "by_category": {cc.RATE_LIMIT: 1}}},
                      "sim": {"failed_attempts": {"S_new": [{"category": cc.RATE_LIMIT}, {"category": "unknown cause"}],
                                                  "S_newmains": [{"category": "unknown cause"}], "S_old": [{"category": "x"}]},
                              "by_set_arm": {"S_new|sim_honest": {"launched": 20}, "S_newmains|sim_honest": {"launched": 15},
                                             "S_paper493|sim_honest": {"launched": 5}, "S_old|sim_honest": {"launched": 9}}}},
        "ext_sp": {
            "item1": {"pool": {"routec": dict(POOL_RC), "v83": dict(POOL_V83)},
                      "detector_cells": {"routec": {c: dict(POOL_RC) for c in cc.CELLS}, "v83": {c: dict(POOL_V83) for c in cc.CELLS}},
                      "n_paths_by_number_of_sources": {"routec": {"0": 1, "1": 4, "2": 1}, "v83": {"0": 1, "1": 3, "2": 1}},
                      "pilot_header_scan": {"n_files": 1, "n_header_read": 1, "n_unknown": 0,
                                            "rows": [{"path": path(PILOT), "has_header": True, "models_any_tokens": A3}]},
                      "v83_counts_equal_relay": False},
            "item2": {"headline_sets": ["S_new", "S_newmains", "S_paper493"], "v83": {"failed": 3, "launched": 40, "rate_limit": 1},
                      "routec": {"failed": 3, "launched": 40, "rate_limit": 1}, "pilot_runs_in_headline_sets": [],
                      "pilot_attempts_S_old_apart": {"files": 1, "counted_attempts_without_eval_simh": 0,
                                                     "manifest_counts": {"files": 1, "usable": 1}}},
            "item4": item4},
    }
    bits = {"rows": [{"path": path(PILOT), "sha_ok": True, "has_header": True, "models_any_tokens": A3}]}
    return ctx, bits


def false_ids(ctx, bits):
    it, payload = cc.derive_all(ctx, bits)
    return [r["id"] for r in it.rows if not r["equal"]], it, payload


def mutated(ctx, keys, value):
    new = copy.deepcopy(ctx)
    d = new
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value
    assert json.dumps(new, sort_keys=True, default=str) != json.dumps(ctx, sort_keys=True, default=str)  # really differs
    return new


# ---------------------------------------------------------------- tests

def test_positive_world_all_equal():
    ctx, bits = world()
    bad, it, payload = false_ids(ctx, bits)
    assert bad == []
    assert len(it.rows) > 150
    assert payload["item1"]["pool"]["routec"] == POOL_RC and payload["item1"]["pool"]["v83"] == POOL_V83
    assert payload["item2"]["routec"] == {"failed": 3, "rate_limit": 1, "launched": 40}
    assert {r["label"] for r in it.rows} <= {"INDEPENDENT", "INTERNAL", "CROSS-RECORD", "copy check", "count / copy", cc.S7_LABEL}
    s7 = [r for r in it.rows if r["id"].startswith("S7.")]
    assert s7 and all(r["label"] == cc.S7_LABEL and r["prestated_false"] and r["note"] == cc.S7_NOTE for r in s7)
    assert not any(r.get("prestated_false") for r in it.rows if not r["id"].startswith("S7."))
    s = cc.summary(it.rows)
    assert s["false_ids_not_prestated"] == [] and sorted(s["prestated_false_ids_that_were_equal"]) == sorted(r["id"] for r in s7)


def test_s7_false_is_reported_apart_from_unexpected_false():
    ctx, bits = world()
    bad, it, _ = false_ids(mutated(ctx, ("ext_sp", "item1", "n_paths_by_number_of_sources", "routec"), {"1": 9}), bits)
    s = cc.summary(it.rows)
    assert bad and all(b.startswith("S7.routec.") for b in bad)
    assert s["false_ids_not_prestated"] == [] and set(s["false_ids"]) == set(bad)


def test_s7_note_sf_l28_reading():
    lines = [f"line {i}" for i in range(1, 28)] + ["   - The number of paths seen by 1, 2 or 3 sources is recorded."]
    out = cc.s7_note_sf_l28("\n".join(lines).encode())
    assert out["verbatim"] == "- The number of paths seen by 1, 2 or 3 sources is recorded."
    assert out["reading"] == cc.S7_L28_READING and out["producer_deviation_claimed"] is False and out["source"].endswith(" L28")


@pytest.mark.parametrize("keys,value,expect", [
    (("ext_rc", "epochs_per_class", "sim"), 8, "H1.sim"),
    (("ext_rc", "per_class_set_epoch", "B", "S_new", "e1"), 2, "H3.B.S_new.e1"),
    (("ext_rc", "zero_row_runs", "reply|L63", "n"), 0, "H9.reply|L63.n"),
    (("ext_rc", "cells", "command|L48", "problems", "a_vs_b", "n_runs", "class1"), 2, "H10.command|L48.a_vs_b.class1"),
    (("ext_sp", "item1", "pool", "routec", "sol_using"), 2, "S1.pool.routec.sol_using"),
    (("ext_sp", "item1", "detector_cells", "v83", "reply|L48", "unknown"), 1, "S6.v83.reply|L48.unknown"),
    (("ext_sp", "item2", "v83", "launched"), 41, "F2.v83.launched"),
    (("ext_sp", "item4", "routec", "command|L63", "n_detector_sim"), 7, "C1.routec.command|L63"),
    (("ext_sp", "item4", "pool_n", "v83"), 6, "C4.v83"),
    (("ext_sp", "item4", "flags", "pool_n"), "unchanged", "C8.pool_n"),
])
def test_one_changed_leaf_flips_exactly_that_item(keys, value, expect):
    ctx, bits = world()
    bad, _, _ = false_ids(mutated(ctx, keys, value), bits)
    assert bad == [expect]


def test_sol_flip_in_pilot_bits_changes_counts_and_row_agreement():
    ctx, bits = world()
    b2 = copy.deepcopy(bits)
    b2["rows"][0]["models_any_tokens"] = Q
    assert b2 != bits
    bad, _, payload = false_ids(ctx, b2)
    assert payload["item1"]["pool"]["routec"]["sol_using"] == 2
    assert "S8b.rows_agree" in bad and "S1.pool.routec.sol_using" in bad


@pytest.mark.parametrize("change", ["pilot_twice", "pilot_in_prior", "sol_and_free", "unmapped_role_s"])
def test_stops(change):
    ctx, bits = world()
    if change == "pilot_twice":
        ctx["ext_rc"]["run_keys"]["sim|t6|e1|zz"] = path(PILOT) + "#e0"
    elif change == "pilot_in_prior":
        ctx["ext_prior"]["run_keys"][PILOT] = path(PILOT) + "#e0"
    elif change == "sol_and_free":
        ctx["usage_bits"].append({"path": path("sim|t1|e1|s1"), "models": A3})
    else:
        ctx["catalogue"]["runs"]["simh_311|t9"] = {"roles": ["S"], "world": "simh_311", "epoch": "e1", "sets": ["S_new"]}
    with pytest.raises(cc.Stop):
        cc.derive_all(ctx, bits)


def test_copy_verdicts():
    assert cc.copy_verdict({}) == "INCOMPLETE"
    assert cc.copy_verdict(None, has_header=False) == "INCOMPLETE"
    assert cc.copy_verdict(A3) == "FREE"
    assert cc.copy_verdict({cc.ASTRA: True, cc.QWEN: False}) == "INCOMPLETE"
    assert cc.copy_verdict({cc.SOL: True, cc.QWEN: True}) == "SOL"
    assert cc.copy_verdict({cc.SOL: True, cc.QWEN: True}, sol_first=False) == "INCOMPLETE"
    assert cc.path_verdict([]) == "unknown" and cc.path_verdict(["FREE", "INCOMPLETE"]) == "unknown"


def run_jq(name, obj):
    out = subprocess.run(["jq", "-c", "-f", str(HERE / name)], input=json.dumps(obj).encode(), capture_output=True, check=True)
    return out.stdout.decode()


def test_honest_filter_drops_values():
    cell = {"rows": 3, "rows_per_run": {"k": 3}, "runs_without_rows": [], "seconds": SENT,
            "certificates": [{"certified_per_column": [True], "max_grad": SENT}],
            "paired_simh_minus_ab": {"observed": {"run_auroc": {"c": SENT}}},
            "problems": {p: {"n_runs": {"class1": 1, "class0": 1}, "observed_complete": True, "null_complete": True,
                             "observed_certified": True, "null": {"certified": True, "run_level_auroc": [SENT]},
                             "run_scores": {"sim|t1|e1|s1": SENT}, "run_level": {"auroc": SENT}} for p in cc.PROBLEMS}}
    rec = {"run_keys": {}, "run_set": {}, "run_epoch": {}, "run_cell": {}, "run_main": {}, "run_folds": {}, "run_source": {},
           "epochs_per_class": {}, "per_class_set_epoch": {}, "settings": {"C": SENT},
           "inputs": {"pool_sizes": {}, "drops": [{"world": "w", "source": "s", "reason": "r", "n": 1}]}, "skip": {"meta_checked": 1},
           "zero_row_runs": {"command|L48": {"n": 0, "by_class": {}, "by_set": {}, "runs": []}},
           "uncut_gates": {"zero_row_runs": {}, "failures": []}, "failed": {}, "required_gaps": [],
           "rows_per_run_equal_across_layers": {}, "cells": {"command|L48": cell}}
    out = run_jq("extract_honest_all.jq", rec)
    assert "0.98765" not in out
    assert json.loads(out)["cells"]["command|L48"]["problems"]["simh_vs_b"]["run_score_keys"] == ["sim|t1|e1|s1"]


def test_simpool_filter_drops_shares_and_percents():
    counts = {"n": 6, "sol_using": 1, "sol_free": 3, "unknown": 2, "shares": {"all_runs_denominator": SENT}}
    rec = {"gates": {"G": "PASS"}, "items": {
        "1_sim_footnote": {"pool": {"routec": counts, "v83": counts}, "percent": {"routec_pool": SENT}, "printed_decimals_percent": 0,
                           "detector_cells": {"routec": {"command|L48": counts}, "v83": {"command|L48": counts},
                                              "routec_cells_agree": True},
                           "n_paths_by_number_of_sources": {"routec": {"1": 1}}, "routec_pool_bounds_print_the_same": True,
                           "pilot_header_scan": {"n_files": 1, "n_header_read": 1, "n_unknown": 0,
                                                 "rows": [{"path": "p", "has_header": True, "models_any_tokens": {}, "n_samples": 7}]},
                           "flags": {"pool_v83_vs_routec": "unchanged"}, "footnote_uses": "x",
                           "v83_reproduces_printed": {"pool counts equal the 10-17 relay (sol, n)": True,
                                                      "pool percent matches the printed v83": True}},
        "2_failed_footnote": {"headline_sets": [], "pilot_runs_in_headline_sets": [],
                              "v83": {"failed": 1, "launched": 2, "rate_limit": 0, "percent": SENT, "percent_without_rate_limit": SENT},
                              "routec": {"failed": 1, "launched": 2, "rate_limit": 0, "percent": SENT},
                              "pilot_attempts_S_old_apart": {"files": 1, "counted_attempts_without_eval": {"simh": 3},
                                                             "manifest_counts": {"files": 1, "usable": 1}},
                              "flags": {"with_rate_limit": "unchanged"}},
        "4_honest_detector_cohort": {"routec": {}, "v83": {}, "pool_n": {}, "flags": {}, "record_A_sizes_S_after_runs": {"x": 5}},
        "5_table7_row5_control": {"run_level_auroc": SENT}}}
    out = json.loads(run_jq("extract_simpool_counts.jq", rec))
    text = json.dumps(out)
    assert "0.98765" not in text and "percent" not in text and "flags" not in json.dumps(out["item1"])
    assert "5_table7" not in text and "record_A_sizes" not in text and "routec_cells_agree" not in text
    assert out["item1"]["v83_counts_equal_relay"] is True
    assert out["item2"]["pilot_attempts_S_old_apart"]["counted_attempts_without_eval_simh"] == 3  # the record's number shape (O4)


def test_usage_filter_reduces_to_bits():
    scan = {"rows": [{"parts": ["L", "arm", "t1", "f.eval"], "model_usage": {cc.SOL: {"input_tokens": 0, "output_tokens": 5},
                                                                             cc.ASTRA: {"input_tokens": 0, "output_tokens": 0}}},
                     {"parts": ["L", "arm", "t2", "g.eval"], "model_usage": None}]}
    out = json.loads(run_jq("extract_usage_bits.jq", scan))
    assert out == [{"path": "L/arm/t1/f.eval", "models": {cc.SOL: True, cc.ASTRA: False}}, {"path": "L/arm/t2/g.eval", "models": {}}]
    usage = {"runs": [{"path": "L/arm/t1/f.eval", "usage": {cc.QWEN: {"total_tokens": 9}}, "usd": SENT}]}
    assert json.loads(run_jq("extract_usage_bits.jq", usage)) == [{"path": "L/arm/t1/f.eval", "models": {cc.QWEN: True}}]


def synthetic_simpool_record():
    def shares(c):
        n, s, u = c["n"], c["sol_using"], c["unknown"]
        return {"all_runs_denominator": float(Fraction(s, n)), "unknown_counted_sol_upper": float(Fraction(s + u, n)),
                "unknown_excluded": float(Fraction(s, n - u))}
    pool = {sc: {**c, "shares": shares(c)} for sc, c in (("routec", POOL_RC), ("v83", POOL_V83))}
    cells = {sc: {c: {**pool[sc]} for c in cc.CELLS} for sc in pool}
    rc, v = pool["routec"]["shares"], pool["v83"]["shares"]
    return {"items": {
        "1_sim_footnote": {"pool": pool, "detector_cells": cells, "printed_decimals_percent": 0,
                           "percent": {"routec_pool": 100 * rc["all_runs_denominator"], "routec_pool_upper": 100 * rc["unknown_counted_sol_upper"],
                                       "routec_pool_unknown_excluded": 100 * rc["unknown_excluded"], "v83_pool": 100 * v["all_runs_denominator"]},
                           "routec_pool_bounds_print_the_same": False,  # 16.67 -> 17 vs 50 -> 50
                           "v83_reproduces_printed": {"pool percent matches the printed v83": False},  # 20 != 6
                           "flags": {"pool_v83_vs_routec": "changed (20 -> 17)",
                                     **{f"cell {c} v83_vs_routec": "changed" for c in cc.CELLS}}},
        "2_failed_footnote": {"printed_decimals_percent": 1,
                              "v83": {"percent": 7.5, "percent_without_rate_limit": 5.0},
                              "routec": {"percent": 7.5, "percent_without_rate_limit": 5.0},
                              "v83_reproduces_printed": {"percent matches the printed v83": False,
                                                         "percent_without_rate_limit matches the printed v83": False},
                              "flags": {"with_rate_limit": "unchanged", "without_rate_limit": "unchanged"}}}}


def test_container_share_compare_positive_and_negative():
    m = load_entry()
    ctx, bits = world()
    _, _, payload = false_ids(ctx, bits)
    rb = json.dumps(synthetic_simpool_record()).encode()
    payload["simpool_sha256"] = hashlib.sha256(rb).hexdigest()
    out = m.compare_record_shares(rb, payload)
    assert out["sha_ok"] and [k for k, r in out["items"].items() if not r["equal"]] == []
    assert len(out["items"]) == 6 + 2 * 4 * 3 + 1 + 4 + 1 + 1 + 1 + 4 + 1 + 4 + 2 + 2
    bad = synthetic_simpool_record()
    bad["items"]["1_sim_footnote"]["percent"]["v83_pool"] = 20.01
    rb2 = json.dumps(bad).encode()
    assert rb2 != rb
    payload["simpool_sha256"] = hashlib.sha256(rb2).hexdigest()
    out2 = m.compare_record_shares(rb2, payload)
    assert [k for k, r in out2["items"].items() if not r["equal"]] == ["S3.v83_pool"]
    assert m.compare_record_shares(rb, payload)["sha_ok"] is False  # bytes no longer match the pinned sha


def test_container_returns_no_record_values():
    m = load_entry()
    ctx, bits = world()
    _, _, payload = false_ids(ctx, bits)
    rec = synthetic_simpool_record()
    rec["items"]["1_sim_footnote"]["percent"]["routec_pool"] = SENT
    rb = json.dumps(rec).encode()
    payload["simpool_sha256"] = hashlib.sha256(rb).hexdigest()
    text = json.dumps(m.compare_record_shares(rb, payload))
    assert "0.98765" not in text
    leaves = [v for r in json.loads(text)["items"].values() for k, v in r.items() if k not in ("record_key", "label", "cite")]
    assert all(isinstance(v, bool) for v in leaves)


def test_bits_from_log():
    m = load_entry()
    u = lambda **kw: SimpleNamespace(**{f: kw.get(f, 0) for f in m.TOKEN_FIELDS})  # noqa: E731
    log = SimpleNamespace(status="success", results=SimpleNamespace(completed_samples=1),
                          stats=SimpleNamespace(model_usage={cc.SOL: u(output_tokens=3), cc.ASTRA: u(), cc.QWEN: u(reasoning_tokens=None, total_tokens=2)}))
    assert m.bits_from_log(log) == {"status": "success", "completed_samples": 1,
                                    "models_any_tokens": {cc.SOL: True, cc.ASTRA: False, cc.QWEN: True}}


def test_entry_imports_at_root_and_name_length():
    src = ENTRY.read_text()
    g = {"__file__": "/root/run_routec_counts_check.py", "__name__": "run_routec_counts_check"}
    exec(compile(src, "/root/run_routec_counts_check.py", "exec"), g)
    assert len(g["APP_NAME"]) <= 63
    assert "import countscheck" not in src.split("def main")[0]  # the container never imports the laptop module


@pytest.mark.parametrize("apps,ok", [
    ([], True),
    ([{"description": "other", "state": "deployed"}, {"description": "N", "state": "stopped"}], True),
    ([{"description": "N", "state": "ephemeral (detached)"}], False),
    ([{"Description": "N", "State": "ephemeral"}], None),  # upper-case keys: schema error, fails closed
    ({"description": "N"}, None),
])
def test_no_live_app_gate(apps, ok):
    p = subprocess.run(["jq", "-e", "--arg", "n", "N", "-f", str(HERE / "no_live_app.jq")], input=json.dumps(apps).encode(),
                       capture_output=True)
    if ok is None:
        assert p.returncode not in (0, 1)
    else:
        assert (p.returncode == 0) is ok and p.stdout.decode().strip() == ("true" if ok else "false")


def test_shape_never_prints_values():
    s = json.dumps(cc.shape({"a": {"b": SENT, "c": [SENT]}, "d": "text"}))
    assert "0.98765" not in s and "text" not in s


@pytest.mark.parametrize("args,msg", [
    ([], "STOP: usage"),
    (["not-a-sha"], "STOP: usage"),
    (["0" * 40, "launch"], "STOP: mode must be run or gate-only"),
    (["0" * 40, "gate-only"], "STOP: HEAD != H"),  # stops at the first git check: no modal call
])
def test_launcher_refuses_before_any_modal_call(args, msg):
    p = subprocess.run(["zsh", "-f", str(HERE / "launch_countscheck.zsh"), *args], capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout.startswith(msg)
