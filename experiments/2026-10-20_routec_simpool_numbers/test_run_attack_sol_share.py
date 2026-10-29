"""Fixtures for run_attack_sol_share.py: synthetic records only; item (c) imported from its real blob. Run:
uv run --group dev pytest experiments/2026-10-20_routec_simpool_numbers/test_run_attack_sol_share.py -q"""

import ast
import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ras", HERE / "run_attack_sol_share.py")
ras = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ras)
SRC = subprocess.run(["git", "show", "d1db55766:experiments/2026-10-15_e2e_sim_pipeline/run_detector_solfree_auroc.py"], capture_output=True,
                     check=True, cwd=ras.REPO).stdout
SF = ras.load_pinned("solfree_script", SRC)
SOL, ASTRA, QWEN = SF.SOL, *SF.EXPECTED
CELLS = ("command|L48", "reply|L48")
SIMS = [f"c{i:02d}" for i in range(12)]
SOLC = {"c00", "c01", "c02"}  # Sol-using per the scan rows


def path(c, world="sim"):
    return f"2026-10-15_atk/{world}/{c}/run.eval"


def attack_record():
    rng = np.random.default_rng(3)
    rk, mains = {}, {}
    for c in SIMS:
        for w in ("sim", "run1"):
            k = f"{w}|{c}|e1|h"
            rk[k], mains[k] = path(c, w) + "#e0", c
    rec = {"run_keys": rk, "run_main": mains, "settings": {"primary": {"command": 63, "reply": 48}}, "cells": {}}
    for cell in CELLS:
        ks = sorted(rk)
        s = {k: float(rng.normal()) for k in ks}
        y = np.array([int(k.startswith("sim|")) for k in ks])
        au = SF.wauc(np.array([s[k] for k in ks]), y, np.ones(len(ks)))
        rec["cells"][cell] = {"problems": {"c_simatk_vs_run1": {"run_scores": s, "run_level": {"auroc": au}, "run_auroc_boot": {}}}}
    return rec


def fixture():
    scan = {"rows": [{"parts": path(c).split("/"), "model_usage": {ASTRA: {"output_tokens": 5}, QWEN: {"output_tokens": 5}}
                      | ({SOL: {"output_tokens": 2}} if c in SOLC else {})} for c in SIMS if c != "c11"]}  # c11: no copy -> UNKNOWN
    runs = [{"family": "sim attack", "routed": True, "arm": ras.LEAD_ARM, "path": path(c),
             "usage": {ASTRA: {"output_tokens": 5}, QWEN: {"output_tokens": 5}} | ({SOL: {"output_tokens": 2}} if c in SOLC else {})}
            for c in SIMS[:10]]  # c10, c11 not routed
    runs += [{"family": "sim attack", "routed": True, "arm": "sim_attack@september_ungrounded", "path": path("u00"),
              "usage": {ASTRA: {"output_tokens": 5}, QWEN: {"output_tokens": 5}, SOL: {"output_tokens": 1}}},
             {"family": "sim attack", "routed": False, "arm": ras.LEAD_ARM, "path": path("c10"), "usage": {}},
             {"family": "real", "routed": True, "arm": "attack", "path": path("c00", "run1"), "usage": {QWEN: {"output_tokens": 5}}}]
    header = {k: copy.deepcopy(v) for k, v in SF.HEADER_PINS.items()} | {"rows": [
        {"path": f"2026-10-14_160932_simh_cont/sim_honest@september/k{i:03d}/run.eval", "has_header": True,
         "models_any_tokens": {ASTRA: True, QWEN: True}} for i in range(SF.HEADER_PINS["n_files"])]}
    blobs = {"solfree_script": SRC, "attack": json.dumps(attack_record()).encode(), "scan": json.dumps(scan).encode(),
             "usage": json.dumps({"runs": runs}).encode(), "header": json.dumps(header).encode()}
    return blobs


@pytest.fixture
def pinned(monkeypatch):
    blobs = fixture()
    monkeypatch.setattr(ras, "PINS", {n: (c, p, ras.sha(blobs[n])) for n, (c, p, _) in ras.PINS.items()})
    return blobs


def test_counts_scope_and_routed_variants(pinned, monkeypatch):
    r = ras.build(pinned, SF, {"git": "x"})[0]
    pc = r["a_reduce_counts"]["per_cell"]
    assert {c: (v["n_class1_sol_using"], v["n_class1_all"], v["n_class1_unknown"]) for c, v in pc.items()} == {c: (3, 12, 2) for c in CELLS}  # c10: an incomplete usage copy; c11: no copy
    assert r["a_reduce_counts"]["pooled_union_of_cells"] == {"n_all": 12, "n_sol_using": 3, "n_sol_free": 7, "n_unknown": 2}
    assert r["gates"]["G_REPRODUCES_546_1309"] == "FAIL in ['cell command|L48', 'cell reply|L48', 'pooled'] (0 of 3)"
    r1, r2 = (r["c_routed_variants"][k] for k in sorted(r["c_routed_variants"]))
    assert (r1["n_runs"], r1["n_sol_using"]) == (11, 4) and (r2["n_runs"], r2["n_sol_using"]) == (10, 3)
    assert r1["gap_vs_reduce_pooled"]["in_routed_only"]["paths"] == [path("u00")]
    assert r2["gap_vs_reduce_pooled"]["in_reduce_only"]["paths"] == [path("c10"), path("c11")]
    assert r2["gap_vs_reduce_pooled"]["in_both_with_different_sol_verdict"] == []
    assert r["gates"]["G_ROUTED_EQUALS_550_1322"] == {k: False for k in r["c_routed_variants"]}
    assert json.loads(json.dumps(r, sort_keys=True)) == r
    monkeypatch.setattr(ras, "EXPECTED", {"reduce_sol_n": (3, 12), "routed_sol_n": (3, 10)})
    r = ras.build(pinned, SF, {"git": "x"})[0]
    assert r["gates"]["G_REPRODUCES_546_1309"] == "PASS (3 of 3)"
    assert r["gates"]["G_ROUTED_EQUALS_550_1322"] == {"R1 family 'sim attack' and routed": False, "R2 = R1 and arm 'sim_attack@september'": True}


def test_changed_input_stops(pinned):
    pinned["usage"] = pinned["usage"] + b" "
    with pytest.raises(ras.Stop, match="input usage differs"):
        ras.build(pinned, SF, {})


def test_item_c_positive_control_still_stops(monkeypatch):
    blobs = fixture()
    rec = json.loads(blobs["attack"])
    rec["cells"]["reply|L48"]["problems"]["c_simatk_vs_run1"]["run_level"]["auroc"] += 0.01
    blobs["attack"] = json.dumps(rec).encode()
    monkeypatch.setattr(ras, "PINS", {n: (c, p, ras.sha(blobs[n])) for n, (c, p, _) in ras.PINS.items()})
    with pytest.raises(SystemExit, match="positive control failed"):
        ras.build(blobs, SF, {})


def test_no_sol_anywhere_stops(monkeypatch):
    blobs = fixture()
    scan = json.loads(blobs["scan"])
    for r in scan["rows"]:
        r["model_usage"].pop(SOL, None)
    blobs["scan"] = json.dumps(scan).encode()
    usage = json.loads(blobs["usage"])
    for r in usage["runs"]:
        r["usage"].pop(SOL, None)
    blobs["usage"] = json.dumps(usage).encode()
    monkeypatch.setattr(ras, "PINS", {n: (c, p, ras.sha(blobs[n])) for n, (c, p, _) in ras.PINS.items()})
    with pytest.raises(SystemExit, match="zero simulated runs flagged Sol-using"):
        ras.build(blobs, SF, {})


def test_scope_line_citations_match_the_blob():
    lines = SRC.decode().splitlines()
    assert "rs = c[\"problems\"][cfg[\"problem\"]][\"run_scores\"]" in lines[141] and "keys = sorted(rs)" in lines[142]
    assert "sim = {k for c in rec[\"cells\"].values()" in lines[179] and "fl = [flags.get(norm(rec[\"run_keys\"][k]))" in lines[191]
    assert "\"n_class1_all\": int(y.sum())" in lines[209] and "\"n_class1_sol_using\": int(" in lines[210]
    assert "for r in scan[\"rows\"]:" in lines[118] and "for r in usage[\"runs\"]:" in lines[123] and "for r in (header or {})" in lines[127]
    assert "flags = {p: True if \"sol\" in x else None if \"incomplete\" in x else False" in lines[134]


def test_container_import_at_root_and_stdlib_only_imports():
    src = (HERE / "run_attack_sol_share.py").read_text()
    ns = {"__file__": "/root/run_attack_sol_share.py", "__name__": "ras_root"}
    exec(compile(src, "/root/run_attack_sol_share.py", "exec"), ns)
    assert ns["REPO"] == Path("/root")
    top = {a.name.split(".")[0] for n in ast.parse(src).body if isinstance(n, ast.Import) for a in n.names}
    top |= {n.module.split(".")[0] for n in ast.parse(src).body if isinstance(n, ast.ImportFrom)}
    assert top <= {"hashlib", "json", "re", "subprocess", "sys", "types", "pathlib", "modal"}
    assert len(ns["APP"]) <= 62 and ras.OUT_NAME.fullmatch("attack_sol_share_2026-10-20_030000.json")


def test_committed_record_has_counts_only_and_item_c_bytes_apart(pinned):
    """Review O1: no AUROC, interval or Sol-free AUROC in the committed record; item (c)'s full record is returned apart (Volume only)."""
    r, item_c = ras.build(pinned, SF, {"git": "x"})
    text = json.dumps(r)
    assert "auroc" not in text.replace("run_detector_solfree_auroc", "") and "ci95" not in text
    assert r["item_c_attack_record"]["sha256"] == ras.sha(item_c) and "auroc_all" in json.loads(item_c)["detectors"]["attack"]["cells"]["reply|L48"]
    assert r["gates"]["G_ROUTED_ROWS"] == {"R1 family 'sim attack' and routed": "kept 11 rows", "R2 = R1 and arm 'sim_attack@september'": "kept 10 rows"}


def test_pinned_item_c_source_matches_its_pin():
    assert ras.sha(SRC) == ras.PINS["solfree_script"][2]


def test_check_pins_before_execution():
    blobs = {n: b"" for n in ras.PINS}
    with pytest.raises(ras.Stop, match="input solfree_script differs"):
        ras.check_pins(blobs)


@pytest.mark.parametrize("field,value", [("routed", "true"), ("family", "sim-attack")])
def test_routed_filter_keeping_no_rows_stops(monkeypatch, field, value):
    blobs = fixture()
    usage = json.loads(blobs["usage"])
    for r in usage["runs"]:
        r[field] = value
    blobs["usage"] = json.dumps(usage).encode()
    monkeypatch.setattr(ras, "PINS", {n: (c, p, ras.sha(blobs[n])) for n, (c, p, _) in ras.PINS.items()})
    with pytest.raises(ras.Stop, match="the filter keeps 0 rows"):
        ras.build(blobs, SF, {})


def test_repeated_routed_paths_stop(monkeypatch):
    blobs = fixture()
    usage = json.loads(blobs["usage"])
    usage["runs"].append(copy.deepcopy(usage["runs"][3]))
    blobs["usage"] = json.dumps(usage).encode()
    monkeypatch.setattr(ras, "PINS", {n: (c, p, ras.sha(blobs[n])) for n, (c, p, _) in ras.PINS.items()})
    with pytest.raises(ras.Stop, match="repeated eval paths"):
        ras.build(blobs, SF, {})


def test_two_sim_runs_sharing_a_path_stop(monkeypatch):
    blobs = fixture()
    rec = json.loads(blobs["attack"])
    rec["run_keys"]["sim|c05|e1|h"] = rec["run_keys"]["sim|c04|e1|h"]
    blobs["attack"] = json.dumps(rec).encode()
    monkeypatch.setattr(ras, "PINS", {n: (c, p, ras.sha(blobs[n])) for n, (c, p, _) in ras.PINS.items()})
    with pytest.raises(ras.Stop, match="share an eval path"):
        ras.build(blobs, SF, {})


def test_routed_verdict_differing_from_item_c_is_listed(monkeypatch):
    """c00 is Sol-using by the scan rows; its routed usage row is incomplete, so the routed verdict differs and is listed by path."""
    blobs = fixture()
    usage = json.loads(blobs["usage"])
    usage["runs"][0]["usage"] = {}
    blobs["usage"] = json.dumps(usage).encode()
    monkeypatch.setattr(ras, "PINS", {n: (c, p, ras.sha(blobs[n])) for n, (c, p, _) in ras.PINS.items()})
    r, _ = ras.build(blobs, SF, {})
    r2 = r["c_routed_variants"]["R2 = R1 and arm 'sim_attack@september'"]
    assert r2["gap_vs_reduce_pooled"]["in_both_with_different_sol_verdict"] == [path("c00")] and r2["n_incomplete"] == 1
