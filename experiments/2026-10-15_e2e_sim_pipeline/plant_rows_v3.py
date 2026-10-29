"""PLANT V3 block (v): V3's OWN code around the ported producer (plant_rows_v3_port.py). PRESTATED_V3 fb34c414 + AMEND1 5e54a709,
AMEND2 748da4ab, AMEND3 94fc0084, AMEND3a 1d0f5696, AMEND4 e84c8e1e, AMEND5 a2eb02fe.

- run_block_v: the ported plant_rows_body on the PLANT record (gz, as committed), the base A record (gz) and plant_rows' FULL config, then a
  two-way, whole-tree comparison of its rows / counts / aggregate with plant_rows (the TARGET), the 23 printed-row booleans (final9 U v17),
  and the controls: the misses fixture (D1), P4 (AMEND4 D3, AMEND5 F1) and A4 (AMEND4 D3, AMEND5 F3).
- cmp_tree: at EVERY dict node the key sets must be equal both ways; lists by length then element; leaves by the caller's eq(); an empty
  container is compared as a value ({} == {} only, [] == [] only). Counts n_leaf and n_empty per block (AMEND5 F3: exact gates).
- selftest: synthetic only (< 1 s): comparator fixtures, and the ported verdict functions on a synthetic record / npz (AMEND4 D4: item 24
  verdicts, 47(2) / 34 unavailable, never average over the attainable subset, draw-0 vs bootstrap-mean disagreement).
Keys, counts and booleans leave this module in the summary; values stay in the detail (the results Volume).
"""

import copy
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

import plant_rows_v3_port as PR

PORT_LABEL = ("producer's field definitions (RULE defines only the verdict rule); ported from run_plant_rows.py @ fd12eca35 sha256 "
              "199b1ab31983da827b7db210fef2380a0ad3019d398312f9919fa1f79040ef5b; excess move: producer's reading (RULE ambiguous; "
              "GATES_routec @ 4e4ab4142 L580)")
EXPECT = {"rows": (2760, 0), "counts": (404, 15), "aggregate": (656, 0)}  # (n_leaf, n_empty) per block (09's count; AMEND5 F3), gated exactly
SEEDS = {"P4": 20261022, "A4": 20261023}

_A = ("aggregate", "by_level")
ROW_PATHS = {  # the 23 printed rows: final9 (d31de34b0, c97aed1c) U v17 (98b5054fb, 220d25d2) key paths into plant_rows (AMEND4 D2)
    47: [_A + (f, "C", "-3", "S", "verdict") for f in ("0.01", "0.05")],
    48: [_A + (f, "C", "-3", "excess", "verdict") for f in ("0.01", "0.05")],
    49: [("counts", "0.01", "-3", "C")], 50: [("counts", "0.01", "+3", "C")],
    51: [_A + (f, "C", "-5", "S", "verdict") for f in ("0.01", "0.05")],
    52: [_A + (f, "C", "-5", "excess", "verdict") for f in ("0.01", "0.05")],
    53: [("counts", "0.01", "-5", "C")], 54: [("counts", "0.01", "+5", "C")],
    55: [_A + (f, "C", "-8", "S", "verdict") for f in ("0.01", "0.05")],
    56: [_A + (f, "C", "-8", "excess", "verdict") for f in ("0.01", "0.05")],
    57: [("counts", "0.01", "-8", "C")], 58: [("counts", "0.01", "+8", "C")],
    59: [_A + (f, "C", "-12", "S", "verdict") for f in ("0.01", "0.05")],
    60: [_A + (f, "C", "-12", "excess", "verdict") for f in ("0.01", "0.05")],
    61: [("counts", "0.01", "-12", "C")], 62: [("counts", "0.01", "+12", "C")],
    79: [_A + ("0.01", "C", "-3", "S", "verdict")],
    80: [_A + ("0.01", "C", d, "S", "verdict") for d in ("-3", "-5", "-8", "-12")],
    81: [("counts", "0.01", "-8", "C"), ("counts", "0.01", "-12", "C", "detected")],
    82: [("counts", "0.01", "-3", "C"), ("counts", "0.01", "-5", "C")],
    83: [_A + ("0.01", "C", d, "excess", "verdict") for d in ("-5", "-8", "-12")],
    84: [_A + ("0.01", "C", "-3", "excess", "verdict")],
    85: [_A + ("0.05", "C", "-3", "excess", "verdict")]}


def cmp_tree(a, b, eq_, path=()):
    """Two-way, whole-tree comparison -> (mismatches [(path, reason)], counts {n_leaf, n_empty, n_dict, n_keyset}). Reasons carry key NAMES only."""
    mm, ct = [], {"n_leaf": 0, "n_empty": 0, "n_dict": 0, "n_keyset": 0}

    def walk(x, y, p):
        if isinstance(x, dict) or isinstance(y, dict):
            if not (isinstance(x, dict) and isinstance(y, dict)):
                mm.append((p, "type"))
                return
            ct["n_dict"] += 1
            ct["n_keyset"] += 1
            if not x and not y:
                ct["n_empty"] += 1
                return
            if set(x) != set(y):
                mm.append((p, "keys", sorted(map(str, set(x) - set(y)))[:5], sorted(map(str, set(y) - set(x)))[:5]))
            for k in sorted(set(x) & set(y), key=str):
                walk(x[k], y[k], p + (k,))
            return
        if isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
            if not (isinstance(x, (list, tuple)) and isinstance(y, (list, tuple))):
                mm.append((p, "type"))
                return
            if not x and not y:
                ct["n_empty"] += 1
                return
            if len(x) != len(y):
                mm.append((p, "length"))
                return
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, p + (i,))
            return
        ct["n_leaf"] += 1
        if not eq_(x, y):
            mm.append((p, "value"))

    walk(a, b, tuple(path))
    return mm, ct


def node(tree, path):
    """The node at a key path, or a sentinel when any key is absent."""
    for k in path:
        if not isinstance(tree, dict) or k not in tree:
            return _ABSENT
        tree = tree[k]
    return tree


class _Absent:
    pass


_ABSENT = _Absent()


def row_booleans(mine, target, eq_):
    """AMEND4 D2: each printed row's boolean = AND over its key paths of the whole-node comparison (an absent node on either side is False)."""
    out = {}
    for r, ps in sorted(ROW_PATHS.items()):
        ok = True
        for p in ps:
            x, y = node(mine, p), node(target, p)
            ok = ok and x is not _ABSENT and y is not _ABSENT and not cmp_tree(x, y, eq_)[0]
        out[str(r)] = bool(ok)
    return out


def _blocks(o):
    return {k: o.get(k) for k in ("rows", "counts", "aggregate")}


def _cmp_blocks(mine, target, eq_):
    res = {}
    for b in ("rows", "counts", "aggregate"):
        mm, ct = cmp_tree(mine.get(b), target.get(b), eq_, (b,))
        res[b] = {"mismatches": mm, **ct, "n_compared": ct["n_leaf"] + ct["n_empty"]}
    return res


def _keys_only(mm):
    return [" | ".join(map(str, x[0])) + f" [{x[1]}]" for x in mm][:40]


def _ctl(ctl, cfg, rec_gz, base_gz):
    """One control copy of the ported body with V3_CTL set; V3_CTL is reset in finally (2e / reviewer: main block (v) first; None between)."""
    assert PR.V3_CTL is None, "V3_CTL is set before a control copy (STOP)"
    PR.V3_CTL = ctl
    try:
        return PR.plant_rows_body(copy.deepcopy(cfg), rec_gz, base_gz)
    finally:
        PR.V3_CTL = None


def misses_fixture(mine_counts, tgt_counts, cfg, eq_):
    """D1 / AMEND5 F3: one probe name NOT already in one misses map (an empty map first in sorted order if any) added to a copy of V3's counts;
    the comparison must flag exactly that misses node, by its key set."""
    cands = []
    for f in sorted(mine_counts):
        for tag in sorted(mine_counts[f]):
            for ro in sorted(mine_counts[f][tag]):
                m_ = mine_counts[f][tag][ro].get("misses")
                if isinstance(m_, dict):
                    cands.append((len(m_) > 0, f, tag, ro))
    assert cands, "misses fixture: no misses map"
    _, f, tag, ro = sorted(cands)[0]
    m_ = mine_counts[f][tag][ro]["misses"]
    rs_ = ro.split("+")
    names = sorted((f"{p} {r_} ({r_})" if "+" in ro else f"{p} {r_}") for p in cfg["probes"] for r_ in rs_)
    add = next(n for n in names if n not in m_)
    mc = copy.deepcopy(mine_counts)
    mc[f][tag][ro]["misses"][add] = "not detected"
    mm, _ = cmp_tree(mc, tgt_counts, eq_, ("counts",))
    want = ("counts", f, tag, ro, "misses")
    ok = len(mm) == 1 and mm[0][0] == want and mm[0][1] == "keys"
    return {"pass": bool(ok), "node": " | ".join(want), "was_empty": not m_, "flagged": _keys_only(mm)}


def _node_of(p):  # the comparison node a mismatch path belongs to (rows: level/readout/label/delta; counts: level/delta/readout; aggregate: level/readout/delta)
    b = p[0]
    if b == "rows":
        return p[:5]
    if b == "counts":
        return p[:4]
    if b == "aggregate":
        return p[:5]
    return p


def p4_control(out, tgt, cfg, rec_gz, base_gz, eq_):
    """AMEND4 D3 P4 / AMEND5 F1: one per-probe entry (verdict detected / not detected, no mechanism mismatch, move in range) chosen by seed;
    its pooled planted array shifted at draw 0 by +2 x delta / 100 in an overlay. Exactly the L546 gate (and L567 when all six arrays are
    present) must fail; the rows node and its counts nodes must mismatch the target; its aggregate node may; nothing else may."""
    cands = []
    for f, by_ro in out["rows"].items():
        for ro, by_lb in by_ro.items():
            for lb, by_d in by_lb.items():
                for tag, v in by_d.items():
                    if v.get("verdict") in ("detected", "not detected") and v.get("mechanism_mismatch") is None and v.get("move_in_range") is True:
                        cands.append((f, ro, lb, tag))
    assert cands, "P4: no eligible per-probe entry"
    cands.sort()
    f, ro, lb, tag = cands[int(np.random.default_rng(SEEDS["P4"]).integers(len(cands)))]
    d = int(tag)
    key = PR.npzkey(lb, cfg["combo"], cfg["mode"], f, "pooled", f"gap_delta_full_plant_{tag}")
    o4 = _ctl({"kind": "P4", "key": key, "shift": 2 * d / 100.0}, cfg, rec_gz, base_gz)
    six = all("planted_point" in out["rows"][f][ro][x].get(tag, {}) for x in out["rows"][f][ro])
    want_g = {("draw-0 move", lb, f, tag)} | ({("mean of the probes", ro, f, tag)} if six else set())
    got_g = set()
    for g_ in o4.get("v3_failed_gates") or []:
        c_ = str(g_.get("check", ""))
        if c_.startswith("draw-0 move"):
            got_g.add(("draw-0 move", g_.get("row"), g_.get("f"), g_.get("delta")))
        elif c_.startswith("mean of the probes"):
            got_g.add(("mean of the probes", g_.get("readout"), g_.get("f"), g_.get("delta")))
        else:
            got_g.add(("other: " + c_[:60], str(g_.get("row")), str(g_.get("f")), str(g_.get("delta"))))
    nodes = set()
    for b in ("rows", "counts", "aggregate"):
        mm, _ = cmp_tree(o4.get(b), tgt.get(b), eq_, (b,))
        nodes |= {_node_of(x[0]) for x in mm}
    req = {("rows", f, ro, lb, tag), ("counts", f, tag, ro), ("counts", f, tag, "+".join(cfg["readouts"]))}
    alw = req | ({("aggregate", "by_level", f, ro, tag)} if d < 0 else set())
    ok = got_g == want_g and req <= nodes <= alw
    return {"pass": bool(ok), "chosen": [f, ro, lb, tag], "gates_expected": sorted(map(list, want_g)), "gates_failed": sorted(map(list, got_g)),
            "nodes_mismatched": sorted(" | ".join(map(str, n)) for n in nodes), "required": sorted(" | ".join(map(str, n)) for n in req)}


def a4_control(out, tgt, cfg, rec_gz, base_gz, eq_):
    """AMEND4 D3 A4 / AMEND5 F3: one testable excess node (|Delta_exp| >= 0.5 points) with move in range, chosen by seed; the planted
    per-draw excess shifted on draws 1..D only by (2 |Delta_exp| + 1) / 100. That excess node must mismatch with move_in_range false;
    nothing outside it may change; no gate may fail."""
    cands = []
    for f, by_ro in out["aggregate"]["by_level"].items():
        for ro, by_d in by_ro.items():
            for tag, v in by_d.items():
                x = (v or {}).get("excess") or {}
                if int(tag) < 0 and x.get("verdict") in ("detected", "not detected") and x.get("move_in_range") is True:
                    cands.append((f, ro, tag))
    assert cands, "A4: no testable in-range excess node"
    cands.sort()
    f, ro, tag = cands[int(np.random.default_rng(SEEDS["A4"]).integers(len(cands)))]
    dexp = out["aggregate"]["by_level"][f][ro][tag]["excess"]["delta_exp_points"]
    o5 = _ctl({"kind": "A4", "f": f, "ro": ro, "tag": tag, "shift": (2 * abs(dexp) + 1) / 100.0}, cfg, rec_gz, base_gz)
    if o5.get("GATE_FAIL"):
        return {"pass": False, "chosen": [f, ro, tag], "why": "a gate failed in the A4 copy"}
    pre = ("aggregate", "by_level", f, ro, tag, "excess")
    outside, inside = [], []
    for b in ("rows", "counts", "aggregate"):
        mm, _ = cmp_tree(o5.get(b), tgt.get(b), eq_, (b,))
        for x in mm:
            (inside if x[0][:len(pre)] == pre else outside).append(x)
    mir = node(o5, ("aggregate", "by_level", f, ro, tag, "excess", "move_in_range"))
    ok = not outside and bool(inside) and mir is False
    return {"pass": bool(ok), "chosen": [f, ro, tag], "inside": _keys_only(inside), "outside": _keys_only(outside), "move_in_range_after": mir is True}


def run_block_v(cfg, rec_gz, base_gz, target_bytes, eq_):
    """The main block (v) FIRST (V3_CTL asserted None and recorded), then the controls. -> (summary: keys / counts / booleans, detail: values)"""
    ctl_none_at_start = PR.V3_CTL is None
    assert ctl_none_at_start, "V3_CTL is set at the start of block (v) (STOP)"
    assert "synthetic_TEST" not in cfg, "AMEND3a: the FULL config has no synthetic_TEST key (synth is not ported) (STOP)"
    out = PR.plant_rows_body(copy.deepcopy(cfg), rec_gz, base_gz)
    tgt = json.loads(target_bytes)
    summ = {"port_label": PORT_LABEL, "v3_ctl_none_at_start": ctl_none_at_start}
    if out.get("GATE_FAIL"):  # the producer's fail-closed withhold (L588-590): a V3 STOP, failed gate names only
        names = sorted({str(g_.get("check", ""))[:80] for g_ in (out.get("gates") or {}).get("failures", [])})
        return summ | {"pass": False, "gate_fail": True, "n_gates": (out.get("gates") or {}).get("n"), "failed_checks": names}, {"out": out}
    cb_ = _cmp_blocks(out, tgt, eq_)
    counts_ok = {b: (cb_[b]["n_leaf"], cb_[b]["n_empty"]) == EXPECT[b] for b in EXPECT}
    rb_ = row_booleans(_blocks(out), _blocks(tgt), eq_)
    mf_ = misses_fixture(out["counts"], tgt["counts"], cfg, eq_)
    p4_ = p4_control(out, tgt, cfg, rec_gz, base_gz, eq_)
    a4_ = a4_control(out, tgt, cfg, rec_gz, base_gz, eq_)
    assert PR.V3_CTL is None, "V3_CTL not reset after the controls (STOP)"
    blocks_ = {b: {k: cb_[b][k] for k in ("n_leaf", "n_empty", "n_dict", "n_keyset", "n_compared")} | {"n_mismatch": len(cb_[b]["mismatches"]),
                                                                                                    "mismatches": _keys_only(cb_[b]["mismatches"]),
                                                                                                    "counts_as_predicted": counts_ok[b]} for b in cb_}
    all_eq = all(not cb_[b]["mismatches"] for b in cb_)
    summ |= {"gates": {"n": out["gates"]["n"], "n_fail": out["gates"]["n_fail"]}, "blocks": blocks_, "all_equal": all_eq,
             "printed_rows": rb_, "printed_rows_all_true": all(rb_.values()) and len(rb_) == 23,
             "controls": {"misses_fixture": mf_, "P4": p4_, "A4": a4_}, "v3_ctl_none_at_end": PR.V3_CTL is None,
             "pass": bool(all_eq and all(counts_ok.values()) and all(rb_.values()) and len(rb_) == 23 and mf_["pass"] and p4_["pass"] and a4_["pass"])}
    return summ, {"out": out, "mismatch_paths": {b: [list(map(str, x[0])) for x in cb_[b]["mismatches"]] for b in cb_}}


# ---------------------------------------------------------------- self-tests (synthetic only) ----------------------------------------------------------------

def _eq_selftest(a, b):  # a stand-in eq for the comparator fixtures (the checker passes its own eq())
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def _synthetic(tmp):
    """A synthetic PLANT-like record, base record, config and npz (no real value). Six probes, readout C, one member origin each, 40 draws."""
    rng = np.random.default_rng(7)
    D = 40
    probes = [f"p{i}" for i in range(6)]
    cb, md, f = "pool", "joint", "0.01"
    cfg = {"combo": cb, "mode": md, "readouts": ["C"], "deltas": [-12, -8, -5, -3, 3, 5, 8, 12], "probes": probes, "calibration_field": ["meta", "window"],
           "calibration_expected": "full", "plant_coheadline_group": "six", "coheadline_group": "six_base", "aggregate": {"group": "G"}}
    z, rows, pooled, bpooled, coh = {}, {}, {}, {}, {}
    unav = {("p0", "+8"), ("p1", "-3")}  # (1) a ceiling and (2) a floor: unattainable member arm-rows
    for i, p in enumerate(probes):
        lb, o = f"{p} C", f"o{i}"
        base = rng.normal(0.0, 0.01, D + 1)
        s1, s2 = base + rng.normal(0, 0.005, D + 1), base + rng.normal(0, 0.005, D + 1)
        base = 0.5 * (s1 + s2)
        yg = rng.normal(0.0, 0.01, D + 1)
        z[PR.npzkey(lb, cb, md, f, "pooled", "gap_delta_full")] = base
        z[PR.npzkey(lb, cb, md, f, "pooled", "gap_sim_minus_run1")] = s1
        z[PR.npzkey(lb, cb, md, f, "pooled", "gap_sim_minus_run2")] = s2
        z[PR.npzkey(lb, cb, md, f, "pooled", "gap_delta_rr")] = yg
        z[PR.npzkey(lb, cb, md, f, o, "D")] = np.full(D + 1, 10.0)
        arms = {}
        for d in cfg["deltas"]:
            tag = f"{d:+d}"
            if (p, tag) in unav:
                arms[tag] = {"status": "unattainable"}
                continue
            mv = np.full(D + 1, d / 100.0)
            if p == "p2" and d == -8:
                mv[0] = 2.5 * d / 100.0  # item 24: a draw-0 move outside 50% of delta -> mechanism mismatch
            if d == -5:
                mv[1:] = 3 * d / 100.0  # draw-0 vs bootstrap-mean disagreement (the excess follows the bootstrap mean; S the draw-0 move)
            pa = base + mv
            z[PR.npzkey(lb, cb, md, f, "pooled", f"gap_delta_full_plant_{tag}")] = pa
            arms[tag] = {"status": "ok", "c": 0.1, "achieved_points": float(100 * (pa[0] - base[0]))}
        rows[lb] = {o: {"plant": {"solve_draw0": {f: {"sim": {"arms": arms}}}}}}
        pooled[lb] = {cb: {"members": [o], md: {f: {"gap": {"delta_full": PR_summ(base)}, "readable": {"all": True}}}}}
        bpooled[lb] = {cb: {md: {f: {"gap": {"delta_full": PR_summ(base)}}}}}
    labs = [f"{p} C" for p in probes]
    q = {}
    for d in cfg["deltas"]:
        tag = f"{d:+d}"
        ks = [PR.npzkey(lb, cb, md, f, "pooled", f"gap_delta_full_plant_{tag}") for lb in labs]
        q[tag] = ({"S": PR_summ(np.mean([z[k] for k in ks], 0)), "rows": labs} if all(k in z for k in ks) else {"unavailable": "synthetic"})
    coh = {cb: {md: {f: {"six|C": q}}}}
    S0 = np.mean([z[PR.npzkey(lb, cb, md, f, "pooled", "gap_delta_full")] for lb in labs], 0)
    Y = np.mean([np.abs(z[PR.npzkey(lb, cb, md, f, "pooled", "gap_delta_rr")]) for lb in labs], 0)
    X0 = np.mean([0.5 * np.abs(z[PR.npzkey(lb, cb, md, f, "pooled", "gap_sim_minus_run1")]) + 0.5 * np.abs(z[PR.npzkey(lb, cb, md, f, "pooled", "gap_sim_minus_run2")])
                  for lb in labs], 0) - Y
    npz_path = os.path.join(tmp, "synthetic_perdraw.npz")
    np.savez(npz_path, **z)
    npz_sha = hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()
    rec = {"meta": {"window": "full", "plant": {"deltas": [-12, -8, -5, -3, 0, 3, 5, 8, 12], "points": [0.01]}}, "pooled": pooled, "rows": rows,
           "coheadline_signed_plant": coh, "npz": {"volume": PR.VOL, "path": npz_path, "sha256": npz_sha}}
    for i, p in enumerate(probes):  # delta 0 planted arrays equal the unplanted ones (L534-536)
        lb = f"{p} C"
        z0k = PR.npzkey(lb, cb, md, f, "pooled", "gap_delta_full_plant_+0")
        z[z0k] = z[PR.npzkey(lb, cb, md, f, "pooled", "gap_delta_full")]
    np.savez(npz_path, **z)
    rec["npz"]["sha256"] = hashlib.sha256(Path(npz_path).read_bytes()).hexdigest()
    base_rec = {"pooled": bpooled, "coheadline_signed": {cb: {md: {f: {"six_base|C": {"S": PR_summ(S0)}}}}},
                "panel_groups": {cb: {"pooled": {md: {f: {"G|C": {"primary": {"mean_G_minus_mean_Y": PR_summ(X0)}}}}}}}}
    rb, bb = json.dumps(rec).encode(), json.dumps(base_rec).encode()
    cfg["record"] = {"git": "synthetic:rec.json.gz", "decompressed_sha256": hashlib.sha256(rb).hexdigest()}
    cfg["base_record"] = {"git": "synthetic:base.json.gz", "decompressed_sha256": hashlib.sha256(bb).hexdigest()}
    return cfg, gzip.compress(rb), gzip.compress(bb)


def PR_summ(v):
    return PR.summ(np.asarray(v, float))


def selftest():
    """Synthetic only. -> keys and booleans. Any failure raises (the caller STOPs)."""
    e = _eq_selftest
    assert cmp_tree({"a": 1}, {"a": 1}, e)[0] == []
    assert [m[1] for m in cmp_tree({"a": 1}, {"a": 1, "b": 2}, e)[0]] == ["keys"], "a key only on one side MISMATCHES"
    assert [m[1] for m in cmp_tree({"a": {}}, {"a": []}, e)[0]] == ["type"], "{} vs [] MISMATCHES"
    assert [m[1] for m in cmp_tree({"a": {}}, {"a": {"x": 1}}, e)[0]] == ["keys"], "{} vs a non-empty map MISMATCHES"
    assert [m[1] for m in cmp_tree([1, 2], [1], e)[0]] == ["length"], "a list-length change MISMATCHES"
    _, ct = cmp_tree({"a": {}, "b": [], "c": [1, {}]}, {"a": {}, "b": [], "c": [1, {}]}, e)
    assert (ct["n_leaf"], ct["n_empty"]) == (1, 3), "empty containers are compared as values and counted"
    with tempfile.TemporaryDirectory() as tmp:
        cfg, rgz, bgz = _synthetic(tmp)
        assert PR.V3_CTL is None
        o = PR.plant_rows_body(copy.deepcopy(cfg), rgz, bgz)
        assert not o.get("GATE_FAIL"), ("synthetic: the ported body gated", (o.get("gates") or {}).get("failures"))
        r, a = o["rows"]["0.01"]["C"], o["aggregate"]["by_level"]["0.01"]["C"]
        assert r["p3 C"]["-8"]["verdict"] == "detected", "item 24: a clean delta move with an excluding interval is detected"
        assert r["p2 C"]["-8"]["mechanism_mismatch"] == "mechanism mismatch" and r["p2 C"]["-8"]["verdict"] == "not detected", "item 24: move outside 50%"
        assert r["p0 C"]["+8"]["verdict"] == "unavailable" and o["counts"]["0.01"]["+8"]["C"]["unavailable"] == 1, "item 47(2) / 34: unavailable"
        assert r["p1 C"]["-3"]["verdict"] == "unavailable", "item 47(2): a floor-unattainable member row"
        assert a["-3"]["S"]["verdict"] == "unavailable" and a["-3"]["excess"]["verdict"] == "unavailable", "never average over the attainable subset"
        assert a["+8"]["S"]["verdict"] == PR.RULE47_POS and a["+8"]["S"]["unattainable_members"] == ["p0 C"], "item 47(1): the positive side, members named"
        assert a["+3"]["S"]["verdict"].startswith("not reported"), "item 47(1): no positive-side verdict without an unattainable member"
        sx, xx = a["-5"]["S"], a["-5"]["excess"]
        assert sx["move_in_range"] is True and xx.get("move_in_range") is False, "draw-0 move (S) vs bootstrap-mean move (excess) disagree as designed"
        assert r["p3 C"]["-5"]["move_in_range"] is True, "per-probe verdicts follow the draw-0 move"
        tgt = json.loads(json.dumps(o))  # a target equal to V3's own output: the controls must pass on it
        mf = misses_fixture(o["counts"], tgt["counts"], cfg, e)
        assert mf["pass"], ("misses fixture", mf)
        p4 = p4_control(o, tgt, cfg, rgz, bgz, e)
        assert p4["pass"], ("P4 on the synthetic record", p4)
        a4 = a4_control(o, tgt, cfg, rgz, bgz, e)
        assert a4["pass"], ("A4 on the synthetic record", a4)
        assert PR.V3_CTL is None
    return {"pass": True, "checks": 17}


if __name__ == "__main__":  # local: uv run --no-project --with numpy==2.3.3 python plant_rows_v3.py (synthetic only)
    print(json.dumps(selftest()))
