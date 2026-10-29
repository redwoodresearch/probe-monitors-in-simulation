"""Synthetic tests for variant_arms.resolve (RULE.md Amendment 19): one passing case and the stops. $0, local, no data.

  uv run --no-project --python 3.12 python experiments/2026-10-13_q3_sce_reply_probe/test_variant_arms.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variant_arms import resolve  # noqa: E402

M = {"base run": "B", "base arm": "sim-attack-fx-full", "override run": "O", "override arm": "sim-attack-fx-fullF9", "override keys": ["k1", "k2"]}
base = {f"k{i}": {} for i in range(1, 6)}
over = {"k1": {}, "k2": {}}


def stops(mapping, b, o, expect, why, says):
    """resolve() must stop with the expected reason (its message names the failed check)."""
    try:
        resolve(mapping, b, o, expect)
    except ValueError as e:
        if says not in str(e):
            raise AssertionError(f"stopped for another reason ({e}), expected: {why}")
        return
    raise AssertionError(f"did not stop: {why}")


r = resolve(M, base, over, 5)
assert len(r) == 5 and r["k1"] == ("O", "sim-attack-fx-fullF9") and r["k3"] == ("B", "sim-attack-fx-full")
stops(M, base, over, 6, "base count differs from the expected total", "the base arm holds")
stops(M | {"override keys": ["k1", "k9"]}, base, {"k1": {}, "k9": {}}, 5, "an override key missing from the base arm", "missing: 1 from the base arm")
stops(M | {"override keys": ["k1", "k2", "k3"]}, base, over, 5, "an override key missing from the override arm", "0 from the base arm, 1 from the override arm")
stops(M | {"override keys": ["k1", "k1"]}, base, {"k1": {}}, 5, "a repeated override key", "override keys repeat")
stops(M, base, over | {"k4": {}}, 5, "an unlisted run in the override arm", "outside the listed keys")
stops({k: v for k, v in M.items() if k != "base arm"}, base, over, 5, "a missing mapping field", "mapping fields")
stops(M | {"extra": 1}, base, over, 5, "an unexpected mapping field", "mapping fields")
print("variant_arms: 1 passing case and 7 stops, each for its own reason")

# resolve_sources (the fix-full / N / F9 job's arm resolution)
from variant_arms import resolve_sources  # noqa: E402

SA = [f"a{i}" for i in range(6)]
metas = {("SIM", "sim-attack"): {k: {} for k in SA}, ("FF", "ff-arm"): {k: {} for k in SA[:4]}, ("F9", "f9-arm"): {"a1": {}, "a2": {}}}
V = {"sim-attack-fx-full": {"of": "sim-attack", "run": "FF", "meta_arm": "ff-arm", "n_extracted": 4, "fallback": ["SIM", "sim-attack"]},
     "sim-attack-F9": {"of": "sim-attack", "mapping": {"base run": "FF", "base arm": "sim-attack-fx-full", "override run": "F9",
                                                      "override arm": "f9-arm", "override keys": ["a1", "a2"]}}}
out = resolve_sources(V, metas, {"sim-attack": SA})
assert out["sim-attack-fx-full"]["a0"] == ("FF", "ff-arm") and out["sim-attack-fx-full"]["a5"] == ("SIM", "sim-attack")
assert out["sim-attack-F9"]["a1"] == ("F9", "f9-arm") and out["sim-attack-F9"]["a0"] == ("FF", "ff-arm") and out["sim-attack-F9"]["a5"] == ("SIM", "sim-attack")


def stops_src(v, m, why, says):
    try:
        resolve_sources(v, m, {"sim-attack": SA})
    except ValueError as e:
        if says not in str(e):
            raise AssertionError(f"stopped for another reason ({e}), expected: {why}")
        return
    raise AssertionError(f"did not stop: {why}")


stops_src(V | {"sim-attack-fx-full": V["sim-attack-fx-full"] | {"n_extracted": 5}}, metas, "a dump's count differs from its record", "its record")
stops_src(V | {"sim-attack-fx-full": V["sim-attack-fx-full"] | {"n_extracted": 5}}, metas | {("FF", "ff-arm"): {k: {} for k in SA[:4] + ["zz"]}},
          "a dump run outside the cohort", "outside the cohort")
stops_src(V, metas | {("SIM", "sim-attack"): {k: {} for k in SA[:5]}}, "an unchanged run with no as-is reading", "no as-is reading")
stops_src(V | {"sim-attack-F9": {"of": "sim-attack", "mapping": V["sim-attack-F9"]["mapping"] | {"base run": "XX"}}}, metas, "the mapping's base is not the resolved arm", "not a resolved variant arm")
stops_src(V, metas | {("F9", "f9-arm"): {"a1": {}}}, "an F9 override key missing from the F9 dump", "from the override arm")
print("resolve_sources: 1 passing case and 5 stops, each for its own reason")

# newline_reading (Amendment 17)
from variant_arms import newline_reading  # noqa: E402

assert newline_reading((-0.05, 0.05), (-0.40, -0.20), (0.35, 0.35, 0.70)) == "newlines are the main cue"
assert newline_reading((0.30, 0.45), (-0.05, 0.05), (0.70, 0.30, 0.70)) == "newlines aren't enough alone"
assert newline_reading((-0.20, 0.40), (-0.40, 0.20), (0.50, 0.40, 0.60)) == "inconclusive"  # Codex's wide-interval case
assert newline_reading((-0.05, 0.08), (-0.25, 0.02), (0.50, 0.45, 0.65)) == "partial"  # 1 fails (interval reaches 0), 2 and 3 fail, N between
assert newline_reading((-0.05, 0.05), (-0.05, 0.08), (0.40, 0.40, 0.42)) == "newlines aren't enough alone"  # 1 fails (N not below as-is), 2 holds
assert newline_reading((0.11, 0.30), (-0.25, -0.11), (0.90, 0.40, 0.70)) == "inconclusive"  # outside both and not between
assert newline_reading((0.2219, 0.3055), (-0.1323, -0.0801), (0.6528, 0.3863, 0.7599)) == "partial"  # Seed (c_N), 1%: narrow intervals outside +-10
print("newline_reading: 7 cases (the 6 the amendment lists, plus Seed's (c_N) regression)")
