"""Lean negative tests for the full-length build (no git, no Modal, no score file): each case must STOP or report a problem.
  python test_inputs_full_neg.py"""

import full_override as FO
import inputs_full as IFU

G = {"commit": "c0ffee0", "path": "gates.json", "sha256": "a" * 64}
E = {"path": "cutfull/x.npz", "sha256": "b" * 64, "staged_shas": ["s" * 64], "runs_without_rows": [], "gate": G | {"file": "f1|p2/fibs__command.npz"}}
OV = {"arms": {"f1": {"p2/fibs__command.npz": E}}, "simh": {}}
OK_FLAGS = {"keys_bound_by_task": True, "runs_without_rows_ok": True, "inside_32k_equal": True, "n_tokens_eq_full_len": True}


def case(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return cond


res = []
gate_ok = {"files": {"f1|p2/fibs__command.npz": OK_FLAGS | {"sha256": "b" * 64}}}
res.append(case("clean entry passes", IFU.entry_problems("S_new", "C", OV, G, gate_ok, set(), None)[0] == []))
gate_sha = {"files": {"f1|p2/fibs__command.npz": OK_FLAGS | {"sha256": "c" * 64}}}
res.append(case("gate sha != entry sha is a problem", any("sha256" in x for x in IFU.entry_problems("S_new", "C", OV, G, gate_sha, set(), None)[0])))
gate_na = {"files": {"f1|p2/fibs__command.npz": OK_FLAGS | {"sha256": "b" * 64, "inside_32k_equal": IFU.NA}}}
res.append(case("n/a inside-32k on a registered S_new file is a problem", bool(IFU.entry_problems("S_new", "C", OV, G, gate_na, set(), None)[0])))
res.append(case("staged sha not covering the inventory's is a problem",
                bool(IFU.entry_problems("S_old", "C", OV, G, gate_ok, set(), {"f1": {"t" * 64}})[0])))
PIN = {"commit": "c0ffee0", "path": "x.json", "sha256": "d" * 64}
good = {"schema": IFU.SCHEMA, "status": "FROZEN x", "test": False} | {k: dict(PIN) for k in IFU.required({}) if "." not in k}
good["ood"] = dict(PIN, signature=dict(PIN))
for blk in ("overrides", "gates", "inventories", "fits"):
    good[blk] = {k.split(".", 1)[1]: dict(PIN) for k in IFU.required({}) if k.startswith(blk + ".")}
res.append(case("fully pinned FROZEN fixture has no schema problem", IFU.schema_problems(good) == []))
bad = dict(good, base={"pending": "x"})
res.append(case("FROZEN with a pending base is exactly a base problem", IFU.schema_problems(bad) == ["base: not a pin (FROZEN needs every pin)"]))
g = FO.read_guard(lambda v, p, s: "read", [("conv", "cutfull/x.npz", "b" * 64)])
try:
    g("conv", "cut/x.npz", "b" * 64)
    res.append(case("guard refuses a path outside the allowlist", False))
except SystemExit:
    res.append(case("guard refuses a path outside the allowlist", True))
res.append(case("guard reads an allowed entry", g("conv", "cutfull/x.npz", "b" * 64) == "read"))
keys_ = {"S_new|C": PIN, "S_new+25f|R": PIN, "IL:S_new+x|R": PIN, "pools_ep2+x|C": PIN, "S_typo+x|R": PIN}
kp_ = IFU.key_problems({"overrides": keys_, "gates": dict(keys_, **{"S_old|C": PIN})})
res.append(case("IL:+part, pools_ep2 part, unknown set and a gate-only key are problems; good keys are not",
                sum("not '<set>|<ro>'" in x for x in kp_) == 3 and sum("only one of" in x for x in kp_) == 1))
res.append(case("a fit entry of the wrong kind STOPs", IFU.fit_problems("sol_a_uncut", {"kind": "4f", "files": {}, "fit": PIN, "companion": PIN}, None) != []))
try:
    FO.combine([{"simh": {"f": [{"root": "r1", "path": "a"}]}}, {"simh": {"f": [{"root": "r1", "path": "b"}]}}])
    res.append(case("two overrides' entries for one H_sim root that differ STOP", False))
except SystemExit:
    res.append(case("two overrides' entries for one H_sim root that differ STOP", True))
res.append(case("two overrides' distinct H_sim roots join by root",
                [e["root"] for e in FO.combine([{"simh": {"f": [{"root": "r1"}]}}, {"simh": {"f": [{"root": "r2"}, {"root": "r1"}]}}])["simh"]["f"]] == ["r1", "r2"]))
for nm_, roles_ in (("a config tagging pooled_all as the lead STOPs", {"pooled_all": "lead", "pooled_all_origins": "lead: four sets"}),
                    ("pooled_all as the only lead STOPs", {"pooled_all": "lead", "pooled_455": "lead-2"})):
    try:
        FO.lead_check(roles_)
        res.append(case(nm_, False))
    except SystemExit:
        res.append(case(nm_, True))
res.append(case("pooled_all_origins as the one lead passes (lead-2 is not a lead)",
                FO.lead_check({"pooled_all_origins": "lead: four sets", "pooled_455": "lead-2", "pooled_all": FO.FULL_COMBO_ROLE["pooled_all"]}) == "pooled_all_origins"))
raise SystemExit(0 if all(res) else 1)
