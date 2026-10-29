"""Negative tests for INPUTS_FINAL: each tampered manifest, NOTE row or S_newmains row must fail closed WITH ITS OWN diagnostic.
Usage: <modal env python> test_inputs_final_neg.py <INPUTS_FINAL.json>. Reads git and the local manifest copies only; no Modal call."""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inputs_final as IF  # noqa: E402
import make_inputs_final as MK  # noqa: E402
import run_peraction311 as R  # noqa: E402

base = json.loads(Path(sys.argv[1]).read_text())
assert not IF.verify(base), "the base manifest must pass"
mem0 = json.loads((HERE / base["sets"]["S_newmains"]["membership710"]["path"]).read_bytes())
sm = base["sets"]["S_newmains"]["s4_source_map"]
smap = json.loads(IF.git_show(sm["commit"], sm["path"]))
staged = [r for r in smap["outside_source_map"] if r["staging"] == "staged"]
o = staged[0]
o2 = next(r for r in staged if any(r["key"] in ks for ks in mem0["folder_keys"].values()))  # an orphan whose 4r row is in the pools
orph = {(r["key"], r["root"]) for r in smap["outside_source_map"]}
fails = []


def expect(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    fails.extend([] if ok else [name])


def tamper(name, f, must):
    """The tampered manifest must fail, and one problem must contain `must` (the intended check, not a side effect)."""
    m = copy.deepcopy(base)
    f(m)
    probs = IF.verify(m)
    expect(f"{name} -> {must!r}", any(must in x for x in probs))


def raises(name, fn, must):
    """fn must stop with a message containing `must` (the intended check)."""
    try:
        fn()
        expect(f"{name} -> {must!r}", False)
    except SystemExit as e:
        expect(f"{name} -> {must!r}", must in str(e))


# ---- S4 orphans: membership resolution (both readers take rows only through it)
mem = copy.deepcopy(mem0)
for ks in mem["folder_keys"].values():
    if o["key"] in ks:
        ks.remove(o["key"])
mem["folder_keys"].setdefault(f"{o['root']}__x", []).append(o["key"])
expect(f"orphan substitution {o['key']} @ {o['root']}", any("owned in the membership" in x for x in IF.s4_orphan_problems(mem, smap)))
mem = copy.deepcopy(mem0)
mem["folder_keys"].setdefault(f"{o2['root']}__x", []).append(o2["key"])
expect(f"orphan double count {o2['key']}", any("two folders" in x for x in IF.s4_orphan_problems(mem, smap)))
expect("clean membership has no S4 problems", not IF.s4_orphan_problems(mem0, smap))
# ---- S4 orphans: the readers' row selection (nm_take is what both the DT and the non-DT S_newmains readers call)
kf0 = {k: f for f, ks in mem0["folder_keys"].items() for k in ks}
own4r = kf0[o2["key"]]
orphan_folder = f"{o2['root']}__{own4r.split('__', 1)[1]}"
mask = R.nm_take([o2["key"]], orphan_folder, kf0, orph, "test (DT reader)")
expect("reader: an orphan row in its orphan folder's DT/score file is not taken", not mask.any())
expect("reader: the same key is taken from its 4r folder", bool(R.nm_take([o2["key"]], own4r, kf0, orph, "test").all()))
kf_bad = dict(kf0, **{o2["key"]: orphan_folder})
raises("reader: a membership that gives an orphan key to its orphan folder stops the job (DT row fed)",
       lambda: R.nm_take([o2["key"]], orphan_folder, kf_bad, orph, "test (DT reader)"), "S4 orphan rows")
raises("reader: a config without s4_orphans stops", lambda: R.nm_take([o2["key"]], own4r, kf0, None, "test"), "no s4_orphans")
# ---- without S4 (09 @ 51a612b29): calibration pools must be equal between the with-S4 and without-S4 rows
cov = {"H_sim_runs": 330, "H_sim_units": 12579, "H_A_runs": 330, "H_A_units": 11659, "H_B_runs": 344, "H_B_units": 12433}
ok_rows = {"FIBS C": {"S_newmains": {"coverage": dict(cov)}, "S_newmains without S4": {"coverage": dict(cov)}}}
expect("without S4: equal pools pass", R.nos4_check(ok_rows)["checked_pairs"] == 1)
bad_rows = {"FIBS C": {"S_newmains": {"coverage": dict(cov)}, "S_newmains without S4": {"coverage": dict(cov, H_B_runs=268, H_B_units=9683)}}}
raises("without S4: S4 dropped from H_B (the 16:13 record's counts) stops", lambda: R.nos4_check(bad_rows), "calibration pools differ")
# ---- orphan .eval sha256: in an input the runner reads (not the orphan root's own folder manifest)
orig = IF.pinned_blobs
IF.pinned_blobs = lambda m: orig(m) + [("experiments/2026-10-15_e2e_sim_pipeline/config/inputs_e2e.json", o["sha256"].encode())]
expect("orphan .eval sha256 in the S_old E2E config", any("orphan .eval sha256" in x for x in IF.verify(base)))
IF.pinned_blobs = orig
expect("pinned_blobs covers the S_old E2E config and membership710's real-arm manifests",
       {base["sets"]["S_old"]["e2e_config"]["path"], "membership710"} <= {n for n, _ in IF.pinned_blobs(base)}
       and any(n.startswith("man710/") for n, _ in IF.pinned_blobs(base)))
# ---- manifest structure
tamper("rms.s4_source_map differs", lambda m: m["rms"].__setitem__("s4_source_map", dict(m["rms"]["s4_source_map"], commit="f3e57dc98")),
       "rms.s4_source_map differs from sets.S_newmains.s4_source_map")
tamper("rms without r36", lambda m: m["rms"].pop("r36"), "rms.r36: missing")
tamper("rms.arm_pin_sources [None]", lambda m: m["rms"].__setitem__("arm_pin_sources", [None]), "rms.arm_pin_sources: must be a non-empty list")
tamper("rms.H_B without S_newmains", lambda m: m["rms"]["H_B"].pop("S_newmains"), "rms.H_B: needs S_new and S_newmains")
tamper("rms.H_sim_roots S_newmains with an S4r root", lambda m: m["rms"]["H_sim_roots"]["S_newmains"].append("simh_nm360_s4r_sim_s1_v1__sim_honest"),
       "rms.H_sim_roots.S_newmains")
tamper("rms.scope reworded", lambda m: m["rms"]["scope"].__setitem__("S_newmains", "S1-S3 + S5"), "rms.scope.S_newmains must be exactly")
tamper("S_newmains s4_source_map missing", lambda m: m["sets"]["S_newmains"].pop("s4_source_map"), "s4_source_map missing")
tamper("gapfill_sim2 None", lambda m: m.__setitem__("gapfill_sim2", None), "gapfill_sim2: status must be")
# epoch 2 release gate, component by component (each manifest carries the other components)
NOTE = {"commit": "d9013b2e7", "path": "experiments/2026-10-14_priority1_spec_09/RULE.md"}
ART = dict(NOTE, sha256=IF.sha(IF.git_show("d9013b2e7", NOTE["path"])))
tamper("epoch2 'in' without 09's verification", lambda m: m["epoch2_honest"].update(status="in", note=NOTE, artifact=ART, verified_09=None),
       "without 09's by-key verification commit")
tamper("epoch2 'in' with a wrong artifact sha", lambda m: m["epoch2_honest"].update(status="in", note=NOTE, artifact=dict(ART, sha256="0" * 64),
                                                                                   verified_09={"commit": "d9013b2e7"}), "pool-key artifact sha256 differs")
m_ = copy.deepcopy(base)
m_["epoch2_honest"].update(status="in", note=NOTE, artifact=ART, verified_09={"commit": "d9013b2e7"})
expect("epoch2 'in' with every gate component: only the runner's refusal remains",
       [x for x in IF.verify(m_) if x.startswith("epoch2")] == ["epoch2_honest: 'in' is not supported by run_peraction311.py (fail closed)"])
tamper("epoch2 old wording", lambda m: m["epoch2_honest"].update(status="not in this build; second refresh"), "epoch2_honest: status must be")
# sel97: two individually valid pins that differ (the membership file in git as a stand-in)
OTHER = {"commit": "0f9d96158", "path": "experiments/2026-10-15_fpr_sweep_507_prep/config/membership710.json"}
OTHER["sha256"] = IF.sha(IF.git_show(OTHER["commit"], OTHER["path"]))
tamper("sel97 vendored copy differs (both pins valid)", lambda m: m["sel97"].__setitem__("cells_vendored", OTHER), "vendored cells copy differs")
tamper("sel97 cells without the vendored copy", lambda m: m["sel97"].pop("cells_vendored"), "pinned together")
# ledger, DT, empty roots
tamper("DT record dropped (root uncovered, not listed pending)", lambda m: m["sets"]["S_new"]["dt_sim_honest"].pop(), "mapped roots without a pinned DT record")
tamper("pending row overlaps excluded", lambda m: m["ledger"]["pending_at_cutoff"].append(dict(m["ledger"]["excluded"][-1])), "excluded and pending at once")
tamper("empty root cell not excluded", lambda m: m["ledger"]["empty_replacement_roots"][0]["cells"].append("t999"), "not excluded by key")
# ---- builder: m_r and class rules
raises("builder: provenance counters disagree (tool_events_final 9, events_by_type 12)",
       lambda: MK.note_mr("k", {"m_r": 9, "m_r_provenance": {"original": {"tool_events_final": 9, "events_by_type_tool": 12}}}), "09's rule: error")
raises("builder: m_r not the max over provenance", lambda: MK.note_mr("k", {"m_r": 40, "m_r_provenance": {"a": {"tool_events_final": 40}, "b": {"tool_events_final": 62}}}),
       "not the max over m_r_provenance")
raises("builder: m_r_inputs counters disagree", lambda: MK.mr_inputs("k", {"m_r": 9, "failed_attempts": {"a": {"value": 9, "tool_events_final": 9, "events_by_type_tool": 12}}}),
       "09's rule: error")
raises("builder: ambiguous prose m_r", lambda: MK.note_mr("k", {"status": "m_r = 40 then m_r = 62"}), "ambiguous or missing")
raises("builder: row m_r differs from m_r_inputs", lambda: MK.note_mr("k", {"m_r": 40}, 62), "differs from the supplied value 62")
expect("builder: explicit 'class unknown' beats 'standing rule'", MK.note_class("k", "standing rule (class unknown, m_r = 23)") == "unknown")
raises("builder: no class in the status text", lambda: MK.note_class("k", "replacement in"), "no class in the status text")
# NOTE rows through the ledger builder (a synthetic NOTE in place of git): relayed m_r disagreement, class field disagreement
_gs = IF.git_show


def with_note(note, extra):
    IF.git_show = lambda c, p_: json.dumps(note).encode() if c == "NOTE" else _gs(c, p_)
    try:
        return MK.ledger_from_notes([{"commit": "NOTE", "path": "n"}], {("S_new", "t1"): "simh311_q1"}, extra)
    finally:
        IF.git_show = _gs


row_ = {"status": "replacement in (RL infrastructure)", "m_r": 9}
rel_ = [{"set": "S_new", "batch": "simh311_q1", "cell_prefix": "t1", "class": "infrastructure", "m_r": 10, "status": "pending", "reason": "r", "ruling": "x"}]
raises("builder: relayed m_r differs from the NOTE row", lambda: with_note({"sensitivities": {"simh_311|t1_x": row_}}, rel_), "differs from the supplied value 10")
raises("builder: class field disagrees with the status text",
       lambda: with_note({"sensitivities": {"simh_311|t1_x": dict(row_, **{"class": "terminal error"})}}, []), "disagrees with the status text")
expect("builder: an agreeing NOTE row builds", with_note({"sensitivities": {"simh_311|t1_x": row_}}, [])["excluded"][0]["m_r"] == 9)
# relay regression (r4): NOTE row and m_r_inputs both 9, relayed 10
raises("builder: relay disagrees with row + m_r_inputs", lambda: with_note({"sensitivities": {"simh_311|t1_x": row_}, "m_r_inputs (09)": {"simh_311|t1_x": {
    "m_r": 9, "failed_attempts": {"original": {"value": 9, "tool_events_final": 9, "events_by_type_tool": 9}}}}}, rel_), "differs")
# ---- S_old DT: exactly one record
for nm_, f_ in (("missing", lambda m: m["sets"]["S_old"].pop("dt_sim_honest")), ("null", lambda m: m["sets"]["S_old"].__setitem__("dt_sim_honest", None)),
                ("empty", lambda m: m["sets"]["S_old"].__setitem__("dt_sim_honest", [])),
                ("duplicate", lambda m: m["sets"]["S_old"].__setitem__("dt_sim_honest", m["sets"]["S_old"]["dt_sim_honest"] * 2))):
    tamper(f"S_old DT {nm_}", f_, "exactly one record is required")
# ---- epoch 2: each release-gate component missing on its own
tamper("epoch2 'in' without the note", lambda m: m["epoch2_honest"].update(status="in", note=None, artifact=ART, verified_09={"commit": "d9013b2e7"}),
       "the Append 38 §4 note is not at its pin")
tamper("epoch2 'in' without the artifact", lambda m: m["epoch2_honest"].update(status="in", note=NOTE, artifact=None, verified_09={"commit": "d9013b2e7"}),
       "the pool-key artifact is not at its pin")
# ---- per-consumer pools (stage_a), the on-policy comparator and its definition
tamper("pools.stage_a H_B n edited", lambda m: m["pools"]["stage_a"]["S_new"]["H_B"].__setitem__("n", 305), "pools.stage_a differs from its recomputation")
tamper("pools.stage_a S_newmains roots without S4r", lambda m: m["pools"]["stage_a"]["S_newmains"].__setitem__(
    "H_sim_roots", [r for r in m["pools"]["stage_a"]["S_newmains"]["H_sim_roots"] if "_s4r_" not in r]), "pools.stage_a differs")
IN_ = base
pr_ok = {"FIBS C": {"S_new": {"pool_runs": {"H_B": {"admitted_runs": base["pools"]["stage_a"]["S_new"]["H_B"]["n"]}}},
                    "S_newmains": {"pool_runs": {"H_B": {"admitted_runs": base["pools"]["stage_a"]["S_newmains"]["H_B"]["n"]}}, "s_newmains": {"s4r_lists_joined": True}}}}
expect("pools_check: matching admitted H_B passes", R.pools_check(pr_ok, IN_)["rows_checked"] == 2)
pr_bad = json.loads(json.dumps(pr_ok))
pr_bad["FIBS C"]["S_newmains"]["s_newmains"]["s4r_lists_joined"] = False
raises("pools_check: S_newmains without the 4r lists must admit the smaller pool", lambda: R.pools_check(pr_bad, IN_), "pools.stage_a mismatch")
pc, note = R.primary_comparator(["own_frac25_L48 C", "own_all10_L48 R", "own_a C", "FIBS C", "Sol (a) R", "DT_transfer_primary C"])
expect("primary_comparator (09 @ 635b14564; 2e 19:4x): own_a leads with sim_minus_run2; own_frac/own_all, DT and others with delta_full",
       pc == {"own_frac25_L48 C": "delta_full", "own_all10_L48 R": "delta_full", "own_a C": "sim_minus_run2", "FIBS C": "delta_full", "Sol (a) R": "delta_full",
              "DT_transfer_primary C": "delta_full"} and set(note) == {"own_frac25_L48 C", "own_all10_L48 R", "own_a C", "DT_transfer_primary C"}
       and "pending confirmation" in note["DT_transfer_primary C"]
       and "sensitivity: run1 shares the training launch" in note["own_all10_L48 R"] and note["own_a C"] == "includes run1 (training arm)")
import re  # noqa: E402

src = (HERE / "run_peraction311.py").read_text()
defs = re.findall(r'"sim_minus_run2": ([^,}\n]+)', src)
expect(f"sim_minus_run2 is sim@S|primary - run2@B at every definition ({len(defs)}), never sim@B",
       len(defs) >= 4 and all(("sim@S|primary" in d_ and "run2@B" in d_) or d_.strip() == "sim - r2" for d_ in defs) and not any("sim@B" in d_ for d_ in defs))
print(json.dumps({"manifest": sys.argv[1], "failed": fails, "pass": not fails}))
sys.exit(1 if fails else 0)
