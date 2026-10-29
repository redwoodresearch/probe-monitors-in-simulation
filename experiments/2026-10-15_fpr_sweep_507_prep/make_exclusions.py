"""The second refresh's ONE exclusions file, by script (09 RULE checklist §6; canonical classes §7). Keys, classes and counts only; no score is read.

Every launched cell of every headline set that is NOT in the final cohort, with key, set, failed arm(s), ONE canonical class (the earliest root
cause, in the order launch -> setup -> honest reference -> sim launch -> staging -> scoring) and the sources' local wording beside it. Sources:
the no-filter list (S_old, S_new, S_newmains; make_nofilter_exclusions.py) and the S_paper493 block built from the v2 pins (the v2 manifest
pins this file and checks the block's sha256). Asserts, per set and in
total: final + excluded (+ pending, TEST only) = launched, no overlap, no unclassed row; two sources giving two different ROOT causes STOP
(a wording difference alone does not). Both N: all sets, and Jeffrey's 1,000 view (S_old + S_new + S_paper493). Known drops confirmed by key.

  python make_exclusions.py --nofilter results/<list>.json@<sha> --pins config/<v2 pins>.json --classes config/canonical_classes.json
      [--gap-addendum <commit>:<path>:<sha>] [--test]
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
STAGES = ("launch", "setup", "honest reference", "sim launch", "staging", "scoring")  # §7: earliest root cause wins
PENDING = "set not routed at freeze"  # S_paper493 held sets: under the waitlist rule (@ 07ad588be) these are pending, never excluded


def load(spec):
    p, h = spec.rsplit("@", 1)
    b = (HERE / p).read_bytes()
    if hashlib.sha256(b).hexdigest() != h:
        raise SystemExit(f"{p}: sha256 differs")
    return json.loads(b), {"path": p, "sha256": h}


def from_wording(wording, C):
    """A source's own root cause from its wording alone: every matching pattern, the EARLIEST stage wins (§7); None if nothing matches."""
    hits = []
    for r in C["patterns"]:
        if re.search(r["match"], wording):
            err = re.search(r"\(([A-Za-z]+Error[^)]*)\)", wording)
            hits.append((STAGES.index(r["stage"]), r["class"].replace("{err}", err[1] if err else "error"), r["stage"], r["ruling"]))
    return min(hits)[1:] if hits else (None, None, None)


def canonical(key, wording, C, ev=None):
    """ONE canonical class for a cell: 09's per-key ruling (exact §7 wording) first, else the earliest-stage pattern of the wording. A ruling with
    "decide" is resolved by key from the evidence (honest_a_reference: an admitted honest_a run of the cell in membership710's folders)."""
    pre = key.split("_", 1)[0]
    if pre in C["by_key"]:
        x = C["by_key"][pre]
        if x.get("decide") == "honest_a_reference":
            has_ = bool((ev or {}).get("honest_a", {}).get(key))
            return ((x["class_if_reference"], x["stage_if_reference"], x["ruling"] + ": usable honest_a reference") if has_ else
                    (x["class"], x["stage"], x["ruling"] + ": no usable honest_a reference (no admitted honest_a run, by key)"))
        if x.get("decide") == "canonical_run":  # 09 (via 507-prep 04:1x): the arm's canonical run, by key in the pinned run table, sets the class
            rt = x["run_table"]
            b_ = subprocess.run(["git", "-C", str(HERE), "show", f"{rt['commit']}:{rt['path']}"], capture_output=True, check=True).stdout
            if hashlib.sha256(b_).hexdigest() != rt["sha256"]:
                raise SystemExit(f"{key}: run table sha256")
            row_ = [c_ for c_ in json.loads(b_)["cells"] if c_["cell"] == key]
            got_ = row_[0].get(f"{x['arm']}_launch") if len(row_) == 1 else None
            if got_ not in (x["rerun_launch"], x["original_launch"]):
                raise SystemExit(f"{key}: the run table's {x['arm']}_launch is {got_!r}, neither the rerun nor the original (STOP)")
            rr_ = got_ == x["rerun_launch"]
            return ((x["class"], x["stage"], f"{x['ruling']}: canonical {x['arm']} run = {got_} (rerun; by key, {rt['path'].rsplit('/', 1)[1]} @ {rt['commit']})")
                    if rr_ else (x["class_if_original"], x["stage_if_original"], f"{x['ruling']}: canonical {x['arm']} run = {got_} (original)"))
        return x["class"], x["stage"], x["ruling"]
    return from_wording(wording, C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nofilter", required=True)
    ap.add_argument("--pins", required=True, help="the v2 pins file; S_paper493's block is built from it (inputs_final_v2.build_sp493), so the v2 manifest can pin this file")
    ap.add_argument("--classes", default="config/canonical_classes.json")
    ap.add_argument("--gap-addendum", default="")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--revision-of", default="", help="<path>@<sha256> of the file this one revises CLASS-ONLY: cohort, launched, final, excluded and pending by key must be identical")
    a = ap.parse_args()
    NF, nf_ref = load(a.nofilter)
    C, c_ref = load(a.classes + "@" + hashlib.sha256((HERE / a.classes).read_bytes()).hexdigest())
    sys.path.insert(0, str(HERE))
    import inputs_final_v2 as IF2

    pb = (HERE / a.pins).read_bytes()
    pins = json.loads(pb)
    sp = IF2.build_sp493(pins["S_paper493"], None, set(pins["S_paper493"]["probe_files"]))
    mref = NF["sources"]["membership710"]
    mb_ = (HERE.parent.parent / mref["path"]).read_bytes() if (HERE.parent.parent / mref["path"]).exists() else (HERE / mref["path"]).read_bytes()
    if hashlib.sha256(mb_).hexdigest() != mref["sha256"]:
        raise SystemExit("membership710: sha256 differs from the no-filter list's source")
    mem = json.loads(mb_)
    ev = {"honest_a": {}}
    for f_, ks_ in mem["folder_keys"].items():
        for k_ in ks_:
            if k_.startswith("honest_a|"):
                ev["honest_a"][k_.split("|", 1)[1]] = f_
    held_ = {}  # 09: a held folder's rows are pending (waitlist side), never excluded, never silently final
    for h in C.get("held", []):
        for k_ in mem["folder_keys"].get(h["folder"], []):
            held_.setdefault(h["set"], {})[k_.split("|", 1)[1]] = h
    in_ref = {"pins": a.pins, "pins_sha256": hashlib.sha256(pb).hexdigest(), "sp493_block_sha256": IF2.block_sha(sp)}
    rows, unclassed, final, launched, pending = [], [], {}, {}, {}  # pending: waitlist side (held or not routed), never counted as excluded
    for o, v in NF["origins"].items():  # the older origins: launched = headline + measurement_excluded (the list's own assert, rechecked)
        final[o] = set(v["headline_cells"])
        ex = v["measurement_excluded"]
        launched[o] = final[o] | set(ex)
        if len(launched[o]) != v["launched_n"] or len(final[o]) != v["headline_n"] or final[o] & set(ex):
            raise SystemExit(f"{o}: the no-filter list's counts do not add up")
        for c, x in sorted(ex.items()):
            cl, st, ru = canonical(c, x["class"], C, ev)
            (rows if cl else unclassed).append({"key": c, "set": o, "failed_arm": sorted(x.get("arms", {})), "class": cl, "stage": st, "ruling": ru,
                                                "local_wording": [x["class"]], "sources": [x.get("source", "no-filter list")]})
    held_rows = []
    for o, hh in held_.items():  # every held cell waits (pending), whether the list called it final or excluded
        hc_ = set(hh) & launched.get(o, set())
        final[o] -= hc_
        rows[:] = [r for r in rows if not (r["set"] == o and r["key"] in hc_)]
        pending[o] = sorted(set(pending.get(o, [])) | hc_)
        held_rows += [{"key": c, "set": o, "class": hh[c]["label"], "stage": "held", "ruling": hh[c]["ruling"], "folder": hh[c]["folder"]} for c in sorted(hc_)]
    final["S_paper493"] = set(sp["cohort"]["no_filter"])
    launched["S_paper493"] = set(sp["cells"])
    pending["S_paper493"] = []
    for x in sp["cohort"]["excluded"]:
        if x["class"] == PENDING:
            pending["S_paper493"].append(x["cell"])
            continue
        arm = x["detail"].split("|", 1)[0] if "|" in x["detail"] else "all"
        cl, st, ru = canonical(x["cell"], x["class"] + " " + x["detail"], C)
        (rows if cl else unclassed).append({"key": x["cell"], "set": "S_paper493", "failed_arm": [arm], "class": cl, "stage": st, "ruling": ru,
                                            "local_wording": [x["class"], x["detail"]], "sources": [f"INPUTS_FINAL v2 S_paper493 block ({x['ruling']})"]})
    zr_ = C.get("zero_row_pending", {})  # zero-row runs not yet classed by 09: TEST reports them as excluded with the exact label; a non-TEST build STOPs
    for o, ks in ((o, v) for o, v in zr_.items() if isinstance(v, list)):
        cells_ = {k.split("|", 1)[1]: k for k in ks}
        if not a.test and set(cells_) & (final.get(o, set()) | {r["key"] for r in rows if r["set"] == o}):
            raise SystemExit(f"{o}: zero-row runs without 09's class: {sorted(ks)} (they cannot enter the cohort until 09 classes them)")
        for c, k in cells_.items():
            if c in final.get(o, set()):
                final[o].discard(c)
                rows.append({"key": c, "set": o, "failed_arm": [k.split("|", 1)[0]], "class": zr_["class"], "stage": "scoring", "ruling": zr_["ruling"],
                             "local_wording": [k], "sources": ["zero_row_pending (config/canonical_classes.json)"]})
    for r in [r for r in rows if r["class"] in set(C.get("pending_classes", []))]:  # recoverable causes wait (09 @ 806da4823), never exclude
        rows.remove(r)
        pending.setdefault(r["set"], [])
        pending[r["set"]] = sorted(set(pending[r["set"]]) | {r["key"]})
    if unclassed:
        raise SystemExit(f"{len(unclassed)} excluded cells without a canonical class, e.g. {[(u['set'], u['key'], u['local_wording']) for u in unclassed[:4]]}")
    conflicts, resolved = [], []
    if a.gap_addendum:  # 4f's GAP_CLASS_ADDENDUM: the same cell with a different ROOT cause STOPs; wording differences go to local_wording
        c_, p_, h_ = a.gap_addendum.split(":")
        gb = subprocess.run(["git", "show", f"{c_}:{p_}"], capture_output=True, check=True).stdout
        if hashlib.sha256(gb).hexdigest() != h_:
            raise SystemExit("gap addendum sha256")
        by = {r["key"]: r for r in rows}
        for cls_, keys in ((g["class"], g["cells"]) for g in json.loads(gb)["gaps"].values()):
            for k in keys:
                r = next((r for kk, r in by.items() if kk.split("_", 1)[0] == k.split("_", 1)[0]), None)
                if r is None:
                    continue  # not excluded here (e.g. own_all_rs-only gaps): the waitlist and the coverage asserts own those
                mine = from_wording(" ".join(r["local_wording"]), C)[:2]  # each source's own root cause, normalized independently
                theirs = from_wording(cls_, C)[:2]
                if mine != theirs:  # different ROOT causes: recorded always; a 09 per-key ruling RESOLVES it (listed), otherwise STOP
                    ru_ = C["by_key"].get(k.split("_", 1)[0])
                    (resolved if ru_ else conflicts).append({"key": r["key"], "here": mine, "gap_addendum": theirs, "gap_wording": cls_}
                                                            | ({"resolved_by": ru_["ruling"], "canonical": ru_["class"]} if ru_ else {}))
                else:
                    r["local_wording"].append(cls_)
                    r["sources"].append(f"GAP_CLASS_ADDENDUM @ {c_}")
    if conflicts and not a.test:
        raise SystemExit(f"root-cause conflicts (09 to rule): {conflicts}")
    per = {}
    for o in launched:
        ex = {r["key"] for r in rows if r["set"] == o}
        pe = set(pending.get(o, []))
        if final[o] & ex or final[o] & pe or ex & pe or final[o] | ex | pe != launched[o]:
            raise SystemExit(f"{o}: final + excluded + pending != launched, or an overlap")
        if pe and not a.test:
            raise SystemExit(f"{o}: {len(pe)} cells pending (sets not routed): the final build waits for them (waitlist rule @ 07ad588be)")
        per[o] = {"launched": len(launched[o]), "final": len(final[o]), "excluded": len(ex), "pending": len(pe),
                  "excluded_by_class": {k: sum(1 for r in rows if r["set"] == o and r["class"] == k) for k in sorted({r["class"] for r in rows if r["set"] == o})}}
    tot = lambda os_: {k: sum(per[o][k] for o in os_) for k in ("launched", "final", "excluded", "pending")}  # noqa: E731
    exp_ = C.get("expected_counts", {})
    for o, n in (exp_.get("excluded") or {}).items():  # asserted by script (0d's recheck): the older origins' exclusion counts
        if per[o]["excluded"] != n:
            raise SystemExit(f"{o}: {per[o]['excluded']} excluded, expected {n}")
    cohort_keys = {o: {"cells": sorted(final[o]), "sha256": hashlib.sha256(json.dumps(sorted(final[o]), separators=(",", ":")).encode()).hexdigest()} for o in final}
    by_stage = {st: sum(1 for r in rows if r["stage"] == st) for st in STAGES}
    SC = C["structural"]
    for r in rows:
        if C["by_key"].get(r["key"].split("_", 1)[0], {}).get("side"):  # e.g. t238: the non-canonical attempt's cause beside the class
            r["side"] = C["by_key"][r["key"].split("_", 1)[0]]["side"]
        r["structural"] = True if r["class"] in SC["structural"] else False if r["class"] in SC["not_structural"] or r["class"].startswith("errored run (") else None
    by_class = {o: {k: sum(1 for r in rows if r["set"] == o and r["class"] == k) for k in sorted({r["class"] for r in rows if r["set"] == o})} for o in per}
    by_struct = {o: {"structural": sum(1 for r in rows if r["set"] == o and r["structural"] is True),
                     "not_structural": sum(1 for r in rows if r["set"] == o and r["structural"] is False),
                     "unresolved": sum(1 for r in rows if r["set"] == o and r["structural"] is None)} for o in per}
    szr = {o: dict(v["structural_zero_runs"]) for o, v in NF["origins"].items()}
    szr["S_paper493"] = {k: {"readouts": [ro for ro, ks in sp["structural_zero_keys"].items() if k in ks]} for k in sorted({k for ks in sp["structural_zero_keys"].values() for k in ks})}
    origins = {o: {"launched_n": per[o]["launched"], "headline_n": per[o]["final"], "headline_cells": sorted(final[o]),
                   "measurement_excluded": {r["key"]: {"class": r["class"], "stage": r["stage"], "structural": r["structural"], "failed_arm": r["failed_arm"],
                                                       "local_wording": r["local_wording"], "ruling": r["ruling"]} for r in rows if r["set"] == o},
                   "structural_zero_runs": szr.get(o, {}), "pending": sorted(pending.get(o, []))} for o in per}
    known = {}
    for pre, want in C["known_drops"].items():  # confirmed by key: each to its class (t619 / t699: recoverable, pending re-stage)
        hit = [r for r in rows if r["key"].split("_", 1)[0] == pre]
        pend = [c for o in pending for c in pending[o] if c.split("_", 1)[0] == pre]
        fin_ = [o for o in final for c in final[o] if c.split("_", 1)[0] == pre]
        known[pre] = ({"set": hit[0]["set"], "class": hit[0]["class"]} if hit else {"pending": pend} if pend else {"status": "final (recovered)", "set": fin_[0]}
                      if fin_ else {"status": "NOT FOUND"}) | ({"expected": want} if want else {})
    out = {"label": "second-refresh exclusions (09 RULE §6, canonical classes §7); keys and classes only; PRELIMINARY" + ("; TEST" if a.test else ""),
           "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "generator": "make_exclusions.py",
           "sources": {"nofilter": nf_ref, "S_paper493": in_ref, "classes": c_ref, "gap_addendum": a.gap_addendum or None},
           "rule": "final + excluded = launched per set and in total; one canonical class per cell (earliest root cause: " + " -> ".join(STAGES) + ")",
           "per_set": per, "N_all_sets": tot(list(per)), "N_jeffrey_1000": tot(["S_old", "S_new", "S_paper493"]), "excluded_by_stage": by_stage,
           "excluded_by_class": by_class, "excluded_structural": by_struct, "held": held_rows, "origins": origins,
           "final_cohort_keys": cohort_keys,
           "known_drops": known, "root_cause_conflicts": conflicts, "root_cause_conflicts_resolved_by_ruling": resolved, "rows": sorted(rows, key=lambda r: (r["set"], r["key"])), "pending": {o: sorted(v) for o, v in pending.items() if v}}
    if a.revision_of:  # a class-only revision: every key set identical; only class / stage / structural / ruling / wording may change
        P0, p0_ref = load(a.revision_of)
        if set(P0["per_set"]) != set(per) or set(P0["final_cohort_keys"]) != set(out["final_cohort_keys"]):
            raise SystemExit(f"revision of {a.revision_of}: the set names differ")
        for o in per:
            if P0["final_cohort_keys"][o]["cells"] != out["final_cohort_keys"][o]["cells"] or P0["per_set"][o]["launched"] != per[o]["launched"] \
                    or sorted(P0["origins"][o]["measurement_excluded"]) != sorted(out["origins"][o]["measurement_excluded"]) \
                    or P0["origins"][o]["pending"] != out["origins"][o]["pending"]:
                raise SystemExit(f"revision of {a.revision_of}: {o}'s cohort / launched / excluded / pending keys differ (a class-only revision may not)")
        old_ = {(r["set"], r["key"]): r for r in P0["rows"]}
        out["revision"] = {"of": p0_ref, "rule": "class-only: cohort, launched, final, excluded and pending identical by key (asserted)",
                           "changed": [{"set": r["set"], "key": r["key"], "from": [old_[(r["set"], r["key"])][x] for x in ("class", "stage", "structural")],
                                        "to": [r[x] for x in ("class", "stage", "structural")]} for r in out["rows"]
                                       if [old_[(r["set"], r["key"])][x] for x in ("class", "stage", "structural")] != [r[x] for x in ("class", "stage", "structural")]]}
    if out["N_jeffrey_1000"]["launched"] != 1000:
        raise SystemExit(f"Jeffrey's view: {out['N_jeffrey_1000']['launched']} launched cells, not 1,000")
    bad_known = [k for k, v in known.items() if v.get("status") == "NOT FOUND" or (v.get("expected") and v.get("class") and v["expected"] != v["class"])]
    if bad_known and not a.test:
        raise SystemExit(f"known drops not confirmed: {bad_known}")
    dest = HERE / "results" / f"exclusions{'_TEST' if a.test else ''}_{out['stamp_pt']}.json"
    blob = json.dumps(out, indent=1).encode()
    dest.open("xb").write(blob)
    print(json.dumps({"file": str(dest.relative_to(HERE)), "sha256": hashlib.sha256(blob).hexdigest(), "per_set": per, "N_all_sets": out["N_all_sets"],
                      "N_jeffrey_1000": out["N_jeffrey_1000"], "known_drops": known, "known_unconfirmed": bad_known,
                      "root_cause_conflicts": conflicts}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
