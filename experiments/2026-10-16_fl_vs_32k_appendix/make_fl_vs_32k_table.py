"""The 32k vs full-length appendix RECORD (table JSON; 09 RULE @ 129b7c4f1, lead §3 (ii)). Reads two Stage A records by key; no recomputation.

Per probe and readout (C lead, R beside): the run-mean Δ_full = TPR_sim(τ_sim) − ½[TPR_run1(τ_B) + TPR_run2(τ_B)] at per-action 1e-2 (joint
95% interval, pooled_all_origins: the four-set lead combo, RULE item 33) from the 32k final record and from the full-length record; plus S, Y and the excess. Both on the same no-filter
cohort and draws (gated below, by key); the two intervals side by side; NO difference test (none is computed or stored). Each bar carries
its probe weights' training length from the resolved FIT-spec file. Prints only paths and gate results, never a value.

Also, descriptive (09 @ 2d61af835 §5): the disagreement flag on the six-probe C S of the two builds; each window is labelled with its
build, calibration and probe set (§4). The excess is secondary and always on (A) (§2).

Gates (any failure stops; nothing is written):
  - 09's fixed settings (f 0.01, joint, pooled_all_origins by name, readouts C then R, gap delta_full); every bar's pooled members plus
    withheld members are exactly the four sets' all-origins rows (FOUR_SETS); no '<...>' placeholder left; inputs match sha256;
  - each record's window marker and meta.lead_readout ("C"); the plan's same_in_both keys (cohort list, draw seeds) equal in both records;
  - COHORT by key: every bar's rows[label][origin].cohort_cell_keys per pooled member are identical across both windows, all probes and
    both readouts (one cohort, 09 §7 (iii)); pooled members, withheld members and D_draw0 origins agree;
  - DRAWS: each record's per-draw npz (sha256 from the record) gives every bar's per-origin main weight D in every draw (draw 0 + draws,
    draw 0 equal to D_draw0); the arrays are identical across windows and the draw counts agree. Equal D cannot by itself prove the same
    resampled tasks (the records hold no per-main draw matrix), so the builder's draws_attestation (a sidecar read by path and sha256,
    naming both records' sha256, its pass key true) is also required;
  - every read path is pinned in code (the plan names only the group keys); S/Y member rows come from the statistic's own node; the
    excess is the registered pooled PANEL group's primary row set;
  - S, Y and the excess: each window's member rows equal the plan's expected members (no silent subset), are bars that passed the
    cohort and draw gate, and share their draw count;
  - the lead comparator is the full-length record's meta.estimator.lead, equal to its selection record (never set in the plan); the full-length
    calibration label is read from the record (507-prep's pinned "full length (lead) ..."); if the lead is (B), its S is read from the
    full-length PANEL record and printed beside, labelled, after checking that record's (A) S equals Stage A's;
  - the excess group key comes from each record's meta.window (cut32768 -> off_policy, full -> off_policy_five_full); an empty or unavailable
    excess group stops; where a PANEL record is supplied it is the cited authority: it must name its Stage A record's sha256, and Stage A's
    copy (rate units) must equal it (points) within 1e-9 points, with the same member rows;
  - every bar's training length is resolved (no pending entry). A bar whose pooled row is not readable keeps no number ("not readable"),
    checked before its point (which may be NaN).

  python make_fl_vs_32k_table.py --plan config/plan_<stamp>.json --stamp _final --npz-root /sol [--res-dir results]
"""

import argparse
import datetime
import json
import math
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

from flio import CONTRACT, FOUR_SETS, LABEL, NO_TEST, SENS_ROLE, WINDOWS, fill, get, keystr, load, node, placeholders, sha_bytes, sha_file, stop, write_new

HERE = Path(__file__).resolve().parent


def npzkey(*parts):
    """Stage A's npz key v2 (run_peraction311.py npzkey, copied): '__'-joined; '@' -> '_at_', '|' and ' ' -> '_', parentheses dropped."""
    out = []
    for p_ in parts:
        q = str(p_).replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")
        if "__" in q or not q:
            stop(f"npz key part {p_!r} would be ambiguous")
        out.append(q)
    return "__".join(out)


# The Stage A all-tasks schema (run_peraction311.py @ refresh2), pinned in code: the plan cannot point a read at another mode, rate, combo,
# stratum or comparator. Groups: the plan names only each window's group key; S/Y and their member rows come from the SAME node, and the
# excess is the registered pooled PANEL group (stratum "pooled", the primary row set).
P = {"probe_gap": ["pooled", "{label}", "{combo}", "{mode}", "{f}", "gap", "delta_full"],
     "members": ["pooled", "{label}", "{combo}", "members"],
     "withheld_members": ["pooled", "{label}", "{combo}", "withheld_members"],
     "D_draw0": ["pooled", "{label}", "{combo}", "{mode}", "{f}", "D_draw0"],
     "readable": ["pooled", "{label}", "{combo}", "{mode}", "{f}", "readable", "all"],
     "cells": ["coverage", "{label}", "{member}", "coverage", "cohort_cells"],
     "cohort_keys": ["rows", "{label}", "{member}", "cohort_cell_keys"]}
GROUP_NODE = {"S": ["coheadline_signed", "{combo}", "{mode}", "{f}", "{key}|{ro}"], "Y": ["coheadline_signed", "{combo}", "{mode}", "{f}", "{key}|{ro}"],
              "excess": ["panel_groups", "{combo}", "pooled", "{mode}", "{f}", "{key}|{ro}", "primary"]}
GROUP_FIELD = {"S": "S", "Y": "Y", "excess": "mean_G_minus_mean_Y"}
# The registered excess group (off-policy five) by the record's own meta.window (2e, 10-16 ~03:2x): the 32k-named group is EMPTY in full-length records
EXCESS_KEY = {"cut32768": "off_policy", "full": "off_policy_five_full"}


def check_contract(plan):
    bad = [k for k in ("f", "mode", "combo", "lead_readout") if plan.get(k) != CONTRACT[k]]
    if list(plan.get("readouts", [])) != CONTRACT["readouts"]:
        bad.append("readouts")
    if "paths" in plan or any("paths" in g or "members_paths" in g for g in plan.get("groups", [])):
        bad.append("read paths are pinned in code; the plan may not carry paths")
    if sorted(g.get("stat") for g in plan.get("groups", [])) != sorted(GROUP_FIELD):
        bad.append(f"groups: exactly one each of {sorted(GROUP_FIELD)}")
    for g in plan.get("groups", []):
        if g.get("stat") == "excess":
            if "key" in g:
                bad.append("groups[excess].key: the excess key is chosen by each record's meta.window (EXCESS_KEY), never set in the plan")
        elif set(g.get("key") or {}) != set(WINDOWS) or not all(isinstance(v, str) and v and "|" not in v for v in g["key"].values()):
            bad.append(f"groups[{g.get('stat')}].key: one group key per window, without '|'")
    if not plan.get("same_in_both"):
        bad.append("same_in_both (empty)")
    if set(plan.get("window_markers", {})) != set(WINDOWS):
        bad.append("window_markers (one per window)")
    if "one_cohort_for_all_probes" in plan:
        bad.append("one_cohort_for_all_probes is not configurable: 09 §7 (iii) keeps one cohort for S, Y, the excess and every probe")
    if "lead_comparator" in plan:
        bad.append("lead_comparator is read from the full-length record (meta.estimator.lead), never set in the plan")
    if not all((plan.get("estimator_selection") or {}).get(k) for k in ("file", "sha256", "commit", "source", "lead_path")):
        bad.append("estimator_selection (file, sha256, commit, source, lead_path): the full-length build's selection record")
    bl = plan.get("build_labels") or {}
    if set(bl) != set(WINDOWS) or not all(isinstance(bl[w].get(k), str) and bl[w][k].strip() for w in bl for k in ("build", "probe_set")) \
            or not str(bl.get("cut32k", {}).get("calibration", "")).strip() or "calibration" in bl.get("full", {}):
        bad.append("build_labels: build and probe_set per window, calibration for cut32k only (the full-length one is read from its record)")
    if not (isinstance(plan.get("full_calibration_label_path"), list) and plan["full_calibration_label_path"]):
        bad.append("full_calibration_label_path: the key path of the full-length record's calibration label (507-prep's pinned label)")
    for w, e in (plan.get("excess_authority") or {}).items():
        if w not in WINDOWS or not all(e.get(k) for k in ("file", "sha256", "commit", "source")):
            bad.append(f"excess_authority.{w}: file, sha256, commit and source of that window's PANEL record")
    if not all(plan.get("draws_attestation", {}).get(k) for k in ("file", "sha256", "commit", "pass_path")):
        bad.append("draws_attestation (file, sha256, commit, pass_path): the builder's statement that both records use the same draw streams")
    if bad:
        stop(f"plan departs from 09 §3 (ii)'s fixed settings: {bad}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--stamp", required=True, help="output-name suffix, e.g. _final or _dryrun1")
    ap.add_argument("--res-dir", default=str(HERE / "results"), help="where the records and the lengths file are read and the table is written")
    ap.add_argument("--npz-root", required=True, help="folder the records' npz paths resolve against (the solavg Volume mount, /sol, in Modal)")
    ap.add_argument("--npz-volume", default="", help="if set, each record's npz.volume must name this Volume (the Modal wrapper sets it)")
    ap.add_argument("--git", default="", help="script commit (default: this checkout's HEAD)")
    a = ap.parse_args()
    import numpy as np

    res, nroot = Path(a.res_dir), Path(a.npz_root)
    pb = Path(a.plan).read_bytes()
    plan = json.loads(pb)
    left = placeholders({k: v for k, v in plan.items() if k != "what"})
    if left:
        stop("unfilled plan placeholders:\n  " + "\n  ".join(left))
    check_contract(plan)
    git = a.git or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    recs = {w: load(res / Path(plan["records"][w]["file"]).name, plan["records"][w]["sha256"]) for w in WINDOWS}
    lengths = load(res / Path(plan["lengths"]["file"]).name, plan["lengths"]["sha256"])
    da = plan["draws_attestation"]
    att = load(res / Path(da["file"]).name, da["sha256"])
    F, combo, mode, readouts = plan["f"], plan["combo"], plan["mode"], plan["readouts"]
    lead = get(recs["full"], ["meta", "estimator", "lead"], "full")  # the full-length build picks (A) or (B) by script (09 @ 2d61af835 §1)
    es = plan["estimator_selection"]
    sel = load(res / Path(es["file"]).name, es["sha256"])
    if lead not in ("A", "B") or get(sel, es["lead_path"], "estimator_selection") != lead:
        stop(f"meta.estimator.lead = {lead!r} is not 'A'/'B' or differs from the selection record's {keystr(es['lead_path'])}")
    ptr = get(recs["full"], ["meta", "estimator", "selection_record"], "full")  # the record's own pointer {commit, path, sha256} must be this file
    pc_, ec_ = (str(ptr.get("commit") or "") if isinstance(ptr, dict) else ""), str(es["commit"])
    if not isinstance(ptr, dict) or ptr.get("sha256") != es["sha256"] or Path(str(ptr.get("path", ""))).name != Path(es["file"]).name \
            or not pc_ or not (pc_.startswith(ec_) or ec_.startswith(pc_)):  # commits match when one sha is a prefix of the other
        stop(f"the plan's estimator_selection ({es['file']} @ {es['commit']}) is not the record's meta.estimator.selection_record {ptr!r}")
    calib = get(recs["full"], plan["full_calibration_label_path"], "full")
    if not (isinstance(calib, str) and calib.startswith("full length (lead)")):
        stop(f"the full-length record's calibration label at {keystr(plan['full_calibration_label_path'])} is missing or not 507-prep's pinned label")
    build_labels = {"cut32k": plan["build_labels"]["cut32k"], "full": plan["build_labels"]["full"] | {"calibration": calib}}
    panel = {w: load(res / Path(e["file"]).name, e["sha256"]) for w, e in (plan.get("excess_authority") or {}).items()}
    for w, e in (plan.get("excess_authority") or {}).items():  # the PANEL record's own code commit, when the plan cites one (2e: cite its meta.git)
        if e.get("code") and not str(get(panel[w], ["meta", "git"], f"{w} PANEL")).startswith(str(e["code"])):
            stop(f"{w}: the PANEL record's meta.git is not the cited code commit {e['code']}")

    def panel_bind(w, labels, what):
        """Each row label a PANEL value is built from must come from exactly one of the PANEL record's npz sources, and that source must be this
        window's Stage A record (its record sha256) and npz (path and sha256): a hash mentioned elsewhere in the record does not count."""
        srcs = get(panel[w], ["meta", "config", "npz"], f"{w} PANEL")
        for lab in labels:
            hit = [x for x in srcs if lab in (x.get("labels") or [])]
            if len(hit) != 1 or (hit[0].get("record") or {}).get("sha256") != plan["records"][w]["sha256"] or hit[0].get("sha256") != recs[w]["npz"]["sha256"] \
                    or hit[0].get("path") != recs[w]["npz"]["path"]:
                stop(f"{w} PANEL {what}: row {lab!r} is not read from this window's Stage A record and npz (one source naming both)")

    def panel_set(w):  # PANEL stores its headline combo's results under "pooled", every other combo under its own name
        return "pooled" if get(panel[w], ["meta", "config", "headline_combo"], f"{w} PANEL") == combo else combo

    def to_rate(n):  # PANEL reports points (x100); the table keeps every value in rate units (fractions)
        return n if "point" not in n else {**n, "point": n["point"] / 100, "ci95": None if n["ci95"] is None else [x / 100 for x in n["ci95"]],
                                           "units_source": "PANEL points / 100"}
    gates = {"contract": CONTRACT, "window_markers": {}, "lead_readout": {}, "same_in_both": [], "cohort_by_key": {}, "draws_by_npz": {},
             "group_members": [], "training_lengths": "every bar resolved",
             "one_cohort_for_all_probes": True}

    for w, rec in recs.items():
        wm = plan["window_markers"][w]
        if get(rec, wm["path"], w) != wm["equals"]:
            stop(f"{w}: window marker {keystr(wm['path'])} differs from the plan's")
        gates["window_markers"][w] = {"key": keystr(wm["path"]), "equals": wm["equals"]}
        lr = get(rec, ["meta", "lead_readout"], w)
        if lr != CONTRACT["lead_readout"]:
            stop(f"{w}: meta.lead_readout = {lr!r}, 09's lead is {CONTRACT['lead_readout']!r}")
        gates["lead_readout"][w] = lr
    # Equal per-origin weights D in every draw do not by themselves prove the same resampled tasks (two mains can swap multiplicities), and
    # the records hold no per-main draw matrix. So the draw identity also rests on the builder's attestation, bound to both records by sha256.
    if get(att, da["pass_path"], "draws_attestation") is not True:
        stop(f"draws_attestation {keystr(da['pass_path'])} is not true")
    att_text = json.dumps(att)
    missing_ = [w for w in WINDOWS if plan["records"][w]["sha256"] not in att_text]
    if missing_:
        stop(f"draws_attestation does not name the {missing_} record sha256 (it must cover both records)")
    # The attestation is a SIDECAR written after both records exist (2e, 10-16 ~03:1x), so no record can name it; its binding is that it
    # names both records' sha256 (checked above). Its pass rule (the builder's): same base INPUTS sha, same cohort-list ref, every seed field
    # equal by key path, byte-identical draw code at both commits, and per-row draws_sha256 equal wherever both records carry it.
    gates["draws_attestation"] = {"file": da["file"], "sha256": da["sha256"], "commit": da["commit"], "source": da.get("source"),
                                  "pass_key": keystr(da["pass_path"]), "kind": "sidecar written after both records; names both records' sha256",
                                  "names_both_records": True}
    for path in plan["same_in_both"]:
        v = {w: get(rec, path, w) for w, rec in recs.items()}
        if v["cut32k"] != v["full"]:
            stop(f"{keystr(path)} differs between the 32k and full-length records")
        gates["same_in_both"].append({"key": keystr(path), "value": v["full"]})
    npz = {}
    for w, rec in recs.items():
        meta_ = get(rec, ["npz"], w)
        if a.npz_volume and meta_.get("volume") != a.npz_volume:
            stop(f"{w}: the record's npz is on {meta_.get('volume')!r}, not {a.npz_volume!r}")
        p_ = nroot / meta_["path"]
        if sha_file(p_) != meta_["sha256"]:
            stop(f"{w}: npz {meta_['path']} does not match the record's sha256")
        npz[w] = np.load(p_)
        gates["draws_by_npz"][w] = {"path": meta_["path"], "sha256": meta_["sha256"], "volume": meta_.get("volume")}

    def length_of(b, ro):
        L = (lengths.get("probes") or {}).get(b["length"])
        if not L:
            stop(f"no training-length entry {b['length']!r}")
        if L["status"] != "resolved":
            stop(f"training length of {b['length']!r} is pending: {L.get('why')}")
        if ro not in L["by_readout"]:
            stop(f"training length of {b['length']!r} has no readout {ro}")
        return L["by_readout"][ro] | {"entry": b["length"], "probe_weights": L["probe"]}

    refs = {}  # one scope: every bar of every probe, readout and window shares one cohort and one set of draws (09 §7 (iii))

    def same(scope, sig, where):
        if scope not in refs:
            refs[scope] = (sig, where)
            return
        s0, w0 = refs[scope]
        bad = [k for k in sig if k != "D" and sig[k] != s0[k]]
        bad += [f"D[{o}]" for o in sig["D"] if not np.array_equal(sig["D"][o], s0["D"][o])]
        if bad:
            stop(f"same cohort and draws: {where} differs from {w0} in {bad} (09 §3 (ii): both on the same no-filter cohort and draws)")

    probe_rows, sens, verified = [], set(), {}  # verified: (window, label) -> draws, for every bar that passed the cohort and draw gate
    for ro in readouts:
        for pr in plan["probes"]:
            bars = []
            for b in pr["bars"]:
                w = b["window"]
                if w not in WINDOWS:
                    stop(f"unknown window {w!r}")
                lab = f"{b['label']} {ro}"
                kw = {"label": lab, "combo": combo, "mode": mode, "f": F, "ro": ro}
                where = f"{w} {lab}"
                rec = recs[w]
                members = get(rec, fill(P["members"], **kw), where)
                if not (isinstance(members, list) and members and all(isinstance(m, str) for m in members)):
                    stop(f"{where}: pooled members missing or empty")
                held = get(rec, fill(P["withheld_members"], **kw), where)
                if sorted(set(members) | set(held or {})) != sorted(FOUR_SETS) or len(set(members) | set(held or {})) != len(members) + len(held or {}):
                    stop(f"{where}: {combo} members {members} + withheld {sorted(held or {})} are not exactly the four sets {list(FOUR_SETS)}")
                d0 = get(rec, fill(P["D_draw0"], **kw), where)
                if not isinstance(d0, dict) or set(d0) != set(members):
                    stop(f"{where}: D_draw0 origins {sorted(d0) if isinstance(d0, dict) else d0} != pooled members {members}")
                keys = {}
                for m in members:
                    ck = get(rec, fill(P["cohort_keys"], member=m, **kw), where)
                    n_ = get(rec, fill(P["cells"], member=m, **kw), where)
                    if not (isinstance(ck, list) and ck == sorted(set(ck)) and len(ck) == n_):
                        stop(f"{where} {m}: cohort_cell_keys not a sorted unique list of coverage cohort_cells length")
                    keys[m] = ck
                rd = get(rec, fill(P["readable"], **kw), where)
                if not isinstance(rd, bool):
                    stop(f"{where}: readable flag is not a boolean")
                gpath = fill(P["probe_gap"], **kw)
                if rd:
                    nd = node(rec, gpath, where)
                    draws = nd.get("draws")
                else:  # Stage A: print 'not readable', never the interval; the point may be NaN, so the strict schema is not applied to it
                    raw = get(rec, gpath, where, unavailable_ok=True)
                    draws = raw.get("draws") if isinstance(raw, dict) else None
                    reading = get(rec, fill(P["readable"], **kw)[:-2], where).get("reading", "not readable")
                    nd = {"key": keystr(gpath), "not_readable": str(reading)}
                if not (isinstance(draws, int) and not isinstance(draws, bool) and draws > 0):
                    stop(f"{where}: no draw count on {keystr(gpath)}")
                Dz = {}
                for m in members:
                    k_ = npzkey(lab, combo, mode, F, m, "D")
                    if k_ not in npz[w].files:
                        stop(f"{where}: npz key {k_} missing")
                    Dz[m] = npz[w][k_]
                    if Dz[m].shape != (draws + 1,) or not math.isclose(float(Dz[m][0]), float(d0[m]), rel_tol=1e-12, abs_tol=0.0):
                        stop(f"{where}: npz {k_} has shape {Dz[m].shape} (want draw 0 + {draws} draws) or its draw 0 differs from D_draw0")
                sig = {"members": members, "withheld_members": sorted(held) if isinstance(held, dict) else held, "cohort_keys": keys, "D": Dz,
                       "draws": draws, "D_shape": {m: list(v.shape) for m, v in Dz.items()}}
                same("all", sig, where)
                verified[(w, lab)] = draws
                if b["role"] == SENS_ROLE:
                    sens.add(lab)
                bars.append({"window": w, "window_label": WINDOWS[w], "record_label": lab, "role": b["role"], "gap": "delta_full (run-mean)",
                             "value": nd, "readable": rd, "training_length": length_of(b, ro),
                             "cohort": {"members": members, "withheld_members": sig["withheld_members"],
                                        "cells": {m: len(v) for m, v in keys.items()},
                                        "cell_keys_sha256": {m: sha_bytes("\n".join(v).encode()) for m, v in keys.items()}}})
            probe_rows.append({"readout": ro, "lead": ro == CONTRACT["lead_readout"], "probe": pr["probe"], "bars": bars})
    for scope, (sig, where) in refs.items():
        gates["cohort_by_key"][str(scope)] = {"reference": where, "members": sig["members"], "cells": {m: len(v) for m, v in sig["cohort_keys"].items()},
                                              "cell_keys_sha256": {m: sha_bytes("\n".join(v).encode()) for m, v in sig["cohort_keys"].items()},
                                              "draws": sig["draws"], "D_shape": sig["D_shape"], "pass": True}

    group_rows = []
    for ro in readouts:
        for g in plan["groups"]:
            out = {"readout": ro, "lead": ro == CONTRACT["lead_readout"], "stat": g["stat"], "what": g["what"], "windows": []}
            for w in WINDOWS:
                key_ = EXCESS_KEY[recs[w]["meta"]["window"]] if g["stat"] == "excess" else g["key"][w]
                kw = {"combo": combo, "mode": mode, "f": F, "ro": ro, "key": key_}
                where = f"{w} group {g['stat']} {ro}"
                base = fill(GROUP_NODE[g["stat"]], **kw)
                nd = node(recs[w], base + [GROUP_FIELD[g["stat"]]], where)
                if g["stat"] == "excess" and ("unavailable" in nd or not get(recs[w], base, where, unavailable_ok=True).get("rows")):
                    stop(f"{where}: the excess group {key_}|{ro} is empty or unavailable (fail closed; 2e: pick the key by meta.window)")
                mem = None
                if "unavailable" not in nd:
                    mem = get(recs[w], base + ["rows"], where)  # the member rows of the SAME node as the statistic
                    want = sorted(f"{p} {ro}" for p in g["expect_members"][w])
                    if not isinstance(mem, list) or sorted(mem) != want:
                        stop(f"{where}: member rows {mem} != the plan's expected members {want} (no silent subset)")
                    outside = [m for m in mem if (w, m) not in verified]
                    if outside:
                        stop(f"{where}: members {outside} are not bars that passed the cohort and draw gate")
                    if any(verified[(w, m)] != nd["draws"] for m in mem):
                        stop(f"{where}: the group's draw count differs from its members'")
                    gates["group_members"].append({"group": g["stat"], "readout": ro, "window": w, "members": want, "pass": True})
                out["windows"].append({"window": w, "window_label": WINDOWS[w], "value": nd, "members": mem, "group_key": f"{key_}|{ro}"})
            group_rows.append(out)

    PTS, TOL = 100.0, 1e-9  # PANEL reports points (x100); Stage A rates. PANEL's own gate holds them equal to 1e-12 in rate units.

    def same_scaled(a_, b_):  # a_: Stage A node (rate), b_: PANEL node (points)
        if ("point" in a_) != ("point" in b_):
            return False
        if "point" not in a_:
            return True
        ok = abs(a_["point"] * PTS - b_["point"]) <= TOL
        if (a_["ci95"] is None) != (b_["ci95"] is None):
            return False
        return ok and (a_["ci95"] is None or all(abs(x * PTS - y) <= TOL for x, y in zip(a_["ci95"], b_["ci95"])))

    for g in [g for g in group_rows if g["stat"] == "excess"]:  # the PANEL record is the cited authority where it exists; Stage A's copy must equal it
        for wv in g["windows"]:
            w = wv["window"]
            if w not in panel:
                wv["authority"] = "PANEL record not supplied yet: Stage A's copy (PANEL's build gates the two equal)"
                continue
            base = ["results", panel_set(w), F, "groups", wv["group_key"], "primary"]
            pn = node(panel[w], base + ["mean_G_minus_mean_Y"], f"{w} PANEL excess {g['readout']}")
            prow = get(panel[w], base + ["rows"], f"{w} PANEL excess {g['readout']}", unavailable_ok=True)
            panel_bind(w, wv["members"] or [], f"excess {g['readout']}")
            if not same_scaled(wv["value"], pn) or ("point" in pn and sorted(prow) != sorted(wv["members"] or [])):
                stop(f"{w} excess {g['readout']}: Stage A's copy differs from the PANEL record ({pn['key']}) in value, interval or member rows")
            wv["authority"] = {"record": plan["excess_authority"][w], "key": pn["key"], "units": "points (x100)", "equals_stage_a_copy": True}

    def flag():  # 09 @ 2d61af835 §5, descriptive: opposite signs of the six-probe C S points (comparator (A)), or exactly one interval excludes 0
        S = next(g for g in group_rows if g["stat"] == "S" and g["readout"] == CONTRACT["lead_readout"])
        v = {x["window"]: x["value"] for x in S["windows"]}
        out = {"rule": "flag = true iff the six-probe C S points of the two builds (comparator (A)) have opposite signs, or exactly one of the two "
                       "intervals excludes 0 (09 @ 2d61af835 §5)",
               "label": "descriptive: one interval excluding 0 while the other doesn't is not a test of a difference between the builds",
               "keys": {w: x["key"] for w, x in v.items()}, "goes_to": "Aswin via 0d, whichever way it reads"}
        if any("point" not in x for x in v.values()):
            return out | {"value": None, "why": "an S node is unavailable"}
        p = [v[w]["point"] for w in WINDOWS]
        opposite = (p[0] > 0 > p[1]) or (p[0] < 0 < p[1])
        if any(v[w]["ci95"] is None for w in WINDOWS):
            return out | {"opposite_signs": opposite, "value": True if opposite else None,
                          "why": None if opposite else "an S interval is withheld: the interval part cannot be read"}
        ex = [v[w]["ci95"][0] > 0 or v[w]["ci95"][1] < 0 for w in WINDOWS]
        return out | {"opposite_signs": opposite, "exactly_one_interval_excludes_0": ex[0] != ex[1], "value": opposite or ex[0] != ex[1]}

    flag_ = flag()  # on (A), before any (B) row is added
    if lead == "B":  # 09 @ 2d61af835 §4b: the comparison stays on (A); the lead's (B) S is printed beside, labelled (from the full PANEL record)
        if "full" not in panel:
            stop("the lead is (B): its S is read from the full-length PANEL record (excess_authority.full), which the plan does not name")
        pan = panel["full"]
        hc = get(pan, ["meta", "config", "headline_combo"], "full PANEL")
        cpath = ["coheadline"] if hc == combo else ["coheadline", "other_combos", combo]
        for ro in readouts:
            e = get(pan, cpath + [F, ro, "six"], "full PANEL (B) S")
            panel_bind("full", e.get("rows") or [], f"(B) S {ro}")
            if sorted(e.get("rows") or []) != sorted(next(x for x in next(g for g in group_rows if g["stat"] == "S" and g["readout"] == ro)["windows"]
                                                        if x["window"] == "full")["members"] or []):
                stop(f"full {ro}: the PANEL (B) S members differ from Stage A's six-probe S members")
            if e.get("S_estimator") != "B":
                stop(f"full PANEL record {keystr(cpath + [F, ro, 'six'])}: S_estimator is not B")
            sa = next(g for g in group_rows if g["stat"] == "S" and g["readout"] == ro)["windows"]
            sa = next(x for x in sa if x["window"] == "full")["value"]
            if not same_scaled(sa, node(pan, cpath + [F, ro, "six", "S_A_beside"], "full PANEL S_A_beside")):
                stop(f"full {ro}: the PANEL record's S_A_beside (its (A) S) differs from Stage A's (A) S")
            group_rows.append({"readout": ro, "lead": ro == CONTRACT["lead_readout"], "stat": "S (B)",
                               "what": "the lead's selected (B) all-runs S, printed beside, labelled; not part of the 32k vs full-length comparison (09 @ 2d61af835 §4b)",
                               "windows": [{"window": "full", "window_label": WINDOWS["full"], "value": to_rate(node(pan, cpath + [F, ro, "six", "S"], "full PANEL (B) S")),
                                            "source": {"record": plan["excess_authority"]["full"], "key": keystr(cpath + [F, ro, "six", "S"])}, "members": e.get("rows")}]})
    now = datetime.datetime.now(ZoneInfo("America/Los_Angeles"))
    table = {
        "label": LABEL,
        "what": ("per-probe run-mean Δ_full (separate thresholds, per-action 1e-2, joint 95%, pooled_all_origins: the four-set lead combo, "
                 "RULE item 33 @ ddadd5f15), C lead, R beside, plus S, Y and the "
                 "excess; the first 32k tokens vs full length (no 32k cut); " + NO_TEST),
        "rule": "09 RULE @ 129b7c4f1, lead §3 (ii); computed in this record only (no figure unless the author asks)",
        "no_difference_test": NO_TEST,
        "units": "fractions (rate units) everywhere, as stored in the Stage A records (x100 = percentage points); values read from a PANEL record (points) are divided by 100 and say so",
        "settings": CONTRACT | {"one_cohort_for_all_probes": True},
        "group_keys": {g["stat"]: g.get("key") for g in plan["groups"]} | {"excess": {w: EXCESS_KEY[recs[w]["meta"]["window"]] for w in WINDOWS}},
        "sensitivity_bars": sorted(sens),
        "build_labels": build_labels,
        "lead_comparator": {"lead": lead, "source": "the full-length record's meta.estimator.lead, equal to the selection record's " + keystr(es["lead_path"]),
                            "selection_record": es, "this_comparison": "A (run-mean Δ_full) in both builds (09 @ 2d61af835 §4b)"},
        "disagreement_flag": flag_,
        "inputs": {"plan": {"path": str(Path(a.plan)), "sha256": sha_bytes(pb)}, "records": {w: plan["records"][w] for w in WINDOWS},
                   "lengths": plan["lengths"]},
        "git": git, "stamp (PT)": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "gates": gates,
        "probe_rows": probe_rows,
        "group_rows": group_rows,
    }
    write_new(res / f"fl_vs_32k_table{a.stamp}_{now.strftime('%Y-%m-%d_%H%M%S')}.json", json.dumps(table, indent=1))
    print("gates passed: contract, window markers, lead readout C, same_in_both", len(gates["same_in_both"]), "| cohort by key and draws by npz:",
          len(refs), "scope(s) |", len(gates["group_members"]), "group member checks | training lengths resolved")


if __name__ == "__main__":
    main()
