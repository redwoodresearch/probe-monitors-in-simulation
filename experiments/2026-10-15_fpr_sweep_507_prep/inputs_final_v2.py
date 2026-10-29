"""INPUTS_FINAL v2: the second refresh's one frozen inputs file (09 @ e3a459bd0 item 5). Build and verify (fail closed).

Schema inputs_final/v2 = v1's keys (inputs_final.py) + "refresh2". Every v1 block is byte-identical (as JSON values) to the base v1 FROZEN file
(INPUTS_FINAL_2026-10-15_203213 @ 32abe4120, sha256 d8be69e6…) unless refresh2.registered_fixes lists its key with a ruling; the v1 verifier
runs on the v1 view (base extensions). New:
  refresh2                    {base {commit, path, sha256}, registered_fixes [{key, ruling}], pins {commit, path, sha256} (the v2 pins file), test bool}
  extensions.S_paper493       "not in this build; second refresh" | the S_paper493 block (SP_KEYS below), status "in"
  extensions.own_all_rs_L48   {status "in" | "not in this build", fit_stamp, pointer {fold_list, score*, census*, fit, verify, ...}, rules}

The S_paper493 block, all by key (run key = "<world>|<cell>"):
  origin_mode         "separate" (09 item 1; the switch's other value "merged into S_new" is refused by the runner)
  facts, names_record {commit, path, sha256}; routing [{commit, path, sha256, staged_set, rev}] (d9's highest revision per staged set)
  cells               {cell: {main, env, kind P_new | P_rep | P_dup, main_origin, r36_main_fold}} from the facts file (493 = 37 + 456 + 0)
  launched_universe   {world: [cells with a routed key]} for run1, run2, sim, honest_a, honest_b (routed sets only; held sets in not_in_build)
  cohort              {no_filter [cells], registered [cells] | null, rule, excluded [{cell, class, detail, ruling}], counts_by_class}
  folders             [{arm, staged_set, world, keys, refused_keys, files {probe file: {path, sha256, vol}}}] (real arms, sim attack, honest A/B)
  simh                [{root, batch, world, keys, files}] (g4 sp493_t196, g5 sp493_new)
  pools               {H_B, H_B_existing, H_B_new, H_B_n, H_sim_new, P_rep_mains {t196, p311}, H_sim_existing_rule, H_A_rule}
  pool_exclusions     [{key, class, detail, ruling}] (the pins' list + the paired-pool scan's, 09 RULE item 26)   P_dup [cells]   not_in_build [{staged_set, worlds, cells, keys, reason}]
  families_pending    {probe file: reason}   score_records {family: [pins]}   draws {own [20261015, 39], all_origins [20261015, 40], D, strata}
  heldout_gates       {famexp | own_a: "passed @ <sha>"} (Append 39 §5 for probes trained without #36 rows; a family without one is WITHHELD)
  structural_zero_keys {C: [run keys], R: [run keys]} (attack runs with no rows at that readout and sample_error null: not caught; 09's zero-unit rule)

  python inputs_final_v2.py build --pins config/inputs_final_v2_pins.json [--status "FROZEN <PT>" | --test]
  python inputs_final_v2.py verify <INPUTS_FINAL_v2.json>
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inputs_final as IF  # noqa: E402

SCHEMA = "inputs_final/v2"
BASE = ("32abe4120", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FINAL_2026-10-15_203213.json")
SP_OFF = "not in this build; second refresh"
SP_KEYS = ("status", "origin_mode", "label", "facts", "names_record", "routing", "cells", "launched_universe", "cohort", "folders", "simh", "pools",
           "pool_exclusions", "P_dup", "not_in_build", "families_pending", "score_records", "draws", "heldout_gates", "simh_zero_unit_keys",
           "structural_zero_evidence", "structural_zero_keys")
DRAWS = {"own": [20261015, 39], "all_origins": [20261015, 40], "D": 10000, "strata": "env (main level)"}
NOT_ROUTED = "set not routed at freeze"
PAIRED = "H_sim cell without an admitted H_A member (paired-pool rule)"  # 09 RULE item 26 @ 3fc5cbce5
FAMILY_KINDS = ("p2", "sol", "famexp", "own_a", "own_all_rs")
# world of a folder by its staged-set kind and meta arm (Amendment 45 §3: main __attack -> run1, run2 -> run2, sim -> sim; g4/g5 by group)
WORLD = {("main", "attack"): "run1", ("main", "honest_a"): "honest_a", ("main", "honest_b"): "honest_b", ("run2", "attack"): "run2",
         ("sim", "sim_attack"): "sim", ("simh_g4", "sim_honest"): "sp493_t196", ("simh_g5", "sim_honest"): "sp493_new"}


def block_sha(block):
    """sha256 of a JSON block in canonical form (sorted keys, no spaces): ties a file built from the same pins to the manifest's block."""
    return IF.sha(json.dumps(block, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


CALIB_LABEL_32K = ("32k record (pre-registered first read): registered epoch-1 calibration; the epoch 1+2 calibration applies to the full-length lead "
                   "(@ 64f00b09e)")  # 09's wording, exact (scope cut 02:10)


HEADLINE_FILES = {"sol/a_tight__cmd_A.npz", "sol/a_tight__avg_top2.npz"} | {f"p2/{x}__{r}.npz" for x in ("fibs", "harmfulness", "refusal", "highstakes")
                                                                         for r in ("command", "reply_mean")} | {f"own_all_rs/own_all_rs_L48__{r}.npz" for r in "CR"}
# the 12 headline labels' S_paper493 files (Sol (a), FIBS, harmfulness, refusal, high-stakes, own_all_rs_L48 x C, R): pending ones STOP a non-TEST build (09)


def completion_meta(m):
    """The record's completion fields from a v2 (or full-length) inputs file, exact names (PANEL matches them by string). Reusable."""
    r = m["refresh2"]
    return {"cohort_rule": "no-filter (09 @ 01012e87e)", "completion_rule": "waitlist (09 RULE @ 07ad588be)", "waitlist": r.get("waitlist"),
            "waitlist_pending": r.get("waitlist_pending"), "exclusions": r.get("exclusions"), "fold_minus1_drops": r.get("fold_minus1_drops")}


def waitlist_check(waitlist, exclusions, test, census=None):
    """09 RULE @ 07ad588be: the final waits for every in-flight cell on the committed waitlist. make_waitlist.py check (the waitlist fork) exits 1
    and prints "pending" on any unresolved cell: STOP. Returns the pending list ([] when resolved)."""
    import subprocess as sp_

    tool = HERE / "make_waitlist.py"
    if not tool.exists():
        if test:
            return None
        raise SystemExit("make_waitlist.py is not in this tree (merge the waitlist fork): the final build cannot check its waitlist")
    rel = lambda e: f"{e['path'].removeprefix('experiments/2026-10-15_fpr_sweep_507_prep/')}@{e['sha256']}"  # noqa: E731  (make_waitlist.py: paths in its folder)
    r = sp_.run([sys.executable, str(tool), "check", "--waitlist", rel(waitlist), "--exclusions", rel(exclusions)] + (["--census", rel(census)] if census else []) + ([] if test else ["--final"]),
                capture_output=True, text=True, cwd=HERE)
    try:
        pend = json.loads(r.stdout)["pending"]
    except (ValueError, KeyError):
        raise SystemExit(f"make_waitlist.py check: unreadable output (rc {r.returncode}): {r.stdout[-300:]} {r.stderr[-300:]}")
    if (r.returncode != 0 or pend) and not test:
        raise SystemExit(f"waitlist: {len(pend)} unresolved cells (the final waits): {pend[:5]}")
    return pend  # TEST: recorded (the runner refuses a non-TEST file whose waitlist_pending is not [])


def pin(commit, path):
    return {"commit": commit, "path": path, "sha256": IF.sha(IF.git_show(commit, path))}


def load_pin(e, what):
    b = IF.git_show(e["commit"], e["path"])
    if IF.sha(b) != e["sha256"]:
        raise SystemExit(f"{what}: sha256 of {e['path']} @ {e['commit']} differs")
    return json.loads(b)


def set_kind(staged_set):
    """sp493 staged-set kind from its name (09's grammar @ 9cb3d0291): main / run2 / sim / simh_g4 / simh_g5."""
    m = re.match(r"^sp493_(?:[abc]_(main|run2)|g([12367])_sim(?:_[xy])?|g([45])_simh)(?:_s[0-9]+)?_v1(?:_try[0-9]+)?$", staged_set)  # x/y: regex v2 @ b9e0b6eda
    if not m:
        raise SystemExit(f"{staged_set}: not an sp493 staged-set name")
    return m[1] or ("sim" if m[2] else f"simh_g{m[3]}")


def refused_names(rec):
    """d9's names record: every staged set and folder listed as superseded / burned / superseded v1 launch / non-routable (A45c §1)."""
    out = set()
    for k in ("superseded", "burned_or_taken", "superseded_v1_launches"):
        for x in rec.get(k, []):
            out.add(x["staged_set"]) if isinstance(x, dict) and "staged_set" in x else out.add(str(x))
            out |= set(x.get("folders", [])) if isinstance(x, dict) else set()
    out |= set((rec.get("non_routable") or {}).get("names", []))
    return out


def record_files(rec, folder):
    """{file name: sha256} of `folder`'s .npz files in a scorer's sealed record: a node whose "arm" is the folder (files, or outputs.files: Sol and
    own_a), or a map entry keyed by the folder or by a path with the folder as one component (p2: "/scaleup/<folder>/p2"); None if absent;
    two different entries for the folder in one record STOP."""
    def files_of(v):
        f = v.get("files") if isinstance(v.get("files"), dict) else (v.get("outputs") or {}).get("files") if isinstance(v.get("outputs"), dict) else None
        return {Path(fn).name: (h if isinstance(h, str) else h["sha256"]) for fn, h in f.items() if fn.endswith(".npz")} if isinstance(f, dict) else None

    hits, stack = [], [rec]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if x.get("arm") == folder and files_of(x) is not None and x.get("published", True) is not False:  # a refused (unpublished) arm is no file
                hits.append(files_of(x))
            for k, v in x.items():
                if isinstance(v, dict) and folder in str(k).strip("/").split("/") and files_of(v) is not None:
                    hits.append(files_of(v))
                stack.append(v)
        elif isinstance(x, list):
            stack += x
    if len({json.dumps(h, sort_keys=True) for h in hits}) > 1:
        raise SystemExit(f"{folder}: two different file lists in one scorer record")
    return hits[0] if hits else None


def manifest_files(rec_list, folder, man_dir, suffix):
    """Manifest-style families: the folder's manifest sha256 in the score records (own_all: results[*].folders[f].manifest_sha256; famexp:
    result.manifests[f].sha256), and the local manifest copy <folder>__<suffix>.json's files (sha-checked); None if no record names the folder."""
    shas = {}
    for rec in rec_list:
        for v in (rec.get("results") or {}).values():
            for f, fi in v.get("folders", {}).items():
                shas.setdefault(f, set()).add(fi["manifest_sha256"])
        for f, fi in ((rec.get("result") or {}).get("manifests") or {}).items():
            shas.setdefault(f, set()).add(fi["sha256"])
    if folder not in shas:
        return None
    if len(shas[folder]) != 1:
        raise SystemExit(f"{folder}: two manifest sha256s across the score records")
    cp = f"{man_dir}/{folder}__{suffix}.json"
    if IF.CP.check(cp) or IF.sha(IF.CP.read(cp)) != next(iter(shas[folder])):
        raise SystemExit(f"{folder}: local manifest copy {cp} missing, outside the copies index, or its sha256 differs from the score record")
    fl = json.loads(IF.CP.read(cp))["files"]
    return {Path(fn).name: (h if isinstance(h, str) else h["sha256"]) for fn, h in fl.items() if fn.endswith(".npz")}


def build_sp493(sp, base, labels_new):
    """The S_paper493 block from the v2 pins' S_paper493 section (see config/inputs_final_v2_pins.json). Deterministic: verify() rebuilds and compares."""
    facts = load_pin(sp["facts"], "facts")
    cells = {}
    for r in facts["rows"]:
        kind = "P_new" if r["kind"] == "new_main" else "P_dup" if "dup" in r["kind"] else "P_rep"
        cells[r["key"]] = {"main": r["main"], "env": r["env"], "kind": kind, "main_origin": r["main_origin"], "r36_main_fold": r.get("r36_main_fold")}
    nk = {k: sum(v["kind"] == k for v in cells.values()) for k in ("P_new", "P_rep", "P_dup")}
    if len(cells) != 493 or nk != {"P_new": 37, "P_rep": 456, "P_dup": 0}:
        raise SystemExit(f"facts: {len(cells)} cells, kinds {nk} (Append 39: 493 = 37 + 456 + 0)")
    refused = refused_names(load_pin(sp["names_record"], "names record"))
    excl = {x.get("key") or x["cell"]: x for x in sp["cohort_exclusions"]}  # by run key ("run2|t615_...") or by cell ("t987_...": every arm)
    pex = {x["key"]: x for x in sp["pool_exclusions"]}
    if len(excl) != len(sp["cohort_exclusions"]) or len(pex) != len(sp["pool_exclusions"]):
        raise SystemExit("exclusions: a key listed twice")
    for k in excl:  # a cell, or <run1|run2|sim>|<cell>, of the facts file
        w_, c_ = k.split("|", 1) if "|" in k else (None, k)
        if c_ not in cells or w_ not in (None, "run1", "run2", "sim"):
            raise SystemExit(f"cohort exclusion {k}: not a facts cell / attack world")
    for k in pex:
        w_, c_ = k.split("|", 1)
        if w_ not in ("honest_a", "honest_b", "sp493_t196", "sp493_new") or c_ not in cells:
            raise SystemExit(f"pool exclusion {k}: not a facts cell / pool world")
    t196 = {c["key"]: c["main"] for c in json.loads(IF.git_show(*sp["t196_cells"][:2]))["cells"]}
    p311 = {c["key"]: c["main"] for c in json.loads(IF.git_show(*sp["p311_cells"][:2]))["cells"]}
    for nm, e in (("t196_cells", sp["t196_cells"]), ("p311_cells", sp["p311_cells"])):
        if IF.sha(IF.git_show(*e[:2])) != e[2]:
            raise SystemExit(f"{nm}: sha256")
    rep_t = sorted({v["main"] for v in cells.values() if v["kind"] == "P_rep" and v["main_origin"] == "t196"})
    rep_p = sorted({v["main"] for v in cells.values() if v["kind"] == "P_rep" and v["main_origin"] == "p311"})
    g4_ok = {c for c, m in t196.items() if m in set(rep_t)}  # A45b: g4's T196 keys = one per t196-origin S_paper493 main
    if len(g4_ok) != len(rep_t):
        raise SystemExit(f"g4 admission: {len(g4_ok)} T196 keys for {len(rep_t)} t196-origin mains (expected one each)")
    folders, simh, routing, seen = [], [], [], {}
    for e in sp["routing"]:  # d9's routing + by-key coverage records (SEALED; every manifest revision carries sample_error)
        rr = load_pin(e, "routing")
        want_ = set(e.get("sets") or [])  # the staged sets taken from this record; every other sp493 set in it must be refused by the names record
        for st in rr["sets"]:
            ss = st["staged_set"]
            if not ss.startswith("sp493_"):
                continue  # the records also hold il-s4rgap-t2
            if want_ and ss not in want_:
                if ss not in refused and ss not in set(sp.get("refused_sets", [])):
                    raise SystemExit(f"routing {e['path']}: {ss} is neither taken nor refused by the names record (no silent drop)")
                continue
            if ss in refused or ss in set(sp.get("refused_sets", [])):
                raise SystemExit(f"routing: {ss} is superseded, burned or non-routable (d9's names record or the pins' refused_sets)")
            k_ = set_kind(ss)
            mp_, mc_ = (x.strip() for x in st["manifest_revision"].split(" @ "))
            if IF.sha(IF.git_show(mc_, mp_)) != st["revision_sha256"]:
                raise SystemExit(f"{ss}: sample_error manifest revision sha256 differs")
            arms = {}
            for k in st["staged"]:
                ad_, c = k.split("|", 1)
                arms.setdefault(ad_.split("@", 1)[0], []).append(c)
            if {a_: len(v) for a_, v in arms.items()} != {a_.split("@", 1)[0]: n for a_, n in st["epochs"].items()} or len(st["staged"]) != st["staged_keys"]:
                raise SystemExit(f"{ss}: staged keys do not match the record's epochs")
            routing.append({k_: v_ for k_, v_ in e.items() if k_ != "sets"} | {"staged_set": ss, "batch": st["batch"], "manifest_revision": st["manifest_revision"],
                                                                             "revision_sha256": st["revision_sha256"]})
            exr = {}
            for x in st["excluded"]:
                ad_, c = x["key"].split("|", 1)
                exr.setdefault(ad_.split("@", 1)[0], {})[c] = x
            for arm in sorted(set(arms) | set(exr)):
                w = WORLD[(k_, arm)]
                folder = f"{ss}__{arm}"
                if folder in refused:
                    raise SystemExit(f"routing: folder {folder} is refused in d9's names record")
                cs = arms.get(arm, [])
                ok_ = all((c in g4_ok) if w == "sp493_t196" else (c in cells) for c in cs)
                if not ok_:
                    raise SystemExit(f"{folder}: a key outside the facts file's cells (g4: outside the admitted T196 keys, A45b)")
                keys = sorted(f"{w}|{c}" for c in cs if f"{w}|{c}" not in excl and c not in excl)
                refk = sorted(f"{w}|{c}" for c in cs if f"{w}|{c}" in excl or c in excl)
                for k in keys:
                    if seen.setdefault(k, folder) != folder:  # one admitted set per cell x world (A45c §3)
                        raise SystemExit(f"{k}: routed by two folders ({seen[k]}, {folder})")
                ent = {"arm": folder, "staged_set": ss, "world": w, "keys": keys, "refused_keys": refk,
                       "staging_excluded": {f"{w}|{c}": {"reason": x["staging_reason"], "sample_error": x["sample_error"]} for c, x in sorted(exr.get(arm, {}).items())}}
                if k_.startswith("simh"):
                    simh.append({"root": folder, "batch": f"sp493_g{k_[-1]}"} | ent)
                else:
                    folders.append(ent)
    # score files per probe file name ("<kind>/<file>"), per folder, from the scorers' sealed records (families_pending where a record has no entry)
    recs = {fam: [load_pin(e, f"{fam} score record") for e in v["records"]] for fam, v in sp["families"].items()}
    idx_pending = {}
    for fam, v in sp["families"].items():  # a scorer's index (one row per routing-authority folder): landed rows' records join, pending rows are listed
        for ix in v.get("indexes", []):
            I = load_pin(ix, f"{fam} index")
            for row in I["folders"]:
                if str(row.get("status", "")).startswith("landed"):
                    r_ = load_pin(row["record"], f"{fam} index record {row['folder']}")
                    if r_.get("arm") != row["folder"]:
                        raise SystemExit(f"{fam} index: {row['folder']}'s record is for {r_.get('arm')}")
                    recs[fam].append(r_)
                else:
                    idx_pending.setdefault(fam, []).append(row["folder"])
        for cv in v.get("coverage", []):  # 4f's per-family coverage listing: covered folders' records join (deduplicated), pending folders are listed
            L_ = load_pin(cv["pin"], f"{fam} coverage listing")
            seen_r = set()
            for folder_, ent in L_["families"][cv["family"]].items():
                if ent.get("status") != "covered":
                    idx_pending.setdefault(fam, []).append(folder_)
                    continue
                rp_, rc_ = (x.strip() for x in ent["record"].split(" @ "))
                if (rc_, rp_) not in seen_r:
                    seen_r.add((rc_, rp_))
                    recs[fam].append(load_pin({"commit": rc_, "path": rp_, "sha256": ent["record_sha256"]}, f"{fam} coverage record {folder_}"))
            idx_pending.setdefault(fam, []).extend(x["folder"] for x in L_.get("pending", []) if x["family"] == cv["family"])
    pending = {}
    for ent in folders + simh:
        ent["files"] = {}
        for fam, v in sp["families"].items():
            if v.get("manifest_copies"):
                fl = manifest_files(recs[fam], ent["arm"], v["manifest_copies"], v.get("copy_suffix", "own_all"))
            else:
                fl = next((x for x in (record_files(r, ent["arm"]) for r in recs[fam]) if x is not None), None)
            tmpl = v["simh_dir"] if "batch" in ent else v["scaleup_dir"]
            for fn, h in (fl or {}).items():
                ent["files"][f"{v['kind']}/{fn}"] = {"path": tmpl.format(folder=ent["arm"], batch=ent.get("batch", "")) + f"/{fn}", "sha256": h, "vol": v.get("vol", "conv")}
    for pn in sorted(labels_new):
        miss = [x["arm"] for x in folders + simh if pn not in x["files"]]
        if miss:
            fam_ = next((f for f, v in sp["families"].items() if v["kind"] == pn.split("/", 1)[0]), None)
            ip_ = sorted(set(miss) & set(idx_pending.get(fam_, [])))
            pending[pn] = f"no score file in {len(miss)} S_paper493 folder(s) at freeze: {sorted(miss)}" + (f"; pending in the {fam_} index: {ip_}" if ip_ else "")
    # held sets: not routed at freeze -> not in this build, by key ("set not routed at freeze"), never silently dropped (09 item 4)
    nib = []
    for h in sp["held_sets"]:
        if any(r["staged_set"] == h["staged_set"] for r in routing):
            raise SystemExit(f"held set {h['staged_set']} is also routed")
        if not h["keys"] or any(k.split("|", 1)[0] not in h["worlds"] or k.split("|", 1)[1] not in cells for k in h["keys"]):
            raise SystemExit(f"held set {h['staged_set']}: keys must be <world>|<facts cell> with a listed world")
        nib.append({"staged_set": h["staged_set"], "worlds": h["worlds"], "cells": sorted({k.split("|", 1)[1] for k in h["keys"]}),
                    "keys": sorted(h["keys"]), "reason": NOT_ROUTED, "source": h["source"]})
    held_keys = {k for x in nib for k in x["keys"]}
    nl_ = {x["key"]: x for x in sp.get("never_launched", [])}  # d9's by-key never-launched arms (a terminal class, with its ruling)
    by_w = {w: {k.split("|", 1)[1] for x in folders for k in x["keys"] if x["world"] == w} for w in ("run1", "run2", "sim", "honest_a", "honest_b")}
    launched = {w: sorted({k.split("|", 1)[1] for x in folders if x["world"] == w for k in x["keys"] + x["refused_keys"] + list(x["staging_excluded"])})
                for w in by_w}
    # no-filter cohort (09 @ 01012e87e; e3a459bd0 item 3): run1, run2 and sim all routed and admitted; every other cell gets one class, by key
    coh, excluded = [], []
    for c in sorted(cells):
        cls = None
        if c in excl:
            cls = (excl[c]["class"], excl[c].get("detail", ""), excl[c]["ruling"])
        else:
            for w in ("run1", "run2", "sim"):
                k = f"{w}|{c}"
                if k in excl:
                    cls = (excl[k]["class"], f"{k}: {excl[k].get('detail', '')}", excl[k]["ruling"])
                elif c in by_w[w]:
                    continue
                elif any(k in x["staging_excluded"] for x in folders):
                    sx_ = next(x["staging_excluded"][k] for x in folders if k in x["staging_excluded"])
                    cls = (f"errored run ({sx_['sample_error']})" if sx_["sample_error"] else "staging exclusion", f"{k}: {sx_['reason']}", "d9's routing record")
                elif k in held_keys:
                    cls = (NOT_ROUTED, k, "09 @ e3a459bd0 item 4")
                elif k in nl_:
                    cls = ("never launched", f"{k}: {nl_[k]['reason']}", nl_[k]["ruling"])
                else:  # 09 RULE @ 07ad588be: not routed yet = in flight; it waits (pending), never an exclusion
                    cls = (NOT_ROUTED, k, "no routed or held set holds this key yet (pending; the final waits)")
                break
        if cls is None:
            coh.append(c)
        else:
            excluded.append({"cell": c, "class": cls[0], "detail": cls[1], "ruling": cls[2]})
    counts = {}
    for x in excluded:
        counts[x["class"]] = counts.get(x["class"], 0) + 1
    # pools (Append 39 §4; 09 item 2): H_B = honest_b of S_paper493's mains, existing (T196 / +311 FINAL pool keys) and new; H_sim new = g4 + g5
    fin = json.loads(IF.git_show(*IF.FINAL_REC))
    po, pn_ = (set(IF.resolve(fin, IF.POOL_REF + x)) for x in (".pool_keys_old", ".pool_keys_new"))
    hb_ex = sorted(k for k in po | pn_ if k.startswith("honest_b|") and ((t196.get(k.split("|", 1)[1]) in rep_t) or (p311.get(k.split("|", 1)[1]) in rep_p)))
    hb_new = sorted(k for x in folders if x["world"] == "honest_b" for k in x["keys"])
    hb = sorted((set(hb_ex) | set(hb_new)) - set(pex))
    hs_new = sorted({k for x in simh for k in x["keys"]} - set(pex))
    # 09 RULE item 26 @ 3fc5cbce5 (paired-pool rule; Append 39 §4 stands): H_A(P) is honest_a on H_sim(P)'s cells, so a g4 / g5 key whose ruled
    # H_A member is absent or not admitted leaves H_sim(P) by key (S_paper493 only). Member: honest_a on the key's cell unless 09 named a split;
    # admitted: FINAL's registered epoch-1 honest_a (its AuB pools) or S_paper493's honest_a folders, minus pool exclusions. The scan must find
    # exactly the pinned list (else STOP); the existing roots stay under the runner's per-row check.
    pp = sp.get("paired_pool") or {}
    if set(pp) != {"split", "expected_excluded", "ruling"} or not all(k in hs_new and m.startswith("honest_a|") and m.split("|", 1)[1] in cells
                                                                   for k, m in pp["split"].items()):
        raise SystemExit("pins: S_paper493.paired_pool must be {split {g4/g5 key: honest_a|<facts cell>}, expected_excluded, ruling} (09 RULE item 26)")
    fa_ = {frozenset(k for k in r["schemes"][r["primary"]]["pool_keys_old"] if k.startswith("honest_a|"))
           for n, r in fin["result"]["calibration_records"]["primary"].items() if n.endswith("|AuB|AuB|AuB")}
    if len(fa_) != 1:
        raise SystemExit(f"FINAL: {len(fa_)} different S_old honest_a sets over its AuB records (expected one)")
    adm_a = (set(next(iter(fa_))) | {k for x in folders if x["world"] == "honest_a" for k in x["keys"]}) - set(pex)
    paired = []
    for k in hs_new:
        m_ = pp["split"].get(k, "honest_a|" + k.split("|", 1)[1])
        if m_ not in adm_a:
            paired.append({"key": k, "class": PAIRED, "detail": f"ruled H_A member {m_}: {'a pool exclusion' if m_ in pex else 'no admitted run'}", "ruling": pp["ruling"]})
    if [x["key"] for x in paired] != sorted(pp["expected_excluded"]):
        raise SystemExit(f"paired-pool scan: {[x['key'] for x in paired]} != the pinned list {sorted(pp['expected_excluded'])} (STOP; tell 09)")
    pex |= {x["key"]: x for x in paired}
    hs_new = sorted(set(hs_new) - set(pex))
    pools = {"H_B": hb, "H_B_existing": hb_ex, "H_B_new": hb_new, "H_B_n": len(hb), "H_sim_new": hs_new, "P_rep_mains": {"t196": rep_t, "p311": rep_p},
             "H_sim_existing_rule": ("the existing admitted sim-honest roots on P_rep mains: A25 primary_v3 for T196 mains; S_new's mapped roots with S_new's "
                                     "exclusions and replacements for +311 mains (resolved by key in each row: pool_runs)"),
             "H_A_rule": "honest_a on H_sim(P)'s cells (by key, per probe)"}
    reg = None
    if sp.get("registered_cohort"):  # the attempt-filtered appendix cohort: cells of the no-filter cohort the pinned label list keeps
        rl = load_pin(sp["registered_cohort"]["pin"], "registered cohort")
        keep = set(IF.resolve(rl, sp["registered_cohort"]["key"]))
        reg = [c for c in coh if c in keep]
    return {"status": "in", "origin_mode": sp["origin_mode"], "label": sp["label"], "facts": sp["facts"], "names_record": sp["names_record"],
            "routing": routing, "cells": cells, "launched_universe": launched,
            "cohort": {"no_filter": coh, "registered": reg, "rule": ("09 @ 01012e87e + e3a459bd0 item 3: every S_paper493 cell whose run1, run2 and sim "
                                                                     "were routed and admitted (scored: each row checks its files by key); exclusions only "
                                                                     "for missing or invalid measurement, by key and class; no attempt-label filter"),
                       "excluded": excluded, "counts_by_class": counts},
            "folders": folders, "simh": simh, "pools": pools, "pool_exclusions": sorted(pex.values(), key=lambda x: x["key"]),
            "P_dup": sorted(c for c, v in cells.items() if v["kind"] == "P_dup"), "not_in_build": nib, "families_pending": pending,
            "score_records": {fam: v["records"] + [{"index": ix} for ix in v.get("indexes", [])] + [{"coverage": cv} for cv in v.get("coverage", [])]
                              for fam, v in sp["families"].items()}, "draws": DRAWS,
            "heldout_gates": {fam: {"pin": g_["pin"], "key": g_["key"], "ruling": g_["ruling"],  # 09's ruling names the field that certifies the by-key gate
                                    "value": IF.resolve(load_pin(g_["pin"], f"{fam} held-out gate"), g_["key"])}
                              for fam, g_ in sp.get("heldout_gates", {}).items() if g_.get("ruling")},
            "simh_zero_unit_keys": {r: sorted(v) for r, v in sp.get("simh_zero_unit_keys", {"C": [], "R": []}).items()},
            "structural_zero_evidence": {k: load_pin(e, f"structural zero {k}") and e for k, e in sp.get("structural_zero_evidence", {}).items()}, "structural_zero_keys": {r: sorted(v) for r, v in sp.get("structural_zero_keys", {"C": [], "R": []}).items()}}


def build(pins_path, status, test):
    pb = (HERE / pins_path).read_bytes()
    pins = json.loads(pb)
    base = load_pin(pins["base"], "base v1 INPUTS_FINAL")
    if base["schema"] != "inputs_final/v1" or not base["status"].startswith("FROZEN"):
        raise SystemExit("base: not a FROZEN v1 file")
    m = {k: base[k] for k in IF.KEYS}
    for fx in pins.get("registered_fixes", []):
        m[fx["key"]] = fx["value"]
    m["schema"], m["status"] = SCHEMA, status
    ext = dict(base["extensions"])
    labels_new = set(pins["S_paper493"]["probe_files"]) if pins["S_paper493"].get("status") == "in" else set()
    ext["S_paper493"] = build_sp493(pins["S_paper493"], base, labels_new) if pins["S_paper493"].get("status") == "in" else SP_OFF
    ext["own_all_rs_L48"] = pins["own_all_rs_L48"]
    m["extensions"] = ext
    m["refresh2"] = {"base": pins["base"], "registered_fixes": [{k: v for k, v in fx.items() if k != "value"} for fx in pins.get("registered_fixes", [])],
                     "pins": {"path": pins_path, "sha256": IF.sha(pb)}, "test": bool(test)}
    m["refresh2"] |= completion(pins, ext["S_paper493"], test)
    if pins.get("m0"):  # the pre-census manifest the census and the fold -1 drops file were built on (freeze order M0 -> census -> M1)
        m["refresh2"]["m0"] = pins["m0"]
        bad_ = m0_problems(m)
        if bad_:
            raise SystemExit(f"M0: {bad_}")
    return m


def m0_problems(m):
    """M1 must equal its M0 in every block except status and refresh2 (the census / drops built on M0 then describe M1's inputs exactly)."""
    e = m["refresh2"]["m0"]
    try:
        b = (HERE / e["path"]).read_bytes()
        if IF.sha(b) != e["sha256"]:
            return ["refresh2.m0: sha256 differs"]
        M0 = json.loads(b)
    except (OSError, KeyError, TypeError, ValueError) as x:
        return [f"refresh2.m0: unreadable ({x})"]
    return [f"{k}: differs from M0" for k in set(m) | set(M0) if k not in ("status", "refresh2") and
            json.dumps(m.get(k), sort_keys=True) != json.dumps(M0.get(k), sort_keys=True)]


def completion(pins, block, test):
    """The waitlist rule (09 RULE @ 07ad588be) and the §6 exclusions file: both pinned; the exclusions file must be built from this block (sha256 of
    its canonical form); make_waitlist.py check must report no pending cell. TEST builds may omit both (recorded as not checked)."""
    wl, ex, ce, fd = pins.get("waitlist"), pins.get("exclusions"), pins.get("census"), pins.get("fold_minus1_drops")
    if isinstance(block, dict) and not test:  # 09: a lead label must cover every S_paper493 cohort cell x arm: pending headline files STOP, never withhold
        hp_ = sorted(set(block["families_pending"]) & HEADLINE_FILES)
        if hp_:
            raise SystemExit(f"headline probe files pending on S_paper493 (a non-TEST build STOPs): { {k: block['families_pending'][k] for k in hp_} }")
    if fd:
        load_pin(fd, "fold -1 drops file")
    elif not test:
        raise SystemExit("a non-TEST build needs the fold_minus1_drops pin (PANEL's Sol (a) / c_asis drop evidence)")
    if ex:  # whenever pinned: built from this very S_paper493 block
        exj = load_pin(ex, "exclusions file")
        if isinstance(block, dict) and exj["sources"]["S_paper493"]["sp493_block_sha256"] != block_sha(block):
            raise SystemExit("exclusions file: built from a different S_paper493 block")
    if not (wl and ex and ce):
        if not test:
            raise SystemExit("a non-TEST build needs the waitlist, exclusions and census pins (the census resolves scored cells)")
        return {"waitlist": wl, "exclusions": ex, "census": ce, "fold_minus1_drops": fd, "waitlist_pending": "not checked (TEST)"}
    load_pin(wl, "waitlist")
    load_pin(ce, "census")
    pend = waitlist_check(wl, ex, test, ce)
    return {"waitlist": wl, "exclusions": ex, "census": ce, "fold_minus1_drops": fd,
            "waitlist_pending": pend if pend is not None else "not checked (TEST: make_waitlist.py not in this tree)"}


def verify(m, pins_path=None):
    """Problems (empty = pass): schema, base equality block by block (listed fixes apart), the v1 verifier on the v1 view, the S_paper493 block
    rebuilt from its pins and compared, the own_all_rs slot's pins re-hashed and its fold list = the committed assignment."""
    bad = []
    if set(m) != set(IF.KEYS) | {"refresh2"} or m.get("schema") != SCHEMA:
        return [f"keys/schema: {sorted(set(m) ^ (set(IF.KEYS) | {'refresh2'}))}, schema {m.get('schema')}"]
    try:
        base = load_pin(m["refresh2"]["base"], "base")
    except (SystemExit, KeyError, TypeError) as e:
        return [f"refresh2.base: {e}"]
    if (m["refresh2"]["base"]["commit"], m["refresh2"]["base"]["path"]) != BASE:
        bad.append("refresh2.base is not INPUTS_FINAL_2026-10-15_203213 @ 32abe4120")
    fixed = {fx["key"] for fx in m["refresh2"]["registered_fixes"]}
    if pins_path:
        pins_ = json.loads((HERE / pins_path).read_bytes())
        want_fx = [{k: v for k, v in fx.items() if k != "value"} for fx in pins_.get("registered_fixes", [])]
        if m["refresh2"]["registered_fixes"] != want_fx or any(json.dumps(m[fx["key"]], sort_keys=True) != json.dumps(fx["value"], sort_keys=True)
                                                               for fx in pins_.get("registered_fixes", [])):
            bad.append("refresh2.registered_fixes: differ from the pins (keys, rulings or values)")
        if json.dumps(m["extensions"].get("own_all_rs_L48"), sort_keys=True) != json.dumps(pins_["own_all_rs_L48"], sort_keys=True):
            bad.append("extensions.own_all_rs_L48: differs from the pins")
        if tuple(m["refresh2"].get(k) for k in ("waitlist", "exclusions", "census", "fold_minus1_drops")) != tuple(pins_.get(k) for k in ("waitlist", "exclusions", "census", "fold_minus1_drops")):
            bad.append("refresh2.waitlist / exclusions / census / fold_minus1_drops: differ from the pins")
        elif pins_.get("exclusions") and isinstance(m["extensions"].get("S_paper493"), dict):
            try:
                if load_pin(pins_["exclusions"], "exclusions")["sources"]["S_paper493"]["sp493_block_sha256"] != block_sha(m["extensions"]["S_paper493"]):
                    bad.append("refresh2.exclusions: built from a different S_paper493 block")
            except (SystemExit, KeyError) as e:
                bad.append(f"refresh2.exclusions: {e}")
        for k_ in ("waitlist", "exclusions", "census", "fold_minus1_drops"):  # required in a non-TEST file, re-hashed whenever present
            e_ = m["refresh2"].get(k_)
            if e_:
                try:
                    load_pin(e_, k_)
                except (SystemExit, KeyError, TypeError) as x_:
                    bad.append(f"refresh2.{k_}: {x_}")
            elif not m["refresh2"].get("test"):
                bad.append(f"refresh2.{k_}: missing in a non-TEST file")
        if m["refresh2"].get("m0") != pins_.get("m0"):
            bad.append("refresh2.m0: differs from the pins")
        if not m["refresh2"].get("test") and m["refresh2"].get("waitlist_pending") != []:
            bad.append("refresh2.waitlist_pending: a non-TEST file must record [] (the final waits for every listed cell)")
    for k in IF.KEYS:
        if k in ("schema", "status", "extensions") or k in fixed:
            continue
        if json.dumps(m[k], sort_keys=True) != json.dumps(base[k], sort_keys=True):
            bad.append(f"{k}: differs from the base v1 file and is not a listed registered fix")
    for k, v in base["extensions"].items():
        if k != "S_paper493" and json.dumps(m["extensions"].get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
            bad.append(f"extensions.{k}: differs from the base v1 file")
    v1 = {k: m[k] for k in IF.KEYS} | {"schema": "inputs_final/v1", "extensions": base["extensions"]}
    bad += [f"v1 view: {x}" for x in IF.verify(v1)]
    sp = m["extensions"].get("S_paper493")
    if sp != SP_OFF:
        if not isinstance(sp, dict) or tuple(sp) != SP_KEYS or sp["status"] != "in":
            bad.append("extensions.S_paper493: not the block (SP_KEYS, status 'in') nor the 'not in this build' string")
        elif pins_path:
            pins = json.loads((HERE / pins_path).read_bytes())
            if IF.sha((HERE / pins_path).read_bytes()) != m["refresh2"]["pins"]["sha256"]:
                bad.append("refresh2.pins: sha256 differs")
            try:
                rb = build_sp493(pins["S_paper493"], base, set(pins["S_paper493"]["probe_files"]))
                if json.dumps(rb, sort_keys=True) != json.dumps(sp, sort_keys=True):
                    bad.append("extensions.S_paper493: differs from its rebuild from the pins")
            except SystemExit as e:
                bad.append(f"extensions.S_paper493 rebuild: {e}")
        else:
            bad.append("extensions.S_paper493: verify needs the pins file (rebuild-and-compare)")
        if isinstance(sp, dict) and sp.get("draws") != DRAWS:
            bad.append("extensions.S_paper493.draws: not Append 39 am.1 §5's streams")
    if m["refresh2"].get("m0"):  # independent of the pins path
        bad += m0_problems(m)
    rs = m["extensions"].get("own_all_rs_L48") or {}
    if rs.get("status") == "in":
        pt = rs.get("pointer") or {}
        for k, e in pt.items():
            try:
                load_pin(e, f"own_all_rs pointer {k}")
            except (SystemExit, KeyError, TypeError) as x:
                bad.append(f"extensions.own_all_rs_L48.pointer.{k}: {x}")
        fl = pt.get("fold_list") or {}
        if (fl.get("commit"), fl.get("path"), fl.get("sha256")) != ("ec56720ad", "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json",
                                                                    "c2fc5d40eb84a34a64cbb9ce7155572d1414f12ea1fbc2ec05708bd73061e299"):
            bad.append("extensions.own_all_rs_L48: the fold list is not the committed assignment @ ec56720ad (09 3a (b))")
        if not any(k.startswith("score") for k in pt) or not rs.get("fit_stamp") or not {"fit", "verify"} <= set(pt):
            bad.append("extensions.own_all_rs_L48: 'in' without a score record, fit_stamp, fit or verify pin")
    elif rs.get("status") != "not in this build":
        bad.append("extensions.own_all_rs_L48: status must be 'in' or 'not in this build'")
    return bad


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pins", default="config/inputs_final_v2_pins.json")
    b.add_argument("--status", default="DRAFT (unfrozen)")
    b.add_argument("--test", action="store_true", help="TEST manifest (stub or partial inputs; values never read)")
    v = sub.add_parser("verify")
    v.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "verify":
        m = json.loads((HERE / a.path).read_bytes())
        bad = verify(m, m["refresh2"]["pins"]["path"])
        print(json.dumps({"problems": bad, "pass": not bad}, indent=1))
        sys.exit(1 if bad else 0)
    m = build(a.pins, ("TEST " if a.test else "") + a.status, a.test)
    bad = verify(m, a.pins)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    tag = "TEST" if a.test else "FROZEN" if a.status.startswith("FROZEN") else "DRAFT"
    dest = HERE / "config" / f"INPUTS_FINAL_V2_{tag}_{stamp}.json"
    blob = json.dumps(m, indent=1, ensure_ascii=False).encode()
    dest.open("xb").write(blob)
    sp = m["extensions"]["S_paper493"]
    summ = {"file": str(dest.relative_to(HERE)), "sha256": IF.sha(blob), "problems": bad}
    if isinstance(sp, dict):
        summ |= {"cohort_no_filter": len(sp["cohort"]["no_filter"]), "excluded_by_class": sp["cohort"]["counts_by_class"], "folders": len(sp["folders"]),
                 "simh_roots": len(sp["simh"]), "H_B": sp["pools"]["H_B_n"], "H_sim_new": len(sp["pools"]["H_sim_new"]), "P_dup": len(sp["P_dup"]),
                 "not_in_build": [(x["staged_set"], len(x["keys"])) for x in sp["not_in_build"]], "families_pending": len(sp["families_pending"])}
    print(json.dumps(summ, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
