"""IL extra attack arms (sim2, run3) for the all-runs estimator (B) and the five-arm cell hits (09 RULE @ 36b9b47df §3, §4, §8; @ 3aaf425ef).
POST HOC. Keys, classes and pins only on the laptop side; scores are read inside the Stage A containers (il_arm_rows).

Pieces:
  candidates()            every extra-arm run key from the committed sources: the IL resolution index v5 (one row per key, supersession
                          resolved) plus the two add-on roots 09 §8 names (ilnm_s4rgap_sim2 incl. its try2, ilold_oldsel = sel97); each with
                          its folder, world, set, status, replacement kind and the counted manifest's .eval path and sha256.
  world_of(folder)        the world from the folder suffix (_sim2_ .. __sim_attack -> sim2; _run3_ .. __attack -> run3); asserted against the
                          counted manifest's per-file "world" field for every candidate.
  build_scope(...)        the scripted scope file: scope = no-filter cohort (@ 48a90b1d2) ∩ (admitted keys), admitted = item 4's gates;
                          exclusions by key and class; per set x arm counts; the 88/257/239 five-arm counts as a cross-check only.
  build_block(...)        the INPUTS "ilarms" block (schema ilarms/v1): per set and folder the admitted keys and per-family score files.
  il_arm_rows(...)        container side: the U-style rows of one probe's admitted sim2/run3 keys (read like run_peraction311's S_newmains arms).

  python il_arms.py scope --sample-errors results/il_sample_errors_<stamp>.json [--test]   # writes results/il_scope_<stamp>.json
  python il_arms.py block --scope results/il_scope_<stamp>.json --man-index <fetch index> [--window cut32768|full]   # config/ilarms_<window>_<stamp>.json
"""

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # inside the Stage A container the module sits at /root/il_arms.py (laptop-side helpers unused)
CH = "experiments/2026-10-14_q3_overnight_chain"
RULE = "09 RULE @ 36b9b47df §3 (all-runs selection), §4 (admitted), §8 (IL scope); @ 3aaf425ef (width-only adopt rule)"
INDEX = ("47b904060", "experiments/2026-10-15_q3_interleaved_run3_sim2/results/il_resolution_index_v5.json",
         "c272725f625ed56699f97bf5f671ec7f5d8ffd5bad1825479e7569eed8b0519e")
SPEC = ("e0fcb51c7", "experiments/2026-10-15_il_analysis_09/specs/il_posthoc_spec_r3.json",
        "dbdce088940140e716f49a7f057b3ded7fd1369697c3792c5f71ecefbd992fe6")  # il.arms / il.arms_extended: per-folder p2 + sol manifest pins and hygiene
FIVE_V2 = ("668a3eb91", "experiments/2026-10-15_il_analysis_09/results/il_five_arm_cells_2026-10-15_v2.json",
           "5ca4574c8d627125f0a02f87e3a1e8f91d94dd307793c9b6f72e5fb629c6db14")  # historical cross-check only (§8)
NOFILTER = ("48a90b1d2", "experiments/2026-10-15_fpr_sweep_507_prep/results/nofilter_exclusions_2026-10-16_012305.json",
            "f154cad6df7e5018485617ab5a22644fbb03549e3b315355b81312764bbcb65f")
ADDONS = (  # 09 §8: ilnm_s4rgap_sim2 (S_newmains) and ilold_oldsel = sel97 (S_old); staging exclusions from their coverage records
    {"root": "il-s4rgap", "set": "S_newmains", "folder": "ilnm_s4rgap_sim2_v1__sim_attack", "manifest": ("5cbd3354c", f"{CH}/config/manifest_ilnm_s4rgap_sim2.json"),
     "coverage": ("75e640282", f"{CH}/results/hygiene/il_coverage_ilnm_s4rgap_2026-10-15.json"), "attestation": ("0452b1c3d", f"{CH}/results/hygiene/attestations_il_il-s4rgap.json")},
    {"root": "il-s4rgap-t2", "set": "S_newmains", "folder": "ilnm_s4rgap_sim2_v1_try2__sim_attack",
     "manifest": ("1f3f914e9", f"{CH}/config/manifest_ilnm_s4rgap_sim2_try2.sample_error_v1.json"), "coverage": None,
     "attestation": ("d34d31ed8", f"{CH}/results/hygiene/attestations_il_retry_il-s4rgap-t2.json")},
    {"root": "il-oldsel", "set": "S_old", "folder": "ilold_oldsel_run3_v1__attack", "manifest": ("5c03a303c", f"{CH}/config/manifest_ilold_oldsel_run3.json"),
     "coverage": ("55dd5d83f", f"{CH}/results/hygiene/il_coverage_ilold_oldsel_2026-10-15.json"), "attestation": ("0452b1c3d", f"{CH}/results/hygiene/attestations_il_il-oldsel.json")},
    {"root": "il-oldsel", "set": "S_old", "folder": "ilold_oldsel_sim2_v1__sim_attack", "manifest": ("5c03a303c", f"{CH}/config/manifest_ilold_oldsel_sim2.json"),
     "coverage": ("55dd5d83f", f"{CH}/results/hygiene/il_coverage_ilold_oldsel_2026-10-15.json"), "attestation": ("0452b1c3d", f"{CH}/results/hygiene/attestations_il_il-oldsel.json")},
)
SET_OF_LAUNCH = {"old": "S_old", "q1": "S_new", "q2": "S_new", "q3": "S_new", "q4": "S_new", "s1": "S_newmains", "s2": "S_newmains", "s3": "S_newmains",
                 "s4d": "S_newmains", "s5": "S_newmains"}
CLASS_OF_STATUS = {  # the index's statuses -> canonical classes (09 @ 806da4823; structural per that ruling)
    "never started": ("never launched", True),
    "run_failed (no policy call)": ("setup failure (no policy call)", True),
    "run_failed (EmptyPolicyCompletionError)": ("EPCE (EmptyPolicyCompletionError)", False),
    "run_failed (OutputContractError)": ("OutputContractError", False),
    "staging blocker: no honest donor with the cover task and the same first user turn": ("no clean donor (recoverable)", True),
    "staging blocker: attack epoch has no system prompt and no assistant messages": ("staging exclusion", True),
}
SWITCHES = {  # 507-prep's defaults (10-16 ~03:10, pending 09); one-line flips, recorded in the scope file
    "require_sample_error_null": True,   # (a) item 4's sample_error == null, from the IL sample-error scan (index "scored" alone is not enough)
    "exclude_rlr_roots": False,          # (b) 09 (10-16 ~03:20): v5's rlr/rlr2/rlr3 designations are the REGISTERED resolution (il311_q*r*, ilold_oldr*
}                                        #     relaunch folders, RULE l.2464 / l.2618); "no _rlr roots" covers only _rlr<k>-SUFFIXED root names
EXCLUDED_BY_KEY = {  # 09's by-key rulings (every family; (B) falls back to (A) for that cell x arm), relayed by 507-prep; label text exact
    "sim2|nm145_model_registry@S_newmains": ("excluded by key (caveat)",
                                             "nm145: 32k pack accepted only by a narrow launcher copy with 5 known verify FAILs; excluded by key",
                                             "09 via 507-prep 10-16 ~04:4x; CUT il_scope_v4 @ b6a29b898"),
}
RLR_CLASS = "excluded: _rlr root without a ruling"
RLR_ROOT_RE = re.compile(r"_rlr\d*(?:_|$)")  # the simh311/nm360 grammar: a root name with an _rlr<k> suffix (never admitted without a ruling)
WORLD_RE = re.compile(r"_(sim2|run3)(?:_s\d+)?_v\d+(?:_try\d+)?__(sim_attack|attack)$")
SUFFIX_OF = {"sim2": "sim_attack", "run3": "attack"}


def git_show(c, p):
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout


def pinned(c, p, h=None):
    b = git_show(c, p)
    s = hashlib.sha256(b).hexdigest()
    if h and s != h:
        raise SystemExit(f"{p} @ {c}: sha256 {s} != pinned {h}")
    return json.loads(b), {"commit": c, "path": p, "sha256": s}


def world_of(folder):
    m = WORLD_RE.search(folder)
    if not m or SUFFIX_OF[m.group(1)] != m.group(2):
        raise SystemExit(f"{folder}: no sim2/run3 world in the folder suffix")
    return m.group(1)


REDRAW_RE = re.compile(r"^(?P<cell>.+?)(?:_redraw(?P<n>\d+))?$")


def cell_of_task(task, cell=None, strict=True):
    """A manifest task is <cell> or <cell>_redraw<n> (redraw dirs); the cohort join is by CELL, never by the raw task string. With cell given, the
    task must be exactly that cell or one of its redraws, else STOP."""
    m = REDRAW_RE.match(task)
    c = m.group("cell") if m else task
    if cell is not None and c != cell:
        if strict:
            raise SystemExit(f"task {task!r} is neither cell {cell!r} nor one of its _redraw<n>")
        return None
    return c


def _eval_of(man, world, cell, launch_id=None):
    """The counted manifest's single .eval for (world, cell): exact join on the manifest's world and task fields (+ the index's launch id), never
    on the path (redraw dirs are <task>_redrawN)."""
    lid = lambda x: str(x or "").removeprefix("runs/")  # noqa: E731  (the IL code's il_canon_lid: "runs/<id>" == "<id>")
    hit = [f for f in man["files"] if f.get("world") == world and cell_of_task(f.get("task") or "", strict=False) == cell
           and (launch_id is None or lid(f.get("launch_id")) == lid(launch_id))]
    if len(hit) != 1:
        raise SystemExit(f"{world}|{cell}: {len(hit)} counted-manifest rows")
    return hit[0]


def candidates():
    """Every extra-arm run key with its folder, world (asserted), set, status, replacement kind and .eval path/sha256; plus the sources."""
    idx, ip = pinned(*INDEX)
    spec, sp = pinned(*SPEC)
    il = spec["il"]
    mans = {}  # counted manifests pinned by the index's own sources map (path -> sha256)
    for path, h in idx["sources"].items():
        if path.startswith(f"{CH}/config/manifest_il"):
            c_ = next((x["commit"] for w in ("run3", "sim2") for x in il["counted"][w] if x["path"] == path), None)
            if c_ is None:
                continue
            mans[path] = pinned(c_, path, h)
    rows, srcs = [], {"index": ip, "spec": sp, "counted_manifests": {p: m[1] for p, m in sorted(mans.items())}, "addons": []}
    for r in idx["rows"]:
        st = SET_OF_LAUNCH[r["launch"]]
        if not r.get("launch_id"):
            raise SystemExit(f"{r['key']}: index row without a launch id")
        x = {"key": r["key"], "set": st, "cell": r["cell"], "world": r["world"], "folder": r["scorer_folder"], "status": r["status"],
             "replacement_kind": r["replacement_kind"], "launch_id": r["launch_id"], "source": "index v5",
             "superseded_in_folders": sorted({fo for so in r["superseded_originals"] for fo in (so.get("folder") or [])})}
        if r["status"] == "scored":
            if world_of(r["scorer_folder"]) != r["world"]:
                raise SystemExit(f"{r['key']}: folder {r['scorer_folder']} is not world {r['world']}")
            if r["key"] != f"{r['world']}|{r['cell']}":
                raise SystemExit(f"{r['key']}: key is not <world>|<cell>")
            mp, mpre = r["counted_manifest"].rsplit("@", 1)
            if mp not in mans or not mans[mp][1]["sha256"].startswith(mpre) or len(mpre) < 12:
                raise SystemExit(f"{r['key']}: counted manifest {r['counted_manifest']} is not a pinned source")
            f = _eval_of(mans[mp][0], r["world"], r["cell"], r["launch_id"])
            cell_of_task(f["task"], r["cell"])  # STOP unless <cell> or <cell>_redraw<n>
            x |= {"eval_path": f["path"], "eval_sha256": f["sha256"], "manifest_world": f["world"], "task": f["task"], "counted_manifest": mans[mp][1]}
            ha_ = (il["arms"] | il["arms_extended"]).get(r["scorer_folder"], {}).get("hygiene", {})
            x["hygiene"] = {"h200_gate": bool(ha_.get("longest_epoch_h200_gate")), "format_check": bool(ha_.get("corrected_render"))}
        rows.append(x)
    seen = {x["key"] for x in rows}
    for a in ADDONS:
        man, mp = pinned(*a["manifest"])
        cov = pinned(*a["coverage"])[0] if a["coverage"] else None
        att = pinned(*a["attestation"])[1]
        w = world_of(a["folder"])
        excl = (cov["worlds"][w]["staging_excluded"] if cov else {}) | {x_["task"]: "staging exclusion (revision excluded_rows)" for x_ in man.get("excluded_rows", [])
                                                                        if x_.get("world") == w}
        only = set(man.get("only_cells") or [])
        srcs["addons"].append({"root": a["root"], "folder": a["folder"], "manifest": mp, "coverage": dict(zip(("commit", "path"), a["coverage"])) if cov else None,
                               "attestation": att})
        for f in man["files"]:
            if f.get("world") != w or (only and f["task"] not in only):
                continue
            cell_ = cell_of_task(f["task"])
            k = f"{w}|{cell_}"
            prev = [y for y in rows if y["key"] == k]
            if prev and any(y["status"] == "scored" for y in prev):  # an add-on only fills keys the index (or an earlier attempt) never scored
                raise SystemExit(f"{k}: scored in {prev[0]['source']} and again in add-on root {a['root']}")
            rows = [y for y in rows if y["key"] != k]  # e.g. the index's 'sel' stage rows (never started) filled by sel97; s4rgap's try2 retry
            seen.add(k)
            ex_ = excl.get(f["task"], excl.get(cell_))
            st_ = "scored" if ex_ is None else "staging blocker: " + ex_
            rows.append({"key": k, "set": a["set"], "cell": cell_, "task": f["task"], "world": w, "folder": a["folder"], "status": st_, "replacement_kind": "original",
                         "replaces": [{"source": y["source"], "status": y["status"], "folder": y["folder"]} for y in prev],
                         "launch_id": f.get("launch_id"), "source": a["root"], "eval_path": f["path"], "eval_sha256": f["sha256"], "manifest_world": f["world"],
                         "counted_manifest": mp, "sample_error_in_manifest": f.get("sample_error", "absent"),
                         "hygiene": {"attestation": True, "format_check": True}})
    return rows, srcs


def cut_key(eval_path):
    """The CUT fork's scope key (il_scope_v2 @ 1d87eb219): the bare eval path with ONE leading 'runs/' removed, plus '#e0' (one epoch)."""
    return eval_path.removeprefix("runs/") + "#e0"


def cut_scope_diff(admitted, cut_ref):
    """STOP unless our admitted keys equal the CUT fork's scope file's admitted_keys by key (commit:path:sha256)."""
    c_, p_, h_ = cut_ref.split(":")
    cut, pin = pinned(c_, p_, h_)
    theirs = set(cut["admitted_keys"])
    ours = {cut_key(x["eval_path"]): x["key"] + "@" + x["set"] for x in admitted}
    if len(ours) != len(admitted):
        raise SystemExit("two admitted keys share one eval path")
    only_ours, only_theirs = sorted(set(ours) - theirs), sorted(theirs - set(ours))
    if only_ours or only_theirs:
        raise SystemExit(json.dumps({"il scope differs from CUT's by key": {"only_ours": [f"{k} ({ours[k]})" for k in only_ours[:20]], "n_only_ours": len(only_ours),
                                                                            "only_theirs": only_theirs[:20], "n_only_theirs": len(only_theirs)}}))
    out = {"equal_by_key": True, "keys": len(theirs)}
    if isinstance(cut.get("rows"), list):  # v3+: each admitted key's scorer folder, set, world and cell must agree too (nm145 -> s4rgap try2, v3_fix)
        their_ = {r["key"]: r for r in cut["rows"] if r.get("admitted")}
        bad = [(k, x["folder"], their_[k].get("scorer_folder")) for x in admitted for k in [cut_key(x["eval_path"])]
               if (their_[k].get("scorer_folder"), their_[k].get("set"), their_[k].get("world"), their_[k].get("cell")) != (x["folder"], x["set"], x["world"], x["cell"])]
        if bad:
            raise SystemExit(json.dumps({"il scope differs from CUT's in folder / set / world / cell": bad[:20], "n": len(bad)}))
        out["equal_folder_set_world_cell"] = True
    return pin | out


def build_scope(sample_errors=None, test=False, cut_ref=None):
    rows, srcs = candidates()
    nf, nfp = pinned(*NOFILTER)
    coh = {o: set(v["headline_cells"]) for o, v in nf["origins"].items()}
    se = {}
    if sample_errors:
        se_b = Path(sample_errors).read_bytes()
        se_j = json.loads(se_b)
        se = {x["key"]: x for x in se_j["rows"]}
        srcs["sample_errors"] = {"path": str(Path(sample_errors).resolve().relative_to(REPO)), "sha256": hashlib.sha256(se_b).hexdigest()}
    elif SWITCHES["require_sample_error_null"] and not test:
        raise SystemExit("a non-TEST scope needs the IL sample-error scan (--sample-errors)")
    adm, exc = [], []
    for x in rows:
        cls, why = None, None
        xk_ = EXCLUDED_BY_KEY.get(f"{x['key']}@{x['set']}")
        if xk_:
            cls, why = xk_[0], {"caveat": xk_[1], "ruling": xk_[2]}
        elif x["cell"] not in coh[x["set"]]:
            cls = "cell not in the no-filter cohort"
        elif x["status"] != "scored":
            cls = CLASS_OF_STATUS.get(x["status"], (None,))[0] or ("staging exclusion" if x["status"].startswith("staging blocker") else None)
            if cls is None:
                raise SystemExit(f"{x['key']}: unmapped status {x['status']!r}")
        elif RLR_ROOT_RE.search(x["folder"]) or (SWITCHES["exclude_rlr_roots"] and x["replacement_kind"].startswith("rlr")):
            cls = RLR_CLASS
        elif not (x["hygiene"].get("h200_gate", True) and x["hygiene"].get("format_check", True)):
            cls, why = "hygiene record missing (H200 gate / format check)", x["hygiene"]
        elif SWITCHES["require_sample_error_null"]:
            s_ = se.get(x["key"])
            if s_ is not None and (s_.get("path") != x["eval_path"] or s_.get("sha256") != x["eval_sha256"] or s_.get("folder") != x["folder"]):
                raise SystemExit(f"{x['key']}: the scan read another .eval than the candidate's (path / sha256 / folder)")
            if s_ is None:
                cls = None if test else "sample error not attested (not in the scan)"
                why = "TEST: not scanned" if test else None
            elif s_.get("sha_ok") is not True:
                cls = "sample error not attested (eval missing or sha differs)"
            elif s_.get("has_error") is True:
                cls, why = f"errored run ({s_.get('sample_error')})", None
            elif not (s_.get("has_error") is False and s_.get("sample_error") is None):
                raise SystemExit(f"{x['key']}: scan row without an explicit null sample error")
        if cls:
            exc.append({"key": x["key"], "set": x["set"], "folder": x["folder"], "class": cls, "source": x["source"],
                        "superseded_in_folders": x.get("superseded_in_folders", [])} | ({"detail": why} if why else {}))
        else:
            adm.append(x | ({"note": why} if why else {}))
    counts = {}
    for st in ("S_old", "S_new", "S_newmains"):
        counts[st] = {w: {"admitted": sum(1 for x in adm if x["set"] == st and x["world"] == w),
                          "excluded": sum(1 for x in exc if x["set"] == st and x["key"].startswith(w + "|"))} for w in ("sim2", "run3")}
        counts[st]["cells_with_sim2"] = len({x["cell"] for x in adm if x["set"] == st and x["world"] == "sim2"})
        counts[st]["cells_with_run3"] = len({x["cell"] for x in adm if x["set"] == st and x["world"] == "run3"})
        counts[st]["cells_with_either"] = len({x["cell"] for x in adm if x["set"] == st})
    by_cls = {}
    for x in exc:
        by_cls[x["class"]] = by_cls.get(x["class"], 0) + 1
    fv, fvp = pinned(*FIVE_V2)
    five = {"S_old": set(fv["S_old_five_arm"]["cells"]), "S_new": set(fv["S_new_on_M"]["cells"]), "S_newmains": set(fv["S_newmains_five_arm"]["cells"])}
    both = {st: {x["cell"] for x in adm if x["set"] == st and x["world"] == "sim2"} & {x["cell"] for x in adm if x["set"] == st and x["world"] == "run3"}
            for st in five}
    out = {"label": "IL extra-arm scope (09 §8): keys and classes only; POST HOC" + ("; TEST" if test else ""), "rule": RULE, "switches": SWITCHES,
           "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "generator": "il_arms.py scope",
           "excluded_by_key_rulings": {k: {"class": v[0], "caveat": v[1], "ruling": v[2]} for k, v in EXCLUDED_BY_KEY.items()},
           "sources": srcs | {"nofilter": nfp, "five_arm_v2": fvp},
           "counts": counts, "excluded_by_class": dict(sorted(by_cls.items())),
           "admitted_by_replacement_kind": {k: sum(1 for x in adm if x["replacement_kind"] == k) for k in sorted({x["replacement_kind"] for x in adm})},
           "canonical_run_rule": ("one run per cell x extra arm: v5's designated row (supersession resolved in the index; a superseded original is never "
                                  "admitted); a canonical run that fails a gate leaves that cell x arm without an extra run (no fallback)"),
           "five_arm_cross_check": {st: {"v2_cells": len(five[st]), "both_arms_admitted_now": len(both[st]), "v2_and_both": len(five[st] & both[st])} for st in five},
           "admitted": sorted(({k: x[k] for k in ("key", "set", "cell", "world", "folder", "replacement_kind", "source", "eval_sha256")}
                               | {"superseded_in_folders": x.get("superseded_in_folders", [])}
                               | {"zero_tool": se.get(x["key"], {}).get("n_tool_nonsubmit") == 0} for x in adm),
                              key=lambda y: (y["set"], y["key"])),
           "excluded": sorted(exc, key=lambda y: (y["set"], y["key"]))}
    keys = [x["key"] + "@" + x["set"] for x in out["admitted"]]
    if len(set(keys)) != len(keys):
        raise SystemExit("an admitted key twice in one set")
    if cut_ref:  # the CUT fork's independent §8 scope file: equal by key, else STOP with the diff
        out["cut_scope_cross_check"] = cut_scope_diff(adm, cut_ref)
    elif not test:
        raise SystemExit("a non-TEST scope needs --cut-scope <commit:path:sha256> (equal by key)")
    for x in out["admitted"]:
        x["cut_key"] = cut_key(next(y["eval_path"] for y in adm if y["key"] == x["key"] and y["set"] == x["set"]))
    return out


def build_block(scope, scope_pin, manifests, families=("p2", "sol"), window="cut32768", sub=None, optional_pending=()):
    """The INPUTS "ilarms" block. manifests: {(folder, family): (bytes, path)} fetched copies; every admitted folder needs every family's manifest
    or that family is 'pending' for the folder (a probe of a pending family uses (A) inside (B) on that folder's keys, labelled)."""
    sub = sub or {"sol": "sol", "p2": "p2", "famexp": "famexp_r2", "own_a": "own_a", "own_all_rs": "own_all_rs"}
    sets = {}
    for x in scope["admitted"]:
        d = sets.setdefault(x["set"], {}).setdefault(x["folder"], {"world": x["world"], "keys": [], "source_keys": [], "structural_zero_keys": [], "manifests": {},
                                                                   "files": {}, "pending_families": []})
        d["keys"].append(x["key"])
        if x.get("zero_tool"):
            d["structural_zero_keys"].append(x["key"])
    for x in scope["admitted"] + scope["excluded"]:  # every key the folder's sources hold (a score file may carry excluded keys; they are dropped by key)
        if x["set"] in sets and x["folder"] in sets[x["set"]]:
            sets[x["set"]][x["folder"]]["source_keys"].append(x["key"])
        for fo in x.get("superseded_in_folders", []):  # a superseded original's rows stay in its old folder's file under the same key (dropped by key)
            if x["set"] in sets and fo in sets[x["set"]]:
                sets[x["set"]][fo]["source_keys"].append(x["key"])
    opt_ = set(optional_pending)  # admitted keys whose (uncut) files are not routed yet: (A) inside (B) there, labelled "no extra runs scored"; never a STOP
    for st, fo in sets.items():
        for folder, d in fo.items():
            d["keys"].sort()
            d["source_keys"] = sorted(set(d["source_keys"]))
            d["optional_pending_keys"] = sorted(k for k in d["keys"] if f"{k}@{st}" in opt_)
            for fam in families:
                m = manifests.get((folder, fam))
                if m is None:
                    d["pending_families"].append(fam)
                    continue
                b_, path_ = m
                d["manifests"][fam] = {"path": path_, "sha256": hashlib.sha256(b_).hexdigest()}
                mj_ = json.loads(b_)
                if fam == "own_all_rs" and mj_.get("folder") != folder:  # own_all-style manifests name their folder (run_peraction311.own_all_files)
                    raise SystemExit(f"own_all_rs {folder}: manifest names folder {mj_.get('folder')}")
                for fn, s_ in mj_["files"].items():
                    if fam == "own_all_rs" and not fn.endswith(".npz"):
                        continue
                    d["files"][f"{fam}/{Path(fn).name}"] = {"path": f"scaleup/{folder}/{sub[fam]}/{Path(fn).name}", "sha256": s_}
    return {"schema": "ilarms/v1", "rule": RULE, "window": window, "scope": scope_pin, "switches": scope["switches"], "sets": sets,
            "counts": scope["counts"], "families": list(families)}


def il_arm_rows(p, origin, block, cohort_cells, read_npz, rows_any):
    """Container side: one probe's admitted sim2/run3 rows on this origin's cohort cells -> (U-style dict of arrays, report). A folder without this
    probe's file (pending family) contributes nothing; its keys are listed, so the row's (B) falls back to (A) there (labelled)."""
    import numpy as np

    add = {x: [] for x in ("run", "score", "fold", "sbm", "source")}
    rep = {"folders_read": 0, "pending_keys": [], "keys_without_rows": [], "structural_zero_keys": [], "rows": 0, "block_window": block["window"]}
    for folder, d in sorted(block["sets"].get(origin, {}).items()):
        want = {k for k in d["keys"] if (cohort_cells is None or k.split("|", 1)[1] in set(cohort_cells)) and k not in set(d.get("optional_pending_keys", []))}
        rep.setdefault("optional_pending_keys", []).extend(sorted(set(d.get("optional_pending_keys", []))))
        if not want:
            continue
        e = d["files"].get(p["new"])
        if e is None:
            rep["pending_keys"] += sorted(want)
            continue
        k, s, f, b = rows_any(read_npz("conv", e["path"], e["sha256"]), p, origin)
        if not {x for x in k.tolist() if x.split("|", 1)[0] == d["world"]} <= set(d["source_keys"]):  # the folder's full source membership (admitted + excluded)
            raise SystemExit(f"{p['label']}: {folder}: rows of {d['world']} keys outside the folder's source membership")
        m = np.isin(k, sorted(want))
        nor_ = want - set(k[m].tolist())
        sz_ = nor_ & set(d.get("structural_zero_keys", []))  # admitted zero-tool runs (sample_error null, 0 tool calls): no row = not caught (§4)
        rep["structural_zero_keys"] += sorted(sz_)
        rep["keys_without_rows"] += sorted(nor_ - sz_)
        rep.setdefault("_scored_keys", []).extend(set(k[m].tolist()))
        for x, y in zip(("run", "score", "fold", "sbm"), (k[m], s[m], f[m], b[m])):
            add[x].append(y)
        add["source"].append(np.array([origin] * int(m.sum())))
        rep["folders_read"] += 1
        rep["rows"] += int(m.sum())
    out = {x: (np.concatenate(v) if v else None) for x, v in add.items()}
    adm_ = sum(len(d["keys"]) for d in block["sets"].get(origin, {}).values())
    opt_ = len(rep.get("optional_pending_keys", []))
    rep["counts"] = {"admitted": adm_, "routed": adm_ - opt_,
                     "scored": len(set(rep.pop("_scored_keys", []))) + len(rep["structural_zero_keys"]),
                     "rule": "report-only (0d/09 10-16 ~03:30): the freeze takes every admitted extra arm that is routed and scored at that moment; "
                             "a probe x cell x arm without one uses (A) inside (B), labelled 'no extra runs scored'"}
    rep["counts"]["in_scope_not_built"] = rep["counts"]["admitted"] - rep["counts"]["scored"]
    return out, rep


def block_main(a):
    """il_arms.py block --scope results/il_scope_<stamp>.json --man-index <ilarms_man index json> [--families p2,sol,famexp] [--window cut32768|full]
    -> config/ilarms_<window>_<stamp>.json (exclusive create). Every manifest copy is re-hashed against the fetch index."""
    sp_ = Path(a[a.index("--scope") + 1])
    mi_ = Path(a[a.index("--man-index") + 1])
    fams = tuple(a[a.index("--families") + 1].split(",")) if "--families" in a else ("p2", "sol", "famexp")
    window = a[a.index("--window") + 1] if "--window" in a else "cut32768"
    if window != "cut32768":  # the block names the 32k score files; the full-length job swaps each read for its pinned uncut file (full fork's read guard)
        raise SystemExit("--window: only cut32768 (the full-length build maps these files to uncut ones through INPUTS_FULL's override)")
    sb = sp_.read_bytes()
    scope = json.loads(sb)
    if "TEST" in scope["label"] and "--test" not in a:
        raise SystemExit("a TEST scope builds only a TEST block (--test)")
    idx = json.loads(mi_.read_bytes())
    man = {}
    for k, v in idx["manifests"].items():
        folder, fam = k.rsplit("__", 1)
        if v["present"] and fam in fams:
            b_ = (mi_.parent / f"{k}.json").read_bytes()
            if hashlib.sha256(b_).hexdigest() != v["sha256"]:
                raise SystemExit(f"{k}: manifest copy differs from the fetch index")
            if v.get("il_spec_pin") and not v.get("equals_il_spec_pin"):
                raise SystemExit(f"{k}: manifest differs from the IL spec's pin")
            man[(folder, fam)] = (b_, v["path"])
    pin = {"path": str(sp_.resolve().relative_to(REPO)), "sha256": hashlib.sha256(sb).hexdigest()}
    blk = build_block(scope, pin, man, fams, window)
    blk["manifest_index"] = {"path": str(mi_), "sha256": hashlib.sha256(mi_.read_bytes()).hexdigest()}
    blk["built_pt"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")  # the freeze moment = this block's sha + time
    blk["report_only"] = ("09/0d 10-16 ~03:30: the IL extra arms never gate the freeze; inclusion by availability at the freeze, never by value; no "
                          "extra arm is added after the selection record is committed; later landings are a labelled later update, not a re-selection")
    if "--optional-pending" in a:  # admitted keys not routed (uncut) at the freeze: '<world>|<cell>@<set>' per line, from the routing state
        opt_ = {x.strip() for x in Path(a[a.index("--optional-pending") + 1]).read_text().split() if x.strip()}
        for st, fo in blk["sets"].items():
            for d in fo.values():
                d["optional_pending_keys"] = sorted(k for k in d["keys"] if f"{k}@{st}" in opt_)
    blob = json.dumps(blk, indent=1, sort_keys=True).encode()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "config" / f"ilarms_{window}{'_TEST' if '--test' in a else ''}_{stamp}.json"
    dest.open("xb").write(blob)
    pend = {st: {fo: d["pending_families"] for fo, d in v.items() if d["pending_families"]} for st, v in blk["sets"].items()}
    print(json.dumps({"file": str(dest.relative_to(HERE)), "sha256": hashlib.sha256(blob).hexdigest(), "folders": {st: len(v) for st, v in blk["sets"].items()},
                      "keys": {st: sum(len(d["keys"]) for d in v.values()) for st, v in blk["sets"].items()}, "pending_families": pend}, indent=1))


def main():
    a = sys.argv[1:]
    if a and a[0] == "block":
        return block_main(a)
    if not a or a[0] != "scope":
        raise SystemExit(__doc__)
    test = "--test" in a
    se = a[a.index("--sample-errors") + 1] if "--sample-errors" in a else None
    out = build_scope(se, test, a[a.index("--cut-scope") + 1] if "--cut-scope" in a else None)
    blob = json.dumps(out, indent=1).encode()
    dest = HERE / "results" / f"il_scope{'_TEST' if test else ''}_{out['stamp_pt']}.json"
    dest.open("xb").write(blob)
    print(json.dumps({"file": str(dest.relative_to(HERE)), "sha256": hashlib.sha256(blob).hexdigest(), "counts": out["counts"],
                      "excluded_by_class": out["excluded_by_class"], "five_arm_cross_check": out["five_arm_cross_check"]}, indent=1))


if __name__ == "__main__":
    main()
