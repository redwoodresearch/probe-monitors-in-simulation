"""INPUTS_FULL: the full-length (uncut) build's one frozen inputs file (09 RULE lead §4 @ cfd94c2e9: own freeze, verifier and waitlist; one
pipeline, two input files). Build and verify, fail closed. Keys, paths and sha256s only: no score is read.

Schema inputs_full/v1. The 32k build's frozen INPUTS_FINAL v2 is the base: cohort, maps, pools and draws are inherited by sha; this file adds
what differs (the uncut score files, fits, pools coverage and OOD table) and the final-build items both builds share (cohort, exclusions,
waitlist). Every pin is {commit, path, sha256} read with git show; a pin may be {"pending": why} only in a TEST file.
  base                 the INPUTS_FINAL v2 (schema inputs_final/v2) the runner's --inputs must equal
  overrides            {"<set>|<readout>": pin}   the CUT fork's override per set and readout (set in SETS + "pools_ep2"; readout C | R)
  gates                {"<set>|<readout>": pin}   its G2/G3 gate record; each override entry's gate must cite this pin and its own "<folder>|<file>"
  inventories          {set: pin + {reattest}}    (a'') v3 with d9's re-attestation pin (cutfull_decisive.reattest_map); others bind staged shas
  roles_old            {S_old folder: real | relaunch | simh}   derived here from inventory_b_v2's worlds (T196 rows)
  fits                 {retrain: {kind sola: fit, contract, refit pins | kind 4f: fit, companion pins; files {C, R} "<scorer>/<file>"}}
  pools_required       {source "inventories", ep2, fill, S_paper493 pins, coverage}: those inventories (= inventories.<same>)
                       and the coverage they give every e12 folder and fill-in root, by key (0d's rule: no epoch-1 fallback)
  ood                  {pin, sha256 dc026dfe..., signature: READ_LOG pin (09's "SIGNED" line names the file and sha)}
  cohort, exclusions, waitlist   pins (the final-build items, shared with the 32k build); waitlist_pending [] when FROZEN
  estimator            {"A": "registered", "B": "unavailable: ..." | pins}   (09 @ 3aaf425ef; (B) off the critical path)
  scope                full_override's labels (headline six, registered-scored-uncut, external four), lead_readout "C"
  pending              every unresolved item; FROZEN needs []

  python inputs_full.py build --pins config/inputs_full_pins.json [--status "FROZEN <PT>" | --test] [--copies-root <dir>]
  python inputs_full.py verify <INPUTS_FULL.json> [--deep] [--copies-root <dir>]   (--deep: also the base's own v2 verification)
  --copies-root: the fpr507 manifest copies' root for copies.py (0d's F2, 10-17): inside the records wrapper the root comes ONLY from this
  flag (copies.py STOPs at the first copy read without it; --deep reads copies via inputs_final_v2.verify -> inputs_final.verify); on the
  laptop, without it, $PROBE_SHIFT_COPIES as before.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import copies as CP  # noqa: E402
import cutfull_decisive as CD  # noqa: E402
import full_override as FO  # noqa: E402
import inputs_final as IF  # noqa: E402

SCHEMA = "inputs_full/v1"
SETS = FO.SETS + ("pools_ep2",)
READOUTS = ("C", "R")
# 0d (05:2x): the provisional freeze's labels, verbatim; every record built on a provisional INPUTS_FULL carries them
PROVISIONAL_LABELS = [
    "PROVISIONAL: override/gate/inventory code since r14 @ 985d4b3ed awaits Codex review (credits out since 04:50); ratified or rebuilt after review. "
    "Unreviewed commits: build_full_override.py 1edb3d39b 434cfaa82 2df13ef6f 07a19ae05 fa521ca45 0b4bc4d86 fd150179b (+ edits in f475a1a30 cb7511c42 "
    "0093df634 1621aa0c9 9c904371e); run_peraction311 gate code 88733fb72 b5c5f9b6f 1621aa0c9 0093df634 cb7511c42 be66df23e; stamp_gates.py 7e649afff; "
    "check_staged.py f475a1a30; inventory builders make_inventory_addendum c9f74aed4 7db048406 2abc53746 1a5a5a4a4 fd65c1d39, make_inventory_b_fix_v1 "
    "cf1ecd53d, make_inventory_il_fix_v1 d86042930 a09b675f3, make_inventory_ha25f_v2 84a73959b, make_il_scope 649081545 49522cb0d 2704c8405, "
    "make_il_scope_v2/v3/v4 1d87eb219 992848516 d933d412e; CUT's ha25f lookup fix efd143756; records committed after 04:50: b0a78bbe5 17136e7c4 "
    "509f46222 e93f297ff b6a29b898.",
    "Codex unavailable at commit",
]
# Dropped labels (0d's item (c), 10-16): the full fork's commits 3c8fc9fa3 d14ba40d9 92cded3d6 d6d9e5a33 a61b38248 38fae86e3 1fb8c4be1 f9911ae44
# 1f796116d (results/codex/full_codex_<sha>.md, 1f796116d in full_codex_1fb8c4be1.md; fixes e90ea046d, rounds r2a / r2b / r3 clean) and "ep2 identity by
# metadata tag; world|cell join assert pending Codex" (E12's 4bd9c7eea: results/codex/e12_codex_index.md @ 9b6d994fe, r9 = git diff 2385f562b 4bd9c7eea)
S4MAP = ("b68ab0a53", "experiments/2026-10-14_q3_overnight_chain/results/s4_source_map/s4_source_map_4r_2026-10-15.json")  # d9's S4 source map
S4_N = 52  # its staged outside-source-map (orphan) entries = inventory_b_v2's s4_orphans_out
# the probe files whose 32k S_old relaunch file exists (cut map: Sol a_tight / c_asis_tight R and C): their relaunch entries need inside_32k_equal
NA_RELAUNCH_WITH_CUT = ("sol/a_tight__avg_top2.npz", "sol/a_tight__cmd_A.npz", "sol/c_asis_tight__avg_top2.npz", "sol/c_asis_tight__cmd_A.npz")
# every inventory INPUTS_FULL pins, in supersession order (a later one's row replaces an earlier one's by run key): (a'') v3 for S_new;
# inventory_b_v2 for S_old / S_newmains; S_paper493 waves 1 and 2-3; the epoch-2 and fill-in pools; the IL extra arms, then their fix
INVENTORY_ORDER = ("S_new", "ha25f", "S_old", "S_newmains", "S_paper493", "S_paper493_w23", "pools_ep2", "pools_fill", "il")  # il = inventory_il_v2 (one complete file; supersedes il_v1 + il_fix_v1)
FLAGS = ("keys_bound_by_task", "runs_without_rows_ok", "inside_32k_equal", "n_tokens_eq_full_len")
NA = "n/a: no cut counterpart"
# (e) 507-prep 09:10: each retrain's fit-record kind and the scorer its uncut files sit under ("<scorer>/<file>" in the overrides). SOLA pins a fit,
# its contract and its refit-match record; 4f pins a fit and its INPUTS_FULL companion. Any other shape STOPs.
FIT_KIND = {"sol_a_uncut": ("sola", "sola_uncut_reselect"), "sol_a_uncut_pinned": ("sola", "sola_uncut_pinned"),
            "c_asis_uncut": ("sola", "casis_uncut_inherit"), "own_all_rs_uncut_L48": ("4f", "own_all_rs_uncut"),
            "own_all_rs25_uncut_L48": ("4f", "own_all_rs_uncut"), "own_all_rs10_uncut_L48": ("4f", "own_all_rs_uncut")}
FIT_PINS = {"sola": ("fit", "contract", "refit"), "4f": ("fit", "companion")}
# Codex r2b #11: the scorers' records that must be cited, by name: SOLA-UNCUT's pointer index and two score records per SOLA scorer
SCORE_RECORD_RULE = {"sola_uncut_pointer_index_": 1} | {f"sola_uncut_score_{sc}_": 2 for k, (kd, sc) in sorted(FIT_KIND.items()) if kd == "sola"}
# Codex r2a #14 / r2b #14: the e12 block's pools that must be present (empty e2 lists are allowed; a missing origin, pool or H_A group is a problem)
E12_POOLS = {"S_old": ("H_A", "H_B", "H_B2"), "S_new": ("H_A", "H_B"), "S_newmains": ("H_A", "H_B", "H_B_coverage")}
E12_S_NEW_H_A_GROUPS = ("DT (no 25f DT honest-A roots scored)", "Sol (a), c_asis", "own family (own_frac10/25/100_L48)",
                        "public probes (FIBS, harmfulness, refusal, high-stakes)")
assert set(FIT_KIND) == set(FO.RETRAIN_OF)
# (f) 507-prep 09:10: the pools-coverage record is the ep2 inventory's epoch2_admitted_subset_check plus inventory_fill_v1 (A25, 8/8); the e12
# block's other fill-in root (g4) is an S_paper493 root, covered by the pinned S_paper493 w1 inventory (0d OK 09:36, 14/14). Any other root STOPs.
FILL_SOURCE = {"simh_t196fill_v1__sim_honest": ("pools_fill", 8), "sp493_g4_simh_v1_try2__sim_honest": ("S_paper493", 14)}  # 0d (09:36): g4 14/14, A25 8/8
RULES = {"lead": "09 RULE @ 129b7c4f1 (lead §1, §3)", "freeze": "09 RULE @ cfd94c2e9 (lead §4)", "waitlist": "09 RULE @ 07ad588be",
         "classes": "09 RULE §7(iv) (canonical earliest root cause)", "estimator": "09 RULE @ 3aaf425ef (width-only adopt rule)",
         "cohort": "no-filter (09 @ 01012e87e)", "pools": "0d: the 32k headline's pools, extracted uncut; no epoch-1 fallback (09 withdrew it)"}


def pin(commit, path):
    return {"commit": commit, "path": path, "sha256": IF.sha(IF.git_show(commit, path))}


def load(e, what):
    b = IF.git_show(e["commit"], e["path"])
    if IF.sha(b) != e["sha256"]:
        raise SystemExit(f"{what}: sha256 of {e['path']} @ {e['commit']} differs")
    return b


def is_pin(e):
    return isinstance(e, dict) and all(isinstance(e.get(k), str) for k in ("commit", "path", "sha256"))


def resolve(x):
    """pins-file entry {commit, path} -> pin; {"pending": why} passes through."""
    return x if "pending" in x else pin(x["commit"], x["path"])


HEX64 = re.compile(r"[0-9a-f]{64}")
# 09 (10-17): the sha fields the writers of INPUTS_FULL, E12, the overrides and the inventories write (keys-only schema scan of 110447's
# files, 10-17): each must be a 64-hex string wherever it appears; E12's epoch1_eval_sha256 is a non-empty list of them.
PIN_SHA = frozenset({"sha256", "eval_sha256", "staged_sha256", "manifest_sha256", "epoch1_members_sha256", "e2_members_sha256",
                     "physical_members_sha256", "script_sha256", "staged_sha"})  # staged_sha: E12 arms (make_e12_block.py L253)
PIN_SHA_LISTS = frozenset({"epoch1_eval_sha256", "staged_shas"})  # staged_shas: override entries (build_full_override.py L270)
# keys that contain "sha" but hold no sha (second review a16c7b88, 10-17: a key-name scan of 110447 and its 22 files found no others)
NOT_SHA = frozenset({"manifests_without_staged_sha", "n_eval_shas", "e2_shas"})


def pin_sha_problems(o, at="$"):
    """Recursive: a PIN_SHA field that is not a 64-hex string (null included), a PIN_SHA_LISTS field that is not a non-empty list of them,
    a dict naming both commit and path without a sha256, a commit that is not a non-empty string, or any other key containing "sha" (any
    case) outside NOT_SHA (unknown to the schema: a new field is reviewed, never passed). Key paths only, never values."""
    out = []
    if isinstance(o, dict):
        if "commit" in o and "path" in o and "sha256" not in o:
            out.append(f"{at}: commit + path without sha256")
        if "commit" in o and not (isinstance(o["commit"], str) and o["commit"]):
            out.append(f"{at}.commit: not a non-empty string")
        for k, v in o.items():
            at_ = f"{at}.{k}"
            if k in PIN_SHA:
                if not (isinstance(v, str) and HEX64.fullmatch(v)):
                    out.append(f"{at_}: not a 64-hex sha256")
            elif k in PIN_SHA_LISTS:
                if not (isinstance(v, list) and v and all(isinstance(x, str) and HEX64.fullmatch(x) for x in v)):
                    out.append(f"{at_}: not a non-empty list of 64-hex sha256s")
            elif isinstance(k, str) and "sha" in k.lower() and k not in NOT_SHA:
                out.append(f"{at_}: unknown sha field")
            out += pin_sha_problems(v, at_)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out += pin_sha_problems(v, f"{at}[{i}]")
    return out


def require_pin_shas(m):
    """09 (10-17): STOP (SystemExit, nothing written) on any null or non-64-hex pin sha256 in INPUTS_FULL itself, E12, every override and
    every inventory, recursively. build calls it once its pins are resolved, and verify calls it first, so both STOP."""
    bad = [f"INPUTS_FULL {x}" for x in pin_sha_problems(m)]
    files = [("e12", m.get("e12"))] + [(f"override {k}", e) for k, e in sorted((m.get("overrides") or {}).items())]
    files += [(f"inventory {k}", e) for k, e in sorted((m.get("inventories") or {}).items())]
    for what, e in files:
        if is_pin(e):  # a pin whose own sha256 is null is already listed from INPUTS_FULL above
            bad += [f"{what} {x}" for x in pin_sha_problems(json.loads(load(e, what)))]
    if bad:
        raise SystemExit(f"STOP: {len(bad)} null or non-64-hex pin sha256 field(s) (09, 10-17): {bad[:8]}")


def roles_from_inventory(inv):
    """S_old folders by role from inventory_b_v2's T196 rows: world relaunch -> relaunch, sim_honest -> simh, other worlds -> real."""
    out = {}
    for r in inv["runs"]:
        if r["task_set"] != "T196":
            continue
        role = "relaunch" if r["world"] == "relaunch" else "simh" if r["world"] == "sim_honest" else "real"
        if out.setdefault(r["original_folder"], role) != role:
            raise SystemExit(f"inventory: {r['original_folder']} holds two roles")
    return out


def build(pins_path, status, test):
    P = json.loads((HERE / pins_path).read_text())
    m = {"schema": SCHEMA, "status": status, "test": test, "built_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z"),
         "rules": RULES, "pins_file": {"path": pins_path, "sha256": IF.sha((HERE / pins_path).read_bytes())}, "lead_readout": "C",
         "scope": {"registered_labels": list(FO.SCOPE), "retrains": FO.RETRAIN_OF, "headline_six": list(FO.HEADLINE_SIX), "labels": FO.LABELS,
                   "auroc_subset": P["auroc_subset"]},
         "base": resolve(P["base"])}
    m["overrides"] = {k: resolve(v) for k, v in P["overrides"].items()}
    m["gates"] = {k: resolve(v) for k, v in P["gates"].items()}
    m["inventories"] = {k: resolve(v) | ({"reattest": v["reattest"]} if "reattest" in v else {}) for k, v in P["inventories"].items()}
    if "s_old_simh_extra_roots" in P:  # 09 route C: the one allowlisted S_old sim-honest root, pinned (verify() checks it is the approved one)
        m["s_old_simh_extra_roots"] = {r: resolve(v) for r, v in P["s_old_simh_extra_roots"].items()} if isinstance(P["s_old_simh_extra_roots"], dict) \
            else P["s_old_simh_extra_roots"]
    m["fits"] = {k: v if "pending" in v else fit_entry(k, v) for k, v in P["fits"].items()}
    m["ood"] = resolve(P["ood"]) | {"signature": resolve(P["ood_signature"])}
    for k in ("cohort", "exclusions", "waitlist", "census", "e12", "ilarms", "relaunch_members", "cutmap"):
        m[k] = resolve(P[k])
    require_pin_shas(m)  # 09 (10-17): before anything else parses a pinned record (the pins above are only hashed)
    inv_b = m["inventories"].get("S_old", {})
    m["roles_old"] = roles_from_inventory(json.loads(load(inv_b, "inventory S_old"))) if is_pin(inv_b) else {"pending": "inventory S_old pending"}
    if P["pools_required"] != {"source": "inventories"} and "pending" not in P["pools_required"]:
        raise SystemExit('pools_required: {"source": "inventories"} or {"pending": why}')
    m["pools_required"] = P["pools_required"] if "pending" in P["pools_required"] else pools_coverage(m)[0]
    m["waitlist_pending"] = waitlist_pending(m, test)
    m["paired_pool_exclusions"] = paired_exclusions(m)  # 09 item 26: listed by key; asserted again at verify
    m["estimator"] = P["estimator"]
    if not isinstance(P.get("provisional"), bool) or not isinstance(P.get("score_records"), list):  # Codex 1fb8c4be1 #3 / d6d9e5a33 #5: stated,
        raise SystemExit('pins: "provisional" (true | false) and "score_records" (a list) must both be stated')  # never defaulted
    m["provisional"] = P["provisional"]
    m["labels_provisional"] = PROVISIONAL_LABELS if P["provisional"] else []
    m["score_records"] = [pin(x["commit"], x["path"]) for x in P["score_records"]]  # scorers' records cited for provenance (sha-checked)
    m["dryrun_counts"] = resolve(P["dryrun_counts"]) if "dryrun_counts" in P else {"pending": "the full-length dry run's substitution counts (sidecar)"}
    m["pending"] = pending_items(m)
    m["relaxations"] = verify(m, want_relax=True)
    m["freeze_checks"] = freeze_checks(m)
    pr = verify(m)  # Codex d6d9e5a33 #1 / 38fae86e3 #6: the full verification runs before any file is written; a FROZEN file never carries a problem
    if not test and (m["pending"] or pr):
        raise SystemExit(f"FROZEN build: pending {m['pending']}; problems {pr}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "config" / f"INPUTS_FULL{'_TEST' if test else ''}_{stamp}.json"
    blob = json.dumps(m, indent=1).encode()
    dest.open("xb").write(blob)
    print(json.dumps({"file": str(dest.relative_to(HERE)), "sha256": IF.sha(blob), "pending": m["pending"], "problems": pr}, indent=1))


PAIRED_CLASS = "H_sim cell without an admitted H_A member (paired-pool rule)"  # 09 RULE item 26 @ 3fc5cbce5 (Append 39 §4 stands)
# 09 (10-16 ~04:4x): item 26 governs ONLY S_paper493's H_sim(P); S_old's tau_sim keeps t48 and t115. The list for both builds, by key:
PAIRED_EXPECTED = [("sp493_g4_simh_v1_try2__sim_honest", "sp493_t196|t115_model_registry")]


def paired_exclusions(m):
    """S_paper493 H_sim(P) cells (the base's extensions.S_paper493.simh keys) without an admitted honest_a member in the uncut inventories, by
    key: excluded from H_sim(P) (attack cells and N unchanged). None while the base or an inventory it needs is pending."""
    if not is_pin(m.get("base")) or not all(is_pin(m["inventories"].get(k)) for k in ("S_new", "S_old", "S_paper493", "S_paper493_w23")):
        return None
    base = json.loads(load(m["base"], "base"))
    sp = (base.get("extensions") or {}).get("S_paper493")
    if not isinstance(sp, dict):
        return None
    ha = set()
    for k in ("S_new", "S_old", "S_newmains", "S_paper493", "S_paper493_w23"):
        if is_pin(m["inventories"].get(k)):
            ha |= {r["task"] for r in json.loads(load(m["inventories"][k], f"inventory {k}"))["runs"] if r["world"] == "honest_a"}
    return sorted(({"root": r["root"], "key": x, "class": PAIRED_CLASS} for r in sp["simh"] for x in r["keys"] if x.split("|", 1)[1] not in ha),
                  key=lambda d: (d["root"], d["key"]))


def fit_entry(k, v):
    """(e) pins-file fit entry -> {kind, files, <its pins>}; a label outside FIT_KIND or any other set of fields STOPs."""
    if k not in FIT_KIND:
        raise SystemExit(f"fits.{k}: not a retrain of full_override.RETRAIN_OF")
    kind = FIT_KIND[k][0]
    if set(v) != set(FIT_PINS[kind]) | {"files"}:
        raise SystemExit(f"fits.{k}: a {kind} fit pins exactly {sorted(FIT_PINS[kind])} + files, not {sorted(v)}")
    return {"kind": kind, "files": v["files"]} | {x: resolve(v[x]) for x in FIT_PINS[kind]}


def _same(rec, p, commit=True):
    """A record's {path, sha256[, commit]} names pin p. commit=True: the record must name the pin's commit; commit=False: a record without a
    commit passes on path + sha256, one WITH a commit must name the pin's (Codex ef_r1 #4). Commits compared as full shas."""
    if not (isinstance(rec, dict) and rec.get("path") == p["path"] and rec.get("sha256") == p["sha256"]):
        return False
    if not commit and "commit" not in rec:
        return True
    rev = lambda c: subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{c}^{{commit}}"], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    return isinstance(rec.get("commit"), str) and rev(rec["commit"]) != "" and rev(rec["commit"]) == rev(p["commit"])


def fit_problems(k, f, get):
    """(e) one fit entry, exactly as specified (507-prep 09:10). SOLA: contract.pass, contract.fit_record == the fit pin, contract.refit_match_record
    == the refit pin, refit.pass, refit.controls_fit_record == the fit pin, contract.score_files naming the files. 4f: companion.pass,
    companion.fit_record.git == the fit pin, companion.refit_match.pass, companion.score_files naming the files. Anything else is a problem."""
    kind, scorer = FIT_KIND.get(k, (None, None))
    if kind is None or f.get("kind") != kind or set(f) != {"kind", "files", *FIT_PINS[kind]} or not all(is_pin(f[x]) for x in FIT_PINS[kind]):
        return [f"fit {k}: not a {kind} fit entry (STOP)"]
    files = f["files"]
    if not (isinstance(files, dict) and sorted(files) == list(READOUTS) and all(isinstance(files[r], str) and files[r].startswith(scorer + "/")
                                                                               and files[r].count("/") == 1 and files[r].endswith(".npz")
                                                                               and len(files[r]) > len(scorer) + 5 for r in READOUTS)):
        return [f"fit {k}: files must be {{C, R}}: '{scorer}/<file>'"]
    b = {x: get(f[x], f"fit {k} {x}") for x in FIT_PINS[kind]}
    if any(v is None for v in b.values()):
        return []  # get() recorded the problem
    if kind == "sola":
        c, r = json.loads(b["contract"]), json.loads(b["refit"])
        sf = c.get("score_files") or {}
        checks = {"contract.pass": c.get("pass") is True, "contract.fit_record == the fit pin": _same(c.get("fit_record"), f["fit"]),
                  "contract.refit_match_record == the refit pin": _same(c.get("refit_match_record"), f["refit"], commit=False),
                  "refit.pass": r.get("pass") is True, "refit.controls_fit_record == the fit pin": _same(r.get("controls_fit_record"), f["fit"]),
                  "contract.score_files name the files": sf.get("scorer") == scorer and all(  # Codex ef_r1 #2: real, non-empty names only
                      isinstance(sf.get(r_), str) and sf[r_] and "/" not in sf[r_] and f"{scorer}/{sf[r_]}" == files[r_] for r_ in READOUTS)}
    else:
        c = json.loads(b["companion"])
        sf = (c.get("score_files") or {}).get(k) or {}
        names = lambda r_: [e.get("path", "") for e in sf.get(r_) or [] if isinstance(e, dict)]  # noqa: E731
        checks = {"companion.pass": c.get("pass") is True,
                  "companion.fit_record.git == the fit pin": _same((c.get("fit_record") or {}).get("git"), f["fit"], commit=False),
                  "companion.refit_match.pass": (c.get("refit_match") or {}).get("pass") is True,
                  "companion.score_files name the files": k in (c.get("probes") or []) and all(
                      names(r_) and len(names(r_)) == len(sf[r_]) and all(f"/{scorer}/" in p and p.endswith("/" + files[r_].split("/")[1]) for p in names(r_))
                      for r_ in READOUTS)}
    return [f"fit {k}: {nm} fails" for nm, ok in checks.items() if not ok]


def pools_coverage(m):
    """(f) -> (pools_required record, problems). The ep2 inventory's epoch2_admitted_subset_check (every admitted epoch-2 key routed, the
    counts consistent) covers the e12 block's epoch-2 folders; each e12 fill-in root is covered cell for cell by its FILL_SOURCE inventory
    (A25 by inventory_fill_v1, 8/8). Keys and counts only. Unpinned inputs give a problem, never a pass."""
    pr, inv = [], m["inventories"]
    if not (is_pin(m.get("e12")) and all(is_pin(inv.get(x)) for x in ("pools_ep2", "pools_fill", "S_paper493"))):
        return {"source": "inventories", "pending": "e12 or the pools_ep2 / pools_fill / S_paper493 inventory unpinned"}, ["pools_required: inputs unpinned"]
    e12 = json.loads(load(m["e12"], "e12"))
    invs = {x: json.loads(load(inv[x], f"inventory {x}")) for x in ("pools_ep2", "pools_fill", "S_paper493")}
    for x, d in invs.items():
        if d.get("status") != "COMPLETE" or d.get("stops") != [] or d.get("n_runs") != len(d.get("runs") or []):
            pr.append(f"pools_required: inventory {x} not COMPLETE, has stops, or n_runs differs from its runs")
    for x in ("pools_ep2", "pools_fill"):
        if invs[x].get("set") != "pools_ep2" or (invs[x].get("collision_test") or {}).get("pass") is not True:
            pr.append(f"pools_required: inventory {x}: set is not pools_ep2 or its collision test did not pass")
    ep2 = invs["pools_ep2"]
    sc = ep2.get("epoch2_admitted_subset_check") or {}
    rna = sc.get("routed_not_admitted")
    if not (sc.get("admitted_not_routed") == [] and isinstance(rna, list) and sc.get("routed_ep2_keys") == len(ep2["runs"])
            and sc.get("admitted_keys") == len(ep2["runs"]) - len(rna)):
        pr.append("pools_required: ep2 epoch2_admitted_subset_check: an admitted key not routed, or its counts differ from the inventory's runs")
    per = lambda d: {f: sum(r["original_folder"] == f for r in d["runs"]) for f in sorted({r["original_folder"] for r in d["runs"]})}  # noqa: E731
    ep2_f = per(ep2)
    want = sorted(e12.get("folders") or {})
    if not want or [f for f in want if not ep2_f.get(f)]:
        pr.append(f"pools_required: e12 epoch-2 folders without ep2 inventory runs: {[f for f in want if not ep2_f.get(f)]}")
    fills, roots = {}, sorted((f["root"], f["cells"]) for f in (e12.get("fill_in") or {}).values())
    for root, cells in roots:
        if root not in FILL_SOURCE:
            pr.append(f"pools_required: e12 fill-in root {root} has no FILL_SOURCE inventory (STOP)")
            continue
        src, n_ = FILL_SOURCE[root]
        tasks = sorted(r["task"].split("_")[0] for r in invs[src]["runs"] if r["original_folder"] == root)
        fills[root] = {"inventory": src, "cells": len(cells), "runs": len(tasks)}
        if tasks != sorted(cells) or len(cells) != n_:  # Codex ef_r1 #3: positive coverage, the expected count (A25 8/8, g4 14/14), one run per cell
            pr.append(f"pools_required: fill-in root {root}: {len(tasks)} runs for {len(cells)} cells in {src} (need {n_}, one per cell)")
    stray = sorted(set(per(invs["pools_fill"])) - {r for r, _ in roots})
    if sorted(r for r, _ in roots) != sorted(FILL_SOURCE) or stray:  # exactly the expected fill-in roots, and no fill inventory root outside them
        pr.append(f"pools_required: no e12 fill-in roots, or fill inventory roots outside the e12 block: {stray}")
    rec = {"source": "inventories", "ep2": inv["pools_ep2"], "fill": inv["pools_fill"], "S_paper493": inv["S_paper493"],
           "coverage": {"ep2": {"runs": len(ep2["runs"]), "admitted": sc.get("admitted_keys"), "routed_not_admitted": len(rna or []),
                                "subset_check_artifact": sc.get("artifact"), "e12_folders": {f: ep2_f.get(f, 0) for f in want}},
                        "fill_in": fills}}
    return rec, pr


def combined_overrides(m):
    """The runner's lookup (run_peraction311 --inputs-full): per set, its pinned C / R overrides plus its part and IL overrides, combined
    (full_override.combine: a shared key with different entries STOPs); pools_ep2 from its C / R pair. Pending entries are left out."""
    ld = lambda k: json.loads(load(m["overrides"][k], f"override {k}"))  # noqa: E731
    out = {}
    for st in SETS:
        ks = [k for k in m["overrides"] if is_pin(m["overrides"][k]) and (k in (f"{st}|C", f"{st}|R") or st != "pools_ep2" and (
            k.split("|")[0].startswith(st + "+") or k in (f"IL:{st}|C", f"IL:{st}|R")))]
        out[st] = FO.combine([ld(k) for k in sorted(ks)])
    return out


def freeze_checks(m):
    """0d's by-key input-check fields (keys, paths, shas only): the lead's epoch-2 pool keys per set and pool vs the ep2 inventory's admitted keys;
    the tau_sim composition; the e12 fill-in root -> uncut entry map per file, looked up as the runner does; problems found (FROZEN needs none)."""
    import hashlib

    if not (is_pin(m.get("e12")) and is_pin(m["inventories"].get("pools_ep2", {}))):
        return None
    pr = []
    e12 = json.loads(load(m["e12"], "e12"))
    inv = json.loads(load(m["inventories"]["pools_ep2"], "inventory pools_ep2"))
    sc = inv.get("epoch2_admitted_subset_check")
    if not (isinstance(sc, dict) and isinstance(sc.get("routed_not_admitted"), list)):  # Codex 38fae86e3 #2: no metadata, no admission
        pr.append("ep2 inventory: no epoch2_admitted_subset_check.routed_not_admitted list")
        sc = {"routed_not_admitted": [{"key": r["key"]} for r in inv["runs"]]}
    adm = {r["key"].removeprefix("runs/") for r in inv["runs"]} - {x["key"].removeprefix("runs/") for x in sc["routed_not_admitted"]}
    hs = lambda xs: hashlib.sha256(json.dumps(sorted(xs)).encode()).hexdigest()  # noqa: E731
    phys = lambda ents: sorted({f"{e['launch_id']}/{e['eval']}#e0" for e in ents})  # noqa: E731
    pools, used = {}, set()
    have_ = {org: sorted(k for k in (P or {}) if k != "H_A_group_of_family") for org, P in (e12.get("pools") or {}).items()}
    if have_ != {org: sorted(v) for org, v in E12_POOLS.items()}:
        pr.append(f"e12 pools: origins / pools {have_} differ from the expected {E12_POOLS}")
    if sorted((((e12.get("pools") or {}).get("S_new") or {}).get("H_A") or {})) != sorted(E12_S_NEW_H_A_GROUPS):
        pr.append("e12 pools: S_new H_A family groups differ from the expected four")
    for org, P in sorted(e12["pools"].items()):  # Codex 38fae86e3 #3: every pool recorded (empty ones with n 0); a malformed one is a problem
        for nm, v in sorted(P.items()):
            if nm == "H_A_group_of_family":
                continue  # the family -> H_A group map (not a pool)
            if isinstance(v, dict) and isinstance(v.get("e2"), list):
                groups = {None: v["e2"]}
            elif nm == "H_A" and isinstance(v, dict) and v and all(isinstance(x, dict) and isinstance(x.get("e2"), list) for x in v.values()):
                groups = {g: x["e2"] for g, x in v.items()}
            else:
                pr.append(f"e12 pool {org}|{nm}: neither {{e2: [...]}} nor H_A family groups of {{e2: [...]}}")
                continue
            ents = [e for g in groups.values() for e in g]
            used |= set(phys(ents))
            pools.setdefault(org, {})[nm] = {"n": len(phys(ents)), "physical_members_sha256": hs(phys(ents)), "e2_members_sha256": hs({e["key_e2"] for e in ents})}
            if None not in groups:
                pools[org][nm]["groups"] = {g: {"n": len(phys(x)), "physical_members_sha256": hs(phys(x))} for g, x in sorted(groups.items())}
    try:
        ovs = combined_overrides(m)
    except SystemExit as x:
        ovs = {}
        pr.append(f"overrides do not combine as the runner combines them: {x}")
    fits = {k: f for k, f in m["fits"].items() if "pending" not in f}
    fill, fills = {}, sorted((e12.get("fill_in") or {}).values(), key=lambda f: f["root"])
    if not fills or any(not f.get("files") for f in fills):  # Codex 38fae86e3 #5: no fill-in, or a fill-in without files, is a problem, not a pass
        pr.append("e12: no fill-in roots, or a fill-in root without files")
    for f in fills:  # Codex 38fae86e3 #4: the registered names and every retrain's uncut names (the runner's aliases), across the combined sets.
        names = sorted(f["files"]) + sorted({fn for x in fits.values() for fn in x["files"].values()})  # Retrains are required at every root
        for fn in names:  # (stricter than the runner, which needs one only where the root holds its base file; every root holds every base today)
            hits = [(st, e) for st, o in sorted(ovs.items()) for e in (o.get("simh") or {}).get(fn, []) if e.get("root") == f["root"]]
            fill[f"{f['root']}|{fn}"] = [{"set": st, "path": e["path"], "sha256": e["sha256"]} for st, e in hits]
    if any(len(v) != 1 for v in fill.values()):
        pr.append(f"fill-in files without exactly one uncut entry: {[k for k, v in fill.items() if len(v) != 1][:6]}")
    if used - adm:
        pr.append(f"epoch-2 members used but not admitted: {sorted(used - adm)[:3]}")
    base = json.loads(load(m["base"], "base")) if is_pin(m.get("base")) else {}
    sp = (base.get("extensions") or {}).get("S_paper493") or {}
    adm_ = {f["root"]: {"n": len(f.get("admitted") or []), "admitted": sorted(f.get("admitted") or [])} for f in fills}
    return {"e12_pools": pools, "e12_physical_keys_used": len(used), "ep2_admitted": len(adm), "used_not_admitted": sorted(used - adm)[:20],
            "admitted_not_used": len(adm - used), "admitted_not_used_rule": "T196 epoch-2 honest_a (09 Q4: S_old H_A gets no epoch 2) and other unpooled admits",
            "tau_sim": {"S_old": {"registered": "primary_v3 (A25 b1 + b2)", "fill_in_admitted": adm_,
                                  "paired_rule": "not applied (09: item 26 governs S_paper493 only; t48 and t115 stay)"},
                        "S_paper493": {"roots": [r["root"] for r in sp.get("simh", [])], "paired_exclusions": m.get("paired_pool_exclusions")}},
            "fill_in_map": fill, "fill_in_map_ok": bool(fill) and all(len(v) == 1 for v in fill.values()), "problems": pr}


def waitlist_pending(m, test):
    """The waitlist rule on the full-length inputs: make_waitlist.py check (via inputs_final_v2.waitlist_check) with the UNCUT census (every
    headline probe scored each waitlisted cell uncut, or it failed terminally and is classed). Not run until all three pins exist."""
    if not all(is_pin(m.get(k)) for k in ("waitlist", "exclusions", "census")):
        return ["waitlist not checked (waitlist, exclusions or uncut census pending)"]
    cen = json.loads(load(m["census"], "census"))
    fl = (cen.get("meta") or {}).get("full_length") or {}
    if not (fl.get("overrides") == m["overrides"] and (cen["meta"].get("inputs_final") or {}).get("sha256") == m["base"].get("sha256")):
        raise SystemExit("census: not a full-length census on these overrides and this base (a 32k census never resolves a full-length cell)")
    ex = json.loads(load(m["exclusions"], "exclusions"))
    final = "TEST" not in json.dumps({k: ex.get(k) for k in ("label", "status")})
    rel = lambda e: f"{e['path'].removeprefix('experiments/2026-10-15_fpr_sweep_507_prep/')}@{e['sha256']}"  # noqa: E731
    r = subprocess.run([sys.executable, str(HERE / "make_waitlist.py"), "check", "--waitlist", rel(m["waitlist"]), "--exclusions", rel(m["exclusions"]),
                        "--census", rel(m["census"]), "--probes", ",".join(FO.HEADLINE_SIX), "--readouts", ",".join(READOUTS)]
                       + (["--final"] if final else []),  # 0d + 09: the IL extra arms are report-only (make_waitlist's default; asserted below)
                       capture_output=True, text=True, cwd=HERE)
    try:
        out = json.loads(r.stdout)
    except ValueError:
        raise SystemExit(f"make_waitlist.py check: unreadable output (rc {r.returncode}): {r.stdout[-300:]} {r.stderr[-300:]}")
    pend = list(out.get("pending") or []) + [f"class pending 09: {x}" for x in out.get("pending_09_class") or []]
    if r.returncode != 0 and not pend:
        raise SystemExit(f"make_waitlist.py check: rc {r.returncode} without a pending list: {r.stderr[-300:]}")
    if out.get("extra_arms_gate") is not False:
        raise SystemExit("make_waitlist.py check: the IL extra arms must be report-only (0d + 09)")
    if sorted(out.get("labels") or []) != sorted(f"{h} {ro}" for h in FO.HEADLINE_SIX for ro in READOUTS):
        raise SystemExit(f"make_waitlist.py check: labels checked {out.get('labels')} are not the full-length headline six")
    return pend


def ood_missing(table):
    """Every INPUTS_FULL label needs a signed (non-pending) training-length row with a class (09: the verifier requires one per label at the
    freeze). Rows are '<name> (<C|R>[; ...])'; a label '<family>|uncut <ro>' reads its registered family's row."""
    import re

    idx, pend = {}, set()
    for r in table.get("rows") or []:
        mo = re.match(r"^(.*?)\s*\((R|C)\b[^)]*\)\s*$", r.get("probe") or "")
        if mo and r.get("class_key"):
            idx[(mo.group(1), mo.group(2))] = r
    for x in table.get("pending") or []:
        mo = re.match(r"^(.*?)\s*\((R|C)\b[^)]*\)\s*$", x)
        if mo:
            pend.add((mo.group(1), mo.group(2)))
    out = []
    for lb in FO.LABELS_OUT:
        fam, ro = lb.rsplit(" ", 1)
        k = (fam.removesuffix(FO.UNCUT), ro)
        if k not in idx or k in pend:
            out.append(lb)
    return out


def pending_items(m):
    out = []
    for k in ("base", "pools_required", "cohort", "exclusions", "waitlist", "census", "e12", "ilarms", "relaunch_members", "cutmap", "dryrun_counts"):
        e = m.get(k) if isinstance(m.get(k), dict) else {"pending": "missing"}
        if "pending" in e:
            out.append(f"{k}: {e['pending']}")
    for blk in ("overrides", "gates", "inventories", "fits"):
        for k, v in m[blk].items():
            if "pending" in v:
                out.append(f"{blk}.{k}: {v['pending']}")
    for st in SETS:
        for ro in READOUTS:
            for blk in ("overrides", "gates"):
                if f"{st}|{ro}" not in m[blk]:
                    out.append(f"{blk}.{st}|{ro}: missing")
    for k in FO.RETRAIN_OF:
        if k not in m["fits"]:
            out.append(f"fits.{k}: missing")
    if m["waitlist_pending"]:
        out.append(f"waitlist_pending: {m['waitlist_pending']}")
    for k, e in m["overrides"].items():  # a pre-schema override (no "set" field, e.g. the decisive members' (a'') files) is TEST-only
        if is_pin(e) and "set" not in json.loads(load(e, f"override {k}")):
            out.append(f"overrides.{k}: pre-schema override (no set field; decisive members only): TEST only")
    if "pending" in m["roles_old"]:
        out.append(f"roles_old: {m['roles_old']['pending']}")
    if is_pin(m.get("ood")):
        miss = ood_missing(json.loads(load(m["ood"], "ood")))
        if miss:
            out.append(f"ood: labels without a signed training-length row: {miss}")
    return out


def has_cut(cm, st, ro, folder, fn, simh, roles_old, extra_roots=()):
    """Does the pinned 32k cut map hold a registered cut counterpart for this uncut entry? (n/a on inside_32k_equal is accepted only when not)
    09 route C (B3): an allowlisted S_old sim-honest root (s_old_simh_extra_roots) has none; every other S_old root keeps the file-name rule."""
    d = (cm.get(st) or {}).get(ro) or {}
    if st == "S_old":
        role = roles_old.get(folder)
        if simh and role == "simh" and folder in extra_roots:
            return False
        return (fn in (d.get("old") or {}) if role == "real" else fn in (d.get("relaunch") or {}) if role == "relaunch"
                else any(k.endswith("|" + fn) for k in d.get("simh") or {}))
    if simh:
        return f"{folder}|{fn}" in (d.get("simh") or {})
    return fn in ((d.get("arms") or {}).get(folder) or {}) or any(k.rsplit("|", 1)[0].split("/", 1)[0] == folder and k.endswith("|" + fn) for k in d.get("ha25f") or {})  # '<root>[/<sub>]|<file>'


def extra_roots_problems(m, get):
    """09 route C (B3; Astra R1 / R2) -> (problems, the allowlist or None). Absent: nothing. Present: exactly the approved allowlist, its pin
    readable at its sha (get), its 20 keys valid, its root a sim-honest folder of the S_old inventory (roles_old); null or anything else is a problem."""
    pr, xr_ = [], None
    if "s_old_simh_extra_roots" not in m:
        return pr, xr_
    try:
        xr_ = FO.s_old_extra_pin(m)
    except SystemExit as x:
        pr.append(str(x))
    for root, e in (xr_ or {}).items():
        b = get(e, f"s_old_simh_extra_roots {root}")
        if b is not None:
            try:
                FO.s_old_extra_keys(json.loads(b))
            except SystemExit as x:
                pr.append(f"s_old_simh_extra_roots {root}: {x}")
        if (m.get("roles_old") or {}).get(root) != "simh":
            pr.append(f"s_old_simh_extra_roots {root}: not a sim-honest folder of the S_old inventory (roles_old)")
    return pr, xr_


def entry_problems(st, ro, ov, gate_pin, gate, retrain_files, inv_staged, na_folders=((), ()), cm=None, roles_old=None, extra_roots=()):
    """Every override entry: path, sha256, staged_shas, runs_without_rows, gate (this set's gate pin, its own file key, four flags true; NA only
    for a retrain's file or a set outside the 32k inputs), staged sha bound to the inventory row (non-(a'') sets)."""
    pr = []
    ents = [(f, fn, e, False) for f, files in (ov.get("arms") or {}).items() for fn, e in files.items()]
    ents += [(e.get("root"), fn, e, True) for fn, es in (ov.get("simh") or {}).items() for e in es]
    if not ents:
        pr.append(f"{st}|{ro}: override has no entries")
    gidx = {g.get("file"): g for g in (gate.get("files") or [])} if isinstance(gate.get("files"), list) else (gate.get("files") or {})
    for folder, fn, e, is_simh in ents:
        tag = f"{st}|{ro}|{folder}|{fn}"
        if not (isinstance(e.get("path"), str) and isinstance(e.get("sha256"), str) and len(e["sha256"]) == 64 and e.get("staged_shas")
                and isinstance(e.get("runs_without_rows"), list)):
            pr.append(f"{tag}: entry fields")
        g = e.get("gate") or {}
        if not ({k: g.get(k) for k in ("path", "sha256")} == {k: gate_pin[k] for k in ("path", "sha256")}  # the file by content (any commit holding it)
                and g.get("file") == f"{folder}|{fn}"):
            pr.append(f"{tag}: gate pointer is not this set's gate record or names another file")
            continue
        fl = gidx.get(f"{folder}|{fn}")
        if fl is None:
            pr.append(f"{tag}: not in the gate record")
            continue
        if fl.get("sha256") != e.get("sha256"):  # the gate tested THIS file (Codex r1 #6)
            pr.append(f"{tag}: the gate record's sha256 is not the entry's")
        for k in FLAGS:
            v = fl.get(k)
            if v is True:
                continue
            if k == "inside_32k_equal" and v == NA:
                if cm is not None:  # the rule: n/a iff the pinned 32k cut map has no registered counterpart for this entry
                    if not has_cut(cm, st, ro, folder, fn, is_simh, roles_old or {}, extra_roots):
                        continue
                elif fn in retrain_files or st == "pools_ep2" or (folder in na_folders[0] and fn not in na_folders[1]):  # TEST without a cut map
                    continue
            pr.append(f"{tag}: gate flag {k} = {v!r}")
        if inv_staged is not None and not (set(inv_staged.get(folder, set())) and set(inv_staged[folder]) <= set(e.get("staged_shas") or [])):
            pr.append(f"{tag}: staged_shas do not cover the inventory's uncut staged sha for {folder}")
    return pr, len(ents)


def required(m):
    """Every item a FROZEN file must pin: name -> entry (a pin, or in TEST only {"pending": <non-empty string>})."""
    out = {k: m.get(k) for k in ("base", "pools_required", "ood", "cohort", "exclusions", "waitlist", "census", "e12", "ilarms", "relaunch_members",
                                 "cutmap", "dryrun_counts")}  # Codex 38fae86e3 #1: the dry-run counts sidecar is required too
    out["ood.signature"] = (m.get("ood") or {}).get("signature")
    main_ = [f"{st}|{ro}" for st in SETS for ro in READOUTS]  # Codex d14ba40d9 #1: every pinned part / IL key is required to be a full pin as well
    for blk, keys in (("overrides", main_ + sorted(set(m.get("overrides") or {}) - set(main_))),
                      ("gates", main_ + sorted(set(m.get("gates") or {}) - set(main_))),
                      ("inventories", list(INVENTORY_ORDER)), ("fits", list(FO.RETRAIN_OF))):
        for k in keys:
            out[f"{blk}.{k}"] = (m.get(blk) or {}).get(k)
    return out


def key_problems(m):
    """Codex d14ba40d9 #3: the override / gate keys the runner reads, and only those: "<set>|<ro>" (set in SETS), "<set>+<part>|<ro>" and
    "IL:<set>|<ro>" (set in full_override.SETS). Every override key has a gate key and every gate key an override key."""
    import re

    pr = []
    for k in sorted(set(m.get("overrides") or {}) | set(m.get("gates") or {})):
        mo = re.fullmatch(r"(IL:)?([A-Za-z0-9_]+?)(?:\+([a-z0-9_]+))?\|(C|R)", k)
        if not mo or (mo.group(1) or mo.group(3)) and mo.group(2) not in FO.SETS or mo.group(2) not in SETS or (mo.group(1) and mo.group(3)):
            pr.append(f"override / gate key {k!r}: not '<set>|<ro>', '<set>+<part>|<ro>' or 'IL:<set>|<ro>' with a known set")
        if k not in (m.get("overrides") or {}) or k not in (m.get("gates") or {}):
            pr.append(f"override / gate key {k!r}: in only one of overrides and gates")
    return pr


def schema_problems(m):
    test, pr = bool(m.get("test")), []
    for k, e in required(m).items():
        if k.startswith("fits.") and isinstance(e, dict) and "kind" in e:
            ok = e["kind"] in FIT_PINS and all(is_pin(e.get(x)) and len(e[x]["sha256"]) == 64 for x in FIT_PINS[e["kind"]])
        elif k == "pools_required" and isinstance(e, dict) and e.get("source") == "inventories" and "pending" not in e:
            ok = all(is_pin(e.get(x)) and len(e[x]["sha256"]) == 64 for x in ("ep2", "fill", "S_paper493"))
        else:
            ok = is_pin(e) and len(e["sha256"]) == 64
        if not ok and not (test and isinstance(e, dict) and isinstance(e.get("pending"), str) and e["pending"]):
            pr.append(f"{k}: not a pin" + ("" if test else " (FROZEN needs every pin)"))
    return pr


# 09 + 0d (relayed by 2e ~03:3x; CUT's il_fix_v1 @ 2b3e6a137): ONE pinned folder exception. Its 66 rows name the registered 32k pack's
# folder (_try2) while the r1 routing record maps the arm to the pre-_try2 original; accepted only for exactly this folder, routing commit and
# staged sha, and only on rows that carry original_folder_exception. Any other mismatch STOPs.
FOLDER_EXCEPTIONS = {"ilnm_s5_sim2_v1_try2__sim_attack": {"routing_original_folder": "ilnm_s5_sim2_v1__sim_attack", "routing_commit": "a668d0432d2a",
                                                          "staged_sha256": "0630d25cf76f39362e40ed5cfaf62028932c4ec0b87a021dcd05a26a4cde304c"}}


def _rev(name):
    """(base, revision) of a routing record file name: '<base>.json' is revision 1, '<base>_r<k>.json' revision k."""
    import re

    mo = re.match(r"^(.*?)(?:_r(\d+))?\.json$", name)
    return mo.group(1), int(mo.group(2) or 1)


def format_problems(inv_pin, inv, allow_partial=False):
    """Non-(a'') sets: per folder, the inventory's uncut_planned format check (passes on every epoch at its staged sha) and routing record (the
    highest revision for its target; the revision the row states)."""
    pr, by, kb, exc_rows = [], {}, {}, {}
    if not (inv.get("status") in (("COMPLETE", "PARTIAL") if allow_partial else ("COMPLETE",)) and not inv.get("stops") and not inv.get("problems")):
        pr.append(f"inventory {inv_pin['path']}: status {inv.get('status')} / stops / problems")
    for r in inv.get("runs") or []:
        u = r.get("uncut_planned") or {}
        by.setdefault(r["original_folder"], []).append(u)
        kb.setdefault(r["original_folder"], set()).add(str(r["key"]).removeprefix("runs/"))  # some stagers prefix keys with runs/ (both sides)
        exc_rows.setdefault(r["original_folder"], []).append(r)
    for folder, us in sorted(by.items()):
        keys = {(u.get("staged_sha256"), u.get("format_check"), u.get("routing"), u.get("revision")) for u in us}
        if len(keys) != 1 or None in next(iter(keys)):
            pr.append(f"{folder}: rows disagree on staged sha / format check / routing / revision, or one is missing")
            continue
        sh, fc, rt, rev = next(iter(keys))
        u0 = us[0]
        try:
            fp, fcm = [x.strip() for x in fc.split("@")]
            f = json.loads(IF.git_show(fcm, fp))
            fk = {str(x.get("key")).removeprefix("runs/") for x in f.get("rows") or []}  # a staged set may serve several original folders: by key
            if not (f.get("passes") is True and f.get("failing_keys") == [] and f.get("shard_errors") == [] and sh in (f.get("staged_sha") or [])
                    and f.get("epochs_checked") == len(f.get("rows") or []) and kb[folder] <= fk):
                pr.append(f"{folder}: format check does not pass on its {len(us)} epochs (by key) at staged sha {sh[:12]}")
            rp, rc = [x.strip() for x in rt.split("@")]
            rec = json.loads(IF.git_show(rc, rp))
            base, k = _rev(Path(rp).name)
            tree = subprocess.run(["git", "ls-tree", "--full-tree", "--name-only", rc, str(Path(rp).parent) + "/"], capture_output=True, check=True, cwd=HERE).stdout.decode().split()
            top = max((_rev(Path(x).name)[1] for x in tree if Path(x).name.endswith(".json") and _rev(Path(x).name)[0] == base), default=-1)
            cf, of = rec.get("cutfull") or {}, rec.get("original_folders") or {}
            # exact combination (Codex r3): S_old attestations: the same world names this original folder AND this uncut folder of this staged
            # set (its staged sha is bound by the format check above); chain routing records: the arm whose meta arm maps to this folder
            # carries this staged sha
            sold = cf.get("staged_set") == u0.get("staged_set") and any(of.get(w) == folder and (cf.get("folders") or {}).get(w) == u0.get("folder") for w in of)
            chain = any(of.get(str(a.get("meta_arm", "")).split("@")[0]) == folder and a.get("staged_sha") == sh for a in rec.get("arms") or [])
            if not chain and folder in FOLDER_EXCEPTIONS and all(u_.get("original_folder_exception") for u_ in exc_rows.get(folder, [])):
                want_ = FOLDER_EXCEPTIONS[folder]  # 09 + 0d's pinned one-folder exception: the routing record names the pre-_try2 original folder
                chain = any(of.get(str(a.get("meta_arm", "")).split("@")[0]) == want_["routing_original_folder"] and a.get("staged_sha") == sh
                            and sh == want_["staged_sha256"] and rc.startswith(want_["routing_commit"]) for a in rec.get("arms") or [])
            if not (sold or chain) or rec.get("failing", {}) not in ({}, None):
                pr.append(f"{folder}: routing record {Path(rp).name} does not bind the inventory's staged set / sha {sh[:12]}, or lists failing sets")
            if not (int(k) == rev == top and rec.get("revision", rev) == rev):
                pr.append(f"{folder}: routing {Path(rp).name} is not the highest revision at {rc} (r{k}, top r{top}) or disagrees with the row")
        except (ValueError, subprocess.CalledProcessError, SystemExit) as x:
            pr.append(f"{folder}: format/routing pointers unreadable: {str(x)[:120]}")
    return pr


def verify(m, deep=False, want_relax=False):
    """-> problems (empty = pass). Fail closed: any unreadable pin, sha difference or missing field is a problem."""
    require_pin_shas(m)  # 09 (10-17): a null or non-64-hex pin sha256 anywhere STOPs (SystemExit), not a listed problem
    pr = []
    if m.get("schema") != SCHEMA:
        return [f"schema {m.get('schema')!r}"]
    pr += schema_problems(m) + key_problems(m)
    frozen = str(m.get("status", "")).startswith("FROZEN")
    if frozen and (m.get("test") or m.get("pending")):
        pr.append("FROZEN with test set or pending items")
    if m.get("lead_readout") != "C":
        pr.append("lead_readout must be C")
    if m.get("scope", {}).get("registered_labels") != list(FO.SCOPE) or m["scope"].get("retrains") != FO.RETRAIN_OF:
        pr.append("scope differs from full_override's")
    if m.get("pending") != pending_items(m):
        pr.append("pending list is not the derived one")

    def get(e, what):
        try:
            return load(e, what)
        except (SystemExit, subprocess.CalledProcessError) as x:
            pr.append(f"{what}: {x}")
            return None

    if is_pin(m["base"]):
        b = get(m["base"], "base")
        if b is not None:
            base = json.loads(b)
            if base.get("schema") != "inputs_final/v2":
                pr.append("base is not an inputs_final/v2 file")
            if frozen and not (str(base.get("status", "")).startswith("FROZEN") and not base["refresh2"]["test"]):
                pr.append("FROZEN INPUTS_FULL needs a FROZEN non-TEST base")
            if deep:
                import inputs_final_v2 as IF2

                pr += [f"base: {x}" for x in IF2.verify(base, base["refresh2"]["pins"]["path"])]
    retrain_files = {fn for f in m["fits"].values() if "files" in f for fn in f["files"].values()}
    rows_by_key, v3_folders = {}, set()
    for st in INVENTORY_ORDER:  # in order: a later inventory's row supersedes an earlier one's by key (il_fix after il)
        e = m["inventories"].get(st)
        if not is_pin(e):
            continue
        b = get(e, f"inventory {st}")
        if b is None:
            continue
        inv = json.loads(b)
        if st == "S_new":  # (a''): reattest_map below
            v3_folders = {r["original_folder"] for r in inv["runs"]}
            continue
        pr += [f"inventory {st}: {x}" for x in format_problems(e, inv, allow_partial=st in ("il", "il_fix"))]  # IL arms are optional (0d)
        for r in inv["runs"]:
            rows_by_key[r["key"]] = r
    glob = {}  # one staged map over every non-(a'') inventory (IL folders sit in the S_old / S_new / S_newmains overrides)
    for r in rows_by_key.values():
        sh = (r.get("uncut_planned") or {}).get("staged_sha256")
        if sh:
            glob.setdefault(r["original_folder"], set()).add(sh)
            glob.setdefault((r.get("uncut_planned") or {}).get("folder"), set()).add(sh)
    if is_pin(m["inventories"].get("S_newmains", {})):  # 0d / 09: the uncut side carries no S4 orphan (d9's map, staged entries, by key)
        smap = json.loads(IF.git_show(*S4MAP))
        orph = {r["record"] + "#e0" for r in smap["outside_source_map"] if r.get("staging") == "staged"}
        invn = json.loads(load(m["inventories"]["S_newmains"], "inventory S_newmains (S4 orphans)"))
        out_ = {x["key"] for x in invn.get("s4_orphans_out") or []}
        if out_ != orph or len(orph) != S4_N:
            pr.append(f"S4 orphans: the inventory's s4_orphans_out ({len(out_)}) differs from d9's map's staged orphans ({len(orph)}) by key")
        if {r["key"] for r in invn["runs"]} & orph:
            pr.append("S4 orphans: an inventory row is an S4 orphan")
    if "pending" not in m.get("roles_old", {"pending": "missing"}) and is_pin(m["inventories"].get("S_old", {})):
        b = get(m["inventories"]["S_old"], "inventory S_old (roles)")
        if b is not None and roles_from_inventory(json.loads(b)) != m["roles_old"]:
            pr.append("roles_old differs from the S_old inventory's worlds")
    p_xr, xr_ = extra_roots_problems(m, get)
    pr += p_xr
    relax = {"reattest_skip": [], "union_exempt": []}  # 0d's two relaxations (05:2x), every file they touch, by path + sha256
    cm_ = None  # the 32k cut map CUT's gates compare against: built by the runner (--cutmap) on THIS base
    if is_pin(m.get("cutmap")):
        cmj_ = json.loads(load(m["cutmap"], "cutmap"))
        if not (is_pin(m.get("base")) and str(cmj_.get("inputs_final", "")).endswith("@" + m["base"]["sha256"])):
            pr.append("cutmap: not built on this base")
        cm_ = cmj_["map"]
        if not all(isinstance(((cm_.get("S_old") or {}).get(ro) or {}).get("old"), dict) and cm_["S_old"][ro]["old"] for ro in READOUTS):
            pr.append("cutmap: no S_old.<C|R>.old block (Codex 92cded3d6 #2: absence of a combined file is never read from a partial map)")
            cm_ = None  # no exemption without the block
    # Codex d14ba40d9 #2: an override's entries must be files of ITS readout (the runner combines C and R into one lookup): the 32k cut map's file
    # names per readout plus the retrains' uncut names; a name of the other readout, or of neither, is a problem
    ro_files = {ro: set() for ro in READOUTS}
    for ro in READOUTS:
        for d in (v.get(ro) or {} for v in (cm_ or {}).values()):
            for blk, x in d.items():
                ro_files[ro] |= ({fn for fs in x.values() for fn in fs} if blk == "arms" else {k.split("|", 1)[-1] for k in x})
        ro_files[ro] |= {f["files"][ro] for f in m["fits"].values() if "files" in f}
    if ro_files["C"] & ro_files["R"]:
        pr.append(f"readout file names shared by C and R: {sorted(ro_files['C'] & ro_files['R'])[:4]}")
    for key, e in m["overrides"].items():
        st, ro = key.split("|")
        il_ = st.startswith("IL:")  # optional IL extra-arm overrides (report-only): checked like the others when pinned, never required
        st = st.removeprefix("IL:")
        st, _, part_ = st.partition("+")  # a part override of a set ("S_new+25f"): same contract, ov.part names it
        if not is_pin(e):
            continue
        b = get(e, f"override {key}")
        gp = m["gates"].get(key, {})
        gb = get(gp, f"gate {key}") if is_pin(gp) else None
        if b is None:
            continue
        ov = json.loads(b)
        if il_ and "IL extra arms" not in str(ov.get("label")):
            pr.append(f"override {key}: not labelled an IL extra-arm override")
        if part_ and ov.get("part") != part_:
            pr.append(f"override {key}: its part field is {ov.get('part')!r}, not {part_!r}")
        if "set" not in ov and m.get("test"):  # pre-schema (a'') override: TEST only (listed in pending); its re-attestation still runs below
            ov = dict(ov, set=st, readout=ov.get("readout", ro))
        if (ov.get("set"), ov.get("readout"), ov.get("status"), ov.get("missing"), ov.get("manifests_without_staged_sha")) != (st, ro, "COMPLETE", [], []):
            pr.append(f"override {key}: set/readout/status/missing/manifests_without_staged_sha")
        fns_ = {fn for x in (ov.get("arms") or {}).values() for fn in x} | {fn for fn, ents in (ov.get("simh") or {}).items() if ents}
        if not fns_:  # Codex d14ba40d9 #4 / d6d9e5a33 #3: an override with no entries is never COMPLETE
            pr.append(f"override {key}: no arm or H_sim entries")
        other_ = [fn for fn in sorted(fns_) if fn not in ro_files[ro] or fn in ro_files[{"C": "R", "R": "C"}[ro]]]
        if other_:
            pr.append(f"override {key}: files not of readout {ro} (cut map / retrain names): {other_[:4]}")
        v3_ = (any(f in v3_folders for f in (ov.get("arms") or {}))  # Codex 92cded3d6 #1 / d6d9e5a33 #2: (a'') folders AND (a'') sim-honest roots
               or any(x.get("root") in v3_folders for ents in (ov.get("simh") or {}).values() for x in ents))
        if gb is None and not ("pending" in gp and m.get("test")):
            pr.append(f"override {key}: no gate record")
            continue
        if gb is not None:
            ov_v3 = {"arms": {f: x for f, x in (ov.get("arms") or {}).items() if f in v3_folders}, "simh": ov.get("simh") or {}} if st == "S_new" else None
            ov_rest = dict(ov, arms={f: x for f, x in (ov.get("arms") or {}).items() if f not in v3_folders}, simh={}) if st == "S_new" else ov
            rl_ = ({f for f, r in (m.get("roles_old") or {}).items() if r == "relaunch"} if st == "S_old" else set(), NA_RELAUNCH_WITH_CUT)
            p_, _ = (entry_problems(st, ro, ov_rest, gp, json.loads(gb), retrain_files, glob, rl_, cm_, m.get("roles_old"), xr_ or {})
                     if (st != "S_new" or ov_rest["arms"]) else ([], 0))
            if st == "S_old":  # the S_old real line is one combined cut file per probe: the real folders' uncut keys must be exactly its keys
                gf_ = json.loads(gb).get("files") or {}
                real_ = [f for f, r in (m.get("roles_old") or {}).items() if r == "real"]
                for fn_ in sorted({fn for f in real_ for fn in (ov.get("arms") or {}).get(f, {})}):
                    if cm_ is not None and fn_ not in ((cm_.get("S_old") or {}).get(ro) or {}).get("old", {}):  # 0d (b)(ii): ONLY a file with NO
                        relax["union_exempt"].append({"override": key, "file": fn_, "entries": sorted(  # 32k combined file (no cut map: no exemption)
                            [f, ov["arms"][f][fn_]["path"], ov["arms"][f][fn_]["sha256"]] for f in real_ if fn_ in (ov.get("arms") or {}).get(f, {}))})
                        continue
                    if (gf_.get(f"S_old|{fn_}") or {}).get("union_keys_equal") is not True:
                        p_.append(f"S_old|{ro}|{fn_}: the gate record does not state union_keys_equal for the real line")
            if st == "S_new" and (ov_v3["arms"] or ov_v3["simh"]):  # the (a'') folders and roots: gate flags here, staged shas by re-attestation
                p2_, _ = entry_problems(st, ro, ov_v3, gp, json.loads(gb), retrain_files, None, cm=cm_, roles_old=m.get("roles_old"))
                p_ += p2_
            pr += p_
        if st == "S_new" and not il_ and not v3_:  # 0d (b)(i): the skip fires ONLY on a part override with 0 v3 folders and 0 v3 roots;
            if not part_ or not fns_:  # any other S_new override without them (or an empty part) is a STOP
                pr.append(f"override {key}: an S_new override with no (a'') folder or root that is not a non-empty part override")
            else:
                relax["reattest_skip"].append({"override": key, "files": sorted(
                    [[f"{f}|{fn}", x_["path"], x_["sha256"]] for f, x in (ov.get("arms") or {}).items() for fn, x_ in x.items()]
                    + [[f"H_sim {x_.get('root')}|{fn}", x_["path"], x_["sha256"]] for fn, ents in (ov.get("simh") or {}).items() for x_ in ents])})
        if st == "S_new" and not il_ and v3_ and is_pin(m["inventories"].get("S_new", {})):  # (a''): d9's re-attestation, every entry's staged sha
            inv = json.loads(load(m["inventories"]["S_new"], "inventory S_new"))
            git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
            try:
                st_, _ = CD.reattest_map(m["inventories"]["S_new"]["reattest"], git, {r["original_folder"] for r in inv["runs"]}, len(inv["runs"]))
                bad, n = CD.reattest_bad({"arms": {f: x for f, x in (ov.get("arms") or {}).items() if f in v3_folders}, "simh": ov.get("simh") or {}}, st_)
                if bad or not n:
                    pr.append(f"override {key}: staged sha differs from the re-attested one: {bad[:5]}")
            except SystemExit as x:
                pr.append(f"override {key}: reattest: {x}")
    for e in m.get("score_records") or []:
        if not is_pin(e):
            pr.append("score_records: not a pin")
        else:
            get(e, f"score record {Path(e['path']).name}")
    relax = {k: sorted(v, key=lambda x: x["override"] + "|" + x.get("file", "")) for k, v in relax.items()}  # Codex d6d9e5a33 (optional): order-free
    if want_relax:
        return relax
    if m.get("freeze_checks") != freeze_checks(m):  # recomputed from the pins (0d's by-key input check reads these fields)
        pr.append("freeze_checks differ from the recomputed fields")
    fc_ = m.get("freeze_checks")
    if fc_ is not None:
        pr += [f"freeze_checks: {x}" for x in fc_.get("problems", ["no problems field"])]
    elif frozen:
        pr.append("freeze_checks: none (e12 or the pools_ep2 inventory unpinned)")
    if is_pin(m.get("dryrun_counts")):
        get(m["dryrun_counts"], "dryrun_counts")
    if not isinstance(m.get("provisional"), bool) or m.get("labels_provisional") != (PROVISIONAL_LABELS if m.get("provisional") else []):
        pr.append("labels_provisional: provisional must be stated (true | false) and the labels must be exactly PROVISIONAL_LABELS when true")
    if not isinstance(m.get("score_records"), list) or (frozen and not m["score_records"]):
        pr.append("score_records: a list, non-empty in a FROZEN file")
    elif m["score_records"] or frozen:  # Codex r2b #11: complete by name (none dropped), no duplicates
        nm_ = [Path(e.get("path", "")).name for e in m["score_records"] if isinstance(e, dict)]
        cnt_ = {pre: sum(n.startswith(pre) for n in nm_) for pre in SCORE_RECORD_RULE}
        if cnt_ != SCORE_RECORD_RULE or len(set(nm_)) != len(nm_) or len(nm_) != sum(SCORE_RECORD_RULE.values()):
            pr.append(f"score_records: by name {cnt_}, need exactly {SCORE_RECORD_RULE} and no other or duplicate record")
    if m.get("relaxations") != relax:  # the build records the relaxations; verify recomputes them (0d checks them by key)
        pr.append("relaxations differ from the recomputed list")
    for k, f in m["fits"].items():
        if "pending" not in f:
            pr += fit_problems(k, f, get)
    if "pending" not in m["pools_required"]:  # (f): recomputed from the pinned inventories; the recorded copy must equal it
        try:
            rec_, p_ = pools_coverage(m)
        except (SystemExit, subprocess.CalledProcessError) as x:
            rec_, p_ = None, [f"pools_required: {x!r}"]
        pr += p_
        if m["pools_required"] != rec_:
            pr.append("pools_required differs from the coverage recomputed from the pinned inventories")
    if is_pin(m["ood"]):
        get(m["ood"], "ood")
        s = get(m["ood"]["signature"], "ood signature (READ_LOG)")
        if s is not None and not any("09 SIGN" in ln and m["ood"]["sha256"][:12] in ln and (Path(m["ood"]["path"]).name in ln or m["ood"]["commit"][:9] in ln)
                                     for ln in s.decode().splitlines()):
            pr.append("ood: no 09 signing line in READ_LOG naming this table (file or commit) and its sha (12 hex)")
    for k in ("cohort", "exclusions", "waitlist", "census", "e12", "ilarms", "relaunch_members"):  # every pin is read and sha-checked (Codex r7)
        if is_pin(m.get(k)):
            get(m[k], k)
    pe_ = paired_exclusions(m)
    if pe_ != m.get("paired_pool_exclusions"):
        pr.append(f"paired-pool exclusions (09 item 26) differ from the recomputed list: file {m.get('paired_pool_exclusions')} vs {pe_}")
    if frozen and pe_ is None:
        pr.append("paired-pool exclusions not computed (base or inventories pending)")
    if pe_ is not None and [(x["root"], x["key"]) for x in pe_] != PAIRED_EXPECTED:
        pr.append(f"paired-pool exclusions are not exactly 09's list {PAIRED_EXPECTED}: {pe_}")
    if pe_ is not None and is_pin(m.get("base")):  # the base (refresh2's builder) must exclude exactly these keys from H_sim(P), by the same class
        bpe_ = sorted(x["key"] for x in ((json.loads(load(m["base"], "base")).get("extensions") or {}).get("S_paper493") or {}).get("pool_exclusions", [])
                      if x.get("class") == PAIRED_CLASS)
        if frozen and bpe_ != sorted(x["key"] for x in pe_):
            pr.append(f"paired-pool exclusions: the base's pool_exclusions {bpe_} differ from the scan {[x['key'] for x in pe_]}")
    if frozen:  # the waitlist is re-checked here, never trusted from the file (Codex r1 #5)
        try:
            pend = waitlist_pending(m, False)
        except SystemExit as x:
            pend = [str(x)[:200]]
        if pend or m.get("waitlist_pending") != []:
            pr.append(f"waitlist not resolved: {pend[:5]} (file states {m.get('waitlist_pending')})")
    if m.get("estimator", {}).get("A") != "registered":
        pr.append("estimator A must be the registered estimator")
    if is_pin(m.get("ilarms")):  # 09 @ 36b9b47df §3: (A) and (B) both computed; the LOADER's ilarms/v1 block on uncut files
        il = json.loads(load(m["ilarms"], "ilarms"))
        if il.get("schema") != "ilarms/v1" or il.get("window") != "cut32768":  # the LOADER names the 32k files; the runner maps them to uncut
            pr.append("ilarms: not an ilarms/v1 block (window cut32768, mapped to uncut files through the overrides)")
    return pr


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--pins", required=True)
    b.add_argument("--status", default="")
    b.add_argument("--test", action="store_true")
    v = sp.add_parser("verify")
    v.add_argument("file")
    v.add_argument("--deep", action="store_true")
    for p_ in (b, v):
        p_.add_argument("--copies-root", default="")
    a = ap.parse_args()
    if a.copies_root:  # F2 (0d, 10-17): set before anything reads a copy; the only root source inside the records wrapper
        CP.set_root(a.copies_root)
    if a.cmd == "build":
        if a.test == bool(a.status):
            raise SystemExit("give exactly one of --test or --status 'FROZEN <PT>'")
        build(a.pins, a.status or f"TEST {datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M %Z')}", a.test)
    else:
        pr = verify(json.loads((HERE / a.file).read_text()), a.deep)
        print(json.dumps({"file": a.file, "problems": pr, "pass": not pr}, indent=1))
        if pr:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
