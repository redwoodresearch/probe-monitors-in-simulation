"""appA: independent re-derivation of Appendix A's "about 17%" / "about 21%" (route C reruns, 2026-10-20).

New code by the fresh appA implementer, written from the spec text quoted in PRESTATED (PRESTATED v2 @ 9b839645d L43-L53 and
L102-L103; RULE @ c3a0cd830 L1591). Gap choices G1-G8 are PRESTATED section 4, all accepted by 09, with 09's rulings (relayed
by 2e, 10-20 ~12:5x): G2 and G3 sensitivities (a printed row is UNRESOLVED if excluding missing-kind units or conflicting runs
changes its rounded pct); G5 the verdict covers H_B (row 250) and run1 (row 251) only; G6 per-set cohort governs; G8 a
mismatch on runs / runs_with_units alone is a definition difference. It never prints a value: only MATCH / MISMATCH / ABSENT
per key, verdicts, booleans, folder and file counts, shas and paths. Values stay in the container and on the output Volume.

Modes
  uv run --no-project --with numpy==2.1.3 --with modal==1.6.1 python run_appa.py toy
      synthetic positive control; exit 0 iff every known-by-construction answer holds.
  uv run --no-project --with numpy==2.1.3 --with modal==1.6.1 python run_appa.py toy-negative
      predicted to FAIL: exit 1 iff the planted wrong record and the perturbed toy mismatch exactly the expected keys
      (else exit 2).
  "$M" run --detach run_appa.py --mode real --repo <worktree> --stamp <yyyy-mm-dd_HHMMSS PT>
      Modal: both toy gates in the container, then the real run. The container commits the results JSON, prints APPA_KEY /
      APPA_VERDICT / APPA_RESULT, then runs the .eval cross-check and prints APPA_XCHECK. Local exit 0: both printed rows
      MATCH; 3: a printed row MISMATCH or UNRESOLVED; other: failed (APPA_FAILED stage=<s> type=<t>; traceback on the Volume).
"""

from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import zipfile
import re
import traceback
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import modal

APP_NAME = "probe-shift_aswin-surya_sub-rerun_appa_kill-2026-11-19"
assert len(APP_NAME) <= 63

SCORES, STAGED, CATDIR, OUT = "/scores", "/staged", "/catalog", "/out"
IN_VOLUMES = {
    SCORES: "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14",
    STAGED: "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14",
    CATDIR: "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13",
    "/ev_chain": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14",
    "/ev_q4own": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13",
    "/ev_pilots": "probe-shift_aswin-surya_sub-q3pilots_runs_kill-2026-11-13",
    "/ev_q2fresh": "probe-shift_aswin-surya_sub-q2-fresh_work_kill-2026-11-13",
}
EVAL_ROOTS = ("/ev_chain/evals_v1", "/ev_q4own/evals_v1", "/ev_pilots", "/ev_q2fresh")
CATALOGUE_REL = ("xfer/fpr507prep/records/results/"
                 "alltasks_CRD_FULL_r2_allorigins_pooledworld_catalogue_e12_2026-10-18_112818_catalogue.json")
CATALOGUE_SHA = "72432318a3703d4675a5c4530d77f598a003af351340d895d305b9bd1050adc8"

_P = "experiments/2026-10-15_fpr_sweep_507_prep/"
PINS = {  # name: (commit, path, sha256)
    "INPUTS_FULL": ("1a842a856", _P + "config/INPUTS_FULL_2026-10-17_233212.json",
                    "1e1868586fe8a786f6e1094b9d5b62e362ec0e2ce241e9b0103d6584af351126"),
    "INPUTS_FINAL_V2": ("a6bde80a2", _P + "config/INPUTS_FINAL_V2_FROZEN_2026-10-16_044120.json",
                        "0e065958b206dbceba8afeb3332ec68b898481da520fe3a3654ab2b6913eaadd"),
    "E12_BLOCK": ("22d7089f5", _P + "config/E12_BLOCK_2026-10-16_092322.json",
                  "683d4de1040f489a768a44fe5ad53f787d35d8ca22117b897a42a75880e1f4dc"),
    "exclusions": ("6414c492b", _P + "results/exclusions_2026-10-16_102159.json",
                   "67a608c25a84e77cb6beb2c70d8c91177e2274d2c96e166469060462d4c9223d"),
    "ov_S_new": ("38b42bf2d", _P + "config/cutfull/override_S_new_C_2026-10-16_104513_gated.json",
                 "908a27ad834a280fb56e9cbe7747bb90a3a3aa726098afe345a9fdc4c91bbadb"),
    "ov_S_old": ("5df9a43ed", _P + "config/cutfull/override_S_old_C_2026-10-17_151056_gated.json",
                 "f8ee4ddc603dd9e38f0fd650055bcc9d324df903bbfd5a51809ca6b59ffceafd"),
    "ov_S_newmains": ("64e9ab71a", _P + "config/cutfull/override_S_newmains_C_2026-10-16_053303_gated.json",
                      "09b766a09208cce60b0882bce7fd538582b9d06d27b30f8ae5a9dda6759c3c3f"),
    "ov_S_paper493": ("4d2f45e73", _P + "config/cutfull/override_S_paper493_C_2026-10-16_101944_gated.json",
                      "1d3208ddc36ce08baf0349bef3d16a3fdf79cd5f8756df80bbc70ca1191b02d0"),
    "ov_pools_ep2": ("93ee9afe6", _P + "config/cutfull/override_pools_ep2_C_2026-10-16_053038_gated.json",
                     "734d2ccb0b4207a7a897b2b6b91776aa35cb8e29e029f2976b4c1f7147968e83"),
    "ov_S_new25f": ("f5a2504dc", _P + "config/cutfull/override_S_new25f_C_2026-10-16_045304_gated.json",
                    "6ebc9d007c99ad5f48bbef7ef60ab31002d5bc30d461f4d70ac0386986b43326"),
    "record": ("a6d37c024", "experiments/2026-10-20_routec_fileedit_share/results/cunits_by_tool_2026-10-20_044328.json",
               "072efe87b87c3dc1d5b67de4e599bba516585924775f0d39edeeb132ba51daa6"),
}
OVERRIDES = ("ov_S_new", "ov_S_old", "ov_S_newmains", "ov_S_paper493", "ov_pools_ep2", "ov_S_new25f")  # G1 order
FULL_BINDING = {  # INPUTS_FULL key path that must pin each config's sha256
    "INPUTS_FINAL_V2": ("base",), "E12_BLOCK": ("e12",), "exclusions": ("exclusions",),
    "ov_S_new": ("overrides", "S_new|C"), "ov_S_old": ("overrides", "S_old|C"),
    "ov_S_newmains": ("overrides", "S_newmains|C"), "ov_S_paper493": ("overrides", "S_paper493|C"),
    "ov_pools_ep2": ("overrides", "pools_ep2|C"), "ov_S_new25f": ("overrides", "S_new+25f|C"),
}
LEAD3 = ("p2/fibs__command.npz", "sola_uncut_reselect/sol_a_uncut__cmd_A.npz",
         "own_all_rs_uncut/own_all_rs_uncut_L48__C.npz")  # PRESTATED v2 L43, L52
ROLE_GROUP = {"A": "H_A", "B": "H_B", "S": "H_sim"}  # L48
ATTACK = ("run1", "run2", "sim")  # L48-L49
GROUPS = ("H_B", "run1", "H_sim", "run2", "sim", "H_A")  # L50-L51: five claim groups + the H_A diagnostic
PRINTED = {"H_B": 17, "run1": 21}  # provenance v17 rows 250 and 251
N_COHORT = 1313
SEED, N_SAMPLE = 20261020, 200
EPOCH_RE = re.compile(r"#e\d+$")
VARIANTS = ("main", "union", "first_eligible")
ALT_KEYS = ("units", "text_editor", "pct_eq_record", "pct_mine_eq_printed")
TIMEOUT_S = 600  # Codex r1 #10: worst case 600 s x ~$0.46/h ~ $0.077, within 120% of the +30% estimate ($0.078)


# ---------------------------------------------------------------- definitions

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(te, units):
    """PRESTATED v2 L102-L103: half up to a whole percent, integer arithmetic."""
    return None if not units else (200 * te + units) // (2 * units)


def half_tie(te, units) -> bool:
    return bool(units) and (200 * te) % (2 * units) == units


def enumerate_folders(overrides, scores_root):
    """G1 + G4. overrides: [(label, override JSON)] in the G1 order. A folder counts only if all three lead readers are listed,
    their files exist, and their staged_shas lists share exactly one sha (L52-L53)."""
    listed, counted, skipped = 0, [], Counter()
    for order, (label, ov) in enumerate(overrides):
        cands = [("arms", root, [ov["arms"][root].get(r) for r in LEAD3]) for root in sorted(ov.get("arms") or {})]
        simh, by = ov.get("simh") or {}, []
        for r in LEAD3:
            m = {}
            for e in simh.get(r) or []:
                if e["folder"] in m:
                    raise ValueError(f"{label}: simh folder listed twice for {r}")
                m[e["folder"]] = e
            by.append(m)
        cands += [("simh", fo, [b.get(fo) for b in by]) for fo in sorted(by[0])]
        for kind, name, ents in cands:
            listed += 1
            if any(e is None for e in ents):
                skipped["reader_not_listed"] += 1
                continue
            if not all(os.path.isfile(os.path.join(scores_root, e["path"])) for e in ents):
                skipped["reader_file_absent"] += 1
                continue
            common = set(ents[0]["staged_shas"]) & set(ents[1]["staged_shas"]) & set(ents[2]["staged_shas"])
            if len(common) != 1:
                skipped[f"common_staged_shas_{min(len(common), 2)}"] += 1
                continue
            counted.append({"order": order, "override": label, "set": ov["set"], "kind": kind, "name": name,
                            "p2_path": ents[0]["path"], "p2_sha256": ents[0]["sha256"],
                            "staged_sha256": next(iter(common))})
    return listed, counted, skipped


def staged_candidates(staged_root: str, p2_path: str) -> list:
    """G2: .../p2/<staged_set>__<arm>/fibs__command.npz -> <staged_set>[_try<n>]/inputs/<arm>.jsonl or <arm>@<x>.jsonl.
    Only a name guess: the sha256 decides (L53), and a hash index of every */inputs/*.jsonl is the fallback."""
    staged_set, arm = p2_path.split("/")[-2].rsplit("__", 1)
    dirs = [os.path.join(staged_root, staged_set)] + glob.glob(os.path.join(staged_root, glob.escape(staged_set) + "_try*"))
    return [p for d in dirs for p in glob.glob(os.path.join(glob.escape(d), "inputs", "*.jsonl"))
            if os.path.basename(p) == arm + ".jsonl" or os.path.basename(p).startswith(arm + "@")]


def parse_staged(path: str) -> dict:
    """G2. One JSONL line per epoch: key, calls[{kind, readout, name, ...}]. Keeps, per epoch key, readout -> [(tool name,
    kind field missing)] of kind-command calls with a readout, and the name multiset of every kind-command call (for the
    cross-check). A call without `kind` counts as command (G2, SPEC-SILENT; the stager's default, run_replay_readouts.py
    @ 5cbd3354c L479) and is flagged, so its units can be excluded in 09's G2 sensitivity."""
    raw = open(path, "rb").read()
    cmd, allnames, n_missing_kind, n_missing_kind_readout, dup_keys = {}, {}, 0, 0, 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        ep = json.loads(line)
        k = ep["key"]
        if k in cmd:
            dup_keys += 1  # merged: a readout then matching calls of both lines counts as 2+ calls (unmapped)
        m, names = cmd.setdefault(k, {}), Counter(allnames.get(k, {}))
        for c in ep["calls"]:
            missing = "kind" not in c  # exactly the stager's c.get("kind", "command") default
            n_missing_kind += missing
            if not missing and c["kind"] != "command":
                continue
            name = c["name"]
            if not isinstance(name, str):
                raise ValueError(f"{path}: a command call's name is not a string")
            names[name] += 1
            if c.get("readout") is not None:
                m.setdefault(int(c["readout"]), []).append((name, missing))
                n_missing_kind_readout += missing  # reviewer 4: predicted 0
        allnames[k] = dict(names)
    return {"path": path, "sha256": sha256_bytes(raw), "cmd": cmd, "allnames": allnames,
            "n_missing_kind": n_missing_kind, "n_missing_kind_readout": n_missing_kind_readout, "dup_keys": dup_keys}


def resolve_staged(folders, staged_root, workers):
    """Hash first, parse only the files whose sha256 a counted folder chose. Returns parsed files by sha, the number of shas
    found by the name guess, by the full hash index, and shas found at more than one path (the sorted first path is used)."""
    need = {f["staged_sha256"] for f in folders}
    first = sorted({p for f in folders for p in staged_candidates(staged_root, f["p2_path"])})
    found = {}
    with ProcessPoolExecutor(workers) as ex:
        for p, h in zip(first, ex.map(sha256_file, first)):
            if h in need:
                found.setdefault(h, []).append(p)
        by_name = len(found)
        if need - set(found):  # fallback: hash every staged input file on the Volume
            rest = sorted(set(glob.glob(os.path.join(staged_root, "*", "inputs", "*.jsonl"))) - set(first))
            for p, h in zip(rest, ex.map(sha256_file, rest)):
                if h in need:
                    found.setdefault(h, []).append(p)
        if need - set(found):
            raise ValueError(f"{len(need - set(found))} staged shas not found on the staged Volume")
        parsed = {}
        for res in ex.map(parse_staged, sorted(min(v) for v in found.values())):
            if res["sha256"] not in need or res["sha256"] in parsed:
                raise ValueError("a staged file changed between hashing and parsing")
            parsed[res["sha256"]] = res
    return parsed, by_name, len(found) - by_name, sum(len(v) > 1 for v in found.values())


def tally(canons, drop_missing_kind: bool = False) -> dict:
    """Units by tool over the canonical appearances of a group's runs (L43-L47). A unit is ("name", tool, kind_missing)."""
    tools, unm, rwu, rwte = Counter(), Counter(), 0, 0
    for canon in canons:
        run_tools = Counter(o[1] for _, o in canon["sig"] if o[0] == "name" and not (drop_missing_kind and o[2]))
        unm.update(o[1] for _, o in canon["sig"] if o[0] == "unmapped")
        tools.update(run_tools)
        rwu += bool(run_tools)
        rwte += bool(run_tools["text_editor"])
    units, te = sum(tools.values()), tools["text_editor"]
    other = {n: c for n, c in sorted(tools.items()) if n not in ("bash", "text_editor") and c}
    return {"units": units, "bash": tools["bash"], "text_editor": te, "other": other, "other_sum": sum(other.values()),
            "unmapped_units": sum(unm.values()), "unmapped_by_reason": dict(sorted(unm.items())), "runs_with_units": rwu,
            "runs_with_text_editor": rwte, "pct": pct(te, units), "half_tie": half_tie(te, units)}


def compute(overrides, cohort, catalogue_runs, scores_root, staged_root, workers):
    import numpy as np

    listed, folders, skipped = enumerate_folders(overrides, scores_root)
    parsed, by_name, by_index, several_paths = resolve_staged(folders, staged_root, workers)
    apps, several = {}, 0
    for fi, f in enumerate(folders):
        raw = open(os.path.join(scores_root, f["p2_path"]), "rb").read()
        if sha256_bytes(raw) != f["p2_sha256"]:
            raise ValueError(f"p2 file sha differs from its override: {f['override']}|{f['name']}")
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        if any(c not in z.files for c in ("key", "run_key", "token")):
            raise ValueError(f"p2 file lacks key/run_key/token: {f['override']}|{f['name']}")
        # Codex r1 #1: in these files `key` is the run key <world token>|<task> and `run_key` is the epoch key
        # <launch>/<eval_rel>#e<k> (run_peraction311.py @ 1a842a856 L4179-L4189); the staged line's `key` is the epoch key.
        rks, keys = z["key"].astype(str).tolist(), z["run_key"].astype(str).tolist()
        toks = z["token"].astype(np.int64).tolist()
        if not len(keys) == len(rks) == len(toks):
            raise ValueError(f"p2 columns differ in length: {f['override']}|{f['name']}")
        if not all("|" in r and not EPOCH_RE.search(r) for r in rks) or not all(EPOCH_RE.search(k) for k in keys):
            raise ValueError(f"p2 key/run_key not in the expected forms: {f['override']}|{f['name']}")
        st, k2rk, per_run = parsed[f["staged_sha256"]]["cmd"], {}, {}
        for k, rk, t in zip(keys, rks, toks):  # L43-L45: map to exactly one kind-command call with readout == token
            k2rk.setdefault(rk, set()).add(k)  # one epoch per run key expected in a folder
            ep = st.get(k)
            if ep is None:
                out = ("unmapped", "no_epoch")
            else:
                calls = ep.get(t, [])  # [(tool name, kind missing)]
                out = (("name",) + calls[0]) if len(calls) == 1 else ("unmapped", "multi_call" if calls else "no_call")
            per_run.setdefault(rk, []).append((k, t, out))
        several += sum(len(v) > 1 for v in k2rk.values())
        for rk, rows in per_run.items():
            apps.setdefault(rk, []).append({"fi": fi, "rows": rows, "sig": sorted((t, o) for _, t, o in rows),
                                            # G3 signature (PRESTATED §4): token + tool name or unmapped reason, without
                                            # the missing-kind flag (Codex r3 #12); the flag stays in "sig" for G2
                                            "csig": sorted((t, o[:2]) for _, t, o in rows),
                                            "keys": sorted({k for k, _, _ in rows})})

    cohort_union = set().union(*cohort.values())
    # "main" = the accepted G3/G6 choices. Two stored alternatives (reviewer 2): "union" = attack membership by the union
    # cohort; "first_eligible" = per-set membership, units from the first appearance in a folder whose set's cohort has
    # the task. Honest runs are the same in all three.
    members = {v: {g: {} for g in GROUPS} for v in VARIANTS}
    outside, outside_rows, rule_diff, keys_differ = Counter(), Counter(), Counter(), Counter()
    conflict_runs = {g: set() for g in GROUPS}
    for rk in sorted(apps):
        a = apps[rk]  # appearances in G1 order
        world, task = rk.split("|", 1)
        honest, cat = set(), catalogue_runs.get(rk)
        if cat is not None:
            honest = {ROLE_GROUP[r] for r in cat["roles"]}
        elig, per_set, union = a, False, False
        if world in ATTACK:
            if cat is not None:
                raise ValueError("an attack-world run key is in the honest catalogue")
            elig = [x for x in a if task in cohort.get(folders[x["fi"]]["set"], ())]
            per_set, union = bool(elig), task in cohort_union
            if per_set != union:
                rule_diff[world] += 1
        gs = {"main": honest | ({world} if per_set else set()), "union": honest | ({world} if union else set()),
              "first_eligible": honest | ({world} if per_set else set())}
        canon = {"main": a[0], "union": a[0], "first_eligible": elig[0] if elig else a[0]}  # G3: first appearance
        for v in VARIANTS:
            for g in gs[v]:
                members[v][g][rk] = canon[v]
        if not gs["main"]:
            outside[world] += 1
            outside_rows[world] += sum(len(x["rows"]) for x in a)
            continue
        conflict = any(x["csig"] != a[0]["csig"] for x in a[1:])
        kdiff = any(x["keys"] != a[0]["keys"] for x in a[1:])
        for g in gs["main"]:
            if conflict:
                conflict_runs[g].add(rk)
            keys_differ[g] += kdiff

    alt = {v: {g: tally(members[v][g].values()) | {"runs": len(members[v][g])} for g in GROUPS} for v in VARIANTS[1:]}
    members = members["main"]
    totals = {}
    for g in GROUPS:
        main = tally(members[g].values())
        no_mk = tally(members[g].values(), drop_missing_kind=True)  # 09's G2 sensitivity
        no_cf = tally(c for rk, c in members[g].items() if rk not in conflict_runs[g])  # 09's G3 sensitivity
        totals[g] = main | {"runs": len(members[g]), "run_keys": sorted(members[g]),
                            "missing_kind_units": main["units"] - no_mk["units"], "pct_excl_missing_kind": no_mk["pct"],
                            "conflict_runs": len(conflict_runs[g]), "conflict_run_keys": sorted(conflict_runs[g]),
                            "pct_excl_conflict_runs": no_cf["pct"]}
    gates = {"folders_listed": listed, "folders_counted": len(folders), "folders_skipped": dict(sorted(skipped.items())),
             "staged_files": len(parsed), "staged_resolved_by_name": by_name, "staged_resolved_by_index": by_index,
             "staged_sha_on_several_paths": several_paths, "run_keys_with_several_epochs": several,
             "dup_epoch_keys": sum(p["dup_keys"] for p in parsed.values()),
             "any_call_missing_kind": any(p["n_missing_kind"] for p in parsed.values()),
             "calls_missing_kind_with_readout": sum(p["n_missing_kind_readout"] for p in parsed.values()),
             # Codex r3 #11: per-group counts are scored-unit counts, so only booleans are printed; counts are in detail
             "missing_kind_units_any": {g: bool(totals[g]["missing_kind_units"]) for g in GROUPS},
             "outside_world_tokens": sorted(outside), "cohort_rules_same": not rule_diff,
             "dedup_conflict_runs": {g: len(conflict_runs[g]) for g in GROUPS},
             "dedup_key_strings_differ_runs": {g: keys_differ[g] for g in GROUPS}}
    detail = {"outside_runs_by_world": dict(sorted(outside.items())), "outside_rows_by_world": dict(sorted(outside_rows.items())),
              "cohort_rule_diff_runs": dict(sorted(rule_diff.items())),
              "calls_missing_kind": sum(p["n_missing_kind"] for p in parsed.values()),
              "missing_kind_units": {g: totals[g]["missing_kind_units"] for g in GROUPS},
              "catalogue_runs_not_seen": {g: sorted(rk for rk, c in catalogue_runs.items()
                                                    if g in {ROLE_GROUP[r] for r in c["roles"]} and rk not in members[g])
                                          for g in ("H_A", "H_B", "H_sim")},
              "folders": folders, "staged": {s: p["path"] for s, p in sorted(parsed.items())}}
    return {"totals": totals, "alt": alt, "gates": gates, "detail": detail, "members": members, "folders": folders,
            "parsed": parsed}


def compare(totals, record, printed):
    """Section 5: exact equality per key; pct by the L102-L103 rule on both sides."""
    rows = []

    def put(g, key, ok):
        rows.append((g, key, "MATCH" if ok else "MISMATCH"))

    for g in GROUPS:
        cell = record["cells"].get(f"pooled|{g}")
        if cell is None:
            rows.append((g, "cell", "ABSENT"))
            continue
        p2, mine = cell["readers"]["p2"], totals[g]
        for key in ("units", "bash", "text_editor", "unmapped_units", "runs_with_units"):
            if key in p2:
                put(g, key, mine[key] == p2[key])
            else:
                rows.append((g, key, "ABSENT"))
        if isinstance(p2.get("other"), dict):
            put(g, "other_by_name", mine["other"] == {n: c for n, c in sorted(p2["other"].items()) if c})
            put(g, "other_sum", mine["other_sum"] == sum(p2["other"].values()))
        elif "other" in p2:
            put(g, "other_sum", mine["other_sum"] == p2["other"])
        else:
            rows.append((g, "other", "ABSENT"))
        if "runs" not in cell:
            rows.append((g, "runs", "ABSENT"))
        elif isinstance(cell["runs"], (list, dict)):
            put(g, "runs", set(mine["run_keys"]) == set(cell["runs"]))  # G8: set equality
        else:
            put(g, "runs", mine["runs"] == cell["runs"])
        if g in printed:
            put(g, "pct_mine_eq_printed", mine["pct"] == printed[g])
        if "units" in p2 and "text_editor" in p2:
            rec_pct = pct(p2["text_editor"], p2["units"])
            put(g, "pct_eq_record", mine["pct"] is not None and mine["pct"] == rec_pct)
            if g in printed:
                put(g, "pct_record_eq_printed", rec_pct == printed[g])
        else:  # Codex r1 #7: a record-derived pct without its operands is ABSENT, not MISMATCH
            rows += [(g, k, "ABSENT") for k in ("pct_eq_record", "pct_record_eq_printed")[: 2 if g in printed else 1]]
    return rows


def compare_alt(alt, record, printed):
    """Reviewer 2: the stored alternatives (union cohort; first eligible appearance), printed rows only, as secondary keys
    `<key>@<variant>`. They never enter the verdict."""
    rows = []
    for v, tot in alt.items():
        for g in printed:
            p2, t = ((record["cells"].get(f"pooled|{g}") or {}).get("readers") or {}).get("p2") or {}, tot[g]
            rows += [(g, f"{k}@{v}", "ABSENT" if k not in p2 else "MATCH" if t[k] == p2[k] else "MISMATCH")
                     for k in ("units", "text_editor")]
            rows.append((g, f"pct_mine_eq_printed@{v}", "MATCH" if t["pct"] == printed[g] else "MISMATCH"))
            if "units" in p2 and "text_editor" in p2:
                ok = t["pct"] is not None and t["pct"] == pct(p2["text_editor"], p2["units"])
                rows.append((g, f"pct_eq_record@{v}", "MATCH" if ok else "MISMATCH"))
            else:
                rows.append((g, f"pct_eq_record@{v}", "ABSENT"))
    return rows


def compare_all(res, record, printed):
    return compare(res["totals"], record, printed) + compare_alt(res["alt"], record, printed)


PRIMARY_KEYS = ("units", "text_editor", "pct_mine_eq_printed", "pct_record_eq_printed")


DEFINITION_KEYS = ("runs", "runs_with_units")  # 09 on G8: a mismatch on these alone is a definition difference


def primary_match(rows, g) -> bool:
    """All four primary keys present for g and MATCH (an absent cell or key is not a match)."""
    return {k for gg, k, s in rows if gg == g and s == "MATCH"} >= set(PRIMARY_KEYS)


def key_class(g, k) -> str:
    """09 on G5: the verdict covers the printed rows (H_B row 250, run1 row 251) only; other groups are diagnostics."""
    if g not in PRINTED:
        return "diagnostic"
    return "primary" if k in PRIMARY_KEYS else "definition" if k in DEFINITION_KEYS else "secondary"


def verdict(rows, t, g) -> dict:
    """09's ruling for a printed row: UNRESOLVED if excluding missing-kind units (G2) or conflicting runs (G3) changes the
    rounded pct; otherwise MATCH iff all four primary keys MATCH; otherwise MISMATCH."""
    mk_same = None if not t["missing_kind_units"] else t["pct_excl_missing_kind"] == t["pct"]
    cf_same = None if not t["conflict_runs"] else t["pct_excl_conflict_runs"] == t["pct"]
    v = "UNRESOLVED" if (mk_same is False or cf_same is False) else "MATCH" if primary_match(rows, g) else "MISMATCH"
    return {"verdict": v, "missing_kind_units_any": bool(t["missing_kind_units"]), "missing_kind_pct_same": mk_same,
            "conflict_runs_any": bool(t["conflict_runs"]), "conflict_pct_same": cf_same}


def eval_tool_names(path: str, epoch: int):
    """Section 8: ONLY assistant tool_calls[].function names; no message text is kept. Returns (names, sample member)."""
    with zipfile.ZipFile(path) as z:
        samples = sorted(n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json"))
        pick = samples if len(samples) == 1 else [n for n in samples if n.endswith(f"_epoch_{epoch + 1}.json")]
        if len(pick) != 1:
            return None, None
        msgs = json.loads(z.read(pick[0])).get("messages") or []
    names = Counter(tc.get("function") for m in msgs if m.get("role") == "assistant" for tc in (m.get("tool_calls") or []))
    return names, pick[0]


def xcheck_one(item) -> dict:
    """One sampled run: per epoch, the .eval found (path, sha256, sample member) and its class (Codex r1 #9 provenance)."""
    roots, rk, epochs = item
    out, recs = Counter(), []
    for key, staged_names, unit_names in epochs:
        rel, _, k = key.rpartition("#e")
        hits = [os.path.join(r, rel) for r in roots if os.path.isfile(os.path.join(r, rel))]
        rec = {"run_key": rk, "epoch_key": key, "eval_paths": hits}
        recs.append(rec)
        if not hits:
            out["eval_not_found"] += 1
            rec["class"] = "eval_not_found"
            continue
        out["eval_on_several_roots"] += len(hits) > 1
        rec["eval_sha256"] = sha256_file(hits[0])
        try:  # reviewer 1: an unreadable .eval is a named class, by a narrow except
            ev, member = eval_tool_names(hits[0], int(k))
        except (zipfile.BadZipFile, json.JSONDecodeError, KeyError):
            out["eval_unreadable"] += 1
            rec["class"] = "eval_unreadable"
            continue
        rec["member"] = member
        if ev is None:
            out["ambiguous_sample"] += 1
            rec["class"] = "ambiguous_sample"
            continue
        st, un = Counter(staged_names), Counter(unit_names)
        out["epochs_resolved"] += 1
        rec["class"] = ("staged_equal" if st == ev else "staged_sub_multiset" if all(ev[n] >= c for n, c in st.items())
                        else "staged_other")
        rec["units_sub_multiset"] = all(ev[n] >= c for n, c in un.items())
        out[rec["class"]] += 1
        out["units_sub_multiset" if rec["units_sub_multiset"] else "units_not_sub_multiset"] += 1
    resolved = [r for r in recs if "units_sub_multiset" in r]
    out["runs_resolved"] += bool(resolved)
    out["runs_resolved_all_units_sub_multiset"] += bool(resolved) and all(r["units_sub_multiset"] for r in resolved)
    return {"counts": dict(out), "records": recs}


def xcheck(res, roots, workers, n_sample=N_SAMPLE, seed=SEED):
    canon = {}
    for g in GROUPS:
        for rk, c in res["members"][g].items():
            if any(o[0] == "name" for _, _, o in c["rows"]):
                canon[rk] = c
    runs = sorted(canon)
    sample = random.Random(seed).sample(runs, min(n_sample, len(runs)))
    items = []
    for rk in sample:
        c = canon[rk]
        allnames = res["parsed"][res["folders"][c["fi"]]["staged_sha256"]]["allnames"]
        items.append((roots, rk, [(k, allnames.get(k, {}), [o[1] for kk, _, o in c["rows"] if kk == k and o[0] == "name"])
                                  for k in c["keys"]]))
    total, records = Counter(), []
    with ProcessPoolExecutor(workers) as ex:
        for out in ex.map(xcheck_one, items):
            total.update(out["counts"])
            records += out["records"]
    return {"seed": seed, "n_runs_eligible": len(runs), "n_runs_sampled": len(sample), "sampled_run_keys": sample,
            "counts": dict(sorted(total.items())), "records": records}


# ---------------------------------------------------------------- toy positive control (synthetic, known by construction)

TOY_PRINTED = {"H_B": 13, "run1": 33}
TOY_PLANTED = ({"H_B.text_editor", "H_B.pct_eq_record", "H_B.pct_record_eq_printed",
                "run1.units", "run1.pct_eq_record", "run1.pct_record_eq_printed"}
               | {f"{g}.{k}@{v}" for v in ("union", "first_eligible")
                  for g, k in (("H_B", "text_editor"), ("H_B", "pct_eq_record"), ("run1", "units"), ("run1", "pct_eq_record"))})
TOY_PERTURBED = ({"H_B.bash", "H_B.text_editor", "H_B.pct_eq_record", "H_B.pct_mine_eq_printed"}  # reviewer 5
                 | {f"H_B.{k}@{v}" for v in ("union", "first_eligible")
                    for k in ("text_editor", "pct_eq_record", "pct_mine_eq_printed")})


def _toy_record(negative: bool) -> dict:
    def cell(runs, units, bash, te, other, unmapped, rwu):
        return {"runs": runs, "readers": {"p2": {"units": units, "bash": bash, "text_editor": te, "other": other,
                                                 "unmapped_units": unmapped, "runs_with_units": rwu}}}
    return {"cells": {
        "pooled|H_B": cell(2, 8, 6, 2 if negative else 1, {"db_bash": 1}, 0, 2),
        "pooled|run1": cell(1, 4 if negative else 3, 1, 1, {"client_bash": 1}, 3, 1),
        "pooled|H_sim": cell(2, 3, 1, 2, {}, 0, 2),
        "pooled|H_A": cell(["honest_c|tc"], 4, 3, 0, {"db_bash": 1}, 0, 1)}}


def _build_toy(base: str, perturb: bool = False, wrong_staged: bool = False, swap_p2: bool = False):
    """Rows below are (epoch key, run key, token). The p2 files store them as the real files do: key = run key
    (<world>|<task>), run_key = epoch key (Codex r1 #1); swap_p2 writes them the other way round, which must stop the run."""
    import numpy as np

    sc, sg, ev = (os.path.join(base, d) for d in ("scores", "staged", "ev"))

    def staged(rel, epochs):
        p = os.path.join(sg, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            for k, calls in epochs:
                f.write(json.dumps({"key": k, "arm": "x", "run": k.split("#")[0], "task": "t", "epoch": 0, "ids": [1, 2],
                                    "calls": calls}) + "\n")
        return sha256_file(p)

    def p2(rel, rows):
        p = os.path.join(sc, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        run_keys, epochs = np.array([r[1] for r in rows]), np.array([r[0] for r in rows])
        np.savez(p, key=epochs if swap_p2 else run_keys, run_key=run_keys if swap_p2 else epochs,
                 token=np.array([r[2] for r in rows], np.int64), score=np.zeros(len(rows)))
        return sha256_file(p)

    def reader(rel):
        p = os.path.join(sc, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(b"r")

    def cmd(r, n, kind="command"):
        return {"kind": kind, "readout": r, "name": n, "irid": "i", "turn": 1}

    KA, KQ, KX, KY1 = "L1/attack/t_a.eval#e0", "L1/attack/t_q.eval#e0", "L1/attack/t_x.eval#e0", "L1/attack/t_y.eval#e0"
    s_att = staged("ta_main_cutfull_v1/inputs/attack.jsonl", [
        (KA, [cmd(10, "bash"), cmd(20, "text_editor"), cmd(20, "text_editor", "call_end"), {"readout": 30, "name": "client_bash"},
              cmd(None, "text_editor"), cmd(40, "bash"), cmd(40, "bash"), cmd(50, "bash", "output_end")]),
        (KQ, [cmd(5, "text_editor")]), (KX, [cmd(3, "bash")]), (KY1, [cmd(7, "bash")])])
    KY2 = "L5/attack/t_y.eval#e0"  # Codex r3 #13: run2|ty first appears in S_new (ineligible), then in S_old (eligible)
    s_tf = staged("tf_main_cutfull_v1/inputs/attack.jsonl", [(KY2, [cmd(7, "text_editor")])])
    KB, KC, KZ, KG = ("L2/honest_b/t_b.eval#e0", "L2/honest_b/t_c.eval#e0", "L2/honest_b/t_z.eval#e0",
                      "L2/honest_b/t_g.eval#e0")
    s_hb = staged("tb_main_cutfull_v1/inputs/honest_b.jsonl", [
        (KB, [cmd(1, "text_editor" if perturb else "bash"), cmd(2, "bash"), cmd(3, "bash"), cmd(4, "text_editor")]),
        (KC, [cmd(1, "bash"), cmd(2, "bash"), cmd(3, "db_bash"), cmd(4, "bash")]),
        (KZ, [cmd(1, "text_editor")]), (KG, [cmd(1, "bash")])])
    KB2, KG2 = "L9/honest_b/t_b.eval#e0", "L3/honest_b/t_g.eval#e0"
    s_hb2 = staged("te_other_v1/inputs/honest_b.jsonl", [  # not at its name-derived path: found by the hash index
        (KB2, [cmd(1, "bash"), {"readout": 2, "name": "bash"}, cmd(3, "bash"), cmd(4, "text_editor")]),  # Codex r3 #12:
        (KG2, [cmd(1, "text_editor")])])  # KB2 differs from KB only by a missing kind, which is not a G3 conflict
    KS = "L4/sim_honest/t_s.eval#e0"
    s_sh = staged("ts_sim_cutfull_v1_try2/inputs/sim_honest@september.jsonl", [
        (KS, [cmd(1, "text_editor"), {"readout": 2, "name": "text_editor"}])])  # second call has no kind
    staged("ts_sim_cutfull_v1/inputs/sim_honest.jsonl", [(KS, [cmd(1, "bash"), cmd(2, "bash")])])  # decoy: other sha

    def entry(path, sha, staged_shas):
        return {"path": path, "sha256": sha, "staged_shas": staged_shas, "runs_without_rows": []}

    def arms(prefix, folder, rows, lists):
        pa = f"{prefix}/p2/{folder}/fibs__command.npz"
        sha = p2(pa, rows)
        out = {}
        for r, lst in zip(LEAD3, lists):
            if lst is None:
                continue
            path = pa if r == LEAD3[0] else f"{prefix}/{r.split('/')[0]}/{folder}/{r.split('/')[1]}"
            if r != LEAD3[0]:
                reader(path)
            out[r] = entry(path, sha if r == LEAD3[0] else "0" * 64, lst)
        return out

    rows_a = [(KA, "run1|ta", 10), (KA, "run1|ta", 20), (KA, "run1|ta", 30), (KA, "run1|ta", 40), (KA, "run1|ta", 99),
              ("L1/attack/t_missing.eval#e0", "run1|ta", 7), (KQ, "run1|tq", 5), (KX, "run2|tx", 3), (KY1, "run2|ty", 7)]
    rows_b = ([(KB, "honest_b|tb", t) for t in (1, 2, 3, 4)] + [(KC, "honest_c|tc", t) for t in (1, 2, 3, 4)]
              + [(KZ, "honest_b|tz", 1), (KG, "honest_b|tg", 1)])
    rows_e = [(KB2, "honest_b|tb", t) for t in (1, 2, 3, 4)] + [(KG2, "honest_b|tg", 1)]
    bad = "0" * 64
    s_te = bad if wrong_staged else s_hb2
    ov1 = {"set": "S_new", "arms": {
        "ta_main_v1__attack": arms("cutfull/q1", "ta_main_cutfull_v1__attack", rows_a,
                                   [[s_att, "f" * 64], [s_att], ["e" * 64, s_att]]),
        "tb_main_v1__honest_b": arms("cutfull/q1", "tb_main_cutfull_v1__honest_b", rows_b, [[s_hb]] * 3),
        "tc_main_v1__honest_b": arms("cutfull/q1", "tc_main_cutfull_v1__honest_b", rows_b, [[s_hb], [s_hb], None]),
        "td_main_v1__honest_b": arms("cutfull/q1", "td_main_cutfull_v1__honest_b", rows_b, [[s_hb, s_att]] * 3)}}
    ps = "cutfull/simh/p2/ts_sim_cutfull_v1__sim_honest/fibs__command.npz"
    sha_ps = p2(ps, [(KS, "simh_x|ts", 1), (KS, "simh_x|ts", 2)])
    simh = {LEAD3[0]: [entry(ps, sha_ps, [s_sh]) | {"folder": "ts_sim_cutfull_v1__sim_honest"}]}
    for r in LEAD3[1:]:
        path = f"cutfull/simh/{r.split('/')[0]}/ts_sim_cutfull_v1__sim_honest/{r.split('/')[1]}"
        reader(path)
        simh[r] = [entry(path, "0" * 64, [s_sh]) | {"folder": "ts_sim_cutfull_v1__sim_honest"}]
    ov1["simh"] = simh
    ov2 = {"set": "S_old", "arms": {
        "te_main_v1__honest_b": arms("cutfull/q2", "te_main_cutfull_v1__honest_b", rows_e, [[s_te]] * 3),
        "tf_main_v1__attack": arms("cutfull/q2", "tf_main_cutfull_v1__attack", [(KY2, "run2|ty", 7)], [[s_tf]] * 3)}}
    catalogue = {"honest_b|tb": {"roles": ["B"]}, "honest_c|tc": {"roles": ["A", "B"]}, "simh_x|ts": {"roles": ["S"]},
                 "honest_b|tg": {"roles": ["S"]}, "honest_b|tn": {"roles": ["B"]}}
    cohort = {"S_new": {"ta", "tb"}, "S_old": {"tx", "ty"}}
    for root, rel, names in (("ev1", "L1/attack/t_a.eval",
                              ["bash", "text_editor", "client_bash", "bash", "bash", "text_editor", "submit"]),
                             ("ev2", "L2/honest_b/t_b.eval", ["bash", "bash", "bash", "text_editor"])):
        p = os.path.join(ev, root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        msgs = [{"role": "user", "content": "x"}] + [{"role": "assistant", "content": "",
                                                      "tool_calls": [{"id": str(i), "function": n, "arguments": {}}]}
                                                     for i, n in enumerate(names)]
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("header.json", "{}")
            z.writestr("samples/1_epoch_1.json", json.dumps({"messages": msgs}))
    open(os.path.join(ev, "ev2", "L2", "honest_b", "t_c.eval"), "wb").write(b"not a zip")  # -> eval_unreadable
    return ([("ov1", ov1), ("ov2", ov2)], cohort, catalogue, sc, sg,
            (os.path.join(ev, "ev1"), os.path.join(ev, "ev2")))


TOY_TOTALS = {  # known by construction (PRESTATED section 6)
    "H_B": {"units": 8, "bash": 6, "text_editor": 1, "other": {"db_bash": 1}, "unmapped_units": 0, "runs": 2,
            "runs_with_units": 2, "pct": 13, "half_tie": True, "missing_kind_units": 0, "conflict_runs": 0},
    "run1": {"units": 3, "bash": 1, "text_editor": 1, "other": {"client_bash": 1}, "unmapped_units": 3, "runs": 1,
             "runs_with_units": 1, "pct": 33, "half_tie": False, "missing_kind_units": 1, "pct_excl_missing_kind": 50,
             "conflict_runs": 0},
    "H_sim": {"units": 3, "bash": 1, "text_editor": 2, "other": {}, "unmapped_units": 0, "runs": 2, "runs_with_units": 2,
              "pct": 67, "half_tie": False, "missing_kind_units": 1, "pct_excl_missing_kind": 50, "conflict_runs": 1,
              "conflict_run_keys": ["honest_b|tg"], "pct_excl_conflict_runs": 100},
    "H_A": {"units": 4, "bash": 3, "text_editor": 0, "other": {"db_bash": 1}, "unmapped_units": 0, "runs": 1,
            "runs_with_units": 1, "pct": 0, "half_tie": False},
    "run2": {"units": 1, "bash": 1, "text_editor": 0, "runs": 1, "pct": 0, "conflict_runs": 1,  # run2|ty, first appearance
             "pct_excl_conflict_runs": None},
    "sim": {"units": 0, "runs": 0, "pct": None},
}
TOY_GATES = {"folders_listed": 7, "folders_counted": 5,
             "folders_skipped": {"common_staged_shas_2": 1, "reader_not_listed": 1},
             "staged_files": 5, "staged_resolved_by_name": 4, "staged_resolved_by_index": 1,
             "staged_sha_on_several_paths": 0, "run_keys_with_several_epochs": 1, "dup_epoch_keys": 0,
             "any_call_missing_kind": True, "calls_missing_kind_with_readout": 3,
             "missing_kind_units_any": {"H_B": False, "run1": True, "H_sim": True, "run2": False, "sim": False, "H_A": False},
             "outside_world_tokens": ["honest_b", "run1", "run2"], "cohort_rules_same": False,
             "dedup_conflict_runs": {"H_B": 0, "run1": 0, "H_sim": 1, "run2": 1, "sim": 0, "H_A": 0},
             "dedup_key_strings_differ_runs": {"H_B": 1, "run1": 0, "H_sim": 1, "run2": 1, "sim": 0, "H_A": 0}}
TOY_XCHECK = {"epochs_resolved": 2, "staged_equal": 1, "staged_sub_multiset": 1, "eval_not_found": 4, "eval_unreadable": 1,
              "eval_on_several_roots": 0, "units_sub_multiset": 2, "runs_resolved": 2, "runs_resolved_all_units_sub_multiset": 2}
TOY_COMPARE = ({(g, k, "MATCH") for g in ("H_B", "run1", "H_sim", "H_A") for k in
                ("units", "bash", "text_editor", "unmapped_units", "runs_with_units", "other_by_name", "other_sum", "runs",
                 "pct_eq_record") + (("pct_mine_eq_printed", "pct_record_eq_printed") if g in TOY_PRINTED else ())}
               | {("run2", "cell", "ABSENT"), ("sim", "cell", "ABSENT")}  # Codex r1 #14: the complete expected row set
               | {(g, f"{k}@{v}", "MATCH") for g in TOY_PRINTED for k in ALT_KEYS for v in ("union", "first_eligible")})


def run_toy(workers: int = 2) -> list:
    """Returns the failed assertions (empty = pass). Every expectation is a hand-written constant above."""
    fails = []
    with tempfile.TemporaryDirectory() as base:
        ovs, cohort, cat, sc, sg, roots = _build_toy(base)
        res = compute(ovs, cohort, cat, sc, sg, workers)
        for g, exp in TOY_TOTALS.items():
            fails += [f"toy {g}.{k}" for k, v in exp.items() if res["totals"][g][k] != v]
        fails += [f"toy gate {k}" for k, v in TOY_GATES.items() if res["gates"][k] != v]
        rows = compare_all(res, _toy_record(False), TOY_PRINTED)
        if set(rows) != TOY_COMPARE or len(rows) != len(TOY_COMPARE):
            fails.append("toy compare rows differ from the complete expected set")
        a = res["alt"]  # reviewer 2 / Codex r3 #13: union adds run2|tx; first_eligible takes run2|ty from its S_old folder
        if ((a["union"]["run2"]["units"], res["totals"]["run2"]["units"], a["first_eligible"]["run2"]["text_editor"],
             a["first_eligible"]["run2"]["bash"], a["first_eligible"]["run1"]["units"]) != (2, 1, 1, 0, 3)):
            fails.append("toy alternative totals")
        if res["detail"]["missing_kind_units"] != {"H_B": 0, "run1": 1, "H_sim": 1, "run2": 0, "sim": 0, "H_A": 0}:
            fails.append("toy missing_kind_units detail")
        if not all(primary_match(rows, g) for g in TOY_PRINTED) or primary_match(rows, "run2"):
            fails.append("toy primary_match")
        part = compare(res["totals"], {"cells": {"pooled|H_B": {"runs": 2, "readers": {"p2": {"text_editor": 1}}}}},
                       TOY_PRINTED)  # Codex r1 #7: record pct operands absent
        want = {("H_B", "pct_eq_record", "ABSENT"), ("H_B", "pct_record_eq_printed", "ABSENT"),
                ("H_B", "pct_mine_eq_printed", "MATCH"), ("H_B", "units", "ABSENT")}
        if not want <= set(part) or primary_match(part, "H_B"):
            fails.append("toy absent-operand rows")
        tot, neg_rows = res["totals"], compare_all(res, _toy_record(True), TOY_PRINTED)
        cf = {"missing_kind_units": 0, "conflict_runs": 1, "pct": 13, "pct_excl_missing_kind": 13}
        for got, want in ((verdict(rows, tot["H_B"], "H_B")["verdict"], "MATCH"),  # clean printed row
                          (verdict(rows, tot["run1"], "run1")["verdict"], "UNRESOLVED"),  # G2: 50 != 33
                          (verdict(rows, tot["run1"], "run1")["missing_kind_pct_same"], False),
                          (verdict(neg_rows, tot["H_B"], "H_B")["verdict"], "MISMATCH"),
                          (verdict(rows, cf | {"pct_excl_conflict_runs": 14}, "H_B")["verdict"], "UNRESOLVED"),  # G3
                          (verdict(rows, cf | {"pct_excl_conflict_runs": 13}, "H_B")["verdict"], "MATCH"),
                          (key_class("H_B", "runs"), "definition"), (key_class("run1", "units"), "primary"),
                          (key_class("H_sim", "units"), "diagnostic"), (key_class("H_B", "bash"), "secondary")):
            if got != want:
                fails.append(f"toy verdict/class: got {got!r} want {want!r}")
        x = xcheck(res, roots, workers)
        fails += [f"toy xcheck {k}" for k, v in TOY_XCHECK.items() if x["counts"].get(k, 0) != v]
        if x["n_runs_sampled"] != 6:
            fails.append("toy xcheck sample size")
    with tempfile.TemporaryDirectory() as base:  # perturbation: one staged bash -> text_editor
        ovs, cohort, cat, sc, sg, _ = _build_toy(base, perturb=True)
        t = compute(ovs, cohort, cat, sc, sg, workers)["totals"]["H_B"]
        if (t["text_editor"], t["units"]) != (TOY_TOTALS["H_B"]["text_editor"] + 1, TOY_TOTALS["H_B"]["units"]):
            fails.append("toy perturbation did not move H_B text_editor by exactly 1")
    with tempfile.TemporaryDirectory() as base:  # a staged sha found nowhere must stop the run
        ovs, cohort, cat, sc, sg, _ = _build_toy(base, wrong_staged=True)
        try:
            compute(ovs, cohort, cat, sc, sg, workers)
            fails.append("toy wrong staged sha did not raise")
        except ValueError as e:
            if "staged shas not found" not in str(e):
                raise
    with tempfile.TemporaryDirectory() as base:  # Codex r1 #1: p2 columns in the other order must stop the run
        ovs, cohort, cat, sc, sg, _ = _build_toy(base, swap_p2=True)
        try:
            compute(ovs, cohort, cat, sc, sg, workers)
            fails.append("toy swapped p2 columns did not raise")
        except ValueError as e:
            if "not in the expected forms" not in str(e):
                raise
    return fails


def run_toy_negative(workers: int = 2) -> bool:
    """Predicted to FAIL: True iff (a) the clean toy against the planted wrong record mismatches exactly TOY_PLANTED and
    (b) the perturbed toy against the clean record mismatches exactly TOY_PERTURBED with H_B's primary match false
    (reviewer 5)."""
    with tempfile.TemporaryDirectory() as base:
        ovs, cohort, cat, sc, sg, _ = _build_toy(base)
        planted = compare_all(compute(ovs, cohort, cat, sc, sg, workers), _toy_record(True), TOY_PRINTED)
    with tempfile.TemporaryDirectory() as base:
        ovs, cohort, cat, sc, sg, _ = _build_toy(base, perturb=True)
        perturbed = compare_all(compute(ovs, cohort, cat, sc, sg, workers), _toy_record(False), TOY_PRINTED)
    got_p = {f"{g}.{k}" for g, k, s in planted if s == "MISMATCH"}
    got_q = {f"{g}.{k}" for g, k, s in perturbed if s == "MISMATCH"}
    print(f"TOY_NEGATIVE planted_exact={got_p == TOY_PLANTED} n_planted={len(got_p)} perturbed_exact={got_q == TOY_PERTURBED} "
          f"n_perturbed={len(got_q)} perturbed_primary_H_B={primary_match(perturbed, 'H_B')}")
    return got_p == TOY_PLANTED and got_q == TOY_PERTURBED and not primary_match(perturbed, "H_B")


# ---------------------------------------------------------------- Modal

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.1.3")
VOLUMES = {m: modal.Volume.from_name(n).with_mount_options(read_only=True) for m, n in IN_VOLUMES.items()}
out_vol = modal.Volume.from_name(APP_NAME, create_if_missing=True)


def _write_x(path: str, text: str) -> str:
    """Append-only (Codex r1 #4): exclusive create, never overwrite."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "x") as f:
        f.write(text)
    return sha256_file(path)


@app.function(image=image, cpu=8.0, memory=16384, timeout=TIMEOUT_S, retries=0, volumes={**VOLUMES, OUT: out_vol})
def run_real(blobs: dict, meta: dict) -> str:
    """Codex r1 #2: a failure leaves the container only as a fixed identifier (stage, exception type). The traceback goes
    to the output Volume, and the identifier is raised outside the except block, so no exception message is chained.
    Codex r3 #4: every attempt writes into its own directory appa/<stamp>/<UTC time>_<random hex>, created exclusively, so a
    restarted or concurrent container can never overwrite another's files."""
    attempt = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:12]}"
    d = os.path.join(OUT, "appa", meta["stamp"], attempt)
    stage, err = ["start"], None
    try:
        os.makedirs(os.path.dirname(d), exist_ok=True)
        os.mkdir(d)  # exclusive: raises if it exists
        return _real(blobs, meta | {"attempt": attempt, "function_call_id": modal.current_function_call_id(),
                                    "input_id": modal.current_input_id()}, stage, d)
    except Exception as e:  # never swallowed: recorded on the Volume, then re-raised sanitized below
        err = (type(e).__name__, traceback.format_exc())
    note = ""
    try:  # Codex r3 #15: a failure to record the traceback must not replace the sanitized failure
        _write_x(os.path.join(d, f"error_appa_{stage[0]}.txt"), err[1])
        out_vol.commit()
    except Exception as e2:
        note = f" error_record={type(e2).__name__}"
    raise RuntimeError(f"APPA_FAILED stage={stage[0]} type={err[0]}{note}")


def _real(blobs: dict, meta: dict, stage: list, d: str) -> str:
    workers = 8
    stage[0] = "pins"
    for name, (_, _, sha) in PINS.items():
        if sha256_bytes(blobs[name]) != sha:
            raise ValueError(f"pinned input differs in the container: {name}")
    cfg = {name: json.loads(blobs[name]) for name in PINS}
    for name, path in FULL_BINDING.items():
        node = cfg["INPUTS_FULL"]
        for k in path:
            node = node[k]
        if node["sha256"] != PINS[name][2]:
            raise ValueError(f"INPUTS_FULL does not pin {name}")
    stage[0] = "toy"
    fails = run_toy(workers)
    if fails:
        raise ValueError(f"toy positive control failed: {sorted(fails)}")
    stage[0] = "toy_negative"
    if not run_toy_negative(workers):
        raise ValueError("toy-negative did not mismatch exactly the planted and perturbed keys")
    stage[0] = "inputs"
    raw = open(os.path.join(CATDIR, CATALOGUE_REL), "rb").read()
    if sha256_bytes(raw) != CATALOGUE_SHA:
        raise ValueError("catalogue sha differs")
    catalogue = json.loads(raw)
    cohort = {s: set(v["cells"]) for s, v in cfg["exclusions"]["final_cohort_keys"].items()}
    if sum(len(v) for v in cohort.values()) != N_COHORT:
        raise ValueError("cohort is not 1,313 cells")
    stage[0] = "compute"
    res = compute([(n, cfg[n]) for n in OVERRIDES], cohort, catalogue["runs"], SCORES, STAGED, workers)
    stage[0] = "compare"
    rows = compare_all(res, cfg["record"], PRINTED)
    rec_cells = {g: cfg["record"]["cells"].get(f"pooled|{g}") for g in GROUPS}
    stamp = meta["stamp"]
    rpath, xpath = os.path.join(d, f"results_appa_{stamp}.json"), os.path.join(d, f"xcheck_appa_{stamp}.json")
    primary = {g: primary_match(rows, g) for g in PRINTED}
    verdicts = {g: verdict(rows, res["totals"][g], g) for g in PRINTED}
    doc = {"what": "appA independent re-derivation of App A 17% / 21% (PRELIMINARY until 09)", "meta": meta | {
               "app": APP_NAME, "utc": datetime.now(timezone.utc).isoformat(), "pins": PINS, "catalogue": [CATALOGUE_REL,
               CATALOGUE_SHA], "volumes": IN_VOLUMES, "config": {"cpu": 8, "memory_mib": 16384, "timeout_s": TIMEOUT_S,
               "workers": workers, "seed": SEED, "n_sample": N_SAMPLE, "groups": GROUPS, "printed": PRINTED}},
           "toy": {"positive_failures": fails, "negative_planted": sorted(TOY_PLANTED),
                   "negative_perturbed": sorted(TOY_PERTURBED)},
           "mine": res["totals"], "mine_alternatives": res["alt"], "record_cells": rec_cells,
           "compare": [{"group": g, "key": k, "status": s, "class": key_class(g, k)} for g, k, s in rows],
           "primary_match": primary, "verdicts": verdicts, "gates": res["gates"], "detail": res["detail"]}
    stage[0] = "write"
    rsha = _write_x(rpath, json.dumps(doc, indent=1, sort_keys=True))
    out_vol.commit()
    # Reviewer 1: the main result is printed from the container, committed, BEFORE the optional cross-check runs.
    for g, k, s in rows:
        print(f"APPA_KEY {g} {k} {s} {key_class(g, k)}", flush=True)
    for g in PRINTED:
        print(f"APPA_VERDICT {g} " + " ".join(f"{k}={v}" for k, v in verdicts[g].items()), flush=True)  # booleans only
    result = {"verdict": {g: verdicts[g]["verdict"] for g in PRINTED}, "sensitivity": verdicts,
              "definition_difference_keys": [f"{g}.{k}" for g, k, s in rows
                                             if s == "MISMATCH" and key_class(g, k) == "definition"],
              "primary_match": primary, "n_match": sum(s == "MATCH" for *_, s in rows),
              "n_mismatch": sum(s == "MISMATCH" for *_, s in rows),
              "mismatch_keys": [f"{g}.{k}" for g, k, s in rows if s == "MISMATCH"],
              "absent_keys": [f"{g}.{k}" for g, k, s in rows if s == "ABSENT"],
              "half_tie": {g: res["totals"][g]["half_tie"] for g in PRINTED}, "gates": res["gates"],
              "toy_positive_pass": True, "toy_negative_exact": True,
              "results": f"{APP_NAME}:{rpath[len(OUT) + 1:]}", "results_sha256": rsha}
    print("APPA_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    stage[0] = "xcheck"  # a crash here can no longer hide the verdict: it is committed and printed above
    x = xcheck(res, EVAL_ROOTS, workers)
    xsha = _write_x(xpath, json.dumps({"meta": {"stamp": stamp, "results_sha256": rsha}, "xcheck": x}, indent=1,
                                      sort_keys=True))
    out_vol.commit()
    print("APPA_XCHECK " + json.dumps(x["counts"] | {"n_runs_sampled": x["n_runs_sampled"],
                                                    "file": f"{APP_NAME}:{xpath[len(OUT) + 1:]}", "sha256": xsha},
                                      sort_keys=True), flush=True)
    return json.dumps({"verdict": result["verdict"]})


@app.local_entrypoint()
def main(mode: str = "real", repo: str = "", stamp: str = ""):
    """Exit 0: both printed rows MATCH. Exit 3: a printed row is MISMATCH or UNRESOLVED (results committed; STOP and report
    keys only, PRESTATED section 5; Codex r1 #3). Any other non-zero exit: the run failed (see APPA_FAILED)."""
    if mode != "real" or not repo or not re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{6}", stamp):
        raise SystemExit("usage: \"$M\" run run_appa.py --mode real --repo <worktree> --stamp <yyyy-mm-dd_HHMMSS>")
    blobs = {}
    for name, (commit, path, sha) in PINS.items():
        b = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
        if sha256_bytes(b) != sha:
            raise SystemExit(f"PIN_MISMATCH {name}")
        blobs[name] = b
    here = os.path.abspath(__file__)
    head = subprocess.run(["git", "-C", os.path.dirname(here), "rev-parse", "HEAD"], capture_output=True, check=True,
                          text=True).stdout.strip()
    meta = {"git_head": head, "script_sha256": sha256_file(here), "stamp": stamp, "repo": repo}
    print(f"APPA_PINS_OK n={len(blobs)} head={head} script_sha256={meta['script_sha256']}")
    out = json.loads(run_real.remote(blobs, meta))  # the container prints APPA_KEY / APPA_VERDICT / APPA_RESULT / APPA_XCHECK
    print("APPA_DONE verdict=" + json.dumps(out["verdict"], sort_keys=True), flush=True)
    if any(v != "MATCH" for v in out["verdict"].values()):
        sys.exit(3)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "toy":
        f = run_toy()
        print("TOY_PASS" if not f else f"TOY_FAIL {sorted(f)}")
        sys.exit(0 if not f else 1)
    if which == "toy-negative":
        sys.exit(1 if run_toy_negative() else 2)
    sys.exit("usage: python run_appa.py toy|toy-negative")
