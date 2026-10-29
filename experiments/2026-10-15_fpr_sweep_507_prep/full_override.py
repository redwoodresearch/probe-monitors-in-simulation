"""Full-length (uncut) substitution for the Stage A all-tasks job: 09's RULE lead §3-§4 (@ 129b7c4f1, cfd94c2e9) and the schema agreed with the
CUT fork (one override per set per readout, cutfull override format + "set"; per-entry G2/G3 gate pointers). Pure functions, no I/O.

The substitution is cutfull_decisive.apply_override (aswin/cutfull-dg-2e @ b3b17eb53), reused verbatim through exact-name shims, one per cfg block:
  S_new        cfg["arms"] (su311_*) + cfg["simh_files"]           (apply_override as the decisive job runs it)
  S_newmains   cfg["arms_nm"] + cfg["simh_nm"]                      (every arm name; prefix "")
  S_paper493   cfg["sp493"]["arms"] + cfg["sp493"]["simh"]          (every arm name; prefix "")
  S_old        probe["old"] -> the S_old override's real folders (inventory world != relaunch / sim_honest); probe["relaunch"] -> its relaunch
               folders; cfg_old["simh_old"][label] -> its sim-honest entries, matched by root (or by batch where the cut entry has no root)
Fail closed everywhere: an arm, root or batch whose cut cfg holds a probe file with no uncut entry stops the job; a read of any file outside the
pinned uncut entries stops the row (read_guard)."""

import copy

import cutfull_decisive as CD

SETS = ("S_old", "S_new", "S_newmains", "S_paper493")
EXTERNAL = ("FIBS", "harmfulness", "refusal", "high-stakes")
# the registered-file families this build scores uncut (C and R); each row leaves as "<family>|uncut <C|R>" (registered weights scored uncut)
REGISTERED = ("Sol (a)",) + EXTERNAL + ("own_all_rs_L48", "own_all_rs25_L48", "own_all_rs10_L48", "c_asis", "own_frac10_L48", "own_frac25_L48",
                                       "own_frac100_L48")
SCOPE = REGISTERED
UNCUT = "|uncut"
# whole-transcript retrains -> their registered counterpart (row rules, folds, pools); each row leaves as "<retrain> <C|R>"
RETRAIN_OF = {"sol_a_uncut": "Sol (a)", "own_all_rs_uncut_L48": "own_all_rs_L48", "own_all_rs25_uncut_L48": "own_all_rs25_L48",
              "own_all_rs10_uncut_L48": "own_all_rs10_L48", "c_asis_uncut": "c_asis",
              "sol_a_uncut_pinned": "Sol (a)"}  # sol_a_uncut with layers pinned to the 32k fit (09 @ 2d61af835): outside every group; sol_a_uncut leads
HEADLINE_SIX = ("sol_a_uncut",) + tuple(x + UNCUT for x in EXTERNAL) + ("own_all_rs_uncut_L48",)
OFF_FIVE = HEADLINE_SIX[:5]
REG_SIX = ("Sol (a)" + UNCUT,) + tuple(x + UNCUT for x in EXTERNAL) + ("own_all_rs_L48" + UNCUT,)
POSITIVE_CONTROL = ("c_asis_uncut", "c_asis" + UNCUT)  # appendix positive control: never in S or any co-headline group
# ADDENDUM labels (4f's exact names): uncut LOO probes, scored by the held-out origin's model. Outside every group and outside LABELS_OUT:
# they do NOT gate the freeze and run AFTER the full-length Stage A as their own labelled job (same cohort and draws; gated on primary-arm
# equality by key with the frozen Stage A record). LOO3 refuses S_paper493; LOO4 needs S_paper493 waves 2-3.
# 09 (10-16 ~04:0x, both builds): fold -1 cells are kept and scored, never dropped; the full-length record asserts 0 actual drops for these
NO_FOLD_DROPS = ("Sol (a)" + "|uncut", "c_asis" + "|uncut", "sol_a_uncut", "sol_a_uncut_pinned", "c_asis_uncut", "own_all_rs_L48|uncut", "own_all_rs25_L48|uncut",
                 "own_all_rs10_L48|uncut", "own_all_rs_uncut_L48", "own_all_rs25_uncut_L48", "own_all_rs10_uncut_L48")
ADDENDUM = {"own_all_LOO4_uncut_L48": "uncut LOO-4 own_all (held-out origin's model; folds T196 / S_new / S_newmains / S_paper493); addendum",
            "own_all_LOO3_uncut_L48": "uncut LOO-3 own_all (held-out origin's model; folds T196 / S_new / S_newmains; refuses S_paper493); addendum"}
ADDENDUM_LABELS = tuple(f"{n} {ro}" for ro in "RC" for n in ADDENDUM)
ADDENDUM_OF = {k: "own_all_LOO_L48" for k in ADDENDUM}  # an addendum runs under the 32k LOO label (its held-out-origin model reader) and is renamed
OWN_ALL_KINDS = ("own_all_t196", "own_all_loo_t196", "own_all_rs_t196")
AUROC06_FULL = {"Sol (a)": "sol_a_uncut", "FIBS": "FIBS" + UNCUT, "harmfulness": "harmfulness" + UNCUT, "own_all_rs_L48": "own_all_rs_uncut_L48"}  # adds (d)
FAMILIES = tuple(x + UNCUT for x in REGISTERED) + tuple(RETRAIN_OF)
LABELS_OUT = tuple(f"{n} {ro}" for ro in "CR" for n in FAMILIES)  # every label the full-length record carries
LABELS = ({x + UNCUT: "registered probe scored on full transcripts (OOD past 32k)" for x in REGISTERED if x not in EXTERNAL}
          | {x + UNCUT: "trained on short external text; scored as is on full transcripts (OOD class per the signed table)" for x in EXTERNAL}
          | {k: "whole-transcript retrain (uncut) of " + v for k, v in RETRAIN_OF.items()}
          | {"sol_a_uncut_pinned": "sol_a_uncut with layers pinned to the 32k fit (09 @ 2d61af835); outside every group (sol_a_uncut is the primary)",
             "c_asis_uncut": "whole-transcript retrain (uncut) of c_asis; appendix positive control, never in S or a co-headline group",
             "c_asis" + UNCUT: "registered c_asis scored on full transcripts; appendix positive control, never in S or a co-headline group"})


def out_label(job_label, retrain=None):
    """The full-length record's label for a job run under a registered label (and optionally as a retrain variant)."""
    base, ro = job_label.rsplit(" ", 1)
    return f"{retrain} {ro}" if retrain else f"{base}{UNCUT} {ro}"


NOT_IN_BUILD = "not in the full-length build (09 RULE lead §1: headline probes, registered-scored-uncut and the four external probes only)"


# 09 (10-16, via 507-prep): in the full-length record the headline is the four-set pooled combo, one main once (pooled_all_origins); pooled_all is
# a labelled line. Exactly one combo (or 'alltasks.<combo>' role entry) is the lead; "lead-2" / "lead-3" are other tiers, not the lead.
FULL_COMBO_ROLE = {"pooled_all": "pooled_all: three origins, per-origin weighting (not the headline)",
                   "pooled_all_without_S4": "beside pooled_all: three origins without S4, per-origin weighting (not the headline)",
                   "pooled_all_origins": "lead: the four-set pooled combo (S_old, S_new, S_newmains, S_paper493), one main once (cells 1/n_main), union stream"}


def lead_check(roles):
    """STOP unless exactly one role is the lead ("lead", "lead:", "lead (...": not "lead-2") and it is pooled_all_origins. -> that key."""
    import re

    leads = sorted(k for k, v in roles.items() if re.match(r"lead(?![-\w])", str(v)))
    if [k.removeprefix("alltasks.") for k in leads] != ["pooled_all_origins"]:
        raise SystemExit(f"full length: the lead must be exactly pooled_all_origins, not {leads}")
    return leads[0]


def combine(ovs):
    """One set's C and R overrides -> one {arms, simh} (probe files are disjoint across readouts; a shared key with different entries stops)."""
    arms, simh = {}, {}
    for ov in ovs:
        for folder, files in (ov.get("arms") or {}).items():
            for fn, e in files.items():
                if arms.setdefault(folder, {}).setdefault(fn, e) != e:
                    raise SystemExit(f"override {folder}|{fn}: two readouts' overrides differ")
        for fn, ents in (ov.get("simh") or {}).items():  # Codex d14ba40d9 #5: a part's roots join the set's list by root; a shared root must agree
            cur = simh.setdefault(fn, [])
            for e in ents:
                same = [x for x in cur if x.get("root") == e.get("root")]
                if same and same[0] != e:
                    raise SystemExit(f"override H_sim {fn} root {e.get('root')}: two overrides' entries differ")
                if not same:
                    cur.append(e)
    return {"arms": arms, "simh": simh}


def _shim(probes, arms, simh, ov, members, prefix, alias):
    """apply_override on one block. alias {registered file: retrain file}: the block's cut entries are looked up under the retrain's file name."""
    fake = {"probes": probes, "arms": arms, "simh_files": simh}
    if alias:
        ov = {"arms": {f: {r: e[a] for r, a in alias.items() if a in e} for f, e in ov.get("arms", {}).items()},
              "simh": {r: ov["simh"][a] for r, a in alias.items() if a in ov.get("simh", {})}}
    c, applied = CD.apply_override(fake, ov, members, snew_prefix=prefix)
    return c["arms"], c["simh_files"], applied


# 09 route C (10-17): the ONE approved S_old sim-honest allowlist (Astra R1: hard-coded, the root bound to its counted-keys file). The root is
# the pilot's ORIGINAL folder (the override's and roles_old's name); its rows are keyed <world>|<task> with world simh_pilot.
S_OLD_EXTRA = {"simh_q4f_v1_pilot__sim_honest": {"commit": "344829af6",
                                                 "path": "experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_pilot_v3.json",
                                                 "sha256": "2149c5cde368aef9e5012c8d9cda5b12fe5a93db3d27ad2d525225bb9f511b27"}}
S_OLD_EXTRA_WORLD, S_OLD_EXTRA_N = "simh_pilot", 20
ROUTEC_PREFLIGHT_WHAT = "route C pre-flight (09 / 2e, 10-17): pilot-key STOP risks, zero-unit runs and filter drops, keys and counts only"
S_OLD_ONLY = "s_old_only"  # tag on an allowlisted entry: S_old tau_sim only (RULE l.4453 §3); every other origin's rows drop it (09 B1 (a))


def s_old_extra_pin(fin):
    """INPUTS_FULL -> None when s_old_simh_extra_roots is absent, else the approved allowlist. Present means EXACTLY S_OLD_EXTRA: null, an
    empty or a second entry, another root or another pin STOPs (Astra R1 / R2)."""
    if "s_old_simh_extra_roots" not in fin:
        return None
    if fin["s_old_simh_extra_roots"] != S_OLD_EXTRA:
        raise SystemExit(f"INPUTS_FULL s_old_simh_extra_roots: present but not the approved allowlist {S_OLD_EXTRA} (09 route C)")
    return S_OLD_EXTRA


def s_old_extra_keys(manifest):
    """The pinned counted-keys file -> its runner keys <world>|<task>, sorted. STOPs unless n = 20 distinct keys, every one world simh_pilot
    and stratum S_old (Astra R3: the file's contents are used, namespace included)."""
    ks_ = manifest.get("keys") or []
    keys = [f"{k.get('world')}|{k.get('task')}" for k in ks_]
    if (manifest.get("n") != S_OLD_EXTRA_N or len(keys) != S_OLD_EXTRA_N or len(set(keys)) != S_OLD_EXTRA_N
            or any(k.get("world") != S_OLD_EXTRA_WORLD or k.get("stratum") != "S_old" or not k.get("task") for k in ks_)):
        raise SystemExit(f"route C counted keys: not {S_OLD_EXTRA_N} distinct {S_OLD_EXTRA_WORLD}|<task> keys of stratum S_old")
    return sorted(keys)


def routec_zero_unit_record(rec, admitted):
    """507-prep's hardening: the record --routec-zero-unit pins must be a CLEAR route C pre-flight (its "what"), name the spec it read by
    sha256, and list only admitted pilot keys per pilot file sha. -> (zero_unit_by_file, the pre-flight spec's sha256). STOPs otherwise."""
    import re

    zu_ = rec.get("zero_unit_by_file")
    sh_ = (rec.get("spec") or {}).get("sha256")
    if rec.get("what") != ROUTEC_PREFLIGHT_WHAT or rec.get("status") != "CLEAR" or not re.fullmatch(r"[0-9a-f]{64}", str(sh_)):
        raise SystemExit("--routec-zero-unit: not a CLEAR route C pre-flight record naming its spec by sha256")
    if not isinstance(zu_, dict) or not all(re.fullmatch(r"[0-9a-f]{64}", str(k_)) and isinstance(v_, list) and set(v_) <= set(admitted)
                                            for k_, v_ in zu_.items()):
        raise SystemExit("--routec-zero-unit: zero_unit_by_file is not {pilot file sha256: [admitted pilot keys]}")
    return zu_, sh_


def flags_set(flags, defaults, own):
    """The flags given a non-default value, other than `own` (--launch-from-pickle takes no other flag: the pickle is the whole job)."""
    return sorted(k for k, v in flags.items() if k in defaults and k not in own and v != defaults[k])


def load_spec_checked(b, want_sha256):
    """O-P1 (10-17), in drive(): the spec pickle's bytes AS READ are hashed and must equal the sha256 the launcher pinned (dispatch: the
    uploaded blob's; launch_from: the gated pickle's; resume: --resume-spec <path>@<sha256>), BEFORE pickle.loads; else STOP, nothing
    unpickled. The same in-memory bytes are then unpickled."""
    import hashlib
    import pickle
    import re

    if not isinstance(want_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", want_sha256) or hashlib.sha256(b).hexdigest() != want_sha256:
        raise SystemExit("drive: the spec pickle's sha256 is not the one the launcher pinned (nothing unpickled; STOP)")
    return pickle.loads(b)


def resume_spec_pin(s):
    """--resume-spec <Volume path>@<sha256> -> (path, sha256); a missing or malformed sha STOPs (O-P1: a resume re-checks the spec too)."""
    import re

    m = re.fullmatch(r"(.+)@([0-9a-f]{64})", s or "")
    if not m:
        raise SystemExit("--resume-spec: <path>@<sha256> (the spec pickle's sha256, 64 hex) is required")
    return m.group(1), m.group(2)


def launch_pickle_checks(pickle_sha, want_sha, verdict, head, porcelain):
    """2e option A (10-17): the gated spec-only pickle, launched as is; every check before ANY upload, and NOTHING unpickled (R2): the
    pickle's sha256 equals the named one AND the passing verdict's side b (the verdict recomputed at launch); side b was built at HEAD
    (verdict b_meta.git); the tracked tree is clean; it names a .json record. -> (the Volume spec path as dispatch() names it, pooled)."""
    from pathlib import Path

    bm_ = (verdict or {}).get("b_meta") or {}
    if pickle_sha != want_sha or not isinstance(verdict, dict) or verdict.get("pass") is not True or verdict.get("b_sha256") != pickle_sha:
        raise SystemExit("--launch-from-pickle: the pickle's sha256 is not the named one, or not the passing gate's side b (STOP before any upload)")
    if porcelain.strip():
        raise SystemExit("--launch-from-pickle: uncommitted tracked changes (STOP before any upload)")
    if bm_.get("git") != head or not str(bm_.get("dest", "")).endswith(".json"):
        raise SystemExit("--launch-from-pickle: side b was not built at HEAD, or names no record (STOP before any upload)")
    pre_ = "cutfull/" if bm_["dest"].startswith("cutfull/") else ""  # as dispatch() names it
    return f"{pre_}xfer/fpr507prep/drivers/{Path(bm_['dest']).name.removesuffix('.json')}.pkl", bool(bm_.get("pooled"))


def missing_inputs(shipped, have):
    """R1: the content-addressed inputs the spec's rows read (meta.shipped_inputs) that are not on the Volume (a listing of inputs/by_sha)."""
    return sorted(k for k, v in (shipped or {}).items() if v.get("sha256") not in have or v.get("path") != f"xfer/fpr507prep/inputs/by_sha/{v.get('sha256')}")


def simh_old_for(v, simh_old):
    """09 B1 (a): the S_old list for a job variant. An S_old variant reads it as is; every other origin's variant (S_paper493 copies it) gets
    it without the S_old-only entries, at build time, so the spec shows the drop. No tagged entry: the same object, unchanged."""
    if v.get("origin") == "S_old" or not any(e.get(S_OLD_ONLY) for es in simh_old.values() for e in es):
        return simh_old
    return {lb: [e for e in es if not e.get(S_OLD_ONLY)] for lb, es in simh_old.items()}


def s_old_only_guard(ents, orig, label):
    """Row time: an S_old-only entry on any other origin's row STOPs (the build drops it; this is the reader's own check)."""
    if orig != "S_old" and any(e.get(S_OLD_ONLY) for e in ents):
        raise SystemExit(f"{label} {orig}: an S_old-only H_sim entry (route C) on a non-S_old row")


def s_old_extra_check(e, keys, label):
    """Row time, where keys are decoded (09 B2 / Astra R3): an allowlisted entry's run keys must be its admitted keys, namespace included; an
    extra key STOPs. An admitted key without rows is a zero-unit run (09, 10-17: kept as a member with 0 units, listed by key, not a STOP),
    admitted as S_paper493's are (Claude review R3): only when the 09-reviewed pre-flight list pinned for this file (e["zero_unit_ok"])
    names it; zero_unit_unlisted (the row is WITHHELD on any) otherwise. -> {n, keys_sha256, zero_unit, zero_unit_unlisted}."""
    import hashlib

    got, want = set(keys), set(e["admitted"])
    if got - want:
        raise SystemExit(f"{label}: route C root {e.get('root')}: {len(got - want)} run keys outside the pinned {len(want)}, e.g. {sorted(got - want)[:3]}")
    return {"n": len(want), "keys_sha256": hashlib.sha256("\n".join(sorted(want)).encode()).hexdigest(), "zero_unit": sorted(want - got),
            "zero_unit_unlisted": sorted(want - got - set(e.get("zero_unit_ok") or []))}


def s_old_simh(cut_, entries, inv_simh_roots, extra_roots, label, zero_unit_ok=None):
    """S_old H_sim for one label, full length -> (registered uncut entries, allowlisted extra entries). The registered roots must equal the
    inventory's sim-honest roots and the 32k list's roots / batches (unchanged checks, Codex r1 #3). 09 route C (10-17): extra_roots
    {root: admitted keys} names roots the 32k list lacks; each must be an inventory sim-honest root with exactly one uncut entry here, and its
    entry is tagged S_old-only and carries its admitted keys (checked at row time). Any root not listed still STOPs by the registered checks.
    With no allowlist the result is the registered list, unchanged. zero_unit_ok {file sha256: [keys]}: the 09-reviewed pre-flight's zero-unit
    pilot runs per pilot file (R3); each tagged entry carries its own file's list (empty when none is pinned)."""
    ext_ = extra_roots or {}
    if any(S_OLD_ONLY in e for e in list(entries) + list(cut_)):  # (d): only this function tags; a registered or 32k entry never carries it
        raise SystemExit(f"override S_old H_sim {label}: an override or 32k entry already carries the S_old-only tag")
    if any(not isinstance(v_, list) or not v_ or len(set(v_)) != len(v_) for v_ in ext_.values()):
        raise SystemExit(f"override S_old H_sim {label}: an allowlisted root without a list of distinct admitted keys")
    unc_ = [e for e in entries if e.get("root") not in ext_]
    add_ = [e for e in entries if e.get("root") in ext_]
    need_ = sorted(r for r in inv_simh_roots if r not in ext_)  # every S_old sim-honest root of the inventory, once each (Codex r1 #3: no root may drop inside a batch)
    if sorted(e.get("root") for e in unc_) != need_ or not cut_:
        raise SystemExit(f"override S_old H_sim {label}: uncut roots {sorted(str(e.get('root')) for e in unc_)} != the inventory's {need_}")
    by_ = "root" if all("root" in e for e in cut_) else "batch"
    if sorted({e[by_] for e in cut_}) != sorted({e[by_] for e in unc_}) or len({e[by_] for e in cut_}) != len(cut_) and by_ == "root":
        raise SystemExit(f"override S_old H_sim {label}: {by_}s differ ({sorted(e[by_] for e in cut_)} vs {sorted({e[by_] for e in unc_})})")
    if ext_ and (sorted(e.get("root") for e in add_) != sorted(ext_) or not set(ext_) <= set(inv_simh_roots)):
        raise SystemExit(f"override S_old H_sim {label}: allowlisted extra roots {sorted(ext_)} need exactly one uncut entry each "
                         f"({sorted(str(e.get('root')) for e in add_)}) and an inventory sim-honest root")
    return unc_, [e | {S_OLD_ONLY: True, "admitted": sorted(ext_[e["root"]]), "zero_unit_ok": sorted((zero_unit_ok or {}).get(e["sha256"], []))}
                  for e in add_]


def apply_full(cfg, cfg_old, ovs, labels, roles_old, alias=None, skip=(), extra_roots=None, zero_unit_ok=None):
    """-> (cfg_u, cfg_old_u, applied, allowed shas). ovs {set: combined override}; labels: the registered-file labels in this job; roles_old
    {folder: "real" | "relaunch" | "simh"} for S_old (from the inventory's worlds); alias {registered file: retrain file} for a retrain variant.
    skip: sets left on their cut files (TEST dry runs only; the runner refuses a scored run with any set skipped)."""
    c, co = copy.deepcopy(cfg), copy.deepcopy(cfg_old)
    applied = []
    probes = [p for p in c["probes"] if p["label"] in labels]
    fns = {p["label"]: (alias or {}).get(p["new"], p["new"]) for p in probes}
    for st, (ak, sk, prefix) in {"S_new": ("arms", "simh_files", "su311_"), "S_newmains": ("arms_nm", "simh_nm", "")}.items():
        if c.get(ak) is None or st in skip:
            continue
        sub_ = {fn: c[sk][fn] for fn in (p["new"] for p in probes) if fn in c[sk]}
        c[ak], sub_u, ap = _shim(c["probes"], c[ak], sub_, ovs[st], labels, prefix, alias)
        c[sk] = c[sk] | sub_u
        applied += [dict(x, set=st) for x in ap]
    if "sp493" in c and "S_paper493" not in skip:
        sp = c["sp493"]
        pend_ = set(sp.get("families_pending") or {})  # 0d (05:2x): a family pending on S_paper493 in the 32k base stays WITHHELD (never admitted)
        lab_sp = [p["label"] for p in probes if p["new"] not in pend_]
        sub_ = {fn: sp["simh"][fn] for fn in (p["new"] for p in probes if p["label"] in lab_sp) if fn in sp["simh"]}
        sp["arms"], sub_u, ap = _shim(c["probes"], sp["arms"], sub_, ovs["S_paper493"], lab_sp, "", alias)
        applied += [{"set": "S_paper493", "withheld_pending_32k": p["label"]} for p in probes if p["label"] not in lab_sp]
        for fn, ents in sub_u.items():  # PERTASK's Codex (10-16): apply_override rebuilds each H_sim entry from the uncut one (keeping only the cut
            w_ = {e["root"]: e.get("sp_world") for e in sub_[fn]}  # entry's batch); S_paper493's rows need the cut entry's world alias (sp_world)
            sub_u[fn] = [dict(e, sp_world=w_[e["root"]]) if w_.get(e["root"]) else e for e in ents]
        sp["simh"] = sp["simh"] | sub_u
        lost_ = sorted(f"{fn}|{e['root']}" for fn, ents in sp["simh"].items() for e in ents if not e.get("sp_world"))
        if lost_:  # every S_paper493 sim-honest entry keeps its world alias (the row's sp_world rename; else its admitted keys never match)
            raise SystemExit(f"S_paper493 H_sim entries without their world alias (sp_world): {lost_[:5]}")
        co["sp493"] = sp
        applied += [dict(x, set="S_paper493") for x in ap]
    if c.get("e12") and "pools_ep2" not in skip:  # --calib e12: the epoch-2 folders and the S_old fill-in roots, from the pools_ep2 override
        e12, ovp = c["e12"], ovs["pools_ep2"]
        no_simh = {"arms": ovp["arms"], "simh": {f: [] for f in set(fns.values()) | {p["new"] for p in probes}}}  # arms only (Codex r6: no H_sim here)
        for org in ("S_old", "S_new", "S_newmains"):
            if org in e12:
                e12[org]["folders"], _, ap = _shim(c["probes"], e12[org]["folders"], {}, no_simh, labels, "", alias)
                applied += [dict(x, set="pools_ep2", e12_origin=org) for x in ap]
        for f in e12.get("fill_S_old", []):  # fill-in roots are sim-honest roots: their uncut entries sit in an override's simh block, by root
            for p in probes:
                if p["new"] not in f["files"]:
                    continue
                fn = fns[p["label"]]
                hits = [(st, e) for st, o in ovs.items() if st not in skip for e in (o.get("simh") or {}).get(fn, []) if e.get("root") == f["root"]]
                if len(hits) != 1:
                    raise SystemExit(f"override e12 fill-in {f['root']}|{fn}: {len(hits)} uncut entries (need exactly one)")
                st, e = hits[0]
                f["files"][p["new"]] = {"path": e["path"], "sha256": e["sha256"]}
                applied.append({"set": st, "e12_fill": f["root"], "label": p["label"], "uncut": {"path": e["path"], "sha256": e["sha256"]}})
        co["e12"] = e12
    if c.get("il_arms"):  # IL extra arms (LOADER's ilarms/v1 block names the 32k files): each to its uncut file from the set's override; a file
        # with no uncut entry is dropped, so the loader lists its keys as pending and that row's (B) falls back to (A) there, labelled (0d: optional)
        for org, folders in c["il_arms"].get("sets", {}).items():
            for folder, d in folders.items():
                for p in probes:
                    fn = p["new"]
                    if fn not in d.get("files", {}):
                        continue
                    e = (ovs.get(org) or {}).get("arms", {}).get(folder, {}).get(fns[p["label"]]) if org not in skip else None
                    if e is None:
                        del d["files"][fn]
                        applied.append({"set": org, "il_without_uncut": f"{folder}|{fns[p['label']]}", "label": p["label"]})
                        continue
                    d["files"][fn] = {"path": e["path"], "sha256": e["sha256"]}
                    applied.append({"set": org, "il_folder": folder, "label": p["label"], "uncut": {"path": e["path"], "sha256": e["sha256"]}})
        co["il_arms"] = c["il_arms"]
    ov = ovs.get("S_old") if "S_old" not in skip else None
    by_role = {r: sorted(f for f, x in roles_old.items() if x == r) for r in ("real", "relaunch", "simh")}
    for p in probes if ov is not None else []:
        fn = fns[p["label"]]
        real = [dict(ov["arms"][f][fn], folder=f, vol="conv") for f in by_role["real"] if fn in ov["arms"].get(f, {})]
        miss = [f for f in by_role["real"] if fn not in ov["arms"].get(f, {})]
        if miss or not real:
            raise SystemExit(f"override S_old {p['label']}: real folders without {fn}: {miss or 'all'}")
        applied += [{"set": "S_old", "label": p["label"], "cut": {k: p["old"].get(k) for k in ("kind", "path", "sha256")}, "uncut": real}]
        kind_ = p["old"]["kind"]
        p["old"] = {"kind": kind_ if kind_ in OWN_ALL_KINDS else "uncut_list", "uncut": True,  # own_all* keep load_own_all (rows_any's readers)
                    "files": {e["folder"]: {"path": e["path"], "sha256": e["sha256"], "vol": "conv"} for e in real}}
        if p.get("relaunch"):
            rl = [dict(ov["arms"][f][fn], folder=f) for f in by_role["relaunch"] if fn in ov["arms"].get(f, {})]
            if not rl:
                raise SystemExit(f"override S_old {p['label']}: no relaunch folder holds {fn}")
            key_ = f"uncut::{p['label']}"
            c["relaunch_files"] = dict(c["relaunch_files"], **{key_: [{"path": e["path"], "sha256": e["sha256"], "vol": "conv"} for e in rl]})
            applied.append({"set": "S_old", "label": p["label"], "relaunch_cut": p["relaunch"], "uncut": rl})
            p["relaunch"] = key_
        rla_ = [dict(ov["arms"][f][fn], folder=f) for f in by_role["relaunch"] if fn in ov["arms"].get(f, {})]
        if rla_:  # every probe's uncut relaunch rows, for the honest_b_relaunch alternate pool line (09 item 15; the RELAUNCH fork reads them)
            k2_ = f"uncut_relaunch::{p['label']}"
            c["relaunch_files"] = dict(c["relaunch_files"], **{k2_: [{"path": e["path"], "sha256": e["sha256"], "vol": "conv"} for e in rla_]})
            p["relaunch_uncut"] = k2_
            applied.append({"set": "S_old", "label": p["label"], "relaunch_pool_line": k2_, "uncut": rla_})
        cut_ = co.get("simh_old", {}).get(p["label"], [])
        unc_, add_ = s_old_simh(cut_, ov["simh"].get(fn, []), by_role["simh"], extra_roots, p["label"], zero_unit_ok)
        co["simh_old"] = dict(co["simh_old"], **{p["label"]: [{"batch": e["batch"], "root": e.get("root"), "path": e["path"], "sha256": e["sha256"]}
                                                             | ({S_OLD_ONLY: True, "admitted": e["admitted"], "zero_unit_ok": e["zero_unit_ok"]}
                                                                if e.get(S_OLD_ONLY) else {})
                                                             for e in unc_ + add_]})
        applied.append({"set": "S_old", "label": p["label"], "simh_cut": cut_, "simh_uncut": unc_ + add_} | ({"simh_extra_roots": add_} if add_ else {}))
    for hr in (c.get("ha25f") or {}).get("roots", []):  # 25f honest-A roots: matched by root folder in exactly one set's override
        folder = hr["root"].split("/")[0]
        for p in probes:
            if p["new"] not in hr["files"]:
                continue
            hits = [(st, o["arms"][folder][fns[p["label"]]]) for st, o in ovs.items() if st not in skip and fns[p["label"]] in o.get("arms", {}).get(folder, {})]
            if not hits and skip:
                continue
            if len(hits) != 1:
                raise SystemExit(f"override 25f root {hr['root']}: {len(hits)} uncut {fns[p['label']]} entries (need exactly one)")
            st, e = hits[0]
            hr["files"][p["new"]] = {"path": e["path"], "sha256": e["sha256"]}
            applied.append({"set": st, "ha25f_root": hr["root"], "label": p["label"], "uncut": {"path": e["path"], "sha256": e["sha256"]}})
    c["probes"] = co["probes"] = [p for p in c["probes"] if p["label"] in labels]
    c["full_uncut"] = co["full_uncut"] = True
    c["cutfull_uncut"] = co["cutfull_uncut"] = True  # row(): G455 compares CUT catches, so it is not applicable here
    allowed = sorted({(x.get("vol", "conv"), x["path"], x["sha256"]) for x in _walk_entries(applied)})
    c["full_allowed"] = co["full_allowed"] = allowed
    return c, co, applied, allowed


def _walk_entries(applied):
    """Every uncut score-file entry {path, sha256} in the applied list (the values under "uncut" / "simh_uncut"; nested gate or manifest
    pointers are not score files and are not collected)."""
    out = []
    for x in applied:
        for k in ("uncut", "simh_uncut"):
            v = x.get(k)
            for e in (v if isinstance(v, list) else [v] if isinstance(v, dict) else []):
                if not (isinstance(e, dict) and isinstance(e.get("path"), str) and isinstance(e.get("sha256"), str)):
                    raise SystemExit(f"applied entry without path/sha256: {x.get('set')} {x.get('label') or x.get('arm') or x.get('simh')}")
                out.append(e)
    return out


def read_guard(read, allowed):
    """read_npz wrapper for full-length rows: a (volume, path, sha256) outside the pinned uncut entries is never read."""
    ok = {tuple(x) for x in allowed}

    def guarded(vol, path, sha):
        if (vol, path, sha) not in ok:
            raise SystemExit(f"full-length: read of a file outside INPUTS_FULL's uncut entries ({vol}:{path})")
        return read(vol, path, sha)

    return guarded


def _pins_in(x, pre=""):
    """Every {path, sha256} inside a probe's S_old block, with its key path ("" for the block itself)."""
    if isinstance(x, dict):
        if isinstance(x.get("path"), str) and isinstance(x.get("sha256"), str):
            return [{"part": pre or "all", "path": x["path"], "sha256": x["sha256"], "vol": x.get("volume") or x.get("vol")}]
        return [e for k, v in sorted(x.items()) for e in _pins_in(v, f"{pre}/{k}" if pre else k)]
    return []


def cut_map(cfg, cfg_old):
    """The 32k all-tasks job's registered cut files per set and readout, keyed like the CUT fork's overrides (its G2/G3 inside-32k check reads
    this): arms[folder][file], simh["<root>|<file>"] (S_old A25: "<batch>|<file>"); S_old's real line is ONE combined file per probe (keys
    world|cell over every T196 world), listed under old[file] as parts; its relaunch under relaunch[file]; 25f honest-A roots under ha25f."""
    ro_of = {}
    for p in cfg["probes"]:
        if ro_of.setdefault(p["new"], p["readout"]) != p["readout"]:
            raise SystemExit(f"cut map: {p['new']} used by two readouts")
    out = {}

    def put(st, fn, blk, key, e):
        d = out.setdefault(st, {}).setdefault(ro_of[fn], {}).setdefault(blk, {})
        if key in d and d[key] != e:
            raise SystemExit(f"cut map: {st} {blk} {key} twice with different files")
        d[key] = e

    for st, arms, simh in (("S_new", cfg.get("arms"), cfg.get("simh_files")), ("S_newmains", cfg.get("arms_nm"), cfg.get("simh_nm")),
                           ("S_paper493", (cfg.get("sp493") or {}).get("arms"), (cfg.get("sp493") or {}).get("simh"))):
        for a in arms or []:
            for fn, e in a["files"].items():
                if fn in ro_of:
                    out.setdefault(st, {}).setdefault(ro_of[fn], {}).setdefault("arms", {}).setdefault(a["arm"], {})[fn] = {"path": e["path"], "sha256": e["sha256"]}
        for fn, ents in (simh or {}).items():
            for e in ents if fn in ro_of else []:
                put(st, fn, "simh", f"{e['root']}|{fn}", {"path": e["path"], "sha256": e["sha256"], "batch": e.get("batch")})
    for p in cfg["probes"]:
        fn = p["new"]
        k_ = str(p["old"].get("kind", ""))  # the Volume each S_old reader uses (load_probe / load_own_all); DT's reader is run_common710's
        dv_ = "p2" if k_.startswith("own_all") or k_ == "fam" else "solavg" if k_.startswith("own_a_") else "conv" if k_ == "uncut_list" else None
        put("S_old", fn, "old", fn, [dict(e, vol=e["vol"] or dv_) for e in _pins_in(p["old"])])
        if p.get("relaunch"):
            e = cfg["relaunch_files"][p["relaunch"]]
            put("S_old", fn, "relaunch", fn, {"path": e["path"], "sha256": e["sha256"]})
        for e in (cfg_old.get("simh_old") or {}).get(p["label"], []):
            put("S_old", fn, "simh", f"{e.get('root') or e['batch']}|{fn}", {"path": e["path"], "sha256": e["sha256"], "batch": e["batch"]})
    for hr in (cfg.get("ha25f") or {}).get("roots", []):
        for fn, e in hr["files"].items():
            if fn in ro_of:
                put("S_new", fn, "ha25f", f"{hr['root']}|{fn}", {"path": e["path"], "sha256": e["sha256"]})
    return out


SUB_KEYS = ("probes", "arms", "simh_files", "arms_nm", "simh_nm", "sp493", "relaunch_files", "ha25f", "e12", "il_arms", "full_uncut", "cutfull_uncut")


def addendum_jobs(jobs, pair, allowed, name):
    """The LOO addendum job (runs after the frozen full-length Stage A): only the 32k LOO label's jobs, on the addendum's uncut files, renamed."""
    return [(lb, v | {k: pair[0][k] for k in SUB_KEYS if k in pair[0]} | ({"simh_old": simh_old_for(v, pair[1]["simh_old"])} if "simh_old" in v else {})
             | {"full_allowed": allowed, "full_as": out_label(lb, name)}, b) for lb, v, b in jobs if lb.rsplit(" ", 1)[0] == ADDENDUM_OF[name]]


def full_jobs(jobs, reg, retrains, allowed):
    """The 32k job list -> the full-length one. reg = (cfg_u, cfg_old_u) for the registered-file labels; retrains {name: (cfg_u, cfg_old_u)}.
    Every job keeps its variant keys (origin, merge, cohort, no_s4, ...) and takes the substituted blocks; a retrain job runs under its
    counterpart's label (its row rules) and is renamed by run_rows (full_as). Labels outside the scope are dropped."""
    def sub(v, pair):
        c, co = pair
        x = {k: c[k] for k in SUB_KEYS if k in c}
        if "simh_old" in v:
            x["simh_old"] = simh_old_for(v, co["simh_old"])
        return v | x | {"full_allowed": allowed}

    out = []
    for lb, v, b in jobs:
        base = lb.rsplit(" ", 1)[0]
        if base in SCOPE:
            out.append((lb, sub(v, reg) | {"full_as": out_label(lb)}, b))
        for name, pair in retrains.items():
            if RETRAIN_OF[name] == base:
                out.append((lb, sub(v, pair) | {"full_as": out_label(lb, name)}, b))
    return out
