"""Noise-matched gap vs the two-real-pipeline yardstick on the POOLED 455 (RULE.md Amendment 1). POST HOC; PRELIMINARY until reviewed.

Input: the 455 fork's pooled per-draw npz (run_peraction311.py @ 1181ef38f/e4e250428, pool_reduce), keys
<label>__joint__<f>__<origin>__<catch> and <label>__joint__<f>__<gap> ('|' -> '__', '@' -> '_at_', ' ' -> '_', parentheses dropped).
Per origin (S_old, S_new): G = 0.5|sim@S|primary - run1@B| + 0.5|sim@S|primary - run2@B|, EXCEPT on-policy rows (own_frac*, own_all*; run 1 is a training arm):
G = |sim@S|primary - run2@B| (0d via 2e 17:5x). Y = |run2@A - run1@B2 (S_old) or run1@B (S_new)| for every row (includes training arm run 1 for on-policy rows).
Headline (option (ii), 0d via 2e 18:0x): the off-policy 5 group; groups with on-policy rows are sensitivities; on-policy rows are shown separately (sim - run 2 only).
Pooled: each catch combined with equal-main weighting, X = (X_old D_old + X_new D_new) / (D_old + D_new) per draw; gate: the pooled gaps
recomputed here equal the npz's saved pooled arrays exactly, and draw 0 of every per-origin catch and pooled gap equals the fork's record.
Analysis (group means, paired difference, readout contrast, summ intervals) as run_panel.py.

  PYTHONPATH=src modal run experiments/2026-10-15_panel_noise_matched/run_pooled.py --cfg pooled455
"""

import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-panelnm_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
LABEL = "POST HOC (judge-panel request, not registered); PRELIMINARY until Codex, 09 or Aswin review; from raw data, by script"
ON = ["own_frac10_L48", "own_frac25_L48", "own_frac100_L48", "own_a"]
OFF = ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes"]
GROUPS = {"on_policy_own_family": ON, "off_policy": OFF, "overall": ON + OFF}
# extra groups: in the record (groups, readout contrast, captions), not in the figure; 2e 17:2x for the writing session's draft figure
EXTRA_GROUPS = {"body6": ["own_frac100_L48", "Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes"],
                # Aswin ~19:55 via 0d/2e: both groups, each pooled and per set; the headline group is a config value (09's pre-stated rule)
                "offpol5_plus_own_all": ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes", "own_all_L48"],
                "offpol5_plus_own_all_own_frac100": ["Sol (a)", "FIBS", "harmfulness", "refusal", "high-stakes", "own_all_L48", "own_frac100_L48"]}
EXTRA_LABELS = {"body6": "6-probe body group (POST HOC; matches the draft figure; own_frac25/own_frac10 in the appendix)",
                "offpol5_plus_own_all": "off-policy 5 + own_all_L48 (6 probes requested: Sol (a), FIBS, harmfulness, refusal, high-stakes, own_all_L48)",
                "offpol5_plus_own_all_own_frac100": "off-policy 5 + own_all_L48 + own_frac100_L48 (7 probes requested: Sol (a), FIBS, harmfulness, refusal, high-stakes, own_all_L48, own_frac100_L48)"}
HEADLINE_PENDING = "pending 09 rule (pre-stated)"
# 09's pre-stated rule @ 1ffffca72 (last section, sec. 2-3): launch-disjoint rows for on-policy probes from an INPUT allowlist; inclusion by availability
LD_TAG = "[launch-disjoint]"
LD_LABEL = "launch-disjoint real arms only"
LD_SHARED_LABEL = "real run 1 shares the probe's training launch"
LD_ORDER = ("own_all_L48", "own_frac100_L48")  # sec. 3 (a), then (b); else (c) none
SUBSET_TAG = "[on-policy-eligible subset]"
SUBSET_LABEL = "on-policy-eligible subset (comparability check: the off-policy 5 on the joining probe's eligible sets)"


def elig_cells(v):
    """Eligible cell keys of one set, in the census fork's v1 (eligible_cells) or v3+ (eligible_and_scored_cells) schema."""
    return list(v.get("eligible_and_scored_cells", v.get("eligible_cells")) or [])


def elig_mains(v):
    return int(v.get("eligible_mains_counted", v.get("eligible_mains", 0)) or 0)


def file_not_covered(v):
    """The eligibility file's own statement of cohort cells not covered (v3+: cohort_check.cohort_cells_not_covered, a dict or list of keys)."""
    x = (v.get("cohort_check") or {}).get("cohort_cells_not_covered") or []
    return sorted(x.keys() if isinstance(x, dict) else x)


def ld_choose(elig, cohort, origins=("S_old", "S_new"), cohort_keys=None, expected_uncovered=None, drops=None, order=LD_ORDER, required_joiner=None, sim_keys=None,
              sim_keys_meta=None, elig_alias=None, excluded_origins=None):
    """09 @ 24fd07866 amendment 1, on the census fork's eligibility file (schema of launch_disjoint_eligibility v1 @ 9268c66b4): availability from
    provenance only, before any score is read. Recomputes each probe's availability (>= 30 eligible mains over its non-withheld sets) and the GLOBAL
    choice (own_all_L48, then own_frac100_L48, then none) and requires both to equal the file's. A set is used for the joiner's launch-disjoint row
    iff it is not withheld and its eligible cells cover the PANEL cohort (checked by COUNT: the per-draw inputs carry no cell list); a partially
    eligible set would need restricted per-draw inputs, so the row is withheld (no fallback). cohort: {origin: cohort cells} from the source records."""
    out = {"rule": "09 @ 1ffffca72 sec. 2-3, amended @ 24fd07866 (eligibility file; provenance only; global choice; no fallback)", "probes": {}, "mismatches": []}
    al = lambda o: (elig_alias or {}).get(o, o)  # an origin whose eligibility is stated under another set (e.g. "S_newmains without S4" under S_newmains); its cohort keys are its own
    for pr in order:  # probes are checked in order up to the first available one (later probes need no entry: amendment 2 item 5 order)
        sets_ = elig.get("sets", {}).get(pr, {})
        tot = sum(int(elig_mains(v)) for v in sets_.values() if not v.get("withheld"))
        fa = (elig.get("availability", {}).get(pr) or (elig.get("others_unchanged") or {}).get(pr) or {})  # LOO file: other probes' availability under others_unchanged
        out["probes"][pr] = {"eligible_mains_total_recomputed": tot, "available": tot >= 30, "file": fa,
                             "sets": {o: {"withheld": v.get("withheld"), "eligible_mains": elig_mains(v), "pools": v.get("pools_failing_or_unknown", v.get("calibration_pools"))}
                                      | {"eligible_cells_n": len(elig_cells(v))}
                                      for o, v in sets_.items()}}
        if not sets_ and fa and pr in (elig.get("others_unchanged") or {}):  # stated in the file from its pinned source (e.g. v5); not recomputable here
            tot = int(fa.get("eligible_mains_total", -1))
            out["probes"][pr] |= {"eligible_mains_total_recomputed": None, "available": bool(fa.get("available")),
                                  "taken_from": f"others_unchanged (source {json.dumps((elig.get('others_unchanged') or {}).get('source'))})"}
        elif not fa or bool(fa.get("available")) != (tot >= 30) or int(fa.get("eligible_mains_total", -1)) != tot:  # fail closed (Codex v12 r1)
            out["mismatches"].append(f"{pr}: file availability {fa} != recomputed ({tot}, {tot >= 30})")
        if out["probes"][pr]["available"]:
            break
    joiner = next((pr for pr in order if pr in out["probes"] and out["probes"][pr]["available"]), None)
    out["order"] = list(order)
    fc = elig.get("global_choice")
    ao = elig.get("availability_order")
    if fc is None and isinstance(ao, str):  # the LOO file states its order instead of a choice: the EXACT sequence must equal ours + "none"; the choice is recomputed
        import re

        toks = [t_.strip() for t_ in re.split(r",\s*then\s+|,\s*|\s+then\s+", ao.split("(")[0].split(";")[0]) if t_.strip()]
        if toks != list(order) + ["none"] or len(set(toks)) != len(toks):
            out["mismatches"].append(f"file availability_order {ao!r} parses to {toks}, not {list(order) + ['none']}")
        out["global_choice_basis"] = "recomputed from the file's availability_order (the file states no global_choice)"
    elif fc is None or (fc if fc != "none" else None) != joiner:  # fail closed: the file must state its global choice
        out["mismatches"].append(f"file global_choice {fc!r} != recomputed {joiner or 'none'!r}")
    out |= {"joiner": joiner, "global_choice_file": fc, "allowed_origins": [], "partial": [], "excluded": {}, "not_in_this_panel": []}
    if required_joiner and joiner != required_joiner:  # selection decided; stop BEFORE any coverage lookup (Codex part-1 r1)
        out["required_mismatch"] = f"the pinned eligibility file makes {joiner or 'none'} the joiner, not {required_joiner}"
        return out
    if joiner:
        sets_ = elig["sets"][joiner]
        for o in origins:
            v = sets_.get(al(o), {})
            n_el = len(elig_cells(v))
            if o in (excluded_origins or {}):  # 09: excluded by a stated rule (e.g. the with-S4 set: S4 cells not in the deciding eligibility file); the gap must exist BY KEY
                row_coh = set((cohort_keys or {}).get(o) or []) - set((drops or {}).get(joiner, {}).get(o, []))
                unc = sorted(row_coh - set(elig_cells(v)))
                if not row_coh or not unc:
                    out.setdefault("stop", []).append(f"{o}: the configured exclusion {excluded_origins[o]!r} is not borne out by key (cohort keys {len(row_coh)}, not eligible {len(unc)})")
                out["excluded"][o] = f"{excluded_origins[o]} (by key: {len(unc)} of {len(row_coh)} cohort cells not eligible)"
                continue
            if not v:  # a set the eligibility file does not state is excluded, never assumed eligible
                out["excluded"][o] = "not in the eligibility file"
            elif v.get("withheld"):
                out["excluded"][o] = ("withheld: " + ", ".join(v.get("pools_failing_or_unknown") or ["see the eligibility file"])) if (v.get("eligible_mains_before_pool_check", 0) or n_el) else "no eligible cell"
            elif (cohort_keys or {}).get(o) is not None:  # 09: BY KEY; cohort within eligible is a REPORTED coverage fact (STOP only if the uncovered set is unexpected)
                row_coh = set(cohort_keys[o]) - set((drops or {}).get(joiner, {}).get(o, []))  # the joiner row's own cohort (minus its fold -1 drops)
                unc = sorted(row_coh - set(elig_cells(v)))
                n_c = len(row_coh)
                src_ = ("membership710, sha-matched to the INPUTS_FINAL pin" if o.startswith("S_newmains") else
                        "Stage A's cohort from FINAL and T196, each sha-matched to Stage A's shipped inputs")
                if not unc and not file_not_covered(v):
                    out["allowed_origins"].append(o)
                    out.setdefault("coverage_by_set", {})[o] = f"by key ({src_}): all {n_c} cohort cells eligible"
                else:
                    exp = sorted(expected_uncovered[o]) if (expected_uncovered or {}).get(o) is not None else sorted(file_not_covered(v))
                    if unc != exp:
                        out.setdefault("stop", []).append(f"{o}: uncovered cohort cells {unc} != the eligibility file's cohort_cells_not_covered {exp}")
                    out["allowed_origins"].append(o)
                    out.setdefault("restricted", {})[o] = unc
                    out.setdefault("coverage_by_set", {})[o] = (f"by key ({src_}): covers {n_c - len(unc)} of {n_c} cohort cells; uncovered (listed): {unc}; "
                                                                "the row and the like-for-like subset are restricted to the covered cells")
            elif n_el >= cohort.get(o, 10**9):
                out["allowed_origins"].append(o)
                out.setdefault("coverage_by_set", {})[o] = "coverage checked by count only (INPUTS_FINAL has no cell-level cohort list for this set)"
            elif n_el > 0:
                out["partial"].append(o)
            else:
                out["excluded"][o] = "no eligible cell"
        out["not_in_this_panel"] = sorted(set(sets_) - {al(o) for o in origins})
        # 09 @ b0a3e0df9: the SIM arm of every eligible cell must be launch-disjoint from the scoring model's training launches; a failing set is withheld,
        # and missing provenance stops (fail closed)
        # BY KEY only (09): each set's sim launch set is the UNION over every eligible cell's sim run key (launch = the key's first path component), from a
        # per-key list derived from the manifests (the eligibility file's own per_key_provenance, else sim_keys from the census fork's file); a set-level list
        # stated in the file must EQUAL that union; no by-key source -> STOP
        for o in list(out["allowed_origins"]):
            tr = ((elig.get("amendment_2_item_4_asserts") or {}).get(al(o)) or {}).get("training_launch_ids") or (elig.get("probes", {}).get(joiner) or {}).get("training_launch_ids")
            el_ = set(elig_cells(sets_[al(o)]))
            src_list = (elig.get("per_key_provenance") or {}).get(f"{al(o)}|sim")
            src_name = "the eligibility file's per_key_provenance"
            if src_list is None and (sim_keys or {}).get(o) is not None:
                src_list, src_name = sim_keys[o], (sim_keys_meta or {}).get("label", "sim_keys")
            if src_list is None or not tr:
                out.setdefault("stop", []).append(f"{o}: cannot derive the sim arm's launches BY KEY (per-key sim list {src_list is not None}, training launches {bool(tr)})")
                continue
            ent = [e_ for e_ in src_list if e_.get("cell") in el_]
            bad_l = [e_["key"] for e_ in ent if not e_.get("launch") or not e_["key"].split("/")[0] or e_.get("launch") != e_["key"].split("/")[0]]
            multi = len(ent) != len({e_["cell"] for e_ in ent})  # several sim runs for a cell (replacements, e.g. S_newmains s5r): only where the file states runs = entries
            dup = len(ent) != len({e_["key"] for e_ in ent}) or (multi and int(((sets_[al(o)].get("arms") or {}).get("sim") or {}).get("runs", -1)) != len(ent))
            if bad_l or dup:
                out.setdefault("stop", []).append(f"{o}: per-key sim entries with an empty or inconsistent launch {bad_l[:5]}, or duplicate keys/cells ({dup})")
                continue
            union = sorted({e_["key"].split("/")[0] for e_ in ent})
            arm_ = ((sets_[al(o)].get("arms") or {}).get("sim") or sets_[al(o)].get("sim_arm") or {})
            stated = sorted(arm_.get("launch_ids") or [])
            # completeness (Codex part-1 r2): every PANEL cohort cell of this set (minus the joiner row's own drops) must have a per-key sim entry, and the
            # number of per-key entries over eligible cells must equal the file's stated sim-arm count; else STOP
            with_key = {e_["cell"] for e_ in ent}
            coh_o = set((cohort_keys or {}).get(o) or []) - set((drops or {}).get(joiner, {}).get(o, []))
            miss = sorted(coh_o - with_key) if coh_o else None
            n_st = arm_.get("cells", arm_.get("runs"))
            if miss is None or miss or not union or n_st is None or not stated or int(n_st) != len({e_["cell"] for e_ in ent}):  # all required (Codex part-1 r3)
                out.setdefault("stop", []).append(f"{o}: per-key sim provenance incomplete (cohort cells without a sim key: {miss if miss is not None else 'no cohort keys'}; "
                                                  f"entries {len(ent)} vs the file's stated {n_st}; launches {union})")
                continue
            if stated != union:
                out.setdefault("stop", []).append(f"{o}: the file's set-level sim launch ids {stated} != the by-key union {union}")
                continue
            shared = sorted(set(union) & set(tr))
            out.setdefault("sim_arm_check", {})[o] = {"basis": f"by key: union over {len(ent)} sim run keys of {len(el_)} eligible cells ({len(el_ - with_key)} eligible cells have no sim key; "
                                                                f"every one of the {len(coh_o)} PANEL cohort cells has one), from {src_name}; equals the file's set-level list: {bool(stated)}",
                                                       "sim_launches": union, "training_launches_n": len(tr), "shared": shared, "disjoint": not shared}
            if shared:
                out["allowed_origins"].remove(o)
                out["excluded"][o] = f"withheld: the sim arm shares training launches {shared} (09 @ b0a3e0df9)"
        out["coverage_check"] = "per set: by key where the frozen file lists the cohort cells, else 'coverage checked by count only' (09)"
    return out


POOLING_CAVEAT = "pooling lets the simulator's per-batch errors cancel, while the rerun yardstick keeps a consistent run1 > run2 launch difference"
# on-policy rows (0d via 2e 17:5x): run 1 is a training arm, so G drops the |sim - run 1| term; Y is unchanged and captioned as including run 1
ONPOLICY = ("own_frac", "own_all")
G_DEF = {"onpolicy": "on-policy rows: G = |sim - run 2| (run 1 is a training arm); Y includes training arm run 1", "default": "G = 0.5|sim - run 1| + 0.5|sim - run 2|"}


# option (ii), 0d via 2e 18:0x: the headline noise-matched group EXCLUDES on-policy rows; on-policy rows are shown separately (sim - run 2 only, no excess);
# groups that include on-policy rows stay as sensitivities
HEADLINE = "off_policy"
HEADLINE_NAME = "off-policy 5 (Sol (a), FIBS, harmfulness, refusal, high-stakes)"
SENS_LABEL = "sensitivity: includes on-policy rows whose yardstick includes the training arm"
ONPOL_CAPTION = "no run1-free real-vs-real pair exists; yardstick includes the training arm (run 1)"
# 09's narrowing @ 635b14564 (via 2e, 19:3x): the separate on-policy rows LEAD with the registered G; the run2-only G sits beside as a sensitivity
ONPOL_R2_LABEL = "sensitivity: run1 shares the training launch"


# noise-subtracted bias (POST HOC; the writing session's judge panel via 0d, 18:2x): per row and for the headline group
BIAS_LABEL = ("POST HOC (judge-panel add-on via 0d): signed mean of sim - real (sim - run 2 for on-policy rows; sim - mean(run 1, run 2) otherwise; companions: the registered "
              "same-launch comparator sim - run 1 (sim - run 2 on-policy) and sim - run 2 for every row) and a noise-subtracted bias "
              "b = sqrt(max(0, mean G2 - mean Y2)), G2 = mean squared sim - real gap over the real comparators of the row's G (0.5 (sim - run 1)^2 + 0.5 (sim - run 2)^2; "
              "(sim - run 2)^2 for on-policy rows only when their G is run2-only), "
              "Y2 = (run 2 - run 1 yardstick)^2; Jeff's RMS scenario A logic. ASSUMES sim - real and run 2 - run 1 carry the same noise variance; checked per row with the across-draw SDs "
              "(task resampling, thresholds re-estimated: sampling noise only, not launch noise); the yardstick's run 2 launch offset is treated as noise. The rescaled version "
              "multiplies each row's Y2 by (SD sim - real / SD run 2 - run 1)^2 (labelled sensitivity). b is an ESTIMATE: where its interval includes 0 there is no established "
              "positive bias.")
BIAS_LABEL_RM = BIAS_LABEL.replace("sim - run 2 for on-policy rows; sim - mean(run 1, run 2) otherwise",
                                   "sim - mean(run 1, run 2) for EVERY row, on-policy rows included (09 Final-build checklist item 2); sim - run 2 beside as the labelled sensitivity")
BIAS_GROUPS = ("off_policy", "offpol5_plus_own_all", "offpol5_plus_own_all_own_frac100", "body6", "overall")  # headline first; the other two for comparison with the panel's body6 context


OFF_HEADLINE = list(OFF)
# every group name a 32k build uses (this script's tables, Stage A's panel_groups, the co-headline, FINAL_TEMPLATE); a full-length record must hold none non-empty
NAMES_32K = frozenset(GROUPS) | frozenset(EXTRA_GROUPS) | frozenset(BIAS_GROUPS) | {"headline_six", "registered_scored_six", "offpol5_plus_own_all_rs_L48", "appendix_B_own_all_rs"}


COUNT_KEYS = ("n", "n_members", "n_rows", "draws", "finite_draws")
DESC_KEYS = ("label", "note", "caveat", "caption", "name", "rule", "source", "definition", "text", "readout", "unit", "units", "status", "reason")
FLAG_KEYS = ("saturated", "degenerate", "readable", "available")


def empty_payload(x, k=None):
    """True only for a payload that carries nothing (Codex final r25/r26): membership fields (rows, unreadable_rows, members, cells, *_rows) are EMPTY LISTS;
    counts are a numeric 0; statistics None or NaN; strings only under descriptive keys, booleans only under flag keys; any other type or value is content."""
    if k in ("rows", "unreadable_rows", "members", "cells") or (isinstance(k, str) and k.endswith("_rows")):
        return isinstance(x, list) and not x
    if k in COUNT_KEYS:
        return x is None or (isinstance(x, (int, float)) and not isinstance(x, bool) and x == 0)
    if x is None:
        return True
    if isinstance(x, bool):
        return k in FLAG_KEYS
    if isinstance(x, str):
        return k in DESC_KEYS
    if isinstance(x, float):
        return math.isnan(x)
    if isinstance(x, list):
        return not x
    if isinstance(x, dict):
        return all(empty_payload(v, kk) for kk, v in x.items())
    return False


def empty_group(g):
    """A group entry that holds nothing, panel_groups' {"primary": {"rows": [], "n": 0, ...}, ...} or a co-headline entry; a malformed primary is not empty."""
    if g is None:
        return True
    if not isinstance(g, dict) or ("primary" in g and not isinstance(g["primary"], dict)):
        return False
    return empty_payload(g)


def drop_origin(cfg, o):
    """The base origin a row's fold -1 drops are listed under (DROPS @ 93f8b7a86 lists only the base origins): o's origin_base, then, if that is a restriction
    such as "S_newmains without S4", the base origin it restricts ("S_newmains"; its may-drop superset covers any subset of its cells) (Codex final r24)."""
    b = (cfg.get("origin_base") or {}).get(o, o)
    bo = (cfg.get("drop_evidence_from_record") or {}).get("base_origins") or []
    m = [x for x in bo if b == x or b.startswith(x + " ")]
    return max(m, key=len) if m else b


def nonempty_32k(tree, where):
    """Paths of non-empty 32k-named group entries anywhere in a nested record block (group keys are '<name>|<readout>')."""
    out = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            if isinstance(k, str) and "|" in k:
                if k.split("|")[0] in NAMES_32K and not empty_group(v):
                    out.append(f"{where}/{k}")
            else:
                out += nonempty_32k(v, f"{where}/{k}")
    return out


def apply_family_config(cfg):
    """The off-policy headline members from the config (e.g. the full-length build: sol_a_uncut in place of Sol (a), the registered weights scored uncut beside);
    default: the off-policy 5. Mutates the module's group tables inside the Modal container (one run per container call)."""
    # 507-prep (via 2e): NO aliasing - "off_policy" keeps the registered members (empty in a full-length record); the full-length off-policy five are the
    # config group off_policy_five_full; only the co-headline's membership check reads off_policy_probes
    if cfg.get("off_policy_probes"):
        OFF_HEADLINE[:] = list(cfg["off_policy_probes"])


def onpol(lab):
    return lab.rsplit(" ", 1)[0].startswith(ONPOLICY)
STRESS = ["c_asis"]
SETS = ("pooled", "S_old", "S_new", "pooled_abs_per_origin")  # the last: SENSITIVITY (2e GO 14:3x), |.| per origin, then D-weighted
ORIGINS = ["S_old", "S_new"]  # default origins (the pooled 455); a config may name N origins and several combos (pooled sets over subsets of them)


def set_names(cfg):
    """Result sets: the headline combo as "pooled" with its per-set sensitivity "pooled_abs_per_origin", each origin, then every other combo as "<combo>"
    and "<combo>|per_set". Without combos (the pooled 455 inputs) this is SETS."""
    o = cfg.get("origins", ORIGINS)
    return ("pooled", *o, "pooled_abs_per_origin") + tuple(x for cb in (cfg.get("combos") or {}) if cb != cfg.get("headline_combo") for x in (cb, f"{cb}|per_set"))


def combo_sets(cfg, cb):
    return ("pooled", "pooled_abs_per_origin") if cb is None or cb == cfg.get("headline_combo") else (cb, f"{cb}|per_set")


def headline_members(cfg):
    return list((cfg.get("combos") or {}).get(cfg.get("headline_combo")) or cfg.get("origins", ORIGINS))


def jn(xs):
    xs = list(xs)
    return " and ".join(xs) if len(xs) <= 2 else ", ".join(xs[:-1]) + " and " + xs[-1]
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6").add_local_python_source("probe_shift")
SOL = modal.Volume.from_name(VOL, version=2)


def san(k):
    return k.replace("|", "__").replace(" ", "_").replace("(", "").replace(")", "").replace("@", "_at_")


def split(lab):
    return lab.rsplit(" ", 1)


def jget(d, path, **kw):
    for p in path:
        d = d[p.format(**kw)] if isinstance(p, str) else d[p]
    return d


def pc(s):
    return {**s, "point": 100 * s["point"], **({"ci95": [100 * s["ci95"][0], 100 * s["ci95"][1]]} if s.get("ci95") else {})}


def stats(rows, members, sat_ok=True):
    import numpy as np

    from probe_shift.fpr_core import summ

    use = [lb for lb in members if rows[lb]["readable"] and (sat_ok or not rows[lb]["saturated"])]
    if not use:
        return {"rows": [], "unavailable": "no readable rows"}
    G = np.mean([rows[lb]["G"] for lb in use], 0)
    Y = np.mean([rows[lb]["Y"] for lb in use], 0)
    return {"rows": use, "n": len(use), "mean_G": pc(summ(G)), "mean_Y": pc(summ(Y)), "mean_G_minus_mean_Y": pc(summ(G - Y)), "_G": G, "_Y": Y}


def dg_(d, *path):
    """Null-safe nested get: None as soon as a level is missing or not a dict."""
    for p_ in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p_)
    return d


def gates_only_mode(cfg):
    """A pooled PILOT or STUB dry run: its return value, Volume copy and local record carry gates only (no statistic, figure or caption)."""
    pw_ = cfg.get("pooled_world") or {}
    return pw_.get("pilot") is True or pw_.get("stub") is True


def delta_full_tol(cfg):
    """The per-draw Delta_full gate's tolerance: the co-headline's (as before; a missing value raises), else a pilot's own stated one, else no such gate."""
    if cfg.get("coheadline"):
        return float(cfg["coheadline"]["delta_full_tol_abs"])
    if (cfg.get("pooled_world") or {}).get("pilot") is True:
        return float(cfg["pooled_world"]["pilot_delta_full_tol_abs"])
    return None


def incomplete(rows, fams, ros):
    """The declared members '<family> <readout>' that are missing or unreadable here (a primary aggregate needs all of them)."""
    return [f"{m} {x}" for m in fams for x in ros if f"{m} {x}" not in rows or not rows[f"{m} {x}"]["readable"]]


def withheld_why(inc_, n_decl):
    return (f"withheld: incomplete membership ({n_decl - len(inc_)} of {n_decl} declared members present and readable; missing or unreadable {inc_}); "
            "a primary aggregate is never averaged over a subset (2e / 0d 10-16)")


def readout_contrast(rows, fams, sat_ok=True):
    from probe_shift.fpr_core import summ

    ok = [f for f in fams if all(f"{f} {ro}" in rows and rows[f"{f} {ro}"]["readable"] and (sat_ok or not rows[f"{f} {ro}"]["saturated"]) for ro in "RC")]
    if not ok:
        return {"families": [], "unavailable": "no family readable at both readouts"}
    inc_ = incomplete(rows, fams, "RC")
    if inc_:  # complete membership (2e / 0d 10-16): every declared family readable at both readouts, else withheld (saturation-dropping stays the labelled sensitivity)
        return {"families": [], "unavailable": withheld_why(inc_, 2 * len(fams))}
    sR, sC = stats(rows, [f"{f} R" for f in ok]), stats(rows, [f"{f} C" for f in ok])
    return {"families": ok, "n_families": len(ok), "reply_excess_minus_command_excess": pc(summ((sR["_G"] - sR["_Y"]) - (sC["_G"] - sC["_Y"])))}


def strip(o):
    return {k: v for k, v in o.items() if not k.startswith("_")}


CELLS_DESCRIPTOR = ("cohort_cell_keys (sorted); one character per cell, '1' = caught at draw 0, '0' = not caught, '-' = no admitted run for that arm (extra arms "
                    "only)")  # 507-prep's compact cell_hits_draw0.cells (32k r2 record, 10-16)


STREAM_LABEL = ("per-origin weighting and draws; point difference = weighting effect (09 B4: a DIFFERENT estimand - a main in two origins counted in each; registered "
                "per-origin streams, superseded for the post); pooled row read from Stage A, not independently recomputed, gated at draw 0")
UNION_LABEL = "lead: union stream (one draw per main; a main in several origins counts once)"


I36_ST5 = ("POST HOC derived (09 RULE item 36 (3)): St5 = the signed S averaged over the five probes other than the deception probe (FIBS), tool-output readout (R), "
           "same draws, 95% joint interval")
I36_DT = "POST HOC derived (09 RULE item 36 (3)): Dt = the deception probe's (FIBS) own signed Delta_full, tool-output readout (R), same draws, 95% joint interval"
I36_NAMES = {"rule": "09 RULE item 36 @ 42609cbf1",
             "S4 / Y4 / E4": "coheadline[<f>][<C|R>]['auroc_subset'] -> S (selected comparator), Y (the signed yardstick run2 - run1 at tau_B), excess ((A)): the four-probe AUROC subset "
                             "fixed @ 19ba8f563, mapped to the full-length probes",
             "Sfilt / Efilt": "coheadline['other_combos']['pooled_all_origins_registered_cohort'][<f>][<C|R>]['six'] -> S and excess (registered cohort; S_paper493 WITHHELD: no attempt labels)",
             "N / K / unlabelled": "record['item36_attempt_filter'] (by pinned key)",
             "St5 / Dt": "coheadline[<f>]['R']['St5'] (S) and ['Dt'] (delta_full)"}


def sat_keys(cfg, c0):
    """The draw-0 catch arms the saturation flag reads: all of them, or under pooled calibration those G and Y use (sim, run1, run2, y1; not run2@A;
    the producer's Astra r1 #9), named in the config."""
    ks_ = (cfg.get("pooled_world") or {}).get("saturation_arms") or list(c0)
    if not set(ks_) <= set(c0):
        raise ValueError(f"saturation arms {ks_} not in the catch block {sorted(c0)}")
    return ks_


def readable_of(cfg, d):
    """A row's readability from its per-pool flags: every flag, or under pooled calibration RULE 37 (the pools the product thresholds on: S and B), named in
    the config; a missing pool flag fails closed."""
    pools_ = (cfg.get("pooled_world") or {}).get("readable_pools")
    return bool(all(d.get(p_) is True for p_ in pools_)) if pools_ else bool(all(d.values()))


PS_CMP_LABEL = "pooled vs registered per-set (item 43)"
NPZ_COLLISION_REASON = "npz key collision (c_asis|uncut → c_asis_uncut_X); record JSON unaffected"  # 0d verbatim (10-16): Stage A's npzkey maps '|' and ' ' to '_'
PILOT_ABSENT = ("coheadline", "panel_groups_gate", "labels_pin")  # a pilot (2e 10-16) runs without the six-probe, Stage A group and full-scope checks (its groups
# stay configured: with two probes they are WITHHELD as incomplete, which the figure and title already handle)
NPZ_COLLISION_APPROVED = frozenset({"c_asis|uncut C", "c_asis|uncut R"})  # 0d / 2e 10-16 (diag_draw0 @ a4a2c958f: the keys hold c_asis_uncut's draws; its partner is exact)
PW_EXTRA_FAMILIES = ("c_asis_uncut", "own_all_rs25_uncut_L48", "own_all_rs10_uncut_L48")  # 09 item 45(3): the PW scope = HEADLINE_SIX + these three


def per_set_compare(cfg, rec):
    """09 #8 (via 2e): each headline quantity of the pooled record beside the registered per-set record's (the RULE 43 source the pooled record names):
    the six-probe S, the co-headline Y, the excess G - |Y| with Y = run2 - run1 at each record's tau_B in BOTH records (like for like; gap_delta_rr is never read) and
    the per-probe Delta_full (A) at C (from the saved per-draw gaps), plus the per-probe C catch arms (per draw, rebuilt from each record's member catches
    and D as Stage A combines them, gated at draw 0 against the record's own catch). Per quantity: the half-width ratio pooled / per-set, a sign flip, an
    'interval includes 0' change, and the paired difference pooled - per-set with its 95% interval on the COMMON valid draws when both records state
    complete, equal union stream identities (meta.merge mains in order, env_of_main for every main, seed), else 'unpaired'. Returns (block, gates)."""
    import numpy as np

    from probe_shift.fpr_core import summ

    pw_, gates = cfg["pooled_world"], []
    gk_ = lambda ok, **kw: gates.append(kw | {"pass": bool(ok)})
    ps_ = ((rec.get("meta") or {}).get("pooled_world") or {}).get("per_set_record") or {}
    if not (ps_.get("path") and ps_.get("sha256")) or ((rec.get("per_set_beside") or {}).get("source") != ps_):
        gk_(False, check="per-set comparison: the pooled record names its per-set record (meta.pooled_world.per_set_record = per_set_beside.source)")
        return None, gates
    pb_ = (Path("/sol") / ps_["path"]).read_bytes()
    gk_(hashlib.sha256(pb_).hexdigest() == ps_["sha256"], check="per-set comparison: the per-set record's sha256")
    psr = json.loads(pb_)
    arrs = {}
    for nm_, r_ in (("pooled", rec), ("per_set", psr)):
        np_ = Path("/sol") / r_["npz"]["path"]
        gk_(hashlib.sha256(np_.read_bytes()).hexdigest() == r_["npz"]["sha256"], check=f"per-set comparison: the {nm_} npz sha256")
        arrs[nm_] = np.load(np_)
    # the draw streams are draws_main(mains in order, env_of_main, seed) (Stage A @ e9b78de6d:2679): pair only when BOTH identities are complete (Astra r1 #2; Codex r3 #4:
    # unique non-empty mains, an environment for every main and no other, a seed that is an int or a non-empty list of ints, e.g. [20261015, 40]) and the three fields
    # are equal; other meta.merge fields (rule text, registered_combos) do not define the draws. Anything less is 'unpaired'.
    mgs_ = [(r_.get("meta") or {}).get("merge") for r_ in (rec, psr)]
    ok_id = lambda m_: (isinstance(m_, dict) and isinstance(m_.get("mains"), list) and bool(m_["mains"]) and all(isinstance(x_, str) and x_ for x_ in m_["mains"])
                        and len(set(m_["mains"])) == len(m_["mains"]) and isinstance(m_.get("env_of_main"), dict) and set(m_["env_of_main"]) == set(m_["mains"])
                        and all(isinstance(v_, str) and v_ for v_ in m_["env_of_main"].values())
                        and all(isinstance(s_, int) and not isinstance(s_, bool) for s_ in (m_.get("seed") if isinstance(m_.get("seed"), list) and m_["seed"] else [m_.get("seed")])))
    sid_ = lambda m_: json.dumps([m_["mains"], m_["env_of_main"], m_["seed"]], sort_keys=True)
    paired = all(ok_id(m_) for m_ in mgs_) and sid_(mgs_[0]) == sid_(mgs_[1])
    hw = lambda z_: (z_["ci95"][1] - z_["ci95"][0]) / 2 if z_ and z_.get("ci95") else None
    inc0 = lambda z_: (z_["ci95"][0] <= 0 <= z_["ci95"][1]) if z_ and z_.get("ci95") else None
    pc_ = lambda s_: {**s_, "point": 100 * s_["point"], **({"ci95": [100 * x for x in s_["ci95"]]} if s_.get("ci95") else {})}

    def cmp_(nm, a_p, a_q):
        sp, sq = pc_(summ(a_p)), pc_(summ(a_q))
        e_ = {"quantity": nm, "pooled": sp, "per_set": sq, "half_width_ratio_pooled_over_per_set": (hw(sp) / hw(sq)) if hw(sp) is not None and hw(sq) else None,
              "sign_flip": bool(sp["point"] * sq["point"] < 0), "includes_0": {"pooled": inc0(sp), "per_set": inc0(sq)},
              "includes_0_changes": None if inc0(sp) is None or inc0(sq) is None else inc0(sp) != inc0(sq)}
        e_["paired_difference_pooled_minus_per_set"] = pc_(summ(a_p - a_q)) if paired and a_p.shape == a_q.shape else "unpaired"  # NaN in either: dropped (common valid draws)
        return e_

    ARM_KEY = {"sim": "sim@S|primary", "run1": "run1@B", "run2": "run2@B"}

    def catch_arr(r_, z_, lb, cb, f, arm):  # a union row's per-draw catch as Stage A combines it: sum_o X_o D_o / sum_o D_o over its members (D = 0 adds nothing)
        md_ = cfg.get("mode", "joint")
        mem_ = (((r_.get("pooled") or {}).get(lb) or {}).get(cb) or {}).get("members") or []
        ks_ = [(keys_for(fmt, lb, f, o=o, catch=ARM_KEY[arm], combo=cb, mode=md_), keys_for(fmt, lb, f, o=o, catch="D", combo=cb, mode=md_)) for o in mem_]
        if not mem_ or any(a_ not in z_.files or d_ not in z_.files for a_, d_ in ks_):
            return None
        Dd = sum(np.asarray(z_[d_], float) for _, d_ in ks_)
        with np.errstate(invalid="ignore", divide="ignore"):
            return sum(np.where(np.asarray(z_[d_], float) > 0, np.asarray(z_[a_], float) * np.asarray(z_[d_], float), 0.0) for a_, d_ in ks_) / Dd

    out = {"label": PS_CMP_LABEL, "per_set_record": ps_, "paired": paired, "paired_basis": "both records' complete union stream identities (meta.merge mains in order, env_of_main for every main, seed) are equal"
           if paired else "unpaired: the records' draw streams differ (meta.merge)",
           "y_definition": ("BOTH records: Y = run2@B - run1@B at the record's own tau_B, rebuilt per draw from the record's members' REAL arms and D (its own NaN mask; "
                            f"equal to sim_minus_run1 - sim_minus_run2 wherever the sim arm is defined, gated against the pooled record's saved {pw_['y_gap']}); the excess = "
                            "G - |Y| in both, like for like. gap_delta_rr is never read: it changes meaning "
                            "on S_old between the builds (pooled run2@A - run1@B, per-set run2@A - run1@B2; 2e via the pooled producer's reviewer)"),
           "by_combo": {}}
    six = list(cfg["coheadline"]["probes"])
    fmt = cfg["npz"][0].get("key_format", "v2c")
    for cb in pw_.get("compare_combos") or [cfg["headline_combo"]]:
        for f, pt in zip(cfg["f_keys"], cfg["points"]):
            for sel in ("C", "R"):
                labs = [f"{p_} {sel}" for p_ in six]
                g_ = {}
                for nm_, z_, extra_ in (("pooled", arrs["pooled"], (pw_["y_gap"],)), ("per_set", arrs["per_set"], ())):
                    ks_ = {(lb, k_): keys_for(fmt, lb, f, gap=k_, combo=cb, mode=cfg.get("mode", "joint")) for lb in labs for k_ in ("delta_full", "sim_minus_run1", "sim_minus_run2") + extra_}
                    if any("delta_rr" in k_ for _, k_ in ks_) or any("delta_rr" in v_ for v_ in ks_.values()):  # 2e: never pair on gap_delta_rr (its S_old meaning differs)
                        raise ValueError("per-set comparison: gap_delta_rr must never be read")
                    miss_ = [v_ for v_ in ks_.values() if v_ not in z_.files]
                    gk_(not miss_, combo=cb, f=f, readout=sel, check=f"per-set comparison: {nm_} arrays present ({len(miss_)} missing)")
                    g_[nm_] = None if miss_ else {lk_: np.asarray(z_[v_], float) for lk_, v_ in ks_.items()}
                if g_["pooled"] is None or g_["per_set"] is None:
                    continue
                for lb in labs:  # the pooled yardstick gap is run2 - run1 at the pooled tau_B = sim_minus_run1 - sim_minus_run2 wherever the sim arm is defined (Codex pilot
                    a_ = g_["pooled"]  # r2 #2: a draw with an undefined S pool has NaN gaps but a finite yardstick; the shared row path gates the yardstick against the real arms)
                    d12_ = a_[(lb, "sim_minus_run1")] - a_[(lb, "sim_minus_run2")]
                    ok_ = np.isfinite(d12_)
                    gk_(np.allclose(a_[(lb, pw_["y_gap"])][ok_], d12_[ok_], rtol=0, atol=1e-12, equal_nan=True), combo=cb, f=f, row=lb,
                        check=f"per-set comparison: the pooled record's {pw_['y_gap']} == sim_minus_run1 - sim_minus_run2 where both are defined")
                # Codex r4 #2: each record's own readability (its pooled row's readable.all, under its own calibration rule); the six-probe quantities need ALL six
                # readable in BOTH records (never a subset mean), a per-probe quantity needs its row readable in both; otherwise the entry is withheld, with the reason
                rdo_ = {nm_: {lb: ((((((r_.get("pooled") or {}).get(lb) or {}).get(cb) or {}).get(cfg.get("mode", "joint")) or {}).get(f) or {}).get("readable") or {}).get("all") is True
                              for lb in labs} for nm_, r_ in (("pooled", rec), ("per_set", psr))}
                unr_ = {nm_: [lb for lb in labs if not d_[lb]] for nm_, d_ in rdo_.items()}
                six_names = ("six-probe S (A)", "Y (run2 - run1 at tau_B)", "excess G - |run2 - run1 at tau_B| (both records)")
                if unr_["pooled"] or unr_["per_set"]:
                    why_ = (f"withheld: incomplete readable membership (unreadable in the pooled record: {unr_['pooled']}; in the per-set record: {unr_['per_set']}; "
                            "each under its own calibration rule); never averaged over a subset")
                    ent = [{"quantity": nm, "withheld": why_} for nm in six_names]
                else:
                    q_ = []
                    for nm_, r_ in (("pooled", rec), ("per_set", psr)):
                        a_ = g_[nm_]
                        yt_ = {}  # Codex pilot r3: Y = run2 - run1 at tau_B from each record's REAL arms, rebuilt per draw from its members (never d1 - d2, never delta_rr)
                        for lb in labs:
                            r2_, r1_ = catch_arr(r_, arrs[nm_], lb, cb, f, "run2"), catch_arr(r_, arrs[nm_], lb, cb, f, "run1")
                            gk_(r2_ is not None and r1_ is not None, combo=cb, f=f, row=lb, check=f"per-set comparison: the {nm_} record's real arms rebuilt from its members")
                            yt_[lb] = (r2_ - r1_) if r2_ is not None and r1_ is not None else np.full_like(a_[(lb, "delta_full")], np.nan)
                        S_ = np.mean([a_[(lb, "delta_full")] for lb in labs], 0)
                        Y_ = np.mean([yt_[lb] for lb in labs], 0)
                        X_ = (np.mean([0.5 * np.abs(a_[(lb, "sim_minus_run1")]) + 0.5 * np.abs(a_[(lb, "sim_minus_run2")]) for lb in labs], 0)
                              - np.mean([np.abs(yt_[lb]) for lb in labs], 0))
                        q_.append((S_, Y_, X_))
                    ent = [cmp_(nm, q_[0][i_], q_[1][i_]) for i_, nm in enumerate(six_names)]
                if sel == "C":  # the per-probe C quantities, each only where its row is readable in both records
                    ok_lb = [lb for lb in labs if rdo_["pooled"][lb] and rdo_["per_set"][lb]]
                    ent += [{"quantity": f"Delta_full (A) {lb} and its catch arms", "withheld": f"row unreadable (pooled readable {rdo_['pooled'][lb]}, per-set readable "
                                                                                                 f"{rdo_['per_set'][lb]})"} for lb in labs if lb not in ok_lb]
                    ent += [cmp_(f"Delta_full (A) {lb}", g_["pooled"][(lb, "delta_full")], g_["per_set"][(lb, "delta_full")]) for lb in ok_lb]
                    for lb in ok_lb:  # the per-probe C catch arms: per draw, reconstructed from each record's member catches and D exactly as Stage A combines them,
                        for arm in ("sim", "run1", "run2"):  # gated at draw 0 against each record's own pooled catch summary
                            ca_ = {}
                            for nm_, r_ in (("pooled", rec), ("per_set", psr)):
                                a_ = catch_arr(r_, arrs[nm_], lb, cb, f, arm)
                                try:
                                    ref_ = jget(r_, ["pooled", lb, cb, cfg.get("mode", "joint"), f, "catch", arm, "point"])
                                except (KeyError, TypeError):
                                    ref_ = None
                                gk_(a_ is not None and ref_ is not None and (abs(float(a_[0]) - float(ref_)) <= 1e-12 or (math.isnan(float(a_[0])) and math.isnan(float(ref_)))), combo=cb, f=f, row=lb, arm=arm,
                                    check=f"per-set comparison: the {nm_} record's {arm} catch reconstructed per draw from its members equals its pooled catch at draw 0")
                                ca_[nm_] = a_
                            if ca_["pooled"] is not None and ca_["per_set"] is not None:
                                ent.append(cmp_(f"catch {arm} {lb}", ca_["pooled"], ca_["per_set"]))
                out["by_combo"].setdefault(cb, {}).setdefault(pt, {})[sel] = ent
    return out, gates


def read_combos(cfg):
    """Combos whose pooled rows are READ from Stage A (never pooled additively here): the main-deduplicated ones and the per-origin-stream ones."""
    return list(cfg.get("dedup_combos") or []) + list(cfg.get("stream_combos") or [])


def stream_diagnostic(cfg, cb, rec):
    """Per combo (2e / 0d): the members, the number of mains in two or more members' pinned cohorts (by the pinned cell -> main map; None if any cell lacks a
    main) and Stage A's draw-0 total weight D_draw0 per headline label and level."""
    cm_, ck_ = cfg.get("cell_main_resolved") or {}, cfg.get("cohort_keys_resolved") or {}
    M = list((cfg.get("combos") or {}).get(cb) or [])
    seen_, shared_ = {}, None
    if M and all(o in ck_ for o in M) and all(k in cm_ for o in M for k in ck_[o]):
        for o in M:
            for mn in {cm_[k] for k in ck_[o]}:
                seen_[mn] = seen_.get(mn, 0) + 1
        shared_ = sum(v > 1 for v in seen_.values())
    D_ = {}
    for f, pt in zip(cfg["f_keys"], cfg["points"]):
        for p_ in (cfg.get("coheadline") or {}).get("probes") or []:
            for sel in ("C", "R"):
                try:
                    D_.setdefault(pt, {})[f"{p_} {sel}"] = jget(rec, ["pooled", f"{p_} {sel}", cb, cfg.get("mode", "joint"), f, "D_draw0"])
                except (KeyError, TypeError):
                    D_.setdefault(pt, {})[f"{p_} {sel}"] = None
    return {"members": M, "shared_mains": shared_, "D_draw0": D_}


def stream_comparison(cfg, res, coh, hg, rec, sets):
    """Aswin 10-16 (via 2e; 09 B4): each headline quantity on BOTH streams, union (lead) and the per-origin twin (beside), per configured pair, level and
    readout: per-probe Delta_full (A), the co-headline S and Y (six, off5, AUROC subset) and the excess; the half-width ratio union / per-origin and an
    'interval includes 0' flip flag. No point assert (09 B4: withdrawn; the twin is a different estimand): the point difference is printed, labelled
    "point difference = weighting effect"."""
    hw = lambda z: (z["ci95"][1] - z["ci95"][0]) if isinstance(z, dict) and z.get("ci95") else None
    inc0 = lambda z: (z["ci95"][0] <= 0 <= z["ci95"][1]) if isinstance(z, dict) and z.get("ci95") else None

    def coh_of(cb, pt, sel):
        b_ = (coh or {}) if cb == cfg.get("headline_combo") else ((coh or {}).get("other_combos") or {}).get(cb) or {}
        return (b_.get(pt) or {}).get(sel) or {}

    out = {"union_label": UNION_LABEL, "per_origin_label": STREAM_LABEL, "point_rule": "09 B4: no point assert; the point difference = weighting effect", "pairs": []}
    INTERVAL_ONLY = "interval effect only (0 shared mains in both combos: same weighting; the check until the runner's same-weighting twin (ii) exists)"
    for pr in cfg.get("stream_pairs") or []:
        u, q = pr["union"], pr["per_origin"]
        su, sq = combo_sets(cfg, u)[0], combo_sets(cfg, q)[0]
        dg_ = {u: stream_diagnostic(cfg, u, rec), q: stream_diagnostic(cfg, q, rec)}
        clean_ = dg_[u]["shared_mains"] == 0 and dg_[q]["shared_mains"] == 0
        ent = {"union": u, "per_origin": q, "diagnostic": dg_, "pair_reading": INTERVAL_ONLY if clean_ else "weighting and interval: point difference = weighting effect", "quantities": []}
        for pt in cfg["points"]:
            for sel in ("C", "R"):
                cu, cq = coh_of(u, pt, sel), coh_of(q, pt, sel)
                qs = [(f"delta_full_A {p_} {sel}", ((res.get(su) or {}).get(pt) or {}).get("rows", {}).get(f"{p_} {sel}", {}).get("delta_full_A"),
                       ((res.get(sq) or {}).get(pt) or {}).get("rows", {}).get(f"{p_} {sel}", {}).get("delta_full_A")) for p_ in (cfg.get("coheadline") or {}).get("probes") or []]
                def xs_(cb, blk, b_, k_, pt=pt):  # the excess where the co-headline omits it (the headline combo's six / off5): from the same rows, as coheadline() does for others
                    if k_ != "excess" or (b_.get(blk) or {}).get("excess") is not None or blk == "auroc_subset":
                        return (b_.get(blk) or {}).get(k_)
                    if (b_.get(blk) or {}).get("unavailable") or not (b_.get(blk) or {}).get("rows"):  # an unavailable block has no excess either (Codex streams r2)
                        return None
                    rows_ = (sets.get(combo_sets(cfg, cb)[0]) or {}).get(pt) or {}
                    labs_ = b_[blk]["rows"]
                    return strip(stats(rows_, labs_))["mean_G_minus_mean_Y"] if all(lb in rows_ and rows_[lb]["readable"] for lb in labs_) else None

                qs += [(f"{blk} {k_} {sel}", xs_(u, blk, cu, k_), xs_(q, blk, cq, k_)) for blk in ("six", "off5", "auroc_subset") for k_ in ("S", "Y", "excess")]
                qs += [(f"excess ({hg}) {sel}", (((res.get(su) or {}).get(pt) or {}).get("groups", {}).get(f"{hg}|{sel}") or {}).get("primary", {}).get("mean_G_minus_mean_Y"),
                        (((res.get(sq) or {}).get(pt) or {}).get("groups", {}).get(f"{hg}|{sel}") or {}).get("primary", {}).get("mean_G_minus_mean_Y"))]
                for nm, zu, zq in qs:
                    if not (isinstance(zu, dict) and isinstance(zq, dict) and "point" in zu and "point" in zq):
                        ent["quantities"].append({"level": pt, "quantity": nm, "unavailable": "missing on one stream"})
                        continue
                    d_ = zu["point"] - zq["point"]
                    e_ = {"level": pt, "quantity": nm, "union": zu, "per_origin": zq, "half_width_ratio_union_over_per_origin": (hw(zu) / hw(zq)) if hw(zu) is not None and hw(zq) else None,
                          "includes_0": {"union": inc0(zu), "per_origin": inc0(zq)}, "includes_0_flips": None if inc0(zu) is None or inc0(zq) is None else inc0(zu) != inc0(zq),
                          "point_difference_union_minus_per_origin": d_, "point_difference_label": "point difference = weighting effect"}
                    ent["quantities"].append(e_)
        out["pairs"].append(ent)
    return out


def hit_cells(e_, row_o):
    """cell_hits_draw0.cells as a list: the explicit list, or - when it is EXACTLY the known descriptor string (09 via 2e, assert (a)) - sorted(row's
    cohort_cell_keys), whose length must equal every per-arm level string, the weights and the mains (assert (b); its equality to the row's validated
    cohort is the callers' check). None (fail closed) for any other value."""
    c_ = (e_ or {}).get("cells")
    if isinstance(c_, list):
        return list(c_)
    ck_ = (row_o or {}).get("cohort_cell_keys")
    if c_ != CELLS_DESCRIPTOR or not isinstance(ck_, list):
        return None
    n_ = len(ck_)
    lens_ = [len(v_) for lv_ in ((e_.get("levels") or {}).values()) for v_ in (lv_ or {}).values()] + [len(e_.get("weight") or []), len(e_.get("mains") or [])]
    return sorted(ck_) if lens_ and all(x_ == n_ for x_ in lens_) else None


def row_drops(row_o):
    """Cohort cells a row dropped (fold -1 rule) on one origin, from the source record."""
    x = ((row_o or {}).get("fold_minus1_dropped") or {}).get("cohort_cells") or {}
    return list(x.get("keys") or []) if isinstance(x, dict) else list(x)


def ldc_joiner_rows_check(r0, ck):
    """Every on-policy row's record cohort_cells must equal the cohort keys minus that row's own fold -1 drops (which must be cohort keys)."""
    for lb, row in r0.items():
        if onpol(lb):
            for o, keys in ck.items():
                if o not in row or "coverage" not in row[o]:  # absent or a withheld stub (no coverage stated)
                    continue
                dr = row_drops(row[o])
                if not set(dr) <= set(keys):
                    return f"{lb} {o}: dropped cells {sorted(set(dr) - set(keys))} are not cohort keys"
                if int(row[o]["coverage"]["cohort_cells"]) != len(set(keys) - set(dr)):
                    return f"{lb} {o}: record cohort_cells {row[o]['coverage']['cohort_cells']} != {len(set(keys) - set(dr))} (cohort keys minus the row's fold -1 drops)"
                if row[o].get("cohort_cell_keys") is not None and set(row[o]["cohort_cell_keys"]) != set(keys) - set(dr):  # by key where the record lists them
                    return f"{lb} {o}: record cohort_cell_keys differ from the cohort keys minus the row's fold -1 drops (by key)"
    return None


def shipped_entry_sha(shipped, name, path):
    """The sha256 of the ONE designated entry in a record's meta.shipped_inputs: a dict key exactly equal to `name`, or an entry whose "path" exactly
    equals `path`. Returns (sha or None, number of matching entries); callers require exactly one match per record (Codex v12 r3)."""
    found = []
    if isinstance(shipped, dict) and name in shipped:
        v = shipped[name]
        found.append(v if isinstance(v, str) else (v or {}).get("sha256"))
    items = shipped.values() if isinstance(shipped, dict) else (shipped if isinstance(shipped, list) else [])
    for v in items:
        if path is not None and isinstance(v, dict) and v.get("path") == path:
            found.append(v.get("sha256"))
    return (found[0] if len(found) == 1 else None), len(found)


def lead_groups_names(res, pt0):
    return {k.split("|")[0] for k in res["pooled"][pt0]["groups"]}


def mkrow(lab, x, r_, n0, cfg):
    """One row from a catches dict x (sim, run1, run2, y2, y1 per draw): G, Y, signed gaps, bias arrays and labels."""
    c0 = {k: float(x[k][0]) for k in ("sim", "run1", "run2", "y2", "y1")}
    y_ = (x["run2"] if cfg.get("pooled_world") else x["y2"]) - x["y1"]  # pooled (RULE 45.1; Stage A's Yo): run2@B - y1 (= run1@B); else run2@A - y1
    return mkrow_gaps(lab, x["sim"] - x["run1"], x["sim"] - x["run2"], y_, c0, x["sim"], x.get("D"), r_, n0, cfg, ytb=x["run2"] - x["run1"])


def mkrow_gaps(lab, d1, d2, y, c0, sim_arr, D_arr, r_, n0, cfg, ytb=None):
    """One row from per-draw gaps d1 = sim - run1, d2 = sim - run2, y = run2@A - run1@B2|B and the draw-0 catches c0. ytb: the signed yardstick run2 - run1 at
    tau_B from the REAL arms (its own NaN mask; Codex pilot r3: d1 - d2 would inherit the sim arm's undefined draws); every caller passes it."""
    import numpy as np

    if ytb is None:
        raise ValueError(f"{lab}: mkrow_gaps needs the real-arm yardstick run2 - run1 (ytb)")
    op = onpol(lab) and cfg.get("g_onpolicy_run2_only", False)
    rm = bool(cfg.get("signed_runmean_all"))  # 09 Final-build checklist item 2: the run-mean Delta_full for EVERY member (on-policy rows included); sim - run 2 beside
    sd_sr = float(np.nanstd((d2 if onpol(lab) and not rm else 0.5 * (d1 + d2))[1:]))  # SD of the signed sim - real (Codex r1 on v9-v11)
    sd_rr = float(np.nanstd(y[1:]))
    ratio = sd_sr / sd_rr if sd_rr > 0 else float("nan")
    bias_ = {"S": d2 if onpol(lab) and not rm else 0.5 * (d1 + d2), "S_reg": d2 if onpol(lab) else d1, "S_r2": d2, **({"Yt": ytb} if rm else {}),
             "G2": d2**2 if op else 0.5 * (d1**2 + d2**2), "Y2": y**2, "Y2r": ratio**2 * y**2, "sd_sr": sd_sr, "sd_rr": sd_rr, "ratio": ratio}
    return {"sim_arr": sim_arr, "D_arr": D_arr, "bias": bias_, "G_reg": 0.5 * np.abs(d1) + 0.5 * np.abs(d2), "G_r2": np.abs(d2), "d1": d1, "d2": d2, "y": y, "ytb": ytb,
            "G": np.abs(d2) if op else 0.5 * np.abs(d1) + 0.5 * np.abs(d2), "Y": np.abs(y), "draw0_catch": c0, "readable": r_,
            "saturated": max(c0[k_] for k_ in sat_keys(cfg, c0)) >= cfg["saturation"], "g_def": G_DEF["onpolicy" if op else "default"], "n0": n0}


def mkrow_sens(rows_o, D, pooled_row):
    """Sensitivity row: |.| per origin, then weighted by each draw's main weight D over the given origins."""
    Dd = sum(D.values())
    w = lambda k, sub=None: sum((rows_o[o][sub][k] if sub else rows_o[o][k]) * D[o] for o in rows_o) / Dd
    return {"d1": None, "d2": None, "y": None, "G": w("G"), "Y": w("Y"), "draw0_catch": pooled_row["draw0_catch"], "readable": pooled_row["readable"],
            "saturated": any(rows_o[o]["saturated"] for o in rows_o), "D0": tuple(float(D[o][0]) for o in rows_o), "D0_origins": list(rows_o), "g_def": rows_o[next(iter(rows_o))]["g_def"],
            "n0": float(Dd[0]), "G_reg": w("G_reg"), "G_r2": w("G_r2"),
            "bias": {k: w(k, "bias") for k in ("S", "S_reg", "S_r2", "G2", "Y2", "Y2r", "Yt") if k in rows_o[next(iter(rows_o))]["bias"]} | {"sd_sr": None, "sd_rr": None, "ratio": None,
                                                                                         "ratio_per_origin": {o: rows_o[o]["bias"]["ratio"] for o in rows_o}}}


def san2(x):
    """v2 part sanitisation (507-prep record key_format v2): '@'->'_at_', '|'->'_', ' '->'_', parentheses dropped."""
    return x.replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")


def keys_for(fmt, lab, f, o=None, catch=None, gap=None, combo=None, mode="joint"):
    """npz key for a per-origin catch / D (o given) or a pooled gap (gap given), in key format v1 (pool_reduce @ 1181ef38f), v2 (@ 214f74de4) or v2c
    (507-prep's all-tasks record: <label>__<combo>__<mode>__<f>__..., every per-origin array repeated under each combo that holds the origin)."""
    if fmt in ("v2", "v2c"):
        base = f"{san2(lab)}__" + (f"{san2(combo)}__" if fmt == "v2c" else "") + f"{mode}__{f}"
        if gap is not None:
            return f"{base}__pooled__gap_{gap}"
        o_ = o.split(" [pooled: ", 1)[0]  # pooled rows: Stage A's npz names the origin WITHOUT its ' [pooled: <combo>]' tag (the combo is its own key part; stub @ 1c7418b1b)
        return f"{base}__{san2(o_)}__D" if catch == "D" else f"{base}__{san2(o_)}__catch_{san2(catch)}"  # every part sanitised (a no-op for S_old / S_new)
    base = f"{san(lab)}__{mode}__{f}"
    if gap is not None:
        return f"{base}__{gap}"
    return f"{base}__{o}__{san(catch)}"


def load(cfg, recs):
    """sets[set][pt][label] = {d1, d2, y, readable, saturated, draw0_catch}; gates list. Origins and combos from the config (default: S_old and S_new, one
    combo = the pooled 455, npz keys without a combo part). With combos, each label's members per combo come from its record (withheld members listed there)."""
    from functools import reduce
    from operator import add

    import numpy as np

    origins, combos, mode = cfg.get("origins", ORIGINS), cfg.get("combos"), cfg.get("mode", "joint")
    hc = cfg.get("headline_combo")
    if combos and hc not in combos:
        raise ValueError(f"headline_combo {hc!r} is not one of the combos {list(combos)}")
    sn = set_names(cfg)  # every emitted set name unique; origins and members unique; no collision after npz-key sanitisation (Codex N-origin r1)
    if (len(set(sn)) != len(sn) or len(set(origins)) != len(origins) or len({san2(o) for o in origins}) != len(origins)
            or len({san2(cb) for cb in (combos or {})}) != len(combos or {}) or any(len(set(M)) != len(M) or not set(M) <= set(origins) for M in (combos or {}).values())):
        raise ValueError(f"set, origin or combo names collide or a combo names an unknown or repeated origin: {sn} / {combos}")
    tol = float(cfg.get("pooled_recompute_tol_abs", 0.0))  # 0 = exact (the pooled 455); else stated in the config up front (09: STOP beyond documented float noise)
    same = ((lambda a, b: bool(np.array_equal(a, b, equal_nan=True))) if tol == 0 else
            (lambda a, b: bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a, b, rtol=0, atol=tol, equal_nan=True))))
    same0 = (lambda a, b: bool(a == b)) if tol == 0 else (lambda a, b: bool(abs(a - b) <= tol))
    tot = lambda xs: reduce(add, xs)  # left-to-right sum without a leading 0 (bitwise the two-origin expressions of the pooled 455)
    sets, gates, dkeys, ld_avail, extra = {s: {} for s in set_names(cfg)}, [], [], {}, {"dk0": {}, "ld_avail_by_combo": {}}
    if cfg.get("ld_restricted"):  # Stage A's per-draw arrays restricted to the covered cells (09: the 268)
        pr_ = Path("/sol") / cfg["ld_restricted"]["npz"]["path"]
        assert hashlib.sha256(pr_.read_bytes()).hexdigest() == cfg["ld_restricted"]["npz"]["sha256"], "restricted npz sha256"
        cfg["_restricted_z"] = np.load(pr_)
    for src, rec in zip(cfg["npz"], recs):
        p = Path("/sol") / src["path"]
        b = p.read_bytes()
        assert hashlib.sha256(b).hexdigest() == src["sha256"], f"npz sha256 {src['path']}"
        z = np.load(p)
        for lab in src["labels"]:
            if lab in ((cfg.get("npz_collision_excluded") or {}).get("labels") or []):  # 2e / 0d 10-16: the npz holds another label's arrays under this label's keys;
                extra.setdefault("npz_collision_excluded", {})[lab] = cfg["npz_collision_excluded"]["reason"]  # no npz-paired gate or row (record JSON unaffected)
                continue
            for f, pt in zip(cfg["f_keys"], cfg["points"]):
                fmt = src.get("key_format", "v1")
                if combos:  # the record's members of this label per combo (headline combo first); members + withheld must be the configured combo exactly
                    mem, unav = {}, {}
                    for cb in sorted(combos, key=lambda c_: c_ != hc):
                        e_ = rec["pooled"][lab][cb]
                        if not e_.get("members") and isinstance(e_.get("pooled"), str) and e_["pooled"].startswith("unavailable"):  # the record states the combo unavailable here
                            unav[cb] = e_["pooled"]
                            gates.append({"row": lab, "f": f, "combo": cb, "check": "unavailable combo: its unavailable members are configured members",
                                          "pass": bool(e_.get("unavailable_members")) and set(e_["unavailable_members"]) <= set(combos[cb])})
                            continue
                        mcb = list(e_["members"])
                        wh = list(e_.get("withheld_members") or {})
                        gates.append({"row": lab, "f": f, "combo": cb, "check": "record members + withheld members = the configured combo",
                                      "pass": bool(mcb) and len(set(mcb)) == len(mcb) and sorted(mcb + wh) == sorted(combos[cb])})
                        mem[cb] = mcb
                    if unav:
                        extra.setdefault("combo_unavailable", {}).setdefault(lab, unav)
                    if not mem:
                        gates.append({"row": lab, "f": f, "check": "at least one available combo", "pass": False})
                        continue
                else:
                    mem = {None: list(origins)}
                if keys_for(fmt, lab, f, gap="delta_rr", combo=next(iter(mem)), mode=mode) not in z:
                    gates.append({"row": lab, "f": f, "check": "pooled arrays present", "pass": False})
                    continue
                org = [o for o in origins if any(o in m for m in mem.values())]
                per, rd = {}, {}
                for o in org:
                    cb0 = next(cb for cb, m in mem.items() if o in m)  # read once (from the first combo holding it); every other copy must be equal
                    g = lambda c, cb_=cb0: z[keys_for(fmt, lab, f, o=o, catch=c, combo=cb_, mode=mode)]  # noqa: B023
                    has_b2 = keys_for(fmt, lab, f, o=o, catch="run1@B2", combo=cb0, mode=mode) in z
                    if cfg.get("pooled_world"):  # RULE 45.1 (via 2e): a pooled row carrying run1@B2 STOPs
                        gates.append({"row": lab, "f": f, "origin": o, "check": "pooled row has no run1@B2 array (RULE 45.1: y1 = run1@B)", "pass": not has_b2})
                    dkeys.append({"key": keys_for(fmt, lab, f, o=o, catch="D", combo=cb0, mode=mode), "draw0": float(g("D")[0]),
                                  "yardstick_run1_arm": "run1@B2" if has_b2 else "run1@B"})
                    extra["dk0"][(lab, f, o)] = dkeys[-1]["draw0"]
                    per[o] = {"sim": g("sim@S|primary"), "run1": g("run1@B"), "run2": g("run2@B"), "y2": g("run2@A"), "y1": g("run1@B2") if has_b2 else g("run1@B"),
                              "D": g("D")}
                    arms = (("sim@S|primary", "sim"), ("run1@B", "run1"), ("run2@B", "run2"), ("run2@A", "y2")) + ((("run1@B2", "y1"),) if has_b2 else ())
                    for cb in mem:
                        if cb != cb0 and o in mem[cb]:  # each copy's own arms, its y1 resolved from its own keys (B2 where present, else B) (Codex N-origin r1)
                            b2c = keys_for(fmt, lab, f, o=o, catch="run1@B2", combo=cb, mode=mode) in z
                            for c, nm in (("sim@S|primary", "sim"), ("run1@B", "run1"), ("run2@B", "run2"), ("run2@A", "y2"), ("run1@B2" if b2c else "run1@B", "y1"), ("D", "D")):
                                gates.append({"row": lab, "f": f, "origin": o, "combo": cb, "arm": c, "check": f"copy equals {cb0}'s", "pass": bool(np.array_equal(g(c, cb), per[o][nm], equal_nan=True))})
                    for c, nm in arms:
                        ref = jget(rec, cfg["paths"]["origin_catch"], label=lab, origin=o, f=f, key=c, mode=mode)
                        gates.append({"row": lab, "f": f, "origin": o, "arm": c, "pass": bool(per[o][nm][0] == ref)})
                    rd[o] = readable_of(cfg, jget(rec, cfg["paths"]["origin_readable"], label=lab, origin=o, f=f, mode=mode))
                ncell = {o: int(jget(rec, ["rows", "{label}", "{origin}", "coverage", "cohort_cells"], label=lab, origin=o)) for o in org}
                dev_all = cfg.get("drop_evidence_resolved")
                if dev_all is not None:  # 507-prep: drops[<row label>][<base origin>] is the may-drop SUPERSET; every stated drop of every row must be in it, by key
                    for o in org:
                        dr_o = set(row_drops(rec["rows"][lab][o]))
                        allow_ = (dev_all.get(lab) or {}).get(drop_origin(cfg, o))
                        if dr_o and (allow_ is None or not dr_o <= set(allow_)):
                            gates.append({"row": lab, "f": f, "origin": o, "check": f"fold -1 drops {sorted(dr_o - set(allow_ or []))[:5]} not in the pinned may-drop list", "pass": False})
                mf_ = cfg.get("moments_fallback")  # 09 (via 2e): Stage A's own fallback-moments text, copied VERBATIM from each member row's moments_source, with its firing count
                mfb = {}
                if mf_:
                    for o in org:
                        ro_ = rec["rows"][lab][o]
                        if "moments_source" not in ro_ or ro_["moments_source"] is None:  # missing on a member row: STOP (null is only for withheld stubs, never members)
                            gates.append({"row": lab, "f": f, "origin": o, "check": "moments_source missing on a member row (fail closed)", "pass": False})
                            continue
                        ms_ = ro_["moments_source"]
                        items_ = list(ms_.items()) if isinstance(ms_, dict) else [("all", ms_)]
                        if not items_ or any(not isinstance(v_, str) or not v_ for _, v_ in items_):  # a string, or a dict of non-empty strings; anything else STOPs (Codex final r21)
                            gates.append({"row": lab, "f": f, "origin": o, "check": "moments_source is not a string or a dict of strings (fail closed)", "pass": False})
                            continue
                        hit_ = {k_: v_ for k_, v_ in items_ if mf_["label"] in v_}
                        fi_ = None
                        if mf_.get("firings_field"):  # 507-prep: rows[<label>][<origin>]["moments_fallback_firings"] = [{pool, fold, n_units, source}] ([] when nothing fires)
                            try:
                                fi_ = jget(ro_, mf_["firings_field"])
                            except (KeyError, TypeError, IndexError):
                                fi_ = None
                        SRC_OK = (mf_["label"], "registered all-unit fallback")  # the documented sources (507-prep)
                        shape_ok = isinstance(fi_, list) and all(isinstance(x_, dict) and {"pool", "fold", "n_units", "source"} <= set(x_) and isinstance(x_["pool"], str) and x_["pool"]
                                                                 and x_["source"] in SRC_OK for x_ in fi_)  # [] is valid; any malformed entry -> "unrecognised shape" (Codex final r23)
                        mine_ = [x_ for x_ in fi_ if x_["source"] == mf_["label"]] if shape_ok else None
                        if hit_ or mine_ or (fi_ is not None and not shape_ok and any(isinstance(x_, dict) and x_.get("source") == mf_["label"] for x_ in (fi_ if isinstance(fi_, list) else []))):
                            # disclosure, never a STOP (2e; Codex final r22/r23): count the firings of 09's label when the list is readable
                            if mine_ is not None:
                                byp_ = {}
                                for x_ in mine_:
                                    byp_[str(x_.get("pool"))] = byp_.get(str(x_.get("pool")), 0) + 1
                                fc_ = len(mine_)
                                fcp = byp_
                            else:
                                fc_ = ("firing count: not recorded in this record (see Stage A record)" if fi_ is None else
                                       "firing count: unrecognised shape in this record (see Stage A record)")
                                fcp = None
                            src_txt = hit_ or ({"(moments_fallback_firings)": mine_[0]["source"]} if mine_ else
                                               {"(moments_fallback_firings)": next(x_["source"] for x_ in fi_ if isinstance(x_, dict) and x_.get("source") == mf_["label"])})
                            mfb[o] = {"moments_source": src_txt, "firings": fc_, "firings_by_pool": fcp}  # Stage A's own text, verbatim
                for o in org:
                    sets[o].setdefault(pt, {})[lab] = mkrow(lab, per[o], rd[o], float(per[o]["D"][0]), cfg) | {"n_cells": ncell[o]} | ({"moments_fallback": {o: mfb[o]}} if o in mfb else {})
                bad_lab = False
                for cb, M in mem.items():
                    ps, ss = combo_sets(cfg, cb)
                    Dd = tot([per[o]["D"] for o in M])
                    if cb in read_combos(cfg):  # 09 @ d799177ae item 5: a main-deduplicated (or a per-origin-stream) combo's pooled row is READ from Stage A's saved gaps, gated at draw 0
                        yg_ = (cfg.get("pooled_world") or {}).get("y_gap")  # pooled: the yardstick gap at the pooled tau_B (RULE 45.1; mapped explicitly, no B2)
                        sv = {nm: z[keys_for(fmt, lab, f, gap=nm, combo=cb, mode=mode)] for nm in ("sim_minus_run1", "sim_minus_run2", "delta_rr") + ((yg_,) if yg_ else ())}
                        for nm, a_ in sv.items():
                            gates.append({"row": lab, "f": f, "origin": "pooled", "combo": cb, "arm": f"{nm} draw 0 vs record (Stage A dedup row)",
                                          "pass": same0(a_[0], jget(rec, cfg["paths"]["pooled_gap"], label=lab, f=f, key=nm, combo=cb, mode=mode))})
                        cat = jget(rec, ["pooled", "{label}", "{combo}", "{mode}", "{f}", "catch"], label=lab, combo=cb, mode=mode, f=f)
                        miss_ = [k for k in ("sim", "run1", "run2", "run2@A", "y1") if not isinstance(cat.get(k), dict) or cat[k].get("point") is None]
                        if miss_:  # 507-prep adds run2@A and y1 to the pooled catch block; saturation needs all five arms (fail closed; Codex checklist r1)
                            gates.append({"row": lab, "f": f, "combo": cb, "check": f"dedup row: pooled catch arms {miss_} missing in the record (fail closed)", "pass": False})
                            bad_lab = True
                            break
                        c0 = {"sim": float(cat["sim"]["point"]), "run1": float(cat["run1"]["point"]), "run2": float(cat["run2"]["point"]), "y2": float(cat["run2@A"]["point"]),
                              "y1": float(cat["y1"]["point"])}
                        if cb in (cfg.get("dedup_combos") or []):  # Codex pilot r2 #1/#2 (every union row, pilot or full): rebuild the row's catches PER DRAW from its members'
                            # arrays and D as Stage A's all-origins comb does (sum_o where(D_o > 0, X_o D_o, 0) / sum_o D_o), gate draw 0 against the record's pooled catch,
                            # and gate the saved gaps per draw against them (NaN where the arm is undefined, on both sides); the yardstick against the REAL arms only
                            with np.errstate(invalid="ignore", divide="ignore"):
                                xc_ = {a_: tot([np.where(per[o]["D"] > 0, per[o][a_] * per[o]["D"], 0.0) for o in M]) / Dd for a_ in ("sim", "run1", "run2")}
                            for a_ in ("sim", "run1", "run2"):
                                gates.append({"row": lab, "f": f, "origin": "pooled", "combo": cb, "arm": a_, "check": "union-row catch rebuilt from members and D equals the record at draw 0",
                                              "pass": bool(abs(float(xc_[a_][0]) - c0[a_]) <= 1e-12 or (math.isnan(float(xc_[a_][0])) and math.isnan(c0[a_])))})
                            for nm, mine in (("sim_minus_run1", xc_["sim"] - xc_["run1"]), ("sim_minus_run2", xc_["sim"] - xc_["run2"])) + (((yg_, xc_["run2"] - xc_["run1"]),) if yg_ else ()):
                                gates.append({"row": lab, "f": f, "origin": "pooled", "combo": cb, "arm": nm, "check": "saved union-row gap equals the member rebuild in every draw",
                                              "pass": bool(np.allclose(sv[nm], mine, rtol=0, atol=1e-12, equal_nan=True))})
                        sh_ = (cfg.get("pooled_world") or {}).get("shared_threshold") or {}
                        rl_ = None
                        if sh_ and (lab in (sh_.get("labels") or []) or isinstance(dg_(rec, "pooled", lab, cb, mode, f, "gap", sh_["gap"]), dict)):  # [SENS shared] (2e 10-16): the
                            # pooled real_line = sim@B - real at the pooled tau_B, union draws: REQUIRED on the configured rows, and gated on every row the record states it for
                            k_rl = keys_for(fmt, lab, f, gap=sh_["gap"], combo=cb, mode=mode)
                            try:
                                ref_rl = jget(rec, cfg["paths"]["pooled_gap"], label=lab, f=f, key=sh_["gap"], combo=cb, mode=mode)
                            except (KeyError, TypeError):
                                ref_rl = None
                            ok_rl = k_rl in z and ref_rl is not None
                            gates.append({"row": lab, "f": f, "origin": "pooled", "combo": cb, "arm": sh_["gap"], "check": "shared-threshold gap present and its draw 0 equals the record",
                                          "pass": bool(ok_rl and same0(z[k_rl][0], ref_rl))})
                            rl_ = z[k_rl] if ok_rl else None
                        try:
                            sat_sa = jget(rec, ["pooled", "{label}", "{combo}", "{mode}", "{f}", "panel", "pooled", "saturated"], label=lab, combo=cb, mode=mode, f=f)
                        except (KeyError, TypeError):
                            sat_sa = None
                        gates.append({"row": lab, "f": f, "combo": cb, "check": "dedup row saturation (max draw-0 arm >= threshold) equals Stage A's panel flag",
                                      "pass": sat_sa is not None and (max(c0[k_] for k_ in sat_keys(cfg, c0)) >= cfg["saturation"]) == bool(sat_sa)})
                        # 09 item 5: an independent draw-0 dedup recompute from per-cell hits (507-prep's compact layout, per ORIGIN row): the combo's members are the
                        # "<origin> (all-origins draws)" rows, each with cell_hits_draw0 = {"cells", "weight" (W0[main] / n_cells_all_origins(main)), "levels": {<f>: {arm:
                        # "0110..."}}, "arms_def": {arm: {threshold, catch_key}}}; the combo's hits are their concatenation. W0[main] is derived independently from the BASE
                        # origin rows (w_c = W0[main] / n_cohort_cells(main)), equal across origins; the expected combo weights are W0[main] / n_cells_all_origins(main) by MY
                        # pinned main map. It runs whenever every member states hits; a CONFIGURED recompute ("dedup_recompute" present, even empty) STOPs otherwise.
                        import math as _m

                        conf_ = "dedup_recompute" in cfg and cb not in (cfg.get("stream_combos") or [])  # a per-origin-stream combo is read only (no one-main recompute)
                        dr_ = cfg.get("dedup_recompute") or {}
                        base_ = cfg.get("origin_base") or {}
                        fin_pos = lambda x_: isinstance(x_, (int, float)) and not isinstance(x_, bool) and _m.isfinite(x_) and x_ > 0
                        parts_ = [(o, (rec["rows"][lab].get(o) or {}).get("cell_hits_draw0")) for o in M]
                        have_ = cb not in (cfg.get("stream_combos") or []) and all(isinstance(e_, dict) and e_.get("cells") for _, e_ in parts_)
                        for o, e_ in parts_:  # 09 assert (a): a string `cells` must be EXACTLY the known descriptor, else STOP
                            if isinstance(e_, dict) and isinstance(e_.get("cells"), str):
                                gates.append({"row": lab, "f": f, "combo": cb, "origin": o, "check": "cell_hits_draw0.cells is exactly the known descriptor string",
                                              "pass": e_["cells"] == CELLS_DESCRIPTOR})
                        if conf_ and not have_:
                            gates.append({"row": lab, "f": f, "combo": cb, "check": f"dedup draw-0 recompute configured but members {[o for o, e_ in parts_ if not e_]} state no cell_hits_draw0",
                                          "pass": False})
                        recomputed = False
                        if have_:
                            cm_ = cfg.get("cell_main_resolved") or {}
                            ckr_ = cfg.get("cohort_keys_resolved") or {}
                            dev_ = cfg.get("drop_evidence_resolved") or {}
                            adx_ = dr_.get("arms_def_expected") or {}
                            vco_, cells_, w_, hs_, orig_of, ok_s = {}, [], [], {a_: "" for a_ in ("sim", "run1", "run2", "run2@A", "y1")}, {}, True
                            for o, e_ in parts_:
                                dr_o = set(row_drops(rec["rows"][lab][o]))
                                allow_ = set((dev_.get(lab) or {}).get(drop_origin(cfg, o)) or [])  # the pinned may-drop list by row label and base origin
                                if not dr_o <= allow_:  # every drop must be in the pinned evidence; none is assumed
                                    gates.append({"row": lab, "f": f, "combo": cb, "origin": o, "check": f"cohort drops {sorted(dr_o - allow_)[:5]} not in the pinned drop evidence", "pass": False})
                                vco_[o] = set(ckr_.get(o) or []) - dr_o - set(cfg.get("dedup_excluded_resolved") or [])
                                if set(rec["rows"][lab][o].get("cohort_cell_keys") or []) != set(ckr_.get(o) or []) - dr_o:
                                    gates.append({"row": lab, "f": f, "combo": cb, "origin": o, "check": "member cohort_cell_keys != the pinned cohort minus validated drops", "pass": False})
                                co_, wo_ = hit_cells(e_, rec["rows"][lab][o]) or [], list(e_.get("weight") or [])  # 09 (b): lengths here; = validated cohort in okp
                                lv_, ad_ = (e_.get("levels") or {}).get(f) or {}, e_.get("arms_def") or {}
                                is_old = base_.get(o, o) == "S_old" and not cfg.get("pooled_world")  # pooled (RULE 45.1 via 2e): y1 = run1@B everywhere, no S_old exception
                                arms_ = ("sim", "run1", "run2", "run2@A") + (("run1@B2",) if is_old else ())
                                if cfg.get("pooled_world"):
                                    gates.append({"row": lab, "f": f, "combo": cb, "origin": o, "check": "pooled member states no run1@B2 arm (RULE 45.1: y1 = run1@B)",
                                                  "pass": "run1@B2" not in ad_ and "run1@B2" not in lv_})
                                okp = (bool(co_) and len(co_) == len(wo_) == len(set(co_)) and set(co_) == vco_[o] and all(k_ in cm_ for k_ in co_) and all(fin_pos(x_) for x_ in wo_)
                                       and all(isinstance(lv_.get(a_), str) and len(lv_[a_]) == len(co_) and not set(lv_[a_]) - {"0", "1"} for a_ in arms_)
                                       and bool(adx_) and all(isinstance(ad_.get(a_), dict) and isinstance(adx_.get(a_), dict) and {"threshold", "catch_key"} <= set(adx_[a_])
                                                              and all(ad_[a_].get(k_) == v_ for k_, v_ in adx_[a_].items()) for a_ in arms_))  # threshold AND catch_key as expected
                                gates.append({"row": lab, "f": f, "combo": cb, "origin": o, "check": "member hits: cells = validated cohort, finite positive weights, 0|1 strings per arm, arms_def as expected",
                                              "pass": bool(okp)})
                                ok_s = ok_s and okp
                                if okp:  # the member's hits reproduce ITS row's draw-0 catches (computed at that pool's T) for every consumed arm (2e)
                                    Wo_ = sum(wo_)
                                    for a_ in arms_:
                                        ck_ = ad_[a_]["catch_key"]
                                        try:
                                            ref_ = jget(rec, cfg["paths"]["origin_catch"], label=lab, origin=o, f=f, key=ck_, mode=mode)
                                        except (KeyError, TypeError):
                                            ref_ = None
                                        mo0 = sum(x_ * (h_ == "1") for x_, h_ in zip(wo_, lv_[a_])) / Wo_
                                        gates.append({"row": lab, "f": f, "combo": cb, "origin": o, "arm": a_, "check": "member hits reproduce the row's draw-0 catch at its pool's T",
                                                      "pass": ref_ is not None and abs(mo0 - ref_) <= float(dr_.get("tol_abs", 1e-12))})
                                    cells_ += co_
                                    w_ += wo_
                                    orig_of |= {k_: o for k_ in co_}
                                    for a_ in ("sim", "run1", "run2", "run2@A"):
                                        hs_[a_] += lv_[a_]
                                    hs_["y1"] += lv_["run1@B2"] if is_old else lv_["run1"]  # y1: run1@B2 on S_old, run1@B elsewhere (Stage A's y1_def)
                            ok_s = ok_s and len(cells_) == len(set(cells_))
                            gates.append({"row": lab, "f": f, "combo": cb, "check": "the concatenated combo cells are unique", "pass": bool(ok_s)})
                            if ok_s and cfg.get("pooled_world"):  # 09 #7 (a): a pooled record has no per-set base rows; W0 FROM SCRATCH from the inputs (the pinned
                                # cell -> canonical main maps and the pinned cohort keys): at draw 0 every canonical main of the union counts once (W0 = 1); the expected
                                # cell weight is 1 / n_cells_union(main), asserted equal to the record's weights below (never derived from the producer's weights)
                                w0_, okw = {cm_[k_]: 1.0 for k_ in cells_}, True
                            elif ok_s:
                                w0_, okw = {}, True
                                for o in M:  # W0[main] from the BASE origin rows (w_c x the main's cells in the validated base cohort), equal across origins
                                    bo_ = base_.get(o, o)
                                    eb_ = (rec["rows"][lab].get(bo_) or {}).get("cell_hits_draw0") or {}
                                    hb_ = hit_cells(eb_, rec["rows"][lab].get(bo_))
                                    if hb_ is None:  # a specific diagnostic (Codex desc r1): the base row's cells are neither a list nor the exact descriptor with aligned lengths
                                        gates.append({"row": lab, "f": f, "combo": cb, "origin": bo_, "check": "base-row cell_hits_draw0 cells readable (a list, or the exact descriptor with "
                                                      "aligned lengths)", "pass": False})
                                    cb_, wb_ = hb_ or [], list(eb_.get("weight") or [])
                                    vb_ = set(ckr_.get(bo_) or []) - set(row_drops(rec["rows"][lab].get(bo_) or {}))
                                    if not cb_ or len(cb_) != len(wb_) or len(set(cb_)) != len(cb_) or set(cb_) != vb_ or any(k_ not in cm_ for k_ in cb_) or not all(fin_pos(x_) for x_ in wb_):
                                        okw = False
                                        break
                                    nmo = {}
                                    for k_ in vb_:
                                        nmo[cm_[k_]] = nmo.get(cm_[k_], 0) + 1
                                    for k_, x_ in zip(cb_, wb_):
                                        v_ = x_ * nmo[cm_[k_]]
                                        if not _m.isfinite(v_) or (cm_[k_] in w0_ and abs(w0_[cm_[k_]] - v_) > 1e-12 * max(1.0, abs(v_))):
                                            okw = False
                                        w0_.setdefault(cm_[k_], v_)
                            if ok_s:  # the record's weights must equal W0[main] / n_cells_all_origins(main) (W0 from the base rows, or from scratch for a pooled record)
                                nall = {}
                                for k_ in cells_:
                                    nall[cm_[k_]] = nall.get(cm_[k_], 0) + 1
                                okw = okw and all(cm_[k_] in w0_ and abs(x_ - w0_[cm_[k_]] / nall[cm_[k_]]) <= 1e-12 * max(1.0, abs(x_)) for k_, x_ in zip(cells_, w_))
                                gates.append({"row": lab, "f": f, "combo": cb, "check": ("dedup weights = 1 / n_cells_union(main), W0 from scratch (pinned cell->main maps, one main once at "
                                                                                          "draw 0; 09 #7)" if cfg.get("pooled_world") else
                                                                                          "dedup weights = W0[main] (from the base origin rows, equal across origins) / n_cells_all_origins(main)"),
                                              "pass": bool(okw)})
                            if ok_s and okw:  # no recompute from unvalidated base rows or weights (Codex desc r1)
                                if cfg.get("pooled_world"):  # Astra pooled r1: the recompute uses the from-scratch weights themselves, not the (equal) record weights
                                    w_ = [w0_[cm_[k_]] / nall[cm_[k_]] for k_ in cells_]
                                W_ = sum(w_)
                                cat_ = {a_: sum(x_ * (h_ == "1") for x_, h_ in zip(w_, hs_[a_])) / W_ for a_ in hs_}
                                tol_d = float(dr_.get("tol_abs", 1e-12))
                                for a_, k0 in (("sim", "sim"), ("run1", "run1"), ("run2", "run2"), ("run2@A", "y2"), ("y1", "y1")):
                                    gates.append({"row": lab, "f": f, "combo": cb, "arm": a_, "check": "dedup draw-0 recompute from the hit strings equals Stage A's pooled catch",
                                                  "pass": abs(cat_[a_] - c0[k0]) <= tol_d})
                                cg_ = (("sim_minus_run1", cat_["sim"] - cat_["run1"]), ("sim_minus_run2", cat_["sim"] - cat_["run2"]), ("delta_rr", cat_["run2@A"] - cat_["y1"]))
                                for nm_, v_ in cg_ + (((yg_, cat_["run2"] - cat_["run1"]),) if yg_ else ()):  # pooled: run2 - run1 at the pooled tau_B
                                    gates.append({"row": lab, "f": f, "combo": cb, "arm": nm_, "check": "the consumed saved gap at draw 0 equals the recompute", "pass": abs(float(sv[nm_][0]) - v_) <= tol_d})
                                recomputed = True
                        if delta_full_tol(cfg) is not None:
                            kf_, tf_ = keys_for(fmt, lab, f, gap="delta_full", combo=cb, mode=mode), delta_full_tol(cfg)
                            S_ = 0.5 * (sv["sim_minus_run1"] + sv["sim_minus_run2"])
                            gates.append({"row": lab, "f": f, "origin": "pooled", "arm": "delta_full (Stage A dedup row)", "combo": cb,
                                          "pass": kf_ in z and bool(np.array_equal(np.isnan(S_), np.isnan(z[kf_])) and np.allclose(S_, z[kf_], rtol=0, atol=tf_, equal_nan=True))})
                        if yg_:  # the real-arm yardstick run2 - run1 at tau_B (Codex pilot r3): pooled = the saved yardstick (gated per draw against the member rebuild)
                            ytb_ = sv[yg_]
                        elif cb in (cfg.get("dedup_combos") or []):  # a union row: the member rebuild (gated above)
                            ytb_ = xc_["run2"] - xc_["run1"]
                        else:  # a per-origin-stream combo: Stage A's plain D-weighted combination of the members' real arms
                            with np.errstate(invalid="ignore", divide="ignore"):
                                ytb_ = tot([per[o]["run2"] * per[o]["D"] for o in M]) / Dd - tot([per[o]["run1"] * per[o]["D"] for o in M]) / Dd
                        sets[ps].setdefault(pt, {})[lab] = mkrow_gaps(lab, sv["sim_minus_run1"], sv["sim_minus_run2"], sv[yg_ or "delta_rr"], c0, None, None, all(rd[o] for o in M), None, cfg,
                                                                     ytb=ytb_) | {
                            "n_cells": None, "members": list(M), "pooled_source": (STREAM_LABEL if cb in (cfg.get("stream_combos") or []) else
                                                                                   "pooled row read from Stage A (main-deduplicated); its draw-0 catches independently recomputed from "
                                                                                   "per-cell hits (a main counts once) and gated" if recomputed else
                                                                                   cfg.get("dedup_label", "pooled row read from Stage A (main-deduplicated); not independently recomputed"))} | (
                            {"real_line": rl_} if rl_ is not None else {})
                        sets[ss].setdefault(pt, {})[lab] = mkrow_sens({o: sets[o][pt][lab] for o in M}, {o: per[o]["D"] for o in M}, sets[ps][pt][lab]) | {
                            "n_cells": None, "pooled_source": "per-set sensitivity over the members' own rows and main weights (a main in several origins is weighted in each)"}
                        continue
                    pool = {k: tot([per[o][k] * per[o]["D"] for o in M]) / Dd for k in ("sim", "run1", "run2", "y2", "y1")}
                    for nm, mine in (("sim_minus_run1", pool["sim"] - pool["run1"]), ("sim_minus_run2", pool["sim"] - pool["run2"]), ("delta_rr", pool["y2"] - pool["y1"])):
                        gates.append({"row": lab, "f": f, "origin": "pooled", "arm": nm, "pass": same(mine, z[keys_for(fmt, lab, f, gap=nm, combo=cb, mode=mode)])} | ({"combo": cb} if combos else {}))
                        ref = jget(rec, cfg["paths"]["pooled_gap"], label=lab, f=f, key=nm, combo=cb, mode=mode)
                        gates.append({"row": lab, "f": f, "origin": "pooled", "arm": f"{nm} draw 0 vs record", "pass": same0(mine[0], ref)} | ({"combo": cb} if combos else {}))
                    if delta_full_tol(cfg) is not None:  # S is built from the same arrays as Stage A's Delta_full: sim - (run1 + run2)/2 per row, gated per draw (stated tolerance)
                        kf_, tf_ = keys_for(fmt, lab, f, gap="delta_full", combo=cb, mode=mode), delta_full_tol(cfg)
                        S_ = 0.5 * ((pool["sim"] - pool["run1"]) + (pool["sim"] - pool["run2"]))
                        gates.append({"row": lab, "f": f, "origin": "pooled", "arm": "delta_full", "combo": cb,
                                      "pass": kf_ in z and bool(np.array_equal(np.isnan(S_), np.isnan(z[kf_])) and np.allclose(S_, z[kf_], rtol=0, atol=tf_, equal_nan=True))})
                        gates.append({"row": lab, "f": f, "origin": "pooled", "arm": "delta_full draw 0 vs record", "combo": cb,
                                      "pass": bool(abs(S_[0] - jget(rec, cfg["paths"]["pooled_gap"], label=lab, f=f, key="delta_full", combo=cb, mode=mode)) <= tf_)})
                    sets[ps].setdefault(pt, {})[lab] = mkrow(lab, pool, all(rd[o] for o in M), float(Dd[0]), cfg) | {"n_cells": sum(ncell[o] for o in M), "members": list(M)}
                    # SENSITIVITY: |.| taken per origin, then pooled with the same per-draw main weights D (no cross-origin cancellation; a different estimand)
                    sets[ss].setdefault(pt, {})[lab] = mkrow_sens({o: sets[o][pt][lab] for o in M}, {o: per[o]["D"] for o in M}, sets[ps][pt][lab]) | {
                        "n_cells": sum(ncell[o] for o in M)}
                if bad_lab:  # a fail-closed dedup gate: nothing else is built for this label (the run STOPs on the failed gate)
                    continue
                for cb, M in mem.items():  # the pooled rows carry their members' fallback statements (verbatim)
                    mm_ = {o: mfb[o] for o in M if o in mfb}
                    for st_ in combo_sets(cfg, cb):
                        if mm_ and lab in sets[st_].get(pt, {}):
                            sets[st_][pt][lab]["moments_fallback"] = mm_
                if cfg.get("estimator_choice") == "B":  # 09 @ 3aaf425ef: the lead's per-probe Delta_full and S use (B) all-runs averaging; Y and the excess stay on (A)
                    gB = cfg["estimator_selection"]["B_gap"]  # the record's gap key (507-prep: "delta_full|B"); the npz name may be sanitised (B_gap_npz)
                    for cb in mem:
                        kB = keys_for(fmt, lab, f, gap=cfg["estimator_selection"].get("B_gap_npz", gB), combo=cb, mode=mode)
                        rowB = sets[combo_sets(cfg, cb)[0]].get(pt, {}).get(lab)
                        okB = kB in z and rowB is not None
                        if okB:
                            rowB["delta_full_B"] = z[kB]
                            try:
                                okB = same0(z[kB][0], jget(rec, cfg["paths"]["pooled_gap"], label=lab, f=f, key=gB, combo=cb, mode=mode))
                            except (KeyError, TypeError):
                                okB = False
                        gates.append({"row": lab, "f": f, "combo": cb, "arm": f"{gB} (B) present and draw 0 vs record", "pass": bool(okB)})
                # 09's rule @ 1ffffca72 sec. 2-3 as amended @ 24fd07866: the joiner (chosen from the ELIGIBILITY file before any score is read) gets a
                # launch-disjoint row over its eligible sets; the off-policy 5 get the same subset (comparability check). Sets must be FULLY eligible here
                # (eligible mains = all mains); a partially eligible set needs restricted per-draw inputs, so the row is withheld (no fallback).
                # With combos: per combo, over the eligible origins among that label's members (the headline combo's validity decides the headline).
                ldc = cfg.get("ld_choice")
                fam, ro = split(lab)
                skip_rest = False
                if ldc and ldc.get("joiner") and (fam == ldc["joiner"] or fam in GROUPS[HEADLINE]):
                    tag = LD_TAG if fam == ldc["joiner"] else SUBSET_TAG
                    lab_ld = f"{fam} {tag} {ro}"
                    if combos and hc in (extra.get("combo_unavailable") or {}).get(lab, {}):  # the record states the headline combo unavailable for this row: no fallback
                        ld_avail.setdefault(fam, {})[(ro, pt)] = {"label": LD_LABEL if tag == LD_TAG else SUBSET_LABEL, "combo": hc, "valid": False,
                                                                  "reason": f"unavailable (Stage A record): {extra['combo_unavailable'][lab][hc]}"}
                    for cb, M in mem.items():
                        ps, ss = combo_sets(cfg, cb)
                        allowed = [o for o in ldc["allowed_origins"] if o in M]
                        info = {"label": LD_LABEL if tag == LD_TAG else SUBSET_LABEL, "allowed_origins": allowed, "withheld_or_ineligible": ldc["excluded"],
                                "mains_draw0": float(sum(per[o]["D"][0] for o in allowed)) if allowed else 0.0} | ({"combo": cb, "row_members": list(M)} if combos else {})
                        avail = ld_avail if cb == (hc if combos else None) else extra["ld_avail_by_combo"].setdefault(cb, {})
                        restr = {o: ldc["restricted"][o] for o in allowed if o in (ldc.get("restricted") or {})}
                        rz = cfg.get("_restricted_z")
                        part_ = [o for o in (ldc.get("partial") or []) if o in M]  # partially eligible sets among this combo's members
                        if allowed and not part_ and restr and (len(allowed) != 1 or rz is None or lab not in cfg["ld_restricted"]["label_map"] or combos):
                            info |= {"valid": False, "reason": ("unavailable: the row must be restricted to the covered cells and " +
                                                                ("restriction across several origins is not supported" if len(allowed) != 1 else
                                                                 "restriction with combos is not supported" if combos else "the restricted inputs are not provided")
                                                                + " (no fallback)"), "uncovered_cells": restr}
                            avail.setdefault(fam, {})[(ro, pt)] = info
                            if combos:  # this combo only: the others still get their rows (Codex N-origin r1)
                                continue
                            skip_rest = True
                            break
                        if allowed and not part_:
                            if restr:  # one origin, restricted to its covered cells: Stage A's restricted per-draw arrays (gated at draw 0 against their record)
                                o1 = allowed[0]
                                lr = cfg["ld_restricted"]["label_map"][lab]
                                fr = cfg["ld_restricted"].get("key_format", "v2")
                                per_r = {nm: rz[keys_for(fr, lr, f, o=o1, catch=c)] for nm, c in (("sim", "sim@S|primary"), ("run1", "run1@B"), ("run2", "run2@B"), ("y2", "run2@A"))}
                                b2r = keys_for(fr, lr, f, o=o1, catch="run1@B2")
                                per_r["y1"] = rz[b2r] if b2r in rz else per_r["run1"]
                                per_r["D"] = rz[keys_for(fr, lr, f, o=o1, catch="D")]
                                rrec = cfg["ld_restricted_rec"]
                                for c, nm in (("sim@S|primary", "sim"), ("run1@B", "run1"), ("run2@B", "run2"), ("run2@A", "y2")):
                                    gates.append({"row": lr, "f": f, "origin": o1, "arm": c, "restricted": True,
                                                  "pass": bool(per_r[nm][0] == jget(rrec, cfg["paths"]["origin_catch"], label=lr, origin=o1, f=f, key=c))})
                                rd_r = readable_of(cfg, jget(rrec, cfg["paths"]["origin_readable"], label=lr, origin=o1, f=f))
                                row_ld = mkrow(lab_ld, per_r, rd_r, float(per_r["D"][0]), cfg) | {"n_cells": int(cfg["ld_restricted_rec"]["rows"][lr][o1]["coverage"]["cohort_cells"])}
                                sens_ld = row_ld | {}
                                info |= {"uncovered_cells": restr, "covered_cells": int(cfg["ld_restricted_rec"]["rows"][lr][o1]["coverage"]["cohort_cells"]),
                                         "mains_draw0": float(per_r["D"][0]), "source": f"restricted inputs {cfg['ld_restricted']['npz']['path']} (label {lr})"}
                            elif len(allowed) == 1:  # one origin: its own arrays, exactly (no re-weighting)
                                o1 = allowed[0]
                                row_ld, sens_ld = mkrow(lab_ld, per[o1], rd[o1], float(per[o1]["D"][0]), cfg) | {"n_cells": ncell[o1]}, sets[o1][pt][lab] | {}
                            elif sorted(allowed) == sorted(M):  # every member eligible: the combo's own pooled row
                                row_ld, sens_ld = sets[ps][pt][lab] | {}, sets[ss][pt][lab] | {}
                            elif cb in read_combos(cfg):  # a subset of a read-only (main-deduplicated or per-origin-stream) combo cannot be pooled here: no fallback
                                info |= {"valid": False, "reason": "unavailable: a strict subset of a main-deduplicated combo has no Stage A array and is not recomputable here (no fallback)"}
                                avail.setdefault(fam, {})[(ro, pt)] = info
                                continue
                            else:  # several but not all members eligible: equal-main pooling over the eligible ones (recomputed here; no Stage A array for this subset)
                                Da = tot([per[o]["D"] for o in allowed])
                                pl = {k: tot([per[o][k] * per[o]["D"] for o in allowed]) / Da for k in ("sim", "run1", "run2", "y2", "y1")}
                                row_ld = mkrow(lab_ld, pl, all(rd[o] for o in allowed), float(Da[0]), cfg) | {"n_cells": sum(ncell[o] for o in allowed), "members": list(allowed)}
                                sens_ld = mkrow_sens({o: sets[o][pt][lab] for o in allowed}, {o: per[o]["D"] for o in allowed}, row_ld) | {"n_cells": sum(ncell[o] for o in allowed)}
                                info["pooled_over"] = "the eligible members only (equal-main weighting, recomputed here; no Stage A array for this subset)"
                            fin = np.isfinite(row_ld["G"][1:]) & np.isfinite(row_ld["Y"][1:])
                            info |= {"complete_draws": int(fin.sum()), "draws": len(fin)}
                            info["readable"] = bool(row_ld["readable"])
                            info["valid"] = info["complete_draws"] >= cfg.get("ld_min_draws", 9500) and info["readable"]
                            if not info["valid"]:
                                info["reason"] = (f"unavailable: {info['complete_draws']} jointly complete draws (need {cfg.get('ld_min_draws', 9500)}), readable {info['readable']} "
                                                  "(no fallback)")
                            for o in allowed:  # a per-origin row is the same in every combo: the first combo holding it (the headline combo first) writes it
                                if lab_ld not in sets[o][pt]:
                                    sets[o][pt][lab_ld] = (row_ld if restr else sets[o][pt][lab]) | {"ld_info": info}
                            sets[ps][pt][lab_ld] = row_ld | {"ld_info": info}
                            sets[ss][pt][lab_ld] = sens_ld | {"ld_info": info}
                        else:
                            info |= {"valid": False, "reason": ("unavailable: partially eligible set(s) " + ", ".join(part_) + " need restricted per-draw inputs (no fallback)")
                                     if part_ else "unavailable: no eligible set" + (" among this row's members of this combo" if combos else "")}
                        avail.setdefault(fam, {})[(ro, pt)] = info
                if skip_rest:
                    continue
                # gate: draw 0 equals the D-weighted per-origin G and Y of the previous pooled record (command rows), else of this run's per-origin draw 0
                pv = (cfg.get("prev") or {}).get("results", {})
                inprev = bool(pv) and lab in pv[org[0]][pt]["rows"]
                for cb, M in mem.items():
                    ss = combo_sets(cfg, cb)[1]
                    for k in ("G", "Y"):
                        g_ = {o: (pv[o][pt]["rows"][lab][k]["point"] / 100 if inprev else sets[o][pt][lab][k][0]) for o in M}
                        e = tot([g_[o] * per[o]["D"][0] for o in M]) / tot([per[o]["D"][0] for o in M])
                        gates.append({"row": lab, "f": f, "origin": ss, "arm": f"{k} draw 0 vs {'previous record' if inprev else 'this run per-origin draw 0'}",
                                      "pass": bool(abs(sets[ss][pt][lab][k][0] - e) <= 1e-12)} | ({"combo": cb} if combos else {}))
    return sets, gates, dkeys, ld_avail, extra


def bias_stats(rows, members):
    """Signed mean, RMS of G and Y, noise-subtracted bias b = sqrt(max(0, mean G2 - mean Y2)) and its rescaled version, per draw on the same draws (points)."""
    import numpy as np

    from probe_shift.fpr_core import summ

    use = [lb for lb in members if rows[lb]["readable"]]
    if not use:
        return {"rows": [], "unavailable": "no readable rows"}
    m = lambda k: np.mean([rows[lb]["bias"][k] for lb in use], 0)
    S, MG, MY, MYr = m("S"), m("G2"), m("Y2"), m("Y2r")
    o = {"rows": use, "n": len(use), "signed_mean_sim_minus_real": pc(summ(S)),
         # companions (same-launch rule): the registered comparator (sim - run 1, same launch as sim; sim - run 2 for on-policy rows) and sim - run 2 for every row;
         # sim - mean(run 1, run 2) = (sim - run 1) + 0.5 (run 1 - run 2) carries half the run 2 launch offset
         "signed_mean_registered_comparator": pc(summ(m("S_reg"))), "signed_mean_sim_minus_run2": pc(summ(m("S_r2"))),
         "rms_G": pc(summ(np.sqrt(MG))), "rms_Y": pc(summ(np.sqrt(MY)))}
    if all("Yt" in rows[lb]["bias"] for lb in use):  # 09 checklist item 3: the group's signed rerun Y = run2 - run1 at tau_B, same draws
        o["signed_rerun_Y_tauB"] = pc(summ(m("Yt")))
    for nm, diff in (("b_hat", MG - MY), ("b_hat_rescaled", MG - MYr)):
        fin = np.isfinite(diff[1:])
        o[nm] = pc(summ(np.sqrt(np.maximum(0.0, diff)) * np.where(np.isfinite(diff), 1.0, np.nan)))
        o[f"{nm}_floor_share_of_draws"] = float(np.mean(diff[1:][fin] < 0)) if fin.any() else None
        o[f"{nm}_floor_binds_at_draw0"] = bool(diff[0] < 0)
        o[f"{nm}_ms_difference_draw0_points2"] = float(1e4 * diff[0])
    sd = {lb: {"sd_sim_minus_real": None if rows[lb]["bias"]["sd_sr"] is None else 100 * rows[lb]["bias"]["sd_sr"],
               "sd_run2_minus_run1": None if rows[lb]["bias"]["sd_rr"] is None else 100 * rows[lb]["bias"]["sd_rr"],
               "ratio": rows[lb]["bias"]["ratio"], "ratio_per_origin": rows[lb]["bias"].get("ratio_per_origin")} for lb in use}
    for v in sd.values():
        rr = [v["ratio"]] if v["ratio"] is not None else list((v["ratio_per_origin"] or {}).values())
        v["ratio_material"] = any(not (0.8 <= q <= 1.25) for q in rr if np.isfinite(q))
    o["noise_check"] = sd
    return o


def add_section_gate(new, old, sections=("bias",)):
    """Gate for code changes that only ADD a section: every existing section of the earlier record is reproduced exactly by key, and the only new sections are `sections`."""
    dump = lambda x: json.dumps(x, sort_keys=True, default=float)
    bad = []
    if set(new) != set(old) or any(set(new[s_]) != set(by) for s_, by in old.items()):
        bad.append("set or point keys differ")
    n = 0
    for s_, by in old.items():
        for pt, r in by.items():
            nr = new.get(s_, {}).get(pt, {})
            if set(nr) - set(r) != set(sections) or set(r) - set(nr):
                bad.append(f"{s_}/{pt}: sections {sorted(set(nr) ^ set(r))}")
            for sec in r:
                n += 1
                if dump(nr.get(sec)) != dump(r[sec]):
                    bad.append(f"{s_}/{pt}/{sec}")
    return {"mode": "add_section", "sections_checked": n, "new_sections": list(sections), "failures": bad[:50], "n_failures": len(bad), "pass": n > 0 and not bad}


def registered_gate(new, old, ref, bias_ref=None):
    """Gate for 09's narrowing (registered G for every row; the separate on-policy rows lead with it, run2-only beside) on v8's inputs: rows, groups, readout
    contrasts and stress equal the pre-v7 record `old` (v6: registered G everywhere) by key once label fields are set aside; the separate rows' run2-only G and
    signed sim - run 2 equal the v8 record `ref` by key, and their registered G equals the row's G; coverage non-empty, key sets equal."""
    dump = lambda x: json.dumps(x, sort_keys=True, default=float)
    ign = ("headline", "name", "sensitivity_label", "g_definition", "members_requested", "members_missing")
    newg = ("offpol5_plus_own_all", "offpol5_plus_own_all_own_frac100")
    bad, n = [], {"rows": 0, "groups": 0, "contrasts": 0, "stress": 0, "onpolicy_separate": 0} | ({"bias_rows": 0, "bias_groups": 0} if bias_ref is not None else {})
    if set(new) != set(old) or any(set(new[s_]) != set(by) for s_, by in old.items()):
        bad.append("set or point keys differ")
    for s_, by in old.items():
        for pt, r in by.items():
            nr = new.get(s_, {}).get(pt, {})
            for sec, cnt in (("rows", "rows"), ("groups", "groups"), ("readout_contrast", "contrasts"), ("stress", "stress")):
                extra = {k for k in set(nr.get(sec, {})) - set(r[sec]) if k.split("|")[0] in newg} if sec in ("groups", "readout_contrast") else set()
                if set(nr.get(sec, {})) - extra != set(r[sec]):
                    bad.append(f"{s_}/{pt}/{sec}: key sets differ")
                if sec == "readout_contrast" and not all(g_ in nr.get(sec, {}) for g_ in newg):
                    bad.append(f"{s_}/{pt}/{sec}: an intended new group is missing")
                if sec == "groups" and not all(any(k.split("|")[0] == g_ for k in nr.get(sec, {})) for g_ in newg):
                    bad.append(f"{s_}/{pt}/{sec}: an intended new group is missing")
                for k, v in r[sec].items():
                    n[cnt] += 1
                    if dump({kk: vv for kk, vv in nr.get(sec, {}).get(k, {}).items() if kk not in ign}) != dump(v):
                        bad.append(f"{s_}/{pt}/{sec}/{k}")
            # groups with the same present members and readout must carry identical statistics (e.g. a requested member absent from the inputs)
            by_mem = {}
            for k, g in nr["groups"].items():
                by_mem.setdefault((k.split("|")[1], tuple(sorted(g["primary"].get("rows") or []))), []).append(dump(g["primary"]))
            for (sel_, mem_), dumps_ in by_mem.items():
                n["same_member_groups"] = n.get("same_member_groups", 0) + (len(dumps_) > 1)
                if len(set(dumps_)) > 1:
                    bad.append(f"{s_}/{pt}: groups with members {list(mem_)} ({sel_}) differ")
            if bias_ref is not None:  # bias rows of non-on-policy probes and bias groups without on-policy members equal the earlier bias record (v9) by key
                bo, bn = bias_ref[s_][pt]["bias"], nr["bias"]
                for lb, q in bo["rows"].items():
                    if not onpol(lb):
                        n["bias_rows"] = n.get("bias_rows", 0) + 1
                        if dump(bn["rows"].get(lb)) != dump(q):
                            bad.append(f"{s_}/{pt}/bias/rows/{lb}")
                for k, q in bo["groups"].items():
                    if not any(onpol(m) for m in q.get("rows", [])):
                        n["bias_groups"] = n.get("bias_groups", 0) + 1
                        if dump({kk: vv for kk, vv in bn["groups"].get(k, {}).items() if kk not in ("headline", "group_label")}) != dump({kk: vv for kk, vv in q.items() if kk != "group_label"}):
                            bad.append(f"{s_}/{pt}/bias/groups/{k}")
            rs = ref[s_][pt]["onpolicy_separate"]
            ns = nr.get("onpolicy_separate", {})
            if set(ns) != set(rs) or set(ns) != {lb for lb in nr["rows"] if onpol(lb)}:
                bad.append(f"{s_}/{pt}/onpolicy_separate: keys differ")
            for lb, q in rs.items():
                n["onpolicy_separate"] += 1
                w = ns.get(lb, {})
                if dump(w.get("G_run2_only")) != dump(q["abs_sim_minus_run2"]) or dump(w.get("signed_sim_minus_run2")) != dump(q["signed_sim_minus_run2"]) \
                        or dump(w.get("G_registered")) != dump(nr["rows"][lb]["G"]):
                    bad.append(f"{s_}/{pt}/onpolicy_separate/{lb}")
    covered = all(v > 0 for v in n.values())
    return {"mode": "registered_onpolicy", **n, "coverage_nonempty": covered, "failures": bad[:50], "n_failures": len(bad), "pass": covered and not bad}


def ld_gate(new, old, expect):
    """Gate for 09's sec. 2-3 (amended) against v11 on the same inputs. expect = {joiner, allowed, dyn, readouts, hg} from the rule (not from the rows).
    Every v11 entry is reproduced by key (headline/label fields aside; bias entries except the SD-fix fields); the ONLY new keys are the rule's
    expected launch-disjoint / subset rows, their separate rows, the dynamic groups and contrasts (exactly, on pooled and per-set-sensitivity sets;
    on eligible origins only for per-origin sets); tagged rows equal their reference rows (the origin's own rows with one eligible origin, the
    pooled/sensitivity rows with several); the flagged headline is the rule's group; empty coverage fails."""
    dump = lambda x: json.dumps(x, sort_keys=True, default=float)
    ign = ("headline", "name", "sensitivity_label", "launch_disjoint", "coverage_note", "shared_launch_label", "n_cells", "supersession_label")
    sd_fields = ("noise_check", "b_hat_rescaled", "b_hat_rescaled_floor_share_of_draws", "b_hat_rescaled_floor_binds_at_draw0", "b_hat_rescaled_ms_difference_draw0_points2")
    j, allowed, dyn, ros = expect.get("joiner"), expect.get("allowed") or [], expect.get("dyn") or [], expect.get("readouts") or []
    sels = list(ros) + (["R+C"] if set(ros) == {"R", "C"} else [])
    bad, n = [], {"rows": 0, "groups": 0, "contrasts": 0, "stress": 0, "onpolicy_separate": 0, "bias_rows": 0, "bias_groups": 0, "tagged_rows_equal_ref": 0, "headline_sets": 0}
    if set(new) != set(old) or any(set(new[s_]) != set(by) for s_, by in old.items()):
        bad.append("set or point keys differ")
    for s_, by in old.items():
        whole = s_ in ("pooled", "pooled_abs_per_origin")
        here = j is not None and (whole or s_ in allowed)
        exp_rows = ({f"{j} {LD_TAG} {ro}" for ro in ros} | {f"{m} {SUBSET_TAG} {ro}" for m in GROUPS[HEADLINE] for ro in ros}) if here else set()
        exp_sep = {f"{j} {LD_TAG} {ro}" for ro in ros} if here else set()
        # exact per set (Codex v12 r2): eligible sets carry every dynamic group and contrast; an ineligible set carries only the headline group's
        # "not computed" entries (its off-policy members exist there) and no contrast
        exp_groups = {f"{d}|{sel}" for d in dyn for sel in sels} if here else ({f"{expect.get('hg')}|{sel}" for sel in sels} if j is not None else set())
        exp_contr = set(dyn) if here else set()
        for pt, r in by.items():
            nr = new.get(s_, {}).get(pt, {})
            for sec, cnt, exp in (("rows", "rows", exp_rows), ("onpolicy_separate", "onpolicy_separate", exp_sep), ("stress", "stress", set()),
                                  ("groups", "groups", exp_groups), ("readout_contrast", "contrasts", exp_contr)):
                extra = set(nr.get(sec, {})) - set(r.get(sec, {}))
                if set(r.get(sec, {})) - set(nr.get(sec, {})) or extra != exp:
                    bad.append(f"{s_}/{pt}/{sec}: new keys {sorted(extra)[:6]} vs expected {sorted(exp)[:6]}")
                for k, v in r.get(sec, {}).items():
                    n[cnt] += 1
                    if dump({kk: vv for kk, vv in nr.get(sec, {}).get(k, {}).items() if kk not in ign}) != dump({kk: vv for kk, vv in v.items() if kk not in ign}):
                        bad.append(f"{s_}/{pt}/{sec}/{k}")
            avail_g = {k for k in exp_groups if nr["groups"].get(k, {}).get("primary", {}).get("rows")}  # bias groups exist for computed groups only
            for part, cnt, exp in (("rows", "bias_rows", exp_rows), ("groups", "bias_groups", avail_g)):
                extra = set(nr["bias"][part]) - set(r["bias"][part])
                if extra != exp or set(r["bias"][part]) - set(nr["bias"][part]):
                    bad.append(f"{s_}/{pt}/bias/{part}: new keys {sorted(extra)[:6]}")
                for k, v in r["bias"][part].items():
                    n[cnt] += 1
                    drop = sd_fields + ("headline", "group_label")
                    if dump({kk: vv for kk, vv in nr["bias"][part].get(k, {}).items() if kk not in drop}) != dump({kk: vv for kk, vv in v.items() if kk not in drop}):
                        bad.append(f"{s_}/{pt}/bias/{part}/{k}")
            for lb in exp_rows & set(nr.get("rows", {})):
                info = nr["rows"][lb].get("launch_disjoint") or {}
                if info.get("valid") != (info.get("complete_draws", 0) >= 9500 and bool(info.get("readable"))):
                    bad.append(f"{s_}/{pt}/rows/{lb}: validity flag inconsistent with its counts and readability")
                base = lb.replace(f" {LD_TAG}", "").replace(f" {SUBSET_TAG}", "")
                ref_ = (new[allowed[0]][pt]["rows"][base] if len(allowed) == 1 else nr["rows"][base])
                n["tagged_rows_equal_ref"] += 1
                if dump({kk: vv for kk, vv in nr["rows"][lb].items() if kk not in ign}) != dump({kk: vv for kk, vv in ref_.items() if kk not in ign}):
                    bad.append(f"{s_}/{pt}/rows/{lb}: differs from its reference row")
            hk = sorted({k.split("|")[0] for k, g in nr["groups"].items() if g.get("headline")})
            n["headline_sets"] += 1
            if hk != [expect.get("hg")]:
                bad.append(f"{s_}/{pt}: flagged headline {hk} != rule's {expect.get('hg')}")
    need = [k for k in n if not (j is None and k == "tagged_rows_equal_ref")]
    return {"mode": "ld_v12", "expect": expect, **n, "failures": bad[:50], "n_failures": len(bad), "pass": not bad and all(n[k] > 0 for k in need)}


PLANT_LABEL = "POST HOC planted check; descriptive (09 @ 6b96950cd); a pipeline test on fixed data, not calibrated power"
PLANT_RULE = ("Aswin's rule (10-10): moves by about the planted amount AND the planted estimate's own 95% interval excludes the unplanted estimate; 'about' is an "
              "IMPLEMENTATION CHOICE, not part of the rule: move / expected in [0.5, 1.5], expected = the achieved shift for the excess and the signed mean, |achieved| for b")
PLANT_PRED = {"excess_change": {3: 0.7, 5: 1.8, 8: 4.2}, "note": "09's folded-normal predictions (sigma ~ 5 points), stated for +delta and applied to -delta by symmetry; "
              "signed-mean change ~ delta; b ~ |delta|; Opus/0d analytic: delta 5 -> excess ~ +1.9"}


def plant_check(cfg, sets, ldc, members_by_sel, pz):
    """09 @ 6b96950cd: per delta (both signs) the PAIRED change, planted minus unplanted on the SAME draws, in the headline excess, the signed mean and b,
    for the headline group (pooled and per-set sensitivity), from Stage A's planted sim arms (plant455's score-level mechanism). DETECTED follows Aswin's
    rule (10-10): the estimate moves by about the planted amount (IMPLEMENTATION CHOICE: move / expected in [0.5, 1.5], expected = the achieved shift for the
    excess and the signed mean, |achieved| for b) AND the planted estimate's own 95% interval excludes the UNPLANTED estimate. The paired change excluding 0
    is a separate diagnostic column. A prediction more than 50% off is flagged "mechanism mismatch". The delta 0 arm must equal the base sim. A pipeline
    test on fixed data, not calibrated power (the shift is a constant solved at draw 0 and reused in every draw)."""
    import numpy as np

    from probe_shift.fpr_core import summ

    P = cfg["plant"]
    synth = bool(P.get("synthetic_TEST"))
    if sorted(P["deltas"]) != [-8, -5, -3, 3, 5, 8]:  # 09: delta 3, 5, 8, both signs (Codex plant r1)
        raise ValueError(f"plant deltas {P['deltas']} != the pre-stated +-3, +-5, +-8")
    if not members_by_sel or not all(members_by_sel.values()):
        raise ValueError("plant check: empty headline membership")
    if ldc.get("restricted"):
        raise ValueError("plant check: restricted launch-disjoint rows are not supported (no restricted planted arms)")
    if cfg.get("headline_combo") in (cfg.get("dedup_combos") or []):
        raise ValueError("plant check: a main-deduplicated headline combo has no additive planted pooling here (not supported)")
    gates, out = [], {"label": PLANT_LABEL + ("; SYNTHETIC catch-level TEST (not the registered mechanism)" if synth else ""), "predictions": PLANT_PRED, "arms": P["deltas"]}

    arms_meta = cfg.get("plant_arms_resolved")  # per row, origin, level and arm: Stage A's status and ACHIEVED shift (points), from the planted records
    rm_ = bool(cfg.get("signed_runmean_all"))  # 09 checklist item 2 (2e): the signed term is the run-mean Delta_full for every member, the on-policy row included

    def arm_meta(lab_base, f, o, d):
        if synth or arms_meta is None:
            return {"status": "ok", "achieved_points": float(d)}
        m_ = ((arms_meta.get(lab_base) or {}).get(o) or {}).get(f, {}).get(f"{d:+g}")
        if m_ is None:
            raise KeyError(f"planted arm not stated in the planted record: {lab_base} {f} {o} {d:+g}")
        return m_

    def sim_arm(lab_base, f, o, d, base):
        if synth:
            return base if d == 0 else np.clip(base + d / 100.0, 0.0, 1.0)
        if arm_meta(lab_base, f, o, d)["status"] != "ok":  # unattainable at draw 0 (the step function cannot reach it): never filled in
            return None
        for tag in (f"{d:+g}",) + (("0", "+0", "-0") if d == 0 else ()):
            k_ = (f"{san2(lab_base)}__" + (f"{san2(cfg['headline_combo'])}__" if P.get("key_format") == "v2c" else "") +
                  f"{cfg.get('mode', 'joint')}__{f}__{san2(o)}__catch_sim_plant_{tag}")
            for z_ in pz:
                if k_ in z_:
                    return z_[k_]
        raise KeyError(f"planted arm missing: {lab_base} {f} {o} {d:+g}")

    pct = lambda s_: {**s_, "point": 100 * s_["point"], **({"ci95": [100 * s_["ci95"][0], 100 * s_["ci95"][1]]} if s_.get("ci95") else {})}

    def fold5(a, sig=5.0):  # 09's folded-normal excess change at shift a (points): E|N(a, sig^2)| - E|N(0, sig^2)| (reproduces 0.7 / 1.8 / 4.2 at 3 / 5 / 8)
        from math import erf, exp, pi, sqrt

        return sig * sqrt(2 / pi) * exp(-a * a / (2 * sig * sig)) + a * erf(a / (sig * sqrt(2))) - sig * sqrt(2 / pi)
    for f, pt in zip(cfg["f_keys"], cfg["points"]):
        for sel, mem in members_by_sel.items():
            for sname in ("pooled", "pooled_abs_per_origin"):
                rows = sets[sname][pt]
                base_G = np.mean([rows[lb]["G"] for lb in mem], 0)
                base_Y = np.mean([rows[lb]["Y"] for lb in mem], 0)
                base_S = np.mean([rows[lb]["bias"]["S"] for lb in mem], 0)
                base_b = np.sqrt(np.maximum(0.0, np.mean([rows[lb]["bias"]["G2"] for lb in mem], 0) - np.mean([rows[lb]["bias"]["Y2"] for lb in mem], 0)))
                for d in [0] + [x for x in P["deltas"] if x != 0]:
                    Gs, Ss, G2s, unatt, ach = [], [], [], [], []
                    for lb in mem:
                        lab_base = lb.replace(f" {LD_TAG}", "").replace(f" {SUBSET_TAG}", "")
                        origins = (sets["pooled"][pt][lb]["ld_info"]["allowed_origins"] if (LD_TAG in lb or SUBSET_TAG in lb)  # the row's own (headline combo) members
                                   else sets["pooled"][pt][lab_base]["members"])
                        per = {}
                        for o in origins:
                            ro = sets[o][pt][lab_base]
                            sp = sim_arm(lab_base, f, o, d, ro["sim_arr"])
                            if sp is None:
                                unatt.append(f"{lab_base} {o}")
                                continue
                            if d == 0:
                                gates.append({"row": lab_base, "f": f, "origin": o, "check": "delta 0 arm equals the base sim", "pass": bool(np.array_equal(sp, ro["sim_arr"], equal_nan=True))})
                            dl = sp - ro["sim_arr"]
                            d1, d2 = ro["d1"] + dl, ro["d2"] + dl
                            per[o] = {"G": 0.5 * np.abs(d1) + 0.5 * np.abs(d2), "S": (d2 if onpol(lab_base) and not rm_ else 0.5 * (d1 + d2)), "G2": 0.5 * (d1 ** 2 + d2 ** 2), "dl": dl, "D": ro["D_arr"]}
                        if any(f"{lab_base} {o}" in unatt for o in origins):
                            continue
                        D0_ = {o: float(per[o]["D"][0]) for o in origins}
                        ach.append(sum(arm_meta(lab_base, f, o, d)["achieved_points"] * D0_[o] for o in origins) / sum(D0_.values()))  # the row's achieved shift at draw 0
                        Dd = sum(per[o]["D"] for o in origins)
                        if sname == "pooled":
                            r_ = rows[lb]
                            dlp = sum(per[o]["dl"] * per[o]["D"] for o in origins) / Dd if len(origins) > 1 else per[origins[0]]["dl"]
                            d1, d2 = r_["d1"] + dlp, r_["d2"] + dlp
                            Gs.append(0.5 * np.abs(d1) + 0.5 * np.abs(d2))
                            Ss.append(d2 if onpol(lab_base) and not rm_ else 0.5 * (d1 + d2))
                            G2s.append(0.5 * (d1 ** 2 + d2 ** 2))
                        else:
                            w = (lambda k: sum(per[o][k] * per[o]["D"] for o in origins) / Dd) if len(origins) > 1 else (lambda k: per[origins[0]][k])  # noqa: B023
                            Gs.append(w("G"))
                            Ss.append(w("S"))
                            G2s.append(w("G2"))
                    if unatt:  # a member's arm is unattainable: this delta has no value here (labelled, never filled in)
                        out.setdefault(pt, {}).setdefault(sname, {}).setdefault(sel, {})[f"{d:+g}"] = {
                            "unattainable": sorted(set(unatt)), "label": "unattainable: the draw-0 sim catch cannot move by this delta (step function at the threshold); not filled in"}
                        continue
                    G_, S_ = np.mean(Gs, 0), np.mean(Ss, 0)
                    b_ = np.sqrt(np.maximum(0.0, np.mean(G2s, 0) - np.mean([rows[lb]["bias"]["Y2"] for lb in mem], 0)))
                    if d == 0:  # the delta 0 recompute must equal the base statistics exactly (or to float rounding on the pooled sums)
                        for nm_, a_, b0 in (("excess", G_ - base_Y, base_G - base_Y), ("signed", S_, base_S), ("b", b_, base_b)):
                            gates.append({"row": f"{sel}|{sname}", "f": f, "check": f"delta 0 {nm_} recompute", "pass": bool(np.allclose(a_, b0, rtol=0, atol=1e-12, equal_nan=True))})
                        out.setdefault(pt, {}).setdefault(sname, {}).setdefault(sel, {})["0"] = {  # the unplanted baseline (Codex plant r1)
                            "excess": pct(summ(base_G - base_Y)), "signed": pct(summ(base_S)), "b": pct(summ(base_b)), "members": list(mem)}
                        continue
                    ch = {"excess": (G_ - base_Y) - (base_G - base_Y), "signed": S_ - base_S, "b": b_ - base_b}
                    e_ = {k: pct(summ(v)) | {"share_of_draws_above_0": float(np.mean(v[1:][np.isfinite(v[1:])] > 0)) if np.isfinite(v[1:]).any() else None} for k, v in ch.items()}
                    planted = {"excess": pct(summ(G_ - base_Y)), "signed": pct(summ(S_)), "b": pct(summ(b_))}
                    paired0 = {k: bool(v.get("ci95") and not (v["ci95"][0] <= 0 <= v["ci95"][1])) for k, v in e_.items()}  # the paired change excludes 0 (a separate column)
                    a_ = float(np.mean(ach))  # the group's ACHIEVED shift (2e: use it, not the nominal delta)
                    # Aswin's standing rule (10-10): DETECTED iff the estimate moves by about the planted amount (here: within +-50% of it, 09's mismatch tolerance;
                    # the expected move is the achieved shift for the excess and the signed mean, |achieved| for b) AND the planted estimate's own 95% interval
                    # excludes the UNPLANTED estimate
                    unp = {"excess": 100 * float((base_G - base_Y)[0]), "signed": 100 * float(base_S[0]), "b": 100 * float(base_b[0])}
                    asw = {}
                    for k_, x_ in (("excess", a_), ("signed", a_), ("b", abs(a_))):
                        ci_, u_ = planted[k_].get("ci95"), unp[k_]
                        ratio = e_[k_]["point"] / x_ if x_ else float("nan")
                        excl = bool(ci_) and not (ci_[0] <= u_ <= ci_[1])
                        marg = (None if not ci_ else (ci_[0] - u_) if u_ < ci_[0] else (u_ - ci_[1]) if u_ > ci_[1] else -min(u_ - ci_[0], ci_[1] - u_))
                        asw[k_] = {"expected_move_points": x_, "move_points": e_[k_]["point"], "move_ratio": ratio, "move_about_planted": bool(0.5 <= ratio <= 1.5),
                                   "unplanted_estimate": u_, "planted_interval_excludes_unplanted": excl, "margin_points": marg, "detected": bool(0.5 <= ratio <= 1.5) and excl}
                    det = {k_: v_["detected"] for k_, v_ in asw.items()}
                    prd = ({"excess": float(np.mean([fold5(x) for x in ach])), "signed": a_, "b_planted": abs(a_),
                            "basis": "at the ACHIEVED shifts (09's folded normal, sigma 5 points, per member row; signed = achieved; b = |achieved|)"}
                           if arms_meta is not None else {"excess": PLANT_PRED["excess_change"].get(abs(d)), "signed": float(d), "b_planted": float(abs(d)), "basis": "nominal delta"})
                    flag = {"excess": prd["excess"] is not None and abs(e_["excess"]["point"] - prd["excess"]) > 0.5 * abs(prd["excess"]),
                            "signed": abs(e_["signed"]["point"] - prd["signed"]) > 0.5 * abs(prd["signed"]),
                            "b_planted": abs(planted["b"]["point"] - prd["b_planted"]) > 0.5 * abs(prd["b_planted"])}
                    out.setdefault(pt, {}).setdefault(sname, {}).setdefault(sel, {})[f"{d:+g}"] = {
                        "achieved_points": {"group_mean": a_ if arms_meta is not None else float(d), "per_row": dict(zip([m_ for m_ in mem], ach)) if arms_meta is not None else None},
                        "paired_change": e_, "planted": planted, "detected": det, "detection_rule": PLANT_RULE,
                        "aswin_rule": asw, "paired_change_excludes_0": paired0, "prediction": prd,
                        "mechanism_mismatch_flag": {k: ("mechanism mismatch" if v else None) for k, v in flag.items()}}
    # caption rule (09 item 5): X = the smallest delta DETECTED (Aswin's rule) for BOTH signs, on the pooled set at the lead level; per readout (C leads, R+C beside)
    ds = sorted({abs(x) for x in P["deltas"] if x})
    pre_ = ("SYNTHETIC TEST (not the registered mechanism): " if synth else "") + "a pipeline test on fixed data, not calibrated power: "

    def caps(sel):
        lead = out.get(cfg["points"][0], {}).get("pooled", {}).get(sel) or {}
        dt = lambda d, k: bool(lead.get(f"{d:+g}", {}).get("detected", {}).get(k))
        a_ = lambda d: lead.get(f"{d:+g}", {}).get("achieved_points", {}).get("group_mean", d)
        achs = lambda x: f" (achieved shifts {a_(x):+.2f} / {a_(-x):+.2f} points)" if arms_meta is not None else ""
        o_ = {"unattainable_at_lead": sorted(k for k, v in lead.items() if isinstance(v, dict) and v.get("unattainable"))}
        for k, nm in (("excess", "the excess"), ("signed", "the signed mean (labelled)"), ("b", "b (labelled)")):
            X = next((d for d in ds if dt(d, k) and dt(-d, k)), None)
            t_ = (f"{nm} detects a uniform sim error of {X} points{achs(X)} in both directions" if X is not None else f"{nm} detects no tested shift in both directions")
            dir_ = {sg: [d for d in ds if dt(sg * d, k)] for sg in (1, -1)}
            part_ = lambda sg, w: (f"{w} shifts detected from {sg * dir_[sg][0]:+g} (achieved {a_(sg * dir_[sg][0]):+.2f})" if dir_[sg] else f"{w} shifts not detected")  # noqa: B023
            o_[k] = pre_ + t_ + f" (tested up to {max(ds)} points; {part_(1, 'positive')}, {part_(-1, 'negative')}; {PLANT_RULE}; no claim that larger errors are ruled out)"
        return o_

    cr_ = P.get("caption_readout", "R+C")
    out["captions_by_readout"] = {sel: caps(sel) for sel in ("C", "R+C")}
    out["caption_readout"] = cr_
    out["caption"], out["caption_signed_mean"], out["caption_b"] = (out["captions_by_readout"][cr_][k] for k in ("excess", "signed", "b"))
    out["unattainable_at_lead"] = out["captions_by_readout"][cr_]["unattainable_at_lead"]
    out["paired_change_excludes_0_definition"] = "paired planted - unplanted on the same draws excludes 0; diagnostic only, NOT Aswin's detection rule"
    out["conservative_note"] = ("the draw-0 paired change sits at or above its own resampled interval because the plant constant is solved at draw 0, so detection is "
                                "conservative; knife-edge verdicts rest on margins of about 0.1 points")
    for cp_ in out["captions_by_readout"].values():
        for k_ in ("excess", "signed", "b"):
            cp_[k_] += f"; {out['conservative_note']}"
    out["caption"], out["caption_signed_mean"], out["caption_b"] = (out["captions_by_readout"][cr_][k] for k in ("excess", "signed", "b"))
    out["calibration_label"] = ("a pipeline test on fixed data, not calibrated power: the shift is a constant solved at draw 0 and reused in every draw, so the achieved shift varies by draw; "
                                "no power reading (e.g. 1.43x the half-width) applies")
    out["gates"] = {"n": len(gates), "n_fail": sum(not g_["pass"] for g_ in gates), "failures": [g_ for g_ in gates if not g_["pass"]][:20]}
    if not gates:
        raise ValueError("plant check: no gate ran")
    return out


B_CAVEAT = "sim2 and run3 come from later launches (run3 shares sim2's launch); tasks with extra runs average across launches"
COH_LABEL = "POST HOC presentation (09 @ be3fcfac0, clarified @ eb8c1673b): a signed co-headline beside the registered noise-matched excess; S - Y is not printed (Y's sign is arbitrary)"


def coheadline(cfg, sets):
    """09 @ be3fcfac0 sec. 1 and 3, clarified @ eb8c1673b sec. 6: on the headline combo's pooled rows, per readout (commands lead, reply beside) and level,
    S = the equal-weight mean over the six headline probes of Delta_full = sim - (run1 + run2)/2 (run-mean for the on-policy probe too; sim - run2 beside as its
    labelled sensitivity) and Y = the mean over the same probes of run2 - run1 at tau_B, on the same draws (joint 95% intervals), with mean |Y| for scale; the
    off-policy-5 versions beside; S, Y and the excess restricted to the probes whose command AUROC (resolved in main from the pinned table, before any PANEL
    value) is >= the threshold (POST HOC sentence). Y_rr (Stage A's Delta_rr: run2 at tau_A - run1 at tau_B2 or tau_B) is an appendix companion."""
    import numpy as np

    from probe_shift.fpr_core import summ

    C = cfg["coheadline"]
    if cfg.get("mode", "joint") != "joint":  # 09: joint 95% intervals (Codex N-origin r1)
        raise ValueError("the signed co-headline is specified in the joint mode only")
    six, op = list(C["probes"]), C["onpolicy_probe"]
    off5 = [p_ for p_ in six if p_ != op]
    if len(six) != 6 or op not in six or sorted(off5) != sorted(OFF_HEADLINE):
        raise ValueError(f"co-headline probes {six} must be the off-policy 5 plus the on-policy probe {op}")
    au = cfg.get("coheadline_auroc_resolved")
    lead = C.get("lead_readout", "C")
    out = {"label": COH_LABEL, "probes": six, "onpolicy_probe": op, "lead_readout": lead, "set": f"pooled ({cfg.get('headline_combo') or 'the pooled set'})",
           "definitions": {"S": ("(B) all-runs (09 @ 3aaf425ef): mean over the probes of the per-task mean over admitted sim runs - the per-task mean over admitted real runs "
                                 "(Stage A's saved (B) gap), per draw; points; (A) sim - (run1 + run2)/2 beside as S_A_beside; " + B_CAVEAT
                                 if cfg.get("estimator_choice") == "B" else "mean over the probes of sim(tau_sim) - (run1(tau_B) + run2(tau_B))/2 (Delta_full), per draw; points"),
                           "excess": "the noise-matched excess is computed on (A) only (09 @ 3aaf425ef: its yardstick assumes single-run arms)" if cfg.get("estimator_choice") == "B" else None,
                           "Y": "mean over the probes of run2(tau_B) - run1(tau_B), same draws (signed rerun yardstick; its sign depends on which run is labelled 1)",
                           "mean_abs_Y": "mean over the probes of |run2(tau_B) - run1(tau_B)| (for scale)",
                           "Y_rr": ("pooled (RULE 45.1): the signed PANEL yardstick per probe, run2(tau_B) - run1(tau_B) at the pooled tau_B (Stage A's gap run2_minus_run1_at_tauB; "
                                    "y1 = run1@B on every origin), the same quantity as Y's per-probe term" if cfg.get("pooled_world") else
                                    "appendix companion: Stage A's Delta_rr per probe (run2 at tau_A - run1 at tau_B2 or tau_B), the signed form of the PANEL yardstick"),
                           "intervals": "95% joint bootstrap (the registered draws)"},
           "Y_label": f"includes {op}, whose run1 shares its training launch ({C.get('launch_ref', 'f1ea93e35')}); the off-policy-5 version beside",
           "onpolicy_sensitivity_label": f"sensitivity: {op}'s term is sim - run 2 ({C.get('onpolicy_caveat') or 'shared-launch caveat: its run 1 shares the training launch'})",
           "onpolicy_caveat": C.get("onpolicy_caveat"), "caption_prefix": C.get("caption_prefix", ""),
           "excess_definition": ("the excess subtracts the POOLED yardstick Y = |run2@tau_B - run1@tau_B| at the pooled tau_B (09 RULE 45.1; y1 = run1@B on every origin, "
                                 "no S_old exception); the same per-probe quantity as the printed mean |Y|" if cfg.get("pooled_world") else
                                 "the excess subtracts the REGISTERED yardstick Y = |Delta_rr| = |run2@tau_A - run1@tau_y1|, where tau_y1 = tau_B2 on S_old and tau_B on other "
                                 "origins (Stage A's y1_def; RULE.md: the two-real-pipeline yardstick); NOT the printed mean |Y| (run2 - run1 at tau_B)")}

    def block(rows, probes, sel, r2=False, excess=False):
        labs = [f"{p_} {sel}" for p_ in probes]
        miss = [lb for lb in labs if lb not in rows or not rows[lb]["readable"]]
        if miss:
            return {"rows": labs, "unavailable": f"missing or unreadable rows {miss} (no fallback)"}
        Sx = [rows[lb]["d2"] if (r2 and split(lb)[0] == op) else 0.5 * (rows[lb]["d1"] + rows[lb]["d2"]) for lb in labs]
        Yx = [rows[lb]["ytb"] for lb in labs]  # the real-arm run2 - run1 at tau_B (Codex pilot r3; equal to d1 - d2 wherever the sim arm is defined)
        o = {"rows": labs, "n": len(labs), "S": pc(summ(np.mean(Sx, 0))), "Y": pc(summ(np.mean(Yx, 0))), "mean_abs_Y": pc(summ(np.mean(np.abs(Yx), 0))),
             "Y_rr": pc(summ(np.mean([rows[lb]["y"] for lb in labs], 0)))}
        sat_ = [lb for lb in labs if rows[lb]["saturated"]]
        if sat_:  # 0d's recheck note: a saturated member (an arm >= the saturation threshold at draw 0) is named in the block, so 'without saturated' is legible
            o["saturated_rows"] = sat_
        if cfg.get("estimator_choice") == "B" and not r2:  # 09 @ 3aaf425ef: S from (B), (A) beside; the r2 sensitivity stays on (A)
            if any("delta_full_B" not in rows[lb] for lb in labs):
                return {"rows": labs, "unavailable": "estimator (B) selected but a member row lacks its (B) Delta_full (no fallback)"}
            o = {"S": pc(summ(np.mean([rows[lb]["delta_full_B"] for lb in labs], 0))), "S_estimator": "B", "S_A_beside": o["S"], "B_caveat": B_CAVEAT} | {k: v for k, v in o.items() if k != "S"}
        if excess:
            o["excess"] = strip(stats(rows, labs))["mean_G_minus_mean_Y"]
        return o

    hc = cfg.get("headline_combo")
    for cb in [hc] + [x for x in (C.get("combos") or []) if x != hc]:  # the headline combo first (out[pt][sel]); optional combos beside (out["other_combos"])
        if cb is not None and cb not in (cfg.get("combos") or {}):
            raise ValueError(f"co-headline combo {cb!r} is not a configured combo")
        dst = out if cb == hc else out.setdefault("other_combos", {}).setdefault(cb, {"label": (cfg.get("combo_labels") or {}).get(cb, cb), "set": combo_sets(cfg, cb)[0]})
        for f, pt in zip(cfg["f_keys"], cfg["points"]):
            rows = sets[combo_sets(cfg, cb)[0]][pt]
            for sel in [lead] + [x for x in ("C", "R") if x != lead]:
                e = {"six": block(rows, six, sel), "off5": block(rows, off5, sel), "six_onpolicy_sim_minus_run2": block(rows, six, sel, r2=True),
                     "role": "lead" if sel == lead else "beside"}
                if cb != hc:  # the registered excess on this combo beside S and Y (the headline combo's is the PANEL headline group itself)
                    e["six"]["excess"], e["off5"]["excess"] = (strip(stats(rows, b_["rows"]))["mean_G_minus_mean_Y"] if not b_.get("unavailable") else None for b_ in (e["six"], e["off5"]))
                if au:
                    e["auroc_subset"] = block(rows, au["subset"], sel, excess=True) | {"label": au["label"], "subset_by": au["basis"]}
                i36 = C.get("item36")
                if i36 and cb == hc and sel == i36["readout"]:  # 09 RULE item 36 (3): St5 (the five probes without the deception probe) and Dt (its own signed Delta_full)
                    e["St5"] = block(rows, i36["St5_probes"], sel) | {"label": I36_ST5}
                    lb_ = f"{i36['deception_probe']} {sel}"
                    r_ = rows.get(lb_)
                    use_b = cfg.get("estimator_choice") == "B"
                    e["Dt"] = ({"row": lb_, "unavailable": "missing or unreadable row (no fallback)", "label": I36_DT} if r_ is None or not r_["readable"] or (use_b and "delta_full_B" not in r_) else
                               {"row": lb_, "S_estimator": "B" if use_b else "A", "delta_full": pc(summ(r_["delta_full_B"] if use_b else 0.5 * (r_["d1"] + r_["d2"]))), "label": I36_DT})
                dst.setdefault(pt, {})[sel] = e
    if au:
        out["auroc_subset"] = {k: v for k, v in au.items()}
    if C.get("item36"):  # the item 36 names of quantities this block already holds (S4 / Y4 / E4 = the AUROC subset; Sfilt / Efilt = the registered-cohort combo)
        out["item36_names"] = I36_NAMES
    return out


def panel_groups_gate(res, ref, cfg, sets):
    """09 (2): every group of the source record's panel_groups (Stage A's computation with this script's estimand names), at this run's mode, must equal this
    run's independent recompute: member rows, n, draws, finite draws and unreadable rows exactly; points and interval ends within the tolerance stated in the
    config (rate units; this run reports points, so x100); Stage A's "degenerate" flag (np.ptp of the finite draws == 0, set where an interval exists) is
    recomputed here from the member rows' per-draw arrays and must be equal. Sets map by combo: Stage A's "pooled" / "pooled_abs_per_origin" of the headline
    combo are this run's "pooled" / "pooled_abs_per_origin"; of another combo, "<combo>" / "<combo>|per_set"; per-origin sets by name."""
    import math

    import numpy as np

    tol, mode = float(cfg["panel_groups_gate"]["tol_abs"]), cfg.get("mode", "joint")
    bad, n, seen, empty_src = [], [0], [], []
    num_eq = lambda a, b: (a is None and b is None) or (a is not None and b is not None and ((math.isnan(a) and math.isnan(b)) or abs(100 * a - b) <= 100 * tol))

    def deg_of(x):
        f_ = x[1:][np.isfinite(x[1:])]
        return bool(np.ptp(f_) == 0) if len(f_) >= 0.95 * (len(x) - 1) and len(f_) else None

    def part(q, m, rows, where):
        if not isinstance(q, dict):
            n[0] += 1
            if where.rsplit("/", 1)[-1] in ("unreadable_rows", "saturated_rows") and isinstance(q, list) and isinstance(m, list):  # label lists: membership, not order
                if len(set(q)) != len(q) or sorted(q) != sorted(m):
                    bad.append(f"{where}: Stage A {q!r} != here {m!r} (as sets)")
            elif q != m:
                bad.append(f"{where}: Stage A {q!r} != here {m!r}")
            return
        if not isinstance(m, dict):
            n[0] += 1
            bad.append(f"{where}: missing here")
            return
        arr = None
        if m.get("rows"):
            G = np.mean([rows[lb]["G"] for lb in m["rows"]], 0)
            Y = np.mean([rows[lb]["Y"] for lb in m["rows"]], 0)
            arr = {"mean_G": G, "mean_Y": Y, "mean_G_minus_mean_Y": G - Y}
        for k, v in q.items():
            n[0] += 1
            if isinstance(v, dict) and "point" in v:
                w = m.get(k)
                ok = isinstance(w, dict) and num_eq(v["point"], w.get("point")) and all(v.get(x) == w.get(x) for x in ("finite_draws", "draws"))
                ok = ok and bool(v.get("ci95")) == bool(w.get("ci95")) and (not v.get("ci95") or all(num_eq(a_, b_) for a_, b_ in zip(v["ci95"], w["ci95"])))
                if "degenerate" in v:
                    ok = ok and arr is not None and k in arr and deg_of(arr[k]) == v["degenerate"]
                if not ok:
                    bad.append(f"{where}/{k}: Stage A {v} vs here {w} (x100)" + (f"; degenerate here {deg_of(arr[k])}" if arr is not None and k in arr else ""))
            elif k == "withheld":  # a withheld group on both sides (texts differ by design; the membership lists beside it are compared as sets)
                if not (isinstance(v, str) and v and isinstance(m.get(k), str) and m.get(k)):
                    bad.append(f"{where}/{k}: Stage A withholds this group but it is not withheld here (or the reverse)")
            elif k in ("rows", "unreadable_rows", "saturated_rows") and isinstance(v, list) and isinstance(m.get(k), list):  # label lists: membership, not order
                if len(set(v)) != len(v) or sorted(v) != sorted(m[k]):  # (Stage A lists HEADLINE_SIX order, this run the npz label order; 10-16 per-set run)
                    bad.append(f"{where}/{k}: Stage A {v!r} != here {m.get(k)!r} (as sets)")
            elif v != m.get(k):
                bad.append(f"{where}/{k}: Stage A {v!r} != here {m.get(k)!r}")

    built_ = set(cfg["combos"]) if cfg.get("combos") else None  # 09 A2: compare exactly the combos this build computes; the others are listed (scope)
    skipped_ = sorted(cb for cb in ref if built_ is not None and cb not in built_)
    for cb in sorted(built_ or []):  # coverage (Codex streams r1): every combo built here must be in Stage A's panel_groups at every level of this run
        if not isinstance(ref.get(cb), dict):  # absent or malformed ancestors are gate failures, never crashes (Codex streams r4)
            bad.append(f"{cb}: built here but absent or malformed in Stage A's panel_groups")
            continue
        for s in ("pooled", "pooled_abs_per_origin"):
            by_m_ = ref[cb].get(s)
            if not isinstance(by_m_, dict) or not isinstance(by_m_.get(mode), dict):
                bad.append(f"{cb}/{s}/{mode}: built here but absent or malformed in Stage A's panel_groups")
                continue
            got_ = by_m_[mode]
            for f in cfg["f_keys"]:
                if not isinstance(got_.get(f), dict) or not got_[f] or not all(isinstance(g_, dict) for g_ in got_[f].values()):  # absent, empty or malformed (Codex streams r2)
                    bad.append(f"{cb}/{s}/{mode}/{f}: built here but absent, empty or malformed in Stage A's panel_groups")
    for cb, by_s in ref.items():
        if cb in skipped_:
            continue
        if not isinstance(by_s, dict):  # malformed branches are gate failures, never crashes (Codex streams r3)
            bad.append(f"{cb}: malformed panel_groups entry")
            continue
        for s, by_m in by_s.items():
            ms = dict(zip(("pooled", "pooled_abs_per_origin"), combo_sets(cfg, cb))).get(s, s)
            if not isinstance(by_m, dict) or ms not in res or not isinstance(by_m.get(mode), dict):
                bad.append(f"{cb}/{s}: set {ms} or mode {mode} not available here (or malformed)")
                continue
            for f, groups in by_m[mode].items():
                if f not in cfg["f_keys"]:
                    bad.append(f"{cb}/{s}/{mode}/{f}: level not in this run")
                    continue
                pt = cfg["points"][cfg["f_keys"].index(f)]
                if pt not in res[ms] or pt not in (sets.get(ms) or {}):  # 09 A2: a missing level is a gate FAILURE, never a crash
                    bad.append(f"{cb}/{s}->{ms}/{mode}/{f}: level missing here")
                    continue
                if not isinstance(groups, dict) or not groups:
                    bad.append(f"{cb}/{s}/{mode}/{f}: empty or malformed groups")
                    continue
                n_lv = n[0]
                for gk, g in groups.items():
                    seen.append(f"{cb}/{s}/{mode}/{f}/{gk}")
                    where = f"{cb}/{s}->{ms}/{mode}/{f}/{gk}"
                    if not isinstance(g, dict) or not (empty_group(g) or isinstance(g.get("primary"), dict)):  # a group body is empty (as a whole) or has its primary block
                        bad.append(f"{where}: malformed group body")
                        continue
                    if gk.startswith("stress_"):  # this script's stress section holds the primary stats only: the rest is recomputed here from the rows
                        sk = gk[len("stress_"):]
                        rows_ = sets[ms][pt]
                        mem_ = [lb for lb in rows_ if split(lb)[0] in STRESS and split(lb)[1] in {"R+C": "RC"}.get(sk.split("|")[1], sk.split("|")[1])]
                        mine = None if sk not in res[ms][pt]["stress"] else {"primary": res[ms][pt]["stress"][sk], "without_saturated": strip(stats(rows_, mem_, sat_ok=False)),
                                                                             "unreadable_rows": [lb for lb in mem_ if not rows_[lb]["readable"]]}
                    else:
                        mine = res[ms][pt]["groups"].get(gk)
                    if mine is None:
                        if empty_group(g):  # analyze() omits a group with no members; a genuinely empty source group (e.g. a full-length record's off_policy|C) is not a mismatch (Codex final r24)
                            empty_src.append(where)
                        else:
                            bad.append(f"{where}: group missing here")
                        continue
                    if not g:  # an empty body for a group that exists here compares nothing (Codex streams r3)
                        bad.append(f"{where}: empty group body for a group computed here")
                        continue
                    for k, v in g.items():  # primary / without_saturated (stat parts) and lists such as unreadable_rows
                        part(v, mine.get(k), sets[ms][pt], f"{where}/{k}")
                if n[0] == n_lv:  # every compared level makes at least one comparison (Codex streams r3)
                    bad.append(f"{cb}/{s}/{mode}/{f}: no comparison at this level")
    return {"rule": "09 (2): recomputed independently; equal to Stage A's panel_groups within the stated tolerance, else STOP", "tolerance_abs_rate_units": tol, "mode": mode,
            "groups_compared": len(seen), "empty_source_groups_absent_here": empty_src, "not_built_here_scope": skipped_, "n": n[0], "n_fail": len(bad), "failures": bad[:50], "pass": not bad and len(seen) > 0 and n[0] > 0}


def sens_label(fams, caveats):
    """SENS_LABEL, or, where per-probe caveats are configured (09's wording), the caveat of each on-policy member instead of the generic training-arm text."""
    onp = [m for m in fams if onpol(f"{m} X")]
    if not caveats or not onp:
        return SENS_LABEL
    return "sensitivity: includes on-policy rows (" + "; ".join(f"{m.replace(' ' + LD_TAG, '')}: {caveats.get(m.replace(' ' + LD_TAG, ''), 'its yardstick includes the training arm')}" for m in onp) + ")"


def analyze(rows, headline=True, hg=HEADLINE, dyn=None, dyn_labels=None, set_name=None, mains=None, sup_labels=None, pooled_set=None, caveats=None, delta_full=False, runmean=False,
            pinned_labels=None):
    allg = GROUPS | EXTRA_GROUPS | (dyn or {})
    pooled_set = set_name in ("pooled", "pooled_abs_per_origin") if pooled_set is None else pooled_set  # a pooled set (any combo) or a per-origin set
    labels_ = EXTRA_LABELS | (dyn_labels or {})

    from probe_shift.fpr_core import summ

    o = {"rows": {}, "groups": {}, "stress": {}, "readout_contrast": {}}
    for lb, r in rows.items():
        G, Y = r["G"], r["Y"]
        o["rows"][lb] = {"G": pc(summ(G)), "Y": pc(summ(Y)), "G_minus_Y": pc(summ(G - Y)), "draw0_catch": r["draw0_catch"], "readable": r["readable"], "saturated": r["saturated"],
                         "g_definition": r["g_def"]} | ({"n_cells": r["n_cells"]} if r.get("n_cells") is not None else {})
        if r["d1"] is not None:
            o["rows"][lb] |= {"sim_minus_run1": pc(summ(r["d1"])), "sim_minus_run2": pc(summ(r["d2"])), "yardstick_signed": pc(summ(r["y"])),
                              "Y_tauB": pc(summ(r["ytb"]))}  # 2e for RENDER's Fig 3: per-probe run2@tau_B - run1@tau_B (real arms), the co-headline Y's per-probe term
        if "D0" in r:
            o["rows"][lb]["D_draw0_S_old_S_new" if r["D0_origins"] == ["S_old", "S_new"] else "D_draw0_" + "_".join(san2(x) for x in r["D0_origins"])] = r["D0"]
        if "ld_info" in r:
            o["rows"][lb]["launch_disjoint"] = r["ld_info"]
        if pinned_labels and split(lb)[0].replace(f" {LD_TAG}", "").replace(f" {SUBSET_TAG}", "") in pinned_labels:  # 507-prep's pinned descriptive label on every row
            o["rows"][lb]["pinned_label"] = pinned_labels[split(lb)[0].replace(f" {LD_TAG}", "").replace(f" {SUBSET_TAG}", "")]
        if r.get("moments_fallback"):  # Stage A's verbatim fallback-moments statement and firing count per member origin (09 via 2e)
            o["rows"][lb]["moments_fallback"] = r["moments_fallback"]
        if r.get("pooled_source"):  # dedup provenance stays visible (Codex checklist r1 / dedup r1)
            o["rows"][lb]["pooled_source"] = r["pooled_source"]
        if runmean and r["d1"] is not None:  # the per-probe (A) run-mean Delta_full on every row (09 @ 2d61af835 sec. 4: the fixed 32k vs full-length comparison uses (A))
            o["rows"][lb]["delta_full_A"] = pc(summ(0.5 * (r["d1"] + r["d2"])))
        if r.get("real_line") is not None:  # [SENS shared] (2e 10-16): Stage A's pooled shared-threshold gap (sim@B - real at the pooled tau_B), union draws, gated at draw 0
            o["rows"][lb]["real_line_shared"] = pc(summ(r["real_line"])) | {"definition": "sim@B - real at the pooled tau_B (shared threshold), union-draw 95% interval; points"}
        if "delta_full_B" in r:  # 09 @ 3aaf425ef: the per-probe Delta_full of the lead uses (B) when selected; (A) is sim_minus_run1/2 beside
            o["rows"][lb] |= {"delta_full_B": pc(summ(r["delta_full_B"])), "delta_full_B_caveat": B_CAVEAT}
    for gname, fams in allg.items():
        for sel, ros in (("R", "R"), ("C", "C"), ("R+C", "RC")):
            mem = [lb for lb in rows if split(lb)[0] in fams and split(lb)[1] in ros]
            if mem and (sel != "R+C" or {split(lb)[1] for lb in mem} == {"R", "C"}):  # "R+C" only when both readouts are present
                req_tagged = [m for m in fams if LD_TAG in m]
                if req_tagged and any(f"{m} {x}" not in rows for m in req_tagged for x in ({"R+C": "RC"}.get(sel, sel))):
                    why = (f"not computed on this set: the joining probe is not eligible here ({set_name}); not the {gname} group"
                           if not pooled_set else "unavailable: the chosen probe's launch-disjoint row is missing (no fallback)")
                    o["groups"][f"{gname}|{sel}"] = {"primary": {"rows": [], "unavailable": why}, "without_saturated": {"rows": [], "unavailable": why}}
                    continue
                bad_m = [lb for lb in mem if (rows[lb].get("ld_info") or {}).get("valid") is False]
                if bad_m:
                    why = f"unavailable: member row(s) {bad_m} failed validity (no fallback)"
                    o["groups"][f"{gname}|{sel}"] = {"primary": {"rows": [], "unavailable": why}, "without_saturated": {"rows": [], "unavailable": why}}
                    continue
                inc_ = incomplete(rows, fams, ros)
                if inc_:  # 2e / 0d (10-16): a primary aggregate needs EVERY declared member (by label), present and readable; else WITHHELD, never averaged over a subset
                    why = withheld_why(inc_, len(fams) * len(ros))
                    o["groups"][f"{gname}|{sel}"] = {"primary": {"rows": [], "withheld": why}, "without_saturated": {"rows": [], "withheld": why},  # Stage A's withheld-group
                                                      "unreadable_rows": [lb for lb in inc_ if lb in rows], "incomplete_members": inc_}  # form (@ 8a15fb10f:3177; stub r1)
                    continue
                o["groups"][f"{gname}|{sel}"] = {"primary": strip(stats(rows, mem)), "without_saturated": strip(stats(rows, mem, sat_ok=False)),
                                                  "unreadable_rows": [lb for lb in mem if not rows[lb]["readable"]], "saturated_rows": [lb for lb in mem if rows[lb]["saturated"]]}
                if gname in labels_:
                    o["groups"][f"{gname}|{sel}"]["label"] = labels_[gname]
                ps_ = sorted({rows[lb]["pooled_source"] for lb in mem if rows[lb].get("pooled_source")})
                if ps_:  # dedup provenance on the group (Codex checklist r1)
                    o["groups"][f"{gname}|{sel}"]["pooled_source"] = "; ".join(ps_)
                if any(LD_TAG in m for m in fams) and not all(LD_TAG in m or SUBSET_TAG in m for m in fams):
                    o["groups"][f"{gname}|{sel}"]["coverage_note"] = (f"group mean mixes coverage: on-policy probe on its launch-disjoint sets only ({(mains or {}).get('joiner')}); "
                                                                      f"off-policy 5 on all tasks ({(mains or {}).get('off5')})")
                if any(rows[lb]["g_def"] == G_DEF["onpolicy"] for lb in mem):
                    o["groups"][f"{gname}|{sel}"]["g_definition_note"] = G_DEF["onpolicy"]
    for sel, ros in (("R", "R"), ("C", "C"), ("R+C", "RC")):
        mem = [lb for lb in rows if split(lb)[0] in STRESS and split(lb)[1] in ros]
        if mem and (sel != "R+C" or {split(lb)[1] for lb in mem} == {"R", "C"}):
            inc_ = incomplete(rows, STRESS, ros)
            o["stress"][f"c_asis|{sel}"] = {"rows": [], "unavailable": withheld_why(inc_, len(STRESS) * len(ros))} if inc_ else strip(stats(rows, mem))
    for gname, fams in allg.items():
        tagged = [m for m in fams if LD_TAG in m or SUBSET_TAG in m]
        if tagged and (any(f"{m} {x}" not in rows for m in tagged for x in "RC") or any((rows[f"{m} {x}"].get("ld_info") or {}).get("valid") is False for m in tagged for x in "RC")):
            if not any(f"{m} {x}" in rows for m in tagged for x in "RC") and not pooled_set:
                continue  # this set has none of the group's tagged rows (not eligible here): no contrast entry
            why = "unavailable: a launch-disjoint or subset member row is missing or failed validity (no fallback)"
            o["readout_contrast"][gname] = {"primary": {"families": [], "unavailable": why}, "without_saturated": {"families": [], "unavailable": why}}
            continue
        o["readout_contrast"][gname] = {"primary": readout_contrast(rows, fams), "without_saturated": readout_contrast(rows, fams, sat_ok=False)}
        if gname in labels_:
            o["readout_contrast"][gname]["label"] = labels_[gname]
        if any(rows[lb]["g_def"] == G_DEF["onpolicy"] for lb in rows if split(lb)[0] in fams):
            o["readout_contrast"][gname]["g_definition_note"] = G_DEF["onpolicy"]
    o["bias"] = {"label": BIAS_LABEL if not runmean else BIAS_LABEL_RM, "rows": {lb: bias_stats(rows, [lb]) for lb in rows},
                 "groups": {k: bias_stats(rows, o["groups"][k]["primary"]["rows"]) | {"group_label": (HEADLINE_NAME if k.split("|")[0] == HEADLINE else
                                                                                                      labels_.get(k.split("|")[0], sens_label(allg[k.split("|")[0]], caveats))),
                                                                                       "headline": k.split("|")[0] == hg}
                            for k in o["groups"] if k.split("|")[0] in BIAS_GROUPS + tuple(dyn or {}) and o["groups"][k]["primary"].get("rows")}}
    if headline:  # option (ii): name the headline group, mark groups with on-policy members as sensitivities, show on-policy rows separately
        for sec in ("groups", "readout_contrast"):
            for k, g in o[sec].items():
                gname = k.split("|")[0]
                req = allg[gname]
                present = sorted({split(lb)[0] for lb in rows if split(lb)[0] in req})
                g |= {"members_requested": list(req), "members_missing": [m for m in req if m not in present]}
                if gname == hg:
                    g |= {"headline": True, "name": HEADLINE_NAME if gname == HEADLINE else labels_.get(gname, gname)}
                elif gname == HEADLINE:
                    g["name"] = HEADLINE_NAME
                if gname != hg and req and all(LD_TAG in m or SUBSET_TAG in m for m in req):
                    g["sensitivity_label"] = "like-for-like comparison on the joining probe's launch-disjoint eligible subset (every member on the same sets)"
                elif gname != hg and any(onpol(f"{m} X") for m in req):
                    g["sensitivity_label"] = sens_label(req, caveats)
        from probe_shift.fpr_core import summ

        cav = lambda lb: (caveats or {}).get(split(lb)[0].replace(f" {LD_TAG}", ""))  # a per-probe caveat (09's wording) REPLACES the generic launch text on all its rows
        o["onpolicy_separate"] = {lb: {"G_registered": pc(summ(rows[lb]["G_reg"])), "G_registered_definition": G_DEF["default"],
                                       "G_run2_only": pc(summ(rows[lb]["G_r2"])),
                                       "G_run2_only_label": ("beside it (see the caption)" if cav(lb) else
                                                             ONPOL_R2_LABEL if LD_TAG not in lb else "beside it, for reference (run 1 is launch-disjoint on this row)"),
                                       "signed_sim_minus_run1": o["rows"][lb].get("sim_minus_run1"), "signed_sim_minus_run2": o["rows"][lb].get("sim_minus_run2"),
                                       "n_main_weight_draw0": rows[lb]["n0"], "n_cells": rows[lb].get("n_cells"),
                                       "caption": cav(lb) or (ONPOL_CAPTION if LD_TAG not in lb else f"{LD_LABEL}: run 1, run 2 and the calibration pools are launch-disjoint for this probe"),
                                       "shared_launch_label": cav(lb) or (LD_SHARED_LABEL if LD_TAG not in lb else None),
                                       "supersession_label": (sup_labels or {}).get(split(lb)[0].replace(f" {LD_TAG}", "")),
                                       "excess": ("in the headline group when this probe joins" if (cav(lb) or LD_TAG in lb) else "not reported (no run1-free real-vs-real pair)")}
                                  | (({"signed_delta_full": pc(summ(rows[lb]["delta_full_B"])), "signed_delta_full_estimator": "B",
                                       "signed_delta_full_A": pc(summ(0.5 * (rows[lb]["d1"] + rows[lb]["d2"]))) if rows[lb]["d1"] is not None else None, "B_caveat": B_CAVEAT,
                                       "lead_signed": "signed_delta_full = (B) all-runs Delta_full (09 @ 3aaf425ef), the lead; (A) sim - (run1 + run2)/2 beside; sim - run1 / sim - run2 beside"}
                                      if "delta_full_B" in rows[lb] else
                                      {"signed_delta_full": pc(summ(0.5 * (rows[lb]["d1"] + rows[lb]["d2"]))) if rows[lb]["d1"] is not None else None, "signed_delta_full_estimator": "A",
                                       "lead_signed": "signed_delta_full = sim - (run1 + run2)/2, the primary comparator (09 @ 635b14564); sim - run1 / sim - run2 beside"}) if delta_full else {})
                                  for lb in rows if onpol(lb)}
    return o


def reproduces(new, old):
    """Gate for code changes that only ADD groups: every statistic of the earlier record (all sets, points, rows, groups incl. the 8-row 'overall', stress,
    readout contrasts) is reproduced exactly, by key, once the extra groups are set aside."""
    extra = tuple(EXTRA_GROUPS)
    trim = {s_: {pt: {sec: ({k: v for k, v in r[sec].items() if k.split("|")[0] not in extra} if sec in ("groups", "readout_contrast") else r[sec]) for sec in r}
                 for pt, r in by.items()} for s_, by in new.items()}
    dump = lambda x: json.dumps(x, sort_keys=True, default=float)
    over = [f"{s_}/{pt}/groups/{k}" for s_, by in old.items() for pt, r in by.items() for k in r["groups"] if k.startswith("overall|")]
    bad_over = [k for k in over if dump(jget(trim, k.split("/", 3)[:3] + [k.split("/", 3)[3]])) != dump(jget(old, k.split("/", 3)[:3] + [k.split("/", 3)[3]]))]
    eq = dump(trim) == dump(old)
    return {"mode": "add_only", "overall_groups_checked": len(over), "overall_groups_differ": bad_over, "all_existing_keys_equal": eq, "extra_groups": list(extra),
            "pass": eq and not bad_over}


def onpolicy_gate(new, old):
    """Gate for the on-policy G definition (0d via 2e 17:5x) against a record made before it: every NON-on-policy row, every group and readout contrast without
    on-policy members, and the stress rows are reproduced exactly by key; on-policy rows keep every statistic except G and G - Y, and their new draw-0 G equals
    |sim - run 2| from the stored draw-0 catches (per origin and pooled) or the D-weighted per-origin G (sensitivity)."""
    dump = lambda x: json.dumps(x, sort_keys=True, default=float)
    drop = lambda d, ks: {k: v for k, v in d.items() if k not in ks}
    bad, n = [], {"rows_equal": 0, "onpolicy_rows": 0, "groups_equal": 0, "contrasts_equal": 0, "stress_equal": 0}
    for s_, by in old.items():
        for pt, r in by.items():
            nr = new[s_][pt]
            for lb, ro in r["rows"].items():
                rn = nr["rows"][lb]
                if not onpol(lb):
                    n["rows_equal"] += 1
                    if dump(drop(rn, ("g_definition",))) != dump(ro):
                        bad.append(f"{s_}/{pt}/rows/{lb}")
                    continue
                n["onpolicy_rows"] += 1
                if dump(drop(rn, ("G", "G_minus_Y", "g_definition"))) != dump(drop(ro, ("G", "G_minus_Y"))):
                    bad.append(f"{s_}/{pt}/rows/{lb} (non-G fields)")
                if s_ == "pooled_abs_per_origin":
                    w = rn["D_draw0_S_old_S_new"]
                    e = (w[0] * new["S_old"][pt]["rows"][lb]["G"]["point"] + w[1] * new["S_new"][pt]["rows"][lb]["G"]["point"]) / (w[0] + w[1])
                else:
                    e = 100 * abs(rn["draw0_catch"]["sim"] - rn["draw0_catch"]["run2"])
                if abs(rn["G"]["point"] - e) > 1e-9 or abs(rn["G_minus_Y"]["point"] - (rn["G"]["point"] - rn["Y"]["point"])) > 1e-9:
                    bad.append(f"{s_}/{pt}/rows/{lb} (G != |sim - run 2|)")
            for sec, key in (("groups", "rows"), ("readout_contrast", "families")):
                for gk, g in r[sec].items():
                    mem = g["primary"].get(key) or []
                    if any(onpol(m if sec == "groups" else f"{m} X") for m in mem):
                        continue
                    n["groups_equal" if sec == "groups" else "contrasts_equal"] += 1
                    if dump(nr[sec][gk]) != dump(g):
                        bad.append(f"{s_}/{pt}/{sec}/{gk}")
            for gk, g in r["stress"].items():
                n["stress_equal"] += 1
                if dump(nr["stress"][gk]) != dump(g):
                    bad.append(f"{s_}/{pt}/stress/{gk}")
    keys_same = set(new) == set(old) and all(set(new[s_]) == set(by) for s_, by in old.items()) and all(
        set(new[s_][pt][sec]) == set(r[sec]) for s_, by in old.items() for pt, r in by.items() for sec in ("rows", "groups", "readout_contrast", "stress"))
    covered = all(v > 0 for v in n.values())  # rows, on-policy rows, groups, contrasts and stress all checked (Codex v8 r1)
    return {"mode": "onpolicy_g", **n, "keys_cover_old": keys_same, "coverage_nonempty": covered, "failures": bad[:50], "n_failures": len(bad), "pass": (not bad) and keys_same and covered}


def headline_gate(new, old):
    """Gate for option (ii) against v7: every row, group, readout contrast and stress row equals v7 exactly by key once the new label fields are set aside
    (so the off-policy 5 group equals v7's group over the same 5 rows and draws); the on-policy separate rows equal their row's |sim - run 2| and signed
    sim - run 2 exactly; each headline group's draw-0 mean G and mean Y equal the mean of its member rows' points; coverage must be non-empty and key sets equal."""
    dump = lambda x: json.dumps(x, sort_keys=True, default=float)
    new_keys = ("headline", "name", "sensitivity_label")
    bad, n = [], {"rows": 0, "groups": 0, "contrasts": 0, "stress": 0, "onpolicy_separate": 0, "headline_groups": 0}
    for s_, by in old.items():
        for pt, r in by.items():
            nr = new[s_][pt]
            if set(nr.get("onpolicy_separate", {})) != {lb for lb in nr["rows"] if onpol(lb)}:
                bad.append(f"{s_}/{pt}/onpolicy_separate: keys differ from the on-policy rows")
            hk = sorted(k for k in nr["groups"] if k.split("|")[0] == HEADLINE)
            ros = {split(lb)[1] for lb in nr["rows"] if split(lb)[0] in GROUPS[HEADLINE]}
            want = sorted([f"{HEADLINE}|{x}" for x in sorted(ros)] + ([f"{HEADLINE}|R+C"] if ros == {"R", "C"} else []))
            if hk != want or not all(nr["groups"][k].get("headline") and nr["groups"][k].get("name") == HEADLINE_NAME for k in hk) or not nr["readout_contrast"].get(HEADLINE, {}).get("headline"):
                bad.append(f"{s_}/{pt}: headline groups {hk} != {want} or flags missing")
            for sec, cnt in (("rows", "rows"), ("groups", "groups"), ("readout_contrast", "contrasts"), ("stress", "stress")):
                if set(nr[sec]) != set(r[sec]):
                    bad.append(f"{s_}/{pt}/{sec}: key sets differ")
                for k, v in r[sec].items():
                    n[cnt] += 1
                    if dump({kk: vv for kk, vv in nr[sec].get(k, {}).items() if kk not in new_keys}) != dump(v):
                        bad.append(f"{s_}/{pt}/{sec}/{k}")
            for lb, q in nr.get("onpolicy_separate", {}).items():
                n["onpolicy_separate"] += 1
                if dump(q["abs_sim_minus_run2"]) != dump(nr["rows"][lb]["G"]) or dump(q["signed_sim_minus_run2"]) != dump(nr["rows"][lb].get("sim_minus_run2")):
                    bad.append(f"{s_}/{pt}/onpolicy_separate/{lb}")
            for k, g in nr["groups"].items():
                if not g.get("headline") or "mean_G" not in g["primary"]:
                    continue
                n["headline_groups"] += 1
                mem = g["primary"]["rows"]
                if any(onpol(m) for m in mem) or abs(g["primary"]["mean_G"]["point"] - sum(nr["rows"][m]["G"]["point"] for m in mem) / len(mem)) > 1e-9 \
                        or abs(g["primary"]["mean_Y"]["point"] - sum(nr["rows"][m]["Y"]["point"] for m in mem) / len(mem)) > 1e-9:
                    bad.append(f"{s_}/{pt}/groups/{k} (headline recompute)")
    covered = all(v > 0 for v in n.values()) and set(new) == set(old) and all(set(new[s_]) == set(by) for s_, by in old.items())  # sets and point keys equal (Codex v8 r1)
    return {"mode": "headline_v8", **n, "coverage_nonempty": covered, "failures": bad[:50], "n_failures": len(bad), "pass": (not bad) and covered}


def diff_table(new, old, moving=("Sol (a)", "c_asis")):
    """Every statistic in results/<set>/<pt>/<section>/<key>: new - old for point and interval ends (points). 'moving' rows are those whose label (or, for
    groups and stress, any member row) is in `moving`; the check is that every moved POINT belongs to a moving row or a group containing one."""
    tab, moved_pt, moved_ci = [], [], []
    for s_, by in new.items():
        for pt, r in by.items():
            for sec in ("rows", "groups", "stress", "readout_contrast"):
                for k, v in r[sec].items():
                    o_ = old.get(s_, {}).get(pt, {}).get(sec, {}).get(k)
                    if o_ is None:
                        tab.append({"key": f"{s_}/{pt}/{sec}/{k}", "status": "absent in old record"})
                        continue
                    vv, oo = (v.get("primary", v), o_.get("primary", o_))
                    mem = vv.get("rows") or vv.get("families") or [k]
                    mov = any(m.split(" ")[0] in ("Sol", "c_asis") if isinstance(m, str) else False for m in mem) or k.split(" ")[0] in ("Sol", "c_asis")
                    for stat, q in vv.items():
                        if not (isinstance(q, dict) and "point" in q) or stat not in oo:
                            continue
                        qo = oo[stat]
                        d = {"key": f"{s_}/{pt}/{sec}/{k}/{stat}", "includes_sol_or_casis": mov, "d_point": q["point"] - qo["point"],
                             "d_ci95": [q["ci95"][0] - qo["ci95"][0], q["ci95"][1] - qo["ci95"][1]] if q.get("ci95") and qo.get("ci95") else None,
                             "finite_draws": [qo.get("finite_draws"), q.get("finite_draws")]}
                        tab.append(d)
                        if abs(d["d_point"]) > 1e-9:
                            moved_pt.append((d["key"], mov))
                        if d["d_ci95"] and max(abs(x) for x in d["d_ci95"]) > 1e-9:
                            moved_ci.append((d["key"], mov))
    return {"table": tab, "n": len(tab), "moved_points": [k for k, _ in moved_pt], "moved_intervals": [k for k, _ in moved_ci],
            "moved_points_outside_sol_casis": [k for k, m in moved_pt if not m], "moved_intervals_outside_sol_casis": [k for k, m in moved_ci if not m],
            "check_points_move_only_with_sol_or_casis": all(m for _, m in moved_pt),
            "moved_by_set": {s_: {"points": sum(k.startswith(s_ + "/") for k, _ in moved_pt), "intervals": sum(k.startswith(s_ + "/") for k, _ in moved_ci),
                                  "statistics": sum(1 for t in tab if t["key"].startswith(s_ + "/") and "d_point" in t)} for s_ in new}}


def panel_body(cfg: dict, rec_blobs: list) -> dict:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    apply_family_config(cfg)

    if cfg.get("eligibility_json") is not None and cfg.get("headline_group") == "auto_09_rule":  # chosen from provenance before any score is read
        r0 = json.loads(rec_blobs[0])["rows"]
        lb0 = next(iter(r0))
        ck = cfg.get("cohort_keys_resolved")
        if ck is not None and ldc_joiner_rows_check(r0, ck) is not None:
            m_ = ldc_joiner_rows_check(r0, ck)
            return {"GATE_FAIL": [{"cohort_keys": m_}], "n_fail": 1, "n": 1}
        drops = {}
        for b_ in rec_blobs:
            for lb_, row_ in json.loads(b_)["rows"].items():
                if onpol(lb_):
                    for o_ in cfg.get("origins", ORIGINS):
                        drops.setdefault(split(lb_)[0], {}).setdefault(o_, set()).update(row_drops(row_.get(o_)))
        coh_n = {o: int(next(r_[o]["coverage"]["cohort_cells"] for r_ in [r0[lb0]] + list(r0.values()) if "coverage" in (r_.get(o) or {}))) for o in cfg.get("origins", ORIGINS)}
        ldc = ld_choose(cfg["eligibility_json"], coh_n, origins=tuple(cfg.get("origins", ORIGINS)), cohort_keys=ck,
                        expected_uncovered=cfg.get("expected_uncovered"), drops=drops, order=tuple(cfg.get("ld_order", LD_ORDER)), required_joiner=cfg.get("required_joiner"),
                        sim_keys=cfg.get("sim_keys_resolved"), sim_keys_meta=cfg.get("sim_keys_meta"), elig_alias=cfg.get("elig_alias"),
                        excluded_origins=cfg.get("ld_origin_excluded"))
        if ldc.get("required_mismatch"):  # checked FIRST: part 1 runs only as a LOO3 record (09 @ d8cf05268)
            return {"GATE_FAIL": [{"required_joiner": ldc["required_mismatch"] + ": this record is not run (decided by availability, never by value)"}], "n_fail": 1, "n": 1}
        if ldc["mismatches"]:
            return {"GATE_FAIL": [{"eligibility": m} for m in ldc["mismatches"]], "n_fail": len(ldc["mismatches"]), "n": len(ldc["mismatches"])}
        if ldc.get("joiner") and (cfg.get("probe_alias") or {}).get(ldc["joiner"]):  # 09: the eligibility key names the probe; the rows carry its identifier
            ldc |= {"joiner_eligibility_key": ldc["joiner"], "joiner": cfg["probe_alias"][ldc["joiner"]],
                    "alias_line": f"{ldc['joiner']} (eligibility file key) = {cfg['probe_alias'][ldc['joiner']]} (amendment 2 item 8 identifier; the rows' label)"}
        if ldc.get("stop"):
            return {"GATE_FAIL": [{"coverage": m} for m in ldc["stop"]], "n_fail": len(ldc["stop"]), "n": len(ldc["stop"])}
        rr_ = cfg.get("ld_restricted_rec")
        if ldc.get("restricted") and rr_ is not None:  # the restricted inputs must exclude exactly the uncovered cells
            for o, unc in ldc["restricted"].items():
                got = sorted(rr_["meta"].get("excluded_cohort_cells", {}).get(o) or [])
                if got != sorted(unc):
                    return {"GATE_FAIL": [{"restricted_inputs": f"{o}: restricted record excludes {got}, the coverage check found {unc}"}], "n_fail": 1, "n": 1}
        if ldc["mismatches"]:
            return {"GATE_FAIL": [{"eligibility": m} for m in ldc["mismatches"]], "n_fail": len(ldc["mismatches"]), "n": len(ldc["mismatches"])}
        cfg = cfg | {"ld_choice": ldc}
    sets, gates, dkeys, ld_avail, extra = load(cfg, [json.loads(b) for b in rec_blobs])
    for rq_ in (cfg.get("required_rows") if isinstance(cfg.get("required_rows"), list) else [cfg["required_rows"]] if cfg.get("required_rows") else []):
        # Codex checklist r1: every required label must exist and be readable in every required combo's pooled set at every level, else STOP (no silent shrinking)
        for cb in rq_["combos"]:
            for pt in cfg["points"]:
                for lab in rq_["labels"]:
                    r_ = sets[combo_sets(cfg, cb)[0]].get(pt, {}).get(lab)
                    gates.append({"row": lab, "combo": cb, "pt": pt, "check": "required row present and readable", "pass": r_ is not None and bool(r_["readable"])})
    hm = headline_members(cfg)
    sgn = 0
    if cfg.get("stageA_summary_gate"):  # 0d via 2e (09 checklist): every pooled row's delta_full, sim - run1 and sim - run2 summaries (point, 95% interval, draws) equal Stage A's record
        from probe_shift.fpr_core import summ

        if not cfg.get("combos"):
            raise ValueError("stageA_summary_gate needs the all-tasks record layout (combos); the legacy pooled layout is not supported (Codex checklist r1)")
        tol_ = float(cfg["stageA_summary_gate"]["tol_abs"])
        recs_ = [json.loads(b) for b in rec_blobs]
        g0 = len(gates)

        def same_summ(a_, b_):
            ca, cb_ = a_.get("ci95"), b_.get("ci95")
            return (a_["finite_draws"] == b_["finite_draws"] and a_["draws"] == b_["draws"] and abs(a_["point"] - b_["point"]) <= tol_ and bool(ca) == bool(cb_)
                    and (not ca or (len(ca) == 2 == len(cb_) and all(abs(x - y) <= tol_ for x, y in zip(ca, cb_)))))

        for cb in cfg["combos"]:
            for f, pt in zip(cfg["f_keys"], cfg["points"]):
                for lab, row in sets[combo_sets(cfg, cb)[0]].get(pt, {}).items():
                    if LD_TAG in lab or SUBSET_TAG in lab:
                        continue
                    rec_ = next(r_ for src, r_ in zip(cfg["npz"], recs_) if lab in src["labels"])
                    pan = jget(rec_, ["pooled", "{label}", "{combo}", "{mode}", "{f}", "panel"], label=lab, combo=cb, mode=cfg.get("mode", "joint"), f=f)
                    mem_lab = list(rec_["pooled"][lab][cb].get("members") or [])  # expected per-origin blocks from the row's membership (Codex final r2)
                    for sa_set, mine_set in (("pooled", combo_sets(cfg, cb)[0]), ("pooled_abs_per_origin", combo_sets(cfg, cb)[1])) + tuple((o_, o_) for o_ in mem_lab):
                        if sa_set not in pan:
                            gates.append({"row": lab, "f": f, "combo": cb, "check": f"Stage A panel block lacks {sa_set}", "pass": False})
                            continue
                        mr = sets[mine_set][pt].get(lab)
                        if mr is None:
                            gates.append({"row": lab, "f": f, "combo": cb, "check": f"row missing here in {mine_set} (Stage A has it)", "pass": False})
                            continue
                        for nm, arr in (("G", mr["G"]), ("Y", mr["Y"]), ("G_minus_Y", mr["G"] - mr["Y"])):
                            gates.append({"row": lab, "f": f, "combo": cb, "set": mine_set, "arm": f"panel {nm} vs Stage A", "pass": bool(same_summ(summ(arr), pan[sa_set][nm]))})
                        gates.append({"row": lab, "f": f, "combo": cb, "set": mine_set, "arm": "saturated flag (a stated boolean) vs Stage A",
                                      "pass": isinstance(pan[sa_set].get("saturated"), bool) and bool(mr["saturated"]) == pan[sa_set]["saturated"]})
        for cb in (cfg.get("combos") or {}):
            ps_ = combo_sets(cfg, cb)[0]
            for f, pt in zip(cfg["f_keys"], cfg["points"]):
                for lab, row in sets[ps_].get(pt, {}).items():
                    if LD_TAG in lab or SUBSET_TAG in lab:
                        continue
                    rec_ = next(r_ for src, r_ in zip(cfg["npz"], recs_) if lab in src["labels"])
                    sa = jget(rec_, ["pooled", "{label}", "{combo}", "{mode}", "{f}", "gap"], label=lab, combo=cb, mode=cfg.get("mode", "joint"), f=f)
                    for nm, arr in (("delta_full", 0.5 * (row["d1"] + row["d2"])), ("sim_minus_run1", row["d1"]), ("sim_minus_run2", row["d2"])):
                        gates.append({"row": lab, "f": f, "combo": cb, "arm": f"{nm} summary vs Stage A record", "pass": bool(same_summ(summ(arr), sa[nm]))})
        sgn = len(gates) - g0
        if not sgn:
            raise ValueError("stageA_summary_gate ran no comparison (fail closed)")
    bad = [g for g in gates if not g["pass"]]
    if bad:
        return {"GATE_FAIL": bad[:30], "n_fail": len(bad), "n": len(gates)}
    if cfg.get("calibration_expected") == "full":  # 507-prep: no aliasing - every consumed full-length record must hold no non-empty 32k-named group (panel_groups and coheadline_signed), else STOP (Codex final r24)
        bad32 = [x for i_, b_ in enumerate(rec_blobs) for blk in ("panel_groups", "coheadline_signed") for x in nonempty_32k(json.loads(b_).get(blk) or {}, f"record[{i_}]/{blk}")]
        if bad32:
            return {"GATE_FAIL": [{"full_window_32k_named_group": x} for x in bad32[:30]], "n_fail": len(bad32), "n": len(bad32)}
    if cfg.get("gates_only"):  # loader + input gates + provenance-only rule checks; no PANEL statistic is computed (pre-launch dry run)
        return {"gates_only": {"n_gates": len(gates), "n_fail": 0, "labels": sorted({g["row"] for g in gates}), "ld_choice": cfg.get("ld_choice"),
                               "ld_rows": {f"{k}|{ro}|{pt}": {kk: vv for kk, vv in v.items() if kk != "required_arms"} for k, d_ in ld_avail.items() for (ro, pt), v in d_.items()}}
                | ({"ld_rows_other_combos": {f"{cb}|{k}|{ro}|{pt}": v for cb, a_ in extra["ld_avail_by_combo"].items() for k, d_ in a_.items() for (ro, pt), v in d_.items()}}
                   if cfg.get("combos") else {})}
    hg = cfg.get("headline_group", HEADLINE_PENDING)
    dyn, dyn_labels, rule = {}, {}, None
    if hg == "auto_09_rule":  # 09 @ 1ffffca72 sec. 3 as amended @ 24fd07866
        ldc = cfg.get("ld_choice") or {"joiner": None}
        joiner = ldc.get("joiner")
        if joiner:
            hg = f"offpol5_plus_{joiner}_launch_disjoint"
            dyn = {hg: list(GROUPS[HEADLINE]) + [f"{joiner} {LD_TAG}"], "offpol5_on_policy_eligible_subset": [f"{m} {SUBSET_TAG}" for m in GROUPS[HEADLINE]]}
            cav_j = (cfg.get("onpolicy_caveats") or {}).get(joiner)  # 09's per-probe wording replaces the inherited shared-launch text (Codex part-1 r1)
            dyn_labels = {hg: f"off-policy 5 + {joiner} ({LD_LABEL}; " + (cav_j if cav_j else f"its shared-launch rows stay beside: {LD_SHARED_LABEL}") + ")",
                          "offpol5_on_policy_eligible_subset": SUBSET_LABEL}
            ros_exp = sorted({split(lb)[1] for by in sets["pooled"].values() for lb in by if split(lb)[0] in GROUPS[HEADLINE]})
            bad_ld = {f"{ro}|{pt}": (ld_avail.get(joiner, {}).get((ro, pt)) or {}).get("reason", "unavailable: the chosen probe has no input row here (no fallback)")
                      for ro in ros_exp for pt in cfg["points"] if not (ld_avail.get(joiner, {}).get((ro, pt)) or {}).get("valid")}
            six = f"offpol5_plus_{joiner}_eligible_subset"
            dyn[six] = [f"{m} {SUBSET_TAG}" for m in GROUPS[HEADLINE]] + [f"{joiner} {LD_TAG}"]
            dyn_labels[six] = (f"like-for-like: all six probes (off-policy 5 + {joiner}) on {joiner}'s launch-disjoint eligible sets "
                               f"({', '.join(ldc.get('allowed_origins') or [])})")
        else:  # rule (c): a first-class outcome - the off-policy 5 lead alone; on-policy rows stay labelled sensitivities
            hg, bad_ld = HEADLINE, {}
        rule = {**ldc, "headline_group": hg, "headline_group_rule_outcome": (f"(a)/(b): {joiner} joins with its launch-disjoint rows" if joiner else
                                                                              "none (rule (c)): off-policy 5 alone leads; on-policy rows stay labelled sensitivities"), "joiner_row_validity": ld_avail.get(joiner, {}) and {f"{ro}|{pt}": v for (ro, pt), v in ld_avail[joiner].items()},
                "headline_status": ("unavailable (the chosen probe's row failed validity; NO fallback): " + json.dumps(bad_ld)) if bad_ld else "available",
                "eligibility": cfg.get("eligibility_meta")}
        if cfg.get("combos"):  # the headline combo decides the headline; the other combos' joiner rows are listed beside
            rule |= {"headline_combo": cfg["headline_combo"], "joiner_row_validity_other_combos": {cb: {f"{ro}|{pt}": v for (ro, pt), v in a_.get(joiner, {}).items()}
                                                                                                 for cb, a_ in extra["ld_avail_by_combo"].items()}}
    if cfg.get("extra_groups"):  # config groups beside the fixed ones (e.g. the DT rows), named as in the source record's panel_groups
        dyn = dyn | cfg["extra_groups"]
        dyn_labels = dyn_labels | {k: (cfg.get("group_labels") or {}).get(k, f"{k} (config group: {', '.join(v)})") for k, v in cfg["extra_groups"].items()}
    cbs = cfg.get("combos") or {}

    def joiner_cov(s_, rows_):  # 09 checklist item 4 (option coverage_per_set): this set's own joiner coverage; else the global allowed list (the legacy text)
        glob_ = ", ".join((cfg.get("ld_choice") or {}).get("allowed_origins") or [])
        if not cfg.get("coverage_per_set"):
            return glob_
        has_ = any(LD_TAG in lb_ for lb_ in rows_)
        if s_ in cfg.get("origins", ORIGINS):
            return s_ if has_ else ""
        al_ = {o_ for lb_, r_ in rows_.items() if LD_TAG in lb_ for o_ in (r_.get("ld_info") or {}).get("allowed_origins", [])}
        return ", ".join(o_ for o_ in cfg.get("origins", ORIGINS) if o_ in al_)
    mains_of = lambda s_: (cfg.get("mains_name", "455 (S_old 183 + S_new 272)") if s_ in ("pooled", "pooled_abs_per_origin") else
                           f"{s_.split('|')[0]} ({' + '.join(cbs[s_.split('|')[0]])})" if s_.split("|")[0] in cbs else s_)
    res = {s: {pt: analyze(rows, headline=cfg.get("headline_excludes_onpolicy", True), hg=hg, dyn=dyn, dyn_labels=dyn_labels, set_name=s, sup_labels=cfg.get("supersession_labels"),
                           mains={"off5": mains_of(s), "joiner": joiner_cov(s, rows)},
                           pooled_set=s in ("pooled", "pooled_abs_per_origin") or s.split("|")[0] in cbs, caveats=cfg.get("onpolicy_caveats"), delta_full=bool(cfg.get("onpolicy_delta_full")),
                           runmean=bool(cfg.get("signed_runmean_all")), pinned_labels=(cfg.get("labels_pinned") or {}).get("LABELS"))
               for pt, rows in sets[s].items()} for s in set_names(cfg)}
    flips = {}  # per point: rows whose signed sim-vs-real gap changes sign between origins, and whether the yardstick keeps its sign
    for pt in cfg["points"]:
        for lb in res["pooled"][pt]["rows"]:
            if not all(lb in res[o_][pt]["rows"] for o_ in hm):  # launch-disjoint / subset rows exist only on their eligible origins
                continue
            sg = {o_: {k: res[o_][pt]["rows"][lb][k]["point"] for k in ("sim_minus_run1", "sim_minus_run2", "yardstick_signed")} for o_ in hm}
            prs = [(a_, b_) for i_, a_ in enumerate(hm) for b_ in hm[i_ + 1:]]  # any pair of the headline combo's origins
            fl = any(sg[a_][k] * sg[b_][k] < 0 for a_, b_ in prs for k in ("sim_minus_run1", "sim_minus_run2"))
            flips.setdefault(pt, {})[lb] = {"signed_points": sg, "sim_gap_sign_differs_between_origins": fl,
                                            "yardstick_reverses_sign_between_origins": any(sg[a_]["yardstick_signed"] * sg[b_]["yardstick_signed"] < 0 for a_, b_ in prs)}  # a zero is not a reversal (Codex RC r1)
    plant = None
    if cfg.get("plant"):
        pz_ = None
        if not cfg["plant"].get("synthetic_TEST"):  # one npz or a list (507-prep: one per readout)
            pz_ = []
            for e_ in (cfg["plant"]["npz"] if isinstance(cfg["plant"]["npz"], list) else [cfg["plant"]["npz"]]):
                pp_ = Path("/sol") / e_["path"]
                assert hashlib.sha256(pp_.read_bytes()).hexdigest() == e_["sha256"], f"plant npz sha256 {e_['path']}"
                pz_.append(np.load(pp_))
        hk_ = [k for k in res["pooled"][cfg["points"][0]]["groups"] if k.split("|")[0] == hg]
        mbs = {k.split("|")[1]: res["pooled"][cfg["points"][0]]["groups"][k]["primary"].get("rows") or [] for k in hk_}
        plant = plant_check(cfg, sets, cfg.get("ld_choice") or {}, mbs, pz_)  # unfiltered: an empty readout membership must STOP inside (Codex plant r1)
        if plant["gates"]["n_fail"]:
            return {"GATE_FAIL": plant["gates"]["failures"], "n_fail": plant["gates"]["n_fail"], "n": plant["gates"]["n"]}
    if cfg.get("calibration_expected") == "full":  # the same no-aliasing rule on this record's own groups (source records were checked before the gates_only return)
        bad32 = [f"here:{s_}/{pt_}/{gk}" for s_, by in res.items() for pt_, r_ in by.items() for sec in ("groups", "stress") for gk, g in r_[sec].items()
                 if (gk.split("|")[0] in NAMES_32K) and not empty_group(g)]
        if bad32:
            return {"GATE_FAIL": [{"full_window_32k_named_group": x} for x in bad32[:30]], "n_fail": len(bad32), "n": len(bad32)}
    pgg = None
    if cfg.get("panel_groups_gate"):  # 09 (2): recomputed independently here; must equal Stage A's panel_groups in the source record within the stated tolerance
        pgg = panel_groups_gate(res, json.loads(rec_blobs[int(cfg["panel_groups_gate"].get("record_index", 0))])["panel_groups"], cfg, sets)
        if pgg["n_fail"] or not pgg["pass"]:
            return {"GATE_FAIL": [{"panel_groups": x} for x in pgg["failures"][:30]] or [{"panel_groups": "coverage"}], "n_fail": max(pgg["n_fail"], 1), "n": pgg["n"]}
    coh = coheadline(cfg, sets) if cfg.get("coheadline") else None
    stream_ = stream_comparison(cfg, res, coh, hg, json.loads(rec_blobs[0]), sets) if cfg.get("stream_pairs") else None  # both streams per headline quantity (Aswin 10-16 via 2e; 09 B4)
    pscmp_ = None
    if (cfg.get("pooled_world") or {}).get("compare_per_set"):  # 09 #8: pooled vs the registered per-set record (item 43); its gates fail closed
        pscmp_, psg_ = per_set_compare(cfg, json.loads(rec_blobs[0]))
        if [x for x in psg_ if not x["pass"]] or pscmp_ is None:
            return {"GATE_FAIL": [x for x in psg_ if not x["pass"]][:30] or [{"check": "per-set comparison unavailable"}], "n_fail": max(1, sum(not x["pass"] for x in psg_)), "n": len(psg_)}
    sac_ = (cfg.get("coheadline") or {}).get("stageA_check")
    if coh and sac_:  # 507-prep: coheadline_signed[<combo>][<mode>][<f>]["headline_six|C"]["S" or "S|B"]; my six-probe S at C must equal it (stated tolerance), else STOP
        rj0 = json.loads(rec_blobs[0])
        for f, pt in zip(cfg["f_keys"], cfg["points"]):
            try:
                fld_ = list(sac_["field"]) + ([sac_["S_key_by_choice"][cfg.get("estimator_choice", "A")]] if sac_.get("S_key_by_choice") else [])  # "S" or "S|B" (507-prep)
                sa_ = jget(rj0, fld_, combo=cfg["headline_combo"], mode=cfg.get("mode", "joint"), f=f)
            except (KeyError, TypeError):
                sa_ = None
            mine_ = (coh.get(pt, {}).get("C") or {}).get("six", {}).get("S")
            ok_ = (isinstance(sa_, dict) and mine_ is not None and abs(mine_["point"] / 100 - sa_["point"]) <= float(sac_["tol_abs"])
                   and bool(mine_.get("ci95")) == bool(sa_.get("ci95")) and (not sa_.get("ci95") or all(abs(x / 100 - y) <= float(sac_["tol_abs"]) for x, y in zip(mine_["ci95"], sa_["ci95"]))))
            ok_ = ok_ and all(isinstance(x_.get("ci95"), list) and len(x_["ci95"]) == 2 for x_ in (mine_, sa_)) and all(mine_.get(k_) == sa_.get(k_) for k_ in ("draws", "finite_draws"))
            if not ok_:
                return {"GATE_FAIL": [{"coheadline_vs_stageA": f"six-probe S at C, f {f}: here {mine_} vs Stage A {sa_}"}], "n_fail": 1, "n": 1}
    outside = []
    for s_, by in res.items():
        for pt, r in by.items():
            for sec in ("rows", "groups", "stress"):
                for k, v in r[sec].items():
                    for kk, q in (v.get("primary", v)).items():
                        if isinstance(q, dict) and q.get("ci95") and not (q["ci95"][0] <= q["point"] <= q["ci95"][1]):
                            outside.append(f"results/{s_}/{pt}/{sec}/{k}/{'primary/' if 'primary' in v else ''}{kk}")
    # ---- figure at the lead point: pooled bars (mean G vs mean Y), and the paired difference for pooled with S_old and S_new beside it ----
    pt0 = cfg["points"][0]
    lead = res["pooled"][pt0]
    hl = cfg.get("headline_excludes_onpolicy", True)
    order = ([hg] if hg in lead_groups_names(res, pt0) and hg not in GROUPS else []) + ([HEADLINE] + [g for g in GROUPS if g != HEADLINE] if hl else list(GROUPS))
    cats = [(g, s) for g in order for s in ("R", "C", "R+C") if f"{g}|{s}" in lead["groups"] and "mean_G" in lead["groups"][f"{g}|{s}"]["primary"]]
    nice = ({"on_policy_own_family": "on-policy own\nfamily (sens.)", "off_policy": "off-policy 5" + ("\n(HEADLINE)" if hg == HEADLINE else ""), "overall": "overall 8\n(sensitivity)"} if hl
            else {"on_policy_own_family": "on-policy\nown family", "off_policy": "off-policy", "overall": "overall"})
    rdn = {"R": "reply", "C": "command", "R+C": "both"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 9.5), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(cats))
    for j, (key, col, nm) in enumerate((("mean_G", "#3b6fb6", "mean noise-matched sim-vs-real gap G (pooled)"), ("mean_Y", "#e08a2c", "mean two-real-pipeline yardstick |Y| (pooled)"))):
        v = [lead["groups"][f"{g}|{s}"]["primary"][key] for g, s in cats]
        p_ = np.array([q["point"] for q in v])
        lo = np.array([q["ci95"][0] if q.get("ci95") else q["point"] for q in v])
        hi = np.array([q["ci95"][1] if q.get("ci95") else q["point"] for q in v])
        xs = x + (j - 0.5) * 0.38
        a1.bar(xs, p_, 0.38, color=col, label=nm)
        a1.vlines(xs, lo, hi, color="black", lw=1.2)  # drawn as lines: a draw-0 point can sit outside its own percentile interval (|gap| shifts draws up)
        a1.plot(np.r_[xs, xs], np.r_[lo, hi], "_", color="black", ms=7)
        for xi, pv, h_ in zip(xs, p_, np.maximum(hi, p_)):
            a1.text(xi, h_ + 0.2, f"{pv:.1f}", ha="center", va="bottom", fontsize=7.5)
    short = lambda g: nice.get(g, ("off-policy 5\n+ " + g.replace("offpol5_plus_", "").replace("_launch_disjoint", "") + "\n(launch-disj.)") if g.endswith("_launch_disjoint") else g)
    a1.set_xticks(x, [f"{short(g)}\n{rdn[s]}\nn={lead['groups'][f'{g}|{s}']['primary']['n']}" for g, s in cats], fontsize=6.5)
    a1.set_ylabel("catch-rate gap, percentage points (lower = closer)")
    ytop_ = max((max(q["point"], q["ci95"][1] if q.get("ci95") else 0) for g, s_ in cats for q in (lead["groups"][f"{g}|{s_}"]["primary"][k_] for k_ in ("mean_G", "mean_Y"))),
                default=None)  # Codex r4 #3: every plotted group may be withheld (complete membership); then the panel says so instead of crashing
    a1.set_ylim(0, 1.15 * ytop_ if ytop_ and ytop_ > 0 else 1.0)
    if not cats:
        a1.text(0.5, 0.5, "no complete group to plot: every group is withheld (incomplete membership) or absent; see the record", ha="center", va="center",
                fontsize=9, transform=a1.transAxes)
    a1.legend(fontsize=8, loc="upper left")
    seen_ = set()
    pal = [("#3b6fb6", "s"), ("#e08a2c", "D"), ("#5a9e5a", "^"), ("#9b59b6", "P"), ("#c0504d", "X")]
    series = [("pooled", "#222222", "o")] + [(o, *pal[i % len(pal)]) for i, o in enumerate(hm)] + [("pooled_abs_per_origin", "#8a8a8a", "v")]
    for j, (s, col, mk) in enumerate(series):
        dv = [res[s][pt0]["groups"].get(f"{g}|{sel}", {}).get("primary", {}).get("mean_G_minus_mean_Y") for g, sel in cats]
        for xi, q in zip(x, dv):
            if q is None or not q.get("ci95"):
                continue
            yy = xi + (j - (len(series) - 1) / 2) * 0.21
            a2.hlines(yy, q["ci95"][0], q["ci95"][1], color=col, lw=1.2)
            a2.plot([q["point"]], [yy], mk, color=col, ms=5 if s == "pooled" else 4, label=s if s not in seen_ else None)
            seen_.add(s)
            a2.text(1.02, yy, f"{s}: {q['point']:+.1f} [{q['ci95'][0]:+.1f}, {q['ci95'][1]:+.1f}]", va="center", fontsize=6.5, color=col, transform=a2.get_yaxis_transform())
    a2.axvline(0, color="#888888", lw=1)
    a2.set_yticks(x, [f"{short(g).replace(chr(10), ' ')} | {rdn[s]}" for g, s in cats], fontsize=7.5)
    a2.invert_yaxis()
    a2.legend(fontsize=7, loc="lower left")
    a2.set_xlabel("mean G - mean Y, points (> 0: sim-vs-real gaps bigger than real-vs-real)")
    tg = (hg if hg in lead_groups_names(res, pt0) else HEADLINE) if hl else "overall"
    ovk = f"{tg}|R+C" if f"{tg}|R+C" in lead["groups"] else f"{tg}|C" if f"{tg}|C" in lead["groups"] else f"{tg}|R"
    ov = lead["groups"][ovk]["primary"].get("mean_G_minus_mean_Y")
    if ov is None or not ov.get("ci95"):  # an unavailable headline (Codex v12 r2): say so, never substitute
        title = (f"{cfg.get('pooled_title', 'Pooled 455 (S_old 183 + S_new 272)')}, per-action FPR {100 * cfg['point_values'][0]:g}%: headline group {hg} is "
                 f"{lead['groups'][ovk]['primary'].get('withheld') or lead['groups'][ovk]['primary'].get('unavailable', 'unavailable')}")
    else:
        title = (f"{cfg.get('pooled_title', 'Pooled 455 (S_old 183 + S_new 272)')}, per-action FPR {100 * cfg['point_values'][0]:g}%: per-probe sim-vs-real gap vs two-real-pipeline gap. "
                 f"{((HEADLINE_NAME if tg == HEADLINE else (GROUPS | EXTRA_GROUPS | dyn).get(tg) and (dyn_labels or {}).get(tg, tg)) + ', ' + ovk.split('|')[1] + ('' if hg == tg else ' [headline group: ' + str(hg) + ']')) if hl else ovk.replace('|', ', ')} excess {ov['point']:+.1f} points [{ov['ci95'][0]:+.1f}, {ov['ci95'][1]:+.1f}]"
                 + (" (interval includes 0)" if ov["ci95"][0] <= 0 <= ov["ci95"][1] else " (interval excludes 0)")
                 + (f"; |.| per origin: {res['pooled_abs_per_origin'][pt0]['groups'][ovk]['primary'].get('mean_G_minus_mean_Y', {}).get('point', float('nan')):+.1f}" if hl else ""))
    if cfg.get("lead_statistic") == "S" and coh:  # 09 @ 2d61af835: the signed S (selected estimator) is PRIMARY; the excess (on (A)) is the SECONDARY check
        e0_ = (coh.get(pt0) or {}).get("C") or {}
        s6_, y6_ = (e0_.get("six") or {}).get("S"), (e0_.get("six") or {}).get("Y")
        exk_ = f"{hg}|C"
        ex6_ = (lead["groups"].get(exk_) or {}).get("primary", {}).get("mean_G_minus_mean_Y")
        fz_ = lambda z: "unavailable" if not z else (f"{z['point']:+.1f} [{z['ci95'][0]:+.1f}, {z['ci95'][1]:+.1f}]" if z.get("ci95") else f"{z['point']:+.1f} (interval withheld)")
        title = (f"{cfg.get('pooled_title', 'Pooled')}, per-action FPR {100 * cfg['point_values'][0]:g}%, COMMANDS: PRIMARY signed S (six probes, "
                 f"{'(B) all-runs' if cfg.get('estimator_choice') == 'B' else '(A) run-mean'}) {fz_(s6_)} points; Y {fz_(y6_)} (for scale; no S - Y); "
                 f"SECONDARY check: the noise-matched excess (on (A)) {fz_(ex6_)}")
    if cfg.get("record_title"):  # a record whose lead is not the headline group (e.g. the AUROC-subset interim) states its own title (2e via 0d)
        title = cfg["record_title"]
    import textwrap

    grp = [lb for lb in flips[pt0] if split(lb)[0] in (GROUPS | EXTRA_GROUPS | dyn)[tg]]
    nfl = [lb for lb in grp if flips[pt0][lb]["sim_gap_sign_differs_between_origins"]]
    nyf = [lb for lb in grp if flips[pt0][lb]["yardstick_reverses_sign_between_origins"]]
    sent = []
    for g, sel in [c_ for c_ in cats if c_[0] == tg]:
        k = f"{g}|{sel}"
        ex = {s_: (res[s_][pt0]["groups"].get(k, {}).get("primary", {}).get("mean_G_minus_mean_Y") or {}).get("point") for s_ in ("pooled", *hm)}
        if any(v is None for v in ex.values()):  # e.g. the joiner is not eligible on one set: that set's value is not computed
            sent.append(f"{tg} {rdn[sel]}: pooled " + ("n/a" if ex["pooled"] is None else f"{ex['pooled']:+.1f}") + "; per set: "
                        + ", ".join(f"{s_} " + ("not computed (withheld: incomplete membership, or joiner not eligible)" if v is None else f"{v:+.1f}") for s_, v in ex.items()
                                    if s_ != "pooled"))
            continue
        all_ = "both" if len(hm) == 2 else "all"
        rel = f"below {all_}" if ex["pooled"] < min(ex[o_] for o_ in hm) else f"above {all_}" if ex["pooled"] > max(ex[o_] for o_ in hm) else "between the"
        sent.append(f"{('off-policy 5' if tg == HEADLINE else tg) if hl else 'overall'} {rdn[sel]}: pooled {ex['pooled']:+.1f}, {rel} per-set excesses ("
                    + ", ".join(f"{o_} {ex[o_]:+.1f}" for o_ in hm) + ")")
    note = (f"At draw 0, {len(nfl)} of {len(grp)} group rows reverse the sign of sim - run 1 or sim - run 2 between {jn(hm) if len(hm) == 2 else 'any two of ' + jn(hm)}, and {len(nyf)} reverse the sign of run 2 - run 1 (a zero is not a reversal) "
            f". Pooling cancels gaps that flip, so the pooled row measures the gap on the {cfg.get('mix_name', '455-task')} mix and can hide within-origin gaps: "
            + "; ".join(sent) + ". Grey (sensitivity): |.| per origin, then weighted by each draw's main weight D (removes the cancellation; a different estimand). "
            f"{sum(1 for k in outside if k.startswith('results/pooled/' + pt0))} pooled statistics at this point have draw 0 outside their own interval (listed in the record). "
            "Main weight at draw 0: " + " + ".join(f"{o_} {extra['dk0'][(cfg['npz'][0]['labels'][0], cfg['f_keys'][0], o_)]:g}" for o_ in hm if (cfg['npz'][0]['labels'][0], cfg['f_keys'][0], o_) in extra["dk0"])
            + f" for {cfg['npz'][0]['labels'][0]}. Off-policy = {OFF[0]} + {', '.join(OFF[1:])}. "
            + ("On-policy rows use the registered G; shown separately with G = |sim - run 2| beside; launch status per probe: "
               + "; ".join(f"{k}: {v}" for k, v in cfg["onpolicy_caveats"].items()) + ". " if cfg.get("onpolicy_caveats") else
               "On-policy rows (own_frac): G = |sim - run 2| (run 1 is a training arm); Y includes training arm run 1. " if cfg.get("g_onpolicy_run2_only", False)
               else "On-policy rows (own_frac) use the registered G; shown separately with G = |sim - run 2| beside (sensitivity: run1 shares the training launch). ")
            + f"Headline set = pooled; per-set values = sensitivity; {POOLING_CAVEAT}. "
            + (f"Headline group: {hg}; other groups with on-policy rows are sensitivities ({SENS_LABEL if not cfg.get('onpolicy_caveats') else 'per-probe launch status as above'}); "
               f"on-policy rows are reported separately in the record (registered G, with |sim - run 2| beside; "
               f"{ONPOL_CAPTION if not cfg.get('onpolicy_caveats') else 'per-probe captions in the record'})." if hl else ""))
    fig.suptitle(textwrap.fill(title, 200) + "\n" + textwrap.fill(cfg["pooled_label"], 230) + "\n" + textwrap.fill(note, 230) + "\n" + textwrap.fill(
        "Bars: 95% joint-bootstrap intervals (tasks resampled, thresholds re-estimated per draw); launch variance not included. Absolute values can shift bootstrap "
        "distributions of G and Y upward; read the paired difference (right). Pooled-first ordering is a POST HOC presentation choice", 230), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.84, 0.97))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    return {"res": res, "gates": {"n": len(gates), "n_fail": 0, "all_pass": True,
                                  "checked": "draw 0 of every per-origin arm equals the fork's record; pooled gaps recomputed from per-origin arms and D equal the saved arrays (all draws) and the record (draw 0)"
                                  + (f"; EXCEPT the main-deduplicated combos {cfg['dedup_combos']}: their pooled rows are read from Stage A's saved gaps, gated at draw 0 against the record, "
                                     "with an independent draw-0 recompute where the record states per-cell hits (each row's pooled_source says which)" if cfg.get("dedup_combos") else "")
                                  + (f"; and the per-origin-stream combos {cfg['stream_combos']}: read from Stage A's saved gaps, gated at draw 0, never recomputed ({STREAM_LABEL})"
                                     if cfg.get("stream_combos") else "")}
            | ({"stageA_summary_gate_comparisons": sgn} if cfg.get("stageA_summary_gate") else {}),
            "D_keys": dkeys, "draw0_outside_own_ci95": outside, "headline_rule": rule, "plant_check": plant, "panel_groups_gate": pgg, "coheadline": coh, "stream_comparison": stream_, "per_set_comparison": pscmp_, "headline_group_resolved": hg, "origin_signs": flips, "title": title, "png": buf.getvalue(),
            "diff": diff_table(res, cfg["diff_old"]["results"]) if cfg.get("diff_old") else None, "npz_collision_excluded": extra.get("npz_collision_excluded"),
            "gate_reproduces_prev": ({"add_only": reproduces, "onpolicy_g": onpolicy_gate, "headline_v8": headline_gate, "add_section": add_section_gate,
                                      "ld_v12": lambda n_, o_: ld_gate(n_, o_, {"joiner": (cfg.get("ld_choice") or {}).get("joiner"), "allowed": (cfg.get("ld_choice") or {}).get("allowed_origins"),
                                                                "dyn": sorted(dyn), "readouts": sorted({split(lb)[1] for lb in sets["pooled"][cfg["points"][0]]}), "hg": hg}), "registered_onpolicy": lambda n_, o_: registered_gate(n_, o_, cfg["check_old"]["results"], (cfg.get("check_old2") or {}).get("results"))}[cfg.get("gate_mode", "add_only")](res, cfg["diff_old"]["results"])
                                     if cfg.get("diff_old") else None)}


@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0, scaledown_window=2, volumes={"/sol": SOL})
def panel(cfg: dict, rec_blobs: list, ctx: dict | None = None) -> dict:
    """panel_body on Modal; with a launch context its return value (and the config it ran) is also saved to the Volume (exclusive name, committed), so a
    detached run whose client dropped can be re-materialised with fetch_panel_output.py (Aswin 10-16 via 2e)."""
    import copy

    cfg0 = copy.deepcopy(cfg)  # the config as launched: panel_body adds runtime objects to cfg (e.g. an open npz), which never enter the payload (Codex detach r1)
    try:
        out = panel_body(cfg, rec_blobs)
    except (Exception, SystemExit):  # the detach gap (2e): the failure itself (an error or a STOP) is saved to the Volume, then re-raised (never swallowed)
        if ctx:
            import traceback

            from probe_shift.panel_out import save_out

            save_out(SOL, "/sol", ctx["out_name"], {"context": ctx, "cfg": cfg0, "exception": traceback.format_exc()})
        raise
    if gates_only_mode(cfg0) and "GATE_FAIL" not in out:  # Codex pilot r1 #4: a pilot's (or stub's) return value AND Volume copy are gates only
        out = {k_: out[k_] for k_ in ("gates", "npz_collision_excluded") if k_ in out}
    if ctx:
        from probe_shift.panel_out import save_out

        out = out | {"volume_copy": save_out(SOL, "/sol", ctx["out_name"], {"context": ctx, "cfg": cfg0, "out": out})}
    return out


def launch_context(kind, cfg_name, stamp, git):
    """What the local side needs to write results/ later from the Volume copy: the stamp, the code at launch (git HEAD, script sha256) and the app id."""
    if not app.app_id:
        raise SystemExit("STOP: no app id at launch (the Volume copy needs one)")
    return {"kind": kind, "cfg_name": cfg_name, "stamp": stamp, "git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "app_id": app.app_id, "out_name": f"{kind}_{cfg_name}_{stamp}_{app.app_id}"}


def captions(rec):
    o = {}
    for s, by in rec["results"].items():
        for pt, r in by.items():
            for gk, g in r["groups"].items():
                p = g["primary"]
                if "mean_G" not in p:
                    continue
                d = p["mean_G_minus_mean_Y"]
                ci = f"[{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]" if d.get("ci95") else "(interval withheld)"
                o[f"results/{s}/{pt}/groups/{gk}/primary"] = (f"{s}, {gk.replace('|', ', ')} at per-action {pt}: mean G {p['mean_G']['point']:.1f} points, mean |Y| "
                                                              f"{p['mean_Y']['point']:.1f}, difference {d['point']:+.1f} {ci} (95% joint-bootstrap interval, the lead mode) over {p['n']} rows ({', '.join(p['rows'])})"
                                                              + (f"; {g['g_definition_note']}" if g.get("g_definition_note") else "")
                                                              + (f"; {g['pooled_source']}" if g.get("pooled_source") else "")
                                                              + ("; SECONDARY check (09 @ 2d61af835: the signed S is primary; the excess is on (A))" if rec.get("lead_statistic") else "")
                                                              + (f"; HEADLINE: {g['name']}" if g.get("headline") else "") + (f"; {g['sensitivity_label']}" if g.get("sensitivity_label") else ""))
            for gk, g in r["readout_contrast"].items():
                q = g["primary"]
                if "reply_excess_minus_command_excess" in q:
                    d = q["reply_excess_minus_command_excess"]
                    ci = f"[{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]" if d.get("ci95") else "(interval withheld)"
                    o[f"results/{s}/{pt}/readout_contrast/{gk}/primary"] = (f"{s}, {gk} at {pt}: reply excess minus command excess {d['point']:+.1f} {ci} (95% joint interval) over {q['n_families']} families"
                                                                            + (f"; {g['g_definition_note']}" if g.get("g_definition_note") else "")
                                                                            + (f"; HEADLINE: {g['name']}" if g.get("headline") else "")
                                                                            + (f"; {g['sensitivity_label']}" if g.get("sensitivity_label") else ""))
            for k, q in r.get("bias", {}).get("groups", {}).items():
                if "b_hat" not in q:
                    continue
                fz = lambda z: f"{z['point']:+.1f} [{z['ci95'][0]:+.1f}, {z['ci95'][1]:+.1f}]" if z.get("ci95") else f"{z['point']:+.1f} (interval withheld)"
                o[f"results/{s}/{pt}/bias/groups/{k}"] = (f"POST HOC, {s}, {k} ({q['group_label']}) at {pt}: signed mean sim - real {fz(q['signed_mean_sim_minus_real'])} points; RMS G "
                                                          f"{q['rms_G']['point']:.1f}, RMS Y {q['rms_Y']['point']:.1f}; noise-subtracted bias {fz(q['b_hat'])} (95% joint interval; floor binds in "
                                                          f"{'n/a' if q['b_hat_floor_share_of_draws'] is None else format(100 * q['b_hat_floor_share_of_draws'], '.0f') + '%'} of draws); with the yardstick rescaled by the SD ratio {fz(q['b_hat_rescaled'])} "
                                                          f"(floor {'n/a' if q['b_hat_rescaled_floor_share_of_draws'] is None else format(100 * q['b_hat_rescaled_floor_share_of_draws'], '.0f') + '%'}); assumes equal noise variance (checked per row: noise_check); "
                                                          + ("an estimate; its interval includes 0, so no established positive bias" if q["b_hat"].get("ci95") and q["b_hat"]["ci95"][0] <= 0 else "an estimate"))
            for lb, rw_ in r["rows"].items():  # a caption for every row whose members state fallback moments: Stage A's text verbatim + the firing count
                if rw_.get("moments_fallback"):
                    o[f"results/{s}/{pt}/rows/{lb}/moments_fallback"] = "; ".join(
                        f"{or_}: " + "; ".join(f"{pk_}: {tx_}" for pk_, tx_ in v_["moments_source"].items())
                        + (f" ({v_['firings']} firings" + (f", by pool {v_['firings_by_pool']}" if v_.get("firings_by_pool") else "") + ")" if isinstance(v_["firings"], int)
                           else f" ({v_['firings']})")
                        for or_, v_ in rw_["moments_fallback"].items())
            for lb, q in r.get("onpolicy_separate", {}).items():
                fmt = lambda z: "n/a" if not z else (f"{z['point']:+.1f} [{z['ci95'][0]:+.1f}, {z['ci95'][1]:+.1f}]" if z.get("ci95") else f"{z['point']:+.1f} (interval withheld)")
                o[f"results/{s}/{pt}/onpolicy_separate/{lb}"] = (f"{s}, {lb} at {pt} (on-policy, shown separately; excluded from the off-policy 5 group): "
                                                                 + ((f"signed delta_full, (B) all-runs (the lead) {fmt(q['signed_delta_full'])}; (A) sim - (run1 + run2)/2 beside {fmt(q.get('signed_delta_full_A'))}; "
                                                                     f"{q.get('B_caveat')}; " if q.get("signed_delta_full_estimator") == "B" else
                                                                     f"signed delta_full (sim - (run1 + run2)/2, the primary comparator) {fmt(q['signed_delta_full'])}; ") if "signed_delta_full" in q else "")
                                                                 + "registered G "
                                                                 f"(0.5|sim - run 1| + 0.5|sim - run 2|) {fmt(q['G_registered'])}; beside it, G = |sim - run 2| {fmt(q['G_run2_only'])} "
                                                                 f"({q['G_run2_only_label']}); signed sim - run 1 {fmt(q['signed_sim_minus_run1'])}, signed sim - run 2 "
                                                                 f"{fmt(q['signed_sim_minus_run2'])} points (95% joint intervals); n = {q.get('n_cells')} cells (from the records), main weight {'n/a (Stage A dedup row)' if q['n_main_weight_draw0'] is None else format(q['n_main_weight_draw0'], 'g')}; {q['caption']}"
                                                                 + (f"; {q['supersession_label']}" if q.get("supersession_label") else ""))
    fz = lambda z: "n/a" if not z else (f"{z['point']:+.1f} [{z['ci95'][0]:+.1f}, {z['ci95'][1]:+.1f}]" if z.get("ci95") else f"{z['point']:+.1f} (interval withheld)")
    co = rec.get("coheadline") or {}
    t_ = lambda b_: (b_.get("unavailable") if b_.get("unavailable") else
                     (f"S [(B) all-runs] {fz(b_['S'])} ((A) beside {fz(b_.get('S_A_beside'))}; {b_.get('B_caveat')})" if b_.get("S_estimator") == "B" else f"S {fz(b_['S'])}")
                     + f", Y {fz(b_['Y'])} (mean |Y| {fz(b_['mean_abs_Y'])})"
                     + ((f", excess {fz(b_['excess'])}" + (" (on (A) only)" if b_.get("S_estimator") == "B" else "")) if b_.get("excess") else ""))
    for key_, lab_, dst in [("coheadline", co.get("set"), co)] + [(f"coheadline/other_combos/{cb}", f"{v['set']}: {v['label']}", v) for cb, v in (co.get("other_combos") or {}).items()]:
        for pt, by in dst.items():
            if not isinstance(by, dict) or not by or not all(isinstance(v, dict) and "six" in v for v in by.values()):
                continue
            for sel, e in by.items():
                o[f"{key_}/{pt}/{sel}"] = (("PRIMARY (09 @ 2d61af835): " if rec.get("lead_statistic") and key_ == "coheadline" else "") + f"{co['label']}. {'COMMANDS (lead)' if sel == 'C' else 'reply'}" + (" (beside)" if e["role"] != "lead" else "") + f", per-action {pt}, {lab_}: "
                                           f"six probes ({', '.join(co['probes'])}): {t_(e['six'])} points (95% joint intervals); Y {co['Y_label']}. Off-policy 5: {t_(e['off5'])}. "
                                           f"{co['onpolicy_sensitivity_label']}: S {fz(e['six_onpolicy_sim_minus_run2'].get('S'))}")
                if e.get("auroc_subset"):
                    a_ = e["auroc_subset"]
                    o[f"{key_}/{pt}/{sel}/auroc_subset"] = (co.get("caption_prefix", "") + f"{a_['label']} ({a_['subset_by']}); {lab_}; {'COMMANDS (lead)' if sel == 'C' else 'reply'}: "
                                                            f"probes {', '.join(x.rsplit(' ', 1)[0] for x in a_['rows'])}: "
                                                            + (a_["unavailable"] if a_.get("unavailable") else t_(a_) + " points")
                                                            + (f"; {co['onpolicy_probe']}: {co['onpolicy_caveat']}" if co.get("onpolicy_caveat") and any(x.startswith(co["onpolicy_probe"] + " ") for x in a_["rows"]) else "")
                                                            + f"; {co['excess_definition']}")
    P, Q = rec["results"].get("pooled", {}), rec["results"].get("pooled_abs_per_origin", {})
    for pt, r in P.items():
        for gk, g in r["groups"].items():
            if "mean_G" not in g["primary"]:
                continue
            q = Q.get(pt, {}).get("groups", {}).get(gk, {}).get("primary", {})
            qg_ = Q.get(pt, {}).get("groups", {}).get(gk, {})
            o[f"results/headline_view/{pt}/groups/{gk}"] = (f"{gk.replace('|', ', ')} at {pt} (members present: {', '.join(g['primary']['rows'])}): pooled excess "
                                                            f"{fz(g['primary']['mean_G_minus_mean_Y'])}" + (f" ({g['pooled_source']})" if g.get("pooled_source") else "")
                                                            + f"; sensitivity (per set) {fz(q.get('mean_G_minus_mean_Y'))}" + (f" ({qg_['pooled_source']})" if qg_.get("pooled_source") else "")
                                                            + " (95% joint intervals)")
        for gk, g in r.get("bias", {}).get("groups", {}).items():
            q = Q.get(pt, {}).get("bias", {}).get("groups", {}).get(gk, {})
            if "b_hat" in g:
                o[f"results/headline_view/{pt}/bias/{gk}"] = (f"POST HOC, {gk.replace('|', ', ')} at {pt}: noise-subtracted bias pooled {fz(g['b_hat'])}, sensitivity (per set) "
                                                              f"{fz(q.get('b_hat'))}; signed mean sim - real {fz(g['signed_mean_sim_minus_real'])} (same in both; 95% joint intervals); b is an estimate"
                                                              + ("; its pooled interval includes 0, so no established positive bias" if g["b_hat"].get("ci95") and g["b_hat"]["ci95"][0] <= 0 else ""))
    return o


@app.local_entrypoint()
def main(cfg: str = "pooled455"):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    c = json.loads((HERE / "config" / f"{cfg}.json").read_text())
    if "TO_FILL" in json.dumps(c) or "TO_CONFIRM" in json.dumps(c):
        raise SystemExit(f"config {cfg} still has TO_FILL fields")
    i36_ = (c.get("coheadline") or {}).get("item36")
    if i36_:  # 09 RULE item 36: St5 = the six minus the deception probe, on a named readout; the attempt filter's sets are configured origins
        six_ = list(c["coheadline"]["probes"])
        af_ = i36_.get("attempt_filter") or {}
        want_reg = {o_: f"{o_} (registered cohort) (all-origins draws)" for o_ in ("S_old", "S_new", "S_newmains")}  # item 36 (2): the three labelled sets (Codex i36 r1)
        st5_bad = (i36_.get("deception_probe") not in six_ or str(i36_["deception_probe"]).split("|")[0] != "FIBS"
                   or sorted(i36_.get("St5_probes") or []) != sorted(p_ for p_ in six_ if p_ != i36_["deception_probe"]) or i36_.get("readout") != "R")
        if c.get("pooled_world"):  # 09 option (B) via 2e (10-16): the registered-cohort combo is DROPPED from pooled scope; item 36's St5 / Dt only, no attempt filter
            if st5_bad or "attempt_filter" in i36_ or "pooled_all_origins_registered_cohort" in set((c.get("coheadline") or {}).get("combos") or []) | set(c.get("combos") or {}):
                raise SystemExit("STOP: a pooled item 36 is St5 / Dt only (FIBS the deception probe, the other five, readout R): no attempt filter, no registered-cohort combo")
        elif (st5_bad or af_.get("registered_of") != want_reg or af_.get("unlabelled") != ["S_paper493"]
                or not all(o_ in c.get("origins", ORIGINS) for o_ in list(want_reg) + list(want_reg.values()) + ["S_paper493"])
                or "pooled_all_origins_registered_cohort" not in ((c.get("coheadline") or {}).get("combos") or [])):
            raise SystemExit("STOP: coheadline.item36 must be 09 RULE item 36: the deception probe FIBS among the six, St5 = the other five, readout R, the labelled sets "
                             "S_old/S_new/S_newmains with their registered-cohort origins, S_paper493 unlabelled, and the registered-cohort combo in the co-headline")
    if "build" in c and c.get("headline_combo") != "pooled_all_origins":  # 09 (via 2e): the lead combo of EVERY comparison, both builds, is pooled_all_origins (by name)
        raise SystemExit(f"STOP: a final build's headline combo is {c.get('headline_combo')!r}, not 'pooled_all_origins'")
    if c.get("stream_combos") or c.get("stream_pairs"):  # both streams (Aswin via 2e; 09 B4): read-only per-origin twins of union-stream (dedup) combos, both in the co-headline
        sc_, dd_ = set(c.get("stream_combos") or []), set(c.get("dedup_combos") or [])
        coh_ = set((c.get("coheadline") or {}).get("combos") or []) | {c.get("headline_combo")}
        if (not sc_ <= set(c.get("combos") or {}) or sc_ & dd_ or not c.get("stream_pairs")
                or any(p_["union"] not in dd_ or p_["per_origin"] not in sc_ or not {p_["union"], p_["per_origin"]} <= coh_ for p_ in c["stream_pairs"])):
            raise SystemExit("STOP: stream_combos must be configured, non-dedup combos; each stream pair = (a dedup union combo, a stream combo), both in the co-headline")
        wh_ = {o_ for o_, ex_ in ((c.get("cohort_keys") or {}).get("extra_origins") or {}).items() if ex_.get("withheld")}  # origins pinned WITHHELD (e.g. S_paper493 registered)
        for p_ in c["stream_pairs"]:  # a pair is the SAME cells on two streams: the union members (net of WITHHELD-pinned origins) are the per-origin members' "(all-origins draws)" rows
            if sorted(m_.replace(" (all-origins draws)", "") for m_ in c["combos"][p_["union"]] if m_ not in wh_) != sorted(c["combos"][p_["per_origin"]]):
                raise SystemExit(f"STOP: stream pair {p_} has different members on its two streams")
    blobs = []
    for src in c["npz"]:
        b = git("show", src["record"]["git"])
        if hashlib.sha256(b).hexdigest() != src["record"]["sha256"]:
            raise SystemExit(f"record sha256 {src['record']['git']}")
        blobs.append(b)
    if c.get("final_checks"):  # 09 @ 07ad588be via 2e; 507-prep's fields (meta.cohort_rule, meta.completion_rule, meta.waitlist, meta.waitlist_pending, meta.exclusions); fail closed
        fc_, fres = c["final_checks"], []
        need_ = ("cohort_rule_field", "cohort_rule_exact", "completion_rule_field", "completion_rule_exact", "waitlist_field", "waitlist_pending_field", "exclusions_field")
        if [k_ for k_ in need_ if not fc_.get(k_)]:
            raise SystemExit(f"STOP: final_checks needs every one of {need_} (fail closed; Codex checklist r1)")
        for src, b_ in zip(c["npz"], blobs):
            rj_ = json.loads(b_)
            try:
                cr_, cp_, wl_, pend_, ex_ = (jget(rj_, fc_[k_]) for k_ in ("cohort_rule_field", "completion_rule_field", "waitlist_field", "waitlist_pending_field", "exclusions_field"))
            except (KeyError, TypeError, IndexError) as e_:
                raise SystemExit(f"STOP: {src['record']['git']} lacks a final-check field ({e_!r})") from None
            if cr_ != fc_["cohort_rule_exact"] or cp_ != fc_["completion_rule_exact"]:
                raise SystemExit(f"STOP: {src['record']['git']} cohort rule {cr_!r} / completion rule {cp_!r} != {fc_['cohort_rule_exact']!r} / {fc_['completion_rule_exact']!r}")
            for nm_, ref_ in (("waitlist", wl_), ("exclusions", ex_)):
                if not isinstance(ref_, dict) or not all(ref_.get(k_) for k_ in ("commit", "path", "sha256")):
                    raise SystemExit(f"STOP: {src['record']['git']} {nm_} field {ref_!r} is not a path@sha (commit, path, sha256)")
                rb2_ = git("show", f"{ref_['commit']}:{ref_['path']}")
                if hashlib.sha256(rb2_).hexdigest() != ref_["sha256"]:
                    raise SystemExit(f"STOP: {nm_} {ref_['commit']}:{ref_['path']} sha256 != the record's {ref_['sha256']}")
            if not isinstance(pend_, list) or pend_ != []:  # the final WAITS: an EMPTY list, nothing else (Codex checklist r1)
                raise SystemExit(f"STOP: the record's waitlist_pending is {pend_!r}, not an empty list: the final waits (09 @ 07ad588be)")
            fres.append({"record": src["record"]["git"], "cohort_rule": cr_, "completion_rule": cp_, "waitlist": wl_, "exclusions": ex_, "waitlist_pending": []})
        c = c | {"final_checks_resolved": fres}
    if "estimator_choice" in c and not c.get("estimator_selection"):  # derived only (the selection block below overwrites it); without one the comparator is (A),
        raise SystemExit("STOP: estimator_choice is derived from estimator_selection, never a config field")  # never config-chosen (any value, null included: Codex r4 #4)
    if c.get("estimator_selection"):  # 09 @ 3aaf425ef: the record's meta must STATE the choice (A or B); S follows it, the excess stays on (A); else STOP
        es_ = c["estimator_selection"]
        chs_ = set()
        for src, b_ in zip(c["npz"], blobs):
            try:
                v_ = jget(json.loads(b_), es_["meta_field"])
            except (KeyError, TypeError, IndexError):
                raise SystemExit(f"STOP: {src['record']['git']} does not state the estimator choice at {es_['meta_field']} (fail closed)") from None
            if v_ not in es_["allowed"]:
                raise SystemExit(f"STOP: {src['record']['git']} estimator choice {v_!r} is not one of {list(es_['allowed'])}")
            if es_.get("rule_field") and jget(json.loads(b_), es_["rule_field"]) != es_["rule_exact"]:  # 507-prep: meta.estimator.rule = "09 @ 3aaf425ef"
                raise SystemExit(f"STOP: {src['record']['git']} estimator rule != {es_['rule_exact']!r}")
            if es_.get("selection_record_field"):  # the width-only selection record by path@sha (verified by git), its widths, and the rule's outcome (Codex final r5)
                sr_ = jget(json.loads(b_), es_["selection_record_field"])
                if not isinstance(sr_, dict) or not all(sr_.get(k_) for k_ in ("commit", "path", "sha256")):
                    raise SystemExit(f"STOP: {src['record']['git']} estimator selection record {sr_!r} is not a path@sha")
                srb_ = git("show", f"{sr_['commit']}:{sr_['path']}")
                if hashlib.sha256(srb_).hexdigest() != sr_["sha256"]:
                    raise SystemExit(f"STOP: estimator selection record {sr_['path']} sha256 != the record's")
                srj_ = json.loads(srb_)
                hw_ = {k_: jget(json.loads(b_), es_["hw_fields"][k_]) for k_ in ("A", "B")}
                hws_ = {k_: jget(srj_, es_["selection_hw_fields"][k_]) for k_ in ("A", "B")}
                import math as _m

                if float(es_.get("ratio", 0.95)) != 0.95:  # the named rule's ratio is fixed (Codex final r6)
                    raise SystemExit("STOP: estimator_selection.ratio must be 09's 0.95")
                if not all(isinstance(x_, (int, float)) and not isinstance(x_, bool) and _m.isfinite(x_) and x_ >= 0 for x_ in list(hw_.values()) + list(hws_.values())) or hw_ != hws_:
                    raise SystemExit(f"STOP: estimator widths {hw_} (record) vs {hws_} (selection record) are not finite, nonnegative and equal")
                if v_ != ("B" if hw_["B"] <= float(es_.get("ratio", 0.95)) * hw_["A"] else "A"):
                    raise SystemExit(f"STOP: the stated lead {v_!r} contradicts 09's rule HW(B) <= 0.95 x HW(A) at widths {hw_}")
                sm_ = {k_: srj_.get(k_) for k_ in ("lead", "threshold", "ratio", "rule", "coverage_counts")}  # the selection record's own decision metadata (Codex final r10)
                req_p = list((c.get("coheadline") or {}).get("probes") or [])  # coverage per probe x set (09 @ 36b9b47df sec. 3): every headline probe, every base set of the headline combo
                req_s = sorted({(c.get("origin_base") or {}).get(o_, o_) for o_ in ((c.get("combos") or {}).get(c.get("headline_combo")) or [])})
                cc_ = sm_["coverage_counts"]
                cc_ok = (isinstance(cc_, dict) and bool(req_p) and all(isinstance(cc_.get(p_), dict) and all(isinstance(cc_[p_].get(s_), int) and not isinstance(cc_[p_].get(s_), bool)
                                                                                                       and cc_[p_][s_] >= 0 for s_ in req_s) for p_ in req_p))
                rcf_ = es_.get("record_coverage_field")  # REQUIRED: Stage A's own statement of the coverage in the main record must equal the selection record's (Codex final r13)
                try:
                    if not rcf_ or jget(json.loads(b_), rcf_) != cc_:
                        cc_ok = False
                except (KeyError, TypeError, IndexError):
                    cc_ok = False
                if (not cc_ok or sm_["lead"] != v_ or sm_["threshold"] != 0.95 or not isinstance(sm_["ratio"], (int, float)) or isinstance(sm_["ratio"], bool) or not _m.isfinite(sm_["ratio"]) or hw_["A"] <= 0
                        or abs(sm_["ratio"] - hw_["B"] / hw_["A"]) > 1e-12 or sm_["rule"] != es_.get("rule_exact") or not isinstance(sm_["coverage_counts"], dict) or not sm_["coverage_counts"]):
                    raise SystemExit(f"STOP: the selection record's lead / threshold / ratio / rule / coverage_counts {sm_} do not match the rule and the record")
                c = c | {"estimator_selection_resolved": (c.get("estimator_selection_resolved") or []) + [{"source": src["record"]["git"], "lead": v_, "HW": hw_, "ratio": sm_["ratio"],
                                                                                                            "coverage_counts": sm_["coverage_counts"], "selection_record": sr_}]}
            chs_.add(es_["allowed"][v_])
        if len(chs_) != 1:
            raise SystemExit(f"STOP: the source records state different estimator choices {chs_}")
        c = c | {"estimator_choice": chs_.pop()}
    MF_LABEL = "fallback moments from out-of-fold scores (cross-fitted probe)"  # 09's exact phrase (via 2e)
    if c.get("build") and (not isinstance(c.get("moments_fallback"), dict) or c["moments_fallback"].get("label") != MF_LABEL):
        raise SystemExit(f"STOP: a final build needs moments_fallback with 09's exact label {MF_LABEL!r} (Codex final r21)")
    c.pop("labels_pinned", None)
    if c.get("build") == "full_length" and not c.get("labels_pin") and (c.get("pooled_world") or {}).get("pilot") is not True:  # mandatory for the full-length build
        # (Codex final r17); a validated PILOT (checked below: labels within the PW scope) runs without the full-scope pin (Codex pilot r1 #1)
        raise SystemExit("STOP: the full-length build needs labels_pin (507-prep's full_override.py constants by sha)")
    if c.get("build") and not c.get("drop_evidence_from_record"):
        raise SystemExit("STOP: a final build needs drop_evidence_from_record (the record-pinned fold -1 drop evidence)")
    if c.get("labels_pin"):  # 507-prep's full-length label constants (full_override.py), pinned by sha: evaluated from their top-level UPPERCASE assignments only, no imports
        import ast

        lb_ = git("show", c["labels_pin"]["git"])
        if hashlib.sha256(lb_).hexdigest() != c["labels_pin"]["sha256"]:
            raise SystemExit("STOP: the pinned label constants file sha256 != config")
        ns_ = {"__builtins__": {"tuple": tuple, "dict": dict, "list": list, "set": set, "sorted": sorted}}
        for n_ in ast.parse(lb_.decode()).body:
            if isinstance(n_, ast.Assign) and len(n_.targets) == 1 and isinstance(n_.targets[0], ast.Name) and n_.targets[0].id.isupper():
                exec(compile(ast.Module(body=[n_], type_ignores=[]), "full_override_constants", "exec"), ns_)  # noqa: S102 (pinned project constants, restricted builtins)
        lp_ = {k: ns_[k] for k in ("HEADLINE_SIX", "OFF_FIVE", "REG_SIX", "POSITIVE_CONTROL", "RETRAIN_OF", "AUROC06_FULL", "LABELS_OUT", "LABELS")}
        chk_ = [(list(c.get("off_policy_probes") or []), list(lp_["OFF_FIVE"]), "off_policy_probes"),
                (list((c.get("coheadline") or {}).get("probes") or []), list(lp_["HEADLINE_SIX"]), "coheadline.probes"),
                (list((c.get("extra_groups") or {}).get("headline_six_full") or []), list(lp_["HEADLINE_SIX"]), "headline_six_full"),
                (list((c.get("extra_groups") or {}).get("off_policy_five_full") or []), list(lp_["OFF_FIVE"]), "off_policy_five_full"),
                (list((c.get("extra_groups") or {}).get("registered_scored_uncut_six") or []), list(lp_["REG_SIX"]) if not c.get("pooled_world") else [],
                 "registered_scored_uncut_six")]  # a pooled build scores the PW scope only (no REG_SIX group)
        lab_cfg = [lb for src in c["npz"] for lb in src["labels"]]  # the consumed label inventory must equal LABELS_OUT exactly (no omitted or extra rows)
        if c.get("pooled_world"):  # pooled build (09 item 45(3)): exactly the configured pooled scope families x C/R, a subset of LABELS_OUT (checked against the record below)
            pw_sc = [f"{fa} {ro}" for fa in c["pooled_world"]["scope_families"] for ro in ("C", "R")]
            chk_.append((sorted(lab_cfg), sorted(pw_sc), "npz labels vs the pooled scope"))
            chk_.append((bool(set(pw_sc) <= set(lp_["LABELS_OUT"])), True, "pooled scope within LABELS_OUT"))
            chk_.append((sorted(c["pooled_world"]["scope_families"]), sorted(list(lp_["HEADLINE_SIX"]) + list(PW_EXTRA_FAMILIES)),
                         "pooled scope families vs the pinned HEADLINE_SIX + the registered PW additions"))  # Astra pooled r1 #5
        else:
            chk_.append((sorted(lab_cfg), sorted(lp_["LABELS_OUT"]), "npz labels vs LABELS_OUT"))
        ab_ = (c.get("extra_groups") or {}).get("appendix_B_own_all_rs_uncut")  # Appendix B = the own_all_rs retrains by RETRAIN_OF
        chk_.append((sorted(ab_ or []), sorted(k for k, v in lp_["RETRAIN_OF"].items() if v.startswith("own_all_rs")), "appendix_B_own_all_rs_uncut vs RETRAIN_OF"))
        fams_used = {lb.rsplit(" ", 1)[0] for lb in lab_cfg}
        for k_, v_ in lp_["LABELS"].items():  # every ON-POLICY pinned family in use MUST have a caveat that starts with its pinned descriptive label (Codex final r18)
            if k_ in fams_used and onpol(f"{k_} X") and not str((c.get("onpolicy_caveats") or {}).get(k_, "")).startswith(v_):
                chk_.append(((c.get("onpolicy_caveats") or {}).get(k_), v_, f"onpolicy_caveats[{k_}] missing or not starting with its pinned LABELS"))
        bad_ = [nm for mine, theirs, nm in chk_ if mine != theirs]
        if bad_ or any(p_ in g_ for g_ in (c.get("extra_groups") or {}).values() for p_ in lp_["POSITIVE_CONTROL"]):
            raise SystemExit(f"STOP: the config's {bad_} differ from the pinned constants, or a positive-control probe sits in a group")
        c = c | {"labels_pinned": {k: (list(v) if isinstance(v, tuple) else v) for k, v in lp_.items()}}
    if c.get("npz_collision_excluded"):  # 2e / 0d 10-16: only a REAL npz key collision may exclude a label, by label, with the verbatim reason, never a grouped row
        nx_ = c["npz_collision_excluded"]
        xl_ = list(nx_.get("labels") or [])
        grp_f = {m for v_ in (c.get("extra_groups") or {}).values() for m in v_} | set((c.get("coheadline") or {}).get("probes") or []) | {
            m for v_ in (GROUPS | EXTRA_GROUPS).values() for m in v_} | set(STRESS)
        bad_x = []
        kf_ = lambda s_, l_: keys_for(s_.get("key_format", "v1"), l_, "F", gap="G", combo="C")  # the label part of that npz's real key format
        for lb in xl_:  # Codex r7: only the approved labels; each collides, in ITS OWN npz under that npz's key format, with a partner that stays checked
            src_ = [s_ for s_ in c["npz"] if lb in s_["labels"]]
            part_ = [o_ for o_ in (src_[0]["labels"] if len(src_) == 1 else []) if o_ != lb and kf_(src_[0], o_) == kf_(src_[0], lb)]
            if lb not in NPZ_COLLISION_APPROVED or len(src_) != 1 or not part_ or any(p_ in xl_ for p_ in part_) or lb.rsplit(" ", 1)[0] in grp_f:
                bad_x.append(lb)
        if nx_.get("reason") != NPZ_COLLISION_REASON or not xl_ or len(set(xl_)) != len(xl_) or bad_x:
            raise SystemExit(f"STOP: npz_collision_excluded needs the verbatim reason and only approved labels ({sorted(NPZ_COLLISION_APPROVED)}) that sit in one npz, "
                             f"collide there under its key format with a partner that is NOT excluded, and belong to no group ({bad_x})")
    if c.get("pooled_world"):  # 09 items 42-46 (pooled calibration, the LEAD): the record states the pooled build, its combos and labels, and the (A) comparator it applied
        pw_ = c["pooled_world"]
        need_ = ("combos", "scope_families", "y_gap", "saturation_arms", "readable_pools", "y1_def", "dropped_combos")
        if [k_ for k_ in need_ if not pw_.get(k_)] or c.get("estimator_selection") or c.get("estimator_choice", "A") != "A" or c.get("stream_combos") or c.get("stream_pairs"):
            raise SystemExit(f"STOP: a pooled build needs pooled_world {need_}, comparator (A) with no estimator_selection, and no per-origin-stream pairs")
        if sorted(pw_["combos"]) != sorted(c.get("combos") or {}) or sorted(pw_["combos"]) != sorted(c.get("dedup_combos") or []) or c.get("headline_combo") not in pw_["combos"]:
            raise SystemExit("STOP: a pooled build's combos are exactly its pooled (union, main-deduplicated) combos, the headline among them")
        if pw_.get("pilot") is not None:  # 2e 10-16: the end-to-end PILOT (a label subset): PANEL's real per-row gates on its rows, a GATES-ONLY record
            lab_p = [lb for src in c["npz"] for lb in src["labels"]]
            sc_p = {f"{fa} {ro}" for fa in pw_["scope_families"] for ro in ("C", "R")}
            if (pw_["pilot"] is not True or "PILOT" not in str(c.get("pooled_label")) or not lab_p or not set(lab_p) <= sc_p or len(set(lab_p)) != len(lab_p)
                    or any(c.get(k_) for k_ in PILOT_ABSENT) or pw_.get("compare_per_set") or not isinstance(pw_.get("pilot_delta_full_tol_abs"), (int, float)) or isinstance(pw_.get("pilot_delta_full_tol_abs"), bool)
                    or not set(lab_p) <= {lb for rq_ in (c.get("required_rows") or []) if c.get("headline_combo") in (rq_.get("combos") or []) for lb in rq_.get("labels") or []}):
                # (Codex pilot r1 #2: every pilot label stays a REQUIRED row - present and readable - on the headline combo)
                raise SystemExit(f"STOP: a pilot needs pooled_world.pilot true and pilot_delta_full_tol_abs, 'PILOT' in pooled_label, unique labels within the PW scope, "
                                 f"compare_per_set off and none of "
                                 f"{PILOT_ABSENT} (a two-probe pilot cannot test them)")
        for src, b_ in zip(c["npz"], blobs):
            rj_ = json.loads(b_)
            m_ = rj_.get("meta") or {}
            mpw_ = m_.get("pooled_world") or {}
            est_ = dg_(m_, "full_length", "estimator", "used") if isinstance(dg_(m_, "full_length", "estimator"), dict) else dg_(m_, "estimator", "used")  # Stage A @ 8a15fb10f
            # states the comparator used at meta.full_length.estimator.used (stub @ 1c7418b1b); an older record at meta.estimator.used
            if "dropped_combos" in pw_ and mpw_.get("dropped_combos") != pw_["dropped_combos"]:  # 09 option (B) via 2e: the record states the dropped combo, label verbatim
                raise SystemExit(f"STOP: {src['record']['git']} meta.pooled_world.dropped_combos {mpw_.get('dropped_combos')!r} != the configured {pw_['dropped_combos']!r}")
            if set(pw_.get("dropped_combos") or {}) & set(pw_["combos"]):
                raise SystemExit("STOP: a dropped combo is also configured as a pooled combo")
            if not mpw_ or sorted(mpw_.get("combos") or []) != sorted(pw_["combos"]) or sorted(mpw_.get("labels") or []) != sorted(src["labels"]) or est_ != "A":
                raise SystemExit(f"STOP: {src['record']['git']} is not the configured pooled build (meta.pooled_world combos / labels, estimator used == 'A')")
            rv_, mg_ = pw_.get("producer_reviewed_sha"), m_.get("git")  # 2e 10-16: the record's producer revision EQUALS or DESCENDS FROM the reviewed PASS sha
            anc_ = (isinstance(rv_, str) and len(rv_) >= 7 and isinstance(mg_, str) and len(mg_) >= 7
                    and subprocess.run(["git", "merge-base", "--is-ancestor", rv_, mg_], capture_output=True, cwd=HERE, check=False).returncode == 0)
            if pw_.get("stub") is True:  # a STUB dry run (2e 10-16: schema stub, synthetic values, placeholder meta.git): no producer revision to check; gates-only output
                if "STUB" not in str(c.get("pooled_label")) or "STUB_" not in str(src["record"]["git"]) or pw_.get("compare_per_set"):
                    raise SystemExit("STOP: a stub dry run needs 'STUB' in pooled_label, a STUB_ record and compare_per_set off")
                anc_ = True
            if not anc_:
                raise SystemExit(f"STOP: {src['record']['git']} meta.git {mg_!r} is not the producer's reviewed sha {rv_!r} or a descendant of it (or a commit is missing locally)")
            yd_ = {(q_.get("catch") or {}).get("y1_def") for r_ in (rj_.get("pooled") or {}).values() for e_ in r_.values() if isinstance(e_, dict)
                   for md_ in ("joint", "fixed") for q_ in (e_.get(md_) or {}).values() if isinstance(q_, dict)}  # the y1 arm the record states, as a string only
            if yd_ != {pw_["y1_def"]}:
                raise SystemExit(f"STOP: {src['record']['git']} states y1 as {sorted(map(str, yd_))}, not the configured pooled_world.y1_def")
    c.pop("drop_evidence_resolved", None)  # never trust a config-supplied value
    if c.get("drop_evidence_from_record"):  # 507-prep: the record pins a keys-only file {probe: {origin: [cells that may drop]}} at meta.fold_minus1_drops = {commit, path, sha256}
        de_cfg, dres_ = c["drop_evidence_from_record"], None
        if not de_cfg.get("base_origins") or not de_cfg.get("required_families"):  # an empty list would make the per-origin presence check vacuous (fail closed)
            raise SystemExit("STOP: drop_evidence_from_record needs non-empty base_origins and required_families")
        for src, b_ in zip(c["npz"], blobs):
            try:
                ptr_ = jget(json.loads(b_), de_cfg["pointer_field"])
            except (KeyError, TypeError, IndexError):
                raise SystemExit(f"STOP: {src['record']['git']} pins no fold -1 drop evidence at {de_cfg['pointer_field']} (fail closed)") from None
            if not isinstance(ptr_, dict) or not all(ptr_.get(k_) for k_ in ("commit", "path", "sha256")):
                raise SystemExit(f"STOP: {src['record']['git']} fold -1 drop pointer {ptr_!r} is not a path@sha")
            db_ = git("show", f"{ptr_['commit']}:{ptr_['path']}")
            if hashlib.sha256(db_).hexdigest() != ptr_["sha256"]:
                raise SystemExit(f"STOP: fold -1 drop evidence {ptr_['path']} sha256 != the record's {ptr_['sha256']}")
            if ptr_.get("field") != de_cfg.get("pointer_contract_field", "drops[<label>][<origin>]"):  # the declared schema (DROPS @ 93f8b7a86), else STOP (Codex final r24)
                raise SystemExit(f"STOP: fold -1 drop pointer field {ptr_.get('field')!r} != the contract 'drops[<label>][<origin>]'")
            if ptr_.get("window") != c.get("calibration_expected") or sorted(ptr_.get("required_families") or []) != sorted(de_cfg.get("required_families") or []):
                raise SystemExit(f"STOP: fold -1 drop pointer window {ptr_.get('window')!r} / required_families {ptr_.get('required_families')} != the build's "
                                 f"{c.get('calibration_expected')!r} / {de_cfg.get('required_families')}")
            d_ = jget(json.loads(db_), de_cfg.get("drops_field", ["drops"]))
            if not isinstance(d_, dict) or not all(isinstance(v_, dict) and all(isinstance(x_, list) for x_ in v_.values()) for v_ in d_.values()):
                raise SystemExit(f"STOP: fold -1 drop evidence {ptr_['path']} is not {{probe: {{origin: [cells]}}}}")
            for fam_ in de_cfg.get("required_families") or []:  # every required family's C and R rows list every base origin (even when empty)
                for ro_ in "CR":
                    e_ = d_.get(f"{fam_} {ro_}")
                    if not isinstance(e_, dict) or not set(de_cfg.get("base_origins") or []) <= set(e_):
                        raise SystemExit(f"STOP: fold -1 drop evidence lacks {fam_} {ro_} or one of its base origins {de_cfg.get('base_origins')}")
            if dres_ is not None and d_ != dres_:
                raise SystemExit("STOP: the source records pin different fold -1 drop evidence")
            dres_ = d_
        c = c | {"drop_evidence_resolved": dres_, "drop_evidence_meta": ptr_}
    c.pop("calibration_evidence_resolved", None)  # never trust a config-supplied value (Codex final r14)
    if c.get("build") and not c.get("calibration_evidence_field"):
        raise SystemExit("STOP: a final build config needs calibration_evidence_field and calibration_expected (fail closed)")
    if c.get("calibration_evidence_field"):  # the build's calibration as Stage A states it (e.g. meta.window), equal to the configured expectation in every source record, else STOP
        if c.get("calibration_expected") in (None, "", [], {}):
            raise SystemExit("STOP: calibration_expected must be stated (non-null) for a calibration check (Codex final r15)")
        ce_ = []
        for src, b_ in zip(c["npz"], blobs):
            try:
                v_ = jget(json.loads(b_), c["calibration_evidence_field"])
            except (KeyError, TypeError, IndexError):
                raise SystemExit(f"STOP: {src['record']['git']} states no calibration evidence at {c['calibration_evidence_field']}") from None
            if v_ in (None, "", [], {}) or v_ != c.get("calibration_expected"):
                raise SystemExit(f"STOP: {src['record']['git']} calibration evidence {v_!r} != the expected {c.get('calibration_expected')!r}")
            ce_.append(v_)
        c = c | {"calibration_evidence_resolved": ce_[0]}
    if c.get("cohort"):  # 09 checklist item 1: the cohort rule (e.g. the no-filter cohort @ 01012e87e) must be STATED by every source record (each required string present), else STOP
        for src, b_ in zip(c["npz"], blobs):
            miss_ = [t_ for t_ in c["cohort"].get("require_in_record", []) if t_ not in b_.decode()]
            if miss_:
                raise SystemExit(f"STOP: source record {src['record']['git']} does not state the configured cohort ({miss_})")
    if c.get("diff_vs"):  # the earlier panel record this run is diffed against (by key, inside the Modal function)
        db = git("show", f"HEAD:{c['diff_vs']['path']}")
        if hashlib.sha256(db).hexdigest() != c["diff_vs"]["sha256"]:
            raise SystemExit("diff_vs record sha256")
        c = c | {"diff_old": json.loads(db)}
    if c.get("eligibility"):  # 09 @ 24fd07866: the census fork's eligibility file, by path@sha (a stub for tests)
        eb = git("show", c["eligibility"]["git"])
        if hashlib.sha256(eb).hexdigest() != c["eligibility"]["sha256"]:
            raise SystemExit("eligibility sha256")
        ej = json.loads(eb)
        c = c | {"eligibility_json": ej, "eligibility_meta": {"git": c["eligibility"]["git"], "sha256": c["eligibility"]["sha256"], "label": ej.get("label")}}
    if c.get("sim_keys"):  # 09: per-key sim run lists derived from the manifests (the census fork's eligibility file's per_key_provenance), by path@sha
        sb = git("show", c["sim_keys"]["git"])
        if hashlib.sha256(sb).hexdigest() != c["sim_keys"]["sha256"]:
            raise SystemExit("sim_keys source sha256")
        pk_ = json.loads(sb)["per_key_provenance"]
        al_ = lambda o: (c.get("elig_alias") or {}).get(o, o)
        c = c | {"sim_keys_resolved": {o: pk_[f"{al_(o)}|sim"] for o in c.get("origins", ORIGINS) if f"{al_(o)}|sim" in pk_},
                 "sim_keys_meta": {"git": c["sim_keys"]["git"], "sha256": c["sim_keys"]["sha256"],
                                   "label": f"per_key_provenance of {c['sim_keys']['git']} (census fork; derived by key from the manifests)"}}
    if_json = if_pins = None
    if c.get("inputs_final"):  # the frozen input set (read first: it pins membership710 below)
        ib = git("show", c["inputs_final"]["git"])
        if hashlib.sha256(ib).hexdigest() != c["inputs_final"]["sha256"]:
            raise SystemExit("inputs_final sha256")
        if_json = if_pins = json.loads(ib)  # if_pins: the file whose pins the cohort sources' pin_field names (the configured inputs file itself)
        if if_json.get("schema") == "inputs_full/v1":  # 2e (10-16): INPUTS_FULL extends a frozen INPUTS_FINAL v2 via .base = {commit, path, sha256}; sets / extensions /
            bp_ = if_json.get("base")  # label come from that base (pinned by sha here), the full-length exclusions pin from INPUTS_FULL itself (if_pins)
            if not (isinstance(bp_, dict) and all(isinstance(bp_.get(k_), str) and bp_[k_] for k_ in ("commit", "path", "sha256"))):
                raise SystemExit("STOP: INPUTS_FULL (inputs_full/v1) .base is not a {commit, path, sha256} pin")
            bb_ = git("show", f"{bp_['commit']}:{bp_['path']}")
            if hashlib.sha256(bb_).hexdigest() != bp_["sha256"]:
                raise SystemExit(f"STOP: INPUTS_FULL's base {bp_['commit']}:{bp_['path']} sha256 != its pin {bp_['sha256']}")
            if_json = json.loads(bb_)
            if if_json.get("schema") == "inputs_full/v1":
                raise SystemExit("STOP: INPUTS_FULL's base is itself an inputs_full/v1 file (one level of base only)")
            c = c | {"inputs_full_resolved": {"inputs_full": dict(c["inputs_final"]), "base": dict(bp_),
                                              "rule": "sets / extensions / label from the base; the exclusions pin from INPUTS_FULL (inputs_full/v1)"}}
    if c.get("cohort_keys"):  # 09: BY-KEY coverage from the files Stage A's run_peraction311.py derives the cohort from
        kc = c["cohort_keys"]
        files = {}
        for nm, f_ in kc["files"].items():  # each file by path@sha; FIRST, a separate step: its sha256 equals the designated entry in EACH source record's shipped inputs
            fb = git("show", f_["git"])
            if hashlib.sha256(fb).hexdigest() != f_["sha256"]:
                raise SystemExit(f"STOP: cohort file {f_['git']} sha256 != config")
            for src, b_ in zip(c["npz"], blobs):
                si = json.loads(b_)["meta"].get("shipped_inputs")
                sha_, n_ = shipped_entry_sha(si or {}, f_["stageA_name"], None)
                if n_ != 1 or sha_ != f_["sha256"]:
                    raise SystemExit(f"STOP: cohort file {f_['git']} sha256 {f_['sha256']} does not equal Stage A's shipped input '{f_['stageA_name']}' in "
                                     f"{src['record']['git']} ({n_} matching entries; sha {sha_})")
            files[nm] = json.loads(fb)
        # Stage A's derivation (run_peraction311.py row()): cohort = FINAL result.cohort.cohort_cells; S_old = the cells in T196; S_new = the rest
        coh = list(files["final"]["result"]["cohort"]["cohort_cells"])
        t196 = {x["key"] for x in files["t196"]["cells"]}
        keys_ = {"S_old": [x for x in coh if x in t196], "S_new": [x for x in coh if x not in t196]}
        m7 = kc.get("membership710")
        if m7:  # 09 (3): S_newmains BY KEY from membership710.json, pinned by the frozen INPUTS_FINAL (sets.S_newmains.membership710)
            if if_json is None:
                raise SystemExit("STOP: membership710 needs the pinned INPUTS_FINAL (config inputs_final)")
            pin = if_json["sets"]["S_newmains"]["membership710"]
            mb = git("show", m7["git"])
            if hashlib.sha256(mb).hexdigest() != pin["sha256"]:  # its own check: the file's sha256 against the INPUTS_FINAL pin
                raise SystemExit(f"STOP: {m7['git']} sha256 != the INPUTS_FINAL pin {pin['sha256']}")
            if m7["sha256"] != pin["sha256"] or not m7["git"].split(":", 1)[1].endswith(pin["path"]):
                raise SystemExit(f"STOP: config membership710 {m7} does not name the INPUTS_FINAL pin {pin}")
            mj = json.loads(mb)
            w4 = list(mj["cohort"]) + list(mj["cohort_4r_sol_only"])  # with S4: the cohort plus the 4r cells (the s4_rule; every probe here covers the 4r folders)
            wo = list(mj["cohort_without_S4"])
            if len(set(w4)) != len(w4) or not set(wo) <= set(w4) or set(keys_["S_old"] + keys_["S_new"]) & set(w4):
                raise SystemExit("STOP: membership710 keys: duplicates, without-S4 not a subset of with-S4, or overlap with S_old/S_new")
            keys_ |= {"S_newmains": w4, "S_newmains without S4": wo}
        for o_, ex_ in (kc.get("extra_origins") or {}).items():  # Codex checklist r1: other origins' frozen cohorts by path@sha (e.g. S_paper493 from INPUTS_FINAL v2)
            eb_ = git("show", ex_["git"])
            if hashlib.sha256(eb_).hexdigest() != ex_["sha256"]:
                raise SystemExit(f"STOP: cohort source {ex_['git']} sha256 != config")
            if ex_.get("pin_field") and (if_pins is None or jget(if_pins, ex_["pin_field"]).get("sha256") != ex_["sha256"]):  # the configured inputs file's pin (INPUTS_FINAL / FULL)
                raise SystemExit(f"STOP: cohort source {ex_['git']} is not the INPUTS_FINAL pin at {ex_['pin_field']}")
            ej_ = json.loads(eb_)
            keys_[o_] = list(jget(ej_, ex_["field"]))
            if ex_.get("list_sha256_field"):  # 507-prep: the sha256 of the compact sorted JSON list of cells, stated beside it
                ls_ = hashlib.sha256(json.dumps(sorted(keys_[o_]), separators=(",", ":")).encode()).hexdigest()
                if ls_ != jget(ej_, ex_["list_sha256_field"]):
                    raise SystemExit(f"STOP: {o_} cohort list sha256 {ls_} != the file's stated {jget(ej_, ex_['list_sha256_field'])}")
        for o_d, o_b in (c.get("origin_base") or {}).items():  # derived origin rows (e.g. "S_old (all-origins draws)") share their base origin's frozen cohort,
            if o_b in keys_ and o_d not in keys_:  # unless their own cohort is pinned (e.g. the registered-cohort rows via extra_origins)
                keys_[o_d] = list(keys_[o_b])
        keys_ = {o: v for o, v in keys_.items() if o in c.get("origins", ORIGINS)}
        dxi_ = kc.get("dedup_exclusions_inputs_final")
        if dxi_:  # 507-prep: INPUTS_FINAL v2 extensions.S_paper493.P_dup (sorted list); each must be a cell of kind P_dup whose main_origin is another configured origin
            if if_json is None:
                raise SystemExit("STOP: P_dup needs the pinned INPUTS_FINAL")
            pd_, info_ = jget(if_json, dxi_["field"]), jget(if_json, dxi_["cells_field"])
            mo_al = dxi_.get("main_origin_alias") or {}  # INPUTS_FINAL names sets t196 / p311 / new; map them to configured origins
            mo_of = lambda k_: mo_al.get((info_.get(k_) or {}).get("main_origin"), (info_.get(k_) or {}).get("main_origin"))
            if not isinstance(pd_, list) or pd_ != sorted(pd_) or any((info_.get(k_) or {}).get("kind") != "P_dup" or mo_of(k_) not in keys_ or mo_of(k_) == dxi_["origin"] for k_ in pd_):
                raise SystemExit(f"STOP: INPUTS_FINAL {dxi_['field']} is not a sorted list of kind-P_dup cells whose main_origin is another configured origin")
            c = c | {"dedup_excluded_resolved": list(pd_), "dedup_excluded_meta": {"class": "P_dup", "origin": dxi_["origin"], "n": len(pd_), "source": f"INPUTS_FINAL {dxi_['field']}"}}
        dx_ = kc.get("dedup_exclusions")
        if dx_:  # cells excluded from the main-deduplicated combos (e.g. P_dup), by path@sha from the FROZEN exclusions artifact (INPUTS_FINAL's pin; Codex final r7)
            xb_ = git("show", dx_["git"])
            if hashlib.sha256(xb_).hexdigest() != dx_["sha256"]:
                raise SystemExit(f"STOP: dedup exclusions {dx_['git']} sha256 != config")
            if if_pins is None or not dx_.get("pin_field") or (jget(if_pins, dx_["pin_field"]) or {}).get("sha256") != dx_["sha256"]:
                raise SystemExit(f"STOP: dedup exclusions {dx_['git']} are not the frozen INPUTS_FINAL pin at {dx_.get('pin_field')}")
            xo_ = jget(json.loads(xb_), dx_["field"])  # schema: {"class": <allowed class, e.g. P_dup>, "origin": <origin>, "cells": [...]} (Codex final r8)
            if not isinstance(xo_, dict) or xo_.get("class") not in (dx_.get("allowed_classes") or []) or not isinstance(xo_.get("cells"), list) or xo_.get("origin") not in keys_:
                raise SystemExit(f"STOP: dedup exclusions at {dx_['field']} are not a {{class, origin, cells}} object of an allowed class {dx_.get('allowed_classes')}")
            c = c | {"dedup_excluded_resolved": sorted(xo_["cells"]), "dedup_excluded_meta": {"class": xo_["class"], "origin": xo_["origin"], "n": len(xo_["cells"])}}
        dev_ = {}
        for fam_, de_ in (kc.get("drop_evidence") or {}).items():  # pinned fold / exclusion evidence: the cells a probe family may drop, per origin (path@sha)
            db_ = git("show", de_["git"])
            if hashlib.sha256(db_).hexdigest() != de_["sha256"]:
                raise SystemExit(f"STOP: drop evidence {de_['git']} sha256 != config")
            dev_[fam_] = jget(json.loads(db_), de_["field"])
        if dev_ and c.get("drop_evidence_resolved") is not None:
            raise SystemExit("STOP: drop evidence is configured twice (cohort_keys.drop_evidence and drop_evidence_from_record); use one")
        if dev_:
            c = c | {"drop_evidence_resolved": dev_}
        if c.get("combos") and [o for o in c.get("origins", ORIGINS) if o not in keys_]:
            raise SystemExit(f"STOP: no cohort keys for {[o for o in c.get('origins', ORIGINS) if o not in keys_]} (every configured origin needs its frozen cohort by key)")
        dup = {o: len(v) - len(set(v)) for o, v in keys_.items() if len(v) != len(set(v))}
        if dup:
            raise SystemExit(f"STOP: duplicate cohort keys {dup}")
        for src, b_ in zip(c["npz"], blobs):  # cohort counts in EVERY source record's on-policy rows must equal the number of keys
            m_ = ldc_joiner_rows_check(json.loads(b_)["rows"], keys_)
            if m_:
                raise SystemExit(f"STOP: {src['record']['git']}: {m_}")
        if c.get("combos"):  # Codex N-origin r1: additive pooling needs MAIN-disjoint members; by key per combo pair (overlap -> STOP)
            mmap = {"S_old": ({x["key"]: x["main"] for x in files["t196"]["cells"]}, "T196 cells' main"),
                    **({o_: (mj["main_of"], "membership710 main_of") for o_ in keys_ if o_.startswith("S_newmains")} if m7 else {})}
            for o_, mm_ in (kc.get("main_maps") or {}).items():  # pinned per-key main maps for the other origins (e.g. S_new from the census fork's per_key_provenance)
                mb_ = git("show", mm_["git"])
                if hashlib.sha256(mb_).hexdigest() != mm_["sha256"]:
                    raise SystemExit(f"STOP: main map {mm_['git']} sha256 != config")
                if mm_.get("cell_main_field"):  # a pinned {cell: main} mapping (e.g. from the exclusions artifact)
                    cm_ = dict(jget(json.loads(mb_), mm_["cell_main_field"]))
                    mmap[o_] = (cm_, f"{mm_['cell_main_field']} of {mm_['git']}")
                    continue
                pk_ = json.loads(mb_)["per_key_provenance"]
                cm_ = {}
                for lst in mm_["lists"]:
                    for e_ in pk_[lst]:
                        if cm_.setdefault(e_["cell"], e_["main"]) != e_["main"]:
                            raise SystemExit(f"STOP: main map {mm_['git']}: cell {e_['cell']} has two mains")
                mmap[o_] = (cm_, f"per_key_provenance {mm_['lists']} of {mm_['git']}")
            for o_, mm_ in (kc.get("main_maps_inputs_final") or {}).items():  # e.g. S_paper493: INPUTS_FINAL v2 extensions.S_paper493.cells[<cell>].main (pinned by INPUTS_FINAL)
                if if_json is None:
                    raise SystemExit(f"STOP: {o_}'s main map needs the pinned INPUTS_FINAL")
                mmap[o_] = ({k: v["main"] for k, v in jget(if_json, mm_["cells_field"]).items()}, f"INPUTS_FINAL {mm_['cells_field']}")
            for o_d, o_b in (c.get("origin_base") or {}).items():
                if o_b in mmap:
                    mmap[o_d] = mmap[o_b]
            mains_, basis_ = {}, {}
            for o, ks in keys_.items():
                if o not in mmap or any(k not in mmap[o][0] for k in ks):  # every cohort cell needs a pinned main (Codex part-1 r1): else STOP
                    raise SystemExit(f"STOP: no pinned main for every cohort cell of {o} ({sum(k not in mmap.get(o, ({}, ''))[0] for k in ks)} missing): main-disjointness cannot be checked")
                mains_[o], basis_[o] = {mmap[o][0][k] for k in ks}, mmap[o][1]
            pairs_ = sorted({tuple(sorted((a_, b_))) for cb_, M in c["combos"].items() if cb_ not in read_combos(c) for a_ in M for b_ in M if a_ != b_})
            shared_ = {}
            for a_, b_ in pairs_:
                if a_ not in keys_ or b_ not in keys_:
                    raise SystemExit(f"STOP: no cohort keys for {a_} or {b_}: the combos' main-disjointness cannot be checked")
                if (mains_[a_] & mains_[b_]) or (set(keys_[a_]) & set(keys_[b_])):
                    shared_[f"{a_} & {b_}"] = sorted(mains_[a_] & mains_[b_])[:20]
            if shared_:
                raise SystemExit(f"STOP: combo members share mains or cells (additive pooling is not valid; 09: a main counts once): {shared_}")
            mdis = {"pairs_checked": [list(x) for x in pairs_], "basis": basis_, "rule": "no shared main (by pinned main name) or cell key between any two members of an additive combo"}
            if c.get("dedup_excluded_resolved"):  # meaning of a P_dup exclusion: an EXACT duplicate (main, side task) of a cell in another BASE origin's pinned cohort
                xm_ = c["dedup_excluded_meta"]
                derived_ = set(c.get("origin_base") or {})
                info_x = (jget(if_json, kc["dedup_exclusions_inputs_final"]["cells_field"]) if kc.get("dedup_exclusions_inputs_final") and if_json else {}) or {}
                t196s_ = {x["key"]: (x["main"], x.get("side")) for x in files["t196"]["cells"]}
                sides_ = {"S_old": {k: v[1] for k, v in t196s_.items()}}
                for o_s, sm_ in (kc.get("side_maps") or {}).items():  # pinned {cell: side task} per other base origin (path@sha); without one, its duplicates cannot be verified
                    sb_ = git("show", sm_["git"])
                    if hashlib.sha256(sb_).hexdigest() != sm_["sha256"]:
                        raise SystemExit(f"STOP: side map {sm_['git']} sha256 != config")
                    sides_[o_s] = dict(jget(json.loads(sb_), sm_["field"]))
                badx_ = []
                for k in c["dedup_excluded_resolved"]:
                    side_ = (info_x.get(k) or {}).get("side")
                    base_o = [o for o in keys_ if o not in derived_ and o != xm_["origin"]]
                    ident_ = (mmap[xm_["origin"]][0].get(k), side_)
                    found_ = any((mmap[o][0].get(k2), (sides_.get(o) or {}).get(k2)) == ident_ for o in base_o if o in sides_ for k2 in keys_[o])
                    if k not in keys_[xm_["origin"]] or side_ is None or not found_:  # no stated side task, or no exact duplicate in a base origin: STOP (fail closed)
                        badx_.append(k)
                if badx_:
                    raise SystemExit(f"STOP: dedup exclusions {badx_[:5]} are not exact (main, side task) duplicates of a cell in another base origin's pinned cohort")
            if c.get("dedup_combos"):  # cell -> main for the independent draw-0 dedup recompute whenever hits are available (popped before the record is written)
                c = c | {"cell_main_resolved": {k: mmap[o][0][k] for o, ks in keys_.items() for k in ks}}
        c.pop("item36_attempt_filter_resolved", None)  # never trust a config-supplied value
        af_ = ((c.get("coheadline") or {}).get("item36") or {}).get("attempt_filter")
        if af_:  # 09 RULE item 36 (2), by pinned key (keys only): N = lead-cohort cells of each labelled set, K = those the registered filter drops, unlabelled sets counted apart
            per_ = {}
            for o_, r_ in af_["registered_of"].items():
                if o_ not in keys_ or r_ not in keys_ or not set(keys_[r_]) <= set(keys_[o_]):
                    raise SystemExit(f"STOP: item 36: {r_} must be a pinned subset of the pinned {o_} cohort")
                per_[o_] = {"N": len(set(keys_[o_])), "K": len(set(keys_[o_]) - set(keys_[r_])), "registered": len(set(keys_[r_]))}
                for src, b_ in zip(c["npz"], blobs):  # 2e: N (no-filter, by pinned key) must equal each source record's stated headline cohort count (meta.cohort_list)
                    st_ = (((json.loads(b_).get("meta") or {}).get("cohort_list") or {}).get("headline_cells") or {}).get(o_)
                    if st_ != per_[o_]["N"]:
                        raise SystemExit(f"STOP: item 36: {o_} N = {per_[o_]['N']} by pinned key != {src['record']['git']} meta.cohort_list.headline_cells {st_!r}")
            unl_ = {o_: {"cells": len(set(keys_[o_])), "status": "WITHHELD from the filtered version: no attempt labels"} for o_ in af_["unlabelled"] if o_ in keys_}
            if set(unl_) != set(af_["unlabelled"]):
                raise SystemExit(f"STOP: item 36: unlabelled sets {af_['unlabelled']} need pinned cohorts")
            c = c | {"item36_attempt_filter_resolved": {"rule": "09 RULE item 36 (2) @ 42609cbf1", "basis": "cohort cells by pinned key, before any probe's fold -1 drops",
                                                        "labelled_sets": per_, "unlabelled_sets": unl_}}
        c = c | {"cohort_keys_resolved": keys_,
                 "cohort_keys_meta": {"files": {nm: {"git": f_["git"], "sha256": f_["sha256"], "stageA_name": f_["stageA_name"]} for nm, f_ in kc["files"].items()},
                                      "derivation": "Stage A run_peraction311.py row(): cohort = FINAL result.cohort.cohort_cells; S_old = cells in T196; S_new = the rest",
                                      "counts": {o: len(v) for o, v in keys_.items()}, "shipped_check": "each file's sha256 equals the designated entry in every source record"}
                 | ({"key_sha256": {o: hashlib.sha256(json.dumps(sorted(v), separators=(",", ":")).encode()).hexdigest() for o, v in keys_.items()}} if c.get("combos") else {})
                 | ({"membership710": {"git": m7["git"], "sha256": m7["sha256"], "pin": "INPUTS_FINAL sets.S_newmains.membership710 (checked first, on its own)",
                                       "derivation": "S_newmains = cohort + cohort_4r_sol_only; S_newmains without S4 = cohort_without_S4 (by key)"}} if m7 else {})
                 | ({"mains_disjoint": mdis} if c.get("combos") else {})}
    if (c.get("coheadline") or {}).get("auroc"):  # 09 sec. 3 / sec. 6: the AUROC subset is fixed from the pinned table HERE, before any PANEL value exists
        A_ = c["coheadline"]["auroc"]
        rb_, tb_ = git("show", A_["record"]["git"]), git("show", A_["table"]["git"])
        if hashlib.sha256(rb_).hexdigest() != A_["record"]["sha256"] or hashlib.sha256(tb_).hexdigest() != A_["table"]["sha256"]:
            raise SystemExit("STOP: AUROC record or table sha256 != config")
        tj = json.loads(tb_)
        if A_["record"]["sha256"] not in json.dumps(tj.get("records", {}).get("new")) or not tj.get("all_pass"):
            raise SystemExit("STOP: the AUROC table does not name the pinned record (records.new) or its gates did not all pass")
        hm_ = sorted((c.get("combos") or {}).get(c.get("headline_combo")) or [])
        vals, der_ = {}, {}
        for p_ in c["coheadline"]["probes"]:
            sp_ = (A_.get("derived") or {}).get(p_, p_)  # 09 co-headline sec. 7: own_all_rs is in iff own_all_L48's C entry in the same record is >= the threshold
            if sp_ != p_:
                der_[p_] = sp_
            e_ = (tj["table"].get(f"{sp_} {A_['readout']}") or {}).get(A_["combo"]) or {}
            if not isinstance(e_.get(A_["field"]), dict) or e_[A_["field"]].get("point") is None:
                raise SystemExit(f"STOP: no {A_['field']} for {sp_} {A_['readout']} / {A_['combo']} in the pinned AUROC table (no fallback)")
            exp_o = sorted(A_.get("expected_origins") or hm_)  # the pinned table's OWN cohort (09 sec. 7 fixes membership from the 767-task table), not FINAL's origins (Codex checklist r1)
            if exp_o and sorted(e_.get("origins") or []) != exp_o:
                raise SystemExit(f"STOP: the AUROC table's {A_['combo']} origins {e_.get('origins')} != the expected {exp_o}")
            vals[p_] = float(e_[A_["field"]]["point"])
        thr = float(A_["threshold"])
        sub = [p_ for p_ in c["coheadline"]["probes"] if vals[p_] >= thr]
        if c.get("labels_pinned") and sorted(sub) != sorted(c["labels_pinned"]["AUROC06_FULL"].values()):  # 507-prep's pinned full-length membership (09 @ 002c1824b (d))
            raise SystemExit(f"STOP: the derived AUROC subset {sub} != the pinned AUROC06_FULL {sorted(c['labels_pinned']['AUROC06_FULL'].values())}")
        if A_.get("stageA_members_field"):  # 507-prep's meta.auroc06_subset (membership pinned @ 19ba8f563) must state the same members, else STOP
            for src, b_ in zip(c["npz"], blobs):
                sm_ = jget(json.loads(b_), A_["stageA_members_field"])
                if A_.get("stageA_pin_field") and jget(json.loads(b_), A_["stageA_pin_field"]) != A_["table"]["git"].split(":")[0]:
                    raise SystemExit(f"STOP: {src['record']['git']} AUROC-subset membership pin != {A_['table']['git'].split(':')[0]}")
                if sorted(sm_ or []) != sorted(sub):
                    raise SystemExit(f"STOP: {src['record']['git']} states AUROC-subset members {sm_} != {sub}")
        c = c | {"coheadline_auroc_resolved": {"values": vals, "derived_from": der_, "threshold": thr, "readout": A_["readout"], "subset": sub, "excluded": [p_ for p_ in vals if p_ not in sub],
                                               "basis": f"{A_['readout']} run AUROC table[label][{A_['combo']}].{A_['field']} >= {thr:g}, from {A_['table']['git']} (sha256 {A_['table']['sha256']}; "
                                                        f"record {A_['record']['git']}, sha256 {A_['record']['sha256']}); resolved before any PANEL value" + (f"; derived memberships (09 sec. 7): {der_}" if der_ else ""),
                                               "label": A_.get("post_hoc_label", "POST HOC: threshold chosen after the AUROCs were seen")}}
    if c.get("combos") and not c.get("cohort_keys_resolved"):  # no evidence of main-disjointness -> STOP (Codex N-origin r1)
        raise SystemExit("STOP: combos need cohort_keys (by key) to check that each combo's members are main-disjoint")
    if (c.get("plant") or {}).get("records"):  # 507-prep's planted records, by path@sha: each arm's status and ACHIEVED shift, by key (2e: achieved, not nominal)
        arms_ = {}
        npz_shas = {e_["sha256"] for e_ in (c["plant"]["npz"] if isinstance(c["plant"]["npz"], list) else [c["plant"]["npz"]])}
        for r_ in c["plant"]["records"]:
            pb_ = git("show", r_["git"])
            if hashlib.sha256(pb_).hexdigest() != r_["sha256"]:
                raise SystemExit(f"STOP: planted record {r_['git']} sha256 != config")
            pj_ = json.loads(pb_)
            if pj_["npz"]["sha256"] not in npz_shas:
                raise SystemExit(f"STOP: planted record {r_['git']} names npz {pj_['npz']['sha256']}, not a configured planted npz")
            for lab_, row_ in pj_["rows"].items():
                for o_, v_ in row_.items():
                    for f_, x_ in ((v_.get("plant") or {}).get("solve_draw0") or {}).items():
                        for tag_, a_ in x_["arms"].items():
                            if lab_ in arms_ and o_ in arms_[lab_] and f_ in arms_[lab_][o_] and tag_ in arms_[lab_][o_][f_]:
                                raise SystemExit(f"STOP: planted arm {lab_} {o_} {f_} {tag_} stated twice")
                            arms_.setdefault(lab_, {}).setdefault(o_, {}).setdefault(f_, {})[tag_] = {"status": a_["status"], "achieved_points": a_.get("achieved_points")}
        c = c | {"plant_arms_resolved": arms_}
    if c.get("ld_restricted"):  # the restricted inputs' record, by path@sha
        rb_ = git("show", c["ld_restricted"]["record"]["git"])
        if hashlib.sha256(rb_).hexdigest() != c["ld_restricted"]["record"]["sha256"]:
            raise SystemExit("restricted record sha256")
        c = c | {"ld_restricted_rec": json.loads(rb_)}
    if if_json is not None:  # the frozen input set; its filled label is carried verbatim
        c = c | {"inputs_final_label": if_json["label"]["filled"]}
    if c.get("check_vs2"):  # a third earlier record (e.g. v9's bias section)
        cb2 = git("show", f"HEAD:{c['check_vs2']['path']}")
        if hashlib.sha256(cb2).hexdigest() != c["check_vs2"]["sha256"]:
            raise SystemExit("check_vs2 record sha256")
        c = c | {"check_old2": json.loads(cb2)}
    if c.get("check_vs"):  # a second earlier record some gates compare against (e.g. v8's separate on-policy rows)
        cb = git("show", f"HEAD:{c['check_vs']['path']}")
        if hashlib.sha256(cb).hexdigest() != c["check_vs"]["sha256"]:
            raise SystemExit("check_vs record sha256")
        c = c | {"check_old": json.loads(cb)}
    if c.get("prev_record"):
        pb = git("show", f"HEAD:{c['prev_record']['path']}")
        if hashlib.sha256(pb).hexdigest() != c["prev_record"]["sha256"]:
            raise SystemExit("previous record sha256")
        c = c | {"prev": json.loads(pb)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    ctx = launch_context("pooled", cfg, stamp, git)
    print(json.dumps({"app_id": ctx["app_id"], "volume_copy_name": ctx["out_name"]}))
    out = panel.remote(c, blobs, ctx)
    print(json.dumps({"app_id": ctx["app_id"], "volume_copy": out.get("volume_copy")}))
    write_outputs(cfg, c, out, ctx)


def write_outputs(cfg, c, out, ctx):
    """results/ from a panel() return value (directly after the run, or later from its Volume copy via fetch_panel_output.py); stamp and code from ctx."""
    from probe_shift.panel_out import write_once

    stamp = ctx["stamp"]
    out = {k: v for k, v in out.items() if k != "volume_copy"}
    if "gates_only" in out:  # a dry-run record (no PANEL statistic), exclusive create
        go = out["gates_only"]
        stamp_ = stamp  # the launch stamp, so a re-materialisation names the same file (Codex detach r1)
        write_once(HERE / "results" / f"gatecheck_{cfg}_{stamp_}.json", json.dumps({"label": "DRY RUN: loader, input gates and the provenance-only rule; no PANEL statistic",
                                                                                         "config": c | {k: None for k in ("eligibility_json", "cohort_keys_resolved")}, **go}, default=str, indent=1).encode())
        ldc_ = go.get("ld_choice") or {}
        print(json.dumps({"file": f"gatecheck_{cfg}_{stamp_}.json", "n_gates": go["n_gates"], "joiner": ldc_.get("joiner"), "allowed": ldc_.get("allowed_origins"),
                          "excluded": ldc_.get("excluded"), "restricted": ldc_.get("restricted"), "coverage": ldc_.get("coverage_by_set"), "mismatches": ldc_.get("mismatches"),
                          "stop": ldc_.get("stop"), "ld_rows": {k: {kk: v.get(kk) for kk in ("valid", "reason")} for k, v in go["ld_rows"].items()}}, default=str))
        return
    if out.get("gate_reproduces_prev") is not None and not out["gate_reproduces_prev"]["pass"]:
        print(json.dumps(out["gate_reproduces_prev"])[:3000])
        raise SystemExit("gate: the earlier record is not reproduced by key")
    c.pop("prev", None)
    c.pop("diff_old", None)
    c.pop("check_old", None)
    c.pop("check_old2", None)
    c.pop("eligibility_json", None)
    c.pop("cohort_keys_resolved", None)
    c.pop("cell_main_resolved", None)
    c.pop("drop_evidence_resolved", None)
    c.pop("dedup_excluded_resolved", None)
    c.pop("sim_keys_resolved", None)
    c.pop("ld_restricted_rec", None)
    if "GATE_FAIL" in out:  # the failure detail can quote values (e.g. the co-headline vs Stage A check): to a local file, never the client (2e 10-16 blinding)
        gf_ = HERE / "results" / f"gatefail_{cfg}_{stamp}.json"
        write_once(gf_, json.dumps(out, default=str, indent=1).encode())
        print(json.dumps({"GATE_FAIL": True, "detail_file": gf_.name, "detail_sha256": hashlib.sha256(gf_.read_bytes()).hexdigest(), "volume_copy": out.get("volume_copy"),
                          "top_level_keys": sorted(out)}))
        raise SystemExit("gate failed (detail in the file above; read its check names with jq, keys only)")
    if gates_only_mode(c):  # 2e 10-16: the PILOT (and a STUB dry run) writes a GATES-ONLY record (no results, figure, captions or other values)
        kind_ = "stub" if (c.get("pooled_world") or {}).get("stub") is True else "pilot"
        pr_ = {"label": f"{kind_.upper()} (gates only, no values): PANEL's pooled per-row gates on the {kind_} record", "pooled_label": c["pooled_label"],
               "meta": {"git": ctx["git"], "stamp_pt": stamp, "script_sha256": ctx["script_sha256"], "app_id": ctx.get("app_id"), "config": c},
               "pilot_scope": {"labels": [lb for src in c["npz"] for lb in src["labels"]], "combos": sorted(c.get("combos") or {})},
               "gates": {k_: out["gates"].get(k_) for k_ in ("n", "n_fail", "all_pass", "checked")}} | (
            {"npz_collision_excluded": out["npz_collision_excluded"]} if out.get("npz_collision_excluded") else {})
        pb_ = json.dumps(pr_, default=str, indent=1).encode()
        write_once(HERE / "results" / f"panel_{kind_}_{cfg}_{stamp}.json", pb_)
        print(json.dumps({"record": f"panel_{kind_}_{cfg}_{stamp}.json", "sha256": hashlib.sha256(pb_).hexdigest(), "gates": pr_["gates"] | {"checked": None}}))
        return
    rec = {"label": LABEL, "pooled_label": c["pooled_label"],
           "meta": {"git": ctx["git"], "stamp_pt": stamp, "script_sha256": ctx["script_sha256"], "config": c,
                    **({"lead_readout": c["lead_readout"], "beside_readout": [x for x in ("C", "R") if x != c["lead_readout"]]} if c.get("lead_readout") else {}),
                    **({"cohort": c["cohort"]} if c.get("cohort") else {}),
                    "rule": "experiments/2026-10-15_panel_noise_matched/RULE.md (Amendments 1 and 2)"},
           "definitions": {"G": ("0.5|sim@S|primary - run1@B| + 0.5|sim@S|primary - run2@B| (points); on-policy rows (own_frac*, own_all*): |sim@S|primary - run2@B| (run 1 is a training arm)"
                                 if c.get("g_onpolicy_run2_only", False) else "0.5|sim@S|primary - run1@B| + 0.5|sim@S|primary - run2@B| (points)"),
                           "headline": (f"headline group = {out.get('headline_group_resolved')} (config: {c.get('headline_group', HEADLINE_PENDING)}); off-policy 5 = {HEADLINE_NAME}; "
                                        f"other groups with on-policy rows are sensitivities ({SENS_LABEL}); on-policy rows separately: {ONPOL_CAPTION}"
                                        if c.get("headline_excludes_onpolicy", True) else None), "Y": "|run2@A - run1@B2 (S_old) or run1@B (S_new)| (points)",
                           "Y_tauB": ("per row: run2@tau_B - run1@tau_B from the real arms (= sim_minus_run1 - sim_minus_run2 wherever the sim arm is defined), same draws, joint 95% interval when the mode is joint; points; "
                                      "the per-probe term of the co-headline Y (whose mean over the probes it is); NOT the registered yardstick (|run2@A - run1@B2 or B|)"),
                           "pooled": "each catch X = (X_old D_old + X_new D_new)/(D_old + D_new) per draw (the 455 fork's pool_reduce)",
                           "group": ("per-draw mean over the member rows; a group with ANY declared member missing or unreadable is WITHHELD, never averaged over a subset "
                                     "(complete membership, 2e / 0d 10-16); NaN if any member is NaN in a draw; summ: point = draw 0, 95% percentile interval over draws 1..D"),
                           "saturated": f"any arm >= {c['saturation']} at draw 0", "ordering": "pooled first is a POST HOC presentation choice; per-set rows have the same columns",
                           "pooled_abs_per_origin": "SENSITIVITY (2e GO 10-15 14:3x): G and Y per origin first, then (G_old D_old + G_new D_new)/(D_old + D_new) per draw; removes cross-origin cancellation; a different estimand (the typical per-origin gap, not the gap on the 455-task mix)"},
           **({"panel_groups_gate": out["panel_groups_gate"]} if out.get("panel_groups_gate") else {}),
           **({"coheadline": out["coheadline"]} if out.get("coheadline") else {}),
           **({"stream_comparison": out["stream_comparison"]} if out.get("stream_comparison") else {}),
           **({"per_set_comparison": out["per_set_comparison"]} if out.get("per_set_comparison") else {}),
           **({"item36_attempt_filter": c["item36_attempt_filter_resolved"]} if c.get("item36_attempt_filter_resolved") else {}),
           "gates": out["gates"], "D_keys": out["D_keys"], "draw0_outside_own_ci95": out["draw0_outside_own_ci95"], "origin_signs": out["origin_signs"], "title": out["title"], "results": out["res"], "gate_reproduces_prev": out.get("gate_reproduces_prev"), "plant_check": out.get("plant_check")}
    if out.get("npz_collision_excluded"):  # labels with no npz-paired gate or PANEL row (their record JSON is unaffected), with the verbatim reason
        rec["npz_collision_excluded"] = out["npz_collision_excluded"]
    rec |= {"headline_group": out.get("headline_group_resolved", c.get("headline_group", HEADLINE_PENDING)), "headline_group_config": c.get("headline_group", HEADLINE_PENDING),
            "headline_rule": out.get("headline_rule"), "inputs_final_label": c.get("inputs_final_label"), "inputs_full": c.get("inputs_full_resolved"), "cohort_keys": c.get("cohort_keys_meta"), "headline_set": "pooled", "sensitivity_set": {"pooled_abs_per_origin": "sensitivity (per set)"},
            "pooling_caveat": POOLING_CAVEAT}
    rec["label"] = f"{LABEL}; {POOLING_CAVEAT}"
    if c.get("lead_statistic"):  # 09 @ 2d61af835
        rec["lead_statistic"] = {"primary": "the six-probe signed S at COMMANDS, per-action 1e-2, joint 95%, in the selected estimator ((A) or (B)); Y beside for scale; no S - Y",
                                 "secondary": "the noise-matched excess, always on (A): 'secondary check'", "rule": "09 @ 2d61af835", "value": c["lead_statistic"]}
    if c.get("onpolicy_caveats") and rec["definitions"]["headline"]:  # 09's per-probe wording REPLACES the generic on-policy texts (Codex interim r1, part-1 r1)
        rec["definitions"]["headline"] = (f"headline group = {out.get('headline_group_resolved')} (config: {c.get('headline_group', HEADLINE_PENDING)}); off-policy 5 = {HEADLINE_NAME}; "
                                          "other groups with on-policy rows are sensitivities; on-policy rows separately, each with its own launch status: "
                                          + "; ".join(f"{k}: {v}" for k, v in c["onpolicy_caveats"].items()))
    if c.get("combos"):  # the generic definitions (Codex interim r1): the pooled set is the headline combo over its member origins
        hm_ = c["combos"][c["headline_combo"]]
        rec["definitions"] |= {"pooled": f"the headline combo {c['headline_combo']} ({' + '.join(hm_)}): each catch X = sum_o X_o D_o / sum_o D_o over the label's members, per draw (Stage A's pooling; gated)",
                               "pooled_abs_per_origin": f"SENSITIVITY: G and Y per origin first, then sum_o G_o D_o / sum_o D_o over the label's members of {c['headline_combo']}; removes cross-origin "
                                                        "cancellation; a different estimand (the typical per-origin gap)",
                               "Y": "|run2@A - run1@B2 (where Stage A has B2, e.g. S_old) or run1@B| (points)"}
    if c.get("pooled_world"):  # 09 RULE 45.1 / 45.2 / 44 (pooled calibration, the lead): the yardstick, saturation and readability as the pooled build defines them
        pw_ = c["pooled_world"]
        rec["definitions"] |= {"Y": f"pooled: yardstick at the pooled tau_B (no separate B2 pool): |{pw_['y_gap']}| = |run2@B - run1@B| (points; RULE 45.1)",
                               "Y_tauB": ("per row: run2@tau_B - run1@tau_B at the pooled tau_B from the real arms (= sim_minus_run1 - sim_minus_run2 wherever the sim arm is "
                                          "defined), same draws; points; under pooled "
                                          "calibration this IS the signed yardstick (RULE 45.1), and the per-probe term of the co-headline Y"),
                               "saturated": f"any of {pw_['saturation_arms']} >= {c['saturation']} at draw 0 (pooled: the catches G and Y use)",
                               "readable": f"RULE 37 under pooled calibration: the pools the product thresholds on ({pw_['readable_pools']}) pass the registered shading test",
                               "dedup_weights": "W0 from scratch (pinned cell -> canonical main, one main once at draw 0), asserted equal to the record's weights (09 #7)"}
    if c.get("combos"):  # N origins and several combos (2e; 09's answers @ 235725ab0): what each set name means
        rec["combos"] = {"origins": c["origins"], "combos": c["combos"], "headline_combo": c["headline_combo"], "mode": c.get("mode", "joint"), "labels": c.get("combo_labels") or {},
                         "set_names": {"pooled / pooled_abs_per_origin": f"the headline combo {c['headline_combo']} (pooled; per-set sensitivity)",
                                       "<combo> / <combo>|per_set": "every other combo (pooled; per-set sensitivity)", "<origin>": "each origin alone"},
                         "members": "per label from its source record (members + withheld members must equal the configured combo; gated)",
                         "pooled": "additive combos: each catch X = sum_o X_o D_o / sum_o D_o over the label's members, per draw; gated against Stage A's saved pooled gaps "
                                   f"(tolerance {c.get('pooled_recompute_tol_abs', 0.0)} in rate units, stated in the config before the run)"
                                   + (f"; main-deduplicated combos {c['dedup_combos']}: {c.get('dedup_label', 'pooled row read from Stage A (main-deduplicated); not independently recomputed')}"
                                      "; their per-set sensitivity weights a main that appears in several origins once per origin (it changes weighting as well as cancellation)"
                                      if c.get("dedup_combos") else "")
                                   + (f"; per-origin-stream combos {c['stream_combos']}: {STREAM_LABEL}" if c.get("stream_combos") else "")}
    if out.get("diff"):
        rec["diff_vs_earlier_record"] = {"record": c["diff_vs"]["path"], "sha256": c["diff_vs"]["sha256"], **out["diff"]}
    rec["captions"] = captions(rec)
    d = HERE / "results"
    blob = json.dumps(rec, default=float, indent=1).encode()
    write_once(d / f"panel_{cfg}_{stamp}.json", blob)
    write_once(d / f"panel_{cfg}_{stamp}.png", out["png"])
    write_once(d / f"panel_{cfg}_{stamp}_captions.json", json.dumps({"record": f"panel_{cfg}_{stamp}.json", "record_sha256": hashlib.sha256(blob).hexdigest(),
                                                                            "title": out["title"], "captions": rec["captions"]}, indent=1).encode())
    print(json.dumps({"record": f"panel_{cfg}_{stamp}.json", "sha256": hashlib.sha256(blob).hexdigest(), "gates": {k_: out["gates"].get(k_) for k_ in ("n", "n_fail", "all_pass")},
                      "title": "withheld from the client (it carries values; 2e 10-16 blinding): in the record and its captions file"}))
