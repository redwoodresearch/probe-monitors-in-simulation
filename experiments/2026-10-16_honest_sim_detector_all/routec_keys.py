"""Route C rerun (2026-10-18): the S_old sim-honest key check. Route C's INPUTS_FULL (1e186858..., 233212) pins S_old to inventory_b_fix_v2
(= inventory_b_fix_v1 + the 20 pilot sim-honest rows, its own label) and the pilot's counted keys (s_old_simh_extra_roots). This module checks,
by key only, that the S_old sim class is exactly the registered 171 runs plus those 20: nothing else may change. Pure functions, no I/O;
honest_inputs.build_all calls them in the container. Keys, tasks and shas only (the counted-keys file and both inventories are keys/names/shas
records by their writers: simh_counted_keys_pilot_v3 (q3_q35 simhonest purchase), inventory_b_fix_v1/v2 (cut_sensitivity, "keys/shas only")).
"""

import hashlib

EXTRA_ROOT = "simh_q4f_v1_pilot__sim_honest"  # route C INPUTS_FULL s_old_simh_extra_roots key
EXTRA_N, EXTRA_WORLD = 20, "simh_pilot"  # full_override.S_OLD_EXTRA_N / S_OLD_EXTRA_WORLD (fbb59a213 L129-137)
# NOTE_routec_census L310/L560 XKSHA: sha256 of the 20 sorted <world>|<task> joined by "\n", the census writer's own form and hash:
# run_peraction311.py @ fbb59a213 L6326 (keys_sha256 = C.sha_b("\n".join(xks_).encode()), xks_ = full_override.s_old_extra_keys L129-137).
# Named PILOT_LIST_DIGEST: the hook's generic-api-key rule fired on the earlier name; it is a digest of public run names, not a credential.
PILOT_LIST_DIGEST = "e7f62cca627f1dfea225ee921fc532c7ed5234a140a7a8bd07e112229eabd25a"
SOLD_SIM_REGISTERED = 171  # RULE.md "Pools": A25 b1 + b2 (149) + fill-in (8) + sp493 g4 on T196 cells (14)


def pilot_keys(manifest: dict) -> tuple:
    """The pinned counted-keys file -> (sorted <world>|<task> keys, sorted (task, eval_sha256) pairs); full_override.s_old_extra_keys' rule."""
    ks = manifest.get("keys") or []
    keys = [f"{k.get('world')}|{k.get('task')}" for k in ks]
    if (manifest.get("n") != EXTRA_N or len(keys) != EXTRA_N or len(set(keys)) != EXTRA_N
            or any(k.get("world") != EXTRA_WORLD or k.get("stratum") != "S_old" or not k.get("task") or not k.get("eval_sha256") for k in ks)):
        raise ValueError(f"pilot counted keys: not {EXTRA_N} distinct {EXTRA_WORLD}|<task> keys of stratum S_old with an eval sha256")
    keys = sorted(keys)
    if hashlib.sha256("\n".join(keys).encode()).hexdigest() != PILOT_LIST_DIGEST:
        raise ValueError("pilot counted keys: sha256 of the sorted keys differs from the route C census pin")
    pairs = sorted((k["task"], k["eval_sha256"]) for k in ks)
    if len(set(pairs)) != EXTRA_N:
        raise ValueError("pilot counted keys: a (task, eval sha256) pair twice")
    return keys, pairs


def pilot_pin(full: dict, spec: dict) -> None:
    """INPUTS_FULL's s_old_simh_extra_roots must be exactly {EXTRA_ROOT: the shipped counted-keys file} (commit by prefix, path, sha256)."""
    x = full.get("s_old_simh_extra_roots")
    if not isinstance(x, dict) or set(x) != {EXTRA_ROOT}:
        raise ValueError(f"INPUTS_FULL s_old_simh_extra_roots is not exactly {{{EXTRA_ROOT}: ...}}")
    p = x[EXTRA_ROOT]
    if not spec["commit"].startswith(p["commit"]) or spec["path"] != p["path"] or spec["sha256"] != p["sha256"]:
        raise ValueError("the shipped pilot counted-keys file is not INPUTS_FULL's s_old_simh_extra_roots pin")


def inventory_delta(v1: dict, v2: dict) -> list:
    """inventory_b_fix_v2 against fix_v1 by key: every v1 row present and byte-equal as JSON objects, s4_orphans_out equal, status equal;
    returns v2's new rows (keys only are compared; values are never printed)."""
    r1, r2 = {r["key"]: r for r in v1["runs"]}, {r["key"]: r for r in v2["runs"]}
    if len(r1) != len(v1["runs"]) or len(r2) != len(v2["runs"]):
        raise ValueError("an inventory key twice")
    if set(r1) - set(r2):
        raise ValueError(f"{len(set(r1) - set(r2))} fix_v1 rows missing from fix_v2")
    changed = sum(r1[k] != r2[k] for k in r1)
    if changed:
        raise ValueError(f"{changed} fix_v1 rows differ in fix_v2")
    if v1.get("s4_orphans_out") != v2.get("s4_orphans_out") or v1.get("status") != v2.get("status"):
        raise ValueError("fix_v2's s4_orphans_out or status differs from fix_v1's")
    return [r2[k] for k in sorted(set(r2) - set(r1))]


def new_rows_are_pilot(new: list, pairs: list) -> set:
    """fix_v2's new rows must be exactly the pilot: T196 sim-honest rows whose (task, eval_sha256) pairs equal the counted keys' pairs."""
    if len(new) != EXTRA_N:
        raise ValueError(f"fix_v2 adds {len(new)} rows, not {EXTRA_N}")
    if any(r.get("task_set") != "T196" or r.get("world") != "sim_honest" for r in new):
        raise ValueError("a fix_v2 new row is not a T196 sim_honest row")
    if sorted((r["task"], r.get("eval_sha256")) for r in new) != pairs:
        raise ValueError("fix_v2's new rows are not the pinned pilot runs by (task, eval sha256)")
    return {r["key"] for r in new}


RUN_FIELDS = ("run_keys", "run_set", "run_folds", "run_main", "run_cell", "run_epoch", "run_source")  # run_honest_all.py's record (L496-499 @ 99362adcc)


def prior_match(prior: dict, kept: list, new_keys: set) -> dict:
    """09's STOPs (b)+(c), in the container: against the 10-16 run's record (honest_all_2026-10-16_120353.json @ 8b0b18e2a), every run of this
    build except the 20 pilot runs has the same name, key, set, fold, main, cell, epoch and source (so A / B / S_newmains / every other key set
    is equal by key), and the main -> fold map equals the 10-16 map exactly (no re-folding; the pilot's mains included). Keys only."""
    if any(not isinstance(prior.get(f), dict) for f in RUN_FIELDS):
        raise ValueError("the prior record lacks a run_* table")
    names = set(prior["run_keys"])
    if any(set(prior[f]) != names for f in RUN_FIELDS):
        raise ValueError("the prior record's run_* tables differ in names")
    new = {e["name"]: (e["key"], e["set"], e["fold"], e["task"], e["cell"], e["epoch"], e["source"]) for e in kept}
    pilot = {e["name"] for e in kept if e["key"] in new_keys}
    if len(pilot) != len(new_keys):
        raise ValueError(f"{len(pilot)} pilot run names for {len(new_keys)} pilot keys")
    rest = set(new) - pilot
    if rest != names:
        raise ValueError(f"run names differ from the 10-16 run's: {len(names - rest)} only there, {len(rest - names)} only here")
    diff = sum(new[n] != tuple(prior[f][n] for f in RUN_FIELDS) for n in names)
    if diff:
        raise ValueError(f"{diff} runs differ from the 10-16 run's in key, set, fold, main, cell, epoch or source")
    old_map, new_map = {}, {}
    for n in names:
        if old_map.setdefault(prior["run_main"][n], prior["run_folds"][n]) != prior["run_folds"][n]:
            raise ValueError("the 10-16 record maps a main to two folds")
    for e in kept:
        if new_map.setdefault(e["task"], e["fold"]) != e["fold"]:
            raise ValueError("this build maps a main to two folds")
    if new_map != old_map:
        raise ValueError(f"main -> fold map differs from the 10-16 run's ({len(set(new_map) ^ set(old_map))} mains only on one side)")
    return {"equal": True, "n_prior": len(names), "n_pilot": len(pilot), "n_mains": len(new_map)}


def sold_sim(kept: list, new_keys: set) -> dict:
    """The built S_old sim class must be the registered 171 plus exactly the 20 pilot keys (every pilot key kept, as sim in S_old)."""
    sold = {e["key"] for e in kept if e["batch"] == "sim" and e["set"] == "S_old"}
    missing = new_keys - sold
    if missing:
        raise ValueError(f"{len(missing)} pilot runs are not in the built S_old sim class")
    if len(sold) != SOLD_SIM_REGISTERED + EXTRA_N or len(sold - new_keys) != SOLD_SIM_REGISTERED:
        raise ValueError(f"S_old sim class has {len(sold)} runs ({len(sold - new_keys)} registered + {len(sold & new_keys)} pilot), "
                         f"not {SOLD_SIM_REGISTERED} + {EXTRA_N}")
    return {"sold_sim": len(sold), "registered": len(sold - new_keys), "pilot": len(sold & new_keys)}
