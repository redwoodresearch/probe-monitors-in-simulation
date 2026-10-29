"""Sol (a)'s content-only command readout (C) on full-length staged inputs: the registered skip rule, unchanged, and 09's rebuild gate
(experiments/2026-10-14_priority1_spec_09/RULE.md @ 36b9b47df, "Full-length lead" §2). Pure functions, no I/O; one implementation for every scorer.

The registered rule (run_scaleup_score.skip_rows / run_asis_fits.b_skip / run_avg_transfer.cmd_prep): a call_end call with a readout is DROPPED when
the staged token id at its readout is one of the fixed call tokens (fixed_call_tokens("q4b_main_v1") = '0', 'n'). Units are (irid, call_index) with
their JSON types kept, as the registered code compares them.
"""

CUT = 32768


def skip_decisions(ids: list, calls: list, fixed_ids: set) -> dict:
    """One staged epoch: {(irid, call_index): (dropped, readout)} for every call_end call with a readout. A unit twice is an error."""
    out = {}
    for r in calls:
        if r["kind"] == "call_end" and r["readout"] is not None:
            u = (r["irid"], r["call_index"])
            if u in out:
                raise ValueError(f"call_end unit {u} twice in one staged epoch")
            out[u] = (ids[r["readout"]] in fixed_ids, r["readout"])
    return out


def dropped(dec: dict) -> list:
    """The registered skip-list format: [[irid, call_index], ...] of the dropped units, in staged order."""
    return [[u[0], u[1]] for u, (s, _) in dec.items() if s]


def compare_skip(registered: dict, rebuilt: dict) -> dict:
    """09's gate for one epoch: on every unit of the REGISTERED (32k) staged inputs, the rebuilt decision equals the registered one, both
    directions (registered drop / rebuilt keep, registered keep / rebuilt drop) and no registered unit missing from the rebuild."""
    a = [list(u) for u, (s, _) in registered.items() if s and u in rebuilt and not rebuilt[u][0]]
    b = [list(u) for u, (s, _) in registered.items() if not s and u in rebuilt and rebuilt[u][0]]
    m = [list(u) for u in registered if u not in rebuilt]
    return {"units": len(registered), "registered_drop_rebuilt_keep": a, "registered_keep_rebuilt_drop": b, "missing_in_rebuilt": m,
            "mismatches": len(a) + len(b) + len(m)}


def past_cut(rebuilt: dict) -> dict:
    """Units of the rebuilt epoch past token 32,768: all, dropped, kept (09: printed per set and arm)."""
    p = [s for s, r in rebuilt.values() if r >= CUT]
    return {"units": len(p), "dropped": sum(p), "kept": len(p) - sum(p)}


def negative_test(registered: dict, rebuilt: dict) -> dict:
    """One perturbed unit (the first registered unit's rebuilt decision flipped) must FAIL compare_skip with exactly one mismatch."""
    u0 = next(iter(registered))
    bad = dict(rebuilt)
    bad[u0] = (not bad[u0][0], bad[u0][1]) if u0 in bad else (not registered[u0][0], registered[u0][1])
    r = compare_skip(registered, bad)
    return {"unit": list(u0), "mismatches": r["mismatches"], "fails_as_required": r["mismatches"] == 1}


def rebuild_skip(rebuilt_epochs: list, registered_epochs: list, fixed_ids: set) -> dict:
    """The rebuild for one arm: the parsed staged epochs ({key, ids, calls}) of the cutfull (rebuilt) and the registered 32k staged inputs ->
    per key the decisions of each side and the dropped lists (the registered skip format). The two files must hold the same epoch keys."""
    dec = {}
    for side, eps in (("rebuilt", rebuilt_epochs), ("registered", registered_epochs)):
        dec[side] = {}
        for ep in eps:
            if ep["key"] in dec[side]:
                raise ValueError(f"{side}: epoch {ep['key']} twice")
            dec[side][ep["key"]] = skip_decisions(ep["ids"], ep["calls"], fixed_ids)
    if set(dec["rebuilt"]) != set(dec["registered"]):
        raise ValueError("the cutfull and the registered staged inputs do not hold the same epoch keys")
    return {"decisions": dec, "lists": {side: {k: dropped(v) for k, v in d.items()} for side, d in dec.items()}}


def gate_report(registered: dict, rebuilt: dict, registered_count=None) -> dict:
    """09's gate for one arm, to be run in its own call: compare_skip per epoch (0 mismatches), the registered dropped count equal to the registered
    pack's count when given, and a negative test (one perturbed unit) that must fail. registered / rebuilt: {key: skip_decisions(...)}."""
    per = {k: compare_skip(registered[k], rebuilt.get(k, {})) for k in registered}  # a key missing from the rebuild counts its units as missing
    mism = sum(v["mismatches"] for v in per.values())
    past = {"units": 0, "dropped": 0, "kept": 0}
    for d in rebuilt.values():
        for k_, v_ in past_cut(d).items():
            past[k_] += v_
    reg_n = sum(s for d in registered.values() for s, _ in d.values())
    k0 = next((k for k in registered if registered[k]), None)
    neg = negative_test(registered[k0], rebuilt.get(k0, {})) | {"key": k0} if k0 is not None else None
    cnt_ok = registered_count is None or reg_n == registered_count
    return {"epochs": len(per), "units_compared": sum(v["units"] for v in per.values()), "mismatches": mism,
            "mismatch_keys": {k: v for k, v in per.items() if v["mismatches"]}, "registered_dropped": reg_n, "registered_count_given": registered_count,
            "registered_count_matches": cnt_ok, "rebuilt_units": sum(len(d) for d in rebuilt.values()),
            "rebuilt_dropped": sum(s for d in rebuilt.values() for s, _ in d.values()), "past_32768": past, "negative_test": neg,
            "passes": set(registered) == set(rebuilt) and mism == 0 and cnt_ok and bool(neg) and neg["fails_as_required"]}
