"""Resolve a variant arm delivered as a mapping (RULE.md Amendment 19's delivery note): a base arm with an override arm's
runs swapped in for listed keys. Pure logic, no I/O; the fix-full/N/F9 fit job will call resolve() on the metas it reads, before any fit (pending).

  mapping = {"base run": <run>, "base arm": <arm>, "override run": <run>, "override arm": <arm>, "override keys": [keys]}
  resolve(mapping, base_metas, override_metas, expect) -> {key: (run, arm)} for exactly `expect` runs

Raises ValueError (numbers and key counts only) unless: the mapping has exactly these fields; the override keys are
distinct; every override key exists in both the base arm and the override arm; the base arm holds exactly `expect` runs;
the override arm holds no key outside the listed ones (a changed-only delivery); and the result holds exactly `expect`
runs, each key once.
"""

FIELDS = {"base run", "base arm", "override run", "override arm", "override keys"}


def resolve(mapping: dict, base_metas: dict, override_metas: dict, expect: int) -> dict:
    if set(mapping) != FIELDS:
        raise ValueError(f"mapping fields {sorted(mapping)} differ from {sorted(FIELDS)}")
    keys = list(mapping["override keys"])
    if len(set(keys)) != len(keys):
        raise ValueError(f"override keys repeat: {len(keys)} listed, {len(set(keys))} distinct")
    miss_base, miss_over = [k for k in keys if k not in base_metas], [k for k in keys if k not in override_metas]
    if miss_base or miss_over:
        raise ValueError(f"override keys missing: {len(miss_base)} from the base arm, {len(miss_over)} from the override arm")
    if len(base_metas) != expect:
        raise ValueError(f"the base arm holds {len(base_metas)} runs, not {expect}")
    extra = set(override_metas) - set(keys)
    if extra:
        raise ValueError(f"the override arm holds {len(extra)} runs outside the listed keys")
    out = {k: (mapping["base run"], mapping["base arm"]) for k in base_metas}
    out |= {k: (mapping["override run"], mapping["override arm"]) for k in keys}
    if len(out) != expect:
        raise ValueError(f"resolved {len(out)} runs, not {expect}")
    return out


def resolve_sources(variants: dict, metas: dict, base_keys: dict) -> dict:
    """Every variant arm's run -> its source, for the fix-full / N / F9 job (Amendments 15, 17, 19). Pure logic.

    variants: arm -> {"of": "sim-attack" | "sim-honest", "run": ext, "meta_arm": arm in that dump, "n_extracted": int,
                      "fallback": [ext, arm]}                              # a complete or changed-only dump, as-is fallback
              or arm -> {"of": ..., "mapping": {base run, base arm, override run, override arm, override keys}}  # F9
              where the mapping's base arm names another variant arm here (fix-full) and its override run/arm a dump.
    metas: (ext, meta_arm) -> {key: meta} for every dump read.
    base_keys: "sim-attack" -> its 939 keys, "sim-honest" -> its 437 keys.
    Returns arm -> {key: (ext, meta_arm)}. Raises ValueError unless every key resolves once, each dump's run count equals its
    n_extracted, every key it holds is one of the arm's keys, and every fallback exists."""
    out = {}
    for arm, v in variants.items():  # plain variant arms first (mapped ones may build on them)
        if "mapping" in v:
            continue
        keys, held = base_keys[v["of"]], metas[(v["run"], v["meta_arm"])]
        if len(held) != v["n_extracted"]:
            raise ValueError(f"{arm}: the dump holds {len(held)} runs, its record {v['n_extracted']}")
        if set(held) - set(keys):
            raise ValueError(f"{arm}: {len(set(held) - set(keys))} runs outside the cohort")
        fb = metas[tuple(v["fallback"])]
        src = {}
        for k in keys:
            if k in held:
                src[k] = (v["run"], v["meta_arm"])
            elif k in fb:
                src[k] = tuple(v["fallback"])
            else:
                raise ValueError(f"{arm}: an unchanged run has no as-is reading")
        out[arm] = src
    for arm, v in variants.items():  # mapped arms (F9): base = a resolved variant arm, override = the listed keys
        if "mapping" not in v:
            continue
        m = v["mapping"]
        if m["base arm"] not in out or variants[m["base arm"]]["run"] != m["base run"]:
            raise ValueError(f"{arm}: the mapping's base arm/run is not a resolved variant arm here")
        base = out[m["base arm"]]
        over = metas[(m["override run"], m["override arm"])]
        got = resolve(m, base, over, len(base_keys[v["of"]]))
        out[arm] = {k: (base[k] if src == (m["base run"], m["base arm"]) else (m["override run"], m["override arm"])) for k, src in got.items()}
    return out


def newline_reading(n_minus_ff: tuple, n_minus_as: tuple, gaps: tuple, margin: float = 0.10) -> str:
    """RULE.md Amendment 17's reading, in its listed order. n_minus_ff, n_minus_as: (lo, hi) 95% intervals of the N gap minus the
    fix-full gap and minus the as-is gap; gaps: point gaps (N, fix-full, as-is).
    1. "newlines are the main cue": (N - fix-full) inside +-margin, and N below as-is with the interval excluding 0;
    2. "newlines aren't enough alone": (N - as-is) inside +-margin;
    3. "inconclusive": both intervals wider than the margin band (width > 2 * margin);
    4. "partial": N's gap between fix-full's and as-is's;
    5. otherwise "inconclusive".
    Point 3 first tested "neither inside +-margin", which made "partial" almost unreachable; Codex (2026-10-13 16:43 PT, after the Seed
    variants result) ruled it a misreading of "wider than the margin" and required this width test. Post hoc: the first output stays beside it."""
    inside = lambda iv: -margin <= iv[0] and iv[1] <= margin  # noqa: E731
    if inside(n_minus_ff) and n_minus_as[1] < 0:
        return "newlines are the main cue"
    if inside(n_minus_as):
        return "newlines aren't enough alone"
    wide = lambda iv: iv[1] - iv[0] > 2 * margin  # noqa: E731
    if wide(n_minus_ff) and wide(n_minus_as):
        return "inconclusive"
    g_n, g_ff, g_as = gaps
    return "partial" if min(g_ff, g_as) <= g_n <= max(g_ff, g_as) else "inconclusive"
