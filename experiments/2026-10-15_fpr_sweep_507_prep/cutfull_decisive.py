"""Decisive ΔG (Append 41 am.1; driver spec v2.1, experiments/2026-10-15_cut_sensitivity/DECISIVE_DG_DRIVER_SPEC.md @ e99d8c891 on
aswin/cut-sensitivity-2e): pure functions used by run_peraction311.py's `g1` and `decisive` modes and by the stub tests. No I/O here.

Members and groups: the v3 +311 six groups, reply readout (LORO manifest amendment 5 @ 6e7c8d928). Gap per member: the run-mean
Δ_full = sim@S|primary − real_mean@B at joint f = 0.01 for EVERY group (narrowing @ 635b14564); the own-real Δ_full,run2 version beside,
as a labelled sensitivity. G_d = mean over groups of the members' mean; ΔG_d = Gfull_d − G32_d, draw by draw. Verdict labels from am.1 §1.
"""

MEMBERS = ["Sol (a) R", "FIBS R", "harmfulness R", "refusal R", "high-stakes R", "own_frac10_L48 R", "own_frac25_L48 R", "own_frac100_L48 R"]
GROUP_OF = {"Sol (a) R": "other-model-real", "FIBS R": "public:FIBS", "harmfulness R": "public:harmfulness", "refusal R": "public:refusal",
            "high-stakes R": "public:high-stakes", "own_frac10_L48 R": "own-real", "own_frac25_L48 R": "own-real", "own_frac100_L48 R": "own-real"}
OWN_REAL = "own-real"
F_KEY, MODE = "0.01", "joint"
MIN_DRAWS = 9500
SCALE = 100.0  # the producer stores fractions; points are percentage points (as the LORO records)
SENS_LABEL = "sensitivity: real run 1 shares the training launch (S_old only)"
UNAVAIL_C = "unavailable: skip list not rebuilt from cutfull inputs"
CATCHES = ("sim@S|primary", "real_mean@B", "run2@B", "run1@B", "run2@A")
# Uncut sensitivity (09 @ 669ae8e09 §2-§3; POST HOC): the decisive cell's members per readout. Sol (a) C has no uncut file (UNAVAIL_C).
MEMBERS_C = ["FIBS C", "harmfulness C", "refusal C", "high-stakes C", "own_frac10_L48 C", "own_frac25_L48 C", "own_frac100_L48 C"]
SENS_LABEL_ROW = "uncut scoring; probes trained within 32k (OOD class per the table @ 1d02b05ec); POST HOC"
Y_LABEL = "Y = TPR_run2(τ_B) − TPR_run1(τ_B), both uncut at the same uncut τ_B (09 @ 669ae8e09 §3); run1 and run2 share a launch on S_new (no launch drift)"


def members_for(readout):
    return {"R": MEMBERS, "C": MEMBERS_C}[readout]


def group_of(m):
    """The six-group map, readout-free (a C member takes its R member's group)."""
    return GROUP_OF[m.rsplit(" ", 1)[0] + " R"]


def sens_summary(cut, unc, members, np, min_draws=MIN_DRAWS):
    """Per window: S = mean over the groups present of the members' mean Δ_full, and Y the same mean of TPR_run2(τ_B) − TPR_run1(τ_B);
    change = uncut − cut, draw by draw. Per member the same four quantities. Percentile 95% over the registered draws (draw 0 = point).
    Descriptive only: no verdict. For R, the five-group version without Sol (a) is beside (like for like with C, which has no Sol (a))."""
    y = lambda a: a["catch"]["run2@B"] - a["catch"]["run1@B"]  # noqa: E731
    subsets = {"all_members": list(members)}
    if any(m.startswith("Sol (a)") for m in members):
        subsets["without_sol_a (like C)"] = [m for m in members if not m.startswith("Sol (a)")]
    out = {"units": "percentage points (fractions x 100)", "interval": "pointwise percentile 95% over the registered draws (draw 0 = point)",
           "yardstick": Y_LABEL, "delta_rr_tau_A": "unavailable: H_A not re-extracted uncut (09 @ 669ae8e09 §3, optional appendix)", "subsets": {}}
    for nm, ms in subsets.items():
        groups = sorted({group_of(m) for m in ms})
        gm = lambda arrs, f: np.mean(np.stack([np.mean(np.stack([f(arrs[m]) for m in ms if group_of(m) == g]), axis=0)  # noqa: E731
                                               for g in groups], axis=1), axis=1)
        q = {}
        for w, arrs in (("cut", cut), ("uncut", unc)):
            q[f"S_{w}"], q[f"Y_{w}"] = gm(arrs, lambda a: a["delta_full"]), gm(arrs, y)
        q["S_change"], q["Y_change"] = q["S_uncut"] - q["S_cut"], q["Y_uncut"] - q["Y_cut"]
        mask = np.all(np.isfinite(np.stack(list(q.values()))[:, 1:]), axis=0)
        D = int(mask.sum())
        pt_ok = all(np.isfinite(v[0]) for v in q.values())  # Codex r12 #2: the point (draw 0) must be finite too
        blk = {"groups": groups, "members": ms, "draws_used": D, "min_draws": min_draws, "status": "ok" if D >= min_draws and pt_ok else "unavailable"}
        if D >= min_draws and pt_ok:
            blk |= {k: {"point": float(v[0] * SCALE), "ci95": [x * SCALE for x in interval(v, mask, np)]} for k, v in q.items()}
        else:
            blk["why"] = f"|D| = {D} < {min_draws}" if D < min_draws else "a point (draw 0) is not finite"
        out["subsets"][nm] = blk
    pm = {}
    for m in members:
        qm = {"delta_full_cut": cut[m]["delta_full"], "delta_full_uncut": unc[m]["delta_full"], "Y_cut": y(cut[m]), "Y_uncut": y(unc[m])}
        qm["delta_full_change"] = qm["delta_full_uncut"] - qm["delta_full_cut"]
        qm["Y_change"] = qm["Y_uncut"] - qm["Y_cut"]  # Codex r12 #3
        mk = np.all(np.isfinite(np.stack(list(qm.values()))[:, 1:]), axis=0)
        ok_m = mk.sum() >= min_draws and all(np.isfinite(v[0]) for v in qm.values())
        pm[m] = {"group": group_of(m), "draws_used": int(mk.sum())} | (
            {k: {"point": float(v[0] * SCALE), "ci95": [x * SCALE for x in interval(v, mk, np)]} for k, v in qm.items()} if ok_m
            else {"status": "unavailable", "why": f"|D| = {int(mk.sum())} < {min_draws} or a point is not finite"})
    out["per_member"] = pm
    st = [b["status"] for b in out["subsets"].values()]
    out["status"] = "ok" if all(x == "ok" for x in st) else "unavailable"
    arrs = {f"{w}__{m.replace(' ', '_').replace('(', '').replace(')', '')}__{k}": v for w, d in (("cut", cut), ("uncut", unc)) for m in members
            for k, v in (("delta_full", d[m]["delta_full"]), ("Y", y(d[m])))}
    return out, arrs


def block_shas(block, fn):
    """The sha256s a scorer manifest's cut-side block cites for file fn (any entry whose path or name ends with fn's basename); None if no block."""
    if block is None:
        return None
    name, found = fn.rsplit("/", 1)[-1], set()

    def walk(x):
        if isinstance(x, dict):
            p_ = str(x.get("path") or x.get("name") or x.get("file") or "")
            if p_.endswith(name) and isinstance(x.get("sha256"), str):
                found.add(x["sha256"])
            for k, v in x.items():
                if k.endswith(name) and isinstance(v, str) and len(v) == 64:
                    found.add(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(block)
    return sorted(found)


def npz_key(label, origin, catch, mode=MODE, f=F_KEY):
    """The producer's per-draw npz key (pool_reduce's sanitizer)."""
    k = f"{label}|{mode}|{f}|{origin}|{catch}"
    return k.replace("|", "__").replace(" ", "_").replace("(", "").replace(")", "").replace("@", "_at_")


def member_arrays(out, np):
    """From one row() output (return_draws on): the member's per-draw catches at joint f = 0.01 and its gaps (fractions)."""
    g = {k: np.asarray(v, dtype=float) for k, v in out["_draws"][MODE][F_KEY].items()}
    missing = [c for c in CATCHES if c not in g]
    if missing:
        raise SystemExit(f"{out['label']}: per-draw catches missing {missing}")
    return {"catch": {c: g[c] for c in CATCHES},
            "delta_full": g["sim@S|primary"] - g["real_mean@B"],
            "delta_full_run2": g["sim@S|primary"] - g["run2@B"],
            "delta_rr": g["run2@A"] - g["run1@B"]}


def t_estimable(out, need=("B", "S")):
    """am.1 §7: the decisive pools' thresholds at f = 0.01 estimable at draw 0: present, k >= 1 and T a finite number (Codex r1 #6)."""
    import math

    pools = out["pools_at_draw0"][F_KEY]
    bad = [nm for nm in need if not (isinstance(pools.get(nm), dict) and pools[nm].get("k", 0) >= 1
                                     and isinstance(pools[nm].get("T"), (int, float)) and math.isfinite(pools[nm]["T"]))]
    return not bad, sorted(bad)


def group_mean(arrs, np, sens=False):
    """G_d (fractions) from {member: arrays}: per group the members' mean, then the mean over groups. NaN where any member is NaN."""
    groups = sorted(set(GROUP_OF.values()))
    cols = []
    for g in groups:
        ms = [m for m in MEMBERS if GROUP_OF[m] == g]
        key = (lambda m: "delta_full_run2" if (sens and GROUP_OF[m] == OWN_REAL) else "delta_full")
        cols.append(np.mean(np.stack([arrs[m][key(m)] for m in ms]), axis=0))
    return np.mean(np.stack(cols, axis=1), axis=1), groups


def interval(x, mask, np):
    v = x[1:][mask]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def verdict(gfull_ci, dg_ci):
    """am.1 §1 labels (endpoints written out; boundary equality satisfies no verdict)."""
    dg_below = dg_ci[1] < 0
    dg_inc0 = dg_ci[0] <= 0 <= dg_ci[1]
    gf_above = gfull_ci[0] > 0
    gf_inc0 = gfull_ci[0] <= 0 <= gfull_ci[1]
    if dg_below and gf_inc0:
        return "Reduction detected; residual unresolved"
    if dg_below and gf_above:
        return "Reduction detected; positive residual remains"
    if gf_above and dg_inc0:
        return "Positive residual remains; change unresolved"
    return "inconclusive"


def decide(cut, uncut, estimable, np, min_draws=MIN_DRAWS, yardstick=False):
    """cut / uncut: {member: member_arrays()} (fractions). estimable: {(window, member): bool}. Returns (result, per-draw arrays)."""
    missing = [m for m in MEMBERS if m not in cut or m not in uncut]
    if missing:
        return {"status": "unavailable", "why": f"members missing: {missing}"}, None
    not_est = sorted(f"{w}|{m}" for (w, m), ok in estimable.items() if not ok)
    if not_est:
        return {"status": "unavailable", "why": f"non-estimable T at draw 0 (am.1 §7): {not_est}"}, None
    g32, groups = group_mean(cut, np)
    gfull, _ = group_mean(uncut, np)
    s32, _ = group_mean(cut, np, sens=True)
    sfull, _ = group_mean(uncut, np, sens=True)
    n = g32.shape[0]
    if any(a.shape != (n,) for a in (gfull, s32, sfull)):
        raise SystemExit("per-draw arrays differ in length")
    if not (np.isfinite(g32[0]) and np.isfinite(gfull[0])):
        return {"status": "unavailable", "why": "draw 0 non-finite"}, None
    allm = np.stack([cut[m]["delta_full"] for m in MEMBERS] + [uncut[m]["delta_full"] for m in MEMBERS])
    mask = np.all(np.isfinite(allm[:, 1:]), axis=0)
    D = int(mask.sum())
    dg = gfull - g32
    out = {"status": "ok" if D >= min_draws else "unavailable", "draws_total": int(n - 1), "draws_used": D, "min_draws": min_draws, "groups": groups,
           "units": "percentage points (fractions x 100)"}
    if D < min_draws:
        out["why"] = f"|D| = {D} < {min_draws}"
        return out, None
    sc = SCALE
    ci = {k: interval(v, mask, np) for k, v in (("G32", g32), ("Gfull", gfull), ("dG", dg))}
    out |= {"G32": {"point": float(g32[0] * sc), "ci95": [ci["G32"][0] * sc, ci["G32"][1] * sc]},
            "Gfull": {"point": float(gfull[0] * sc), "ci95": [ci["Gfull"][0] * sc, ci["Gfull"][1] * sc]},
            "dG": {"point": float(dg[0] * sc), "ci95": [ci["dG"][0] * sc, ci["dG"][1] * sc]},
            "verdict": verdict(ci["Gfull"], ci["dG"])}
    smask = mask & np.all(np.isfinite(np.stack([cut[m]["delta_full_run2"] for m in MEMBERS if GROUP_OF[m] == OWN_REAL]
                                                + [uncut[m]["delta_full_run2"] for m in MEMBERS if GROUP_OF[m] == OWN_REAL])[:, 1:]), axis=0)
    sdg = sfull - s32
    out["sensitivity_own_real_run2"] = {"label": SENS_LABEL, "draws_used": int(smask.sum()),
                                        **{k: {"point": float(v[0] * sc), "ci95": [x * sc for x in interval(v, smask, np)]} for k, v in (("G32", s32), ("Gfull", sfull), ("dG", sdg))}}
    out["per_member"] = {m: {"group": GROUP_OF[m], "delta_full_cut": float(cut[m]["delta_full"][0] * sc), "delta_full_uncut": float(uncut[m]["delta_full"][0] * sc),
                             "change": float((uncut[m]["delta_full"][0] - cut[m]["delta_full"][0]) * sc),
                             "yardstick_delta_rr_cut": float(cut[m]["delta_rr"][0] * sc),
                             "yardstick_delta_rr_uncut": (float(uncut[m]["delta_rr"][0] * sc) if yardstick else
                                                          "unavailable: H_A differs by key between windows or its 25f roots are not re-extracted")} for m in MEMBERS}
    arrs = {"G32": g32, "Gfull": gfull, "dG": dg, "mask": mask}
    for m in MEMBERS:
        k = m.replace(" ", "_").replace("(", "").replace(")", "")
        arrs[f"{k}__delta_full_cut"], arrs[f"{k}__delta_full_uncut"] = cut[m]["delta_full"], uncut[m]["delta_full"]
        arrs[f"{k}__delta_rr_cut"] = cut[m]["delta_rr"]
        if yardstick:
            arrs[f"{k}__delta_rr_uncut"] = uncut[m]["delta_rr"]
    return out, arrs


def apply_override(cfg, ov, members, snew_prefix="su311_"):
    """The uncut window's cfg: every S_new arm folder's file for each member probe, and every member's H_sim file list, replaced from the
    override (paths on the conv Volume under cutfull/...). Fail closed: an S_new arm or H_sim root without its uncut file stops the job.
    Returns (cfg_uncut, applied) without modifying cfg."""
    import copy

    c = copy.deepcopy(cfg)
    c["cutfull_uncut"] = True  # row(): G455 compares CUT catches, so the uncut window skips it (the cut window keeps it)
    news = {p["label"]: p["new"] for p in c["probes"] if p["label"] in members}
    if sorted(news) != sorted(members):
        raise SystemExit(f"override: probes missing from the config: {sorted(set(members) - set(news))}")
    applied, missing = [], []
    for a in c["arms"]:
        if not a["arm"].startswith(snew_prefix):
            continue
        for lb, fn in news.items():
            if fn not in a["files"]:
                continue
            e = ov.get("arms", {}).get(a["arm"], {}).get(fn)
            if e is None:
                missing.append(f"{a['arm']}|{fn}")
                continue
            applied.append({"arm": a["arm"], "file": fn, "cut": a["files"][fn], "uncut": {"path": e["path"], "sha256": e["sha256"]},
                            "runs_without_rows": list(e.get("runs_without_rows", [])),  # the scorer's zero-unit runs (Codex r3 #1)
                            "cut_side_block": e.get("cut_side_registered_score_files"), "staged_shas": list(e.get("staged_shas") or [])})
            a["files"][fn] = {"path": e["path"], "sha256": e["sha256"]}
    for lb, fn in news.items():
        cut_list = c["simh_files"].get(fn, [])
        unc = ov.get("simh", {}).get(fn)
        if unc is None:
            missing.append(f"H_sim|{fn}")
            continue
        if sorted(e["root"] for e in unc) != sorted(e["root"] for e in cut_list):
            raise SystemExit(f"override H_sim {fn}: roots differ from the cut window's ({len(unc)} vs {len(cut_list)})")
        cut_by_root = {e["root"]: e for e in cut_list}
        applied += [{"simh": fn, "root": e["root"], "cut": {"path": cut_by_root[e["root"]]["path"], "sha256": cut_by_root[e["root"]]["sha256"]},
                     "uncut": {"path": e["path"], "sha256": e["sha256"]}, "runs_without_rows": list(e.get("runs_without_rows", [])),
                     "staged_shas": list(e.get("staged_shas") or [])} for e in unc]
        c["simh_files"][fn] = [dict(e, batch=cut_by_root[e["root"]]["batch"]) for e in unc]  # the cut entry's batch (exclusions are by batch)
    if missing:
        raise SystemExit(f"override: uncut files missing (the decisive test would be unavailable): {missing[:10]} (+{max(0, len(missing) - 10)})")
    return c, applied


STEP0_KEYLIST = "cf5c4f5e3e0e0664"  # the registered step-0 key list (step0_keys_v2_2026-10-15.json sha256 prefix; aswin/cut-sensitivity-2e @ f3bfe9208)
STEP0_SPEC_SHA256 = "17062a1ac155b3fe12690b770d4ebe2d98ef4f0effea1c7b7f4716ecf2dc46e6"  # step0_spec_v2 (same 26 keys) that d9's step-0 record cites
DECISIVE_POOLS = ("cohort_cells", "H_B", "H_sim")


ROUTING_DIR = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing"


def reattest_map(pin, git, folders, n_epochs):
    """d9's re-attestation at a NEW routing revision (09 via 4f). pin = '<pointer commit>@<rev>x<n>,...' (e.g. bccb0f153835@3x40,4x5): the
    (a'') pointer files committed at <pointer commit> -> each routing record -> per-arm staged sha and every-epoch format check (git reads only).
    Fails closed unless: every pointer's revision is the highest for its target at the pin, is newer than the revision it states the staged
    sets / roots / per-arm staged shas are unchanged from, and has no failing set; every arm's format check passes on all its epochs with
    no failing keys or shard errors and cites the arm's staged sha; the folders are exactly the inventory's and the epochs sum to its runs;
    the revision counts equal the pin's. Returns ({original folder: staged sha}, summary)."""
    import json
    import re
    from pathlib import PurePosixPath

    commit, want = pin.split("@", 1)
    want = {int(r): int(n) for r, n in (x.split("x") for x in want.split(","))}
    names = [n for n in git("show", "--name-only", "--format=", commit).decode().split()
             if re.fullmatch(rf"{re.escape(ROUTING_DIR)}/[^/]+_cutfull_r\d+_attest\.json", n)]
    tree = [PurePosixPath(t).name for t in git("ls-tree", "--full-tree", "--name-only", commit, ROUTING_DIR + "/").decode().split()]
    out, revs, epochs, bad, recs = {}, {}, 0, [], []
    for n in sorted(names):
        p = json.loads(git("show", f"{commit}:{n}"))
        stem = PurePosixPath(n).name.removesuffix("_attest.json")
        base, k = stem.rsplit("_r", 1)
        top = max((int(m.group(1)) for t in tree if (m := re.fullmatch(rf"{re.escape(base)}_r(\d+)\.json", t))), default=-1)
        rp, rc = [x.strip() for x in p["routing_record"].split("@")]
        rec = json.loads(git("show", f"{rc}:{rp}"))
        u = rec.get("unchanged_from_revision") or {}
        if (p.get("failing") != {} or not p.get("sets") or int(k) != p.get("revision") or rec.get("revision") != p["revision"]
                or p["revision"] != top or rp != f"{ROUTING_DIR}/{stem}.json"):
            bad.append(f"{stem}: pointer/record revision, highest-revision or failing-set check")
        if not (u.get("staged_sha_per_arm") is True and u.get("staged_set") is True and u.get("extraction_root") is True
                and isinstance(u.get("revision"), int) and u["revision"] < rec.get("revision", -1)):
            bad.append(f"{stem}: not stated unchanged from an earlier revision")
        for a in rec.get("arms") or []:
            folder = (rec.get("original_folders") or {}).get(str(a.get("meta_arm", "")).split("@")[0])
            fc, sh = a.get("format_check"), a.get("staged_sha")
            if not (folder and fc and isinstance(sh, str) and re.fullmatch(r"[0-9a-f]{64}", sh)):
                bad.append(f"{stem}|{a.get('meta_arm')}: no folder, format check or staged sha")
                continue
            fp, fcm = [x.strip() for x in fc.split("@")]
            f = json.loads(git("show", f"{fcm}:{fp}"))
            if not (f.get("passes") is True and f.get("failing_keys") == [] and f.get("shard_errors") == [] and sh in (f.get("staged_sha") or [])
                    and f.get("epochs_checked") == a.get("epochs")):
                bad.append(f"{stem}|{a['meta_arm']}: format check does not pass on every epoch at this staged sha")
            if folder in out and out[folder] != sh:
                bad.append(f"{folder}: two staged shas")
            out[folder] = sh
            epochs += a.get("epochs") or 0
            recs.append({"folder": folder, "staged_sha": sh, "routing_record": f"{rp} @ {rc}", "format_check": fc})
        revs[p["revision"]] = revs.get(p["revision"], 0) + 1
    if set(out) != set(folders):
        bad.append(f"folders differ from the inventory's: {sorted(set(folders) ^ set(out))[:5]}")
    if epochs != n_epochs:
        bad.append(f"epochs {epochs} != the inventory's {n_epochs} runs")
    if revs != want:
        bad.append(f"revision counts {revs} != the pin's {want}")
    if bad:
        raise SystemExit(f"--reattest fails ({len(bad)}): {bad[:6]}")
    return out, {"pin": pin, "pointers": len(names), "revisions": {str(r): n for r, n in sorted(revs.items())}, "folders": len(out),
                 "epochs": epochs, "arms": recs}


def reattest_bad(ov, st):
    """Override entries whose scorer pack does not cite its folder's re-attested staged sha (every entry is checked; returns the bad ones)."""
    bad, n = [], 0
    for folder, files in (ov.get("arms") or {}).items():
        for fn, e in files.items():
            n += 1
            if st.get(folder) is None or st[folder] not in (e.get("staged_shas") or []):
                bad.append(f"{folder}|{fn}")
    for fn, ents in (ov.get("simh") or {}).items():
        for e in ents:
            n += 1
            if st.get(e.get("root")) is None or st[e["root"]] not in (e.get("staged_shas") or []):
                bad.append(f"{e.get('root')}|{fn}")
    return bad, n


def boundary_units(s0):
    """The step-0 compare's boundary-excluded units by key (am.1 §3), for the record: [{run_key, unit}]."""
    out = [{"run_key": k, "unit": "|".join(map(str, u))} for k, us in (s0.get("excluded_boundary_units") or {}).items() for u in us]
    for r in s0.get("runs", []):
        for u in (r.get("checks", {}).get("units", {}).get("excluded_boundary") or []):
            x = {"run_key": r.get("key"), "unit": "|".join(map(str, u))}
            if x not in out:
                out.append(x)
    return out


def pools_equal(cut_o, unc_o):
    """0d: the decisive pools and cohort must be EQUAL BY KEY across windows for every member (row()'s pool_keys: n + sha256 of the sorted
    keys); H_A (yardstick only) separately. A missing or different key set makes the test unavailable, never a shrunk pool."""
    diffs, ha_diffs = [], []
    for c, u in zip(cut_o, unc_o):
        if c["label"] != u["label"] or "pool_keys" not in c or "pool_keys" not in u:
            diffs.append(f"{c['label']}: pool_keys missing or rows misaligned")
            continue
        for k in DECISIVE_POOLS:
            if c["pool_keys"][k] != u["pool_keys"][k]:
                diffs.append(f"{c['label']}|{k}: cut {c['pool_keys'][k]['n']} vs uncut {u['pool_keys'][k]['n']}")
        if c["pool_keys"]["H_A"] != u["pool_keys"]["H_A"]:
            ha_diffs.append(f"{c['label']}|H_A")
    return {"decisive_pass": not diffs, "differences": diffs, "H_A_pass": not ha_diffs, "H_A_differences": ha_diffs,
            "pools": list(DECISIVE_POOLS), "rule": "equal by key (sha256 of sorted keys) per member, cut vs uncut; any difference -> unavailable"}


def ood_by_member(table, members=None):
    """Append 41 §5 amendment: each member's OOD class and caveat from 09's signed training-length table (rows: probe "<name> (R...)",
    class, class_key, optional caveat; classes: {key: wording}; pending: [probes]). Fail closed: a member with no row, no class_key,
    or listed as pending stops the decisive run."""
    import re

    classes = table.get("classes") or {}
    pending = {re.sub(r"\s*\((?:R|C|STRESS)[^)]*\)\s*$", "", x) for x in (table.get("pending") or [])}
    idx = {}
    for r in table.get("rows") or []:
        pr = r.get("probe") or ""
        mo = re.match(r"^(.*?)\s*\((R|C)\b[^)]*\)\s*$", pr)
        if mo:
            idx[(mo.group(1), mo.group(2))] = r
    out, missing = {}, []
    for m in members or MEMBERS:
        base, ro = m.rsplit(" ", 1)
        r = idx.get((base, ro))
        key = r and r.get("class_key")
        cav = r and (r.get("caveat") or classes.get(key))
        if base in pending or not (key and cav):
            missing.append(m + (" (pending in the table)" if base in pending else ""))
            continue
        out[m] = {"class_key": key, "class": r.get("class"), "caveat": cav}
    if missing:
        raise SystemExit(f"OOD table: no class/caveat for {missing}")
    return out
