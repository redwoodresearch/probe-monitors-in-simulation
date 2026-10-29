"""Clean-room recompute of route C's downstream group steps (addendum 20, a1, 2026-10-20): pw_shares, group_catch, group_excess four_probe.

Definitions (registered; no production code read; SPEC_READING addendum 20 cites each line):
  pw_shares   per label x level x side: the run-weighted share of the pooled-world honest pool's runs with >= 1 unit above that pool's
              draw-0 threshold, zero-unit runs counted non-alarming (RULE l.4525, l.3642); one value per label = my A18 value, which
              must be equal on all 4 pooled origin rows (else UNRESOLVED). S: verdict. B and S - B: diagnostic under two readings,
              l.3002 (structural run left out on honest_b; 09's dated line) and the record's convention (structural run counted).
  group means equal-weight mean over the registered members (six: RULE l.4153 / l.4870; four: l.4537, l.4843-4844), C lead and R
              (catch) or C (excess), joint, 1e-2, pooled_all_origins. Point = mean of MY draw-0 member values (A18 comparison file).
              Interval = 95% percentile interval of the per-draw group mean over the members' COMMON finite draws (l.4803, l.4807,
              l.4936) of A's per-draw npz (a shared input; G7b compared it). G_p = 0.5|sim - run1| + 0.5|sim - run2|,
              Y_p = |run2 - run1| at tau_B (l.4840, l.4925; READ_LOG l.830; r3).
  controls    the draw-index rule and the percentile method are those that reproduce A's own member intervals (gaps; then catch),
              and the group rule must reproduce A's six-probe panel_groups node; a control that fails makes the rows it gates UNRESOLVED.
The laptop gets booleans and counts only; values and the full comparison go to the output Volume.

  DS_CFG=<folder of the pinned inputs> uv run --no-project --python 3.12 --with modal==1.6.1 modal run cleanroom_downstream.py
"""

import json
import os
import time
from pathlib import Path

import modal

OUT = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
NPZ_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
NPZ = ("xfer/fpr507prep/alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327_perdraw.npz",
       "5e6cd10e382d8b03d409982c67efd72512176d7e95d2f3c94752d2b90e5650d1")
CMP = ("cleanroom_comparison_A18_2026-10-20_040452.json", "584c9d8c9564df728aaa692bc03c696446100bd1e68ba17406d64f24571a61d7")
REC = "routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz"
PINS = {REC: "191c467ab095b0f9ddfe2271822655bc5b49ee02a1a2c5a5b0e11bc855a00865",
        "group_catch.json": "f9dfda2349e2ead23b543eccd7a722aa125513e50e766e141eb61b30c3a7fae1",
        "group_excess.json": "f4fe20de538d63c1d2e769755220b3d76574e8b80213c9d2b0439d2b70fe8169",
        "pw_shares.json": "32a6549f1a40319442f703b3fe694728bdcb62cb2de8781c634f1f888b436bae"}
A_SHA = "54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a"  # A decompressed
COMBO, MODE, LVL = "pooled_all_origins", "joint", "0.01"
SIX = ["sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48"]
FOUR = ["sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "own_all_rs_uncut_L48"]
GROUPS = {"six": SIX, "auroc_subset": FOUR}
FLOOR = {f"{n} {ro}" for n in SIX for ro in ("C", "R")}  # 20b: Figs 1/6's floor_key labels (09, by key in the figure sources)
ARMS = ("real_mean", "sim")
CATCH_O = {"real_mean": "real_mean@B", "sim": "sim@S|primary"}  # 20c (R2 P1): A's per-origin catch names (arms_def catch_key; points.catch)
GAPS = ("delta_full", "sim_minus_run1", "sim_minus_run2", "run2_minus_run1_at_tauB")
METHODS = ("linear", "hazen", "weibull", "median_unbiased", "normal_unbiased", "interpolated_inverted_cdf", "inverted_cdf",
           "averaged_inverted_cdf", "closest_observation", "lower", "higher", "nearest", "midpoint")  # numpy order; linear first
RTOL = 1e-9

app = modal.App(OUT)
OUT_VOL = modal.Volume.from_name(OUT, version=2)
SVOL = modal.Volume.from_name(NPZ_VOL)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if modal.is_local():
    for f in PINS:
        image = image.add_local_file(str(Path(os.environ["DS_CFG"]) / f), f"/ds/{f}")


def tol(ref):
    return RTOL * max(1.0, abs(ref))


def enc(s):
    """The npz key_format's part encoding (A's .npz.key_format): '@' -> '_at_', '|' -> '_', ' ' -> '_', parentheses dropped."""
    return s.replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")


@app.function(image=image, cpu=2, memory=8192, timeout=1200, retries=0, volumes={"/o": OUT_VOL, "/sv": SVOL.read_only()})
def run(stamp: str) -> dict:
    import gzip
    import hashlib

    import numpy as np

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    raw = {f: Path("/ds", f).read_bytes() for f in PINS}
    for f, want in PINS.items():
        assert sha(raw[f]) == want, f"input sha {f}"
    araw = gzip.decompress(raw[REC])
    assert sha(araw) == A_SHA, "A decompressed sha"
    A = json.loads(araw)
    gc, ge, pw = (json.loads(raw[f]) for f in ("group_catch.json", "group_excess.json", "pw_shares.json"))
    craw = Path("/o", CMP[0]).read_bytes()
    assert sha(craw) == CMP[1], "A18 comparison sha"
    cmp = json.loads(craw)["comparison"]
    by = {}
    for r in cmp:
        by.setdefault(r["what"], []).append(r)
    h = hashlib.sha256()
    with open(Path("/sv", NPZ[0]), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    assert h.hexdigest() == NPZ[1], "npz sha"
    Z = np.load(Path("/sv", NPZ[0]))
    names = set(Z.files)
    stops, rows, controls = [], [], []
    for nm, rec in (("group_catch", gc), ("group_excess", ge)):  # the group records must be on A, the npz and the registered cell
        m_ = rec.get("meta") or {}
        for k_, want in (("record.sha256", A_SHA), ("npz.sha256", NPZ[1]), ("combo", COMBO), ("mode", MODE), ("level", LVL)):
            got = m_
            for part in k_.split("."):
                got = (got or {}).get(part) if isinstance(got, dict) else None
            if got != want:
                stops.append(f"{nm} meta.{k_} != pinned")
    if (ge.get("meta") or {}).get("readout") != "C":
        stops.append("group_excess meta.readout != C")
    if sorted(pw.get("shares") or {}) != sorted(A["pooled"]) or len(A["pooled"]) != 18:
        stops.append("pw_shares labels != A's 18 pooled labels")
    rf = (pw.get("provenance") or {}).get("record_file") or {}
    if rf.get("sha256_pin") != A_SHA or rf.get("sha256_as_read") != A_SHA:  # 20c (R1 P2): pw_shares must be built on A
        stops.append("pw_shares provenance.record_file sha != A")

    def mine(what):  # one row of MY A18 comparison, by name; absent or ambiguous STOPs
        rs = by.get(what) or []
        if len(rs) != 1 or not isinstance(rs[0].get("mine"), (int, float)) or isinstance(rs[0].get("mine"), bool):
            stops.append(f"mine {what}: {len(rs)} rows")
            return None, None
        return float(rs[0]["mine"]), float(rs[0].get("flip_bound") or 0.0)

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)

    def row(fam, key, m, ref, bound, verdict, kind="value", note=None):
        r = {"family": fam, "key": key, "verdict": verdict, "kind": kind, "mine": m, "ref": ref, "bound": bound, "note": note}
        if kind == "bool":
            r["match"] = bool(isinstance(ref, bool) and m is ref)
        elif kind == "int":
            r["match"] = bool(isinstance(ref, int) and not isinstance(ref, bool) and m == ref)
        elif kind == "unresolved":
            r["match"] = None
        else:
            r["match"] = bool(num(m) and num(ref) and abs(m - ref) <= (bound or 0.0) + tol(ref))
            r["match_x100"] = bool(num(m) and num(ref) and abs(100 * m - ref) <= 100 * (bound or 0.0) + tol(ref))  # diagnostic only
        rows.append(r)

    def arr(label, origin, name):
        k = "__".join([enc(label), enc(COMBO), MODE, LVL, enc(origin), name])
        if k not in names:
            stops.append(f"npz key absent: {k}")
            return None
        return np.asarray(Z[k], dtype=float)

    def summ(x, idx, method):
        x = x[1:] if idx == "drop0" else x
        f = x[np.isfinite(x)]
        lo, hi = (np.percentile(f, [2.5, 97.5], method=method) if len(f) else (np.nan, np.nan))
        return {"draws": int(len(x)), "finite_draws": int(len(f)), "ci95": [float(lo), float(hi)]}

    def same(s, node):  # a summary vs a record node: ci95 within b, draws / finite_draws exact
        c = node.get("ci95") or [None, None]
        return (s["draws"] == node.get("draws") and s["finite_draws"] == node.get("finite_draws")
                and all(num(c[i]) and abs(s["ci95"][i] - c[i]) <= tol(c[i]) for i in (0, 1)))

    def pnode(label):
        return A["pooled"][label][COMBO][MODE][LVL]

    # ---- (1) member gap control: identifies the draw-index rule and the percentile method on A's own pooled gap intervals
    gap = {(n, g): arr(f"{n} C", "pooled", f"gap_{enc(g)}") for n in SIX for g in GAPS}
    if stops:
        return {"stamp": stamp, "stops": stops[:40], "n_stops": len(stops)}
    idx_rule = {}
    for (n, g), x in gap.items():
        d = pnode(f"{n} C")["gap"][g].get("draws")
        idx_rule[(n, g)] = "all" if len(x) == d else "drop0" if len(x) == d + 1 else None
    rules = set(idx_rule.values())
    IDX = rules.pop() if len(rules) == 1 and None not in rules else None
    METHOD = None
    if IDX:
        for mth in METHODS:
            if all(same(summ(x, IDX, mth), pnode(f"{n} C")["gap"][g]) for (n, g), x in gap.items()):
                METHOD = mth
                break
    controls.append({"name": "member_gap", "n": len(gap), "idx_rule": IDX, "method": METHOD, "pass": bool(IDX and METHOD)})
    other = {"all": "drop0", "drop0": "all"}.get(IDX)  # negative control N2: the other index rule must fail >= 1 member
    n2 = bool(other and METHOD and not all(same(summ(x, other, METHOD), pnode(f"{n} C")["gap"][g]) for (n, g), x in gap.items()))
    controls.append({"name": "N2_index_rule_detected", "pass": n2})

    def gmean(mem, f):  # per-draw group mean over members on their COMMON finite draws (l.4936); NaN where any member is non-finite
        M = np.vstack([f(n) for n in mem])
        return np.where(np.all(np.isfinite(M), axis=0), M.mean(axis=0), np.nan)

    def Gd(n):
        return 0.5 * np.abs(gap[(n, "sim_minus_run1")]) + 0.5 * np.abs(gap[(n, "sim_minus_run2")])

    def Yd(n):
        return np.abs(gap[(n, "run2_minus_run1_at_tauB")])

    def mine_GY(n, q):
        return mine(f"{n} C|{MODE}|{LVL}|panel_{q}")

    def excess(mem):  # point (mine, draw 0) and per-draw arrays (npz) for mean_G, mean_Y, mean_G_minus_mean_Y
        mg, my = [mine_GY(n, "G") for n in mem], [mine_GY(n, "Y") for n in mem]
        if any(v is None for v, _ in mg + my):
            return None
        pg, py = np.mean([v for v, _ in mg]), np.mean([v for v, _ in my])
        bg, by_ = np.mean([b for _, b in mg]), np.mean([b for _, b in my])
        dg, dy = gmean(mem, Gd), gmean(mem, Yd)
        return {"mean_G": (pg, bg, dg), "mean_Y": (py, by_, dy), "mean_G_minus_mean_Y": (pg - py, bg + by_, dg - dy)}

    # ---- (2) six-group control: my group rule on A's own six-probe excess node (panel_groups headline_six_full|C primary)
    six_ok, ex6 = None, None
    if METHOD:
        ex6 = excess(SIX)
        node6 = A["panel_groups"][COMBO]["pooled"][MODE][LVL]["headline_six_full|C"]["primary"]
        res6 = []
        for q, (p, b, d) in (ex6 or {}).items():
            s, nd = summ(d, IDX, METHOD), node6[q]
            res6.append(same(s, nd) and num(nd.get("point")) and abs(p - nd["point"]) <= b + tol(nd["point"]))
        six_ok = bool(ex6) and len(res6) == 3 and all(bool(x) for x in res6)
        controls.append({"name": "six_group_excess", "n": len(res6), "n_true": int(sum(res6)), "pass": six_ok})

    # ---- (3) group_excess four_probe (verdict) and its 17g strings (record consistency)
    fp = ge["four_probe"]
    reg4 = sorted(FOUR)
    for k_, m_ in (("rows", fp.get("rows")), ("meta.four_membership.members", ge["meta"]["four_membership"].get("members"))):
        got = sorted(str(x).removesuffix(" C") for x in (m_ or []))
        row("group_excess", f"four_probe.{k_}=registered four", got == reg4, True, None, True, "bool")
    row("group_excess", "four_probe.n", len(FOUR), fp.get("n"), None, True, "int")
    ex4 = excess(FOUR)  # points never depend on the interval controls
    for q in ("mean_G", "mean_Y", "mean_G_minus_mean_Y"):
        nd = fp[q]
        if ex4 is None:
            for part in ("point", "ci95[0]", "ci95[1]", "draws", "finite_draws"):
                row("group_excess", f"four_probe.{q}.{part}", None, None, None, True, "unresolved", "mine absent")
            continue
        p, b, d = ex4[q]
        row("group_excess", f"four_probe.{q}.point", float(p), nd.get("point"), float(b), True)
        g_ok = bool(METHOD and six_ok)  # (1) and (3) gate the interval rows
        s = summ(d, IDX, METHOD) if g_ok else None
        for i in (0, 1):
            if g_ok:
                row("group_excess", f"four_probe.{q}.ci95[{i}]", s["ci95"][i], (nd.get("ci95") or [None, None])[i], 0.0, True)
            else:
                row("group_excess", f"four_probe.{q}.ci95[{i}]", None, None, None, True, "unresolved", "control (1) or (3) failed")
        for part in ("draws", "finite_draws"):
            if g_ok:
                row("group_excess", f"four_probe.{q}.{part}", s[part], nd.get(part), None, True, "int")
            else:
                row("group_excess", f"four_probe.{q}.{part}", None, None, None, True, "unresolved", "control (1) or (3) failed")
    s17 = ge.get("four_probe_excess_17g") or {}
    fl = fp["mean_G_minus_mean_Y"]
    for part, sv, fv in (("point", s17.get("point"), fl.get("point")),
                         ("ci95[0]", (s17.get("ci95") or [None, None])[0], (fl.get("ci95") or [None, None])[0]),
                         ("ci95[1]", (s17.get("ci95") or [None, None])[1], (fl.get("ci95") or [None, None])[1])):
        ok = isinstance(sv, str) and num(fv) and float(sv) == fv
        rows.append({"family": "group_excess_17g", "key": f"four_probe_excess_17g.{part}==float key", "verdict": False,
                     "record_consistency": True, "kind": "bool", "match": bool(ok)})
    # negative control N1: the six's members in place of the four must NOT reproduce the four_probe points
    if ex4 and ex6:
        n1 = any(not (num(fp[q].get("point")) and abs(ex6[q][0] - fp[q]["point"]) <= ex6[q][1] + tol(fp[q]["point"]))
                 for q in ("mean_G", "mean_Y", "mean_G_minus_mean_Y"))
        controls.append({"name": "N1_membership_detected", "pass": bool(n1)})

    # ---- (4) group_catch: member catch control (A's pooled catch nodes), then the four groups (verdict)
    origins = sorted(pnode(f"{SIX[0]} C")["D_draw0"].keys())

    def pooled_catch(label, a):
        k = "__".join([enc(label), enc(COMBO), MODE, LVL, "pooled", f"catch_{enc(a)}"])
        if k in names:
            return np.asarray(Z[k], dtype=float), "direct"
        num_, den, bad = 0.0, 0.0, None
        for o in origins:  # rebuild (SPEC-AMBIGUOUS reading, identified only by the member control): D-weighted mean over origins
            op = enc(o.split(" [pooled:")[0])  # 20d (listing @ npz_keys20d_2026-10-20_121835): the origin part drops " [pooled: …]"
            kc, kw = ("__".join([enc(label), enc(COMBO), MODE, LVL, op, n_]) for n_ in (f"catch_{enc(CATCH_O[a])}", "D"))
            if kc not in names or kw not in names:  # 20c (R1 P2): an absent array makes the catch intervals UNRESOLVED, not a STOP
                absent.append(kc if kc not in names else kw)
                return None, "absent"
            c, w = np.asarray(Z[kc], dtype=float), np.asarray(Z[kw], dtype=float)
            pos = w > 0
            bad = (pos & ~np.isfinite(c)) if bad is None else (bad | (pos & ~np.isfinite(c)))
            num_ = num_ + np.where(pos, w * np.nan_to_num(c), 0.0)
            den = den + np.where(pos, w, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(bad | (den <= 0), np.nan, num_ / den)
        return out, "rebuilt"

    cat, how, absent = {}, set(), []
    for n in SIX:
        for ro in ("C", "R"):
            for a in ARMS:
                x, hw = pooled_catch(f"{n} {ro}", a)
                cat[(n, ro, a)] = x
                how.add(hw)
    catch_ok = None
    if METHOD and all(v is not None for v in cat.values()):
        res = [same(summ(x, IDX, METHOD), pnode(f"{n} {ro}")["catch"][a]) for (n, ro, a), x in cat.items()]
        catch_ok = all(bool(x) for x in res)
        # 20d diagnostic (not gating): the rebuilt per-draw pooled sim - real_mean equals the npz's pooled gap_delta_full within b
        ident = []
        for n in SIX:
            for ro in ("C", "R"):
                kg = "__".join([enc(f"{n} {ro}"), enc(COMBO), MODE, LVL, "pooled", "gap_delta_full"])
                if kg not in names:  # 20d (R1 / R2 P2): the diagnostic never crashes the job
                    ident.append(None)
                    continue
                g = np.asarray(Z[kg], dtype=float)
                dlt = cat[(n, ro, "sim")] - cat[(n, ro, "real_mean")]
                if g.shape != dlt.shape:
                    ident.append(None)
                    continue
                f = np.isfinite(g) & np.isfinite(dlt)
                ident.append(bool(np.array_equal(np.isfinite(g), np.isfinite(dlt)) and np.all(np.abs(dlt[f] - g[f]) <= RTOL * np.maximum(1.0, np.abs(g[f])))))
        controls.append({"name": "member_catch", "how": sorted(how), "n": len(res), "n_true": int(sum(res)), "pass": catch_ok,
                         "diag_rebuilt_delta_full_eq_pooled_gap": {"n": len(ident), "n_true": sum(x is True for x in ident),
                                                                    "n_na": sum(x is None for x in ident)}})
    else:
        controls.append({"name": "member_catch", "how": sorted(how), "n_absent": len(absent), "absent": absent[:12], "pass": False})
    for g, mem in GROUPS.items():
        for ro in ("C", "R"):
            gk = f"{g}|{ro}"
            node = (gc.get("groups") or {}).get(gk) or {}
            got = sorted(str(x).removesuffix(f" {ro}") for x in (node.get("members") or []))
            row("group_catch", f"groups.{gk}.members=registered", got == sorted(mem), True, None, True, "bool")
            row("group_catch", f"groups.{gk}.n", len(mem), node.get("n"), None, True, "int")
            for a in ARMS:
                nd = ((node.get("catch") or {}).get(a)) or {}
                ms = [mine(f"{n} {ro}|{MODE}|{LVL}|catch_{a}") for n in mem]
                if all(v is not None for v, _ in ms):
                    row("group_catch", f"groups.{gk}.catch.{a}.point", float(np.mean([v for v, _ in ms])), nd.get("point"),
                        float(np.mean([b for _, b in ms])), True)
                else:
                    row("group_catch", f"groups.{gk}.catch.{a}.point", None, None, None, True, "unresolved", "mine absent")
                if catch_ok:
                    s = summ(gmean(mem, lambda n: cat[(n, ro, a)]), IDX, METHOD)
                    for i in (0, 1):
                        row("group_catch", f"groups.{gk}.catch.{a}.ci95[{i}]", s["ci95"][i], (nd.get("ci95") or [None, None])[i], 0.0, True)
                    for part in ("draws", "finite_draws"):
                        row("group_catch", f"groups.{gk}.catch.{a}.{part}", s[part], nd.get(part), None, True, "int")
                else:
                    for part in ("ci95[0]", "ci95[1]", "draws", "finite_draws"):
                        row("group_catch", f"groups.{gk}.catch.{a}.{part}", None, None, None, True, "unresolved", "member catch control failed")
                # record consistency: the record's group point == the mean of A's member points (exact within b)
                ap = [pnode(f"{n} {ro}")["catch"][a].get("point") for n in mem]
                ok = all(num(v) for v in ap) and num(nd.get("point")) and abs(float(np.mean(ap)) - nd["point"]) <= tol(nd["point"])
                rows.append({"family": "group_catch_rc", "key": f"groups.{gk}.catch.{a}.point==mean(A member points)", "verdict": False,
                             "record_consistency": True, "kind": "bool", "match": bool(ok)})

    # ---- (5) pw_shares: my A18 pool values, one per label x level x side (4 pooled origin rows must agree)
    def pool_mine(label, lvl, side):
        os_ = [o for o in ((A.get("rows") or {}).get(label) or {}) if f"[pooled: {COMBO}]" in o]
        vals = []
        for o in os_:
            b0 = f"{label}|{o}|{lvl}|{side}"
            sw, fb = mine(f"{b0}|per_run_alarm_share_with_no_unit_runs")
            sh, _ = mine(f"{b0}|per_run_alarm_share")
            ru, _ = mine(f"{b0}|runs")
            zs, _ = mine(f"{b0}|zero_unit_sum")
            if None in (sw, sh, ru, zs):
                return None
            al = round(sh * ru)
            assert abs(sh * ru - al) <= tol(sh * ru), f"alarm count not an integer: {b0}"
            vals.append((sw, fb, al / (ru + zs) if ru + zs else None))  # (registered reading, its allowance, the record-convention reading)
        if len(os_) != 4:
            stops.append(f"pw {label} {lvl}: {len(os_)} pooled origin rows")
            return None
        return vals[0] if all(v == vals[0] for v in vals) else "unequal"

    for label in sorted(pw["shares"]):
        for lvl in ("0.01", "0.05"):
            node = pw["shares"][label].get(lvl) or {}
            ref = {sd: ((node.get(sd) or {}).get("registered_with_no_unit_runs") or {}).get("run_weighted") for sd in ("S", "B", "S_minus_B")}
            S, B = pool_mine(label, lvl, "S"), pool_mine(label, lvl, "B")
            k0 = f"shares.{label}.{lvl}"
            if not isinstance(S, tuple) or not isinstance(B, tuple):
                for sd, vd in (("S", True), ("B|l3002", False), ("B|record_convention", lvl == "0.01" and label in FLOOR), ("S_minus_B|l3002", False),
                               ("S_minus_B|record_convention", False)):
                    row("pw_shares", f"{k0}.{sd}", None, None, None, vd, "unresolved", "4 origin rows unequal or absent")
                continue
            row("pw_shares", f"{k0}.S", S[0], ref["S"], S[1], True)
            row("pw_shares", f"{k0}.B|l3002", B[0], ref["B"], B[1], False)
            plotted = lvl == "0.01" and label in FLOOR  # 20b (0d / Aswin): the 12 plotted floor points are verdict on the record's convention
            row("pw_shares", f"{k0}.B|record_convention", B[2], ref["B"], B[1], plotted,
                note="record convention; deviates from RULE L3002 (A18 finding (b)), disclosed" if plotted else None)
            row("pw_shares", f"{k0}.S_minus_B|l3002", S[0] - B[0], ref["S_minus_B"], S[1] + B[1], False)
            row("pw_shares", f"{k0}.S_minus_B|record_convention", S[0] - B[2], ref["S_minus_B"], S[1] + B[1], False)

    vr = [r for r in rows if r["verdict"]]
    fam = {}
    for r in rows:
        f_ = fam.setdefault(r["family"], {"rows": 0, "verdict": 0, "verdict_match": 0, "unresolved": 0, "diag_match": 0, "rc": 0, "rc_match": 0})
        f_["rows"] += 1
        if r.get("record_consistency"):
            f_["rc"] += 1
            f_["rc_match"] += bool(r["match"])
        elif r["verdict"]:
            f_["verdict"] += 1
            f_["verdict_match"] += r["match"] is True
            f_["unresolved"] += r["kind"] == "unresolved"
        else:
            f_["diag_match"] += r["match"] is True
            f_["unresolved"] += r["kind"] == "unresolved"
    diag = {}
    for r in rows:
        if r["family"] == "pw_shares" and not r["verdict"] and r["kind"] != "unresolved":
            k_ = r["key"].split(".")[-1]
            d_ = diag.setdefault(k_, {"n": 0, "match": 0})
            d_["n"] += 1
            d_["match"] += bool(r["match"])
    failing = [r["key"] for r in vr if r["match"] is not True]
    vals = {"rows": rows, "controls": controls, "idx_rule": IDX, "method": METHOD, "stops": stops}
    vpath = Path("/o", f"cleanroom_downstream_values_{stamp}.json")
    vb = json.dumps(vals, default=float).encode()
    with vpath.open("xb") as fh:
        fh.write(vb)
    OUT_VOL.commit()
    ctl_ok = all(c["pass"] for c in controls)
    return {"stamp": stamp, "n_stops": len(stops), "stops": stops[:40], "controls": controls, "by_family": fam, "pw_diagnostic": diag,
            "n_verdict": len(vr), "n_verdict_match": sum(r["match"] is True for r in vr), "n_unresolved": sum(r["kind"] == "unresolved" for r in vr),
            "failing_verdict_keys": failing[:200], "n_failing_verdict": len(failing),
            "rc_all_true": all(r["match"] for r in rows if r.get("record_consistency")),
            "values_file": {"volume": OUT, "path": vpath.name, "sha256": sha(vb)},
            "PASS": not stops and ctl_ok and not failing and all(r["match"] for r in rows if r.get("record_consistency"))}


@app.local_entrypoint()
def main():
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out = run.remote(stamp)
    path = Path(__file__).parent / "results" / f"cleanroom_downstream_{stamp}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out.get(k) for k in ("n_stops", "n_verdict", "n_verdict_match", "n_unresolved", "n_failing_verdict", "rc_all_true", "PASS")}))
    print("wrote", path)
