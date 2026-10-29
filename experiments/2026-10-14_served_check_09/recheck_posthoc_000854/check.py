"""Independent check of the Append 28f post-hoc served record. Read-only; numbers only (no transcript text)."""

import json
import math
import re
import sys
from collections import Counter

S = "/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/80d5a744-25a9-4506-a519-d6d787a845e5/scratchpad"
PH = f"{S}/wt_served_ph/experiments/2026-10-14_served_check_09/results/served_read_real_posthoc_2026-10-15_000854.json"
PR = f"{S}/wt_served/experiments/2026-10-14_served_check_09/results/served_read_real_2026-10-14_212330.json"
V6 = f"{S}/xe/in/xerr_v6_2026-10-14_161122.json"
CL = f"{S}/xe/posthoc_claims_table.txt"
EXPECT = {"t11_media_processing", "t13_auto_workflow", "t55_user_tracking", "t69_web_scraping", "t123_media_processing"}

ph_all = json.load(open(PH))
ph = ph_all["result"]
pr = json.load(open(PR))["result"]
v6 = json.load(open(V6))["result"]
out = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


def tag(ok):
    return "MATCH" if ok else "MISMATCH"


# ---------- 1. gates, fatal, task list, counts ----------
say("== 1. gates / fatal / tasks")
g = ph["gates"]
say("gates", {k: v.get("pass") for k, v in g.items()}, "| V detail", {k: g["V_reader_reproduces_v6"][k] for k in ("threshold", "moments", "catch", "ci", "n")})
say("fatal key in result:", "fatal" in ph, "| fatal key at top:", "fatal" in ph_all, "| posthoc gates string:", repr(ph["posthoc_exclude"]["gates"]))
pe = ph["posthoc_exclude"]
tr = pe["tasks_removed"]
say("tasks_removed", tr, tag(set(tr) == EXPECT and len(tr) == 5))
say("source_record sha", pe["source_record"]["sha256"], tag(pe["source_record"]["sha256"] == "c3e3bc0a871c244a808daed0d2e2ee5b4bf3d80f28b8e3b3d1620a623751c71a"))
coh = set(v6["cohort"]["tasks"])
est = set(v6["universe_and_split"]["estimation_half"])
cest = coh & est
say("v6 cohort∩estimation_half n =", len(cest), "(v6 stated", v6["universe_and_split"]["cohort_in_estimation_half"], ")")
say("removed ⊆ cohort∩est:", set(tr) <= cest, "| missing:", sorted(set(tr) - cest))
env = v6["universe_and_split"]["env_map"]
before = Counter(env[t] for t in cest)
rem = Counter(env[t] for t in tr)
ret = Counter(env[t] for t in cest - set(tr))
tc = pe["task_counts"]
say("before n", len(cest), "rec", tc["before"]["n"], tag(len(cest) == tc["before"]["n"] == 90))
say("removed n", len(tr), "rec", tc["removed"]["n"], tag(tc["removed"]["n"] == 5))
say("retained n", len(cest) - len(tr), "rec", tc["retained"]["n"], tag(tc["retained"]["n"] == 85))
for nm, c in (("before", before), ("removed", rem), ("retained", ret)):
    say(f"env {nm}", dict(sorted(c.items())), tag(dict(c) == tc[nm]["by_environment"]))
sc = v6["universe_and_split"]["strata_cohort_counts"]["estimation"]
say("strategy before v6", sc, "rec", tc["before"]["by_strategy_stratum"], tag(sc == tc["before"]["by_strategy_stratum"]))
say("strategy removed rec", tc["removed"]["by_strategy_stratum"], "retained rec", tc["retained"]["by_strategy_stratum"],
    tag(tc["retained"]["by_strategy_stratum"] == {"with_strategy": sc["with_strategy"] - tc["removed"]["by_strategy_stratum"].get("with_strategy", 0),
                                                   "without_strategy": sc["without_strategy"] - tc["removed"]["by_strategy_stratum"].get("without_strategy", 0)}))
# stratum n_tasks in the reduced Append 33 per-probe rows (independent of the count block)
fvr = pe["full_vs_reduced"]["real"]
ns = Counter()
for row, rv in fvr.items():
    red = rv["append33_per_probe"]["reduced"]
    for L, probes in red.items():
        if not isinstance(probes, dict):
            continue
        for p, strata in probes.items():
            ns[tuple((sn, m.get("n_tasks")) for sn, m in sorted(strata.items()))] += 1
say("reduced append33 stratum n_tasks patterns:", dict(ns))
nsf = Counter()
for row, rv in fvr.items():
    for L, probes in rv["append33_per_probe"]["full"].items():
        if not isinstance(probes, dict):
            continue
        for p, strata in probes.items():
            nsf[tuple((sn, m.get("n_tasks")) for sn, m in sorted(strata.items()))] += 1
say("full append33 stratum n_tasks patterns:", dict(nsf))
sat = Counter()
for row, rr in ph["scenarios"]["real"]["rows"].items():
    for cfg, vv in rr["variants"].items():
        for L, x in vv.items():
            if isinstance(x, dict) and "saturation" in x:
                sat[x["saturation"]["n_tasks"]] += 1
say("reduced saturation n_tasks:", dict(sat))

# ---------- 2. calibration unchanged ----------
say("== 2. calibration")
say("calibration_unchanged", pe["calibration_unchanged"])
ncmp = nmis = 0
mis = []
for row in pr["scenarios"]["real"]["rows"]:
    for cfg, vv in pr["scenarios"]["real"]["rows"][row]["variants"].items():
        for L, x in vv.items():
            if not isinstance(x, dict):
                continue
            y = ph["scenarios"]["real"]["rows"][row]["variants"][cfg].get(L)
            for fld in ("thresholds", "pools", "sol_moments"):
                if fld in x:
                    if fld == "thresholds":
                        for p, t in x[fld].items():
                            ncmp += 1
                            if y[fld][p] != t:
                                nmis += 1
                                mis.append((row, cfg, L, p))
                    else:
                        ncmp += 1
                        if y[fld] != x[fld]:
                            nmis += 1
                            mis.append((row, cfg, L, fld))
say(f"scenario threshold/pool/moment entries compared: {ncmp}, mismatches: {nmis}", mis[:5])
# matched-units branch thresholds
mp = pr["sensitivity_endpoint_token_matched_units"]["result"]
mh = ph["sensitivity_endpoint_token_matched_units"]["result"]
mc = mm = 0


def walk_thr(a, b, path):
    global mc, mm
    if isinstance(a, dict):
        for k, v in a.items():
            if k in ("thresholds", "pools", "sol_moments"):
                mc += 1
                if b.get(k) != v:
                    mm += 1
            elif isinstance(v, dict):
                walk_thr(v, b.get(k, {}), path + [k])


walk_thr(mp, mh, [])
say(f"matched-units branch threshold/pool/moment blocks compared: {mc}, mismatches: {mm}")
# example thresholds
ex = []
for row in pr["scenarios"]["real"]["rows"]:
    vv = pr["scenarios"]["real"]["rows"][row]["variants"]["primary_reader_pool_membership"]
    for L in ("A0", "A", "B"):
        for p, t in vv[L]["thresholds"].items():
            ex.append((row, L, p, t, ph["scenarios"]["real"]["rows"][row]["variants"]["primary_reader_pool_membership"][L]["thresholds"][p]))
for e in ex[:6]:
    say("  thr", e[0], e[1], e[2], "primary", e[3], "posthoc", e[4])

# ---------- 3. full side == primary ----------
say("== 3. full side vs primary")


def pick(b):
    return {"point": b["point"], "ci95": b["ci95"]}


def lv_primary(row, cfg, L):
    v = pr["scenarios"]["real"]["rows"][row]["variants"][cfg][L]
    if "L" not in v:
        return {"status": v.get("reading")}
    return {
        "reading": v["reading"],
        "L_pooled": pick(v["L"]["pooled"]) | {"ci90": v["L"]["pooled"]["ci90"]},
        "L_per_probe": {p: pick(b) for p, b in v["L"]["per_probe"].items()},
        "L_rerun_pooled": pick(v["L_rerun"]["pooled"]),
        "L_rerun_per_probe": {p: pick(b) for p, b in v["L_rerun"]["per_probe"].items()},
        "catch_change_vs_reader_pooled": {w: pick(b["pooled"]) for w, b in v["catch_change_vs_reader"].items()},
    }


def lv_ph(row, cfg, L):
    v = ph["scenarios"]["real"]["rows"][row]["variants"][cfg][L]
    if "L" not in v:
        return {"status": v.get("reading")}
    return lv_primary.__wrapped__(v) if False else {
        "reading": v["reading"],
        "L_pooled": pick(v["L"]["pooled"]) | {"ci90": v["L"]["pooled"]["ci90"]},
        "L_per_probe": {p: pick(b) for p, b in v["L"]["per_probe"].items()},
        "L_rerun_pooled": pick(v["L_rerun"]["pooled"]),
        "L_rerun_per_probe": {p: pick(b) for p, b in v["L_rerun"]["per_probe"].items()},
        "catch_change_vs_reader_pooled": {w: pick(b["pooled"]) for w, b in v["catch_change_vs_reader"].items()},
    }


nums = lambda o: [x for x in json.dumps(o) and _flat(o)]


def _flat(o):
    if isinstance(o, dict):
        for k in sorted(o):
            yield from _flat(o[k])
    elif isinstance(o, list):
        for x in o:
            yield from _flat(x)
    else:
        yield o


nf = nfm = nr = nrm = nvals = 0
bad = []
for row, rv in fvr.items():
    for cfg, lays in rv["configurations"].items():
        for L, fr in lays.items():
            a, b = fr["full"], lv_primary(row, cfg, L)
            nf += 1
            nvals += len(list(_flat(a)))
            if a != b:
                nfm += 1
                bad.append(("full", row, cfg, L))
            c, d = fr["reduced"], lv_ph(row, cfg, L)
            nr += 1
            if c != d:
                nrm += 1
                bad.append(("reduced", row, cfg, L))
say(f"full cells (row x cfg x layout) compared exactly to primary: {nf}, mismatches {nfm}; leaf values {nvals}")
say(f"reduced cells compared exactly to posthoc scenarios: {nr}, mismatches {nrm}", bad[:5])
cfgs_pr = set(pr["scenarios"]["real"]["rows"]["R_reply_meanpooled|alpha=0.05"]["variants"])
say("primary variant keys:", sorted(cfgs_pr))
# x_err full vs primary
xm = xc = 0
for row, rv in fvr.items():
    for L, x in rv["x_err_U"]["full"].items():
        p = pr["scenarios"]["real"]["x_err"][row][L]
        xc += 1
        if {k: p.get(k) for k in ("X_err", "U_approx_upper_95", "note")} != x:
            xm += 1
    for L, x in rv["x_err_U"]["reduced"].items():
        p = ph["scenarios"]["real"]["x_err"][row][L]
        if {k: p.get(k) for k in ("X_err", "U_approx_upper_95", "note")} != x:
            xm += 1
say(f"x_err/U full vs primary + reduced vs posthoc scenarios: {xc} full cells, mismatches {xm}")
# append33 full reader vs v6 per-probe pooled
am = ac = 0
for row, rv in fvr.items():
    ro, al = row.split("|alpha=")
    pp6 = v6["results"][f"{ro}|mechanical|alpha={al}"]["primary_eligible_estimation_half"]
    rd = rv["append33_per_probe"]["full"]["reader"]
    for p, strata in rd.items():
        for sn, m in strata.items():
            s6 = pp6["sim_minus_mean_real"][sn]["per_probe"][p]
            ac += 1
            if not (m["delta"] == s6["delta"] and m["ci95"] == s6["ci95"] and m["discordant_tasks"] == s6["discordant_tasks"]):
                am += 1
say(f"append33 full reader per-probe (delta, ci95, discordant) vs v6: {ac} cells, mismatches {am}")
# matched units intersection full side vs primary matched branch
mu = pe["full_vs_reduced"]["matched_units_intersection"]
mmc = mmm = 0
for row, lays in mu["rows"].items():
    for L, fr in lays.items():
        v = mp["rows"][row]["variants"]["primary_reader_pool_membership"][L]
        if v is None:
            continue
        b = {"status": v.get("reading")} if "L" not in v else {
            "reading": v["reading"], "L_pooled": pick(v["L"]["pooled"]) | {"ci90": v["L"]["pooled"]["ci90"]},
            "L_per_probe": {p: pick(x) for p, x in v["L"]["per_probe"].items()},
            "L_rerun_pooled": pick(v["L_rerun"]["pooled"]),
            "L_rerun_per_probe": {p: pick(x) for p, x in v["L_rerun"]["per_probe"].items()},
            "catch_change_vs_reader_pooled": {w: pick(x["pooled"]) for w, x in v["catch_change_vs_reader"].items()}}
        mmc += 1
        if b != fr["full"]:
            mmm += 1
say(f"matched-units intersection full vs primary matched branch: {mmc} cells, mismatches {mmm}")

# ---------- 4. claims ----------
say("== 4. claims")
txt = open(CL).read().splitlines()
r1 = lambda x: f"{x:+.1f}" if abs(round(x, 1)) > 0 or x >= 0 else "-0.0"


def fmt(x):
    s = f"{x:+.1f}"
    return "+0.0" if s == "-0.0" else s


def fmt_int(b):
    return f"{fmt(b['point'])}[{fmt(b['ci95'][0])},{fmt(b['ci95'][1])}]"


pat_pool = re.compile(r"^(\S+)\s+(A0|A|B)\s+pooled L full (\S+) \[(.*?)\] \| reduced (\S+) \[(.*?)\]$")
pat_pp = re.compile(r"^\s+(\w+)\s+full (\S+)\s+reduced (\S+)$")
cfg = "primary_reader_pool_membership"
nchk = nbad = 0
badl = []
probes_seen = {}
cur = None
for ln in txt:
    m = pat_pool.match(ln)
    if m:
        row, L, fs, frd, rs, rrd = m.groups()
        cur = (row, L)
        probes_seen[cur] = []
        cell = fvr[row]["configurations"][cfg][L]
        for side, s, rd in (("full", fs, frd), ("reduced", rs, rrd)):
            got = fmt_int(cell[side]["L_pooled"])
            nchk += 1
            if got != s:
                nbad += 1
                badl.append((row, L, side, s, got))
            nchk += 1
            if not cell[side]["reading"].startswith(rd.strip()):
                nbad += 1
                badl.append((row, L, side, "reading", rd, cell[side]["reading"]))
        continue
    m = pat_pp.match(ln)
    if m and cur:
        p, fs, rs = m.groups()
        probes_seen[cur].append(p)
        cell = fvr[cur[0]]["configurations"][cfg][cur[1]]
        for side, s in (("full", fs), ("reduced", rs)):
            got = fmt_int(cell[side]["L_per_probe"][p])
            nchk += 1
            if got != s:
                nbad += 1
                badl.append((cur, p, side, s, got))
say(f"claims numeric+reading items checked: {nchk}, mismatches: {nbad}", badl)
miss = [(k, sorted(fvr[k[0]]["configurations"][cfg][k[1]]["full"]["L_per_probe"]), v) for k, v in probes_seen.items()
        if sorted(v) != sorted(fvr[k[0]]["configurations"][cfg][k[1]]["full"]["L_per_probe"])]
say("claims cells:", len(probes_seen), "| probe sets complete vs record:", not miss, miss)
say("claims header lines: tasks_removed", tag(txt[0] == f"tasks_removed {sorted(tr)}"),
    "| calibration", tag(txt[2] == f"calibration_unchanged {json.dumps(pe['calibration_unchanged'])}"),
    "| task_counts prefix", tag(json.dumps(tc).startswith(txt[1].split(" ", 1)[1])))

# reading rule
say("-- reading rule re-applied (90% inside ±5 -> no material; 95% beyond ±5 -> material; else inconclusive)")


def rule(b):
    lo90, hi90 = b["ci90"]
    lo95, hi95 = b["ci95"]
    if (lo90 > -5 and hi90 < 5) if STRICT else (lo90 >= -5 and hi90 <= 5):
        return "no material"
    if lo95 > 5 or hi95 < -5:
        return "material"
    return "inconclusive"


STRICT = "--strict" in sys.argv
say("STRICT interior:", STRICT)
mrc = mrb = 0
for row, lays in pe["full_vs_reduced"]["matched_units_intersection"]["rows"].items():
    for L, fr in lays.items():
        for side in ("full", "reduced"):
            x = fr[side]
            if "L_pooled" in x:
                mrc += 1
                w = rule(x["L_pooled"])
                g_ = x["reading"]
                if not ((w == "no material" and g_.startswith("no material")) or (w == "inconclusive" and g_.startswith("inconclusive")) or (w == "material" and g_.startswith("material"))):
                    mrb += 1
say(f"reading rule on matched-units intersection: {mrc} values, disagreements {mrb}")
rc = rb = 0
rbad = []
read_changes = []
for row, rv in fvr.items():
    for c_, lays in rv["configurations"].items():
        for L, fr in lays.items():
            for side in ("full", "reduced"):
                x = fr[side]
                if "L_pooled" not in x:
                    continue
                want = rule(x["L_pooled"])
                got = x["reading"]
                ok = (want == "no material" and got.startswith("no material change")) or (want == "inconclusive" and got.startswith("inconclusive")) or (want == "material" and got.startswith("material"))
                rc += 1
                if not ok:
                    rb += 1
                    rbad.append((row, c_, L, side, want, got))
            if "reading" in fr["full"] and fr["full"].get("reading") != fr["reduced"].get("reading"):
                read_changes.append((row, c_, L, fr["full"]["reading"][:30], fr["reduced"]["reading"][:30]))
say(f"reading rule re-applied to {rc} pooled L values (all cfgs, full+reduced): disagreements {rb}", rbad)
say("distinct readings:", Counter(fr[s]["reading"] for rv in fvr.values() for lays in rv["configurations"].values() for fr in lays.values() for s in ("full", "reduced") if "reading" in fr[s]))
say("reading changes full->reduced:", read_changes)

# ---------- 5. largest changes ----------
say("== 5. largest |full-reduced|")
best_pp = (0, None)
best_pool = (0, None)
best_pp_prim = (0, None)
best_pool_prim = (0, None)
ez_changes = []
ezr_changes = []
ez = lambda ci: ci[0] > 0 or ci[1] < 0
for row, rv in fvr.items():
    for c_, lays in rv["configurations"].items():
        for L, fr in lays.items():
            f, r = fr["full"], fr["reduced"]
            if "L_pooled" not in f:
                continue
            dpool = abs(f["L_pooled"]["point"] - r["L_pooled"]["point"])
            if dpool > best_pool[0]:
                best_pool = (dpool, (row, c_, L, f["L_pooled"]["point"], r["L_pooled"]["point"]))
            if c_ == cfg and dpool > best_pool_prim[0]:
                best_pool_prim = (dpool, (row, L, f["L_pooled"]["point"], r["L_pooled"]["point"]))
            if ez(f["L_pooled"]["ci95"]) != ez(r["L_pooled"]["ci95"]):
                ez_changes.append(("POOLED", row, c_, L, f["L_pooled"]["ci95"], r["L_pooled"]["ci95"]))
            for p in f["L_per_probe"]:
                a, b = f["L_per_probe"][p], r["L_per_probe"][p]
                dd = abs(a["point"] - b["point"])
                if dd > best_pp[0]:
                    best_pp = (dd, (row, c_, L, p, a["point"], b["point"]))
                if c_ == cfg and dd > best_pp_prim[0]:
                    best_pp_prim = (dd, (row, L, p, a["point"], b["point"]))
                if ez(a["ci95"]) != ez(b["ci95"]):
                    ez_changes.append((row, c_, L, p, a["ci95"], b["ci95"]))
            for p in f["L_rerun_per_probe"]:
                a, b = f["L_rerun_per_probe"][p], r["L_rerun_per_probe"][p]
                if ez(a["ci95"]) != ez(b["ci95"]):
                    ezr_changes.append((row, c_, L, p, a["ci95"], b["ci95"]))
say("max |dL| per-probe, all cfgs:", best_pp)
say("max |dL| pooled, all cfgs:", best_pool)
say("max |dL| per-probe, primary cfg:", best_pp_prim)
say("max |dL| pooled, primary cfg:", best_pool_prim)
say("per-probe/pooled L 95% excludes-zero status changes (strict lo>0 or hi<0):", len(ez_changes))
for e in ez_changes:
    say("   ", e)
say("L_rerun per-probe excludes-zero changes:", len(ezr_changes), ezr_changes[:6])
# matched-units intersection
best_m = (0, None)
mez = []
for row, lays in mu["rows"].items():
    for L, fr in lays.items():
        f, r = fr["full"], fr["reduced"]
        if "L_pooled" not in f or "L_pooled" not in r:
            continue
        for p in f["L_per_probe"]:
            dd = abs(f["L_per_probe"][p]["point"] - r["L_per_probe"][p]["point"])
            if dd > best_m[0]:
                best_m = (dd, (row, L, p))
            if ez(f["L_per_probe"][p]["ci95"]) != ez(r["L_per_probe"][p]["ci95"]):
                mez.append((row, L, p, f["L_per_probe"][p]["ci95"], r["L_per_probe"][p]["ci95"]))
        if f["reading"] != r["reading"]:
            mez.append(("reading", row, L, f["reading"][:30], r["reading"][:30]))
say("matched-units intersection: max |dL| per-probe", best_m, "| status/reading changes", mez)
open(f"{S}/chk/ph28f/check_out.txt", "w").write("\n".join(out) + "\n")
