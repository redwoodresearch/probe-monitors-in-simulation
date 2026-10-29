"""Figure 4: the catch-rate gap (simulated minus real) against alarm strictness, from ONE all-tasks sweep record, on the lead union combination,
with JOINT 95% intervals only (thresholds re-estimated in every resample; no fixed-threshold band). Writing's choice (A), via 2e (10-16):
  - panel 1, the six-probe average (coheadline_signed[combo]["joint"][f][--average-group]): orange S = the mean over the six probes of
    (simulated minus the mean of the two real runs); gray Y = the mean of (real run 2 minus real run 1, both at the real threshold);
  - panels 2-7, the six probes (pooled[label][combo]["joint"][f]["gap"]): orange only, unless --probe-gray names a per-probe gray series in a
    pooled-world record (run2_minus_run1_at_tauB; absent = STOP; never on the per-set record);
  - levels 1e-3 .. 5e-2 only (the registered 15), no "unreadable" markers, no status word in the title, caption or image.
gate() below is the ONE predicate both this renderer and fig4_gate_count.py (counts only, before the render) apply; every failure is listed,
then the render STOPs. Asserted for every plotted point: the probes' and the average's levels are the same registered 15; for every member
origin the readable flags hold B, A and S (B2 where present) and all are True, and readable["all"] is True; no withheld or unavailable member,
no nos4 flag or withheld twin on the probe combo, the combo not in the record's without-S4 flagged or unavailable lists (required for the
per-set record), no row on the nos4 allowlist, no nos4 twin check on a member's coverage, no withheld/unavailable/nos4 key under joint; the
average's rows are the six; every plotted interval is a finite pair; the rows are the command readout; on the pooled-world record, every
probe's readable_real_line is True at every level (the real run 2 minus run 1 lines' own readability flag; required, so absent fails). The source JSON holds every plotted
value by key plus average_per_level (the per-point table for 0d's diff), the not-readable counts per pool type, dropped-draw and
degenerate counts, the estimator and the average key.
--record-type names the record: sweep15_perset (the per-set sweep15 re-reduce) or pooled_world_B (the pooled-world app B). Nothing is recomputed.
Module level is stdlib only, so the gate job imports gate() without matplotlib.

  python post_fig4_sweep.py --record <file> <sha256> <path@commit> --record-type sweep15_perset --rows "sol_a_uncut C" ... \
      --labels '{...}' --stamp _fig4 [--probe-gray none]
"""

import json
import math
import re
import sys
import textwrap

REG15 = [10 ** (-3 + 0.125 * i) for i in range(14)] + [0.05]  # the registered sweep15 grid (1e-3 .. 5e-2); record keys carry 6 significant digits
POOLS = ("B", "A", "S", "B2")  # the readable flags' pool types (draw 0, per member origin); B2 only on S_old
REQ_POOLS = {"B", "A", "S"}
ORIGIN, GROUP = "pooled_all_origins", "headline_six_full|C"  # the lead union combination and the six-probe average group (shared with the gate)
SERIES = {"A": ("delta_full", "S"), "B": ("delta_full|B", "S|B")}  # estimator -> (probe gap key, average key); gray average = "Y"
TYPES = {"sweep15_perset": ("none",), "pooled_world_B": ("none", "run2_minus_run1_at_tauB")}  # the per-probe gray series each record type allows
GRAY_CAPTION = "real run 2 minus real run 1 shown for the six-probe average"  # writing's caption line (via 2e), verbatim
# writing via 2e (10-16): NOT in the post caption (the post's interval footnote covers it), but verbatim in the record's _text.md (and the NOTE)
RECORD_NOTE = "Post hoc; joint intervals (thresholds re-estimated per resample); registered sweep: fixed thresholds"
SHOW = (1e-3, 1e-2, 5e-2)  # labelled ticks and printed values
# --intervals fixed+joint (09's ruling, writing's spec via 0d / 2e, 10-17), verbatim
LEG_FIXED = "thick bar: thresholds fixed at their estimate (registered; leaves out threshold noise)"
LEG_JOINT = "thin whisker: thresholds re-estimated in each resample (added after seeing results)"
CAP_WHISKER = "At 1 in 100 the thin whisker is the interval used everywhere else in the post."


def one_in(t):
    return f"1 in {'~' if abs(1 / t - round(1 / t)) > 1e-9 / t else ''}{round(1 / t):,}"


def finite_ci(v):
    return (isinstance(v, dict) and isinstance(v.get("ci95"), (list, tuple)) and len(v["ci95"]) == 2
            and all(isinstance(z, (int, float)) and not isinstance(z, bool) and math.isfinite(z) for z in [v.get("point"), *v["ci95"]]))


def flagged_keys(x):
    """Keys anywhere under x that name a withhold, an unavailable read or a nos4 flag."""
    if isinstance(x, dict):
        return sum((("withheld" in k) or ("unavailable" in k) or ("nos4" in k.lower())) + flagged_keys(v) for k, v in x.items())
    if isinstance(x, list):
        return sum(flagged_keys(v) for v in x)
    return 0


OUT_DPI = 200  # postfig Out.save's savefig dpi
ZERO_LW = 1.0  # the zero line's width in points (ax.axhline(0, lw=1))


def label_ok(fig, ann):
    """(True, "") when the printed value's full drawn extent (text plus its padded box) lies inside its axes and clear of the zero line's stroke,
    measured at the figure's current dpi; else (False, reason). Ids only in the reason, never the value."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    rdr = fig.canvas.get_renderer()
    ax = ann.axes
    boxes = [ann.get_window_extent(rdr)] + ([ann.get_bbox_patch().get_window_extent(rdr)] if ann.get_bbox_patch() is not None else [])
    bb, ab = Bbox.union(boxes), ax.get_window_extent(rdr)
    if not (bb.x0 >= ab.x0 and bb.x1 <= ab.x1 and bb.y0 >= ab.y0 and bb.y1 <= ab.y1):
        return False, "leaves its axes"
    y0 = ax.transData.transform((ann.xy[0], 0.0))[1]
    hw = ZERO_LW * fig.dpi / 72 / 2  # half the zero line's stroke, in pixels
    if bb.y0 - hw <= y0 <= bb.y1 + hw:
        return False, "sits on the zero line"
    return True, ""


def reg_levels(node):
    lvs = sorted(node, key=float) if isinstance(node, dict) else []
    return lvs, len(lvs) == 15 and all(abs(float(x) - y) <= 1e-5 * y for x, y in zip(lvs, REG15))


def series_info(v, c):
    """Counts only: degenerate intervals and non-finite (dropped) resamples of one plotted node."""
    c["degenerate"] += bool(isinstance(v, dict) and v.get("degenerate"))
    if isinstance(v, dict) and isinstance(v.get("draws"), int) and isinstance(v.get("finite_draws"), int):
        c["dropped_draws"] += v["draws"] - v["finite_draws"]
    else:
        c["no_draw_count"] += 1


def gate(rec, rows, origin, group, est, record_type, probe_gray="none"):
    """Every predicate Figure 4 asserts, over every probe, member and level (no early exit). Returns (fails, info): keys, labels and counts only,
    never a value."""
    gap, avg_s = SERIES[est]
    keys = [gap] + ([probe_gray] if probe_gray != "none" else [])
    fails, nr = [], {p: 0 for p in POOLS} | {"other": 0}
    info = {"series": {"probe": keys, "average": [avg_s, "Y"]}, "probes": {}, "b_keys_present": 0}
    if len(rows) != 6 or any(not r.endswith(" C") for r in rows) or group.rsplit("|", 1)[-1] != "C":
        fails.append(f"six command-readout rows and a C group expected (rows {rows}, group {group})")
    meta = rec.get("meta") or {}
    allow = [e.get("label") for e in (meta.get("nos4_twin_withhold") or {}).get("allowlist") or []]
    info["nos4_allowlist"] = allow
    fails += [f"{r}: on the nos4 twin-withhold allowlist" for r in rows if r in allow]
    tw = (meta.get("without_S4") or {}).get("twin_withheld")
    if isinstance(tw, dict) and isinstance(tw.get("combos_flagged"), list) and isinstance(tw.get("combos_unavailable"), list):
        bad = set(tw["combos_flagged"]) | set(tw["combos_unavailable"])
        fails += [f"{r}|{origin}: in meta.without_S4.twin_withheld combos_flagged/unavailable" for r in rows if f"{r}|{origin}" in bad]
    elif record_type == "sweep15_perset":
        fails.append("meta.without_S4.twin_withheld.combos_flagged/combos_unavailable missing (required for the per-set record)")
    lv_all = None
    for lab in rows:
        pn = ((rec.get("pooled") or {}).get(lab) or {}).get(origin)
        pc = info["probes"][lab] = {"series": {k: {"degenerate": 0, "dropped_draws": 0, "no_draw_count": 0} for k in keys}}
        if not isinstance(pn, dict) or not pn.get("members") or not isinstance(pn.get("joint"), dict):
            fails.append(f"{lab}: no pooled[{lab}][{origin}] with members and a joint node")
            continue
        members = pn["members"]
        pc["members"] = list(members)
        if pn.get("withheld_members") or pn.get("unavailable_members"):
            fails.append(f"{lab}: withheld or unavailable members")
        if "nos4_flag" in pn or pn.get("nos4_twin_withheld_members"):
            fails.append(f"{lab}: nos4_flag or nos4_twin_withheld_members on the combo")
        cov = (rec.get("coverage") or {}).get(lab) or {}
        fails += [f"{lab} {m}: nos4_twin_check on the member's coverage" for m in members if "nos4_twin_check" in (cov.get(m) or {})]
        cells = [((cov.get(m) or {}).get("coverage") or {}).get("cohort_cells") for m in members]
        pc["tasks"] = sum(cells) if all(isinstance(x, int) and not isinstance(x, bool) for x in cells) else None
        lvs, ok = reg_levels(pn["joint"])
        if not ok:
            fails.append(f"{lab}: levels are not the registered 15 (1e-3 .. 5e-2): {lvs}")
        elif lv_all is None:
            lv_all = lvs
        elif lvs != lv_all:
            fails.append(f"{lab}: level keys differ from the first probe's")
        n_flag = flagged_keys(pn["joint"])
        if n_flag:
            fails.append(f"{lab}: {n_flag} withheld/unavailable/nos4 keys under joint")
        for lv in lvs:
            n = pn["joint"][lv] if isinstance(pn["joint"][lv], dict) else {}
            rd = n.get("readable") or {}
            if rd.get("all") is not True:
                fails.append(f"{lab} {lv}: readable['all'] is not True")
            if record_type == "pooled_world_B" and n.get("readable_real_line") is not True:  # the gray lines' own flag (fail closed when absent)
                fails.append(f"{lab} {lv}: readable_real_line is not True")
            for m in members:
                r = rd.get(m)
                if not isinstance(r, dict) or not REQ_POOLS <= set(r):
                    fails.append(f"{lab} {lv} {m}: readable flags missing for {sorted(REQ_POOLS - set(r or {}))}")
                    continue
                for p, flag in r.items():
                    if flag is not True:
                        nr[p if p in POOLS else "other"] += 1
                        fails.append(f"{lab} {lv} {m}: readable[{p}] is not True")
            g = n.get("gap") or {}
            info["b_keys_present"] += "delta_full|B" in g  # information only
            for k in keys:
                if not finite_ci(g.get(k)):
                    fails.append(f"{lab} {lv} {k}: absent, withheld or not a finite interval")
                series_info(g.get(k), pc["series"][k])
    cn = ((rec.get("coheadline_signed") or {}).get(origin) or {}).get("joint")
    clvs, ok = reg_levels(cn)
    if not ok or clvs != lv_all:
        fails.append(f"coheadline_signed[{origin}][joint]: level keys are not the probes' registered 15")
    ac = info["average"] = {"series": {k: {"degenerate": 0, "dropped_draws": 0, "no_draw_count": 0} for k in (avg_s, "Y")}}
    for lv in clvs:
        g = cn[lv].get(group) if isinstance(cn[lv], dict) else None
        if not isinstance(g, dict):
            fails.append(f"{group} {lv}: absent")
            continue
        if flagged_keys(g):
            fails.append(f"{group} {lv}: unavailable, nos4-flagged or withheld")
        if sorted(g.get("rows") or []) != sorted(rows):
            fails.append(f"{group} {lv}: its rows are not the six panel rows")
        for k in (avg_s, "Y"):
            if not finite_ci(g.get(k)):
                fails.append(f"{group} {lv} {k}: absent, withheld or not a finite interval")
            series_info(g.get(k), ac["series"][k])
    info["levels"] = lv_all
    info["not_readable"] = {"per_pool_type": nr, "total": sum(nr.values())}
    return fails, info


R37_LEVELS = ("0.0001", "0.0002", "0.0005")  # RULE 37's strict levels (the separate R37 record)


def gate_mode(rec, rows, origin, group, est, record_type, probe_gray, mode, levels):
    """The same predicates as gate() for one more mode ("fixed": the registered sweep's draw-0 thresholds) or record (R37), at the given level
    keys: per probe the level keys present, readable["all"] (and readable_real_line on the pooled world) True, every member's B / A / S flags
    True, no withheld / unavailable / nos4 key under the mode, finite plotted intervals; the average group's rows are the six. Keys, labels and
    counts only. On the R37 record, readable_rule37 must also be True at every drawn level."""
    gap, avg_s = SERIES[est]
    keys = [gap] + ([probe_gray] if probe_gray != "none" else [])
    fails, nr = [], {p: 0 for p in POOLS} | {"other": 0}
    for lab in rows:
        pn = ((rec.get("pooled") or {}).get(lab) or {}).get(origin)
        node = (pn or {}).get(mode)
        if not isinstance(node, dict) or not pn.get("members"):
            fails.append(f"{lab}: no pooled[{lab}][{origin}][{mode}] with members")
            continue
        if sorted(node, key=float) != sorted(levels, key=float):
            fails.append(f"{lab} {mode}: level keys are not {list(levels)}")
            continue
        if flagged_keys(node):
            fails.append(f"{lab}: {flagged_keys(node)} withheld/unavailable/nos4 keys under {mode}")
        for lv in levels:
            n = node[lv] if isinstance(node[lv], dict) else {}
            rd = n.get("readable") or {}
            if rd.get("all") is not True:
                fails.append(f"{lab} {mode} {lv}: readable['all'] is not True")
            if record_type == "pooled_world_B" and n.get("readable_real_line") is not True:
                fails.append(f"{lab} {mode} {lv}: readable_real_line is not True")
            if lv in R37_LEVELS and n.get("readable_rule37") is not True:
                fails.append(f"{lab} {mode} {lv}: readable_rule37 is not True")
            for m in pn["members"]:
                r = rd.get(m)
                if not isinstance(r, dict) or not REQ_POOLS <= set(r):
                    fails.append(f"{lab} {mode} {lv} {m}: readable flags missing")
                    continue
                for pl, flag in r.items():
                    if flag is not True:
                        nr[pl if pl in POOLS else "other"] += 1
                        fails.append(f"{lab} {mode} {lv} {m}: readable[{pl}] is not True")
            for k in keys:
                if not finite_ci((n.get("gap") or {}).get(k)):
                    fails.append(f"{lab} {mode} {lv} {k}: absent, withheld or not a finite interval")
    cn = (((rec.get("coheadline_signed") or {}).get(origin) or {}).get(mode)) or {}
    if sorted(cn, key=float) != sorted(levels, key=float):
        fails.append(f"coheadline_signed[{origin}][{mode}]: level keys are not {list(levels)}")
    for lv in levels:
        g = (cn.get(lv) or {}).get(group) if isinstance(cn.get(lv), dict) else None
        if not isinstance(g, dict):
            fails.append(f"{group} {mode} {lv}: absent")
            continue
        if flagged_keys(g):
            fails.append(f"{group} {mode} {lv}: unavailable, nos4-flagged or withheld")
        if sorted(g.get("rows") or []) != sorted(rows):
            fails.append(f"{group} {mode} {lv}: its rows are not the six panel rows")
        for k in (avg_s, "Y"):
            if not finite_ci(g.get(k)):
                fails.append(f"{group} {mode} {lv} {k}: absent, withheld or not a finite interval")
    return fails, {"mode": mode, "levels": list(levels), "not_readable": {"per_pool_type": nr, "total": sum(nr.values())}}


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from postfig import GRAY, INK, MUTED, RC, RES, SIM, Out, args, check_record_meta, comparator, estimator, label, routec_label

    a = args(__doc__.splitlines()[0], [("--record", "all-tasks sweep record")],
             {"--name": {"default": "fig4_sweep"}, "--rows": {"nargs": "+", "required": True},
              "--record-type": {"required": True, "choices": list(TYPES)}, "--origin": {"default": ORIGIN},
              "--average-group": {"default": GROUP}, "--probe-gray": {"default": "none"},
              "--labels": {"default": "{}"}, "--title": {"default": "Catch-rate gap between simulated and real runs, by alarm strictness"},
              "--text-note": {"default": "", "help": "a status line written verbatim into _text.md after the record note, never in the image or the "
                                                    "caption (0d via 2e, 10-17: no status words on the page)"},
              "--label-levels": {"nargs": "+", "default": None, "help": "print values only at these registered levels (0d's layout fix, 10-17: "
                                                                        "0.01); every printed label is then kept inside its axes and off the zero "
                                                                        "line, else STOP. Absent = the earlier three labels, unchanged"},
              "--intervals": {"default": "joint", "choices": ["joint", "fixed+joint"],
                              "help": "fixed+joint: each point gets the registered FIXED interval as a thick short bar over the JOINT interval as a "
                                      "thin longer whisker (09's ruling, 10-17); joint = the earlier whiskers only, unchanged"},
              "--r37-record": {"nargs": 3, "metavar": ("FILE", "SHA256", "SOURCE"), "default": None,
                               "help": "OPTIONAL, off by default: RULE 37 points (1e-4 / 2e-4 / 5e-4) from the separate R37 record, with run-count labels"},
              "--r37-count-key": {"default": None, "help": "JSON list: the key path, under each R37 level node, of the run count printed beside the "
                                                           "point (required with --r37-record; absent in the record = STOP)"},
              "--label-sidecar": {"nargs": 3, "metavar": ("FILE", "SHA256", "SOURCE"), "default": None,
                                  "help": "route C: B's own JSON label sidecar (09, 10-18); its record_sha256 must be the record drawn; the label is "
                                          "composed from it (postfig.routec_label), written to the source JSON and _text.md, never the image or "
                                          "caption; needs --label-sha and --record-meta, no --text-note"},
              "--label-sha": {"default": None, "help": "route C: the sha256 of 09's registered label string for this record (B: READ_LOG @ "
                                                       "598169ded L2226), which the sidecar's label must hash to"},
              "--record-meta": {"nargs": 3, "metavar": ("GIT", "SCRIPT_SHA256", "INPUTS_FULL_SHA256"), "default": None,
                                "help": "route C (09 S2 item 10): the drawn record's meta.git, meta.script_sha256 and FROZEN inputs_full sha256, "
                                        "re-asserted by key here before the gate (postfig.check_record_meta); required with --label-sidecar"}})
    labels = json.loads(a.labels)
    rlabel = None
    rc_ = (a.label_sidecar, a.label_sha, a.record_meta)
    if any(x is not None for x in rc_):  # route C (09, 10-17 / 10-18): all three, and no typed status line
        if any(x is None for x in rc_) or a.text_note:
            sys.exit("STOP: --label-sidecar, --label-sha and --record-meta go together, without --text-note")
        rlabel = routec_label(a.label_sidecar, a.record[1], f"Figure 4 (post_fig4_sweep) on route C pooled B @ {a.record[1]}", a.label_sha)
    if comparator() != "run_mean":  # orange = simulated minus the mean of the two real runs
        sys.exit("STOP: Figure 4 reads the run-mean comparator (delta_full); the plan's comparator is not run_mean")
    show_lab = SHOW
    if a.label_levels is not None:
        show_lab = tuple(float(x) for x in a.label_levels)
        if not show_lab or not all(any(abs(t - y) <= 1e-5 * y for y in REG15) for t in show_lab):
            sys.exit(f"STOP: --label-levels {a.label_levels} are not registered levels")
    est = estimator()  # A = Fig 1's (507-prep 10-16); recorded in the source JSON
    if est != "A":  # a plan edited after the gate run cannot render B
        sys.exit(f"STOP: Figure 4 renders estimator A (Fig 1's); the plan gives {est!r}")
    gap, avg_s = SERIES[est]
    if a.probe_gray not in TYPES[a.record_type]:
        sys.exit(f"STOP: --probe-gray {a.probe_gray!r} is not allowed for record type {a.record_type} (allowed: {TYPES[a.record_type]}); the "
                 "per-set record has no per-probe signed run 2 minus run 1")
    if a.r37_record is not None:  # refresh2's Astra (10-17, REQUIRED): gate_mode() does not apply the full record / combo / member exclusions
        # (combo-level nos4_flag, withholding meta, record-level no-S4 exclusions) to the separate R37 record; R37 is off in every plan and
        # whether to show it is Aswin's open question, so the switch STOPs; the drawing code below stays for when it is gated in full
        sys.exit("STOP: R37 points are disabled until gated with the full record/combo/member exclusions")
    if a.r37_record is not None and not a.r37_count_key:
        sys.exit("STOP: --r37-record needs --r37-count-key (the run count printed beside each R37 point)")
    o = Out(a.name, a, {"rec": a.record, **({"r37": a.r37_record} if a.r37_record is not None else {})})
    rec = o.recs["rec"]
    if a.record_meta is not None:  # 2e / 09 (10-18): B's meta by key before the gate and any drawing
        o.src["record_meta"] = check_record_meta(rec.get("meta"), *a.record_meta)
    fixed = a.intervals == "fixed+joint"
    fails, info = gate(rec, a.rows, a.origin, a.average_group, est, a.record_type, a.probe_gray)
    if fixed:  # the registered fixed-threshold intervals pass the same predicates at the same level keys
        f_fx, info["fixed"] = gate_mode(rec, a.rows, a.origin, a.average_group, est, a.record_type, a.probe_gray, "fixed", info["levels"] or ())
        fails += f_fx
    if a.r37_record is not None:  # the R37 record's strict levels, each mode drawn
        for md in (("fixed", "joint") if fixed else ("joint",)):
            f_r, info[f"r37_{md}"] = gate_mode(o.recs["r37"], a.rows, a.origin, a.average_group, est, a.record_type, a.probe_gray, md, R37_LEVELS)
            fails += f_r
    if fails:
        print("\n".join(fails), file=sys.stderr)
        sys.exit(f"STOP: {len(fails)} gate failures (all listed above; not-readable flags by pool type {info['not_readable']['per_pool_type']})")
    lv_all = info["levels"]

    # every plotted value read by key (logged in the source JSON)
    series = {lab: [(lv, key, o.val("rec", ["pooled", lab, a.origin, "joint", lv, "gap", key]), col)
                    for lv in lv_all for key, col in [(gap, SIM)] + ([(a.probe_gray, GRAY)] if a.probe_gray != "none" else [])]
              for lab in a.rows}
    series["average"] = [(lv, key, o.val("rec", ["coheadline_signed", a.origin, "joint", lv, a.average_group, key]), col)
                         for lv in lv_all for key, col in ((avg_s, SIM), ("Y", GRAY))]
    apl = {}  # the per-point table for 0d's diff: the average at each level, both series, with joint intervals
    for lv, key, v, _ in series["average"]:
        apl.setdefault(lv, {})["sim_minus_real" if key == avg_s else "run2_minus_run1"] = {"point": v["point"], "ci95": v["ci95"]}
    fx = {}  # (panel, level, series key) -> the FIXED node, read by key (logged in the source JSON)
    r37 = {}  # panel -> [(level, series key, joint node, fixed node or None, colour, run count)]
    if fixed:
        for lab in a.rows:
            for lv, key, _, _ in series[lab]:
                fx[(lab, lv, key)] = o.val("rec", ["pooled", lab, a.origin, "fixed", lv, "gap", key])
        for lv, key, _, _ in series["average"]:
            fx[("average", lv, key)] = o.val("rec", ["coheadline_signed", a.origin, "fixed", lv, a.average_group, key])
            apl[lv]["sim_minus_real_fixed" if key == avg_s else "run2_minus_run1_fixed"] = {"point": fx[("average", lv, key)]["point"],
                                                                                            "ci95": fx[("average", lv, key)]["ci95"]}
    if a.r37_record is not None:
        ck = json.loads(a.r37_count_key)
        for panel in ["average"] + list(a.rows):
            r37[panel] = []
            for lv in R37_LEVELS:
                for key, col in ([(gap, SIM)] + ([(a.probe_gray, GRAY)] if a.probe_gray != "none" else [])) if panel != "average" else \
                        ((avg_s, SIM), ("Y", GRAY)):
                    base = (["pooled", panel, a.origin] if panel != "average" else ["coheadline_signed", a.origin])
                    tail = (["gap", key] if panel != "average" else [a.average_group, key])
                    vj = o.val("r37", base + ["joint", lv] + tail)
                    vf = o.val("r37", base + ["fixed", lv] + tail) if fixed else None
                    cnt = o.val("r37", base + ["fixed" if fixed else "joint", lv] + ([] if panel != "average" else [a.average_group]) + ck)
                    if isinstance(cnt, bool) or not isinstance(cnt, int):
                        sys.exit(f"STOP: {panel} {lv}: the R37 run count at {ck} is not an integer")
                    r37[panel].append((lv, key, vj, vf, col, cnt))
    n_tasks = {lab: sum(o.val("rec", ["coverage", lab, m, "coverage", "cohort_cells"]) for m in rec["pooled"][lab][a.origin]["members"])
               for lab in a.rows}

    plt.rcParams.update({**RC, "font.size": 13, **({"axes.grid": False} if a.label_levels is not None else {})})  # no gridlines under the label check
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.4), sharey=True, squeeze=False)
    axes = axes.ravel()
    not_nested = []  # ids of points whose fixed interval is not inside the joint one (listed in the source JSON; ids only)
    for ax, panel in zip(axes, ["average"] + list(a.rows)):
        for col in (SIM, GRAY):
            pts = [(float(lv), v) for lv, _, v, c in series[panel] if c == col]
            if not pts:
                continue
            off = 1.0 if col == SIM else 1.06  # the gray series sits just right of the orange one at each level (log axis)
            xs, ys = [x * off for x, _ in pts], [100 * v["point"] for _, v in pts]
            ax.plot(xs, ys, color=col, lw=2, ls="-" if col == SIM else "--", marker="o", ms=5, mfc=col, mec="white", mew=1,
                    solid_capstyle="round", zorder=3)  # dataviz skill: 2-pt lines, filled points with a surface ring; dashes = 2nd encoding
            for x, (_, v) in zip(xs, pts):  # endpoints drawn directly: a percentile interval may exclude its point
                ax.plot([x, x], [100 * v["ci95"][0], 100 * v["ci95"][1]], color=col, lw=1.1, zorder=2)
                ax.plot([x, x], [100 * v["ci95"][0], 100 * v["ci95"][1]], ls="none", marker="_", ms=7, mew=1.1, color=col, zorder=2)
                if fixed:  # the JOINT whisker above is drawn first; the registered FIXED interval goes on top as a thick short bar
                    lv_ = next(lv for lv, _, vv, c in series[panel] if c == col and vv is v)
                    key_ = next(k for lv2, k, vv, c in series[panel] if c == col and vv is v)
                    sid = f"{panel} | {'orange' if col == SIM else 'gray'} | {lv_}"
                    ax.lines[-2].set_gid(f"{sid} | joint")
                    ax.lines[-1].set_gid(f"{sid} | joint caps")
                    f_ = fx[(panel, lv_, key_)]
                    ax.plot([x, x], [100 * f_["ci95"][0], 100 * f_["ci95"][1]], color=col, lw=4.5, solid_capstyle="butt", zorder=2.5,
                            gid=f"{sid} | fixed")
                    if not (f_["ci95"][0] >= v["ci95"][0] and f_["ci95"][1] <= v["ci95"][1]):
                        not_nested.append(sid)
            for (x0, v), x in zip(pts, xs):  # values in ink, not the series colour (orange above its point, gray below)
                if any(abs(x0 - t) <= 1e-5 * t for t in show_lab):
                    ann = ax.annotate(f"{100 * v['point']:+.1f}", (x, 100 * v["point"]), xytext=(4, 6 if col == SIM else -13),
                                      textcoords="offset points", fontsize=9, color=INK if col == SIM else MUTED, zorder=4,
                                      bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.85})  # legible over error bars
                    ann.set_gid(f"{panel} | {'orange' if col == SIM else 'gray'} | {x0:g}")  # ids for the placement check's STOP (no value)
        for lv, key, vj, vf, col, cnt in r37.get(panel, []):  # OPTIONAL R37 points (off by default): unconnected, with run counts
            x = float(lv) * (1.0 if col == SIM else 1.06)
            sid = f"{panel} | {'orange' if col == SIM else 'gray'} | {lv} | r37"
            ax.plot([x, x], [100 * vj["ci95"][0], 100 * vj["ci95"][1]], color=col, lw=1.1, zorder=2, gid=f"{sid} | joint")
            if vf is not None:
                ax.plot([x, x], [100 * vf["ci95"][0], 100 * vf["ci95"][1]], color=col, lw=4.5, solid_capstyle="butt", zorder=2.5,
                        gid=f"{sid} | fixed")
                if not (vf["ci95"][0] >= vj["ci95"][0] and vf["ci95"][1] <= vj["ci95"][1]):
                    not_nested.append(sid)
            ax.plot([x], [100 * vj["point"]], ls="none", marker="o", ms=5, mfc=col, mec="white", mew=1, zorder=3, gid=f"{sid} | point")
            ax.annotate(f"{cnt:,} runs", (x, 100 * vj["ci95"][0]), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=7,
                        color=MUTED, zorder=4, gid=f"{sid} | count")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xscale("log")
        ax.set_xlim((1e-4 if r37 else 1e-3) / 1.25, 0.05 * 1.35)
        ax.set_xticks(([1e-4] if r37 else []) + list(SHOW), [one_in(t) for t in ([1e-4] if r37 else []) + list(SHOW)])
        ax.minorticks_off()
        title = "Six-probe average" if panel == "average" else label(panel.rsplit(" ", 1)[0], labels)
        ax.set_title("\n".join(textwrap.wrap(title, 30)), loc="left", fontsize=13, fontweight="bold" if panel == "average" else "normal")
        ax.tick_params(labelleft=True)
    for i in (3, 4, 5, 6):  # the bottom of each column (the 8th cell holds the legend)
        axes[i].set_xlabel("Honest actions that raise an alarm\n(stricter to the left)")
    for i in (0, 4):
        axes[i].set_ylabel("Catch-rate gap, percentage points\n(above 0: simulation catches more)", fontsize=12)
    axes[7].axis("off")
    rr_txt = "Real run 2 minus real run 1 (same real threshold" + ("; first panel only)" if a.probe_gray == "none" else ")")
    hand = [Line2D([], [], color=SIM, lw=2, marker="o", ms=5, mec="white", label="Simulated minus real (real = mean of the two real runs)"),
            Line2D([], [], color=GRAY, lw=2, ls="--", marker="o", ms=5, mec="white", label=rr_txt),
            Line2D([], [], color=MUTED, marker="|", ms=12, mew=1.5, ls="none",
                   label="Error bars: 95% intervals over resampled tasks,\nthresholds re-estimated in each resample")]
    if fixed:  # the two interval kinds, writing's legend verbatim (wrapped for the 8th cell)
        hand[2:] = [Line2D([], [], color=MUTED, lw=4.5, solid_capstyle="butt", label="\n".join(textwrap.wrap(LEG_FIXED, 44))),
                    Line2D([], [], color=MUTED, lw=1.1, label="\n".join(textwrap.wrap(LEG_JOINT, 44)))]
    axes[7].legend(handles=hand, frameon=False, loc="center left", fontsize=11)
    fig.tight_layout(w_pad=2.0)
    if a.label_levels is not None:  # 0d's layout fix: every printed value inside its axes and off the zero line (flip to the other side once)
        fig.set_dpi(OUT_DPI)  # measured at the saved resolution (postfig Out.save writes 200 dpi)
        for ann in [t for ax in axes for t in ax.texts if not (t.get_gid() or "").endswith("| count")]:
            for attempt in range(2):
                if label_ok(fig, ann)[0]:
                    break
                if attempt == 0:  # the other side of its point
                    dx, dy = ann.xyann
                    ann.xyann = (dx, -13 if dy > 0 else 6)
            else:
                sys.exit(f"STOP: a printed value ({ann.get_gid()}) cannot be placed inside its axes and clear of the zero line on either side of "
                         "its point")
        o.src["label_levels"] = list(a.label_levels)

    n = sorted(set(n_tasks.values()))
    all_series = list(info["probes"].values()) + [info["average"]]
    dropped = sum(c["dropped_draws"] for s in all_series for c in s["series"].values())
    no_count = sum(c["no_draw_count"] for s in all_series for c in s["series"].values())
    degen = sum(c["degenerate"] for s in all_series for c in s["series"].values())
    o.src |= {"comparator": "run_mean", "estimator": est, "record_type": a.record_type, "origin": a.origin, "six_probes": list(a.rows),
              "probe_series": gap, "probe_gray": a.probe_gray, "average_series": {"orange": avg_s, "gray": "Y"},
              "average_key": f'coheadline_signed["{a.origin}"]["joint"][f]["{a.average_group}"]', "average_per_level": apl, "levels": lv_all,
              "interval": ("fixed (registered; thick bar) and joint (post hoc; thin whisker)" if fixed
                           else "joint (thresholds re-estimated in every resample)"), "record_note": RECORD_NOTE, "gate": info, "n_tasks": n_tasks,
              "gates_asserted": ["the registered 15 levels (1e-3 .. 5e-2), the same keys for the probes and the average",
                                 "readable flags B, A, S (B2 where present) all True for every member origin, and readable['all']",
                                 ("no withheld or unavailable member; no nos4_flag or nos4_twin_withheld_members on the combo; the combo not in "
                                  "meta.without_S4.twin_withheld combos_flagged/combos_unavailable; no row on the nos4 allowlist; no "
                                  "nos4_twin_check on a member's coverage; no withheld/unavailable/nos4 key under joint"),
                                 "the average's rows are the six", "every plotted interval a finite pair", "the command readout"]
              + (["readable_real_line True for every probe at every level (pooled-world record)"] if a.record_type == "pooled_world_B" else [])}
    o.text(a.title,
           f"{n[-1]:,} tasks" + (f" ({n[0]:,} for some probes)" if len(n) > 1 else "") + ", probes scoring commands. "
           "Orange: simulated minus real catch rate, the real rate being the mean of the two real runs, each pipeline on its own alarm line; "
           "the first panel averages the six probes. "
           f"Gray: {GRAY_CAPTION}" + (" and for each probe" if a.probe_gray != "none" else "") + ". "
           + ("Thick bars: 95% intervals over resampled tasks with the alarm lines fixed at their full-data estimate (the registered sweep "
              "method); thin whiskers: 95% intervals with the alarm lines re-estimated in each resample (added after seeing results). "
              + CAP_WHISKER + " " if fixed else "Error bars: 95% intervals over resampled tasks, alarm lines re-estimated in each resample. ")
           + ("Unconnected points at 1 in 10,000, 1 in 5,000 and 1 in 2,000: the separate RULE 37 record, labelled with run counts. " if r37 else "")
           + "Levels from 1 in 1,000 to 1 in 20 honest actions raising an alarm; every plotted point is readable (at least 10 honest actions above "
           f"each alarm line in every contributing pool; not-readable pool flags at these levels: {info['not_readable']['total']}). "
           + (f"Non-finite resamples dropped: {dropped:,} in total. " if dropped else "")
           + (f"{no_count} plotted points carry no resample count. " if no_count else "")
           + (f"{degen} plotted intervals are flagged degenerate in the record. " if degen else "")
           + ("Values printed at 1 in 1,000, 1 in 100 and 1 in 20." if a.label_levels is None
              else f"Values printed at {', '.join(one_in(t) for t in show_lab)}."),
           [lab.rsplit(" ", 1)[0] for lab in a.rows], n=n[-1])
    if a.text_note:  # the pooled status, verbatim from the plan (no status string in code); absent = the outputs are unchanged
        o.src["text_note"] = a.text_note
    if rlabel:  # route C: the sidecar-composed label and its pin (09's rule), beside the record citation
        o.src |= {"label": rlabel, "label_sidecar": {"file": a.label_sidecar[0], "sha256": a.label_sidecar[1], "source": a.label_sidecar[2],
                                                    "label_sha256": a.label_sha}}
    if fixed or r37:  # ids only: points whose registered fixed interval is not inside the joint one
        o.src |= {"intervals": a.intervals, "fixed_not_within_joint": not_nested, "r37": bool(r37)}
    for s in [o.title, o.caption] + [t.get_text() for t in fig.findobj(Text)]:
        if re.search(r"(?i)prelim", s or ""):
            sys.exit("STOP: 'preliminary' in the title, caption or image (writing: no status word on Figure 4)")
    o.save(fig)
    with open(RES / f"{a.name}{a.stamp}_text.md", "a") as f:  # the record-only note, after the caption and source line (never in the caption)
        f.write(f"\nRecord note (not for the post caption): {RECORD_NOTE}\n")
        if a.text_note:
            f.write(f"\nStatus (not in the image or the caption): {a.text_note}\n")
        if rlabel:
            f.write(f"\nLabel (not in the image or the caption): {rlabel}\n")


if __name__ == "__main__":
    main()
