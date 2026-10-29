"""Post figures from the PANEL record (ports of figs_simple/make_excess_C.py and make_excess_combined.py), 0d's layout via 2e (09 @ 2d61af835):

--variant signed (main text, Figs 3 and 11): per probe, the signed difference with its joint 95% interval beside the rerun's signed run 2 − run 1
  (gray), and the six-probe S and Y as the summary bar (the record's coheadline block). The signed difference follows the plan's comparator:
  run_mean = 09's lead, the record's per-probe run-mean Δ_full (delta_full_B when the record carries (B), else delta_full_A); run1 = sim − run 1.
--variant excess (appendix): the noise-matched excess per probe on estimator (A), with the group average, labelled with its yardstick
  definition and the planted-check caveat.
Every value read by key from the PANEL record; the banner's task count from the count record's coverage over the plotted task sets.

--source pooled_record (0d via 2e, 10-17; --variant excess, one readout): the per-probe gap comparison from the pooled-world Stage A record A
and no PANEL record (never mixed): per probe pooled["<p> <ro>"]["pooled_all_origins"]["joint"][F]["panel"]["pooled"]["G_minus_Y"]; the
six-probe row panel_groups["pooled_all_origins"]["pooled"]["joint"][F]["<group>|<ro>"]["primary"]["mean_G_minus_mean_Y"]; the four-probe
row four_probe.mean_G_minus_mean_Y from run_group_excess.py's record (--four-record, built on this record A, gate passed). Every key read
strictly (a missing key STOPs); fractions drawn x100; value labels with two decimals and a true minus sign, checked inside the axes and off
the zero line at the saved dpi. Writing's spec (10-17): no in-image title; the x-axis label and the axis note verbatim, the note split at zero
so its left arrow ("simulation closer to real than a rerun": G < Y) sits on the negative side, measured at the saved dpi.

  python post_excess.py --panel panel_<stamp>.json <sha> <path@commit> --count alltasks_<stamp>.json <sha> <path@commit> --name fig2_signed_C
      --rows <key> ... --readouts C --variant signed --stamp _x
  python post_excess.py --source pooled_record --record <A>.json <sha> <path@commit> --four-record group_excess_<stamp>.json <sha> <path@commit>
      --name fig5_excess_C --rows <key> ... --readouts C --variant excess --group headline_six_full --subset-members <4 keys> --stamp _x
"""

import json
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figio import POOL  # the build profile's pooled combination, checked by name against LEAD
from postfig import (BAR, BUILD, F, GRAY, INK, INPUT_SIDECAR, MUTED, RC, RES, SIM, SUM, Out, args, check_record_meta, comparator, estimator, has_ci,
                     label, routec_inputs, routec_label, sidecar_label, wmark)

RO_TXT = {"C": "Command readout", "R": "Tool-output readout"}
RERUN_KEY = "Y_tauB"  # PANEL (via 2e, 10-16): per probe run2@τ_B − run1@τ_B per draw, points; the yardstick_signed fallback is dropped
EXCESS_DEF = ("Excess = mean |G| − mean of |run2@τA − run1@τB| (run1 at the separate B2 pool on S_old; the noise-matched yardstick), on estimator "
              "(A); the planted check is blind to under-catching")  # 0d via 2e, 10-16 (the B2 clause: 0d's v3 caption fixes)
SUBSET_RULE = ("C run AUROC table", ">= 0.6", "resolved before any PANEL value")  # as post_main: the four-probe block's subset_by, by key
LEAD = "pooled_all_origins"  # 09 (via 2e): the lead combination, read by name
# --source pooled_record reads the excess the pooled-world Stage A runner stored; its definition, run_peraction311.py @ 704122edf (the runner
# commit of record A; cited so the review can check it):
#   l.3176       G = 0.5 * np.abs(sim - r1) + 0.5 * np.abs(sim - r2)    per draw, the mean of |sim - run 1| and |sim - run 2|; never |delta_full|
#   l.3177       Y = np.abs((r2 if meta.get("pooled_world") else y2) - y1)    pooled world: y1 = run 1 at tau_B (Y1_DEF_PW, l.3027): |run 2 - run 1|
#   l.3190       "panel": {"pooled": {"G": summ(G), "Y": summ(Y), "G_minus_Y": summ(G - Y), ...}}    the per-probe row: G_minus_Y (ONE key)
#   l.3273-3275  Gm, Ym = np.mean over the group's rows; primary["mean_G_minus_mean_Y"] = summ(Gm - Ym)    the six-probe row
# The four-probe row: run_group_excess.py @ 5bae3788b rebuilds that group from the record's per-draw arrays with the same lines (its gate and
# its six-probe control against the record must pass): four_probe["mean_G_minus_mean_Y"].
PROBE_EXCESS = ["panel", "pooled", "G_minus_Y"]
EXCESS_DEF_PW = ("Simulation gap G = the mean of |simulated − real run 1| and |simulated − real run 2|; rerun gap Y = |real run 2 − real run 1|, "
                 "both runs at τ_B; plotted: G − Y (computed in every resampled draw)")
SUBSET_TXT = ("Four-probe average: membership fixed before any of these results were computed, from earlier AUROC values; the 0.6 cut-off itself "
              "was chosen after those AUROCs were seen (post hoc).")  # post_main's, verbatim (0d via 2e, v3.1)
ZERO_LW = 1  # the zero line's stroke (ax.axvline below)
FOUR_ROW = "Four-probe average"  # the row label and the caption's name, as post_main's (writing via 2e, 10-17: no "(AUROC ≥ 0.6)" in either)
# 09 (via 2e, 10-18): the four-probe (E4) row may render; whenever it does, its caption carries RULE l.4409's registered label VERBATIM
# (RULE.md @ 91eefa7ca on aswin/convergence-audit-09; SUBSET_TXT above names the 0.6 cut-off). The prose around it stays Aswin's.
# Aswin (10-18 23:08 PT via 0d; 09's condition binds the figure's record note, 23:10): KEEP the row; the label goes once into the
# _text.md caption and _source.json, never into the image; the generic " Post hoc." it duplicated is dropped; route C renders only.
E4_LABEL = "POST HOC: threshold chosen after the AUROCs were seen"
CITE_RX = re.compile(r"[^\x00-\x1f\x7f]*[^\s@]+@[0-9a-f]{40}[^\x00-\x1f\x7f]*[0-9a-f]{64}[^\x00-\x1f\x7f]*")  # the AUROC citation's form


def with_e4(cap):
    """The caption with E4's registered label appended verbatim (route C pooled_record figures only; see e4_caption)."""
    return f"{cap} {E4_LABEL}."


def e4_caption(cap, routec):
    """The pooled_record caption's end: on route C, E4's registered label once, in place of the generic " Post hoc." (Aswin, 10-18: the two
    said the same thing twice); any other pooled_record render keeps " Post hoc." and carries no E4 label. The caption goes to _text.md
    only (postfig's Out.save refuses caption text in the PNG)."""
    return with_e4(cap) if routec else f"{cap} Post hoc."
XLABEL_PW = "Simulation gap minus rerun gap (percentage points)"  # writing via 0d / 2e (10-17), verbatim
# writing's axis note, verbatim "← simulation closer to real than a rerun | farther →", split at zero: the left part ends at the zero line (the
# negative side: G < Y, the simulation closer to the real runs than run 2 is to run 1) and the right part starts there with its "|"
AXIS_NOTE = ("← simulation closer to real than a rerun ", "| farther →")


def sgn2(x):
    """A value label: two decimals, signed, with a true minus sign (U+2212), never ASCII '-' (0d via 2e, 10-17)."""
    return f"{x:+.2f}".replace("-", "\u2212")


def labels_placed(fig, texts, dpi=200):
    """Ids (never values) of the value labels whose drawn text leaves its axes or crosses the zero line's stroke, at the saved dpi (Out.save)."""
    old = fig.dpi
    fig.set_dpi(dpi)
    fig.canvas.draw()
    rdr, bad = fig.canvas.get_renderer(), []
    for id_, t in texts:
        bb, ab = t.get_window_extent(rdr), t.axes.get_window_extent(rdr)
        x0, hw = t.axes.transData.transform((0.0, 0.0))[0], ZERO_LW * dpi / 72 / 2
        if not (bb.x0 >= ab.x0 and bb.x1 <= ab.x1 and bb.y0 >= ab.y0 and bb.y1 <= ab.y1):
            bad.append(f"{id_}: leaves its axes")
        elif bb.x0 - hw <= x0 <= bb.x1 + hw:
            bad.append(f"{id_}: sits on the zero line")
    fig.set_dpi(old)
    return bad


def bar(ax, y, v, h, col, txt=True, bold=False, fmt=None, off=0.6):
    p_ = v["point"]
    ax.barh(y, p_, height=h, color=col, zorder=2)
    if has_ci(v):
        ax.plot(v["ci95"], [y, y], color=INK, lw=1.2, zorder=3)
    if txt:
        hi = v["ci95"][1] if has_ci(v) else p_
        return ax.text(max(p_, hi) + off, y, f"{fmt(p_) if fmt else f'{p_:+.1f}'}{wmark(v)}", fontsize=10, va="center", ha="left", color=INK,
                       fontweight="bold" if bold else "normal")
    elif not has_ci(v):
        wmark(v)


def note_placed(fig, ax, notes, dpi=200):
    """Problems (words only) with the axis note at the saved dpi: the left part must end at or left of the zero line and the right part start at
    or right of it, both inside the figure and clear of the x tick labels and the x-axis label."""
    old = fig.dpi
    fig.set_dpi(dpi)
    fig.canvas.draw()
    rdr, bad = fig.canvas.get_renderer(), []
    x0, fb = ax.transData.transform((0.0, 0.0))[0], fig.bbox
    (lb, rb) = (n.get_window_extent(rdr) for n in notes)
    if ax.xaxis_inverted() or lb.x1 > x0 + 0.5 or rb.x0 < x0 - 0.5:  # not inverted: pixel-left of zero is the negative side
        bad.append("the note's left part is not on the negative side, or its right part not on the positive side")
    others = [t.get_window_extent(rdr) for t in ax.get_xticklabels() if t.get_visible() and t.get_text()] + [ax.xaxis.label.get_window_extent(rdr)]
    for nm, b in (("left", lb), ("right", rb)):
        if not (b.x0 >= fb.x0 and b.x1 <= fb.x1 and b.y0 >= fb.y0 and b.y1 <= fb.y1):
            bad.append(f"the note's {nm} part leaves the figure")
        if any(b.overlaps(o_) for o_ in others):
            bad.append(f"the note's {nm} part overlaps a tick label or the axis label")
    fig.set_dpi(old)
    return bad


def pooled_excess(a, labels):
    """--source pooled_record: the per-probe gap comparison (simulation gap minus rerun gap) from the pooled-world record A, read by key."""
    if a.variant != "excess" or len(a.readouts) != 1 or a.scale_readouts or a.subset_block:
        sys.exit("STOP: --source pooled_record draws --variant excess on ONE readout, with no --scale-readouts or --subset-block (the four-probe "
                 "row comes from --four-record)")
    if a.panel or a.count or not (a.record and a.four_record):
        sys.exit("STOP: --source pooled_record takes --record and --four-record, and no --panel or --count (never mixed)")
    if BUILD and POOL != LEAD:
        sys.exit(f"STOP: the lead must be pooled_all_origins (Stage A pool {POOL!r})")
    ro = a.readouts[0]
    o = Out(a.name, a, {"record": a.record, "four": a.four_record})
    if a.record_meta is not None:  # route C: --record's meta by key, before anything is read for drawing
        o.src["record_meta"] = check_record_meta(o.recs["record"].get("meta"), *a.record_meta)
    if a.routec_labels:  # route C: the sidecar-composed labels and their pins (09's rule; every drawn record)
        o.src |= {"label": a.routec_labels[0], "label_sidecar": {"file": a.label_sidecar[0], "sha256": a.label_sidecar[1], "source": a.label_sidecar[2],
                                                                "label_sha256": a.label_sha},
                  "input_labels": a.routec_labels[1], "input_sidecars": [{"role": r_, "file": f_, "sha256": s_, "source": c_, "step": l_}
                                                                         for r_, f_, s_, c_, l_ in a.input_sidecar or []]}
        if not CITE_RX.fullmatch(a.auroc_citation or ""):  # 09 (10-18): the E4 row's AUROC source, cited; the plan holds a placeholder until 09 confirms
            sys.exit("STOP: route C: the four-probe row needs --auroc-citation (its AUROC membership source by path@commit and sha256); the plan "
                     "holds a placeholder until 09 confirms the citation text")
        o.src["four_probe_auroc_citation"] = a.auroc_citation
        o.src["four_probe_label"] = E4_LABEL  # route C only, as the caption's label (Aswin, 10-18)
    want_rows = sorted(f"{p} {ro}" for p in a.rows)
    four_rows = sorted(f"{p} {ro}" for p in a.subset_members)
    if len(a.subset_members) != 4 or not set(a.subset_members) <= set(a.rows):
        sys.exit("STOP: --subset-members must be four of the plotted --rows")
    node = lambda lab: ["pooled", lab, LEAD, "joint", F]
    grp = ["panel_groups", LEAD, "pooled", "joint", F, f"{a.group}|{ro}"]
    bad = [f"{p} {ro}: readable['all'] not True" for p in a.rows if o.val("record", node(f"{p} {ro}") + ["readable", "all"]) is not True]
    if bad:
        sys.exit(f"STOP: not readable at {F}: {bad}")
    for p in a.rows:  # the per-probe node is G - Y by key: its draw-0 point equals G's minus Y's exactly (a key mix-up STOPs)
        g_, y_, d_ = (o.val("record", node(f"{p} {ro}") + ["panel", "pooled", k, "point"]) for k in ("G", "Y", "G_minus_Y"))
        if not all(isinstance(x, float) for x in (g_, y_, d_)) or d_ != g_ - y_:
            sys.exit(f"STOP: {p} {ro}: panel.pooled.G_minus_Y is not G - Y at draw 0")
    if o.val("record", grp + ["unreadable_rows"]) != [] or sorted(o.val("record", grp + ["primary", "rows"])) != want_rows:
        sys.exit(f"STOP: panel_groups {a.group}|{ro}: unreadable rows, or its primary rows are not the plotted rows")
    rm_ = o.val("record", ["meta", "auroc06_subset", "members"])
    sl_ = o.val("record", ["meta", "auroc06_subset", "label"])
    if sorted(rm_) != sorted(a.subset_members) or not (isinstance(sl_, str) and "POST HOC" in sl_ and ">= 0.6" in sl_):
        sys.exit("STOP: meta.auroc06_subset's members are not --subset-members, or its label is not 'POST HOC ... AUROC >= 0.6' (as post_main)")
    fm = {k: o.val("four", ["meta", k]) for k in ("combo", "mode", "level", "readout", "stream", "tag")}
    checks = {"gate.pass": o.val("four", ["gate", "pass"]) is True,
              "built on this record A": o.val("four", ["meta", "record", "sha256"]) == a.record[1],
              "combo / mode / level / readout / stream / tag": fm == {"combo": LEAD, "mode": "joint", "level": F, "readout": ro, "stream": "pooled",
                                                                      "tag": "primary"},
              "members": sorted(o.val("four", ["meta", "four_membership", "members"])) == sorted(a.subset_members),
              "rows": sorted(o.val("four", ["four_probe", "rows"])) == four_rows,
              "six-probe control MATCH": all(o.val("four", ["positive_control_six", "matches_record", k, "verdict"]) == "MATCH"
                                             for k in ("mean_G", "mean_Y", "mean_G_minus_mean_Y"))}
    if not all(checks.values()):
        sys.exit(f"STOP: the four-probe record fails: {[k for k, v in checks.items() if not v]}")

    def pts(path, rec="record"):  # a node in POINTS: the record's fraction x100, read by key (logged as stored)
        v = o.val(rec, path)
        if not (isinstance(v, dict) and "point" in v and isinstance(v["point"], float)):
            sys.exit(f"STOP: no numeric point at {path}")
        return {**v, "point": 100 * v["point"], **({"ci95": [100 * x for x in v["ci95"]]} if has_ci(v) else {})}
    rows = [(f"{a.group}|{ro}", a.group_label, pts(grp + ["primary", "mean_G_minus_mean_Y"]), True),
            ("four_probe", FOUR_ROW, pts(["four_probe", "mean_G_minus_mean_Y"], "four"), True)]
    rows += [(f"{p} {ro}", label(p, labels), pts(node(f"{p} {ro}") + PROBE_EXCESS), False) for p in a.rows]
    o.src["keys"] = {"probe": f'pooled["<p> {ro}"]["{LEAD}"]["joint"]["{F}"]{PROBE_EXCESS} x100',
                     "six": f"{grp} ['primary']['mean_G_minus_mean_Y'] x100", "four": "four record ['four_probe']['mean_G_minus_mean_Y'] x100"}
    ex = [x for _, _, v, _ in rows for x in [v["point"]] + (list(v["ci95"]) if has_ci(v) else [])]
    lo_, hi_ = min(ex + [0.0]), max(ex + [0.0])
    sp = (hi_ - lo_) or 1.0
    plt.rcParams.update({**RC, "font.size": 12, "axes.spines.left": False})
    fig, ax = plt.subplots(figsize=(9.0, 1.2 + 0.62 * len(rows)))  # the per-set excess layout (one readout, two average rows)
    ys, texts = [], []
    for i, (id_, _, v, summary) in enumerate(rows):
        y = i + (0 if summary else 0.6)
        texts.append((id_, bar(ax, y, v, 0.62, SUM if summary else BAR, bold=summary, fmt=sgn2, off=0.02 * sp)))
        ys.append(y)
    ax.set_xlim(lo_ - 0.05 * sp, hi_ + 0.25 * sp)  # room on the right for the value labels
    ax.set_xlabel(XLABEL_PW, fontsize=10.5, labelpad=20)  # below the axis note
    notes = [ax.annotate(t, xy=(0.0, 0.0), xycoords=("data", "axes fraction"), xytext=(0, -27), textcoords="offset points", ha=h, va="top",
                         fontsize=9.5, color=MUTED) for t, h in zip(AXIS_NOTE, ("right", "left"))]
    ax.axvline(0, color=MUTED, lw=ZERO_LW, zorder=1)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(ys, [r[1] for r in rows])
    ax.invert_yaxis()
    for t_, r_ in zip(ax.get_yticklabels(), rows):
        if r_[3]:
            t_.set_fontweight("bold")
    fig.tight_layout()
    bad = labels_placed(fig, texts)
    if bad:
        sys.exit(f"STOP: {a.name}: value labels misplaced at 200 dpi: {bad}")
    bad = note_placed(fig, ax, notes)
    if bad:
        sys.exit(f"STOP: {a.name}: axis note misplaced at 200 dpi: {bad}")
    o.src |= {"source": "pooled_record", "value_labels": "two decimals, signed, U+2212 minus; inside the axes and off the zero line at 200 dpi",
              "x_label": XLABEL_PW, "axis_note": list(AXIS_NOTE),
              "axis_note_check": "left part ends at or left of zero, right part starts at or right of it; clear of tick and axis labels at 200 dpi",
              "four_record_checks": sorted(checks)}
    sat = [p for p in a.rows if o.val("record", node(f"{p} {ro}") + ["panel", "pooled", "saturated"]) is True]
    o.src["saturated_rows"] = sat
    n_by = {p: sum(o.val("record", ["coverage", f"{p} {ro}", m, "coverage", "cohort_cells"]) for m in o.val("record", node(f"{p} {ro}")[:3] + ["members"]))
            for p in a.rows}
    ns = sorted(set(n_by.values()))
    if len(ns) != 1:
        sys.exit(f"STOP: {a.name}: the plotted rows' task counts differ ({ns})")
    o.src["n_by_probe"] = n_by
    names = ", ".join(label(p, labels) for p in a.subset_members)
    cap = (f"Probes scoring {'commands' if ro == 'C' else 'tool outputs'}, scored on all {ns[0]:,} tasks. {EXCESS_DEF_PW}. Each pipeline on its "
           "own alarm line at 1 alarm per 100 honest actions; lines: 95% intervals over resampled main tasks (thresholds re-estimated in every "
           "draw; the union stream, pooled_all_origins: one draw per main across origins; pointwise, not simultaneous across probes). "
           f"{a.group_label}: the mean of G − Y over all six probes; {FOUR_ROW}: the same mean over {names}. {SUBSET_TXT}"
           + (f" Saturated in the record (a catch of at least 99% at draw 0): {', '.join(label(p, labels) for p in sat)}." if sat else ""))
    o.text(a.title or "Simulation gap minus rerun gap, per probe", e4_caption(cap, bool(a.routec_labels)), list(a.rows), n=ns[0])
    o.save(fig)


def main():
    rec3 = {"nargs": 3, "metavar": ("FILE", "SHA256", "SOURCE"), "default": None}
    a = args(__doc__.splitlines()[0], [],
             {"--panel": {**rec3, "help": "PANEL record (--source panel)"},
              "--count": {**rec3, "help": "record whose coverage gives the task count (the all-tasks Stage A record; --source panel)"},
              "--source": {"default": "panel", "choices": ["panel", "pooled_record"],
                           "help": "pooled_record: the excess from the pooled-world Stage A record's own keys, no PANEL record (0d via 2e, 10-17)"},
              "--record": {**rec3, "help": "the pooled-world Stage A record A (--source pooled_record)"},
              "--four-record": {**rec3, "help": "run_group_excess.py's four-probe record on that record A (--source pooled_record)"},
              "--text-note": {"default": "", "help": "a status line written verbatim into the figure's _text.md after the caption and Source line"},
              "--name": {"required": True}, "--rows": {"nargs": "+", "required": True}, "--readouts": {"nargs": "+", "default": ["C"]},
              "--variant": {"required": True, "choices": ["signed", "excess"]}, "--group": {"default": "body6", "help": "excess: group key for the average"},
              "--group-label": {"default": "Average of these probes"}, "--combo": {"default": "pooled"},
              "--summary": {"default": "coheadline", "choices": ["coheadline", "none"], "help": "signed: the six-probe S and Y bar (none: schema tests only)"},
              "--summary-block": {"default": "six"},
              "--count-probe": {"default": "FIBS C", "help": "the probe label whose pooled cohort gives the task count"},
              "--count-origins": {"nargs": "+", "default": None, "help": "the combination's task sets, when the PANEL record does not list them"},
              "--expect-headline-combo": {"default": None, "help": "the PANEL record's combos.headline_combo must equal this (09 RULE item 30: pooled_all_origins)"},
              "--scale-readouts": {"nargs": "+", "default": None,
                                   "help": "one x scale from every plotted value of these readouts (Figs 3 and 12: C and R on the same scale)"},
              "--subset-block": {"default": None, "help": "excess: a second top row, the PANEL coheadline block's excess (auroc_subset; the writing session via 2e)"},
              "--subset-members": {"nargs": "*", "default": [], "help": "the --subset-block's four probe keys, asserted by key"},
              "--subset-label": {"default": "Four-probe average (AUROC ≥ 0.6)"},
              "--labels": {"default": "{}"}, "--title": {"default": None},
              "--label-sidecar": {**rec3, "help": "route C: --record's own JSON label sidecar (09); its record_sha256 must be --record's; with "
                                                  "--label-sha, --record-meta and an --input-sidecar for --four-record; no --text-note"},
              "--label-sha": {"default": None, "help": "route C: the sha256 of 09's registered label string for --record"},
              "--record-meta": {"nargs": 3, "metavar": ("GIT", "SCRIPT_SHA256", "INPUTS_FULL_SHA256"), "default": None,
                                "help": "route C (09 S2 item 10): --record's meta.git, meta.script_sha256 and FROZEN inputs_full sha256, by key"},
              "--input-sidecar": INPUT_SIDECAR,
              "--auroc-citation": {"default": "", "help": "route C: the four-probe (E4) row's AUROC membership source, cited verbatim in _source.json "
                                                          "and _text.md (path@commit and sha256; 0d / 09 supply it)"}})
    labels = json.loads(a.labels)
    a.routec_labels = None
    rc_ = (a.label_sidecar, a.label_sha, a.record_meta)
    if any(x is not None for x in rc_) or a.input_sidecar:  # route C (09 / 2e, 10-18): the pooled source, all of them, no typed status line
        if any(x is None for x in rc_) or a.text_note or a.source != "pooled_record" or not a.record:
            sys.exit("STOP: --label-sidecar, --label-sha and --record-meta go together (with --input-sidecar), on --source pooled_record, "
                     "without --text-note")
        a.routec_labels = (routec_label(a.label_sidecar, a.record[1], f"{a.name} (post_excess) on route C pooled A @ {a.record[1]}", a.label_sha),
                           routec_inputs({k: v for k, v in (("four", a.four_record),) if v}, a.input_sidecar,
                                         sidecar_label(a.label_sidecar, a.record[1]), a.label_sha, a.record[1]))
    if a.source == "pooled_record":
        pooled_excess(a, labels)
    else:
        if not (a.panel and a.count) or a.record or a.four_record:
            sys.exit("STOP: --source panel takes --panel and --count, and no --record or --four-record")
        panel_excess(a, labels)
    if a.text_note:  # after the caption and Source line Out.save wrote (exclusive create), so never inside the caption or the image
        with open(RES / f"{a.name}{a.stamp}_text.md", "a") as f:
            f.write(f"\nStatus (not in the image or the caption): {a.text_note}\n")
    if a.routec_labels:
        with open(RES / f"{a.name}{a.stamp}_text.md", "a") as f:
            f.write(f"\nLabel (not in the image or the caption): {a.routec_labels[0]}\n"
                    + "".join(f"Input label ({r_}): {l_}\n" for r_, l_ in sorted(a.routec_labels[1].items()))
                    + f"{FOUR_ROW} AUROC membership source (not in the image or the caption): {a.auroc_citation}\n")


def panel_excess(a, labels):
    o = Out(a.name, a, {"panel": a.panel, "count": a.count})
    cmp_ = comparator() if a.variant == "signed" else None
    est = estimator() if a.variant == "signed" else None
    if a.variant == "signed" and cmp_ == "run1" and a.summary == "coheadline":  # Codex ports r5: the coheadline S is run-mean; it would not summarise run-1 bars
        sys.exit("STOP: the coheadline S and Y summarise run-mean differences; with comparator run1 use --summary none")
    base = ["results", a.combo, F]
    panel_origins = a.count_origins or ((sorted(k for k in o.val("panel", ["results"]) if re.fullmatch(r"S_[A-Za-z0-9]+", k)) if a.combo.startswith("pooled")
                                         else [a.combo]))  # the task sets behind the plotted combination (Codex ports r1/r2, r28: origins only)
    o.src["count_origins"] = panel_origins
    hc = (o.val("panel", ["combos"]).get("headline_combo") if isinstance(o.recs["panel"].get("combos"), dict) else None) if a.combo == "pooled" else a.combo
    if BUILD:  # 09: under a build profile the PANEL lead must be the union combination, by name; no override (Codex r34/r35)
        if a.expect_headline_combo not in (None, "pooled_all_origins"):
            sys.exit(f"STOP: --expect-headline-combo {a.expect_headline_combo!r} conflicts with 09's lead pooled_all_origins")
        if a.combo != "pooled":  # the PANEL stores its headline combination under results["pooled"]; any other combo is not the lead
            sys.exit(f"STOP: --combo {a.combo!r}: under a build profile the figure reads the PANEL headline results['pooled']")
        rec_hc = (o.recs["panel"].get("combos") or {}).get("headline_combo") if isinstance(o.recs["panel"].get("combos"), dict) else None
        if rec_hc != "pooled_all_origins":  # the record's own field, independent of any CLI value
            sys.exit(f"STOP: the PANEL record's combos.headline_combo is {rec_hc!r}, not pooled_all_origins (09)")
        a.expect_headline_combo = "pooled_all_origins"
    if a.expect_headline_combo and hc != a.expect_headline_combo:
        sys.exit(f"STOP: the PANEL combination plotted is {hc!r}, expected {a.expect_headline_combo!r} (09 RULE item 30)")
    stream = ("the union stream (pooled_all_origins: one draw per main across origins; 09 RULE item 30)" if hc == "pooled_all_origins"
              else f"the PANEL combination {hc!r}")
    if BUILD:
        o.src["interval_stream"] = stream
    def nodes_for(ro):  # every node the figure plots for one readout, by key (for --scale-readouts)
        if a.variant == "excess":
            ps = [base + ["groups", f"{a.group}|{ro}", "primary", "mean_G_minus_mean_Y"]] + [base + ["rows", f"{p} {ro}", "G_minus_Y"] for p in a.rows]
            ps += [["coheadline", F, ro, a.subset_block, "excess"]] if a.subset_block else []
        else:
            sk = "sim_minus_run1" if cmp_ == "run1" else f"delta_full_{est}"
            ps = [["coheadline", F, ro, a.summary_block, x] for x in ("S", "Y")] if a.summary == "coheadline" else []
            ps += [base + ["rows", f"{p} {ro}", k] for p in a.rows for k in (sk, RERUN_KEY)]
        return [o.val("panel", p_) for p_ in ps]
    if a.subset_block and a.variant != "excess":
        sys.exit("STOP: --subset-block is the excess figure's second average row")
    sat_txt = []  # the averages' saturated members, by key, with the PANEL record's definition (0d via 2e)
    for ro in (a.readouts if a.variant == "excess" else []):
        avgs = [(a.group_label, base + ["groups", f"{a.group}|{ro}"])]
        if a.subset_block:
            b_ = o.val("panel", ["coheadline", F, ro, a.subset_block])
            want = sorted(f"{p} {ro}" for p in a.subset_members)
            if len(a.subset_members) != 4 or b_.get("unavailable") or sorted(b_.get("rows") or []) != want:
                sys.exit(f"STOP: coheadline {a.subset_block} rows {b_.get('rows')} are not the expected four {want}")
            sb, lb = o.val("panel", ["coheadline", F, ro, a.subset_block, "subset_by"]), o.val("panel", ["coheadline", F, ro, a.subset_block, "label"])
            if not (isinstance(sb, str) and sb.startswith(SUBSET_RULE[0]) and all(t in sb for t in SUBSET_RULE[1:])) or not isinstance(lb, str):
                sys.exit(f"STOP: coheadline {a.subset_block} subset_by is not 'commands AUROC >= 0.6, resolved before any PANEL value', or it has no label")
            o.src.setdefault("subset", {})[ro] = {"block": a.subset_block, "rows": b_["rows"], "subset_by": sb, "label": lb}
            avgs.append((a.subset_label, ["coheadline", F, ro, a.subset_block]))
        for lab_, path_ in avgs:
            sat = o.val("panel", path_ + ["saturated_rows"])
            if not isinstance(sat, list):
                sys.exit(f"STOP: {path_}: saturated_rows is not a list")
            o.src.setdefault("saturated_rows", {})[f"{lab_} ({ro})"] = sat
            if sat:
                names = ", ".join(label(r.rsplit(" ", 1)[0], labels) for r in sat)
                sat_txt.append(f"{lab_} includes the {names}, which the PANEL record marks saturated")
    sat_def = o.val("panel", ["definitions", "saturated"]) if sat_txt else None
    xr = None
    if a.scale_readouts:  # Aswin via 2e: Fig 12 (R) on Fig 3's (C) scale; both figures take the range over both readouts
        if not set(a.readouts) <= set(a.scale_readouts):
            sys.exit(f"STOP: --scale-readouts {a.scale_readouts} must include the plotted readouts {a.readouts}")
        ex = [x for ro in a.scale_readouts for v in nodes_for(ro) for x in [v["point"]] + (list(v["ci95"]) if has_ci(v) else [])]
        lo_, hi_ = min(ex + [0.0]), max(ex + [0.0])
        sp = (hi_ - lo_) or 1.0
        xr = (lo_ - 0.05 * sp, hi_ + 0.2 * sp)  # room on the right for the value labels
        o.src["x_scale"] = {"readouts": a.scale_readouts, "xlim": list(xr)}
    plt.rcParams.update({**RC, "font.size": 12, "axes.spines.left": False})
    n_rows = len(a.rows) + 1 + (1 if a.subset_block else 0)  # the second average row adds height
    fig, axes = plt.subplots(1, len(a.readouts), figsize=(6.5 * len(a.readouts) + 2.5, 1.2 + (0.75 if a.variant == "signed" else 0.62) * n_rows),
                             sharey=True, squeeze=False)
    axes = list(axes[0])
    for j, ro in enumerate(a.readouts):
        ax = axes[j]
        if a.variant == "excess":
            rows = [(a.group_label, base + ["groups", f"{a.group}|{ro}", "primary", "mean_G_minus_mean_Y"], True)]
            rows += [(a.subset_label, ["coheadline", F, ro, a.subset_block, "excess"], True)] if a.subset_block else []  # second top row
            rows += [(label(p, labels), base + ["rows", f"{p} {ro}", "G_minus_Y"], False) for p in a.rows]
            ys = []
            for i, (lab, path, summary) in enumerate(rows):
                y = i + (0 if summary else 0.6)
                bar(ax, y, o.val("panel", path), 0.62, SUM if summary else BAR, bold=summary)
                ys.append(y)
            ax.set_xlabel("Simulation gap minus rerun gap (points)\n← simulation closer to real · farther →", fontsize=10.5)  # no estimator (writing session)
        else:  # signed: summary S and Y, then each probe's signed difference beside the rerun's
            hh, ys = 0.36, []
            if a.summary == "coheadline":
                blk = o.val("panel", ["coheadline", F, ro, a.summary_block])
                if blk.get("unavailable") or "S" not in blk or "Y" not in blk:
                    sys.exit(f"STOP: coheadline[{F}][{ro}][{a.summary_block}] has no S and Y: {blk.get('unavailable')}")
                o.src.setdefault("summary_rows", {})[ro] = blk.get("rows")
                o.src.setdefault("S_estimator", {})[ro] = blk.get("S_estimator", "A")
                if blk.get("S_estimator", "A") != est:
                    sys.exit(f"STOP: coheadline S is estimator {blk.get('S_estimator', 'A')}, the plan says {est}")
                if sorted(blk.get("rows") or []) != sorted(f"{p} {ro}" for p in a.rows):  # the summary must summarise the plotted probes
                    sys.exit(f"STOP: coheadline {a.summary_block} rows {blk.get('rows')} differ from the plotted rows")
                bar(ax, -hh / 2, o.val("panel", ["coheadline", F, ro, a.summary_block, "S"]), hh * 0.9, SUM, bold=True)
                bar(ax, hh / 2, o.val("panel", ["coheadline", F, ro, a.summary_block, "Y"]), hh * 0.9, GRAY, bold=True)
                ys.append(0)
            for i, p in enumerate(a.rows):
                y = i + 1.4
                row_ = o.val("panel", base + ["rows", f"{p} {ro}"])
                skey = "sim_minus_run1" if cmp_ == "run1" else f"delta_full_{est}"
                if est == "B" and row_.get("delta_full_B_caveat"):
                    o.src.setdefault("B_caveat", row_["delta_full_B_caveat"])
                if skey not in row_:
                    sys.exit(f"STOP: {p} {ro}: the PANEL record has no {skey} (the run-mean Δ_full needs the panel's run-mean rows)")
                o.src.setdefault("signed_key", {})[f"{p} {ro}"] = skey
                bar(ax, y - hh / 2, o.val("panel", base + ["rows", f"{p} {ro}", skey]), hh * 0.9, SIM)
                bar(ax, y + hh / 2, o.val("panel", base + ["rows", f"{p} {ro}", RERUN_KEY]), hh * 0.9, GRAY, txt=False)
                ys.append(y)
            rows = ([("Six-probe S and Y\n(Y: run 2 − run 1 at τ_B)", None, True)] if a.summary == "coheadline" else []) + [(label(p, labels), None, False) for p in a.rows]
            ax.set_xlabel("Signed difference in catch rate (points)", fontsize=10.5)
        ax.axvline(0, color=MUTED, lw=1, zorder=1)
        if xr:
            ax.set_xlim(*xr)
        ax.tick_params(axis="y", length=0)
        if j == 0:
            ax.set_yticks(ys, [r[0] for r in rows])
            ax.invert_yaxis()
            for t_, r_ in zip(ax.get_yticklabels(), rows):
                if r_[2]:  # every average row bold
                    t_.set_fontweight("bold")
        if len(a.readouts) > 1:
            ax.set_title(RO_TXT.get(ro, ro), loc="left", fontsize=13)
    if a.variant == "signed":
        fig.legend(handles=[Patch(color=SIM, label="Simulated minus real run 1" if cmp_ == "run1" else
                                  ("Simulated minus real (mean of two runs)" if est == "A" else "Simulated minus real (all-runs mean, estimator (B))")),
                            Patch(color=GRAY, label="Real run 2 minus real run 1 (both at τ_B)")]
                   + ([Patch(color=SUM, label="Six-probe S (the mean signed difference)")] if a.summary == "coheadline" else []),
                   frameon=False, loc="upper right", ncol=3, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if a.variant == "signed" else 1))
    if BUILD:  # the Stage A union node's own members, by key (the pooled-world record names its rows '<set> (all-origins draws) [pooled: ...]'; 2e)
        cmem = o.val("count", ["pooled", a.count_probe, "pooled_all_origins"])["members"]
        if not cmem:
            sys.exit(f"STOP: pooled[{a.count_probe!r}]['pooled_all_origins'] in the count record has no members")
        o.src["count_members"] = cmem
    else:
        cmem = panel_origins
    n = sum(o.val("count", ["coverage", a.count_probe, m, "coverage", "cohort_cells"]) for m in cmem)
    cnt = f"task count from {a.count[0]} coverage ({', '.join(cmem)})"
    if a.variant == "signed":
        what = ("sim − real run 1" if cmp_ == "run1" else "run-mean Δ_full = sim − mean(run 1, run 2), estimator (A) per-arm" if est == "A"
                else "Δ_full on estimator (B), the all-runs mean: " + (o.src.get("B_caveat") or "see the PANEL record's B_caveat"))
        rerun_txt = (f"per probe run 2 at τ_B minus run 1 at τ_B (PANEL record key {RERUN_KEY}), on the same draws as the summary Y, "
                     "whose six per-probe points it averages")
        o.src["rerun_definition"] = rerun_txt
        cap = (f"Signed difference per probe ({what}) with its joint 95% interval, beside the rerun's signed run 2 − run 1 in gray ({rerun_txt})"
               + ("; summary bar: the six-probe S and Y (Y = run 2 − run 1, both at τ_B) from the PANEL record's coheadline block" if a.summary == "coheadline" else "")
               + f"; 1 alarm per 100 honest actions, each pipeline on its own line; " + (f"intervals: {stream} (pointwise, not simultaneous across probes); "
                                                                                       if BUILD else "") + f"{cnt}. Post hoc.")
        title = a.title or "Signed catch-rate difference, simulated vs real, and the rerun"
    else:
        sub = o.src.get("subset", {}).get(a.readouts[0]) if a.subset_block else None
        cap = (f"{EXCESS_DEF}. Each pipeline on its own alarm line at 1 alarm per 100 honest actions; lines: 95% intervals over resampled tasks"
               + (f" ({stream}, pointwise, not simultaneous across probes)" if BUILD else "") + f"; {cnt}."
               + (f" {a.subset_label}: the excess of the PANEL record's coheadline {a.subset_block} block (members {', '.join(sub['rows'])}), the "
                  f"probes whose commands AUROC on real runs is at least 0.6, membership fixed before any PANEL value (the PANEL record's label: "
                  f"{sub['label']})." if sub else "")
               + (f" {'; '.join(sat_txt)} (PANEL definition: {sat_def})." if sat_txt else "") + " Post hoc.")
        title = a.title or "Appendix: noise-matched excess (simulation gap minus rerun gap)"
    o.text(title, f"Probes scoring {' and '.join({'C': 'commands', 'R': 'tool outputs'}[r] for r in a.readouts)}. " + cap, [p for p in a.rows], n=n)
    o.save(fig)


if __name__ == "__main__":
    main()
