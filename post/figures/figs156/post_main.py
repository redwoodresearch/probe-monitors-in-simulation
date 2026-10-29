"""Post figure (v81 Figure 1, and the same layout for appendices E1 and F1): two panels, one row per probe, rows aligned. LEFT: real catch
(the plan's comparator: the mean of the two real runs) and simulated catch, with a tick at each probe's no-skill floor (the share of honest
runs that raise at least one alarm, real honest pool B, threshold fixed at its point estimate; RUN-LENGTH's union metrics record by key).
RIGHT: simulated minus real beside real run 2 minus real run 1 (both at tau_B), the PANEL record's per-probe rows, with the PANEL record's
coheadline averages (--avg-blocks) as the top rows. Every value read by key: catches from the Stage A record's pooled[<probe> <readout>]
[pooled_all_origins] (by name), differences from the PANEL record's results["pooled"] (its headline combination must be pooled_all_origins).

--right-source pooled_record (0d via 2e, 10-16 23:31): the RIGHT panel reads the pooled-world Stage A record's own keys and no PANEL record
(never mixed): per probe pooled["<p> <ro>"]["pooled_all_origins"]["joint"][F]["gap"]["delta_full"] (the lead, estimator A; lead block v2 @
3428f7b18) and ["run2_minus_run1_at_tauB"]; the averages coheadline_signed["pooled_all_origins"]["joint"][F]["headline_six_full|<ro>" |
"auroc06_subset|<ro>"]["S" | "Y"], the four-probe membership from meta.auroc06_subset by key. Record values are fractions and are drawn x100
(points, like the PANEL's). Every drawn probe must be readable at F (readable["all"] and readable_real_line True), else STOP.
Value labels on the right (each probe's difference and each average's S): two decimals, signed, with a true minus sign (U+2212), on every
source (0d via 2e, 10-17: the text quotes -0.35 where one decimal printed -0.4); each one inside the right axes and off the zero line at the
saved dpi, else STOP with ids only.
  python post_main.py --record <A>.json <sha> <path@commit> --panel <P>.json <sha> <path@commit> --metrics <M>.json <sha> <path@commit>
      --name fig1_main --rows <key> ... --readout C --avg-blocks six auroc_subset --subset-members <4 keys> --stamp _x
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from postfig import BUILD, F, GRAY, INK, MUTED, RC, REAL, SIM, SUM, RES, Out, args, INPUT_SIDECAR, check_record_meta, comparator, estimator, has_ci, label, routec_inputs, routec_label, sidecar_label, wmark
from figio import POOL  # noqa: E402  the build profile's pooled combination, checked by name against LEAD

LEAD = "pooled_all_origins"  # 09 (via 2e): the lead combination, read by name in every record
STREAM = ("Intervals and points: the union stream (pooled_all_origins: one draw per main across origins; 09 RULE item 30); intervals are pointwise, "
          "not simultaneous across probes.")
NOUN = {"C": "commands", "R": "tool outputs"}
RERUN_KEY = "Y_tauB"  # PANEL: per probe run2@tau_B - run1@tau_B (as post_excess)
FLOOR_KEY = "0.01|B|fixed_T|with_no_unit_runs"  # RUN-LENGTH via 2e: the registered share (09 @ 51a612b29); the unsuffixed key is descriptive only
FLOOR_TXT = "share of honest runs that raise at least one alarm (real honest pool B, threshold fixed at its point estimate)"  # 0d via 2e, verbatim
POOLED_BLK = {"six": "headline_six_full", "auroc_subset": "auroc06_subset"}  # --avg-blocks -> the pooled record's coheadline_signed groups
POOLED_GAP = {"delta_full_A": "delta_full", "sim_minus_run1": "sim_minus_run1", RERUN_KEY: "run2_minus_run1_at_tauB"}  # PANEL key -> record gap
# the row labels and the caption's names: writing via 2e (10-17), "Four-probe average" with no "(AUROC ≥ 0.6)" in the image or the caption;
# the record side still requires meta.auroc06_subset's "POST HOC" / ">= 0.6" label (provenance, never drawn)
AVG_TXT = {"six": "Six-probe average", "auroc_subset": "Four-probe average"}
RR_LEGEND = "Real run 2 minus run 1"  # James's Doc comment via 0d/2e, 10-16 (one legend row per panel)
SUBSET_RULE = ("C run AUROC table", ">= 0.6", "resolved before any PANEL value")  # the PANEL block's subset_by must say all three, by key
SUBSET_TXT = ("Four-probe average: membership fixed before any of these results were computed, from earlier AUROC values; the 0.6 cut-off itself "
              "was chosen after those AUROCs were seen.")  # 0d via 2e (v3.1, 10-16), verbatim; true by key: subset_by above, 'POST HOC' in the label; its parenthetical "post hoc" removed (Aswin 10-20 00:4x via 0d: "both removed")
SUBSET_TXT_R = "On tool outputs it is the same four probes, chosen by commands AUROC, not by tool-output AUROC."  # 0d's v3 fix for Figure 5
ZERO_LW = 1  # the right panel's zero line stroke (axr.axvline below)


def caption_end(routec):
    """The caption's last words: a route C render ends without the generic " Post hoc." (Aswin 10-20 01:1x via 0d, "Remove it", Figures
    1 and 6; the same pattern as post_excess.e4_caption); any other render keeps it."""
    return "" if routec else " Post hoc."


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


def main():
    a = args(__doc__.splitlines()[0], [("--record", "Stage A all-tasks record")],
             {"--name": {"required": True},
              "--panel": {"nargs": 3, "metavar": ("FILE", "SHA256", "SOURCE"), "default": None, "help": "the PANEL record (right panel, --right-source panel)"},
              "--right-source": {"default": "panel", "choices": ["panel", "pooled_record"],
                                 "help": "pooled_record: the right panel from the pooled-world Stage A record's own keys, no PANEL record"}, "--rows": {"nargs": "+", "required": True, "help": "probe keys (no readout)"},
              "--readout": {"required": True, "choices": ["C", "R"]},
              "--metrics": {"nargs": 3, "metavar": ("FILE", "SHA256", "SOURCE"), "default": None, "help": "RUN-LENGTH's union metrics record (floor tick)"},
              "--floor-pw": {"nargs": 3, "metavar": ("FILE", "SHA256", "SOURCE"), "default": None,
                             "help": "pooled world: RUN-LENGTH's analyze_pw_shares output, the tick as a point at draw 0, no interval (2e, 10-16)"},
              "--group-catch": {"nargs": 3, "metavar": ("FILE", "SHA256", "SOURCE"), "default": None,
                                "help": "run_group_catch.py's record: the LEFT panel's average rows (equal-weight group catch; 2e via Aswin, 10-16)"},
              "--no-floor": {"action": "store_true", "help": "no floor tick (0d: Figure 1 renders first without it; a schema test without the union block)"},
              "--no-floor-note": {"default": "", "help": "caption text when there is no tick, verbatim (0d: draft: no-skill tick pending)"},
              "--text-note": {"default": "", "help": "a status line written verbatim into the figure's _text.md after the caption and Source line, never "
                                                    "in the image or the caption (0d via 2e, 10-17: no status words on the page)"},
              "--avg-blocks": {"nargs": "*", "default": [], "help": "PANEL coheadline blocks drawn as the top rows (e.g. six auroc_subset)"},
              "--subset-members": {"nargs": "*", "default": [], "help": "the auroc_subset block's four probe keys, asserted by key (0d via 2e)"},
              "--scale-readouts": {"nargs": "+", "default": None, "help": "one right-panel scale over these readouts (Figure 1 and F1)"},
              "--labels": {"default": "{}"}, "--title": {"default": "Catch rate and catch-rate difference, simulated vs real"},
              "--label-sidecar": {"nargs": 3, "metavar": ("FILE", "SHA256", "SOURCE"), "default": None,
                                  "help": "route C: --record's own JSON label sidecar (09); its record_sha256 must be --record's; with --label-sha, "
                                          "--record-meta and an --input-sidecar for every other drawn input; no --text-note"},
              "--label-sha": {"default": None, "help": "route C: the sha256 of 09's registered label string for --record"},
              "--record-meta": {"nargs": 3, "metavar": ("GIT", "SCRIPT_SHA256", "INPUTS_FULL_SHA256"), "default": None,
                                "help": "route C (09 S2 item 10): --record's meta.git, meta.script_sha256 and FROZEN inputs_full sha256, by key"},
              "--input-sidecar": INPUT_SIDECAR})
    labels, ro = json.loads(a.labels), a.readout
    rlabel, ilabels = None, {}
    rc_ = (a.label_sidecar, a.label_sha, a.record_meta)
    if any(x is not None for x in rc_) or a.input_sidecar:  # route C (09 / 2e, 10-18): all of them, and no typed status line
        if any(x is None for x in rc_) or a.text_note:
            sys.exit("STOP: --label-sidecar, --label-sha and --record-meta go together (with --input-sidecar), without --text-note")
        rlabel = routec_label(a.label_sidecar, a.record[1], f"{a.name} (post_main, readout {ro}) on route C pooled A @ {a.record[1]}", a.label_sha)
        ilabels = routec_inputs({k: v for k, v in (("panel", a.panel), ("metrics", a.metrics), ("pw", a.floor_pw), ("group", a.group_catch)) if v},
                                a.input_sidecar, sidecar_label(a.label_sidecar, a.record[1]), a.label_sha, a.record[1])
    a.readouts = [ro]  # Out.text picks each probe's training-length text for this readout (Codex r56)
    cmp_, est = comparator(), estimator()
    if [bool(a.metrics), bool(a.floor_pw), a.no_floor].count(True) != 1:
        sys.exit("STOP: give exactly one of --metrics (the per-set floor tick), --floor-pw (the pooled-world tick) or --no-floor")
    est_b = est == "B"
    pooled = a.right_source == "pooled_record"
    if pooled == bool(a.panel):  # never mix: the pooled right panel takes no PANEL record; the PANEL right panel needs one
        sys.exit("STOP: --right-source pooled_record takes no --panel; --right-source panel needs --panel")
    if pooled and (cmp_, est) != ("run_mean", "A"):
        sys.exit(f"STOP: the pooled right panel reads delta_full (comparator run_mean, estimator A); the plan gives {cmp_}/{est}")
    ckeys = ({"run_mean": "real_mean", "run1": "run1"}[cmp_] + ("|B" if est_b else ""), "sim|B" if est_b else "sim")
    skey = "sim_minus_run1" if cmp_ == "run1" else f"delta_full_{est}"
    if cmp_ == "run1" and a.avg_blocks:  # Codex ports r5: the coheadline S and Y summarise run-mean differences
        sys.exit("STOP: the coheadline averages summarise run-mean differences; with comparator run1 pass no --avg-blocks")
    o = Out(a.name, a, {"record": a.record, **({"panel": a.panel} if a.panel else {}), **({"metrics": a.metrics} if a.metrics else {}), **({"pw": a.floor_pw} if a.floor_pw else {}),
                        **({"group": a.group_catch} if a.group_catch else {})})
    if a.record_meta is not None:  # route C: --record's meta by key, before anything is read for drawing
        o.src["record_meta"] = check_record_meta(o.recs["record"].get("meta"), *a.record_meta)
    o.src |= {"comparator": cmp_, "catch_keys": list(ckeys), "signed_key": skey, "rerun_key": RERUN_KEY, "readout": ro, "right_source": a.right_source}

    def rnode(key_or_blk, k, r_, avg=False):
        """A right-panel node in POINTS: the PANEL's as stored, or the pooled record's (a fraction) x100, both read by key (logged as stored)."""
        if not pooled:
            return o.val("panel", ["coheadline", F, r_, key_or_blk, k] if avg else ["results", "pooled", F, "rows", f"{key_or_blk} {r_}", k])
        v = o.val("record", ["coheadline_signed", LEAD, "joint", F, f"{POOLED_BLK[key_or_blk]}|{r_}", k] if avg
                  else ["pooled", f"{key_or_blk} {r_}", LEAD, "joint", F, "gap", POOLED_GAP[k]])
        return {**v, "point": 100 * v["point"], **({"ci95": [100 * x for x in v["ci95"]]} if has_ci(v) else {})}

    if pooled:
        o.src["right_keys"] = {"probe": f'pooled["<p> {ro}"]["{LEAD}"]["joint"]["{F}"]["gap"][{[POOLED_GAP[skey], POOLED_GAP[RERUN_KEY]]}] x100',
                               "averages": f'coheadline_signed["{LEAD}"]["joint"]["{F}"][<{"|".join(POOLED_BLK.values())}>|{ro}]["S" | "Y"] x100'}
        bad = []
        for r_ in dict.fromkeys([ro] + list(a.scale_readouts or [])):  # every drawn probe readable at F, by key (fail closed when absent)
            for p in a.rows:
                n_ = o.val("record", ["pooled", f"{p} {r_}", LEAD, "joint", F])
                flags = {"readable['all']": (n_.get("readable") or {}).get("all"), "readable_real_line": n_.get("readable_real_line")}
                bad += [f"{p} {r_}: {k} not True" for k, v in flags.items() if v is not True]
        if bad:
            sys.exit(f"STOP: not readable at {F}: {bad}")
    else:
        rec_hc = (o.recs["panel"].get("combos") or {}).get("headline_combo") if isinstance(o.recs["panel"].get("combos"), dict) else None
        if rec_hc != LEAD:  # 09: the lead combination, by name, in both records (read as LEAD always; Codex r56)
            sys.exit(f"STOP: the lead must be pooled_all_origins (PANEL headline_combo {rec_hc!r})")
    if BUILD and POOL != LEAD:
        sys.exit(f"STOP: the lead must be pooled_all_origins (Stage A pool {POOL!r})")
    for blk in a.avg_blocks:  # the averages' members, checked by key before anything is drawn
        if pooled and blk not in POOLED_BLK:
            sys.exit(f"STOP: no pooled coheadline group for --avg-blocks {blk!r}")
        b_ = (o.val("record", ["coheadline_signed", LEAD, "joint", F, f"{POOLED_BLK[blk]}|{ro}"]) if pooled
              else o.val("panel", ["coheadline", F, ro, blk]))
        if b_.get("unavailable") or "S" not in b_ or "Y" not in b_:
            sys.exit(f"STOP: coheadline[{F}][{ro}][{blk}] has no S and Y: {b_.get('unavailable')}")
        if b_.get("S_estimator", "A") != est:
            sys.exit(f"STOP: coheadline {blk} S is estimator {b_.get('S_estimator', 'A')}, the plan says {est}")
        want = {"six": [f"{p} {ro}" for p in a.rows], "auroc_subset": [f"{p} {ro}" for p in a.subset_members]}.get(blk)
        if want is None or (blk == "auroc_subset" and len(a.subset_members) != 4) or sorted(b_.get("rows") or []) != sorted(want):
            sys.exit(f"STOP: coheadline {blk} rows {b_.get('rows')} are not the expected {want}")
        o.src.setdefault("average_members", {})[blk] = b_.get("rows")
        if blk == "auroc_subset" and pooled:  # the pooled record's own membership, by key (0d via 2e: meta.auroc06_subset)
            m_ = o.val("record", ["meta", "auroc06_subset"])
            if sorted(m_.get("members") or []) != sorted(a.subset_members) or not (isinstance(m_.get("label"), str) and "POST HOC" in m_["label"]
                                                                                 and ">= 0.6" in m_["label"]):
                sys.exit("STOP: meta.auroc06_subset's members are not --subset-members, or its label is not 'POST HOC ... AUROC >= 0.6'")
            o.src["subset_by"], o.src["subset_label"] = f"meta.auroc06_subset (rule {m_.get('rule')}, pin {m_.get('membership_pin')})", m_["label"]
        elif blk == "auroc_subset":  # the caption's selection rule (commands AUROC >= 0.6, membership fixed before any PANEL value), by key
            sb, lb = o.val("panel", ["coheadline", F, ro, blk, "subset_by"]), o.val("panel", ["coheadline", F, ro, blk, "label"])
            if not (isinstance(sb, str) and sb.startswith(SUBSET_RULE[0]) and all(t in sb for t in SUBSET_RULE[1:])) or not (isinstance(lb, str) and "POST HOC" in lb):
                sys.exit(f"STOP: coheadline {blk} subset_by is not 'commands AUROC >= 0.6, resolved before any PANEL value', or its label is not POST HOC")
            o.src["subset_by"], o.src["subset_label"] = sb, lb
    nav = len(a.avg_blocks)
    ys = {p: nav + (0.4 if nav else 0) + i for i, p in enumerate(a.rows)}  # the averages on top, then one row per probe
    plt.rcParams.update({**RC, "font.size": 11})
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(14, 1.3 + 0.62 * (len(a.rows) + nav)), sharey=True, gridspec_kw={"width_ratios": [1, 1.1]})
    h, n_by, floors, f_none = 0.36, {}, {}, {}
    if a.floor_pw and (o.val("pw", ["pass"]) is not True or o.val("pw", ["combo"]) != LEAD or o.val("pw", ["point"]) != "draw 0 (no interval)"):
        sys.exit("STOP: the pooled-world share record did not pass, is not pooled_all_origins, or is not a draw-0 point record (Codex r96)")
    for p in a.rows:
        lab = f"{p} {ro}"
        y = ys[p]
        for dy, k, col in ((-h / 2, ckeys[0], REAL), (h / 2, ckeys[1], SIM)):
            c = o.val("record", ["pooled", lab, LEAD, "joint", F, "catch", k])
            v = 100 * c["point"]
            axl.barh(y + dy, v, h * 0.92, color=col, zorder=2)
            hi = v
            if has_ci(c):  # endpoints drawn directly: a percentile interval may exclude its point
                lo_, hi = 100 * c["ci95"][0], 100 * c["ci95"][1]
                axl.plot([lo_, hi], [y + dy] * 2, color=INK, lw=1.1, zorder=3)
            axl.text(max(v, hi) + 1.2, y + dy, f"{v:.0f}{wmark(c)}", va="center", fontsize=9, color=INK)
        n_by[lab] = sum(o.val("record", ["coverage", lab, m, "coverage", "cohort_cells"]) for m in o.val("record", ["pooled", lab, LEAD])["members"])
        if a.metrics:  # the no-skill floor: RUN-LENGTH's union record, pooled exactly like the bars
            mnode = o.val("metrics", ["pooled", lab, "pooled_all_origins"])
            if isinstance(mnode.get("unavailable"), str):
                f_none[lab] = mnode["unavailable"]
            else:
                fl = o.val("metrics", ["pooled", lab, "pooled_all_origins", "metrics", FLOOR_KEY])  # missing: STOP; never the unsuffixed key
                floors[lab] = fl["point"]
                axl.plot([100 * fl["point"]] * 2, [y - h - 0.04, y + h + 0.04], color=INK, lw=2.2, zorder=4)
        if a.floor_pw:  # pooled world: shares[label][level]["B"]["registered_with_no_unit_runs"]["run_weighted"], a point at draw 0 (2e, 10-16)
            x = o.val("pw", ["shares", lab, F, "B", "registered_with_no_unit_runs", "run_weighted"])  # missing: STOP
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not (0.0 <= x <= 1.0):
                sys.exit(f"STOP: {lab}: the pooled-world share is not a number in [0, 1]")
            floors[lab] = x
            axl.plot([100 * x] * 2, [y - h - 0.04, y + h + 0.04], color=INK, lw=2.2, zorder=4)
    if nav and not a.group_catch:  # the left average rows are never left empty (2e via Aswin, 10-16)
        sys.exit("STOP: --avg-blocks needs --group-catch (the left panel's average rows)")
    if a.group_catch and nav:  # LEFT average rows: the group catch record by key; its members must be the right panel's block rows
        if o.val("group", ["meta", "record", "sha256"]) != a.record[1]:  # built on this figure's own Stage A record (Codex r92)
            sys.exit("STOP: the group catch record was built on a different Stage A record")
        if ckeys != ("real_mean", "sim"):
            sys.exit("STOP: the group catch record holds estimator (A)'s real_mean and sim only")
        if o.val("group", ["gate", "pass"]) is not True:
            sys.exit("STOP: the group catch record's gate did not pass")
        for i, blk in enumerate(a.avg_blocks):
            gn = o.val("group", ["groups", f"{blk}|{ro}"])
            if sorted(gn.get("members") or []) != sorted(o.src["average_members"][blk]) or (gn.get("level"), gn.get("mode"), gn.get("combo")) != (F, "joint", LEAD):
                sys.exit(f"STOP: group catch {blk}|{ro}: members or level/mode/combination differ from the right panel's block")
            for dy, k, col in ((-h / 2, ckeys[0], REAL), (h / 2, ckeys[1], SIM)):
                c = o.val("group", ["groups", f"{blk}|{ro}", "catch", k])
                v = 100 * c["point"]
                axl.barh(i + dy, v, h * 0.92, color=col, zorder=2)
                hi = v
                if has_ci(c):  # endpoints drawn directly, as the probe rows
                    lo_, hi = 100 * c["ci95"][0], 100 * c["ci95"][1]
                    axl.plot([lo_, hi], [i + dy] * 2, color=INK, lw=1.1, zorder=3)
                axl.text(max(v, hi) + 1.2, i + dy, f"{v:.0f}{wmark(c)}", va="center", fontsize=9, color=INK, fontweight="bold")
        o.src["left_average_rows"] = {b: f"group catch record: groups['{b}|{ro}']['catch'][{list(ckeys)}]" for b in a.avg_blocks}
    axl.set_xlim(0, 112)
    axl.set_xlabel("Attack runs caught (%)")
    vlab = []  # the right panel's value labels, by id (probe / average and readout), for the placement check
    for p in a.rows:  # RIGHT: the per-probe signed difference and the rerun (the PANEL record's, or the pooled record's own)
        if not pooled:
            row_ = o.val("panel", ["results", "pooled", F, "rows", f"{p} {ro}"])
            if skey not in row_:
                sys.exit(f"STOP: {p} {ro}: the PANEL record has no {skey}")
            if est_b and row_.get("delta_full_B_caveat"):  # estimator (B) carries its later-launch caveat into the caption (as post_excess)
                o.src.setdefault("B_caveat", row_["delta_full_B_caveat"])
        for dy, k, col in ((-h / 2, skey, SIM), (h / 2, RERUN_KEY, GRAY)):
            v = rnode(p, k, ro)
            axr.barh(ys[p] + dy, v["point"], h * 0.92, color=col, zorder=2)
            if has_ci(v):
                axr.plot(v["ci95"], [ys[p] + dy] * 2, color=INK, lw=1.1, zorder=3)
            if k == skey:
                vlab.append((f"{p} {ro}", axr.text(max(v["point"], v["ci95"][1] if has_ci(v) else v["point"]) + 0.6, ys[p] + dy,
                                                   f"{sgn2(v['point'])}{wmark(v)}", va="center", fontsize=9, color=INK)))
            elif not has_ci(v):
                wmark(v)
    for i, blk in enumerate(a.avg_blocks):  # the averages: S (simulated minus real) and Y (the rerun), the PANEL's coheadline block
        for dy, k, col in ((-h / 2, "S", SUM), (h / 2, "Y", GRAY)):
            v = rnode(blk, k, ro, avg=True)
            axr.barh(i + dy, v["point"], h * 0.92, color=col, zorder=2)
            if has_ci(v):
                axr.plot(v["ci95"], [i + dy] * 2, color=INK, lw=1.1, zorder=3)
            if k == "S":
                vlab.append((f"{blk}|{ro}", axr.text(max(v["point"], v["ci95"][1] if has_ci(v) else v["point"]) + 0.6, i + dy,
                                                     f"{sgn2(v['point'])}{wmark(v)}", va="center", fontsize=9, color=INK, fontweight="bold")))
            elif not has_ci(v):
                wmark(v)
    if a.scale_readouts:  # one right-panel scale across the readouts (Figure 1 and F1)
        if ro not in a.scale_readouts:
            sys.exit(f"STOP: --scale-readouts {a.scale_readouts} must include {ro}")
        ex = []
        for r_ in a.scale_readouts:
            nodes = [rnode(p, k, r_) for p in a.rows for k in (skey, RERUN_KEY)]
            nodes += [rnode(blk, k, r_, avg=True) for blk in a.avg_blocks for k in ("S", "Y")]
            ex += [x for v in nodes for x in [v["point"]] + (list(v["ci95"]) if has_ci(v) else [])]
        lo_, hi_ = min(ex + [0.0]), max(ex + [0.0])
        sp = (hi_ - lo_) or 1.0
        axr.set_xlim(lo_ - 0.05 * sp, hi_ + 0.2 * sp)
        o.src["right_x_scale"] = {"readouts": a.scale_readouts, "xlim": list(axr.get_xlim())}
    axr.axvline(0, color=MUTED, lw=ZERO_LW, zorder=1)
    axr.set_xlabel("Difference in catch rate (percentage points)")
    avg_lbl = {b: AVG_TXT.get(b, b) for b in a.avg_blocks}  # one row label at both readouts; no in-image word on how the four were chosen (Aswin via 2e, 10-16)
    ticks = [(i, avg_lbl[blk]) for i, blk in enumerate(a.avg_blocks)] + [(ys[p], label(p, labels)) for p in a.rows]
    axl.set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
    for t_ in axl.get_yticklabels()[:nav]:
        t_.set_fontweight("bold")
    axl.invert_yaxis()
    axl.tick_params(axis="y", length=0)
    axr.tick_params(axis="y", length=0)
    if nav:
        for ax in (axl, axr):
            ax.axhline(nav - 0.3, color="#dddddd", lw=1)
    real_lbl = ("Real (run 1)" if cmp_ == "run1" else "Real (all-runs mean, estimator (B))" if est_b
                else "Real (mean of two runs)")  # the legend follows the estimator (Codex r56); James's wording via 0d/2e, 10-16
    hand_l = [Patch(color=REAL, label=real_lbl),
              Patch(color=SIM, label="Simulated")] + ([Line2D([], [], color=INK, lw=2.2, label="No-skill floor")] if floors else [])
    hand_r = [Patch(color=SIM, label="Simulated minus real"), Patch(color=GRAY, label=RR_LEGEND)] + (
        [Patch(color=SUM, label="Average")] if nav else [])
    for fs in (9, 8, 7):  # ONE legend row above each panel (James via 0d/2e); smaller type until the two rows clear each other and the figure
        legs = [ax_.legend(handles=h_, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=len(h_), fontsize=fs, borderaxespad=0.0,
                           columnspacing=1.2, handlelength=1.6) for ax_, h_ in ((axl, hand_l), (axr, hand_r))]
        fig.tight_layout()
        fig.canvas.draw()
        r_ = fig.canvas.get_renderer()
        bl, br = (lg.get_window_extent(r_) for lg in legs)
        if not bl.overlaps(br) and bl.x1 <= br.x0 and br.x1 <= fig.bbox.x1 + 0.5 and min(bl.x0, br.x0) >= -0.5 and max(bl.y1, br.y1) <= fig.bbox.y1 + 0.5:
            break
    else:
        sys.exit("STOP: the one-row legends overlap each other or leave the figure at 7 pt")
    o.src["legend_fontsize"] = fs
    bad = labels_placed(fig, vlab)  # after the final layout
    if bad:
        sys.exit(f"STOP: {a.name}: value labels misplaced at 200 dpi: {bad}")
    o.src["value_labels"] = "two decimals, signed, U+2212 minus; inside the right axes and off the zero line at 200 dpi"
    o.src |= {"floor_key": (f'pooled["<probe> {ro}"]["pooled_all_origins"]["metrics"]["{FLOOR_KEY}"]["point"]' if a.metrics else
                            f'shares["<probe> {ro}"]["{F}"]["B"]["registered_with_no_unit_runs"]["run_weighted"] (draw 0, no interval)' if a.floor_pw else None),
              "floor_unavailable": f_none, "n_by_probe": n_by}
    ns = sorted(set(n_by.values()))
    if len(ns) != 1:  # one cohort for every plotted row, by key; the caption states that one n (the writing session, 10-16)
        sys.exit(f"STOP: {a.name}: the plotted rows' task counts differ ({ns}); the caption's single 'scored on all N tasks' would be wrong")
    sub_txt = (" " + SUBSET_TXT + (" " + SUBSET_TXT_R if ro == "R" else "") if "auroc_subset" in a.avg_blocks else "")  # no record names (2e v3.1)

    def members(b):  # the average's members by key, in plain names; the record keys stay in the source JSON (average_members)
        m = o.src["average_members"][b]
        return f"all {len(m)} probes" if sorted(m) == sorted(f"{p} {ro}" for p in a.rows) else ", ".join(label(r.rsplit(" ", 1)[0], labels) for r in m)
    avg_txt = "; ".join(f"{avg_lbl[b]}: the mean over {members(b)}" for b in a.avg_blocks)
    o.text(a.title, f"Probes scoring {NOUN[ro]}, scored on all {ns[0]:,} tasks; 1 alarm per 100 honest actions, each pipeline on its own alarm line "
           "(separate thresholds). Left: catch rates with 95% intervals over "
           "resampled main tasks (thresholds re-estimated in every draw)"
           + ("; average rows: equal-weight means of their probes' catch rates, intervals from the per-draw means over the same resampled tasks"
              if a.group_catch and nav else "")
           + (f"; tick: no-skill floor, the {FLOOR_TXT}" if a.metrics or a.floor_pw else "") + ("; point estimate at draw 0 (no interval)" if a.floor_pw else "")
           + ("".join(f"; {lb_}: no floor ({why})" for lb_, why in sorted(f_none.items())))
           + ". Right: simulated minus real catch in percentage points (" + ("sim − real run 1" if cmp_ == "run1" else f"run-mean Δ_full, estimator ({est})")
           + ") beside real run 2 minus real run 1 (both at τ_B), 95% intervals" + (f"; {avg_txt}" if nav else "")
           + "." + sub_txt + " " + STREAM + (f" Estimator (B): {o.src['B_caveat']}" if o.src.get("B_caveat") else "") + caption_end(bool(rlabel)), a.rows, n=ns[-1])
    if a.no_floor and a.no_floor_note:  # 0d: the note, verbatim, HEADS the caption (after Out.text's build and training-length text; Codex r62)
        o.caption = f"{a.no_floor_note}. {o.caption}"
        o.src["no_floor_note"] = a.no_floor_note
    if a.text_note:
        o.src["text_note"] = a.text_note
    if rlabel:  # route C: the sidecar-composed labels and their pins (09's rule; every drawn record)
        o.src |= {"label": rlabel, "label_sidecar": {"file": a.label_sidecar[0], "sha256": a.label_sidecar[1], "source": a.label_sidecar[2],
                                                    "label_sha256": a.label_sha},
                  "input_labels": ilabels, "input_sidecars": [{"role": r_, "file": f_, "sha256": s_, "source": c_, "step": l_}
                                                              for r_, f_, s_, c_, l_ in a.input_sidecar or []]}
    o.save(fig)
    if a.text_note:  # after the caption and Source line Out.save wrote (exclusive create), so never inside the caption
        with open(RES / f"{a.name}{a.stamp}_text.md", "a") as f:
            f.write(f"\nStatus (not in the image or the caption): {a.text_note}\n")
    if rlabel:
        with open(RES / f"{a.name}{a.stamp}_text.md", "a") as f:
            f.write(f"\nLabel (not in the image or the caption): {rlabel}\n" + "".join(f"Input label ({r_}): {l_}\n" for r_, l_ in sorted(ilabels.items())))


if __name__ == "__main__":
    main()
