"""Post figure: per-task agreement (port of figs_simple/make_pertask.py). --mode scatter: per probe, each task's most suspicious action minus
its own alarm line, the anchor real run (x) against the other real run (gray) and against the simulated run (orange), R² from the anchor
record. --mode summary: R² with the anchor run for every probe, the other real run beside the simulated run. The anchor is real run 1
everywhere: the anchor record's `lead` block is read by name (sensitivity_run2_anchor never is), and its anchor field must equal --anchor for
every drawn probe, else STOP. --union-stream reads union_stream.lead[label].sets["<set> (union stream)"] (PERTASK via 2e, 10-16; the record's
gate.pass must be true) and prints the record's union_stream.caption and meta.interval_labels.thresholds by key.
--pooled-world (0d via 2e, 10-17; Figures 7 and 8 on the pooled anchor record): the scatter's origin rows come from the per-task record's own
bootstrap.sets[--set] by key (each member's base origin = its first word; no --origin-keys); the anchor record's label, from its
label sidecar (--label-sidecar, sha-checked), is quoted verbatim in the _text.md status line after the caption (as Figure 4: never in the
caption or the image; save_status), under 09's rule keyed on the CITED A (--cited-a, an explicit pin; READ_LOG @
8a4e2bf84 L2213, 2e 10-17; pooled_label below): d8fcb945f (the 171-pool A) needs the .md sidecar's one "**Label:**" line naming 171-pool and
PRELIMINARY; route C's A needs A's JSON label sidecar (--a-label-sidecar) and an anchor JSON sidecar whose label STARTS WITH A's label; the
summary's R² axis comes from --r2-axis (below; the v2 plan fixes it at 0 to 0.30). Any drawn set the anchor record labels not
readable STOPs (ids only). Value labels use a true minus sign (U+2212).
--r2-axis (the summary's R² axis, a plan field; writing via 0d / 2e, 10-17): "auto" (default; the per-set autoscale) or "LO,HI" (e.g. "0,0.30"):
a FIXED shared axis from LO = 0 to HI (bars start at 0, so LO must be 0); every plotted point and interval end must lie in [0, HI] (never
clipped silently) and
every value label inside the axes at the saved dpi, else STOP with ids only.

  python post_pertask.py --pertask <rec>.json <sha> <path@commit> --anchor-record <rec>.json <sha> <path@commit> --mode scatter --rows "FIBS C" --stamp _x
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from postfig import BUILD, GRAY, INK, MUTED, RC, RES, SIM, Out, args, has_ci, label, wmark

ALLO = {"pooled_all_origins": ["S_old", "S_new", "S_newmains", "S_paper493"],  # run_pertask_anchor.ALLO @ ea843ebc9 (union sets' base origins)
        "pooled_all_origins_without_S4": ["S_old", "S_new", "S_newmains without S4", "S_paper493"], "paper_cells": ["S_old", "S_new", "S_paper493"]}
LABEL_LINE = re.compile(r"^\*\*Label:\*\* (.+)$", re.M)  # the LABEL sidecar's one label line (…_LABEL.md @ 25b59ef3f)
A171 = "d8fcb945f"  # the 171-pool pooled A's commit, as the 171-pool anchor's --final cites it (SPEND L573 @ 2c9377bec)


def sidecar_json(raw, what):
    """A route C label sidecar: exactly {record, record_sha256, label}, each a non-empty string, else STOP."""
    try:
        sc = json.loads(raw)
    except ValueError:
        sys.exit(f"STOP: {what} is not JSON (a route C sidecar is {{record, record_sha256, label}})")
    if not isinstance(sc, dict) or set(sc) != {"record", "record_sha256", "label"} or not all(isinstance(v, str) and v.strip() for v in sc.values()):
        sys.exit(f"STOP: {what} is not {{record, record_sha256, label}} as non-empty strings")
    return sc


def pooled_label(cited_a, final_record, anchor_sha, anchor_meta_label, side, a_side):
    """09's label rule (READ_LOG @ 8a4e2bf84 L2213), keyed on the CITED A: an explicit CLI pin (2e 10-17), never the records' own labels.
    side / a_side: (bytes, pinned sha256) of the anchor record's label sidecar and of A's label sidecar (a_side None when not given); each sha
    is checked before its bytes are parsed. The anchor record must have read the cited A (its meta.union_stream.final_record is
    "<cited A>:<path>@<sha>"). Cited A d8fcb945f (171-pool): the .md sidecar's one **Label:** line names 171-pool and PRELIMINARY, and no A
    sidecar is given. Any other cited A (route C): A's JSON sidecar names that A (record_sha256 = final_record's sha) and route C; the anchor's
    JSON sidecar names the anchor record (record_sha256 = its sha), its label STARTS WITH A's label exactly, has no "171-pool", contains
    PRELIMINARY (no FINAL-release record is pinned, and no argument for one exists yet) and appears in the anchor record's own meta.label.
    Anything else STOPs. Returns the label to quote verbatim."""
    if not (isinstance(cited_a, str) and re.fullmatch(r"[0-9a-f]{7,40}", cited_a)):
        sys.exit("STOP: --cited-a is not a 7-40 hex commit")
    if not isinstance(final_record, str) or final_record.split(":", 1)[0] != cited_a or "@" not in final_record:
        sys.exit("STOP: the anchor record's union stream did not read the cited A (meta.union_stream.final_record)")
    for s_, what in ((side, "the label sidecar"), (a_side, "A's label sidecar")):
        if s_ is not None and hashlib.sha256(s_[0]).hexdigest() != s_[1]:
            sys.exit(f"STOP: sha mismatch for {what}")
    if A171.startswith(cited_a) or cited_a.startswith(A171):  # the 171-pool A
        if a_side is not None:
            sys.exit("STOP: an A label sidecar is for route C's A, not the 171-pool A d8fcb945f")
        lines_ = LABEL_LINE.findall(side[0].decode("utf-8", "replace"))
        if len(lines_) != 1 or "171-pool" not in lines_[0] or "PRELIMINARY" not in lines_[0]:
            sys.exit("STOP: cited A d8fcb945f: the label sidecar has no single **Label:** line naming 171-pool and PRELIMINARY")
        return lines_[0].strip()
    if len(cited_a) != 40:  # 2e (10-17): route C cites its A by the FULL 40-hex commit everywhere (the pins, the --pertask reference, the labels)
        sys.exit("STOP: a route C A is cited by its full 40-hex commit")
    if a_side is None:
        sys.exit("STOP: a route C A needs --a-label-sidecar (A's {record, record_sha256, label})")
    asc, sc = sidecar_json(a_side[0], "A's label sidecar"), sidecar_json(side[0], "the label sidecar")
    if asc["record_sha256"] != final_record.rsplit("@", 1)[1] or "route C" not in asc["label"]:
        sys.exit("STOP: A's label sidecar does not name the A the anchor record read, or does not name route C")
    lab = sc["label"]
    if sc["record_sha256"] != anchor_sha or not lab.startswith(asc["label"]) or "171-pool" in lab or "PRELIMINARY" not in lab:
        sys.exit("STOP: the label sidecar does not name the anchor record, or its label does not start with A's label, or names 171-pool, "
                 "or is not PRELIMINARY")
    if not isinstance(anchor_meta_label, str) or lab not in anchor_meta_label:
        sys.exit("STOP: the label sidecar's label is not in the anchor record's own meta.label")
    return lab


def r2_axis(spec):
    """'auto' -> None; 'LO,HI' -> (LO, HI) with LO == 0 < HI <= 1 (bars start at 0), else STOP (the plan's --r2-axis field)."""
    if spec == "auto":
        return None
    try:
        lo, hi = (float(x) for x in spec.split(","))
    except ValueError:
        sys.exit(f"STOP: --r2-axis {spec!r} is neither 'auto' nor 'LO,HI'")
    if not (lo == 0.0 and 0.0 < hi <= 1.0):  # bars are drawn from 0: a lower bound above 0 would clip every bar base (refresh2 r2 opt. 5)
        sys.exit(f"STOP: --r2-axis {spec!r} is not LO = 0 < HI <= 1 (bars start at 0)")
    return lo, hi


def m2(x):
    """A two-decimal value label with a true minus sign (U+2212), never ASCII '-' (Figs 1/5/6 convention)."""
    return f"{x:.2f}".replace("-", "\u2212")


def save_status(o, fig, plabel, a, dpi=200):
    """Figure 4's pattern (a8953dad R-1 via 2e, 10-18; post_fig4_sweep.py @ 9617995a5 L473-480): no status word in the title, caption or
    image ("prelim" STOPs); the anchor record's label goes in the _text.md file AFTER the caption and the source line, as a status line,
    and stays in _source.json (o.src["label_sidecar"]); never in the caption or the image."""
    from matplotlib.text import Text

    for s in [o.title, o.caption] + [t.get_text() for t in fig.findobj(Text)]:
        if re.search(r"(?i)prelim", s or ""):
            sys.exit("STOP: 'preliminary' in the title, caption or image (as Figure 4: the status goes in the text file's status line)")
    o.save(fig, dpi=dpi)
    if plabel:
        with open(RES / f"{a.name}{a.stamp}_text.md", "a") as f:  # after the caption and source line (never in the caption)
            f.write(f'\nStatus (not in the image or the caption): label, verbatim from the anchor record\'s label sidecar: "{plabel}"\n')


def outside_axes(fig, texts, dpi):
    """Ids (never values) of the value labels whose drawn text leaves its axes, measured at the saved dpi."""
    old = fig.dpi
    fig.set_dpi(dpi)
    fig.canvas.draw()
    rdr = fig.canvas.get_renderer()
    bad = []
    for id_, t in texts:
        bb, ab = t.get_window_extent(rdr), t.axes.get_window_extent(rdr)
        if not (bb.x0 >= ab.x0 and bb.x1 <= ab.x1 and bb.y0 >= ab.y0 and bb.y1 <= ab.y1):
            bad.append(id_)
    fig.set_dpi(old)
    return bad


def main():
    a = args(__doc__.splitlines()[0], [("--pertask", "per-task record"), ("--anchor-record", "per-task anchor (R²) record")],
             {"--name": {"required": True}, "--mode": {"required": True, "choices": ["scatter", "summary"]},
              "--rows": {"nargs": "+", "required": True, "help": "probe labels as in the records, e.g. 'FIBS C'"},
              "--anchor": {"default": "run1"}, "--set": {"default": "pooled", "help": "the anchor record's set whose R² is printed"},
              "--union-stream": {"action": "store_true", "help": "read the set's union-stream row (union_stream.lead[...]['<set> (union stream)'])"},
              "--origin-keys": {"default": None, "help": "JSON {base origin: the per-task record's row key} for the scatter (explicit; absent row = STOP)"},
              "--per-origin-appendix": {"action": "store_true", "help": "allow per-origin-stream intervals (an appendix figure, labelled)"},
              "--pooled-world": {"action": "store_true", "help": "the pooled anchor record: origins from the per-task record by key, the label quoted"},
              "--label-sidecar": {"nargs": 3, "metavar": ("FILE", "SHA256", "SOURCE"), "default": None,
                                  "help": "--pooled-world: the anchor record's label sidecar (171-pool: .md, its **Label:** line; route C: JSON); quoted verbatim"},
              "--cited-a": {"default": None, "help": "--pooled-world: the A commit the anchor record read (d8fcb945f = 171-pool; else route C)"},
              "--a-label-sidecar": {"nargs": 3, "metavar": ("FILE", "SHA256", "SOURCE"), "default": None,
                                    "help": "--pooled-world, route C A only: A's label sidecar {record, record_sha256, label}"},
              "--r2-axis": {"default": "auto", "help": "summary R² axis: 'auto' or a fixed 'LO,HI' (plan field; ends checked, never clipped)"},
              "--labels": {"default": "{}"}, "--title": {"default": "Per-task agreement with the anchor real run"}})
    labels = json.loads(a.labels)
    okeys = json.loads(a.origin_keys) if a.origin_keys else None
    o = Out(a.name, a, {"pertask": a.pertask, "anchor": a.anchor_record})
    plabel = None
    if a.pooled_world:  # 0d via 2e (10-17): never mixed with an explicit mapping; the union row of the lead combination only
        if not a.label_sidecar or not a.cited_a or a.origin_keys or not a.union_stream or a.set != "pooled_all_origins":
            sys.exit("STOP: --pooled-world takes --cited-a, --label-sidecar, --union-stream and --set pooled_all_origins, and no --origin-keys")
        rd_ = lambda s: None if s is None else ((RES / Path(s[0]).name).read_bytes(), s[1])  # noqa: E731  (bytes, pinned sha256)
        plabel = pooled_label(a.cited_a, o.val("anchor", ["meta", "union_stream", "final_record"]), a.anchor_record[1],
                              (o.recs["anchor"].get("meta") or {}).get("label"), rd_(a.label_sidecar), rd_(a.a_label_sidecar))
        o.src["label_sidecar"] = {"file": a.label_sidecar[0], "sha256": a.label_sidecar[1], "source": a.label_sidecar[2], "label": plabel,
                                  "cited_a": a.cited_a, "rule": "09 @ READ_LOG 8a4e2bf84 L2213, keyed on the cited A (post_pertask.pooled_label)"}
        if a.a_label_sidecar:
            o.src["a_label_sidecar"] = {"file": a.a_label_sidecar[0], "sha256": a.a_label_sidecar[1], "source": a.a_label_sidecar[2]}
        bs = o.val("pertask", ["bootstrap", "sets", a.set])  # the per-task record's own members of the set, by key (missing: STOP)
        base_ = {k.split(" ", 1)[0]: k for k in bs} if isinstance(bs, list) and all(isinstance(k, str) for k in bs) else {}
        if len(base_) != len(bs or []) or sorted(base_) != sorted(ALLO[a.set]):
            sys.exit(f"STOP: the per-task record's bootstrap.sets[{a.set!r}] members do not map one-to-one onto {ALLO[a.set]}")
        okeys = base_
        o.src["origin_keys"] = {"from": f'bootstrap.sets["{a.set}"] (per-task record, by key; base origin = first word)', "map": okeys}
    elif a.label_sidecar or a.cited_a or a.a_label_sidecar:
        sys.exit("STOP: --label-sidecar, --cited-a and --a-label-sidecar are for --pooled-world")
    other, blk = {}, "lead"  # PERTASK via 2e: the lead block anchors every probe on run 1; sensitivity_run2_anchor is never the lead
    for p in a.rows:
        anc = o.val("anchor", [blk, p, "anchor"])
        if anc != a.anchor:
            sys.exit(f"STOP: {p}: the anchor record's lead anchor is {anc!r}, not {a.anchor!r}")
        other[p] = o.val("anchor", [blk, p, "other_real"])
    if a.union_stream:  # the union row; the record's own gate must have passed
        if o.val("anchor", ["gate", "pass"]) is not True:
            sys.exit("STOP: the anchor record's gate.pass is not true")
        sbase = lambda p: ["union_stream", blk, p, "sets", f"{a.set} (union stream)"]  # noqa: E731
    else:
        sbase = lambda p: [blk, p, "sets", a.set]  # noqa: E731
    nr_ = [f"{p} | {sbase(p)[-1]}" for p in a.rows if "readability" in o.val("anchor", sbase(p))]  # labelled not readable (09 item 45(2))
    if nr_:
        sys.exit(f"STOP: the anchor record labels these drawn sets not readable: {nr_}")
    r2 = {(p, k): o.val("anchor", sbase(p) + ["r2", k]) for p in a.rows for k in ("yardstick", "main")}
    il = (o.recs["anchor"].get("meta") or {}).get("interval_labels") or {}
    thr = il.get("thresholds") if isinstance(il.get("thresholds"), str) else "thresholds fixed at the point estimate"
    stream = ""  # 09 RULE item 30 (Codex r31): the anchor record's pooled set combines independent per-origin draws unless shown otherwise
    if a.union_stream:  # the record's caption, verbatim, whenever the union row is read (Codex r52)
        cap = o.val("anchor", ["union_stream", "caption"])
        if not isinstance(cap, str) or not cap.strip():
            sys.exit("STOP: the anchor record's union_stream.caption is missing")
        stream = f"Intervals ({a.set}, union stream; 09's caption, verbatim): {cap.strip().rstrip('.')}. "
    if BUILD:
        if a.union_stream:
            if a.set != "pooled_all_origins":  # Codex r34: the union label needs exactly the lead combination's name
                sys.exit(f"STOP: --set {a.set!r} is not pooled_all_origins; other streams need --per-origin-appendix")
        elif not a.per_origin_appendix:  # 09: a body figure reads the union stream; per-origin streams only as a labelled appendix
            sys.exit("STOP: no --union-stream: pass it (the record's union row) or --per-origin-appendix for a labelled appendix figure")
        else:
            lab_ = (il.get("sets") or {}).get(a.set) or "per-origin streams (09 RULE item 30: appendix)"
            stream = f"Intervals: {lab_}; pointwise, {thr}. "
        o.src["interval_stream"] = stream
    probes = [p.rsplit(" ", 1)[0] for p in a.rows]
    if a.mode == "summary":
        xr = r2_axis(a.r2_axis)
        if xr:  # no silent clipping: every plotted point and interval end inside the fixed axis (ids only)
            out_ = [f"{p} | {k}" for p in a.rows for k in ("yardstick", "main")
                    if not all(xr[0] <= x_ <= xr[1] for x_ in [r2[(p, k)]["point"]] + (list(r2[(p, k)]["ci95"]) if has_ci(r2[(p, k)]) else []))]
            if out_:
                sys.exit(f"STOP: {a.name}: plotted R² points or interval ends outside the fixed axis {a.r2_axis}: {out_}")
        plt.rcParams.update(RC)
        fig, ax = plt.subplots(figsize=(8, 1.2 + 0.55 * len(a.rows)))
        h, vl = 0.38, []
        for i, p in enumerate(a.rows):
            for dy, k, col in ((-h / 2, "yardstick", GRAY), (h / 2, "main", SIM)):
                v = r2[(p, k)]
                ax.barh(i + dy, v["point"], height=h * 0.9, color=col, zorder=2)
                if has_ci(v):
                    ax.plot(v["ci95"], [i + dy] * 2, color=INK, lw=1, zorder=3)
                off = 0.01 * (xr[1] - xr[0]) if xr else 0.01  # the label's gap after the interval: 1% of a fixed axis (per-set: 0.01)
                vl.append((f"{p} | {k}", ax.text((v["ci95"][1] if has_ci(v) else v["point"]) + off, i + dy, f"{m2(v['point'])}{wmark(v)}",
                                                 va="center", fontsize=9)))
        if xr:
            ax.set_xlim(*xr)
        ax.set_yticks(range(len(a.rows)), [label(q, labels) for q in probes])
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(f"R² with real run {a.anchor[-1]} across tasks (higher = more alike)")
        ax.legend(handles=[Patch(color=GRAY, label="The other real run"), Patch(color=SIM, label="Simulated run")], frameon=False, loc="lower right")
        fig.tight_layout()
        if xr:
            bad = outside_axes(fig, vl, 200)
            if bad:
                sys.exit(f"STOP: {a.name}: value labels outside the axes at 200 dpi: {bad}")
            o.src["r2_axis"] = {"fixed": list(xr), "rule": "every plotted point and interval end inside (STOP otherwise); bars from 0"}
            o.src["value_labels"] = f"two decimals, U+2212 minus; inside the axes (R² axis fixed at {xr[0]:g} to {xr[1]:g}) at 200 dpi"
        nc = [o.val("anchor", sbase(p) + ["n_cells"]) for p in a.rows]  # each probe's own n, by key (2e: never a hard-coded cohort size)
        ncs = sorted(set(nc))
        n_txt = (f"{ncs[0]:,} tasks with all three runs scored" if len(ncs) == 1 else "tasks with all three runs scored, per probe: "
                 + "; ".join(f"{label(q, labels)} {c:,}" for q, c in zip(probes, nc)))
        o.text(a.title, stream + f"Commands, 1 in 100 honest actions; {n_txt} (the anchor record's n_cells); 95% intervals over resampled tasks; "
               f"anchor: real run {a.anchor[-1]} for every probe. "
               "Post hoc.", probes, n=max(nc))
        save_status(o, fig, plabel, a)
        return
    plt.rcParams.update({**RC, "font.size": 12})
    fig, axes = plt.subplots(len(a.rows), 2, figsize=(10, 4.6 * len(a.rows)), squeeze=False)
    for r, p in enumerate(a.rows):
        rows = o.val("pertask", ["rows", p])
        cells = {}
        mem = ALLO.get(a.set)  # the union set's origins, as the anchor runner's ALLO (ea843ebc9); "pooled" = every origin; else the one set
        if okeys is not None:  # explicit mapping, base origin -> the record's row key (pooled-world rows carry suffixes; 2e): absent = STOP
            if not mem or set(okeys) != set(mem):
                sys.exit(f"STOP: --origin-keys must map exactly the set's origins {mem} (got {sorted(okeys)})")
            miss = [okeys[b] for b in mem if not (isinstance(rows.get(okeys[b]), dict) and "per_task" in rows[okeys[b]])]
            if miss:
                sys.exit(f"STOP: {p}: the per-task record has no per_task rows {miss}")
            origins = [okeys[b] for b in mem]
        else:
            if any(" [pooled: " in k for k in rows):  # a pooled-world record's suffixed rows need the explicit mapping (Astra r7 optional)
                sys.exit(f"STOP: {p}: the per-task record has pooled-world rows; pass --origin-keys")
            origins = sorted(k for k, v in rows.items() if isinstance(v, dict) and "per_task" in v
                             and (k in mem if mem else (a.set == "pooled" or k == a.set)))
        for origin in origins:
            for e in rows[origin]["per_task"]:
                cells.setdefault((origin, e["cell"]), {})[e["arm"]] = e["max_minus_tau"]
        fin = lambda x: isinstance(x, (int, float)) and np.isfinite(x)  # noqa: E731  the runner's cells: finite run 1, run 2 and sim (Codex r52)
        keep = {c: v for c, v in cells.items() if all(fin(v.get(k)) for k in (a.anchor, other[p], "sim"))}
        pts = list(keep.values())
        n_rec = o.val("anchor", sbase(p) + ["n_cells"])
        if len(pts) != n_rec:  # Codex ports r1: the plotted points must be the set whose R² is printed
            sys.exit(f"STOP: {p}: {len(pts)} plotted tasks but the anchor record's {a.set} set has n_cells {n_rec}")
        o.src.setdefault("n_points", {})[p] = len(pts)
        o.src.setdefault("scatter_points", {})[p] = {"key": f'rows["{p}"][<origin>]["per_task"][*]["max_minus_tau"]',
                                                     "points": [{"origin": c[0], "cell": c[1], **{k: v[k] for k in (a.anchor, other[p], "sim")}}
                                                                for c, v in sorted(keep.items())]}
        x = np.array([q[a.anchor] for q in pts])
        ys = {arm: np.array([q[arm] for q in pts]) for arm in (other[p], "sim")}
        allv = np.r_[x, ys[other[p]], ys["sim"]]  # EVERY point inside the limits (0d via 2e, v2): min/max of all values, small padding,
        pad = 0.03 * ((allv.max() - allv.min()) or 1.0)  # the same limits on both panels and both axes
        lo, hi = float(allv.min() - pad), float(allv.max() + pad)
        o.src.setdefault("axis_limits", {})[p] = [lo, hi]
        for c, (arm, col, k) in enumerate(((other[p], GRAY, "yardstick"), ("sim", SIM, "main"))):
            ax, y = axes[r, c], ys[arm]
            ax.scatter(x, y, s=10, alpha=0.35, color=col, lw=0)
            b, a0 = np.polyfit(x, y, 1)  # drawing aid only; the printed R² is the record's
            xx = np.array([lo, hi])
            ax.plot(xx, a0 + b * xx, color=INK, lw=2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.axhline(0, color=MUTED, lw=0.9, ls="--", zorder=0)
            ax.axvline(0, color=MUTED, lw=0.9, ls="--", zorder=0)
            ax.text(0.04, 0.96, f"R² = {m2(r2[(p, k)]['point'])}", transform=ax.transAxes, va="top", fontsize=13, color=INK, fontweight="bold")
            ax.set_xlabel(f"Real run {a.anchor[-1]}: most suspicious action,\ndistance above the alarm line")
            ax.set_ylabel(("Real run " + arm[-1] if arm != "sim" else "Simulated run") + ": most suspicious action,\ndistance above the alarm line")
            xl_, yl_ = ax.get_xlim(), ax.get_ylim()  # render-time check: every point of this panel is drawn (none outside the limits)
            inside = int(((x >= xl_[0]) & (x <= xl_[1]) & (y >= yl_[0]) & (y <= yl_[1])).sum())
            if inside != len(pts) or inside != n_rec:
                sys.exit(f"STOP: {p} {k}: {inside} points inside the limits, {len(pts)} plotted, anchor n_cells {n_rec}")
            o.src.setdefault("drawn_points_per_panel", {}).setdefault(p, {})[k] = inside
            if c == 0 and r > 0:  # the first probe's name is one figure-level label above both panels (0d via 2e, v2: no overlap)
                ax.set_title(label(p.rsplit(" ", 1)[0], labels), loc="left", fontsize=13)
            if r == 0:
                ax.text(0.5, 1.08, "Two real runs (how well a rerun agrees)" if c == 0 else "Real against simulated", transform=ax.transAxes,
                        ha="center", fontsize=12, color=INK, fontweight="bold")
    head = fig.text(0.5, 0.995, label(a.rows[0].rsplit(" ", 1)[0], labels), ha="center", va="top", fontsize=14, fontweight="bold", color=INK,
                    gid="panel-label")  # a panel label, not a title (postfig.save allows only this gid)
    top = 0.95
    fig.set_dpi(170)  # the overlap check at the save DPI (Codex r99 optional)
    for _ in range(12):  # lower the panels until the label meets no panel header or axes (titles included)
        fig.tight_layout(rect=(0, 0, 1, top))
        fig.canvas.draw()
        r_ = fig.canvas.get_renderer()
        hb = head.get_window_extent(r_)
        if not any(ax.get_tightbbox(r_).overlaps(hb) for ax in axes[0]):
            break
        top -= 0.02
    else:
        sys.exit("STOP: the probe label still overlaps a panel")
    n = sorted(set(o.src["n_points"].values()))
    o.text(a.title, "One point per task with all three runs scored (" + (f"{n[0]:,} tasks" if len(n) == 1 else "per probe: " + "; ".join(
               f"{label(q, labels)} {o.src['n_points'][p]:,}" for q, p in zip(probes, a.rows))) + "; equal to the anchor record's n_cells by key), "
           "probes scoring commands. "
           f"Each axis shows how far a run's most suspicious action sits above (or below) its own alarm line; dashed lines mark zero, so points right "
           f"of or above them are caught. Anchor: real run {a.anchor[-1]}. R² from the anchor record ({a.set} set). " + stream + "Post hoc.",
           probes, n=n[-1])
    save_status(o, fig, plabel, a, dpi=170)


if __name__ == "__main__":
    main()
