"""Post figure: the certified attack detector's run-level AUROC ("activations gave the simulation away"; James's Doc comments via 0d/2e, 10-16).
--mode bars: per readout cell (tool outputs, commands) the AUROC telling simulated attack runs from real run 1 (problem c_simatk_vs_run1) beside
the real-vs-real control, run 2 against run 1 (problem d_run2_vs_run1), with the record's 95% bootstrap intervals over main tasks.
--mode roc: ROC curves from the record's per-run held-out scores (problems[...].run_scores, refit3's repaired values in place), simulated vs
run 1 solid and run 2 vs run 1 dashed, one panel per readout; each curve's AUC is recomputed from those scores and must equal the record's
run_level.auroc, else STOP. Every value is read by key from the certified record (refit3 @ 80d75b27e; certification NOTE @ cb1da6112): its
refit3.accepted and each problem's observed_certified and observed_complete must be true, and each cell's layer must be the record's
settings.primary layer for its readout.

  python post_attack.py --record attack_all_refit3_<stamp>.json <sha256> <path@commit> --name fig_attack_auroc --mode bars \
      --cells "reply|L48" "command|L63" --stamp _x
"""

import math
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import json

from postfig import GRAY, INK, MUTED, RC, RES, SIM, Out, args, no_stale

PROB = {"c_simatk_vs_run1": "Simulated vs real attack runs (run 1)", "d_run2_vs_run1": "Real run 2 vs real run 1 (control)"}
RO = {"reply": "Tool outputs", "command": "Commands"}


def auc_ranks(pos, neg):  # Mann-Whitney AUROC, ties count one half: P(pos > neg) + 0.5 P(pos == neg)
    ns = np.sort(neg)
    lt, le = np.searchsorted(ns, pos, "left"), np.searchsorted(ns, pos, "right")
    return float((lt.sum() + 0.5 * (le - lt).sum()) / (len(pos) * len(neg)))


def roc(pos, neg):  # (fpr, tpr) at every distinct score, highest first; ties step together
    s = np.r_[pos, neg]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o_ = np.argsort(-s, kind="mergesort")
    s, y = s[o_], y[o_]
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp, fp = np.cumsum(y)[last], np.cumsum(1 - y)[last]
    return np.r_[0.0, fp / len(neg)], np.r_[0.0, tp / len(pos)]


def main():
    a = args(__doc__.splitlines()[0], [("--record", "the certified attack-detector record (refit3)")],
             {"--name": {"required": True}, "--mode": {"required": True, "choices": ["bars", "roc", "calltable"]},
              "--allow-nonprimary": {"action": "store_true", "help": "a cell off its readout's primary layer is drawn, labelled non-primary (Aswin via 0d/2e)"},
              "--layer-choice": {"default": "", "help": "with --allow-nonprimary: the reason a non-primary layer is THE layer of this figure (labels drop "
                                                        "'non-primary'; the caption and _source.json state the registered primary and this reason)"},
              "--cells": {"nargs": "+", "required": True, "help": "record cells, e.g. 'reply|L48' 'command|L63'"},
              "--note": {"default": "", "help": "a note that heads the caption, verbatim (e.g. 'data check open (nm172)'; 2e, 10-16)"},
              "--title": {"default": "Telling simulated from real attack runs with the attack detector"}})
    if a.layer_choice.strip() and not a.allow_nonprimary:
        sys.exit("STOP: --layer-choice goes with --allow-nonprimary")
    o = Out(a.name, a, {"record": a.record})
    if o.val("record", ["refit3", "accepted"]) is not True:
        sys.exit("STOP: the record's refit3.accepted is not true")
    worlds = o.val("record", ["settings", "worlds"])  # run_scores key prefix -> world name
    probs = {p["name"]: p for p in o.val("record", ["settings", "problems"])}
    if set(PROB) - set(probs):
        sys.exit(f"STOP: the record lacks problems {sorted(set(PROB) - set(probs))}")
    cells = []
    for c in a.cells:
        ro_, ly = o.val("record", ["cells", c, "readout"]), o.val("record", ["cells", c, "layer"])
        prim = o.val("record", ["settings", "primary", ro_])
        if ro_ not in RO or c != f"{ro_}|L{ly}" or (ly != prim and not a.allow_nonprimary):
            sys.exit(f"STOP: cell {c}: readout {ro_!r}, layer {ly} is not the record's primary layer {prim} for that readout")
        for p in PROB:
            b = ["cells", c, "problems", p]
            if o.val("record", b + ["observed_certified"]) is not True or o.val("record", b + ["observed_complete"]) is not True:
                sys.exit(f"STOP: {c} {p}: not observed_certified and observed_complete")
        cells.append((c, ro_, ly))
        if ly != prim:
            o.src.setdefault("nonprimary", {})[c] = {"layer": ly, "primary_layer": prim}
    if a.layer_choice.strip():  # the chosen layer is recorded with the registered primary it departs from (Aswin, 10-16, via 2e)
        if not o.src.get("nonprimary"):
            sys.exit("STOP: --layer-choice given but every cell is at its primary layer")
        o.src["layer_choice"] = {"reason": a.layer_choice.strip(), "cells": o.src["nonprimary"]}
    lyt = lambda c, ly: f"layer {ly}, non-primary" if c in o.src.get("nonprimary", {}) and not a.layer_choice.strip() else f"layer {ly}"  # noqa: E731
    plt.rcParams.update({**RC, "font.size": 11})
    vals = {}
    for c, _, _ in cells:
        for p in PROB:
            b = ["cells", c, "problems", p]
            au = o.val("record", b + ["run_level", "auroc"])
            ci = o.val("record", b + ["run_auroc_boot", "ci95"])
            bt = {k: o.val("record", b + ["run_auroc_boot", k]) for k in ("n_draws", "failed_draws", "n_tasks")}
            ok_ci = isinstance(ci, list) and len(ci) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in ci) and ci[0] <= ci[1]
            if not (isinstance(au, (int, float)) and 0.0 <= au <= 1.0) or not ok_ci:
                sys.exit(f"STOP: {c} {p}: the run-level AUROC or its interval is malformed")
            vals[(c, p)] = (au, ci, bt)
    YLO = 0.4  # the writing session's y range (0.4 to 1); an interval endpoint below it STOPs, never clips (2e, 10-16)
    if a.mode == "calltable":  # per-action (call-level) AUROC with its bootstrap interval, every cell and problem, by key; a table, no figure
        lines = ["| Readout | " + " | ".join(PROB.values()) + " |", "|---|" + "---|" * len(PROB)]
        for c, ro_, ly in cells:
            row = []
            for p in PROB:
                b = ["cells", c, "problems", p]
                au, ci = o.val("record", b + ["call_level", "auroc"]), o.val("record", b + ["call_auroc_boot", "ci95"])
                nr = o.val("record", b + ["n_rows"])
                ok = isinstance(ci, list) and len(ci) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in ci) and ci[0] <= ci[1]
                if not (isinstance(au, (int, float)) and 0.0 <= au <= 1.0) or not ok:
                    sys.exit(f"STOP: {c} {p}: the call-level AUROC or its interval is malformed")
                row.append(f"{au:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] ({nr['class1']:,} vs {nr['class0']:,} actions)")
            lines.append(f"| {RO[ro_]} ({lyt(c, ly)}) | " + " | ".join(row) + " |")
        bt = {o.val("record", ["cells", c, "problems", p, "call_auroc_boot", k]) for c, _, _ in cells for p in PROB for k in ("n_draws",)}
        fd = sum(o.val("record", ["cells", c, "problems", p, "call_auroc_boot", "failed_draws"]) for c, _, _ in cells for p in PROB)
        md = (f"# {a.title}\n\n" + (f"{a.note.strip().rstrip('.')}. " if a.note.strip() else "")
              + "Per-action (call-level) AUROC of the certified attack detector: each action scored by its held-out logit, simulated attack actions "
              "against real run 1, beside real run 2 against real run 1; 95% intervals: bootstrap over main tasks "
              f"({'/'.join(f'{x:,}' for x in sorted(bt))} draws, {fd} failed). Keys: cells[<cell>].problems[<problem>].call_level.auroc and "
              ".call_auroc_boot.ci95. Values from the certified record (refit3, its repaired fits in place). Post hoc.\n\n"
              + "\n".join(lines) + f"\n\nSource: {a.name}{a.stamp}_source.json (inputs by path@sha; every value with its key).\n")
        no_stale(md, a.name)
        o.src |= {"cells": [c for c, _, _ in cells], "problems": PROB, "mode": a.mode, "note": a.note.strip() or None}
        with open(RES / f"{a.name}{a.stamp}.md", "x") as f:
            f.write(md)
        with open(RES / f"{a.name}{a.stamp}_source.json", "x") as f:
            f.write(json.dumps(o.src, indent=1))
        print(RES / f"{a.name}{a.stamp}.md")
        return
    if a.mode == "bars":
        low = [f"{c} {p}" for (c, p), (au, ci, _) in vals.items() if min(au, ci[0]) < YLO]
        if low:
            sys.exit(f"STOP: an AUROC or interval endpoint is below the y-axis floor {YLO}: {low} (not clipped; tell 2e)")
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        w = 0.36
        for i, (c, ro_, ly) in enumerate(cells):
            for j, (p, col) in enumerate((("c_simatk_vs_run1", SIM), ("d_run2_vs_run1", GRAY))):  # Figure 1's colours: orange sim vs real, gray real vs real
                au, ci, _ = vals[(c, p)]
                x = i + (j - 0.5) * w
                ax.bar(x, au - YLO, w * 0.92, bottom=YLO, color=col, zorder=2)
                ax.plot([x, x], ci, color=INK, lw=1.2, zorder=3)  # endpoints drawn directly: a percentile interval may exclude its point
                if ci[1] + 0.05 <= 1.0:  # the printed value above the interval, or inside the bar below it when the top is too close to 1
                    ax.text(x, ci[1] + 0.012, f"{au:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
                elif min(au, ci[0]) - 0.05 >= YLO:  # inside the bar: below both the bar top and the interval's low end (CUT: an interval may exclude its point)
                    ax.text(x, min(au, ci[0]) - 0.012, f"{au:.2f}", ha="center", va="top", fontsize=10, color="white", fontweight="bold")
                else:
                    sys.exit(f"STOP: {c} {p}: no room to print the value on its bar")
        ax.axhline(0.5, color=MUTED, lw=1, ls="--", zorder=1)
        ax.set_xlim(-0.6, len(cells) - 0.4)
        ax.text(len(cells) - 0.42, 0.5, "chance", ha="right", va="bottom", fontsize=9, color=MUTED)
        ax.set_xticks(range(len(cells)), [f"{RO[ro_]}\n({lyt(c, ly)})" for c, ro_, ly in cells])
        ax.set_ylim(YLO, 1.0)
        ax.set_ylabel("Run-level AUROC\n(higher = easier to tell apart)")
        ax.legend(handles=[Patch(color=SIM, label=PROB["c_simatk_vs_run1"]), Patch(color=GRAY, label=PROB["d_run2_vs_run1"])], frameon=False,
                  loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=1, fontsize=9, borderaxespad=0.0)
        fig.tight_layout()
    else:  # two panels: simulated vs real (orange shades), then the real-vs-real control (gray shades), each with the diagonal
        shade = {("c_simatk_vs_run1", 0): (SIM, "-"), ("c_simatk_vs_run1", 1): ("#a8441c", "--"),
                 ("d_run2_vs_run1", 0): (GRAY, "-"), ("d_run2_vs_run1", 1): ("#4f4f4b", "--")}
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), squeeze=False)
        for ax, p in zip(axes[0], PROB):
            for i, (c, ro_, ly) in enumerate(cells):
                sc = o.val("record", ["cells", c, "problems", p, "run_scores"])
                pos_w, neg_w = probs[p]["pos"], probs[p]["neg"]
                pos = np.array([v for k, v in sc.items() if worlds.get(k.split("|", 1)[0]) in pos_w], dtype=float)
                neg = np.array([v for k, v in sc.items() if worlds.get(k.split("|", 1)[0]) in neg_w], dtype=float)
                n_ = o.val("record", ["cells", c, "problems", p, "n_runs"])
                if len(pos) + len(neg) != len(sc) or (len(pos), len(neg)) != (n_["class1"], n_["class0"]) or not (np.isfinite(pos).all() and np.isfinite(neg).all()):
                    sys.exit(f"STOP: {c} {p}: the per-run scores do not split into the record's class counts, or are not finite")
                auc = auc_ranks(pos, neg)
                if abs(auc - vals[(c, p)][0]) > 1e-9:  # the curve drawn must be the one whose AUROC the record states
                    sys.exit(f"STOP: {c} {p}: the AUROC recomputed from run_scores differs from run_level.auroc")
                fpr, tpr = roc(pos, neg)
                col, ls = shade[(p, i % 2)]
                ax.plot(fpr, tpr, color=col, lw=2, ls=ls, label=f"{RO[ro_]} (layer {ly}), AUROC {vals[(c, p)][0]:.2f}")
                o.src.setdefault("roc", {}).setdefault(c, {})[p] = {"key": f'cells["{c}"]["problems"]["{p}"]["run_scores"]', "n_pos": len(pos), "n_neg": len(neg),
                                                                     "auc_recomputed": auc, "fpr": fpr.tolist(), "tpr": tpr.tolist()}
            ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(PROB[p], loc="left", fontsize=12)
            ax.set_xlabel("False-positive rate (real run 1 flagged)")
            ax.set_ylabel(f"True-positive rate ({'simulated' if p == 'c_simatk_vs_run1' else 'real run 2'} flagged)")
            ax.legend(frameon=False, loc="lower right", fontsize=9)
        fig.tight_layout()
    folds = sorted(set(o.val("record", ["run_folds"]).values()))  # the CV folds and the mains they group, by key (CUT optional via 2e)
    mains = o.val("record", ["inputs", "folds", "mains_used"])
    if mains != len(set(o.val("record", ["run_main"]).values())):
        sys.exit("STOP: inputs.folds.mains_used differs from the number of distinct run_main values")
    bt0 = {vals[k][2]["n_draws"] for k in vals}
    fd = sum(vals[k][2]["failed_draws"] for k in vals)
    nt = sorted({vals[k][2]["n_tasks"] for k in vals})
    n_txt = "; ".join(f"{RO[ro_]} {p_}: {o.val('record', ['cells', c, 'problems', p, 'n_runs', 'class1']):,} vs "
                      f"{o.val('record', ['cells', c, 'problems', p, 'n_runs', 'class0']):,} runs" for c, ro_, _ in cells
                      for p, p_ in (("c_simatk_vs_run1", "simulated vs run 1"), ("d_run2_vs_run1", "run 2 vs run 1")))
    cap = (f"Run-level AUROC of the certified attack detector (held-out scores from cross-validation: {len(folds)} folds, grouped by main task "
           f"({mains:,} mains); a run's "
           "score is the mean of its rows' held-out logits), telling simulated attack runs from real "
           "run 1, beside the real-vs-real control (real run 2 against real run 1); "
           + ("each readout at its primary layer. " if not o.src.get("nonprimary") else
              "layers: " + ", ".join(f"{RO[ro_].lower()} {ly}" for _, ro_, ly in cells) + " ("
              + "; ".join(f"the record's registered primary layer for {RO[ro_].lower()} is {o.src['nonprimary'][c]['primary_layer']}"
                          for c, ro_, _ in cells if c in o.src["nonprimary"]) + f"; {a.layer_choice.strip().rstrip('.')}). "
              if a.layer_choice.strip() else "each readout at its primary layer, and "
              + "; ".join(f"{RO[ro_]} also at layer {ly} (a non-primary layer; the record's primary layer for {RO[ro_].lower()} is "
                          f"{o.src['nonprimary'][c]['primary_layer']})" for c, ro_, ly in cells if c in o.src["nonprimary"]) + ". ")
           + (f"Bars start at {YLO} (y-axis {YLO} to 1; no interval endpoint lies below it). " if a.mode == "bars" else "")
           + (f"95% intervals: bootstrap over main tasks ({'/'.join(f'{x:,}' for x in sorted(bt0))} draws, {fd} failed; {', '.join(f'{x:,}' for x in nt)} "
              "tasks). " if a.mode == "bars" else "Curves from the record's per-run held-out scores; each AUC equals the record's run-level AUROC. ")
           + f"Runs: {n_txt}. Values from the certified record (refit3, its repaired fits in place). Post hoc.")
    o.src |= {"cells": [c for c, _, _ in cells], "problems": PROB, "mode": a.mode,
              "keys": {"auroc": 'cells["<cell>"]["problems"]["<problem>"]["run_level"]["auroc"]',
                       "ci95": 'cells["<cell>"]["problems"]["<problem>"]["run_auroc_boot"]["ci95"]'}}
    o.text(a.title, (f"{a.note.strip().rstrip('.')}. " if a.note.strip() else "") + cap)
    o.src["note"] = a.note.strip() or None
    o.save(fig)


if __name__ == "__main__":
    main()
