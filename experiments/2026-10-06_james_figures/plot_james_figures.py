"""Four figures for the 2026-10-07 meeting with James, read from existing result files only (config/sources.yaml).

  uv run --with matplotlib --with pyyaml python experiments/2026-10-06_james_figures/plot_james_figures.py

Writes figures/fig1..fig4 PNGs and figures.md (every bar's value, 95% interval, n, source file and key). Nothing is
recomputed: each number is the one stored in its track's result file.
"""

import hashlib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).parent
SRC = yaml.safe_load((HERE / "config" / "sources.yaml").read_text())
OUT = HERE / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # dataviz reference palette, categorical slots 1-3
NEUTRAL = "#a3a3a0"  # Figure 3a's bars: blue/orange mean real/simulated in 3b and 3c, so 3a uses neither
INK, MUTED, GRID = "#262624", "#6b6b68", "#e8e8e5"
ROWS = []  # figures.md rows
HASHES = {}  # source -> sha256 of the file each value came from


SNAPSHOT = HERE / "inputs" / "snapshot.json"
REFRESH = "--refresh-snapshot" in sys.argv  # accept a source file whose sha256 differs from the snapshot's
_OLD = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {"source": {}, "sha256": {}, "values": {}}
_SNAP, _FILES = {"source": {}, "sha256": {}, "values": {}}, {}
RESULT_KEYS = {"headline", "headline_held_out_environment", "by_layer", "results"}  # the bulky per-run results


def _where(label: str) -> tuple:
    """Where a source lives: ("git", commit, path) for committed result files, ("local", path) for the Q1 files."""
    if label.startswith("q1/"):
        return ("local", Path(SRC["q1"]["local"]).expanduser() / SRC["q1"]["dir"] / label[3:])
    if "repo_path" in SRC[label.split(":")[0]]:  # a result file of this experiment, committed beside the script
        return ("local", HERE / SRC[label]["repo_path"])
    section, field = (label.split(":") + ["path"])[:2]
    return ("git", SRC[section]["commit"], SRC[section][field])


def _read(where: tuple) -> bytes | None:
    if where[0] == "local":
        return where[1].read_bytes() if where[1].exists() else None
    try:
        return subprocess.check_output(["git", "show", f"{where[1]}:{where[2]}"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:  # the commit is not in this clone (e.g. its branch was squash-merged)
        return None


def _provenance(key: str, value):
    """A Q1 file's provenance field as saved: whole, except a manifest's per-run list (about 380 KB), which is
    summarized as its length, runs per arm, and the sha256 of its canonical JSON."""
    if key != "manifest" or not isinstance(value, dict) or "runs" not in value:
        return value
    runs = value["runs"]
    per_arm = {}
    for r in runs if isinstance(runs, list) else []:  # each run is [arm, sample key]
        arm = r.get("arm") if isinstance(r, dict) else r[0] if isinstance(r, list) and r else None
        if arm is not None:
            per_arm[arm] = per_arm.get(arm, 0) + 1
    digest = hashlib.sha256(json.dumps(runs, sort_keys=True).encode()).hexdigest()
    return value | {"runs": {"n": len(runs), "per_arm": per_arm, "sha256": digest}}


def get(label: str, *keys: str):
    """One value from a source, recorded with the file's sha256 in inputs/snapshot.json. A source that cannot be
    read (an uncommitted Q1 file, a commit missing from this clone) falls back to the snapshot. A source whose
    sha256 differs from the snapshot's stops the run unless --refresh-snapshot is given."""
    if label not in _FILES:
        data = _read(_where(label))
        if data is not None:
            digest = hashlib.sha256(data).hexdigest()
            old = _OLD["sha256"].get(label)
            if old and old != digest and not REFRESH:
                raise ValueError(f"{label} changed since the snapshot ({old[:12]} -> {digest[:12]}); "
                                 "rerun with --refresh-snapshot to accept it")
            _SNAP["sha256"][label] = digest
            data = q2_parse(data.decode()) if label == "q2" else json.loads(data)
            if label.startswith("q1/"):  # uncommitted files: keep each run's resolved config and provenance too
                for k in data:
                    if k not in RESULT_KEYS:
                        _SNAP["values"].setdefault(label, {})[json.dumps([k])] = _provenance(k, data[k])
        else:  # the snapshot must hold this very source (same commit and path), not one the config used to name
            if _OLD["source"].get(label) != cite(label):
                raise ValueError(f"{label}: {cite(label)} is unavailable and the snapshot holds "
                                 f"{_OLD['source'].get(label)}, not it")
            _SNAP["sha256"][label] = _OLD["sha256"][label]
            _SNAP["values"][label] = dict(_OLD["values"][label])  # carry everything saved, provenance included
        _SNAP["source"][label] = cite(label)
        _FILES[label] = data
    key = json.dumps(keys)
    if _FILES[label] is None:
        value = _OLD["values"][label][key]
    else:
        value = _FILES[label]
        for k in keys:
            value = value[k]
    _SNAP["values"].setdefault(label, {})[key] = value
    HASHES[label] = _SNAP["sha256"][label] + ("" if _FILES[label] is not None else " (from inputs/snapshot.json)")
    return value


def q1(name: str, *keys: str):
    return get(f"q1/{name}", *keys)


def cite(label: str) -> str:
    """The source as figures.md cites it: commit:path, or the Q1 path."""
    w = _where(label)
    if w[0] == "git":
        return f"{w[1]}:{w[2]}"
    return f'{SRC["q1"]["dir"]}/{label[3:]}' if label.startswith("q1/") else f"{HERE.parent.name}/{HERE.name}/{SRC[label]['repo_path']}"


def rec(fig: str, label: str, x: dict, source: str, key: str, value="auroc") -> dict:
    """Record one bar (or reference mark) for figures.md and return {v, lo, hi}."""
    v = x[value] if value in x else x["value"]
    lo, hi = x.get("ci95", [None, None])
    n = x.get("n", x.get("n_pairs", ""))
    ROWS.append((fig, label, v, lo, hi, n, source, key))
    return {"v": v, "lo": lo, "hi": hi}


def style(ax, ylabel: str, base: float):
    ax.set_ylabel(ylabel, color=INK, fontsize=9.5)
    ax.axhline(base, color=MUTED, lw=0.8, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


def bars(ax, xs, vals, color, label, base=0.5, width=0.36, hatch=None, fmt="{:.3f}"):
    """Bars from `base` (chance), 95% error bars, and the value printed above each bar."""
    for i, (x, b) in enumerate(zip(xs, vals)):
        ax.bar(x, b["v"] - base, width, bottom=base, color=color, edgecolor="white", linewidth=1.5,
               hatch=hatch, label=label if i == 0 else None, zorder=2)
        if b["lo"] is not None:
            ax.errorbar(x, b["v"], yerr=[[b["v"] - b["lo"]], [b["hi"] - b["v"]]], fmt="none", ecolor=INK,
                        elinewidth=1, capsize=2.5, zorder=3)
        top = b["hi"] if b["hi"] is not None else b["v"]
        ax.text(x, top + 0.008, fmt.format(b["v"]), ha="center", va="bottom", fontsize=7.5, color=INK, zorder=4)


def refmarks(ax, xs, vals, width=0.36, label="real vs real (same method)", null=False):
    """A tick across each bar: black at that method's own real-vs-real value, or dashed grey (null=True) at the
    probe's label-shuffle null 95th percentile."""
    for i, (x, b) in enumerate(zip(xs, vals)):
        if b is not None:
            ax.hlines(b["v"], x - width / 2, x + width / 2, colors=MUTED if null else INK, lw=1.6 if null else 2.2,
                      linestyles=(0, (2, 1.5)) if null else "solid", zorder=5, label=label if i == 0 else None)


def shuffle_p95(fig: str, label: str, level, position: str, variant: str) -> dict:
    """The selection-aware label-shuffle null's 95th percentile for a Q1 headline (held-out environments)."""
    name, key = SRC["q1"]["selection_null"][level], f"results.max63__{position}__{variant}.p95"
    return rec(fig, label, {"auroc": q1(name, *key.split(".")), "n": q1(name, *f"results.max63__{position}__{variant}.n".split("."))},
               f'{SRC["q1"]["dir"]}/{name}', key)


def caption(fig, text: str, bottom: float):
    fig.text(0.012, 0.012, "\n".join(textwrap.wrap(text, 175)), fontsize=8.2, color=INK, va="bottom", ha="left")
    fig.subplots_adjust(bottom=bottom)


def fig1():
    """Probe vs TF-IDF at every text level, environments never trained on, real e1 vs sim, per run."""
    tsrc = cite("tfidf_levels")
    probe, probe_rr, probe_null, tfidf, tfidf_rr = [], [], [], [], []
    for level, name in SRC["q1"]["levels"].items():
        src = f'{SRC["q1"]["dir"]}/{name}'
        key = "headline_held_out_environment.output_end.{}.all_outputs.run"
        probe.append(rec("1", f"probe, level {level}", q1(name, *key.format("real_vs_sim").split(".")), src, key.format("real_vs_sim")))
        probe_rr.append(rec("1", f"probe real vs real, level {level}", q1(name, *key.format("null_e1_vs_e2").split(".")), src, key.format("null_e1_vs_e2")))
        probe_null.append(shuffle_p95("1", f"probe label-shuffle null p95, level {level}", level, "output_end", "all_outputs"))
        k = f"level{level}_outputs_only/{{}}/all"
        tfidf.append(rec("1", f"TF-IDF, level {level}", get("tfidf_levels", "results", k.format("real_vs_sim"), "all_cells", "runs"), tsrc, f'results["{k.format("real_vs_sim")}"].all_cells.runs'))
        tfidf_rr.append(rec("1", f"TF-IDF real vs real, level {level}", get("tfidf_levels", "results", k.format("null_e1_vs_e2"), "all_cells", "runs"), tsrc, f'results["{k.format("null_e1_vs_e2")}"].all_cells.runs'))
    fig, ax = plt.subplots(figsize=(12, 6.2))
    xs = list(range(len(probe)))
    w = 0.38
    bars(ax, [x - w / 2 for x in xs], probe, BLUE, "Probe (Qwen3-32B, output_end)", width=w)
    bars(ax, [x + w / 2 for x in xs], tfidf, ORANGE, "TF-IDF (each output's own text)", width=w)
    refmarks(ax, [x - w / 2 for x in xs], probe_rr, w)
    refmarks(ax, [x + w / 2 for x in xs], tfidf_rr, w, label=None)
    refmarks(ax, [x - w / 2 for x in xs], probe_null, w, label="probe's label-shuffle null, 95th percentile", null=True)
    ax.set_xticks(xs, [f"level {x}" for x in xs])
    ax.set_xlabel("Text level  →  more of the harness format normalized away (level 0: the text as the reader sees it)", color=INK, fontsize=9.5)
    ax.set_ylim(0.45, 1.03)
    style(ax, "AUROC, real vs simulated attack runs (per run)\n↑ higher = easier to tell real from simulated; 0.5 = chance", 0.5)
    ax.set_title("Probe vs TF-IDF at every text level, on environments never trained on", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    caption(fig, "Compared: per-run AUROC for telling real attack runs (epoch 1) from simulated attack runs on the 939 paired "
            "tasks, each run scored by a model trained on the other nine environments (leave one environment out). Probe: "
            "linear probe at each tool output's last content token, headline layer chosen per fold by inner CV; a run's score "
            "is the mean over its outputs. TF-IDF: logistic regression on each output's own text at the same text level, "
            "averaged per run. Black ticks: the same method telling two real runs apart (epoch 1 vs epoch 2); TF-IDF's level-0 "
            "tick (0.68) reflects clock times, which differ between real runs. Dashed grey ticks (probe only): 95th percentile of "
            "the selection-aware label-shuffle null (100 shuffles through the whole pipeline, layer choice included); at level 7 "
            f"the probe ({probe[-1]['v']:.3f}) is above it ({probe_null[-1]['v']:.3f}). At level 0 the probe's readout token sometimes "
            f"ends in a trailing newline, about twice as often in simulated runs ({newline_share('sim-attack'):.1%} of output_end tokens "
            f"vs {newline_share('real-attack-e1'):.1%} in real runs; not measured at other levels; see figures.md); a check splitting by "
            "trailing newline is pending. Error bars: 95% intervals from resampling main "
            "tasks. Bars start at chance (0.5). Sources and exact values: figures.md.", 0.27)
    fig.savefig(OUT / "fig1_probe_vs_tfidf_by_level.png", dpi=200)
    plt.close(fig)


def fig2():
    """Every baseline on original text, main-task folds, real e1 vs sim attack."""
    name, bname = SRC["q1"]["original_text"], SRC["q1"]["original_baselines"]
    src, bsrc = f'{SRC["q1"]["dir"]}/{name}', f'{SRC["q1"]["dir"]}/{bname}'
    L = SRC["q1"]["random_layer"]
    tv = SRC["tfidf_original"]["variant"]
    t = lambda cmp_: get("tfidf_original", "baselines", tv, "results", cmp_, "all_cells", "runs")  # noqa: E731
    tsrc, jsrc = cite("tfidf_original"), cite("judge")
    j = get("judge", "judges", SRC["judge"]["judge"], "our_cells", SRC["judge"]["slice"])
    g = lambda f, key: q1(f, *key.split("."))  # noqa: E731
    layers = q1(name, "config", "layers")
    nulls = [rec("2", f"probe fixed-layer label-shuffle null p95, layer {lay}", {"auroc": g(name, k := f"by_layer.{lay}.results.output_end.real_vs_sim.all_outputs.run.label_shuffle_null.p95"), "n": 100}, src, k)["v"]
             for lay in layers]
    items = [  # (label, bar, real-vs-real, colour, hatch)
        ("Probe\n(headline layer)", rec("2", "probe", g(name, k := "headline.output_end.real_vs_sim.all_outputs.run"), src, k),
         rec("2", "probe real vs real", g(name, k := "headline.output_end.null_e1_vs_e2.all_outputs.run"), src, k), BLUE, None),
        ("TF-IDF\n(each output's text)", rec("2", "TF-IDF", t("real_vs_sim/all"), tsrc, 'baselines.tfidf_cleaned.results["real_vs_sim/all"].all_cells.runs'),
         rec("2", "TF-IDF real vs real", t("null_e1_vs_e2/all"), tsrc, 'baselines.tfidf_cleaned.results["null_e1_vs_e2/all"].all_cells.runs'), ORANGE, None),
        ("LLM judge\n(pairwise accuracy)", rec("2", "LLM judge (pairwise accuracy)", j, jsrc, "judges.gpt-5.6-sol_medium.our_cells.no_run_over_32k", value="accuracy"),
         None, AQUA, "//"),
        (f"Best of 1,000\nrandom directions (L{L})", rec("2", f"random directions, layer {L}", g(name, k := f"by_layer.{L}.results.output_end.real_vs_sim.all_outputs.best_random_direction.run"), src, k),
         rec("2", f"random directions real vs real, layer {L}", g(name, k := f"by_layer.{L}.results.output_end.null_e1_vs_e2.all_outputs.best_random_direction.run"), src, k), BLUE, None),
        ("Output length", rec("2", "output length", g(bname, k := "results.real_vs_sim.all_outputs.output_length.run"), bsrc, k),
         rec("2", "output length real vs real", g(bname, k := "results.null_e1_vs_e2.all_outputs.output_length.run"), bsrc, k), ORANGE, None),
        ("Tool-output\nperplexity", rec("2", "perplexity", g(bname, k := "results.real_vs_sim.all_outputs.perplexity.run"), bsrc, k),
         rec("2", "perplexity real vs real", g(bname, k := "results.null_e1_vs_e2.all_outputs.perplexity.run"), bsrc, k), ORANGE, None),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    w = 0.6
    seen = set()
    for x, (label, bar, rr, color, hatch) in enumerate(items):
        family = {BLUE: "Activation-based (Qwen3-32B)", ORANGE: "Text statistics", AQUA: "LLM judge (gpt-5.6-sol)"}[color]
        bars(ax, [x], [bar], color, None if family in seen else family, width=w, hatch=hatch)
        seen.add(family)
        refmarks(ax, [x], [rr], w, label="real vs real (same method)" if x == 0 else None)
    ax.set_xticks(range(len(items)), [i[0] for i in items])
    ax.set_ylim(0.45, 1.03)
    style(ax, "AUROC, real vs simulated attack runs (per run)\n↑ higher = easier to tell real from simulated; 0.5 = chance", 0.5)
    ax.set_title("Every baseline on the original text (main-task folds)", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    caption(fig, "Compared: real attack runs (epoch 1) vs simulated attack runs on the 939 paired tasks, original text (as the "
            "reader sees it, no normalization), cross-validated over the shared main-task folds. Probe: inner-CV headline "
            "layer, 8-layer analysis. TF-IDF: each tool output's own text, averaged per run. Random directions: the best of "
            "1,000 picked on the training folds, at layer 48. Length and perplexity: per-output values averaged per run. LLM "
            "judge: the LinuxArena paper's stored verdicts (gpt-5.6-sol, medium effort), shown a real and a simulated run of the "
            "same task and asked which is real; its metric is pairwise accuracy (590 pairs with no run over 32k, whole "
            "transcripts), and chance is 0.5 for both metrics; it has no real-vs-real counterpart. Black ticks: the same method "
            "telling two real runs apart; TF-IDF's (0.71) reflects clock times. No selection-aware null (repeating the headline's layer "
            "choice) was run for this 8-layer analysis; the fixed-layer label-shuffle nulls (100 shuffles at each layer) have 95th "
            f"percentiles of at most {max(nulls):.3f}, which bound only a single-layer probe. Error bars: 95% intervals from resampling main "
            "tasks. The level-7 comparison is Figure 1's rightmost pair. Sources: figures.md.", 0.265)
    fig.savefig(OUT / "fig2_baselines_original_text.png", dpi=200)
    plt.close(fig)


def q2_parse(text: str) -> dict:
    """{readout: {train→test: {value, ci95, n}}} from the Q2 table's '2×2, all eligible calls' section."""
    block = text.split("## 2×2, all eligible calls")[1].split("\n## ")[0]
    out = {}
    for line in block.splitlines():
        m = re.match(r"\| (call_end|pre_call|prev_content) \| (real → real|real → sim) \| ([\d.]+) \[([\d.]+), ([\d.]+)\] \|.*\| ([\d /]+) \|$", line)
        if m:
            out.setdefault(m[1], {})[m[2]] = {"value": float(m[3]), "ci95": [float(m[4]), float(m[5])], "n": m[6].strip()}
    assert sum(map(len, out.values())) == 6, f"expected 6 rows in the Q2 table, found {sum(map(len, out.values()))}"
    return out


def fig3():
    """Agent calls vs tool outputs: Q1 at level 7, Q2 recall, Q3 catch rate."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.8), gridspec_kw={"width_ratios": [1.15, 1.1, 1]})
    # (a) Q1, level 7, environments never trained on
    ce_name, oe_name = SRC["q1"]["callend_level7"], SRC["q1"]["levels"][7]
    cs, os_ = f'{SRC["q1"]["dir"]}/{ce_name}', f'{SRC["q1"]["dir"]}/{oe_name}'
    k = "headline_held_out_environment.{}.{}.{}.run"
    a = [("Agent calls\n(call_end, content\ntokens, no editor)", ce_name, cs, "call_end", "no_editor_outputs", NEUTRAL),
         ("Tool outputs\n(output_end,\nno editor)", oe_name, os_, "output_end", "no_editor_outputs", NEUTRAL),
         ("Tool outputs\n(output_end,\nall)", oe_name, os_, "output_end", "all_outputs", NEUTRAL)]
    ax = axes[0]
    for x, (label, f, s, pos, var, color) in enumerate(a):
        b = rec("3a", f"Q1 L7 {pos} {var}", q1(f, *(kk := k.format(pos, "real_vs_sim", var)).split(".")), s, kk)
        rr = rec("3a", f"Q1 L7 {pos} {var} real vs real", q1(f, *(kk := k.format(pos, "null_e1_vs_e2", var)).split(".")), s, kk)
        bars(ax, [x], [b], color, None, width=0.55, hatch="//" if var == "all_outputs" else None)
        refmarks(ax, [x], [rr], 0.55, label="real vs real" if x == 0 else None)
        if pos == "output_end":
            refmarks(ax, [x], [shuffle_p95("3a", f"Q1 L7 {pos} {var} label-shuffle null p95", 7, pos, var)], 0.55,
                     label="label-shuffle null, 95th pct" if x == 1 else None, null=True)
        else:  # the corrected call_end rerun has no matching selection-aware null run: say so on the bar
            ax.text(x, 0.455, "no shuffle-null\nrun yet", ha="center", va="bottom", fontsize=7.5, color=MUTED, zorder=6)
    ax.set_xticks(range(3), [i[0] for i in a], fontsize=8.5)
    ax.set_ylim(0.45, 0.62)
    style(ax, "AUROC, real vs simulated attack runs (per run)\n↑ higher = easier to tell real from simulated; 0.5 = chance", 0.5)
    ax.set_title("Q1: telling real from simulated at level 7\n(environments never trained on)", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    # (b) Q2 recall at 1% false positives
    qsrc = cite("q2")
    ax = axes[1]
    w = 0.36
    for j, (split, color) in enumerate([("real → real", BLUE), ("real → sim", ORANGE)]):
        vals = [rec("3b", f"Q2 {r} {split}", get("q2", r, split), qsrc, f"2×2, all eligible calls: {r} | {split} | recall") for r in ("call_end", "pre_call", "prev_content")]
        bars(ax, [x + (j - 0.5) * w for x in range(3)], vals, color, f"trained on real, tested on {split.split()[-1]}", base=0, width=w)
    ax.set_xticks(range(3), ["Command\nprobe\n(call_end)", "Before-\ncommand\n(pre_call)", "Its content-\ntoken control\n(prev_content)"], fontsize=8)
    ax.set_ylim(0, 1.05)
    style(ax, "Recall of impactful calls at 1% false positives\n↑ higher = catches more impactful calls", 0)
    ax.set_title("Q2: impactful-call probes\n(trained on real runs)", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    # (c) Q3 catch rate at 1% false alarms
    msrc = cite("q3")
    ax = axes[2]
    for j, (arm, color, label) in enumerate([("e1", BLUE, "real attacks (epoch 1)"), ("sim", ORANGE, "simulated attacks")]):
        vals, refs = [], []
        for readout in ("probe_call_end_content_mean", "probe_output_end_mean"):
            sl = ("readouts", readout, "task", "slices", "all_runs")
            c, n = get("q3", *sl, "catch_rate", "fpr_0.01"), get("q3", *sl, "n_runs")
            key = f"readouts.{readout}.task.slices.all_runs.catch_rate.fpr_0.01.{{}}"
            vals.append(rec("3c", f"Q3 {readout} {arm}", c[arm] | {"n": f'{n[arm]} attack / {n["honest"]} honest runs'}, msrc, key.format(arm)))
            if j == 0:
                refs.append(rec("3c", f"Q3 {readout} e2 (real vs real)", c["e2"] | {"n": f'{n["e2"]} attack / {n["honest"]} honest runs'}, msrc, key.format("e2")))
        xs = [x + (j - 0.5) * w for x in range(2)]
        bars(ax, xs, vals, color, label, base=0, width=w)
        if j == 0:
            refmarks(ax, xs, refs, w, label="second real sample (epoch 2)")
    ax.set_xticks(range(2), ["Agent calls\n(call_end, content\ntokens; 681 / 911 runs)", "Tool outputs\n(output_end;\n939 / 990 runs)"], fontsize=8)
    ax.set_ylim(0, 0.5)
    style(ax, "Attack catch rate at 1% false alarms on real honest runs\n↑ higher = catches more attacks", 0)
    ax.set_title("Q3: attack detector\n(trained on real runs)", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Agent calls vs tool outputs", x=0.012, ha="left", fontsize=12, color=INK)
    caption(fig, "Compared. (a) Q1, text level 7, environments never trained on: per-run AUROC for real attack (epoch 1) vs simulated "
            "attack, probe at the last content token of each agent call (call_end, the rerun that skips the <n> placeholder slot; "
            "no editor calls; 826 runs) or of each tool output (output_end; no editor outputs, or all outputs, hatched); black ticks: "
            "epoch 1 vs epoch 2; dashed grey ticks: the selection-aware label-shuffle null's 95th percentile. No null was run for the corrected call_end rerun, so its bar makes no claim of signal: its interval includes 0.5 and it is below both output_end nulls. (b) Q2: recall of impactful calls at a threshold frozen at 1% false positives on training-domain "
            "validation, probes trained on real runs and tested on real or simulated runs; command probe = call_end, before-command "
            "probe = pre_call (the <tool_call> token), shown beside prev_content, its content-token control: the last content token before the call, which sees the same context without template tokens. (c) Q3: share of attack runs flagged by an attack-vs-honest probe trained on "
            "real runs, threshold frozen at 1% false alarms on real honest runs; main-task folds; the call readout covers only runs with a content-token call (681 attack, 911 honest), the output readout all runs (939, 990), shown as attack / honest runs under each group; black ticks: a second real sample "
            "(epoch 2). Error bars: 95% intervals from resampling main tasks. Panels use different metrics, so compare within a "
            "panel. Sources: figures.md.", 0.3)
    fig.savefig(OUT / "fig3_calls_vs_outputs.png", dpi=200)
    plt.close(fig)


def fig4():
    """Today's vs September's simulator: TF-IDF on 20 cells, attack and honest."""
    groups, sept, today, rr = [], [], [], []
    for cohort, real, other in [("attack", "real-attack-e1", "epoch2_vs_epoch1_scored_by_this_model"),
                                ("honest", "real-honest", "real-honest-2_vs_real-honest_scored_by_this_model")]:
        label = f"pilots:{cohort}"
        psrc = cite(label)
        cells, today_key = f"cells_with_sim-{cohort}-today", f"sim-{cohort}-today_vs_{real}_scored_by_this_model"
        rr_cells = "all_cells" if cohort == "attack" else "cells_with_real-honest-2"
        for lvl, var, name in [("level0", "all", "level 0"), ("level0", "no_editor", "level 0,\nno editor outputs"), ("level7", "all", "level 7")]:
            res = lambda *k: get(label, "baselines", lvl, "results", f"real_vs_sim/{var}", *k)  # noqa: E731
            base = f'baselines.{lvl}.results["real_vs_sim/{var}"]'
            groups.append(f"{cohort.capitalize()}\n{name}")
            sept.append(rec("4", f"{cohort} {lvl}/{var} September simulator", res(cells, "runs"), psrc, f"{base}.{cells}.runs"))
            today.append(rec("4", f"{cohort} {lvl}/{var} today's simulator", res(today_key, cells, "runs"), psrc, f"{base}.{today_key}.{cells}.runs"))
            rr.append(rec("4", f"{cohort} {lvl}/{var} real vs real", res(other, rr_cells, "runs"), psrc, f"{base}.{other}.{rr_cells}.runs"))
    fig, ax = plt.subplots(figsize=(12, 6.2))
    w = 0.38
    xs = list(range(len(groups)))
    bars(ax, [x - w / 2 for x in xs], sept, ORANGE, "September simulator (the cohort we study)", width=w)
    bars(ax, [x + w / 2 for x in xs], today, AQUA, "Today's simulator", width=w)
    refmarks(ax, [x - w / 2 for x in xs], rr, w, label="real vs real (all cells)")
    refmarks(ax, [x + w / 2 for x in xs], rr, w, label=None)
    ax.axvline(2.5, color=GRID, lw=1.2)
    ax.set_xticks(xs, groups, fontsize=9)
    ax.set_ylim(0.45, 1.17)  # headroom for the legend above the near-1.0 bars
    style(ax, "TF-IDF AUROC, simulated vs real runs (per run)\n↓ lower = harder to tell from real (more realistic); 0.5 = indistinguishable", 0.5)
    ax.set_title("Today's simulator vs September's, on the same 20 tasks (TF-IDF)", loc="left", fontsize=12, color=INK)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3)
    caption(fig, "Compared: per-run AUROC of a TF-IDF classifier (each tool output's own text, averaged per run) telling simulated runs "
            "from real runs on the 20 tasks that have a run from today's simulator (40 runs per bar), for attack runs (vs real "
            "epoch 1) and honest runs (vs real honest), at text level 0 (all outputs, or no editor outputs) and level 7. The "
            "classifier is trained on real vs September-simulator runs over the main-task folds; today's runs are scored by that "
            "same model, so a lower bar can mean September's tells don't transfer, not only that today's simulator is more "
            "realistic. Black ticks: the same classifier telling two real samples apart (attack: epoch 2 vs 1, 939 tasks; honest: "
            "a second real honest sample vs the first, 258 tasks). Error bars: 95% intervals from resampling main tasks (wide: 20 "
            "tasks). Sources: figures.md.", 0.21)
    fig.savefig(OUT / "fig4_today_vs_september_simulator.png", dpi=200)
    plt.close(fig)


INTRO = ("Generated by `plot_james_figures.py` from the files in `config/sources.yaml`; nothing is recomputed. Each row "
         "is one bar or reference mark: value, 95% interval, n (runs, or pairs for the judge, or impactful / read-only calls "
         "for Q2), source file (`commit:path` for committed files) and the key read.")
PROVENANCE = ("**Inputs.** Every value used is saved, with the sha256 of the file it came from (and, for the Q1 files, each "
              "run's resolved config, manifest and null scheme), in `inputs/snapshot.json`; the "
              "script falls back on it for any source it cannot read (the Q1 files below, which were not committed on any "
              "branch on 2026-10-06 and were read from the main session's checkout, or a pinned commit missing from a "
              "clone), and stops if a source file's sha256 differs from the snapshot's unless run with `--refresh-snapshot`.")
ARM_NAMES = {"real-attack-e1": "real attack epoch 1", "sim-attack": "simulated attack", "real-attack-e2": "real attack epoch 2"}


def newline_share(arm: str, field: str = "ending_in_newline") -> float:
    return get("newlines", "counts", arm, field) / get("newlines", "counts", arm, "output_end_tokens")


def newline_note() -> str:
    """Figure 1 caption's newline numbers, from analyze_readout_newlines.py's result file."""
    shares = "; ".join(f"{newline_share(a):.1%} of {get('newlines', 'counts', a, 'output_end_tokens'):,} in {name} "
                       f"({newline_share(a, 'newline_only'):.1%} newline-only)" for a, name in ARM_NAMES.items())
    return (f"`analyze_readout_newlines.py` (result `{cite('newlines')}`), on the transcript viewer's cache (cache version "
            f"{get('newlines', 'cache', 'version')}), level-0 text, the paired tasks' valid runs, readout positions inside the "
            "32k window: the share of `output_end` tokens (from `probe_shift.reader.segment_end_positions`) whose decoded "
            f"text ends in a newline is {shares}. Not measured at other levels.")
TITLES = {"1": "Figure 1: probe vs TF-IDF at every text level (environments never trained on)",
          "2": "Figure 2: every baseline on original text (main-task folds)",
          "3a": "Figure 3a: Q1 at level 7, calls vs outputs", "3b": "Figure 3b: Q2 recall at 1% false positives",
          "3c": "Figure 3c: Q3 catch rate at 1% false alarms",
          "4": "Figure 4: today's vs September's simulator (TF-IDF, 20 tasks)"}


def write_md():
    lines = ["# Figures for the 2026-10-07 meeting with James", "", INTRO, "", PROVENANCE, ""]
    for fig, title in TITLES.items():
        lines += [f"## {title}", "", "| bar | value | 95% interval | n | source | key |", "|---|---|---|---|---|---|"]
        for f, label, v, lo, hi, n, source, key in ROWS:
            if f == fig:
                ci = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "–"
                lines.append(f"| {label} | {v:.3f} | {ci} | {n} | `{source}` | `{key.replace('|', chr(92) + '|')}` |")
        lines.append("")
    lines += ["## Note: trailing newlines in the readout token (Figure 1 caption)", "", newline_note(), ""]
    lines += ["## Source hashes", "", "| source | sha256 of the file read |", "|---|---|"]
    lines += [f"| `{cite(label)}` | `{h}` |" for label, h in sorted(HASHES.items())]
    (HERE / "figures.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in (fig1, fig2, fig3, fig4):
        f()
    write_md()  # reads values too (the newline note), so the snapshot is saved after it
    SNAPSHOT.parent.mkdir(exist_ok=True)  # every value used, so any checkout can regenerate the figures
    SNAPSHOT.write_text(json.dumps(_SNAP, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(ROWS)} bars and marks to figures.md and 4 PNGs in {OUT}")
