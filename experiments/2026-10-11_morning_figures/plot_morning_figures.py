"""Morning figures: one question per figure, every number read from git at a named commit.

  uv run --no-project --with matplotlib --with numpy --with pyyaml --with scikit-learn \
      python experiments/2026-10-11_morning_figures/plot_morning_figures.py --out <dir> [--only Q2.2 Q2.4]

figures_q*.yaml list each figure's panels. A panel names its source (git ref and path) and reference lines. Each bar
names a key into that JSON file (slash path, list indices as numbers) or, for markdown, a quote that must appear in the
file, plus the value the spec's author read. The script resolves the ref, reads the file with `git show`, and stops on
any bar whose value differs from the spec, so a misread or a changed file cannot reach a figure. A source not yet at
its ref draws as an empty "pending" slot. Each figure's footer lists its sources as file@commit (the last commit that
touched the file); captions.json holds each figure's caption, full source paths and pending items.
Run `git fetch` first; regenerate after a result lands by updating its entry and rerunning.
"""

import argparse
import json
import re
import subprocess
import textwrap
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
COLORS = {"real": "#2f6db5", "real2": "#8fb3e0", "sim": "#e8833a", "gap": "#3d4450", "control": "#a9b0b8",
          "other": "#7a5fb0"}
FMT = {"percent": "{:.1f}%", "points": "{:+.1f}", "AUROC": "{:.3f}", "AUROC diff": "{:+.3f}", "SD": "{:+.2f}", "ratio": "{:.2f}×", "ratio3": "{:.3f}×", "diff3": "{:+.3f}",
       "logits": "{:+.2f}", "steps": "{:+.1f}", "tokens": "{:+,.0f}"}
GRAY = "#5a6371"
_cache = {}


def git(*args, check=True):
    r = subprocess.run(["git", "-C", str(ROOT), *args], check=False, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r


def read(ref, path):
    """(file@commit, parsed body) at ref; a source that cannot be read stops the run (only `pending` entries wait)."""
    if (ref, path) not in _cache:
        commit = git("rev-parse", f"{ref}^{{commit}}").stdout.strip()  # one immutable commit for the read and the stamp
        r = git("show", f"{commit}:{path}")
        sha = git("log", "-1", "--format=%h", "--abbrev=8", commit, "--", path).stdout.strip()
        _cache[ref, path] = (f"{path}@{sha}", json.loads(r.stdout) if path.endswith(".json") else r.stdout)
    return _cache[ref, path]


def dig(obj, key):
    for k in key if isinstance(key, list) else str(key).split("/"):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    return obj


def close(got, want, floor=2):
    """True when got rounds to want at want's precision, never coarser than `floor` decimals (the printed precision)."""
    decimals = max(floor, -Decimal(str(float(want))).normalize().as_tuple().exponent)
    return abs(got - want) <= 0.51 * 10 ** -decimals + 1e-9


def values(panel, bar, errors):
    """(value, lo, hi, source) for a bar, or (None, None, None, source) if its file is pending."""
    if "pending" in bar:  # a result expected overnight whose file name is not known yet
        return None, None, None, f"pending: {bar['pending']}"
    src, body = read(bar.get("ref", panel.get("ref", "origin/main")), bar.get("path", panel.get("path")))
    if body is None:
        return None, None, None, src
    s = bar.get("scale", 1)
    fl = 3 if panel.get("unit") in ("AUROC", "AUROC diff", "ratio3", "diff3") else 2  # never looser than what the figure prints
    if bar.get("key") is None:  # markdown: the quote must be in the file and hold each number the spec states
        if bar["quote"] not in body:
            errors.append(f"{bar['label']}: quote {bar['quote']!r} not in {src}")
        if "count" in bar:  # the file states k/n; the bar is k/n scaled
            k, n = bar["count"]
            if not re.search(rf"(?<![\d.,]){k}/{n}(?!\d|[.,]\d)", bar["quote"]):
                errors.append(f"{bar['label']}: {k}/{n} not in its quote")
            bar = {**bar, "value": round(s * k / n, 4)}
        quote = bar["quote"].replace("\u2212", "-")
        for name, raw in zip(("value", "lo", "hi"), bar.get("raw", [])):  # the exact source string behind each number
            if raw is None or bar.get(name) is None:
                continue
            if not re.search(rf"(?<![\d.,eE+-]){re.escape(raw)}(?!\d|[.,]\d|[eE][-+]?\d)", quote) or not close(float(raw) * s, bar[name], fl):
                errors.append(f"{bar['label']}: {name} {bar[name]} is not {raw!r} x {s} in the quote from {src}")
        missing = [n for n, r in zip(("value", "lo", "hi"), bar.get("raw", []) + [None] * 3) if bar.get(n) is not None
                   and r is None and not (n == "value" and "count" in bar)]
        if missing:
            errors.append(f"{bar['label']}: no raw source string for {missing}")
        return bar["value"], bar.get("lo"), bar.get("hi"), src
    got = [dig(body, bar[k]) * s if bar.get(k) is not None else None for k in ("key", "lo_key", "hi_key")]
    if got[1] is not None and got[2] is not None and got[1] > got[2]:  # a negative scale flips the sign
        got[1], got[2] = got[2], got[1]
    for name, g in zip(("value", "lo", "hi"), got):
        if g is not None and bar.get(name) is not None and not close(g, bar[name], fl):
            errors.append(f"{bar['label']}: {name} reads {g:.6g} from {src}, spec says {bar[name]}")
    v, lo, hi = got
    return v, lo, hi, src


def sourced(item, ys, errors, sources):
    """A margin or null taken from a committed file: a JSON key, or a raw number inside a quote, must give each y."""
    so = item["source"]
    src, body = read(so["ref"], so["path"])
    sources.add(src)
    if body is None:
        errors.append(f"{item['label']}: {src}")
        return
    if "key" in so and isinstance(dig(body, so["key"]), list):  # a null band [lower, upper] must match element by element
        band = [x * so.get("scale", 1) for x in dig(body, so["key"])]
        if len(band) != len(ys) or not all(close(a, b) for a, b in zip(band, ys)):
            errors.append(f"{item['label']}: band not backed by its source in {src}")
        return
    if "key" in so:  # a JSON value; pm marks a symmetric margin
        v, both, ok = dig(body, so["key"]) * so.get("scale", 1), so.get("pm", False), True
    else:  # the quote must be in the file; both signs only for a ± margin
        quote, text = so["quote"].replace("\u2212", "-"), body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        ok = so["quote"] in text and re.search(rf"(?<![\d.,eE+-]){re.escape(so['raw'])}(?!\d|[.,]\d|[eE][-+]?\d)", quote)
        v, both = float(so["raw"]) * so.get("scale", 1), so.get("pm", False) or "\u00b1" + so["raw"] in so["quote"]
    if not ok or not all(close(v, y) or (both and close(abs(v), abs(y))) for y in ys):
        errors.append(f"{item['label']}: value not backed by its source in {src}")


def check_text(panel, bar, errors):
    """A printed sample size needs a quote from its source; numbers in a line's label must be the line's own values."""
    if bar.get("n"):
        bar = {**bar, "n": str(bar["n"])}  # YAML may read a bare count as an integer
        _, body = read(bar.get("ref", panel.get("ref", "origin/main")), bar.get("path", panel.get("path")))
        quote = bar.get("n_quote") or ""
        if not re.search(rf"(?<![\d.,]){re.escape(bar['n'])}(?!\d|[.,]\d|[eE][-+]?\d)", quote) or body is None or quote not in json.dumps(body, ensure_ascii=False) + str(body):
            errors.append(f"{bar['label']}: sample size {bar['n']!r} has no quote in its source")


def draw_bars(ax, panel, errors, sources, pending):
    fmt = FMT[panel["unit"]]
    for line in panel.get("lines", []):
        if "source" in line:
            sourced(line, line.get("band", [line.get("y", 0)]), errors, sources)
        own = [float(x) for x in line.get("band", [line.get("y", 0)])]
        signed = {"+": [1], "-": [-1], "\u2212": [-1], "\u00b1": [1, -1], "": [1, -1]}  # an unsigned number may be either side
        if any(not any(close(k * float(x), o) for o in own for k in signed[sg])
               for sg, x in re.findall(r"([+\-\u2212\u00b1]?)(\d+\.?\d*)", line["label"])):
            errors.append(f"line {line['label']!r}: its numbers are not its own values {sorted(own)}")
        if "band" in line:
            ax.axhspan(*line["band"], color="#dfeee0", zorder=0)
            ax.text(1.0, line["band"][1], f" {line['label']}", transform=ax.get_yaxis_transform(), va="bottom",
                    fontsize=8, color=GRAY)
        else:
            ax.axhline(line["y"], ls="--", lw=1, color=GRAY, zorder=1)
            ax.text(1.0, line["y"], f" {line['label']}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=8, color=GRAY)
    if panel["unit"] in ("points", "SD", "logits"):
        ax.axhline(0, lw=0.8, color="black")
    labels = []
    for bar in panel["bars"]:
        v, lo, hi, src = values(panel, bar, errors)
        if v is None:  # recorded in captions.json, never drawn: no status markers on figures
            pending.append(f"{bar['label']}: {src}")
            continue
        check_text(panel, bar, errors)
        i = len(labels)
        labels.append(bar["label"] + (f"\n({bar['n']})" if bar.get("n") else ""))
        sources.add(src)
        ax.bar(i, v, width=0.6, color=COLORS[bar["role"]], zorder=2)
        if "null" in bar:  # this bar's own shuffled-label null (a 95th percentile, or a 95% band), dotted across the bar
            ys = bar["null"].get("band", [bar["null"].get("y")])
            sourced({"label": f"{bar['label']} null", **bar["null"]}, ys, errors, sources)
            ax.hlines(ys, i - 0.36, i + 0.36, colors="black", linestyles=":", lw=1.6, zorder=4)
        top = hi if hi is not None else v
        if lo is not None and hi is not None:
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color="black", capsize=5, lw=1.4, zorder=3)
        tight = lo is not None and hi - lo < 3  # a sub-point gap with a narrow interval keeps two decimals
        f = fmt.replace(".1f", ".2f") if panel["unit"] == "points" and abs(v) < 1 and tight else fmt
        num = lambda x, f=f: f.format(x).replace("-", "\u2212")
        y, up = (top, True) if v >= 0 else ((lo if lo is not None else v), False)
        lines = [(num(v), 12, "bold")] + ([(f"[{num(lo)}, {num(hi)}]", 8.5, "normal")] if lo is not None else [])
        for k, (t, size, weight) in enumerate(lines if not up else lines[::-1]):
            off = (4 + 13 * k) * (1 if up else -1)
            ax.annotate(t, (i, y), xytext=(0, off), textcoords="offset points", ha="center",
                        va="bottom" if up else "top", fontsize=size, fontweight=weight)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.margins(y=0.25)
    if "ylim" in panel:
        ax.set_ylim(*panel["ylim"])
    elif panel["unit"] == "percent":
        ax.set_ylim(0, 128)
        ax.set_yticks(range(0, 101, 20))


def draw_roc(ax, panel, errors, sources, pending):
    import numpy as np
    from sklearn.metrics import roc_curve

    for curve in panel["curves"]:
        if "pending" in curve:
            pending.append(f"{curve['label']}: pending: {curve['pending']}")
            continue
        src, body = read(curve.get("ref", panel.get("ref", "origin/main")), curve.get("path", panel.get("path")))
        if body is None:
            pending.append(f"{curve['label']}: {src}")
            continue
        if "fpr_key" in curve:
            fpr, tpr = np.array(dig(body, curve["fpr_key"])), np.array(dig(body, curve["tpr_key"]))
        else:
            fpr, tpr, _ = roc_curve(dig(body, curve["labels_key"]), dig(body, curve["scores_key"]))
        sources.add(src)
        ax.plot(np.clip(fpr, 1e-4, 1), tpr, color=COLORS[curve["role"]], lw=2, label=curve["label"])
    grid = np.logspace(-4, 0, 200)  # y = x, sampled so it stays the diagonal on a log axis
    ax.plot(grid, grid, ls=":", color=GRAY, lw=1, label="chance")
    for x in panel.get("fpr_lines", [0.01]):
        ax.axvline(x, ls="--", lw=1, color=GRAY)
        ax.text(x, 0.02, f" {x:.0%}", fontsize=8, color=GRAY)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-positive rate (log scale; left is stricter)", fontsize=9)
    ax.set_ylabel(panel.get("ylabel", "True-positive rate"), fontsize=9)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, frameon=False, loc="lower right")


def drawable(panel):
    return any("pending" not in b for b in panel.get("bars") or panel.get("curves") or [])


def make(fig_spec, out):
    panels = [p for p in fig_spec["panels"] if drawable(p)]
    skipped = [f"panel {p['title']}: nothing committed yet" for p in fig_spec["panels"] if not drawable(p)]
    if not panels:  # nothing committed for this figure yet: no image, only its pending entry
        return {"fig": fig_spec["fig"], "png": None, "caption": fig_spec["caption"], "sources": [], "pending": skipped}
    ncols = fig_spec.get("ncols", len(panels))
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(7.5, 4.6 * ncols + 0.8), 4.6 * nrows), squeeze=False)
    errors, sources, pending = [], set(), []
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, panel in zip(axes.flat, panels):
        (draw_roc if panel["kind"] == "roc" else draw_bars)(ax, panel, errors, sources, pending)
        ax.set_title(panel["title"], fontsize=10)
        if panel.get("ylabel"):
            ax.set_ylabel(panel["ylabel"], fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if errors:
        raise SystemExit(f"{fig_spec['fig']}: values differ from the spec:\n  " + "\n  ".join(errors))
    fig.suptitle(textwrap.fill(fig_spec["title"], int(9.5 * fig.get_figwidth())), fontsize=12, fontweight="bold")
    shown = sorted(s.split("/")[-1] for s in sources)
    fig.text(0.01, 0.01, "Black lines: 95% intervals. Sources: " + "; ".join(shown), fontsize=6.5, color=GRAY, wrap=True)
    fig.tight_layout(rect=(0, 0.06 / nrows, 1, 1 - 0.05 / nrows))
    png = out / f"{fig_spec['fig'].replace('.', '_')}_{fig_spec['slug']}.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)
    return {"fig": fig_spec["fig"], "png": str(png), "caption": fig_spec["caption"], "sources": sorted(sources),
            "pending": pending + skipped, "notes": fig_spec.get("notes", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--spec", nargs="*", type=Path, default=sorted(HERE.glob("figures_*.yaml")))
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figs = [f for spec in args.spec for f in yaml.safe_load(spec.read_text())["figures"] if not args.only or f["fig"] in args.only]
    made = [make(f, args.out) for f in figs]
    cap = args.out / "captions.json"  # merged by figure, so --only keeps the other figures' entries
    old = {m["fig"]: m for m in json.loads(cap.read_text())} if cap.exists() else {}
    cap.write_text(json.dumps(list({**old, **{m["fig"]: m for m in made}}.values()), indent=1, ensure_ascii=False))
    for m in made:
        print(f"{m['png'] or m['fig'] + ': not drawn yet'}\n  {m['caption']}" + (f"\n  pending: {m['pending']}" if m["pending"] else ""))


if __name__ == "__main__":
    main()
