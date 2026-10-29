"""Aswin's key figure (0d, 10-13): real attack catch at 1% (x) vs overstatement (y), one point per recipe x readout,
Qwen3-32B L48 and L63 panels. Values are read by key from the Check 1 result (why_q1_readouts_2026-10-13_174537.json,
aswin/why-steps-q1 @ e8c1d18b); the drawing runs on Modal CPU and only the PNG comes back.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run \\
      experiments/2026-10-13_why_steps/plot_key_figure.py --q1 <path to why_q1_readouts_*.json>
"""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-whysteps_plot_kill-2026-11-13"
RECIPES = {"a": "real-trained (a)", "b": "fake/fake (b)", "c": "mixed (c)"}
COLORS = {"a": "#0072B2", "b": "#E69F00", "c": "#D55E00"}  # Okabe-Ito, colour-blind safe
READOUTS = {"call_end": ("command (call_end)", "o"), "output_end": ("reply, last token", "s"), "averaged": ("reply, averaged", "^")}

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("matplotlib==3.10.6", "numpy==2.3.3", "adjustText==1.3.0")


@app.function(image=image, cpu=2, memory=4096, timeout=600, retries=0)
def draw(points: list[dict], title: str, footer: str) -> bytes:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), sharex=True, sharey=True)
    for ax, layer in zip(axes, ("48", "63")):
        ax.axhline(0, color="0.6", lw=1, zorder=0)
        texts, xs, ys = [], [], []
        for p in [p for p in points if p["layer"] == layer]:
            x, y = p["x"], p["y"]
            hl = p["highlight"]
            ax.errorbar(x, y, xerr=[[x - p["x_ci"][0]], [p["x_ci"][1] - x]], yerr=[[y - p["y_ci"][0]], [p["y_ci"][1] - y]],
                        fmt="none", ecolor=COLORS[p["recipe"]], alpha=0.45, lw=1.2, zorder=1)
            hollow = p.get("hollow", False)  # fix939 averaged, beside the full-repair averaged points
            ax.scatter(x, y, s=260 if hl else 150, marker=READOUTS[p["readout"]][1], linewidth=2.4 if hl or hollow else 1,
                       facecolor="none" if hollow else COLORS[p["recipe"]],
                       edgecolor=COLORS[p["recipe"]] if hollow else ("black" if hl else "white"), zorder=3)
            if hollow:  # unlabelled: their values are in the sidecar (and in the v2 figure)
                continue
            label = (f"{hl}: " if hl else "") + f"{x:.2f}, {y:+.2f}" + (f" ({p['repair']})" if p["repair"] else "")
            texts.append(ax.text(x, y, label, fontsize=10, fontweight="bold" if hl else "normal", color="0.1", zorder=4))
            xs.append(x)
            ys.append(y)
        ax.set_xlim(0, 0.9)
        ax.set_ylim(-0.12, 0.72)
        adjust_text(texts, x=xs, y=ys, ax=ax, expand=(1.3, 1.6), arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
        ax.set_title(f"Layer {layer}", fontsize=14)
        ax.set_xlabel("Real attack catch at 1% false alarms  (→ more real attacks caught)", fontsize=12)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Overstatement: simulated − real catch  (closer to 0 = more faithful grade)", fontsize=12)
    handles = [Line2D([], [], ls="", marker="o", ms=11, color=c, label=RECIPES[r]) for r, c in COLORS.items()]
    handles += [Line2D([], [], ls="", marker=m, ms=11, color="0.35", label=n) for n, m in READOUTS.values()]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=11.5, frameon=False, bbox_to_anchor=(0.5, 0.07))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01, footer, ha="center", fontsize=10, color="0.3", wrap=True)
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    return buf.getvalue()


def points_from(q1: dict) -> list[dict]:
    """Reads by key only: call_end and reply-last-token rows (fix-full world) and d9's quoted averaged rows."""
    out = []
    for L in ("48", "63"):
        for r in RECIPES:
            for ro in ("call_end", "output_end"):
                y = q1["rows"][f"L{L}__{ro}__{r}"]["fpr_0.01"]
                out.append({"layer": L, "recipe": r, "readout": ro, "repair": "", "x": y["real_attack_catch_e2"]["value"],
                            "x_ci": y["real_attack_catch_e2"]["ci95"], "y": y["overstatement_fake_minus_e2"]["value"],
                            "y_ci": y["overstatement_fake_minus_e2"]["ci95"], "highlight": ""})
            cell = {"a": "a", "b": "b_fixed", "c": "c_fixed"}[r]
            y = q1["output_end_averaged_quoted"]["values"][L][cell]["fpr_0.01"]
            out.append({"layer": L, "recipe": r, "readout": "averaged", "repair": "simulated arm as-is" if r == "a" else "fix939",
                        "x": y["real_attack_catch_e2"]["value"], "x_ci": y["real_attack_catch_e2"]["ci95"],
                        "y": y["overstatement"]["value"], "y_ci": y["overstatement"]["ci95"],
                        "highlight": {"a": "real-trained reference", "b": "fake/fake"}.get(r, "")})
    return out


def points_from_steps(m: dict, world: str) -> list[dict]:
    """Seed (Amendment 4): all three readouts fit here, in one world, read by key from a why_metrics_steps file."""
    out = []
    for L in ("48", "63"):
        for r in RECIPES:
            for ro, key in (("call_end", "call_end"), ("output_end", "output_end"), ("averaged", "output_end_mean")):
                y = m["rows"][f"L{L}__{world}_{key}__{r}"]["fpr_0.01"]
                out.append({"layer": L, "recipe": r, "readout": ro, "repair": "", "x": y["real_attack_catch_e2"]["value"],
                            "x_ci": y["real_attack_catch_e2"]["ci95"], "y": y["overstatement_fake_minus_e2"]["value"],
                            "y_ci": y["overstatement_fake_minus_e2"]["ci95"],
                            "highlight": {"a": "real-trained reference", "b": "fake/fake"}.get(r, "") if ro == "averaged" else ""})
    return out


def add_full_repair_averaged(pts: list[dict], m: dict) -> list[dict]:
    """Amendment 7: full + F9 averaged points (filled, highlighted as the averaged ones were); fix939 averaged become hollow."""
    out = []
    for p in pts:
        out.append(p | ({"hollow": True, "highlight": ""} if p["readout"] == "averaged" else {}))
    for L in ("48", "63"):
        for r in RECIPES:
            y = m["rows"][f"L{L}__ffavg_output_end_mean__{r}"]["fpr_0.01"]
            out.append({"layer": L, "recipe": r, "readout": "averaged", "repair": "full + F9", "x": y["real_attack_catch_e2"]["value"],
                        "x_ci": y["real_attack_catch_e2"]["ci95"], "y": y["overstatement_fake_minus_e2"]["value"],
                        "y_ci": y["overstatement_fake_minus_e2"]["ci95"],
                        "highlight": {"a": "real-trained reference", "b": "fake/fake"}.get(r, "")})
    return out


@app.local_entrypoint()
def main(avg_steps: str = "", q1: str = "", tag: str = "v2", steps: str = "", world: str = "seed_ff", model: str = "qwen3-32b", title: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    if steps and not title:
        raise SystemExit("a Seed figure needs --title, written from its own numbers")
    raw = Path(steps or q1).read_bytes()
    pts = points_from_steps(json.loads(raw), world) if steps else points_from(json.loads(raw))
    if avg_steps:
        if not title:
            raise SystemExit("with --avg-steps, write the title from the new numbers (--title)")
        pts = add_full_repair_averaged(pts, json.loads(Path(avg_steps).read_bytes()))
    title = title.replace("|", "\n") or ("Fake/fake training with averaged replies comes closest to the real-trained detector;\n"
             "the command readout grades honestly but catches few real attacks")
    footer = (f"{'Seed-OSS-36B' if steps else 'Qwen3-32B'}. Threshold: 1% false alarms on held-out real honest runs. Bars: 95% intervals, 2,000 main-task resamples.\n"
              + ({"seed_ff": "Simulated runs fully repaired (full, without F9).",
                  "seed_asis": "Simulated runs as-is (unrepaired)."}[world] + " Command points: 782 attack cells with content command rows; reply points: all 939."
                 if steps else
                 ("Fully repaired (averaged replies: full + F9). Hollow triangles: averaged replies at fix939 (real-trained: its simulated arm as-is).\n"
               "Command points: 782 attack cells with content command rows; reply points: all 939."
               if avg_steps else "Simulated runs fully repaired unless marked (fix939 = earlier repair; averaged-reply fits exist only there).")))
    png = draw.remote(pts, title, footer)
    model = "seed-oss-36b_" + world if steps else model
    out = HERE / "results" / f"key_figure_{model}_2026-10-13_{tag}.png"
    with out.open("xb") as fh:
        fh.write(png)
    side = {"png": out.name, "png_sha256": hashlib.sha256(png).hexdigest(), "q1_input": Path(q1).name if q1 else None,
            "avg_steps_input": Path(avg_steps).name if avg_steps else None,
            "avg_steps_sha256": hashlib.sha256(Path(avg_steps).read_bytes()).hexdigest() if avg_steps else None,
            "steps_input": Path(steps).name if steps else None, "title": title, "footer": footer, "world": world if steps else None,
            "q1_sha256": hashlib.sha256(raw).hexdigest(), "git": git("rev-parse", "HEAD"), "points": pts}
    with (HERE / "results" / f"key_figure_{model}_2026-10-13_{tag}.json").open("x") as fh:
        fh.write(json.dumps(side, indent=1))
    print(f"wrote {out.name} {side['png_sha256'][:12]}", flush=True)
