"""Figure for the WHOLE-RUN (uncut) sim-vs-real detector (POST HOC, PRELIMINARY): plot_detector.py @ 648f33e61 whose only changes are
the inputs (this folder's honest_uncut / attack_uncut records; each record's own null, since the uncut fits carry it), a second row for
the ADDED per-action AUROC (rows scored by their fold model), and the app name. Every plotted number is read by key; captions.json
lists each key path. Rendered on Modal (CPU).

  <modal> run experiments/2026-10-16_cmd_sim_detector_uncut/plot_detector_uncut.py --honest results/honest_uncut_<stamp>.json --attack results/attack_uncut_<stamp>.json
"""

import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-cmdsimdet_uncut_kill-2026-11-14")
img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("matplotlib==3.10.6", "numpy==2.3.3")
LAYERS = (48, 63)
GROUPS = [("honest: sim vs real B ∪ C", "H", "simh_vs_BuC"),
          ("honest: sim vs real B,\nsame main tasks", "H", "simh_vs_Bmatched"),
          ("control: real honest\nC vs B", "H", "c_vs_b"),
          ("attack: sim vs real run 1", "A", "c_simatk_vs_run1"),
          ("attack: sim vs held-out\nrun 2 (sim-vs-run1 model)", "A", "hold:sim_vs_heldout_run2"),
          ("control: held-out run 2\nvs run 1 (same model)", "A", "hold:heldout_run2_vs_run1"),
          ("control: real run 2\nvs run 1 (own fit)", "A", "d_run2_vs_run1")]


def values(H: dict, A: dict) -> dict:
    """{level: {readout: {group: {L: (auroc, lo, hi, null_p95 or None, key path)}}}}, level = run | action, read by key."""
    out = {"run": {}, "action": {}}
    for ro in ("command", "reply"):
        for lvl in out:
            out[lvl][ro] = {}
        for _lab, src, key in GROUPS:
            for lvl in out:
                out[lvl][ro][key] = {}
            for L in LAYERS:
                cell = f"{ro}|L{L}"
                if src == "H":
                    r = H["cells"][cell]["problems"][key]
                    p = f"honest.cells['{cell}'].problems['{key}']"
                    o = r["observed"]
                    out["run"][ro][key][L] = (o["auroc"], *o["auroc_boot"]["ci95"], r["null"]["p95"], f"{p}.observed.auroc / .auroc_boot.ci95 / .null.p95")
                    out["action"][ro][key][L] = (o["row_auroc"], *o["row_auroc_boot"]["ci95"], r["null_row"]["p95"],
                                                 f"{p}.observed.row_auroc / .row_auroc_boot.ci95 / .null_row.p95")
                elif key.startswith("hold:"):
                    k = key.split(":", 1)[1]
                    r = A["cells"][cell]["heldout_run2"]["contrasts"][k]
                    p = f"attack.cells['{cell}'].heldout_run2.contrasts['{k}']"
                    out["run"][ro][key][L] = (r["auroc"], *r["boot"]["ci95"], None, f"{p}.auroc / .boot.ci95 (no null)")
                    out["action"][ro][key][L] = (r["call_level"]["auroc"], *r["call_level"]["boot"]["ci95"], None, f"{p}.call_level.auroc / .call_level.boot.ci95 (no null)")
                else:
                    r = A["cells"][cell]["problems"][key]
                    p = f"attack.cells['{cell}'].problems['{key}']"
                    out["run"][ro][key][L] = (r["run_level"]["auroc"], *r["run_auroc_boot"]["ci95"], r["null"]["run_level_auroc_summary"]["p95"],
                                              f"{p}.run_level.auroc / .run_auroc_boot.ci95 / .null.run_level_auroc_summary.p95")
                    out["action"][ro][key][L] = (r["call_level"]["auroc"], *r["call_auroc_boot"]["ci95"], r["null"]["call_level_auroc_summary"]["p95"],
                                                 f"{p}.call_level.auroc / .call_auroc_boot.ci95 / .null.call_level_auroc_summary.p95")
    return out


@app.function(image=img, cpu=2, memory=4096, timeout=300)
def render(vals: dict, title: str, caption: str) -> bytes:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    col = {48: "#4C78A8", 63: "#F58518"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
    for r_i, lvl in enumerate(("run", "action")):
        for c_i, ro in enumerate(("command", "reply")):
            ax = axes[r_i][c_i]
            x = np.arange(len(GROUPS))
            for i, L in enumerate(LAYERS):
                for g, (_lab, _src, key) in enumerate(GROUPS):
                    a, lo, hi, nul, _ = vals[lvl][ro][key][str(L)]
                    xx = g + (i - 0.5) * 0.36
                    ax.bar(xx, a - 0.5, bottom=0.5, width=0.34, color=col[L], label=f"layer {L}" if g == 0 else None)
                    ax.errorbar(xx, a, yerr=[[a - lo], [hi - a]], color="black", capsize=3, lw=1)
                    ax.text(xx, hi + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
                    if nul is not None:
                        ax.plot([xx - 0.17, xx + 0.17], [nul, nul], color="#666666", ls="--", lw=1.2, label="shuffle null 95th pct." if (g == 0 and i == 0) else None)
            ax.axhline(0.5, color="black", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([g[0] for g in GROUPS] if r_i == 1 else [""] * len(GROUPS), rotation=30, ha="right", fontsize=8.5)
            ax.set_title(f"{'per run' if lvl == 'run' else 'per action'}, {ro} readout ({'content-only end of call' if ro == 'command' else 'averaged reply'})", fontsize=11)
            ax.set_ylim(0.35, 1.0)
            ax.grid(axis="y", alpha=0.3)
        axes[r_i][0].set_ylabel(f"{'run' if lvl == 'run' else 'action'}-level AUROC (higher = more separable)")
    axes[0][0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(title, fontsize=11.5, x=0.01, ha="left")
    fig.text(0.01, 0.005, caption, fontsize=8, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    return buf.getvalue()


@app.local_entrypoint()
def main(honest: str, attack: str):
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    hb, ab = (HERE / honest).read_bytes(), (HERE / attack).read_bytes()
    H, A = json.loads(hb), json.loads(ab)
    bad = []  # (Codex r2) every validity gate the runners record must pass, else no figure
    for nm, R in (("honest", H), ("attack", A)):
        if R.get("inputs") != "uncut" or R["failed"] or len(R["cells"]) != 4 or not all(R["rows_per_run_equal_across_layers"].values()):
            bad.append(f"{nm}: not uncut, failed cells, missing cells or rows differ across layers")
        if (R.get("uncut_gates") or {"failures": ["absent"]})["failures"] or (R.get("pilot") or {}).get("stopped"):
            bad.append(f"{nm}: uncut gates failed or the pilot stopped")
        if not all(r["null_complete"] for c in R["cells"].values() for r in c["problems"].values()):
            bad.append(f"{nm}: an incomplete null")
    if A["required_gaps"]:
        bad.append("attack: required gaps")
    if bad:
        raise SystemExit(f"no figure: {bad}")
    v = values(H, A)
    c, r = v["run"]["command"], v["run"]["reply"]
    title = (f"Whole runs (uncut). Command readout: sim vs real run-level AUROC {c['simh_vs_BuC'][63][0]:.2f} honest, {c['c_simatk_vs_run1'][63][0]:.2f} attack (layer 63), "
             f"vs {c['c_vs_b'][63][0]:.2f} and {c['d_run2_vs_run1'][63][0]:.2f} for the real-vs-real controls;\naveraged reply readout: {r['simh_vs_BuC'][48][0]:.2f} honest, "
             f"{r['c_simatk_vs_run1'][48][0]:.2f} attack (layer 48), vs {r['c_vs_b'][48][0]:.2f} and {r['d_run2_vs_run1'][48][0]:.2f} for the controls")
    nH = H["cells"]["command|L63"]["problems"]
    nA = A["cells"]["command|L63"]["problems"]
    caption = (f"POST HOC, PRELIMINARY. Qwen3.5-27B's own T196 runs, whole-run activations (no 32k cut). L2 logistic regression (C = 0.01) on one row per readout unit, "
               f"5-fold CV grouped by main task (#36 folds); run score = mean row logit; per action = each row's own score. Honest: sim honest = A25's "
               f"{nH['simh_vs_BuC']['n_runs']['class1']} runs vs real honest B ∪ C ({nH['simh_vs_BuC']['n_runs']['class0']}); matched = B on the same "
               f"{nH['simh_vs_Bmatched']['n_mains']} mains. Attack: sim {nA['c_simatk_vs_run1']['n_runs']['class1']} vs run 1 {nA['c_simatk_vs_run1']['n_runs']['class0']}; "
               f"run 2 ({nA['d_run2_vs_run1']['n_runs']['class1']}) also scored held-out by the sim-vs-run1 fold models. Bars: 95% task-bootstrap intervals (2,000 draws, scores fixed; "
               f"per action: tasks drawn with all their rows). Dashed: 95th percentile of a within-task label-shuffle null refit through the CV (honest 50, attack 100 draws; "
               f"none for held-out contrasts).")
    png = render.remote({lv: {ro: {k: {str(L): list(x) for L, x in d.items()} for k, d in v[lv][ro].items()} for ro in v[lv]} for lv in v}, title, caption)
    stamp = A["stamp"]
    out = HERE / "results" / f"detector_uncut_auroc_{stamp}.png"
    out.open("xb").write(png)
    caps = {"figure": out.name, "figure_sha256": sha(png), "title": title, "caption": caption,
            "sources": {"honest": {"path": honest, "sha256": sha(hb)}, "attack": {"path": attack, "sha256": sha(ab)}},
            "values": {lv: {ro: {k: {str(L): {"auroc": x[0], "ci95": [x[1], x[2]], "null_p95": x[3], "key": x[4]} for L, x in d.items()} for k, d in v[lv][ro].items()}
                            for ro in v[lv]} for lv in v}}
    (HERE / "results" / f"detector_uncut_auroc_{stamp}_captions.json").open("x").write(json.dumps(caps, indent=1))
    print(out.name)
    print(title)
