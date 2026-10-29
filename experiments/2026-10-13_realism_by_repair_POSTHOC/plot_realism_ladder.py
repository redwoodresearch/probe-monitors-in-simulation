"""Item 7c (v2): honest realism and the overstated attack catch by repair level, one figure. Renders on Modal CPU.

Inputs, each checked against its sha256 before anything is sent:
- top bars and 95% intervals: the as-run realism JSON (results/manifest_realism_v1.json), levels[...]["summary_ii"] and
  control["summary_ii"]; the tab and insert fixes from the F10/F11 rows (results/manifest_fx_realism_v2.json), fx_1 and fx_2;
- top ticks (each bar's label-shuffle null, 95th percentile): the null JSONs (results/manifest_null_gpu_full_v1.json) and fx_1, fx_2;
- bottom bars: d9's fix-full metrics (sha256 9d3e6f8a…), layers[L][row]["fpr_0.01"]["fake_minus_real"] for rows c, c_fixed (c′),
  c_full (c″), c_F9; df's why-steps metrics (aswin/why-steps @ 6cec82c3, sha256 3ae21656…),
  rows["L{L}__ffavgF10_output_end__c"] and ["L{L}__ffavgF11_output_end__c"]["fpr_0.01"]["overstatement_fake_minus_e2"].
  df's full + F9 row must equal d9's c_F9 (checked; the job stops if not).
Plain level names: unrepaired = as-is; two repairs = fix939; full repair = fix-full; + ninth fix = full + F9;
+ tab fix = full + F9 + F10; + insert fix = full + F9 + F11 (tab and insert fixes each added alone).

  uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run experiments/2026-10-13_realism_by_repair_POSTHOC/plot_realism_ladder.py \\
      --asrun <as-run JSON> --null-dir <rr JSONs> --fx-dir <fx JSONs> --metrics <d9 metrics> --why <df metrics> --out <png>
"""
import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-realism-null_plot_kill-2026-11-13")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("matplotlib==3.10.6", "numpy==2.3.3")
SHA = {"metrics": "9d3e6f8a5123c9f242ea96a584db40d7b009092ba47a31541e12930f52a2a7dd", "why": "3ae21656589e0f33a14a30e668cdb2f3946df33d83d32fa3e611870ef9dc8f06"}
# (plain name, as-run level, null file stem, fx file stem, d9 row, df row)
LEVELS = [("unrepaired", "as-is", "rr_as-is", None, "c", None), ("two repairs", "fix939 (SH′)", "rr_fix939_(SH′)", None, "c_fixed (c′)", None),
          ("full repair", "fix-full (SH″)", "rr_fix-full_(SH″)", None, "c_full (c″)", None), ("+ ninth fix", "full + F9", "rr_full_+_F9", None, "c_F9", None),
          ("+ tab fix", None, None, "fx_1", None, "ffavgF10"), ("+ insert fix", None, None, "fx_2", None, "ffavgF11")]


@app.function(image=image, cpu=2, memory=4096, timeout=300, retries=0)
def render(asrun: str, files: dict, metrics: str, why: str) -> bytes:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    A = {r["layer"]: r for r in json.loads(asrun)["layers"]}
    J = {k: json.loads(v) for k, v in files.items()}
    M, W, copies = json.loads(metrics)["layers"], json.loads(why)["rows"], json.loads(metrics)["copies_by_source"]
    for L in (48, 63):  # the attack panel's n, by key: 939 simulated and 939 real (e2) attack runs in every row
        for row in ("ffavg", "ffavgF10", "ffavgF11"):
            k = W[f"L{L}__{row}_output_end__c"]["runs_kept"]
            if (k["sim-attack"], k["real-attack-e2"]) != (939, 939):
                raise RuntimeError("df's attack runs are not 939 and 939")
    if any(sum(copies[a].values()) != 939 for a in ("sim-attack-fullfix", "sim-attack-f9")):
        raise RuntimeError("d9's repaired attack arms are not 939 runs")
    for L in (48, 63):  # df's full + F9 must be d9's c_F9
        if abs(W[f"L{L}__ffavg_output_end__c"]["fpr_0.01"]["overstatement_fake_minus_e2"]["value"] - M[str(L)]["c_F9"]["fpr_0.01"]["fake_minus_real"]["value"]) > 1e-9:
            raise RuntimeError("df's full + F9 overstatement differs from d9's")
    top, bot = {48: [], 63: []}, {48: [], 63: []}
    for L in (48, 63):
        for name, lv, rr, fx, row9, rowdf in LEVELS:
            if lv:
                s = A[L]["levels"][lv]["summary_ii"]
                top[L].append((name, s["value"], *s["ci95"], J[f"L{L}__{rr}"]["null_p95"]))
            else:
                r = J[f"L{L}__{fx}"]
                top[L].append((name, r["summary_ii"], *r["ci95"], r["null_p95"]))
            o = M[str(L)][row9]["fpr_0.01"]["fake_minus_real"] if row9 else W[f"L{L}__{rowdf}_output_end__c"]["fpr_0.01"]["overstatement_fake_minus_e2"]
            bot[L].append((name, 100 * o["value"], 100 * o["ci95"][0], 100 * o["ci95"][1]))
        s = A[L]["control"]["summary_ii"]
        top[L].append(("real vs real\n(control)", s["value"], *s["ci95"], J[f"L{L}__rr_control"]["null_p95"]))
    colors = {48: "#0072B2", 63: "#E69F00"}  # Okabe-Ito
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(18, 12.5), gridspec_kw={"height_ratios": [1.2, 1]})
    bw, off = 0.27, 0.14
    tw = 0.6 * bw  # each null tick: 60% of a bar's width, beside its bar, clear of the neighbouring pair's
    texts, obstacles = {a1: [], a2: []}, {a1: [], a2: []}
    for ax, data, fmt, pad in ((a1, top, "{:.3f}", 0.012), (a2, bot, "+{:.1f}", 1.5)):
        for j, L in enumerate((48, 63)):
            for i, item in enumerate(data[L]):
                xc = i + (-off if L == 48 else off)
                v, lo, hi = item[1], item[2], item[3]
                obstacles[ax].append(ax.bar(xc, v, bw, color=colors[L], label=f"layer {L}" if (i == 0 and ax is a1) else None)[0])
                cw = bw * 0.22  # the whisker and its caps as plain lines, so each has a tight bounding box
                obstacles[ax] += [ax.plot([xc, xc], [lo, hi], color="black", lw=1.4)[0], ax.plot([xc - cw, xc + cw], [lo, lo], color="black", lw=1.4)[0],
                                  ax.plot([xc - cw, xc + cw], [hi, hi], color="black", lw=1.4)[0]]
                if ax is a1:  # the null tick, on the outer side of the pair
                    x0 = xc - bw / 2 - tw if L == 48 else xc + bw / 2
                    obstacles[ax].append(ax.plot([x0, x0 + tw], [item[4]] * 2, color="black", lw=3.5, solid_capstyle="butt",
                                                 label="95th percentile of 100 label shuffles" if (i == 0 and L == 48) else None)[0])
                texts[ax].append(ax.text(xc, hi + pad, fmt.format(v), ha="center", va="bottom", fontsize=12, fontweight="bold"))
        names = [it[0] for it in data[48]]
        ax.set_xticks(np.arange(len(names)), names, fontsize=13)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.tick_params(axis="y", labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    a1.set_ylim(0.4, 1.06)
    a1.set_ylabel("AUROC, real vs simulated honest runs\n(higher = easier to tell simulated from real)", fontsize=13.5)
    a2.set_ylim(0, 100)
    a2.set_title("Detector trained on simulated attacks vs real honest runs: how much simulated attacks overstate its catch rate", fontsize=14)
    a2.set_ylabel("Catch on simulated minus real attacks,\npoints, at 1% false alarms\n(higher = simulated runs overstate more)", fontsize=13.5)
    fig.suptitle("Repairs shrink but don't remove how easily simulated honest runs are told from real ones\n"
                 "Qwen3-32B reading GPT-5.6-Sol's runs (real and simulated), layers 48 and 63", fontsize=16, fontweight="bold")
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, fontsize=12.5, frameon=False, bbox_to_anchor=(0.5, 0.91))
    fig.text(0.5, 0.008, "n: 437 simulated vs 885 real honest runs (control 256 vs 256), 939 simulated vs 939 real attack runs; last-token readout; "
             "whiskers: 95% task bootstrap; tab and insert fixes each added alone", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.025, 1, 0.895))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in (a1, a2):  # move each value label up until it touches no whisker, tick, bar or earlier label
        step = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.005
        obs = [o.get_window_extent(r) for o in obstacles[ax]]
        placed = []
        for t in texts[ax]:
            for _ in range(400):
                b = t.get_window_extent(r).expanded(1.12, 1.15)
                if not any(b.overlaps(o) for o in obs + placed):
                    break
                x, y = t.get_position()
                t.set_position((x, y + step))
            else:
                hits = [type(o).__name__ for o, e in zip(obstacles[ax], obs) if b.overlaps(e)] + ["label"] * sum(b.overlaps(e) for e in placed)
                raise RuntimeError(f"label {t.get_text()} could not be placed without overlap: {hits}")
            placed.append(t.get_window_extent(r))
        top_px = ax.get_window_extent(r).y1
        if any(p.y1 > top_px for p in placed):
            raise RuntimeError("a label runs past the top of its panel")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    return buf.getvalue()


def checked(paths: list, manifest: Path) -> dict:
    m = {Path(f["key"]).name: f["sha256"] for f in json.loads(manifest.read_text())["files"]}
    out = {}
    for p in paths:
        b = p.read_bytes()
        if hashlib.sha256(b).hexdigest() != m.get(p.name):
            raise SystemExit(f"{p.name}: sha256 differs from {manifest.name}")
        out[p.stem] = b.decode()
    return out


@app.local_entrypoint()
def main(asrun: str, null_dir: str, fx_dir: str, metrics: str, why: str, out: str):
    res = HERE / "results"
    a = checked([Path(asrun)], res / "manifest_realism_v1.json")
    files = checked(list(Path(null_dir).glob("L*__rr_*.json")), res / "manifest_null_gpu_full_v1.json")
    files |= checked(list(Path(fx_dir).glob("L*__fx_*.json")), res / "manifest_fx_realism_v2.json")
    src = {}
    for k, p in (("metrics", metrics), ("why", why)):
        b = Path(p).read_bytes()
        if hashlib.sha256(b).hexdigest() != SHA[k]:
            raise SystemExit(f"{p}: sha256 differs from the expected {k} file")
        src[k] = b.decode()
    Path(out).write_bytes(render.remote(next(iter(a.values())), files, src["metrics"], src["why"]))
    print(f"wrote {out}")
