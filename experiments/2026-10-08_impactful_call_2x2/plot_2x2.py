"""2x2 for the impactful-command probe read at the end of the command's content: trained on {real, simulated} x tested on
{real, simulated}, attack runs only (impactful vs read-only calls), per reader, with 95% intervals from main-task
resamples (seed 0) on every cell, gap and transfer loss.

Readout: call_end restricted to calls whose last token is the command's content (Q2's `call_end_content` flag: not a
trailing timeout, list, empty string or escaped newline). The probes are Q2's saved per-fold headline probes, trained on
all calls; only the test calls are restricted.
Metrics per cell (Stage 1's definitions, ../2026-10-06_q2_impactful_call_transfer/shift/stage1.py):
  - primary: catch rate at 1% false alarms calibrated in the test world (per fold, on that world's own read-only
    calls, in sample), and AUROC;
  - secondary: catch rate at the threshold frozen at 1% false alarms on the training world's validation calls, with
    the false-alarm rate it achieves.
Contrasts on the same resamples: the domain gap (the probe fixed, the test world changed: real->real minus real->sim;
and tested on simulated minus real for each training world), and the transfer loss (the target fixed, the training world
changed: sim->sim minus real->sim, and real->real minus sim->real).
Resamples redo the calibration but not the fits or the layer choice (those are the saved per-fold probes).

  uv run --with pandas --with pyarrow --with matplotlib --with joblib --with scikit-learn python \\
    experiments/2026-10-08_impactful_call_2x2/plot_2x2.py                     # off-policy, both readers
  ... plot_2x2.py --table "Qwen3.5-27B=<table>=<manifest>" --setting "On-policy: ..." --tag onpolicy
  ... plot_2x2.py --from-json results/impactful_2x2_<tag>_<stamp>.json      # redraw from saved results
  ... plot_2x2.py --james results/<off-policy>.json [results/<on-policy>.json]  # meeting figure, one column per setting
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "2026-10-06_q2_impactful_call_transfer" / "shift"))
import stage1  # noqa: E402

Q2 = ROOT / ".run_cache" / "q2_shift"
DEFAULT = [f"Qwen3.5-27B={Q2 / 'tables' / 'qwen3.5__call_end__attack_only.jsonl.gz'}={Q2 / 'vol_qwen35' / 'tw_manifest.json'}",
           f"Qwen3-32B={Q2 / 'tables' / 'qwen3-32b__call_end__attack_only.jsonl.gz'}={Q2 / 'vol' / 'base_manifest.json'}"]
CACHE = ROOT / ".run_cache" / "impactful_2x2"
TITLE = "Does the impactful-command probe work the same on simulated runs?"
CAPTION = ("Caught = an impactful command (one that changes something) scores above a threshold that flags 1% of the read-only "
           "commands in the world it is tested on (left).\nMiddle: the threshold is set on the training world instead (flags {fa} of "
           "read-only commands on the test world). Attack runs; the probe reads the last token of each command's content ({cov} of "
           "commands).\nError bars: 95% intervals from resampling tasks.")
COLORS = {"real": "#1f77b4", "sim": "#ff7f0e"}
NAMES = {"real": "real", "sim": "simulated"}
METRICS = {"catch, calibrated in test world": "recall_recal", "catch, frozen training-world threshold": "recall", "auroc": "auroc"}
K = ["arm", "key", "step", "action"]


def content_table(reader: str, table: Path, manifest: Path) -> tuple[Path, dict]:
    """The table restricted to content call_ends, written under a Stage 1 style name (<reader>__<readout>__<view>)."""
    t = pd.read_json(table, lines=True)
    m = pd.DataFrame([{**{k: r[k] for k in K}, "call_end_content": r.get("call_end_content")}  # extra arms may lack it
                      for r in json.loads(manifest.read_text()) if r["arm"] in ("real-attack-e1", "sim-attack")])
    t = t.merge(m, on=K, how="left", validate="many_to_one")
    if t.call_end_content.isna().any():
        raise ValueError(f"{reader}: {int(t.call_end_content.isna().sum())} calls missing from the manifest")
    cov = {d: round(float(t[(t.domain == d) & (t.train_set == "real")].call_end_content.mean()), 4) for d in ("real", "sim")}
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{reader.lower().replace(' ', '')}__call_end-content__attack_only.jsonl.gz"
    t[t.call_end_content.astype(bool)].drop(columns="call_end_content").to_json(out, orient="records", lines=True, compression="gzip")
    return out, cov


def per_draw(table: Path, train: str):
    """Stage 1's quantities per resample (row 0: the point estimate) for one training world, as stage1.analyze computes
    them; kept as arrays so contrasts across training worlds use the same resamples."""
    df = pd.read_json(table, lines=True)
    df = df[df.train_set == train].reset_index(drop=True)
    df["margin"] = df.score - df.threshold
    counts, inv = stage1.task_counts(df.main_task.to_numpy())
    W_all = np.vstack([np.ones((1, counts.shape[1])), counts])
    per = {"real": [], "sim": []}
    for k in range(0, len(W_all), stage1.CHUNK):
        Wc = W_all[k:k + stage1.CHUNK][:, inv].astype(float)
        for dom in per:
            i = np.flatnonzero(df.domain == dom)
            per[dom].append(stage1.domain_stats(df.iloc[i], Wc[:, i]))
    res = {dom: {m: np.concatenate([c[m] for c in chunks]) for m in chunks[0]} for dom, chunks in per.items()}
    n = {dom: {"impactful": int(((df.domain == dom) & (df.y == 1)).sum()), "read_only": int(((df.domain == dom) & (df.y == 0)).sum()),
               "main_tasks": int(df[df.domain == dom].main_task.nunique())} for dom in ("real", "sim")}
    return res, np.unique(df.main_task), n


def summ(v):
    return {"value": round(float(v[0]), 4), "ci95": [round(float(x), 4) for x in np.nanpercentile(v[1:], [2.5, 97.5])]}


def cells(table: Path) -> dict:
    d, tasks = {}, {}
    out = {}
    for train in ("real", "sim"):
        d[train], tasks[train], n = per_draw(table, train)
        out[train] = {test: {**{name: summ(d[train][test][key]) for name, key in METRICS.items()},
                             "false-alarm rate (frozen)": summ(d[train][test]["fpr"]), "n": n[test]} for test in ("real", "sim")}
        out[train]["gap (tested on simulated minus real)"] = {name: summ(d[train]["sim"][key] - d[train]["real"][key]) for name, key in METRICS.items()}
    if not np.array_equal(tasks["real"], tasks["sim"]):  # the resamples are paired only over the same tasks
        raise ValueError("the two training worlds' tables cover different main tasks")
    # Domain gap: the probe held fixed, the test world changed. Transfer loss: the target held fixed, the training world changed.
    out["domain gap, trained on real (real->real minus real->sim)"] = {name: summ(d["real"]["real"][k] - d["real"]["sim"][k]) for name, k in METRICS.items()}
    out["transfer loss on the simulated target (sim->sim minus real->sim)"] = {name: summ(d["sim"]["sim"][k] - d["real"]["sim"][k]) for name, k in METRICS.items()}
    out["transfer loss on the real target (real->real minus sim->real)"] = {name: summ(d["real"]["real"][k] - d["sim"]["real"][k]) for name, k in METRICS.items()}
    return out


def panel(ax, res, metric, ylabel):
    x = np.arange(2)
    for j, test in enumerate(("real", "sim")):
        v = [res[train][test][metric] for train in ("real", "sim")]
        val = np.array([c["value"] for c in v])
        lo, hi = val - np.array([c["ci95"][0] for c in v]), np.array([c["ci95"][1] for c in v]) - val
        pos = x + (j - 0.5) * 0.38
        ax.bar(pos, val, 0.36, color=COLORS[test], label=f"tested on {NAMES[test]} runs", yerr=[lo, hi], capsize=4)
        for p, y_, h in zip(pos, val, hi):
            ax.text(p, y_ + h + 0.015, f"{y_:.1%}" if metric != "auroc" else f"{y_:.3f}", ha="center", va="bottom", fontsize=9)
    pct = metric != "auroc"
    unit = (lambda v: f"{v * 100:+.1f}") if pct else (lambda v: f"{v:+.3f}")
    gap = lambda train: res[train]["gap (tested on simulated minus real)"][metric]
    labels = [f"trained on {NAMES[t]} runs\nsimulated − real:\n{unit(gap(t)['value'])}{' pts' if pct else ''} "
              f"[{unit(gap(t)['ci95'][0])}, {unit(gap(t)['ci95'][1])}]" for t in ("real", "sim")]
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(ylabel)


def draw(results, png):
    readers = list(results["readers"])
    fig, axes = plt.subplots(len(readers), 3, figsize=(17, 4.8 * len(readers)), squeeze=False)
    for i, reader in enumerate(readers):
        res = results["readers"][reader]
        panel(axes[i, 0], res, "catch, calibrated in test world", "Impactful commands caught at 1% false alarms\n(higher is better)")
        panel(axes[i, 1], res, "catch, frozen training-world threshold", "Impactful commands caught\n(higher is better)")
        panel(axes[i, 2], res, "auroc", "AUROC, impactful vs read-only\n(higher is better)")
        head = " (headline)" if i == 0 and len(readers) > 1 else ""
        axes[i, 0].set_title(f"{reader}{head}: caught, 1% false alarms in the test world", fontsize=10, loc="left")
        axes[i, 1].set_title(f"{reader}: caught, threshold from the training world (secondary)", fontsize=10, loc="left")
        axes[i, 2].set_title(f"{reader}: ranking (AUROC)", fontsize=10, loc="left")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f"{results['question']}\n{results['setting']}", fontsize=12)
    fig.text(0.5, 0.005, results["caption"], ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    fig.savefig(png, dpi=150)


def draw_james(paths, png, reader=None):
    """The meeting version: the headline reader only; catch at 1% false alarms in the test world (large) over AUROC
    (small); one column per setting (off-policy now, on-policy beside it later), with shared axes, colors and labels."""
    res = [json.loads(Path(p).read_text()) for p in paths]
    reader = reader or next(iter(res[0]["readers"]))
    fig, axes = plt.subplots(2, len(res), figsize=(6.2 * len(res), 7.2), squeeze=False, gridspec_kw={"height_ratios": [3, 1.25]})
    for j, r in enumerate(res):
        cells_ = r["readers"][reader]
        panel(axes[0, j], cells_, "catch, calibrated in test world",
              "Impactful commands caught\nat 1% false alarms (higher is better)" if j == 0 else "")
        panel(axes[1, j], cells_, "auroc", "AUROC (higher is better)" if j == 0 else "")
        axes[1, j].set_ylim(0.9, 1.005)
        axes[1, j].set_yticks([0.9, 0.95, 1.0])
        for t in axes[1, j].texts:
            t.set_fontsize(8)
        axes[0, j].set_title(r["setting"], fontsize=11)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f"{res[0]['question']}\n{reader}", fontsize=12)
    import textwrap
    caption = ("Caught = an impactful command (one that changes something) scores above a threshold that flags 1% of the read-only "
               "commands in the world it is tested on. The probe reads the last token of each command's content, in attack runs. "
               "Error bars: 95% intervals from resampling tasks. AUROC axis starts at 0.9. Preliminary.")
    fig.text(0.5, 0.005, "\n".join(textwrap.wrap(caption, 95 * len(res))), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    fig.savefig(png, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="append", help="reader=table=manifest (repeatable; the first is the headline)")
    ap.add_argument("--setting", default="Off-policy: Qwen reads Sol's transcripts")
    ap.add_argument("--tag", default="offpolicy")
    ap.add_argument("--from-json", help="redraw the figure from a saved results file")
    ap.add_argument("--james", nargs="+", help="meeting figure from saved results files, one column per setting")
    args = ap.parse_args()
    if args.james:
        png = HERE / "results" / f"impactful_2x2_james_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.png"
        draw_james(args.james, png)
        print(png)
        return
    if args.from_json:  # a redraw gets its own name: results/ is append-only
        js = Path(args.from_json)
        png = js.with_name(f"{js.stem}_redrawn_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.png")
        draw(json.loads(js.read_text()), png)
        print(png)
        return
    results = {"question": TITLE, "setting": args.setting, "readers": {}}
    covs, fas = [], []
    for spec in (args.table or DEFAULT):
        reader, table, manifest = spec.split("=", 2)
        t, cov = content_table(reader, Path(table), Path(manifest))
        res = cells(t)
        rel = lambda p: str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else str(p)
        results["readers"][reader] = {"inputs": {rel(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (table, manifest)},
                                      "content call_end coverage (share of attack calls)": cov, **res}
        covs += list(cov.values())
        fas += [res[tr][te]["false-alarm rate (frozen)"]["value"] for tr in ("real", "sim") for te in ("real", "sim")]
    results["caption"] = CAPTION.format(cov=f"{min(covs):.0%}–{max(covs):.0%}", fa=f"{min(fas):.1%}–{max(fas):.1%}")
    results["note"] = ("Primary catch rate: threshold at 1% false alarms on the test world's own read-only calls, per fold, in "
                       "sample (not task-disjoint). Secondary: threshold frozen on the training world's validation calls (false-alarm "
                       f"rates achieved on the test worlds: {min(fas):.1%}–{max(fas):.1%}). Resamples redo calibration, not fitting or layer choice. Label-shuffle null for this slice: run_content_null.py (results/content_null_*.json).")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    js = HERE / "results" / f"impactful_2x2_{args.tag}_{stamp}.json"
    js.write_text(json.dumps(results, indent=1))
    draw(results, js.with_suffix(".png"))
    for reader, r in results["readers"].items():
        f = lambda c: f"{c['value']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"
        print(reader, "coverage", r["content call_end coverage (share of attack calls)"])
        for train in ("real", "sim"):
            print(f"  trained on {train}: " + " | ".join(f"{m}: real {f(r[train]['real'][m])} sim {f(r[train]['sim'][m])}" for m in METRICS))
        for c in ("domain gap, trained on real (real->real minus real->sim)", "transfer loss on the simulated target (sim->sim minus real->sim)",
                  "transfer loss on the real target (real->real minus sim->real)"):
            print(f"  {c}:", {m: f(v) for m, v in r[c].items()})
    print(js.with_suffix(".png"))


if __name__ == "__main__":
    main()
