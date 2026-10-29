"""Side-by-side table: Qwen3-32B vs Qwen3.5-27B as the reader, for every Q1 headline number (2026-10-06).

Reads finished Q1 GPU results files (analyze_q1_gpu.py::main) listed in config/side_by_side.yaml: per text level, the
Qwen3-32B run (first 32k tokens) and the Qwen3.5-27B runs on the matched rows (the positions Qwen3-32B read) and on
whole transcripts. For each readout, comparison and variant it reports run-level AUROC with its 95% interval:
  - the headline (task split, and held-out environment), when the run has one (full mode);
  - the best layer and the layer-curve values at fixed depths, from the per-layer results every run has;
  - the label-shuffle null's 95th percentile next to each per-layer number.
Writes results/side_by_side_<stamp>.json and .md. Local and CPU only.

  uv run python experiments/2026-10-06_qwen35_replication/analyze_side_by_side.py
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

HERE = Path(__file__).parent
READOUTS = ["output_end", "call_end"]
COMPARISONS = ["real_vs_sim", "null_e1_vs_e2"]
VARIANTS = ["all_outputs", "no_editor_outputs"]
DEPTHS = [0.25, 0.5, 0.75, 1.0]  # fixed relative depths for the layer curves: layer round(q * 63) = 16, 32, 47, 63


def fmt(m: dict | None) -> str:
    if not m:
        return "—"
    null = m.get("label_shuffle_null")
    return (f"{m['auroc']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]" + (f" L{m['layer']}" if "layer" in m else "")
            + (f" (null p95 {null['p95']:.3f})" if null and "p95" in null else ""))


def numbers(path: Path) -> dict:
    """The Q1 headline numbers of one results file, keyed by (readout, comparison, variant, statistic)."""
    d = json.loads(path.read_text())
    out = {}
    for name, key in (("headline_task", "headline"), ("headline_env", "headline_held_out_environment")):
        for r, by_c in (d.get(key) or {}).items():
            for c, by_v in by_c.items():
                for v, h in by_v.items():
                    if isinstance(h, dict) and "run" in h:
                        out[(r, c, v, name)] = h["run"] | {"layers": h.get("chosen_layer_by_fold")}
    layers = sorted(d["by_layer"], key=int)
    for r in READOUTS:
        for c in COMPARISONS:
            for v in VARIANTS + [f"held_out_environment/{v}" for v in VARIANTS]:
                def at(l):  # results[readout][comparison][variant], or [...]["held_out_environment"][variant]
                    x = d["by_layer"][l]["results"].get(r, {}).get(c, {})
                    for part in v.split("/"):
                        x = x.get(part, {}) if isinstance(x, dict) else {}
                    return x.get("run") if isinstance(x, dict) else None
                curve = {int(l): at(l) for l in layers}
                curve = {l: x for l, x in curve.items() if x and "auroc" in x}
                if not curve:
                    continue
                best = max(curve, key=lambda l: curve[l]["auroc"])
                out[(r, c, v, "best_layer")] = curve[best] | {"layer": best}
                n = max(int(l) for l in layers)
                for q in DEPTHS:
                    l = round(q * n)
                    if l in curve:
                        out[(r, c, v, f"L{l}")] = curve[l]
    return out


def resolve(p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else HERE / q


def protocol_rows(files: dict) -> list[str]:
    """The frozen level-7 decision per reader: probe vs TF-IDF run AUROC on shared outputs, gap with its paired 95%
    interval, and the clean-cell gap, for both splits, output_end."""
    out = ["| split | variant | " + " | ".join(f"{n}: probe / TF-IDF / gap [95%] / clean-cell gap" for n in files) + " |",
           "|---|---|" + "---|" * len(files)]
    reports = {n: json.loads(resolve(p).read_text())["results"] for n, p in files.items()}
    for split in ("A_held_out_environment", "B_main_task"):
        for var in ("all_outputs", "no_editor_outputs"):
            cells = []
            for n, r in reports.items():
                e = r[split][f"output_end__real_vs_sim__{var}"]
                g, c = e["gap_probe_minus_tfidf"], (e.get("clean_cells") or {}).get("gap_probe_minus_tfidf") or {}
                cells.append(f"{g['probe']:.3f} / {g['baseline']:.3f} / {g['gap']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] / "
                             + (f"{c['gap']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]" if c else "—"))
            out.append(f"| {split} | {var} | " + " | ".join(cells) + " |")
    return out


def q2_rows(groups: dict) -> list[str]:
    """Q2 headlines (the layer chosen per outer fold on training-domain validation): AUROC [95%], recall and FPR at the
    frozen 1% threshold, for every view, position and train/test cell, one column per reader."""
    out = []
    for label, files in groups.items():
        res = {n: json.loads(resolve(p).read_text()) if p else None for n, p in files.items()}
        def views(d):
            return d["views"] if d and "views" in d else ({"attack_only (attack-only run)": d} if d else {})
        names = sorted(set().union(*[views(d) for d in res.values()]))
        out += [f"\n{label}:\n", "| view | position | cell | " + " | ".join(f"{n}: AUROC / recall / FPR" for n in files) + " |",
                "|---|---|---|" + "---|" * len(files)]
        for v in names:
            for pos in ("call_end", "pre_call", "prev_content"):
                cells = sorted(set().union(*[set(views(d).get(v, {}).get("positions", {}).get(pos, {}).get("headline", {})) for d in res.values()]))
                for c in cells:
                    if " minus " in c:
                        continue
                    vals = []
                    for d in res.values():
                        h = views(d).get(v, {}).get("positions", {}).get(pos, {}).get("headline", {}).get(c)
                        vals.append("—" if not h else f"{h['auroc']['value']:.3f} [{h['auroc']['ci95'][0]:.3f}, {h['auroc']['ci95'][1]:.3f}]"
                                    f" / {h['recall']['value']:.3f} / {h['fpr']['value']:.3f}")
                    out.append(f"| {v} | {pos} | {c} | " + " | ".join(vals) + " |")
    return out


def key_table(rows: list[dict], names: list[str]) -> list[str]:
    """README key numbers: per level/design, real vs sim run AUROC on held-out environments, the headline where a run
    is full mode, else its best layer on held-out environments (marked *); output_end all and no editor; call_end from
    the amendment-3 rows where a level has them."""
    def get(level, readout, stat, var):
        return next((r for r in rows if r["level"] == level and r["readout"] == readout and r["comparison"] == "real_vs_sim"
                     and r["variant"] == var and r["statistic"] == stat), None)

    def pick(level, readout, variant):
        h, b = get(level, readout, "headline_env", variant), get(level, readout, "best_layer", f"held_out_environment/{variant}")
        out = []
        for n in names:
            m, tag = (h or {}).get(n), ""
            if not m and b and b.get(n):
                m, tag = b[n], "*"
            out.append("—" if not m else f"{m['auroc']:.3f}{tag}")
        return " / ".join(out)

    levels = list(dict.fromkeys(r["level"] for r in rows))
    lines = ["| level / design | output_end, all outputs | output_end, no editor | call_end, all outputs |", "|---|---|---|---|"]
    for lv in levels:
        if "> filter" in lv:
            continue
        base, every = lv.split(" (")[0].replace(", every real run", ""), "every real run" in lv
        ce = next((l for l in levels if "> filter" in l and l.startswith(base) and ("every real run" in l) == every), lv)
        lines.append(f"| {lv} | {pick(lv, 'output_end', 'all_outputs')} | {pick(lv, 'output_end', 'no_editor_outputs')} | "
                     f"{pick(ce, 'call_end', 'all_outputs')} |")
    return lines


def main(config: str = "side_by_side.yaml"):
    cfg = yaml.safe_load((HERE / "config" / config).read_text())
    rows, table = [], []
    sources = {}
    for level, files in cfg["levels"].items():
        files = {name: (p if p and resolve(p).exists() else None) for name, p in files.items()}  # not landed yet: blank
        for name, p in files.items():
            if p:
                sources[f"{level} | {name}"] = {"file": str(resolve(p)), "sha256": hashlib.sha256(resolve(p).read_bytes()).hexdigest()[:16]}
        cols = {name: numbers(resolve(p)) if p else {} for name, p in files.items()}
        drop = set((cfg.get("exclude_readouts") or {}).get(level, []))  # e.g. call_end computed under different filters
        cols = {name: {k: v for k, v in col.items() if k[0] not in drop} for name, col in cols.items()}
        keys = sorted(set().union(*cols.values()), key=lambda k: (READOUTS.index(k[0]), COMPARISONS.index(k[1]), k[2], k[3]))
        for k in keys:
            entry = {"level": level, "readout": k[0], "comparison": k[1], "variant": k[2], "statistic": k[3],
                     **{name: col.get(k) for name, col in cols.items()}}
            rows.append(entry)
            table.append(f"| {level} | {k[0]} | {k[1]} | {k[2]} | {k[3]} | " + " | ".join(fmt(col.get(k)) for col in cols.values()) + " |")
    names = list(next(iter(cfg["levels"].values())))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    out = HERE / "results" / f"side_by_side_{stamp}"
    out.with_suffix(".json").write_text(json.dumps({"config": cfg, "sources": sources, "rows": rows}, indent=1))
    md = ["| level | readout | comparison | variant | statistic | " + " | ".join(names) + " |",
          "|---" * (5 + len(names)) + "|"] + table
    proto = protocol_rows(cfg["protocol"]) if cfg.get("protocol") else []
    q2 = q2_rows(cfg["q2"]) if cfg.get("q2") else []
    out.with_suffix(".md").write_text("Run-level AUROC [95% interval]. " + cfg.get("note", "") + "\n\n"
                                      + ("Level-7 frozen-protocol decision (output_end):\n\n" + "\n".join(proto) + "\n\n" if proto else "")
                                      + "\n".join(md) + "\n" + ("\n## Q2\n" + "\n".join(q2) + "\n" if q2 else ""))
    # compact: headlines where a run has them (full mode), else the best layer (lite levels; optimistic for both
    # readers alike); real vs sim and the real-vs-real null; Q2 attack-only and three-way headlines
    headline_full = {(r["level"], r["readout"]) for r in rows  # level/readout where every reader with a result has a headline
                     if r["statistic"] == "headline_env" and all(r.get(n) for n in names)}
    keep = lambda r: (r["statistic"] in ("headline_env", "headline_task")
                      or (r["statistic"] == "best_layer" and r["comparison"] == "real_vs_sim"
                          and (r["level"], r["readout"]) not in headline_full))
    compact = [t for r, t in zip(rows, table) if keep(r)]
    out.with_name(out.name + "_compact").with_suffix(".md").write_text(
        "| level | readout | comparison | variant | statistic | " + " | ".join(names) + " |\n" + "|---" * (5 + len(names)) + "|\n"
        + "\n".join(compact) + "\n" + ("\n## Q2\n" + "\n".join(q2) + "\n" if q2 else ""))
    out.with_name(out.name + "_key").with_suffix(".md").write_text("\n".join(key_table(rows, names)) + "\n")
    print(f"{len(rows)} rows -> {out}.md, .json; {len(compact)} compact rows -> {out.name}_compact.md; key table -> {out.name}_key.md")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="side_by_side.yaml", help="in config/: per level, the results file of each reader")
    main(ap.parse_args().config)
