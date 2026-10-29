"""Print the refit result's headline table (local) and save it as results/refit_table_<refit stamp>.txt: per reader,
each cell's AUROC, standardized pAUC 0-5% (from analyze_refit_pauc.py) and catch at 1% (calibrated in the test world,
and frozen) with 95% intervals and the shuffle null, then the contrasts, signed simulated minus real (amendment 43: the
refit file stores real minus simulated for the domain gaps and the real-target transfer; those are negated with their
interval ends swapped; the pAUC file is already signed simulated minus real).

  python3 experiments/2026-10-08_impactful_refit_bootstrap/refit_table.py results/refit_<stamp>.json results/refit_pauc_<stamp>.json
"""
import json
import sys
from pathlib import Path

refit, pauc = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
M = [("auroc", "AUROC"), ("catch, calibrated in test world", "catch@1% calibrated"), ("catch, frozen training-world threshold", "catch@1% frozen")]
FLIP = {"domain gap, trained on real (real->real minus real->sim)": ("domain gap, trained on real: real->sim − real->real", -1),
        "domain gap, trained on sim (sim->real minus sim->sim)": ("domain gap, trained on sim: sim->sim − sim->real", -1),
        "transfer loss on the simulated target (sim->sim minus real->sim)": ("transfer, tested on sim: sim->sim − real->sim", 1),
        "transfer loss on the real target (real->real minus sim->real)": ("transfer, tested on real: sim->real − real->real", -1)}
null = lambda c: f" (null mean {c['null']['mean']}, max {c['null']['max']})"
lines = [f"refit: {Path(sys.argv[1]).name}; pAUC: {Path(sys.argv[2]).name}"]
for reader, v in refit["readers"].items():
    p = pauc["readers"][reader]
    if p["fingerprint"] != v["fingerprint"]:
        raise ValueError(f"{reader}: the pAUC file was computed from other replicates")
    # the fingerprint leaves out the replicate counts, so compare them too
    if [refit["config"][k] for k in ("n_boot", "n_null")] != [pauc["config"][k] for k in ("n_boot", "n_null")] or \
            [v["replicates"][k] for k in ("bootstrap", "null draws")] != [p["replicates"][k] for k in ("bootstrap", "null draws")]:
        raise ValueError(f"{reader}: the two files differ in n_boot, n_null or the replicates they summarize")
    lines.append(f"== {reader}: {v['replicates']}; analyzed calls {v['analyzed calls']}; pAUC AUROC check {p['auroc check (max abs difference)']}")
    for cell in ("real->real", "real->sim", "sim->real", "sim->sim"):
        c = v[cell]
        parts = [f"{lab} {c[m]['value']:.4f} {c[m]['ci95']}" + (null(c[m]) if "null" in c[m] else "") for m, lab in M]
        parts.insert(1, f"pAUC 0-5% {p[cell]['value']:.4f} {p[cell]['ci95']}" + null(p[cell]))
        lines.append(f"  {cell} | " + " | ".join(parts))
    for key, (label, sign) in FLIP.items():
        c = v[key]
        parts = [f"{lab} {sign * c[m]['value']:+.4f} {sorted(round(sign * x, 4) for x in c[m]['ci95'])}" for m, lab in M]
        parts.insert(1, f"pAUC 0-5% {p[label]['value']:+.4f} {p[label]['ci95']}")
        lines.append(f"  {label} | " + " | ".join(parts))
    lines.append(f"  layer picks: {v['picks']['layers']} | C: {v['picks']['C']}")
out = Path(sys.argv[1]).parent / f"refit_table_{Path(sys.argv[1]).stem.removeprefix('refit_')}.txt"
text = "\n".join(lines) + "\n"
if out.exists() and out.read_text() != text:  # results are append-only
    raise FileExistsError(f"{out} exists with other contents")
out.write_text(text)
print("\n".join(lines))
print(out)
