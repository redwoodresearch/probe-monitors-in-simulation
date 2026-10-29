"""Print run_null_gpu.py's checkpoint JSONs (realism by repair level, world-direction realism-left null, scaling
diagnostic) as plain tables. Reads the JSONs only; computes nothing beyond a max over printed differences.

  uv run --no-project --python 3.12 python experiments/2026-10-13_realism_by_repair_POSTHOC/print_null_results.py \
      <folder holding the L{layer}__{unit}.json files>
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
RR = ["as-is", "fix939_(SH′)", "fix-full_(SH″)", "newline-only", "full_+_F9", "control"]
print("Realism by repair level, label-shuffle null (Amendment 1; summary (ii), 100 within-task shuffles)")
for L in (48, 63):
    for u in RR:
        r = json.loads((d / f"L{L}__rr_{u}.json").read_text())
        print(f"L{L} {r['unit'][3:]:15s} observed {r['observed']:.4f} (as run {r['as_run']:.4f}, gate {r['gate_passes']}) | null mean {r['null_mean']:.4f}, "
              f"95th {r['null_p95']:.4f}, max {max(r['null']):.4f} | p {r['p']:.4f} | runs (sim, real) {r['runs']}")
print("\nWorld-direction realism-left null (world-direction Amendment 1; mean of the 20 models' held-out AUROC, 100 shuffles)")
for L in (48, 63):
    r = json.loads((d / f"L{L}__wd.json").read_text())
    print(f"L{L}: honest runs without output rows: {r['honest_runs_without_rows']}")
    for k, v in r["variants"].items():
        print(f"  {k:4s} observed {v['observed']:.4f} (as run {v['as_run']:.4f}, gate {v['gate_passes']}) | null mean {v['null_mean']:.4f}, "
              f"5th {v['null_p5']:.4f}, 95th {v['null_p95']:.4f} | p above {v['p_above']:.4f}, p below {v['p_below']:.4f}")
print("\nScaling diagnostic: point D with directions fit without re-standardization (alt) vs as run")
worst = (0.0, "")
for L in (48, 63):
    r = json.loads((d / f"L{L}__k17p.json").read_text())
    for k, v in r["points"].items():
        print(f"L{L} {k:22s} D as run {v['as_run']['D']:+.4f} | alt {v['alt']['D']:+.4f} | alt - as run {v['alt_minus_as_run_D']:+.4f}")
        worst = max(worst, (abs(v["alt_minus_as_run_D"]), f"L{L} {k}"))
print(f"largest |alt - as run| in D: {worst[0]:.4f} ({worst[1]})")
