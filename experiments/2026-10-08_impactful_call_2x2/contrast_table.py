"""The README's contrast table from the off-policy 2x2's full result (impactful_2x2_offpolicy_2026-10-08_114908.json, on
S3 per results/manifest_v1.json), every gap signed simulated minus real (amendment 43): domain gap = the test world's
simulated minus real (real->sim minus real->real); transfer = simulated-trained minus real-trained, per test world. The
result file stores real-minus-simulated contrasts for the domain gap and the real-target transfer, so those are negated
with their interval ends swapped; the simulated-target transfer (sim->sim minus real->sim) is already signed this way.

  python3 experiments/2026-10-08_impactful_call_2x2/contrast_table.py <path to impactful_2x2_offpolicy_2026-10-08_114908.json>
"""
import json
import sys

ROWS = [("domain gap, trained on real (real->real minus real->sim)", -1, "domain gap: real→sim − real→real"),
        ("transfer loss on the simulated target (sim->sim minus real->sim)", 1, "transfer, tested on simulated: sim→sim − real→sim"),
        ("transfer loss on the real target (real->real minus sim->real)", -1, "transfer, tested on real: sim→real − real→real")]
METRICS = [("catch, calibrated in test world", "pts"), ("catch, frozen training-world threshold", "pts"), ("auroc", "auroc")]


def cell(v: dict, sign: int, unit: str) -> str:
    x, (lo, hi) = sign * v["value"], sorted(sign * c for c in v["ci95"])
    return f"{x * 100:+.1f} pts [{lo * 100:+.1f}, {hi * 100:+.1f}]" if unit == "pts" else f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]"


with open(sys.argv[1]) as fh:
    r = json.load(fh)["readers"]
print("| Reader | Contrast (simulated minus real) | Caught (1% in test world) | Caught (training-world threshold) | AUROC |")
print("|---|---|---|---|---|")
for reader, v in r.items():
    for key, sign, label in ROWS:
        print(f"| {reader} | {label} | " + " | ".join(cell(v[key][m], sign, u) for m, u in METRICS) + " |")
