"""Write RESULT.md for the halves mask split (arms N and W) from its two result JSONs, every number read by key after a
sha256 check against results_manifest_v1.json (the JSONs are on S3; each equals its blob at e064ed40).
  uv run --no-project --python 3.12 python experiments/2026-10-12_claimB_halves_mask_split/make_result_md.py \\
      <halves_splitN_2026-10-12_143658.json> <halves_splitW_2026-10-12_143656.json>
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ARMS = [("N", "numbers/IDs only", "halves_splitN_2026-10-12_143658.json"),
        ("W", "whitespace only", "halves_splitW_2026-10-12_143656.json")]
COMMIT = "e064ed40"

man = json.loads((HERE / "results_manifest_v1.json").read_text())["files"]
if len(sys.argv) != len(ARMS) + 1:
    raise SystemExit(__doc__)
data = {}
for (arm, _, name), path in zip(ARMS, sys.argv[1:]):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != man[f"experiments/{HERE.name}/results/{name}"]["sha256"]:
        raise SystemExit(f"{path} does not match the manifest's sha256 for {name}")
    data[arm] = json.loads(raw)

pct = lambda x: f"{100 * x:.2f}%"  # noqa: E731
pts = lambda x: f"{100 * x:+.2f}"  # noqa: E731
iv = lambda v: f"[{100 * v[0]:+.2f}, {100 * v[1]:+.2f}]"  # noqa: E731
DIRS = [("dev → test", "dev_to_test"), ("test → dev", "test_to_dev")]


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


primary = table(["Arm", "Primary reading (analyzer)", "Simulated caught", "e2 flagged (direct)", "Direct gap [95%]",
                 "Sim catch null p", "Null columns", "Both planted checks pass"],
                [[f"{a} ({desc})", f'"{data[a]["primary"]["reading"]}"', pct(data[a]["primary"]["sim_catch"]),
                  pct(data[a]["primary"]["direct_e2_flagged"]),
                  f'{pts(data[a]["primary"]["direct_gap"])} {iv(data[a]["primary"]["direct_gap_ci95"])}',
                  f'{data[a]["primary"]["sim_null_p"]:.4f}', str(data[a]["null_columns_used"]["pooled_sim_catch"]),
                  "yes" if data[a]["primary"]["planted_both_pass"] else "no"] for a, desc, _ in ARMS])
atten = table(["Arm", "Halves direct gap", "Arm direct gap", "Change [95%]", "Reading (analyzer)",
               "Unmasked selection: arm gap [95%]", "Unmasked: change [95%]"],
              [[a, pts(data[a]["attenuation"]["halves_direct_gap"]), pts(data[a]["attenuation"]["arm_direct_gap"]),
                f'{pts(data[a]["attenuation"]["change"])} {iv(data[a]["attenuation"]["change_ci95"])}',
                f'"{data[a]["attenuation"]["reading"]}"',
                f'{pts(data[a]["unmasked_selection"]["arm_direct_gap"])} {iv(data[a]["unmasked_selection"]["arm_direct_gap_ci95"])}',
                f'{pts(data[a]["unmasked_selection"]["change"])} {iv(data[a]["unmasked_selection"]["change_ci95"])}']
               for a, _, _ in ARMS])
planted = table(["Arm", "Direction", "Pairs", "Held-out AUROC", "Floor", "Refit null max", "p", "Null columns", "Check"],
                [[a, dn, str(p["pairs"]), f'{p["auroc"]:.3f}', f'{p["floor"]:.3f}', f'{p["refit_null_max"]:.3f}', f'{p["p"]:.4f}',
                  str(p["null_columns"]), "pass" if p["pass"] else "**fail**"]
                 for a, _, _ in ARMS for dn, dk in DIRS for p in [data[a]["results"][dk]["planted"]]])
catch = table(["Arm", "Set", "Simulated caught", "e2 flagged (direct)", "Direct gap [95%]", "e2 flagged (separate)",
               "Separate gap [95%]", "Separate-yardstick reading (analyzer)"],
              [[a, sn, pct(c["sim_catch"]), pct(c["direct_e2_flagged"]), f'{pts(c["direct_gap"])} {iv(c["direct_gap_ci95"])}',
                pct(c["e2_flagged"]), f'{pts(c["gap"])} {iv(c["gap_ci95"])}', f'"{r["catch_reading"]}"']
               for a, _, _ in ARMS for sn, sk in DIRS + [("pooled", "pooled")]
               for r in [data[a]["results"][sk]] for c in [r["catch_1pct"]]])
auroc = table(["Arm", "Direction", "Simulated vs e1", "e2 vs e1", "Difference [5th, 95th percentile]", "X, the AUROC margin",
               "Readings (analyzer)"],
              [[a, dn, f'{u["fixed_vs_e1"]:.3f}', f'{u["e2_vs_e1"]:.3f}', f'{u["delta"]:+.3f} [{u["lower_95"]:+.3f}, {u["upper_95"]:+.3f}]',
                f'{u["X"]:.3f}', "; ".join(f'{k}: "{v}"' for k, v in data[a]["results"][dk]["auroc_reading"].items())]
               for a, _, _ in ARMS for dn, dk in DIRS for u in [data[a]["results"][dk]["auroc"]]])
conv = table(["Arm", "Array", "Entries", "Unconverged entries", "Columns without a converged entry",
              "Real-label unmasked choice converged"],
             [[a, k.replace("|", ", "), str(v["entries"]), str(v["unconverged_entries"]), str(v["columns_without_converged_entry"]),
               "yes" if v["column0_unmasked_choice_converged"] else "no"] for a, _, _ in ARMS for k, v in data[a]["convergence"].items()])
runs = ", ".join(f'{a} {data[a]["git"][:8]}' for a, _, _ in ARMS)

md = f"""# Halves mask split result (arms N and W; exploratory; preliminary until reviewed)

{data['N']['label']}; and {data['W']['label']}. Rule: `RULE.md` ("Reading rule (per arm; fixed now)", with amendment
1). Results: `results/{ARMS[0][2]}` and `results/{ARMS[1][2]}` (committed at {COMMIT}; run git {runs}), on S3 behind
`results_manifest_v1.json`; `make_result_md.py` checks their sha256 and writes this file. Percentages are shares of runs
flagged at the 1% alarm; gaps are in percentage points with task-clustered 95% intervals. AUROC values, differences
and margins are in AUROC units. Quoted strings are the analyzer's readings, copied from the result files.

## Primary: pooled catch gap at the 1% alarm, direct yardstick, convergence mask

The direct yardstick scores the second real run (e2) with the detector trained on the simulated side, at the same
threshold.

{primary}

## Change against halves, same cells

{atten}

## Planted checks (plant-2 pairs against refit shuffles)

{planted}

## Each direction, and the separate yardstick

The separate yardstick scores e2 with its own e2-vs-e1 detector.

{catch}

## AUROC, simulated vs e1 against e2 vs e1

{auroc}

## Convergence

{conv}
"""
(HERE / "RESULT.md").write_text(md)
print(md)
