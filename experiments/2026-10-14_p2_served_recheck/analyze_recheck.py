"""Summarize a recheck record (counts, hashes, max |diff| only) claim by claim. Reads one small JSON; no scores are printed.

  python3 experiments/2026-10-14_p2_served_recheck/analyze_recheck.py results/recheck_<stamp>.json
"""

import json
import sys
from collections import Counter

r = json.load(open(sys.argv[1]))
P = print
P(f"record {sys.argv[1]}  stamp {r['stamp (PT)']}  git {r['git']['commit'][:9]}  smoke={r['smoke']}")
P("hashes:", {k: v["match"] for k, v in r["hashes"].items()}, "UA rows", r["hashes"]["unit_assignment"]["rows"])

g = r["claims"]["1_gate"]
P(f"\n[1] gate: {g['verified_cells']}/{g['cells']} cells verified (same 194 keys, every run within 1e-6)")
P("    max |diff| over cells:", max(v["max_abs_diff"] for v in g["mine"].values()),
  "| runs per cell:", sorted({v["runs_mine"] for v in g["mine"].values()}), "| rows:", g["row_counts"])
P("    claimed:", g["claimed (manifest A)"], "A0/B pass:", g["claimed (manifests A0, B) pass"])

c2 = r["claims"]["2_units"]
P("\n[2] npz units per layout:", {L: {ro: v["n"] for ro, v in d.items()} for L, d in c2["per_layout"].items()})
P("    runs per layout:", {L: sorted({v["runs"] for v in d.values()}) for L, d in c2["per_layout"].items()}, "scored runs", c2["scored runs"])
P("    identical across layouts:", c2["identical_across_layouts"])
for k, v in c2["npz_vs_reader_units (layout A)"].items():
    P(f"    {k}: reader {v['reader_units']} npz {v['npz_units']} extra {v['npz_extra']} unread {v['npz_unread']} runs {v['runs_reader']}/{v['runs_npz']}")

c3 = r["claims"]["3_A0_vs_A_honest"]
P(f"\n[3] A0 vs A honest rows: {c3['identical_npz_files']}/20 npz identical;",
  "run means identical:", {k: v["identical"] for k, v in c3["mine"].items() if "run_means" in k})

c4 = r["claims"]["4_unit_assignment"]
P("\n[4] UA:", {k: v for k, v in c4["ua"].items() if k != "columns"})
P("    claimed provenance:", c4["manifest_provenance (claimed)"])
P("    log scan:", c4["log_scan"])
tot = Counter()
for w, v in c4["sample_checks"].items():
    for k, x in v["counts"].items():
        tot[k] += x
for k in sorted(tot):
    P(f"    {k}: {tot[k]}")

c5 = r["claims"]["5_28d"]
cl = c5["claimed"]
P("\n[5] 28d (mismatched / units, affected runs): mine vs claimed")
for ro in ("reply_mean", "reply_last", "command", "call_end_content"):
    for w in ("run1", "run2", "sim", "honest_a", "honest_b", "honest_c"):
        m = c5["mine"].get(f"{ro}|{w}")
        if m is None:
            continue
        if ro.startswith("reply"):
            c = cl["reply"][w]
        elif ro == "command":
            c = (cl["command_mismatched"][w], cl["command_units"][w], None)
        else:
            c = (cl["command_mismatched"][w], cl["call_end_units"][w], None)
        P(f"    {ro:16s} {w:8s} {m['mismatched']}/{m['units']} ({m['affected_runs']})  claimed {c[0]}/{c[1]}"
          + (f" ({c[2]})" if c[2] is not None else "") + f"  flags agree {m['npz_flag_agrees_with_mine']}/{m['units']}"
          + f"  npz False per layout {m['npz_flag_false_per_layout']}  no pair {m['units_without_token_pair']}  tools {m['mismatched_by_tool']}")

c6 = r["claims"]["6_layouts_and_positions"]
P("\n[6] layout and position checks (pass / FAIL summed over worlds)")
agg = Counter()
for w, v in c6.items():
    if w == "stage_record_B_audit":
        continue
    for k, x in v["counts"].items():
        agg[k] += x
for k in sorted(agg):
    if k.endswith(": pass") or k.endswith(": FAIL"):
        P(f"    {k}: {agg[k]}")
for k in sorted(agg):
    if not (k.endswith(": pass") or k.endswith(": FAIL") or k.startswith("FAIL")):
        P(f"    (info) {k}: {agg[k]}")
for w, v in c6.items():
    if w != "stage_record_B_audit":
        a = c6["stage_record_B_audit"][w]
        P(f"    think {w}: mine {v['think']}  | stage audit emptied {a['B: think blocks emptied']} removed {a['B: think tokens removed']}"
          f" unclosed {a['B: unclosed think blocks kept']}  | spans {v['span_convention']}  multiplicity {v['multiplicity']}  sys_hash {v['sys_hash']}")
        if v["fails"]:
            P(f"    FAIL keys {w}: {json.dumps(v['fails'])[:1500]}")

c7 = r["claims"]["7_per_row"]
P("\n[7] per-row scores (stored vs mine)")
for L in ("A0", "A", "B"):
    cells = c7[L]["cells"]
    rows = sum(v["rows"] for v in cells.values())
    P(f"  {L}: {len(cells)} cells, {rows} rows; verified cells {sum(1 for v in cells.values() if [x for k, x in v.items() if k.startswith('verified')][0])};"
      f" unmatched {sum(v['unmatched_rows'] for v in cells.values())}; nonfinite/missing {sum(v['nonfinite_or_missing'] for v in cells.values())};"
      f" max|diff| f32 {max(v['max_abs_diff_f32'] for v in cells.values()):.3g}; rows >1e-6 (f32) {sum(v['n_gt_1e-6_f32'] for v in cells.values())};"
      f" max|diff| vs exact f64 {max(v['max_abs_diff_f64'] for v in cells.values()):.3g};"
      f" max rounding ratio {max(v['max_rounding_ratio |stored - exact| / (eps32 * sum|x_i w_i|)'] for v in cells.values()):.3g}")
    P(f"     run means: max|diff| {max(v['run_means']['max_abs_diff'] for v in cells.values()):.3g};"
      f" runs >1e-6 {sum(v['run_means']['n_gt_1e-6_or_nonfinite'] for v in cells.values())}; same keys all {all(v['run_means']['same_keys'] for v in cells.values())};"
      f" runs per cell {sorted({v['run_means']['runs_mine'] for v in cells.values()})}")
    for k, v in cells.items():
        P(f"     {k:34s} rows {v['rows']:6d} max|d| {v['max_abs_diff_f32']:.3g} >1e-6 {v['n_gt_1e-6_f32']:5d} ratio {v['max_rounding_ratio |stored - exact| / (eps32 * sum|x_i w_i|)']:.3g}"
          + (f" logits max|d| {v['fibs_logits_max_abs_diff_f32']:.3g} ratio {v['fibs_logits_max_rounding_ratio']:.3g} layers ok {v['fibs_layers_stored == mine']}" if k.startswith("fibs") else ""))
    P("     counts:", c7[L]["counts"], "fails:", {k: len(v) for k, v in c7[L]["fails"].items()})
P("  projection:", c7["projection_stats"])
