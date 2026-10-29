"""Print an addendum record (counts, yes/no, ratios only; no thresholds, catches or scores).

  python3 experiments/2026-10-14_p2_served_recheck/analyze_addendum.py results/addendum_<stamp>.json
"""

import json
import sys

r = json.load(open(sys.argv[1]))
P = print
P(f"addendum {sys.argv[1]}  stamp {r['stamp (PT)']}  git {r['git'][:9]}")
P("[1] wording (from the main record):", json.dumps(r["1_wording (from the main record, by key)"]))
t = r["2_tolerance (POST HOC, k = 8)"]
P(f"\n[2] tolerance: {t['definition']}")
for L, cells in t["served"].items():
    P(f"  served {L}: row-matching problems {t['served row-matching problems'][L]}")
    for k, c in cells.items():
        if not c.get("complete", True):
            P(f"    {k:32s} INCOMPLETE {c}")
            continue
        s = f"    {k:32s} rows {c['rows']['n']:6d} max ratio {c['rows']['max_ratio']:.3g} pass {c['rows']['pass']} | runs(mean) max {c['runs (mean)']['max_ratio']:.3g} pass {c['runs (mean)']['pass']}"
        if "runs (max)" in c:
            s += f" | runs(max) max {c['runs (max)']['max_ratio']:.3g} pass {c['runs (max)']['pass']}"
        if "fibs logits" in c:
            s += f" | logits max {c['fibs logits']['max_ratio']:.3g} pass {c['fibs logits']['pass']}"
        s += f" | stored==run_means {c['stored run means == run_means.json (within 1e-9)']}"
        P(s)
P("  reader (all worlds):")
for k, c in t["reader (all worlds)"].items():
    s = f"    {k:32s} paired rows {c['rows (paired)']['n']:6d} max {c['rows (paired)']['max_ratio']} pass {c['rows (paired)']['pass']} | runs(mean) max {c['runs (mean)']['max_ratio']:.3g} pass {c['runs (mean)']['pass']}"
    if "runs (max)" in c:
        s += f" | runs(max) max {c['runs (max)']['max_ratio']:.3g} pass {c['runs (max)']['pass']}"
    s += f" | stored==record {c['stored row means == record run_scores (within 1e-9)']}"
    P(s)
P("  reader pairing:", json.dumps(t["reader pairing"]))
P("  honest_b gate (runs):")
for k, c in t["honest_b gate (runs)"].items():
    P(f"    {k:32s} runs {c['runs']} max|rec-mine f32| {c['max |record - mine f32|']:.3g} n>1e-6 {c['n |record - mine f32| > 1e-6']} | tol max ratio {c['tolerance (record vs exact)']['max_ratio']:.3g} pass {c['tolerance (record vs exact)']['pass']}")
P("\n[3] t16:", json.dumps(r["3_t16"], indent=1))
P("\n[4a] files:", json.dumps(r["4a_run_score_files"]))
f = r["4b_flip_counts"]
P("\n[4b] flips (per layout|probe|readout|alpha): eligible, threshold differs, then per arm flips / within d / within 2d")
for k, c in f["cells"].items():
    if c.get("incomplete"):
        P(f"  {k:48s} INCOMPLETE missing {c['missing keys (stored, mine)']}")
        continue
    arms = " ".join(f"{w}:{c[w]['hit status differs']}/{c[w]['within max|diff| of the stored threshold']}/{c[w]['within 2*max|diff| (upper bound on flips)']}"
                    for w in ("run1", "run2", "sim", "honest_b", "honest_c"))
    P(f"  {k:48s} elig {str(c['eligible (v6 frozen)'])[0]} pool {c['pool']:3s} thrdiff {c['threshold differs (stored vs mine)']:3s} {arms}")
P("  totals:", f["totals"])
