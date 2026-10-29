"""Unmatched (v3) -> length-matched table by key, from the committed records only (no statistics computed here).
Usage (repo root): python3 -I experiments/2026-10-15_loro_sep_09/compare_runlen.py"""

import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
V3 = {"311": "loro_sep311v3_CR_2026-10-15_162243.json", "455": "loro_sep455v3_CR_2026-10-15_162243.json"}
VAR = {"cap30": "loro_runlen_cap30_CR_2026-10-15_165508.json", "captaskmin": "loro_runlen_captaskmin_CR_2026-10-15_165527.json",
       "cappaired": "loro_runlen_cappaired_CR_2026-10-15_165436.json"}


def f2(x):
    return "-" if x is None else f"{x:+.2f}"


def rg(r):
    return "-" if not isinstance(r, list) else f"[{r[0]:+.2f}, {r[1]:+.2f}]"


def nread(c, origins):
    return sum(1 for v in c["readability_by_member_reported_only"].values() if any(v[o]["readable_contrast"] is False for o in origins))


for nm, fn in {**V3, **VAR}.items():
    print(f"# {fn} sha256 {hashlib.sha256((R / fn).read_bytes()).hexdigest()}")
for entry, origins in (("311", ("S_new",)), ("455", ("S_old", "S_new"))):
    base = json.loads((R / V3[entry]).read_text())["cells"]
    print(f"\n== {'+311 (S_new)' if entry == '311' else 'pooled 455'}: cell | variant | Sol (a) group d0 | 6-group m [95% CI] | tau2 | (b) range new-recipe mean | true-gap range | LOO | |D| | not readable | c_asis")
    for ro in ("R", "C"):
        for f in ("0.01", "0.001", "0.05"):
            for inp in ("primary", "yardstick"):
                key = f"{ro}|{f}|{inp}"
                rows = [("v3 unmatched", base[key])]
                for var, fn in VAR.items():
                    cells = json.loads((R / fn).read_text())["entries"][entry]["cells"]
                    if key in cells:
                        rows.append((var, cells[key]))
                for tag, c in rows:
                    st = c["stress"][0] if c.get("stress") else None
                    print(f"{key} | {tag} | {f2(c['group_gaps_draw0'].get('other-model-real'))} | {f2(c.get('a_mean_signed_gap'))} {rg(c.get('a_mean_signed_gap_ci95'))} | {f2(c.get('tau2'))}"
                          f" | {rg(c.get('b_range_new_recipe_mean_observed_independent_cohort'))} | {rg(c.get('b_range_new_recipe_mean_true_gap'))} | {c.get('loo_inside')} | {c.get('D')}"
                          f" | {nread(c, origins)} | {(f2(st['gap']) + ' ' + st['position_vs_b_observed'] + '/' + st['position_vs_b_true']) if st else '-'}")
print("\n== +311 per-group draw-0 Delta_full at R 0.01 and R 0.05 (v3 unmatched -> cap30 -> captaskmin -> cappaired)")
groups = ["other-model-real", "own-real", "public:FIBS", "public:harmfulness", "public:high-stakes", "public:refusal"]
for key in ("R|0.01|primary", "R|0.05|primary", "C|0.01|primary", "C|0.05|primary"):
    vals = [json.loads((R / V3["311"]).read_text())["cells"][key]["group_gaps_draw0"]] + [json.loads((R / fn).read_text())["entries"]["311"]["cells"][key]["group_gaps_draw0"] for fn in VAR.values()]
    print(key + " | " + " | ".join(f"{g}: " + " -> ".join(f2(v[g]) for v in vals) for g in groups))
