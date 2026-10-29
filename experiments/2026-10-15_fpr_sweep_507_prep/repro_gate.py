"""Reproduction gate (final rebuild): a new record must equal the old one by key. Catch/gap summaries exact; threshold metadata (T, T_*) within
1e-12 (max diff stated). Ignored: meta, npz, resources and fields the new code adds (listed). Usage: repro_gate.py <new.json> <old.json> [--expect-changed <substring>] [--allow-new <substring>]
--expect-changed: leaves whose path holds the substring (e.g. "without S4", the registered rule's expected change @ 51a612b29) are reported with
counts (changed / unchanged), never compared exactly; every other leaf must still match."""
import json
import math
import sys

IGNORE_TOP = ("meta", "npz")
NEW_FIELDS = ("/resources", "/replacements", "/gate", "/pool_runs", "/calibration_digest", "/cohort_cell_keys")  # diagnostics added by later code;
# reported, not compared (calibration_digest / cohort_cell_keys: the without-S4 checks @ ae010b45a)


def flat(o, pre=""):
    if isinstance(o, dict):
        return {k2: v2 for k, v in o.items() for k2, v2 in flat(v, f"{pre}/{k}").items()}
    if isinstance(o, list):
        return {k2: v2 for i, v in enumerate(o) for k2, v2 in flat(v, f"{pre}[{i}]").items()}
    return {pre: o}


n, o = (json.load(open(p)) for p in sys.argv[1:3])
EXP = sys.argv[sys.argv.index("--expect-changed") + 1] if "--expect-changed" in sys.argv else None
NEWOK = sys.argv[sys.argv.index("--allow-new") + 1] if "--allow-new" in sys.argv else None  # a label the new record adds (e.g. own_all_LOO_L48)
if EXP == "" or NEWOK == "":
    sys.exit("empty --expect-changed / --allow-new substring")
new_allowed = 0
n_compared, nan_pairs = 0, 0  # leaves actually compared (after the resource, expected-change and NaN-pair skips); NaN pairs counted apart
exp_changed, exp_same, exp_paths = 0, 0, []
fa = flat({k: v for k, v in n.items() if k not in IGNORE_TOP})
fb = flat({k: v for k, v in o.items() if k not in IGNORE_TOP})
exact_bad, t_diffs, new_only, old_only, other = [], [], [], [], []
for p in sorted(set(fa) | set(fb)):
    if "/resources/" in p:  # container timing, peak memory and the requested resources: run facts, not results
        continue
    if p not in fb:
        if NEWOK and NEWOK in p:
            new_allowed += 1
            continue
        (new_only if any(x in p for x in NEW_FIELDS) else other).append(p)
        continue
    if p not in fa:
        old_only.append(p)
        continue
    a, b = fa[p], fb[p]
    if EXP and (EXP in p or EXP.replace(" ", "_") in p):
        same = a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
        exp_same += same
        exp_changed += not same
        if not same and len(exp_paths) < 20:
            exp_paths.append(p)
        continue
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        nan_pairs += 1
        continue
    n_compared += 1
    if a == b:
        continue
    last = p.rsplit("/", 1)[-1]
    if isinstance(a, float) and isinstance(b, float) and (last == "T" or last.startswith("T_")):
        if math.isnan(a) != math.isnan(b):  # a NaN on one side only is a mismatch (abs(nan) > tol is False)
            exact_bad.append(p)
            continue
        t_diffs.append(abs(a - b))
        if abs(a - b) > 1e-12:
            exact_bad.append(p)
    else:
        exact_bad.append(p)
res = {"new": sys.argv[1], "old": sys.argv[2], "leaves_compared": len(set(fa) & set(fb)), "mismatch": exact_bad[:40], "n_mismatch": len(exact_bad),
       "threshold_diffs": len(t_diffs), "threshold_max_abs_diff": max(t_diffs) if t_diffs else 0.0, "new_only_fields": len(new_only),
       "new_only_unexpected": other[:20], "old_only": old_only[:20],
       "leaves_actually_compared": n_compared, "nan_pairs": nan_pairs,
       "pass": not exact_bad and not other and not old_only and n_compared > 0 and (not NEWOK or new_allowed > 0),
       "expected_change": ({"substring": EXP, "changed_leaves": exp_changed, "unchanged_leaves": exp_same, "examples": exp_paths} if EXP else None),
       "allowed_new_label": ({"substring": NEWOK, "new_leaves": new_allowed} if NEWOK else None)}
print(json.dumps(res, indent=1))
sys.exit(0 if res["pass"] else 1)
