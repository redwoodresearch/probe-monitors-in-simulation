"""Cell (ii), df's two "why" steps, Qwen3-32B, cell (c). Values print as stored, not in points: fractions,
ratios and (Step 2) probe scores.

Step 1, the placeholder world, catch at 1% false alarms: the overstatement (fake catch − real second-epoch catch, a
fraction) of the fix-full ("ff") and placeholder ("ph", PRIMARY: L48 ph_call_end_mff) probes at the call_end
content-only readout, and the ratio S1 = placeholder overstatement / fix-full overstatement. S1 is read only if the
frozen rule says so (`read`).
Step 2, the simulator-rewritten honest runs (primary view "changed"; df's score_step2.py @ 6f77682c), in probe-score
units, not catch: delta = mean over the retained pairs of (rewritten − original) run score, its 95% interval from
2,000 resamples of those pairs; G = mean simulated-honest − mean real-honest out-of-fold run score; S2 = delta / G, a
ratio, whose interval is delta's interval / G with G held fixed.

Sources (git, sha256-checked; df's branch aswin/why-steps, folder experiments/2026-10-13_why_steps/results/):
- Step 1: why_metrics_steps_2026-10-13_175358.json @ 43ff8f69;
- Step 2: why_step2_2026-10-13_182312.json @ 6f77682c.
Run from the repo after `git fetch origin aswin/why-steps`: python3 -I cell_ii_why_steps.py
"""

import hashlib
import json
import subprocess

R = "experiments/2026-10-13_why_steps/results"
STEP1 = ("43ff8f69ee479cb5540c4b9765e7d87c0528f510", f"{R}/why_metrics_steps_2026-10-13_175358.json",
         "746d628c06a872b040da7405f4e9e2c351f3c25f5cd6ed77f0ab5577d940e4c9")
STEP2 = ("6f77682c32599f376daa9f11f9a5c4b2030d489d", f"{R}/why_step2_2026-10-13_182312.json",
         "51a24934269a7ec87a2f9092c160466b2befde6fad7e0b4dacce7d4fee527670")


def git_json(commit, path, sha):
    b = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"sha256 mismatch: {commit[:8]}:{path}")
    return json.loads(b)


def get(d, key):
    for k in key.split("/"):
        d = d[k]
    return d


def show(d, key):
    x = get(d, key)
    lo, hi = x["ci95"]
    print(f"  {key:66s} {x['value']:.4f} [{lo:.4f}, {hi:.4f}]")


def main():
    s1, s2 = git_json(*STEP1), git_json(*STEP2)
    print("sha256 OK for both files\n")
    print(f"STEP 1  {STEP1[1]} @ {STEP1[0][:8]}")
    for row in ("L48__ff_call_end__c", "L63__ff_call_end__c", "L48__ph_call_end_mff__c", "L63__ph_call_end_mff__c"):
        show(s1, f"rows/{row}/fpr_0.01/overstatement_fake_minus_e2")
    for layer in ("L48", "L63"):
        show(s1, f"rows/{layer}__S1/fpr_0.01")
        read = get(s1, f"rows/{layer}__S1/read")
        print(f"  rows/{layer}__S1/read = {read}" + ("" if read else "  -> S1 not read under the frozen rule"))
    print(f"\nSTEP 2  {STEP2[1]} @ {STEP2[0][:8]}")
    for ro in ("ff_call_end_L48_c", "ff_call_end_L63_c", "ff_output_end_L63_c"):
        print(f"  {'readouts/' + ro + '/changed/pairs':66s} {get(s2, f'readouts/{ro}/changed/pairs')}")
        show(s2, f"readouts/{ro}/changed/delta")
        show(s2, f"readouts/{ro}/changed/S2")
        print(f"  {'readouts/' + ro + '/G':66s} {get(s2, f'readouts/{ro}/G'):.4f}")
        d, g = get(s2, f"readouts/{ro}/changed/delta/value"), get(s2, f"readouts/{ro}/G")
        print(f"  check: delta / G = {d / g:.4f} (approximate: the file rounds delta, G and S2 to 4 decimals each)")


if __name__ == "__main__":
    main()
