"""POST HOC (0d's request after the k = 1 result, 10-11; not the committed rule), $0 from the stored per-site scores
(claimB/k1_v1/scores_<tag>.json; numbers only): 198 of 400 simulated replies are byte-identical to the real ones after
the fixes and the level-1 scrub, so they enter Δ1 as zero shifts. Splits the sites by that flag:
  1. Δ1, G and R1 = Δ1 / G on the non-identical sites only;
  2. G on the identical sites (and on the non-identical ones), and the difference G(non-identical) − G(identical);
  3. counts of identical vs non-identical sites per environment and per tool.
Intervals: 2,000 main-task draws (default_rng(20261011), a fresh generator per quantity), resampling the main tasks of
the rows the quantity uses (for the difference, all 400 sites' tasks jointly), as analyze_k1.py. L63 and step A's layer.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python \\
      experiments/2026-10-11_claimB_k_replies/analyze_POSTHOC_k1_identical.py --tag 4be7958c7157
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
OUT, DRAWS, SEED = "claimB/k1_v1", 2000, 20261011


def boot(rows, stat):
    groups = defaultdict(list)
    for r in rows:
        groups[r["main_task"]].append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(DRAWS):
        v = stat([r for k in rng.choice(keys, len(keys)) for r in groups[k]])
        if v is not None:
            vals.append(v)
    return ([float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]), DRAWS - len(vals)


def d1(rows, w):
    return float(np.mean([r[w]["sim"] - r[w]["self"] for r in rows])) if rows else None


def g(rows, w):
    return float(np.mean([r[w]["stored_fixed_sim"] - r[w]["stored_e1"] for r in rows])) if rows else None


def r1(rows, w):
    gg = g(rows, w)
    return d1(rows, w) / gg if rows and gg > 0 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    raw = b"".join(vol.read_file(f"{OUT}/scores_{a.tag}.json"))
    sites = json.loads(raw)["sites"]
    ident = [r for r in sites if r["sim_text_equals_self_text"]]
    diff = [r for r in sites if not r["sim_text_equals_self_text"]]
    out = {"label": "POST HOC (not the committed rule): k = 1 split by identical simulated replies", "sites": len(sites),
           "identical": len(ident), "non_identical": len(diff), "inputs": {f"scores_{a.tag}.json": hashlib.sha256(raw).hexdigest()}}
    for w in ("63", "A"):
        x = {}
        x["non_identical"] = {"delta1": d1(diff, w), "delta1_ci95": boot(diff, lambda rr: d1(rr, w))[0], "G": g(diff, w),
                              "G_ci95": boot(diff, lambda rr: g(rr, w))[0], "R1": r1(diff, w)}
        x["non_identical"]["R1_ci95"], x["non_identical"]["draws_with_nonpositive_G"] = boot(diff, lambda rr: r1(rr, w))
        x["identical"] = {"delta1": d1(ident, w), "G": g(ident, w), "G_ci95": boot(ident, lambda rr: g(rr, w))[0]}

        def gap_diff(rr, w=w):
            a_, b_ = [r for r in rr if not r["sim_text_equals_self_text"]], [r for r in rr if r["sim_text_equals_self_text"]]
            return g(a_, w) - g(b_, w) if a_ and b_ else None
        x["G_non_identical_minus_identical"] = {"value": gap_diff(sites), "ci95": boot(sites, gap_diff)[0]}
        out[f"layer_{w}"] = x
    out["by_environment"] = {e: {"identical": c[True], "non_identical": c[False]} for e, c in sorted(
        {e: Counter(r["sim_text_equals_self_text"] for r in sites if r["environment"] == e) for e in {r["environment"] for r in sites}}.items())}
    out["by_tool"] = {t: {"identical": c[True], "non_identical": c[False]} for t, c in sorted(
        {t: Counter(r["sim_text_equals_self_text"] for r in sites if r["tool"] == t) for t in {r["tool"] for r in sites}}.items())}
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"k1_identical_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("sites", "identical", "non_identical", "layer_63", "layer_A")}, indent=1))
    print("by_environment", json.dumps(out["by_environment"]))
    print("by_tool", json.dumps(out["by_tool"]))
    print("->", dst)


if __name__ == "__main__":
    main()
