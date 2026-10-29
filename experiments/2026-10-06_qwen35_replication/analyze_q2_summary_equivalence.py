"""Does this checkout's Q2 summary reproduce a committed Q2 result exactly (2026-10-07)?

The efficiency changes to run_q2_alllayers_qwen35.py (container sizes, each fit file read once in the summary) must not
change a number. This reruns `summary` from this checkout on a committed features run's saved fits (its
fits_index.jsonl and the config stored in its result.json) and compares the output with that result.json, key by key
after a JSON round trip. Modal CPU only (the summary container: 32 cores, about 15 minutes); no fits, no GPU.

  uv run python experiments/2026-10-06_qwen35_replication/analyze_q2_summary_equivalence.py [--features-run ...] [--out-sub threeway]
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_q2_alllayers_qwen35 as q2  # noqa: E402


def compare(a, b, path: list, out: dict):
    """Walk two JSON values: the largest absolute difference of each top-level key's numbers, and every place where
    the structure, a non-number or a NaN pattern differs."""
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in a and k in b:
                compare(a[k], b[k], path + [k], out)
            else:
                out["mismatches"].append(path + [k])
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + [i], out)
    elif num(a) and num(b):
        top = out["keys"].setdefault(path[0], {"numbers": 0, "max_abs_diff": 0.0})
        top["numbers"] += 1
        if math.isnan(a) or math.isnan(b):
            if not (math.isnan(a) and math.isnan(b)):
                out["mismatches"].append(path)
        else:
            top["max_abs_diff"] = max(top["max_abs_diff"], abs(a - b))
    elif a != b:
        out["mismatches"].append(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-run", default="2026-10-06_q2_qwen35_matched_eb014307fd1aab31_as_labeled")
    ap.add_argument("--out-sub", default="threeway")
    a = ap.parse_args()
    base = f"{a.features_run}/{a.out_sub}"
    committed = json.loads(b"".join(q2.feats_vol.read_file(f"{base}/result.json")))
    fits = [json.loads(l) for l in b"".join(q2.feats_vol.read_file(f"{base}/fits_index.jsonl")).decode().splitlines()]
    cfg = committed["config"]
    with q2.app.run():
        out = json.loads(json.dumps(q2.summary.remote(cfg, fits)))
    keys = sorted(set(committed) - {"config", "index"})  # every summary key the committed result has (orchestrate adds those two)
    cmp = {"keys": {}, "mismatches": [[k] for k in sorted(set(out) ^ set(keys))]}
    for k in keys:
        if k in out:
            compare(committed[k], out[k], [k], cmp)
    identical = not cmp["mismatches"] and all(v["max_abs_diff"] == 0 for v in cmp["keys"].values())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q2_summary_equivalence_{stamp}.json"
    path.write_text(json.dumps({"features_run": a.features_run, "out_sub": a.out_sub, "fits": len(fits), "identical": identical,
                                "per_key": cmp["keys"], "mismatches": cmp["mismatches"][:100],
                                "n_mismatches": len(cmp["mismatches"])}, indent=1))
    for k, v in cmp["keys"].items():
        print(f"{k:13} {v['numbers']:8} numbers, max |difference| {v['max_abs_diff']:.3g}")
    print(f"{base}: {len(fits)} fits; {len(cmp['mismatches'])} structural or non-numeric mismatches; "
          f"{'IDENTICAL' if identical else 'DIFFERENT'} -> {path}")


if __name__ == "__main__":
    main()
