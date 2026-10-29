"""Cell (i), the trap: how much the mixed recipe (cell (c): probe trained on simulated attack vs real honest) overstates
real catch, per reader. Overstatement = catch of simulated attacks − catch of real second-epoch attacks, at the 1%
false-alarm threshold, 95% main-task bootstrap interval. Claimed: 52–79 points across readers.

Sources: five POSTHOC metrics files, S3 only. Each sha256 below is copied from its manifest, in
experiments/2026-10-13_q3_sce_reply_probe/results/: manifest_sce_results_v1.json, manifest_sce_q35_results_v1.json and
manifest_sce_readers_results_v1.json @ origin/aswin/q3-sceg-9-results; manifest_seed_variants_s3_v1.json @
origin/aswin/q3-sce-seed-grid. Fetch each file into one folder:
  aws s3 cp --profile redwood-sandbox --region us-east-2 \
    s3://individual-aswin-surya-probe-shift-kill-2027-04-06/experiments/2026-10-13_q3_sce_reply_probe/results/<file> <dir>/
Run: python3 -I cell_i_trap.py <dir>
"""

import hashlib
import json
import sys
from pathlib import Path

FILES = {"sce_metrics_POSTHOC_2026-10-13_123007.json": "0b567cc412212b7255fcfcff8c6aba722825d8d935f822bdd69d669c46854974",
         "sce_q35_metrics_POSTHOC_2026-10-13_131109.json": "2c0c2c7128f863b0e6e9be6733bc4d9ba35f5b5e456be67bf5b581efcfcfca27",
         "sce_olmo_metrics_POSTHOC_2026-10-13_134244.json": "de48af4f3f2accf44247746a6a7bde59105ecf7fb1cf6706045d50c01fda4ab3",
         "sce_llama_metrics_POSTHOC_2026-10-13_134251.json": "fff9fedfcaf5e6561eb28c1cd02ce013f2c6eea2d4534f0ab3c72299b2bdab56",
         "sce_seed_variants_metrics_POSTHOC_2026-10-13_164540.json": "a0c85ed61899b6c8d31f6079eb006517f4f569f9e9859284a58beb4127783e79"}
OVER = "c/fpr_0.01/sce_reported_minus_real"  # a {"value", "ci95"} dict, stored as fractions
ROWS = [  # (reader and layer, file, key)
    ("Qwen3-32B, layer 48", "sce_metrics_POSTHOC_2026-10-13_123007.json", f"layers/48/{OVER}"),
    ("Qwen3-32B, layer 63", "sce_metrics_POSTHOC_2026-10-13_123007.json", f"layers/63/{OVER}"),
    ("Qwen3.5-27B, layer 63", "sce_q35_metrics_POSTHOC_2026-10-13_131109.json", f"layers/63/{OVER}"),
    ("Olmo-3.1-32B, its layer per fold", "sce_olmo_metrics_POSTHOC_2026-10-13_134244.json", f"results/reader_layer_per_fold/{OVER}"),
    ("Olmo-3.1-32B, layer 56 (beside)", "sce_olmo_metrics_POSTHOC_2026-10-13_134244.json", f"results/L56/{OVER}"),
    ("Llama-3.3-70B, its layer per fold", "sce_llama_metrics_POSTHOC_2026-10-13_134251.json", f"results/reader_layer_per_fold/{OVER}"),
    ("Llama-3.3-70B, layer 60 (beside)", "sce_llama_metrics_POSTHOC_2026-10-13_134251.json", f"results/L60/{OVER}"),
    ("Seed-OSS-36B, its layer per fold", "sce_seed_variants_metrics_POSTHOC_2026-10-13_164540.json", f"results/g3_layer_per_fold/{OVER}"),
    ("Seed-OSS-36B, layer 63 (beside)", "sce_seed_variants_metrics_POSTHOC_2026-10-13_164540.json", f"results/L63/{OVER}")]


def load(path, sha):
    b = Path(path).read_bytes()
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"sha256 mismatch: {path}")
    return json.loads(b)


def get(d, key):
    for k in key.split("/"):
        d = d[k]
    return d


def main(folder):
    data = {f: load(Path(folder) / f, sha) for f, sha in FILES.items()}
    print("sha256 OK for all 5 files\n")
    vals = []
    for name, f, key in ROWS:
        x = get(data[f], key)
        vals.append(x["value"])
        lo, hi = x["ci95"]
        print(f"{name:36s} {100 * x['value']:5.1f} [{100 * lo:.1f}, {100 * hi:.1f}]   {f} :: {key}")
    print(f"\nrange across the rows above: {100 * min(vals):.1f} to {100 * max(vals):.1f} points")


if __name__ == "__main__":
    main(sys.argv[1])
