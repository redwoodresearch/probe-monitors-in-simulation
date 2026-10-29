"""Key reads (no computation) of the trap figure's numbers from the five readers' committed SCE metrics files.

For each reader, at its headline view and at 1% and 5% false alarms: the real-trained cell (a)'s real (second real run,
e2) catch, the mixed cell (c)'s real catch and simulated-attack catch, and (c)'s overstatement (simulated − real), each
with its stored 95% main-task bootstrap interval. Cell (c) = unrepaired simulated attacks vs real honest runs; readout =
output_end, the last content token of each tool output. Each input's sha256 must equal its committed manifest's.

  uv run --no-project --python 3.12 python extract_trap_numbers.py --dir <folder holding the five metrics files> --out trap_numbers.csv
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

# reader -> (file, sha256 from its manifest, (top key, view key), manifest@commit, view label)
SOURCES = {
    "Qwen3-32B": ("sce_metrics_POSTHOC_2026-10-13_123007.json", "0b567cc41221", ("layers", "63"),
                  "manifest_sce_results_v1.json @ 1911ca4e (aswin/q3-sce-reply-probe)", "layer 63 (#36's reply layer)"),
    "Qwen3-32B (layer 48, beside)": ("sce_metrics_POSTHOC_2026-10-13_123007.json", "0b567cc41221", ("layers", "48"),
                  "manifest_sce_results_v1.json @ 1911ca4e (aswin/q3-sce-reply-probe)", "layer 48"),
    "Qwen3.5-27B": ("sce_q35_metrics_POSTHOC_2026-10-13_131109.json", "2c0c2c7128f8", ("layers", "63"),
                    "manifest_sce_q35_results_v1.json @ 67e53532 (aswin/q3-sce-reply-probe)", "layer 63"),
    "Seed-OSS-36B": ("sce_seed_metrics_POSTHOC_2026-10-13_131217.json", "afbed348f78e", ("results", "g3_layer_per_fold"),
                     "manifest_sce_seed_results_v1.json @ 5140c1ca (aswin/q3-sce-seed-grid)", "its per-fold layers"),
    "Olmo-3.1-32B": ("sce_olmo_metrics_POSTHOC_2026-10-13_134244.json", "de48af4f3f2a", ("results", "reader_layer_per_fold"),
                     "manifest_sce_readers_results_v1.json @ 69a51ad7 (aswin/q3-sce-seed)", "its per-fold layers"),
    "Llama-3.3-70B": ("sce_llama_metrics_POSTHOC_2026-10-13_134251.json", "fff9fedfcaf5", ("results", "reader_layer_per_fold"),
                      "manifest_sce_readers_results_v1.json @ 69a51ad7 (aswin/q3-sce-seed)", "its per-fold layers"),
}
FIELDS = ["reader", "view", "rate", "quantity", "value", "ci_lo", "ci_hi", "file", "sha256_prefix", "manifest", "key"]


def main(dir_: str, out: str):
    rows = []
    for reader, (fname, sha_pre, (top, view), manifest, label) in SOURCES.items():
        p = Path(dir_) / fname
        if hashlib.sha256(p.read_bytes()).hexdigest()[:12] != sha_pre:
            raise SystemExit(f"{fname}: sha256 differs from its manifest")
        V = json.loads(p.read_text())[top][view]
        for rate in ("fpr_0.01", "fpr_0.05"):
            sim = V["c"]["sim_attacks_of_this_cell"]
            picks = [("real-trained (a): real catch", f"a/{rate}/catch/real-attack-e2", V["a"][rate]["catch"]["real-attack-e2"]),
                     ("mixed (c): real catch", f"c/{rate}/catch/real-attack-e2", V["c"][rate]["catch"]["real-attack-e2"]),
                     ("mixed (c): simulated catch", f"c/{rate}/catch/{sim}", V["c"][rate]["catch"][sim]),
                     ("mixed (c): overstatement (sim − real)", f"c/{rate}/sce_reported_minus_real", V["c"][rate]["sce_reported_minus_real"])]
            for q, key, x in picks:
                lo, hi = x["ci95"]
                rows.append({"reader": reader, "view": label, "rate": rate, "quantity": q, "value": x["value"], "ci_lo": lo,
                             "ci_hi": hi, "file": fname, "sha256_prefix": sha_pre, "manifest": manifest, "key": f"{top}/{view}/{key}"})
    with open(out, "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args().dir, ap.parse_args().out)
