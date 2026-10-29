"""Which entries of the amendment-40 rerun lost draws? (2026-10-08, saved outputs only, $0)

Reads the two current full outputs of analyze_phase2_fold_calibration_noise.py (per-task folds, 14e77d6; in .run_cache
and on S3) and prints each one's sha256. Their summaries, committed in 3d80227 and listed in
phase2_fold_calibration_a40_manifest.json, record the same values under full_output_sha256. For every
draw set x N x pairing x measure it lists the draws whose gap is undefined (NaN), and any summary value left undefined
(null). A40's decision entries are set 1's three catch@1% schemes and set 2's chosen scheme and partial AUROC 0-5%,
paired, at N = 60.

  uv run python experiments/2026-10-06_qwen35_replication/analyze_phase2_fold_calibration_invalid.py
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
FULL = ["phase2_fold_calibration_a40_first_2026-10-08_164346.json", "phase2_fold_calibration_a40_all_2026-10-08_164416.json"]


def main():
    for name in FULL:
        full = ROOT / ".run_cache" / "fold_calibration" / name
        d = json.loads(full.read_text())
        if d["script_commit"] != "14e77d609967c345529b557576b10d02e200d6a3":
            raise ValueError(f"{name}: not the current per-task run")
        print(f"== {name} (cells {d['cells']}, full-output sha256 {hashlib.sha256(full.read_bytes()).hexdigest()[:16]})")
        decision = {("set 1", m) for m in d["set 1 selection at N = 60 (catch@1% gap half-width)"]} | {
            ("set 2", m) for m in d["set 2 comparison at N = 60"]}
        lost = 0
        for st, by in d["by_n"].items():
            for key, row in by.items():
                for m, r in row.items():
                    if not isinstance(r, dict):
                        continue
                    nulls = [k for k, v in r.items() if v is None or (isinstance(v, list) and None in v)]
                    if r["invalid draws"] or nulls:
                        lost += 1
                        print(f"  {st} | {key} | {m}: {r['invalid draws']} of {d['draws_per_n_per_set']} draws undefined; null fields: {nulls}")
                    if key == "60 | paired" and (st, m) in decision:
                        print(f"  DECISION {st} | {key} | {m}: {r['invalid draws']} undefined draws")
        print(f"  entries with undefined draws or null fields: {lost}")


if __name__ == "__main__":
    main()
