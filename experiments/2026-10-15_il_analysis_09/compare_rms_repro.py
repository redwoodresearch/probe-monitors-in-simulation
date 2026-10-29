"""Pooled-RMS patch gate (09 via 2e, 10-15 18:0x): with spec.il.posthoc_pooled_rms absent, the patched run_il_posthoc.py must reproduce
the run-5 record's RMS rows exactly by key. Compares two records' il_posthoc_POST_HOC blocks leaf by leaf and writes paths only (never a value).

  python3 experiments/2026-10-15_il_analysis_09/compare_rms_repro.py <reference record> <new record> [--expect-extra=<key>,...]
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
extra_ok = set(next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--expect-extra=")), []))
ref_b, new_b = Path(args[0]).read_bytes(), Path(args[1]).read_bytes()
ref, new = json.loads(ref_b)["result"]["il_posthoc_POST_HOC"], json.loads(new_b)["result"]["il_posthoc_POST_HOC"]
diff, missing, extra, n, rel = [], [], [], [0], {}


def walk(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k not in b:
                missing.append(f"{path}.{k}")
            else:
                walk(a[k], b[k], f"{path}.{k}")
        extra.extend(f"{path}.{k}" for k in b if k not in a)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
    else:
        n[0] += 1
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            diff.append(path)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
                rel[path] = abs(a - b) / max(abs(a), abs(b), 1e-300)  # size of a difference only, never a value


if not all(isinstance(x.get(b_), dict) and x.get(b_) for x in (ref, new) for b_ in ("rms_bias", "rms_bias_per_action")):
    raise SystemExit("an RMS block is absent or empty in one of the records (Codex r2)")
blocks = {}
for blk in ("rms_bias", "rms_bias_per_action", "nm_yardstick", "five_arm_cells", "resolved_sources", "rms_bias_pooled_across_sets"):
    d0, m0, e0, n0 = len(diff), len(missing), len(extra), n[0]
    walk(ref.get(blk), new.get(blk), blk)
    rb_ = [rel.get(x) for x in diff[d0:]]
    blocks[blk] = {"leaves_compared": n[0] - n0, "differing": len(diff) - d0, "missing_in_new": len(missing) - m0, "extra_in_new": len(extra) - e0,
                   "differing_non_numeric": sum(r_ is None for r_ in rb_),
                   "max_relative_numeric_difference": max([r_ for r_ in rb_ if r_ is not None], default=0.0)}
unexpected_extra = [e for e in extra if e.rsplit(".", 1)[-1] not in extra_ok]
rms_exact = all(blocks[b]["differing"] == 0 and blocks[b]["missing_in_new"] == 0 for b in ("rms_bias", "rms_bias_per_action")) and not any(
    e.startswith(("rms_bias.", "rms_bias_per_action.")) for e in unexpected_extra)
new_top = sorted(set(new) - set(ref))
within_ = all(blocks[b_]["missing_in_new"] == 0 and blocks[b_]["differing_non_numeric"] == 0 and blocks[b_]["max_relative_numeric_difference"] <= 1e-12
              for b_ in ("rms_bias", "rms_bias_per_action")) and not any(e.startswith(("rms_bias.", "rms_bias_per_action.")) for e in unexpected_extra)
out = {"rms_rows_within_1e-12_relative": within_, "label": "pooled-RMS patch gate: reproduction of the reference record's POST HOC blocks by key (paths only; no values)",
       "reference": {"path": args[0], "sha256": hashlib.sha256(ref_b).hexdigest()}, "new": {"path": args[1], "sha256": hashlib.sha256(new_b).hexdigest()},
       "blocks": blocks, "rms_rows_exact": rms_exact, "expected_extra_keys": sorted(extra_ok), "unexpected_extra": unexpected_extra[:50],
       "differing_paths": diff[:50], "missing_paths": missing[:50], "new_top_level_keys": new_top,
       "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")}
b = json.dumps(out, indent=1).encode()
fn = HERE / "results" / f"pooled_rms_gate_{Path(args[1]).stem}_vs_{Path(args[0]).stem}_{out['stamp_pt']}.json"
fn.open("xb").write(b)
print(fn.name, hashlib.sha256(b).hexdigest(), json.dumps({"rms_rows_exact": rms_exact, "rms_rows_within_1e-12_relative": within_, "blocks": blocks}))
