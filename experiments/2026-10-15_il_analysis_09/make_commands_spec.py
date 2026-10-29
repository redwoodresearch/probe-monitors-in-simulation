"""Spec for the stage B posthoc rerun with 09's POST HOC commands row (c68587df6): the stage B spec (sha-checked) plus
il.posthoc_commands_all_readable = true and stage_b_meta.commands_row (rule, readability record, code commit). By key, nothing else may differ.

  python3 -I experiments/2026-10-15_il_analysis_09/make_commands_spec.py --code-commit <sha> --out specs/<name>.json
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = ("specs/il_posthoc_spec_stageB_wtpooled_2026-10-15_221909.json", "26a406c3e8ca96aa83fbb0506d852661b2682bcc88ef376d5997ba4ff790e311")
READ = ("results/readability_commands_2026-10-15_225321.json", "8795cb3479e876ea6f52195e99e5d1b50db1f612a688da2cdf00bda0cc1795bb")
ap = argparse.ArgumentParser()
ap.add_argument("--code-commit", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
bb = (HERE / BASE[0]).read_bytes()
rb = (HERE / READ[0]).read_bytes()
if hashlib.sha256(bb).hexdigest() != BASE[1] or hashlib.sha256(rb).hexdigest() != READ[1]:
    raise SystemExit("stage B spec or readability record sha256 differs")
code_ = (HERE / "run_il_posthoc.py").read_bytes()
if code_ != subprocess.run(["git", "show", f"{a.code_commit}:experiments/2026-10-15_il_analysis_09/run_il_posthoc.py"], capture_output=True, check=True, cwd=REPO).stdout:
    raise SystemExit("run_il_posthoc.py differs from --code-commit")
base_, spec = json.loads(bb), json.loads(bb)
if not spec["il"].get("posthoc_pooled_rms"):
    raise SystemExit("the base spec is not in pooled mode")
spec["il"]["posthoc_commands_all_readable"] = True
spec["stage_b_meta"]["commands_row"] = {"rule": "09 RULE @ c68587df6 (POST HOC all-readable-probes pooled RMS row on commands at per-action 1e-2)",
                                        "readability_record": f"experiments/2026-10-15_il_analysis_09/{READ[0]} sha256 {READ[1]}",
                                        "code_commit": a.code_commit, "code_sha256": hashlib.sha256(code_).hexdigest(),
                                        "base_stage_b_spec": f"experiments/2026-10-15_il_analysis_09/{BASE[0]} sha256 {BASE[1]}"}


def diffs(x, y, p=()):
    if isinstance(x, dict) and isinstance(y, dict):
        o = []
        for k in set(x) | set(y):
            o += [p + (k,)] if (k in x) != (k in y) else diffs(x[k], y[k], p + (k,))
        return o
    return [] if json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) else [p]


bad = [d for d in diffs(base_, spec) if d not in (("il", "posthoc_commands_all_readable"), ("stage_b_meta", "commands_row"))]
if bad:
    raise SystemExit(f"BUILD ERROR: differs from the stage B spec outside the two allowed keys: {bad[:5]}")
Path(HERE / a.out).open("x").write(json.dumps(spec, indent=1) + "\n")
print(a.out, hashlib.sha256((HERE / a.out).read_bytes()).hexdigest())
