"""A2: does C's scoring path (run_POSTHOC_score_honest_c.py) reproduce run_score_onpolicy.honest_scores exactly? Before any
C score is read, B's runs of one existing extraction are scored through C's path (`--check`) and compared, call by
call, with honest_scores.json's B rows from that extraction: the same calls in the same order, and the same five
fold-model scores. $0, local; scores and keys only.

  uv run --no-project --python 3.12 --with numpy --with pyyaml --with 'modal>=1.4.3' python analyze_POSTHOC_honest_c_scorer_check.py \\
      <honest_scores.json> <honest_scores_Ccheck_<x>.json> <acts meta root> <extraction x>
"""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from analyze_honest_ab_audit import local_load
from run_POSTHOC_score_honest_c import c_config
from run_score_onpolicy import frozen_folds, rows_sha

HERE = Path(__file__).parent


def main(honest_path: str, check_path: str, root: str, x: str):
    raw_h, raw_c = Path(honest_path).read_bytes(), Path(check_path).read_bytes()
    H, C = json.loads(raw_h), json.loads(raw_c)
    if C["key"] != H["key"] or C["choices"] != H["choices"] or C["c_extraction"] != x:
        raise SystemExit("the check was not scored with honest_scores.json's models, or not from this extraction")
    ext = yaml.safe_load((HERE / "config" / "score_scaleup196_simhon.yaml").read_text())
    folds_full = frozen_folds(HERE.parents[1] / ext["pilot_tasks"])
    if any(folds_full[r["task"]] != r["fold"] for r in H["rows"]):
        raise SystemExit("honest_scores.json's folds differ from the frozen folds")
    ab = [r for r in local_load(ext, folds_full, Path(root)) if (r[0] in ("honA", "honB") and r[5] == 0) or r[0] == "simH"]
    if [[r[0], r[1], r[5]] for r in ab] != [[h["world"], h["task"], h["y"]] for h in H["rows"]]:
        raise SystemExit("honest_scores.json's rows differ from the local join")
    ref = [(r[1], r[6], r[8], h["s"]) for r, h in zip(ab, H["rows"]) if r[0] == "honB" and r[2] == x]
    cr = [r for r in local_load(c_config(ext, x), folds_full, Path(root)) if r[0] == "honB" and r[5] == 0]
    if len(cr) != len(C["rows"]):
        raise SystemExit(f"the check file has {len(C['rows'])} rows, the local join {len(cr)}")
    # the check file's own identities: its stored task, fold and label per row, and the hash of the ordered activation
    # rows it scored, recomputed from the local join (load()'s src_rows format)
    if [[c["task"], c["fold"], c["y"]] for c in C["rows"]] != [[r[1], folds_full[r[1]], r[5]] for r in cr]:
        raise SystemExit("the check file's task, fold or label sequence differs from the local join")
    src = [(r[0], r[1], ("acts", r[2]), r[3], r[4], r[5], r[7]) for r in cr]
    if C.get("rows_sha256") != rows_sha(src):
        raise SystemExit("the check file's rows hash differs from the local join's")
    for rows in (C["rows"], [h for h in H["rows"]]):
        if any(len(r["s"]) != 5 or any(v != v or v in (float("inf"), float("-inf")) for v in r["s"]) for r in rows):
            raise SystemExit("a row without exactly five finite scores")
    got = [(r[1], r[6], r[8], c["s"]) for r, c in zip(cr, C["rows"])]
    same_calls = [g[:3] for g in got] == [f[:3] for f in ref]
    diff = max((abs(a - b) for g, f in zip(got, ref) for a, b in zip(g[3], f[3])), default=None) if same_calls else None
    out = {"label": "A2: C's scoring path against honest_scores (equivalence check, before any C score)",
           "inputs": {Path(honest_path).name: hashlib.sha256(raw_h).hexdigest(), Path(check_path).name: hashlib.sha256(raw_c).hexdigest()},
           "extraction": x, "rows_reference": len(ref), "rows_check": len(got), "tasks": len({g[0] for g in got}),
           "same_calls_in_order": same_calls, "rows_sha256_matches": True, "max_abs_score_difference": diff,
           "precision": "scores as stored, rounded to 6 decimals", "passed": bool(same_calls and diff == 0.0)}
    print(json.dumps(out))
    if not out["passed"]:
        raise SystemExit("C's path does not reproduce honest_scores exactly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("honest_path", "check_path", "root", "x"):
        ap.add_argument(a)
    a = ap.parse_args()
    main(a.honest_path, a.check_path, a.root, a.x)
