"""Revision 6s reproduction check: every leaf of the FINAL record's `result` (6r @ 3ce6ea55e, sha256 7dceb8bb…) must exist
in the 6s record's `result` with the same value (numbers within 1e-12, NaN == NaN; strings, booleans and None equal).
Allowed to differ: result.record.code_sha256 and result.record.rule_sha256 (provenance of the new code). Prints counts
and mismatching key paths only, never a value. Modal CPU.

  CHK_A=<final record json> CHK_B=<6s record json> $MODAL run experiments/2026-10-14_xerr_scaleup_09/check_repro_rev6s.py
"""

import hashlib
import json
import os
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-convaudit_xerrcheck_kill-2026-11-14"
A, B = Path(os.environ.get("CHK_A", "/nonexistent")), Path(os.environ.get("CHK_B", "/nonexistent"))
ALLOWED = {("record", "code_sha256"), ("record", "rule_sha256")}
TOL = 1e-12
FINAL_SHA256 = "7dceb8bb25752aa3619cfde64add702fd4f7d13a1309a56dbc8880a6b27af9f5"  # 6r @ 3ce6ea55e

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")
if A.exists() and B.exists():
    image = image.add_local_file(A, "/root/a.json").add_local_file(B, "/root/b.json")


@app.function(image=image, cpu=2, memory=8192, timeout=900, retries=0)
def compare() -> dict:
    import math

    raw_a, raw_b = Path("/root/a.json").read_bytes(), Path("/root/b.json").read_bytes()
    if hashlib.sha256(raw_a).hexdigest() != FINAL_SHA256:  # Codex r1 optional: the baseline is pinned
        return {"MATCH": False, "error": "CHK_A is not the pinned FINAL 6r record"}
    if raw_a == raw_b:
        return {"MATCH": False, "error": "CHK_B is byte-identical to CHK_A"}
    a, b = json.loads(raw_a)["result"], json.loads(raw_b)["result"]
    out = {"leaves": 0, "numeric": 0, "max_abs_diff": 0.0, "missing": [], "type_mismatch": [], "value_mismatch": []}

    def walk(x, y, path):
        if path in ALLOWED:
            return
        if isinstance(x, dict):
            if not isinstance(y, dict):
                out["type_mismatch"].append("/".join(path))
                return
            for k, v in x.items():
                if k not in y:
                    out["missing"].append("/".join(path + (k,)))
                else:
                    walk(v, y[k], path + (k,))
            return
        if isinstance(x, list):
            if not isinstance(y, list) or len(x) != len(y):
                out["type_mismatch"].append("/".join(path))
                return
            for i, (v, w) in enumerate(zip(x, y, strict=True)):
                walk(v, w, path + (str(i),))
            return
        out["leaves"] += 1
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
        isint = lambda v: isinstance(v, int) and not isinstance(v, bool)  # noqa: E731
        if num(x) and num(y) and isint(x) != isint(y):  # Codex r2: an int/float type change is a mismatch
            out["value_mismatch"].append("/".join(path))
        elif isint(x) and isint(y):
            out["numeric"] += 1
            if x != y:  # exact for integers (Codex r1 optional)
                out["value_mismatch"].append("/".join(path))
        elif num(x) and num(y):
            out["numeric"] += 1
            if math.isnan(x) or math.isnan(y):
                if not (math.isnan(x) and math.isnan(y)):
                    out["value_mismatch"].append("/".join(path))
                return
            d = abs(x - y)
            out["max_abs_diff"] = max(out["max_abs_diff"], d)
            if d > TOL:
                out["value_mismatch"].append("/".join(path))
        elif type(x) is not type(y) or x != y:
            out["value_mismatch"].append("/".join(path))

    if not a:
        return {"MATCH": False, "error": "the FINAL result is empty"}
    walk(a, b, ())
    out["has_rev6s_key"] = "append21j_2b" in ((b.get("results") or {}).get("R|0.05") or {}).get("primary", {})
    n_new = sum(1 for k in b if k not in a)
    out |= {
        "sha256_a": hashlib.sha256(raw_a).hexdigest(),
        "sha256_b": hashlib.sha256(raw_b).hexdigest(),
        "new_top_level_keys_in_b": n_new,
        "MATCH": not (out["missing"] or out["type_mismatch"] or out["value_mismatch"]) and out["has_rev6s_key"],
    }
    for k in ("missing", "type_mismatch", "value_mismatch"):
        out[f"n_{k}"] = len(out[k])
        out[k] = out[k][:200]
    return out


@app.local_entrypoint()
def main():
    res = compare.remote()
    keep = ("MATCH", "error", "has_rev6s_key", "leaves", "numeric", "n_missing", "n_type_mismatch", "n_value_mismatch", "missing",
            "type_mismatch", "value_mismatch", "sha256_a", "sha256_b", "new_top_level_keys_in_b")
    out = {k: res[k] for k in keep if k in res}  # Codex r1 item 4: counts and key paths only, never a value
    out["float_max_abs_diff_within_1e-12"] = (res["max_abs_diff"] <= TOL) if "max_abs_diff" in res else None
    print(json.dumps(out, indent=1))
    if not res.get("MATCH"):
        raise SystemExit("NOT MATCH")
