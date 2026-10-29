"""Regression check (JSON reads only): two PANEL records from the same config must be byte-identical outside meta (git, stamp, script sha256); captions,
title and figure too. Writes results/regression_<stamp>.json (exclusive create).

  python3 experiments/2026-10-15_panel_noise_matched/check_regression.py <baseline record> <new record> [added key names, comma-separated]
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
A, B = (Path(x) for x in sys.argv[1:3])
ADDED = sys.argv[3].split(",") if len(sys.argv) > 3 else []  # key names the change ADDS (e.g. Y_tauB): removed from the new record wherever the baseline lacks them
a, b = json.loads(A.read_bytes()), json.loads(B.read_bytes())
n_added = [0]


def drop_added(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in [k for k in y if k in ADDED and k not in x]:
            del y[k]
            n_added[0] += 1
        for k in x:
            if k in y:
                drop_added(x[k], y[k])
    elif isinstance(x, list) and isinstance(y, list):
        for u, v in zip(x, y):
            drop_added(u, v)


drop_added(a, b)
body = lambda r: json.dumps({k: v for k, v in r.items() if k != "meta"}).encode()
cap = lambda p: json.loads(p.with_name(p.stem + "_captions.json").read_bytes())
out = {"label": "REGRESSION TEST (not a result): same config, code before vs after a change",
       "baseline": {"path": str(A), "sha256": hashlib.sha256(A.read_bytes()).hexdigest(), "git": a["meta"]["git"]},
       "new": {"path": str(B), "sha256": hashlib.sha256(B.read_bytes()).hexdigest(), "git": b["meta"]["git"]},
       "top_keys_same_order": list(a) == list(b), "differing_top_keys": [k for k in a if json.dumps(a[k]) != json.dumps(b.get(k))],
       "differing_meta_keys": [k for k in a["meta"] if json.dumps(a["meta"][k]) != json.dumps(b["meta"].get(k))],
       "non_meta_bytes_equal": body(a) == body(b), "non_meta_sha256": [hashlib.sha256(body(a)).hexdigest(), hashlib.sha256(body(b)).hexdigest()],
       "captions_and_title_equal": cap(A)["captions"] == cap(B)["captions"] and cap(A)["title"] == cap(B)["title"],
       "png_bytes_equal": A.with_suffix(".png").read_bytes() == B.with_suffix(".png").read_bytes(),
       "added_keys_ignored": {"names": ADDED, "n_removed_from_new": n_added[0]}}
out["pass"] = (out["top_keys_same_order"] and out["differing_top_keys"] == ["meta"] and set(out["differing_meta_keys"]) <= {"git", "stamp_pt", "script_sha256"}
               and out["non_meta_bytes_equal"] and out["captions_and_title_equal"] and out["png_bytes_equal"])
p = HERE / "results" / f"regression_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
p.open("x").write(json.dumps(out, indent=1))
print(json.dumps({"file": p.name, "pass": out["pass"], "differing_meta_keys": out["differing_meta_keys"]}))
if not out["pass"]:  # a failed regression is never a successful run (Codex runner r1)
    raise SystemExit(1)
