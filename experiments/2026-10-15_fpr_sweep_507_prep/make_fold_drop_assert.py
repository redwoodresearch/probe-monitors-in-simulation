"""Keys-only assert on Stage A records (09, relayed 03:5x): Sol (a), c_asis and own_all_rs rows dropped NO fold -1 cell (the fold -1 units are
kept and scored; the drops file is a may-drop list). Reads each row's fold_minus1_dropped / fold_minus1_rule only; no score.
Output results/checks/fold_drop_assert_<stamp>.json; exit 1 if any such row dropped a cell (every dropped key listed).

  python make_fold_drop_assert.py results/<record A>.json results/<record B>.json
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
FAMS = ("Sol (a)", "c_asis", "own_all_rs_L48", "own_all_rs25_L48", "own_all_rs10_L48")

if not sys.argv[1:]:
    raise SystemExit("usage: make_fold_drop_assert.py <record> [<record> ...]")
out = {"what": "fold -1 drops in the Stage A records, keys only (09: Sol (a) / c_asis / own_all_rs must drop 0 cells)", "records": {}, "dropped": [],
       "missing_evidence": []}
for rp in sys.argv[1:]:
    b = (HERE / rp).read_bytes()
    R = json.loads(b)
    n = 0
    for lb, d in R["rows"].items():
        if lb.rsplit(" ", 1)[0] not in FAMS:
            continue
        for o, r in d.items():
            if "withheld" in r:
                continue
            n += 1
            if "fold_minus1_dropped" not in r:  # every computed row carries it ({} when nothing dropped); missing evidence is a failure
                out["missing_evidence"].append({"record": rp, "label": lb, "origin": o})
                continue
            dr = r["fold_minus1_dropped"].get("cohort_cells", {})
            if dr.get("n", 0):
                out["dropped"].append({"record": rp, "label": lb, "origin": o, "n": dr["n"], "keys": dr.get("keys", []), "rule": r.get("fold_minus1_rule")})
    out["records"][rp] = {"sha256": hashlib.sha256(b).hexdigest(), "rows_checked": n}
out["pass"] = not out["dropped"] and not out["missing_evidence"] and all(v["rows_checked"] > 0 for v in out["records"].values())
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
dest = HERE / "results" / "checks" / f"fold_drop_assert_{stamp}.json"
blob = json.dumps(out, indent=1).encode()
dest.open("xb").write(blob)
print(json.dumps({"file": str(dest.relative_to(HERE)), "sha256": hashlib.sha256(blob).hexdigest(), "pass": out["pass"],
                  "rows_checked": {k: v["rows_checked"] for k, v in out["records"].items()}, "dropped": out["dropped"][:10],
                  "missing_evidence": out["missing_evidence"][:10]}, indent=1))
sys.exit(0 if out["pass"] else 1)
