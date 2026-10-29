"""Option C's gate (warm-started primal fits): the same draws' values from a warm run and a reference run (cold primal,
or the dual) must agree, metric by metric, within TOL; choices are compared where both runs saved them. Local, free.

  python3 experiments/2026-10-08_q3_onpolicy_figures/analyze_warm_gate.py <warm draws file> <reference draws file>
"""

import json
import sys
from pathlib import Path

TOL = 1e-4


def main(warm: str, ref: str):
    a, b = (json.loads(Path(p).read_text()) for p in (warm, ref))
    da, db = ({d["draw"]: d for d in x["draws"]} for x in (a, b))
    common = sorted(set(da) & set(db))
    if not common or set(da) - set(db) - {0}:
        raise ValueError(f"warm draws {sorted(da)} are not all in the reference's {sorted(db)}")
    worst, rows, choice_diff, choices_checked = 0.0, [], [], 0
    for k in common:
        if set(da[k]["cells"]) != set(db[k]["cells"]):
            raise ValueError(f"draw {k}: cells differ")
        for cell, v in da[k]["cells"].items():
            if set(v["values"]) != set(db[k]["cells"][cell]["values"]):
                raise ValueError(f"draw {k} {cell}: metrics differ")
            for m, x in v["values"].items():
                y = db[k]["cells"][cell]["values"][m]
                if any(z is not None and z != z for z in (x, y)):
                    raise ValueError(f"draw {k} {cell} {m}: NaN")
                d = None if x is None or y is None else abs(x - y)
                if (x is None) != (y is None):
                    raise ValueError(f"draw {k} {cell} {m}: one run failed the metric and the other did not")
                if d is not None:
                    worst = max(worst, d)
                    if d > TOL:
                        rows.append({"draw": k, "cell": cell, "metric": m, "warm": x, "reference": y})
        if "choices" in db[k] and da[k].get("choices") != db[k]["choices"]:
            choice_diff.append(k)
        choices_checked = choices_checked + 1 if "choices" in db[k] else choices_checked
    out = {"warm": Path(warm).name, "reference": Path(ref).name, "draws_compared": common, "max_abs_diff": worst, "tol": TOL,
           "over_tol": rows, "draws_with_different_choices": choice_diff,
           "warm_seconds": {k: da[k]["seconds"] for k in common}, "reference_seconds": {k: db[k]["seconds"] for k in common},
           "draws_with_choices_compared": choices_checked, "passes": not rows and not choice_diff}
    print(json.dumps(out, indent=1))
    if not out["passes"]:
        sys.exit(1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
