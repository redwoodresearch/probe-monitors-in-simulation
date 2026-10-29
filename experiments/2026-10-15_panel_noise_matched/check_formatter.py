"""Test of the shared formatter (2e, 10-16): every printed value in a value table equals round(x, 2) of its stored raw float, and the three known slips print
correctly from their stored floats. JSON reads only.

  PYTHONPATH=src python3 experiments/2026-10-15_panel_noise_matched/check_formatter.py <table.json> [...]
"""

import json
import sys
from pathlib import Path

from probe_shift.fmtpts import p2, stat2

KNOWN = [(2.004563233376792, "+2.00"), (-2.8846153846153815, "-2.88"), (2.5254578997573653, "+2.53"), (-0.001, "+0.00")]
bad = [(x, p2(x), want) for x, want in KNOWN if p2(x) != want]
n = 0


def walk(o):
    global n
    if isinstance(o, dict):
        if "printed" in o and "point" in o:
            n += 1
            want = stat2({"point": o["point"], "ci95": o.get("ci95")})
            exp = f"{round(o['point'], 2) + 0.0:+.2f}"
            if o["printed"] != want or not o["printed"].startswith(exp):
                bad.append((o["point"], o["printed"], exp))
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)


for fn in sys.argv[1:]:
    walk(json.loads(Path(fn).read_text()))
print(json.dumps({"checked_printed_values": n, "known_cases": len(KNOWN), "failures": bad[:10], "pass": not bad}))
if bad:
    raise SystemExit("formatter test failed")
