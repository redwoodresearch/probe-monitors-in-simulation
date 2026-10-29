"""Print the panel check (a) table from results/false_zero_*.json (by script)."""

import json
import sys

d = json.load(open(sys.argv[1]))  # noqa: SIM115
for key, bins in d["bins"].items():
    cells = []
    for b, v in bins.items():
        if v["n"]:
            lo, hi = v["cp95"]
            cells.append(f"{b}: {v['k_zero']}/{v['n']}={v['share_zero']:.2f} [{lo:.2f},{hi:.2f}]")
        else:
            cells.append(f"{b}: n=0")
    print(key, "|", "; ".join(cells))
