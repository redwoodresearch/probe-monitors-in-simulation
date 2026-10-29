"""Dry run of run_dt_q4b.py main() with no Modal app: the module is imported (Modal objects are lazy), arm/relaunch are replaced by a stub
that records what main() would send and stops before any remote call, so nothing is launched and nothing is written to results/. Every
local check of main() runs as in a launch. Usage (from the dts worktree): python dryrun_main.py <the launch's run_dt_q4b.py arguments>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "experiments/2026-10-14_p2_dt_scaleup")
import run_dt_q4b as m  # noqa: E402


class Stop(Exception):
    pass


class Stub:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *a):
        if self.fn == "simh_primary":  # (counted_raw, cells, strata, name, meta)
            rr, cells, strata, declared, name, meta = [("simh_full_b1+b2", json.loads(a[0])["keys"])], a[1], a[2], {}, a[3], a[4]
        else:
            rr, cells, strata, declared, name, meta = (a[0], a[1], a[2], a[3], a[4], a[5]) if self.fn == "relaunch" else (a[0], a[1], a[2], a[4], a[5], a[7])
        print("DRYRUN", json.dumps({"function": self.fn, "name": name, "runs": [r for r, _ in rr], "manifest_rows": [len(f) for _, f in rr], "cells": len(cells),
                                    "strata": sorted({tuple(v) for v in strata.values()}), "declared": len(declared), "stop_usd": meta["stop_usd"],
                                    "preflight_bound_usd": meta["preflight_bound_usd"], "roots_read": meta.get("roots_read"), "dump_evidence": meta.get("dump_evidence"),
                                    "git": meta["git"]}, indent=1), flush=True)
        raise Stop


m.arm, m.relaunch = Stub("arm"), Stub("relaunch")
if hasattr(m, "simh_primary"):
    m.simh_primary = Stub("simh_primary")
kw, args = {}, sys.argv[1:]
for i in range(0, len(args), 2):
    kw[args[i].lstrip("-").replace("-", "_")] = args[i + 1]
fn = getattr(m.main, "info", None)
raw = fn.raw_f if fn is not None else m.main.raw_f
try:
    raw(**kw)
except Stop:
    print("DRYRUN-OK main() passed every local check; stopped before the remote call; no app, nothing written", flush=True)
    sys.exit(0)
raise SystemExit("main() returned without reaching the remote call")
