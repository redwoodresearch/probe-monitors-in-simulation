"""Draw-0 reproduction gate (RULE.md): the rescoring's per-call draw 0 must reproduce the stored draw 0 on the five
existing cells. Fails closed. Reads only those cells; never prints or saves the new sim->run2 cell's values.

  python -I check_gate.py <stored draws0-0 per-call json> <new draws0-0 per-call json>
"""
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STORED_SHA = "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf"
CELLS = ["real->real", "real->sim", "sim->real", "sim->sim", "real->run2"]
TOL = 1e-9


def main(stored_path, new_path):
    if (got := hashlib.sha256(Path(stored_path).read_bytes()).hexdigest()) != STORED_SHA:
        raise SystemExit(f"{stored_path}: sha256 {got} is not {STORED_SHA}")
    A, B = json.loads(Path(stored_path).read_text()), json.loads(Path(new_path).read_text())
    bad = []
    if [d["draw"] for d in B["draws"]] != [0]:
        bad.append("the new file must hold draw 0 only")
    a, b = A["draws"][0], B["draws"][0]
    if "sim->run2" not in b["cells"]:
        bad.append("the new file has no sim->run2 cell")
    if a.get("choices") != b.get("choices"):
        bad.append("per-fold (layer, C) choices differ")
    meta = lambda F: [[m["world"], m["task"], m["irid"], m["call"], m["y"]] for m in F["meta"]]
    if meta(A) != meta(B):
        bad.append("meta rows differ")
    for c in CELLS:
        ca, cb = a["cells"].get(c), b["cells"].get(c)
        if ca is None or cb is None:
            bad.append(f"{c}: missing")
            continue
        va, vb = ca["values"], cb["values"]
        if sorted(va) != sorted(vb):
            bad.append(f"{c}: metric keys differ")
        for m in va:
            x, y = va[m], vb.get(m)
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y)
                    and abs(x - y) <= TOL):
                bad.append(f"{c}: {m} differs or is not finite")
        if ca.get("failed") != cb.get("failed"):
            bad.append(f"{c}: failed records differ")
        pa, pb = ca.get("per_call"), cb.get("per_call")
        if pa is None or pb is None:
            bad.append(f"{c}: no per-call output")
            continue
        for k in ("row", "flag_frozen", "flag_calibrated"):
            if pa[k] != pb[k]:
                bad.append(f"{c}: per-call {k} differs")
    out = {"gate": "PASS" if not bad else "FAIL", "mismatches": bad, "cells_checked": CELLS, "tolerance": TOL,
           "stored": Path(stored_path).name, "stored_sha256": STORED_SHA, "new": Path(new_path).name,
           "new_sha256": hashlib.sha256(Path(new_path).read_bytes()).hexdigest(),
           "new_git": B.get("git"), "new_scorer_sha256": B.get("scorer_sha256"), "new_checkpoint_folder": B.get("checkpoint_folder")}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = Path(__file__).parent / "results" / f"gate_draw0_{stamp}.json"
    with dst.open("x") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print("->", dst)
    if bad:
        raise SystemExit("GATE FAILED: no bootstrap job launches")


if __name__ == "__main__":
    main(*sys.argv[1:3])
