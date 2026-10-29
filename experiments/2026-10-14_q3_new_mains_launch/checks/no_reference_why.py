"""ON THE BOX, read-only (Aswin via 0d and d9, 10-15 ~09:30 PT): why a cell had no usable real reference. For each cell, from
the main launch's launch.json: its reference_state entry (tries, state, unusable run folders) and, for every real arm of
the cell, the run key, state and each attempt's start, end, exit code, kind, redrawn_because, eval_status, clean,
usable_reference, policy-call count, error CLASS names (no messages) and its .eval file name. Metadata only.
  python3 no_reference_why.py <main run dir> <cell> [<cell> ...]"""

import collections
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
raw = (run / "launch.json").read_bytes()
d = json.loads(raw)


def classes(errs):
    out = collections.Counter()
    for e in errs or []:
        s = str(e)
        m = re.search(r"raised (\w+)", s)
        out[(m.group(1) if m else re.split(r"[(:\s]", s, maxsplit=1)[0])[:40]] += 1
    return dict(out)


def attempt(a):
    o = a.get("outcome") or {}
    ev = sorted(glob.glob(a["log_dir"] + "/*.eval")) if a.get("log_dir") else []
    return {"start": a.get("start"), "end": a.get("end"), "exit_code": a.get("exit_code"), "kind": a.get("kind"),
            "redrawn_because": a.get("redrawn_because"), "policy_called": a.get("policy_called"),
            "eval_status": o.get("eval_status"), "clean": o.get("clean"), "usable_reference": o.get("usable_reference"),
            "policy_calls": len(o.get("policy_calls") or []), "error_classes": classes(o.get("errors")),
            "sample_error_classes": classes(o.get("sample_errors")), "log_dir": a.get("log_dir"),
            "eval_files": [Path(x).name for x in ev]}


out = {"launch": run.name, "launch_json_sha256": hashlib.sha256(raw).hexdigest(), "cells": {}}
for c in sys.argv[2:]:
    out["cells"][c] = {
        "reference_state": d.get("reference_state", {}).get(c),
        "runs": [{"key": r["key"], "real": r["real"], "state": r["state"], "why": r.get("why"),
                  "attempts": [attempt(a) for a in r["attempts"]]}
                 for r in d["runs"] if r["key"].split("/")[-1] in (c, c + "_r2")]}
print(json.dumps(out, indent=1))
