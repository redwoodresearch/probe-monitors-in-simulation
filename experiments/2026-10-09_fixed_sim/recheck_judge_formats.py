"""Review check for #516 (thread LZbV), local and free: for every describe-judge answer file (data/describe_judge_*.json),
how each seed's answer was read (strict JSON, or the prose fallback), how many raw answers were JSON wrapped in a code fence
or preamble, and the picks when such wrapped JSON is parsed directly. Numbers only; also written to
<out dir>/recheck_judge_formats_<stamp>.json with the inputs' sha256.

  python3 experiments/2026-10-09_fixed_sim/recheck_judge_formats.py <out dir> experiments/2026-10-09_fixed_sim/data/describe_judge_*.json
"""
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def wrapped_json(raw: str):
    """The JSON object inside a fence or preamble, if the answer is one; else None."""
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.S) or re.search(r"(\{.*\})", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) and "pairs" in d else None


rows = []
for path in sys.argv[2:]:
    d = json.load(open(path))
    for a in d["answers"]:
        fmt = a.get("format", "json")
        raw = a["answer"].get("raw", "")
        w = wrapped_json(raw) if fmt == "prose" else None
        line = f"{path.rsplit('/', 1)[-1]} | {d['model']} {d['reasoning_effort']} | {a['seed'][-1]} | read as {fmt} | picked {a['picked_simulated']}/{a['pairs_answered']}"
        if w is not None:
            picked = {p["pair"]: p["simulated"] for p in w["pairs"]}
            hits = sum(picked.get(k["pair"]) == k["simulated"] for k in a["key"])
            line += f" | WRAPPED JSON: {hits}/{len(picked)}"
        elif fmt == "prose":
            line += " | prose (no JSON object inside)"
        print(line)
        rows.append({"file": path.rsplit("/", 1)[-1], "model": d["model"], "effort": d["reasoning_effort"], "seed": a["seed"], "read_as": fmt,
                     "picked": a["picked_simulated"], "answered": a["pairs_answered"], "wrapped_json": w is not None})
here = Path(__file__)
out = {"answers": rows, "counts": {"strict_json": sum(r["read_as"] == "json" for r in rows), "prose": sum(r["read_as"] == "prose" for r in rows),
                                   "wrapped_json": sum(r["wrapped_json"] for r in rows)},
       "inputs_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sys.argv[2:]},
       "script_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
       "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=here.parent).stdout.strip()}
dst = Path(sys.argv[1]) / f"recheck_judge_formats_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
with open(dst, "x") as fh:
    fh.write(json.dumps(out, indent=1))
print(out["counts"], "->", dst)
