"""POST HOC (after the pre-set reading; no verdict): E1's D computed from the cached scores and from the one-pass scores
on the checked runs whose hide and instructed sequences were both rescored, with the confirmation's own readout (the
frozen masks' "scored" tokens, at least 10) and its frozen SD (0.5396). Local, $0.

  uv run --no-project --with numpy python experiments/2026-10-12_q2_e1_equiv_check/analyze_d_change_POSTHOC.py \\
      <one_pass json> <archived stage folder> <confirm_masks_2026-10-11_185020.json>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
SD0, MIN_TOKENS, MASKS_SHA = 0.5396148830137296, 10, "b3db91f08e361a49d8c1f559edf878c5d26eeb81fdfabbb564865f42e1b6974b"


def main(one_p: Path, folder: Path, masks_p: Path):
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    if hashlib.sha256(masks_p.read_bytes()).hexdigest() != MASKS_SHA:
        raise SystemExit("not the confirmation's masks result")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    masks = json.loads(masks_p.read_text())["masks"]
    stored = {(r["key"], r["seq"]): r for r in (json.loads(p.read_text()) for p in (folder / "scores").glob("*.json"))}
    one = {(r["key"], r["seq"]): r for r in json.loads(one_p.read_text())["rows"] if not r.get("oom")}

    def readout(pt, key, seq):
        idx = masks.get(f"{key}|{seq}", {}).get("scored", [])
        return (float(np.asarray(pt)[idx].mean()), len(idx)) if len(idx) >= MIN_TOKENS else (None, len(idx))
    rows = []
    for key in sorted({k for k, _ in one}):
        if not all((key, s) in one for s in ("hide", "instructed_honest")):
            continue
        c = [readout(stored[(key, s)]["per_token"], key, s) for s in ("hide", "instructed_honest")]
        o = [readout(one[(key, s)]["one_pass_per_token"], key, s) for s in ("hide", "instructed_honest")]
        if None in (c[0][0], c[1][0]):
            rows.append({"key": key, "readout": "fewer than 10 scored tokens in a report", "scored tokens (hide, instructed)": [c[0][1], c[1][1]]}); continue
        dc, do = (c[0][0] - c[1][0]) / SD0, (o[0][0] - o[1][0]) / SD0
        rows.append({"key": key, "scored tokens (hide, instructed)": [c[0][1], c[1][1]], "D cached": dc, "D one pass": do, "|change in D|": abs(dc - do)})
    ch = [r["|change in D|"] for r in rows if "|change in D|" in r]
    out = {"label": "POST HOC (after the pre-set reading; no verdict)", "git": head, "one-pass file": {one_p.name: hashlib.sha256(one_p.read_bytes()).hexdigest()},
           "runs with both reports rescored": len(rows), "runs with a D (both readouts >= 10 tokens)": len(ch),
           "max |change in D|": max(ch) if ch else None, "mean |change in D|": float(np.mean(ch)) if ch else None, "rows": rows}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"d_change_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
