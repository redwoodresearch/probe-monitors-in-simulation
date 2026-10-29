"""POST HOC (Codex on #718): the ±5 sizing of the same-batch real-vs-real gap, from the farther interval endpoint beside
the committed half-width-average one. Reads only the committed result file. Local, $0.

  uv run --no-project python experiments/2026-10-11_q2_transfer_rvr/analyze_sizing_endpoint_POSTHOC.py \\
      experiments/2026-10-11_q2_transfer_rvr/results/transfer_rvr_2026-10-11_133901.json
"""
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
PIN = "transfer_rvr_2026-10-11_133901.json"
TARGET, PER_RUN, BUFFER = 0.05, 1.00, 1.30  # ±5 points; df's whole-job $ per real Sol honest run, raw and with 30%


def main(p: Path):
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    if p.name != PIN:
        raise SystemExit(f"{p.name} is not {PIN}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    d = json.loads(p.read_text())
    out = {"label": "POST HOC (Codex on #718)", "git": head, "input": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()}, "rates": {}}
    for rate in ("1%", "5%"):
        r = d[f"target {rate}"]["same-batch pairs"]; g = r["pooled gap"]; n = r["pairs needed for ±5 (95% half-width)"]["pairs now"]
        lo, hi, pt = g["95%"][0], g["95%"][1], g["point"]
        res = {"point (points)": 100 * pt, "below / above the point (points)": [100 * (pt - lo), 100 * (hi - pt)], "pairs now": n}
        for name, h in (("half the width (as committed: total width 10)", (hi - lo) / 2), ("farther endpoint (±5 on both sides)", max(pt - lo, hi - pt))):
            need = math.ceil(n * (h / TARGET) ** 2); more = max(0, need - n)
            res[name] = {"half-width used (points)": 100 * h, "pairs needed": need, "more pairs": more, "more runs (2 per pair)": 2 * more,
                         "cost raw ($)": 2 * more * PER_RUN, "cost with 30% ($)": round(2 * more * BUFFER, 2)}
        if res["half the width (as committed: total width 10)"]["pairs needed"] != r["pairs needed for ±5 (95% half-width)"]["pairs needed"]:
            raise SystemExit(f"{rate}: does not reproduce the committed pairs needed")
        out["rates"][rate] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"sizing_endpoint_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
