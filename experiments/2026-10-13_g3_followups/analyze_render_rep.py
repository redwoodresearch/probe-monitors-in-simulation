"""G3 follow-up (2), POST HOC: the render gate for the repaired arm (run_extract_g3_rep.py --mode render --which rep), G3's
gate rules applied to one arm (../2026-10-13_g3_new_reader/analyze_render_g3.py's arm_rows(), imported unchanged): 823 runs,
0 render failures, 0 tool outputs without a Qwen3-32B position, 0 readouts whitespace-only or holding a newline (fail if
not), and >= 99% of required rows with an eligible readout, and every run with at least one (unevaluable if not). Every left-out row is listed for the
scoring. Also the repaired arm's Seed forward tokens and their squares, for the price. Local, $0. Exits 1 unless it passes.

  uv run --no-project python experiments/2026-10-13_g3_followups/analyze_render_rep.py results/g3_render_rep_<stamp>.json
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-13_g3_new_reader"))
from analyze_render_g3 import REVISION, arm_rows  # noqa: E402

ARM, EXPECT = "sim-attack-fix939", {"sim-attack-fix939": 823}


def main(p):
    R = json.loads(Path(p).read_text())
    if R["which"] != "rep" or R["reader"]["revision"] != REVISION:
        raise SystemExit("not a repaired-arm render record of G3's reader")
    n, runs = {}, {}
    for r in R["rows"]:
        n.setdefault(r["arm"], {})[r["count"]] = r["n"]
    for x in R["runs"]:
        if x["key"] in runs.setdefault(x["arm"], {}):
            raise SystemExit("a run listed twice")
        runs[x["arm"]][x["key"]] = x
    if set(n) != {ARM} or set(runs) != {ARM}:
        raise SystemExit(f"arms {sorted(n)} are not the repaired arm alone")
    fails, unevaluable = [], []
    out = {"scope": "rep", "POST HOC": True, "render files": {Path(p).name: {"sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(), "git (render)": R["git"],
                                                                             "which": "rep", "ids sha256": R["ids sha256"]}},
           "arms": arm_rows(n, runs, EXPECT, fails, unevaluable, (ARM,))}
    out["frozen row drops (repaired, own)"] = {k: x["left out"] for k, x in sorted(runs[ARM].items()) if x.get("left out")}
    empty = sorted(k for k, x in runs[ARM].items() if len(x.get("left out", [])) == x.get("required", 0))  # no eligible row left (0 required included)
    out["repaired runs with no eligible row"] = empty
    if empty:
        unevaluable.append(f"{len(empty)} repaired runs have no eligible row (no score for their cells)")
    out["gate"] = "fail" if fails else "unevaluable" if unevaluable else "pass"
    out["failing"], out["unevaluable"] = fails, unevaluable
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"render_rep_gate_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "frozen row drops (repaired, own)"}, indent=1)); print(dest)
    if out["gate"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1])
