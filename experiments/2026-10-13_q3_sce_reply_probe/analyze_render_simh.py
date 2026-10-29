"""SCE grid, Seed-OSS-36B (RULE.md Amendment 13, this folder), POST HOC: the render gate for run_extract_g3_simh.py --mode render
(the 437 simulated honest runs and the two gate runs). A copy of q2's ../2026-10-13_g3_followups/analyze_render_rep.py (@ 1c277bfc)
with the arms changed, applying G3's gate rules per arm (G3's analyze_render_g3.py arm_rows(), imported unchanged from
g3_vendored/, a byte copy at a3efa7ea): the plan's run counts, 0 render failures, 0 tool outputs without a Qwen3-32B position,
0 readouts whitespace-only or holding a newline (fail if not), and >= 99% of required rows with an eligible readout, and every
run with at least one (unevaluable if not). Every left-out row is listed for the scoring. Local, $0. Exits 1 unless it passes.

  uv run --no-project python experiments/2026-10-13_q3_sce_reply_probe/analyze_render_simh.py results/g3_render_simh_<stamp>.json
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "g3_vendored"))
from analyze_render_g3 import REVISION, arm_rows  # noqa: E402

EXPECT = {"sim-honest": 437, "sim-attack": 2}


def main(p):
    R = json.loads(Path(p).read_text())
    if R["which"] != "simh" or R["reader"]["revision"] != REVISION:
        raise SystemExit("not a simh render record of G3's reader")
    n, runs = {}, {}
    for r in R["rows"]:
        n.setdefault(r["arm"], {})[r["count"]] = r["n"]
    for x in R["runs"]:
        if x["key"] in runs.setdefault(x["arm"], {}):
            raise SystemExit("a run listed twice")
        runs[x["arm"]][x["key"]] = x
    if set(n) != set(EXPECT) or set(runs) != set(EXPECT):
        raise SystemExit(f"arms {sorted(n)} are not the simh arms")
    fails, unevaluable = [], []
    out = {"scope": "simh", "POST HOC": True, "render files": {Path(p).name: {"sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(), "git (render)": R["git"],
                                                                              "which": "simh", "ids sha256": R["ids sha256"]}},
           "arms": arm_rows(n, runs, EXPECT, fails, unevaluable, tuple(EXPECT))}
    out["frozen row drops, by arm"] = {a: {k: x["left out"] for k, x in sorted(runs[a].items()) if x.get("left out")} for a in sorted(EXPECT)}
    empty = {a: sorted(k for k, x in runs[a].items() if len(x.get("left out", [])) == x.get("required", 0)) for a in sorted(EXPECT)}  # 0 required included
    out["runs with no eligible row, by arm"] = empty
    if any(empty.values()):
        unevaluable.append(f"runs with no eligible row: { {a: len(v) for a, v in empty.items()} }")
    out["gate"] = "fail" if fails else "unevaluable" if unevaluable else "pass"
    out["failing"], out["unevaluable"] = fails, unevaluable
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"render_simh_gate_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "frozen row drops, by arm"}, indent=1)); print(dest)
    if out["gate"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1])
