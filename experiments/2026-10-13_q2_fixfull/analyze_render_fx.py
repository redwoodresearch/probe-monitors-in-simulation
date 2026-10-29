"""Fix-full (PLAN.md here), POST HOC: the render gate for the fx arms (run_extract_g3_fx.py --mode render --which fx), a copy of
../2026-10-13_g3_followups/analyze_render_rep.py (1c277bfc) applied to every staged arm, each arm's expected count from the stage
record. Its docstring follows.

G3 follow-up (2), POST HOC: the render gate for the repaired arm (run_extract_g3_rep.py --mode render --which rep), G3's
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



def main(p, stage):
    R = json.loads(Path(p).read_text())
    if R["which"] != "fx" or R["reader"]["revision"] != REVISION:
        raise SystemExit("not an fx render record of G3's reader")
    EXPECT = {a: len(ks) for a, ks in json.loads(Path(stage).read_text())["runs by arm"].items()}
    ARMS = tuple(sorted(EXPECT))
    n, runs = {}, {}
    for r in R["rows"]:
        n.setdefault(r["arm"], {})[r["count"]] = r["n"]
    for x in R["runs"]:
        if x["key"] in runs.setdefault(x["arm"], {}):
            raise SystemExit("a run listed twice")
        runs[x["arm"]][x["key"]] = x
    if set(n) != set(ARMS) or set(runs) != set(ARMS):
        raise SystemExit(f"arms {sorted(n)} are not the staged arms")
    fails, unevaluable = [], []
    out = {"scope": "fx", "POST HOC": True, "render files": {Path(p).name: {"sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(), "git (render)": R["git"],
                                                                             "which": "fx", "ids sha256": R["ids sha256"]}},
           "arms": arm_rows(n, runs, EXPECT, fails, unevaluable, ARMS)}
    out["frozen row drops (fx, own)"] = {a: {k: x["left out"] for k, x in sorted(runs[a].items()) if x.get("left out")} for a in ARMS}
    empty = {a: sorted(k for k, x in runs[a].items() if len(x.get("left out", [])) == x.get("required", 0)) for a in ARMS}  # no eligible row left
    out["fx runs with no eligible row"] = empty
    for a, e in empty.items():
        if e:
            unevaluable.append(f"{a}: {len(e)} runs have no eligible row (no score for their cells)")
    out["gate"] = "fail" if fails else "unevaluable" if unevaluable else "pass"
    out["failing"], out["unevaluable"] = fails, unevaluable
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"render_fx_gate_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "frozen row drops (fx, own)"}, indent=1)); print(dest)
    if out["gate"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
