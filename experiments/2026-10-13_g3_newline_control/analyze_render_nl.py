"""G3 newline-only controls, POST HOC: the render gate for one edited arm (run_extract_g3_nl.py --mode render --which
nlstrip|nladd), G3's gate rules applied to that arm (../2026-10-13_g3_new_reader/analyze_render_g3.py's arm_rows(), imported
unchanged): 939 runs, 0 render failures (edited and unedited renderings), 0 tool outputs without a Qwen3-32B position, 0
readouts whitespace-only or holding a newline (fail if not), >= 99% of required rows with an eligible readout, and every run
with at least one (unevaluable if not). It also reports the edit's counts by family and how many edited outputs' readout
token text differs from the unedited rendering's (reported, not gated). Every left-out row is listed for the scoring.
Local, $0. Exits 1 unless it passes.

  uv run --no-project python experiments/2026-10-13_g3_newline_control/analyze_render_nl.py results/g3_render_<which>_<stamp>.json
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

ARMS = {"nlstrip": "sim-attack-nlstrip", "nladd": "real-attack-e2-nladd"}


def main(p):
    R = json.loads(Path(p).read_text())
    which = R["which"]
    if which not in ARMS or R["reader"]["revision"] != REVISION:
        raise SystemExit("not an nl render record of G3's reader")
    arm = ARMS[which]
    n, runs = {}, {}
    for r in R["rows"]:
        n.setdefault(r["arm"], {})[r["count"]] = r["n"]
    for x in R["runs"]:
        if x["key"] in runs.setdefault(x["arm"], {}):
            raise SystemExit("a run listed twice")
        runs[x["arm"]][x["key"]] = x
    if set(n) != {arm} or set(runs) != {arm}:
        raise SystemExit(f"arms {sorted(n)} are not {arm} alone")
    fails, unevaluable = [], []
    out = {"scope": which, "POST HOC": True,
           "render files": {Path(p).name: {"sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(), "git (render)": R["git"], "which": which,
                                           "ids sha256": R["ids sha256"]}},
           "arms": arm_rows(n, runs, {arm: 939}, fails, unevaluable, (arm,))}
    unedited_fail = sum(v for k, v in n[arm].items() if k.startswith("render failure (unedited)"))
    if unedited_fail:
        fails.append(f"{unedited_fail} unedited renderings failed")
    out["edit"] = {k: v for k, v in sorted(n[arm].items()) if k.startswith("edited outputs") or k == "runs with at least one edited output"}
    out["frozen row drops (own)"] = {k: x["left out"] for k, x in sorted(runs[arm].items()) if x.get("left out")}
    empty = sorted(k for k, x in runs[arm].items() if len(x.get("left out", [])) == x.get("required", 0))
    out["runs with no eligible row"] = empty
    if empty:
        unevaluable.append(f"{len(empty)} runs have no eligible row")
    out["gate"] = "fail" if fails else "unevaluable" if unevaluable else "pass"
    out["failing"], out["unevaluable"] = fails, unevaluable
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"render_{which}_gate_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "frozen row drops (own)"}, indent=1)); print(dest)
    if out["gate"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1])
