"""Archive analyze_warm_gate.py's verdicts (the solver amendment's evidence) as residuals only. Local, free.

For each `warm=reference` pair of draws files, runs analyze_warm_gate.main unchanged and keeps only what is not a
metric value: draws compared, the largest absolute difference, the tolerance, the draws whose layer and C choices
differ, how many draws had choices compared, seconds per draw, the verdict, and each input's sha256. A pair with any
metric over tolerance stops the script before anything is written, because those rows would carry metric values.
Writes results/warm_gate_record_<stamp>.json.

  python3 experiments/2026-10-08_q3_onpolicy_figures/run_warm_gate_record.py <warm.json>=<reference.json> ...
"""

import contextlib
import hashlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import analyze_warm_gate as gate

HERE = Path(__file__).parent
KEEP = ("draws_compared", "max_abs_diff", "tol", "draws_with_different_choices", "draws_with_choices_compared",
        "warm_seconds", "reference_seconds", "passes")


def one(warm: str, ref: str) -> dict:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            gate.main(warm, ref)
        except SystemExit:  # a failed verdict still prints its record first
            pass
    out = json.loads(buf.getvalue())
    if out["over_tol"]:
        raise SystemExit(f"{Path(warm).name} vs {Path(ref).name}: {len(out['over_tol'])} metric(s) over tolerance; nothing written")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    return {"warm": Path(warm).name, "reference": Path(ref).name, "warm_sha256": sha(warm), "reference_sha256": sha(ref),
            "over_tol": 0} | {k: out[k] for k in KEEP}


def main(pairs: list[str]):
    rows = [one(*p.split("=", 1)) for p in pairs]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"warm_gate_record_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps({"gate": "analyze_warm_gate.py", "tol": gate.TOL, "pairs": rows}, indent=1))
    print(json.dumps([{k: r[k] for k in ("warm", "reference", "draws_compared", "max_abs_diff", "passes")} for r in rows], indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
