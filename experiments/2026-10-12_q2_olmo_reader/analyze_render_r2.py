"""The gate to the GPU (PLAN.md step 2a, pre-set; committed before the re-render's counts were read). Local, $0.
In every arm: render failures <= 0.5% of runs; R2 keeps >= 99% of the outputs Qwen3-32B kept; R2 counts 0
whitespace-only and 0 newline-containing readouts at C2; 0 tool outputs lack a Qwen3-32B position. Reports the newline
membership of R0, R1 and R2 at C2 by arm beside it, and the forward tokens for the cost projection.

  uv run --no-project python experiments/2026-10-12_q2_olmo_reader/analyze_render_r2.py results/olmo_render_r2_<stamp>.json
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent


def main(p):
    R = json.loads(Path(p).read_text()); n = defaultdict(dict)
    for r in R["rows"]:
        n[r["arm"]][r["count"]] = r["n"]
    arms, out, fails = sorted(n), {"render file": Path(p).name, "git (render)": R["git"], "arms": {}}, []
    for a in arms:
        c = n[a]; runs = c["runs"]; qk = c.get("outputs Qwen3-32B kept", 0)
        failures = sum(v for k, v in c.items() if k.startswith("render failure"))
        kept = c.get("of those, kept under C2 (R2 readout below the context length)", 0)
        row = {"runs": runs, "render failures": failures, "outputs Qwen3-32B kept": qk, "kept under C2 (R2)": kept,
               "R2 coverage of Qwen3-32B's kept outputs": kept / qk if qk else None,
               "R2 at index 32,768 or beyond": c.get("of those, R2 at index 32,768 or beyond (kept if below 65,536)", 0),
               "tool outputs with no Qwen3-32B position": c.get("tool outputs with no Qwen3-32B position", 0),
               "forward tokens": c.get("forward tokens (to the last kept readout)", 0), "Olmo tokens": c.get("Olmo tokens", 0)}
        for rule in ("R0", "R1", "R2"):
            t = c.get(f"{rule} readouts at C2", 0)
            row[f"{rule} at C2: readouts, whitespace-only, contains a newline"] = [t, c.get(f"{rule} readouts at C2: whitespace-only", 0), c.get(f"{rule} readouts at C2: contains a newline", 0)]
        out["arms"][a] = row
        if failures > 0.005 * runs:
            fails.append(f"{a}: render failures {failures} of {runs}")
        if not qk or kept < 0.99 * qk:
            fails.append(f"{a}: R2 keeps {kept} of {qk}")
        if row["R2 at C2: readouts, whitespace-only, contains a newline"][1:] != [0, 0]:
            fails.append(f"{a}: R2 has whitespace-only or newline readouts")
        if row["tool outputs with no Qwen3-32B position"]:
            fails.append(f"{a}: {row['tool outputs with no Qwen3-32B position']} outputs lack a Qwen3-32B position")
    out["gate"] = "pass" if not fails else "fail"; out["failing"] = fails
    out["forward tokens, all arms"] = sum(r["forward tokens"] for r in out["arms"].values())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"render_r2_gate_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(sys.argv[1])
