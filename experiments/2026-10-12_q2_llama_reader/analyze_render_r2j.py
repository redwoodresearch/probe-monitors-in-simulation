"""The gate to the GPU (PLAN.md step A, pre-set; committed before the render's counts are read). Local, $0.
In every arm: render failures <= 0.5% of runs; R2J keeps >= 99% of the outputs Qwen3-32B kept; R2J counts 0 readouts
that encode only whitespace and 0 that encode a newline, at C2; 0 tool outputs lack a Qwen3-32B position; at least
99.9% of tool outputs' rendered spans JSON-decode exactly to the raw content. Beside it: newline membership of R0, R1
and R2J at C2, slip survival per component, and the forward tokens for the cost projection.

  uv run --no-project python experiments/2026-10-12_q2_llama_reader/analyze_render_r2j.py results/llama_render_r2j_<stamp>.json
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
COMPONENTS = ("(a) trailing newline", "(b) str_replace gutter", "(c) insert review note", "(d) hunk header")


def main(p):
    R = json.loads(Path(p).read_text()); n = defaultdict(dict)
    for r in R["rows"]:
        n[r["arm"]][r["count"]] = r["n"]
    out, fails = {"render file": Path(p).name, "git (render)": R["git"], "arms": {}}, []
    for a in sorted(n):
        c = n[a]; runs = c["runs"]; qk = c.get("outputs Qwen3-32B kept", 0); outs = c.get("tool outputs", 0)
        failures = sum(v for k, v in c.items() if k.startswith("render failure"))
        kept = c.get("of those, kept under C2 (R2J readout below the context length)", 0)
        rt = c.get("tool outputs whose rendered span JSON-decodes exactly to the raw content", 0)
        row = {"runs": runs, "render failures": failures, "outputs Qwen3-32B kept": qk, "kept under C2 (R2J)": kept,
               "R2J coverage of Qwen3-32B's kept outputs": kept / qk if qk else None,
               "R2J at index 32,768 or beyond": c.get("of those, R2J at index 32,768 or beyond", 0),
               "tool outputs with no Qwen3-32B position": c.get("tool outputs with no Qwen3-32B position", 0),
               "tool outputs": outs, "JSON round trip exact": rt, "JSON round trip share": rt / outs if outs else None,
               "forward tokens": c.get("forward tokens (to the last kept readout)", 0), "Llama tokens": c.get("Llama tokens", 0),
               "slip survival (in raw, in raw and rendered)": {k: [c.get(f"slip {k}: in the raw output", 0), c.get(f"slip {k}: in the raw output and in the rendered span", 0)] for k in COMPONENTS}}
        for rule in ("R0", "R1", "R2J"):
            row[f"{rule} at C2: readouts, encode only whitespace, encode a newline"] = [c.get(f"{rule} readouts at C2", 0), c.get(f"{rule} readouts at C2: encodes only whitespace", 0),
                                                                                       c.get(f"{rule} readouts at C2: encodes a newline", 0)]
        out["arms"][a] = row
        if failures > 0.005 * runs:
            fails.append(f"{a}: render failures {failures} of {runs}")
        if not qk or kept < 0.99 * qk:
            fails.append(f"{a}: R2J keeps {kept} of {qk}")
        if row["R2J at C2: readouts, encode only whitespace, encode a newline"][1:] != [0, 0]:
            fails.append(f"{a}: R2J has whitespace-only or newline readouts")
        if row["tool outputs with no Qwen3-32B position"]:
            fails.append(f"{a}: {row['tool outputs with no Qwen3-32B position']} outputs lack a Qwen3-32B position")
        if not outs or rt < 0.999 * outs:
            fails.append(f"{a}: JSON round trip exact for {rt} of {outs}")
    out["gate"] = "pass" if not fails else "fail"; out["failing"] = fails
    out["forward tokens, all arms"] = sum(r["forward tokens"] for r in out["arms"].values())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"render_r2j_gate_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(sys.argv[1])
