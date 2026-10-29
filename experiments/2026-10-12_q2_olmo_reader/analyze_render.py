"""A third reader, step 1: the pre-set go / no-go (PLAN.md) applied to the render's counts. Local, $0; committed before the
render's counts are read.

  uv run --no-project python experiments/2026-10-12_q2_olmo_reader/analyze_render.py <olmo_render json>
"""
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ARMS = ("real honest", "real e1", "real e2", "simulated (as simulated)", "simulated, re-rendered (823)")
CATS = ("whitespace only", "content with a trailing newline inside the token", "other content", "no token")


def main(p: Path):
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    d = json.loads(p.read_text())
    n = defaultdict(dict)
    for r in d["rows"]:
        n[r["arm"]][r["count"]] = r["n"]
    if set(n) != set(ARMS):
        raise SystemExit(f"arms differ: {sorted(n)}")
    per, ok_fail, ok_r1, ok_c1 = {}, True, True, True
    for a in ARMS:
        c = n[a]; runs = c["runs"]
        fails = {k: v for k, v in c.items() if k.startswith("render failure")}
        win, r1 = c.get("outputs in Olmo's window (R0 token below the cut)", 0), c.get("of those, with an R1 readout", 0)
        outs, one = c.get("tool outputs", 0), c.get("kept by exactly one reader (C1 R1 vs Qwen3-32B)", 0)
        share = lambda rule, cut: {k: c.get(f"readout {rule} at {cut}: {k}", 0) for k in CATS}  # noqa: E731
        res = {"runs": runs, "render failures": sum(fails.values()), "render failures by kind": fails,
               "render failure share": sum(fails.values()) / runs, "R1 readout share of outputs in Olmo's window": r1 / win if win else None,
               "outputs kept by exactly one reader, share of outputs": one / outs if outs else None,
               "truncated runs: Olmo, Qwen3-32B": [c.get("truncated: Olmo", 0), c.get("truncated: Qwen3-32B", 0)],
               "tool outputs": outs, "tool outputs with no Qwen3-32B position": c.get("tool outputs with no Qwen3-32B position", 0),
               "R1 and R0 differ": c.get("R1 and R0 differ", 0)}
        for rule in ("R0", "R1"):
            for cut in ("C1", "C2"):
                s_ = share(rule, cut); tot = sum(s_.values())
                res[f"readout {rule} at {cut}: share by category"] = {k: (v / tot if tot else None) for k, v in s_.items()} | {"n": tot}
                ni = c.get(f"insert outputs ending with the review note, readout {rule} at {cut}", 0)
                inside = c.get(f"insert outputs ending with the review note, readout {rule} at {cut}: inside the note", 0)
                res[f"insert outputs ending with the review note, {rule} at {cut}"] = {"n": ni, "readout inside the note": inside, "share": inside / ni if ni else None}
        ok_fail &= res["render failure share"] <= 0.005
        ok_r1 &= res["R1 readout share of outputs in Olmo's window"] is not None and res["R1 readout share of outputs in Olmo's window"] >= 0.99
        ok_c1 &= res["outputs kept by exactly one reader, share of outputs"] is not None and res["outputs kept by exactly one reader, share of outputs"] <= 0.02
        per[a] = res
    reading = "go" if ok_fail and ok_r1 and ok_c1 else "go with C2 only" if ok_fail and ok_r1 else "no-go"
    why = [w for w, ok in (("render failures above 0.5% in some arm", ok_fail), ("R1 readout below 99% of window outputs in some arm", ok_r1),
                           ("C1 kept sets differ by more than 2% in some arm", ok_c1)) if not ok]
    out = {"plan": "PLAN.md (this folder), go / no-go", "git": head, "render file": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()},
           "lengths": d["lengths"], "by arm": per, "reading (pre-set)": reading, "conditions not met": why}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"render_reading_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
