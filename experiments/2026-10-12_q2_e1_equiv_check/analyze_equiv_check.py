"""E1 confirmation, the long-context equivalence check: the pre-set reading (PLAN.md here, committed before it runs).
Local, $0. Compares run_equiv_check.py's one-pass scores with the confirmation's stored cached scores.

  uv run --no-project python experiments/2026-10-12_q2_e1_equiv_check/analyze_equiv_check.py \\
      <one_pass json> <archived stage folder (with scores/ and staged.json)>
"""
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
TOL_SD, STAGE1_TOL, D_SD, N_LONGEST = 0.02, 0.0042, 0.5396, 3  # stage 1's rule; its absolute tolerance; E1's frozen SD


def main(one_p: Path, folder: Path):
    me = Path(__file__).resolve()
    for f in (me, HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    one = json.loads(one_p.read_text())
    if hashlib.sha256((folder / "staged.json").read_bytes()).hexdigest()[:16] != one["staged.json sha256 prefix"]:
        raise SystemExit("the archived stage folder isn't the one the one-pass job read")
    stored = {(r["key"], r["seq"]): r for r in (json.loads(p.read_text()) for p in (folder / "scores").glob("*.json"))}
    staged = json.loads((folder / "staged.json").read_text())
    if {(s_["key"], s_["seq"]) for s_ in staged} != set(stored) or any(r.get("oom") or "score" not in r for r in stored.values()):
        raise SystemExit("stored scores don't cover the staged sequences exactly, each with a score")
    honest = [r["score"] for r in stored.values() if r["seq"] == "honest"]
    sd = statistics.stdev(honest); tol = TOL_SD * sd
    keys = one["selected runs (3 longest, then 5 by seed)"]
    want = {(s_["key"], s_["seq"]) for s_ in staged if s_["key"] in set(keys)}
    got = [(r["key"], r["seq"]) for r in one["rows"]]
    if len(got) != len(set(got)) or set(got) != want:
        raise SystemExit("the one-pass rows aren't exactly the selected runs' staged sequences, each once")
    rows, checked, oom = [], 0, 0
    for r in one["rows"]:
        c = stored[(r["key"], r["seq"])]
        if r.get("oom"):
            oom += 1; rows.append({"key": r["key"], "seq": r["seq"], "tokens": r["tokens"], "checked": False}); continue
        if r["ids_sha256"] != c["ids_sha256"] or len(r["one_pass_per_token"]) != len(c["per_token"]):
            raise SystemExit(f"{r['key']} {r['seq']}: the one-pass sequence isn't the stored one")
        checked += 1
        dm = abs(c["score"] - r["one_pass_score"]); dt = max(abs(a - b) for a, b in zip(c["per_token"], r["one_pass_per_token"]))
        rows.append({"key": r["key"], "seq": r["seq"], "tokens": r["tokens"], "checked": True, "report-mean difference": dm,
                     "max per-token difference (stored rounded to 4 dp)": dt, "within tolerance": dm <= tol})
    full = {k for k in keys if all(r["checked"] for r in rows if r["key"] == k)}  # runs with every sequence checked
    longest_checked = bool(full & set(keys[:N_LONGEST]))
    paired = [r for r in rows if r["checked"] and r["seq"] in ("hide", "instructed_honest")
              and {(x["seq"]) for x in rows if x["key"] == r["key"] and x["checked"]} >= {"hide", "instructed_honest"}]
    worst_m = max((r["report-mean difference"] for r in rows if r["checked"]), default=None)
    worst_t = max((r["max per-token difference (stored rounded to 4 dp)"] for r in rows if r["checked"]), default=None)
    worst_pair = max((r["max per-token difference (stored rounded to 4 dp)"] for r in paired), default=None)
    if any(r["checked"] and not r["within tolerance"] for r in rows):
        reading = "not equivalent"
    elif checked >= 0.8 * len(rows) and longest_checked:
        reading = "cached scoring verified on these contexts"
    else:
        reading = "incomplete"
    out = {"plan": "PLAN.md (this folder)", "git": head, "one-pass file": {one_p.name: hashlib.sha256(one_p.read_bytes()).hexdigest()},
           "selected runs": keys, "sequences": len(rows), "checked": checked, "out of memory": oom, "a longest run fully checked": longest_checked,
           "runs fully checked": len(full), "runs with both hide and instructed checked": len({r["key"] for r in paired}),
           "honest report means: n, SD": [len(honest), sd], "tolerance (0.02 x SD)": tol, "stage 1's absolute tolerance": STAGE1_TOL,
           "max report-mean difference": worst_m, "max per-token difference": worst_t,
           "bound on |change in D| for runs with both hide and instructed checked (2 x max per-token / 0.5396)": (2 * worst_pair / D_SD) if worst_pair is not None else None,
           "reading (pre-set)": reading, "rows": rows}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"equiv_check_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
