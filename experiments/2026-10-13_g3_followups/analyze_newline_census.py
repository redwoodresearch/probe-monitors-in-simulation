"""G3 follow-ups (3), the newline census table and PLAN.md (3)'s conclusion rule (POST HOC, preliminary until Codex). Local,
$0: reads run_newline_census.py's record and analyze_newline_sites_b7.py's, prints the table (world x set x family: outputs,
trailing newlines 0/1/2/3+ by my definition as shares, two or more, b7's mean, internal blank-line runs share and mean, b7's
blank-lines mean) and writes it with the rule's evaluation. Written before the census record existed. The rule compares
simulated minus real for shell and view on all outputs in three populations (LinuxArena attack: as simulated vs second real;
LinuxArena honest: as simulated vs real; September: fixed simulated vs real run 1) and at b7's sites (simulated vs first
real); its families are "shell, error field ignored" and "view, error field ignored" in every population, because the
September runs have no error table (Codex), so all four compare the same thing.

  uv run --no-project --python 3.12 python experiments/2026-10-13_g3_followups/analyze_newline_census.py --census <record> --sites <record>
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
FAMS = ("bash", "other shell", "shell (bash + other shell)", "shell, error-like text", "shell, no error-like text", "view", "create",
        "str_replace", "insert", "editor-other", "other", "error (field set)", "shell, error field ignored", "view, error field ignored", "all families")
POPS = {"LinuxArena attack": ("as simulated", "second real"), "LinuxArena honest": ("honest: as simulated", "honest: real"),
        "September on policy": ("September: fixed simulated", "September: real run 1")}


def stats(C, p):
    n = C.get(f"{p} | outputs", 0)
    if not n:
        return None
    tm = [C.get(f"{p} | trailing (mine) | {i}", 0) for i in range(4)]
    return {"outputs": n, "trailing (mine) 0/1/2/3+ (%)": [round(100 * x / n, 2) for x in tm], "two or more (%)": round(100 * (tm[2] + tm[3]) / n, 2),
            "trailing (b7) mean": round(C.get(f"{p} | trailing (b7) | sum", 0) / n, 4),
            "internal runs: share with one or more (%)": round(100 * C.get(f"{p} | internal runs | outputs with one or more", 0) / n, 2),
            "internal runs: mean count": round(C.get(f"{p} | internal runs | sum", 0) / n, 3), "blank lines (b7) mean": round(C.get(f"{p} | blank lines (b7) | sum", 0) / n, 3)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--census", required=True, type=Path); ap.add_argument("--sites", required=True, type=Path); a = ap.parse_args()
    git = lambda *x: subprocess.run(["git", *x], cwd=HERE, capture_output=True, text=True)  # noqa: E731
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):  # the two input records are over 300 lines: kept local, pinned by sha256 in the output and in results/newline_census_inputs_*.json
        if git("ls-files", "--error-unmatch", str(f)).returncode or git("diff", "--quiet", "HEAD", "--", str(f)).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    R, B = json.loads(a.census.read_text()), json.loads(a.sites.read_text())
    C, BC = R["counts (world | set | family | measure | value)"], B["counts (group | world | measure | value)"]
    worlds = sorted({k.split(" | ")[0] for k in C})
    table = {w: {s: {f: st for f in FAMS if (st := stats(C, f"{w} | {s} | {f}"))} for s in ("all outputs", "Qwen3-32B-kept rows", "first output")} for w in worlds}
    sites = {}
    for g in ("all sites", "tool shell", "tool view", "first command (k = 0) | tool shell", "later command (k > 0) | tool shell"):
        for w, lab in (("e1", "first real"), ("e2", "second real"), ("sim", "simulated"), ("sim raw output", "simulated, raw output (new sites)")):
            n = BC.get(f"{g} | {w} | sites", 0)
            if n:
                tm = [BC.get(f"{g} | {w} | trailing (mine) | {i}", 0) for i in range(4)]
                sites.setdefault(g, {})[lab] = {"outputs": n, "trailing (mine) 0/1/2/3+ (%)": [round(100 * x / n, 2) for x in tm],
                                               "two or more (%)": round(100 * (tm[2] + tm[3]) / n, 2), "trailing (b7) mean": round(BC.get(f"{g} | {w} | trailing (b7) | sum", 0) / n, 4),
                                               "internal runs: share with one or more (%)": round(100 * BC.get(f"{g} | {w} | internal runs | outputs with one or more", 0) / n, 2)}
    rule = {}
    for kind, fam, site_g in (("shell", "shell, error field ignored", "tool shell"), ("view", "view, error field ignored", "tool view")):
        d = {}
        for pop, (sim, real) in POPS.items():
            s, r = table[sim]["all outputs"][fam], table[real]["all outputs"][fam]
            d[pop] = {"simulated - real, two or more (points)": round(s["two or more (%)"] - r["two or more (%)"], 2),
                      "simulated - real, b7 mean": round(s["trailing (b7) mean"] - r["trailing (b7) mean"], 4), "outputs (simulated, real)": [s["outputs"], r["outputs"]]}
        s, r = sites[site_g]["simulated"], sites[site_g]["first real"]
        d["b7's 91 sites"] = {"simulated - real, two or more (points)": round(s["two or more (%)"] - r["two or more (%)"], 2),
                              "simulated - real, b7 mean": round(s["trailing (b7) mean"] - r["trailing (b7) mean"], 4), "outputs (simulated, real)": [s["outputs"], r["outputs"]]}
        signs = {(x["simulated - real, two or more (points)"] > 0, x["simulated - real, b7 mean"] > 0) for x in d.values()}
        nonzero = all(x["simulated - real, two or more (points)"] != 0 and x["simulated - real, b7 mean"] != 0 for x in d.values())
        same = nonzero and len(signs) == 1 and len(set(next(iter(signs)))) == 1  # one sign pair everywhere, and the two measures agree
        big = sum(abs(d[p]["simulated - real, two or more (points)"]) >= 2 for p in POPS)
        d["same sign everywhere, both measures"], d["populations (of 3) with |two-or-more difference| >= 2 points"] = same, big
        d["consistent (PLAN.md (3) rule)"] = same and big >= 2
        rule[kind] = d
    out = {"POST HOC": True, "status": "preliminary until Codex", "plan": "PLAN.md (3)", "git": git("rev-parse", "HEAD").stdout.strip(),
           "inputs sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.census, a.sites)},
           "census join checks": R["join checks (runs failing each, by arm; original arms must be 0)"],
           "table (world | set | family)": table, "b7's 91 sites": sites, "conclusion rule (PLAN.md (3))": rule,
           "shell and view trailing newlines differ consistently": all(rule[k]["consistent (PLAN.md (3) rule)"] for k in rule)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    txt = json.dumps(out, separators=(",", ":")) + "\n"  # one line: the table has about 600 rows
    with open(dest := HERE / "results" / f"newline_census_table_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(txt)
    for w in worlds:
        for s in ("all outputs", "Qwen3-32B-kept rows", "first output"):
            for f, st in table[w][s].items():
                print(f"{w:42s} | {s:19s} | {f:28s} | n {st['outputs']:6d} | 0/1/2/3+ {st['trailing (mine) 0/1/2/3+ (%)']} | >=2 {st['two or more (%)']:5.1f} | b7 {st['trailing (b7) mean']:.3f} | runs {st['internal runs: share with one or more (%)']:5.1f}% x{st['internal runs: mean count']:.2f} | blank {st['blank lines (b7) mean']:.2f}")
    print(json.dumps(rule, indent=1)); print(dest)


if __name__ == "__main__":
    main()
