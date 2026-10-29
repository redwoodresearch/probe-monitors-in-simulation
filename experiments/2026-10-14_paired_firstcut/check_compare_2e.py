"""By-key comparison of this first cut (cohort view, pooled A∪B threshold) with 2e's first cut, on Modal CPU. Both
result files are read from their commits; a row disagrees if any rate, Δ or Tango bound differs by more than 0.5 points,
or n or a discordant count differs. Keys present in only one table are listed, not compared.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run \\
      experiments/2026-10-14_paired_firstcut/check_compare_2e.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-paired_firstcut_kill-2026-11-14"
MINE = ("17dbdef2c", "experiments/2026-10-14_paired_firstcut/results/paired_firstcut_2026-10-14_120054.jsonl")
THEIRS = ("2a940845f", "experiments/2026-10-14_priority1_firstcut/results/firstcut_2026-10-14_115701.json")
TOL = 0.5  # points

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=1, memory=2048, timeout=600, retries=0)
def compare(mine_lines: list, theirs: dict) -> dict:
    readout = {"averaged avg_top2": "avg_top2", "command A (content-only call_end, L63)": "command_A",
               "averaged avg_top2 (curve column)": "avg_top2"}
    base = lambda b: "E2" if b.startswith("E2") else "all_runs"  # noqa: E731

    def their_name(r):
        if r["readout"].endswith("(curve column)"):
            return f"adapted_{r['probe']}|avg_top2"
        return f"{'own_a' if r['probe'] == 'own_a' else 'sol_' + r['probe']}|{readout[r['readout']]}"

    T = {}
    for r in theirs["rows"]:
        for comp, (x, y) in (("sim_minus_run2", ("sim", "run2")), ("run2_minus_run1", ("run2", "run1"))):
            b = r[comp]["binary"]
            T[(their_name(r), base(r["base"]), comp, r["alpha"])] = {
                "n": r["n"], "threshold": r["threshold_pooled"], f"catch_{x}": r["catch"][x], f"catch_{y}": r["catch"][y],
                "delta": b["mean"], "tango95_lo": b["tango95"][0], "tango95_hi": b["tango95"][1], "first_only": b["x_only"],
                "second_only": b["y_only"], "mcnemar_p": b["mcnemar_p"], "z_mean": r[comp]["continuous_z"]["mean"],
                "honest_above": round(r["realized_fa_pooled_in_sample"] * r["honest_pooled_n"]), "honest_n": r["honest_pooled_n"]}
    M = {}
    for line in mine_lines:
        r = json.loads(line)
        if "probe" not in r or r["view"] != "cohort" or r["threshold"] != "pooled_AB":
            continue
        name = r["probe"].rsplit("|", 1)[0]
        x, y = r["comparison"].split("_minus_")
        M[(name, r["base"], r["comparison"], r["alpha"])] = {
            "n": r["n_tasks"], "threshold": r["threshold_value"], f"catch_{x}": r[f"catch_{x}"], f"catch_{y}": r[f"catch_{y}"],
            "delta": r["delta"], "tango95_lo": r["ci95"][0], "tango95_hi": r["ci95"][1], "first_only": r["discordant_first_only"],
            "second_only": r["discordant_second_only"], "mcnemar_p": r["mcnemar_p"], "z_mean": r["paired_t_std"]["mean"],
            "honest_above": r["honest_above"], "honest_n": r["honest_n"]}
    rates = ("catch_run1", "catch_run2", "catch_sim", "delta", "tango95_lo", "tango95_hi")
    exact = ("n", "first_only", "second_only", "honest_above", "honest_n")
    rows, worst = [], {f: 0.0 for f in rates + ("threshold", "mcnemar_p", "z_mean")}
    for k in sorted(set(M) & set(T), key=str):
        m, t = M[k], T[k]
        diff = {f: abs(m[f] - t[f]) for f in rates if f in m} | {"threshold": abs(m["threshold"] - t["threshold"]),
                                                                "mcnemar_p": abs(m["mcnemar_p"] - t["mcnemar_p"]),
                                                                "z_mean": abs(m["z_mean"] - t["z_mean"])}
        for f, v in diff.items():
            worst[f] = max(worst[f], v)
        bad = [f for f in rates if f in diff and 100 * diff[f] > TOL] + [f for f in exact if m[f] != t[f]]
        rows.append({"key": list(k), "mine": m, "theirs": t, "abs_diff": diff, "disagree": bad})
    return {"compared": len(rows), "disagreeing": sum(bool(r["disagree"]) for r in rows), "max_abs_diff": worst,
            "only_mine": sorted(map(list, set(M) - set(T)), key=str), "only_theirs": sorted(map(list, set(T) - set(M)), key=str),
            "rows": rows}


def show(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=REPO).stdout


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    mine, theirs = show(*MINE), show(*THEIRS)
    res = compare.remote(mine.decode().splitlines(), json.loads(theirs))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = {"git": git("rev-parse", "HEAD"), "mine": f"{MINE[1]}@{MINE[0]}", "mine_sha256": hashlib.sha256(mine).hexdigest(),
            "theirs": f"{THEIRS[1]}@{THEIRS[0]}", "theirs_sha256": hashlib.sha256(theirs).hexdigest(), "tolerance_points": TOL}
    p = HERE / "results" / f"compare_2e_{stamp}.json"
    with p.open("x") as fh:
        json.dump(head | res, fh, indent=1)
    print(f"wrote {p.name} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"compared {res['compared']} keys; disagreeing (> {TOL} pts or unequal counts) {res['disagreeing']}")
    print("max abs diff:", {f: round(v, 6) for f, v in res["max_abs_diff"].items()})
    for r in res["rows"]:
        if r["disagree"]:
            print("DISAGREE", r["key"], r["disagree"], {f: (r["mine"][f], r["theirs"][f]) for f in r["disagree"]})
    print(f"only mine ({len(res['only_mine'])}):", res["only_mine"])
    print(f"only 2e ({len(res['only_theirs'])}):", res["only_theirs"])
