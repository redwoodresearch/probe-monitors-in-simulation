"""By-key comparison of this sweep with 2e's (Amendment 22d), on Modal CPU. Both files are read from their commits. A key
disagrees if any rate, difference or band endpoint differs by more than 0.5 points, or k or the grey flag differs.
Keys in only one table are listed, not compared.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run \\
      experiments/2026-10-14_fpr_sweep_xcheck/check_compare_2e.py
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
APP = "probe-shift_aswin-surya_sub-fpr_xcheck_kill-2026-11-14"
MINE = ("ab095924e", "experiments/2026-10-14_fpr_sweep_xcheck/results/sweep_xcheck_2026-10-14_130926.jsonl")
THEIRS = ("0cb12e437", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_sweep_2026-10-14_125005.json")
TOL = 0.5  # points

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=1, memory=2048, timeout=600, retries=0)
def compare(mine_lines: list, theirs: dict) -> dict:
    ro_map = {"averaged_reply": "reply", "command_call_end": "command"}
    pool_map = {"primary_honest_B_u_C": ("BC", ("honest_b", "honest_c")), "sensitivity_honest_B": ("B", ("honest_b",)),
                "sensitivity_honest_A_u_B": ("AB", ("honest_a", "honest_b"))}
    lv_map = {"unit_level": ("unit", "fpr_unit", "realized_run_false_alarm_any_unit"), "run_level": ("run_mean", "fpr_run", "realized_run_false_alarm")}
    dmap = {"sim_minus_real_mean": "sim_minus_real", "sim_minus_run1": "sim_minus_run1", "sim_minus_run2": "sim_minus_run2",
            "run2_minus_run1": "run2_minus_run1"}
    key = lambda ro, lv, r, p, s: (ro, lv, f"{r:.6e}", p, s)  # noqa: E731
    T, extra = {}, {}
    for tro, block in theirs["sweep"].items():
        ro = ro_map[tro]
        extra[ro] = {"cohort": block["cohort_tasks"], "universe": block["universe_tasks"], "honest_units": block["honest_units"],
                     "honest_runs": block["honest_runs"]}
        for tlv, (lv, rate_field, fa_field) in lv_map.items():
            for tp, (p, arms) in pool_map.items():
                for r in block[tlv][tp]:
                    nunits = sum(block["honest_units" if lv == "unit" else "honest_runs"][a] for a in arms)
                    rfu = r.get("realized_unit_fpr")
                    common = {"k": r["k"], "threshold": r["threshold"], "greyed": bool(r.get("greyed_lt10_honest_units_above", False)),
                              "n_pool": nunits,
                              "pool_above": round(sum(rfu[a] * block["honest_units"][a] for a in arms)) if lv == "unit"
                              else round(sum(r[fa_field][a] * block["honest_runs"][a] for a in arms)),
                              "pool_runs_flagged": sum(r[fa_field][a] * block["honest_runs"][a] for a in arms) / sum(block["honest_runs"][a] for a in arms),
                              "honest_a_runs_flagged": r[fa_field]["honest_a"],
                              **{f"tpr_{a}": r["catch"][a] for a in ("run1", "run2", "sim")}, "tpr_real": r["catch"]["real_mean"],
                              **{f"d_{dmap[d]}": v for d, v in r["diff"].items()}}
                    for s, src in (("env", r), ("fold", r["fold_stratified"])):
                        b = {f"tpr_{a if a != 'real_mean' else 'real'}_ci{i}": src["catch_ci95"][a][i] for a in src["catch_ci95"] for i in (0, 1)}
                        b |= {f"d_{dmap[d]}_ci{i}": src["diff_ci95"][d][i] for d in src["diff_ci95"] for i in (0, 1)}
                        T[key(ro, lv, r[rate_field], p, s)] = common | b
    M, listing = {}, None
    for line in mine_lines:
        r = json.loads(line)
        if "listing" in r:
            listing = r["listing"]
        if "readout" not in r:
            continue
        v = {"k": r["k"], "threshold": r["threshold"], "greyed": r["greyed"], "n_pool": r["n_pool"],
             "pool_above": r["pool_above"], "pool_runs_flagged": r["pool_runs_flagged"],
             "honest_a_runs_flagged": r["honest_a_runs_flagged"], **{f"tpr_{a}": x for a, x in r["tpr"].items()},
             **{f"d_{d}": x["point"] for d, x in r["delta"].items()}}
        v |= {f"tpr_{a}_ci{i}": x[i] for a, x in r["tpr_ci95"].items() for i in (0, 1)}
        v |= {f"d_{d}_ci{i}": x["ci95"][i] for d, x in r["delta"].items() for i in (0, 1)}
        M[key(r["readout"], r["level"], r["rate"], r["pool"], r["strata"])] = v
    exact = ("k", "greyed", "n_pool", "pool_above")
    rows, worst = [], {}
    for k in sorted(set(M) & set(T)):
        m, t = M[k], T[k]
        fields = sorted((set(m) & set(t)) - set(exact) - {"threshold"})
        diff = {f: abs(m[f] - t[f]) for f in fields if m[f] is not None and t[f] is not None}
        diff["threshold"] = abs(m["threshold"] - t["threshold"])
        for f, x in diff.items():
            worst[f] = max(worst.get(f, 0.0), x)
        bad = [f for f in fields if f in diff and 100 * diff[f] > TOL] + [f for f in exact if m[f] != t[f]]
        rows.append({"key": list(k), "disagree": bad, "mine": {f: m[f] for f in bad}, "theirs": {f: t[f] for f in bad}, "abs_diff": diff})
    no_unit_all = None  # my universe = T196 tasks minus those with no unit in any arm of either readout
    for ro in listing:
        s = set.intersection(*[set(v) for v in listing[ro]["no_units"].values()])
        no_unit_all = s if no_unit_all is None else no_unit_all & s
    mine_extra = {ro: {"cohort": listing[ro]["cohort_n"]} for ro in listing} | {"universe": 196 - len(no_unit_all)}
    return {"compared": len(rows), "disagreeing": sum(bool(r["disagree"]) for r in rows), "max_abs_diff": worst,
            "only_mine": sorted(map(list, set(M) - set(T))), "only_theirs": sorted(map(list, set(T) - set(M))),
            "counts": {"mine": mine_extra, "theirs": extra}, "rows": rows}


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
    print(f"compared {res['compared']} keys; disagreeing (> {TOL} pts, or k / grey / counts unequal) {res['disagreeing']}")
    print("max abs diff:", {f: round(v, 6) for f, v in sorted(res["max_abs_diff"].items())})
    print("counts:", json.dumps(res["counts"]))
    for r in res["rows"]:
        if r["disagree"]:
            print("DISAGREE", r["key"], {f: (r["mine"][f], r["theirs"][f]) for f in r["disagree"]})
    print(f"only mine ({len(res['only_mine'])}):", sorted({(k[0], k[1]) for k in res["only_mine"]}))
    print(f"only 2e ({len(res['only_theirs'])}):", res["only_theirs"][:12])
