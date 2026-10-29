"""H1 amendment 1's $0 gate (written before it ran): on the dev environments, does the frozen L63 probe separate
simulated from real e1 bash outputs at identical calls in B-dev's evened runs (gate 1: G_dev's 95% lower bound > 0), and
is the projected SE of the output share small enough (gate 2: SE(mean d) / G_dev <= 0.15)? Keys and numbers only.

Scores: d9's per-output L63 scores of B-dev's runs (`claimB/q11_remainder_v1/records.json`, `after` = the evened text,
from B-dev's frozen refits). Pairs: identical bash calls inside the cut, matched as in `analyze_identical_calls.py`,
joined by the pack's output_end (step, action) at the call's ordinal (alignment checked as there). Intervals: 2,000
main-task draws, default_rng(20261011). Also reported (not gates): the same for e2 - e1, and before vs after the first
detected file change.

  PYTHONPATH=src uv run --no-project --python 3.14 --with 'modal>=1.4.3' --with numpy --with boto3 --with pyyaml \\
      python experiments/2026-10-11_claimB_h1_text_vs_layout/analyze_dev_gate.py
"""

import hashlib
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

import analyze_identical_calls as IC

HERE = Path(__file__).parent
RECORDS = "claimB/q11_remainder_v1/records.json"
SEED, DRAWS, SE_BAR = 20261011, 2000, 0.15
REC_ARM = {"sim": "sim-attack-fixed", "e1": "real-attack-e1", "e2": "real-attack-e2"}


def boot(v, rng):
    tasks = sorted({m for m, _ in v})
    by = defaultdict(list)
    for m, d in v:
        by[m].append(d)
    sums, ns = np.array([sum(by[m]) for m in tasks]), np.array([len(by[m]) for m in tasks])
    w = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=DRAWS)
    b = (w @ sums) / (w @ ns)
    return {"pairs": int(ns.sum()), "main_tasks": len(tasks), "mean": float(sums.sum() / ns.sum()),
            "ci95": [float(x) for x in np.percentile(b, [2.5, 97.5])], "se": float(b.std())}


def main():
    ex = json.loads(IC.EXCLUSIONS.read_text())
    env_of = json.loads(IC.FOLDS.read_text())["main_task_environment"]
    cells = [c for c in ex["paired_attack_cells"]["cells"] if env_of[c.split(":")[0]] not in IC.TEST]
    chosen, cell_set = {}, set(cells)
    for s in sorted(ex["samples"], key=lambda s: s["key"]):
        if s["arm"] in IC.ARMS and s["cell"] in cell_set:
            chosen.setdefault((IC.ARMS[s["arm"]], s["cell"]), s)
    jobs = defaultdict(list)
    for (arm, cell), s in chosen.items():
        jobs[str(IC.DATA / s["cohort"] / s["file"])].append((f"{arm}|{cell}", s["sample_id"], s["epoch"]))
    with ProcessPoolExecutor(14) as pool:
        rec = dict(r for part in pool.map(IC.load_file, jobs.items()) for r in part)

    vol = modal.Volume.from_name(IC.VOL, version=2)
    runs_raw = b"".join(vol.read_file(f"{IC.PACK}/runs.L19.json"))
    pos = {}
    for r in json.loads(runs_raw):
        a = {v: k for k, v in IC.PACK_ARM.items()}.get(r["arm"])
        if a and r["cell"] in cell_set:
            pos[(a, r["cell"])] = (r["key"], [(p["step"], p["action"], p["kept"]) for p in r["positions"] if p["kind"] == "output_end"])
    rec_raw = b"".join(vol.read_file(RECORDS))
    after = {}
    for x in json.loads(rec_raw):
        k = (x["cell"], x["arm"], x["step"], x["action"])
        if k in after:
            raise SystemExit("duplicate record")
        after[k] = x["after"]

    def score(a, c, i):
        key, ps = pos[(a, c)]
        if key != chosen[(a, c)]["key"] or len(ps) != len(rec[f"{a}|{c}"]["calls"]) or not rec[f"{a}|{c}"]["results_in_call_order"]:
            raise SystemExit(f"{a}|{c}: the pack run does not line up with the archive sample")
        if not ps[i][2]:
            return None, "outside_cut"
        v = after.get((c, REC_ARM[a], ps[i][0], ps[i][1]))
        return (v, None) if v is not None and np.isfinite(v) else (None, "no_record")

    rows, dropped = defaultdict(list), defaultdict(int)
    for a, b in (("sim", "e1"), ("e2", "e1")):
        for c in cells:
            ca, cb = rec[f"{a}|{c}"]["calls"], rec[f"{b}|{c}"]["calls"]
            qa, qb = defaultdict(list), defaultdict(list)
            for i, x in enumerate(ca):
                qa[x["key"]].append(i)
            for i, x in enumerate(cb):
                qb[x["key"]].append(i)
            for k in qa.keys() & qb.keys():
                for i, j in zip(qa[k], qb[k]):
                    if ca[i]["family"] != "bash":
                        continue
                    (sa, ra), (sb, rb) = score(a, c, i), score(b, c, j)
                    if ra or rb:
                        dropped[f"{a}_vs_{b}|{ra or rb}"] += 1
                        continue
                    pre = ca[i]["prior_edits"] == 0 and cb[j]["prior_edits"] == 0
                    for sub in ("all", "before_first_change" if pre else "after_first_change"):
                        rows[(f"{a}_vs_{b}", sub)].append((c.split(":")[0], sa - sb))
    rng = np.random.default_rng(SEED)
    out = {"label": "H1 amendment 1 dev gate ($0; keys and numbers only; preliminary until reviewed)", "pairs_dropped": dict(dropped),
           "rows": {f"{t}|{sub}": boot(v, rng) for (t, sub), v in sorted(rows.items())}}
    g = out["rows"]["sim_vs_e1|all"]
    proj = g["se"] / g["mean"] if g["mean"] > 0 else float("inf")
    out["gate"] = {"G_dev": g["mean"], "G_dev_ci95": g["ci95"], "gate1_lower_bound_above_0": g["ci95"][0] > 0,
                   "projected_se_of_S": proj, "gate2_projected_se_at_most": SE_BAR, "gate2_pass": proj <= SE_BAR}
    out["gate"]["enough"] = out["gate"]["gate1_lower_bound_above_0"] and out["gate"]["gate2_pass"]
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": {"records_json": hashlib.sha256(rec_raw).hexdigest(), "pack_runs_json": hashlib.sha256(runs_raw).hexdigest(),
                       IC.EXCLUSIONS.name: hashlib.sha256(IC.EXCLUSIONS.read_bytes()).hexdigest()}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"dev_gate_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("pairs dropped:", dict(dropped))
    for k, v in out["rows"].items():
        print(k, {x: (round(y, 4) if isinstance(y, float) else [round(z, 4) for z in y] if isinstance(y, list) else y) for x, y in v.items()})
    print("gate:", out["gate"])
    print("->", dst)


if __name__ == "__main__":
    main()
