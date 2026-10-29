"""H1 planning: does step A's probe separate simulated from real outputs at identical calls, and how much spread is
there? Local and free; keys and numbers only (stored per-output scores, no text). Written before it ran.

The whole-output swap (Aswin 10:22:01 PT via 0d) can only move the probe at identical calls if the probe's per-output
score already differs there. Here, on the 5 test environments, with the identical-call pairs of
`analyze_identical_calls.py` (inside the 32k cut):
- d = score(simulated output) - score(real e1 output) under step A's fixed-vs-e1 per-output scores (the catch table's
  input), per pair; its mean, by tool family (bash, text_editor), all identical calls and the pre-edit subset;
- the same for real e2 - real e1 under step A's e2-vs-e1 per-output scores (its own probes: the pipeline's
  real-vs-real baseline; e2 has no per-output scores under the fixed-vs-e1 probes);
- the overall per-output gap (every kept simulated output minus every kept real e1 output, same family) for scale;
- the task-clustered standard error of mean d (2,000 main-task draws, default_rng(20261011)). 2.486 x SE is reported
  only as a rough guide to H1's sensitivity: it is the SE of natural differences across different histories, not of
  paired score changes after a swap (H1's own pilot sizes that). A zero gap here would not prove swaps can't move scores.
"Pre-edit" means no detected edit before the call (the failure-hiding heuristic: editor writes and common bash write
patterns); other writes can still make the two runs' states differ.

  PYTHONPATH=src uv run --no-project --python 3.14 --with 'modal>=1.4.3' --with numpy --with boto3 --with pyyaml \\
      python experiments/2026-10-11_claimB_h1_text_vs_layout/analyze_identical_call_gap.py
"""

import gzip
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

import analyze_identical_calls as IC  # same folder: calls_of, load_file, the pack and archive paths

HERE = Path(__file__).parent
SCORES = Path.home() / "probe-shift-sims/experiments/2026-10-09_fixed_sim/data"  # git-ignored; hashed below
FILES = {"fixed_vs_e1": SCORES / "claimB_output_scores_A_2026-10-10_022646.json.gz",
         "e2_vs_e1": SCORES / "claimB_output_scores_A_e2_vs_e1_2026-10-10_024204.json.gz"}
SEED, DRAWS = 20261011, 2000


def main():
    ex = json.loads(IC.EXCLUSIONS.read_text())
    cells = [c for c in ex["paired_attack_cells"]["cells"]]
    env_of = json.loads(IC.FOLDS.read_text())["main_task_environment"]
    cells = [c for c in cells if env_of[c.split(":")[0]] in IC.TEST]
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
    pos = {}  # (arm, cell) -> (sample key, [(step, action, kept)] of output_end positions in order)
    for r in json.loads(runs_raw):
        a = {v: k for k, v in IC.PACK_ARM.items()}.get(r["arm"])
        if a and r["cell"] in cell_set:
            pos[(a, r["cell"])] = (r["key"], [(p["step"], p["action"], p["kept"]) for p in r["positions"] if p["kind"] == "output_end"])
    score = {}  # (comparison, arm, sample key, step, action) -> per-output score
    for comp, f in FILES.items():
        for r in json.loads(gzip.decompress(f.read_bytes()))["levels"]["level1"]["rows"]:
            if (comp, r["arm"], r["key"], r["step"], r["action"]) in score:
                raise SystemExit(f"{comp}: duplicate score row")
            score[(comp, r["arm"], r["key"], r["step"], r["action"])] = r["score"]

    def sc(comp, a, c, i):
        """(score, None) or (None, reason): outside the cut is expected; anything else is a failed join and stops."""
        key, ps = pos[(a, c)]
        if key != chosen[(a, c)]["key"] or len(ps) != len(rec[f"{a}|{c}"]["calls"]) or not rec[f"{a}|{c}"]["results_in_call_order"]:
            raise SystemExit(f"{a}|{c}: the pack run does not line up with the archive sample")
        if not ps[i][2]:
            return None, "outside_cut"
        v = score.get((comp, IC.PACK_ARM[a], key, ps[i][0], ps[i][1]))
        if v is None or not np.isfinite(v):
            raise SystemExit(f"{a}|{c}: a kept output has no finite score in {comp}")
        return v, None

    rows = defaultdict(list)  # (pair type, subset, family) -> [(main task, d)]
    missing = defaultdict(int)
    for a, b, comp in (("sim", "e1", "fixed_vs_e1"), ("e2", "e1", "e2_vs_e1")):
        for c in cells:
            ca, cb = rec[f"{a}|{c}"]["calls"], rec[f"{b}|{c}"]["calls"]
            qa, qb = defaultdict(list), defaultdict(list)
            for i, x in enumerate(ca):
                qa[x["key"]].append(i)
            for i, x in enumerate(cb):
                qb[x["key"]].append(i)
            for k in qa.keys() & qb.keys():
                for i, j in zip(qa[k], qb[k]):
                    (sa, ra), (sb, rb) = sc(comp, a, c, i), sc(comp, b, c, j)
                    fam = "bash" if ca[i]["family"] == "bash" else ("text_editor" if ca[i]["family"].startswith("text_editor") else "other")
                    if ra or rb:
                        missing[f"{a}_vs_{b}|{ra or rb}"] += 1
                        continue
                    pre = ca[i]["prior_edits"] == 0 and cb[j]["prior_edits"] == 0
                    for sub in ("all", "pre_edit") if pre else ("all",):
                        for fm in (fam, "any"):
                            rows[(f"{a}_vs_{b}", sub, fm)].append((c.split(":")[0], sa - sb))

    overall = {}
    for comp, (a, b) in (("fixed_vs_e1", ("sim-attack-fixed", "real-attack-e1")), ("e2_vs_e1", ("real-attack-e2", "real-attack-e1"))):
        rs = [r for r in json.loads(gzip.decompress(FILES[comp].read_bytes()))["levels"]["level1"]["rows"] if r["environment"] in IC.TEST]
        for fm in ("bash", "text_editor", "any"):
            sel = [r for r in rs if fm == "any" or (r["tool"] == fm if fm == "text_editor" else r["tool"].endswith("bash"))]
            overall[f"{comp}|{fm}"] = float(np.mean([r["score"] for r in sel if r["arm"] == a]) - np.mean([r["score"] for r in sel if r["arm"] == b]))

    rng = np.random.default_rng(SEED)
    out = {"label": "H1 planning: per-output score differences at identical calls, test environments (numbers only; preliminary until reviewed)",
           "pairs_dropped": dict(missing), "overall_per_output_gap": overall, "rows": {}}
    for (t, sub, fm), v in sorted(rows.items()):
        tasks = sorted({m for m, _ in v})
        by = defaultdict(list)
        for m, d in v:
            by[m].append(d)
        sums, ns = np.array([sum(by[m]) for m in tasks]), np.array([len(by[m]) for m in tasks])
        w = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=DRAWS)
        boot = (w @ sums) / (w @ ns)
        mean, se = float(sums.sum() / ns.sum()), float(boot.std())
        out["rows"][f"{t}|{sub}|{fm}"] = {"pairs": int(ns.sum()), "main_tasks": len(tasks), "mean_d": mean, "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                                          "se": se, "sd_d": float(np.std([d for _, d in v])), "mde80_one_sided": 2.486 * se}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": {**{f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in FILES.values()}, "pack_runs_json": hashlib.sha256(runs_raw).hexdigest(),
                       IC.EXCLUSIONS.name: hashlib.sha256(IC.EXCLUSIONS.read_bytes()).hexdigest()}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"identical_call_gap_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("pairs dropped:", dict(missing))
    print("overall per-output gap:", {k: round(v, 3) for k, v in overall.items()})
    for k, v in out["rows"].items():
        if not k.endswith("|other"):
            print(k, {x: (round(y, 4) if isinstance(y, float) else [round(z, 4) for z in y] if isinstance(y, list) else y) for x, y in v.items()})
    print("->", dst)


if __name__ == "__main__":
    main()
