"""Queue item 2, steps A–C (PLAN.md in this folder, committed before any number): what drives step A's fixed-sim probe's
run score, how much sits on the four leftover slips, and a score-level rescore with slip-carrying outputs dropped on both
sides (placebo drops and planted check 2 beside it). Local CPU, reads stored scores and the build's level-0 texts.

  PYTHONPATH=src uv run --no-project --python 3.13 --with numpy --with scikit-learn --with pyyaml --with 'modal>=1.4.3' \\
      python experiments/2026-10-10_claimB_probe_attribution/analyze_attribution_abc.py <fixed-sim build dir>
Writes results/attribution_abc_<stamp PT>.json (numbers only) and prints a summary.
"""

import gzip
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
FS = HERE.parent / "2026-10-09_fixed_sim"
sys.path.insert(0, str(FS))
import analyze_tpr_at_fpr as T  # noqa: E402  (cross-fit thresholds, task bootstrap, AUROC)
import fixes  # noqa: E402  (calls_of: the reader's step/action indexing)
import run_tfidf_slips_evened_modal as SL  # noqa: E402  (the four slips, as evened for TF-IDF)

SCORES = FS / "data/claimB_output_scores_A_2026-10-10_022646.json.gz"
PLANT2 = FS / "data/step_a_output_scores_e402bcc321a8/level1/output_scores_A_plant2_e402bcc321a8.jsonl.gz"
SIM, REAL, SEED, DRAWS, PLACEBOS = "sim-attack-fixed", "real-attack-e1", 20261010, 2000, 20


def texts(build: Path) -> dict:
    """(run key, step, action) -> (tool, editor command, output text, slips) from the build's level-0 inputs."""
    out = {}
    for f in sorted((build / "inputs").glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["arm"] not in ("sim-attack", REAL):
                continue
            for c in fixes.calls_of(r["messages"]):
                text = r["messages"][c["msg"]]["content"]
                text = text if isinstance(text, str) else json.dumps(text)
                cmd = c["args"].get("command") if c["tool"] == "text_editor" else None
                _, hit = SL.even(text, cmd, c["tool"])
                if SL.RANGE in text:
                    hit = hit + ["range"]
                out[(r["key"], c["step"], c["action"])] = (c["tool"], cmd, hit)
    return out


def category(tool, cmd):
    return f"text_editor {cmd}" if tool == "text_editor" else tool


def main(build: Path):
    rows = [r for r in json.loads(gzip.decompress(SCORES.read_bytes()))["levels"]["level1"]["rows"] if r["environment"] in T.TEST]
    meta = texts(build)
    missing = [r for r in rows if (r["key"], r["step"], r["action"]) not in meta]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(rows)} score rows have no text (e.g. {missing[0]['key']}); join failed")
    runs = defaultdict(list)
    for r in rows:
        tool, cmd, hit = meta[(r["key"], r["step"], r["action"])]
        runs[r["key"]].append({"s": r["score"], "cat": category(tool, cmd), "slips": hit, "arm": r["arm"], "task": r["main_task"]})
    keys = sorted(runs)
    arm = {k: runs[k][0]["arm"] for k in keys}
    task = {k: runs[k][0]["task"] for k in keys}

    # A and B: the sim-minus-real gap in mean run score, split exactly by category (score/n summed per category)
    def contrib(pred):
        return {k: sum(o["s"] for o in runs[k] if pred(o)) / len(runs[k]) for k in keys}

    def gap(c, ks):
        sim = [c[k] for k in ks if arm[k] == SIM]
        real = [c[k] for k in ks if arm[k] == REAL]
        return float(np.mean(sim) - np.mean(real))

    tasks = sorted(set(task.values()))
    by_task = defaultdict(list)
    for k in keys:
        by_task[task[k]].append(k)
    rng = np.random.default_rng(SEED)
    draws = [[k for t in rng.choice(tasks, len(tasks)) for k in by_task[t]] for _ in range(DRAWS)]

    def share(pred):
        c, total = contrib(pred), contrib(lambda o: True)
        point = gap(c, keys) / gap(total, keys)
        boot = [gap(c, d) / gap(total, d) for d in draws]
        return {"gap": gap(c, keys), "share": point, "share_ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])]}

    cats = sorted({o["cat"] for k in keys for o in runs[k]})
    res = {"total_gap": gap(contrib(lambda o: True), keys),
           "by_category": {c: share(lambda o, c=c: o["cat"] == c) for c in cats},
           "slip_carrying": share(lambda o: bool(o["slips"])),
           "by_slip": {s: share(lambda o, s=s: s in o["slips"]) for s in ("create", "view", "order", "range")}}
    res["output_counts"] = {a: {"outputs": sum(len(runs[k]) for k in keys if arm[k] == a),
                                "slip_carrying": sum(bool(o["slips"]) for k in keys if arm[k] == a for o in runs[k])} for a in (SIM, REAL)}
    top = [(sum(bool(o["slips"]) for o in sorted(runs[k], key=lambda o: -o["s"])[:3]) / min(3, len(runs[k])),
            sum(bool(o["slips"]) for o in runs[k]) / len(runs[k])) for k in keys if arm[k] == SIM]
    res["sim_top3_slip_share"] = {"top3": float(np.mean([a for a, _ in top])), "base_rate": float(np.mean([b for _, b in top]))}

    # C: planted check first (drop the str_replace outputs from planted check 2's copies and twins)
    p2 = [json.loads(line) for line in gzip.open(PLANT2, "rt")]
    p2 = [r for r in p2 if r["environment"] in T.TEST]
    sr = {(k, s, a) for (k, s, a), (tool, cmd, _) in meta.items() if tool == "text_editor" and cmd == "str_replace"}
    kept = [r for r in p2 if (r["key"].removesuffix("__plant2"), r["step"], r["action"]) not in sr]
    out = {}
    for name, rs in (("plant2_all_outputs", p2), ("plant2_str_replace_dropped", kept)):
        y, s, t = T.select(T.runs_from_outputs(rs, np.mean), "plant2", T.TEST)
        out[name] = T.evaluate({"id": name, "y": y, "s": s, "task": t})
    res["planted_check"] = {k: {"auroc": v["auroc"], "auroc_ci95": v["auroc_ci95"], "runs": v["positive_runs"] + v["negative_runs"]} for k, v in out.items()}
    res["planted_check"]["passed"] = out["plant2_str_replace_dropped"]["auroc"] <= 0.60

    def evaluate(name, keep):
        """Run score = mean over the outputs `keep` retains; runs left empty are dropped (counted)."""
        sc = {k: [o["s"] for o in runs[k] if keep(k, o)] for k in keys}
        empty = [k for k in keys if not sc[k]]
        live = [k for k in keys if sc[k]]
        y = np.array([arm[k] == SIM for k in live])
        r = T.evaluate({"id": name, "y": y, "s": np.array([np.mean(sc[k]) for k in live]), "task": np.array([task[k] for k in live])})
        return {"auroc": r["auroc"], "auroc_ci95": r["auroc_ci95"], "runs_dropped_empty": len(empty),
                **{f"caught_at_{p}pct": r[f"at_{p}pct"]["caught_share"] for p in T.TARGETS},
                **{f"caught_at_{p}pct_ci95": r[f"at_{p}pct"]["caught_ci95"] for p in T.TARGETS},
                **{f"false_alarms_at_{p}pct": r[f"at_{p}pct"]["fpr"] for p in T.TARGETS}}

    res["C"] = {"all_outputs": evaluate("all", lambda k, o: True), "slips_dropped": evaluate("drop", lambda k, o: not o["slips"])}
    placebo = []
    for i in range(PLACEBOS):  # drop as many random non-slip outputs as each run has slip-carrying ones
        drop = {}
        for k in keys:
            idx = [j for j, o in enumerate(runs[k]) if not o["slips"]]
            n = sum(bool(o["slips"]) for o in runs[k])
            drop[k] = set(random.Random(f"placebo:{i}:{k}").sample(idx, min(n, len(idx))))
        ids = {k: {id(o): j for j, o in enumerate(runs[k])} for k in keys}
        placebo.append(evaluate(f"placebo{i}", lambda k, o: ids[k][id(o)] not in drop[k])["auroc"])
    res["C"]["placebo_auroc"] = {"min": min(placebo), "max": max(placebo), "seeds": PLACEBOS}
    base, dropped = res["C"]["all_outputs"]["auroc"], res["C"]["slips_dropped"]["auroc"]
    reading = ("not read: the planted check failed" if not res["planted_check"]["passed"] else
               "the slips explain most of it" if dropped <= 0.5 + 0.5 * (base - 0.5) and min(placebo) > dropped else
               "the slips explain little" if dropped >= 0.5 + 0.75 * (base - 0.5) else "partly")
    res["C"]["reading"] = reading
    res["C"]["label"] = "score-level diagnostic; ignores context effects (a slip can shift later outputs' activations)"
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    res["provenance"] = {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "scores_sha256": hashlib.sha256(SCORES.read_bytes()).hexdigest(), "plant2_sha256": hashlib.sha256(PLANT2.read_bytes()).hexdigest(),
                         "build_manifest_sha256": hashlib.sha256((build / "manifest.json").read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"attribution_abc_{stamp}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "by_category"}, indent=1))
    print("top categories by share of the gap:")
    for c, v in sorted(res["by_category"].items(), key=lambda x: -abs(x[1]["share"]))[:8]:
        print(f"  {c}: {v['share']:+.3f} [{v['share_ci95'][0]:+.3f}, {v['share_ci95'][1]:+.3f}]")
    print("->", path)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
