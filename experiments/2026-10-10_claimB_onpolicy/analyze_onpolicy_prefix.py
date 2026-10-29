"""Follow-up (c), the prefix-only clean version (local, free; from run_onpolicy_scores_modal.py's per-output scores and the
staged level-0 text). A run is scored only on its outputs before its first rule-type output, the outputs no fix rule could
have touched and whose context holds no fixed text: a text_editor insert, str_replace or create output (R1, R2, R8), a
dpkg output (R5), or any output holding a parseable clock stamp (as the review's prefix test v2, analyze_review_checks.py;
create added because round two's R8 changes creates). Rule types are read on the level-0 text of each run (a planted copy
uses its twin's); the same (step, action) set serves level 1. Per level, for fixed sim vs real run 1 and real run 2 vs run
1, on the 5 test environments and all 10: run-level AUROC of the mean over all outputs (the refit reproduced the grid's
run scores, run_onpolicy_scores_modal.py) and over prefix outputs only, the runs and outputs that qualify per side, a 95% main-task interval, and a
probe-fixed null (1,000 draws, one coin per main task swapping its cells' two runs). Writes results/onpolicy_prefix_<stamp>.json.

  PYTHONPATH=src:experiments/2026-10-09_fixed_sim uv run --no-project --with numpy --with scikit-learn --with scipy --with 'modal>=1.4.3' \
      python experiments/2026-10-10_claimB_onpolicy/analyze_onpolicy_prefix.py data/onpolicy_output_scores_<stamp>.json.gz
"""

import gzip
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
FS = HERE.parent / "2026-10-09_fixed_sim"
sys.path.insert(0, str(FS))
L0 = HERE / "data" / "2026-10-10_la_september-onpolicy-l0_v1"
TEST = set(json.loads((FS / "split.json").read_text())["test"])
POS = {"fixed_vs_e1": "sim-attack-fixed", "e2_vs_e1": "real-attack-e2"}
DRAWS, BOOT, SEED = 1000, 1000, 0


def prefix_of(messages: list[dict]) -> set[tuple[int, int]]:
    """(step, action) of the outputs before the first rule-type output, in transcript order."""
    import fixes
    from analyze_review_checks import rule_type

    keep = set()
    for c in fixes.calls_of(messages):
        text = messages[c["msg"]]["content"] if isinstance(messages[c["msg"]]["content"], str) else ""
        if rule_type(c["tool"], c["args"], text, True) or (c["tool"] == "text_editor" and c["args"].get("command") == "create"):
            break
        keep.add((c["step"], c["action"]))
    return keep


def evaluate(rows: list[dict], pos: str, keep: dict | None, rng) -> dict:
    """Run-level AUROC of the mean output score (over `keep[key]` outputs only, if given), interval and null."""
    by = {}
    for r in rows:
        if keep is None or (r["step"], r["action"]) in keep.get(r["key"].split("__plant")[0], set()):
            by.setdefault(r["key"], []).append(r)
    cells = {}
    for k in by:  # exactly one run per side per cell, so the null swaps every retained run
        cells.setdefault(by[k][0]["cell"], []).append(by[k][0]["arm"] == pos)
    whole = {c for c, v in cells.items() if sorted(v) == [False, True]}
    dropped = sum(by[k][0]["cell"] not in whole for k in by)
    keys = sorted(k for k in by if by[k][0]["cell"] in whole)
    y = np.array([by[k][0]["arm"] == pos for k in keys], bool)
    s = np.array([np.mean([x["score"] for x in by[k]]) for k in keys])
    task, cell = (np.array([by[k][0][f] for k in keys]) for f in ("main_task", "cell"))
    out = {"runs": {"positive": int(y.sum()), "negative": int((~y).sum()), "dropped_without_their_pair": int(dropped)},
           "outputs": {"positive": int(sum(len(by[k]) for k in keys if by[k][0]["arm"] == pos)),
                       "negative": int(sum(len(by[k]) for k in keys if by[k][0]["arm"] != pos))}}
    if len(set(y)) < 2:
        return out | {"auroc": None}
    tasks = np.unique(task)
    ix = {t: np.flatnonzero(task == t) for t in tasks}
    boot = [roc_auc_score(y[i], s[i]) for i in (np.concatenate([ix[t] for t in rng.choice(tasks, len(tasks))]) for _ in range(BOOT)) if len(set(y[i])) == 2]
    pairs = [np.flatnonzero(cell == c) for c in np.unique(cell)]
    pairs = [p for p in pairs if len(p) == 2 and y[p].sum() == 1]
    t_of = np.unique([task[p[0]] for p in pairs])
    null = []
    for _ in range(DRAWS):
        flip, yy = dict(zip(t_of, rng.random(len(t_of)) < 0.5)), y.copy()
        for p in pairs:
            if flip[task[p[0]]]:
                yy[p] = yy[p[::-1]]
        null.append(roc_auc_score(yy, s))
    auc = roc_auc_score(y, s)
    return out | {"auroc": float(auc), "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                  "null_p95": float(np.percentile(null, 95)), "p": float((1 + sum(v >= auc for v in null)) / (1 + DRAWS))}


def main(path: str):
    d = json.loads(gzip.decompress(Path(path).read_bytes()))
    bad = [k for k, v in d["scores"].items() for c in v["check"].values()
           if not (c["converged"] and np.isfinite(c["corr_with_saved"]) and c["corr_with_saved"] >= 0.999)]
    bad += [k for k, v in d["scores"].items() if not all(np.isfinite(r["score"]) for r in v["rows"])]
    if bad:
        raise SystemExit(f"refits failed their checks in {sorted(set(bad))}: not analyzed")
    keep = {}
    for f in sorted((L0 / "inputs").glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            keep[r["key"]] = prefix_of(r["messages"])
    rng, out = np.random.default_rng(SEED), {"source": Path(path).name, "git_of_scores": d["git"], "rows": {}}
    for name, v in sorted(d["scores"].items()):
        level, comp = name.split("|")
        if comp not in POS:
            continue
        for scope, envs in (("test_envs", TEST), ("all_envs", None)):
            rows = [r for r in v["rows"] if envs is None or r["environment"] in envs]
            out["rows"][f"{name}|{scope}"] = {"all_outputs": evaluate(rows, POS[comp], None, rng), "prefix_only": evaluate(rows, POS[comp], keep, rng)}
            a, p = out["rows"][f"{name}|{scope}"]["all_outputs"], out["rows"][f"{name}|{scope}"]["prefix_only"]
            if a["auroc"] is None or p["auroc"] is None:
                print(f"{name} {scope}: no AUROC (a side is empty) | all {a['runs']} | prefix {p['runs']}")
                continue
            print(f"{name} {scope}: all outputs {a['auroc']:.4f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] p95 {a['null_p95']:.3f} | prefix only "
                  f"{p['auroc']:.4f} [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] p95 {p['null_p95']:.3f}, runs {p['runs']}, outputs {p['outputs']}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"onpolicy_prefix_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", HERE / "results" / f"onpolicy_prefix_{stamp}.json")


if __name__ == "__main__":
    main(sys.argv[1])
