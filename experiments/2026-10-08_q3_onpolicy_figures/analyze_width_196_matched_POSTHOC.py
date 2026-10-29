"""Matched-cohort sensitivity for width_196: drop the 4 empty-streak tasks from every world (POST HOC). $0, local.

The orchestrator for Aswin, 2026-10-13, after Codex's review of #1300: the as-run sensitivity dropped only the 4 tasks'
simulated rows, so its real - simulated contrast compared the first real run on 191 tasks with the simulated run on 187
(unmatched). Here the 4 tasks (width_196's dropped set, read from its result JSON) leave every world, in evaluation and
calibration alike. Then real - simulated (real->real - real->sim) and first real run - second real run (real->real -
real->run2) are recomputed with (a) thresholds re-estimated in every draw and (b) thresholds fixed at this cohort's
draw-0 values. There are 2,000 draws, seed 20261012, and the retained tasks are resampled with replacement within each
fold. The threshold rule and the statistics are imported from the as-run width script. The probes are not refit (they
were trained with those tasks in). Also records, for #1301's review, the --evals root and the sha256 of every simulated
run file the drop criterion read.

PREDICTION (committed before running): matched real - simulated (a) point within 0.3 points of +0.46 and width within
0.5 of 3.69; first - second real run within 0.3 of +1.98.

  uv run --no-project --python 3.14 --with numpy python -I analyze_width_196_matched_POSTHOC.py --draw0 <json> \\
      --width <width_196 result json> --id-sheet <csv> --evals <bundle_src dir> --out <json>
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze_width_196_POSTHOC import CELLS, DIFFS, FPR, LAUNCHES, NB, SEED, SHA, stats, threshold  # noqa: E402

SHA_WIDTH = "c21444b8245aa574b3e49a45f805b9ffa078adb124c45de61ca037503a4468ff"


def main(draw0: str, width: str, id_sheet: str, evals: str, out: str):
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    git = g("rev-parse", "HEAD").strip()
    for name, p, h in (("draw0", draw0, SHA["draw0"]), ("id_sheet", id_sheet, SHA["id_sheet"]), ("width", width, SHA_WIDTH)):
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h:
            raise SystemExit(f"{name} sha256 mismatch")
    D, Wd = json.load(open(draw0)), json.load(open(width))
    dropped = Wd["sensitivity_empty_replies"]["dropped_tasks"]
    if len(dropped) != 4:
        raise SystemExit("gate failed: width_196 did not drop exactly 4 tasks")
    d0, meta, folds = D["draws"][0], D["meta"], D["folds"]
    world = np.array([m["world"] for m in meta])
    tasks = sorted({m["task"] for m in meta})
    ti = {t: i for i, t in enumerate(tasks)}
    task = np.array([ti[m["task"]] for m in meta])
    y = np.array([m["y"] for m in meta])
    fold_of_task = np.array([folds[t] for t in tasks])
    fold = fold_of_task[task]
    ev = {c: (np.array(d0["cells"][c]["per_call"]["row"]), np.array(d0["cells"][c]["per_call"]["score"])) for c in CELLS}
    cal = {}
    for c in CELLS:
        W, cl = c.split("->")[0], d0["cells"][c]["per_call"]["calibration"]
        rr, ss, mf = np.array(cl["row"]), np.array(cl["score"]), np.array(cl["model_fold"])
        for k in range(5):
            m = (mf == k) & (world[rr] == W) & (y[rr] == 0)
            cal.setdefault((W, k), (rr[m], ss[m]))
    t_stored = {c: {f["fold"]: f["threshold"] for f in d0["cells"][c]["folds"] if f["threshold_kind"] == "frozen"} for c in CELLS}
    drop_idx = [ti[t] for t in dropped]
    keep_all, keep_m = np.ones(len(meta), bool), ~np.isin(task, drop_idx)
    retained = {k: np.array([i for i in np.flatnonzero(fold_of_task == k) if i not in drop_idx]) for k in range(5)}

    def thresholds(c, wt, keep):
        W, ts = c.split("->")[0], {}
        for k in range(5):
            rr, ss = cal[(W, k)]
            m = keep[rr] & (wt[task[rr]] > 0)
            ts[k] = threshold(ss[m], wt[task[rr[m]]], FPR)
        return ts

    def catch_fa(c, wt, keep, ts):
        r, s = ev[c]
        m = keep[r] & (wt[task[r]] > 0)
        r, s = r[m], s[m]
        w = wt[task[r]]
        f = s > np.array([ts[k] for k in fold[r]])
        pos, neg = y[r] == 1, y[r] == 0
        return float((w * f)[pos].sum() / w[pos].sum()), float((w * f)[neg].sum() / w[neg].sum())

    def weights(b):
        w = np.zeros(len(tasks))
        if b == 0:
            w[np.concatenate(list(retained.values()))] = 1
            return w
        rng = np.random.default_rng([SEED, b])
        for k in range(5):
            np.add.at(w, rng.choice(retained[k], len(retained[k]), replace=True), 1)
        return w

    # gate: all tasks at draw 0 reproduce the stored catch and FA, with the stored thresholds and with re-estimated ones
    # (scores are stored to 6 decimals, so the re-estimated thresholds need not equal the stored ones bit for bit)
    for c in CELLS:
        st = d0["cells"][c]["values"]
        for ts in (t_stored[c], thresholds(c, np.ones(len(tasks)), keep_all)):
            ca, fa = catch_fa(c, np.ones(len(tasks)), keep_all, ts)
            if abs(ca - st["catch_frozen"]) > 1e-12 or abs(fa - st["false_alarm_frozen"]) > 1e-12:
                raise SystemExit(f"gate failed: {c}")
    w0 = weights(0)
    t_matched = {c: thresholds(c, w0, keep_m) for c in CELLS}
    res = {meth: {c: [] for c in CELLS} for meth in ("a", "b")}
    for b in range(NB + 1):
        wt = weights(b)
        for c in CELLS:
            res["a"][c].append(catch_fa(c, wt, keep_m, thresholds(c, wt, keep_m))[0])
            res["b"][c].append(catch_fa(c, wt, keep_m, t_matched[c])[0])
    matched = {}
    for meth in ("a", "b"):
        o = {c: stats(v[1:], v[0]) for c, v in res[meth].items()}
        for dname, (ca, cb) in DIFFS.items():
            d = [x - z for x, z in zip(res[meth][ca], res[meth][cb])]
            o[dname] = stats(d[1:], d[0])
        matched[meth] = o
    stored_t_point = {c: catch_fa(c, w0, keep_m, t_stored[c]) for c in CELLS}
    compare = {dname: {"all_tasks": {m: Wd["bootstrap"][m][dname] for m in ("a", "b")},
                       "unmatched_sim_rows_only": {m: Wd["sensitivity_empty_replies"]["bootstrap_without"][m][dname] for m in ("a", "b")},
                       "matched": {m: matched[m][dname] for m in ("a", "b")}} for dname in DIFFS}
    # provenance of the drop criterion's inputs (#1301 review): every simulated run file in the sheet, by sha256
    sim_files = sorted({r["run_key"].rsplit("#e", 1)[0]: r["task"] for r in csv.DictReader(open(id_sheet, newline="")) if r["arm"] == "sim_attack"}.items())
    prov = []
    for key, t in sim_files:
        hits = [Path(evals) / L / key for L in LAUNCHES if (Path(evals) / L / key).exists()]
        if len(hits) != 1:
            raise SystemExit("run file not found exactly once")
        prov.append({"task": t, "launch": hits[0].parent.name if hits[0].parent.parent == Path(evals) else str(hits[0].relative_to(evals)).split("/")[0],
                     "run_key": key, "sha256": hashlib.sha256(hits[0].read_bytes()).hexdigest(), "dropped": t in dropped})
    report = {"label": "POST HOC, descriptive; matched-cohort sensitivity for width_196", "git": git,
              "inputs_sha256": {"draw0": SHA["draw0"], "id_sheet": SHA["id_sheet"], "width_196": SHA_WIDTH}, "n_boot": NB, "seed": SEED,
              "dropped_tasks": dropped, "tasks_retained": int(len(tasks) - len(drop_idx)),
              "rows_dropped_by_world": {w: int(((world == w) & ~keep_m).sum()) for w in ("real", "sim", "run2")},
              "thresholds": {"a": "re-estimated in every draw on the matched cohort", "b": "fixed at the matched cohort's draw-0 values",
                             "matched_draw0": t_matched, "stored": t_stored},
              "matched": matched, "matched_draw0_with_stored_thresholds": stored_t_point, "compare": compare,
              "eval_provenance": {"evals_root": str(Path(evals).resolve()), "simulated_run_files": prov,
                                  "digest_of_sorted_sha256": hashlib.sha256("\n".join(sorted(p["sha256"] for p in prov)).encode()).hexdigest()},
              "written": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")}
    Path(out).write_text(json.dumps(report, indent=1))
    pct = lambda v: f"{100 * v:6.2f}"  # noqa: E731
    print(f"git {git[:8]} | dropped {dropped} from every world; rows dropped {report['rows_dropped_by_world']}; tasks retained {report['tasks_retained']}")
    for c in CELLS:
        a, b = matched["a"][c], matched["b"][c]
        print(f"{c:16s} matched point {pct(a['point'])} (stored thresholds {pct(stored_t_point[c][0])}) | (b) [{pct(b['ci95'][0])},{pct(b['ci95'][1])}] w {pct(b['width'])} | (a) [{pct(a['ci95'][0])},{pct(a['ci95'][1])}] w {pct(a['width'])}")
    for dname, v in compare.items():
        for lab, x in v.items():
            print(f"{dname:16s} {lab:24s} " + " | ".join(f"({m}) {pct(x[m]['point'])} [{pct(x[m]['ci95'][0])},{pct(x[m]['ci95'][1])}] w {pct(x[m]['width'])}" for m in ("a", "b")))
    print(f"simulated run files hashed: {len(prov)} (dropped {sum(p['dropped'] for p in prov)}); digest {report['eval_provenance']['digest_of_sorted_sha256'][:16]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--draw0", "--width", "--id-sheet", "--evals", "--out"):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    main(a.draw0, a.width, a.id_sheet, a.evals, a.out)
