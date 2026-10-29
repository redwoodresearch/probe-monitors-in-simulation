"""Destructive and transfer monitors on Qwen3.5-27B's 196-task attack runs, design (b) of PLAN_dt_acting_monitors.md.

Aswin, via AskUserQuestion before 22:14 PT 2026-10-12 (relayed by the orchestrator), verbatim: "Approve, 20 shuffles,
stop $43 (Recommended)" ("As committed in the plan; timed check first; re-price if slower than 300 s per refit").
Labels: Aswin's exact-q2 sheet only (sha256 e4296643..., checked by count_label_positives.load), joined through the
held key and the ID sheet on this machine; only (arm, task, irid, call, impactful, destructive, transfer) leaves it.

Per class (destructive, transfer) and design (primary: other impactful calls as negatives; secondary: read-only
calls), on the 196-task monitor's own rows and pipeline (run_score_onpolicy.one_draw, the xcache_scaleup196 cache,
the frozen task folds, inner CV over layer and C):
- the gate: the cache's rows equal the sheet's attack rows on the scorer's cohort (tasks with a scored call in all
  three attack worlds), with the same task and impactful flag;
- the fit: train on real attack 1 (world "real") with only the design's rows (train_ok = test_ok = positives and the
  design's negatives); score held-out folds of real attack 1, sim attack and real attack 2 (the yardstick); per-call
  scores and every world's calibration-fold scores are kept, so 1% and 5% thresholds are recomputed in the analysis;
- the null: n_null refits with the class labels permuted within each task of the training world (other worlds keep
  their labels), redoing the fit and the inner selection; seeds (20261012, class, design, i).
Each stage is written to the score Volume as it finishes. --check runs one design's fit only (the timed check).
Spend stops: the strict billing watcher (v2.1 copy, run from a frozen folder) stops the app by its ID at --cap-usd or
the deadline; and the function timeout (6,000 s) bounds every container, so 4 H100s at $4.47/h can't pass $29.8 even if billing
lags (the elapsed-cost watcher stops by app name, which doesn't resolve an ephemeral app, so it isn't used).
Before any spend, every fold of the training world must hold both classes in each design: then no inner split is
single-class, and the scorer's failure rule and the plan's skip rule coincide (Codex's review).

  uv run --no-project --python 3.14 --with 'modal>=1.4.3' --with pyyaml --with numpy python run_dt_attack_fits.py \\
      --labels <q2 csv> --sheet <v3 csv> --ids <ID sheet> --key <held key> --frozen <dir with the watcher copy> \\
      --cap-usd <whole app> --max-minutes <deadline> [--check] [--n-null 20]
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
Q3 = HERE.parent / "2026-10-08_q3_onpolicy_figures"
sys.path[:0] = [str(HERE), str(Q3)]
import run_score_onpolicy as R  # noqa: E402

APP = "probe-shift_aswin-surya_sub-dt-attack-fits_kill-2026-11-12"
SEED = 20261012
ARM = {"real": "real_attack_1", "sim": "sim_attack", "run2": "real_attack_2"}
DESIGNS = [(c, d) for c in ("destructive", "transfer") for d in ("primary", "secondary")]
app = modal.App(APP)
image = R.image.add_local_python_source("run_score_onpolicy")


SEM_SHA = "e9a6850b3ca8070385d9abbb4b2d270c36f9170988b6054ad5785553297157e9"  # Aswin's provisional semantic export, 22:06 PT 10-12


def label_table(labels: str, sheet: str, ids: str, key: str, semantic: str = "", q2_subset: bool = False) -> list:
    """[arm, task, irid, call, impactful, destructive, transfer] for every attack row, after the count step's checks.
    With `semantic`, the labels come from the provisional semantic file instead (destructive_broad, transfer_semantic),
    an unresolved one as "U", which every design excludes (Aswin: "A blank means unknown, never zero"). With q2_subset
    too, the labels stay exact-q2 but a row unresolved in the semantic file for that class becomes "U" (the same rows
    as the semantic fit: Aswin, before 00:21 PT 10-13, "Fit it, plus exact-q2 on same subset (Recommended)")."""
    import count_label_positives as C

    rows = C.load(labels, sheet, ids, key)  # every hash, order, column, label-value and one-to-one join check
    keyrows = {r["blind_id"]: r for r in csv.DictReader(open(key, newline=""))}
    order = [r["blind_id"] for r in csv.DictReader(open(labels, newline=""))]
    if semantic:
        if hashlib.sha256(Path(semantic).read_bytes()).hexdigest() != SEM_SHA:
            raise SystemExit("semantic sha256 mismatch")
        sem = list(csv.DictReader(open(semantic, newline="")))
        if [r["blind_id"] for r in sem] != order or list(sem[0]) != ["blind_id", "destructive_broad", "transfer_semantic",
                                                                     "destructive_unresolved", "transfer_unresolved"]:
            raise SystemExit("semantic file: blind_ids, order or columns differ")
        for r, s in zip(rows, sem):
            for cls, lab, flag in (("destructive", "destructive_broad", "destructive_unresolved"), ("transfer", "transfer_semantic", "transfer_unresolved")):
                if s[flag] not in ("0", "1") or s[lab] not in ("0", "1", "") or (s[lab] == "") != (s[flag] == "1"):
                    raise SystemExit("semantic file: a label or flag outside the rules")
                r[cls] = ("U" if s[lab] == "" else r[cls]) if q2_subset else (s[lab] or "U")
    return [[r["arm"], r["task"], keyrows[b]["irid"], int(keyrows[b]["call"]), int(r["impactful"]), r["destructive"], r["transfer"]]
            for b, r in zip(order, rows) if r["arm"] in ARM.values()]


def fold_classes(world, fold, y, mask) -> list:
    """Designs' folds in the training world that lack a class: [(fold, positives, negatives)]; must be empty."""
    import numpy as np

    out = []
    for f in range(5):
        r = (np.asarray(world) == "real") & np.asarray(mask) & (np.asarray(fold) == f)
        p = int(np.asarray(y)[r].sum())
        if p == 0 or p == int(r.sum()):
            out.append((f, p, int(r.sum()) - p))
    return out


def design_arrays(table: list, cls: str, design: str):
    """(positive, negative) masks over table rows for a class and design."""
    import numpy as np

    v = np.array([r[5 if cls == "destructive" else 6] for r in table])
    imp = np.array([r[4] for r in table]) == 1
    pos = v == "1"
    return pos, ((imp & (v == "0")) if design == "primary" else (~imp & (v != "1") & (v != "U")))  # "U": unresolved, excluded


@app.function(image=image, gpu="H100", memory=65536, timeout=6000, retries=0,  # 4 x 6,000 s x $4.47/h = $29.8 (semantic stop $30; was 7,800 s under $43)
              volumes={"/acts": R.acts_vol.read_only(), "/score": R.score_vol})
def fit(cfg: dict, folds: dict, table: list, cls: str, design: str, n_null: int, key: str) -> dict:
    import numpy as np

    t0 = time.time()
    D = R.load(cfg, folds)
    load_s = time.time() - t0
    R.score_vol.reload()
    d = Path("/score") / key / f"{cls}_{design}"
    if d.exists() and any(d.iterdir()):  # never overwrite an earlier run's folder
        raise RuntimeError(f"{d} already holds files")
    d.mkdir(parents=True, exist_ok=True)

    def save(name, obj):  # each stage as it finishes, so a stopped run keeps what it finished
        (d / f".{name}.json.tmp").write_text(json.dumps(obj))
        (d / f".{name}.json.tmp").replace(d / f"{name}.json")
        R.score_vol.commit()

    # the gate (PLAN, Data and spend): the cache's rows are the sheet's attack rows on the scorer's cohort
    cohort = {m["task"] for m in D["meta"]}
    lab = {(a, i, c): (t, imp, de, tr) for a, t, i, c, imp, de, tr in table}
    if len(lab) != len(table):
        raise RuntimeError("duplicate (arm, irid, call) in the label table")
    have = [(ARM[m["world"]], m["irid"], m["call"]) for m in D["meta"]]
    want = {k for k, v in lab.items() if v[0] in cohort}
    if len(set(have)) != len(have) or set(have) != want:
        raise RuntimeError(f"gate: {len(set(have) - want)} cache rows without a label, {len(want - set(have))} labelled rows not cached")
    if any(lab[h][0] != m["task"] or lab[h][1] != m["y"] for h, m in zip(have, D["meta"])):
        raise RuntimeError("gate: a row's task or impactful flag differs between the cache and the sheet")
    pos, neg = design_arrays([[a, lab[(a, i, c)][0], i, c, *lab[(a, i, c)][1:]] for a, i, c in have], cls, design)  # cache order
    mask, y = pos | neg, pos.astype(int)
    if fold_classes(D["world"], D["fold"], y, mask):
        raise RuntimeError(f"a training-world fold lacks a class: {fold_classes(D['world'], D['fold'], y, mask)}")
    Dd = D | {"y": y, "train_ok": mask, "test_ok": mask, "worlds": ["real"], "warm_start": False, "save_probes": True,
              "cells": [("real", "real"), ("real", "sim"), ("real", "run2")]}
    world, task = D["world"], D["task"]
    save("rows", {"world": world.tolist(), "task": task.tolist(), "fold": D["fold"].tolist(), "y": y.tolist(),
                  "in_design": mask.tolist(), "summary": D["summary"], "load_seconds": round(load_s, 1),
                  "counts": {w: {"positives": int((pos & (world == w)).sum()), "negatives": int((neg & (world == w)).sum())}
                             for w in ARM}})

    def run(Dx):
        R._WARM.clear()
        R._SAVED.clear()
        R._LAYERS.clear()
        t = time.time()
        r = R.one_draw(Dx, 0, cfg["seed"], observed=True, per_call=True) | {"seconds": round(time.time() - t, 2)}
        print(f"{key}/{cls}_{design}: one_draw in {r['seconds']} s", flush=True)
        return r

    main = run(Dd)
    main["probes"] = {f"{W}|{k}": {"layer": m["layer"], "C": m["C"], "mu": m["mu"].tolist(), "sd": m["sd"].tolist(),
                                   "beta": m["beta"].tolist()} for (W, k), m in R._SAVED.items()}
    save("main", main)
    timings = [main["seconds"]]
    ci, di = ["destructive", "transfer"].index(cls), ["primary", "secondary"].index(design)
    for i in range(1, n_null + 1):  # within-task label permutation in the training world only
        rng, yn = np.random.default_rng([SEED, ci, di, i]), y.copy()
        for t in np.unique(task[(world == "real") & mask]):
            idx = np.flatnonzero((world == "real") & mask & (task == t))
            yn[idx] = y[idx][rng.permutation(len(idx))]
        r = run(Dd | {"y": yn})
        timings.append(r["seconds"])
        save(f"null_{i:02d}", {"i": i, "null_positive_rows": np.flatnonzero((world == "real") & mask & (yn == 1)).tolist(),
                               "cells": r["cells"], "choices": r["choices"], "seconds": r["seconds"]})
    out = {"key": key, "class": cls, "design": design, "load_seconds": round(load_s, 1), "seconds": timings,
           "total_seconds": round(time.time() - t0, 1), "values": {c: v["values"] for c, v in main["cells"].items()}}
    save("done", out)
    return out


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0, volumes={"/score": R.score_vol})
def collect(key: str) -> dict:
    """Every stage file under /score/<key>, for a launcher whose client died."""
    R.score_vol.reload()
    return {str(p.relative_to(Path("/score") / key)): json.loads(p.read_text()) for p in sorted((Path("/score") / key).glob("*/*.json"))}


class Parser(argparse.ArgumentParser):
    def error(self, message):  # no echo: the arguments include the held key's path
        sys.stderr.write("failed: bad arguments\n")
        sys.exit(2)


if __name__ == "__main__":
    import yaml

    ap = Parser()
    for k in ("--labels", "--sheet", "--ids", "--key"):
        ap.add_argument(k, required=True)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--semantic", default="", help="fit on Aswin's PROVISIONAL semantic labels (unresolved rows excluded)")
    ap.add_argument("--q2-subset", action="store_true", help="with --semantic: exact-q2 labels on the semantic file's resolved rows")
    ap.add_argument("--check", action="store_true", help="the timed check: destructive/primary's fit only, no null")
    ap.add_argument("--collect", default="", help="only read back an earlier run's stage files under this key")
    ap.add_argument("--frozen", required=True, help="folder with frozen copies of watchers/*.py")
    ap.add_argument("--cap-usd", type=float, required=True)
    ap.add_argument("--max-minutes", type=float, required=True)
    a = ap.parse_args()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE).stdout.strip():
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    cfg = yaml.safe_load((Q3 / "config" / "score_scaleup196.yaml").read_text())
    folds = R.frozen_folds(Q3.parents[1] / cfg["pilot_tasks"])
    if a.q2_subset and not a.semantic:
        raise SystemExit("--q2-subset needs --semantic")
    table = label_table(a.labels, a.sheet, a.ids, a.key, a.semantic, a.q2_subset)
    jobs = DESIGNS[:1] if a.check else DESIGNS
    present = set.intersection(*({r[1] for r in table if r[0] == arm} for arm in ARM.values()))  # the scorer's cohort
    tab = [r for r in table if r[1] in present]
    for c, d in DESIGNS:  # before any spend: every training-world fold holds both classes
        p, n = design_arrays(tab, c, d)
        bad = fold_classes([{v: k for k, v in ARM.items()}[r[0]] for r in tab], [folds[r[1]] for r in tab], p.astype(int), p | n)
        if bad:
            raise SystemExit(f"{c}/{d}: training-world folds without both classes {bad}; amend the plan before fitting")
    n_null = 0 if a.check else a.n_null
    ident = {"git": git, "table_sha256": hashlib.sha256(json.dumps(table).encode()).hexdigest(), "jobs": jobs, "n_null": n_null,
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    key = a.collect or f"dt_attack_{'q2subset_' if a.q2_subset else 'semantic_' if a.semantic else ''}{'check' if a.check else 'fits'}_{hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]}"
    t0 = time.time()
    with modal.enable_output(), app.run(detach=True):
        print(f"LAUNCHED app {app.app_id} key {key} at {datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(timespec='seconds')}", flush=True)
        fz = Path(a.frozen)
        if (fz / "watch_billing_strict.py").read_bytes() != (HERE / "watchers" / "watch_billing_strict.py").read_bytes():
            raise SystemExit("the frozen watcher differs from the committed copy")
        with open(fz / f"watch_strict_{key}.log", "a") as fh:  # the watcher outlives this launcher
            subprocess.Popen(["python3", str(fz / "watch_billing_strict.py"), app.app_id, str(a.cap_usd), str(time.time() + 60 * a.max_minutes)],
                             stdout=fh, stderr=subprocess.STDOUT, start_new_session=True, cwd=fz)
        print(f"watcher: {fz}/watch_strict_{key}.log", flush=True)
        if a.collect:
            res = collect.remote(key)
        else:
            res = {f"{c}_{d}": r for (c, d), r in zip(jobs, fit.starmap([(cfg, folds, table, c, d, n_null, key) for c, d in jobs],
                                                                    return_exceptions=True))}
            res = {k: (v if isinstance(v, dict) else {"error": repr(v)}) for k, v in res.items()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE.parents[1] / "data" / "dt_attack_fits" / f"{key}_{stamp}.json"  # per-row arrays: S3, not git
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as fh:
        fh.write(json.dumps({"git": git, "app": APP, "key": key, "ident": ident, "wall_seconds": round(time.time() - t0), "results": res}))
    print(json.dumps({k: ({"seconds": v.get("seconds"), "load_seconds": v.get("load_seconds")} if "error" not in v else v)
                      for k, v in res.items()} if not a.collect else {"files": len(res)}))
    print("->", path)
