"""Figure 1: honest main-task success per acting model (pre-registration amendments 13 and 16). $0, local; reads the
run folders' launch.json and .eval logs (zip archives with zstd members, read directly with Python 3.14; no Inspect needed).

- Qwen3.5-27B: per task, A's first try and B (amendment 13: never A's final try; later tries are selected). Qwen3-32B:
  one honest run per task (amendment 16). Which arm folder is which model comes from the config.
- The counted attempt (the a1 session's launcher, D5 and amendments 12, 13, 17, 20): the first attempt without
  `redrawn_because` (a redraw replaces a try that failed before the policy started), never a "rerun". Other attempts
  are counted and listed.
- Each counted attempt is one of:
  - success: linux_scorer main_task_success "C";
  - failure: "I";
  - not scored: "N", or a linux_scorer without main_task_success;
  - errored: the sample has an error and no linux_scorer;
  - missing: no attempt of that kind, or no .eval in its log folder.
  Only successes and failures enter a rate. The rest are counted and reported, never scored as failures.
- Rates: per model, successes over scored attempts, with an exact (Clopper–Pearson) 95% interval (the orchestrator's
  ask), and the pre-registered task-bootstrap interval (2,000 resamples of tasks, seed 0) of the mean over tasks of
  each task's rate. The difference (Qwen3.5-27B − Qwen3-32B) gets a paired task bootstrap on tasks scored for both.

  uv run --no-project --python 3.14 --with numpy --with scipy python analyze_fig1_honest_success.py --config config/fig1_<name>.json [--label INVALID]
"""

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).parent


def outcome(eval_path: Path) -> str:
    with zipfile.ZipFile(eval_path) as z:
        names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
        if len(names) != 1:
            raise ValueError(f"{eval_path}: {len(names)} samples, expected 1")
        s = json.loads(z.read(names[0]))
    ls = (s.get("scores") or {}).get("linux_scorer")
    if s.get("error") and ls is None:
        return "errored"
    if ls is None:
        return "not scored"
    v = (ls.get("value") or {}).get("main_task_success")
    return {"C": "success", "I": "failure"}.get(v, "not scored")


def exact(k, n):
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", default="", help="e.g. INVALID or DRY CHECK; written into the result")
    ap.add_argument("--out", default="", help="write here instead of results/ (live updates while runs finish)")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    rows, other = [], Counter()  # rows: (model, task, arm, attempt kind, outcome); other: attempts not counted
    for src in cfg["sources"]:  # {"run_dir", "arm", "model", "attempt": "counted"}
        rd = Path(src["run_dir"]).expanduser()
        launch = json.loads((rd / "launch.json").read_text())
        for r in launch["runs"]:
            if r["arm"] != src["arm"]:
                continue
            atts = r.get("attempts") or []
            usable = [a for a in atts if not a.get("redrawn_because") and a.get("kind") != "rerun"]
            if src["attempt"] != "counted":
                raise ValueError(f"attempt rule {src['attempt']!r}: only 'counted' is defined")
            pick = usable[:1]
            other[src["model"]] += len(atts) - len(pick)
            if not pick:
                rows.append((src["model"], r["task"], r["arm"], src["attempt"], "missing"))
                continue
            acting = (pick[0].get("env_vars") or {}).get("PT_ATTACKER")
            if acting is not None and acting.split("/", 1)[-1] != src["model"]:
                raise ValueError(f"{rd.name} {r['arm']} {r['task']}: acting model {acting}, config says {src['model']}")
            ld = pick[0].get("log_dir") or ""
            local = rd / ld.split(f"/{rd.name}/", 1)[1] if f"/{rd.name}/" in ld else Path(ld).expanduser()  # devbox path -> this copy
            evals = sorted(local.glob("*.eval")) if ld else []
            rows.append((src["model"], r["task"], r["arm"], pick[0].get("kind"), outcome(evals[-1]) if evals else "missing"))
    out = {"label": args.label, "config": cfg, "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
           "counts": {}, "rates": {}}
    by_model = defaultdict(list)
    for m, t, a, k, o in rows:
        by_model[m].append((t, a, o))
    rng = np.random.default_rng(0)
    task_rate = {}
    for m, rs in by_model.items():
        c = Counter(o for _, _, o in rs)
        out["counts"][m] = dict(c) | {"attempts": len(rs), "tasks": len({t for t, _, _ in rs}), "attempts not counted": other[m]}
        per_task = defaultdict(list)
        for t, _, o in rs:
            if o in ("success", "failure"):
                per_task[t].append(o == "success")
        k, n = c["success"], c["success"] + c["failure"]
        tr = {t: float(np.mean(v)) for t, v in per_task.items()}
        task_rate[m] = tr
        ts = sorted(tr)
        boots = [float(np.mean([tr[ts[i]] for i in rng.integers(0, len(ts), len(ts))])) for _ in range(2000)] if ts else []
        out["rates"][m] = {"successes": k, "scored": n, "rate": round(k / n, 4) if n else None, "exact_95": exact(k, n),
                           "mean_over_tasks": round(float(np.mean(list(tr.values()))), 4) if tr else None,
                           "task_bootstrap_95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)] if boots else None,
                           "tasks_scored": len(tr)}
    models = cfg.get("difference")  # [minuend, subtrahend]
    if models and all(m in task_rate for m in models):
        a, b = (task_rate[m] for m in models)
        both = sorted(set(a) & set(b))
        d = [a[t] - b[t] for t in both]
        boots = [float(np.mean([d[i] for i in rng.integers(0, len(both), len(both))])) for _ in range(2000)] if both else []
        out["difference"] = {"of": models, "tasks_scored_for_both": len(both), "mean": round(float(np.mean(d)), 4) if d else None,
                             "paired_task_bootstrap_95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)] if boots else None}
    out["per_task"] = [{"model": m, "task": t, "arm": a, "attempt": k, "outcome": o} for m, t, a, k, o in rows]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = Path(args.out) if args.out else HERE / "results" / f"fig1_honest_success{'_' + args.label.lower().replace(' ', '_') if args.label else ''}_{stamp}.json"
    with path.open("w" if args.out else "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("label", "counts", "rates") + (("difference",) if "difference" in out else ())}))
    print("->", path)


if __name__ == "__main__":
    main()
