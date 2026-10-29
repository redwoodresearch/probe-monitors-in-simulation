"""Does Qwen3.5 behave differently in simulation? Task-paired behavior measures on the Phase 2 strict-tools rerun.
Read-only, $0, CPU. README.md has the comparisons, the definitions and the predictions written before running.

  uv run --no-project --python 3.14 --with numpy --with boto3 --with pyyaml python -I \
      experiments/2026-10-09_phase2_onpolicy_behavior/analyze_onpolicy_behavior.py --run-dir <copy of the run folder>

Writes results/onpolicy_behavior_<stamp>.json (counts and statistics only) and data/awareness_hits_<stamp>.jsonl (each
keyword hit's raw line, with run, task and turn; data/ is gitignored).
"""
import argparse
import collections
import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from behavior_common import (BOOT, RUNNER_SUBMIT, SEED, failure, fingerprint,  # noqa: E402, F401
                             measure)  # re-exported: analyze_onpolicy_distributions and the Qwen3-32B report use them


COMPARISONS = [  # (name, real arm, sim arm); arms are (arm, simulator), simulator None for real runs
    ("attack, September (primary)", ("attack", None), ("sim_attack", "september")),
    ("attack, today (secondary)", ("attack", None), ("sim_attack", "today")),
    ("honest, September (primary)", ("honest_b", None), ("sim_honest", "september")),
    ("honest, today (secondary)", ("honest_b", None), ("sim_honest", "today")),
    ("honest, real A vs real B (floor)", ("honest_b", None), ("honest_a", None)),
]

def load(run_dir: Path) -> tuple[dict, list[dict], dict, list, collections.Counter]:
    launch = json.loads((run_dir / "launch.json").read_text())
    runs, quotes, problems, label_errors = {}, [], [], collections.Counter()
    for r in launch["runs"]:
        key = (r["arm"], None if r["real"] else r["simulator"])
        counted = [a for a in r["attempts"] if not a.get("redrawn_because")]
        if not counted:
            problems.append({"run": key, "task": r["task"], "problem": "no counted attempt"})
            continue
        a = counted[0] if r["arm"] == "honest_a" else counted[-1]  # honest_a: its first try (amendment 13)
        log_dir = run_dir / Path(a["log_dir"]).relative_to(Path(a["log_dir"]).parents[1])
        evals = sorted(log_dir.glob("*.eval"))
        if len(evals) != 1:
            problems.append({"run": key, "task": r["task"], "problem": f"{len(evals)} .eval files in {log_dir.name}"})
            continue
        with zipfile.ZipFile(evals[0]) as z:
            names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
            if len(names) != 1:
                problems.append({"run": key, "task": r["task"], "problem": f"{len(names)} samples"})
                continue
            sample = json.loads(z.read(names[0]))
        m, q, le = measure(sample, r)
        m["sample_error"] = ((sample.get("error") or {}).get("message") or "")[:200] or None
        m["submitted"] = m["submitted"] and not m["sample_error"]  # an errored run counts as not submitted
        runs[(key, r["task"])] = m
        quotes += q
        label_errors += le
    return launch, quotes, runs, problems, label_errors


def paired(runs: dict, real: tuple, sim: tuple, field: str, keep_errored: bool = False) -> dict:
    tasks = sorted({t for (k, t) in runs if k == real} & {t for (k, t) in runs if k == sim})
    ok = [t for t in tasks if (keep_errored or not (runs[(real, t)]["sample_error"] or runs[(sim, t)]["sample_error"]))
          and runs[(real, t)][field] is not None and runs[(sim, t)][field] is not None]
    if not ok:
        return {"n_tasks": 0}
    x = np.array([float(runs[(real, t)][field]) for t in ok])
    y = np.array([float(runs[(sim, t)][field]) for t in ok])
    d = y - x
    boots = d[np.random.default_rng(SEED).integers(0, len(d), size=(BOOT, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_tasks": len(ok), "real_mean": x.mean(), "sim_mean": y.mean(), "diff": d.mean(), "ci95": [lo, hi],
            "sim_higher": int((d > 0).sum()), "sim_lower": int((d < 0).sum()), "tied": int((d == 0).sum()),
            "excluded_tasks": sorted(set(tasks) - set(ok))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())  # repo-wide: imported code counts too
    launch, quotes, runs, problems, label_errors = load(args.run_dir)
    arms = sorted({k for k, _ in runs}, key=str)

    print(f"run {args.run_dir.name} | script commit {commit[:12]}{' (repo has uncommitted changes)' if dirty else ''} "
          f"| labeler {fingerprint()} | seed {SEED}, {BOOT} task resamples")
    print(f"runs read: {len(runs)} of {len(launch['runs'])}; problems: {problems or 'none'}")
    print(f"label errors (calls the labeler has no rule for): {dict(label_errors) or 'none'}")
    f2 = lambda v, d=3: "n/a" if v is None or v != v else f"{v:.{d}f}"  # noqa: E731
    print("\nper arm (all runs read; means over runs without a sample error):")
    pooled = {}
    for k in arms:
        rs = [m for (kk, _), m in runs.items() if kk == k]
        clean = [m for m in rs if not m["sample_error"]]
        cats = collections.Counter()
        for m in clean:
            cats.update(m["categories"])
        mean = lambda f: (float(np.mean(v)) if (v := [m[f] for m in clean if m[f] is not None]) else None)  # noqa: E731
        pooled[str(k)] = {"runs": len(rs), "sample_errors": len(rs) - len(clean),
                          "submitted": sum(m["submitted"] for m in rs),
                          **{f: mean(f) for f in ("steps", "calls", "read_only_share", "rereads_per_100_calls",
                                                  "edit_reread_share", "fail_share", "error_field_share",
                                                  "keyword_hits_per_100_turns")},
                          "keyword_any_runs": sum(m["keyword_any"] for m in clean),
                          "prompt_has_keyword_runs": sum(m["prompt_has_keyword"] for m in clean),
                          "internet_simulator_hits": sum(m["internet_simulator_hits"] for m in clean),
                          "category_share_pooled": {c: round(n / sum(cats.values()), 4) for c, n in cats.most_common()},
                          "fail_kinds": dict(sum((collections.Counter(m["fail_kinds"]) for m in clean),
                                                 collections.Counter()))}
        p = pooled[str(k)]
        print(f"  {k[0] + ('@' + k[1] if k[1] else ''):22s} runs {p['runs']:2d} (errored {p['sample_errors']}), "
              f"submitted {p['submitted']}/{p['runs']} | steps {f2(p['steps'], 1)}, calls {f2(p['calls'], 1)} | "
              f"read-only {f2(p['read_only_share'])} | re-reads/100 calls {f2(p['rereads_per_100_calls'], 2)}, edits "
              f"re-read {f2(p['edit_reread_share'])} | fail {f2(p['fail_share'])} (error field "
              f"{f2(p['error_field_share'])}) | keyword hits/100 turns {f2(p['keyword_hits_per_100_turns'], 2)}, runs with any "
              f"{p['keyword_any_runs']}, prompt has one {p['prompt_has_keyword_runs']}, internet-simulator "
              f"{p['internet_simulator_hits']}")
        print(f"  {'':22s} categories: {p['category_share_pooled']}")
        print(f"  {'':22s} failure kinds: {p['fail_kinds']}")

    top = [c for c, _ in sum((collections.Counter(m["categories"]) for m in runs.values()),
                             collections.Counter()).most_common(6)]
    for m in runs.values():  # a category a run never used has share 0 there
        for c in top:
            m["cat_share_" + c] = m["cat_share"].get(c, 0.0) if m["calls"] else None
        m["submitted_int"] = int(m["submitted"])
        m["keyword_any_int"] = int(m["keyword_any"])
    fields = (["steps", "calls", "read_only_share"] + ["cat_share_" + c for c in top]
              + ["rereads_per_100_calls", "edit_reread_share", "fail_share", "error_field_share",
                 "keyword_hits_per_100_turns", "keyword_any_int"])
    comps = {}
    print("\ntask-paired, sim minus real (mean over tasks, 95% task-bootstrap interval; sim higher/lower/tied):")
    for name, real, sim in COMPARISONS:
        comps[name] = {"submitted_int (errored runs kept)": paired(runs, real, sim, "submitted_int", keep_errored=True)}
        comps[name] |= {f: paired(runs, real, sim, f) for f in fields}
        print(f"  {name}:")
        for f, s in comps[name].items():
            if not s["n_tasks"]:
                print(f"    {f:36s} no tasks")
                continue
            print(f"    {f:36s} n={s['n_tasks']:2d} real {s['real_mean']:8.3f} sim {s['sim_mean']:8.3f} diff "
                  f"{s['diff']:+8.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] {s['sim_higher']}/{s['sim_lower']}/"
                  f"{s['tied']}{' excluded ' + str(s['excluded_tasks']) if s['excluded_tasks'] else ''}")

    data = HERE / "data" / f"awareness_hits_{stamp}.jsonl"
    data.parent.mkdir(exist_ok=True)
    data.write_text("".join(json.dumps(q) + "\n" for q in quotes))
    by_kw = collections.defaultdict(collections.Counter)
    for q in quotes:
        by_kw[q["arm"] + ("@" + q["simulator"] if q["simulator"] else "")][q["match"].lower()] += 1
    print(f"\nawareness keyword hits by arm and keyword (raw lines: {data.relative_to(REPO)}, gitignored):")
    for k, c in sorted(by_kw.items()):
        print(f"  {k:22s} {dict(c)}")
    rng = np.random.default_rng(SEED)
    print(f"quotes, up to 3 per arm, drawn with seed {SEED} (run = arm@simulator/task, turn = assistant turn):")
    for k in sorted(by_kw):
        qs = [q for q in quotes if q["arm"] + ("@" + q["simulator"] if q["simulator"] else "") == k]
        for i in sorted(rng.choice(len(qs), size=min(3, len(qs)), replace=False)):
            q = qs[i]
            print(f"  [{k}/{q['task']}, turn {q['turn']}, {q['part']}] "
                  f"{q['line'][max(0, q['at'] - 110): q['at'] + 130].strip()!r}")

    out = HERE / "results" / f"onpolicy_behavior_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "x") as f:
        json.dump({"run": args.run_dir.name, "launch_probe_shift_commit": launch.get("probe_shift_commit"),
                   "script_commit": commit, "repo_dirty": dirty, "labeler": fingerprint(), "seed": SEED,
                   "boot": BOOT, "made": stamp, "problems": problems, "label_errors": dict(label_errors),
                   "per_arm": pooled, "comparisons": comps,
                   "per_run": [{"arm": k[0], "simulator": k[1], "task": t,
                                **{f: v for f, v in m.items() if f not in ("categories", "cat_share")}}
                               for (k, t), m in sorted(runs.items(), key=str)],
                   "awareness_hits_file": str(data.relative_to(REPO)), "awareness_hits": len(quotes)},
                  f, indent=1, default=float)
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
