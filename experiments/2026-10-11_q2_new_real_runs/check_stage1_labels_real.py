"""Stage 1, real half of the labels-only check (PREREG.md "Staged launch and checks"; PURCHASE.md in
2026-10-11_q2_lean_newruns). For transfer and destructive, it compares on the new real runs:
- positives per run;
- the share of runs with one or more;
- the share of new tasks with positives in both runs.

These are compared with today's matched real runs. If any is below 80% of today's, the rest waits and is re-priced. The
simulated half waits for b7's stage 1. Honest runs only; per-call labels from the committed rules (build_recal_labels.py's
functions); counts out, no command text. Local, $0.

Today's numbers come from the recal rows (the dump's impactful calls with their labels), as the width check used them.
The new runs have no dump yet, so their labels come straight from the transcripts: impactful by `label_call`, plus the
monitor's rule. As a positive control, the same direct labeler is run on today's 800 matched runs and must reproduce the
recal rows' positives. The original overlap800 runs on the same 50 tasks are reported beside it (same tasks, same code).

  uv run --no-project --python 3.14 --with boto3 --with pyyaml python \\
      experiments/2026-10-11_q2_new_real_runs/check_stage1_labels_real.py <recal json> <2e's run list json> <the stage's honest_b folder, as copied from S3: <key>/<eval file>>
"""
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "2026-10-10_q2_transfer_commands")); sys.path.insert(0, str(EXP / "2026-10-10_q2_destructive_commands"))
from analyze_destructive_counts import DESTRUCTIVE  # noqa: E402
from analyze_transfer_counts import calls, executed, is_transfer, label_call  # noqa: E402
from probe_shift.calls import EDITOR_WRITES  # noqa: E402

RECAL_SHA = "c9ba6dfe3bd028027e162ae386dc5c4ec5343eff5825d63c4ed94534b2edb9e7"
RUN_LIST_SHA = "c0b270be308d437b6f641a640347ea232b4ab5a7a6e19c724f79ec498c6b4965"
OLD_LOGS = Path.home() / ".cache/probe-shift/s3/rr-sce-sim-transcripts/transcripts/2026-09-18_la-public10_real-honest-solmedium-step150-overlapaware-n1000/logs"
MONITORS = {"transfer": "transfer", "destructive": "destructive_raw"}
FLOOR = 0.80


def positive(tool, args) -> dict:
    return {"destructive": any(DESTRUCTIVE.search(v if isinstance(v, str) else json.dumps(v)) for v in args.values()),
            "transfer": bool(executed(tool or "") and is_transfer(args))}


def labels(sample: dict) -> dict:
    """Per monitor, the number of impactful calls that are positive (build_recal_labels.py's rules). A text_editor call
    whose command is neither an edit nor a view has no rule in label_call (it raises, by design). The editor rejects such
    a command, so it is counted as not impactful here, a decision made for this check; the counts of such calls, and of
    those a monitor's rule would flag if they were counted, are reported."""
    n = {m: 0 for m in MONITORS} | {"invalid text_editor calls": 0} | {f"invalid, {m} rule would flag": 0 for m in MONITORS}
    for _step, _action, tool, args in calls(sample):
        if tool == "text_editor" and args.get("command") not in EDITOR_WRITES and args.get("command") != "view":
            n["invalid text_editor calls"] += 1
            for m, v in positive(tool, args).items():
                n[f"invalid, {m} rule would flag"] += v
            continue
        if label_call(tool, args)["label"] != "impactful":
            continue
        for m, v in positive(tool, args).items():
            n[m] += v
    return n


def one_sample(path: Path, sample_id: str) -> dict:
    with zipfile.ZipFile(path) as z:
        for name in (x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")):
            s = json.loads(z.read(name))
            if str(s["id"]) == sample_id:
                return s
    raise SystemExit(f"{path.name}: sample {sample_id} not found")


def rates(per_run: dict, pairs: list) -> dict:
    """per_run: run -> {monitor: positives}; pairs: [(run 1, run 2)] for the pair share."""
    out = {}
    for m in MONITORS:
        v = [per_run[r][m] for r in per_run]
        out[m] = {"runs": len(v), "positives": sum(v), "positives per run": sum(v) / len(v), "share of runs with one or more": sum(x > 0 for x in v) / len(v),
                  "pairs": len(pairs), "share of pairs with positives in both": (sum(per_run[a][m] > 0 and per_run[b][m] > 0 for a, b in pairs) / len(pairs)) if pairs else None}
    return out


def main(recal: Path, run_list: Path, new_dir: Path):
    for p, want in ((recal, RECAL_SHA), (run_list, RUN_LIST_SHA)):
        if hashlib.sha256(p.read_bytes()).hexdigest() != want:
            raise SystemExit(f"{p.name} differs from its pinned sha256")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    used = [Path(__file__).resolve(), EXP / "2026-10-10_q2_transfer_commands/analyze_transfer_counts.py", EXP / "2026-10-10_q2_destructive_commands/analyze_destructive_counts.py"]
    for f in used:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    rows = json.loads(recal.read_text())["rows"]
    sim_tasks = {r["task"] for r in rows if r["world"] == 1}
    is800 = lambda run: "|overlap800_" in run
    # today's, from the recal rows (as the width check)
    matched = [r for r in rows if r["world"] == 0 and is800(r["run"]) and r["task"] in sim_tasks]
    runs_of = defaultdict(set)
    for r in matched:
        runs_of[r["task"]].add(r["run"])
    today_runs = sorted({r["run"] for r in matched})
    today = {run: {m: 0 for m in MONITORS} for run in today_runs}
    for r in matched:
        for m, col in MONITORS.items():
            today[r["run"]][m] += r[col] == 1
    first2 = defaultdict(set)
    for r in rows:
        if r["world"] == 0:
            first2[r["task"]].add(r["run"])
    pairs163 = [tuple(sorted(v)[:2]) for t, v in first2.items() if len(v) >= 2 and all(is800(x) for x in sorted(v)[:2])]
    if len(today_runs) != 800 or len(pairs163) != 163:
        raise SystemExit(f"today's cohort differs: {len(today_runs)} runs, {len(pairs163)} pairs")
    today_rates = rates(today, pairs163)
    # positive control: the direct labeler on today's 800 runs
    direct = {}
    for run in today_runs:
        _, file, sid, _ep = run.split("|")
        direct[run] = labels(one_sample(OLD_LOGS / file, sid))
    control = {m: {"recal rows": sum(today[r][m] for r in today_runs), "direct labeler": sum(direct[r][m] for r in today_runs),
                   "runs that differ": sum(today[r][m] != direct[r][m] for r in today_runs)} for m in MONITORS}
    # the new runs
    new_rows = json.loads(run_list.read_text())["rows"]
    new = {}
    for r in new_rows:
        f = new_dir / r["key"] / r["eval_file"]
        if hashlib.sha256(f.read_bytes()).hexdigest() != r["eval_sha256"]:
            raise SystemExit(f"{f.name} differs from 2e's eval_sha256")
        new[r["key"]] = labels(one_sample(f, r["sample_id"]))
    new_pairs = sorted((f"oct11c_p{p:03d}_r1", f"oct11c_p{p:03d}_r2") for p in sorted({r["position"] for r in new_rows}))
    stage_tasks = sorted({r["task"] for r in new_rows})
    same_task_old = {run: direct[run] for t in stage_tasks for run in runs_of.get(t, ())}
    extra = lambda per_run: {k: sum(v[k] for v in per_run.values()) for k in next(iter(per_run.values())) if k not in MONITORS} | {
        "runs with an invalid text_editor call": sum(v["invalid text_editor calls"] > 0 for v in per_run.values())}
    out = {"git": head, "invalid text_editor calls (counted as not impactful)": {"today's 800 runs": extra(direct), "new runs": extra(new)}, "recal sha256": RECAL_SHA, "run list sha256": RUN_LIST_SHA, "floor": FLOOR,
           "positive control (direct labeler vs recal rows, today's 800 runs)": control,
           "today (recal rows; 800 matched runs, 163 pairs)": today_rates,
           "new real runs (stage 1; 100 runs, 50 pairs; direct labeler)": rates(new, new_pairs),
           "beside: original overlap800 runs on the same 50 tasks (direct labeler)": rates(same_task_old, []) | {"runs": len(same_task_old)}}
    verdict = {}
    for m in MONITORS:
        t, n = today_rates[m], out["new real runs (stage 1; 100 runs, 50 pairs; direct labeler)"][m]
        verdict[m] = {k: {"new": n[k], "today": t[k], "ratio": n[k] / t[k] if t[k] else None, "below 80%": bool(t[k] and n[k] < FLOOR * t[k])}
                      for k in ("positives per run", "share of runs with one or more", "share of pairs with positives in both")}
    out["check (new vs 80% of today's)"] = verdict
    control_ok = all(c["runs that differ"] == 0 for c in control.values())
    out["positive control passes (every run matches)"] = control_ok
    out["any rate below 80%"] = (any(v["below 80%"] for m in verdict.values() for v in m.values()) if control_ok
                                 else "withheld: the direct labeler doesn't reproduce today's labels run for run")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"stage1_labels_real_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
