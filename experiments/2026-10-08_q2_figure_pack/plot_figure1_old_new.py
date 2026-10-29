"""Figure 1, old vs new Qwen (James's request), in the minimal style: honest main-task success of the on-policy
Qwen3-32B (old) and Qwen3.5-27B (new) agents, with Sol's real honest runs as the reference, on the tasks scored for
all three (local). The analysis is plot_figure1.py's (amendment 16), imported, not copied: per-task C/I counts with
unscored (N) runs left out of both counts, the matched tasks, pooled (run-weighted) rates, and one bootstrap over those
tasks (each task's runs together; 10,000 draws, seed 0) shared by every model, so differences are paired. A rate of 0
gives a degenerate task interval ([0, 0]); the exact (Clopper-Pearson) interval over runs is printed beside it, and
the bar shows that one.

Inputs: Sol, honest_success_2026-10-08_173215.json (S3, experiments/2026-10-08_q2_free_checks_3/results/); Qwen3.5-27B,
results/honest_success_qwen3.5-27b_2026-10-09_100002.json; Qwen3-32B, the native-window arm's report_relaunch.json and
report_first.json (S3, phase2/runs/qwen3_32b_native_report/; #281), whose rows name the task folder, mapped to the main
task by the pilot draw (pilot_tasks_amendment9.json). The first run's runs that never scored (stopped and relaunched)
are dropped; each (arm, task) must then appear once.

With a fifth argument, find_placeholder_runs.py's output (Aswin, 2026-10-09): placeholder-graded runs (a "Dummy scorer"
that marks every run I) count as unscored, so a task left with no scored run drops out of the matched set for all three
bars, as t09 (scorer timeout) does. The bars are then task-weighted (each task's share of runs passed, averaged over
tasks), with 95% intervals from one bootstrap over tasks (10,000 draws, seed 0) shared by every model; a bar at 0 shows
the exact interval over tasks (0 of n). The pooled numbers stay in the JSON, and the as-reported figure (pooled, no
placeholder handling) is printed beside the corrected one.

  uv run --no-project --with matplotlib --with numpy --with scipy python experiments/2026-10-08_q2_figure_pack/plot_figure1_old_new.py \
      <Sol json> <Qwen3.5 json> <report_relaunch.json> <report_first.json> [<placeholder_runs json>]
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_figure1 import exact, paired_boot, per_task, task_averaged

HERE = Path(__file__).parent
PILOT = HERE.parents[0] / "2026-10-07_phase2_attack_check" / "config" / "pilot_tasks_amendment9.json"


def qwen3_32b_rows(relaunch: Path, first: Path) -> list[tuple[str, str, str]]:
    main_of = {c["key"]: c["main"] for c in json.loads(PILOT.read_text())["cells"]}
    seen, out = set(), []
    for path in (relaunch, first):
        for r in json.loads(path.read_text())["rows"]:
            if r.get("main_task_raw") is None:  # not started, or stopped before scoring, and relaunched
                continue
            if (r["arm"], r["task"]) in seen:
                raise ValueError(f"{r['arm']} {r['task']} appears twice")
            seen.add((r["arm"], r["task"]))
            out.append((main_of[r["task"]], r["main_task_raw"], f"{r['arm']}/{r['task']}"))
    return out


def tables(sol: dict, q35: dict, rows32: list, drop: dict) -> dict:
    """Per-task (successes, scored runs) per model; runs listed in `drop` (placeholder-graded) count as unscored."""
    mark = lambda m, rows: [(t, "N" if k in drop.get(m, ()) else v) for t, v, k in rows]  # noqa: E731
    return {"Qwen3-32B": per_task(mark("Qwen3-32B", rows32)),
            "Qwen3.5-27B": per_task(mark("Qwen3.5-27B", [(r["sample_id"].split(":")[0], r["main_task_success"],
                                                            f"{r['arm']}/{r['task_dir']}") for r in q35["rows"]])),
            "Sol": per_task(mark("Sol", [(r["main_task"], r["main_task_success"], r["key"]) for r in sol["rows"]]))}


def summarize(tabs: dict) -> dict:
    ci = lambda d: [round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])]  # noqa: E731
    tasks, res = paired_boot(list(tabs.values()))
    res = dict(zip(tabs, res))
    tw = dict(zip(tabs, task_averaged(list(tabs.values()), tasks, draws=10_000, seed=0)))
    out = {"matched tasks": len(tasks), "tasks": tasks}
    for m, (r, d, k, n) in res.items():
        ntask = sum(1 for t in tasks if tabs[m][t][0] > 0)  # tasks with any run passed
        out[m] = {"rate": round(r, 4), "successes": k, "scored runs": n, "task interval": ci(d),
                  "exact interval over runs": [round(x, 4) for x in exact(k, n)],
                  "task-weighted rate": round(float(tw[m][0]), 4), "task-weighted interval": ci(tw[m][1]),
                  "exact interval over tasks (shown when the bar is 0)": [round(x, 4) for x in exact(ntask, len(tasks))]}
    for a, b in (("Qwen3.5-27B", "Qwen3-32B"), ("Qwen3.5-27B", "Sol"), ("Qwen3-32B", "Sol")):
        out[f"{a} minus {b} (points)"] = [round(100 * (res[a][0] - res[b][0]), 1)] + [round(100 * x, 1) for x in ci(res[a][1] - res[b][1])]
        out[f"{a} minus {b} (points, task-weighted)"] = [round(100 * float(tw[a][0] - tw[b][0]), 1)] + [
            round(100 * x, 1) for x in ci(tw[a][1] - tw[b][1])]
    return out


def main(sol_path: Path, q35_path: Path, relaunch: Path, first: Path, placeholder: Path | None = None):
    sol, q35 = json.loads(sol_path.read_text()), json.loads(q35_path.read_text())["Qwen3.5-27B"]
    rows32 = qwen3_32b_rows(relaunch, first)
    plain = tables(sol, q35, rows32, {})
    if tuple(map(sum, zip(*plain["Qwen3.5-27B"].values()))) != (q35["successes"], q35["runs"]):
        raise ValueError("Qwen3.5-27B rows disagree with its counts")
    inputs = (sol_path, q35_path, relaunch, first, *([placeholder] if placeholder else []))
    drop = {}
    if placeholder:  # the list must come from these same four inputs: run keys repeat across reruns
        ph = json.loads(placeholder.read_text())
        want = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (sol_path, q35_path, relaunch, first)}
        if ph["inputs sha256"] != want:
            raise ValueError(f"{placeholder.name} was made from other inputs: {ph['inputs sha256']} vs {want}")
        drop = ph["placeholder runs"]
    out = {"inputs sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
           "Qwen3-32B runs read (all tasks)": {v: [x for _, x, _ in rows32].count(v) for v in ("C", "I", "N")},
           "placeholder runs counted as unscored": {m: len(v) for m, v in drop.items()}} | summarize(tables(sol, q35, rows32, drop))
    print(json.dumps(out, indent=1))
    old = summarize(plain)  # the as-reported figure: pooled, no placeholder handling
    if placeholder:  # printed beside the corrected one (task-weighted)
        print(f"as reported ({old['matched tasks']} tasks, pooled) -> corrected ({out['matched tasks']} tasks, task-weighted); "
              f"left out: {sorted(set(old['tasks']) - set(out['tasks']))}")
        for m in ("Qwen3-32B", "Qwen3.5-27B", "Sol"):
            print(f"  {m:12s} {100 * old[m]['rate']:5.1f}% {[round(100 * x, 1) for x in old[m]['task interval']]} -> "
                  f"{100 * out[m]['task-weighted rate']:5.1f}% {[round(100 * x, 1) for x in out[m]['task-weighted interval']]}")
        for a, b in (("Qwen3.5-27B", "Sol"), ("Qwen3.5-27B", "Qwen3-32B")):
            print(f"  {a} minus {b}: {old[f'{a} minus {b} (points)']} -> {out[f'{a} minus {b} (points, task-weighted)']}")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    names = {"Qwen3-32B": "Old Qwen\n(Qwen3-32B)", "Qwen3.5-27B": "New Qwen\n(Qwen3.5-27B)",
             "Sol": "Sol\n(reference,\ndifferent setup)"}
    for i, (m, c) in enumerate(zip(names, ("#c4661a", "#2f6db5", "#c9ced4"))):
        v = out[m]["task-weighted rate"]
        lo, hi = out[m]["task-weighted interval"] if v > 0 else out[m]["exact interval over tasks (shown when the bar is 0)"]
        ax.bar(i, 100 * v, color=c, width=0.6)
        ax.errorbar(i, 100 * v, yerr=[[100 * (v - lo)], [100 * (hi - v)]], color="black", capsize=6, lw=1.5)
        label = f"{100 * v:.0f}%" if v else f"0%\n(0 of {out['matched tasks']} tasks)"
        ax.text(i + 0.32 if v else i, 100 * v if v else 100 * hi + 14, label, ha="left" if v else "center", va="center",
                fontsize=12, fontweight="bold")
    ax.set_xticks(range(3), list(names.values()), fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Honest runs passed (%)\n(averaged over tasks; higher is better)", fontsize=11)
    old_r, new_r = out["Qwen3-32B"]["task-weighted rate"], out["Qwen3.5-27B"]["task-weighted rate"]
    assert old_r == 0 and 0.6 <= new_r <= 0.72, "the title's takeaway no longer matches the numbers"  # "about two-thirds"
    ax.set_title(f"Old Qwen passes no honest tasks; new Qwen passes about two-thirds\n(same {out['matched tasks']} "
                 "LinuxArena tasks)", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # Pilot tasks out of the matched set: a placeholder grader (dropped here) or no scored run at all (t09: the grader
    # timed out on every run). The README gives which is which; the figure carries one short method line (Aswin's rule).
    pilot = sorted({c["main"] for c in json.loads(PILOT.read_text())["cells"]})
    out["left out"] = {"placeholder-graded": sorted(set(old["tasks"]) - set(out["tasks"])),
                       "no scored run": sorted(set(pilot) - set(old["tasks"]))}
    n_left = sum(map(len, out["left out"].values()))
    left = f"; {n_left} of {len(pilot)} pilot tasks dropped for broken graders" if n_left else ""
    fig.text(0.01, 0.01, f"Task-weighted, 95% intervals over tasks{left}.", fontsize=8, color="#5a6371")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    png = HERE / "results" / f"figure1_old_new_qwen_{stamp}.png"
    fig.savefig(png, dpi=160)
    (HERE / "results" / f"figure1_old_new_qwen_{stamp}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(png)


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:6]))
