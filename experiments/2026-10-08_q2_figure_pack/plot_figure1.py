"""Figure 1: honest main-task success of the on-policy Qwen agents against Sol's (gpt-5.6-sol) real honest runs, on
the pilot tasks scored for every model shown, with 95% intervals and paired Qwen - Sol differences from one bootstrap
over those tasks (local).

Inputs: Sol's rate from analyze_honest_success.py (branch aswin/q2-free-checks-3; result on S3 at
experiments/2026-10-08_q2_free_checks_3/results/honest_success_2026-10-08_173215.json, sha256 f274f2fc…b1585), and,
a JSON per analyze_honest_rerun.py: {"<model>": {"successes": k, "runs": n, "rows": [{"sample_id": "<task>:<id>",
"main_task_success": "C|I|N"}, ...], "note": ..., "source": ...}}. Models missing from it are drawn as placeholders.

  python3 experiments/2026-10-08_q2_figure_pack/plot_figure1.py <honest_success json> [<qwen json>]
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

HERE = Path(__file__).parent
MODELS = ("Qwen3.5-27B",)  # Aswin dropped the Qwen3-32B agent arm (2026-10-09; the footer says why)
TOKEN_RATIO = 1.054  # Qwen3.5 tokens per Qwen3 token, mean over 60 staged transcripts (experiments/2026-10-06_qwen35_reader/README.md)


def exact(k: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson 95% interval."""
    return (beta.ppf(0.025, k, n - k + 1) if k else 0.0), (beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


def per_task(outcomes: list[tuple[str, str]]) -> dict[str, tuple[int, int]]:
    """(successes, scored runs) per main task from (task, C/I/N) per run; unscored (N) runs count in neither."""
    out = {}
    for t, v in outcomes:
        c, n = out.get(t, (0, 0))
        out[t] = (c + (v == "C"), n + (v in ("C", "I")))
    return out


def paired_boot(tables: list[dict], draws: int = 10_000, seed: int = 0) -> tuple[list[str], list[tuple]]:
    """The matched tasks (a scored run in every table) and, per table, the pooled (run-weighted) success rate there,
    its bootstrap draws, successes and scored runs. One bootstrap over the matched tasks (each task's runs together)
    serves every table, so differences between tables are paired. James asked for intervals over tasks."""
    tasks = sorted(set.intersection(*({t for t, (_, n) in tab.items() if n > 0} for tab in tables)))
    if not tasks:
        raise ValueError("no task is scored in every table")
    pick = np.random.default_rng(seed).integers(0, len(tasks), (draws, len(tasks)))
    out = []
    for tab in tables:
        c, n = (np.array([tab[t][i] for t in tasks]) for i in (0, 1))
        out.append((c.sum() / n.sum(), c[pick].sum(1) / n[pick].sum(1), int(c.sum()), int(n.sum())))
    return tasks, out


def runs_per_task(tab: dict, tasks: list[str]) -> str:
    """'1 run on 7 tasks, 2 on 8, ...': how many scored runs each matched task has."""
    counts = {}
    for t in tasks:
        counts[tab[t][1]] = counts.get(tab[t][1], 0) + 1
    parts = [f"{n} on {k}" for n, k in sorted(counts.items())]
    return f"{parts[0].replace(' on ', ' run on ', 1)} tasks" + "".join(f", {x}" for x in parts[1:])


def task_averaged(tables: list[dict], tasks: list[str], draws: int = 2_000, seed: int = 20261009) -> list[tuple]:
    """Sensitivity (coordinator, 2026-10-09): each table's success share averaged within each matched task first, then
    over tasks, with its bootstrap draws (one task resample shared by every table, so differences are paired)."""
    pick = np.random.default_rng(seed).integers(0, len(tasks), (draws, len(tasks)))
    shares = [np.array([tab[t][0] / tab[t][1] for t in tasks]) for tab in tables]
    return [(sh.mean(), sh[pick].mean(1)) for sh in shares]


def main(sol_path: Path, qwen_path: Path | None):
    sol = json.loads(sol_path.read_text())
    elo, ehi = exact(sol["successes"], sol["runs"])
    if abs(elo - sol["exact 95% interval"][0]) > 1e-3 or abs(ehi - sol["exact 95% interval"][1]) > 1e-3:
        raise ValueError(f"Sol's interval {sol['exact 95% interval']} differs from Clopper-Pearson ({elo:.4f}, {ehi:.4f})")
    qwen = json.loads(qwen_path.read_text()) if qwen_path else {}
    present = [m for m in MODELS if m in qwen]  # final only when every required model's results are in the input
    status = "final" if len(present) == len(MODELS) else "partial" if present else "template"
    tabs = {"Sol": per_task([(r["main_task"], r["main_task_success"]) for r in sol["rows"]])}
    tabs |= {m: per_task([(r["sample_id"].split(":")[0], r["main_task_success"]) for r in qwen[m]["rows"]]) for m in MODELS if m in qwen}
    for m, tab in tabs.items():  # the rows must reproduce each input's own counts
        want = (sol["successes"], sol["runs"]) if m == "Sol" else (qwen[m]["successes"], qwen[m]["runs"])
        if tuple(map(sum, zip(*tab.values()))) != want:
            raise ValueError(f"{m}: rows disagree with the counts")
    tasks, res = paired_boot(list(tabs.values()))
    avg = dict(zip(tabs, task_averaged(list(tabs.values()), tasks)))
    res = dict(zip(tabs, res))
    ci = lambda d: np.percentile(d, [2.5, 97.5])
    s_rate, s_draws, sk, sn = res["Sol"]
    lo, hi = ci(s_draws)
    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    ax.axhspan(lo, hi, color="0.85", zorder=0, label=f"Sol, real honest runs: {s_rate:.0%} passed [{lo:.0%}, {hi:.0%}] ({sk} of {sn} runs)")
    ax.axhline(s_rate, color="0.3", ls="--", zorder=1)
    for x, m in enumerate(MODELS):
        if m in res:
            r, d, k, n = res[m]
            (a, b), (dl, dh) = ci(d), ci(d - s_draws)
            ax.bar(x, r, color="tab:blue", width=0.45, zorder=2)
            ax.errorbar(x, r, yerr=[[r - a], [b - r]], color="black", capsize=6, zorder=3)
            ax.text(x, b + 0.03, f"{r:.0%} passed [{a:.0%}, {b:.0%}] ({k} of {n} runs)\n{m} minus Sol: {100 * (r - s_rate):+.1f} points "
                    f"[{100 * dl:+.1f}, {100 * dh:+.1f}]", ha="center", fontsize=9.5)
            v = [x_["main_task_success"] for x_ in qwen[m]["rows"]]
            timed_out = all("timed out" in u.get("explanation", "") for u in qwen[m]["unscored"])
            ax.text(x, 0.03, f"all {len({r_['sample_id'].split(':')[0] for r_ in qwen[m]['rows']})} tasks,\ntwo honest runs each:\n"
                    f"passed {v.count('C')}, failed {v.count('I')}\nnot scored {v.count('N')}"
                    + ("\n(scorer timed out)" if v.count("N") and timed_out else ""), ha="center", fontsize=8, color="white")
        else:
            ax.text(x, 0.5, f"{m}:\nno result yet", ha="center", va="center", fontsize=11, color="0.4")
    ax.set_xticks(range(len(MODELS)), [f"{m}\n(acting as the agent)" for m in MODELS])
    ax.set_xlim(-0.9, len(MODELS) - 0.1)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Share of honest runs that passed the task (higher is better)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.set_title(f"Does Qwen3.5-27B finish honest LinuxArena tasks as often as Sol?\n({len(tasks)} tasks scored for both models)", fontsize=12)
    first = next((m for m in MODELS if m in tabs), None)
    ties = sum(tabs[first][t][0] * tabs["Sol"][t][1] == tabs["Sol"][t][0] * tabs[first][t][1] for t in tasks) if first else 0
    sha = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    foot = [f"Preliminary. Bars and band: share of scored runs that passed, on the {len(tasks)} tasks scored for both; 95% intervals from resampling tasks.",
            f"Sol ran in a different harness; Qwen and Sol tie on {ties} of {len(tasks)} tasks." if first else "Sol ran in a different harness.",
            f"Data: {sol_path.name}" + (f", {qwen_path.name}" if qwen_path else "") + f"; script at commit {sha}."]
    fig.text(0.01, 0.01, "\n".join(foot), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for m in MODELS:  # printed, not drawn: the sensitivity and the context check (the README carries both)
        if m in avg:
            print(f"task-averaged: Sol {avg['Sol'][0]:.4f}, {m} {avg[m][0]:.4f}, difference {100 * (avg[m][0] - avg['Sol'][0]):+.1f} points "
                  f"{[round(100 * v_, 1) for v_ in ci(avg[m][1] - avg['Sol'][1])]}; Sol runs per task: {runs_per_task(tabs['Sol'], tasks)}")
        if m in qwen and "runs over 32,768 tokens (all, successful)" in qwen[m]:
            print(f"{m} runs over 32,768 tokens (all, successful): {qwen[m]['runs over 32,768 tokens (all, successful)']}; successful runs over it "
                  f"after dividing by {TOKEN_RATIO}: {sum(r_['peak_context_tokens'] / TOKEN_RATIO > 32768 for r_ in qwen[m]['rows'] if r_['main_task_success'] == 'C')}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"figure1_{'honest_success' if status == 'final' else status}_plain_{stamp}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
