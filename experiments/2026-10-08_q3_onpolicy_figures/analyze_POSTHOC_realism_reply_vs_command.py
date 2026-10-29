"""Realism at the reply tokens vs the command tokens, same runs and reader (RULE_POSTHOC_realism_reply_196.md; POST HOC).
$0, local. Reads the two run_realism_probe.py job outputs (the command-token job and the reply-token job, separation
only), restricts both to the tasks they share (same folds), and reports for each: call- and run-level AUROC (mean within
fold), catch at 1%, the run 2 vs run 1 control and the 6-shuffle null; then paired differences (reply − command) on the
same 2,000 task resamples. One figure. Counts only.

  uv run --no-project --with numpy --with scipy --with scikit-learn --with matplotlib python \\
      analyze_POSTHOC_realism_reply_vs_command.py <command job output> <reply job output>
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_POSTHOC_realism_196 import SEED, separation

HERE = Path(__file__).parent
METRICS = ("call_auroc_mean_within_fold", "run_auroc_mean_within_fold", "call_catch_at_1pct", "run_catch_at_1pct")


def arm(J, task_fold):
    J2 = J | {"real": J["realism"]["cells"]["rs->rs"]["per_call"], "ctrl": J["control"]["cells"]["rr->rr"]["per_call"]}
    draws = []
    out = {"realism": separation(J2, "real", "sim", task_fold, draws_out=draws), "control": separation(J2, "ctrl", "run2", task_fold)}
    tasks = sorted({m["task"] for m in J["meta"]})
    ti = {t: i for i, t in enumerate(tasks)}  # the job's own task index, as its shuffles drew flips
    nulls = []
    for nl in J["nulls"]:
        flip = np.random.default_rng([SEED, nl["i"]]).integers(0, 2, len(tasks))
        ylab = {r: int((m["world"] == "sim") ^ flip[ti[m["task"]]]) for r, m in enumerate(J["meta"]) if m["world"] in ("real", "sim")}
        nulls.append(separation(J2 | {"null": nl["per_call"]}, "null", "sim", task_fold, ylab=ylab, n_boot=0))
    for m in ("call_auroc_mean_within_fold", "run_auroc_mean_within_fold"):
        v = np.array([x[m] for x in nulls])
        o = out["realism"][m]
        out.setdefault("null", {})[m] = {"values": v.tolist(), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()),
                                         "max": float(v.max()), "p": (1 + int((v >= o).sum())) / (1 + len(v))}
    out["layers"] = [{k: d[k] for k in ("fold", "layer", "C", "inner_auroc")} for d in J["realism"]["cells"]["rs->rs"]["folds"]
                     if d["threshold_kind"] == "calibrated"]
    return out, draws


def main(command: str, reply: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    raw = {k: Path(p).read_bytes() for k, p in (("command", command), ("reply", reply))}
    J = {k: json.loads(v) for k, v in raw.items()}
    if J["command"]["config"].get("readout_kind", "command") != "command" or J["reply"]["config"].get("readout_kind") != "output_end":
        raise SystemExit("inputs: the first must be the command-token job, the second the reply-token job")
    tasks = {k: {m["task"] for m in j["meta"]} for k, j in J.items()}
    common = sorted(tasks["command"] & tasks["reply"])
    if any(J["command"]["folds_map"][t] != J["reply"]["folds_map"][t] for t in common):
        raise SystemExit("the two jobs' folds differ")
    task_fold = {t: J["command"]["folds_map"][t] for t in common}
    out = {"git": git, "rule": "RULE_POSTHOC_realism_reply_196.md", "inputs": {k: [Path(p).name, hashlib.sha256(raw[k]).hexdigest()]
                                                                             for k, p in (("command", command), ("reply", reply))},
           "jobs": {k: {"key": j["key"], "app_id": j["app_id"], "git": j["git"]} for k, j in J.items()},
           "tasks": {"command": len(tasks["command"]), "reply": len(tasks["reply"]), "common": len(common)},
           "rows": {k: {w: sum(m["world"] == w and m["task"] in task_fold for m in j["meta"]) for w in ("real", "sim", "run2")} for k, j in J.items()}}
    D = {}
    for k in ("command", "reply"):
        out[k], D[k] = arm(J[k], task_fold)
    out["paired_reply_minus_command"] = {}
    for m in METRICS:
        pts = [(b[m], a[m]) for a, b in zip(D["command"], D["reply"]) if a[m] is not None and b[m] is not None]
        diff = np.array([r - c for r, c in pts])
        out["paired_reply_minus_command"][m] = {"point": out["reply"]["realism"][m] - out["command"]["realism"][m],
                                                "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))], "n_draws": len(diff)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"realism_reply_vs_command_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    pr = lambda x: f"{x:.3f}"  # noqa: E731
    print("tasks:", out["tasks"], "rows:", out["rows"])
    for k in ("command", "reply"):
        for part in ("realism", "control"):
            X = out[k][part]
            print(f"{k} {part}: call AUROC {pr(X['call_auroc_mean_within_fold'])} {[pr(v) for v in X['call_auroc_mean_within_fold_ci95']]}; "
                  f"run AUROC {pr(X['run_auroc_mean_within_fold'])} {[pr(v) for v in X['run_auroc_mean_within_fold_ci95']]}; "
                  f"call catch@1% {pr(X['call_catch_at_1pct'])} {[pr(v) for v in X['call_catch_at_1pct_ci95']]}; run catch@1% {pr(X['run_catch_at_1pct'])}")
        print(f"{k} null:", json.dumps({m: {x: out[k]['null'][m][x] for x in ('mean', 'min', 'max', 'p')} for m in out[k]['null']}))
        print(f"{k} layers:", out[k]["layers"])
    for m, v in out["paired_reply_minus_command"].items():
        print(f"paired reply - command {m}: {pr(v['point'])} {[pr(x) for x in v['ci95']]}")
    figure(out, dst, stamp)
    print("->", dst)


def figure(out, dst, stamp):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.4))
    cols = {"realism": "#d62728", "control": "#7f7f7f"}
    for i, lvl in enumerate(("call", "run")):
        for j, k in enumerate(("command", "reply")):
            for h, part in enumerate(("realism", "control")):
                x = i * 5.5 + j * 2.4 + h
                X = out[k][part]
                v, (lo, hi) = X[f"{lvl}_auroc_mean_within_fold"], X[f"{lvl}_auroc_mean_within_fold_ci95"]
                ax.bar(x, v, color=cols[part], width=0.9, label=("real vs simulated" if part == "realism" else "real run 1 vs run 2 (control)")
                       if i == 0 and j == 0 else None)
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color="k", capsize=3)
                ax.text(x, 0.04, f"{v:.2f}\n[{lo:.2f},\n{hi:.2f}]", ha="center", va="bottom", color="white", fontsize=7)
            nv = out[k]["null"][f"{lvl}_auroc_mean_within_fold"]
            ax.plot([i * 5.5 + j * 2.4 - 0.45, i * 5.5 + j * 2.4 + 1.45], [nv["mean"]] * 2, color="#1f77b4", ls="--", lw=1.2,
                    label=f"label-shuffle null, mean (command: {len(out['command']['null']['call_auroc_mean_within_fold']['values'])} "
                          f"shuffles; reply: {len(out['reply']['null']['call_auroc_mean_within_fold']['values'])})" if i == 0 and j == 0 else None)
            ax.text(i * 5.5 + j * 2.4 + 0.5, 1.0, f"{k}\ntokens", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([1.7, 7.2], ["call level", "run level (mean over a run's calls)"])
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("AUROC, mean over held-out folds\n(0.5 = can't tell apart; higher = easier to tell apart)")
    ax.set_title("Realism at command vs reply tokens, same runs and reader (Qwen3.5-27B, 196-task acting set-up; POST HOC)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    P = out["paired_reply_minus_command"]
    txt = (f"Paired reply − command (same {out['tasks']['common']} tasks, 2,000 task resamples within folds, thresholds re-estimated, not refits): "
           + "; ".join(f"{m.replace('_mean_within_fold', '').replace('_', ' ')} {v['point']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                       for m, v in P.items())
           + f". Shuffle nulls: command {len(out['command']['null']['call_auroc_mean_within_fold']['values'])}, reply "
           f"{len(out['reply']['null']['call_auroc_mean_within_fold']['values'])} (the command job was stopped after shuffle 5). "
           f"Rows (real/sim/run2): command {out['rows']['command']}, reply {out['rows']['reply']}. {dst.name} @ {out['git'][:8]}")
    fig.text(0.01, 0.005, "\n".join(txt[i: i + 160] for i in range(0, len(txt), 160)), fontsize=6.5, family="monospace", va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fdst = HERE / "figures" / f"realism_reply_vs_command_POSTHOC_{stamp}.png"
    fig.savefig(fdst, dpi=150, bbox_inches="tight")
    print("figure ->", fdst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command")
    ap.add_argument("reply")
    main(**vars(ap.parse_args()))
