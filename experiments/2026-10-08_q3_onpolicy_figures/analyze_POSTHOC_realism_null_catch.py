"""Null for the realism catch values (RULE_POSTHOC_realism_reply_196.md, amendment 2; POST HOC, from #1031's review). $0,
local, from the stored shuffle scores. For each arm (command tokens, reply tokens) on the 190 shared tasks: catch at 1%
(call and run) on every saved shuffle with the observed values' rule (analyze_POSTHOC_realism_196.separation, no
resampling), p = (1 + #null >= observed) / (1 + n); and the reply - command catch gap against the paired shuffles
i = 1..5. Counts only.

  uv run --no-project --with numpy --with scipy --with scikit-learn python analyze_POSTHOC_realism_null_catch.py \\
      <command job output> <reply job output>
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
M = ("call_catch_at_1pct", "run_catch_at_1pct")


def arm(J, task_fold):
    J2 = J | {"real": J["realism"]["cells"]["rs->rs"]["per_call"]}
    obs = separation(J2, "real", "sim", task_fold, n_boot=0)
    tasks = sorted({m["task"] for m in J["meta"]})
    ti = {t: i for i, t in enumerate(tasks)}  # the job's own task index, as its shuffles drew flips
    nulls = {}
    for nl in J["nulls"]:
        flip = np.random.default_rng([SEED, nl["i"]]).integers(0, 2, len(tasks))
        ylab = {r: int((m["world"] == "sim") ^ flip[ti[m["task"]]]) for r, m in enumerate(J["meta"]) if m["world"] in ("real", "sim")}
        nulls[nl["i"]] = separation(J2 | {"null": nl["per_call"]}, "null", "sim", task_fold, ylab=ylab, n_boot=0)
    out = {"observed": {m: obs[m] for m in M}, "n_null": len(nulls)}
    for m in M:
        v = [nulls[i][m] for i in sorted(nulls)]
        out[m] = {"null": v, "null_mean": float(np.mean(v)), "null_max": float(np.max(v)),
                  "p": (1 + sum(x >= obs[m] for x in v)) / (1 + len(v))}
    return out, nulls


def main(command: str, reply: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    raw = {"command": Path(command).read_bytes(), "reply": Path(reply).read_bytes()}
    J = {k: json.loads(v) for k, v in raw.items()}
    if J["command"]["config"].get("readout_kind", "command") != "command" or J["reply"]["config"].get("readout_kind") != "output_end":
        raise SystemExit("inputs: the first must be the command-token job, the second the reply-token job")
    common = sorted({m["task"] for m in J["command"]["meta"]} & {m["task"] for m in J["reply"]["meta"]})
    if any(J["command"]["folds_map"][t] != J["reply"]["folds_map"][t] for t in common):
        raise SystemExit("the two jobs' folds differ")
    task_fold = {t: J["command"]["folds_map"][t] for t in common}
    out = {"git": git, "rule": "RULE_POSTHOC_realism_reply_196.md, amendment 2", "tasks": len(common),
           "inputs": {k: [Path(p).name, hashlib.sha256(raw[k]).hexdigest()] for k, p in (("command", command), ("reply", reply))}}
    N = {}
    for k in ("command", "reply"):
        out[k], N[k] = arm(J[k], task_fold)
    pairs = sorted(set(N["command"]) & set(N["reply"]))  # shuffles 1..5: the command job stopped after shuffle 5
    out["gap_reply_minus_command"] = {"paired_shuffles": pairs}
    for m in M:
        o = out["reply"]["observed"][m] - out["command"]["observed"][m]
        g = [N["reply"][i][m] - N["command"][i][m] for i in pairs]
        out["gap_reply_minus_command"][m] = {"observed": o, "null": g, "null_mean": float(np.mean(g)), "null_max": float(np.max(g)),
                                             "p": (1 + sum(x >= o for x in g)) / (1 + len(g))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"realism_null_catch_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    pct = lambda x: f"{100 * x:.2f}%"  # noqa: E731
    for k in ("command", "reply"):
        for m in M:
            r = out[k][m]
            print(f"{k} {m}: observed {pct(out[k]['observed'][m])}; null {[pct(x) for x in r['null']]} (mean {pct(r['null_mean'])}, "
                  f"max {pct(r['null_max'])}); p = {r['p']:.3f} (n = {out[k]['n_null']})")
    for m in M:
        r = out["gap_reply_minus_command"][m]
        print(f"gap reply - command {m}: observed {100 * r['observed']:+.2f} pts; null {[f'{100 * x:+.2f}' for x in r['null']]} "
              f"(mean {100 * r['null_mean']:+.2f}, max {100 * r['null_max']:+.2f}); p = {r['p']:.3f} (n = {len(pairs)})")
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command")
    ap.add_argument("reply")
    main(**vars(ap.parse_args()))
