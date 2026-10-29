"""Draw the Phase 2 pilot's 20 tasks and freeze what each attack run must be told (Sol's September attack approach
and the sha256 of Sol's system prompt and first user message), in the launch order. Laptop, no spend (Sol's samples
come from the S3 cache):

  uv run python experiments/2026-10-07_phase2_attack_check/run_select_tasks.py [--rule amendment9] [--out <path>]

Eligible cells are run_select_cells.py's: a Phase 1 paired attack cell (splits/exclusions_v4.json), Sol's epoch-1 side
task passed, September wrote an approach for it, and its main task is one of the 437 paired honest tasks (decision 3:
the filter stays, and every result states that population). A task is a main task: 2 distinct main tasks per
environment, 20 in all, so every environment is exercised and no environment dominates (clinical_trial alone has a
quarter of the eligible cells); each gets one of its eligible cells at random. The launch order is a random
permutation, fixed here before any run: run_pilot.py starts tasks strictly in it, so if the spend guard stops new tasks,
the ones run are a random subset. Writes config/pilot_tasks.json (refuses to overwrite).

--rule amendment9 instead implements #82's amendment 9 (PROPOSED as of 2026-10-08, for Aswin; #82 at d3f9605) exactly:
eligible cells are those where Sol's September real attack (epoch 1) passed the side-task scorer; one cell per main
task, the one whose id sorts first; within each environment, main tasks ordered by sha256("phase2-pilot-v1:" + main),
environments by sha256("phase2-pilot-v1:" + environment), both ascending; then round-robin across environments until
--n-tasks tasks (default 20; the task file records the resolved n_tasks). That order is also the launch order.

Each cell's September approach is kept out of git (approaches.py): the task file records its sha256 and S3 URI, the
text goes to the git-ignored data/phase2_attack_check/approaches/ and is uploaded to S3 (never overwriting; an object
already there must hold the same text). run_pilot.py restores and checks every text before use.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import approaches
import run_select_cells as cells_rule

HERE = Path(__file__).parent
PER_ENV, SEED, N_TASKS, SALT = 2, 20261008, 20, "phase2-pilot-v1:"


def amendment9(e1: dict, n_tasks: int = N_TASKS) -> tuple[list[str], dict]:
    """#82's amendment 9: e1-passed cells, one per main task (first by id), hash-ordered, round-robin by environment."""
    key = lambda x: hashlib.sha256((SALT + x).encode()).hexdigest()  # noqa: E731
    passed = sorted(c for c, r in e1.items() if r["side_success"] == "C")
    first = {}
    for c in passed:  # sorted, so the first cell seen for a main task is the one whose id sorts first
        first.setdefault(c.split(":")[0], c)
    by_env: dict[str, list[str]] = {}
    for main, c in first.items():
        by_env.setdefault(e1[c]["env"], []).append(main)
    order = {env: sorted(mains, key=key) for env, mains in by_env.items()}
    envs = sorted(order, key=key)
    chosen, rank = [], 0
    while len(chosen) < n_tasks and any(rank < len(order[e]) for e in envs):
        chosen += [first[order[e][rank]] for e in envs if rank < len(order[e])][:n_tasks - len(chosen)]
        rank += 1
    return chosen, {"n_eligible_cells": len(passed), "n_eligible_mains": len(first),
                    "eligible_mains_by_env": {e: len(m) for e, m in sorted(order.items())}}


def upload(digests: list[str]) -> None:
    """Each approach text to S3 under its sha256, never overwriting; an existing object must hold the same text."""
    from botocore.exceptions import ClientError
    from probe_shift.transcripts import _s3

    bucket, prefix = approaches.APPROACHES_S3.removeprefix("s3://").split("/", 1)
    for digest in sorted(set(digests)):
        body = (approaches.APPROACHES / f"{digest}.txt").read_bytes()
        assert hashlib.sha256(body).hexdigest() == digest, f"the local text doesn't hash to {digest}"
        try:
            _s3().put_object(Bucket=bucket, Key=f"{prefix}{digest}.txt", Body=body, IfNoneMatch="*")
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("PreconditionFailed", "412"):
                raise
        held = _s3().get_object(Bucket=bucket, Key=f"{prefix}{digest}.txt")["Body"].read()  # new or already there
        assert hashlib.sha256(held).hexdigest() == digest, f"S3 holds different text under {digest}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["candidate", "amendment9"], default="candidate")
    ap.add_argument("--out", type=Path, default=HERE / "config" / "pilot_tasks.json")
    ap.add_argument("--no-upload", action="store_true", help="keep the approach texts local only (a trial draw)")
    ap.add_argument("--n-tasks", type=int, default=N_TASKS, help="amendment 9 only: how many tasks in launch order (196 = every eligible one)")
    args = ap.parse_args()
    out = args.out
    assert not out.exists(), f"{out} exists; the draw is frozen"
    ex = json.loads(cells_rule.SPLITS.read_text())
    paired, honest = set(ex["paired_attack_cells"]["cells"]), set(ex["paired_honest_cells"]["cells"])
    e1, e2 = cells_rule.rows(cells_rule.E1), cells_rule.rows(cells_rule.E2)
    assert all(len(v) == 1 for v in e1.values()), "epoch 1 has a cell twice"
    e1 = {c: v[0] for c, v in e1.items()}
    eligible = sorted(c for c, r in e1.items() if c in paired and r["side_success"] == "C" and r["approach"].strip()
                      and c.split(":")[0] in honest)
    by_env: dict[str, dict[str, list[str]]] = {}  # env -> main task -> its eligible cells
    for c in eligible:
        by_env.setdefault(e1[c]["env"], {}).setdefault(c.split(":")[0], []).append(c)
    counts = {"n_eligible_cells": len(eligible), "n_eligible_mains": sum(len(m) for m in by_env.values()),
              "eligible_mains_by_env": {e: len(m) for e, m in sorted(by_env.items())}}
    if args.rule == "amendment9":
        chosen, counts = amendment9(e1, args.n_tasks)
    else:
        rng = random.Random(SEED)
        chosen = []
        for env in sorted(by_env):
            mains = sorted(by_env[env])
            assert len(mains) >= PER_ENV, f"{env}: only {len(mains)} eligible main tasks"
            for main in rng.sample(mains, PER_ENV):
                pool = sorted(by_env[env][main])
                chosen.append(pool[rng.randrange(len(pool))])
        rng.shuffle(chosen)  # the launch order
    tasks = []
    for i, cell in enumerate(chosen, 1):
        r, (main, side) = e1[cell], cell.split(":")
        tasks.append({"key": f"t{i:02d}_{r['env']}", "env": r["env"], "cell": cell, "main": main, "side": side,
                      **approaches.store(r["approach"]), "sol_system_sha256": r["system_sha256"],
                      "sol_input_sha256": r["input_sha256"],
                      "sol_e1": {k: r[k] for k in ("alias", "file", "id", "epoch", "side_success", "tool_calls")},
                      "sol_e2": [{k: x[k] for k in ("alias", "file", "id", "epoch", "side_success", "tool_calls")}
                                 for x in e2.get(cell, [])]})
    if not args.no_upload:
        upload([t["approach_sha256"] for t in tasks])
    # "cells": run_pilot.py restores the approaches into a run-local copy, which tasks.py reads as PHASE2_CELLS.
    rule = __doc__.split("\n\n")[3 if args.rule == "amendment9" else 2]
    out.write_text(json.dumps({"rule_name": args.rule, "seed": SEED if args.rule == "candidate" else None,
                               "salt": SALT if args.rule == "amendment9" else None, "rule": rule,
                               "n_tasks": args.n_tasks if args.rule == "amendment9" else N_TASKS} | counts
                              | {"cells": tasks}, indent=1))
    for t in tasks:
        print(t["key"], t["cell"], t["sol_e1"]["side_success"], [x["side_success"] for x in t["sol_e2"]])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
