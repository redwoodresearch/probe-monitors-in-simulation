"""Step 2 pilot: the September Astra simulator rewrites every tool reply of 30 real honest runs, each call given the REAL
history before it (Oct 7's `run_simulate_pairs.simulate`, imported unchanged, one call per site as the k = 1 stage).

Runs are taken in draw order; up to --runs-in-parallel runs at once, and each run's calls go in order k = 0, 1, ...
so the provider's prompt cache can reuse the run's shared prefix (system prompt and earlier calls). Every call is cached
by the sha256 of its record and appended to <cache>/sim_results.jsonl at once (resumable). Spend is enforced before each
call from provider usage (priced by sim.prices, as Oct 7): spent + every in-flight call at $3 (k = 1's max was $1.16,
Oct 7's $1.43) must stay at or below the $560 API stop, else nothing new starts. $3 is not a provable bound (a call can
repair, judge and fall back), so the stop leaves $25 under the $600 cap after Modal's $15, and any single call above $3
stops new starts. A failed call counts as at least $3 (a timed-out request may be billed without reported usage). A
failed call is retried once; a call that fails twice marks its run failed (counted, never scored), and 3 failed runs stop
the pilot. Attempts and failed runs are rebuilt from the ledger on resume. Fallback, refusal,
block and network substitutions are the September pipeline's normal paths: recorded per call and counted, not stops.
Keys come only from op run (OPENAI_API_KEY in the environment); run from a folder with no .env above it.

  cd $TMPDIR && op run --env-file=<repo>/.env.tpl -- uv run --project ~/simulated-control-evals-sept --no-sync python \\
      <repo>/experiments/2026-10-13_step2_sim_rewrite/run_rewrite_simulate.py --cache <dir> [--check-only N | --max-runs N]
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "2026-10-06_matched_step_pairs"))
import run_simulate_pairs as RSIM  # noqa: E402  (simulate, cost, log_judges, no_dotenv_above, tree_sha256, MODEL, SEPT)

STOP_USD, WORST_CALL, MAX_FAILED_RUNS = 560.0, 3.0, 3
SUBS = ("fallback_substitutions", "refusal_substitutions", "block_substitutions", "net_substitutions")


def load(cache: Path):
    runs = {r["run"]: r for r in map(json.loads, (cache / "runs.jsonl").read_text().splitlines())}
    calls = {}
    for c in map(json.loads, (cache / "calls.jsonl").read_text().splitlines()):
        calls.setdefault(c["run"], []).append(c)
    run_sha = {i: hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for i, r in runs.items()}
    items = {}
    for i, cs in calls.items():
        assert [c["k"] for c in cs] == list(range(runs[i]["n_actions"])), f"run {i}: calls out of order or missing"
        items[i] = [(runs[i] | c | {"site": f"{i}:{c['k']}"},
                     hashlib.sha256((run_sha[i] + json.dumps(c, sort_keys=True)).encode()).hexdigest()) for c in cs]
    return runs, items


def charge(row: dict) -> float:
    """What a finished call counts against the stop: its priced usage, and at least $3 if it failed."""
    return max(row["cost_usd"], WORST_CALL) if row["status"] == "failed" else row["cost_usd"]


def ledger(results: Path, build: dict):
    spent, done, counts, fails = 0.0, set(), {}, {}
    starts = results.with_name("sim_starts.jsonl")
    started = [json.loads(l)["record_sha256"] for l in starts.read_text().splitlines()] if starts.exists() else []
    finished = []
    if results.exists():
        for line in results.read_text().splitlines():
            r = json.loads(line)
            if r["build"] != build:
                raise SystemExit("sim_results.jsonl holds rows from another simulator build or script; use a new cache")
            spent += charge(r)
            finished.append(r["record_sha256"])
            if r["status"] == "ok":
                done.add(r["record_sha256"])
            elif r["status"] == "failed":
                fails[r["record_sha256"]] = fails.get(r["record_sha256"], 0) + 1
            counts[r["status"]] = counts.get(r["status"], 0) + 1
    if len(started) != len(finished):  # a call that started and never finished may have been billed: reconcile by hand
        raise SystemExit(f"{len(started) - len(finished)} started calls have no result; reconcile before resuming")
    return spent, done, counts, fails


async def one(rec: dict, digest: str, check_only: bool, build: dict) -> dict:
    from inspect_ai.model._model import init_sample_model_usage, sample_model_usage

    t0 = time.time()
    init_sample_model_usage()  # this task's own usage ledger, so a failed attempt's spend is still counted (as Oct 7)
    judge_log = []
    RSIM.JUDGE_LOG.set(judge_log)
    try:
        out = await RSIM.simulate(rec, check_only=check_only)
        status = "check_ok" if check_only else "ok"
    except Exception as e:  # noqa: BLE001  (recorded and counted, never scored)
        out, status = {"error_type": type(e).__name__, "error": f"{type(e).__name__}: {e}"[:2000]}, "failed"
    usage = {m: u.model_dump(exclude_none=True) for m, u in sample_model_usage().items()}
    return {"site": rec["site"], "run": rec["run"], "k": rec["k"], "record_sha256": digest, "status": status,
            "cost_usd": RSIM.cost(usage) if usage else 0.0, "usage": usage, "build": build,
            "seconds": round(time.time() - t0, 1), "t": time.strftime("%H:%M:%S"), "result": out, "judge_calls": judge_log}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--check-only", type=int, default=0, help="build N runs' first simulators and compare prompts; no model call")
    p.add_argument("--max-runs", type=int, default=0)
    p.add_argument("--runs-in-parallel", type=int, default=10)
    a = p.parse_args()
    RSIM.no_dotenv_above(Path.cwd())
    import logging
    import warnings

    logging.disable(logging.CRITICAL)  # the snapshot's warnings can echo transcript fragments; flags are kept per row instead
    warnings.filterwarnings("ignore")
    assert a.check_only or os.environ.get("OPENAI_API_KEY"), "no OPENAI_API_KEY: run under op run"
    RSIM.log_judges()
    runs, items = load(a.cache)
    sims = lambda *x: RSIM.subprocess.run(["git", "-C", str(RSIM.SEPT), *x], capture_output=True, text=True).stdout  # noqa: E731
    envs = sorted({r["environment"] for r in runs.values()})
    build = {"simulator_model": RSIM.MODEL, "sims_commit": sims("rev-parse", "HEAD").strip(),
             "sims_src_diff_sha256": hashlib.sha256(sims("diff", "HEAD", "--", "src").encode()).hexdigest(),
             "sims_lock_sha256": hashlib.sha256((RSIM.SEPT / "uv.lock").read_bytes()).hexdigest(),
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "oct7_script_sha256": hashlib.sha256(Path(RSIM.__file__).read_bytes()).hexdigest(),
             "codebase_sha256": {e: RSIM.tree_sha256(RSIM.SEPT / ".settings" / "linuxarena" / e / "codebase") for e in envs}}
    order = sorted(items)
    if a.check_only:  # the deepest call of each run: the whole real prefix must pass the September contract audit
        out = [await one(items[i][-1][0], items[i][-1][1], True, build) for i in order[: a.check_only]]
        print(json.dumps([{k: o[k] for k in ("site", "status")} | {k: o["result"].get(k) for k in ("system_prompt_matches_september", "trusted_prefix", "memory_messages", "error")} for o in out], indent=1))
        return
    results = a.cache / "sim_results.jsonl"
    spent, done, counts, fails = ledger(results, build)
    if a.max_runs:
        order = order[: a.max_runs]
    dead = sorted({i for i in items for _, d in items[i] if fails.get(d, 0) >= 2 and d not in done})
    state = {"spent": spent, "in_flight": 0, "stop": None, "statuses": dict(counts), "failed_runs": dead}
    if len(dead) >= MAX_FAILED_RUNS:
        state["stop"] = f"{len(dead)} runs failed before this launch; stop and read"
    lock = asyncio.Lock()

    async def call(rec, digest):
        async with lock:  # a reservation: spent + every in-flight call at $3 stays <= the API stop
            if state["stop"] or state["spent"] + (state["in_flight"] + 1) * WORST_CALL > STOP_USD:
                state["stop"] = state["stop"] or f"the next call could pass the ${STOP_USD:.0f} API stop"
                return None
            state["in_flight"] += 1
            with open(results.with_name("sim_starts.jsonl"), "a") as fh:  # written before the call is sent
                fh.write(json.dumps({"site": rec["site"], "record_sha256": digest, "t": time.strftime("%H:%M:%S")}) + "\n")
        o = await one(rec, digest, False, build)
        async with lock:
            with open(results, "a") as fh:  # every finished call is written at once
                fh.write(json.dumps(o) + "\n")
            state["in_flight"] -= 1
            state["spent"] += charge(o)
            state["statuses"][o["status"]] = state["statuses"].get(o["status"], 0) + 1
            if o["cost_usd"] > WORST_CALL:
                state["stop"] = f"site {o['site']} cost ${o['cost_usd']:.2f}, above the ${WORST_CALL:.0f} reservation; stop and read"
        return o

    async def do_run(i):
        t0, cost, subs = time.time(), 0.0, {k: 0 for k in SUBS}
        if i in state["failed_runs"]:
            return
        for rec, digest in items[i]:
            if digest in done:
                continue
            o = {"result": {"error_type": "failed twice before this launch"}}
            for _attempt in range(max(0, 2 - fails.get(digest, 0))):
                o = await call(rec, digest)
                if o is None:
                    return
                cost += o["cost_usd"]
                if o["status"] == "ok":
                    for k in SUBS:
                        subs[k] += int(bool(o["result"].get(k)))
                    break
            else:
                async with lock:
                    state["failed_runs"].append(i)
                    if len(state["failed_runs"]) >= MAX_FAILED_RUNS:
                        state["stop"] = f"{len(state['failed_runs'])} runs failed; stop and read"
                print(json.dumps({"run": i, "failed_at_k": rec["k"], "error": o["result"].get("error_type")}), flush=True)
                return
        print(json.dumps({"t": time.strftime("%H:%M:%S"), "run_done": i, "calls": len(items[i]), "usd_this_launch": round(cost, 2),
                          "minutes": round((time.time() - t0) / 60, 1), "substitutions": subs, "spent_usd": round(state["spent"], 2)}), flush=True)

    queue = asyncio.Queue()
    for i in order:
        queue.put_nowait(i)

    async def worker():
        while not state["stop"]:
            try:
                i = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await do_run(i)

    print(json.dumps({"started": time.strftime("%H:%M:%S"), "runs": len(order), "calls": sum(len(items[i]) for i in order),
                      "already_done": len(done), "spent_before_usd": round(spent, 2)}), flush=True)
    await asyncio.gather(*(worker() for _ in range(a.runs_in_parallel)))
    print(json.dumps({"finished": time.strftime("%H:%M:%S"), "spent_usd": round(state["spent"], 2), "statuses": state["statuses"],
                      "failed_runs": state["failed_runs"], "runs_not_started": queue.qsize(), "stop": state["stop"]}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
