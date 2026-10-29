"""k = 1 stage: one September Astra simulator call per site (Oct 7's `run_simulate_pairs.simulate`, imported unchanged),
cached by the sha256 of the site's record and appended to <cache>/sim_results.jsonl (resumable: cached sites are
skipped). Spend is enforced in code before each batch: spent (priced from each call's usage by sim.prices, as Oct 7) +
the $35 Modal reserve + the batch's worst case (its calls at Oct 7's max, $1.43, times 1.5) must stay at or below $400,
else the run stops. Failures are recorded and counted, never scored; any refusal substitution or failure stops further
batches until read. Keys: OPENAI_API_KEY from the repo's .env, loaded into this process only; run from a folder with no
.env above it. Prints keys and numbers only.

  cd $TMPDIR && uv run --project ~/simulated-control-evals-sept --no-sync python \\
      <repo>/experiments/2026-10-11_claimB_k_replies/run_k1_simulate.py --cache <dir> [--check-only N | --limit N]
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "2026-10-06_matched_step_pairs"))
import run_simulate_pairs as RSIM  # noqa: E402  (simulate, cost, log_judges, no_dotenv_above, MODEL)

STOP_USD, SLACK, MODAL_RESERVE, WORST_CALL, CONCURRENCY = 400.0, 20.0, 35.0, 5.0, 8
# $5 a call reserved in flight (3.5x Oct 7's max of $1.43); not a provable bound (the simulator can repair and re-ground),
# so $20 of slack is kept under the $400 stop.
STOPS = ("fallback_substitutions", "refusal_substitutions", "block_substitutions")  # refusal / filter paths: stop and read
# net_substitutions (the router sending network commands to the fallback by design) is recorded, not a stop
ENV = Path.home() / "probe-distribution-shift-from-sims" / ".env"


def load_key():
    for line in ENV.read_text().splitlines():
        if line.split("=", 1)[0].strip() == "OPENAI_API_KEY":
            key = line.split("=", 1)[1].strip().strip('"').strip("'").strip()
            if not re.fullmatch(r"[A-Za-z0-9_\-]+", key):
                raise SystemExit("OPENAI_API_KEY malformed (not shown)")
            os.environ["OPENAI_API_KEY"] = key
            return
    raise SystemExit("OPENAI_API_KEY missing (not shown)")


def ledger(results: Path, build: dict | None = None) -> tuple[float, set, dict]:
    spent, done, counts = 0.0, set(), {}
    starts = results.with_name("sim_starts.jsonl")
    started = {json.loads(l)["record_sha256"] for l in starts.read_text().splitlines()} if starts.exists() else set()
    if results.exists():
        for line in results.read_text().splitlines():
            r = json.loads(line)
            if build is not None and r["build"] != build:
                raise SystemExit("sim_results.jsonl holds rows from another simulator build or script; use a new cache")
            spent += r.get("cost_usd") or 0.0
            done.add(r["record_sha256"])
            counts[r["status"]] = counts.get(r["status"], 0) + 1
    if started - done:  # a call that started and never finished may have been billed: reconcile by hand first
        raise SystemExit(f"{len(started - done)} started calls have no result; reconcile before resuming (each may have cost up to $5)")
    return spent, done, counts


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
    cost = RSIM.cost(usage) if usage else 0.0
    if status == "ok" and any(out.get(k) for k in STOPS):
        status = "substitution"
    return {"site": rec["site"], "record_sha256": digest, "status": status, "cost_usd": cost, "usage": usage, "build": build,
            "seconds": round(time.time() - t0, 1), "result": out, "judge_calls": judge_log}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--check-only", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    RSIM.no_dotenv_above(Path.cwd())
    import logging
    import warnings
    logging.disable(logging.CRITICAL)  # the snapshot's warnings can echo transcript fragments; flags are kept per row instead
    warnings.filterwarnings("ignore")
    load_key()
    RSIM.log_judges()
    recs = [json.loads(l) for l in (a.cache / "records.jsonl").read_text().splitlines()]
    sims = lambda *x: RSIM.subprocess.run(["git", "-C", str(RSIM.SEPT), *x], capture_output=True, text=True).stdout  # noqa: E731
    envs = sorted({r["environment"] for r in recs})
    build = {"simulator_model": RSIM.MODEL, "sims_commit": sims("rev-parse", "HEAD").strip(),
             "sims_src_diff_sha256": hashlib.sha256(sims("diff", "HEAD", "--", "src").encode()).hexdigest(),
             "sims_lock_sha256": hashlib.sha256((RSIM.SEPT / "uv.lock").read_bytes()).hexdigest(),
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "oct7_script_sha256": hashlib.sha256(Path(RSIM.__file__).read_bytes()).hexdigest(),
             "codebase_sha256": {e: RSIM.tree_sha256(RSIM.SEPT / ".settings" / "linuxarena" / e / "codebase") for e in envs}}
    digests = [hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in recs]
    if a.check_only:
        out = [await one(r, d, True, build) for r, d in list(zip(recs, digests))[: a.check_only]]
        print(json.dumps([{k: o[k] for k in ("site", "status", "cost_usd")} | {k: o["result"].get(k) for k in ("system_prompt_matches_september", "trusted_prefix", "memory_messages", "error_type")} for o in out], indent=1))
        return
    results = a.cache / "sim_results.jsonl"
    spent, done, counts = ledger(results, build)
    if any(k != "ok" for k in counts):
        print(json.dumps({"stopped": "a failed or substituted call needs reading first", "statuses": counts, "spent_usd": round(spent, 2)}))
        return
    todo = [(r, d) for r, d in zip(recs, digests) if d not in done]
    if a.limit:
        todo = todo[: a.limit]
    state = {"spent": spent, "in_flight": 0, "stop": None, "statuses": dict(counts)}
    lock = asyncio.Lock()

    async def worker(queue):
        while True:
            try:
                r, d = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            async with lock:  # a reservation: spent + Modal reserve + every in-flight call at $5 stays <= $400 - slack
                if state["stop"] or state["spent"] + MODAL_RESERVE + (state["in_flight"] + 1) * WORST_CALL > STOP_USD - SLACK:
                    state["stop"] = state["stop"] or "the next call could pass the $400 stop"
                    return
                state["in_flight"] += 1
                with open(results.with_name("sim_starts.jsonl"), "a") as fh:  # written before the call is sent
                    fh.write(json.dumps({"site": r["site"], "record_sha256": d, "t": time.strftime("%H:%M:%S")}) + "\n")
            o = await one(r, d, False, build)
            async with lock:
                with open(results, "a") as fh:  # every completed call is written at once
                    fh.write(json.dumps(o) + "\n")
                state["in_flight"] -= 1
                state["spent"] += o["cost_usd"]
                state["statuses"][o["status"]] = state["statuses"].get(o["status"], 0) + 1
                res = o["result"]
                if o["status"] != "ok" or any(res.get(k) for k in STOPS):
                    state["stop"] = f"site {o['site']}: status {o['status']} or a substitution; stop and read"
                n = sum(state["statuses"].values())
                if n % 20 == 0:
                    print(json.dumps({"t": time.strftime("%H:%M:%S"), "done": n, "spent_usd": round(state["spent"], 2)}), flush=True)

    queue = asyncio.Queue()
    for item in todo:
        queue.put_nowait(item)
    await asyncio.gather(*(worker(queue) for _ in range(CONCURRENCY)))
    print(json.dumps({"finished": time.strftime("%H:%M:%S"), "spent_usd": round(state["spent"], 2), "statuses": state["statuses"],
                      "left": queue.qsize(), "stop": state["stop"]}))

if __name__ == "__main__":
    asyncio.run(main())
