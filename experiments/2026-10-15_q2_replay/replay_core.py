"""The replay's simulator pass (df's RULE v7 @ 604bc7c60, frozen): rewrite each eligible tool output of a fixed real run
with the frozen September simulator, conditioned on the run's fixed actions and the simulator's OWN earlier outputs.
Imported inside the container (B's sims tree 1ea37722 on the path); no Modal code here.

Live shape (the 1ea37722 tree): one simulator session per run; the eligible calls of one assistant message go together
through `Simulator.simulate_concurrent(action, session, expected=<eligible calls in that message>)` (sim/simulator.py:432),
which coalesces them into one fabricated step and fills them into the session's memory (`fill_output`, the same
`_committed_text` as `commit_output`: the memory output cap). Inspect-rejected calls (error type "parsing") are never
sent and never committed (RULE v7). A session is rebuilt, when needed (resume, after a failed step), by the commit path:
`action_message` + `commit_output` per committed output in order, which reproduced B's stored memories 191/191
(tree_swap_check_2026-10-15_014915.json @ 6614ee1d). The run's model clients (with the network router's remembered
payloads and route decisions) are kept across rebuilds, and their state is saved in every result row so a cold resume
restores it.

Per run: the eligible outputs come from the frozen predicate (replay_predicate.py); the simulated range is the initial
allocation K_r (c_r from the run's corrected-B staged epoch), whole steps (a step straddling K_r is simulated whole), or
a larger limit passed in for an extension. Per step, under one lock: no dispatch while PAUSE exists, after a cost
overrun, or if settled spend + every in-flight reservation + this step's reservation would pass the hard stop. The
reservation is $5 + $2 per sibling call (the largest priced call in 1,253 ledger calls was $1.32); a step that costs more
than its reservation stops all new dispatch (Step 2's trip-wire) and is reported. One attempt =
  - success: outputs recorded raw; status simulated_identical / simulated_changed (the Inspect-truncated output vs the
    real transcript text);
  - RateLimitError (anywhere in the cause chain): never a fallback; backoff on the injected clock (30, 60, 120, 240, 300,
    300... s). The per-call allowance (30 minutes) counts every backoff sleep and the whole wall time of every attempt of
    the step, so Inspect's own transport retries inside an attempt count too; it persists in the ledger, bounds each
    attempt at the transport boundary (a timeout), and when it is used up the run stops, its current and later outputs
    not_processed_before_stop;
  - any other failure: charged at least $3; after 2 failed attempts the step's REAL outputs are committed once, as
    kept_real_after_refusal if any request of the step ended with the provider's content filter, else
    kept_real_after_failure; the run continues. A resume re-applies both limits before dispatching again.
After an attempt, requests a failed sibling left running are drained before settlement.
Durability: a journal (a Modal Dict) holds, per run, the cumulative charge and the open dispatch, written BEFORE each
request can be billed and after each result; the ledger (one JSONL per run, every row chained: sha256 of the previous
digest and the row) holds the rows. Before any dispatch the pass verifies every ledger, charges every open dispatch
without a result (at its reservation, at least $3), takes the larger of the ledger's and the journal's charges, and
restores the overrun trip-wire latch. Backoff, kept-real, stop and failure rows are committed to the Volume at once.
Every tick (15 s): commit, reload, read PAUSE, then write and commit the status file (counts only; "finished": false
until a final write after pass_end sets it true with finished_pt; rate-limit errors
are those that surfaced after Inspect's internal retries), also while paused and before the first dispatch. An aborted
pass cancels its workers and writes no pass_end.
"""

import asyncio
import hashlib
import json
import math
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

RULE_SHA = "604bc7c60"
FAIL_MIN_USD, RL_CAP_S, RL_SCHEDULE = 3.0, 1800.0, (30, 60, 120, 240, 300)
INSPECT_TRUNC = 16 * 1024
SUBS = ("fallback_substitutions", "block_substitutions", "refusal_substitutions", "net_substitutions")
PT = ZoneInfo("America/Los_Angeles")
CURRENT: ContextVar = ContextVar("replay_ctx", default=None)  # {run, step, attempt, n, requests}: request logging and the stub


def reserve_usd(n_calls: int) -> float:
    """What one step reserves against the hard stop before dispatch."""
    return 5.0 + 2.0 * (n_calls - 1)


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def rate_limited(e: BaseException | None) -> bool:
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if "RateLimit" in type(e).__name__ or getattr(e, "status_code", None) == 429:
            return True
        e = e.__cause__ or e.__context__
    return False


class SimulatedCrash(BaseException):
    """Stub only: a container death between steps, for the resume test."""


class RealClock:
    async def sleep(self, s: float) -> None:
        await asyncio.sleep(s)


class Ledger:
    def __init__(self, path: Path, header: dict):
        self.path, self.rows = path, []
        if path.exists():
            self.rows = [json.loads(line) for line in path.read_text().splitlines()]
            prev = ""
            for r in self.rows:  # the chain must verify before anything is resumed
                d = r.pop("digest")
                assert d == h(prev + json.dumps(r, sort_keys=True)), f"{path.name}: ledger chain broken"
                r["digest"], prev = d, d
            assert self.rows and self.rows[0]["kind"] == "header" and self.rows[0]["config"] == header["config"], f"{path.name}: other config"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.append(header)

    def append(self, row: dict) -> None:
        prev = self.rows[-1]["digest"] if self.rows else ""
        row = dict(row, t_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        row["digest"] = h(prev + json.dumps(row, sort_keys=True))
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        self.rows.append(row)


class Pass:
    """One pass over a list of prepared runs, in one container, `concurrency` runs at once."""

    def __init__(self, *, root: Path, label: str, mode: str, runs: list, build, hard_stop: float, concurrency: int,
                 status_path: Path, pause_path: Path, config: dict, journal, vol=None, clock=None, harness=None,
                 fake_cost_per_step: float = 0.0, cost_fn=None, tick_s: float = 15.0, cap_margin: float = 0.0):
        self.root, self.label, self.mode, self.runs = root / label, label, mode, runs
        self.build, self.hard_stop, self.concurrency = build, hard_stop, concurrency
        self.status_path, self.pause_path, self.config, self.vol = status_path, pause_path, config, vol
        self.clock, self.harness, self.fake_cost, self.cost_fn = clock or RealClock(), harness, fake_cost_per_step, cost_fn
        self.tick_s = tick_s
        self.lock = asyncio.Lock()
        self.spent, self.inflight, self.paused, self.finished, self.overrun = 0.0, 0.0, False, False, None
        self.in_flight_runs, self.done_runs = set(), set()
        self.rl_events, self.end_events, self.dispatch_log, self.tick_log = [], [], [], []
        self.max_committed_exposure = 0.0
        self.finished_pt = None  # set only when every run finished (pass_end written); an aborted pass never sets it
        self.ledgers = {}
        self.journal = journal  # a durable key-value store (a Modal Dict): spend and open dispatches survive the container
        self.cap_margin = cap_margin  # dispatch stops this far below the hard stop: room for in-flight steps' overruns
        self.charged = {}

    # -- Volume, spend and control -----------------------------------------------------------------
    def _commit_reload(self, reload: bool) -> None:
        if self.vol is not None:
            self.vol.commit()
            if reload:
                self.vol.reload()

    async def commit_now(self) -> None:
        async with self.lock:
            await asyncio.to_thread(self._commit_reload, False)

    def jkey(self, rid: str) -> str:
        return f"{self.label}|{rid}"

    async def journal_open(self, rid: str, s: int, attempt: int, res: float) -> None:
        await self.journal.put(self.jkey(rid), {"charged": self.charged.get(rid, 0.0), "open": [s, attempt, res, time.time()], "overrun": self.overrun})

    async def journal_close(self, rid: str) -> None:
        """One put: the run's cumulative charge and the trip-wire latch as it stands (set under the lock at settlement)."""
        await self.journal.put(self.jkey(rid), {"charged": self.charged[rid], "open": None, "overrun": self.overrun})

    async def tick(self) -> None:
        async with self.lock:
            await asyncio.to_thread(self._commit_reload, True)  # every row so far, then the latest PAUSE state
            self.paused = self.pause_path.exists()
            now = time.time()
            self.tick_log.append((now, self.paused))
            bins = {}
            for t, kind in self.rl_events + self.end_events:
                end = datetime.fromtimestamp(math.ceil(t / 600) * 600, PT).isoformat(timespec="seconds")  # bin END, ISO with offset
                b = bins.setdefault(end, {"rate_limit_errors": 0, "runs_ended": 0})
                b["rate_limit_errors" if kind == "rl" else "runs_ended"] += 1
            status = {"status_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "status_written_pt": datetime.now(PT).isoformat(timespec="seconds"), "label": self.label,
                      "mode": self.mode, "rule_sha": RULE_SHA, "paused": self.paused, "overrun_stop": self.overrun,
                      "runs_total": len(self.runs), "runs_in_flight": len(self.in_flight_runs), "runs_done": len(self.done_runs),
                      "runs_pending": len(self.runs) - len(self.done_runs) - len(self.in_flight_runs),
                      "rate_limit_errors_last_10min": sum(1 for t, _ in self.rl_events if t > now - 600),
                      "rate_limit_errors_by_10min": dict(sorted(bins.items())), "spend_logged_usd": round(self.spent, 4),
                      "reserved_in_flight_usd": round(self.inflight, 2), "hard_stop_usd": self.hard_stop,
                      "finished": self.finished_pt is not None, "finished_pt": self.finished_pt}  # d9's census: a stopped pass is not stale
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(status, indent=1))
            tmp.replace(self.status_path)
            await asyncio.to_thread(self._commit_reload, False)

    async def control(self) -> None:
        while not self.finished:
            await asyncio.sleep(self.tick_s)
            await self.tick()

    async def write(self, ledger: Ledger, row: dict, commit: bool = False) -> None:
        async with self.lock:
            ledger.append(row)
            if commit:
                await asyncio.to_thread(self._commit_reload, False)

    async def claim(self, rid: str, s: int, res: float) -> str | None:
        """Wait out PAUSE, then, under the lock, either reserve and record the dispatch or say why not."""
        while True:
            while self.paused:
                await asyncio.sleep(0.5)
            async with self.lock:
                if self.paused:
                    continue
                if self.overrun:
                    return "overrun"
                if self.spent + self.inflight + res > self.hard_stop - self.cap_margin:
                    if self.inflight < 1e-9:  # settled spend alone leaves no room: the run stops
                        return "budget"
                    wait = True  # in-flight steps may settle below their reservations: re-check after them
                else:
                    wait = False
                    self.inflight += res
                if not wait:
                    self.max_committed_exposure = max(self.max_committed_exposure, self.spent + self.inflight)
                    self.dispatch_log.append((time.time(), rid, s))
                    return None
            await asyncio.sleep(0.5)

    # -- one run -----------------------------------------------------------------------------------
    @staticmethod
    def committed_entries(ledger: Ledger) -> list:
        """(j, text) in commit order: simulated outputs as returned, kept-real outputs as the agent saw them."""
        out = []
        for r in ledger.rows:
            if r["kind"] == "result" and r["status"] == "ok":
                out += list(zip(r["js"], r["outputs"]))
            elif r["kind"] == "kept_real":
                out += list(zip(r["js"], r["real_outputs"]))
        return out

    async def rebuild(self, run: dict, ledger: Ledger, old=None):
        clients = None
        if old is not None:
            clients = old["clients"]
            await old["sim"].close()
        state = next((r["router_state"] for r in reversed(ledger.rows) if r.get("router_state") is not None), None)
        st = self.build(run, clients=clients, router_state=None if clients else state)
        await st["sim"].setup([])
        st["session"] = st["sim"].open()
        for j, text in self.committed_entries(ledger):
            st["sim"].commit_output(await st["sim"].action_message(run["actions"][j], st["session"]), text, st["session"])
        return st

    def open_ledger(self, run: dict) -> Ledger:
        if run["id"] not in self.ledgers:
            self.ledgers[run["id"]] = Ledger(self.root / "ledger" / f"{run['id']}.jsonl",
                                             {"kind": "header", "run": run["id"], "rule_sha": RULE_SHA, "config": self.config,
                                              "eligible": len(run["actions"]), "c_r": run["c_r"], "K_r": run["K_r"]})
        return self.ledgers[run["id"]]

    async def run_one(self, run: dict, limit: int) -> None:
        ledger = self.open_ledger(run)
        rows = ledger.rows
        if any(r["kind"] in ("stop", "excluded") for r in rows) or any(r["kind"] == "done" and r["limit"] >= limit for r in rows):
            return
        st = await self.rebuild(run, ledger)
        if st["sim"].system_prompt() != run["stored_system_prompt"]:
            await self.write(ledger, {"kind": "excluded", "reason": "system prompt differs from the sample's stored prompt"}, commit=True)
            await st["sim"].close()
            return
        done_steps = {r["step"] for r in rows if (r["kind"] == "result" and r["status"] == "ok") or r["kind"] == "kept_real"}
        rejected_logged = {r["step"] for r in rows if r["kind"] == "rejected"}
        try:
            for s, step in enumerate(run["steps"]):
                if min(step) >= limit:
                    break
                eligible = [j for j in step if not run["rejected"][j]]
                if s not in rejected_logged and len(eligible) < len(step):  # Inspect-rejected calls: never sent, never committed
                    await self.write(ledger, {"kind": "rejected", "step": s, "js": [j for j in step if run["rejected"][j]]})
                if not eligible or s in done_steps:
                    continue
                outcome, st = await self.step(run, ledger, st, s, eligible)
                if outcome in ("rate_limit", "budget", "overrun"):
                    await self.write(ledger, {"kind": "stop", "reason": outcome, "step": s}, commit=True)
                    return
            await self.write(ledger, {"kind": "done", "limit": limit}, commit=True)
        finally:
            await st["sim"].close()

    async def keep_real(self, run, ledger, st, s, js):
        refusal = any(r.get("refusal") for r in ledger.rows if r.get("step") == s and r["kind"] == "result")
        await self.write(ledger, {"kind": "kept_real", "step": s, "js": js, "real_outputs": [run["real_outputs"][j] for j in js],
                                  "status": "kept_real_after_refusal" if refusal else "kept_real_after_failure"}, commit=True)
        st = await self.rebuild(run, ledger, st)
        if self.harness:
            await self.harness.after_commit(run, [(j, run["real_outputs"][j]) for j in js])
        return "kept_real", st

    async def step(self, run: dict, ledger: Ledger, st: dict, s: int, js: list):
        rid, n = run["id"], len(js)
        actions = [run["actions"][j] for j in js]
        srows = [r for r in ledger.rows if r.get("step") == s]
        attempt = 1 + max((r["attempt"] for r in srows if r["kind"] in ("start", "result")), default=-1)  # monotonic, orphans included
        failures = sum(r["kind"] == "result" and r["status"] in ("failed", "orphaned") for r in srows)
        # the per-call cap counts every backoff sleep and the whole wall time of every attempt (so Inspect's own transport
        # retries inside an attempt count too); it persists in the ledger
        used = sum(r["seconds"] for r in srows if r["kind"] == "backoff") + sum(r.get("seconds", 0) for r in srows if r["kind"] == "result")
        n_backoff = sum(r["kind"] == "backoff" for r in srows)
        while True:
            if failures >= 2:  # also after a crash between the second failure and its kept-real row
                return await self.keep_real(run, ledger, st, s, js)
            if used >= RL_CAP_S:
                return "rate_limit", st
            res = reserve_usd(n)
            why = await self.claim(rid, s, res)
            if why:
                return why, st
            from inspect_ai.model._model import init_sample_model_usage, sample_model_usage

            await self.journal_open(rid, s, attempt, res)  # durable before the request can be billed
            await self.write(ledger, {"kind": "start", "step": s, "attempt": attempt, "js": js})
            init_sample_model_usage()
            log = []
            ctx = {"run": rid, "step": s, "attempt": attempt, "n": n, "requests": log, "active": 0}
            CURRENT.set(ctx)
            before = st["counters"]()
            t0 = time.time()
            self.in_flight_runs.add(rid)
            if self.harness:
                await self.harness.before_step(run, s, js, attempt)
            err, outs = None, None
            try:  # the remaining per-call allowance bounds the attempt at the transport boundary
                outs = await asyncio.wait_for(asyncio.gather(*(st["sim"].simulate_concurrent(a, st["session"], expected=n) for a in actions)),
                                              timeout=max(1.0, RL_CAP_S - used))
            except Exception as e:  # noqa: BLE001  (classified and recorded below; never scored)
                err = e
            finally:
                me = asyncio.current_task()
                stray = [t for t in asyncio.all_tasks() if t is not me and not t.done() and t.get_context().get(CURRENT) is ctx]
                for t in stray:  # work a failed sibling left running (e.g. the separate-fabrication fallback) is cancelled
                    t.cancel()
                if stray:
                    await asyncio.gather(*stray, return_exceptions=True)
                self.in_flight_runs.discard(rid)
            seconds = time.time() - t0
            used += seconds
            usage = {m: u.model_dump(exclude_none=True) for m, u in sample_model_usage().items()}
            cost = (self.cost_fn(usage) if usage and self.cost_fn else 0.0) + self.fake_cost
            after = st["counters"]()
            base = {"kind": "result", "step": s, "attempt": attempt, "js": js, "cost_usd": round(cost, 6), "usage": usage,
                    "subs": {k: after[k] - before[k] for k in SUBS}, "requests": len(log), "seconds": round(seconds, 1),
                    "router_state": st["router_state"](), "reserved_usd": res}
            attempt += 1
            timed_out = isinstance(err, (TimeoutError, asyncio.TimeoutError))
            limited = err is not None and (rate_limited(err) or timed_out)
            # billing unknown (a cancelled request, a timeout): the whole reservation is charged
            if timed_out or stray:  # also after a successful return that left cancelled strays (Codex r4)
                charge = max(cost, res)
            elif err is None or limited:
                charge = cost
            else:
                charge = max(cost, FAIL_MIN_USD)
            overrun = f"{rid} step {s} cost ${cost:.2f} above its ${res:.0f} reservation" if cost > res else None
            async with self.lock:  # release the reservation, charge, and latch an overrun, in one step
                self.inflight -= res
                self.spent += charge
                self.charged[rid] = self.charged.get(rid, 0.0) + charge
                if overrun and not self.overrun:
                    self.overrun = overrun  # Step 2's trip-wire: no new dispatch anywhere until a human clears it
                if limited and not timed_out:
                    self.rl_events.append((time.time(), "rl"))
            await self.journal_close(rid)
            if err is None:
                staged = [run["truncate"](run["actions"][j]["function"], o) for j, o in zip(js, outs)]
                statuses = ["simulated_identical" if t == run["real_outputs"][j] else "simulated_changed" for j, t in zip(js, staged)]
                await self.write(ledger, base | {"status": "ok", "charge_usd": charge, "outputs": list(outs), "statuses": statuses})
                if self.harness:
                    await self.harness.after_commit(run, list(zip(js, outs)))
                return "ok", st
            if limited:
                await self.write(ledger, base | {"status": "rate_limited", "charge_usd": charge, "error": type(err).__name__, "time_cap": timed_out})
                if used >= RL_CAP_S:
                    return "rate_limit", st
                wait = min(RL_SCHEDULE[min(n_backoff, len(RL_SCHEDULE) - 1)], RL_CAP_S - used)
                used, n_backoff = used + wait, n_backoff + 1
                await self.write(ledger, {"kind": "backoff", "step": s, "seconds": wait, "total_s": round(used, 1)}, commit=True)
                await self.clock.sleep(wait)
                st = await self.rebuild(run, ledger, st)
                continue
            failures += 1
            refusal = any(q.get("stop_reason") == "content_filter" for q in log)
            await self.write(ledger, base | {"status": "failed", "charge_usd": charge, "error": type(err).__name__, "refusal": refusal}, commit=True)
            st = await self.rebuild(run, ledger, st)

    # -- the pass ----------------------------------------------------------------------------------
    async def run_all(self, limits: dict) -> dict:
        events = self.root / "pass_events.jsonl"
        events.parent.mkdir(parents=True, exist_ok=True)
        with open(events, "a") as fh:
            fh.write(json.dumps({"kind": "pass_start", "t_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
        self.spent = 0.0
        for run in self.runs:  # every ledger verified, every orphan charged, before any dispatch
            ledger = self.open_ledger(run)
            j = await self.journal.get(self.jkey(run["id"]), None) or {"charged": 0.0, "open": None, "overrun": None}
            if j.get("overrun") and not self.overrun:  # the trip-wire latch survives relaunches
                self.overrun = j["overrun"]
            in_ledger = sum(r["charge_usd"] for r in ledger.rows if r["kind"] == "result")
            results = {(r["step"], r["attempt"]) for r in ledger.rows if r["kind"] == "result"}
            starts = {(r["step"], r["attempt"]) for r in ledger.rows if r["kind"] == "start"}
            if j["open"] is not None:  # durable even if the start row was lost with the container
                starts.add((j["open"][0], j["open"][1]))
            orphan = 0.0
            for s, a in sorted(starts - results):
                from_journal = j["open"] is not None and (s, a) == (j["open"][0], j["open"][1])
                res = j["open"][2] if from_journal else FAIL_MIN_USD
                t_open = j["open"][3] if from_journal else next((datetime.fromisoformat(r["t_utc"]).timestamp() for r in ledger.rows
                                                                 if r["kind"] == "start" and (r["step"], r["attempt"]) == (s, a)), time.time() - RL_CAP_S)
                # its elapsed time counts against the step's 30-minute allowance (downtime included: conservative)
                ledger.append({"kind": "result", "step": s, "attempt": a, "status": "orphaned", "charge_usd": max(res, FAIL_MIN_USD),
                               "cost_usd": 0.0, "js": [], "seconds": round(min(RL_CAP_S, time.time() - t_open), 1),
                               "error": "start without result (resume)"})
                orphan += max(res, FAIL_MIN_USD)
            charged = max(in_ledger, j["charged"]) + orphan  # the journal also covers result rows lost before a commit
            self.charged[run["id"]] = charged
            self.spent += charged
            await self.journal.put(self.jkey(run["id"]), {"charged": charged, "open": None, "overrun": j.get("overrun")})
        await self.tick()  # status written and PAUSE read before the first dispatch
        sem = asyncio.Semaphore(self.concurrency)
        control = asyncio.create_task(self.control())
        crashes = []

        async def one(run):
            async with sem:
                while self.paused:  # no new run while PAUSE exists
                    await asyncio.sleep(0.5)
                for _ in range(3):  # a stub crash resumes from the ledger, like a relaunch
                    try:
                        await self.run_one(run, limits.get(run["id"], run["K_r"]))
                        break
                    except SimulatedCrash:
                        crashes.append(run["id"])
                self.done_runs.add(run["id"])
                self.end_events.append((time.time(), "end"))

        tasks = [asyncio.create_task(one(r)) for r in self.runs]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:  # no worker outlives an aborted pass
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.finished = True
            await control
            await self.commit_now()
            raise
        self.finished = True
        await control
        with open(events, "a") as fh:  # only a pass that finished every run
            fh.write(json.dumps({"kind": "pass_end", "t_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "spent_usd": round(self.spent, 4), "overrun": self.overrun}) + "\n")
        self.finished_pt = datetime.now(PT).isoformat(timespec="seconds")
        await self.tick()  # the final status: finished true, final counts and spend (committed)
        await self.commit_now()
        return {"spent_usd": round(self.spent, 4), "crash_resumes": crashes, "overrun": self.overrun,
                "max_committed_exposure_usd": round(self.max_committed_exposure, 4)}


def final_statuses(run: dict, rows: list) -> list:
    """One terminal status per eligible output (outputs outside the simulated range are left to the staging gate)."""
    st = ["beyond_limit"] * len(run["actions"])
    for j, rej in enumerate(run["rejected"]):
        if rej:
            st[j] = "kept_real_rejected"
    for r in rows:
        if r["kind"] == "result" and r["status"] == "ok":
            for j, s in zip(r["js"], r["statuses"]):
                st[j] = s
        elif r["kind"] == "kept_real":
            for j in r["js"]:
                st[j] = r["status"]
    stop = next((r for r in rows if r["kind"] == "stop"), None)
    if stop is not None:
        for s in range(stop["step"], len(run["steps"])):
            for j in run["steps"][s]:
                if not run["rejected"][j]:
                    st[j] = "not_processed_before_stop"
    return st
