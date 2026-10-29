"""Do the two formatting fixes move the paper's LLM trajectory monitor, as they move the tool-output probe?

Approved by Aswin via the orchestrator (2026-10-08): gpt-5.4-nano only, the paper's own settings, on q2's 100 swap
runs, two arms scored the same day:
- (a) as simulated: the simulator's recorded actions, as the paper's monitor reads them;
- (b) both fixes: the same actions with q2's combined rerender applied to their outputs (the insert review note,
  `diff_with_note`, and str_replace in the real format, `sr_gutter_no_newline`; config/swap_combined.yaml there).

Everything else is the sims repo's own code, read-only: the `.eval` reader (`sim.adapters.inspect_eval`), the monitor
(`metrics.monitor.trajectory`: frozen template sha 14ec7f…, whole trajectory, 0–10, k = 3) and the cached, cost-logged
client (`runner.metrics.clients.JudgeClient`) at the paper's medium reasoning effort and concurrency 64 (configs
`la_current_trajectory_nano_*.yml`). Completions are cached in `.run_cache/scoring_calls/` and every paid call is
logged in `.run_cache/paper/total_cost.jsonl` (both git-ignored). Run from the repo root.

  --check  $0: build both arms, check (b) against q2's stage report run by run, count tokens, price it.
  --archive-inputs  $0, after the run: write every trajectory's exact monitor prompt for both arms to
           data/monitor_inputs_<stamp>.jsonl.gz (git-ignored; model input text) after checking that each prompt keys
           the run's own k cached completions, so these are the inputs that were scored.
  --run    the paid run, stopped hard once the logged cost passes the cap (default $8, minus a concurrency margin).
           Both arms run at once, so a shared gate holds at most IN_FLIGHT calls in flight in total, and the stop
           leaves room for all of them at a generous per-call cost. The run name is fixed, so a restart after a stop
           counts the spend of every attempt. The first attempt refuses to start if the completion cache already holds
           entries (so (a) is fresh); `--resume` continues a stopped attempt from its own cached calls.

  op run --env-file=<file with only OPENAI_API_KEY> -- uv run --project ~/simulated-control-evals --frozen \
      --with tiktoken python experiments/2026-10-08_q3_swap_llm_monitor/run_monitor_swap.py --check
"""

import argparse
import asyncio
import collections
import hashlib
import json
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import attrs
import yaml

HERE = Path(__file__).parent
REPO = HERE.parents[1]
Q2 = REPO / "experiments/2026-10-07_q2_rendering_swap"
sys.path[:0] = [str(REPO / "src"), str(Q2)]
import rerender as rr  # noqa: E402
import str_replace_tells as st  # noqa: E402
from metrics.monitor import trajectory as tm  # noqa: E402
from metrics.provenance import corpus_id  # noqa: E402
from runner.metrics.clients import JudgeClient, ModelSettings  # noqa: E402
from sim.adapters.inspect_eval import transcripts  # noqa: E402
from sim.identity import Mode  # noqa: E402
from sim.prices import load_prices  # noqa: E402

MODEL, EFFORT, K, CONCURRENCY = "openai/gpt-5.4-nano", "medium", 3, 64
OPTIONS = {"scale": "0-10", "aggregation": "whole-trajectory", "k_rescores": K, "concurrency": CONCURRENCY}
FIX = ("diff_with_note", "sr_gutter_no_newline")
SIMS = next(p for p in (REPO.parent / "simulated-control-evals", Path.home() / "simulated-control-evals") if p.exists())  # sibling checkout
CACHE = Path.home() / ".cache/probe-shift/s3"
LEDGER = Path(".run_cache/paper/total_cost.jsonl")
IN_FLIGHT = 8  # shared across both arms; transport only (the paper's 64 is per run and changes no answer)
# The paper's client sets no output limit, and adding one would change the instrument. So the reserve assumes every
# admitted call runs to the model's maximum output (assumed 128k tokens: $0.160) on a 60k-token input ($0.012), which
# makes the cap hard: spend at admission <= cap - MARGIN, plus at most IN_FLIGHT calls of at most MAX_CALL_USD each.
MAX_CALL_USD = 0.172
MARGIN = IN_FLIGHT * MAX_CALL_USD  # $1.38: new calls stop at $6.62 under the $8 cap
RUN_NAME = "probe-shift_q3_swap_llm_monitor_2026-10-08"  # fixed, so spend across restarts adds up
SCORING_CACHE = Path(".run_cache/scoring_calls")


class BudgetStop(AssertionError):  # an AssertionError, so the monitor re-raises it instead of counting a failure
    pass


def load_a() -> dict:
    keys = json.loads((Q2 / "config/swap_runs.json").read_text())["runs"]
    meta = {(s["arm"], s["key"]): s for s in json.loads((REPO / "splits/exclusions_v4.json").read_text())["samples"]}
    uris = yaml.safe_load((SIMS / "src/paper/configs/real_runs.yml").read_text())["runs"]
    by_file, out = collections.defaultdict(list), {}
    for k in keys:
        m = meta[("sim-attack", k)]
        by_file[(m["cohort"], m["file"])].append((k, f"{m['sample_id']}#{m['epoch']}"))
    for (cohort, file), items in by_file.items():
        bucket, prefix = uris[cohort]["uri"].removeprefix("s3://").split("/", 1)
        ts = {corpus_id(t): t for t in transcripts(CACHE / bucket / f"{prefix}logs/{file}", simulated=True, mode=Mode.ATTACK)}
        out |= {k: ts[cid] for k, cid in items}
    return out


def fixed(t):
    """(b): q2's combined rerender on the recorded actions. The actions become reader-style messages (one assistant
    message per step, its tool outputs after it), the two fixes run in turn, and only changed outputs are replaced."""
    acts, steps, msgs, at = list(t.actions), {}, [], []
    for i, a in enumerate(acts):
        if "step" not in a:
            raise ValueError(f"{t.session_id}: action {i} has no step")
        steps.setdefault(a["step"], []).append(i)
    for ids in steps.values():
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": acts[i]["function"], "arguments": acts[i].get("arguments") or {}}} for i in ids]})
        for i in ids:
            msgs.append({"role": "tool", "content": acts[i].get("output") or ""})
            at.append((len(msgs) - 1, i))
    new, inserts = rr.rerender_messages(msgs, FIX[0])
    new, srs = st.swap_messages(new, FIX[1])
    changed = [i for m, i in at if new[m]["content"] != msgs[m]["content"]]
    for m, i in at:
        if i in changed:
            acts[i] = {**acts[i], "output": new[m]["content"]}
    return attrs.evolve(t, actions=tuple(acts)), inserts, srs, len(changed)


def check_against_q2(key, inserts, srs, n_changed, report):
    """the same fixes as q2's staged arm: per run, which inserts are re-rendered and which str_replaces change, in call
    order, and the number of changed outputs. An insert's category (exact vs approximate window) can differ with the
    file views in its history, but `diff_with_note` appends the note to the simulator's own diff whatever the category,
    so only whether it is re-rendered decides the text."""
    r = report[key]
    mine = ([x["category"] in rr.RENDERED for x in inserts], [x["changed"] for x in srs], n_changed)
    theirs = ([x["category"] in rr.RENDERED for x in r["inserts"]], [x["changed"] for x in r["str_replaces sim-attack-note-sr-both"]],
              len(r["changed_messages sim-attack-note-sr-both"]))
    return mine == theirs, {"mine": mine, "q2": theirs}


def spent(run_name: str) -> float:
    if not LEDGER.exists():
        return 0.0
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    return sum((r["run_cost"] if r["run_cost"] is not None else 0.02) for r in rows if r["run"] == run_name)


@attrs.define
class Guarded:
    inner: JudgeClient
    cap: float
    gate: asyncio.Semaphore

    @property
    def name(self) -> str:
        return self.inner.name

    async def complete(self, *args, **kwargs):
        async with self.gate:
            cost = spent(self.inner.settings.run_name)
            if cost > self.cap - MARGIN:
                raise BudgetStop(f"logged cost ${cost:.2f} passed the stop at ${self.cap - MARGIN:.2f} (cap ${self.cap})")
            return await self.inner.complete(*args, **kwargs)


def git(cwd) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cwd, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--archive-inputs", action="store_true")
    ap.add_argument("--cap", type=float, default=8.0)
    ap.add_argument("--resume", action="store_true", help="continue a stopped attempt from its own cached calls")
    args = ap.parse_args()
    if sum([args.check, args.run, args.archive_inputs]) != 1:
        raise SystemExit("pass exactly one of --check, --run or --archive-inputs")
    if Path.cwd().resolve() != REPO.resolve():
        raise SystemExit(f"run from the repo root ({REPO}): the cache and the cost ledger are cwd-relative")
    tm._template()  # asserts the frozen template sha
    report = json.loads((Q2 / "results/stage_report_swap_combined_2026-10-08_031201.json").read_text())["report"]
    a = load_a()
    b, checks, n_changed = {}, {}, {}
    for k, t in a.items():
        b[k], inserts, srs, n_changed[k] = fixed(t)
        checks[k] = check_against_q2(k, inserts, srs, n_changed[k], report)
    bad = {k: v[1] for k, v in checks.items() if not v[0]}
    print(f"(b) built for {len(b)} runs; {sum(n_changed.values())} outputs changed; runs differing from q2's stage report: {len(bad)}")
    if bad:
        print(json.dumps(dict(list(bad.items())[:3]), indent=1)[:3000])
        raise SystemExit("(b) does not reproduce q2's combined arm")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    meta = {"stamp": stamp, "model": MODEL, "reasoning_effort": EFFORT, "options": OPTIONS, "fix": FIX,
            "template_sha256": hashlib.sha256(tm._TEMPLATE_PATH.read_bytes()).hexdigest(), "git": git(REPO), "sims_git": git(SIMS),
            "runs": len(a), "changed_outputs": sum(n_changed.values()), "q2_stage_report_match": True}
    if args.archive_inputs:
        import gzip

        from runner.metrics import clients
        from sim.keys import canonical_json  # noqa: F401  (the cache key's serialization, used by FileCache.path)

        rows, missing = [], 0
        for arm, d in (("a_as_simulated", a), ("b_both_fixes", b)):
            for k, t in d.items():
                prompt = tm._prompt(t)
                identity = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "reasoning_effort": EFFORT,
                            "response_schema": tm.RESPONSE_SCHEMA}
                hits = [clients._CACHE.path({**identity, "sample_idx": i, "this_call_uuid": clients._CALL_UUID}).exists() for i in range(K)]
                missing += K - sum(hits)
                rows.append({"arm": arm, "swap_key": k, "corpus_id": corpus_id(t), "prompt": prompt,
                             "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
        if missing:
            raise SystemExit(f"{missing} of {len(rows) * K} cached completions not found for these prompts: not the scored inputs")
        out = REPO / "data" / f"monitor_inputs_{stamp}.jsonl.gz"
        out.parent.mkdir(exist_ok=True)
        with gzip.open(out, "xt") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"{len(rows)} prompts, every one keyed to its {K} cached completions -> {out}")
        print("sha256", hashlib.sha256(out.read_bytes()).hexdigest())
        return
    if args.check:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        tok = {arm: [len(enc.encode(tm._prompt(t))) for t in d.values()] for arm, d in (("a", a), ("b", b))}
        p = load_prices()[MODEL]
        est = {out: sum(K * p.usd(input_tokens=x, output_tokens=out) for v in tok.values() for x in v) for out in (1000, 2000, 4000)}
        print(json.dumps(meta | {"input_tokens_mean": {arm: sum(v) / len(v) for arm, v in tok.items()},
                                 "estimate_usd_by_output_tokens_per_call": est}, indent=1))
        return
    run_name = RUN_NAME
    cached_before = len(list(SCORING_CACHE.glob("*.json"))) if SCORING_CACHE.exists() else 0
    if cached_before and not args.resume:
        raise SystemExit(f"{SCORING_CACHE} already holds {cached_before} completions: (a) would not be fresh")
    if spent(run_name) > args.cap - MARGIN:
        raise SystemExit(f"{run_name} has already logged ${spent(run_name):.2f}")
    t0 = datetime.now(ZoneInfo("America/Los_Angeles"))
    print("start", t0.isoformat(), run_name, f"cached before {cached_before}", flush=True)

    async def both():
        judge = Guarded(JudgeClient(model=MODEL, settings=ModelSettings(reasoning_effort=EFFORT, run_name=run_name)), args.cap,
                        asyncio.Semaphore(IN_FLIGHT))
        settings = types.SimpleNamespace(options=OPTIONS, judge=judge, reasoning_effort=EFFORT, seed=0)
        return await asyncio.gather(tm.run_metric({"subject": tuple(a.values())}, settings),
                                    tm.run_metric({"subject": tuple(b.values())}, settings), return_exceptions=True)

    ra, rb = asyncio.run(both())
    t1 = datetime.now(ZoneInfo("America/Los_Angeles"))
    errors = {arm: f"{type(r).__name__}: {r}" for arm, r in (("a", ra), ("b", rb)) if isinstance(r, BaseException)}
    if errors:  # a stop or a crash: record it (no model text), then fail loudly; completed calls stay cached for --resume
        (HERE / "results" / "partial").mkdir(parents=True, exist_ok=True)
        with (HERE / "results" / "partial" / f"monitor_stopped_{stamp}.json").open("x") as f:
            f.write(json.dumps(meta | {"run_name": run_name, "start": t0.isoformat(), "stop": t1.isoformat(),
                                       "cost_usd_logged": spent(run_name), "errors": errors}, indent=1, default=str))
        raise SystemExit(f"stopped: {errors}; logged cost ${spent(run_name):.2f}")
    key_of = {corpus_id(t): k for k, t in a.items()}
    out = meta | {"run_name": run_name, "start": t0.isoformat(), "stop": t1.isoformat(), "cost_usd_logged": spent(run_name),
                  "cached_completions_before": cached_before, "in_flight": IN_FLIGHT,
                  "arms": {arm: {"scores": r.payload["scores"], "dropped": r.payload["dropped"], "attrition": r.payload["attrition"],
                                 "prompt_sha256": r.prompt_sha256, "summary": r.summary} for arm, r in (("a_as_simulated", ra), ("b_both_fixes", rb))},
                  "swap_key_of_corpus_id": {cid: key_of.get(cid) for cid in ra.payload["scores"]}}
    path = HERE / "results" / f"monitor_scores_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1, default=str))
    print("stop", t1.isoformat(), f"logged cost ${out['cost_usd_logged']:.2f}",
          {arm: (len(v["scores"]), len(v["dropped"])) for arm, v in out["arms"].items()}, "->", path)


if __name__ == "__main__":
    main()
