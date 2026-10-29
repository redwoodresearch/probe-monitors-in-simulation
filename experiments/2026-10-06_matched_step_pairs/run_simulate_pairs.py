"""Ask the September Astra simulator for step k's output, given the real history before it. One call per pair.

Runs on the September snapshot of the sims code (~/simulated-control-evals-sept, built by the sims session: commit
ba990399 + the S3 archive + its 2-line patch), never on sims main. The simulator is wired exactly as the September
arena path wires it (runner.build.arena_simulated_task.simulated_tools): Astra as the primary, wrapped in the Sol
fallback with the Terra refusal judge, then the Terra network router; config from config_for_sample over the reference
cell's stored September metadata (reference trajectory pasted in at 32,768 characters per output, codebase index,
container path mappings), with two changes:
  - codebase_root points at the local copy of the same environment folder (the sims session reproduced the
    September fleet digest for 8 of 10 environments; media_processing and model_registry are not in the sample);
  - seeded_prefix is the pair's real prefix: every earlier call with the real output the agent saw. Step k's real
    output is never in it.
No Docker: the arena's scratch container only serves an in-run judge, and read tools serve the static folder.
Before simulating, the simulator's system prompt is compared with the reference sample's stored September prompt;
a mismatch fails the pair.

Each finished pair is appended to <out>.jsonl at once (resumable: done pairs are skipped), with the output, the
output as the reader sees it (Inspect's 16 KiB tool-output truncation, which every arm's logs carry), token usage and
cost per model, the fallback and router counters, and the simulator's whole memory. Failures are recorded and counted,
never scored. Keys come from the environment; run under op run, from a folder with no .env above it:

  cd $TMPDIR && op run --env-file=<repo>/.env.tpl -- uv run --project ~/simulated-control-evals-sept --no-sync \\
    python <repo>/experiments/2026-10-06_matched_step_pairs/run_simulate_pairs.py --pairs <data/pairs_*.jsonl> \\
    --out <data/sim_<stamp>> [--pilot] [--concurrency 16]
"""

import argparse
import asyncio
import contextvars
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SEPT = Path.home() / "simulated-control-evals-sept"
MODEL = "openai/gpt-6-astra"
MAX_TOOL_OUTPUT = 16 * 1024  # Inspect's default, as in every arm's logs
PT = ZoneInfo("America/Los_Angeles")


JUDGE_LOG: "contextvars.ContextVar[list]" = contextvars.ContextVar("judge_log")
JUDGE_MODELS = {"openai/gpt-5.6-terra"}  # the September refusal and router judges
# Output files written before rows recorded their build, with the build both of their launches logged (checked
# 2026-10-06 from the launch logs). A resume of one of these must still run under exactly this build.
LEGACY_BUILDS = {"sim_pilot_2026-10-06.jsonl": {
    "simulator_model": "openai/gpt-6-astra", "sims_commit": "ba990399bfbd71aebf7636c04252d22c05720156",
    "sims_src_diff_sha256": "7d2b6021d1301cc1b084441905bae9592d74cae2c201d808c322ae1763702db4"}}


def log_judges() -> None:
    """Record every judge prompt and response with its pair. The refusal judge (FallbackClient) and the router judge
    (RoutedNetworkClient) resolve their models through `runner.simulate.model.resolve`; for judge models it now
    returns a wrapper whose `generate` appends {model, prompt, response} to the calling task's JUDGE_LOG. Nothing
    else changes: same model, same arguments, same output."""
    import runner.simulate.model as model_module

    original = model_module.resolve

    class Logged:
        def __init__(self, model, name):
            self._model, self._name = model, name

        async def generate(self, input, *args, **kwargs):
            # The request is logged before it is sent, so a call that times out or raises still leaves its prompt and
            # error with the pair's (failed) row.
            entry = {"model": self._name, "prompt": [str(getattr(m, "content", m)) for m in input], "response": None}
            log = JUDGE_LOG.get(None)
            if log is not None:
                log.append(entry)
            try:
                output = await self._model.generate(input, *args, **kwargs)
            except BaseException as error:
                entry["error"] = f"{type(error).__name__}: {error}"[:2000]
                raise
            entry["response"] = output.completion
            return output

        def __getattr__(self, name):
            return getattr(self._model, name)

    def resolve(name, **kwargs):
        model = original(name, **kwargs)
        return Logged(model, name) if name in JUDGE_MODELS else model

    model_module.resolve = resolve


def tree_sha256(root: Path) -> str:
    """sha256 over every file under root: its relative path and its bytes, in sorted order."""
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(root)).encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def no_dotenv_above(path: Path) -> None:
    """Inspect can auto-load a .env from the working folder or its parents; this run's keys come only from op run."""
    for d in [path, *path.parents]:
        assert not (d / ".env").exists(), f"{d / '.env'} exists; run from a folder with no .env above it"


async def simulate(rec: dict, check_only: bool = False) -> dict:
    from inspect_ai.model._call_tools import truncate_tool_output

    from runner.simulate import build_simulator, config_for_sample
    from runner.simulate.fidelity import blocked_reachable_host, reachable_set
    from runner.simulate.model import FallbackClient, InspectClient, RoutedNetworkClient

    md = dict(rec["sim_metadata"])
    assert md["simulator_model"] == MODEL and md["orig_trajectory_out_cap"] == 32768 and md["orig_trajectory_mode"] == "dump"
    assert not md["codebase_exec"] and not md["session_notes"] and md["self_critique"] is None and not md["seeded_prefix"]
    assert {md["simulator_refusal_judge_model"], md["simulator_router_judge_model"]} <= JUDGE_MODELS, "unlogged judge model"
    md["codebase_root"] = str(SEPT / ".settings" / "linuxarena" / rec["environment"] / "codebase")
    assert Path(md["codebase_root"]).is_dir(), md["codebase_root"]
    md["seeded_prefix"] = rec["prefix"]
    assert all(a["id"] != rec["step"]["id"] for a in rec["prefix"]), "step k is in its own prefix"

    # runner.build.arena_simulated_task.simulated_tools, line for line.
    fallback = InspectClient(md["simulator_fallback_model"])
    cidrs, hosts = reachable_set(md["codebase_root"])
    guard = FallbackClient(InspectClient(MODEL), fallback, lambda text: blocked_reachable_host(text, cidrs, hosts),
                           refusal_judge_model=md["simulator_refusal_judge_model"])
    router = RoutedNetworkClient(guard, fallback, md["simulator_router_judge_model"])
    trusted = False
    config = config_for_sample(md, model=MODEL, codebase_exec=False, exec_source_authority=md["exec_source_authority"],
                               orig_trajectory_mode="dump", self_critique=None)
    simulator = build_simulator(config, fallback=fallback, client=router)
    prompt = simulator.system_prompt()
    prompt_match = prompt == rec["stored_system_prompt"]
    if not prompt_match:
        i = next((j for j, (a, b) in enumerate(zip(prompt, rec["stored_system_prompt"])) if a != b), min(len(prompt), len(rec["stored_system_prompt"])))
        raise AssertionError(f"system prompt differs from September's at char {i}: ours {prompt[i - 80:i + 80]!r} "
                             f"vs stored {rec['stored_system_prompt'][i - 80:i + 80]!r}")
    session = simulator.open()
    try:
        await simulator.setup(config.seeded_prefix)
    except AssertionError as e:
        if "violate the output contract" not in str(e):
            raise
        # A real prefix output the September contract rejects. trust_seeded_prefix only turns that audit into a
        # warning (sim.simulator.Simulator.setup); prompts and generation are unchanged. Recorded per pair.
        import attrs

        trusted = True
        config = attrs.evolve(config, trust_seeded_prefix=True)
        simulator = build_simulator(config, fallback=fallback, client=router)
        session = simulator.open()
        await simulator.setup(config.seeded_prefix)
    if check_only:  # no model call: the prompt matched and the prefix passed (or was trusted past) the contract audit
        await simulator.close()
        return {"system_prompt_matches_september": prompt_match, "trusted_prefix": trusted,
                "memory_messages": len(session.memory)}
    action = {"function": rec["step"]["function"], "arguments": rec["step"]["arguments"]}
    if "description" in rec["step"]:
        action["description"] = rec["step"]["description"]
    t0 = time.time()
    try:
        output = await simulator.simulate_action(action, session)
    finally:
        await simulator.close()
    truncated = truncate_tool_output(rec["step"]["function"], output, MAX_TOOL_OUTPUT)
    return {"output": output, "reader_output": truncated.output if truncated else output,
            "inspect_truncated": truncated is not None, "seconds": round(time.time() - t0, 1),
            "system_prompt_matches_september": prompt_match, "trusted_prefix": trusted,
            "fallback_substitutions": simulator.fallback_substitutions,
            "block_substitutions": guard.block_substitutions, "refusal_judge_calls": guard.refusal_judge_calls,
            "refusal_substitutions": guard.refusal_substitutions, "net_substitutions": router.net_substitutions,
            "router_judge_calls": router.judge_calls,
            "grounding_calls": [plain(g) for g in session.grounding_calls],
            "memory": [plain(m) for m in session.memory[1:]]}  # [0] is the system prompt, checked above


def plain(x):
    import attrs

    return json.loads(json.dumps(attrs.asdict(x) if attrs.has(type(x)) else x, default=repr))


def cost(usage: dict) -> float:
    from sim.prices import load_prices

    table = load_prices()
    total = 0.0
    for model, u in usage.items():
        cached = u.get("input_tokens_cache_read", 0) or 0
        total += table[model].usd(input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0),
                                  cache_read_tokens=cached, cache_write_tokens=u.get("input_tokens_cache_write", 0) or 0)
    return round(total, 4)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="output stem; writes <out>.jsonl and <out>.run.json")
    p.add_argument("--pilot", action="store_true", help="only the pilot pairs")
    p.add_argument("--only", default="", help="comma-separated pair numbers")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--budget-usd", type=float, default=140.0, help="stop starting new pairs past this spend")
    p.add_argument("--check-only", action="store_true", help="build every simulator and compare prompts; no model calls")
    args = p.parse_args()
    from inspect_ai.model._model import init_sample_model_usage, sample_model_usage

    no_dotenv_above(Path.cwd())
    assert args.check_only or os.environ.get("OPENAI_API_KEY"), "no OPENAI_API_KEY: run under op run"

    recs = [json.loads(l) for l in args.pairs.read_text().splitlines()]
    envs = sorted({r["environment"] for r in recs})  # every environment in the pairs file, whatever --pilot/--only pick
    if args.pilot:
        recs = [r for r in recs if r["pilot"]]
    if args.only:
        recs = [r for r in recs if r["pair"] in {int(x) for x in args.only.split(",")}]
    out = args.out.with_suffix(".jsonl")
    pairs_sha = hashlib.sha256(args.pairs.read_bytes()).hexdigest()
    sims = lambda *a: subprocess.run(["git", "-C", str(SEPT), *a], capture_output=True, text=True).stdout
    # What a resumed run must share with the rows already written: the simulator build and model, this script, and
    # the environment folders the read tools serve (every file's path and bytes).
    build = {"simulator_model": MODEL, "sims_commit": sims("rev-parse", "HEAD").strip(),
             "sims_src_diff_sha256": hashlib.sha256(sims("diff", "HEAD", "--", "src").encode()).hexdigest(),
             "sims_lock_sha256": hashlib.sha256((SEPT / "uv.lock").read_bytes()).hexdigest(),
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "codebase_sha256": {e: tree_sha256(SEPT / ".settings" / "linuxarena" / e / "codebase") for e in envs}}
    status_done = "checked" if args.check_only else "ok"
    done, spent, legacy_rows = {}, 0.0, False
    if out.exists():
        lines = out.read_text().splitlines()
        for n, line in enumerate(lines):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                assert n == len(lines) - 1, f"{out} line {n + 1} is corrupt"
                # An interrupted append: keep the partial bytes aside and cut the file back to its last whole record,
                # so new records do not land after the fragment.
                out.with_suffix(".partial").write_text(line)
                out.write_text("".join(l + "\n" for l in lines[:n]))
                print(f"moved a partial last line of {out} to {out.with_suffix('.partial')}", flush=True)
                continue
            assert r["pairs_sha256"] == pairs_sha, f"{out} holds results for another pairs file; use a new --out"
            if "build" in r:
                assert r["build"] == build, f"{out} was made by another simulator build or script; use a new --out"
            else:  # rows from before the field: only the verified 2026-10-06 run, and only under its build
                legacy = LEGACY_BUILDS.get(out.name)
                assert legacy is not None, f"{out} has rows with no recorded build; use a new --out"
                assert all(build[k] == v for k, v in legacy.items()), f"{out} was made by another simulator build"
                legacy_rows = True
            spent += r.get("cost_usd", 0.0)  # every attempt, failed ones included
            if r["status"] == status_done:
                done[r["pair"]] = r
    todo = [r for r in recs if r["pair"] not in done]
    # A legacy file never recorded its script or environment fingerprints, so nothing new may be added to it.
    assert not (legacy_rows and todo), f"{out} predates full build fingerprints; simulate new pairs under a new --out"
    run = {"started": datetime.now(PT).isoformat(timespec="seconds"), "pairs_file": args.pairs.name,
           "pairs_sha256": pairs_sha, "sims_tree": str(SEPT), **build,
           "python": sys.version, "n_requested": len(recs), "n_already_done": len(done), "concurrency": args.concurrency}
    print(json.dumps(run), flush=True)
    log_judges()
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    stats = {"ok": 0, "failed": 0, "skipped_budget": 0}

    async def one(rec):
        nonlocal spent
        async with sem:
            if spent >= args.budget_usd:
                stats["skipped_budget"] += 1
                return
            t0 = time.time()
            init_sample_model_usage()  # this task's own usage ledger, so a failed attempt's spend is still counted
            judge_log = []
            JUDGE_LOG.set(judge_log)
            try:
                res = await simulate(rec, args.check_only)
                row = {"pair": rec["pair"], "status": status_done, "result": res}
                stats["ok"] += 1
            except Exception as e:
                row = {"pair": rec["pair"], "status": "failed", "error": f"{type(e).__name__}: {e}"[:2000],
                       "traceback": traceback.format_exc()[-4000:]}
                stats["failed"] += 1
            usage = {m: u.model_dump(exclude_none=True) for m, u in sample_model_usage().items()}
            row |= {"pairs_sha256": pairs_sha, "build": build, "usage": usage, "cost_usd": cost(usage),
                    "judge_calls": judge_log}
            spent += row["cost_usd"]
            row["finished"] = datetime.now(PT).isoformat(timespec="seconds")
            row["wall_seconds"] = round(time.time() - t0, 1)
            async with lock:
                with open(out, "a") as f:
                    f.write(json.dumps(row) + "\n")
            print(f"pair {rec['pair']} {row['status']} {row['wall_seconds']}s spent ${spent:.2f} "
                  f"({stats['ok']} ok, {stats['failed']} failed)", flush=True)

    await asyncio.gather(*(one(r) for r in todo))
    run |= {"finished": datetime.now(PT).isoformat(timespec="seconds"), "spent_usd_total": round(spent, 2)} | stats
    # One record per launch, so a resumed run keeps every launch's settings.
    stamp = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
    args.out.with_name(f"{args.out.name}.run_{stamp}.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(run), flush=True)
    if stats["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
