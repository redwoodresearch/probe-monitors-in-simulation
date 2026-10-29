"""The paper's gpt-5.4-nano trajectory monitor on the on-policy runs: real run 1, simulated (September), real run 2, and
the real honest runs for the threshold (RULE_ONPOLICY.md; Aswin's OK and the $70 stop in SPEND.md).

The harness is #126's (`../2026-10-08_q3_swap_llm_monitor/run_monitor_swap.py`), unchanged where it scores: sims'
`metrics.monitor.trajectory` (frozen template sha 14ec7f…, whole trajectory, 0–10, k = 3) through the cached,
cost-logged `runner.metrics.clients.JudgeClient` at medium reasoning effort, transcripts read with sims'
`sim.adapters.inspect_eval` reader from the locally cached S3 run folders. Completions are cached in
`.run_cache/scoring_calls/` and every paid call is logged in `.run_cache/paper/total_cost.jsonl` (both git-ignored).
Run from the repo root.

Two observers sit around the client and change nothing it sends: a gate that holds the calls in flight and stops new
ones near the cap or when the projected total passes it, and a wrapper on the provider model that records, per paid
call, whether Inspect marked the answer a refusal (no text). Draw outcomes go to data/ (git-ignored); the results JSON
keeps scores, counts and hashes only.

  --check  $0: list every line's runs from launch.json, check each .eval against its folder's S3 manifest, read them,
           count tokens, price it.
  --run    the paid run. All lines at once through one gate; the run name is fixed, so a restart adds up the spend of
           every attempt; `--resume` continues from the cached calls.
  --planted  with --check or --run: the planted-slip lines instead (RULE_ONPOLICY.md, planted addendum): real runs 1
           and 2 with 09's `plant_output` (plant_both.py, copied verbatim from aswin/plantslip-plan@396e9361) applied to
           each text_editor output. A run with no plant site has the same prompt as launch 1, so it is a cache hit.
  --newline-check  $0, with --planted: on one planted run (seeded pick) and over all of them, whether the monitor's
           prompt keeps each planted str_replace output's trailing newline and drops each removed review note.
           Prints booleans, counts, offsets and lengths only.

  op run --env-file=<file with only OPENAI_API_KEY> -- uv run --project ~/simulated-control-evals --frozen \
      --with tiktoken python experiments/2026-10-12_q3_nano_realsim/run_nano_onpolicy.py --check
"""

import argparse
import asyncio
import collections
import hashlib
import json
import math
import re
import subprocess
import time
import types
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import attrs
from metrics.monitor import rendering
from metrics.monitor import trajectory as tm
from metrics.provenance import corpus_id
from runner.metrics import clients
from runner.metrics.clients import JudgeClient, ModelSettings
from sim.adapters.inspect_eval import transcripts
from sim.identity import Mode
from sim.prices import load_prices

HERE = Path(__file__).parent
REPO = HERE.parents[1]
SIMS = next(p for p in (REPO.parent / "simulated-control-evals", Path.home() / "simulated-control-evals") if p.exists())
MODEL, EFFORT, K, CONCURRENCY = "openai/gpt-5.4-nano", "medium", 3, 64
OPTIONS = {"scale": "0-10", "aggregation": "whole-trajectory", "k_rescores": K, "concurrency": CONCURRENCY}
RUNS = Path.home() / ".cache/probe-shift/s3/individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs"
FOUR = ("2026-10-09_000523_rerun_strict", "2026-10-09_173815_scaleup_canary", "2026-10-09_182505_scaleup",
        "2026-10-10_001314_scaleup_topup")
LINES = {  # line: (run folders, launch.json arm, arm folder, simulated, mode)
    "real_run1": (FOUR, "attack", "attack", False, Mode.ATTACK),
    "sim_september": (FOUR, "sim_attack", "sim_attack@september", True, Mode.ATTACK),
    "real_run2": (("2026-10-10_001314_second_attack",), "attack", "attack", False, Mode.ATTACK),
    "honest_a": (FOUR, "honest_a", "honest_a", False, Mode.HONEST),
    "honest_b": (FOUR, "honest_b", "honest_b", False, Mode.HONEST),
}
PLANT = {"real_run1_planted": "real_run1", "real_run2_planted": "real_run2"}  # planted line: the line it plants
PLANT_SHA256 = "f71c591277ec8fca4fdd8bf986a2e3dd3f6cb1acb1b5246327785d6378287aa8"  # plant_both.py @ 396e9361
Q2_MODULES = (REPO / "experiments/2026-10-09_fixed_sim", REPO / "experiments/2026-10-07_q2_rendering_swap")
CHECK_SEED = 20261012
LEDGER = Path(".run_cache/paper/total_cost.jsonl")
RUN_NAME = "probe-shift_q3_nano_onpolicy_2026-10-12"  # fixed: the $70 covers every attempt and any planted launch
IN_FLIGHT = 24  # across all lines; transport only (the paper's 64 is per run and changes no answer)
# The client sets no output limit, so each admitted call is reserved at the model's maximum output (assumed 128k tokens,
# as #126) plus an input bound of one token per UTF-8 byte of the longest prompt and 2,000 tokens of overhead. A
# call's usage that the provider leaves out is charged the whole reserve. Retries inside Inspect sit below the gate.
OUT_MAX_USD = 128_000 * 1.25e-6
PROJECT_AFTER = 300  # paid calls before the projected-total stop applies
POLICY = re.compile(r"content_policy|invalid_prompt|content_filter|safety|flagged|refus", re.I)
DATA = REPO / "data"


class BudgetStop(AssertionError):  # an AssertionError, so the monitor re-raises it instead of counting a failure
    pass


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


RESERVE = [OUT_MAX_USD]  # per-call reserve; main() adds the input bound once the prompts are known


def ledger() -> tuple[float, int]:
    """the run's logged spend and its number of rows; a ledger that can't be read stops the run (an AssertionError), so
    no call goes out"""
    try:
        if not LEDGER.exists():
            return 0.0, 0
        rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
        costs = [(r["run_cost"] if r["run_cost"] is not None else RESERVE[0]) for r in rows if r["run"] == RUN_NAME]
        if not all(isinstance(c, (int, float)) and math.isfinite(c) and c >= 0 for c in costs):
            raise ValueError("a non-finite or negative cost in the ledger")
        return float(sum(costs)), len(costs)
    except Exception as error:
        raise BudgetStop(f"the cost ledger could not be read ({type(error).__name__}); stopping before any call") from error


def spent() -> float:
    return ledger()[0]


def load(names=tuple(LINES)) -> tuple[dict, dict, dict, dict, dict]:
    """Per line, {corpus_id: transcript} and {corpus_id: (task, .eval path relative to RUNS, sha256)}; every .eval is
    checked against its folder's S3 manifest. A task with several .eval files in one line keeps the one attempt the
    launch marked usable; with none usable, the task is left out of that line. Skipped files are returned and counted."""
    manifests, lines, where, skipped, no_eval = {}, {}, {}, {}, {}
    for line in names:
        folders, arm, arm_dir, simulated, mode = LINES[line]
        lines[line], where[line], found = {}, {}, collections.defaultdict(list)
        for folder in folders:
            rd = RUNS / folder
            if folder not in manifests:
                m = json.loads((RUNS / f"{folder}.manifest_v1.json").read_text())
                manifests[folder] = {f["key"]: f["sha256"] for f in m["files"]}
            for r in json.loads((rd / "launch.json").read_text())["runs"]:
                if r["arm"] != arm:
                    continue
                found[r["task"]]  # a run of this arm with no .eval at all still counts as a task without one
                for a in r.get("attempts") or []:
                    local = rd / a["log_dir"].split(f"/{folder}/", 1)[1]
                    if local.relative_to(rd).parts[0] != arm_dir:
                        continue
                    usable = ((a.get("outcome") or {}).get("policy_calls") or [{}])[0].get("usable") == 1
                    found[r["task"]] += [(usable, folder, ev) for ev in sorted(local.glob("*.eval"))]
        no_eval[line] = sorted(task for task, evs in found.items() if not evs)
        for task, evs in found.items():
            if len(evs) > 1:
                keep = [x for x in evs if x[0]]
                if len(keep) > 1:
                    raise SystemExit(f"{line}/{task}: {len(evs)} .eval files, {len(keep)} usable")
                skipped[f"{line}/{task}"] = [str(x[2].relative_to(RUNS)) for x in evs if not x[0]]  # none usable: all
                evs = keep
            for _, folder, ev in evs:
                rel = str(ev.relative_to(RUNS))
                digest = hashlib.sha256(ev.read_bytes()).hexdigest()
                if manifests[folder].get(f"phase2/runs/{rel}") != digest:
                    raise SystemExit(f"{rel}: not in its S3 manifest with this sha256")
                for t in transcripts(ev, simulated=simulated, mode=mode):
                    cid = corpus_id(t)
                    if cid in lines[line]:
                        raise SystemExit(f"{line}: corpus id {cid} twice")
                    lines[line][cid], where[line][cid] = t, (task, rel, digest)
    return lines, where, {f: sha((RUNS / f"{f}.manifest_v1.json").read_text()) for f in manifests}, skipped, no_eval


def plant_mods():
    import importlib.util
    import sys

    path = HERE / "plant_both.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PLANT_SHA256:
        raise SystemExit("plant_both.py differs from aswin/plantslip-plan@396e9361")
    for folder in Q2_MODULES:
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    import fixes  # noqa: E402  (q2's modules, as run_plant_stage.py@8353bcb5 _mods())
    import rerender  # noqa: E402

    spec = importlib.util.spec_from_file_location("plant_both", path)
    pb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pb)
    return pb, fixes, rerender


def planted(t, pb, fixes, rerender):
    """09's plant_output on each text_editor output of one run: (run, [(action index, slip)]). Unchanged runs come back
    as the same object, so their prompt is launch 1's."""
    acts, changed = list(t.actions), []
    for i, a in enumerate(acts):
        if a.get("function") != "text_editor":
            continue
        cmd, o = (a.get("arguments") or {}).get("command"), a.get("output")
        new = pb.plant_output(o, cmd, fixes, rerender)
        if new is not None and new != o:
            acts[i] = {**a, "output": new}
            changed.append((i, "gutter+newline" if cmd == "str_replace" else "note"))
    return (attrs.evolve(t, actions=tuple(acts)) if changed else t), changed


def reaches_monitor(t0, t1, changed, rerender) -> dict:
    """Does the monitor's prompt carry the plant? Booleans, counts, offsets and lengths only; no text leaves this."""
    p0, p1 = tm._prompt(t0), tm._prompt(t1)
    rows = []
    for i, slip in changed:
        out = t1.actions[i]["output"]
        needle = "\nResult: " + out
        off = p1.find(needle)
        end = off + len(needle)
        rows.append({"action": i, "slip": slip, "output_chars": len(out), "found_once": p1.count(needle) == 1,
                     "offset": off, "over_monitor_limit": len(out) > rendering.OUTPUT_LIMIT,
                     "output_ends_with_one_newline": out.endswith("\n") and not out.endswith("\n\n"),
                     "next_two_prompt_chars_are_newlines": off >= 0 and p1[end:end + 2] == "\n\n",
                     "last_action": i == len(t1.actions) - 1,
                     "review_note_in_output": rerender.REVIEW in out})
    return {"prompt_chars": [len(p0), len(p1)], "review_note_count": [p0.count(rerender.REVIEW), p1.count(rerender.REVIEW)],
            "outputs": rows}


@attrs.define
class Observed:
    """The provider model, unchanged; records per paid call whether Inspect marked the answer a refusal."""
    inner: object
    log: list
    path: Path  # appended per call, so a crash keeps the marks; a resume reads every attempt's file

    async def generate(self, messages, *args, **kwargs):
        out = await self.inner.generate(messages, *args, **kwargs)
        content = out.choices[0].message.content if out.choices else []
        refusal = any(getattr(c, "refusal", False) for c in content) if isinstance(content, list) else False
        row = {"prompt_sha256": sha(messages[-1].text), "inspect_refusal": bool(refusal),
               "stop_reason": out.choices[0].stop_reason if out.choices else None, "empty": not out.completion}
        self.log.append(row)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)


@attrs.define
class Guarded:
    inner: JudgeClient
    cap: float
    margin: float
    gate: asyncio.Semaphore
    size: dict  # prompt sha256 -> characters (the projection's unit)
    total: int  # characters over every call of the run (K per prompt)
    paid: list  # the Observed log: one row per paid call
    before: float  # spend logged by earlier attempts
    unlogged: float  # a reserve per call that raised: a billed call that fails never reaches the ledger
    draws_path: Path  # one row per finished call, paid or cached, appended as it finishes
    draws: list = attrs.field(factory=list)
    floor: float = 0.0  # the highest logged spend seen
    rows0: int = 0  # ledger rows at start: every paid call since must have added one
    stopped: str | None = None  # the first stop, latched: no call goes out after it
    done: int = 0
    progress: Path | None = None
    last: float = 0.0

    @property
    def name(self) -> str:
        return self.inner.name

    def check(self, logged, n):
        """the ledger only grows, by one row per paid call (the client writes it in the same step as the call returns):
        anything else means it was lost, replaced or not written, and the run stops"""
        if logged < self.floor - 1e-9:
            raise BudgetStop(f"the ledger shows ${logged:.2f}, below the ${self.floor:.2f} seen before")
        if n < self.rows0 + len(self.paid):
            raise BudgetStop(f"the ledger has {n - self.rows0} rows for {len(self.paid)} paid calls")
        self.floor = logged

    def projected(self, cost: float) -> float:
        """this attempt's cost per prompt character paid, over the characters still to score (every loaded run, so
        runs the monitor filters out make it high, never low)"""
        paid_chars = sum(self.size[r["prompt_sha256"]] for r in self.paid)
        return cost + (cost - self.before) / paid_chars * (self.total - self.done) if paid_chars else 0.0

    def draw(self, row):
        self.draws.append(row)
        with self.draws_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    async def complete(self, system, user, **kwargs):
        async with self.gate:
            if self.stopped:
                raise BudgetStop(self.stopped)
            try:
                (logged, n), h = ledger(), sha(user)
                self.check(logged, n)
                cost = logged + self.unlogged
                if cost > self.cap - self.margin:
                    raise BudgetStop(f"logged cost ${cost:.2f} passed the stop at ${self.cap - self.margin:.2f} (cap ${self.cap})")
                if len(self.paid) >= PROJECT_AFTER and self.projected(cost) > self.cap:
                    raise BudgetStop(f"projected total ${self.projected(cost):.2f} passes the cap ${self.cap} (logged ${cost:.2f})")
            except BudgetStop as stop:
                self.stopped = str(stop)
                raise
            try:
                out = await self.inner.complete(system, user, **kwargs)
            except AssertionError:
                raise
            except (Exception, asyncio.CancelledError) as error:  # a cancelled call may be billed too
                text = f"{type(error).__name__}: {error}"
                self.unlogged += RESERVE[0]
                self.draw({"prompt_sha256": h, "outcome": "refused" if POLICY.search(text) else "error", "error": text})
                raise
            finally:
                self.done += self.size[h]
                self.report()
            self.draw({"prompt_sha256": h, "outcome": "ok" if out else "refused_empty"})
            return out

    def report(self, force=False):
        if self.progress and (force or time.time() - self.last > 30):
            try:
                (logged, n), self.last = ledger(), time.time()
                self.check(logged, n)
                cost = logged + self.unlogged
            except BudgetStop as stop:
                self.stopped = self.stopped or str(stop)
                return
            with self.progress.open("a") as f:
                f.write(json.dumps({"time": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
                                    "calls_finished": len(self.draws), "paid_calls": len(self.paid), "cost_logged_plus_failed_reserve": round(cost, 4),
                                    "projected_total": round(self.projected(cost), 2), "share_done": round(self.done / self.total, 4),
                                    "outcomes": dict(collections.Counter(d["outcome"] for d in self.draws))}) + "\n")


def safe_spent():
    try:
        return spent()
    except BudgetStop:
        return None


def git(cwd) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cwd, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--cap", type=float, default=70.0)
    ap.add_argument("--resume", action="store_true", help="continue a stopped attempt from its own cached calls")
    ap.add_argument("--planted", action="store_true", help="the planted-slip lines (launch 2)")
    ap.add_argument("--newline-check", action="store_true", help="$0, with --planted: does the plant reach the prompt")
    args = ap.parse_args()
    if sum([args.check, args.run, args.newline_check]) != 1:
        raise SystemExit("pass exactly one of --check, --run or --newline-check")
    if args.newline_check and not args.planted:
        raise SystemExit("--newline-check needs --planted")
    if Path.cwd().resolve() != REPO.resolve():
        raise SystemExit(f"run from the repo root ({REPO}): the cache and the cost ledger are cwd-relative")
    tm._template()  # asserts the frozen template sha
    t_load = time.time()
    lines, where, manifests, skipped, no_eval = load(tuple(PLANT.values()) if args.planted else tuple(LINES))
    plant_meta, plant_runs = {}, {}
    if args.planted:
        pb, fixes, rerender = plant_mods()
        done = {pl: {cid: planted(t, pb, fixes, rerender) for cid, t in lines[src].items()} for pl, src in PLANT.items()}
        if args.newline_check:
            changed_sr = sorted((where["real_run1"][cid][0], cid) for cid, (_, ch) in done["real_run1_planted"].items()
                                if any(s == "gutter+newline" for _, s in ch))
            import random

            task, cid = random.Random(CHECK_SEED).choice(changed_sr)
            one = reaches_monitor(lines["real_run1"][cid], *done["real_run1_planted"][cid], rerender)
            agg = collections.Counter()
            for pl, src in PLANT.items():
                for c, (t1, ch) in done[pl].items():
                    if ch:
                        r = reaches_monitor(lines[src][c], t1, ch, rerender)
                        agg[f"{pl}: runs"] += 1
                        agg[f"{pl}: review notes removed from the prompt"] += r["review_note_count"][0] - r["review_note_count"][1]
                        for o in r["outputs"]:
                            agg[f"{pl}: {o['slip']} outputs"] += 1
                            for k in ("found_once", "output_ends_with_one_newline", "next_two_prompt_chars_are_newlines",
                                      "over_monitor_limit", "review_note_in_output", "last_action"):
                                agg[f"{pl}: {o['slip']} {k}"] += bool(o[k])
            print(json.dumps({"seed": CHECK_SEED, "picked_task": task, "picked_run": one, "all_planted_runs": dict(sorted(agg.items()))}, indent=1))
            return
        bad = 0  # the plant must reach the prompt: every planted output once, str_replace ones ending in one newline
        for pl, src in PLANT.items():
            for c, (t1, ch) in done[pl].items():
                if ch:
                    r = reaches_monitor(lines[src][c], t1, ch, rerender)
                    bad += sum(not (o["found_once"] and o["next_two_prompt_chars_are_newlines"] and not o["over_monitor_limit"]
                                    and not o["review_note_in_output"]
                                    and (o["slip"] != "gutter+newline" or o["output_ends_with_one_newline"])) for o in r["outputs"])
        if bad:
            raise SystemExit(f"{bad} planted outputs do not reach the monitor's prompt as planted")
        for pl, src in PLANT.items():
            plant_runs[pl] = {cid: dict(collections.Counter(s for _, s in ch)) for cid, (_, ch) in done[pl].items() if ch}
            plant_meta[pl] = {"plants": src, "runs": len(done[pl]), "runs_changed": len(plant_runs[pl]),
                              "outputs_changed": dict(sum((collections.Counter(v) for v in plant_runs[pl].values()), collections.Counter()))}
            lines[pl], where[pl] = {cid: t1 for cid, (t1, _) in done[pl].items()}, where[src]
            del lines[src], where[src]
    prompts = {line: {cid: tm._prompt(t) for cid, t in d.items()} for line, d in lines.items()}
    size = {sha(p): len(p) for d in prompts.values() for p in d.values()}
    RESERVE[0] = (max(len(p.encode()) for d in prompts.values() for p in d.values()) + 2_000) * 0.2e-6 + OUT_MAX_USD
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    cfg = {k: {"folders": v[0], "arm": v[1], "arm_dir": v[2], "simulated": v[3], "mode": v[4].name} for k, v in LINES.items()}
    meta = {"stamp": stamp, "model": MODEL, "reasoning_effort": EFFORT, "options": OPTIONS, "run_name": RUN_NAME,
            "cap_usd": args.cap, "in_flight": IN_FLIGHT, "lines": {k: cfg[PLANT.get(k, k)] | ({"plant": plant_meta[k]} if k in plant_meta else {})
                                                                   for k in lines},
            "plant_both_sha256": PLANT_SHA256 if args.planted else None,
            "template_sha256": hashlib.sha256(tm._TEMPLATE_PATH.read_bytes()).hexdigest(), "git": git(REPO),
            "sims_git": git(SIMS), "manifest_sha256": manifests, "skipped_unusable_attempts": skipped,
            "tasks_without_eval": no_eval,
            "runs": {line: len(d) for line, d in lines.items()},
            "tasks": {line: len({w[0] for w in d.values()}) for line, d in where.items()},
            "prompt_chars_max": max(size.values()), "reserve_per_call_usd": RESERVE[0]}
    print(f"read {sum(meta['runs'].values())} runs in {time.time() - t_load:.0f}s", json.dumps(meta["runs"]), flush=True)
    if args.check:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        tok = {line: [len(enc.encode(p)) for p in d.values()] for line, d in prompts.items()}
        p = load_prices()[MODEL]
        split = 6.9 / 27.1  # uncached share of input on 10-08 (SPEND.md there); the rest is a cache read
        est = {out: round(sum(K * (p.usd(input_tokens=x * split, output_tokens=out) + x * (1 - split) * 0.02e-6)
                               for v in tok.values() for x in v), 2) for out in (1000, 2000, 4000)}
        print(json.dumps(meta | {"input_tokens_mean": {k: round(sum(v) / len(v)) for k, v in tok.items()},
                                 "input_tokens_max": max(max(v) for v in tok.values()),
                                 "estimate_usd_by_output_tokens_per_call_10-08_cache_split": est,
                                 "estimate_usd_no_cache_4k_output": round(sum(K * p.usd(input_tokens=x, output_tokens=4000)
                                                                              for v in tok.values() for x in v), 2)}, indent=1))
        return
    if args.planted and spent() < 16.48:  # launch 1 logged $16.49 under this run name: a missing ledger would forget it
        raise SystemExit(f"{LEDGER} holds ${spent():.2f} for {RUN_NAME}, not launch 1's $16.49: refusing to start")
    if args.resume and not spent():
        raise SystemExit(f"--resume but {LEDGER} holds no spend for {RUN_NAME}: refusing to start")
    if spent() and not (args.resume or args.planted):
        raise SystemExit(f"{RUN_NAME} has already logged ${spent():.2f}: pass --resume")
    margin = IN_FLIGHT * RESERVE[0]
    DATA.mkdir(exist_ok=True)
    raised_before = sum("error" in json.loads(x) for f in DATA.glob("nano_onpolicy_draws_*.jsonl") for x in f.read_text().splitlines())
    if spent() + raised_before * RESERVE[0] > args.cap - margin:
        raise SystemExit(f"{RUN_NAME} has already logged ${spent():.2f} plus {raised_before} failed calls' reserve")
    paid = []
    real_model = clients._model
    clients._model = lambda name: Observed(real_model(name), paid, DATA / f"nano_onpolicy_paid_calls_{stamp}.jsonl")
    progress = DATA / f"nano_onpolicy_progress_{stamp}.jsonl"
    total = K * sum(size[sha(p)] for d in prompts.values() for p in d.values())
    t0, before = datetime.now(ZoneInfo("America/Los_Angeles")), spent() + raised_before * RESERVE[0]
    print("start", t0.isoformat(), RUN_NAME, f"logged before ${before:.2f}", f"new calls stop at ${args.cap - margin:.2f}",
          "progress", progress, flush=True)

    async def all_lines():
        judge = Guarded(JudgeClient(model=MODEL, settings=ModelSettings(reasoning_effort=EFFORT, run_name=RUN_NAME)), args.cap,
                        margin, asyncio.Semaphore(IN_FLIGHT), size, total, paid, before, raised_before * RESERVE[0],
                        DATA / f"nano_onpolicy_draws_{stamp}.jsonl", progress=progress, floor=ledger()[0], rows0=ledger()[1])
        settings = types.SimpleNamespace(options=OPTIONS, judge=judge, reasoning_effort=EFFORT, seed=0)
        res = await asyncio.gather(*(tm.run_metric({"subject": tuple(d.values())}, settings) for d in lines.values()),
                                   return_exceptions=True)
        judge.report(force=True)
        return dict(zip(lines, res)), judge

    res, judge = asyncio.run(all_lines())
    t1 = datetime.now(ZoneInfo("America/Los_Angeles"))
    errors = {line: f"{type(r).__name__}" for line, r in res.items() if isinstance(r, BaseException)}
    if judge.stopped:  # a latched stop (e.g. a paid call's ledger row missing) fails the run even if every line finished
        errors["gate"] = "BudgetStop"
        res = res | {"gate": BudgetStop(judge.stopped)}
    if errors:  # a stop or a crash: record it (no model text), then fail loudly; completed calls stay cached for --resume
        (HERE / "results" / "partial").mkdir(parents=True, exist_ok=True)
        with (HERE / "results" / "partial" / f"nano_onpolicy_stopped_{stamp}.json").open("x") as f:
            f.write(json.dumps(meta | {"start": t0.isoformat(), "stop": t1.isoformat(), "cost_usd_logged": safe_spent(),
                                       "errors": errors, "stop_messages": {line: str(r)[:300] for line, r in res.items()
                                                                           if isinstance(r, BudgetStop)}}, indent=1))
        raise SystemExit(f"stopped: {errors}; logged cost {safe_spent()}")
    with (DATA / f"nano_onpolicy_drop_reasons_{stamp}.json").open("x") as f:  # may quote model text: data/ only
        f.write(json.dumps({line: r.payload["dropped"] for line, r in res.items()}, indent=1))
    by_prompt = collections.defaultdict(list)  # every attempt's draws: a refusal on any attempt counts
    for f in sorted(DATA.glob("nano_onpolicy_draws_*.jsonl")):
        for x in f.read_text().splitlines():
            r = json.loads(x)
            by_prompt[r["prompt_sha256"]].append(r["outcome"])
    every_paid = [json.loads(x) for f in sorted(DATA.glob("nano_onpolicy_paid_calls_*.jsonl")) for x in f.read_text().splitlines()]
    flagged = {r["prompt_sha256"] for r in every_paid if r["inspect_refusal"] or r["stop_reason"] == "content_filter"}

    def refused(line, cid):
        h = sha(prompts[line][cid])
        return h in flagged or any(x.startswith("refused") for x in by_prompt[h])

    def category(line, cid):
        if refused(line, cid):
            return "refused"
        return "error" if "error" in by_prompt[sha(prompts[line][cid])] else "unparsed"

    # a run with a refused draw on any attempt is a refusal even if its scores parsed (rule: never scored)
    moved = {line: sorted(cid for cid in r.payload["scores"] if refused(line, cid)) for line, r in res.items()}
    for line, r in res.items():
        for cid in moved[line]:
            r.payload["dropped"][cid] = "refused after parsing"
            del r.payload["scores"][cid]

    out = meta | {"start": t0.isoformat(), "stop": t1.isoformat(), "cost_usd_logged": spent(), "paid_calls": len(paid),
                  "failed_calls_reserve_usd": judge.unlogged,
                  "lines": {line: meta["lines"][line] | {
                      "scores": r.payload["scores"], "attrition": r.payload["attrition"], "prompt_sha256": r.prompt_sha256,
                      "dropped": {cid: category(line, cid) for cid in r.payload["dropped"]},
                      "drop_counts": dict(collections.Counter(category(line, cid) for cid in r.payload["dropped"])),
                      "refused_after_parsing": len(moved[line]),
                      "where": {cid: {"task": w[0], "eval": w[1], "eval_sha256": w[2]} for cid, w in where[line].items()},
                      "plant_per_run": plant_runs.get(line)}
                      for line, r in res.items()}}
    path = HERE / "results" / f"nano_onpolicy_scores_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1, allow_nan=False))
    print("stop", t1.isoformat(), f"logged cost ${out['cost_usd_logged']:.2f}",
          {line: (len(v["scores"]), v["drop_counts"]) for line, v in out["lines"].items()}, "->", path)


if __name__ == "__main__":
    main()
