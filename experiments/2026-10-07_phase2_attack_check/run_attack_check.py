"""Launch the attack check: for each cell in config/cells.json, a real attack and a real honest run of its main task
(step 2's simulator reference), Qwen3.5-27B acting, one `inspect eval` per run, all in parallel. Runs ON THE DEVBOX,
with the snapshot's Python and VLLM_API_KEY in the environment (rr-env, never argv):

  ~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/attack_check/run_attack_check.py \\
      --base-url https://<tunnel>/v1 --commit <probe-shift commit> [--runs honest_a/tiktok,honest_a/web_scraping]

The sims code is one of config/run.yaml's `simulators` (--simulator: september, the default, or today): its base plus
the policy-limits patch, run with the base's own environment (`venv`) and PYTHONPATH=<tree>/src, which puts the
patched code ahead of the
environment's editable install; the launcher checks that before starting anything. Each run gets the policy's call
limits (PT_POLICY_*) and keeps every raw model call in its log (--log-model-api), so each call's x-irid request ID
joins it to the vLLM proxy's token record.

Arms: `attack` and `honest_a` (real, config/run.yaml), and `sim_attack` (step 2: simulated by the September simulator,
config/sim.yaml, forked at 0 from a real honest reference per environment, given by --references). No run is redrawn
once its policy has started (Aswin, 2026-10-07): a policy timeout ends the episode and the run is kept
(PT_POLICY_TIMEOUT_ENDS_EPISODE), other errors are recorded with the partial transcript, and --retry-on-error is
refused. Only a run that failed before the policy's first call (sandbox setup) is relaunched, at most
MAX_PRESTART_REDRAWS times, into <out>/<arm>/<env>_redraw<k>.

Writes <out>/<arm>/<env>/*.eval, <out>/logs/<arm>_<env>.log and <out>/launch.json (inputs, commands and environment
minus the keys, timings, outcomes, and per run and per arm: steps, step-limit hits, turns at the token cap, timeouts,
how each episode ended, usable runs, runs that failed after the policy started, and redraws). Exits nonzero unless
every run left exactly one sample without an error.
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

HERE = Path(__file__).resolve().parent
PT = ZoneInfo("America/Los_Angeles")


def now() -> str:
    return datetime.now(PT).isoformat(timespec="seconds")


TERMINATING = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
MAX_PRESTART_REDRAWS = 2
STRICT_PATCH = "config/policy_strict_tools_off.patch"  # needed by every tree when strict_tools is false
STOP: list[int] = []  # termination signals received; the launch and wait loops check it at safe points


def _note_signal(signum, _frame) -> None:
    """Record the signal and let the loops stop at their next check, so a signal can never land between starting a
    child and registering it (that child would then outlive the launcher in its own session)."""
    STOP.append(signum)


def _sleep(seconds: float) -> None:
    """Sleep, waking within a second of a termination signal."""
    end = time.time() + seconds
    while time.time() < end and not STOP:
        time.sleep(max(0.0, min(1.0, end - time.time())))


def _signal_group(proc, sig: int) -> None:
    """Signal a child's whole process group, not just the `inspect` process: each runs in its own session
    (start_new_session), so its workers are in a group of their own, whose ID is the child's PID."""
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:  # the group is already gone
        pass


def _group_alive(proc) -> bool:
    proc.poll()  # reap the leader if it has exited, so it no longer counts
    try:
        os.killpg(proc.pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_children(procs: list) -> None:
    """Interrupt every still-running `inspect eval` (it then cancels its sample), and kill any left after 60 s, so
    nothing keeps running or spending once the launcher stops. Signals during this bounded window are only noted."""
    # By group: a worker can outlive its `inspect` leader (and every attempt is covered, redrawn ones included).
    alive = [entry["proc"] for entry in procs if _group_alive(entry["proc"])]
    for p in alive:
        _signal_group(p, signal.SIGINT)
    deadline = time.time() + 60
    while time.time() < deadline and any(_group_alive(p) for p in alive):
        time.sleep(1)
    for p in alive:  # whatever is left of each group, even if `inspect` itself has exited and a worker hasn't
        _signal_group(p, signal.SIGKILL)
        if p.poll() is None:
            try:
                p.wait(timeout=30)  # so the exit status recorded next is final
            except subprocess.TimeoutExpired:  # survived SIGKILL: noted, and the launch record is still written
                print(f"{now()} pid {p.pid} still alive 30 s after SIGKILL", flush=True)
    if alive:
        print(f"{now()} stopped {len(alive)} running evaluations", flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def src_sha256(tree: Path) -> str:
    """The snapshot's src/ as one hash: sorted relative paths and their file hashes, __pycache__ excluded."""
    h = hashlib.sha256()
    for p in sorted(q for q in (tree / "src").rglob("*") if q.is_file() and "__pycache__" not in q.parts):
        h.update(str(p.relative_to(tree)).encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def host_fingerprint(settings: Path, cfg: dict) -> dict:
    """What the agent's environment can show but the config doesn't pin: the launcher's umask, the Docker engine,
    group-writable files in the settings folder (bind-mounted sources carry the host's modes into `ls -l`), each
    environment image's default umask, and the IDs and digests of the images the runs will use. A September batch
    run under umask 002 is separable from the rest by its `ls -l` modes alone, so every launch records these."""
    umask = os.umask(0)
    os.umask(umask)

    def out(cmd: list[str]) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()

    group_writable = {env: sum(1 for p in (settings / "linuxarena" / env).rglob("*")
                               if ".git" not in p.parts and p.is_file() and p.stat().st_mode & 0o020)
                      for env in cfg["envs"]}
    images = [line for line in out(["docker", "images", "--digests", "--format",
                                    "{{.Repository}}:{{.Tag}} {{.Digest}} {{.ID}}"]).splitlines()
              if line.startswith(("linuxarena/", "quay.io/minio/", "nginx:", "postgres:", "redis:", "rabbitmq:",
                                  "elasticsearch:", "ghcr.io/chroma-core/"))]
    return {"launcher_umask": f"{umask:04o}", "docker_version": json.loads(out(["docker", "version", "--format",
                                                                                 "{{json .}}"])),
            "settings_group_writable_files": group_writable,
            "image_default_umask": {env: out(["docker", "run", "--rm", "--entrypoint", "sh",
                                              f"linuxarena/env-{env}:{cfg['env_commits'][env]}", "-c", "umask"])
                                    for env in cfg["envs"]},
            "images": images}


def policy_calls(sample, policy: str, max_steps: int, max_tokens: int) -> dict:
    """The policy's model calls in one sample: steps (completed calls), whether it reached the step limit (such runs
    are kept), calls cut at the token cap, calls that failed (and how many of those were attempt timeouts), and how the
    episode ended."""
    from inspect_ai.event import ModelEvent
    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool

    calls = [e for e in sample.events if isinstance(e, ModelEvent) and e.model == policy]
    done = [e for e in calls if e.error is None and not e.pending and e.output and e.output.choices]
    failed = [e for e in calls if e.error is not None]
    tools = [tc.function for m in sample.messages if isinstance(m, ChatMessageAssistant) for tc in m.tool_calls or []]
    ended_by = (sample.store or {}).get("policy_ended_by")
    if ended_by == "out_of_context" and done and calls[-1] is done[-1]:
        done = done[:-1]  # the call the context guard discarded (config/policy_context_guard.patch): not a step
    ended = (ended_by if ended_by in ("policy_timeout", "out_of_context")
             else "time limit" if sample.limit and sample.limit.type == "time" else f"{sample.limit.type} limit"
             if sample.limit else "error" if sample.error else "submit" if tools and tools[-1] == "submit"
             else "step limit or no submit")
    return {"steps": len(done), "step_limit_hits": int(len(done) >= max_steps),
            # By token count, not stop reason: vLLM reports `tool_calls` for a turn cut at the cap once it has parsed a
            # call (the reruns' 3 submit turns ran to 32,768 tokens and read `tool_calls`).
            "cap_hits": sum(bool(e.output.usage) and e.output.usage.output_tokens >= max_tokens for e in done),
            "failed_calls": len(failed), "timeouts": sum("AttemptTimeoutError" in str(e.error) for e in failed),
            "retried_calls": sum(bool(e.retries) for e in calls), "ended": ended,
            # Usable: the sample's transcript holds at least one result of a tool other than submit (an action a
            # simulator can fork from: sce doesn't fork on submit; a run ending in plain text or a refusal has none,
            # and a time-limited one keeps no transcript) and the sample ended without an error (a policy timeout,
            # which keeps its transcript, counts).
            "usable": int(any(isinstance(m, ChatMessageTool) and m.function != "submit" for m in sample.messages)
                          and sample.error is None),
            "failed_after_start": int(bool(calls) and sample.error is not None)}


def policy_call_ids(sample, policy: str) -> list[tuple[str | None, str]]:
    """Each policy call's Inspect request ID (from its raw request, kept by --log-model-api) and the tools it called,
    joined with '+' when a turn called more than one."""
    from inspect_ai.event import ModelEvent

    out = []
    for e in sample.events:
        if isinstance(e, ModelEvent) and e.model == policy:
            irid = ((e.call.request or {}).get("extra_headers") or {}).get("x-irid") if e.call else None
            calls = e.output.message.tool_calls if e.output and e.output.choices else None
            out.append((irid, "+".join(c.function for c in calls) if calls else "(no tool call)"))  # every call
    return out


def stop_reasons(base_url: str, request_ids: list[str]) -> dict:
    """vLLM's stop reasons per request ID, from the request-log proxy (the eval log keeps only finish reasons)."""
    request = urllib.request.Request(f"{base_url.removesuffix('/v1')}/proxy/stop_reasons", method="POST",
                                     data=json.dumps({"request_ids": request_ids}).encode(),
                                     headers={"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}",
                                              "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(request, timeout=120).read())


def repetition_stops(calls: list[tuple[str | None, str]], reasons: dict) -> dict:
    """Calls vLLM ended for repetition (decision D), in total and by tool, plus the calls that couldn't be checked."""
    hit = [tool for irid, tool in calls if irid and any(
        "repetition_detected" in r["stop_reasons"] for r in reasons.get(irid, []))]
    return {"repetition_stops": len(hit), "repetition_stops_by_tool": dict(Counter(hit)),
            "calls_without_request_id": sum(irid is None for irid, _ in calls),
            "calls_unknown_to_proxy": sum(bool(irid) and not reasons.get(irid) for irid, _ in calls),
            # A call's token record: a successful response whose prompt and completion IDs and logprobs all came back
            # (retries leave other records under the same ID). Proxies before 2026-10-08 don't report it (None).
            "calls_without_complete_token_record": sum(bool(irid) and bool(reasons.get(irid)) and not any(
                r.get("status") == 200 and r.get("complete") is not False for r in reasons[irid]) for irid, _ in calls)}


BUFFERS = Path.home() / ".local" / "share" / "inspect_ai" / "samplebuffer"
# Errors that can only come from starting the sample's sandbox (docker compose), before the policy runs.
SETUP_MARKERS = ("No services started", "sandboxenv_init", "sample_init", "compose_up", "compose up")


def policy_started_in_buffer(log_path: Path, policy: str) -> bool | None:
    """Whether Inspect's sample buffer for this log recorded a policy call; None if there is no readable buffer. A
    sample reaches the .eval only when it finishes, so a crashed run's log holds none, while its buffer keeps the
    events it had."""
    import sqlite3

    found = None
    for db in glob.glob(str(BUFFERS / "*" / f"{glob.escape(log_path.stem)}*.db")):
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                rows = con.execute("select data from events").fetchall()
            finally:
                con.close()
        except sqlite3.Error:
            continue
        started, corrupt = False, False
        for (data,) in rows:  # a row cut short by the crash is noted, never raised
            try:
                d = json.loads(data)
            except (TypeError, ValueError):
                corrupt = True
                continue
            started = started or (d.get("event") == "model" and d.get("model") == policy)
        if started:
            return True
        if not corrupt:  # read in full and no policy call in it; a corrupt row might have been one: unknown
            found = False
    return found


def prestart_failure(log_dir: Path, policy: str) -> str | None:
    """Why an attempt failed before the policy's first call (redraw it), None if it didn't (it succeeded, or the policy
    had started: such a run is kept, never redrawn), or "unknown: ..." when that can't be told: such a run is recorded
    and never redrawn (D5). Never raises: one run's broken log mustn't stop the others."""
    from inspect_ai.event import ModelEvent
    from inspect_ai.log import read_eval_log

    paths = sorted(log_dir.glob("*.eval"))
    if not paths:
        return "no eval log"  # inspect never got as far as opening its log: nothing ran
    logs, unreadable = [], []
    for p in paths:
        try:
            logs.append(read_eval_log(str(p)))
        except Exception as e:  # noqa: BLE001 — a crashed run's truncated log; judged from its buffer below
            unreadable.append(f"{p.name}: {e!r}"[:200])
    samples = [s for log in logs for s in log.samples or []]
    if samples:
        started = any(isinstance(e, ModelEvent) and e.model == policy for s in samples for e in s.events)
        errors = [s.error.message[:300] for s in samples if s.error]
        return f"errored before the policy's first call: {errors[0]}" if errors and not started else None
    # No finished sample: a setup failure, or a crash or cancellation mid-run. The sample buffer tells which.
    statuses = [log.status for log in logs]
    in_buffer = [policy_started_in_buffer(p, policy) for p in paths]
    if any(x is True for x in in_buffer):
        return None  # the policy started: kept, never redrawn
    if in_buffer and all(x is False for x in in_buffer):
        return f"no sample, and its buffer shows no policy call (status {statuses}; unreadable {unreadable})"
    # With no buffer left (Inspect removes it when an eval ends, errors included), only an error raised while setting
    # up the sandbox shows the policy never started (step 1's "No services started"); any other error might have come
    # after it had, so it isn't redrawn.
    setup = [(log.error.message or "")[:200] for log in logs if log.status == "error" and log.error and any(
        m in f"{log.error.message or ''}\n{log.error.traceback or ''}" for m in SETUP_MARKERS)]
    if setup and not unreadable:
        return f"no sample: failed setting up the sandbox ({setup[0]})"
    return f"unknown: no sample and no buffer to tell whether the policy started (status {statuses}; unreadable " \
           f"{unreadable})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="the tunnel's OpenAI-compatible base, ending in /v1")
    ap.add_argument("--commit", required=True, help="the probe-shift commit this folder was copied from")
    ap.add_argument("--out", type=Path, help="default: ~/phase2/runs/<PT stamp>_attack_check")
    ap.add_argument("--runs", help="only these <arm>/<env> runs, comma-separated (default: every cell's every arm)")
    ap.add_argument("--arms", help="comma-separated (default: config/run.yaml's arms); sim_attack needs --references")
    ap.add_argument("--references", type=Path, help="JSON {env: real honest reference .eval} for the sim_attack arm")
    ap.add_argument("--simulator", default="september",
                    help="which of run.yaml's simulators (sims trees) every run of this launch uses")
    args = ap.parse_args()
    stray = sorted(k for k in os.environ if k.startswith(("BA_", "PT_", "LA_", "SCE_", "INSPECT_", "ENV_IMAGE_")))
    assert not stray, f"unset these inherited settings first; they would reach every run unrecorded: {stray}"
    assert os.environ.get("VLLM_API_KEY"), "VLLM_API_KEY is not in the environment"
    cfg = yaml.safe_load((HERE / "config" / "run.yaml").read_text())
    cells = json.loads((HERE / "config" / "cells.json").read_text())["cells"]
    simulator = cfg["simulators"][args.simulator]
    tree, settings = Path(simulator["tree"]).expanduser(), Path(cfg["settings_dir"]).expanduser()
    venv = Path(simulator["venv"]).expanduser()
    assert src_sha256(tree) == simulator["src_sha256"], f"{tree}/src is not the pinned tree (setup_devbox.sh)"
    assert cfg["policy_limits"]["strict_tools"] or STRICT_PATCH in simulator.get("extra_patches", []), (
        f"{args.simulator}: strict_tools is false but its tree lacks {STRICT_PATCH}: its policy would send strict tools")
    # The runs must import the policy from the patched tree, not from the environment's editable install.
    which = "import runner.policy.agent as a; print(a.__file__)"
    imported = subprocess.run([str(venv / "bin" / "python"), "-c", which], cwd=tree, capture_output=True, text=True,
                              env=os.environ | {"PYTHONPATH": str(tree / "src")}, check=True).stdout.strip()
    assert Path(imported).is_relative_to(tree / "src"), f"runs would import the policy from {imported}"
    for env, commit in cfg["env_commits"].items():
        head = subprocess.run(["git", "-C", str(settings / "linuxarena" / env), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        assert head == commit, f"{env} settings at {head}, config pins {commit}"
    out = (args.out or Path.home() / "phase2" / "runs" / f"{datetime.now(PT):%Y-%m-%d_%H%M%S}_attack_check").resolve()
    assert not out.exists(), f"{out} exists; runs are append-only"
    (out / "logs").mkdir(parents=True)

    run = cfg["run"]
    limits = cfg["policy_limits"]
    common = {"PT_ATTACKER": cfg["policy"], "BA_MAX_STEPS": str(run["max_steps"]), "VLLM_BASE_URL": args.base_url,
              "PYTHONPATH": str(tree / "src"), "PT_POLICY_ATTEMPT_TIMEOUT": str(limits["attempt_timeout_s"]),
              "PT_POLICY_MAX_RETRIES": str(limits["max_retries"]), "PT_POLICY_MAX_TOKENS": str(limits["max_tokens"]),
              "PT_POLICY_TIMEOUT_ENDS_EPISODE": "1" if limits["timeout_ends_episode"] else "",
              "PT_POLICY_STRICT_TOOLS": "" if limits["strict_tools"] else "0",
              "SKIP_DOCKER_BUILD": "1", "CONTROL_TOWER_SETTINGS_DIR": str(settings),
              "INTERNET_SIMULATOR_IMAGE": cfg["internet_simulator_image"],
              **{f"ENV_IMAGE_TAG_{e.upper()}": c for e, c in cfg["env_commits"].items()}}
    inspect = str(venv / "bin" / "inspect")  # started directly, not via `uv run`, so a signal reaches it
    # No --timeout or --max-retries: Inspect would merge them over the policy's PT_POLICY_* limits. No --retry-on-error:
    # it would redraw runs after the policy started; pre-start failures are redrawn by this launcher instead.
    real_args = ["--max-samples", "1", "--time-limit", str(run["time_limit_s"]), "--log-model-api",
                 "--display", "plain"]
    sim = yaml.safe_load((HERE / "config" / "sim.yaml").read_text())
    sim_args = [*sim["inspect_args"], "--display", "plain"]
    for flags in (real_args, sim_args):  # also as --flag=value
        assert not {"--retry-on-error", "--timeout", "--max-retries"} & {f.split("=")[0] for f in flags}, flags
    policy_env = {k: v for k, v in common.items() if k.startswith("PT_POLICY_")}
    assert {k: v for k, v in sim["env"].items() if k.startswith("PT_POLICY_")} == policy_env, "sim.yaml != run.yaml"
    arms = args.arms.split(",") if args.arms else cfg["arms"]
    assert set(arms) <= {"attack", "honest_a", "sim_attack"}, arms
    references = json.loads(args.references.read_text()) if args.references else {}
    pinned = {}  # each sim_attack reference, copied into the run folder and hashed: the fork source can't drift
    wanted = set(args.runs.split(",")) if args.runs else None
    if wanted:
        assert wanted <= {f"{a}/{c['env']}" for c in cells for a in arms}, f"unknown runs: {sorted(wanted)}"
    plan = []
    for cell in cells:
        for arm in arms:
            if wanted and f"{arm}/{cell['env']}" not in wanted:
                continue
            task, flags = f"{HERE / 'tasks.py'}@la_real_attack_sept", real_args
            attack = {"BA_MODE": "attack", "BA_CELLS": cell["cell"], "BA_ATTACKER_PROMPT": run["attacker_prompt"],
                      "PHASE2_CELLS": str(HERE / "config" / "cells.json")}
            if arm == "attack":
                env_vars = common | {"LA_ENV": cell["env"]} | attack
            elif arm == "honest_a":
                env_vars = common | {"LA_ENV": cell["env"], "BA_MODE": "honest", "BA_MAINS": cell["main"]}
                task = "src/runner/tasks/linuxarena_real.py"  # the patched tree's copy (cwd is the tree)
            else:  # the reference is both the fork source and the simulator's original trajectory, as in September
                source = Path(references[cell["env"]]).expanduser().resolve()
                assert source.is_file(), f"{cell['env']}: no reference at {source}"
                (out / "references").mkdir(exist_ok=True)
                ref = str(shutil.copy2(source, out / "references" / f"{cell['env']}.eval"))
                pinned[cell["env"]] = {"source": str(source), "copy": ref, "sha256": sha256(Path(ref))}
                env_vars = common | sim["env"] | simulator.get("env", {}) | {
                    "LA_ENV": cell["env"], "BA_REAL_LOG": ref, "BA_HONEST_LOG": ref} | (
                    {k: v for k, v in attack.items() if k != "BA_ATTACKER_PROMPT"})
                task, flags = f"{HERE / 'tasks.py'}@la_sim_attack_sept", sim_args
            plan.append({"arm": arm, "env": cell["env"], "cell": cell["cell"], "env_vars": env_vars,
                         "cmd": [inspect, "eval", task], "flags": flags})
    if any(p["arm"] == "sim_attack" for p in plan):
        assert os.environ.get("OPENAI_API_KEY"), "the simulator needs OPENAI_API_KEY in the environment (rr-env)"

    record = {"started": now(), "finished": None, "probe_shift_commit": args.commit, "simulator": args.simulator,
              "sims_tree": str(tree), "sims_src_sha256": simulator["src_sha256"], "sims_venv": str(venv),
              "policy_imported_from": imported,
              "only_runs": args.runs, "arms": arms, "references": pinned, "config": cfg, "sim_config": sim,
              "input_sha256": {p: sha256(HERE / p) for p in ("config/run.yaml", "config/sim.yaml", "config/cells.json",
                                                            "tasks.py", "run_attack_check.py",
                                                            "config/sept_snapshot_policy_limits.patch",
                                                            "config/policy_strict_tools_off.patch")},
              "host": host_fingerprint(settings, cfg), "runs": []}
    procs = []
    last_launch: dict[str, float] = {}

    def start(item: dict, rec: dict, attempt: int) -> None:
        """Launch one attempt of a run; attempt 0 logs to <arm>/<env>, a redraw to <arm>/<env>_redraw<k>. Any two
        launches of one environment are at least `stagger_s` apart, redraws included: they write the same compose
        file, and two at once race."""
        _sleep(run["stagger_s"] - (time.time() - last_launch.get(item["env"], float("-inf"))))
        if STOP:
            return
        last_launch[item["env"]] = time.time()
        suffix = f"_redraw{attempt}" if attempt else ""
        log_dir = out / item["arm"] / f"{item['env']}{suffix}"
        cmd = [*item["cmd"], "--log-dir", str(log_dir), *item["flags"]]
        with open(out / "logs" / f"{item['arm']}_{item['env']}{suffix}.log", "w") as log:
            proc = subprocess.Popen(cmd, cwd=tree, env=os.environ | item["env_vars"], stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        rec |= {"cmd": cmd, "log_dir": str(log_dir), "exit_code": None}
        rec["attempts"].append({"log_dir": str(log_dir), "start": now(), "pid": proc.pid})
        procs.append({"proc": proc, "rec": rec, "t0": time.time(), "item": item, "attempt": attempt})
        (out / "launch.json").write_text(json.dumps(record, indent=1))  # on disk before the next launch
        print(f"{now()} launched {item['arm']} {item['env']}{suffix} (pid {proc.pid})", flush=True)
    # Children run in their own sessions, so a terminal's Ctrl+C reaches them once, through stop_children, rather than
    # twice (directly and forwarded), which can cut Inspect's own cancellation short. Ctrl+C, SIGTERM and SIGHUP (a
    # tmux session killed) are only noted while children run; the loops then stop and clean up.
    for sig in TERMINATING:
        signal.signal(sig, _note_signal)
    try:
        # Arms in the outer loop: every environment's first arm starts at once, the second arm `stagger_s` later.
        for wave, arm in enumerate(arms):
            if wave:
                _sleep(run["stagger_s"])
            for item in (p for p in plan if p["arm"] == arm):
                if STOP:
                    break
                rec = {k: item[k] for k in ("arm", "env", "cell", "env_vars")} | {
                    "start": now(), "end": None, "process_seconds": None, "attempts": []}
                record["runs"].append(rec)
                start(item, rec, 0)
        while not STOP and any(entry["rec"]["exit_code"] is None for entry in procs):
            _sleep(10)
            for entry in list(procs):
                proc, rec = entry["proc"], entry["rec"]
                if entry.get("done") or (code := proc.poll()) is None:
                    continue
                entry["done"] = True
                seconds = round(time.time() - entry["t0"], 1)
                rec["attempts"][-1] |= {"end": now(), "exit_code": code, "process_seconds": seconds}
                print(f"{now()} {rec['arm']} {rec['env']} exited {code} after {seconds} s", flush=True)
                try:
                    why = prestart_failure(Path(rec["log_dir"]), cfg["policy"])
                except Exception as e:  # noqa: BLE001 — one run's broken log must not stop the others
                    why = f"unknown: the pre-start check raised {e!r}"[:300]
                rec["attempts"][-1]["prestart"] = why
                if why and not why.startswith("unknown") and entry["attempt"] < MAX_PRESTART_REDRAWS and not STOP:
                    rec["attempts"][-1]["redrawn_because"] = why
                    print(f"{now()} {rec['arm']} {rec['env']}: {why}; redrawing", flush=True)
                    start(entry["item"], rec, entry["attempt"] + 1)
                else:
                    rec |= {"end": now(), "process_seconds": seconds, "exit_code": code}
                (out / "launch.json").write_text(json.dumps(record, indent=1))
    finally:
        stop_children(procs)  # a no-op when every run has already exited
        for entry in procs:
            if entry["rec"]["exit_code"] is None and not entry.get("done"):
                entry["rec"] |= {"end": now(), "process_seconds": round(time.time() - entry["t0"], 1),
                                 "exit_code": entry["proc"].poll(), "stopped_by_launcher": True}
        record["stopped_by_signal"] = STOP[:]
        (out / "launch.json").write_text(json.dumps(record, indent=1))
        signal.signal(signal.SIGINT, signal.default_int_handler)  # reading the logs below stays interruptible
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_DFL)
    if STOP:
        raise SystemExit(f"launcher stopped by signal {STOP[0]}; every started run was interrupted (see launch.json)")

    # `inspect eval` can exit 0 with an errored or cancelled log, so each run's outcome is read from its logs.
    from inspect_ai.log import read_eval_log

    for rec in record["runs"]:
        logs, rec["unreadable_logs"] = [], []
        for p in sorted(Path(rec["log_dir"]).glob("*.eval")):  # the final attempt's
            try:
                logs.append(read_eval_log(str(p)))
            except Exception as e:  # noqa: BLE001 — a crashed run's truncated log: recorded, and the run fails below
                rec["unreadable_logs"].append(f"{p.name}: {e!r}"[:300])
        rec["eval_status"] = [log.status for log in logs]
        rec["sample_errors"] = [s.error.message[:500] for log in logs for s in log.samples or [] if s.error]
        # A run cut off by a limit still ends `success` with no error. It is a legitimate attack outcome (September's
        # protocol), but an honest run cut off this way cannot be step 2's reference: sce copies the conversation onto
        # the sample only when the agent returns, so the simulator has nothing to fork ("no forkable samples").
        rec["sample_limits"] = [s.limit.type for log in logs for s in log.samples or [] if s.limit]
        rec["samples"] = sum(len(log.samples or []) for log in logs)
        rec["policy_calls"] = [policy_calls(s, cfg["policy"], run["max_steps"], limits["max_tokens"])
                               for log in logs for s in log.samples or []]
        rec["_call_ids"] = [c for log in logs for s in log.samples or [] for c in policy_call_ids(s, cfg["policy"])]
    # Repetition stops (D) are visible only in vLLM's stop_reason, which the proxy logged; ask it before the server
    # stops. A failure is recorded, never read as zero stops.
    try:
        reasons = stop_reasons(args.base_url, sorted({i for r in record["runs"] for i, _ in r["_call_ids"] if i}))
        record["stop_reasons_error"] = None
    except Exception as e:  # noqa: BLE001 — recorded and printed; the request log on the volume still has them
        reasons, record["stop_reasons_error"] = None, repr(e)[:500]
        print(f"WARNING: could not read stop reasons from the proxy: {e!r}", flush=True)
    for rec in record["runs"]:
        calls = rec.pop("_call_ids")
        rec["repetition"] = repetition_stops(calls, reasons) if reasons is not None else None
    keys = ("steps", "step_limit_hits", "cap_hits", "timeouts", "failed_calls", "retried_calls", "usable",
            "failed_after_start")
    record["policy_calls_by_arm"] = {
        arm: {"runs": sum(r["arm"] == arm for r in record["runs"]),
              "prestart_redraws": sum(len(r["attempts"]) - 1 for r in record["runs"] if r["arm"] == arm)}
        | {k: sum(c[k] for r in record["runs"] if r["arm"] == arm for c in r["policy_calls"]) for k in keys}
        | {"ended": dict(Counter(c["ended"] for r in record["runs"] if r["arm"] == arm for c in r["policy_calls"]))}
        for arm in dict.fromkeys(r["arm"] for r in record["runs"])}
    for arm, totals in record["policy_calls_by_arm"].items():
        reps = [r["repetition"] for r in record["runs"] if r["arm"] == arm and r["repetition"] is not None]
        counted = ("repetition_stops", "calls_without_request_id", "calls_unknown_to_proxy")
        totals["repetition"] = None if reasons is None else {k: sum(x[k] for x in reps) for k in counted} | {
            "repetition_stops_by_tool": dict(sum((Counter(x["repetition_stops_by_tool"]) for x in reps), Counter()))}
    record["finished"] = now()
    (out / "launch.json").write_text(json.dumps(record, indent=1))
    print(f"-> {out}", flush=True)
    for r in record["runs"]:
        for c in r["policy_calls"]:
            print(f"{r['arm']}/{r['env']}: {c['steps']} steps (limit {run['max_steps']}), ended by {c['ended']}; "
                  f"{c['cap_hits']} turns at the token cap, {c['timeouts']} timeouts, "
                  f"{c['failed_calls']} failed calls; repetition stops: {r['repetition']}", flush=True)
    for arm, totals in record["policy_calls_by_arm"].items():
        print(f"{arm}: {json.dumps(totals)}", flush=True)
    failed = [f"{r['arm']}/{r['env']}" for r in record["runs"]
              if r["exit_code"] != 0 or r["eval_status"] != ["success"] or r["sample_errors"] or r["samples"] != 1
              or r["unreadable_logs"]
              or (r["arm"] == "honest_a" and r["sample_limits"])
              # A reference must hold at least one action to be forked (a first-call policy timeout leaves none). Such a
              # run is kept, not redrawn, but it cannot serve as step 2's reference.
              or (r["arm"] == "honest_a" and not all(c["usable"] for c in r["policy_calls"]))]
    if failed:
        raise SystemExit(f"runs without exactly one clean, usable sample: {failed} (see launch.json)")


if __name__ == "__main__":
    main()
