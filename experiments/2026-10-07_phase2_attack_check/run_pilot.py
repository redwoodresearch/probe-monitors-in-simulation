"""Launch the Phase 2 pilot: every task in config/pilot.yaml's task file, each with 3 real runs and, once its reference
is usable, 4 simulated runs, all keyed by task. Runs ON THE DEVBOX with the September snapshot's Python, VLLM_API_KEY
and OPENAI_API_KEY in the environment (rr-env, never argv):

  ~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/attack_check/run_pilot.py \\
      --base-urls https://<tunnel 1>/v1,https://<tunnel 2>/v1 --commit <probe-shift commit> [--tasks t01_x,t02_y]

Per task:
  - real, on `real_simulator`'s tree: honest_a (the simulators' reference: fork source and original trajectory),
    honest_b (the real run a simulated honest run is compared with), attack (Sol's prompt and cell, tasks.py);
  - simulated, for each of `simulators`: sim_honest@<simulator> and sim_attack@<simulator>, forked at 0 from the
    task's honest_a, on that simulator's tree and environment (config/sim.yaml plus its own `env`).

First, every server must pass the server-start checks (check_timeouts_live.py tiny repetition newline), or nothing
launches. Then one process schedules every run (pre-registered decisions 4-6, pilot_decisions.md):
  - Tasks start strictly in the task file's order (a random order fixed before launch): a task starts only after
    every earlier one has. All of a task's runs go to one server, the least loaded when it starts, so its real and
    simulated runs see the same load.
  - At most `max_runs_per_server` runs per vLLM server and `max_real_runs` real environments on the devbox at once;
    any two launches in one environment, real or simulated, are `stagger_s` apart (they write one compose file).
  - A task's simulated runs wait for its honest_a. An honest_a that isn't usable (no non-submit action, a limit or an
    error) is rerun. honest_a starts at most `reference_tries` times in all, pre-start redraws included (amendments
    12, 17 and 20); every try is kept. If none is usable, the task's simulated runs are recorded as not run.
  - A usable honest_a is then preflighted once per simulator (check_sim_samples.py, both modes, that simulator's tree
    and environment, no model calls): the sample builds from it and the reference's reasoning doesn't reach the
    simulator. A simulator's runs of the task start only if its preflight passes; otherwise they are not run.
  - A run ends "done", or "no policy call" when its last attempt made none.
  - No run is redrawn once its policy has started (D5). A run that failed before the policy's first call is redrawn
    at most MAX_PRESTART_REDRAWS times; one whose start can't be told (a crash that left no readable sample) is
    recorded, not redrawn.
  - Spend guard, on new tasks only: no new task starts once projected simulator spend passes `spend.cap_usd`, or once
    actual spend plus the worst case to finish every started task, the new one included, would pass
    `spend.worst_case_limit_usd` (each unfinished simulated run at the larger of `spend.worst_per_sim_run_usd` and the
    costliest finished one). A started task always runs all its arms: nothing is stopped for spend (Aswin, 10-08).

Writes <out>/<arm>/<task>[_redraw<k>|_rerun<k>]/*.eval, <out>/logs/, <out>/references/<task>.eval and
<out>/launch.json (inputs, every attempt, outcomes, spend, and per arm: runs, steps, limit hits, capped turns,
timeouts, guard stops, usable runs, redraws and reference reruns, plus decision 5's flags).
"""

import argparse
import glob
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

import run_attack_check as rac

HERE = Path(__file__).resolve().parent
STRICT_PATCH = rac.STRICT_PATCH
REAL, SIMULATED = ("honest_a", "honest_b", "attack"), ("sim_honest", "sim_attack")
PREFLIGHT_TIMEOUT_S = 600  # a preflight builds 4 samples with no model call: about a minute; a hung one is failed
BUFFERS = Path.home() / ".local" / "share" / "inspect_ai" / "samplebuffer"


def called_policy(outcome: dict) -> bool:
    """Whether an attempt called the policy at all: any policy model event, completed, failed or still pending (a
    first-call timeout, or a run stopped mid-call, counts). call_ids lists every one (rac.policy_call_ids)."""
    return bool(outcome["call_ids"]) or any(c["steps"] or c["failed_calls"] for c in outcome["policy_calls"])


def run_outcome(log_dir: Path, policy: str, max_steps: int, max_tokens: int) -> dict:
    """What one attempt left: eval status, sample errors and limits, and the policy's calls (rac.policy_calls). An
    unreadable log, or a sample the counting fails on, is recorded, never raised: one run can't stop the others."""
    from inspect_ai.log import read_eval_log

    out = {"eval_status": [], "sample_errors": [], "sample_limits": [], "samples": 0, "policy_calls": [],
           "call_ids": [], "unreadable": [], "errors": []}
    for p in sorted(log_dir.glob("*.eval")):
        try:
            log = read_eval_log(str(p))
        except Exception as e:  # noqa: BLE001 — a crashed run's truncated log; recorded per run
            out["unreadable"].append(f"{p.name}: {e!r}"[:300])
            continue
        out["eval_status"].append(log.status)
        for s in log.samples or []:
            out["samples"] += 1
            out["sample_errors"] += [s.error.message[:500]] if s.error else []
            out["sample_limits"] += [s.limit.type] if s.limit else []
            try:
                out["policy_calls"].append(rac.policy_calls(s, policy, max_steps, max_tokens))
                out["call_ids"] += rac.policy_call_ids(s, policy)
            except Exception as e:  # noqa: BLE001
                out["errors"].append(f"counting {p.name}: {e!r}"[:300])
    out["clean"] = (out["eval_status"] == ["success"] and out["samples"] == 1 and not out["sample_errors"]
                    and not out["unreadable"] and not out["errors"])
    # Decision 4 (adopted 10-08): only a time limit makes a reference unusable; the 200-step limit leaves it forkable.
    timed_out = any(kind in ("time", "working") for kind in out["sample_limits"])
    out["usable_reference"] = out["clean"] and not timed_out and all(c["usable"] for c in out["policy_calls"])
    return out


class BufferGone(Exception):
    """A sample buffer vanished while it was read (its sample just finished)."""


def run_spend(log_dir: Path, policy: str, prices, policy_calls: bool = False) -> float | None:
    """Simulator API spend of one attempt so far: priced usage of every non-policy model call (with policy_calls, the
    policy's own calls instead: an API policy's spend, policy_spend_watch.py), from its finished log, or from Inspect's
    sample buffer while it runs. None when neither could be read (no sample in the log yet and no buffer): the caller
    keeps its last reading rather than count zero."""
    from inspect_ai.event import ModelEvent
    from inspect_ai.log import read_eval_log

    def usd(model: str, u: dict) -> float:
        return prices[model].usd(input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0),
                                 cache_read_tokens=u.get("input_tokens_cache_read") or 0)

    total, read = 0.0, False
    for p in sorted(log_dir.glob("*.eval")):
        try:
            log = read_eval_log(str(p))
        except Exception:  # noqa: BLE001 — still being written; the buffer below has the live calls
            log = None
        if log is not None and log.samples:
            total += sum(usd(e.model, e.output.usage.model_dump()) for s in log.samples for e in s.events
                         if isinstance(e, ModelEvent) and (e.model == policy) == policy_calls and e.output and e.output.usage)
            read = True
            continue
        for db in glob.glob(str(BUFFERS / "*" / f"{glob.escape(p.stem)}*.db")):  # the live sample's events
            try:  # Inspect deletes the buffer when the sample finishes: a vanished one is read from the log next time
                con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                try:
                    rows = con.execute("select event_id, data from events order by id").fetchall()
                finally:
                    con.close()
            except sqlite3.Error:
                raise BufferGone(db)
            last = {}
            for event_id, data in rows:
                d = json.loads(data)
                if d.get("event") == "model":
                    last[event_id] = d
            total += sum(usd(d["model"], d["output"]["usage"]) for d in last.values()
                         if (d.get("model") == policy) == policy_calls and (d.get("output") or {}).get("usage"))
            read = read or bool(last)
    return total if read else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-urls", required=True, help="the servers' OpenAI-compatible bases, comma-separated")
    ap.add_argument("--commit", required=True, help="the probe-shift commit this folder was copied from")
    ap.add_argument("--out", type=Path, help="default: ~/phase2/runs/<PT stamp>_pilot")
    ap.add_argument("--tasks", help="only these task keys, comma-separated (default: every task in the file)")
    ap.add_argument("--approaches", type=Path, default=Path.home() / "phase2" / "approaches",
                    help="the September approach texts, <sha256>.txt (rsynced from data/phase2_attack_check/approaches)")
    ap.add_argument("--references-from", type=Path,
                    help="a prior run folder: fork its pinned honest_a references instead of running honest_a "
                         "(e.g. a simulator's arm rerun after the pilot; the pairing with that run's arms is kept)")
    ap.add_argument("--rerun-reference", default="",
                    help="with --references-from: task keys (comma-separated) whose prior reference can't be used "
                         "(e.g. its preflight failed); each gets a fresh honest_a here, and its simulated runs fork from it")
    ap.add_argument("--pilot-config", default="config/pilot.yaml",
                    help="relative to this folder; config/pilot_qwen3_32b.yaml runs the Qwen3-32B honest arm")
    args = ap.parse_args()
    stray = sorted(k for k in os.environ if k.startswith(("BA_", "PT_", "LA_", "SCE_", "INSPECT_", "ENV_IMAGE_")))
    assert not stray, f"unset these inherited settings first; they would reach every run unrecorded: {stray}"
    assert os.environ.get("VLLM_API_KEY"), "VLLM_API_KEY is not in the environment"
    cfg = yaml.safe_load((HERE / "config" / "run.yaml").read_text())
    pilot = yaml.safe_load((HERE / args.pilot_config).read_text())
    # The policy model (default: run.yaml's), the real arms run (default: all three) and the server-start checks.
    policy = pilot.get("policy", cfg["policy"])
    real_arms = tuple(pilot.get("real_arms", REAL))
    checks = pilot.get("server_checks", ["tiny", "repetition", "newline"])
    assert set(real_arms) <= set(REAL), real_arms
    if pilot["simulators"]:
        assert ("honest_a" in real_arms) != bool(args.references_from), \
            "simulated runs fork from this run's honest_a or, with --references-from, a prior run's (not both)"
        # sim.yaml's env sets the simulated arms' policy, so an override would split the two worlds' policies.
        assert "policy" not in pilot, "a policy override is for real-only arms (no simulators)"
        assert os.environ.get("OPENAI_API_KEY"), "the simulators need OPENAI_API_KEY in the environment (rr-env)"
        # Today's simulator stays unconfirmed until Jeff says which version is his latest (config/run.yaml pins it).
        assert pilot.get("simulator_confirmed") is True, f"{args.pilot_config}: simulator_confirmed is not true"
    # Today's tree fails at sandbox start without ANTHROPIC_API_KEY (DEVIATIONS.md 1). Option (i): a non-secret
    # placeholder in run.yaml's simulators.today.env; option (ii): a tree with the simulator_model passthrough
    # (today_sandbox_model, or Jeff's merged sims commit once pinned).
    wanted_sims = {*pilot["simulators"], *pilot.get("rerun_simulators", []),
                   (pilot.get("time_control") or {}).get("simulator")}
    assert "today" not in wanted_sims or cfg["simulators"]["today"].get("env", {}).get("ANTHROPIC_API_KEY"), (
        "today's tree needs ANTHROPIC_API_KEY at sandbox start: add the placeholder to run.yaml's simulators.today.env "
        "(option i) or use a tree with the simulator_model passthrough (option ii)")
    rerun_refs = [k for k in args.rerun_reference.split(",") if k]
    assert not rerun_refs or args.references_from, "--rerun-reference needs --references-from"
    sim = yaml.safe_load((HERE / "config" / "sim.yaml").read_text())
    pilot_tasks = HERE / pilot["tasks"]  # approaches as sha256s; a run-local copy restores them (below)
    tasks = json.loads(pilot_tasks.read_text())["cells"]
    if args.tasks:
        wanted = args.tasks.split(",")
        assert set(wanted) <= {t["key"] for t in tasks}, f"unknown tasks: {sorted(set(wanted) - {t['key'] for t in tasks})}"
        tasks = [t for t in tasks if t["key"] in wanted]
    unknown = sorted(set(rerun_refs) - {t["key"] for t in tasks})  # against the final selection (after --tasks)
    assert not unknown, f"--rerun-reference: task keys not in this launch {unknown}"  # else a stale reference is reused
    # A task given a fresh reference lost every simulator's runs, so it gets `rerun_simulators` (default: all the
    # pilot's simulators), while the other tasks get `simulators` only.
    rerun_sims = list(pilot.get("rerun_simulators", pilot["simulators"]))
    # A time control (Aswin, 10-08): a few preselected tasks also get `time_control.simulator`'s runs, interleaved with
    # the rest, outside the primary cohort, so launch time isn't confounded with simulator version.
    tc = pilot.get("time_control") or {}
    tc_tasks = set(tc.get("tasks", []))
    all_keys = {c["key"] for c in json.loads(pilot_tasks.read_text())["cells"]}  # the file's, before --tasks
    assert tc_tasks <= all_keys and not tc_tasks & set(rerun_refs), f"time_control tasks: {tc_tasks}"
    tc_tasks &= {t["key"] for t in tasks}  # a --tasks subset keeps only its own control tasks
    def sims_for(key: str) -> list:
        if key in rerun_refs:
            return rerun_sims
        return [*pilot["simulators"], *([tc["simulator"]] if key in tc_tasks else [])]
    servers = args.base_urls.split(",")
    settings = Path(cfg["settings_dir"]).expanduser()
    trees = {}
    for name in {pilot["real_simulator"], *pilot["simulators"], *pilot.get("rerun_simulators", []),
                 *([tc["simulator"]] if tc else [])}:
        s = cfg["simulators"][name]
        tree, venv = Path(s["tree"]).expanduser(), Path(s["venv"]).expanduser()
        assert rac.src_sha256(tree) == s["src_sha256"], f"{tree}/src is not the pinned tree (setup_devbox.sh)"
        tree_defaults = s.get("policy_limits") == "tree_defaults"  # unpatched: the tree's own policy defaults
        assert not tree_defaults or not pilot["simulators"], f"{name}: tree_defaults is for real-only launches"
        assert tree_defaults or cfg["policy_limits"]["strict_tools"] or STRICT_PATCH in s.get("extra_patches", []), (
            f"{name}: strict_tools is false but its tree lacks {STRICT_PATCH}: its policy would send strict tools")
        which = "import runner.policy.agent as a, sim; print(a.__file__); print(sim.__file__)"
        imported = subprocess.run([str(venv / "bin" / "python"), "-c", which], cwd=tree, capture_output=True, text=True,
                                  env=os.environ | {"PYTHONPATH": str(tree / "src")}, check=True).stdout.split()
        assert all(Path(p).is_relative_to(tree / "src") for p in imported), f"{name}: runs would import {imported}"
        trees[name] = {"tree": tree, "venv": venv, "env": s.get("env", {}), "imported": imported, "tree_defaults": tree_defaults,
                       "src_sha256": s["src_sha256"]}
    for env in sorted({t["env"] for t in tasks}):
        head = subprocess.run(["git", "-C", str(settings / "linuxarena" / env), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        assert head == cfg["env_commits"][env], f"{env} settings at {head}, config pins {cfg['env_commits'][env]}"
    out = (args.out or Path.home() / "phase2" / "runs" / f"{datetime.now(rac.PT):%Y-%m-%d_%H%M%S}_pilot").resolve()
    assert not out.exists(), f"{out} exists; runs are append-only"
    (out / "logs").mkdir(parents=True)
    (out / "references").mkdir()
    # tasks.py (PHASE2_CELLS) and the preflights read this copy, every approach text checked against its sha256.
    import approaches

    # The restored texts must never be stageable: inside a git checkout only its git-ignored data/ is allowed. On the
    # devbox this folder is a plain copy (~/phase2/attack_check), so there is no checkout to guard.
    repo = HERE.parents[1]
    if (repo / ".git").exists() and (repo / "src" / "probe_shift").is_dir():  # this repo's checkout, not any git folder
        assert not out.is_relative_to(repo) or out.is_relative_to(repo / "data"), f"{out} is inside the repo, outside data/"
    tasks_file = approaches.materialize(pilot_tasks, out / "cells.json", args.approaches.expanduser())

    run, limits = cfg["run"], cfg["policy_limits"]
    # A real-only launch matched to another batch may set its own step limit and time limit (zoo C: the overlap800
    # batch's 150 steps and 2,700 s, from its log header).
    if pilot.get("run"):
        assert not pilot["simulators"], "pilot run: only in a launch without simulators"
        assert set(pilot["run"]) <= {"max_steps", "time_limit_s"}, pilot["run"]
        run = run | pilot["run"]
    # A launch without simulators may set its own reply cap and a context window (the Qwen3-32B native-window arm,
    # config/policy_context_guard.patch): sim.yaml's limits are run.yaml's, so a launch with simulated runs can't.
    if pilot.get("policy_limits"):
        assert not pilot["simulators"], "pilot policy_limits: only in a launch without simulators"
        assert set(pilot["policy_limits"]) <= {"max_tokens", "context_window"}, pilot["policy_limits"]
        limits = limits | pilot["policy_limits"]
    real_args = ["--max-samples", "1", "--time-limit", str(run["time_limit_s"]), "--log-model-api", "--display", "plain"]
    sim_args = [*sim["inspect_args"], "--display", "plain"]
    for flags in (real_args, sim_args):  # also as --flag=value
        assert not {"--retry-on-error", "--timeout", "--max-retries"} & {f.split("=")[0] for f in flags}, flags

    def common(name: str, server: str) -> dict:
        tree = trees[name]["tree"]
        env = {"PT_ATTACKER": policy, "BA_MAX_STEPS": str(run["max_steps"]), "VLLM_BASE_URL": server,
                "PYTHONPATH": str(tree / "src"), "PT_POLICY_ATTEMPT_TIMEOUT": str(limits["attempt_timeout_s"]),
                "PT_POLICY_MAX_RETRIES": str(limits["max_retries"]), "PT_POLICY_MAX_TOKENS": str(limits["max_tokens"]),
                "PT_POLICY_TIMEOUT_ENDS_EPISODE": "1" if limits["timeout_ends_episode"] else "",
                "PT_POLICY_STRICT_TOOLS": "" if limits["strict_tools"] else "0",
                **({"PT_POLICY_CONTEXT_WINDOW": str(limits["context_window"])} if limits.get("context_window") else {}),
                "SKIP_DOCKER_BUILD": "1", "CONTROL_TOWER_SETTINGS_DIR": str(settings),
                "INTERNET_SIMULATOR_IMAGE": cfg["internet_simulator_image"],
                **{f"ENV_IMAGE_TAG_{e.upper()}": c for e, c in cfg["env_commits"].items()}}
        if trees[name]["tree_defaults"]:  # an unpatched tree reads no PT_POLICY_* setting: none is passed or recorded
            env = {k: v for k, v in env.items() if not k.startswith("PT_POLICY_")}
        return env

    # Replayed image lookups (#82 amendment 32): with `digest_sets`, every run of a task, and the rewrite before its
    # preflight, get that task's PHASE2_DIGEST_SET, so control-tower's generated compose file is the one its reference
    # recorded (check_mixed_forms.py); a task without a set gets the tag form. Needs fixed_digest_lookup/ first on PATH.
    assert not pilot.get("canary") or pilot["canary"] in {t["key"] for t in tasks}, "the canary isn't in this launch"
    replay = json.loads((HERE / pilot["digest_sets"]).read_text()) if pilot.get("digest_sets") else None
    digest_sets = replay["sets"] if replay else None
    if digest_sets is not None:
        assert Path(shutil.which("docker") or "").parent == HERE / "fixed_digest_lookup", "PATH: fixed_digest_lookup/"
        assert os.environ.get("IMAGES_JSON"), "fixed_digest_lookup needs IMAGES_JSON (setup_devbox's images.json)"
        # The sets were derived against one verified manifest: the wrapper must answer from that same one.
        replay["images_json_used"] = {"path": os.environ["IMAGES_JSON"], "sha256": rac.sha256(Path(os.environ["IMAGES_JSON"]))}
        assert replay["images_json_used"]["sha256"] == replay["images_json_sha256"], \
            f"IMAGES_JSON isn't the manifest {pilot['digest_sets']} was derived from ({replay['images_json']})"
        # One generated compose file per environment, so its tasks in one launch must replay the same set (tag form
        # for all, with config/digest_sets_tagform_*.json, lets every task of an environment share a launch).
        per_env = {}
        for t in tasks:
            per_env.setdefault(t["env"], set()).add(json.dumps(sorted(digest_sets.get(t["key"], []))))
        mixed = sorted(env for env, forms in per_env.items() if len(forms) > 1)
        assert not mixed, f"replay: these environments' tasks replay different lookup sets in one launch: {mixed}"
    lookup_env = lambda key: (  # noqa: E731
        {} if digest_sets is None else {"PHASE2_DIGEST_SET": json.dumps(digest_sets.get(key, []))})

    policy_env = {k: v for k, v in common(pilot["real_simulator"], "").items() if k.startswith("PT_POLICY_")}
    if pilot["simulators"]:  # simulated runs must get the real runs' policy limits
        assert {k: v for k, v in sim["env"].items() if k.startswith("PT_POLICY_")} == policy_env, "sim.yaml != run.yaml"

    def spec(r: dict) -> dict:
        """The command, environment and working directory of one run (a reference must be pinned for a simulated
        run first)."""
        t, name = r["task"], r["simulator"]
        tree = trees[name]
        attack = lambda: {"BA_MODE": "attack", "BA_CELLS": t["cell"], "PHASE2_CELLS": str(tasks_file)}  # noqa: E731
        if r["arm"] in ("honest_a", "honest_b"):
            env_vars = common(name, r["server"]) | {"LA_ENV": t["env"], "BA_MODE": "honest", "BA_MAINS": t["main"]}
            if tree["tree_defaults"]:  # unused in honest mode, but September's fleet set it: the header then matches
                env_vars["BA_ATTACKER_PROMPT"] = run["attacker_prompt"]
            task, flags = "src/runner/tasks/linuxarena_real.py", real_args  # the tree's own copy (cwd is the tree)
        elif r["arm"] == "attack":
            env_vars = common(name, r["server"]) | {"LA_ENV": t["env"], "BA_ATTACKER_PROMPT": run["attacker_prompt"]} \
                | attack()
            task, flags = f"{HERE / 'tasks.py'}@la_real_attack_sept", real_args
        else:
            ref = references[t["key"]]["copy"]
            env_vars = common(name, r["server"]) | sim["env"] | tree["env"] | {
                "LA_ENV": t["env"], "BA_REAL_LOG": ref, "BA_HONEST_LOG": ref}
            if r["arm"] == "sim_attack":
                env_vars |= attack()
                task = f"{HERE / 'tasks.py'}@la_sim_attack_sept"
            else:  # as the Astra honest batch: the same simulator settings, honest mode, the tree's own task
                env_vars |= {"BA_MODE": "honest"}
                task = "src/runner/tasks/linuxarena_sim.py"
            flags = sim_args
        return {"cmd": [str(tree["venv"] / "bin" / "inspect"), "eval", task], "flags": flags,
                "env_vars": env_vars | lookup_env(t["key"]), "cwd": str(tree["tree"])}

    runs, references = [], {}
    for t in tasks:
        server = None  # chosen when the task starts: the least-loaded server, for all of its runs
        for arm in (*real_arms, *(["honest_a"] if t["key"] in rerun_refs else [])):
            runs.append({"key": f"{arm}/{t['key']}", "arm": arm, "simulator": pilot["real_simulator"], "task": t,
                         "server": server, "state": "ready", "attempts": [], "real": True})
        for name in sims_for(t["key"]):
            for arm in SIMULATED:
                runs.append({"key": f"{arm}@{name}/{t['key']}", "arm": arm, "simulator": name, "task": t,
                             "server": server, "state": "waiting", "attempts": [], "real": False,
                             "time_control": t["key"] in tc_tasks and name == tc.get("simulator")})
    label = lambda r: r["arm"] if r["real"] else f"{r['arm']}@{r['simulator']}"  # noqa: E731

    from sim.prices import load_prices  # the September snapshot's prices (its venv runs this launcher)

    prices = load_prices()
    record = {"started": rac.now(), "finished": None, "probe_shift_commit": args.commit, "servers": servers,
              "replayed_lookups": replay,
              "pilot_config": args.pilot_config, "pilot": pilot, "policy": policy, "real_arms": real_arms,
              # an unpatched tree reads no PT_POLICY_*: its own defaults apply; max_tokens is only the cap-hit threshold
              "policy_limits": {"applied": "the tree's own defaults (no PT_POLICY_* set)",
                                "cap_hit_threshold_tokens": limits["max_tokens"]}
              if trees[pilot["real_simulator"]]["tree_defaults"] else limits,
              "tasks_file": str(tasks_file), "pilot_tasks": str(pilot_tasks), "tasks": tasks,
              "config": cfg | {"policy": policy}, "sim_config": sim,
              "trees": {k: {"tree": str(v["tree"]), "src_sha256": v["src_sha256"], "imported": v["imported"],
                            "env": v["env"]} for k, v in trees.items()},
              "input_sha256": {p: rac.sha256(HERE / p) for p in (
                  "config/run.yaml", "config/sim.yaml", args.pilot_config, pilot["tasks"], "tasks.py",
                  "run_attack_check.py", "run_pilot.py", "check_sim_samples.py", "check_timeouts_live.py",
                  "config/sept_snapshot_policy_limits.patch", "config/policy_strict_tools_off.patch",
                  *sorted({p for n in trees for p in cfg["simulators"][n].get("extra_patches", [])}))},
              "host": rac.host_fingerprint(settings, cfg | {"envs": sorted({t["env"] for t in tasks})}),
              "references": references, "spend": [], "runs": []}
    procs, last_launch, task_ref = [], {}, {t["key"]: {"tries": 0, "state": "pending"} for t in tasks}
    preflights = {}  # (task key, simulator) -> running preflight process
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith(("BA_", "PT_", "LA_", "SCE_", "INSPECT_"))}

    def start_preflights(key: str) -> None:
        """Check the task's new reference with each simulator's own code before its runs fork from it."""
        if not sims_for(key):  # no simulated run forks from it (an honest-only launch): nothing to check
            return
        refs = out / "references" / f"{key}.json"
        refs.write_text(json.dumps({key: references[key]["copy"]}))
        (out / "preflight").mkdir(exist_ok=True)
        if digest_sets is not None:  # the fork check reads the generated file before any run rewrites it: write it first
            tree, env = trees[sims_for(key)[0]], next(t["env"] for t in tasks if t["key"] == key)
            compose = settings / "linuxarena" / env / "codebase" / "compose.yml"
            subprocess.run([str(tree["venv"] / "bin" / "python"), "-c", "import sys; from pathlib import Path; from "
                            "control_tower.sandbox.docker import strip_build_sections as s; s(Path(sys.argv[1]))",
                            str(compose)], cwd=tree["tree"], check=True, timeout=600,
                           env=clean_env | common(sims_for(key)[0], "") | lookup_env(key))
        for name in sims_for(key):
            tree = trees[name]
            result = out / "preflight" / f"{key}_{name}.json"
            cmd = [str(tree["venv"] / "bin" / "python"), str(HERE / "check_sim_samples.py"), "--simulator", name,
                   "--tasks-file", str(tasks_file), "--references", str(refs), "--only", key, "--modes",
                   "attack,honest", "--out", str(result)]
            with open(out / "logs" / f"preflight_{key}_{name}.log", "w") as log:
                proc = subprocess.Popen(cmd, cwd=tree["tree"], env=clean_env | {"PYTHONPATH": str(tree["tree"] / "src")},
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            preflights[(key, name)] = (proc, time.time())
            procs.append({"proc": proc})
            task_ref[key].setdefault("preflight", {})[name] = {"state": "running", "result": str(result),
                                                               "start": rac.now()}

    def check_preflights() -> None:
        for (key, name), (proc, t0) in list(preflights.items()):
            if (code := proc.poll()) is None:
                if time.time() - t0 < PREFLIGHT_TIMEOUT_S:
                    continue
                rac._signal_group(proc, signal.SIGKILL)  # hung: its whole group, then judged as failed
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:  # recorded; the launch goes on, and stop_children retries at the end
                    task_ref[key]["preflight"][name]["kill_unconfirmed"] = True
                code = f"timed out after {PREFLIGHT_TIMEOUT_S} s"
            del preflights[(key, name)]
            ok = code == 0  # a timeout is a string, so it fails
            task_ref[key]["preflight"][name] |= {"state": "passed" if ok else "failed", "exit_code": code,
                                                 "end": rac.now()}
            print(f"{rac.now()} preflight {key} {name}: {'passed' if ok else f'FAILED (exit {code})'}", flush=True)
            for r in runs:
                if r["task"]["key"] == key and not r["real"] and r["simulator"] == name and r["state"] == "waiting":
                    r |= {"state": "ready"} if ok else {"state": "not run", "why": f"{name} preflight failed (exit {code})"}

    def save() -> None:
        record["runs"] = [{k: v for k, v in r.items() if k not in ("task", "proc")} | {"task": r["task"]["key"]}
                          for r in runs]
        record["reference_state"] = task_ref
        (out / "launch.json").write_text(json.dumps(record, indent=1, default=str))

    def start(r: dict, kind: str = "") -> None:
        n = sum(a.get("kind") == kind for a in r["attempts"]) if kind else 0
        suffix = f"_{kind}{n + 1}" if kind else ""
        log_dir = out / label(r) / f"{r['task']['key']}{suffix}"
        s = spec(r)
        cmd = [*s["cmd"], "--log-dir", str(log_dir), *s["flags"]]
        with open(out / "logs" / f"{label(r)}_{r['task']['key']}{suffix}.log", "w") as log:
            proc = subprocess.Popen(cmd, cwd=s["cwd"], env=os.environ | s["env_vars"], stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        if kind == "redraw":  # the attempt it replaces is superseded only now that the redraw has launched
            r["attempts"][-1]["redrawn_because"] = r["attempts"][-1]["prestart"]
        r["attempts"].append({"kind": kind or "first", "log_dir": str(log_dir), "start": rac.now(), "pid": proc.pid,
                              "cmd": cmd, "env_vars": s["env_vars"]})
        r |= {"state": "running", "proc": proc, "t0": time.time()}
        procs.append({"proc": proc, "rec": r})
        last_launch[r["task"]["env"]] = time.time()  # every launch: simulated runs write the compose file too
        save()  # on disk before the next launch
        print(f"{rac.now()} launched {label(r)} {r['task']['key']}{suffix} on server {servers.index(r['server'])} "
              f"(pid {proc.pid})", flush=True)

    def reap() -> None:
        for r in runs:
            if r["state"] != "running" or (code := r["proc"].poll()) is None:
                continue
            attempt = r["attempts"][-1]
            attempt |= {"end": rac.now(), "exit_code": code, "process_seconds": round(time.time() - r["t0"], 1)}
            log_dir = Path(attempt["log_dir"])
            try:
                why = rac.prestart_failure(log_dir, policy)
            except Exception as e:  # noqa: BLE001 — never let one run's log stop the others; never redrawn
                why = f"unknown: the pre-start check raised {e!r}"[:300]
            redraws = 0  # since this run's first try or its latest reference rerun
            for a in reversed(r["attempts"]):
                if a["kind"] != "redraw":
                    break
                redraws += 1
            attempt["prestart"] = why
            # Read even when it will be redrawn: if the launcher stops before the redraw starts, this attempt stays
            # the run's last and is reported with its outcome. It is marked superseded only once the redraw starts.
            attempt["outcome"] = run_outcome(log_dir, policy, run["max_steps"], limits["max_tokens"])
            # True on any policy event in the log. A log read and counted in full with no policy event: False. Else
            # (no readable sample, or its counting failed), the pre-start check (which reads Inspect's sample buffer) decides: None from it means
            # the buffer shows the policy started (True); "unknown: ..." can't tell (None); anything else is a
            # definite pre-start failure (False).
            out = attempt["outcome"]
            counted = out["samples"] and not out["unreadable"] and not out["errors"]  # read and counted in full
            attempt["policy_called"] = (True if called_policy(out) else False if counted
                                        else True if why is None else None if why.startswith("unknown") else False)
            a_full = r["arm"] == "honest_a" and len(r["attempts"]) >= pilot["reference_tries"]  # A: 3 starts in all
            if why and not why.startswith("unknown") and redraws < rac.MAX_PRESTART_REDRAWS and not a_full \
                    and not rac.STOP:
                print(f"{rac.now()} {label(r)} {r['task']['key']}: {why}; redrawing", flush=True)
                r["state"] = "ready"
                r["next_kind"] = "redraw"
                continue
            print(f"{rac.now()} {label(r)} {r['task']['key']} exited {code}: clean={attempt['outcome']['clean']}",
                  flush=True)
            # A run whose last attempt never called the policy (no completed or failed call: a pre-start failure with
            # its redraws spent, or no readable sample) is not "done": viewers and counts of finished runs must not
            # take it for one (10-08 audit: all 32 sim@today attempts were recorded as done with zero calls).
            r["state"] = "no policy call" if attempt["policy_called"] is False else "done"
            if r["arm"] == "honest_a":
                try:
                    reference_done(r, attempt)
                except Exception as e:  # noqa: BLE001 — recorded; the task's simulated runs can't fork, the rest go on
                    task_ref[r["task"]["key"]] |= {"state": "error", "error": repr(e)[:300]}
                    for other in runs:
                        if other["task"]["key"] == r["task"]["key"] and not other["real"] and other["state"] == "waiting":
                            other |= {"state": "not run", "why": "the reference couldn't be checked or pinned"}
        save()

    def reference_done(r: dict, attempt: dict) -> None:
        """Pin a usable honest_a as its task's reference and release the simulated runs, or rerun it."""
        key, ref = r["task"]["key"], task_ref[r["task"]["key"]]
        ref["tries"] = len(r["attempts"])  # starts so far, redraws included: A gets reference_tries in all
        usable = attempt["outcome"]["usable_reference"]
        if usable:
            try:
                (source,) = sorted(Path(attempt["log_dir"]).glob("*.eval"))
                copy = out / "references" / f"{key}.eval"
                shutil.copy2(source, copy)
                references[key] = {"source": str(source), "copy": str(copy), "sha256": rac.sha256(copy),
                                   "try": ref["tries"]}
            except Exception as e:  # noqa: BLE001 — couldn't pin it: treated as not usable, like any other failure
                ref.setdefault("errors", []).append(f"pinning try {ref['tries']}: {e!r}"[:300])
                usable = False
        if usable:
            ref["state"] = "usable"
            start_preflights(key)  # the simulated runs wait for their simulator's preflight
        elif len(r["attempts"]) < pilot["reference_tries"] and not rac.STOP:  # every start counts, redraws too
            ref.setdefault("unusable", []).append(attempt["log_dir"])
            print(f"{rac.now()} honest_a {key} is not a usable reference (try {ref['tries']}); rerunning", flush=True)
            r["state"], r["next_kind"] = "ready", "rerun"
        else:
            ref |= {"state": "none usable"}
            ref.setdefault("unusable", []).append(attempt["log_dir"])
            for other in runs:
                if other["task"]["key"] == key and not other["real"]:
                    other |= {"state": "not run", "why": f"no usable reference in {ref['tries']} tries"}

    def spend() -> dict:
        per_run = {}
        for r in runs:
            if r["real"] or not r["attempts"]:
                continue
            total = 0.0
            for a in r["attempts"]:  # a finished attempt is priced once, from its log; a running one from its buffer
                try:
                    if "sim_usd" not in a:
                        usd = run_spend(Path(a["log_dir"]), policy, prices)
                        if usd is not None:  # nothing readable (a vanished buffer): keep the last reading
                            a["live_usd"] = usd
                            if "end" in a:  # finished and read: priced once, from its log
                                a["sim_usd"] = usd
                    a["live_usd"] = a.get("sim_usd", a.get("live_usd", 0.0))
                except Exception as e:  # noqa: BLE001 — a read race or a half-written log: keep the last reading
                    a.setdefault("spend_read_errors", []).append(repr(e)[:200])
                total += a.get("live_usd", 0.0)
            per_run[r["key"]] = total
        done = [per_run[r["key"]] for r in runs if not r["real"] and r["state"] == "done"]
        est = max(pilot["spend"]["prior_per_sim_run_usd"], sum(done) / len(done) if done else 0.0)
        started = {r["task"]["key"] for r in runs if r["attempts"]}
        pending = sum(1 for r in runs if not r["real"] and r["task"]["key"] in started
                      and r["state"] in ("waiting", "ready"))
        running = [per_run.get(r["key"], 0.0) for r in runs if not r["real"] and r["state"] == "running"]
        actual = sum(per_run.values())  # every attempt so far, stopped ones included
        # A finished attempt not yet priced from its log (only a live reading, maybe none) counts at the estimate.
        # An ended attempt that never called the policy (policy_called False) reserves nothing more: no simulator
        # call can follow; whatever it did spend is already in `actual` (GitHub Codex on #245).
        unpriced = [a.get("live_usd", 0.0) for r in runs if not r["real"] for a in r["attempts"]
                    if "end" in a and "sim_usd" not in a and a.get("policy_called") is not False]
        projected = actual + sum(max(0.0, est - x) for x in running + unpriced) + pending * est
        worst = max(pilot["spend"]["worst_per_sim_run_usd"], max(done, default=0.0))
        worst_case = actual + sum(max(0.0, worst - x) for x in running + unpriced) + sum(
            max(0.0, worst - per_run.get(r["key"], 0.0)) for r in runs if not r["real"] and r["task"]["key"] in started
            and r["state"] in ("waiting", "ready"))
        snap = {"time": rac.now(), "actual_usd": round(actual, 2), "projected_usd": round(projected, 2),
                "per_run_estimate_usd": round(est, 2), "pending_simulated_runs": pending,
                "finished_unpriced_attempts": len(unpriced), "worst_case_usd": round(worst_case, 2),
                "worst_per_sim_run_usd": round(worst, 2),
                "next_task_fits": projected + 2 * len(rerun_sims) * est <= pilot["spend"]["cap_usd"]
                and worst_case + 2 * len(rerun_sims) * worst <= pilot["spend"]["worst_case_limit_usd"]}
        snap["_projected"], snap["_est"], snap["_worst_case"], snap["_worst"] = projected, est, worst_case, worst
        record["spend"].append({k: v for k, v in snap.items() if not k.startswith("_")})
        return snap

    # An API policy's spend (pilot `policy_stop_usd`, real-only): every start, redraws included, waits for
    # policy_spend_watch.py's reading (under 3 min old). It goes ahead only if the reading's committed spend, plus the
    # worst case of every attempt the reading hasn't seen and of the new one, stays under the stop. Otherwise, or with
    # no fresh reading for 5 min (the watcher is down: POLICY_HALT, final), nothing more starts, and the launch ends
    # once its running processes do (pending redraws are recorded as not run).
    stop_usd = pilot.get("policy_stop_usd")
    assert stop_usd is None or all(r["real"] and r["arm"] == "honest_b" for r in runs), \
        "policy_stop_usd: real-only honest_b launches (policy_spend_watch.py prices <run>/honest_b only)"
    t_launch = time.time()

    def policy_fits() -> bool:
        if stop_usd is None:
            return True
        if (out / "POLICY_HALT").exists():
            return False
        try:
            rd = json.loads((out / "policy_spend.json").read_text())
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(rd["time"])).total_seconds()
        except (OSError, ValueError, KeyError):
            rd, age = None, float("inf")
        if rd is not None and rd["stop_usd"] != stop_usd:
            (out / "POLICY_HALT").write_text(json.dumps({"why": "the watcher's stop differs", "reading": rd}))
            return False
        if age > 180:
            if age > 300 and time.time() - t_launch > 300:
                (out / "POLICY_HALT").write_text(json.dumps({"why": "no fresh policy spend reading", "reading": rd}))
            return False
        seen = set(rd["attempts_seen"])  # attempt folders the reading priced (each running one at its worst)
        unseen = sum(Path(a["log_dir"]).name not in seen and not ("end" in a and a.get("policy_called") is False)
                     for r in runs for a in r["attempts"])
        if rd["committed_usd"] + (unseen + 1) * rd["worst_per_run_usd"] <= stop_usd:
            return True
        (out / "POLICY_STOP").write_text(json.dumps(rd | {"unseen_attempts": unseen}))
        return False

    def read_after_last_end() -> bool:
        """Whether the watcher's latest reading came after every attempt's end and reserves nothing for a running one
        (so a just-ended run's worst case no longer holds budget)."""
        try:
            rd = json.loads((out / "policy_spend.json").read_text())
            t0 = datetime.fromisoformat(rd["time"])
        except (OSError, ValueError, KeyError):
            return False
        ends = [datetime.fromisoformat(a["end"]) for r in runs for a in r["attempts"] if "end" in a]
        return rd.get("running") == 0 and (not ends or t0 > max(ends))

    def launch(snap: dict) -> None:
        """Start what capacity allows. Started tasks first, in order; then new tasks strictly in the file's order (a
        new task waits until every earlier one has started), each only if the projection with it stays under the cap."""
        busy = Counter(r["server"] for r in runs if r["state"] == "running")
        real_busy = sum(1 for r in runs if r["state"] == "running" and r["real"])
        projected, per_task = snap["_projected"], 2 * len(rerun_sims) * snap["_est"]  # the larger set: conservative
        worst_case, worst_task = snap["_worst_case"], 2 * len(rerun_sims) * snap["_worst"]
        # The canary (Aswin, 10-08): until canary_watch.py writes <out>/CANARY_OK after a real str_replace of the
        # canary task passed, only that task's real arms may start; it stops the launcher on one without new_str.
        gate_closed = bool(pilot.get("canary")) and not (out / "CANARY_OK").exists()
        started_keys = {r["task"]["key"] for r in runs if r["attempts"]}
        # Started tasks first (the canary may have started out of order), then new ones in the file's order: a sort
        # keeps each group's order, so the spend guard's `break` on a new task never strands a started one.
        for t in sorted(tasks, key=lambda t: t["key"] not in started_keys):
            if gate_closed and t["key"] != pilot["canary"]:
                continue
            mine = [r for r in runs if r["task"]["key"] == t["key"]]
            if not any(r["attempts"] or r["state"] in ("ready", "waiting") for r in mine):
                continue  # nothing of it will ever run (e.g. a reused reference missing or failing its preflight)
            new = not any(r["attempts"] for r in mine)
            if new and (projected + per_task > pilot["spend"]["cap_usd"]
                        or worst_case + worst_task > pilot["spend"]["worst_case_limit_usd"]):
                break  # the guard: no new task from here on
            if new:  # the least-loaded server takes the whole task
                server = min(servers, key=lambda x: busy[x])
                for r in mine:
                    r["server"] = server
            for r in mine:
                if r["state"] != "ready" or rac.STOP or busy[r["server"]] >= pilot["max_runs_per_server"]:
                    continue
                if gate_closed and not r["real"]:
                    continue  # the canary task's simulated runs wait for the gate too
                if time.time() - last_launch.get(t["env"], float("-inf")) < run["stagger_s"]:
                    continue  # one environment's launches are stagger_s apart, simulated ones included
                if r["real"] and real_busy >= pilot["max_real_runs"]:
                    continue
                if not policy_fits():
                    break  # the API policy's stop (above): nothing more starts until it fits
                start(r, r.pop("next_kind", ""))
                busy[r["server"]] += 1
                real_busy += r["real"]
            if new:
                if not any(r["attempts"] for r in mine):
                    for r in mine:
                        r["server"] = None
                    break  # it couldn't start yet (capacity or stagger): no later task may start before it
                projected += per_task
                worst_case += worst_task

    for sig in rac.TERMINATING:
        signal.signal(sig, rac._note_signal)
    save()
    # Every server must pass the server-start checks (check_timeouts_live.py: tiny, repetition, newline) before
    # anything launches; they run with the real arms' tree, as the runs' policy does.
    tree = trees[pilot["real_simulator"]]
    record["server_checks"] = {}  # (clean_env: defined above, for the preflights)
    for i, server in enumerate(servers):
        if not checks:  # `server_checks: []`: a launch with no live server (the canary abort test, $0)
            record["server_checks"][server] = {"exit_code": 0, "output_tail": "skipped: no checks configured",
                                               "summary": None}
            continue
        if rac.STOP:  # a stop signal (Ctrl+C, SIGTERM, SIGHUP) skips the remaining servers' checks (#101's Codex P2)
            record["server_checks"][server] = {"exit_code": "skipped: stopping", "output_tail": "", "summary": None}
            continue
        summary = out / f"server_checks_{i}.json"
        cmd = [str(tree["venv"] / "bin" / "python"), str(HERE / "check_timeouts_live.py"), *checks,
               "--policy", policy, "--summary", str(summary)]
        check_log = out / "logs" / f"server_checks_{i}.log"  # a file, not a pipe: a chatty check can't block on it
        with open(check_log, "w") as log:  # no Popen context manager: its exit waits without a bound
            proc = subprocess.Popen(cmd, cwd=tree["tree"], stdout=log, stderr=subprocess.STDOUT,
                                    env=clean_env | {"PYTHONPATH": str(tree["tree"] / "src"), "VLLM_BASE_URL": server})
            deadline = time.time() + 900
            while proc.poll() is None and not rac.STOP and time.time() < deadline:
                time.sleep(1)
            ended_by = None
            if proc.poll() is None:  # stopping, or past 900 s: end the check now rather than let it run on
                ended_by = "stopped" if rac.STOP else "timed out after 900 s"
                for stop in (proc.terminate, proc.kill):
                    stop()
                    try:
                        proc.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                else:
                    ended_by += "; still alive 30 s after SIGKILL"
        code = ended_by or proc.returncode
        tail = check_log.read_text()[-1500:]
        record["server_checks"][server] = {"exit_code": code, "output_tail": tail,
                                           "summary": json.loads(summary.read_text()) if summary.exists() else None}
        print(f"{rac.now()} server {i} checks: {'passed' if code == 0 else f'FAILED ({code})'}", flush=True)
    save()
    if rac.STOP:  # a stop during the checks ends the launch here, before references are copied or preflighted
        raise SystemExit(f"stopped during the server-start checks ({', '.join(str(x) for x in rac.STOP)})")
    failed = [s for s, v in record["server_checks"].items() if v["exit_code"] != 0]
    if failed:
        raise SystemExit(f"server-start checks failed for {failed}; nothing launched (see launch.json)")
    if args.references_from:  # pin the prior run's references, checked against its record, and preflight them
        prior = json.loads((args.references_from.expanduser() / "launch.json").read_text())["references"]
        record["references_from"] = str(args.references_from)
        for t in tasks:
            key = t["key"]
            if key in rerun_refs:  # a fresh honest_a runs in this launch; its simulated runs wait for it as usual
                task_ref[key]["prior_reference_not_used"] = prior.get(key, {}).get("copy")
                continue
            if key not in prior:
                for r in runs:
                    if r["task"]["key"] == key and not r["real"]:
                        r |= {"state": "not run", "why": f"no pinned reference in {args.references_from}"}
                task_ref[key]["state"] = "none in the prior run"
                continue
            copy = out / "references" / f"{key}.eval"
            shutil.copy2(prior[key]["copy"], copy)
            assert rac.sha256(copy) == prior[key]["sha256"], f"{key}: the prior reference doesn't match its sha256"
            references[key] = prior[key] | {"copy": str(copy), "reused_from": prior[key]["copy"]}
            task_ref[key] |= {"state": "usable", "reused_from": str(args.references_from)}
            start_preflights(key)
        save()
    try:
        while not rac.STOP:
            if (out / "CANARY_ABORT").exists():  # canary_watch.py saw a real str_replace without new_str: stop it all
                record["canary_abort"] = (out / "CANARY_ABORT").read_text()[:2000]
                print(f"{rac.now()} CANARY ABORT: a real str_replace without new_str; stopping every run", flush=True)
                rac.STOP.append("canary abort")
                break
            reap()
            check_preflights()
            snap = spend()
            if snap["actual_usd"] > pilot["spend"]["worst_case_limit_usd"] and not record.get("over_limit"):
                print(f"{rac.now()} OVER LIMIT: simulator spend ${snap['actual_usd']} passed "
                      f"${pilot['spend']['worst_case_limit_usd']}; started tasks still finish", flush=True)
                record["over_limit"] = snap  # the worst case was too low: reported, nothing is stopped
            launch(snap)
            started = {r["task"]["key"] for r in runs if r["attempts"]}
            active = [r for r in runs if r["state"] == "running"
                      or (r["state"] in ("waiting", "ready") and r["task"]["key"] in started)]
            unstarted = [r for r in runs if r["state"] in ("waiting", "ready") and r["task"]["key"] not in started]
            if stop_usd is not None and not any(r["state"] == "running" for r in runs) and (
                    (out / "POLICY_HALT").exists()
                    or ((out / "POLICY_STOP").exists() and read_after_last_end() and not policy_fits())):
                for r in runs:  # nothing runs and nothing more may start: pending runs and redraws end here
                    if r["state"] in ("waiting", "ready"):
                        r |= {"state": "not run", "why": "policy spend stop: never started"}
                break
            if not active and (not unstarted or not snap["next_task_fits"]):
                for r in unstarted:  # tasks the guard never let start
                    r |= {"state": "not run", "why": "spend cap: the task never started"}
                break
            print(f"{rac.now()} running {sum(r['state'] == 'running' for r in runs)}, done "
                  f"{sum(r['state'] == 'done' for r in runs)} of {len(runs)}; simulator ${snap['actual_usd']} "
                  f"(projected ${snap['projected_usd']})", flush=True)
            rac._sleep(30)
    finally:
        if (out / "CANARY_ABORT").exists() and not record.get("canary_abort"):  # stopped by the watcher's Ctrl+C first
            record["canary_abort"] = (out / "CANARY_ABORT").read_text()[:2000]
        rac.stop_children(procs)  # preflights included
        for (key, name) in list(preflights):
            task_ref[key]["preflight"][name] |= {"state": "stopped", "end": rac.now()}
        for r in runs:
            if r["state"] == "running":  # kept with what they reached, and in the diagnostics
                attempt = r["attempts"][-1]
                attempt |= {"end": rac.now(), "exit_code": r["proc"].poll(), "stopped_by_launcher": True}
                try:
                    attempt["outcome"] = run_outcome(Path(attempt["log_dir"]), policy, run["max_steps"],
                                                     limits["max_tokens"])
                    attempt["policy_called"] = called_policy(attempt["outcome"]) or None  # stopped: None = can't tell
                except Exception as e:  # noqa: BLE001 — recorded; the other runs' records still get written
                    attempt["outcome_error"] = repr(e)[:300]
                r["state"] = "stopped"
            elif r["state"] in ("ready", "waiting"):
                r |= {"state": "not run", "why": f"launcher stopped ({', '.join(str(x) for x in rac.STOP)})"}
        try:
            spend()  # the final reading, stopped runs included
        except Exception as e:  # noqa: BLE001
            record["final_spend_error"] = repr(e)[:300]
        record["stopped_by_signal"] = [str(s) for s in rac.STOP]
        save()
        signal.signal(signal.SIGINT, signal.default_int_handler)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_DFL)

    summarize(runs, record, servers, label)
    save()
    print(f"-> {out}", flush=True)


def summarize(runs: list, record: dict, servers: list, label) -> None:
    """Guard stops from each server's proxy, then per-arm totals and decision 5's flags."""
    # Every attempt that ran (not one superseded by a pre-start redraw): honest_a's tries are runs too.
    finals = [(r, a["outcome"]) for r in runs for a in r["attempts"] if a.get("outcome") and not a.get("redrawn_because")]
    reasons = {}
    for server in servers:
        if server == "none":  # an API policy (--base-urls none): no vLLM proxy to ask
            reasons[server] = None
            record["stop_reasons_not_applicable"] = "no policy server: an API policy"
            continue
        ids = sorted({i for r, o in finals if o and r["server"] == server for i, _ in o["call_ids"] if i})
        try:
            reasons[server] = rac.stop_reasons(server, ids)
        except Exception as e:  # noqa: BLE001 — recorded, never read as zero stops
            reasons[server] = None
            record.setdefault("stop_reasons_errors", {})[server] = repr(e)[:300]
    per_arm = {}
    for r, o in finals:
        if not o:
            continue
        rep = rac.repetition_stops(o["call_ids"], reasons[r["server"]]) if reasons[r["server"]] is not None else None
        o["repetition"] = rep
        a = per_arm.setdefault(label(r), Counter())
        a["runs"] += 1
        for c in o["policy_calls"]:
            a["steps"] += c["steps"]
            a["runs_step_limit"] += c["step_limit_hits"]
            a["runs_time_limit"] += c["ended"] == "time limit"
            a["runs_with_capped_turn"] += c["cap_hits"] > 0
            a["runs_with_timeout"] += c["timeouts"] > 0
            a["runs_failed_after_start"] += c["failed_after_start"]
            a["runs_usable"] += c["usable"]
            a["runs_ended_submit"] += c["ended"] == "submit"
            a["runs_out_of_context"] += c["ended"] == "out_of_context"  # a failure the sample doesn't show (the guard)
        a["runs_with_guard_stop"] += bool(rep and rep["repetition_stops"])
        if rep is None:  # the proxy couldn't be asked: the token log's completeness is unknown for these runs
            a["runs_proxy_unknown" if r["server"] != "none" else "runs_without_proxy"] += 1
        else:  # every policy call must join a token record (request ID known to the proxy)
            a["calls_unknown_to_proxy"] += rep["calls_unknown_to_proxy"]
            a["calls_without_request_id"] += rep["calls_without_request_id"]
            a["calls_without_complete_token_record"] += rep["calls_without_complete_token_record"]
        a["runs_without_sample"] += o["samples"] == 0
        a["runs_with_counting_errors"] += bool(o.get("errors"))  # its counts are missing, not zero

    for r in runs:  # once per run, not per attempt
        if r["attempts"]:
            a = per_arm.setdefault(label(r), Counter())
            a["prestart_redraws"] += sum(x["kind"] == "redraw" for x in r["attempts"])
            a["reference_reruns"] += sum(x["kind"] == "rerun" for x in r["attempts"])
    record["by_arm"] = {k: dict(v) for k, v in per_arm.items()}
    record["not_run"] = Counter(label(r) for r in runs if r["state"] == "not run")
    flags = []
    rate = lambda arm, k: per_arm[arm][k] / per_arm[arm]["runs"] if per_arm.get(arm, {}).get("runs") else None  # noqa: E731
    keys = ("runs_step_limit", "runs_time_limit", "runs_with_capped_turn", "runs_with_timeout",
            "runs_failed_after_start", "runs_with_guard_stop", "runs_without_sample", "runs_out_of_context")
    for arm, a in per_arm.items():
        gaps = ("calls_unknown_to_proxy", "calls_without_request_id", "calls_without_complete_token_record",
                "runs_proxy_unknown")
        if a["runs_with_counting_errors"]:
            flags.append(f"{arm}: diagnostics incomplete ({a['runs_with_counting_errors']} runs whose counts failed; "
                         f"see each attempt's outcome errors)")
        if any(a[k] for k in gaps):
            flags.append(f"{arm}: token log incomplete ({', '.join(f'{k} {a[k]}' for k in gaps)})")
    for arm in per_arm:
        for k in keys:
            if (x := rate(arm, k)) is not None and x > 0.10:
                flags.append(f"{arm}: {k} {x:.0%} (over 10%)")
            real = {"sim_honest": "honest_b", "sim_attack": "attack"}.get(arm.split("@")[0])
            if real and real in per_arm and (x := rate(arm, k)) is not None and (y := rate(real, k)) is not None:
                big = max(per_arm[arm][k], per_arm[real][k]) >= 3 and (min(x, y) == 0 or max(x, y) / min(x, y) >= 2)
                if abs(x - y) > 0.10 or big:
                    flags.append(f"{arm} vs {real}: {k} {x:.0%} vs {y:.0%}")
    record["flags"] = flags
    record["finished"] = rac.now()
    for arm, totals in record["by_arm"].items():
        print(f"{arm}: {json.dumps(totals)}", flush=True)
    for f in flags:
        print(f"FLAG {f}", flush=True)


if __name__ == "__main__":
    main()
