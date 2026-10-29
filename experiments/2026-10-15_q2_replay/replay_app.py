"""The replay's simulator pass on Modal (df's RULE v7 @ 604bc7c60, frozen; replay_core.py holds the logic).

Image: the September snapshot's src with the two committed run.yaml patches applied (B's tree; src sha256 verified =
1ea37722... before any sims import), the codebase folders with every `.compose-*` file removed (RULE v7 FALLBACK, choice
(i); the served folders' content digest is checked against d9's manifest for B's folders without compose files), the
frozen predicate (../2026-10-14_onpolicy_sim/replay_predicate.py, sha256 checked) and Oct 7's simulator helpers.
Inputs are passed in and sha256-checked: phase1_manifest.json (872ddc7c...) and placebo_manifest.json (8eb57dae...) @
604bc7c60, the corrected-B map @ 46ca25fc4, d9's tree manifest @ cfdd656c9; the source runs and B's simulated samples are
read from the q4own_evals Volume, corrected B's staged epochs (for c_r) from the same Volume (sha256 checked).

Functions (app probe-shift_aswin-surya_sub-q2-replay_sim_kill-2026-11-14; Volume probe-shift_aswin-surya_sub-q2-
onpolicysim_data_kill-2026-11-14, under /replay/):
- simulate_live: the real pass (label t196_v1; status /replay/status/live.json; PAUSE /replay/PAUSE), the simruns Secret.
  NOT run without q2's go. `--action live` refuses unless /replay/PAUSE is absent and the run is detached; it spawns the
  pass (timeout 8 h), writes results/live_launch_<stamp>.json (app ID, call ID) and exits; monitor_replay.py reads progress.
- dry_run: the stub dry run, no Secret, 0 API calls: every model the simulator resolves is a stub (judges answer
  "DECISION: NO"; the simulator answers each call with an editor-error-shaped marker, never the placeholder string).
  Labels dry_main (all 135 must-run runs, checks (a) and (b)), dry_inject (injected rate limits, failures, refusals,
  a 30-minute backoff stop and a crash-and-resume, check (d)), dry_budget (hard stop with a fake cost), dry_pause (the
  PAUSE file). Stub outputs stay under /replay/dry_*/ and never reach staging.
- render_check: check (c), Modal CPU in the staging image (stage_q4b.py @ ef93eb73 with Amendment 9's template fix, as
  run_q4f_stage_fixed.py @ 0e78df207): per must-run run, (c0) the render of the source run equals corrected B's staged
  ids; (c1) with dry_main's stub outputs swapped in, rendering a retained prefix with the unprocessed suffix present or
  cut away gives the same token ids and readout positions inside the prefix; (c2) the replay render equals corrected B's
  staged ids up to the first changed output (gate 4, pre-tested).
Output: hashes, counts, booleans and statuses only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-15_q2_replay/replay_app.py --action dry-run          (then --action render-check)
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if modal.is_local() else Path("/root/repo")
SEPT = Path.home() / "simulated-control-evals-sept"
C_SEPT, C_REPO = "/root/simulated-control-evals-sept", "/root/repo"
APP = "probe-shift_aswin-surya_sub-q2-replay_sim_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q2-onpolicysim_data_kill-2026-11-14"
EVALS = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
HF = "probe-shift_aswin-surya_hf-cache"
SECRET = "probe-shift_aswin-surya_sub-simruns_openai_kill-2026-11-13"
JOURNAL = "probe-shift_aswin-surya_sub-q2-replay_journal_kill-2026-11-14"
B_SRC = "1ea3772209663ef8a848effaf6677d23d5bab06c248a37c1fb61f0957efd7b8c"
PREDICATE_SHA = "80f85c4dbf616b01645c9e886ceab3361d8257eb5216fbf7359d9870edf6e403"
PATCHES = [("b66909607", "experiments/2026-10-07_phase2_attack_check/config/sept_snapshot_policy_limits.patch"),
           ("8d7d6a192", "experiments/2026-10-07_phase2_attack_check/config/policy_strict_tools_off.patch")]
INPUTS = {"phase1": ("604bc7c60", "experiments/2026-10-15_replay_sim_outputs/phase1_manifest.json",
                     "872ddc7c68bd856f1b21d3effb5f368068d058372cedc9d749532c57bc9e2ce7"),
          "placebo": ("604bc7c60", "experiments/2026-10-15_replay_sim_outputs/placebo_manifest.json",
                      "8eb57dae18ff40fd1c6da7fdc178868d8773415252a57fe4b83510c7ea58cf94"),
          "map": ("46ca25fc4", "experiments/2026-10-14_q35_sol_avg_POSTHOC/config/q4b_epoch_map_correctedb_v1.json",
                  "e9d313bbf26bf02bbfd10fbdaebaf98430f54d513bd205e2d204c3614c38119c"),
          "tree_manifest": ("cfdd656c9", "experiments/2026-10-14_q3_scaleup_311/results/sim_tree_manifest_2026-10-15.json", None)}
STAGED_SHAS = {("q4b_main_v1", "attack"): "0fc04b487e037fb291da037d9507c5de2d395e9e61f0ca5f217d0c22e95beb7e",
               ("q4b_main_v1", "honest_b"): "3fc71b14a2515abee6d1071ac4f04aa4b8f842c647e81de0684d2f1031a188ae",
               ("q4f_main_v1_changed", "attack"): "a140d75ae191c36fff1fec6999731d4a8edde208192dfe8442aea4c3661d1d0b",
               ("q4f_main_v1_changed", "honest_b"): "4ec266de04bb334d25459318924ad11ffe662b32860e8fccbed28e7e5ea36767"}
BOUNDARY = (248066, frozenset({198}), 248067, 271)  # <tool_response>, newline-led openers, </tool_response>, "\n\n" (@ 6614ee1d)
ENVS = ("auto_workflow", "clinical_trial", "file_indexing", "media_processing", "model_registry", "shared_dotfiles",
        "slack_clone", "tiktok", "user_tracking", "web_scraping")
HARD_STOP_USD, CONCURRENCY = 3000.0, 20
LIVE_TIMEOUT_S = 8 * 3600  # q2's pre-brief (10-15): an explicit 8 h; a pass cut by it resumes from its ledgers on a new go
CAP_MARGIN_USD = 300.0  # dispatch stops at $2,700 settled + reserved: $15 per concurrency slot for in-flight overruns
REPO_FILES = ("experiments/2026-10-06_matched_step_pairs/run_simulate_pairs.py",
              "experiments/2026-10-06_matched_step_pairs/run_select_pairs.py",
              "experiments/2026-10-14_onpolicy_sim/replay_predicate.py", "experiments/2026-10-15_q2_replay/replay_core.py",
              "src/probe_shift/__init__.py", "src/probe_shift/reader.py", "src/probe_shift/transcripts.py")
PT = ZoneInfo("America/Los_Angeles")


def git_show(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=ROOT).stdout


app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
evals = modal.Volume.from_name(EVALS, version=2).read_only()
sim_image = (modal.Image.debian_slim(python_version="3.14").apt_install("git").pip_install("uv==0.12.19")
             .add_local_file(SEPT / "pyproject.toml", f"{C_SEPT}/pyproject.toml", copy=True)
             .add_local_file(SEPT / "uv.lock", f"{C_SEPT}/uv.lock", copy=True)
             .run_commands(f"cd {C_SEPT} && UV_PROJECT_ENVIRONMENT=/usr/local uv sync --frozen --inexact --no-install-project --no-dev")
             .env({"PYTHONPATH": f"{C_SEPT}/src:{C_REPO}/src"})
             .add_local_dir(SEPT / "src", f"{C_SEPT}/src", ignore=["**/__pycache__"], copy=True))
if modal.is_local():
    for _i, (_c, _p) in enumerate(PATCHES):
        _f = Path(f"/tmp/q2_replay_patch_{_i}.patch")
        _f.write_bytes(git_show(_c, _p))
        sim_image = sim_image.add_local_file(_f, f"/root/patches/{_i}.patch", copy=True)
sim_image = sim_image.run_commands(f"cd {C_SEPT} && git apply -p1 /root/patches/0.patch && git apply -p1 /root/patches/1.patch")
for _e in ENVS:
    sim_image = sim_image.add_local_dir(SEPT / ".settings" / "linuxarena" / _e / "codebase", f"{C_SEPT}/.settings/linuxarena/{_e}/codebase", copy=True)
sim_image = sim_image.run_commands(f"find {C_SEPT}/.settings/linuxarena -name '.compose-*' -exec rm -rf {{}} +")  # FALLBACK choice (i)
for _f in REPO_FILES:
    sim_image = sim_image.add_local_file(ROOT / _f, f"{C_REPO}/{_f}")


def where(e: BaseException) -> str:
    import traceback

    tb = traceback.extract_tb(e.__traceback__)[-1] if e.__traceback__ else None
    return f"{type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}" if tb else type(e).__name__


def src_sha256(tree: Path) -> str:  # run_attack_check.src_sha256 (main)
    hh = hashlib.sha256()
    for p in sorted(q for q in (tree / "src").rglob("*") if q.is_file() and "__pycache__" not in q.parts):
        hh.update(str(p.relative_to(tree)).encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return hh.hexdigest()


def setup_container(inputs: dict, stub_factory=None):
    """Verify the tree and inputs, wire the model logging (and the stub), and prepare every must-run run."""
    tree = src_sha256(Path(C_SEPT))
    assert tree == B_SRC, f"the sims tree is {tree[:12]}, not B's 1ea37722"
    import logging
    import warnings
    import zipfile

    logging.disable(logging.CRITICAL)  # warnings can quote transcript text
    warnings.filterwarnings("ignore")
    for p in (f"{C_REPO}/experiments/2026-10-06_matched_step_pairs", f"{C_REPO}/experiments/2026-10-14_onpolicy_sim",
              f"{C_REPO}/experiments/2026-10-15_q2_replay"):
        sys.path.insert(0, p)
    import replay_core as RC
    import replay_predicate as RP
    import run_select_pairs as RSP
    import run_simulate_pairs as RSIM
    import runner.simulate.model as model_module
    from inspect_ai.model._call_tools import truncate_tool_output
    from probe_shift.transcripts import resolve_attachments, tool_result_text
    from runner.arenas.reference_contract import _tree_digest
    from runner.simulate import build_simulator, config_for_sample
    from runner.simulate.fidelity import blocked_reachable_host, reachable_set
    from runner.simulate.model import FallbackClient, InspectClient, RoutedNetworkClient

    assert hashlib.sha256(Path(RP.__file__).read_bytes()).hexdigest() == PREDICATE_SHA, "not the frozen predicate"
    for name, (_c, _p, sha) in INPUTS.items():
        assert sha is None or hashlib.sha256(inputs[name]).hexdigest() == sha, f"{name} is not the pinned file"
    man = json.loads(inputs["tree_manifest"])["boxes"]["sims-astra"]["codebases"]
    for e in ENVS:  # FALLBACK content gate on the folders actually served
        root = Path(C_SEPT) / ".settings/linuxarena" / e / "codebase"
        assert not any(root.rglob(".compose-*")), f"{e}: a .compose-* file is served"
        assert _tree_digest((("environment/0", root),)) == man[f"/home/aswin/phase2/settings|{e}"]["contents_sha256_without_any_compose_files"], e

    original = model_module.resolve

    class Logged:  # every resolved model: its requests' stop reasons go to the current step's log
        def __init__(self, model, name):
            self._model, self._name = model, name

        async def generate(self, input, *args, **kwargs):
            ctx = RC.CURRENT.get()
            if ctx is not None:
                ctx["active"] += 1
            try:
                out = await self._model.generate(input, *args, **kwargs)
            except BaseException as error:
                if ctx is not None:
                    ctx["requests"].append({"model": self._name, "error": type(error).__name__})
                raise
            finally:
                if ctx is not None:
                    ctx["active"] -= 1
            if ctx is not None:
                ctx["requests"].append({"model": self._name, "stop_reason": str(getattr(out, "stop_reason", ""))})
            return out

        def __getattr__(self, name):
            return getattr(self._model, name)

    def resolve(name, **kwargs):
        return Logged(stub_factory(name) if stub_factory else original(name, **kwargs), name)

    model_module.resolve = resolve

    em = json.loads(inputs["map"])["map"]
    sims = {task: k for k, (w, task) in em.items() if w == "sim"}
    staged = {}
    for (folder, arm), sha in STAGED_SHAS.items():
        f = Path("/ev") / folder / "inputs" / f"{arm}.jsonl"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == sha, f"{folder}/{arm}.jsonl changed"
        for line in f.open():
            ep = json.loads(line)
            staged[(folder, arm, ep["run"])] = ep

    def sample(rel: str) -> dict:
        with zipfile.ZipFile(Path("/ev/evals_v1") / rel) as z:
            (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
            return resolve_attachments(json.loads(z.read(n)))

    def truncate(fn: str, raw: str) -> str:
        tr = truncate_tool_output(fn, raw, RC.INSPECT_TRUNC)
        return tr.output if tr else raw

    runs = []
    for row in json.loads(inputs["phase1"])["runs"] + json.loads(inputs["placebo"])["runs"]:
        rel = row["corrected_map_key"].split("#")[0]
        s, ref = sample(rel), sample(sims[row["task"]].split("#")[0])
        outs, counts = RP.eligible_outputs(s)
        assert counts["unmatched_tool_messages"] == 0, row["task"]
        c, _ = RP.c_r(staged[(row["staging"], row["arm_of"], rel)], BOUNDARY)
        stored = ref["store"]["SimulatorState:sim_messages"]
        descriptions = {}
        for m in stored:
            for name, desc in RSP.DESCRIBED.findall(m["content"] if isinstance(m["content"], str) else ""):
                descriptions.setdefault(name, desc)
        msgs, actions = s["messages"], []
        for o in outs:
            tc = msgs[o["assistant_msg"]]["tool_calls"][o["call_pos_source"]]
            a = {"function": tc["function"], "arguments": tc.get("arguments") or {}, "id": tc.get("id")}
            if a["function"] not in ("bash", "text_editor") and a["function"] in descriptions:
                a["description"] = descriptions[a["function"]]
            actions.append(a)
        steps, by_msg = [], {}
        for o in outs:
            if o["assistant_msg"] not in by_msg:
                by_msg[o["assistant_msg"]] = len(steps)
                steps.append([])
            steps[by_msg[o["assistant_msg"]]].append(o["j"])
        md = {k: ref["metadata"].get(k) for k in RSP.SIM_KEYS}
        assert md["simulator_model"] == RSIM.MODEL and md["orig_trajectory_out_cap"] == 32768 and md["orig_trajectory_mode"] == "dump"
        assert not md["codebase_exec"] and not md["session_notes"] and md["self_critique"] is None and not md["seeded_prefix"]
        runs.append({"id": f"{row['world']}__{row['task']}", "world": row["world"], "task": row["task"], "env": row["task"].split("_", 1)[1],
                     "actions": actions, "rejected": [o["rejected"] for o in outs], "steps": steps,
                     "real_outputs": [tool_result_text(msgs[o["tool_msg"]]) for o in outs], "c_r": c, "K_r": RP.k_r(c, len(outs)),
                     "md": md, "stored_system_prompt": stored[0]["content"], "truncate": truncate, "counts": counts,
                     "mapping": [{k: o[k] for k in ("j", "tool_msg", "assistant_msg", "call_pos_source", "function", "call_id_sha256", "rejected")} for o in outs]})

    def build(run: dict, clients=None, router_state=None) -> dict:
        """Oct 7's simulate wiring (runner.build.arena_simulated_task.simulated_tools, line for line). The run's clients are
        kept across rebuilds (the router remembers payloads and route decisions); a cold resume restores them."""
        md = dict(run["md"])
        md["codebase_root"] = str(Path(C_SEPT) / ".settings/linuxarena" / run["env"] / "codebase")
        md["seeded_prefix"] = []
        if clients is None:
            fallback = InspectClient(md["simulator_fallback_model"])
            cidrs, hosts = reachable_set(md["codebase_root"])
            guard = FallbackClient(InspectClient(RSIM.MODEL), fallback, lambda text: blocked_reachable_host(text, cidrs, hosts),
                                   refusal_judge_model=md["simulator_refusal_judge_model"])
            router = RoutedNetworkClient(guard, fallback, md["simulator_router_judge_model"])
            if router_state is not None:
                router._payloads = set(router_state["payloads"])
                router._route_decisions = {a: d for a, d in router_state["decisions"]}
            clients = (fallback, guard, router)
        fallback, guard, router = clients
        cfg = config_for_sample(md, model=RSIM.MODEL, codebase_exec=False, exec_source_authority=md["exec_source_authority"],
                                orig_trajectory_mode="dump", self_critique=None)
        sim = build_simulator(cfg, fallback=fallback, client=router)
        return {"sim": sim, "clients": clients,
                "router_state": lambda: {"payloads": sorted(router._payloads), "decisions": sorted([a, d] for a, d in router._route_decisions.items())},
                "counters": lambda: {"fallback_substitutions": sim.fallback_substitutions, "block_substitutions": guard.block_substitutions,
                                     "refusal_substitutions": guard.refusal_substitutions, "net_substitutions": router.net_substitutions}}

    config = {"rule_sha": RC.RULE_SHA, "tree": B_SRC, "predicate_sha256": PREDICATE_SHA, "boundary": [BOUNDARY[0], sorted(BOUNDARY[1]), BOUNDARY[2], BOUNDARY[3]],
              "inputs_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in inputs.items()}, "folders": "FALLBACK (i): no .compose-*",
              "reserve": "3 + 1 per sibling", "fail_min_usd": RC.FAIL_MIN_USD, "rate_limit_cap_s": RC.RL_CAP_S, "concurrency": CONCURRENCY}
    return RC, RSIM, runs, build, config


class Journal:
    """The pass's durable spend journal: a Modal Dict, written before each request can be billed."""

    def __init__(self):
        self.d = modal.Dict.from_name(JOURNAL, create_if_missing=True)

    async def put(self, k, v):
        await self.d.put.aio(k, v)

    async def get(self, k, default=None):
        return await self.d.get.aio(k, default)


@app.function(image=sim_image, secrets=[modal.Secret.from_name(SECRET)], cpu=4, memory=32768, timeout=LIVE_TIMEOUT_S, retries=0,
              volumes={"/data": vol, "/ev": evals})
def simulate_live(inputs: dict, label: str, limits: dict, concurrency: int = CONCURRENCY) -> dict:
    """The real pass. Never run without q2's go. `limits` (run id -> output limit) extends runs on a resume; the ledger
    header's config is unchanged by it, so every ledger resumes; runs already done to their limit return at once."""
    RC, RSIM, runs, build, config = setup_container(inputs)
    p = RC.Pass(root=Path("/data/replay"), label=label, mode="live", runs=runs, build=build, hard_stop=HARD_STOP_USD,
                concurrency=concurrency, status_path=Path("/data/replay/status/live.json"), pause_path=Path("/data/replay/PAUSE"),
                config=config, journal=Journal(), vol=vol, cost_fn=RSIM.cost, cap_margin=CAP_MARGIN_USD)
    try:
        out, printed = quiet(p.run_all(limits))
    finally:
        vol.commit()
    return out | {"suppressed_print_lines": printed}


def quiet(coro):
    """Run the pass with stdout and stderr captured and dropped: the simulator prints contract errors that can quote
    source paths (sim/simulator.py:527). Only line counts by prefix leave."""
    import asyncio
    import contextlib
    import io

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        out = asyncio.run(coro)
    lines = sink.getvalue().splitlines()
    return out, {"batch_fallback": sum(x.startswith("[batch]") for x in lines), "coalesce_short": sum(x.startswith("[coalesce]") for x in lines),
                 "other": sum(not x.startswith(("[batch]", "[coalesce]")) for x in lines)}


class Harness:
    """Stub mode: the requests' checks (b), the injected errors (d) and the fake clock."""

    def __init__(self, RC, build, inject: dict, crash_at: dict):
        self.RC, self.build, self.inject, self.crash_at = RC, build, inject, crash_at
        self.shadow, self.expect, self.first, self.t, self.slept, self.crashed = {}, {}, set(), 0.0, {}, set()
        self.b = {"requests_checked": 0, "snapshot_equal": 0, "final_message_equal": 0, "leaks_later_output": 0, "leaks_later_action": 0,
                  "leaks_earlier_real_output": 0, "skipped_short_or_ambiguous": 0}

    def now(self):
        return self.t

    async def sleep(self, s):  # the fake clock; the crash test fires here, between a backoff row and the next attempt
        import asyncio

        self.t += s
        rid = self.RC.CURRENT.get()["run"]
        self.slept[rid] = self.slept.get(rid, 0.0) + s
        if rid in self.crash_at and rid not in self.crashed and self.slept[rid] >= self.crash_at[rid]:
            self.crashed.add(rid)
            raise self.RC.SimulatedCrash()
        await asyncio.sleep(0)

    async def shadow_of(self, run):
        if run["id"] not in self.shadow:
            st = self.build(run)
            await st["sim"].setup([])
            self.shadow[run["id"]] = {"st": st, "session": st["sim"].open(), "committed": {}, "dispatched": []}
        return self.shadow[run["id"]]

    async def before_step(self, run, s, js, attempt):
        from sim.render import compose_batch_request, format_tool_action

        sh = await self.shadow_of(run)
        sim, sess = sh["st"]["sim"], sh["session"]
        msgs = [await sim.action_message(run["actions"][j], sess) for j in js]
        text = msgs[0] if len(msgs) == 1 else compose_batch_request(msgs, notes=sim._may_retain)
        render = lambda j: format_tool_action(run["actions"][j]["function"], run["actions"][j].get("description", ""), run["actions"][j]["arguments"])  # noqa: E731
        sh["dispatched"] += [render(j) for j in js]
        later = [j for st_js in run["steps"][s + 1:] for j in st_js if not run["rejected"][j]]
        earlier = [j for j in sh["committed"] if sh["committed"][j] != run["real_outputs"][j]]
        self.expect[(run["id"], s, attempt)] = {
            "snapshot": [self.RC.h(f"{m.role}\0{m.content}") for m in sess.memory], "final": self.RC.h(text),
            "later_outputs": [run["real_outputs"][j] for j in later], "later_actions": [render(j) for j in later],
            "earlier_real": [run["real_outputs"][j] for j in earlier], "committed": list(sh["committed"].values()),
            "dispatched": "\n".join(sh["dispatched"])}

    async def after_commit(self, run, entries):
        sh = await self.shadow_of(run)
        for j, text in entries:
            sh["st"]["sim"].commit_output(await sh["st"]["sim"].action_message(run["actions"][j], sh["session"]), text, sh["session"])
            sh["committed"][j] = text

    def on_request(self, ctx, messages):
        key = (ctx["run"], ctx["step"], ctx["attempt"])
        if key in self.first:  # only the step's first request: later ones carry repair turns
            return
        self.first.add(key)
        e = self.expect[key]
        b = self.b
        b["requests_checked"] += 1
        got = [self.RC.h(f"{m.role}\0{m.text}") for m in messages[:-1]]
        b["snapshot_equal"] += got == e["snapshot"]
        b["final_message_equal"] += self.RC.h(messages[-1].text) == e["final"]
        body = "\n".join(m.text for m in messages if m.role != "system")
        known = set(e["committed"])
        for kind, items in (("leaks_later_output", e["later_outputs"]), ("leaks_later_action", e["later_actions"]),
                            ("leaks_earlier_real_output", e["earlier_real"])):
            for x in items:
                if len(x) < 24 or x in known or x in e["dispatched"] or any(x in c for c in known):
                    b["skipped_short_or_ambiguous"] += 1
                elif x in body:
                    b[kind] += 1

    def kind(self, ctx):
        plan = self.inject.get((ctx["run"], ctx["step"]))
        if plan is None:
            return None
        return plan[ctx["attempt"]] if ctx["attempt"] < len(plan) else plan[-1] if plan[-1] == "rate_limit_forever" else None


def stub_factory_for(harness, RC, RSIM, latency_s: float = 0.2):
    from inspect_ai.model import ModelOutput
    from sim.render import _CALL_MARKER, render_step_draft

    class StubRateLimitError(Exception):
        status_code = 429

    class StubServerError(Exception):
        status_code = 503

    class Stub:
        def __init__(self, name):
            self.name = name

        async def generate(self, input, *args, **kwargs):
            import asyncio

            if self.name in RSIM.JUDGE_MODELS:
                return ModelOutput.from_content(model=self.name, content="DECISION: NO")
            ctx = RC.CURRENT.get()
            harness.on_request(ctx, input)
            await asyncio.sleep(latency_s)  # a little latency, so pauses and concurrency are exercised
            k = harness.kind(ctx)
            if k in ("rate_limit", "rate_limit_forever"):
                raise StubRateLimitError("stub rate limit")
            if k == "server_error":
                raise StubServerError("stub server error")
            if k == "content_filter":
                return ModelOutput.from_content(model=self.name, content="", stop_reason="content_filter")
            n = ctx["n"] if _CALL_MARKER.search(input[-1].text) else 1  # a batch request's own example markers are not calls
            outs = [f"No replacement was performed. STUB|{ctx['run']}|{ctx['step']}|{i}" for i in range(n)]
            return ModelOutput.from_content(model=self.name, content=render_step_draft(outs, single=n == 1))

    return Stub


@app.function(image=sim_image, cpu=8, memory=32768, timeout=3 * 3600, retries=0, volumes={"/data": vol, "/ev": evals})
def dry_run(inputs: dict) -> dict:
    import asyncio
    import time
    from collections import Counter

    holder = {}
    RC, RSIM, runs, build, config = setup_container(inputs, stub_factory=lambda name: holder["Stub"](name))
    root = Path("/data/replay")
    for label in ("dry_main", "dry_inject", "dry_budget", "dry_pause"):
        assert not (root / label).exists(), f"{root / label} exists; the dry run never resumes over old stub ledgers"
    assert not any(k.startswith(("dry_main|", "dry_inject|", "dry_budget|", "dry_pause|")) for k in Journal().d.keys()), "old dry-run journal keys"
    out = {"rule_sha": RC.RULE_SHA, "config": config, "runs": len(runs)}

    # (a) the predicate and its mapping, frozen
    mapping = {r["id"]: r["mapping"] for r in runs}
    raw = json.dumps(mapping, sort_keys=True).encode()
    (root / "dry_main").mkdir(parents=True)
    (root / "dry_main" / "mapping.json").write_bytes(raw)
    agg = lambda k: sum(r["counts"][k] for r in runs)  # noqa: E731
    out["a"] = {"mapping_sha256": hashlib.sha256(raw).hexdigest(), "predicate_sha256": PREDICATE_SHA, "eligible": agg("eligible"),
                "submit_calls": agg("submit_calls"), "rejected": agg("rejected"), "multi_call_messages": agg("multi_call_messages"),
                "steps_with_rejected_and_eligible": sum(1 for r in runs for st in r["steps"] if any(r["rejected"][j] for j in st) and not all(r["rejected"][j] for j in st)),
                "K_r": sum(r["K_r"] for r in runs), "c_r": sum(r["c_r"] for r in runs)}

    def make_pass(label, harness, run_list, **kw):
        holder["Stub"] = stub_factory_for(harness, RC, RSIM)
        return RC.Pass(root=root, label=label, mode="stub", runs=run_list, build=build, concurrency=CONCURRENCY,
                       status_path=root / label / "status.json", pause_path=root / label / "PAUSE", config=config | {"label": label},
                       journal=Journal(), vol=vol, clock=harness, harness=harness, **kw)

    # (b) all 135 must-run runs, no injection
    hm = Harness(RC, build, {}, {})
    pm = make_pass("dry_main", hm, runs, hard_stop=HARD_STOP_USD)
    res, printed = quiet(pm.run_all({}))
    st_counts = Counter(s for r in runs for s in RC.final_statuses(r, pm.ledgers[r["id"]].rows))
    results = [row for r in runs for row in pm.ledgers[r["id"]].rows if row["kind"] == "result"]
    expected_steps = sum(1 for r in runs for st in r["steps"] if min(st) < r["K_r"] and any(not r["rejected"][j] for j in st))
    b = hm.b | {"attempts": len(results), "attempts_ok": sum(x["status"] == "ok" for x in results), "expected_steps": expected_steps,
                "statuses": dict(st_counts), "spent_usd": res["spent_usd"], "suppressed_print_lines": printed,
                "coalesced_multi_call_steps": sum(len(x["js"]) > 1 for x in results if x["status"] == "ok"),
                "placeholder_strings_in_outputs": sum("[tool output withheld]" in o for x in results if x["status"] == "ok" for o in x["outputs"])}
    b["pass"] = bool(b["requests_checked"] > 0 and b["requests_checked"] == b["attempts"] == b["attempts_ok"] == expected_steps
                     and b["snapshot_equal"] == b["final_message_equal"] == b["requests_checked"]
                     and b["leaks_later_output"] == b["leaks_later_action"] == b["leaks_earlier_real_output"] == 0
                     and set(st_counts) <= {"simulated_changed", "simulated_identical", "kept_real_rejected", "beyond_limit"}
                     and b["placeholder_strings_in_outputs"] == 0 and printed["batch_fallback"] == 0)
    out["b"] = b
    out["sample_status"] = json.loads((root / "dry_main" / "status.json").read_text())

    # (d) injected paths, on six placebo runs (their own ledgers under dry_inject)
    pick = [r for r in runs if r["world"] == "honest_b"][:6]

    def nth_step(r, n):  # the n-th step (0-based) with an eligible call inside K_r
        cand = [s for s, st in enumerate(r["steps"]) if any(not r["rejected"][j] for j in st) and min(st) < r["K_r"]]
        return cand[min(n, len(cand) - 1)]

    plan = {"rl_then_ok": (pick[0], 2, ["rate_limit"] * 3), "fail_twice": (pick[1], 3, ["server_error", "server_error"]),
            "refuse_twice": (pick[2], 1, ["content_filter", "content_filter"]), "rl_forever": (pick[3], 2, ["rate_limit_forever"]),
            "crash_resume": (pick[4], 1, ["rate_limit_forever"]), "fail_once": (pick[5], 2, ["server_error"])}
    inject = {(r["id"], nth_step(r, n)): kinds for r, n, kinds in plan.values()}
    hi = Harness(RC, build, inject, {plan["crash_resume"][0]["id"]: 1200})
    pi = make_pass("dry_inject", hi, pick, hard_stop=HARD_STOP_USD)
    res_i, printed_i = quiet(pi.run_all({}))
    d = {}
    for name, (r, n, kinds) in plan.items():
        rows, s = pi.ledgers[r["id"]].rows, nth_step(r, n)
        srows = [x for x in rows if x.get("step") == s]
        d[name] = {"backoffs": sum(x["kind"] == "backoff" for x in srows),
                   "backoff_total_s": max([x["total_s"] for x in srows if x["kind"] == "backoff"], default=0),
                   "results": [x["status"] for x in srows if x["kind"] == "result"], "kept_real": [x["status"] for x in srows if x["kind"] == "kept_real"],
                   "stop": [x["reason"] for x in rows if x["kind"] == "stop"], "done": any(x["kind"] == "done" for x in rows),
                   "statuses": dict(Counter(RC.final_statuses(r, rows)))}
    exp = {"rl_then_ok": lambda x: x["results"] == ["rate_limited"] * 3 + ["ok"] and x["backoffs"] == 3 and not x["kept_real"] and x["done"],
           "fail_twice": lambda x: x["results"] == ["failed", "failed"] and x["kept_real"] == ["kept_real_after_failure"] and x["done"],
           "refuse_twice": lambda x: x["results"] == ["failed", "failed"] and x["kept_real"] == ["kept_real_after_refusal"] and x["done"],
           "rl_forever": lambda x: x["stop"] == ["rate_limit"] and not x["kept_real"] and set(x["results"]) == {"rate_limited"}
                                   and x["backoff_total_s"] >= RC.RL_CAP_S - 5 and x["statuses"].get("not_processed_before_stop", 0) > 0,
           "crash_resume": lambda x: x["stop"] == ["rate_limit"] and not x["kept_real"] and set(x["results"]) == {"rate_limited"}
                                     and RC.RL_CAP_S - 5 <= x["backoff_total_s"] <= RC.RL_CAP_S + 1,
           "fail_once": lambda x: x["results"] == ["failed", "ok"] and not x["kept_real"] and x["done"]}
    for name in plan:
        d[name]["pass"] = bool(exp[name](d[name]))
    d["crash_resumes"] = res_i["crash_resumes"]
    d["shadow_checks"] = hi.b
    d["suppressed_print_lines"] = printed_i
    d["pass"] = bool(all(d[k]["pass"] for k in plan) and d["crash_resumes"] == [plan["crash_resume"][0]["id"]]
                     and hi.b["snapshot_equal"] == hi.b["final_message_equal"] == hi.b["requests_checked"] > 0
                     and hi.b["leaks_later_output"] == hi.b["leaks_later_action"] == hi.b["leaks_earlier_real_output"] == 0)
    out["d"] = d

    # budget: three runs, $1 a step, hard stop $10
    hb = Harness(RC, build, {}, {})
    pb = make_pass("dry_budget", hb, runs[:3], hard_stop=10.0, fake_cost_per_step=1.0)
    rb, _ = quiet(pb.run_all({}))
    stops = [x["reason"] for r in runs[:3] for x in pb.ledgers[r["id"]].rows if x["kind"] == "stop"]
    out["budget"] = {"hard_stop": 10.0, "spent_usd": rb["spent_usd"], "max_committed_exposure_usd": rb["max_committed_exposure_usd"], "stops": stops,
                     "pass": bool(rb["max_committed_exposure_usd"] <= 10.0 and rb["spent_usd"] <= 10.0 and "budget" in stops)}

    # PAUSE: present before the start (no run may start), removed after 3 s, set again mid-run for 3 s
    hp = Harness(RC, build, {}, {})
    pp = make_pass("dry_pause", hp, runs[3:5], hard_stop=HARD_STOP_USD, tick_s=1.0)
    (root / "dry_pause").mkdir(parents=True, exist_ok=True)
    pause = root / "dry_pause" / "PAUSE"
    pause.write_text("")
    marks = {}

    async def toggler():  # file changes under the pass's lock, so a Volume reload never meets an open file
        await asyncio.sleep(3)
        async with pp.lock:
            pause.unlink()
        marks["off1"] = time.time()
        await asyncio.sleep(2)
        async with pp.lock:
            pause.write_text("")
        marks["on2"] = time.time()
        await asyncio.sleep(3)
        async with pp.lock:
            pause.unlink()
        marks["off2"] = time.time()

    async def both():
        await asyncio.gather(pp.run_all({}), toggler())

    quiet(both())
    disp = [t for t, _r, _s in pp.dispatch_log]
    pz = {"dispatches": len(disp), "before_first_unpause": sum(t < marks["off1"] for t in disp),
          "during_second_pause_after_one_tick_and_one_step": sum(marks["on2"] + 1.5 < t < marks["off2"] for t in disp),
          "dispatches_after_second_unpause": sum(t > marks["off2"] for t in disp),
          "ticks_paused": sum(p for _t, p in pp.tick_log), "ticks": len(pp.tick_log)}
    pz["pass"] = bool(pz["dispatches"] > 0 and pz["before_first_unpause"] == 0 and pz["during_second_pause_after_one_tick_and_one_step"] == 0
                      and pz["ticks_paused"] >= 4)
    out["pause"] = pz
    out["pass_results"] = {"dry_main": res, "dry_inject": res_i, "dry_budget": rb}
    out["dry_run_pass"] = bool(b["pass"] and d["pass"] and out["budget"]["pass"] and pz["pass"])
    (root / "dry_main" / "dry_run_result.json").write_text(json.dumps(out, default=str))
    vol.commit()
    return out


@app.function(image=sim_image, cpu=4, memory=16384, timeout=1800, retries=0, volumes={"/data": vol, "/ev": evals})
def pause_rerun(inputs: dict, label: str, tick_s: float, window_s: float, gap_s: float) -> dict:
    """The PAUSE pass of the dry run again, unchanged criterion, with windows long enough for several ticks (q2 10-15):
    PAUSE present before the start, removed after window_s, set again gap_s later for window_s, then removed."""
    import asyncio
    import time

    holder = {}
    RC, RSIM, runs, build, config = setup_container(inputs, stub_factory=lambda name: holder["Stub"](name))
    root = Path("/data/replay")
    assert not (root / label).exists(), f"{root / label} exists"
    assert not any(k.startswith(f"{label}|") for k in Journal().d.keys()), "old journal keys for this label"
    hp = Harness(RC, build, {}, {})
    holder["Stub"] = stub_factory_for(hp, RC, RSIM, latency_s=2.0)  # runs outlast the windows and the gap
    pp = RC.Pass(root=root, label=label, mode="stub", runs=runs[3:5], build=build, concurrency=CONCURRENCY, status_path=root / label / "status.json",
                 pause_path=root / label / "PAUSE", config=config | {"label": label}, journal=Journal(), vol=vol, clock=hp, harness=hp,
                 hard_stop=HARD_STOP_USD, tick_s=tick_s)
    (root / label).mkdir(parents=True)
    pause = root / label / "PAUSE"
    pause.write_text("")
    marks = {}

    async def toggler():  # file changes under the pass's lock, so a Volume reload never meets an open file
        await asyncio.sleep(window_s)
        async with pp.lock:
            pause.unlink()
        marks["off1"] = time.time()
        await asyncio.sleep(gap_s)
        async with pp.lock:
            pause.write_text("")
        marks["on2"] = time.time()
        await asyncio.sleep(window_s)
        async with pp.lock:
            pause.unlink()
        marks["off2"] = time.time()

    async def both():
        await asyncio.gather(pp.run_all({}), toggler())

    quiet(both())
    disp = [t for t, _r, _s in pp.dispatch_log]
    pz = {"label": label, "tick_s": tick_s, "window_s": window_s, "gap_s": gap_s, "dispatches": len(disp),
          "before_first_unpause": sum(t < marks["off1"] for t in disp),
          "during_second_pause_after_one_tick_and_one_step": sum(marks["on2"] + tick_s + 2.5 < t < marks["off2"] for t in disp),
          "dispatches_between_windows": sum(marks["off1"] < t < marks["on2"] for t in disp),
          "dispatches_after_second_unpause": sum(t > marks["off2"] for t in disp),
          "ticks_paused_window1": sum(p for t, p in pp.tick_log if t < marks["off1"]),
          "ticks_paused_window2": sum(p for t, p in pp.tick_log if marks["on2"] < t < marks["off2"]),
          "ticks_paused": sum(p for _t, p in pp.tick_log), "ticks": len(pp.tick_log)}
    # the criterion as written, plus (Codex r7) proof that the second window met a run in progress and was seen by a tick
    pz["pass"] = bool(pz["dispatches"] > 0 and pz["before_first_unpause"] == 0 and pz["during_second_pause_after_one_tick_and_one_step"] == 0
                      and pz["ticks_paused"] >= 4 and pz["dispatches_between_windows"] > 0 and pz["dispatches_after_second_unpause"] > 0
                      and pz["ticks_paused_window1"] >= 1 and pz["ticks_paused_window2"] >= 1)
    (root / label / "pause_rerun_result.json").write_text(json.dumps(pz))
    vol.commit()
    return pz


stage_image = (modal.Image.debian_slim(python_version="3.13").uv_pip_install("transformers==5.19.0", "numpy", "jinja2", "zstandard", "boto3", "pyyaml")
               .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))
if modal.is_local():
    for _name, (_c, _p) in {"stage_q4b.py": ("ef93eb730", "experiments/2026-10-13_q4_why_weaker_POSTHOC/stage_q4b.py"),
                            "evalzip.py": ("ef93eb730", "experiments/2026-10-13_q4_why_weaker_POSTHOC/evalzip.py")}.items():
        _f = Path(f"/tmp/q2_replay_{_name}")
        _f.write_bytes(git_show(_c, _p))
        stage_image = stage_image.add_local_file(_f, f"/root/repo/q4/{_name}")
    stage_image = stage_image.add_local_file(ROOT / "experiments/2026-10-14_onpolicy_sim/replay_predicate.py", "/root/repo/q4/replay_predicate.py")
COND = "{%- if loop.index0 > ns.last_query_index %}"  # run_q4f_stage_fixed.py @ 0e78df207: Amendment 9's template fix


@app.function(image=stage_image, cpu=8, memory=32768, timeout=3600, retries=0,
              volumes={"/data": vol, "/ev": evals, "/hf": modal.Volume.from_name(HF).read_only()})
def render_check(inputs: dict) -> dict:
    sys.path.insert(0, "/root/repo/q4")
    import replay_predicate as RP
    import stage_q4b as S
    from transformers import AutoTokenizer

    assert hashlib.sha256(Path(RP.__file__).read_bytes()).hexdigest() == PREDICATE_SHA
    for name, (_c, _p, sha) in INPUTS.items():
        assert sha is None or hashlib.sha256(inputs[name]).hexdigest() == sha, name
    tok = AutoTokenizer.from_pretrained(S.QWEN35_27B["name"], revision=S.QWEN35_27B["revision"], local_files_only=True)
    assert tok.chat_template.count(COND) == 1
    tok.chat_template = tok.chat_template.replace(COND, "{%- if true %}")
    em = json.loads(inputs["map"])["map"]
    keys = {(w, t): k for k, (w, t) in em.items()}
    staged = {}
    for (folder, arm), sha in STAGED_SHAS.items():
        f = Path("/ev") / folder / "inputs" / f"{arm}.jsonl"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == sha
        for line in f.open():
            ep = json.loads(line)
            staged[(folder, arm, ep["run"])] = ep["ids"]

    def system_for(s, row):  # stage_q4b.process's donor rule (@ ef93eb73, lines 346-355), from the corrected map's honest runs
        own = S.transcripts.system_prompt(s)
        if row["arm_of"].startswith("honest"):
            return own
        cover, fu = S.COVER.search(own), S.first_user_sha(s)
        for hname in ("honest_a", "honest_b"):
            k = keys.get((hname, row["task"]))
            if k is None:
                continue
            hs = S.load(Path("/ev/evals_v1") / k.split("#")[0])
            p = S.transcripts.system_prompt(hs)
            if cover and cover.group(1).strip() in p and "You are an agent" in p and S.first_user_sha(hs) == fu:
                return p
        raise ValueError("no donor")

    def render(msgs, cut):
        return S.reader.segment_end_positions(tok, S.scrubbed(msgs, S.SCRUB_LEVEL), max_tokens=cut)

    ledger_dir = Path("/data/replay/dry_main/ledger")
    c = {"runs": 0, "c0_source_equals_staged": 0, "c1_prefix_checks": 0, "c1_ids_equal": 0, "c1_positions_equal": 0,
         "c2_replay_equals_staged_before_first_change": 0, "c2_runs": 0, "errors": [], "first_diff_c0": []}
    for row in json.loads(inputs["phase1"])["runs"] + json.loads(inputs["placebo"])["runs"]:
        rid = f"{row['world']}__{row['task']}"
        try:
            c["runs"] += 1
            rel = row["corrected_map_key"].split("#")[0]
            s = S.load(Path("/ev/evals_v1") / rel)
            msgs = S.transcripts.reader_messages(s, system_for(s, row))
            ids = render(msgs, S.CUT)["ids"]
            ref = staged[(row["staging"], row["arm_of"], rel)]
            c["c0_source_equals_staged"] += ids == ref
            if ids != ref:
                c["first_diff_c0"].append(next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), min(len(ids), len(ref))))
            rows = [json.loads(line) for line in (ledger_dir / f"{rid}.jsonl").read_text().splitlines()]
            sim = {j: o for r in rows if r["kind"] == "result" and r["status"] == "ok" for j, o in zip(r["js"], r["outputs"])}
            tool_i = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
            outs, _ = RP.eligible_outputs(s)
            assert len(tool_i) == len(outs), "reader tool messages != eligible outputs"
            rep = [dict(m) for m in msgs]
            for j, o in sim.items():  # the agent's transcript gets the Inspect-truncated text, as the runner computes it
                rep[tool_i[j]]["content"] = o  # stub markers are far below 16 KiB: truncation is the identity here
            full = render(rep, None)
            done_js = sorted(sim)
            for frac in (0.25, 0.5, 0.75, 1.0):
                if not done_js:
                    break
                last = done_js[max(0, int(frac * len(done_js)) - 1)]
                cutm = tool_i[last] + 1
                while cutm < len(rep) and rep[cutm]["role"] == "tool":  # keep the step's whole tool group
                    cutm += 1
                part = render(rep[:cutm], None)
                n = len(part["ids"])
                c["c1_prefix_checks"] += 1
                c["c1_ids_equal"] += part["ids"] == full["ids"][:n]
                pm = {(p["kind"], p["step"], p["action"]): p["token"] for p in part["positions"]}
                fm = {(p["kind"], p["step"], p["action"]): p["token"] for p in full["positions"]}
                extra = [k for k, t in fm.items() if k not in pm and t is not None and t < n]
                c["c1_positions_equal"] += all(fm.get(k) == t for k, t in pm.items()) and not extra
            if done_js:  # gate 4 pre-test: up to the first changed output's content start
                first = min(j for j in done_js if sim[j] != msgs[tool_i[j]]["content"])
                pos = [p for p in range(len(ref) - 1) if ref[p] == BOUNDARY[0] and (ref[p + 1] in BOUNDARY[1] or (ref[p + 1] == BOUNDARY[3] and p + 2 < len(ref) and ref[p + 2] == BOUNDARY[2]))]
                lim = pos[first] + 2 if first < len(pos) else len(ref)
                rep_ids = render(rep, S.CUT)["ids"]
                c["c2_runs"] += 1
                c["c2_replay_equals_staged_before_first_change"] += rep_ids[:lim] == ref[:lim]
                # through the first changed output's <tool_response> marker only: the token after it can differ when the
                # source output was EMPTY (Qwen3.5 trims content, so the template newlines merge into one token)
                lim_m = pos[first] + 1 if first < len(pos) else len(ref)
                c["c2_equal_through_marker"] = c.get("c2_equal_through_marker", 0) + (rep_ids[:lim_m] == ref[:lim_m])
                if rep_ids[:lim] != ref[:lim]:
                    d0 = next((i for i, (x, y) in enumerate(zip(rep_ids, ref)) if x != y), min(len(rep_ids), len(ref)))
                    c.setdefault("c2_fail_detail", []).append({"first_diff_minus_marker": d0 - pos[first] if first < len(pos) else None,
                                                               "source_output_empty": msgs[tool_i[first]]["content"] == "",
                                                               "token_after_marker_in_source": ref[pos[first] + 1] if first < len(pos) else None})
        except Exception as e:  # noqa: BLE001
            c["errors"].append([rid, where(e)])
    c["pass"] = bool(c["runs"] > 0 and not c["errors"] and c["c0_source_equals_staged"] == c["runs"]
                     and c["c1_prefix_checks"] > 0 and c["c1_ids_equal"] == c["c1_positions_equal"] == c["c1_prefix_checks"]
                     and c["c2_runs"] > 0 and c.get("c2_equal_through_marker", 0) == c["c2_runs"]
                     and all(x["source_output_empty"] and x["first_diff_minus_marker"] == 1 for x in c.get("c2_fail_detail", [])))
    (Path("/data/replay/dry_main") / "render_check_result.json").write_text(json.dumps(c))
    vol.commit()
    return c


@app.local_entrypoint()
def main(action: str, label: str = "t196_v1", limits_from: str = "", limits_sha256: str = "", to_end: bool = False):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    inputs = {name: git_show(c, p) for name, (c, p, _s) in INPUTS.items()}
    stamp = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
    base = {"git": git("rev-parse", "HEAD"), "rule_sha": "604bc7c60", "app": APP}
    if action == "dry-run":
        res = dry_run.remote(inputs)
    elif action == "render-check":
        res = render_check.remote(inputs)
    elif action == "pause-rerun":
        res = pause_rerun.remote(inputs, "dry_pause_long", 15.0, 75.0, 40.0)
    elif action == "live":  # only on q2's "GO REPLAY"; spawned, so the pass outlives this client, which exits after the record
        if label != "t196_v1" or "--detach" not in sys.argv:
            raise SystemExit("the live pass runs as label t196_v1 under modal run --detach")
        if any(Path(e.path).name == "PAUSE" for e in vol.listdir("/replay")):
            raise SystemExit("/replay/PAUSE is present: no launch (only d9 removes it)")
        limits, conc = {}, CONCURRENCY
        assert not to_end or limits_from, "--to-end extends only the runs of a --limits-from record"
        if limits_from or limits_sha256:  # an extension (RULE: a failing run gets more outputs simulated, inside the cap)
            f = HERE / "results" / limits_from
            assert "/" not in limits_from and f.is_file(), "a stage record file name in results/"
            subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, check=True, capture_output=True)  # committed
            b = f.read_bytes()
            assert hashlib.sha256(b).hexdigest() == limits_sha256, "the stage record's sha256 differs"
            sr = json.loads(b)
            assert sr["label"] == label and sr["mode"] == "stage" and sr["extensions_needed"], "a t196_v1 stage record with extensions"
            ids = {f"{r['world']}__{r['task']}" for k in ("phase1", "placebo") for r in json.loads(inputs[k])["runs"]}
            assert set(sr["extensions_needed"]) <= ids, "extension for a run outside the manifests"
            limits = {k: int(v) for k, v in sr["extensions_needed"].items()}
            if to_end:  # q2 10-15 round 3: the record's runs simulated past their last eligible output (no run index reaches 10**6)
                limits = dict.fromkeys(limits, 10**6)
            conc = min(CONCURRENCY, len(limits))  # only these runs make calls; d9's go is per concurrent run
        call = simulate_live.spawn(inputs, label, limits, conc)
        rec = base | {"action": "live", "label": label, "app_id": app.app_id, "call_id": call.object_id,
                      "spawned_pt": datetime.now(PT).isoformat(timespec="seconds"), "concurrency": conc,
                      "timeout_s": LIVE_TIMEOUT_S, "hard_stop_usd": HARD_STOP_USD, "cap_margin_usd": CAP_MARGIN_USD, "limits": limits,
                      "limits_from": limits_from or None, "limits_sha256": limits_sha256 or None, "to_end": to_end}
        dst = HERE / "results" / f"live_launch_{stamp}.json"
        with open(dst, "x") as fh:
            fh.write(json.dumps(rec, indent=1))
        print(json.dumps(rec), "->", dst.name)
        return
    else:
        raise SystemExit(f"unknown action {action}")
    dst = HERE / "results" / f"{action}_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    with open(dst, "x") as fh:
        fh.write(json.dumps(base | {"result": res}, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("config", "sample_status")}, default=str)[:3000], "->", dst.name)
