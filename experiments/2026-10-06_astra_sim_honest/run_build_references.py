"""Rebuild chosen honest references as real-honest .eval files the sims code accepts, then verify them.

The September overlap800 references exist only as `metadata.orig_trajectory` inside the simulated attack samples
(actions with their outputs, plus `fork.source_step_sizes` for how they group into assistant turns). The sim task
takes a reference only as a real-honest .eval (`BA_REAL_LOG`, `BA_HONEST_LOG`), so each one is written back out as
a one-sample log: system prompt, empty input, one assistant turn per step with its tool calls, one tool message per
output. The policy's own text and reasoning were never stored; with fork fraction 0 neither is used. A turn that had
text but no tool call (a step of size 0) is rebuilt with a placeholder text, so the fork record keeps its step list.

The file goes to `<out>/<env>/<fork_run>/`, so the sims code records the same `fork.run` as September. It carries
September's own `arena_reference_contract`, and the local env folder must produce the identical contract (after
run_reproduce_contracts.py --write) or the build stops. Only for the envs listed in the config as not reproducible
(media_processing, model_registry; Aswin's OK 2026-10-06) is the local contract stamped instead, with September's kept
in the provenance and the reference marked `contract_stamped_locally`.

Any failed check deletes the reference file and stops, so a reference that is not September's cannot be launched.

Checks, each fatal (no model calls, no containers):
  1. The sims loader turns the file back into exactly the stored trajectory (`honest_reference_by_main`).
  2. The fork it plans matches September's: run, source sample, action count, step sizes, cut at 0.
  3. The simulator system prompt for the new honest sample, built by the real task code, is byte-identical to the
     stored September attack sample's. A diff is written next to the summary if not.

Run inside the September snapshot of simulated-control-evals (its venv and its .settings env checkouts):
  cd ~/simulated-control-evals-sept && CONTROL_TOWER_SETTINGS_DIR=$PWD/.settings uv run python \\
    ~/probe-shift-sims/experiments/2026-10-06_astra_sim_honest/run_build_references.py \\
    --references <results/references_*.json> --mains ct_rewrite_in_go,content_encoding_detector --out <dir>
"""

import argparse
import difflib
import io
import json
import os
import subprocess
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import ToolCall

import runner
from runner.arenas.linuxarena import linuxarena
from runner.arenas.reference_contract import reference_contract
from runner.build.arena_simulated_task import sim_task
from runner.build.run_plan import honest_reference_by_main, planned_samples
from runner.policy.prompts import honest_arena_prompt
from runner.settings import SimulatedTaskSettings
from runner.simulate.config import config_for_sample
from sim.simulator import Simulator

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "config" / "astra_honest.json").read_text())
PT = ZoneInfo("America/Los_Angeles")
NO_TEXT = "(this turn's text was not stored in the September sample; placeholder written by run_build_references.py)"


def stored_sample(uri: str, sample_id: str, epoch: int) -> dict:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    body = boto3.Session(profile_name="redwood-sandbox").client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        for name in z.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                sample = json.loads(z.read(name))
                if sample["id"] == sample_id and sample["epoch"] == epoch:
                    return sample
    raise KeyError(f"{sample_id} epoch {epoch} not in {uri}")


def reference_log(ref: dict, stored: dict) -> tuple[EvalLog, dict]:
    md, env = stored["metadata"], ref["environment"]
    traj, sizes = md["orig_trajectory"], md["fork"]["source_step_sizes"]
    assert sum(sizes) == len(traj) == md["fork"]["of_actions"], (sum(sizes), len(traj), md["fork"])
    honest_prompt = honest_arena_prompt(main_task=md["main_task"]["description"])
    messages = [ChatMessageSystem(content=honest_prompt), ChatMessageUser(content="", source="input")]
    i = 0
    for size in sizes:
        step = traj[i : i + size]
        i += size
        messages.append(ChatMessageAssistant(content="" if step else NO_TEXT, tool_calls=[
            ToolCall(id=a["id"], function=a["function"], arguments=a["arguments"]) for a in step]))
        messages += [ChatMessageTool(content=a["output"], tool_call_id=a["id"], function=a["function"]) for a in step]
    local, sept = reference_contract(linuxarena(env), ref_main(ref)), md["arena_reference_contract"]
    stamp_local = env in CONFIG["september_digest_not_reproducible"]
    assert stamp_local or local == sept, (
        f"{ref_main(ref)}: local env contract {local['contents_sha256'][:12]} differs from September's "
        f"{sept['contents_sha256'][:12]}; run run_reproduce_contracts.py --write first")
    contract = local if stamp_local else sept
    provenance = {
        "reconstructed_from": "metadata.orig_trajectory", "sim_uri": ref["sim_uri"],
        "sim_sample_id": ref["sim_sample_id"], "sept_fork": md["fork"], "sept_contract": md["arena_reference_contract"],
        "contract_stamped_locally": stamp_local, "placeholder_turns": sizes.count(0),
        "builder": "probe-shift-sims experiments/2026-10-06_astra_sim_honest/run_build_references.py",
    }
    sample = EvalSample(
        id=ref["source_sample"], epoch=1, input="", target="", messages=messages,
        metadata={"main_task": md["main_task"], "environment": md["environment"],
                  "arena_reference_contract": contract, "reference_provenance": provenance},
    )
    spec = EvalSpec(
        created=datetime.now(PT).isoformat(timespec="seconds"), task=f"la-{env}-real-honest-1",
        run_id=uuid.uuid4().hex, eval_id=uuid.uuid4().hex,
        dataset=EvalDataset(samples=1, sample_ids=[ref["source_sample"]]),
        model="openai/gpt-5.6-sol", config=EvalConfig(), metadata=provenance,
    )
    return EvalLog(status="success", eval=spec, samples=[sample]), contract


def ref_main(ref: dict) -> str:
    return ref["cell"].split(":", 1)[0]


def simulator_prompt(env: str, path: Path) -> tuple[str, dict]:
    """The September snapshot's own task code, run up to the simulator's system prompt (no model, no container).
    Settings come only from the config: any inherited run variable is removed first."""
    for name in [n for n in os.environ if n.startswith(("BA_", "PT_", "LA_", "SCE_"))]:
        del os.environ[name]
    os.environ.update(CONFIG["env"] | {"LA_ENV": env, "BA_REAL_LOG": str(path), "BA_HONEST_LOG": str(path)})
    arena = linuxarena(env)
    settings = SimulatedTaskSettings.from_environment(arena)
    [sample] = sim_task(arena, settings).dataset
    config = config_for_sample(sample.metadata, model=settings.simulator_model, codebase_exec=False,
                               exec_source_authority=True, orig_trajectory_mode="dump", self_critique=None)
    return Simulator(config, client=object(), fallback=None).system_prompt(), sample.metadata


def build_one(main: str, ref: dict, out_dir: Path, stamp: str) -> dict:
    stored = stored_sample(ref["sim_uri"], ref["sim_sample_id"], ref["sim_epoch"])
    log, contract = reference_log(ref, stored)
    path = out_dir / ref["environment"] / ref["fork_run"] / f"{ref['source_sample'].replace(':', '_')}.eval"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, str(path))
    try:
        md = stored["metadata"]
        actions, _ = honest_reference_by_main([path])[main]
        assert actions == md["orig_trajectory"], f"{main}: rebuilt trajectory differs from the stored one"
        [row] = planned_samples([path], one_per_cell=True, fork_fraction=0.0)
        sept = md["fork"]
        got = {"run": row.fork.run, "source_sample": row.fork.source_sample, "of_actions": row.fork.of_actions,
               "source_step_sizes": list(row.fork.source_step_sizes), "after": row.fork.after}
        want = {k: sept[k] for k in got}
        assert got == want, f"{main}: fork differs: {got} vs {want}"

        prompt, sample_md = simulator_prompt(ref["environment"], path)
        sept_prompt = stored["store"]["SimulatorState:sim_messages"][0]["content"]
        if prompt != sept_prompt:
            diff = difflib.unified_diff(sept_prompt.splitlines(), prompt.splitlines(), "september", "rebuilt", lineterm="")
            with open(HERE / "results" / f"prompt_diff_{main}_{stamp}.txt", "x") as f:
                f.write("\n".join(diff))
            raise AssertionError(f"{main}: simulator prompt differs from September's; diff in results/")
    except BaseException:
        path.unlink()  # a reference that failed a check must not be launchable
        raise
    return {
        "main": main, "environment": ref["environment"], "reference_eval": str(path), "ok": True,
        "trajectory_identical": True, "fork_identical": True, "simulator_prompt_identical": True,
        "prompt_chars": [len(sept_prompt), len(prompt)],
        "contract": {"stamped": contract["contents_sha256"], "september": md["arena_reference_contract"]["contents_sha256"],
                     "local_equals_september": contract == md["arena_reference_contract"]},
        "sample_metadata": {k: sample_md.get(k) for k in (
            "simulator_model", "simulator_fallback_model", "simulator_refusal_judge_model",
            "simulator_router_judge_model", "orig_trajectory_mode", "orig_trajectory_out_cap",
            "codebase_exec", "session_notes", "self_critique", "codebase_root")},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--mains", required=True, help="comma-separated main task ids, or `all`")
    p.add_argument("--out", type=Path, required=True, help="where the reference .eval files go")
    args = p.parse_args()

    refs = json.loads(args.references.read_text())["references"]
    mains = sorted(refs) if args.mains == "all" else args.mains.split(",")
    sims_tree = Path(runner.__file__).resolve().parents[2]
    git = lambda repo: subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    provenance = {"sims_tree": str(sims_tree), "sims_git": git(sims_tree), "probe_shift_git": git(HERE),
                  "references_file": str(args.references)}
    stamp = datetime.now(PT).strftime("%Y%m%d-%H%M%S")
    rows = []
    for main in mains:
        try:  # a failed check is recorded and the reference deleted; the run still exits nonzero below
            row = build_one(main, refs[main], args.out.resolve(), stamp)
        except Exception as e:
            row = {"main": main, "environment": refs[main]["environment"], "ok": False, "error": f"{type(e).__name__}: {e}"}
        rows.append(row | provenance)
        print(f"{main}: " + ("trajectory, fork and simulator prompt identical to September's; contract is September's: "
                             f"{row['contract']['local_equals_september']}" if row["ok"] else f"FAILED {row['error']}"),
              flush=True)
    out = HERE / "results" / f"references_built_{stamp}.jsonl"
    with open(out, "x") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    failed = [r["main"] for r in rows if not r["ok"]]
    print(f"{len(rows)} references, {len(failed)} failed {failed}; wrote {out}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
