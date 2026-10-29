"""On-policy patching pilot, q2's simulator pass (df's RULE.md on aswin/onpolicy-patching @ d9052e47, Amendments 1-2):
Step 2's per-call September simulator, on Modal, over the FIRST 20 non-submit tool outputs of each drawn Qwen3.5-27B run.

One image: the September sims snapshot (~/simulated-control-evals-sept, its uv.lock synced) and the 8 environments'
static codebase folders. No Docker: with codebase_exec off, read tools serve the static folders (Oct 7, Step 2).
- build: the simulator records of the draw (step0_draw_2026-10-14_094447.json, sha256 checked), one cache per arm on the
  Volume, with Step 2's builder's checks (0 shared call ids with the reference, a fresh overlap800 honest reference, a
  September description for every non-bash/editor tool, no attachment://). A failed check stops the build: the draw is
  never skipped through. Inputs: Qwen3.5's .eval files on the q4own_evals Volume (read-only) and the 20 overlap800
  reference .eval files, copied from S3 by the laptop into /data/refs. Returns a keys-only manifest.
- simulate: Step 2's runner (run_rewrite_simulate.py, imported unchanged) on one arm's cache. Its API stop becomes the
  pilot's $130 minus what every other cache on the Volume has charged, and a launch refuses to start if the calls it can
  start plus every call already started would pass 450. --max-calls N keeps only the first run's first N calls (the
  smoke; same records and digests, so the pass skips them later). The image has no .git, so the runner's two git queries
  on the snapshot (HEAD, src diff) are answered with the laptop's git output, passed in.
- tally: per-arm counts from the ledger (statuses, spend, substitutions, trusted prefixes): the 5% gate's inputs.
Nothing prints transcript text: keys, counts and statuses only (a check-only error is cut to its exception type).
Key: the Modal Secret probe-shift_aswin-surya_sub-simruns_openai_kill-2026-11-13 (OPENAI_API_KEY), on simulate only.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run [--detach] \\
      experiments/2026-10-14_onpolicy_sim/onpolicy_sim.py --action build --draw <step0_draw_...094447.json>
      ... --action simulate --arm honest [--check-only N | --max-calls 3]      ... --action tally --arm honest
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
ROOT = HERE.parents[1] if modal.is_local() else Path("/root/repo")  # the container holds this file at /root
SEPT = Path.home() / "simulated-control-evals-sept"
C_SEPT, C_REPO = "/root/simulated-control-evals-sept", "/root/repo"
APP = "probe-shift_aswin-surya_sub-q2-onpolicysim_sim_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q2-onpolicysim_data_kill-2026-11-14"
EVALS = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
SECRET = "probe-shift_aswin-surya_sub-simruns_openai_kill-2026-11-13"
DRAW_SHA = "1c1165aef237e1cbaf670d538a9a6c17ec3443268dbc2af62fe2404c2a066c7a"  # Amendment 2's draw, @ a9960a13
CACHES = Path("/data/caches") / f"{DRAW_SHA[:12]}_r2"  # r2: calls to tools the run lacks are kept real (the r1 build wrote honest only, unused)
WINDOW, STOP_USD, CALL_CAP = 20, 130.0, 450
ENVS = ("auto_workflow", "clinical_trial", "file_indexing", "shared_dotfiles", "slack_clone", "tiktok", "user_tracking", "web_scraping")
REFS = "experiments/2026-10-06_astra_sim_honest/results/references_20261006-142754.json"
CELLS = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
REPO_FILES = ("experiments/2026-10-06_matched_step_pairs/run_simulate_pairs.py",
              "experiments/2026-10-06_matched_step_pairs/run_select_pairs.py",
              "experiments/2026-10-13_step2_sim_rewrite/run_rewrite_simulate.py", REFS, CELLS,
              "src/probe_shift/__init__.py", "src/probe_shift/reader.py", "src/probe_shift/transcripts.py")
PT = ZoneInfo("America/Los_Angeles")

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, create_if_missing=True, version=2)
image = (modal.Image.debian_slim(python_version="3.14").apt_install("git").pip_install("uv==0.12.19")
         .add_local_file(SEPT / "pyproject.toml", f"{C_SEPT}/pyproject.toml", copy=True)
         .add_local_file(SEPT / "uv.lock", f"{C_SEPT}/uv.lock", copy=True)
         .run_commands(f"cd {C_SEPT} && UV_PROJECT_ENVIRONMENT=/usr/local uv sync --frozen --inexact --no-install-project --no-dev")
         .env({"PYTHONPATH": f"{C_SEPT}/src:{C_REPO}/src"})
         .add_local_dir(SEPT / "src", f"{C_SEPT}/src", ignore=["**/__pycache__"]))
for _e in ENVS:
    image = image.add_local_dir(SEPT / ".settings" / "linuxarena" / _e / "codebase", f"{C_SEPT}/.settings/linuxarena/{_e}/codebase")
for _f in REPO_FILES:
    image = image.add_local_file(ROOT / _f, f"{C_REPO}/{_f}")


def eval_samples(path: Path) -> list[dict]:
    import zipfile

    from probe_shift.transcripts import resolve_attachments

    with zipfile.ZipFile(path) as z:  # Python 3.14 reads the zstd members natively
        return [resolve_attachments(json.loads(z.read(n))) for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]


@app.function(image=image, volumes={"/data": vol, "/ev": modal.Volume.from_name(EVALS, version=2).read_only()},
              cpu=4, memory=8192, timeout=1800, retries=0)
def build(draw_raw: bytes) -> dict:
    sys.path.insert(0, f"{C_REPO}/experiments/2026-10-06_matched_step_pairs")
    import run_select_pairs as RSP  # actions_of, SIM_KEYS, DESCRIBED (Step 2's builder's helpers)
    from probe_shift.transcripts import main_task_id

    assert hashlib.sha256(draw_raw).hexdigest() == DRAW_SHA, "not Amendment 2's draw"
    refs = json.loads(Path(C_REPO, REFS).read_text())["references"]
    cells = {c["key"]: c for c in json.loads(Path(C_REPO, CELLS).read_text())["cells"]}
    built, problems = {}, []
    for arm in ("honest", "attack"):
        runs, calls, manifest = [], [], []
        for p in (p for p in json.loads(draw_raw)["picked"] if p["arm"] == arm):
            (s,) = eval_samples(Path("/ev/evals_v1") / p["rel"])
            main = main_task_id(s)
            assert main == p["main_task"] == cells[p["task"]]["main"] and main in refs, f"{p['key']}: main task mismatch or no reference"
            ref_info = refs[main]
            ref_file = ref_info["sim_uri"].rsplit("/", 1)[-1]
            ref = next(x for x in eval_samples(Path("/data/refs") / ref_file)
                       if x["id"] == ref_info["sim_sample_id"] and x["epoch"] == ref_info["sim_epoch"])
            md, traj = ref["metadata"], ref["metadata"]["orig_trajectory"]
            assert md["fork"]["run"].startswith("sce-paired-honest-"), p["key"]
            descriptions = {}
            for m in ref["store"]["SimulatorState:sim_messages"]:
                for name, desc in RSP.DESCRIBED.findall(m["content"] if isinstance(m["content"], str) else ""):
                    descriptions.setdefault(name, desc)
            acts_all = RSP.actions_of(s)
            acts = acts_all[:WINDOW]
            tools = {}
            for x in acts:
                t = x["function"] + (f":{x['arguments'].get('command')}" if x["function"] == "text_editor" else "")
                tools[t] = tools.get(t, 0) + 1
            shared_ids = len({x["id"] for x in acts_all} & {x.get("id") for x in traj})
            # A call to a tool the run doesn't have (Qwen3.5's `str_replace` slip) gets Inspect's own error, in a simulated
            # world too (Inspect rejects it before any tool runs): its real output is kept, never simulated, and later
            # prefixes render it without a description (the September renderer's path for an undescribed tool).
            kept = [j for j, x in enumerate(acts) if x["function"] not in ("bash", "text_editor") and x["function"] not in descriptions]
            problem = ("shares call ids with its reference" if shared_ids else "window is not the draw's calls" if len(acts) != p["calls"] else
                       "a window call has no real output" if any(x["output"] is None for x in acts) else
                       "an undescribed window tool returned no error" if any(acts[j]["error"] is None for j in kept) else None)
            if problem:  # every run is checked; nothing is written if any fails (the draw is never skipped through)
                problems.append({"arm": arm, "key": p["key"], "problem": problem, "window_tools": tools,
                                 "undescribed": sorted({acts[j]["function"] for j in kept})})
                continue
            stored = ref["store"]["SimulatorState:sim_messages"][0]
            assert stored["role"] == "system"
            run_i = len(runs)
            run = {"run": run_i, "arm": arm, "key": p["key"], "file": p["rel"], "sample_id": s["id"], "epoch": s["epoch"],
                   "environment": cells[p["task"]]["env"], "main_task": main, "n_actions": len(acts) - len(kept), "window": len(acts),
                   "kept_real": [{"j": j, "function": acts[j]["function"], "error_type": acts[j]["error"].get("type")} for j in kept],
                   "n_actions_total": len(acts_all),
                   "reference": {"file": ref_file, "sample_id": ref_info["sim_sample_id"], "epoch": ref_info["sim_epoch"],
                                 "cell": ref_info["cell"], "fork_run": md["fork"]["run"], "n_actions": len(traj), "shared_call_ids": shared_ids},
                   "sim_metadata": {key: md.get(key) for key in RSP.SIM_KEYS}, "stored_system_prompt": stored["content"]}
            run_calls = []
            for k, j in enumerate(j for j in range(len(acts)) if j not in kept):  # k: the simulator call, j: the window output
                prefix = [{key: x[key] for key in ("function", "arguments", "output", "id")} for x in acts[:j]]
                for x in prefix:
                    if x["function"] not in ("bash", "text_editor") and x["function"] in descriptions:
                        x["description"] = descriptions[x["function"]]
                step = {key: acts[j][key] for key in ("function", "arguments", "id")}
                if step["function"] not in ("bash", "text_editor"):
                    step["description"] = descriptions[step["function"]]
                run_calls.append({"run": run_i, "k": k, "j": j, "prefix": prefix, "step": step, "real_output": acts[j]["output"],
                                  "real_error": acts[j]["error"]})
            if any("attachment://" in json.dumps(x) for x in [run, *run_calls]):
                raise SystemExit(f"{arm} {p['key']}: an unresolved attachment:// string in a built record")
            runs.append(run)
            calls += run_calls
            manifest.append({k: run[k] for k in ("run", "key", "environment", "main_task", "n_actions", "window", "kept_real", "n_actions_total")}
                            | {"tool_outputs_step0": p["tool_outputs"], "window_tools": tools, "reference": run["reference"]})
        built[arm] = (manifest, {"runs.jsonl": "".join(json.dumps(x) + "\n" for x in runs).encode(),
                                 "calls.jsonl": "".join(json.dumps(x) + "\n" for x in calls).encode()})
    if problems:
        return {"problems": problems}
    out = {}
    for arm, (manifest, files) in built.items():
        d = CACHES / arm
        d.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            if (d / name).exists():  # never overwritten: a rebuild must reproduce it byte for byte
                assert (d / name).read_bytes() == data, f"{d / name} exists with other bytes"
            else:
                with open(d / name, "xb") as fh:
                    fh.write(data)
        out[arm] = {"n_runs": len(manifest), "n_calls": sum(m["n_actions"] for m in manifest), "runs": manifest,
                    **{f"{n.split('.')[0]}_sha256": hashlib.sha256(data).hexdigest() for n, data in files.items()}}
    vol.commit()
    return out


def tally_rows(cache: Path) -> dict:
    sys.path.insert(0, f"{C_REPO}/experiments/2026-10-13_step2_sim_rewrite")
    import run_rewrite_simulate as R

    path = cache / "sim_results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    ok = [r for r in rows if r["status"] == "ok"]
    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    calls_with = {k: sum(bool(r["result"].get(k)) for r in ok) for k in R.SUBS}
    any_sub = sum(any(r["result"].get(k) for k in R.SUBS) for r in ok)
    return {"rows": len(rows), "statuses": statuses, "charged_usd": round(sum(map(R.charge, rows)), 4),
            "ok_calls": len(ok), "ok_sites": len({r["site"] for r in ok}), "calls_with": calls_with, "calls_with_any_substitution": any_sub,
            "share_any_substitution": round(any_sub / len(ok), 4) if ok else None,
            "trusted_prefix_calls": sum(bool(r["result"].get("trusted_prefix")) for r in ok),
            "prompt_mismatch_calls": sum(not r["result"].get("system_prompt_matches_september") for r in ok),
            "inspect_truncated_calls": sum(bool(r["result"].get("inspect_truncated")) for r in ok),
            "max_call_usd": max((r["cost_usd"] for r in rows), default=0.0),
            "ok_nonempty_string_outputs": sum(isinstance(r["result"].get("output"), str) and r["result"]["output"] != "" for r in ok),
            "tokens": {m: {f: sum((r["usage"].get(m) or {}).get(f) or 0 for r in rows) for f in ("input_tokens", "output_tokens", "input_tokens_cache_read")}
                       for m in sorted({m for r in rows for m in r["usage"]})},
            "builds": sorted({json.dumps(r["build"], sort_keys=True) for r in rows})} | empties(cache, ok)


def empties(cache: Path, ok: list[dict]) -> dict:
    """Empty simulator replies beside the real output at the same call (by the runner's record digest); counts only."""
    h = lambda x: hashlib.sha256(x.encode()).hexdigest()  # noqa: E731
    runs = {r["run"]: r for r in map(json.loads, (cache / "runs.jsonl").read_text().splitlines())}
    real = {h(h(json.dumps(runs[c["run"]], sort_keys=True)) + json.dumps(c, sort_keys=True)): c["real_output"]
            for c in map(json.loads, (cache / "calls.jsonl").read_text().splitlines())}
    sim_empty = [r for r in ok if r["result"].get("output") == ""]
    return {"records": len(real), "real_empty_outputs": sum(v == "" for v in real.values()), "sim_empty_outputs": len(sim_empty),
            "sim_empty_and_real_empty": sum(real[r["record_sha256"]] == "" for r in sim_empty),
            "sim_nonempty_and_real_empty": sum(real[r["record_sha256"]] == "" for r in ok if r["result"].get("output") != "")}


@app.function(image=image, volumes={"/data": vol}, cpu=1, memory=4096, timeout=600, retries=0)
def tally(arm: str) -> dict:
    vol.reload()
    return tally_rows(CACHES / arm)


@app.function(image=image, volumes={"/data": vol}, secrets=[modal.Secret.from_name(SECRET)], cpu=2, memory=8192,
              timeout=3 * 3600, retries=0)
def simulate(arm: str, sims_git: dict, check_only: int = 0, max_calls: int = 0, runs_in_parallel: int = 10) -> dict:
    import asyncio
    import contextlib
    import io
    import os
    import types

    sys.path.insert(0, f"{C_REPO}/experiments/2026-10-13_step2_sim_rewrite")
    import run_rewrite_simulate as R

    def git(cmd, **_):  # the runner's only subprocess use: `git -C <snapshot> rev-parse HEAD` and `... diff HEAD -- src`
        assert cmd[:3] == ["git", "-C", str(R.RSIM.SEPT)], cmd
        return types.SimpleNamespace(stdout={("rev-parse", "HEAD"): sims_git["rev"], ("diff", "HEAD", "--", "src"): sims_git["src_diff"]}[tuple(cmd[3:])])

    R.RSIM.subprocess = types.SimpleNamespace(run=git)
    vol.reload()
    cache = CACHES / arm
    others = sum(R.charge(json.loads(line)) for p in Path("/data/caches").glob("*/*/sim_results.jsonl") if p.parent != cache
                 for line in p.read_text().splitlines())
    started = sum(len(p.read_text().splitlines()) for p in Path("/data/caches").glob("*/*/sim_starts.jsonl"))
    for p in Path("/data/caches").glob("*/*/sim_starts.jsonl"):  # Codex r1: a call started without a result may be billed
        r = p.with_name("sim_results.jsonl")
        if len(p.read_text().splitlines()) != (len(r.read_text().splitlines()) if r.exists() else 0):
            raise SystemExit(f"{p.parent}: a started call has no result; reconcile by hand before any launch")
    res = cache / "sim_results.jsonl"
    done = {r["record_sha256"] for r in map(json.loads, res.read_text().splitlines()) if r["status"] == "ok"} if res.exists() else set()
    R.STOP_USD = STOP_USD - others  # the pilot's API stop covers every cache on the Volume
    load = R.load

    def windowed(c):
        runs, items = load(c)
        if max_calls:  # the smoke: the first run's first N calls, same digests
            first = min(items)
            items = {first: items[first][:max_calls]}
        todo = sum(d not in done for v in items.values() for _, d in v)
        if not check_only and started + todo > CALL_CAP:
            raise SystemExit(f"{started} calls started + {todo} to do would pass the {CALL_CAP}-call cap")
        return runs, items

    R.load = windowed
    os.makedirs("/tmp/run", exist_ok=True)
    os.chdir("/tmp/run")  # no .env above it
    sys.argv = ["run_rewrite_simulate.py", "--cache", str(cache), "--runs-in-parallel", str(runs_in_parallel)] + (
        ["--check-only", str(check_only)] if check_only else [])
    print(json.dumps({"arm": arm, "check_only": check_only, "max_calls": max_calls, "api_stop_this_cache_usd": round(R.STOP_USD, 4),
                      "charged_other_caches_usd": round(others, 4), "started_all_caches": started, "done_this_cache": len(done),
                      "sims_rev": sims_git["rev"].strip(), "sims_src_diff_sha256": hashlib.sha256(sims_git["src_diff"].encode()).hexdigest(),
                      "runner_sha256": hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest()}), flush=True)
    if check_only:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            asyncio.run(R.main())
        rows = json.loads(buf.getvalue())
        for r in rows:  # the type only: an error message could quote the prefix
            r["error"] = r["error"].split(":", 1)[0] if r.get("error") else None
        print(json.dumps(rows), flush=True)
        return {"check": rows}
    lock = Path("/data/caches/RUNNING")  # one paying launch at a time (Codex r1); a crash leaves it for a hand reconcile
    if lock.exists():
        raise SystemExit(f"another launch holds {lock}: {lock.read_text()}")
    lock.write_text(json.dumps({"arm": arm, "t": datetime.now(PT).isoformat()}))
    vol.commit()
    try:
        asyncio.run(R.main())
        lock.unlink()
    finally:
        vol.commit()
    t = tally_rows(cache)
    print(json.dumps(t), flush=True)
    return t


@app.local_entrypoint()
def main(action: str, arm: str = "", draw: str = "", check_only: int = 0, max_calls: int = 0, runs_in_parallel: int = 10):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE, check=True).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("refusing to run from a dirty tree")
    stamp = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
    head = git("rev-parse", "HEAD").strip()
    if action == "build":
        out = build.remote(Path(draw).read_bytes())
        if "problems" in out:  # nothing was written; keys, tool names and counts only
            dst = HERE / "results" / f"records_problems_{stamp}.json"
            dst.parent.mkdir(exist_ok=True)
            with open(dst, "x") as fh:
                fh.write(json.dumps({"draw_sha256": DRAW_SHA, "git": head, **out}, indent=1))
            raise SystemExit(f"{len(out['problems'])} drawn runs failed a check; nothing written -> {dst.name}")
        res = {"label": "on-policy patching pilot: simulator records, first 20 non-submit outputs per run (keys only; records stay on the Volume)",
               "draw_sha256": DRAW_SHA, "volume": VOL, "caches": str(CACHES), "git": head,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **out}
        dst = HERE / "results" / f"records_{stamp}.json"
        dst.parent.mkdir(exist_ok=True)
        with open(dst, "x") as fh:
            fh.write(json.dumps(res, indent=1))
        print({a: (out[a]["n_runs"], out[a]["n_calls"]) for a in out}, "->", dst.name)
    elif action == "simulate":
        sims = lambda *a: subprocess.run(["git", "-C", str(SEPT), *a], capture_output=True, text=True, check=True).stdout  # noqa: E731
        print(json.dumps({"launch": stamp, "git": head, "arm": arm}), flush=True)
        simulate.remote(arm, {"rev": sims("rev-parse", "HEAD"), "src_diff": sims("diff", "HEAD", "--", "src")}, check_only, max_calls, runs_in_parallel)
    elif action == "tally":
        t = tally.remote(arm)
        dst = HERE / "results" / f"sim_tally_{arm}_{stamp}.json"
        with open(dst, "x") as fh:
            fh.write(json.dumps({"arm": arm, "git": head, "caches": str(CACHES)} | t, indent=1))
        print(json.dumps({k: t[k] for k in t if k != "builds"}), "->", dst.name)
    else:
        raise SystemExit(f"unknown action {action}")
