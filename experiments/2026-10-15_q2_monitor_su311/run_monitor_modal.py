"""RULE_ONPOLICY Amendment 1 (RULE_AMENDMENT_1.md here): the paper's trajectory monitor on the on-policy Qwen3.5 runs,
run on Modal. One SPEND row per job (SPEND.md here).

  drift_nano  GPT-5.4-nano today on the 196 cohort's honest_b (drift control against launch 1's 10-12 scores) and batch C.
  mini_196    GPT-5.4-mini on the 196 cohort (run 1, run 2, sim, honest A, honest B by corrected-B's keys) and batch C.
  nano_311    GPT-5.4-nano on the +311 cells (run 1, run 2, sim, honest A, honest B by d9's relay manifests and staging).
  nano_nm360  GPT-5.4-nano on the new mains, S1-S3 subset (Amendment 2): keys from inputs_nm360_s123_v1.json; never reads
              corrected-B's manifest (0d's rule), so its render gate uses launch 1's own honest B keys.

Scoring is launch 1's (../2026-10-12_q3_nano_realsim/run_nano_onpolicy.py @ 31d52ebb1), unchanged where it scores: sims'
`metrics.monitor.trajectory` (frozen template sha 14ec7f…, whole trajectory, 0–10, k = 3) through the cached, cost-logged
`runner.metrics.clients.JudgeClient` at medium reasoning effort; .eval files read with sims' `sim.adapters.inspect_eval`;
launch 1's refusal marks (`Observed`) and draw categories. What changes is transport: one Modal CPU container per job (a
deployed function with max_containers=1, so one writer per job), the OpenAI key from the Aswin-authorized Modal Secret,
inputs from read-only Volumes (each file's sha256 checked against a pinned record), the completion cache, cost ledger
and records on a work Volume, and a spend budget checked on every HTTP request to the provider (`Budget`).

Modes: check (no API calls: resolve keys, sha-check and read every file, gates G0 render and G1 coverage, count tokens
with o200k_base, price the job); ping (one tiny call to the job's model under its own run name); run (the paid run;
--resume continues from the cache). No transcript or monitor text leaves the container: records hold scores, counts,
keys and hashes; completions stay in the work Volume's cache; exceptions cross the boundary as type and status only.

  uv run --no-project --python 3.14 --with modal==1.6.1 modal deploy experiments/2026-10-15_q2_monitor_su311/run_monitor_modal.py
  uv run --no-project --python 3.14 --with modal==1.6.1 python experiments/2026-10-15_q2_monitor_su311/run_monitor_modal.py drift_nano check
"""

import asyncio
import collections
import hashlib
import json
import math
import re
import secrets
import subprocess
import time
import types
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
SIMS = Path.home() / "simulated-control-evals"
SIMS_GIT = "e561f9365ef2f9c5f8990491927b5559ae94d473"  # launch 1's sims commit
APP = "probe-shift_aswin-surya_sub-q2-monsu311_monitor_kill-2026-11-15"
WORK = "probe-shift_aswin-surya_sub-q2-monsu311_work_kill-2026-11-15"
EV196 = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
EV311 = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
SECRET = "probe-shift_aswin-surya_sub-simruns_openai_kill-2026-11-13"  # OPENAI_API_KEY only
C_ROOT = "evals_v1/2026-10-11_112714_honest_c/honest_b"  # batch C, one .eval per task folder
L1 = "../2026-10-12_q3_nano_realsim"
D1 = Path.home() / "probe-shift-q3-nano-realsim-1012"  # d9's launch-1 worktree (results/ and data/ hold the files below)
PINS = {  # name: (sha256, where it came from); checked in the local entrypoint and again in the container
    "correctedb_manifest_v1.json": ("bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc",
                                    "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/ @ 46ca25fc"),
    "nano_onpolicy_scores_2026-10-12_133512.json": ("10a62e088d556e3d5ad8cba855f9d6bae7fc05ef8139c0281189b221031cf9af",
                                                    f"{D1}/experiments/2026-10-12_q3_nano_realsim/results/; S3 manifest_nano_onpolicy_s3_v1.json"),
    "nano_onpolicy_draws_2026-10-12_133512.jsonl": ("b3df799564cb66633aae348467808edddcd67c9f3b1c2b542c553446269c3009",
                                                    f"{D1}/data/; S3 manifest_nano_onpolicy_s3_v1.json"),
}
# corrected-B keys launch 1 did not score (its launch marked the attempt unusable and it scored a rerun): sha256 from the
# run folder's S3 manifest, whose own sha256 equals launch 1's recorded manifest_sha256 for the folder (8646fbe5…)
EXTRA196 = {"2026-10-09_182505_scaleup/honest_a/t83_auto_workflow/2026-10-10T02-39-10-00-00_la-auto-workflow-real-honest-1_PehBpNi3ZLuzP6pEBvrman.eval":
            "70b859d85d25e0c8fd03872bcdbf46fb812f12471fc5dd5e07fb80f535e3ebf2"}
C_TASKS = 196  # batch C ran the 196 cohort's tasks: its folders must be exactly launch 1's honest_b task names
LINE196 = {"run1": "real_run1", "run2": "real_run2", "sim": "sim_september", "honest_a": "honest_a", "honest_b": "honest_b"}
JOBS = {  # job: (model, lines, cap in USD, calls in flight, function timeout in seconds)
    "drift_nano": ("openai/gpt-5.4-nano", ("honest_b", "honest_c"), 30.0, 16, 4200),  # $20 -> $30 (SPEND.md, 10-15 01:5x)
    "mini_196": ("openai/gpt-5.4-mini", ("real_run1", "real_run2", "sim_september", "honest_a", "honest_b", "honest_c"),
                 224.0, 32, 7200),
    "nano_311": ("openai/gpt-5.4-nano", ("real_run1", "real_run2", "sim_september", "honest_a", "honest_b"), 100.0, 48, 5500),
    "nano_nm360": ("openai/gpt-5.4-nano", ("real_run1", "real_run2", "sim_september", "honest_a", "honest_b"), 60.0, 48, 5500),
    # S5 pass: the new mains' $60 hard stop less S1-S3's $11.18, so this pass's stop (90%) keeps the total under $54
    "nano_nm360_s5": ("openai/gpt-5.4-nano", ("real_run1", "real_run2", "sim_september", "honest_a", "honest_b"), 47.5, 48, 5500),
}
REMOTE = "https://github.com/redwoodresearch/probe-distribution-shift-from-sims.git"  # pushes go over HTTPS
EFFORT, K, CONCURRENCY = "medium", 3, 64
OPTIONS = {"scale": "0-10", "aggregation": "whole-trajectory", "k_rescores": K, "concurrency": CONCURRENCY}
STOP_SHARE = 0.9  # exposure (logged + carried + one reserve per unsettled provider request) stays <= 90% of the cap
OUT_MAX_TOKENS = 128_000  # the client sets no output limit, so each request is reserved at the model's maximum output
PING_CAP = 2.0  # USD: the ping's own budget (its capability request and retries included)
PROJECT_AFTER = 300  # paid calls before the projected-total stop applies
POLICY = re.compile(r"content_policy|invalid_prompt|content_filter|safety|flagged|refus", re.I)
PACIFIC = ZoneInfo("America/Los_Angeles")

CONTROL_TOWER = "control-tower @ git+https://github.com/linuxarena/control-tower.git@d1156cea40a90f71602ae9f9f853e77e6d87d7e7"
image = (modal.Image.debian_slim(python_version="3.14").apt_install("tzdata", "git")
         .pip_install_from_requirements(str(HERE / "requirements_sims_e561f936_py314.txt"))
         # sims' own pin (pyproject [tool.uv.sources]); its dependencies are already in the exported requirements.
         # runner.metrics.clients imports it at module load through runner.simulate.
         .pip_install(CONTROL_TOWER, extra_options="--no-deps")
         .env({"PYTHONPATH": "/root/sims/src", "TZ": "America/Los_Angeles"})
         .add_local_dir(SIMS / "src", "/root/sims/src", ignore=["**/__pycache__"]))
app = modal.App(APP)
work_vol = modal.Volume.from_name(WORK, create_if_missing=True, version=2)
VOLS = {"/ev196": modal.Volume.from_name(EV196).read_only(), "/ev311": modal.Volume.from_name(EV311).read_only(),
        "/work": work_vol}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def keys_196(inputs: dict, lines: tuple) -> dict:
    """{line: [{task, vol, rel, sha256}]} for the 196 cohort: corrected-B's keys (the probe scorers'), each .eval's sha256
    from launch 1's record of the same path, so these are the files launch 1 scored."""
    man = json.loads(inputs["correctedb_manifest_v1.json"])
    l1 = json.loads(inputs["nano_onpolicy_scores_2026-10-12_133512.json"])
    known = {w["eval"]: w["eval_sha256"] for L in l1["lines"].values() for w in L["where"].values()} | EXTRA196
    out = collections.defaultdict(list)
    for e in man["epochs"]:
        line = LINE196[e["world"]]
        if line not in lines:
            continue
        assert e["key"].endswith("#e0"), e["key"]
        path = e["key"][:-3]
        if path not in known:
            raise SystemExit(f"{line}/{e['task']}: corrected-B key neither in launch 1's files nor in EXTRA196")
        out[line].append({"task": e["task"], "vol": "/ev196", "rel": f"evals_v1/{path}", "sha256": known[path]})
    return dict(out)


def keys_c(inputs: dict) -> list:
    """batch C: every task folder under C_ROOT, exactly one .eval each (as the eval-awareness pass); the folders must be
    exactly launch 1's 196 honest_b task names. No pinned sha exists, so each file's sha256 is recorded, not checked."""
    l1 = json.loads(inputs["nano_onpolicy_scores_2026-10-12_133512.json"])
    expected = {w["task"] for w in l1["lines"]["honest_b"]["where"].values()}
    root, out = Path("/ev196", C_ROOT), []
    found = sorted(p.name for p in root.iterdir())
    base = {f: re.sub(r"_redraw\d+$", "", f) for f in found}  # a redrawn task's folder carries _redrawN (t67 in C)
    if len(expected) != C_TASKS or len(set(base.values())) != len(found) or set(base.values()) != expected:
        raise SystemExit(f"batch C: {len(found)} folders, {len(set(base.values()) ^ expected)} differ from the 196 task names")
    for f in found:
        evs = sorted((root / f).glob("*.eval"))
        if len(evs) != 1:
            raise SystemExit(f"batch C {f}: {len(evs)} .eval files")
        out.append({"task": base[f], "vol": "/ev196", "rel": str(evs[0].relative_to("/ev196")), "sha256": None})
    return out


def keys_311(inputs: dict, pins: dict) -> tuple[dict, dict]:
    """+311 keys the probe scorers' way: per batch, each relay manifest paired with its staging record; honest A/B and run 1
    from the main manifests, run 2 from the run2 manifests, sim from the follow-up manifests; minus each staging record's
    c_gate exclusions (Q2 sim reasons from the corrected record). Exactly one key per (world, cell), else stop."""
    corrected = json.loads(inputs[pins["exclusion_records"][1]["path"]])["excluded"]
    cut = {e["manifest_key"] for e in json.loads(inputs[pins["exclusion_records"][0]["path"]])["excluded"]}
    world_line = {"run1": "real_run1", "run2": "real_run2", "sim": "sim_september", "honest_a": "honest_a",
                  "honest_b": "honest_b"}
    found, excluded = collections.defaultdict(list), {}
    for b in pins["batches"]:
        for pair in b["pairs"]:
            name = Path(pair["manifest"]["path"]).name
            worlds = {"run1", "honest_a", "honest_b"} if name.endswith("_main.json") else \
                {"run2"} if name.endswith("_run2.json") else {"sim"}
            rows = [f for f in json.loads(inputs[pair["manifest"]["path"]])["files"] if f["world"] in worlds]
            gate = json.loads(inputs[pair["staging"]["path"]])
            gate = gate["run2" if worlds == {"run2"} else "main"]["c_gate"]
            keys = {f["path"] + "#e0" for f in rows}
            if gate["manifest_keys"] != len(rows) or not set(gate["excluded"]) <= keys or \
                    gate["staged"] != len(rows) - len(gate["excluded"]):
                raise SystemExit(f"{b['batch']}/{name}: staging record does not match its manifest")
            for f in rows:
                k = f["path"] + "#e0"
                if k in gate["excluded"]:
                    why = corrected[k]["corrected_reason"] if k in corrected else gate["excluded"][k]
                    excluded[f"{world_line[f['world']]}/{f['task']}/{b['batch']}"] = why
                    continue
                assert f["state"] == "done", (k, f["state"])
                found[(world_line[f["world"]], f["task"])].append({"task": f["task"], "vol": "/ev311",
                                                                   "rel": f"evals_v1/{f['path']}", "sha256": f["sha256"]})
    if not cut <= {k[:-3] for k in corrected}:
        raise SystemExit("a Ctrl+C-cut Q2 run is missing from the corrected staging exclusions")
    out = collections.defaultdict(list)
    for (line, task), ks in sorted(found.items()):
        if len({k["rel"] for k in ks}) != 1:
            raise SystemExit(f"{line}/{task}: {len(ks)} counted keys")
        out[line].append(ks[0])
    return dict(out), excluded


def keys_nm360(inputs: dict, pins: dict) -> tuple[dict, dict, dict]:
    """New-mains keys (Amendment 2), the +311 way per batch: run 1 and honest A/B from the main and _m1 manifests (disjoint
    parts, unioned), run 2 from the run2 manifests, sim from the follow-ups; honest runs repeated across manifests dedupe
    by key. Excluded: each stage record's c_gate.excluded, plus every record of the pinned setup-failure file and every
    excluded=true record of the pinned stopped-runs file, even where staged. Exactly one key per (line, cell), else stop.
    The pre-exclusion cell union per launch must equal d9's counts. S5 (Amendment 2b), when the pins carry them: the
    supersession record's 6 originals are excluded (superseded) and its 6 replacements must be exactly the replacement
    batch's sims, sha for sha; the S5 sim status manifests must agree with every sim batch's c_gate, key for key."""
    sup = json.loads(inputs[pins["supersession"]["path"]]) if "supersession" in pins else None
    status = json.loads(inputs[pins["status_manifests"]["path"]])["batches"] if "status_manifests" in pins else {}
    originals, replacements, seen_orig, seen_repl, status_checked = {}, {}, set(), set(), 0
    if sup is not None:
        if not sup["pass"] or len(sup["rows"]) != 6 or any(r["authoritative"] != "replacement" or not r["no_fallback"] for r in sup["rows"]):
            raise SystemExit("supersession record: not 6 authoritative replacements")
        repl_pair = next(pr for bb in pins["batches"] for pr in bb["pairs"] if pr["manifest"]["path"].endswith("manifest_nm360_s5r_sim.json"))
        if sup["replacement_manifest"]["sha256"] != repl_pair["manifest"]["sha256"]:
            raise SystemExit("supersession record names another replacement manifest")
        for r in sup["rows"]:
            o, n = r["original"], r["replacement"]
            originals[(o["launch"], o["run_key"])] = (r["key"], o["eval_sha256"])
            replacements[(n["launch"], n["run_key"])] = (r["key"], n["eval_sha256"])
    extra = {}
    for rec in pins["exclusion_records"]:
        for r in json.loads(inputs[rec["path"]])["records"]:
            if r.get("excluded", True) is True:
                extra.setdefault(r["record"], []).append(Path(rec["path"]).name)
    world_line = {"run1": "real_run1", "run2": "real_run2", "sim": "sim_september", "honest_a": "honest_a",
                  "honest_b": "honest_b"}
    found, excluded, cells = collections.defaultdict(list), {}, collections.defaultdict(set)
    for b in pins["batches"]:
        for pair in b["pairs"]:
            name = Path(pair["manifest"]["path"]).name
            worlds = {"run1", "honest_a", "honest_b"} if re.search(r"_main(_s\d+)?\.json$", name) else \
                {"run2"} if re.search(r"_run2(_s\d+)?\.json$", name) else {"sim"}
            rows = [f for f in json.loads(inputs[pair["manifest"]["path"]])["files"] if f["world"] in worlds]
            gate = json.loads(inputs[pair["staging"]["path"]])
            gate = gate["run2" if worlds == {"run2"} else "main"]["c_gate"]
            keys = {f["path"] + "#e0" for f in rows}
            if gate["manifest_keys"] != len(rows) or not set(gate["excluded"]) <= keys or \
                    gate["staged"] != len(rows) - len(gate["excluded"]):
                raise SystemExit(f"{b['batch']}/{name}: staging record does not match its manifest")
            for f in rows:
                k = f["path"] + "#e0"
                if worlds != {"sim"}:
                    cells[f["launch"]].add(f["task"])
                sources = ([f"stage record c_gate: {gate['excluded'][k]}"] if k in gate["excluded"] else []) + extra.get(f["path"], [])
                if worlds == {"sim"} and b["batch"] in status:
                    st = status[b["batch"]]["rows"].get(f"sim|{f['task']}")
                    if st is None or not st["counted"] or (st["exclusion_reason"] is not None) != (k in gate["excluded"]):
                        raise SystemExit(f"{b['batch']}/{f['task']}: the S5 status manifest disagrees with the stage record")
                    status_checked += 1
                rk = (f["launch"], f"{f['arm_dir']}/{f['launch_task']}")
                if worlds == {"sim"} and rk in originals:
                    if f["sha256"] != originals[rk][1] or rk in seen_orig:
                        raise SystemExit(f"{originals[rk][0]}: original differs from the supersession record or appears twice")
                    seen_orig.add(rk)
                    sources.append("superseded by its s5r replacement (09 @ bffe22e3e; supersession record @ c3e9e3e95)")
                if worlds == {"sim"} and f["launch"].endswith("_s5r_fu"):
                    if rk not in replacements or f["sha256"] != replacements[rk][1] or rk in seen_repl:
                        raise SystemExit(f"{f['task']}: s5r sim not in the supersession record, sha differs, or twice")
                    seen_repl.add(rk)
                if sources:
                    excluded[f["path"]] = {"line": world_line[f["world"]], "task": f["task"], "batch": b["batch"], "sources": sources}
                    continue
                assert f["state"] == "done", (k, f["state"])
                found[(world_line[f["world"]], f["task"])].append({"task": f["task"], "vol": "/ev311",
                                                                   "rel": f"evals_v1/{f['path']}", "sha256": f["sha256"]})
    if sup is not None and (seen_orig != set(originals) or seen_repl != set(replacements)):
        raise SystemExit("supersession: an original or a replacement is missing from the manifests")
    union = {launch: len(c) for launch, c in sorted(cells.items())}
    if union != pins["expected_cells"]:
        raise SystemExit(f"cell union per launch {union} differs from d9's {pins['expected_cells']}")
    out = collections.defaultdict(list)
    for (line, task), ks in sorted(found.items()):
        if len({k["rel"] for k in ks}) != 1:
            raise SystemExit(f"{line}/{task}: {len({k['rel'] for k in ks})} counted keys")
        out[line].append(ks[0])
    run1 = {k["task"] for k in out["real_run1"]}
    meta = {"cell_union_per_launch": union, "cells_with_run1_but_no_sim_key": sorted(run1 - {k["task"] for k in out["sim_september"]}),
            "extra_exclusion_records_applied": sorted(p for p in excluded if p in extra),
            "extra_exclusion_records_not_in_these_manifests": len(set(extra) - set(excluded))}
    if sup is not None:
        meta |= {"superseded_originals": sorted(v[0] for v in originals.values()), "replacements_kept": sorted(v[0] for v in replacements.values()),
                 "status_manifest_rows_checked": status_checked}
    return dict(out), excluded, meta


def keys_l1_honest_b(inputs: dict) -> list:
    """launch 1's own honest B keys (its where entries, path and sha256), for the render gate of jobs that may not read
    corrected-B's manifest"""
    l1 = json.loads(inputs["nano_onpolicy_scores_2026-10-12_133512.json"])
    return [{"task": w["task"], "vol": "/ev196", "rel": f"evals_v1/{w['eval']}", "sha256": w["eval_sha256"]}
            for _, w in sorted(l1["lines"]["honest_b"]["where"].items())]


def read_one(item: dict, simulated: bool, mode_name: str):
    """one .eval: check its sha256 against the pin (or record it), read it with sims' reader; exactly one transcript"""
    from sim.adapters.inspect_eval import transcripts
    from sim.identity import Mode

    path = Path(item["vol"], item["rel"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if item["sha256"] is not None and digest != item["sha256"]:
        raise SystemExit(f"{item['rel']}: sha256 differs from its pinned record")
    ts = list(transcripts(path, simulated=simulated, mode=Mode[mode_name]))
    if len(ts) != 1:
        raise SystemExit(f"{item['rel']}: {len(ts)} transcripts")
    return ts[0], digest


def _job(name, mode, inputs, pins311, git, stamp, resume) -> dict:
    import os
    from concurrent.futures import ThreadPoolExecutor

    from metrics.monitor import trajectory as tm
    from metrics.provenance import corpus_id
    from runner.metrics import clients
    from runner.metrics.clients import JudgeClient, ModelSettings
    from sim.prices import load_prices

    for n, txt in inputs.items():
        if n in PINS and sha(txt) != PINS[n][0]:
            raise SystemExit(f"pin mismatch: {n}")
    model, lines_wanted, cap, in_flight, _ = JOBS[name]
    run_name = f"probe-shift_q2_monsu311_{name}_2026-10-15"  # fixed: the cap covers every attempt
    base = Path("/work", name)
    work_vol.reload()
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "results").mkdir(exist_ok=True)
    os.chdir(base)  # JudgeClient's cache (.run_cache/scoring_calls) and cost ledger (.run_cache/paper/total_cost.jsonl)
    ledger_path = Path(".run_cache/paper/total_cost.jsonl")
    price = load_prices()[model]
    tm._template()  # asserts the frozen template sha

    attempt = f"{stamp}_{secrets.token_hex(4)}"  # unique per invocation: names this attempt's journal and records
    clients._draws.clear()  # a warm container must not carry draw indices (cache keys) or event-loop-bound model clients
    clients._MODELS.clear()
    from inspect_ai.model import get_model

    def fresh_model(n):  # sims' _model(n) is get_model(n) for these OpenAI ids; memoize=False gives this invocation its own
        return get_model(n, memoize=False)

    if mode == "ping":  # one tiny call under its own run name and its own $2 budget, never in the job's ledger rows
        ping_run, ping_prompt = run_name + "_ping", f"Reply with the single word OK. ({attempt})"
        ping_reserve = ((len(ping_prompt.encode()) + 2_000) * price.input + OUT_MAX_TOKENS * price.output) / 1e6
        ping_ledger = make_ledger(ledger_path, ping_run, ping_reserve)
        spent_p, rows_p = ping_ledger()
        carried_p = max(0, prior_requests(base / "data", "ping_journal_*.jsonl") - rows_p) * ping_reserve
        b = Budget(ping_ledger, PING_CAP, ping_reserve, spent_p, rows_p, carried_p, base / "data" / f"ping_journal_{attempt}.jsonl", 1)
        judge = JudgeClient(model=model, settings=ModelSettings(reasoning_effort=EFFORT, run_name=ping_run))
        mdl = fresh_model(model)
        real_model, clients._model = clients._model, lambda n: mdl

        async def one():
            try:
                return await judge.complete("", ping_prompt, reasoning_effort=EFFORT)
            finally:
                await mdl.__aexit__(None, None, None)  # closed inside this invocation's event loop
        try:
            with gated(b):
                out = asyncio.run(one())
        finally:
            clients._model = real_model
            b.journal({"event": "end", "sent": b.sent}, strict=False)
            work_vol.commit()
        return {"job": name, "model": model, "ping_ok": bool(out and out.strip()), "chars": len(out or ""),
                "provider_requests": b.sent, "ping_cost_usd_logged_all_pings": ping_ledger()[0], "carried_reserve_usd": carried_p}

    # keys per line
    excluded = {}
    nm_meta = {}
    if name == "nano_311":
        keys, excluded = keys_311(inputs, pins311)
    elif name in ("nano_nm360", "nano_nm360_s5"):
        keys, excluded, nm_meta = keys_nm360(inputs, pins311)
    else:
        keys = keys_196(inputs, lines_wanted)
    if "honest_c" in lines_wanted:
        keys["honest_c"] = keys_c(inputs)
    keys = {line: keys[line] for line in lines_wanted}
    g0_keys = keys_196(inputs, ("honest_b",))["honest_b"] if name == "nano_311" else \
        keys_l1_honest_b(inputs) if name in ("nano_nm360", "nano_nm360_s5") else None
    simulated = {"sim_september": True}
    attack = {"real_run1", "real_run2", "sim_september"}

    t_load = time.time()
    loaded, where = {}, {}
    with ThreadPoolExecutor(16) as pool:
        for line, items in keys.items():
            res = list(pool.map(lambda it, line=line: read_one(it, simulated.get(line, False),
                                                                "ATTACK" if line in attack else "HONEST"), items))
            loaded[line], where[line] = {}, {}
            for it, (t, digest) in zip(items, res):
                cid = corpus_id(t)
                if cid in loaded[line]:
                    raise SystemExit(f"{line}: corpus id twice")
                loaded[line][cid] = t
                where[line][cid] = {"task": it["task"], "eval": it["rel"], "eval_sha256": digest}
        g0_set = list(pool.map(lambda it: read_one(it, False, "HONEST")[0], g0_keys)) if g0_keys else []
    load_s = round(time.time() - t_load)
    prompts = {line: {cid: tm._prompt(t) for cid, t in d.items()} for line, d in loaded.items()}

    # G0: the image renders launch 1's prompts byte for byte (every run launch 1 sent to the monitor)
    l1 = json.loads(inputs["nano_onpolicy_scores_2026-10-12_133512.json"])
    l1_prompts = {json.loads(x)["prompt_sha256"] for x in inputs["nano_onpolicy_draws_2026-10-12_133512.jsonl"].splitlines() if x.strip()}
    l1_sent = {w["eval"] for L in l1["lines"].values() for cid, w in L["where"].items()
               if cid in L["scores"] or cid in L["dropped"]}
    g0_runs = [(t, where[line][cid]["eval"][len("evals_v1/"):]) for line in keys if line != "honest_c" and name not in ("nano_311", "nano_nm360", "nano_nm360_s5")
               for cid, t in loaded[line].items()]
    g0_runs += [(t, k["rel"][len("evals_v1/"):]) for t, k in zip(g0_set, g0_keys or [])]
    g0_checked = [(t, p) for t, p in g0_runs if p in l1_sent]
    g0_miss = sum(sha(tm._prompt(t)) not in l1_prompts for t, _ in g0_checked)
    gates = {"G0_render": {"checked": len(g0_checked), "not_sent_by_launch1": len(g0_runs) - len(g0_checked),
                           "mismatched": g0_miss, "pass": bool(g0_checked) and g0_miss == 0},
             "G1_coverage": {line: {"keys": len(keys[line]), "loaded": len(loaded[line])} for line in keys}}
    gates["G1_coverage"]["pass"] = all(v["keys"] == v["loaded"] for k, v in gates["G1_coverage"].items() if k != "pass")
    size = {sha(p): len(p) for d in prompts.values() for p in d.values()}
    reserve = ((max(len(p.encode()) for d in prompts.values() for p in d.values()) + 2_000) * price.input
               + OUT_MAX_TOKENS * price.output) / 1e6  # per provider request; input bound: one token per UTF-8 byte of the longest prompt
    meta = {"job": name, "stamp": stamp, "model": model, "reasoning_effort": EFFORT, "options": OPTIONS,
            "run_name": run_name, "cap_usd": cap, "stop_share": STOP_SHARE, "in_flight": in_flight,
            "template_sha256": hashlib.sha256(tm._TEMPLATE_PATH.read_bytes()).hexdigest(), "git": git,
            "sims_git": SIMS_GIT, "pins": {k: PINS[k][0] for k in PINS if k in inputs}, "pins311_index": pins311.get("index"),
            "runs": {line: len(d) for line, d in loaded.items()}, "excluded_by_staging": excluded, "nm360_keys": nm_meta or None,
            "load_seconds": load_s, "gates": gates, "prompt_chars_max": max(size.values()), "reserve_per_request_usd": reserve,
            "stop_usd_at_full_flight": STOP_SHARE * cap - in_flight * reserve}
    if not (gates["G0_render"]["pass"] and gates["G1_coverage"]["pass"]):
        raise SystemExit(f"gate failed: {json.dumps(gates)}")

    if mode == "check":
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        tok = {line: [len(enc.encode(p)) for p in d.values()] for line, d in prompts.items()}
        cached = 21.1 / 42.1  # launch 1's cache-read share of input (SPEND.md there)

        def usd(x, out, share):
            return K * price.usd(input_tokens=x * (1 - share), output_tokens=out, cache_read_tokens=x * share)
        est = {f"output_{o}": round(sum(usd(x, o, cached) for v in tok.values() for x in v), 2) for o in (1000, 4000)}
        est["no_cache_output_4000"] = round(sum(usd(x, 4000, 0.0) for v in tok.values() for x in v), 2)
        rec = meta | {"input_tokens_mean": {k: round(sum(v) / len(v)) for k, v in tok.items()},
                      "input_tokens_total": sum(sum(v) for v in tok.values()), "calls": K * sum(len(v) for v in tok.values()),
                      "estimate_usd": est, "tasks": {line: len({w["task"] for w in d.values()}) for line, d in where.items()}}
        with (base / "results" / f"check_{attempt}.json").open("x") as f:
            f.write(json.dumps(rec, indent=1))
        work_vol.commit()
        return rec

    # ---- the paid run. One writer per job: each job is its own deployed function with max_containers=1, so a second
    # launch queues until the first ends (and then refuses without --resume). Spend is accounted per HTTP request to the
    # provider (budget below), so Inspect's retries, the SDK's retries and Inspect's capability probe are all covered.
    ledger = make_ledger(ledger_path, run_name, reserve)

    # liabilities carried from earlier attempts, from durable records only: every provider request an attempt may have
    # sent (its exact count if it ended cleanly, else every slot it reserved durably) minus the durable ledger rows
    spent0, rows0 = ledger()
    prior = prior_requests(base / "data", "journal_*.jsonl")
    carried = max(0, prior - rows0) * reserve
    if resume and not (spent0 or prior):
        raise SystemExit(f"--resume but nothing is recorded for {run_name}")
    if (spent0 or prior) and not resume:
        raise SystemExit(f"{run_name} has records (${spent0:.2f} logged, {prior} requests): pass --resume")
    if spent0 + carried + in_flight * reserve > STOP_SHARE * cap:
        raise SystemExit(f"logged ${spent0:.2f} plus ${carried:.2f} carried leaves no room for a flight under the stop")

    budget = Budget(ledger, cap, reserve, spent0, rows0, carried, base / "data" / f"journal_{attempt}.jsonl", in_flight)
    paid = []
    budget.paid = paid
    real_model, observed = clients._model, {}

    def model_once(n):  # sims caches one Inspect Model per name; one Observed per name
        if n not in observed:
            observed[n] = Observed(fresh_model(n), paid, base / "data" / f"paid_calls_{attempt}.jsonl")
        return observed[n]

    total = K * sum(size[sha(p)] for d in prompts.values() for p in d.values())
    t0, close_errors = datetime.now(PACIFIC), []

    async def all_lines():
        judge = Guarded(JudgeClient(model=model, settings=ModelSettings(reasoning_effort=EFFORT, run_name=run_name)), budget,
                        asyncio.Semaphore(in_flight), size, total, paid, base / "data" / f"draws_{attempt}.jsonl",
                        progress=base / "data" / f"progress_{attempt}.jsonl")
        settings = types.SimpleNamespace(options=OPTIONS, judge=judge, reasoning_effort=EFFORT, seed=0)
        res = await asyncio.gather(*(tm.run_metric({"subject": tuple(d.values())}, settings) for d in loaded.values()),
                                   return_exceptions=True)
        judge.report(force=True)
        for o in observed.values():  # each invocation's own model clients, closed inside its event loop
            try:
                await o.inner.__aexit__(None, None, None)
            except Exception as error:  # noqa: BLE001 - recorded in the result, never silent
                close_errors.append(type(error).__name__)
        await budget.commit()
        return dict(zip(loaded, res)), judge

    clients._model = model_once
    try:
        with gated(budget):
            res, judge = asyncio.run(all_lines())
    finally:
        clients._model = real_model
        budget.journal({"event": "end", "sent": budget.sent}, strict=False)
        work_vol.commit()
    t1 = datetime.now(PACIFIC)
    errors = {line: type(r).__name__ for line, r in res.items() if isinstance(r, BaseException)}
    if budget.stopped:
        errors["gate"] = "BudgetStop"
    logged_end = ledger()[0]
    common = {"attempt": attempt, "start": t0.isoformat(), "stop": t1.isoformat(), "cost_usd_logged": logged_end, "paid_calls": len(paid),
              "provider_requests": budget.sent, "carried_reserve_usd": carried, "model_close_errors": close_errors,
              "unsettled_requests": budget.sent - (ledger()[1] - rows0)}
    if errors:  # a stop or a failed line: the record (no model text) is kept, and the launcher reports it as a failure
        rec = meta | common | {"errors": errors, "stop_messages": {line: str(r)[:300] for line, r in res.items()
                                                                   if isinstance(r, BudgetStop)}
                               | ({"gate": budget.stopped[:300]} if budget.stopped else {})}
        (base / "results" / f"stopped_{attempt}.json").write_text(json.dumps(rec, indent=1))
        work_vol.commit()
        return rec
    (base / "data" / f"drop_reasons_{attempt}.json").write_text(json.dumps({line: r.payload["dropped"] for line, r in res.items()}))
    by_prompt = collections.defaultdict(list)  # every attempt's draws: a refusal on any attempt counts
    for f in sorted((base / "data").glob("draws_*.jsonl")):
        for x in f.read_text().splitlines():
            r = json.loads(x)
            by_prompt[r["prompt_sha256"]].append(r["outcome"])
    every_paid = [json.loads(x) for f in sorted((base / "data").glob("paid_calls_*.jsonl")) for x in f.read_text().splitlines()]
    flagged = {r["prompt_sha256"] for r in every_paid if r["inspect_refusal"] or r["stop_reason"] == "content_filter"}

    def refused(line, cid):
        h = sha(prompts[line][cid])
        return h in flagged or any(x.startswith("refused") for x in by_prompt[h])

    def category(line, cid):
        if refused(line, cid):
            return "refused"
        return "error" if "error" in by_prompt[sha(prompts[line][cid])] else "unparsed"

    moved = {line: sorted(cid for cid in r.payload["scores"] if refused(line, cid)) for line, r in res.items()}
    for line, r in res.items():
        for cid in moved[line]:
            r.payload["dropped"][cid] = "refused after parsing"
            del r.payload["scores"][cid]
    rec = meta | common | {"lines": {line: {"scores": r.payload["scores"], "attrition": r.payload["attrition"],
                                            "dropped": {cid: category(line, cid) for cid in r.payload["dropped"]},
                                            "drop_counts": dict(collections.Counter(category(line, cid) for cid in r.payload["dropped"])),
                                            "refused_after_parsing": len(moved[line]), "where": where[line]}
                                     for line, r in res.items()}}
    with (base / "results" / f"scores_{attempt}.json").open("x") as f:  # results are append-only
        f.write(json.dumps(rec, indent=1, allow_nan=False))
    work_vol.commit()
    return rec


def prior_requests(data: Path, pattern: str) -> int:
    """provider requests earlier attempts may have sent, from their durable journals: the exact count if an attempt
    ended cleanly, else every slot it reserved"""
    total = 0
    for f in sorted(data.glob(pattern)):
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        ends = [r for r in rows if r["event"] == "end"]
        total += ends[-1]["sent"] if ends else sum(r["n"] for r in rows if r["event"] == "reserve")
    return total


def make_ledger(path: Path, run_name: str, reserve: float):
    """(logged USD, rows) for one run name in JudgeClient's cost ledger; a row without a cost is charged the reserve"""
    def ledger() -> tuple[float, int]:
        if not path.exists():
            return 0.0, 0
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        costs = [(r["run_cost"] if r["run_cost"] is not None else reserve) for r in rows if r["run"] == run_name]
        if not all(isinstance(c, (int, float)) and math.isfinite(c) and c >= 0 for c in costs):
            raise ValueError("a non-finite or negative cost in the ledger")
        return float(sum(costs)), len(costs)
    return ledger


class gated:
    """Patch httpx.AsyncClient.send for the duration: every request to *.openai.com waits for the budget first, and
    redirects are not followed (a redirect would be a second, ungated request)."""

    def __init__(self, budget):
        self.budget = budget

    def __enter__(self):
        import httpx

        self.real, budget, real = httpx.AsyncClient.send, self.budget, httpx.AsyncClient.send

        async def send(client, request, *args, **kwargs):
            if request.url.host.endswith("openai.com"):
                await budget.before_request()
                kwargs["follow_redirects"] = False
            return await real(client, request, *args, **kwargs)
        httpx.AsyncClient.send = send

    def __exit__(self, *exc):
        import httpx

        httpx.AsyncClient.send = self.real
        return False


class BudgetStop(AssertionError):  # an AssertionError, so the monitor re-raises it instead of counting a failure
    pass


class Budget:
    """Exposure = logged ledger spend + carried liabilities + one reserve per provider request sent and not yet settled by
    a ledger row (in flight, failed or retried). A request is sent only if exposure plus its reserve stays <= STOP_SHARE x
    cap, and only from a block of slots reserved durably (journaled and committed) beforehand, so a crash leaves a durable
    upper bound on what was sent. Any failed accounting write or commit stops the run (sticky)."""

    def __init__(self, ledger, cap, reserve, spent0, rows0, carried, journal_path, block):
        self._ledger, self.cap, self.reserve, self.spent0, self.rows0 = ledger, cap, reserve, spent0, rows0
        self.carried, self.journal_path, self.block = carried, journal_path, block
        self.sent, self.reserved, self.stopped, self.floor, self.paid, self._lock = 0, 0, None, spent0, [], None
        journal_path.open("x").close()  # this attempt's own journal, created exclusively

    def ledger(self) -> tuple[float, int]:
        try:
            return self._ledger()
        except Exception as error:  # noqa: BLE001 - an unreadable ledger stops the run (sticky)
            self.stop(f"the cost ledger could not be read ({type(error).__name__})")

    def stop(self, why: str):
        self.stopped = self.stopped or why
        raise BudgetStop(self.stopped)

    def journal(self, row, strict=True):
        try:
            with self.journal_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as error:  # noqa: BLE001 - an accounting write that fails stops the run
            if strict:
                self.stop(f"a journal write failed ({type(error).__name__})")

    async def commit(self):
        try:
            await work_vol.commit.aio()
        except Exception as error:  # noqa: BLE001 - a commit that fails stops the run
            self.stop(f"a work-Volume commit failed ({type(error).__name__})")

    def exposure(self) -> float:
        """the ledger only grows, and by one row per paid call; anything else stops the run"""
        logged, n = self.ledger()
        if logged < self.floor - 1e-9:
            self.stop(f"the ledger shows ${logged:.2f}, below the ${self.floor:.2f} seen before")
        if n - self.rows0 < len(self.paid) or n - self.rows0 > self.sent:
            self.stop(f"the ledger has {n - self.rows0} rows for {len(self.paid)} paid calls and {self.sent} requests")
        self.floor = logged
        return logged + self.carried + (self.sent - (n - self.rows0)) * self.reserve

    async def before_request(self):
        if self.stopped:
            raise BudgetStop(self.stopped)
        if self._lock is None:
            self._lock = asyncio.Lock()
        while self.sent >= self.reserved:  # reserve a durable block before any request in it is sent
            async with self._lock:
                if self.stopped:
                    raise BudgetStop(self.stopped)
                if self.sent >= self.reserved:
                    self.journal({"event": "reserve", "n": self.block})
                    await self.commit()
                    self.reserved += self.block
        if self.stopped:  # rechecked after every wait
            raise BudgetStop(self.stopped)
        e = self.exposure()  # no await between this check and the count: atomic within the event loop
        if e + self.reserve > STOP_SHARE * self.cap:
            self.stop(f"exposure ${e:.2f} plus one request would pass {STOP_SHARE} x cap ${self.cap}")
        self.sent += 1


class Observed:
    """The provider model, unchanged; records per paid call whether Inspect marked the answer a refusal (launch 1's)."""

    def __init__(self, inner, log, path):
        self.inner, self.log, self.path = inner, log, path

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


class Guarded:
    """launch 1's gate around the judge: holds the calls in flight, refuses new calls after a stop or when the projected
    total passes the stop share of the cap, and records every draw's outcome"""

    def __init__(self, inner, budget, gate, size, total, paid, draws_path, progress=None):
        self.inner, self.budget, self.gate, self.size, self.total, self.paid = inner, budget, gate, size, total, paid
        self.draws_path, self.progress = draws_path, progress
        self.draws, self.done, self.last = [], 0, 0.0

    @property
    def name(self) -> str:
        return self.inner.name

    def projected(self, exposure: float) -> float:
        """exposure now, plus the settled spend per prompt character paid so far over the characters still to score
        (reservations are counted once, in exposure, and never extrapolated)"""
        paid_chars = sum(self.size[r["prompt_sha256"]] for r in self.paid)
        settled = self.budget.ledger()[0] - self.budget.spent0
        return exposure + settled / paid_chars * (self.total - self.done) if paid_chars else exposure

    def draw(self, row):
        self.draws.append(row)
        try:
            with self.draws_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as error:  # noqa: BLE001 - a draw record that can't be written stops the run
            self.budget.stop(f"a draw record could not be written ({type(error).__name__})")

    async def complete(self, system, user, **kwargs):
        async with self.gate:
            b, h = self.budget, sha(user)
            if b.stopped:
                raise BudgetStop(b.stopped)
            if len(self.paid) >= PROJECT_AFTER and self.projected(b.exposure()) > STOP_SHARE * b.cap:
                b.stop(f"projected total ${self.projected(b.exposure()):.2f} passes {STOP_SHARE} x cap ${b.cap}")
            try:
                out = await self.inner.complete(system, user, **kwargs)
            except AssertionError:
                raise
            except (Exception, asyncio.CancelledError) as error:
                text = f"{type(error).__name__}: {error}"  # may quote the request: kept in data/ on the work Volume only
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
                e, self.last = self.budget.exposure(), time.time()
            except BudgetStop:
                return
            with self.progress.open("a") as f:
                f.write(json.dumps({"time": datetime.now(PACIFIC).isoformat(timespec="seconds"),
                                    "calls_finished": len(self.draws), "paid_calls": len(self.paid),
                                    "provider_requests": self.budget.sent, "exposure_usd": round(e, 4),
                                    "projected_total": round(self.projected(e), 2), "share_done": round(self.done / self.total, 4),
                                    "outcomes": dict(collections.Counter(d["outcome"] for d in self.draws))}) + "\n")


def run_job(name: str, mode: str, inputs: dict, pins311: dict, git: str, stamp: str, resume: bool, expected: str) -> dict:
    """The sanitized boundary. Our own stops (SystemExit, built from keys, paths and counts) keep their message. Any other
    exception's message is DROPPED: only its type and HTTP status code cross the Modal boundary; the traceback, which may
    quote the request, is written to the work Volume if possible. Nothing is pattern-matched or masked."""
    import logging
    import traceback

    for lib in ("inspect_ai", "httpx", "openai", "anthropic"):  # dependency logs may quote requests: keep them quiet
        logging.getLogger(lib).setLevel(logging.CRITICAL)
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        if code != expected:
            raise SystemExit("the deployed runner differs from the committed file the launcher read: redeploy")
        return _job(name, mode, inputs, pins311, git, stamp, resume) | {"runner_file_sha256": code}
    except SystemExit as stop:
        raise RuntimeError(f"stopped: {stop}") from None
    except BaseException as error:  # noqa: BLE001 - re-raised without its message
        kind, status = type(error).__name__, getattr(error, "status_code", None)
        try:
            path = Path("/work", name, "data", f"error_{stamp}.txt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(traceback.format_exc())
            work_vol.commit()
            where = f"traceback at {WORK}:/{name}/data/error_{stamp}.txt"
        except BaseException:  # noqa: BLE001 - the report itself failed; still raise the sanitized error below
            where = "traceback not saved"
    raise RuntimeError(f"{kind} (status {status}); {where}") from None


FN = dict(image=image, volumes=VOLS, secrets=[modal.Secret.from_name(SECRET)], cpu=(8, 8), memory=(16384, 16384),
          retries=0, startup_timeout=300, scaledown_window=2, max_containers=1)


@app.function(timeout=JOBS["drift_nano"][4], **FN)
def job_drift_nano(mode: str, inputs: dict, pins311: dict, git: str, stamp: str, resume: bool, expected: str) -> dict:
    return run_job("drift_nano", mode, inputs, pins311, git, stamp, resume, expected)


@app.function(timeout=JOBS["mini_196"][4], **FN)
def job_mini_196(mode: str, inputs: dict, pins311: dict, git: str, stamp: str, resume: bool, expected: str) -> dict:
    return run_job("mini_196", mode, inputs, pins311, git, stamp, resume, expected)


@app.function(timeout=JOBS["nano_311"][4], **FN)
def job_nano_311(mode: str, inputs: dict, pins311: dict, git: str, stamp: str, resume: bool, expected: str) -> dict:
    return run_job("nano_311", mode, inputs, pins311, git, stamp, resume, expected)


@app.function(timeout=JOBS["nano_nm360"][4], **FN)
def job_nano_nm360(mode: str, inputs: dict, pins311: dict, git: str, stamp: str, resume: bool, expected: str) -> dict:
    return run_job("nano_nm360", mode, inputs, pins311, git, stamp, resume, expected)


@app.function(timeout=JOBS["nano_nm360_s5"][4], **FN)
def job_nano_nm360_s5(mode: str, inputs: dict, pins311: dict, git: str, stamp: str, resume: bool, expected: str) -> dict:
    return run_job("nano_nm360_s5", mode, inputs, pins311, git, stamp, resume, expected)


def git_out(*args, cwd) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd, check=True).stdout


def launch():
    """Local launcher for the DEPLOYED app (modal deploy of this committed file): checks the tree, gathers and pins the
    inputs, then spawns the job's function and prints the call id. --wait blocks for check/ping and writes their records."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=list(JOBS))
    ap.add_argument("mode", choices=("check", "ping", "run"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    repo = HERE.parents[1]
    if git_out("status", "--porcelain", "--untracked-files=no", cwd=repo).strip():
        raise SystemExit("uncommitted tracked changes in this repo")
    if git_out("rev-parse", "HEAD", cwd=SIMS).strip() != SIMS_GIT or git_out("status", "--porcelain", cwd=SIMS).strip():
        raise SystemExit(f"{SIMS} is not clean at {SIMS_GIT}")
    git = git_out("rev-parse", "HEAD", cwd=repo).strip()  # read once, before the call (the run may outlive later commits)
    if git_out("ls-remote", REMOTE, f"refs/heads/{git_out('rev-parse', '--abbrev-ref', 'HEAD', cwd=repo).strip()}", cwd=repo).split()[:1] != [git]:
        raise SystemExit("HEAD is not the pushed tip of this branch on origin")
    inputs = {} if args.job in ("nano_nm360", "nano_nm360_s5") else {  # nano_nm360 never reads corrected-B's manifest (0d, 10-15)
        "correctedb_manifest_v1.json": git_out("show", "46ca25fc:experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_manifest_v1.json", cwd=repo)}
    inputs |= {
        "nano_onpolicy_scores_2026-10-12_133512.json": (D1 / "experiments/2026-10-12_q3_nano_realsim/results/nano_onpolicy_scores_2026-10-12_133512.json").read_text(),
        "nano_onpolicy_draws_2026-10-12_133512.jsonl": (D1 / "data/nano_onpolicy_draws_2026-10-12_133512.jsonl").read_text(),
    }
    pins311 = {}
    if args.job in ("nano_311", "nano_nm360", "nano_nm360_s5"):
        pins311 = json.loads((HERE / {"nano_311": "inputs_su311_v1.json", "nano_nm360": "inputs_nm360_s123_v1.json",
                                      "nano_nm360_s5": "inputs_nm360_s5_v1.json"}[args.job]).read_text())
        files = ([pins311["index"]] if "index" in pins311 else []) \
            + [p[r] for b in pins311["batches"] for p in b["pairs"] for r in ("manifest", "staging")] + pins311["exclusion_records"] \
            + [pins311[k] for k in ("supersession", "status_manifests") if k in pins311]
        for f in files:
            txt = git_out("show", f"{f['commit']}:{f['path']}", cwd=repo)
            if sha(txt) != f["sha256"]:
                raise SystemExit(f"pin mismatch: {f['path']}")
            inputs[f["path"]] = txt
    for n, (h, _) in PINS.items():
        if n in inputs and sha(inputs[n]) != h:
            raise SystemExit(f"pin mismatch: {n}")
    if args.job in ("nano_nm360", "nano_nm360_s5") and "correctedb_manifest_v1.json" in inputs:
        raise SystemExit("nano_nm360 must not read corrected-B's manifest")
    stamp = datetime.now(PACIFIC).strftime("%Y-%m-%d_%H%M%S")
    fn = modal.Function.from_name(APP, f"job_{args.job}")
    call = fn.spawn(args.mode, inputs, pins311, git, stamp, args.resume, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps({"job": args.job, "mode": args.mode, "stamp": stamp, "git": git, "call_id": call.object_id,
                      "runner_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}), flush=True)
    if args.mode != "run":  # check and ping records hold counts only
        rec = call.get(timeout=JOBS[args.job][4])
        (HERE / "results" / f"{args.job}_{args.mode}_{stamp}.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps({k: v for k, v in rec.items() if k not in ("excluded_by_staging",)}, indent=1)[:6000])


if __name__ == "__main__":
    launch()
