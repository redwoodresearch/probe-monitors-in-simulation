"""On-policy patching pilot (df's RULE.md, Amendments 1-4): assemble and gate the simulator's replies, stage the six versions
of the 20 drawn Qwen3.5-27B runs for Qwen3-32B extraction, and build the reply map. All in one Modal CPU function; only
keys, counts and sha256s leave the Volumes.

Versions (arm `patch-<version>`), all on the window = each run's first 20 non-submit tool outputs; outputs 21+ stay real:
  original; rewritten (every simulated window output swapped); bash; edits (text_editor str_replace / insert); createview
  (text_editor create / view); placeholder (every window output -> Step 1's "[tool output withheld]"). Amendment 4: a call
  to a tool the run lacks keeps its real Inspect error in every version (kept_real). A swap puts in the simulator's reply
  as the reader sees it (`reader_output`, Inspect's 16 KiB truncation, as Step 2).
Reader messages: `transcripts.reader_messages` with the run's own system prompt (honest) or the honest system prompt of the
same task (attack; Qwen3.5's honest_a run of that task, every attempt agreeing), as #36 staged Sol's runs.
Gates (all or nothing; every failure listed):
  1. per arm, each result is for a current record (the runner's digest), exactly one ok reply per simulated call, and the
     share of ok calls with any substitution (fallback, refusal, block, network) is at most 5% (RULE gate 4);
  2. the reader messages' calls are the records' calls: same count, names in order, each window tool message equal to the
     real output (sha256), each simulated call's record real output equal too;
  3. only window tool-message content changes, the message count is unchanged, 0 `attachment://`;
  4. every window output_end and the first 21 call_end positions lie inside the 32,768-token cut (Qwen3-32B tokenizer
     @ the pinned revision, `reader.segment_end_positions`, as the extractor computes them).
Writes <ACTS>/<DST>/inputs/shard_0.jsonl, then stage_record.json last (never overwritten). Locally: stage_<stamp>.json
(with every window output's step, action, tool, kept_real and real/simulated sha256 and length) and reply_map_<stamp>.json
in df's shape (scorer @ 55eeb0bb): {"keys": {key: {"window": [{j (1-20), step, action, tool, editor_command, kept_real}],
"versions": {version: {"changed": [[step, action], ...]}}}}}, changed = the window output's content differs from the
original's in that version (sha256). Keys are the draw's `rel` with "/" -> "__" (`file` keeps rel).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run \\
      experiments/2026-10-14_onpolicy_sim/stage_onpolicy.py --draw <step0_draw_2026-10-14_094447.json>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-onpolicysim_stage_kill-2026-11-14"
DATA = "probe-shift_aswin-surya_sub-q2-onpolicysim_data_kill-2026-11-14"
EVALS = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
DRAW_SHA = "1c1165aef237e1cbaf670d538a9a6c17ec3443268dbc2af62fe2404c2a066c7a"
CACHES = f"/data/caches/{DRAW_SHA[:12]}_r2"
DST = "2026-10-14_q2_onpolicy_patching_inputs_v1"
WINDOW, CALLS_SCORED, MAX_TOKENS, SUB_MAX = 20, 21, 32768, 0.05
PLACEHOLDER = "[tool output withheld]"  # Step 1's (aswin/q2-step1-placeholder, stage_placeholder.py)
SUBS = ("fallback_substitutions", "refusal_substitutions", "block_substitutions", "net_substitutions")
SWAP = {"rewritten": lambda tool, cmd: True, "bash": lambda tool, cmd: tool == "bash",
        "edits": lambda tool, cmd: tool == "text_editor" and cmd in ("str_replace", "insert"),
        "createview": lambda tool, cmd: tool == "text_editor" and cmd in ("create", "view")}
VERSIONS = ("original", *SWAP, "placeholder")
WORLD = {"honest": "honest_a", "attack": "run1"}
PT = ZoneInfo("America/Los_Angeles")

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("transformers==4.56.2", "jinja2", "boto3", "pyyaml", "zstandard==0.23.0")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift")
         .add_local_file(HERE / "evalzip.py", "/root/evalzip.py")  # byte copy of df's (zstd .eval members on Python 3.12)
         .add_local_file(HERE.parent / "2026-10-06_matched_step_pairs" / "run_select_pairs.py",  # at its repo depth: it reads
                         "/root/repo/experiments/2026-10-06_matched_step_pairs/run_select_pairs.py"))  # HERE.parents[1] at import
h = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731


def text(content) -> str:
    return content if isinstance(content, str) else json.dumps(content)


@app.function(image=image, cpu=8, memory=32768, timeout=1800, retries=0,
              volumes={"/acts": acts_vol, "/data": modal.Volume.from_name(DATA, version=2).read_only(),
                       "/ev": modal.Volume.from_name(EVALS, version=2).read_only(), "/hf": modal.Volume.from_name(HF_CACHE).read_only()})
def stage(draw_raw: bytes) -> dict:
    try:
        return _stage(draw_raw)
    except Exception as e:  # nothing from the exception (which may quote transcript text) crosses the boundary (Codex r1)
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"stage failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}") from None


def _stage(draw_raw: bytes) -> dict:
    import sys
    import zipfile

    sys.path[:0] = ["/root", "/root/repo/experiments/2026-10-06_matched_step_pairs"]
    import evalzip  # noqa: F401
    import run_select_pairs as RSP  # actions_of, as the records builder
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B
    from probe_shift.reader import segment_end_positions
    from probe_shift.transcripts import main_task_id, reader_messages, resolve_attachments, submit_calls, system_prompt

    assert hashlib.sha256(draw_raw).hexdigest() == DRAW_SHA
    picked = json.loads(draw_raw)["picked"]
    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])

    def sample(path: Path) -> dict:
        with zipfile.ZipFile(path) as z:
            (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
            return resolve_attachments(json.loads(z.read(n)))

    failures, arm_report, replies, records = [], {}, {}, {}
    for arm in WORLD:  # gate 1
        d = Path(CACHES) / arm
        read = lambda name: [json.loads(line) for line in (d / name).read_text().splitlines()]  # noqa: E731
        runs = {r["run"]: r for r in read("runs.jsonl")}
        run_sha = {i: h(json.dumps(r, sort_keys=True)) for i, r in runs.items()}
        calls = read("calls.jsonl")
        digest = {h(run_sha[c["run"]] + json.dumps(c, sort_keys=True)): c for c in calls}
        rows = read("sim_results.jsonl")
        stale = sum(r["record_sha256"] not in digest for r in rows)
        ok = {}
        for r in rows:
            if r["status"] == "ok" and r["record_sha256"] in digest:
                ok.setdefault(r["record_sha256"], []).append(r)
        missing = sum(len(ok.get(x, [])) != 1 for x in digest)
        n_sub = sum(any(v[0]["result"].get(k) for k in SUBS) for v in ok.values())
        arm_report[arm] = {"runs": len(runs), "calls": len(calls), "result_rows": len(rows), "stale_rows": stale,
                           "calls_without_exactly_one_ok": missing, "ok_calls": len(ok), "calls_with_any_substitution": n_sub,
                           "calls_with": {k: sum(bool(v[0]["result"].get(k)) for v in ok.values()) for k in SUBS},
                           "share_any_substitution": round(n_sub / max(len(ok), 1), 4),
                           "api_usd_all_attempts": round(sum(r["cost_usd"] for r in rows), 4)}
        if stale or missing:
            failures.append({"arm": arm, "gate": "one ok reply per simulated call", "stale": stale, "missing": missing})
        if n_sub / max(len(ok), 1) > SUB_MAX:
            failures.append({"arm": arm, "gate": "substitution share above 5% (RULE gate 4)", "share": arm_report[arm]["share_any_substitution"]})
        for x, c in digest.items():
            if len(ok.get(x, [])) == 1:
                replies[(runs[c["run"]]["key"], c["j"])] = (ok[x][0]["result"]["reader_output"], h(c["real_output"]), c["step"])
        records |= {r["key"]: r for r in runs.values()}
    if failures:
        return {"arms": arm_report, "failures": failures, "staged": None}

    honest_prompt = {}
    for p in picked:  # the honest system prompt of each attack run's task, from every honest_a attempt of that task
        if p["arm"] == "attack":
            prompts = {(main_task_id(s), system_prompt(s)) for s in map(sample, Path("/ev/evals_v1").glob(f"*/honest_a/{p['task']}/*.eval"))}
            if len(prompts) != 1 or next(iter(prompts))[0] != p["main_task"]:
                failures.append({"key": p["key"], "gate": "one honest system prompt for the task", "found": len(prompts)})
                continue
            honest_prompt[p["key"]] = next(iter(prompts))[1]

    staged, run_report, outputs, keys = [], [], {}, {}
    for p in picked:
        rec, s = records[p["key"]], sample(Path("/ev/evals_v1") / p["rel"])
        key = p["rel"].replace("/", "__")
        acts = RSP.actions_of(s)
        kept = {x["j"] for x in rec["kept_real"]}
        msgs = reader_messages(s, system_prompt(s) if p["arm"] == "honest" else honest_prompt.get(p["key"], ""))
        turns = [m for m in msgs if m["role"] == "assistant"]
        calls = [(step, action, c) for step, m in enumerate(turns) for action, c in enumerate(m["tool_calls"])]
        tool_i = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
        g2 = (p["arm"] == "honest" or p["key"] in honest_prompt) and len(calls) == len(tool_i) == len(acts) and all(
            c["function"]["name"] == a["function"] for (_, _, c), a in zip(calls, acts)) and all(
            h(text(msgs[tool_i[j]]["content"])) == h(acts[j]["output"]) for j in range(WINDOW)) and all(
            replies[(p["key"], j)][1] == h(acts[j]["output"]) for j in range(WINDOW) if j not in kept) and all(
            (p["key"], j) not in replies for j in kept) and all(  # the record's call is the sample's: id, name, arguments
            {k: replies[(p["key"], j)][2][k] for k in ("function", "arguments", "id")} == {k: acts[j][k] for k in ("function", "arguments", "id")}
            for j in range(WINDOW) if j not in kept) and all(
            (c["function"].get("arguments") or {}) == (a["arguments"] or {}) for (_, _, c), a in zip(calls, acts)) and all(
            acts[x["j"]]["function"] == x["function"] for x in rec["kept_real"])
        if not g2:
            failures.append({"key": p["key"], "gate": "reader messages match the records (calls, names, real outputs)"})
            continue
        entries = []
        for j in range(WINDOW):
            step, action, c = calls[j]
            tool, cmd = c["function"]["name"], (c["function"].get("arguments") or {}).get("command") if c["function"]["name"] == "text_editor" else None
            real = text(msgs[tool_i[j]]["content"])
            sim = None if j in kept else replies[(p["key"], j)][0]
            entries.append({"j": j, "step": step, "action": action, "tool": tool, "command": cmd, "kept_real": j in kept,
                            "changed": sim is not None and sim != real, "real_sha256": h(real), "sim_sha256": None if sim is None else h(sim),
                            "real_chars": len(real), "sim_chars": None if sim is None else len(sim)})
        outputs[key] = {"arm": p["arm"], "file": p["rel"], "entries": entries}
        st = (s.get("metadata") or {}).get("side_task")
        side = (st.get("id") if isinstance(st, dict) else st) if p["arm"] == "attack" else None
        keys[key] = {"window": [{"j": e["j"] + 1, "step": e["step"], "action": e["action"], "tool": e["tool"],
                                 "editor_command": e["command"], "kept_real": e["kept_real"]} for e in entries], "versions": {}}
        for v in VERSIONS:
            new = [dict(m) for m in msgs]
            for e in entries:
                if e["kept_real"] or v == "original":
                    continue
                if v == "placeholder":
                    new[tool_i[e["j"]]]["content"] = PLACEHOLDER
                elif SWAP[v](e["tool"], e["command"]):
                    new[tool_i[e["j"]]]["content"] = replies[(p["key"], e["j"])][0]
            diff = [i for i, (a, b) in enumerate(zip(msgs, new)) if a != b]
            win = set(tool_i[:WINDOW])
            g3 = len(new) == len(msgs) and all(i in win and set(msgs[i]) == set(new[i]) and all(
                msgs[i][f] == new[i][f] for f in msgs[i] if f != "content") for i in diff)
            seg = segment_end_positions(tok, new, MAX_TOKENS)
            pos = {(x["kind"], x["step"], x["action"]): x for x in seg["positions"]}
            out_kept = sum(bool(pos.get(("output_end", e["step"], e["action"]), {}).get("kept")) for e in entries)
            out_none = sum(pos.get(("output_end", e["step"], e["action"]), {}).get("token", 0) is None for e in entries)
            call_kept = sum(bool(pos.get(("call_end", st, ac), {}).get("kept")) for st, ac, _ in calls[:CALLS_SCORED])
            row = {"key": key, "arm": f"patch-{v}", "cohort": f"q4own-{WORLD[p['arm']]}", "file": p["rel"], "sample_id": s["id"],
                   "epoch": s["epoch"], "environment": rec["environment"], "main_task": p["main_task"], "side_task": side,
                   "cell": f"{p['main_task']}:{side}" if side else p["main_task"], "sim_reference": rec["reference"], "n_submit_calls": submit_calls(s),
                   "tokens": seg["n_tokens"], "truncated": seg["n_tokens"] > MAX_TOKENS, "version": v, "pilot_arm": p["arm"], "messages": new}
            g4 = out_kept + out_none == WINDOW and call_kept == min(CALLS_SCORED, len(calls))
            if not (g3 and g4 and "attachment://" not in json.dumps(row)):
                failures.append({"key": p["key"], "version": v, "gate": "only window tool content" if not g3 else "positions inside the cut" if not g4 else "attachment string",
                                 "output_end_kept": out_kept, "output_end_none": out_none, "call_end_kept": call_kept})
                continue
            if v != "original":
                keys[key]["versions"][v] = {"changed": [[e["step"], e["action"]] for e in entries if tool_i[e["j"]] in diff]}
            staged.append(row)
            run_report.append({"key": key, "arm": p["arm"], "version": v, "changed_messages": len(diff), "tokens": seg["n_tokens"],
                               "output_end_kept": out_kept, "output_end_none": out_none, "call_end_kept": call_kept,
                               "staged_sha256": h(json.dumps(row, sort_keys=True))})
    for arm in WORLD:  # empty simulator replies, beside the real output at the same call (counts only)
        es = [e for o in outputs.values() if o["arm"] == arm for e in o["entries"] if e["sim_chars"] is not None]
        arm_report[arm] |= {"sim_empty_replies": sum(e["sim_chars"] == 0 for e in es),
                            "sim_empty_and_real_empty": sum(e["sim_chars"] == 0 and e["real_chars"] == 0 for e in es)}
    out = {"arms": arm_report, "failures": failures, "rows": run_report}
    if failures or len(staged) != len(picked) * len(VERSIONS):
        return out | {"staged": None}
    dst = Path("/acts") / DST
    dst.mkdir()  # exclusive: never overwritten
    (dst / "inputs").mkdir()
    body = "".join(json.dumps(r) + "\n" for r in staged).encode()
    tmp = dst / "inputs" / "shard_0.jsonl.partial"  # the extractor globs shard_*.jsonl: a partial write is never read
    tmp.write_bytes(body)
    tmp.rename(dst / "inputs" / "shard_0.jsonl")
    out |= {"staged": DST, "shard_0_sha256": hashlib.sha256(body).hexdigest(), "shard_0_rows": len(staged),
            "outputs": outputs, "keys": keys}
    (dst / "stage_record.json").write_text(json.dumps({k: v for k, v in out.items() if k != "keys"}))  # last
    acts_vol.commit()
    return out


@app.local_entrypoint()
def main(draw: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    stamp = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
    out = stage.remote(Path(draw).read_bytes())
    base = {"git": git("rev-parse", "HEAD"), "app": APP, "draw_sha256": DRAW_SHA, "caches": CACHES,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with open(HERE / "results" / f"stage_{stamp}.json", "x") as fh:
        fh.write(json.dumps(base | {k: v for k, v in out.items() if k != "keys"}, indent=1))
    if out.get("staged"):
        with open(HERE / "results" / f"reply_map_{stamp}.json", "x") as fh:
            fh.write(json.dumps({"keys": out["keys"], "provenance": base | {"staged": out["staged"], "window": WINDOW}}, indent=1))
    print(json.dumps({"staged": out.get("staged"), "rows": len(out.get("rows", [])), "failures": out["failures"][:10],
                      "arms": out["arms"]}), "->", f"stage_{stamp}.json")
