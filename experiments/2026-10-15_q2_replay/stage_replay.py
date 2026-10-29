"""Replay staging, T196 (df's RULE v7 @ 604bc7c60; gate 4 as clarified @ 4d579eb8c): render each replay or placebo run
exactly as corrected B was rendered (run_q4f_stage_fixed.py @ 0e78df207: stage_q4b.py @ ef93eb73 with Amendment 9's
chat-template fix; cut 32,768, scrub level 1, labeler eb014307), with the simulator's outputs in place of the real tool
outputs it rewrote. One Modal CPU function; hashes, counts, statuses only.

How: stage_q4b.process runs unchanged on the run's counted attempt (the q4b upload manifest, sha256-checked on the
Volume); only `transcripts.reader_messages` is wrapped for the call, so the j-th tool message (the j-th eligible output of
the frozen predicate) carries the simulated output, truncated by Inspect's own rule (`truncate_tool_output`, 16 KiB; the
agent's transcript in a live run), when that output's status is simulated_*; kept-real, rejected and unprocessed outputs
stay real. Everything after (scrub, render, positions, rows, ids) is stage_q4b's own code.
Gates (all or nothing; every failure listed):
  1. one terminal status per eligible output from the ledger (chain verified); a run that stopped (rate limit, budget,
     overrun) or was excluded is unfinished and not staged; the completion invariant (RULE: processed until the next
     unprocessed eligible output begins at token >= 32,768 in the replay render) holds, else the run's needed limit is
     listed for an extension and it is not staged; outputs past it become not_needed_after_cut;
  2. only tool-message content changes (sha256 per message), and exactly at the swapped outputs that differ;
  3. 0 `attachment://` strings and 0 "[tool output withheld]" strings in staged rows and in the ledger outputs;
  4. GATE4_DEFINITION below, against the run's corrected-B staged row (the map's run_of: q4b_main_v1 or
     q4f_main_v1_changed); plus the count of runs whose first difference is the merged boundary token after an EMPTY
     source output;
  5. per arm: kept-real fallbacks (after failure + after refusal) at most 5% of processed eligible outputs (RULE); a run
     above 10% fallbacks is excluded and listed. Sol-answered steps are simulator-written, not a gate (RULE @ 9445e96c6):
     their calls are reported per arm and per run (calls_in_substituted_steps, substituted_share), never gated.
Gate 6's control: 10 Phase-1 run1 epochs (order sha256("replay|control|" + task)) staged with no swap, equal to corrected
B's rows, as arm replay_control_run1, so the extraction and df's scorer can reproduce stored per-row scores.
Modes: `check` writes only the record (used on the stub dry-run ledgers: stub outputs are never staged); `stage` writes
<label>/<out>/inputs/replay_attack.jsonl and replay_honest_b.jsonl (arms renamed; keys unchanged) and stage_record.json
last, only if every gate passes. Never overwrites.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-15_q2_replay/stage_replay.py --label dry_main --mode check
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if modal.is_local() else Path("/root/repo")
APP = "probe-shift_aswin-surya_sub-q2-replay_stage_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q2-onpolicysim_data_kill-2026-11-14"
EVALS = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
HF = "probe-shift_aswin-surya_hf-cache"
Q4F = "0e78df207f55dfda58227ec25d51000ba033d814"  # run_q4f_stage_fixed.py's commit (corrected B's staging)
PREDICATE_SHA = "80f85c4dbf616b01645c9e886ceab3361d8257eb5216fbf7359d9870edf6e403"
INPUTS = {"phase1": ("604bc7c60", "experiments/2026-10-15_replay_sim_outputs/phase1_manifest.json",
                     "872ddc7c68bd856f1b21d3effb5f368068d058372cedc9d749532c57bc9e2ce7"),
          "placebo": ("604bc7c60", "experiments/2026-10-15_replay_sim_outputs/placebo_manifest.json",
                      "8eb57dae18ff40fd1c6da7fdc178868d8773415252a57fe4b83510c7ea58cf94")}
STAGED_SHAS = {("q4b_main_v1", "attack"): "0fc04b487e037fb291da037d9507c5de2d395e9e61f0ca5f217d0c22e95beb7e",
               ("q4b_main_v1", "honest_b"): "3fc71b14a2515abee6d1071ac4f04aa4b8f842c647e81de0684d2f1031a188ae",
               ("q4f_main_v1_changed", "attack"): "a140d75ae191c36fff1fec6999731d4a8edde208192dfe8442aea4c3661d1d0b",
               ("q4f_main_v1_changed", "honest_b"): "4ec266de04bb334d25459318924ad11ffe662b32860e8fccbed28e7e5ea36767"}
BOUNDARY = (248066, frozenset({198}), 248067, 271)  # <tool_response>, newline-led openers, </tool_response>, "\n\n" (@ 6614ee1d)
GATE4_DEFINITION = "through_first_changed_marker"
GATE4_DOC = ("each replay or placebo run's staged token ids equal its own unmodified source run's staged token ids for every position "
             "through and including the first changed output's `<tool_response>` marker token")  # df's RULE.md @ 4d579eb8c
FALLBACK_MAX, RUN_FALLBACK_MAX, CUT = 0.05, 0.10, 32768
ARM_OUT = {"attack": "replay_attack", "honest_b": "replay_honest_b"}
# RULE gate 6's 10 seeded original run1 epochs; seed pinned by df in RULE.md @ 3b3b00303 (expected first 10 listed there)
CONTROL_N, CONTROL_SEED, CONTROL_ARM = 10, "replay|control|", "replay_control_run1"
PLACEHOLDER = "[tool output withheld]"
Q4_FILES = {"q4/stage_q4b.py": ("ef93eb730", "experiments/2026-10-13_q4_why_weaker_POSTHOC/stage_q4b.py"),
            "q4/evalzip.py": ("ef93eb730", "experiments/2026-10-13_q4_why_weaker_POSTHOC/evalzip.py"),
            "manifest_q4b_eval_upload_v1.json": (Q4F, "experiments/2026-10-13_q4_why_weaker_POSTHOC/results/manifest_q4b_eval_upload_v1.json")}
for _m in ("experiments/2026-10-09_fixed_sim/fixes.py", "experiments/2026-10-12_plantslip_transfer/plant_both.py",
           "experiments/2026-10-07_q2_rendering_swap/rerender.py", "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py",
           "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"):
    Q4_FILES[_m] = (Q4F, _m)
COND = "{%- if loop.index0 > ns.last_query_index %}"  # run_q4f_stage_fixed.py @ 0e78df207: Amendment 9's template fix
PT = ZoneInfo("America/Los_Angeles")


def git_show(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=ROOT).stdout


app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
image = (modal.Image.debian_slim(python_version="3.13")
         .uv_pip_install("transformers==5.19.0", "numpy", "jinja2", "zstandard", "boto3", "pyyaml")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))
if modal.is_local():
    for _dst, (_c, _p) in Q4_FILES.items():
        _f = Path("/tmp") / f"q2_stage_{_dst.replace('/', '__')}"
        _f.write_bytes(git_show(_c, _p))
        image = image.add_local_file(_f, f"/root/repo/{_dst}")
    image = image.add_local_file(ROOT / "experiments/2026-10-14_onpolicy_sim/replay_predicate.py", "/root/repo/q4/replay_predicate.py")


# Inspect 0.3.251 (the sims lock's) conflicts with transformers 5.19.0 in one environment, so Inspect's truncation runs in
# its own container: the staging function sends (j, tool name, raw output) per simulated output and gets the text back.
trunc_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("inspect_ai==0.3.251")


@app.function(image=trunc_image, cpu=2, memory=8192, timeout=1200, retries=0)
def truncations(payload: dict) -> dict:
    from inspect_ai.model._call_tools import truncate_tool_output

    out = {}
    for rid, items in payload.items():
        out[rid] = {}
        for j, fn, raw in items:
            tr = truncate_tool_output(fn, raw, 16 * 1024)  # Inspect's default, as every arm's transcript
            out[rid][str(j)] = tr.output if tr else raw
    return out


def verified_rows(path: Path) -> list:
    rows, prev = [json.loads(line) for line in path.read_text().splitlines()], ""
    for r in rows:
        d = r.pop("digest")
        assert d == hashlib.sha256((prev + json.dumps(r, sort_keys=True)).encode()).hexdigest(), f"{path.name}: ledger chain broken"
        r["digest"], prev = d, d
    return rows


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0,
              volumes={"/data": vol, "/ev": modal.Volume.from_name(EVALS, version=2).read_only(), "/hf": modal.Volume.from_name(HF).read_only()})
def stage(inputs: dict, label: str, mode: str, out: str) -> dict:
    import sys
    import traceback
    from collections import Counter, defaultdict

    sys.path.insert(0, "/root/repo/q4")
    import replay_predicate as RP
    import stage_q4b as S
    from transformers import AutoTokenizer

    assert mode in ("check", "stage") and GATE4_DEFINITION == "through_first_changed_marker"
    assert hashlib.sha256(Path(RP.__file__).read_bytes()).hexdigest() == PREDICATE_SHA
    for name, (_c, _p, sha) in INPUTS.items():
        assert hashlib.sha256(inputs[name]).hexdigest() == sha, name
    import re

    assert re.fullmatch(r"[a-z0-9_]+", label), "a plain label"
    if mode == "stage":
        assert not label.startswith("dry"), "stub ledgers are never staged"
        assert not (Path("/data/replay") / label / out).exists(), "never overwritten"
    if not S.fingerprint().startswith(S.LABELER):
        raise ValueError("labeler fingerprint differs")
    tok = AutoTokenizer.from_pretrained(S.QWEN35_27B["name"], revision=S.QWEN35_27B["revision"], local_files_only=True)
    assert tok.chat_template.count(COND) == 1
    tok.chat_template = tok.chat_template.replace(COND, "{%- if true %}")
    folds = S.frozen_folds(json.loads(S.T196.read_text()))
    counted = S.counted_attempts("main")
    prompts = defaultdict(dict)  # run_q4f_stage_fixed._setup's donor table
    for (arm_dir, task), c in counted.items():
        if arm_dir in ("honest_a", "honest_b"):
            hs = S.load(c["eval"])
            prompts[task][arm_dir] = (S.transcripts.system_prompt(hs), S.first_user_sha(hs))
    staged_ref = {}
    for (folder, arm), sha in STAGED_SHAS.items():
        f = Path("/ev") / folder / "inputs" / f"{arm}.jsonl"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == sha
        for line in f.open():
            ep = json.loads(line)
            staged_ref[(folder, arm, ep["run"])] = ep["ids"]

    def boundaries(ids):
        mid, nl, close, empty = BOUNDARY
        return [p for p in range(len(ids) - 1) if ids[p] == mid and (ids[p + 1] in nl or (ids[p + 1] == empty and p + 2 < len(ids) and ids[p + 2] == close))]

    original_rm = S.transcripts.reader_messages
    payload = {}  # every simulated output, truncated by Inspect in its own container
    for row in json.loads(inputs["phase1"])["runs"] + json.loads(inputs["placebo"])["runs"]:
        rid = f"{row['world']}__{row['task']}"
        p = Path("/data/replay") / label / "ledger" / f"{rid}.jsonl"
        if not p.exists():
            continue
        outs_fn = RP.eligible_outputs(S.load(counted[(row["arm_of"], row["task"])]["eval"]))[0]
        payload[rid] = [[j, outs_fn[j]["function"], o] for r in verified_rows(p) if r["kind"] == "result" and r["status"] == "ok"
                        for j, o in zip(r["js"], r["outputs"])]
    trunc = truncations.remote(payload)
    out_rows, failures, unfinished, extensions, run_recs = defaultdict(list), [], [], {}, []
    per_arm = defaultdict(Counter)
    for row in json.loads(inputs["phase1"])["runs"] + json.loads(inputs["placebo"])["runs"]:
        rid, arm, task = f"{row['world']}__{row['task']}", row["arm_of"], row["task"]
        rel = row["corrected_map_key"].split("#")[0]
        try:
            c = counted[(arm, task)]
            assert f"{c['launch']}/{c['rel']}" == rel, "the counted attempt is not the manifest's run"
            rows = verified_rows(Path("/data/replay") / label / "ledger" / f"{rid}.jsonl")
            if any(r["kind"] in ("stop", "excluded") for r in rows) or not any(r["kind"] == "done" for r in rows):
                unfinished.append([rid, next((r.get("reason", r["kind"]) for r in rows if r["kind"] in ("stop", "excluded")), "no done row")])
                continue
            s = S.load(c["eval"])
            outs, _ = RP.eligible_outputs(s)
            status = ["unprocessed"] * len(outs)
            raw, swaps = {}, {}
            for o in outs:
                if o["rejected"]:
                    status[o["j"]] = "kept_real_rejected"
            assigned = Counter()
            kept_texts = []
            for r in rows:
                if r["kind"] == "result" and r["status"] == "ok":
                    assert len(r["js"]) == len(r["outputs"]) == len(r["statuses"]), "a result row's lists differ in length"
                    for j, o, st in zip(r["js"], r["outputs"], r["statuses"]):
                        status[j], raw[j] = st, o
                        assigned[j] += 1
                elif r["kind"] == "kept_real":
                    for j, t in zip(r["js"], r["real_outputs"]):
                        status[j] = r["status"]
                        assigned[j] += 1
                        kept_texts.append(t)
            if any(v > 1 for v in assigned.values()) or any(outs[j]["rejected"] for j in assigned):
                raise ValueError("an eligible output has more than one terminal status, or a rejected one was processed")
            if mode == "stage" and (rows[0]["config"].get("label") or any("STUB|" in o for o in raw.values())):
                raise ValueError("a stub ledger: never staged")
            subs_calls = sum(len(r["js"]) for r in rows if r["kind"] == "result" and r["status"] == "ok" and any(r["subs"].values()))
            for j in raw:
                swaps[j] = trunc[rid][str(j)]
            diffs = {}

            def patched(sample, system):
                msgs = original_rm(sample, system)
                tool_i = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
                assert len(tool_i) == len(outs), "reader tool messages != eligible outputs"
                new = [dict(m) for m in msgs]
                for j, t in swaps.items():
                    new[tool_i[j]]["content"] = t
                h = lambda m: hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()  # noqa: E731
                changed = [i for i, (a, b) in enumerate(zip(msgs, new)) if h(a) != h(b)]
                expected = sorted(tool_i[j] for j, t in swaps.items() if t != msgs[tool_i[j]]["content"])
                only_tool = len(new) == len(msgs) and all(msgs[i]["role"] == "tool" and {k: v for k, v in msgs[i].items() if k != "content"}
                                                          == {k: v for k, v in new[i].items() if k != "content"} for i in changed)
                rendered = json.dumps(new)
                diffs.update(attachments=rendered.count("attachment://"), placeholders=rendered.count(PLACEHOLDER))
                diffs.update(changed=changed, expected=expected, only_tool=only_tool, tool_i=tool_i,
                             first=min((j for j, t in swaps.items() if t != msgs[tool_i[j]]["content"]), default=None),
                             first_source_empty=None)
                if diffs["first"] is not None:
                    diffs["first_source_empty"] = msgs[tool_i[diffs["first"]]]["content"] == ""
                return new

            S.transcripts.reader_messages = patched
            try:
                ep, rec = S.process(c, arm, task, "main", prompts, tok, folds, Counter())
            finally:
                S.transcripts.reader_messages = original_rm
            ids, ref = ep["ids"], staged_ref[(row["staging"], arm, rel)]
            bref, brep = boundaries(ref), boundaries(ids)
            first = diffs["first"]
            lim = bref[first] + 1 if first is not None and first < len(bref) else len(ref)
            g4 = ids[:lim] == ref[:lim]
            d0 = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), None)
            merged_empty = bool(first is not None and first < len(bref) and d0 == bref[first] + 1 and diffs["first_source_empty"])
            # completion invariant: the first unprocessed eligible output must begin at or past the cut
            u = next((j for j in range(len(outs)) if status[j] == "unprocessed"), None)
            invariant = u is None or u >= len(brep) or brep[u] + 2 >= CUT
            if not invariant:
                extensions[rid] = sum(p + 2 < CUT for p in brep)
            for j in range(len(outs)):
                if status[j] == "unprocessed":
                    status[j] = "not_needed_after_cut" if invariant else "unprocessed_in_cut"
            processed = [st for st in status if st not in ("not_needed_after_cut", "not_processed_before_stop", "unprocessed_in_cut")]
            fb = sum(st in ("kept_real_after_failure", "kept_real_after_refusal") for st in processed)
            text = json.dumps(ep)
            g = {"only_tool_content": diffs["only_tool"] and diffs["changed"] == diffs["expected"],
                 "attachments": diffs["attachments"] + text.count("attachment://"),
                 "placeholders_staged": diffs["placeholders"] + text.count(PLACEHOLDER),
                 "placeholders_ledger": sum(PLACEHOLDER in o for o in list(raw.values()) + kept_texts),
                 "gate4": g4, "first_diff_is_merged_empty_boundary": merged_empty, "invariant": invariant}
            rr = {"run": rid, "arm": arm, "statuses": dict(Counter(status)), "processed": len(processed), "fallbacks": fb,
                  "simulated_calls": len(raw), "calls_in_substituted_steps": subs_calls, "truncated": sum(swaps[j] != raw[j] for j in raw),
                  "changed_messages": len(diffs["changed"]), "tokens": len(ids), **g}
            run_recs.append(rr)
            bad = [k for k in ("only_tool_content", "gate4") if not g[k]] + [k for k in ("attachments", "placeholders_staged", "placeholders_ledger") if g[k]]
            if bad:
                failures.append([rid, bad])
            for k in ("processed", "fallbacks", "simulated_calls", "calls_in_substituted_steps"):  # arm gates before removing any run
                per_arm[arm][k] += rr[k]
            if not invariant:
                continue
            if processed and fb / len(processed) > RUN_FALLBACK_MAX:
                rr["excluded"] = "fallbacks above 10%"
                continue
            out_rows[ARM_OUT[arm]].append(ep | {"arm": ARM_OUT[arm]})
        except Exception as e:  # noqa: BLE001  (type and location only)
            tb = traceback.extract_tb(e.__traceback__)[-1]
            failures.append([rid, f"{type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}"])
    gate5 = all(v["fallbacks"] <= FALLBACK_MAX * v["processed"] for v in per_arm.values())  # unrounded; fallbacks only (RULE @ 9445e96c6)
    # RULE gate 6's control: 10 Phase-1 run1 epochs in the order sha256("replay|control|" + task), staged through this same
    # pipeline with NO swap; each must equal corrected B's staged row exactly (then df's scorer reproduces stored scores)
    control = []
    p1 = sorted(json.loads(inputs["phase1"])["runs"], key=lambda r: hashlib.sha256((CONTROL_SEED + r["task"]).encode()).hexdigest())[:CONTROL_N]
    for row in p1:
        rel = row["corrected_map_key"].split("#")[0]
        c = counted[(row["arm_of"], row["task"])]
        ep, _ = S.process(c, row["arm_of"], row["task"], "main", prompts, tok, folds, Counter())
        same = ep["ids"] == staged_ref[(row["staging"], row["arm_of"], rel)]
        control.append({"task": row["task"], "ids_equal_corrected_b": same})
        if same:
            out_rows[CONTROL_ARM].append(ep | {"arm": CONTROL_ARM})
        else:
            failures.append([row["task"], "control epoch differs from corrected B"])
    arms = {a: dict(v) | {"fallback_share": round(v["fallbacks"] / max(v["processed"], 1), 6),
                          "substituted_share": round(v["calls_in_substituted_steps"] / max(v["simulated_calls"], 1), 6)} for a, v in per_arm.items()}
    rec = {"label": label, "mode": mode, "gate4_definition": GATE4_DEFINITION, "gate4_doc": GATE4_DOC, "runs": len(run_recs),
           "failures": failures, "unfinished": unfinished, "extensions_needed": extensions, "arms": arms, "gate5": gate5,
           "gate4_pass": sum(r["gate4"] for r in run_recs), "first_diff_merged_empty_boundary": sum(r["first_diff_is_merged_empty_boundary"] for r in run_recs),
           "invariant_pass": sum(r["invariant"] for r in run_recs), "excluded_runs": [r["run"] for r in run_recs if r.get("excluded")],
           "staged_rows": {a: len(v) for a, v in out_rows.items()}, "control": control, "per_run": run_recs}
    rec["all_gates_pass"] = bool(not failures and not unfinished and not extensions and gate5 and run_recs)
    dst = Path("/data/replay") / label / out
    if mode == "stage" and rec["all_gates_pass"]:
        tmp = dst.with_name(dst.name + ".partial")
        (tmp / "inputs").mkdir(parents=True)
        for a, eps in sorted(out_rows.items()):
            body = "".join(json.dumps(e) + "\n" for e in sorted(eps, key=lambda e: e["key"])).encode()
            (tmp / "inputs" / f"{a}.jsonl").write_bytes(body)
            rec.setdefault("inputs_sha256", {})[f"{a}.jsonl"] = hashlib.sha256(body).hexdigest()
        (tmp / "stage_record.json").write_text(json.dumps(rec))
        tmp.rename(dst)
        rec["staged"] = str(dst)
    else:
        rec["staged"] = None
        (Path("/data/replay") / label / f"{out}_record_{datetime.now(PT).strftime('%Y-%m-%d_%H%M%S')}.json").write_text(json.dumps(rec))
    vol.commit()
    return rec


@app.local_entrypoint()
def main(label: str, mode: str = "check", out: str = "staged_v1"):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    inputs = {name: git_show(c, p) for name, (c, p, _s) in INPUTS.items()}
    res = stage.remote(inputs, label, mode, out)
    stamp = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"stage_{label}_{mode}_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps({"git": git("rev-parse", "HEAD"), "app": APP} | res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_run"})[:3000], "->", dst.name)
