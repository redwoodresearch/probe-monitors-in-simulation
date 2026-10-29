"""Step 2 pilot: assemble, check and stage the rewritten runs for Qwen3-32B extraction, all on Modal CPU (no GPU).

The laptop only uploads the simulator's files (runs.jsonl, calls.jsonl, sim_results.jsonl) to a new folder on the Volume
and launches one function. That function does the rest:
  - **Assembly gates, per run** (were assemble_rewritten.py's; Aswin 10-13 17:24: nothing runs locally):
    1. exactly one ok simulated reply per call k = 0..n-1, with no duplicates;
    2. cost per run from every attempt's priced provider usage, split by model, with the cached share of input tokens.
  - **Staging,** from #36's staged real-honest rows (`2026-10-06_la_segment-end_qwen3-32b_v1/inputs/`, the text of the
    originals in `2026-10-06_la_segment-end_qwen3-32b_alllayers_v1`). Each tool message's content becomes the
    simulator's reply for its call (`reader_output`). Every other field and message stays as in the source row. Gates:
    3. the row's tool-message count equals the run's call count;
    4. tool messages map to calls by tool-call id when the rows carry ids, else by position;
    5. either way, each source tool message must equal its call's REAL output (sha256), so the mapping is proven;
    6. only `role: tool` content changes, and the message count is unchanged (as stage_fixfull.py's variant gate);
    7. 0 `attachment://` strings in the staged rows.
All or nothing: if any run fails a gate, nothing is staged; the record lists every failure. Writes
`<DST>/inputs/shard_0.jsonl` (arm `real-honest-simrewrite`) and `<DST>/stage_record.json`. Never overwrites.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run \\
      experiments/2026-10-13_step2_sim_rewrite/stage_simrewrite.py --cache <dir>
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q2-step2_stage_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
BASE = "2026-10-06_la_segment-end_qwen3-32b_v1"
DST = "2026-10-13_q2_step2_simrewrite_inputs_v1"
WORK = "2026-10-13_q2_step2_simrewrite_work"
ARM = "real-honest-simrewrite"
FILES = ("runs.jsonl", "calls.jsonl", "sim_results.jsonl")

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12")
h = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731


def text(content) -> str:
    return content if isinstance(content, str) else json.dumps(content)


@app.function(image=image, volumes={"/acts": acts}, cpu=4, memory=32768, timeout=1800, retries=0)
def stage(upload: str) -> dict:
    up = Path("/acts") / WORK / upload
    read = lambda name: [json.loads(l) for l in (up / name).read_text().splitlines()]  # noqa: E731
    run_list = read("runs.jsonl")
    runs = {r["run"]: r for r in run_list}
    assert len(runs) == len(run_list), "duplicate run ids in runs.jsonl"
    run_sha = {i: h(json.dumps(r, sort_keys=True)) for i, r in runs.items()}
    calls, rows, digest = defaultdict(list), defaultdict(list), {}
    for c in read("calls.jsonl"):
        assert c["run"] in runs, f"a call of unknown run {c['run']}"
        calls[c["run"]].append({"k": c["k"], "id": c["step"]["id"], "real_sha256": h(c["real_output"])})
        digest[(c["run"], c["k"])] = h(run_sha[c["run"]] + json.dumps(c, sort_keys=True))  # the runner's record hash
    for r in read("sim_results.jsonl"):
        assert r["run"] in runs, f"a result of unknown run {r['run']}"
        assert r["record_sha256"] == digest[(r["run"], r["k"])], f"result {r['site']} is for another record (stale or mixed file)"
        rows[r["run"]].append(r)
    report, failures, payload = [], [], {}
    for i, run in sorted(runs.items()):
        rs, cs = rows.get(i, []), calls[i]
        assert [c["k"] for c in cs] == list(range(run["n_actions"])), f"run {i}: calls out of order or missing"
        usd, tok = 0.0, defaultdict(lambda: defaultdict(int))
        for r in rs:
            usd += r["cost_usd"]
            for m, u in r["usage"].items():
                for f in ("input_tokens", "input_tokens_cache_read", "output_tokens"):
                    tok[m][f] += u.get(f, 0) or 0
        ok = defaultdict(list)
        for r in rs:
            if r["status"] == "ok":
                ok[r["k"]].append(r)
        e = {"run": i, "key": run["key"], "environment": run["environment"], "n_actions": run["n_actions"],
             "attempts": len(rs), "failed_attempts": sum(r["status"] == "failed" for r in rs), "usd": round(usd, 4),
             "tokens_by_model": {m: dict(v) for m, v in tok.items()},
             "substitutions": {k: sum(bool(v[0]["result"].get(k)) for v in ok.values())
                               for k in ("fallback_substitutions", "refusal_substitutions", "block_substitutions", "net_substitutions")},
             "trusted_prefix_calls": sum(bool(v[0]["result"].get("trusted_prefix")) for v in ok.values()),
             "gate_one_ok_reply_per_call": sorted(ok) == list(range(run["n_actions"])) and all(len(v) == 1 for v in ok.values())}
        report.append(e)
        if not e["gate_one_ok_reply_per_call"]:
            failures.append({"run": i, "gate": "one ok reply per call", "ok_calls": len(ok), "n_actions": run["n_actions"]})
            continue
        payload[run["key"]] = {"run": i, "replies": [ok[k][0]["result"]["reader_output"] for k in range(run["n_actions"])],
                               "real_sha256": [c["real_sha256"] for c in cs], "call_ids": [c["id"] for c in cs]}

    src = {}
    for p in sorted((Path("/acts") / BASE / "inputs").glob("shard_*.jsonl")):
        for line in p.read_text().splitlines():
            it = json.loads(line)
            if it["arm"] == "real-honest" and it["key"] in payload:
                assert it["key"] not in src, "a run appears twice in the source inputs"
                src[it["key"]] = it
    staged, record = [], []
    for key, x in sorted(payload.items()):
        if key not in src:
            failures.append({"run": x["run"], "gate": "source row missing"})
            continue
        it = src[key]
        msgs = it["messages"]
        tool_i = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
        if len(tool_i) != len(x["replies"]):
            failures.append({"run": x["run"], "gate": "tool-message count", "tool_messages": len(tool_i), "calls": len(x["replies"])})
            continue
        ids = [msgs[i].get("tool_call_id") for i in tool_i]
        if all(ids) and len(set(ids)) == len(ids) and sorted(ids) == sorted(x["call_ids"]):  # by id (Codex), one to one
            ks, mapped_by = [x["call_ids"].index(t) for t in ids], "id"
        elif not any(ids):  # no ids in the reader rows: position, proven by the sha256 gate below
            ks, mapped_by = list(range(len(tool_i))), "order"
        else:
            failures.append({"run": x["run"], "gate": "tool-call ids differ from the records'"})
            continue
        assert sorted(ks) == list(range(len(tool_i))), "the mapping is not a permutation of the calls"
        mism = sum(h(text(msgs[i]["content"])) != x["real_sha256"][k] for i, k in zip(tool_i, ks))
        if mism:
            failures.append({"run": x["run"], "gate": "source reply equals real output", "mismatches": mism})
            continue
        new = [dict(m) for m in msgs]
        for i, k in zip(tool_i, ks):
            new[i]["content"] = x["replies"][k]
        changed = [i for i, (a, b) in enumerate(zip(msgs, new)) if a != b]
        only_tool = len(new) == len(msgs) and all(msgs[i]["role"] == "tool" and set(msgs[i]) == set(new[i]) and
                                                  all(msgs[i][f] == new[i][f] for f in msgs[i] if f != "content") for i in changed)
        row = it | {"arm": ARM, "messages": new}
        no_att = "attachment://" not in json.dumps(row)
        if not (only_tool and no_att):
            failures.append({"run": x["run"], "gate": "only tool content changes" if not only_tool else "attachment string"})
            continue
        staged.append(row)
        record.append({"run": x["run"], "key": key, "tool_messages": len(tool_i), "changed_messages": len(changed),
                       "mapped_by": mapped_by, "source_sha256": h(json.dumps(it, sort_keys=True)),
                       "staged_sha256": h(json.dumps(row, sort_keys=True))})
    upload_sha = {n: hashlib.sha256((up / n).read_bytes()).hexdigest() for n in FILES}
    out = {"upload": upload, "upload_sha256": upload_sha, "runs": len(runs), "passed": len(record), "failures": failures,
           "api_usd_all_attempts": round(sum(e["usd"] for e in report), 2), "per_run": report, "staged_runs": record}
    if failures or len(record) != len(runs):
        out["staged"] = None  # all or nothing
        (up / "assembly_record.json").write_text(json.dumps(out))
        acts.commit()
        return out
    dst = Path("/acts") / DST
    dst.mkdir()  # exclusive: raises if the staged run exists (never overwritten)
    (dst / "inputs").mkdir()
    body = "".join(json.dumps(r) + "\n" for r in staged).encode()
    tmp = dst / "inputs" / "shard_0.jsonl.partial"  # the extractor globs shard_*.jsonl, so a partial write is never read
    tmp.write_bytes(body)
    assert hashlib.sha256(tmp.read_bytes()).hexdigest() == hashlib.sha256(body).hexdigest()
    tmp.rename(dst / "inputs" / "shard_0.jsonl")
    out |= {"staged": DST, "shard_0_sha256": hashlib.sha256(body).hexdigest(), "shard_0_rows": len(staged)}
    (dst / "stage_record.json").write_text(json.dumps(out))  # written last: extraction launches only once it exists
    (up / "assembly_record.json").write_text(json.dumps(out))
    acts.commit()
    return out


@app.local_entrypoint()
def main(cache: str):
    import io
    from datetime import datetime
    from zoneinfo import ZoneInfo

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    upload = f"upload_{stamp}"
    with acts.batch_upload() as batch:  # no force: an existing path fails the upload (never overwritten)
        for name in FILES:
            batch.put_file(io.BytesIO((Path(cache) / name).read_bytes()), f"{WORK}/{upload}/{name}")
    out = stage.remote(upload)
    here = Path(__file__).parent
    with open(here / "results" / f"stage_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out | {"app": APP}, indent=1))
    print({k: out.get(k) for k in ("staged", "runs", "passed", "api_usd_all_attempts", "shard_0_sha256")}, "failures", out["failures"][:5])
