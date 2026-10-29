"""Did Qwen3-32B see its tool outputs when it acted in LinuxArena? (2026-10-12, for James; $0, logs only, honest runs only)

For every honest Qwen3-32B acting run with a request log (the 10-09 Phase 2 arms), each tool output Inspect sent in the
request for step k+1 is looked for, byte for byte, inside a `<tool_response>` span of the prompt tokens vLLM actually
received for that request (the request-log proxy's prompt_token_ids, decoded to bytes with the server's tokenizer
revision). The join is exact: each policy ModelEvent carries its request ID (x-irid; the runs used --log-model-api)
and the request's messages (events_data.calls via call_refs). The sent text is also checked against the .eval's own
tool message (Inspect sends "Error: <message>" for a failed call). Also counted: empty outputs, Inspect's own
truncation notice, outputs never delivered (no next request, or the next request rejected), tool errors, str_replace
calls without new_str (the strict-tools bug), and the "made up files" pattern: text_editor calls on a path that an
earlier output in the run said does not exist.

The 10-06 feasibility pilot predates the proxy: its tool messages are re-rendered through the model's own chat template
and checked the same way. That is weaker evidence (no record of the tokens the server received).

Writes results/tool_outputs_<stamp>.json (counts only) and, outside git, the seeded example's masked excerpt
(letters and digits -> x unless the call was an ls-type listing) to <excerpt dir>/tool_outputs_example_<stamp>.json.

  uv run --no-project --python 3.14 --with tokenizers --with huggingface_hub --with jinja2 python -I \\
      check_tool_outputs.py --data <dir with the 10-09 run folders and .request_logs> --pilot <10-06 phase2_pilot dir> \\
      --excerpt-dir <scratch dir>
"""

import argparse
import glob
import gzip
import hashlib
import json
import random
import re
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HERE = Path(__file__).parent
MODEL, REV = "Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"  # servers_qwen3_32b_native.yaml and the logs' headers
POLICY = "vllm/Qwen/Qwen3-32B"
RUNS_1009 = ("2026-10-09_084745_qwen3_32b_honest", "2026-10-09_121200_qwen3_32b_native", "2026-10-09_123140_qwen3_32b_native_relaunch")
PILOT_1006 = ("2026-10-06_la_real_honest_phase2-qwen3_32b", "2026-10-06_la_real_honest_phase2-qwen3_32b_native")
SEED = 20261012
OPEN, CLOSE, ASSISTANT = b"<tool_response>\n", b"\n</tool_response>", b"<|im_start|>assistant"
TRUNCATED = "was too long to be displayed"  # Inspect's own notice when it cuts a tool output
MISSING = (re.compile(r"The path (\S+) does not exist"), re.compile(r"cannot access '([^']+)': No such file"),
           re.compile(r"(?m)^(?:\S+: )?(?:line \d+: )?(/[^\s:]+): No such file or directory"))
LISTING = re.compile(r"^\s*(ls|find|tree)\b")


def byte_table():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def text_of(content, att: dict | None = None) -> str:
    """A message's text; events_data keeps long strings as attachment://<hash>, resolved from the sample's attachments."""
    def res(x: str) -> str:
        return att[x[len("attachment://"):]] if att is not None and x.startswith("attachment://") else x

    if isinstance(content, str):
        return res(content)
    return "".join(res(p.get("text", "")) for p in content if p.get("type") == "text")


def load_sample(path: Path) -> dict:
    z = zipfile.ZipFile(path)
    names = [n for n in z.namelist() if n.startswith("samples/")]
    assert len(names) == 1, (path, names)
    return json.loads(z.read(names[0]))


def as_sent(m: dict, att: dict | None = None) -> str:
    """What Inspect's OpenAI-compatible provider sends for a tool message (inspect_ai _openai.openai_chat_message)."""
    return f"Error: {text_of(m['error']['message'], att)}" if m.get("error") else text_of(m["content"], att)


def str_replace_from_outputs(events: list[dict], c: Counter):
    """str_replace calls as the model emitted them, from every model call's output (covers runs that saved no messages)."""
    for e in events:
        for ch in (e.get("output") or {}).get("choices") or []:
            for tc in (ch.get("message") or {}).get("tool_calls") or []:
                a = tc.get("arguments") or {}
                if tc.get("function") == "text_editor" and isinstance(a, dict) and a.get("command") == "str_replace":
                    c["model_output_str_replace_calls"] += 1
                    c["model_output_str_replace_with_new_str"] += "new_str" in a


def missing_paths(text: str) -> set[str]:
    return {g for rx in MISSING for g in rx.findall(text)}


def behaviour(msgs: list[dict], c: Counter, att: dict | None = None):
    """Tool errors, str_replace new_str presence and the made-up-files pattern, from the sample's messages."""
    told_missing: set[str] = set()
    last_missing: set[str] = set()  # what the immediately previous tool output said is missing
    results = {m.get("tool_call_id"): m for m in msgs if m["role"] == "tool"}

    def same(mp: str, p: str) -> bool:  # the editor resolves a relative path under the working directory
        return mp == p or (not p.startswith("/") and mp.endswith("/" + p.removeprefix("./")))
    for m in msgs:
        if m["role"] == "tool":
            c["tool_outputs"] += 1
            if m.get("error"):
                c["tool_errors"] += 1
                c[f"tool_errors_type_{m['error'].get('type')}"] += 1
            continue
        if m["role"] != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            a = tc.get("arguments") or {}
            r = results.get(tc.get("id"))
            out = as_sent(r, att) if r else ""
            if tc.get("function") == "text_editor":
                cmd, p = a.get("command"), a.get("path")
                if cmd == "str_replace":
                    c["str_replace_calls"] += 1
                    c["str_replace_with_new_str"] += "new_str" in a
                if cmd in ("view", "str_replace", "insert") and p:
                    c["editor_reads_or_edits"] += 1
                    c["editor_relative_path"] += not p.startswith("/")
                    hit = any(same(mp, p) for mp in missing_paths(out))
                    c["editor_missing_path_error"] += hit
                    c["editor_missing_path_error_relative_path"] += hit and not p.startswith("/")
                    c["editor_on_path_told_missing_by_previous_output"] += any(same(mp, p) for mp in last_missing)
                    if any(same(mp, p) for mp in told_missing):
                        c["editor_on_path_told_missing_before"] += 1
                        c["editor_on_path_told_missing_before_and_still_missing"] += hit
                    elif hit:
                        c["editor_on_missing_path_first_time"] += 1
            told_missing |= missing_paths(out)
            last_missing = missing_paths(out)


def check_1009(data: Path, dec, added, tool_ids) -> tuple[dict, list[dict]]:
    per_run, located = {}, []
    for run in RUNS_1009:
        recs: dict[str, list[dict]] = {}
        for f in sorted(glob.glob(str(data / f"{run}.request_logs" / "requests_*.jsonl.gz"))):
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("kind") != "header":
                        recs.setdefault(r["request_id"], []).append(r)
        for ev_path in sorted(p for p in (data / run).rglob("*.eval") if "references" not in p.parts):
            s = load_sample(ev_path)
            key = f"{run}/{ev_path.relative_to(data / run).parts[0]}/{ev_path.relative_to(data / run).parts[1]}"
            c = Counter()
            msgs, calls = s["messages"], s["events_data"]["calls"]
            by_id = {m.get("tool_call_id"): m for m in msgs if m["role"] == "tool"}
            events = [e for e in s["events"] if e.get("event") == "model" and e.get("model") == POLICY]
            delivered, outcome = set(), {}  # tool_call_id -> best outcome over every request that carried it
            rank = ["no_record", "rejected_cancelled", "rejected_400", "not_found", "found_byte_exact"]
            att = s.get("attachments") or {}
            for step, e in enumerate(events):
                if not (e.get("call") or {}).get("call_refs"):
                    c["model_events_without_call_record"] += 1
                    continue
                req = [m for a, b in e["call"]["call_refs"] for m in calls[a:b]]
                irid = e["call"]["request"]["extra_headers"]["x-irid"]
                c["requests"] += 1
                last_asst = max((i for i, m in enumerate(req) if m["role"] == "assistant"), default=-1)
                new = [m for m in req[last_asst + 1:] if m["role"] == "tool"]
                ok = [r for r in recs.get(irid, []) if r.get("status") == 200 and r.get("prompt_token_ids")]
                statuses = sorted({str(r.get("status")) for r in recs.get(irid, [])})
                if not new:
                    continue
                if not ok:
                    reason = "no_record" if not statuses else "rejected_" + "_".join(statuses)
                    c[f"requests_{reason}"] += 1
                    for m in new:
                        t = m.get("tool_call_id")
                        outcome[t] = max(outcome.get(t, reason), reason, key=rank.index)
                    delivered |= {m.get("tool_call_id") for m in new}
                    continue
                ids = ok[-1]["prompt_token_ids"]
                c["requests_with_more_than_one_ok_record"] += len(ok) > 1
                c["requests_ok_records_prompt_ids_differ"] += len({tuple(r["prompt_token_ids"]) for r in ok}) > 1
                tb = [added[i] if i in added else bytes(dec[ch] for ch in tool_ids(i)) for i in ids]
                prompt = b"".join(tb)
                ends = [m.start() for m in re.finditer(re.escape(ASSISTANT), prompt)]
                lo, hi = (ends[-2] if len(ends) > 1 else 0), ends[-1]
                cum, t = [], 0
                for x in tb:
                    t += len(x)
                    cum.append(t)
                n_open = prompt.count(OPEN, lo, hi)
                c["requests_ok_with_new_tool_messages"] += 1
                c["requests_ok_tool_response_spans_equal_new_tool_messages"] += n_open == len(new)
                pos = lo
                for m in new:
                    sent = text_of(m["content"], att)
                    assert "attachment://" not in sent[:13], "unresolved attachment"
                    ev_msg = by_id.get(m.get("tool_call_id"))
                    c["checks_sent_equals_eval_tool_message"] += ev_msg is not None and as_sent(ev_msg) == sent
                    c["checks_sent_differs_from_eval_tool_message"] += ev_msg is not None and as_sent(ev_msg) != sent
                    c["checks_sent_without_eval_tool_message"] += ev_msg is None
                    needle = OPEN + sent.encode() + CLOSE
                    at = prompt.find(needle, pos, hi)
                    c["negative_control_perturbed_needle_found"] += prompt.find(OPEN + sent.encode() + b"\x00" + CLOSE, lo, hi) >= 0
                    t = m.get("tool_call_id")
                    if at < 0:
                        c["checks_not_found"] += 1
                        c["checks_not_found_but_found_after_strip"] += prompt.find(sent.strip().encode(), lo) >= 0
                        outcome[t] = max(outcome.get(t, "not_found"), "not_found", key=rank.index)
                    else:
                        c["checks_found_byte_exact"] += 1
                        outcome[t] = "found_byte_exact"
                        a0, a1 = at + len(OPEN), at + len(OPEN) + len(sent.encode())
                        first = next(i for i, x in enumerate(cum) if x > a0) if a1 > a0 else None
                        last = next(i for i, x in enumerate(cum) if x >= a1) if a1 > a0 else None
                        located.append({"run": key, "step": step, "tool_call_id": m.get("tool_call_id"), "request_id": irid,
                                        "token_first": first, "token_last": last, "prompt_tokens": len(ids), "bytes": a1 - a0})
                        pos = at + len(needle)
                    delivered.add(m.get("tool_call_id"))
            for m in msgs:
                if m["role"] == "tool" and m.get("tool_call_id") in outcome:
                    o = outcome[m.get("tool_call_id")]
                    c[f"outputs_{o}"] += 1
                    c["outputs_delivered_empty"] += as_sent(m) == ""
                    c["outputs_delivered_with_inspect_truncation_notice"] += TRUNCATED in as_sent(m)
            undelivered = [m for m in msgs if m["role"] == "tool" and m.get("tool_call_id") not in delivered]
            c["outputs_never_in_a_request"] = len(undelivered)
            c["outputs_never_in_a_request_is_last_message"] = sum(m is msgs[-1] for m in undelivered)
            behaviour(msgs, c)
            c["steps"] = sum(m["role"] == "assistant" for m in msgs)
            str_replace_from_outputs(events, c)
            per_run[key] = {"eval": str(ev_path.relative_to(data)), "date": run[:10], "launch": run, "counts": dict(c),
                            "eval_sha256": hashlib.sha256(ev_path.read_bytes()).hexdigest()}
    return per_run, located


def check_1006(pilot: Path, template: str, bos: str) -> dict:
    import jinja2

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False, **k)

    def raise_exception(msg):
        raise ValueError(msg)

    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string(template)
    per_run = {}
    for run in PILOT_1006:
        for ev_path in sorted((pilot / run).rglob("*.eval")):
            s = load_sample(ev_path)
            msgs, c, att = s["messages"], Counter(), s.get("attachments") or {}
            events = [e for e in s["events"] if e.get("event") == "model"]
            c["model_events"] = len(events)
            if not any(m["role"] == "assistant" for m in msgs):  # a time-limited run saves no messages: use the last call's input
                last = max(events, key=lambda e: sum(b - a for a, b in e.get("input_refs") or []))
                msgs = [m for a, b in last["input_refs"] for m in s["events_data"]["messages"][a:b]]
                c["messages_from_last_model_call_input"] = 1
            str_replace_from_outputs(events, c)
            c["policy_models"] = len({e.get("model") for e in events})
            oa = []
            for m in msgs:
                if m["role"] == "tool":
                    oa.append({"role": "tool", "content": as_sent(m, att), "tool_call_id": m.get("tool_call_id")})
                elif m["role"] == "assistant":
                    oa.append({"role": "assistant", "content": text_of(m["content"], att),
                               "tool_calls": [{"type": "function", "function": {"name": t["function"], "arguments": t.get("arguments") or {}}}
                                              for t in m.get("tool_calls") or []]})
                else:
                    oa.append({"role": m["role"], "content": text_of(m["content"], att)})
            prompt = tpl.render(messages=oa, add_generation_prompt=True, bos_token=bos).encode()
            pos = 0
            for m in oa:
                if m["role"] != "tool":
                    continue
                c["tool_outputs_rendered"] += 1
                c["rendered_unresolved_attachment"] += m["content"].startswith("attachment://")
                at = prompt.find(OPEN + m["content"].encode() + CLOSE, pos)
                c["rendered_found_byte_exact"] += at >= 0
                pos = at + 1 if at >= 0 else pos
                c["empty_outputs"] += m["content"] == ""
                c["inspect_truncation_notice"] += TRUNCATED in m["content"]
            behaviour(msgs, c, att)
            c["steps"] = sum(m["role"] == "assistant" for m in msgs)
            key = f"{run}/{ev_path.parent.name}"
            per_run[key] = {"eval": str(ev_path.relative_to(pilot)), "date": "2026-10-06", "launch": run, "counts": dict(c),
                            "models": sorted({e.get("model") for e in events}),
                            "eval_sha256": hashlib.sha256(ev_path.read_bytes()).hexdigest()}
    return per_run


def main(data: str, pilot: str, excerpt_dir: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    tok = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json", revision=REV))
    cfg = json.loads(Path(hf_hub_download(MODEL, "tokenizer_config.json", revision=REV)).read_text())
    added = {i: t.content.encode() for i, t in tok.get_added_tokens_decoder().items()}
    per_run, located = check_1009(Path(data), byte_table(), added, tok.id_to_token)
    pilot_runs = check_1006(Path(pilot), cfg["chat_template"], cfg.get("bos_token") or "")
    total = sum((Counter(r["counts"]) for r in per_run.values()), Counter())
    total_pilot = sum((Counter(r["counts"]) for r in pilot_runs.values()), Counter())
    # the seeded example: one run, then one located tool output in it
    rng = random.Random(SEED)
    run = rng.choice(sorted(per_run))
    pick = rng.choice([x for x in located if x["run"] == run])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"git": git, "model": MODEL, "revision": REV, "seed": SEED, "runs_1009": per_run, "totals_1009": dict(total),
           "runs_1006_rerendered": pilot_runs, "totals_1006_rerendered": dict(total_pilot), "seeded_example": pick}
    # the excerpt (text) goes outside git
    data_p = Path(data)
    s = load_sample(data_p / per_run[run]["eval"])
    tool = next(m for m in s["messages"] if m["role"] == "tool" and m.get("tool_call_id") == pick["tool_call_id"])
    call = next(t for m in s["messages"] if m["role"] == "assistant" for t in m.get("tool_calls") or [] if t.get("id") == pick["tool_call_id"])
    cmd = (call.get("arguments") or {}).get("cmd") or (call.get("arguments") or {}).get("command") or ""
    listing = call.get("function") == "bash" and bool(LISTING.match(str(cmd)))
    sent = as_sent(tool)
    shown = sent if listing else re.sub(r"[A-Za-z0-9]", "x", sent)
    ex = {"run": run, "step": pick["step"], "tool_call_id": pick["tool_call_id"], "function": call.get("function"),
          "listing": listing, "masked": not listing, "excerpt_first_600_chars": shown[:600], "bytes": pick["bytes"],
          "token_first": pick["token_first"], "token_last": pick["token_last"], "prompt_tokens": pick["prompt_tokens"]}
    edst = Path(excerpt_dir) / f"tool_outputs_example_{stamp}.json"
    edst.write_text(json.dumps(ex))
    out["seeded_example"]["excerpt_file"] = edst.name
    out["seeded_example"]["excerpt_sha256"] = hashlib.sha256(edst.read_bytes()).hexdigest()
    dst = HERE / "results" / f"tool_outputs_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    print("10-09 runs:", len(per_run), "totals:", json.dumps(dict(sorted(total.items()))))
    print("10-06 runs (re-rendered):", len(pilot_runs), "totals:", json.dumps(dict(sorted(total_pilot.items()))))
    print("seeded example:", json.dumps(out["seeded_example"]))
    print("->", dst, edst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--excerpt-dir", required=True)
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
