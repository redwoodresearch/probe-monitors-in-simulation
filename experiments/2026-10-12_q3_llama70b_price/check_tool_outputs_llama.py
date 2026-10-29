"""Did Llama-3.3-70B see its tool outputs when it acted in LinuxArena? The Llama variant of 2e's check (Option (b), agreed
with 2e on 2026-10-12): experiments/2026-10-12_qwen3_acting_tool_outputs/check_tool_outputs.py, blob 14b4abec, at
05a4bceb (the #988 -> #992 stack). $0, logs only, honest runs only.

Unchanged from 2e's check (2e's list, kept identical):
1. the join: each policy ModelEvent's x-irid (the runs use --log-model-api) to the request-log proxy's 200 record, whose
   prompt_token_ids are decoded to bytes per token (added tokens from their content, byte-level BPE through the byte
   table);
2. attachments: attachment:// in events_data.calls resolved through sample["attachments"];
3. counting per tool_call_id, the best outcome over every request that carried it
   (found > not_found > rejected_400 > cancelled > no_record), with the same outcome keys;
4. a failed call is compared as Inspect sends it, "Error: <message>", and against the .eval's own tool message;
5. the search window: from the start of the previous assistant turn to the generation prompt;
6. the negative control: the same needle with one extra byte must be found 0 times;
7. references/ copies skipped; request tool messages with no .eval tool message (stopped runs) counted apart;
8. str_replace counted from EVERY model call's output, with whether new_str is present.

Changed for Llama (and only this):
- the tokenizer: the pinned snapshot's tokenizer.json (meta-llama/Llama-3.3-70B-Instruct 6f6073b4…), read from a local
  copy (--tokenizer-dir), since the repo is gated on the Hub, checked by its vocabulary size and the template markers'
  token IDs (hard-coded file hashes trip the repo's secret scanner), and by the positive control below;
- the markers: the model's own chat template (that snapshot's tokenizer_config.json) renders each tool result as
  `<|start_header_id|>ipython<|end_header_id|>\\n\\n` + `content | tojson` + `<|eot_id|>`, one block per result, so the
  needle is that header, json.dumps(content, ensure_ascii=False) (transformers' tojson), and `<|eot_id|>`; the window
  marker is `<|start_header_id|>assistant<|end_header_id|>\\n\\n`, whose last occurrence is the generation prompt;
- the span count: ipython blocks in the window against the request's new tool messages (2e's <tool_response> count);
- also counted: `checks_not_found_but_found_after_strip_json` (the stripped text, JSON-escaped, found), beside 2e's
  raw-text `checks_not_found_but_found_after_strip`, since a Llama prompt carries tool text JSON-escaped;
- a REAL-LOG HAND CHECK, a hard gate (2e, 10-12): one checked output picked by seed, from the outputs holding <, >, &
  or ' when there are any (an HTML-escaping tojson, jinja's own, would hide exactly those), must have its needle in its
  request's window exactly once, or the script writes no results; not-founds are also split by those characters and
  tested against jinja's escaping (`checks_not_found_but_found_htmlsafe`);
- no 10-06 re-render arm (there is no earlier Llama run); instead a POSITIVE CONTROL before any run is read: a small
  conversation with a tool result holding quotes, a newline, a tab and non-ASCII text is rendered through the snapshot's
  own template, tokenized with its tokenizer, decoded back to bytes by the same decoder, and the needle must be found
  exactly once (and the one-extra-byte needle 0 times); the script stops otherwise.

Writes results/tool_outputs_llama_<stamp>.json (counts only) and, outside git, the seeded example's masked excerpt.

  uv run --no-project --python 3.14 --with tokenizers --with jinja2 python -I check_tool_outputs_llama.py \\
      --data <dir with the run folders and their .request_logs> --runs <run folder>[,<run folder>...] \\
      --tokenizer-dir <dir with tokenizer.json and tokenizer_config.json> --excerpt-dir <scratch dir>
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

from tokenizers import Tokenizer

HERE = Path(__file__).parent
MODEL, REV = "meta-llama/Llama-3.3-70B-Instruct", "6f6073b423013f6a7d4d9f39144961bfbfbc386b"  # servers_llama33_70b.yaml
POLICY = "vllm/meta-llama/Llama-3.3-70B-Instruct"
# The snapshot's tokenizer, checked by what the search relies on: its size and the template markers' token IDs.
VOCAB, MARKER_IDS = 128256, {"<|begin_of_text|>": 128000, "<|start_header_id|>": 128006, "<|end_header_id|>": 128007, "<|eot_id|>": 128009}
SEED = 20261012
OPEN, CLOSE = b"<|start_header_id|>ipython<|end_header_id|>\n\n", b"<|eot_id|>"
ASSISTANT = b"<|start_header_id|>assistant<|end_header_id|>\n\n"
TRUNCATED = "was too long to be displayed"  # Inspect's own notice when it cuts a tool output
MISSING = (re.compile(r"The path (\S+) does not exist"), re.compile(r"cannot access '([^']+)': No such file"),
           re.compile(r"(?m)^(?:\S+: )?(?:line \d+: )?(/[^\s:]+): No such file or directory"))
LISTING = re.compile(r"^\s*(ls|find|tree)\b")


def tojson(x) -> bytes:
    """transformers' chat-template tojson (json.dumps, ensure_ascii=False, no indent) as bytes."""
    return json.dumps(x, ensure_ascii=False).encode()


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


def decoder(tok: Tokenizer):
    """token id -> bytes: added tokens from their content, the rest through the byte table (2e's decode)."""
    dec, added = byte_table(), {i: t.content.encode() for i, t in tok.get_added_tokens_decoder().items()}
    return lambda i: added[i] if i in added else bytes(dec[ch] for ch in tok.id_to_token(i))


def positive_control(tok: Tokenizer, template: str, bos: str) -> dict:
    """Render a known conversation through the snapshot's own template (transformers' jinja settings), tokenize it,
    decode it back with the check's decoder, and find the known tool result's needle exactly once."""
    import jinja2

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False, **k)

    def raise_exception(msg):
        raise ValueError(msg)

    env.globals["raise_exception"] = raise_exception
    known = 'line one "quoted"\n\tline two: café ≠ cafe\\path'
    msgs = [{"role": "system", "content": "You are a test."}, {"role": "user", "content": "List the files."},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": "bash", "arguments": {"cmd": "ls"}}}]},
            {"role": "tool", "content": known, "tool_call_id": "call_1"}]
    tools = [{"type": "function", "function": {"name": "bash", "description": "Run a command.",
                                                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}}]
    text = env.from_string(template).render(messages=msgs, tools=tools, add_generation_prompt=True, bos_token=bos, tools_in_user_message=False)
    to_bytes = decoder(tok)
    ids = tok.encode(text, add_special_tokens=False).ids
    prompt = b"".join(to_bytes(i) for i in ids)
    needle = OPEN + tojson(known) + CLOSE
    out = {"rendered_bytes": len(prompt), "decoded_equals_rendered": prompt == text.encode(),
           "needle_found": prompt.count(needle), "negative_found": prompt.count(OPEN + tojson(known) + b"\x00" + CLOSE),
           "ipython_blocks": prompt.count(OPEN), "generation_prompt_is_last_assistant_marker":
               prompt.rstrip(b"\n").endswith(ASSISTANT.rstrip(b"\n"))}
    assert out["decoded_equals_rendered"] and out["needle_found"] == 1 and out["negative_found"] == 0 and \
        out["ipython_blocks"] == 1 and out["generation_prompt_is_last_assistant_marker"], f"positive control failed: {out}"
    return out


def htmlsafe(x: str) -> bytes:  # jinja's own tojson filter (htmlsafe_json_dumps: json.dumps, ASCII, then these four)
    return json.dumps(x).encode().replace(b"<", b"\\u003c").replace(b">", b"\\u003e").replace(b"&", b"\\u0026").replace(b"'", b"\\u0027")


def check_runs(data: Path, runs: list[str], to_bytes) -> tuple[dict, list[dict], list[dict]]:
    per_run, located, checked = {}, [], []  # checked: every (output, request) pair, for the real-log hand check (2e)
    for run in runs:
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
                tb = [to_bytes(i) for i in ids]
                prompt = b"".join(tb)
                ends = [m.start() for m in re.finditer(re.escape(ASSISTANT), prompt)]
                lo, hi = (ends[-2] if len(ends) > 1 else 0), ends[-1]
                c["requests_generation_prompt_not_at_end"] += prompt[hi:].rstrip(b"\n") != ASSISTANT.rstrip(b"\n")
                cum, t = [], 0
                for x in tb:
                    t += len(x)
                    cum.append(t)
                n_open = prompt.count(OPEN, lo, hi)
                c["requests_ok_with_new_tool_messages"] += 1
                c["requests_ok_ipython_blocks_equal_new_tool_messages"] += n_open == len(new)
                pos = lo
                sents = [text_of(m["content"], att) for m in new]
                for m in new:
                    sent = text_of(m["content"], att)
                    assert "attachment://" not in sent[:13], "unresolved attachment"
                    ev_msg = by_id.get(m.get("tool_call_id"))
                    c["checks_sent_equals_eval_tool_message"] += ev_msg is not None and as_sent(ev_msg) == sent
                    c["checks_sent_differs_from_eval_tool_message"] += ev_msg is not None and as_sent(ev_msg) != sent
                    c["checks_sent_without_eval_tool_message"] += ev_msg is None
                    needle = OPEN + tojson(sent) + CLOSE
                    at = prompt.find(needle, pos, hi)
                    special = any(ch in sent for ch in "<>&'")
                    checked.append({"run": key, "step": step, "tool_call_id": m.get("tool_call_id"), "request_id": irid,
                                    "special_chars": special, "same_text_in_request": sents.count(sent),
                                    "needle_count_in_window": prompt.count(needle, lo, hi), "ordered_match": at >= 0})
                    c["negative_control_perturbed_needle_found"] += prompt.find(OPEN + tojson(sent) + b"\x00" + CLOSE, lo, hi) >= 0
                    t = m.get("tool_call_id")
                    if at < 0:
                        c["checks_not_found"] += 1
                        c["checks_not_found_with_special_chars"] += special
                        c["checks_not_found_but_found_htmlsafe"] += prompt.find(OPEN + htmlsafe(sent) + CLOSE, lo, hi) >= 0
                        c["checks_not_found_but_found_after_strip"] += prompt.find(sent.strip().encode(), lo) >= 0
                        c["checks_not_found_but_found_after_strip_json"] += prompt.find(tojson(sent.strip())[1:-1], lo) >= 0
                        outcome[t] = max(outcome.get(t, "not_found"), "not_found", key=rank.index)
                    else:
                        c["checks_found_byte_exact"] += 1
                        outcome[t] = "found_byte_exact"
                        a0, a1 = at + len(OPEN), at + len(OPEN) + len(tojson(sent))
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
    return per_run, located, checked


def main(data: str, runs: str, tokenizer_dir: str, excerpt_dir: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    td = Path(tokenizer_dir)
    tok = Tokenizer.from_file(str(td / "tokenizer.json"))
    assert tok.get_vocab_size(with_added_tokens=True) == VOCAB, tok.get_vocab_size(with_added_tokens=True)
    assert {m: tok.token_to_id(m) for m in MARKER_IDS} == MARKER_IDS, f"not the snapshot {REV} tokenizer"
    cfg = json.loads((td / "tokenizer_config.json").read_text())
    control = positive_control(tok, cfg["chat_template"], cfg.get("bos_token") or "")
    per_run, located, checked = check_runs(Path(data), runs.split(","), decoder(tok))
    # The real-log hand check, a hard gate (2e): one checked output, by seed, preferring outputs with <, >, & or ' (an
    # HTML-escaping tojson would hide exactly those); its needle must be in its request's window exactly once, and be
    # this output's own ordered match. Eligible: outputs whose text no other new tool message of the request shares
    # (chosen before looking at matches), so "exactly once" is what a correct render gives (Codex 10-12).
    unique = [x for x in checked if x["same_text_in_request"] == 1]
    pool = [x for x in unique if x["special_chars"]] or unique
    if not pool:
        raise SystemExit("no tool output was checked against a served request: no hand check, no results")
    hand = random.Random(f"{SEED}-hand-check").choice(pool) | {"pool": "special_chars" if pool is not unique else "all",
                                                              "pool_size": len(pool), "checked": len(checked)}
    print("hand check:", json.dumps(hand))
    if hand["needle_count_in_window"] != 1 or not hand["ordered_match"]:
        raise SystemExit(f"hand check failed: needle found {hand['needle_count_in_window']} times (ordered match "
                         f"{hand['ordered_match']}), not once; no results")
    total = sum((Counter(r["counts"]) for r in per_run.values()), Counter())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"git": git, "model": MODEL, "revision": REV, "seed": SEED, "base": "2e check_tool_outputs.py blob 14b4abec @ 05a4bceb",
           "positive_control": control, "hand_check": hand, "runs": per_run, "totals": dict(total), "seeded_example": None}
    rng = random.Random(SEED)  # the seeded example: one run (2e's population: every run), then one located output in it
    run = rng.choice(sorted(per_run))
    if not [x for x in located if x["run"] == run]:
        out["seeded_example"] = {"run": run, "located_outputs": 0}
    else:  # its excerpt (text) goes outside git
        pick = rng.choice([x for x in located if x["run"] == run])
        s = load_sample(Path(data) / per_run[run]["eval"])
        tool = next(m for m in s["messages"] if m["role"] == "tool" and m.get("tool_call_id") == pick["tool_call_id"])
        call = next(t for m in s["messages"] if m["role"] == "assistant" for t in m.get("tool_calls") or [] if t.get("id") == pick["tool_call_id"])
        cmd = (call.get("arguments") or {}).get("cmd") or (call.get("arguments") or {}).get("command") or ""
        listing = call.get("function") == "bash" and bool(LISTING.match(str(cmd)))
        sent = as_sent(tool)
        shown = sent if listing else re.sub(r"[A-Za-z0-9]", "x", sent)
        ex = {"run": run, "step": pick["step"], "tool_call_id": pick["tool_call_id"], "function": call.get("function"),
              "listing": listing, "masked": not listing, "excerpt_first_600_chars": shown[:600], "bytes": pick["bytes"],
              "token_first": pick["token_first"], "token_last": pick["token_last"], "prompt_tokens": pick["prompt_tokens"]}
        edst = Path(excerpt_dir) / f"tool_outputs_llama_example_{stamp}.json"
        edst.write_text(json.dumps(ex))
        out["seeded_example"] = pick | {"excerpt_file": edst.name, "excerpt_sha256": hashlib.sha256(edst.read_bytes()).hexdigest()}
    dst = HERE / "results" / f"tool_outputs_llama_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    print("positive control:", json.dumps(control))
    print("runs:", len(per_run), "totals:", json.dumps(dict(sorted(total.items()))))
    print("seeded example:", json.dumps(out["seeded_example"]))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True, help="run folder names under --data, comma-separated")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--excerpt-dir", required=True)
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
