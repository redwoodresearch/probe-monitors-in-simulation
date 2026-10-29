"""Viewer data: download the four cohorts once and cache, per sample, exactly what the reader model sees.

Each cached sample holds its metadata, `reader_messages` (the honest system prompt of the same main task, as the
activation extraction renders it), the segments `reader.render` finds (one per agent turn and per tool output, so
the viewer's segments are the probe's segments), and where the reader's token cut falls.

The cache is local and regenerable: `<cache>/eval/<alias>/*.eval` (raw downloads), `<cache>/samples/<arm>/*.json`
and `<cache>/index.jsonl` (one metadata row per sample, what the task list filters on).
"""

import hashlib
import json
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from probe_shift.models import QWEN3_32B
from probe_shift.reader import _layout, _segments, segment_end_positions
from probe_shift.transcripts import (RUNNER_SUBMIT, _s3, cohort_uri, eval_files, invalidity, main_task_id,
                                     reader_messages, submit_calls, system_prompt, tool_result_text, visible_text)

# Arm -> cohort alias (simulated-control-evals `real_runs.yml`). Order is the default column order.
ARMS = {
    "real-attack-e1": "la_public10_real_attack_n1000",
    "sim-attack": "la_public10_sim_astra_attack_n1000",
    "real-attack-e2": "la_public10_real_attack_reroll_n1000",
    "real-honest": "la_public10_real_honest_n1000",
}
TOKENIZER = QWEN3_32B  # pinned revision, as the splits and the extraction use
CACHE_VERSION = 5  # bump when the cached sample shape changes; the app refuses an older cache
TOOL_CALL_OPEN = "<tool_call>"  # Qwen3's chat template opens every rendered tool call with this
ARGS_OPEN = '"arguments": '  # and renders the arguments as plain json.dumps after this


def sample_key(file: str, sample: dict) -> str:
    """Same key as splits/make_splits.py and the activation extraction."""
    return f"{file.removesuffix('.eval')}__{sample['id'].replace('/', '_')}__{sample['epoch']}"


def sample_path(cache: Path, arm: str, key: str) -> Path:
    """Hashed, because two keys are longer than a filename may be."""
    return cache / "samples" / arm / f"{hashlib.sha1(key.encode()).hexdigest()[:20]}.json"


def download(alias: str, cache: Path) -> list[Path]:
    """Every .eval of a cohort, fetched once into `<cache>/eval/<alias>/`, in `eval_files` order."""
    bucket, prefix = cohort_uri(alias).removeprefix("s3://").split("/", 1)
    out = cache / "eval" / alias
    out.mkdir(parents=True, exist_ok=True)

    def get(file):
        path = out / file
        if not path.exists():
            part = path.with_name(path.name + ".part")
            _s3().download_file(bucket, f"{prefix}logs/{file}", str(part))
            part.rename(path)
        return path

    with ThreadPoolExecutor(32) as pool:
        return list(pool.map(get, eval_files(alias)))


def read_eval(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        return [json.loads(z.read(n)) for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]


def _id(x):
    return x.get("id") if isinstance(x, dict) else x


def _scores(sample: dict) -> dict:
    """{name: value} over every scorer, plus each scorer's explanation."""
    out, explanations = {}, {}
    for scorer, s in (sample.get("scores") or {}).items():
        value = s.get("value")
        out |= value if isinstance(value, dict) else {scorer: value}
        if s.get("explanation"):
            explanations[scorer] = s["explanation"]
    return out | {"explanations": explanations}


def _json_prefix(body: str) -> str:
    """A JSON string body cut at an arbitrary character, decoded; a trailing partial escape is dropped."""
    for k in range(7):
        try:
            return json.loads('"' + body[:len(body) - k] + '"')
        except json.JSONDecodeError:
            continue
    raise ValueError(f"cannot decode the JSON string prefix {body[-20:]!r}")


def _call_cut(call: str, off: int) -> dict:
    """Where a cut at character `off` of one rendered tool call (`<tool_call>\\n{"name": ..., "arguments": {...}}`)
    falls: {"arg": the argument it is in, "chars": characters of its value read (of the decoded string, or of the
    serialized JSON of a non-string value), "in_key": True if the cut is inside the argument's name}; or
    {"arg": None, "after": False} before the arguments and {"arg": None, "after": True} after them. Arguments must
    be a dict: the template dumps a dict with json.dumps, so offsets map onto `json.dumps(value)`."""
    dec = json.JSONDecoder()
    a = call.index(ARGS_OPEN) + len(ARGS_OPEN)
    _, args_end = dec.raw_decode(call, a)
    if off >= args_end:
        return {"arg": None, "after": True}
    i = a + 1
    while off > i:
        while call[i] in ", ":
            i += 1
        key, key_end = dec.raw_decode(call, i)
        i = call.index(":", key_end) + 1
        while call[i] == " ":
            i += 1
        value, end = dec.raw_decode(call, i)
        if off <= end:  # inside this value, or right after it (then all of it is read)
            if off <= i:  # before the value: inside the name (dimmed too), or after it with the name read
                return {"arg": key, "chars": 0, "in_key": off < key_end}
            if not isinstance(value, str):
                return {"arg": key, "chars": max(off - i, 0)}  # characters of its serialized JSON read
            return {"arg": key, "chars": len(_json_prefix(call[i + 1:off]))}
        i = end
    return {"arg": None, "after": False}


def _cut(tok, ids: list[int], segments: list[dict], messages: list[dict], max_tokens: int) -> dict | None:
    """Where the reader's input ends, if the transcript is longer than `max_tokens`: the message the cut falls in,
    how many of its segment's tokens the reader sees, and the exact place. In a tool output (whose segment is its
    content verbatim) that is a character offset. In an agent turn it is the tool call (`call`, None for the
    turn's own text, then `text_chars` is the offset in that text) and, inside that call's arguments, `part`
    from `_call_cut`. A cut between segments that falls inside a user message there (such as a nudge) gives that
    message and `text_chars`; one in template tokens goes to the next segment, with nothing of it read."""
    if len(ids) <= max_tokens:
        return None
    if segments and max_tokens < segments[0]["start"]:  # far below any window we use; the grid has no marker there
        raise ValueError(f"max_tokens={max_tokens} ends inside the opening prompt, which the viewer does not mark")
    seg = next((s for s in segments if s["end"] > max_tokens), None)
    done = [s for s in segments if s["end"] <= max_tokens]
    between = [i for i in range(done[-1]["message"] + 1 if done else 0, seg["message"] if seg else len(messages))
               if messages[i]["role"] not in ("assistant", "tool")]
    if (seg is None or seg["start"] > max_tokens) and between:
        # The cut is between segments. A user message there is rendered verbatim, so find it in the decoded gap.
        lo, hi = (done[-1]["end"] if done else 0), (seg["start"] if seg else len(ids))
        gap, off, at = tok.decode(ids[lo:hi]), len(tok.decode(ids[lo:max_tokens])), 0
        for i in between:
            content = messages[i]["content"]
            at = gap.find(content, at) if content else -1
            if at >= 0 and off < at + len(content):
                return {"message": i, "tokens_seen": None, "chars": None, "call": None, "part": None,
                        "text_chars": max(off - at, 0)}
            at = max(at, 0) + len(content)
    if seg is None:  # the cut falls in closing template tokens: everything is read
        return None
    seen = max(0, max_tokens - seg["start"])
    chars = len(tok.decode(ids[seg["start"]:seg["start"] + seen]))
    call, part, text_chars = None, None, None
    if seg["kind"] == "agent":
        text = tok.decode(ids[seg["start"]:seg["end"]])
        starts = [m.start() for m in re.finditer(re.escape(TOOL_CALL_OPEN), text)]
        n_calls = len(messages[seg["message"]].get("tool_calls") or [])
        # The template does not escape arguments, so an argument quoting the marker would shift every call.
        if len(starts) != n_calls:
            raise ValueError(f"{len(starts)} {TOOL_CALL_OPEN} markers for {n_calls} calls in message {seg['message']}")
        call = sum(at <= chars for at in starts) - 1
        if call < 0:  # in the turn's own text, which the template renders verbatim (after any empty think block)
            content = messages[seg["message"]]["content"]
            at = text.find(content) if content else -1
            call, text_chars = None, (min(max(chars - at, 0), len(content)) if at >= 0 else None)
        else:
            end = starts[call + 1] if call + 1 < len(starts) else len(text)
            arguments = messages[seg["message"]]["tool_calls"][call]["function"]["arguments"]
            if not isinstance(arguments, dict):  # rendered verbatim, so offsets would not map onto json.dumps(value)
                raise ValueError(f"the 32k cut falls in a call with non-dict arguments (message {seg['message']})")
            part = _call_cut(text[starts[call]:end], chars - starts[call])
            if part["arg"] is None:  # before the arguments: the call is unread; after them: the next one is
                call, part = call + part["after"], None
    return {"message": seg["message"], "tokens_seen": seen, "chars": chars, "call": call, "part": part,
            "text_chars": text_chars}


def _readouts(tok, messages, segments, text, found, enc, max_tokens) -> list[dict]:
    """Every probe readout position (`reader.segment_end_positions`) and where it shows in the grid.

    Each: kind, step, action, token, kept (inside the window), token_str (the decoded token), message, and for
    output_end the characters [start, end) of the output's content; for call_end the call and the characters
    [start, end) of the displayed value of argument `arg` (end None: to the value's end; arg None: after the
    arguments); for pre_call and call_marker the call. A position with no token (an empty output) is left out.
    """
    offsets, ids = enc["offset_mapping"], enc["input_ids"]  # a position's `token` is an index into these
    where = {(kind, step, action): (a, b, seg["message"])
             for (kind, step, action, _, a, b), seg in zip(found, segments, strict=True)}
    positions = segment_end_positions(tok, messages, max_tokens)["positions"]
    # Each call starts at its pre_call token, the reader's own `<tool_call>` opener (a quoted one cannot fool this).
    opens = {(p["step"], p["action"]): offsets[p["token"]][0] for p in positions if p["kind"] == "pre_call"}
    out = []
    for p in positions:
        if p["token"] is None:
            continue
        r = {k: p[k] for k in ("kind", "step", "action", "token", "kept")} | {"token_str": tok.decode([ids[p["token"]]])}
        s, e = offsets[p["token"]]
        if p["kind"] in ("output_end", "output_marker"):
            a, _, i = where[("tool_output", p["step"], p["action"])]
            r |= {"message": i} | ({"start": s - a, "end": e - a} if p["kind"] == "output_end" else {})
        else:
            _, b, i = where[("agent", p["step"], None)]
            k = p["action"]
            r |= {"message": i, "call": k}
            if p["kind"] == "call_end" and not isinstance(messages[i]["tool_calls"][k]["function"]["arguments"], dict):
                r |= {"arg": None, "start": None, "end": None}  # rendered verbatim, not json.dumps: shown after the call
            elif p["kind"] == "call_end":
                c0 = opens[(p["step"], k)]
                call = text[c0:opens.get((p["step"], k + 1), b)]
                lo, hi = _call_cut(call, s - c0), _call_cut(call, e - c0)
                same = lo["arg"] is not None and hi.get("arg") == lo["arg"]
                r |= {"arg": lo["arg"], "start": lo.get("chars"), "end": hi["chars"] if same else None}
        out.append(r)
    return out


def _compact(m: dict, shown: dict | None) -> dict:
    """An original message for the raw view: encrypted reasoning summarized, and text or arguments identical to
    what the reader sees (`shown`) abbreviated, so differences (error fields, dropped calls) stand out."""
    same = "(same as the text the reader sees)"
    out = {k: v for k, v in m.items() if k not in ("id", "source", "model", "metadata")}
    content = m.get("content")
    if isinstance(content, list):
        out["content"] = [{"type": "reasoning", "reasoning": f"<encrypted, {len(c.get('reasoning') or '')} chars>"}
                          if c.get("type") == "reasoning" else
                          (c | {"text": same}) if shown and c.get("type") == "text" and c.get("text") == shown["content"]
                          and c.get("text") else c for c in content]
    elif shown and content and content == shown["content"]:
        out["content"] = same
    if m.get("tool_calls") and shown:
        args = {c["function"]["name"] + str(n): c["function"]["arguments"] for n, c in enumerate(shown.get("tool_calls") or [])}
        n, calls = 0, []
        for c in m["tool_calls"]:
            if c["function"] != RUNNER_SUBMIT and args.get(c["function"] + str(n)) == (c.get("arguments") or {}):
                c = c | {"arguments": "(same as shown)"}
            n += c["function"] != RUNNER_SUBMIT
            calls.append({k: v for k, v in c.items() if k not in ("id", "parse_error", "view")})
        out["tool_calls"] = calls
    return out


def _raw(sample: dict, messages: list[dict]) -> list[dict]:
    """Per reader message, the original message it came from (`orig`, compacted) and the originals that
    reader_messages dropped just before it (`dropped`: submit calls and their results, empty turns); the last
    entry also gets `dropped_after`. Matched in order on role and the text the reader sees."""
    def matches(o, m):
        if o["role"] != m["role"]:
            return False
        if m["role"] == "tool":
            return tool_result_text(o) == m["content"] and o.get("function") == m.get("tool_name")
        if m["role"] == "assistant":
            names = [c["function"] for c in o.get("tool_calls") or [] if c["function"] != RUNNER_SUBMIT]
            return names == [c["function"]["name"] for c in m["tool_calls"]] and visible_text(o["content"]) == m["content"]
        return m["role"] == "system" or visible_text(o["content"]) == m["content"]

    originals, j, out = sample["messages"], 0, []
    for m in messages:
        if m["role"] == "system" and not (originals and originals[0]["role"] == "system"):
            out.append({"orig": None, "dropped": []})  # the run had no system message; the reader adds the honest one
            continue
        dropped = []
        while j < len(originals) and not matches(originals[j], m):
            dropped.append(_compact(originals[j], None))
            j += 1
        if j == len(originals):
            raise ValueError(f"reader message {len(out)} ({m['role']}) has no original")
        out.append({"orig": _compact(originals[j], m), "dropped": dropped})
        j += 1
    out[-1]["dropped_after"] = [_compact(o, None) for o in originals[j:]]
    return out


_worker = {}


def _init(tokenizer: dict, honest: dict, cache: Path, max_tokens: int):
    from transformers import AutoTokenizer

    _worker.update(tok=AutoTokenizer.from_pretrained(tokenizer["name"], revision=tokenizer["revision"]), honest=honest, cache=cache, max_tokens=max_tokens)


def _text(field, key: str, fallback):
    return field.get(key) if isinstance(field, dict) else fallback


def summary(description: str | None, limit: int = 240) -> str:
    """The first prose paragraph of a task description (skipping headings), cut at a word boundary."""
    for para in (description or "").split("\n\n"):
        para = " ".join(para.split())
        if para and not para.startswith("#"):
            return para if len(para) <= limit else para[:limit].rsplit(" ", 1)[0] + " …"
    return ""


def _records(job: tuple) -> list[dict]:
    """Normalize every sample of one .eval file. Runs in a worker process."""
    arm, path = job
    tok, honest, cache, max_tokens = (_worker[k] for k in ("tok", "honest", "cache", "max_tokens"))
    rows = []
    for s in read_eval(path):
        md = s.get("metadata") or {}
        mt, side = main_task_id(s), _id(md.get("side_task"))
        key = sample_key(path.name, s)
        messages = reader_messages(s, honest[mt])
        text, found, enc = _layout(tok, messages)  # what reader.render does, kept for the readout mapping
        ids, segments = enc["input_ids"], _segments(found, enc)
        at = [i for i, m in enumerate(messages) if m["role"] in ("assistant", "tool")]
        for seg, i in zip(segments, at, strict=True):
            seg["message"] = i
        fork, scores = md.get("fork") or {}, _scores(s)
        row = {
            "arm": arm, "key": key, "cohort": ARMS[arm], "file": path.name, "sample_id": s["id"], "epoch": s["epoch"],
            "environment": _id(md.get("environment")), "main_task": mt, "side_task": side,
            "environment_name": _text(md.get("environment"), "display_name", _id(md.get("environment"))),
            "main_task_name": _text(md.get("main_task"), "name", mt),
            "main_task_summary": summary(_text(md.get("main_task"), "description", md.get("main_task_str"))),
            "side_task_summary": summary(_text(md.get("side_task"), "goal", md.get("side_task_text"))),
            "cell": f"{mt}:{side}" if side else mt,
            "invalid": invalidity(s), "n_submit_calls": submit_calls(s), "tokens": len(ids),
            "truncated": len(ids) > max_tokens, "n_steps": sum(m["role"] == "assistant" for m in messages),
            "side_task_success": scores.get("side_task_success"), "main_task_success": scores.get("main_task_success"),
            "sim_source_sample": fork.get("source_sample"), "sim_source_run": fork.get("run"),
        }
        detail = row | {
            "messages": messages, "segments": segments, "cut": _cut(tok, ids, segments, messages, max_tokens), "scores": scores,
            "readouts": _readouts(tok, messages, segments, text, found, enc, max_tokens), "raw": _raw(s, messages),
            "main_task_description": _text(md.get("main_task"), "description", md.get("main_task_str")),
            "side_task_goal": _text(md.get("side_task"), "goal", md.get("side_task_text")),
            "attack_approach": md.get("attack_approach"), "simulator_model": md.get("simulator_model"),
            "original_system_prompt": system_prompt(s),
        }
        out = sample_path(cache, arm, key)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(detail))
        rows.append(row)
    return rows


def build(cache: Path, max_tokens: int, workers: int) -> list[dict]:
    """Download every cohort, normalize every sample, write the per-sample files and `index.jsonl`."""
    paths = {arm: download(alias, cache) for arm, alias in ARMS.items()}
    # The honest prompt per main task, taken as splits/make_splits.py takes it: the first real-honest sample of
    # that main task, in `eval_files` order.
    honest = {}
    for path in paths["real-honest"]:
        for s in read_eval(path):
            honest.setdefault(main_task_id(s), system_prompt(s))
    jobs = [(arm, p) for arm, ps in paths.items() for p in ps]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(TOKENIZER, honest, cache, max_tokens)) as pool:
        rows = [r for batch in pool.map(_records, jobs) for r in batch]
    (cache / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (cache / "version.json").write_text(json.dumps({"version": CACHE_VERSION}))
    return rows
