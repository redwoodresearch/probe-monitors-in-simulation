"""Does "strict": false bring back str_replace's new_str? (10-08: in the pilot's real runs, tools went out strict and
vLLM's ordered constrained decoding dropped new_str.) On the devbox, against one running server, $0 beyond its GPU:

  check_strict_replay.py <pilot run dir> <base url> <out.jsonl> --raw-dir <dir> [--n 5] [--samples 2] [--model <name>]

1. Replay: n recorded requests of the pilot's real runs whose reply was a str_replace without new_str, sent as
   recorded (strict true) and with every tool strict false, `samples` times each.
2. Crafted edits, with the real text_editor schema and the recorded system prompt, strict false and true: a
   replacement, two replacements in one file, an insertion, and an intentional deletion of a line.

Writes one JSON line per reply: the case, the strict flag, and each text_editor call's command and argument keys in
order, with new_str's length (never model text). Each request as sent and its full reply go to --raw-dir (model text:
a git-ignored data/ folder or the devbox, archived to S3 from the laptop), so every verdict can be audited. Passes if
every strict-false reply does the asked edit with its text: each str_replace carries new_str (empty only for the
deletion), and the insertion carries insert_text (or a str_replace with new_str). --model replays against another
served model (the Qwen3-32B arm's server). It also fails unless strict-on replies still drop new_str (the positive
control).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from inspect_ai.log import read_eval_log
from openai import OpenAI
from openai.types.chat import ChatCompletion

FILE = "/app/settings.py"
TEXT = "DEBUG = True\na = 1\nb = 2\nc = 4\nNAME = 'svc'\n"
SHOWN = "".join(f"{i}\t{line}" for i, line in enumerate(TEXT.splitlines(keepends=True), 1))
LINES = TEXT.splitlines(keepends=True)
CRAFTED = {  # the ask, and every file state that counts as done (one edit per turn is fine for replace_twice)
    "replace": (f"In {FILE} replace `a = 1` with `a = 10`.", [TEXT.replace("a = 1\n", "a = 10\n")]),
    "replace_twice": (f"In {FILE} replace `a = 1` with `a = 10` and `b = 2` with `b = 20`.",
                      [TEXT.replace("a = 1\n", "a = 10\n").replace("b = 2\n", "b = 20\n"),
                       TEXT.replace("a = 1\n", "a = 10\n"), TEXT.replace("b = 2\n", "b = 20\n")]),
    "insert": (f"In {FILE} insert the line `TIMEOUT = 30` after line 3.", ["".join([*LINES[:3], "TIMEOUT = 30\n",
                                                                                 *LINES[3:]])]),
    # A deletion of the line's text that leaves it empty (old_str without its newline) is the same edit, not the bug.
    "delete": (f"In {FILE} delete the line `c = 4` entirely, using str_replace.", [TEXT.replace("c = 4\n", ""),
                                                                                  TEXT.replace("c = 4", "")]),
}


def apply(text: str, args: dict) -> str:
    """text_editor's edit on an in-memory file; raises on anything the real tool would refuse or misapply."""
    command = args.get("command")
    if command == "view":
        return text  # looking first is fine: a view changes nothing
    assert args.get("path") == FILE, f"path must be {FILE}"
    if command == "str_replace":
        old, new = args.get("old_str"), args.get("new_str")
        assert isinstance(old, str) and isinstance(new, str), "old_str and new_str must both be strings"
        assert text.count(old) == 1, "old_str must occur exactly once"
        return text.replace(old, new, 1)
    if command == "insert":
        line = args.get("insert_line")
        new = args["new_str"] if args.get("new_str") is not None else args.get("insert_text")  # the tool's precedence
        lines = text.splitlines(keepends=True)
        assert isinstance(line, int) and not isinstance(line, bool) and 0 <= line <= len(lines), "insert_line in range"
        assert isinstance(new, str) and new, "inserted text non-empty"
        return "".join([*lines[:line], new if new.endswith("\n") else new + "\n", *lines[line:]])
    raise AssertionError(f"not an edit: {command!r}")


def norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip("\n").splitlines())


def parse(raw) -> dict | None:  # noqa: ANN001
    """A call's arguments, or None when they aren't a JSON object (counted as an error, never as a pass)."""
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def resolved(value, attachments: dict):  # noqa: ANN001, ANN201
    """The logged request or response with Inspect's `attachment://<hash>` references (long strings stored once per
    sample) replaced by their text, so a replay sends what was sent."""
    if isinstance(value, str) and value.startswith("attachment://"):
        return attachments[value.removeprefix("attachment://")]
    if isinstance(value, dict):
        return {k: resolved(v, attachments) for k, v in value.items()}
    if isinstance(value, list):
        return [resolved(v, attachments) for v in value]
    return value


def broken_requests(run_dir: Path, n: int) -> list[dict]:
    found = []
    for path in sorted(run_dir.glob("*/*/*.eval"), key=lambda p: (p.parts[-2], p.parts[-3])):  # arms interleaved
        if path.parts[-3] not in ("honest_a", "honest_b", "attack"):
            continue
        sample = read_eval_log(path).samples[0]
        for event in sample.events:
            call = getattr(event, "call", None)
            if getattr(event, "event", "") != "model" or not call or not isinstance(call.response, dict):
                continue
            message = (call.response.get("choices") or [{}])[0].get("message", {})
            for tc in message.get("tool_calls") or []:
                args = parse(resolved(tc["function"]["arguments"], sample.attachments or {}))
                if args and args.get("command") == "str_replace" and "new_str" not in args:
                    request = resolved(call.request, sample.attachments or {})
                    assert "attachment://" not in json.dumps(request), f"{path}: unresolved attachment"
                    found.append({"source": str(path), "request": request})
                    break
            if len(found) and found[-1]["source"] == str(path):
                break  # one per run, so the n come from n runs
        if len(found) == n:
            return found
    return found


def with_strict(request: dict, strict: bool) -> dict:
    out = copy.deepcopy(request)
    for tool in out.get("tools", []):
        tool["function"]["strict"] = strict
    return out


def calls(reply) -> list[tuple[dict, dict | None]]:  # noqa: ANN001
    """Every tool call as (a loggable row: tool, command, keys, new_str length; its parsed arguments, in memory)."""
    out = []
    for tc in reply.choices[0].message.tool_calls or []:
        args = parse(tc.function.arguments)
        row = {"tool": tc.function.name, "command": (args or {}).get("command"), "keys": list(args or {}) or None,
               "unparsed": args is None,
               "new_str_len": len(args["new_str"]) if args and isinstance(args.get("new_str"), str) else None}
        out.append((row, args))
    return out


def passes(name: str, pairs: list[tuple[dict, dict | None]]) -> bool:
    """A strict-false reply passes when every call parses and is a text_editor call, every str_replace carries a
    string new_str, and, for a crafted case, applying its edits in order gives one of the expected files."""
    if not pairs or any(row["unparsed"] or row["tool"] != "text_editor" for row, _ in pairs):
        return False
    if any(row["command"] == "str_replace" and row["new_str_len"] is None for row, _ in pairs):
        return False  # the bug: a replacement without its text
    if name not in CRAFTED:  # a replayed request: it asked for a replacement before, so its text must be non-empty
        replaces = [row for row, _ in pairs if row["command"] == "str_replace"]
        return bool(replaces) and all(row["new_str_len"] > 0 for row in replaces)
    text = TEXT
    try:
        for _, args in pairs:
            text = apply(text, args)
    except AssertionError:
        return False
    return norm(text) in {norm(t) for t in CRAFTED[name][1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("base_url")
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--model", help="the served model to ask instead of the recorded one")
    ap.add_argument("--raw-dir", type=Path, required=True, help="where each request and reply is saved (not git)")
    args = ap.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=False)  # one folder per replay run: never overwrite raw evidence
    assert not args.out.exists(), f"{args.out} exists: each replay run writes its own file"
    client = OpenAI(base_url=args.base_url, api_key=os.environ["VLLM_API_KEY"], timeout=1800)
    recorded = broken_requests(args.run_dir, args.n)
    assert len(recorded) == args.n, f"found {len(recorded)} broken requests, wanted {args.n}"
    cases = []
    for i, item in enumerate(recorded):
        for strict in (True, False):
            cases.append((f"replay_{i}", strict, with_strict(item["request"], strict), item["source"]))
    base = recorded[0]["request"]
    system = [m for m in base["messages"] if m.get("role") == "system"][:1]
    for name, (ask, _) in CRAFTED.items():
        content = f"{FILE} currently reads (cat -n):\n{SHOWN}\n{ask} Use the text_editor tool."
        messages = [*system, {"role": "user", "content": content}]
        for strict in (False, True):
            request = with_strict({k: v for k, v in base.items() if k not in ("messages",)} | {"messages": messages},
                                  strict)
            cases.append((name, strict, request, "crafted"))
    failures, total = [], 0
    with args.out.open("a") as f:
        for name, strict, request, source in cases:
            for k in range(args.samples):
                body = {key: value for key, value in request.items() if key not in ("extra_headers",)}
                if args.model:
                    body["model"] = args.model
                # The recorded body as-is (Qwen's `reasoning` fields included), not re-typed by the SDK's create().
                reply = client.post("/chat/completions", body=body, cast_to=ChatCompletion)
                raw = args.raw_dir / f"{name}_{'strict' if strict else 'unstrict'}_{k}.json"
                raw.write_text(json.dumps({"request": body, "reply": reply.model_dump(), "source": source}))
                pairs = calls(reply)
                rows = [row for row, _ in pairs]
                ok = passes(name, pairs)
                row = {"case": name, "strict": strict, "sample": k, "source": source, "calls": rows, "passes": ok,
                       "raw": str(raw),
                       "finish_reason": reply.choices[0].finish_reason}
                f.write(json.dumps(row) + "\n")
                f.flush()
                if not strict:
                    total += 1
                    if not ok:
                        failures.append((name, k, rows))
                print(name, "strict" if strict else "unstrict", k, [(r["command"], r["keys"]) for r in rows],
                      flush=True)
    print(f"strict-false replies: {total}, failing: {len(failures)}")
    for failure in failures:
        print("FAIL", failure)
    # Positive control: strict on must still drop new_str, or a pass doesn't show strict mode was the cause.
    rows = [json.loads(line) for line in args.out.read_text().splitlines()]  # this run's own file (asserted new)
    strict_replaces = [c for r in rows if r["strict"] and r["case"] != "delete"  # a deletion may omit new_str
                       for c in r["calls"] if c["command"] == "str_replace"]
    dropped = sum(c["new_str_len"] is None for c in strict_replaces)
    print(f"positive control: strict-on str_replace without new_str: {dropped} of {len(strict_replaces)}")
    raise SystemExit(1 if failures or not dropped else 0)


if __name__ == "__main__":
    main()
