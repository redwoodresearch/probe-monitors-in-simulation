"""The Llama-3.3-70B smoke's verdict inputs, from its raw .eval files and its server's request log. Read-only, $0.

  uv run --no-project --python 3.14 --with tokenizers python analyze_smoke.py --run <smoke run dir copy> \\
      --tokenizer-dir <dir with tokenizer.json> [--compare <a Qwen3-32B native .eval of the same task>]

Per run (.eval messages): the first user message's length, plain-text assistant replies, tool calls by function
(submit apart), tool messages, the ending and the real scorer's main-task value. Per policy request (joined to the
proxy's 200 record by x-irid, prompt_token_ids decoded with the snapshot's tokenizer): the prompt's turns by role
(Llama's <|start_header_id|>role<|end_header_id|> headers), whether the tool list is in the first user turn (the
template's "Given the following functions" preamble), and how many bytes follow the tool list in that turn. --compare
counts the same .eval fields for a Qwen run. Writes results/smoke_<stamp>.json (counts only); the assistant's text
replies print to the console, quoted with run and step.
"""

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from check_tool_outputs_llama import POLICY, decoder, load_sample, text_of
from tokenizers import Tokenizer

HERE = Path(__file__).parent
HEADER = re.compile(rb"<\|start_header_id\|>(\w+)<\|end_header_id\|>\n\n")
PREAMBLE = b"Given the following functions"
SYSTEM_TOOLS = b"You have access to the following functions"  # the template with tools_in_user_message=false (fix A)


def eval_counts(path: Path) -> tuple[dict, list[str]]:
    s = load_sample(path)
    msgs = s["messages"]
    users = [m for m in msgs if m["role"] == "user"]
    calls = Counter(t["function"] for m in msgs if m["role"] == "assistant" for t in m.get("tool_calls") or [])
    texts = [text_of(m["content"]) for m in msgs if m["role"] == "assistant" and not m.get("tool_calls")]
    score = (((s.get("scores") or {}).get("linux_scorer") or {}).get("value") or {}).get("main_task_success")
    return {"eval": path.name, "eval_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "messages": len(msgs),
            "first_user_message_chars": len(text_of(users[0]["content"])) if users else None,
            "system_message_chars": sum(len(text_of(m["content"])) for m in msgs if m["role"] == "system"),
            "assistant_plain_text_replies": len(texts), "tool_calls_by_function": dict(calls),
            "non_submit_tool_calls": sum(n for f, n in calls.items() if f != "submit"),
            # both Llama templates refuse to render an assistant message with more than one tool call (README)
            "assistant_messages_with_more_than_one_tool_call": sum(len(m.get("tool_calls") or []) > 1 for m in msgs if m["role"] == "assistant"),
            "tool_messages": sum(m["role"] == "tool" for m in msgs),
            "ended_by": (s.get("store") or {}).get("policy_ended_by"), "main_task_raw": score}, texts


def main(run: str, tokenizer_dir: str, compare: str | None):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    run_dir = Path(run)
    to_bytes = decoder(Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json")))
    recs, other = {}, {}
    for f in sorted((run_dir.parent / f"{run_dir.name}.request_logs").glob("requests_*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("kind") != "header" and r.get("status") == 200 and r.get("prompt_token_ids"):
                    recs[r["request_id"]] = r
                elif r.get("kind") != "header":  # refused or cancelled requests (e.g. a template render error: HTTP 400)
                    other.setdefault(r.get("request_id"), []).append(str(r.get("status")))
    out = {"git": git, "run": run_dir.name, "request_records_ok": len(recs), "runs": {}}
    for ev_path in sorted(p for p in run_dir.rglob("*.eval") if "references" not in p.parts):
        key = "/".join(ev_path.relative_to(run_dir).parts[:2])
        counts, texts = eval_counts(ev_path)
        for i, t in enumerate(texts):
            print(f"[{run_dir.name} {key}, plain-text reply {i + 1}] {t!r}")
        reqs, events = [], [e for e in load_sample(ev_path)["events"] if e.get("event") == "model" and e.get("model") == POLICY]
        counts["policy_events_with_error"] = sum(bool(e.get("error")) for e in events)
        counts["policy_requests_with_non_200_record_by_status"] = dict(Counter(
            st for e in events for st in other.get((((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid"), [])))
        for e in events:
            if e.get("error"):
                print(f"[{run_dir.name} {key}, policy event error] {str(e['error'])[:160]!r}")
        for step, e in enumerate(events):
            irid = (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid")
            r = recs.get(irid)
            if r is None:
                reqs.append({"step": step, "proxy_200_record": False})
                continue
            prompt = b"".join(to_bytes(i) for i in r["prompt_token_ids"])
            heads = list(HEADER.finditer(prompt))
            roles = [h.group(1).decode() for h in heads]
            first_user = next((k for k, h in enumerate(heads) if h.group(1) == b"user"), None)
            turn = b""
            if first_user is not None:
                end = heads[first_user + 1].start() if first_user + 1 < len(heads) else len(prompt)
                turn = prompt[heads[first_user].end():end]
            tools_end = turn.rfind(b"}\n\n")  # the template ends the tool list with "\n\n", then the user's content
            k_sys = next((k for k, h in enumerate(heads) if h.group(1) == b"system"), None)
            system = prompt[heads[k_sys].end():heads[k_sys + 1].start()] if k_sys is not None and k_sys + 1 < len(heads) else b""
            reqs.append({"step": step, "proxy_200_record": True, "prompt_tokens": len(r["prompt_token_ids"]),
                         "roles_in_prompt": roles, "tool_list_in_first_user_turn": turn.startswith(PREAMBLE),
                         "tool_list_in_system_turn": SYSTEM_TOOLS in system,
                         "first_user_turn_content_bytes": len(turn.removesuffix(b"<|eot_id|>")),
                         "bytes_after_tool_list_in_first_user_turn": len(turn) - (tools_end + 3) - len(b"<|eot_id|>")
                         if turn.startswith(PREAMBLE) and tools_end >= 0 else None})
        out["runs"][key] = counts | {"policy_requests": reqs}
    if compare:
        out["compare"], texts = eval_counts(Path(compare))
        for i, t in enumerate(texts[:2]):
            print(f"[compare {Path(compare).name}, plain-text reply {i + 1}] {t[:300]!r}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"smoke_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--compare")
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
