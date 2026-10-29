"""A pilot smoke's verdict inputs (gpt-oss-120b or Qwen3-Coder-Next), from its raw .eval files and its server's request
log. Read-only, $0. The Seed pilot's analyze_smoke_seed.py (83465c61), changed only for each model's prompt format:
qcoder's `<|im_start|>role\n` turns with the tool list (`<tools>`) in the system turn; gptoss's harmony
`<|start|>role` turns with the tool list (`namespace functions`) in the developer turn.

  uv run --no-project --python 3.14 --with tokenizers python analyze_smoke_pilots.py --model <gptoss|qcoder> \\
      --run <smoke run dir copy> --tokenizer-dir <dir with tokenizer.json>

Per run (.eval messages): the first user message's length, plain-text assistant replies, tool calls by function
(submit apart), tool messages, the ending and the real scorer's main-task value. Per policy request (joined to the
proxy's 200 record by x-irid, prompt_token_ids decoded with the snapshot's tokenizer): the prompt's turns by role
(per model, above), whether the tool list is in the system or developer turn
blocks), the first user turn's content bytes (the September harness sends it empty), and the prompt's length. --compare
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

import check_tool_outputs_pilots as CK
from check_tool_outputs_pilots import decoder, load_sample, text_of
from tokenizers import Tokenizer

HERE = Path(__file__).parent
FMT = {"qcoder": {"header": re.compile(rb"<\|im_start\|>(\w+)\n?"), "tools": b"<tools>", "tools_turn": b"system", "eos": b"<|im_end|>",
                  "gen": b"<|im_start|>assistant\n"},
       "gptoss": {"header": re.compile(rb"<\|start\|>(\w+)"), "tools": b"namespace functions", "tools_turn": b"developer",
                  "eos": b"<|end|>", "gen": b"<|start|>assistant"}}
HEADER = SYSTEM_TOOLS = EOS = GEN = TOOLS_TURN = POLICY = None  # set by main() from --model


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
            "assistant_messages_with_more_than_one_tool_call": sum(len(m.get("tool_calls") or []) > 1 for m in msgs if m["role"] == "assistant"),
            "tool_messages": sum(m["role"] == "tool" for m in msgs),
            "ended_by": (s.get("store") or {}).get("policy_ended_by"), "main_task_raw": score}, texts


def main(model: str, run: str, tokenizer_dir: str, compare: str | None):
    global HEADER, SYSTEM_TOOLS, EOS, GEN, TOOLS_TURN, POLICY
    f = FMT[model]
    HEADER, SYSTEM_TOOLS, EOS, GEN, TOOLS_TURN = f["header"], f["tools"], f["eos"], f["gen"], f["tools_turn"]
    POLICY = "vllm/" + CK.MODELS[model]["model"]
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
            k_sys = next((k for k, h in enumerate(heads) if h.group(1) == TOOLS_TURN), None)
            system = prompt[heads[k_sys].end():heads[k_sys + 1].start()] if k_sys is not None and k_sys + 1 < len(heads) else b""
            reqs.append({"step": step, "proxy_200_record": True, "prompt_tokens": len(r["prompt_token_ids"]),
                         "roles_in_prompt": roles, "tool_list_in_first_user_turn": SYSTEM_TOOLS in turn,
                         "tool_list_in_system_turn": SYSTEM_TOOLS in system,
                         "first_user_turn_content_bytes": len(turn.removesuffix(EOS).removeprefix(b"<|message|>").removesuffix(EOS)),
                         "generation_prompt_at_end": prompt.endswith(GEN)})
        out["runs"][key] = counts | {"policy_requests": reqs}
    if compare:
        out["compare"], texts = eval_counts(Path(compare))
        for i, t in enumerate(texts[:2]):
            print(f"[compare {Path(compare).name}, plain-text reply {i + 1}] {t[:300]!r}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"smoke_{model}_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(FMT))
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--compare")
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
