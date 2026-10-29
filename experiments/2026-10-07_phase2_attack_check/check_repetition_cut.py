"""Checks (a) and (a') for the repetition guard, vLLM's side: would cutting a runaway turn where the guard stops it
change what vLLM returns? (a) is decision D on the reruns' 3 runaway submit turns; (a') is the multi_token setting on
step 2's reasoning loop. On Modal (CPU, the server's own vLLM 0.30.0), with the request logs and the HF cache on
volumes:

  uv run --no-project --python 3.12 --with modal==1.2.6 --with pyyaml modal run \\
      experiments/2026-10-07_phase2_attack_check/check_repetition_cut.py --calls <analyze_repetition_cut.py prepare's JSON> \\
      [--setting multi_token] [--log requests_2026-10-07_235557.jsonl.gz]

For each call: the token at which vLLM 0.30's own repetition check (check_sequence_repetition, applied after every
token as the scheduler does) stops the generation under D's settings; the full and the cut output decoded as vLLM
decodes (skip_special_tokens, its default); and both parsed by the server's parser (ParserManager with the qwen3
reasoning parser and the qwen3_coder tool parser, as `vllm serve` builds it), against the request's own tools. For a
call with a next policy call (prepare records it), also whether the next call's logged prompt holds any 40-token
chunk of the turn's output (sampled every 250 tokens, before and after the cut) and how the turn is rendered there.
analyze_repetition_cut.py compare then takes both parses through Inspect's conversion.

The full parse (the policy's reasoning and text) is written to the git-ignored data/phase2_attack_check/ and archived
to S3; results/ gets only its lengths, hashes, stop points and verdicts.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data" / "phase2_attack_check"  # git-ignored: raw model text never goes in git
APP = "probe-shift_aswin-surya_sub-phase2_rep-cut_kill-2026-11-06"  # Modal names: at most 64 characters
REQUEST_LOG = "probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"

image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([]).uv_pip_install("vllm==0.30.0", "pyyaml")  # run_vllm_tunnel.py's layers, so they're cached
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))
app = modal.App(APP)


@app.function(image=image, cpu=2, memory=16384, timeout=1800,
              volumes={"/logs": modal.Volume.from_name(REQUEST_LOG), "/hf": modal.Volume.from_name(HF_CACHE)})
def parse(log_name: str, calls: dict, setting: dict) -> str:
    import gzip

    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser import ParserManager
    from vllm.sampling_params import RepetitionDetectionParams
    from vllm.v1.core.sched.utils import check_sequence_repetition

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    parser_cls = ParserManager.get_parser(tool_parser_name="qwen3_coder", reasoning_parser_name="qwen3",
                                          enable_auto_tools=True, model_name=MODEL, is_harmony=False)
    params = RepetitionDetectionParams(**setting)
    with gzip.open(f"/logs/{log_name}", "rt") as f:
        records = {r.get("request_id"): r for r in map(json.loads, f) if r.get("request_id")}
    out = {"vllm": __import__("vllm").__version__, "log": log_name, "repetition_detection": setting, "calls": {}}
    for irid, call in calls.items():
        ids = records[irid]["choices"][0]["token_ids"]
        # The scheduler checks after each new token; the first prefix that trips the check is what vLLM returns.
        stop = next((n for n in range(1, len(ids) + 1) if check_sequence_repetition(ids[:n], params)), None)
        request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "-"}], tools=call["tools"],
                                        tool_choice=call.get("tool_choice") or "auto")
        row = {"tokens": len(ids), "stop_at": stop}
        for name, part in (("full", ids), ("cut", ids[:stop] if stop else None)):
            if part is None:
                row[name] = None
                continue
            text = tokenizer.decode(part, skip_special_tokens=True)
            parser = parser_cls(tokenizer, request.tools, chat_template_kwargs={}, model_config=None)
            reasoning, content, tool_calls = parser.parse(text, request, enable_auto_tools=True,
                                                          model_output_token_ids=part)
            row[name] = {"text_chars": len(text), "reasoning": reasoning, "content": content,
                         "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls or []]}
        nxt = call.get("next_request_id")
        if nxt and stop:  # does the next turn see this one's output? (it can't differ if it sees none of it)
            prompt = records[nxt]["prompt_token_ids"]
            grams = {tuple(prompt[i:i + 40]) for i in range(len(prompt) - 39)}
            found = [k for k in range(0, len(ids) - 39, 250) if tuple(ids[k:k + 40]) in grams]
            text = tokenizer.decode(prompt, skip_special_tokens=False)
            row["next_call"] = {"request_id": nxt, "prompt_tokens": len(prompt), "chunks_checked": len(range(0, len(ids) - 39, 250)),
                                "chunks_found_before_cut": [k for k in found if k + 40 <= stop],
                                "chunks_found_after_cut": [k for k in found if k + 40 > stop],
                                "empty_assistant_turns": text.count("<|im_start|>assistant\n<|im_end|>"),
                                "tail": text[-400:]}
        out["calls"][irid] = row
        print(irid, {k: v for k, v in row.items() if k in ("tokens", "stop_at")},
              {n: row[n] and row[n]["tool_calls"] for n in ("full", "cut")}, flush=True)
    return json.dumps(out)


def _sha(text: str | None) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def without_text(result: dict) -> dict:
    """The parse result with every piece of model text replaced by its length and sha256, for results/."""
    def parse(row):
        return None if row is None else {
            "text_chars": row["text_chars"], "reasoning_chars": len(row["reasoning"] or ""),
            "reasoning_sha256": _sha(row["reasoning"]), "content_chars": len(row["content"] or ""),
            "content_sha256": _sha(row["content"]),
            "tool_calls": [{"name": tc["name"], "arguments_sha256": _sha(tc["arguments"])} for tc in row["tool_calls"]]}

    out = {k: v for k, v in result.items() if k != "calls"}
    out["calls"] = {i: {k: parse(v) if k in ("full", "cut") else {kk: vv for kk, vv in v.items() if kk != "tail"}
                        if k == "next_call" else v for k, v in row.items()} for i, row in result["calls"].items()}
    return out


@app.local_entrypoint()
def main(calls: str, log: str = "requests_2026-10-07_211507.jsonl.gz", setting: str = "d"):
    import yaml  # read here, not at import: the container has no config folder

    settings = yaml.safe_load((HERE / "config" / "servers.yaml").read_text())["repetition_settings"]
    result = json.loads(parse.remote(log, json.loads(Path(calls).read_text()), settings[setting]))
    result["setting"] = setting
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    DATA.mkdir(parents=True, exist_ok=True)
    full = DATA / f"repetition_cut_parse_{stamp}.json"  # compare reads this one
    with open(full, "x") as f:
        json.dump(result, f, indent=1)
    path = HERE / "results" / f"repetition_cut_parse_{stamp}.json"
    with open(path, "x") as f:
        json.dump(without_text(result) | {"full_parse": str(full.relative_to(HERE.parents[1]))}, f, indent=1)
    print(f"-> {path} (full parse, with the model's text: {full}; archive it to S3)")
