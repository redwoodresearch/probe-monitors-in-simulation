"""Render check for the two pilot models ($0, local files only; synthetic tool messages, never real transcripts).
Reported, never selected on (RULE.md).

For each model's pinned chat template (HF chat_template.jinja) and tokenizer: does a tool output's trailing newline
survive into the prompt, and does a failed call's error text (Inspect sends it as "Error: <message>") appear? Each
render is tokenized and decoded back; the decode must equal the render. For gpt-oss-120b, vLLM 0.30.0 does not use the
jinja template: it renders through openai-harmony (vllm/entrypoints/openai/parser/harmony_utils.py), so the same
check also runs through openai-harmony's encoder when --harmony is given.

  uv run --no-project --python 3.14 --with tokenizers --with jinja2 [--with openai-harmony] python render_check.py \\
      --gptoss-dir <dir: chat_template.jinja, tokenizer.json> --qcoder-dir <dir: same> [--harmony]
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import jinja2
from tokenizers import Tokenizer

HERE = Path(__file__).parent
NEWLINE_OUT = "line one\nline two\n"        # ends in a newline
ERROR_TEXT = "Error: The path /app/missing.txt does not exist."
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a command.",
                                            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]


def convo(tool_text: str) -> list[dict]:
    return [{"role": "system", "content": "TASK TEXT"}, {"role": "user", "content": ""},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": {"cmd": "ls"}}}]},
            {"role": "tool", "content": tool_text, "tool_call_id": "c1", "name": "bash"}]


def jinja_render(template: str, msgs: list[dict]) -> str:
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False, **k)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    env.globals["strftime_now"] = lambda f: "2026-10-13"
    return env.from_string(template).render(messages=msgs, tools=TOOLS, add_generation_prompt=True)


def check(render: str, tok: Tokenizer) -> dict:
    ids = tok.encode(render, add_special_tokens=False).ids
    decoded = tok.decode(ids, skip_special_tokens=False)
    return {"decode_equals_render": decoded == render, "tokens": len(ids)}


def one(name: str, d: Path) -> dict:
    tmpl, tok = (d / "chat_template.jinja").read_text(), Tokenizer.from_file(str(d / "tokenizer.json"))
    a, b = jinja_render(tmpl, convo(NEWLINE_OUT)), jinja_render(tmpl, convo(NEWLINE_OUT.rstrip("\n")))
    e = jinja_render(tmpl, convo(ERROR_TEXT))
    return {"renderer": "HF chat_template.jinja", "trailing_newline_kept": a != b and (NEWLINE_OUT in a),
            "with_and_without_newline_render_differently": a != b, "error_text_shown": ERROR_TEXT in e,
            "tool_output_json_escaped": json.dumps(NEWLINE_OUT)[1:-1] in a and NEWLINE_OUT not in a,
            "first_user_turn_empty_rendered": True, "positive_control": check(a, tok) | {"needle_found": NEWLINE_OUT in a or json.dumps(NEWLINE_OUT)[1:-1] in a}}


def harmony(name: str) -> dict:
    from openai_harmony import (Conversation, DeveloperContent, HarmonyEncodingName, Message, Role, SystemContent,
                                ToolDescription, load_harmony_encoding)
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def render(tool_text: str) -> str:
        dev = DeveloperContent.new().with_instructions("TASK TEXT").with_function_tools(
            [ToolDescription.new("bash", "Run a command.", parameters=TOOLS[0]["function"]["parameters"])])
        msgs = [Message.from_role_and_content(Role.SYSTEM, SystemContent.new()), Message.from_role_and_content(Role.DEVELOPER, dev),
                Message.from_role_and_content(Role.USER, ""),
                Message.from_role_and_content(Role.ASSISTANT, '{"cmd": "ls"}').with_channel("commentary").with_recipient("functions.bash"),
                Message.from_author_and_content(__import__("openai_harmony").Author.new(Role.TOOL, "functions.bash"), tool_text)
                .with_channel("commentary").with_recipient("assistant")]
        ids = enc.render_conversation_for_completion(Conversation.from_messages(msgs), Role.ASSISTANT)
        return enc.decode(ids)
    a, b, e = render(NEWLINE_OUT), render(NEWLINE_OUT.rstrip("\n")), render(ERROR_TEXT)
    return {"renderer": "openai-harmony (vLLM 0.30.0's gpt-oss path)", "trailing_newline_kept": a != b and NEWLINE_OUT in a,
            "error_text_shown": ERROR_TEXT in e}


def main(gptoss_dir: str, qcoder_dir: str, harmony_on: bool):
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out = {"git": git, "synthetic_only": True, "selected_on": False,
           "gpt-oss-120b@b5c939de": {"jinja": one("gptoss", Path(gptoss_dir))},
           "Qwen3-Coder-Next@a7fbcb5c": {"jinja": one("qcoder", Path(qcoder_dir))}}
    if harmony_on:
        out["gpt-oss-120b@b5c939de"]["harmony"] = harmony("gptoss")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"render_check_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), "\n->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gptoss-dir", required=True)
    ap.add_argument("--qcoder-dir", required=True)
    ap.add_argument("--harmony", dest="harmony_on", action="store_true")
    main(**vars(ap.parse_args()))
