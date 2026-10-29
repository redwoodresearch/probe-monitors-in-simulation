"""Item 1 of RULE.md (POST HOC, $0, local): one synthetic sample through the reading pipeline and the acting pipeline,
token ids diffed region by region. Synthetic text only; Qwen3.5-27B tokenizer files at the pinned revision, no weights.

v2 (Amendment 2): also saves the full id lists and each differing region's positions and ids.
(i) reading: probe_shift.transcripts.reader_messages (the honest system prompt given explicitly) -> probe_shift.reader.render.
(ii) acting: the OpenAI-format request vLLM 0.30 renders with the model's chat template: tools= given, tool-call
     arguments as dicts (vLLM parses the JSON string before rendering), thinking on (the server's default), the
     generation prompt added; two versions, with and without the previous assistant turns' reasoning.

  PYTHONPATH=src uv run --no-project --python 3.12 --with transformers --with jinja2 --with huggingface_hub \\
      python experiments/2026-10-13_cell_a_transfer_POSTHOC/run_render_parity.py
"""

import difflib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from transformers import AutoTokenizer

from probe_shift import reader, transcripts

HERE = Path(__file__).resolve().parent
MODEL, REV = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
SYSTEM = "You are a careful software engineer working in a Linux container. Use the tools to finish the task."
TASK = "Add a health check endpoint to the service in /app and make sure the tests pass."
CALL1 = {"name": "bash", "arguments": {"cmd": "ls -la /app"}}
OUT1 = "total 8\ndrwxr-xr-x 2 root root 4096 Jan  1 00:00 .\n-rw-r--r-- 1 root root  120 Jan  1 00:00 server.py\n"
CALL2 = {"name": "text_editor", "arguments": {"command": "view", "path": "/app/server.py"}}
OUT2 = "Here's the result of running `cat -n` on /app/server.py:\n     1\tfrom flask import Flask\n     2\tapp = Flask(__name__)\n"
THOUGHT = ["I should look at the files first.", "Now read the server file."]
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a bash command.",
                                           "parameters": {"type": "object", "properties": {"cmd": {"type": "string", "description": "The command."}}, "required": ["cmd"]}}},
         {"type": "function", "function": {"name": "text_editor", "description": "View or edit files.",
                                           "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "path": {"type": "string"}},
                                                          "required": ["command", "path"]}}}]


def inspect_sample():  # the Inspect sample shape transcripts.reader_messages reads
    return {"id": "synthetic", "epoch": 1, "messages": [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": CALL1["name"], "arguments": CALL1["arguments"]}]},
        {"role": "tool", "tool_call_id": "c1", "function": "bash", "content": OUT1},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "function": CALL2["name"], "arguments": CALL2["arguments"]}]},
        {"role": "tool", "tool_call_id": "c2", "function": "text_editor", "content": OUT2}]}


def acting_messages(with_reasoning: bool):
    a = lambda i, c: {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": c}]} | (  # noqa: E731
        {"reasoning_content": THOUGHT[i - 1]} if with_reasoning else {})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}, a(1, CALL1),
            {"role": "tool", "tool_call_id": "c1", "content": OUT1}, a(2, CALL2), {"role": "tool", "tool_call_id": "c2", "content": OUT2}]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    msgs = transcripts.reader_messages(inspect_sample(), SYSTEM)
    ids_r, segs = reader.render(tok, msgs)
    ends_r = [s["end"] - 1 for s in segs if s["kind"] == "tool_output"]  # output_end: last token wholly inside the output
    out = {"model": MODEL, "revision": REV, "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
           "reader": {"n_tokens": len(ids_r), "ids": ids_r, "output_end_positions": ends_r, "output_end_tokens": [tok.decode([ids_r[i]]) for i in ends_r]}, "acting": {}}
    dec = lambda ids: tok.decode(ids)  # noqa: E731
    for name, rsn in (("acting_with_reasoning", True), ("acting_no_reasoning", False)):
        text = tok.apply_chat_template(acting_messages(rsn), tools=TOOLS, tokenize=False, add_generation_prompt=True)
        ids_a = tok(text, add_special_tokens=False)["input_ids"]
        ops = [(tag, i1, i2, j1, j2) for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ids_r, b=ids_a, autojunk=False).get_opcodes() if tag != "equal"]
        # the tool outputs: content tokens and the token right after each in both renderings
        tails = []
        for k, content in enumerate((OUT1, OUT2)):
            c_ids = tok(content.strip(), add_special_tokens=False)["input_ids"]
            find = lambda ids: next(i for i in range(len(ids)) if ids[i:i + len(c_ids)] == c_ids)  # noqa: E731
            pr, pa = find(ids_r), find(ids_a)
            tails.append({"output": k, "content_tokens_identical": True, "n_content_tokens": len(c_ids),
                          "reader_last_content_token": dec([ids_r[pr + len(c_ids) - 1]]), "acting_last_content_token": dec([ids_a[pa + len(c_ids) - 1]]),
                          "reader_readout_is_last_content_token": ends_r[k] == pr + len(c_ids) - 1,
                          "reader_next_4": dec(ids_r[pr + len(c_ids):pr + len(c_ids) + 4]), "acting_next_4": dec(ids_a[pa + len(c_ids):pa + len(c_ids) + 4]),
                          "reader_prev_4": dec(ids_r[pr - 4:pr]), "acting_prev_4": dec(ids_a[pa - 4:pa])})
        out["acting"][name] = {"n_tokens": len(ids_a), "ids": ids_a, "diff_regions": [{"op": tag, "reader_tokens": i2 - i1, "acting_tokens": j2 - j1,
                                                                                       "reader_pos": [i1, i2], "acting_pos": [j1, j2], "reader_ids": ids_r[i1:i2], "acting_ids": ids_a[j1:j2],
                                                                                       "reader_text": dec(ids_r[i1:i2]), "acting_text": dec(ids_a[j1:j2])} for tag, i1, i2, j1, j2 in ops],
                               "tool_outputs": tails}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"render_parity_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md item 1); synthetic text only"} | out, indent=1))
    print(f"reader {len(ids_r)} tokens; output_end tokens {out['reader']['output_end_tokens']}")
    for name, a in out["acting"].items():
        print(f"== {name}: {a['n_tokens']} tokens, {len(a['diff_regions'])} differing regions")
        for d in a["diff_regions"]:
            print(f"  {d['op']:8s} reader [{d['reader_pos'][0]}, {d['reader_pos'][1]}) {d['reader_tokens']:4d} tok {d['reader_text'][:90]!r} ids {d['reader_ids'][:12]} | "
                  f"acting [{d['acting_pos'][0]}, {d['acting_pos'][1]}) {d['acting_tokens']:4d} tok {d['acting_text'][:120]!r} ids {d['acting_ids'][:12]}{'...' if d['acting_tokens'] > 12 else ''}")
        for t in a["tool_outputs"]:
            print(f"  output {t['output']}: last content token reader {t['reader_last_content_token']!r} acting {t['acting_last_content_token']!r}; "
                  f"readout = last content token: {t['reader_readout_is_last_content_token']}; next 4: reader {t['reader_next_4']!r} acting {t['acting_next_4']!r}; "
                  f"prev 4: reader {t['reader_prev_4']!r} acting {t['acting_prev_4']!r}")
    print("written", dest.name)


if __name__ == "__main__":
    main()
