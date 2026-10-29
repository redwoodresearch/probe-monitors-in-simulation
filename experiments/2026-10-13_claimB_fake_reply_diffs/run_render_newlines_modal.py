"""POST HOC (RULE.md Amendment 1 @ 266ecf07, items 2(i) and 2(iii); 2(ii)'s rendering side). Extends the reader render check
@ 8d46b355 (script run_render_readers.py @ c9aa0c61, whose rendering paths and readout rules are copied here unchanged).
Synthetic fixtures only, no real text. One Modal CPU container; tokenizer files only; the gated Llama tokenizer through
the Modal secret, never read or printed.
  (i)  shell and file-view outputs ending in 0, 1 and 2 newlines: per reader and step (0->1, 1->2), does the added newline
       reach the reader, is the readout unaffected (same index, identical ids through it), and the readout tokens.
  (ii)/(iii) one synthetic Inspect sample (a content marker, four clock stamps, an errored tool message) through
       reader_messages, the staging scrub's tool branch at levels 0 and 1, and each reader's prep and template: which
       markers appear in the final rendered text.
Writes results/render_newlines_<PT stamp>.json.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-13_claimB_fake_reply_diffs/run_render_newlines_modal.py
"""

import difflib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-render-newlines_kill-2026-11-13"
READERS = {"Qwen3-32B": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", "qwen"),
           "Qwen3.5-27B": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", "qwen"),
           "Olmo-3.1-32B": ("allenai/Olmo-3.1-32B-Instruct", "ac0587e4a7744a551c059d8cd17ba220bc940dae", "r2"),
           "Llama-3.3-70B": ("meta-llama/Llama-3.3-70B-Instruct", "6f6073b423013f6a7d4d9f39144961bfbfbc386b", "r2j"),
           "Seed-OSS-36B": ("ByteDance-Seed/Seed-OSS-36B-Instruct", "497f1dca95ebdec98e41d517b9f060ee753c902f", "r2s")}
LLAMA_DATE = "26 Jul 2024"
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub==0.35.3", "sentencepiece==0.2.0", "protobuf==5.29.3", "tiktoken==0.12.0",
                         "pyyaml==6.0.2", "boto3>=1.34")  # boto3 and pyyaml: probe_shift.transcripts imports them
         .env({"HF_HOME": "/tmp/hf"}).add_local_python_source("probe_shift"))

# --- fixtures (synthetic) ---
VIEW = [{"role": "user", "content": "Show /tmp/demo.py."},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "text_editor", "arguments": {"command": "view", "path": "/tmp/demo.py"}}}]},
        {"role": "tool", "content": "Here's the result of running `cat -n` on /tmp/demo.py:\n     1\tx = 1\n     2\ty = 2", "tool_name": "text_editor"},
        {"role": "assistant", "content": "Done.", "tool_calls": []}]
STAMPS = {"date_cmd": "Thu Sep 18 10:23:45 UTC 2026", "iso": "2026-09-18T10:24:01Z", "ls": "Sep 18 10:25", "http": "Thu, 18 Sep 2026 10:26:07 GMT"}
# each stamp's whole output line, so its masked form is unique in the rendering
LINES = {"date_cmd": STAMPS["date_cmd"], "iso": STAMPS["iso"], "ls": f"-rw-r--r-- 1 root root 13 {STAMPS['ls']} notes.txt",
         "http": f"Date: {STAMPS['http']}"}
LEFTOVER = ["Thu", "UTC", "2026", "18 Sep", "GMT"]
CONTENT_MARK, ERR_MARK, ERR_TYPE = "MARKERCONTENTQ7", "MARKERERRMSGQ7 command timed out", "timeout"
SAMPLE = {"id": "fixture", "epoch": 1, "messages": [
    {"role": "system", "content": "sample system prompt"},
    {"role": "user", "content": "Check the clock and the files."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": "bash", "arguments": {"cmd": "date; ls -l; curl -sI localhost"}}]},
    {"role": "tool", "tool_call_id": "c1", "function": "bash",
     "content": CONTENT_MARK + "\n" + "\n".join(LINES.values()) + "\n"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "function": "bash", "arguments": {"cmd": "sleep 999"}}]},
    {"role": "tool", "tool_call_id": "c2", "function": "bash", "content": "", "error": {"type": ERR_TYPE, "message": ERR_MARK}},
    {"role": "assistant", "content": "Done.", "tool_calls": []}]}


def with_newlines(msgs, n):
    return [dict(m, content=m["content"].rstrip("\n") + "\n" * n) if m["role"] == "tool" else m for m in msgs]


# --- copied unchanged from run_render_readers.py @ c9aa0c61 ---
def prep_olmo(msgs):  # run_extract_olmo.py prep(): string content, tool-call arguments as a dict
    out = []
    for m in msgs:
        m = dict(m); c = m.get("content")
        m["content"] = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**(tc.get("function") or {}), "arguments": (lambda a: json.loads(a) if isinstance(a, str) else (a or {}))(
                (tc.get("function") or {}).get("arguments"))}} for tc in m["tool_calls"]]
        out.append(m)
    return out


def prep_llama(msgs):  # run_extract_llama.py prep(): one assistant message per tool call
    out = []
    for m in prep_olmo(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            if m["content"].strip():
                out.append({"role": "assistant", "content": m["content"]})
            out += [{"role": "assistant", "content": "", "tool_calls": [tc]} for tc in m["tool_calls"]]
        else:
            m.pop("tool_calls", None)
            out.append(m)
    return out


def bad(t: str) -> bool:  # run_extract_olmo.py bad()
    return t.strip() == "" or "\n" in t


def readout(tok, rule, msgs, special):
    """(ids, token names, readout index) by the reader's own rendering path and rule: render_all's body @ c9aa0c61."""
    import bisect

    from probe_shift.reader import segment_end_positions
    content = next(m["content"] for m in msgs if m["role"] == "tool")
    if rule == "qwen":
        P = segment_end_positions(tok, msgs)
        ids = P["ids"]
        return ids, tok.convert_ids_to_tokens(ids), next(p["token"] for p in P["positions"] if p["kind"] == "output_end")
    prepped = prep_llama(msgs) if rule == "r2j" else prep_olmo(msgs)
    text = tok.apply_chat_template(prepped, tokenize=False, **({"date_string": LLAMA_DATE} if rule == "r2j" else {}))
    body = json.dumps(content, ensure_ascii=False)[1:-1] if rule == "r2j" else content
    assert text.count(body) == 1, f"the rendered content occurs {text.count(body)} times"
    a = text.index(body); b = a + len(body)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]; st = [s for s, _ in enc["offset_mapping"]]; en = [e for _, e in enc["offset_mapping"]]
    i0, i1 = bisect.bisect_left(st, a), bisect.bisect_right(en, b)
    if rule == "r2j":
        t = next((k for k in range(i1 - 1, i0 - 1, -1) if "\\n" not in text[st[k]:en[k]] and text[st[k]:en[k]].strip()), None)
    else:
        t = next((k for k in range(i1 - 1, i0 - 1, -1) if not bad(text[st[k]:en[k]]) and en[k] > st[k]
                  and (rule != "r2s" or ids[k] not in special)), None)
    return ids, tok.convert_ids_to_tokens(ids), t
# --- end of the copy ---


def differing(a: list, b: list) -> int:
    """Tokens of the longer side not in a matching block: insertions, deletions and substitutions after the readout."""
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return max(len(a), len(b)) - sum(m.size for m in sm.get_matching_blocks())


def rendered(tok, rule, msgs):
    """The final text each reader's extraction feeds the model (Qwen: reader.prepare, then the template as _layout)."""
    from probe_shift.reader import prepare
    if rule == "qwen":
        return tok.apply_chat_template(prepare(tok, msgs), tokenize=False, enable_thinking=False)
    pre = prep_llama(msgs) if rule == "r2j" else prep_olmo(msgs)
    return tok.apply_chat_template(pre, tokenize=False, **({"date_string": LLAMA_DATE} if rule == "r2j" else {}))


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret-aswin")], cpu=2, memory=8192, timeout=900, retries=0)
def render_all(shell: list[dict]) -> dict:
    from transformers import AutoTokenizer

    from probe_shift.scrub import scrub
    from probe_shift.transcripts import reader_messages
    staged = reader_messages(SAMPLE, "You are a helpful assistant.")
    res = {"newlines": {}, "markers": {}}
    for reader, (repo, rev, rule) in READERS.items():
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        special = set(tok.all_special_ids)
        for fam, base in (("shell", shell), ("view", VIEW)):
            v = {n: with_newlines(base, n) for n in (0, 1, 2)}
            r = {n: readout(tok, rule, v[n], special) for n in v}
            for a, b in ((0, 1), (1, 2)):
                (ia, na, ta), (ib, nb, tb) = r[a], r[b]
                res["newlines"][f"{reader}|{fam}|{a}->{b}"] = {
                    "reaches": rendered(tok, rule, v[a]) != rendered(tok, rule, v[b]),
                    "readout_unaffected": ta == tb and ta is not None and ia[:ta + 1] == ib[:tb + 1],
                    "readout_index": [ta, tb], "readout_token": [na[ta] if ta is not None else None, nb[tb] if tb is not None else None],
                    "tokens_after_readout": [len(ia) - ta - 1 if ta is not None else None, len(ib) - tb - 1 if tb is not None else None],
                    "differing_tokens_after_readout": differing(ia[ta + 1:], ib[tb + 1:]) if None not in (ta, tb) else None,
                    "next_tokens": [na[ta + 1:ta + 4] if ta is not None else [], nb[tb + 1:tb + 4] if tb is not None else []]}
        for level in (0, 1):
            msgs = [m | {"content": scrub(m["content"], level)} if m["role"] == "tool" else m for m in staged]  # scrubbed()'s tool branch
            text = rendered(tok, rule, msgs)
            res["markers"][f"{reader}|level{level}"] = {
                "content_marker": CONTENT_MARK in text, "error_message": ERR_MARK in text, "error_type": ERR_TYPE in text,
                "stamps_verbatim": {k: s in text for k, s in STAMPS.items()},
                "stamp_lines_masked_form": {k: scrub(line, 1) in text for k, line in LINES.items()},
                "placeholders": {p: text.count(p) for p in ("<time>", "<date>", "<epoch>")},
                "leftover_parts": {p: p in text for p in LEFTOVER}}
    return res


@app.local_entrypoint()
def main():
    me = Path(__file__).resolve()
    if subprocess.run(["git", "status", "--porcelain", str(me)], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit(f"commit {me.name} first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    shell = json.loads((HERE.parent / "2026-10-13_g3_new_reader/fixtures/tool_newline.json").read_text())["messages"]  # G3's synthetic fixture
    res = render_all.remote(shell)
    out = HERE / "results" / f"render_newlines_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
    out.open("x").write(json.dumps({"label": "POST HOC; synthetic fixtures only", "rule": "RULE.md Amendment 1 @ 266ecf07, item 2",
                                    "git": git, "readers": READERS, **res}, indent=1))
    for k, r in res["newlines"].items():
        print(f"{k:34s} reaches {r['reaches']!s:5s} readout unaffected {r['readout_unaffected']!s:5s} readout {r['readout_token']} next {r['next_tokens']}")
    for k, r in res["markers"].items():
        print(f"{k:24s} content {r['content_marker']} err msg {r['error_message']} err type {r['error_type']} stamps {r['stamps_verbatim']} "
              f"placeholders {r['placeholders']} leftover {r['leftover_parts']}")
    print("written", out.name)
