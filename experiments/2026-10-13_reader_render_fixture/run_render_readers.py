"""What each reader sees at the tool-reply readout: synthetic fixtures through five readers' chat templates (POST HOC,
pennies of Modal CPU; no real text). The orchestrator for Aswin, 2026-10-13.

Fixtures: the G3 screen's ../2026-10-13_g3_new_reader/fixtures/tool_newline.json (a bash output ending in a newline) and
fixtures/editor_slips.json (synthetic text_editor outputs: str_replace in the real gutter form, the same in the
simulator's form with one trailing newline, and an insert ending in the review note). Per reader, at its pinned revision,
each fixture is rendered with that reader's own rendering path and readout rule, as its extraction used them:
  Qwen3-32B, Qwen3.5-27B: probe_shift.reader.segment_end_positions (output_end: the last token wholly inside the output);
  Olmo-3.1-32B-Instruct: run_extract_olmo.py's prep() and R2 (the last token wholly inside the content that is not
    whitespace-only and holds no newline);
  Llama-3.3-70B-Instruct: run_extract_llama.py's prep(), date_string and R2J on the JSON-escaped body (no token holding an
    escaped newline, at least one non-whitespace character);
  Seed-OSS-36B-Instruct: G3 Amendment 1's readout (R2 that is also not a special token) on the content located once in
    the rendering.
Prints, per reader and fixture: whether the output's trailing newline reaches the reader (G3's guard: the rendering with
the newline differs from the rendering with it removed; a template that trims content and writes its own newline gives
identical renderings), and
the tokens from 8 before the readout to 4 after it, with the readout marked. One Modal CPU container; the Hugging Face
token stays in the Modal secret's environment and is never read or printed. Tokenizer files only, no weights.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-13_reader_render_fixture/run_render_readers.py
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-render-readers_kill-2026-11-13"
READERS = {"Qwen3-32B": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", "qwen"),
           "Qwen3.5-27B": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", "qwen"),
           "Olmo-3.1-32B": ("allenai/Olmo-3.1-32B-Instruct", "ac0587e4a7744a551c059d8cd17ba220bc940dae", "r2"),
           "Llama-3.3-70B": ("meta-llama/Llama-3.3-70B-Instruct", "6f6073b423013f6a7d4d9f39144961bfbfbc386b", "r2j"),
           "Seed-OSS-36B": ("ByteDance-Seed/Seed-OSS-36B-Instruct", "497f1dca95ebdec98e41d517b9f060ee753c902f", "r2s")}
LLAMA_DATE = "26 Jul 2024"
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub==0.35.3", "sentencepiece==0.2.0", "protobuf==5.29.3", "tiktoken==0.12.0")
         .env({"HF_HOME": "/tmp/hf"}).add_local_python_source("probe_shift"))


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


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret-aswin")], cpu=2, memory=8192, timeout=900, retries=0)
def render_all(fixtures: dict) -> dict:
    import bisect

    from transformers import AutoTokenizer

    from probe_shift.reader import segment_end_positions
    res = {}
    for reader, (repo, rev, rule) in READERS.items():
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        special = set(tok.all_special_ids)
        for case, msgs in fixtures.items():
            content = next(m["content"] for m in msgs if m["role"] == "tool")
            r = {"rule": rule}
            if rule == "qwen":
                P = segment_end_positions(tok, msgs)
                ids, names = P["ids"], tok.convert_ids_to_tokens(P["ids"])
                t = next(p["token"] for p in P["positions"] if p["kind"] == "output_end")
                text = tok.apply_chat_template(msgs, tokenize=False)
            else:
                prepped = prep_llama(msgs) if rule == "r2j" else prep_olmo(msgs)
                text = tok.apply_chat_template(prepped, tokenize=False, **({"date_string": LLAMA_DATE} if rule == "r2j" else {}))
                body = json.dumps(content, ensure_ascii=False)[1:-1] if rule == "r2j" else content
                if text.count(body) != 1:
                    r["error"] = f"the rendered content occurs {text.count(body)} times"; res[f"{reader}|{case}"] = r; continue
                a = text.index(body); b = a + len(body)
                enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
                ids = enc["input_ids"]; st = [s for s, _ in enc["offset_mapping"]]; en = [e for _, e in enc["offset_mapping"]]
                names = tok.convert_ids_to_tokens(ids)
                i0, i1 = bisect.bisect_left(st, a), bisect.bisect_right(en, b)
                if rule == "r2j":
                    t = next((k for k in range(i1 - 1, i0 - 1, -1) if "\\n" not in text[st[k]:en[k]] and text[st[k]:en[k]].strip()), None)
                else:
                    t = next((k for k in range(i1 - 1, i0 - 1, -1) if not bad(text[st[k]:en[k]]) and en[k] > st[k]
                              and (rule != "r2s" or ids[k] not in special)), None)
            nl = content.endswith("\n")
            if nl:
                cut = [dict(m, content=m["content"].rstrip("\n")) if m["role"] == "tool" else m for m in msgs]
                kw = {"date_string": LLAMA_DATE} if rule == "r2j" else {}
                pre = (lambda x: x) if rule == "qwen" else (prep_llama if rule == "r2j" else prep_olmo)
                r["newline_reaches_reader"] = tok.apply_chat_template(pre(msgs), tokenize=False, **kw) != tok.apply_chat_template(pre(cut), tokenize=False, **kw)
            r |= {"output_ends_in_newline": nl, "readout": t, "window": [{"i": k, "token": names[k], "readout": k == t} for k in range(max(0, t - 8), min(len(ids), t + 5))] if t is not None else []}
            res[f"{reader}|{case}"] = r
    return res


@app.local_entrypoint()
def main():
    g3 = json.loads((HERE.parent / "2026-10-13_g3_new_reader/fixtures/tool_newline.json").read_text())
    fx = {"bash_trailing_newline": g3["messages"]} | json.loads((HERE / "fixtures/editor_slips.json").read_text())["cases"]
    res = render_all.remote(fx)
    out = HERE / "results" / f"render_readers_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
    out.open("x").write(json.dumps({"label": "POST HOC; synthetic fixtures only", "readers": READERS, "results": res}, indent=1))
    for key, r in res.items():
        print(f"== {key}: rule {r['rule']}; ends in newline {r.get('output_ends_in_newline')}; newline reaches reader "
              f"{r.get('newline_reaches_reader', 'n/a')}; readout {r.get('readout')} {r.get('error', '')}")
        print("   " + "  ".join(("[[" + repr(w["token"]) + "]]") if w["readout"] else repr(w["token"]) for w in r["window"]))
    print("written", out.name)
