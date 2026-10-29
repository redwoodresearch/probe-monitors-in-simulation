"""G3 step 1 (RULE.md): screen the nine fixed candidates and pick the reader. $0: Hugging Face metadata, config, index and
tokenizer files only; no weights. Hugging Face access is anonymous (no login exists on the laptop; token=False, NETRC
disabled), so a gated repository fails (a). It runs once, from a clean commit; there is no rerun after the pick.

Per candidate, at the revision model_info returns (recorded; later steps use it). Every check runs for every candidate
and records its reason on failure; any error fails its own check only. A candidate passes if (a)-(d) all pass.
  (a) available: model_info gives a revision sha and a 30-day download count; the root safetensors index and
      config.json download at that revision.
  (b) size: index metadata.total_size / bytes per parameter of config.json's dtype, in [20e9, 40e9] inclusive. The dtype
      is the top-level torch_dtype or dtype; a multimodal config that keeps it only in text_config uses that one
      (recorded; this assumes one storage dtype for the checkpoint).
  (c) template: the fixture renders through prep() + tok.apply_chat_template(..., tokenize=False), the rendering path of
      run_extract_olmo.py@726c953e (prep() is copied below and byte-checked against that blob; the Olmo extraction's
      launch record holds the same blob sha256), and the tool output's exact text appears in the rendered string.
  (d) newline: the content span is the tool output's text located in the rendered string (it must occur once). Tokenized
      with offset mapping (no special tokens added):
        literal (RULE.md): every token covering the span's final "\\n" lies inside the span (none crosses its end), at
          least one does, and output_end (the last token wholly inside the span) is not a special token;
        guard: the final "\\n" is the tool output's own: the fixture rendered without that newline equals the full
          rendering minus exactly that character at the span end. A template that trims content and writes its own
          "\\n" right after it passes the literal test with a template newline (Qwen3.5-27B does; --selftest).
      Both are recorded separately; (d) passes only if both pass.
Pick: the most 30-day downloads among passing candidates, ties to list order. If the literal-only pick (guard ignored)
differs from the pick, the record says so and holds pick = null: the guard is beyond RULE.md's text, so 09 resolves
that case by amendment from this record, with no rerun.

  uv run --no-project --python 3.12 --with transformers==4.57.1 --with jinja2==3.1.6 --with huggingface_hub==0.35.3 \\
      --with sentencepiece==0.2.0 --with protobuf==5.29.3 python experiments/2026-10-13_g3_new_reader/screen_readers.py
--selftest screens two non-candidate controls and prints only (no record): Qwen3-32B (expect a-c pass; literal (d)
fails, its content "\\n" merging with the template's "\\n" into one "\\n\\n" token across the span end) and Qwen3.5-27B
(trims tool output: literal (d) passes on the template's newline, the guard fails).
"""
import ast
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANDIDATES = ["google/gemma-3-27b-it", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "openai/gpt-oss-20b",
              "zai-org/GLM-4-32B-0414", "ByteDance-Seed/Seed-OSS-36B-Instruct", "LGAI-EXAONE/EXAONE-4.0-32B",
              "01-ai/Yi-1.5-34B-Chat", "CohereLabs/c4ai-command-r-08-2024", "internlm/internlm2_5-20b-chat"]
RENDER_SRC = ("726c953e", "experiments/2026-10-12_q2_olmo_reader/run_extract_olmo.py",
              "40fd64b31aa4778b38659a7d07fa7d191bb2ec06063412b97b3671b835049c24")
BYTES = {"float64": 8, "float32": 4, "bfloat16": 2, "float16": 2, "float8_e4m3fn": 1, "float8_e5m2": 1, "int8": 1}
LO, HI = 20e9, 40e9


def prep(msgs):
    """Olmo's template takes string content and tool-call arguments as a dict; the staged messages may hold either form."""
    out = []
    for m in msgs:
        m = dict(m); c = m.get("content")
        m["content"] = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**(tc.get("function") or {}), "arguments": (lambda a: json.loads(a) if isinstance(a, str) else (a or {}))(
                (tc.get("function") or {}).get("arguments"))}} for tc in m["tool_calls"]]
        out.append(m)
    return out


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def preflight():
    """prep() must equal the as-run extractor's byte for byte; this script and its fixture must be committed, unmodified."""
    commit, path, sha = RENDER_SRC
    blob = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    if hashlib.sha256(blob).hexdigest() != sha:
        raise SystemExit(f"{path}@{commit} sha256 mismatch")
    src = blob.decode()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "prep")
    if ast.get_source_segment(src, fn) != inspect.getsource(prep).rstrip("\n"):
        raise SystemExit("prep() differs from the pinned extractor's")
    mine = [str(p.relative_to(ROOT)) for p in (Path(__file__).resolve(), HERE / "fixtures" / "tool_newline.json")]
    if git("status", "--porcelain", "--", *mine).strip() or any(not git("ls-files", "--", p).strip() for p in mine):
        raise SystemExit("commit screen_readers.py and fixtures/tool_newline.json before the screen runs")


def err(e: Exception) -> str:
    s = str(e).strip()
    return f"{type(e).__name__}: {s.splitlines()[0][:200] if s else ''}"


def valid_downloads(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def check_a(api, repo):
    from huggingface_hub import hf_hub_download
    r, info, idx, cfg = {"pass": False}, None, None, None
    try:
        info = api.model_info(repo)
        if not (isinstance(info.sha, str) and re.fullmatch(r"[0-9a-f]{40}", info.sha)) or not valid_downloads(info.downloads):
            raise ValueError(f"missing or invalid metadata: sha {info.sha!r}, downloads {info.downloads!r}")
        names = [s.rfilename for s in info.siblings or []]
        idx_names = ["model.safetensors.index.json"] if "model.safetensors.index.json" in names else \
            [n for n in names if "/" not in n and n.endswith(".safetensors.index.json")]
        if len(idx_names) != 1:
            raise FileNotFoundError(f"root safetensors index files: {idx_names}")
        idx = json.loads(Path(hf_hub_download(repo, idx_names[0], revision=info.sha, token=False)).read_text())
        cfg = json.loads(Path(hf_hub_download(repo, "config.json", revision=info.sha, token=False)).read_text())
        r = {"pass": True, "index": idx_names[0]}
    except Exception as e:
        r["reason"] = err(e)
    return r, info, idx, cfg


def check_b(idx, cfg):
    try:
        size = ((idx or {}).get("metadata") or {}).get("total_size")
        tc = (cfg or {}).get("text_config") if isinstance((cfg or {}).get("text_config"), dict) else {}
        src, dtype = next(((k, v) for k, v in [("torch_dtype", (cfg or {}).get("torch_dtype")), ("dtype", (cfg or {}).get("dtype")),
                           ("text_config.torch_dtype", tc.get("torch_dtype")), ("text_config.dtype", tc.get("dtype"))] if v), (None, None))
        if not isinstance(size, (int, float)) or dtype not in BYTES:
            return {"pass": False, "reason": f"total_size {size!r}, dtype {dtype!r}"}
        n = size / BYTES[dtype]
        r = {"pass": LO <= n <= HI, "total_size": size, "dtype": dtype, "dtype_source": src, "params": n}
        if not r["pass"]:
            r["reason"] = f"{n / 1e9:.2f}B parameters, outside [20B, 40B]"
        return r
    except Exception as e:
        return {"pass": False, "reason": err(e)}


def check_cd(repo, rev, fixture):
    from transformers import AutoTokenizer
    content = next(m["content"] for m in fixture["messages"] if m["role"] == "tool")
    c, d = {"pass": False}, {"pass": False}
    try:
        if not rev:
            raise ValueError("no revision")
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, trust_remote_code=False, token=False)
        text = tok.apply_chat_template(prep(fixture["messages"]), tokenize=False)
        c = {"pass": isinstance(text, str) and content in text, "occurrences": text.count(content), "rendered": text}
        if not c["pass"]:
            c["reason"] = "the tool output's text is not in the rendered string"
    except Exception as e:
        c["reason"] = err(e)
    if not c["pass"]:
        d["reason"] = "no content span ((c) failed)"
        return c, d
    try:
        if c["occurrences"] != 1:
            raise ValueError(f"content span not unique ({c['occurrences']} occurrences)")
        a = text.index(content); b = a + len(content)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
        if len(ids) != len(offs) or any(not (0 <= s <= e <= len(text)) for s, e in offs) \
                or any(offs[i][0] < offs[i - 1][0] for i in range(1, len(offs))):
            raise ValueError("offset mapping malformed (lengths, bounds or start order)")
        cover = [i for i, (s, e) in enumerate(offs) if s <= b - 1 < e]
        inside = [i for i, (s, e) in enumerate(offs) if e > s and s >= a and e <= b]
        oe = inside[-1] if inside else None
        special = set(tok.all_special_ids) | {i for i, t in tok.added_tokens_decoder.items() if t.special}
        crossing = [i for i, (s, e) in enumerate(offs) if s < a < e or s < b < e]  # RULE.md: a token crossing the span boundary fails (d)
        nl_ok = bool(cover) and all(a <= offs[i][0] and offs[i][1] <= b for i in cover)
        oe_ok = oe is not None and ids[oe] not in special
        spell = tok.convert_ids_to_tokens(ids)
        piece = lambda i: {"id": ids[i], "token": spell[i], "offsets": list(offs[i]), "text": text[offs[i][0]:offs[i][1]]}  # noqa: E731
        d = {"pass_literal": nl_ok and oe_ok and not crossing, "newline_tokens_inside": nl_ok, "output_end_is_content": oe_ok,
             "crossing_tokens": [piece(i) for i in crossing], "span": [a, b], "newline_tokens": [piece(i) for i in cover],
             "output_end": None if oe is None else piece(oe), "next_token": piece(oe + 1) if oe is not None and oe + 1 < len(ids) else None}
        reasons = [w for w, bad in [("a token covering the final newline crosses the span end, or none covers it", not nl_ok),
                                    ("a token crosses a span boundary", bool(crossing)),
                                    ("output_end is a special token or missing", not oe_ok)] if bad]
    except Exception as e:
        d, reasons = {"pass_literal": False}, [f"literal: {err(e)}"]
    try:  # the guard, separately: its own error never touches the literal result
        trimmed = [{**m, "content": m["content"][:-1]} if m["role"] == "tool" else m for m in fixture["messages"]]
        text_trim = tok.apply_chat_template(prep(trimmed), tokenize=False)
        b = text.index(content) + len(content)
        own = text == text_trim[:b - 1] + "\n" + text_trim[b - 1:]
        d |= {"guard_newline_is_the_outputs": own, "rendered_without_newline": text_trim}
        if not own:
            reasons.append("guard: the final newline is not the tool output's (the template trims or rewrites it)")
    except Exception as e:
        own = False
        d["guard_newline_is_the_outputs"] = False
        reasons.append(f"guard: {err(e)}")
    d["pass"] = bool(d["pass_literal"]) and own
    if reasons:
        d["reason"] = "; ".join(reasons)
    return c, d


def screen(api, repo, fixture):
    a, info, idx, cfg = check_a(api, repo)
    raw = info.sha if info is not None else None
    rev = raw if isinstance(raw, str) and re.fullmatch(r"[0-9a-f]{40}", raw) else None  # (c)/(d) run at any valid sha
    c, d = check_cd(repo, rev, fixture)
    dl = info.downloads if info is not None else None
    r = {"repo": repo, "revision": rev, "revision_raw": None if rev else repr(raw),
         "downloads_30d": dl if valid_downloads(dl) else None, "downloads_raw": None if valid_downloads(dl) else repr(dl),
         "a": a, "b": check_b(idx, cfg), "c": c, "d": d}
    r["passes"] = all(r[k]["pass"] for k in "abcd")
    r["passes_literal"] = all(r[k]["pass"] for k in "abc") and bool(d.get("pass_literal"))
    return r


def pick_of(rows, key):
    ok = [(i, r) for i, r in enumerate(rows) if r[key]]
    return max(ok, key=lambda t: (t[1]["downloads_30d"], -t[0]))[1]["repo"] if ok else None


def main(selftest: bool):
    from huggingface_hub import HfApi
    os.environ["NETRC"] = "/dev/null"  # requests' implicit .netrc auth off: nothing is sent but anonymous requests
    fx_path = HERE / "fixtures" / "tool_newline.json"
    fixture = json.loads(fx_path.read_text())
    api = HfApi(token=False)
    if selftest:
        for repo in ("Qwen/Qwen3-32B", "Qwen/Qwen3.5-27B"):
            r = screen(api, repo, fixture)
            print(repo, {k: r[k]["pass"] for k in "abcd"}, "literal (d):", r["d"].get("pass_literal"),
                  {k: r[k].get("reason") for k in "abcd" if not r[k]["pass"]},
                  {k: r["d"].get(k) for k in ("output_end", "next_token")})
        return
    preflight()
    rows = []
    for repo in CANDIDATES:
        rows.append(screen(api, repo, fixture))
        print(f"screened {repo}", flush=True)
    pick, literal = pick_of(rows, "passes"), pick_of(rows, "passes_literal")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"what": "G3 step 1: the reader screen (RULE.md)", "created_pt": stamp, "git": git("rev-parse", "HEAD").strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "render_function": {"file": RENDER_SRC[1], "commit": RENDER_SRC[0], "blob_sha256": RENDER_SRC[2],
                               "function": "prep() + tok.apply_chat_template(..., tokenize=False)"},
           "fixture": {"path": str(fx_path.relative_to(ROOT)), "sha256": hashlib.sha256(fx_path.read_bytes()).hexdigest(),
                       "note": "RULE.md's four turns, no system message (reader_messages always prepends one in staging)"},
           "versions": {p: version(p) for p in ("transformers", "tokenizers", "jinja2", "huggingface_hub", "sentencepiece", "protobuf")},
           "hf_access": "anonymous: HfApi(token=False), token=False on every download, NETRC=/dev/null; no login exists on the laptop",
           "trust_remote_code": False, "size_bounds": [LO, HI], "candidates": rows,
           "pick_literal_rule_only": literal, "pick": pick if pick == literal else None,
           "outcome": ("picked" if pick and pick == literal else "no candidate passes: G3 stops" if not pick and not literal
                       else "the guard changes the pick: pick held for 09's amendment (no rerun)")}
    out = HERE / "results" / f"screen_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    with out.open("x") as f:
        json.dump(rec, f, indent=1)
    print(f"{'candidate':48} {'a':>2} {'b':>2} {'c':>2} {'d':>2} {'d lit':>5} {'params':>8} {'downloads':>10}  first failing reason")
    for r in rows:
        why = next((f"{k}: {r[k]['reason']}" for k in "abcd" if not r[k]["pass"]), "")
        n = r["b"].get("params")
        print(f"{r['repo']:48} " + " ".join(f"{'Y' if r[k]['pass'] else 'n':>2}" for k in "abcd")
              + f" {'Y' if r['d'].get('pass_literal') else 'n':>5} {'' if n is None else f'{n / 1e9:.2f}B':>8}"
              + f" {r['downloads_30d'] if r['downloads_30d'] is not None else '-':>10}  {why[:110]}")
    print(f"pick: {rec['pick']} ; literal-rule pick: {literal} ; outcome: {rec['outcome']} ; record {out.relative_to(ROOT)}")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--selftest"]):
        raise SystemExit("usage: screen_readers.py [--selftest]")
    main(selftest=sys.argv[1:] == ["--selftest"])
