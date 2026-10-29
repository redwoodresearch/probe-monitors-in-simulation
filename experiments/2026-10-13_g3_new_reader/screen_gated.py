"""G3 RULE.md Amendment 1, point 3 (dbb1cd03): re-screen the two gated candidates (google/gemma-3-27b-it,
CohereLabs/c4ai-command-r-08-2024) with the project's Hugging Face credentials, then combine them with the seven unchanged
records of results/screen_2026-10-13_012449.json (c2785c4d) and apply the unchanged pick rule. An adapter around
screen_readers.py@f934d81f (imported unchanged; its fixture unchanged), with only these changes:
  - authentication, for these two only: the Modal secret `huggingface-secret-aswin` (HF_TOKEN, as
    ../2026-10-06_apollo_llama_check/run_apollo_llama.py on main) inside one Modal CPU container. The token stays in that
    container's environment: it is never read, printed or returned, and nothing on the laptop holds it;
  - the frozen inputs: each candidate's revision and 30-day download count are the original record's, not re-read; the
    checks run at that revision (the current hub sha and downloads are recorded beside, as diagnostics only);
  - the readout in (d) (Amendment 1, point 1): output_end is the last non-special token whose offsets have positive
    width, lie wholly inside the content span, and pass Olmo's pinned bad() test (text.strip() != "" and no "\\n"). The
    newline test (every token covering the span's final "\\n" inside the span, none crossing either boundary) and the
    guard (the final "\\n" is the tool output's own, Amendment 1 point 2) are unchanged.
Pick: the most recorded downloads among candidates passing (a)-(d), ties to list order (screen_readers.pick_of). Runs once.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with transformers==4.57.1 --with jinja2==3.1.6 \\
      --with huggingface_hub==0.35.3 modal run experiments/2026-10-13_g3_new_reader/screen_gated.py
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q2-g3_screen_kill-2026-11-12"
GATED = ("google/gemma-3-27b-it", "CohereLabs/c4ai-command-r-08-2024")
ORIGINAL = ("c2785c4d", "experiments/2026-10-13_g3_new_reader/results/screen_2026-10-13_012449.json",
            "662bedd89be33d8f6a4d777e53f036c2a36054f53887dc862d3a121fef9c0944")
SCREEN = ("f934d81f", "4058010f1a7df0d952c0a33c403b88a4643f94bb982867419180a4316c7508a4")  # screen_readers.py: commit, sha256
PIP = ["transformers==4.57.1", "tokenizers==0.22.2", "jinja2==3.1.6", "huggingface_hub==0.35.3", "sentencepiece==0.2.0", "protobuf==5.29.3"]
SRC = "/opt/g3/screen"
FIXTURE_SHA = "14edd6d0f72b78f464b27118ad44e4ff97394e4c0f3d2f017ca804591c675437"  # fixtures/tool_newline.json as committed at f934d81f and recorded in the original screen

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/tmp/hf"})
         .add_local_file(HERE / "screen_readers.py", f"{SRC}/screen_readers.py"))


def reason(e: Exception) -> str:
    """A returned reason holds the exception's type and HTTP status only, never its text (which a hub error could fill)."""
    st = getattr(getattr(e, "response", None), "status_code", None)
    return f"{type(e).__name__}" + (f" (HTTP {st})" if st else "")


def check_cd_amended(S, repo, rev, fixture):
    """screen_readers.check_cd with Amendment 1's readout and authenticated downloads (token=None reads HF_TOKEN from the
    container's environment). Everything else is check_cd's, line for line."""
    from transformers import AutoTokenizer
    content = next(m["content"] for m in fixture["messages"] if m["role"] == "tool")
    c, d = {"pass": False}, {"pass": False}
    try:
        if not rev:
            raise ValueError("no revision")
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, trust_remote_code=False, token=None)
        text = tok.apply_chat_template(S.prep(fixture["messages"]), tokenize=False)
        c = {"pass": isinstance(text, str) and content in text, "occurrences": text.count(content), "rendered": text}
        if not c["pass"]:
            c["reason"] = "the tool output's text is not in the rendered string"
    except Exception as e:
        c["reason"] = reason(e)
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
        special = set(tok.all_special_ids) | {i for i, t in tok.added_tokens_decoder.items() if t.special}
        inside = [i for i, (s, e) in enumerate(offs) if e > s and s >= a and e <= b]
        tt = lambda i: text[offs[i][0]:offs[i][1]]  # noqa: E731
        eligible = [i for i in inside if ids[i] not in special and tt(i).strip() != "" and "\n" not in tt(i)]  # Amendment 1's readout
        oe = eligible[-1] if eligible else None
        crossing = [i for i, (s, e) in enumerate(offs) if s < a < e or s < b < e]
        nl_ok = bool(cover) and all(a <= offs[i][0] and offs[i][1] <= b for i in cover)
        oe_ok = oe is not None
        spell = tok.convert_ids_to_tokens(ids)
        piece = lambda i: {"id": ids[i], "token": spell[i], "offsets": list(offs[i]), "text": tt(i)}  # noqa: E731
        d = {"pass_literal": nl_ok and oe_ok and not crossing, "newline_tokens_inside": nl_ok, "output_end_is_content": oe_ok,
             "crossing_tokens": [piece(i) for i in crossing], "span": [a, b], "newline_tokens": [piece(i) for i in cover],
             "output_end": None if oe is None else piece(oe), "next_token": piece(oe + 1) if oe is not None and oe + 1 < len(ids) else None,
             "readout": "Amendment 1: last non-whitespace content token"}
        reasons = [w for w, bad in [("a token covering the final newline crosses the span end, or none covers it", not nl_ok),
                                    ("a token crosses a span boundary", bool(crossing)),
                                    ("no eligible (non-whitespace, non-special) content token", not oe_ok)] if bad]
    except Exception as e:
        d, reasons = {"pass_literal": False}, [f"literal: {reason(e)}"]
    try:
        trimmed = [{**m, "content": m["content"][:-1]} if m["role"] == "tool" else m for m in fixture["messages"]]
        text_trim = tok.apply_chat_template(S.prep(trimmed), tokenize=False)
        b = text.index(content) + len(content)
        own = text == text_trim[:b - 1] + "\n" + text_trim[b - 1:]
        d |= {"guard_newline_is_the_outputs": own, "rendered_without_newline": text_trim}
        if not own:
            reasons.append("guard: the final newline is not the tool output's (the template trims or rewrites it)")
    except Exception as e:
        own = False
        d["guard_newline_is_the_outputs"] = False
        reasons.append(f"guard: {reason(e)}")
    d["pass"] = bool(d["pass_literal"]) and own
    if reasons:
        d["reason"] = "; ".join(reasons)
    return c, d


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret-aswin")], cpu=2, memory=8192, timeout=900, retries=0)
def rescreen(frozen: dict, fixture: dict) -> list[dict]:
    """The two gated candidates at their frozen revisions. Returns records only: no environment value is read or returned."""
    import os
    import sys
    sys.path.insert(0, SRC)
    import screen_readers as S
    from huggingface_hub import HfApi, hf_hub_download
    os.environ["NETRC"] = "/dev/null"
    if not os.environ.get("HF_TOKEN"):  # presence only; the value is never read into a variable, printed or returned
        raise RuntimeError("the Modal secret did not provide HF_TOKEN: no anonymous correction")
    api = HfApi()  # token from the container's HF_TOKEN
    rows = []
    for repo in GATED:
        rev, dl = frozen[repo]["revision"], frozen[repo]["downloads_30d"]
        a, idx, cfg, now = {"pass": False}, None, None, {}
        try:
            info = api.model_info(repo, revision=rev)
            now = {"sha at frozen revision": info.sha}
            names = [s.rfilename for s in info.siblings or []]
            idx_names = ["model.safetensors.index.json"] if "model.safetensors.index.json" in names else \
                [n for n in names if "/" not in n and n.endswith(".safetensors.index.json")]
            if len(idx_names) != 1:
                raise FileNotFoundError(f"root safetensors index files: {idx_names}")
            idx = json.loads(Path(hf_hub_download(repo, idx_names[0], revision=rev)).read_text())
            cfg = json.loads(Path(hf_hub_download(repo, "config.json", revision=rev)).read_text())
            a = {"pass": info.sha == rev, "index": idx_names[0]}
            if not a["pass"]:
                a["reason"] = "the hub returned a different sha for the frozen revision"
        except Exception as e:
            a["reason"] = reason(e)
        try:
            cur = api.model_info(repo)
            now |= {"current hub sha": cur.sha, "current downloads": cur.downloads}
        except Exception as e:
            now |= {"current lookup": reason(e)}
        c, d = check_cd_amended(S, repo, rev, fixture)
        r = {"repo": repo, "revision": rev, "downloads_30d": dl, "frozen from": "screen_2026-10-13_012449.json",
             "diagnostics (not used)": now, "a": a, "b": S.check_b(idx, cfg), "c": c, "d": d}
        r["passes"] = all(r[k]["pass"] for k in "abcd")
        r["passes_literal"] = all(r[k]["pass"] for k in "abc") and bool(d.get("pass_literal"))
        rows.append(r)
    return rows


@app.local_entrypoint()
def main():
    import sys
    git = lambda *x: subprocess.run(["git", *x], cwd=HERE, capture_output=True, text=True, check=True).stdout  # noqa: E731
    mine = [Path(__file__).resolve(), HERE / "screen_readers.py", HERE / "fixtures" / "tool_newline.json", HERE / "RULE.md"]
    if git("status", "--porcelain", "--", *map(str, mine)).strip():
        raise SystemExit("commit the adapter (and leave the screen, fixture and RULE unchanged) before the re-screen runs")
    if hashlib.sha256((HERE / "screen_readers.py").read_bytes()).hexdigest() != SCREEN[1]:
        raise SystemExit("screen_readers.py differs from f934d81f's")
    if subprocess.run(["git", "merge-base", "--is-ancestor", "dbb1cd03", "HEAD"], cwd=HERE).returncode:
        raise SystemExit("Amendment 1 (dbb1cd03) is not in this branch's history")
    blob = subprocess.run(["git", "show", f"{ORIGINAL[0]}:{ORIGINAL[1]}"], cwd=HERE, capture_output=True, check=True).stdout
    if hashlib.sha256(blob).hexdigest() != ORIGINAL[2]:
        raise SystemExit("the original screen record differs from its pin")
    orig = json.loads(blob)
    sys.path.insert(0, str(HERE))
    import screen_readers as S
    frozen = {r["repo"]: {"revision": r["revision"], "downloads_30d": r["downloads_30d"]} for r in orig["candidates"] if r["repo"] in GATED}
    if set(frozen) != set(GATED) or any(not v["revision"] or not S.valid_downloads(v["downloads_30d"]) for v in frozen.values()):
        raise SystemExit("the original record lacks a frozen revision or download count for a gated candidate")
    fx = (HERE / "fixtures" / "tool_newline.json").read_bytes()
    if hashlib.sha256(fx).hexdigest() != FIXTURE_SHA or orig["fixture"]["sha256"] != FIXTURE_SHA:
        raise SystemExit("the fixture differs from the original screen's")
    fixture = json.loads(fx)
    new = {r["repo"]: r for r in rescreen.remote(frozen, fixture)}
    rows = [new.get(r["repo"], r) for r in orig["candidates"]]
    if [r["repo"] for r in rows] != S.CANDIDATES:
        raise SystemExit("candidate order differs from the RULE's list")
    pick = S.pick_of(rows, "passes")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"what": "G3 Amendment 1 point 3: the gated re-screen and the final pick", "created_pt": stamp, "git": git("rev-parse", "HEAD").strip(),
           "adapter sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "screen_readers.py": {"commit": SCREEN[0], "sha256": SCREEN[1]},
           "original record": {"commit": ORIGINAL[0], "path": ORIGINAL[1], "sha256": ORIGINAL[2]},
           "re-screened": list(GATED), "unchanged": [r["repo"] for r in rows if r["repo"] not in GATED],
           "hf_access": "the two gated candidates: Modal secret huggingface-secret-aswin inside one Modal CPU container (app " + APP + ")",
           "candidates": rows, "pick": pick, "outcome": "picked" if pick else "no candidate passes: G3 stops"}
    out = HERE / "results" / f"screen_final_{stamp}.json"
    with out.open("x") as f:
        json.dump(rec, f, indent=1)
    print(f"{'candidate':48} {'a':>2} {'b':>2} {'c':>2} {'d':>2} {'params':>8} {'downloads':>10}  first failing reason")
    for r in rows:
        why = next((f"{k}: {r[k]['reason']}" for k in "abcd" if not r[k]["pass"]), "")
        n = r["b"].get("params")
        print(f"{r['repo']:48} " + " ".join(f"{'Y' if r[k]['pass'] else 'n':>2}" for k in "abcd")
              + f" {'' if n is None else f'{n / 1e9:.2f}B':>8} {r['downloads_30d'] if r['downloads_30d'] is not None else '-':>10}"
              + f"  {'(re-screened) ' if r['repo'] in GATED else ''}{why[:100]}")
    print(f"pick: {pick} ; record {out.relative_to(HERE.parents[1])}")
