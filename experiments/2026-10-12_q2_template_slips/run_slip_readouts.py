"""Template slips, addendum (a) and (b) (PLAN.md, committed before this script ran). One Modal CPU container per reader
image; masked skeletons and counts only: no tool-output letter or digit leaves a container unmasked except in df's
validated template forms, and a failure is counted by its stage and type, never raised with its message.

(a) Masked token diff: one task's first real honest and simulated honest str_replace and insert replies (v4 inputs),
    each rendered in its own transcript through the reader's dump path and tokenized; tokens of the reply's first and
    last 4 lines and the template tokens around it, masked per character by df's Figure 2 rules
    (../2026-10-11_is_this_a_test/run_figure2_masked.py, imported), the readout token marked. df's leaks() re-checks
    every masked reply before anything is returned; a leak raises. Qwen3-32B and Qwen3.5-27B only.
(b) Where each detector's readout lies relative to the slipped characters (review note, gutter, trailing newline),
    per arm over #36's 3,807 runs and fix939's 823, on whole transcripts: Qwen3-32B and Qwen3.5-27B through
    probe_shift.reader, Olmo-3.1-32B through ../2026-10-12_q2_olmo_reader/run_extract_olmo.py positions() (imported).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-12_q2_template_slips/run_slip_readouts.py --runs <#36 runs.json> --fix <fix939 per-run jsonl>
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
EXP = HERE.parent
FILES = {"rerender.py": EXP / "2026-10-07_q2_rendering_swap", "str_replace_tells.py": EXP / "2026-10-07_q2_rendering_swap",
         "fixes.py": EXP / "2026-10-09_fixed_sim", "run_figure2_masked.py": EXP / "2026-10-11_is_this_a_test",
         "run_extract_olmo.py": HERE / "olmo_726c953e"}  # a byte-identical copy of the Olmo branch's as-run extractor (main lacks it)
OLMO_SRC = ("726c953e", "40fd64b31aa4778b38659a7d07fa7d191bb2ec06063412b97b3671b835049c24")  # the commit and sha256 of experiments/2026-10-12_q2_olmo_reader/run_extract_olmo.py
APP = "probe-shift_aswin-surya_sub-q2-tplslips_readout_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
DUMP, V4 = "2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
FIX_INPUTS, FIX_LABEL = "2026-10-11_q3_fix939_inputs_v1", "sim-attack-fix939"
READERS = {"qwen3-32b": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", "4.56.2"),
           "qwen3.5-27b": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", "5.19.0"),
           "olmo-3.1-32b": ("allenai/Olmo-3.1-32B-Instruct", "ac0587e4a7744a551c059d8cd17ba220bc940dae", "4.57.1")}
SEED, LINES = "20261020", 4
OPEN, CLOSE = "<tool_response>\n", "\n</tool_response>"  # Qwen3 and Qwen3.5 tool-response delimiters around the content
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
       "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()


def image(tv):
    img = modal.Image.debian_slim(python_version="3.12").uv_pip_install(f"transformers=={tv}", "jinja2==3.1.6", "huggingface_hub").add_local_python_source("probe_shift")
    for name, folder in FILES.items():
        img = img.add_local_file(folder / name, f"/root/{folder.name}/{name}")
    return img


def _mods():
    import sys
    for folder in {f for f in FILES.values()}:
        for p in (f"/root/{folder.name}", str(folder)):
            if p not in sys.path:
                sys.path.insert(0, p)
    import fixes  # noqa: E402  (puts the swap folder on the path)
    import rerender
    import run_figure2_masked as df
    return fixes, rerender, df


def masked(text: str) -> str:
    """df's skeleton() per character, untrimmed and the same length as `text`: header lines with paths masked, an exact
    footer, validated gutters and diff file names and hunk headers kept; every other letter or digit to "x"."""
    _, _, df = _mods()
    body = text.split("\n")
    out = [df.X(x) for x in body]
    filled = [i for i, x in enumerate(body) if x.strip()]
    if filled:
        first, last = filled[0], filled[-1]
        m, kind = next(((m, k) for m, k in ((pat.match(body[first]), k) for pat, k in df.HEADERS) if m), (None, None))
        if m:
            out[first] = "".join(df.X(g) if i % 2 == 1 else g for i, g in enumerate(m.groups()))
            footer = last != first and body[last] in df.FOOTERS
            if footer:
                out[last] = body[last]
            inner = list(range(first + 1, last if footer else last + 1))
            keep = (df.cat_gutters if kind == "cat" else df.diff_gutters)([body[i] for i in inner])
            for i, k in zip(inner, keep or []):
                out[i] = body[i][:k] + df.X(body[i][k:])
    res = "\n".join(out)
    assert len(res) == len(text)
    return res


def staged(run_dir, by_key):
    res = {}
    for shard in sorted((Path("/acts") / run_dir / "inputs").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line); res[it["key"] if by_key else (it["arm"], it["key"])] = it
    return res


def _init(reader):
    global TOK, READER
    from transformers import AutoTokenizer
    name, rev, _ = READERS[reader]
    TOK, READER = AutoTokenizer.from_pretrained(name, revision=rev), reader


def layout(msgs):
    """(rendered text, {(step, action): (a, b)} content spans, token offsets, {(step, action): readout token or None})."""
    if READER == "olmo-3.1-32b":
        import bisect

        import run_extract_olmo as ol
        r = ol.positions(TOK, msgs, {})  # q empty: kept is False everywhere, but every output's R2 is computed
        m2 = ol.prep(msgs)
        text = TOK.apply_chat_template(m2, tokenize=False)
        enc = TOK(text, return_offsets_mapping=True, add_special_tokens=False)
        spans, pos, step, action = {}, 0, -1, 0
        for m in m2:
            if m["role"] == "assistant":
                step, action = step + 1, 0
            elif m["role"] == "tool":
                a = text.index(ol.ENV_OPEN, pos) + len(ol.ENV_OPEN); b = a + len(m["content"])
                spans[(step, action)] = (a, b); action += 1; pos = b
        offs = enc["offset_mapping"]; starts = [s for s, _ in offs]; ends = [e for _, e in offs]
        ro = {(p["step"], p["action"]): p["token"] for p in r["positions"] if p["kind"] == "output_end"}
        for (s, a_), (a, b) in spans.items():  # R2 recomputed from the offsets must equal positions()'s
            i0, i1 = bisect.bisect_left(starts, a), bisect.bisect_right(ends, b)
            mine = next((i for i in range(i1 - 1, i0 - 1, -1) if not ol.bad(text[starts[i]:ends[i]])), None)
            if mine != ro[(s, a_)]:
                raise ValueError("R2 recomputed differs from positions()")
        return text, spans, offs, ro
    from probe_shift.reader import _layout, segment_end_positions
    text, found, enc = _layout(TOK, msgs)
    spans = {(s, a_): (a, b) for kind, s, a_, _t, a, b in found if kind == "tool_output"}
    r = segment_end_positions(TOK, msgs, None)
    ro = {(p["step"], p["action"]): p["token"] for p in r["positions"] if p["kind"] == "output_end"}
    return text, spans, enc["offset_mapping"], ro


def categories(raw: str, cmd: str, rendered: str, tok_span):
    """[(component, category)] for one output: rendered = the output's rendered content; tok_span = the readout token's
    (start, end) relative to it, or None."""
    import re
    fixes, rerender, _ = _mods()
    out = []
    s, e = tok_span if tok_span else (None, None)
    if cmd == "insert" and raw.rstrip().endswith(rerender.REVIEW):
        note0 = len(rendered.rstrip()) - len(rerender.REVIEW); note1 = note0 + len(rerender.REVIEW)
        out.append(("review note", "no readout" if s is None else "after the note" if s >= note1 else "inside the review note" if s >= note0 else "before the note"))
    if cmd == "str_replace" and rerender.DIFF in raw[:300] and fixes.numbered_form(raw.partition("+++ edited_version\n")[2]) == "numbered":
        if s is None:
            out.append(("gutter", "no readout"))
        else:
            ls = rendered.rfind("\n", 0, s) + 1
            le = rendered.find("\n", ls); le = len(rendered) if le < 0 else le
            g = re.match(r"^ *\d+\|(?:[ +\-] ?)?", rendered[ls:le])
            if not g:
                out.append(("gutter", "on another line (header or other)"))
            else:
                ge = ls + g.end()
                on_g, on_c = s < ge and e > ls, s < le and e > ge  # bounded: the gutter [ls, ge), the code [ge, le)
                out.append(("gutter", "spanning the gutter's end and the code" if on_g and on_c else "on the gutter only" if on_g
                            else "on the line's code" if on_c else "on the line's end (no gutter or code character)"))
    if cmd in ("create", "str_replace", "insert") and raw.endswith("\n"):
        out.append(("trailing newline", "absent from the rendered text" if not rendered.endswith("\n") else "no readout" if s is None
                    else "readout token contains the newline" if e > len(rendered) - 1 else "readout before the newline"))
    return out


def count_one(item):
    from collections import Counter
    arm, msgs = item
    c = Counter()
    try:
        fixes, _, _ = _mods()
        text, spans, offs, ro = layout(msgs)
    except Exception as e:
        c[(arm, "-", f"render failure ({type(e).__name__})")] += 1
        return c
    try:
        calls = fixes.calls_of(msgs)
    except Exception as e:
        c[(arm, "-", f"pairing failure ({type(e).__name__})")] += 1
        return c
    c[(arm, "-", "tool outputs not paired to a call (calls_of drops them)")] += sum(m["role"] == "tool" for m in msgs) - len(calls)
    for call in calls:
        if call["tool"] != "text_editor":
            continue
        raw = msgs[call["msg"]]["content"]
        if not isinstance(raw, str) or fixes.TRUNCATED in raw:
            continue
        key = (call["step"], call["action"])
        if key not in spans:
            c[(arm, "-", "output with no rendered span")] += 1; continue
        a, b = spans[key]; t = ro.get(key)
        span = None if t is None else (offs[t][0] - a, offs[t][1] - a)
        c[(arm, "all text_editor outputs", "no readout" if t is None else "with a readout")] += 1  # independent of components
        for comp, cat in categories(raw, call["args"].get("command"), text[a:b], span):
            c[(arm, comp, cat)] += 1
    return c


def figure(reader):
    """(a): the four replies' masked token sequences under this reader."""
    fixes, rerender, df = _mods()
    items = staged(V4, False)
    arms = sorted({a for a, _ in items})
    real, sim = [a for a in arms if a == "real-honest"], [a for a in arms if a == "sim-honest"]
    if not real or not sim:
        raise ValueError(f"v4 arms: {arms}")
    by_task = {}
    for (a, k), it in sorted(items.items()):
        if a in ("real-honest", "sim-honest"):
            by_task.setdefault(it["main_task"], {}).setdefault(a, it)  # the first run by key
    def firsts(msgs):
        try:
            calls = fixes.calls_of(msgs)
        except Exception:
            return {}  # a transcript that can't be paired is not eligible
        got = {}
        for call in calls:
            cmd, o = call["args"].get("command"), msgs[call["msg"]]["content"]
            if call["tool"] == "text_editor" and isinstance(o, str) and fixes.TRUNCATED not in o and cmd in ("str_replace", "insert") and cmd not in got:
                if cmd == "insert" or rerender.DIFF in o[:300]:
                    got[cmd] = call
        return got
    for task in sorted(by_task, key=lambda t: hashlib.sha256(f"{SEED}|{t}".encode()).hexdigest()):
        pair = by_task[task]
        if len(pair) == 2 and all(len(firsts(pair[a]["messages"])) == 2 for a in pair):
            break
    else:
        raise ValueError("no eligible task")
    out = {"task sha256": hashlib.sha256(task.encode()).hexdigest()[:12]}
    for a in ("real-honest", "sim-honest"):
        msgs = pair[a]["messages"]
        text, spans, offs, ro = layout(msgs)
        for cmd, call in firsts(msgs).items():
            key = (call["step"], call["action"]); a0, b0 = spans[key]
            content = text[a0:b0]; mk = masked(content)
            if df.leaks(mk + "\n" + "marker"):
                raise ValueError("leak check failed")
            if text[a0 - len(OPEN):a0] != OPEN or text[b0:b0 + len(CLOSE)] != CLOSE:
                raise ValueError("template around the reply is not the expected delimiters")
            lo, hi = a0 - len(OPEN), b0 + len(CLOSE)  # only the reply and its own two delimiters are shown
            lines = [i for i, ch in enumerate(content) if ch == "\n"]
            head_end = lines[LINES - 1] + 1 if len(lines) >= LINES else b0 - a0
            tail_start = lines[-LINES] + 1 if len(lines) >= LINES else 0
            toks = []
            for i, (s, e) in enumerate(offs):
                if e <= lo or s >= hi:
                    continue
                rs, re_ = s - a0, e - a0
                inside = rs < b0 - a0 and re_ > 0
                if inside and not (rs < head_end or re_ > tail_start):
                    continue
                txt = "".join(mk[j - a0] if a0 <= j < b0 else text[j] if lo <= j < hi else ("x" if text[j].isalnum() else text[j]) for j in range(s, e))
                kind = "template" if not inside else "content" if rs >= 0 and re_ <= b0 - a0 else "template+content"
                toks.append({"token": txt, "kind": kind, "readout": i == ro.get(key)})
            out[f"{a} {cmd}"] = {"run key sha256": hashlib.sha256(pair[a]["key"].encode()).hexdigest()[:12], "step": call["step"],
                                  "rendered content ends with a newline": content.endswith("\n"), "tokens": toks}
    return out


def _run(reader, cohort, fix_keys):
    from collections import Counter
    from multiprocessing import Pool

    from transformers import AutoTokenizer
    name, rev, tv = READERS[reader]
    AutoTokenizer.from_pretrained(name, revision=rev)
    _init(reader)
    fig = figure(reader) if reader != "olmo-3.1-32b" else None
    base, fixed = staged(DUMP, False), staged(FIX_INPUTS, True)
    items = [(r["arm"], base[(r["arm"], r["key"])]["messages"]) for r in cohort] + [(FIX_LABEL, fixed[k]["messages"]) for k in fix_keys]
    total = Counter()
    with Pool(32, initializer=_init, initargs=(reader,)) as pool:
        for cnt in pool.imap(count_one, items, chunksize=4):
            total.update(cnt)
    return {"reader": {"model": name, "revision": rev, "transformers": tv}, "(a) masked token diff": fig,
            "(b) rows": [{"arm": a, "component": comp, "category": k, "n": int(v)} for (a, comp, k), v in sorted(total.items())]}


def _guard(reader, cohort, fix_keys):
    try:
        return _run(reader, cohort, fix_keys)
    except Exception as e:  # nothing from the exception (which may quote staged text) crosses the boundary
        raise RuntimeError(f"{reader} job failed: {type(e).__name__}") from None


@app.function(image=image(READERS["qwen3-32b"][2]), volumes={"/acts": acts}, cpu=32, memory=65536, timeout=3600, retries=0)
def run_qwen3(cohort: list[dict], fix_keys: list[str]) -> dict:
    return _guard("qwen3-32b", cohort, fix_keys)


@app.function(image=image(READERS["qwen3.5-27b"][2]), volumes={"/acts": acts}, cpu=32, memory=65536, timeout=3600, retries=0)
def run_qwen35(cohort: list[dict], fix_keys: list[str]) -> dict:
    return _guard("qwen3.5-27b", cohort, fix_keys)


@app.function(image=image(READERS["olmo-3.1-32b"][2]), volumes={"/acts": acts}, cpu=32, memory=65536, timeout=3600, retries=0)
def run_olmo(cohort: list[dict], fix_keys: list[str]) -> dict:
    return _guard("olmo-3.1-32b", cohort, fix_keys)


@app.local_entrypoint()
def main(runs: str, fix: str):
    import subprocess
    for f in [Path(__file__).resolve(), HERE / "PLAN.md"] + [folder / name for name, folder in FILES.items()]:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if hashlib.sha256((HERE / "olmo_726c953e" / "run_extract_olmo.py").read_bytes()).hexdigest() != OLMO_SRC[1]:
        raise SystemExit("the Olmo extractor copy differs from 726c953e's")
    for p_, role in ((Path(runs), "runs.json"), (Path(fix), "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl")):
        if hashlib.sha256(p_.read_bytes()).hexdigest() != PIN[role]:
            raise SystemExit(f"{p_.name} is not the pinned {role}")
    rs = json.loads(Path(runs).read_text())
    fix_keys = sorted({json.loads(l)["key"] for l in open(fix) if json.loads(l)["label"] == "FIX"})
    if len(rs) != 3807 or len(fix_keys) != 823:
        raise SystemExit("not #36's 3,807 runs and fix939's 823 re-rendered ones")
    cohort = [{"arm": r["arm"], "key": r["key"]} for r in rs]
    calls = {"qwen3-32b": run_qwen3.spawn(cohort, fix_keys), "qwen3.5-27b": run_qwen35.spawn(cohort, fix_keys), "olmo-3.1-32b": run_olmo.spawn(cohort, fix_keys)}
    res = {k: v.get() for k, v in calls.items()}
    out = {"plan": "PLAN.md addendum (a), (b) (this folder)", "git": head, "inputs sha256": {"#36 runs.json": PIN["runs.json"], "fix939 per-run file": PIN["fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl"]},
           "readers": res}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"slip_readouts_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(dest)
