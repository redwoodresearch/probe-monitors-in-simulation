"""Template slips, step 1 (PLAN.md, committed before this script ran): which editor-format slip components survive each
reader's chat template, as each dump rendered (probe_shift.reader). One Modal CPU container per reader image
(Qwen3-32B with transformers 4.56.2; Qwen3.5-27B with 5.19.0), counts only: no tool-output text leaves a container, and
a failure is counted by its stage and type, never raised with its message.

For each text_editor output and component, two versions differ only in that component (PLAN.md). Each is rendered in
a minimal conversation; it "survives in the prompt" if the rendered prompts differ, and "at its own readout" if the
token ids up to and including its output_end readout differ (segment_end_positions, the dump's path). A seeded context
check repeats both tests in the full transcripts (up to 25 outputs per arm and component).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-12_q2_template_slips/run_template_slips.py --runs <#36 runs.json> --fix <fix939 per-run jsonl>
"""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
SWAP, FIXED = HERE.parent / "2026-10-07_q2_rendering_swap", HERE.parent / "2026-10-09_fixed_sim"
APP = "probe-shift_aswin-surya_sub-q2-tplslips_render_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
DUMP = "2026-10-06_la_segment-end_qwen3-32b_v1"
FIX_DUMP, FIX_INPUTS, FIX_ARM, FIX_LABEL = ("2026-10-11_q3_fix939_qwen3-32b_l48-63_v1", "2026-10-11_q3_fix939_inputs_v1",
                                            "sim-attack-note-sr-both", "sim-attack-fix939")
READERS = {"qwen3-32b": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", "4.56.2"),
           "qwen3.5-27b": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", "5.19.0")}
SEED, N_CONTEXT = 20261020, 25
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
       "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"}
GUTTER = re.compile(r"^ *\d+\|(?:([ +\-])(?: (.*))?)?$")  # fixes.REAL_GUTTER_LINE, with the sign and code captured
CANON = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@(.*)$")
ONE = re.compile(r"^@@ -\d+(,\d+)? \+\d+(,\d+)? @@")

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()


def image(tv):
    return (modal.Image.debian_slim(python_version="3.12").uv_pip_install(f"transformers=={tv}", "jinja2==3.1.6", "huggingface_hub")
            .add_local_python_source("probe_shift")
            .add_local_file(SWAP / "rerender.py", "/root/2026-10-07_q2_rendering_swap/rerender.py")
            .add_local_file(SWAP / "str_replace_tells.py", "/root/2026-10-07_q2_rendering_swap/str_replace_tells.py")
            .add_local_file(FIXED / "fixes.py", "/root/2026-10-09_fixed_sim/fixes.py"))


def _mods():
    import sys
    for p in ("/root/2026-10-09_fixed_sim", str(FIXED)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import fixes  # noqa: E402  (it puts the swap folder on the path and imports rerender, str_replace_tells)
    import rerender
    import str_replace_tells
    return fixes, rerender, str_replace_tells


def strip_gutter(out: str) -> str:
    head, sep, body = out.partition("+++ edited_version\n")
    lines = []
    for line in body.split("\n"):
        m = GUTTER.match(line)
        lines.append(line if line.startswith("@@") or not m else (m.group(1) or " ") + (m.group(2) or ""))
    return head + sep + "\n".join(lines)


def pairs_of(arm: str, messages: list[dict]):
    """[(msg, step, action, component, present, with, without)] for the text_editor outputs, plus counts of what was
    left out, by reason. Versions differ only in the component (PLAN.md)."""
    from collections import Counter
    fixes, rerender, srt = _mods()
    out, skip = [], Counter()
    snippets = {}
    if arm == "sim-attack":  # (e): the real snippet, built from the as-simulated transcript
        new, recs = rerender.rerender_messages(messages, "snippet_no_note")
        for r in recs:
            if r["category"] in rerender.RENDERED:
                snippets[r["msg"]] = new[r["msg"]]["content"]
            else:
                skip[f"(e) insert {r['category']}"] += 1
    for c in fixes.calls_of(messages):
        if c["tool"] != "text_editor":
            continue
        cmd, o, i = c["args"].get("command"), messages[c["msg"]]["content"], c["msg"]
        skip[f"text_editor {cmd} outputs"] += 1
        if not isinstance(o, str):
            skip["non-text output (left out)"] += 1; continue
        if fixes.TRUNCATED in o:
            skip["harness-truncated output (left out)"] += 1; continue
        add = lambda comp, present, w, wo: out.append((i, c["step"], c["action"], comp, present, w, wo))  # noqa: E731
        if cmd in ("create", "str_replace", "insert"):
            base = o.rstrip("\n"); add("a", o.endswith("\n"), base + "\n", base)
        diff = rerender.DIFF in o[:300]
        if cmd == "str_replace" and diff:
            form = fixes.numbered_form(o.partition("+++ edited_version\n")[2])
            if form == "numbered":
                add("b", True, o, strip_gutter(o))
            elif form == "sign_first":
                skip["(b) sign-first diff (left out)"] += 1
            else:
                numbered, cat = srt.add_gutter(o)
                if cat == "plain":
                    add("b", False, numbered, o)
                else:
                    skip[f"(b) {cat} diff (left out)"] += 1
        if cmd == "insert":
            base = o.rstrip("\n"); present = base.rstrip().endswith(rerender.REVIEW)
            if present:
                base = base.rstrip()[: -len(rerender.REVIEW)].rstrip("\n")
            add("c", present, base + "\n" + rerender.REVIEW, base)
            skip["(e) insert in snippet form (preamble)" if rerender.PREAMBLE in o[:400] else "(e) insert not in snippet form"] += 1
            if i in snippets:
                add("e", False, snippets[i], o.rstrip("\n"))
        if cmd in ("insert", "str_replace") and diff:
            heads = [l for l in o.split("\n") if l.startswith("@@")]
            for h in heads:
                skip["(d) header form: " + ("both counts" if CANON.match(h) else "one count or none" if ONE.match(h) else "other")] += 1
            if any(CANON.match(h) for h in heads):
                short = "\n".join(CANON.sub(lambda m: f"@@ -{m.group(1)} +{m.group(3)} @@{m.group(5)}", l) if l.startswith("@@") else l for l in o.split("\n"))
                add("d", True, o, short)
    return out, skip


def _work_items(cohort, fix_keys):
    def staged(run_dir, by_key):
        res = {}
        for shard in sorted((Path("/acts") / run_dir / "inputs").glob("shard_*.jsonl")):
            for line in shard.read_text().splitlines():
                it = json.loads(line); res[it["key"] if by_key else (it["arm"], it["key"])] = it["messages"]
        return res
    base, fixed = staged(DUMP, False), staged(FIX_INPUTS, True)
    items = [(r["arm"], r["key"], base[(r["arm"], r["key"])], str(Path("/acts") / DUMP / "acts" / r["arm"] / f"{r['filename']}.json")) for r in cohort]
    fname = {r["key"]: r["filename"] for r in cohort if r["arm"] == "sim-attack"}
    items += [(FIX_LABEL, k, fixed[k], str(Path("/acts") / FIX_DUMP / "meta" / FIX_ARM / f"{fname[k]}.json")) for k in fix_keys]
    return items


def keys_one(item):
    arm, key, msgs, _ = item
    try:
        return [(arm, key, i, comp) for i, _s, _a, comp, *_ in pairs_of(arm, msgs)[0]]
    except Exception:
        return []  # counted in the main pass


def _init(reader):
    global TOK, READER
    from transformers import AutoTokenizer
    name, rev, _ = READERS[reader]
    TOK, READER = AutoTokenizer.from_pretrained(name, revision=rev), reader


def conv(content):
    return [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "text_editor", "arguments": {"command": "view", "path": "/app/x"}}}]},
            {"role": "tool", "content": content}]


def tests(msgs, step, action):
    """{path: (prompt text, own-readout key)} for this reader's paths. The prompt is rendered first and on its own; the
    own-readout key is the tuple of token ids through the output's readout, None if it has no readout, or
    "failed: <type>" if the reader's boundary checks reject the transcript (counted, the prompt test still stands)."""
    from probe_shift.reader import prepare, segment_end_positions
    render = lambda m: TOK.apply_chat_template(m, tokenize=False, enable_thinking=False)  # noqa: E731
    try:
        r = segment_end_positions(TOK, msgs, None)
        p = next(p for p in r["positions"] if p["kind"] == "output_end" and p["step"] == step and p["action"] == action)
        own = None if p["token"] is None else tuple(r["ids"][: p["token"] + 1])
    except Exception as e:
        own = f"failed: {type(e).__name__}"
    if READER == "qwen3-32b":
        return {"qwen3-32b": (render(msgs), own)}
    return {"qwen3.5-27b (dump path: prepare, then the template)": (render(prepare(TOK, msgs)), own),
            "qwen3.5-27b (raw template, prompt only)": (render(msgs), None)}


def count_one(arg):
    from collections import Counter
    (arm, key, msgs, qpath), context = arg
    c = Counter()
    try:
        prs, skip = pairs_of(arm, msgs)
    except Exception as e:
        c[("all", arm, "-", f"pairing failure ({type(e).__name__})")] += 1
        return c
    for k, v in skip.items():
        c[("all", arm, "-", k)] += v
    try:
        kept = {(p["step"], p["action"]): bool(p["kept"]) for p in json.loads(Path(qpath).read_text())["positions"] if p["kind"] == "output_end"}
    except Exception as e:
        kept = {}; c[("all", arm, "-", f"Qwen3-32B meta failure ({type(e).__name__})")] += 1
    for i, step, action, comp, present, w, wo in prs:
        c[("all", arm, comp, "outputs with a constructible pair")] += 1
        c[("all", arm, comp, "present in the output")] += present
        c[("all", arm, comp, "constructible pairs inside Qwen3-32B's 32k window")] += kept.get((step, action), False)
        try:
            tw, two = tests(conv(w), 0, 0), tests(conv(wo), 0, 0)
        except Exception as e:
            c[("all", arm, comp, f"render failure ({type(e).__name__})")] += 1
            continue
        for path in tw:
            sp, so = tw[path][0] != two[path][0], tw[path][1] != two[path][1]
            c[(path, arm, comp, "survive in the prompt")] += sp
            if "raw template" not in path:
                if isinstance(tw[path][1], str) or isinstance(two[path][1], str):
                    c[(path, arm, comp, "own-readout check failed (reader boundary checks)")] += 1
                else:
                    c[(path, arm, comp, "survive at the output's own readout")] += so
            if (arm, key, i, comp) in context:
                try:
                    fw, fwo = [dict(m) for m in msgs], [dict(m) for m in msgs]
                    fw[i]["content"], fwo[i]["content"] = w, wo
                    xw, xwo = tests(fw, step, action), tests(fwo, step, action)
                    own_ok = "raw template" in path or not any(isinstance(x[path][1], str) for x in (tw, two, xw, xwo))
                    agree = ((xw[path][0] != xwo[path][0]) == sp) and ("raw template" in path or not own_ok or (xw[path][1] != xwo[path][1]) == so)
                    c[(path, arm, comp, "context check: own-readout comparison skipped (boundary checks)")] += not own_ok
                    c[(path, arm, comp, "context check: cases")] += 1
                    c[(path, arm, comp, "context check: agree")] += agree
                except Exception as e:
                    c[(path, arm, comp, f"context check failure ({type(e).__name__})")] += 1
    return c


def _render(reader, cohort, fix_keys):
    global TOK, READER
    import random
    from collections import Counter
    from multiprocessing import Pool

    from transformers import AutoTokenizer
    name, rev, tv = READERS[reader]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)  # download once; the pool's processes load from the cache
    ph = {}
    from probe_shift.reader import prepare
    TOK, READER = tok, reader
    for label, content in (("output without a trailing newline", "<<OUTPUT>>"), ("output with a trailing newline", "<<OUTPUT>>\n")):
        m = conv(content)
        txt = tok.apply_chat_template(m, tokenize=False, enable_thinking=False)
        ph[label] = {"raw template": txt[txt.index("<<OUTPUT>>") - 40:]}
        if reader != "qwen3-32b":
            t2 = tok.apply_chat_template(prepare(tok, m), tokenize=False, enable_thinking=False)
            ph[label]["dump path"] = t2[t2.index("<<OUTPUT>>") - 40:]
    items = _work_items(cohort, fix_keys)
    with Pool(32) as pool:
        keys = [k for ks in pool.imap(keys_one, items, chunksize=8) for k in ks]
    by = {}
    for k in sorted(keys):
        by.setdefault((k[0], k[3]), []).append(k)
    rng = random.Random(SEED)
    context = {k for g in sorted(by) for k in (rng.sample(by[g], N_CONTEXT) if len(by[g]) > N_CONTEXT else by[g])}
    total = Counter()
    with Pool(32, initializer=_init, initargs=(reader,)) as pool:
        for cnt in pool.imap(count_one, [(it, {k for k in context if k[0] == it[0] and k[1] == it[1]}) for it in items], chunksize=4):
            total.update(cnt)
    return {"reader": {"model": name, "revision": rev, "transformers": tv}, "template around a placeholder output": ph,
            "rows": [{"path": p, "arm": a, "component": comp, "count": k, "n": int(v)} for (p, a, comp, k), v in sorted(total.items())]}


@app.function(image=image(READERS["qwen3-32b"][2]), volumes={"/acts": acts}, cpu=32, memory=65536, timeout=3600, retries=0)
def render_qwen3(cohort: list[dict], fix_keys: list[str]) -> dict:
    try:
        return _render("qwen3-32b", cohort, fix_keys)
    except Exception as e:  # nothing from the exception (which may quote staged text) crosses the boundary
        raise RuntimeError(f"render job failed: {type(e).__name__}") from None


@app.function(image=image(READERS["qwen3.5-27b"][2]), volumes={"/acts": acts}, cpu=32, memory=65536, timeout=3600, retries=0)
def render_qwen35(cohort: list[dict], fix_keys: list[str]) -> dict:
    try:
        return _render("qwen3.5-27b", cohort, fix_keys)
    except Exception as e:
        raise RuntimeError(f"render job failed: {type(e).__name__}") from None


@app.local_entrypoint()
def main(runs: str, fix: str):
    import subprocess
    for f in (Path(__file__).resolve(), HERE / "PLAN.md", SWAP / "rerender.py", SWAP / "str_replace_tells.py", FIXED / "fixes.py"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for p_, role in ((Path(runs), "runs.json"), (Path(fix), "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl")):
        if hashlib.sha256(p_.read_bytes()).hexdigest() != PIN[role]:
            raise SystemExit(f"{p_.name} is not the pinned {role}")
    rs = json.loads(Path(runs).read_text())
    fix_keys = sorted({json.loads(l)["key"] for l in open(fix) if json.loads(l)["label"] == "FIX"})
    if len(rs) != 3807 or len(fix_keys) != 823:
        raise SystemExit("not #36's 3,807 runs and fix939's 823 re-rendered ones")
    cohort = [{"arm": r["arm"], "key": r["key"], "filename": r["filename"]} for r in rs]
    calls = {"qwen3-32b": render_qwen3.spawn(cohort, fix_keys), "qwen3.5-27b": render_qwen35.spawn(cohort, fix_keys)}
    res = {k: v.get() for k, v in calls.items()}
    out = {"plan": "PLAN.md (this folder)", "git": head, "inputs sha256": {"#36 runs.json": PIN["runs.json"], "fix939 per-run file": PIN["fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl"]},
           "readers": {k: v["reader"] for k, v in res.items()}, "template around a placeholder output": {k: v["template around a placeholder output"] for k, v in res.items()},
           "rows": [r | {"container": k} for k, v in res.items() for r in v["rows"]]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"template_slips_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(dest)
