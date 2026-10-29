"""A third reader, step 1 (PLAN.md, committed with this script before it runs): Olmo-3.1-32B-Instruct's chat template
and tokenizer on #36's staged transcripts and fix939's re-rendered ones, in one Modal CPU container (no weights). For
each tool output it finds the readout token under R0 (#36's rule) and R1 (trailing whitespace removed first), and
compares the 32k cut with Qwen3-32B's extraction. Counts only: no token or output text leaves the container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-12_q2_olmo_reader/run_olmo_render.py --runs <#36 runs.json> --fix <fix939 per-run jsonl>
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-olmo-render_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
DUMP = "2026-10-06_la_segment-end_qwen3-32b_v1"
FIX_DUMP, FIX_INPUTS, FIX_ARM = "2026-10-11_q3_fix939_qwen3-32b_l48-63_v1", "2026-10-11_q3_fix939_inputs_v1", "sim-attack-note-sr-both"
MODEL, REVISION, CUT = "allenai/Olmo-3.1-32B-Instruct", "ac0587e4a7744a551c059d8cd17ba220bc940dae", 32768
ENV_OPEN, END = "<|im_start|>environment\n", "<|im_end|>"
SWAP = HERE.parent / "2026-10-07_q2_rendering_swap"
LABEL = {"real-honest": "real honest", "real-attack-e1": "real e1", "real-attack-e2": "real e2", "sim-attack": "simulated (as simulated)"}
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
       "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"}
app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub==0.35.3")
         .add_local_file(SWAP / "rerender.py", "/root/rerender.py"))


def _init():
    import sys

    from transformers import AutoTokenizer
    sys.path.insert(0, "/root")
    from rerender import REVIEW
    global TOK, REV
    TOK, REV = AutoTokenizer.from_pretrained(MODEL, revision=REVISION), REVIEW.strip()


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


def category(t: str) -> str:
    if t.strip() == "":
        return "whitespace only"
    return "content with a trailing newline inside the token" if t.endswith("\n") else "other content"


def one(item):
    """(label, messages, Qwen meta summary) -> counts for one run. Any failure is counted by its stage and type only (an
    exception's message or attributes may quote content), never raised."""
    from collections import Counter
    label, msgs, q = item
    c = Counter(); c[(label, "runs")] += 1
    stage = "render"
    try:
        return _one(label, msgs, q, c)
    except Exception as e:
        c[(label, f"render failure: {getattr(e, 'stage', stage)} ({type(e).__name__})")] += 1
        return c, None


class Fail(Exception):
    def __init__(self, stage):
        super().__init__(stage); self.stage = stage


def _one(label, msgs, q, c):
    import bisect
    msgs = prep(msgs)
    text = TOK.apply_chat_template(msgs, tokenize=False)
    tools = [m for m in msgs if m["role"] == "tool"]
    if text.count(ENV_OPEN) != len(tools) or any(ENV_OPEN in m["content"] or END in m["content"] for m in tools):
        raise Fail("a delimiter quoted in content")
    spans, pos, step, action, cmds = [], 0, -1, 0, {}
    for m in msgs:
        if m["role"] == "assistant":
            step, action = step + 1, 0
            for i, tc in enumerate(m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                cmds[(step, i)] = (fn.get("arguments") or {}).get("command") if fn.get("name") == "text_editor" else None
        elif m["role"] == "tool":
            a = text.index(ENV_OPEN, pos) + len(ENV_OPEN); b = a + len(m["content"])
            if text[a:b] != m["content"] or not text.startswith(END, b):
                raise Fail("a tool output not verbatim")
            spans.append((step, action, a, b, m["content"])); action += 1; pos = b
    enc = TOK(text, return_offsets_mapping=True, add_special_tokens=False)
    starts, ends = [s_ for s_, _ in enc["offset_mapping"]], [e for _, e in enc["offset_mapping"]]
    n = len(enc["input_ids"])

    def last_inside(a, b):
        i0, i1 = bisect.bisect_left(starts, a), bisect.bisect_right(ends, b)
        return i1 - 1 if i1 > i0 else None
    c[(label, "truncated: Olmo")] += n > CUT; c[(label, "truncated: Qwen3-32B")] += q["truncated"]
    for s, a_, a, b, content in spans:
        stripped = content.rstrip()
        r0, r1 = last_inside(a, b), last_inside(a, a + len(stripped))
        in0, k1 = r0 is not None and r0 < CUT, r1 is not None and r1 < CUT
        c[(label, "tool outputs")] += 1
        c[(label, "outputs in Olmo's window (R0 token below the cut)")] += in0
        c[(label, "of those, with an R1 readout")] += in0 and r1 is not None
        c[(label, "kept: Olmo C1 (R1)")] += k1
        c[(label, "R1 and R0 differ")] += r1 != r0
        qk = q["kept"].get(f"{s},{a_}")
        c[(label, "tool outputs with no Qwen3-32B position")] += qk is None  # counted as not kept by Qwen3-32B
        c[(label, "kept: Qwen3-32B")] += bool(qk); c[(label, "kept by exactly one reader (C1 R1 vs Qwen3-32B)")] += k1 != bool(qk)
        note0 = a + len(stripped) - len(REV) if stripped.endswith(REV) else None  # the review note's start, by offset
        for rule, r in (("R0", r0), ("R1", r1)):
            for cut, kept in (("C1", r is not None and r < CUT), ("C2", bool(qk))):
                if kept:
                    c[(label, f"readout {rule} at {cut}: {'no token' if r is None else category(text[starts[r]:ends[r]])}")] += 1
                    if cmds.get((s, a_)) == "insert" and note0 is not None:
                        c[(label, f"insert outputs ending with the review note, readout {rule} at {cut}")] += 1
                        c[(label, f"insert outputs ending with the review note, readout {rule} at {cut}: inside the note")] += r is not None and starts[r] >= note0
    return c, (n, q["n_tokens"])


@app.function(image=image, volumes={"/acts": acts}, cpu=32, memory=65536, timeout=3600, retries=0)
def render(cohort: list[dict], fix_keys: list[str]) -> dict:
    try:
        return _render(cohort, fix_keys)
    except Exception as e:  # nothing from the exception (which may quote staged text) crosses the boundary
        raise RuntimeError(f"render job failed: {type(e).__name__}") from None


def _render(cohort, fix_keys):
    from collections import Counter
    from multiprocessing import Pool

    import numpy as np
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(MODEL, revision=REVISION)  # download once; the pool's processes load from the cache

    def staged(run_dir, by_key):
        out = {}
        for shard in sorted((Path("/acts") / run_dir / "inputs").glob("shard_*.jsonl")):
            for line in shard.read_text().splitlines():
                it = json.loads(line); out[it["key"] if by_key else (it["arm"], it["key"])] = it["messages"]
        return out

    def qsum(meta):  # Qwen3-32B's output_end positions: kept flag by "step,action", plus the run's length
        return {"n_tokens": meta["n_tokens"], "truncated": bool(meta["truncated"]),
                "kept": {f"{p['step']},{p['action']}": bool(p["kept"]) for p in meta["positions"] if p["kind"] == "output_end"}}
    base, fixed = staged(DUMP, False), staged(FIX_INPUTS, True)
    items = []
    for r in cohort:
        meta = json.loads((Path("/acts") / DUMP / "acts" / r["arm"] / f"{r['filename']}.json").read_text())
        items.append((LABEL[r["arm"]], base[(r["arm"], r["key"])], qsum(meta)))
    fname = {r["key"]: r["filename"] for r in cohort if r["arm"] == "sim-attack"}
    for k in fix_keys:
        meta = json.loads((Path("/acts") / FIX_DUMP / "meta" / FIX_ARM / f"{fname[k]}.json").read_text())
        items.append(("simulated, re-rendered (823)", fixed[k], qsum(meta)))
    total, lengths = Counter(), {}
    with Pool(32, initializer=_init) as pool:
        for (cnt, nl), it in zip(pool.imap(one, items, chunksize=4), items):
            total.update(cnt)
            if nl:
                lengths.setdefault(it[0], []).append(nl)
    rows = [{"arm": a, "count": k, "n": int(v)} for (a, k), v in sorted(total.items())]
    ratio = {a: {"runs rendered": len(v), "median tokens, Olmo": float(np.median([x[0] for x in v])), "median tokens, Qwen3-32B": float(np.median([x[1] for x in v])),
                 "Olmo / Qwen3-32B tokens per run: median, 10th, 90th": [float(x) for x in np.percentile([x[0] / x[1] for x in v], [50, 10, 90])]} for a, v in lengths.items()}
    return {"rows": rows, "lengths": ratio}


@app.local_entrypoint()
def main(runs: str, fix: str):
    import subprocess
    for f in (Path(__file__).resolve(), HERE / "PLAN.md", SWAP / "rerender.py"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for p_, role in ((Path(runs), "runs.json"), (Path(fix), "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl")):
        if hashlib.sha256(p_.read_bytes()).hexdigest() != PIN[role]:
            raise SystemExit(f"{p_.name} is not the pinned {role}")
    rs = json.loads(Path(runs).read_text())
    fix_keys = sorted({json.loads(l)["key"] for l in open(fix) if json.loads(l)["label"] == "FIX"})
    if len(rs) != 3807 or len(fix_keys) != 823 or not set(fix_keys) <= {r["key"] for r in rs if r["arm"] == "sim-attack"}:
        raise SystemExit("not #36's 3,807 runs and fix939's 823 re-rendered ones")
    res = render.remote([{"arm": r["arm"], "key": r["key"], "filename": r["filename"]} for r in rs], fix_keys)
    out = {"plan": "PLAN.md (this folder)", "git": head, "reader": {"model": MODEL, "revision": REVISION, "cut": CUT},
           "inputs sha256": {"#36 runs.json": PIN["runs.json"], "fix939 per-run file": PIN["fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl"]}} | res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"olmo_render_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(dest)
