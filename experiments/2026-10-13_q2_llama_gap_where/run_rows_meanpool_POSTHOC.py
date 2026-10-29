"""POST HOC (0d, 10-13 about 11:10; go about 11:30 under Aswin's standing "<$5 just run"): does the output category that
carries Llama-3.3-70B's last-token gap also carry its mean-pooled gap? Step 1: per-output probe logits and category codes in
one Modal CPU container, about $0.06. Not part of RULE.md; RULE.md and the registered analysis are unchanged.

run_rows.py's job, restricted to Llama-3.3-70B and widened to two readouts and the arms avg_vs_last reads (bd4a077e):
- mean-pooled (`output_end_mean`, extraction 2026-10-12_la_output-end-mean_llama33-70b_l2_v1, fit run 2026-10-13_011737):
  real e2, as simulated (939) and fix939's re-rendered runs (823; avg_vs_last's repaired arm takes the other 116 as simulated);
- last-token (`output_end`, part 1's extraction, fit run 2026-10-12_143129): real e2 and as simulated.
Every kept row gets the logit of each nested model (f, v), v != f, of its run's test fold f at both layers 60 and 70 (never
row 2's layers), so the local side can read avg_vs_last's three configurations: L60, L70 and per fold (60, 70, 60, 60, 70).
Categories and Llama's R2J readout class come from run_rows.py's functions, copied unchanged (main checks the copy). No
transcript text leaves the container; a failed check crosses with its count-only message, any other error with its type,
file and line only. Gates as run_rows.py's: joins, kept rows, the meta's R2J token equal to positions()'s, no whitespace-only
or newline readout. The activations stay on their Volumes; only the per-row logits and codes come back.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_q2_llama_gap_where/run_rows_meanpool_POSTHOC.py --runs <#36 runs.json dir> --fix <fix939 per-run jsonl>
"""
import ast
import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REGISTERED = HERE / "run_rows.py"
APP = "probe-shift_aswin-surya_sub-q2-gapwhere-mp_rows_kill-2026-11-13"
LA = "probe-shift_aswin-surya_sub-la-activations"
L_ACTS, L_WORK = "probe-shift_aswin-surya_sub-q2-llama_acts_kill-2026-11-12", "probe-shift_aswin-surya_sub-q2-llama_work_kill-2026-11-12"
M_ACTS, M_WORK = "probe-shift_aswin-surya_sub-q2-mpllama_acts_kill-2026-11-12", "probe-shift_aswin-surya_sub-q2-mpllama_work_kill-2026-11-12"
WEIGHTS = "probe-shift_aswin-surya_sub-apollo_llama-cache"
DUMP = "2026-10-06_la_segment-end_qwen3-32b_v1"
FIX_DUMP, FIX_INPUTS, FIX_ARM = "2026-10-11_q3_fix939_qwen3-32b_l48-63_v1", "2026-10-11_q3_fix939_inputs_v1", "sim-attack-note-sr-both"
FIX_INPUTS_SHA = "57cfbfbf1e4571690769605d2093d73edc93b958e06fed5e8efe8fa88587530d"
L_EXTRACT, M_EXTRACT = "2026-10-12_la_output-end-r2j_llama33-70b_alllayers_v1", "2026-10-12_la_output-end-mean_llama33-70b_l2_v1"
MODEL, REVISION, CTX = "meta-llama/Llama-3.3-70B-Instruct", "6f6073b423013f6a7d4d9f39144961bfbfbc386b", 131072
DATE = "26 Jul 2024"
HDR, END = "<|start_header_id|>ipython<|end_header_id|>\n\n", "<|eot_id|>"
LAYERS, ROW2 = (60, 70), (20, 30, 40, 50, 79)
assert not set(LAYERS) & set(ROW2)
READOUTS = {  # readout -> (probe dir, meta kind, probe key prefix, {source: (extraction dir, label)})
    "last-token": ("/lwork/runs/2026-10-12_143129", "output_end", "output_end",
                   {"e2": (f"/lacts/{L_EXTRACT}", "real-attack-e2"), "sim": (f"/lacts/{L_EXTRACT}", "sim-attack")}),
    "mean-pooled": ("/mwork/runs/2026-10-13_011737", "output_end_mean", "output_end_mean",
                    {"e2": (f"/macts/{M_EXTRACT}", "real-attack-e2"), "sim": (f"/macts/{M_EXTRACT}", "sim-attack"),
                     "fix": (f"/macts/{M_EXTRACT}", "sim-attack-fix939")})}
FAMILY = ("bash", "view", "create", "str_replace", "insert", "editor-other", "other")
EDITORS, CMDS, LENGTH = ("text_editor", "str_replace_based_edit_tool"), ("view", "create", "str_replace", "insert"), (100, 1000, 5000)
ENDING = ("empty", "newline", "whitespace", "alphanumeric", "punctuation")
POSITION = ("early", "middle", "late")
RCLASS = ("whitespace-only", "newline", "letters", "digits", "punctuation")
COPIED = ("Fail", "prep", "classes", "positions", "qsum", "categories", "_init", "readouts", "Gate", "check")  # from run_rows.py, byte for byte

app = modal.App(APP)
vols = {"/la": modal.Volume.from_name(LA, version=2).read_only(), "/lacts": modal.Volume.from_name(L_ACTS, version=2).read_only(),
        "/lwork": modal.Volume.from_name(L_WORK, version=2).read_only(), "/macts": modal.Volume.from_name(M_ACTS, version=2).read_only(),
        "/mwork": modal.Volume.from_name(M_WORK, version=2).read_only(), "/llama": modal.Volume.from_name(WEIGHTS).read_only()}
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub==0.35.3", "numpy==2.3.3")
         .env({"HF_HUB_CACHE": "/llama", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}))


# ---------- copied unchanged from ../2026-10-12_q2_llama_reader/run_extract_llama.py (main checks it) ----------
class Fail(Exception):
    def __init__(self, stage):
        super().__init__(stage); self.stage = stage


def prep(msgs):
    """Llama's template, per PLAN.md: string content; tool-call arguments as a dict; each assistant turn split into its
    text (if any, after trimming) and one message per tool call. Tool outputs and everything else in their order."""
    out = []
    for m in msgs:
        m = dict(m); c = m.get("content")
        m["content"] = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = [{**tc, "function": {**(tc.get("function") or {}), "arguments": (lambda a: json.loads(a) if isinstance(a, str) else (a or {}))(
                (tc.get("function") or {}).get("arguments"))}} for tc in m["tool_calls"]]
            if m["content"].strip():
                out.append({"role": "assistant", "content": m["content"]})
            out += [{"role": "assistant", "content": "", "tool_calls": [tc]} for tc in calls]
        else:
            m.pop("tool_calls", None)
            out.append(m)
    return out


def classes(e: str) -> tuple[list[str], list[int]]:
    """Per character of a JSON string body (no quotes): the class of the character it encodes ("nl" for a newline,
    "ws" for other whitespace, "c" for content; an escape sequence's characters all take its decoded character's class),
    and the decoded index each escaped character belongs to."""
    esc = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}
    cls, dec, i, k = [], [], 0, 0
    while i < len(e):
        if e[i] == "\\":
            ch, L = (chr(int(e[i + 2:i + 6], 16)), 6) if e[i + 1] == "u" else (esc[e[i + 1]], 2)
        else:
            ch, L = e[i], 1
        c = "nl" if ch == "\n" else "ws" if ch.strip() == "" else "c"
        cls += [c] * L; dec += [k] * L; i += L; k += 1
    return cls, dec


def positions(tok, msgs, q):
    """One staged transcript under Llama's template. q: Qwen3-32B's output_end positions, {"step,action": [kept, tool]}.
    Returns ids, the meta positions (every call's call_end, not kept; every output's output_end with R2J as its token,
    kept under C2 and below CTX), and per output (qwen kept, R0, R1, R2J as (index, classes), raw content, the
    rendered span (a, b) of the escaped body, its classes) for the render and diff modes. Raises Fail(stage)."""
    import bisect
    staged = [dict(m) for m in msgs]
    tools = [m for m in staged if m["role"] == "tool"]
    for m in tools:
        c = m.get("content")
        m["content"] = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
    text = tok.apply_chat_template(prep(msgs), tokenize=False, date_string=DATE)
    if text.count(HDR) != len(tools) or any(HDR in m["content"] or END in m["content"] for m in tools):
        raise Fail("a delimiter quoted in content")
    spans, calls, pos, step, action = [], [], 0, -1, 0
    for m in staged:
        if m["role"] == "assistant":
            step, action = step + 1, 0
            calls += [(step, i, (tc.get("function") or {}).get("name")) for i, tc in enumerate(m.get("tool_calls") or [])]
        elif m["role"] == "tool":
            a = text.index(HDR, pos) + len(HDR); enc = json.dumps(m["content"], ensure_ascii=False); b = a + len(enc)
            if text[a:b] != enc or not text.startswith(END, b):
                raise Fail("a tool output not rendered as its JSON string")
            spans.append((step, action, a + 1, b - 1, m["content"])); action += 1; pos = b
    enc_ = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc_["input_ids"]; starts, ends = [s for s, _ in enc_["offset_mapping"]], [e for _, e in enc_["offset_mapping"]]

    def inside(a, b):  # the tokens lying entirely inside [a, b)
        return bisect.bisect_left(starts, a), bisect.bisect_right(ends, b)
    out = [{"kind": "call_end", "step": s, "action": i, "tool": name, "token": None, "kept": False, "token_text": None} for s, i, name in calls]
    per = []
    for s, a_, a, b, content in spans:
        cls, dec = classes(text[a:b])
        tc = lambda t: cls[starts[t] - a:ends[t] - a]  # noqa: E731
        i0, i1 = inside(a, b)
        r0 = i1 - 1 if i1 > i0 else None
        last_c = max((j for j, c in enumerate(cls) if c == "c"), default=None)
        j0, j1 = inside(a, a + last_c + 1) if last_c is not None else (0, 0)
        r1 = j1 - 1 if j1 > j0 else None
        r2 = next((t for t in range(i1 - 1, i0 - 1, -1) if "nl" not in tc(t) and any(c != "ws" for c in tc(t))), None)
        qk = q.get(f"{s},{a_}")
        kept = bool(qk and qk[0]) and r2 is not None and r2 < CTX
        out.append({"kind": "output_end", "step": s, "action": a_, "tool": qk[1] if qk else None, "token": r2, "kept": kept, "token_text": None})
        per.append({"qk": qk, "rules": {r: (t, None if t is None else tc(t)) for r, t in (("R0", r0), ("R1", r1), ("R2J", r2))},
                    "content": content, "span": (a, b), "cls": cls, "dec": dec})
    return {"ids": ids, "n_tokens": len(ids), "positions": out, "outputs": per, "text": text, "starts": starts, "ends": ends}
# ---------- end of the copy ----------


def qsum(meta):  # Qwen3-32B's output_end positions: [kept, tool] by "step,action" (run_extract_llama.py's)
    return {f"{p['step']},{p['action']}": [bool(p["kept"]), p.get("tool")] for p in meta["positions"] if p["kind"] == "output_end"}


def categories(msgs):
    """Per tool output, by (step, action) as positions() numbers them: [family, length bin, ending, position, tool name];
    and how many calls' arguments weren't a JSON object (their family is editor-other or other)."""
    outs, step, action, calls, bad = [], -1, 0, [], 0
    for m in msgs:
        if m["role"] == "assistant":
            step, action, calls = step + 1, 0, m.get("tool_calls") or []
        elif m["role"] == "tool":
            fn = (calls[action].get("function") or {}) if action < len(calls) else {}
            name, args = fn.get("name"), fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = None
            if not isinstance(args, dict):
                bad += 1; args = {}
            cmd = args.get("command")
            fam = 0 if name == "bash" else (1 + CMDS.index(cmd) if isinstance(cmd, str) and cmd in CMDS else 5) if name in EDITORS else 6
            c = m.get("content")
            text = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
            ch = text[-1] if text else ""
            end = 0 if not text else 1 if ch in "\n\r" else 2 if ch.isspace() else 3 if ch.isalnum() else 4
            outs.append([step, action, fam, sum(len(text) >= t for t in LENGTH), end, name]); action += 1
    n = len(outs)
    return {(o[0], o[1]): [o[2], o[3], o[4], 0 if 3 * k < n else 1 if 3 * k < 2 * n else 2, o[5]] for k, o in enumerate(outs)}, n, bad


def _init():
    from transformers import AutoTokenizer
    global TOK
    TOK = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def readouts(item):
    """One Llama transcript: per output (step, action, R2J token, kept, readout class, merges whitespace, body decodes to
    the full content). A render failure returns its stage only."""
    j, msgs, q = item
    try:
        r = positions(TOK, msgs, q)
    except Fail as e:
        return j, e.stage
    res = []
    for p, o in zip((p for p in r["positions"] if p["kind"] == "output_end"), r["outputs"]):
        full = (o["dec"][-1] + 1 if o["dec"] else 0) == len(o["content"])
        t = p["token"]
        if t is None:
            res.append((p["step"], p["action"], None, p["kept"], -1, False, full)); continue
        a = o["span"][0]; lo, hi = r["starts"][t] - a, r["ends"][t] - a
        cl, chars = o["cls"][lo:hi], [o["content"][d] for d in sorted(set(o["dec"][lo:hi]))]
        rc = (0 if all(c != "c" for c in cl) else 1 if "nl" in cl else 2 if any(x.isalpha() for x in chars)
              else 3 if all(x.isdigit() for x in chars if not x.isspace()) else 4)
        res.append((p["step"], p["action"], t, p["kept"], rc, "ws" in cl and "c" in cl, full))
    return j, res


class Gate(Exception):
    """Our own checks only: the message holds counts and names, never staged text."""


def check(ok, msg):
    if not ok:
        raise Gate(msg)


@app.function(image=image, volumes=vols, cpu=8, memory=32768, timeout=3600, retries=0)
def rows(units: list[dict]) -> dict:
    try:
        return _rows(units)
    except Gate as e:
        raise RuntimeError(f"check failed: {e}") from None
    except Exception as e:  # any other error crosses as its type, file and line only (its message may quote staged text)
        import traceback
        f = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"rows failed: {type(e).__name__} at {Path(f.filename).name}:{f.lineno}") from None


def _rows(units):
    import time
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import Pool

    import numpy as np
    t0, log = time.time(), lambda s: print(f"{time.time() - t0:7.1f}s {s}", flush=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    need, msgs, rec = {(u["src"], u["key"]) for u in units}, {}, {"inputs sha256": {}}

    def take(src, key, m):
        if (src, key) in need:
            check((src, key) not in msgs, f"a staged {src} transcript appears twice")
            msgs[(src, key)] = m
    for p in sorted((Path("/la") / DUMP / "inputs").glob("shard_*.jsonl")):
        data = p.read_bytes(); rec["inputs sha256"][f"{DUMP}/inputs/{p.name}"] = hashlib.sha256(data).hexdigest()
        for line in data.splitlines():
            it = json.loads(line)
            if it["arm"] in ("real-attack-e2", "sim-attack"):
                take("e2" if it["arm"] == "real-attack-e2" else "sim", it["key"], it["messages"])
        del data
    fx = sorted((Path("/la") / FIX_INPUTS / "inputs").glob("shard_*.jsonl"))
    check([p.name for p in fx] == ["shard_0.jsonl"] and sha(fx[0]) == FIX_INPUTS_SHA, "fix939's inputs are not the pinned ones")
    for it in map(json.loads, fx[0].read_text().splitlines()):
        take("fix", it["key"], it["messages"])
    check(set(msgs) == need, f"{len(need - set(msgs))} needed staged transcripts not found")
    rec["inputs sha256"][f"{FIX_INPUTS}/inputs/shard_0.jsonl"] = FIX_INPUTS_SHA
    cats = {sk: categories(m) for sk, m in msgs.items()}
    rec["calls whose arguments aren't a JSON object"] = sum(c[2] for c in cats.values())
    log(f"staged: {len(msgs)} transcripts")
    qdir = {"e2": f"{DUMP}/acts/real-attack-e2", "sim": f"{DUMP}/acts/sim-attack", "fix": f"{FIX_DUMP}/meta/{FIX_ARM}"}
    items = sorted(((j, msgs[(u["src"], u["key"])], qsum(json.loads((Path("/la") / qdir[u["src"]] / f"{u['filename']}.json").read_text())))
                    for j, u in enumerate(units)), key=lambda it: -sum(len(json.dumps(m)) for m in it[1]))  # longest first
    _init()  # fails here, once, rather than in every pool worker
    with Pool(8, initializer=_init) as pool:
        ro_all = dict(pool.imap_unordered(readouts, items, chunksize=1))
    items.clear(); msgs.clear()
    fails = Counter(v for v in ro_all.values() if isinstance(v, str))
    check(not fails, f"Llama render failures: {dict(fails)}")
    log("Llama readouts located")

    out, gate = {}, Counter()
    for readout, (pdir, kind, pkey, srcs) in READOUTS.items():
        probes = {}
        for l in LAYERS:
            pf = Path(pdir) / f"probes_L{l}.npz"
            rec["inputs sha256"][f"{readout} {pf.name}"] = sha(pf)
            with np.load(pf) as z:
                probes[l] = {k: z[k] for k in z.files if k.startswith(f"{pkey}__task_nested_")}
        groups, R, files, n_rows = {}, [], {}, []
        mine = [(j, u) for j, u in enumerate(units) if u["src"] in srcs]
        for j, u in mine:
            d, label = srcs[u["src"]]
            m = json.loads((Path(d) / "meta" / label / f"{u['filename']}.json").read_text())
            check(m["key"] == u["key"], f"{readout}: a meta of another run")
            kept = [p for p in m["positions"] if p["kind"] == kind and p["kept"]]
            sa = [(p["step"], p["action"]) for p in kept]; rws = [p["row"] for p in kept]
            c, n_out, _ = cats[(u["src"], u["key"])]
            ro = {(s, a): (t, k, rc, mw, full) for s, a, t, k, rc, mw, full in ro_all[j]}
            gate[f"{readout}: kept outputs repeated"] += len(sa) - len(set(sa))
            gate[f"{readout}: kept outputs without a staged output"] += sum(x not in c for x in sa)
            gate[f"{readout}: kept outputs without a row"] += sum(r is None for r in rws)
            gate[f"{readout}: kept rows not 0..n_rows-1"] += rws != list(range(m["n_rows"]))
            gate[f"{readout}: outputs differ from positions()'s"] += set(ro) != {(p["step"], p["action"]) for p in m["positions"] if p["kind"] == kind}
            gate[f"{readout}: kept outputs differ from positions()'s"] += {x for x, v in ro.items() if v[1]} != set(sa)
            gate[f"{readout}: kept readout tokens differ from positions()'s"] += sum(ro.get(x, (None,))[0] != p["token"] for x, p in zip(sa, kept))
            gate[f"{readout}: outputs not counted by categories()"] += len(ro) != n_out
            gate[f"{readout}: outputs whose body doesn't decode to the full content"] += sum(not v[4] for v in ro.values())
            for l in LAYERS:
                path = str(Path(d) / "by_layer" / f"L{l}" / f"{m['chunk']}.npy")
                groups.setdefault((path, l), []).append((j, u["fold"], [m["row_start"] + r for r in rws], len(R)))
            fid = files.setdefault(str(Path(d) / "by_layer" / f"L*/{m['chunk']}.npy"), len(files))
            for x, r in zip(sa, rws):
                cc = c.get(x, [-1, -1, -1, -1, None])
                R.append([j, x[0], x[1], fid, m["row_start"] + r, *cc[:4], *ro.get(x, (None, None, -1, False))[2:4]])
            n_rows.append(len(sa))

        def score(key):
            path, l = key
            X = np.load(path)  # whole file: row-indexed mmap is slow on a Volume
            res = []
            for j, f, rr, r0 in groups[key]:
                check(not rr or (min(rr) >= 0 and max(rr) < len(X)), f"{readout}: rows outside an activation file")
                x = X[rr].astype(np.float32)
                s = np.full((len(rr), 5), np.nan, np.float32)
                for v in range(5):
                    if v != f:
                        k = f"{pkey}__task_nested_{f}_{v}__"; z = probes[l]
                        s[:, v] = ((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])
                res.append((l, r0, s))
            return res
        logit = {l: np.full((len(R), 5), np.nan, np.float32) for l in LAYERS}
        with ThreadPoolExecutor(8) as ex:
            for res in ex.map(score, list(groups)):
                for l, r0, s in res:
                    logit[l][r0:r0 + len(s)] = s
        A = np.array(R, dtype=np.int64)
        check(len(A) and (A[:, 5:] >= 0).all(), f"{readout}: a row without categories or readout class")
        own = np.arange(5)[None, :] == np.array([units[j]["fold"] for j in A[:, 0]])[:, None]
        check(all(np.isfinite(logit[l][~own]).all() for l in LAYERS), f"{readout}: a non-finite logit at a model that reads the row")
        rc = A[:, 9]
        gate[f"{readout}: kept readouts whitespace-only or holding a newline"] += int(((rc == 0) | (rc == 1)).sum())
        names = ["unit", "step", "action", "activation file", "activation row", "family", "length", "ending", "position", "readout class", "merges whitespace"]
        out |= {f"{readout}__{n}": A[:, i] for i, n in enumerate(names)} | {f"{readout}__logit L{l}": logit[l] for l in LAYERS}
        rec[f"{readout}: activation files (arrays' 'activation file' indexes this)"] = list(files)
        rec[f"{readout}: units, rows by source"] = {s: [sum(u["src"] == s for _, u in mine), sum(n for (_, u), n in zip(mine, n_rows) if u["src"] == s)] for s in srcs}
        log(f"{readout}: {len(R)} rows scored at L60 and L70")
    bad = {k: int(v) for k, v in gate.items() if v}
    check(not bad, f"gates failed: {bad}")
    rec["gates: every count 0"] = sorted(gate)
    buf = io.BytesIO(); np.savez(buf, **out)
    return {"npz": buf.getvalue(), "record": rec}


@app.local_entrypoint()
def main(runs: str, fix: str):
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    for f in (Path(__file__).resolve(), REGISTERED):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()

    def defs(p):
        src = Path(p).read_text(); tree = ast.parse(src)
        return {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in COPIED}
    if defs(__file__) != defs(REGISTERED) or len(defs(__file__)) != len(COPIED):
        raise SystemExit("the copied functions differ from run_rows.py's")
    runs_p, fix_p = Path(runs) / "runs.json", Path(fix)
    if sha(runs_p) != "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96" or sha(fix_p) != "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd":
        raise SystemExit("not #36's runs.json or fix939's per-run file")
    rr = json.loads(runs_p.read_text())
    fk = {json.loads(l)["key"] for l in fix_p.open() if json.loads(l)["label"] == "FIX"}
    units = [{"i": i, "key": x["key"], "filename": x["filename"], "fold": int(x["fold"]), "src": "e2" if x["arm"] == "real-attack-e2" else "sim"}
             for i, x in enumerate(rr) if x["arm"] in ("real-attack-e2", "sim-attack")]
    units += [{"i": i, "key": x["key"], "filename": x["filename"], "fold": int(x["fold"]), "src": "fix"} for i, x in enumerate(rr) if x["arm"] == "sim-attack" and x["key"] in fk]
    if [sum(u["src"] == s for u in units) for s in ("e2", "sim", "fix")] != [939, 939, 823]:
        raise SystemExit("not 939 real e2, 939 as simulated and 823 re-rendered runs")
    res = rows.remote(units)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    npz = HERE / "results" / f"rows_meanpool_POSTHOC_{stamp}.npz"
    with open(npz, "xb") as fh:
        fh.write(res["npz"])
    rec = {"POST HOC": True, "question": "0d, 10-13 about 11:10", "git": head, "app": APP, "layers": LAYERS,
           "readouts": {k: {"probe run": v[0], "meta kind": v[1], "sources": v[3]} for k, v in READOUTS.items()},
           "codes": {"family": FAMILY, "length (characters, lower bounds after 0)": LENGTH, "ending": ENDING, "position": POSITION, "readout class": RCLASS},
           "inputs sha256": {"runs.json": sha(runs_p), fix_p.name: sha(fix_p)}, "tokenizer": f"{MODEL} @ {REVISION}",
           "units (runs.json index, source, fold; arrays' unit indexes this)": {"runs.json index": [u["i"] for u in units], "source": [u["src"] for u in units], "fold": [u["fold"] for u in units]},
           "container": res["record"], "rows file": npz.name, "rows sha256": sha(npz)}
    with open(dest := HERE / "results" / f"rows_meanpool_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    shown = rec | {"units (runs.json index, source, fold; arrays' unit indexes this)": "in the file",
                   "container": {k: v for k, v in rec["container"].items() if "activation files" not in k}}  # file stems are run keys: not printed
    print(json.dumps(shown, indent=1))
    print(dest)
