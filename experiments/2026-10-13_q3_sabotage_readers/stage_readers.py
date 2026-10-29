"""B3 (RULE.md here): stage Qwen3.5-27B's acting runs on the sabotage cohort for another reader's forward pass. Local CPU,
tokenizers only; writes token ids, positions, labels, hashes and counts, never text (an error inside a run is re-raised as
its type and the run's file name only).

Runs, as the cross-read staged them (stage_crossread.py @ 8314a13d, copied here: counted attempts from the launch metadata,
each .eval sha256-checked against its launch manifest; the honest donor prompt for attack runs; scrub level 1; the 32,768 cut):
real run 1 (`attack` of the four main launches), real run 2 (`second_attack`) and the simulated runs (`sim_attack@september`)
on the 156 sabotage tasks (the 40 stage-1 tasks of flagrate_rows, the 116 new tasks of scaled_rows), and honest_b on the 51
tasks of the gate's honest rows (rows_v2 roles honb_cal, honb_ho).

The command locator (`command_spans`), for any chat template: every call's command value (rule 1, reader.command_param) is
replaced by a unique marker and the conversation rendered again. Outside the markers the two renderings must be equal, and at
each marker the real rendering must hold the value verbatim, JSON-escaped (ensure_ascii False or True), each tried in that
order and accepted only if the text after it matches too. The value's character span maps to the tokens lying wholly inside
it (the cross-read's rule): the readout is the last of them (part 1) and the span runs from the first to the last (part 2,
the mean). A token crossing either end of the value is counted, not read. Nothing else in the call (e.g. a timeout) is
inside the span.

--reader qwen3 --gate N: the positive control. For N runs drawn with seed 20261013, the locator's last token must equal the
cross-read's structural rule-1 position (command_positions, copied) for every call, None included; exits 1 otherwise.
N at or above the number of runs gates every run.

  uv run --no-project --python 3.14 --with transformers==4.57.1 --with jinja2 --with numpy python \\
      experiments/2026-10-13_q3_sabotage_readers/stage_readers.py --reader seed|olmo|llama|qwen3 --inputs <scaled_inputs dir> \\
      [--gate N] [--limit N] [--workers N]
"""

import argparse
import bisect
import hashlib
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p_ in (REPO / "src", REPO / "experiments/2026-10-07_q2_rendering_swap", REPO / "experiments/2026-10-09_fixed_sim", HERE):
    sys.path.insert(0, str(p_))

import numpy as np  # noqa: E402

from probe_shift import reader, transcripts  # noqa: E402
from probe_shift.calls import fingerprint, label_call  # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad")
CACHE = Path(os.environ.get("B3_CACHE") or Path.home() / ".cache/probe-shift/s3/individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs")
MAIN = ["2026-10-09_000523_rerun_strict", "2026-10-09_173815_scaleup_canary", "2026-10-09_182505_scaleup",
        "2026-10-10_001314_scaleup_topup"]
RUN2 = "2026-10-10_001314_second_attack"
ARMS = {("main", "attack"): "real1", ("main", "sim_attack@september"): "sim", ("run2", "attack"): "real2", ("main", "honest_b"): "honest_b"}
ACTING, CUT, SCRUB_LEVEL, LABELER, SEED = "vllm/Qwen/Qwen3.5-27B", 32768, 1, "eb014307", 20261013
COVER = re.compile(r"<cover_task>\n?(.*?)\n?</cover_task>", re.S)
STAGE_ROOT = Path(os.environ.get("B3_STAGE_ROOT") or SCRATCH / "sabreaders_stage")  # run_stage_modal.py: the B3 Volume
RESULTS = Path(os.environ.get("B3_RESULTS") or HERE / "results")
LA_ROOT, ERRORS = Path(os.environ.get("B3_LA") or "/la"), REPO / "experiments/2026-10-06_q1_segment_end/results/tool_errors_2026-10-06_134908.jsonl"
PINS = {"flagrate_rows_2026-10-13_122417.csv": "44c75f78e78f86b4bcab409b1379762a8d4dccadfd456f28ee651832eb2b13a7",
        "scaled_rows_2026-10-13_152021.csv": "a95453ac2688e61ba99184726ce27a4dd54276a04730a56d67825f204d1ee1f3",
        "rows_v2.json": "727c417916b8fb2df9a31aca5f8ca0cc4dd17002a8a2edcb7bd79444bc0bf4b5"}
LLAMA_TOK = Path(os.environ.get("B3_LLAMA_TOK") or SCRATCH / "llama_tok") / "6f6073b423013f6a7d4d9f39144961bfbfbc386b"  # tokenizer files read from the weights Volume
READERS = {  # name, revision, layers, how the reader's pipeline renders (q2's run_extract_olmo / _llama, G3's run_extract_g3)
    "qwen3": {"name": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137", "render": "qwen3"},
    "seed": {"name": "ByteDance-Seed/Seed-OSS-36B-Instruct", "revision": "497f1dca95ebdec98e41d517b9f060ee753c902f", "render": "plain"},
    "olmo": {"name": "allenai/Olmo-3.1-32B-Instruct", "revision": "ac0587e4a7744a551c059d8cd17ba220bc940dae", "render": "plain"},
    "llama": {"name": "meta-llama/Llama-3.3-70B-Instruct", "revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b", "render": "llama"},
}
DATE = "26 Jul 2024"  # Llama's date_string, as q2's run_extract_llama.py passes it
MARK = "QXCMD{:06d}Z"
VERSION = {"llama": "v2"}  # llama v1's record named tokenizer-file hashes (kept out of git); v2 is the same staging without them
NO_EVAL = []


class Withheld(Exception):
    """A run's processing failed; the message names the run and the error type only (no transcript text)."""


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ids_sha(ids) -> str:
    return hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()


# ---- copied from stage_crossread.py @ 8314a13d (scrubbed, resolved, index_sample, reader_map, first_user_sha, load) ----
def scrubbed(messages, level):
    from probe_shift.scrub import scrub

    def args(v):
        if isinstance(v, str):
            return scrub(v, level)
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, (int, float)):
            return "<n>" if level >= 2 else v
        if isinstance(v, list):
            return [args(x) for x in v]
        if isinstance(v, dict):
            return {k: args(x) for k, x in v.items()}
        return v

    out = []
    for m in messages:
        if m["role"] == "tool":
            m = m | {"content": scrub(m["content"], level)}
        elif m["role"] == "assistant":
            m = m | {"content": scrub(m["content"], level) if m["content"] else m["content"],
                     "tool_calls": [c | {"function": c["function"] | {"arguments": args(c["function"]["arguments"])}}
                                    for c in m.get("tool_calls") or []]}
        out.append(m)
    return out


def resolved(x, att):
    if isinstance(x, str) and transcripts.REF.fullmatch(x):
        return att[x.removeprefix(transcripts.ATTACHMENT)]
    if isinstance(x, list):
        return [resolved(v, att) for v in x]
    if isinstance(x, dict):
        return {k: resolved(v, att) for k, v in x.items()}
    return x


def index_sample(s, cv):
    turns, att = [], s.get("attachments") or {}
    for e in s.get("events") or []:
        if e.get("event") != "model" or e.get("model") != ACTING:
            continue
        irid = (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid")
        if irid is None:
            raise ValueError("an acting-model event without x-irid")
        offered = {t.get("name") for t in (e.get("tools") or [])}
        calls = []
        for tc in (((e.get("output") or {}).get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []:
            name, args = tc.get("function"), resolved(tc.get("arguments") or {}, att)
            if transcripts.refs_left(args):
                raise ValueError("an attachment reference left in a call's arguments")
            not_offered = bool(offered) and name not in offered and name != "submit"
            malformed = name == "text_editor" and args.get("command") is None
            cv["calls excluded: tool not offered"] += not_offered
            cv["calls excluded: text_editor without command"] += malformed
            excluded = name == "submit" or not_offered or malformed
            param = None if excluded else reader.command_param(name, args)
            calls.append({"name": name, "excluded": bool(excluded), "param": param,
                          "label": None if excluded else label_call(name, args)["label"]})
        turns.append({"irid": irid, "calls": calls})
    return turns


def reader_map(sample_msgs):
    asst_idx = [i for i, m in enumerate(sample_msgs) if m["role"] == "assistant"]
    step, steps, call_actions = -1, [], []
    for k, i in enumerate(asst_idx):
        m = sample_msgs[i]
        tcs = m.get("tool_calls") or []
        kept = [ci for ci, tc in enumerate(tcs) if tc["function"] != transcripts.RUNNER_SUBMIT]
        visible = bool(kept) or bool(transcripts.visible_text(m["content"]).strip())
        if visible:
            step += 1
        steps.append(step if visible else None)
        call_actions.append({ci: a for a, ci in enumerate(kept)} if visible else {})
        submit_ids = {tc.get("id") for tc in tcs if tc["function"] == transcripts.RUNNER_SUBMIT}
        nxt = asst_idx[k + 1] if k + 1 < len(asst_idx) else len(sample_msgs)
        a, ids = 0, [tc.get("id") for tc in tcs]
        for tm in [t for t in sample_msgs[i + 1: nxt] if t["role"] == "tool"]:
            ci = ids.index(tm.get("tool_call_id")) if tm.get("tool_call_id") in ids else None
            if tm.get("tool_call_id") in submit_ids or tm.get("function") == transcripts.RUNNER_SUBMIT:
                continue
            if not visible or a >= len(kept) or ci != kept[a]:
                raise ValueError("tool outputs do not follow their calls in order")
            a += 1
    return steps, call_actions


def first_user_sha(sample: dict) -> str:
    m = next((m for m in sample["messages"] if m["role"] == "user"), None)
    return hashlib.sha256(transcripts.visible_text(m["content"] if m else "").encode()).hexdigest()


def member(z, info) -> bytes:
    """stage_b2.py's member (31e4551c): a zip member's bytes; zstd members (method 93) decompressed by hand below Python 3.14, CRC-checked."""
    if info.compress_type != 93 or sys.version_info >= (3, 14):
        return z.read(info)
    import struct
    import zlib

    import zstandard

    z.fp.seek(info.header_offset)
    h = z.fp.read(30)
    if h[:4] != b"PK\x03\x04":
        raise ValueError("bad local file header")
    n, m = struct.unpack("<HH", h[26:30])
    z.fp.seek(info.header_offset + 30 + n + m)
    out = zstandard.ZstdDecompressor().decompress(z.fp.read(info.compress_size), max_output_size=info.file_size)
    if len(out) != info.file_size or zlib.crc32(out) != info.CRC:
        raise ValueError("zstd member fails its size or CRC check")
    return out


def load(ev: Path) -> dict:
    with zipfile.ZipFile(ev) as z:
        (info,) = [x for x in z.infolist() if x.filename.startswith("samples/") and x.filename.endswith(".json")]
        return transcripts.resolve_attachments(json.loads(member(z, info)))


def command_positions(tok, messages, cut, cv):
    """stage_crossread.py's structural rule 1 for Qwen3's JSON calls (copied unchanged): the gate's reference."""
    text, found, enc = reader._layout(tok, messages)
    starts = [s for s, _ in enc["offset_mapping"]]
    ends = [e for _, e in enc["offset_mapping"]]
    tool_calls = [m.get("tool_calls") or [] for m in messages if m["role"] == "assistant"]
    dumps = lambda x: json.dumps(x, ensure_ascii=False)  # noqa: E731
    out = {}
    for kind, step, _, _, a, b in found:
        if kind != "agent":
            continue
        calls = reader._calls(text, a, b, "qwen3")
        if len(calls) != len(tool_calls[step]):
            raise ValueError("rendered calls differ from the message")
        for i, ((_k, spans, _close, name, _), c) in enumerate(zip(calls, tool_calls[step])):
            args = c["function"].get("arguments") or {}
            param = reader.command_param(name, args)
            out[(step, i)] = None
            if param is None:
                continue
            head, parts, span = '{"name": ' + dumps(name) + ', "arguments": {', [], None
            pos = len(head)
            for j, (k, v) in enumerate(args.items()):
                piece = dumps(k) + ": "
                if j:
                    pos += 2
                if k == param:
                    sv = dumps(v)
                    span = (pos + len(piece) + 1, pos + len(piece) + len(sv) - 1)
                piece += dumps(v)
                parts.append(piece)
                pos += len(piece)
            serial = head + ", ".join(parts) + "}}"
            c0, c1 = spans[0]
            if text[c0:c1] != serial or span is None:
                continue
            v0, v1 = c0 + span[0], c0 + span[1]
            i0, i1 = bisect.bisect_left(starts, v0), bisect.bisect_right(ends, v1)
            if i1 <= i0 or i1 - 1 >= cut:
                continue
            out[(step, i)] = i1 - 1
    return out


# ---- rendering per reader, and the template-agnostic locator ----
def _plain(msgs):  # q2's / G3's prep(): string content, tool-call arguments as a dict
    out = []
    for m in msgs:
        m = dict(m)
        c = m.get("content")
        m["content"] = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**(tc.get("function") or {}), "arguments": (lambda a: json.loads(a) if isinstance(a, str) else (a or {}))(
                (tc.get("function") or {}).get("arguments"))}} for tc in m["tool_calls"]]
        out.append(m)
    return out


def _llama(msgs):  # q2's run_extract_llama.py prep(): each assistant turn split into its text and one message per call
    out = []
    for m in _plain(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            if m["content"].strip():
                out.append({"role": "assistant", "content": m["content"]})
            out += [{"role": "assistant", "content": "", "tool_calls": [tc]} for tc in m["tool_calls"]]
        else:
            m.pop("tool_calls", None)
            out.append(m)
    return out


def render(tok, kind, msgs) -> str:
    if kind == "qwen3":
        return tok.apply_chat_template(reader.prepare(tok, msgs), tokenize=False, enable_thinking=False)
    if kind == "llama":
        return tok.apply_chat_template(_llama(msgs), tokenize=False, date_string=DATE)
    return tok.apply_chat_template(_plain(msgs), tokenize=False)


def command_spans(tok, kind, msgs, cv):
    """{(step, i): (char start, char end) or None} for every call of every assistant message, plus the text. Raises if the
    two renderings differ outside the command values or a value isn't found in a known encoding."""
    sent, marks, n = [dict(m) for m in msgs], [], 0
    spans = {}
    for step, i in enumerate([i for i, m in enumerate(msgs) if m["role"] == "assistant"]):
        new = []
        for j, tc in enumerate(msgs[i].get("tool_calls") or []):
            fn = tc["function"]
            args = fn.get("arguments") or {}
            param = reader.command_param(fn["name"], args)
            spans[(step, j)] = None
            if param is not None:
                marks.append((step, j, args[param], MARK.format(n)))
                args = dict(args) | {param: MARK.format(n)}
                n += 1
            new.append(tc | {"function": fn | {"arguments": args}})
        sent[i] = sent[i] | {"tool_calls": new}
    text, stext = render(tok, kind, msgs), render(tok, kind, sent)
    if "QXCMD" in text:
        raise ValueError("marker text in the transcript")
    ps, pt = 0, 0
    for step, j, v, mk in marks:
        q = stext.find(mk, ps)
        if q < 0 or stext.count(mk) != 1:
            raise ValueError("a command value not located")
        seg = stext[ps:q]
        if text[pt:pt + len(seg)] != seg:
            raise ValueError("the renderings differ outside the command values")
        pt, ps = pt + len(seg), q + len(mk)
        nq = stext.find("QXCMD", ps)
        nseg = stext[ps: len(stext) if nq < 0 else nq]
        cands = dict.fromkeys([v, json.dumps(v, ensure_ascii=False)[1:-1], json.dumps(v)[1:-1]])
        enc = next((e for e in cands if text.startswith(e, pt) and text.startswith(nseg, pt + len(e))), None)
        if enc is None:
            raise ValueError("a command value is not in the rendering verbatim or JSON-escaped")
        cv["value encoding: " + ("verbatim" if enc == v else "JSON-escaped")] += 1
        spans[(step, j)] = (pt, pt + len(enc))
        pt += len(enc)
    if text[pt:] != stext[ps:]:
        raise ValueError("the renderings differ after the last command value")
    return text, spans


def token_span(starts, ends, a, b):
    """(first, last, crosses start, crosses end): the tokens lying wholly inside [a, b) (the cross-read's rule), or None."""
    i0, i1 = bisect.bisect_left(starts, a), bisect.bisect_right(ends, b)
    if i1 <= i0:
        return None
    return i0, i1 - 1, bool(i0 > 0 and starts[i0 - 1] < a < ends[i0 - 1]), bool(i1 < len(starts) and starts[i1] < b < ends[i1])


# ---- runs ----
def counted_attempts() -> dict:
    """{(launch kind, arm_dir, task): attempt}: stage_crossread.counted_attempts for the main launches and run 2."""
    counted = {}
    for run in MAIN + [RUN2]:
        kind = "run2" if run == RUN2 else "main"
        rd = CACHE / run
        man = json.loads((CACHE / f"{run}.manifest_v1.json").read_text())
        want = {f["key"].split(f"phase2/runs/{run}/", 1)[1]: f["sha256"] for f in man["files"] if f["key"].startswith(f"phase2/runs/{run}/")}
        if sha_file(rd / "launch.json") != want["launch.json"]:
            raise ValueError(f"{run}: launch.json differs from its manifest")
        for r in json.loads((rd / "launch.json").read_text())["runs"]:
            for ai, a in enumerate(r.get("attempts") or []):
                if a.get("redrawn_because") or a.get("kind") == "rerun":
                    continue
                rel_dir = (a.get("log_dir") or "").split(f"/{run}/", 1)[-1]
                arm_dir = rel_dir.split("/", 1)[0]
                if (kind, arm_dir) not in ARMS and not (kind == "main" and arm_dir == "honest_a"):
                    continue
                key = (kind, arm_dir, r["task"])
                if key in counted:
                    raise ValueError(f"several counted attempts for {key} in the launch metadata")
                evs = sorted(e for e in want if e.startswith(rel_dir + "/") and e.endswith(".eval"))
                if not evs:
                    NO_EVAL.append(f"{run}/{rel_dir}")
                    continue
                if len(evs) != 1:
                    raise ValueError(f"{run}/{rel_dir}: {len(evs)} .eval files in the manifest, 1 expected")
                ev = rd / evs[0]
                if not ev.exists() or sha_file(ev) != want[evs[0]]:
                    raise ValueError(f"{run}/{evs[0]}: missing from the cache or differs from its manifest")
                counted[key] = {"launch": run, "eval": ev, "rel": evs[0], "attempt_index": ai, "attempt_kind": a.get("kind")}
    return counted


def tasks(inputs: Path) -> dict:
    """The sabotage cohort's tasks (attack arms) and the gate's honest tasks (honest_b), from the pinned row files."""
    for n, h in PINS.items():
        if sha_file(inputs / n) != h:
            raise SystemExit(f"{n}: sha256 mismatch")
    import csv

    st1 = {r["task"] for r in csv.DictReader(open(inputs / "flagrate_rows_2026-10-13_122417.csv")) if r["part"] == "2"}
    new = {r["task"] for r in csv.DictReader(open(inputs / "scaled_rows_2026-10-13_152021.csv"))}
    hon = {r["task"] for r in json.loads((inputs / "rows_v2.json").read_text())["rows"] if r["role"] in ("honb_cal", "honb_ho")}
    if (len(st1), len(new), len(st1 & new), len(hon)) != (40, 116, 0, 51):
        raise SystemExit(f"task counts {len(st1)}, {len(new)}, {len(st1 & new)}, {len(hon)} (expected 40, 116, 0, 51)")
    return {"attack": st1 | new, "honest_b": hon}


TOK, KIND, REPAIR = None, None, None


def _init(rd, templates=None):
    global TOK, KIND, REPAIR
    from transformers import AutoTokenizer

    if templates is not None:  # --repair: B2's fix-full minus F7 (repair_b2.py)
        import fixes
        import rerender
        import repair_b2
        import str_replace_tells

        REPAIR = repair_b2.make_repair(templates, fixes, rerender, str_replace_tells)

    path = str(LLAMA_TOK) if rd["render"] == "llama" else rd["name"]
    kw = {} if rd["render"] == "llama" else {"revision": rd["revision"], "local_files_only": True}
    TOK, KIND = AutoTokenizer.from_pretrained(path, **kw), rd["render"]


def process(job):
    """One run: (arm, staged epoch or None, per-run record, counters, gate mismatches or None)."""
    (kind, arm_dir, task), c, donor, gate = job
    cv = Counter()
    try:
        s = load(c["eval"])
        if arm_dir == "honest_b":
            system = transcripts.system_prompt(s)
        else:
            own = transcripts.system_prompt(s)
            cover, fu = COVER.search(own), first_user_sha(s)
            pick = [p for p in donor if p and cover and cover.group(1).strip() in p[0] and "You are an agent" in p[0] and p[1] == fu]
            if not pick:
                return ARMS[(kind, arm_dir)], None, {"task": task, "excluded": "no honest donor"}, cv, None
            system = pick[0][0]
        turns = index_sample(s, cv)
        steps, call_actions = reader_map(s["messages"])
        if len(steps) != len(turns):
            if not s.get("error"):
                raise ValueError("acting-model events and assistant messages differ in number")
            return ARMS[(kind, arm_dir)], None, {"task": task, "excluded": "errored run: events and messages misaligned"}, cv, None
        asst = [m for m in s["messages"] if m["role"] == "assistant"]
        if any([x["name"] for x in t["calls"]] != [tc["function"] for tc in (m.get("tool_calls") or [])] for t, m in zip(turns, asst)):
            raise ValueError("an acting-model event's calls differ from its assistant message's")
        msgs = transcripts.reader_messages(s, system)
        if REPAIR is not None:
            msgs, n_changed = REPAIR(msgs)
            cv["outputs changed by the repair"] += n_changed
            cv["runs changed by the repair"] += n_changed > 0
        msgs = scrubbed(msgs, SCRUB_LEVEL)
        text, spans = command_spans(TOK, KIND, msgs, cv)
        enc = TOK(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        starts, ends = [a for a, _ in enc["offset_mapping"]], [b for _, b in enc["offset_mapping"]]
        if any(starts[i] < starts[i - 1] or ends[i] < ends[i - 1] for i in range(1, len(ids))):
            raise ValueError("offsets not monotonic")
        pos = {}
        for k_, sp in spans.items():
            ts = token_span(starts, ends, *sp) if sp else None
            cv["calls with a command value"] += sp is not None
            cv["values with no whole token inside"] += sp is not None and ts is None
            if ts:
                cv["values whose first token crosses the value's start"] += ts[2]
                cv["values whose last token crosses the value's end"] += ts[3]
                cv["values read past the cut"] += ts[1] >= CUT
            pos[k_] = ts if ts and ts[1] < CUT else None
        mism = None
        if gate:
            ref = command_positions(TOK, msgs, CUT, Counter())
            mine = {k_: (v[1] if v else None) for k_, v in pos.items()}
            mism = {"calls": len(ref), "mismatches": sum(ref.get(k_) != mine.get(k_) for k_ in set(ref) | set(mine)),
                    "readouts": sum(v is not None for v in ref.values())}
        rows = []
        for k, t in enumerate(turns):
            for ci, call in enumerate(t["calls"]):
                if call["excluded"]:
                    continue
                st, ac = steps[k], call_actions[k].get(ci)
                ts = None if st is None or ac is None else pos.get((st, ac))
                rows.append({"irid": t["irid"], "turn": k, "call_index": ci, "name": call["name"], "param": call["param"],
                             "label": call["label"], "kind": "command", "readout": ts[1] if ts else None,
                             "span": [ts[0], ts[1]] if ts else None})
        idents = [(r["irid"], r["call_index"]) for r in rows]
        if len(set(idents)) != len(idents):
            raise ValueError("duplicate row identity")
        last = max((r["readout"] for r in rows if r["readout"] is not None), default=None)
        rec = {"task": task, "tokens": len(ids), "rows": len(rows), "rows_read": sum(r["readout"] is not None for r in rows),
               "forward_tokens": 0 if last is None else last + 1}
        if last is None:
            return ARMS[(kind, arm_dir)], None, rec | {"excluded": "no readout below the cut"}, cv, mism
        fwd = ids[: last + 1]
        ep = {"key": f"{c['launch']}/{c['rel']}#e0", "arm": ARMS[(kind, arm_dir)], "run": f"{c['launch']}/{c['rel']}",
              "run_id": c["rel"], "task": task, "attempt_index": c["attempt_index"], "attempt_kind": c["attempt_kind"],
              "epoch": 0, "n_tokens": len(ids), "ids": [int(x) for x in fwd], "ids_sha256": ids_sha(fwd), "calls": rows}
        return ep["arm"], ep, rec, cv, mism
    except Exception as e:  # never let a message carry transcript text
        raise Withheld(f"{c['launch']}/{c['rel']}: {type(e).__name__} (message withheld)") from None


def main(reader_name: str, inputs: str, gate: int, limit: int, workers: int, repair: bool = False):
    if not fingerprint().startswith(LABELER):
        raise ValueError(f"labeler fingerprint {fingerprint()[:8]} is not {LABELER}")
    rd = READERS[reader_name]
    if gate and reader_name != "qwen3":
        raise SystemExit("--gate is the Qwen3-32B positive control")
    T = tasks(Path(inputs))
    counted = counted_attempts()
    prompts = defaultdict(dict)
    for (kind, arm_dir, task), c in counted.items():
        if arm_dir in ("honest_a", "honest_b"):
            try:
                hs = load(c["eval"])
                prompts[task][arm_dir] = (transcripts.system_prompt(hs), first_user_sha(hs))
            except Exception as e:
                raise Withheld(f"{c['launch']}/{c['rel']}: {type(e).__name__} (message withheld)") from None
    todo = sorted(k for k in counted if k[1] != "honest_a" and k[2] in T["honest_b" if k[1] == "honest_b" else "attack"])
    if gate:
        todo = sorted(random.Random(SEED).sample(todo, min(gate, len(todo))))  # every run when N >= the runs
    elif limit:
        todo = todo[:limit]
    templates = None
    if repair:  # B2: only the simulated runs, repaired
        import fixes
        import repair_b2

        todo = [k for k in todo if k[1] == "sim_attack@september"]
        templates = repair_b2.learn_templates(LA_ROOT, ERRORS, fixes)
    jobs = [(k, counted[k], [prompts[k[2]].get(h) for h in ("honest_a", "honest_b")], bool(gate)) for k in todo]
    staged, per_run, cov, mism = defaultdict(list), defaultdict(list), defaultdict(Counter), []
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(rd, templates)) as pool:
        for arm, ep, rec, cv, mm in pool.imap_unordered(process, jobs, chunksize=1):
            cov[arm].update(cv)
            per_run[arm].append(rec)
            if ep is not None:
                staged[arm].append(ep)
            if mm is not None:
                mism.append(mm)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    summary = {a: {"runs": len(r), "runs excluded": dict(Counter(x["excluded"] for x in r if "excluded" in x)), "runs staged": len(staged[a]),
                   "rows": sum(x.get("rows", 0) for x in r), "rows read": sum(x.get("rows_read", 0) for x in r),
                   "forward tokens": sum(x.get("forward_tokens", 0) for x in r), "counters": dict(cov[a])} for a, r in sorted(per_run.items())}
    record = {"reader": reader_name, "model": rd, "stamp": stamp, "cut": CUT, "scrub_level": SCRUB_LEVEL, "labeler_fingerprint": fingerprint(),
              "inputs_sha256": PINS, "tasks": {k: len(v) for k, v in T.items()}, "counted_attempts_without_an_eval": NO_EVAL,
              "gate": gate, "limit": limit, "repair": "B2 fix-full minus F7 (repair_b2.py)" if repair else None, "summary": summary}
    if gate:
        tot = {k: sum(m[k] for m in mism) for k in ("calls", "mismatches", "readouts")}
        record["gate_result"] = tot | {"runs": len(mism), "passes": tot["mismatches"] == 0 and tot["readouts"] > 0}
    elif not limit:
        d = STAGE_ROOT / f"sabreaders_{reader_name}{'_repair' if repair else ''}_{'v1' if repair else VERSION.get(reader_name, 'v1')}" / "inputs"
        if d.exists():
            raise FileExistsError(f"{d} exists; stage under a new name")
        d.mkdir(parents=True)
        record["staged_inputs_sha256"] = {}
        for arm, eps in sorted(staged.items()):
            b = "".join(json.dumps(e) + "\n" for e in sorted(eps, key=lambda e: e["key"])).encode()
            (d / f"{arm}.jsonl").write_bytes(b)
            record["staged_inputs_sha256"][f"{arm}.jsonl"] = hashlib.sha256(b).hexdigest()
        record["staged_dir"] = str(d.parent)
    out = RESULTS / f"{'gate' if gate else 'stage'}_{reader_name}{'_repair' if repair else ''}{'_limit' + str(limit) if limit else ''}_{stamp}.json"
    with out.open("x") as fh:
        fh.write(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("summary", "gate_result") if k in record}, indent=1))
    print("->", out)
    if gate and not record["gate_result"]["passes"]:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", dest="reader_name", choices=sorted(READERS), required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--repair", action="store_true", help="B2: the simulated runs only, repaired (repair_b2.py)")
    main(**vars(ap.parse_args()))
