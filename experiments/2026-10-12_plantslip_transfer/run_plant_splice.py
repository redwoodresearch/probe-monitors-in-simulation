"""Planted-slip transfer, on-policy pairs P1, P7 and P8 (RULE.md, Amendment 1): splice plant_both's edit into the
logged token ids of a replay staging, re-tokenizing each planted tool output and nothing else ($0, Modal CPU).

A staging (`/acts/<inputs>/inputs/<arm>.jsonl`, written by run_replay_readouts.py --stage) holds per context epoch the
ids vLLM served and per call its readout, its request's length n and the sha256 of that request's ids. Qwen3.5's chat
template renders a tool message as `<tool_response>` + "\n" + trim(content) + "\n" + `</tool_response>`, and both tags
are single added tokens (248066, 248067), so each response's interior is token-aligned and can be re-tokenized on its own.

Per epoch: the tool calls are read from the decoded assistant turns (run_replay_readouts.py's regexes) and paired in order
with the responses in the user turns that follow. A text_editor response gets plant_output(content, command) (plant_both.py,
q2's selection and edits); the planted content is rendered as the template would, trim() included, and spliced in.
Readouts after a changed response move by its length change; an output_end readout inside a changed response is placed
again by run_replay_readouts.py's rule (the last token whose bytes lie wholly inside the content); request lengths move
and their hashes are recomputed. Checks: staged hashes first; every token outside the changed responses identical after
the shift; every moved readout's token id unchanged.

Modes:
- gate: the plant switched OFF. Every response of every epoch (all tools) is re-tokenized from its own decoded content
  and every output_end readout inside it placed again; the rebuilt ids, readouts, lengths and hashes must equal the
  staged ones exactly, or the function raises. Nothing is written.
- plant: writes only the epochs with at least one changed response to `/acts/<inputs>_plantslip/inputs/<arm>.jsonl`
  (exclusive), and returns the reach: runs, epochs and distinct outputs changed per slip, and whether the plant's
  appended newline reaches the ids. Counts and hashes only; no text leaves the function.

  modal deploy experiments/2026-10-12_plantslip_transfer/run_plant_splice.py
  uv run --no-project --with modal==1.6.1 python run_plant_splice.py <gate|plant> <inputs staging> <arm> [<inputs> <arm> ...]
"""

import hashlib
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-plantslip_splice_kill-2026-11-12"
ACTS, HF_CACHE = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", "probe-shift_aswin-surya_hf-cache"
REV = "fc05daec18b0a78c049392ed2e771dde82bdf654"  # Qwen3.5-27B, as run_read_tests.swap_stage
FILES = {"plant_both.py": HERE, "fixes.py": REPO / "experiments/2026-10-09_fixed_sim",
         "rerender.py": REPO / "experiments/2026-10-07_q2_rendering_swap", "str_replace_tells.py": REPO / "experiments/2026-10-07_q2_rendering_swap"}

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "transformers==5.19.0").env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
for name, folder in FILES.items():
    image = image.add_local_file(folder / name, f"/root/plantmods/{name}")


def bytes_to_unicode() -> dict:  # GPT-2's byte <-> unicode table, as run_replay_readouts.py defines it
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


@app.function(image=image, cpu=8, memory=65536, timeout=3600, retries=0, volumes={"/acts": acts_vol, "/hf": modal.Volume.from_name(HF_CACHE)})
def splice(inputs: str, arm: str, mode: str) -> dict:
    """Errors leave this function as their type and line only: a decoding error can carry transcript bytes."""
    import traceback

    try:
        return _splice(inputs, arm, mode)
    except Exception as e:  # noqa: BLE001 - re-raised without its message or arguments
        line = traceback.extract_tb(e.__traceback__)[-1].lineno
        raise RuntimeError(f"splice failed: {type(e).__name__} at line {line}") from None


def _splice(inputs: str, arm: str, mode: str) -> dict:
    import itertools
    import re
    from collections import Counter

    import numpy as np
    from transformers import AutoTokenizer

    sys.path.insert(0, "/root/plantmods")
    import fixes  # noqa: E402  (puts nothing else on the path; rerender and str_replace_tells sit beside it)
    import rerender  # noqa: E402
    from plant_both import plant_output  # noqa: E402

    if mode not in ("gate", "plant"):
        raise ValueError(mode)
    acts_vol.reload()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", revision=REV)
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    tr_open, tr_close = tok.convert_tokens_to_ids("<tool_response>"), tok.convert_tokens_to_ids("</tool_response>")
    if (tr_open, tr_close) != (248066, 248067):
        raise ValueError("unexpected tool_response token ids")
    byte_of = {u: b for b, u in bytes_to_unicode().items()}
    added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    sha = lambda ids: hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()

    def tbytes(seq):
        return [added[t].encode() if t in added else bytes(byte_of[ch] for ch in tok.convert_ids_to_tokens(t)) for t in seq]

    def last_inside(tb, lo, hi):  # run_replay_readouts.py's rule
        ends = list(itertools.accumulate(len(x) for x in tb))
        inside = [i for i, (s, e) in enumerate(zip([0] + ends[:-1], ends)) if s >= lo and e <= hi and e > s]
        return inside[-1] if inside else None

    call_re = re.compile(r"<tool_call>\n<function=([^>\n]+)>\n(.*?)</function>\n</tool_call>", re.S)
    param_re = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", re.S)

    def responses(ids):
        """[(open index, close index, tool name, command)] for every tool response, paired in order with its call."""
        out, pending = [], []
        starts = [k for k, x in enumerate(ids) if x == im_start]
        for s in starts:
            e = next((k for k in range(s + 1, len(ids)) if ids[k] == im_end), len(ids))
            role = b"".join(tbytes(ids[s + 1: s + 3])).decode("utf-8", "replace")
            if role.startswith("assistant"):
                text = b"".join(tbytes(ids[s + 1: e])).decode("utf-8", "strict")
                pending = [(m.group(1), {p.group(1): p.group(2) for p in param_re.finditer(m.group(2))}) for m in call_re.finditer(text)]
            elif role.startswith("user"):
                opens = [k for k in range(s, e) if ids[k] == tr_open]
                for o in opens:
                    c = next((k for k in range(o + 1, e) if ids[k] == tr_close), None)
                    if c is None:
                        raise ValueError("a tool response without its closing tag inside the turn")
                    if not pending:
                        raise ValueError("a tool response with no pending call")
                    name, params = pending.pop(0)
                    out.append((o, c, name, (params.get("command") or "").strip()))
                if opens:
                    pending = []
        n_open, n_close = sum(x == tr_open for x in ids), sum(x == tr_close for x in ids)
        if n_open != n_close or n_open != len(out):
            raise ValueError("tool_response tags unpaired, or outside user turns")
        return out

    src = Path("/acts") / inputs / "inputs" / f"{arm}.jsonl"
    raw = src.read_bytes()
    cnt, changed_runs, outs, out_eps = Counter(), set(), Counter(), []
    for line in raw.decode().splitlines():
        ep = json.loads(line)
        ids = ep["ids"]
        for c in ep["calls"]:
            if sha(ids[: c["n"]]) != c["sha256"]:
                raise ValueError("staged ids differ from the staged hash")
        cnt["epochs"] += 1
        spans = responses(ids)
        cnt["responses"] += len(spans)
        new, cur, edits = [], 0, []  # edits: (old open, old close, new open, new close)
        for o, c, name, cmd in spans:
            seg = b"".join(tbytes(ids[o + 1: c])).decode("utf-8", "strict")
            if not (seg.startswith("\n") and seg.endswith("\n")):
                raise ValueError("a tool response not wrapped in newlines as the template renders it")
            content = seg[1:-1]
            cnt["contents_not_trimmed"] += content != content.strip()
            planted = None
            if mode == "plant" and name == "text_editor":
                cnt[f"text_editor {cmd if cmd in ('view', 'create', 'str_replace', 'insert', 'undo_edit') else 'other or no command'}"] += 1
                p = plant_output(content, cmd, fixes, rerender)
                if p is not None and p != content:
                    planted = p
            if mode == "gate" or planted is not None:
                rendered = (planted if planted is not None else content).strip()  # the template's trim
                new_seg = tok.encode("\n" + rendered + "\n", add_special_tokens=False)
                new.extend(ids[cur: o + 1])
                n_o = len(new) - 1
                new.extend(new_seg)
                edits.append((o, c, n_o, len(new), "\n" + rendered + "\n", planted is not None))
                cur = c
                if planted is not None:
                    slip = "gutter+newline" if cmd == "str_replace" else "note"
                    cnt[f"changed {slip}"] += 1
                    cnt["planted ends with a newline"] += planted.endswith("\n")
                    cnt["newline reaches the ids"] += rendered.endswith("\n")
                    outs[(ep["run"], slip, hashlib.sha256(content.encode()).hexdigest())] += 1
        new.extend(ids[cur:])
        if mode == "plant" and not any(e[5] for e in edits):
            continue
        # tokens outside the edited responses: identical after the shift
        prev_o, prev_n = 0, 0
        for o, c, n_o, n_c, _, _ in edits:
            if new[prev_n: n_o + 1] != ids[prev_o: o + 1]:
                raise ValueError("tokens outside the edited responses changed")
            prev_o, prev_n = c, n_c
        if new[prev_n:] != ids[prev_o:]:
            raise ValueError("tokens after the last edited response changed")

        def moved(pos):
            """new index of an old position outside every edited response"""
            d = sum((n_c - n_o) - (c - o) for o, c, n_o, n_c, _, _ in edits if c <= pos)
            return pos + d

        calls = []
        for call in ep["calls"]:
            call = dict(call)
            r = call["readout"]
            if r is not None:
                inside = [e for e in edits if e[0] < r < e[1]]
                if inside:
                    if call.get("kind") != "output_end":
                        raise ValueError("a non-output readout inside a tool response")
                    o, c, n_o, n_c, seg_text, _ = inside[0]
                    tb = tbytes(new[n_o + 1: n_c])
                    i_in = last_inside(tb, 1, 1 + len(seg_text[1:-1].encode()))
                    call["readout"] = None if i_in is None else n_o + 1 + i_in
                    cnt["output_end readouts placed again"] += 1
                else:
                    nr = moved(r)
                    if new[nr] != ids[r]:
                        raise ValueError("a moved readout's token changed")
                    call["readout"] = nr
            if any(e[0] < call["n"] < e[1] for e in edits):
                raise ValueError("a request ends inside a tool response")
            call["n"] = moved(call["n"])
            call["sha256"] = sha(new[: call["n"]])
            calls.append(call)
        if mode == "gate":
            if new != ids or calls != ep["calls"]:
                cnt["epochs not reproduced"] += 1
                raise ValueError(f"gate: an epoch is not reproduced exactly (ids equal: {new == ids})")
            cnt["epochs reproduced exactly"] += 1
            continue
        changed_runs.add(ep["run"])
        cnt["epochs changed"] += 1
        out_eps.append(ep | {"ids": new, "calls": calls, "plantslip": {"responses_changed": sum(e[5] for e in edits)}})
    res = {"inputs": inputs, "arm": arm, "mode": mode, "source_sha256": hashlib.sha256(raw).hexdigest(), "counts": dict(cnt)}
    if mode == "plant":
        runs_all = {json.loads(line)["run"] for line in raw.decode().splitlines()}
        res |= {"runs": len(runs_all), "runs_changed": len(changed_runs),
                "distinct_outputs_changed": dict(Counter(k[1] for k in outs))}
        d = Path("/acts") / f"{inputs}_plantslip" / "inputs"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{arm}.jsonl"
        if f.exists():
            raise FileExistsError(f"{f} exists")
        b = "".join(json.dumps(e) + "\n" for e in out_eps).encode()
        f.write_bytes(b)
        acts_vol.commit()
        res |= {"written": str(f.relative_to("/acts")), "written_sha256": hashlib.sha256(b).hexdigest(), "epochs_written": len(out_eps)}
    return res


def main(mode: str, pairs: list[str]):
    fn = modal.Function.from_name(APP, "splice")
    jobs = [(pairs[i], pairs[i + 1], mode) for i in range(0, len(pairs), 2)]
    res = list(fn.starmap(jobs, return_exceptions=True))
    for (inputs, arm, _), r in zip(jobs, res):
        print(json.dumps(r if isinstance(r, dict) else {"inputs": inputs, "arm": arm, "error": type(r).__name__}))  # no message: it can quote text


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
