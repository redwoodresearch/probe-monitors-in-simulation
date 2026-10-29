"""The retrain's masked inputs (plan_attack_detector_retrain_2026-10-10.md: dates and times masked, scrub level 1): each
staged replay epoch with times and dates masked, re-staged for run_extract_replay.py. Tokenizer only; Modal CPU.

Source: a run_replay_readouts.py stage (`<stage>/inputs/<arm>.jsonl` on the activations Volume); every call's staged
sha256 is checked against the staged ids first. Scope, as the cached off-policy level-1 extraction
(run_extract_segment_end.py's scrubbed()): every assistant turn (its text, reasoning and tool-call arguments) and every
tool-output turn (a user turn opening with `<tool_response>`) is passed through `probe_shift.scrub.scrub(text, 1)`; the
system prompt and the task's user message are not. A turn is the tokens strictly between `<|im_start|>` and its
`<|im_end|>` (or the end of the epoch), split at any request boundary inside it, each piece masked by the turn's role.
A piece whose text the scrub leaves unchanged keeps its tokens exactly; otherwise the masked text is tokenized again (no
special tokens added; it must decode back to the same text, and keep its number of parameter values and tool outputs)
and spliced in. Every position outside a changed piece moves by the length change before it.
Readouts in a changed piece are recomputed by their rule on the masked text: the last token wholly inside the same
`<parameter=...>` value (command, call_end) or `<tool_response>` block (output_end), by its order in the piece. The same
recomputation on the unmasked piece must return the staged readout, or the stage stops. Each call's request length n and
the sha256 of the masked ids[:n] are rewritten, so the extractor's gate checks the masked inputs' own consistency (they
are a reading of Qwen's runs with dates removed, not a replay of the served tokens). Writes `<stage>_scrub1/inputs/` and
returns per arm: epochs, pieces changed, readouts recomputed and lost, and the file sha256 for the extraction config.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \
      experiments/2026-10-08_q3_onpolicy_figures/run_mask_stage.py --stage-name scaleup_l2_segends_v1 --arms attack,honest_a,honest_b,sim_attack@september
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q3onpol_maskstage_kill-2026-11-10"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
MODEL, REVISION, LEVEL = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", 1
REGION = {"output_end": re.compile(r"<tool_response>\n(.*?)\n</tool_response>", re.S),
          "param": re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", re.S)}

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "numpy")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


def bytes_to_unicode() -> dict:
    """GPT-2's byte-level BPE alphabet (Qwen's tokenizer uses it), as run_replay_readouts.bytes_to_unicode."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, k = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + k)
            k += 1
    return dict(zip(bs, map(chr, cs)))


def make_masker(tok):
    """mask_epoch(ep, n) -> the epoch with its turns masked and its calls' readouts, n and sha256 rewritten."""
    import itertools

    import numpy as np

    from probe_shift.scrub import SCRUB_VERSION, scrub

    byte_of = {u: b for b, u in bytes_to_unicode().items()}
    added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")

    def tbytes(seq):
        return [added[t].encode() if t in added else bytes(byte_of[ch] for ch in tok.convert_ids_to_tokens(t)) for t in seq]

    def regions(text, kind):
        return [tuple(len(text[:x].encode()) for x in m.span(1 if kind == "output_end" else 2)) for m in REGION[kind].finditer(text)]

    def spans_of(seq):
        ends = list(itertools.accumulate(len(b) for b in tbytes(seq)))
        return list(zip([0] + ends[:-1], ends))

    def readout_in(seq, text, kind, k):
        """The last token of `seq` wholly inside the k-th region of `kind` in `text` (= the decoded seq), or None."""
        regs = regions(text, kind)
        if k >= len(regs):
            return None
        inside = [i for i, (s, e) in enumerate(spans_of(seq)) if s >= regs[k][0] and e <= regs[k][1] and e > s]
        return inside[-1] if inside else None

    def region_of(seq, text, kind, i):
        """The order k of the region (of `kind`) wholly containing token i of `seq`, or None."""
        s, e = spans_of(seq)[i]
        hits = [k for k, (a, b) in enumerate(regions(text, kind)) if a <= s and e <= b]
        return hits[0] if len(hits) == 1 else None

    def mask_epoch(ep, n):
        ids = ep["ids"]
        for c in ep["calls"]:  # the staged inputs as written by run_replay_readouts.py, before anything is rewritten
            if hashlib.sha256(np.asarray(ids[: c["n"]], np.int32).tobytes()).hexdigest() != c["sha256"]:
                raise ValueError(f"{ep['key']} call {c.get('irid')}: the staged ids differ from the staged hash")
        starts = [i for i, x in enumerate(ids) if x == im_start]
        bounds = sorted({c["n"] for c in ep["calls"]})  # a completion without <|im_end|> ends its request mid-turn
        new, old_to_new, spans = [], {}, []  # spans: (old lo, old hi, new lo, new seq, text, masked, old seq) of changed turns
        pos = 0
        for s in starts + [len(ids)]:
            for i in range(pos, s):  # tokens outside any turn interior (specials, separators)
                old_to_new[i] = len(new)
                new.append(ids[i])
            if s == len(ids):
                break
            old_to_new[s] = len(new)
            new.append(ids[s])
            end = next((j for j in range(s + 1, len(ids)) if ids[j] == im_end), len(ids))
            head = b"".join(tbytes(ids[s + 1: min(end, s + 8)])).decode("utf-8", errors="ignore")
            # the role, from the turn's first tokens; a tool turn is a user turn that opens with a tool response
            maskable = head.startswith("assistant") or head.startswith("user\n<tool_response>")
            cuts = [s + 1] + [b for b in bounds if s + 1 < b < end] + [end]  # request boundaries split a turn
            n["turns cut at a request boundary"] += len(cuts) > 2
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                old_to_new[lo] = len(new)  # a piece's start (a request boundary) maps to its new start, changed or not
                seq = ids[lo:hi]
                text = b"".join(tbytes(seq)).decode("utf-8", errors="strict")
                masked = scrub(text, LEVEL) if maskable else text
                if masked == text:
                    for j, t in enumerate(seq):
                        old_to_new[lo + j] = len(new)
                        new.append(t)
                    continue
                mseq = tok.encode(masked, add_special_tokens=False)
                if b"".join(tbytes(mseq)).decode("utf-8") != masked:
                    raise ValueError(f"{ep['key']}: the masked turn does not round-trip through the tokenizer")
                if any(len(regions(text, k)) != len(regions(masked, k)) for k in REGION):
                    raise ValueError(f"{ep['key']}: masking changed the number of parameter values or tool outputs in a turn")
                spans.append((lo, hi, len(new), mseq, text, masked, seq))
                new.extend(mseq)
                n["turns changed"] += 1
            pos = end
        old_to_new[len(ids)] = len(new)
        calls = []
        for c in ep["calls"]:
            c = dict(c)
            r = c["readout"]
            hit = [sp for sp in spans if r is not None and sp[0] <= r < sp[1]]
            if hit:
                lo, _, nlo, mseq, text, masked, seq = hit[0]
                kind = "output_end" if c.get("kind") == "output_end" else "param"
                k = region_of(seq, text, kind, r - lo)
                if k is None or readout_in(seq, text, kind, k) != r - lo:
                    raise ValueError(f"{ep['key']} call {c.get('irid')}: the staged readout is not its region's last token")
                i = readout_in(mseq, masked, kind, k)
                c["readout"] = None if i is None else nlo + i
                n["readouts recomputed"] += 1
                n["readouts lost to masking"] += i is None
            elif r is not None:
                c["readout"] = old_to_new[r]
            if c["n"] not in old_to_new or any(sp[0] < c["n"] < sp[1] for sp in spans):
                raise ValueError(f"{ep['key']}: a request ends inside a changed turn")
            c["n"] = old_to_new[c["n"]]
            c["sha256"] = hashlib.sha256(np.asarray(new[: c["n"]], np.int32).tobytes()).hexdigest()
            calls.append(c)
        n["epochs"] += 1
        n["tokens"] += len(new)
        return ep | {"ids": new, "calls": calls, "scrub_level": LEVEL, "scrub_version": SCRUB_VERSION}

    return mask_epoch


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0,
              volumes={"/acts": modal.Volume.from_name(ACTS, version=2), "/hf": modal.Volume.from_name(HF_CACHE)})
def stage(src: str, arm: str) -> dict:
    from transformers import AutoTokenizer

    mask_epoch, n = make_masker(AutoTokenizer.from_pretrained(MODEL, revision=REVISION)), Counter()
    out_eps = [mask_epoch(json.loads(line), n) for line in (Path("/acts") / src / "inputs" / f"{arm}.jsonl").read_text().splitlines()]
    d = Path("/acts") / f"{src}_scrub{LEVEL}" / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{arm}.jsonl"
    if f.exists():
        raise FileExistsError(f"{f} exists; stage under a new name")
    b = "".join(json.dumps(e) + "\n" for e in out_eps).encode()
    f.write_bytes(b)
    modal.Volume.from_name(ACTS, version=2).commit()
    return {"src": src, "arm": arm, "sha256": hashlib.sha256(b).hexdigest(), **n}


@app.local_entrypoint()
def main(stage_name: str, arms: str):
    for r in stage.map([stage_name] * len(arms.split(",")), arms.split(",")):
        print(json.dumps(r), flush=True)
