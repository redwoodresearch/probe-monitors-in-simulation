"""POST HOC (for #1367's review): does the token that sets each chosen run's score lie WHOLLY inside a content span? The
spans follow the Qwen3.5 chat template as rendered: a system or user body; the assistant's think block and its text
before the first tool call; each parameter value (between `<parameter=…>\\n` and `\\n</parameter>`); each tool output
(between `<tool_response>\\n` and `\\n</tool_response>`). Everything else is template. A token is "inside" if its bytes
lie wholly in one span, "crosses" if it overlaps a span without lying inside it, and "template" otherwise. Offsets come
from each token's own bytes (the tokenizer's byte-level vocabulary), so there is no decode-then-encode step.
The per-run classes are saved. Modal CPU, about $0.05 (Aswin's "<$5 just run"); only classes, scores and run ids leave.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal modal run experiments/2026-10-13_simcue_onpolicy/check_POSTHOC_argmax_spans.py
"""

import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
ROWS, ROWS_SHA = ROOT / ".run_cache/simcue_onpolicy/rows_2026-10-13_121721.json", "1abd1dc0e7437db5708ffc306a831b1a3ba6a918ec1390d9287593618decfab1"
ARM = {"real1": "attack", "real2": "attack", "sim_sept": "sim_attack@september", "honest_a": "honest_a", "honest_b": "honest_b"}
T1 = 6.018515315380688  # results/analysis_2026-10-13_124152.json
app = modal.App("probe-shift_aswin-surya_sub-simcue-spans_kill-2026-11-13")
q3 = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", version=2).read_only()
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "huggingface_hub")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


def bytes_to_unicode():
    """GPT-2's byte-to-unicode map, which byte-level BPE vocabularies (Qwen's included) use for their token strings."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for c in range(256):
        if c not in bs:
            bs.append(c)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def content_spans(b):
    """Byte spans of content in a rendered Qwen3.5 transcript (bytes)."""
    out = []
    for m in re.finditer(rb"<\|im_start\|>(system|user|assistant)\n(.*?)(?:<\|im_end\|>|\Z)", b, re.S):
        role, s, body = m.group(1), m.start(2), m.group(2)
        if role == b"user" and body.startswith(b"<tool_response>"):
            out += [(s + t.start(1), s + t.end(1)) for t in re.finditer(rb"<tool_response>\n(.*?)\n</tool_response>", body, re.S)]
        elif role == b"assistant":
            th = re.match(rb"<think>\n?(.*?)\n?</think>\n*", body, re.S)
            k = 0
            if th:
                out.append((s + th.start(1), s + th.end(1)))
                k = th.end()
            tc = body.find(b"<tool_call>", k)
            text_end = len(body) if tc < 0 else tc
            text = body[k:text_end].rstrip(b"\n")
            out.append((s + k, s + k + len(text)))
            out += [(s + p.start(1), s + p.end(1)) for p in re.finditer(rb"<parameter=[^>\n]*>\n(.*?)\n</parameter>", body, re.S)]
        else:
            out.append((s, m.end(2)))
    return [x for x in out if x[1] > x[0]]


@app.function(image=image, volumes={"/q3": q3, "/hf": hf}, cpu=8, memory=32768, timeout=3600, retries=0)
def classify(items: list) -> list:
    import os
    import sys
    from collections import defaultdict

    from transformers import AutoTokenizer

    from probe_shift.models import QWEN35_27B
    sys.stdout.flush(), sys.stderr.flush()
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1), os.dup2(null, 2)  # nothing printed leaves
    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    dec = {v: k for k, v in bytes_to_unicode().items()}
    added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    vocab_bytes = lambda i: added[i].encode() if i in added else bytes(dec[c] for c in tok.convert_ids_to_tokens(i))
    want = defaultdict(dict)
    for it in items:
        want[(it["stage"], ARM[it["world"]])][(it["run"], it["epoch"])] = it
    out = []
    for (stage, arm), need in want.items():
        for line in (Path("/q3") / stage / "inputs" / f"{arm}.jsonl").read_text().splitlines():
            ep = json.loads(line)
            it = need.get((ep["run"], ep["epoch"]))
            if it is None:
                continue
            pieces = [vocab_bytes(i) for i in ep["ids"]]
            starts, pos = [], 0
            for p in pieces:
                starts.append(pos)
                pos += len(p)
            b = b"".join(pieces)
            a, e = starts[it["argmax"]], starts[it["argmax"]] + len(pieces[it["argmax"]])
            spans = content_spans(b)
            inside = any(s <= a and e <= t for s, t in spans)
            overlap = any(s < e and t > a for s, t in spans)
            out.append(it | {"class": "inside" if inside else ("crosses" if overlap else "template")})
    return out


@app.local_entrypoint()
def main():
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    st = git("status", "--porcelain", "--untracked-files=no")
    if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
        raise SystemExit("commit first")
    if hashlib.sha256(ROWS.read_bytes()).hexdigest() != ROWS_SHA:
        raise SystemExit("the rows file is not the collected one")
    chosen = json.loads((HERE / "results/check_2026-10-13_121629_check.json").read_text())["result"]["chosen"]
    best = {}
    for r in json.loads(ROWS.read_text()):
        k = (r["world"], r["run"])
        if r["run"] in set(chosen[r["world"]]) and (k not in best or r["max"] > best[k]["max"]):
            best[k] = r
    items = [{"world": r["world"], "stage": r["stage"], "run": r["run"], "epoch": r["epoch"], "argmax": r["argmax"], "score": r["max"]} for r in best.values()]
    res = classify.remote(items)
    if len(res) != len(items):
        raise ValueError(f"{len(items) - len(res)} runs were not classified")
    top_real1 = sorted((x for x in res if x["world"] == "real1"), key=lambda x: -x["score"])[:2]
    out = {"git": git("rev-parse", "HEAD").stdout.strip(), "runs": len(res),
           "class per world": {w: dict(Counter(x["class"] for x in res if x["world"] == w)) for w in ARM},
           "runs above the 1% threshold 6.0185": sorted([[x["world"], x["class"], x["score"]] for x in res if x["score"] > T1]),
           "the two highest first-real runs (they set the threshold)": [[x["class"], x["score"]] for x in top_real1],
           "per run [world, run, epoch, argmax, score, class]": sorted([[x["world"], x["run"], x["epoch"], x["argmax"], x["score"], x["class"]] for x in res])}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"argmax_spans_POSTHOC_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")  # one key per line
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("per run")}, indent=1), f"\n-> {path}")
