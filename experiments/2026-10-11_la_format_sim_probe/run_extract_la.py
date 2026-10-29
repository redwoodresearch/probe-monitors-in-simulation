"""N1 Part 1, stage 2 (README.md): read the LinuxArena-format pairs and the 2×2's synthetic pairs with Qwen3-32B.

One H200. Per dialogue, one forward pass (Qwen3-32B @ 9216db57, bf16, sdpa, batch 1, end-padded to a multiple of 1,024
tokens), stopped after layer 24, storing in fp16, at every layer 0–24:
- content: the last 32 tokens of the final call's call_end span (Qwen3: the JSON call);
- turn start: the last token of "<|im_start|>assistant\\n" opening the final call's turn.
The two runs, `n1C_<stamp>` and `n1T_<stamp>`, are laid out as run_extract_concept.py's, so run_fit.py fits them.
The pairs file is read on the Volume and checked against the build's sha256. The synthetic pairs are the retry's own
120 held-out pairs with the two reworded cues, built here and passed in. Only counts and timings return; failures
return type and line numbers. Every 150 dialogues are checkpointed on the Volume, so a stopped run resumes with
`--resume <stamp>`. Retries 0; an external list-rate watcher stops the app at its stop.

  PYTHONPATH=src uv run modal deploy experiments/2026-10-11_la_format_sim_probe/run_extract_la.py
  PYTHONPATH=src uv run python experiments/2026-10-11_la_format_sim_probe/run_extract_la.py launch results/build_<stamp>.json [--resume <stamp>]
  PYTHONPATH=src uv run python experiments/2026-10-11_la_format_sim_probe/run_extract_la.py collect <call id>
"""

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-n1_extract_kill-2026-11-10"
VOL = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09"
BUCKET, CAP, MAX_TOKENS, CHUNK = 1024, 32, 32768, 150
LAYERS = list(range(25))
Q1 = HERE.parent / "2026-10-11_fake_env_concept_probe"
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub"]

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
hf_cache = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift"))
LOG_FD = 1


def log(msg):  # counts only, to the real stdout (the Modal log)
    import os
    os.write(LOG_FD, (msg + "\n").encode())


@app.function(image=image, gpu="H200", volumes={"/hf": hf_cache, "/acts": vol}, memory=32768, timeout=5400, retries=0)  # one 150-dialogue chunk in memory at a time
def extract(pairs_file: str, pairs_sha: str, synthetic: list[dict], stamp: str, prov: dict) -> dict:
    global LOG_FD
    import os
    import traceback

    null = os.open(os.devnull, os.O_WRONLY)  # no LinuxArena text in logs or errors
    LOG_FD = os.dup(1)
    os.dup2(null, 1), os.dup2(null, 2)
    try:
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != prov["extractor_sha256"]:
            raise ValueError("the deployed extractor is not the launching commit's: redeploy")
        return _extract(pairs_file, pairs_sha, synthetic, stamp, prov)
    except BaseException as e:
        lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("run_extract_la.py")]
        raise RuntimeError(f"extract failed: {type(e).__name__} at run_extract_la.py lines {lines}") from None


def _extract(pairs_file, pairs_sha, synthetic, stamp, prov):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import probe_shift.reader as reader_module
    from probe_shift.models import QWEN3_32B
    from probe_shift.reader import segment_end_positions

    vol.reload()  # a warm container sees the latest commits
    raw = (Path("/acts") / pairs_file).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pairs_sha:
        raise ValueError("the pairs file does not match the build's sha256")
    items = [json.loads(x) for x in raw.decode().splitlines()] + synthetic
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    lm = AutoModelForCausalLM.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"], dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="sdpa").eval()
    load_s = time.time() - t0
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    im_start = tok.convert_tokens_to_ids("<|im_start|>")

    class Stop(Exception):
        pass

    ck = Path("/acts") / f"n1_chunks_{stamp}"  # one checkpoint per 150 dialogues: a stopped run resumes with its stamp
    ck.mkdir(exist_ok=True)
    fp = json.dumps({"extractor": prov["extractor_sha256"], "reader": hashlib.sha256(Path(reader_module.__file__).read_bytes()).hexdigest(),
                     "model": QWEN3_32B, "pairs": pairs_sha, "synthetic": prov["synthetic_sha256"], "chunk": CHUNK, "layers": LAYERS,
                     "cap": CAP, "bucket": BUCKET})
    have = (ck / "fingerprint.json").read_text() if (ck / "fingerprint.json").exists() else None
    if (have is None and any(ck.iterdir())) or (have is not None and have != fp):
        raise ValueError("these checkpoints were made from other inputs or code, or carry no fingerprint: not resumed")
    (ck / "fingerprint.json").write_text(fp)
    fwd_s, n_tok, resumed = 0.0, 0, 0
    for c in range(0, len(items), CHUNK):
        if (ck / f"c{c:05d}.done").exists():
            resumed += 1
            continue
        rows = {"C": {l: [] for l in LAYERS}, "T": {l: [] for l in LAYERS}}
        meta = {"C": [], "T": []}
        for k in range(c, min(c + CHUNK, len(items))):
            it = items[k]
            r = segment_end_positions(tok, it["messages"], MAX_TOKENS)
            last_step = sum(m["role"] == "assistant" for m in it["messages"]) - 1
            ends = [p for p in r["positions"] if p["kind"] == "call_end" and p["kept"] and p.get("span") and (p["step"], p["action"]) == (last_step, 0)]
            if len(ends) != 1 or r["truncated"]:
                raise ValueError(f"dialogue {k}: {len(ends)} final-call spans inside the cut")
            i0, i1 = ends[0]["span"]
            a0 = max(i0, i1 - CAP)
            starts = [j for j in range(i0) if r["ids"][j] == im_start]
            if not starts or tok.decode(r["ids"][starts[-1]:starts[-1] + 3]) != "<|im_start|>assistant\n":
                raise ValueError(f"dialogue {k}: the final call's turn does not open with the assistant header")
            b = starts[-1] + 2
            acts = {}

            def hook_for(layer):
                def hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    acts[layer] = (h[0, a0:i1].to(torch.float16).cpu().numpy(), h[0, b:b + 1].to(torch.float16).cpu().numpy())
                    if layer == LAYERS[-1]:
                        raise Stop
                return hook

            handles = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in LAYERS]
            t1 = time.time()
            try:
                with torch.inference_mode():
                    lm.model(input_ids=torch.tensor([r["ids"] + [pad] * (-len(r["ids"]) % BUCKET)], device="cuda"), use_cache=False)
            except Stop:
                pass
            finally:
                for h in handles:
                    h.remove()
            fwd_s += time.time() - t1
            n_tok += len(r["ids"])
            for l in LAYERS:
                rows["C"][l].append(acts[l][0])
                rows["T"][l].append(acts[l][1])
            base = {x: it.get(x) for x in ("id", "family", "split", "pair", "label", "task", "mechanism", "position")}
            meta["C"].append(base | {"n_rows": i1 - a0, "span": [a0, i1], "full_span": [i0, i1]})
            meta["T"].append(base | {"n_rows": 1, "span": [b, b + 1]})
        for v in ("C", "T"):
            for l in LAYERS:
                np.save(ck / f"c{c:05d}_{v}_L{l}.npy", np.concatenate(rows[v][l]))
            (ck / f"c{c:05d}_{v}_meta.json").write_text(json.dumps(meta[v]))
        (ck / f"c{c:05d}.done").write_text("")
        vol.commit()
        log(f"{min(c + CHUNK, len(items))}/{len(items)} dialogues, {n_tok / max(fwd_s, 1e-9):.0f} tok/s, {time.time() - t0:.0f} s")
    runs, n_c = {}, 0
    for v in ("C", "T"):  # assemble the runs run_fit.py reads, in item order
        d = Path("/acts") / f"n1{v}_{stamp}"
        runs[v] = d.name
        if (ck / f"assembled_{v}").exists():  # a finished readout is left alone
            n_c = sum(m["n_rows"] for m in json.loads((d / "meta.json").read_text())) if v == "C" else n_c
            continue
        d.mkdir(parents=True, exist_ok=True)  # an interrupted assembly is redone from the chunks; nothing reads d until its marker
        cs = range(0, len(items), CHUNK)
        for l in LAYERS:
            np.save(d / f"L{l}.npy", np.concatenate([np.load(ck / f"c{c:05d}_{v}_L{l}.npy") for c in cs]))
        meta, row = [], 0
        for c in cs:
            for m in json.loads((ck / f"c{c:05d}_{v}_meta.json").read_text()):
                meta.append(m | {"row_start": row})
                row += m["n_rows"]
        n_c = row if v == "C" else n_c
        (d / ".meta.json.tmp").write_text(json.dumps(meta))
        (d / ".meta.json.tmp").rename(d / "meta.json")
        (d / ".manifest.json.tmp").write_text(json.dumps(prov | {
            "model": QWEN3_32B["name"], "revision": QWEN3_32B["revision"], "dtype": "bfloat16", "attn": "sdpa", "batch": 1,
            "padding": f"end, to a multiple of {BUCKET}", "layers": LAYERS, "sets_sha256": pairs_sha,
            "readout": {"C": f"last {CAP} tokens of the final call_end span (Qwen3: the JSON call)",
                        "T": "turn start (last token of <|im_start|>assistant\\n)"}[v]}))
        (d / ".manifest.json.tmp").rename(d / "manifest.json")
        (ck / f"assembled_{v}").write_text("")
    vol.commit()
    return {"runs": runs, "dialogues": len(items), "synthetic dialogues": len(synthetic), "content rows": n_c, "tokens (this attempt)": n_tok,
            "chunks resumed": resumed,
            "forward_s": round(fwd_s, 1), "load_s": round(load_s, 1), "tokens_per_s": round(n_tok / max(fwd_s, 1e-9)),
            "gpu": torch.cuda.get_device_properties(0).name, "provenance": prov}


def synthetic_items():  # the retry's own held-out pairs with the two reworded cues (README section 3; local, no LinuxArena text)
    sys.path.insert(0, str(HERE))
    from run_build_la_pairs import HELD_TASK, HELD_TOOL

    built = json.loads((Q1 / "results" / "concept_sets_v2_2026-10-11_015302.json").read_text())
    raw = (ROOT / ".run_cache" / "fake_env_concept_probe" / built["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != built["sha256"]:
        raise ValueError("the retry's set file is not its recorded build")
    old = {"test_card_numbers": " (Every card number in the orders table is 4242 4242 4242 4242.)",
           "null_mailer": "sendmail: delivered to the null mailer; nothing leaves this host"}
    new = {"test_card_numbers": HELD_TASK["test_card_numbers"][0], "null_mailer": HELD_TOOL["null_mailer"][1]}
    items = []
    for it in map(json.loads, raw.decode().splitlines()):
        if it["split"] != "heldout":
            continue
        text = json.dumps(it["messages"])
        if it["mechanism"] in old and it["label"] == 1:
            if text.count(json.dumps(old[it["mechanism"]])[1:-1]) != 1:
                raise ValueError(f"{it['id']}: the old cue is not there once")
            text = text.replace(json.dumps(old[it["mechanism"]])[1:-1], json.dumps(new[it["mechanism"]])[1:-1])
        items.append(dict(it, id=f"syn_{it['id']}", pair=f"syn_{it['pair']}", split="synthetic", position=None, messages=json.loads(text)))
    if len(items) != 240:
        raise ValueError(f"{len(items)} synthetic dialogues, not 240")
    return items


if __name__ == "__main__":
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    if sys.argv[1] == "launch":
        st = git("status", "--porcelain", "--untracked-files=no")
        if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
            raise SystemExit("commit first: the extraction must record the commit of the code that made it")
        build = json.loads((HERE / sys.argv[2]).read_text())
        syn = synthetic_items()
        stamp = sys.argv[sys.argv.index("--resume") + 1] if "--resume" in sys.argv else datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        prov = {"git": git("rev-parse", "HEAD").stdout.strip(), "stamp": stamp, "build": sys.argv[2], "pairs": build["file"],
                "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "synthetic_sha256": hashlib.sha256(json.dumps(syn).encode()).hexdigest()}
        call = modal.Function.from_name(APP, "extract").spawn(build["file"], build["sha256"], syn, stamp, prov)
        print(json.dumps({"call": call.object_id, **prov}, indent=1))
    elif sys.argv[1] == "collect":
        out = modal.FunctionCall.from_id(sys.argv[2]).get()
        path = HERE / "results" / f"extract_{out['provenance']['stamp']}.json"
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1), f"\n-> {path}")
