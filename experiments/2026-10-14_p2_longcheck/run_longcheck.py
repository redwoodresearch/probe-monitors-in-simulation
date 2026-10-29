"""RULE.md: one H200 container re-reads each shard's 8 longest items of the high-stakes and cyber stage-1 reads with the same
capture code, and compares with the stored rows (cosine per item, readout and layer; pass >= 0.9999).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-14_p2_longcheck/run_longcheck.py \\
      --hs-stamp 2026-10-14_131233 --cy-stamp <cyber stage-1 stamp>
"""

import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
APP = "probe-shift_aswin-surya_sub-p2probes_longcheck_kill-2026-11-13"
JOBS = {"highstakes": "2026-10-14_p2_highstakes", "cyber": "2026-10-14_p2_cyber"}
N_LONG, PASS = 8, 0.9999
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy==2.3.3", "flash-linear-attention==0.5.2", "jinja2"]

app = modal.App(APP)
reads = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_reads_kill-2026-11-13", version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
for name, folder in JOBS.items():
    image = image.add_local_file(EXP / folder / "run_read_fit.py", f"/root/{name}_reader.py")


@app.function(image=image, gpu=["H200"], cpu=(8, 8), memory=(65536, 65536), timeout=900, retries=0,
              volumes={"/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(), "/reads": reads})
def recheck(stamps: dict) -> dict:
    import importlib

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mods = {n: importlib.import_module(f"{n}_reader") for n in JOBS}
    m0 = next(iter(mods.values()))
    tok = AutoTokenizer.from_pretrained(m0.MODEL, revision=m0.REV)
    special = set(tok.all_special_ids)
    base = AutoModelForCausalLM.from_pretrained(m0.MODEL, revision=m0.REV, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval().model
    gpu, out = torch.cuda.get_device_name(0), {}
    for name, mod in mods.items():
        items, _ = mod.load_items()
        d = Path("/reads") / name / stamps[name]
        for shard in range(mod.SHARDS):
            stored = []  # (n_tokens, part, row, item) for this shard's non-anchor rows
            for f in sorted(d.glob(f"shard{shard}_part*.json")):
                pm = json.loads(f.read_text())
                for k, e in enumerate(pm["meta"]):
                    if e["excluded"] is None and not (f.stem.endswith("part00") and k < mod.N_ANCHOR):
                        stored.append((e["n_tokens"], f.stem, k, e["item"]))
            rows = []
            for n_tok, part, k, i in sorted(stored, reverse=True)[:N_LONG]:
                seg = mod.segments(items[i][2])  # the reader's capture code, as in read_shard
                text = "".join(t for t, _ in seg)
                if text != tok.apply_chat_template(items[i][2], tokenize=False, add_generation_prompt=False):
                    raise ValueError(f"{name} item {i}: rendering differs")
                spans, at = [], 0
                for t, is_c in seg:
                    if is_c and t:
                        spans.append((at, at + len(t)))
                    at += len(t)
                enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                idx = [j for j, ((a_, b_), tid) in enumerate(zip(enc["offset_mapping"], enc["input_ids"]))
                       if b_ > a_ and tid not in special and any(s_ <= a_ and b_ <= e_ for s_, e_ in spans)]
                ids, ix, got = torch.tensor([enc["input_ids"]], device="cuda"), torch.tensor(idx, device="cuda"), {}

                def grab(l, o, got=got, ix=ix, last=idx[-1]):
                    h = (o[0] if isinstance(o, tuple) else o)[0]
                    got[l] = torch.stack([h.index_select(0, ix).float().mean(0).to(h.dtype), h[last]])

                hooks = [base.layers[l].register_forward_hook(lambda m_, a_, o, l=l: grab(l, o)) for l in range(64)]
                try:
                    with torch.inference_mode():
                        base(input_ids=ids, use_cache=False)
                finally:
                    for h in hooks:
                        h.remove()
                new = torch.stack([got[l] for l in range(64)], 1).to(torch.float16).cpu().numpy().astype(np.float64)  # (2, 64, 5120)
                old = np.load(d / f"{part}.npy", mmap_mode="r")[k].astype(np.float64)
                cos = (new * old).sum(-1) / (np.linalg.norm(new, axis=-1) * np.linalg.norm(old, axis=-1))
                rows.append({"item": i, "n_tokens": n_tok, "stored_part": part, "min_cosine": float(cos.min()),
                             "max_abs_diff": float(np.abs(new - old).max()), "bit_equal": bool(np.array_equal(new, old))})
            out[f"{name}|shard{shard}"] = rows
    return {"gpu": gpu, "results": out, "passes": all(r["min_cosine"] >= PASS for v in out.values() for r in v)}


@app.local_entrypoint()
def main(hs_stamp: str, cy_stamp: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--", "."):
        raise SystemExit("uncommitted changes")
    head = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit("HEAD is not pushed")
    r = recheck.remote({"highstakes": hs_stamp, "cyber": cy_stamp})
    from datetime import datetime
    from zoneinfo import ZoneInfo

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "RULE.md in this folder", "git": head, "app": APP, "stamps": {"highstakes": hs_stamp, "cyber": cy_stamp}, "pass_cosine": PASS} | r
    (HERE / "results").mkdir(exist_ok=True)
    dest = HERE / "results" / f"longcheck_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    lines = [f"gpu {r['gpu']}; passes {r['passes']}"] + [f"{k}: " + "; ".join(f"item {x['item']} ({x['n_tokens']} tok) cos {x['min_cosine']:.6f} "
                                                         f"maxdiff {x['max_abs_diff']:.3g} bit-equal {x['bit_equal']}" for x in v) for k, v in r["results"].items()]
    dest.with_suffix(".printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(dest.name)
