"""RULE.md Amendment 1: the longest-input H200 recheck of a stage-1 stamp, v2 (review fixes on run_longcheck.py). One fresh H200
container re-reads each stored shard's 8 longest items with the job's capture code and compares with the stored rows (cosine per
item, readout and layer; pass >= 0.9999). v2 takes the shard inventory from the stamp's stored part files (not the reader's current
SHARDS), requires exactly 8 comparisons per shard, checks the reader's blob against the stage-1 record, and writes its record to
the Volume from inside the container. Launch detached (0d's rule, 10-14 15:19):

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach \\
      experiments/2026-10-14_p2_longcheck/run_longcheck_v2.py --job cyber --stamp 2026-10-14_135713
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
APP = "probe-shift_aswin-surya_sub-p2probes_longcheck_kill-2026-11-13"
JOBS = {"highstakes": "2026-10-14_p2_highstakes", "cyber": "2026-10-14_p2_cyber"}
N_LONG, PASS, TIMEOUT = 8, 0.9999, 240  # timeout: about 190 s expected + 25% (27B load 40-60 s, two dataset downloads, 8 short items)
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy==2.3.3", "flash-linear-attention==0.5.2", "jinja2"]

app = modal.App(APP)
reads = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_reads_kill-2026-11-13", version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
for name, folder in JOBS.items():
    image = image.add_local_file(EXP / folder / "run_read_fit.py", f"/root/{name}_reader.py")


@app.function(image=image, gpu=["H200"], cpu=(8, 8), memory=(65536, 65536), timeout=TIMEOUT, retries=0,
              volumes={"/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(), "/reads": reads})
def recheck(job: str, stamp: str, run_stamp: str, provenance: dict, want_items: dict) -> dict:
    import importlib

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mod = importlib.import_module(f"{job}_reader")
    d = Path("/reads") / job / stamp
    parts = sorted(d.glob("shard*_part*.json"))
    shards = sorted({int(f.stem.split("_")[0].removeprefix("shard")) for f in parts})
    if not shards:  # v2: the inventory comes from the stored run; an empty one stops the check
        raise ValueError(f"{d}: no stored part files")
    stored_gpus = sorted({json.loads(f.read_text())["gpu"] or "none" for f in parts})
    for s in shards:  # v2 (Codex): a complete, contiguous part set per shard, row counts matching, before choosing the longest
        fs = sorted(d.glob(f"shard{s}_part*.json"))
        if [f.stem for f in fs] != [f"shard{s}_part{p:02d}" for p in range(len(fs))]:
            raise ValueError(f"shard {s}: parts are not contiguous from 00: {[f.stem for f in fs]}")
        n = [len(json.loads(f.read_text())["meta"]) for f in fs]
        if any(np.load(f.with_suffix('.npy'), mmap_mode='r').shape[0] != k for f, k in zip(fs, n)) or sum(n) != want_items[str(s)]:
            raise ValueError(f"shard {s}: {sum(n)} stored rows, the stage-1 record says {want_items[str(s)]}, or an array's rows differ from its meta")
    tok = AutoTokenizer.from_pretrained(mod.MODEL, revision=mod.REV)
    special = set(tok.all_special_ids)
    base = AutoModelForCausalLM.from_pretrained(mod.MODEL, revision=mod.REV, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval().model
    gpu, out = torch.cuda.get_device_name(0), {}
    items, _ = mod.load_items()
    for shard in shards:
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
                raise ValueError(f"{job} item {i}: rendering differs")
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
        if len(rows) != N_LONG:  # v2: never pass on fewer comparisons than planned
            raise ValueError(f"{job} shard {shard}: {len(rows)} comparisons, not {N_LONG}")
        out[f"{job}|shard{shard}"] = rows
    rec = {"label": "RULE.md Amendment 1 in this folder", "stamp (PT)": run_stamp, "app": APP, "job": job, "stage1_stamp": stamp,
           "pass_cosine": PASS, "n_long": N_LONG, "timeout_s": TIMEOUT, "stored_shards": shards, "stored_part_files": [f.name for f in parts],
           "stored_gpus": stored_gpus, "gpu": gpu, "results": out, "passes": all(r["min_cosine"] >= PASS for v in out.values() for r in v)} | provenance
    dest = Path("/reads") / "longcheck" / f"longcheck_{run_stamp}.json"  # 0d's rule: the record is written from inside the job
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    reads.commit()
    return rec


@app.local_entrypoint()
def main(job: str, stamp: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--", "."):
        raise SystemExit("uncommitted changes")
    head = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit("HEAD is not pushed")
    fit = json.loads((EXP / JOBS[job] / "results" / f"read_fit_{stamp}.json").read_text())
    blob = git("rev-parse", f"HEAD:experiments/{JOBS[job]}/run_read_fit.py")
    if fit["blobs_at_launch"]["run_read_fit.py"] != blob:
        raise SystemExit("the reader's blob differs from the stage-1 record's: not the same capture code")
    run_stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    prov = {"git": head, "reader_blob": blob, "stage1_record_git": fit["git"]}
    print(f"launching {APP} {job} {stamp} -> /reads/longcheck/longcheck_{run_stamp}.json; timeout {TIMEOUT} s", flush=True)
    want = {str(x["shard"]): x["items"] for x in fit["shards"]}
    r = recheck.remote(job, stamp, run_stamp, prov | {"stage1_items_per_shard": want}, want)
    dest = HERE / "results" / f"longcheck_{run_stamp}.json"
    dest.open("x").write(json.dumps(r, indent=1))
    lines = [f"gpu {r['gpu']}; stored shards {r['stored_shards']} on {r['stored_gpus']}; passes {r['passes']}"]
    lines += [f"{k}: " + "; ".join(f"item {x['item']} ({x['n_tokens']} tok) cos {x['min_cosine']:.6f} maxdiff {x['max_abs_diff']:.3g} "
                                   f"bit-equal {x['bit_equal']}" for x in v) for k, v in r["results"].items()]
    dest.with_suffix(".printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(dest.name)
