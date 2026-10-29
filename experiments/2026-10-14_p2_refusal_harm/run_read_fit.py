"""RULE.md stages 1 and 1b: read Arditi et al.'s harmful and harmless prompts with Qwen3.5-27B (one Modal H100) at the last
instruction token ("inst") and the last template token ("post"), every layer; fit the harmfulness (inst) and refusal (post)
directions by difference in means on train; choose each one's layer on val (layers 0-50); gate on val AUROC >= 0.9. The
local entrypoint checks git, launches, and writes the record and the two directions (safetensors) to results/.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach experiments/2026-10-14_p2_refusal_harm/run_read_fit.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-p2probes_refread_kill-2026-11-13"
READS = "probe-shift_aswin-surya_sub-p2probes_reads_kill-2026-11-13"
MODEL, REV = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
SRC = "https://raw.githubusercontent.com/andyrdt/refusal_direction/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/splits/"
BLOBS = {"harmful_train": "5ca6b46750e06bc401ebd93171a6b0dc0590cdd8", "harmful_val": "c3128d37e8b255e765f4f51f1a6715572595e3e7",
         "harmful_test": "2dc705dc1a50e7773efca46fedab71229169b3bb", "harmless_train": "700a497bd1d20ab074fcc576e9bd79ac604543c5",
         "harmless_val": "6b9ee6e9c789799354b618a046758276da445bd8", "harmless_test": "6033b711a3c0bf3d49fb88a4824fdac8be792f25"}
N = {"train": 128, "val": 32, "test": 64}
CAND = range(51)  # Arditi et al.: exclude the last 20% of layers (64 layers)
POS = ("inst", "post")
DIRS = {"harmfulness": "inst", "refusal": "post"}
GATE = 0.9
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy==2.3.3", "safetensors==0.8.0", "flash-linear-attention==0.5.2"]

app = modal.App(APP)
reads = modal.Volume.from_name(READS, create_if_missing=True, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})


def _auroc(pos, neg):  # ties count one half
    import numpy as np

    s = np.concatenate([pos, neg])
    o = np.argsort(s, kind="mergesort")
    first, cnt = np.unique(s[o], return_index=True, return_counts=True)[1:]
    rk = np.empty(len(s))
    rk[o] = np.repeat(first + (cnt - 1) / 2 + 1, cnt)
    return float((rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@app.function(image=image, gpu="H100", cpu=(8, 8), memory=(98304, 98304), timeout=1800, retries=0,
              volumes={"/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(), "/reads": reads})
def read_fit(stamp: str) -> dict:
    import random
    import time
    import urllib.request
    from collections import Counter

    import numpy as np
    import safetensors.numpy as stn
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = {}
    for name, blob in BLOBS.items():  # data only: parsed as JSON, never executed
        raw = urllib.request.urlopen(SRC + f"{name}.json", timeout=60).read()
        if hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest() != blob:
            raise ValueError(f"{name}: git blob sha1 differs from the pin")
        data[name] = [x["instruction"] for x in json.loads(raw)]
    rng, items = random.Random(20261014), []
    for split, n in N.items():
        for lab, kind in ((1, "harmful"), (0, "harmless")):
            items += [(split, lab, t) for t in rng.sample(data[f"{kind}_{split}"], n)]
    texts = [t for _, _, t in items]
    overlap = len(texts) - len(set(texts))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    base, L = lm.model, lm.config.num_hidden_layers
    load_s = round(time.time() - t0)
    start_id, end_id, special = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>"), set(tok.all_special_ids)
    X = np.zeros((len(items), len(POS), L, 5120), np.float16)  # Qwen3.5-27B hidden size; a different width fails on assignment
    pos_tok, ntok, where = {p: Counter() for p in POS}, [], []
    t1 = time.time()
    for i, text in enumerate(texts):
        ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"]
        u = [j for j in range(len(ids) - 1) if ids[j] == start_id and tok.decode([ids[j + 1]]).strip() == "user"]
        ends = [j for j in range(u[-1] + 2, len(ids)) if ids[j] == end_id] if u else []
        if len(u) != 1 or not ends or ends[0] - 1 <= u[0] + 2 or ids[ends[0] - 1] in special:
            raise ValueError(f"item {i}: no single user turn with a content token before its <|im_end|>")
        at = (ends[0] - 1, len(ids) - 1)
        where.append(list(at))
        for p, j in zip(POS, at):
            pos_tok[p][tok.convert_ids_to_tokens(ids[j])] += 1
        ntok.append(len(ids))
        rows = {}
        idx = torch.tensor(at, device="cuda")

        def grab(l, out, rows=rows, idx=idx):
            rows[l] = (out[0] if isinstance(out, tuple) else out)[0].index_select(0, idx).to(torch.float16).cpu().numpy()

        hooks = [base.layers[l].register_forward_hook(lambda m, a, out, l=l: grab(l, out)) for l in range(L)]
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        for l in range(L):
            X[i, :, l] = rows[l]
    fwd_s = round(time.time() - t1)
    if not np.isfinite(X).all() or (np.abs(X).max(axis=-1) == 0).any():
        raise ValueError("a non-finite or all-zero row")
    out = Path("/reads") / "refusal" / stamp
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "rows.npy", X)
    (out / "items.json").write_text(json.dumps([{"split": s, "label": y, "text_sha256": hashlib.sha256(t.encode()).hexdigest(), "n_tokens": n, "positions": w}
                                                for (s, y, t), n, w in zip(items, ntok, where)]))
    reads.commit()
    split, lab = np.array([s for s, _, _ in items]), np.array([y for _, y, _ in items])
    res, files = {}, {}
    for name, p in DIRS.items():
        pi, per, units = POS.index(p), {}, {}
        for l in range(L):
            tr = X[split == "train", pi, l].astype(np.float32)
            y = lab[split == "train"]
            r = tr[y == 1].mean(0) - tr[y == 0].mean(0)
            units[l] = (r / np.linalg.norm(r)).astype(np.float32)
            m = {}
            for s_ in ("val", "test"):
                sc, ys = X[split == s_, pi, l].astype(np.float32) @ units[l], lab[split == s_]
                a, b = sc[ys == 1], sc[ys == 0]
                m[f"{s_}_auroc"] = _auroc(a, b)
                m[f"{s_}_d"] = float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
            per[l] = m
        ch = max(CAND, key=lambda l: (per[l]["val_auroc"], per[l]["val_d"]))
        res[name] = {"position": p, "layer": ch, "chosen": per[ch], "passes": per[ch]["val_auroc"] >= GATE, "per_layer": per}
        files[name] = stn.save({"direction": units[ch]}, metadata={"layer": str(ch), "position": p, "model": f"{MODEL}@{REV}", "stamp": stamp})
    return {"directions": res, "files": files, "position_tokens": {p: dict(c.most_common()) for p, c in pos_tok.items()},
            "prompt_tokens": {"min": min(ntok), "median": float(np.median(ntok)), "max": max(ntok)}, "duplicate_texts": overlap,
            "pool_sizes": {k: len(v) for k, v in data.items()}, "load_s": load_s, "forward_s": fwd_s, "reads": f"{READS}:/refusal/{stamp}"}


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    head = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit("HEAD is not pushed")
    blobs = {}
    for f in ("RULE.md", "run_read_fit.py"):  # both must be committed and unchanged before any GPU time (Codex)
        if git("status", "--porcelain", "--", f) or not git("ls-files", "--", f):
            raise SystemExit(f"{f} is not committed")
        blobs[f] = git("rev-parse", f"{head}:experiments/2026-10-14_p2_refusal_harm/{f}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    r = read_fit.remote(stamp)
    rec = {"label": "PRELIMINARY until Codex and an independent check; RULE.md in this folder", "git": head, "stamp (PT)": stamp, "app": APP,
           "blobs_at_launch": blobs, "source": SRC, "blobs": BLOBS, "n": N,
           "gate": f"val AUROC >= {GATE} at the chosen layer (layers 0-50; max val AUROC, then val Cohen's d)"}
    rec |= {k: v for k, v in r.items() if k != "files"}
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    for name, b in r["files"].items():
        f = res / f"{name}_direction_{stamp}.safetensors"
        f.open("xb").write(b)
        rec["directions"][name]["file"] = {"path": f.name, "sha256": hashlib.sha256(b).hexdigest()}
    dest = res / f"read_fit_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    lines = [f"load {r['load_s']} s, forward {r['forward_s']} s; prompt tokens {r['prompt_tokens']}; duplicate texts {r['duplicate_texts']}",
             f"position tokens {r['position_tokens']}"]
    lines += [f"{n}: layer {d['layer']} ({d['position']}), val AUROC {d['chosen']['val_auroc']:.3f} d {d['chosen']['val_d']:.2f}; "
              f"test AUROC {d['chosen']['test_auroc']:.3f}; gate passes {d['passes']}" for n, d in r["directions"].items()]
    dest.with_suffix(".printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(dest.name)
    if not all(d["passes"] for d in r["directions"].values()):
        raise SystemExit("a stage-1 gate failed: stop, no B scoring (RULE.md)")
