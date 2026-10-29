"""RULE.md stages 1 and 1b: read McKenzie et al.'s high/low-stakes data with Qwen3.5-27B on two H200 shards (mean over content
tokens and the last content token, every layer; each shard also reads the same 16 anchor items), then on CPU: check the shards
agree on the anchors, fit a logistic probe per layer on the mean readout, choose the layer on a seeded held-out 20% of train, gate
on test AUROC >= 0.85. The local entrypoint checks git, launches, and writes the record and the probe (safetensors) to results/.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach experiments/2026-10-14_p2_highstakes/run_read_fit.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-p2probes_hsread_kill-2026-11-13"
READS = "probe-shift_aswin-surya_sub-p2probes_reads_kill-2026-11-13"
MODEL, REV = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
SRC = "https://huggingface.co/datasets/Arrrlex/models-under-pressure/resolve/940d830f639617752f54fdf2bca0da7e0d856a9e/"
FILES = {"train": ("training/train.jsonl", "04216df3a8b3ebf3ffb617f1a6701b0429964d28a6c6df677896359891fe2f34"),
         "test": ("training/test.jsonl", "6af73f76248f46547cfbf346ba15cf0fafb5eac3986faae43de61c5e1660621f"),
         "hh": ("anthropic_hh_balanced/test.jsonl", "9ea4beff6de77023044373f44c67de1dbea09708e2af63d9dfcaa964998daf1d")}
LABEL = {"high-stakes": 1, "low-stakes": 0}
SHARDS, N_ANCHOR, PART, GATE = 2, 16, 1000, 0.85
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy==2.3.3", "flash-linear-attention==0.5.2", "jinja2"]

app = modal.App(APP)
reads = modal.Volume.from_name(READS, create_if_missing=True, version=2)
gpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
cpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scikit-learn==1.7.2", "joblib", "safetensors==0.8.0")


def load_items() -> tuple:
    """Every high/low-stakes item of the three pinned files, in file order: (split, label, messages); and other-label counts."""
    import urllib.request
    from collections import Counter

    items, other = [], Counter()
    for split, (path, want) in FILES.items():
        raw = urllib.request.urlopen(SRC + path, timeout=120).read()
        if hashlib.sha256(raw).hexdigest() != want:
            raise ValueError(f"{path}: sha256 differs from the pinned LFS sha256")
        for line in raw.decode().splitlines():  # data only: parsed as JSON, never executed
            row = json.loads(line)
            if row["labels"] not in LABEL:
                other[f"{split}|{row['labels']}"] += 1
                continue
            msgs = json.loads(row["inputs"]) if split == "hh" else [{"role": "user", "content": row["inputs"]}]
            items.append((split, LABEL[row["labels"]], msgs))
    return items, dict(other)


def segments(msgs: list) -> list | None:
    """The pinned Qwen3.5 template's rendering of a tool-free chat without a generation prompt, as (text, is_content)
    segments: role headers and delimiters are template text; each message's trimmed content is content. None when the
    template would transform a message (a think block inside assistant content, a non-string content, a misplaced
    system turn); the caller then excludes the item. The caller asserts the joined text equals the official rendering."""
    if any(not isinstance(m.get("content"), str) or m.get("role") not in ("system", "user", "assistant") for m in msgs):
        return None
    if any(m["role"] == "system" for m in msgs[1:]) or not any(m["role"] == "user" for m in msgs):
        return None
    last_user = max(j for j, m in enumerate(msgs) if m["role"] == "user")
    seg = []
    for j, m in enumerate(msgs):
        c = m["content"].strip()
        if m["role"] == "assistant" and "</think>" in c:
            return None
        head = f"<|im_start|>{m['role']}\n" + ("<think>\n\n</think>\n\n" if m["role"] == "assistant" and j > last_user else "")
        seg += [(head, False), (c, True), ("<|im_end|>\n", False)]
    return seg


def shard_plan(items: list) -> dict:
    """Per shard, its item list (the 16 anchors first) and its parts of PART items; the fit requires exactly these."""
    test0 = [i for i, it in enumerate(items) if it[0] == "test"][:N_ANCHOR]
    mine = {s: test0 + [i for i in range(len(items)) if i % SHARDS == s] for s in range(SHARDS)}
    return {"anchors": test0, "parts": {s: [m[p: p + PART] for p in range(0, len(m), PART)] for s, m in mine.items()}}


@app.function(image=gpu_image, gpu=["H200"], cpu=(8, 8), memory=(65536, 65536), timeout=1500, retries=0,
              volumes={"/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(), "/reads": reads})
def read_shard(stamp: str, shard: int, script_blob: str) -> dict:
    """This shard's items (the 16 anchors first), in parts of PART items; each finished part is saved and committed, and a
    rerun skips parts already on the Volume. Per item, every layer: the mean over content tokens (float32 sum, cast to bf16
    as extract_q4b.py does, then fp16) and the last content token."""
    import time

    import jinja2
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items, other = load_items()
    parts = shard_plan(items)["parts"][shard]
    mine = [i for pt in parts for i in pt]
    d = Path("/reads") / "highstakes" / stamp
    d.mkdir(parents=True, exist_ok=True)
    cfg = d / f"shard{shard}_config.json"  # a resumed stamp must come from the same script
    if cfg.exists() and json.loads(cfg.read_text())["script_blob"] != script_blob:
        raise ValueError(f"stamp {stamp} was started by another version of this script")
    cfg.write_text(json.dumps({"script_blob": script_blob}))
    todo = [p for p in range(len(parts)) if not (d / f"shard{shard}_part{p:02d}.json").exists()]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    special = set(tok.all_special_ids)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval() if todo else None
    base, gpu = (lm.model, torch.cuda.get_device_name(0)) if todo else (None, None)
    load_s, t1, lens = round(time.time() - t0), time.time(), []
    for p in todo:
        out, meta = np.zeros((len(parts[p]), 2, 64, 5120), np.float16), []  # [:, 0] content mean, [:, 1] last content token
        for k, i in enumerate(parts[p]):
            seg = segments(items[i][2])
            text = "".join(t for t, _ in seg) if seg else None
            try:
                official = tok.apply_chat_template(items[i][2], tokenize=False, add_generation_prompt=False) if seg else None
            except jinja2.TemplateError:  # e.g. "No user query found" (a chat whose only user turn is a tool response)
                seg, official = None, None
            if seg is None or text != official:
                meta.append({"item": i, "excluded": "template transform or error" if seg is None else "rendering differs from the template"})
                continue
            spans, at = [], 0
            for t, is_c in seg:
                if is_c and t:
                    spans.append((at, at + len(t)))
                at += len(t)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            idx = [j for j, ((a_, b_), tid) in enumerate(zip(enc["offset_mapping"], enc["input_ids"]))
                   if b_ > a_ and tid not in special and any(s_ <= a_ and b_ <= e_ for s_, e_ in spans)]
            if not idx:
                meta.append({"item": i, "excluded": "no content token"})
                continue
            ids, ix, got = torch.tensor([enc["input_ids"]], device="cuda"), torch.tensor(idx, device="cuda"), {}

            def grab(l, o, got=got, ix=ix, last=idx[-1]):
                h = (o[0] if isinstance(o, tuple) else o)[0]
                got[l] = torch.stack([h.index_select(0, ix).float().mean(0).to(h.dtype), h[last]])

            hooks = [base.layers[l].register_forward_hook(lambda mod, a, o, l=l: grab(l, o)) for l in range(64)]
            try:
                with torch.inference_mode():
                    base(input_ids=ids, use_cache=False)
            finally:
                for h in hooks:
                    h.remove()
            out[k] = torch.stack([got[l] for l in range(64)], 1).to(torch.float16).cpu().numpy()  # one device-to-host copy per item
            meta.append({"item": i, "excluded": None, "n_tokens": len(enc["input_ids"]), "n_content": len(idx)})
            lens.append(len(enc["input_ids"]))
        if not np.isfinite(out).all():
            raise ValueError(f"shard {shard} part {p}: non-finite rows")
        np.save(d / f"shard{shard}_part{p:02d}.npy", out)
        (d / f"shard{shard}_part{p:02d}.json").write_text(json.dumps({"items": parts[p], "meta": meta, "gpu": gpu}))  # written last: marks the part done
        reads.commit()
        print(f"shard {shard}: part {p + 1}/{len(parts)} saved, {time.time() - t1:.0f} s", flush=True)
    return {"shard": shard, "gpu": gpu, "items": len(mine), "parts_done_before": len(parts) - len(todo), "load_s": load_s,
            "forward_s": round(time.time() - t1), "n_items_total": len(items), "other_labels": other,
            "tokens": {"min": min(lens), "median": float(np.median(lens)), "max": max(lens)} if lens else None}


@app.function(image=cpu_image, cpu=(16, 16), memory=(65536, 65536), timeout=1200, retries=0, volumes={"/reads": reads})
def fit(stamp: str) -> dict:
    import random

    import numpy as np
    import safetensors.numpy as stn
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    reads.reload()
    items, _ = load_items()
    d = Path("/reads") / "highstakes" / stamp
    plan = shard_plan(items)
    want = {f"shard{s}_part{p:02d}": pt for s, ps in plan["parts"].items() for p, pt in enumerate(ps)}
    have = {f.stem for f in d.glob("shard*_part*.json")}
    if have != set(want):
        raise ValueError(f"part inventory differs: missing {sorted(set(want) - have)}, extra {sorted(have - set(want))}")
    A, M, where = {}, {}, {}
    for sp_ in sorted(want):
        A[sp_], M[sp_] = np.load(d / f"{sp_}.npy", mmap_mode="r"), json.loads((d / f"{sp_}.json").read_text())
        if M[sp_]["items"] != want[sp_] or len(M[sp_]["meta"]) != len(want[sp_]) or [m["item"] for m in M[sp_]["meta"]] != want[sp_] \
                or A[sp_].shape != (len(want[sp_]), 2, 64, 5120):
            raise ValueError(f"{sp_}: items, meta or array shape differ from the plan")
    gpus = {m["gpu"] for m in M.values()}
    first = [f"shard{s}_part00" for s in range(SHARDS)]
    if len(gpus) != 1 or any(M[f]["items"][:N_ANCHOR] != plan["anchors"] for f in first):
        raise ValueError(f"anchor items differ from the plan, or GPUs {gpus}")
    if any(m["excluded"] for f in first for m in M[f]["meta"][:N_ANCHOR]):
        raise ValueError("an anchor item was excluded")
    anc = [np.asarray(A[f][:N_ANCHOR]).astype(np.float64) for f in first]  # (16, 2, 64, 5120): both readouts, every layer
    na, nb = np.linalg.norm(anc[0], axis=-1), np.linalg.norm(anc[1], axis=-1)
    cosv = (anc[0] * anc[1]).sum(-1) / (na * nb)
    if not (np.isfinite(anc[0]).all() and np.isfinite(anc[1]).all() and (na > 0).all() and (nb > 0).all() and np.isfinite(cosv).all()):
        raise ValueError("non-finite or zero anchor rows")
    cos = float(cosv.min())
    if cos < 0.9999:
        raise ValueError(f"shards disagree on the anchors: min cosine {cos}")
    excluded = {}
    for f, m in M.items():
        for k, (i, e) in enumerate(zip(m["items"], m["meta"])):
            if f.endswith("part00") and k < N_ANCHOR:
                continue
            if e["excluded"]:
                excluded[e["excluded"]] = excluded.get(e["excluded"], 0) + 1
            elif i in where:
                raise ValueError(f"item {i} appears twice")
            else:
                where[i] = (f, k)
    split = {sp: [i for i, it in enumerate(items) if it[0] == sp and i in where] for sp in FILES}
    y = {sp: np.array([items[i][1] for i in split[sp]]) for sp in FILES}
    tr = list(range(len(split["train"])))
    rng = random.Random(20261014)
    ho = set()
    for c in (0, 1):  # stratified 20% held out of train
        idx = [j for j in tr if y["train"][j] == c]
        ho |= set(rng.sample(idx, round(0.2 * len(idx))))
    fit_i, ho_i = np.array([j for j in tr if j not in ho]), np.array(sorted(ho))

    rows = {sp: [where[i] for i in split[sp]] for sp in FILES}

    def X(sp, l):  # (items, 5120) float32 content-mean rows at layer l, gathered per part
        out = np.empty((len(rows[sp]), 5120), np.float32)
        for f in A:
            pos = [j for j, (ff, _) in enumerate(rows[sp]) if ff == f]
            if pos:
                out[pos] = A[f][[rows[sp][j][1] for j in pos], 0, l]
        return out

    def one(l):
        Xtr, Xte, Xhh = X("train", l), X("test", l), X("hh", l)
        mu, sd = Xtr[fit_i].mean(0), Xtr[fit_i].std(0) + 1e-6
        clf = LogisticRegression(C=1.0, max_iter=2000, tol=1e-6).fit((Xtr[fit_i] - mu) / sd, y["train"][fit_i])
        dec = lambda Z: clf.decision_function((Z - mu) / sd)
        return l, {"heldout_auroc": float(roc_auc_score(y["train"][ho_i], dec(Xtr[ho_i]))), "n_iter": int(clf.n_iter_[0]),
                   "test_auroc (descriptive)": float(roc_auc_score(y["test"], dec(Xte))), "hh_auroc (descriptive)": float(roc_auc_score(y["hh"], dec(Xhh)))}

    per = dict(Parallel(n_jobs=16)(delayed(one)(l) for l in range(64)))
    ch = max(per, key=lambda l: per[l]["heldout_auroc"])
    Xtr, Xte, Xhh = X("train", ch), X("test", ch), X("hh", ch)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=2000, tol=1e-6).fit((Xtr - mu) / sd, y["train"])
    w, b = clf.coef_[0].astype(np.float32), float(clf.intercept_[0])
    r, c = (w / sd).astype(np.float32), np.float32(b - float((mu * w / sd).sum()))  # decision = x·r + c
    test_auc, hh_auc = float(roc_auc_score(y["test"], Xte @ r + c)), float(roc_auc_score(y["hh"], Xhh @ r + c))
    f = stn.save({"direction": r, "bias": np.array([c], np.float32), "W": w, "intercept": np.array([b], np.float32), "mean": mu.astype(np.float32), "std": sd.astype(np.float32)},
                 metadata={"layer": str(ch), "readout": "mean over content tokens", "model": f"{MODEL}@{REV}", "stamp": stamp})
    return {"anchor_min_cosine": cos, "gpu": gpus.pop(), "excluded": excluded, "n": {sp: len(split[sp]) for sp in FILES}, "n_heldout": len(ho_i), "per_layer": per,
            "directions": {"highstakes": {"position": "mean over content tokens", "layer": ch, "passes": test_auc >= GATE,
                                          "chosen": {"heldout_auroc": per[ch]["heldout_auroc"], "test_auroc": test_auc, "hh_auroc (descriptive)": hh_auc,
                                                     "n_iter_refit": int(clf.n_iter_[0]), "refit_hit_max_iter": int(clf.n_iter_[0]) >= 2000}}}, "file": f}


@app.local_entrypoint()
def main(resume_stamp: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    head = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit("HEAD is not pushed")
    blobs = {}
    for f in ("RULE.md", "run_read_fit.py"):  # committed and unchanged before any GPU time
        if git("status", "--porcelain", "--", f) or not git("ls-files", "--", f):
            raise SystemExit(f"{f} is not committed")
        blobs[f] = git("rev-parse", f"{head}:experiments/2026-10-14_p2_highstakes/{f}")
    stamp = resume_stamp or datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")  # a resumed stamp skips finished parts
    shards = list(read_shard.starmap([(stamp, s, blobs["run_read_fit.py"]) for s in range(SHARDS)]))
    r = fit.remote(stamp)
    rec = {"label": "PRELIMINARY until Codex and an independent check; RULE.md in this folder", "git": head, "stamp (PT)": stamp, "app": APP,
           "blobs_at_launch": blobs, "source": SRC, "files": FILES, "shards": shards, "reads": f"{READS}:/highstakes/{stamp}",
           "gate": f"test AUROC >= {GATE} at the layer with the best held-out-20% AUROC (logistic, C=1, mean readout)"}
    rec |= {k: v for k, v in r.items() if k != "file"}
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    f = res / f"highstakes_probe_{stamp}.safetensors"
    f.open("xb").write(r["file"])
    rec["directions"]["highstakes"]["file"] = {"path": f.name, "sha256": hashlib.sha256(r["file"]).hexdigest()}
    dest = res / f"read_fit_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    d = rec["directions"]["highstakes"]
    lines = [f"shards {[(s['shard'], s['gpu'], s['items'], s['load_s'], s['forward_s'], s['tokens']) for s in shards]}; other labels {shards[0]['other_labels']}",
             f"anchor min cosine {r['anchor_min_cosine']:.6f} (both readouts, every layer); excluded {r['excluded']}; n {r['n']}; held out {r['n_heldout']}",
             (f"highstakes: layer {d['layer']}, held-out AUROC {d['chosen']['heldout_auroc']:.3f}; test AUROC {d['chosen']['test_auroc']:.3f}; "
              f"HH AUROC {d['chosen']['hh_auroc (descriptive)']:.3f}; gate passes {d['passes']}"),
             "max_iter hit at layers: " + str([l for l, v in r["per_layer"].items() if v["n_iter"] >= 2000]) + f"; refit hit max_iter: {d['chosen']['refit_hit_max_iter']}"]
    dest.with_suffix(".printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(dest.name)
    if not d["passes"]:
        raise SystemExit("the stage-1 gate failed: stop, no B scoring (RULE.md)")
