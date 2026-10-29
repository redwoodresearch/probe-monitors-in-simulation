"""Per-shard spot reproduction of cell (a)'s Qwen3.5 dump on a pinned H200 (RULE.md in this folder).

Re-reads 2 seeded runs from each of the 40 original shards of `2026-10-06_la_segment-end_qwen35-27b_alllayers_v1` with
the as-run extractor's forward pass (run_extract_qwen35.py @ e08de48, copied below line for line) and the as-run reader
(reader_e08de48.py), and compares every stored row at all 64 layers. Nothing is written to the dump; the containers
return numbers and run keys only, and the orchestrator writes one result JSON to the work Volume.
  uv run --no-project --python 3.12 --with modal==1.6.1 modal run [--detach] \\
      experiments/2026-10-14_q2_b200_cella_repro/run_cella_repro.py --git <commit> [--check-only]
--check-only: the CPU selection, Gate S, Gate P and the cost estimate from the original per-run forward seconds; no GPU.
"""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-b200cella_repro_kill-2026-11-14"
WORK = "probe-shift_aswin-surya_sub-q2-b200cella_work_kill-2026-11-14"
DUMP_VOL = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06"
INPUTS = "probe-shift_aswin-surya_sub-la-activations"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
RUN = "2026-10-06_la_segment-end_qwen35-27b_alllayers_v1"
INPUTS_FROM = "2026-10-06_la_segment-end_qwen3-32b_v1"
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
SHARDS, PAD_TO, N_TRANSCRIPTS = 40, 4096, 3933
SEED, PER_SHARD, CONTAINERS = 20261014, 2, 4
COS_MIN, PEAK_REL_MAX = 0.9999, 2.0 ** -6
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]

app = modal.App(APP)
work = modal.Volume.from_name(WORK, create_if_missing=True, version=2)
dump = modal.Volume.from_name(DUMP_VOL, version=2).read_only()
inp = modal.Volume.from_name(INPUTS, version=2).read_only()
hf = modal.Volume.from_name(HF_CACHE)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_file(HERE / "reader_e08de48.py", "/root/reader_e08de48.py"))


def _staged():
    """The staged items in the as-run extractor's order, and each transcript's round-robin shard."""
    src = Path("/in") / INPUTS_FROM / "inputs"
    files = sorted(src.glob("shard_*.jsonl"))
    raw = {p.name: p.read_bytes() for p in files}
    items = [json.loads(l) for b in raw.values() for l in b.decode().splitlines()]
    items.sort(key=lambda it: (-it["tokens"], it["arm"], it["key"]))
    return items, {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}


@app.function(image=image, cpu=8, memory=32768, timeout=1800, retries=0, volumes={"/in": inp, "/dump": dump, "/hf": hf})
def select(check_positions: bool) -> dict:
    import numpy as np

    run_dir = Path("/dump") / RUN
    cfg = json.loads((run_dir / "config.json").read_text())
    for k, v in {"revision": REVISION, "pad_to": PAD_TO, "shards": SHARDS, "inputs_from": INPUTS_FROM, "model": MODEL,
                 "dtype": "bfloat16", "attn_implementation": "sdpa", "max_tokens": None}.items():
        if cfg.get(k) != v:
            raise ValueError(f"stored config {k}={cfg.get(k)!r}, expected {v!r}")
    items, inputs_sha = _staged()
    rr = {(it["arm"], it["key"]): i % SHARDS for i, it in enumerate(items)}
    listed, chunk_of = {}, {}
    for f in sorted((run_dir / "chunks").glob("shard*_chunk*.json")):
        s = int(f.name[5:8])
        for t in json.loads(f.read_text())["transcripts"]:
            listed.setdefault(s, set()).add((t["arm"], t["key"]))
            chunk_of.setdefault((t["arm"], t["key"]), set()).add(f.stem)
    distinct = set().union(*listed.values())
    gate_s = {"distinct transcripts in chunk files": len(distinct),
              "transcripts in more than one chunk file": sum(len(c) > 1 for c in chunk_of.values()),
              "chunk-file shard differs from round-robin shard": sum(rr.get(k) != s for s, ks in listed.items() for k in ks),
              "listed but not staged": len(distinct - set(rr))}
    gate_s["pass"] = (gate_s["distinct transcripts in chunk files"] == N_TRANSCRIPTS
                      and gate_s["chunk-file shard differs from round-robin shard"] == 0 and gate_s["listed but not staged"] == 0)
    stats = {}
    for f in sorted((run_dir / "stats").glob("shard_*.json")):
        st = json.loads(f.read_text())
        stats[f.stem] = sorted(k for k in st if "gpu" in k.lower() or "device" in k.lower())
    rng, picks = np.random.default_rng(SEED), []
    for s in range(SHARDS):
        cand = sorted(listed.get(s, ()))
        for j in sorted(rng.choice(len(cand), PER_SHARD, replace=False).tolist()):
            arm, key = cand[j]
            stem = key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"
            m = json.loads((run_dir / "meta" / arm / f"{stem}.json").read_text())
            if m["chunk"] not in chunk_of[(arm, key)]:
                raise ValueError(f"meta chunk {m['chunk']} is not a chunk file listing {arm}/{stem}")
            kept = [p["token"] for p in m["positions"] if p["kept"]]
            picks.append({"shard": s, "arm": arm, "key": key, "chunk": m["chunk"], "row_start": m["row_start"],
                          "n_rows": m["n_rows"], "n_tokens": m["n_tokens"], "kept_tokens": m["kept_tokens"],
                          "kept_token_idx": kept, "forward_s": m.get("forward_s"), "padded_tokens": m.get("padded_tokens")})
    gate_p = None
    if check_positions:
        import reader_e08de48 as reader
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        by = {(it["arm"], it["key"]): it for it in items}
        bad = []
        for p in picks:
            r = reader.segment_end_positions(tok, by[(p["arm"], p["key"])]["messages"], None)
            if not (r["n_tokens"] == p["n_tokens"] and len(r["ids"]) == p["kept_tokens"]
                    and [q["token"] for q in r["positions"] if q["kept"]] == p["kept_token_idx"]):
                bad.append([p["shard"], p["arm"], p["key"]])
        gate_p = {"runs checked": len(picks), "runs whose positions differ": bad, "pass": not bad}
    loads = [[] for _ in range(CONTAINERS)]  # longest first onto the lightest container
    for p in sorted(picks, key=lambda p: -(p["forward_s"] or 0)):
        min(loads, key=lambda b: sum(q["forward_s"] or 0 for q in b)).append(p)
    return {"stored config": cfg, "inputs sha256": inputs_sha, "gate S": gate_s, "gate P (CPU)": gate_p,
            "stats files and their GPU-like fields": stats, "picks": picks, "batches": loads,
            "original forward seconds of the picks": round(sum(p["forward_s"] or 0 for p in picks), 1),
            "original forward seconds by batch": [round(sum(p["forward_s"] or 0 for p in b), 1) for b in loads]}


@app.function(image=image, gpu="H200", memory=65536, cpu=4, timeout=3600, retries=0,
              volumes={"/in": inp, "/dump": dump, "/hf": hf})
def repro(batch: list[dict]) -> dict:
    import importlib.metadata as md

    import numpy as np
    import reader_e08de48 as reader
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.cuda.get_device_name(0)
    want = {(b["arm"], b["key"]) for b in batch}
    items, _ = _staged()
    by = {(it["arm"], it["key"]): it for it in items if (it["arm"], it["key"]) in want}
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda",
                                              attn_implementation="sdpa").eval()
    base = lm.model
    layers = list(range(lm.config.num_hidden_layers))
    load_s = round(time.time() - t_load)

    def read(ids, idx):  # run_extract_qwen35.py @ e08de48, read(), unchanged
        acts = {}

        def hook_for(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                acts[layer] = h[0].index_select(0, idx).to(torch.float16).cpu().numpy()
            return hook

        handles = [base.layers[l].register_forward_hook(hook_for(l)) for l in layers]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
            torch.cuda.synchronize()
        finally:
            for h in handles:
                h.remove()
        return acts, time.time() - t0, torch.cuda.max_memory_allocated() / 2**30

    out = []
    for b in batch:
        rec = {k: b[k] for k in ("shard", "arm", "key", "chunk", "row_start", "n_rows")}
        r = reader.segment_end_positions(tok, by[(b["arm"], b["key"])]["messages"], None)
        kept = [p for p in r["positions"] if p["kept"]]
        rec["positions identical"] = (r["n_tokens"] == b["n_tokens"] and len(r["ids"]) == b["kept_tokens"]
                                      and [p["token"] for p in kept] == b["kept_token_idx"])
        if not rec["positions identical"]:
            out.append(rec)
            continue
        idx = torch.tensor([p["token"] for p in kept], device="cuda", dtype=torch.long)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = r["ids"] + [pad] * (-len(r["ids"]) % PAD_TO)
        acts, seconds, peak = read(ids, idx)
        per_layer = {}
        for l in layers:
            stored = np.load(Path("/dump") / RUN / "by_layer" / f"L{l}" / f"{b['chunk']}.npy", mmap_mode="r")
            s = np.asarray(stored[b["row_start"]: b["row_start"] + b["n_rows"]], dtype=np.float32)
            n = acts[l].astype(np.float32)
            if s.shape != n.shape:
                raise ValueError(f"{b['arm']}/{b['key']} L{l}: stored {s.shape} vs re-read {n.shape}")
            d = np.abs(n - s)
            ns, nn = np.linalg.norm(s, axis=1), np.linalg.norm(n, axis=1)
            cos = np.where((ns > 0) & (nn > 0), (s * n).sum(1) / np.maximum(ns * nn, 1e-30), (ns == nn).astype(np.float32))
            peak_rel = d.max(1) / np.maximum(np.abs(s).max(1), 1e-30)
            per_layer[str(l)] = {"rows": int(len(s)), "bit-exact rows": int((d.max(1) == 0).sum()),
                                 "max abs diff": float(d.max(initial=0)), "min cosine": float(cos.min(initial=1.0)),
                                 "max peak-relative diff": float(peak_rel.max(initial=0)),
                                 "non-finite": int((~np.isfinite(acts[l])).sum())}
        rec |= {"device": device, "forward_s": round(seconds, 2), "peak_gib": round(peak, 1), "padded_tokens": len(ids),
                "per layer": per_layer}
        out.append(rec)
    versions = {d: md.version(d) for d in ("torch", "transformers", "accelerate", "numpy", "huggingface_hub",
                                           "flash-linear-attention")}
    return {"device": device, "load_s": load_s, "versions": versions | {"cuda": torch.version.cuda}, "runs": out}


def verdicts(runs: list[dict]) -> dict:
    shards = {}
    for r in runs:
        shards.setdefault(r["shard"], []).append(r)
    res = {}
    for s, rs in sorted(shards.items()):
        if len(rs) != PER_SHARD or not all(r["positions identical"] for r in rs):
            res[str(s)] = {"verdict": "INCONCLUSIVE", "runs": len(rs)}
            continue
        L = [v for r in rs for v in r["per layer"].values()]
        exact = all(v["bit-exact rows"] == v["rows"] for v in L)
        ok = all(v["min cosine"] >= COS_MIN and v["max peak-relative diff"] <= PEAK_REL_MAX and v["non-finite"] == 0 for v in L)
        l63 = [r["per layer"]["63"] for r in rs]
        res[str(s)] = {"verdict": "EXACT" if exact else ("PASS" if ok else "FAIL"),
                       "devices": sorted({r["device"] for r in rs}),
                       "min cosine": min(v["min cosine"] for v in L), "max abs diff": max(v["max abs diff"] for v in L),
                       "max peak-relative diff": max(v["max peak-relative diff"] for v in L),
                       "bit-exact rows / rows": [sum(v["bit-exact rows"] for v in L), sum(v["rows"] for v in L)],
                       "L63": {"min cosine": min(v["min cosine"] for v in l63), "max abs diff": max(v["max abs diff"] for v in l63),
                               "bit-exact rows / rows": [sum(v["bit-exact rows"] for v in l63), sum(v["rows"] for v in l63)]},
                       "per layer (min cosine, max abs diff)": {
                           l: [min(r["per layer"][l]["min cosine"] for r in rs), max(r["per layer"][l]["max abs diff"] for r in rs)]
                           for l in rs[0]["per layer"]}}
    counts = {k: sum(v["verdict"] == k for v in res.values()) for k in ("EXACT", "PASS", "FAIL", "INCONCLUSIVE")}
    return {"counts": counts, "shards": res}


@app.function(image=image, cpu=1, memory=4096, timeout=3 * 3600, retries=0, volumes={"/work": work})
def orchestrate(git: str, stamp: str, check_only: bool) -> str:
    sel = select.remote(check_only)
    rec = {"POST HOC": True, "rule": "RULE.md (this folder)", "git": git, "stamp (PT)": stamp, "run": RUN, "seed": SEED,
           "bar": {"cosine >=": COS_MIN, "max peak-relative diff <=": PEAK_REL_MAX}, "selection": sel}
    name = f"cella_repro_{'check_' if check_only else ''}{stamp}.json"
    if not check_only:
        if not sel["gate S"]["pass"]:
            rec["status"] = "stopped: gate S failed"
        else:
            parts = list(repro.map(sel["batches"]))
            runs = [r for p in parts for r in p["runs"]]
            rec |= {"containers": [{k: p[k] for k in ("device", "load_s", "versions")} for p in parts],
                    "runs": runs, "result": verdicts(runs), "status": "preliminary"}
    else:
        rec["status"] = "check only"
    (Path("/work") / "results").mkdir(parents=True, exist_ok=True)
    (Path("/work") / "results" / name).write_text(json.dumps(rec, indent=1))
    work.commit()
    return f"results/{name}"


@app.local_entrypoint()
def main(git: str, check_only: bool = False):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(orchestrate.remote(git, stamp, check_only), flush=True)
