"""B3 (RULE.md here): Seed-OSS-36B, Olmo-3.1-32B or Llama-3.3-70B reading Qwen3.5-27B's acting runs on the sabotage cohort,
staged by stage_readers.py. One forward pass per run (its ids end at its last readout, below the 32,768 cut); the residual
stream after the stored decoder blocks, at two readouts of every staged command row:
- `last/L<l>/<shard>.npy`: the last token wholly inside the command value (part 1);
- `mean/L<l>/<shard>.npy`: the mean over the tokens from the first to the last wholly inside it (part 2).
Row j of both files is row j of the shard's `meta/<shard>.json` (per run: key, arm, run_id, task and its rows' irid, turn,
call_index, label and row). fp16. Written once per shard, under temporary names renamed into place, after every check.

Checks (any failure raises; nothing of that shard is written):
- the staged files' sha256s equal the config's (from the staging record), and the container's tokenizer.json at the
  pinned revision has the sha256 the staging used;
- per run, before the model loads and again before its forward pass: sha256(ids) equals the staged hash, every readout
  lies inside the ids and every span is [first <= last];
- non-finite activations; and in --test: the run read twice (max abs difference), the mean of one row recomputed from the
  stored hidden states in float64 (max abs difference).
Writers: one per shard and attempt (a Modal Dict put-if-absent; a second container of the same attempt stops). A later
--attempt reruns only the shards without a meta file. Function retries are off.

  uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach \\
      experiments/2026-10-13_q3_sabotage_readers/run_extract_readers.py --reader seed|olmo|llama --record <stage record> [--test] [--attempt 1]
"""

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
READER = os.environ.get("SABR_READER", "olmo")
APP = f"probe-shift_aswin-surya_sub-q3sabr_{READER}_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3sabr_acts_kill-2026-11-13"
LEASES = "probe-shift_aswin-surya_sub-q3sabr_leases_kill-2026-11-13"
R = {  # layers: G3's seven (round_half_up(n k / 8), k = 2..7, plus n - 1), which hold each reader's committed reply layers
    "seed": {"model": "ByteDance-Seed/Seed-OSS-36B-Instruct", "revision": "497f1dca95ebdec98e41d517b9f060ee753c902f", "gpu": "H200",
             "layers": [16, 24, 32, 40, 48, 56, 63], "shards": 16, "hidden": 5120, "weights": ("probe-shift_aswin-surya_hf-cache", "/hf")},
    "olmo": {"model": "allenai/Olmo-3.1-32B-Instruct", "revision": "ac0587e4a7744a551c059d8cd17ba220bc940dae", "gpu": "H100!",
             "layers": [16, 24, 32, 40, 48, 56, 63], "shards": 16, "hidden": 5120, "weights": ("probe-shift_aswin-surya_hf-cache", "/hf")},
    "qwen3": {"model": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137", "gpu": "H100!",  # + the cross-read's 39, 51
              "layers": [16, 24, 32, 39, 40, 48, 51, 56, 63], "shards": 16, "hidden": 5120, "weights": ("probe-shift_aswin-surya_hf-cache", "/hf")},
    "llama": {"model": "meta-llama/Llama-3.3-70B-Instruct", "revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b", "gpu": "H100:2",
              "layers": [20, 30, 40, 50, 60, 70, 79], "shards": 12, "hidden": 8192, "weights": ("probe-shift_aswin-surya_sub-apollo_llama-cache", "/llama")},
}[READER]
ENV = ({"HF_HUB_CACHE": "/llama"} if READER == "llama" else {"HF_HOME": "/hf"}) | {"HF_HUB_OFFLINE": "1", "SABR_READER": READER,
                                                                                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
PIP = ["torch==2.8.0", "accelerate==1.10.1", "transformers==4.57.1", "jinja2==3.1.6", "huggingface_hub==0.35.3", "numpy"]
if READER == "qwen3":  # Amendment 2's gate failed at 4.57.1 (min cosine 0.99877 vs the cross-read): the cross-read's own stack and padding
    PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub"]
PAD_TO = 1024 if READER == "qwen3" else 1  # run_extract_crossread.py pads after the last id to a multiple of 1024 (no position read changes)
TIMEOUT_S = 3 * 3600

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, create_if_missing=True, version=2)
leases = modal.Dict.from_name(LEASES, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env(ENV)


def ids_sha(ids) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()


def gate(ep: dict) -> None:
    n = len(ep["ids"])
    bad = ids_sha(ep["ids"]) != ep["ids_sha256"] or any(
        c["readout"] is not None and not (0 < c["span"][0] <= c["span"][1] == c["readout"] < n) for c in ep["calls"])
    if bad:
        raise ValueError(f"gate: {ep['key']} differs from its staged hash or has a readout outside its ids")


def snapshot_file(name: str) -> Path:
    root = Path(R["weights"][1]) / ("" if READER == "llama" else "hub")
    return root / f"models--{R['model'].replace('/', '--')}" / "snapshots" / R["revision"] / name


XREAD = "probe-shift_aswin-surya_sub-q3xread_acts_kill-2026-11-13"  # qwen3 --test: the cross-read's stored layer-63 command rows


@app.function(image=image, gpu=R["gpu"], memory=131072 if READER == "llama" else 65536, timeout=TIMEOUT_S, retries=0,
              volumes={R["weights"][1]: modal.Volume.from_name(R["weights"][0]).read_only(), "/acts": acts_vol}
              | ({"/xr": modal.Volume.from_name(XREAD, version=2).read_only()} if READER == "qwen3" else {}))
def extract_shard(cfg: dict, shard: int) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    me = f"{os.environ.get('MODAL_TASK_ID', '?')}@{time.time():.0f}"
    if not leases.put(f"{cfg['run']}/shard{shard:03d}/a{cfg['attempt']}", me, skip_if_exists=True):
        raise RuntimeError(f"shard {shard} attempt {cfg['attempt']} already has a writer; rerun with a new --attempt")
    acts_vol.reload()
    run_dir, src = Path("/acts") / cfg["run"], Path("/acts") / cfg["inputs"] / "inputs"
    name = f"shard{shard:03d}_a{cfg['attempt']}"
    if list((run_dir / "meta").glob(f"shard{shard:03d}_a*.json")) and not cfg["test"]:  # a smoke always reads
        return {"shard": shard, "skipped": "a meta file exists for this shard"}
    if hashlib.sha256(snapshot_file("tokenizer.json").read_bytes()).hexdigest() != cfg["tokenizer_sha256"]:
        raise ValueError("the snapshot's tokenizer.json differs from the one the staging used")
    raw = {p.name: p.read_bytes() for p in sorted(src.glob("*.jsonl"))}
    if {n: hashlib.sha256(b).hexdigest() for n, b in raw.items()} != cfg["inputs_sha256"]:
        raise ValueError(f"the staged inputs in {src} differ from the config's sha256s")
    eps = sorted((json.loads(line) for b in raw.values() for line in b.decode().splitlines()), key=lambda e: (-len(e["ids"]), e["key"]))
    if cfg["test"] and READER == "qwen3":  # the reproduction gate reads a first-real-run epoch the cross-read stored
        eps = [e for e in eps if e["arm"] == "real1"]
    mine = eps[:1] if cfg["test"] else eps[shard:: cfg["shards"]]
    for ep in mine:
        gate(ep)
    t0 = time.time()
    dmap = "cuda"
    if READER == "llama":  # q2's split by hand: layers 0-39 and the embeddings on GPU 0, the rest on GPU 1
        dmap = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1} | {f"model.layers.{i}": int(i >= 40) for i in range(80)}
    lm = AutoModelForCausalLM.from_pretrained(R["model"], revision=R["revision"], dtype=torch.bfloat16, device_map=dmap,
                                              attn_implementation="sdpa").eval()
    if any(str(v) in ("cpu", "disk") for v in getattr(lm, "hf_device_map", {}).values()):
        raise ValueError("the model did not fit on the GPUs")
    base, L = lm.model, cfg["layers"]
    load_s = round(time.time() - t0)
    pad_id = None
    if PAD_TO > 1:
        from transformers import AutoTokenizer

        tk = AutoTokenizer.from_pretrained(R["model"], revision=R["revision"])
        pad_id = tk.pad_token_id if tk.pad_token_id is not None else tk.eos_token_id

    def read(ids, F, Rr, keep=None):
        acts = {}

        def hook(l):
            def f(m, a, out):
                h = (out[0] if isinstance(out, tuple) else out)[0]
                last = h.index_select(0, torch.tensor(Rr, device=h.device))
                mean = torch.stack([h[a_: b_ + 1].float().mean(0) for a_, b_ in zip(F, Rr)])
                acts[l] = (last.to(torch.float16).cpu().numpy(), mean.to(torch.float16).cpu().numpy())
                if keep is not None and l == L[-1]:
                    keep["h"] = h[keep["a"]: keep["b"] + 1].double().cpu().numpy()
            return f
        hooks = [base.layers[l].register_forward_hook(hook(l)) for l in L]
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([ids + [pad_id] * (-len(ids) % PAD_TO) if PAD_TO > 1 else ids], device="cuda:0"), use_cache=False)
            torch.cuda.synchronize()
        finally:
            for h in hooks:
                h.remove()
        return acts, round(time.time() - t, 2), round(max(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())) / 2**30, 1)

    bufs, metas, stats, test = {(d, l): [] for d in ("last", "mean") for l in L}, [], [], None
    for ep in mine:
        gate(ep)
        rows = [c for c in ep["calls"] if c["readout"] is not None]
        F, Rr = [c["span"][0] for c in rows], [c["span"][1] for c in rows]
        keep = {"a": F[0], "b": Rr[0]} if cfg["test"] else None
        acts, sec, peak = read(ep["ids"], F, Rr, keep)
        if cfg["test"]:
            again, sec2, _ = read(ep["ids"], F, Rr)
            ref = keep["h"].mean(0)
            test = {"tokens": len(ep["ids"]), "rows": len(rows), "forward_s": sec, "forward_s_repeat": sec2, "peak_gib": peak, "load_s": load_s,
                    "repeat_max_abs_diff": max(float(np.abs(acts[l][k].astype(np.float32) - again[l][k].astype(np.float32)).max()) for l in L for k in (0, 1)),
                    "mean_vs_float64_max_abs_diff": float(np.abs(acts[L[-1]][1][0].astype(np.float64) - ref).max()),
                    "mean_row0_tokens": Rr[0] - F[0] + 1}
            if READER == "qwen3":  # the cross-read's layer-63 command rows for the same calls: cosine >= 0.999 per row passes
                xd = Path("/xr/2026-10-13_xread_main_v1")
                xm = json.loads((xd / "meta" / "attack" / f"{hashlib.sha1(ep['key'].encode()).hexdigest()[:20]}.json").read_text())
                xrow = {(c["irid"], c["call_index"]): xm["row_start"] + c["row"] for c in xm["calls"] if c.get("kind") == "command" and c.get("row") is not None}
                X = np.load(xd / "by_layer" / "L63" / f"{xm['chunk']}.npy", mmap_mode="r")
                pairs = [(j, xrow[(c["irid"], c["call_index"])]) for j, c in enumerate(rows) if (c["irid"], c["call_index"]) in xrow]
                a = np.stack([acts[63][0][j].astype(np.float64) for j, _ in pairs])
                b = np.stack([np.asarray(X[r], np.float64) for _, r in pairs])
                cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
                test["xread_l63"] = {"rows_compared": len(pairs), "rows_staged": len(rows), "max_abs_diff": float(np.abs(a - b).max()),
                                     "min_cosine": float(cos.min()), "passes": bool(len(pairs) == len(rows) and cos.min() >= 0.999)}
                if not test["xread_l63"]["passes"]:  # Codex 22:11: the gate stops the smoke, not just records it
                    raise ValueError(f"Qwen3-32B reproduction gate failed: {test['xread_l63']}")
        for l in L:
            for k, d in enumerate(("last", "mean")):
                if not np.isfinite(acts[l][k]).all():
                    raise ValueError(f"{ep['key']}: non-finite activations; nothing of this shard is written")
                bufs[(d, l)].append(acts[l][k])
        start = sum(len(m["rows"]) for m in metas)
        metas.append({k: ep[k] for k in ("key", "arm", "run", "run_id", "task", "n_tokens", "ids_sha256")} | {"forward_tokens": len(ep["ids"]), "forward_s": sec,
                     "peak_gib": peak, "rows": [{k: c[k] for k in ("irid", "turn", "call_index", "name", "label", "readout", "span")} | {"row": start + j}
                                                for j, c in enumerate(rows)]})
        stats.append(sec)
    for d in ("last", "mean"):
        for l in L:
            (run_dir / d / f"L{l}").mkdir(parents=True, exist_ok=True)
            x = np.concatenate(bufs[(d, l)]) if bufs[(d, l)] else np.zeros((0, R["hidden"]), np.float16)
            tmp = run_dir / d / f"L{l}" / f".{name}.npy.tmp"
            with open(tmp, "wb") as fh:
                np.save(fh, x)
            os.replace(tmp, run_dir / d / f"L{l}" / f"{name}.npy")
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "meta" / f".{name}.json.tmp"
    tmp.write_text(json.dumps({"shard": shard, "attempt": cfg["attempt"], "writer": me, "layers": L, "runs": metas}))
    os.replace(tmp, run_dir / "meta" / f"{name}.json")
    acts_vol.commit()
    return {"shard": shard, "runs": len(mine), "rows": sum(len(m["rows"]) for m in metas), "load_s": load_s,
            "forward_s": round(sum(stats), 1), "seconds": round(time.time() - t0), "test": test}


@app.local_entrypoint()
def main(reader: str, record: str, test: bool = False, attempt: int = 1, shards: int = 0, smoke: str = ""):
    if reader != READER:
        raise SystemExit(f"set SABR_READER={reader} (the app, GPU and weights are chosen at import)")
    rec = json.loads(Path(record).read_text())
    if reader == "qwen3" and not test:  # Codex 22:11: a full Qwen3-32B run needs a passed smoke on the same staged inputs
        sm = json.loads(Path(smoke).read_text()) if smoke else {}
        t = ((sm.get("shards") or [{}])[0].get("test") or {}).get("xread_l63") or {}
        if not t.get("passes") or sm["cfg"]["inputs_sha256"] != rec["staged_inputs_sha256"]:
            raise SystemExit("pass --smoke <a passed Qwen3-32B smoke record on these staged inputs>")
    if rec["reader"] != reader or rec.get("gate") or rec.get("limit"):
        raise SystemExit("the record is not this reader's full staging")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    if not test and git("status", "--porcelain", "--", str(Path(__file__).name), "RULE.md", "SPEND.md"):
        raise SystemExit("commit this script, RULE.md and SPEND.md before a launch")
    local = (Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad/llama_tok") / R["revision"] if reader == "llama" else  # stage_readers.LLAMA_TOK
             Path.home() / f".cache/huggingface/hub/models--{R['model'].replace('/', '--')}/snapshots/{R['revision']}")
    tok_sha = hashlib.sha256((local / "tokenizer.json").read_bytes()).hexdigest()
    sname = Path(rec["staged_dir"]).name
    run = f"2026-10-13_{sname}" if "_repair_" in sname or reader == "qwen3" else f"2026-10-13_sabreaders_{reader}_v1"
    cfg = {"run": run + ("_test" if test else ""), "inputs": sname, "inputs_sha256": rec["staged_inputs_sha256"],
           "tokenizer_sha256": tok_sha, "layers": R["layers"], "shards": shards or R["shards"], "attempt": attempt, "test": test}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"cfg": cfg, "reader": R, "git": git("rev-parse", "HEAD"), "launched": stamp, "app": APP, "record": Path(record).name}
    res = list(extract_shard.map([cfg] * (1 if test else cfg["shards"]), range(1 if test else cfg["shards"])))
    out["shards"] = res
    dst = HERE / "results" / f"extract_{reader}{'_test' if test else ''}_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(res if not test else res[0]["test"], indent=1), "\n->", dst)
