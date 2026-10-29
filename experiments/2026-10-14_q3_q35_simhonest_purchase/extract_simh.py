"""Simulated-honest purchase (experiments/2026-10-14_q3_q35_simhonest_purchase/RULE.md, gate 4 and amendment 5): a copy of
2e's extract_q4b.py @ 0bf6ec95 whose only changes are: this job's app, lease and Volume names (staged inputs and outputs
on probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13); and the PER-SHARD REPRODUCTION GATE: after the model
loads and before any of its own epochs, every shard reads batch C's CORRECTED staged epochs (2e's q4f_c_v1, extracted as
q4b_extract_c_v1, on its Volume mounted read-only; RULE.md amendment 11: uncorrected B is no reference): 2 drawn with
seed 20261014 from C's staged keys and C's LONGEST epoch by token count, and compares every row, at every layer, with
C's stored rows: cosine >= 0.9999 per row (bit-exact expected on H200); and it reads its OWN longest epoch twice, cosine
>= 0.9999 per row between the two reads (amendment 5: long contexts are where hardware numerics broke). Any failure
raises before the shard writes anything. gpu="H200" only, as the original. The original docstring follows.

Q4 (B), RULE.md Amendment 5: a copy of d9's run_extract_crossread.py (aswin/q3-qwen3-crossread-score @ 3188dc6e) whose only
changes are: the reader Qwen3.5-27B (probe_shift.models QWEN35_27B) on one H200 per shard with flash-linear-attention, as the
original run_extract_replay.py; this job's app, Volume and lease names; the Volume (staged inputs and outputs) set by
Q4OWN_VOLUME; and, beside each output_end last-token row, an output_end_mean row at every layer: the mean of the residual
stream over the output's content span [start, end) staged by stage_q4b.py (meta: each output_end call's `row_mean`).
Gate per epoch: each output_end span ends at its readout (end - 1 == readout), else the shard stops.
The original docstring follows.

POST HOC (RULE.md): Qwen3-32B's residual stream after all 64 decoder blocks at each staged readout of Qwen3.5-27B's
acting runs, rendered in #36's reader format (stage_crossread.py). A copy of
experiments/2026-10-08_q3_onpolicy_figures/run_extract_replay.py whose only changes are: Qwen3-32B (probe_shift.models
QWEN3_32B) on one H100 per shard, this job's app, Volume and lease names, and no flash-linear-attention (Qwen3.5 only); from the plant step on, the app name comes from XREAD_APP so each step
has its own (the elapsed watcher stops by name).
The gate, layout, resume and lease logic are unchanged. One forward pass per run (one epoch, cut at 32,768 tokens).

The original docstring follows.

Figure 3, step 2: the residual stream after all 64 decoder blocks of Qwen3.5-27B at each policy call's rule-1 readout,
read on the exact tokens vLLM served (amendment 7). One H200 forward pass per context epoch (EXTRACTION_PLAN.md).

Input: the epochs staged by run_replay_readouts.py --stage (inputs/<arm>.jsonl on the output Volume): per epoch its token
ids (the last request's logged prompt + completion), and per call its readout position, its request's length n and the
sha256 of that request's logged prompt + completion ids (int32 bytes).
The fidelity gate, before the model loads and again right before each forward pass: for every call, sha256(ids[:n])
equals the staged hash, and its readout lies before n. Any mismatch raises; nothing of that shard is written after it.
Padding (to `pad_to`) goes after the last id only: in this causal model it changes no position read.

Output, as the off-policy extraction's layout: `by_layer/L<l>/<chunk>.npy` (fp16 rows), `chunks/<chunk>.json`, and each
epoch's `meta/<arm>/<filename>.json` (its calls with their row and the staged file's sha256, marking it done). Every file
is written under a temporary name and renamed into place. A rerun skips an epoch only if its meta parses, names the
current staged file's sha256, and its chunk index lists it. The run's settings are stored once (`config.json`); a rerun
with other settings stops. Non-finite activations raise before their chunk is published.
Writers: function retries are off, and Modal may still reschedule a crashed input. Each shard has one current writer in a
Modal Dict: an attempt claims it put-if-absent (a second container of the same attempt stops), a later --attempt takes
it over, and every write re-checks it, so a superseded writer stops before its next write. Chunk names carry the
attempt.

  uv run --no-project --python 3.14 --with 'modal>=1.2' --with pyyaml modal run --detach \
      experiments/2026-10-08_q3_onpolicy_figures/run_extract_replay.py --config <config/extract_*.yaml> [--test] [--attempt 1]
`--test`: one shard, only the longest epoch, read twice (the memory and time check on one H200).
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
APP = os.environ.get("XREAD_APP") or "probe-shift_aswin-surya_sub-q3simh_extract_kill-2026-11-13"
if not (APP.startswith("probe-shift_aswin-surya_sub-q3simh_") and APP.endswith("_kill-2026-11-13") and len(APP) < 64):
    raise ValueError(f"app name {APP!r} is not this job's")
ACTS = "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"  # pinned: the only writable Volume (never B's)
BREF = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"  # batch C's Volume (2e): staged inputs and rows, read-only
REPRO = {"inputs": "q4f_c_v1/inputs/honest_b.jsonl",  # batch C, CORRECTED (2e, amendment 11 of Q4; config q4b_extract_c.yaml)
         "inputs_sha256": "513f621a9e71e91f8188cccc76cb222e9c9f4acb5a91fc90c45a283964729b31",
         "run": "q4b_extract_c_v1", "seed": 20261014, "n_drawn": 2, "min_cosine": 0.9999}
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
LEASES = "probe-shift_aswin-surya_sub-q3simh_leases_kill-2026-11-13"
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]
TIMEOUT_S = 30 * 60  # 4 h in the original; 30 min here: the gates' spend bound (RULE.md (g))

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, create_if_missing=True, version=2)
leases = modal.Dict.from_name(LEASES, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "XREAD_APP": APP, "Q4OWN_VOLUME": ACTS})


def ids_sha(ids) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()


def gate(ep: dict) -> None:
    """Raise unless every call's logged prompt + completion is exactly the epoch's first n ids, with its readout inside."""
    bad = [c["irid"] for c in ep["calls"] if c["n"] > len(ep["ids"]) or ids_sha(ep["ids"][: c["n"]]) != c["sha256"]
           or (c["readout"] is not None and not 0 <= c["readout"] < c["n"])]
    if bad:
        raise ValueError(f"fidelity gate: {len(bad)} calls of {ep['key']} differ from the logged tokens, e.g. {bad[:3]}")


def save_npy(path: Path, x) -> None:
    import numpy as np

    with open(path, "wb") as fh:
        np.save(fh, x)


def filename(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:20]


@app.function(image=image, gpu="H200", cpu=(4.0, 4.0), memory=(65536, 65536), timeout=TIMEOUT_S, retries=0, startup_timeout=300, scaledown_window=60,
              volumes={"/hf": modal.Volume.from_name(HF_CACHE).read_only(), "/acts": acts_vol,
                       "/bref": modal.Volume.from_name(BREF, version=2).read_only()})
def extract_shard(cfg: dict, shard: int) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    me = f"{os.environ.get('MODAL_TASK_ID', '?')}@{time.time():.0f}"
    key = f"{cfg['run']}/shard{shard:03d}"
    if not leases.put(f"{key}/a{cfg['attempt']}", me, skip_if_exists=True):
        raise RuntimeError(f"shard {shard} attempt {cfg['attempt']} already has a writer; rerun with a new --attempt")
    held = leases.get(key)
    if held is not None and held["attempt"] >= cfg["attempt"]:
        raise RuntimeError(f"shard {shard} is held by attempt {held['attempt']}; rerun with a later --attempt")
    leases[key] = {"writer": me, "attempt": cfg["attempt"]}

    def check():
        if (leases.get(key) or {}).get("writer") != me:
            raise RuntimeError(f"shard {shard}: a later attempt took over; stopping before writing")

    def put(path: Path, write):
        tmp = path.with_name(f".{path.name}.{cfg['attempt']}.tmp")
        write(tmp)
        check()
        os.replace(tmp, path)

    acts_vol.reload()
    run_dir, src = Path("/acts") / cfg["run"], Path("/acts") / cfg["inputs"] / "inputs"
    raw = {p.name: p.read_bytes() for p in sorted(src.glob("*.jsonl"))}
    if {n: hashlib.sha256(b).hexdigest() for n, b in raw.items()} != cfg["inputs_sha256"]:
        raise ValueError(f"the staged inputs in {src} differ from the config's sha256s")
    eps = sorted((json.loads(l) for b in raw.values() for l in b.decode().splitlines()), key=lambda e: (-len(e["ids"]), e["key"]))
    mine = eps[:1] if cfg.get("test") else eps[shard:: cfg["shards"]]
    need = {"key", "arm", "run", "task", "epoch", "attempt_index", "attempt_kind", "redrawn_because", "policy_called", "ids", "calls"}
    for ep in mine:  # the whole shard passes the gate and has every field before the model loads
        if need - set(ep):
            raise ValueError(f"{ep.get('key')}: staged without {sorted(need - set(ep))}; restage")
        gate(ep)

    def done(ep) -> bool:
        f = run_dir / "meta" / ep["arm"] / f"{filename(ep['key'])}.json"
        if not f.exists():
            return False
        m = json.loads(f.read_text())  # a truncated file raises: inspect it, don't skip it
        idx = json.loads((run_dir / "chunks" / f"{m['chunk']}.json").read_text())
        if m["inputs_sha256"] != cfg["inputs_sha256"][f"{ep['arm']}.jsonl"] or m["key"] not in {e["key"] for e in idx["epochs"]}:
            raise ValueError(f"{f} was written from other inputs or its chunk does not list it")
        return True

    todo = [ep for ep in mine if not done(ep)]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=torch.bfloat16, device_map="cuda",
                                              attn_implementation="sdpa").eval()
    base, layers = lm.model, list(range(lm.config.num_hidden_layers))
    load_s = round(time.time() - t0)

    def read(ids, idx, spans=None):
        acts = {}

        def grab(l, out):  # last-token rows, then (output_end) span-mean rows: [start, end) via a cumulative sum in float32
            h = (out[0] if isinstance(out, tuple) else out)[0]
            rows = h.index_select(0, idx)
            if spans is not None and len(spans):
                cs = torch.cumsum(h.float(), 0)
                a, b = spans[:, 0], spans[:, 1]
                lo = torch.where((a > 0).unsqueeze(1), cs.index_select(0, (a - 1).clamp(min=0)), torch.zeros_like(cs[:1]))
                rows = torch.cat([rows, ((cs.index_select(0, b - 1) - lo) / (b - a).unsqueeze(1).float()).to(rows.dtype)])
                del cs
            acts[l] = rows.to(torch.float16).cpu().numpy()

        hooks = [base.layers[l].register_forward_hook(lambda m, a, out, l=l: grab(l, out)) for l in layers]
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
            torch.cuda.synchronize()
        finally:
            for h in hooks:
                h.remove()
        return acts, round(time.time() - t, 2), round(torch.cuda.max_memory_allocated() / 2**30, 1)

    def calls_of(ep):
        kept = [c for c in ep["calls"] if c["readout"] is not None]
        means = [c for c in kept if c["kind"] == "output_end" and c.get("span")]
        if any(c["span"][1] - 1 != c["readout"] or not 0 <= c["span"][0] < c["span"][1] for c in means):
            raise ValueError(f"{ep['key']}: an output_end span does not end at its readout")
        idx = torch.tensor([c["readout"] for c in kept], device="cuda", dtype=torch.long)
        spans = torch.tensor([c["span"] for c in means], device="cuda", dtype=torch.long).reshape(-1, 2)
        return kept, means, idx, spans

    # The per-shard reproduction gate (RULE.md amendment 5), before any of this shard's own epochs is read or written.
    raw_ref = Path("/bref", REPRO["inputs"]).read_bytes()
    if hashlib.sha256(raw_ref).hexdigest() != REPRO["inputs_sha256"]:
        raise ValueError("batch C's staged inputs differ from their pinned sha256")
    import random

    all_ref = sorted(map(json.loads, raw_ref.decode().splitlines()), key=lambda e: e["key"])
    ref_eps = random.Random(REPRO["seed"]).sample(all_ref, REPRO["n_drawn"])  # the same draw in every shard
    longest_ref = max(all_ref, key=lambda e: (len(e["ids"]), e["key"]))  # C's longest epoch (amendment 5)
    if longest_ref["key"] not in {e["key"] for e in ref_eps}:
        ref_eps.append(longest_ref)
    repro = []
    for ep in ref_eps:
        gate(ep)
        kept, means, idx, spans = calls_of(ep)
        acts, _, _ = read(ep["ids"] + [pad_id] * (-len(ep["ids"]) % cfg["pad_to"]), idx, spans)
        m = json.loads(Path("/bref", REPRO["run"], "meta", ep["arm"], f"{filename(ep['key'])}.json").read_text())
        if m["n_rows"] != len(kept) + len(means):
            raise ValueError(f"reproduction {ep['task']}: {len(kept) + len(means)} rows vs the reference's {m['n_rows']}")
        worst_cos, worst_diff = 1.0, 0.0
        for l in layers:
            b = np.load(Path("/bref", REPRO["run"], "by_layer", f"L{l}", f"{m['chunk']}.npy"), mmap_mode="r")
            b = np.asarray(b[m["row_start"]: m["row_start"] + m["n_rows"]], dtype=np.float32)
            a = acts[l].astype(np.float32)
            if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError(f"shard {shard}: reproduction {ep['task']} layer {l}: shape or non-finite values")
            cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
            if not np.isfinite(cos).all():
                raise ValueError(f"shard {shard}: reproduction {ep['task']} layer {l}: non-finite cosine")
            worst_cos, worst_diff = min(worst_cos, float(cos.min())), max(worst_diff, float(np.abs(a - b).max()))
        repro.append({"reference": "batch C", "key_sha256": hashlib.sha256(ep["key"].encode()).hexdigest()[:16],
                      "task": ep["task"], "tokens": len(ep["ids"]), "rows": m["n_rows"], "min_cosine": worst_cos,
                      "max_abs_diff": worst_diff})
        if worst_cos < REPRO["min_cosine"]:
            raise ValueError(f"shard {shard}: reproduction gate failed on {ep['task']} (min cosine {worst_cos:.6f}, "
                             f"max |diff| {worst_diff:.4g}); nothing of this shard is written")
    if mine:  # the shard's own longest epoch, read twice (eps are sorted longest first, so mine[0] is it)
        own = mine[0]
        kept, means, idx, spans = calls_of(own)
        ids = own["ids"] + [pad_id] * (-len(own["ids"]) % cfg["pad_to"])
        first, _, _ = read(ids, idx, spans)
        second, _, _ = read(ids, idx, spans)
        cos_own = 1.0
        for l in layers:
            a, b = first[l].astype(np.float32), second[l].astype(np.float32)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError(f"shard {shard}: its longest epoch has non-finite values at layer {l}")
            cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
            if not np.isfinite(cos).all():
                raise ValueError(f"shard {shard}: its longest epoch: non-finite cosine at layer {l}")
            cos_own = min(cos_own, float(cos.min()) if len(cos) else 1.0)
        repro.append({"task": own["task"], "tokens": len(own["ids"]), "own_longest_read_twice_min_cosine": cos_own})
        if cos_own < REPRO["min_cosine"]:
            raise ValueError(f"shard {shard}: its longest epoch read twice differs (min cosine {cos_own:.6f}); "
                             f"nothing of this shard is written")

    for l in layers:
        (run_dir / "by_layer" / f"L{l}").mkdir(parents=True, exist_ok=True)
    (run_dir / "chunks").mkdir(parents=True, exist_ok=True)
    buf, metas, stats, peaks, chunk = {l: [] for l in layers}, [], [], [], 0

    def flush():
        nonlocal chunk
        if not metas:
            return
        name = f"shard{shard:03d}_a{cfg['attempt']}_chunk{chunk:03d}"
        xs = {l: np.concatenate(buf[l]) for l in layers}
        nonfinite = sum(int((~np.isfinite(x)).sum()) for x in xs.values())
        if nonfinite:
            raise ValueError(f"{name}: {nonfinite} non-finite activations; nothing of it is published")
        for l in layers:
            put(run_dir / "by_layer" / f"L{l}" / f"{name}.npy", lambda t, x=xs[l]: save_npy(t, x))
            buf[l].clear()
        start = 0
        for m in metas:
            m |= {"chunk": name, "row_start": start, "inputs_sha256": cfg["inputs_sha256"][f"{m['arm']}.jsonl"]}
            start += m["n_rows"]
        index = {"layers": layers, "rows": start, "epochs": [{k: m[k] for k in ("arm", "key", "row_start", "n_rows")} for m in metas]}
        put(run_dir / "chunks" / f"{name}.json", lambda t: t.write_text(json.dumps(index)))
        for m in metas:  # last: each epoch's meta marks it done
            d = run_dir / "meta" / m["arm"]
            d.mkdir(parents=True, exist_ok=True)
            put(d / f"{filename(m['key'])}.json", lambda t, m=m: t.write_text(json.dumps(m)))
        stats.append({"chunk": name, "epochs": len(metas), "rows": start})
        metas.clear()
        chunk += 1
        acts_vol.commit()

    for ep in todo:
        check()
        gate(ep)  # again, on the ids about to be read
        kept, means, idx, spans = calls_of(ep)
        ids = ep["ids"] + [pad_id] * (-len(ep["ids"]) % cfg["pad_to"])
        acts, seconds, peak = read(ids, idx, spans)
        peaks.append(peak)
        timing = {"forward_s": seconds, "peak_gib": peak, "padded_tokens": len(ids)}
        if cfg.get("test"):
            again, seconds, _ = read(ids, idx, spans)
            timing |= {"forward_s_repeat": seconds, "repeat_max_abs_diff": max(float(np.abs(acts[l].astype(np.float32) - again[l].astype(np.float32)).max(initial=0)) for l in layers)}
        for l in layers:
            buf[l].append(acts[l])
        row, row_mean = iter(range(len(kept))), iter(range(len(kept), len(kept) + len(means)))
        mean_ids = {id(c) for c in means}
        metas.append({k: ep[k] for k in ("key", "arm", "run", "task", "epoch", "attempt_index", "attempt_kind", "redrawn_because",
                                         "policy_called")} | {"n_tokens": len(ep["ids"]), "n_rows": len(kept) + len(means), **timing,
                     "calls": [{k: v for k, v in c.items() if k != "sha256"} | {"row": next(row) if c["readout"] is not None else None}
                               | ({"row_mean": next(row_mean)} if id(c) in mean_ids else {}) for c in ep["calls"]]})
        if len(metas) == cfg["chunk_size"]:
            flush()
    flush()
    return {"shard": shard, "epochs": len(mine), "done_before": len(mine) - len(todo), "chunks": stats, "load_s": load_s,
            "reproduction": repro,
            "seconds": round(time.time() - t0), "max_peak_gib": max(peaks, default=None),
            "gpu": torch.cuda.get_device_name(), "device_count": torch.cuda.device_count()}  # 0d: each record proves its hardware


@app.function(image=image, cpu=(2.0, 2.0), memory=(8192, 8192), timeout=900, retries=0, startup_timeout=300, scaledown_window=60, volumes={"/acts": acts_vol})
def verify_metas(cfg: dict) -> dict:
    """Coverage before completion: every staged epoch has exactly one meta, written from these inputs and listed by its
    chunk index; no meta without a staged epoch."""
    acts_vol.reload()
    run_dir, src = Path("/acts") / cfg["run"], Path("/acts") / cfg["inputs"] / "inputs"
    staged = {(e["arm"], e["key"]) for p in sorted(src.glob("*.jsonl")) for e in map(json.loads, p.read_text().splitlines())}
    metas, bad = {}, []
    for f in sorted((run_dir / "meta").glob("*/*.json")):
        m = json.loads(f.read_text())
        idx = json.loads((run_dir / "chunks" / f"{m['chunk']}.json").read_text())
        ok = (m["inputs_sha256"] == cfg["inputs_sha256"][f"{m['arm']}.jsonl"] and m["key"] in {e["key"] for e in idx["epochs"]}
              and f.name == f"{filename(m['key'])}.json")
        if not ok or (m["arm"], m["key"]) in metas:
            bad.append(f.name)
        metas[(m["arm"], m["key"])] = f.name
    missing, extra = sorted(staged - set(metas)), sorted(set(metas) - staged)
    return {"staged": len(staged), "metas": len(metas), "missing": [k for _, k in missing][:20],
            "extra": [k for _, k in extra][:20], "bad": bad[:20],
            "complete": bool(staged) and not missing and not extra and not bad}


@app.local_entrypoint()
def main(config: str, test: bool = False, attempt: int = 1):
    import yaml

    git = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
           "dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=HERE).stdout.strip())}
    from probe_shift.models import QWEN35_27B

    cfg = yaml.safe_load((HERE / "config" / config).read_text()) | {"attempt": attempt, "test": test}
    if cfg["model"] != QWEN35_27B["name"]:
        raise ValueError(f"{cfg['model']} is not this job's pinned reader")
    cfg["revision"] = QWEN35_27B["revision"]
    if not isinstance(cfg.get("shards"), int) or cfg["shards"] < 1:
        raise ValueError(f"shards must be a positive integer, got {cfg.get('shards')!r}")
    if test:
        cfg["run"] += "_test"
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    settings = {k: v for k, v in cfg.items() if k != "attempt"}
    try:
        stored = json.loads(b"".join(acts_vol.read_file(f"{cfg['run']}/config.json")))
    except (FileNotFoundError, modal.exception.NotFoundError):
        stored = None
    if stored is not None and stored != settings:
        raise ValueError(f"{cfg['run']} exists with other settings; use a new run name")
    import io

    with acts_vol.batch_upload(force=True) as b:
        if stored is None:
            b.put_file(io.BytesIO(json.dumps(settings, indent=1).encode()), f"{cfg['run']}/config.json")
        b.put_file(io.BytesIO(json.dumps({"attempt": attempt, "git": git}, indent=1).encode()), f"{cfg['run']}/attempt_{attempt}.json")
    n = 1 if test else cfg["shards"]
    res = list(extract_shard.starmap([(cfg, s) for s in range(n)], return_exceptions=True))
    out = {"git": git, "config": cfg, "app": APP, "volume": ACTS, "shards": [r if isinstance(r, dict) else {"error": repr(r)[:500]} for r in res]}
    path = HERE / "results" / f"extract_simh_{cfg['run']}_a{attempt}_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out["shards"], indent=1)[:3000])
    print("->", path)
    failed = [s for s in out["shards"] if "error" in s]
    if failed:
        raise SystemExit(f"{len(failed)} of {n} shards failed; rerun with --attempt {attempt + 1} (done epochs are kept)")
    if not test:  # the scorer reads only complete extractions: every staged epoch has exactly one valid meta
        cov = verify_metas.remote(cfg)
        print("coverage:", json.dumps(cov))
        if not cov["complete"]:
            raise SystemExit(f"extraction incomplete: {cov}; complete.json not written")
        with acts_vol.batch_upload(force=True) as b:
            b.put_file(io.BytesIO(json.dumps({"epochs": sum(s["epochs"] for s in out["shards"]), "attempt": attempt,
                                              "inputs_sha256": cfg["inputs_sha256"]}).encode()), f"{cfg['run']}/complete.json")
        print("complete:", sum(s["epochs"] for s in out["shards"]), "epochs")
