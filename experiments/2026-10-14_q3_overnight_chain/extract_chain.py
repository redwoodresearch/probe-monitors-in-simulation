"""Overnight chain (RULE.md here): the real-honest fork's extract_rh.py (origin/aswin/q3-q35-realhonest-1e4 @ 2f0932469), changed only in
its app, Volume and lease names (the chain's). Its docstring follows.

1e-4 real honest runs (RULE.md here, Amendment 6): 2e's extract_q4b.py (aswin/q4-why-weaker @ 26d036293, as batch C used it
for q4b_extract_c_v1) whose only changes are this job's app, Volume and lease names and its result file name. One H200 per shard
(gpu="H200", never a type that allows B200), flash-linear-attention, every layer: command, call_end and output_end last-token
rows plus output_end span means. The original docstring follows.

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
APP = os.environ.get("XREAD_APP") or "probe-shift_aswin-surya_sub-q3chain_extr_kill-2026-11-14"
if not (APP.startswith("probe-shift_aswin-surya_sub-q3chain_") and APP.endswith("_kill-2026-11-14") and len(APP) < 64):
    raise ValueError(f"app name {APP!r} is not this job's")
ACTS = os.environ.get("Q4OWN_VOLUME") or "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
LEASES = os.environ.get("XREAD_LEASES") or "probe-shift_aswin-surya_sub-q3chain_leases_kill-2026-11-14"  # d9's CUT fork: a separate batch's own
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]
STARTUP_S, SCALEDOWN_S = 10 * 60, 60  # the chain: billable startup and idle time bounded too (the reservation counts both)
TIMEOUT_S = int(os.environ.get("XREAD_TIMEOUT_S") or 30 * 60)  # the chain: about 10 epochs per shard at under 1 min each, plus the model load; a bounded worst case per shard
# (d9's CUT fork, Append 41: an uncut batch sets its own bound by env; unset for the chain)

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, create_if_missing=True, version=2)
leases = modal.Dict.from_name(LEASES, create_if_missing=True)
_OVR_ENV = {k: os.environ[k] for k in ("XREAD_LEASES", "XREAD_TIMEOUT_S") if os.environ.get(k)}  # empty for the chain (image unchanged)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "XREAD_APP": APP, "Q4OWN_VOLUME": ACTS} | _OVR_ENV)


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


@app.function(image=image, gpu="H200", memory=65536, timeout=TIMEOUT_S, retries=0, startup_timeout=STARTUP_S, scaledown_window=SCALEDOWN_S,
              volumes={"/hf": modal.Volume.from_name(HF_CACHE).read_only(), "/acts": acts_vol})
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
        kept = [c for c in ep["calls"] if c["readout"] is not None]
        means = [c for c in kept if c["kind"] == "output_end" and c.get("span")]
        if any(c["span"][1] - 1 != c["readout"] or not 0 <= c["span"][0] < c["span"][1] for c in means):
            raise ValueError(f"{ep['key']}: an output_end span does not end at its readout")
        idx = torch.tensor([c["readout"] for c in kept], device="cuda", dtype=torch.long)
        spans = torch.tensor([c["span"] for c in means], device="cuda", dtype=torch.long).reshape(-1, 2)
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
        uncut = {"cut": "full", "tokens_total": len(ep["ids"]), "truncated": False,
                 "max_position": max((c["readout"] for c in kept), default=None)} if cfg.get("cut") == "full" else {}  # Append 41 (cutfull roots only)
        metas.append({k: ep[k] for k in ("key", "arm", "run", "task", "epoch", "attempt_index", "attempt_kind", "redrawn_because",
                                         "policy_called")} | {"n_tokens": len(ep["ids"]), "n_rows": len(kept) + len(means), **timing, **uncut,
                     "calls": [{k: v for k, v in c.items() if k != "sha256"} | {"row": next(row) if c["readout"] is not None else None}
                               | ({"row_mean": next(row_mean)} if id(c) in mean_ids else {}) for c in ep["calls"]]})
        if len(metas) == cfg["chunk_size"]:
            flush()
    flush()
    out = {"shard": shard, "epochs": len(mine), "done_before": len(mine) - len(todo), "chunks": stats, "load_s": load_s,
           "seconds": round(time.time() - t0), "max_peak_gib": max(peaks, default=None),
           "gpu_name": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count()}  # the chain: each shard logs its own hardware
    # the writer's own record on the Volume (10-15, 2e's finding on ilnm_s5_sim2: Modal re-executed an orchestrator, whose second
    # fan-out hit the leases and was the only result recorded): a re-executed orchestrator reads this instead of the lease error
    (run_dir / "shard_results").mkdir(parents=True, exist_ok=True)
    put(run_dir / "shard_results" / f"shard{shard:03d}_a{cfg['attempt']}.json",
        lambda t: t.write_text(json.dumps(out | {"writer": me, "function_call_id": modal.current_function_call_id(), "input_id": modal.current_input_id()})))
    acts_vol.commit()
    return out


WRITER_WAIT_S = TIMEOUT_S + STARTUP_S + 120  # a lease holder of the same attempt finishes (or dies) within its own timeout bound


def writer_record(cfg: dict, shard: int, attempt: int, deadline: float) -> dict | None:
    """A shard whose lease another execution of the SAME attempt holds: wait (until the shared deadline, the writers' own timeout
    bound) for that writer's own record (shard_results/) and return it, marked as recovered; None if none appears (the error stands)."""
    p = Path("/acts") / cfg["run"] / "shard_results" / f"shard{shard:03d}_a{attempt}.json"
    while True:
        acts_vol.reload()
        if p.exists():
            rec = json.loads(p.read_text())
            return {k: v for k, v in rec.items() if k not in ("writer", "function_call_id", "input_id")} | {
                "recovered_from_writer_record": f"{cfg['run']}/shard_results/{p.name}",
                "writer": rec.get("writer"), "writer_function_call_id": rec.get("function_call_id"), "writer_input_id": rec.get("input_id")}
        if time.time() >= deadline:
            return None
        time.sleep(30)


ORCH_TIMEOUT_S = 4 * 3600


@app.function(image=image, cpu=1, memory=2048, timeout=ORCH_TIMEOUT_S, retries=0, volumes={"/acts": acts_vol})
def orchestrate(cfg: dict, attempt: int, n: int, git: dict, test: bool) -> dict:
    """The chain (10-14 ~21:00, 0d: a laptop disconnect must not cancel shards): the shard fan-out runs HERE, on Modal, not in the
    laptop's client. Its result goes to the Volume ({run}/result_a<attempt>.json, and complete.json when every shard succeeded), so
    a re-attached or restarted chain reads it from the Volume."""
    import io

    t_start = time.time()
    with acts_vol.batch_upload(force=True) as b:  # the started marker: a chain that never saw the client's call id finds it here
        b.put_file(io.BytesIO(json.dumps({"orchestrator_call_id": modal.current_function_call_id(), "attempt": attempt}).encode()),
                   f"{cfg['run']}/orchestrator_a{attempt}.json")
    res = list(extract_shard.starmap([(cfg, s) for s in range(n)], return_exceptions=True))
    shards, deadline = [], min(time.time() + WRITER_WAIT_S, t_start + ORCH_TIMEOUT_S - 600)  # 10 min kept to publish the result
    for s, r in enumerate(res):
        if not isinstance(r, dict) and f"attempt {attempt} already has a writer" in repr(r):  # this orchestrator was re-executed
            r = writer_record(cfg, s, attempt, deadline) or r
        shards.append(r if isinstance(r, dict) else {"error": repr(r)[:500]})
    out = {"git": git, "config": cfg, "app": APP, "volume": ACTS, "attempt": attempt, "shards": shards}
    acts_vol.reload()
    with acts_vol.batch_upload(force=True) as b:
        b.put_file(io.BytesIO(json.dumps(out, indent=1).encode()), f"{cfg['run']}/result_a{attempt}.json")
        if not test and not any("error" in x for x in out["shards"]):  # the scorer reads only complete extractions
            b.put_file(io.BytesIO(json.dumps({"epochs": sum(x["epochs"] for x in out["shards"]), "attempt": attempt,
                                              "inputs_sha256": cfg["inputs_sha256"]}).encode()), f"{cfg['run']}/complete.json")
    return out


@app.local_entrypoint()
def main(config: str, test: bool = False, attempt: int = 1):
    import yaml

    git = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
           "dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ":/*.py", ":/src"], capture_output=True, text=True, cwd=HERE).stdout.strip())}
    # the chain: dirty = a modified tracked CODE file (*.py, src/); the records, configs and ACTIVATIONS rows the chain writes and commits
    # concurrently (per-batch workers) are not code
    from probe_shift.models import QWEN35_27B

    cfg = yaml.safe_load((HERE / "config" / config).read_text()) | {"attempt": attempt, "test": test}
    if cfg["model"] != QWEN35_27B["name"]:
        raise ValueError(f"{cfg['model']} is not this job's pinned reader")
    cfg["revision"] = QWEN35_27B["revision"]
    if "cut" in cfg and (cfg["cut"] != "full" or ACTS == "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
                         or LEASES == "probe-shift_aswin-surya_sub-q3chain_leases_kill-2026-11-14"
                         or not APP.startswith("probe-shift_aswin-surya_sub-q3chain_cutfull")):  # an allowlist: the chain never names an app so
        raise SystemExit("a cutfull config runs only with its own app, Volume and leases (XREAD_APP, Q4OWN_VOLUME, XREAD_LEASES)")
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
    call = orchestrate.spawn(cfg, attempt, n, git, test)  # the fan-out runs on Modal; run with --detach so a disconnect cannot cancel it
    print(json.dumps({"orchestrator_call_id": call.object_id, "run": cfg["run"], "attempt": attempt}), flush=True)
    out = call.get()
    path = HERE / "results" / f"extract_chain_{cfg['run']}_a{attempt}_{stamp}.json"
    if path.exists():
        raise SystemExit(f"{path.name} exists")
    part = path.with_suffix(".part")  # the chain: atomic, so a kill or a sleep never leaves a truncated record under the final name
    part.write_text(json.dumps(out, indent=1))
    part.rename(path)
    print(json.dumps(out["shards"], indent=1)[:3000])
    print("->", path)
    failed = [s for s in out["shards"] if "error" in s]
    if failed:
        raise SystemExit(f"{len(failed)} of {n} shards failed; rerun with --attempt {attempt + 1} (done epochs are kept)")
    if not test:  # complete.json was written by the orchestrator on Modal (the one writer)
        print("complete:", sum(s["epochs"] for s in out["shards"]), "epochs")
