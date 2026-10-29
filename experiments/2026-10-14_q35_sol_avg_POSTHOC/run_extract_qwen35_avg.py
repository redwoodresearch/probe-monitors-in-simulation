"""Training-side averaged extraction (RULE.md here, POST HOC; Aswin "go up to $80"): a copy of
../2026-10-06_qwen35_replication/run_extract_qwen35.py (main @ the copy's source) with only these changes: app, Volume
and lease names; the staged inputs and the outputs on this folder's Volume; the GPU list ["H200", "B200"]; only the
call_end and output_end last-token rows kept, plus an output_end_mean row per output_end (the plain mean over its
content span from segment_end_positions, the span stage_q4b.py uses for B); and a per-shard reproduction gate before
anything is written: the shard's first real run must reproduce the stored Sol dump's output_end rows (v4) at layers
16, 32, 48 and 63 (every row's cosine >= 0.9999 and max |difference| <= 0.125), else the shard stops. The original
docstring follows.

All-layers extraction with Qwen3.5-27B as the reader: the segment-end positions at every one of its 64 layers.

The qwen35 track's copy of the main session's prepared script (experiments/2026-10-06_qwen35_reader at a0a4cda), with
four changes: this track's app name; inputs read only from the shared Volume and outputs written to this track's own
Volume; each transcript's forward time, peak GPU memory and padded length kept in its metadata; and a test mode that
reads a spread of lengths with a timed second pass (the pilot). The rest of the original docstring follows.

Prepared for the Qwen3.5 redo of Phase 1 (2026-10-06); launch only after Aswin confirms whole transcripts. Differences
from the Qwen3-32B all-layers extraction (experiments/2026-10-06_all_layers_extraction): the model (Qwen3.5-27B loaded
text-only as Qwen3_5ForCausalLM, transformers 5.19, flash-linear-attention kernels), whole transcripts (no 32k cut),
H200s (a 153k-token transcript projects to about 76 GB), and each sequence right-padded to a multiple of `pad_to`
tokens so the linear-attention kernels autotune once per length bucket instead of once per transcript; padding sits
after every readout position, so in this causal model it changes none of them.

Same transcripts and text as an existing segment-end run (its staged inputs on the Volume are read directly, so
nothing is restaged), same per-action positions (`probe_shift.reader.segment_end_positions`: pre_call, call_end,
call_marker, output_end, output_marker), the residual stream after every decoder block, fp16; no log-probs or EMA.

Layout, so an analysis can load one layer at a time from a few large files (GPU containers read many small files very
slowly): each shard buffers `chunk_size` transcripts, then writes `by_layer/L<l>/shard<s>_chunk<c>.npy` for every layer
and `chunks/shard<s>_chunk<c>.json` (each transcript's arm, key, filename and row range), and last each transcript's
`meta/<arm>/<filename>.json` (positions with rows relative to its block, plus `chunk` and `row_start`), which marks it
done: a resumed shard skips finished transcripts and continues its chunk numbering. Rerun the same command to resume.
One writer per shard (a q2 incident, 2026-10-08: Modal rescheduled a shard onto a second GPU container while the first
still ran). Function retries are off, but Modal also reschedules an input after a container failure. So each shard first
claims a numbered lease epoch in a Modal Dict (put-if-absent; ShardLease), before the model loads. The lease names the
container and carries a heartbeat refreshed every minute. A second container finding a live lease stops at once, and a
lease is taken over only once its heartbeat is 10 minutes stale. A resume or takeover writes chunks with an `_e<epoch>`
suffix, so two writers never write the same file. Consumers find chunks through each transcript's meta, so the suffix
changes nothing for them. Chunk files are written under temporary names and renamed into place after re-checking the
lease.

  uv run modal run --detach experiments/2026-10-06_qwen35_replication/run_extract_qwen35.py --config qwen35_alllayers.yaml

Throughput options (2026-10-07; off by default, so earlier runs' code path is unchanged):
  - `shard_by: length`: each shard takes a contiguous block of lengths (about equal tokens per shard) instead of every
    `shards`-th transcript. A container tunes the linear-attention kernels once per 4,096-token bucket it meets (about
    9.5 s each); round-robin shards met about 18 buckets each, 22% of the v4 run's forward time.
  - `prefetch: true`: the next transcript's tokenization and readout positions are computed on a CPU thread while the
    GPU reads the current one (7% of the v4 run's shard time was this CPU work with the GPU idle).
Each shard's stats also record its peak host memory (`max_rss_gib`), to size `memory` from data.

`transcripts: <file in config/>` (2026-10-07, the call_end fix) extracts only the listed transcripts (its `affected` and
`controls`) and refuses staged inputs whose shard files differ from the file's `inputs_sha256`. Everything else is as
above, so the listed transcripts are read exactly as in the full run apart from the reader fix.
"""

import json
import os
import re
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q35solavg_extract_kill-2026-11-13"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
INPUTS = "probe-shift_aswin-surya_sub-la-activations"  # the main session's Volume: staged inputs, mounted read only
# The output Volume; QWEN35_OUTPUT_VOLUME overrides it for a run (claim B reader run, 2026-10-10). The image carries the
# same value, so containers resolve the same Volume. Unset, every earlier run's path is unchanged.
OUTPUT = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"  # staged inputs and outputs
V4VOL, V4RUN = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", "2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1"
GATE_LAYERS, GATE_COS, GATE_ABS, KEEP_KINDS = (16, 32, 48, 63), 0.9999, 0.125, ("call_end", "output_end")
GPU = ["H200", "B200"]
TIMEOUT_S = 3 * 3600
LEASES = "probe-shift_aswin-surya_sub-q35solavg_leases_kill-2026-11-13"  # Modal Dict: (run/shard) -> its writer
LEASE_STALE_S, HEARTBEAT_S, RPC_TIMEOUT_S = 600, 60, 30
THROUGHPUT_DEFAULTS = {"shard_by": "round_robin", "prefetch": False}  # the options' defaults, recorded with every run
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]

app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE)
in_vol = modal.Volume.from_name(V4VOL, version=2).read_only()  # the stored Sol dump, for the gate
out_vol = modal.Volume.from_name(OUTPUT, create_if_missing=True, version=2)
leases = modal.Dict.from_name(LEASES, create_if_missing=True)


class LeaseTimeout(RuntimeError):
    """A Modal Dict call with no answer in time. A RuntimeError, not a TimeoutError, so main() records the shard as
    failed instead of retrying the result fetch as a connection timeout."""


def _rpc(fn, *args, timeout: float = RPC_TIMEOUT_S, **kwargs):
    """A Modal Dict call that fails closed: LeaseTimeout if Modal gives no answer within `timeout` seconds (Dict calls
    take no timeout of their own). The call runs on a daemon thread, so a hung one cannot keep the container alive."""
    import threading

    out = {}

    def run():
        try:
            out["value"] = fn(*args, **kwargs)
        except BaseException as e:  # re-raised in the caller
            out["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise LeaseTimeout(f"Modal Dict call gave no answer within {timeout} s")
    if "error" in out:
        raise out["error"]
    return out["value"]


class ShardLease:
    """One writer per shard across containers, built only from put-if-absent (Modal Dicts have no compare-and-swap). Claims
    are numbered epochs: `<run>/shard<s>/epoch<n>`. A claimer stops if any epoch entry is live (not released,
    heartbeat under 10 minutes old). Otherwise it creates epoch n = 1 + the largest epoch in the Dict or among the
    shard's chunk files on the Volume, with put-if-absent, which exactly one claimer can win. Counting the chunk
    files keeps names unique after Dict entries expire (seven days untouched), and so does counting release markers;
    a live holder's entry never expires. Each holder writes only its own entry (heartbeats) and its release marker
    `<...>/epoch<n>/released`, a separate key, so a heartbeat write still in flight cannot undo a release. A holder
    has lost the shard once epoch n+1 exists. Correctness does not rest on the lease: chunks of epoch n > 0 carry an
    `_e<n>` suffix, so two writers never share a file. The lease only keeps a second container from spending GPU
    time on the same shard."""

    def __init__(self, key: str, me: str | None = None):
        import threading

        self.key, self.lost, self.beat, self.epoch, self.entry = key, threading.Event(), None, None, None
        self.me = me or os.environ.get("MODAL_TASK_ID") or f"{socket.gethostname()}-{os.getpid()}"

    def _at(self, n: int) -> str:
        return f"{self.key}/epoch{n}"

    def claim(self, used: set[int] = frozenset()):
        """used: the epochs whose chunks are already on the Volume."""
        import threading

        prefix = f"{self.key}/epoch"
        items = _rpc(lambda: list(leases.items()), timeout=120)
        held = {int(k[len(prefix):]): v for k, v in items if k.startswith(prefix) and k[len(prefix):].isdigit()}
        released = {int(k[len(prefix):-len("/released")]) for k, _ in items
                    if k.startswith(prefix) and k.endswith("/released") and k[len(prefix):-len("/released")].isdigit()}
        for e, v in sorted(held.items()):
            if e not in released and not v.get("released") and time.time() - v["heartbeat"] < LEASE_STALE_S:  # (an in-entry
            # flag: the first version of this lease)
                raise RuntimeError(f"{self.key} is held by {v['task']} (epoch {e}, heartbeat {time.time() - v['heartbeat']:.0f} s ago); "
                                   f"not starting a second writer. Rerun once it has ended")
        n = max(set(held) | released | set(used), default=-1) + 1  # a marker can outlive its entry
        now = time.time()
        self.entry = {"task": self.me, "claimed": now, "heartbeat": now}
        if not _rpc(leases.put, self._at(n), self.entry, skip_if_exists=True):
            raise RuntimeError(f"another container claimed {self.key} epoch {n} at the same time; not starting")
        self.epoch = n
        self.beat = threading.Thread(target=self._beat, daemon=True)
        self.beat.start()

    def _beat(self):
        try:
            while not self.lost.wait(HEARTBEAT_S):
                if _rpc(leases.get, self._at(self.epoch + 1)) is not None:  # a successor claimed the shard
                    self.lost.set()
                    return
                self.entry = self.entry | {"heartbeat": time.time()}
                _rpc(leases.put, self._at(self.epoch), self.entry)  # only this container writes this entry
        except Exception:
            self.lost.set()  # fail closed: without a heartbeat the lease would go stale while this container writes
            raise

    def check(self):
        """Raise unless this container still holds the shard: its heartbeat is alive and recent, and no later epoch
        exists. Called before every forward pass and every write to shared files; a Dict call with no answer within
        30 s counts as a lost lease."""
        stale = time.time() - self.entry["heartbeat"] > LEASE_STALE_S - 2 * HEARTBEAT_S  # a successor may take over soon
        try:
            taken = self.lost.is_set() or not self.beat.is_alive() or stale or _rpc(leases.get, self._at(self.epoch + 1)) is not None
        except LeaseTimeout:
            taken = True
        if taken:
            self.lost.set()
            raise RuntimeError(f"lost the lease on {self.key} (epoch {self.epoch}); stopping before the next forward pass or write")

    def release(self):
        self.lost.set()  # stops the heartbeat
        if self.beat is not None:
            self.beat.join(RPC_TIMEOUT_S + 5)  # the heartbeat's own Dict calls time out
            try:
                _rpc(leases.put, f"{self._at(self.epoch)}/released", True)
            except LeaseTimeout:  # unreleased, the lease goes stale in 10 minutes
                print(f"could not release {self.key} epoch {self.epoch}: Modal Dict gave no answer", flush=True)


image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "QWEN35_OUTPUT_VOLUME": OUTPUT})
         .add_local_python_source("probe_shift"))


def filename(key: str) -> str:
    """Same file stems as the segment-end extraction."""
    import hashlib

    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, gpu=GPU, memory=65536, volumes={"/hf": hf_cache, "/v4": in_vol, "/out": out_vol}, timeout=TIMEOUT_S,
              retries=0)  # resume by rerunning instead (see the docstring)
def extract_shard(cfg: dict, shard: int) -> list[dict]:
    lease = ShardLease(f"{cfg['run']}/shard{shard:03d}")
    out_vol.reload()  # every chunk the shard's previous writers committed
    chunks = (Path("/out") / cfg["run"] / "chunks").glob(f"shard{shard:03d}_chunk*.json")
    lease.claim({int(m.group(1)) if (m := re.search(r"_e(\d+)$", f.stem)) else 0 for f in chunks})  # before the model loads
    try:
        return _extract_shard(cfg, shard, lease)
    finally:
        lease.release()


def _extract_shard(cfg: dict, shard: int, lease: ShardLease) -> list[dict]:
    import resource
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.reader import segment_end_positions

    import hashlib

    src = Path("/out") / cfg["inputs_from"] / "inputs"
    raw = {p.name: p.read_bytes() for p in sorted(src.glob("shard_*.jsonl"))}
    if cfg.get("inputs_sha256") and {n: hashlib.sha256(b).hexdigest() for n, b in raw.items()} != cfg["inputs_sha256"]:
        raise ValueError(f"the staged inputs in {src} differ from the sha256s the transcript list was made from")
    items = [json.loads(l) for b in raw.values() for l in b.decode().splitlines()]
    ids = [(it["arm"], it["key"]) for it in items]
    if len(set(ids)) != len(ids):  # e.g. stale shard files left by a resume with fewer shards
        raise ValueError(f"{len(ids) - len(set(ids))} duplicate transcripts in {src}")
    if cfg.get("only"):  # a listed subset; every listed transcript must be staged
        only = {tuple(x) for x in cfg["only"]}
        if only - set(ids):
            raise ValueError(f"{len(only - set(ids))} listed transcripts are not staged in {src}")
        items = [it for it in items if (it["arm"], it["key"]) in only]
    items.sort(key=lambda it: (-it["tokens"], it["arm"], it["key"]))  # longest first, dealt round-robin: balanced
    if cfg.get("shard_by", "round_robin") == "length":  # contiguous lengths, about total/shards tokens per shard
        total, before, owner = sum(it["tokens"] for it in items), 0, []
        for it in items:
            owner.append(min(before * cfg["shards"] // total, cfg["shards"] - 1))
            before += it["tokens"]
        mine = [it for it, o in zip(items, owner) if o == shard]
    elif cfg.get("shard_by", "round_robin") == "round_robin":
        mine = items[shard::cfg["shards"]]
    else:
        raise ValueError(f"shard_by must be round_robin or length, not {cfg['shard_by']!r}")
    if cfg.get("test_items"):  # pilot: evenly spaced through this shard's lengths, longest first
        mine = mine[:: max(1, len(mine) // cfg["test_items"])][: cfg["test_items"]]
    run_dir = Path("/out") / cfg["run"]
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=getattr(torch, cfg["dtype"]),
                                              device_map="cuda", attn_implementation=cfg["attn_implementation"]).eval()
    base = lm.model
    layers = list(range(lm.config.num_hidden_layers))
    load_s = round(time.time() - t_load)

    def read(ids, idx, spans=()):
        """One forward pass; returns the rows at idx, then one mean row per span [a, b), for every layer, its seconds and
        peak GPU memory (GiB)."""
        acts = {}

        def hook_for(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                parts = [h[0].index_select(0, idx).float()]
                if spans:
                    parts.append(torch.stack([h[0, a:b].float().mean(0) for a, b in spans]))
                acts[layer] = torch.cat(parts).to(torch.float16).cpu().numpy()
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

    for l in layers:
        (run_dir / "by_layer" / f"L{l}").mkdir(parents=True, exist_ok=True)
    (run_dir / "chunks").mkdir(parents=True, exist_ok=True)
    chunk = len(list((run_dir / "chunks").glob(f"shard{shard:03d}_chunk*.json")))  # continue numbering after a retry
    buf, buf_meta, stats = {l: [] for l in layers}, [], []

    def flush():
        nonlocal chunk
        if not buf_meta:
            return
        name = f"shard{shard:03d}_chunk{chunk:03d}" + (f"_e{lease.epoch}" if lease.epoch else "")  # one writer per name
        lease.check()
        nonfinite, staged = 0, {}  # final path -> this writer's temporary file
        for l in layers:
            x = np.concatenate(buf[l]) if buf[l] else np.zeros((0, 5120), np.float16)
            nonfinite += int((~np.isfinite(x)).sum())
            final = run_dir / "by_layer" / f"L{l}" / f"{name}.npy"
            staged[final] = final.with_name(f".{name}.{lease.me}.npy.tmp")
            with open(staged[final], "wb") as fh:
                np.save(fh, x)
            buf[l].clear()
        index, start = [], 0
        for m in buf_meta:
            index.append({"arm": m["arm"], "key": m["key"], "filename": m["filename"], "row_start": start, "n_rows": m["n_rows"]})
            m |= {"chunk": name, "row_start": start}
            start += m["n_rows"]
        final = run_dir / "chunks" / f"{name}.json"
        staged[final] = final.with_name(f".{name}.{lease.me}.json.tmp")
        staged[final].write_text(json.dumps({"layers": layers, "rows": start, "transcripts": index}))
        lease.check()
        for f, tmp in staged.items():  # layer arrays, then the chunk index: names are this epoch's alone
            os.replace(tmp, f)
        for m in buf_meta:  # written last: each transcript's metadata marks it done
            d = run_dir / "meta" / m["arm"]
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".{m['filename']}.json.tmp"
            tmp.write_text(json.dumps(m))
            os.replace(tmp, d / f"{m['filename']}.json")
        stats.append({"chunk": name, "transcripts": len(buf_meta), "rows": start, "nonfinite": nonfinite, "gpu": gpu_name})  # Amendment 23: GPU per chunk
        buf_meta.clear()
        chunk += 1
        out_vol.commit()

    t_all = time.time()
    gpu_name = torch.cuda.get_device_name(0)
    todo = [it for it in mine if not (run_dir / "meta" / it["arm"] / f"{filename(it['key'])}.json").exists()]
    gate_prefix = cfg.get("gate_arm_prefix", "real-")  # Amendment 23: the as-is simulated arms gate on their own v4 rows ("sim-")
    reals = [it for it in mine if it["arm"].startswith(gate_prefix)]
    gate_item = reals[-1] if reals else None  # the shard's shortest real run (or gate_prefix run), read first
    if gate_item is None:
        raise RuntimeError(f"shard {shard}: no {gate_prefix} run to gate on")
    gate_already_done = (run_dir / "meta" / gate_item["arm"] / f"{filename(gate_item['key'])}.json").exists()
    todo = [gate_item] + [it for it in todo if it is not gate_item]

    def check_gate(it, kept, acts):
        m = json.loads((Path("/v4") / V4RUN / "meta" / it["arm"] / f"{filename(it['key'])}.json").read_text())
        sp = [p for p in m["positions"] if p["kind"] == "output_end" and p["kept"]]
        mine_rows = [i for i, p in enumerate(kept) if p["kind"] == "output_end"]
        if len(sp) != len(mine_rows) or [p["token"] for p in sp] != [kept[i]["token"] for i in mine_rows]:
            return {"passes": False, "why": "output_end positions differ", "stored": len(sp), "mine": len(mine_rows)}
        worst_cos, worst_abs = 1.0, 0.0
        for l in GATE_LAYERS:
            S = np.load(Path("/v4") / V4RUN / "by_layer" / f"L{l}" / f"{m['chunk']}.npy", mmap_mode="r")
            a = np.asarray(S[[m["row_start"] + p["row"] for p in sp]]).astype(np.float32)
            b = acts[l][mine_rows].astype(np.float32)
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
            worst_cos, worst_abs = min(worst_cos, float(cos.min())), max(worst_abs, float(np.abs(a - b).max()))
        return {"passes": bool(worst_cos >= GATE_COS and worst_abs <= GATE_ABS), "rows": len(sp), "min_cosine": worst_cos,
                "max_abs_diff": worst_abs, "gpu": gpu_name}
    positions = lambda it: segment_end_positions(tok, it["messages"], cfg["max_tokens"])
    pool = ThreadPoolExecutor(1) if cfg.get("prefetch") else None  # the tokenizer is used only on that thread
    ahead = pool.submit(positions, todo[0]) if pool and todo else None
    for n, it in enumerate(todo, 1):
        lease.check()  # before every forward pass: a superseded or cut-off holder stops here
        stem = filename(it["key"])
        if pool:
            r = ahead.result()
            ahead = pool.submit(positions, todo[n]) if n < len(todo) else None
        else:
            r = positions(it)
        kept = [p for p in r["positions"] if p["kept"] and p["kind"] in KEEP_KINDS]
        mean_src = [p for p in kept if p["kind"] == "output_end" and p.get("span")]
        for p in mean_src:
            if p["span"][1] - 1 != p["token"] or p["span"][0] >= p["span"][1]:
                raise ValueError("an output_end span does not end at its readout token")
        idx = torch.tensor([p["token"] for p in kept], device="cuda", dtype=torch.long)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = r["ids"] + [pad] * (-len(r["ids"]) % cfg["pad_to"])  # after every position read: no effect on them
        acts, seconds, peak = read(ids, idx, [tuple(p["span"]) for p in mean_src])
        if it is gate_item:  # the per-shard reproduction gate, before anything of this shard is written
            gate = check_gate(it, kept, acts)
            print(f"shard {shard}: gate {json.dumps(gate)}", flush=True)
            if not gate["passes"]:
                raise RuntimeError(f"shard {shard}: reproduction gate failed")
            if gate_already_done:  # a resumed shard re-checks its GPU but keeps the rows already written
                continue
        timing = {"forward_s": round(seconds, 2), "peak_gib": round(peak, 1), "padded_tokens": len(ids)}
        if cfg.get("repeat_forward"):  # pilot: a second, warm pass separates one-time kernel tuning from per-run cost
            again, seconds, _ = read(ids, idx)
            timing |= {"forward_s_repeat": round(seconds, 2),
                       "repeat_max_abs_diff": max(float(np.abs(acts[l].astype(np.float32) - again[l].astype(np.float32)).max(initial=0))
                                                  for l in layers)}
        for l in layers:
            buf[l].append(acts[l])
        row = iter(range(len(kept) + len(mean_src)))
        plain_pos = [p | {"row": next(row) if (p["kept"] and p["kind"] in KEEP_KINDS) else None} for p in r["positions"]]
        mean_pos = [p | {"kind": "output_end_mean", "row": next(row)} for p in mean_src]
        buf_meta.append({**{k: it[k] for k in ("key", "arm", "cohort", "file", "sample_id", "epoch", "environment", "main_task",
                                               "side_task", "cell", "n_submit_calls", "sim_reference")},
                         "filename": stem, "n_tokens": r["n_tokens"], "truncated": r["truncated"], "kept_tokens": len(r["ids"]),
                         "n_rows": len(kept) + len(mean_src), **timing, "gpu": gpu_name,
                         "positions": plain_pos + mean_pos})
        if len(buf_meta) == cfg["chunk_size"]:
            flush()
            print(f"shard {shard}: {n}/{len(todo)} after {(time.time() - t_all) / 60:.1f} min", flush=True)
    flush()
    if pool:
        pool.shutdown()
    summary = {"shard": shard, "transcripts": len(mine), "chunks": stats, "load_seconds": load_s, "seconds": round(time.time() - t_all),
               "max_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)}  # Linux: KiB
    lease.check()
    (run_dir / "stats").mkdir(exist_ok=True)
    (run_dir / "stats" / f"shard_{shard}.json").write_text(json.dumps(summary))
    out_vol.commit()
    return [summary]


@app.local_entrypoint()
def main(config: str = "qwen35_alllayers.yaml", test: bool = False, no_wait: bool = False, gpu: str = ""):
    """`--test` (the pilot): one shard, 8 transcripts spread over its lengths, each read twice, chunks of 4, under a
    separate run name. `--no-wait` (with `modal run --detach`): spawn the shards and return, so a laptop outage cannot
    stop them; shards resume past finished transcripts, and their stats are on the Volume."""
    import yaml

    from probe_shift.models import QWEN35_27B

    git = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
           "dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=HERE).stdout.strip())}
    cfg = yaml.safe_load((HERE / "config" / config).read_text())
    if cfg["model"] != QWEN35_27B["name"]:
        raise ValueError(f"{cfg['model']} has no pinned revision in probe_shift.models")
    cfg["revision"] = QWEN35_27B["revision"]
    cfg = THROUGHPUT_DEFAULTS | cfg  # resolved, so config.json and the local result record what ran
    if cfg.get("transcripts"):  # the subset and the input hashes it was chosen from travel in the stored config
        t = json.loads((HERE / "config" / cfg["transcripts"]).read_text())
        if t["inputs_run"] != cfg["inputs_from"]:
            raise ValueError(f"{cfg['transcripts']} lists transcripts of {t['inputs_run']}, not {cfg['inputs_from']}")
        cfg |= {"only": sorted(map(list, t["affected"] + t["controls"])), "inputs_sha256": t["inputs_sha256"]}
    if test:
        cfg |= {"run": cfg["run"] + "_test", "test_items": 8, "chunk_size": 4, "repeat_forward": True}
    try:
        stored = json.loads(b"".join(out_vol.read_file(f"{cfg['run']}/config.json")))
    except (FileNotFoundError, modal.exception.NotFoundError):
        stored = None
    if stored is not None and THROUGHPUT_DEFAULTS | stored != cfg:  # a run stored before these options had their defaults
        raise ValueError(f"{cfg['run']} exists with different settings; use a new run name")
    with out_vol.batch_upload(force=True) as batch:
        import io

        batch.put_file(io.BytesIO(json.dumps(cfg, indent=1).encode()), f"{cfg['run']}/config.json")
    n = 1 if test else cfg["shards"]
    started = time.time()
    if "," in gpu:
        raise SystemExit("--gpu takes one GPU type (Function.with_options accepts a single type)")
    gpus = [gpu] if gpu else GPU  # Amendment 23 (0d): `--gpu H200` resumes on H200 only; not in cfg, so a resume's config check is unchanged
    fn = extract_shard.with_options(gpu=gpu) if gpu else extract_shard
    calls = [fn.spawn(cfg, i) for i in range(n)]
    print(f"{cfg['run']}: git {git['commit'][:7]}{' (dirty)' if git['dirty'] else ''}, spawned {n} shards reading inputs of {cfg['inputs_from']}", flush=True)
    if no_wait:
        print(f"not waiting; output Volume {OUTPUT}, run {cfg['run']}", flush=True)
        return
    stats = []
    for shard, call in enumerate(calls):
        for attempt in range(120):
            try:
                stats.extend(call.get())
                break
            except (socket.gaierror, ConnectionError, TimeoutError, modal.exception.ConnectionError, modal.exception.ServiceError) as e:
                print(f"lost the connection while waiting on shard {shard} ({e!r}); retry {attempt + 1} in 30 s", flush=True)
                time.sleep(30)
            except Exception as e:
                print(f"shard {shard} crashed: {e!r}")
                stats.append({"shard": shard, "crashed": repr(e)[:500]})
                break
        else:  # an hour of network errors: unknown outcome; its stats are on the Volume if it finished
            stats.append({"shard": shard, "unreachable": True})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    out = {"wall_seconds": round(time.time() - started), "git": git,
           "config": cfg | {"modal": {"app": APP, "gpu": gpus, "pip": PIP, "inputs_volume": INPUTS, "output_volume": OUTPUT}},
           "shards": stats}
    (HERE / "results" / f"extract_{cfg['run']}_{stamp}.json").write_text(json.dumps(out, indent=1))
    ok = [s for s in stats if "chunks" in s]
    print(f"done in {out['wall_seconds'] / 3600:.2f} h: {len(ok)} shards ok, {len(stats) - len(ok)} crashed or unreachable; "
          f"transcripts {sum(c['transcripts'] for s in ok for c in s['chunks'])}, rows {sum(c['rows'] for s in ok for c in s['chunks'])}, "
          f"non-finite {sum(c['nonfinite'] for s in ok for c in s['chunks'])}")
    if cfg.get("gate_arm_prefix") and len(ok) != len(stats):  # Amendment 23 (Codex): a failed shard fails the run (earlier runs, without the key, unchanged)
        raise SystemExit(f"{len(stats) - len(ok)} shards crashed or unreachable")
