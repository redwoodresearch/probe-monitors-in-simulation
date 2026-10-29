"""Qwen3-32B reads both versions of every matched pair (real history + real output, real history + simulated output)
and saves the residual stream after every one of its 64 blocks at five positions, plus the output's token log-probs.

Each version is the pair's reader messages (honest system prompt, as in every extraction) through the next agent turn
that calls a tool; the two versions differ only in step k's tool output. Positions, from
`probe_shift.reader.segment_end_positions`, with action 0 of each turn:
  output_end (step k)     the output's last content token: the readout the Q1 probes use
  output_marker (step k)  the `</tool_response>` after it (template token; reported only next to output_end)
  pre_call (next call)    the `<tool_call>` token opening the next call
  call_end (next call)    the next call's last content token (the real next command, identical in both versions)
  call_marker (next call) the `</tool_call>` after it
The next call's "previous content token" (Q2's prev_content) is output_end of step k: only text-only turns or user
messages can sit between them. Tokens after the last position are not run (a causal model never sees them), and every
position must sit inside the first 32,768 tokens; a version whose positions do not is recorded and skipped.

Inputs (pairs file and simulator results, both git-ignored) are uploaded to the pairs Volume with the items; outputs go
to `<run>/acts/<pair>_<version>.npz` (array [5, 64, 5120] fp16 and log-probs) plus `.json` (positions). A retried or
resumed shard skips items already written.

  uv run modal run --detach experiments/2026-10-06_matched_step_pairs/run_read_pairs.py \\
    --pairs data/pairs_<stamp>.jsonl --sims data/sim_<stamp>.jsonl [--sims data/sim_<stamp2>.jsonl] --run <name>
"""

import io
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-matched_read_kill-2026-11-06"
VOLUME = "probe-shift_aswin-surya_sub-matched_pairs_kill-2026-11-06"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
GPU = "H100!"
TIMEOUT_S = 2 * 3600
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "huggingface_hub[hf_transfer]"]
KINDS = [("output_end", "turn"), ("output_marker", "turn"), ("pre_call", "next_turn"), ("call_end", "next_turn"),
         ("call_marker", "next_turn")]
MAX_TOKENS = 32768
# Simulator files written before rows recorded their build, verified to come from one build (ba990399 + src diff
# 7d2b6021; see run_simulate_pairs.LEGACY_BUILDS). sim_pilot20 is a copy of the first 20 rows.
LEGACY_SIM_FILES = {"sim_pilot_2026-10-06.jsonl", "sim_pilot20_2026-10-06.jsonl"}
PT = ZoneInfo("America/Los_Angeles")

app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf"})
         .add_local_python_source("probe_shift"))


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache, "/vol": vol}, timeout=TIMEOUT_S,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def read_shard(cfg: dict, shard: int) -> list[dict]:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.reader import segment_end_positions

    vol.reload()
    run_dir = Path("/vol") / cfg["run"]
    items = [json.loads(l) for l in (run_dir / "inputs" / f"shard_{shard}.jsonl").read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="sdpa").eval()
    base, head = lm.model, lm.lm_head
    layers = list(range(len(base.layers)))
    (run_dir / "acts").mkdir(parents=True, exist_ok=True)
    stats = []
    for n, it in enumerate(items, 1):
        stem = run_dir / "acts" / f"{it['pair']:03d}_{it['version']}"
        if stem.with_suffix(".json").exists():
            stats.append({"pair": it["pair"], "version": it["version"], "status": "already_done"})
            continue
        r = segment_end_positions(tok, it["messages"])
        want = {(k, it[w]): k for k, w in KINDS}
        pos = {want[(p["kind"], p["step"])]: p for p in r["positions"] if (p["kind"], p["step"]) in want and p["action"] == 0}
        missing = [k for k, _ in KINDS if pos.get(k) is None or pos[k]["token"] is None]
        if missing or max(p["token"] for p in pos.values()) >= MAX_TOKENS:
            stats.append({"pair": it["pair"], "version": it["version"], "status": "skipped",
                          "why": f"missing {missing}" if missing else "past the 32k cut"})
            continue
        idx = [pos[k]["token"] for k, _ in KINDS]
        cut = max(idx) + 1
        out = next(o for o in r["outputs"] if o["step"] == it["turn"] and o["action"] == 0)
        acts = {}

        def hook_for(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                acts[layer] = h[0].index_select(0, torch.tensor(idx, device=h.device)).to(torch.float16).cpu()
            return hook

        handles = [base.layers[l].register_forward_hook(hook_for(l)) for l in layers]
        t0 = time.time()
        try:
            with torch.inference_mode():
                ids = torch.tensor([r["ids"][:cut]], device="cuda")
                hidden = base(input_ids=ids, use_cache=False).last_hidden_state[0]
                # log p(token t | tokens before t) for step k's output content tokens
                targets = torch.arange(max(out["start"], 1), out["end"], device="cuda")
                lp = (head(hidden.index_select(0, targets - 1)).float().log_softmax(-1)
                      .gather(1, ids[0, targets][:, None])[:, 0].cpu().numpy() if len(targets) else np.zeros(0))
        finally:
            for h in handles:
                h.remove()
        arr = torch.stack([acts[l] for l in layers], 1).numpy()  # [position, layer, hidden]
        np.savez(stem.with_suffix(".npz"), acts=arr, logprobs=lp.astype(np.float32))
        meta = {"pair": it["pair"], "version": it["version"], "kinds": [k for k, _ in KINDS], "tokens": idx,
                "token_text": [pos[k]["token_text"] for k, _ in KINDS], "n_tokens_full": r["n_tokens"], "n_tokens_run": cut,
                "output_tokens": [out["start"], out["end"]], "seconds": round(time.time() - t0, 2),
                "nonfinite": int((~np.isfinite(arr)).sum())}
        stem.with_suffix(".json").write_text(json.dumps(meta))
        stats.append({"pair": it["pair"], "version": it["version"], "status": "ok", "seconds": meta["seconds"],
                      "n_tokens_run": cut, "nonfinite": meta["nonfinite"]})
        if n % 20 == 0:
            vol.commit()
            print(f"shard {shard}: {n}/{len(items)}", flush=True)
    (run_dir / "stats").mkdir(exist_ok=True)
    (run_dir / "stats" / f"shard_{shard}.json").write_text(json.dumps(stats))
    vol.commit()
    return stats


@app.local_entrypoint()
def main(pairs: str, sims: str, run: str, shards: int = 4):
    """`--sims` takes one file or several, comma-separated (pilot and scale-up); a pair's first ok result wins."""
    import hashlib

    from probe_shift import reader as reader_module
    from probe_shift.models import QWEN3_32B

    pairs_path = Path(pairs)
    recs = {json.loads(l)["pair"]: json.loads(l) for l in pairs_path.read_text().splitlines()}
    pairs_sha = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    sim, builds = {}, set()
    sim_paths = [Path(s) for s in sims.split(",")]
    for p in sim_paths:
        for line in p.read_text().splitlines():
            r = json.loads(line)
            assert r["pairs_sha256"] == pairs_sha, f"{p} holds simulator results for another pairs file"
            if r["status"] == "ok":
                # One simulator build per read run; rows with no recorded build only from the verified legacy files.
                assert "build" in r or p.name in LEGACY_SIM_FILES, f"{p} has rows with no recorded simulator build"
                builds.add(json.dumps(r.get("build", "legacy-2026-10-06"), sort_keys=True))
                sim.setdefault(r["pair"], r["result"])
    assert len(builds) <= 1, f"--sims mixes {len(builds)} simulator builds; read one build at a time"
    items = []
    for pair, res in sorted(sim.items()):
        rec = recs[pair]
        msgs = rec["reader_messages"]
        i = rec["out_msg_index"]
        assert msgs[i]["role"] == "tool" and msgs[i]["content"] == rec["real_output"], f"pair {pair}: step k's output is not at out_msg_index"
        sim_msgs = [dict(m) for m in msgs]
        sim_msgs[i]["content"] = res["reader_output"]
        for version, m in (("real", msgs), ("sim", sim_msgs)):
            items.append({"pair": pair, "version": version, "turn": rec["turn"], "next_turn": rec["next_turn"], "messages": m})
    cfg = {"run": run, "model": QWEN3_32B["name"], "revision": QWEN3_32B["revision"], "dtype": "bfloat16",
           "layers": "all 64 (decoder block outputs)", "kinds": KINDS, "max_tokens": MAX_TOKENS,
           "pairs_file": pairs_path.name, "pairs_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
           "sims_files": [p.name for p in sim_paths],
           "sims_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in sim_paths],
           # The code and environment that write the activations: a resume under different ones is refused below.
           "code_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in (Path(__file__), Path(reader_module.__file__))},
           "pip": PIP, "image_python": "3.12", "gpu": GPU}
    items.sort(key=lambda it: -sum(len(json.dumps(m)) for m in it["messages"]))
    shard_items = [items[s::shards] for s in range(shards)]
    try:
        stored = json.loads(b"".join(vol.read_file(f"{run}/config.json")))
    except (FileNotFoundError, modal.exception.NotFoundError):
        stored = None
    if stored is not None and stored != json.loads(json.dumps(cfg)):
        raise ValueError(f"{run} exists with different settings; use a new run name")
    with vol.batch_upload(force=True) as batch:
        for s, sh in enumerate(shard_items):
            batch.put_file(io.BytesIO("".join(json.dumps(x) + "\n" for x in sh).encode()), f"{run}/inputs/shard_{s}.jsonl")
        batch.put_file(io.BytesIO(json.dumps(cfg, indent=1).encode()), f"{run}/config.json")
        # The raw inputs and simulator outputs live on the Volume, not in git.
        batch.put_file(str(pairs_path), f"raw/{pairs_path.name}")
        for p in sim_paths:
            batch.put_file(str(p), f"raw/{p.name}")
    print(f"uploaded {len(items)} items ({len(sim)} pairs) in {shards} shards to {VOLUME}/{run}", flush=True)
    t0 = time.time()
    stats = [s for shard_stats in read_shard.starmap([(cfg, s) for s in range(shards)]) for s in shard_stats]
    stamp = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
    out = {"wall_seconds": round(time.time() - t0), "config": cfg | {"modal": {"app": APP, "gpu": GPU, "volume": VOLUME,
                                                                              "shards": shards, "pip": PIP}},
           "items": stats}
    with open(HERE / "results" / f"read_{run}_{stamp}.json", "x") as f:
        f.write(json.dumps(out, indent=1))
    from collections import Counter
    print(Counter(s["status"] for s in stats), "GPU seconds", round(sum(s.get("seconds", 0) for s in stats)),
          "non-finite", sum(s.get("nonfinite", 0) for s in stats))
