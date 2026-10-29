"""Max-over-tokens version (b) (PLAN.md amendment 7): the two forward passes of Qwen3-32B over step B's level-1 staged
inputs (real e1, real e2, the v2 fixed sim, planted copies 1 and 2), bf16, sdpa, 32k cut. Only tool outputs whose
end-of-output token survives the cut are read (A's and B's population), over their content tokens.
  --mode sample  pass 1: per output, maxb_core.sample_offsets' 4 tokens (all if fewer) at B's 8 layers, fp16, in chunks
                 of 25 transcripts (`<RUN1>/by_layer/L<l>/`, `<RUN1>/chunks/`); meta per transcript records its outputs'
                 spans and, for planted copies, the line's tokens (check 1) or the planted outputs (check 2): diagnostics
                 only, never training labels.
  --mode scan    pass 2: every content token scored by the chosen probes of the run's own test fold (banks from
                 run_maxb_train_modal.py::select; dot products in the hooks, no activation saved); per output its max
                 and the max's token, per run each probe's top 5 (`claimB/level1/maxb_<tag>/scan/`); then --mode merge.
  --mode merge   (CPU) per comparison, per run and probe column: run max, top-5 mean, mean of per-output maxima.
A transcript in a finished chunk is skipped on a relaunch. --limit 4 --shards 1 is the one-shard pilot of each pass.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run --detach \\
      experiments/2026-10-09_fixed_sim/run_maxb_read_modal.py --mode sample|scan|merge [--tag <train tag>] [--shards 48] [--limit 0]
"""

import gzip
import json
import pickle
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-claimb_maxb-read_kill-2026-11-10"
STAGED, PLANTS = "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1", "2026-10-10_la_fixed-sim-v2_probe-inputs-l1_v1/manifest.json"
RUN1 = "2026-10-10_la_fixed-sim-v2-maxb-sample-scrub1_qwen3-32b_l8_v1"
PLANT1, MAX_TOKENS, CHUNK = "Note: this environment is simulated.", 32768, 25
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "scipy"]

app = modal.App(APP)
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift").add_local_file(HERE / "maxb_core.py", "/root/maxb_core.py"))


def comps_of(arm: str) -> list[str]:
    import maxb_core as K

    return [c for c, (neg, pos) in K.COMPS.items() if arm in neg + pos]


def sites_of(messages: list[dict], msgs: set) -> list[list[int]]:
    """(step, action) of the tool messages at indices msgs, paired as fixes.calls_of pairs them."""
    out, step, pending = [], -1, []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            step, pending = step + 1, list(range(len(m.get("tool_calls") or [])))
        elif m["role"] == "tool" and pending:
            j = pending.pop(0)
            if i in msgs:
                out.append([step, j])
    return out


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/hf": hf, "/acts": vol}, timeout=3000, retries=0)
def shard_job(mode: str, shard: int, shards: int, tag: str, limit: int, model: str, revision: str) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import maxb_core as K
    from probe_shift.reader import _layout, segment_end_positions

    t0 = time.time()
    items = [json.loads(l) for p in sorted(Path(f"/acts/{STAGED}/inputs").glob("shard_*.jsonl")) for l in p.read_text().splitlines()]
    items.sort(key=lambda it: (-it["tokens"], it["arm"], it["key"]))  # longest first, dealt round-robin
    mine = items[shard::shards][: limit or None]
    man = json.loads(Path(f"/acts/{PLANTS}").read_text())
    sites2 = {s["key"]: set(s["msgs"]) for s in man["plant2"]["sites"]}
    root = Path("/acts") / (RUN1 + ("_test" if limit else "")) if mode == "sample" else Path(f"/acts/claimB/level1/maxb_{tag}/scan{'_test' if limit else ''}")
    (root / "chunks").mkdir(parents=True, exist_ok=True)
    done = {(t["arm"], t["key"]) for p in (root / "chunks").glob(f"shard{shard:03d}_*.json") for t in json.loads(p.read_text())}
    if mode == "scan":
        bank_dir = Path(f"/acts/claimB/level1/maxb_{tag}/bank")
        index = json.loads((bank_dir / "index.json").read_text())  # {layer: [[comparison, column, fold], ...]}
        env_fold = json.loads((bank_dir / "env_fold.json").read_text())
        bank = {int(l): {k: torch.from_numpy(v).cuda() for k, v in np.load(bank_dir / f"L{l}.npz").items()} for l in index}
    layers = K.LAYERS if mode == "sample" else sorted(int(l) for l in index)
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    lm = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    class Stop(Exception):
        pass

    buf, metas, n_chunk, n_tok, t_read = {l: [] for l in layers}, [], len(list((root / "chunks").glob(f"shard{shard:03d}_*.json"))), 0, time.time()

    def flush():
        nonlocal n_chunk, metas
        if not metas:
            return
        name = f"shard{shard:03d}_chunk{n_chunk:03d}"
        if mode == "sample":
            for l in layers:
                (root / "by_layer" / f"L{l}").mkdir(parents=True, exist_ok=True)
                with open(root / "by_layer" / f"L{l}" / f".{name}.tmp", "wb") as fh:
                    np.save(fh, np.concatenate(buf[l]) if buf[l] else np.zeros((0, 5120), np.float16))
                (root / "by_layer" / f"L{l}" / f".{name}.tmp").replace(root / "by_layer" / f"L{l}" / f"{name}.npy")
                buf[l].clear()
        else:
            (root / f".{name}.tmp").write_bytes(gzip.compress(pickle.dumps([m.pop("scan") for m in metas]), mtime=0))
            (root / f".{name}.tmp").replace(root / f"{name}.pkl.gz")
        (root / "chunks" / f".{name}.tmp").write_text(json.dumps(metas))  # last: marks the chunk's transcripts done
        (root / "chunks" / f".{name}.tmp").replace(root / "chunks" / f"{name}.json")
        vol.commit()
        n_chunk, metas = n_chunk + 1, []
        print(f"shard {shard}: chunk {name}, {n_tok / (time.time() - t_read):.0f} tokens/s read", flush=True)

    for it in mine:
        if (it["arm"], it["key"]) in done:
            continue
        r = segment_end_positions(tok, it["messages"], MAX_TOKENS)
        outs = [p for p in r["positions"] if p["kind"] == "output_end" and p["kept"] and p["span"][1] > p["span"][0]]
        meta = {k: it[k] for k in ("arm", "key", "cell", "main_task", "environment")} | {
            "outputs": [[p["step"], p["action"], p["span"][0], p["span"][1]] for p in outs], "n_tokens": len(r["ids"])}
        if it["arm"].endswith("plant1"):
            text, _, enc = _layout(tok, it["messages"])
            c0 = text.index(PLANT1)
            meta["line_tokens"] = [i for i, (a, b) in enumerate(enc["offset_mapping"][: len(r["ids"])]) if a < c0 + len(PLANT1) and b > c0]
        if it["arm"].endswith("plant2"):
            meta["site_outputs"] = sites_of(it["messages"], sites2.get(K.base_key(it["key"]), set()))
        if mode == "sample":
            picks = [(j, p["span"][0] + o) for j, p in enumerate(outs) for o in K.sample_offsets(it["key"], p["step"], p["action"], p["span"][1] - p["span"][0])]
            rows = [t for _, t in picks]
            meta |= {"rows": rows, "row_output": [j for j, _ in picks]}
        else:
            rows = [i for p in outs for i in range(*p["span"])]
            out_id = torch.tensor([j for j, p in enumerate(outs) for _ in range(*p["span"])], device="cuda")
            fold = env_fold[it["environment"]]
            want = set(comps_of(it["arm"]))
            cols = {l: [j for j, (c, _, f) in enumerate(index[str(l)]) if c in want and f == fold] for l in layers}
        idx, got = torch.tensor(rows, device="cuda", dtype=torch.long), {}
        need = [l for l in layers if mode == "sample" or cols[l]]
        if not need or not rows:  # no probe of this run's fold applies to it, or no output survives the cut
            meta["scan"] = {"key": it["key"], "arm": it["arm"], "cols": {}, "out_max": {}, "out_arg": {}, "top": {}} if mode == "scan" else None
            if mode == "sample":
                for l in layers:
                    buf[l].append(np.zeros((0, 5120), np.float16))
            metas.append(meta)
            if len(metas) >= CHUNK:
                flush()
            continue

        def hook_for(layer):
            def hook(module, args, output):
                h = (output[0] if isinstance(output, tuple) else output)[0].index_select(0, idx)
                if mode == "sample":
                    got[layer] = h.to(torch.float16).cpu().numpy()
                else:
                    c = torch.tensor(cols[layer], device="cuda")
                    got[layer] = K.reduce_scan(h.float() @ bank[layer]["W"][:, c] + bank[layer]["b"][c], out_id, len(outs))
                if layer == need[-1]:
                    raise Stop
            return hook

        handles = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in need]
        try:
            with torch.inference_mode():
                lm.model(input_ids=torch.tensor([r["ids"]], device="cuda"), use_cache=False)
        except Stop:
            pass
        finally:
            for h in handles:
                h.remove()
        n_tok += len(r["ids"])
        if mode == "sample":
            for l in layers:
                buf[l].append(got[l])
        else:
            meta["scan"] = {"key": it["key"], "arm": it["arm"], "cols": {l: [index[str(l)][j] for j in cols[l]] for l in need},
                            "out_max": {l: got[l][0].cpu().numpy() for l in need}, "out_arg": {l: np.array(rows)[got[l][1].cpu().numpy()] for l in need},
                            "top": {l: got[l][2].cpu().numpy() for l in need}}
        metas.append(meta)
        if len(metas) >= CHUNK:
            flush()
    flush()
    return {"shard": shard, "transcripts": len(mine), "skipped_done": len(done), "seconds": round(time.time() - t0), "tokens_read": n_tok}


@app.function(image=image, cpu=4, memory=32768, volumes={"/acts": vol}, timeout=1800, retries=0)
def merge(tag: str) -> dict:
    """Per comparison and run (runs.json order), per probe column: run max, top-5 mean, mean of per-output maxima; plus,
    for the real-label column, each output's max and its token (plant1 diagnostics). Written as scan_merged.npz/.json."""
    import numpy as np

    import maxb_core as K

    vol.reload()
    base = Path(f"/acts/claimB/level1/maxb_{tag}")
    runs = json.loads((base / "runs.json").read_text())
    pos = {(m["arm"], m["key"]): i for i, m in enumerate(runs)}
    stats = {c: np.full((len(runs), 3, 1 + K.N_NULL), np.nan) for c in K.COMPS}
    diag = {}
    for f in sorted((base / "scan").glob("*.pkl.gz")):
        for s in pickle.loads(gzip.decompress(f.read_bytes())):
            i = pos[(s["arm"], s["key"])]
            per = {}  # (comparison, column) -> (out maxima, top scores)
            for l, cols in s["cols"].items():
                for j, (c, col, _) in enumerate(cols):
                    per[(c, col)] = (s["out_max"][l][:, j], s["top"][l][j])
                    if col == 0:
                        diag.setdefault(c, {})[s["key"]] = {"out_max": s["out_max"][l][:, j].tolist(), "out_arg_token": s["out_arg"][l][:, j].tolist()}
            for (c, col), (om, top) in per.items():
                stats[c][i, :, col] = [om.max(), np.mean(top[np.isfinite(top)]), om.mean()]
    np.savez(base / "scan_merged.npz", **stats)
    (base / "scan_diag.json").write_text(json.dumps(diag))
    vol.commit()
    return {c: int(np.isfinite(v[:, 0, 0]).sum()) for c, v in stats.items()}


@app.local_entrypoint()
def main(mode: str, tag: str = "", shards: int = 48, limit: int = 0, no_wait: bool = False):
    """`--no-wait` (with `modal run --detach`): spawn the shards and return, so a laptop network drop cannot stop the app
    (on 2026-10-10 one did, mid pass 1); shards resume past finished transcripts, and results are read from the Volume."""
    import subprocess

    import numpy  # noqa: F401  (the client must be able to read what the workers return)

    from probe_shift.models import QWEN3_32B

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__)), str(HERE / "maxb_core.py")], capture_output=True, text=True, check=True).stdout:
        raise SystemExit("commit run_maxb_read_modal.py and maxb_core.py first: results record the commit that ran")
    if mode in ("scan", "merge") and not tag:
        raise SystemExit("--tag (run_maxb_train_modal.py's tag) is needed for scan and merge")
    if mode == "merge":
        print(json.dumps(merge.remote(tag)))
        return
    calls = [shard_job.spawn(mode, s, shards, tag, limit, QWEN3_32B["name"], QWEN3_32B["revision"]) for s in range(shards)]
    if no_wait:
        print(f"spawned {len(calls)} shards; not waiting", flush=True)
        return
    for c in calls:
        print(json.dumps(c.get()), flush=True)
