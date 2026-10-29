"""Llama-3.2-1B through the halves test (RULE.md here; Aswin about 11:55 PT 10-12, via 0d: "Buy full evened run (~$9–13)"):
the reads. Llama reads the halves' evened texts exactly as the small-model baseline read the level-1 texts. Keys, counts and
activations only; no text leaves the containers.

  --mode gate     1 L4: the 4 unchanged gate runs staged with the evened texts (kind h3_gate_unchanged) are read here and
                  must equal the stored Llama dump (2026-10-10_la_fixed-sim-v2-b-scrub1_llama32-1b_l8_v1) in positions and
                  rows bitwise at all 8 modules (cosine reported as a diagnostic only), before any bulk read. gate.json; never overwritten.
  --mode extract  SHARDS L4 shards, spawned only after the gate passed: every other staged item (H3's 831 changed test
                  runs, B-dev's 852 changed dev runs and 100 dev planted copies) read once; one .npz per item (rows at
                  the 8 modules, positions, meta). Resumable: an item whose file exists from this script is skipped.

The reader is copied from experiments/2026-10-10_claimB_small_lm/run_small_lm_modal.py @ 6d7caaf5 (MODELS["llama32-1b"],
load, locate, read; same pins and image), so the gate can be bitwise. The staged texts are the Qwen3.5 halves staging's
(run_stage_evened_modal.py, sha256 23e04d5c..., read-only).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal deploy \\
      experiments/2026-10-12_claimB_1b_halves/run_1b_read_modal.py
  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python \\
      experiments/2026-10-12_claimB_1b_halves/run_1b_read_modal.py gate|extract
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
MODEL, REVISION, LAYERS = "meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6", [4, 7, 9, 10, 11, 12, 14, 15]
QWEN, QWEN_REV = "Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"  # the layout's tokenizer (probe_shift.models)
MAX_TOKENS = 32768
STAGED = "2026-10-12_la_fixed-sim-v2_probe-inputs-b-l1-evened-halves_v1"
STAGE_SHA = "23e04d5cef13542c398ccfc190d0663d025508297e461c15a380535f2a785ac6"
DUMP = "2026-10-10_la_fixed-sim-v2-b-scrub1_llama32-1b_l8_v1"  # the stored Llama reads (the gate's reference)
DUMP_SCRIPT = "8ddde7f9"  # run_small_lm_modal.py's sha256 prefix in that dump's manifest (ACTIVATIONS.md)
SHARDS = 8
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT = f"claimB/halves_1b_v1/read_{SCRIPT_SHA[:12]}"


def download():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, revision=REVISION)
    snapshot_download(QWEN, revision=QWEN_REV, allow_patterns=["*.json", "*.txt", "*.model", "merges.txt", "vocab.json"])


app = modal.App("probe-shift_aswin-surya_sub-claimb-1b-read_kill-2026-11-12")
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "scipy", "scikit-learn==1.7.2", "huggingface_hub")
         .run_function(download, secrets=[modal.Secret.from_name("huggingface-secret-aswin")], timeout=30 * 60)
         .env({"HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift"))


def load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    qtok = AutoTokenizer.from_pretrained(QWEN, revision=QWEN_REV)
    ltok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    return qtok, ltok, lm


def locate(qtok, ltok, item):
    """run_small_lm_modal.locate @6d7caaf5, unchanged."""
    import numpy as np

    from probe_shift.reader import _layout, segment_end_positions

    text, found, enc = _layout(qtok, item["messages"])
    seg = segment_end_positions(qtok, item["messages"], MAX_TOKENS)
    cut = enc["offset_mapping"][len(seg["ids"]) - 1][1]
    spans = {(f[1], f[2]): (f[4], f[5], f[3]) for f in found if f[0] == "tool_output"}
    qkept = {(p["step"], p["action"]): p["kept"] for p in seg["positions"] if p["kind"] == "output_end"}
    lenc = ltok(text[:cut], return_offsets_mapping=True, add_special_tokens=True)
    offs = np.array(lenc["offset_mapping"])
    positions, idx = [], []
    for (s, a), qk in sorted(qkept.items()):  # every output Qwen read a readout for; kept only if inside Qwen's cut
        ca, cb, tool = spans[(s, a)]
        inside = np.flatnonzero((offs[:, 0] >= ca) & (offs[:, 1] <= cb) & (offs[:, 1] > offs[:, 0])) if qk else []
        if len(inside):
            positions.append({"kind": "output_end", "step": s, "action": a, "tool": tool, "kept": True, "row": len(idx), "token": int(inside[-1])})
            idx.append(int(inside[-1]))
        else:
            positions.append({"kind": "output_end", "step": s, "action": a, "tool": tool, "kept": False, "token": None,
                              "why": "outside Qwen's cut" if not qk else "no Llama token inside the output"})
    return positions, idx, lenc["input_ids"]


def read(qtok, ltok, lm, item) -> tuple[dict, list]:
    """run_small_lm_modal.read @6d7caaf5, unchanged: {layer: rows [n, d] fp16} and the positions for one transcript."""
    import torch

    positions, idx, ids = locate(qtok, ltok, item)
    if not idx:
        raise ValueError(f"{item['key']}: no output has a readout")
    got = {}

    def hook_for(layer):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            got[layer] = h[0].index_select(0, torch.tensor(idx, dtype=torch.long, device="cuda")).to(torch.float16).cpu().numpy()
        return hook

    handles = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in LAYERS]
    try:
        with torch.inference_mode():
            lm.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return got, positions


def staged() -> tuple[dict, dict]:
    """(manifest, {(arm, key): item}) of the staged evened run, after checking its staging script and file hashes."""
    base = Path("/acts") / STAGED
    man = json.loads((base / "manifest.json").read_text())
    if man["script_sha256"] != STAGE_SHA:
        raise ValueError("the staged run was made by another staging script")
    items = {}
    for name, want in man["files"].items():
        body = (base / "inputs" / name).read_bytes()
        if hashlib.sha256(body).hexdigest() != want:
            raise ValueError(f"{name}: sha256 differs from the staging manifest")
        for line in body.decode().splitlines():
            it = json.loads(line)
            items[(it["arm"], it["key"])] = it
    if len(items) != len(man["items"]):
        raise ValueError("staged items and manifest rows differ in number")
    return man, items


def fid(arm: str, key: str) -> str:
    return hashlib.sha256(f"{arm}|{key}".encode()).hexdigest()[:20]  # run_small_lm_modal's meta file id


def dump_rows(arm: str, key: str) -> tuple[dict, list]:
    """The stored Llama dump's rows (8 modules) and output_end positions for one transcript."""
    import numpy as np

    m = json.loads((Path("/acts") / DUMP / "meta" / arm / f"{fid(arm, key)}.json").read_text())
    kept = [p for p in m["positions"] if p["kind"] == "output_end" and p["kept"]]
    rows = {l: np.load(Path("/acts") / DUMP / "by_layer" / f"L{l}" / f"{m['chunk']}.npy")[m["row_start"] + np.array([p["row"] for p in kept], int)]
            for l in LAYERS}
    return rows, m["positions"]


@app.function(image=image, gpu="L4", cpu=4, memory=32768, volumes={"/acts": acts}, timeout=30 * 60, retries=0)
def gate() -> dict:
    import numpy as np

    out = Path("/acts") / OUT
    if (out / "gate.json").exists():
        raise ValueError("gate.json exists for this script version")
    dman = json.loads((Path("/acts") / DUMP / "manifest.json").read_text())
    if not dman["script_sha256"].startswith(DUMP_SCRIPT) or (dman["model"], dman["revision"], dman["layers"]) != (MODEL, REVISION, LAYERS):
        raise ValueError("the stored dump was made by another extractor, model or layer set")
    man, items = staged()
    qtok, ltok, lm = load()
    res = []
    for row in (r for r in man["items"] if r["kind"] == "h3_gate_unchanged"):
        got, pos = read(qtok, ltok, lm, items[(row["arm"], row["key"])])
        ref, rpos = dump_rows(row["arm"], row["key"])
        g = {"key": row["key"], "positions_equal": pos == rpos}
        for l in LAYERS:
            a, b = got[l].astype(np.float32), ref[l].astype(np.float32)
            same = a.shape == b.shape
            g[f"L{l}"] = {"bitwise": bool(same and np.array_equal(got[l], ref[l])),
                          "min_cosine": float(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))).min()) if same else None}
        g["pass"] = g["positions_equal"] and all(g[f"L{l}"]["bitwise"] for l in LAYERS)  # bitwise only (RULE.md)
        res.append(g)
    out_res = {"gate": res, "passed": len(res) == 4 and all(g["pass"] for g in res), "script_sha256": SCRIPT_SHA}
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(out_res))
    acts.commit()
    return out_res


@app.function(image=image, gpu="L4", cpu=4, memory=32768, volumes={"/acts": acts}, timeout=100 * 60, retries=0)
def extract(shard: int) -> dict:
    import io

    import numpy as np

    t0 = time.time()
    out = Path("/acts") / OUT
    g = json.loads((out / "gate.json").read_text())
    if not g["passed"] or g["script_sha256"] != SCRIPT_SHA:
        raise ValueError("no passed gate from this script version")
    man, items = staged()
    rows = sorted((r for r in man["items"] if r["kind"] != "h3_gate_unchanged"), key=lambda r: (r["arm"], r["key"]))[shard::SHARDS]
    (out / "reads").mkdir(parents=True, exist_ok=True)
    meta0 = {"script_sha256": SCRIPT_SHA, "model": MODEL, "revision": REVISION, "layers": LAYERS, "staged": STAGED}
    todo = [r for r in rows if not (out / "reads" / f"{fid(r['arm'], r['key'])}.npz").exists()]
    qtok, ltok, lm = load()
    stats = {"shard": shard, "items": len(rows), "skipped_done": len(rows) - len(todo), "rows": 0, "nonfinite": 0, "no_readout": 0}
    for n, r in enumerate(todo):
        got, pos = read(qtok, ltok, lm, items[(r["arm"], r["key"])])
        stats["rows"] += len(got[LAYERS[0]])
        stats["nonfinite"] += sum(int((~np.isfinite(got[l])).sum()) for l in LAYERS)
        stats["no_readout"] += sum(not q["kept"] for q in pos)
        meta = meta0 | {k: r[k] for k in ("kind", "arm", "key", "cell", "environment", "main_task", "range_tool_ordinals")} | {"positions": pos}
        buf = io.BytesIO()
        np.savez(buf, meta=json.dumps(meta), **{f"L{l}": got[l] for l in LAYERS})
        dst = out / "reads" / f"{fid(r['arm'], r['key'])}.npz"
        (dst.parent / f".{dst.name}.tmp").write_bytes(buf.getvalue())
        (dst.parent / f".{dst.name}.tmp").replace(dst)
        if n % 25 == 24:
            acts.commit()
            print(f"[1b-read] shard {shard}: {n + 1}/{len(todo)} read, {time.time() - t0:.0f}s", flush=True)
    stats["seconds"] = round(time.time() - t0)
    (out / "stats").mkdir(exist_ok=True)
    (out / "stats" / f"shard_{shard}.json").write_text(json.dumps(stats))
    acts.commit()
    return stats


@app.function(image=image, cpu=1, memory=1024, timeout=120, retries=0)
def version() -> str:
    return SCRIPT_SHA


def main(mode: str):
    from probe_shift.models import QWEN3_32B

    if (QWEN3_32B["name"], QWEN3_32B["revision"]) != (QWEN, QWEN_REV):
        raise SystemExit("the layout's Qwen3-32B pin differs from probe_shift.models")
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    if modal.Function.from_name(app.name, "version").remote() != SCRIPT_SHA:
        raise SystemExit("the deployed app runs other code: `modal deploy` this revision first")
    if mode == "gate":
        print(json.dumps(modal.Function.from_name(app.name, "gate").remote(), indent=1))
    elif mode == "extract":
        calls = [modal.Function.from_name(app.name, "extract").spawn(s) for s in range(SHARDS)]
        print("spawned:", [c.object_id for c in calls], flush=True)
        for c in calls:
            print(json.dumps(c.get(timeout=110 * 60)), flush=True)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main(sys.argv[1])
