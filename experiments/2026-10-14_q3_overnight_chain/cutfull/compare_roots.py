"""d9's CUT fork: the writer attestation of a cutfull extraction root (09 via d9 10-16, the Q4 precedent @ 7c86d8397): a FULL fresh
re-extraction B of the same staged inputs is compared BIT FOR BIT with the routed root A on EVERY staged epoch, at all 64 layers
(every row of the epoch: last-token and span-mean). Identical everywhere -> A stands; otherwise B replaces A at a new revision.
Also records each shard's GPU from both roots' shard results (B must be H200 on every shard). Modal CPU; keys, counts, hashes only.

  CUTCMP_ROOTS='{"staged_set": S, "a": rootA, "b": rootB}' uv run --no-project --python 3.12 --with modal==1.6.1 \
      modal run --detach experiments/2026-10-14_q3_overnight_chain/cutfull/compare_roots.py
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
REPO = CHAIN.parents[1] if len(CHAIN.parents) > 1 else CHAIN
APP = "probe-shift_aswin-surya_sub-q3cutfull_cmproots_kill-2026-11-14"
UNC_VOL = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
N_LAYERS = 64
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=image, cpu=8, memory=32768, timeout=3 * 3600, retries=0, volumes={"/unc": modal.Volume.from_name(UNC_VOL, version=2).read_only()})
def compare(spec: dict) -> dict:
    import numpy as np

    S, A, B = (Path("/unc") / spec[k] for k in ("staged_set", "a", "b"))
    out = {**spec, "epochs": {}, "fail": []}
    gpus = {}
    for name, R in (("a", A), ("b", B)):
        gpus[name] = {p.name: json.loads(p.read_text()).get("gpu_name") for p in sorted((R / "shard_results").glob("shard*.json"))}
    out["gpus"] = gpus
    ca, cb = json.loads((A / "config.json").read_text()), json.loads((B / "config.json").read_text())
    want_b = {f"shard{k:03d}_a1.json" for k in range(cb["shards"])}  # root b is ONE fresh attempt: every shard's own record, on H200
    if set(gpus["b"]) != want_b or any(g != "NVIDIA H200" for g in gpus["b"].values()):
        out["fail"].append(f"root b: shard records {sorted(gpus['b'])} vs {sorted(want_b)}, or a shard not on NVIDIA H200")
    keys_cfg = sorted((set(ca) | set(cb)) - {"run"})  # every setting but the root's own name
    out["config"] = {k: [ca.get(k), cb.get(k)] for k in keys_cfg}
    if any(ca.get(k) != cb.get(k) for k in keys_cfg) or not all(k in ca for k in ("inputs", "inputs_sha256", "model", "revision", "pad_to", "chunk_size")):
        out["fail"].append("configs differ or lack a required setting")
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((S / "inputs").glob("*.jsonl"))}
    if files != ca.get("inputs_sha256") or ca.get("inputs") != spec["staged_set"]:  # the staged files ARE the ones both roots read
        out["fail"].append("staged files differ from the roots' inputs")
    cache = {}

    def arr(R, l, chunk):
        k = (str(R), l, chunk)
        if k not in cache:
            cache[k] = np.load(R / "by_layer" / f"L{l}" / f"{chunk}.npy", mmap_mode="r")
        return cache[k]

    n_ep = 0
    for f in sorted((S / "inputs").glob("*.jsonl")):
        arm = f.stem
        for line in f.read_text().splitlines():
            key = json.loads(line)["key"]
            fn = hashlib.sha1(key.encode()).hexdigest()[:20]
            ma, mb = (json.loads((R / "meta" / arm / f"{fn}.json").read_text()) for R in (A, B))
            if key in out["epochs"]:
                raise ValueError(f"{key}: staged twice")
            ep = json.loads(line)
            kept = [c for c in ep["calls"] if c["readout"] is not None]
            n_exp = len(kept) + sum(1 for c in kept if c["kind"] == "output_end" and c.get("span"))  # last-token rows + span-mean rows
            row_ok = (ma["n_rows"] == mb["n_rows"] == n_exp and ma["calls"] == mb["calls"] and ma["n_tokens"] == mb["n_tokens"] == len(ep["ids"])
                      and all(isinstance(m["row_start"], int) and m["row_start"] >= 0 and m["n_rows"] > 0 for m in (ma, mb)))
            diff_layers = []
            if row_ok:
                for l in range(N_LAYERS):
                    fa, fb = arr(A, l, ma["chunk"]), arr(B, l, mb["chunk"])
                    if (fa.ndim != 2 or fb.ndim != 2 or fa.dtype != np.float16 or fb.dtype != np.float16 or fa.shape[1] != fb.shape[1]
                            or ma["row_start"] + ma["n_rows"] > fa.shape[0] or mb["row_start"] + mb["n_rows"] > fb.shape[0]):
                        diff_layers.append(l)  # malformed or out of bounds: never identical
                        continue
                    xa = fa[ma["row_start"]: ma["row_start"] + ma["n_rows"]]
                    xb = fb[mb["row_start"]: mb["row_start"] + mb["n_rows"]]
                    if not np.array_equal(np.asarray(xa).view(np.uint16), np.asarray(xb).view(np.uint16)):  # bit for bit
                        diff_layers.append(l)
            n_ep += 1
            out["epochs"][key] = {"arm": arm, "rows": ma["n_rows"], "meta_equal": row_ok, "bit_identical": row_ok and not diff_layers,
                                  "layers_differing": diff_layers, "a_chunk": ma["chunk"], "b_chunk": mb["chunk"]}
            cache.clear()
    out["n_epochs"] = n_ep
    out["n_bit_identical"] = sum(e["bit_identical"] for e in out["epochs"].values())
    out["identical"] = not out["fail"] and n_ep > 0 and out["n_bit_identical"] == n_ep
    return out


@app.local_entrypoint()
def main():
    spec = json.loads(os.environ["CUTCMP_ROOTS"])
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    r = compare.remote(spec) | {"git": head}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = CHAIN / "results" / "cutfull" / f"compare_roots_{spec['b']}_{stamp}.json"
    dest.open("x").write(json.dumps(r, indent=1))
    print(dest.name, "IDENTICAL" if r["identical"] else "DIFFERENT", f"{r['n_bit_identical']}/{r['n_epochs']}", r["fail"])
