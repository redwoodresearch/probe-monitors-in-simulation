"""Overnight chain (RULE.md here; 09 + 0d's Q4 rule via 4f, 10-14 ~21:30): compare a re-extraction (v2) with the original (v1) of
the SAME staged inputs, EVERY epoch of every staged arm, at EVERY layer. Per epoch (matched by key; the extractor's meta file name):
same calls (kind, readout, row present, row_mean present) and n_tokens, then every last-token and span-mean row compared in
float32 at each layer: bit-exact or not, max |diff|. Keys, counts and numbers only. Runs on Modal (one container per arm and layer
group, spawned from a remote driver so a laptop disconnect cannot cancel it); the result goes to the Volume
(<v2 root>/compare_v1v2.json) and to results/ when the client is still attached.

  modal run --detach experiments/2026-10-14_q3_overnight_chain/compare_v1v2.py --v1 <root> --v2 <root>
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q3chain_cmpv12_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
GROUPS = 8  # layer groups per arm
app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=image, cpu=4, memory=32768, timeout=3600, retries=0, volumes={"/ev": vol.read_only()})
def compare_part(v1: str, v2: str, arm: str, layers: list) -> dict:
    import numpy as np

    cache = {}

    def rows_of(root, chunk, layer):
        k = (root, chunk, layer)
        if k not in cache:
            cache[k] = np.load(f"/ev/{root}/by_layer/L{layer}/{chunk}.npy", mmap_mode="r")
        return cache[k]

    out = {}
    for f in sorted(Path("/ev", v1, "meta", arm).glob("*.json")):
        a = json.loads(f.read_text())
        g = Path("/ev", v2, "meta", arm, f.name)
        if not g.exists():
            out[a["key"]] = {"missing_in_v2": True}
            continue
        b = json.loads(g.read_text())
        same = a["key"] == b["key"] and a["n_tokens"] == b["n_tokens"] and len(a["calls"]) == len(b["calls"]) and all(
            (x["kind"], x["readout"], x["row"] is None, "row_mean" in x) == (y["kind"], y["readout"], y["row"] is None, "row_mean" in y)
            for x, y in zip(a["calls"], b["calls"]))
        if not same:
            out[a["key"]] = {"calls_or_length_differ": True}
            continue
        pairs = [(a["row_start"] + x[k], b["row_start"] + y[k]) for x, y in zip(a["calls"], b["calls"]) for k in ("row", "row_mean")
                 if x.get(k) is not None]
        ai, bi = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        exact, diff = True, 0.0
        for layer in layers:
            x = np.asarray(rows_of(v1, a["chunk"], layer)[ai], np.float32)
            y = np.asarray(rows_of(v2, b["chunk"], layer)[bi], np.float32)
            exact &= bool(np.array_equal(x, y))
            diff = max(diff, float(np.abs(x - y).max(initial=0.0)))
        out[a["key"]] = {"rows": len(pairs), "bit_exact": exact, "max_abs_diff": diff, "chunk_v1": a["chunk"], "chunk_v2": b["chunk"]}
    v2_keys = {json.loads(g.read_text())["key"] for g in Path("/ev", v2, "meta", arm).glob("*.json")}
    return {"arm": arm, "layers": layers, "epochs": out, "v2_only": sorted(v2_keys - set(out))}


@app.function(image=image, cpu=1, memory=2048, timeout=2 * 3600, retries=0, volumes={"/ev": vol})
def driver(v1: str, v2: str) -> dict:
    import io

    for r in (v1, v2):
        if not Path("/ev", r, "complete.json").exists():
            raise FileNotFoundError(f"{r}: no complete.json")
    c1, c2 = (json.loads(Path("/ev", r, "config.json").read_text()) for r in (v1, v2))
    if c1["inputs_sha256"] != c2["inputs_sha256"]:
        raise ValueError("v1 and v2 extracted different staged inputs")
    arms = sorted(k[:-len(".jsonl")] for k in c1["inputs_sha256"])
    groups = [list(range(g, 64, GROUPS)) for g in range(GROUPS)]
    parts = list(compare_part.starmap([(v1, v2, arm, ls) for arm in arms for ls in groups]))
    per = {}
    for p in parts:
        for k, r in p["epochs"].items():
            e = per.setdefault(p["arm"], {}).setdefault(k, {"layers": 0, "bit_exact": True, "max_abs_diff": 0.0, "problems": []})
            if "rows" not in r:
                e["problems"].append(sorted(r))
                e["bit_exact"] = False
                continue
            e["layers"] += len(p["layers"])
            e["bit_exact"] &= r["bit_exact"]
            e["max_abs_diff"] = max(e["max_abs_diff"], r["max_abs_diff"])
            e["rows"], e["chunk_v1"], e["chunk_v2"] = r["rows"], r["chunk_v1"], r["chunk_v2"]
        if p["v2_only"]:
            per.setdefault(p["arm"], {})["__v2_only__"] = p["v2_only"]
    summary = {arm: {"compared": sum(1 for k in d if k != "__v2_only__"), "identical": sum(1 for k, e in d.items() if k != "__v2_only__" and e["bit_exact"] and e["layers"] == 64),
                     "max_abs_diff": max((e["max_abs_diff"] for k, e in d.items() if k != "__v2_only__"), default=0.0), "v2_only": len(d.get("__v2_only__", []))}
               for arm, d in per.items()}
    out = {"v1": v1, "v2": v2, "inputs_sha256": c1["inputs_sha256"], "layers": 64, "summary": summary,
           "all_identical": all(s["compared"] == s["identical"] and s["v2_only"] == 0 for s in summary.values()), "per_epoch": per}
    with vol.batch_upload(force=True) as b:
        b.put_file(io.BytesIO(json.dumps(out, indent=1).encode()), f"{v2}/compare_v1v2.json")
    return out


@app.local_entrypoint()
def main(v1: str, v2: str):
    call = driver.spawn(v1, v2)
    print(json.dumps({"driver_call_id": call.object_id, "v1": v1, "v2": v2}), flush=True)
    r = call.get()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"compare_v1v2_{v2}_{stamp}.json"
    part = dest.with_suffix(".part")
    part.write_text(json.dumps(r, indent=1))
    part.rename(dest)
    print(json.dumps({"file": dest.name, "summary": r["summary"], "all_identical": r["all_identical"]}))
