"""1e-4 real honest runs (RULE.md, Amendment 6): compare re-read rows with batch C's stored rows (q4b_extract_c_v1 on 2e's
Volume, read only). Used twice: the positive control (`--run rh_c_ref_extract_v1 --arm honest_b`: batch C's 16 shard-longest and
4 seeded epochs, re-staged and re-extracted with this job's copies) and the per-shard gate of the main extraction
(`--run rh1e4_extract_v1 --arm repro-c`: in every shard, batch C's longest staged epoch and one seeded epoch).
Per re-read epoch (its key minus any "~ref<j>" suffix is batch C's key): the calls must match batch C's meta call for call
(kind, readout, row present, row_mean present); at layers 16, 32, 48, 63 every last-token and span-mean row is compared in
float32: bit-exact or not, cosine, max |diff|. Pass (2e's bar): every row bit-exact, or every row cosine >= 0.9999 and
max |diff| <= 0.125. Results per epoch and per shard (from the chunk name); if any shard fails, the whole extraction is
redone under a fresh run name on H200 (RULE.md, Amendment 6, Codex round 6). Keys, counts and numbers only.

Completeness first (Codex round 6): the run's complete.json must exist; the re-read metas of the arm must be exactly the keys of
the staged <inputs>/inputs/<arm>.jsonl; for `repro-c`, every one of the run's shards (config.json) must hold exactly one copy of
each reference source. Anything missing fails the comparison.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-14_q3_q35_realhonest_1e4/compare_repro.py \\
      --run <extraction run> --arm <arm>
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q3realhonest_cmprep_kill-2026-11-14"
MINE = "probe-shift_aswin-surya_sub-q3realhonest_data_kill-2026-11-14"
C_VOL, C_RUN, C_ARM = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "q4b_extract_c_v1", "honest_b"
LAYERS = (16, 32, 48, 63)
BAR = {"cosine": 0.9999, "max_abs_diff": 0.125}
app = modal.App(APP)


@app.function(image=modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3"), cpu=4, memory=32768, timeout=3600,
              retries=0, volumes={"/ev": modal.Volume.from_name(MINE, version=2).read_only(),
                                  "/c": modal.Volume.from_name(C_VOL, version=2).read_only()})
def compare(run: str, arm: str) -> dict:
    import numpy as np

    fn = lambda key: hashlib.sha1(key.encode()).hexdigest()[:20]  # noqa: E731  (the extractor's filename())
    cache = {}

    def rows_of(root, chunk, layer):
        k = (root, chunk, layer)
        if k not in cache:
            cache[k] = np.load(f"{root}/by_layer/L{layer}/{chunk}.npy", mmap_mode="r")
        return cache[k]

    mine_root, c_root = f"/ev/{run}", f"/c/{C_RUN}"
    cfg = json.loads(Path(mine_root, "config.json").read_text())
    if not Path(mine_root, "complete.json").exists():
        raise FileNotFoundError(f"{run}: no complete.json; the extraction did not finish")
    want = {json.loads(line)["key"] for line in Path("/ev", cfg["inputs"], "inputs", f"{arm}.jsonl").read_text().splitlines()}
    metas = sorted(Path(mine_root, "meta", arm).glob("*.json"))
    got = {json.loads(f.read_text())["key"] for f in metas}
    if got != want:
        raise ValueError(f"{run}/{arm}: {len(want - got)} staged keys without a meta, {len(got - want)} metas not staged")
    out, shards = [], {}
    for f in metas:
        m = json.loads(f.read_text())
        ckey = m["key"].split("~ref", 1)[0]
        c = json.loads(Path(c_root, "meta", C_ARM, f"{fn(ckey)}.json").read_text())
        res = {"key": m["key"], "c_key": ckey, "chunk": m["chunk"], "n_tokens": [m["n_tokens"], c["n_tokens"]]}
        same_calls = len(m["calls"]) == len(c["calls"]) and all(
            (a["kind"], a["readout"], a["row"] is None, "row_mean" in a) == (b["kind"], b["readout"], b["row"] is None, "row_mean" in b)
            for a, b in zip(m["calls"], c["calls"]))
        res["calls_match"] = same_calls
        if same_calls and m["n_tokens"] == c["n_tokens"]:
            pairs = [(m["row_start"] + a[k], c["row_start"] + b[k]) for a, b in zip(m["calls"], c["calls"]) for k in ("row", "row_mean")
                     if a.get(k) is not None]
            mi, ci = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
            exact, cos_min, diff_max = True, 1.0, 0.0
            for layer in LAYERS:
                x = np.asarray(rows_of(mine_root, m["chunk"], layer)[mi], np.float32)
                y = np.asarray(rows_of(c_root, c["chunk"], layer)[ci], np.float32)
                if not (np.isfinite(x).all() and np.isfinite(y).all()):
                    raise ValueError(f"{m['key']}: non-finite rows at L{layer}")
                exact &= bool(np.array_equal(x, y))
                cos = (x * y).sum(1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12)
                cos_min, diff_max = min(cos_min, float(cos.min(initial=1.0))), max(diff_max, float(np.abs(x - y).max(initial=0.0)))
            res |= {"rows": len(pairs), "bit_exact": exact, "min_cosine": cos_min, "max_abs_diff": diff_max,
                    "passes": exact or (cos_min >= BAR["cosine"] and diff_max <= BAR["max_abs_diff"])}
        else:
            res |= {"passes": False, "why": "calls or length differ from batch C's meta"}
        out.append(res)
        s = m["chunk"].split("_", 1)[0]
        shards.setdefault(s, []).append(res["passes"])
    coverage = None
    if arm == "repro-c":  # each shard: exactly one copy of each reference source
        per = {}
        for r in out:
            per.setdefault(r["chunk"].split("_", 1)[0], []).append(r["c_key"])
        sources = sorted({r["c_key"] for r in out})
        coverage = {"shards_expected": cfg["shards"], "shards_seen": len(per), "sources": len(sources),
                    "every_shard_one_of_each": len(per) == cfg["shards"] and all(sorted(v) == sources for v in per.values())}
        if not coverage["every_shard_one_of_each"]:
            raise ValueError(f"reference coverage: {coverage}")
    return {"run": run, "arm": arm, "layers": LAYERS, "bar": BAR, "coverage": coverage, "epochs": len(out), "passing": sum(r["passes"] for r in out),
            "bit_exact": sum(bool(r.get("bit_exact")) for r in out), "min_cosine": min((r.get("min_cosine", -1.0) for r in out), default=None),
            "shards": {s: all(v) for s, v in sorted(shards.items())}, "failing_shards": sorted(s for s, v in shards.items() if not all(v)),
            "per_epoch": out}


@app.local_entrypoint()
def main(run: str, arm: str):
    r = compare.remote(run, arm)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"compare_repro_{run}_{arm}_{stamp}.json"
    dest.open("x").write(json.dumps(r, indent=1))
    print(json.dumps({k: r[k] for k in ("run", "arm", "epochs", "passing", "bit_exact", "min_cosine", "failing_shards")}))
    print("->", dest)
    if r["passing"] != r["epochs"]:
        raise SystemExit(f"{r['epochs'] - r['passing']} of {r['epochs']} epochs fail the bar; shards {r['failing_shards']}")
