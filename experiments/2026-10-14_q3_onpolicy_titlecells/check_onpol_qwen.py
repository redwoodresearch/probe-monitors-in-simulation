"""On-policy title cells (RULE.md here), POST HOC: the check and the reproduction gates of the Qwen3-32B extraction of
Qwen3.5-27B's own runs (config/onpol_qwen3.yaml). q2's check_fresh_qwen.py (aswin/q2-fresh-honest @ f55ccfea) with:
  - layers: the run's own (all 64) for the structural checks; the reproduction compares the layers both sides hold;
  - G1 (unchanged from q2's): each repro-real-honest run against d9's mean-pool dump (2026-10-12_la_mean-pool36_qwen3-32b_l7_v1,
    real-honest; the title cell's real-arm features): positions identical, every row at the dump's 7 layers; a shard passes
    if its rows are bit-exact (H100) or min cosine >= 0.999 (H200, B200);
  - G2 (new): each repro-xread-s1 run (a run-1 attack run's messages at scrub level 1) against the cross-read's own extraction
    (2026-10-13_xread_main_v1 on the cross-read Volume, arm attack): output_end rows matched by readout token index (the staged
    ids are identical, run_onpol_stage.py's G0), at all 64 layers. Different code (transformers 5.19.0, padding to 1,024), so
    not bit-exact: pass if min cosine >= 0.999 over every matched row, with at least 95% of the cross-read's rows matched;
  - G3 (new): every meta's positions equal run_onpol_stage.py's Qwen3-32B position metas (QMETA, Seed's keep flags) in kind,
    step, action, token, kept and span.
Modal CPU; counts only (no text, no activations leave the container). v2 (02:25 PT): the same checks split into 12 parallel
parts (the one-container v1 was stopped by its $0.50 watcher after 36 minutes, before it reported).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_q3_onpolicy_titlecells/check_onpol_qwen.py --run <run> --staged <staged run>
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3onpol_check_kill-2026-11-13"
D9 = {"repro-real-honest": ("2026-10-12_la_mean-pool36_qwen3-32b_l7_v1", "real-honest")}
D9_LAYERS = (16, 24, 32, 40, 48, 56, 63)
XREAD_RUN, XREAD_ARM = "2026-10-13_xread_main_v1", "attack"
QMETA = "2026-10-14_q3onpol_q35own_qmeta_v1"
KINDS = ("output_end", "output_end_mean", "call_end", "call_end_mean")

app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only()
xread = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3xread_acts_kill-2026-11-13", version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, volumes={"/acts": acts, "/xr": xread}, cpu=4, memory=32768, timeout=3600, retries=0)
def check(RUN: str, STAGED: str, part: str) -> dict:
    try:
        return _check(RUN, STAGED, part)
    except Exception as e:  # noqa: BLE001 — type and location only
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"check failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}") from None


def _xread_stem(key: str) -> str:  # run_extract_crossread.py's meta file stems
    import hashlib
    return hashlib.sha1(key.encode()).hexdigest()[:20]


def _check(RUN: str, STAGED: str, part: str) -> dict:
    """One part of the check (the parts run in parallel, v2): "layers:a:b" (each chunk's layer files LAYERS[a:b] against its
    index, and the first and last layers' non-finite recount), "index" (metas, indexes, kept readouts), "g1", "g2", "g3"."""
    from collections import Counter, defaultdict

    import numpy as np
    root, run = Path("/acts"), Path("/acts") / RUN
    LAYERS = json.loads((run / "chunks" / sorted(p.name for p in (run / "chunks").glob("shard*_chunk*.json"))[0]).read_text())["layers"]
    if part.startswith("layers:"):
        a_, b_ = map(int, part.split(":")[1:])
        bad, recount = 0, 0
        for p in sorted((run / "chunks").glob("shard*_chunk*.json")):
            c = json.loads(p.read_text())
            for l in LAYERS[a_:b_]:
                x = np.load(run / "by_layer" / f"L{l}" / f"{p.stem}.npy", mmap_mode="r")
                bad += x.shape[0] != c["rows"]
                if l in (LAYERS[0], LAYERS[-1]):
                    recount += int((~np.isfinite(np.asarray(x))).sum())
        return {"part": part, "layer files whose rows differ from the index": bad, "non-finite recount": recount}
    staged = {(r["arm"], r["key"]) for r in json.loads((root / STAGED / "stage_record.json").read_text())["runs"]}
    metas = {}
    for d in sorted((run / "meta").iterdir()):
        for p in d.glob("*.json"):
            m = json.loads(p.read_text())
            metas[(m["arm"], m["key"])] = m
    prob = Counter()
    prob["staged runs without a meta"] = len(staged - set(metas))
    prob["metas not staged"] = len(set(metas) - staged)
    index = {}
    for p in sorted((run / "chunks").glob("shard*_chunk*.json")):
        c = json.loads(p.read_text())
        prob["chunk indexes without .done"] += not (run / "chunks" / f"{p.stem}.done").exists()
        prob["chunk indexes whose layers differ from the run's"] += c["layers"] != LAYERS
        index[p.stem] = c
    entries = Counter()
    for name, c in index.items():
        for t in c["transcripts"]:
            m = metas.get((t["arm"], t["key"]))
            if m is not None and m["chunk"] == name:
                entries[(t["arm"], t["key"])] += 1
                prob["meta and index disagree (row_start or n_rows)"] += (m["row_start"], m["n_rows"]) != (t["row_start"], t["n_rows"])
    prob["metas with no index entry pointing back"] = sum(entries[k] == 0 for k in metas)
    prob["metas with more than one index entry"] = sum(entries[k] > 1 for k in metas)
    nonfinite_index = sum(c["nonfinite"] for c in index.values())
    kept = defaultdict(Counter)
    for (a, _), m in metas.items():
        for p in m["positions"]:
            if p["kind"] in KINDS and p.get("row") is not None:
                kept[a][p["kind"]] += 1
    if part == "index":
        return {"part": part, "run": RUN, "layers": len(LAYERS), "metas by arm": dict(Counter(a for a, _ in metas)), "problems": dict(prob),
                "non-finite (chunk indexes)": nonfinite_index, "kept readouts by arm and kind": {a: dict(v) for a, v in sorted(kept.items())}}

    # G3: the metas' positions against the staged Qwen3-32B position metas (Seed's keep flags)
    sigq = lambda ps: [(p["kind"], p.get("step"), p.get("action"), p.get("token"), p.get("kept"), p.get("span")) for p in ps if not p["kind"].endswith(("_mean", "_max")) and "_ema" not in p["kind"]]  # noqa: E731
    g3 = Counter()
    for (a, k), m in (metas.items() if part == "g3" else []):
        if a == "repro-xread-s1":
            continue
        q = root / QMETA / "meta" / a / f"{m['filename']}.json"
        if not q.exists():
            g3["qmeta missing"] += 1
            continue
        g3["identical" if sigq(m["positions"]) == sigq(json.loads(q.read_text())["positions"]) else "differ"] += 1
    if part == "g3":
        return {"part": part, "G3 positions vs the staged position metas": dict(g3),
                "pass": g3.get("differ", 0) == 0 and g3.get("qmeta missing", 0) == 0 and g3.get("identical", 0) > 0}

    gpu = {int(p.stem.split("_")[1]): json.loads(p.read_text()).get("gpu") for p in (run / "stats").glob("shard_*.json")}

    def rows_of(run_dir, m, layers):
        return {l: np.load(run_dir / "by_layer" / f"L{l}" / f"{m['chunk']}.npy", mmap_mode="r") for l in layers}

    # G1: repro-real-honest vs d9's mean-pool dump (q2's gate, unchanged)
    shard_res, rep = {}, defaultdict(lambda: {"shards": 0, "shards passing": 0, "rows": 0, "bit-exact rows": 0, "max abs diff": 0.0, "min cosine": 1.0})
    for (a, k), m in (sorted(metas.items()) if part == "g1" else []):
        if a not in D9:
            continue
        dump, darm = D9[a]
        d = json.loads((root / dump / "meta" / darm / f"{m['filename']}.json").read_text())
        sig = lambda mm: [(p["kind"], p.get("step"), p.get("action"), p.get("token"), p.get("kept"), p.get("span"), p.get("row") is not None) for p in mm["positions"]]  # noqa: E731
        shard = int(m["chunk"].split("_")[0].removeprefix("shard"))
        g = gpu.get(shard) or "unknown"
        r = {"positions identical": sig(m) == sig(d), "rows": 0, "bit-exact rows": 0, "max abs diff": 0.0, "min cosine": 1.0}
        ours, theirs = rows_of(run, m, D9_LAYERS), rows_of(root / dump, d, D9_LAYERS)
        for p, q in zip(m["positions"], d["positions"]):
            if p.get("row") is None or q.get("row") is None:
                continue
            for l in D9_LAYERS:
                u = np.asarray(ours[l][m["row_start"] + p["row"]], dtype=np.float32)
                v = np.asarray(theirs[l][d["row_start"] + q["row"]], dtype=np.float32)
                cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
                r["rows"] += 1
                r["bit-exact rows"] += bool(np.array_equal(u, v))
                r["max abs diff"] = max(r["max abs diff"], float(np.abs(u - v).max()))
                r["min cosine"] = min(r["min cosine"], cos)
        exact = r["rows"] > 0 and r["bit-exact rows"] == r["rows"]
        r["pass"] = bool(r["positions identical"] and r["rows"] > 0 and (exact if "H100" in g else r["min cosine"] >= 0.999))
        shard_res[shard] = {"gpu": g, **r}
        t = rep[g]
        t["shards"] += 1; t["shards passing"] += r["pass"]; t["rows"] += r["rows"]; t["bit-exact rows"] += r["bit-exact rows"]
        t["max abs diff"] = max(t["max abs diff"], r["max abs diff"]); t["min cosine"] = min(t["min cosine"], r["min cosine"])

    if part == "g1":
        return {"part": part, "G1 reproduction by GPU type": dict(rep), "G1 reproduction by shard": {str(s_): v for s_, v in sorted(shard_res.items())},
                "G1 failing shards (their rows are not used)": sorted(s_ for s_, v in shard_res.items() if not v["pass"]),
                "pass": bool(shard_res) and all(v["pass"] for v in shard_res.values())}

    # G2: repro-xread-s1 vs the cross-read's rows, matched by output_end readout token
    xr = Path("/xr") / XREAD_RUN
    g2, cos_l, ad_l = {}, defaultdict(list), defaultdict(list)  # per-layer cosines and max abs differences (the v3 diagnostic)
    off_l = defaultdict(list)  # v3: (new layer l, cross-read layer l + o) cosines, o in -1, 0, +1 (an index or hook shift check)
    for (a, k), m in (sorted(metas.items()) if part == "g2" else []):
        if a != "repro-xread-s1":
            continue
        stg = next(json.loads(l)["staged_epoch_key"] for l in open(root / STAGED / "inputs" / next(
            f for f, v in json.loads((root / STAGED / "stage_record.json").read_text())["shards_files"].items() if v["arm"] == a).split("/")[-1])
            if json.loads(l)["key"] == k)
        xm = json.loads((xr / "meta" / XREAD_ARM / f"{_xread_stem(stg)}.json").read_text())
        xrows = {c["readout"]: c["row"] for c in xm["calls"] if c["kind"] == "output_end" and c.get("row") is not None}
        orow = {p["token"]: p["row"] for p in m["positions"] if p["kind"] == "output_end" and p.get("row") is not None}
        both = sorted(set(xrows) & set(orow))
        layers = [l for l in LAYERS if (xr / "by_layer" / f"L{l}" / f"{xm['chunk']}.npy").exists()]
        ours, theirs = rows_of(run, m, layers), rows_of(xr, xm, layers)
        r = {"cross-read output_end rows": len(xrows), "matched": len(both), "layers": len(layers), "rows x layers": 0,
             "bit-exact": 0, "max abs diff": 0.0, "min cosine": 1.0}
        for t_ in both:
            for l in layers:
                u = np.asarray(ours[l][m["row_start"] + orow[t_]], dtype=np.float32)
                v = np.asarray(theirs[l][xm["row_start"] + xrows[t_]], dtype=np.float32)
                r["rows x layers"] += 1
                r["bit-exact"] += bool(np.array_equal(u, v))
                r["max abs diff"] = max(r["max abs diff"], float(np.abs(u - v).max()))
                cs_ = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
                r["min cosine"] = min(r["min cosine"], cs_)
                cos_l[l].append(cs_); ad_l[l].append(float(np.abs(u - v).max()))
                if part == "g2":
                    for o in (-1, 1):
                        if (l + o) in theirs:
                            w = np.asarray(theirs[l + o][xm["row_start"] + xrows[t_]], dtype=np.float32)
                            off_l[(l, o)].append(float(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-12)))
                    off_l[(l, 0)].append(cs_)
        r["pass"] = bool(len(both) >= 0.95 * max(len(xrows), 1) and r["rows x layers"] > 0 and r["min cosine"] >= 0.999)
        g2[k] = r
    per_layer = {str(l): {"rows": len(v), "min cosine": float(np.min(v)), "1st percentile cosine": float(np.percentile(v, 1)),
                          "median cosine": float(np.median(v)), "share >= 0.999": float(np.mean(np.asarray(v) >= 0.999)),
                          "median max abs diff": float(np.median(ad_l[l]))} for l, v in sorted(cos_l.items())}
    offsets = {}
    for l in sorted(cos_l):
        row = {str(o): {"median cosine": float(np.median(off_l[(l, o)])), "min cosine": float(np.min(off_l[(l, o)]))}
               for o in (-1, 0, 1) if off_l.get((l, o))}
        row["best offset (by median)"] = int(max((o for o in (-1, 0, 1) if off_l.get((l, o))), key=lambda o: np.median(off_l[(l, o)])))
        offsets[str(l)] = row
    return {"part": part, "G2 cross-read rows (scrub 1)": g2, "G2 per layer (diagnostic)": per_layer, "G2 layer offsets (diagnostic)": offsets,
            "offset 0 best at every layer": all(v["best offset (by median)"] == 0 for v in offsets.values()), "pass": bool(g2) and all(v["pass"] for v in g2.values())}


@app.local_entrypoint()
def main(run: str, staged: str, only: str = ""):
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__))], cwd=HERE).returncode:
        raise SystemExit("commit this script before running")
    parts = ["index", "g1", "g2", "g3"] + [f"layers:{i}:{i + 8}" for i in range(0, 64, 8)]
    if only:  # one part alone (v3: --only g2, the per-layer diagnostic); its record is written as is, with no overall gate
        r = check.remote(run, staged, only)
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        with open(p := HERE / "results" / f"check_{only}_{run}_{stamp}.json", "x") as fh:
            fh.write(json.dumps({"POST HOC": True, "git": head, "run": run} | r, indent=1) + "\n")
        print(p)
        print(json.dumps(r.get("G2 per layer (diagnostic)", r), indent=1)[:6000])
        return
    res = {r["part"]: r for r in check.starmap([(run, staged, p) for p in parts])}
    lay = [res[p] for p in parts if p.startswith("layers:")]
    idx = res["index"]
    struct_ok = (not any(idx["problems"].values()) and idx["non-finite (chunk indexes)"] == 0
                 and sum(x["layer files whose rows differ from the index"] for x in lay) == 0 and sum(x["non-finite recount"] for x in lay) == 0)
    gates = {"structure": struct_ok, "G1": res["g1"]["pass"], "G2": res["g2"]["pass"], "G3": res["g3"]["pass"]}
    out = {"run": run, "parts": res, "layer files whose rows differ from the index": sum(x["layer files whose rows differ from the index"] for x in lay),
           "non-finite recount (first and last layer)": sum(x["non-finite recount"] for x in lay), "gates": gates,
           "gate": "pass" if all(gates.values()) else "FAIL"}
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(p := HERE / "results" / f"check_{run}_{stamp}.json", "x") as fh:
        fh.write(json.dumps({"POST HOC": True, "plan": "RULE.md (on-policy title cells)", "git": head} | out, indent=1) + "\n")
    print(p)
    print(json.dumps({"metas by arm": idx["metas by arm"], "problems": idx["problems"], "G1": res["g1"]["G1 reproduction by GPU type"],
                      "G2": {k: {kk: v[kk] for kk in ("matched", "cross-read output_end rows", "min cosine", "max abs diff", "pass")} for k, v in res["g2"]["G2 cross-read rows (scrub 1)"].items()},
                      "G3": res["g3"]["G3 positions vs the staged position metas"], "gates": gates, "gate": out["gate"]}, indent=1))
