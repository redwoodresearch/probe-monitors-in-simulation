"""Step 1 (PLAN.md here): the check of the placeholder extraction (Modal CPU, reads the Volume; writes counts only).
  - every staged run extracted exactly once (the metas are exactly the stage record's (arm, key) set, each with exactly one
    matching chunk-index entry), every chunk done, chunk index and meta agree (rows cumulative from 0, row_start and
    n_rows), each layer file's row count, non-finite values per layer;
  - every kept output_end span has the same token length and the same last token (the placeholder renders identically);
  - coverage: kept call_end readouts per arm, and how many sit on a fixed-slot token (q1_eval.FIXED_CALL_TOKENS, level 0);
  - reproduction (a sample, not a full check): for SAMPLE = 15 runs per arm (seed 20261013), the rows at positions before the first tool reply (step 0's
    pre_call, call_end, call_marker) against #36's all-layers run (sim-honest: the v4 all-layers run) at layers 48 and 63,
    per GPU type: bit-exact count, max abs diff, min cosine. Gate: min cosine >= 0.999 (prefix rows can differ in the last
    bits when the sequence length changes the matmul shapes).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run experiments/2026-10-13_q2_step1_placeholder/check_ph.py
"""
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-step1ph_check_kill-2026-11-13"
RUN, STAGED = "2026-10-13_step1ph_qwen3-32b_l48-63_v1", "2026-10-13_q2_step1_placeholder_inputs_v1"
REF = {"sim-honest": "2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1"}
REF_DEFAULT = "2026-10-06_la_segment-end_qwen3-32b_alllayers_v1"
LAYERS, SAMPLE, SEED = (48, 63), 15, 20261013
FIXED = ("0", "n")  # q1_eval.fixed_call_tokens at level 0 (no scrub): ">" only from scrub level 2

app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/acts": acts}, cpu=8, memory=32768, timeout=3600, retries=0)
def check() -> dict:
    import numpy as np
    root, P = Path("/acts") / RUN, Counter()
    rec = json.loads((Path("/acts") / STAGED / "stage_record.json").read_text())
    expect = Counter(r["arm"] for r in rec["runs"])
    expect_ids = {(r["arm"], r["key"]) for r in rec["runs"]}
    gpu = {int(p.stem.split("_")[1]): json.loads(p.read_text()).get("gpu") for p in (root / "stats").glob("shard_*.json")}
    metas = {}
    for p in (root / "meta").glob("*/*.json"):
        m = json.loads(p.read_text())
        if (m["arm"], m["key"]) in metas:
            P["meta twice"] += 1
        metas[(m["arm"], m["key"])] = m
    got = Counter(a for a, _ in metas)
    P["metas not exactly the staged (arm, key) set"] += set(metas) != expect_ids
    nonfinite, rows_total, entries = {l: 0 for l in LAYERS}, 0, Counter()
    for idx in sorted((root / "chunks").glob("shard*_chunk*.json")):
        c = json.loads(idx.read_text())
        P["chunk without its done marker"] += not (root / "chunks" / f"{idx.stem}.done").exists()
        off = 0
        for t in c["transcripts"]:
            entries[(t["arm"], t["key"], idx.stem)] += 1
            m = metas.get((t["arm"], t["key"]))
            P["index entry without its meta, or meta in another chunk"] += m is None or m["chunk"] != idx.stem
            if m is not None and m["chunk"] == idx.stem:
                P["row_start or n_rows differ"] += (m["row_start"], m["n_rows"]) != (t["row_start"], t["n_rows"])
            P["offsets not cumulative from 0"] += t["row_start"] != off
            off += t["n_rows"]
        rows_total += off
        for l in LAYERS:
            X = np.load(root / "by_layer" / f"L{l}" / f"{idx.stem}.npy", mmap_mode="r")
            P["layer file with the wrong row count"] += X.shape[0] != off
            nonfinite[l] += int((~np.isfinite(np.asarray(X, np.float32))).sum())
    P["meta without exactly one matching chunk-index entry"] += sum(entries[(a, k, m["chunk"])] != 1 for (a, k), m in metas.items())
    spans, last_tok, ce = Counter(), Counter(), Counter()
    for (a, k), m in metas.items():
        P["meta rows != kept positions"] += m["n_rows"] != sum(q["kept"] for q in m["positions"])
        for q in m["positions"]:
            if q["kind"] == "output_end" and q["kept"]:
                spans[q["span"][1] - q["span"][0]] += 1
                last_tok[q["token_text"]] += 1
            if q["kind"] == "call_end" and q["kept"]:
                ce[f"{a}|kept"] += 1
                ce[f"{a}|on a fixed-slot token"] += q["token_text"] in FIXED
    rng = np.random.default_rng(SEED)
    repro = {}
    for arm in sorted(expect):
        src = arm.removesuffix("-ph")
        ref_root = Path("/acts") / REF.get(src, REF_DEFAULT)
        keys = sorted(k for a, k in metas if a == arm)
        for k in rng.choice(keys, min(SAMPLE, len(keys)), replace=False):
            m = metas[(arm, k)]
            rp = ref_root / "meta" / src / f"{filename(k)}.json"
            if not rp.exists():
                P["reproduction: reference meta missing"] += 1
                continue
            r = json.loads(rp.read_text())
            pre = {(q["kind"], q["step"], q["action"]): q["row"] for q in m["positions"] if q["kept"] and q["step"] == 0 and q["kind"] in ("pre_call", "call_end", "call_marker")}
            rr = {(q["kind"], q["step"], q["action"]): q["row"] for q in r["positions"] if q["kept"] and q["step"] == 0 and q["kind"] in ("pre_call", "call_end", "call_marker")}
            if set(pre) != set(rr) or not pre:
                P["reproduction: step-0 positions differ from the reference"] += 1
                continue
            g = gpu.get(int(m["chunk"][5:8]), "unknown")
            for l in LAYERS:
                A = np.load(root / "by_layer" / f"L{l}" / f"{m['chunk']}.npy", mmap_mode="r")
                Bm = np.load(ref_root / "by_layer" / f"L{l}" / f"{r['chunk']}.npy", mmap_mode="r")
                for pk in sorted(pre):
                    x = np.asarray(A[m["row_start"] + pre[pk]], np.float32); y = np.asarray(Bm[r["row_start"] + rr[pk]], np.float32)
                    d = repro.setdefault(f"{g} L{l}", {"rows": 0, "bit-exact": 0, "max abs diff": 0.0, "min cosine": 1.0})
                    d["rows"] += 1; d["bit-exact"] += bool((x == y).all())
                    d["max abs diff"] = max(d["max abs diff"], float(np.abs(x - y).max()))
                    d["min cosine"] = min(d["min cosine"], float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))
    out = {"run": RUN, "staged": STAGED, "metas by arm": dict(sorted(got.items())), "expected by arm": dict(sorted(expect.items())),
           "rows": rows_total, "non-finite by layer": nonfinite, "problems": dict(P), "output_end span lengths": dict(spans),
           "output_end last tokens": dict(last_tok), "call_end coverage": dict(sorted(ce.items())), "reproduction (step-0 rows vs the all-layers run)": repro,
           "GPU by shard": {str(k): v for k, v in sorted(gpu.items())}}
    out["clean"] = (got == expect and not any(P.values()) and not any(nonfinite.values()) and len(spans) == 1 and len(last_tok) == 1
                    and bool(repro) and all(v["min cosine"] >= 0.999 for v in repro.values()))
    return out


@app.local_entrypoint()
def main():
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", __file__], cwd=HERE).returncode:
        raise SystemExit("commit check_ph.py before running")
    out = check.remote()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"check_{RUN}_{stamp}.json"
    p.write_text(json.dumps({"git": head} | out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "GPU by shard"}, indent=1)); print(p)
