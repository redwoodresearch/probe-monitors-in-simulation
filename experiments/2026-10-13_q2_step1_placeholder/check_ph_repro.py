"""Step 1 (PLAN.md here): the reproduction part of check_ph.py again, with diagnostics (Modal CPU; counts and keys only).
check_ph.py (2026-10-13_174034) flagged one sampled run whose step-0 positions differ from the reference and sampled no
run from the one shard on an H100 NVL. This re-runs check_ph.py's same seeded sample (identical code path for the sample),
records each flagged run's arm, key and reason (no step-0 call in either; or the position sets differ, with the
differing (kind, step, action) tuples), and adds SAMPLE runs from every GPU type's shards (seed 20261014).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run experiments/2026-10-13_q2_step1_placeholder/check_ph_repro.py
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-step1ph_check_kill-2026-11-13"
RUN, STAGED = "2026-10-13_step1ph_qwen3-32b_l48-63_v1", "2026-10-13_q2_step1_placeholder_inputs_v1"
REF = {"sim-honest": "2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1"}
REF_DEFAULT = "2026-10-06_la_segment-end_qwen3-32b_alllayers_v1"
LAYERS, SAMPLE, SEED, SEED_GPU = (48, 63), 15, 20261013, 20261014
PRE = ("pre_call", "call_end", "call_marker")

app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


def filename(key: str) -> str:
    import hashlib
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/acts": acts}, cpu=2, memory=8192, timeout=1800, retries=0)
def repro() -> dict:
    import numpy as np
    root = Path("/acts") / RUN
    rec = json.loads((Path("/acts") / STAGED / "stage_record.json").read_text())
    arms = sorted({r["arm"] for r in rec["runs"]})
    gpu = {int(p.stem.split("_")[1]): json.loads(p.read_text()).get("gpu") for p in (root / "stats").glob("shard_*.json")}
    metas = {}
    for p in (root / "meta").glob("*/*.json"):
        m = json.loads(p.read_text())
        metas[(m["arm"], m["key"])] = m
    rng = np.random.default_rng(SEED)
    picks = []
    for arm in arms:  # check_ph.py's sample, same order and generator
        keys = sorted(k for a, k in metas if a == arm)
        picks += [("check_ph sample", arm, k) for k in rng.choice(keys, min(SAMPLE, len(keys)), replace=False)]
    g2 = np.random.default_rng(SEED_GPU)
    for gt in sorted(set(gpu.values())):
        cand = sorted((a, k) for (a, k), m in metas.items() if gpu.get(int(m["chunk"][5:8])) == gt)
        picks += [(f"per GPU type: {gt}", *cand[i]) for i in g2.choice(len(cand), min(SAMPLE, len(cand)), replace=False)]
    out, flagged = {}, []
    for src_label, arm, k in picks:
        m = metas[(arm, k)]
        src = arm.removesuffix("-ph")
        ref_root = Path("/acts") / REF.get(src, REF_DEFAULT)
        r = json.loads((ref_root / "meta" / src / f"{filename(k)}.json").read_text())
        pre = {(q["kind"], q["step"], q["action"]): q["row"] for q in m["positions"] if q["kept"] and q["step"] == 0 and q["kind"] in PRE}
        rr = {(q["kind"], q["step"], q["action"]): q["row"] for q in r["positions"] if q["kept"] and q["step"] == 0 and q["kind"] in PRE}
        allpos = sorted({(q["kind"], q["step"], q["action"], q["kept"]) for q in m["positions"] if q["step"] == 0})
        if set(pre) != set(rr) or not pre:
            flagged.append({"sample": src_label, "arm": arm, "key": k, "reason": "no kept step-0 call position in either" if not pre and not rr else "position sets differ",
                            "only here": sorted(map(list, set(pre) - set(rr))), "only in the reference": sorted(map(list, set(rr) - set(pre))),
                            "step-0 positions here (kind, step, action, kept)": [list(x) for x in allpos],
                            "step-0 positions in the reference (kind, step, action, kept)": [list(x) for x in sorted({(q["kind"], q["step"], q["action"], q["kept"]) for q in r["positions"] if q["step"] == 0})]})
            continue
        g = gpu.get(int(m["chunk"][5:8]), "unknown")
        for l in LAYERS:
            A = np.load(root / "by_layer" / f"L{l}" / f"{m['chunk']}.npy", mmap_mode="r")
            Bm = np.load(ref_root / "by_layer" / f"L{l}" / f"{r['chunk']}.npy", mmap_mode="r")
            for pk in sorted(pre):
                x = np.asarray(A[m["row_start"] + pre[pk]], np.float32); y = np.asarray(Bm[r["row_start"] + rr[pk]], np.float32)
                d = out.setdefault(f"{src_label} | {g} L{l}", {"runs": set(), "rows": 0, "bit-exact": 0, "max abs diff": 0.0, "min cosine": 1.0})
                d["runs"].add((arm, k)); d["rows"] += 1; d["bit-exact"] += bool((x == y).all())
                d["max abs diff"] = max(d["max abs diff"], float(np.abs(x - y).max()))
                d["min cosine"] = min(d["min cosine"], float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))
    for d in out.values():
        d["runs"] = len(d["runs"])
    return {"run": RUN, "GPU types": sorted(set(gpu.values())), "reproduction": out, "flagged runs": flagged}


@app.local_entrypoint()
def main():
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", __file__], cwd=HERE).returncode:
        raise SystemExit("commit check_ph_repro.py before running")
    out = repro.remote()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"check_repro_{RUN}_{stamp}.json"
    p.write_text(json.dumps({"git": head} | out, indent=1) + "\n")
    print(json.dumps(out, indent=1)[:6000]); print(p)
