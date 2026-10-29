"""F12 (RULE.md here): check_f11.py (aswin/q2-f10-tabs @ 51846989) with only the app, run, staged-run names and result name changed.
It was: F11 (PLAN_F11.md): check_fxavg.py (aswin/q2-f10-tabs @ 9c23081d) for the F11 Qwen3-32B run 2026-10-13_q2_f11_qwen3-32b_l7_v1
(the staged set from stage_record.json's "planned shard" keys). It was: the check and the reproduction gate of the Qwen3-32B run
2026-10-13_q2_fxavg_qwen3-32b_l7_v1. Modal CPU; counts only (no text, no activations leave the container).
  - every staged run extracted exactly once (metas = the stage record's (arm, key) set; one chunk-index entry per meta, same
    row_start and n_rows); every chunk index has its .done marker; each layer file's rows equal its index's;
  - non-finite: the chunk indexes' counts, and a full recount of the first and last layer;
  - kept readouts per arm and kind (output_end, call_end, and their means);
  - reproduction gate: each shard's reproduction run against d9's dump (positions identical; every row at every layer):
    bit-exact rows, max |difference|, min cosine, by the shard's GPU. A shard passes if its rows are bit-exact (H100) or
    min cosine >= 0.999 (H200, B200); a failing shard's rows are not used.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach experiments/2026-10-13_q2_fixfull/check_fxavg.py
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-ffm1_check_kill-2026-11-13"
RUN, STAGED = "2026-10-14_q2_f12_qwen3-32b_l7_v1", "2026-10-14_q2_f12q_inputs_v1"
D9 = {"repro-sim-attack": ("2026-10-12_la_mean-pool36_qwen3-32b_l7_v1", "sim-attack"),
      "repro-sim-honest": ("2026-10-13_q3_mean-pool36-honestsim_qwen3-32b_l7_v1", "sim-honest")}
LAYERS = (16, 24, 32, 40, 48, 56, 63)
KINDS = ("output_end", "output_end_mean", "call_end", "call_end_mean")

app = modal.App(APP)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, volumes={"/acts": acts}, cpu=8, memory=32768, timeout=3600, retries=0)
def check() -> dict:
    try:
        return _check()
    except Exception as e:  # noqa: BLE001 — type and location only
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"check failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}") from None


def _check() -> dict:
    from collections import Counter, defaultdict

    import numpy as np
    root, run = Path("/acts"), Path("/acts") / RUN
    staged = {tuple(k.split("|", 1)) for k in json.loads((root / STAGED / "stage_record.json").read_text())["planned shard"]}
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
    recount = 0
    for name, c in index.items():
        for l in LAYERS:
            x = np.load(run / "by_layer" / f"L{l}" / f"{name}.npy", mmap_mode="r")
            prob["layer files whose rows differ from the index"] += x.shape[0] != c["rows"]
            if l in (LAYERS[0], LAYERS[-1]):
                recount += int((~np.isfinite(np.asarray(x))).sum())
    kept = defaultdict(Counter)
    for (a, _), m in metas.items():
        for p in m["positions"]:
            if p["kind"] in KINDS and p.get("row") is not None:
                kept[a][p["kind"]] += 1

    # reproduction gate
    gpu = {int(p.stem.split("_")[1]): json.loads(p.read_text()).get("gpu") for p in (run / "stats").glob("shard_*.json")}

    def rows(run_dir, m):
        return {l: np.load(run_dir / "by_layer" / f"L{l}" / f"{m['chunk']}.npy", mmap_mode="r") for l in LAYERS}
    shard_res, rep = {}, defaultdict(lambda: {"shards": 0, "shards passing": 0, "rows": 0, "bit-exact rows": 0, "max abs diff": 0.0, "min cosine": 1.0})
    for (a, k), m in sorted(metas.items()):
        if a not in D9:
            continue
        dump, darm = D9[a]
        d = json.loads((root / dump / "meta" / darm / f"{m['filename']}.json").read_text())
        sig = lambda mm: [(p["kind"], p.get("step"), p.get("action"), p.get("token"), p.get("kept"), p.get("span"), p.get("row") is not None) for p in mm["positions"]]  # noqa: E731
        shard = int(m["chunk"].split("_")[0].removeprefix("shard"))
        g = gpu.get(shard) or "unknown"
        r = {"positions identical": sig(m) == sig(d), "rows": 0, "bit-exact rows": 0, "max abs diff": 0.0, "min cosine": 1.0}
        ours, theirs = rows(run, m), rows(root / dump, d)
        for p, q in zip(m["positions"], d["positions"]):
            if p.get("row") is None or q.get("row") is None:
                continue
            for l in LAYERS:
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
    return {"run": RUN, "metas by arm": dict(Counter(a for a, _ in metas)), "problems": dict(prob),
            "non-finite (chunk indexes)": nonfinite_index, "non-finite recount (first and last layer)": recount,
            "kept readouts by arm and kind": {a: dict(v) for a, v in sorted(kept.items())},
            "reproduction by GPU type": dict(rep), "reproduction by shard": {str(s): v for s, v in sorted(shard_res.items())},
            "shards without a reproduction run": sorted(set(gpu) - set(shard_res)),
            "failing shards (their rows are not used)": sorted(s for s, v in shard_res.items() if not v["pass"]),
            "gate": "pass" if shard_res and all(v["pass"] for v in shard_res.values()) and len(shard_res) == len(gpu) else "FAIL"}


@app.local_entrypoint()
def main():
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__))], cwd=HERE).returncode:
        raise SystemExit("commit this script before running")
    out = check.remote()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(p := HERE / "results" / f"check_f12_{stamp}.json", "x") as fh:
        fh.write(json.dumps({"POST HOC": True, "plan": "RULE.md (ffm iteration 1)", "git": head} | out, indent=1) + "\n")
    print(p)
    print(json.dumps({k: out[k] for k in ("metas by arm", "problems", "non-finite (chunk indexes)", "non-finite recount (first and last layer)",
                                          "reproduction by GPU type", "failing shards (their rows are not used)", "gate")}, indent=1))
