"""Gate for q2's fresh-task confirmation: the saved weights of the averaged reply readout reproduce the stored nested
scores. For Qwen3-32B full + F9 (part avg, stamp 2026-10-13_185308), Seed full without F9 (part seed, stamp
2026-10-13_221350; d9's headline Seed cell) and Seed full + F9 (part seedf11, stamp 2026-10-13_223111), layers 48 and 63, cells (a) and (b): 50 runs per arm, drawn with a seed from a sha256, are
rescored from their kept output_end_mean rows with the saved W/b/mean/std of each of their four nested models (f, v ≠ f),
and compared with the stored nested scores. Modal CPU; no text leaves the Volumes.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run \\
      experiments/2026-10-13_why_steps/check_weights_rescore.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-whysteps_rescore_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-whysteps_work_kill-2026-11-13"
ACTS, SEED_VOL = "probe-shift_aswin-surya_sub-la-activations", "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"
MP36, FXAVG, SEEDCALL = "2026-10-12_la_mean-pool36_qwen3-32b_l7_v1", "2026-10-13_q2_fxavg_qwen3-32b_l7_v1", "2026-10-13_seedcall_seed-oss-36b_v1"
H_, E1, E2, SA, SH = "real-honest", "real-attack-e1", "real-attack-e2", "sim-attack", "sim-honest"
ARMS = (H_, E1, E2, SA, SH)
# world -> (root, scores record, weights stamp, arm -> sources tried in order); the same sources as run_why_steps.py
WORLDS = {
    "ffavg": ("/acts", "why_avg_run_2026-10-13_185308.json", "2026-10-13_185308",
              {H_: [(MP36, H_)], E1: [(MP36, E1)], E2: [(MP36, E2)], SA: [(FXAVG, "sim-attack-fx-fullF9")], SH: [(FXAVG, "sim-honest-fx-fullF9")]}),
    "seed_ff": ("/acts_seed", "why_seed_run_2026-10-13_221350.json", "2026-10-13_221350",  # d9's headline Seed cell (no F9)
                {H_: [(SEEDCALL, H_)], E1: [(SEEDCALL, E1)], E2: [(SEEDCALL, E2)],
                 SA: [(SEEDCALL, "sim-attack-fx-full")], SH: [(SEEDCALL, "sim-honest-fx-full")]}),
    "seed_ffF9": ("/acts_seed", "why_seedf11_run_2026-10-13_223111.json", "2026-10-13_223111",
                  {H_: [(SEEDCALL, H_)], E1: [(SEEDCALL, E1)], E2: [(SEEDCALL, E2)],
                   SA: [(SEEDCALL, "sim-attack-fx-fullF9"), (SEEDCALL, "sim-attack-fx-full")],
                   SH: [(SEEDCALL, "sim-honest-fx-fullF9"), (SEEDCALL, "sim-honest-fx-full")]}),
}
LAYERS, CELLS, PER_ARM, KIND = (48, 63), ("a", "b"), 50, "output_end_mean"

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0,
              volumes={"/acts": modal.Volume.from_name(ACTS, version=2).read_only(),
                       "/acts_seed": modal.Volume.from_name(SEED_VOL, version=2).read_only(),
                       "/work": modal.Volume.from_name(WORK, version=2).read_only()})
def rescore(world: str, sample: list[dict], stored: dict) -> dict:
    root, _, stamp, sources = WORLDS[world]
    root = Path(root)
    keys = {(s["arm"], s["key"]) for s in sample}
    metas = {}
    for srcs in sources.values():
        for ext, arm in srcs:
            for p in (root / ext / "meta" / arm).glob("*.json"):
                m = json.loads(p.read_text())
                metas.setdefault((ext, arm), {})[m["key"]] = m
    out = {}
    for layer in LAYERS:
        w = {c: np.load(Path("/work") / "weights" / stamp / f"{world}_{KIND}_L{layer}_{c}.npz") for c in CELLS}
        diffs = {c: [] for c in CELLS}
        for s in sample:
            src = next(x for x in sources[s["arm"]] if s["key"] in metas.get(x, {}))
            m = metas[src][s["key"]]
            idx = [m["row_start"] + p["row"] for p in m["positions"] if p["kind"] == KIND and p["kept"]]
            X = np.load(root / src[0] / "by_layer" / f"L{layer}" / f"{m['chunk']}.npy", mmap_mode="r")[idx].astype(np.float32)
            f = s["fold"]
            for c in CELLS:
                for v in range(5):
                    if v == f:
                        continue
                    mine = float((((X - w[c]["mean"][f, v]) / w[c]["std"][f, v]) @ w[c]["W"][f, v] + w[c]["b"][f, v]).mean())
                    diffs[c].append(abs(mine - stored[f"L{layer}__{c}"][s["i"]][v]))
        out[f"L{layer}"] = {c: {"max_abs_diff": float(max(d)), "comparisons": len(d)} for c, d in diffs.items()}
    return {"world": world, "runs": len(sample), "by_arm": {a: sum(s["arm"] == a for s in sample) for a in ARMS}, "results": out}


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    res = {"git": git("rev-parse", "HEAD"), "per_arm": PER_ARM, "kind": KIND, "worlds": {}}
    for world, (_, rec_name, _, _) in WORLDS.items():  # reads by key; the arithmetic is remote
        rec = json.loads((HERE / "results" / rec_name).read_text())
        npz = HERE / "results" / rec["scores_file"]
        if hashlib.sha256(npz.read_bytes()).hexdigest() != rec["scores_sha256"]:
            raise SystemExit(f"{npz.name}: sha256 differs from its record")
        Z = np.load(npz)
        arm, key, fold = Z["runs__arm"], Z["runs__key"], Z["runs__fold"].astype(int)
        rng = np.random.default_rng(int(hashlib.sha256(f"rescore|{world}".encode()).hexdigest()[:8], 16))
        pick = sorted(int(i) for a in ARMS for i in rng.choice(np.flatnonzero(arm == a), PER_ARM, replace=False))
        sample = [{"i": n, "arm": str(arm[i]), "key": str(key[i]), "fold": int(fold[i])} for n, i in enumerate(pick)]
        stored = {}  # per sampled run n: the stored nested score of model (its fold f, v) for v = 0..4 (NaN at v == f)
        for L in LAYERS:
            for c in CELLS:
                Zc = Z[f"L{L}__{world}_{KIND}__{c}__nested_real"]
                stored[f"L{L}__{c}"] = [[float(Zc[fold[i], v, i]) for v in range(5)] for i in pick]
        r = rescore.remote(world, sample, stored)
        res["worlds"][world] = r | {"scores_file": rec["scores_file"], "scores_sha256": rec["scores_sha256"]}
        print(json.dumps({world: r["results"]}), flush=True)
    res["max_abs_diff_overall"] = max(x["max_abs_diff"] for w in res["worlds"].values() for L in w["results"].values() for x in L.values())
    res["pass_1e-3"] = res["max_abs_diff_overall"] <= 1e-3
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"check_weights_rescore_{stamp}.json"
    with p.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(f"wrote {p.name}; max |difference| {res['max_abs_diff_overall']:.2e}", flush=True)
