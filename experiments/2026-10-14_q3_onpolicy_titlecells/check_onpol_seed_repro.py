"""On-policy title cells (RULE.md here), POST HOC: the Seed-OSS-36B reproduction gate (G4). Each repro-real-honest run of the
Seed extraction (run_extract_seed_onpol.py; #36 real honest runs copied from d9's mean-pool dump's inputs) against q2's seedcall
dump (2026-10-13_seedcall_seed-oss-36b_v1, arm real-honest: the Seed title cell's real-arm features): positions identical (kind,
step, action, token, kept, span), and every kept row at the dump's 7 layers. Pass per run: bit-exact rows, or (another GPU type
than the dump's shard) min cosine >= 0.999. Modal CPU; counts only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_q3_onpolicy_titlecells/check_onpol_seed_repro.py --run <seed run>
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3onpol_seedcheck_kill-2026-11-13"
REF, REF_ARM, ARM = "2026-10-13_seedcall_seed-oss-36b_v1", "real-honest", "repro-real-honest"
LAYERS = (16, 24, 32, 40, 48, 56, 63)

app = modal.App(APP)
out_ro = modal.Volume.from_name("probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12", version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, volumes={"/out": out_ro}, cpu=4, memory=16384, timeout=1800, retries=0)
def check(run: str) -> dict:
    try:
        return _check(run)
    except Exception as e:  # noqa: BLE001 — type and location only
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"check failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}") from None


def _check(run: str) -> dict:
    import numpy as np
    ours_dir, ref_dir = Path("/out") / run, Path("/out") / REF
    gpu = {int(p.stem.split("_")[1]): json.loads(p.read_text()).get("gpu") for p in (ours_dir / "stats").glob("shard_*.json")}
    sig = lambda m: [(p["kind"], p.get("step"), p.get("action"), p.get("token"), p.get("kept"), p.get("span")) for p in m["positions"]]  # noqa: E731
    res = {}
    for p in sorted((ours_dir / "meta" / ARM).glob("*.json")):
        m = json.loads(p.read_text())
        d = json.loads((ref_dir / "meta" / REF_ARM / p.name).read_text())
        shard = int(m["chunk"].split("_")[0].removeprefix("shard"))
        r = {"gpu": gpu.get(shard), "positions identical": sig(m) == sig(d), "rows x layers": 0, "bit-exact": 0, "max abs diff": 0.0, "min cosine": 1.0}
        if r["positions identical"]:
            a = {l: np.load(ours_dir / "by_layer" / f"L{l}" / f"{m['chunk']}.npy", mmap_mode="r") for l in LAYERS}
            b = {l: np.load(ref_dir / "by_layer" / f"L{l}" / f"{d['chunk']}.npy", mmap_mode="r") for l in LAYERS}
            for x, y in zip(m["positions"], d["positions"]):
                if x.get("row") is None or y.get("row") is None:
                    continue
                for l in LAYERS:
                    u = np.asarray(a[l][m["row_start"] + x["row"]], np.float32)
                    v = np.asarray(b[l][d["row_start"] + y["row"]], np.float32)
                    r["rows x layers"] += 1
                    r["bit-exact"] += bool(np.array_equal(u, v))
                    r["max abs diff"] = max(r["max abs diff"], float(np.abs(u - v).max()))
                    r["min cosine"] = min(r["min cosine"], float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
        r["pass"] = bool(r["positions identical"] and r["rows x layers"] > 0 and (r["bit-exact"] == r["rows x layers"] or r["min cosine"] >= 0.999))
        res[p.stem] = r
    return {"run": run, "reference": REF, "runs": len(res), "by run": res, "gate": "pass" if res and all(v["pass"] for v in res.values()) else "FAIL"}


@app.local_entrypoint()
def main(run: str):
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__))], cwd=HERE).returncode:
        raise SystemExit("commit this script before running")
    out = check.remote(run)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(p := HERE / "results" / f"check_seed_repro_{run}_{stamp}.json", "x") as fh:
        fh.write(json.dumps({"POST HOC": True, "git": head} | out, indent=1) + "\n")
    print(p)
    print(json.dumps({"runs": out["runs"], "gate": out["gate"]}, indent=1))
