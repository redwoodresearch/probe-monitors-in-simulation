"""RULE.md: fold the frozen Apollo D2 probe (block_output layer 31; per-token logistic, no intercept) into the shared scorer's
form, decision = x · r + c with r = coef / scale and c = -sum(mean * coef / scale), from the stored weights; write it as
safetensors and a record the scorer reads (directions.apollo_d2 = {layer, passes, file}). Modal CPU; nothing is fit.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-14_p2_apollo/run_fold_apollo.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-p2probes_apollofold_kill-2026-11-13"
WEIGHTS = "/runs/2026-10-09_191400/probes_block_output.npz"
LAYER = 31
GATE = {"chosen on": "Roleplaying dev half, seed 0 (analyze_step2_qwen35.py lines 1-10 @ main)", "Roleplaying dev AUROC": 0.949,
        "held out (README L404 @ main)": {"insider-trading report": 0.995, "double-down": 0.925, "Roleplaying test half": 0.875}}

app = modal.App(APP)
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q2-deception-qwen35_kill-2026-11-08").read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0")


@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=600, retries=0, volumes={"/q2": vol})
def fold() -> dict:
    import numpy as np
    import safetensors.numpy as stn

    raw = (Path("/q2") / WEIGHTS.lstrip("/")).read_bytes()
    import io

    z = np.load(io.BytesIO(raw))
    if set(z.files) != {"coef", "scaler_mean", "scaler_scale"} or any(z[k].shape != (64, 5120) for k in z.files):
        raise ValueError(f"unexpected keys or shapes: {[(k, z[k].shape) for k in z.files]}")
    coef, mean, scale = (z[k][LAYER].astype(np.float64) for k in ("coef", "scaler_mean", "scaler_scale"))
    if not (np.isfinite(coef).all() and np.isfinite(mean).all() and (scale > 0).all()):
        raise ValueError("non-finite coef/mean or non-positive scale")
    r, c = coef / scale, -float((mean * coef / scale).sum())
    x = np.random.default_rng(0).standard_normal((4, 5120)) * 10 + mean  # fold check against the original decision on 4 random rows
    if not np.allclose(((x - mean) / scale) @ coef, x @ r + c, rtol=1e-9, atol=1e-6):
        raise ValueError("fold check failed")
    f = stn.save({"direction": r.astype(np.float32), "bias": np.array([c], np.float32), "coef": coef.astype(np.float32),
                  "mean": mean.astype(np.float32), "scale": scale.astype(np.float32)},
                 metadata={"layer": str(LAYER), "readout": "block_output", "source": WEIGHTS})
    return {"weights_sha256": hashlib.sha256(raw).hexdigest(), "coef_norm": float(np.linalg.norm(coef)), "bias": c, "file": f}


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if git("status", "--porcelain", "--", ".") or git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted changes")
    head = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit("HEAD is not pushed")
    r = fold.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    f = res / f"apollo_d2_L{LAYER}_{stamp}.safetensors"
    f.open("xb").write(r["file"])
    rec = {"label": "PRELIMINARY; RULE.md in this folder", "git": head, "app": APP, "stamp (PT)": stamp, "weights": WEIGHTS,
           "weights_sha256": r["weights_sha256"], "coef_norm": r["coef_norm"], "bias": r["bias"],
           "directions": {"apollo_d2": {"position": "block_output (per-token probe; mean rows exact)", "layer": LAYER, "passes": True, "gate_scope": "source-domain (agent-format control outstanding)", "gate": GATE,
                                        "file": {"path": f.name, "sha256": hashlib.sha256(r["file"]).hexdigest()}}}}
    dest = res / f"fold_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "directions"}, indent=1))
    print(dest.name)
