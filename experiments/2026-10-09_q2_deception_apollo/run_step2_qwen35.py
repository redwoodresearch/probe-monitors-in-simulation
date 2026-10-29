"""Step 2 of the deception tests: Apollo's probe retrained on Qwen3.5-27B, the agent this project monitors (Modal, one
H100). Runs only because step 1 reproduced Apollo's shipped scores on these sets (results/step1_llama_parity_*.json).

`probe_shift.apollo_sweep.sweep` fits the probe on Apollo's instructed pairs (held-out 20% of pairs, seed 0) at every
one of the 64 layers with both readouts (raw block output; Apollo's final-norm readout), and scores every labelled
dialogue of Roleplaying, the insider-trading report and double-down sets and sandbagging, plus 2,000 Alpaca controls
(seed 0), each by its mean over the detected tokens. Spans are content only (no template tokens), as in the earlier
Qwen sweeps; the Qwen3.5 template runs with enable_thinking false. Probes are kept on this experiment's Volume; every
per-dialogue score comes back for analyze_step2_qwen35.py (layer and readout chosen on half of Roleplaying, then frozen
for the other sets).

  uv run modal run experiments/2026-10-09_q2_deception_apollo/run_step2_qwen35.py --pilot   # 40 pairs, 12 per set
  uv run modal run experiments/2026-10-09_q2_deception_apollo/run_step2_qwen35.py
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

from probe_shift.models import QWEN35_27B

HERE = Path(__file__).parent
APOLLO_REPO = Path.home() / "deception-detection"
NAME = "probe-shift_aswin-surya_sub-q2-deception-qwen35_kill-2026-11-08"
app = modal.App(NAME)
out_vol = modal.Volume.from_name(NAME, create_if_missing=True)
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "scikit-learn==1.7.2", "threadpoolctl",
                         "flash-linear-attention==0.5.2")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))
CFG = {"model": QWEN35_27B["name"], "revision": QWEN35_27B["revision"], "dtype": "bfloat16", "chat_template": {"enable_thinking": False},
       "reg_coeff": 10, "max_iter": 2000, "val_fraction": 0.2, "split_seed": 0, "fit_workers": 8, "blas_threads_per_fit": 2}


@app.function(image=image, gpu="H100!", volumes={"/hf": hf.read_only(), "/out": out_vol}, cpu=16, memory=98304, timeout=7200, retries=0)
def sweep_remote(cfg: dict, train: list[dict], evals: dict[str, list[dict]]) -> dict:
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift import apollo_sweep

    out_dir = Path("/out/runs") / cfg["run"]
    if out_dir.exists() and any(out_dir.iterdir()):  # checked before any GPU work
        raise FileExistsError(f"{out_dir} already holds a run")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=getattr(torch, cfg["dtype"]), device_map="cuda",
                                                attn_implementation="sdpa").model.eval()
    load_s = time.time() - t0
    print(f"loaded in {load_s:.0f}s: {len(base.layers)} blocks", flush=True)
    result, probes = apollo_sweep.sweep(cfg, base, tok, train, evals, log=lambda m: print(m, flush=True))
    out_dir.mkdir(parents=True)
    for r, p in probes.items():
        np.savez(out_dir / f"probes_{r}.npz", **p)
    (out_dir / "result.json").write_text(json.dumps(result))
    out_vol.commit()
    import resource

    return result | {"load_seconds": round(load_s, 1), "container_seconds": round(time.time() - t0, 1), "device": torch.cuda.get_device_name(),
                     "peak GPU memory (GiB)": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                     "peak RAM (GiB)": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
                     "probes_on_volume": f"{NAME}:/runs/{cfg['run']}"}


def build(pilot: bool) -> tuple[list[dict], dict[str, list[dict]]]:
    import numpy as np

    from probe_shift import apollo

    train = apollo.instructed_pairs(APOLLO_REPO)
    alpaca = apollo.from_rollouts(APOLLO_REPO, "alpaca__plain__llama-70b-3.3.json", "alpaca", None)
    evals = {"roleplaying": apollo.roleplaying(APOLLO_REPO), "insider_trading": apollo.insider_trading(APOLLO_REPO),
             "insider_trading_doubledown": apollo.insider_trading(APOLLO_REPO, double_down=True),
             "sandbagging": apollo.from_rollouts(APOLLO_REPO, "sandbagging_v2__wmdp_mmlu__llama-70b-3.3.json", "sandbagging", "<step4>")}
    evals = {k: [d for d in v if d["label"] is not None] for k, v in evals.items()}  # ambiguous and skipped dialogues are not scored
    evals["alpaca"] = [alpaca[i] for i in sorted(np.random.default_rng(0).choice(len(alpaca), 2000, replace=False).tolist())]
    if pilot:
        train = [d for d in train if d["pair"] < 40]
        evals = {k: v[:12] for k, v in evals.items()}
    return train, evals


@app.local_entrypoint()
def main(pilot: bool = False):
    train, evals = build(pilot)
    git = lambda *a: subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True, check=True).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    cfg = CFG | {"run": f"{'pilot_' if pilot else ''}{stamp}"}
    res = sweep_remote.remote(cfg, train, evals)
    out = {"config": cfg, "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
           "apollo_repo_commit": subprocess.run(["git", "-C", str(APOLLO_REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
           "runner sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "pilot": pilot} | res
    # no AUROC is printed here: the layer and readout are chosen on half of Roleplaying in analyze_step2_qwen35.py
    print(json.dumps({k: res[k] for k in ("load_seconds", "forward_seconds", "fit_seconds", "container_seconds", "peak GPU memory (GiB)", "peak RAM (GiB)")}))
    path = HERE / "results" / f"step2_qwen35_{cfg['run']}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out) + "\n")
    print(path)
