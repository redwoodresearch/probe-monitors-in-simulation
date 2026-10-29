"""Retrain Apollo's deception probe on Qwen3-32B and test it on their Roleplaying set (Modal, one H100).

Apollo (github.com/ApolloResearch/deception-detection, config `repe.yaml`) fit a logistic-regression probe on
Llama-3.3-70B, layer 22 of 80. Here the same data and fitting recipe run on Qwen3-32B: train on their
instructed honest/deceptive pairs, then score their Roleplaying responses (Llama-written, Apollo-graded).
The primary layer is a middle layer (32 of 64); a few other layers are fit alongside for reference.

  uv run modal run experiments/2026-10-05_apollo_deception_probe/run_apollo_probe.py
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APOLLO_REPO = Path.home() / "deception-detection"
APP = "probe-shift_aswin-surya_sub-apollo_deception-probe"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
OUTPUT = "probe-shift_aswin-surya_sub-apollo_deception-probe"
GPU = "H100!"
TIMEOUT_S = 3600
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "scikit-learn==1.7.2"]

app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE, create_if_missing=True)
out_vol = modal.Volume.from_name(OUTPUT, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*PIP)
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("probe_shift")
)


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache, "/out": out_vol}, timeout=TIMEOUT_S, memory=65536)
def fit_and_score(cfg: dict, train: list[dict], evals: list[dict]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift import apollo

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["dtype"]), device_map="cuda",
                                                attn_implementation="sdpa").model.eval()
    return apollo.fit_and_score(cfg, train, evals, tok, base, out_dir=Path("/out") / cfg["run"]) | {"device": torch.cuda.get_device_name()}


@app.local_entrypoint()
def main():
    import yaml

    from probe_shift.apollo import instructed_pairs, roleplaying

    cfg = yaml.safe_load((HERE / "config" / "apollo.yaml").read_text())
    commit = subprocess.run(["git", "-C", str(APOLLO_REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    train, evals = instructed_pairs(APOLLO_REPO), roleplaying(APOLLO_REPO)
    print(f"Apollo repo {commit[:8]}: {len(train)} instructed-pair dialogues, {len(evals)} roleplaying responses")
    result = fit_and_score.remote(cfg, train, evals)
    result["config"] = cfg | {"apollo_repo_commit": commit,
                              "modal": {"app": APP, "gpu": GPU, "timeout_s": TIMEOUT_S, "pip": PIP, "hf_cache": HF_CACHE, "output_volume": OUTPUT}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"apollo_probe_{stamp}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"\n{result['device']}: forward passes {result['forward_seconds']}s, total {result['total_seconds']}s")
    print(f"train dialogues {result['n_train_dialogues']} (no tokens: {result['n_train_dialogues_without_tokens']}), val {result['n_val_dialogues']}; roleplaying {result['n_eval']}")
    for l, r in result["by_layer"].items():
        mark = "  <- primary" if int(l) == cfg["primary_layer"] else ""
        print(f"  layer {l:>2}: roleplaying AUROC {r['roleplaying_auroc']:.3f}   instructed-pairs val AUROC {r['instructed_pairs_val_auroc']:.3f}   converged {r['converged']}{mark}")
    print(f"results -> {path}")
