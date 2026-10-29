"""Llama check of the Apollo port: does the port reproduce Apollo's probe on the model Apollo used?

(1) `original`: Apollo's code, unchanged (github.com/ApolloResearch/deception-detection), run on
    Llama-3.3-70B-Instruct with their `repe.yaml`. Expect Roleplaying AUROC about 0.964.
(2) `port`: `probe_shift.apollo` on the same model at layer 22 with the same settings. Also scores the port's
    activations with Apollo's shipped probe (`example_results/instructed_pairs/detector.pt`) and compares the
    per-response scores with their shipped `scores.json`, which checks the activations and token spans directly.

Weights are cached on a Modal Volume at Apollo's model directory (/data/huggingface). The tokenizer comes from
meta-llama/Meta-Llama-3.1-70B-Instruct, as in Apollo's code; its template dates every prompt 26 Jul 2024.

  uv run modal run experiments/2026-10-06_apollo_llama_check/run_apollo_llama.py::download   # access check + weights
  uv run modal run experiments/2026-10-06_apollo_llama_check/run_apollo_llama.py::original
  uv run modal run experiments/2026-10-06_apollo_llama_check/run_apollo_llama.py::port
"""

import json
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APOLLO_REPO = Path.home() / "deception-detection"
APP = "probe-shift_aswin-surya_sub-apollo_llama-check"
LLAMA_CACHE = "probe-shift_aswin-surya_sub-apollo_llama-cache"
OUTPUT = "probe-shift_aswin-surya_sub-apollo_deception-probe"
SECRET = modal.Secret.from_name("huggingface-secret-aswin")  # HF_TOKEN
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
TOKENIZER = "meta-llama/Meta-Llama-3.1-70B-Instruct"
GPU = "H100:2"  # 141 GB of fp16 weights
TIMEOUT_S = 6 * 3600
CACHE_ENV = {"HF_HUB_CACHE": "/data/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "1"}

app = modal.App(APP)
llama_cache = modal.Volume.from_name(LLAMA_CACHE, create_if_missing=True)
out_vol = modal.Volume.from_name(OUTPUT, create_if_missing=True)

# Apollo's own environment: their package and pinned dependencies (Python 3.11, torch~=2.2), with transformers
# held at a release from their last commit's era (Feb 2025) so it stays compatible with torch 2.2.
apollo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .add_local_dir(APOLLO_REPO, "/root/deception-detection", copy=True, ignore=[".git", "results"])
    .run_commands("cd /root/deception-detection && pip install -e . 'transformers==4.48.3' hf_transfer")
    .env(CACHE_ENV)
)
port_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "scikit-learn==1.7.2",
                    "huggingface_hub[hf_transfer]")
    .env(CACHE_ENV)
    .add_local_python_source("probe_shift")
)


@app.function(image=port_image, volumes={"/data/huggingface": llama_cache}, secrets=[SECRET], cpu=8, memory=32768, timeout=TIMEOUT_S)
def download() -> dict:
    from huggingface_hub import model_info, snapshot_download

    access = {}
    for repo in (MODEL, TOKENIZER):
        try:
            model_info(repo, files_metadata=False)
            access[repo] = "ok"
        except Exception as e:  # report and stop rather than download half of what is needed
            access[repo] = f"{type(e).__name__}: {str(e)[:200]}"
    print(json.dumps(access, indent=1), flush=True)
    if any(v != "ok" for v in access.values()):
        return {"access": access}
    snapshot_download(TOKENIZER, allow_patterns=["*.json", "tokenizer*"])
    snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    llama_cache.commit()
    return {"access": access, "downloaded": [MODEL, TOKENIZER]}


@app.function(image=apollo_image, gpu=GPU, volumes={"/data/huggingface": llama_cache, "/out": out_vol}, secrets=[SECRET],
              timeout=TIMEOUT_S, memory=131072)
def run_original() -> dict:
    import shutil

    repo = Path("/root/deception-detection")
    subprocess.run(["python", "deception_detection/scripts/experiment.py", "run", "--config_file", "repe"], cwd=repo, check=True)
    folder = max((repo / "results").iterdir(), key=lambda p: p.stat().st_mtime)
    dest = Path("/out") / "2026-10-06_apollo_original_llama-70b-3.3" / folder.name
    shutil.copytree(folder, dest)
    out_vol.commit()
    return {"folder": folder.name, "results_table": (folder / "results_table.csv").read_text(),
            "scores": json.loads((folder / "scores.json").read_text()).get("roleplaying__plain")}


@app.function(image=port_image, gpu=GPU, volumes={"/data/huggingface": llama_cache, "/out": out_vol}, secrets=[SECRET],
              timeout=TIMEOUT_S, memory=131072)
def run_port(cfg: dict, train: list[dict], evals: list[dict], detector_bytes: bytes) -> dict:
    import io

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.apollo import fit_and_score

    det = pickle.load(io.BytesIO(detector_bytes))  # Apollo's probe: layers, directions, scaler mean/scale (torch tensors)
    reference = {k: (v.float().numpy().tolist() if hasattr(v, "numpy") else v) for k, v in det.items()
                 if k in ("layers", "directions", "scaler_mean", "scaler_scale")}
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["dtype"]), device_map="auto",
                                                attn_implementation="sdpa").model.eval()
    return fit_and_score(cfg, train, evals, tok, base, out_dir=Path("/out") / cfg["run"], reference=reference) | {
        "device": torch.cuda.get_device_name()}


def _save(name: str, result: dict) -> Path:
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"{name}_{stamp}.json"
    path.write_text(json.dumps(result, indent=1))
    return path


@app.local_entrypoint()
def original():
    commit = subprocess.run(["git", "-C", str(APOLLO_REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    result = run_original.remote() | {"apollo_repo_commit": commit, "gpu": GPU}
    print(result["results_table"])
    print(f"results -> {_save('apollo_original', result)}")


@app.local_entrypoint()
def port():
    import numpy as np
    import yaml
    from sklearn.metrics import roc_auc_score

    from probe_shift.apollo import instructed_pairs, roleplaying

    cfg = yaml.safe_load((HERE / "config" / "port_llama.yaml").read_text())
    shipped = APOLLO_REPO / "example_results" / "instructed_pairs"
    train, evals = instructed_pairs(APOLLO_REPO), roleplaying(APOLLO_REPO)
    result = run_port.remote(cfg, train, evals, (shipped / "detector.pt").read_bytes())

    # Compare with Apollo's shipped per-response scores (their probe, their activations).
    theirs = json.loads((shipped / "scores.json").read_text())["roleplaying__plain"]
    their_mean = np.array([np.mean(x) if len(x) else np.nan for x in theirs["scores"]])
    their_ntok = np.array([len(x) for x in theirs["scores"]])
    # Apollo shuffles a dataset on load (np.random.seed(42); permutation), so their j-th score is our perm[j]-th dialogue.
    np.random.seed(42)
    perm = np.random.permutation(len(evals))
    ours = [result["reference_scores"][i] for i in perm]
    mine_mean = np.array([r["score"] for r in ours])
    mine_ntok = np.array([r["n_tokens"] for r in ours])
    label_name = {1: "DECEPTIVE", 0: "HONEST", None: "AMBIGUOUS"}
    labels_match = float(np.mean([label_name[evals[i]["label"]] == l for i, l in zip(perm, theirs["labels"])]))
    labels = np.array([{"HONEST": 0, "DECEPTIVE": 1}.get(l, -1) for l in theirs["labels"]])
    ok = labels >= 0
    result["comparison_with_shipped"] = {
        "label_alignment_share": labels_match,
        "same_token_count_share": float((mine_ntok == their_ntok).mean()),
        "pearson_r_mean_scores": float(np.corrcoef(their_mean, mine_mean)[0, 1]),
        "max_abs_diff_mean_scores": float(np.nanmax(np.abs(their_mean - mine_mean))),
        "auroc_shipped_scores": float(roc_auc_score(labels[ok], their_mean[ok])),
        "auroc_shipped_probe_on_port_activations": float(roc_auc_score(labels[ok], mine_mean[ok])),
    }
    print(json.dumps(result["comparison_with_shipped"], indent=1))
    for l, r in result["by_layer"].items():
        print(f"layer {l}: retrained port Roleplaying AUROC {r['roleplaying_auroc']:.4f}, instructed-pairs val AUROC {r['instructed_pairs_val_auroc']:.4f}")
    print(f"results -> {_save('port_llama', result | {'config': cfg})}")
