"""Why does Apollo's deception probe score lower on Qwen3-32B than on Llama-3.3-70B? GPU passes on Modal.

  # weights for the model-size check (into this experiment's volume)
  uv run modal run experiments/2026-10-06_apollo_qwen_gap/run_qwen_gap.py::download --model Qwen/Qwen3-8B
  uv run modal run experiments/2026-10-06_apollo_qwen_gap/run_qwen_gap.py::download --model Qwen/Qwen3-14B
  # (Llama-3.1-8B-Instruct is gated: its download needs a project-approved HF Secret, see download_remote)
  # Qwen3-32B writes its own Roleplaying responses (then grade them with run_grade_roleplaying.py)
  uv run modal run experiments/2026-10-06_apollo_qwen_gap/run_qwen_gap.py::generate
  # every layer, both readouts; config/sweep_<name>.yaml; --pilot runs a small subset first
  uv run modal run experiments/2026-10-06_apollo_qwen_gap/run_qwen_gap.py::sweep --config qwen3-32b
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APOLLO_REPO = Path.home() / "deception-detection"
# The app and this experiment's volume. Label shortened from "qwen-layer-sweep": Modal names must be under 64 characters.
NAME = "probe-shift_aswin-surya_sub-apollo_qwen-gap_kill-2026-11-06"
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "scikit-learn==1.7.2",
       "huggingface_hub[hf_transfer]"]

app = modal.App(NAME)
vol = modal.Volume.from_name(NAME, create_if_missing=True)
VOLUMES = {
    "/qwen-cache": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only(),
    "/llama-cache": modal.Volume.from_name("probe-shift_aswin-surya_sub-apollo_llama-cache").read_only(),
    "/vol": vol,
}
CACHE_DIR = {"qwen-cache": "/qwen-cache/hub", "llama-cache": "/llama-cache", "own": "/vol/hf-hub"}
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*PIP)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("probe_shift")
)


def _load(cfg: dict, causal_lm: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = CACHE_DIR[cfg["weights"]]
    rev = cfg.get("revision")  # pinned snapshot when the config records one
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", cfg["model"]), cache_dir=cache, local_files_only=True, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], cache_dir=cache, local_files_only=True, revision=rev, dtype=getattr(torch, cfg["dtype"]),
                                                 device_map="auto" if cfg.get("n_gpus", 1) > 1 else "cuda", attn_implementation="sdpa").eval()
    return tok, (model if causal_lm else model.model)


# No credentials: personal tokens stay out of Modal. Ungated models (Qwen) download as is; a gated one (Llama) needs a
# project-approved Modal Secret added here first.
@app.function(image=image, volumes={"/vol": vol}, cpu=8, memory=32768, timeout=3600)
def download_remote(repo_id: str, revision: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id, revision=revision, cache_dir=CACHE_DIR["own"], allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    vol.commit()
    return path


@app.function(image=image, gpu="H100!", volumes=VOLUMES, timeout=3600, memory=65536)
def generate_remote(cfg: dict, prompts: list[dict]) -> dict:
    import time

    import torch

    t0 = time.time()
    torch.manual_seed(cfg["seed"])
    tok, model = _load(cfg, causal_lm=True)
    tok.padding_side = "left"
    kwargs = cfg["chat_template"]
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]

    def messages(p, assistant):
        return [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}, {"role": "assistant", "content": assistant}]

    texts = [tok.apply_chat_template(messages(p, p["prefix"]), tokenize=False, continue_final_message=cfg["continue_final_message"], **kwargs)
             for p in prompts]
    assert all(t.endswith(p["prefix"]) for t, p in zip(texts, prompts)), "prompt must end with the answer prefix"
    out = []
    for b in range(0, len(prompts), cfg["batch_size"]):
        enc = tok(texts[b:b + cfg["batch_size"]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, do_sample=True, temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"],
                                 max_new_tokens=cfg["max_new_tokens"], eos_token_id=stop, pad_token_id=tok.pad_token_id)
        for j, row in enumerate(gen[:, enc["input_ids"].shape[1]:].tolist()):
            p = prompts[b + j]
            ended = [k for k, t in enumerate(row) if t in stop]
            new = row[:ended[0]] if ended else row
            response = tok.decode(new, skip_special_tokens=True).rstrip("\n ").lstrip("\n")  # Apollo's postprocess
            # The probe later renders prefix + response with the same template; it must extend exactly this prompt.
            full = tok.apply_chat_template(messages(p, p["prefix"] + response), tokenize=False, **kwargs)
            assert full.startswith(texts[b + j]), f"{p['id']}: probe-time rendering does not extend the generation prompt"
            out.append(p | {"response": response, "n_new_tokens": len(new), "hit_cap": not ended})
        print(f"{len(out)}/{len(prompts)} generated, {time.time() - t0:.0f}s", flush=True)
    return {"generations": out, "example_prompt": texts[0], "device": torch.cuda.get_device_name(), "seconds": round(time.time() - t0, 1)}


def _sweep(cfg: dict, train: list[dict], evals: dict[str, list[dict]]) -> dict:
    import time

    import numpy as np
    import torch

    from probe_shift import apollo_sweep

    out_dir = Path("/vol/runs") / cfg["run"]
    if out_dir.exists() and any(out_dir.iterdir()):  # checked before any GPU work
        raise FileExistsError(f"{out_dir} already holds a run; change cfg['run'] to rerun")
    t0 = time.time()
    tok, base = _load(cfg)
    load_s = time.time() - t0
    print(f"loaded {cfg['model']} in {load_s:.0f}s: {len(base.layers)} blocks, {torch.cuda.device_count()} GPUs", flush=True)
    result, probes = apollo_sweep.sweep(cfg, base, tok, train, evals, log=lambda m: print(m, flush=True))
    out_dir.mkdir(parents=True)
    for r, p in probes.items():
        np.savez(out_dir / f"probes_{r}.npz", **p)
    (out_dir / "result.json").write_text(json.dumps(result))
    vol.commit()
    return result | {"load_seconds": round(load_s, 1), "container_seconds": round(time.time() - t0, 1),
                     "device": torch.cuda.get_device_name(), "n_gpus": torch.cuda.device_count(),
                     "probes_on_volume": f"{NAME}:/runs/{cfg['run']}"}


@app.function(image=image, gpu="H100!", volumes=VOLUMES, timeout=2 * 3600, memory=98304, cpu=16)
def sweep_1gpu(cfg: dict, train: list[dict], evals: dict[str, list[dict]]) -> dict:
    return _sweep(cfg, train, evals)


@app.function(image=image, gpu="H100:2", volumes=VOLUMES, timeout=3 * 3600, memory=131072, cpu=16)
def sweep_2gpu(cfg: dict, train: list[dict], evals: dict[str, list[dict]]) -> dict:
    return _sweep(cfg, train, evals)


def _provenance() -> dict:
    def git(*args, repo=HERE):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()

    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "apollo_repo_commit": git("rev-parse", "HEAD", repo=APOLLO_REPO)}


def _results_dir() -> Path:
    """Created before any paid work starts, so a finished GPU job can always save (git does not keep empty folders)."""
    (HERE / "results").mkdir(exist_ok=True)
    return HERE / "results"


def _save(name: str, result: dict) -> Path:
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = _results_dir() / f"{name}_{stamp}.json"
    with open(path, "x") as f:  # results/ is append-only: "x" raises if the file exists, atomically
        f.write(json.dumps(result, indent=1))
    return path


def _latest(pattern: str) -> Path:
    paths = sorted((HERE / "results").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no results/{pattern}")
    return paths[-1]


@app.local_entrypoint()
def download(model: str):
    print(download_remote.remote(model))


@app.local_entrypoint()
def generate():
    import yaml

    cfg = yaml.safe_load((HERE / "config" / "generate_qwen3-32b.yaml").read_text())
    _results_dir()
    rollouts = json.loads((APOLLO_REPO / "data" / "rollouts" / f"{cfg['prompts']}.json").read_text())["rollouts"]
    assert cfg["n_per_scenario"] == 1
    prompts = []
    for i, r in enumerate(rollouts):  # ids match probe_shift.apollo.roleplaying, so the two sets pair by scenario
        system, user, prefix = (m["content"] for m in r["input_messages"][:3])
        prompts.append({"id": f"roleplaying_{i}", "system": system, "user": user, "prefix": prefix,
                        "scenario": r["metadata"]["scenario"], "question": r["metadata"]["question"]})
    result = generate_remote.remote(cfg, prompts)
    gens = result["generations"]
    print(f"{len(gens)} responses; hit the {cfg['max_new_tokens']}-token cap: {sum(g['hit_cap'] for g in gens)}; "
          f"empty: {sum(not g['response'].strip() for g in gens)}; containing '<think>': {sum('<think>' in g['response'] for g in gens)}")
    for g in gens[:3]:
        print(f"  {g['id']}: {g['prefix']}{g['response'][:200]!r}")
    print(f"results -> {_save('generations_qwen3-32b', result | {'config': cfg | _provenance()})}")


@app.local_entrypoint()
def sweep(config: str, pilot: bool = False, skip_generated: bool = False, run_suffix: str = ""):
    """--pilot: a small subset, to check the pipeline end to end. --skip-generated: only Apollo's Llama-written set.
    --run-suffix rerun: rerun a config into its own volume folder and results file (sweep_<config>_<suffix>_*.json),
    which the analysis does not pick up as that model's sweep."""
    import yaml

    from probe_shift.apollo import instructed_pairs, roleplaying
    from probe_shift.apollo_sweep import roleplaying_from_generations

    cfg = yaml.safe_load((HERE / "config" / f"sweep_{config}.yaml").read_text())
    _results_dir()
    train = instructed_pairs(APOLLO_REPO)
    evals = {"roleplaying_llama-70b-3.3": roleplaying(APOLLO_REPO)}
    provenance = _provenance()
    if not skip_generated:
        gen_path = _latest("generations_qwen3-32b_*.json")
        evals["roleplaying_qwen3-32b"] = roleplaying_from_generations(json.loads(gen_path.read_text())["generations"])
        provenance["generations_file"] = str(gen_path.relative_to(HERE))
    assert list(evals) == cfg["eval_sets"][:len(evals)], (list(evals), cfg["eval_sets"])
    if pilot:
        cfg["run"] += "_pilot"
        train = [d for d in train if d["pair"] < cfg["pilot"]["pairs"]]
        evals = {k: v[::cfg["pilot"]["eval_every"]] for k, v in evals.items()}
    if run_suffix:
        assert run_suffix.replace("-", "").isalnum() and not run_suffix[0].isdigit(), "suffix: letters, digits, dashes; not starting with a digit"
        cfg["run"] += f"_{run_suffix}"
    fn = {1: sweep_1gpu, 2: sweep_2gpu}[cfg["n_gpus"]]
    result = fn.remote(cfg, train, evals)
    name = f"sweep_{config}" + ("_pilot" if pilot else "") + (f"_{run_suffix}" if run_suffix else "")
    gpu = {1: "H100! (sweep_1gpu)", 2: "H100:2 (sweep_2gpu)"}[cfg["n_gpus"]]
    path = _save(name, result | {"config": cfg | provenance | {"modal_app": NAME, "gpu": gpu, "pip": PIP}})

    print(f"\n{result['device']} x{result['n_gpus']}: load {result['load_seconds']}s, forward {result['forward_seconds']}s, "
          f"fit {result['fit_seconds']}s, container {result['container_seconds']}s")
    rows = result["by_probe"]
    print(f"probes not converged: {sum(not r['converged'] for r in rows)} of {len(rows)}")
    for r in rows:
        if r["layer"] % 4 == 0 or r["layer"] == result["n_blocks"] - 1:
            sets = "  ".join(f"{k}: {'-' if r[f'{k}_auroc'] is None else round(r[f'{k}_auroc'], 3)}" for k in evals)
            print(f"  {r['readout']:>12} L{r['layer']:>2} depth {r['depth']:.2f}  instructed val {r['instructed_val_auroc']:.3f}  {sets}")
    print(f"results -> {path}")
