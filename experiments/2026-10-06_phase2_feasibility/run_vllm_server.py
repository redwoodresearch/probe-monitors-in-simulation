"""OpenAI-compatible vLLM servers on Modal for the Phase 2 policy: Qwen3-32B, and Qwen3.5-27B as the fallback.

Every request must carry `Authorization: Bearer <key>`: vLLM reads the key from VLLM_API_KEY, which comes from a
Modal Secret (made by run_setup_key.py). Settings are in config/servers.yaml. Weights come from the HF-cache volume,
read-only, offline. Each endpoint scales to zero after `scaledown_window_s` idle and never above one GPU.

  M="uv run --no-project --with modal==1.2.6 --with pyyaml modal"
  $M run experiments/2026-10-06_phase2_feasibility/run_vllm_server.py::check     # L4 for a minute: vLLM version, flags
  $M deploy experiments/2026-10-06_phase2_feasibility/run_vllm_server.py         # both endpoints
  $M app stop probe-shift_aswin-surya_sub-phase2_vllm_kill-2026-11-06            # tear down
"""

import os
import subprocess
import threading
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-phase2_vllm_kill-2026-11-06"
SECRET = "probe-shift_aswin-surya_sub-phase2_vllm-key_kill-2026-11-06"  # holds VLLM_API_KEY
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
CONFIG_IN_IMAGE = "/root/servers.yaml"
PORT = 8000


def _config() -> dict:
    import yaml

    path = HERE / "config" / "servers.yaml" if modal.is_local() else Path(CONFIG_IN_IMAGE)
    return yaml.safe_load(path.read_text())


CFG = _config()
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={CFG['vllm_version']}", "pyyaml")
    .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
    .add_local_file(HERE / "config" / "servers.yaml", CONFIG_IN_IMAGE)
)
app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE)
GPU_FUNCTION = dict(
    image=image,
    gpu=CFG["gpu"],
    volumes={"/hf": hf_cache.read_only()},
    secrets=[modal.Secret.from_name(SECRET)],
    timeout=CFG["request_timeout_s"],
    scaledown_window=CFG["scaledown_window_s"],
    max_containers=CFG["max_containers"],
)


def _serve(key: str) -> None:
    model = CFG["models"][key]
    cmd = ["vllm", "serve", model["hf_id"], f"--revision={model['revision']}", f"--served-model-name={model['hf_id']}",
           "--host=0.0.0.0", f"--port={PORT}", *model["args"]]
    print("launching:", cmd, flush=True)
    process = subprocess.Popen(cmd)

    def exit_with_vllm() -> None:
        # Without this a vLLM that dies at startup leaves the GPU container waiting out startup_timeout.
        code = process.wait()
        print(f"vllm exited with {code}; stopping the container", flush=True)
        os._exit(1)

    threading.Thread(target=exit_with_vllm, daemon=True).start()


@app.function(**GPU_FUNCTION)
@modal.concurrent(max_inputs=CFG["max_concurrent_requests"])
@modal.web_server(PORT, startup_timeout=CFG["startup_timeout_s"])
def serve_qwen3_32b():
    _serve("qwen3_32b")


@app.function(**GPU_FUNCTION)
@modal.concurrent(max_inputs=CFG["max_concurrent_requests"])
@modal.web_server(PORT, startup_timeout=CFG["startup_timeout_s"])
def serve_qwen35_27b():
    _serve("qwen35_27b")


@app.function(**GPU_FUNCTION)
@modal.concurrent(max_inputs=CFG["max_concurrent_requests"])
@modal.web_server(PORT, startup_timeout=CFG["startup_timeout_s"])
def serve_qwen3_32b_native():
    _serve("qwen3_32b_native")


@app.function(image=image, gpu="L4", timeout=600)  # vLLM will not parse its own flags without a GPU
def check_flags() -> dict:
    """The installed vLLM version, and whether each flag and parser in the config exists, on a cheap GPU."""
    runs = {name: subprocess.run(cmd, capture_output=True, text=True) for name, cmd in
            {"version": ["vllm", "--version"], "help": ["vllm", "serve", "--help=all"]}.items()}
    failed = {name: r.stderr[-3000:] for name, r in runs.items() if r.returncode != 0}
    if failed:
        return {"failed": failed}
    version, help_text = runs["version"].stdout.strip(), runs["help"].stdout
    args = [a for m in CFG["models"].values() for a in m["args"]]
    flags = sorted({a.split("=")[0] for a in args})
    parsers = sorted({a.split("=", 1)[1] for a in args if a.startswith(("--tool-call-parser", "--reasoning-parser"))})
    return {"vllm": version, "flags_missing_from_help": [f for f in flags if f not in help_text],
            "parsers_missing_from_help": [p for p in parsers if p not in help_text]}


@app.local_entrypoint()
def check():
    import json

    result = check_flags.remote()
    print(json.dumps(result, indent=1))
    if result.get("failed") or result.get("flags_missing_from_help") or result.get("parsers_missing_from_help"):
        raise SystemExit("vLLM check failed; do not deploy this config")
