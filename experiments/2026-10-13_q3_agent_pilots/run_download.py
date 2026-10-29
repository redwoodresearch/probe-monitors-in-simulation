"""Download the two pilot models' pinned weights into the shared HF cache Volume (probe-shift_aswin-surya_hf-cache,
mounted at /hf as HF_HOME, the Volume run_vllm_tunnel.py serves from), once, on CPU. One function per model, one
container each, retries 0, an explicit timeout. RULE.md; SPEND.md.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run run_download.py
"""
import modal

APP = "probe-shift_aswin-surya_sub-agentpilots_download_kill-2026-11-13"
MODELS = {
    "gptoss": ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a", ["original/*", "metal/*"]),
    "qcoder": ("Qwen/Qwen3-Coder-Next", "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb", []),
}
app = modal.App(APP)
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub[hf_transfer]==0.36.0")
         .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}))


@app.function(image=image, volumes={"/hf": hf}, cpu=8, memory=16384, timeout=3600, retries=0)
def download(key: str) -> dict:
    import os
    from huggingface_hub import snapshot_download
    repo, rev, ignore = MODELS[key]
    path = snapshot_download(repo, revision=rev, ignore_patterns=ignore or None)
    hf.commit()
    files = sorted(os.listdir(path))
    size = sum(os.path.getsize(os.path.join(path, f)) for f in files if os.path.isfile(os.path.join(path, f)))
    return {"key": key, "repo": repo, "revision": rev, "path": path, "files": len(files), "bytes": size,
            "safetensors": sum(f.endswith(".safetensors") for f in files)}


@app.local_entrypoint()
def main():
    for r in download.map(list(MODELS)):
        print(r)
