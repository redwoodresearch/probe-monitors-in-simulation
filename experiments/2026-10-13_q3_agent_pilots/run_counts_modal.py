"""The agent pilots' slip counts and gpt-oss's empty-new_str matcher on Modal CPU (Aswin, 10-13 17:24: nothing runs on
the laptop). The honest run copies and their request logs are uploaded beforehand to the Volume below
(`modal volume put`); the pinned gpt-oss tokenizer is sent with the call (its sha256 is in the matcher's output). One CPU container runs
slip_counts_pilots.py for both models and check_empty_new_str_gptoss.py for gpt-oss, and returns their result files,
which the laptop writes into results/ beside the earlier ones (append-only).

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-13_q3_agent_pilots/run_counts_modal.py \\
      --git <commit> --tokenizer <gpt-oss tokenizer.json at b5c939de> [--parts slips,matcher]
"""

import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
VOLUME = "probe-shift_aswin-surya_sub-q3pilots_runs_kill-2026-11-13"
RUNS = {"qcoder": ["2026-10-13_155403_qwen3_coder_next_smoke", "2026-10-13_161201_qwen3_coder_next"],
        "gptoss": ["2026-10-13_155631_gpt_oss_120b_smoke", "2026-10-13_165239_gpt_oss_120b"]}
TOKENIZER = "/root/gptoss_tokenizer.json"

app = modal.App("probe-shift_aswin-surya_sub-q3pilots_counts_kill-2026-11-13")
image = (modal.Image.debian_slim(python_version="3.14").pip_install("tokenizers")
         .add_local_file(HERE / "slip_counts_pilots.py", "/root/pilots/slip_counts_pilots.py")
         .add_local_file(HERE / "check_empty_new_str_gptoss.py", "/root/pilots/check_empty_new_str_gptoss.py"))
runs_vol = modal.Volume.from_name(VOLUME)


@app.function(image=image, cpu=2, memory=4096, timeout=1800, retries=0, volumes={"/data": runs_vol})
def counts(git: str, tokenizer_json: bytes, parts: str) -> dict[str, str]:
    Path(TOKENIZER).write_bytes(tokenizer_json)
    out = Path("/root/pilots/results")
    out.mkdir(exist_ok=True)
    py = [sys.executable, "-I"]
    for model, runs in RUNS.items() if "slips" in parts else []:
        subprocess.run(py + ["/root/pilots/slip_counts_pilots.py", "--model", model, "--git", git,
                             "--runs", *[f"/data/{r}" for r in runs]], check=True)
    if "matcher" in parts:
        subprocess.run(py + ["/root/pilots/check_empty_new_str_gptoss.py", "--tokenizer", TOKENIZER, "--git", git,
                             "--runs", *[f"/data/{r}" for r in RUNS["gptoss"]]], check=True)
    return {f.name: f.read_text() for f in out.glob("*.json")}


@app.local_entrypoint()
def main(git: str, tokenizer: str, parts: str = "slips,matcher"):
    files = counts.remote(git, Path(tokenizer).read_bytes(), parts)
    for name, text in sorted(files.items()):
        dst = HERE / "results" / name
        with dst.open("x") as fh:  # append-only: never overwrite an earlier result
            fh.write(text)
        print("->", dst)
