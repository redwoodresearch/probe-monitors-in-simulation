"""POST HOC, descriptive (09 via q2, 10-14): model_split_2026-10-14_100204.json's tell per tool output. Qwen3.5-27B's runs are
about 2.5 times longer than Sol's and the probe's run score is a mean over output rows, so per-output rates are what it sees.
Per acting model, arm and window: CRLF-bearing outputs per tool output = mean CRLF-bearing outputs per run / mean replies
per run (the arm's total over its total, every run weighted 1); G per output = real - simulated; Qwen3.5 / Sol. Points only:
the file holds no per-run draws, so no intervals. The frozen reading stays on R as written. Modal CPU.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_q2_ffm_iter1/addendum_per_output.py
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
SRC = "model_split_2026-10-14_100204.json"
SRC_SHA = "27842c06e0b67c83"  # prefix; committed in 7f87fb28
APP = "probe-shift_aswin-surya_sub-q2-ffm1_addendum_kill-2026-11-13"

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").add_local_file(HERE / "results" / SRC, f"/root/{SRC}")


@app.function(image=image, cpu=0.25, memory=512, timeout=600, retries=0)
def convert() -> dict:
    raw = Path(f"/root/{SRC}").read_bytes()
    if not hashlib.sha256(raw).hexdigest().startswith(SRC_SHA):
        raise SystemExit("the model_split result differs from its pin")
    M = json.loads(raw)["means"]
    out = {}
    for win in ("first 21 replies", "whole run"):
        for side in ("attack", "honest"):
            row = {}
            for model in ("Qwen3.5-27B", "Sol"):
                rate = {w: M[model][f"{w} {side}|{win}|CRLF-bearing outputs"]["point"] / M[model][f"{w} {side}|{win}|replies"]["point"]
                        for w in ("real", "sim")}
                row[model] = {"replies per run (real, sim)": [M[model][f"{w} {side}|{win}|replies"]["point"] for w in ("real", "sim")],
                              "CRLF-bearing outputs per tool output (real, sim)": [rate["real"], rate["sim"]],
                              "G per tool output (real - sim)": rate["real"] - rate["sim"]}
            g = row["Sol"]["G per tool output (real - sim)"]
            row["G per tool output, Qwen3.5 / Sol"] = row["Qwen3.5-27B"]["G per tool output (real - sim)"] / g if g else None
            out[f"{side}|{win}"] = row
    return out


@app.local_entrypoint()
def main():
    f = Path(__file__)
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
            or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
        raise SystemExit("commit this script before running")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = convert.remote()
    p = HERE / "results" / f"model_split_per_output_POSTHOC_{stamp}.json"
    with open(p, "x") as fh:
        fh.write(json.dumps({"POST HOC": True, "descriptive": True, "source": f"results/{SRC} @ 7f87fb28", "git": git,
                             "stamp": stamp, "per output": res}, indent=1) + "\n")
    print("wrote", p)
