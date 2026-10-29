"""POST HOC (RULE.md Amendment 6, 09's note): the honest insertion Δ with and without the one honest run whose original
carries a harness user message after its first assistant turn. Modal CPU; reads the staged inputs (counts and keys only)
and the pilot's per-run scores; same bootstrap as the scorer (2,000 resamples of the runs, seed 20261013).

  uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run \\
      experiments/2026-10-14_onpolicy_patching/check_POSTHOC_late_message.py --scores <patching_scores json> --draw <draw json>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-patching_check_kill-2026-11-14"
INPUTS = "2026-10-14_q2_onpolicy_patching_inputs_v1"

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=image, cpu=2, memory=4096, timeout=900, retries=0,
              volumes={"/acts": modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only()})
def late(keys: dict, per_run: list[dict]) -> dict:
    import numpy as np

    late_keys = []
    for line in (Path("/acts") / INPUTS / "inputs" / "shard_0.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["arm"] == "patch-original" and r["key"] in keys:
            msgs = r["messages"]
            i0 = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"), len(msgs))
            if any(m["role"] == "user" for m in msgs[i0 + 1:]):
                late_keys.append(r["key"])
    late_tasks = {keys[k]["task"] for k in late_keys}
    out = {"late_runs": [{"task": keys[k]["task"], "arm": keys[k]["arm"]} for k in late_keys]}
    rng = np.random.default_rng(20261013)
    for ro in ("reply", "command"):
        for layer in (48, 63):
            for c in ("c", "b", "a"):
                rows = [r for r in per_run if r["readout"] == ro and r["layer"] == layer and r["version"] == "rewritten"
                        and r["view"] == "changed" and r["cell"] == c and r["arm"] == "honest"]
                for name, rr in (("all", rows), ("without_late", [r for r in rows if r["task"] not in late_tasks])):
                    d = np.array([r["rewritten_score"] - r["original_score"] for r in rr])
                    bd = np.array([d[i].mean() for i in rng.integers(0, len(d), (2000, len(d)))])
                    out[f"{ro}_L{layer}_{c}_{name}"] = {"runs": len(d), "delta": round(float(d.mean()), 4),
                                                        "ci95": [round(float(np.percentile(bd, q)), 4) for q in (2.5, 97.5)]}
    return out


@app.local_entrypoint()
def main(scores: str, draw: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    raw = Path(scores).read_bytes()
    per_run = json.loads(raw)["per_run"]
    keys = {p["rel"].replace("/", "__"): {"task": p["task"], "arm": p["arm"]} for p in json.loads(Path(draw).read_bytes())["picked"]}
    res = late.remote(keys, per_run)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"git": git("rev-parse", "HEAD"), "scores": Path(scores).name, "scores_sha256": hashlib.sha256(raw).hexdigest()} | res
    p = HERE / "results" / f"check_POSTHOC_late_message_{stamp}.json"
    with p.open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("late", "reply_L48_c", "reply_L63_c"))}), flush=True)
