"""New-task pool for more simulated honest runs (POST HOC, $0, keys and counts only; 0d's question 10-13 ~18:1x PT).

d9's count curve varied the number of distinct tasks (131/197/262), so more simulated honest training negatives only
follow it on NEW tasks. This counts, from committed records only, the public10 main tasks that have a usable September
reference (the n437 settings' overlap800 rule) but are not among the 437 and not in zoo C's sealed 274; how many of
those have real honest runs; and the cost of +N new-task runs. One Modal CPU function does the counting (Aswin 17:24:
no compute on the laptop); the local entrypoint only ships the pinned inputs and writes the returned JSON.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run \
      experiments/2026-10-13_q2_new_task_pool/run_task_pool.py
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if modal.is_local() else HERE  # the container imports this file from /root
APP = "probe-shift_aswin-surya_sub-q2-taskpool_count_kill-2026-11-13"
INPUTS = {  # name in the container: (repo path, what it gives)
    "refs.json": ("experiments/2026-10-06_astra_sim_honest/results/references_20261006-142754.json",
                  "the n437 batch's overlap800 references (437 covered mains) and the 105 uncovered mains"),
    "sim_honest_runs.json": ("experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json",
                             "the 437 simulated honest runs as scored (arm sim-honest)"),
    "runs36.json": ("experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json",
                    "#36's scored runs (real honest, e1, e2, sim attack)"),
    "exclusions_v3.json": ("splits/exclusions_v3.json", "every sample of the September n1000 cohorts, with exclusion reasons"),
    "batch_spec.json": ("experiments/2026-10-11_q2_new_real_runs/results/batch_spec_2026-10-11_151207.json",
                        "zoo C's 274 unpaired simulated-cohort tasks (sealed 10-11 19:21:54 as a confirmation set)"),
}
EXCLUDED_ENVS = ("media_processing", "model_registry")  # the Step 2 pilot's and Oct 7's exclusion (not reproduced)
PRICE = {"api_low": 6.03, "api_high": 7.86, "extract": 0.05}  # lean stage 1's $301.66/50; b7's n437 sample; Qwen3-32B + Seed

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    for name, (path, _) in INPUTS.items():
        image = image.add_local_file(ROOT / path, f"/inputs/{name}")


@app.function(image=image, cpu=2, memory=4096, timeout=600, retries=0)
def count() -> dict:
    def load(n):
        return json.loads(Path(f"/inputs/{n}").read_bytes())

    refs = load("refs.json")
    covered, uncovered = set(refs["references"]), set(refs["uncovered_mains"])
    env_of = {m: v["environment"] for m, v in refs["references"].items()}
    s437 = {r["main_task"] for r in load("sim_honest_runs.json") if r["arm"] == "sim-honest"}
    runs36 = load("runs36.json")
    real36 = {r["main_task"] for r in runs36 if r["arm"] == "real-honest"}
    for r in runs36:
        env_of.setdefault(r["main_task"], r["environment"])
    samples = load("exclusions_v3.json")["samples"]
    for r in samples:
        env_of.setdefault(r["main_task"], r["environment"])
    n1000_all = {r["main_task"] for r in samples if r["arm"] == "real-honest"}
    n1000_valid = {r["main_task"] for r in samples if r["arm"] == "real-honest" and not r["reason"]}
    sim_attack_n1000 = {r["main_task"] for r in samples if r["arm"] == "sim-attack"}
    every_main = {r["main_task"] for r in samples} | covered | uncovered | real36
    sealed = set(load("batch_spec.json")["task order"])

    def envs(ms):
        out = {}
        for m in sorted(ms):
            out[env_of.get(m, "?")] = out.get(env_of.get(m, "?"), 0) + 1
        return dict(sorted(out.items()))

    def real(ms):
        return {"with real honest in #36": len(ms & real36), "with real honest in n1000 (any)": len(ms & n1000_all),
                "with valid real honest in n1000": len(ms & n1000_valid)}

    avail_all_envs = covered - s437 - sealed
    avail = {m for m in avail_all_envs if env_of.get(m) not in EXCLUDED_ENVS}
    unc_unsealed = uncovered - sealed

    def cost(n):
        lo, hi = n * (PRICE["api_low"] + PRICE["extract"]), n * (PRICE["api_high"] + PRICE["extract"])
        return {"runs": n, "raw": [round(lo, 2), round(hi, 2)], "with 30%": [round(1.3 * lo, 2), round(1.3 * hi, 2)]}

    n_av = len(avail)
    return {
        "public10 main tasks seen in the records (all n1000 cohorts, references, #36)": len(every_main),
        "simulated attack n1000 main tasks (exclusions_v3 sim-attack)": len(sim_attack_n1000),
        "references file: covered + uncovered": [len(covered), len(uncovered), len(covered | uncovered)],
        "covered and uncovered overlap": len(covered & uncovered),
        "sim attack n1000 mains outside covered + uncovered": len(sim_attack_n1000 - covered - uncovered),
        "the 437 (sim-honest mains as scored)": len(s437),
        "the 437 minus covered (should be 0)": len(s437 - covered),
        "covered minus the 437": len(covered - s437),
        "sealed 274: count, in the 437, in covered, in uncovered, elsewhere": [
            len(sealed), len(sealed & s437), len(sealed & covered), len(sealed & uncovered), len(sealed - covered - uncovered)],
        "AVAILABLE (usable reference, not in the 437, not sealed), all environments": len(avail_all_envs),
        "AVAILABLE, outside media_processing and model_registry": n_av,
        "available: real honest coverage": real(avail),
        "available: by environment": envs(avail_all_envs),
        "the 437 by environment (media_processing + model_registry included in the batch)": envs(s437),
        "uncovered (no overlap800 reference; base200 only), not sealed": len(unc_unsealed),
        "uncovered, not sealed: real honest coverage": real(unc_unsealed),
        "uncovered, not sealed: by environment": envs(unc_unsealed),
        "real honest n1000 mains with no simulated attack run (outside covered + uncovered)": len(n1000_all - covered - uncovered),
        "cost per run": PRICE,
        "cost": {"+N available": cost(n_av), "+N/2": cost(n_av // 2), "+50": cost(50), "+100": cost(100),
                 "+uncovered unsealed (only if new references existed)": cost(len(unc_unsealed))},
    }


@app.local_entrypoint()
def main():
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(Path(__file__).resolve())], capture_output=True,
                           text=True, cwd=HERE).stdout.strip()
    if dirty:
        raise SystemExit("commit this script before running it")
    inputs = {n: {"path": p, "what": w, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}
              for n, (p, w) in INPUTS.items()}
    out = {"POST HOC": True, "status": "preliminary", "question": "0d, 10-13 ~18:1x PT: public10 main tasks with a usable "
           "September reference, not among the 437, not sealed; real honest coverage; cost of +N new-task runs",
           "definitions": {"usable reference": "the n437 settings' overlap800 reference (references file's 'references')",
                           "sealed": "zoo C's 274 unpaired simulated-cohort tasks (batch spec 'task order')",
                           "excluded environments": list(EXCLUDED_ENVS)},
           "git": git, "inputs": inputs, "result": count.remote()}
    path = HERE / "results" / f"task_pool_POSTHOC_{datetime.now(ZoneInfo('America/Los_Angeles')):%Y-%m-%d_%H%M%S}.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} exists; never overwritten")
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(path)
    print(json.dumps(out["result"], indent=1))
