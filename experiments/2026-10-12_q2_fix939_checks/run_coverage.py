"""fix939 follow-up (H), truncation coverage (PLAN.md addendum H, committed with this script before it runs): one Modal
CPU container reads the extraction metadata of fix939's 823 re-rendered runs and their originals and returns, per run,
token counts and readout counts only: the run's length and truncation, outputs past the 32,768-token cut, outputs pooled,
and whether the pooled (step, action) set is identical. No token text is read or returned.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-12_q2_fix939_checks/run_coverage.py --runs <#36 runs.json> --fix <fix939 per-run jsonl>
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-fix939-coverage_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
DUMP = "2026-10-06_la_segment-end_qwen3-32b_v1"
FIX_DUMP, FIX_ARM = "2026-10-11_q3_fix939_qwen3-32b_l48-63_v1", "sim-attack-note-sr-both"
CUT = 32768  # max_tokens in both extraction configs
app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12")
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96",
       "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl": "5f6ecbc42b1c6bb9586cb58e474d74c5c1d914bedb95d80a5a07dd078b0e5ebd"}


@app.function(image=image, volumes={"/acts": acts}, cpu=4, memory=8192, timeout=1800, retries=0)
def coverage(runs: list[dict]) -> list[dict]:
    def summary(path):
        m = json.loads(path.read_text())
        out = [p for p in m["positions"] if p["kind"] == "output_end"]
        for p in out:  # fail loudly: kept must mean a token inside the cut
            if p["kept"] != (p["token"] is not None and p["token"] < min(m["n_tokens"], CUT)):
                raise ValueError("an output_end position's kept flag disagrees with its token index and the cut")
        if m["truncated"] != (m["n_tokens"] > CUT):
            raise ValueError("a run's truncated flag disagrees with its length and the cut")
        return {"n_tokens": m["n_tokens"], "truncated": m["truncated"], "outputs": len(out),
                "past the cut": sum(p["token"] is not None and not p["kept"] for p in out),
                "empty (no content token)": sum(p["token"] is None for p in out),
                "pooled": sum(p["kept"] for p in out)}, {(p["step"], p["action"]) for p in out if p["kept"]}

    res = []
    for r in runs:
        a, sa = summary(Path("/acts") / DUMP / "acts" / "sim-attack" / f"{r['filename']}.json")
        b, sb = summary(Path("/acts") / FIX_DUMP / "meta" / FIX_ARM / f"{r['filename']}.json")
        res.append({"key": r["key"], "original": a, "re-rendered": b, "pooled set identical": sa == sb,
                    "pooled lost": len(sa - sb), "pooled gained": len(sb - sa)})
    return res


@app.local_entrypoint()
def main(runs: str, fix: str):
    for p_, role in ((Path(runs), "runs.json"), (Path(fix), "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl")):
        if hashlib.sha256(p_.read_bytes()).hexdigest() != PIN[role]:
            raise SystemExit(f"{p_.name} is not the pinned {role}")
    fname = {r["key"]: r["filename"] for r in json.loads(Path(runs).read_text()) if r["arm"] == "sim-attack"}
    fkeys = sorted({json.loads(l)["key"] for l in open(fix) if json.loads(l)["label"] == "FIX"})
    if len(fkeys) != 823 or not set(fkeys) <= set(fname):
        raise SystemExit("not fix939's 823 re-rendered simulated runs")
    res = coverage.remote([{"key": k, "filename": fname[k]} for k in fkeys])
    out = {"plan": "PLAN.md addendum H (this folder)", "cut (tokens)": CUT,
           "inputs sha256": {Path(runs).name: PIN["runs.json"], Path(fix).name: PIN["fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl"]},
           "runs": res}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    d = HERE.parents[1] / "data" / "fix939_coverage"; d.mkdir(parents=True, exist_ok=True)
    with open(dest := d / f"coverage_{stamp}.json", "x") as fh:  # numbers only, but per run: data/, then S3 with a manifest
        fh.write(json.dumps(out) + "\n")
    print(dest)
