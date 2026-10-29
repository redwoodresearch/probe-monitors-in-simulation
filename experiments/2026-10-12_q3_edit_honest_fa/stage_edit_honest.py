"""POST HOC (RULE.md), step 1: stage the edited copy of every simulated honest run with the exact fix939 recipe, plus a
smoke set of 4 unedited simulated honest runs for the pipeline check. Modal CPU. Honest runs only: no attack run is read.

The recipe is ../2026-10-08_q3_swap_cue/run_fix939_stage.py's, with the same two modules (blobs pinned below): q2's
rerender.rerender_messages(messages, "diff_with_note") (insert outputs gain the real review note and lose the trailing
newline), then str_replace_tells.swap_messages(messages, "sr_gutter_no_newline") (str_replace diffs lose the trailing
newline; plain diffs gain the real numbered gutter). As there, a run stops the staging if any message other than a fixed
output changed. A run the recipe leaves unchanged is not staged: its inputs are byte-identical, so its as-is scores stand
for "edited" (as fix939's 116 unchanged runs).

Sources: arm sim-honest of 2026-10-06_la_segment-end-v4_qwen3-32b_v1 (the staged inputs of the v4 all-layers
extraction, which the committed false-alarm numbers used), for the 437 runs of sim_honest_scores_2026-10-07_124717.
Writes new staged runs only (never overwrites); returns counts, keys and hashes, no message text.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run \
      experiments/2026-10-12_q3_edit_honest_fa/stage_edit_honest.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
SWAP = REPO / "experiments" / "2026-10-07_q2_rendering_swap"
BLOBS = {"rerender.py": "31e658b4c7676f81c7b7c4d4a98582a2b0c732a4", "str_replace_tells.py": "5c6e5a846f8bc793e84857cf1c5578610ebf63a3"}
APP = "probe-shift_aswin-surya_sub-q3edithonest_stage_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
SRC, ARM_IN = "2026-10-06_la_segment-end-v4_qwen3-32b_v1", "sim-honest"
DST, ARM = "2026-10-12_q3_edithonest_inputs_v1", "sim-honest-note-sr-both"
SMOKE = "2026-10-12_q3_edithonest_smoke_inputs_v1"
RUNS = REPO / "experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json"
SMOKE_N, SMOKE_SEED = 4, 20261012

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
         .add_local_file(SWAP / "rerender.py", "/root/rerender.py")
         .add_local_file(SWAP / "str_replace_tells.py", "/root/str_replace_tells.py")
         .add_local_python_source("probe_shift"))  # rerender.read_only imports probe_shift.calls (as fix939's image)


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=4, memory=32768, timeout=1800, retries=0)
def stage(keys: list[str], smoke_keys: list[str]) -> dict:
    try:
        return _stage(keys, smoke_keys)
    except Exception as e:  # noqa: BLE001 — re-raised with type and position only, never swallowed
        raise RuntimeError(f"stage failed: {type(e).__name__} at {getattr(e, 'where', 'unknown position')}") from None


def _stage(keys: list[str], smoke_keys: list[str]) -> dict:
    import sys

    sys.path.insert(0, "/root")
    import rerender as rr
    import str_replace_tells as st

    for d in (DST, SMOKE):
        if (Path("/acts") / d).exists():
            raise FileExistsError(f"{d} exists: a staged run is never overwritten")
    want, smoke = set(keys), set(smoke_keys)
    out, smoke_out, report = [], [], {}
    for p in sorted((Path("/acts") / SRC / "inputs").glob("shard_*.jsonl")):
        with p.open() as fh:
            for ln, line in enumerate(fh):
                it = json.loads(line)
                if it["arm"] != ARM_IN or it["key"] not in want:
                    continue
                try:
                    new, rec_i = rr.rerender_messages(it["messages"], "diff_with_note")
                    new, rec_s = st.swap_messages(new, "sr_gutter_no_newline")
                except Exception as e:
                    e.where = f"{p.name} line {ln}"
                    raise
                rendered = sorted({r["msg"] for r in rec_i if r["category"] in rr.RENDERED} | {r["msg"] for r in rec_s if r["changed"]})
                changed = [i for i, (a, b) in enumerate(zip(it["messages"], new)) if a != b]
                if changed != rendered or len(new) != len(it["messages"]):
                    raise ValueError(f"{it['key']}: changed messages differ from the fixed outputs")
                if it["key"] in report:
                    raise ValueError(f"{it['key']} appears twice in the source inputs")
                report[it["key"]] = {"changed_messages": changed, "tool_outputs": sum(m["role"] == "tool" for m in it["messages"]),
                                     "inserts_rendered": sum(r["category"] in rr.RENDERED for r in rec_i),
                                     "str_replace_changed": sum(r["changed"] for r in rec_s), "tokens": it["tokens"]}
                if changed:
                    out.append(it | {"arm": ARM, "messages": new})
                if it["key"] in smoke:
                    smoke_out.append(it)  # byte-identical original
    if set(report) != want:
        raise ValueError(f"found {len(report)} of {len(want)} runs")
    if len(smoke_out) != len(smoke):
        raise ValueError("smoke runs missing")
    bodies = {}
    for d, items in ((DST, out), (SMOKE, smoke_out)):
        (Path("/acts") / d / "inputs").mkdir(parents=True)
        body = "".join(json.dumps(x) + "\n" for x in sorted(items, key=lambda x: x["key"]))
        (Path("/acts") / d / "inputs" / "shard_0.jsonl").write_text(body)
        bodies[d] = hashlib.sha256(body.encode()).hexdigest()
    (Path("/acts") / DST / "stage_report.json").write_text(json.dumps(report))
    acts_vol.commit()
    return {"runs": len(report), "changed_runs": len(out), "unchanged_runs": len(report) - len(out),
            "changed_messages": sum(len(v["changed_messages"]) for v in report.values()),
            "tool_outputs": sum(v["tool_outputs"] for v in report.values()),
            "inserts_rendered": sum(v["inserts_rendered"] for v in report.values()),
            "str_replace_changed": sum(v["str_replace_changed"] for v in report.values()),
            "tokens_changed_runs_uncut": sum(v["tokens"] for k, v in report.items() if v["changed_messages"]),
            "tokens_changed_runs_cut_32k": sum(min(v["tokens"], 32768) for k, v in report.items() if v["changed_messages"]),
            "inputs_sha256": bodies[DST], "smoke_inputs_sha256": bodies[SMOKE], "smoke_keys": sorted(smoke),
            "per_run": {k: {"changed_messages": len(v["changed_messages"]), "first_changed_message": (v["changed_messages"] or [None])[0]}
                        for k, v in report.items()}}


@app.local_entrypoint()
def main():
    import numpy as np

    for f, blob in BLOBS.items():  # the exact fix939 editor code
        got = subprocess.run(["git", "hash-object", str(SWAP / f)], capture_output=True, text=True, check=True).stdout.strip()
        if got != blob:
            raise SystemExit(f"{f} is {got}, not the fix939 blob {blob}")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()
    runs = json.loads(RUNS.read_text())
    keys = sorted(r["key"] for r in runs if r["arm"] == ARM_IN)
    if len(keys) != 437 or len(set(keys)) != 437:
        raise SystemExit(f"{len(keys)} simulated honest runs, not 437")
    smoke = sorted(np.random.default_rng(SMOKE_SEED).choice(keys, SMOKE_N, replace=False).tolist())
    res = stage.remote(keys, smoke)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"stamp": stamp, "git": head, "app": APP, "source": SRC, "staged": DST, "arm": ARM, "smoke_staged": SMOKE,
           "runs_json_sha256": hashlib.sha256(RUNS.read_bytes()).hexdigest(), "editor_blobs": BLOBS} | res
    (HERE / "results").mkdir(exist_ok=True)
    with (HERE / "results" / f"stage_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "per_run"}, indent=1))
