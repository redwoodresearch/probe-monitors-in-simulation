"""POST HOC (plan_fix939_POSTHOC_2026-10-11.md), step 1: stage the both-fixes copy of every simulated attack run in #36's
base dump whose messages the fixes change. As ../2026-10-07_q2_rendering_swap/run_swap.py stage: the variant
diff_with_note+sr_gutter_no_newline, applied in that order, and a stop if any message other than a fixed output differs.
Modal CPU; writes only to a new staged run and returns counts and keys (no message text; failures are re-raised as type
and position only).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \
      experiments/2026-10-08_q3_swap_cue/run_fix939_stage.py --coverage results/fix_coverage_POSTHOC_2026-10-11_010321.json \
      --scores-dir <#36 run_scores_2026-10-06_181548>
"""

import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
SWAP = HERE.parent / "2026-10-07_q2_rendering_swap"
APP = "probe-shift_aswin-surya_sub-q3swapcue_fix939stage_kill-2026-11-10"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
BASE, DST, ARM = "2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-11_q3_fix939_inputs_v1", "sim-attack-note-sr-both"

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "pyyaml")
         .add_local_file(SWAP / "rerender.py", "/root/rerender.py")
         .add_local_file(SWAP / "str_replace_tells.py", "/root/str_replace_tells.py")
         .add_local_python_source("probe_shift"))


@app.function(image=image, volumes={"/acts": acts_vol}, cpu=4, memory=32768, timeout=1800, retries=0)
def stage(keys: list[str]) -> dict:
    try:
        return _stage(keys)
    except Exception as e:  # noqa: BLE001 — sanitized and re-raised, never swallowed
        raise RuntimeError(f"stage failed: {type(e).__name__} at {getattr(e, 'where', 'unknown position')}") from None


def _stage(keys: list[str]) -> dict:
    import sys

    sys.path.insert(0, "/root")
    import rerender as rr
    import str_replace_tells as st

    dst = Path("/acts") / DST
    if dst.exists():
        raise FileExistsError(f"{DST} exists: a staged run is never overwritten")
    want, out, report = set(keys), [], {}
    for p in sorted((Path("/acts") / BASE / "inputs").glob("shard_*.jsonl")):
        with p.open() as fh:
            for ln, line in enumerate(fh):
                try:
                    it = json.loads(line)
                    if it["arm"] != "sim-attack" or it["key"] not in want:
                        continue
                    new, rec_i = rr.rerender_messages(it["messages"], "diff_with_note")
                    new, rec_s = st.swap_messages(new, "sr_gutter_no_newline")
                except Exception as e:
                    e.where = f"{p.name} line {ln}"
                    raise
                rendered = sorted({r["msg"] for r in rec_i if r["category"] in rr.RENDERED} | {r["msg"] for r in rec_s if r["changed"]})
                changed = [i for i, (a, b) in enumerate(zip(it["messages"], new)) if a != b]
                if changed != rendered or len(new) != len(it["messages"]) or not changed:
                    raise ValueError(f"{it['key']}: changed messages differ from the fixed outputs, or none changed")
                if it["key"] in report:
                    raise ValueError(f"{it['key']} appears twice in the base inputs")
                out.append(it | {"arm": ARM, "messages": new})
                report[it["key"]] = {f"changed_messages {ARM}": changed}
    if set(report) != want:
        raise ValueError(f"staged {len(report)} of {len(want)} runs")
    (dst / "inputs").mkdir(parents=True)
    body = "".join(json.dumps(x) + "\n" for x in sorted(out, key=lambda x: x["key"]))
    (dst / "inputs" / "shard_0.jsonl").write_text(body)
    (dst / "rerender_report.json").write_text(json.dumps(report))
    acts_vol.commit()
    return {"staged": len(out), "inputs_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "changed_messages": sum(len(v[f"changed_messages {ARM}"]) for v in report.values())}


@app.local_entrypoint()
def main(coverage: str, scores_dir: str):
    cov = json.loads(Path(coverage).read_text())
    runs = json.loads((Path(scores_dir) / "runs.json").read_text())
    cells = {r["cell"]: r["key"] for r in runs if r["arm"] == "sim-attack"}
    keys = sorted(cells[r["cell"]] for r in cov["per_run"] if r["any_changed"])
    if len(keys) != cov["runs_with_any_changed_message_including_past_the_cut"]:
        raise ValueError("the coverage file's changed runs don't match its count")
    print(json.dumps(stage.remote(keys) | {"keys": len(keys)}, indent=1))
