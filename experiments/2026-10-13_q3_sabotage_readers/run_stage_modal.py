"""B3: stage_readers.py on Modal CPU (Aswin, 17:24 PT 10-13: no compute on the laptop). One container per staging job, with
many cores; the .eval cache (uploaded unchanged to the B3 Volume under evals_cache/, each file still sha256-checked against its
launch manifest by stage_readers.counted_attempts), the pinned row files (b3_inputs/), the Llama tokenizer files (llama_tok/),
the HF cache Volume for the other tokenizers, and the LA Volume for the repair's F8 templates. Staged inputs are written on
the B3 Volume (sabreaders_<reader>[_repair]_v1/inputs/), where run_extract_readers.py reads them; the record comes back
here and is written into results/.

  uv run --no-project --python 3.12 --with "modal==1.6.1" modal run run_stage_modal.py --jobs "qwen3,qwen3:repair,seed:repair,..."
"""

import subprocess
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q3sabr_stage_kill-2026-11-13"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root/repo")  # the container imports this file from /root
E = "/root/repo/experiments"

app = modal.App(APP)
b3 = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3sabr_acts_kill-2026-11-13", version=2)
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
la = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("transformers==4.57.1", "jinja2==3.1.6", "numpy", "boto3", "zstandard==0.23.0")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "B3_CACHE": "/v/evals_cache", "B3_STAGE_ROOT": "/v", "B3_RESULTS": "/tmp/results",
               "B3_LLAMA_TOK": "/v/llama_tok", "B3_LA": "/la"})
         .add_local_dir(REPO / "src", "/root/repo/src")
         .add_local_file(HERE / "stage_readers.py", f"{E}/2026-10-13_q3_sabotage_readers/stage_readers.py")
         .add_local_file(HERE / "repair_b2.py", f"{E}/2026-10-13_q3_sabotage_readers/repair_b2.py")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/rerender.py", f"{E}/2026-10-07_q2_rendering_swap/rerender.py")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py", f"{E}/2026-10-07_q2_rendering_swap/str_replace_tells.py")
         .add_local_file(REPO / "experiments/2026-10-09_fixed_sim/fixes.py", f"{E}/2026-10-09_fixed_sim/fixes.py")
         .add_local_file(REPO / "experiments/2026-10-09_fixed_sim/clock.py", f"{E}/2026-10-09_fixed_sim/clock.py")
         .add_local_file(REPO / "experiments/2026-10-06_q1_segment_end/results/tool_errors_2026-10-06_134908.jsonl",
                         f"{E}/2026-10-06_q1_segment_end/results/tool_errors_2026-10-06_134908.jsonl"))


@app.function(image=image, cpu=32, memory=65536, timeout=1800, retries=0,
              volumes={"/v": b3, "/hf": hf.read_only(), "/la": la.read_only()})
def stage(reader: str, repair: bool) -> dict:
    import os
    import sys

    sys.path.insert(0, f"{E}/2026-10-13_q3_sabotage_readers")
    os.makedirs("/tmp/results", exist_ok=True)
    import stage_readers

    stage_readers.main(reader, "/v/b3_inputs", 0, 0, 30, repair)
    files = {p.name: p.read_bytes() for p in Path("/tmp/results").iterdir()}
    os.makedirs("/v/stage_records", exist_ok=True)  # a lost client (18:11 PT: DNS failure) must not lose the record
    for n, b in files.items():
        Path(f"/v/stage_records/{n}").write_bytes(b)
    b3.commit()
    return files


@app.function(image=image, cpu=4, memory=16384, timeout=900, retries=0, volumes={"/v": b3})
def rebuild(name: str, reader: str, repair: bool) -> dict:
    """A staged folder's record rebuilt from its files, after the staging client lost its connection (18:11 PT) and with it
    the records the containers returned: sha256 per file and, per arm, runs, rows, rows read and forward tokens. The staging
    counters (encodings, crossing tokens, repaired outputs) are not recoverable this way and are marked missing."""
    import hashlib
    import json
    from collections import Counter

    d = Path("/v") / name / "inputs"
    shas, summary = {}, {}
    for p in sorted(d.glob("*.jsonl")):
        b = p.read_bytes()
        shas[p.name] = hashlib.sha256(b).hexdigest()
        c = Counter()
        for line in b.decode().splitlines():
            e = json.loads(line)
            c["runs staged"] += 1
            c["rows"] += len(e["calls"])
            c["rows read"] += sum(x["readout"] is not None for x in e["calls"])
            c["forward tokens"] += len(e["ids"])
        summary[p.stem] = dict(c)
    rec = {"reader": reader, "rebuilt_from_volume": True, "counters": "missing (the staging client lost its connection)", "gate": 0, "limit": 0,
           "repair": "B2 fix-full minus F7 (repair_b2.py)" if repair else None, "staged_dir": f"/v/{name}", "staged_inputs_sha256": shas, "summary": summary}
    return {f"stage_{reader}{'_repair' if repair else ''}_rebuilt.json": json.dumps(rec, indent=1).encode()}


@app.local_entrypoint()
def main(jobs: str, rebuild_only: bool = False):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--", "stage_readers.py", "repair_b2.py", Path(__file__).name):
        raise SystemExit("commit the staging code before it runs")
    args = [(j.split(":")[0], j.endswith(":repair")) for j in jobs.split(",")]
    if rebuild_only:
        args = [(f"sabreaders_{r}{'_repair' if rp else ''}_v1", r, rp) for r, rp in args]
    for files in (rebuild if rebuild_only else stage).starmap(args):
        for name, b in files.items():
            with (HERE / "results" / name).open("xb") as fh:
                fh.write(b)
            print("->", HERE / "results" / name)
