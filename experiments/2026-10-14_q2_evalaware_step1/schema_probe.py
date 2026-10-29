"""Eval-awareness step 1, schema probe (honest only; prints structure, never text).

Opens ONE honest_a .eval of the corrected-B cohort and reports where Qwen3.5-27B's reasoning lives: per assistant message,
the content's type, the item types and keys, character counts per item type, and whether '<think>' appears (boolean).
Also counts, for batch C (evals_v1/2026-10-11_112714_honest_c), .eval files per task and samples per file.
  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach experiments/2026-10-14_q2_evalaware_step1/schema_probe.py
"""
import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q2-evalaware_schema_kill-2026-11-14"
EVALS = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
HERE = Path(__file__).resolve().parent
KEY = ("2026-10-09_000523_rerun_strict/honest_a/t01_media_processing")  # one honest_a task folder (first task of the cohort)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("zstandard==0.23.0")
         .add_local_file(HERE / "evalzip.py", "/root/evalzip.py"))
app = modal.App(APP)


@app.function(image=image, volumes={"/ev": modal.Volume.from_name(EVALS, version=2).read_only()}, cpu=1, memory=2048,
              timeout=300)
def probe() -> dict:
    import collections
    import sys
    import zipfile
    sys.path.insert(0, "/root")
    import evalzip  # noqa: F401  (registers zstd)
    out = {}
    folder = Path("/ev/evals_v1") / KEY
    evs = sorted(folder.glob("*.eval"))
    out["honest_a_files_in_folder"] = len(evs)
    with zipfile.ZipFile(evs[0]) as z:
        names = z.namelist()
        samples = [n for n in names if n.startswith("samples/") and n.endswith(".json")]
        out["zip_members"], out["sample_members"] = len(names), len(samples)
        s = json.loads(z.read(samples[0]))
    out["sample_keys"] = sorted(s.keys())
    msgs = s.get("messages", [])
    out["n_messages"] = len(msgs)
    out["roles"] = dict(collections.Counter(m.get("role") for m in msgs))
    shapes = collections.Counter()
    item_chars = collections.Counter()
    think_in_text = 0
    msg_keys = collections.Counter()
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        msg_keys.update(m.keys())
        c = m.get("content")
        if isinstance(c, str):
            shapes["str"] += 1
            item_chars["str"] += len(c)
            think_in_text += "<think>" in c
        elif isinstance(c, list):
            shapes["list"] += 1
            for it in c:
                t = it.get("type", "?")
                item_chars[t] += sum(len(v) for k, v in it.items() if isinstance(v, str) and k != "type")
                shapes["item:" + t + ":" + ",".join(sorted(it.keys()))] += 1
                if t == "text":
                    think_in_text += "<think>" in it.get("text", "")
    out["assistant_message_keys"] = dict(msg_keys)
    out["assistant_content_shapes"] = dict(shapes)
    out["assistant_chars_by_item_type"] = dict(item_chars)
    out["assistant_text_with_think_tag"] = think_in_text
    c_root = Path("/ev/evals_v1/2026-10-11_112714_honest_c/honest_b")
    per_task = collections.Counter()
    samples_per_file = collections.Counter()
    for task in sorted(c_root.iterdir()):
        evs_c = sorted(task.glob("*.eval"))
        per_task[len(evs_c)] += 1
        for e in evs_c:
            with zipfile.ZipFile(e) as z:
                samples_per_file[sum(n.startswith("samples/") and n.endswith(".json") for n in z.namelist())] += 1
    out["batch_c_tasks_by_eval_count"] = dict(per_task)
    out["batch_c_files_by_sample_count"] = dict(samples_per_file)
    return out


@app.local_entrypoint()
def main():
    res = probe.remote()
    p = HERE / "results" / "schema_probe_2026-10-14.json"
    p.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
