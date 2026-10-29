"""Add a cohort read by S3 prefix or local folder (`probe_shift.cohorts`, e.g. today's-simulator pilot) to an existing
staging as one more arm: the same rows and checks as run_stage_examples.py, rendered with each main task's honest
system prompt (taken from the extraction's reader inputs, where every transcript carries it) as the probe reads it.
Each of the cohort's cells must be in the base staging, with exactly one valid run. Writes a new
.run_cache/examples_<stamp>/ holding the base staging's rows plus the new arm's.

Cohorts are read with `probe_shift.cohorts` (#26). The 2026-10-06 stagings were made before #26 merged, from its branch's
src on PYTHONPATH (8655b5a); the functions used here are the same in #26, which only added manifest versions.
  uv run --with transformers==4.56.2 --with jinja2 python \
      experiments/2026-10-06_q1_text_baselines/run_stage_cohort.py config/stage_pilot_attack.yaml
"""

import gzip
import json
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

import probe_shift
from probe_shift.cohorts import eval_files, file_samples, root_of
from probe_shift.models import QWEN3_32B
from probe_shift.transcripts import invalidity, main_task_id, reader_messages
from run_stage_examples import raw_outputs, side_task_id, stage_run

HERE = Path(__file__).parent


def main():
    cfg = yaml.safe_load((HERE / sys.argv[1]).read_text())
    from transformers import AutoTokenizer

    assert cfg["tokenizer"] == QWEN3_32B, f"config tokenizer {cfg['tokenizer']} is not the shared pin {QWEN3_32B}"
    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    base = HERE / ".run_cache" / cfg["base"]
    base_runs = [json.loads(l) for l in (base / "runs.jsonl").read_text().splitlines()]
    base_cells = {r["cell"] for r in base_runs}
    assert cfg["arm"] not in {r["arm"] for r in base_runs}, f"{cfg['arm']} is already in {cfg['base']}"

    prompts = {}  # main task -> its honest system prompt, as the extraction rendered every arm
    shards = sorted((HERE / ".run_cache" / "reader_inputs" / cfg["honest_prompts_from"]).rglob("shard_*.jsonl"))
    assert shards, f"no reader inputs for {cfg['honest_prompts_from']} (run_stage_reader_inputs.py downloads them)"
    for f in shards:
        for t in map(json.loads, f.read_text().splitlines()):
            assert t["messages"][0]["role"] == "system"
            prompts.setdefault(t["main_task"], set()).add(t["messages"][0]["content"])
    assert all(len(p) == 1 for p in prompts.values()), "a main task has more than one system prompt"
    prompts = {mt: p.pop() for mt, p in prompts.items()}

    files = eval_files(cfg["cohort"])
    with ThreadPoolExecutor(cfg["threads"]) as ex:
        samples = [(f, s) for f, ss in zip(files, ex.map(lambda f: file_samples(cfg["cohort"], f), files)) for s in ss]
    chosen, dropped = {}, Counter()
    for f, s in samples:
        side = side_task_id(s)
        cell = main_task_id(s) if side is None else f"{main_task_id(s)}:{side}"  # honest cells are main tasks
        assert cell in base_cells, f"{cell} is not in {cfg['base']}"
        if reason := invalidity(s):
            dropped[reason] += 1
            continue
        assert cell not in chosen, f"two valid runs for cell {cell}"
        chosen[cell] = (f, s)

    rows, calls, runs = [], [], []
    for cell, (f, s) in sorted(chosen.items()):
        mt, env = main_task_id(s), (s.get("metadata") or {}).get("environment")
        meta = {"arm": cfg["arm"], "key": f"{f.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}", "file": f,
                "sample_id": s["id"], "epoch": s["epoch"], "environment": env.get("id") if isinstance(env, dict) else env,
                "main_task": mt, "cell": cell, "sim_reference": None}
        _, r, c, run = stage_run(tok, reader_messages(s, prompts[mt]), cfg["max_tokens"], meta, raw_outputs(s))
        rows += r
        calls += c
        runs.append(run)
    print(f"{cfg['arm']}: {len(runs)} runs, {len(rows)} outputs ({sum(r['kept'] for r in rows)} kept), {len(calls)} calls; "
          f"invalid: {dict(dropped)}", flush=True)

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / ".run_cache" / f"examples_{stamp}"
    out.mkdir(parents=True)
    for name, new in (("outputs.jsonl.gz", rows), ("calls.jsonl.gz", calls)):
        with gzip.open(out / name, "wt") as fh:
            with gzip.open(base / name, "rt") as old:
                shutil.copyfileobj(old, fh)
            fh.writelines(json.dumps(r) + "\n" for r in new)
    (out / "runs.jsonl").write_text((base / "runs.jsonl").read_text() + "".join(json.dumps(r) + "\n" for r in runs))
    src = Path(probe_shift.__file__).parent
    git = {p: subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=p).stdout.strip() for p in (HERE, src)}
    (out / "stage.json").write_text(json.dumps({"config": cfg, "cohort_root": root_of(cfg["cohort"]), "git": {str(k): v for k, v in git.items()},
                                                "probe_shift": str(src), "runs_added": len(runs), "invalid": dict(dropped)}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
