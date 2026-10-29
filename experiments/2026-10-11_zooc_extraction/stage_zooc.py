"""Stage a Phase 2 run folder of real Sol honest runs (zoo C) as inputs for the v4 Qwen3.5 extraction.

The v4 dump (`2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1`) read inputs staged by
experiments/2026-10-06_segment_end_extraction/run_extract_segment_end.py (`segment_end_v4.yaml`) from S3 cohorts with a
`logs/` folder. A Phase 2 run folder has one .eval per run under `<arm>/<run key>/`, so this adapter stages it the
same way, item by item:
  - which samples: the run list's rows, each read from the one file under `<arm>/` with the row's eval_file name (a
    redrawn run sits in `<run key>_redraw<n>/`), its sha256 checked; exactly one sample per .eval. Every other .eval
    under `<arm>/` is left unread, and must be exactly the ones given by `--expect-unlisted` (stage 4's pre-start
    attempt of oct11c_p169_r1); they are recorded with their sha256;
  - validity: `transcripts.invalidity` (errored, stalled, no_actions), excluded and counted, as v4;
  - messages: `transcripts.reader_messages(sample, system_prompt(sample))`, scrub level 0. v4 gave each run the honest
    system prompt of its main task, which for an honest run is its own;
  - attachments: samples are loaded through `transcripts.resolve_attachments` (on main since #760/#778), so a reference
    is resolved or the run stops; v4's loader had no such step (its texts had none, per the 10-11 census);
  - tokens: as `splits/make_splits.py` (Qwen3-32B at its pinned revision, chat template, thinking off), which only
    orders and deals transcripts to shards;
  - item fields: those of a v4 real-honest item, with arm `real-honest` and cohort `--cohort`, cell = main task,
    side_task and sim_reference None.
Inputs are written as `<run>/inputs/shard_<i>.jsonl` on the shared Volume `probe-shift_aswin-surya_sub-la-activations`,
which run_extract_qwen35.py reads through `inputs_from`. Counts and hashes only are printed.

  uv run --no-project --python 3.14 --with 'transformers==5.19.0' --with jinja2 --with pyyaml --with boto3 --with 'modal>=1.4.3' \
      python experiments/2026-10-11_zooc_extraction/stage_zooc.py --run-dir <local run folder> --run-list <run list json> \
      --run <staging run name> --cohort <cohort label> [--upload]
"""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from probe_shift.models import QWEN3_32B  # noqa: E402
from probe_shift.transcripts import invalidity, main_task_id, reader_messages, resolve_attachments, submit_calls, system_prompt  # noqa: E402

VOLUME = "probe-shift_aswin-surya_sub-la-activations"
MAX_TOKENS = 32768  # v4 staging's cut, used only for its `truncated` flag and the dealing cost, as there
TIME_FIT = (1.51e-4, 1.55e-8)  # run_extract_segment_end.py's dealing cost


def main(run_dir: str, run_list: str, run: str, cohort: str, arm_dir: str, shards: int, upload: bool, local_out: str,
         expect_unlisted: list[str]):
    from transformers import AutoTokenizer

    rd, rows = Path(run_dir), json.loads(Path(run_list).read_text())["rows"]
    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    items, excluded = [], Counter()
    found = {}
    for p in sorted((rd / arm_dir).glob("*/*.eval")):
        found.setdefault(p.name, []).append(p)
    listed = {row["eval_file"] for row in rows}
    unlisted = sorted(str(p.relative_to(rd / arm_dir)) for v in found.values() for p in v if p.name not in listed)
    if unlisted != sorted(expect_unlisted):
        raise ValueError(f"unlisted .eval files {unlisted} differ from --expect-unlisted {sorted(expect_unlisted)}")
    read, path_of = {}, {}
    for row in rows:
        if len(found.get(row["eval_file"], [])) != 1:
            raise ValueError(f"{row['key']}: {len(found.get(row['eval_file'], []))} files named as its eval_file")
        ev = found[row["eval_file"]][0]
        body = ev.read_bytes()
        read[str(ev.relative_to(rd / arm_dir))] = hashlib.sha256(body).hexdigest()
        path_of[row["key"]] = str(ev.relative_to(rd / arm_dir))
        if hashlib.sha256(body).hexdigest() != row["eval_sha256"]:
            raise ValueError(f"{row['key']}: eval sha256 differs from the run list")
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
            if len(names) != 1:
                raise ValueError(f"{row['key']}: {len(names)} samples; one is expected")
            s = resolve_attachments(json.loads(z.read(names[0])))
        if s["id"] != row["sample_id"]:
            raise ValueError(f"{row['key']}: sample id differs from the run list")
        reason = invalidity(s)
        if reason:
            excluded[reason] += 1
            continue
        messages = reader_messages(s, system_prompt(s))
        text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False)
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        md = s.get("metadata") or {}
        env = md.get("environment") or md.get("env")  # as splits/make_splits.py
        mt = main_task_id(s)
        items.append({"key": f"{ev.name.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}", "arm": "real-honest",
                      "cohort": cohort, "file": ev.name, "sample_id": s["id"], "epoch": s["epoch"],
                      "environment": env.get("id") if isinstance(env, dict) else env, "main_task": mt, "side_task": None, "cell": mt,
                      "sim_reference": None, "tokens": n, "truncated": n > MAX_TOKENS, "n_submit_calls": submit_calls(s),
                      "messages": messages, "run_key": row["key"]})
    if sorted((p.rsplit("/", 1)[1], h) for p, h in read.items()) != sorted((r["eval_file"], r["eval_sha256"]) for r in rows):
        raise ValueError("the files read differ from the run list's eval_files and sha256s")
    if len({it["key"] for it in items}) != len(items):
        raise ValueError("duplicate transcript keys")
    a, b = TIME_FIT
    cost = lambda it: a * min(it["tokens"], MAX_TOKENS) + b * min(it["tokens"], MAX_TOKENS) ** 2  # noqa: E731
    dealt, load = [[] for _ in range(shards)], [0.0] * shards
    for it in sorted(items, key=lambda it: (-cost(it), it["key"])):
        i = min(range(shards), key=load.__getitem__)
        dealt[i].append(it)
        load[i] += cost(it)
    files = {f"shard_{i}.jsonl": "".join(json.dumps(x) + "\n" for x in sh).encode() for i, sh in enumerate(dealt)}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    cfg = {"run": run, "stage_only": True, "adapter": "experiments/2026-10-11_zooc_extraction/stage_zooc.py", "git": git,
           "cohort": cohort, "arm": "real-honest", "source_run_dir": rd.name, "arm_dir": arm_dir, "run_list": Path(run_list).name,
           "tokenizer": QWEN3_32B, "scrub_level": 0, "max_tokens_for_flags": MAX_TOKENS,
           "normalization": "as segment_end_v4.yaml (own honest system prompt; reasoning dropped; submit calls dropped)"}
    out = {"written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "git": git, "staging_run": run,
           "volume": VOLUME, "runs_listed": len(rows), "eval_files_found": sum(map(len, found.values())),
           "read_equals_run_list": True,  # every listed eval_file read once, by name and sha256 (else stopped above)
           "not_read": [{"path": u, "sha256": hashlib.sha256((rd / arm_dir / u).read_bytes()).hexdigest(),
                         "reason": "not in the run list; named by --expect-unlisted"} for u in unlisted],
           "staged": len(items), "excluded": dict(excluded),
           "tokens": {"total": sum(it["tokens"] for it in items), "max": max(it["tokens"] for it in items),
                      "over_32k": sum(it["truncated"] for it in items)},
           "inputs_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()},
           "per_run": [{"run_key": it["run_key"], "eval_path": path_of[it["run_key"]]}
                       | {k: it[k] for k in ("key", "main_task", "environment", "tokens", "n_submit_calls")} for it in items]}
    if local_out:  # the same bytes as uploaded, for offline checks (gitignored data/ folder)
        lo = Path(local_out) / run / "inputs"
        lo.mkdir(parents=True, exist_ok=True)
        for n, b in files.items():
            (lo / n).write_bytes(b)
    if upload:
        import modal

        vol = modal.Volume.from_name(VOLUME, version=2)
        try:
            exists = bool(b"".join(vol.read_file(f"{run}/config.json")))  # read_file is lazy: a missing file raises while read
        except (FileNotFoundError, modal.exception.NotFoundError):
            exists = False
        if exists:
            raise SystemExit(f"{run} already exists on {VOLUME}; use a new run name")
        with vol.batch_upload() as batch:
            for n, b in files.items():
                batch.put_file(io.BytesIO(b), f"{run}/inputs/{n}")
            batch.put_file(io.BytesIO(json.dumps(cfg, indent=1).encode()), f"{run}/config.json")
        out["uploaded"] = True
    path = HERE / "results" / f"stage_{run}_{out['written']}.json"  # the summary; per-run IDs and tokens one line each beside it
    path.parent.mkdir(exist_ok=True)
    per_run = out.pop("per_run")
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    with path.with_name(path.stem + "_runs.jsonl").open("x") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in per_run))
    print(json.dumps(out, indent=1), "\n->", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-list", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--arm-dir", default="honest_b")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--local-out", default="")
    ap.add_argument("--expect-unlisted", nargs="*", default=[], help="paths under --arm-dir of .eval files the run list leaves out")
    main(**vars(ap.parse_args()))
