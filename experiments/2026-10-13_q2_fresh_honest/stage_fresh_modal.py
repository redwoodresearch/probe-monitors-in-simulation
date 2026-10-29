"""Fresh-task honest runs (PLAN.md here), stage C: stage one batch of new real Sol honest runs as extractor inputs, on Modal
CPU (no laptop compute). The staging is ../2026-10-11_zooc_extraction/stage_zooc.py's (cecc8fb3), item by item:
  - which samples: the run list's rows (list_fresh_runs.py), each read from the one .eval under <run folder>/honest_b/ with the
    row's eval_file name, its sha256 checked; one sample per file. Rows the run list counts as errored (errored_keys) are not
    staged and are counted (RULE.md leaves errored runs out). Every other .eval there must sit in a listed run's own folders
    (<run id>/ or the launcher's pre-start redraws <run id>_redraw<n>/: earlier attempts), counted; anything else stops;
  - validity: transcripts.invalidity, excluded and counted;
  - messages: transcripts.reader_messages(sample, system_prompt(sample)), scrub level 0, attachments resolved;
  - tokens: Qwen3-32B at its pinned revision, chat template, thinking off (dealing only);
  - item fields: a v4 real-honest item's, arm --arm, cohort --cohort, cell = main task, plus run_id.
Plus reproduction runs (stage_fxavg.py's way): --n-repro real-honest runs drawn with seed 20261014 from d9's mean-pool dump
2026-10-12_la_mean-pool36_qwen3-32b_l7_v1, copied from that dump's own inputs; only `arm` (repro-real-honest) and `tokens`
(-1..-n, so they sort last and the extractor deals one to each shard) change, the source count kept as source_tokens.
Inputs: the run folder and run list, put on the work Volume by transfer_run.sh. Output: <staged run>/inputs/shard_<i>.jsonl
(one arm per file) and stage_record.json on the activation Volume; a record (counts and hashes, no text) to results/.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-13_q2_fresh_honest/stage_fresh_modal.py --run-dir <run folder name> --run-list <run list file name> \\
      --staged <staged run name> --arm <arm> --cohort <label> --n-repro <shards>
"""
import hashlib
import io
import json
import random
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-fresh_stage_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q2-fresh_work_kill-2026-11-13"
D9 = ("2026-10-12_la_mean-pool36_qwen3-32b_l7_v1", "real-honest")
MAX_TOKENS, SEED = 32768, 20261014

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
work = modal.Volume.from_name(WORK, version=2, create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("transformers==5.19.0", "jinja2", "zstandard==0.25.0", "huggingface_hub", "boto3", "pyyaml")  # transcripts.py imports boto3 and yaml (no AWS call is made)
         .add_local_file(HERE / "evalzip.py", "/root/evalzip.py")
         .add_local_python_source("probe_shift"))


class Gate(Exception):
    """A failed check; its message holds counts only."""


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def msha(msgs):
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()


def run_job(run_dir: str, run_list: str, staged: str, arm: str, cohort: str, n_repro: int, repro_pool: list) -> dict:
    import evalzip  # noqa: F401  (zstd members for Python 3.12's zipfile)
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B
    from probe_shift.transcripts import invalidity, main_task_id, reader_messages, resolve_attachments, submit_calls, system_prompt

    rd, root = Path("/work") / run_dir, Path("/acts")
    rl = json.loads((Path("/work") / run_list).read_text())
    rows, errored = rl["rows"], set(rl["head"]["errored_keys"])
    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    found = {}
    for p in sorted((rd / "honest_b").glob("*/*.eval")):
        found.setdefault(p.name, []).append(p)
    listed = {r["eval_file"] for r in rows if r.get("eval_file")}
    ids = {r["run_id"] for r in rows}
    unlisted = sorted(str(p.relative_to(rd / "honest_b")) for v in found.values() for p in v if p.name not in listed)
    own = lambda d: d in ids or (d.rsplit("_redraw", 1)[0] in ids and d.rsplit("_redraw", 1)[-1].isdigit())  # noqa: E731
    if any(not own(u.split("/")[0]) for u in unlisted):
        raise Gate(f"{sum(not own(u.split('/')[0]) for u in unlisted)} unlisted .eval files outside a listed run's own folders")
    items, excluded, no_log, not_staged = [], Counter(), 0, []
    for row in rows:
        if row["run_id"] in errored:
            not_staged.append(row["run_id"])
            continue
        if not row.get("eval_file"):
            no_log += 1
            continue
        if len(found.get(row["eval_file"], [])) != 1:
            raise Gate("a run list row has no single file of its eval_file name")
        body = found[row["eval_file"]][0].read_bytes()
        if hashlib.sha256(body).hexdigest() != row["eval_sha256"]:
            raise Gate("an eval sha256 differs from the run list")
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
            if len(names) != 1:
                raise Gate("an .eval without exactly one sample")
            s = resolve_attachments(json.loads(z.read(names[0])))
        if s["id"] != row["sample_id"]:
            raise Gate("a sample id differs from the run list")
        reason = invalidity(s)
        if reason:
            excluded[reason] += 1
            continue
        messages = reader_messages(s, system_prompt(s))
        text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False)
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        md = s.get("metadata") or {}
        env = md.get("environment") or md.get("env")
        mt = main_task_id(s)
        if mt != row["task"]:
            raise Gate("a sample's main task differs from the run list's")
        ev = found[row["eval_file"]][0]
        items.append({"key": f"{ev.name.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}", "arm": arm,
                      "cohort": cohort, "file": ev.name, "sample_id": s["id"], "epoch": s["epoch"],
                      "environment": env.get("id") if isinstance(env, dict) else env, "main_task": mt, "side_task": None, "cell": mt,
                      "sim_reference": None, "tokens": n, "truncated": n > MAX_TOKENS, "n_submit_calls": submit_calls(s),
                      "messages": messages, "run_id": row["run_id"]})
    if len({it["key"] for it in items}) != len(items):
        raise Gate("duplicate transcript keys")

    repro, chosen = [], None
    if n_repro:
        dump, src_arm = D9
        cfg = json.loads((root / dump / "config.json").read_text())
        have = {p.stem for p in (root / dump / "meta" / src_arm).glob("*.json")}
        src = {}
        for p in sorted((root / cfg["inputs_from"] / "inputs").glob("shard_*.jsonl")):
            for line in p.read_text().splitlines():
                it = json.loads(line)
                if it["arm"] == src_arm and filename(it["key"]) in have and it["key"] in set(repro_pool):
                    src[filename(it["key"])] = it
        pick = random.Random(SEED).sample(sorted(src), n_repro)
        for i, s_ in enumerate(pick, 1):
            it = src[s_]
            repro.append(it | {"arm": "repro-real-honest", "tokens": -i, "source_tokens": it["tokens"]})
        chosen = {"dump": dump, "dump arm": src_arm, "inputs_from": cfg["inputs_from"], "pool": len(src),
                  "transcripts": sorted(it["key"] for it in repro)}
    if any(it["tokens"] < 0 for it in items):
        raise Gate("a staged run has a negative token count")

    dst = root / staged
    if dst.exists():
        raise FileExistsError(f"{staged} exists: a staged run is never overwritten")
    (dst / "inputs").mkdir(parents=True)
    shards, by_arm = {}, {}
    for i, (a, rs) in enumerate(sorted({x["arm"]: None for x in items + repro}.items())):
        rs = sorted((x for x in items + repro if x["arm"] == a), key=lambda x: x["key"])
        body = "".join(json.dumps(x) + "\n" for x in rs).encode()
        (dst / "inputs" / f"shard_{i}.jsonl").write_bytes(body)
        shards[f"inputs/shard_{i}.jsonl"] = {"arm": a, "runs": len(rs), "sha256": hashlib.sha256(body).hexdigest()}
        by_arm[a] = [x["key"] for x in rs]
    record = {"runs": [{"arm": x["arm"], "key": x["key"], "messages_sha256": msha(x["messages"])} for x in sorted(items + repro, key=lambda x: (x["arm"], x["key"]))],
              "shards": shards, "reproduction draw": chosen, "seed": SEED}
    rb = json.dumps(record).encode()
    (dst / "stage_record.json").write_bytes(rb)
    acts.commit()
    return {"mode": "stage", "staged": staged, "gates pass": True, "shards_files": shards, "runs by arm": by_arm,
            "record_sha256": hashlib.sha256(rb).hexdigest(), "reproduction draw": chosen, "run list rows": len(rows),
            "rows without a log": no_log, "errored rows, not staged": len(not_staged), "errored run ids, not staged": not_staged,
            "staged runs": len(items), "excluded": dict(excluded), "unlisted (earlier attempts)": unlisted,
            "tokens": {"total": sum(x["tokens"] for x in items), "max": max((x["tokens"] for x in items), default=0),
                       "over_32k": sum(x["truncated"] for x in items)},
            "per run": [{"run_id": x["run_id"], "main_task": x["main_task"], "environment": x["environment"], "tokens": x["tokens"],
                         "n_submit_calls": x["n_submit_calls"], "transcript": x["key"]} for x in items]}


@app.function(image=image, volumes={"/acts": acts, "/work": work.read_only()}, cpu=8, memory=32768, timeout=1800, retries=0)
def job(run_dir: str, run_list: str, staged: str, arm: str, cohort: str, n_repro: int, repro_pool: list) -> dict:
    try:
        return run_job(run_dir, run_list, staged, arm, cohort, n_repro, repro_pool)
    except Exception as e:  # noqa: BLE001 — type, location and our own count messages only, never staged text
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        msg = str(e) if isinstance(e, (Gate, FileExistsError)) else ""
        raise RuntimeError(f"stage failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}: {msg[:200]}") from None


@app.local_entrypoint()
def main(run_dir: str, run_list: str, staged: str, arm: str, cohort: str, n_repro: int = 0):
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__))], cwd=HERE).returncode:
        raise SystemExit("commit this script before running it")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    pool36 = [r["key"] for r in json.loads((HERE.parent / "2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json").read_text()) if r["arm"] == "real-honest"]
    out = job.remote(run_dir, run_list, staged, arm, cohort, n_repro, pool36)  # reproduction runs only from #36's 990 valid real honest runs
    out = {"git": git, "made": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"),
           "inputs": {"work volume": WORK, "run folder": run_dir, "run list": run_list}} | out
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"stage_{staged}_{out['made']}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("per run", "runs by arm", "shards_files")}, indent=1))
    print("->", dest)
