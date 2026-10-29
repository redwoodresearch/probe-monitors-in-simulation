"""Step 1 (PLAN.md here), POST HOC: behaviour alone. Stage every run of both worlds with EVERY tool reply replaced by one
fixed placeholder, PLACEHOLDER (5 Qwen3-32B tokens); system, user and assistant messages and tool-call arguments stay
byte-identical, and a tool message's other fields too. Modal CPU; writes no message text locally (counts and hashes).

Sources (staged inputs on the Volume, read as the fix-full stager does): real-attack-e1, real-attack-e2, real-honest and
sim-attack from #36's segment-end run (#36's runs.json keys: 939, 939, 990, 939), sim-honest from the v4 run (437 keys).
Arms are the source arm + "-ph"; keys, cells and every other field of a staged row stay the source's.
Gates (any failure stops before anything is written): every source run found once; the same number of messages; every
non-tool message equal to the source's; every tool message's fields other than content equal; every tool content equal to
PLACEHOLDER; no "attachment://" anywhere in a staged row; at least one tool message per run.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run \\
      experiments/2026-10-13_q2_step1_placeholder/stage_placeholder.py --mode survey|stage
"""
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
APP = "probe-shift_aswin-surya_sub-q2-step1ph_stage_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
BASE = "2026-10-06_la_segment-end_qwen3-32b_v1"  # #36's staged inputs: real-honest, real-attack-e1/e2, sim-attack
HONEST = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"  # sim-honest (the fix-full stager's source)
DST = "2026-10-13_q2_step1_placeholder_inputs_v1"
PLACEHOLDER = "[tool output withheld]"
RUNS36 = REPO / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
HONEST_RUNS = REPO / "experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json"
EXPECT = {"real-attack-e1": 939, "real-attack-e2": 939, "real-honest": 990, "sim-attack": 939, "sim-honest": 437}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12")


def run_job(mode: str, keys: dict) -> dict:
    def load(run_dir, want):
        out = {}
        for p in sorted((Path("/acts") / run_dir / "inputs").glob("shard_*.jsonl")):
            for line in p.read_text().splitlines():
                it = json.loads(line)
                if it["key"] in want.get(it["arm"], ()):
                    if (it["arm"], it["key"]) in out:
                        raise ValueError("a run appears twice in the source inputs")
                    out[(it["arm"], it["key"])] = it
        return out

    want = {a: set(ks) for a, ks in keys.items()}
    src = load(BASE, {a: v for a, v in want.items() if a != "sim-honest"}) | load(HONEST, {"sim-honest": want["sim-honest"]})
    found = Counter(a for a, _ in src)
    if dict(found) != {a: len(v) for a, v in want.items()}:
        raise ValueError(f"source runs found {dict(found)}, wanted {({a: len(v) for a, v in want.items()})}")
    c, rows, record = Counter(), {}, []
    for (arm, k), it in sorted(src.items()):
        m = it["messages"]
        new = [x | {"content": PLACEHOLDER} if x["role"] == "tool" else x for x in m]
        tools = [i for i, x in enumerate(m) if x["role"] == "tool"]
        if not tools:
            raise ValueError("a run without a tool message")
        if len(new) != len(m) or any(new[i] != m[i] for i in range(len(m)) if m[i]["role"] != "tool") \
                or any({f: v for f, v in new[i].items() if f != "content"} != {f: v for f, v in m[i].items() if f != "content"} for i in tools) \
                or any(new[i]["content"] != PLACEHOLDER for i in tools):
            raise ValueError("gate: a non-tool message, a tool message's other field, or a placeholder differs")
        row = it | {"arm": f"{arm}-ph", "messages": new}
        if "attachment://" in json.dumps(row):
            raise ValueError("gate: attachment:// in a staged row")
        c[f"{arm}|runs"] += 1
        c[f"{arm}|tool messages"] += len(tools)
        c[f"{arm}|tool messages whose content was not a string"] += sum(not isinstance(m[i]["content"], str) for i in tools)
        c[f"{arm}|tool messages already equal to the placeholder"] += sum(m[i]["content"] == PLACEHOLDER for i in tools)
        c[f"{arm}|source characters in tool messages"] += sum(len(json.dumps(m[i]["content"])) for i in tools)
        rows.setdefault(row["arm"], []).append(row)
        record.append({"arm": row["arm"], "key": k, "tool_messages": len(tools),
                       "messages_sha256": hashlib.sha256(json.dumps(new, sort_keys=True).encode()).hexdigest(),
                       "source_messages_sha256": hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()})
    out = {"placeholder": PLACEHOLDER, "counts": dict(sorted(c.items())), "runs by arm": {a: len(r) for a, r in sorted(rows.items())}}
    if mode == "stage":
        dst = Path("/acts") / DST
        if dst.exists():
            raise FileExistsError(f"{DST} exists: a staged run is never overwritten")
        (dst / "inputs").mkdir(parents=True)
        shas = {}
        for n, (arm, rs) in enumerate(sorted(rows.items())):
            body = "".join(json.dumps(x) + "\n" for x in sorted(rs, key=lambda x: x["key"])).encode()
            (dst / "inputs" / f"shard_{n}.jsonl").write_bytes(body)
            shas[f"inputs/shard_{n}.jsonl"] = {"arm": arm, "runs": len(rs), "sha256": hashlib.sha256(body).hexdigest()}
        rec = json.dumps({"placeholder": PLACEHOLDER, "runs": record, "shards": shas}).encode()
        (dst / "stage_record.json").write_bytes(rec)
        acts.commit()
        out |= {"staged": DST, "shards": shas, "record_sha256": hashlib.sha256(rec).hexdigest()}
    return out


@app.function(image=image, volumes={"/acts": acts}, cpu=8, memory=65536, timeout=3600, retries=0)
def job(mode: str, keys: dict) -> dict:
    try:
        return run_job(mode, keys)
    except Exception as e:  # noqa: BLE001 — sanitized (no staged text crosses the boundary) and re-raised
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"stage failed: {type(e).__name__}: {str(e)[:200]} at {Path(tb.filename).name}:{tb.lineno}") from None


@app.local_entrypoint()
def main(mode: str = "survey"):
    if mode not in ("survey", "stage"):
        raise SystemExit("--mode survey|stage")
    for f in (Path(__file__), HERE / "PLAN.md"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    keys = {}
    for r in json.loads(RUNS36.read_text()):
        keys.setdefault(r["arm"], []).append(r["key"])
    keys["sim-honest"] = [r["key"] for r in json.loads(HONEST_RUNS.read_text()) if r["arm"] == "sim-honest"]
    if {a: len(set(v)) for a, v in keys.items()} != EXPECT:
        raise SystemExit(f"cohort sizes {({a: len(set(v)) for a, v in keys.items()})} differ from {EXPECT}")
    out = job.remote(mode, {a: sorted(set(v)) for a, v in keys.items()})
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    p = HERE / "results" / f"{mode}_{stamp}.json"
    p.write_text(json.dumps({"POST HOC": True, "git": head, "mode": mode, "inputs": {"#36 runs.json": hashlib.sha256(RUNS36.read_bytes()).hexdigest(),
                             "honest runs.json": hashlib.sha256(HONEST_RUNS.read_bytes()).hexdigest()}} | out, indent=1) + "\n")
    print(p)
    for k, v in out["counts"].items():
        print(f"  {k}: {v}")
