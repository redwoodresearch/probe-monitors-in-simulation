"""Did the same scorer grade Sol's real honest runs and the strict rerun's Qwen3.5-27B honest runs? (local, read-only,
$0; PREDICTIONS.md first). For every run on both sides, from its Inspect log: the runner's recorded revision, the
inspect_ai version, the scorer configuration, the arena reference contract (a sha256 of the environment's codebase
directory: runner/arenas/linuxarena.py hashes `codebase_path` only, so it does not cover the main-task scorer), and
sha256s of the main task's description and scorer description. Then, per main task present on both sides, whether
each field agrees. .eval files are zstd zips: Python 3.14.

  uv run --no-project --python 3.14 python experiments/2026-10-09_q2_scorer_check/analyze_scorer_versions.py \
      <Sol honest_success json> <Sol logs dir> <rerun dir>
"""
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sha = lambda text: hashlib.sha256((text or "").encode()).hexdigest()[:16]


def record(path: Path, sample_id: str | None = None) -> dict:
    """The version fields of one run (its sample; `sample_id` picks it when a log holds several)."""
    with zipfile.ZipFile(path) as z:
        e = json.loads(z.read("header.json"))["eval"]
        samples = [json.loads(z.read(n)) for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
    s = next(x for x in samples if sample_id is None or x["id"] == sample_id)
    m, c = s["metadata"]["main_task"], s["metadata"]["arena_reference_contract"]
    return {"main_task": m["id"], "environment": m["environment"], "sample_id": s["id"], "file": path.name,
            "revision": e.get("revision"), "inspect_ai": e["packages"].get("inspect_ai"),
            "scorers": json.dumps(e["scorers"], sort_keys=True), "contract_sha256": c["contract_sha256"], "contents_sha256": c["contents_sha256"],
            "description_sha": sha(m.get("description")), "scorer_description_sha": sha(m.get("scorer_description")),
            "main_task_success": s["scores"]["linux_scorer"]["value"]["main_task_success"]}


def main(sol_json: Path, sol_logs: Path, rerun: Path):
    sol_rows = json.loads(sol_json.read_text())["rows"]
    sides = {"sol": [record(sol_logs / r["file"], r["sample_id"]) for r in sol_rows],
             "qwen3.5": [record(f) for arm in ("honest_a", "honest_b") for f in sorted(rerun.glob(f"{arm}/*/*.eval"))]}
    by = {k: defaultdict(list) for k in sides}
    for k, rows in sides.items():
        for r in rows:
            by[k][r["main_task"]].append(r)
    fields = ("contents_sha256", "description_sha", "scorer_description_sha", "scorers", "inspect_ai")
    tasks = sorted(set(by["sol"]) & set(by["qwen3.5"]))
    per_task = {}
    for t in tasks:
        per_task[t] = {f: sorted({r[f] for r in by["sol"][t]}) == sorted({r[f] for r in by["qwen3.5"][t]}) for f in fields}
        per_task[t]["qwen3.5 scored"] = any(r["main_task_success"] in ("C", "I") for r in by["qwen3.5"][t])
    summary = {"main tasks on both sides": len(tasks),
               "revisions": {k: sorted({json.dumps(r["revision"]) for r in rows}) for k, rows in sides.items()},
               "inspect_ai": {k: sorted({r["inspect_ai"] for r in rows}) for k, rows in sides.items()},
               "scorer configs": {k: sorted({r["scorers"] for r in rows}) for k, rows in sides.items()},
               "codebase digests (contents_sha256) per environment": {k: {env: sorted({r["contents_sha256"][:12] for r in rows if r["environment"] == env})
                                                                          for env in sorted({r["environment"] for r in rows})} for k, rows in sides.items()},
               "tasks where every field agrees": sum(all(v[f] for f in fields) for v in per_task.values()),
               "tasks disagreeing, by field": {f: sorted(t for t, v in per_task.items() if not v[f]) for f in fields}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "results" / f"scorer_versions_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_task": per_task, "runs": sides}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(out)


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:4]))
