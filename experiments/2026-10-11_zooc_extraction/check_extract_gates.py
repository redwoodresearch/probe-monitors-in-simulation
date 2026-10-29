"""Gates for a zoo C Qwen3.5 extraction, from the Volume's small files only (config, stats, chunk indexes, metas; no
activations): every staged transcript has a meta and sits in a chunk; the staged input files still have the hashes the
staging recorded; no shard crashed and no chunk has non-finite values; rows per readout kind, per run. IDs and counts
only. Writes a gate file and exits non-zero if a gate fails.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' python experiments/2026-10-11_zooc_extraction/check_extract_gates.py \
      --run <extraction run> --staging <results/stage_<inputs run>_<stamp>.json>
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ACTS, INPUTS = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", "probe-shift_aswin-surya_sub-la-activations"


def read(vol, path):
    return b"".join(vol.read_file(path))


def provenance(run: str, launch_logs: str):
    """The launch lines that name the commit and the app (verbatim, IDs only) and the test's repeat reads, from the
    test run's metas: each transcript read twice, the largest absolute difference between the two reads."""
    lines = [line.strip() for f in launch_logs.split(",") for line in Path(f).read_text().splitlines()
             if any(w in line for w in ("spawned", "modal.com/apps/", "done in", '"calls"', '"call"'))]  # modal run logs or spawn_extract.py records
    acts = modal.Volume.from_name(ACTS, version=2)
    test = [json.loads(read(acts, e.path)) for e in acts.listdir(f"{run}_test/meta", recursive=True) if e.path.endswith(".json")]
    keep = ("key", "n_tokens", "n_rows", "forward_s", "forward_s_repeat", "repeat_max_abs_diff")
    out = {"written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "run": run, "launch_log_lines": lines,
           "test_reads": sorted(({k: m.get(k) for k in keep} for m in test), key=lambda r: r["key"]),
           "test_max_repeat_difference": max(m["repeat_max_abs_diff"] for m in test)}
    path = HERE / "results" / f"provenance_{run}_{out['written']}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("launch_log_lines", "test_max_repeat_difference")}, indent=1), "\n->", path)


def main(run: str, staging: str, launch_logs: str = ""):
    if launch_logs:
        return provenance(run, launch_logs)
    st = json.loads(Path(staging).read_text())
    runs_file = Path(staging).with_name(Path(staging).stem + "_runs.jsonl")
    staged = {json.loads(line)["key"]: json.loads(line)["run_key"] for line in runs_file.read_text().splitlines()}
    acts, inp = modal.Volume.from_name(ACTS, version=2), modal.Volume.from_name(INPUTS, version=2)
    cfg = json.loads(read(acts, f"{run}/config.json"))
    inputs_sha = {e.path.rsplit("/", 1)[1]: hashlib.sha256(read(inp, e.path)).hexdigest() for e in inp.listdir(f"{cfg['inputs_from']}/inputs")}
    stats = [json.loads(read(acts, e.path)) for e in acts.listdir(f"{run}/stats")]
    metas = [json.loads(read(acts, e.path)) for e in acts.listdir(f"{run}/meta", recursive=True) if e.path.endswith(".json")]
    chunked = Counter(t["key"] for e in acts.listdir(f"{run}/chunks") if e.path.endswith(".json")
                      for t in json.loads(read(acts, e.path))["transcripts"])
    per_run, kinds = [], Counter()
    for m in metas:
        k = Counter(p["kind"] for p in m["positions"] if p["kept"])
        kinds.update(k)
        per_run.append({"run_key": staged.get(m["key"]), "key": m["key"], "n_tokens": m["n_tokens"], "rows": m["n_rows"],
                        "call_end": k["call_end"], "output_end": k["output_end"]})
    gates = {
        "staged inputs unchanged (sha256 as recorded by staging)": inputs_sha == st["inputs_sha256"],
        "config reads the staged inputs": cfg["inputs_from"] == st["staging_run"],
        f"every staged transcript has a meta ({len(staged)})": sorted(m["key"] for m in metas) == sorted(staged),
        "every meta is in exactly one chunk": set(chunked) == set(staged) and set(chunked.values()) == {1},
        f"stats for every shard ({cfg['shards']})": sorted(s["shard"] for s in stats) == list(range(cfg["shards"])),
        "no non-finite values": all(c["nonfinite"] == 0 for s in stats for c in s["chunks"]),
        "rows in chunks = rows in metas": sum(c["rows"] for s in stats for c in s["chunks"]) == sum(m["n_rows"] for m in metas),
    }
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out = {"written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "git": git, "run": run,
           "volume": ACTS, "inputs_from": cfg["inputs_from"], "config": cfg, "gates": gates, "transcripts": len(metas),
           "rows": sum(m["n_rows"] for m in metas), "kept_positions_by_kind": dict(kinds),
           "tokens": sum(m["n_tokens"] for m in metas), "shards": stats, "per_run": sorted(per_run, key=lambda r: r["key"])}
    path = HERE / "results" / f"gates_{run}_{out['written']}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("run", "transcripts", "rows", "tokens", "kept_positions_by_kind", "gates")}, indent=1), "\n->", path)
    if not all(gates.values()):
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--staging", default="")
    ap.add_argument("--launch-logs", default="", help="comma-separated launch logs: record provenance instead of the gates")
    main(**vars(ap.parse_args()))
