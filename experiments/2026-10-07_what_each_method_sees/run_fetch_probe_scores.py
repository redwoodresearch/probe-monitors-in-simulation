"""Fetch the probe's headline-layer step scores (output_end) for levels 0 and 7 from the S3 archive, nothing else.

Reads each analysis summary and the archive manifest from git (probe_results_commit), takes the layers chosen in any
outer fold (both splits, both comparisons, all outputs), lists the exact keys and their total size, refuses above the
cap, then copies key by key (never a prefix) and checks each file's sha256 against the manifest. Files already present
with the right hash are skipped.

  uv run python experiments/2026-10-07_what_each_method_sees/run_fetch_probe_scores.py [--cap-mb 300]
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
SEG = "experiments/2026-10-06_q1_segment_end"


def git_show(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


def headline_layers(summary: dict) -> set[int]:
    return {l for head in ("headline", "headline_held_out_environment") for comp in summary[head]["output_end"].values()
            for l in comp["all_outputs"]["chosen_layer_by_fold"].values()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config/what_sees.yaml"))
    ap.add_argument("--cap-mb", type=float, default=300)
    ap.add_argument("--profile", default="redwood-sandbox")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    commit = cfg["probe_results_commit"]
    manifest = {r["path"]: r for r in map(json.loads, git_show(commit, f"{SEG}/s3_archive_manifest.jsonl").decode().splitlines())}
    cache = HERE / ".run_cache"
    wanted = []
    for level, p in cfg["probe"].items():
        name = f"q1_gpu_{p['analysis']}.json"
        summary_path = cache / "q1_results" / name
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_bytes(git_show(commit, f"{SEG}/results/{name}"))
        summary = json.loads(summary_path.read_text())
        run = summary["config"]["extraction_run"]
        for l in sorted(headline_layers(summary)):
            wanted.append(manifest[f"{run}__q1gpu/{p['analysis']}/L{l}_step_scores_output_end.npz"] | {"analysis": p["analysis"]})
    total = sum(r["size"] for r in wanted)
    for r in wanted:
        print(r["size"], r["uri"])
    print(f"{len(wanted)} files, {total / 1e6:.1f} MB")
    assert total <= args.cap_mb * 1e6, f"{total / 1e6:.1f} MB is over the {args.cap_mb} MB cap"
    for r in wanted:
        dest = cache / "probe_step_scores" / r["analysis"] / Path(r["uri"]).name
        if not (dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == r["sha256"]):
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["aws", "s3", "cp", "--quiet", "--profile", args.profile, r["uri"], str(dest)], check=True)
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == r["sha256"], f"sha256 mismatch: {dest}"
    print("all files present, sha256 match")


if __name__ == "__main__":
    main()
