"""The sim-trained monitor's real-vs-sim catch, by the real-trained headline's method ($0, stored files only).

Acting set-up (Qwen did the 196 tasks): onpolicy_2x2_scaleup196_draws0-500_2026-10-10_163614.json (S3; sha256 listed in
experiments/2026-10-08_q3_onpolicy_figures/results/s3_manifest_scaleup196_v1.json). value / diff / endpoints are
imported unchanged from analyze_scaleup196.py (4a8ed021), the script behind the real-trained headline
(+0.46 [-2.32, +2.50]): paired over refit draws 1..500 (threshold recomputed in each draw), failed draws at the
floor/ceiling, 2.5th/97.5th percentiles; draw 0 is the observed value. The real-trained gap is recomputed here as a
check that the method matches. Per-cell ranges use floor 0 / ceiling 1; the headline script reports none.

Reading set-up (off-policy): refit_2026-10-08_232032.json, q2's 500-replicate refit bootstrap (S3; sha256 listed in
experiments/2026-10-09_q2_tpr_at_1pct/results/manifest_v2.json), read by key readers.qwen3.5.

  uv run --no-project --with numpy python -I analyze_simtrained_range.py \
      ../2026-10-08_q3_onpolicy_figures/analyze_scaleup196.py <draws0-500 json> <refit json> <out json>
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SHA = {"draws": "b8ddf7c20aecb210b90cb67544dcd7f9cdbae6ca9e9ead8f758fe405f895d18d",
       "refit": "31f88cd5da9594a18235ca43abb5088360f8fb082ed8bef2a398b2672cd02cfb"}
GAP = "domain gap, trained on sim (sim->real minus sim->sim)"
METRICS = ("catch_frozen", "false_alarm_frozen", "catch", "false_alarm", "auroc")


def load(path, key):
    raw = Path(path).read_bytes()
    if (got := hashlib.sha256(raw).hexdigest()) != SHA[key]:
        raise SystemExit(f"{path}: sha256 {got} is not the manifest's {SHA[key]}")
    return json.loads(raw)


def main(module_path, draws_path, refit_path, out_path):
    spec = importlib.util.spec_from_file_location("scaleup196", module_path)
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    D, R = load(draws_path, "draws"), load(refit_path, "refit")

    draws = {d["draw"]: d for d in D["draws"]}
    if len(draws) != len(D["draws"]) or sorted(draws) != list(range(A.FROZEN_DRAWS + 1)):
        raise SystemExit(f"the draw file must hold draws 0..{A.FROZEN_DRAWS} exactly once")
    obs, boots = draws[0], [draws[b] for b in range(1, A.FROZEN_DRAWS + 1)]
    cells = sorted(obs["cells"])
    if "sim->run2" in cells:
        raise SystemExit("the file has a sim->run2 cell: compute the sim-trained yardstick before reporting")

    git = subprocess.run(["git", "-C", str(Path(module_path).parent), "rev-parse", "HEAD"], capture_output=True, text=True)
    act = {"source": Path(draws_path).name, "sha256": SHA["draws"], "cells": cells, "git_head": git.stdout.strip(),
           "method": "analyze_scaleup196.py value/diff/endpoints; paired over draws 1..500; observed = draw 0",
           "sim-trained real-vs-real yardstick": "not available: the file has no sim->run2 cell"}
    for cell in ("sim->real", "sim->sim", "real->real", "real->sim"):
        act[cell] = {m: {"observed": A.value(obs, cell, m), **A.endpoints([A.value(d, cell, m) for d in boots], 0.0, 1.0)}
                     for m in METRICS}
    pairs = {"sim-trained real - sim (sim->real minus sim->sim)": ("sim->real", "sim->sim"),
             "real-trained real - sim (real->real minus real->sim), reproduces the headline": ("real->real", "real->sim")}
    for name, (a, b) in pairs.items():
        act[name] = {m: {"observed": A.diff(obs, a, b, m), **A.endpoints([A.diff(d, a, b, m) for d in boots])}
                     for m in METRICS}

    q = R["readers"]["qwen3.5"]
    read = {"source": Path(refit_path).name + " readers.qwen3.5", "sha256": SHA["refit"], "config": R["config"],
            "replicates": q["replicates"], GAP: q[GAP], "sim->real": q["sim->real"], "sim->sim": q["sim->sim"]}
    # one entry per line, so the file stays a few dozen lines
    blocks = [f' {json.dumps(k)}: {{\n' + ",\n".join(f'  {json.dumps(n)}: {json.dumps(v)}' for n, v in d.items()) + "\n }"
              for k, d in (("acting set-up", act), ("reading set-up", read))]
    Path(out_path).write_text("{\n" + ",\n".join(blocks) + "\n}\n")

    def pts(e, s="+"):
        return f"{100 * e['observed']:{s}.2f} [{100 * e['ci95_exact'][0]:{s}.2f}, {100 * e['ci95_exact'][1]:{s}.2f}]" \
               f" failed {e['failed']}/{e['draws']}"

    print("cells:", ", ".join(cells), "| yardstick:", act["sim-trained real-vs-real yardstick"])
    for name in pairs:
        for m in METRICS:
            print(f"{name[:38]:38s} {m:20s} {pts(act[name][m])}")
    for c in ("sim->real", "sim->sim"):
        for m in METRICS[:4]:
            print(f"{c:10s} {m:20s} {pts(act[c][m], '')}")
    for m in ("catch, frozen training-world threshold", "catch, calibrated in test world",
              "false-alarm rate, frozen", "false-alarm rate, calibrated in test world"):
        print("reading", m, q[GAP][m]["value"], q[GAP][m]["ci95"])


if __name__ == "__main__":
    main(*sys.argv[1:5])
