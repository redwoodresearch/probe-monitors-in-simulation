"""Header check before the reroll yardstick ($0, laptop, S3 reads; 0d's GO relayed after 22:33 PT 10-11): do the
independent reroll (S2) and claim B's first sim (S1) share the simulator settings, the reference and the start
condition? 20 seeded cells (random.Random(20261011)) of the 97 test-environment overlap cells. For each cell the
S2 log (staged corpus) and S1's sample (claim B's `la_public10_sim_astra_attack_n1000`, the file named in
splits/exclusions_v3.json) are downloaded to a temporary folder (deleted at exit) and compared on metadata only:
scalars are printed, every text field is compared by sha256 and never printed.

Decision rule, fixed before running. Comparability BREAKS (stop and report, no reads bought) if, in any checked
cell: the simulator model, its fallback, or its refusal/router judge model differ; codebase_exec or
exec_source_authority differ; the reasoning effort of any model both arms
called differs; orig_trajectory_mode or the output cap differ; self-critique or session notes differ; or
the start differs (fork `after`/`prefix_actions`/`fraction`, or a non-empty seeded prefix in one arm only).
Fork `of_actions`/`unit` (the reference's length and unit) are reported, not breaking. Missing evidence (a scalar,
fork field or model event absent, no model both arms called) or an S2 log whose reroll_provenance does not point to
the S1 sample stops the script with an error instead of passing.
Reported, NOT breaking: a different reference (orig_trajectory sha256: overlap800 references are fresh rollouts per
job), a different attack-approach text, a different simulator source commit, different eval config, a model
called in one arm only (e.g. the fallback).

  uv run --no-project --python 3.14 python -I experiments/2026-10-11_claimB_yardsticks/check_reroll_headers.py \
      <repo root> <sce root>      (AWS_PROFILE=redwood-sandbox from the laptop)
"""

import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO, SCE = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
HERE = REPO / "experiments/2026-10-11_claimB_yardsticks"
BUCKET = "rr-sce-sim-transcripts"
S1_PREFIX = "transcripts/2026-09-18_la-public10_sim-astra-medium-attack-origtraj-read-nocritic-fork0-overlapaware-n1000/logs/"
N_CHECK, SEED = 20, 20261011
SCALARS = ["simulator_model", "simulator_fallback_model", "simulator_refusal_judge_model", "simulator_router_judge_model",
           "session_notes", "self_critique", "orig_trajectory_mode", "orig_trajectory_out_cap", "codebase_exec",
           "exec_source_authority"]
TEXTS = ["orig_trajectory", "attack_approach", "main_task_str", "side_task_text", "env_context"]
FORK_START = ["after", "prefix_actions", "fraction"]
FORK_INFO = ["of_actions", "unit"]
BREAKING = set(SCALARS) | {"efforts", "fork_start", "seeded_prefix_empty"}


def h(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fetch(key: str, dst: Path) -> Path:
    subprocess.run(["aws", "s3", "cp", "--quiet", f"s3://{BUCKET}/{key}", str(dst)], check=True)
    return dst


def sample_of(path: Path, sample_id: str | None) -> tuple[dict, dict]:
    with zipfile.ZipFile(path) as z:
        header = json.loads(z.read("header.json"))
        members = [n for n in z.namelist() if n.startswith("samples/")]
        found = []
        for n in members:
            d = json.loads(z.read(n))
            if sample_id is None or d.get("id") == sample_id:
                found.append(d)
        if len(found) != 1:
            raise ValueError(f"{path.name}: {len(found)} samples match {sample_id}")
        return header, found[0]


def efforts(sample: dict) -> dict:
    out = {}
    for ev in sample.get("events", []):
        if ev.get("event") == "model":
            out.setdefault(str(ev.get("model")), set()).add(str((ev.get("config") or {}).get("reasoning_effort")))
    return {k: sorted(v) for k, v in sorted(out.items())}


def facts(header: dict, sample: dict) -> dict:
    md = sample.get("metadata")
    if not md or any(k not in md for k in SCALARS + TEXTS + ["fork", "seeded_prefix"]):
        raise ValueError(f"{sample.get('id')}: required metadata missing")
    fork = md["fork"] or {}
    if any(k not in fork for k in FORK_START + FORK_INFO):
        raise ValueError(f"{sample.get('id')}: fork fields missing")
    eff = efforts(sample)
    if not eff:
        raise ValueError(f"{sample.get('id')}: no model events")
    return {"scalars": {k: md.get(k) for k in SCALARS},
            "texts": {k: h(md.get(k)) for k in TEXTS},
            "fork_start": {k: fork.get(k) for k in FORK_START},
            "fork_info": {k: fork.get(k) for k in FORK_INFO},
            "fork_source_sample": fork.get("source_sample"),
            "seeded_prefix_len": len(md.get("seeded_prefix") or []),
            "source_commit": (md.get("source_tree") or {}).get("commit"),
            "efforts": eff,
            "reroll_provenance": {k: v for k, v in (md.get("reroll_provenance") or {}).items() if k != "initial_state_sha256"},
            "config": header.get("eval", {}).get("config", {})}


def main():
    staged = json.loads((SCE / "experiments/2026-09-29_monitor_score_variance/staged_rerolls.json").read_text())
    excl = json.loads((REPO / "splits/exclusions_v3.json").read_text())
    test = set(json.loads((REPO / "experiments/2026-10-09_fixed_sim/split.json").read_text())["test"])
    paired = set(excl["paired_attack_cells"]["cells"])
    sims = [s for s in excl["samples"] if s["arm"] == "sim-attack"]
    s1 = {s["cell"]: s for s in sims}
    if len(s1) != len(sims) or len({r["cell"] for r in staged}) != len(staged):
        raise ValueError("duplicate cells in S1 or in the staged rerolls")
    rows = []
    for r in staged:
        env = re.search(r"_la-([a-z-]+?)-sim-reroll-\d+_", Path(r["source"]).name).group(1).replace("-", "_")
        if r["cell"] in paired and env in test:
            rows.append(r)
    if len(rows) != 97:
        raise ValueError(f"expected 97 test overlap cells, got {len(rows)}")
    pick = random.Random(SEED).sample(sorted(rows, key=lambda r: r["cell"]), N_CHECK)
    per_cell, agree = [], Counter()
    with tempfile.TemporaryDirectory() as tmp:
        for r in pick:
            a = s1[r["cell"]]
            h1, smp1 = sample_of(fetch(S1_PREFIX + a["file"], Path(tmp) / "s1.eval"), a["sample_id"])
            h2, smp2 = sample_of(fetch(r["destination"], Path(tmp) / "s2.eval"), None)
            f1, f2 = facts(h1, smp1), facts(h2, smp2)
            shared = set(f1["efforts"]) & set(f2["efforts"])
            if not shared:
                raise ValueError(f"{r['cell']}: no model called in both arms")
            prov = f2["reroll_provenance"]
            if not (prov.get("original_sample_id") == a["sample_id"] and str(prov.get("original_file")).endswith(a["file"])):
                raise ValueError(f"{r['cell']}: S2's reroll_provenance does not point to the S1 sample")
            same = {**{k: f1["scalars"][k] == f2["scalars"][k] for k in SCALARS},
                    **{f"text:{k}": f1["texts"][k] == f2["texts"][k] for k in TEXTS},
                    "efforts": all(f1["efforts"][m] == f2["efforts"][m] for m in shared),
                    "fork_info": f1["fork_info"] == f2["fork_info"],
                    "models_used_same": set(f1["efforts"]) == set(f2["efforts"]),
                    "fork_start": f1["fork_start"] == f2["fork_start"],
                    "seeded_prefix_empty": (f1["seeded_prefix_len"] == 0) == (f2["seeded_prefix_len"] == 0),
                    "source_commit": f1["source_commit"] == f2["source_commit"], "config": f1["config"] == f2["config"]}
            agree.update(k for k, v in same.items() if v)
            per_cell.append({"cell": r["cell"], "s1_reference_tag": a.get("sim_reference"), "same": same,
                             "s1": {k: f1[k] for k in ("scalars", "fork_start", "fork_info", "seeded_prefix_len", "source_commit", "efforts")},
                             "s2": {k: f2[k] for k in ("scalars", "fork_start", "fork_info", "seeded_prefix_len", "source_commit", "efforts")},
                             "config_diff": sorted(k for k in set(f1["config"]) | set(f2["config"])
                                                   if f1["config"].get(k) != f2["config"].get(k))})
    keys = sorted({k for c in per_cell for k in c["same"]})
    breaks = sorted({k for c in per_cell for k, v in c["same"].items() if not v and k in BREAKING})
    out = {"label": "reroll header check (metadata only; texts by sha256)", "cells_checked": N_CHECK, "seed": SEED,
           "agree_counts": {k: agree[k] for k in keys}, "breaking_fields_with_any_mismatch": breaks,
           "verdict": "BREAKS comparability" if breaks else "comparable on the breaking fields",
           "distinct": {"s1_efforts": sorted({json.dumps(c["s1"]["efforts"]) for c in per_cell}),
                        "s2_efforts": sorted({json.dumps(c["s2"]["efforts"]) for c in per_cell}),
                        "s1_fork_start": sorted({json.dumps(c["s1"]["fork_start"], sort_keys=True) for c in per_cell}),
                        "s2_fork_start": sorted({json.dumps(c["s2"]["fork_start"], sort_keys=True) for c in per_cell}),
                        "s1_commits": sorted({str(c["s1"]["source_commit"]) for c in per_cell}),
                        "s2_commits": sorted({str(c["s2"]["source_commit"]) for c in per_cell}),
                        "config_diff_keys": sorted({k for c in per_cell for k in c["config_diff"]})},
           "per_cell": per_cell,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"reroll_headers_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("agree_counts", "breaking_fields_with_any_mismatch", "verdict", "distinct")}, indent=1, default=str))
    print("->", dst)


if __name__ == "__main__":
    main()
