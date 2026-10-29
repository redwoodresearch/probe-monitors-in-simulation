"""The task and reference list for one stage of q2's lean230 simulated honest runs (PLAN.md), keys and hashes only.

Tasks: positions [--start, --end) (1-based, end exclusive) of "task order" in q2's batch spec
(experiments/2026-10-11_q2_new_real_runs/results/batch_spec_2026-10-11_151207.json @ 1aff2647), checked against its
recorded order sha256 (sha256 of the newline-joined order). References: each task's n437 reference, the one its
2026-10-06 simulated run got: the entry in the n437 selection file (experiments/2026-10-06_astra_sim_honest/results/
references_20261006-142754.json, sha256 checked against the n437 manifest), cross-checked against the n437 S3
manifest's per-task reference file and sha256. The reference .eval files are copied from the n437 prefix's
references/ into --refs-dir (git-ignored data/) and their sha256 checked. Also records each task's n437 cost, used by
the stage driver's stop projection. Writes results/lean230_refs_<stage>_<stamp>.json in the launcher's --references
format ({"references": {main: entry}}) plus provenance.

  uv run --no-project --python 3.12 python experiments/2026-10-11_q2_sim_honest_lean230/make_stage_refs.py \\
      --start 1 --end 51 --stage stage1 --refs-dir data/lean230_refs
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SPEC = ("1aff2647", "experiments/2026-10-11_q2_new_real_runs/results/batch_spec_2026-10-11_151207.json")
SELECTION = "experiments/2026-10-06_astra_sim_honest/results/references_20261006-142754.json"
N437 = "s3://rr-sce-sim-transcripts/transcripts/2026-10-06_la-public10_sim-astra-medium-honest-origtraj-read-nocritic-fork0/"


def git_show(rev, path):
    return subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, check=True, cwd=HERE).stdout


def aws(*a):
    return subprocess.run(["aws", "--profile", "redwood-sandbox", *a], capture_output=True, check=True).stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--refs-dir", type=Path, required=True)
    args = p.parse_args()
    spec_raw = git_show(*SPEC)
    spec = json.loads(spec_raw)
    order = spec["task order"]
    assert hashlib.sha256("\n".join(order).encode()).hexdigest() == spec["task order sha256"], "task order hash"
    mains = order[args.start - 1:args.end - 1]
    assert len(mains) == args.end - args.start and len(set(mains)) == len(mains)
    sel_raw = git_show("origin/main", SELECTION)
    sel = json.loads(sel_raw)["references"]
    with tempfile.TemporaryDirectory() as tmp:
        m = Path(tmp) / "manifest_v2.json"
        aws("s3", "cp", "--quiet", N437 + "manifest_v2.json", str(m))
        man_raw = m.read_bytes()
    man = json.loads(man_raw)
    assert man["reference_selection"]["sha256"] == hashlib.sha256(sel_raw).hexdigest(), "selection file differs from n437's"
    tasks = {t["main"]: t for t in man["tasks"]}
    refs, rows = {}, []
    for pos, main in enumerate(mains, start=args.start):
        t, s = tasks[main], sel[main]
        assert t["outcome"] == "valid", f"{main}: n437 run not valid"
        rel = f"{s['environment']}/{s['fork_run']}/{s['source_sample'].replace(':', '_')}.eval"
        assert t["reference"]["file"] == f"references/{rel}", f"{main}: manifest reference differs from the selection"
        dst = args.refs_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            aws("s3", "cp", "--quiet", N437 + "references/" + rel, str(dst))
        got = hashlib.sha256(dst.read_bytes()).hexdigest()
        assert got == t["reference"]["sha256"], f"{main}: reference sha256 differs from the n437 manifest"
        refs[main] = s
        rows.append({"position": pos, "main": main, "environment": s["environment"], "reference": rel, "reference_sha256": got,
                     "n437_usd": t["usd"], "n437_total_time_s": t["total_time_s"]})
    out = {"label": f"lean230 {args.stage}: tasks and their n437 references (keys and hashes only)", "positions": [args.start, args.end - 1],
           "spec": {"commit": SPEC[0], "path": SPEC[1], "sha256": hashlib.sha256(spec_raw).hexdigest(), "task_order_sha256": spec["task order sha256"]},
           "selection": {"path": SELECTION, "sha256": hashlib.sha256(sel_raw).hexdigest()},
           "n437_manifest": {"s3": N437 + "manifest_v2.json", "sha256": hashlib.sha256(man_raw).hexdigest()},
           "n437_usd_sum": round(sum(r["n437_usd"] for r in rows), 2), "tasks": rows, "references": refs,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"lean230_refs_{args.stage}_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("positions", "n437_usd_sum")} | {"tasks": len(rows),
                     "environments": sorted({r["environment"] for r in rows})}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
