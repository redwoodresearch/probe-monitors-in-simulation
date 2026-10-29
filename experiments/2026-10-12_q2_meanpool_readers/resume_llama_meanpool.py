"""Resume the Llama mean-pooled extraction (run 2026-10-12_la_output-end-mean_llama33-70b_l2_v1) without a remote
orchestrator. On 10-12 at about 22:10 PT the first launch's orchestrator container was restarted and re-spawned all 24
shards while the originals still ran (24 containers started 21:36, 25 at 22:10); the app was stopped at 22:19. This
launcher imports the extractor unchanged (run_extract_llama_meanpool.py, whose source sha256 the run's stored config
pins), re-checks that pin, and spawns the 24 shards from the local entrypoint, so no remote function can restart and spawn
them again. Each shard resumes as the extractor does: it drops chunks without their done marker (and their metas), skips
transcripts whose meta exists, and writes every file to a temp name then renames it, the done marker last.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach \\
      experiments/2026-10-12_q2_meanpool_readers/resume_llama_meanpool.py --runs <#36 runs.json> --fix <fix939 per-run jsonl>
"""
import hashlib
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_extract_llama_meanpool as X  # noqa: E402  (unchanged; defines the app, the shard function and the Volumes)

app = X.app


@app.local_entrypoint()
def resume(runs: str, fix: str):
    for f in (Path(__file__).resolve(), Path(X.__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    for p_, role in ((Path(runs), "runs.json"), (Path(fix), "fix939_frozen_runs_POSTHOC_2026-10-11_020458.jsonl")):
        if hashlib.sha256(p_.read_bytes()).hexdigest() != X.PIN[role]:
            raise SystemExit(f"{p_.name} is not the pinned {role}")
    rs = json.loads(Path(runs).read_text())
    fix_keys = sorted({json.loads(l)["key"] for l in open(fix) if json.loads(l)["label"] == "FIX"})
    if len(rs) != 3807 or len(fix_keys) != 823:
        raise SystemExit("not #36's 3,807 runs and fix939's 823 re-rendered ones")
    cohort = [{"arm": r["arm"], "key": r["key"], "filename": r["filename"]} for r in rs]
    cfg = json.loads(b"".join(X.out_vol.read_file(f"{X.RUN}/config.json")))  # the first launch's stored config, used as is
    src = hashlib.sha256(Path(X.__file__).read_bytes()).hexdigest()
    if cfg["run"] != X.RUN or cfg["scope"] != "all" or cfg["shards"] != X.SHARDS or cfg["source sha256"] != src or cfg.get("limit_items"):
        raise SystemExit("the stored config is not this run's full extraction with this extractor's source")
    # refuse while any other launch of this app is live: an earlier (detached) launch could still hold shards
    rows = json.loads(subprocess.run(["uv", "run", "--no-project", "--with", "modal==1.6.1", "modal", "app", "list", "--json"],
                                     capture_output=True, text=True, check=True, timeout=180).stdout)
    live = [r for r in rows if r["description"] == X.APP and r["app_id"] != app.app_id and r["state"] != "stopped"]
    if live:
        raise SystemExit(f"another launch of {X.APP} is live ({[r['app_id'] for r in live]}): stop it first")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec_p = HERE / "results" / f"extract_resume_{cfg['run']}_{stamp}.json"
    rec = {"resumed": stamp, "git": head, "app": app.app_id, "run": cfg["run"], "source sha256": src, "function calls": []}
    calls = []
    for i in range(cfg["shards"]):  # spawned once, from here only; each call id is saved as soon as it exists
        calls.append(X.extract_shard.spawn(cfg, i, cohort, fix_keys))
        rec["function calls"].append(calls[-1].object_id)
        tmp = rec_p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(rec, indent=1) + "\n"); tmp.replace(rec_p)  # atomic: no saved id is lost
    print(f"{cfg['run']}: {len(calls)} shards spawned from the local entrypoint", flush=True)
    stats = []
    for shard, call in enumerate(calls):  # the orchestrator's wait loop, run locally
        for _ in range(240):
            try:
                stats.extend(call.get())
                break
            except (socket.gaierror, ConnectionError, TimeoutError, modal.exception.ConnectionError, modal.exception.ServiceError):
                time.sleep(30)
            except Exception as e:  # the shard itself failed: recorded by type only (a message may quote staged text)
                stats.append({"shard": shard, "crashed": type(e).__name__})
                break
        else:
            stats.append({"shard": shard, "unreachable": True})
    ok = [x for x in stats if "chunks" in x]
    summary = {"resumed": stamp, "shards": stats, "n_ok": len(ok), "n_crashed": len(stats) - len(ok),
               "transcripts (chunks live at each shard's end)": sum(c["transcripts"] for x in ok for c in x["chunks"]),
               "rows": sum(c["rows"] for x in ok for c in x["chunks"]), "nonfinite": sum(c["nonfinite"] for x in ok for c in x["chunks"])}
    with open(dest := HERE / "results" / f"extract_resume_summary_{cfg['run']}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "shards"})); print(dest)
