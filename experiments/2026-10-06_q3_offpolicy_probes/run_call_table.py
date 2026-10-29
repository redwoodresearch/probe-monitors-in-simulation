"""Per-call table for all four cohorts: tool, last argument, and its output's error, keyed as the segment-end
extraction keys calls (arm, key, step, action). Used for the slices (no editor calls; no calls after a tool-error
output; calls ending in content vs in a fixed argument). Streams the .eval archives from S3 (network-bound, no model).

Writes data/call_table_<stamp>.jsonl.gz (git-ignored; also uploaded to the Q3 Volume) and a summary in results/.

  PYTHONPATH=src uv run --no-project --python 3.14 --with-editable ../simulated-control-evals --with boto3 \
      --with pyyaml --with 'modal>=1.2' python experiments/2026-10-06_q3_offpolicy_probes/run_call_table.py
"""

import gzip
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import yaml

from probe_shift.offpolicy import call_table, provenance
from probe_shift.transcripts import eval_files, file_samples

HERE = Path(__file__).parent
cfg = yaml.safe_load((HERE / "config" / "q3.yaml").read_text())
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
(HERE / "results").mkdir(exist_ok=True)
(HERE / "data").mkdir(exist_ok=True)
out, record = HERE / "data" / f"call_table_{stamp}.jsonl.gz", HERE / "results" / f"call_table_{stamp}.json"
if out.exists() or record.exists():  # append-only: fail before the long S3 pass, and again at write time ("x")
    raise FileExistsError(f"{out} or {record} exists")

rows, summary = [], {}
for arm, alias in cfg["cohorts"].items():
    files = eval_files(alias)
    with ThreadPoolExecutor(32) as pool:
        for f, samples in zip(files, pool.map(lambda f: file_samples(alias, f), files)):
            for s in samples:
                key = f"{f.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}"
                rows += [{"arm": arm, "key": key} | r for r in call_table(s)]
    mine = [r for r in rows if r["arm"] == arm]
    summary[arm] = {"samples": len({r["key"] for r in mine}), "calls": len(mine),
                    "tools": dict(Counter(r["tool"] for r in mine).most_common()),
                    "errors": dict(Counter(r["error"] for r in mine if r["error"]).most_common()),
                    "last_arg_by_tool": {t: dict(Counter(r["last_arg"] for r in mine if r["tool"] == t).most_common(8))
                                         for t in sorted({r["tool"] for r in mine})}}
    print(arm, json.dumps(summary[arm]), flush=True)

with gzip.open(out, "xt") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
vol = modal.Volume.from_name(cfg["output_volume"], create_if_missing=True)
with vol.batch_upload() as batch:  # force=False (the default) refuses an existing Volume path
    batch.put_file(str(out), f"call_tables/{out.name}")
payload = json.dumps(  # provenance before the record exists, so a clean tree is not reported dirty
    {"file": str(out.relative_to(HERE)), "volume": f"{cfg['output_volume']}/call_tables/{out.name}",
     "cohorts": cfg["cohorts"], "summary": summary,
     "provenance": provenance(__file__, HERE.parent.parent / "src/probe_shift/offpolicy.py", HERE.parent.parent / "src/probe_shift/transcripts.py")}, indent=1)
with record.open("x") as fh:
    fh.write(payload)
print(f"{len(rows)} calls -> {out}")
