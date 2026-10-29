"""Token volume of Phase 2 request logs (counts only; no text). Modal CPU, seconds.

For each named request-log file on the request-log Volume: requests, prompt and completion tokens, the longest request.
Used to size the on-policy replay extraction.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.2' modal run \
      experiments/2026-10-08_q3_onpolicy_figures/run_count_request_logs.py --names <file>,<file>,... --label <name>
"""

import gzip
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
app = modal.App("probe-shift_aswin-surya_sub-q3onpol_reqcount_kill-2026-11-06")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-phase2_request-log_kill-2026-11-06")


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=2, memory=16384, timeout=900, retries=0,
              volumes={"/rl": vol.read_only()})
def count(names: list) -> dict:
    out = {}
    for n in names:
        p = Path("/rl") / n
        if not p.exists():
            raise FileNotFoundError(n)
        nreq = ptok = ctok = mx = 0
        with gzip.open(p, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                req, resp = r.get("request") or r, r.get("response") or r
                pt = len(req.get("prompt_token_ids") or resp.get("prompt_token_ids") or [])
                ct = sum(len(c.get("token_ids") or []) for c in (resp.get("choices") or []))
                nreq, ptok, ctok, mx = nreq + 1, ptok + pt, ctok + ct, max(mx, pt + ct)
        out[n] = {"requests": nreq, "prompt_tokens": ptok, "completion_tokens": ctok, "max_request_tokens": mx}
    return out


@app.local_entrypoint()
def main(names: str, label: str):
    res = count.remote(names.split(","))
    tot = {k: sum(v[k] for v in res.values()) for k in ("requests", "prompt_tokens", "completion_tokens")}
    tot["max_request_tokens"] = max(v["max_request_tokens"] for v in res.values())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"request_log_counts_{label}_{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps({"label": label, "files": res, "total": tot}, indent=1))
    print(json.dumps(tot))
    print("->", path)
