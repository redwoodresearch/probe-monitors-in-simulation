"""09's like-with-like job count (10-18, R-5 re-ruled (a)): the PS spec's jobs counted by the route C pre-flight's OWN selection rule,
against the committed pre-flight record's job count. The pre-flight (preflight_routec.py @ 37224810, analyze L53-56) takes the jobs whose
cfg.simh_old holds any entry tagged s_old_only, and files a tagged job whose origin is not S_old as a defect (L60-62); its record's
len(.jobs) counts the tagged S_old jobs. This script counts the same way in a Modal CPU container (the pickle is unpickled only there;
spec_diff's loader and image) and reports all jobs as AUDIT ONLY (the full factorization, registered-cohort rows included, rp L6627-6654).
Keys and counts only.

  modal run -n <description> tools_routec/preflight_job_count.py --b <local .pkl> --preflight-n <len(.jobs) of the committed record>
      --preflight-ref <path@sha256> --out results/spec_diff/<name>.json        (--out relative to the 507-prep folder, written exclusively)
"""

import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import modal

TAG = "s_old_only"  # = full_override.S_OLD_ONLY = preflight_routec's TAG
app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_jobcount_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")  # = spec_diff's image


def selection(spec):
    """The pre-flight's selection (L53-56) and its origin check (L60-62), by count; all jobs as audit."""
    jobs = spec.get("jobs") or []
    tagged = [i for i, j in enumerate(jobs) if any(isinstance(e, dict) and e.get(TAG) for es in ((j[1] or {}).get("simh_old") or {}).values() for e in es)]
    s_old = [i for i in tagged if (jobs[i][1] or {}).get("origin") == "S_old"]
    return {"n_tagged": len(tagged), "n_tagged_s_old": len(s_old), "n_tagged_not_s_old": len(tagged) - len(s_old), "n_jobs_all_audit": len(jobs)}


@app.function(image=image, cpu=(2, 4), memory=(4096, 32768), timeout=1800, retries=0)
def count_remote(b_bytes, preflight_n: int) -> dict:
    s = selection(pickle.loads(b_bytes))
    return s | {"like_with_like_equal": s["n_tagged_s_old"] == preflight_n, "pass": s["n_tagged_s_old"] == preflight_n and s["n_tagged_not_s_old"] == 0}


@app.local_entrypoint()
def main(b: str, preflight_n: int, preflight_ref: str, out: str):
    here = Path(__file__).resolve().parents[1]  # the 507-prep folder
    blob = Path(b).read_bytes()
    res = count_remote.remote(blob, preflight_n)
    rec = {"what": "like-with-like job count (09 R-5 (a)): tagged S_old jobs by the pre-flight's rule vs the pre-flight record",
           "b": {"local": b, "sha256": hashlib.sha256(blob).hexdigest()}, "preflight": {"ref": preflight_ref, "n_jobs": preflight_n},
           "rule": "preflight_routec.py @ 37224810 analyze L53-56 (selection), L60-62 (non-S_old tagged = defect); all jobs audit only",
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True, cwd=here).stdout.decode().strip()} | res
    data = json.dumps(rec, indent=1).encode()
    dest = here / out
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as fh:
        fh.write(data)
    print(json.dumps({"out": out, "sha256": hashlib.sha256(data).hexdigest()} | res))
