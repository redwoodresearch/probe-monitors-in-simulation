"""Keys-only (2e for 0d / 09, 10-15 22:5x; second refresh): which of the four pinned T196 score files lack an honest_b row for cell t83? Runs on Modal;
reads each file (sha-checked against run_il_posthoc.py's PINS), searches every string key and value for 'honest_b|t83_' and reports counts only.

  PYTHONPATH=src modal run experiments/2026-10-15_il_analysis_09/check_t83_honest_b.py --xe-in <dir with the pinned files>
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
FILES = {"fibs_scores_2026-10-14_124948.json": "366d594cd719d52e9bb11d34bf4d3491b6cbd82d64f61efa73f476773cb49b1a",
         "scores_2026-10-14_145748.json": "d0b7f028f8ac3795dd946a261f384f474748a296772fb1f2f2f22cecaac6c961",
         "highstakes_scores_133519.json": "2d251dee71a4a0366bd9549fcc3726511fe576aafce0bb6a913353d7fc06f38a",
         "asis_scores_142228.json": "59c051a4ff9cc40811745a119251224f031cdaf479b8d3fb909c1059ab94da31"}
app = modal.App("probe-shift_aswin-surya_sub-ilz_kill-2026-11-14")


@app.function(cpu=1, memory=2048, timeout=600, retries=0, image=modal.Image.debian_slim(python_version="3.12"))
def scan(blobs: dict) -> dict:
    out = {}
    for name, b in blobs.items():
        if hashlib.sha256(b).hexdigest() != FILES[name]:
            raise SystemExit(f"{name}: sha256 differs from the pin")
        hits, t83_any = [], 0

        def walk(o, path):
            nonlocal t83_any
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(k, str) and k.startswith("honest_b|t83_"):
                        hits.append(path + [k])
                    if isinstance(k, str) and "t83_" in k:
                        t83_any += 1
                    walk(v, path + [k] if len(path) < 3 else path)
            elif isinstance(o, list):
                for v in o:
                    walk(v, path)
            elif isinstance(o, str):
                if o.startswith("honest_b|t83_"):
                    hits.append(path + ["(value)"])
                if "t83_" in o:
                    t83_any += 1
        walk(json.loads(b), [])
        dropped = [h for h in hits if any("drop" in str(x).lower() or "exclu" in str(x).lower() for x in h)]
        rows_ = [h for h in hits if h not in dropped]
        paths_ = sorted({" / ".join(map(str, h[:4])) for h in hits})
        out[name] = {"honest_b_t83_occurrences": len(hits), "listed_as_dropped_or_excluded": len(dropped), "other_occurrences": len(rows_),
                     "occurrence_paths": paths_[:12], "any_t83_string_occurrences": t83_any,
                     "status": "LACKS honest_b|t83 (absent)" if not hits else "LACKS a scored honest_b|t83 row (listed as dropped/excluded only)"
                     if not rows_ else "has honest_b|t83 outside a dropped/excluded list (see occurrence_paths)"}
    return out


@app.local_entrypoint()
def main(xe_in: str):
    blobs = {n: (Path(xe_in) / n).read_bytes() for n in FILES}
    res = scan.remote(blobs)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    b = json.dumps({"label": "keys-only: honest_b|t83 in the four pinned T196 score files (counts only; no scores)", "files": FILES, "result": res,
                    "stamp_pt": stamp, "app_id": app.app_id}, indent=1).encode()
    fn = HERE / "results" / f"check_t83_honest_b_{stamp}_{app.app_id}.json"
    fn.open("xb").write(b)
    print(fn.name, hashlib.sha256(b).hexdigest(), json.dumps({k: v["status"] for k, v in res.items()}))
