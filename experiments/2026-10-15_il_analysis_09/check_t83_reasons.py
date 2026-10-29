"""Keys-only (09 via 2e, item C): the `dropped` REASON for honest_b|t83 in the three pinned score files, and asis's run_folds entry. On Modal; each file
sha-checked against run_il_posthoc.py's PINS. Output: the dropped entry's string (key or key+reason), its container's keys, and any reason field;
for asis the run_folds key and its fold label. No score is read or copied.

  PYTHONPATH=src modal run experiments/2026-10-15_il_analysis_09/check_t83_reasons.py --xe-in <dir with the pinned files>
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
        found = []

        def walk(o, path):
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(k, str) and k.startswith("honest_b|t83_"):
                        found.append({"path": path + [k], "kind": "dict key", "value_type": type(v).__name__,
                                      "value_if_label": v if isinstance(v, (str, int)) and not isinstance(v, bool) and (isinstance(v, str) or abs(v) < 10) else None})
                    walk(v, path + [k])
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    if isinstance(v, str) and v.startswith("honest_b|t83_"):
                        found.append({"path": path + [f"[{i}]"], "kind": "list item", "string": v,
                                      "list_len": len(o), "siblings_are_strings": all(isinstance(x, str) for x in o),
                                      "list_items": [x for x in o if isinstance(x, str)] if len(o) <= 10 else None})  # run keys only
                    elif isinstance(v, dict) and any(isinstance(x, str) and x.startswith("honest_b|t83_") for x in v.values()):
                        found.append({"path": path + [f"[{i}]"], "kind": "dict in list",
                                      "entry": {k2: v2 for k2, v2 in v.items() if isinstance(v2, str)}})
                    walk(v, path + [f"[{i}]"])
            elif isinstance(o, str) and o.startswith("honest_b|t83_") and not path[-1:] == ["__list__"]:
                pass
        d = json.loads(b)
        walk(d, [])
        reasons = {}

        def find_reasons(o, path):  # any dict key mentioning reason / why near a 'dropped' container
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(k, str) and ("reason" in k.lower() or "why" in k.lower()) and "drop" in "/".join(path + [k]).lower():
                        reasons["/".join(path + [k])] = v if isinstance(v, str) else (v if isinstance(v, (list, dict)) and len(json.dumps(v)) < 400 else type(v).__name__)
                    find_reasons(v, path + [k])
        find_reasons(d, [])
        out[name] = {"occurrences": found, "drop_reason_fields": reasons}
    return out


@app.local_entrypoint()
def main(xe_in: str):
    res = scan.remote({n: (Path(xe_in) / n).read_bytes() for n in FILES})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    b = json.dumps({"label": "keys-only: the dropped reason for honest_b|t83 (no scores)", "files": FILES, "result": res, "stamp_pt": stamp,
                    "app_id": app.app_id}, indent=1, ensure_ascii=False).encode()
    fn = HERE / "results" / f"check_t83_reasons_{stamp}_{app.app_id}.json"
    fn.open("xb").write(b)
    print(fn.name, hashlib.sha256(b).hexdigest())
