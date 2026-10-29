"""cell_hits_ext TEST gate (Z fork; 2e's GO 03:0x 10-16): two _TEST all-tasks records from the same head, one with --cell-hits-ext and one without,
compared by key in a Modal container (values never printed or read here). PASS iff every difference is (a) an addition on the flag-on record that
CELLX_RULE allows (rows.*.*.cell_hits_draw0.{plant, ext}, its levels.*.{sim2, run3, <sim|sim2>|plant<+d>} strings and arms_def entries for those arms,
meta.adds.cell_hits_ext) or (b) a per-launch field (stamps, app/driver ids, timings, Volume paths of the per-draw npz); anything else FAILS. Prints
key paths and counts only, and writes them to results/ (exclusive create).

  PYTHONPATH=src modal run experiments/2026-10-15_fpr_sweep_507_prep/compare_cellx.py --on <Volume path>@<sha256> --off <Volume path>@<sha256>
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"  # run_common455.VOLS["solavg"]
app = modal.App("probe-shift_aswin-surya_sub-ilz_kill-2026-11-14")
image = modal.Image.debian_slim(python_version="3.12")
EXT_ARM = re.compile(r"^(sim2|run3|(sim|sim2)\|plant[+-]\d+)$")
LAUNCH = [re.compile(x) for x in (  # exact per-launch leaves (Codex gate r1): everything else, deletions included, must match
    r"^meta\.(stamp|app_id|read_started_pt)$", r"^meta\.shipped_inputs_sent_now(\..+)?$",
    r"^meta\.queue\.(submit_unix|first_container_start_unix|last_container_start_unix|pending_s|peak_rss_mb_max|elapsed_s_max)$",
    r"^rows\.[^.]+\.[^.]+\.resources\.(container_start_unix|elapsed_s|peak_rss_mb)$", r"(^|\.)npz\.path$")]


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes={"/sol": modal.Volume.from_name(VOL, version=2)})
def compare(on: str, off: str) -> dict:
    def load(spec):
        p_, h_ = spec.rsplit("@", 1)
        b = (Path("/sol") / p_).read_bytes()
        if hashlib.sha256(b).hexdigest() != h_:
            raise SystemExit(f"{p_}: sha256 differs")
        return json.loads(b)

    diffs = []  # (path tuple, kind)

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b), key=str):
                if k not in b:
                    diffs.append((path + (k,), "only_on"))
                elif k not in a:
                    diffs.append((path + (k,), "only_off"))
                else:
                    walk(a[k], b[k], path + (k,))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                diffs.append((path, "length"))
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    walk(x, y, path + (i,))
        elif type(a) is not type(b) or (a != b and not (isinstance(a, float) and a != a and b != b)):
            diffs.append((path, "value"))

    walk(load(on), load(off), ())

    def allowed(p, kind):
        s = [str(x) for x in p]
        if (kind == "value" and any(r.search(".".join(s)) for r in LAUNCH)) or s[:2] == ["meta", "shipped_inputs_sent_now"]:  # the upload list: cache state
            return "launch"
        if kind != "only_on":
            return None
        if s[:3] == ["meta", "adds", "cell_hits_ext"]:
            return "ext"
        if len(s) >= 4 and s[0] == "rows" and s[3] == "cell_hits_draw0":
            t = s[4:]
            if t in (["plant"], ["ext"]) or (len(t) == 3 and t[0] == "levels" and EXT_ARM.match(t[2])) or (len(t) == 2 and t[0] == "arms_def" and EXT_ARM.match(t[1])):
                return "ext"
        return None

    cls = {"ext": [], "launch": [], "unexpected": []}
    for p, kind in diffs:
        cls[allowed(p, kind) or "unexpected"].append(".".join(str(x) for x in p) + f" [{kind}]")
    return {"pass": not cls["unexpected"], "n_diffs": len(diffs), "counts": {k: len(v) for k, v in cls.items()},
            "unexpected": cls["unexpected"][:200], "launch": sorted(set(cls["launch"]))[:200],
            "ext_by_kind": sorted({re.sub(r"^rows\.[^.]+\.[^.]+\.", "rows.<label>.<origin>.", x) for x in cls["ext"]})[:200]}


@app.local_entrypoint()
def main(on: str, off: str):
    import subprocess

    res = compare.remote(on, off)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"label": "cell_hits_ext TEST gate (keys only; values never read); POST HOC", "on": on, "off": off, "result": res,
           "meta": {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "app_id": app.app_id, "stamp_pt": stamp}}
    b = json.dumps(out, indent=1).encode()
    fn = HERE / "results" / f"compare_cellx_TEST_{stamp}_{app.app_id}.json"
    fn.open("xb").write(b)
    print(fn.name, hashlib.sha256(b).hexdigest(), "PASS" if res["pass"] else "FAIL", res["counts"])
    for x in res["unexpected"][:40]:
        print("UNEXPECTED", x)
