"""The nm360 S4 sidecars (4f, 10-15 about 00:20; 09's ruling): manifest_dt.json is append-only, so each S4 folder gets
manifest_dt_addendum_2026-10-15.json (exclusive-create). Every non_ended_staged key must be accounted for: "stopped" keys must be
ended by d9's header verdict (stopped_runs_nm360_v2 @ e241130bf: header status success, 1 sample, not excluded); "no policy call"
keys must be in the setup-failure record @ b0176a82c. Anything else stops before any write. Keys only.
  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-14_p2_dt_scaleup/write_s4_addendum.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_dtq4b_kill-2026-11-13"
WVOL = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
NAME = "manifest_dt_addendum_2026-10-15.json"
FOLDERS = ("nm360_s4_main_v1__attack", "nm360_s4_main_v1__honest_a", "nm360_s4_main_v1__honest_b", "nm360_s4_run2_v1__attack")
EXPECT = {"nm360_s4_main_v1__attack": (4, 0), "nm360_s4_main_v1__honest_a": (2, 1), "nm360_s4_main_v1__honest_b": (4, 1), "nm360_s4_run2_v1__attack": (1, 0)}  # (ended, excluded) by key
HEADER = ("e241130bf", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/stopped_runs_nm360_v2_2026-10-15.json",
          "2c31e8ef09580fc53463643a26addc1864a58b3b7590def6cd7b7fbc43d5e3ff")
SETUP = ("b0176a82c", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/setup_failures_nm360_2026-10-14.json")
wv = modal.Volume.from_name(WVOL, version=2)
app = modal.App(APP)


@app.function(image=modal.Image.debian_slim(python_version="3.13"), cpu=(1, 1), memory=(2048, 2048), timeout=300, retries=0,
              scaledown_window=10, startup_timeout=120, volumes={"/work": wv})
def write(header: dict, setup: dict, cites: dict) -> dict:
    wv.reload()
    hdr = {r["record"] + "#e0": r for r in header["records"]}
    sf = {r["record"] + "#e0": r for r in setup["records"]}
    out, plan = {}, {}
    for g in FOLDERS:  # plan every folder before writing any
        d = Path("/work/scaleup") / g
        m = json.loads((d / "manifest_dt.json").read_text())
        ended, excl = {}, {}
        for k, v in m["non_ended_staged"].items():
            h = hdr.get(k)
            if v["state"] == "stopped" and h and h["eval_header_status"] == "success" and h["eval_samples"] == 1 and h["excluded"] is False:
                ended[k] = {"join": v["join"], "chain_state": "stopped", "eval_header_status": h["eval_header_status"], "eval_samples": h["eval_samples"]}
            elif v["state"] == "no policy call" and k in sf:
                excl[k] = {"join": v["join"], "state": v["state"], "record": cites["setup"]}
            else:
                raise ValueError(f"{g}: {k} ({v['state']}) is neither ended by the header verdict nor in the setup-failure record")
        if (len(ended), len(excl)) != EXPECT[g]:
            raise ValueError(f"{g}: {len(ended)} ended and {len(excl)} excluded keys, expected {EXPECT[g]}")
        plan[g] = {"folder": g, "supersedes_fields": ["non_ended_staged"], "header_record": cites["header"], "ended_by_header_verdict": ended,
                   "excluded_at_read": excl, "manifest_dt_sha256": hashlib.sha256((d / "manifest_dt.json").read_bytes()).hexdigest(),
                   "ruling": "09 via 4f 10-15 about 00:20: score S4 as staged; the 11 relabelled runs are ended and read; the 2 nm126 runs are excluded at read"}
    raws = {g: json.dumps(body, indent=1).encode() for g, body in plan.items()}  # deterministic: no timestamps in the body
    for g, raw in raws.items():  # a rerun after a partial write: an existing sidecar must be byte-identical, else stop; never overwrite
        f = Path("/work/scaleup") / g / NAME
        if f.exists() and f.read_bytes() != raw:
            raise ValueError(f"{g}: {NAME} exists with other bytes; nothing overwritten")
    for g, body in plan.items():
        raw, f = raws[g], Path("/work/scaleup") / g / NAME
        if not f.exists():
            with f.open("xb") as fh:
                fh.write(raw)
        out[g] = {"sha256": hashlib.sha256(raw).hexdigest(), "ended_by_header_verdict": len(body["ended_by_header_verdict"]), "excluded_at_read": len(body["excluded_at_read"])}
    wv.commit()
    return out


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout
    me = "experiments/2026-10-14_p2_dt_scaleup/write_s4_addendum.py"
    if git("status", "--porcelain", "--", me) or not git("branch", "-r", "--contains", git("rev-parse", "HEAD").strip()):
        raise SystemExit("commit and push this script first")
    hraw, sraw = git("show", f"{HEADER[0]}:{HEADER[1]}"), git("show", f"{SETUP[0]}:{SETUP[1]}")
    if hashlib.sha256(hraw).hexdigest() != HEADER[2]:
        raise SystemExit("the header-verdict record differs from its pin")
    full = lambda c: git("rev-parse", "--verify", f"{c}^{{commit}}").decode().strip()
    cites = {"header": {"path": HEADER[1], "commit": full(HEADER[0]), "sha256": HEADER[2]},
             "setup": {"path": SETUP[1], "commit": full(SETUP[0]), "sha256": hashlib.sha256(sraw).hexdigest()}}
    r = write.remote(json.loads(hraw), json.loads(sraw), cites)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "keys only; nm360 S4 sidecars", "git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": stamp, "volume": WVOL, "file": NAME,
           "cites": cites, "folders": r}
    dest = HERE / "results" / f"s4_addendum_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    print(json.dumps(r, indent=1), dest.name)
