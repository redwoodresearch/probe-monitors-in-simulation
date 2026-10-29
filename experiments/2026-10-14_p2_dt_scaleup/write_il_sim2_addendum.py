"""The routing-1 IL sim2 correction sidecars (4f approved, 10-15 about 01:25; d9's correction): the three folders' manifest_dt.json
(append-only) embed preflight.donor_record citing il_donor_shas_2026-10-15_011125.json @ f655bdc4d with a note that the excluded
runs' first-user-turn sha matches the pinned donor. That hash comparison is invalid (every hash is sha256 of the empty string).
Each folder gets manifest_dt_addendum_2026-10-15b.json (exclusive-create; a rerun accepts only byte-identical existing files).
Keys only. One small Modal job; the record is written by the local entrypoint.
  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-14_p2_dt_scaleup/write_il_sim2_addendum.py
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
NAME = "manifest_dt_addendum_2026-10-15b.json"
FOLDERS = ("ilnm_s2_sim2_s1_v1__sim_attack", "ilnm_s1_sim2_s1_v1__sim_attack", "il311_q2_sim2_s1_v1__sim_attack")
CITED = {"path": "experiments/2026-10-14_q3_overnight_chain/results/il_donor_check/il_donor_shas_2026-10-15_011125.json", "commit": "f655bdc4d",
         "sha256": "6734b535516648f60e1e350ddae4c43ccd9e99196f884ca47ca297d43588f85a"}
CORRECTION = ("f655bdc4d is cited ONLY as evidence that the excluded runs were 1-message RateLimitError failures; its first-user-turn hashes "
              "are sha256 of the empty string and are NOT donor-hash evidence; the staged runs' donor attachment comes from the stage records")
wv = modal.Volume.from_name(WVOL, version=2)
app = modal.App(APP)


@app.function(image=modal.Image.debian_slim(python_version="3.13"), cpu=(1, 1), memory=(2048, 2048), timeout=300, retries=0,
              scaledown_window=10, startup_timeout=120, volumes={"/work": wv})
def write() -> dict:
    wv.reload()
    raws = {}
    for g in FOLDERS:  # plan every folder before writing any: each manifest must cite exactly the record being corrected
        m = json.loads((Path("/work/scaleup") / g / "manifest_dt.json").read_text())
        dr = m.get("preflight", {}).get("donor_record", {})
        if (dr.get("path"), dr.get("commit"), dr.get("sha256")) != (CITED["path"], CITED["commit"], CITED["sha256"]):
            raise ValueError(f"{g}: manifest_dt.json does not cite the record being corrected")
        body = {"folder": g, "supersedes_fields": ["preflight.donor_record.note"], "cited_record": CITED, "correction": CORRECTION,
                "corrected_record": None, "manifest_dt_sha256": hashlib.sha256((Path("/work/scaleup") / g / "manifest_dt.json").read_bytes()).hexdigest(),
                "source": "d9's correction via 4f, 2026-10-15 about 01:20 PT; 4f approved this sidecar about 01:25 PT"}
        raws[g] = json.dumps(body, indent=1).encode()  # deterministic: no timestamps in the body
    for g, raw in raws.items():  # a rerun after a partial write: an existing sidecar must be byte-identical, else stop; never overwrite
        f = Path("/work/scaleup") / g / NAME
        if f.exists() and f.read_bytes() != raw:
            raise ValueError(f"{g}: {NAME} exists with other bytes; nothing overwritten")
    out = {}
    for g, raw in raws.items():
        f = Path("/work/scaleup") / g / NAME
        if not f.exists():
            with f.open("xb") as fh:
                fh.write(raw)
        out[g] = {"sha256": hashlib.sha256(raw).hexdigest()}
    wv.commit()
    return out


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()
    me = "experiments/2026-10-14_p2_dt_scaleup/write_il_sim2_addendum.py"
    if git("status", "--porcelain", "--", me) or not git("branch", "-r", "--contains", git("rev-parse", "HEAD")):
        raise SystemExit("commit and push this script first")
    raw = subprocess.run(["git", "-C", str(REPO), "show", f"{CITED['commit']}:{CITED['path']}"], capture_output=True, check=True).stdout
    if hashlib.sha256(raw).hexdigest() != CITED["sha256"]:
        raise SystemExit("the cited record differs from its pin")
    r = write.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "keys only; routing-1 IL sim2 correction sidecars", "git": git("rev-parse", "HEAD"), "stamp (PT)": stamp, "volume": WVOL,
           "file": NAME, "cited_record": CITED, "correction": CORRECTION, "folders": r}
    dest = HERE / "results" / f"il_sim2_addendum_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    print(json.dumps(r, indent=1), dest.name)
