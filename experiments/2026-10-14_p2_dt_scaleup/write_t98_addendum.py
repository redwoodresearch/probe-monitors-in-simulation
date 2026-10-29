"""t98 class-record sidecar (4f relaying 09, 10-15 about 02:55): ilold_old_run3_s2's manifest_dt.json (append-only) lists its one
declared exclusion, run3|t98_model_registry, with the stager's verbatim reason, and its preflight cites a general class record
(il_exclusions_corrected_snap1 @ 99d8b5ac3) that covers snapshot 1, not t98 in snapshot 2. 09's Append 35 ruling @ 221bb94ea classes
it as a staging blocker ("no usable donor: no donor row in the pinned reference"), the same as t139. This writes
manifest_dt_addendum_2026-10-15c.json (exclusive-create; a rerun accepts only a byte-identical file). Keys and text only.
  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run experiments/2026-10-14_p2_dt_scaleup/write_t98_addendum.py
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
NAME = "manifest_dt_addendum_2026-10-15c.json"
FOLDERS = ("ilold_old_run3_s2_v1__attack",)
RULING = {"record": "experiments/2026-10-14_priority1_spec_09/RULE.md (Append 35 ruling, run3|t98_model_registry)", "commit": "221bb94ea",
          "class": "staging blocker: no usable donor: no donor row in the pinned reference (stager rule section 6); the run itself is complete; same as run3|t139_model_registry"}
CELL = "t98_model_registry"
EXPECTED_KEY = "runs/2026-10-14_il_old/attack/t98_model_registry/2026-10-15T07-54-40-00-00_la-model-registry-real-attack-1_jVhciUsUsBpBNqTnbfLs7q.eval#e0"  # the stage record's one exclusion (chain_stage_ilold_old_run3_s2_2026-10-15_015808.json @ 22396107c, c_gate.excluded)
NOTE = "preflight.citations.class_record (il_exclusions_corrected_snap1 @ 99d8b5ac3) covers IL snapshot 1 and does NOT cover run3|t98 in snapshot 2"
wv = modal.Volume.from_name(WVOL, version=2)
app = modal.App(APP)


@app.function(image=modal.Image.debian_slim(python_version="3.13"), cpu=(1, 1), memory=(2048, 2048), timeout=300, retries=0,
              scaledown_window=10, startup_timeout=120, volumes={"/work": wv})
def write() -> dict:
    wv.reload()
    raws = {}
    for g in FOLDERS:  # plan before writing: exactly one declared exclusion, the t98 run3 epoch, with the stager's reason kept verbatim
        m = json.loads((Path("/work/scaleup") / g / "manifest_dt.json").read_text())
        dec = m["declared_unstaged_epochs"]
        if set(dec) != {EXPECTED_KEY}:
            raise ValueError(f"{g}: declared exclusions are not exactly the pinned run3 {CELL} epoch")
        (key, reason), = dec.items()
        body = {"folder": g, "adds": "the class record for the declared run3|t98_model_registry exclusion", "excluded_key": key,
                "stager_reason_verbatim": reason, "class_ruling": RULING, "note": NOTE,
                "manifest_dt_sha256": hashlib.sha256((Path("/work/scaleup") / g / "manifest_dt.json").read_bytes()).hexdigest(),
                "source": "09's ruling @ 221bb94ea via 4f, 2026-10-15 about 02:55 PT"}
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
    me = "experiments/2026-10-14_p2_dt_scaleup/write_t98_addendum.py"
    if git("status", "--porcelain", "--", me) or not git("branch", "-r", "--contains", git("rev-parse", "HEAD")):
        raise SystemExit("commit and push this script first")
    if not git("rev-parse", "--verify", f"{RULING['commit']}^{{commit}}"):
        raise SystemExit("09's ruling commit is not available")
    r = write.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "keys only; t98 class-record sidecar", "git": git("rev-parse", "HEAD"), "stamp (PT)": stamp, "volume": WVOL,
           "file": NAME, "class_ruling": RULING, "note": NOTE, "folders": r}
    dest = HERE / "results" / f"t98_addendum_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    print(json.dumps(r, indent=1), dest.name)
