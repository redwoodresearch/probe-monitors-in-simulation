"""Codex's required change 2 on the like-for-like run (codex/review_a_like.md): check the two stored D_G_rows files, which
run_layers_a_like_resume2.py read for gate 2, against the sha256 digests version 2 recorded (layers_a_v2_2026-10-13_235012.json,
"D_G_rows_sha256"). One small Modal CPU function; writes results/check_dg_sha_<stamp>.json.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach experiments/2026-10-13_q3_layer_stability/check_dg_sha.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3layers_alike_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
V2 = ("layer_stability/a/layers_a_v2_2026-10-13_235012.json", "9995f91ed2ac0bc348bc67af4369ca13c8d71f873e551e8456633f35aedda47d")
DG = "recipe1_amsplits/2026-10-13_184928/D_G_rows_fpr_{a}_2026-10-13_185609.npz"

app = modal.App(APP)
work_vol = modal.Volume.from_name(WORK, version=2)


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=1, memory=1024, timeout=300, retries=0, volumes={"/work": work_vol})
def check() -> dict:
    b = (Path("/work") / V2[0]).read_bytes()
    if hashlib.sha256(b).hexdigest() != V2[1]:
        raise RuntimeError("version 2's result differs from its sha256")
    rec = json.loads(b)["D_G_rows_sha256"]
    out = {}
    for a in ("0.01", "0.05"):
        got = hashlib.sha256((Path("/work") / DG.format(a=a)).read_bytes()).hexdigest()
        out[f"fpr_{a}"] = {"file": DG.format(a=a), "sha256": got, "recorded_by_version_2": rec[f"fpr_{a}"], "match": got == rec[f"fpr_{a}"]}
    return out


@app.local_entrypoint()
def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
    out = check.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = {"label": "Codex required change 2 (like-for-like): D_G_rows digests", "git": head, "written": stamp, "results": out,
           "all_match": all(x["match"] for x in out.values())}
    with (HERE / "results" / f"check_dg_sha_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
