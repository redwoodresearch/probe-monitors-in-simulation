"""Amendment 2 checks (checks_read10.py) on Modal CPU: no local compute (Aswin 10-13 17:24). The 24 .eval files (12 sim,
12 real) were uploaded with `modal volume put` to VOLUME under RUN/{sim,real}/; the function reads them there, records
their sha256, and returns numbers, booleans, keys and check ids only. The local side writes results/checks_<stamp>.json.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_claimB_read10/run_checks_modal.py
"""

import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
NAME = "probe-shift_aswin-surya_sub-claimb-read10_kill-2026-11-12"
VOLUME = "probe-shift_aswin-surya_sub-claimb-read10_inputs_kill-2026-11-12"
RUN = "/vol/run_2026-10-13_read10"

app = modal.App(NAME)
vol = modal.Volume.from_name(VOLUME)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("zstandard==0.23.0")
    .add_local_file(HERE / "checks_read10.py", "/root/read10/checks_read10.py")
)


@app.function(image=image, volumes={"/vol": vol}, cpu=1.0, memory=2048, timeout=900, retries=0)
def checks(draw: dict, movers: dict) -> dict:
    import sys

    sys.path.insert(0, "/root/read10")
    import checks_read10 as C

    return C.compute(f"{RUN}/sim", f"{RUN}/real", draw, movers)


@app.local_entrypoint()
def main():
    import sys

    sys.path.insert(0, str(HERE))
    import checks_read10 as C

    draw = json.loads((HERE / C.DRAW).read_text())
    movers = json.loads((HERE / C.MOVERS).read_text())
    out = checks.remote(draw, movers)
    print(C.write_result(out).name)
