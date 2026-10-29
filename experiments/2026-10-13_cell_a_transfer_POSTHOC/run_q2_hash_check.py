"""Cell (a) transfer, provenance for #1539 (WiEC, WiEL): the q2 inputs of item 4 against q2 run 2026-10-12_011944's
originals. One read-only Modal CPU container hashes the originals on q2's work Volume (runs/<run>/probes_L*.npz and
scores_L*.npz; data/<data>/runs.json and tables.npz). The local copies the as-run job and analysis read were hashed on the
laptop with `shasum -a 256` (the JSON passed in), so nothing is uploaded. Compared per file: original vs local copy; and
for the probes, original vs the hashes run_xfer_layers.py recorded (`probes_sha256` in xfer_layers_POSTHOC_2026-10-13_155055.json).
The result is written to the Volume in the container, and locally.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_cell_a_transfer_POSTHOC/run_q2_hash_check.py --local <local_sha256.json> --xfer-layers <xfer_layers json>
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-cella_q2hash_kill-2026-11-13"
Q2, OUT = "probe-shift_aswin-surya_sub-q2-qwen35-reply_work_kill-2026-11-12", "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
RUN, DATA, LAYERS = "2026-10-12_011944", "d403ff8ad9c6", (16, 24, 32, 40, 48, 56, 63)
XFER_LAYERS_SHA = "9142c575fcdc867a18a5fc4fc8aaf670195afad7a21e3f9849abd2a2e55ea4d4"  # manifest_cell_a_transfer_v1.json
app = modal.App(APP)
out_vol = modal.Volume.from_name(OUT, version=2)


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=1, memory=2048, timeout=900, retries=0,
              volumes={"/q2": modal.Volume.from_name(Q2, version=2).read_only(), "/out": out_vol})
def check(local: dict, recorded: dict, git: str, stamp: str) -> dict:
    import hashlib

    files = {f"probes_L{L}.npz": f"runs/{RUN}/probes_L{L}.npz" for L in LAYERS} | {f"scores_L{L}.npz": f"runs/{RUN}/scores_L{L}.npz" for L in LAYERS}
    files |= {"runs.json": f"data/{DATA}/runs.json", "tables.npz": f"data/{DATA}/tables.npz"}
    rows = {}
    for name, path in files.items():
        h = hashlib.sha256(Path("/q2", path).read_bytes()).hexdigest()
        r = {"volume_path": path, "original_sha256": h, "local_sha256": local.get(name), "local_equal": local.get(name) == h}
        if name.startswith("probes_"):
            rec = recorded.get(name.removeprefix("probes_L").removesuffix(".npz"))
            r |= {"recorded_sha256": rec, "recorded_equal": rec == h}
        rows[name] = r
    res = {"label": "cell (a) transfer: q2 input provenance (#1539)", "git": git, "q2_volume": Q2, "q2_run": RUN, "q2_data": DATA, "files": rows,
           "all_local_equal": all(r["local_equal"] for r in rows.values()),
           "all_recorded_probes_equal": all(r["recorded_equal"] for k, r in rows.items() if k.startswith("probes_"))}
    d = Path("/out") / "cella_q2_hash_checks"
    d.mkdir(exist_ok=True)
    (d / f"q2_hash_check_{stamp}.json").write_text(json.dumps(res, indent=1))
    out_vol.commit()
    return res


@app.local_entrypoint()
def main(local: str, xfer_layers: str):
    import hashlib

    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head = g("rev-parse", "HEAD")  # read once, before any spawn
    b = Path(xfer_layers).read_bytes()
    if hashlib.sha256(b).hexdigest() != XFER_LAYERS_SHA:
        raise SystemExit("xfer_layers JSON sha256 differs from manifest v1")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = check.remote(json.loads(Path(local).read_text())["files"], json.loads(b)["probes_sha256"], head, stamp)
    dest = HERE / "results" / f"q2_hash_check_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    print("git", head)
    for k, r in res["files"].items():
        print(f"{k:16s} {r['volume_path']:40s} original {r['original_sha256'][:12]} local {'=' if r['local_equal'] else 'DIFFERS'}"
              + (f" recorded {'=' if r['recorded_equal'] else 'DIFFERS'}" if "recorded_equal" in r else ""))
    print("all local equal:", res["all_local_equal"], "| all recorded probes equal:", res["all_recorded_probes_equal"])
    print(dest.name)
