"""Post-review check (generic-probe PR threads on #815 and #817): are the direction files that score_part1.py and
score_part2.py read on the score Volume (generic_contrast_v1/v.npy, r.npy, B.npy, rB.npy) still exactly the frozen
directions archived on S3 (manifest_frozen_direction_v1.json, manifest_frozen_direction_b_v1.json)? One small Modal CPU
container reads the Volume read-only and returns each array's sha256 (C-order float32 bytes), plus the folder's file
sizes; the laptop compares them with the same hashes of the S3 npz arrays (directions only, no activations).

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run \
      experiments/2026-10-11_generic_contrast_probe/check_frozen_direction_hash.py --npz-dir <dir with the two S3 npz>
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

app = modal.App("probe-shift_aswin-surya_sub-gcp-hashcheck_kill-2026-11-10")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
PAIRS = {"v.npy": ("direction_L48.npz", "v_all"), "r.npy": ("direction_L48.npz", "r_all"),
         "B.npy": ("direction_B_L46.npz", "B_all"), "rB.npy": ("direction_B_L46.npz", "rB_all")}


def digest(a) -> dict:
    import numpy as np

    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


@app.function(image=image, cpu=1, memory=2048, timeout=300, retries=0, volumes={"/score": vol.read_only()})
def volume_hashes() -> dict:
    import numpy as np

    d = Path("/score/generic_contrast_v1")
    return {"arrays": {n: digest(np.load(d / n, allow_pickle=False)) for n in PAIRS},
            "files": {p.name: p.stat().st_size for p in sorted(d.iterdir())}}


@app.local_entrypoint()
def main(npz_dir: str):
    import numpy as np

    here = Path(__file__).parent / "results"
    for m in ("manifest_frozen_direction_v1.json", "manifest_frozen_direction_b_v1.json"):  # the local npz must be the frozen S3 file
        for f in json.loads((here / m).read_text())["files"]:
            if f["file"].endswith(".npz") and hashlib.sha256((Path(npz_dir) / f["file"]).read_bytes()).hexdigest() != f["sha256"]:
                raise SystemExit(f"{f['file']}: sha256 differs from {m}")
    got = volume_hashes.remote()
    rows, ok = {}, True
    for n, (f, k) in PAIRS.items():
        z = digest(np.load(Path(npz_dir) / f, allow_pickle=False)[k])
        same = z == got["arrays"][n]
        ok &= same
        rows[n] = {"volume": got["arrays"][n], "s3": {"file": f, "key": k, **z}, "identical": same}
    out = {"written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "all_identical": ok,
           "pairs": rows, "volume_files_bytes": got["files"]}
    path = here / f"frozen_direction_hash_{out['written']}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({"all_identical": ok, "files": got["files"]}), "->", path)
