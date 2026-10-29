"""Post-review check (#860 thread on check_frozen_direction_hash.py): were part 2's scores made with the frozen
direction? score_part2.py did not record a direction hash, so this re-runs its own `score` (imported unchanged, via
`.local` in a Modal CPU container with the same Volumes) with the two direction reads, v.npy and r.npy, served from the
archived S3 file direction_L48.npz (sha256 checked in the container against manifest_frozen_direction_v1.json), and
compares the per-run projections with part 2's stored scores (manifest_part2_scores_v1.json). Keys and numbers only;
no activations leave the container.

  uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run \
      experiments/2026-10-11_generic_contrast_probe/verify_part2_frozen.py --npz <direction_L48.npz> --stored <part2 scores json>
"""

import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import score_part2 as S  # the as-run scorer, unchanged since fd78e7a1

app = modal.App("probe-shift_aswin-surya_sub-gcp-p2verify_score_kill-2026-11-12")
image = S.image.add_local_file(HERE / "score_part2.py", "/root/score_part2.py")


@app.function(image=image, cpu=16, memory=65536, timeout=1800, retries=0, volumes={"/score": S.score_vol.read_only(), "/acts": S.acts_vol.read_only()})
def verify(npz_bytes: bytes, expected_sha256: str, layer: int) -> dict:
    import numpy as np

    if hashlib.sha256(npz_bytes).hexdigest() != expected_sha256:
        raise ValueError("direction_L48.npz differs from manifest_frozen_direction_v1.json")
    z = np.load(io.BytesIO(npz_bytes), allow_pickle=False)
    frozen, served, real_load = {"v.npy": z["v_all"], "r.npy": z["r_all"]}, [], np.load

    def load(path, *a, **k):
        p = Path(path)
        if p.parent == Path("/score") / S.OUT and p.name in frozen:
            served.append(p.name)
            return frozen[p.name]
        return real_load(path, *a, **k)

    np.load = load
    try:
        out = S.score.local(layer)
    finally:
        np.load = real_load
    if sorted(served) != ["r.npy", "v.npy"]:
        raise ValueError(f"direction reads not served from the frozen file: {served}")
    return {**out, "served_from_frozen_npz": served}


@app.local_entrypoint()
def main(npz: str, stored: str):
    import numpy as np

    res = HERE / "results"
    want = {f["file"]: f["sha256"] for m in ("manifest_frozen_direction_v1.json", "manifest_part2_scores_v1.json") for f in json.loads((res / m).read_text())["files"]}
    for p in (npz, stored):
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != want[Path(p).name]:
            raise SystemExit(f"{p}: sha256 differs from its manifest")
    old = json.loads(Path(stored).read_text())
    new = verify.remote(Path(npz).read_bytes(), want["direction_L48.npz"], old["layer"])
    same = {k: new[k] == old[k] for k in ("layer", "rows", "runs", "columns", "calls_per_run", "detector_runs", "detector_score", "layer39_check")}
    diff, cols = {}, {"v": slice(0, 1), "v_res": slice(1, 2), "r": slice(2, 3), "random_100": slice(3, None)}
    for agg in ("mean", "first"):
        a, b = np.array(new[agg]), np.array(old[agg])
        diff[agg] = {g: {"max_abs_diff": float(np.abs(a[:, c] - b[:, c]).max()), "max_abs_stored": float(np.abs(b[:, c]).max()), "exact": bool((a[:, c] == b[:, c]).all())}
                     for g, c in cols.items()}
    ok = all(same.values()) and all(x["max_abs_diff"] <= 1e-5 * x["max_abs_stored"] for d in diff.values() for x in d.values())
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip())
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(npz), Path(stored), HERE / "score_part2.py", Path(__file__))}
    out = {"written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), "git": git, "tracked_dirty": dirty, "sha256": sha,
           "served_from_frozen_npz": new["served_from_frozen_npz"], "reproduces_part2_with_frozen_direction": ok,
           "rule": "every non-projection field equal, and in each column group (v, v_res, r, the 100 random) the largest difference within 1e-5 x that group's largest stored |value|",
           "fields_equal": same, "projections": diff}
    path = res / f"part2_frozen_direction_verify_{out['written']}.json"
    path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
    print(json.dumps({"ok": ok, "fields_equal": same}), "->", path)
    if not ok:
        raise SystemExit("part 2 NOT reproduced with the frozen direction (report written)")
