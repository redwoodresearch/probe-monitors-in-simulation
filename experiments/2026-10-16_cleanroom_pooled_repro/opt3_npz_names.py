"""Option (3), stage 0: npz MEMBER NAMES only (09's spec via 2e, 2026-10-21 ~09:1x; PRESTATEMENT_OPT3.md @ 3c95cb9d1, §4 C1).

Lists the member names (zip entries) of three of the producer's D1 calibration npz files and of B's per-draw gap npz, so that C1's
comparison can be coded. It never opens an array: no np.load, and no ZipInfo size, shape or dtype is read into the output. Each file's
sha256 is recomputed in the container and compared with its pin; only booleans leave, and any mismatch STOPs. The pins come from 2e's
strings-only JSONs: opt3_calibration_paths_AB.json (sha256 c1ea9431…) for the calibration npz files, and opt3_perdraw_npz_AB.json
(sha256 28c8302c…) for B's per-draw npz, which B's record names. Output: names, counts and booleans in one keys-only file.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run opt3_npz_names.py
"""

import hashlib
import json
import time
import zipfile
from pathlib import Path

import modal

DEADLINE_S = 420  # per attempt: startup_timeout 120 + timeout 240 + scaledown 60 (OPT3 addendum 3 §1)
APP = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
POOLED = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"  # the xfer/fpr507prep tree (as cleanroom_downstream.py NPZ_VOL)
PINNED = {  # role -> (path on POOLED, sha256), copied from opt3_calibration_paths_AB.json (sha256 c1ea9431…)
    "A_calib_single_FIBS_C": ("xfer/fpr507prep/pooledworld/alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327/calib/"
                              "FIBS_uncut_C__pooled_all_origins_013a35fdd258ea22_ab94a17f.npz",
                              "013a35fdd258ea22de3726131dbd9fbb1ec4b8566202305c0c4f32ed8aac3b4e"),
    "A_calib_five_sol_a_C": ("xfer/fpr507prep/pooledworld/alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327/calib/"
                             "sol_a_uncut_C__pooled_all_origins_a27e63c3086962b1_5e17d278.npz",
                             "a27e63c3086962b1f270b2568169719b47c5d08e9e12ee3dcf1e8ac46d3bdccc"),
    "B_calib_five_sol_a_C": ("xfer/fpr507prep/pooledworld/alltasks_CRD_FULL_r2_allorigins_sweep15_pooledworld_scored_e12_2026-10-19_180652/calib/"
                             "sol_a_uncut_C__pooled_all_origins_808a11fc187108d2_0a8c068d.npz",
                             "808a11fc187108d245530cc63ebca7536f69bf8103ecd5279f715ab656df00ca"),
}
# A's per-draw gap npz is already listed (addendum 20d, npz_keys20d). B's, as B's record names it: 2e's opt3_perdraw_npz_AB.json (28c8302c…)
PINNED["B_gap_perdraw"] = ("xfer/fpr507prep/alltasks_CRD_FULL_r2_allorigins_sweep15_pooledworld_scored_e12_2026-10-19_180652_perdraw.npz",
                           "8a2a2afda1ff5271695191be04911fcd9ecb4bf7199d5715fe6a526c9b2ffaa3")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(image=image, cpu=(1, 1), memory=(1024, 1024), timeout=240, startup_timeout=120, scaledown_window=60, retries=0,
              volumes={"/p": modal.Volume.from_name(POOLED, version=2).read_only()})
def names(launch_epoch: float) -> dict:
    if time.time() > launch_epoch + DEADLINE_S:  # 09 (#5 ii): a platform retry after the deadline exits at once
        raise RuntimeError("STOP: past the launch deadline")
    out, err = {"files": {}}, None
    try:  # nothing but a fixed string or an exception type name leaves on failure; raised after the except block
        for role, (path, want) in PINNED.items():
            p = Path("/p", path)
            ok = file_sha(p) == want
            out["files"][role] = {"path": path, "sha256_matches_pin": ok}
            if not ok:
                err = f"sha mismatch: {role}"
                raise ValueError
            with zipfile.ZipFile(p) as z:  # namelist() reads the central directory only; no member is opened
                nl = z.namelist()
            out["files"][role].update(n_members=len(nl), members=nl)
    except Exception as e:
        err = err or type(e).__name__
    if err:
        raise RuntimeError(f"STOP: {err}")
    return out


@app.local_entrypoint()
def main():
    out = names.remote(time.time())  # the launch epoch for the in-function deadline (OPT3 addendum 3 §1)
    out["pins_from"] = [{"file": "opt3_calibration_paths_AB.json", "sha256": "c1ea94312e9b5ce07a4d7f24f54ed0f28e7a4c83d9c45aa982deabe44480dbeb"},
                        {"file": "opt3_perdraw_npz_AB.json", "sha256": "28c8302c7c2128e1d802ba9e7684d45fa25dddb631104a045fceb9c801aab111"}]
    path = Path(__file__).parent / "results" / f"opt3_npz_names_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
    with open(path, "x") as fh:  # exclusive create (append-only results)
        fh.write(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({r: {k: v for k, v in f.items() if k != "members"} for r, f in out["files"].items()}))
    print("wrote", path)
