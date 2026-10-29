"""own_frac{10,25,100}_L48's fit spec, read by key on Modal (no download): the spec's sha256, the training worlds (pass 1's label rule at the fit's
git, applied to the spec's map), the training launch ids (each training key's first path component) with counts per world, and any cut /
max-length setting in the spec. Keys, counts and settings only; no rows or scores are read. Read-only; writes one record to results/.
  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run <this file>"""

import collections
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC = ("2026-10-14_163027", "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61")
FIT_GIT = "b5a0586fa"  # pass 1's fit run 2026-10-14_175636 (results/fit_2026-10-14_175636.json "git")
app = modal.App("probe-shift_aswin-surya_sub-p2probes_specread_kill-2026-11-14")
VOL = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2).read_only()


@app.function(image=modal.Image.debian_slim(python_version="3.12"), cpu=1, memory=2048, timeout=300, retries=0, volumes={"/out": VOL})
def read(train_worlds: list) -> dict:
    b = (Path("/out/famexp") / SPEC[0] / f"spec_{SPEC[0]}.json").read_bytes()
    spec = json.loads(b)
    worlds = collections.Counter(w for w, _ in spec["map"].values())
    train = {k: w for k, (w, _) in spec["map"].items() if w in train_worlds}
    launches = collections.Counter((k.split("/", 1)[0], w) for k, w in train.items())
    cutlike = {k: v for k, v in spec.items() if any(s in k.lower() for s in ("cut", "max", "len", "trunc", "token"))}
    return {"sha256": hashlib.sha256(b).hexdigest(), "spec_keys": sorted(spec), "map_worlds": dict(sorted(worlds.items())),
            "training_keys": len(train), "training_launches": {f"{l_}|{w}": n for (l_, w), n in sorted(launches.items())},
            "training_launch_ids": sorted({l_ for l_, _ in launches}), "cut_or_max_length_in_spec": cutlike or "none"}


@app.local_entrypoint()
def main():
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{FIT_GIT}:experiments/2026-10-14_p2_family_expansion/run_famexp.py"], capture_output=True, check=True).stdout.decode()
    rule = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("y = np.where(np.isin(role")]
    if rule != ['y = np.where(np.isin(role, ["run1", "sol:E1"]), 1.0, np.where(np.isin(role, ["honest_a", "sol:H"]), 0.0, np.nan))']:
        raise SystemExit(f"pass 1's label rule at {FIT_GIT} is not the expected one: {rule}")
    sol_cut = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("C, CUT, DROP =")]
    out = read.remote(["run1", "honest_a"])  # own-family rows: role = the map world; y = 1 run1, 0 honest_a, else NaN (never trained)
    rec = {"label": "own_frac{10,25,100}_L48 fit spec read by key on Modal (no download): sha, training worlds, training launches, cut setting; keys and counts only",
           "stamp (PT)": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"),
           "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(), "app_id": app.app_id,
           "spec": {"volume": "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", "path": f"/famexp/{SPEC[0]}/spec_{SPEC[0]}.json", "pinned_sha256": SPEC[1]},
           "sha256_equals_pin": out["sha256"] == SPEC[1],
           "training_worlds": {"rule": f"run_famexp.py @ {FIT_GIT}: {rule[0]} (own family: role = the spec map's world)", "worlds": ["run1", "honest_a"]},
           "code_cut_constant_at_fit_git": {"line": sol_cut, "applies_to": "Sol E1/H rows of the ad family only (_sol_rows); own-family rows come from the B staging as staged"}} | out
    (HERE / "results" / f"fit_spec_read_{rec['stamp (PT)']}.json").open("x").write(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
