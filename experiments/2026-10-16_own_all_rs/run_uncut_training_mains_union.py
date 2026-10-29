"""DISTINCT training mains of own_all_rs{,25,10}_uncut_L48 (4f for 0d's v80 number map, 10-16): per variant x readout, the union of the 5 fold
models' training manifests (/out/own_all_rs_uncut/<stamp>/training/<variant>__<R|C>__f<k>.json on the scores Volume, each sha-checked against the
fit record's training_manifest_sha256), with per-fold sizes (= fit.fits[*].mains) and the union's mains by key. One Modal CPU function (not the
laptop); the entrypoint only checks git and writes the APPEND-ONLY NOTE. Keys / counts only.
  modal run --detach experiments/2026-10-16_own_all_rs/run_uncut_training_mains_union.py"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_ownall-mainsunion_kill-2026-11-14"
FIT_GIT = ("experiments/2026-10-14_p2_family_expansion/results/own_all_rs_uncut_fit_2026-10-16_030614.json", "f62f4d660109fff5d3d207a024299f3c254fbfb2c603ddc7fc62b40b38760442")
OUT = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2)
app = modal.App(APP, image=modal.Image.debian_slim(python_version="3.12"))


@app.function(cpu=1, memory=2048, timeout=600, retries=0, volumes={"/out": OUT.read_only()})
def union(stamp: str, fit_sha: str, fits: list) -> dict:
    od = Path("/out/own_all_rs_uncut") / stamp
    if hashlib.sha256((od / "fit_record.json").read_bytes()).hexdigest() != fit_sha:
        raise SystemExit("the Volume fit record differs from the git record's fit_record_sha256")
    per = {}
    for f in fits:
        b = (od / "training" / f"{f['variant']}__{f['readout']}__f{f['fold']}.json").read_bytes()
        if hashlib.sha256(b).hexdigest() != f["training_manifest_sha256"]:
            raise SystemExit(f"{f['variant']} {f['readout']} f{f['fold']}: training manifest differs from the fit record")
        ms = json.loads(b)
        if len(ms) != f["mains"] or len(set(ms)) != len(ms):
            raise SystemExit(f"{f['variant']} {f['readout']} f{f['fold']}: manifest size differs from fit.fits mains, or repeats a main")
        per.setdefault(f"{f['variant']}|{f['readout']}", {})[f["fold"]] = ms
    out = {}
    for k, v in sorted(per.items()):
        u = sorted(set().union(*v.values()))
        cnt = {m: sum(m in set(x) for x in v.values()) for m in u}
        out[k] = {"distinct_training_mains": len(u), "per_fold_mains": {str(f): len(x) for f, x in sorted(v.items())},
                  "mains_by_number_of_fold_models": {str(n): sum(c == n for c in cnt.values()) for n in range(1, 6)}, "union_mains": u}
    return out


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/own-all-loo"):
        raise SystemExit("uncommitted tracked changes, or HEAD is not on origin")
    fb = git("show", f"HEAD:{FIT_GIT[0]}")
    if hashlib.sha256(fb).hexdigest() != FIT_GIT[1]:
        raise SystemExit("the git fit record differs from its pin")
    fr = json.loads(fb)
    fits = [{k: f[k] for k in ("variant", "readout", "fold", "mains", "training_manifest_sha256")} for f in fr["fit"]["fits"]]
    res = union.remote(fr["stamp (PT)"], fr["fit"]["fit_record_sha256"], fits)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    note = {"label": "own_all_rs{,25,10}_uncut_L48 distinct training mains (union of the 5 fold models' training manifests), per variant x readout (APPEND-ONLY NOTE; 4f / 0d v80, 10-16); keys / counts only",
            "stamp (PT)": stamp, "writer": {"commit": git("rev-parse", "HEAD").decode().strip(), "app": APP},
            "fit": {"git_record": {"path": FIT_GIT[0], "sha256": FIT_GIT[1]}, "fit_record_sha256": fr["fit"]["fit_record_sha256"]},
            "training_manifest_sha256": {f"{f['variant']}|{f['readout']}|f{f['fold']}": f["training_manifest_sha256"] for f in fits},
            "results": res}
    out = REPO / "experiments" / "2026-10-14_p2_family_expansion" / "results" / f"own_all_rs_uncut_training_mains_union_NOTE_{stamp}.json"
    out.open("x").write(json.dumps(note, indent=1))
    print(json.dumps({k: (v["distinct_training_mains"], v["per_fold_mains"]) for k, v in res.items()} | {"path": str(out.relative_to(REPO))}))
