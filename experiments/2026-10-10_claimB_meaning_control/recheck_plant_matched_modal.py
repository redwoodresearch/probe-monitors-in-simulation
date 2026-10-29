"""Review check for #527 (thread Lzva): the meaning control's planted check compared each copy's mean over its OWN kept
outputs; in a run where the plant moved the 32k cut, the two copies kept different outputs. Here, on Modal CPU: for every
such run, refit the frozen probe exactly as run_meaning_plant_modal.score does (CPU, float64), check that it reproduces the
stored per-run shift, and recompute the run- and site-level shifts over the outputs both copies kept (the intersection).
Locally: reproduce the published summary from the result file (asserted), replace those runs' values, and recompute.
Writes <out dir>/recheck_plant_matched_<stamp>.json (numbers, provenance, inputs' sha256).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run \\
      experiments/2026-10-10_claimB_meaning_control/recheck_plant_matched_modal.py --result <meaning_plant result json> --out-dir <dir>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_meaning_plant_modal as MP  # noqa: E402

M, P = MP.M, MP.P
app = modal.App("probe-shift_aswin-surya_sub-claimb_review-checks_kill-2026-11-10")
image = MP.image.add_local_file(HERE / "run_meaning_plant_modal.py", "/root/run_meaning_plant_modal.py")


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": P.vol}, timeout=1800, retries=0)
def recheck(env_to_fold: dict) -> list:
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.q1_eval import step_table

    torch.set_num_threads(8)
    files = sorted(x for x in (Path("/acts") / MP.OUT / "runs").glob("*.npz") if not x.name.endswith(".tmp.npz"))
    todo = []
    for i, f in enumerate(files):  # the result's per_run order is these files sorted by name
        z, y = np.load(f), np.load(Path("/acts") / M.OUT / "runs" / f.name)
        pm, nm = json.loads(str(z["meta"])), json.loads(str(y["meta"]))["none"]
        if pm["r2"]["steps"] != nm["steps"]:
            todo.append((i, f.name, pm, nm, z, y))
    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    arm_of = np.array([pruns[i]["arm"] for i in t["run"]])
    efold = np.array([env_to_fold[pruns[i]["environment"]] for i in t["run"]])
    out = []
    for i, name, pm, nm, z, y in todo:
        e = pm["environment"]
        rec = {"index": i, "file": name, "cell": pm["cell"], "environment": e, "main_task": pm["main_task"],
               "outputs_planted": len(pm["r2"]["steps"]), "outputs_none": len(nm["steps"])}
        sp, sn = [tuple(x) for x in pm["r2"]["steps"]], [tuple(x) for x in nm["steps"]]
        both = sorted(set(sp) & set(sn))
        rec["outputs_both"] = len(both)
        for label, layer in (("L63", 63), ("A_layer", M.A_LAYER[e])):
            X = np.load(Path("/acts") / M.PACK / f"L{layer}_output_end.npy")
            tr = np.isin(arm_of, [M.REAL, M.SIM]) & (efold != env_to_fold[e])
            Xa = torch.from_numpy(X[tr]).double()
            Ya = torch.from_numpy((arm_of[tr] == M.SIM).astype(float)[:, None]).double()
            mean, std = Xa.mean(0), Xa.std(0, unbiased=False)
            std = std.where(std > 0, std.new_ones(()))
            Xa = (Xa - mean) / std * M.C**0.5
            tol, mi, prev = 1e-4 * min(M.C**0.5, 1.0), 5000, None
            while True:  # the same refit loop as the scoring job
                W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol)
                if evals < mi or evals == prev or mi * 4 > 80000:
                    break
                mi, prev = mi * 4, evals

            def s(x):
                return ((((torch.from_numpy(x).double() - mean) / std * M.C**0.5) @ W + b)[:, 0]).numpy()

            a, b0 = s(z[f"r2_L{layer}"]), s(y[f"none_L{layer}"])
            ia, ib = [sp.index(k) for k in both], [sn.index(k) for k in both]
            site = tuple(pm["r2"]["site"])
            rec[f"{label}|run_own"] = float(a.mean() - b0.mean())
            rec[f"{label}|run_matched"] = float(a[ia].mean() - b0[ib].mean())
            rec[f"{label}|site"] = float(a[sp.index(site)] - b0[sn.index(site)]) if site in sn else None
            del Xa, Ya, X
        out.append(rec)
    return out


def summary(rows, key, seed=b"meaning-plant-r2"):
    import numpy as np

    rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))  # as run_meaning_plant_modal.score
    tasks = sorted({r["main_task"] for r in rows})
    by = {k: [r for r in rows if r["main_task"] == k] for k in tasks}
    draws = [[r for k in rng.choice(tasks, len(tasks)) for r in by[k]] for _ in range(M.BOOT)]
    use = [r for r in rows if r[key] is not None]
    boot = np.array([np.mean([r[key] for r in d if r[key] is not None]) for d in draws])
    return {"mean": float(np.mean([r[key] for r in use])), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "toward_sim": sum(r[key] > 0 for r in use)}


@app.local_entrypoint()
def main(result: str, out_dir: str):
    res = json.loads(Path(result).read_text())
    per_run = res["per_run"]
    for key in ("L63|run", "A_layer|run"):  # reproduce the published numbers before changing anything
        got = summary(per_run, key)
        pub = res["results"][key]
        assert abs(got["mean"] - pub["mean"]) < 1e-12 and all(abs(x - y) < 1e-12 for x, y in zip(got["ci95"], pub["ci95"])), key
    folds = json.loads((M.REPO / "splits" / "folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
    fixed = recheck.remote(folds)
    matched = [dict(r) for r in per_run]
    for f in fixed:
        r = matched[f["index"]]
        assert (r["main_task"], r["environment"]) == (f["main_task"], f["environment"]), "run order mismatch"
        for label in ("L63", "A_layer"):
            if abs(f[f"{label}|run_own"] - r[f"{label}|run"]) > 1e-4:  # the CPU refit must reproduce the stored shift
                raise ValueError(f"{f['cell']} {label}: refit gives {f[label + '|run_own']}, stored {r[label + '|run']}")
            r[f"{label}|run"] = f[f"{label}|run_matched"]
    out = {"published": {k: res["results"][k] for k in ("L63|run", "A_layer|run")},
           "matched_outputs": {k: summary(matched, k) for k in ("L63|run", "A_layer|run")}, "changed_runs": fixed,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "inputs_sha256": {Path(result).name: hashlib.sha256(Path(result).read_bytes()).hexdigest()}}
    path = Path(out_dir) / f"recheck_plant_matched_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("published", "matched_outputs")}, indent=1))
    print(json.dumps(fixed, indent=1))
    print("->", path)
