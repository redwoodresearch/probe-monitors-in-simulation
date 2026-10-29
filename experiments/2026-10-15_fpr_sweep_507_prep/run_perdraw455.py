"""09's per-draw npz for leave-one-recipe-out on the 455 cohort (2e's relay; 0d approved). Schema of the FPR fork's
crossrecipe_2026-10-15_103328_perdraw.npz (its RULE @ 7bc924e08, "### Second commit"), plus task_origin (S_old / S_new) and task_main.
Rows, pools, transformations, thresholds, cohort and draws are run455's (run_common455.py @ the run455 record); draw 0 must equal the
run455 record exactly for every row x point x comparator and arm, or the job stops. Draws 1..10,000 are run455's shared main-level draws.
BLIND: the npz goes to the Volume; the record holds its path, sha256, shapes and gate counts only, never a value.

  PYTHONPATH=src <modal 1.6.1> run --detach experiments/2026-10-15_fpr_sweep_507_prep/run_perdraw455.py
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_common455 as C  # noqa: E402
from run_fprsweep_507 import A3, GRID, draws_main, transform, wmean  # noqa: E402

app = modal.App(C.APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
         .add_local_file(HERE / "run_fprsweep_507.py", "/root/run_fprsweep_507.py").add_local_file(HERE / "run_common455.py", "/root/run_common455.py"))
vol_out = modal.Volume.from_name(C.VOLS["solavg"], version=2)
RUN455 = ("results/common_run455_2026-10-15_101451.json", "c7b348575e5e1a2c2a4c97e192659d3a432d7c99721b0ab996c92e36f306d810")
POINTS = [("run0.01", "run", 0.01), ("run0.02", "run", 0.02), ("run0.05", "run", 0.05), ("run0.1", "run", 0.1),
          ("unit0.001", "unit", 1e-3), ("unit0.00133", "unit", 10 ** -2.875), ("unit0.00178", "unit", 10 ** -2.75)]
COMPS = ["sim_minus_run1", "sim_minus_run2", "sim_minus_real_mean", "run2_minus_run1"]
ARMS = ["run1", "run2", "sim", "real_mean"]


def gaps_of(c):
    return {"sim_minus_run1": c["sim"] - c["run1"], "sim_minus_run2": c["sim"] - c["run2"], "sim_minus_real_mean": c["sim"] - c["real_mean"],
            "run2_minus_run1": c["run2"] - c["run1"]}


def unit_catches(V, z, pools, coh, unit_of_cell, Wm, ucol, fs):
    """run_common455.perdraw's arithmetic (sweep()'s union threshold, any unit strictly above T, equal-weighted mains), all four arms."""
    import numpy as np

    runs = {}
    for i, r in enumerate(V["run"]):
        runs.setdefault(r, []).append(i)
    rmax = {r: float(z[ix].max()) for r, ix in runs.items()}
    allp = np.concatenate([pools[s] for s in sorted(pools)])
    d_ = z[allp][np.argsort(-z[allp], kind="mergesort")]
    n = len(d_)
    mem = {}
    for c in coh:
        mem.setdefault(unit_of_cell[c], []).append(c)
    us = sorted(mem, key=lambda x: ucol[x])
    Wc = Wm[:, [ucol[x] for x in us]]
    out = {}
    for f in fs:
        T = float(d_[min(int(np.floor(f * n + 1e-9)), n - 1)])
        h = {w: np.array([np.mean([float(rmax[f"{w}|{c}"] > T) for c in mem[x]]) for x in us]) for w in A3}
        h["real_mean"] = (h["run1"] + h["run2"]) / 2
        out[f] = {w: wmean(Wc, h[w]) for w in h}
    return out


@app.function(image=image, cpu=(16, 16), memory=(65536, 65536), timeout=3600, retries=0, scaledown_window=2,
              volumes={"/sol": vol_out, "/p2": C.vol["p2"], "/conv": C.vol["conv"]})
def compute(cfg: dict, blobs: dict, meta: dict) -> dict:
    import numpy as np

    rec = json.loads(blobs["run455"])
    t196 = json.loads(blobs["t196"])
    canon = sorted(c["key"] for c in t196["cells"])
    env_t = {c["key"]: c["env"] for c in t196["cells"]}
    main_t = {c["key"]: c["main"] for c in t196["cells"]}
    fin = json.loads(blobs["final"])["result"]
    coh = sorted(fin["cohort"]["cohort_cells"])
    mp = fin["maps"]
    main_of = mp["main_of"]
    reg_new = sorted(set(fin["design"]["primary_universe_mains"]) - set(main_t.values()))
    universe = sorted(set(main_t.values()) | set(reg_new))
    cls = lambda v: "both" if set(v) == {"old", "new"} else {"old": "old-only", "new": "new-only"}[v[0]]  # noqa: E731
    units, W = draws_main(universe, {m: (mp["env_of_main"][m], cls(mp["origin_of_main"][m])) for m in universe})
    ucol = {m: i for i, m in enumerate(units)}
    ref = fin["calibration_records"]["primary"][cfg["pool_reference_key"]]
    sch = ref["schemes"][ref["primary"]]
    pool_old, pool_new = set(sch["pool_keys_old"]), set(sch["pool_keys_new"])
    gi = {nm: next(i for i, g in enumerate(GRID) if abs(np.log10(g) - np.log10(f)) < 1e-9) for nm, lv, f in POINTS if lv == "unit"}
    probes = C.probe_specs(cfg)
    R, D1 = len(probes), W.shape[0]
    gaps = np.full((R, len(POINTS), len(COMPS), D1), np.nan)
    catches = np.full((R, len(POINTS), len(ARMS), D1), np.nan)
    gate = {"compared_values": 0, "mismatch": []}
    for r, p in enumerate(probes):
        U = C.load_probe(p, cfg, "455")
        world = np.array([x.split("|", 1)[0] for x in U["run"]])
        cell = np.array([x.split("|", 1)[1] for x in U["run"]])
        keep = np.flatnonzero((np.isin(cell, coh) & np.isin(world, list(A3))) | np.isin(U["run"], list(pool_old | pool_new)) | (U["source"] == "relaunch")
                              | ((world == "honest_a") & (np.isin(cell, coh) | np.isin(cell, canon))))
        V = {k_: v[keep] for k_, v in U.items()}
        vcell = cell[keep]
        V["stratum"] = np.array(["S_old" if (s in ("S_old", "relaunch")) else "S_new" for s in V["source"]])
        V["cell_stratum"] = {c: ("S_old" if c in env_t else "S_new") for c in set(vcell.tolist())}
        pools = {"S_old": np.flatnonzero(np.isin(V["run"], list(pool_old)) | (V["source"] == "relaunch")), "S_new": np.flatnonzero(np.isin(V["run"], list(pool_new)))}
        have = {w: set(vcell[np.array([x.split("|", 1)[0] == w for x in V["run"]], dtype=bool)].tolist()) for w in A3}
        if [c for c in coh if all(c in have[w] for w in A3)] != coh:
            raise SystemExit(f"{p['label']}: cohort incomplete (the 455 record has all 455 for every row)")
        mains = {c: main_of[c] for c in coh}
        store = {}
        C.run_level(V, np.arange(len(V["run"])), pools, coh, mains, W, ucol, C.ALPHAS, p["agg"], p["folds"] == 5, blocks={"all": coh},
                    catch_fn=lambda a, pc: store.__setitem__(a, pc["all"]))
        z, _, why = transform(V, pools, "union", p["folds"] == 5)
        if why:
            raise SystemExit(f"{p['label']}: transform withheld ({why})")
        uc = unit_catches(V, z, pools, coh, mains, W, ucol, [GRID[gi[nm]] for nm, lv, _ in POINTS if lv == "unit"])
        rp = rec["probes"][p["label"]]
        for i, (nm, lv, f) in enumerate(POINTS):
            if lv == "run":
                c_ = store[f]
                refb = next(q for q in rp["run"]["points"] if abs(q["alpha"] - f) < 1e-12)["blocks"]["all"]
            else:
                c_ = uc[GRID[gi[nm]]]
                refq = rp["unit"][gi[nm]]
                if refq["fpr_target"] != GRID[gi[nm]]:
                    raise SystemExit("unit grid index")
                refb = refq["blocks"]["combined"]
            g_ = gaps_of(c_)
            for j, a in enumerate(ARMS):
                catches[r, i, j] = c_[a]
                gate["compared_values"] += 1
                if float(c_[a][0]) != refb["tpr"][a]["point"]:
                    gate["mismatch"].append(f"{p['label']}|{nm}|catch|{a}")
            for j, k_ in enumerate(COMPS):
                gaps[r, i, j] = g_[k_]
                gate["compared_values"] += 1
                if float(g_[k_][0]) != refb["diff"][k_]["point"]:
                    gate["mismatch"].append(f"{p['label']}|{nm}|gap|{k_}")
    if sorted(p["label"] for p in probes) != sorted(rec["probes"]) or gate["compared_values"] != R * len(POINTS) * (len(ARMS) + len(COMPS)):
        raise SystemExit("draw-0 gate incomplete: rows or comparison count differ from the run455 record")
    if gate["mismatch"]:
        raise SystemExit(f"draw-0 gate failed at {len(gate['mismatch'])} values: {gate['mismatch'][:10]}")
    arrs = {"gaps": gaps, "catches": catches, "row_labels": np.array([p["label"] for p in probes]), "row_recipe": np.array([p["recipe"] for p in probes]),
            "row_family": np.array([p["family"] for p in probes]), "row_readout": np.array([p["readout"] for p in probes]),
            "row_registered": np.array([p["registered"] for p in probes]), "points": np.array([nm for nm, _, _ in POINTS]), "comparators": np.array(COMPS),
            "arms": np.array(ARMS), "task_multiplicities": W[:, [ucol[main_of[c]] for c in coh]], "cohort_tasks": np.array(coh),
            "task_origin": np.array(["S_old" if c in env_t else "S_new" for c in coh]), "task_main": np.array([main_of[c] for c in coh])}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    b = buf.getvalue()
    name = f"perdraw455_{meta['stamp']}"
    path = Path("/sol/xfer/fpr507prep") / name / f"{name}_perdraw.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fh:  # exclusive create: a name collision fails loudly
        fh.write(b)
    vol_out.commit()
    return {"label": "09's per-draw npz for LORO on the 455 cohort (blinded record: no values)", "meta": meta,
            "npz": {"volume": C.VOLS["solavg"], "path": str(path).removeprefix("/sol/"), "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b),
                    "shapes": {k: list(v.shape) for k, v in arrs.items()}, "dtypes": {k: str(v.dtype) for k, v in arrs.items()}},
            "gate_draw0_vs_run455": {"compared_values": gate["compared_values"], "mismatches": 0, "pass": True},
            "draws": "run455's shared main-level draws over the 506-main universe (09's construction); task_multiplicities = each cohort task's main's multiplicity; "
                     "a main's catch is the mean over its cohort tasks (task_main gives the grouping)",
            "schema": "crossrecipe_2026-10-15_103328_perdraw.npz (FPR fork RULE @ 7bc924e08) + task_origin, task_main", "pass": True}


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    cfg = json.loads((HERE / "config" / "common455.json").read_text())
    ucfg = json.loads((C.UNIT / "config" / "inputs_b.json").read_text())
    blobs = {}
    for k in ("t196",):
        spec = ucfg["local"][k]
        b = git("show", spec["git"]) if "git" in spec else Path(spec["path"]).read_bytes()
        if C.sha_b(b) != spec["sha256"]:
            raise SystemExit(f"G1: {k}")
        blobs[k] = b
    fb = git("show", f"{C.FINAL[0]}:{C.FINAL[1]}")
    if C.sha_b(fb) != C.FINAL[2]:
        raise SystemExit("FINAL record sha256")
    blobs["final"] = fb
    rb = (HERE / RUN455[0]).read_bytes()
    if C.sha_b(rb) != RUN455[1]:
        raise SystemExit("run455 record sha256")
    blobs["run455"] = rb
    ob = git("show", cfg["own_a_record"]["git"])
    if C.sha_b(ob) != cfg["own_a_record"]["sha256"]:
        raise SystemExit("G1: own_a record")
    own = json.loads(ob)
    for p in cfg["probes"]:
        if p["old"]["kind"] == "own_a_R":
            p["old"]["files"] = own["saved"]["tight"]["avg"]
        if p["old"]["kind"] == "own_a_C":
            e = own["saved"]["cmd"]["cmdA"]["rows|own_a_tight"]
            p["old"] |= {"path": e["path"], "sha256": e["sha256"]}
    cfg["arms"] = C.arms_from_spec(json.loads(git("show", f"{C.SPEC[0]}:{C.SPEC[1]}")), git)
    rr = git("show", cfg["relaunch_record"]["git"])
    if C.sha_b(rr) != cfg["relaunch_record"]["sha256"]:
        raise SystemExit("G1: relaunch record")
    cfg["relaunch_files"] = {fn: {"path": f"ref/relaunch/sol/{fn}", "sha256": json.loads(rr)["outputs"]["files"][fn]} for fn in cfg["relaunch_names"]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    meta = {"git": git("rev-parse", "HEAD").decode().strip(), "stamp": stamp, "script_sha256": C.sha_b(Path(__file__).read_bytes()),
            "common455_sha256": C.sha_b((HERE / "run_common455.py").read_bytes()), "core_sha256": C.sha_b((HERE / "run_fprsweep_507.py").read_bytes()),
            "run455_record": f"{RUN455[0]} @ {RUN455[1]}", "inputs": {k: C.sha_b(v) for k, v in blobs.items()}}
    out = compute.remote(cfg, blobs, meta)
    blob = json.dumps(out, indent=1).encode()
    dest = HERE / "results" / f"perdraw455_{stamp}.json"
    dest.open("xb").write(blob)
    print(json.dumps({"record": dest.name, "sha256": C.sha_b(blob), "pass": out.get("pass"), "npz_sha256": out["npz"]["sha256"]}), flush=True)
