"""RULE.md "Scoring": rescore every rule-1 command row (attack: the 196-task cache; honest A, B, simulated honest: the simhon
extractions; honest C: its extraction) with the stored I1, I2 and DT monitor models, each row by its task's fold model, in
float64; gate on reproducing every stored per-call score; write one per-row npz per monitor in 2e's schema. No statistic here
(analyze_callfacet.py). Modal CPU; the local entrypoint checks git and launches.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with pyyaml --with numpy modal run --detach \\
      experiments/2026-10-14_p2_callfacet/run_callfacet_score.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
Q3 = HERE.parent / "2026-10-08_q3_onpolicy_figures" if modal.is_local() else Path("/root")
sys.path.insert(0, str(Q3))
import run_score_onpolicy as S  # (unchanged)

APP = "probe-shift_aswin-surya_sub-p2probes_callscore_kill-2026-11-13"
OUTVOL = "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"
KEY, DT, CX = "scaleup196_probes_14b33da3d97a_percall", "dt_attack_fits_00f4994d6c1e", "2026-10-11_honest_c_segends_v1"
DESIGNS = ("destructive_primary", "transfer_primary", "destructive_secondary", "transfer_secondary")
ARM = {"real": "run1", "run2": "run2", "sim": "sim", "honA": "honest_a", "honB": "honest_b", "honC": "honest_c", "simH": "sim_honest"}
app = modal.App(APP)
out_vol = modal.Volume.from_name(OUTVOL, create_if_missing=True, version=2)
image = S.image.uv_pip_install("pyyaml").add_local_file(Q3 / "run_score_onpolicy.py", "/root/run_score_onpolicy.py")


@app.function(image=image, cpu=16, memory=65536, timeout=3600, retries=0,
              volumes={"/acts": S.acts_vol.read_only(), "/score": S.score_vol.read_only(), "/out": out_vol})
def score(cfg: dict, ext: dict, pilot: str, stamp: str) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    tmp = Path("/tmp/pilot.json")
    tmp.write_text(pilot)
    folds = S.frozen_folds(tmp)
    d = Path("/score") / KEY
    pz = np.load(d / "probes_draw0.npz")
    models = {"I1": {}, "I2": {}}  # monitor -> {fold k: (layer, mu, sd, beta)}
    for W, k, layer, _ in json.loads(str(pz["meta"])):
        models["I1" if W == "real" else "I2"][k] = (layer, *(np.asarray(pz[f"{W}|{k}|{n}"], np.float64) for n in ("mu", "sd", "beta")))
    dtj, dtrows = {}, {}
    for des in DESIGNS:
        dtj[des] = json.loads((Path("/score") / DT / des / "main.json").read_text())
        dtrows[des] = json.loads((Path("/score") / DT / des / "rows.json").read_text())
        models[f"DT_{des}"] = {int(p.split("|")[1]): (v["layer"], *(np.asarray(v[n], np.float64) for n in ("mu", "sd", "beta")))
                               for p, v in dtj[des]["probes"].items()}
    for m, ks in models.items():
        if sorted(ks) != list(range(5)) or any(a.shape != (5120,) or not np.isfinite(a).all() for v in ks.values() for a in v[1:]) \
                or any((v[2] <= 0).any() for v in ks.values()):
            raise ValueError(f"{m}: not five finite fold models of width 5120 with positive scales")
    draw = json.loads((d / "draw_00000.json").read_text())
    for W, m in (("real", "I1"), ("sim", "I2")):
        if any(draw["choices"][f"{W}|{k}"][0] != models[m][k][0] for k in range(5)):
            raise ValueError(f"{m}: saved layers differ from the draw's choices")
    for des in DESIGNS:
        if any(dtj[des]["choices"][f"real|{k}"][0] != models[f"DT_{des}"][k][0] for k in range(5)):
            raise ValueError(f"DT {des}: saved layers differ from the fit's choices")

    main = S.load(cfg | {"_rows_only": True}, folds)["rows"]
    c = Path("/score") / cfg["x_cache"]
    if json.loads((c / "rows.json").read_text())["rows_sha256"] != S.rows_sha(main):
        raise ValueError("the cache holds other rows")
    for des in DESIGNS:
        if dtrows[des]["world"] != [r[0] for r in main] or dtrows[des]["fold"] != [folds[r[1]] for r in main] or len(dtrows[des]["y"]) != len(main):
            raise ValueError(f"DT {des}: rows.json is not row-aligned with the cache")
    hon = [r for r in S.load(ext | {"_rows_only": True}, folds)["rows"] if r[0] in ("honA", "honB", "simH")]
    worlds_c = {w: v for w, v in ext["worlds"].items() if w not in ("honA", "honB", "simH")} | {"honB": f"honest_b@{CX}"}
    cfg_c = ext | {"label": "scaleup196_honc", "extractions": ext["extractions"] + [CX], "worlds": worlds_c}
    hc = [("honC",) + tuple(r[1:]) for r in S.load(cfg_c | {"_rows_only": True}, folds)["rows"] if r[0] == "honB" and r[2][1] == CX]
    if not hc or not hon:
        raise ValueError("no honest rows")
    rows = main + hon + hc
    n_main = len(main)
    fold = np.array([folds[r[1]] for r in rows])
    layers = sorted({v[0] for ks in models.values() for v in ks.values()})
    scores = {m: np.full(len(rows), np.nan) for m in models}
    Xc = np.load(c / "X.npy", mmap_mode="r")
    if Xc.shape != (cfg["n_layers"], n_main, cfg["hidden"]):
        raise ValueError(f"cache shape {Xc.shape}")
    groups = {}
    for i, r in enumerate(rows[n_main:], start=n_main):
        groups.setdefault((r[2], r[3]), ([], []))
        groups[(r[2], r[3])][0].append(i), groups[(r[2], r[3])][1].append(r[4])
    for L in layers:
        X = np.empty((len(rows), 5120), np.float16)
        X[:n_main] = Xc[L]

        def one(item, X=X, L=L):
            ((vol, run), chunk), (pos, sel) = item
            X[pos] = np.load(Path("/" + vol) / run / "by_layer" / f"L{L}" / f"{chunk}.npy")[sel]

        with ThreadPoolExecutor(32) as ex:
            list(ex.map(one, list(groups.items())))
        for m, ks in models.items():
            for k, (layer, mu, sd, beta) in ks.items():
                if layer == L:
                    ix = np.flatnonzero(fold == k)
                    scores[m][ix] = ((X[ix].astype(np.float64) - mu) / sd) @ beta
        del X
    if any(not np.isfinite(v).all() for v in scores.values()):
        raise ValueError("non-finite scores")

    gate = {}
    stored = [("I1", draw["cells"], ("real->real", "real->sim", "real->run2"))] + [("I2", draw["cells"], ("sim->sim", "sim->real"))]
    stored += [(f"DT_{des}", dtj[des]["cells"], ("real->real", "real->sim", "real->run2")) for des in DESIGNS]
    for m, cells, names in stored:
        for nm in names:
            pc = cells[nm]["per_call"]
            diff = float(np.abs(np.round(scores[m][np.array(pc["row"])], 6) - np.array(pc["score"], np.float64)).max())
            gate[f"{m}|{nm}"] = {"calls": len(pc["row"]), "max_abs_difference_6dp": diff}
            if not diff <= 1.000001e-6:
                raise ValueError(f"gate: {m} {nm} does not reproduce the stored scores (max difference {diff})")

    # the loader's rows are (world, task, ("acts", run), chunk, absolute row, y, matched) (run_score_onpolicy.py l.469): irid, call
    # index, epoch and turn come from the extraction meta, joined by (run, chunk, absolute row), as analyze_POSTHOC_A2_batches does
    info = {}
    for run in sorted({r[2][1] for r in rows}):
        for f in sorted((Path("/acts") / run / "meta").glob("*/*.json")):
            mm = json.loads(f.read_text())
            for cc in mm["calls"]:
                if cc.get("kind", "command") == "command" and cc["row"] is not None:
                    info[(run, mm["chunk"], mm["row_start"] + cc["row"])] = (str(cc["irid"]), int(cc["call"]), int(mm["epoch"]), int(cc["turn"]))
    meta = [info.get((r[2][1], r[3], r[4])) for r in rows]
    if any(x is None for x in meta):
        raise ValueError(f"{sum(x is None for x in meta)} rows without extraction meta")
    keys = [f"{ARM[r[0]]}|{r[1]}" for r in rows]
    order = sorted(range(len(rows)), key=lambda i: (keys[i], meta[i][2], meta[i][3], meta[i][1]))  # chronological within each run
    unit, seen = np.zeros(len(rows), np.int64), {}
    for i in order:
        unit[i] = seen.get(keys[i], 0)
        seen[keys[i]] = unit[i] + 1
    y = np.array([r[5] for r in rows], np.int64)
    o = np.array(order)
    out, files = Path("/out") / "2026-10-14_p2_callfacet" / stamp, {}
    out.mkdir(parents=True, exist_ok=True)
    for m, ks in models.items():
        lay = np.array([ks[f][0] for f in fold], np.int64)
        extra = {}
        if m.startswith("DT_"):
            des = m[3:]
            pos = np.full(len(rows), -1, np.int64)  # -1: no class label (honest rows)
            pos[:n_main] = np.array(dtrows[des]["y"], np.int64) * np.array(dtrows[des]["in_design"], np.int64) \
                - (1 - np.array(dtrows[des]["in_design"], np.int64))  # 1 class positive, 0 design negative, -1 outside the design
            extra["dt_class"] = pos
        p = out / f"{m}__rule1_command.npz"
        np.savez(p, key=np.array(keys)[o], unit=unit[o], call_index=np.array([x[1] for x in meta], np.int64)[o], irid=np.array([x[0] for x in meta])[o],
                 epoch=np.array([x[2] for x in meta], np.int64)[o], turn=np.array([x[3] for x in meta], np.int64)[o], score=scores[m][o],
                 fold=fold.astype(np.int64)[o], layer=lay[o], impactful=y[o], **{k: v[o] for k, v in extra.items()})
        files[m] = {"path": str(p.relative_to("/out")), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rows": len(rows)}
    out_vol.commit()
    counts = {}
    for r in rows:
        counts[f"{ARM[r[0]]}|y{r[5]}"] = counts.get(f"{ARM[r[0]]}|y{r[5]}", 0) + 1
    return {"gate": gate, "files": files, "counts": counts, "layers": layers, "rows_sha256": {"attack": S.rows_sha(main)},
            "dt_design_counts": {des: dtrows[des]["counts"] for des in DESIGNS},
            "models": {m: {str(k): {"layer": v[0]} for k, v in ks.items()} for m, ks in models.items()}}


@app.local_entrypoint()
def main():
    import yaml

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    head = git("rev-parse", "HEAD")
    for f in ("RULE.md", "run_callfacet_score.py"):
        if git("status", "--porcelain", "--", f) or not git("ls-files", "--", f):
            raise SystemExit(f"{f} is not committed unchanged")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit("HEAD is not pushed")
    cfg = yaml.safe_load((Q3 / "config" / "score_scaleup196.yaml").read_text())
    ext = yaml.safe_load((Q3 / "config" / "score_scaleup196_simhon.yaml").read_text())
    pilot = (HERE.parents[1] / cfg["pilot_tasks"]).read_text()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    r = score.remote(cfg, ext, pilot, stamp)
    rec = {"label": "PRELIMINARY; RULE.md in this folder", "git": head, "stamp (PT)": stamp, "app": APP,
           "blobs": {f: git("rev-parse", f"{head}:experiments/2026-10-14_p2_callfacet/{f}") for f in ("RULE.md", "run_callfacet_score.py")}} | r
    dest = HERE / "results" / f"callfacet_score_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("gate", "counts", "files")}, indent=1))
    print(dest.name)
