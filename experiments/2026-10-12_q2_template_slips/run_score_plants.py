"""Template slips, step 2, scoring (PLAN.md addendum 3). One Modal CPU container; saved activations and frozen probes
only, no fit, no model, no text.

The frozen probes of the Qwen3.5 reply-readout run 2026-10-12_011944 (probes_L<l>.npz on its work Volume: per nested
model (f, v) and outer model f, w, b, mean, std) score output_end rows as ((x - mean) / std) . w + b in float32 from the
float16 rows; a run score is the mean over its kept output_end rows (the fit's "mean" pooling).
- `--mode gate` (G1, before any GPU work): re-scores the 939 real e2 runs' rows from Qwen3.5's dump; the local side
  compares them with the stored nested and outer scores (tolerance 1e-3, every layer and model of the run's fold).
- `--mode pilot` / `--mode full`: also scores the extraction's planted and control runs, and reports, per row and per
  model of the run's fold, the largest |new - dump| for the controls' rows (G2) and for each planted run's rows before
  its first changed output (G3), tolerance 0.01 logits. `full` requires every changed run of the staging record and all
  8 controls, and writes the planted runs' nested and outer scores (npz) and a record (json) to results/.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-12_q2_template_slips/run_score_plants.py --mode <gate|pilot|full> --scores-dir <Qwen3.5 run_scores_2026-10-12_011944> \\
      --stage <results/plant_stage_*.json> [--extract-run <run name>]
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-tplslips_score_kill-2026-11-12"
DUMP_VOL, DUMP = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", "2026-10-06_la_segment-end_qwen35-27b_alllayers_v1"
OUT_VOL = "probe-shift_aswin-surya_sub-q2-tplslips_acts_kill-2026-11-12"
WORK_VOL, RUN_ID = "probe-shift_aswin-surya_sub-q2-qwen35-reply_work_kill-2026-11-12", "2026-10-12_011944"
LAYERS, E2, PLANTS, CONTROL = (16, 24, 32, 40, 48, 56, 63), "real-attack-e2", ("e2-gutter-stripped", "e2-note-dropped"), "e2-control"
TOL_G1, TOL_ROW = 1e-3, 0.01
PIN = {"runs.json": "3c028aca246181950e460349067ca27d0fb85bff60bc4c01b518c9b27a234ec1"}

app = modal.App(APP)
dump_vol = modal.Volume.from_name(DUMP_VOL, version=2).read_only()
out_vol = modal.Volume.from_name(OUT_VOL, version=2, create_if_missing=True).read_only()
work_vol = modal.Volume.from_name(WORK_VOL, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/dump": dump_vol, "/out": out_vol, "/work": work_vol}, cpu=8, memory=65536, timeout=3600, retries=0)
def score(e2: list[dict], extract_run: str | None, mode: str) -> dict:
    """e2: {j, key, fold, plants: {plant: first changed [step, action]}, control: bool}."""
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    def meta(root, arm, key):
        p = Path(root) / "meta" / arm / f"{filename(key)}.json"
        if not p.exists():
            return None
        m = json.loads(p.read_text())
        rows = {(q["step"], q["action"]): q["row"] for q in m["positions"] if q["kind"] == "output_end" and q["kept"]}
        return m["chunk"], m["row_start"], rows
    jobs = {}  # (arm, j) -> (root, meta)
    missing = Counter()
    for r in e2:
        jobs[("orig", r["j"])] = ("/dump/" + DUMP, meta("/dump/" + DUMP, E2, r["key"]))
        if jobs[("orig", r["j"])][1] is None:
            raise ValueError("an e2 run has no meta in the dump")
        if mode == "gate":
            continue
        for arm in [*r["plants"], *([CONTROL] if r["control"] else [])]:
            m = meta(f"/out/{extract_run}", arm, r["key"])
            if m is None:
                missing[arm] += 1
            else:
                jobs[(arm, r["j"])] = (f"/out/{extract_run}", m)
    by_j = {r["j"]: r for r in e2}
    n = len(e2)

    def one_layer(layer):
        with np.load(Path("/work") / "runs" / RUN_ID / f"probes_L{layer}.npz") as z:
            P = {k: z[k] for k in z.files}

        def sc(x, name):
            return ((x - P[name + "__mean"]) / P[name + "__std"]) @ P[name + "__w"] + np.float32(P[name + "__b"])
        nested = {a: np.full((5, 5, n), np.nan) for a in ("orig", *PLANTS, CONTROL)}
        outer = {a: np.full(n, np.nan) for a in ("orig", *PLANTS, CONTROL)}
        g = Counter(); worst = {"G2": 0.0, "G3": 0.0}; rowsets = Counter(); orig_rows = {}
        # originals first, then the new runs; one chunk file in memory at a time (whole file: mmap is slow on a Volume)
        for phase in ("orig", "new"):
            todo = sorted(((root, mt[0], arm, j_) for (arm, j_), (root, mt) in jobs.items() if (arm == "orig") == (phase == "orig")))
            cur, block = None, None
            for root, c, arm, j in todo:
                if cur != (root, c):
                    cur, block = (root, c), np.load(Path(root) / "by_layer" / f"L{layer}" / f"{c}.npy")
                _, s0, rows = jobs[(arm, j)][1]
                keys = sorted(rows)
                if not keys:
                    continue
                x = block[[s0 + rows[k] for k in keys]].astype(np.float32)
                f = by_j[j]["fold"]
                per = {"outer": sc(x, f"output_end__task_outer_{f}")}
                outer[arm][j] = float(np.mean(per["outer"].astype(np.float64)))
                for v in range(5):
                    if v != f:
                        per[v] = sc(x, f"output_end__task_nested_{f}_{v}")
                        nested[arm][f, v, j] = float(np.mean(per[v].astype(np.float64)))
                if arm == "orig":
                    orig_rows[j] = (keys, per)
                    continue
                g["nonfinite row scores (new runs)"] += sum(int((~np.isfinite(v_)).sum()) for v_ in per.values())
                rowsets["runs compared"] += 1
                okeys, oper = orig_rows[j]
                if okeys != keys:
                    rowsets["runs whose output_end rows differ from the dump's"] += 1
                    continue
                first = tuple(by_j[j]["plants"][arm]) if arm in PLANTS else None
                sel = [i for i, k in enumerate(keys) if first is None or k < first]
                if not sel:
                    continue
                gate = "G2" if arm == CONTROL else "G3"
                for v, s in per.items():  # every nested model of the run's fold, and its outer model
                    dv = np.abs(s[sel] - oper[v][sel])
                    if not np.isfinite(dv).all():
                        g[f"{gate} nonfinite differences"] += int((~np.isfinite(dv)).sum()); continue
                    worst[gate] = max(worst[gate], float(dv.max()))
                g[f"{gate} rows compared ({arm})"] += len(sel)
            block = None
        return layer, {"nested": {a: v.tolist() for a, v in nested.items()}, "outer": {a: v.tolist() for a, v in outer.items()},
                       "worst": worst, "counts": dict(g), "rowsets": dict(rowsets)}
    with ThreadPoolExecutor(len(LAYERS)) as pool:
        res = dict(pool.map(one_layer, LAYERS))
    return {"missing metas by arm": dict(missing), "layers": {str(l): v for l, v in res.items()}}


@app.local_entrypoint()
def main(mode: str, scores_dir: str, stage: str, extract_run: str = "", pilot_list: str = ""):
    import numpy as np
    me = Path(__file__).resolve()
    for f in (me, HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    if mode not in ("gate", "pilot", "full") or (mode != "gate" and not extract_run) or (mode == "pilot") != bool(pilot_list):
        raise SystemExit("--mode gate; pilot with --extract-run and --pilot-list; or full with --extract-run")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    d, st_p = Path(scores_dir), Path(stage)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    if sha(d / "runs.json") != PIN["runs.json"]:
        raise SystemExit("not the Qwen3.5 run's runs.json")
    runs = json.loads((d / "runs.json").read_text()); st = json.loads(st_p.read_text())
    idx = [i for i, r in enumerate(runs) if r["arm"] == E2]
    plants = {}
    for x in st["per run"]:
        plants.setdefault(x["key"], {})[x["plant"]] = x["first changed (step, action)"]
    e2 = [{"j": j, "key": runs[i]["key"], "fold": int(runs[i]["fold"]), "plants": plants.get(runs[i]["key"], {}),
           "control": runs[i]["key"] in st["controls"]} for j, i in enumerate(idx)]
    keys = {runs[i]["key"] for i in idx}
    if set(plants) - keys or set(st["controls"]) - keys or len(set(st["controls"])) != 8:
        raise SystemExit("the staging record names runs that are not this run list's e2 runs, or not 8 distinct controls")
    res = score.remote(e2, extract_run or None, mode)
    fold = np.array([r["fold"] for r in e2]); n = len(e2); m = np.zeros((5, 5, n), bool)
    for j, f in enumerate(fold):
        m[f, :, j] = True; m[f, f, j] = False
    g1 = {}
    for l in LAYERS:
        z = np.load(d / f"scores_L{l}.npz")
        sn, so = z["output_end__mean__task_nested"][:, :, idx], z["output_end__mean__task_outer"][idx]
        mn, mo = np.array(res["layers"][str(l)]["nested"]["orig"]), np.array(res["layers"][str(l)]["outer"]["orig"])
        if not (np.isfinite(sn[m]).all() and np.isfinite(mn[m]).all() and np.isfinite(so).all() and np.isfinite(mo).all()):
            raise SystemExit(f"L{l}: a missing stored or re-scored e2 score")
        g1[str(l)] = [float(np.abs(mn[m] - sn[m]).max()), float(np.abs(mo - so).max())]
    rec = {"plan": "PLAN.md addendum 3 (this folder)", "git": head, "mode": mode, "extraction run": extract_run or None,
           "inputs sha256": {"runs.json": PIN["runs.json"], st_p.name: sha(st_p)} | {f"scores_L{l}.npz": sha(d / f"scores_L{l}.npz") for l in LAYERS},
           "G1: max |re-scored - stored| e2 score by layer [nested, outer] (tolerance 1e-3)": g1,
           "G1 passes": all(max(v) <= TOL_G1 for v in g1.values())}
    if mode != "gate":
        L = res["layers"]
        rec |= {"missing metas by arm": res["missing metas by arm"],
                "G2: max |control row - dump row| by layer (tolerance 0.01)": {l: L[l]["worst"]["G2"] for l in L},
                "G3: max |planted prefix row - dump row| by layer (tolerance 0.01)": {l: L[l]["worst"]["G3"] for l in L},
                "rows compared by layer": {l: L[l]["counts"] for l in L}, "row sets by layer": {l: L[l]["rowsets"] for l in L}}
        # the runs that must be present: the pilot's list, or every staged run
        if mode == "pilot":
            pl = json.loads(Path(pilot_list).read_text()); expect = [tuple(x) for x in pl["affected"] + pl["controls"]]
        else:
            expect = [(a, r["key"]) for r in e2 for a in r["plants"]] + [(CONTROL, r["key"]) for r in e2 if r["control"]]
        k_of = {r["key"]: r["j"] for r in e2}
        absent = [x for x in expect if not np.isfinite(np.array(L["63"]["outer"][x[0]])[k_of[x[1]]])]
        need = {CONTROL, *PLANTS}
        compared = all(L[l]["counts"].get(f"G{2 if a == CONTROL else 3} rows compared ({a})", 0) > 0 for l in L for a in need)
        clean = all(L[l]["counts"].get(k, 0) == 0 for l in L for k in ("nonfinite row scores (new runs)", "G2 nonfinite differences", "G3 nonfinite differences"))
        rec |= {"expected runs absent": len(absent), "rows compared for the controls and both plants": compared, "no nonfinite row score or difference": clean}
        rec["G2 and G3 pass"] = not absent and compared and clean and all(max(L[l]["worst"].values()) <= TOL_ROW for l in L) and all(
            L[l]["rowsets"].get("runs whose output_end rows differ from the dump's", 0) == 0 for l in L)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if mode == "full":
        want = {p: sum(p in r["plants"] for r in e2) for p in PLANTS} | {CONTROL: sum(r["control"] for r in e2)}
        have = {a: int(np.isfinite(np.array(res["layers"]["63"]["outer"][a])).sum()) for a in want}
        rec["runs scored vs staged"] = {a: [have[a], want[a]] for a in want}
        if have != want or res["missing metas by arm"]:
            raise SystemExit(f"not every staged run was extracted: {rec['runs scored vs staged']}, missing {res['missing metas by arm']}")
        npz = HERE / "results" / f"plant_scores_{stamp}.npz"
        np.savez(npz, run_index=np.array(idx), **{f"L{l}__{a}__{k}": np.array(res["layers"][str(l)][k][a]) for l in LAYERS
                                                  for a in (*PLANTS, CONTROL) for k in ("nested", "outer")})
        rec |= {"scores file": npz.name, "scores sha256": sha(npz)}
    with open(dest := HERE / "results" / f"plant_score_{mode}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
