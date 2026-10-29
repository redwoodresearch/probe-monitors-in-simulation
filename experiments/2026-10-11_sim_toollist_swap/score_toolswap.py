"""Row 4's rescoring: the 196-task impactful-command monitor, frozen, on the edited extractions. Modal CPU.

The monitor is draw 0's five real-trained fold models as saved (`scaleup196_probes_14b33da3d97a_percall/probes_draw0.npz`:
per fold k its layer, the training part's mean and scale, and beta; no refit), each at its frozen threshold (calibrated
on real run 1's read-only calls, recorded in the main per-call file). A call in task T is scored by the model of T's fold,
((x[layer] - mu) / sd) @ beta in float64, and flagged when its score exceeds that fold's frozen threshold, as
run_score_onpolicy.py scores the real->sim, real->real and real->run2 cells. Calls are joined to the main per-call file's
rows by (request id, the command entry's own call number), so only the counted attempts the scorer used enter.

Gates, in order; each must pass before the next condition is read:
1. the unedited extractions reproduce every main per-call score of real->sim and real->real to 6 decimals, and their
   frozen flags;
2. the no-op extraction (20 simulated and 20 run-2 tasks, each tool line spliced to itself) reproduces its rows' main
   per-call scores within 1e-5 and their flags;
then the edited simulated arm and the mirror are scored. Per-call scores and flags are returned; the statistics run
locally (analyze_toolswap.py).

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run score_toolswap.py --main <draw0 per-call json> --out <json>
"""

import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q3toolswap_score_kill-2026-11-08"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
SCORE = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
PROBES = "scaleup196_probes_14b33da3d97a_percall"
ORIGINAL = {"sim_attack@september": ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1",
                                     "2026-10-10_scaleup_l2_segends_v1", "2026-10-10_scaleup_topup_segends_v1"],
            "attack": ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1",
                       "2026-10-10_scaleup_l2_segends_v1", "2026-10-10_scaleup_topup_segends_v1"]}
EDITED = {"noop": "2026-10-11_toolswap_noop_v2", "sim": "2026-10-11_toolswap_sim_v2", "mirror": "2026-10-11_toolswap_mirror_v2"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
score_vol = modal.Volume.from_name(SCORE, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, cpu=8, memory=65536, timeout=3 * 3600, retries=0,
              volumes={"/acts": acts.read_only(), "/score": score_vol.read_only()})
def rescore(main: dict, conditions: list, noop_tasks: dict) -> dict:
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    meta, folds, thr = main["meta"], main["folds"], {int(k): v for k, v in main["frozen_threshold"].items()}
    pz = np.load(Path("/score") / PROBES / "probes_draw0.npz")
    real = sorted((k, layer) for W, k, layer, C in json.loads(str(pz["meta"])) if W == "real")
    if [k for k, _ in real] != list(range(5)) or any(main["choices"][f"real|{k}"][0] != layer for k, layer in real):
        raise ValueError("the saved models are not draw 0's five real-trained models")
    P = {k: (layer, pz[f"real|{k}|mu"], pz[f"real|{k}|sd"], pz[f"real|{k}|beta"]) for k, layer in real}
    by_key = {(m["irid"], m["call"]): i for i, m in enumerate(meta)}

    def complete(run):
        """The run's complete.json epoch count equals its meta files (every staged epoch extracted)."""
        n_meta = len(list((Path("/acts") / run / "meta").glob("*/*.json")))
        done = json.loads((Path("/acts") / run / "complete.json").read_text())["epochs"]
        if n_meta != done:
            raise ValueError(f"{run}: {n_meta} metas vs {done} epochs in complete.json")
        return done

    def index(run, arm):
        """(row of main meta) -> (chunk, row in chunk), for the run's command entries of that arm."""
        out = {}
        for f in sorted((Path("/acts") / run / "meta" / arm).glob("*.json")):
            m = json.loads(f.read_text())
            for c in m["calls"]:
                if c.get("kind", "command") != "command" or c.get("row") is None:
                    continue
                i = by_key.get((c["irid"], c["call"]))
                if i is None:
                    continue
                if i in out:
                    raise ValueError(f"{run}: main row {i} appears twice")
                out[i] = (m["chunk"], m["row_start"] + c["row"])
        return out

    def scores(run, idx):
        """main row -> float64 score with its task's fold model, reading each (chunk, layer) file whole once."""
        jobs = {}
        for i, (chunk, r) in idx.items():
            k = folds[meta[i]["task"]]
            jobs.setdefault((chunk, P[k][0]), []).append((i, r, k))
        res = {}

        def one(job):
            (chunk, layer), items = job
            X = np.load(Path("/acts") / run / "by_layer" / f"L{layer}" / f"{chunk}.npy")
            out = {}
            for i, r, k in items:
                _, mu, sd, beta = P[k]
                out[i] = float(((X[r].astype(np.float64) - mu) / sd) @ beta)
            return out

        with ThreadPoolExecutor(16) as ex:
            for o in ex.map(one, list(jobs.items())):
                res.update(o)
        if not all(np.isfinite(v) for v in res.values()):
            raise ValueError(f"{run}: non-finite scores")
        return res

    def check(res, ref, tol, label):
        miss = [i for i in ref if i not in res]
        if miss:
            raise ValueError(f"{label}: {len(miss)} reference rows not found")
        diff = max(abs(round(res[i], 6) - ref[i][0]) if tol <= 1.1e-6 else abs(res[i] - ref[i][0]) for i in ref)
        flags = sum((res[i] > thr[folds[meta[i]['task']]]) != ref[i][1] for i in ref)
        if not diff <= tol or flags:
            raise ValueError(f"{label}: max score difference {diff}, {flags} flags differ")
        return {"rows": len(ref), "max_abs_difference": diff, "flags_differ": flags}

    ref = {cell: {int(r): (s, f) for r, s, f in zip(v["row"], v["score"], v["flag_frozen"])} for cell, v in main["per_call"].items()}
    out = {"gates": {}, "scores": {}}
    if ({"sim", "mirror"} & set(conditions)) and "noop" not in conditions:
        raise ValueError("the edited conditions are read only after the no-op gate, in the same call")
    acts.reload()
    out["gates"]["complete"] = {run: complete(run) for run in sorted(set(ORIGINAL["attack"]) | {EDITED[c] for c in conditions})}
    # gate 1: unedited extractions reproduce the main per-call scores and flags
    for arm, cell in (("sim_attack@september", "real->sim"), ("attack", "real->real")):
        res = {}
        for run in ORIGINAL[arm]:
            res.update(scores(run, index(run, arm)))
        out["gates"][f"unedited {cell}"] = check(res, ref[cell], 1.000001e-6, f"unedited {cell}")
        out["scores"][f"before {cell}"] = {str(i): res[i] for i in ref[cell]}
    if "noop" in conditions:  # gate 2
        run = EDITED["noop"]
        s_sim, s_r2 = scores(run, index(run, "sim_attack@september")), scores(run, index(run, "attack"))
        # exact coverage: every main per-call row of the preselected no-op tasks, nothing else
        want_sim = {i: ref["real->sim"][i] for i in ref["real->sim"] if meta[i]["task"] in set(noop_tasks["sim"])}
        want_r2 = {i: ref["real->run2"][i] for i in ref["real->run2"] if meta[i]["task"] in set(noop_tasks["run2"])}
        if set(s_sim) != set(want_sim) or set(s_r2) != set(want_r2):
            raise ValueError(f"no-op coverage: sim {len(s_sim)} of {len(want_sim)}, run2 {len(s_r2)} of {len(want_r2)}")
        out["gates"]["noop sim"] = check(s_sim, want_sim, 1e-5, "noop sim")
        out["gates"]["noop run2"] = check(s_r2, want_r2, 1e-5, "noop run2")
        out["gates"]["noop tasks"] = {"sim": len({meta[i]["task"] for i in s_sim}), "run2": len({meta[i]["task"] for i in s_r2})}
    for cond, arm, cell in (("sim", "sim_attack@september", "real->sim"), ("mirror", "attack", "real->real")):
        if cond in conditions:
            run = EDITED[cond]
            res = scores(run, index(run, arm))
            found = sum(i in res for i in ref[cell])
            if found != len(ref[cell]):
                raise ValueError(f"{cond}: {found} of {len(ref[cell])} main rows found in the edited extraction")
            out["scores"][f"after {cell}"] = {str(i): res[i] for i in ref[cell]}
            out["gates"][f"{cond} coverage"] = {"rows_found": found, "rows": len(ref[cell])}
    return out


@app.local_entrypoint()
def main(main_file: str, out: str, noop_tasks: str, conditions: str = "noop,sim,mirror"):
    d = json.loads(Path(main_file).read_text())
    cells = d["draws"][0]["cells"]
    thr = {f["fold"]: f["threshold"] for f in cells["real->sim"]["folds"] if f["threshold_kind"] == "frozen"}
    if len(thr) != 5 or any(thr[f["fold"]] != f["threshold"] for f in cells["real->real"]["folds"] if f["threshold_kind"] == "frozen"):
        raise ValueError("the frozen thresholds are not five, or differ between cells")
    payload = {"meta": d["meta"], "folds": d["folds"], "frozen_threshold": thr, "choices": d["draws"][0]["choices"],
               "per_call": {c: {k: cells[c]["per_call"][k] for k in ("row", "score", "flag_frozen")} for c in ("real->sim", "real->real", "real->run2")}}
    res = rescore.remote(payload, conditions.split(","), json.loads(Path(noop_tasks).read_text()))
    Path(out).write_text(json.dumps(res))
    print(json.dumps(res["gates"], indent=1))
