"""POST HOC (RULE.md, this folder): the frozen Llama-3.3-70B fit (q2's run 2026-10-12_143129) scores the simulated honest
runs read by run_extract_simhonest_llama.py, as simulated and edited. Modeled on q2's run_score_fix_llama.py (the Olmo copy's code, Llama's names, layers and pins): one Modal CPU
container, saved activations and probes only, no fit, no model. The probes come from the laptop (S3 copies, pinned by
sha256), so no other experiment's Volume is mounted.

Per run of fold g and saved layer l: every nested model with g as its test fold, (g, v), every nested model with g as
its validation fold, (f, g), and the outer model g score the run's rows as ((x - mean) / std) . w + b in float32 from
the float16 rows; the run score is the mean over its rows, as the fit pools them (RULE.md: both nested roles, as the
stored base scores hold). Gates (RULE.md), checked locally before anything is written:
  (1) the smoke's 4 real honest runs, re-extracted, re-score to q2's stored nested and outer scores (<= 1e-3);
  (2) in every edited run, each output before its first changed output (the Qwen3-32B scoring record's
      first_changed_output) sits at the same token and scores as in the as-simulated run (outer model, <= 1e-3).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_q3_reader_matched_fa/run_score_simhonest_olmo.py --fit-dir <S3 copy of run 2026-10-12_110549> \\
      --sim-runs <sim_honest_scores_2026-10-07_124717/runs.json> --eh-scoring <edithonest_scoring_2026-10-12_022927.json>
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3rmfa_llama-s_kill-2026-11-13"
OUT = "probe-shift_aswin-surya_sub-q3rmfa_acts_kill-2026-11-13"
RUN = "2026-10-13_q3rmfa_simhonest_llama33-70b_r2j_l60-70_v1"
SIM_LABEL, ED_LABEL, EXTRA_LABEL = "sim-honest", "sim-honest-edited", "real-honest-extra"
SAVE_LAYERS, TOL = (60, 70), 1e-3
PIN = {"runs.json": "ad10475afccc285bde39a389a17cb8b278a15d670c5d673933a094c6ffbc5254",
       "probes_L60.npz": "7f08e5462bcf302e11fec341f30042aeb374cdebae21a590d14fd339a99aa99c",
       "probes_L70.npz": "c108606d8d0cf655be817958b85a1cff7e38a4a0273ecaddab0bec96b103cae2",
       "scores_L60.npz": "6d21272535104de7cf6aa179cde98070dea0cbc823fc0aee9639763c8938906b",
       "scores_L70.npz": "f58f71edc48333cc362d3814fb83880b1abab74d3270b5a358ce3385f8b16c0c",
       "sim_runs": "f420ed2401dcc776cbbcfd414bbe9d6d1132a3bbd012b529a33aba7fa5ecd068",
       "eh_scoring": "bcb57de46be2a724fd78174cf08824fda1aec4d30e2d1cc3535ba1fe488afffd"}

app = modal.App(APP)
out_vol = modal.Volume.from_name(OUT, version=2).read_only()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, volumes={"/out": out_vol}, cpu=8, memory=32768, timeout=3600, retries=0)
def score(probes: dict, folds: dict) -> dict:
    """probes: {layer: npz bytes}; folds: {label: {run key: fold}} for the runs to score (labels sim-honest,
    sim-honest-edited, real-honest-extra and, from the smoke run, real-honest). Returns per label and run key: nested
    [l][5][5] (filled where the run's fold is the test or the validation fold, NaN elsewhere), outer [l], and per-row
    outer scores with their (step, action, token)."""
    import io

    import numpy as np
    Z = {int(l): dict(np.load(io.BytesIO(b))) for l, b in probes.items()}
    res = {}
    for run, labels in ((RUN, (SIM_LABEL, ED_LABEL, EXTRA_LABEL)), (RUN + "_smoke", ("real-honest",))):
        d = Path("/out") / run
        for lab in labels:
            want = folds.get(lab, {})
            for p in sorted((d / "meta" / lab).glob("*.json")) if (d / "meta" / lab).exists() else []:
                m = json.loads(p.read_text())
                if m["key"] not in want:
                    continue
                f = want[m["key"]]
                kept = [q for q in m["positions"] if q["kept"]]
                rec = {"n_rows": m["n_rows"], "rows": [[q["step"], q["action"], q["token"]] for q in kept], "nested": {}, "outer": {}, "row_outer": {}}
                for l in SAVE_LAYERS:
                    x = np.asarray(np.load(d / "by_layer" / f"L{l}" / f"{m['chunk']}.npy", mmap_mode="r")[m["row_start"]: m["row_start"] + m["n_rows"]], np.float32)
                    if x.shape != (m["n_rows"], 8192) or x.shape[0] == 0 or x.shape[0] != len(kept):
                        raise ValueError("a scoring slice has the wrong shape")
                    z = Z[l]
                    nest = [[float("nan")] * 5 for _ in range(5)]
                    for a in range(5):
                        for v in range(5):
                            if a != v and f in (a, v) and len(x):
                                k = f"output_end__task_nested_{a}_{v}__"
                                nest[a][v] = float(np.mean((((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])).astype(np.float64)))
                    k = f"output_end__task_outer_{f}__"
                    so = (((x - z[k + "mean"]) / z[k + "std"]) @ z[k + "w"] + np.float32(z[k + "b"])).astype(np.float64) if len(x) else np.zeros(0)
                    rec["nested"][str(l)], rec["outer"][str(l)] = nest, float(so.mean()) if len(so) else float("nan")
                    rec["row_outer"][str(l)] = so.tolist()
                res.setdefault(lab, {})[m["key"]] = rec
    return res


@app.local_entrypoint()
def main(fit_dir: str, sim_runs: str, eh_scoring: str, smoke_record: str):
    import numpy as np
    for f in (Path(__file__).resolve(), HERE / "RULE.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    fit = Path(fit_dir)
    files = {"runs.json": fit / "data" / "runs.json", "sim_runs": Path(sim_runs)} | {f"probes_L{l}.npz": fit / f"probes_L{l}.npz" for l in SAVE_LAYERS} \
        | {f"scores_L{l}.npz": fit / f"scores_L{l}.npz" for l in SAVE_LAYERS}
    for role, p in files.items():
        if sha(p) != PIN[role]:
            raise SystemExit(f"{p} is not the pinned {role}")
    if sha(eh_scoring) != PIN["eh_scoring"]:
        raise SystemExit("not the pinned Qwen3-32B edited-honest scoring record (first_changed_output)")
    ehs = json.loads(Path(eh_scoring).read_text())
    checks = [json.loads(p.read_text()) for p in sorted((HERE / "results").glob("extract_check_llama_2*.json"))]
    if not checks or sum(checks[-1]["problems"].values()) or checks[-1]["run"] != RUN or checks[-1]["meta files by label"] != {SIM_LABEL: 437, ED_LABEL: 381, EXTRA_LABEL: 10}:
        raise SystemExit("scoring needs a passing completeness check of the full extraction (RULE.md)")
    smoke = json.loads(Path(smoke_record).read_text())  # the smoke launch record: which real honest runs it read
    base = json.loads(files["runs.json"].read_text())
    bidx = {r["key"]: i for i, r in enumerate(base)}
    srun = [r for r in json.loads(files["sim_runs"].read_text()) if r["arm"] == "sim-honest"]
    edited = set(ehs["edited_keys"])
    # the extraction's filenames by key come from its metas; the folds from the runs files (#36's frozen main-task folds)
    xrun = [r for r in json.loads(files["sim_runs"].read_text()) if r["arm"] == "real-honest" and not r["in_base"]]
    folds = {SIM_LABEL: {r["key"]: int(r["fold"]) for r in srun}, ED_LABEL: {r["key"]: int(r["fold"]) for r in srun if r["key"] in edited},
             EXTRA_LABEL: {r["key"]: int(r["fold"]) for r in xrun},
             "real-honest": {r["key"]: int(base[bidx[r["key"]]]["fold"]) for r in smoke["cohort"]["base"]}}
    res = score.remote({l: (fit / f"probes_L{l}.npz").read_bytes() for l in SAVE_LAYERS}, folds)
    gates, vals = {}, {}
    # gate 1: the smoke's real honest runs against q2's stored scores
    worst = 0.0
    for r in smoke["cohort"]["base"]:
        i, rec = bidx[r["key"]], res["real-honest"][r["key"]]
        f = int(base[i]["fold"])
        for l in SAVE_LAYERS:
            z = np.load(fit / f"scores_L{l}.npz")
            st_n, st_o = z["output_end__mean__task_nested"][:, :, i], z["output_end__mean__task_outer"][i]
            mine = np.array(rec["nested"][str(l)])
            m = np.array([[a != v and f in (a, v) for v in range(5)] for a in range(5)])  # both roles
            if not (np.isfinite(mine[m]).all() and np.isfinite(st_n[m]).all() and np.isfinite(rec["outer"][str(l)]) and np.isfinite(st_o)):
                worst = float("inf")
                continue
            worst = max(worst, float(np.abs(mine[m] - st_n[m]).max()), abs(rec["outer"][str(l)] - float(st_o)))
    n_smoke = len(res.get("real-honest", {}))
    gates["smoke_real_honest_rescore_matches_stored"] = worst <= TOL and n_smoke == len(smoke["cohort"]["base"]) == 4
    vals["smoke_rescore_worst_abs_diff"], vals["smoke_runs_rescored"] = worst, n_smoke
    # gate 2: prefix outputs of every edited run
    fco = ehs["first_changed_output"]
    worst2, mism, compared, runs_with_prefix = 0.0, 0, 0, 0
    for k, rec in res[ED_LABEL].items():
        a = res[SIM_LABEL][k]
        s0, a0 = fco[k]
        pre = [j for j, (s, ac, _) in enumerate(rec["rows"]) if (s, ac) < (s0, a0)]
        pa = {(s, ac): (j, t) for j, (s, ac, t) in enumerate(a["rows"])}
        runs_with_prefix += bool(pre)
        # the prefix position SETS must be equal: (step, action, token) of every kept output before the first change
        mism += sorted(tuple(x) for x in rec["rows"] if (x[0], x[1]) < (s0, a0)) != sorted(tuple(x) for x in a["rows"] if (x[0], x[1]) < (s0, a0))
        for j in pre:
            s, ac, t = rec["rows"][j]
            if (s, ac) not in pa or pa[(s, ac)][1] != t:
                mism += 1
                continue
            for l in SAVE_LAYERS:
                worst2 = max(worst2, abs(rec["row_outer"][str(l)][j] - a["row_outer"][str(l)][pa[(s, ac)][0]]))
            compared += 1
    gates["edited_prefix_rows_match"] = mism == 0 and worst2 <= TOL and compared > 0
    vals |= {"prefix_rows_compared": compared, "prefix_rows_position_mismatch": mism, "prefix_worst_abs_diff": worst2,
             "edited_runs_with_a_prefix_row": runs_with_prefix}
    gates["counts"] = (sorted(res.get(SIM_LABEL, {})) == sorted(folds[SIM_LABEL]) and len(folds[SIM_LABEL]) == 437
                       and sorted(res.get(ED_LABEL, {})) == sorted(edited) and len(edited) == 381
                       and sorted(res.get(EXTRA_LABEL, {})) == sorted(folds[EXTRA_LABEL]) and len(xrun) == 10)
    req = np.array([[[a != v and g in (a, v) for v in range(5)] for a in range(5)] for g in range(5)])  # [g][a][v]: required slots
    gates["required_slots_finite"] = all(np.isfinite(np.array(rec_["nested"][str(l)])[req[folds[lab][k]]]).all() and np.isfinite(rec_["outer"][str(l)])
                                         for lab in (SIM_LABEL, ED_LABEL, EXTRA_LABEL) for k, rec_ in res[lab].items() for l in SAVE_LAYERS)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"rule": "RULE.md (this folder)", "git": head, "run": RUN, "inputs sha256": dict(PIN) | {"eh_scoring": sha(eh_scoring), "smoke_record": sha(smoke_record)},
           "gates": gates | {"all": all(gates.values())}, "gate values": vals,
           "n": {lab: len(v) for lab, v in res.items()}}
    if not rec["gates"]["all"]:
        print(json.dumps(rec, indent=1))
        raise SystemExit("a scoring gate FAILED: nothing written")
    arrs = {}
    for lab, rs in ((SIM_LABEL, srun), (ED_LABEL, srun), (EXTRA_LABEL, xrun)):
        keys = [r["key"] for r in rs]
        arrs[f"{lab}__keys"] = np.array(keys)
        for l in SAVE_LAYERS:
            N = np.full((5, 5, len(keys)), np.nan); O = np.full(len(keys), np.nan)
            for j, r in enumerate(rs):
                if r["key"] in res[lab]:
                    N[:, :, j] = np.array(res[lab][r["key"]]["nested"][str(l)])
                    O[j] = res[lab][r["key"]]["outer"][str(l)]
            arrs[f"L{l}__{lab}__nested"], arrs[f"L{l}__{lab}__outer"] = N, O
    npz = HERE / "results" / f"simhonest_scores_llama_{stamp}.npz"
    with open(npz, "xb") as fh:
        np.savez(fh, **arrs)
    rec |= {"scores_file": npz.name, "scores_sha256": sha(npz),
            "keys_order": "per label, the runs.json order of arm sim-honest (sim-honest, sim-honest-edited: NaN where unchanged) or of the real honest runs outside the fit"}
    with open(dest := HERE / "results" / f"simhonest_scoring_llama_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(dest)
