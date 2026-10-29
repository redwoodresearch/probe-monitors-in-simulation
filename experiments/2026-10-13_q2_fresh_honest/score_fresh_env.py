"""Fresh-task honest runs: the per-environment diagnostic (POST HOC), AMENDMENT_ENV_2026-10-14.md revision 2 (2588336e) as
written. It runs only after the all-stage verdict (score_fresh.py --stages 1,...,7) is committed. score_fresh._analyze does the
loading, the rescore gates and the per-run scores (keep=True), unchanged; this file adds the per-environment measures, the
stratified task bootstrap and the readings. Modal CPU; numbers only, no text.

  --analysis 4-7    the confirmatory analysis (held out from the auto_workflow selection)
  --analysis 1-7    beside it, labelled as including the selection data (stages 1-3)
  --test            a pipeline test: the parity batch as the fresh runs, with fresh and cohort environment labels permuted
                    by a fixed seed, so no real per-environment rate is computed; written to results/, never printed

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' --with numpy modal run --detach \\
      experiments/2026-10-13_q2_fresh_honest/score_fresh_env.py --analysis 4-7
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
sys.path.insert(0, str(HERE))  # score_fresh.py beside this file (in the container: /root)
APP = "probe-shift_aswin-surya_sub-q2-fresh_scoreenv_kill-2026-11-13"
AMENDMENT, AMENDMENT_COMMIT = HERE / "AMENDMENT_ENV_2026-10-14.md", "2588336e"
AW, B, ALPHAS, SEED = "auto_workflow", 2000, (0.01, 0.05), 20261014

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_python_source("probe_shift")
         .add_local_file(HERE / "score_fresh.py", "/root/score_fresh.py"))


VOLS = {"/acts": modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only(),  # score_fresh.VOLS
        "/seed": modal.Volume.from_name("probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12", version=2).read_only(),
        "/q3w": modal.Volume.from_name("probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", version=2).read_only(),
        "/g3w": modal.Volume.from_name("probe-shift_aswin-surya_sub-q2-g3_work_kill-2026-11-12", version=2).read_only(),
        "/df": modal.Volume.from_name("probe-shift_aswin-surya_sub-whysteps_work_kill-2026-11-13", version=2).read_only()}


def env_rows(kept, fresh, base, universe, env_of_task, planned, all654):
    """The amendment's measures, bootstrap and readings for every primary row; kept = score_fresh._analyze(keep=True)'s arrays."""
    import numpy as np

    import score_fresh as sf
    pairs = sf.PAIRS
    envs = sorted(set(env_of_task[t] for t in universe) | {env_of_task[r["main_task"]] for r in base})
    if AW not in envs:
        raise ValueError("auto_workflow is not among the environments")
    f_tasks = {e: sorted(t for t in universe if env_of_task[t] == e) for e in envs}
    folds = sorted({r["fold"] for r in base})
    if folds != [0, 1, 2, 3, 4] or len(envs) != 10:
        raise ValueError(f"expected 5 folds and 10 environments, got {len(folds)} and {len(envs)}")
    c_tasks = {(f, e): sorted({r["main_task"] for r in base if r["fold"] == f and env_of_task[r["main_task"]] == e}) for f in folds for e in envs}
    if any(not v for v in c_tasks.values()):
        raise ValueError("an empty (fold, environment) cell in the cohort")
    # multiplicities: row 0 = point estimate (all ones), rows 1..B = draws (amendment "Interval": the call order is fixed)
    g_f, g_c = np.random.default_rng([SEED, 11]), np.random.default_rng([SEED, 12])
    mF = {e: np.ones((B + 1, len(f_tasks[e]))) for e in envs}
    mC = {k: np.ones((B + 1, len(v))) for k, v in c_tasks.items()}
    for b in range(1, B + 1):
        for e in envs:
            n = len(f_tasks[e])
            if n:
                mF[e][b] = np.bincount(g_f.integers(0, n, n), minlength=n)
        for f in folds:
            for e in envs:
                n = len(c_tasks[f, e])
                mC[f, e][b] = np.bincount(g_c.integers(0, n, n), minlength=n)
    fidx = {e: {t: j for j, t in enumerate(f_tasks[e])} for e in envs}
    cidx = {k: {t: j for j, t in enumerate(v)} for k, v in c_tasks.items()}
    ftask = [r["main_task"] for r in fresh]

    def ci(x):
        x = x[1:][np.isfinite(x[1:])]
        return [float(v) for v in np.quantile(x, [0.025, 0.975], method="linear")] if len(x) else [None, None]

    out = {}
    for name in sf.PRIMARY:
        k = kept[name]
        SF, elig, S_c, fold_c, task_c, c_elig = k["SF"], k["elig"], k["S_c"], k["fold_c"], k["task_c"], k["c_elig"]
        row = {"ineligible fresh runs (analysed stages, planned - eligible)": int(planned - elig.sum()), "planned": planned,
               "ineligible fresh runs over all 654 (the verdict record)": all654.get(name)}
        gate_fail = planned - elig.sum() > 0.05 * planned
        for a in ALPHAS:
            thr = k["thr"][a]
            # per-task numerators (per model) and denominators
            dF = {e: np.zeros(len(f_tasks[e])) for e in envs}
            nF = {e: np.zeros((len(pairs), len(f_tasks[e]))) for e in envs}
            for i in np.flatnonzero(elig):
                e = env_of_task[ftask[i]]
                j = fidx[e][ftask[i]]
                dF[e][j] += 1
                for m, (f, v) in enumerate(pairs):
                    nF[e][m, j] += SF[f, v, i] > thr[f, v]
            dC = {kk: np.zeros(len(v)) for kk, v in c_tasks.items()}
            nC = {kk: np.zeros((len(pairs), len(v))) for kk, v in c_tasks.items()}
            for i in np.flatnonzero(c_elig):
                f, t = int(fold_c[i]), str(task_c[i])
                e = env_of_task[t]
                j = cidx[f, e][t]
                dC[f, e][j] += 1
                for m, (f2, v) in enumerate(pairs):
                    if f2 == f:
                        nC[f, e][m, j] += S_c[f2, v, i] > thr[f2, v]
            FA_f, FA_c, nM = {}, {}, {}
            drops = 0
            for e in envs:
                denF = mF[e] @ dF[e]  # [B+1]
                rf = np.stack([mF[e] @ nF[e][m] for m in range(len(pairs))]) / np.where(denF > 0, denF, np.nan)  # [20, B+1]
                rc, okm = [], []
                for m, (f, v) in enumerate(pairs):
                    den = mC[f, e] @ dC[f, e]
                    rc.append((mC[f, e] @ nC[f, e][m]) / np.where(den > 0, den, np.nan))
                    okm.append(den > 0)
                rc, okm = np.stack(rc), np.stack(okm)  # [20, B+1]
                drops += int((~okm[:, 1:]).sum())
                with np.errstate(invalid="ignore"):
                    cnt = okm.sum(0)
                    FA_c[e] = np.where(cnt > 0, np.nansum(np.where(okm, rc, 0), 0) / np.maximum(cnt, 1), np.nan)
                    FA_f[e] = np.where((cnt > 0) & (denF > 0), np.nansum(np.where(okm, rf, 0), 0) / np.maximum(cnt, 1), np.nan)
                nM[e] = int(cnt[0])
            D = {e: FA_f[e] - FA_c[e] for e in envs}
            # one shared mask: e is defined in a draw iff both of its rates are; an undefined e leaves both medians and,
            # for auto_workflow, every one of its quantiles (D, FA_fresh, FA_cohort, E_fresh, E_cohort)
            ok = {e: np.isfinite(D[e]) for e in envs}
            for e in envs:
                FA_f[e], FA_c[e] = np.where(ok[e], FA_f[e], np.nan), np.where(ok[e], FA_c[e], np.nan)
            others = [e for e in envs if e != AW]
            medD = np.array([np.median([D[e][b] for e in others if ok[e][b]]) if any(ok[e][b] for e in others) else np.nan for b in range(B + 1)])
            medC = np.array([np.median([FA_c[e][b] for e in others if ok[e][b]]) if any(ok[e][b] for e in others) else np.nan for b in range(B + 1)])
            E_f, E_c = D[AW] - medD, FA_c[AW] - medC
            aw_undef = int((~ok[AW][1:]).sum())
            pt = lambda x: float(x[0] * 100) if np.isfinite(x[0]) else None  # noqa: E731
            iv = lambda x: [None if v is None else v * 100 for v in ci(x)]  # noqa: E731
            res = {"by environment": {e: {"models used (|M_e|, point)": nM[e], "fresh tasks": len(f_tasks[e]),
                                          "FA_fresh (points)": [pt(FA_f[e])] + iv(FA_f[e]), "FA_cohort (points)": [pt(FA_c[e])] + iv(FA_c[e]),
                                          "D (points)": [pt(D[e])] + iv(D[e])} for e in envs},
                   "E_fresh (points)": [pt(E_f)] + iv(E_f), "E_cohort (points)": [pt(E_c)] + iv(E_c),
                   "draws with auto_workflow undefined": aw_undef, "model drops across draws": drops,
                   "undefined draws by environment": {e: int((~ok[e][1:]).sum()) for e in envs},
                   "environments undefined at the point estimate": [e for e in envs if not ok[e][0]]}
            lbD, lbEf, lbEc = res["by environment"][AW]["D (points)"][1], res["E_fresh (points)"][1], res["E_cohort (points)"][1]
            if gate_fail:
                reading = "no reading: more than 5% of the analysed planned runs ineligible"
            elif aw_undef > 0.01 * B or None in (lbD, lbEf, lbEc) or res["by environment"][AW]["D (points)"][0] is None:
                reading = "no reading: auto_workflow undefined"
            elif lbEc > 0 and lbD <= 0:
                reading = "cohort excess, no demonstrated fresh increase"
            elif lbEc > 0 and lbD > 0:
                reading = ("cohort excess and an auto_workflow-specific fresh increase" if lbEf > 0
                           else "cohort excess and a fresh increase not shown to be specific")
            elif lbD > 0 and lbEf > 0:
                reading = "auto_workflow-specific fresh increase, no demonstrated cohort excess"
            else:
                reading = "neither / inconclusive"
            row[str(a)] = res | {"reading (marginal, unadjusted; POST HOC)": reading}
        out[name] = row
    return out


@app.function(image=image, cpu=16, memory=131072, timeout=3600, retries=0, volumes=VOLS)
def analyze(fresh: list[dict], base: list[dict], universe: list[str], planned: int, pre_left: int, env_of_task: dict, all654: dict) -> dict:
    try:
        import score_fresh as sf
        main_out, kept = sf._analyze(fresh, base, universe, planned, False, pre_left, keep=True)
        return {"rescore gates": main_out["gates"], "sha256": main_out["sha256"],
                "rows": env_rows(kept, fresh, base, universe, env_of_task, planned, all654)}
    except Exception as e:  # noqa: BLE001 — type and location only, never text
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"analyze failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}") from None


@app.local_entrypoint()
def main(analysis: str = "", test: bool = False):
    import random

    import score_fresh as sf
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    if git("log", "-1", "--format=%h", "--abbrev=8", "--", str(AMENDMENT)) != AMENDMENT_COMMIT:
        raise SystemExit("the amendment changed after revision 2")
    verdicts = [x for x in git("ls-files", "results").splitlines() if Path(x).name.startswith("score_fresh_s1234567_")]
    if not test and len(verdicts) != 1:
        raise SystemExit(f"{len(verdicts)} all-stage verdict records (score_fresh_s1234567_*.json) committed in HEAD; need exactly one")
    all654 = {}
    if not test:
        v = json.loads((HERE / verdicts[0]).read_text())
        if v["universe"]["planned fresh runs"] != 654 or not v["universe"]["the verdict run (all seven stages)"]:
            raise SystemExit("the verdict record is not the all-stage run")
        all654 = {n: v["rows"][n]["fresh runs left out"] + v["rows"][n]["never staged (invalid, no log, or errored in the run list), all rows"]
                  for n in sf.PRIMARY}
    stages = {"4-7": ["4", "5", "6", "7"], "1-7": [str(k) for k in range(1, 8)]}.get(analysis)
    if not test and not stages:
        raise SystemExit("--analysis 4-7 or 1-7")
    runs36 = json.loads(sf.BASE_RUNS.read_text())
    env_of_task = {}
    for r in runs36:
        if env_of_task.setdefault(r["main_task"], r["environment"]) != r["environment"]:
            raise SystemExit("a cohort main task in two environments")
    cells = json.loads(sf.TASKS.read_text())["cells"]
    for c in cells:
        if env_of_task.setdefault(c["main_task"], c["environment"]) != c["environment"]:
            raise SystemExit("a main task in two environments")
    base = [{"key": r["key"], "main_task": r["main_task"], "fold": r["fold"], "i36": i, "environment": r["environment"]}
            for i, r in enumerate(runs36) if r["arm"] == sf.H]
    labels = ["parity"] if test else stages
    batches = ["parity"] if test else [f"s{k}" for k in stages]
    fresh, inputs, pre_left = [], {}, 0
    for b, lab in zip(batches, labels):  # score_fresh.main's own input checks and fields
        st_p, qc_p, sc_p = (sf.one(f"stage_2026-10-13_q2_fresh_{b}_inputs_v1_*.json"), sf.one(f"check_2026-10-13_q2_fresh_{b}_qwen3-32b_l7_v1_*.json"),
                            sf.one(f"extract_check_2026-10-13_q2_fresh_{b}_seedcall_seed-oss-36b_v1_*.json"))
        rl_p = sf.one(f"fresh_runs_fresh1013_stage{lab}_*.json")
        for p in (st_p, qc_p, sc_p, rl_p):
            inputs[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        st, qc, sc, rl = (json.loads(p.read_text()) for p in (st_p, qc_p, sc_p, rl_p))
        if any(qc["problems"].values()) or qc["non-finite (chunk indexes)"] or qc["non-finite recount (first and last layer)"]:
            raise SystemExit(f"{b}: Qwen3-32B check problems")
        if any(sc["problems"].values()) or sc["non-finite values: every layer (chunk indexes)"] or sc["non-finite values: first and last layer recounted"]:
            raise SystemExit(f"{b}: Seed check problems")
        bad = sorted({str(s) for s in qc["failing shards (their rows are not used)"]} | {str(s) for s in qc["shards without a reproduction run"]})
        errored = set(rl["head"]["errored_keys"])
        pre_left += sum(st["excluded"].values()) + st["rows without a log"] + st.get("errored rows, not staged", 0)
        arm = "real-honest-parity" if b == "parity" else "real-honest-fresh"
        for x in st["per run"]:
            fresh.append({"run_id": x["run_id"], "main_task": x["main_task"], "environment": x["environment"], "transcript": x["transcript"],
                          "arm": arm, "stage": lab, "qwen_run": f"2026-10-13_q2_fresh_{b}_qwen3-32b_l7_v1",
                          "seed_run": f"2026-10-13_q2_fresh_{b}_seedcall_seed-oss-36b_v1", "eligible before scoring": x["run_id"] not in errored,
                          "qwen bad shards": bad})
    planned = sum(c["stage"] in labels for c in cells)
    universe = sorted({c["main_task"] for c in cells if c["stage"] in labels})
    if test:  # permute environment labels over tasks (fixed seed), fresh runs' labels too: no real per-environment rate
        tasks = sorted(env_of_task)
        perm = [env_of_task[t] for t in tasks]
        random.Random(7).shuffle(perm)
        env_of_task = dict(zip(tasks, perm))
        if "auto_workflow" not in {env_of_task[t] for t in universe}:
            env_of_task[universe[0]] = "auto_workflow"
        for x in fresh:
            x["environment"] = env_of_task[x["main_task"]]
        for r in base:
            r["environment"] = env_of_task[r["main_task"]]
    res = {"POST HOC": True, "rule": f"AMENDMENT_ENV_2026-10-14.md revision 2 ({AMENDMENT_COMMIT})", "git": git("rev-parse", "HEAD"),
           "analysis": "TEST (parity batch, permuted environment labels): not a result" if test else
           ("stages 4-7: held out from the auto_workflow selection (confirmatory)" if analysis == "4-7" else
            "stages 1-7: includes the selection data (stages 1-3)"),
           "label": "POST HOC diagnostic; marginal, unadjusted; preliminary until Aswin reviews it",
           "inputs sha256": inputs | {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (sf.BASE_RUNS, sf.TASKS, AMENDMENT)}}
    res |= analyze.remote(fresh, base, universe, planned, pre_left, env_of_task, all654)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    tag = "test_parity" if test else f"s{analysis.replace('-', 'to')}"
    with open(p := HERE / "results" / f"score_fresh_env_{tag}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")
    if not test:
        for name, row in res["rows"].items():
            r1 = row["0.01"]
            print(f"{name} | 1%: aw D {r1['by environment'][AW]['D (points)']}, E_fresh {r1['E_fresh (points)']}, E_cohort {r1['E_cohort (points)']}"
                  f" -> {r1['reading (marginal, unadjusted; POST HOC)']}")
    else:
        print("test written (not printed):", "rescore gates pass" if all(g["pass"] for g in res["rescore gates"].values()) else "GATE FAIL")
    print(p)
