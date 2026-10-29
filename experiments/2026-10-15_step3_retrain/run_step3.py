"""Step 3, item 0 (RULE.md): own-real probes retrained on run1 / run2 / a per-task mix, on famexp's pass-1 rows and folds.

Jobs (each local launch triggers ONE remote function; --detach keeps everything):
  gate: refit own_frac100 at L48 (R and C, folds 0-4) through THIS file's fit path; it must be bit-equal to famexp pass 1
        (fit 2026-10-14_175636): weights and held-out open rows (sha256 and arrays), sealed sim files by sha256 only (never opened).
        Also records: the spec's folds vs the Sol #36 git map, per-task world coverage counts, timings and GPU.
  fit:  the three step-3 recipes (r1only, r2only, r12) x L16/32/48 x R/C x folds 0-4; only after RULE.md is committed (checked).
Rows: 2e's pinned row code (run_avg_transfer.py / run_correctedb.py @ 4d1eec572), materialized and sha256-checked by run_famexp.py's
import; fitter: torch_probe @ 341ee1585 (C 0.01, certified), the same H100 type as pass 1. Writes only to the step-3 work Volume.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach <this file> --job gate|fit
"""

import hashlib
import inspect
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
FAM = REPO / "experiments/2026-10-14_p2_family_expansion"
APP = {"gate": "probe-shift_aswin-surya_sub-step3_gate_kill-2026-10-22", "fit": "probe-shift_aswin-surya_sub-step3_fit_kill-2026-10-22"}
SPEC, P1 = "2026-10-14_163027", "2026-10-14_175636"  # famexp's prep spec and pass-1 fit run
SPEC_SHA = "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61"  # prep_2026-10-14_163027.json result.spec_sha256
SOL36 = ("aae9ec5f1", "experiments/2026-10-14_own_s_09/sol36_fold_map.json")
RECIPES = ("r1only", "r2only", "r12")
LAYERS, C = (16, 32, 48), 0.01
if modal.is_local():
    sys.path.insert(0, str(FAM))
    import run_famexp as rf  # materializes the pinned 2e row code + fitter into /tmp/famexp_build, refusing any byte difference

    BUILD = rf.BUILD
else:
    BUILD = Path("/tmp/famexp_build")


def _fit_one(tp, X, y, tr, st, dev):
    """Certified fit on rows tr, standardized by rows st (float64). Returns weights, cert, seconds."""
    import torch
    t0 = time.time()
    Xs = X[torch.as_tensor(st, device=dev)].double()
    mean, std = Xs.mean(0), Xs.std(0, unbiased=False)
    std = std.where(std > 0, std.new_ones(()))
    del Xs
    Xt = (X[torch.as_tensor(tr, device=dev)].double() - mean) / std
    W, b, cert = tp.fit(Xt, torch.as_tensor(y[tr], device=dev, dtype=torch.float64), C, allow_uncertified=True, return_certificate=True)
    torch.cuda.synchronize()
    return (W, b, mean, std), {k: cert[k] for k in ("n", "max_grad", "certified", "n_evals")}, time.time() - t0


def _train_tasks(rname, fold, f):
    """own_a's training tasks for outer fold f: tasks in folds != f with run1 or honest_a ROWS in this readout, seeded order."""
    t = sorted({u.split("|", 1)[1] for u, g in zip(rname.tolist(), fold.tolist(), strict=True) if g != f and u.split("|", 1)[0] in ("run1", "honest_a")})
    random.Random(f"famexp|own_a|f{f}").shuffle(t)
    return t


if modal.is_local():  # the two copied functions must be famexp's, byte for byte (the gate tests this file's path)
    for fn in (_fit_one, _train_tasks):
        if inspect.getsource(fn) != inspect.getsource(getattr(rf, fn.__name__)):
            raise SystemExit(f"{fn.__name__} differs from run_famexp.py's")

app = modal.App(APP["fit" if modal.is_local() and "fit" in sys.argv else "gate"])  # the launch's job names the app
WORK = modal.Volume.from_name("probe-shift_aswin-surya_sub-step3_work_kill-2026-11-13", create_if_missing=True, version=2)
FAMV = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2).read_only()
BV = modal.Volume.from_name("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2).read_only()
img = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3", "transformers==5.19.0", "jinja2", "safetensors==0.8.0")
       .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift").add_local_dir(BUILD, "/root/x2e"))
VOLS = {"/b": BV, "/fam": FAMV, "/out": WORK}


def _x2e():
    sys.path.insert(0, "/root/x2e")
    import run_correctedb as rc  # it imports run_avg_transfer as rat, as in 2e's containers
    import torch_probe_pinned as tp

    return rc, rc.rat, tp


def _j(p: Path, obj) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as fh:  # write-once
        fh.write(json.dumps(obj, indent=1, default=float))
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _spec() -> dict:
    """famexp's spec, copied once onto the step-3 Volume (sha256-checked both sides); every job reads the copy."""
    dst = Path("/out/step3") / SPEC / f"spec_{SPEC}.json"
    if not dst.exists():
        raw = (Path("/fam/famexp") / SPEC / f"spec_{SPEC}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != SPEC_SHA:
            raise SystemExit("famexp spec sha256 differs from its prep record")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.open("xb").write(raw)
        WORK.commit()
    raw = dst.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPEC_SHA:
        raise SystemExit("step-3 spec copy sha256 differs")
    return json.loads(raw)


def _positives(role, task, fold, f, T_f, rcode):
    """Within-task matched positives for outer fold f (RULE.md): per training task t, m_t = min(#run1 rows, #run2 rows); one fixed
    subsample of m_t rows per (t, run), drawn by default_rng([20261015, f, readout, j, run]) (j = t's index in sorted T_f). r1only
    takes every task's run1 subsample, r2only its run2 subsample, r12 the run1 subsample on an exact seeded half of the tasks and the
    run2 subsample on the rest. So every task has the same positive count in all three recipes, and r12 equals r1only on its run1 half."""
    import numpy as np
    order = sorted(T_f)
    half = order[:]
    random.Random(f"step3|r12|f{f}").shuffle(half)
    half = set(half[: len(order) // 2])  # these tasks give r12 their run1 rows
    sub = {"r1only": [], "r2only": [], "r12": []}
    for j, t in enumerate(order):
        idx = {w: np.flatnonzero((task == t) & (role == w) & (fold != f)) for w in ("run1", "run2")}
        m = min(len(idx["run1"]), len(idx["run2"]))
        pick = {w: np.sort(np.random.default_rng([20261015, f, rcode, j, wc]).choice(idx[w], m, replace=False)) for wc, w in enumerate(("run1", "run2"), 1)}
        sub["r1only"].append(pick["run1"])
        sub["r2only"].append(pick["run2"])
        sub["r12"].append(pick["run1"] if t in half else pick["run2"])
    return {r: np.concatenate(v) for r, v in sub.items()}, {"r12_run1_tasks": len(half), "r12_run1_task_set_sha256": hashlib.sha256("\n".join(sorted(half)).encode()).hexdigest()}


def _group3(readout: str, L: int, folds: list, recipes: list, run: str) -> dict:
    """Load famexp's own rows once at (readout, L); fit each recipe x fold; score every held-out row of fold f (all worlds;
    sim rows written apart, never read here); write npz, weights and a completion record (resume-safe, write-once)."""
    import numpy as np
    import torch
    rc, rat, tp = _x2e()
    t0, dev = time.time(), "cuda"
    od = Path("/out/step3") / SPEC / f"run_{run}"
    done = od / "groups" / f"{readout}_L{L}_f{'-'.join(map(str, folds))}_{'-'.join(recipes)}.json"
    if done.exists():
        try:
            return json.loads(done.read_text())
        except json.JSONDecodeError:
            raise SystemExit(f"{done.name}: truncated completion record: stop and inspect") from None
    spec = _spec()
    own = {"layout": "b", "kind": "output_end_mean" if readout == "R" else "call_end", "map": spec["map"], "arm_of": spec["arm_of"],
           "run_of": spec["run_of"]} | ({"skip": spec["skip"]} if readout == "C" else {})
    Xo, no, fo = (rat._b_rows if readout == "R" else rat._b_rows_cmd)(own, L, spec["fot"])
    ci, ir = rc._row_calls(own, cmd=(readout == "C"))
    X = torch.from_numpy(Xo).to(dev)
    load_s = time.time() - t0
    role = np.array([u.split("|", 1)[0] for u in no])
    task = np.array([u.split("|", 1)[1] for u in no])
    have = {w: {t for t, r in zip(task.tolist(), role.tolist(), strict=True) if r == w} for w in ("honest_a", "run1", "run2")}
    rcode = {"R": 0, "C": 1}[readout]
    out, fits = {"run": run, "readout": readout, "layer": L, "folds": folds, "recipes": recipes, "load_seconds": load_s, "rows": int(len(no)),
                 "gpu": torch.cuda.get_device_name(0)}, []
    for f in folds:
        T_f = {t for t in have["honest_a"] & have["run1"] & have["run2"] if spec["fot"][t] != f}  # the same training tasks for all 3 recipes
        pos_by, coin = _positives(role, task, fo, f, T_f, rcode)
        if len({len(v) for v in pos_by.values()}) != 1:
            raise SystemExit("within-task matching failed: unequal positive counts")
        neg = np.flatnonzero((role == "honest_a") & (fo != f) & np.isin(task, sorted(T_f)))
        for r in recipes:
            if r == "own_frac100":  # famexp's exact path (the gate)
                y = np.where(role == "run1", 1.0, np.where(role == "honest_a", 0.0, np.nan))
                tt = set(_train_tasks(no, fo, f))
                tr = np.flatnonzero(~np.isnan(y) & (fo != f) & np.array([t in tt for t in task.tolist()]))
                info, name = {"train_tasks": len(tt)}, f"own_frac100_L{L}"
            else:
                pos = np.sort(pos_by[r])
                y = np.full(len(no), np.nan)
                y[neg], y[pos] = 0.0, 1.0
                tr = np.sort(np.concatenate([neg, pos]))
                info = {"train_tasks": len(T_f), "n_pos": int(len(pos)), "n_neg": int(len(neg)), **coin,
                        "pos_index_sha256": hashlib.sha256(np.ascontiguousarray(pos, dtype=np.int64).tobytes()).hexdigest(),
                        "train_task_set_sha256": hashlib.sha256("\n".join(sorted(T_f)).encode()).hexdigest()}
                name = f"step3_{r}_L{L}"
            m_, cert, secs = _fit_one(tp, X, y, tr, tr, dev)
            sc = np.flatnonzero(fo == f)  # held-out rows of fold f, every world
            z = tp.decision((X[torch.as_tensor(sc, device=dev)].double() - m_[2]) / m_[3], m_[0], m_[1]).cpu().numpy()[:, 0]
            if not np.isfinite(z).all():
                raise ValueError(f"non-finite scores {name} f{f}")
            hs = {}
            for part, sel in (("open", role[sc] != "sim"), ("sim_sealed", role[sc] == "sim")):
                hs[part] = rc._save_schema(od / part / f"{name}__{readout}__L{L}__f{f}.npz", no[sc][sel], np.asarray(ci)[sc][sel], np.asarray(ir)[sc][sel],
                                           fo[sc][sel], z[sel], layer=np.full(int(sel.sum()), L, dtype=np.int64))
            from safetensors.numpy import save_file
            wp = od / "weights" / f"{name}__{readout}__L{L}__f{f}.safetensors"
            wp.parent.mkdir(parents=True, exist_ok=True)
            save_file({k: np.ascontiguousarray(t.cpu().numpy()) for k, t in zip(("W", "b", "mean", "std"), m_, strict=True)}, str(wp))
            hs["weights"] = hashlib.sha256(wp.read_bytes()).hexdigest()
            fits.append({"variant": name, "recipe": r, "fold": f, "layer": L, "readout": readout, "seconds": secs, **cert, **info, "sha256": hs})
            print(f"[{readout} L{L}] {name} f{f}: {secs:.1f} s, certified {cert['certified']}", flush=True)
    out["fits"], out["seconds"] = fits, time.time() - t0
    _j(done, out)
    WORK.commit()
    return out


def _fit_path_sha() -> str:
    """sha256 over the source of every function on the fit path plus the pinned row code and fitter: the gate's pass binds to it."""
    h = hashlib.sha256()
    for fn in (_x2e, _spec, _positives, _group3, _fit_one, _train_tasks):
        h.update(inspect.getsource(fn).encode() + b"\0")
    for name in ("run_avg_transfer.py", "run_correctedb.py", "torch_probe_pinned.py", "corrected_map.json", "merge.json", "avg_v1.json", "t196.json"):
        h.update(name.encode() + b"\0" + ((BUILD if modal.is_local() else Path("/root/x2e")) / name).read_bytes())
    return h.hexdigest()


def _gate_compare(run: str, readout: str) -> dict:
    """own_frac100_L48 here vs famexp pass 1, per fold: weights + open npz by sha256 and by array; sealed sim by file sha256 only."""
    import numpy as np
    from safetensors.numpy import load_file
    p1 = {(x["fold"], x["readout"]): x for g in json.loads((Path("/fam/famexp") / SPEC / f"fit_{P1}.json").read_text())["groups"]
          for x in g["fits"] if x["variant"] == "own_frac100_L48"}
    mine = Path("/out/step3") / SPEC / f"run_{run}"
    theirs = Path("/fam/famexp") / SPEC / f"run_{P1}"
    res = {}
    for f in range(5):
        nm = f"own_frac100_L48__{readout}__L48__f{f}"
        sh = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
        r = {"p1_record_found": (f, readout) in p1}
        for part, ext in (("weights", "safetensors"), ("open", "npz"), ("sim_sealed", "npz")):
            a, b = mine / part / f"{nm}.{ext}", theirs / part / f"{nm}.{ext}"
            r[f"{part}_sha_equal"] = sh(a) == sh(b)
            r[f"{part}_sha_equals_p1_record"] = r["p1_record_found"] and sh(a) == p1[(f, readout)]["sha256"][part]
        wa, wb = load_file(str(mine / "weights" / f"{nm}.safetensors")), load_file(str(theirs / "weights" / f"{nm}.safetensors"))
        r["weights_arrays_equal"] = all(np.array_equal(wa[k], wb[k]) for k in ("W", "b", "mean", "std"))
        r["weights_max_abs_diff"] = max(float(np.abs(wa[k].astype(np.float64) - wb[k].astype(np.float64)).max()) for k in ("W", "b", "mean", "std"))
        oa, ob = np.load(mine / "open" / f"{nm}.npz"), np.load(theirs / "open" / f"{nm}.npz")
        r["open_fields_equal"] = sorted(oa.files) == sorted(ob.files) and all(np.array_equal(oa[k], ob[k]) for k in oa.files)
        r["open_rows"] = int(len(oa["score"]))
        r["open_score_max_abs_diff"] = float(np.abs(oa["score"] - ob["score"]).max()) if len(oa["score"]) == len(ob["score"]) else None
        r["bit_equal"] = all(r[k] for k in ("weights_sha_equal", "open_sha_equal", "sim_sealed_sha_equal", "weights_arrays_equal", "open_fields_equal",
                                             "weights_sha_equals_p1_record", "open_sha_equals_p1_record", "sim_sealed_sha_equals_p1_record"))
        res[str(f)] = r
    return res


def _coverage(spec: dict, split: dict, gitmap: dict) -> dict:
    """Counts only: per T196 task, which worlds have >= 1 row at R / C (spec.n_rows); the folds vs the Sol #36 git map."""
    by = {}
    for k, (w, t) in spec["map"].items():
        nr = spec["n_rows"][k]
        for i, rd in enumerate(("R", "C")):
            if nr[i] > 0:
                by.setdefault(rd, {}).setdefault(t, set()).add(w)
    four = ("honest_a", "run1", "run2", "sim")
    sets = {"cohort_183": split["cohort"], "estimation_90": split["estimation"], "universe_195": split["universe"]}
    cov = {rd: {nm: {"n": len(ts), "all_four_worlds": sum(set(four) <= by[rd].get(t, set()) for t in ts),
                     **{w: sum(w in by[rd].get(t, set()) for t in ts) for w in (*four, "honest_b", "honest_c")}} for nm, ts in sets.items()} for rd in ("R", "C")}
    fot = spec["fot"]
    gm = gitmap["fold_of_task"]
    fold_check = {"git_map_tasks": len(gm), "spec_tasks": len(fot), "agree": sum(fot.get(t) == v for t, v in gm.items()),
                  "disagree": sorted(t for t, v in gm.items() if t in fot and fot[t] != v), "in_git_map_not_spec": sorted(set(gm) - set(fot)),
                  "in_spec_not_git_map": sorted(set(fot) - set(gm))}
    return {"coverage": cov, "fold_check": fold_check}


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(32768, 32768), timeout=1800, retries=0, scaledown_window=2, volumes=VOLS)
def gate(stamp: str, split: dict, gitmap: dict) -> dict:
    t0 = time.time()
    run = f"gate_{stamp}"
    rec = {"groups": {}, "compare": {}, "fit_path_sha256": _fit_path_sha()}
    rec |= _coverage(_spec(), split, gitmap)
    for rd in ("R", "C"):
        g = _group3(rd, 48, [0, 1, 2, 3, 4], ["own_frac100"], run)
        rec["groups"][rd] = {k: v for k, v in g.items() if k != "fits"} | {"fit_seconds": [x["seconds"] for x in g["fits"]],
                                                                          "certified": [x["certified"] for x in g["fits"]]}
        rec["compare"][rd] = _gate_compare(run, rd)
    rec["bit_equal_all"] = all(r["bit_equal"] for c in rec["compare"].values() for r in c.values())
    rec["container_seconds"] = time.time() - t0
    _j(Path("/out/step3") / SPEC / f"gate_{stamp}.json", rec)
    WORK.commit()
    return rec


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(32768, 32768), timeout=1200, retries=0, scaledown_window=2, volumes=VOLS)
def fit_group(readout: str, L: int, run: str) -> dict:
    return _group3(readout, L, [0, 1, 2, 3, 4], list(RECIPES), run)


@app.function(image=img, cpu=(2, 2), memory=(4096, 4096), timeout=3600, retries=0, scaledown_window=2, volumes={"/out": WORK})
def orchestrate(stamp: str, gate_stamp: str, rule: dict) -> dict:
    """One container per (readout, layer); refuses unless this Volume's gate record is bit-equal AND was made by the same fit-path
    code. A failed group is rerun once, into a fresh attempt folder (files are never overwritten)."""
    g = json.loads((Path("/out/step3") / SPEC / f"gate_{gate_stamp}.json").read_text())
    if not g.get("bit_equal_all") or g.get("fit_path_sha256") != _fit_path_sha():
        raise SystemExit("the gate record is not bit-equal, or its fit-path code differs from this deployment's: phase 1 does not run")
    run = f"fit_{stamp}"
    jobs = [(rd, L) for rd in ("R", "C") for L in LAYERS]
    calls = [fit_group.spawn(rd, L, run) for rd, L in jobs]
    res, first, fails = [], [], []
    for attempt in (1, 2):
        for (rd, L), c in zip(jobs, calls, strict=True):
            try:
                res.append(c.get())
            except Exception as e:  # noqa: BLE001 - recorded per group, rerun once, never dropped silently
                (first if attempt == 1 else fails).append({"group": [rd, L], "error": type(e).__name__, "detail": str(e)[:300], "attempt": attempt})
        if attempt == 1 and first:
            jobs = [tuple(x["group"]) for x in first]
            calls = [fit_group.spawn(rd, L, f"{run}_retry") for rd, L in jobs]
        else:
            break
    fits = [x for g_ in res for x in g_["fits"]]
    rec = {"rule": rule, "gate": gate_stamp, "fit_path_sha256": g["fit_path_sha256"], "run": run, "retried_groups": first, "failed_groups": fails, "fits": len(fits),
           "uncertified": [x for x in fits if not x["certified"]], "groups": res}
    _j(Path("/out/step3") / SPEC / f"fit_{stamp}.json", rec)
    WORK.commit()
    return {k: v for k, v in rec.items() if k != "groups"} | {"group_seconds": [[g_["readout"], g_["layer"], g_["load_seconds"], g_["seconds"]] for g_ in res]}


@app.local_entrypoint()
def main(job: str, gate_stamp: str = ""):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    me = [str((HERE / f).relative_to(REPO)) for f in ("run_step3.py", "RULE.md")]
    if git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--untracked-files=all", "--", *me[:1]):
        raise SystemExit("uncommitted changes, or this launcher is untracked: commit and push first")
    head = git("rev-parse", "HEAD")
    if not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", head).splitlines()):
        raise SystemExit("HEAD is not on origin: push first")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"git": head, "stamp (PT)": stamp, "app": APP[job], "job": job}
    if job == "gate":
        sys.path.insert(0, str(FAM))
        import run_famexp_pack as rfp
        _, sp = rfp.load_split()
        pin = json.loads((FAM / "v6_pin.json").read_text())
        us = json.loads(subprocess.run(["git", "-C", str(REPO), "show", f"{pin['commit']}:{pin['path']}"], capture_output=True, check=True).stdout)["result"]["universe_and_split"]
        excl = {sp["main_of"][t] for t in sp["eligibility_half"]}
        split = {"cohort": sp["cohort"], "estimation": [t for t in sp["cohort"] if sp["main_of"][t] not in excl], "universe": us["universe"]}
        if (len(split["cohort"]), len(split["estimation"]), len(split["universe"])) != (183, 90, 195):
            raise SystemExit("split sizes differ from 183 / 90 / 195")
        gitmap = json.loads(git("show", f"{SOL36[0]}:{SOL36[1]}"))
        out = gate.remote(stamp, split, gitmap)
    elif job == "fit" and gate_stamp:
        if not (HERE / "RULE.md").exists() or git("status", "--porcelain", "--untracked-files=all", "--", me[1]):
            raise SystemExit("RULE.md must be committed and pushed before any step-3 fit")
        rule = {"commit": git("log", "-1", "--format=%H", "--", me[1]), "sha256": hashlib.sha256((HERE / "RULE.md").read_bytes()).hexdigest()}
        out = orchestrate.remote(stamp, gate_stamp, rule)
    else:
        raise SystemExit("--job gate | fit --gate-stamp <stamp>")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{job}_{stamp}.json").open("x").write(json.dumps(rec | {"result": out}, indent=1, default=float))
    print(json.dumps(rec | {"result": out}, indent=1, default=float)[:6000])
