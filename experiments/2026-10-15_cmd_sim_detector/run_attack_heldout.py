"""POST HOC (Aswin 10-15 12:07 via 0d; 2e's detector fork): the ATTACK-ONLY sim-vs-real detector at the command (content-only call_end)
and reply readouts, with run 2 as a HELD-OUT real world. A copy of run_detector_refit.py @ 0b8bdd422 (aswin/projection-why-small-gaps)
whose ONLY changes are: (a) observed fits only (no shuffle null: A2's record @ db69919f6 holds the 100-draw within-task null for these
exact fits, and the reproduction gate below proves the fits are A2's); (b) in each fold of (c) sim attack vs run1, run 2's runs on that
fold's held-out main tasks are scored with the fold's weights (never seen in training: their tasks are in the test fold), giving
AUROC(sim vs held-out run 2) and AUROC(held-out run 2 vs run 1) under the sim-vs-run1 detector; (c) a task bootstrap (2,000 draws,
scores fixed) for 95% intervals on every run AUROC; (d) weights go to xfer/cmd_sim_detector/<stamp>/, the record is
attack_heldout_<stamp>.json. Reproduction gate vs A2's stored record kept unchanged.

  PYTHONPATH=src <modal> run --detach experiments/2026-10-15_cmd_sim_detector/run_attack_heldout.py::main_a2 --r36 <runs36.json>

Docstring of the copied file (run_detector_refit.py):
DETECTOR REFIT WITH WEIGHTS SAVED (projection add-on, RULE.md Addendum 1, 0d via 2e 10-15): a copy of
run_controls_a2.py @ 2188f0ad1 (aswin/cmd-readout-simh-control) whose ONLY changes are (1) each observed fit's per-fold W, b and its
standardization (mean, std, computed with torch_probe.standardize's own formulas) are saved as npz on the solavg Volume with sha256
recorded, with a check that the saved weights reproduce the held-out row scores; (2) a gate that every cell's held-out run scores and
metrics reproduce A2's stored record (controls_a2_2026-10-15_004048.json @ db69919f6); (3) relocated paths (the A1 consistency
reference is a byte-identical copy in config/), the app name (in run_simh_control.py) and the record's name. Same data, folds, C,
layers, readouts, nulls and bootstrap as A2.

  <scratch>/modalenv/bin/modal run --detach experiments/2026-10-15_projection_why_small_gaps/run_detector_refit.py::main_a2 --r36 <runs36.json>

Original docstring (A2):
RULE.md Amendment 2: (c) simulated attack (class 1) vs real attack run1 (class 0) and (d) real attack run2 (class 1) vs run1 (class 0),
the real-vs-real noise floor for (c). Same pipeline as run_simh_control.py / run_controls_a1.py, both imported unchanged (loader,
shuffle, catch_at with k_caught/n, realized FPR and Wilson intervals, null summaries), on run_simh_control's app. Thresholds per fold
on held-out run1 units. Run level and call level; catch at 1% and 5% primary, AUROC secondary. Stored activations only; attack runs
are read as activations (and staged token ids for the fixed-call-token rule) only; no text is read.

  PYTHONPATH=src <scratch>/modalenv/bin/modal run --detach experiments/2026-10-14_cmd_readout_simh_control/run_controls_a2.py::main_a2 \\
      --r36 <runs36.json>
"""

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import modal.exception

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_simh_control as base  # noqa: E402  (unchanged)
import run_controls_a1 as a1  # noqa: E402  (unchanged)

WORLDS = {"run1": "run1", "run2": "run2", "sim": "sim_attack"}  # corrected-B manifest world -> batch label
PROBLEMS = [{"name": "c_simatk_vs_run1", "pos": ["sim_attack"], "neg": ["run1"]},
            {"name": "d_run2_vs_run1", "pos": ["run2"], "neg": ["run1"]}]
C0 = 0.01
REF_A1 = {"path": "config/ref_controls_a1_2026-10-14_235656.json", "sha256": "0f9024985a9a99f915329c8ffe812acc6ee1ed859c2f791741eadfd2a0130205"}  # relocated copy
REF_A2 = {"commit": "db69919f6ad12ec5d2861713a02039f7b83f8be2", "path": "experiments/2026-10-14_cmd_readout_simh_control/results/controls_a2_2026-10-15_004048.json",
          "sha256": "8182a82008d2069c1bdf391d5e1118e972a34d7dd4fef1c126d8fd5106da7caf"}  # the record the refit must reproduce
TOL_RUN, TOL_MET, TOL_SAVED = 1e-6, 1e-6, 1e-9  # refit vs A2 run scores and metrics; saved weights vs the fit's own held-out row scores
SOLVOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
sol_vol = modal.Volume.from_name(SOLVOL, version=2)
EXPECT = {"run1": 193, "run2": 192, "sim_attack": 191}
FIT_TIMEOUT, MARGIN_S, SKIP_TIMEOUT, IDLE_S, START_GPU, START_CPU = 480, 75, 300, 2, 90, 60  # the margin leaves room for metrics + bootstrap
SKIP_DEADLINE_S, FIT_DEADLINE_S = 600, 720  # overall deadlines from spawn: outstanding calls are cancelled (containers terminated) after them
BOOT_B, BOOT_SEED = 2000, 20261015  # the task-paired bootstrap of (c) - (d)
FPR_NOTE = ("realized on the same real-attack run1 units that set each fold's threshold (in-sample for the threshold), not a prospective FPR; "
            "run1 has 35-43 runs per fold, so at run level nominal 1% means k = 0 per fold and nominal 5% about 1-2 per fold")
DIRECTION = ("class 1 = simulated attack (c) or run2 (d), class 0 = real attack run1; detectors are trained to score class 1 higher; the "
             "threshold is set per fold on held-out run1 units; catch = share of class-1 units above the score exceeded by only 1% (5%) of run1 units")

app = base.app  # run_simh_control's app: these functions are added to it
_files = [(HERE / "run_simh_control.py", "/root/run_simh_control.py"), (HERE / "run_controls_a1.py", "/root/run_controls_a1.py")]
tok_image = base.tok_image.add_local_file(*_files[0]).add_local_file(*_files[1])
gpu_image = base.gpu_image.add_local_file(*_files[0]).add_local_file(*_files[1])


@app.function(image=tok_image, cpu=(4, 4), memory=(16384, 16384), timeout=SKIP_TIMEOUT, retries=0, scaledown_window=IDLE_S,
              startup_timeout=START_CPU, volumes={"/b": base.b_vol, "/hf": base.hf_vol})
def skip_b_a2(want: dict, shas: dict, guard) -> dict:
    if not guard.put("skip", time.time(), skip_if_exists=True):  # the cost guard (see fit_a2)
        raise RuntimeError("skip list: a rescheduled attempt; refused by the cost guard")
    return base.skip_rows("/b", want, shas)


class Watchdog:
    """The independent deadline (Amendment 2): a daemon thread that, every 5 s, cancels with terminate_containers=True any registered
    call past its own deadline (counted from when its spawn returned) and not yet marked done. Every cancel runs in its own daemon
    thread, so a stalled or failing cancel cannot delay the others; the log records each cancel as attempted, then confirmed or
    failed. The main thread's blocking SDK calls do not delay it."""

    def __init__(self):
        import threading

        self.calls, self.done, self.cancelled, self.log, self.threads = {}, set(), set(), [], {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def add(self, name: str, call, seconds: float):
        self.calls[name] = (call, time.time() + seconds)

    def _cancel(self, name: str, why: str):
        import threading

        def go(c):
            try:
                c.cancel(terminate_containers=True)
                self.log.append(f"{name}: cancel confirmed")
            except Exception as e:  # noqa: BLE001 - logged in the record; the per-attempt timeout still bounds that container
                self.log.append(f"{name}: cancel failed: {type(e).__name__}: {str(e)[:200]}")

        with self._lock:  # claim, start and publish in one acquisition (the same lock guards cancel_outstanding's snapshot)
            if name in self.done or name in self.cancelled or name not in self.calls:
                return
            self.cancelled.add(name)
            self.log.append(f"{name}: cancel attempted ({why})")
            th = threading.Thread(target=go, args=(self.calls[name][0],), daemon=True)
            th.start()
            self.threads[name] = th

    def _run(self):
        while not self._stop.wait(5):
            for name, (_, dl) in list(self.calls.items()):
                if time.time() >= dl:
                    self._cancel(name, "deadline")

    def cancel_outstanding(self, why: str, join_s: float = 30.0):
        """Cancel every registered call not marked done (independently), then wait at most join_s in total for confirmations."""
        for name in list(self.calls):
            self._cancel(name, why)
        end = time.time() + join_s
        with self._lock:
            snap = list(self.threads.items())
        for name, t in snap:
            t.join(max(0.0, end - time.time()))
            if t.is_alive():
                self.log.append(f"{name}: cancel not confirmed within {join_s:.0f} s")

    def stop(self):
        self._stop.set()


@app.function(image=gpu_image, gpu="H100!", cpu=(4, 4), memory=(24576, 24576), timeout=FIT_TIMEOUT, retries=0, scaledown_window=IDLE_S,
              startup_timeout=START_GPU, volumes={"/b": base.b_vol, "/sol": sol_vol})
def fit_a2(L: int, readout: str, epochs: list, skip: dict, guard, wdir: str) -> dict:
    """guard: an ephemeral modal.Dict; the first attempt of a cell claims its key, and a rescheduled attempt (crash or preemption,
    which retries=0 does not stop) finds it taken and stops before any work (the cost bound)."""
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

    if not guard.put(f"{readout}|L{L}", time.time(), skip_if_exists=True):
        raise RuntimeError(f"{readout}|L{L}: a rescheduled attempt; refused by the cost guard")
    t0 = time.time()
    X16, row_run, keep, no_rows, chunks = base._load(L, readout, epochs, skip, {})  # every epoch is on the corrected-B manifest (Amendment 23)
    runs = [epochs[i] for i in keep]  # epochs arrive sorted by name
    t_load = time.time() - t0
    XT = torch.from_numpy(X16).cuda()
    del X16
    N = base.N_NULL
    prep = {}
    for pr in PROBLEMS:
        ids = np.array([j for j, r in enumerate(runs) if r["batch"] in pr["pos"] + pr["neg"]])
        y = np.array([int(runs[j]["batch"] in pr["pos"]) for j in ids], np.int64)
        perms, pinfo = base.within_task_perms([runs[j]["task"] for j in ids], y, N, base.SEED)
        pos = np.full(len(runs), -1)
        pos[ids] = np.arange(len(ids))
        rows_p = np.flatnonzero(np.isin(row_run, ids))
        rpos = np.full(len(row_run), -1)
        rpos[rows_p] = np.arange(len(rows_p))
        prep[pr["name"]] = {"ids": ids, "Y": np.vstack([y[None, :], perms]), "pos": pos, "fold": np.array([runs[j]["fold"] for j in ids]), "pinfo": pinfo,
                            "perm_sha256": hashlib.sha256(perms.tobytes()).hexdigest(), "rows": rows_p, "rpos": rpos,
                            "oof": np.full((1 + N, len(ids)), np.nan), "oof_row": np.full((1 + N, len(rows_p)), np.nan)}
    NULL = list(range(1, N + 1))
    work = [("P1", p["name"], [0]) for p in PROBLEMS]  # (detector fork) observed fits only; A2's record holds the null for these fits
    heldout_run2 = {}  # (detector fork) run 2 run name -> mean held-out score under the (c) fold model of its main task's fold
    certs, took, done = [], {"single": [], "null": []}, {}
    saved_w, saved_chk = {}, {}  # (refit) observed fits only: problem -> {fold: (W, b, mean, std)}; max |saved-weights score - S|
    for prio, p, cols in work:
        d = prep[p]
        kind = "null" if len(cols) > 1 else "single"
        for f in range(5):
            if prio != "P1":  # reserve (run_controls_a1's rule)
                est = max(took[kind]) if took[kind] else 3 * max(took["single"])
                if time.time() - t0 + 1.25 * est > FIT_TIMEOUT - MARGIN_S:
                    certs.append({"priority": prio, "problem": p, "fold": f, "skipped": f"reserve: elapsed {time.time() - t0:.0f} s, estimate {est:.0f} s"})
                    continue
            tr_runs, te_runs = d["ids"][d["fold"] != f], d["ids"][d["fold"] == f]
            tr = np.flatnonzero(np.isin(row_run, tr_runs))
            te = np.flatnonzero(np.isin(row_run, te_runs))
            Xtr, Xte = tp.standardize(XT[torch.from_numpy(tr).cuda()].double(), XT[torch.from_numpy(te).cuda()].double())
            if prio == "P1":  # (refit) the standardization, with torch_probe.standardize's own formulas on the same rows
                Xr = XT[torch.from_numpy(tr).cuda()].double()
                mu_, sd_ = Xr.mean(0), Xr.std(0, unbiased=False)
                sd_ = sd_.where(sd_ > 0, sd_.new_ones(()))
                del Xr
            Ytr = torch.from_numpy(d["Y"][cols][:, d["pos"][row_run[tr]]].T.astype(np.float64)).cuda()
            ts = time.time()
            W, b, cert = tp.fit(Xtr, Ytr, C0, allow_uncertified=True, return_certificate=True)
            S = (Xte @ W + b).cpu().numpy()  # [test rows, columns]
            if prio == "P1":  # (refit) save, and check the saved weights reproduce S from the raw held-out rows
                Wn, bn, mn, sn = (t.detach().cpu().numpy().astype(np.float64) for t in (W[:, 0], b.reshape(-1)[:1], mu_, sd_))
                Xe = XT[torch.from_numpy(te).cuda()].double().cpu().numpy()
                saved_chk.setdefault(p, {})[f] = float(np.max(np.abs(((Xe - mn) / sn) @ Wn + bn[0] - S[:, 0])))
                saved_w.setdefault(p, {})[f] = (Wn, bn, mn, sn)
                del Xe
            if prio == "P1" and p == "c_simatk_vs_run1":  # (detector fork) score run 2's held-out runs of this fold with this fold's (c) model
                r2 = np.array([j for j, r in enumerate(runs) if r["batch"] == "run2" and r["fold"] == f], np.int64)
                rows2 = np.flatnonzero(np.isin(row_run, r2))
                X2 = XT[torch.from_numpy(rows2).cuda()].double()
                S2 = (((X2 - mu_) / sd_) @ W[:, 0] + b.reshape(-1)[0]).cpu().numpy()
                if not np.isfinite(S2).all():
                    raise ValueError("non-finite held-out run 2 scores")
                rr2 = row_run[rows2]
                for j in r2:
                    m_ = rr2 == j
                    if not m_.any():
                        raise ValueError("a held-out run 2 run without rows")
                    heldout_run2[runs[j]["name"]] = float(S2[m_].mean())
                del X2
            loc = d["pos"][row_run[te]]
            tot = np.zeros((len(d["ids"]), len(cols)))
            np.add.at(tot, loc, S)
            cnt = np.bincount(loc, minlength=len(d["ids"]))
            tp_ = d["pos"][te_runs]
            if (cnt[tp_] == 0).any():
                raise ValueError("a test run without rows")
            d["oof"][np.ix_(cols, tp_)] = (tot[tp_] / cnt[tp_, None]).T
            d["oof_row"][np.ix_(cols, d["rpos"][te])] = S.T
            g = np.array(cert["max_grad_per_column"])
            took[kind].append(time.time() - ts)
            certs.append({"priority": prio, "problem": p, "C": C0, "fold": f, "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
                          "n_train_runs": int(len(tr_runs)), "n_test_runs": int(len(te_runs)), "n_evals": int(cert["n_evals"]), "columns": [int(c) for c in cols],
                          "max_grad_per_column": [float(x) for x in g], "certified_per_column": [bool(x <= tp.CERT_BAR) for x in g], "max_grad": float(g.max()),
                          "uncertified_columns": [int(cols[i]) for i in np.flatnonzero(~(g <= tp.CERT_BAR))], "seconds": round(took[kind][-1], 2),
                          "columns_note": "column 0 = observed labels, column k = null draw k"})
            done.setdefault((p, kind), set()).add(f)
            del Xtr, Xte, Ytr, W, b
            torch.cuda.empty_cache()
    import io

    wsaved = {}
    for p_, fw in saved_w.items():  # (refit) one npz per (cell, problem): f{f}__{W,b,mean,std}, float64
        buf = io.BytesIO()
        np.savez(buf, **{f"f{f}__{k}": v for f, t in sorted(fw.items()) for k, v in zip(("W", "b", "mean", "std"), t)})
        rel = f"xfer/cmd_sim_detector/{wdir}/w_{readout}_L{L}__{p_}.npz"
        Path("/sol", rel).parent.mkdir(parents=True, exist_ok=True)
        Path("/sol", rel).open("xb").write(buf.getvalue())
        wsaved[p_] = {"volume": SOLVOL, "path": rel, "sha256": hashlib.sha256(buf.getvalue()).hexdigest(), "folds": sorted(int(f) for f in fw),
                      "max_abs_saved_weights_score_diff": max(saved_chk[p_].values()), "standardization": "torch_probe.standardize formulas, training rows of the fold"}
    sol_vol.commit()
    out = {"layer": L, "readout": readout, "gpu": torch.cuda.get_device_name(0), "saved_weights": wsaved, "load_seconds": round(t_load), "seconds": round(time.time() - t0),
           "rows": int(len(row_run)), "chunks_read": chunks, "runs_without_rows": no_rows, "training_unit": a1.TRAINING_UNIT, "direction": DIRECTION,
           "rows_per_run": {r["name"]: int((row_run == j).sum()) for j, r in enumerate(runs)}, "certificates": certs, "problems": {}}
    for p, d in prep.items():
        y, fold = d["Y"][0], d["fold"]
        rrun = d["pos"][row_run[d["rows"]]]
        rfold = fold[rrun]
        unc = {(c["fold"], u) for c in certs if c.get("problem") == p for u in c.get("uncertified_columns", [])}
        obs_ok = done.get((p, "single"), set()) == set(range(5)) and bool(np.isfinite(d["oof"][0]).all()) and bool(np.isfinite(d["oof_row"][0]).all())
        null_ok = done.get((p, "null"), set()) == set(range(5)) and bool(np.isfinite(d["oof"][1:]).all()) and bool(np.isfinite(d["oof_row"][1:]).all())
        res = {"n_runs": {"class1": int(y.sum()), "class0": int((y == 0).sum())},
               "n_rows": {"class1": int((y[rrun] == 1).sum()), "class0": int((y[rrun] == 0).sum())},
               "n_runs_per_fold": {str(f): {"class1": int(((fold == f) & (y == 1)).sum()), "class0": int(((fold == f) & (y == 0)).sum())} for f in range(5)},
               "shuffle": d["pinfo"], "perm_sha256": d["perm_sha256"], "observed_complete": obs_ok, "null_complete": null_ok,
               "observed_certified": not any(c == 0 for _, c in unc), "uncertified_fold_columns": sorted([list(x) for x in unc])}
        if obs_ok:
            res["run_level"] = a1._metrics(d["oof"][0], y, fold)
            res["call_level"] = a1._metrics(d["oof_row"][0], y[rrun], rfold)
            for lvl in ("run_level", "call_level"):
                for a in ("0.01", "0.05"):
                    res[lvl][f"catch_{a}"]["fpr_note"] = FPR_NOTE  # Amendment 2: class 0 = real attack run1, not real honest
            res["call_level_note"] = "units of one run share its fold and label; the Wilson intervals treat units as independent, so they are too narrow"
            res["run_scores"] = {runs[j]["name"]: float(d["oof"][0][i]) for i, j in enumerate(d["ids"])}
        if obs_ok and null_ok:
            bad = sorted({c for _, c in unc if c > 0})
            res["null"] = {"uncertified_draws": bad, "certified": not bad}
            for lvl, sc, yy, ff in (("run_level", d["oof"], lambda k: d["Y"][k], fold), ("call_level", d["oof_row"], lambda k: d["Y"][k][rrun], rfold)):
                for a in (0.01, 0.05):
                    nt = np.array([a1.catch_at(sc[k], yy(k), ff, a)["tpr"] for k in range(1, N + 1)])
                    res["null"][f"{lvl}_catch_{a:g}"] = nt.tolist()
                    res["null"][f"{lvl}_catch_{a:g}_summary"] = a1._summ(nt, res[lvl][f"catch_{a:g}"]["tpr"])
                na = np.array([base.auroc(sc[k], yy(k)) for k in range(1, N + 1)])
                res["null"][f"{lvl}_auroc"] = na.tolist()
                res["null"][f"{lvl}_auroc_summary"] = a1._summ(na, res[lvl]["auroc"])
        out["problems"][p] = res
    out["paired_c_minus_d"] = _paired(prep, runs, row_run) if all(r.get("run_level") for r in out["problems"].values()) else None
    # (detector fork) task-bootstrap 95% intervals on run AUROC, and the held-out run 2 contrasts under the (c) detector
    for p, d in prep.items():
        if out["problems"][p].get("run_scores"):
            out["problems"][p]["run_auroc_boot"] = boot_pairs(d["oof"][0], d["Y"][0], [runs[j]["task"] for j in d["ids"]])
    c = out["problems"].get("c_simatk_vs_run1", {}).get("run_scores")
    n2 = sorted(r["name"] for r in runs if r["batch"] == "run2")
    if c and sorted(heldout_run2) == n2:
        sim = {n: v for n, v in c.items() if n.startswith("sim|")}
        r1 = {n: v for n, v in c.items() if n.startswith("run1|")}
        task = lambda n: n.split("|", 1)[1]  # noqa: E731
        hold = {}
        for name, pos, neg in (("sim_vs_heldout_run2", sim, heldout_run2), ("heldout_run2_vs_run1", heldout_run2, r1)):
            sc = np.array([*pos.values(), *neg.values()])
            yy = np.array([1] * len(pos) + [0] * len(neg))
            hold[name] = {"auroc": base.auroc(sc, yy), "n_pos": len(pos), "n_neg": len(neg),
                          "boot": boot_pairs(sc, yy, [task(n) for n in [*pos, *neg]])}
        out["heldout_run2"] = {"n_runs": len(heldout_run2), "run_scores": heldout_run2, "contrasts": hold,
                               "note": "run 2 runs scored with the (c) sim-vs-run1 fold model of their main task's fold (task unseen in that model's training); "
                                       "sim and run 1 scores are (c)'s out-of-fold scores; class 1 = first world named"}
    else:
        out["heldout_run2"] = {"error": f"held-out run 2 scores missing: {len(heldout_run2)} of {len(n2)}"}
    return out


def boot_pairs(score, y, task, B: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    """(detector fork) Task bootstrap of run AUROC on fixed scores: main tasks drawn with replacement, every run of a drawn task taken
    with multiplicity; a draw missing a class is counted as failed, never dropped silently."""
    import numpy as np

    score, y = np.asarray(score, np.float64), np.asarray(y)
    groups = {}
    for i, t in enumerate(task):
        groups.setdefault(t, []).append(i)
    keys = sorted(groups)
    idx = {t: np.array(groups[t]) for t in keys}
    rng = np.random.default_rng(seed)
    vals, failed = [], 0
    for _ in range(B):
        sel = np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        if y[sel].min() == y[sel].max():
            failed += 1
            continue
        vals.append(base.auroc(score[sel], y[sel]))
    v = np.array(vals)
    return {"n_draws": B, "failed_draws": failed, "seed": seed, "n_tasks": len(keys), "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]}


def _paired(prep: dict, runs: list, row_run) -> dict:
    """Task-paired bootstrap of (c) - (d) on the fixed out-of-fold scores (no refit): BOOT_B draws of the main tasks with replacement
    (seed BOOT_SEED); each draw takes every run (and row) of the drawn tasks, with multiplicity, in both problems, and recomputes
    run-level AUROC and catch at 1% and 5% and call-level AUROC and catch at 5% (thresholds per fold on the drawn run1 units). A draw
    whose metric cannot be computed (a fold without run1 or a class missing) is counted, never dropped silently."""
    import numpy as np

    def index(d):
        rt, rw = {}, {}
        for i, j in enumerate(d["ids"]):
            rt.setdefault(runs[j]["task"], []).append(i)
        for r_i, i in enumerate(d["pos"][row_run[d["rows"]]]):
            rw.setdefault(runs[d["ids"][i]]["task"], []).append(r_i)
        return {t: np.array(v) for t, v in rt.items()}, {t: np.array(v) for t, v in rw.items()}

    probs = {"c": prep["c_simatk_vs_run1"], "d": prep["d_run2_vs_run1"]}
    idx = {k: index(d) for k, d in probs.items()}
    rows_y = {k: d["Y"][0][d["pos"][row_run[d["rows"]]]] for k, d in probs.items()}
    rows_f = {k: d["fold"][d["pos"][row_run[d["rows"]]]] for k, d in probs.items()}
    tasks = sorted(set(idx["c"][0]) | set(idx["d"][0]))

    def metrics(k, sel_runs, sel_rows):
        d = probs[k]
        s, y, f = d["oof"][0][sel_runs], d["Y"][0][sel_runs], d["fold"][sel_runs]
        sr, yr, fr = d["oof_row"][0][sel_rows], rows_y[k][sel_rows], rows_f[k][sel_rows]
        return {"run_auroc": base.auroc(s, y), "run_catch_0.01": a1.catch_at(s, y, f, 0.01)["tpr"], "run_catch_0.05": a1.catch_at(s, y, f, 0.05)["tpr"],
                "call_auroc": base.auroc(sr, yr), "call_catch_0.05": a1.catch_at(sr, yr, fr, 0.05)["tpr"]}

    every = {k: (np.concatenate(list(idx[k][0].values())), np.concatenate(list(idx[k][1].values()))) for k in probs}
    obs = {k: metrics(k, *every[k]) for k in probs}
    rng = np.random.default_rng(BOOT_SEED)
    diffs, failed = {m: [] for m in obs["c"]}, 0
    for _ in range(BOOT_B):
        draw = [tasks[i] for i in rng.integers(0, len(tasks), len(tasks))]
        try:
            got = {k: metrics(k, np.concatenate([idx[k][0][t] for t in draw if t in idx[k][0]]),
                              np.concatenate([idx[k][1][t] for t in draw if t in idx[k][1]])) for k in probs}
            if not all(np.isfinite(v) for g in got.values() for v in g.values()):
                raise ValueError("non-finite")
        except (ValueError, ZeroDivisionError, FloatingPointError):
            failed += 1
            continue
        for m in diffs:
            diffs[m].append(got["c"][m] - got["d"][m])
    out = {"n_draws": BOOT_B, "failed_draws": failed, "seed": BOOT_SEED, "n_tasks": len(tasks), "note": "out-of-fold scores fixed (no refit); task-paired",
           "observed": {m: {"c": obs["c"][m], "d": obs["d"][m], "c_minus_d": obs["c"][m] - obs["d"][m]} for m in obs["c"]}}
    for m, v in diffs.items():
        v = np.array(v)
        out["observed"][m] |= {"ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else None,
                               "share_draws_le_0": float((v <= 0).mean()) if len(v) else None}
    return out


@app.local_entrypoint()
def main_a2(r36: str, config: str = base.CONFIG):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--", str(HERE)):
        raise SystemExit("uncommitted tracked changes, or uncommitted files in this experiment folder")
    git_at_launch = git("rev-parse", "HEAD")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    if sha(r36) != base.R36_SHA:
        raise SystemExit("runs36.json sha256 mismatch")
    if sha(HERE / REF_A1["path"]) != REF_A1["sha256"]:
        raise SystemExit("Amendment 1's record (consistency reference) sha256 mismatch")
    ref = json.loads((HERE / REF_A1["path"]).read_text())
    cfg = json.loads((HERE / config).read_text())
    fold_of_main = {}
    for r in json.loads(Path(r36).read_text()):  # folds: r36 ONLY
        if fold_of_main.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("a main task in two folds")
    REPO_T196 = base.REPO / base.T196
    main_of = {c["key"]: c["main"] for c in json.loads(REPO_T196.read_text())["cells"]}
    mf = json.loads(base._git_bytes(cfg["real"]["manifest"]))
    epochs = [{k: e[k] for k in ("key", "world", "task", "arm", "extraction_run", "staged_inputs_run", "meta_file")} | {"batch": WORLDS[e["world"]], "cls": "attack", "vol": "b"}
              for e in mf["epochs"] if e["world"] in WORLDS]
    shas = {e["staged_inputs_file"]: e["staged_inputs_sha256"] for e in mf["epochs"] if e["world"] in WORLDS}
    want = {}
    for e in epochs:
        want.setdefault(e["staged_inputs_run"], {}).setdefault(e["arm"], []).append(e["key"])
    wd = Watchdog()
    guard_cm = modal.Dict.ephemeral()  # the cost guard: one attempt per call (skip list and each cell); closed in the finally below
    guard = guard_cm.__enter__()
    cells, failed, fatal = {}, {}, None
    try:  # one lifecycle: any failure cancels every outstanding call (independently), writes a failure record and closes the Dict
        call = skip_b_a2.spawn(want, shas, guard)
        wd.add("skip", call, SKIP_DEADLINE_S)
        sk = call.get(timeout=SKIP_DEADLINE_S)  # a timeout (builtin or Modal's class) is fatal: handled below
        wd.done.add("skip")
        if sk["fixed_tokens"] != base.FIXED_TOKENS:
            raise SystemExit("fixed call tokens differ from Amendment 12's")
        for e in epochs:
            e["name"] = f"{e['world']}|{e['task']}"
            e["fold"] = fold_of_main[main_of[e["task"]]]  # KeyError = a task outside T196 or without a #36 fold: stop
        if len({e["key"] for e in epochs}) != len(epochs) or len({e["name"] for e in epochs}) != len(epochs):
            raise SystemExit("duplicate epoch key or run name")
        epochs.sort(key=lambda e: e["name"])
        counts = {b_: sum(e["batch"] == b_ for e in epochs) for b_ in sorted({e["batch"] for e in epochs})}
        if counts != EXPECT:
            raise SystemExit(f"epoch counts {counts} != {EXPECT}")
        print(json.dumps({"stamp": stamp, "git": git_at_launch, "epochs_per_batch": counts, "skip_dropped_rows": sk["dropped_rows"]}), flush=True)
        jobs = {}
        for ro in base.READOUTS:
            for L in base.LAYERS:
                jobs[(L, ro)] = fit_a2.spawn(L, ro, epochs, sk["skip"], guard, stamp)
                wd.add(f"{ro}|L{L}", jobs[(L, ro)], FIT_DEADLINE_S)
        t_spawn = time.time()
        for (L, ro), c in jobs.items():
            try:
                cells[f"{ro}|L{L}"] = c.get(timeout=max(1.0, FIT_DEADLINE_S - (time.time() - t_spawn)))
                wd.done.add(f"{ro}|L{L}")
            except Exception as e:  # noqa: BLE001 - a timeout or a failed cell: recorded; the call is cancelled in the finally
                failed[f"{ro}|L{L}"] = f"{type(e).__name__}: {str(e)[:500]} (at {time.time() - t_spawn:.0f} s from spawn; deadline {FIT_DEADLINE_S} s)"
    except BaseException as e:
        fatal = f"{type(e).__name__}: {str(e)[:1000]}"
        raise
    finally:
        try:  # each cleanup step is isolated: an error in one never skips the next
            wd.cancel_outstanding("fatal error" if fatal else "not finished", join_s=30.0)
        except Exception as e:  # noqa: BLE001 - logged in the record
            wd.log.append(f"cancel_outstanding failed: {type(e).__name__}: {str(e)[:200]}")
        wd.stop()
        try:
            guard_cm.__exit__(None, None, None)
        except Exception as e:  # noqa: BLE001 - logged; the Dict is ephemeral and expires with the client anyway
            wd.log.append(f"guard close failed: {type(e).__name__}")
        if fatal is not None:  # a sanitized failure record (no scores), best effort; the original error propagates
            try:
                (HERE / "results" / f"attack_heldout_FAILED_{stamp}.json").open("x").write(json.dumps(
                    {"label": "RULE.md Amendment 2: FAILED before the record", "git": git_at_launch, "stamp": stamp, "error": fatal,
                     "failed": failed, "cells_done": sorted(cells), "watchdog_log": wd.log}, indent=1))
            except Exception as e:  # noqa: BLE001 - best effort; never masks the original error
                print(f"could not write the failure record: {type(e).__name__}", flush=True)
    same = {ro: len({json.dumps(cells[f"{ro}|L{L}"]["rows_per_run"], sort_keys=True) for L in base.LAYERS if f"{ro}|L{L}" in cells}) <= 1 for ro in base.READOUTS}
    consistency = {}  # run1 and simulated-attack rows per run must equal Amendment 1's
    for k, c in cells.items():
        mine = {n: v for n, v in c["rows_per_run"].items() if n.split("|", 1)[0] in ("run1", "sim")}
        theirs = {n: v for n, v in ref["cells"][k]["rows_per_run"].items() if n.split("|", 1)[0] in ("run1", "sim")}
        consistency[k] = {"runs_compared": len(theirs), "equal": mine == theirs}
    gaps = [f"{k}|{p}|observed" for k, c in cells.items() for p, r in c["problems"].items() if not r["observed_complete"]]  # (detector fork) no null here
    gaps += [f"{k}|heldout_run2" for k, c in cells.items() if "error" in c.get("heldout_run2", {"error": 1})]
    gaps += [f"{k}|paired" for k, c in cells.items() if not c.get("paired_c_minus_d")]
    rec = {"label": "RULE.md Amendment 2 (2026-10-14_cmd_readout_simh_control); PRELIMINARY until 09 and Codex review", "git": git_at_launch, "stamp": stamp,
           "app": base.APP, "training_unit": a1.TRAINING_UNIT, "direction": DIRECTION, "volume": base.BVOL,
           "config": {"path": config, "sha256": sha(HERE / config)}, "r36": {"path": r36, "sha256": base.R36_SHA}, "t196": {"path": base.T196, "sha256": sha(REPO_T196)},
           "consistency_reference": REF_A1, "settings": {"C": C0, "layers": base.LAYERS, "primary": base.PRIMARY, "n_null": 0, "null_reference": "A2 @ db69919f6 (100 within-task shuffles of these exact fits; reproduction gate below)", "seed": base.SEED,
                                                         "problems": PROBLEMS, "worlds": WORLDS, "fit_timeout_s": FIT_TIMEOUT},
           "epochs_per_batch": counts, "skip": {"dropped_rows": sk["dropped_rows"], "staged_sha256": sk["staged_sha256"]},
           "run_folds": {e["name"]: e["fold"] for e in epochs}, "rows_per_run_equal_across_layers": same, "consistency_with_a1": consistency,
           "required_gaps": gaps, "watchdog_log": wd.log, "cells": cells, "failed": failed}
    # (refit) the reproduction gate against A2's stored record, and the saved-weights check
    a2 = subprocess.run(["git", "show", f"{REF_A2['commit']}:{REF_A2['path']}"], capture_output=True, check=True, cwd=HERE).stdout
    if hashlib.sha256(a2).hexdigest() != REF_A2["sha256"]:
        raise SystemExit("A2's record sha256 differs from the pin")
    a2 = json.loads(a2)
    repro, rfail = {}, [f"missing cell {k}" for k in sorted(set(a2["cells"]) - set(cells))]
    for k in sorted(set(a2["cells"]) & set(cells)):
        c = cells[k]
        for p, ra in a2["cells"][k]["problems"].items():
            try:
                r = c["problems"][p]
                mine, theirs = r.get("run_scores", {}), ra["run_scores"]
                d = max(abs(mine[n] - theirs[n]) for n in theirs) if set(mine) == set(theirs) else float("inf")
                dm = 0.0
                for lvl in ("run_level", "call_level"):
                    dm = max(dm, abs(r[lvl]["auroc"] - ra[lvl]["auroc"]), *(abs(r[lvl][f"catch_{a}"]["tpr"] - ra[lvl][f"catch_{a}"]["tpr"]) for a in ("0.01", "0.05")))
                sw = c["saved_weights"][p]["max_abs_saved_weights_score_diff"]
                repro[f"{k}|{p}"] = {"runs": len(theirs), "same_run_keys": set(mine) == set(theirs), "max_abs_run_score_diff": d, "max_abs_metric_diff": dm,
                                     "max_abs_saved_weights_score_diff": sw, "saved_folds": c["saved_weights"][p]["folds"]}
                ok = d <= TOL_RUN and dm <= TOL_MET and sw <= TOL_SAVED and c["saved_weights"][p]["folds"] == list(range(5))
            except (KeyError, TypeError, ValueError) as e:  # a missing piece is a gate failure, recorded, never a silent pass
                repro[f"{k}|{p}"], ok = {"error": f"{type(e).__name__}: {str(e)[:200]}"}, False
            if not ok:
                rfail.append(f"{k}|{p}")
    rec["reproduction_of_a2"] = {"reference": REF_A2, "tolerances": {"run_scores": TOL_RUN, "metrics": TOL_MET, "saved_weights_vs_held_out_rows": TOL_SAVED},
                                 "cells": repro, "failed": rfail, "pass": not rfail}
    rec["label"] = ("POST HOC attack-only sim-vs-real detector with held-out run 2 (2e detector fork; a copy of run_detector_refit.py @ 0b8bdd422, observed fits "
                    "only; the null is A2's @ db69919f6); PRELIMINARY until 09 and Codex review")
    dest = HERE / "results" / f"attack_heldout_{stamp}.json"
    dest.open("x").write(json.dumps(rec, indent=1))
    for k, c in cells.items():
        for p, r in c["problems"].items():
            n = r.get("null", {})
            row = {"cell": k, "problem": p, "runs": r["n_runs"], "rows": r["n_rows"], "observed_certified": r["observed_certified"], "null_complete": r["null_complete"],
                   "null_certified": n.get("certified")}
            for lvl in ("run_level", "call_level"):
                m = r.get(lvl, {})
                for a in ("0.01", "0.05"):
                    c_ = m.get(f"catch_{a}", {})
                    row[f"{lvl}|catch_{a}"] = {x: c_.get(x) for x in ("tpr", "k_caught", "n", "tpr_wilson95", "fpr_realized", "class0_above", "n_class0")}
                    row[f"{lvl}|catch_{a}"]["null"] = n.get(f"{lvl}_catch_{a}_summary")
                row[f"{lvl}|auroc"] = {"auroc": m.get("auroc"), "null": n.get(f"{lvl}_auroc_summary")}
            print(json.dumps(row), flush=True)
    for k, c in cells.items():
        print(json.dumps({"cell": k, "paired_c_minus_d": c.get("paired_c_minus_d")}), flush=True)
        print(json.dumps({"cell": k, "run_auroc_boot": {p: r.get("run_auroc_boot", {}).get("ci95") for p, r in c["problems"].items()},
                          "heldout_run2": {n: {"auroc": v["auroc"], "ci95": v["boot"]["ci95"]} for n, v in c.get("heldout_run2", {}).get("contrasts", {}).items()}}), flush=True)
    print(json.dumps({"consistency_with_a1": consistency, "required_gaps": gaps, "failed": failed, "rows_equal": same}), flush=True)
    print(json.dumps({"reproduction_of_a2": rec["reproduction_of_a2"]}), flush=True)
    print(dest.name)
    if rfail:
        raise SystemExit(f"GATE: the refit does not reproduce A2's stored record for {rfail}: STOP and tell 2e")
    if failed or gaps or not all(same.values()) or not all(v["equal"] for v in consistency.values()):
        raise SystemExit("a cell failed, a required piece is missing, rows differ across layers, or run1/simulated-attack rows differ from Amendment 1's")
