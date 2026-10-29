"""UNCUT (RULE.md here; 4f detU fork, 10-16, for 0d's post): run_attack_heldout.py @ 648f33e61 (aswin/cmd-sim-detector-2e) whose ONLY
changes are (a) --inputs uncut: the same runs by key from their whole-run uncut roots (uncut_inputs.py; Volume /u), with every
epoch's meta checked; --inputs 32k keeps its construction verbatim (the reproduction gate vs A2 and vs 2e's held-out record);
(b) A2's 100-draw within-task null restored for the uncut fits (A2's own work list: observed, then the null, with the reserve rule);
(c) the ADDED per-action pieces: a task bootstrap on call-level AUROC, and held-out run 2 at call level; (d) the timeouts (uncut
runs are longer), weights to this fork's own Volume, --n-null (0 = observed only, the gate) and the record's name. The copied
docstring follows.

  PYTHONPATH=src <modal> run experiments/2026-10-16_cmd_sim_detector_uncut/run_attack_uncut.py::main_a --r36 <runs36.json> \
      --inputs 32k --n-null 0          # gate 2 (reproduction); never --detach (the client's exit must stop the app: RULE.md Cost)
  PYTHONPATH=src <modal> run .../run_attack_uncut.py::main_a --r36 <runs36.json> --inputs uncut --config config/inputs_uncut_v<k>.json

POST HOC (Aswin 10-15 12:07 via 0d; 2e's detector fork): the ATTACK-ONLY sim-vs-real detector at the command (content-only call_end)
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
import uncut_inputs as ui  # noqa: E402

WORLDS = {"run1": "run1", "run2": "run2", "sim": "sim_attack"}  # corrected-B manifest world -> batch label
PROBLEMS = [{"name": "c_simatk_vs_run1", "pos": ["sim_attack"], "neg": ["run1"]},
            {"name": "d_run2_vs_run1", "pos": ["run2"], "neg": ["run1"]}]
C0 = 0.01
REF_A1 = {"path": "config/ref_controls_a1_2026-10-14_235656.json", "sha256": "0f9024985a9a99f915329c8ffe812acc6ee1ed859c2f791741eadfd2a0130205"}  # relocated copy
REF_A2 = {"commit": "db69919f6ad12ec5d2861713a02039f7b83f8be2", "path": "experiments/2026-10-14_cmd_readout_simh_control/results/controls_a2_2026-10-15_004048.json",
          "sha256": "8182a82008d2069c1bdf391d5e1118e972a34d7dd4fef1c126d8fd5106da7caf"}  # the record the refit must reproduce
TOL_RUN, TOL_MET, TOL_SAVED = 1e-6, 1e-6, 1e-9  # refit vs A2 run scores and metrics; saved weights vs the fit's own held-out row scores
SOLVOL = "probe-shift_aswin-surya_sub-cmdsimdet_uncut_w_kill-2026-11-14"  # (uncut) this fork's own weights Volume, not the solavg one
sol_vol = modal.Volume.from_name(SOLVOL, version=2, create_if_missing=True)
u_vol = modal.Volume.from_name(ui.VOL, version=2).read_only()
REF_H = {"commit": "648f33e6115ebbe19276981dcbd79f9274d9f1ba", "path": "experiments/2026-10-15_cmd_sim_detector/results/attack_heldout_2026-10-15_122057.json",
         "sha256": "632ff9d234060061afa0041a71df12e4761e8196e8d4b6bdbfe32ec22882e0ad"}  # (uncut) 2e's held-out record: the gate's second reference
UNCUT_WORLDS = {"run1": "run1", "run2": "run2", "sim": "sim_attack"}
RATE_H = 4.33  # $/h: H100 3.95 + 4 CPU x 0.0473 + 24 GiB x 0.008 (Modal list prices; 2e's rate), for the pilot's projection
EXPECT = {"run1": 193, "run2": 192, "sim_attack": 191}
FIT_TIMEOUT, MARGIN_S, SKIP_TIMEOUT, IDLE_S, START_GPU, START_CPU = 1800, 300, 900, 2, 90, 60  # (uncut) 480/75/300 -> 1800/300/900: ~1.6x rows, null, row bootstraps
SKIP_DEADLINE_S, FIT_DEADLINE_S, GATE_DEADLINE_S = 1200, 2100, 600  # overall deadlines from spawn: outstanding calls are cancelled (containers
# terminated) after them; (Codex r2) the 32k gate (observed only, about 3 min expected) gets 600 s
BOOT_B, BOOT_SEED = 2000, 20261015  # the task-paired bootstrap of (c) - (d)
FPR_NOTE = ("realized on the same real-attack run1 units that set each fold's threshold (in-sample for the threshold), not a prospective FPR; "
            "run1 has 35-43 runs per fold, so at run level nominal 1% means k = 0 per fold and nominal 5% about 1-2 per fold")
DIRECTION = ("class 1 = simulated attack (c) or run2 (d), class 0 = real attack run1; detectors are trained to score class 1 higher; the "
             "threshold is set per fold on held-out run1 units; catch = share of class-1 units above the score exceeded by only 1% (5%) of run1 units")

app = base.app  # run_simh_control's app: these functions are added to it
_files = [(HERE / "run_simh_control.py", "/root/run_simh_control.py"), (HERE / "run_controls_a1.py", "/root/run_controls_a1.py"),
          (HERE / "uncut_inputs.py", "/root/uncut_inputs.py")]
tok_image = base.tok_image.add_local_file(*_files[0]).add_local_file(*_files[1]).add_local_file(*_files[2])
gpu_image = base.gpu_image.add_local_file(*_files[0]).add_local_file(*_files[1]).add_local_file(*_files[2])


@app.function(image=tok_image, cpu=(4, 4), memory=(16384, 16384), timeout=SKIP_TIMEOUT, retries=0, scaledown_window=IDLE_S,
              startup_timeout=START_CPU, volumes={"/b": base.b_vol, "/hf": base.hf_vol})
def skip_b_a2(want: dict, shas: dict, guard) -> dict:
    if not guard.put("skip", time.time(), skip_if_exists=True):  # the cost guard (see fit_a2)
        raise RuntimeError("skip list: a rescheduled attempt; refused by the cost guard")
    return base.skip_rows("/b", want, shas)


@app.function(image=tok_image, cpu=(4, 4), memory=(16384, 16384), timeout=SKIP_TIMEOUT, retries=0, scaledown_window=IDLE_S,
              startup_timeout=START_CPU, volumes={"/u": u_vol, "/hf": base.hf_vol})
def skip_u_a2(want: dict, shas: dict, full_len: dict, roots: dict, guard) -> dict:
    """(uncut) the same skip rule on the uncut staged inputs, plus every epoch's meta check; the same cost guard."""
    if not guard.put("skip", time.time(), skip_if_exists=True):
        raise RuntimeError("skip list: a rescheduled attempt; refused by the cost guard")
    return ui.uncut_skip("/u", want, shas, full_len, roots)


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
              startup_timeout=START_GPU, volumes={"/b": base.b_vol, "/sol": sol_vol, "/u": u_vol})
def fit_a2(L: int, readout: str, epochs: list, skip: dict, guard, wdir: str, allowed: dict, n_null: int) -> dict:
    """guard: an ephemeral modal.Dict; the first attempt of a cell claims its key, and a rescheduled attempt (crash or preemption,
    which retries=0 does not stop) finds it taken and stops before any work (the cost bound)."""
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

    if not guard.put(f"{readout}|L{L}", time.time(), skip_if_exists=True):
        raise RuntimeError(f"{readout}|L{L}: a rescheduled attempt; refused by the cost guard")
    t0 = time.time()
    X16, row_run, keep, no_rows, chunks = base._load(L, readout, epochs, skip, allowed)  # 32k: {} (corrected-B manifest); uncut: every root's chunk list
    runs = [epochs[i] for i in keep]  # epochs arrive sorted by name
    t_load = time.time() - t0
    XT = torch.from_numpy(X16).cuda()
    del X16
    N = n_null  # (uncut) A2's 100 for the uncut fits; 0 = observed only (the gate)
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
    work = [("P1", p["name"], [0]) for p in PROBLEMS] + ([("P2", p["name"], NULL) for p in PROBLEMS] if N else [])  # (uncut) A2's work list
    heldout_run2 = {}  # (detector fork) run 2 run name -> mean held-out score under the (c) fold model of its main task's fold
    heldout_rows2 = {}  # (uncut) run 2 run name -> its rows' held-out scores (per action)
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
                    heldout_rows2[runs[j]["name"]] = S2[m_].tolist()
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
        rel = f"xfer/cmd_sim_detector_uncut/{wdir}/w_{readout}_L{L}__{p_}.npz"
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
            rr_ = d["pos"][row_run[d["rows"]]]  # (uncut) per action: call-level AUROC's task bootstrap (rows of the drawn tasks)
            out["problems"][p]["call_auroc_boot"] = boot_pairs(d["oof_row"][0], d["Y"][0][rr_], [runs[d["ids"][i]]["task"] for i in rr_])
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
        dc = prep["c_simatk_vs_run1"]  # (uncut) per action: held-out run 2 rows vs (c)'s out-of-fold sim and run 1 rows
        rr_c = dc["pos"][row_run[dc["rows"]]]
        names_c = [runs[dc["ids"][i]]["name"] for i in rr_c]
        sim_rows = [float(v) for n, v in zip(names_c, dc["oof_row"][0]) if n.startswith("sim|")]
        r1_rows = [float(v) for n, v in zip(names_c, dc["oof_row"][0]) if n.startswith("run1|")]
        sim_t = [n.split("|", 1)[1] for n in names_c if n.startswith("sim|")]
        r1_t = [n.split("|", 1)[1] for n in names_c if n.startswith("run1|")]
        h_rows = [v for n in sorted(heldout_rows2) for v in heldout_rows2[n]]
        h_t = [n.split("|", 1)[1] for n in sorted(heldout_rows2) for _ in heldout_rows2[n]]
        for name, pos, pt, neg, nt in (("sim_vs_heldout_run2", sim_rows, sim_t, h_rows, h_t), ("heldout_run2_vs_run1", h_rows, h_t, r1_rows, r1_t)):
            sc = np.array([*pos, *neg])
            yy = np.array([1] * len(pos) + [0] * len(neg))
            hold[name]["call_level"] = {"auroc": base.auroc(sc, yy), "n_pos_rows": len(pos), "n_neg_rows": len(neg), "boot": boot_pairs(sc, yy, [*pt, *nt])}
        out["heldout_run2"] = {"n_runs": len(heldout_run2), "run_scores": heldout_run2, "contrasts": hold,
                               "note": "run 2 runs scored with the (c) sim-vs-run1 fold model of their main task's fold (task unseen in that model's training); "
                                       "sim and run 1 scores are (c)'s out-of-fold scores; class 1 = first world named"}
    else:
        out["heldout_run2"] = {"error": f"held-out run 2 scores missing: {len(heldout_run2)} of {len(n2)}"}
    out["seconds_total"] = round(time.time() - t0)  # (Codex r1) through the bootstraps
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
def main_a(r36: str, inputs: str = "uncut", config: str = "", n_null: int = base.N_NULL, stage: str = "fit", budget: float = 16.0, resume: str = ""):
    """(uncut) inputs: 32k (the copied construction; gate 2, --n-null 0) or uncut. stage: dry (local checks only, no Modal call),
    skip (+ the CPU skip list and meta check, then stop) or fit."""
    if inputs not in ("32k", "uncut") or stage not in ("dry", "skip", "fit") or n_null < 0 or n_null == 1:
        raise SystemExit("--inputs 32k|uncut, --stage dry|skip|fit, --n-null 0 or >= 2 (Codex r1: one draw would be classed as a single fit)")
    if (inputs == "32k") != (n_null == 0) or (inputs == "32k" and stage != "fit"):
        raise SystemExit("32k mode is the reproduction gate (observed only, --n-null 0, stage fit); uncut fits carry the null")
    config = config or (base.CONFIG if inputs == "32k" else "config/inputs_uncut_v2.json")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    if stage == "fit" and (git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--", str(HERE), f":(exclude){HERE}/results/partial")):
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
    cfg_v1 = json.loads((HERE / base.CONFIG).read_text())
    mf = json.loads(base._git_bytes(cfg_v1["real"]["manifest"]))
    allowed, uprov, same32, pilot = {}, None, None, None
    if inputs == "32k":
        epochs = [{k: e[k] for k in ("key", "world", "task", "arm", "extraction_run", "staged_inputs_run", "meta_file")} | {"batch": WORLDS[e["world"]], "cls": "attack", "vol": "b"}
                  for e in mf["epochs"] if e["world"] in WORLDS]
        shas = {e["staged_inputs_file"]: e["staged_inputs_sha256"] for e in mf["epochs"] if e["world"] in WORLDS}
        want = {}
        for e in epochs:
            want.setdefault(e["staged_inputs_run"], {}).setdefault(e["arm"], []).append(e["key"])
    else:  # (uncut) the same runs by key (the corrected-B manifest's run1 / run2 / sim keys), from their uncut roots
        u = ui.build(base.REPO, cfg, UNCUT_WORLDS, dry_run=stage != "fit")
        epochs = [e | {"cls": "attack"} for e in u["epochs"]]
        want, shas, allowed, uprov = u["want"], u["shas"], u["allowed"], u["provenance"]
        same32 = {}
        for w, b_ in WORLDS.items():
            ref_k, got = {e["key"] for e in mf["epochs"] if e["world"] == w}, {e["key"] for e in epochs if e["batch"] == b_}
            if ref_k != got:
                raise SystemExit(f"uncut {w} keys differ from the 32k detector's ({len(got - ref_k)} extra, {len(ref_k - got)} missing)")
            same32[w] = len(got)
        print(json.dumps({"stamp": stamp, "git": git_at_launch, "stage": stage, "same_runs_as_32k": same32, "routing": uprov["routing"]["mode"],
                          "roots": {k: v["epochs"] for k, v in uprov["roots"].items()}}), flush=True)
        if stage == "dry":
            return
    wd = Watchdog()
    guard_cm = modal.Dict.ephemeral()  # the cost guard: one attempt per call (skip list and each cell); closed in the finally below
    guard = guard_cm.__enter__()
    cells, failed, fatal = {}, {}, None
    try:  # one lifecycle: any failure cancels every outstanding call (independently), writes a failure record and closes the Dict
        call = (skip_b_a2.spawn(want, shas, guard) if inputs == "32k" else
                skip_u_a2.spawn(want, shas, {e["key"]: e["full_len"] for e in epochs}, {e["staged_inputs_run"]: e["extraction_run"] for e in epochs}, guard))
        wd.add("skip", call, SKIP_DEADLINE_S)
        sk = call.get(timeout=SKIP_DEADLINE_S)  # a timeout (builtin or Modal's class) is fatal: handled below
        wd.done.add("skip")
        if sk["fixed_tokens"] != base.FIXED_TOKENS:
            raise SystemExit("fixed call tokens differ from Amendment 12's")
        if inputs == "uncut":
            if sk["meta_checked"] != len(epochs):
                raise SystemExit("not every epoch's meta was checked")
            print(json.dumps({"skip_dropped_rows": sk["dropped_rows"], "meta_checked": sk["meta_checked"], "meta_tokens": sk["meta_tokens"]}), flush=True)
            if stage == "skip":
                return
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
        jobs, t_sp, walls = {}, {}, {}
        fit_deadline = GATE_DEADLINE_S if inputs == "32k" else FIT_DEADLINE_S
        ui.hard_exit_after(fit_deadline * (1 if inputs == "32k" else 2) + 300)  # (Codex r4) pilot window + the rest's window + 300 s for records
        cmeta = ui.ckpt_meta(git_at_launch, sha(HERE / config), n_null, epochs, sk["skip"], allowed)  # (4f) per-cell checkpoints; --resume <stamp>
        cdir = HERE / "results" / "partial" / f"attack_{inputs}_{resume or stamp}"  # (Codex r5) a resume appends to the resumed set
        if resume:
            for cell, (o, w) in ui.ckpt_load(HERE / "results" / "partial" / f"attack_{inputs}_{resume}", cmeta).items():
                cells[cell], walls[cell] = o, w
            print(json.dumps({"resumed_from": resume, "cells": sorted(cells)}), flush=True)
        order = [(63, "command")] + [(L, ro) for ro in base.READOUTS for L in base.LAYERS if (L, ro) != (63, "command")]  # pilot = primary cell

        def launch(keys):
            for k in keys:
                jobs[k], t_sp[k] = fit_a2.spawn(k[0], k[1], epochs, sk["skip"], guard, stamp, allowed, n_null), time.time()
                wd.add(f"{k[1]}|L{k[0]}", jobs[k], fit_deadline)

        def collect(keys):
            for k in keys:
                L, ro = k
                try:
                    cells[f"{ro}|L{L}"] = jobs[k].get(timeout=max(1.0, fit_deadline - (time.time() - t_sp[k])))
                    wd.done.add(f"{ro}|L{L}")
                    walls[f"{ro}|L{L}"] = time.time() - t_sp[k]
                    ui.ckpt_save(cdir, f"{ro}|L{L}", cells[f"{ro}|L{L}"], cmeta, walls[f"{ro}|L{L}"])
                except Exception as e:  # noqa: BLE001 - a timeout or a failed cell: recorded; the call is cancelled in the finally
                    failed[f"{ro}|L{L}"] = f"{type(e).__name__}: {str(e)[:500]} (at {time.time() - t_sp[k]:.0f} s from spawn; deadline {fit_deadline} s)"
        todo = [k for k in order if f"{k[1]}|L{k[0]}" not in cells]  # (4f) a resume runs only the missing cells
        first = todo if inputs == "32k" else ([order[0]] if order[0] in todo else [])  # (Codex r1) uncut: the pilot cell alone first
        launch(first)
        collect(first)
        if inputs == "uncut":
            wall = walls.get("command|L63", float("inf"))
            proj = wall * len(order) * RATE_H / 3600
            pilot = {"cell": "command|L63", "wall_seconds": round(wall) if wall != float("inf") else None, "container_seconds_total": cells.get("command|L63", {}).get("seconds_total"),
                     "projected_usd_all_cells": round(proj, 2), "budget_usd": budget, "rate_usd_per_h": RATE_H}
            print(json.dumps({"pilot": pilot}), flush=True)
            if failed or proj > budget:
                pilot["stopped"] = "the pilot failed" if failed else f"projected ${proj:.2f} > budget ${budget:.2f}"
            else:
                rest = [k for k in todo if k != order[0]]
                launch(rest)
                collect(rest)
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
                (HERE / "results" / f"attack_{inputs}_FAILED_{stamp}.json").open("x").write(json.dumps(
                    {"label": "RULE.md Amendment 2: FAILED before the record", "git": git_at_launch, "stamp": stamp, "error": fatal,
                     "failed": failed, "cells_done": sorted(cells), "watchdog_log": wd.log}, indent=1))
            except Exception as e:  # noqa: BLE001 - best effort; never masks the original error
                print(f"could not write the failure record: {type(e).__name__}", flush=True)
    same = {ro: len({json.dumps(cells[f"{ro}|L{L}"]["rows_per_run"], sort_keys=True) for L in base.LAYERS if f"{ro}|L{L}" in cells}) <= 1 for ro in base.READOUTS}
    consistency = {}  # run1 and simulated-attack rows per run must equal Amendment 1's (32k only: uncut runs carry more rows by design)
    if inputs == "32k":
        for k, c in cells.items():
            mine = {n: v for n, v in c["rows_per_run"].items() if n.split("|", 1)[0] in ("run1", "sim")}
            theirs = {n: v for n, v in ref["cells"][k]["rows_per_run"].items() if n.split("|", 1)[0] in ("run1", "sim")}
            consistency[k] = {"runs_compared": len(theirs), "equal": mine == theirs}
    gaps = [f"{k}|{p}|observed" for k, c in cells.items() for p, r in c["problems"].items() if not r["observed_complete"]]
    gaps += [f"{k}|{p}|null" for k, c in cells.items() for p, r in c["problems"].items() if n_null and not r["null_complete"]]  # (uncut) the null is required
    gaps += [f"{k}|heldout_run2" for k, c in cells.items() if "error" in c.get("heldout_run2", {"error": 1})]
    gaps += [f"{k}|paired" for k, c in cells.items() if not c.get("paired_c_minus_d")]
    rec = {"label": f"UNCUT detector attack-only with held-out run 2, inputs {inputs} (4f detU fork, 10-16; run_attack_heldout @ 648f33e61 + A2's null + per-action "
                    "pieces); PRELIMINARY until Codex and 0d review", "git": git_at_launch, "stamp": stamp, "inputs": inputs,
           "app": base.APP, "training_unit": a1.TRAINING_UNIT, "direction": DIRECTION, "volume": base.BVOL if inputs == "32k" else ui.VOL, "weights_volume": SOLVOL,
           "config": {"path": config, "sha256": sha(HERE / config)}, "r36": {"path": r36, "sha256": base.R36_SHA}, "t196": {"path": base.T196, "sha256": sha(REPO_T196)},
           "uncut_inputs": uprov, "same_runs_as_32k": same32,
           "consistency_reference": REF_A1 if inputs == "32k" else None,
           "settings": {"C": C0, "layers": base.LAYERS, "primary": base.PRIMARY, "n_null": n_null, "seed": base.SEED, "problems": PROBLEMS, "worlds": WORLDS,
                        "fit_timeout_s": FIT_TIMEOUT, "boot_b": BOOT_B, "boot_seed": BOOT_SEED,
                        "null_note": "uncut: A2's within-task shuffle null (n_null draws, base.SEED), refit through the whole CV; 32k: observed only (gate 2)"},
           "epochs_per_batch": counts, "skip": {k: sk[k] for k in ("dropped_rows", "staged_sha256", "meta_checked", "meta_tokens") if k in sk},
           "run_folds": {e["name"]: e["fold"] for e in epochs}, "run_keys": {e["name"]: e["key"] for e in epochs},
           "rows_per_run_equal_across_layers": same, "consistency_with_a1": consistency,
           "required_gaps": gaps, "watchdog_log": wd.log, "pilot": pilot, "resumed_from": resume or None, "checkpoints": str(cdir.relative_to(HERE)),
           "cell_wall_seconds": {k: round(v) for k, v in walls.items()}, "cells": cells, "failed": failed}
    if inputs == "uncut":  # (Codex r1) rows equal the staged expectation; observed fits and null draws certified
        rec["uncut_gates"] = ui.check_cells(cells, epochs, sk["expected_rows"])
    rfail = []
    if inputs == "32k":  # gate 2: A2's stored record (run scores, run/call metrics, saved weights) and 2e's held-out record (held-out run 2, bootstrap)
        a2 = subprocess.run(["git", "show", f"{REF_A2['commit']}:{REF_A2['path']}"], capture_output=True, check=True, cwd=HERE).stdout
        if hashlib.sha256(a2).hexdigest() != REF_A2["sha256"]:
            raise SystemExit("A2's record sha256 differs from the pin")
        a2 = json.loads(a2)
        hb = subprocess.run(["git", "show", f"{REF_H['commit']}:{REF_H['path']}"], capture_output=True, check=True, cwd=HERE).stdout
        if hashlib.sha256(hb).hexdigest() != REF_H["sha256"]:
            raise SystemExit("2e's held-out record sha256 differs from the pin")
        hr = json.loads(hb)
        repro = {}
        rfail = [f"missing cell {k}" for k in sorted(set(a2["cells"]) - set(cells))]
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
                    rb, hb_ = r["run_auroc_boot"]["ci95"], hr["cells"][k]["problems"][p]["run_auroc_boot"]["ci95"]
                    dm = max(dm, *(abs(x - y) for x, y in zip(rb, hb_)))
                    sw = c["saved_weights"][p]["max_abs_saved_weights_score_diff"]
                    repro[f"{k}|{p}"] = {"runs": len(theirs), "same_run_keys": set(mine) == set(theirs), "max_abs_run_score_diff": d, "max_abs_metric_diff": dm,
                                         "max_abs_saved_weights_score_diff": sw, "saved_folds": c["saved_weights"][p]["folds"]}
                    ok = d <= TOL_RUN and dm <= TOL_MET and sw <= TOL_SAVED and c["saved_weights"][p]["folds"] == list(range(5))
                except (KeyError, TypeError, ValueError) as e:  # a missing piece is a gate failure, recorded, never a silent pass
                    repro[f"{k}|{p}"], ok = {"error": f"{type(e).__name__}: {str(e)[:200]}"}, False
                if not ok:
                    rfail.append(f"{k}|{p}")
            try:  # held-out run 2: run scores and both contrasts (AUROC, bootstrap interval) equal 2e's record
                mh, th = c["heldout_run2"], hr["cells"][k]["heldout_run2"]
                d = max(abs(mh["run_scores"][n] - th["run_scores"][n]) for n in th["run_scores"]) if set(mh["run_scores"]) == set(th["run_scores"]) else float("inf")
                dm = max(max(abs(mh["contrasts"][n]["auroc"] - v["auroc"]), *(abs(x - y) for x, y in zip(mh["contrasts"][n]["boot"]["ci95"], v["boot"]["ci95"])))
                         for n, v in th["contrasts"].items())
                repro[f"{k}|heldout_run2"] = {"runs": len(th["run_scores"]), "max_abs_run_score_diff": d, "max_abs_metric_diff": dm}
                ok = d <= TOL_RUN and dm <= TOL_MET
            except (KeyError, TypeError, ValueError) as e:
                repro[f"{k}|heldout_run2"], ok = {"error": f"{type(e).__name__}: {str(e)[:200]}"}, False
            if not ok:
                rfail.append(f"{k}|heldout_run2")
        rec["reproduction_of_a2_and_heldout"] = {"references": [REF_A2, REF_H], "tolerances": {"run_scores": TOL_RUN, "metrics": TOL_MET, "saved_weights_vs_held_out_rows": TOL_SAVED},
                                                 "cells": repro, "failed": rfail, "pass": not rfail}
    dest = HERE / "results" / f"attack_{inputs}_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
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
            row["run_auroc_ci95"] = r.get("run_auroc_boot", {}).get("ci95")
            row["call_auroc_ci95"] = r.get("call_auroc_boot", {}).get("ci95")
            print(json.dumps(row), flush=True)
    for k, c in cells.items():
        print(json.dumps({"cell": k, "paired_c_minus_d": c.get("paired_c_minus_d")}), flush=True)
        print(json.dumps({"cell": k, "heldout_run2": {n: {"auroc": v["auroc"], "ci95": v["boot"]["ci95"], "call_auroc": v.get("call_level", {}).get("auroc"),
                                                          "call_ci95": v.get("call_level", {}).get("boot", {}).get("ci95")}
                                                      for n, v in c.get("heldout_run2", {}).get("contrasts", {}).items()},
                          "seconds": c["seconds"], "load_seconds": c["load_seconds"], "rows": c["rows"]}), flush=True)
    print(json.dumps({"consistency_with_a1": consistency, "required_gaps": gaps, "failed": failed, "rows_equal": same}), flush=True)
    if inputs == "32k":
        print(json.dumps({"reproduction": {k: rec["reproduction_of_a2_and_heldout"][k] for k in ("failed", "pass")}}), flush=True)
    print(dest.name)
    if rfail:
        raise SystemExit(f"GATE 2 (attack): the refit does not reproduce A2's / 2e's stored records for {rfail}: STOP and tell 4f")
    if inputs == "uncut":
        print(json.dumps({"pilot": pilot, "uncut_gates": rec["uncut_gates"]}), flush=True)
        if pilot and pilot.get("stopped"):
            raise SystemExit(f"PILOT STOP: {pilot['stopped']}")
        if rec["uncut_gates"]["failures"] or len(cells) != 4:
            raise SystemExit(f"uncut gates {rec['uncut_gates']['failures'][:5]} or {len(cells)} of 4 cells")
    if failed or gaps or not all(same.values()) or not all(v["equal"] for v in consistency.values()):
        raise SystemExit("a cell failed, a required piece is missing, rows differ across layers, or run1/simulated-attack rows differ from Amendment 1's")
