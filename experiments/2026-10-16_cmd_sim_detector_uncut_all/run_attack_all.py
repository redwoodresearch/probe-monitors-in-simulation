"""ALL-SETS uncut sim-vs-real detector, attack only (detAll fork of 4f, 10-16). Pre-statement: 09 RULE.md @ 3685810b4 L4693 ("All-tasks
sim-vs-real detector"): recipe pinned from 648f33e61, only the cohort and the fold map change. Code: detU's run_attack_uncut.py
(T196 uncut job, aswin/cmd-sim-detector-uncut-4f @ 816958fda; its gate 2 reproduced 648f33e61 exactly and its records passed an
independent recheck) with these changes only:
- inputs: five per-set configs (S_old = detU's inputs_uncut_v2 copy; S_new, S_newmains, S_paper493 w1 + w23), read with
  uncut_inputs_all.build (detU's loader + the "set" field + the 52 refused S4 orphans counted in the whole-arm check, never read);
- cohort: the frozen 32k cohort by key (exclusions_2026-10-16_033635.json final_cohort_keys, 1,313 cells; 0d 10-16: the full-length
  cohort is the same cells). Runs outside it are dropped by key and counted. Every cohort cell must have exactly one run1, run2 and
  sim run (else STOP);
- cell names: run2 rows' "_r2" suffix and "_redraw<n>" are stripped to the cell (S_paper493), by key; the cell's set must be the run's;
- folds: the own_all_rs re-split fold list @ ec56720ad (by_main), main of a cell from the fold list's own pinned cell files; coverage
  STOP if a cohort main has no fold;
- the shuffle null and every bootstrap group by MAIN task (e["task"] = main): 09's whole-run within-main shuffle, 100 draws;
- zero-row runs (no readout rows) are dropped from that cell's AUROC by key (4f/09), listed and counted by world and by set;
- no 32k mode (no reproduction gate here; detU's gate 2 covers the code), app / weights Volume renamed, timeouts and memory scaled.
Run (from this folder; caffeinate keeps the laptop awake; checkpoints per cell, --resume <stamp>):
  caffeinate -i <modal> run experiments/2026-10-16_cmd_sim_detector_uncut_all/run_attack_all.py::main_all --stage dry|skip|fit [--budget 90]
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
import uncut_inputs as ui  # noqa: E402  (unchanged copy)
import uncut_inputs_all as uia  # noqa: E402  (detAll: + set field, + refused S4 orphans)

WORLDS = {"run1": "run1", "run2": "run2", "sim": "sim_attack"}  # corrected-B manifest world -> batch label
PROBLEMS = [{"name": "c_simatk_vs_run1", "pos": ["sim_attack"], "neg": ["run1"]},
            {"name": "d_run2_vs_run1", "pos": ["run2"], "neg": ["run1"]}]
C0 = 0.01
SOLVOL = "probe-shift_aswin-surya_sub-cmdsimdet_uncutall_w_kill-2026-11-14"  # (detAll) this fork's own weights Volume
sol_vol = modal.Volume.from_name(SOLVOL, version=2, create_if_missing=True)
u_vol = modal.Volume.from_name(ui.VOL, version=2).read_only()
UNCUT_WORLDS = {"run1": "run1", "run2": "run2", "sim": "sim_attack"}
RATE_H = 4.53  # $/h: H100 3.95 + 4 CPU x 0.0473 + 48 GiB x 0.008 (Modal list prices), for the pilot's projection
FIT_TIMEOUT, MARGIN_S, SKIP_TIMEOUT, IDLE_S, START_GPU, START_CPU = 7200, 600, 3600, 2, 90, 60  # (detAll) ~7x T196 uncut's rows (its cells took 254-323 s)
SKIP_DEADLINE_S, FIT_DEADLINE_S = 3900, 7500  # overall deadlines from spawn: outstanding calls are cancelled (containers terminated) after them
BOOT_B, BOOT_SEED = 2000, 20261015  # the task-paired bootstrap of (c) - (d)
FPR_NOTE = ("realized on the same real-attack run1 units that set each fold's threshold (in-sample for the threshold), not a prospective FPR; "
            "(detAll) run1 has about 260 runs per fold (1,313 over 5 folds), so at run level nominal 1% means k = 2 per fold")
DIRECTION = ("class 1 = simulated attack (c) or run2 (d), class 0 = real attack run1; detectors are trained to score class 1 higher; the "
             "threshold is set per fold on held-out run1 units; catch = share of class-1 units above the score exceeded by only 1% (5%) of run1 units")

app = base.app  # run_simh_control's app: these functions are added to it
_files = [(HERE / "run_simh_control.py", "/root/run_simh_control.py"), (HERE / "run_controls_a1.py", "/root/run_controls_a1.py"),
          (HERE / "uncut_inputs.py", "/root/uncut_inputs.py"), (HERE / "uncut_inputs_all.py", "/root/uncut_inputs_all.py")]  # (detAll) this folder's copies;
# (Codex runner r1) uncut_inputs_all is imported at module scope, so the remote containers need it too
tok_image = base.tok_image
gpu_image = base.gpu_image
for _f in _files:
    tok_image, gpu_image = tok_image.add_local_file(*_f), gpu_image.add_local_file(*_f)


@app.function(image=tok_image, cpu=(4, 4), memory=(32768, 32768), timeout=SKIP_TIMEOUT, retries=0, scaledown_window=IDLE_S,
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


@app.function(image=gpu_image, gpu="H100!", cpu=(4, 4), memory=(49152, 49152), timeout=FIT_TIMEOUT, retries=0, scaledown_window=IDLE_S,
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
            W, b, cert = tp.fit(Xtr, Ytr, C0, allow_uncertified=True, return_certificate=True, max_iter=20000)
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
        main_of_name = {r["name"]: r["task"] for r in runs}  # (detAll) task = the main task: bootstrap groups by main
        task = lambda n: main_of_name[n]  # noqa: E731
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
        sim_t = [task(n) for n in names_c if n.startswith("sim|")]
        r1_t = [task(n) for n in names_c if n.startswith("run1|")]
        h_rows = [v for n in sorted(heldout_rows2) for v in heldout_rows2[n]]
        h_t = [task(n) for n in sorted(heldout_rows2) for _ in heldout_rows2[n]]
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


COHORT = {"commit": "adbc791b6", "path": "experiments/2026-10-15_fpr_sweep_507_prep/results/exclusions_2026-10-16_033635.json", "sha256_prefix": "4511f6c928c91f10"}
FOLDS = {"commit": "ec56720adde66baabe642a7fc842172c83737679", "path": "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json"}
SET_CONFIGS = [("S_old", "config/inputs_all_S_old_v1.json"), ("S_new", "config/inputs_all_S_new_v1.json"), ("S_newmains", "config/inputs_all_S_newmains_v1.json"),
               ("S_paper493", "config/inputs_all_S_paper493_w1_v1.json"), ("S_paper493", "config/inputs_all_S_paper493_w23_v1.json")]
SETS = ("S_old", "S_new", "S_newmains", "S_paper493")


def _gitb(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout


def cell_of(e: dict) -> str:
    """(detAll) the cohort cell of a run: run2 rows of S_paper493 carry "_r2", and two redraw tasks carry "_redraw<n>"."""
    import re

    t = re.sub(r"_r2$", "", e["task"]) if e["world"] == "run2" else e["task"]
    return re.sub(r"_redraw[0-9]+$", "", t)


def build_inputs(dry: bool) -> dict:
    """Local, fail-closed: per-set loader builds, the cohort filter, cell normalization, folds and mains. No Modal call."""
    raw = _gitb(COHORT["commit"], COHORT["path"])
    if not hashlib.sha256(raw).hexdigest().startswith(COHORT["sha256_prefix"]):
        raise SystemExit("cohort (exclusions) file: sha256 is not the relayed one")
    coh = json.loads(raw)
    if coh.get("pending") or coh["N_all_sets"]["final"] != sum(len(coh["final_cohort_keys"][s]["cells"]) for s in SETS):
        raise SystemExit("cohort file: pending entries, or final count != cells listed")
    cohort = {s: set(coh["final_cohort_keys"][s]["cells"]) for s in SETS}
    fraw = _gitb(FOLDS["commit"], FOLDS["path"])
    F = json.loads(fraw)
    fold_of_main = {m: v[0] for m, v in F["by_main"].items()}
    main_of_cell = {}
    for s, (c, p, sh) in F["pins"]["cells"].items():  # the fold list's own cell files (sha256 pinned there)
        b = _gitb(c, p)
        if hashlib.sha256(b).hexdigest() != sh:
            raise SystemExit(f"{p}: sha256 differs from the fold list's pin")
        for x in json.loads(b)["cells"]:
            main_of_cell[x["key"]] = x["main"]
    c, p, pre = F["pins"]["sp493_facts"]
    b = _gitb(c, p)
    if not hashlib.sha256(b).hexdigest().startswith(pre):
        raise SystemExit(f"{p}: sha256 differs from the fold list's pin")
    for r in json.loads(b)["rows"]:
        if main_of_cell.setdefault(r["key"], r["main"]) != r["main"]:
            raise SystemExit(f"{r['key']}: two mains")
    epochs, want, shas, allowed, prov = [], {}, {}, {}, {}
    for s, path in SET_CONFIGS:
        cfg = json.loads((HERE / path).read_text())
        inv = ui.git_json(base.REPO, cfg["inventory"])
        ro = uia.refused_out_counts(inv, cfg["routing"]["records"], base.REPO) if s == "S_newmains" else {}  # the S4 orphans are nm360 S4 runs
        u = uia.build(base.REPO, cfg, UNCUT_WORLDS, dry_run=dry, refused_out=ro)
        for st, arms in u["want"].items():
            if st in want:
                raise SystemExit(f"staged set {st} in two configs")
            want[st] = arms
        for k, v in u["shas"].items():
            if shas.setdefault(k, v) != v:
                raise SystemExit(f"{k}: two staged shas")
        for k, v in u["allowed"].items():
            if allowed.setdefault(k, v) != v:
                raise SystemExit(f"{k}: two chunk lists")
        prov[path] = {"set": s, "config_sha256": hashlib.sha256((HERE / path).read_bytes()).hexdigest(), "refused_out": {f"{a}/{b_}": n for (a, b_), n in ro.items()},
                      "provenance": u["provenance"]}
        epochs += [e | {"cls": "attack", "set": s} for e in u["epochs"]]
    if len({e["key"] for e in epochs}) != len(epochs):
        raise SystemExit("a run key in two configs")
    renamed, outside, kept = {}, [], []
    for e in epochs:
        cell = cell_of(e)
        if cell != e["task"]:
            renamed[e["key"]] = {"task": e["task"], "cell": cell}
        if F["cell_set"].get(cell) != e["set"]:
            raise SystemExit(f"{e['key']}: cell {cell} is not a {e['set']} cell of the fold list")
        if cell not in cohort[e["set"]]:
            outside.append({"key": e["key"], "set": e["set"], "world": e["world"], "cell": cell})
            continue
        main = main_of_cell[cell]
        if main not in fold_of_main:
            raise SystemExit(f"COVERAGE STOP: main {main} (cell {cell}) has no fold in the fold list")
        kept.append(e | {"cell": cell, "task": main, "fold": fold_of_main[main], "name": f"{e['world']}|{cell}"})
    per = {}
    for e in kept:
        per.setdefault((e["set"], e["cell"]), []).append(e["world"])
    bad = sorted(f"{s}|{c}" for s in SETS for c in cohort[s] if sorted(per.get((s, c), [])) != ["run1", "run2", "sim"])
    if bad:
        raise SystemExit(f"{len(bad)} cohort cells without exactly one run1, run2 and sim run at full length, e.g. {bad[:5]}")
    if len({e["name"] for e in kept}) != len(kept):
        raise SystemExit("duplicate run name")
    kept.sort(key=lambda e: e["name"])
    want_k = {}
    for e in kept:
        want_k.setdefault(e["staged_inputs_run"], {}).setdefault(e["arm"], []).append(e["key"])
    shas_k = {f"{st}/inputs/{arm}.jsonl": shas[f"{st}/inputs/{arm}.jsonl"] for st, arms in want_k.items() for arm in arms}
    allowed_k = {e["extraction_run"]: allowed[e["extraction_run"]] for e in kept}
    n = len(set().union(*cohort.values()))
    return {"epochs": kept, "want": want_k, "shas": shas_k, "allowed": allowed_k,
            "expect": {"run1": n, "run2": n, "sim_attack": n},
            "provenance": {"cohort": {"commit": COHORT["commit"], "path": COHORT["path"], "sha256": hashlib.sha256(raw).hexdigest(),
                                      "cells": {s: len(cohort[s]) for s in SETS}},
                           "folds": FOLDS | {"sha256": hashlib.sha256(fraw).hexdigest(), "mains_used": len({e["task"] for e in kept})},
                           "configs": prov, "renamed_to_cell": renamed,
                           "outside_cohort_dropped": {"n": len(outside), "by_set_world": {f"{s}|{w}": sum(1 for o in outside if o["set"] == s and o["world"] == w)
                                                                                          for s in SETS for w in ("run1", "run2", "sim")}, "keys": outside}}}


def zero_row_summary(cells: dict, epochs: list) -> dict:
    """(detAll; 4f + 09) runs with no rows in a cell are dropped from that cell's AUROC by key; counted by world and by set."""
    by_name = {e["name"]: e for e in epochs}
    out = {}
    for k, c in cells.items():
        z = sorted(c.get("runs_without_rows", []))
        out[k] = {"n": len(z), "by_world": {w: sum(1 for n in z if by_name[n]["world"] == w) for w in ("run1", "run2", "sim")},
                  "by_set": {s: sum(1 for n in z if by_name[n]["set"] == s) for s in SETS}, "runs": [{"name": n, "key": by_name[n]["key"], "set": by_name[n]["set"]} for n in z]}
    return out


@app.local_entrypoint()
def main_all(n_null: int = base.N_NULL, stage: str = "fit", budget: float = 90.0, resume: str = ""):
    """stage: dry (local checks only, no Modal call), skip (+ the CPU skip list and meta check, then stop) or fit (pilot cell first;
    the rest only if the pilot's projection is within the budget)."""
    if stage not in ("dry", "skip", "fit") or n_null < 2:
        raise SystemExit("--stage dry|skip|fit; --n-null >= 2 (the null is required)")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    if stage == "fit" and (git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--", str(HERE), f":(exclude){HERE}/results/partial")):
        raise SystemExit("uncommitted tracked changes, or uncommitted files in this experiment folder")
    git_at_launch = git("rev-parse", "HEAD")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    u = build_inputs(dry=stage != "fit")
    epochs, want, shas, allowed, uprov, EXPECT = u["epochs"], u["want"], u["shas"], u["allowed"], u["provenance"], u["expect"]
    counts = {b_: sum(e["batch"] == b_ for e in epochs) for b_ in sorted({e["batch"] for e in epochs})}
    if counts != EXPECT:
        raise SystemExit(f"epoch counts {counts} != {EXPECT}")
    print(json.dumps({"stamp": stamp, "git": git_at_launch, "stage": stage, "epochs_per_batch": counts, "per_set": {s: sum(e["set"] == s for e in epochs) for s in SETS},
                      "mains": uprov["folds"]["mains_used"], "outside_cohort_dropped": uprov["outside_cohort_dropped"]["by_set_world"],
                      "renamed_to_cell": len(uprov["renamed_to_cell"]), "roots": len(allowed)}), flush=True)
    if stage == "dry":
        return
    wd = Watchdog()
    guard_cm = modal.Dict.ephemeral()  # the cost guard: one attempt per call (skip list and each cell); closed in the finally below
    guard = guard_cm.__enter__()
    cells, failed, fatal, pilot, walls = {}, {}, None, None, {}
    cdir = HERE / "results" / "partial" / f"attack_all_{resume or stamp}"
    try:
        call = skip_u_a2.spawn(want, shas, {e["key"]: e["full_len"] for e in epochs}, {e["staged_inputs_run"]: e["extraction_run"] for e in epochs}, guard)
        wd.add("skip", call, SKIP_DEADLINE_S)
        sk = call.get(timeout=SKIP_DEADLINE_S)
        wd.done.add("skip")
        if sk["fixed_tokens"] != base.FIXED_TOKENS:
            raise SystemExit("fixed call tokens differ from Amendment 12's")
        if sk["meta_checked"] != len(epochs):
            raise SystemExit("not every epoch's meta was checked")
        print(json.dumps({"skip_dropped_rows": sk["dropped_rows"], "meta_checked": sk["meta_checked"], "meta_tokens": sk["meta_tokens"]}), flush=True)
        if stage == "skip":
            return
        jobs, t_sp = {}, {}
        ui.hard_exit_after(FIT_DEADLINE_S * 2 + 300)  # pilot window + the rest's window + 300 s for records
        cmeta = ui.ckpt_meta(git_at_launch, sha(HERE / "run_attack_all.py"), n_null, epochs, sk["skip"], allowed)
        if resume:
            for cell, (o, w) in ui.ckpt_load(cdir, cmeta).items():
                cells[cell], walls[cell] = o, w
            print(json.dumps({"resumed_from": resume, "cells": sorted(cells)}), flush=True)
        order = [(63, "command")] + [(L, ro) for ro in base.READOUTS for L in base.LAYERS if (L, ro) != (63, "command")]  # pilot = primary cell

        def launch(keys):
            for k in keys:
                jobs[k], t_sp[k] = fit_a2.spawn(k[0], k[1], epochs, sk["skip"], guard, stamp, allowed, n_null), time.time()
                wd.add(f"{k[1]}|L{k[0]}", jobs[k], FIT_DEADLINE_S)

        def collect(keys):
            for k in keys:
                L, ro = k
                try:
                    cells[f"{ro}|L{L}"] = jobs[k].get(timeout=max(1.0, FIT_DEADLINE_S - (time.time() - t_sp[k])))
                    wd.done.add(f"{ro}|L{L}")
                    walls[f"{ro}|L{L}"] = time.time() - t_sp[k]
                    ui.ckpt_save(cdir, f"{ro}|L{L}", cells[f"{ro}|L{L}"], cmeta, walls[f"{ro}|L{L}"])
                except Exception as e:  # noqa: BLE001 - a timeout or a failed cell: recorded; the call is cancelled in the finally
                    failed[f"{ro}|L{L}"] = f"{type(e).__name__}: {str(e)[:500]} (at {time.time() - t_sp[k]:.0f} s from spawn; deadline {FIT_DEADLINE_S} s)"
        todo = [k for k in order if f"{k[1]}|L{k[0]}" not in cells]
        first = [order[0]] if order[0] in todo else []
        launch(first)
        collect(first)
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
        try:
            wd.cancel_outstanding("fatal error" if fatal else "not finished", join_s=30.0)
        except Exception as e:  # noqa: BLE001 - logged in the record
            wd.log.append(f"cancel_outstanding failed: {type(e).__name__}: {str(e)[:200]}")
        wd.stop()
        try:
            guard_cm.__exit__(None, None, None)
        except Exception as e:  # noqa: BLE001 - logged; the Dict is ephemeral
            wd.log.append(f"guard close failed: {type(e).__name__}")
        if fatal is not None:  # a sanitized failure record (no scores), best effort; the original error propagates
            try:
                (HERE / "results" / f"attack_all_FAILED_{stamp}.json").open("x").write(json.dumps(
                    {"label": "detAll: FAILED before the record", "git": git_at_launch, "stamp": stamp, "error": fatal,
                     "failed": failed, "cells_done": sorted(cells), "watchdog_log": wd.log}, indent=1))
            except Exception as e:  # noqa: BLE001
                print(f"could not write the failure record: {type(e).__name__}", flush=True)
    same = {ro: len({json.dumps(cells[f"{ro}|L{L}"]["rows_per_run"], sort_keys=True) for L in base.LAYERS if f"{ro}|L{L}" in cells}) <= 1 for ro in base.READOUTS}
    gaps = [f"{k}|{p}|observed" for k, c in cells.items() for p, r in c["problems"].items() if not r["observed_complete"]]
    gaps += [f"{k}|{p}|null" for k, c in cells.items() for p, r in c["problems"].items() if not r["null_complete"]]
    gaps += [f"{k}|heldout_run2" for k, c in cells.items() if "error" in c.get("heldout_run2", {"error": 1})]
    gaps += [f"{k}|paired" for k, c in cells.items() if not c.get("paired_c_minus_d")]
    rec = {"label": "ALL-SETS UNCUT detector, attack only, with held-out run 2 (detAll fork of 4f, 10-16; 09 RULE @ 3685810b4 L4693; detU's code @ 816958fda "
                    "with the cohort and fold map changed); PRELIMINARY until the independent recheck", "git": git_at_launch, "stamp": stamp,
           "app": base.APP, "training_unit": a1.TRAINING_UNIT, "direction": DIRECTION, "volume": ui.VOL, "weights_volume": SOLVOL,
           "inputs": uprov,
           "settings": {"C": C0, "layers": base.LAYERS, "primary": base.PRIMARY, "n_null": n_null, "seed": base.SEED, "problems": PROBLEMS, "worlds": WORLDS,
                        "fit_timeout_s": FIT_TIMEOUT, "boot_b": BOOT_B, "boot_seed": BOOT_SEED,
                        "null_note": "09 @ 3685810b4: whole-run labels shuffled within each MAIN task (task = main here), class counts preserved, n_null draws "
                                     "(base.SEED), refit through the whole CV; every bootstrap draws main tasks",
                        "zero_row_rule": "a run with no rows in a cell has no run score: dropped from that cell's AUROC by key (4f + 09, 10-16)"},
           "epochs_per_batch": counts, "per_set": {s: sum(e["set"] == s for e in epochs) for s in SETS},
           "skip": {k: sk[k] for k in ("dropped_rows", "staged_sha256", "meta_checked", "meta_tokens") if k in sk},
           "run_folds": {e["name"]: e["fold"] for e in epochs}, "run_keys": {e["name"]: e["key"] for e in epochs},
           "run_set": {e["name"]: e["set"] for e in epochs}, "run_main": {e["name"]: e["task"] for e in epochs},
           "rows_per_run_equal_across_layers": same, "zero_row_runs": zero_row_summary(cells, epochs),
           "required_gaps": gaps, "watchdog_log": wd.log, "pilot": pilot, "resumed_from": resume or None, "checkpoints": str(cdir.relative_to(HERE)),
           "cell_wall_seconds": {k: round(v) for k, v in walls.items()}, "cells": cells, "failed": failed}
    rec["uncut_gates"] = ui.check_cells(cells, epochs, sk["expected_rows"])
    dest = HERE / "results" / f"attack_all_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({"record": dest.name, "pilot": pilot, "required_gaps": gaps, "failed": failed, "rows_equal": same,
                      "zero_row_runs": {k: {x: v[x] for x in ("n", "by_world", "by_set")} for k, v in rec["zero_row_runs"].items()},
                      "uncut_gates_failures": rec["uncut_gates"]["failures"][:5]}), flush=True)
    if pilot and pilot.get("stopped"):
        raise SystemExit(f"PILOT STOP: {pilot['stopped']}")
    if rec["uncut_gates"]["failures"] or len(cells) != 4 or gaps:
        raise SystemExit(f"gates {rec['uncut_gates']['failures'][:5]}, gaps {gaps[:5]} or {len(cells)} of 4 cells")
