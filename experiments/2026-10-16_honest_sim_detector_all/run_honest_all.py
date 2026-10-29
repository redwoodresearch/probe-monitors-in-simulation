"""ALL-TASKS honest-run sim-vs-real detector (a1, 10-16; POST HOC; RULE.md here). Aswin via 0d: "i don't want subset of tasks, take everything
together"; "that's fine i give upermission for honest run detector" (GPU, cap $40). Code: 4f's all-sets attack runner run_attack_all.py @ 59acf625e
(the code of attack_all_2026-10-16_042630.json, iteration cap 5,000; NOT the held 960ab0ed3) with these changes only:
- inputs: the honest pools of every set (sim honest, real honest A, real honest B; epoch 1 + admitted epoch 2), built REMOTELY by
  honest_inputs.build_all from committed files shipped as bytes (config/sources_v1.json, each git show'd and sha256-checked here); 4f's loader
  (uncut_inputs_all.build, unchanged copy) per source; folds by main from the own_all_rs re-split @ ec56720ad (coverage STOP);
- problems: simh_vs_b (sim honest = class 1 vs real honest B = class 0) and a_vs_b (real honest A vs real honest B: the real-vs-real control);
- no held-out run 2 block; the task-paired bootstrap is simh_vs_b minus a_vs_b; app, weights Volume and record names changed.
Unchanged: C = 0.01, certification max |grad| <= 1e-6 (else uncertified, counted), run score = mean of row logits, per-action AUROC, 100
within-main shuffles, 2,000-draw main bootstrap (seed 20261015), cells L48 / L63 x command / reply (primary command L63, reply L48), the pilot
cell first with a cost projection, the cost guard, the watchdog and per-cell checkpoints (--resume <stamp>). Stored activations only; no text.
Run (from this folder; caffeinate keeps the laptop awake):
  caffeinate -i <modal> run experiments/2026-10-16_honest_sim_detector_all/run_honest_all.py::main_h --stage dry|skip|fit [--budget 30]
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
import run_simh_control as base  # noqa: E402  (unchanged copy except the app name)
import run_controls_a1 as a1  # noqa: E402  (unchanged copy)
import uncut_inputs as ui  # noqa: E402  (unchanged copy)

PROBLEMS = [{"name": "simh_vs_b", "pos": ["sim"], "neg": ["B"]},
            {"name": "a_vs_b", "pos": ["A"], "neg": ["B"]}]
C0 = 0.01
SOLVOL = "probe-shift_aswin-surya_sub-honsimdet_routec_w_kill-2026-11-15"  # this job's own weights Volume (route C rerun 2026-10-18)
sol_vol = modal.Volume.from_name(SOLVOL, version=2, create_if_missing=True)
u_vol = modal.Volume.from_name(ui.VOL, version=2).read_only()
SOURCES = "config/sources_v2.json"  # route C rerun 2026-10-18 (make_sources_v2.sh); v1 = the 10-16 run's
EXPECT = {"sim": 869, "A": 1516, "B": 1714}  # RULE.md "Pools" (849 sim) + route C's 20 pilot sim-honest runs in S_old (routec_keys.sold_sim)
OUT = "honest_all_routec"  # record / checkpoint / failure-record name prefix (route C rerun)
# 09's STOPs (b)+(c): the 10-16 run's record, shipped as bytes (git show + sha256 here, as analyze_honest_slices.py:27-28 pins it) and
# parsed only in the container by routec_keys.prior_match (keys / folds / sets only; the laptop never parses it)
PRIOR = {"commit": "8b0b18e2ab61ab911b2a7a46bcf09e46a578ed80", "path": "experiments/2026-10-16_honest_sim_detector_all/results/honest_all_2026-10-16_120353.json",
         "sha256": "c9ec2530e368c990136d536aa5ff74aefc3be44615698ed00f913f7b52d97b14", "role": "prior_record"}
RATE_H = 4.53  # $/h: H100 3.95 + 4 CPU x 0.0473 + 48 GiB x 0.008 (Modal list prices), for the pilot's projection
FIT_TIMEOUT, MARGIN_S, SKIP_TIMEOUT, IDLE_S, START_GPU, START_CPU = 7200, 600, 3600, 2, 90, 60
SKIP_DEADLINE_S, FIT_DEADLINE_S, PREP_DEADLINE_S = 3900, 7500, 1200  # overall deadlines from spawn: outstanding calls are cancelled after them
BOOT_B, BOOT_SEED = 2000, 20261015  # the task-paired bootstrap of simh_vs_b - a_vs_b
FPR_NOTE = ("realized on the same real honest B units that set each fold's threshold (in-sample for the threshold), not a prospective FPR; "
            "B has about 340 runs per fold (1,714 over 5 folds), so at run level nominal 1% means k = 3 per fold")
DIRECTION = ("class 1 = simulated honest (simh_vs_b) or real honest A (a_vs_b), class 0 = real honest B; detectors are trained to score class 1 "
             "higher; the threshold is set per fold on held-out B units; catch = share of class-1 units above the score exceeded by only 1% (5%) of B units")

app = base.app  # run_simh_control's app: these functions are added to it
_files = [(HERE / "run_simh_control.py", "/root/run_simh_control.py"), (HERE / "run_controls_a1.py", "/root/run_controls_a1.py"),
          (HERE / "uncut_inputs.py", "/root/uncut_inputs.py"), (HERE / "uncut_inputs_all.py", "/root/uncut_inputs_all.py"),
          (HERE / "honest_inputs.py", "/root/honest_inputs.py"), (HERE / "routec_keys.py", "/root/routec_keys.py")]
tok_image = base.tok_image
gpu_image = base.gpu_image
for _f in _files:
    tok_image, gpu_image = tok_image.add_local_file(*_f), gpu_image.add_local_file(*_f)


@app.function(image=tok_image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, scaledown_window=IDLE_S, startup_timeout=START_CPU)
def prep_h(files: list, data: dict, src: dict, guard) -> dict:
    """The input build (honest_inputs.build_all) on the shipped committed files; no Volume is read."""
    if not guard.put("prep", time.time(), skip_if_exists=True):
        raise RuntimeError("input build: a rescheduled attempt; refused by the cost guard")
    import honest_inputs

    return honest_inputs.build_all(files, data, src)


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
        rel = f"xfer/honest_sim_detector_all/{wdir}/w_{readout}_L{L}__{p_}.npz"
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
                    res[lvl][f"catch_{a}"]["fpr_note"] = FPR_NOTE  # class 0 = real honest B
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
    out["paired_simh_minus_ab"] = _paired(prep, runs, row_run) if all(r.get("run_level") for r in out["problems"].values()) else None
    # (detector fork) task-bootstrap 95% intervals on run AUROC
    for p, d in prep.items():
        if out["problems"][p].get("run_scores"):
            out["problems"][p]["run_auroc_boot"] = boot_pairs(d["oof"][0], d["Y"][0], [runs[j]["task"] for j in d["ids"]])
            rr_ = d["pos"][row_run[d["rows"]]]  # (uncut) per action: call-level AUROC's task bootstrap (rows of the drawn tasks)
            out["problems"][p]["call_auroc_boot"] = boot_pairs(d["oof_row"][0], d["Y"][0][rr_], [runs[d["ids"][i]]["task"] for i in rr_])
    out["seconds_total"] = round(time.time() - t0)  # (Codex r1) through the bootstraps
    crel = f"xfer/honest_sim_detector_all/{wdir}/cell_{readout}_L{L}.json"  # (0d 10-16) the finished cell, durable if the client drops
    Path("/sol", crel).parent.mkdir(parents=True, exist_ok=True)
    Path("/sol", crel).open("x").write(json.dumps(out))
    sol_vol.commit()
    return out | {"volume_copy": {"volume": SOLVOL, "path": crel, "note": "this cell's result as returned, minus this field"}}


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
    """Task-paired bootstrap of simh_vs_b (c) - a_vs_b (d) on the fixed out-of-fold scores (no refit): BOOT_B draws of the main tasks with replacement
    (seed BOOT_SEED); each draw takes every run (and row) of the drawn tasks, with multiplicity, in both problems, and recomputes
    run-level AUROC and catch at 1% and 5% and call-level AUROC and catch at 5% (thresholds per fold on the drawn B units). A draw
    whose metric cannot be computed (a fold without B or a class missing) is counted, never dropped silently."""
    import numpy as np

    def index(d):
        rt, rw = {}, {}
        for i, j in enumerate(d["ids"]):
            rt.setdefault(runs[j]["task"], []).append(i)
        for r_i, i in enumerate(d["pos"][row_run[d["rows"]]]):
            rw.setdefault(runs[d["ids"][i]]["task"], []).append(r_i)
        return {t: np.array(v) for t, v in rt.items()}, {t: np.array(v) for t, v in rw.items()}

    probs = {"c": prep["simh_vs_b"], "d": prep["a_vs_b"]}  # c = simh_vs_b, d = a_vs_b (4f's key names kept)
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


SETS = ("S_old", "S_new", "S_newmains", "S_paper493")


def _gitb(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=HERE).stdout


def zero_row_summary(cells: dict, epochs: list) -> dict:
    """Runs with no rows in a cell are dropped from that cell's AUROC by key (4f + 09); counted by class and by set."""
    by_name = {e["name"]: e for e in epochs}
    out = {}
    for k, c in cells.items():
        z = sorted(c.get("runs_without_rows", []))
        out[k] = {"n": len(z), "by_class": {w: sum(1 for n in z if by_name[n]["batch"] == w) for w in EXPECT},
                  "by_set": {s: sum(1 for n in z if by_name[n]["set"] == s) for s in SETS}, "runs": [{"name": n, "key": by_name[n]["key"], "set": by_name[n]["set"]} for n in z]}
    return out


@app.local_entrypoint()
def main_h(n_null: int = base.N_NULL, stage: str = "fit", budget: float = 30.0, resume: str = ""):
    """stage: dry (the remote input build only: one CPU call), skip (+ the CPU skip list and meta check, then stop) or fit (pilot cell
    first; the rest only if the pilot's projection is within the budget). The laptop only git-shows the listed files and checks their sha256."""
    if stage not in ("dry", "skip", "fit") or n_null < 2:
        raise SystemExit("--stage dry|skip|fit; --n-null >= 2 (the null is required)")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    if stage == "fit" and (git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--", str(HERE), f":(exclude){HERE}/results/partial")):
        raise SystemExit("uncommitted tracked changes, or uncommitted files in this experiment folder")
    git_at_launch = git("rev-parse", "HEAD")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    src = json.loads((HERE / SOURCES).read_text())  # the launch list (make_sources.sh): which committed files to ship
    if any(f["role"] == PRIOR["role"] for f in src["files"]):
        raise SystemExit(f"{SOURCES} already lists a {PRIOR['role']}")
    src["files"] = src["files"] + [PRIOR]  # route C rerun: the 10-16 record for routec_keys.prior_match (bytes only here)
    data = {}
    for f in src["files"]:
        b = _gitb(f["commit"], f["path"])
        if hashlib.sha256(b).hexdigest() != f["sha256"]:
            raise SystemExit(f"{f['path']} @ {f['commit'][:12]}: sha256 differs from {SOURCES}")
        data[f"{f['commit']}:{f['path']}"] = b
    wd = Watchdog()
    guard_cm = modal.Dict.ephemeral()  # the cost guard: one attempt per call (input build, skip list and each cell); closed in the finally below
    guard = guard_cm.__enter__()
    cells, failed, fatal, pilot, walls, sk = {}, {}, None, None, {}, None
    cdir = HERE / "results" / "partial" / f"{OUT}_{resume or stamp}"
    try:
        call = prep_h.spawn(src["files"], data, {k: src[k] for k in ("routing_ref", "routing_dir", "routing_names")}, guard)
        wd.add("prep", call, PREP_DEADLINE_S)
        u = call.get(timeout=PREP_DEADLINE_S)
        wd.done.add("prep")
        epochs, want, shas, allowed, uprov = u["epochs"], u["want"], u["shas"], u["allowed"], u["provenance"]
        counts = {b_: sum(e["batch"] == b_ for e in epochs) for b_ in sorted({e["batch"] for e in epochs})}
        print(json.dumps({"stamp": stamp, "git": git_at_launch, "stage": stage, "epochs_per_class": counts, "per_class_set_epoch": u["counts"],
                          "mains": len({e["task"] for e in epochs}), "drops": uprov["drops"], "roots": len(allowed), "pool_sizes": uprov["pool_sizes"],
                          "routec_sold_sim": {k: uprov["routec_sold_sim"][k] for k in ("sold_sim", "registered", "pilot")},
                          "routec_prior_match": {k: uprov["routec_prior_match"][k] for k in ("equal", "n_prior", "n_pilot", "n_mains")}}), flush=True)
        if counts != EXPECT:
            raise SystemExit(f"epoch counts {counts} != {EXPECT}")
        if stage == "dry":
            return
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
        cmeta = ui.ckpt_meta(git_at_launch, sha(HERE / "run_honest_all.py"), n_null, epochs, sk["skip"], allowed)
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
        if fatal is not None and stage == "fit":  # a sanitized failure record (no scores), best effort; the original error propagates
            try:
                (HERE / "results" / f"{OUT}_FAILED_{stamp}.json").open("x").write(json.dumps(
                    {"label": "honest all-tasks detector: FAILED before the record", "git": git_at_launch, "stamp": stamp, "error": fatal,
                     "failed": failed, "cells_done": sorted(cells), "watchdog_log": wd.log}, indent=1))
            except Exception as e:  # noqa: BLE001
                print(f"could not write the failure record: {type(e).__name__}", flush=True)
    same = {ro: len({json.dumps(cells[f"{ro}|L{L}"]["rows_per_run"], sort_keys=True) for L in base.LAYERS if f"{ro}|L{L}" in cells}) <= 1 for ro in base.READOUTS}
    gaps = [f"{k}|{p}|observed" for k, c in cells.items() for p, r in c["problems"].items() if not r["observed_complete"]]
    gaps += [f"{k}|{p}|null" for k, c in cells.items() for p, r in c["problems"].items() if not r["null_complete"]]
    gaps += [f"{k}|paired" for k, c in cells.items() if not c.get("paired_simh_minus_ab")]
    rec = {"label": "ROUTE C RERUN (2026-10-18; route C INPUTS_FULL 233212, S_old sim + the 20 pilot runs, routec_keys.py) of the "
                    "ALL-TASKS UNCUT honest-run detector: sim honest vs real honest B, real honest A vs B as the control (a1, 10-16; POST HOC; "
                    "4f's run_attack_all.py @ 59acf625e with the inputs and problems changed); PRELIMINARY until the independent recheck",
           "git": git_at_launch, "stamp": stamp, "app": base.APP, "training_unit": a1.TRAINING_UNIT, "direction": DIRECTION, "volume": ui.VOL,
           "weights_volume": SOLVOL, "inputs": uprov | {"sources_manifest": {"file": SOURCES, "sha256": sha(HERE / SOURCES)}},
           "settings": {"C": C0, "layers": base.LAYERS, "primary": base.PRIMARY, "n_null": n_null, "seed": base.SEED, "problems": PROBLEMS,
                        "fit_timeout_s": FIT_TIMEOUT, "boot_b": BOOT_B, "boot_seed": BOOT_SEED,
                        "null_note": "whole-run labels shuffled within each MAIN task (task = main here), class counts preserved, n_null draws (base.SEED), "
                                     "refit through the whole CV; every bootstrap draws main tasks",
                        "zero_row_rule": "a run with no rows in a cell has no run score: dropped from that cell's AUROC by key (4f + 09, 10-16)"},
           "epochs_per_class": counts, "per_class_set_epoch": u["counts"],
           "skip": {k: sk[k] for k in ("dropped_rows", "staged_sha256", "meta_checked", "meta_tokens") if k in sk},
           "run_folds": {e["name"]: e["fold"] for e in epochs}, "run_keys": {e["name"]: e["key"] for e in epochs},
           "run_set": {e["name"]: e["set"] for e in epochs}, "run_main": {e["name"]: e["task"] for e in epochs},
           "run_cell": {e["name"]: e["cell"] for e in epochs}, "run_epoch": {e["name"]: e["epoch"] for e in epochs},
           "run_source": {e["name"]: e["source"] for e in epochs},
           "rows_per_run_equal_across_layers": same, "zero_row_runs": zero_row_summary(cells, epochs),
           "required_gaps": gaps, "watchdog_log": wd.log, "pilot": pilot, "resumed_from": resume or None, "checkpoints": str(cdir.relative_to(HERE)),
           "cell_wall_seconds": {k: round(v) for k, v in walls.items()}, "cells": cells, "failed": failed}
    rec["uncut_gates"] = ui.check_cells(cells, epochs, sk["expected_rows"])
    dest = HERE / "results" / f"{OUT}_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({"record": dest.name, "pilot": pilot, "required_gaps": gaps, "failed": failed, "rows_equal": same,
                      "zero_row_runs": {k: {x: v[x] for x in ("n", "by_class", "by_set")} for k, v in rec["zero_row_runs"].items()},
                      "uncut_gates_failures": rec["uncut_gates"]["failures"][:5]}), flush=True)
    if pilot and pilot.get("stopped"):
        raise SystemExit(f"PILOT STOP: {pilot['stopped']}")
    if rec["uncut_gates"]["failures"] or len(cells) != 4 or gaps:
        raise SystemExit(f"gates {rec['uncut_gates']['failures'][:5]}, gaps {gaps[:5]} or {len(cells)} of 4 cells")
