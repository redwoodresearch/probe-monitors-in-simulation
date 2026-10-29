"""UNCUT (RULE.md here; 4f detU fork, 10-16, for 0d's post): run_honest_v2.py @ 648f33e61 (aswin/cmd-sim-detector-2e) whose ONLY changes
are (a) --inputs uncut: the same runs by key, read from their whole-run uncut roots (uncut_inputs.py; Volume /u), with the meta check
of every epoch; --inputs 32k keeps v2's construction verbatim (the reproduction gate); (b) the ADDED per-action AUROC: every row
scored by its fold model, AUROC over rows, its null from the same draws and a task bootstrap (rows of the drawn main tasks);
(c) --n-null (0 = observed only, for the reproduction gate), the fit timeout (uncut runs are longer) and the record's name; and
(d) in 32k mode, a gate that run scores and observed metrics equal v2's stored record (honest_detector_2026-10-15_122054.json)
within 1e-6. Its docstring follows.

  PYTHONPATH=src <modal> run experiments/2026-10-16_cmd_sim_detector_uncut/run_honest_uncut.py::main_h --r36 <runs36.json> \\
      --inputs 32k --n-null 0          # gate 2 (reproduction); never --detach (the client's exit must stop the app: RULE.md Cost)
  PYTHONPATH=src <modal> run .../run_honest_uncut.py::main_h --r36 <runs36.json> --inputs uncut --config config/inputs_uncut_v<k>.json

POST HOC (Aswin 10-15 12:07 via 0d; 2e's detector fork): a simulated-honest vs real-honest detector at the content-only end-of-call
(command) readout, beside the averaged reply readout, on Qwen3.5-27B's own T196 runs. Honest only, so attack behaviour cannot drive it.

A thin extension of run_simh_control.py (copied unchanged from aswin/cmd-readout-simh-control @ c663d0204, app renamed): same loader,
row selection (fixed call tokens dropped; content-only call_end rows), fitter (C = 0.01, certified), #36 folds by main task, run score =
mean of row logits, within-task label-shuffle null. Changes, all here:
  - inputs v2: the A25 sim-honest counted set (149 = batch 1 + batch 2), not v1's batch 1 + pilot;
  - three problems per cell: simh_vs_BuC (sim honest vs real honest B and C, v1's design), simh_vs_Bmatched (sim honest vs real
    honest B on the SAME main tasks, both sides restricted to mains that have both), c_vs_b (real C vs real B, the negative control);
  - N_NULL = 50 within-task shuffles (time budget for three problems), FIT timeout 1,200 s;
  - a task bootstrap (2,000 draws of main tasks with replacement, out-of-fold scores fixed, no refit) for a 95% interval on run AUROC.
Stored activations only; no text is read.

  PYTHONPATH=src <modal> run --detach experiments/2026-10-15_cmd_sim_detector/run_honest_v2.py::main_v2 --r36 <runs36.json>
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_simh_control as base  # noqa: E402  (unchanged except the app name)
import uncut_inputs as ui  # noqa: E402

CONFIG_V2 = "config/inputs_v2.json"
N_NULL_V2, FIT_TIMEOUT_V2, MARGIN_V2 = 50, 1800, 300  # (uncut) timeout 1,200 -> 1,800 s and margin 60 -> 300 s: ~2.4x the honest rows, row bootstraps
BOOT_B, BOOT_SEED = 2000, 20261015
PROBLEMS_V2 = ("simh_vs_BuC", "simh_vs_Bmatched", "c_vs_b")
REF_V2 = {"commit": "648f33e6115ebbe19276981dcbd79f9274d9f1ba", "path": "experiments/2026-10-15_cmd_sim_detector/results/honest_detector_2026-10-15_122054.json",
          "sha256": "e909cb93836a252c437d32e36aa14a689db9476e8aa63700074ce102f03f165d"}
TOL = 1e-6
UNCUT_WORLDS = {"honest_b": "B", "honest_c": "C", "sim_honest": "sim"}
START_GPU, FIT_DEADLINE_V2, GATE_DEADLINE_S = 90, FIT_TIMEOUT_V2 + 90 + 60, 600  # (Codex r1/r2) bounded startup; the client cancels any call
# past its deadline (32k gate: observed fits only, about 3 min expected, so 600 s)
RATE_H = 4.59  # $/h: H100 3.95 + 8 CPU x 0.0473 + 32 GiB x 0.008 (Modal list prices), for the pilot's projection
EXPECT_UNCUT = {"B": 195, "C": 192, "sim": 149}
app = base.app
u_vol = modal.Volume.from_name(ui.VOL, version=2).read_only()
_files = [(HERE / "run_simh_control.py", "/root/run_simh_control.py"), (HERE / "uncut_inputs.py", "/root/uncut_inputs.py")]
tok_image = base.tok_image.add_local_file(*_files[0]).add_local_file(*_files[1])
gpu_image = base.gpu_image.add_local_file(*_files[0]).add_local_file(*_files[1])


@app.function(image=tok_image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, volumes={"/b": base.b_vol, "/hf": base.hf_vol})
def skip_b_v2(want: dict, shas: dict) -> dict:
    return base.skip_rows("/b", want, shas)


@app.function(image=tok_image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, volumes={"/d": base.d_vol, "/hf": base.hf_vol})
def skip_d_v2(extract_run: str, staged_run: str, arm: str, staged_sha: str) -> dict:
    return base.skip_d.local(extract_run, staged_run, arm, staged_sha)  # base's body, run in this container


@app.function(image=tok_image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, startup_timeout=60, scaledown_window=2,
              volumes={"/u": u_vol, "/hf": base.hf_vol})
def skip_u_v2(want: dict, shas: dict, full_len: dict, roots: dict) -> dict:
    """(uncut) the same skip rule on the uncut staged inputs, plus every epoch's meta check (uncut_inputs.uncut_skip)."""
    return ui.uncut_skip("/u", want, shas, full_len, roots)


def boot_auroc(score, y, task, B: int, seed: int) -> dict:
    """Task bootstrap of run-level AUROC on fixed out-of-fold scores: draw main tasks with replacement, take every run of each drawn
    task (with multiplicity). A draw with one class missing is counted as failed, never dropped silently."""
    import numpy as np

    groups = {}
    for i, t in enumerate(task):
        groups.setdefault(t, []).append(i)
    keys = sorted(groups)
    idx = {t: np.array(groups[t]) for t in keys}
    rng = np.random.default_rng(seed)
    vals, failed = [], 0
    for _ in range(B):
        sel = np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        yy = y[sel]
        if yy.min() == yy.max():
            failed += 1
            continue
        vals.append(base.auroc(score[sel], yy))
    v = np.array(vals)
    return {"n_draws": B, "failed_draws": failed, "seed": seed, "n_tasks": len(keys), "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
            "note": "main tasks resampled with replacement; out-of-fold scores fixed (no refit)"}


@app.function(image=gpu_image, gpu="H100!", cpu=(8, 8), memory=(32768, 32768), timeout=FIT_TIMEOUT_V2, retries=0, startup_timeout=START_GPU,
              scaledown_window=2, volumes={"/b": base.b_vol, "/d": base.d_vol, "/u": u_vol})
def fit_cell_v2(L: int, readout: str, epochs: list, skip: dict, allowed: dict, n_null: int, guard) -> dict:
    """base.fit_cell with three problems (PROBLEMS_V2), n_null shuffles and a task-bootstrap interval on run AUROC; (uncut) plus the
    per-action AUROC (rows) with its null and task bootstrap, and a cost guard (a rescheduled attempt stops before any work)."""
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

    if not guard.put(f"{readout}|L{L}", time.time(), skip_if_exists=True):
        raise RuntimeError(f"{readout}|L{L}: a rescheduled attempt; refused by the cost guard")
    N_NULL_V2 = n_null  # noqa: N806  (uncut: the module constant becomes an argument; 0 = observed only)
    t0 = time.time()
    X16, row_run, keep, no_rows, chunks = base._load(L, readout, epochs, skip, allowed)
    runs = [epochs[i] for i in keep]  # epochs arrive sorted by name
    t_load = time.time() - t0
    XT = torch.from_numpy(X16).cuda()
    del X16
    main_sim = {r["main"] for r in runs if r["cls"] == "sim"}
    main_b = {r["main"] for r in runs if r["batch"] == "B"}
    both = main_sim & main_b
    problems = {"simh_vs_BuC": ([j for j, r in enumerate(runs) if r["cls"] in ("sim", "real")], lambda r: int(r["cls"] == "sim")),
                "simh_vs_Bmatched": ([j for j, r in enumerate(runs) if (r["cls"] == "sim" or r["batch"] == "B") and r["main"] in both], lambda r: int(r["cls"] == "sim")),
                "c_vs_b": ([j for j, r in enumerate(runs) if r["batch"] in ("B", "C")], lambda r: int(r["batch"] == "C"))}
    prep = {}
    for p, (ids, lab) in problems.items():
        ids = np.array(ids)
        y = np.array([lab(runs[j]) for j in ids], np.int64)
        perms, pinfo = base.within_task_perms([runs[j]["task"] for j in ids], y, N_NULL_V2, base.SEED)
        Y = np.vstack([y[None, :], perms])
        pos = np.full(len(runs), -1)
        pos[ids] = np.arange(len(ids))
        fold = np.array([runs[j]["fold"] for j in ids])
        rows_p = np.flatnonzero(np.isin(row_run, ids))  # (uncut) the problem's rows, for the per-action AUROC
        rpos = np.full(len(row_run), -1)
        rpos[rows_p] = np.arange(len(rows_p))
        prep[p] = {"ids": ids, "Y": Y, "pos": pos, "fold": fold, "pinfo": pinfo, "perm_sha256": hashlib.sha256(perms.tobytes()).hexdigest(),
                   "oof": np.full(Y.shape, np.nan), "rows": rows_p, "rpos": rpos, "oof_row": np.full((Y.shape[0], len(rows_p)), np.nan)}
    certs, null_done, took = [], {p: set() for p in problems}, {"observed": [], "null": []}
    for phase, cols in (("observed", [0]), ("null", list(range(1, N_NULL_V2 + 1)))):
        if not cols:  # (uncut) n_null = 0: observed fits only
            continue
        for p, d in prep.items():
            for f in range(5):
                if phase == "null":
                    est = max(took["null"]) if took["null"] else 3 * max(took["observed"])
                    if time.time() - t0 + 1.25 * est > FIT_TIMEOUT_V2 - MARGIN_V2:
                        certs.append({"phase": phase, "problem": p, "fold": f, "skipped": f"reserve: elapsed {time.time() - t0:.0f} s, estimate {est:.0f} s"})
                        continue
                tr_runs, te_runs = d["ids"][d["fold"] != f], d["ids"][d["fold"] == f]
                tr = np.flatnonzero(np.isin(row_run, tr_runs))
                te = np.flatnonzero(np.isin(row_run, te_runs))
                Xtr, Xte = tp.standardize(XT[torch.from_numpy(tr).cuda()].double(), XT[torch.from_numpy(te).cuda()].double())
                Ytr = torch.from_numpy(d["Y"][cols][:, d["pos"][row_run[tr]]].T.astype(np.float64)).cuda()
                ts = time.time()
                W, b, cert = tp.fit(Xtr, Ytr, base.C, allow_uncertified=True, return_certificate=True)
                S = (Xte @ W + b).cpu().numpy()
                loc = d["pos"][row_run[te]]
                tot = np.zeros((len(d["ids"]), len(cols)))
                np.add.at(tot, loc, S)
                cnt = np.bincount(loc, minlength=len(d["ids"]))
                tp_ = d["pos"][te_runs]
                if (cnt[tp_] == 0).any():
                    raise ValueError("a test run without rows")
                d["oof"][np.ix_(cols, tp_)] = (tot[tp_] / cnt[tp_, None]).T
                d["oof_row"][np.ix_(cols, d["rpos"][te])] = S.T  # (uncut) per-action: each held-out row's own score
                g = np.array(cert["max_grad_per_column"])
                took[phase].append(time.time() - ts)
                certs.append({"phase": phase, "problem": p, "fold": f, "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
                              "n_train_runs": int(len(tr_runs)), "n_test_runs": int(len(te_runs)), "n_evals": int(cert["n_evals"]),
                              "max_grad": float(g.max()), "uncertified_columns": [int(cols[i]) for i in np.flatnonzero(~(g <= tp.CERT_BAR))],
                              "seconds": round(took[phase][-1], 2)})
                if phase == "null":
                    null_done[p].add(f)
                del Xtr, Xte, Ytr, W, b
                torch.cuda.empty_cache()
    out = {"layer": L, "readout": readout, "gpu": torch.cuda.get_device_name(0), "load_seconds": round(t_load), "seconds": round(time.time() - t0),
           "rows": int(len(row_run)), "chunks_read": chunks, "runs_without_rows": no_rows, "matched_mains": len(both),
           "rows_per_run": {r["name"]: int((row_run == j).sum()) for j, r in enumerate(runs)}, "certificates": certs, "problems": {}}
    for p, d in prep.items():
        y, fold, oof = d["Y"][0], d["fold"], d["oof"]
        if not np.isfinite(oof[0]).all():
            raise ValueError(f"{p}: observed out-of-fold scores incomplete or non-finite")
        unc = sorted({(c["fold"], u) for c in certs if c.get("problem") == p for u in c.get("uncertified_columns", [])})
        obs = {"auroc": base.auroc(oof[0], y)}
        for a in (0.01, 0.05):
            obs[f"tpr_{a:g}"], _ = base.tpr_at(oof[0], y, fold, a)
        tasks = [runs[j]["main"] for j in d["ids"]]
        obs["auroc_boot"] = boot_auroc(oof[0], y, tasks, BOOT_B, BOOT_SEED)
        rrun = d["pos"][row_run[d["rows"]]]  # (uncut) per-action AUROC: rows carry their run's label and main task
        if not np.isfinite(d["oof_row"][0]).all():
            raise ValueError(f"{p}: observed per-row scores incomplete or non-finite")
        obs["row_auroc"] = base.auroc(d["oof_row"][0], y[rrun])
        obs["row_auroc_boot"] = boot_auroc(d["oof_row"][0], y[rrun], [tasks[i] for i in rrun], BOOT_B, BOOT_SEED)
        obs["row_note"] = "per action: every command (or reply) row scored by its fold model, AUROC over rows; bootstrap draws main tasks, all their rows"
        res = {"n_runs": {"class1": int(y.sum()), "class0": int((y == 0).sum())},
               "n_rows": {"class1": int(np.isin(row_run, d["ids"][y == 1]).sum()), "class0": int(np.isin(row_run, d["ids"][y == 0]).sum())},
               "n_mains": len(set(tasks)), "shuffle": d["pinfo"], "perm_sha256": d["perm_sha256"], "observed": obs,
               "observed_certified": not any(c == 0 for _, c in unc), "uncertified_fold_columns": [list(x) for x in unc],
               "run_scores": {runs[j]["name"]: float(oof[0][i]) for i, j in enumerate(d["ids"])},
               "n_rows_problem": int(len(d["rows"])),
               "null_complete": N_NULL_V2 > 0 and null_done[p] == set(range(5)) and bool(np.isfinite(oof[1:]).all()) and bool(np.isfinite(d["oof_row"][1:]).all())}
        if res["null_complete"]:
            na = np.array([base.auroc(oof[k], d["Y"][k]) for k in range(1, N_NULL_V2 + 1)])
            bad = sorted({c for _, c in unc if c > 0})
            nr = np.array([base.auroc(d["oof_row"][k], d["Y"][k][rrun]) for k in range(1, N_NULL_V2 + 1)])  # (uncut) per-action null, same draws
            res["null_row"] = {"auroc": nr.tolist(), "n_draws": int(len(nr)), "p95": float(np.quantile(nr, 0.95)), "max": float(nr.max()),
                               "mean": float(nr.mean()), "p_add_one": float((1 + (nr >= obs["row_auroc"]).sum()) / (1 + len(nr)))}
            res["null"] = {"auroc": na.tolist(), "n_draws": int(len(na)), "p95": float(np.quantile(na, 0.95)), "max": float(na.max()),
                           "mean": float(na.mean()), "p_add_one": float((1 + (na >= obs["auroc"]).sum()) / (1 + len(na))),
                           "uncertified_draws": bad, "certified": not bad}
        out["problems"][p] = res
    out["seconds_total"] = round(time.time() - t0)  # (Codex r1) through the bootstraps
    return out


def build_32k(cfg: dict, main_of: dict, fold_of_main: dict):
    """v2's construction (main_v2 @ 648f33e61), verbatim apart from returning its pieces: real honest B (corrected-B manifest) + C,
    simulated honest from d9's dumps (provenance, counted keys), skip lists remote. Returns (epochs, skip, allowed, extra record)."""
    rc = cfg["real"]
    mf = json.loads(base._git_bytes(rc["manifest"]))
    up, st, ex = (json.loads(base._git_bytes(rc[k])) for k in ("c_upload", "c_stage", "c_extract"))
    real = [{k: e[k] for k in ("key", "world", "task", "arm", "extraction_run", "staged_inputs_run", "meta_file")} | {"batch": "B"}
            for e in mf["epochs"] if e["world"] in rc["worlds_from_manifest"]]
    shas = {e["staged_inputs_file"]: e["staged_inputs_sha256"] for e in mf["epochs"] if e["world"] in rc["worlds_from_manifest"]}
    excl = set(st["c"]["c_gate"]["excluded"])
    ce = [{"key": f"{f_['path']}#e0", "world": "honest_c", "task": f_["task"], "arm": "honest_b", "extraction_run": ex["config"]["run"],
           "staged_inputs_run": ex["config"]["inputs"], "batch": "C"} for f_ in up["files"] if f"{f_['path']}#e0" not in excl]
    if len(ce) != st["c"]["c_gate"]["staged"]:
        raise SystemExit("batch C epochs differ from its staging gate's count")
    real += ce
    allowed = {ex["config"]["run"]: sorted(c["chunk"] for s_ in ex["shards"] for c in s_["chunks"])}
    shas |= {f"{ex['config']['inputs']}/inputs/{a}": s for a, s in ex["config"]["inputs_sha256"].items()}
    want = {}
    for e in real:
        want.setdefault(e["staged_inputs_run"], {}).setdefault(e["arm"], []).append(e["key"])
    sims, dump_rec = [], {}
    for dmp in cfg["sim_dumps"]:  # base.main's provenance (B200 rule) and counted-keys checks, verbatim
        man = json.loads(base._git_bytes(dmp["record"]))
        pv = dmp["prov"]
        if pv["gpu_pin"]:
            raise SystemExit("v2 dumps carry per-shard device fields; pin mode is not used here")
        prov = base.provenance(man, dmp["extract_run"], dmp["staged_run"], dmp["arm"], pv["shards_key"], pv["gpu_key"], pv["longest_key"], pv["run_key"],
                               pv["inputs_sha_key"])
        ck = json.loads(base._git_bytes(dmp["counted_keys"]))
        counted = {k["task"] for k in ck["keys"] if k.get("world") == dmp["world"] and k.get("usable", True) is True}
        bk = json.loads(base._git_bytes(dmp["batch_keys"])) if dmp["batch_keys"] else None
        if any(s_["done_before"] for s_ in man["shards"]) or sum(c["epochs"] for s_ in man["shards"] for c in s_["chunks"]) != sum(s_["epochs"] for s_ in man["shards"]):
            raise SystemExit(f"{dmp['name']}: the extraction record does not list every chunk")
        allowed[dmp["extract_run"]] = sorted(c["chunk"] for s_ in man["shards"] for c in s_["chunks"])
        dump_rec[dmp["name"]] = {"provenance": prov, "record": dmp["record"], "counted_keys": dmp["counted_keys"], "batch_keys": dmp["batch_keys"],
                                 "n_counted_usable": len(counted), "_counted": counted,
                                 "_batch": ({k.split("/", 1)[1] for k in bk["usable_keys"]} if bk else None)}
    calls = {"B+C": skip_b_v2.spawn(want, shas)} | {d["name"]: skip_d_v2.spawn(d["extract_run"], d["staged_run"], d["arm"],
                                                                                   dump_rec[d["name"]]["provenance"]["staged_inputs_sha256"]) for d in cfg["sim_dumps"]}
    sk = {k: c.get() for k, c in calls.items()}
    if any(s_["fixed_tokens"] != base.FIXED_TOKENS for s_ in sk.values()):
        raise SystemExit("fixed call tokens differ from Amendment 12's")
    all_counted = None
    for dmp in cfg["sim_dumps"]:
        r_ = dump_rec[dmp["name"]]
        staged = {t for _, t in sk[dmp["name"]]["pairs"]}
        counted = r_.pop("_counted")
        all_counted = counted if all_counted is None else all_counted
        if counted != all_counted:
            raise SystemExit("the dumps name different counted-key sets")
        bt = r_.pop("_batch")
        if not staged <= counted or (bt is not None and staged != bt):
            raise SystemExit(f"{dmp['name']}: staged tasks are not counted usable keys (or differ from the batch's usable keys)")
        r_ |= {"staged_sha256": sk[dmp["name"]]["staged_sha256"], "skip_dropped_rows": sk[dmp["name"]]["dropped_rows"], "n_staged": len(staged)}
        sims += [{"key": k, "world": dmp["world"], "task": t, "arm": dmp["arm"], "extraction_run": dmp["extract_run"], "batch": dmp["batch"], "vol": "d"}
                 for k, t in sk[dmp["name"]]["pairs"]]
    sim_tasks = {e["task"] for e in sims}
    if sim_tasks != all_counted or len(sims) != len(all_counted):  # the A25 set exactly: 149 runs, one per counted task
        raise SystemExit(f"sim-honest epochs ({len(sims)}, {len(sim_tasks)} tasks) are not exactly the {len(all_counted)} counted keys")
    epochs = [e | {"cls": "real", "vol": "b"} for e in real] + [e | {"cls": "sim"} for e in sims]
    skip = sk["B+C"]["skip"] | {k: v for n, s_ in sk.items() if n != "B+C" for k, v in s_["skip"].items()}
    extra = {"volumes": {"real": base.BVOL, "sim": base.DVOL}, "real_skip": {"dropped_rows": sk["B+C"]["dropped_rows"], "staged_sha256": sk["B+C"]["staged_sha256"]},
             "sim_dumps": dump_rec, "skip_dropped_rows": {k: v["dropped_rows"] for k, v in sk.items()}}
    return epochs, skip, allowed, extra


def same_runs_as_32k(cfg_v2: dict, epochs: list) -> dict:
    """(uncut) the run sets equal the 32k detector's, rebuilt from its input records (never its result files): B = the corrected-B
    manifest's honest_b keys; C = batch C's upload minus its staging gate's exclusions; simulated honest = the A25 counted usable
    keys, matched by eval sha256 (and one run per counted task)."""
    rc = cfg_v2["real"]
    mf = json.loads(base._git_bytes(rc["manifest"]))
    up, st = (json.loads(base._git_bytes(rc[k])) for k in ("c_upload", "c_stage"))
    excl = set(st["c"]["c_gate"]["excluded"])
    ref = {"B": {e["key"] for e in mf["epochs"] if e["world"] in rc["worlds_from_manifest"]},
           "C": {f"{f_['path']}#e0" for f_ in up["files"] if f"{f_['path']}#e0" not in excl}}
    got = {b_: {e["key"] for e in epochs if e["batch"] == b_} for b_ in ("B", "C")}
    out = {}
    for b_ in ("B", "C"):
        if got[b_] != ref[b_]:
            raise SystemExit(f"uncut {b_} keys differ from the 32k detector's ({len(got[b_] - ref[b_])} extra, {len(ref[b_] - got[b_])} missing)")
        out[b_] = len(got[b_])
    ck = json.loads(base._git_bytes(cfg_v2["sim_dumps"][0]["counted_keys"]))
    usable = [k for k in ck["keys"] if k.get("world") == cfg_v2["sim_dumps"][0]["world"] and k.get("usable", True) is True]
    sims = [e for e in epochs if e["batch"] == "sim"]
    if {k["eval_sha256"] for k in usable} != {e["eval_sha256"] for e in sims} or {k["task"] for k in usable} != {e["task"] for e in sims} or len(sims) != len(usable):
        raise SystemExit("uncut simulated-honest runs are not exactly the A25 counted keys (eval sha256, tasks)")
    out["sim"] = len(sims)
    return out


def reproduction_gate(cells: dict) -> dict:
    """(32k mode) every cell's run scores and observed metrics equal v2's stored record within TOL."""
    b = subprocess.run(["git", "show", f"{REF_V2['commit']}:{REF_V2['path']}"], capture_output=True, check=True, cwd=HERE).stdout
    if hashlib.sha256(b).hexdigest() != REF_V2["sha256"]:
        raise SystemExit("v2's record sha256 differs from the pin")
    ref = json.loads(b)
    rep, fail = {}, [f"missing cell {k}" for k in sorted(set(ref["cells"]) - set(cells))]
    for k in sorted(set(ref["cells"]) & set(cells)):
        for p, ra in ref["cells"][k]["problems"].items():
            try:
                r = cells[k]["problems"][p]
                mine, theirs = r["run_scores"], ra["run_scores"]
                d = max(abs(mine[n] - theirs[n]) for n in theirs) if set(mine) == set(theirs) else float("inf")
                o, oa = r["observed"], ra["observed"]
                dm = max(abs(o["auroc"] - oa["auroc"]), abs(o["tpr_0.01"] - oa["tpr_0.01"]), abs(o["tpr_0.05"] - oa["tpr_0.05"]),
                         *(abs(x - y) for x, y in zip(o["auroc_boot"]["ci95"], oa["auroc_boot"]["ci95"])))
                rep[f"{k}|{p}"] = {"runs": len(theirs), "same_run_names": set(mine) == set(theirs), "max_abs_run_score_diff": d, "max_abs_metric_diff": dm}
                ok = d <= TOL and dm <= TOL
            except (KeyError, TypeError, ValueError) as e:
                rep[f"{k}|{p}"], ok = {"error": f"{type(e).__name__}: {str(e)[:200]}"}, False
            if not ok:
                fail.append(f"{k}|{p}")
    return {"reference": REF_V2, "tolerance": TOL, "cells": rep, "failed": fail, "pass": not fail}


@app.local_entrypoint()
def main_h(r36: str, inputs: str = "uncut", config: str = "", n_null: int = N_NULL_V2, stage: str = "fit", budget: float = 18.0, resume: str = ""):
    """inputs: 32k (v2's construction; gate 2) or uncut. stage: dry (local checks only, no Modal call), skip (+ the CPU skip list and
    meta check, then stop) or fit. budget ($): the uncut pilot (the first cell, alone) must project the four cells within it."""
    if inputs not in ("32k", "uncut") or stage not in ("dry", "skip", "fit") or n_null < 0 or (inputs == "uncut" and n_null < 2):
        raise SystemExit("--inputs 32k|uncut, --stage dry|skip|fit, --n-null >= 0 (uncut: >= 2, the null is required)")
    if inputs == "32k" and n_null != 0:
        raise SystemExit("32k mode is the reproduction gate: observed fits only (--n-null 0)")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    if stage == "fit" and (git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--", str(HERE), f":(exclude){HERE}/results/partial")):
        raise SystemExit("uncommitted tracked changes, or uncommitted files in this experiment folder")
    git_at_launch = git("rev-parse", "HEAD")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    if sha(r36) != base.R36_SHA:
        raise SystemExit("runs36.json sha256 mismatch")
    config = config or (CONFIG_V2 if inputs == "32k" else "config/inputs_uncut_v2.json")
    cfg = json.loads((HERE / config).read_text())
    cfg_v2 = json.loads((HERE / CONFIG_V2).read_text())
    fold_of_main = {}
    for r in json.loads(Path(r36).read_text()):
        if fold_of_main.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("a main task in two folds")
    main_of = {c["key"]: c["main"] for c in json.loads((base.REPO / base.T196).read_text())["cells"]}
    if inputs == "uncut":
        u = ui.build(base.REPO, cfg, UNCUT_WORLDS, dry_run=stage != "fit")
        inv = {r["key"]: r for r in ui.git_json(base.REPO, cfg["inventory"])["runs"]}
        epochs = [e | {"cls": "sim" if e["batch"] == "sim" else "real", "eval_sha256": inv[e["key"]]["eval_sha256"]} for e in u["epochs"]]
        same = same_runs_as_32k(cfg_v2, epochs)
        counts = {b_: sum(e["batch"] == b_ for e in epochs) for b_ in sorted({e["batch"] for e in epochs})}
        if counts != EXPECT_UNCUT:
            raise SystemExit(f"uncut epoch counts {counts} != {EXPECT_UNCUT}")
        print(json.dumps({"stamp": stamp, "git": git_at_launch, "stage": stage, "epochs_per_batch": counts, "same_runs_as_32k": same,
                          "routing": u["provenance"]["routing"]["mode"], "roots": {k: v["epochs"] for k, v in u["provenance"]["roots"].items()}}), flush=True)
        if stage == "dry":
            return
        roots = {e["staged_inputs_run"]: e["extraction_run"] for e in epochs}
        sk = skip_u_v2.remote(u["want"], u["shas"], {e["key"]: e["full_len"] for e in epochs}, roots)
        if sk["fixed_tokens"] != base.FIXED_TOKENS or sk["meta_checked"] != len(epochs):
            raise SystemExit("fixed call tokens differ from Amendment 12's, or not every epoch's meta was checked")
        skip, allowed = sk["skip"], u["allowed"]
        extra = {"volumes": {"uncut": ui.VOL}, "uncut_inputs": u["provenance"], "same_runs_as_32k": same,
                 "skip": {"dropped_rows": sk["dropped_rows"], "staged_sha256": sk["staged_sha256"], "meta_checked": sk["meta_checked"], "meta_tokens": sk["meta_tokens"]}}
        print(json.dumps({"skip_dropped_rows": sk["dropped_rows"], "meta_checked": sk["meta_checked"], "meta_tokens": sk["meta_tokens"]}), flush=True)
        if stage == "skip":
            return
    else:
        if stage != "fit":
            raise SystemExit("32k mode runs only as the gate (stage fit)")
        epochs, skip, allowed, extra = build_32k(cfg, main_of, fold_of_main)
    for e in epochs:
        e["name"] = f"{e['world']}|{e['task']}"
        e["main"] = main_of[e["task"]]
        e["fold"] = fold_of_main[e["main"]]
    if len({e["key"] for e in epochs}) != len(epochs) or len({e["name"] for e in epochs}) != len(epochs):
        raise SystemExit("duplicate epoch key or run name")
    epochs.sort(key=lambda e: e["name"])
    counts = {b_: sum(e["batch"] == b_ for e in epochs) for b_ in sorted({e["batch"] for e in epochs})}
    print(json.dumps({"stamp": stamp, "git": git_at_launch, "inputs": inputs, "n_null": n_null, "epochs_per_batch": counts}), flush=True)
    order = [(63, "command")] + [(L, ro) for ro in base.READOUTS for L in base.LAYERS if (L, ro) != (63, "command")]  # pilot = primary cell
    cells, failed, pilot = {}, {}, None
    cmeta = ui.ckpt_meta(git_at_launch, sha(HERE / config), n_null, epochs, skip, allowed)  # (4f) per-cell checkpoints; --resume <stamp>
    cdir = HERE / "results" / "partial" / f"honest_{inputs}_{resume or stamp}"  # (Codex r5) a resume appends to the resumed set
    walls = {}
    if resume:
        for cell, (o, w) in ui.ckpt_load(HERE / "results" / "partial" / f"honest_{inputs}_{resume}", cmeta).items():
            cells[cell], walls[cell] = o, w
        print(json.dumps({"resumed_from": resume, "cells": sorted(cells)}), flush=True)
    deadline = GATE_DEADLINE_S if inputs == "32k" else FIT_DEADLINE_V2
    ui.hard_exit_after(deadline * (1 if inputs == "32k" else 2) + 300)  # (Codex r4) pilot window + the rest's window + 300 s for records
    wd = ui.Watchdog()  # (Codex r2) independent, bounded cancellation of any call past its deadline
    with modal.Dict.ephemeral() as guard:  # (uncut) the cost guard: one attempt per cell
        jobs, t_sp = {}, {}

        def launch(keys):
            for k in keys:
                jobs[k], t_sp[k] = fit_cell_v2.spawn(k[0], k[1], epochs, skip, allowed, n_null, guard), time.time()
                wd.add(f"{k[1]}|L{k[0]}", jobs[k], deadline)

        def collect(keys):
            for k in keys:
                L, ro = k
                try:
                    cells[f"{ro}|L{L}"] = jobs[k].get(timeout=max(1.0, deadline - (time.time() - t_sp[k])))
                    wd.done.add(f"{ro}|L{L}")
                    walls[f"{ro}|L{L}"] = time.time() - t_sp[k]
                    ui.ckpt_save(cdir, f"{ro}|L{L}", cells[f"{ro}|L{L}"], cmeta, walls[f"{ro}|L{L}"])
                except Exception as e:  # noqa: BLE001 - recorded (a timeout too), then the run exits nonzero after the record is written
                    failed[f"{ro}|L{L}"] = f"{type(e).__name__}: {str(e)[:500]} (at {time.time() - t_sp[k]:.0f} s from spawn; deadline {deadline} s)"
        try:
            todo = [k for k in order if f"{k[1]}|L{k[0]}" not in cells]  # (4f) a resume runs only the missing cells
            first = todo if inputs == "32k" else ([order[0]] if order[0] in todo else [])  # (Codex r1) uncut: the pilot cell alone first
            launch(first)
            collect(first)
            if inputs == "uncut":
                L0, ro0 = order[0]
                wall = walls.get(f"{ro0}|L{L0}", float("inf"))
                proj = wall * len(order) * RATE_H / 3600
                pilot = {"cell": f"{ro0}|L{L0}", "wall_seconds": round(wall) if wall != float("inf") else None, "container_seconds_total": cells.get(f"{ro0}|L{L0}", {}).get("seconds_total"),
                         "projected_usd_all_cells": round(proj, 2), "budget_usd": budget, "rate_usd_per_h": RATE_H}
                print(json.dumps({"pilot": pilot}), flush=True)
                if failed or proj > budget:
                    pilot["stopped"] = "the pilot failed" if failed else f"projected ${proj:.2f} > budget ${budget:.2f}"
                else:
                    rest = [k for k in todo if k != order[0]]
                    launch(rest)
                    collect(rest)
        finally:
            try:  # nothing outstanding past here; each cancel runs in its own thread, joined at most 30 s in total
                wd.cancel_outstanding("not finished", join_s=30.0)
            except Exception as e:  # noqa: BLE001 - logged in the record
                wd.log.append(f"cancel_outstanding failed: {type(e).__name__}: {str(e)[:200]}")
            wd.stop()
    same = {ro: len({json.dumps(cells[f"{ro}|L{L}"]["rows_per_run"], sort_keys=True) for L in base.LAYERS if f"{ro}|L{L}" in cells}) <= 1 for ro in base.READOUTS}
    rec = {"label": f"UNCUT detector honest-only, inputs {inputs} (4f detU fork, 10-16; run_honest_v2 @ 648f33e61 + per-action AUROC); PRELIMINARY until Codex and 0d review",
           "git": git_at_launch, "stamp": stamp, "app": base.APP, "inputs": inputs, "config": {"path": config, "sha256": sha(HERE / config)},
           "r36": {"path": r36, "sha256": base.R36_SHA}, "t196": {"path": base.T196, "sha256": sha(base.REPO / base.T196)},
           "settings": {"C": base.C, "layers": base.LAYERS, "readouts": base.READOUTS, "n_null": n_null, "seed": base.SEED, "boot_b": BOOT_B,
                        "boot_seed": BOOT_SEED, "problems": PROBLEMS_V2, "fixed_tokens": base.FIXED_TOKENS, "fit_timeout_s": FIT_TIMEOUT_V2,
                        "readout_rule": "command = content-only call_end rows minus fixed call tokens; reply = output_end row_mean; no end-of-sequence or template-token readout",
                        "fitter": "probe_shift.torch_probe.fit defaults, certificate bar 1e-6"},
           "epochs_per_batch": counts, "run_folds": {e["name"]: e["fold"] for e in epochs}, "run_keys": {e["name"]: e["key"] for e in epochs},
           "rows_per_run_equal_across_layers": same, "pilot": pilot, "resumed_from": resume or None, "checkpoints": str(cdir.relative_to(HERE)),
           "cell_wall_seconds": {k: round(v) for k, v in walls.items()}, "watchdog_log": wd.log, "deadline_s": deadline, "cells": cells, "failed": failed} | extra
    if inputs == "uncut":  # (Codex r1) rows equal the staged expectation; observed fits and null draws certified
        rec["uncut_gates"] = ui.check_cells(cells, epochs, sk["expected_rows"])
    if inputs == "32k":
        rec["reproduction_of_v2"] = reproduction_gate(cells)
    dest = HERE / "results" / f"honest_{inputs}_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(rec, indent=1))
    for k, c in cells.items():
        for p, r in c["problems"].items():
            o, n, nr = r["observed"], r.get("null", {}), r.get("null_row", {})
            print(json.dumps({"cell": k, "problem": p, "runs": r["n_runs"], "rows": r["n_rows"], "mains": r["n_mains"], "auroc": round(o["auroc"], 4),
                              "ci95": o["auroc_boot"]["ci95"], "boot_failed": o["auroc_boot"]["failed_draws"], "null_p95": n.get("p95"), "null_p": n.get("p_add_one"),
                              "row_auroc": round(o["row_auroc"], 4), "row_ci95": o["row_auroc_boot"]["ci95"], "row_null_p95": nr.get("p95"), "row_null_p": nr.get("p_add_one"),
                              "tpr_1pct": o["tpr_0.01"], "tpr_5pct": o["tpr_0.05"], "observed_certified": r["observed_certified"],
                              "null_complete": r["null_complete"], "seconds": c["seconds"], "load_seconds": c["load_seconds"]}), flush=True)
    print(dest.name)
    if inputs == "32k":
        print(json.dumps({"reproduction_of_v2": {k: rec["reproduction_of_v2"][k] for k in ("failed", "pass")}}), flush=True)
        if not rec["reproduction_of_v2"]["pass"] or failed or not all(same.values()):
            raise SystemExit(f"GATE 2 (honest): reproduction failed {rec['reproduction_of_v2']['failed']}, failed cells {sorted(failed)}, or rows differ {same}")
        return
    print(json.dumps({"pilot": pilot, "uncut_gates": rec["uncut_gates"]}), flush=True)
    incomplete = sorted(f"{k}|{p}" for k, c in cells.items() for p, r in c["problems"].items() if not r["null_complete"])
    if pilot and pilot.get("stopped"):
        raise SystemExit(f"PILOT STOP: {pilot['stopped']}")
    if failed or not all(same.values()) or incomplete or rec["uncut_gates"]["failures"] or len(cells) != len(order):
        raise SystemExit(f"failed cells {sorted(failed)}, rows differ across layers {same}, incomplete nulls {incomplete}, "
                         f"or uncut gates {rec['uncut_gates']['failures'][:5]}")
