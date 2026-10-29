"""POST HOC (Aswin 10-15 12:07 via 0d; 2e's detector fork): a simulated-honest vs real-honest detector at the content-only end-of-call
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_simh_control as base  # noqa: E402  (unchanged except the app name)

CONFIG_V2 = "config/inputs_v2.json"
N_NULL_V2, FIT_TIMEOUT_V2, MARGIN_V2 = 50, 1200, 60
BOOT_B, BOOT_SEED = 2000, 20261015
PROBLEMS_V2 = ("simh_vs_BuC", "simh_vs_Bmatched", "c_vs_b")
app = base.app
_base_file = (HERE / "run_simh_control.py", "/root/run_simh_control.py")
tok_image = base.tok_image.add_local_file(*_base_file)
gpu_image = base.gpu_image.add_local_file(*_base_file)


@app.function(image=tok_image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, volumes={"/b": base.b_vol, "/hf": base.hf_vol})
def skip_b_v2(want: dict, shas: dict) -> dict:
    return base.skip_rows("/b", want, shas)


@app.function(image=tok_image, cpu=(4, 4), memory=(16384, 16384), timeout=900, retries=0, volumes={"/d": base.d_vol, "/hf": base.hf_vol})
def skip_d_v2(extract_run: str, staged_run: str, arm: str, staged_sha: str) -> dict:
    return base.skip_d.local(extract_run, staged_run, arm, staged_sha)  # base's body, run in this container


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


@app.function(image=gpu_image, gpu="H100!", cpu=(8, 8), memory=(32768, 32768), timeout=FIT_TIMEOUT_V2, retries=0, volumes={"/b": base.b_vol, "/d": base.d_vol})
def fit_cell_v2(L: int, readout: str, epochs: list, skip: dict, allowed: dict) -> dict:
    """base.fit_cell with three problems (PROBLEMS_V2), N_NULL_V2 shuffles and a task-bootstrap interval on run AUROC; everything else as base."""
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

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
        prep[p] = {"ids": ids, "Y": Y, "pos": pos, "fold": fold, "pinfo": pinfo, "perm_sha256": hashlib.sha256(perms.tobytes()).hexdigest(),
                   "oof": np.full(Y.shape, np.nan)}
    certs, null_done, took = [], {p: set() for p in problems}, {"observed": [], "null": []}
    for phase, cols in (("observed", [0]), ("null", list(range(1, N_NULL_V2 + 1)))):
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
        res = {"n_runs": {"class1": int(y.sum()), "class0": int((y == 0).sum())},
               "n_rows": {"class1": int(np.isin(row_run, d["ids"][y == 1]).sum()), "class0": int(np.isin(row_run, d["ids"][y == 0]).sum())},
               "n_mains": len(set(tasks)), "shuffle": d["pinfo"], "perm_sha256": d["perm_sha256"], "observed": obs,
               "observed_certified": not any(c == 0 for _, c in unc), "uncertified_fold_columns": [list(x) for x in unc],
               "run_scores": {runs[j]["name"]: float(oof[0][i]) for i, j in enumerate(d["ids"])},
               "null_complete": null_done[p] == set(range(5)) and bool(np.isfinite(oof[1:]).all())}
        if res["null_complete"]:
            na = np.array([base.auroc(oof[k], d["Y"][k]) for k in range(1, N_NULL_V2 + 1)])
            bad = sorted({c for _, c in unc if c > 0})
            res["null"] = {"auroc": na.tolist(), "n_draws": int(len(na)), "p95": float(np.quantile(na, 0.95)), "max": float(na.max()),
                           "mean": float(na.mean()), "p_add_one": float((1 + (na >= obs["auroc"]).sum()) / (1 + len(na))),
                           "uncertified_draws": bad, "certified": not bad}
        out["problems"][p] = res
    return out


@app.local_entrypoint()
def main_v2(r36: str, config: str = CONFIG_V2):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--", str(HERE)):
        raise SystemExit("uncommitted tracked changes, or uncommitted files in this experiment folder")
    git_at_launch = git("rev-parse", "HEAD")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    if sha(r36) != base.R36_SHA:
        raise SystemExit("runs36.json sha256 mismatch")
    cfg = json.loads((HERE / config).read_text())
    fold_of_main = {}
    for r in json.loads(Path(r36).read_text()):
        if fold_of_main.setdefault(r["main_task"], r["fold"]) != r["fold"]:
            raise SystemExit("a main task in two folds")
    main_of = {c["key"]: c["main"] for c in json.loads((base.REPO / base.T196).read_text())["cells"]}
    # real honest B (corrected-B manifest) + C: base.main's construction, verbatim
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
    for e in epochs:
        e["name"] = f"{e['world']}|{e['task']}"
        e["main"] = main_of[e["task"]]
        e["fold"] = fold_of_main[e["main"]]
    if len({e["key"] for e in epochs}) != len(epochs) or len({e["name"] for e in epochs}) != len(epochs):
        raise SystemExit("duplicate epoch key or run name")
    epochs.sort(key=lambda e: e["name"])
    skip = sk["B+C"]["skip"] | {k: v for n, s_ in sk.items() if n != "B+C" for k, v in s_["skip"].items()}
    counts = {b_: sum(e["batch"] == b_ for e in epochs) for b_ in sorted({e["batch"] for e in epochs})}
    print(json.dumps({"stamp": stamp, "git": git_at_launch, "epochs_per_batch": counts, "skip_dropped_rows": {k: v["dropped_rows"] for k, v in sk.items()}}), flush=True)
    jobs = {(L, ro): fit_cell_v2.spawn(L, ro, epochs, skip, allowed) for ro in base.READOUTS for L in base.LAYERS}
    cells, failed = {}, {}
    for (L, ro), c in jobs.items():
        try:
            cells[f"{ro}|L{L}"] = c.get()
        except Exception as e:  # noqa: BLE001 - recorded, then the run exits nonzero after the record is written
            failed[f"{ro}|L{L}"] = f"{type(e).__name__}: {str(e)[:500]}"
    same = {ro: len({json.dumps(cells[f"{ro}|L{L}"]["rows_per_run"], sort_keys=True) for L in base.LAYERS if f"{ro}|L{L}" in cells}) <= 1 for ro in base.READOUTS}
    rec = {"label": "POST HOC cmd_sim_detector honest-only (2e detector fork, Aswin 10-15 12:07); PRELIMINARY until 09 and Codex review", "git": git_at_launch,
           "stamp": stamp, "app": base.APP, "volumes": {"real": base.BVOL, "sim": base.DVOL}, "config": {"path": config, "sha256": sha(HERE / config)},
           "r36": {"path": r36, "sha256": base.R36_SHA}, "t196": {"path": base.T196, "sha256": sha(base.REPO / base.T196)},
           "settings": {"C": base.C, "layers": base.LAYERS, "readouts": base.READOUTS, "n_null": N_NULL_V2, "seed": base.SEED, "boot_b": BOOT_B,
                        "boot_seed": BOOT_SEED, "problems": PROBLEMS_V2, "fixed_tokens": base.FIXED_TOKENS,
                        "readout_rule": "command = content-only call_end rows minus fixed call tokens; reply = output_end row_mean; no end-of-sequence or template-token readout",
                        "fitter": "probe_shift.torch_probe.fit defaults, certificate bar 1e-6"},
           "epochs_per_batch": counts, "real_skip": {"dropped_rows": sk["B+C"]["dropped_rows"], "staged_sha256": sk["B+C"]["staged_sha256"]}, "sim_dumps": dump_rec,
           "run_folds": {e["name"]: e["fold"] for e in epochs}, "rows_per_run_equal_across_layers": same, "cells": cells, "failed": failed}
    dest = HERE / "results" / f"honest_detector_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(rec, indent=1))
    for k, c in cells.items():
        for p, r in c["problems"].items():
            o, n = r["observed"], r.get("null", {})
            print(json.dumps({"cell": k, "problem": p, "runs": r["n_runs"], "mains": r["n_mains"], "auroc": round(o["auroc"], 4), "ci95": o["auroc_boot"]["ci95"],
                              "boot_failed": o["auroc_boot"]["failed_draws"], "null_p95": n.get("p95"), "null_p": n.get("p_add_one"),
                              "tpr_1pct": o["tpr_0.01"], "tpr_5pct": o["tpr_0.05"], "observed_certified": r["observed_certified"],
                              "null_complete": r["null_complete"], "seconds": c["seconds"]}), flush=True)
    print(dest.name)
    incomplete = sorted(f"{k}|{p}" for k, c in cells.items() for p, r in c["problems"].items() if not r["null_complete"])
    if failed or not all(same.values()) or incomplete:
        raise SystemExit(f"failed cells {sorted(failed)}, rows differ across layers {same}, or incomplete nulls {incomplete}")
