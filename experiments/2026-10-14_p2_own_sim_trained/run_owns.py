"""RULE.md (Append 29, sim-trained probes own_s / own_x). Rows come from 2e's pinned row code (run_avg_transfer.py @ 4d1eec572) on
pass 1's spec plus the counted sim-honest epochs; fits are the certified torch_probe @ be9dbba7c, as pass 1's _fit_one.
  g2pilot: the TIMING pilot for G2's permuted-label refits; scores computed and DISCARDED (never written, printed or logged).
  prep:    pass 1's spec + the sim-honest epochs of dumps.json (pinned, checked in-job); frozen training-key lists.
  fit:     6 H100 containers (readout x layer), 5 folds each: own_frac100 refit control, own_s (and own_x), held-out scores split
           open / sim_sealed; then the open pack.
  pack:    --part sim_sealed only after 09's phase-1 record is on origin (checked in-job, before any sim file is opened).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach <this file> --job g2pilot --device h100|cpu
  ... --job prep | --job fit --spec-owns <prep stamp> --spec-sha <sha256> --variants own_s,own_x
  ... --job pack --spec-owns <stamp> --run <fit stamp> --part sim_sealed --elig-commit <sha> --elig-path <path> --elig-sha256 <sha256>
"""

import hashlib
import io
import json
import math
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
APP = "probe-shift_aswin-surya_sub-p2probes_owns_kill-2026-11-13"
X2E = "experiments/2026-10-14_q35_sol_avg_POSTHOC"
PINNED = {  # name -> (commit, path, sha256): materialized from git at launch, refused on any byte difference
    "run_avg_transfer.py": ("4d1eec572", f"{X2E}/run_avg_transfer.py", "aa55026b169f3e9c1b16ab24986963c062cfb874710de686b18b8fe56357f1df"),
    "run_correctedb.py": ("4d1eec572", f"{X2E}/run_correctedb.py", "41ad64e585b6a4b582bf1906c1bf1fe142a2c17140c3ce6a52ffdcde39d1bac2"),
    "torch_probe_pinned.py": ("be9dbba7c", "src/probe_shift/torch_probe.py", "c4b9cc4f10442d026510b8d0777d3ab8cb0467272d403d455fa4071d3208cf23"),
    "t196.json": ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")}
SPEC = "2026-10-14_163027"  # pass 1's spec (map, arm_of, run_of, skip, fot, n_rows) on the scores Volume
PASS1 = {"spec_sha256": "5e2409a75edd6a92a1eae6c222e032911a17140bcefac80de28fdc4780b70b61", "fit_run": "2026-10-14_175636",
         "recipe_commit": "b5a0586fa946ffd7913568e60c00ab58a30d1e40", "recipe": "experiments/2026-10-14_p2_family_expansion/run_famexp.py"}
APPEND29 = ("a3e89200d0f835ff205f3b0c57d288b3e318715e", "experiments/2026-10-14_priority1_spec_09/RULE.md")
RUN_FOLDS = ("1b21885ec6d61dc044f9ed8b5a80b59add719015", "experiments/2026-10-14_own_s_09/sol36_fold_map.json",
             "67803a26e3b0dcd0a131550dfc97010181a4b1b47ce162b91ac861ab1f1ca59d")  # 09's SCORE-FREE Sol #36 fold map (cell -> fold), as its phase 1 reads
LAYERS, VARIANTS, SEALED = (16, 32, 48), ("own_s", "own_x"), ("sim", "simh_pilot", "simh_full")  # sealed worlds: sim attack and sim honest
TIMEOUT = {"prep": 600, "R": 420, "C": 540, "orch": 1800, "pack": 600}  # expected about 220 s per fit container; worst case under 0d's $5 stop (RULE.md Cost)
G2_AMEND = "7f205d254cddcc5c820892afa5ddc6189ef28a40"  # 09's Append 29b FINAL (200 permutations; strict completeness; p_MC wording), on origin
G2 = {"perms": 200, "block": 50, "timeout": 2450, "margin_s": 200}  # Aswin's 200 (RULE.md Amendment G2); 24 H100 containers; worst under the $76 stop
CTRL_TOL = 1e-4  # positive control: refit own_frac100 vs pass 1's weights, per-run mean held-out score (pass 1's pilot rule)
C, SEED, PERM_SEED = 0.01, 20261014, [20261014, 29]
PILOT = {"readout": "C", "L": 16, "perms": 10, "budget_s": {"h100": 600, "cpu": 840}, "timeout": {"h100": 900, "cpu": 1500}}
BUILD = Path("/tmp/owns_build")

if modal.is_local():
    BUILD.mkdir(exist_ok=True)
    for name, (commit, path, want) in PINNED.items():
        raw = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
        if hashlib.sha256(raw).hexdigest() != want:
            raise SystemExit(f"{name}: pinned bytes differ ({commit}:{path})")
        (BUILD / name).write_bytes(raw)

app = modal.App(APP)
OUT = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2)
VOLS = {"/b": modal.Volume.from_name("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2).read_only(),
        "/sol": modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2).read_only(), "/out": OUT,
        "/q3": modal.Volume.from_name("probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13", version=2).read_only()}  # sim-honest dumps
img = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3", "transformers==5.19.0", "jinja2", "safetensors==0.8.0")
       .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift").add_local_dir(BUILD, "/root/x2e"))


def _x2e():
    sys.path.insert(0, "/root/x2e")
    import run_correctedb as rc  # it imports run_avg_transfer as rat, as in 2e's containers
    import torch_probe_pinned as tp

    return rc, rc.rat, tp


def select(spec: dict, readout: str, pos_world: str, neg_world, ref_pos: str = "run1", ref_neg: str = "honest_a") -> dict:
    """Append 29 item 1: per fold f, the training keys of each class = a prefix of a fresh random.Random(20261014) shuffle of the
    sorted unique run keys ("<world>|<cell>") with rows in this readout and main fold != f; prefix length = min(own_frac100's
    count of that class, available). own_frac100's counts: run1 / honest_a runs with rows in folds != f (all its train tasks)."""
    ri = 0 if readout == "R" else 1
    runs = {}
    for k, (w, t) in spec["map"].items():
        if spec["n_rows"][k][ri] > 0:
            runs.setdefault(w, set()).add(f"{w}|{t}")
    out = {}
    for f in range(5):
        sel = {}
        for cls, ws, ref in ((1, (pos_world,), ref_pos), (0, (neg_world,) if isinstance(neg_world, str) else tuple(neg_world), ref_neg)):
            avail = sorted(u for w in ws for u in runs.get(w, ()) if spec["fot"][u.split("|", 1)[1]] != f)
            n_ref = sum(spec["fot"][u.split("|", 1)[1]] != f for u in runs.get(ref, ()))
            keys = list(avail)
            random.Random(SEED).shuffle(keys)
            sel[cls] = {"keys": sorted(keys[: min(n_ref, len(avail))]), "own_frac100_count": n_ref, "available": len(avail),
                        "matched": len(avail) >= n_ref}
        out[str(f)] = sel
    return out


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(32768, 32768), timeout=PILOT["timeout"]["h100"], retries=0, scaledown_window=2, volumes=VOLS)
def g2pilot_h100(stamp: str) -> dict:
    return _g2pilot(stamp, "h100")


@app.function(image=img, cpu=(16, 16), memory=(32768, 32768), timeout=PILOT["timeout"]["cpu"], retries=0, scaledown_window=2, volumes=VOLS)
def g2pilot_cpu(stamp: str) -> dict:
    return _g2pilot(stamp, "cpu")


def _g2pilot(stamp: str, device: str) -> dict:
    """10 within-main whole-run label permutations of own_x-shaped data (sim positives count-matched to own_frac100, honest_a
    negatives), each refitting the five-fold pipeline at PILOT's readout x layer; stops starting fits after the budget."""
    import numpy as np
    import torch

    t0 = time.time()
    _, rat, tp = _x2e()
    if device == "cpu":
        torch.set_num_threads(16)
    dev = "cuda" if device == "h100" else "cpu"
    spec = json.loads((Path("/out/famexp") / SPEC / f"spec_{SPEC}.json").read_text())
    readout, L = PILOT["readout"], PILOT["L"]
    sel = select(spec, readout, "sim", "honest_a")
    keep = {k: v for k, v in spec["map"].items() if v[0] in ("sim", "honest_a")}
    own = {"layout": "b", "kind": "call_end" if readout == "C" else "output_end_mean", "map": keep, "arm_of": spec["arm_of"], "run_of": spec["run_of"],
           "skip": spec["skip"]}
    Xn, names, fold = (rat._b_rows_cmd if readout == "C" else rat._b_rows)(own, L, spec["fot"])
    X = torch.from_numpy(Xn).to(dev)
    load_s = time.time() - t0
    main_of = {c["key"]: c["main"] for c in json.loads(Path("/root/x2e/t196.json").read_text())["cells"]}
    runs = sorted(set(names.tolist()))
    lab0 = {u: 1.0 if u.startswith("sim|") else 0.0 for u in runs}
    rng = np.random.default_rng(PERM_SEED)
    by_main = {}
    for u in runs:
        by_main.setdefault(main_of[u.split("|", 1)[1]], []).append(u)
    mixed = sum(len({lab0[u] for u in us}) == 2 for us in by_main.values())
    fits, perms_done, stopped = [], 0, False
    for p in range(PILOT["perms"]):
        lab = {}
        for m in sorted(by_main):  # whole-run labels permuted within main (class counts kept)
            us = by_main[m]
            for u, y_ in zip(us, rng.permutation([lab0[u] for u in us]), strict=True):
                lab[u] = float(y_)
        y = np.array([lab[u] for u in names.tolist()])
        for f in range(5):
            if time.time() - t0 > PILOT["budget_s"][device] + load_s:
                stopped = True
                break
            tr_runs = set(sel[str(f)][1]["keys"]) | set(sel[str(f)][0]["keys"])
            tr = np.flatnonzero(np.array([u in tr_runs for u in names.tolist()]) & (fold != f))
            t1 = time.time()
            Xs = X[torch.as_tensor(tr, device=dev)].double()
            mean, std = Xs.mean(0), Xs.std(0, unbiased=False)
            std = std.where(std > 0, std.new_ones(()))
            W, b, cert = tp.fit((Xs - mean) / std, torch.as_tensor(y[tr], device=dev, dtype=torch.float64), C, allow_uncertified=True, return_certificate=True)
            del Xs
            t2 = time.time()
            sc = np.flatnonzero(fold == f)
            z = tp.decision((X[torch.as_tensor(sc, device=dev)].double() - mean) / std, W, b)
            if dev == "cuda":
                torch.cuda.synchronize()
            del z  # DISCARDED: held-out scores are never written, printed or logged
            fits.append({"perm": p, "fold": f, "fit_s": round(t2 - t1, 3), "score_s": round(time.time() - t2, 3), "n_rows": len(tr),
                         "n_runs": len(tr_runs), "n_evals": int(cert["n_evals"]), "certified": bool(cert["certified"]), "max_grad": float(cert["max_grad"])})
            print(f"[{device}] perm {p} fold {f}: fit {t2 - t1:.1f} s, evals {cert['n_evals']}, certified {cert['certified']}", flush=True)
        if stopped:
            break
        perms_done += 1
    ft = [x["fit_s"] for x in fits]
    rec = {"device": device, "gpu": torch.cuda.get_device_name() if dev == "cuda" else None, "readout": readout, "layer": L, "load_s": round(load_s, 1),
           "rows": len(names), "runs": len(runs), "mains": len(by_main), "mixed_class_mains": mixed, "single_class_mains": len(by_main) - mixed,
           "perms_done": perms_done, "fits_done": len(fits), "stopped_at_budget": stopped, "fit_s_mean": float(np.mean(ft)) if ft else None,
           "fit_s_median": float(np.median(ft)) if ft else None, "fit_s_max": max(ft) if ft else None, "all_certified": all(x["certified"] for x in fits),
           "selection_counts": {f: {c: {k: v for k, v in s.items() if k != "keys"} for c, s in d.items()} for f, d in sel.items()},
           "fits": fits, "seconds": round(time.time() - t0, 1), "label": "TIMING ONLY; scores discarded"}
    p_ = Path("/out/owns/g2pilot") / f"g2pilot_{device}_{stamp}.json"
    p_.parent.mkdir(parents=True, exist_ok=True)
    with p_.open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    OUT.commit()
    return {k: v for k, v in rec.items() if k != "fits"}


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _j(p: Path, obj) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as fh:  # write-once
        fh.write(json.dumps(obj, indent=1, default=float))
    return _sha(p)


def _main_of() -> dict:
    return {c["key"]: c["main"] for c in json.loads(Path("/root/x2e/t196.json").read_text())["cells"]}


@app.function(image=img, cpu=(4, 4), memory=(16384, 16384), timeout=TIMEOUT["prep"], retries=0, scaledown_window=2, volumes=VOLS)
def prep_owns(stamp: str, dumps: list, counted: dict, fold_of_task: dict) -> dict:
    """Pass 1's spec + every counted sim-honest epoch (one per counted key; staged inputs, completion record and metas checked as
    2e's cb_c_gate does; content-only command skips by its rule); the fold map checked against 09's score-free Sol #36 map; then
    the frozen own_s / own_x training-key lists (Append 29 item 1) and their run / main / row counts."""
    import numpy as np

    if {d["world"] for d in dumps} != {"simh_full", "simh_pilot"}:
        raise SystemExit("own_s's negatives are the pilot AND the continuation (Append 29): dumps.json needs both worlds")
    key_of = lambda x: x["volume_path"].split("/", 1)[1] if "volume_path" in x else f"{x['launch']}/{x['eval_file']}"  # noqa: E731  (pilot v3 / primary_v3)
    want_w = {w: {key_of(x): x for x in xs} for w, xs in counted.items()}
    seen_w = {w: set() for w in want_w}
    sp = Path("/out/famexp") / SPEC / f"spec_{SPEC}.json"
    if _sha(sp) != PASS1["spec_sha256"]:
        raise SystemExit("pass 1's spec differs from its pin")
    spec, main_of, fails = json.loads(sp.read_text()), _main_of(), []
    fixed = set(spec["fixed_ids"])
    fails += [f"Sol #36 fold map {t}: {fold_of_task[t]} vs {f}" for t, f in spec["fot"].items() if t in fold_of_task and fold_of_task[t] != f]  # same fold wherever both map
    absent_09 = sorted(t for t in spec["fot"] if t not in fold_of_task)  # 09's map holds the 195-task universe; such a cell keeps pass 1's fold
    fails += [f"Sol #36 fold map {t}: {f} vs {spec['fot'][t]}" for t, f in fold_of_task.items() if t in spec["fot"] and spec["fot"][t] != f]
    fom = {}
    for t, f in spec["fot"].items():
        fom.setdefault(main_of[t], set()).add(f)
    fails += [f"main {m} has two folds" for m, fs in fom.items() if len(fs) != 1]
    add = {k: {} for k in ("map", "arm_of", "run_of", "skip", "n_rows")}
    for d in dumps:
        root, inp = Path("/q3") / d["run"], Path("/q3") / d["staging"] / "inputs" / f"{d['arm']}.jsonl"
        if _sha(inp) != d["staged_sha256"]:
            fails.append(f"{d['world']}: staged inputs sha256 differs")
            continue
        eps_l = [json.loads(x) for x in inp.read_text().splitlines() if x]
        eps = {e["key"]: e for e in eps_l}
        fails += [f"{d['world']}: duplicate staged keys"] if len(eps) != len(eps_l) else []
        done = json.loads((root / "complete.json").read_text())
        if done.get("inputs_sha256", {}).get(f"{d['arm']}.jsonl") != d["staged_sha256"] or done.get("epochs") != len(eps) or len(eps) != d["epochs"]:
            fails.append(f"{d['world']}: complete.json or epoch count ({done.get('epochs')} done, {len(eps)} staged, {d['epochs']} pinned)")
        metas = [json.loads(q.read_text()) for q in sorted((root / "meta" / d["arm"]).glob("*.json"))]
        by_run = {m["run"]: m for m in metas}
        want_all = want_w[d["world"]]  # a world's counted keys may be split over several dumps (simh_full = b1 + b2)
        want = {r_: want_all[r_] for r_ in by_run if r_ in want_all}
        mkeys = [m["key"] for m in metas]
        if len(by_run) != len(metas) or len(want_all) != len(counted[d["world"]]) or len(set(mkeys)) != len(mkeys) or \
                set(by_run) - set(want_all) or set(mkeys) != set(eps) or len({x["task"] for x in want_all.values()}) != len(want_all) or \
                set(by_run) & seen_w[d["world"]]:
            fails.append(f"{d['world']} {d['run']}: metas ({len(metas)}) not a disjoint subset of the counted keys ({len(want_all)}), or metas != staged epochs ({len(eps)})")
        seen_w[d["world"]] |= set(by_run)
        chunk_idx = {}
        for r_, x in sorted(want.items()):
            m = by_run.get(r_)
            if m is None:
                continue
            k, ep = m["key"], eps.get(m["key"], {})
            if m.get("inputs_sha256") != d["staged_sha256"] or m["task"] != x["task"] or len(m["calls"]) != len(ep.get("calls", [])) or \
                    not (root / "meta" / d["arm"] / f"{hashlib.sha1(k.encode()).hexdigest()[:20]}.json").exists() or m["task"] not in spec["fot"]:
                fails.append(f"{d['world']} {x['task']}: meta hash, task, call count, file name or fold")
                continue
            bad = ""
            for c, x_ in zip(ep["calls"], m["calls"], strict=True):  # as cb_c_gate: identity, readout row, span-mean row
                want_mean = c["kind"] == "output_end" and c["readout"] is not None and bool(c.get("span"))
                if ((c["irid"], c["call_index"], c["kind"]) != (x_["irid"], x_["call_index"], x_["kind"]) or (c["readout"] is not None) != (x_.get("row") is not None)
                        or want_mean != (x_.get("row_mean") is not None)):
                    bad = f"call mismatch at irid {c['irid']}"
                    break
            rows_ = [x_[f_] for x_ in m["calls"] for f_ in ("row", "row_mean") if x_.get(f_) is not None]
            n_, r0 = m.get("n_rows"), m.get("row_start")
            if not bad and not (isinstance(n_, int) and isinstance(r0, int) and r0 >= 0 and n_ == len(rows_) and sorted(rows_) == list(range(n_))):
                bad = "rows are not 0..n_rows-1 or row_start invalid"
            if not bad:
                if m["chunk"] not in chunk_idx:
                    chunk_idx[m["chunk"]] = json.loads((root / "chunks" / f"{m['chunk']}.json").read_text())
                ent = [e for e in chunk_idx[m["chunk"]]["epochs"] if e["key"] == k]
                if len(ent) != 1 or (ent[0]["arm"], ent[0]["row_start"], ent[0]["n_rows"]) != (d["arm"], r0, n_):
                    bad = "chunk index entry differs from the meta"
                for L_ in (0, *LAYERS, 63):
                    A = np.load(root / "by_layer" / f"L{L_}" / f"{m['chunk']}.npy", mmap_mode="r")
                    if A.shape[0] != chunk_idx[m["chunk"]]["rows"] or r0 + n_ > A.shape[0]:
                        bad = f"rows beyond chunk (or chunk size != index) at L{L_}"
            if bad:
                fails.append(f"{d['world']} {x['task']}: {bad}")
                continue
            sk = [[c["irid"], c["call_index"]] for c in ep["calls"] if c["kind"] == "call_end" and c["readout"] is not None and ep["ids"][c["readout"]] in fixed]
            st = {tuple(c) for c in sk}
            add["map"][k], add["arm_of"][k], add["run_of"][k], add["skip"][k] = [d["world"], m["task"]], d["arm"], str(root), sk  # Path("/b") / "/q3/x" = /q3/x
            add["n_rows"][k] = [sum(c["kind"] == "output_end" and c.get("row_mean") is not None for c in m["calls"]),
                                sum(c["kind"] == "call_end" and c.get("row") is not None and (c["irid"], c["call_index"]) not in st for c in m["calls"])]
    fails += [f"{w}: dumps hold {len(seen_w[w])} of {len(want_w[w])} counted keys" for w in want_w if seen_w[w] != set(want_w[w])]
    wt = [f"{w}|{t}" for w, t in list(spec["map"].values()) + list(add["map"].values())]
    fails += ["two epochs share a world|cell"] if len(wt) != len(set(wt)) else []
    if fails:
        raise SystemExit(f"prep failed ({len(fails)}): {fails[:6]}")
    so = {"fot": spec["fot"], "fixed_ids": spec["fixed_ids"]} | {k: spec[k] | add[k] for k in add}
    so["selection"] = {"own_s": {r: select(so, r, "sim", tuple(sorted({d["world"] for d in dumps}))) for r in ("R", "C")},
                       "own_x": {r: select(so, r, "sim", "honest_a") for r in ("R", "C")}}
    rows = {f"{w}|{t}": so["n_rows"][k] for k, (w, t) in so["map"].items()}
    so["counts"] = {v: {r: {f: {c: {"runs": len(x["keys"]), "mains": len({main_of[u.split("|", 1)[1]] for u in x["keys"]}),
                                    "rows": sum(rows[u][r == "C"] for u in x["keys"]), "own_frac100_count": x["own_frac100_count"],
                                    "available": x["available"], "matched": x["matched"]} for c, x in d_.items()}
                            for f, d_ in sel.items()} for r, sel in by_r.items()} for v, by_r in so["selection"].items()}
    so["t196_cells_absent_from_09_map"] = {t: spec["fot"][t] for t in absent_09}
    so["pins"] = {"pass1_spec": [SPEC, PASS1["spec_sha256"]], "dumps": dumps, "append29": APPEND29, "sol36_fold_map": RUN_FOLDS, "recipe": PASS1}
    h = _j(Path("/out/owns") / stamp / f"spec_owns_{stamp}.json", so)
    OUT.commit()
    return {"spec_sha256": h, "sim_honest_epochs": len(add["map"]), "selection": so["selection"], "counts": so["counts"]}


def _fit_one(tp, X, y, tr, dev):
    """pass 1's _fit_one: float64 standardization on the training rows, then a certified fit (C = 0.01)."""
    import torch

    t0 = time.time()
    Xs = X[torch.as_tensor(tr, device=dev)].double()
    mean, std = Xs.mean(0), Xs.std(0, unbiased=False)
    std = std.where(std > 0, std.new_ones(()))
    del Xs
    Xt = (X[torch.as_tensor(tr, device=dev)].double() - mean) / std
    W, b, cert = tp.fit(Xt, torch.as_tensor(y[tr], device=dev, dtype=torch.float64), C, allow_uncertified=True, return_certificate=True)
    torch.cuda.synchronize()
    return (W, b, mean, std), {k: cert[k] for k in ("n", "max_grad", "certified", "n_evals")}, time.time() - t0


def _fit_rl(spec_stamp: str, readout: str, L: int, run_stamp: str, variants: list, attempt: int) -> dict:
    import numpy as np
    import torch
    from safetensors.numpy import load_file, save_file

    rc, rat, tp = _x2e()
    t0, dev = time.time(), "cuda"
    base = Path("/out/owns") / spec_stamp
    od = base / f"run_{run_stamp}" / f"{readout}_L{L}_a{attempt}"
    done = od / "group.json"
    if done.exists():
        try:
            return json.loads(done.read_text())
        except json.JSONDecodeError:
            raise SystemExit(f"{done}: truncated completion record: stop and inspect") from None
    if od.exists():
        raise SystemExit(f"{od}: a partial attempt folder exists (no completion record): the next attempt uses a new folder")
    so, main_of = json.loads((base / f"spec_owns_{spec_stamp}.json").read_text()), _main_of()
    own = {"layout": "b", "kind": "output_end_mean" if readout == "R" else "call_end", "map": so["map"], "arm_of": so["arm_of"],
           "run_of": so["run_of"]} | ({"skip": so["skip"]} if readout == "C" else {})
    Xo, no, fo = (rat._b_rows if readout == "R" else rat._b_rows_cmd)(own, L, so["fot"])
    ci, ir = rc._row_calls(own, cmd=(readout == "C"))
    X, names = torch.from_numpy(Xo).to(dev), no.tolist()
    wo = np.array([u.split("|", 1)[0] for u in names])
    load_s, fits, ctrl = time.time() - t0, [], []
    p1 = Path("/out/famexp") / SPEC / f"run_{PASS1['fit_run']}"
    for f in range(5):
        ho = np.flatnonzero(fo == f)  # every world's held-out rows of fold f: its main's designated model
        test_mains = {main_of[u.split("|", 1)[1]] for u in set(no[ho].tolist())}
        # positive control: own_frac100_L{L} refit exactly as pass 1 (all run1 / honest_a rows of folds != f), vs pass 1's weights
        yc = np.where(wo == "run1", 1.0, np.where(wo == "honest_a", 0.0, np.nan))
        trc = np.flatnonzero(~np.isnan(yc) & (fo != f))
        mc, certc, secs = _fit_one(tp, X, yc, trc, dev)
        g1 = json.loads((p1 / "groups" / f"own_{readout}_L{L}_f{f}.json").read_text())
        rec1 = [x for x in g1["fits"] if x["variant"] == f"own_frac100_L{L}" and x["fold"] == f]
        wp = p1 / "weights" / f"own_frac100_L{L}__{readout}__L{L}__f{f}.safetensors"
        if len(rec1) != 1 or _sha(wp) != rec1[0]["sha256"]["weights"]:
            raise SystemExit(f"pass 1's own_frac100_L{L} {readout} f{f} weights differ from its fit record")
        R1 = {k: torch.as_tensor(v, device=dev) for k, v in load_file(str(wp)).items()}
        hr = ho[~np.isin(wo[ho], SEALED)]  # the control reads REAL rows only (Codex round 25: no sealed-score diagnostics)
        Xh = X[torch.as_tensor(hr, device=dev)].double()
        zr = tp.decision((Xh - R1["mean"]) / R1["std"], R1["W"], R1["b"]).cpu().numpy()[:, 0]
        zm = tp.decision((Xh - mc[2]) / mc[3], mc[0], mc[1]).cpu().numpy()[:, 0]
        _, inv = np.unique(no[hr], return_inverse=True)
        rmd = float(np.abs(np.bincount(inv, weights=zr - zm) / np.bincount(inv)).max())
        ctrl.append({"fold": f, "max_abs_row_diff": float(np.abs(zr - zm).max()), "max_abs_run_mean_diff": rmd, "bit_identical": bool((zr == zm).all()),
                     "max_abs_W_diff": float((R1["W"] - mc[0]).abs().max()), "certified": certc["certified"], "seconds": secs, "passes": rmd <= CTRL_TOL})
        del Xh, zr, zm
        if not ctrl[-1]["passes"]:
            _j(od / "control_failed.json", {"control": ctrl})
            OUT.commit()
            raise SystemExit(f"positive control failed: own_frac100_L{L} {readout} f{f} refit differs from pass 1 by {rmd:.3g} (> {CTRL_TOL})")
        for v in variants:
            sel = so["selection"][v][readout][str(f)]
            pos, neg = set(sel["1"]["keys"]), set(sel["0"]["keys"])
            lab = {u: 1.0 for u in pos} | {u: 0.0 for u in neg}
            y = np.array([lab.get(u, np.nan) for u in names])
            tr = np.flatnonzero(~np.isnan(y))
            tr_mains = {main_of[u.split("|", 1)[1]] for u in pos | neg}
            if set(no[tr].tolist()) != pos | neg or (fo[tr] == f).any() or tr_mains & test_mains:  # Append 29: in-sample leakage assert
                raise SystemExit(f"LEAKAGE or missing rows: {v} {readout} L{L} f{f}: a training run in fold f, a shared main, or a run without rows")
            m_, cert, secs = _fit_one(tp, X, y, tr, dev)
            z = tp.decision((X[torch.as_tensor(ho, device=dev)].double() - m_[2]) / m_[3], m_[0], m_[1]).cpu().numpy()[:, 0]
            if not np.isfinite(z).all():
                raise ValueError(f"non-finite scores {v} {readout} L{L} f{f}")
            name, hs = f"{v}_L{L}", {}
            for part, keep in (("open", ~np.isin(wo[ho], SEALED)), ("sim_sealed", np.isin(wo[ho], SEALED))):  # the firewall: sim attack + sim honest apart
                hs[part] = rc._save_schema(od / part / f"{name}__{readout}__L{L}__f{f}.npz", no[ho][keep], np.asarray(ci)[ho][keep], np.asarray(ir)[ho][keep],
                                           fo[ho][keep], z[keep], layer=np.full(int(keep.sum()), L, dtype=np.int64))
            wpath = od / "weights" / f"{name}__{readout}__L{L}__f{f}.safetensors"
            wpath.parent.mkdir(parents=True, exist_ok=True)
            save_file({k: np.ascontiguousarray(t.cpu().numpy()) for k, t in zip(("W", "b", "mean", "std"), m_, strict=True)}, str(wpath))
            hs["weights"] = _sha(wpath)
            fits.append({"variant": name, "fold": f, "layer": L, "readout": readout, "seconds": secs, **cert, "sha256": hs,
                         "train": {"pos_runs": len(pos), "neg_runs": len(neg), "mains": len(tr_mains), "rows": int(len(tr))}})
            print(f"[{readout} L{L}] {name} f{f}: {secs:.1f} s, certified {cert['certified']}", flush=True)
    out = {"readout": readout, "layer": L, "attempt": attempt, "dir": str(od.relative_to("/out")), "load_seconds": load_s, "seconds": time.time() - t0,
           "control": ctrl, "fits": fits, "gpu": torch.cuda.get_device_name()}
    _j(done, out)
    OUT.commit()
    return out


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(32768, 32768), timeout=TIMEOUT["R"], retries=0, scaledown_window=2, volumes=VOLS)
def fit_R(spec_stamp: str, L: int, run_stamp: str, variants: list, attempt: int) -> dict:
    return _fit_rl(spec_stamp, "R", L, run_stamp, variants, attempt)


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(32768, 32768), timeout=TIMEOUT["C"], retries=0, scaledown_window=2, volumes=VOLS)
def fit_C(spec_stamp: str, L: int, run_stamp: str, variants: list, attempt: int) -> dict:
    return _fit_rl(spec_stamp, "C", L, run_stamp, variants, attempt)


@app.function(image=img, cpu=(2, 2), memory=(16384, 16384), timeout=TIMEOUT["orch"], retries=0, scaledown_window=2, volumes=VOLS)
def orchestrate(spec_stamp: str, spec_sha: str, run_stamp: str, variants: list, split: dict) -> dict:
    """The 6 fit containers in parallel (one infra retry each, into its own attempt folder), the fit manifest, then the open pack."""
    base = Path("/out/owns") / spec_stamp
    if _sha(base / f"spec_owns_{spec_stamp}.json") != spec_sha:
        raise SystemExit("spec_owns differs from the sha the launcher committed (frozen training keys)")
    fn = {"R": fit_R, "C": fit_C}
    groups, failed, retries = {}, [], 1  # ONE infra retry in the whole job (keeps the worst case under the $5 stop)
    calls = {(r, L): fn[r].spawn(spec_stamp, L, run_stamp, variants, 1) for r in ("R", "C") for L in LAYERS}
    for (r, L), c in calls.items():
        for attempt in (1, 2):
            try:
                groups[f"{r}_L{L}"] = (c if attempt == 1 else fn[r].spawn(spec_stamp, L, run_stamp, variants, 2)).get()
                break
            except BaseException as e:  # noqa: BLE001 - one infra retry into its own folder, then recorded (type only)
                stop = "positive control failed" in str(e) or "LEAKAGE" in str(e)
                if stop or attempt == 2 or retries == 0:
                    failed.append({"group": f"{r}_L{L}", "attempt": attempt, "error": type(e).__name__, "stop": str(e)[:200] if stop else ""})
                    break
                retries -= 1
    OUT.reload()
    fits = [x for g in groups.values() for x in g["fits"]]
    man = {"spec_stamp": spec_stamp, "run_stamp": run_stamp, "variants": variants, "groups": groups, "failed_groups": failed,
           "uncertified": [f"{x['variant']} {x['readout']} f{x['fold']}" for x in fits if not x["certified"]],
           "control_all_pass": all(c["passes"] for g in groups.values() for c in g["control"]) and not failed}
    h = _j(base / f"fit_{run_stamp}.json", man)
    OUT.commit()
    res = {"fit_manifest_sha256": h, "fits": len(fits), "failed_groups": failed, "uncertified": man["uncertified"], "control_all_pass": man["control_all_pass"]}
    if failed and any(x["stop"] for x in failed):
        return res | {"open_pack": "not built: a stop condition (control or leakage)"}
    try:
        res["open_pack"] = pack_core(spec_stamp, run_stamp, "open", run_stamp, split, None, OUT.commit)
    except BaseException as e:  # noqa: BLE001 - the fits stand; the pack is rerun by --job pack
        res["open_pack"] = {"error": type(e).__name__, "detail": str(e)[:300]}
    return res


def _save(path, r, readout):
    """run_famexp_pack._save: per-row key/unit/call_index/score/fold plus run_key/run_score/run_fold (reply mean, command MAX)."""
    import numpy as np

    keys, inv = np.unique(r["key"], return_inverse=True)
    rs = (np.bincount(inv, weights=r["score"]) / np.bincount(inv)) if readout == "R" else np.full(len(keys), -np.inf)
    if readout == "C":
        np.maximum.at(rs, inv, r["score"])
    rf = np.zeros(len(keys), dtype=np.int64)
    rf[inv] = r["fold"]
    if any(len(set(r["fold"][inv == i].tolist())) != 1 for i in range(len(keys))):
        raise ValueError("a run has rows from two folds")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, key=r["key"], unit=r["unit"], call_index=r["call_index"], score=r["score"], fold=r["fold"], run_key=keys, run_score=rs, run_fold=rf)
    path.open("xb").write(buf.getvalue())
    return {"path": str(path.relative_to("/out")), "sha256": hashlib.sha256(buf.getvalue()).hexdigest(), "rows": int(len(r["key"])), "runs": int(len(keys))}


@app.function(image=img, cpu=(4, 4), memory=(32768, 32768), timeout=TIMEOUT["pack"], retries=0, scaledown_window=2, volumes=VOLS)
def pack(spec_stamp: str, run_stamp: str, part: str, stamp: str, split: dict, auth: dict | None = None) -> dict:
    return pack_core(spec_stamp, run_stamp, part, stamp, split, auth, OUT.commit)


def phase1_frozen(auth: dict, decided: set) -> bool:
    """Codex rounds 25-26: per decided variant x readout, numeric per-fold moments (5 folds, sd > 0) and a numeric threshold per
    alpha; paired_layer_sets with exactly R|C x 0.01|0.05, each a list of layers from 16/32/48; primary_cell a frozen choice."""
    try:
        return _phase1_frozen(auth, decided)
    except (TypeError, AttributeError, KeyError):  # malformed nesting fails closed (Codex round 27)
        return False


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _phase1_frozen(auth: dict, decided: set) -> bool:
    num = _num
    dec = auth.get("decisions") or {}
    for n, r in decided:
        d = dec.get(f"{n}|{r}") or {}
        mo = d.get("moments") or {}
        if not all(num((mo.get("mu") or {}).get(str(f))) and num((mo.get("sd") or {}).get(str(f))) and mo["sd"][str(f)] > 0 for f in range(5)):
            return False
        if not all(num((d.get(a) or {}).get("threshold")) for a in ("0.01", "0.05")):
            return False
    E = auth.get("paired_layer_sets")
    if not isinstance(E, dict) or set(E) != {f"{r}|{a}" for r in ("R", "C") for a in ("0.01", "0.05")}:
        return False
    if not all(isinstance(v, list) and set(v) <= set(LAYERS) and all(isinstance(x, int) for x in v) for v in E.values()):
        return False
    return auth.get("primary_cell") in ("R|0.05", "C|0.05", "unavailable")


def pack_core(spec_stamp: str, run_stamp: str, part: str, stamp: str, split: dict, auth: dict | None, commit) -> dict:
    """One npz per (variant, readout), as run_famexp_pack: open = real arms and honest; sim_sealed = sim attack AND sim honest,
    only with 09's phase-1 record (verified here before any sealed file is opened)."""
    import numpy as np

    base = Path("/out/owns") / spec_stamp
    man = json.loads((base / f"fit_{run_stamp}.json").read_text())
    roster = [f"{v}_L{L}" for v in man["variants"] for L in LAYERS]
    if part == "sim_sealed":
        auth = auth or {}
        rec_ = auth.get("result", auth).get("phase1_eligibility_record", auth.get("result", auth))
        auth = rec_ | {"rows": rec_.get("eligibility_table", rec_.get("rows", [])), "eligibility_record_sha256": auth.get("eligibility_record_sha256"),
                       "commit": auth.get("commit"), "path": auth.get("path")}
        op = base / f"pack_open_{auth.get('open_pack_stamp', 'missing')}" / "manifest.json"
        if not op.exists() or _sha(op) != auth.get("open_pack_sha256") or (auth.get("spec_stamp"), auth.get("run_stamp")) != (spec_stamp, run_stamp):
            raise SystemExit("the phase-1 record does not name this spec, run and open pack")
        rows = {(x.get("variant"), x.get("readout"), str(x.get("alpha"))): x for x in auth.get("rows", [])}
        want = {(n, r, a) for n in roster for r in ("R", "C") for a in ("0.01", "0.05")}
        ok = lambda x: (x.get("status") in ("eligible", "ineligible") and _num(x.get("real_tpr")) and x.get("pool") in ("B", "BuC")) or \
            (x.get("status") == "incomplete" and bool(x.get("reason")))  # noqa: E731
        decided = {(n, r) for (n, r, a), x in rows.items() if x.get("status") in ("eligible", "ineligible")}
        x_view = auth | {"paired_layer_sets": auth.get("paired_layer_sets_own_x"), "primary_cell": auth.get("primary_cell_own_x")}  # 09: own_x's own keys
        if set(rows) != want or not all(map(ok, rows.values())) or not phase1_frozen(auth, decided) or \
                ("own_x" in man["variants"] and not phase1_frozen(x_view, decided)):
            raise SystemExit(f"the phase-1 record is not a complete freeze: {len(want)} rows, per-row calibration (per-fold moments, thresholds), "
                             "paired_layer_sets for R|C x 0.01|0.05 (layer lists), and primary_cell in R|0.05 / C|0.05 / unavailable (Append 29 item 2)")
    elig, cohort, main_of = set(split["eligibility_half"]), split["cohort"], split["main_of"]
    sel = sorted(t for t in cohort if t in elig)
    ex_mains = sorted({main_of[t] for t in elig})
    est = sorted(t for t in cohort if t not in elig and main_of[t] not in set(ex_mains))
    if len(elig) != 97 or len(sel) != 93:
        raise SystemExit(f"v6 split mismatch: {len(elig)} / {len(sel)} (want 97 / 93)")
    so = json.loads((base / f"spec_owns_{spec_stamp}.json").read_text())
    fits = [x for g in man["groups"].values() for x in g["fits"]]
    gdir = {(g["readout"], g["layer"]): Path("/out") / g["dir"] for g in man["groups"].values()}
    out = {"part": part, "spec_stamp": spec_stamp, "run_stamp": run_stamp, "pack_stamp": stamp, **({"open_pack_stamp": stamp} if part == "open" else {}),
           "failed_groups": [x["group"] for x in man["failed_groups"]], "spec": "09's Append 29 @ " + APPEND29[0],
           "authorization": None if part == "open" else {"eligibility_record_sha256": auth.get("eligibility_record_sha256"),
                                                         "open_manifest_sha256": auth.get("open_pack_sha256"), "spec_stamp": spec_stamp, "run_stamp": run_stamp},
           "eligibility_record_ref": None if part == "open" else {"commit": auth.get("commit"), "path": auth.get("path")},
           "selection_tasks": sel, "estimation_tasks_7b": est, "excluded_mains_7b": ex_mains, "variants": {}}
    sealed_w = set(SEALED)
    for name in roster:
        v, L = name.rsplit("_L", 1)[0], int(name.rsplit("_L", 1)[1])
        for readout in ("R", "C"):
            fs = {x["fold"]: x for x in fits if x["variant"] == name and x["readout"] == readout}
            e = {"recipe_id": name, "family": v, "layer": L, "readout": readout, "sealed": part == "sim_sealed",
                 "training": {"positives": "v6 sim (September sim epoch-1 attacks on T196)",
                              "negatives": "counted sim-honest runs" if v == "own_s" else "own_a's real honest batch (honest_a)",
                              "counts": so["counts"][v][readout]},
                 "folds": {f"{f}|L{L}": {k: x.get(k) for k in ("fold", "layer", "certified", "max_grad", "n", "n_evals", "sha256")} for f, x in sorted(fs.items())},
                 "control_own_frac100": (man["groups"].get(f"{readout}_L{L}") or {}).get("control")}
            probs = [f"fold {f} has no fit" for f in range(5) if f not in fs] + [f"fold {f} not certified" for f, x in fs.items() if not x["certified"]]
            parts_ = []
            for f, x in sorted(fs.items()):
                q = gdir[(readout, L)] / part / f"{name}__{readout}__L{L}__f{f}.npz"
                if not q.exists() or _sha(q) != x["sha256"].get(part):
                    probs.append(f"fold {f}: file differs from its certified fit")
                    continue
                z = np.load(q)
                if set(z["fold"].tolist()) - {f} or set(z["layer"].tolist()) - {L}:
                    probs.append(f"fold {f}: a row from another fold or layer")
                    continue
                parts_.append({k: z[k] for k in ("key", "unit", "call_index", "score", "fold")})
            r = {k: np.concatenate([p_[k] for p_ in parts_]) for k in ("key", "unit", "call_index", "score", "fold")} if parts_ else None
            if r is not None:
                world = np.array([k.split("|", 1)[0] for k in r["key"].tolist()])
                keep = np.isin(world, list(sealed_w)) if part == "sim_sealed" else ~np.isin(world, list(sealed_w))
                r = {k: a[keep] for k, a in r.items()}
            have = set() if r is None else set(r["key"].tolist())
            ri = readout == "C"
            if part == "open":
                need = {f"{w}|{t}" for t in cohort for w in ("run1", "run2")} | \
                       {f"{w}|{t}" for k, (w, t) in so["map"].items() if w in ("honest_a", "honest_b", "honest_c") and so["n_rows"][k][ri] > 0}
            else:
                need = {f"sim|{t}" for t in cohort} | {f"{w}|{t}" for k, (w, t) in so["map"].items() if w in sealed_w - {"sim"} and so["n_rows"][k][ri] > 0}
            miss = sorted(need - have)
            probs += [f"{len(miss)} required keys without a score"] if miss else []
            e |= {"complete": not probs and r is not None, "incomplete_reasons": probs, "missing": {"n": len(miss), "keys": miss}}
            if part == "open":  # sealed coverage as COUNTS only
                qs = [gdir[(readout, L)] / "sim_sealed" / f"{name}__{readout}__L{L}__f{f}.npz" for f in range(5)] if (readout, L) in gdir else []
                sk = [k for q in qs if q.exists() for k in np.load(q)["key"].tolist()]
                e["sealed_runs_present"] = {w: len({k for k in sk if k.split("|", 1)[0] == w}) for w in SEALED}
            if r is None:
                r = {"key": np.array([], dtype="<U1"), "unit": np.array([], dtype=np.int64), "call_index": np.array([], dtype=np.int64),
                     "score": np.array([], dtype=np.float64), "fold": np.array([], dtype=np.int64)}
            e["npz"] = _save(base / f"pack_{part}_{stamp}" / f"{name}__{readout}.npz", r, readout)
            out["variants"][f"{name}|{readout}"] = e
    js = json.dumps(out)
    if part == "open" and ("sim|" in js or "simh_pilot|" in js or "simh_full|" in js):
        raise SystemExit("internal: a sealed key reached the open manifest")
    mp = base / f"pack_{part}_{stamp}" / "manifest.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.open("x").write(json.dumps(out, indent=1, default=float))
    commit()
    return {"manifest": "/p2/" + str(mp.relative_to("/out")), "sha256": _sha(mp), "selection": len(sel), "estimation": len(est),
            "incomplete": [k for k, v in out["variants"].items() if not v["complete"]]}


def _auc(s, y):
    """AUROC of run scores s for labels y (1 = attack): P(pos > neg) + 0.5 P(tie)."""
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None, 0  # undefined: Append 29b makes the permutation (and so the cell) incomplete
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))), len(pos) * len(neg)


def _g2_block(spec_stamp: str, g2_stamp: str, readout: str, L: int, perms: list) -> dict:
    """G2's within-main shuffle reference (Append 29 item 2; Amendment G2): per permutation, whole-run labels permuted within main
    over own_s's population (sim and counted sim-honest runs with rows), then the COMPLETE five-fold pipeline: count-matched
    selection, training-only preprocessing, certified fit, held-out run scores, AUROC within each scoring model aggregated by
    pair counts. perm -1 = the observed labels. SEALED: AUROCs go only to write-once files under sealed/; nothing is printed or returned."""
    import numpy as np
    import torch

    _, rat, tp = _x2e()
    t0, dev = time.time(), "cuda"
    base = Path("/out/owns") / spec_stamp
    so, main_of = json.loads((base / f"spec_owns_{spec_stamp}.json").read_text()), _main_of()
    keep = {k: v for k, v in so["map"].items() if v[0] in SEALED}
    own = {"layout": "b", "kind": "output_end_mean" if readout == "R" else "call_end", "map": keep, "arm_of": so["arm_of"],
           "run_of": so["run_of"]} | ({"skip": so["skip"]} if readout == "C" else {})
    Xo, no, _ = (rat._b_rows if readout == "R" else rat._b_rows_cmd)(own, L, so["fot"])
    X, names = torch.from_numpy(Xo).to(dev), no.tolist()
    load_s = time.time() - t0
    runs = sorted(set(names))
    rfold = {u: so["fot"][u.split("|", 1)[1]] for u in runs}
    lab0 = {u: 1 if u.startswith("sim|") else 0 for u in runs}
    by_main = {}
    for u in runs:
        by_main.setdefault(main_of[u.split("|", 1)[1]], []).append(u)
    order = sorted(by_main)
    rng, labs = np.random.default_rng(PERM_SEED), {-1: lab0}
    for p in range(G2["perms"]):  # the whole sequence is drawn in every container; each uses its own block
        lab = {}
        for m in order:
            us = by_main[m]
            lab |= dict(zip(us, (int(x) for x in rng.permutation([lab0[u] for u in us])), strict=True))
        if p in perms:
            labs[p] = lab
    nref = {(f, c): so["counts"]["own_s"][readout][str(f)][str(c)]["own_frac100_count"] for f in range(5) for c in (0, 1)}
    row_idx = {}
    for i, u in enumerate(names):
        row_idx.setdefault(u, []).append(i)
    od = base / f"g2_{g2_stamp}" / "sealed" / f"{readout}_L{L}"
    done, fit_s, cert_all = [], [], True
    for p in ([-1] if -1 in perms else []) + [q for q in perms if q >= 0]:
        if time.time() - t0 > G2["timeout"] - G2["margin_s"]:
            break
        fp = od / f"perm_{p}.json"
        if fp.exists():
            done.append(p)
            continue
        lab, per_fold = labs[p], []
        for f in range(5):
            sel = {}
            for c in (0, 1):  # Append 29 item 1's selection, on THIS permutation's labels
                avail = sorted(u for u in runs if lab[u] == c and rfold[u] != f)
                ks = list(avail)
                random.Random(SEED).shuffle(ks)
                sel[c] = sorted(ks[: min(nref[(f, c)], len(avail))])
            if p == -1:
                frozen = so["selection"]["own_s"][readout][str(f)]
                if sel[1] != frozen["1"]["keys"] or sel[0] != frozen["0"]["keys"]:
                    raise SystemExit(f"G2 observed selection differs from own_s's frozen keys ({readout} L{L} f{f})")
            trr = sel[0] + sel[1]
            if {main_of[u.split("|", 1)[1]] for u in trr} & {main_of[u.split("|", 1)[1]] for u in runs if rfold[u] == f}:
                raise SystemExit(f"LEAKAGE in G2 ({readout} L{L} f{f} perm {p})")
            tr = np.array(sorted(i for u in trr for i in row_idx[u]))
            y = np.array([float(lab[u]) for u in names])
            m_, cert, secs = _fit_one(tp, X, y, tr, dev)
            fit_s.append(secs)
            cert_all &= bool(cert["certified"])
            ho = [u for u in runs if rfold[u] == f]
            hi = np.array([i for u in ho for i in row_idx[u]])
            z = tp.decision((X[torch.as_tensor(hi, device=dev)].double() - m_[2]) / m_[3], m_[0], m_[1]).cpu().numpy()[:, 0]
            if not cert["certified"] or not np.isfinite(z).all():
                per_fold = None  # Codex round 26: an uncertified fit or a non-finite score makes this permutation incomplete
                break
            pos_, rs = 0, []
            for u in ho:  # run score: reply mean, command max (rows in row order, as v7)
                k_ = len(row_idx[u])
                rs.append(float(np.mean(z[pos_: pos_ + k_])) if readout == "R" else float(np.max(z[pos_: pos_ + k_])))
                pos_ += k_
            a, pairs = _auc(np.array(rs), np.array([lab[u] for u in ho]))
            if a is None:
                per_fold = None
                break
            per_fold.append({"fold": f, "auc": a, "pairs": pairs, "certified": bool(cert["certified"])})
        if per_fold is None:
            _j(fp, {"perm": p, "complete": False, "reason": "an uncertified fit, a non-finite score or an undefined fold AUROC"})
        else:
            tot = sum(x["pairs"] for x in per_fold)
            _j(fp, {"perm": p, "complete": True, "auc": sum(x["auc"] * x["pairs"] for x in per_fold) / tot, "per_fold": per_fold})
        OUT.commit()
        done.append(p)
        print(f"[G2 {readout} L{L}] perm {p} done at {time.time() - t0:.0f} s", flush=True)  # no AUROC printed
    mixed = sum(len({lab0[u] for u in us}) == 2 for us in by_main.values())
    return {"readout": readout, "layer": L, "perms_requested": len(perms), "perms_done": len(done), "load_s": load_s, "fits": len(fit_s),
            "fit_s_mean": float(np.mean(fit_s)) if fit_s else None, "all_certified": cert_all, "mains": len(by_main), "mixed_class_mains": mixed,
            "single_class_mains": len(by_main) - mixed, "seconds": time.time() - t0}


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(32768, 32768), timeout=G2["timeout"], retries=0, scaledown_window=2, volumes=VOLS)
def g2_block(spec_stamp: str, g2_stamp: str, readout: str, L: int, perms: list) -> dict:
    return _g2_block(spec_stamp, g2_stamp, readout, L, perms)


@app.function(image=img, cpu=(2, 2), memory=(8192, 8192), timeout=G2["timeout"] + 300, retries=0, scaledown_window=2, volumes=VOLS)
def g2_orchestrate(spec_stamp: str, spec_sha: str, g2_stamp: str, resume: bool = False) -> dict:
    """24 blocks (readout x layer x 4 blocks of 50 permutations; block 0 also runs the observed labels), no retries; then the
    SEALED summary (pointwise p per readout x layer) and an open completion record (counts and timing only)."""
    base = Path("/out/owns") / spec_stamp
    if _sha(base / f"spec_owns_{spec_stamp}.json") != spec_sha:
        raise SystemExit("spec_owns differs from the committed prep record")
    gd = base / f"g2_{g2_stamp}"
    if gd.exists() != resume:
        raise SystemExit("a new G2 run never reuses a stamp; an existing G2 folder is resumed only with --g2-resume")
    att = 1 + len(list(gd.glob("attempt_a*.json")))  # the attempt number is reserved before any block starts (Codex round 27)
    _j(gd / f"attempt_a{att}.json", {"attempt": att, "resume": resume})
    OUT.commit()
    blocks = [(r, L, ([-1] if b == 0 else []) + list(range(b * G2["block"], (b + 1) * G2["block"])))
              for r in ("R", "C") for L in LAYERS for b in range(G2["perms"] // G2["block"])]
    if resume:  # only blocks with a missing permutation file are respawned (bounds a resume's worst case to those blocks)
        blocks = [(r, L, ps) for r, L, ps in blocks if any(not (gd / "sealed" / f"{r}_L{L}" / f"perm_{p}.json").exists() for p in ps)]
    calls = [(r, L, ps, g2_block.spawn(spec_stamp, g2_stamp, r, L, ps)) for r, L, ps in blocks]
    res, failed = [], []
    for r, L, ps, c in calls:
        try:
            res.append(c.get())
        except BaseException as e:  # noqa: BLE001 - recorded (type only); completed permutations stay on the Volume
            failed.append({"readout": r, "layer": L, "first_perm": ps[-1] - G2["block"] + 1, "error": type(e).__name__})
    OUT.reload()
    summ = {}
    for r in ("R", "C"):
        for L in LAYERS:
            d = base / f"g2_{g2_stamp}" / "sealed" / f"{r}_L{L}"
            got = {int(q.stem.split("_", 1)[1]): json.loads(q.read_text()) for q in d.glob("perm_*.json")} if d.exists() else {}
            obs = got.get(-1) if (got.get(-1) or {}).get("complete") else None
            ok_ = sorted(k for k, v in got.items() if k >= 0 and v.get("complete"))
            perm = [got[k]["auc"] for k in ok_]
            done_ = obs is not None and ok_ == list(range(G2["perms"]))  # Codex round 26: the approved 200, all complete, or no p
            summ[f"{r}|L{L}"] = {"complete": done_, "observed_auc": obs["auc"] if obs else None, "observed_per_fold": obs["per_fold"] if obs else None,
                                 "n_perms_complete": len(perm), "n_perms_incomplete": sum(1 for k, v in got.items() if k >= 0 and not v.get("complete")),
                                 "n_perms_missing": G2["perms"] - sum(1 for k in got if k >= 0), "perm_aucs": perm,
                                 "n_perm_ge_observed": sum(a >= obs["auc"] for a in perm) if done_ else None,
                                 "p_mc": (1 + sum(a >= obs["auc"] for a in perm)) / (1 + G2["perms"]) if done_ else None,
                                 "p_pointwise": (1 + sum(a >= obs["auc"] for a in perm)) / (1 + G2["perms"]) if done_ else None,  # 09's reader's name (Codex round 32)
                                 "p_text": (f"p_MC = {1 + sum(a >= obs['auc'] for a in perm)}/201; {sum(a >= obs['auc'] for a in perm)}/200 permutation AUROCs >= observed"
                                            if done_ else "incomplete (no p)"),
                                 "perm_p95": sorted(perm)[int(0.95 * len(perm))] if done_ else None}
    _j(gd / "sealed" / f"summary_a{att}.json", {"spec": "Append 29 item 2 + Amendment G2", "attempt": att, "readout_layer": summ,
                                                 "p_rule": "Append 29b @ 7f205d254: p_MC = (1 + #{perm AUROC >= observed}) / 201, upper tail, ties inclusive, pointwise per probe x readout; "
                                                           "complete only if all 5 observed fold fits and all 5 fold fits of every one of the 200 permutations certify and every fold AUROC is defined; "
                                                           "at the floor read 'p_MC = 1/201; 0/200 permutation AUROCs >= observed', never an exhaustive p <= 0.005"})
    rec = {"g2_stamp": g2_stamp, "attempt": att, "blocks": len(blocks), "failed_blocks": failed, "per_block": res,
           "perms_done": {f"{r}|L{L}": sum(x["perms_done"] for x in res if x["readout"] == r and x["layer"] == L) for r in ("R", "C") for L in LAYERS},
           "cells_complete": {k: v["complete"] for k, v in summ.items()}}
    h = _j(gd / f"completion_a{att}.json", rec)
    OUT.commit()
    return {"completion_sha256": h, "attempt": att, "failed_blocks": failed, "perms_done": rec["perms_done"], "cells_complete": rec["cells_complete"],
            "mains": {r: next(({k: x[k] for k in ("mains", "mixed_class_mains", "single_class_mains")} for x in res if x["readout"] == r), None) for r in ("R", "C")}}


@app.function(image=img, cpu=(1, 1), memory=(2048, 2048), timeout=120, retries=0, scaledown_window=2, volumes=VOLS)
def g2_read(spec_stamp: str, fit_run: str, g2_stamp: str, phase2: dict) -> dict:
    """Release: only a structured phase-2 record (phase 2, this spec and fit run, and the sha256 of THIS spec's sealed pack
    manifest on the Volume) opens G2's sealed summary (the latest attempt)."""
    base = Path("/out/owns") / spec_stamp
    r = phase2.get("result", phase2)
    sp = base / f"pack_sim_sealed_{r.get('sealed_pack_stamp', 'missing')}" / "manifest.json"
    sm = json.loads(sp.read_text()) if sp.exists() else {}
    if r.get("phase") != 2 or (r.get("spec_stamp"), r.get("run_stamp")) != (spec_stamp, fit_run) or not sp.exists() or _sha(sp) != r.get("sealed_manifest_sha256") \
            or (sm.get("part"), sm.get("spec_stamp"), sm.get("run_stamp")) != ("sim_sealed", spec_stamp, fit_run):
        raise SystemExit("not a phase-2 record for this spec, fit run and sealed pack (fields: phase 2, spec_stamp, run_stamp, sealed_pack_stamp, sealed_manifest_sha256)")
    att = max(int(q.stem.rsplit("_a", 1)[1]) for q in (base / f"g2_{g2_stamp}" / "sealed").glob("summary_a*.json"))
    return json.loads((base / f"g2_{g2_stamp}" / "sealed" / f"summary_a{att}.json").read_text())


def load_split():
    """run_famexp_pack.load_split: the pinned v6 split (sha256-checked) and the T196 cell -> main map."""
    pin = json.loads((HERE.parent / "2026-10-14_p2_family_expansion" / "v6_pin.json").read_text())
    raw = subprocess.run(["git", "-C", str(REPO), "show", f"{pin['commit']}:{pin['path']}"], capture_output=True, check=True).stdout
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise SystemExit("the pinned X_err v6 record differs from its sha256")
    v6 = json.loads(raw)["result"]
    t196 = json.loads((BUILD / "t196.json").read_text())
    return pin, {"eligibility_half": v6["universe_and_split"]["eligibility_half"], "cohort": v6["cohort"]["tasks"], "main_of": {c["key"]: c["main"] for c in t196["cells"]}}


@app.local_entrypoint()
def main(job: str, device: str = "h100", spec_owns: str = "", spec_sha: str = "", variants: str = "own_s,own_x", run: str = "", part: str = "",
         elig_commit: str = "", elig_path: str = "", elig_sha256: str = "", g2_resume: str = "", g2: str = ""):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    show = lambda c, p: subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
    me = [str((HERE / f).relative_to(REPO)) for f in ("run_owns.py", "RULE.md", "dumps.json") if (HERE / f).exists()]
    if git("status", "--porcelain", "--untracked-files=all", "--", *me):
        raise SystemExit("commit this script, RULE.md and dumps.json before launching")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"git": git("rev-parse", "HEAD"), "stamp (PT)": stamp, "app": APP, "job": job}
    if job == "g2pilot" and device in ("h100", "cpu"):
        rec |= {"device": device, "result": (g2pilot_h100 if device == "h100" else g2pilot_cpu).remote(stamp)}
        name = f"g2pilot_{device}_{stamp}.json"
    elif job == "prep":
        if not git("branch", "-r", "--contains", APPEND29[0]):
            raise SystemExit("Append 29 FINAL is not on a fetched remote branch")
        dumps = json.loads((HERE / "dumps.json").read_text())["dumps"]
        counted = {}
        for d in dumps:
            raw = show(d["counted_keys"]["commit"], d["counted_keys"]["path"])
            if hashlib.sha256(raw).hexdigest() != d["counted_keys"]["sha256"]:
                raise SystemExit(f"{d['world']}: counted-keys manifest differs from its pin")
            ks = json.loads(raw)["keys"]
            if any(k.get("world") != d["world"] for k in ks):
                raise SystemExit(f"{d['world']}: a counted key names another world")
            counted[d["world"]] = ks
        rf = show(RUN_FOLDS[0], RUN_FOLDS[1])
        if hashlib.sha256(rf).hexdigest() != RUN_FOLDS[2]:
            raise SystemExit("09's Sol #36 fold map differs from its pin")
        rec |= {"dumps": dumps, "result": prep_owns.remote(stamp, dumps, counted, json.loads(rf)["fold_of_task"])}
        name = f"prep_owns_{stamp}.json"
    elif job == "fit" and spec_owns and len(spec_sha) == 64:
        vs = [v for v in variants.split(",") if v]
        if not vs or set(vs) - set(VARIANTS):
            raise SystemExit("--variants: own_s[,own_x]")
        pr = str((HERE / "results" / f"prep_owns_{spec_owns}.json").relative_to(REPO))
        try:
            blob = show("HEAD", pr)
        except subprocess.CalledProcessError:
            raise SystemExit("the prep record (frozen training keys) is not in HEAD") from None
        if git("status", "--porcelain", "--", pr) or blob != (REPO / pr).read_bytes() or json.loads(blob)["result"]["spec_sha256"] != spec_sha \
                or not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", "HEAD").splitlines()):
            raise SystemExit("the prep record differs from HEAD, its spec sha256 differs, or HEAD is not pushed")
        pin, split = load_split()
        call = orchestrate.spawn(spec_owns, spec_sha, stamp, vs, split)
        rec |= {"spec_owns": [spec_owns, spec_sha], "variants": vs, "v6_pin": pin, "call": call.object_id, "result": call.get()}
        name = f"fit_{stamp}.json"
    elif job == "g2" and spec_owns and len(spec_sha) == 64:
        if not G2_AMEND or git("cat-file", "-t", G2_AMEND) != "commit" or not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", G2_AMEND).splitlines()):
            raise SystemExit("G2 HOLD (09 via 4f): no G2 number before 09's Append 29 amendment (200 permutations) is on origin and pinned in G2_AMEND")
        pr = str((HERE / "results" / f"prep_owns_{spec_owns}.json").relative_to(REPO))
        if git("status", "--porcelain", "--", pr) or json.loads(show("HEAD", pr))["result"]["spec_sha256"] != spec_sha:
            raise SystemExit("the prep record is not committed in HEAD, or its spec sha256 differs")
        call = g2_orchestrate.spawn(spec_owns, spec_sha, g2_resume or stamp, bool(g2_resume))
        rec |= {"spec_owns": [spec_owns, spec_sha], "g2_stamp": g2_resume or stamp, "resume": bool(g2_resume), "call": call.object_id, "result": call.get(),
                "sealed": "AUROCs stay on the Volume until 09's phase 2"}
        name = f"g2_{stamp}.json"
    elif job == "g2release" and spec_owns and run and g2 and elig_commit and elig_path and len(elig_sha256) == 64:  # run = own_s's fit run; elig = 09's PHASE-2 record
        if git("cat-file", "-t", elig_commit) != "commit" or not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", elig_commit).splitlines()):
            raise SystemExit("09's phase-2 record commit is not on origin")
        rb = show(elig_commit, elig_path)
        if hashlib.sha256(rb).hexdigest() != elig_sha256 or spec_owns.encode() not in rb:
            raise SystemExit("the phase-2 record's sha256 differs, or it does not name this own_s spec")
        rec |= {"phase2_record": {"commit": elig_commit, "path": elig_path, "sha256": elig_sha256}, "g2_stamp": g2, "result": g2_read.remote(spec_owns, run, g2, json.loads(rb))}
        name = f"g2_release_{stamp}.json"
    elif job == "pack" and part in ("open", "sim_sealed") and spec_owns and run:
        pin, split = load_split()
        auth = None
        if part == "sim_sealed":
            if not (elig_commit and elig_path and len(elig_sha256) == 64) or git("cat-file", "-t", elig_commit) != "commit" or \
                    not any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", elig_commit).splitlines()):
                raise SystemExit("--elig-commit (on origin) --elig-path --elig-sha256 are required for the sealed pack")
            rb = show(elig_commit, elig_path)
            if hashlib.sha256(rb).hexdigest() != elig_sha256:
                raise SystemExit("the phase-1 record's sha256 differs from the one 09 sent")
            auth = json.loads(rb) | {"commit": elig_commit, "path": elig_path, "eligibility_record_sha256": elig_sha256}
        rec |= {"part": part, "v6_pin": pin, "result": pack.remote(spec_owns, run, part, stamp, split, auth)}
        name = f"pack_{part}_{stamp}.json"
    else:
        raise SystemExit("--job g2pilot --device h100|cpu | prep | fit|g2 --spec-owns --spec-sha | pack --spec-owns --run --part ... | g2release ...")
    out = HERE / "results" / name
    out.parent.mkdir(exist_ok=True)
    with out.open("x") as fh:  # results/ is append-only
        fh.write(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float)[:4000])
