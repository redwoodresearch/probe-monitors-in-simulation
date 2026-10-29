"""RULE.md (probe-family expansion): prep, pilot, and the fits of Option T (one H100 per fit group x fold) or C (one per group).
Every Modal call is detached-safe: the local entrypoint triggers ONE remote function; records are written to the Volume inside.
Rows come from 2e's pinned row code (run_avg_transfer.py / run_correctedb.py @ 4d1eec572), fits from torch_probe @ 341ee1585.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach <this file> --job prep|pilot|fit|pack ...
(fit --mode T only on 0d's explicit "launch T"; --mode C only on 0d's explicit go). RULE.md "Jobs" lists every argument.
"""

import hashlib
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
APP = "probe-shift_aswin-surya_sub-p2probes_famexp_kill-2026-11-13"
X2E = "experiments/2026-10-14_q35_sol_avg_POSTHOC"
PINNED = {  # name -> (commit, path, sha256): materialized from git at launch, refused on any byte difference
    "run_avg_transfer.py": ("4d1eec572", f"{X2E}/run_avg_transfer.py", "aa55026b169f3e9c1b16ab24986963c062cfb874710de686b18b8fe56357f1df"),
    "run_correctedb.py": ("4d1eec572", f"{X2E}/run_correctedb.py", "41ad64e585b6a4b582bf1906c1bf1fe142a2c17140c3ce6a52ffdcde39d1bac2"),
    "torch_probe_pinned.py": ("341ee1585", "src/probe_shift/torch_probe.py", "ab336c91c8b9696f29f6f0e9f1c1587296b818604ec1a198c3d74ddced8cf64a"),
    "corrected_map.json": ("4d1eec572", f"{X2E}/config/q4b_epoch_map_correctedb_v1.json", "e9d313bbf26bf02bbfd10fbdaebaf98430f54d513bd205e2d204c3614c38119c"),
    "merge.json": ("4d1eec572", f"{X2E}/results/correctedb_merge_2026-10-14_121609.json", "48c1dc8582f41ff393c047cc42f6600d0ce63a975a43f66e058384f88fb677fc"),
    "avg_v1.json": ("4d1eec572", f"{X2E}/results/avg_transfer_avg_v1_2026-10-14_011347.json", "cc90ed914bdee4119e9648f7bc85f0e835f9fcd61e2ade3a6185b448b95fd4f4"),
    "t196.json": ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb")}
LOCAL_PINS = {"r36": "76b11486cc353ab1eec3", "c_manifest": "98cbbbd52aeb73cfce884b08350130ddd7cc23bf637d1603896f3ae0137be157"}
BUILD = Path("/tmp/famexp_build")  # the pinned copies, rebuilt and re-checked on every launch
C, CUT, DROP = 0.01, 32768, ("0", "n")  # own_a's and Sol (a)'s C; Sol's position cut; Amendment 12's fixed call tokens
FRACTIONS, LAYERS_OWN, KS = (0.10, 0.25, 1.0), (16, 32, 48), (5, 10, 20)
TIMEOUT = {"pilot": 1500, "own": 264, "ad": 359, "own_C": 540, "ad_C": 680, "prep": 900, "orch": 2100}  # expected + 25% (RULE.md Cost); orch includes the open pack

if modal.is_local():
    BUILD.mkdir(exist_ok=True)
    for name, (commit, path, want) in PINNED.items():
        raw = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
        if hashlib.sha256(raw).hexdigest() != want:
            raise SystemExit(f"{name}: pinned bytes differ ({commit}:{path})")
        (BUILD / name).write_bytes(raw)

app = modal.App(APP)
OUT = modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2)
VOLS_RO = {"/b": modal.Volume.from_name("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2).read_only(),
           "/sol": modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2).read_only()}
base = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3", "transformers==5.19.0", "jinja2", "safetensors==0.8.0")
img = (base.env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift")
       .add_local_dir(BUILD, "/root/x2e").add_local_file(HERE / "run_famexp_pack.py", "/root/famexp/run_famexp_pack.py"))


def _x2e():
    sys.path.insert(0, "/root/x2e")
    import run_correctedb as rc  # it imports run_avg_transfer as rat, as in 2e's containers
    import torch_probe_pinned as tp

    return rc, rc.rat, tp


def _j(p: Path, obj) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as fh:  # write-once (proven on the Volume by prep); a truncated record is caught on resume
        fh.write(json.dumps(obj, indent=1, default=float))
    return hashlib.sha256(p.read_bytes()).hexdigest()


@app.function(image=img, cpu=(4, 4), memory=(32768, 32768), timeout=TIMEOUT["prep"], retries=0, scaledown_window=2,
              volumes=VOLS_RO | {"/out": OUT, "/acts": modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only(),
                                 "/hf": modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only()})
def prep(stamp: str, r36: bytes, c_manifest: bytes) -> dict:
    """2e's corrected B + batch C row spec, exactly as run_correctedb's score_remote builds it (cmap, skip_c, folds)."""
    if not hashlib.sha256(r36).hexdigest().startswith(LOCAL_PINS["r36"]) or hashlib.sha256(c_manifest).hexdigest() != LOCAL_PINS["c_manifest"]:
        raise SystemExit("in-job: r36 or C manifest sha256 differs from its pin")
    rc, rat, _ = _x2e()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    X = Path("/root/x2e")
    mrec, cm = json.loads((X / "merge.json").read_text()), json.loads((X / "corrected_map.json").read_text())
    if not mrec.get("gate_passes") or sha(X / "corrected_map.json") != mrec["corrected_map"]["sha256"]:
        raise SystemExit("merge record did not pass, or the corrected map differs from its record")
    fom = {r["main_task"]: r["fold"] for r in json.loads(r36)}  # Sol #36's folds (never canonical_folds)
    fot = {c["key"]: fom[c["main"]] for c in json.loads((X / "t196.json").read_text())["cells"]}
    Path("/tmp/cman.json").write_bytes(c_manifest)
    p = rat.cmd_prep.local()
    cg = rc.cb_c_gate.local(p["fixed_ids"])
    fails = rc._c_checks(cg, "/tmp/cman.json", sha, fot) + [f"B task {t} without a fold" for t in sorted({t for _, t in cm["map"].values()} - set(fot))]
    if fails:
        raise SystemExit(f"C or fold gate failed: {fails[:5]}")
    spec = {"map": cm["map"] | {k: ["honest_c", e["task"]] for k, e in cg["epochs"].items()},
            "arm_of": cm["arm_of"] | {k: rc.C_ARM for k in cg["epochs"]}, "run_of": cm["run_of"] | {k: rc.C_RUN for k in cg["epochs"]},
            "skip": dict(p["skip"]) | {e["key"]: e["skip_corrected"] for e in mrec["epochs"]} | {k: e["skip"] for k, e in cg["epochs"].items()},
            "fot": fot, "sol_runs": [r | {"role": "H"} for r in json.loads(r36) if r["arm"] == rat.H_] + [r | {"role": "E1"} for r in json.loads(r36) if r["arm"] == rat.E1],
            "layer_choice_a": json.loads((X / "avg_v1.json").read_text())["analysis"]["layer_choice"]["a"], "fixed_ids": p["fixed_ids"]}

    def nrows(k):  # rows per epoch at R and C with 2e's own filters (_b_rows / _b_rows_cmd): the frozen required-run set for the pack
        m = json.loads((Path("/b") / spec["run_of"][k] / "meta" / spec["arm_of"][k] / f"{rc.stem(k)}.json").read_text())
        sk = {tuple(x) for x in spec["skip"].get(k, [])}
        return [sum(c["kind"] == "output_end" and c.get("row_mean") is not None for c in m["calls"]),
                sum(c["kind"] == "call_end" and c.get("row") is not None and (c["irid"], c["call_index"]) not in sk for c in m["calls"])]
    spec["n_rows"] = {k: nrows(k) for k in spec["map"]}
    h = _j(Path("/out/famexp") / stamp / f"spec_{stamp}.json", spec)
    OUT.commit()
    return {"spec_sha256": h, "epochs": len(spec["map"]), "c_epochs": cg["n"], "sol_runs": len(spec["sol_runs"])}


def _sol_rows(rat, spec, L, readout):
    """Sol's E1/H rows at L (Arm R's loader): R = output_end_mean, C = content-only call_end (fixed tokens dropped by token_text)."""
    import numpy as np
    root, src, names, folds = Path("/sol") / rat.SOLRUN, [], [], []
    for r in spec["sol_runs"]:
        m = json.loads((root / "meta" / r["arm"] / f"{rat.filename(r['key'])}.json").read_text())
        if m["key"] != r["key"]:
            raise ValueError("Sol meta key mismatch")
        kind = "output_end_mean" if readout == "R" else "call_end"
        if readout == "C" and any(q["kind"] == kind and "token_text" not in q for q in m["positions"]):
            raise ValueError("Sol call_end positions without token_text: cannot drop fixed call tokens")
        rows = [q["row"] for q in m["positions"] if q["kind"] == kind and q.get("row") is not None and q["token"] < CUT
                and (readout == "R" or q["token_text"] not in DROP)]
        if rows:
            src.append((m["chunk"], m["row_start"], rows))
            names += [f"sol:{r['role']}|{r['key']}"] * len(rows)
            folds += [r["fold"]] * len(rows)
    X, cache = [], {}
    for ch, s0, rows in src:
        if ch not in cache:
            cache[ch] = np.load(root / "by_layer" / f"L{L}" / f"{ch}.npy", mmap_mode="r")
        X.append(np.asarray(cache[ch][[s0 + q for q in rows]]).astype(np.float32))
    return np.concatenate(X), np.array(names), np.array(folds)


def _train_tasks(rname, fold, f):
    """own_a's training tasks for outer fold f: tasks in folds != f with run1 or honest_a ROWS in this readout, seeded order."""
    t = sorted({u.split("|", 1)[1] for u, g in zip(rname.tolist(), fold.tolist(), strict=True) if g != f and u.split("|", 1)[0] in ("run1", "honest_a")})
    random.Random(f"famexp|own_a|f{f}").shuffle(t)
    return t


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


def _group(spec_stamp: str, fam: str, readout: str, L: int, folds: list, extra: dict | None = None) -> dict:
    import numpy as np
    import torch
    rc, rat, tp = _x2e()
    t0, dev = time.time(), "cuda"
    od = Path("/out/famexp") / spec_stamp / f"run_{extra.get('run', 'pilot') if extra else 'pilot'}"
    done = od / "groups" / f"{fam}_{readout}_L{L}_f{'-'.join(map(str, folds))}.json"
    if done.exists():  # a retry after a completed attempt (Codex round 6): resume, never overwrite bound files
        try:
            return json.loads(done.read_text())
        except json.JSONDecodeError:  # Codex round 7: a truncated record stops loudly, never recomputes over bound files
            raise SystemExit(f"{done.name}: truncated completion record: stop and inspect") from None
    spec = json.loads((Path("/out/famexp") / spec_stamp / f"spec_{spec_stamp}.json").read_text())
    ph = {}
    tick = lambda name: (ph.__setitem__(name, round(time.time() - t0, 1)), print(f"[{fam} {readout} L{L}] {name} at {ph[name]} s", flush=True))  # noqa: E731
    own = {"layout": "b", "kind": "output_end_mean" if readout == "R" else "call_end", "map": spec["map"], "arm_of": spec["arm_of"],
           "run_of": spec["run_of"]} | ({"skip": spec["skip"]} if readout == "C" else {})
    Xo, no, fo = (rat._b_rows if readout == "R" else rat._b_rows_cmd)(own, L, spec["fot"])
    tick("own rows loaded")
    ci, ir = rc._row_calls(own, cmd=(readout == "C"))
    tick("own row calls read")
    wo = np.array([u.split("|", 1)[0] for u in no])
    if fam == "ad":
        Xs_, ns_, fs_ = _sol_rows(rat, spec, L, readout)
        tick("Sol rows loaded")
        X = torch.from_numpy(np.concatenate([Xs_, Xo])).to(dev)
        role = np.concatenate([np.array([u.split("|", 1)[0] for u in ns_]), wo])
        rname, fold, off = np.concatenate([ns_, no]), np.concatenate([fs_, fo]), len(ns_)
    else:
        X, role, rname, fold, off = torch.from_numpy(Xo).to(dev), wo, no, fo, 0
    load_s = time.time() - t0
    tick("rows on GPU")
    y = np.where(np.isin(role, ["run1", "sol:E1"]), 1.0, np.where(np.isin(role, ["honest_a", "sol:H"]), 0.0, np.nan))
    own_runs = list(dict.fromkeys(no.tolist()))  # map order, runs with rows (Arm R's pool order)
    task_of = lambda u: u.split("|", 1)[1]  # noqa: E731
    out, fits = {"fam": fam, "readout": readout, "layer": L, "folds": folds, "load_seconds": load_s, "rule": "H200 inputs; H100 certified fits"}, []
    for f in folds:
        for v in (extra or {}).get("variants") or (FRACTIONS if fam == "own" else KS):
            if fam == "own":
                ts = _train_tasks(rname, fold, f)
                tt = set(ts[: math.ceil(v * len(ts))])
                tr = np.flatnonzero(~np.isnan(y) & (fold != f) & np.array([task_of(u) in tt for u in rname]))
                st, info = tr, {"train_tasks": len(tt)}
            else:
                pool = [i for i, u in enumerate(own_runs) if u.startswith("run1|") and spec["fot"][task_of(u)] != f]
                pick = {own_runs[i] for i in np.random.default_rng([20261017, v, 0, f]).choice(np.array(pool), v, replace=False)}
                keep = (np.char.startswith(role.astype(str), "sol:") & (fold != f)) | np.isin(rname, sorted(pick)) | ((role == "honest_a") & (fold != f))
                tr = np.flatnonzero(keep & ~np.isnan(y))
                st, info = np.flatnonzero(np.char.startswith(role.astype(str), "sol:") & (fold != f)), {"draw": sorted(pick)}
            m_, cert, secs = _fit_one(tp, X, y, tr, st, dev)
            tick(f"fit {v} f{f} done ({secs:.1f} s)")
            name = f"own_frac{int(round(v * 100))}_L{L}" if fam == "own" else f"ad_k{v}"
            fits.append({"variant": name, "fold": f, "layer": L, "readout": readout, "seconds": secs, **cert, **info})
            sc = np.flatnonzero((fold == f) & (np.arange(len(fold)) >= off))  # held-out own rows of fold f, every world
            z = tp.decision((X[torch.as_tensor(sc, device=dev)].double() - m_[2]) / m_[3], m_[0], m_[1]).cpu().numpy()[:, 0]
            if not np.isfinite(z).all():
                raise ValueError(f"non-finite scores {name} f{f}")
            j, hs = sc - off, {}
            for part, sel in (("open", wo[j] != "sim"), ("sim_sealed", wo[j] == "sim")):  # RULE.md firewall: sim rows apart
                hs[part] = rc._save_schema(od / part / f"{name}__{readout}__L{L}__f{f}.npz", no[j][sel], np.asarray(ci)[j][sel], np.asarray(ir)[j][sel],
                                           fo[j][sel], z[sel], layer=np.full(int(sel.sum()), L, dtype=np.int64))
            from safetensors.numpy import save_file
            wp = od / "weights" / f"{name}__{readout}__L{L}__f{f}.safetensors"
            wp.parent.mkdir(parents=True, exist_ok=True)
            save_file({k: np.ascontiguousarray(t.cpu().numpy()) for k, t in zip(("W", "b", "mean", "std"), m_, strict=True)}, str(wp))
            hs["weights"] = hashlib.sha256(wp.read_bytes()).hexdigest()
            fits[-1]["sha256"] = hs
            if extra and extra.get("ref"):  # pilot reproduction gate: 2e's tight outer weights on the same rows
                out["repro"] = _repro(extra["ref"], X, sc, rname, m_, tp, dev)
    out["fits"], out["seconds"], out["phase_seconds"] = fits, time.time() - t0, ph
    _j(done, out)
    OUT.commit()
    return out


def _repro(ref, X, sc, rname, m_, tp, dev):
    """22c option 2 (tight vs tight): 2e's weights (bare W/b/mean/std or own_a__f<f>__ keys), scored as cb does, vs this fit; per-run means <= 1e-4."""
    import io

    import numpy as np
    import torch
    raw = Path("/sol", ref["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise SystemExit("reference weights sha256 differs from the pin")
    Z = np.load(io.BytesIO(raw))
    pre = "" if "W" in Z.files else f"own_a__f{ref['fold']}__"
    g = lambda k: torch.as_tensor(Z[pre + k], device=dev)  # noqa: E731
    Xf = X[torch.as_tensor(sc, device=dev)]
    ref_z = tp.decision(((Xf - g("mean").float()) / g("std").float()).double(), g("W").double().reshape(-1, 1), g("b").double().reshape(-1)).cpu().numpy()[:, 0]
    mine = tp.decision((Xf.double() - m_[2]) / m_[3], m_[0], m_[1]).cpu().numpy()[:, 0]
    runs, inv = np.unique(rname[sc], return_inverse=True)
    rm = lambda z: np.bincount(inv, weights=z) / np.bincount(inv)  # noqa: E731
    d = float(np.abs(rm(ref_z) - rm(mine)).max())
    return {"runs": int(len(runs)), "rows": int(len(sc)), "format": "bare" if not pre else pre, "max_abs_run_mean_diff": d,
            "max_abs_row_diff": float(np.abs(ref_z - mine).max()), "passes_1e-4": d <= 1e-4}


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(32768, 32768), timeout=TIMEOUT["pilot"], retries=0, scaledown_window=2,
              volumes=VOLS_RO | {"/out": OUT})
def fit_group(spec_stamp: str, fam: str, readout: str, L: int, folds: list, extra: dict | None = None) -> dict:
    return _group(spec_stamp, fam, readout, L, folds, extra)


@app.function(image=img, gpu="H100!", cpu=(4, 4), memory=(32768, 32768), timeout=TIMEOUT["pilot"], retries=0, scaledown_window=2,
              volumes=VOLS_RO | {"/out": OUT})
def pilot(spec_stamp: str, ref: dict, stamp: str) -> dict:
    """RULE.md stop rule 1: one timing fit at the largest n (adapted k=20, fold 0, L46, R) and the own_a reproduction fit."""
    t = _group(spec_stamp, "ad", "R", 46, [0], {"variants": [20], "run": f"pilot_{stamp}"})
    r = _group(spec_stamp, "own", ref["readout"], ref["layer"], [ref["fold"]], {"variants": [1.0], "run": f"pilot_{stamp}", "ref": ref})
    sec = t["fits"][0]["seconds"]
    rec = {"timing_fit_seconds": sec, "assumed_seconds": 21, "timing_ok_1.5x": sec <= 1.5 * 21, "load_seconds": [t["load_seconds"], r["load_seconds"]],
           "timing_cert": t["fits"][0], "repro_cert": r["fits"][0], "repro": r.get("repro"),
           "pass": sec <= 1.5 * 21 and t["fits"][0]["certified"] and r["fits"][0]["certified"] and bool(r.get("repro", {}).get("passes_1e-4"))}
    _j(Path("/out/famexp") / spec_stamp / f"pilot_{stamp}.json", rec)
    OUT.commit()
    return rec


@app.function(image=img, cpu=(4, 4), memory=(32768, 32768), timeout=600, retries=0, scaledown_window=2, volumes={"/out": OUT, "/sol": VOLS_RO["/sol"]})
def pack_open(spec_stamp: str, run_stamp: str, stamp: str, split: dict) -> dict:
    """The OPEN pack right after the fits (critical path): run_famexp_pack.pack_core, the same code as its own launcher."""
    sys.path.insert(0, "/root/famexp")
    import run_famexp_pack as rfp

    return rfp.pack_core(spec_stamp, run_stamp, "open", stamp, split, None, OUT.commit)


@app.function(image=img, cpu=(2, 2), memory=(8192, 8192), timeout=TIMEOUT["orch"], retries=0, scaledown_window=2, volumes={"/out": OUT})
def orchestrate(spec_stamp: str, mode: str, stamp: str, pilot_stamp: str, split: dict) -> dict:
    """Option T: one container per (group, fold); Option C: one per group. Spawned here so a detached launch keeps every call.
    Refuses unless this spec's pilot record passed (stop rule 1)."""
    pr = json.loads((Path("/out/famexp") / spec_stamp / f"pilot_{pilot_stamp}.json").read_text())
    if not pr.get("pass"):
        raise SystemExit("the pilot record did not pass: the full job does not run")
    spec = json.loads((Path("/out/famexp") / spec_stamp / f"spec_{spec_stamp}.json").read_text())
    jobs = [("own", r, L, [f]) for r in ("R", "C") for L in LAYERS_OWN for f in range(5)]
    jobs += [("ad", "R", L, [f]) for f, pair in enumerate(spec["layer_choice_a"]) for L in pair] + [("ad", "C", 63, [f]) for f in range(5)]
    if mode == "C":  # merge folds per (fam, readout, layer)
        g = {}
        [g.setdefault(j[:3], []).extend(j[3]) for j in jobs]
        jobs = [(*k, fs) for k, fs in g.items()]
    calls = [fit_group.with_options(timeout=TIMEOUT[fam if mode == "T" else f"{fam}_C"]).spawn(spec_stamp, fam, r, L, fs, {"run": stamp})
             for fam, r, L, fs in jobs]
    res, fails, first = [], [], []
    for attempt in (1, 2):  # RULE.md: a failed group is rerun once (same code, same spec); only what still fails is recorded failed
        for (fam, r, L, fs), c in zip(jobs, calls, strict=True):
            try:
                res.append(c.get())
            except Exception as e:  # noqa: BLE001 - recorded per group, never silently dropped
                (first if attempt == 1 else fails).append({"group": [fam, r, L, fs], "error": type(e).__name__, "attempt": attempt})
        if attempt == 1 and first:
            jobs = [tuple(g["group"]) for g in first]
            calls = [fit_group.with_options(timeout=TIMEOUT[fam if mode == "T" else f"{fam}_C"]).spawn(spec_stamp, fam, r, L, fs, {"run": stamp})
                     for fam, r, L, fs in jobs]
        else:
            break
    fits = [x for g_ in res for x in g_["fits"]]
    rec = {"mode": mode, "pilot": pilot_stamp, "retried_groups": first, "failed_groups": fails, "fits": len(fits), "uncertified": [x for x in fits if not x["certified"]],
           "fit_seconds": [min(x["seconds"] for x in fits), max(x["seconds"] for x in fits)] if fits else None, "groups": res}
    _j(Path("/out/famexp") / spec_stamp / f"fit_{stamp}.json", rec)
    OUT.commit()
    try:  # the open pack, auto-run after the fits (4f / 0d): its manifest path + sha256 go to 09 for phase 1
        pk = pack_open.remote(spec_stamp, stamp, datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S"), split)
    except BaseException as e:  # noqa: BLE001 - SystemExit included (Codex round 9); the fits stand; run_famexp_pack.py reruns the pack
        pk = {"error": type(e).__name__, "detail": str(e)[:300]}
    return {k: v for k, v in rec.items() if k != "groups"} | {"open_pack": pk}


@app.local_entrypoint()
def main(job: str, spec: str = "", r36: str = "", c_manifest: str = "", mode: str = "", ref_path: str = "", ref_sha: str = "",
         ref_layer: int = -1, ref_kind: str = "R", ref_fold: int = 0, pilot_stamp: str = ""):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    me = [str((HERE / f).relative_to(REPO)) for f in ("run_famexp.py", "RULE.md", "v6_pin.json")]
    if git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--untracked-files=all", "--", *me):
        raise SystemExit("uncommitted changes, or this launcher / RULE.md is untracked: commit and push first (frozen analysis)")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = {"git": git("rev-parse", "HEAD"), "stamp (PT)": stamp, "app": APP, "job": job, "pinned": {k: v[2] for k, v in PINNED.items()}}

    def cohort():  # v6_pin.json: the X_err v6 record (commit, path, sha256) whose split selection uses (RULE.md; Amendment 7b)
        pin = json.loads((HERE / "v6_pin.json").read_text())
        raw = subprocess.run(["git", "-C", str(REPO), "show", f"{pin['commit']}:{pin['path']}"], capture_output=True, check=True).stdout
        if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise SystemExit("the pinned X_err v6 record differs from its sha256")
        return json.loads(raw)["result"]["cohort"]["tasks"]

    if job == "prep":
        b1, b2 = Path(r36).read_bytes(), Path(c_manifest).read_bytes()
        if not hashlib.sha256(b1).hexdigest().startswith(LOCAL_PINS["r36"]) or hashlib.sha256(b2).hexdigest() != LOCAL_PINS["c_manifest"]:
            raise SystemExit("r36 or C manifest sha256 differs from its pin")
        out = prep.remote(stamp, b1, b2)
    elif job == "pilot":
        if ref_layer < 0 or not ref_path or len(ref_sha) != 64:
            raise SystemExit("the reproduction reference (2e's own_a tight outer weights) is required: --ref-path --ref-sha --ref-layer")
        out = pilot.remote(spec, {"path": ref_path, "sha256": ref_sha, "layer": ref_layer, "readout": ref_kind, "fold": ref_fold}, stamp)
    elif job == "fit" and mode in ("T", "C") and pilot_stamp:
        cohort()  # the v6 pin is a hard precondition of the full job
        sys.path.insert(0, str(HERE))
        import run_famexp_pack as rfp

        out = orchestrate.remote(spec, mode, stamp, pilot_stamp, rfp.load_split()[1])
    else:
        raise SystemExit("--job prep | pilot | fit --mode T|C --pilot-stamp <stamp> (packs: run_famexp_pack.py)")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{job}_{stamp}.json").open("x").write(json.dumps(head | {"result": out}, indent=1, default=float))
    print(json.dumps(head | {"result": out}, indent=1, default=float)[:4000])
