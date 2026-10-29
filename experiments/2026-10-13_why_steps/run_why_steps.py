"""Why Steps 1 and 2 (RULE.md): d9's SCE recipe refit at call_end content-only, in the fix-full world (ff) and the
placeholder world (ph), Qwen3-32B L48 and L63, cells (a) e1 vs H, (b) SA vs SH, (c) SA vs H. Every nested model's
weights are saved for Step 2. Modal; stored activations only; no model calls; no text leaves the Volumes (failures are
re-raised by type only; a Gate's message is numbers only).

  --part ff: pack ff output_end at L63 and refit (c″) as gate 1 (checked here against d9's stored scores, --d9-npz),
             then pack ff call_end content-only at L48/L63 and fit (a), (b), (c).
  --part ph: after q2's check_ph passes (gate 2), pack ph call_end content-only and fit (a), (b), (c).
Writes results/why_<part>_scores_<stamp>.npz (git-ignored; S3 with a manifest) and results/why_<part>_run_<stamp>.json.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run \\
      experiments/2026-10-13_why_steps/run_why_steps.py --part ff --d9-npz <path>
"""

import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-whysteps_fit_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-whysteps_work_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
V4 = "2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1"
FX = "2026-10-13_fixfull_qwen3-32b_l48-63_v2"
PH = "2026-10-13_step1ph_qwen3-32b_l48-63_v1"
D9_NPZ_SHA256 = "48fc5eb52546f8884bf1083a3d8530a6c80624b52c619678edf72c28443826be"
D9_SEED_NPZ_SHA256 = "57c8081115f6daf3cc0fbf93dbf3422392bec5a1779653a44b7d2f72eb3ba6ae"  # sce_shortcut_2026-10-13_170259.npz (d9)
D9_SEED_ARMS = {"sim-attack": "sim-attack-fx-full", "sim-honest": "sim-honest-fx-full"}  # d9: c_ff = sim-attack-fx-full vs real-honest
R36 = REPO / "experiments/2026-10-06_q3_attack_detector/results"
H_, E1, E2, SA, SH = "real-honest", "real-attack-e1", "real-attack-e2", "sim-attack", "sim-honest"
ARMS = (H_, E1, E2, SA, SH)
# world -> arm -> (extraction run, meta arm)
SOURCES = {"ff": {H_: (V4, H_), E1: (V4, E1), E2: (V4, E2), SA: (FX, "sim-attack-fx-full"), SH: (FX, "sim-honest-fx-full")},
           "ph": {a: (PH, f"{a}-ph") for a in ARMS},
           "asis": {a: (V4, a) for a in ARMS}}  # Amendment 1, Q2: unrepaired
# Amendment 4: Seed-OSS-36B, filled in from q2's message; the job refuses to run while any name is "TBD".
SEED_VOL = "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"  # q2, 10-13 ~18:25: one run holds every arm
SEED_ASIS = SEED_FX = "2026-10-13_seedcall_seed-oss-36b_v1"
SEED_LAYERS = (48, 63)  # of G3's 7 saved layers, the relative-depth match to Qwen3-32B (Amendment 4)
ROOT_OF = {"seed_asis": "acts_seed", "seed_ff": "acts_seed"}  # every other world: "acts"
# Amendment 7: the full-repair averaged reply readout. Real arms from d9's mean-pool36 dump; simulated from q2's fxavg dump
# (full + F9, complete arms). ffavgF10: F10-changed runs swapped in by key (a list = sources tried in order).
MP36, FXAVG = "2026-10-12_la_mean-pool36_qwen3-32b_l7_v1", "2026-10-13_q2_fxavg_qwen3-32b_l7_v1"
SOURCES |= {"ffavg": {H_: (MP36, H_), E1: (MP36, E1), E2: (MP36, E2), SA: (FXAVG, "sim-attack-fx-fullF9"),
                      SH: (FXAVG, "sim-honest-fx-fullF9")},
            "ffavgF10": {H_: (MP36, H_), E1: (MP36, E1), E2: (MP36, E2),
                         SA: [(FXAVG, "sim-attack-fx-fullF9F10"), (FXAVG, "sim-attack-fx-fullF9")],
                         SH: [(FXAVG, "sim-honest-fx-fullF9F10"), (FXAVG, "sim-honest-fx-fullF9")]}}
F11 = "2026-10-13_q2_f11_qwen3-32b_l7_v1"  # Amendment 8: F11-changed runs only (427 attack, 226 honest)
SOURCES |= {"ffavgF11": {H_: (MP36, H_), E1: (MP36, E1), E2: (MP36, E2),
                         SA: [(F11, "sim-attack-fx-fullF9F11"), (FXAVG, "sim-attack-fx-fullF9")],
                         SH: [(F11, "sim-honest-fx-fullF9F11"), (FXAVG, "sim-honest-fx-fullF9")]}}
WORLDS_OF = {"seed": ["seed_ff", "seed_asis"], "avg": ["ffavg", "ffavgF10"],  # parts that run several worlds
             "f11": ["ffavg", "ffavgF10", "ffavgF11"], "seed40": ["seed_ff", "seed_asis"], "seedf11": ["seed_ffF9", "seed_ffF11"]}
KINDS_PART = {"f11": ("output_end", "output_end_mean"), "seed40": ("output_end_mean",), "seedf11": ("output_end", "output_end_mean")}
LAYERS_PART = {"seed40": (40,)}  # d9's per-fold Seed cell needs layer 40 (G3: fold 0 -> L40), averaged only  # Amendment 8: last token and averaged, all three repair levels
SOURCES |= {"seed_asis": {a: (SEED_ASIS, a) for a in ARMS},
            "seed_ff": {H_: (SEED_ASIS, H_), E1: (SEED_ASIS, E1), E2: (SEED_ASIS, E2),  # fix-full without F9, as d9's c_ff
                        SA: (SEED_FX, "sim-attack-fx-full"), SH: (SEED_FX, "sim-honest-fx-full")}}
# Amendment 9: Seed F9 and F11 as their own arms (0d relaying Aswin, descriptive only), keys swapped in by key
SEED_F11 = "2026-10-13_f11_seedcall_seed-oss-36b_v1"
SOURCES |= {"seed_ffF9": {H_: (SEED_ASIS, H_), E1: (SEED_ASIS, E1), E2: (SEED_ASIS, E2),
                          SA: [(SEED_FX, "sim-attack-fx-fullF9"), (SEED_FX, "sim-attack-fx-full")],
                          SH: [(SEED_FX, "sim-honest-fx-fullF9"), (SEED_FX, "sim-honest-fx-full")]},
            "seed_ffF11": {H_: (SEED_ASIS, H_), E1: (SEED_ASIS, E1), E2: (SEED_ASIS, E2),
                           SA: [(SEED_F11, "sim-attack-fx-fullF9F11"), (SEED_FX, "sim-attack-fx-fullF9"), (SEED_FX, "sim-attack-fx-full")],
                           SH: [(SEED_F11, "sim-honest-fx-fullF9F11"), (SEED_FX, "sim-honest-fx-fullF9"), (SEED_FX, "sim-honest-fx-full")]}}
ROOT_OF |= {"seed_ffF9": "acts_seed", "seed_ffF11": "acts_seed"}
LAYERS_OF = {"seed_asis": SEED_LAYERS, "seed_ff": SEED_LAYERS, "seed_ffF9": SEED_LAYERS, "seed_ffF11": SEED_LAYERS}
# Q1 for Seed: command (#36's content-only calls, field "primary"), command (Seed's own filter, "content_only", beside),
# reply last token, reply averaged (d9's rule)
KINDS = ("call_end", "call_end_own", "output_end", "output_end_mean")
KINDS_OF = {"seed_asis": KINDS, "seed_ff": KINDS}
KINDS_OF |= {"ffavg": ("output_end_mean",), "ffavgF10": ("output_end_mean",)}
CELLS = {"a": (E1, H_), "b": (SA, SH), "c": (SA, H_)}  # (label 1, label 0)
LAYERS, H, C, MAX_ITER = (48, 63), 5120, 0.01, 1000
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
ROOTS = {"acts": Path("/acts"), "acts_seed": Path("/acts_seed"), "work": Path("/work")}

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
work_vol = modal.Volume.from_name(WORK, version=2, create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")  # #36's and d9's image, unchanged
         .uv_pip_install("torch==2.8.0", "numpy==2.3.3", "scikit-learn==1.7.2", "scipy==1.16.2")
         .add_local_python_source("probe_shift"))


class Gate(Exception):
    """A failed check; its message holds numbers only."""


def sanitized(fn):
    def run(*a):
        try:
            return fn(*a)
        except Gate as e:
            raise RuntimeError(f"{fn.__name__} gate failed: {e}") from None
        except Exception as e:  # noqa: BLE001 — re-raised by type only, never swallowed: a message could carry text
            raise RuntimeError(f"{fn.__name__} failed: {type(e).__name__}") from None
    return run


@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0,
              volumes={"/acts": acts_vol.read_only(), "/work": work_vol}
              | ({} if SEED_VOL in ("TBD", ACTS) else {"/acts_seed": modal.Volume.from_name(SEED_VOL, version=2).read_only()}))
def pack(stamp: str, world: str, layer: int, kind: str, runs: list[dict], match: str | None = None) -> dict:
    return sanitized(_pack)(stamp, world, layer, kind, runs, match)


def _pack(stamp, world, layer, kind, runs, match=None):
    """match: another world whose dump's kept (step, action) positions of this kind restrict this world's (Amendment 3)."""
    from concurrent.futures import ThreadPoolExecutor

    from probe_shift.q1_eval import fixed_call_tokens

    t0 = time.time()
    acts = ROOTS["acts" if SEED_VOL == ACTS else ROOT_OF.get(world, "acts")]
    srcs = lambda w: [s for v in SOURCES[w].values() for s in (v if isinstance(v, list) else [v])]
    need = sorted(set(srcs(world)) | (set(srcs(match)) if match else set()))

    def metas(ext, arm):
        out = {}
        for p in (acts / ext / "meta" / arm).glob("*.json"):
            m = json.loads(p.read_text())
            if m["key"] in out:
                raise Gate(f"{need.index((ext, arm))}: a run extracted twice")
            out[m["key"]] = m
        return out

    with ThreadPoolExecutor(len(need)) as ex:
        M = dict(zip(need, ex.map(lambda t: metas(*t), need)))
    def source(w, a, key):  # the first source holding the run
        v = SOURCES[w][a]
        return next((s for s in (v if isinstance(v, list) else [v]) if key in M[s]), None)

    used = Counter()
    for a in ARMS:  # every run of the arm found in the dump (or, for a list, in one of its sources)
        want = {r["key"] for r in runs if r["arm"] == a}
        miss = [k for k in want if source(world, a, k) is None]
        if miss:
            raise Gate(f"{ARMS.index(a)}: {len(miss)} runs missing from the dump")
        used |= Counter(f"{a}<-{source(world, a, k)[1]}" for k in want)
    fixed = {ext: set(fixed_call_tokens(ext)) for ext, _ in need}
    pkind = "call_end" if kind.startswith("call_end") else kind

    def content(p, ext):  # Seed's dumps carry flags (token_text is None there): required, never defaulted
        if pkind != "call_end":
            return True
        if world.startswith("seed"):
            return bool(p["primary"] if kind == "call_end" else p["content_only"])
        return p["token_text"] not in fixed[ext]
    rows, dropped, none = [], Counter(), Counter()
    for i, r in enumerate(runs):
        ext, arm = source(world, r["arm"], r["key"])
        m = M[(ext, arm)][r["key"]]
        ps = [p for p in m["positions"] if p["kind"] == pkind and p["kept"]]
        if match:
            mm = M[source(match, r["arm"], r["key"])][r["key"]]
            allowed = {(p["step"], p["action"]) for p in mm["positions"] if p["kind"] == pkind and p["kept"]}
            ps = [p for p in ps if (p["step"], p["action"]) in allowed]
        keep = [p for p in ps if content(p, ext)]
        dropped[r["arm"]] += len(ps) - len(keep)
        none[r["arm"]] += not keep
        rows += [(i, ext, m["chunk"], m["row_start"] + p["row"]) for p in keep]
    n_arm = Counter(r["arm"] for r in runs)
    if any(none[a] > 0.2 * n_arm[a] for a in n_arm):  # a missing readout kind in a dump must not pass silently
        raise Gate(f"L{layer} {kind}: runs without rows {dict(none)}")
    X = np.zeros((len(rows), H), np.float16)
    groups = {}
    for j, (_, ext, ch, row) in enumerate(rows):
        groups.setdefault((ext, ch), []).append((j, row))

    def load(item):
        (ext, ch), its = item
        X[[j for j, _ in its]] = np.load(acts / ext / "by_layer" / f"L{layer}" / f"{ch}.npy", mmap_mode="r")[[r for _, r in its]]

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(load, groups.items()))
    if not np.isfinite(X).all():
        raise Gate(f"L{layer}: {int((~np.isfinite(X)).sum())} non-finite features")
    run = np.array([r[0] for r in rows], np.int32)
    d = ROOTS["work"] / "packs" / stamp
    d.mkdir(parents=True, exist_ok=True)
    name = f"{world}_{kind}{'_m' + match if match else ''}_L{layer}"
    with open(d / f"{name}.tmp", "wb") as fh:  # written whole, then renamed
        np.save(fh, X)
    os.replace(d / f"{name}.tmp", d / f"{name}.npy")
    np.savez(d / f"rows_{name}.npz", run=run)
    work_vol.commit()
    return {"world": world, "layer": layer, "kind": kind, "match": match, "rows": len(rows), "runs_by_source": dict(used), "rows_by_arm": dict(Counter(runs[i]["arm"] for i in run.tolist())),
            "fixed_token_rows_dropped": dict(dropped), "runs_without_rows": dict(none),
            "features_sha256": hashlib.sha256(X.tobytes()).hexdigest(), "seconds": round(time.time() - t0)}


@app.function(image=image, gpu="H100!", cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0, volumes={"/work": work_vol})
def fit(stamp: str, world: str, layer: int, kind: str, runs: list[dict], cells: list[str], ref=None, match: str | None = None,
        ref_stop: bool = True) -> dict:
    return sanitized(_fit)(stamp, world, layer, kind, runs, cells, ref, match, ref_stop)


def _fit(stamp, world, layer, kind, runs, cells, ref=None, match=None, ref_stop=True):
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import pool

    t0 = time.time()
    work_vol.reload()
    d, name = ROOTS["work"] / "packs" / stamp, f"{world}_{kind}{'_m' + match if match else ''}_L{layer}"
    r_run = np.load(d / f"rows_{name}.npz")["run"].astype(np.int64)
    X = torch.from_numpy(np.load(d / f"{name}.npy")).cuda().float()
    n = len(runs)
    arm, fold = np.array([r["arm"] for r in runs]), np.array([r["fold"] for r in runs])
    out, info = {}, {"world": world, "layer": layer, "kind": kind, "match": match, "cells": {}}
    for cell in cells:
        tc = time.time()
        pos, neg = CELLS[cell]
        y = np.where(arm == pos, 1.0, np.where(arm == neg, 0.0, np.nan))
        nested = np.full((5, 5, n), np.nan, np.float32)
        Wsave, bsave = np.full((5, 5, H), np.nan, np.float32), np.full((5, 5), np.nan, np.float32)
        msave, ssave = np.full((5, 5, H), np.nan, np.float32), np.full((5, 5, H), np.nan, np.float32)
        evals = []
        for f, v in PAIRS:
            te = (fold == f) | (fold == v)
            tr, sc = np.flatnonzero((~np.isnan(y) & ~te)[r_run]), np.flatnonzero(te[r_run])
            Xtr = X[torch.from_numpy(tr).cuda()]
            mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
            std = std.where(std > 0, std.new_ones(()))
            W, b, n_eval = tp.fit((Xtr - mean) / std, torch.from_numpy(y[r_run[tr]]).cuda().float(), C, MAX_ITER)
            del Xtr
            s = tp.decision((X[torch.from_numpy(sc).cuda()] - mean) / std, W, b).double().cpu().numpy()
            if not np.isfinite(s).all():
                raise Gate(f"L{layer} {cell}: non-finite output scores")
            S = pool(r_run[sc], s, n, "mean")
            S[~te] = np.nan
            nested[f, v] = S[:, 0]
            Wsave[f, v], bsave[f, v] = W.float().cpu().numpy().reshape(-1), float(b)
            msave[f, v], ssave[f, v] = mean.cpu().numpy(), std.cpu().numpy()
            evals.append(int(n_eval))
        wd = ROOTS["work"] / "weights" / stamp
        wd.mkdir(parents=True, exist_ok=True)
        np.savez(wd / f"{name}_{cell}.npz", W=Wsave, b=bsave, mean=msave, std=ssave)
        work_vol.commit()
        info["cells"][cell] = {"train_runs": {"1": int((y == 1).sum()), "0": int((y == 0).sum())}, "max_evals": max(evals),
                               "fits_at_cap": sum(e >= MAX_ITER for e in evals), "seconds": round(time.time() - tc),
                               "weights": f"weights/{stamp}/{name}_{cell}.npz"}
        out[cell] = nested
        if ref is not None:  # gate 1, here in the container: d9's stored nested scores for this cell, by arm and key
            same = bool(np.array_equal(np.isnan(ref), np.isnan(nested)))
            info["gate1"] = {"same_missing": same, "max_abs_diff": float(np.nanmax(np.abs(ref - nested))), "runs_compared": n}
            info["gate1"]["pass"] = bool(same and info["gate1"]["max_abs_diff"] <= 1e-3)
            if ref_stop and not info["gate1"]["pass"]:
                raise Gate(f"gate 1: {info['gate1']}")
    info["seconds"] = round(time.time() - t0)
    return {"info": info, "nested": out}


def build_runs() -> list[dict]:
    """#36's 3,807 runs, then the 437 simulated honest runs, in that order (d9's main-job order)."""
    keep = ("arm", "key", "cell", "main_task", "fold")
    base = [{k: r[k] for k in keep} for r in json.loads((R36 / "run_scores_2026-10-06_181548/runs.json").read_text())]
    sh = [{k: r[k] for k in keep} for r in json.loads((R36 / "sim_honest_scores_2026-10-07_124717/runs.json").read_text()) if r["arm"] == SH]
    return base + sh


def d9_seed_ref(runs: list[dict], d9_npz: Path) -> np.ndarray:
    """d9's refit Seed (c_ff) output_end L63 nested scores, in this job's run order (a read by key; compared remotely)."""
    if hashlib.sha256(d9_npz.read_bytes()).hexdigest() != D9_SEED_NPZ_SHA256:
        raise SystemExit("d9's Seed scores file differs from the pinned sha256")
    Z = np.load(d9_npz)
    at = {(a, k): i for i, (a, k) in enumerate(zip(Z["seed-oss-36b__runs__arm"], Z["seed-oss-36b__runs__key"]))}
    idx = np.array([at[(D9_SEED_ARMS.get(r["arm"], r["arm"]), r["key"])] for r in runs])
    return Z["seed-oss-36b__L63__c_ff__nested_real"][..., idx]


def d9_ref(runs: list[dict], d9_npz: Path) -> np.ndarray:
    """d9's stored (c″) output_end L63 nested scores, in this job's run order (a read by key; the comparison is remote)."""
    if hashlib.sha256(d9_npz.read_bytes()).hexdigest() != D9_NPZ_SHA256:
        raise SystemExit("d9's scores file differs from the pinned sha256")
    Z = np.load(d9_npz)
    d9arm = {SA: "sim-attack-fullfix", SH: "sim-honest-fullfix"}
    at = {(a, k): i for i, (a, k) in enumerate(zip(Z["runs__arm"], Z["runs__key"]))}
    idx = np.array([at[(d9arm.get(r["arm"], r["arm"]), r["key"])] for r in runs])
    return Z["L63__c_full__nested_real"][..., idx]


@app.function(image=image, cpu=1, memory=4096, timeout=3600, retries=0, volumes={"/work": work_vol})
def orchestrate(stamp: str, head: str, part: str, runs: list[dict], ref=None, gate2: str = "") -> dict:
    """The whole job, remote, so a detached run survives the laptop (0d, 10-13 18:1x): gate, packs, fits; the scores and
    the record are written to the work Volume (results/<stamp>/) and returned."""
    import io

    worlds = WORLDS_OF.get(part, [part])
    rec = {"stamp": stamp, "git": head, "app": APP, "part": part, "rule": "RULE.md", "sources": {w: SOURCES[w] for w in worlds},
           "settings": {"cells": CELLS, "C": C, "max_iter": MAX_ITER}, "pack": [], "fit": [], "gate2_evidence": gate2 or None}
    arrays = {}
    if part == "ff":  # gate 1 first: (c″) at output_end L63 against d9's stored scores
        rec["pack"].append(pack.remote(stamp, "ff", 63, "output_end", runs))
        g = fit.remote(stamp, "ff", 63, "output_end", runs, ["c"], ref)  # raises unless gate 1 passes
        rec["fit"].append(g["info"])
        rec["gate1"] = g["info"]["gate1"]
        arrays["L63__ff_output_end__c__nested_real"] = g["nested"]["c"]
    if part == "seed":  # Amendment 4's gate: Seed (c) at output_end L63 against d9's refit Seed probe, reported, not a stop
        rec["pack"].append(pack.remote(stamp, "seed_ff", 63, "output_end", runs))
        g = fit.remote(stamp, "seed_ff", 63, "output_end", runs, ["c"], ref, None, False)
        rec["fit"].append(g["info"])
        rec["gate_seed"] = g["info"]["gate1"]
    jobs = [(w, layer, kind, m) for w in worlds for m in ((None, "ff") if w == "ph" else (None,))
            for kind in KINDS_PART.get(part, KINDS_OF.get(w, ("call_end",))) for layer in LAYERS_PART.get(part, LAYERS_OF.get(w, LAYERS))]
    rec["settings"]["jobs"] = jobs
    rec["pack"] += list(pack.starmap([(stamp, w, layer, kind, runs, m) for w, layer, kind, m in jobs]))
    for x in fit.starmap([(stamp, w, layer, kind, runs, list(CELLS), None, m) for w, layer, kind, m in jobs]):
        rec["fit"].append(x["info"])
        i = x["info"]
        arrays |= {f"L{i['layer']}__{i['world']}_{i['kind']}{'_m' + i['match'] if i['match'] else ''}__{c}__nested_real": v
                   for c, v in x["nested"].items()}
    buf = io.BytesIO()
    np.savez(buf, **arrays, **{f"runs__{k}": np.array([r[k] for r in runs]) for k in ("arm", "key", "cell", "main_task", "fold")})
    data = buf.getvalue()
    rec |= {"scores_file": f"why_{part}_scores_{stamp}.npz", "scores_sha256": hashlib.sha256(data).hexdigest()}
    d = ROOTS["work"] / "results" / stamp
    d.mkdir(parents=True, exist_ok=True)
    (d / rec["scores_file"]).write_bytes(data)
    (d / f"why_{part}_run_{stamp}.json").write_text(json.dumps(rec, indent=1))
    work_vol.commit()
    return {"rec": rec, "npz": data}


@app.local_entrypoint()
def main(part: str, d9_npz: str = "", gate2: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    if part not in ("ff", "ph", "asis", "seed", "avg", "f11", "seed40", "seedf11") or (part in ("ff", "seed") and not d9_npz) \
            or (part in ("ph", "avg", "f11", "seedf11") and not gate2):
        raise SystemExit("--part ff|ph|asis|seed|avg; ff and seed need --d9-npz (their gates), ph and avg need --gate2")
    worlds = WORLDS_OF.get(part, [part])
    if "TBD" in json.dumps([[SOURCES[w], LAYERS_OF.get(w, LAYERS)] for w in worlds] + ([SEED_VOL] if part.startswith("seed") else [])):
        raise SystemExit(f"{part}: the dump names or layers are not filled in yet")
    runs = build_runs()
    if Counter(r["arm"] for r in runs) != {H_: 990, E1: 939, E2: 939, SA: 939, SH: 437} or len({(r["arm"], r["key"]) for r in runs}) != len(runs):
        raise SystemExit("unexpected runs")
    ref = d9_ref(runs, Path(d9_npz)) if part == "ff" else d9_seed_ref(runs, Path(d9_npz)) if part == "seed" else None
    head = git("rev-parse", "HEAD")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(f"stamp {stamp}, git {head}, part {part}; results also on the work Volume at results/{stamp}/", flush=True)
    out = orchestrate.remote(stamp, head, part, runs, ref, gate2)  # the only call: it survives a detached client's exit
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / out["rec"]["scores_file"], "xb") as fh:  # git-ignored (*.npz): archived to S3 with a manifest
        fh.write(out["npz"])
    with (HERE / "results" / f"why_{part}_run_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(out["rec"], indent=1))
    print(json.dumps({k: out["rec"][k] for k in ("stamp", "scores_file", "scores_sha256")}), flush=True)
