"""RULE.md Addendum 1, item 2: the geometry of the refit A2 detector (sim attack vs run1, weights saved by run_detector_refit.py),
from stored activations and stored weights only (no fit). (a) The cosine of the detector's direction with each probe's direction
at the SAME readout and layer (own_frac100_L48 R at reply L48; own_frac100_L48 C at command L48; own_a tight command A at command
L63), per fold model (detector and probe both held out on fold f), in two common spaces: raw activations (a probe's functional
u = W / std, so per-fold standardization is folded into u) and honest-standardized coordinates (u * sigma_H, sigma_H = honest
B u C unit SD per coordinate). (b) At reply L48, the share of each direction's squared norm in the bottom k% of the honest B u C
unit covariance's eigen-directions (raw space; and the correlation matrix's, honest-standardized space), against random
directions; the run2-vs-run1 detector (the same recipe, no signal) and own_frac100_L48 R as references. Honest rows are loaded
with run_simh_control._load (A2's loader, unchanged copy); command skip rows with run_simh_control.skip_rows (tokenizer, unchanged).

  <scratch>/modalenv/bin/modal run --detach experiments/2026-10-15_projection_why_small_gaps/run_detector_geometry.py \\
      --refit experiments/2026-10-15_projection_why_small_gaps/results/detector_refit_<stamp>.json
"""

import hashlib
import io
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_simh_control as base  # noqa: E402  (the unchanged loader; only its app name differs from A2's copy)

APP = "probe-shift_aswin-surya_sub-projection_why_geometry_kill-2026-11-14"
SOLVOL, P2VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"
CELLS = [("reply", 48), ("command", 48), ("command", 63)]
PAIRS = {"reply|L48": "own_frac100_L48_R", "command|L48": "own_frac100_L48_C", "command|L63": "own_a_tight_cmd_A"}
KS, N_RAND, SEED = (1, 5, 10, 25, 50), 1000, 20261015
TIMEOUT = 1200

app = modal.App(APP)
sol_vol = modal.Volume.from_name(SOLVOL, version=2)
p2_vol = modal.Volume.from_name(P2VOL, version=2).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy==2.3.3", "safetensors==0.8.0")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"}).add_local_python_source("probe_shift")
         .add_local_file(HERE / "run_simh_control.py", "/root/run_simh_control.py"))  # base.tok_image's packages plus a pinned safetensors


def _geometry(name: str, spec: dict, meta: dict, od: Path) -> dict:  # noqa: C901
    import numpy as np
    import safetensors.numpy as stn

    t0, log, gates, read_sha = time.time(), [], {}, {}

    def note(msg):
        log.append(f"{time.time() - t0:7.1f} s  {msg}")
        print(log[-1], flush=True)

    def gate(ok, key, info):
        gates[key] = {"pass": bool(ok)} | info
        if not ok:
            (od / f"{name}_GATE_FAILED.json").write_text(json.dumps({"label": "GATE FAILED: stop and ask 2e (no result numbers)", "where": key, "meta": meta,
                                                                     "gates": gates, "log": log}, default=str, indent=1))
            sol_vol.commit()
            raise SystemExit(f"gate {key} failed: {json.dumps(info, default=str)[:800]}")

    def read(path, sha):
        b = Path(path).read_bytes()
        h = hashlib.sha256(b).hexdigest()
        read_sha[str(path)] = h
        gate(h == sha, f"G1 {path}", {"got": h, "pinned": sha})
        return b

    # ---- weights: u = W / std per fold (the functional in raw activation space) ----
    def det(cell, prob):
        e = spec["detector"][cell][prob]
        z = np.load(io.BytesIO(read(Path("/sol") / e["path"], e["sha256"])), allow_pickle=False)
        return {f: (z[f"f{f}__W"].reshape(-1), z[f"f{f}__std"].reshape(-1)) for f in range(5)}

    def famexp(ro):
        out = {}
        for f in range(5):
            e = spec["famexp"][f"{ro}|{f}"]
            t = stn.load(read(Path("/p2") / e["path"], e["sha256"]))
            out[f] = (t["W"].astype(np.float64).reshape(-1), t["std"].astype(np.float64).reshape(-1))
        return out

    def own_cmd():
        c = spec["own_a_cmd"]
        zt = np.load(io.BytesIO(read(Path("/sol") / c["own_a_tight_weights"]["path"], c["own_a_tight_weights"]["sha256"])), allow_pickle=False)
        zo = np.load(io.BytesIO(read(Path("/sol") / c["own_a_weights"]["path"], c["own_a_weights"]["sha256"])), allow_pickle=False)
        out = {}
        for f in range(5):
            z = zt if f"own_a__f{f}__W" in zt else zo
            out[f] = (z[f"own_a__f{f}__W"].astype(np.float64).reshape(-1), z[f"own_a__f{f}__std"].astype(np.float64).reshape(-1))
        return out

    U = {}
    for ro, L in CELLS:
        k = f"{ro}|L{L}"
        for prob in ("c_simatk_vs_run1", "d_run2_vs_run1"):
            U[(k, prob)] = {f: w / s for f, (w, s) in det(k, prob).items()}
    U[("reply|L48", "own_frac100_L48_R")] = {f: w / s for f, (w, s) in famexp("R").items()}
    U[("command|L48", "own_frac100_L48_C")] = {f: w / s for f, (w, s) in famexp("C").items()}
    U[("command|L63", "own_a_tight_cmd_A")] = {f: w / s for f, (w, s) in own_cmd().items()}
    for key, d in U.items():
        if not all(v.shape == (5120,) and np.isfinite(v).all() for v in d.values()):
            raise SystemExit(f"G1: {key}: bad weight vectors")
    note("weights loaded")

    # ---- honest B u C unit rows (A24 membership), per readout and layer ----
    sk = base.skip_rows("/b", spec["want"], spec["shas"])
    gate(sk["fixed_tokens"] == base.FIXED_TOKENS, "G3 fixed call tokens", {"fixed_tokens": sk["fixed_tokens"]})
    epochs = spec["epochs"]
    H = {}
    for ro, L in CELLS:
        X16, row_run, keep, no_rows, chunks = base._load(L, ro, epochs, sk["skip"], spec["allowed"])
        exp = spec["expected"][ro]
        runs_kept = sorted(epochs[i]["name"] for i in keep)
        gate(len(X16) == exp["n_units"] and runs_kept == exp["pool_runs"], f"G4 honest rows {ro}|L{L}",
             {"rows": int(len(X16)), "expected_rows": exp["n_units"], "runs": len(runs_kept), "expected_runs": len(exp["pool_runs"]), "runs_without_rows": no_rows})
        X = X16.astype(np.float64)
        del X16
        mu, sd = X.mean(0), X.std(0)
        if not (sd > 0).all():
            raise SystemExit(f"{ro}|L{L}: a zero-variance coordinate")
        H[f"{ro}|L{L}"] = {"sd": sd, "n": int(len(X))}
        if (ro, L) == ("reply", 48):
            Xc = X - mu
            S = Xc.T @ Xc / len(X)
            del Xc
            lam, V = np.linalg.eigh(S)  # ascending
            Rm = S / np.outer(sd, sd)
            lamR, VR = np.linalg.eigh(Rm)
            H["reply|L48"] |= {"lam": lam, "V": V, "lamR": lamR, "VR": VR}
            del S, Rm
        del X
        note(f"honest {ro}|L{L}: {H[f'{ro}|L{L}']['n']} rows")

    # ---- (a) cosines in two common spaces ----
    rng = np.random.default_rng(SEED)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    space = {"raw": lambda u, k: u, "honest_standardized": lambda u, k: u * H[k]["sd"]}
    cos_res = {}
    for ro, L in CELLS:
        k = f"{ro}|L{L}"
        dc, dd, pr = U[(k, "c_simatk_vs_run1")], U[(k, "d_run2_vs_run1")], U[(k, PAIRS[k])]
        r = {"probe": PAIRS[k]}
        for sp, tf in space.items():
            t = lambda d, f: tf(d[f], k)  # noqa: E731
            per = {
                "detector_vs_probe": [cos(t(dc, f), t(pr, f)) for f in range(5)],
                "detector_vs_run2run1_detector": [cos(t(dc, f), t(dd, f)) for f in range(5)],
                "run2run1_detector_vs_probe": [cos(t(dd, f), t(pr, f)) for f in range(5)],
            }
            cross = {"detector_cross_fold": [cos(t(dc, f), t(dc, g)) for f in range(5) for g in range(f + 1, 5)],
                     "probe_cross_fold": [cos(t(pr, f), t(pr, g)) for f in range(5) for g in range(f + 1, 5)]}
            R = rng.standard_normal((N_RAND, 5120))
            rc = np.array([[abs(cos(t(dc, f), x)) for x in R] for f in range(5)]).ravel()
            r[sp] = {**{m: {"by_fold": v, "mean": float(np.mean(v)), "mean_abs": float(np.mean(np.abs(v)))} for m, v in per.items()},
                     **{m: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for m, v in cross.items()},
                     "random_abs_cos": {"mean": float(rc.mean()), "q95": float(np.quantile(rc, 0.95)), "n": int(len(rc)), "one_over_sqrt_d": float(1 / np.sqrt(5120))}}
        cos_res[k] = r
    note("cosines done")

    # ---- (b) share of the squared norm in the bottom k% of honest variance directions (reply L48) ----
    hk = H["reply|L48"]
    lowvar = {}
    for sp, (lam, V, tf) in {"raw": (hk["lam"], hk["V"], lambda u: u), "honest_standardized": (hk["lamR"], hk["VR"], lambda u: u * hk["sd"])}.items():
        ms = {kk: int(round(kk / 100 * len(lam))) for kk in KS}

        def prof(M, lam=lam, V=V, ms=ms):  # rows of M = directions; per row its squared-coefficient profile in the eigenbasis
            C2 = (M / np.linalg.norm(M, axis=1, keepdims=True)) @ V
            C2 = C2 ** 2
            return [{**{f"bottom_{kk}pct": float(c[:m].sum()) for kk, m in ms.items()}, "variance_ratio": float((c * lam).sum() / lam.mean())} for c in C2]

        out = {}
        for key in (("reply|L48", "c_simatk_vs_run1"), ("reply|L48", "d_run2_vs_run1"), ("reply|L48", "own_frac100_L48_R")):
            pf = prof(np.stack([tf(U[key][f]) for f in range(5)]))
            out[key[1]] = {"by_fold": pf, "mean": {m: float(np.mean([p[m] for p in pf])) for m in pf[0]}}
        rp = prof(rng.standard_normal((N_RAND, 5120)))
        out["random_directions"] = {m: {"mean": float(np.mean([p[m] for p in rp])), "q025": float(np.quantile([p[m] for p in rp], 0.025)),
                                        "q975": float(np.quantile([p[m] for p in rp], 0.975))} for m in rp[0]}
        out["eigen_directions"] = {"n": int(len(lam)), "bottom_counts": ms, "eigenvalue_min": float(lam[0]), "eigenvalue_median": float(np.median(lam)),
                                   "eigenvalue_max": float(lam[-1])}
        lowvar[sp] = out
    note("low-variance shares done")

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        return x

    rec = clean({"label": "RULE.md Addendum 1, item 2: detector geometry (descriptive); PRELIMINARY until 09, Codex and Aswin review", "name": name, "meta": meta,
                 "spaces": {"raw": "u = W / std per fold (the score's functional on raw activations)",
                            "honest_standardized": "u * sigma_H, sigma_H = per-coordinate SD of the honest B u C units at that readout and layer"},
                 "cosines": cos_res, "low_variance_share_reply_L48": lowvar, "honest_rows": {k: v["n"] for k, v in H.items()},
                 "skip": {"fixed_tokens": sk["fixed_tokens"], "dropped_rows": sk["dropped_rows"], "staged_sha256": sk["staged_sha256"]},
                 "settings": {"ks_percent": KS, "n_random": N_RAND, "seed": SEED, "cells": [f"{r}|L{L}" for r, L in CELLS], "pairs": PAIRS},
                 "gates": gates, "gates_pass": True, "read_sha256": read_sha, "log": log})
    blob = json.dumps(rec, allow_nan=False).encode()
    (od / f"{name}.json").open("xb").write(blob)
    sol_vol.commit()
    return {"file": f"{name}.json", "bytes": blob, "sha256": hashlib.sha256(blob).hexdigest()}


@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=TIMEOUT, retries=0,
              volumes={"/b": base.b_vol, "/hf": base.hf_vol, "/sol": sol_vol, "/p2": p2_vol})
def geometry(name: str, spec: dict, meta: dict) -> dict:
    od = Path("/sol/xfer/projection_why") / name
    od.mkdir(parents=True, exist_ok=False)  # the cost guard: a rescheduled attempt stops here
    sol_vol.commit()
    try:
        return _geometry(name, spec, meta, od)
    except BaseException as e:
        if not (od / f"{name}_GATE_FAILED.json").exists():
            (od / f"{name}_FAILED.json").write_text(json.dumps({"label": "FAILED (no result numbers): stop and ask 2e", "meta": meta,
                                                                "error": f"{type(e).__name__}: {str(e)[:2000]}"}, indent=1))
            sol_vol.commit()
        raise


@app.local_entrypoint()
def main(refit: str):
    from zoneinfo import ZoneInfo

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip() or git("status", "--porcelain", "--", str(HERE)).strip():
        raise SystemExit("uncommitted tracked changes, or uncommitted files in this experiment folder")
    head = git("rev-parse", "HEAD").decode().strip()
    rp = base.REPO / refit
    git("ls-files", "--error-unmatch", str(rp))
    rr = json.loads(rp.read_text())
    if not rr.get("reproduction_of_a2", {}).get("pass"):
        raise SystemExit("the refit record did not pass its reproduction gate")
    detector = {k: {p: c["saved_weights"][p] for p in ("c_simatk_vs_run1", "d_run2_vs_run1")} for k, c in rr["cells"].items()}
    cfg = json.loads((HERE / "config/inputs.json").read_text())

    def pinned(spec):
        b = git("show", f"{spec['commit']}:{spec['path']}")
        if hashlib.sha256(b).hexdigest() != spec["sha256"]:
            raise SystemExit(f"{spec['path']}: sha256 mismatch")
        return json.loads(b)

    fpr, own = pinned(cfg["git"]["fpr_record"]), pinned(cfg["git"]["own_a_record"])
    universe = set(fpr["universe"])
    # honest B u C epochs, exactly as run_simh_control.main builds them (config/inputs_v1.json, copied from A2's folder)
    rc = json.loads((HERE / base.CONFIG).read_text())["real"]
    mf = json.loads(base._git_bytes(rc["manifest"]))
    up, st, ex = (json.loads(base._git_bytes(rc[k])) for k in ("c_upload", "c_stage", "c_extract"))
    real = [{k: e[k] for k in ("key", "world", "task", "arm", "extraction_run", "staged_inputs_run", "meta_file")} for e in mf["epochs"] if e["world"] in rc["worlds_from_manifest"]]
    shas = {e["staged_inputs_file"]: e["staged_inputs_sha256"] for e in mf["epochs"] if e["world"] in rc["worlds_from_manifest"]}
    excl = set(st["c"]["c_gate"]["excluded"])
    ce = [{"key": f"{f_['path']}#e0", "world": "honest_c", "task": f_["task"], "arm": "honest_b", "extraction_run": ex["config"]["run"],
           "staged_inputs_run": ex["config"]["inputs"]} for f_ in up["files"] if f"{f_['path']}#e0" not in excl]
    if len(ce) != st["c"]["c_gate"]["staged"]:
        raise SystemExit("batch C epochs differ from its staging gate's count")
    allowed = {ex["config"]["run"]: sorted(c["chunk"] for s_ in ex["shards"] for c in s_["chunks"])}
    shas |= {f"{ex['config']['inputs']}/inputs/{a}": s for a, s in ex["config"]["inputs_sha256"].items()}
    epochs = sorted([e | {"vol": "b", "name": f"{e['world']}|{e['task']}"} for e in real + ce if e["task"] in universe], key=lambda e: e["name"])
    if len({e["name"] for e in epochs}) != len(epochs):
        raise SystemExit("duplicate honest run name")
    want = {}
    for e in epochs:
        want.setdefault(e["staged_inputs_run"], {}).setdefault(e["arm"], []).append(e["key"])
    # expected honest units (the FPR record's B u C pools): reply = Sol (a)'s pool; command = own_a command A's B u C block
    sol = fpr["probes"]["sol_a|avg_top2|corrected|none|tight"]
    (cb,) = [fpr["own_real"]["own_a_tight cmd_A"][b] for b in ("primary", "sensitivity") if fpr["own_real"]["own_a_tight cmd_A"][b]["pool"] == "honest_b+honest_c"]
    expected = {"reply": {"n_units": sum(v["n"] for v in sol["z_moments_unit"].values()), "pool_runs": sorted(sol["pool_runs"])},
                "command": {"n_units": cb["n_pool_units"], "pool_runs": sorted(sol["pool_runs"])}}
    if cb["pool_runs"] != len(expected["command"]["pool_runs"]):
        raise SystemExit("command pool run count differs from Sol (a)'s pool")
    spec = {"detector": detector, "famexp": cfg["weights"]["own_frac100_L48"], "own_a_cmd": own["saved"]["cmd"]["cmdA"], "epochs": epochs, "allowed": allowed,
            "want": want, "shas": shas, "expected": expected}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    name = f"detector_geometry_{stamp}"
    meta = {"git": head, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "stamp_pt": stamp, "app": APP,
            "refit_record": {"path": refit, "sha256": hashlib.sha256(rp.read_bytes()).hexdigest()}, "rule": "experiments/2026-10-15_projection_why_small_gaps/RULE.md (Addendum 1)"}
    print(json.dumps({"name": name, "git": head, "honest_epochs": len(epochs)}), flush=True)
    out = geometry.remote(name, spec, meta)
    dest = HERE / "results" / out["file"]
    dest.open("xb").write(out["bytes"])
    print(json.dumps({"record": str(dest.relative_to(base.REPO)), "sha256": out["sha256"], "volume": f"{SOLVOL}:/xfer/projection_why/{name}/"}), flush=True)
