"""The fix-full training-recipe cells (RULE.md Amendment 14), Qwen3-32B last token (output_end), L48 and L63:
(a) e1 vs real honest, refit as the gate against the main job's stored cell (a); (c″) fix-full sim attack vs real honest;
(b″) fix-full sim attack vs fix-full sim honest, each with a 100-shuffle refit null; with NEWLINE filled in (Amendment 16),
also (c_N) and (b_N) on the newline-only variant, and with F9 filled in (Amendment 18), (c_F9) and (b_F9) on "Full + F9",
all in the same job.
  pack (CPU, one container per layer): the main job's packed matrix (sha256 checked), plus a fix-full copy of every sim
      attack and sim honest run, read from q2's dumps. A run missing from a dump keeps its original reading, and is
      counted.
  fit (H100, one container per layer), twice: cell (a) alone as the gate, then (c″) and (b″) with batched shuffles.
The dump names are filled in from q2's message (FIXFULL below); the job refuses to run while any is "TBD". No text
leaves the Volumes, and failures are re-raised by type only (a Gate's own message is numbers only).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run \\
      experiments/2026-10-13_q3_sce_reply_probe/run_sce_fixfull.py
"""

import hashlib
import json
import os
import subprocess
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q3sce_fixfull_kill-2026-11-13"
WORK = "probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06"
# q2's fix-full dumps: (Volume, extraction run, meta arm) per original arm. Filled in when q2 sends the names.
FX_VOL, FX_RUN = "probe-shift_aswin-surya_sub-la-activations", "2026-10-13_fixfull_qwen3-32b_l48-63_v2"  # q2, relayed by d9 10-13 ~15:20
FIXFULL = {"volume": FX_VOL, "sim-attack": (FX_RUN, "sim-attack-fx-full"), "sim-honest": (FX_RUN, "sim-honest-fx-full"),
           "n": {"sim-attack": 939, "sim-honest": 437}}  # how many runs each dump holds, as q2 reports them
# Amendment 16's newline-only variant N: None until q2 sends its names; then the same shape as FIXFULL. It runs in the
# same job (one model load) when both sets of names are in.
NEWLINE = {"volume": FX_VOL, "sim-attack": (FX_RUN, "sim-attack-fx-NL"), "sim-honest": (FX_RUN, "sim-honest-fx-NL"),
           "n": {"sim-attack": 939, "sim-honest": 437}}
# Amendment 18's "Full + F9": None until q2 sends its names; then FIXFULL's shape plus "mode": "complete" (every run) or
# "changed" (changed runs only: the rest take their fix-full reading, and the counts must total 939 and 437).
F9_MAPPING = HERE / "inputs/f9_mapping_qwen3_2026-10-13_151934.json"  # q2's, @ b6530d33 on aswin/q2-fixfull
F9_MAPPING_SHA256 = "99fee1bd43c7c6232feb1ab4e8200c6cb8ee56f8350c8cd76fd5bc3845aa4b4c"
F9 = {"volume": FX_VOL, "sim-attack": ("2026-10-13_fixfull_f9_qwen3-32b_l48-63_v1", "sim-attack-fx-fullF9"),
      "sim-honest": ("2026-10-13_fixfull_f9_qwen3-32b_l48-63_v1", "sim-honest-fx-fullF9"), "n": {"sim-attack": 36, "sim-honest": 45},
      "mode": "changed"}
MAIN = "sce/2026-10-13_121538"  # the main job's packed features and stored cells (app ap-JXedkWSJhaFEow9OOu24pm)
MAIN_SHA256 = {48: "084dcf1cf056ab6211bef45a13af99ada3ef7b51da577867f9b403e1cd741250", 63: "42b3236e47fe59461f931cb927e66a741af94c878b679c35b9af8b767f9d3a0e"}
MAIN_ROWS, N_MAIN = 135963, 5620
R36 = REPO / "experiments/2026-10-06_q3_attack_detector/results"
LAYERS, H = (48, 63), 5120
H_, E1, SA, SH = "real-honest", "real-attack-e1", "sim-attack", "sim-honest"
# new arm -> (its original arm, mount, spec, the arm whose reading fills a run missing from the dump: None = the original)
COPY = {"sim-attack-fullfix": (SA, "acts", FIXFULL, None), "sim-honest-fullfix": (SH, "acts", FIXFULL, None)}
CELLS = {"a": (E1, H_), "c_full": ("sim-attack-fullfix", H_), "b_full": ("sim-attack-fullfix", "sim-honest-fullfix")}  # (1, 0)
if NEWLINE is not None:
    mn = "acts" if NEWLINE["volume"] == FIXFULL["volume"] else "acts_n"  # one mount per Volume
    COPY |= {"sim-attack-newline": (SA, mn, NEWLINE, None), "sim-honest-newline": (SH, mn, NEWLINE, None)}
    CELLS |= {"c_N": ("sim-attack-newline", H_), "b_N": ("sim-attack-newline", "sim-honest-newline")}
if F9 is not None:
    fill = "sim-attack-fullfix", "sim-honest-fullfix"
    m9 = "acts" if F9["volume"] == FIXFULL["volume"] else "acts_f9"
    COPY |= {"sim-attack-f9": (SA, m9, F9, fill[0] if F9.get("mode") == "changed" else None),
             "sim-honest-f9": (SH, m9, F9, fill[1] if F9.get("mode") == "changed" else None)}
    CELLS |= {"c_F9": ("sim-attack-f9", H_), "b_F9": ("sim-attack-f9", "sim-honest-f9")}
C, MAX_ITER, N_NULL, ALPHAS = 0.01, 1000, 100, (0.01, 0.05)
PAIRS = [(f, v) for f in range(5) for v in range(5) if v != f]
ROOTS = {"acts": Path("/acts"), "acts_n": Path("/acts_n"), "acts_f9": Path("/acts_f9"), "work": Path("/work")}
TBD = "TBD" in json.dumps([FIXFULL, NEWLINE, F9])
if F9 is not None and F9.get("mode") not in ("complete", "changed"):
    raise SystemExit("F9's mode must be 'complete' or 'changed'")
if TBD and not os.environ.get("SCE_FIXFULL_DRYRUN"):  # before any app, image or Volume is built: nothing reaches Modal
    raise SystemExit("the fix-full dump names are not filled in yet")

app = modal.App(APP)
work_vol = modal.Volume.from_name(WORK, version=2)
lease_store = modal.Dict.from_name(f"{APP}-leases", create_if_missing=True)  # writers' leases, 0d's duplicate-writer rule
BEAT, FRESH, WAIT = 15, 45, 60  # seconds: heartbeat, a lease counts as fresh, how long to watch a fresh lease advance
KILL_AFTER = 35  # seconds without a successful beat before a writer kills itself (< FRESH: it dies before its lease looks stale)
fix_vol = modal.Volume.from_name(WORK if TBD else FIXFULL["volume"], version=2)  # a placeholder mount until the names are in
mounts = {"/acts": fix_vol.read_only(), "/work": work_vol} | (
    {} if NEWLINE is None or TBD or NEWLINE["volume"] == FIXFULL["volume"] else {"/acts_n": modal.Volume.from_name(NEWLINE["volume"], version=2).read_only()}) | (
    {} if F9 is None or TBD or F9["volume"] == FIXFULL["volume"] else {"/acts_f9": modal.Volume.from_name(F9["volume"], version=2).read_only()})
image = (modal.Image.debian_slim(python_version="3.12")  # #36's image, unchanged
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


class Lease:
    """A writer's lease on one input (0d's duplicate-writer rule): a heartbeat every BEAT s in a modal.Dict, keyed by a
    unique id per execution. A writer that can't beat for KILL_AFTER s (< FRESH) kills its own process, through a watchdog
    thread that doesn't wait on the write, so a stale lease always
    means its writer is gone. On entry, another lease on the same input that is fresh AND still advancing over WAIT s
    means two live writers: a sticky conflict marker is written, then Gate (stop). A stale lease (a preempted
    predecessor) is an alert: recorded, the run goes on. A clean exit writes a done marker. The client then confirms
    exactly one completed call per input, no conflict marker, and no lease still beating (final_check)."""

    def __init__(self, store, stamp, input_id):
        self.store, self.prefix = store, f"{stamp}|{input_id}|"
        self.key = self.prefix + uuid.uuid4().hex[:16]
        self.alerts, self.stop = [], threading.Event()

    def _beat(self):
        while not self.stop.wait(BEAT):
            pre = time.monotonic()  # the deadline runs from before the write, like the stored stamp
            try:
                self.store[self.key] = time.time()
                self.last_ok = pre
            except Exception:  # noqa: BLE001 — retried next beat; the watchdog enforces the deadline
                pass

    def _watchdog(self):  # independent of the write, which may block: a writer that can't beat must not live on
        while not self.stop.wait(1 if BEAT >= 1 else BEAT / 4):
            if time.monotonic() - self.last_ok > KILL_AFTER:
                os._exit(3)

    def __enter__(self):
        pre = time.monotonic()
        self.store[self.key] = time.time()
        self.last_ok = pre
        if time.monotonic() - pre > KILL_AFTER:  # the first write took so long that the lease may already look stale
            raise Gate("lease acquisition slower than the kill deadline")
        threading.Thread(target=self._beat, daemon=True).start()
        threading.Thread(target=self._watchdog, daemon=True).start()
        others = {k: self.store[k] for k in list(self.store.keys()) if k.startswith(self.prefix) and k != self.key}
        fresh = {k: t for k, t in others.items() if time.time() - t < FRESH}
        if fresh:
            time.sleep(WAIT)
            if any(self.store[k] > t for k, t in fresh.items()):
                self.stop.set()
                self.store["conflict|" + self.key] = time.time()  # sticky: seen by the client even if this error is dropped
                raise Gate(f"two live writers on one input: {len(fresh) + 1} fresh, advancing leases")
        self.alerts = sorted(k[len(self.prefix):] for k in others)  # preempted predecessors: their leases stopped
        if time.monotonic() - self.last_ok > KILL_AFTER:  # never start the work behind a lease that may look stale
            self.stop.set()
            raise Gate("lease not renewed within the kill deadline before the work started")
        return self

    def __exit__(self, et, ev, tb):
        self.stop.set()
        if et is None:
            self.store["done|" + self.key] = time.time()


def final_check(store, stamp, expected):
    """After every call returned: wait out FRESH + WAIT so a duplicate still running would show a fresh lease, then
    require exactly one done marker per expected input, no conflict marker and no fresh lease without a done marker."""
    time.sleep(FRESH + WAIT)
    keys = list(store.keys())
    done = Counter(k.split("|")[2] for k in keys if k.startswith(f"done|{stamp}|"))
    conflicts = [k for k in keys if k.startswith(f"conflict|{stamp}|")]
    live = [k for k in keys if k.startswith(f"{stamp}|") and "done|" + k not in keys and time.time() - store[k] < FRESH]
    ok = set(done) == set(expected) and all(v == 1 for v in done.values()) and not conflicts and not live
    return ok, {"done_markers": dict(done), "conflict_markers": len(conflicts), "live_leases_at_end": len(live)}


def leased(input_id, fn, stamp, *a):
    with Lease(lease_store, stamp, input_id) as lease:
        out = fn(stamp, *a)
    return out, lease.alerts


@app.function(image=image, cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0, volumes=mounts)
def pack(stamp: str, layer: int, runs: list[dict], f9_keys: dict | None = None) -> dict:
    out, alerts = sanitized(leased)(f"pack:{layer}", _pack, stamp, layer, runs, f9_keys)
    return out | {"lease_alerts": alerts}


def _pack(stamp, layer, runs, f9_keys=None):
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    base = np.load(ROOTS["work"] / MAIN / f"L{layer}.npy")
    if base.shape[0] != MAIN_ROWS or hashlib.sha256(base.tobytes()).hexdigest() != MAIN_SHA256[layer]:
        raise Gate(f"L{layer}: the main job's packed features differ")
    base_run = np.load(ROOTS["work"] / MAIN / f"rows_L{layer}.npz")["run"].astype(np.int64)
    index = {(r["arm"], r["key"]): i for i, r in enumerate(runs[:N_MAIN])}
    metas, src = {}, []
    for new, (orig, mount, spec, _) in COPY.items():
        ext, arm = spec[orig]
        out = {}
        for p in (ROOTS[mount] / ext / "meta" / arm).glob("*.json"):
            m = json.loads(p.read_text())
            if m["key"] in out:
                raise Gate(f"{new}: a run extracted twice")
            out[m["key"]] = m
        if len(out) != spec["n"][orig]:  # a missing or partial dump must not fall back silently to the originals
            raise Gate(f"{new}: {len(out)} runs in the dump, {spec['n'][orig]} expected")
        if spec is F9 and f9_keys is not None and set(out) != set(f9_keys[orig]):  # q2's mapping names exactly these overrides
            raise Gate(f"{new}: the dump's keys differ from the F9 mapping's ({len(out)} vs {len(f9_keys[orig])})")
        metas[new] = (mount, ext, out)
        if not set(out) <= {r["key"] for r in runs[:N_MAIN] if r["arm"] == orig}:
            raise Gate(f"{new}: dump keys outside the main job's runs")
    rows_new, run_new = [], []  # per new run: (mount, ext, chunk, row) of its kept output_end rows, or the original's rows
    for i, r in enumerate(runs[N_MAIN:], start=N_MAIN):
        mount, ext, ms = metas[r["arm"]]
        m = ms.get(r["key"])
        fill = COPY[r["arm"]][3]
        if m is None and fill is not None and r["key"] in metas[fill][2]:  # changed-only F9: the fix-full reading, by key
            mount, ext, ms = metas[fill]
            m = ms[r["key"]]
            rows_new += [(mount, ext, m["chunk"], m["row_start"] + p["row"]) for p in m["positions"] if p["kind"] == "output_end" and p["kept"]]
            src.append("filled from fix-full")
        elif m is None:  # unchanged: the original reading's rows from the main matrix
            j = index[(COPY[r["arm"]][0], r["key"])]
            rows_new += [("main", None, None, int(k)) for k in np.flatnonzero(base_run == j)]
            src.append("original")
        else:
            rows_new += [(mount, ext, m["chunk"], m["row_start"] + p["row"]) for p in m["positions"] if p["kind"] == "output_end" and p["kept"]]
            src.append("repaired")
        run_new += [i] * (len(rows_new) - len(run_new))
    Xn = np.zeros((len(rows_new), H), np.float16)
    groups = {}
    for j, (mount, ext, ch, row) in enumerate(rows_new):
        groups.setdefault((mount, ext, ch), []).append((j, row))

    def load(item):
        (mount, ext, ch), its = item
        block = base if mount == "main" else np.load(ROOTS[mount] / ext / "by_layer" / f"L{layer}" / f"{ch}.npy", mmap_mode="r")
        Xn[[j for j, _ in its]] = block[[r for _, r in its]]

    for new, (orig, _, spec, fill) in COPY.items():  # changed-only F9: its dump plus fix-full fills must cover every run
        got = Counter(s for s, r in zip(src, runs[N_MAIN:]) if r["arm"] == new)
        total = 939 if orig == SA else 437
        if fill is not None and got["repaired"] + got["filled from fix-full"] != total:
            raise Gate(f"{new}: {got['repaired']} in the dump + {got['filled from fix-full']} filled from fix-full != {total}")
        if spec is F9 and fill is None and got["repaired"] != total:
            raise Gate(f"{new}: complete delivery holds {got['repaired']} runs, {total} expected")
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(load, groups.items()))
    if not np.isfinite(Xn).all():
        raise Gate(f"L{layer}: non-finite features")
    X = np.concatenate([base, Xn])
    run = np.concatenate([base_run, np.array(run_new, np.int64)]).astype(np.int32)
    d = ROOTS["work"] / "sce_fixfull" / stamp
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"L{layer}.tmp", "wb") as fh:
        np.save(fh, X)
    os.replace(d / f"L{layer}.tmp", d / f"L{layer}.npy")
    np.savez(d / f"rows_L{layer}.npz", run=run)
    work_vol.commit()
    return {"layer": layer, "rows": int(len(run)), "new_rows": len(rows_new), "source": src,
            "copies_by_source": {a: dict(Counter(s for s, r in zip(src, runs[N_MAIN:]) if r["arm"] == a)) for a in COPY},
            "features_sha256": hashlib.sha256(X.tobytes()).hexdigest(), "seconds": round(time.time() - t0)}


@app.function(image=image, gpu="H100!", cpu=(8, 8), memory=(32768, 32768), timeout=1800, retries=0, volumes={"/work": work_vol})
def fit(stamp: str, layer: int, runs: list[dict], cells: list[str]) -> dict:
    out, alerts = sanitized(leased)(f"fit:{layer}:{'+'.join(cells)}", _fit, stamp, layer, runs, cells)
    out["info"]["lease_alerts"] = alerts
    return out


def _fit(stamp, layer, runs, cells):
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.attack_eval import pool, shuffle_within_task

    t0 = time.time()
    work_vol.reload()
    d = ROOTS["work"] / "sce_fixfull" / stamp
    r_run = np.load(d / f"rows_L{layer}.npz")["run"].astype(np.int64)
    X = torch.from_numpy(np.load(d / f"L{layer}.npy")).cuda().float()
    n = len(runs)
    arm, task = np.array([r["arm"] for r in runs]), np.array([r["main_task"] for r in runs])
    fold, honest = np.array([r["fold"] for r in runs]), np.array([r["arm"] == H_ for r in runs])

    def fit_score(train_runs, score_runs, Y):
        tr, te = np.flatnonzero(train_runs[r_run]), np.flatnonzero(score_runs[r_run])
        Xtr = X[torch.from_numpy(tr).cuda()]
        mean, std = Xtr.mean(0), Xtr.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        W, b, n_eval = tp.fit((Xtr - mean) / std, torch.from_numpy(Y[r_run[tr]]).cuda().float(), C, MAX_ITER)
        del Xtr
        s = tp.decision((X[torch.from_numpy(te).cuda()] - mean) / std, W, b).double().cpu().numpy()
        if not np.isfinite(s).all():
            raise Gate(f"L{layer}: non-finite output scores")
        S = pool(r_run[te], s, n, "mean")
        S[~score_runs] = np.nan
        return S, int(n_eval)

    out, info = {}, {"layer": layer, "cells": {}}
    for cell in cells:
        tc = time.time()
        pos, neg = CELLS[cell]
        y = np.where(arm == pos, 1.0, np.where(arm == neg, 0.0, np.nan))
        trainable = ~np.isnan(y)
        rng = np.random.default_rng(int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16))
        variants = {"real": y[:, None]} | ({} if cell == "a" else
                                           {"null": np.stack([shuffle_within_task(y, task, trainable, rng) for _ in range(N_NULL)], 1)})
        a_ = {"nested_real": np.full((5, 5, n), np.nan, np.float32)}
        if cell != "a":
            a_ |= {f"null_caught_{a:g}": np.zeros((n, N_NULL), np.uint8) for a in ALPHAS}
            a_["y_null"] = np.nan_to_num(variants["null"], nan=-1).astype(np.int8)
        evals = {k: [] for k in variants}
        for f, v in PAIRS:
            te = (fold == f) | (fold == v)
            for k, Y in variants.items():
                S, ne = fit_score(trainable & ~te, te, Y)
                evals[k].append(ne)
                if k == "real":
                    a_["nested_real"][f, v] = S[:, 0]
                    continue
                hv = honest & (fold == v) & np.isfinite(S[:, 0])
                Sh = np.sort(S[hv], 0)[::-1]
                for a in ALPHAS:
                    a_[f"null_caught_{a:g}"][fold == f] += (S[fold == f] > Sh[int(np.floor(a * hv.sum()))]).astype(np.uint8)
        info["cells"][cell] = {"train_runs": {"1": int((y == 1).sum()), "0": int((y == 0).sum())},
                               "max_evals": {k: max(e) for k, e in evals.items()},
                               "fits_at_cap": {k: sum(x >= MAX_ITER for x in e) for k, e in evals.items()}, "seconds": round(time.time() - tc)}
        if cell == "a":  # the gate: the main job's stored cell (a), on its runs
            ref = np.load(ROOTS["work"] / MAIN / f"cell_a_L{layer}.npz")["nested_real"]
            mine = a_["nested_real"][..., :N_MAIN]
            same = bool(np.array_equal(np.isnan(ref), np.isnan(mine)))
            info["gate"] = {"same_missing": same, "max_abs_diff": float(np.nanmax(np.abs(ref - mine)))}
            if not (same and info["gate"]["max_abs_diff"] <= 1e-3):
                raise Gate(f"L{layer} cell (a): {info['gate']}")
        out[cell] = a_
    info["seconds"] = round(time.time() - t0)
    return {"info": info, "arrays": out}


def build_runs() -> list[dict]:
    """The main job's 5,620 runs (run_sce_probe.build_runs at f6d50a21), then a fix-full copy of each sim attack and
    sim honest run (and, with NEWLINE, an N copy of each)."""
    keep = ("arm", "key", "cell", "main_task", "fold")
    base = [{k: r[k] for k in keep} | {"in_base": True} for r in json.loads((R36 / "run_scores_2026-10-06_181548/runs.json").read_text())]
    sh = [{k: r[k] for k in keep} | {"in_base": False}
          for r in json.loads((R36 / "sim_honest_scores_2026-10-07_124717/runs.json").read_text()) if r["arm"] == SH]
    main_runs = (base + sh + [r | {"arm": "sim-attack-fixed", "in_base": False} for r in base if r["arm"] == SA]
                 + [r | {"arm": "sim-honest-edited", "in_base": False} for r in sh])
    return main_runs + [r | {"arm": new, "in_base": False} for new, (orig, *_) in COPY.items() for r in main_runs[:3807 + 437] if r["arm"] == orig]


@app.local_entrypoint()
def main():
    if TBD:
        raise SystemExit("the fix-full dump names are not filled in yet")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head, t0 = git("rev-parse", "HEAD"), time.time()
    runs = build_runs()
    if len(runs) != N_MAIN + (939 + 437) * (len(COPY) // 2) or len({(r["arm"], r["key"]) for r in runs}) != len(runs):
        raise SystemExit("unexpected runs")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(f"stamp {stamp}, git {head}", flush=True)
    f9_keys = None
    if F9 is not None:  # q2's F9 mapping, sha256 checked: its override keys must equal the F9 dump's
        if hashlib.sha256(F9_MAPPING.read_bytes()).hexdigest() != F9_MAPPING_SHA256:
            raise SystemExit("the F9 mapping file differs from q2's")
        mp = json.loads(F9_MAPPING.read_text())["arms"]
        f9_keys = {SA: mp["full + F9, attack"]["override keys"], SH: mp["full + F9, honest"]["override keys"]}
        if {a: len(k) for a, k in f9_keys.items()} != {SA: F9["n"][SA], SH: F9["n"][SH]}:
            raise SystemExit("the F9 mapping's override counts differ from F9's")
    packs = list(pack.starmap([(stamp, l, runs, f9_keys) for l in LAYERS]))
    print(json.dumps([{k: v for k, v in p.items() if k != "source"} for p in packs]), flush=True)
    fits = list(fit.starmap([(stamp, l, runs, ["a"]) for l in LAYERS]))  # the gate first
    rest = [c for c in CELLS if c != "a"]
    fits += list(fit.starmap([(stamp, l, runs, rest) for l in LAYERS]))
    # 0d's rule: exactly one completed call per input, no conflict, nothing still writing; preempted predecessors are alerts
    expected = {f"pack:{l}" for l in LAYERS} | {f"fit:{l}:{c}" for l in LAYERS for c in ("a", "+".join(rest))}
    one_call, writers = final_check(lease_store, stamp, expected)
    (HERE / "results").mkdir(exist_ok=True)
    npz = HERE / "results" / f"sce_fixfull_scores_{stamp}.npz"  # git-ignored (*.npz): archived to S3 with a manifest
    src = ["original"] * N_MAIN + packs[0]["source"]
    with open(npz, "xb") as fh:
        np.savez(fh, **{f"L{x['info']['layer']}__{c}__{k}": v for x in fits for c, arrs in x["arrays"].items() for k, v in arrs.items()},
                 **{f"runs__{k}": np.array([r[k] for r in runs]) for k in ("arm", "key", "cell", "main_task", "fold", "in_base")},
                 runs__repaired=np.array([s == "repaired" for s in src]))
    rec = {"stamp": stamp, "git": head, "app": APP, "rule": "RULE.md Amendments 14, 16 and 18", "fixfull": FIXFULL, "newline": NEWLINE, "f9": F9,
           "settings": {"layers": LAYERS, "cells": CELLS, "C": C, "max_iter": MAX_ITER, "n_null": N_NULL, "alphas": ALPHAS, "main": MAIN},
           "pack": [{k: v for k, v in p.items() if k != "source"} for p in packs], "fit": [x["info"] for x in fits],
           "one_completed_call_per_input": one_call, "writers": writers,
           "scores_file": npz.name, "scores_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(), "client_seconds": round(time.time() - t0)}
    with (HERE / "results" / f"sce_fixfull_run_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("stamp", "scores_file", "scores_sha256", "client_seconds", "one_completed_call_per_input")}), flush=True)
    if not one_call:  # recorded above; a nonzero exit sends the launcher into its stop-and-confirm cleanup
        raise SystemExit(f"duplicate-writer check failed: {writers}")
