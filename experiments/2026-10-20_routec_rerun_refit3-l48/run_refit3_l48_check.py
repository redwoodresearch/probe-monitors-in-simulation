"""refit3-l48 (route C reruns 2026-10-20, item 3 step D): VALUE check of refit3's command|L48 attack-detector cell.

Pre-statement: PRESTATED_2026-10-20_125239.md (sha256 0f1a966f…), PRESTATED_AMEND1_2026-10-20_130336.md (sha256 0a998c40…) and
PRESTATED_AMEND2_2026-10-20_130749.md (sha256 731ae87c…), PRESTATED_AMEND3_2026-10-20_131434.md (sha256 38e644f3…) and
PRESTATED_AMEND4_2026-10-20_134050.md (sha256 4586d45c…: G3 finiteness on gathered rows only; slot 23's 4 printed numbers gate, all 6 reported).
Record: attack_all_refit3_2026-10-16_141813.json @ 80d75b27e. The VERDICT cell is command|L48; reply|L48 rides the same chunk reads as a
DIAGNOSTIC (AMEND1 A1) and never enters `pass` or the exit code.
- G1-G4: input gates (run set, fixed-token skip list, rows per run and chunks, stored weights), by key against the record.
- D3: replay of the 10-16 stored weights on independently rebuilt rows (scoring and row mapping), tolerance 1e-9.
- D2: independent refit (damped Newton in float64 from zero, certified), tolerance 1e-4 on run scores and run-level AUROC.
- Diagnostics: STATS_MATCH, STORED_CERT, RESID_EXPLAINED (AMEND1 A5), MY_FIT_ERR, AUROC_DEF, the D3 rounding bound.
- Controls, each predicted to FAIL: D2 against the L63 cell's scores; D3 with W[0] += 1e-3; D3 with the d_run2_vs_run1 npz; a toy self-test.
- `reading.class` (AMEND1 A6), in this precedence: STOP_D3 (D3 MISMATCH); NO_VERDICT (D3 bound exceeded, or a fold of mine
  uncertified); STOP_DEFINITION_GAP_OR_BUG (STATS_MATCH, STORED_CERT or RESID_EXPLAINED false); REPRODUCED (D2 MATCH); else
  CERTIFIED_FIT_WITHIN_DELTA_HAT. `reading.ruling_09` (AMEND2 B2, 09's pre-stated ruling): CERTIFIED_FIT_WITHIN_DELTA_HAT is
  "PASS (value half)" iff rank_identical (B1) or max|dhat| <= 1e-3, else UNRESOLVED. Exit 0 iff ruling_09 is REPRODUCED or
  PASS (value half) and all 3 controls failed as predicted.
Values stay on Modal: the container compares by key and writes one results JSON to the output Volume (create-only). stdout gets one
structured line, {"refit3l48": ...}, holding booleans, counts, keys and shas only.
Not the producer's code: every rule it follows is quoted and labelled in the pre-statement §3.

Run from the root of the committed worktree (M = the named modal binary):
  "$M" run experiments/2026-10-20_routec_rerun_refit3-l48/run_refit3_l48_check.py::main --test       # toy self-test only (CPU)
  "$M" run --detach experiments/2026-10-20_routec_rerun_refit3-l48/run_refit3_l48_check.py::main > <fresh log> 2>&1
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-rerun_refit3l48_kill-2026-11-19"
ACTS_VOL = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
W_VOL = "probe-shift_aswin-surya_sub-cmdsimdet_uncutall_w_kill-2026-11-14"
HF_VOL = "probe-shift_aswin-surya_hf-cache"
LAYER, HDIM = 48, 5120
SETS, WORLDS = ("S_old", "S_new", "S_newmains", "S_paper493"), ("run1", "run2", "sim")
REC = ("80d75b27e9ac00127b44d052a9fa1aaa7da47d47", "experiments/2026-10-16_attack_refit3/results/attack_all_refit3_2026-10-16_141813.json",
       "df7171a98d82d28ae86653480be922d978fc36104c63f622b3a69fb43a074eac")
COHORT = ("adbc791b690a831870bcf223e09550aef5dfb6bd", "experiments/2026-10-15_fpr_sweep_507_prep/results/exclusions_2026-10-16_033635.json",
          "4511f6c928c91f10dc2074a5d06bac4ab47ff57dca178f57fe14f8c106353477")
FOLDS = ("ec56720adde66baabe642a7fc842172c83737679", "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json",
         "c2fc5d40eb84a34a64cbb9ce7155572d1414f12ea1fbc2ec05708bd73061e299")
_CFG = "experiments/2026-10-16_attack_refit3/config/"
CONFIGS = [("S_old", _CFG + "inputs_all_S_old_v1.json", "758d274c0aafcd60cc6bfe7b5c4d150e489b633584529a99c46cecacc6854665"),
           ("S_new", _CFG + "inputs_all_S_new_v1.json", "661427e8755759083c96c2a587e3e0e2eb85206de7dc1e8fce9f879a4e18c493"),
           ("S_newmains", _CFG + "inputs_all_S_newmains_v1.json", "a05be412901d8a3c69b50ca6ad07b3365bcf8e5e94b8bb286fe5acd81a22630b"),
           ("S_paper493", _CFG + "inputs_all_S_paper493_w1_v1.json", "4a6ec898c8f78bfbae6c4c2189e31d35d80e12e1cd2d263c00006b81bf24152e"),
           ("S_paper493", _CFG + "inputs_all_S_paper493_w23_v1.json", "61adc9e9b8fc94a4ed92cf19f12ecdfcc99ba2903d12df4c09a6f028e3f20a19")]
WDIR = "xfer/cmd_sim_detector_uncut/2026-10-16_042630"
VERDICT_CELL = "command|L48"
CELLS = {  # readout, control cell (PC-D2), stored weights {problem: (path, sha256)}; reply|L48 is a diagnostic only (AMEND1 A1)
    "command|L48": {"readout": "command", "control": "command|L63",
                    "npz": {"c": (f"{WDIR}/w_command_L48__c_simatk_vs_run1.npz", "bd7158998c5016a7c71c6d16ff18a1ca33c4566750ee14177968252375e1fe02"),
                            "d": (f"{WDIR}/w_command_L48__d_run2_vs_run1.npz", "3df9f0d936eb26f29bf2757b4e7e280dd7e89eee84632246d485828dcfbf7c67")}},
    "reply|L48": {"readout": "reply", "control": "reply|L63",
                  "npz": {"c": (f"{WDIR}/w_reply_L48__c_simatk_vs_run1.npz", "f050111d45a3179488a14f1bf9d475a1fa589acd1ef600a30dc58af88fe7c2af"),
                          "d": (f"{WDIR}/w_reply_L48__d_run2_vs_run1.npz", "22d39a89fb29d99f00110357ced11003f7da5bc245966c7674a1363bf1e64e44")}}}
VOCAB = "hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654/vocab.json"
FIXED = ("0", "n")  # the fixed call tokens (648f33e61 RULE L24; p2 RULE L149)
C, CERT_BAR, NEWTON_TOL, MAX_NEWTON, CHUNK = 0.01, 1e-6, 1e-11, 100, 16384
TOL_D2, TOL_D3, TOL_DIAG, SEED = 1e-4, 1e-9, 1e-9, 20261020
L48_SWITCH = [  # AMEND1 A2 and AMEND2 B3
    {"path": "experiments/2026-10-14_priority1_spec_09/GATES_routec_2026-10-17.md", "commit": "f1d0f437d55612f7ee7bda21c77330105aff6d7a",
     "file_sha256": "d170cc75b3a2fb79f15dfa833ea6ed305d2fc25dd72d76dbfd8d17d7f7932dbb", "line": 638,
     "heading": "detector_vs_drop's detector: commands L63 → commands L48 (author's choice, POST HOC); Fig 2 / App E at L48 (disclosure) "
                "(09, 2026-10-20 12:56 PDT by date)"},
    {"path": "experiments/2026-10-14_priority1_spec_09/GATES_routec_2026-10-17.md", "commit": "11879c4407916fb43f6585972f9cf5d749387b1f",
     "file_sha256": "aac47a242ceccb2f0713520bf2da10bdaf5544fc3ecf2693abf491ac4fec1a67", "lines": "645-649",
     "heading": "Addition to \"detector_vs_drop's detector … Fig 2 / App E at L48\" (09, 2026-10-20 13:02 PDT by date; step D reviewer's "
                "evidence via 2e, checked by 09 by key)",
     "fig2_render": {"commit": "19d9b1bd2f96cc1479151929061c3299fe05629a",
                     "source": "experiments/2026-10-15_fpr_sweep_507_prep/results/fig_attack_roc_attack_v5_source.json",
                     "cells": ["reply|L48", "command|L48"]}}]
LABELS = {  # 09's labels (AMEND2 B2, AMEND3 C1)
    "pass_ranks": "x = the stored certified fit; exact optimum within max|Δ̂|; printed Spearman unaffected",
    "pass_slot": "x = the stored certified fit; exact optimum within max|Δ̂|; printed slot 23 values unchanged at print rounding "
                 "(recomputed from the refit x)",
    "reproduced_no_auroc": "x reproduced; refit3 run-level AUROC not compared (AUROC definition not identified)"}
# AMEND3 C3: slot 23 inputs. Stage D (item 3 §2 A) and the L48 detector_vs_drop record (item 3 step A, committed at step B)
STAGE_D = ("c22edd8d3cc1221225dec418ab00b40515d5ba43", "experiments/2026-10-15_fpr_sweep_507_prep/results/routec_records/pertask_full_2026-10-20_004115.json",
           "468655154f826ec1a93e08333eb802e68abea6db65a8e2f80a3e0f23465f36d9")
DETDROP = ("a66fa8d80e2fd56d78b3f4c5d533dd8480da81ee",
           "experiments/2026-10-15_fpr_sweep_507_prep/results/detector_vs_drop_2026-10-20_125606_ap-NuUKhGaNZaLhqdKopouezY.json",
           "a536b65f78a40582c4e9491866c566a2139db906279ae0100d2ca23eb357af2c")
SLOT_DET = {"command|L48": "commands L48", "reply|L48": "replies L48"}
SLOT_SEED, SLOT_D, SLOT_COMBO = [20261015, 41], 10000, "pooled_all_origins"

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3", "tzdata==2025.2")
acts = modal.Volume.from_name(ACTS_VOL, version=2).read_only()
wvol = modal.Volume.from_name(W_VOL, version=2).read_only()
hfvol = modal.Volume.from_name(HF_VOL).read_only()
out = modal.Volume.from_name(APP, version=2, create_if_missing=True)


class Stop(RuntimeError):
    """A failed gate inside the container: no D2/D3 verdict. The message holds booleans, counts and keys only."""


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- statistics (my code; definitions: PRESTATED §3)
def auroc(s, y) -> float:
    """Mann-Whitney AUROC, ties counted half (class 1 = y true)."""
    import numpy as np

    s, y = np.asarray(s, np.float64), np.asarray(y, bool)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def compare(mine: dict, rec: dict, tol: float) -> dict:
    """By key: MATCH iff the key sets are equal and every |difference| <= tol (non-finite counts as over). '_' fields hold values."""
    import math

    common = sorted(set(mine) & set(rec))
    d = {k: abs(mine[k] - rec[k]) for k in common}
    over = [k for k in common if not (math.isfinite(d[k]) and d[k] <= tol)]
    return {"keys_equal": set(mine) == set(rec), "n_mine": len(mine), "n_record": len(rec), "n_compared": len(common), "n_over_tol": len(over),
            "match": set(mine) == set(rec) and not over and bool(common), "over_keys_first10": over[:10],
            "_max_abs": max(d.values()) if d else None, "_over_keys": over}


def rank_swaps(rec: dict, mine: dict) -> dict:
    """AMEND2 B1: over the record's keys, the pairs (i < j) whose order differs, sign(rec_i - rec_j) != sign(mine_i - mine_j)
    (a tie on one side only counts). keys_equal is False if mine lacks a key or has an extra one."""
    import numpy as np

    ks = sorted(set(rec) & set(mine))
    a, b = np.array([rec[k] for k in ks]), np.array([mine[k] for k in ks])
    sw = np.triu(np.sign(a[:, None] - a[None, :]) != np.sign(b[:, None] - b[None, :]), 1)
    return {"n": len(ks), "keys_equal": set(rec) == set(mine), "swaps": int(sw.sum())}


def slot23(xs: dict, x2: dict, stage_d: dict, run_main: dict) -> dict:
    """AMEND3 C3. PORTED FROM THE PRODUCER: run_detector_vs_drop.py @ 76032545d (blob c51f7a59) analyse() L195-L274, restricted to
    sol_a_uncut C on the route C pooled rows (combo pooled_all_origins). xs = {"sim|<cell>": x}, x2 = {"run2|<cell>": x}. Returns
    sim_score_vs_drop {point, ci95, n_cells} (main-clustered bootstrap, 10,000 shared draws) and the null's {point, n_cells}."""
    import numpy as np

    def rows_of(og):  # L198-L203
        r_ = (stage_d["rows"].get("sol_a_uncut C") or {}).get(f"{og} (all-origins draws) [pooled: {SLOT_COMBO}]")
        if r_ is None:
            raise Stop(f"slot23: no Stage D row for {og}")
        return r_["per_task"]

    def rank_avg(a):  # L205-L212, verbatim
        o = np.argsort(a, kind="mergesort")
        s_ = a[o]
        st = np.flatnonzero(np.r_[True, s_[1:] != s_[:-1]])
        cnt = np.diff(np.r_[st, len(a)])
        r = np.empty(len(a))
        r[o] = np.repeat(st + (cnt - 1) / 2.0, cnt) + 1.0
        return r

    def spear(x, y):  # L214-L220, verbatim
        if len(x) < 3:
            return np.nan
        rx, ry = rank_avg(x), rank_avg(y)
        rx, ry = rx - rx.mean(), ry - ry.mean()
        den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        return float((rx * ry).sum() / den) if den > 0 else np.nan

    cells, seen = [], set()
    for og in SETS:  # L222-L229
        for r in rows_of(og):
            if r["arm"] == "sim":
                if r["cell"] in seen:
                    raise Stop(f"slot23: cell {r['cell']} has two sim rows")
                seen.add(r["cell"])
                cells.append(r["cell"])
    main = {}
    for c in cells:  # L230-L235
        ms = {run_main.get(f"{w}|{c}") for w in ("sim", "run1", "run2")} - {None}
        if len(ms) != 1:
            raise Stop(f"slot23: cell {c} has no single main")
        main[c] = ms.pop()
    mains = sorted(set(main.values()))
    mcol = {m: i for i, m in enumerate(mains)}
    W = np.random.default_rng(SLOT_SEED).multinomial(len(mains), np.full(len(mains), 1 / len(mains)), size=SLOT_D)  # L192, L236-L238
    mt = {}
    for og in SETS:  # L253-L260
        for r in rows_of(og):
            if r["arm"] in mt.get(r["cell"], {}):
                raise Stop(f"slot23: cell {r['cell']} has two {r['arm']} rows")
            mt.setdefault(r["cell"], {})[r["arm"]] = r["max_minus_tau"]
    if set(mt) != set(cells):
        raise Stop("slot23: Stage D cells differ from the cohort")

    def fin_(v):
        return v is not None and bool(np.isfinite(v))

    y_drop = {c: v["sim"] - (v["run1"] + v["run2"]) / 2 for c, v in mt.items() if all(fin_(v.get(a)) for a in ("sim", "run1", "run2"))}
    y_null = {c: v["run2"] - v["run1"] for c, v in mt.items() if all(fin_(v.get(a)) for a in ("run1", "run2"))}
    out = {"y_drop": y_drop, "y_null": y_null}
    for nm, xmap, ymap in (("sim_score_vs_drop", {c: xs.get(f"sim|{c}") for c in cells}, y_drop),
                           ("null_run2_score_vs_run2_minus_run1", {c: x2.get(f"run2|{c}") for c in cells}, y_null)):  # L269-L274
        cs = [c for c in cells if fin_(xmap[c]) and c in ymap]
        x, y = np.array([xmap[c] for c in cs]), np.array([ymap[c] for c in cs])
        cols = np.array([mcol[main[c]] for c in cs], int)  # boot(), L240-L248, for both analyses (the whole slot string)
        bt = np.array([spear(np.repeat(x, W[b, cols]), np.repeat(y, W[b, cols])) for b in range(SLOT_D)])
        fin = bt[np.isfinite(bt)]
        out[nm] = {"point": spear(x, y), "n_cells": len(cs),
                   "ci95": [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) >= 0.95 * SLOT_D else None}
    return out


def slot_vals(node: dict) -> list:
    """The 6 numbers of the two slot-23 strings: for sim_score_vs_drop and then the null, point, ci95 low, ci95 high.
    A withheld interval (ci95 None) gives None for both ends."""
    v = []
    for an in ("sim_score_vs_drop", "null_run2_score_vs_run2_minus_run1"):
        ci = node[an].get("ci95")
        v += [node[an]["point"], *(ci if isinstance(ci, list) and len(ci) == 2 else [None, None])]
    return v


def close(a, b, tol: float) -> bool:
    """Equal within tol; both None (withheld) or both NaN count as equal; one None is unequal; never raises."""
    import math

    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol


def k2s(v) -> str:
    """Print rounding of a slot-23 component: slot_table.py @ 26112da71 L81-L84 / L232-L233 (signed, 2 decimals; '-' shown as '−').
    A withheld end prints as 'withheld' on both sides (summ_str L111-L112), so it equals only another withheld end."""
    return "withheld" if v is None else f"{v:+.2f}".replace("-", "−")


def standardize_stats(XT, idx, dev):
    """Mean and population std (ddof 0) of the rows idx of the float16 matrix XT, in float64; std 0 -> 1 (PRESTATED §3)."""
    import torch

    n, s1 = len(idx), torch.zeros(XT.shape[1], dtype=torch.float64, device=dev)
    for i in range(0, n, CHUNK):
        s1 += XT[idx[i:i + CHUNK]].double().sum(0)
    mean = s1 / n
    s2 = torch.zeros_like(mean)
    for i in range(0, n, CHUNK):
        s2 += ((XT[idx[i:i + CHUNK]].double() - mean) ** 2).sum(0)
    std = (s2 / n).sqrt()
    return mean, torch.where(std > 0, std, torch.ones_like(std))


def objective(XT, idx, y, mean, std, w, b, need_grad=True, need_hess=False):
    """G = (sum_i logloss(x~_i.w + b, y_i) + |w|^2 / (2C)) / n with x~ = (x - mean) / std, intercept unpenalized; its gradient and Hessian."""
    import torch

    n, d = len(idx), w.shape[0]
    G = torch.zeros((), dtype=torch.float64, device=w.device)
    gw, gb = torch.zeros_like(w), torch.zeros_like(b)
    H = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=w.device) if need_hess else None
    for i in range(0, n, CHUNK):
        X = (XT[idx[i:i + CHUNK]].double() - mean) / std
        yc = y[i:i + CHUNK]
        z = X @ w + b
        G += (torch.logaddexp(torch.zeros_like(z), z) - yc * z).sum()
        if need_grad:
            p = torch.sigmoid(z)
            gw += X.T @ (p - yc)
            gb += (p - yc).sum()
            if need_hess:
                Xa = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], 1)
                H += Xa.T @ (Xa * (p * (1 - p))[:, None])
    G = (G + (w @ w) / (2 * C)) / n
    if not need_grad:
        return G, None, None
    g = torch.cat([(gw + w / C) / n, (gb / n).reshape(1)])
    if need_hess:
        H /= n
        H[:d, :d] += torch.eye(d, dtype=H.dtype, device=H.device) / (C * n)
    return G, g, H


def newton(XT, idx, y, mean, std, dev):
    """Damped Newton from w = 0, b = 0 until max|dG| <= NEWTON_TOL (or the rounding floor). Returns w, b, the certificate, delta = H^-1 g
    at the end (its score on a row is that row's remaining optimization error, to second order) and the Cholesky factor of the final H."""
    import torch

    d = XT.shape[1]
    w, b = torch.zeros(d, dtype=torch.float64, device=dev), torch.zeros((), dtype=torch.float64, device=dev)
    hist, steps = [], []
    for _ in range(MAX_NEWTON):
        G, g, H = objective(XT, idx, y, mean, std, w, b, need_hess=True)
        hist.append(float(g.abs().max()))
        if hist[-1] <= NEWTON_TOL or (len(hist) > 5 and hist[-1] < 1e-9 and hist[-1] > 0.5 * hist[-2]):
            break
        delta = torch.cholesky_solve(g[:, None], torch.linalg.cholesky(H))[:, 0]
        dec, t = float(g @ delta), 1.0
        while True:
            w1, b1 = w - t * delta[:d], b - t * delta[d]
            G1 = objective(XT, idx, y, mean, std, w1, b1, need_grad=False)[0]
            if float(G1) <= float(G) - 1e-4 * t * dec + 1e-13 * abs(float(G)) or t < 1e-8:
                break
            t /= 2
        steps.append(t)
        w, b = w1, b1
    G, g, H = objective(XT, idx, y, mean, std, w, b, need_hess=True)
    L = torch.linalg.cholesky(H)
    delta = torch.cholesky_solve(g[:, None], L)[:, 0]
    mg = float(g.abs().max())
    return w, b, {"n_iter": len(steps), "steps": steps, "max_grad_history": hist, "max_grad": mg, "certified": mg <= CERT_BAR}, delta, L


def row_scores(XT, idx, mean, std, w, b, with_abs=False):
    """Per-row logits x~.w + b in float64; with_abs also returns sum_j |w_j x~_j| + |b| (for the rounding bound)."""
    import torch

    s = torch.empty(len(idx), dtype=torch.float64, device=w.device)
    a = torch.empty_like(s) if with_abs else None
    for i in range(0, len(idx), CHUNK):
        X = (XT[idx[i:i + CHUNK]].double() - mean) / std
        s[i:i + CHUNK] = X @ w + b
        if with_abs:
            a[i:i + CHUNK] = X.abs() @ w.abs() + b.abs()
    return s, a


def run_means(s, seg, names):
    """Mean of the per-row scores s within each run (seg = run position per row) -> {name: float}."""
    import torch

    tot = torch.zeros(len(names), dtype=torch.float64, device=s.device).index_add_(0, seg, s)
    cnt = torch.bincount(seg, minlength=len(names)).to(torch.float64)
    return dict(zip(names, (tot / cnt).cpu().tolist()))


# ---------------------------------------------------------------- toy self-test (runs before any Volume read)
def selftest() -> dict:
    import numpy as np
    import torch

    rng = np.random.default_rng(SEED)
    n_runs, k, d = 40, 15, 12
    y_run = np.arange(n_runs) % 2
    fold_run = (np.arange(n_runs) // 2) % 5
    X = rng.normal(size=(n_runs * k, d)) + 0.4 * np.repeat(y_run, k)[:, None] * rng.normal(size=d)[None, :]
    X[:, 3] = 2.5  # a zero-std column
    XT = torch.from_numpy(X.astype(np.float16))
    seg = np.repeat(np.arange(n_runs), k)
    tr_runs, te_runs = np.flatnonzero(fold_run != 0), np.flatnonzero(fold_run == 0)
    tr = torch.from_numpy(np.flatnonzero(np.isin(seg, tr_runs)))
    te = torch.from_numpy(np.flatnonzero(np.isin(seg, te_runs)))
    ytr = torch.from_numpy(np.repeat(y_run, k)[tr.numpy()].astype(np.float64))
    mean, std = standardize_stats(XT, tr, "cpu")
    t = {"std_zero_to_one": bool(std[3] == 1.0) and bool((((XT[tr].double() - mean) / std)[:, 3] == 0).all())}
    w, b, cert, delta, L = newton(XT, tr, ytr, mean, std, "cpu")
    t["newton_tol"] = cert["max_grad"] <= NEWTON_TOL
    # an independent optimizer on the same objective (torch L-BFGS, float64, from zero)
    w2 = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    b2 = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w2, b2], lr=1, max_iter=20000, tolerance_grad=1e-13, tolerance_change=0, history_size=50,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        G = objective(XT, tr, ytr, mean, std, w2, b2, need_grad=False)[0]
        G.backward()
        return G

    opt.step(closure)
    s1 = row_scores(XT, te, mean, std, w, b)[0]
    s2 = row_scores(XT, te, mean, std, w2.detach(), b2.detach())[0]
    t["newton_vs_lbfgs_1e-7"] = float((s1 - s2).abs().max()) <= 1e-7
    G0 = float(objective(XT, tr, ytr, mean, std, w, b, need_grad=False)[0])
    v = torch.from_numpy(rng.normal(size=d))
    t["is_minimum"] = all(float(objective(XT, tr, ytr, mean, std, w + e * v, b, need_grad=False)[0]) > G0 for e in (1e-3, -1e-3))
    t["fit_err_tiny"] = float(row_scores(XT, te, mean, std, delta[:d], delta[d])[0].abs().max()) <= 1e-9
    # RESID_EXPLAINED's one-step prediction: from a perturbed point theta_p, x~.(theta_p - theta*) ~= x~.H^-1 g(theta_p)
    wp, bp = w + 1e-3 * v, b + 1e-3
    gp = objective(XT, tr, ytr, mean, std, wp, bp)[1]
    dp = torch.cholesky_solve(gp[:, None], L)[:, 0]
    true_gap = row_scores(XT, te, mean, std, wp, bp)[0] - s1
    pred_gap = row_scores(XT, te, mean, std, dp[:d], dp[d])[0]
    t["one_step_prediction_5pct"] = float((true_gap - pred_gap).abs().max()) <= max(1e-6, 0.05 * float(pred_gap.abs().max()))
    segte = torch.from_numpy(np.searchsorted(te_runs, seg[te.numpy()]))
    names = [f"r{j}" for j in te_runs]
    rm = run_means(s1, segte, names)
    t["run_means"] = all(abs(rm[f"r{j}"] - float(s1.numpy()[seg[te.numpy()] == j].mean())) <= 1e-12 for j in te_runs)
    t["auroc_cases"] = (auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0 and auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
                        and auroc([1, 1, 1, 1], [0, 0, 1, 1]) == 0.5 and auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75)
    sc, yy = np.round(rng.normal(size=60), 1), rng.integers(0, 2, 60)
    pairs = sum((1.0 if sc[i] > sc[j] else 0.5 if sc[i] == sc[j] else 0.0) for i in range(60) for j in range(60) if yy[i] and not yy[j])
    t["auroc_vs_pairs"] = bool(abs(auroc(sc, yy) - pairs / (int(yy.sum()) * (60 - int(yy.sum())))) <= 1e-12)  # Codex r2 P1: a native bool
    base = {f"k{i}": float(x) for i, x in enumerate(rng.normal(size=30))}
    moved = dict(base, k7=base["k7"] + 2 * TOL_D2)
    t["compare_identical_match"] = compare(base, dict(base), TOL_D2)["match"]
    cm = compare(moved, base, TOL_D2)
    t["compare_moved_mismatch_1"] = (not cm["match"]) and cm["n_over_tol"] == 1
    t["compare_missing_mismatch"] = not compare({k: v for k, v in base.items() if k != "k3"}, base, TOL_D2)["match"]
    r3 = {"a": 1.0, "b": 2.0, "c": 3.0}
    t["rank_swaps_cases"] = (rank_swaps(r3, dict(r3))["swaps"] == 0 and rank_swaps(r3, {"a": 1.0, "b": 3.0, "c": 2.0})["swaps"] == 1
                             and rank_swaps(r3, {"a": 3.0, "b": 2.0, "c": 1.0})["swaps"] == 3
                             and rank_swaps(r3, {"a": 1.0, "b": 2.0, "c": 2.0})["swaps"] == 1
                             and rank_swaps(r3, {"a": 1.5, "b": 2.5, "c": 9.0})["swaps"] == 0
                             and not rank_swaps(r3, {"a": 1.0, "b": 2.0})["keys_equal"])
    return {"tests": t, "all_pass": all(t.values())}


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=600, retries=0, scaledown_window=2)
def selftest_cpu() -> dict:
    return selftest()


# ---------------------------------------------------------------- inputs (PRESTATED §2-§3; built here, never taken from the record)
def cell_of(world: str, task: str) -> str:
    t = re.sub(r"_r2$", "", task) if world == "run2" else task
    return re.sub(r"_redraw[0-9]+$", "", t)


def build_runs(blobs: dict, pins: dict) -> dict:
    coh = json.loads(blobs[pins["cohort"]])
    if coh.get("pending") or coh["N_all_sets"]["final"] != sum(len(coh["final_cohort_keys"][s]["cells"]) for s in SETS):
        raise Stop("STOP cohort file: pending entries, or final count != cells listed")
    cohort = {s: set(coh["final_cohort_keys"][s]["cells"]) for s in SETS}
    F = json.loads(blobs[pins["folds"]])
    fold_of_main = {m: v[0] for m, v in F["by_main"].items()}
    main_of_cell = {}
    for c, p, _ in F["pins"]["cells"].values():
        for x in json.loads(blobs[f"{c}:{p}"])["cells"]:
            main_of_cell[x["key"]] = x["main"]
    c, p, _ = F["pins"]["sp493_facts"]
    for r in json.loads(blobs[f"{c}:{p}"])["rows"]:
        if main_of_cell.setdefault(r["key"], r["main"]) != r["main"]:
            raise Stop(f"STOP {r['key']}: two mains")
    runs, outside, allowed = [], 0, {}
    for s, path, _ in CONFIGS:
        cfg = json.loads(blobs[pins["configs"][path]])
        inv = json.loads(blobs[pins["inventories"][path]])
        if inv.get("status") != cfg["inventory_status"]:
            raise Stop(f"STOP {path}: inventory status")
        for root, spec in cfg["extraction_records"].items():
            ex = json.loads(blobs[f"{spec['commit']}:{spec['path']}"])
            ch = {cc["chunk"]: cc["rows"] for x in ex["shards"] for cc in x["chunks"]}
            if allowed.setdefault(root, ch) != ch:
                raise Stop(f"STOP {root}: two extraction records")
        for r in inv["runs"]:
            if r.get("task_set", r.get("set")) != cfg["task_set"] or r["world"] not in WORLDS:
                continue
            u = r["uncut_planned"]
            if u["volume"] != ACTS_VOL or u["staged_inputs_file"] != f"{r['arm']}.jsonl" or u["extraction_root"] not in cfg["extraction_records"]:
                raise Stop(f"STOP {r['key']}: inventory row not on the activation Volume, or an unpinned root")
            cell = cell_of(r["world"], r["task"])
            if F["cell_set"].get(cell) != s:
                raise Stop(f"STOP {r['key']}: cell is not a {s} cell of the fold list")
            if cell not in cohort[s]:
                outside += 1
                continue
            main = main_of_cell[cell]
            if main not in fold_of_main:
                raise Stop(f"STOP coverage: main {main} has no fold")
            runs.append({"name": f"{r['world']}|{cell}", "key": r["key"], "world": r["world"], "set": s, "main": main, "fold": fold_of_main[main],
                         "arm": r["arm"], "root": u["extraction_root"], "staged": u["staged_set"], "staged_sha256": u["staged_sha256"],
                         "full_len": r["full_len"]})
    per = {}
    for e in runs:
        per.setdefault((e["set"], e["name"].split("|", 1)[1]), []).append(e["world"])
    bad = [k for s in SETS for k in cohort[s] if sorted(per.get((s, k), [])) != ["run1", "run2", "sim"]]
    if bad or len({e["name"] for e in runs}) != len(runs) or len({e["key"] for e in runs}) != len(runs):
        raise Stop(f"STOP {len(bad)} cohort cells without exactly one run per world, or duplicate names/keys")
    return {"runs": sorted(runs, key=lambda e: e["name"]), "outside_cohort_dropped": outside, "allowed": allowed}


def skip_file(args):
    """One staged-inputs file: its sha256 and, per wanted key, the (irid, call_index) of call_end calls on a fixed token."""
    path, keys, fixed_ids = args
    h, ks, skip = hashlib.sha256(), set(keys), {}
    with open(path, "rb") as f:
        for line in f:
            h.update(line)
            ep = json.loads(line)
            if ep["key"] in ks:
                if ep["key"] in skip:
                    raise Stop(f"STOP {path}: a key twice in one staged file")
                skip[ep["key"]] = [[c["irid"], c["call_index"]] for c in ep["calls"]
                                   if c["kind"] == "call_end" and c["readout"] is not None and ep["ids"][c["readout"]] in fixed_ids]
    if set(skip) != ks:
        raise Stop(f"STOP {path}: {len(ks - set(skip))} wanted keys not staged")
    return path, h.hexdigest(), skip


def read_meta(e: dict, skip: set) -> dict:
    """One run's meta: command rows = call_end with a row, minus the fixed-token calls; reply rows = output_end with a row_mean."""
    m = json.loads((Path("/u") / e["root"] / "meta" / e["arm"] / f"{hashlib.sha1(e['key'].encode()).hexdigest()[:20]}.json").read_text())
    if m["key"] != e["key"] or m.get("cut") != "full" or m.get("truncated") is not False or m["n_tokens"] != e["full_len"]:
        raise Stop(f"STOP {e['name']}: meta key, cut, truncated or n_tokens")
    sel = {"command": [c for c in m["calls"] if c["kind"] == "call_end" and c.get("row") is not None and (c["irid"], c["call_index"]) not in skip],
           "reply": [c for c in m["calls"] if c["kind"] == "output_end" and c.get("row_mean") is not None]}
    rows = {"command": [c["row"] for c in sel["command"]], "reply": [c["row_mean"] for c in sel["reply"]]}
    invalid = {}  # Codex r2 P2: per readout, so a reply-only problem never stops the command verdict (AMEND1 A1)
    for ro, cs in sel.items():
        invalid[ro] = (len({(str(c["src_irid"]), int(c["call_index"])) for c in cs}) != len(cs)
                       or any(type(r) is not int or r < 0 or r >= m["n_rows"] for r in rows[ro]))
        if invalid[ro]:
            rows[ro] = []
    return {"chunk": m["chunk"], "row_start": m["row_start"], "rows": rows, "invalid": invalid}


# ---------------------------------------------------------------- one cell: G3, G4, D2, D3, diagnostics, controls (PRESTATED §4; AMEND1)
def cell_check(cell: str, R: dict, runs: list, metas: dict, rows: dict, stage_d: dict, DD: dict) -> tuple:
    """One cell: G3, G4, D2, D3, diagnostics, controls, slot 23 and the reading. R = refit3's record; DD = the L48 detector_vs_drop record."""
    import io

    import numpy as np
    import torch

    spec, Rc = CELLS[cell], R["cells"][cell]
    ro = spec["readout"]
    n_bad = sum(m["invalid"][ro] for m in metas.values())
    if n_bad:
        raise Stop(f"STOP {cell}: {n_bad} runs' metas have duplicate {ro} units or a row outside the epoch")
    by_name = {e["name"]: e for e in runs}
    cp = Rc["problems"]["c_simatk_vs_run1"]
    rr = rows[ro]
    names = sorted(rr)  # runs with rows in this readout, sorted by name
    counts = {n: int(rr[n].shape[0]) for n in names}
    c_names = [n for n in names if by_name[n]["world"] in ("sim", "run1")]
    r2_names = [n for n in names if by_name[n]["world"] == "run2"]
    chunks = sorted({f"{by_name[n]['root']}/{metas[n]['chunk']}" for n in names})
    nrow = {"class1": sum(counts[n] for n in c_names if by_name[n]["world"] == "sim"),
            "class0": sum(counts[n] for n in c_names if by_name[n]["world"] == "run1")}
    g3 = {"rows_per_run_equal": counts == Rc["rows_per_run"], "runs_with_rows": len(names),
          "runs_without_rows_equal": sorted(set(by_name) - set(names)) == sorted(Rc["runs_without_rows"]),
          "chunks_equal": chunks == sorted(Rc["chunks_read"]), "n_chunks": len(chunks), "rows_total_equal": sum(counts.values()) == Rc["rows"],
          "finite": all(bool(np.isfinite(rr[n]).all()) for n in names),
          "c_runs": {"class1": sum(by_name[n]["world"] == "sim" for n in c_names), "class0": sum(by_name[n]["world"] == "run1" for n in c_names)},
          "c_n_rows_equal": nrow == cp["n_rows"], "run2_runs": len(r2_names)}
    g3["pass"] = (g3["rows_per_run_equal"] and g3["runs_without_rows_equal"] and g3["chunks_equal"] and g3["rows_total_equal"] and g3["finite"]
                  and g3["c_runs"] == cp["n_runs"] and g3["c_n_rows_equal"] and len(r2_names) == 1313)
    if not g3["pass"]:
        raise Stop(f"STOP {cell} G3 {json.dumps(g3)}")
    W, g4 = {}, {}
    for k, (p, h) in spec["npz"].items():
        b = (Path("/w") / p).read_bytes()
        g4[k] = sha(b) == h and sha(b) == Rc["saved_weights"][{"c": "c_simatk_vs_run1", "d": "d_run2_vs_run1"}[k]]["sha256"]
        if not g4[k]:  # Codex r2 P1: the hash is checked BEFORE the bytes are parsed
            raise Stop(f"STOP {cell} G4 {k}: npz sha256 differs from its pin")
        z = np.load(io.BytesIO(b), allow_pickle=False)
        want = {f"f{f}__{q}" for f in range(5) for q in ("W", "b", "mean", "std")}
        if set(z.files) != want or any(z[n].dtype != np.float64 for n in want):
            raise Stop(f"STOP {cell} G4 {k}: npz keys or dtypes are not f0-f4 x W/b/mean/std float64")
        W[k] = {f: tuple(np.asarray(z[f"f{f}__{q}"], np.float64) for q in ("W", "b", "mean", "std")) for f in range(5)}
        g4[k] = all(t[0].shape == (HDIM,) and t[1].shape == (1,) and t[2].shape == (HDIM,) and t[3].shape == (HDIM,) for t in W[k].values())
    g4["pass"] = g4["c"] and g4["d"]
    if not g4["pass"]:
        raise Stop(f"STOP {cell} G4 {json.dumps(g4)}")

    dev = "cuda"
    XT = torch.from_numpy(np.concatenate([rr[n] for n in names])).to(dev)  # float16 as stored, rows in name order
    start = dict(zip(names, np.concatenate([[0], np.cumsum([counts[n] for n in names])[:-1]]).tolist()))

    def idx_of(ns):
        return torch.from_numpy(np.concatenate([np.arange(start[n], start[n] + counts[n]) for n in ns])).to(dev)

    def seg_of(ns):
        return torch.from_numpy(np.repeat(np.arange(len(ns)), [counts[n] for n in ns])).to(dev)

    def t64(a):
        return torch.from_numpy(a).to(dev)

    mine, d3, pa, pb, dhat, eps = ({} for _ in range(6))
    folds = []
    diag = {"stats_match": True, "stats_match_std_units": True, "stored_cert": True, "my_fit_err_le_1e-6": True, "d3_bound_le_1e-9": True,
            "my_certified": True}
    u = 2.0 ** -53
    for f in range(5):
        tr = [n for n in c_names if by_name[n]["fold"] != f]
        te = [n for n in c_names if by_name[n]["fold"] == f]
        r2 = [n for n in r2_names if by_name[n]["fold"] == f]
        itr, ite, ir2 = idx_of(tr), idx_of(te), idx_of(r2)
        hold = te + r2  # held-out runs of this fold: c test runs, then run2
        segh = seg_of(hold)
        ih = torch.cat([ite, ir2])
        ytr = torch.from_numpy(np.repeat([float(by_name[n]["world"] == "sim") for n in tr], [counts[n] for n in tr])).to(dev)
        # D2: my standardization and my Newton fit; my remaining error on held-out rows
        mu, sd = standardize_stats(XT, itr, dev)
        w, b, cert, delta, Lh = newton(XT, itr, ytr, mu, sd, dev)
        mine.update(run_means(row_scores(XT, ih, mu, sd, w, b)[0], segh, hold))
        e_rows = row_scores(XT, ih, mu, sd, delta[:-1], delta[-1])[0]  # my own leftover error per row, x~.H^-1 g (AMEND3 C2: eps)
        err = float(e_rows.abs().max())
        eps.update(run_means(e_rows, segh, hold))
        # D3: the stored weights, standardized with the stored stats, plus the rounding bound of the comparison
        Ws, bs, ms, ss = (t64(a) for a in W["c"][f])
        sc, ab = row_scores(XT, ih, ms, ss, Ws, bs[0], with_abs=True)
        d3.update(run_means(sc, segh, hold))
        bound = 2 * ((HDIM + 3) * u * float(ab.max()) + max(counts[n] for n in hold) * u * float(sc.abs().max()))
        # diagnostics: my stats vs the stored stats (in units of the column's std); the stored fit's gradient on my rows;
        # RESID_EXPLAINED's prediction dhat = run mean of x~.H^-1 g_s (x~ with the stored stats; H = my final Hessian)
        stat_abs = max(float((mu - ms).abs().max()), float((sd - ss).abs().max()))  # PRESTATED §4: within 1e-9 (absolute; gates STATS_MATCH)
        stat_dev = max(float(((mu - ms).abs() / sd).max()), float(((sd - ss).abs() / sd).max()))  # the same in std units (reported only)
        _, g_s, _ = objective(XT, itr, ytr, ms, ss, Ws, bs[0])
        stored_mg = float(g_s.abs().max())
        ds = torch.cholesky_solve(g_s[:, None], Lh)[:, 0]
        dhat.update(run_means(row_scores(XT, ih, ms, ss, ds[:-1], ds[-1])[0], segh, hold))
        # controls: a perturbed weight; the wrong npz (d_run2_vs_run1)
        Wp = Ws.clone()
        Wp[0] += 1e-3
        pa.update(run_means(row_scores(XT, ih, ms, ss, Wp, bs[0])[0], segh, hold))
        Wd, bd, md, sdd = (t64(a) for a in W["d"][f])
        pb.update(run_means(row_scores(XT, ih, md, sdd, Wd, bd[0])[0], segh, hold))
        diag["stats_match"] &= stat_abs <= TOL_DIAG
        diag["stats_match_std_units"] &= stat_dev <= TOL_DIAG
        diag["stored_cert"] &= stored_mg <= CERT_BAR
        diag["my_fit_err_le_1e-6"] &= err <= 1e-6
        diag["d3_bound_le_1e-9"] &= bound <= TOL_D3
        diag["my_certified"] &= cert["certified"]
        folds.append({"fold": f, "n_train_runs": len(tr), "n_test_runs": len(te), "n_run2_runs": len(r2), "n_train_rows": len(itr),
                      "n_test_rows": len(ite), "n_run2_rows": len(ir2), "newton": cert, "_my_fit_err_max": err, "_d3_rounding_bound": bound,
                      "_stats_dev_abs": stat_abs, "_stats_dev_in_std_units": stat_dev, "_stored_max_grad_on_my_rows": stored_mg})
        del ytr, itr, ite, ir2, ih, Lh
        torch.cuda.empty_cache()
    del XT
    torch.cuda.empty_cache()

    rec_c, rec_h = cp["run_scores"], Rc["heldout_run2"]["run_scores"]
    rec_all = {**rec_c, **rec_h}
    mine_c = {k: v for k, v in mine.items() if not k.startswith("run2|")}
    mine_h = {k: v for k, v in mine.items() if k.startswith("run2|")}
    cmp = {"D2_c": compare(mine_c, rec_c, TOL_D2), "D2_run2": compare(mine_h, rec_h, TOL_D2),
           "D3_c": compare({k: v for k, v in d3.items() if not k.startswith("run2|")}, rec_c, TOL_D3),
           "D3_run2": compare({k: v for k, v in d3.items() if k.startswith("run2|")}, rec_h, TOL_D3),
           "PC_D2_vs_L63": compare(mine_c, R["cells"][spec["control"]]["problems"]["c_simatk_vs_run1"]["run_scores"], TOL_D2),
           "PC_D3a_perturbed_w": compare(pa, rec_all, TOL_D3), "PC_D3b_wrong_npz": compare(pb, rec_all, TOL_D3)}
    ks = sorted(mine_c)
    my_auc = auroc([mine_c[k] for k in ks], [k.startswith("sim|") for k in ks])
    rk = sorted(rec_c)
    auc_rec = auroc([rec_c[k] for k in rk], [k.startswith("sim|") for k in rk])
    auc = {"_mine": my_auc, "_record": cp["run_level"]["auroc"], "_record_recomputed": auc_rec,
           "auroc_def_match": abs(auc_rec - cp["run_level"]["auroc"]) <= 1e-12}
    auc["d2_iii"] = "NO VERDICT" if not auc["auroc_def_match"] else ("MATCH" if abs(my_auc - cp["run_level"]["auroc"]) <= TOL_D2 else "MISMATCH")
    # RESID_EXPLAINED (AMEND1 A5, AMEND3 C2): (record - mine) vs (dhat - eps), over all 3,933 keys
    keys = sorted(set(rec_all) & set(mine) & set(dhat) & set(eps))
    resid = max(abs((rec_all[k] - mine[k]) - (dhat[k] - eps[k])) for k in keys)
    dmax = max(abs(dhat[k]) for k in keys)
    diag["resid_explained"] = len(keys) == len(rec_all) == 3933 and resid <= max(1e-6, 0.05 * dmax)
    diag["dhat_max_le"] = {"1e-4": dmax <= 1e-4, "1e-3": dmax <= 1e-3, "1e-2": dmax <= 1e-2}
    diag["auroc_def_match"] = auc["auroc_def_match"]
    # AMEND2 B1: ranks over detector_vs_drop's x (sim) and null (run2) sets, record vs my refit
    rs = {"x": rank_swaps({k: v for k, v in rec_c.items() if k.startswith("sim|")}, {k: v for k, v in mine_c.items() if k.startswith("sim|")}),
          "null": rank_swaps(rec_h, mine_h)}
    rank_identical = all(r["keys_equal"] and r["swaps"] == 0 for r in rs.values())
    # AMEND3 C3: slot 23 recomputed (port of run_detector_vs_drop.analyse) from refit3's own x, from my x, and from x := y (control)
    rnode = DD["result"]["probes"]["sol_a_uncut C"][SLOT_DET[cell]]
    rv = slot_vals(rnode)
    rn = [rnode["sim_score_vs_drop"]["n_cells"], rnode["null_run2_score_vs_run2_minus_run1"]["n_cells"]]
    sp_rec = slot23(rec_c, rec_h, stage_d, R["run_main"])
    sp_mine = slot23(mine_c, mine_h, stage_d, R["run_main"])
    sp_ctl = slot23({f"sim|{c}": v for c, v in sp_rec["y_drop"].items()}, {f"run2|{c}": v for c, v in sp_rec["y_null"].items()}, stage_d, R["run_main"])
    # the port's own control: refit3's x with the highest- and lowest-scored sim cells (among cells with a finite y) swapped
    xc = {c: rec_c[f"sim|{c}"] for c in sp_rec["y_drop"] if f"sim|{c}" in rec_c}
    hi_, lo_ = max(xc, key=xc.get), min(xc, key=xc.get)
    sp_swap = slot23(dict(rec_c, **{f"sim|{hi_}": xc[lo_], f"sim|{lo_}": xc[hi_]}), rec_h, stage_d, R["run_main"])

    def ncells(sp):
        return [sp["sim_score_vs_drop"]["n_cells"], sp["null_run2_score_vs_run2_minus_run1"]["n_cells"]]

    # AMEND4 D2: the 4 PRINTED numbers (rows 119-122: sim point, ci95 low, ci95 high; null point) gate; all 6 (+ the null's ci95) are reported
    printed, every = (0, 1, 2, 3), (0, 1, 2, 3, 4, 5)

    def same_print(sp, idx):
        v = slot_vals(sp)
        return [k2s(v[i]) for i in idx] == [k2s(rv[i]) for i in idx] and ncells(sp) == rn

    def repro(sp, idx):
        v = slot_vals(sp)
        return ncells(sp) == rn and all(close(v[i], rv[i], 1e-12) for i in idx)

    slot = {"port_repro": repro(sp_rec, printed), "port_repro_all": repro(sp_rec, every),
            "pc_port_swap_failed_as_predicted": not repro(sp_swap, printed),
            "unchanged_printed": same_print(sp_mine, printed), "unchanged_all": same_print(sp_mine, every),
            "pc_x_eq_y_failed_as_predicted": not same_print(sp_ctl, printed), "n_printed": len(printed), "n_all": len(every)}
    diag["slot23"] = slot
    # verdicts and the reading: AMEND3 C1's ordered list, top to bottom
    # Codex r2 P1: the rounding bound first (PRESTATED §4: bound exceeded -> NO VERDICT, whatever the comparison says)
    d3v = "NO VERDICT" if not diag["d3_bound_le_1e-9"] else ("MATCH" if cmp["D3_c"]["match"] and cmp["D3_run2"]["match"] else "MISMATCH")
    d2 = {"i": cmp["D2_c"]["match"], "ii": cmp["D2_run2"]["match"], "iii": auc["d2_iii"]}
    if d3v == "MISMATCH":
        cls = "STOP_D3"
    elif d3v == "NO VERDICT" or not diag["my_certified"] or not diag["my_fit_err_le_1e-6"]:
        cls = "NO_VERDICT"
    elif not (diag["stats_match"] and diag["stored_cert"] and diag["resid_explained"]):
        cls = "STOP_DEFINITION_GAP_OR_BUG"
    elif d2["i"] and d2["ii"] and (d2["iii"] == "MATCH" or not diag["auroc_def_match"]):
        cls = "REPRODUCED"
    else:
        cls = "CERTIFIED_FIT_WITHIN_DELTA_HAT"
    ruling, label = {"REPRODUCED": "REPRODUCED", "STOP_D3": "STOP", "STOP_DEFINITION_GAP_OR_BUG": "STOP", "NO_VERDICT": "NO_VERDICT"}.get(cls), None
    if cls == "REPRODUCED":
        label = "x reproduced" if diag["auroc_def_match"] else LABELS["reproduced_no_auroc"]
    elif cls == "CERTIFIED_FIT_WITHIN_DELTA_HAT":
        if rank_identical:
            ruling, label = "PASS (value half)", LABELS["pass_ranks"]
        elif (diag["dhat_max_le"]["1e-3"] and slot["unchanged_printed"] and slot["port_repro"] and slot["pc_x_eq_y_failed_as_predicted"]
              and slot["pc_port_swap_failed_as_predicted"]):
            ruling, label = "PASS (value half)", LABELS["pass_slot"]
        else:
            ruling = "UNRESOLVED"
    disclose = (["slot23.unchanged_all = false: the null's ci95 (not printed) moves at print rounding; the 4 printed numbers do not (AMEND4 D2)"]
                if label == LABELS["pass_slot"] and not slot["unchanged_all"] else [])
    reading = {"class": cls, "ruling_09": ruling, "label": label, "disclosures": disclose, "rank_identical": rank_identical,
               "rank_swaps": {k: r["swaps"] for k, r in rs.items()}, "rank_n": {k: r["n"] for k, r in rs.items()},
               "rank_keys_equal": {k: r["keys_equal"] for k, r in rs.items()}, "dhat_max_le": diag["dhat_max_le"], "slot23": slot}
    controls = {k: not cmp[k]["match"] for k in ("PC_D2_vs_L63", "PC_D3a_perturbed_w", "PC_D3b_wrong_npz")}
    full = {"gates": {"G3": g3, "G4": g4}, "folds": folds, "diagnostics": diag, "compare": cmp, "auroc": auc, "controls_failed_as_predicted": controls,
            "verdict": {"D2": d2, "D3": d3v}, "reading": reading, "_resid_max": resid, "_dhat_max": dmax,
            "mine": {"run_scores": mine}, "replay": {"run_scores": d3}, "dhat": dhat, "eps": eps,
            "slot23": {"_record": rv, "_record_n_cells": rn, "_port_from_record_x": slot_vals(sp_rec), "_from_my_x": slot_vals(sp_mine),
                       "_control_x_eq_y": slot_vals(sp_ctl), "_control_swap": slot_vals(sp_swap), "swap_cells": [hi_, lo_],
                       "_n_cells": {"port": ncells(sp_rec), "mine": ncells(sp_mine), "control": ncells(sp_ctl), "swap": ncells(sp_swap)}},
            "record_values": {"c_run_scores": rec_c, "heldout_run2_run_scores": rec_h, "run_level_auroc": cp["run_level"]["auroc"]}}
    summary = {"verdict": full["verdict"], "reading": reading, "gates": {"G3": g3["pass"], "G4": g4["pass"]},
               "counts": {"runs_with_rows": len(names), "chunks": len(chunks), "c_runs": g3["c_runs"], "run2_runs": len(r2_names),
                          "keys_resid": len(keys), "folds_certified": sum(fd["newton"]["certified"] for fd in folds),
                          "newton_iters": [fd["newton"]["n_iter"] for fd in folds]},
               "compare": {k: {q: v for q, v in c.items() if not q.startswith("_")} for k, c in cmp.items()},
               "controls_failed_as_predicted": controls, "diagnostics": diag}
    return full, summary


# ---------------------------------------------------------------- the check (A100-40GB; PRESTATED §4, §7)
@app.function(image=image, gpu="A100-40GB", cpu=(8, 8), memory=(49152, 49152), timeout=3600, retries=0, scaledown_window=2,
              volumes={"/u": acts, "/w": wvol, "/hf": hfvol, "/out": out})
def check(blobs: dict, pins: dict, meta: dict) -> dict:
    """The remote boundary (Codex r2 P1): my own Stop messages (booleans, counts, keys) pass; any other exception is replaced by its type
    and the stage it hit, raised outside the except block so its message and context never reach the laptop."""
    stage = {"now": "start"}
    failed = None
    try:
        return _check(blobs, pins, meta, stage)
    except Stop:
        raise
    except Exception as e:  # noqa: BLE001 - re-raised below as a Stop with its type and stage
        failed = f"FAILED {type(e).__name__} during stage {stage['now']} (message withheld: it may hold values)"
    raise Stop(failed)


def _check(blobs: dict, pins: dict, meta: dict, stage: dict) -> dict:
    import io
    import time
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import numpy as np

    t0, secs = time.time(), {}
    stage["now"] = "selftest"
    st = selftest()
    if not st["all_pass"]:
        raise Stop(f"STOP self-test failed: {sorted(k for k, v in st['tests'].items() if not v)}")
    secs["selftest"] = time.time() - t0
    stage["now"] = "pins"
    for k, v in pins["sha256"].items():  # every git input again, in the container
        if sha(blobs[k]) != v:
            raise Stop(f"STOP {k}: sha256 differs in the container")
    R = json.loads(blobs[pins["record"]])
    stage_d, DD = json.loads(blobs[pins["stage_d"]]), json.loads(blobs[pins["detdrop"]])

    # G1: the run set, folds, mains and sets, by key
    stage["now"] = "G1 run set"
    B = build_runs(blobs, pins)
    runs = B["runs"]
    by_name = {e["name"]: e for e in runs}
    g1 = {"n_runs": len(runs), "per_world": {w: sum(e["world"] == w for e in runs) for w in WORLDS},
          "per_set": {s: sum(e["set"] == s for e in runs) for s in SETS}, "mains": len({e["main"] for e in runs}),
          "outside_cohort_dropped": B["outside_cohort_dropped"],
          "run_keys_equal": {e["name"]: e["key"] for e in runs} == R["run_keys"],
          "run_folds_equal": {e["name"]: e["fold"] for e in runs} == R["run_folds"],
          "run_main_equal": {e["name"]: e["main"] for e in runs} == R["run_main"],
          "run_set_equal": {e["name"]: e["set"] for e in runs} == R["run_set"]}
    g1["pass"] = (g1["n_runs"] == 3939 and g1["mains"] == 864 and g1["per_set"] == R["per_set"]
                  and all(g1[k] for k in ("run_keys_equal", "run_folds_equal", "run_main_equal", "run_set_equal")))
    if not g1["pass"]:
        raise Stop(f"STOP G1 {json.dumps(g1)}")

    # G2: fixed-token skip list from the staged inputs; the 3,939 metas
    stage["now"] = "G2 skip list"
    vocab_bytes = (Path("/hf") / VOCAB).read_bytes()
    vocab = json.loads(vocab_bytes)
    fixed_ids = {vocab[t] for t in FIXED}
    want = {}
    for e in runs:
        want.setdefault((e["staged"], e["arm"]), {"keys": [], "sha": set()})
        want[(e["staged"], e["arm"])]["keys"].append(e["key"])
        want[(e["staged"], e["arm"])]["sha"].add(e["staged_sha256"])
    if any(len(v["sha"]) != 1 for v in want.values()):
        raise Stop("STOP G2: two staged sha256 for one staged file")
    jobs = [(f"/u/{st_}/inputs/{arm}.jsonl", v["keys"], fixed_ids) for (st_, arm), v in sorted(want.items())]
    with ThreadPoolExecutor(8) as ex:  # about 75 files of about 16 MB: threads suffice (no fork after torch)
        got = list(ex.map(skip_file, jobs))
    skip, file_sha, dropped = {}, {}, {}
    for path, h, sk in got:
        rel = path.removeprefix("/u/")
        file_sha[rel], dropped[rel] = h, sum(len(v) for v in sk.values())
        skip.update(sk)
    staged_ok = all(file_sha[f"{st_}/inputs/{arm}.jsonl"] == next(iter(v["sha"])) for (st_, arm), v in want.items())
    secs["skip"] = time.time() - t0
    stage["now"] = "G2 metas"
    with ThreadPoolExecutor(32) as ex:
        metas = dict(zip([e["name"] for e in runs], ex.map(lambda e: read_meta(e, {tuple(x) for x in skip[e["key"]]}), runs)))
    g2 = {"n_files": len(file_sha), "staged_sha_equal_inventory": staged_ok, "staged_sha_equal_record": file_sha == R["skip"]["staged_sha256"],
          "dropped_rows_equal_record": dropped == R["skip"]["dropped_rows"], "fixed_ids_n": len(fixed_ids), "metas_read": len(metas)}
    g2["pass"] = staged_ok and g2["staged_sha_equal_record"] and g2["dropped_rows_equal_record"] and len(metas) == 3939 and len(fixed_ids) == 2
    if not g2["pass"]:
        raise Stop(f"STOP G2 {json.dumps(g2)}")
    secs["metas"] = time.time() - t0

    # rows: every chunk either readout needs is read once and hashed in full (the activation manifest)
    by_chunk = {}
    for n, m in metas.items():
        if m["rows"]["command"] or m["rows"]["reply"]:
            by_chunk.setdefault((by_name[n]["root"], m["chunk"]), []).append(n)

    def read_chunk(item):
        (root, ch), ns = item
        if ch not in B["allowed"][root]:
            raise Stop(f"STOP G3 {root}/{ch}: not in its extraction record")
        p = Path("/u") / root / "by_layer" / f"L{LAYER}" / f"{ch}.npy"
        b = p.read_bytes()
        A = np.load(io.BytesIO(b), allow_pickle=False)
        if A.shape != (B["allowed"][root][ch], HDIM) or A.dtype != np.float16:
            raise Stop(f"STOP G3 {root}/{ch}: shape or dtype differs from the extraction record")
        # Codex r4: absolute row indices are validated PER READOUT before gathering; a reply-only problem is returned, never raised
        got_, oob = {"command": {}, "reply": {}}, {"command": [], "reply": []}
        used = np.zeros(A.shape[0], bool)
        for ro in ("command", "reply"):
            for n in ns:
                if not metas[n]["rows"][ro]:
                    continue
                q = [metas[n]["row_start"] + r for r in metas[n]["rows"][ro]]
                if min(q) < 0 or max(q) >= A.shape[0]:
                    oob[ro].append(n)
                    continue
                got_[ro][n] = A[q]
                used[q] = True
        if oob["command"]:
            raise Stop(f"STOP G3 {root}/{ch}: {len(oob['command'])} runs' command rows lie outside the chunk")
        # AMEND4 D1: gathered rows are checked per cell in G3 (a STOP); rows neither readout gathers only get a count of non-finite values
        mf = {"path": str(p.relative_to("/u")), "size": len(b), "sha256": sha(b), "nonfinite_unused": int((~np.isfinite(A[~used])).sum())}
        return mf, got_, oob["reply"]

    stage["now"] = "G3 chunk reads"
    rows, manifest = {"command": {}, "reply": {}}, []
    with ThreadPoolExecutor(8) as ex:
        for mf, g, bad_reply in ex.map(read_chunk, sorted(by_chunk.items())):
            manifest.append(mf)
            for ro in rows:
                rows[ro].update(g[ro])
            for n in bad_reply:  # reply rows outside the chunk: the reply diagnostic's cell_check then STOPs on its own (AMEND1 A1)
                metas[n]["invalid"]["reply"] = True
    manifest.sort(key=lambda x: x["path"])
    manifest_sha = sha(json.dumps(manifest, sort_keys=True).encode())
    secs["rows"] = time.time() - t0

    cells_full, cells_sum = {}, {}
    stage["now"] = "cell command|L48"
    cells_full[VERDICT_CELL], cells_sum[VERDICT_CELL] = cell_check(VERDICT_CELL, R, runs, metas, rows, stage_d, DD)  # a Stop here ends the job
    del rows["command"]
    secs[VERDICT_CELL] = time.time() - t0
    stage["now"] = "cell reply|L48 (diagnostic)"
    msg = None
    try:  # reply|L48 is a DIAGNOSTIC (AMEND1 A1): a failure is recorded and printed by type, and never touches the verdict
        cells_full["reply|L48"], cells_sum["reply|L48"] = cell_check("reply|L48", R, runs, metas, rows, stage_d, DD)
    except Stop as e:  # my own gate messages: booleans, counts and keys only
        msg = f"Stop: {e}"
    except Exception as e:  # noqa: BLE001 - recorded by type in the results and on the printed line, not swallowed
        msg = f"{type(e).__name__} (message withheld: it may hold values)"  # Codex r2 P1: never str(e) of a foreign exception
    if msg is not None:  # Codex r3: both handlers record the diagnostic entry
        cells_full["reply|L48"], cells_sum["reply|L48"] = {"error": msg}, {"error": msg, "reading": {"class": "NO_VERDICT", "ruling_09": "NO_VERDICT"}}
    secs["reply|L48"] = time.time() - t0

    v = cells_sum[VERDICT_CELL]
    ok = v["reading"]["ruling_09"] in ("REPRODUCED", "PASS (value half)") and all(v["controls_failed_as_predicted"].values())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rel = f"refit3l48/refit3l48_check_{stamp}_{meta.get('app_id') or 'noappid'}.json"
    res = {"label": "refit3-l48 VALUE check of refit3 command|L48 (reply|L48 diagnostic); PRELIMINARY until 09 rules FINAL", "stamp_pt": stamp,
           "meta": meta, "pins": pins, "l48_switch": L48_SWITCH, "selftest": st, "verdict_cell": VERDICT_CELL, "pass": ok,
           "settings": {"C": C, "cert_bar": CERT_BAR, "newton_tol": NEWTON_TOL, "max_newton": MAX_NEWTON, "chunk_rows": CHUNK, "tol_d2": TOL_D2,
                        "tol_d3": TOL_D3, "tol_diag": TOL_DIAG, "fixed_tokens": FIXED, "layer": LAYER, "cells": CELLS, "seed": SEED, "labels": LABELS,
                        "slot23": {"seed": SLOT_SEED, "draws": SLOT_D, "combo": SLOT_COMBO, "det": SLOT_DET, "print": "f'{x:+.2f}' with U+2212 for '-'",
                                   "port": "run_detector_vs_drop.py @ 76032545d (blob c51f7a59) analyse() L195-L274", "stage_d": STAGE_D,
                                   "detdrop": DETDROP}},
           "gates": {"G1": g1, "G2": g2}, "cells": cells_full, "manifest": manifest, "manifest_sha256": manifest_sha,
           "tokenizer": {"path": VOCAB, "sha256": sha(vocab_bytes), "fixed_ids": sorted(fixed_ids)}, "seconds": {k: round(x, 1) for k, x in secs.items()}}
    stage["now"] = "write results"
    body = json.dumps(res, indent=1, sort_keys=True).encode()
    dest = Path("/out") / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as fh:
        fh.write(body)
    out.commit()
    return {"verdict_cell": VERDICT_CELL, **v, "gates": {"G1": g1["pass"], "G2": g2["pass"], **v["gates"]},
            "g3_unused_rows": {"label": "G3 narrowed to gathered rows (producer's rule) by AMEND4, before any value",
                               "nonfinite_values": sum(m["nonfinite_unused"] for m in manifest),
                               "chunks_with_nonfinite": sum(m["nonfinite_unused"] > 0 for m in manifest)},
            "counts": {"runs": len(runs), "mains": g1["mains"], "staged_files": g2["n_files"], "metas": len(metas), "chunks_read": len(by_chunk),
                       **v["counts"]},
            "diagnostic_cells": {"reply|L48": cells_sum["reply|L48"]}, "selftest_all_pass": st["all_pass"],
            "results": {"volume": APP, "path": rel, "sha256": sha(body)}, "manifest_sha256": manifest_sha, "tokenizer_sha256": sha(vocab_bytes),
            "seconds": {k: round(x) for k, x in secs.items()}, "pass": ok}


# ---------------------------------------------------------------- local side: git pins only (keys and shas; no value is parsed or printed)
@app.local_entrypoint()
def main(test: bool = False):
    if test:
        r = selftest_cpu.remote()
        print(json.dumps({"refit3l48_test": r}), flush=True)
        if not r["all_pass"]:
            raise SystemExit(1)
        return

    def git(*a) -> bytes:
        return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout

    repo = Path(git("rev-parse", "--show-toplevel").decode().strip())
    me = Path(__file__).resolve().relative_to(repo)
    if git("status", "--porcelain", "--untracked-files=no") or git("status", "--porcelain", "--", str(HERE)):
        raise SystemExit("STOP uncommitted tracked changes, or uncommitted files in this experiment folder")
    head = git("rev-parse", "HEAD").decode().strip()
    blob = git("rev-parse", f"HEAD:{me}").decode().strip()
    if git("hash-object", str(Path(__file__).resolve())).decode().strip() != blob:
        raise SystemExit("STOP the running script is not the committed blob")
    blobs, shas = {}, {}

    def pin(commit: str, path: str, want: str) -> str:
        b = git("show", f"{commit}:{path}")
        if len(want) < 16 or not sha(b).startswith(want):
            raise SystemExit(f"STOP {path} @ {commit[:9]}: sha256 differs from its pin")
        k = f"{commit}:{path}"
        blobs[k], shas[k] = b, sha(b)
        return k

    pins = {"record": pin(*REC), "cohort": pin(*COHORT), "folds": pin(*FOLDS), "stage_d": pin(*STAGE_D), "detdrop": pin(*DETDROP),
            "configs": {}, "inventories": {}}
    F = json.loads(blobs[pins["folds"]])
    for c, p, h in [*F["pins"]["cells"].values(), F["pins"]["sp493_facts"]]:
        pin(c, p, h)
    for _, path, h in CONFIGS:
        k = pin(REC[0], path, h)
        pins["configs"][path] = k
        cfg = json.loads(blobs[k])
        inv = cfg["inventory"]
        pins["inventories"][path] = pin(inv["commit"], inv["path"], inv["sha256"])
        for spec in cfg["extraction_records"].values():
            pin(spec["commit"], spec["path"], spec["sha256"])
    pins["sha256"] = shas
    prestated = {p.name: sha(p.read_bytes()) for p in sorted(HERE.glob("PRESTATED_*.md"))}
    meta = {"git_head": head, "script": str(me), "script_blob": blob, "script_sha256": sha(Path(__file__).read_bytes()),
            "prestated": prestated, "app_id": getattr(app, "app_id", None), "n_git_inputs": len(blobs),
            "git_input_bytes": sum(len(b) for b in blobs.values())}
    print(json.dumps({"refit3l48_launch": {k: meta[k] for k in ("git_head", "script_blob", "app_id", "n_git_inputs", "git_input_bytes")}}), flush=True)
    r = check.remote(blobs, pins, meta)
    print(json.dumps({"refit3l48": r}), flush=True)
    if not r["pass"] or not all(r["gates"].values()):
        raise SystemExit(1)
