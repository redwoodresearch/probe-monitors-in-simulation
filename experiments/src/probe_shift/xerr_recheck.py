"""Independent X_err recheck machinery (4f's checker). PRELIMINARY.

Written from the spec only (experiments/2026-10-14_xerr_scaleup_recheck/RULE.md lists the versions), never from session
09's implementation. Pure numpy + stdlib so it ships to a Modal CPU container as one file. No score value is ever printed
here; callers write values only into the result JSON.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re

import numpy as np

NDRAW = 10_000
SEED = (20261014, 1)
ALPHAS = ("0.01", "0.05")
ARMS = ("run1", "run2", "sim")
FIVE = ("sol_a", "fibs", "harmfulness", "refusal", "highstakes")
COMPS = ("sim_minus_mean_real", "sim_minus_run1", "sim_minus_run2", "reexecution_run2_minus_run1")
FOLDER_RE = re.compile(
    r"^(su311|nm360)_(q[0-9]+r?|s[0-9]+)_(main|run2|sim)(_s[0-9]+)?_v1(_try[0-9]+)?__(honest_a|honest_b|attack|sim_attack)$"
)
WORLDS = ("run1", "run2", "sim", "honest_a", "honest_b", "honest_c")
CERT_OK = "ok (>= 95% complete draws)"
CERT_NO = "no across-probe certification (< 95% complete draws)"
FEW = "too few members (< 2): spread undefined, no X_err"
UNDER = "underpowered (n < 30 mains)"
DIFFER = "strata differ (legacy label: equivalence not established); stratum results lead"
POOLED_OK = "strata pooled (TOST passed)"


class Stop(Exception):
    """A registered stop: the job ends and the record says why."""


# ---------------------------------------------------------------------------------------------------------- input helpers
def _nodup(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise Stop(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def loads(b: bytes):
    return json.loads(b, object_pairs_hook=_nodup)


def pinned(b: bytes, want: str, what: str) -> bytes:
    got = sha256(b)
    if got != want:
        raise Stop(f"sha256 mismatch for {what}: {got} != {want}")
    return b


def split_key(k: str) -> tuple[str, str]:
    w, _, c = k.partition("|")
    if not c or "|" in c:
        raise Stop(f"bad run key {k!r}")
    return w, c


def npz_runs(raw: bytes, what: str, values: bool = True, skip=frozenset()) -> dict:
    """Per-run aggregates of a per-row npz: mean and max of `score`, folds, and per-model aggregates if present.

    values=False (parse mode, blinding): keys, row counts, folds and non-finite counts only; no score is aggregated.
    Rows (and run_* entries) of keys in `skip` are removed before any validation or aggregation."""
    z0 = np.load(io.BytesIO(raw), allow_pickle=False)
    kf = "key" if "key" in z0.files else "keys"
    sf = "score" if "score" in z0.files else ("scores" if "scores" in z0.files else None)
    keys = z0[kf].astype(str)
    if keys.ndim != 1:
        raise Stop(f"{what}: key array is not 1-D")
    present = set(keys.tolist()) | (set(z0["run_key"].astype(str).tolist()) if "run_key" in z0.files else set())
    skip = set(skip) & present
    if not skip:
        return _npz_runs(z0, keys, sf, what, values)  # unchanged lazy path when no skipped key is present
    keep = ~np.isin(keys, sorted(skip))
    z = {f: z0[f] for f in z0.files}
    for f in list(z):
        if f.startswith("run_"):
            continue
        if z[f].shape[:1] == keys.shape:
            z[f] = z[f][keep]
    if "run_key" in z:
        rkeep = ~np.isin(z["run_key"].astype(str), sorted(skip)) if skip else np.ones(len(z["run_key"]), bool)
        for f in ("run_key", "run_score", "run_fold", "run_stratum"):
            if f in z and z[f].shape[:1] == rkeep.shape:
                z[f] = z[f][rkeep]
    keys = keys[keep]

    class _Z(dict):
        @property
        def files(self):
            return list(self.keys())

    return _npz_runs(_Z(z), keys, sf, what, values)


def _npz_runs(z, keys, sf, what, values) -> dict:
    out = {"files": sorted(z.files), "nrows": int(len(keys))}
    uk, inv = np.unique(keys, return_inverse=True)
    out["keys"] = uk.tolist()
    cnt = np.bincount(inv, minlength=len(uk)).astype(float)

    def agg(col):
        col = col.astype(np.float64)
        tot = np.zeros(len(uk))
        mx = np.full(len(uk), -np.inf)
        np.add.at(tot, inv, col)
        np.maximum.at(mx, inv, col)
        return tot / np.maximum(cnt, 1), mx

    if sf is not None:
        sc = z[sf].astype(np.float64)
        if sc.shape != keys.shape:
            raise Stop(f"{what}: score shape {sc.shape} != key shape {keys.shape}")
        out["n_nonfinite_rows"] = int((~np.isfinite(sc)).sum())
        if values:
            out["mean"], out["max"] = agg(sc)
    if "score_by_model" in z.files and values:
        sbm = z["score_by_model"].astype(np.float64)
        if sbm.ndim != 2 or sbm.shape[0] != len(keys):
            raise Stop(f"{what}: score_by_model shape {sbm.shape}")
        out["by_model_mean"] = np.stack([agg(sbm[:, j])[0] for j in range(sbm.shape[1])], 1)
        out["by_model_max"] = np.stack([agg(sbm[:, j])[1] for j in range(sbm.shape[1])], 1)
    if "fold" in z.files:
        fo = z["fold"].astype(np.int64)
        lo = np.full(len(uk), 10**9)
        hi = np.full(len(uk), -(10**9))
        np.minimum.at(lo, inv, fo)
        np.maximum.at(hi, inv, fo)
        if (lo != hi).any():
            raise Stop(f"{what}: rows of one run disagree on fold")
        out["fold"] = lo.tolist()
    for f in ("run_key", "run_score", "run_fold", "run_stratum"):
        if f in z.files and (values or f != "run_score"):
            out[f] = z[f].tolist()
    return out


def _integral(a: np.ndarray) -> bool:
    if np.issubdtype(a.dtype, np.integer):
        return True
    if np.issubdtype(a.dtype, np.floating):
        return bool(np.isfinite(a).all() and (a == np.round(a)).all())
    return False


def npz_meta(raw: bytes, what: str, require: tuple = ()) -> dict:
    """Metadata only (parse mode; blinding): array names and exact shapes, per-run keys, row-fold consistency, run_key /
    run_fold, and non-finite COUNTS. No score is summarized, aggregated or returned. `require` names arrays that must
    exist; `score` must be 1-D (n,), `score_by_model` (n, 5), folds integral."""
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    out = {"arrays": {f: list(z[f].shape) for f in z.files}, "problems": []}
    for f in require:
        if f not in z.files:
            out["problems"].append(f"required array {f} missing")
    kf = "key" if "key" in z.files else ("keys" if "keys" in z.files else None)
    if kf is None:
        out["problems"].append("no key array")
        return out
    keys = z[kf].astype(str)
    if keys.ndim != 1:
        out["problems"].append("key array is not 1-D")
        return out
    n = len(keys)
    out["nrows"] = n
    for f in ("score", "scores", "fold", "unit", "call_index"):
        if f in z.files and z[f].shape != (n,):
            out["problems"].append(f"{f} shape {list(z[f].shape)} != [{n}]")
    for f in ("fold", "run_fold"):
        if f in z.files and not _integral(z[f]):
            out["problems"].append(f"{f} is not integral")
    for f in ("score", "scores"):
        if f in z.files and z[f].shape == (n,):
            bad_rows = ~np.isfinite(z[f].astype(np.float64))
            out["n_nonfinite_score"] = int(bad_rows.sum())
            out["runs_with_nonfinite_score"] = sorted(set(keys[bad_rows].tolist()))  # run keys only (blinding)
    for f in ("score_by_model", "cmd_A_by_model"):  # the second is 2e's alias; validated identically
        if f not in z.files:
            continue
        sbm = z[f]
        if sbm.ndim != 2 or sbm.shape[0] != n or sbm.shape[1] != 5:
            out["problems"].append(f"{f} shape {list(sbm.shape)} (want [{n}, 5])")
        else:
            bad_rows = ~np.isfinite(sbm.astype(np.float64)).all(1)
            out["n_nonfinite_score_by_model"] = out.get("n_nonfinite_score_by_model", 0) + int((~np.isfinite(sbm.astype(np.float64))).sum())
            out["runs_with_nonfinite_by_model"] = sorted(set(out.get("runs_with_nonfinite_by_model", [])) | set(keys[bad_rows].tolist()))
    uk, inv = np.unique(keys, return_inverse=True)
    out["keys"] = uk.tolist()
    if "fold" in z.files and z["fold"].shape == (n,) and _integral(z["fold"]):
        fo = z["fold"].astype(np.int64)
        lo = np.full(len(uk), 10**9)
        hi = np.full(len(uk), -(10**9))
        np.minimum.at(lo, inv, fo)
        np.maximum.at(hi, inv, fo)
        if (lo != hi).any():
            out["problems"].append(f"{int((lo != hi).sum())} runs whose rows disagree on fold")
        out["fold"] = dict(zip(uk.tolist(), lo.tolist()))
    if "run_key" in z.files:
        rk = z["run_key"].astype(str).tolist()
        out["run_key"] = rk
        if "run_fold" in z.files and _integral(z["run_fold"]):
            rf = z["run_fold"].astype(np.int64).tolist()
            if len(rf) != len(rk):
                out["problems"].append("run_fold and run_key lengths differ")
            else:
                out["run_fold"] = dict(zip(rk, rf))
        if len(set(rk)) != len(rk):
            out["problems"].append("duplicate run_key entries")
        if set(rk) != set(out["keys"]):
            out["problems"].append("run_key set differs from the rows' key set")
        if "fold" in out and "run_fold" in out and any(out["fold"][k] != out["run_fold"][k] for k in out["keys"] if k in out["run_fold"]):
            out["problems"].append("run_fold differs from the rows' fold")
    return out


# ---------------------------------------------------------------------------------------------------------- statistics
def order_threshold(values, alpha: str):
    """k = floor(alpha*n + 1e-9); the (k+1)-th largest value; a hit is strictly above it."""
    s = np.sort(np.asarray(list(values), dtype=np.float64))[::-1]
    n = len(s)
    k = int(math.floor(float(alpha) * n + 1e-9))
    if n == 0 or k >= n:
        raise Stop(f"threshold undefined: n={n}, k={k}")
    return float(s[k]), n, k


def stratified_draws(units: list, stratum_of: dict, ndraw: int = NDRAW, seed=SEED, strata_key=None) -> np.ndarray:
    """(ndraw+1, len(units)) integer multiplicities; row 0 is all ones.

    One fresh generator; strata visited in sorted order (by `strata_key` if given); within a stratum the units keep the
    order of `units` (callers pass them sorted); one multinomial call per stratum of size ndraw.
    """
    rng = np.random.default_rng(list(seed))
    W = np.zeros((ndraw + 1, len(units)))
    W[0] = 1.0
    lab = [stratum_of[u] for u in units]
    for s in sorted(set(lab), key=strata_key):
        ix = [i for i, x in enumerate(lab) if x == s]
        W[1:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1.0 / len(ix)), size=ndraw)
    return W


def weighted_auroc(pos, neg, W: np.ndarray, within_group: bool) -> np.ndarray:
    """AUROC per draw of positives over negatives; ties 1/2; a pair weighs the product of its two units' multiplicities.

    pos, neg: lists of (unit_column, group, value). With within_group, only pairs of equal group count.
    """
    if not pos or not neg:
        return np.full(W.shape[0], np.nan)
    pu = np.array([p[0] for p in pos])
    nu = np.array([q[0] for q in neg])
    pv = np.array([p[2] for p in pos], dtype=np.float64)
    nv = np.array([q[2] for q in neg], dtype=np.float64)
    if within_group:
        ok = np.array([p[1] for p in pos], dtype=object)[:, None] == np.array([q[1] for q in neg], dtype=object)[None, :]
    else:
        ok = np.ones((len(pos), len(neg)), dtype=bool)
    ind = ((pv[:, None] > nv[None, :]).astype(float) + 0.5 * (pv[:, None] == nv[None, :])) * ok
    U = W.shape[1]
    A = np.zeros((U, U))
    N = np.zeros((U, U))
    np.add.at(A, (np.repeat(pu, len(nu)), np.tile(nu, len(pu))), ind.ravel())
    np.add.at(N, (np.repeat(pu, len(nu)), np.tile(nu, len(pu))), ok.ravel().astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((W @ A) * W).sum(1) / ((W @ N) * W).sum(1)


def sameness(auc: np.ndarray, ndraw: int = NDRAW) -> dict:
    f = auc[1:][np.isfinite(auc[1:])]
    lo, hi = (float(np.quantile(f, 0.025)), float(np.quantile(f, 0.975))) if len(f) else (float("nan"), float("nan"))
    usable = len(f) >= 0.95 * ndraw
    return {"auroc": float(auc[0]), "ci95": [lo, hi], "nan_draws": int(ndraw - len(f)),
            "pass": bool(usable and lo >= 0.45 and hi <= 0.55)}


def q(x: np.ndarray, p: float):
    x = x[np.isfinite(x)]
    return float(np.quantile(x, p)) if len(x) else None


def family_block(est: np.ndarray, names: list, *, u_label: str, ndraw: int = NDRAW, disc=None, counts=None) -> dict:
    """Append 19 estimator on per-draw member deltas est (ndraw+1, P), percent units in the output."""
    P = est.shape[1]
    full = np.isfinite(est).all(1) if P else np.zeros(est.shape[0], bool)
    complete = int(full[1:].sum())
    out = {"complete_draws": complete, "unstable_over_1pct_nan": bool(ndraw - complete > 0.01 * ndraw), "per_probe": {}}
    for i, nm in enumerate(names):
        lo, hi = q(est[1:, i], 0.025), q(est[1:, i], 0.975)
        pp = {"delta": 100 * float(est[0, i]),
              "ci95": [100 * lo if lo is not None else None, 100 * hi if hi is not None else None],
              "U_abs_approx": 100 * max(abs(lo), abs(hi)) if lo is not None else None}
        if disc is not None:
            pp["discordant_tasks"] = int(disc[i])
        out["per_probe"][nm] = pp
    if P < 2:
        out["X_err"] = None
        out["note"] = FEW
    elif not full[0]:
        out["X_err"] = None
        out["note"] = "draw 0 undefined (empty subset): no X_err"
    elif complete < 2:
        out["X_err"] = None
        out["note"] = "fewer than 2 complete draws: no X_err"
    else:
        d = est[1:][full[1:]]
        m0, S0 = float(est[0].mean()), float(est[0].var(ddof=1))
        md, Sd = d.mean(1), d.var(1, ddof=1)
        w = (d - md[:, None]).var(0, ddof=1)
        spread2 = max(0.0, S0 - float(w.sum()) / (P - 1))
        bias2 = max(0.0, m0**2 - float(md.var(ddof=1)))
        x = 100 * math.sqrt(bias2 + spread2)
        out.update({
            "X_err": x,
            "X_err_wording": "point estimate 0 (floored)" if x == 0.0 else None,
            "bias_mean": 100 * m0,
            "bias_ci95": [100 * float(np.quantile(md, 0.025)), 100 * float(np.quantile(md, 0.975))],
            "bias2_corrected_floored": bias2 == 0.0,
            "spread": 100 * math.sqrt(spread2),
            "spread_floored": spread2 == 0.0,
            "spread_plugin_ci95_inflated_by_noise": [100 * float(np.quantile(np.sqrt(Sd), 0.025)),
                                                     100 * float(np.quantile(np.sqrt(Sd), 0.975))],
            "X_err_plugin_no_deconvolution": 100 * math.sqrt(m0**2 + S0),
            "U_approx_upper_95": 100 * float(np.quantile(np.sqrt(md**2 + Sd), 0.95)),
            "U_label": u_label,
            "fixed_family_rms": 100 * math.sqrt(bias2 + (P - 1) * spread2 / P),
            "certification": CERT_OK if complete >= 0.95 * ndraw else CERT_NO,
        })
    if counts is not None:
        out["n_cells"], out["n_mains"] = int(counts[0]), int(counts[1])
    return out


def tost_block(est_a: np.ndarray, est_b: np.ndarray, names: list, n_cells: list, n_mains: list,
               ndraw: int = NDRAW) -> dict:
    """Paired TOST at X = 5 points on diff = a - b (both per-draw member deltas on shared draws)."""
    full = np.isfinite(est_a).all(1) & np.isfinite(est_b).all(1)
    dd = (est_a - est_b)[1:][full[1:]]
    per, within, beyond = {}, [], []
    for i, nm in enumerate(names):
        x = dd[:, i]
        c90 = [100 * float(np.quantile(x, 0.05)), 100 * float(np.quantile(x, 0.95))] if len(x) else [None, None]
        c95 = [100 * float(np.quantile(x, 0.025)), 100 * float(np.quantile(x, 0.975))] if len(x) else [None, None]
        w5 = bool(len(x) and c90[0] > -5 and c90[1] < 5)
        b5 = bool(len(x) and (c95[0] > 5 or c95[1] < -5))
        per[nm] = {"diff": 100 * float(est_a[0, i] - est_b[0, i]), "ci90": c90, "ci95": c95, "within_5": w5,
                   "difference_beyond_5_established": b5}
        within.append(w5)
        beyond.append(b5)
    under = min(n_mains) < 30
    complete = int(full[1:].sum())
    if names and all(within):
        pooled = POOLED_OK
    else:
        pooled = DIFFER + (f"; {UNDER}" if under else "")
    if any(beyond):
        reading = "difference established (95% interval wholly outside +-5)"
    else:
        reading = "inconclusive" + (f"; {UNDER}" if under else "")
    return {"per_member": per, "n_cells_with_without": list(n_cells), "n_mains_with_without": list(n_mains),
            "complete_draws": complete,
            "complete_draws_warning": None if complete >= 0.95 * ndraw else "fewer than 95% complete draws",
            "pooled_label": pooled, "difference_reading": reading}


def corr_block(M: np.ndarray, names: list) -> dict:
    with np.errstate(invalid="ignore", divide="ignore"):
        cm = np.atleast_2d(np.corrcoef(M)) if len(names) > 1 else np.ones((1, 1))
    off = cm[~np.eye(len(names), dtype=bool)]
    return {"members": list(names), "matrix": [[float(x) if np.isfinite(x) else None for x in r] for r in cm],
            "mean_off_diagonal": float(np.nanmean(off)) if len(off) and np.isfinite(off).any() else None}


def comps_of(h: dict) -> dict:
    return {"sim_minus_mean_real": h["sim"] - (h["run1"] + h["run2"]) / 2, "sim_minus_run1": h["sim"] - h["run1"],
            "sim_minus_run2": h["sim"] - h["run2"], "reexecution_run2_minus_run1": h["run2"] - h["run1"]}


class Subsetter:
    """Per-draw member deltas over a subset of cells: within-main mean, then multiplicity-weighted mean over mains."""

    def __init__(self, W: np.ndarray, units: list, cells: list, main_of: dict):
        self.W = W
        self.upos = {u: i for i, u in enumerate(units)}
        self.cpos = {c: i for i, c in enumerate(cells)}
        self.main_of = main_of

    def est(self, M: np.ndarray, subset: list) -> np.ndarray:
        """M: (P, n_cells) per-cell differences in `cells` order. Returns (ndraw+1, P)."""
        mains = sorted({self.main_of[c] for c in subset})
        mpos = {m: i for i, m in enumerate(mains)}
        S = np.zeros((M.shape[0], len(mains)))
        n = np.zeros(len(mains))
        for c in subset:
            S[:, mpos[self.main_of[c]]] += M[:, self.cpos[c]]
            n[mpos[self.main_of[c]]] += 1
        A = S / np.maximum(n, 1)
        Wt = self.W[:, [self.upos[m] for m in mains]] if mains else np.zeros((self.W.shape[0], 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            return (Wt @ A.T) / Wt.sum(1, keepdims=True)


# ---------------------------------------------------------------------------------------------------------- comparator
WORDING_KEYS = {"U_label", "X_err_wording", "certification", "strata_label", "stratum_label", "pooled_label",
                "difference_reading", "note", "complete_draws_warning", "sensitivity_d_T196_only", "label", "withheld",
                "reading"}


class Comparator:
    """Walks my result against the target by key.

    Every leaf is compared exactly by type: numbers (|diff| <= atol; integers exactly), booleans, None, identifiers and
    other strings. Only strings under a descriptive-label key (WORDING_KEYS, or any value under a `labels` dict) go to
    the separate wording class; everything else counts toward the numeric verdict.
    """

    def __init__(self, atol: float = 1e-9, strict: bool = False):
        self.atol = atol
        self.strict = strict  # strict: target dict keys missing on my side are mismatches too
        self.n = {"numeric": 0, "wording": 0}
        self.mism = {"numeric": [], "wording": []}
        self.maxabs = 0.0
        self.max_at = None
        self.ok_paths = []  # leaf paths compared and equal (so "fixed" can be proven, not inferred from absence)

    def walk(self, mine, theirs, path: str, wording: bool = False):
        if isinstance(mine, dict):
            if not isinstance(theirs, dict):
                self._bad("numeric", f"{path}: target has {type(theirs).__name__}, not a dict")
                return
            for k, v in mine.items():
                if k not in theirs:
                    self._bad("numeric", f"{path}/{k}: missing in target")
                    continue
                self.walk(v, theirs[k], f"{path}/{k}", wording or k in WORDING_KEYS or k == "labels")
            if self.strict:
                for k in theirs:
                    if k not in mine:
                        self._bad("numeric", f"{path}/{k}: in the target, missing here")
            return
        if isinstance(mine, (list, tuple)):
            if not isinstance(theirs, (list, tuple)) or len(theirs) != len(mine):
                self._bad("numeric", f"{path}: list length or type differs ({len(mine)} vs "
                                     f"{len(theirs) if isinstance(theirs, (list, tuple)) else type(theirs).__name__})")
                return
            for i, (a, b) in enumerate(zip(mine, theirs)):
                self.walk(a, b, f"{path}[{i}]", wording)
            return
        cls = "wording" if wording and (isinstance(mine, str) or mine is None) and (isinstance(theirs, str) or theirs is None) \
            else "numeric"
        self.n[cls] += 1
        before = len(self.mism[cls])
        self._leaf(mine, theirs, path, cls)
        if len(self.mism[cls]) == before:
            self.ok_paths.append(path)

    def _leaf(self, mine, theirs, path, cls):
        tm, tt = _kind(mine), _kind(theirs)
        if tm != tt:
            self._bad(cls, f"{path}: type {tm} vs target {tt} (mine {mine!r}, target {theirs!r})")
            return
        if tm in ("str", "bool", "none"):
            if mine != theirs:
                self._bad(cls, f"{path}: mine {mine!r} vs target {theirs!r}")
            return
        if isinstance(mine, int) or isinstance(theirs, int):  # an integer quantity: exact
            if float(mine) != float(theirs) or not float(mine).is_integer():
                self._bad(cls, f"{path}: mine {mine!r} vs target {theirs!r} (integer quantity)")
            return
        a, b = float(mine), float(theirs)
        if not (math.isfinite(a) and math.isfinite(b)):
            if not (math.isnan(a) and math.isnan(b)) and a != b:
                self._bad(cls, f"{path}: mine {a} vs target {b} (non-finite)")
            return
        d = abs(a - b)
        if d > self.maxabs:
            self.maxabs, self.max_at = d, path
        if d > self.atol:
            self._bad(cls, f"{path}: mine {a!r} vs target {b!r} (|diff| {d:.3g})")

    def _bad(self, cls, msg):
        self.mism[cls].append(msg)

    def summary(self, cap: int | None = 500) -> dict:
        return {"compared": dict(self.n), "n_mismatch": {k: len(v) for k, v in self.mism.items()},
                "max_abs_diff": self.maxabs, "max_abs_diff_at": self.max_at,
                "mismatches": {k: (v if cap is None else v[:cap]) for k, v in self.mism.items()},
                "mismatch_paths": {k: [m.split(": ")[0] for m in v] for k, v in self.mism.items()},
                **({"matched_paths": list(self.ok_paths)} if cap is None else {})}


def _kind(x) -> str:
    if x is None:
        return "none"
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, (int, float)):
        return "number"
    if isinstance(x, str):
        return "str"
    return type(x).__name__


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path


def uncovered(mine, theirs, path="") -> list:
    """Target leaves that I do not produce (coverage, not mismatches)."""
    out = []
    if isinstance(theirs, dict):
        if not isinstance(mine, dict):
            return list(leaves(theirs, path))
        for k, v in theirs.items():
            out += uncovered(mine.get(k), v, f"{path}/{k}") if k in mine else list(leaves(v, f"{path}/{k}"))
    elif isinstance(theirs, (list, tuple)) and isinstance(mine, (list, tuple)) and len(mine) == len(theirs):
        for i, (a, b) in enumerate(zip(mine, theirs)):
            out += uncovered(a, b, f"{path}[{i}]")
    return out


def negative_test(mine_by_cell: dict, plants: list) -> dict:
    """Each plant is applied alone to a target equal to MY result (zero mismatches by construction) and must be flagged
    exactly once, at its own path. Kinds: float (+1e-8), bool (flip), int (+1), str (number -> string), drop (delete
    the key), extend (append to a list)."""
    import copy

    base = Comparator()
    for cell, m in mine_by_cell.items():
        base.walk(m, m, cell)
    out = {"baseline_mismatches": len(base.mism["numeric"]) + len(base.mism["wording"]), "plants": []}
    for path, kind in plants:
        pert = copy.deepcopy(mine_by_cell)
        node = pert
        for k in path[:-1]:
            node = node[k]
        last = path[-1]
        if kind == "float":
            node[last] = node[last] + 1e-8
        elif kind == "bool":
            node[last] = not node[last]
        elif kind == "int":
            node[last] = node[last] + 1
        elif kind == "str":
            node[last] = str(node[last])
        elif kind == "drop":
            del node[last]
        elif kind == "extend":
            node[last] = list(node[last]) + [0.0]
        c = Comparator()
        for cell, m in mine_by_cell.items():
            c.walk(m, pert[cell], cell)
        flags = c.mism["numeric"] + c.mism["wording"]
        want = "/".join(map(str, path))
        ok = len(flags) == 1 and flags[0].startswith(want)
        out["plants"].append({"path": want, "kind": kind, "flagged": len(flags), "at_path": ok})
    out["pass"] = out["baseline_mismatches"] == 0 and all(p["at_path"] for p in out["plants"])
    return out
