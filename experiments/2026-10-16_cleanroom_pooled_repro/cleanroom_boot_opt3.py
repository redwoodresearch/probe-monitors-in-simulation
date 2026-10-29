"""Option (3): clean-room bootstrap of route C's interval ends (a1; PRESTATEMENT_OPT3.md @ 3c95cb9d1 + addenda 1-8f in opt3/). r2.7, after
Codex r2 and 09's rulings on R1 and Codex r2 (addendum 8), BLAS threads (8a), C4 before any verdict (8b), Codex / R1 on r2.2 (8c),
the main run's KeyError (the co-headline key shape) with self-locating STOPs (8d), one shared node path with a resolve gate (8e), and
the rerun's FileNotFoundError: three unshipped /cfg inputs, an inputs gate and stop_file (8f).

Re-generates the union-stream draws and the per-draw pooled estimator in a second implementation written from registered text, then
compares the 300 in-scope nodes' interval ends (A 60, B 240) with the producer records by key, inside Modal. Values stay on the output
Volume; the laptop gets the exact-schema summary: booleans, integer counts, fixed codes and validated identifiers (no free text).

Shape (§6): ONE driver (cpu 1, 2 GiB) started by one .spawn under `modal run --detach`. It exclusive-creates a spawn marker
(idempotency), then runs exactly two workers in parallel, A and B (cpu 16, 64 GiB each). It validates the summary against the exact
schema before writing, writes a STOP summary on every other exit, and the write itself never raises. Each worker:
  1. code pins (the reducer blob, the pins config, q1_eval.py, the node list) and the inputs gate (every /cfg file the reducer opens,
     sha-checked, 8f), then the self-tests against the frozen predictions
     (opt3/PREDICTIONS_r2_8.json) BEFORE any producer read: D5, verdicts, the comparison core, C0, C2, calib layouts, gap identifiers,
     the schema, stop codes, the memory projection, the main-path D7 fixture, C5 (300 trials, the addendum 7 / 8 guards);
  2. input pins: the scope file, 2e's JSONs, A's rc1a + D5 parse-readiness; the producer record;
  3. manifests: meta.merge, the calibration npz shas vs the record's own field, the per-draw npz pin; the node-list rebuild; the resolve
     gate (every node through the shared path to a five-key leaf, 8e); then
     C4's removed-key fixtures, BEFORE any verdict (addendum 8b); C2 on A's fixed-mode delta_full with A's own index rule (8c);
  4. the draw-0 inputs through the reducer's own functions; engine-vs-reducer draw-0 identity (tau 1e-9, TPR 1e-12, no flip term);
  5. C0 on draw 0, BEFORE the draws: one unit scale or NOT LIKE-FOR-LIKE;
  6. D1: candidates #1..#12 on draws 1..20 (B: D8's readings and the fixed tau too); first match wins, no switching;
  7. the 200-draw pilot, checked during the run for time and memory; then draws 1..10,000; per-node NaN-reason counts;
  8. C1 (calibration npz, strict per-quantity layouts; every gap member required), intervals (>= 9,500 finite draws), the
     comparison (EXACT tier; MC diagnostic otherwise; record counts 9,500 <= finite <= draws == 10,000 per node),
     the values saved, then D5 into the verdict (flip-sensitive elements UNRESOLVED, 8c).
"""

import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------- one BLAS / OpenMP thread per process (09, addendum 8a)
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def _env_code(v):
    """An env value as an allow-listed small int: unset -1, a 1-4 digit integer as itself, anything else -2."""
    return -1 if v is None else int(v) if re.fullmatch(r"[0-9]{1,4}", v) else -2


THREAD_ENV_PRE = {k: _env_code(os.environ.get(k)) for k in THREAD_VARS}  # as the runtime set them, before the update
if not modal.is_local():  # in the container, before any numpy import in this process (forked children inherit it)
    os.environ.update({k: "1" for k in THREAD_VARS})

# ---------------------------------------------------------------- names, pins and constants
APP = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
SCORES = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
POOLED = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
HF = "probe-shift_aswin-surya_hf-cache"
REDUCER = ("cleanroom_reduce_v2.py", "991a34be9f290797bfbf89c543341e59990b53a72ec11bf045ec6d426cf21f30")  # reviewed blob (run_routec.sh)
PINS = {"A": "config/routec_A18_pins.json", "B": "config/routec_BS18_pins.json"}  # the configs whose draw-0 points passed (A18, BS18)
PINS_SHA = {"A": "5195c68aeef459230f27b1a2580e8bc30ac3552b51d499f9f7465beb4939c837",  # Codex r1 #4: immutable pins of the configs
            "B": "8f49555dead5766d5305dc69b5e0de0c9c091f1e13954e6a19a0f4ad73f8bc52"}
Q1_EVAL_SHA = "2e313de531cc72c5fabd4b256449a6d4aeca347a1be1b8decb10855aa7ddedb1"  # src/probe_shift/q1_eval.py (blob 0a08acf4)
A_RECORD = ("routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz",
            "54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a")  # decompressed; C2 runs on A's fixed arrays in both workers
MIN_FINITE = 9500  # R:4935 / PT:37: the >= 9,500 complete-draw gate
NODES = {"A": ("opt3/opt3_nodes_A_rlext.txt", "a5c12f43cfa9d2358ea189409701a2f6067f79870a524f08715b440495c1b5d7"),
         "B": ("opt3/opt3_nodes_fig4.txt", "7d984dcf9107a43db4ae9188aaab693280f55502b011e445247e718d95a1b35c")}
SCOPE = ("a1_interval_scope_rlext_20261025.jsonl", "4dbffaa36a5569b3f76dae0f701558357539c71bec2288c75849bc0e9d29e41b")
CALIB_JSON = ("opt3_calibration_paths_AB.json", "c1ea94312e9b5ce07a4d7f24f54ed0f28e7a4c83d9c45aa982deabe44480dbeb")
FIGS = {"fig1_main": ("fig1_main_figs156_routecA_source.json", "4f31f71b9d8b92fe25845d715c91d1dcfbd9910fbd9c11f8ab185b3cb97c74e1"),
        "fig1_main_R": ("fig1_main_R_figs156_routecA_source.json", "69b30ddcd3fcd8db7d7d634e4a258849848ca00168792c68ccb25af202ed6536"),
        "fig4": ("fig4_pooled_fig4_routecB_source.json", "0de398ad8a6e6115a67e473c42a772f4def525dae11a481e23654e4d504e9e5d")}  # @ f35c8a0dc
RC1A = ("slots_routeC_writing_rc1a.json", "3d7b036b445dca6b467ce709071c7003ab94fe3b854c0bc3d0f82ec4bedc9238")  # @ 6bee032a9 (D5's slot strings)
PERDRAW_JSON = ("opt3_perdraw_npz_AB.json", "28c8302c7c2128e1d802ba9e7684d45fa25dddb631104a045fceb9c801aab111")  # 2e's strings-only extraction
GAP_NPZ = {"A": ("xfer/fpr507prep/alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327_perdraw.npz",
                 "5e6cd10e382d8b03d409982c67efd72512176d7e95d2f3c94752d2b90e5650d1"),
           "B": ("xfer/fpr507prep/alltasks_CRD_FULL_r2_allorigins_sweep15_pooledworld_scored_e12_2026-10-19_180652_perdraw.npz",
                 "8a2a2afda1ff5271695191be04911fcd9ecb4bf7199d5715fe6a526c9b2ffaa3")}  # both = 2e's opt3_perdraw_npz_AB.json (28c8302c)
COMBO = "pooled_all_origins"
N_DRAWS, ID_DRAWS, PILOT_DRAWS = 10000, 20, 200  # TB:16 / R:3357; D1 identification on draws 1..20; §6 pilot
SEED, CAL_SEED = [20261015, 40], [20261014, 31]  # R:3856 / meta.merge.seed; R:2194, TB:25
TIMEOUT, PILOT_FRACTION = 3600, 0.8
MEM_LIMIT_MIB, MEM_FRACTION = 65536, 0.8  # 09 (R1 #19): STOP if the pilot's memory projection > 0.8 x 64 GiB
CHUNK = 50  # draws per task, the pilot's and the full run's alike (addendum 8 §3.7)
RTOL = 1e-9  # addendum 5: 1e-9 * max(1, |x|)
WORKERS = 16
D1_LIST = [(i, s, p) for i in ("I1", "I2") for s in ("S1", "S2") for p in ("Pa", "Pb", "Pc")]  # PRESTATEMENT_OPT3 §1 D1-list, #1..#12
GROUPS = {"headline_six_full": (0, 1, 2, 3, 4, 5), "auroc06_subset": (0, 1, 2, 5)}  # R:4402-4406, R:4843 (as the reducer's GROUPS)
SIX = ["sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48"]

app = modal.App(APP)
OUT_VOL = modal.Volume.from_name(APP, create_if_missing=True, version=2)
VOLS = {"/s": modal.Volume.from_name(SCORES, version=2).read_only(), "/p": modal.Volume.from_name(POOLED, version=2).read_only(),
        "/a": modal.Volume.from_name(ACTS, version=2).read_only(), "/hf": modal.Volume.from_name(HF).read_only(), "/o": OUT_VOL}
base = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "threadpoolctl==3.6.0", "tokenizers==0.22.1")
        .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}))  # Codex r1 (b): one BLAS thread per process
images = {"A": base, "B": base, "driver": base}
if modal.is_local():  # /cfg = the reducer's pinned inputs (CR_CFG); /opt3in = OPT3_CFG (scope file, 2e's JSONs, figure sources, rc1a)
    _here, _root = Path(__file__).parent, Path(__file__).resolve().parents[2]
    for rec in ("A", "B"):  # one image per record: the reducer reads its config at the fixed path /run.json
        images[rec] = (base.add_local_dir(os.environ["CR_CFG"], "/cfg").add_local_dir(os.environ["OPT3_CFG"], "/opt3in")
                       .add_local_dir(str(_here / "opt3"), "/opt3").add_local_file(str(_here / PINS[rec]), "/run.json")
                       .add_local_file(str(_here / REDUCER[0]), f"/root/{REDUCER[0]}")
                       .add_local_file(str(_root / "src/probe_shift/q1_eval.py"), "/root/q1_eval.py"))
    images["driver"] = base.add_local_dir(str(_here / "opt3"), "/opt3")  # the node lists, for the summary's membership checks


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tol(x):
    return RTOL * max(1.0, abs(x))


class Unresolved(Exception):
    pass


# Fixed STOP codes (Codex r2 #1; 09): the summary carries a code from this list, never free text. The .jq gate holds the same lists.
STOP_BASES = (
    "not started", "unknown stop code", "reducer blob", "pins config / q1_eval sha", "node list sha", "predictions file sha",
    "self-tests differ from the frozen predictions", "pin scope", "pin calib_json", "pin perdraw_json", "pin fig1_main",
    "pin fig1_main_R", "pin fig4", "pin rc1a", "D5 not ready: key_path form", "D5 not ready: slot missing",
    "D5 not ready: slot not a string", "D5 not ready: aggregator unknown", "D5 not ready: node not in A's list",
    "D5 not ready: no halfwidth elements", "record sha", "meta.merge layout", "calibration npz sha / manifest", "per-draw npz pin",
    "node list rebuild", "scope a_node form", "unparsed node", "a node level is not among the record's levels",
    "PLUMBING: pool / cohort mains outside meta.merge.mains", "engine draw 0 differs from compute_label", "unknown call pattern",
    "pilot projection over 0.8 x timeout", "pilot memory projection over 0.8 x 64 GiB", "calibration member missing",
    "calibration layout", "C1: a stored gap member is missing", "C1: stored gap array length vs the identified index rule",
    "C2: A record sha", "C2: A per-draw npz sha", "C2: expected members missing", "C2: stored array layout",
    "C2: no percentile method reproduces A's fixed-mode delta_full ci95", "fixture key absent", "C4 fixture did not stop",
    "report schema", "past the launch deadline", "worker report schema", "spawn marker exists (idempotency)", "summary schema",
    "BLAS threads not 1", "no BLAS pool found", "allow-list sha", "D5 not ready: slot list", "a record level is not in the pinned level list",
    "resolve:catch", "resolve:coheadline", "resolve:gap", "input missing", "input sha", "input pinned twice", "input missing: catalogue",
    "input missing: manifest")
EXC_NAMES = (
    "AssertionError", "AttributeError", "BadZipFile", "BrokenProcessPool", "ConnectionError", "EOFError", "ExecutionError",
    "FileExistsError", "FileNotFoundError", "FloatingPointError", "FunctionTimeoutError", "ImportError", "IndexError",
    "InputCancellation", "InternalFailure", "JSONDecodeError", "KeyError", "LinAlgError", "MemoryError", "ModuleNotFoundError",
    "NotADirectoryError", "OSError", "OverflowError", "PermissionError", "RecursionError", "RemoteError", "RuntimeError",
    "TimeoutError", "TypeError", "UnicodeDecodeError", "ValueError", "ZeroDivisionError", "other")
STOP_CODES = tuple(f"STOP: {b}" for b in STOP_BASES) + tuple(f"STOP: {w}exception {n}" for w in ("", "worker ", "driver ")
                                                             for n in EXC_NAMES)


class Stop(Exception):
    def __init__(self, code, ids=(), input_file=None):
        super().__init__(code)
        self.code, self.ids, self.input_file = code, [str(x) for x in ids], input_file


def stop(base, ids=(), input_file=None):
    """Raise a fixed STOP code; an unlisted base raises 'STOP: unknown stop code' instead (no free text can be built here). input_file:
    the inputs gate's file name (addendum 8f), reported only if it is one of INPUT_FILES."""
    code = f"STOP: {base}"
    raise Stop(code if code in STOP_CODES else "STOP: unknown stop code", ids, input_file)


def exc_code(where, e):
    """'STOP: [worker |driver ]exception <Name>' with Name from EXC_NAMES, else 'other'. `where` is '', 'worker ' or 'driver '."""
    n = type(e).__name__
    return f"STOP: {where}exception {n if n in EXC_NAMES else 'other'}"


BOOT_NLINES = Path(__file__).read_text().count("\n")  # this file's line count (= `wc -l` of its blob): stop_line's bound for "boot"
REDUCER_NLINES = 1549  # cleanroom_reduce_v2.py, the pinned reducer (blob b8c136c8, sha256 991a34be...): stop_line's bound for "reducer"
STOP_FILES = {"cleanroom_boot_opt3.py": "boot", "cleanroom_reduce_v2.py": "reducer"}  # a1's own files on the call path (addendum 8f)
NLINES = {"boot": BOOT_NLINES, "reducer": REDUCER_NLINES}


def _stop_where(e):
    """09 (addenda 8d, 8f): (file id, line) of the deepest traceback frame of `e` in any of a1's own files (STOP_FILES), skipping stop()
    itself so a fixed STOP points at its call site; (None, None) if none. Integers and fixed ids only: no message text."""
    import traceback

    fr = [(STOP_FILES[Path(f.filename).name], f.lineno) for f in traceback.extract_tb(e.__traceback__)
          if Path(f.filename).name in STOP_FILES and f.name != "stop"]
    return fr[-1] if fr else (None, None)


def stop_code_at(code, stg, rec):
    """09 (addendum 8d): a worker STOP code with the stage that was running, ' @<stage>': the first stage of the record's sequence
    (STAGES; B's ends at 'values') not yet completed. No suffix once every stage is done."""
    seq = STAGES if rec == "A" else STAGES[:STAGES.index("values") + 1]
    left = [s for s in seq if s not in stg]
    return f"{code} @{left[0]}" if left else code


# ---------------------------------------------------------------- D1: multiplicity generators (PRESTATEMENT_OPT3 §1, D1-list)
def strata_order(mains, env_of, s_rule):
    """S1: environments in sorted string order; S2: in order of first appearance walking `mains` in stored (sorted) order."""
    if s_rule == "S1":
        envs = sorted(set(env_of[m] for m in mains))
    else:
        envs = []
        for m in mains:
            if env_of[m] not in envs:
                envs.append(env_of[m])
    return [(e, [i for i, m in enumerate(mains) if env_of[m] == e]) for e in envs]  # mains within a stratum in stored order


def multiplicities(mains, env_of, s_rule, p_rule, seed, n_draws):
    """Rows 0..n_draws; row 0 all ones (TB:20). One default_rng(seed); one stream over all strata in S order."""
    import numpy as np

    rng = np.random.default_rng(seed)
    st = strata_order(mains, env_of, s_rule)
    M = np.zeros((n_draws + 1, len(mains)), dtype=np.int32)
    M[0] = 1
    if p_rule == "Pa":  # per stratum, multinomial(n_s, uniform, size=n_draws) gives draws 1..n_draws (XS:806-812)
        for _e, idx in st:
            n = len(idx)
            M[1:, idx] = rng.multinomial(n, [1.0 / n] * n, size=n_draws)
    elif p_rule == "Pb":  # per draw, per stratum, multinomial(n_s, uniform)
        for d in range(1, n_draws + 1):
            for _e, idx in st:
                n = len(idx)
                M[d, idx] = rng.multinomial(n, [1.0 / n] * n)
    elif p_rule == "Pc":  # per draw, per stratum, bincount(integers(0, n_s, n_s))
        for d in range(1, n_draws + 1):
            for _e, idx in st:
                n = len(idx)
                M[d, idx] = np.bincount(rng.integers(0, n, n), minlength=n)
    else:
        stop("unknown call pattern")
    assert np.all(M[1:].sum(1) == len(mains)), "each draw resamples every stratum's size"
    return M


# ---------------------------------------------------------------- the per-draw weighted engine (registered rules, §3)
def side_data(R, pool_runs, extra_units, main_idx, single):
    """Pool units of one side (B or S) as arrays, from the reducer's own units(): score, fold (-9 single / 0..4 / -1), sbm, the unit's
    universe main index, its run index; plus per fold and main the sufficient statistics for weighted moments. extra_units: folds
    of attack runs that need this side's transform (as compute_label's `needed`)."""
    import numpy as np

    s, f, sbm, (ks, _us) = R.units(pool_runs)
    keys = [str(k) for k in ks]
    run_of = {}
    run_idx = np.array([run_of.setdefault(k, len(run_of)) for k in keys], dtype=np.int64)
    by_key = {x["key"]: x for x in pool_runs}
    run_main = np.array([main_idx[by_key[k]["main"]] for k in run_of], dtype=np.int64)
    d = {"s": np.asarray(s, float), "f": np.asarray(f, int), "sbm": sbm, "run_idx": run_idx, "run_main": run_main, "ukeys": keys, "uids": np.asarray(_us),
         "unit_main": run_main[run_idx] if len(run_idx) else np.zeros(0, np.int64), "single": single, "n_runs": len(run_of)}
    if single:
        d["needed"] = set()
        return d
    needed = set(np.unique(d["f"]).tolist()) | set(extra_units)
    if -1 in needed:
        needed |= {0, 1, 2, 3, 4}
    d["needed"] = needed
    U = len(main_idx)
    d["fold_stats"] = {}
    for k in sorted(x for x in needed if x >= 0):
        sel = d["f"] == k
        c = float(np.mean(d["s"][sel])) if sel.any() else 0.0  # centring constant (draw-0 mean) for numerical stability
        x = d["s"][sel] - c
        um = d["unit_main"][sel]
        n = np.bincount(um, minlength=U).astype(float)
        s1 = np.bincount(um, weights=x, minlength=U)
        s2 = np.bincount(um, weights=x * x, minlength=U)
        runs_k = np.zeros(U)
        for r in np.unique(d["run_idx"][sel]):
            runs_k[run_main[r]] += 1
        d["fold_stats"][k] = (c, n, s1, s2, runs_k)
    if -1 in needed:
        sel = d["f"] == -1
        d["f1_sel"] = sel
        d["f1_sbm"] = sbm[sel] if sel.any() else np.zeros((0, 5))
        d["f1_main"] = d["unit_main"][sel]
        d["f1_run"] = d["run_idx"][sel]
    return d


def weighted_moments(d, Mf):
    """Per draw (rows of Mf, float), multiplicity-weighted fold moments, ddof 0 (TB:31, R:3857), with validity: >= 2 distinct runs of
    positive weight, a finite SD > 0 (R:2167; D7). Also 507 l.139's '>= 2 units' validity, reported beside (pre-statement D7 row).
    Returns {fold: (mu, sd, ok, ok_units)} arrays over draws, and fold -1's (mu0, sd0, ok0, ok0_units)."""
    import numpy as np

    out = {}
    for k, (c, n, s1, s2, runs_k) in d["fold_stats"].items():
        W = Mf @ n
        A1 = Mf @ s1
        A2 = Mf @ s2
        with np.errstate(divide="ignore", invalid="ignore"):
            m1 = A1 / W
            var = np.maximum(A2 / W - m1 * m1, 0.0)
        mu, sd = c + m1, np.sqrt(var)
        pos = (Mf > 0).astype(float)
        fin = (W > 0) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
        out[k] = (mu, sd, fin & (pos @ runs_k >= 2), fin & (pos @ n >= 2))
    if "f1_sel" in d:
        out[-1] = f1_moments(d, Mf, out)
    return out


def f1_coeffs(mom, D):
    """m_u = (1/5) sum_k (sbm_uk - mu_k) / sd_k = sbm_u @ a + b per draw (SPEC_READING §2; R:3857 re-standardization)."""
    import numpy as np

    a = np.stack([1.0 / (5.0 * mom[k][1]) for k in range(5)], axis=1) if D else np.zeros((0, 5))
    b = -np.sum(np.stack([mom[k][0] / (5.0 * mom[k][1]) for k in range(5)], axis=1), axis=1)
    return a, b


def f1_moments(d, Mf, mom):
    import numpy as np

    D = Mf.shape[0]
    ok5 = np.all(np.stack([mom[k][2] for k in range(5)]), axis=0)
    ok5u = np.all(np.stack([mom[k][3] for k in range(5)]), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = f1_coeffs(mom, D)
        m = a @ d["f1_sbm"].T + b[:, None]  # D x n1
        w = Mf[:, d["f1_main"]]
        W = w.sum(1)
        mu0 = (w * m).sum(1) / W
        sd0 = np.sqrt(np.maximum((w * (m - mu0[:, None]) ** 2).sum(1) / W, 0.0))
    runs = np.zeros((D, d["n_runs"]))
    if len(d["f1_run"]):
        np.add.at(runs.T, d["f1_run"], (w > 0).T.astype(float))
    fin = (W > 0) & np.isfinite(mu0) & np.isfinite(sd0) & (sd0 > 0)
    return (mu0, sd0, ok5 & fin & ((runs > 0).sum(1) >= 2), ok5u & fin & ((w > 0).sum(1) >= 2))


def transform_draw(d, mom, j):
    """z of every pool unit at draw j (row j of the moment arrays). Never None (09, R1 #4): NaN-filled, so a unit of an invalid or
    unmapped fold stays NaN; callers use the validity masks and only threshold valid draws."""
    import numpy as np

    if d["single"]:
        return d["s"]
    z = np.full(len(d["s"]), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(5):
            sel = d["f"] == k
            if sel.any():
                z[sel] = (d["s"][sel] - mom[k][0][j]) / mom[k][1][j]
        if "f1_sel" in d and d["f1_sel"].any():
            a = np.array([1.0 / (5.0 * mom[k][1][j]) for k in range(5)])
            b = -sum(mom[k][0][j] / (5.0 * mom[k][1][j]) for k in range(5))
            z[d["f1_sel"]] = (d["f1_sbm"] @ a + b - mom[-1][0][j]) / mom[-1][1][j]
    return z


def weighted_tau(z, w, levels):
    """R:3342 / TB:30 weighted: stable mergesort descending; k_d = floor(alpha * n_d + 1e-9), n_d = sum of unit multiplicities;
    T = the first sorted value whose cumulative weight exceeds k_d; k_d = 0 is non-estimable (D6) -> NaN. Returns (values, the
    selected unit's index per level, -1 where NaN): the engine's ACTUAL selection, which C5 checks (Codex r2 #6)."""
    import numpy as np

    order = np.argsort(-z, kind="stable")
    zs, cw = z[order], np.cumsum(w[order])
    n_d = float(cw[-1]) if len(cw) else 0.0
    vals, idx = [], []
    for lvl in levels:
        k = int(np.floor(float(lvl) * n_d + 1e-9))
        i = int(np.searchsorted(cw, k, side="right")) if k >= 1 and n_d > 0 else len(zs)
        vals.append(float(zs[i]) if i < len(zs) else np.nan)
        idx.append(int(order[i]) if i < len(zs) else -1)
    return vals, idx


# ---------------------------------------------------------------- draw-0 inputs through the reducer's own functions (C0's basis)
# ---------------------------------------------------------------- the inputs gate (09, addendum 8f): every /cfg file the reducer opens
INPUT_FILES = (  # the 27 /cfg basenames this run's reducer path opens (A18 / BS18 files[] union + 2 S_old overrides + the pilot registry)
    "E12_BLOCK_2026-10-16_092322.json",
    "INPUTS_FINAL_V2_FROZEN_2026-10-16_044120.json",
    "INPUTS_FULL_2026-10-17_233212.json",
    "authorized_overlaps_routec_2026-10-18_123612.json",
    "exclusions_2026-10-16_102159.json",
    "new_mains_cells_v1.json",
    "nofilter_exclusions_2026-10-16_012305.json",
    "override_S_new25f_C_2026-10-16_045304_gated.json",
    "override_S_new25f_R_2026-10-16_045058_gated.json",
    "override_S_new_C_2026-10-16_104513_gated.json",
    "override_S_new_R_2026-10-16_104459_gated.json",
    "override_S_newmains_C_2026-10-16_053303_gated.json",
    "override_S_newmains_R_2026-10-16_053350_gated.json",
    "override_S_old_C_2026-10-17_151056_gated.json",
    "override_S_old_R_2026-10-17_151100_gated.json",
    "override_S_paper493_C_2026-10-16_101944_gated.json",
    "override_S_paper493_R_2026-10-16_102114_gated.json",
    "override_pools_ep2_C_2026-10-16_053038_gated.json",
    "override_pools_ep2_R_2026-10-16_052820_gated.json",
    "own_all_rs_folds_2026-10-16.json",
    "pilot_tasks_amendment9_all196.json",
    "routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz",
    "routec_B_alltasks_CRD_FULL_r2_allorigins_sweep15_pooledworld_scored_e12_2026-10-19_180652.json.gz",
    "runs.json",
    "scaleup_tasks_plus311_2026-10-14_125135.json",
    "simh_counted_keys_pilot_v3.json",
    "sp493_facts_2026-10-15_165730.json",
)
CFG_KEYS = ("v2", "e12", "excl", "nofilter", "S_old", "S_new", "S_newmains", "sp493", "r36", "oars")  # the reducer's cfg(k) on this path


def reducer_inputs(run, full, IN):
    """Every /cfg file the reducer opens on this path, from its own config references, with the sha256 each pins: files[] (their sha),
    the cfg(k) inputs (each must be among files[]), INPUTS_FULL's overrides (their sha) and the s_old_simh_extra_roots registry of each
    of the run's extra_root_worlds (its sha). Returns {basename: sha256}; a name pinned twice with two shas, or a cfg(k) file outside
    files[], STOPs naming it."""
    want = {}

    def add(name, h):
        if want.setdefault(name, h) != h:
            stop("input pinned twice", input_file=name)

    for h, _c, p in run["files"]:
        add(Path(p).name, h)
    for k in CFG_KEYS:
        if IN[k] not in want:
            stop("input missing", input_file=IN[k])
    if full is not None:
        for o in full["overrides"].values():
            add(Path(o["path"]).name, o["sha256"])
        for root in run.get("extra_root_worlds") or {}:
            reg = full["s_old_simh_extra_roots"][root]
            add(Path(reg["path"]).name, reg["sha256"])
    return want


def inputs_gate(run, IN, cfg_dir=Path("/cfg"), pool_dir=Path("/p")):
    """09 (addendum 8f): BEFORE the self-tests and any record read, every /cfg file the reducer will open exists with its pinned sha (in
    sorted order: the first failure STOPs naming it), INPUTS_FULL read only after it passed as one of files[]; then the /p catalogue and
    manifest exist (load_inputs checks their sha). Returns the number of /cfg files checked."""
    def check(want):
        for name in sorted(want):
            p = cfg_dir / name
            if not p.is_file():
                stop("input missing", input_file=name)
            if file_sha(p) != want[name]:
                stop("input sha", input_file=name)

    first = reducer_inputs(run, None, IN)
    check(first)
    if run["inputs_full"] not in first:  # the reducer asserts INPUTS_FULL is a pinned file; read it only once it passed as one
        stop("input missing", input_file=run["inputs_full"])
    full = json.loads((cfg_dir / run["inputs_full"]).read_bytes())
    want = reducer_inputs(run, full, IN)
    check({k: v for k, v in want.items() if k not in first})
    for key in ("catalogue", "manifest"):
        if not (pool_dir / run[key]["path"]).is_file():
            stop(f"input missing: {key}")
    return len(want)


def load_labels(R, want):
    """The reducer's run_stage('compute') preamble and per-label loop, unchanged in substance (membership gates included), for the
    label x readout pairs in `want` only. Returns {label: dict(run, pool, used, single, lr, d0)} with d0 = compute_label(..., ext=True)."""
    from concurrent.futures import ProcessPoolExecutor


    run_cfg, full, ovs, cat, man = R.load_inputs()
    R.LEVELS, R.CELL_LEVELS = [str(x) for x in run_cfg["levels"]], [str(x) for x in run_cfg["cell_levels"]]
    assert R.LEVELS and len(set(R.LEVELS)) == len(R.LEVELS) and set(R.CELL_LEVELS) <= set(R.LEVELS), "config levels / cell_levels"
    assert full["e12"]["sha256"] == R.sha(R.cfg("e12").read_bytes()), "E12 block is not the one INPUTS_FULL pins"
    roles = {r for v in cat["runs"].values() for r in v["roles"]}  # R1 #16: the reducer's compute-stage catalogue asserts
    assert roles <= {"A", "B", "S"}, "catalogue roles"
    hb, hs, ha = ({k for k, v in cat["runs"].items() if r in v["roles"]} for r in ("B", "S", "A"))
    auth = {o["key"] for o in R.jload(Path("/cfg", run_cfg["authorized_overlaps"]["name"]))["authorized_overlaps"]}
    assert not (hb & hs) and not (ha & hs) and not ((ha & hb) - auth), "catalogue overlaps"
    fx = R.fixed_ids()
    six = R.probes(full)
    allp = six + R.sensitivity(full, run_cfg.get("sensitivity_labels", []))
    cohort, main, fmaps, cohort_chk = R.cells_and_maps(full)
    assert not any(cohort_chk["vs_nofilter"].values()) and not cohort_chk["vs_inputs_final_v2_sp493"], "cohort cross-check"
    ns = R.namespaces(R.jload(R.cfg("e12")), R.jload(R.cfg("v2")), run_cfg.get("extra_root_worlds"))
    p_root, p_v, p_want = R.pilot_registry(run_cfg, full)
    cstaged, jobs = R.c_staged_map(ovs, ns, {p[2]["C"] for p in allp[len(six):]})
    with ProcessPoolExecutor(8, initializer=_child_init) as px:  # one BLAS thread in each pool process too (addendum 8a)
        got = list(px.map(_staged_expect, [{"path": p, "sha": s_, "fixed": fx["ids"]} for p, s_ in sorted(jobs.items())]))
    _blas_gate([b for _r, b in got])
    expect = dict(r for r, _b in got)
    nf_zero = {k for o in R.jload(R.cfg("nofilter"))["origins"].values() for k in (o.get("structural_zero_runs") or {})}
    sz = R.jload(R.cfg("v2"))["extensions"]["S_paper493"]["structural_zero_keys"]
    known_zero = {ro: nf_zero | set(sz[ro]) for ro in ("C", "R")}
    out = {}
    for name, fm, files in allp:
        for ro in ("C", "R"):
            label = f"{name} {ro}"
            if label not in want:
                continue
            family = R.C_FAMILY[files[ro].split("/")[0]] if ro == "C" else None
            entries, folders_without, skipped = R.file_entries(ovs, ro, files[ro], ns, cstaged)
            rows, worlds, alias = R.load_rows(entries, expect, family)
            run, pool, rep = R.resolve(rows, worlds, entries, cohort, main, fmaps.get(fm), cat)
            used, dropped = R.cohort_used(run, cohort)
            lr = gate_inputs(R, label, entries, rows, worlds, run, pool, used, rep, alias, main, family, expect, run_cfg,
                             p_root, p_v, p_want, man)
            R.gate(label, lr, pool, lr["_me"], known_zero)  # the reducer's compute-stage STOPs, unchanged
            d0 = R.compute_label(run, pool, used, main, fm is None, ext=True, struct=lr.get("structural_no_row"))
            out[label] = {"run": run, "pool": pool, "used": used, "single": fm is None, "d0": d0, "main": main}
            del rows
    missing = sorted(set(want) - set(out))
    assert not missing, f"labels not built: {missing}"
    return out, run_cfg


def gate_inputs(R, label, entries, rows, worlds, run, pool, used, rep, alias, main, family, expect, run_cfg, p_root, p_v, p_want, man):
    """The `lr` fields the reducer's gate() reads, built exactly as run_stage builds them (cleanroom_reduce_v2.py L816-866)."""
    mk = alias["c_units"].pop("mismatch_keys")
    if family:
        lead = {f"{a}|{c}" for _, c in used for a in R.ARMS} | {k for k, x in pool.items() if x["r"] is not None}
        alias["c_units"].update(n_mismatch_lead=len(mk & lead), n_mismatch_nonlead=len(mk - lead))
        lead_zero = {f"{a}|{c}" for _, c in used for a in R.ARMS if run[f"{a}|{c}"]["zero"]} | {k for k, x in pool.items() if x["zero"]}
        alias["c_units"].update(R.zero_unit_check(lead_zero, R.zero_row_index(entries, worlds, main), expect, family))
    attack_worlds = sorted({w for e in entries if e["kind"] == "arm" and e["folder"].endswith(("__attack", "__sim_attack"))
                            for w in worlds[e["path"]]})
    lr = {"c_units": alias["c_units"], "alias": alias, "attack_worlds": attack_worlds, "n_unlocated": len(rep["unlocated"]),
          "n_attack_not_in_own_set": len(rep["attack_not_in_own_set"]), "zero_unit_attack": rep["zero_unit_attack"],
          "zero_unit_pool": rep["zero_unit_pool"], "pool_main_vs_task_main": rep["pool_main_vs_task_main"][:20],
          "fold_col_mismatch_mapped": rep["fold_col_mismatch_mapped"], "fold_col_not_minus1": rep["fold_col_not_minus1"],
          "score_vs_sbm_fold": [rep["score_vs_sbm_fold_checked"], rep["score_vs_sbm_fold_mismatch"]],
          "missing_pool": rep["missing"]["pool"], "cells_used_by_set": {s: sum(1 for x, _ in used if x == s) for s in R.SETS}}
    lr["pilot"] = R.pilot_check(entries, rows, worlds, pool, p_root, p_v, p_want, main)
    zr = R.zero_row_index(entries, worlds, main)  # addendum 18b / 18c (the A18 / BS18 configs set addendum18)
    pw = {r: {k.split("|", 1)[0] for k, x in pool.items() if x["role"] == r} for r in ("B", "S")}
    reg = {r: set(run_cfg["structural_no_row_registered"][r]) for r in ("B", "S")}
    known = {r: set(run_cfg["zero_row_nonstructural_known"][r]) for r in ("B", "S")}
    der = {r: sorted(k for k in zr if k.split("|", 1)[0] in pw[r] and k not in pool) for r in ("B", "S")}
    lr["structural_no_row"] = {r: [k for k in der[r] if k.split("|", 1)[1] in reg[r] and k not in known[r]] for r in der}
    lr["zero_row_unexpected"] = {r: [k for k in der[r] if k.split("|", 1)[1] not in reg[r] and k not in known[r]] for r in der}
    lr["zero_row_world_unresolved"] = sorted(k for k in zr if k.startswith("?|"))
    lr["structural_registered"] = {r: sorted(reg[r]) for r in ("B", "S")}
    me = man["per_label_combo"].get(f"{label} | {R.COMBO}")
    lr["_me"] = me
    if me is not None:
        lr["manifest"] = {"eligible_units_vs_mine": {"agree": 0, "disagree_n": 0}}
        eu = me["sizes"].get("eligible_units_by_run", {})
        agree, dis = 0, 0
        for key, v in eu.items():
            for role in ("B", "S"):
                if isinstance(v, dict) and role in v and key in pool and pool[key]["role"] == role:
                    mine = 0 if pool[key]["zero"] else len(pool[key]["r"]["score"])
                    agree, dis = (agree + 1, dis) if v[role] == mine else (agree, dis + 1)
        lr["manifest"]["eligible_units_vs_mine"] = {"agree": agree, "disagree_n": dis}
    return lr


# ---------------------------------------------------------------- per-label arrays for the bootstrap
def build_label(R, L, main_idx):
    """Arrays for one label x readout: both pool sides (side_data) and the attack arms (per used cell, vectorized), in the reducer's order."""
    import numpy as np

    run, pool, used, single = L["run"], L["pool"], L["used"], L["single"]
    att = {a: [run[f"{a}|{c}"] for _, c in used] for a in R.ARMS}
    nonzero_folds = {a: {x["fold"] for x in att[a] if not x["zero"]} for a in R.ARMS}
    pB = [x for x in pool.values() if x["role"] == "B"]
    pS = [x for x in pool.values() if x["role"] == "S"]
    sides = {"B": side_data(R, pB, nonzero_folds["run1"] | nonzero_folds["run2"] | nonzero_folds["sim"], main_idx, single),
             "S": side_data(R, pS, nonzero_folds["sim"], main_idx, single)}  # B also transforms the sim units (real line, ext)
    cells_main = np.array([main_idx[L["main"][c]] for _, c in used], dtype=np.int64)
    n_main = np.bincount(cells_main, minlength=len(main_idx)).astype(float)
    arms = {}
    for a in R.ARMS:
        xs = att[a]
        n = len(xs)
        maxs = np.full(n, -np.inf)
        fold = np.full(n, -9, dtype=np.int64)  # -9: single-model or zero-unit; 0..4 mapped; -1 fold -1
        f1_units, f1_owner = [], []
        for i, x in enumerate(xs):
            if x["zero"]:
                continue
            if single or x["fold"] is None:
                maxs[i] = float(np.max(x["r"]["score"]))
            elif x["fold"] >= 0:
                maxs[i], fold[i] = float(np.max(x["r"]["score"])), int(x["fold"])
            else:
                fold[i] = -1
                f1_units.append(np.asarray(x["r"]["sbm"], float))
                f1_owner.append(np.full(len(x["r"]["sbm"]), i))
        zero = np.array([bool(x["zero"]) for x in xs])
        f1 = np.concatenate(f1_units) if f1_units else np.zeros((0, 5))
        own = np.concatenate(f1_owner) if f1_owner else np.zeros(0, np.int64)
        order = np.argsort(own, kind="stable")
        f1, own = f1[order], own[order]
        starts = np.flatnonzero(np.r_[True, own[1:] != own[:-1]]) if len(own) else np.zeros(0, np.int64)
        arms[a] = {"maxs": maxs, "fold": fold, "zero": zero, "f1": f1, "f1_runs": own[starts] if len(own) else own, "f1_starts": starts}
    return {"sides": sides, "arms": arms, "cells_main": cells_main, "inv_n": 1.0 / n_main[cells_main], "single": single,
            "n_cells": len(used)}


def zmax_arm(arm, side, mom, D):
    """Every attack run's largest transformed unit, for D draws at once (D x n_runs; -inf for zero-unit runs), under `side`'s moments."""
    import numpy as np

    Z = np.tile(arm["maxs"], (D, 1))
    if side["single"]:
        return Z
    for k in range(5):
        sel = arm["fold"] == k
        if sel.any():
            Z[:, sel] = (arm["maxs"][sel][None, :] - mom[k][0][:, None]) / mom[k][1][:, None]
    if len(arm["f1_runs"]):
        a, b = f1_coeffs(mom, D)
        m = a @ arm["f1"].T + b[:, None]  # D x n_f1_units
        z = (m - mom[-1][0][:, None]) / mom[-1][1][:, None]
        Z[:, arm["f1_runs"]] = np.maximum.reduceat(z, arm["f1_starts"], axis=1)
    Z[:, arm["zero"]] = -np.inf
    return Z


def side_ok(side, mom, D, which=2):
    """Per draw: every needed moment of the side valid (single-model sides are always valid). which = 2: the registered >= 2-runs
    rule; which = 3: 507 l.139's >= 2-units rule (reported beside, addendum 8 §3.4)."""
    import numpy as np

    ok = np.ones(D, bool)
    if not side["single"]:
        for k in side["needed"]:
            ok &= mom[k][which]
    return ok


REASONS = {0: "ok", 1: "d7", 2: "d6", 3: "empty", 4: "other", 5: "d7_units_ok"}  # per-draw reason codes (addendum 8 §3.4)
COUNT_KEYS = (("d7", 1), ("d7_units_ok", 5), ("d6", 2), ("empty", 3), ("other", 4))


def eval_chunk(LB, M, rows, levels, mode, hits=False):
    """Per-draw quantities for one label over draw rows `rows` of M (row 0 = draw 0). mode 'joint' refits moments and tau each draw
    (R:3857; PT:15-16); 'fixed' holds draw-0 moments and tau (PT:17); 'joint_s507' holds draw-0 moments but redraws tau (D8's
    S507:138 reading, B only). Returns arrays over the rows: tau per side x level and the engine's selected unit (tau_idx, -1 if
    none), moments per side, TPR per arm x level (arms run1, run2, sim at its side; simB = sim at tau_B under B's transform), the
    per-draw flip weight (the weight share of cells whose top unit lies within 2b of tau), and a reason code per TPR draw and arm
    (REASONS: 1 d7 / 5 d7_units_ok where a needed moment is invalid, 2 d6 where k_d = 0, 3 empty where no cohort main has weight).
    Nothing is None (09, R1 #4): invalid draws are NaN with their code. hits=True adds the engine's caught flags per cell (C5)."""
    import numpy as np

    Mf = M[rows].astype(float)
    D, nl = len(rows), len(levels)
    sides = LB["sides"]
    mom = {}
    for sd in ("B", "S"):
        if sides[sd]["single"]:
            mom[sd] = {}
        elif mode == "joint":
            mom[sd] = weighted_moments(sides[sd], Mf)
        else:  # draw-0 moments held: row 0's moments broadcast to every draw
            m0 = weighted_moments(sides[sd], np.ones((1, M.shape[1])))
            mom[sd] = {k: tuple(np.repeat(v[:1], D) for v in t) for k, t in m0.items()}
    ok = {sd: side_ok(sides[sd], mom[sd], D, 2) for sd in ("B", "S")}
    oku = {sd: side_ok(sides[sd], mom[sd], D, 3) for sd in ("B", "S")}
    tau = {sd: np.full((D, nl), np.nan) for sd in ("B", "S")}
    tidx = {sd: np.full((D, nl), -1, dtype=np.int64) for sd in ("B", "S")}
    reason = {sd: np.repeat(np.where(ok[sd], 0, np.where(oku[sd], 5, 1)).astype(np.int8)[:, None], nl, axis=1) for sd in ("B", "S")}
    with np.errstate(invalid="ignore", divide="ignore"):
        for sd in ("B", "S"):
            unit_w = sides[sd]["unit_main"]
            if mode == "joint":
                for j in np.flatnonzero(ok[sd]):
                    tau[sd][j], tidx[sd][j] = weighted_tau(transform_draw(sides[sd], mom[sd], j), Mf[j, unit_w], levels)
            elif ok[sd][0]:
                z0 = transform_draw(sides[sd], mom[sd], 0)
                if mode == "fixed":
                    tau[sd][:], tidx[sd][:] = weighted_tau(z0, np.ones(len(z0)), levels)
                else:  # joint_s507
                    for j in range(D):
                        tau[sd][j], tidx[sd][j] = weighted_tau(z0, Mf[j, unit_w], levels)
            reason[sd][(reason[sd] == 0) & ~np.isfinite(tau[sd])] = 2  # D6: k_d = 0
        V = Mf[:, LB["cells_main"]] * LB["inv_n"][None, :]
        Vs = V.sum(1)
        tpr, flip, rr, cg = {}, {}, {}, {}
        for a, sd in (("run1", "B"), ("run2", "B"), ("sim", "S"), ("simB", "B")):
            Z = zmax_arm(LB["arms"]["sim" if a == "simB" else a], sides[sd], mom[sd], D)
            tpr[a], flip[a] = np.full((D, nl), np.nan), np.zeros((D, nl))
            rr[a] = reason[sd].copy()
            rr[a][(rr[a] == 0) & ~(Vs > 0)[:, None]] = 3  # an empty arm (no positive-weight cohort main)
            if hits:
                cg[a] = np.zeros((D, Z.shape[1], nl), bool)
            for li in range(nl):
                T = tau[sd][:, li]
                good = np.isfinite(T) & (Vs > 0)
                caught = Z > T[:, None]
                tpr[a][:, li] = np.where(good, (V * caught).sum(1) / np.where(Vs > 0, Vs, 1.0), np.nan)
                near = np.abs(np.where(np.isfinite(Z), Z, np.inf) - T[:, None]) <= 2 * RTOL * np.maximum(1.0, np.abs(T))[:, None]
                flip[a][:, li] = np.where(good, (V * near).sum(1) / np.where(Vs > 0, Vs, 1.0), 0.0)  # a bound term, 0 off-support
                if hits:
                    cg[a][:, :, li] = caught
    mom_out = {sd: {str(k): [np.asarray(x) for x in t] for k, t in mom[sd].items()} for sd in mom}
    out = {"tau": tau, "tau_idx": tidx, "tpr": tpr, "flip": flip, "mom": mom_out, "ok": ok, "ok_units": oku, "reason": rr}
    if hits:
        out["caught"] = cg
    return out


def draw0_consistency(LB, M, levels, d0):
    """Addendum 4 §1 (Codex r1 #12): the engine at draw 0 (all ones) reproduces compute_label's draw-0 tau within 1e-9 relative and
    its TPRs within 1e-12, with NO flip allowance. Returns (failures, the draw-0 eval) - the eval feeds C0's draw-0 flip bound."""
    e = eval_chunk(LB, M, [0], levels, "joint")
    bad = []
    for li, lvl in enumerate(levels):
        dl = d0["levels"][lvl]
        for sd, key in (("B", "tau_B"), ("S", "tau_S")):
            if not (abs(e["tau"][sd][0, li] - dl[key]["T"]) <= tol(dl[key]["T"])):
                bad.append(f"tau_{sd}@{lvl}")
        for a in ("run1", "run2", "sim"):
            if not (abs(e["tpr"][a][0, li] - dl["tpr"][a]) <= 1e-12):
                bad.append(f"tpr_{a}@{lvl}")
        if not (abs(e["tpr"]["simB"][0, li] - dl["tpr_sim_at_tauB"]) <= 1e-12):
            bad.append(f"tpr_simB@{lvl}")
    return bad, e


# ---------------------------------------------------------------- nodes (scope file / node lists) and the record's node fields
LEAVES = {"catch": ("sim", "real_mean"), "gap": ("delta_full", "run2_minus_run1_at_tauB", "real_line", "sim_minus_run1", "sim_minus_run2"),
          "coheadline": ("S", "Y")}  # the leaves the engine computes (addendum 8e); a node naming any other leaf is "unparsed"


def parse_node(node):
    """'pooled|<label>|pooled_all_origins|<mode>|<level>|<gap|catch>|<leaf>' or
    'coheadline_signed|pooled_all_origins|<mode>|<level>|<group>|<readout>|<S|Y>' -> dict. The label (it may contain '|', 'FIBS|uncut C')
    is everything between 'pooled|' and '|pooled_all_origins|', never split; the rest is split with a bounded maxsplit so a leaf keeps
    any '|' (addendum 8e), and only LEAVES are accepted."""
    if node.startswith("pooled|") and f"|{COMBO}|" in node:
        label, rest = node[len("pooled|"):].split(f"|{COMBO}|", 1)
        mode, level, kind, name = rest.split("|", 3)
        if label and mode in ("joint", "fixed") and kind in ("gap", "catch") and name in LEAVES[kind]:
            return {"family": kind, "label": label, "mode": mode, "level": level, "name": name}
    elif node.startswith(f"coheadline_signed|{COMBO}|"):
        mode, level, group, ro, sy = node[len(f"coheadline_signed|{COMBO}|"):].split("|", 4)
        if mode in ("joint", "fixed") and group in GROUPS and ro in ("C", "R") and sy in LEAVES["coheadline"]:
            return {"family": "coheadline", "label": None, "mode": mode, "level": level, "group": group, "readout": ro, "name": sy}
    stop("unparsed node", [node])


LEVEL = ("<level>",)  # node_path's placeholder for the level key, resolved by number (level_key); a tuple, never a record key


def node_path(p):
    """THE record path of a parsed node (addendum 8e: the one builder every lookup uses). A co-headline node's group and readout are ONE
    record key, "<group>|<readout>" (addendum 8d; confirmed keys-only by 2e on A and B)."""
    if p["family"] == "coheadline":
        return ["coheadline_signed", COMBO, p["mode"], LEVEL, f"{p['group']}|{p['readout']}", p["name"]]
    return ["pooled", p["label"], COMBO, p["mode"], LEVEL, p["family"], p["name"]]


def level_key(d, level):
    """The record's level key equal to `level` as a number (exactly one), so '0.01' / 0.01 / '1e-02' spellings can't silently differ."""
    ks = [k for k in d if _num(k) is not None and float(k) == float(level)]
    if len(ks) != 1:
        raise KeyError(f"level {level}: {len(ks)} keys")
    return ks[0]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def walk(H, p):
    """Follow node_path(p) in record H: (the leaf, the chain of (parent, key) taken). Every step is a direct index; a missing key raises."""
    d, chain = H, []
    for k in node_path(p):
        key = level_key(d, p["level"]) if k is LEVEL else k
        chain.append((d, key))
        d = d[key]
    return d, chain


def record_node(H, p):
    """The record's leaf for a parsed node, through the shared path (walk / node_path)."""
    return walk(H, p)[0]


LEAF_KEYS = {"ci95", "degenerate", "draws", "finite_draws", "point"}  # 09 (addendum 8e): the exact leaf key set


def _leaf_ok(leaf):
    """09's types (addendum 8e): exactly LEAF_KEYS; ci95 a 2-list of numbers or nulls; draws / finite_draws ints; point a number or null;
    degenerate a bool."""
    return (isinstance(leaf, dict) and set(leaf) == LEAF_KEYS and isinstance(leaf["ci95"], list) and len(leaf["ci95"]) == 2
            and all(x is None or _isnum(x) for x in leaf["ci95"]) and _isint(leaf["draws"]) and _isint(leaf["finite_draws"])
            and (leaf["point"] is None or _isnum(leaf["point"])) and isinstance(leaf["degenerate"], bool))


def resolve_check(H, nodes):
    """The resolve gate (addendum 8e): every node resolves through walk to a leaf of 09's shape, else STOP 'resolve:<family>' (the node
    in stop_ids). Returns n_degenerate, the count of nodes whose degenerate is true (read only to count; it gates nothing)."""
    n_deg = 0
    for n in nodes:
        p = parse_node(n)
        try:
            leaf = walk(H, p)[0]
        except (KeyError, TypeError):
            stop(f"resolve:{p['family']}", [n])
        if not _leaf_ok(leaf):
            stop(f"resolve:{p['family']}", [n])
        n_deg += int(leaf["degenerate"] is True)
    return n_deg


def members(p):
    """Label x readouts a node needs."""
    if p["family"] == "coheadline":
        return [f"{SIX[i]} {p['readout']}" for i in GROUPS[p["group"]]]
    return [p["label"]]


def node_series(p, per, lv):
    """My per-draw values of a node from per-label TPR arrays (per[label][mode] = eval output over draws 0..N), at level index lv."""
    import numpy as np

    def one(label, name, fam):
        t = per[label][p["mode"]]["tpr"]
        r1, r2, sm, sb = (t[a][:, lv] for a in ("run1", "run2", "sim", "simB"))
        real = 0.5 * (r1 + r2)
        if fam == "catch":
            return {"sim": sm, "real_mean": real}[name]
        return {"delta_full": sm - real, "run2_minus_run1_at_tauB": r2 - r1, "real_line": sb - real,
                "sim_minus_run1": sm - r1, "sim_minus_run2": sm - r2}[name]

    if p["family"] != "coheadline":
        return one(p["label"], p["name"], p["family"])
    xs = np.stack([one(m, "delta_full" if p["name"] == "S" else "run2_minus_run1_at_tauB", "gap") for m in members(p)])
    return xs.mean(0)  # NaN in any member -> NaN (common finite draws, R:4936; a group is complete or absent, PT:39)


def node_nan_counts(p, per, lv, x):
    """Addendum 8 §3.4: the node's NaN draws among 1..N counted by reason. A NaN draw takes the first non-zero code among its
    constituent arms (members in order, arms in COEF order); a NaN draw whose constituents are all ok counts as 'other'."""
    import numpy as np

    nm = p["name"] if p["family"] != "coheadline" else ("delta_full" if p["name"] == "S" else "run2_minus_run1_at_tauB")
    code = np.zeros(len(x), np.int8)
    for lab in members(p):
        rr = per[lab][p["mode"]]["reason"]
        for a in COEF[nm]:
            code = np.where(code == 0, rr[a][:, lv], code)
    nan = ~np.isfinite(x)
    y = np.where(nan, np.where(code == 0, 4, code), 0)[1:]
    return {name: int(np.sum(y == k)) for name, k in COUNT_KEYS}


def interval(x, method, drop0=True):
    """95% percentile interval over the finite draws 1..N (TB:79: draw 0 excluded), numpy method `method` (C2). NaN ends if no draw
    is finite (never None, 09)."""
    import numpy as np

    y = x[1:] if drop0 else x
    fin = y[np.isfinite(y)]
    if not len(fin):
        return float("nan"), float("nan"), int(len(y)), 0
    lo, hi = np.percentile(fin, [2.5, 97.5], method=method)
    return float(lo), float(hi), int(len(y)), int(len(fin))


# ---------------------------------------------------------------- C5: the engine vs explicit duplication (synthetic; TB:83-86; Codex r1 #12)
def _c5_world(R, rng, single):
    """A synthetic label (addendum 7 §2): 15 mains; folds 0-4 each hold >= 2 mains (mains 0-9), mains 10-14 random (fold -1 included);
    every third main has cells in both origins; one zero-unit attack run; unequal clusters; 2-decimal scores (ties)."""
    import numpy as np

    U = 15
    sets = ("S_old", "S_new")
    folds = [None] * U if single else [m % 5 for m in range(10)] + [int(rng.choice([0, 1, 2, 3, 4, -1])) for _ in range(5)]
    if not single and folds.count(-1) == 1:
        folds[folds.index(-1)] = 0  # fold -1 needs >= 2 mains too, or none
    main_of, used, run, pool = {}, [], {}, {}
    for m in range(U):
        mn = f"m{m}"
        ncell = 2 if m % 3 == 0 else 1
        for j in range(ncell):
            c = f"c{m}_{j}"
            main_of[c] = mn
            used.append((sets[j] if ncell == 2 else sets[m % 2], c))
            for a in R.ARMS:
                zero = m == U - 1 and a == "run2"
                n = int(rng.integers(1, 5))
                sc = np.round(rng.normal(size=n), 2)
                sb = rng.normal(size=(n, 5)) if folds[m] == -1 else None
                run[f"{a}|{c}"] = {"r": None if zero else {"score": sc, "sbm": sb, "unit": np.arange(n)}, "fold": folds[m], "main": mn, "zero": zero}
        for role in ("B", "S"):
            for r in range(2 + int(rng.integers(0, 2))):
                n = int(rng.integers(1, 6))
                sc = np.round(rng.normal(size=n), 2)
                sb = rng.normal(size=(n, 5)) if folds[m] == -1 else None
                key = f"w{role}{r}|p{m}"
                pool[key] = {"r": {"score": sc, "sbm": sb, "unit": np.arange(n)}, "fold": folds[m], "main": mn, "zero": False, "role": role, "key": key}
    return {"run": run, "pool": pool, "used": used, "single": single, "main": main_of, "folds": folds, "U": U}


def _c5_weights(rng, L, trial):
    """Integer weights 1-3 with 0-2 zeros, drawn only among mains whose fold keeps >= 2 positive-weight mains; every 50th trial
    all-zero (empty subsets); addendum 7 §2."""
    import numpy as np

    U = L["U"]
    if trial % 50 == 6:
        return np.zeros(U, dtype=np.int64)
    w = rng.integers(1, 4, size=U)
    folds = L["folds"]
    for _ in range(int(rng.integers(0, 3))):
        m = int(rng.integers(0, U))
        same = [k for k in range(U) if folds[k] == folds[m] and w[k] > 0]
        if len(same) >= 3 or (folds[m] is None and len(same) >= 2):
            w[m] = 0
    return w



def _c5_dup(R, L, w):
    """Explicit duplication of main clusters: main m (its cells in every origin, its pool runs) repeated w[m] times as distinct mains."""
    run, pool, used, main = {}, {}, [], {}
    for s, c in L["used"]:
        m = L["main"][c]
        for t in range(int(w[int(m[1:])])):
            ct = f"{c}#{t}"
            main[ct] = f"{m}#{t}"
            used.append((s, ct))
            for a in R.ARMS:
                run[f"{a}|{ct}"] = dict(L["run"][f"{a}|{c}"], main=f"{m}#{t}")
    for key, x in L["pool"].items():
        for t in range(int(w[int(x["main"][1:])])):
            pool[f"{key}#{t}"] = dict(x, key=f"{key}#{t}", main=f"{x['main']}#{t}")
    return run, pool, used, main


def _c5_d7_valid(L, w, single):
    """Codex r2 #7 / 09: each side's validity computed INDEPENDENTLY of side_data / weighted_moments, with plain numpy on the explicitly
    duplicated units. Per side, the folds the engine needs (the pool runs' folds plus the folds of the non-zero attack runs the side
    transforms; -1 -> all of 0..4) must each hold >= 2 distinct positive-weight pool runs and a finite SD > 0 (ddof 0) over the
    duplicated scores; fold -1 likewise on m built from those fold moments. Single-model sides are always valid."""
    import numpy as np

    if single:
        return True
    for role, arms in (("B", ("run1", "run2", "sim")), ("S", ("sim",))):
        runs = [x for x in L["pool"].values() if x["role"] == role and not x["zero"]]
        needed = {x["fold"] for x in runs} | {x["fold"] for k, x in L["run"].items() if k.split("|", 1)[0] in arms and not x["zero"]}
        if -1 in needed:
            needed |= {0, 1, 2, 3, 4}
        mom = {}
        for k in sorted(needed):
            pos = [x for x in runs if x["fold"] == k and w[int(x["main"][1:])] > 0]
            if len({x["key"] for x in pos}) < 2:
                return False
            if k == -1:
                continue
            s = np.concatenate([np.tile(x["r"]["score"], int(w[int(x["main"][1:])])) for x in pos])
            mom[k] = (float(np.mean(s)), float(np.std(s)))
            if not (np.isfinite(mom[k][1]) and mom[k][1] > 0):
                return False
        if -1 in needed:
            pos = [x for x in runs if x["fold"] == -1 and w[int(x["main"][1:])] > 0]
            sb = np.concatenate([np.tile(x["r"]["sbm"], (int(w[int(x["main"][1:])]), 1)) for x in pos])
            m = sum((sb[:, k] - mom[k][0]) / mom[k][1] for k in range(5)) / 5.0
            if not (np.isfinite(np.std(m)) and np.std(m) > 0):
                return False
    return True


def ref_selected_unit(zd, ks, us, k, pos, order="shared"):
    """Addendum 8c (Codex r22 #1; 09): the reference's selected threshold unit at rank k (as the reducer indexes it, min(k, n - 1)) under
    the SHARED tie order: duplicated copies of a unit placed contiguously, in the weighted path's stable mergesort order, i.e. by
    descending z, then the original unit's position in the engine's unit order (pos), then the copy index. order="copy_major" is the
    duplicate's own list order (the naive order; the mismatch negative). Returns (original run key, unit id, the z at that position)."""
    import numpy as np

    orig = [str(x).rsplit("#", 1) for x in ks]
    zz = np.asarray(zd, float)
    if order == "shared":
        p = np.array([pos[(o[0], int(u))] for o, u in zip(orig, us)])
        t = np.array([int(o[1]) for o in orig])
        idx = np.lexsort((t, p, -zz))
    else:
        idx = np.argsort(-zz, kind="stable")
    j = int(idx[min(k, len(idx) - 1)])
    return orig[j][0], int(us[j]), float(zz[j])


def c5_weighted_path(R, levels=("0.05", "0.2"), n_trials=300, force_skip=False):
    """C5 (addenda 4, 6 §12, 7, 8 §2.6-7): the engine's ACTUAL outputs at integer weights vs compute_label on the explicitly duplicated
    mains: moments (folds 0-4 and fold -1) and every TPR / delta_full / real line within absolute 1e-12 (TB:86); tau the SAME
    order-statistic unit as the engine actually selected, its value within 1e-12 * max(1, |tau|); caught flags bit-equal per attack
    cell, arm and level (the engine's own flags). Validity: the engine's must equal _c5_d7_valid's on every trial; a skip needs both
    invalid, and the engine's tau / TPRs NaN with a D7 code. Negatives (isolated): per-cell weights (V only) and unweighted tau
    (threshold weights only) must each be detected. force_skip routes every trial to the skip path (addendum 7 §2 (d))."""
    import numpy as np

    rng = np.random.default_rng(20261021)
    R.LEVELS = list(levels)
    classes = ("weighted_tpr_delta", "cross_origin", "zero_weights", "empty_subsets", "threshold_ties", "unequal_clusters", "fold_minus1", "single_model")
    cov = {c: 0 for c in classes}
    fails, failed_trials, compared_trials, skipped, empty = [], set(), set(), 0, 0
    neg = {"per_cell_weights": False, "unweighted_tau": False}

    def fail(trial, what):
        fails.append(f"trial {trial}: {what}")
        failed_trials.add(trial)

    for trial in range(n_trials):
        single = trial % 7 == 1
        L = _c5_world(R, rng, single)
        U = L["U"]
        w = _c5_weights(rng, L, trial)
        if force_skip:  # addendum 7 §2 (d): every trial routed to the skip path, before any work
            skipped += 1
            continue
        main_idx = {f"m{i}": i for i in range(U)}
        LB = build_label(R, L, main_idx)
        Mw = w[None, :].astype(np.int64)
        e = eval_chunk(LB, Mw, [0], list(levels), "joint", hits=True)
        run_d, pool_d, used_d, main_d = _c5_dup(R, L, w)
        eng_ok = bool(e["ok"]["B"][0] and e["ok"]["S"][0])
        if not used_d:  # all-zero weights: an empty cohort; the engine's TPRs must be NaN with a non-ok code
            if all(np.isnan(e["tpr"][a][0]).all() and (e["reason"][a][0] != 0).all() for a in ("run1", "run2", "sim")):
                empty += 1
                cov["empty_subsets"] += 1
            else:
                fail(trial, "empty cohort but a finite TPR")
            continue
        if _c5_d7_valid(L, w, single) != eng_ok:
            fail(trial, "engine validity differs from the independent D7 check")
            continue
        if not eng_ok:  # both say invalid: the engine's outputs must be NaN with a D7 code (1 or 5) on every invalid side
            bad_sides = [sd for sd in ("B", "S") if not e["ok"][sd][0]]
            arms = [a for a, sd in (("run1", "B"), ("run2", "B"), ("sim", "S")) if sd in bad_sides]
            if all(np.isnan(e["tau"][sd][0]).all() for sd in bad_sides) and all(np.isin(e["reason"][a][0], (1, 5)).all() for a in arms):
                skipped += 1
            else:
                fail(trial, "invalid engine outputs")
            continue
        try:
            d0 = R.compute_label(run_d, pool_d, used_d, main_d, single, ext=True)
        except AssertionError:
            fail(trial, "reducer invalid where the engine is valid")
            continue
        compared_trials.add(trial)
        pos = w[w > 0]
        cov["weighted_tpr_delta"] += int(len(set(pos.tolist())) > 1)
        cov["cross_origin"] += int(any(m % 3 == 0 and w[m] > 0 for m in range(U)))
        cov["zero_weights"] += int((w == 0).any() and (w > 0).any())
        sizes = {len(x["r"]["score"]) for x in L["pool"].values() if w[int(x["main"][1:])] > 0}
        cov["unequal_clusters"] += int(len(sizes) >= 2)
        cov["fold_minus1"] += int(not single and any(f == -1 and w[m] > 0 for m, f in enumerate(L["folds"])))
        cov["single_model"] += int(single)
        tied = False
        for li, lvl in enumerate(levels):
            dl = d0["levels"][lvl]
            for sd, key in (("B", "tau_B"), ("S", "tau_S")):
                T = dl[key]["T"]
                if not abs(e["tau"][sd][0, li] - T) <= 1e-12 * max(1.0, abs(T)):
                    fail(trial, f"tau_{sd}@{lvl} value")
                side, i = LB["sides"][sd], int(e["tau_idx"][sd][0, li])  # the unit the engine actually selected (Codex r2 #6)
                zd, ks, us = d0["_pool_z"][sd]
                same = {(str(k).rsplit("#", 1)[0], int(u)) for k, u, v in zip(ks, us, zd) if v == T}
                tied |= len(same) > 1
                ref_pos_by_unit = {(side["ukeys"][q], int(side["uids"][q])): q for q in range(len(side["ukeys"]))}
                rk, ru, rz = ref_selected_unit(zd, ks, us, int(dl[key]["k"]), ref_pos_by_unit)  # the reference's selection, shared tie order (8c)
                if rz != T:
                    fail(trial, f"tau_{sd}@{lvl} reference rank")
                if i < 0 or (side["ukeys"][i], int(side["uids"][i])) != (rk, ru):  # exact identity, not set membership
                    fail(trial, f"tau_{sd}@{lvl} unit")
            for a in ("run1", "run2", "sim"):  # hits bit-equal per original cell: the engine's own caught flags (09 #1)
                eng = {c: bool(e["caught"][a][0, j, li]) for j, (_s, c) in enumerate(L["used"]) if w[int(L["main"][c][1:])] > 0}
                dup = {}
                for ct, v in dl["caught"][a].items():
                    dup.setdefault(ct.rsplit("#", 1)[0], set()).add(bool(v))
                if any(len(v) != 1 for v in dup.values()) or {c: next(iter(v)) for c, v in dup.items()} != eng:
                    fail(trial, f"hits_{a}@{lvl}")
            for a in ("run1", "run2", "sim"):
                if not abs(e["tpr"][a][0, li] - dl["tpr"][a]) <= 1e-12:
                    fail(trial, f"tpr_{a}@{lvl}")
            if not abs(e["tpr"]["simB"][0, li] - dl["tpr_sim_at_tauB"]) <= 1e-12:
                fail(trial, f"real line@{lvl}")
            df_e = e["tpr"]["sim"][0, li] - 0.5 * (e["tpr"]["run1"][0, li] + e["tpr"]["run2"][0, li])
            if not abs(df_e - dl["delta_full"]) <= 1e-12:
                fail(trial, f"delta_full@{lvl}")
        cov["threshold_ties"] += int(tied)
        if not single:  # the engine's own moments (eval_chunk's output), folds 0-4 and fold -1, absolute 1e-12 (TB:86)
            for sd in ("B", "S"):
                em, rm = e["mom"][sd], d0["moments"][sd]
                pairs = [(str(k), mu, sdv) for k, (mu, sdv) in rm["fold"].items()]
                if rm["f1"] is not None or "-1" in em:
                    pairs.append(("-1", *(rm["f1"] if rm["f1"] is not None else (np.nan, np.nan))))
                for k, mu, sdv in pairs:
                    if k not in em or not (abs(em[k][0][0] - mu) <= 1e-12 and abs(em[k][1][0] - sdv) <= 1e-12):
                        fail(trial, f"moments {sd} fold {k}")
        if len(set(pos.tolist())) > 1:
            LBn = dict(LB, inv_n=np.ones_like(LB["inv_n"]))  # per-cell weights: only V changes
            en = eval_chunk(LBn, Mw, [0], list(levels), "joint")
            if any(abs(en["tpr"][a][0, li] - d0["levels"][lv]["tpr"][a]) > 1e-12 for a in ("run1", "run2", "sim") for li, lv in enumerate(levels)):
                neg["per_cell_weights"] = True
            for sd, key in (("B", "tau_B"), ("S", "tau_S")):  # unweighted tau: the weighted moments kept, threshold weights all 1 (#12)
                side = LB["sides"][sd]
                mw = weighted_moments(side, Mw.astype(float)) if not side["single"] else {}
                tu, _ = weighted_tau(transform_draw(side, mw, 0), np.ones(len(side["s"])), list(levels))
                if any(abs(tu[li] - d0["levels"][lv][key]["T"]) > 1e-12 * max(1.0, abs(d0["levels"][lv][key]["T"])) for li, lv in enumerate(levels)):
                    neg["unweighted_tau"] = True
    return {"failures": fails, "n_trials": n_trials, "compared": len(compared_trials), "skipped_d7": skipped, "empty": empty,
            "n_failed_trials": len(failed_trials - compared_trials), "coverage": cov, "negatives_detected": neg}


C5_MIN_COMPARED, C5_MAX_SKIP_FRAC = 200, 0.10  # addendum 7 §2 (b)


def c5_guard(res):
    """Addendum 7 §2 (b)-(d) and addendum 8 §2.7: STOP-worthy problems of a C5 result (empty list = pass)."""
    probs = []
    if res["compared"] + res["skipped_d7"] + res["empty"] + res["n_failed_trials"] != res["n_trials"]:
        probs.append("accounting")
    if res["skipped_d7"] > C5_MAX_SKIP_FRAC * res["n_trials"]:
        probs.append("skipped_d7 over 10% of trials")
    if res["compared"] < C5_MIN_COMPARED:
        probs.append("fewer than 200 compared trials")
    probs += [f"no compared trial: {c}" for c, n in res["coverage"].items() if n < 1]
    if res["failures"]:
        probs.append("failures")
    if not all(res["negatives_detected"].values()):
        probs.append("a negative control was not detected")
    return probs


# ---------------------------------------------------------------- D5: halfwidth slot parse (addenda 2 §2, 3 §2, 6) and its fixtures
def d5_fit(slot_string, reading):
    """09's rule per scale (addendum 2 §2; Codex r1 #10): on each scale (fraction, x100 points) count the numeric tokens equal to the
    reading rounded to the token's own decimals. A scale fits iff its count is exactly 1. Exactly one fitting scale -> 'fit:<scale>';
    both -> 'unresolved' (ambiguous); none -> 'unresolved' if some scale has >= 2 equal tokens, else 'none'. A reading of exactly 0
    is scale-invariant (addendum 3 #7): exactly one equal token -> 'fit:invariant'. No tokens -> 'unresolved'."""
    import re

    toks = re.findall(r"[-+]?\d+(?:\.\d+)?", slot_string)
    if not toks:
        return "unresolved"

    def n_equal(v):
        return len([t for t in toks if round(v, len(t.split(".")[1]) if "." in t else 0) == float(t)])

    if reading == 0:
        n = n_equal(0.0)
        return "fit:invariant" if n == 1 else ("unresolved" if n >= 2 else "none")
    res = {"fraction": n_equal(reading), "points": n_equal(100.0 * reading)}
    fits = [s for s, n in res.items() if n == 1]
    if len(fits) == 2:
        return "unresolved"
    if len(fits) == 1:
        return f"fit:{fits[0]}"
    return "unresolved" if any(n >= 2 for n in res.values()) else "none"


def _d5_allowed(v):
    """The scales an element's per-reading result is compatible with: an invariant fit (a zero reading) with either (addendum 3 #7)."""
    if v == "fit:invariant":
        return {"fraction", "points", "invariant"}
    return {v[4:]} if v in ("fit:fraction", "fit:points") else set()


D5_STATUS = ("fit", "ambiguous", "mixed", "unresolved", "not_reproduced", "error", "flip_sensitive")


def d5_classify(per_element, flip=frozenset()):
    """Addendum 8 §3.1 (R1 #1; 09) and 8c §4: the (reading, scale) pairs that fit EVERY element. One pair -> 'fit' (a fit stays a fit);
    two or more -> 'ambiguous' (UNRESOLVED (D5 ambiguous)). No common pair: an element 'none' under both readings is NOT REPRODUCED (09
    ruling 3) unless it is flip-sensitive (`flip`: an element with a constituent node that passed C1 / the end check only via the flip
    term), which moves it to the flip list -> 'not_reproduced' if any NOT REPRODUCED element remains, else 'flip_sensitive' if the flip
    list is non-empty (UNRESOLVED (D5 flip-sensitive)); else an element with no fit at all (unresolved tokens, addendum 2) ->
    'unresolved'; else every element fits some pair but none is common -> 'mixed' (addendum 3 (a)). No elements -> 'error'.
    Returns (status, pair or None, nr keys, unresolved keys, flip-relabelled keys)."""
    if not per_element:
        return "error", None, [], [], []
    cands = []
    for r in sorted({r for res in per_element.values() for r in res}):
        sc = None
        for res in per_element.values():
            a = _d5_allowed(res.get(r))
            sc = a if sc is None else sc & a
        if sc:
            cands.append((r, "invariant") if sc == {"fraction", "points", "invariant"} else (r, next(iter(sc - {"invariant"}))))
    if len(cands) == 1:
        return "fit", cands[0], [], [], []
    if len(cands) >= 2:
        return "ambiguous", None, [], [], []
    nr_all = sorted(e for e, res in per_element.items() if all(v == "none" for v in res.values()))
    fl = [e for e in nr_all if e in flip]
    nr = [e for e in nr_all if e not in flip]
    unres = sorted(e for e, res in per_element.items() if e not in nr_all and not any(_d5_allowed(v) for v in res.values()))
    if nr:
        return "not_reproduced", None, nr, unres, fl
    if fl:
        return "flip_sensitive", None, [], unres, fl
    return ("unresolved", None, [], unres, []) if unres else ("mixed", None, [], [], [])


D5_AGG = {"max_all": max, "max_excl_on_policy": max, "min_excl_on_policy": min}


def d5_ready(slots, lines, nodes_a, slot_list):
    """09 (R1 #18; addendum 8c): D5's inputs must parse BEFORE the draws. Every halfwidth scope line has key_path ['slots', <key>] with
    the key in the pinned slot list; the slot exists in rc1a and is a string; its aggregator (the key's last part) is in D5_AGG; every
    node it names is one of A's nodes; the slots found equal the pinned list. Returns {slot: node set}; anything else STOPs."""
    by_slot = {}
    for e in lines:
        if not str(e.get("element", "")).startswith("halfwidth"):
            continue
        kp = e.get("key_path")
        if not (isinstance(kp, list) and len(kp) == 2 and kp[0] == "slots" and isinstance(kp[1], str)):
            stop("D5 not ready: key_path form")
        if kp[1] not in slot_list:  # the summary reports slot keys by membership in this pinned list
            stop("D5 not ready: slot list")
        if kp[1] not in slots:
            stop("D5 not ready: slot missing")
        if not isinstance(slots[kp[1]], str):
            stop("D5 not ready: slot not a string")
        if kp[1].rsplit(".", 1)[-1] not in D5_AGG:
            stop("D5 not ready: aggregator unknown")
        node = str(e.get("a_node", "")).rsplit("|ci95[", 1)[0]
        if node not in nodes_a:
            stop("D5 not ready: node not in A's list")
        by_slot.setdefault(kp[1], set()).add(node)
    if not by_slot:
        stop("D5 not ready: no halfwidth elements")
    if set(by_slot) != set(slot_list):
        stop("D5 not ready: slot list")
    return by_slot


# ---------------------------------------------------------------- the calibration npz (addendum 5 §3; Codex r1 #7: in-scope members only)
_CAL_CACHE = {}


def _calib_arrays(npz_path, want_fixed):
    """Only addendum 5's read set: <mode>__<level>__{B,S}__T (joint; fixed only for B), joint__{B,S}__{ok,mu,sd,mu0,sd0}."""
    import zipfile

    import numpy as np

    ck = (str(npz_path), bool(want_fixed))
    if ck in _CAL_CACHE:
        return _CAL_CACHE[ck]
    out = {}
    with zipfile.ZipFile(npz_path) as z:
        for nm in z.namelist():
            if not nm.endswith(".npy"):
                continue
            p = nm[:-4].split("__")
            t_ok = len(p) == 4 and p[3] == "T" and p[2] in ("B", "S") and (p[0] == "joint" or (p[0] == "fixed" and want_fixed))
            m_ok = len(p) == 3 and p[0] == "joint" and p[1] in ("B", "S") and p[2] in ("ok", "mu", "sd", "mu0", "sd0")
            if not (t_ok or m_ok):
                continue
            with z.open(nm) as fh:
                arr = np.load(io.BytesIO(fh.read()))
            out[(p[0], float(p[1]), p[2], "T") if t_ok else (p[0], p[1], p[2])] = arr
    _CAL_CACHE[ck] = out
    return out


def _row(arr, d, i_rule, kind):
    """Element(s) for draw d under the index rule, or None if the array is not in its quantity's ONE expected layout (Codex r2 #9;
    09). 'series' (tau, mu0, sd0): 1-D of the rule's length. 'folds' (mu, sd): 2-D, the rule's length x 5, draw axis first or second
    (addendum 5 §3). 'ok' (never gates; addendum 8 §2.9): either."""
    want = N_DRAWS + 1 if i_rule == "I1" else N_DRAWS
    j = d if i_rule == "I1" else d - 1
    shp = getattr(arr, "shape", None)
    if kind in ("series", "ok") and shp == (want,):
        return arr[j]
    if kind in ("folds", "ok") and shp == (want, 5):
        return arr[j]
    if kind in ("folds", "ok") and shp == (5, want):
        return arr[:, j]
    return None


def calib_check(npz_path, ev, rows, levels, i_rule, fixed_ev=None, strict=False, label="?"):
    """C1 against one label's D1 calibration npz (addendum 5 §3). Per draw d in `rows`, under index rule i_rule: tau_B / tau_S (joint,
    every level) both non-finite or within 2b; fold moments (folds my side needs) and fold -1 mu0 / sd0 within b; B's fixed tau when
    fixed_ev is given (identification included, Codex r1 #7). ok masks are counted, not gated. A missing member or a layout off its
    quantity's one expected layout is a mismatch during identification (strict=False) and a STOP after it (strict=True; the
    member's identifier goes to stop_ids). Returns (n_compared, n_mismatch, first '<quantity>@draw<d>', n_ok_diff)."""
    import numpy as np

    st = _calib_arrays(npz_path, fixed_ev is not None)
    n_cmp = n_bad = n_okd = 0
    first = None
    mom = {}
    for sd in ("B", "S"):
        if ev["mom"] and ev["mom"][0].get(sd):
            mom[sd] = {k: [np.concatenate([c[sd][k][i] for c in ev["mom"]]) for i in range(2)] for k in ev["mom"][0][sd]}

    def bad(what, d):
        nonlocal n_bad, first
        n_bad += 1
        if first is None:
            first = f"{what}@draw{d}"

    def get(key, name, d, kind):
        if key not in st:
            if strict:
                stop("calibration member missing", [f"{label} {name}"])
            return None
        v = _row(st[key], d, i_rule, kind)
        if v is None and strict and kind != "ok":
            stop("calibration layout", [f"{label} {name}"])
        return v

    def close_tau(s_, m_):
        return s_ is not None and np.ndim(s_) == 0 and (np.isfinite(s_) == np.isfinite(m_)) and (not np.isfinite(s_) or abs(m_ - s_) <= 2 * tol(float(s_)))

    for d in rows:
        for li, lvl in enumerate(levels):
            for sd in ("B", "S"):
                n_cmp += 1
                if not close_tau(get(("joint", float(lvl), sd, "T"), f"joint__{lvl}__{sd}__T", d, "series"), ev["tau"][sd][d, li]):
                    bad(f"tau_{sd}@{lvl}", d)
                if fixed_ev is not None:
                    n_cmp += 1
                    if not close_tau(get(("fixed", float(lvl), sd, "T"), f"fixed__{lvl}__{sd}__T", d, "series"), fixed_ev["tau"][sd][d, li]):
                        bad(f"fixed_tau_{sd}@{lvl}", d)
        for sd in ("B", "S"):
            o = get(("joint", sd, "ok"), f"joint__{sd}__ok", d, "ok")
            if o is None or bool(np.all(o)) != bool(ev["ok"][sd][d]):
                n_okd += 1
            if sd not in mom:
                continue
            for nm, idx in (("mu", 0), ("sd", 1)):
                s_ = get(("joint", sd, nm), f"joint__{sd}__{nm}", d, "folds")
                if s_ is None:
                    bad(f"{nm}_{sd} layout", d)
                    continue
                for k in range(5):
                    if str(k) not in mom[sd]:
                        continue
                    m_ = mom[sd][str(k)][idx][d]
                    n_cmp += 1
                    if (np.isfinite(s_[k]) != np.isfinite(m_)) or (np.isfinite(s_[k]) and abs(m_ - s_[k]) > tol(float(s_[k]))):
                        bad(f"{nm}_{sd}[{k}]", d)
            if "-1" in mom[sd]:
                for nm, idx in (("mu0", 0), ("sd0", 1)):
                    s_ = get(("joint", sd, nm), f"joint__{sd}__{nm}", d, "series")
                    m_ = mom[sd]["-1"][idx][d]
                    n_cmp += 1
                    if s_ is None or (np.isfinite(s_) != np.isfinite(m_)) or (np.isfinite(s_) and abs(m_ - s_) > tol(float(s_))):
                        bad(f"{nm}_{sd}", d)
    return n_cmp, n_bad, first, n_okd


def gap_compare(stored, parsed, values, fbs, scale, i_rule):
    """C1's gap members (addendum 4 §5; Codex r2 #10): every in-scope gap node needs its stored array (else STOP) of the identified
    rule's length (else STOP). Per draw 1..N: both non-finite, or |scale x mine - stored| <= b(stored) + scale x the draw's flip bound.
    Returns (compared, mismatches, flip-attributable, ['gap <node>@draw<d>' at each failing node's first failing draw], {node: its
    flip-attributable draws} (addendum 8c §4))."""
    import numpy as np

    want = N_DRAWS + 1 if i_rule == "I1" else N_DRAWS
    n_cmp = n_mis = n_flip = 0
    first, by_node = [], {}
    for n, p in parsed.items():
        if p["family"] != "gap":
            continue
        key = (p["label"], p["mode"], float(p["level"]), p["name"])
        if key not in stored:
            stop("C1: a stored gap member is missing", [n])
        arr = stored[key]
        if getattr(arr, "shape", None) != (want,):
            stop("C1: stored gap array length vs the identified index rule", [n])
        s_ = np.asarray(arr[1:] if i_rule == "I1" else arr, float)
        m_ = scale * np.asarray(values[n][1:], float)
        fs, fm = np.isfinite(s_), np.isfinite(m_)
        both = fs & fm
        with np.errstate(invalid="ignore"):
            b = RTOL * np.maximum(1.0, np.abs(np.where(fs, s_, 0.0)))
            dif = np.where(both, np.abs(m_ - np.where(fs, s_, 0.0)), 0.0)
            lim = b + scale * np.asarray(fbs[n][1:], float)
        mis = (fs != fm) | (both & (dif > lim))
        n_cmp += len(s_)
        n_mis += int(mis.sum())
        by_node[n] = int((both & (dif > b) & (dif <= lim)).sum())
        n_flip += by_node[n]
        if mis.any():
            first.append(f"gap {n}@draw{int(np.argmax(mis)) + 1}")
    return n_cmp, n_mis, n_flip, first, by_node


def npz_label(label):
    """The per-draw npz's label encoding (addendum 8e; the B stage-0 names match it, e.g. "FIBS|uncut C" -> "FIBS_uncut_C")."""
    return label.replace("|", "_").replace(" ", "_")


def npz_gap_member(label, mode, level, leaf):
    """A pooled gap member's npz name: <npz_label>__pooled_all_origins__<mode>__<level>__pooled__gap_<leaf>.npy (addendum 8e)."""
    return f"{npz_label(label)}__{COMBO}__{mode}__{level}__pooled__gap_{leaf}.npy"


def gap_arrays(npz_path, label, mode):
    """The stored pooled gap arrays of one label x mode, by member name (A's form, 20d: <lab>__<combo>__<mode>__<level>__pooled__gap_<name>;
    B's from stage 0). Returns {(level_str, name): array}; level matched by number later."""
    import zipfile

    import numpy as np

    lab = npz_label(label)
    out = {}
    with zipfile.ZipFile(npz_path) as z:
        for nm in z.namelist():
            parts = nm[:-4].split("__") if nm.endswith(".npy") else None
            if parts and len(parts) == 6 and parts[0] == lab and parts[1] == COMBO and parts[2] == mode and parts[4] == "pooled" and parts[5].startswith("gap_"):
                with z.open(nm) as fh:
                    out[(parts[3], parts[5][4:])] = np.load(io.BytesIO(fh.read()))
    return out


# ---------------------------------------------------------------- C2 on A's fixed-mode delta_full (PRESTATEMENT §4; Codex r1 #6)
A_LEVELS = ("0.01", "0.05")  # A's registered levels (config/routec_A18_pins.json)
METHODS = ("linear", "inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull",
           "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint", "nearest")


def c2_expected():
    """Every expected C2 node: A's fixed-mode pooled delta_full for the six headline probes x C / R x A's levels."""
    return [{"family": "gap", "label": f"{name} {ro}", "mode": "fixed", "level": lvl, "name": "delta_full"}
            for name in SIX for ro in ("C", "R") for lvl in A_LEVELS]


def c2_index_rule(stored):
    """Codex r2 #8 / 09: A's C2 index rule from A's OWN stored arrays (pre-statement §4 C2: "percentile and index rule re-identified
    on A's stored arrays"), independent of the worker's record: all 1-D and of one length; N + 1 -> I1, N -> I2; else STOP."""
    if not stored:
        stop("C2: expected members missing")
    shapes = {getattr(a, "shape", None) for a in stored.values()}
    if len(shapes) != 1 or next(iter(shapes)) not in ((N_DRAWS + 1,), (N_DRAWS,)):
        stop("C2: stored array layout")
    return "I1" if next(iter(shapes)) == (N_DRAWS + 1,) else "I2"


def c2_select(stored, rec_nodes):
    """The first numpy method (linear first) reproducing every node's record ci95 and finite_draws from the producer's own array,
    under the index rule of those arrays. stored / rec_nodes: {node_key: ...} for EVERY expected node; a missing or empty set STOPs
    (no silent intersection). Returns (method, index rule)."""
    import numpy as np

    if not stored or set(stored) != set(rec_nodes):
        stop("C2: expected members missing")
    i_rule = c2_index_rule(stored)
    for mth in METHODS:
        ok = True
        for n, arr in stored.items():
            y = arr[1:] if i_rule == "I1" else arr
            fin = y[np.isfinite(y)]
            c, fd = rec_nodes[n]["ci95"], rec_nodes[n]["finite_draws"]
            if not len(fin):
                ok = False
                break
            lo, hi = np.percentile(fin, [2.5, 97.5], method=mth)
            if not (abs(lo - c[0]) <= tol(c[0]) and abs(hi - c[1]) <= tol(c[1]) and _isint(fd) and fd == len(fin)):
                ok = False
                break
        if ok:
            return mth, i_rule
    stop("C2: no percentile method reproduces A's fixed-mode delta_full ci95")


def c2_identify():
    """C2 in either worker: A's record (pinned, in /cfg) and A's per-draw npz (pinned), the expected fixed-mode delta_full members.
    Returns (method, n expected, A's index rule)."""
    import gzip
    import zipfile

    import numpy as np

    raw = gzip.decompress(Path("/cfg", A_RECORD[0]).read_bytes())
    if sha(raw) != A_RECORD[1]:
        stop("C2: A record sha")
    HA = json.loads(raw)
    del raw
    if file_sha(Path("/p", GAP_NPZ["A"][0])) != GAP_NPZ["A"][1]:
        stop("C2: A per-draw npz sha")
    exp = c2_expected()
    want = {npz_gap_member(p["label"], "fixed", p["level"], "delta_full"): i for i, p in enumerate(exp)}
    stored = {}
    with zipfile.ZipFile(Path("/p", GAP_NPZ["A"][0])) as z:
        names = set(z.namelist())
        for nm, i in want.items():
            if nm in names:
                with z.open(nm) as fh:
                    stored[i] = np.load(io.BytesIO(fh.read()))
    rec = {i: record_node(HA, p) for i, p in enumerate(exp)}
    method, rule = c2_select(stored, rec)
    return method, len(exp), rule


# ---------------------------------------------------------------- the comparison core (pure; every read a direct index) + C4 fixtures
def _isint(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _isnum(x):
    import math

    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def compare_core(H, nodes, mine, scale, exact, mc_se):
    """Per node: my (lo, hi, draws, finite) vs the record's ci95 / draws / finite_draws (Codex r1 #8: exact integer types, draws ==
    10,000, finite >= 9,500 on both sides). C0 was decided before the draws (mine[n]['c0']); scale is the identified unit scale
    (1 or 100; never None here). EXACT tier: ends within tol (+ the flip bound -> 'EXACT (flip-attributable)'). Otherwise the
    diagnostic MC tier: ends within 4 sqrt(2) SE AND finite counts within 4 binomial SE (Codex r1 #9). Every record read is a direct
    index, so a missing key raises (C4)."""
    import math

    rows = {}
    for n in nodes:
        p = parse_node(n)
        r = record_node(H, p)
        ci, dr, fd, pt = r["ci95"], r["draws"], r["finite_draws"], r["point"]
        if not (isinstance(ci, list) and len(ci) == 2 and all(_isnum(v) for v in ci) and _isnum(pt)):
            raise ValueError("node value layout")
        if not (_isint(dr) and _isint(fd)):
            raise ValueError("node counts are not integers")
        m = mine[n]
        rec_ok = dr == N_DRAWS and MIN_FINITE <= fd <= dr  # per node; impossible counts -> NOT LIKE-FOR-LIKE (Codex r22 #2; 09)
        mine_ok = _isint(m["draws"]) and _isint(m["finite"]) and m["draws"] == N_DRAWS and m["finite"] >= MIN_FINITE and _isnum(m["lo"]) and _isnum(m["hi"])
        ends_eq = mine_ok and all(abs(scale * m[k] - ci[i]) <= tol(ci[i]) for i, k in enumerate(("lo", "hi")))
        flip_ok = mine_ok and all(abs(scale * m[k] - ci[i]) <= tol(ci[i]) + scale * m["flip_bound"] for i, k in enumerate(("lo", "hi")))
        counts_eq = dr == m["draws"] and fd == m["finite"]
        row = {"c0": bool(m["c0"]), "record_meta_ok": bool(rec_ok), "mine_complete": bool(mine_ok), "ends_equal": bool(ends_eq),
               "ends_flip_attributable": bool(flip_ok and not ends_eq), "counts_equal": bool(counts_eq), "label": None}
        if not m["c0"] or not rec_ok:
            row["verdict"] = "NOT LIKE-FOR-LIKE"
        elif exact:
            row["verdict"] = "REPRODUCED" if (flip_ok and counts_eq) else "NOT REPRODUCED"
            if row["verdict"] == "REPRODUCED":
                row["label"] = "EXACT (flip-attributable)" if row["ends_flip_attributable"] else "EXACT"
        else:
            row["verdict"] = "UNRESOLVED (stream gate)"
            se = mc_se[n]
            p_nan = ((N_DRAWS - fd) + (N_DRAWS - m["finite"])) / (2.0 * N_DRAWS) if mine_ok else 1.0
            fin_ok = mine_ok and abs(m["finite"] - fd) <= 4.0 * math.sqrt(N_DRAWS * p_nan * (1.0 - p_nan))
            row["mc_diagnostic_agrees"] = bool(mine_ok and fin_ok and all(math.isfinite(se[i]) and abs(scale * m[k] - ci[i]) <= 4 * math.sqrt(2) * scale * se[i]
                                                                              for i, k in enumerate(("lo", "hi"))))
        rows[n] = row
    return rows


def c4_fixtures(H, nodes, mine, scale, exact, mc_se):
    """Each record key compare_core reads is removed in turn, on the first node of each family; compare_core must raise. The keys are
    walk's chain (the shared path, addendum 8e) plus the node's four compared fields. A fixture whose key is absent STOPs (a no-op proves
    nothing). Returns {name: stopped}."""
    firsts = {}
    for n in nodes:
        firsts.setdefault(parse_node(n)["family"], n)
    stopped = {}
    for fam, n in sorted(firsts.items()):
        leaf, chain = walk(H, parse_node(n))
        targets = chain + [(leaf, f) for f in ("ci95", "draws", "finite_draws", "point")]
        for parent, key in targets:
            if not isinstance(parent, dict) or key not in parent:
                stop("fixture key absent")
            saved = parent.pop(key)
            try:
                compare_core(H, [n], mine, scale, exact, mc_se)
                stopped[f"{fam}:{key}"] = False
            except Exception:
                stopped[f"{fam}:{key}"] = True
            finally:
                parent[key] = saved
    return stopped


COEF = {"delta_full": {"sim": 1.0, "run1": 0.5, "run2": 0.5}, "run2_minus_run1_at_tauB": {"run1": 1.0, "run2": 1.0},
        "real_line": {"simB": 1.0, "run1": 0.5, "run2": 0.5}, "sim_minus_run1": {"sim": 1.0, "run1": 1.0},
        "sim_minus_run2": {"sim": 1.0, "run2": 1.0}, "sim": {"sim": 1.0}, "real_mean": {"run1": 0.5, "run2": 0.5}}


def node_flip(p, per, lv):
    """Per draw, the flip-implied bound on the node's value: sum over arms of coef x (weight share of near-threshold cells)."""
    def one(label, name):
        f = per[label][p["mode"]]["flip"]
        return sum(c * f[a][:, lv] for a, c in COEF[name].items())

    if p["family"] != "coheadline":
        return one(p["label"], p["name"])
    nm = "delta_full" if p["name"] == "S" else "run2_minus_run1_at_tauB"
    xs = [one(m, nm) for m in members(p)]
    return sum(xs) / len(xs)


def mc_se(x, method, n_batches=10):
    """C3: batch standard error of each interval end: 10 batches of 1,000 draws (R:132's practice); SE = sd(batch ends) / sqrt(10)."""
    import numpy as np

    y = x[1:]
    B = len(y) // n_batches
    ends = []
    for b in range(n_batches):
        z = y[b * B:(b + 1) * B]
        z = z[np.isfinite(z)]
        ends.append(np.percentile(z, [2.5, 97.5], method=method) if len(z) else [np.nan, np.nan])
    e = np.array(ends)
    return tuple(float(np.std(e[:, i], ddof=1) / np.sqrt(n_batches)) for i in (0, 1))


# ---------------------------------------------------------------- the exact report / summary schema (Codex r1 #2, r2 #1-2; 09; addenda 8 §6, 8c)
# The .jq gate holds the same vocabularies, grammars and rules; a1 diffs them before every commit.
STAGES = ("code_pins", "selftests", "pins", "manifests", "nodes", "resolve", "c4", "c2", "draw0", "c0", "d1", "pilot", "draws", "c1", "compare", "values", "d5")
D5_PAIRS = tuple(f"{r}:{s}" for r in ("half_range", "max_side") for s in ("fraction", "points", "invariant"))
VERDICTS = ("INTERVALS REPRODUCED (EXACT)", "NOT REPRODUCED", "NOT LIKE-FOR-LIKE", "NOT LIKE-FOR-LIKE (C0 scale unidentified)",
            "UNRESOLVED (stream gate)", "UNRESOLVED (D5 ambiguous)", "UNRESOLVED (D5 mixed readings)",
            "UNRESOLVED (D5 unresolved elements)", "UNRESOLVED (D5 error)", "UNRESOLVED (D5 flip-sensitive)", "STOP")
LEVEL_LIST = ("0.001", "0.00133352", "0.00177828", "0.00237137", "0.00316228", "0.00421697", "0.00562341", "0.00749894", "0.01",
              "0.0133352", "0.0177828", "0.0237137", "0.0316228", "0.0421697", "0.05")  # B's registered levels; A's 0.01 / 0.05 among them
_LABEL = r"[A-Za-z0-9_|\-]{1,40} [CR]"
_LEVEL = "(?:" + "|".join(x.replace(".", r"\.") for x in LEVEL_LIST) + ")"  # 09 (addendum 8c): membership, never a free decimal
_QTY = r"(?:(?:fixed_)?tau_[BS]@" + _LEVEL + r"|(?:mu|sd)_[BS]\[[0-4]\]|(?:mu0|sd0)_[BS]|(?:mu|sd)_[BS] layout)"
_MEMBER = r"(?:(?:joint|fixed)__" + _LEVEL + r"__[BS]__T|joint__[BS]__(?:ok|mu|sd|mu0|sd0))"
ID_RE = {"draw0": _LABEL + r":(?:tau_[BS]|tpr_(?:run1|run2|sim|simB))@" + _LEVEL,
         "d1": r"#(?:[1-9]|1[0-2]) (?:joint|joint_s507) " + _LABEL + " " + _QTY + r"@draw[0-9]{1,5}",
         "c1": _LABEL + " " + _QTY + r"@draw[0-9]{1,5}",
         "stop_id": _LABEL + " " + _MEMBER}  # selftest ids and slot keys: by membership in the pinned lists (load_allow)
VALUES_DIR_RE = r"/o/opt3_[AB]_[0-9]{8}_[0-9]{6}_[0-9a-f]{8}"
LAUNCH_ID_RE = r"[0-9]{10,16}"
D5_SLOTS = ("opt3/opt3_d5_slots.txt", "01e0c64c309959e492c1e8b40a1a3358ff2fc0b903538add71d3aaabfb11b7fa")  # addendum 8c @ 4b1204d43
NO_ALLOW = {"selftest": frozenset(), "slot": frozenset()}
REPORT_FIELDS = {
    "record": ("enum", ("A", "B")), "stop": ("stop", None), "stop_file": ("enum|null", ("boot", "reducer")), "stop_line": ("stop_line", None),
    "stop_ids": ("ids", "stop_id"), "input_file": ("enum|null", INPUT_FILES),
    "stages": ("list_enum", STAGES),
    "thread_env_pre": ("thread_map", None), "blas_pools_n": ("int|null", None), "blas_threads_max": ("int|null", None),
    "selftests_ok": ("bool|null", None), "selftests_failed": ("ids", "selftest"), "calib_sha_ok": ("bool|null", None),
    "c5_trials": ("int|null", None), "c5_compared": ("int|null", None), "c5_skipped_d7": ("int|null", None),
    "n_degenerate": ("count|null", None), "mains_outside_universe": ("int|null", None), "draw0_ok": ("bool|null", None), "draw0_failures": ("ids", "draw0"),
    "c0_scale": ("enum|null", ("fraction", "points")), "c0_ok": ("bool|null", None),
    "d1_identified": ("int|null", None), "d1_reading": ("enum|null", ("joint", "joint_s507")), "d1_s1_equals_s2": ("bool|null", None),
    "d1_candidate_failures": ("ids", "d1"),
    "pilot_projected_s": ("int|null", None), "pilot_elapsed_s": ("int|null", None), "pilot_rss_peak_mib": ("int|null", None),
    "pilot_rss_child_mib": ("int|null", None), "pilot_mem_projected_mib": ("int|null", None),
    "pilot_mem_source": ("enum|null", ("cgroup_v2", "cgroup_v1", "getrusage")),
    "c1_ok": ("bool|null", None), "c1_n_labels": ("int|null", None), "c1_n_labels_ok": ("int|null", None), "c1_gap_compared": ("int|null", None),
    "c1_gap_mismatch": ("int|null", None), "c1_gap_flip": ("int|null", None), "c1_ok_mask_diffs": ("int|null", None), "c1_first_failures": ("ids", "c1"),
    "c2_method": ("enum|null", METHODS), "c2_n_nodes": ("int|null", None), "c4_n": ("int|null", None), "c4_n_stopped": ("int|null", None),
    "n_nodes": ("int|null", None), "n_reproduced": ("int|null", None), "n_reproduced_flip": ("int|null", None), "n_not_reproduced": ("int|null", None),
    "n_not_like_for_like": ("int|null", None), "n_unresolved": ("int|null", None), "n_incomplete_draws": ("int|null", None), "mc_n_agree": ("int|null", None),
    "nan_by_node": ("nan_map", None), "d5_status": ("enum|null", D5_STATUS), "d5_pair": ("enum|null", D5_PAIRS), "d5_ok": ("bool|null", None),
    "d5_not_reproduced": ("ids", "slot"), "d5_unresolved": ("ids", "slot"), "n_d5_flip_sensitive": ("int|null", None),
    "verdict": ("enum", VERDICTS), "failing_nodes": ("nodes", None),
    "values_dir": ("values_dir", None), "values_sha256_node_series": ("hex64|null", None), "values_sha256_compare": ("hex64|null", None),
    "seconds": ("int", None)}


def load_allow():
    """09 (addendum 8c): the identifier allow-lists. Self-test ids = the pinned predictions file's group names and "group.key"s; D5
    slot keys = the pinned slot list. Both files are sha-checked. Returns {"selftest": set, "slot": set}."""
    for path, want in (PREDICTIONS, D5_SLOTS):
        if file_sha(Path("/", path)) != want:
            stop("allow-list sha")
    pred = json.loads(Path("/", PREDICTIONS[0]).read_bytes())
    ids = set()
    for g, v in pred.items():
        if not g.startswith("_"):
            ids.add(g)
            if isinstance(v, dict):
                ids |= {f"{g}.{k}" for k in v}
    return {"selftest": frozenset(ids), "slot": frozenset(ln for ln in Path("/", D5_SLOTS[0]).read_text().splitlines() if ln)}


def _id_ok(kind, x, nodes, allow):
    """One identifier: a string of <= 200 characters. Self-test ids and slot keys by membership in the pinned lists; node keys by
    membership in the record's node list; the rest by their grammar, whose level part is one of LEVEL_LIST."""
    if not isinstance(x, str) or len(x) > 200:
        return False
    if kind in ("selftest", "slot"):
        return x in allow[kind]
    if kind == "c1" and x.startswith("gap "):
        n, _sep, d = x[4:].rpartition("@draw")
        return n in nodes and re.fullmatch(r"[0-9]{1,5}", d) is not None
    if kind == "stop_id" and x in nodes:
        return True
    return re.fullmatch(ID_RE[kind], x) is not None


def _kind_ok(kind, vocab, v, nodes, allow):
    if kind == "enum":
        return isinstance(v, str) and v in vocab
    if kind == "enum|null":
        return v is None or (isinstance(v, str) and v in vocab)
    if kind == "stop":  # a fixed code, optionally " @<stage>" from STAGES (addendum 8d)
        if v is None:
            return True
        parts = v.split(" @") if isinstance(v, str) else []
        return isinstance(v, str) and (v in STOP_CODES or (len(parts) == 2 and parts[0] in STOP_CODES and parts[1] in STAGES))
    if kind == "count|null":
        return v is None or (_isint(v) and v >= 0)
    if kind == "stop_line":  # its per-file bound is a rule (addendum 8f)
        return v is None or (_isint(v) and v >= 1)
    if kind == "bool|null":
        return v is None or isinstance(v, bool)
    if kind == "int|null":
        return v is None or _isint(v)
    if kind == "int":
        return _isint(v)
    if kind == "list_enum":
        return isinstance(v, list) and all(isinstance(x, str) and x in vocab for x in v)
    if kind == "ids":
        return isinstance(v, list) and all(_id_ok(vocab, x, nodes, allow) for x in v)
    if kind == "nodes":
        return isinstance(v, list) and all(isinstance(x, str) and x in nodes for x in v) and v == sorted(set(v))
    if kind == "nan_map":
        keys = {k for k, _c in COUNT_KEYS}
        return isinstance(v, dict) and all(k in nodes and isinstance(c, dict) and set(c) == keys and all(_isint(c[q]) and c[q] >= 0 for q in keys)
                                           for k, c in v.items())
    if kind == "thread_map":
        return isinstance(v, dict) and (v == {} or set(v) == set(THREAD_VARS)) and all(_isint(x) and -2 <= x <= 9999 for x in v.values())
    if kind == "values_dir":
        return v is None or (isinstance(v, str) and re.fullmatch(VALUES_DIR_RE, v) is not None)
    if kind == "hex64|null":
        return v is None or (isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None)
    return False


def blank_report(rec, code="STOP: not started"):
    """Every field present (the crashed-worker / late fallback too, 09); `code` must be a STOP code (None only while running)."""
    out = {k: ([] if kind in ("ids", "nodes", "list_enum") else {} if kind in ("nan_map", "thread_map") else None) for k, (kind, _v) in REPORT_FIELDS.items()}
    out.update(record=rec, stop=code, verdict="STOP", seconds=0)
    return out


def validate_report(r, nodes, allow):
    """Problems (key names only) of one worker report against the exact schema and the shared rules (addenda 8 §6, 8b, 8c); [] if
    valid. `nodes` is the record's committed node list; `allow` the pinned self-test and slot lists (load_allow)."""
    if not isinstance(r, dict):
        return ["report not an object"]
    probs = [f"extra:{k}" for k in sorted(set(r) - set(REPORT_FIELDS))] + [f"missing:{k}" for k in sorted(set(REPORT_FIELDS) - set(r))]
    if probs:
        return probs
    probs = [f"type:{k}" for k, (kind, vocab) in REPORT_FIELDS.items() if not _kind_ok(kind, vocab, r[k], nodes, allow)]
    if probs:
        return probs
    stg = r["stages"]
    if (r["stop"] is None) != (r["verdict"] != "STOP"):
        probs.append("rule:stop<->verdict")
    if (r["stop_line"] is None) != (r["stop_file"] is None) or (r["stop_line"] is not None and (r["stop"] is None or r["stop_line"] > NLINES[r["stop_file"]])):
        probs.append("rule:stop_line")  # both or neither; only on a STOP; within that file's line count (addenda 8d, 8f)
    if r["input_file"] is not None and r["stop"] is None:
        probs.append("rule:input_file")
    if r["stop"] is not None and len(r["stop"].split(" @")) == 2 and r["stop"].split(" @")[1] in stg:
        probs.append("rule:stop_stage")  # the running stage cannot be a completed one (addendum 8d)
    if (r["c4_n"] != r["c4_n_stopped"] or (r["c4_n"] is not None and r["c4_n"] < 1) or (("c4" in stg) != (r["c4_n"] is not None))
            or (r["verdict"] != "STOP" and "c4" not in stg) or ("compare" in stg and r["c4_n"] is None)):
        probs.append("rule:c4")  # 09 (addendum 8b): a verdict needs C4 done, both counts non-null and equal
    if set(r["nan_by_node"]) != (set(nodes) if "draws" in stg else set()):
        probs.append("rule:nan_by_node")
    if r["verdict"] == "INTERVALS REPRODUCED (EXACT)" and not (
            all(r[k] is True for k in ("selftests_ok", "draw0_ok", "c0_ok", "c1_ok")) and r["n_nodes"] == len(nodes)
            and r["n_reproduced"] == r["n_nodes"] and r["failing_nodes"] == [] and "values" in stg
            and r["d5_status"] == ("fit" if r["record"] == "A" else None)):
        probs.append("rule:exact")
    if r["verdict"] == "NOT LIKE-FOR-LIKE (C0 scale unidentified)" and r["c0_ok"] is not False:
        probs.append("rule:c0")
    if len(stg) != len(set(stg)):
        probs.append("rule:stages")
    return probs


def validate_summary(s, nodes_ab, allow):
    """Problems of the driver summary: exact top keys {launch_epoch, launch_id, fan_out, driver_stop, workers}; launch_epoch a finite
    number, launch_id 10-16 digits, fan_out an integer; fan_out 2 with driver_stop null and workers [A, B], each valid against its own
    node list and the pinned allow-lists; or fan_out 0 with no workers and a driver_stop code."""
    if not isinstance(s, dict):
        return ["summary not an object"]
    top = {"launch_epoch", "launch_id", "fan_out", "driver_stop", "workers"}
    probs = [f"top:{k}" for k in sorted(set(s) ^ top)]
    if probs:
        return probs
    if not (_isnum(s["launch_epoch"]) and isinstance(s["launch_id"], str) and re.fullmatch(LAUNCH_ID_RE, s["launch_id"])
            and _isint(s["fan_out"]) and (s["driver_stop"] is None or (isinstance(s["driver_stop"], str) and s["driver_stop"] in STOP_CODES))):
        return ["top:types"]
    w = s["workers"]
    if s["fan_out"] == 2 and s["driver_stop"] is None and isinstance(w, list) and len(w) == 2:
        if [x.get("record") if isinstance(x, dict) else None for x in w] != ["A", "B"]:
            return ["workers:order"]
        for x, rec in zip(w, ("A", "B")):
            probs += validate_report(x, nodes_ab[rec], allow)
    elif not (s["fan_out"] == 0 and w == [] and s["driver_stop"] is not None):
        probs.append("fan_out/workers")
    return probs


# ---------------------------------------------------------------- C0 before the draws (Codex r1 #5; 09: no scale -> NOT LIKE-FOR-LIKE)
def c0_points(H, parsed, per0, levels):
    """Per node: (my draw-0 value, my draw-0 flip bound, the record's point). per0[label][mode] = the draw-0 eval (fixed == joint at draw 0)."""
    out = {}
    for n, p in parsed.items():
        lv = [i for i, x in enumerate(levels) if float(x) == float(p["level"])][0]
        out[n] = (float(node_series(p, per0, lv)[0]), float(node_flip(p, per0, lv)[0]), record_node(H, p)["point"])
    return out


def c0_identify(pts):
    """C0 (addendum 4 §4): the ONE scale (fraction x1 / points x100) under which every node's draw-0 point equals the record's within
    b + the draw-0 flip bound. 'fraction', 'points', or None (neither or both: C0 fails, every node NOT LIKE-FOR-LIKE)."""
    ok = []
    for name, s in (("fraction", 1.0), ("points", 100.0)):
        if all(_isnum(rp) and abs(s * m - rp) <= tol(rp) + s * fb for m, fb, rp in pts.values()):
            ok.append(name)
    return ok[0] if len(ok) == 1 else None


# ---------------------------------------------------------------- self-tests against frozen predictions (opt3/PREDICTIONS_r2_4.json, pinned)
PREDICTIONS = ("opt3/rlext/PREDICTIONS_r2_8_rlext.json", "d0b665dbcae5852587beab0c8b530e435a51d6e2ed33c55457b75220ce16d337")  # rlext (ADDENDUM_OPT3_10_rlext): r2_8 with templates.a_nodes_resolve 72|72
SELFTEST_DIR = Path("/tmp")  # the calibration fixtures' synthetic npz files
TEMPLATES = ("opt3/opt3_record_path_templates.tsv", "58785b3064ac5a5d21666bc9f9b7365e5f85d02a8812734296196270c3b5649c")  # 2e's keys-only map


_DROP = ("<drop>",)  # a fixture marker: remove this key from the synthetic leaf


def _fx_raise():
    raise KeyError("fixture")


def _fx_stop():
    stop("unknown call pattern")


def _fx_library():
    return json.loads("{")


def build_template_record(rec, split_coheadline=False):
    """Addendum 8e: a synthetic record with the REAL key layout of record `rec`, from 2e's keys-only path map: each leaf template
    expanded over the record's label keys, the modes joint / fixed and its level keys; every leaf a synthetic five-key object.
    split_coheadline nests "<group>|<readout>" as two keys (r2.5's wrong assumption; a negative). None if the map's sha differs."""
    if file_sha(Path("/", TEMPLATES[0])) != TEMPLATES[1]:
        return None
    tok = re.compile(r'"(?:[^"\\]|\\.)*"|<[a-z]+>|[A-Za-z0-9_]+')
    paths, sets = [], {}
    for ln in Path("/", TEMPLATES[0]).read_text().splitlines():
        r, _sep, body = ln.partition("\t")
        if r != rec or not body:
            continue
        if body.startswith("<"):  # a key-set line: '<label> keys under pooled ...: "a", "b"' / '<lvl> keys under coheadline...'
            name = body.split(" ", 1)[0] + (":coheadline" if " under coheadline_signed" in body else "")
            sets[name] = [json.loads(x) for x in re.findall(r'"(?:[^"\\]|\\.)*"', body.split(":", 1)[1])]
        else:
            paths.append([json.loads(t) if t.startswith('"') else t for t in tok.findall(body)])
    leaf = {"point": 0.2, "ci95": [0.1, 0.3], "draws": N_DRAWS, "finite_draws": N_DRAWS, "degenerate": False}
    H = {}
    for path in paths:
        lv = sets["<lvl>:coheadline"] if path[0] == "coheadline_signed" else sets["<lvl>"]
        combos = [[]]
        for t in path:
            opts = {"<label>": sets.get("<label>", []), "<mode>": ["joint", "fixed"], "<lvl>": lv}.get(t, [t])
            combos = [c + [o] for c in combos for o in opts]
        for keys in combos:
            if split_coheadline and keys[0] == "coheadline_signed":
                keys = keys[:4] + keys[4].split("|", 1) + keys[5:]
            d = H
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = dict(leaf, ci95=list(leaf["ci95"]))
    return H


def _template_fixtures():
    """Addendum 8e §3: the real key layout (2e's keys-only map) through the shared path, the resolve gate and C4."""
    out = {}
    nodes = {"A": read_nodes("A"), "B": read_nodes("B")}
    for rec in ("A", "B"):
        H = build_template_record(rec)
        if H is None:
            return {"templates": "sha"}
        ok = 0
        for n in nodes[rec]:
            try:
                ok += int(_leaf_ok(walk(H, parse_node(n))[0]))
            except (KeyError, TypeError, Stop):
                pass
        out[f"{rec.lower()}_nodes_resolve"] = f"{ok}|{len(nodes[rec])}"
        try:
            out[f"{rec.lower()}_resolve_gate"] = f"ok|{resolve_check(H, nodes[rec])}"
        except Stop as e:
            out[f"{rec.lower()}_resolve_gate"] = e.code
        nan_ = float("nan")
        ph = {n: {"lo": nan_, "hi": nan_, "draws": 0, "finite": 0, "point": nan_, "flip_bound": 0.0, "flip_bound_point": 0.0, "c0": False}
              for n in nodes[rec]}
        try:
            fx = c4_fixtures(H, nodes[rec], ph, 1.0, False, {n: (nan_, nan_) for n in nodes[rec]})
            out[f"{rec.lower()}_c4_counts"] = f"{len(fx)}|{sum(fx.values())}"
        except Stop:
            out[f"{rec.lower()}_c4_counts"] = "stop"
        if rec == "A":
            try:
                resolve_check(build_template_record("A", split_coheadline=True), nodes["A"])
                out["split_shape_gate"] = "passed"
            except Stop as e:
                out["split_shape_gate"] = e.code
            try:
                walk(H, parse_node(f"pooled|FIBS|uncut C|{COMBO}|joint|0.01|gap|delta_full"))
                out["label_pipe_kept"] = "ok"
            except (KeyError, TypeError, Stop):
                out["label_pipe_kept"] = "fail"
    out["leaf_pipe_rejected"] = _val(lambda: parse_node(f"pooled|FIBS|uncut C|{COMBO}|joint|0.01|catch|sim|B"))
    return out


def _inputs_fixtures():
    """Addendum 8f §4: the inputs gate on synthetic files in temp dirs (names only matter; the contents are a few bytes)."""
    import shutil
    import tempfile

    def h(b):
        return sha(b)

    def scene(files, cfg_have, pool_have, overrides, reg=("z/reg.json", b"r"), IN=None):
        root = Path(tempfile.mkdtemp(prefix="opt3_inputs_", dir=SELFTEST_DIR))
        cfg_dir, pool_dir = root / "cfg", root / "p"
        cfg_dir.mkdir()
        pool_dir.mkdir()
        for name, b in cfg_have.items():
            (cfg_dir / name).write_bytes(b)
        for name in pool_have:
            (pool_dir / name).write_bytes(b"{}")
        run = {"files": [[h(b), "c", f"x/{n}"] for n, b in files.items()], "inputs_full": "full.json", "extra_root_worlds": {"root1": {}},
               "catalogue": {"path": "cat.json"}, "manifest": {"path": "man.json"}}
        full = {"overrides": {k: {"path": f"y/{n}", "sha256": h(b)} for k, (n, b) in overrides.items()},
                "s_old_simh_extra_roots": {"root1": {"path": reg[0], "sha256": h(reg[1])}}}
        if "full.json" in cfg_have:
            (cfg_dir / "full.json").write_bytes(json.dumps(full).encode())
            run["files"] = [[h(json.dumps(full).encode()) if n == "full.json" else h(b), "c", f"x/{n}"] for n, b in files.items()]
        try:
            return f"ok|{inputs_gate(run, IN or {k: 'a.json' for k in CFG_KEYS}, cfg_dir, pool_dir)}"
        except Stop as e:
            return f"{e.code}|{e.input_file}"
        finally:
            shutil.rmtree(root)

    base_files = {"a.json": b"a", "full.json": b"f"}
    ov = {"S_old|R": ("ov.json", b"o")}
    have = {"a.json": b"a", "full.json": b"f", "ov.json": b"o", "reg.json": b"r"}
    pool = ("cat.json", "man.json")
    real_ov = {"S_old|C": ("override_S_old_C_2026-10-17_151056_gated.json", b"c"), "S_old|R": ("override_S_old_R_2026-10-17_151100_gated.json", b"r")}
    return {"all_present": scene(base_files, have, pool, ov),
            "missing_override": scene(base_files, {k: v for k, v in have.items() if k != "ov.json"}, pool, ov),
            "missing_registry": scene(base_files, {k: v for k, v in have.items() if k != "reg.json"}, pool, ov),
            "missing_catalogue": scene(base_files, have, ("man.json",), ov),
            "sha_mismatch": scene(base_files, dict(have, **{"ov.json": b"X"}), pool, ov),
            "cfg_key_not_pinned": scene(base_files, have, pool, ov, IN=dict({k: "a.json" for k in CFG_KEYS}, v2="b.json")),
            "pinned_twice": scene(dict(base_files, **{"ov.json": b"o2"}), dict(have, **{"ov.json": b"o2"}), pool, ov),
            "r26_input_set": scene(base_files, {"a.json": b"a", "full.json": b"f"}, pool, real_ov,
                                   reg=("z/simh_counted_keys_pilot_v3.json", b"s"))}


def _d7_world(R, drop_fold0=False):
    """09 (R1 #4) main-path fixture world (addendum 8 §3.2): 20 mains, folds m % 5. Mains 0-9 carry the cohort (one cell each, one
    3-unit attack run per arm); mains 10-19 are pool-only. Every main has ONE 3-unit pool run per role. drop_fold0 keeps only main
    0's pool runs in fold 0, so fold 0 has one run at draw 0 (D7 under the runs rule, valid under 507's units rule)."""
    import numpy as np

    rng = np.random.default_rng(20261022)
    run, pool, used, main_of = {}, {}, [], {}
    for m in range(20):
        mn, fold = f"m{m}", m % 5
        if m < 10:
            c = f"c{m}"
            main_of[c] = mn
            used.append(("S_old", c))
            for a in R.ARMS:
                run[f"{a}|{c}"] = {"r": {"score": rng.normal(size=3), "sbm": None, "unit": np.arange(3)}, "fold": fold, "main": mn, "zero": False}
        if drop_fold0 and fold == 0 and m != 0:
            continue
        for role in ("B", "S"):
            key = f"w{role}|p{m}"
            pool[key] = {"r": {"score": rng.normal(size=3), "sbm": None, "unit": np.arange(3)}, "fold": fold, "main": mn, "zero": False,
                         "role": role, "key": key}
    return {"run": run, "pool": pool, "used": used, "single": False, "main": main_of}


def _d7_fixture(R):
    """Rows: 0 all ones; 1 fold 0 keeps only main 0 (one run, 3 units); 2 fold 0 all zero; 3 the cohort mains zero, the pool-only
    mains one (valid pools, empty cohort); 4 all ones. Levels 0.01 (k_d = 0 at 60 or 30 units: D6) and 0.2."""
    import numpy as np

    lv = ["0.01", "0.2"]
    M = np.ones((5, 20), dtype=np.int32)
    M[1, [5, 10, 15]] = 0
    M[2, [0, 5, 10, 15]] = 0
    M[3, :10] = 0
    main_idx = {f"m{i}": i for i in range(20)}
    rows = list(range(5))

    def codes(e, li):
        return ",".join(str(int(c)) for c in e["reason"]["run1"][:, li])

    def fin(a):
        return "".join("T" if np.isfinite(v) else "F" for v in a)

    out = {}
    try:
        LB, LBd = build_label(R, _d7_world(R), main_idx), build_label(R, _d7_world(R, drop_fold0=True), main_idx)
        ej, ef = eval_chunk(LB, M, rows, lv, "joint"), eval_chunk(LB, M, rows, lv, "fixed")
        dj, df = eval_chunk(LBd, M, rows, lv, "joint"), eval_chunk(LBd, M, rows, lv, "fixed")
        out.update(joint_codes_l20=codes(ej, 1), joint_codes_l01=codes(ej, 0), joint_tau_b_finite_l20=fin(ej["tau"]["B"][:, 1]),
                   joint_tpr_run1_finite_l20=fin(ej["tpr"]["run1"][:, 1]), fixed_codes_l20=codes(ef, 1), fixed_codes_l01=codes(ef, 0),
                   drop_joint_codes_l20=codes(dj, 1), drop_fixed_codes_l20=codes(df, 1))
        p = {"family": "gap", "label": "D7 C", "mode": "joint", "level": "0.2", "name": "delta_full"}
        x = node_series(p, {"D7 C": {"joint": ej}}, 1)
        out["node_counts"] = ",".join(str(v) for v in node_nan_counts(p, {"D7 C": {"joint": ej}}, 1, x).values())
        pf = dict(p, mode="fixed")
        xf = node_series(pf, {"D7 C": {"fixed": df}}, 1)
        interval(xf, "linear")
        node_nan_counts(pf, {"D7 C": {"fixed": df}}, 1, xf)
        out["no_crash"] = "ok"
    except Exception as e:  # recorded as the fixture's outcome (it must equal "ok"), not swallowed
        out["no_crash"] = f"raise {type(e).__name__}"
    return out


def _val(fn):
    """A fixture's outcome: its value, 'stop' on a Stop, or 'raise <Name>' on any other exception (compared with the prediction)."""
    try:
        return fn()
    except Stop:
        return "stop"
    except Exception as e:
        return f"raise {type(e).__name__}"


def _summary_fixtures():
    """The 60 schema fixtures (addenda 8 §6, 8b §2, 8c §3 and §6, 8d §4, 8e §3, 8f §4), on synthetic node lists and allow-lists {A: [nA], B: [nB]}; mirrored by the results RUN_ME's jq fixtures."""
    nodes_ab = {"A": ["nA"], "B": ["nB"]}
    allow = {"selftest": frozenset({"c5", "c5.failures"}), "slot": frozenset({"s1.max_all"})}  # synthetic allow-lists (addendum 8c §6)

    def exact(rec):
        r = blank_report(rec, None)
        r.update(stages=[s for s in STAGES if rec == "A" or s != "d5"], selftests_ok=True, draw0_ok=True, c0_ok=True, c0_scale="fraction",
                 c1_ok=True, c4_n=5, c4_n_stopped=5, n_nodes=1, n_reproduced=1, nan_by_node={nodes_ab[rec][0]: {k: 0 for k, _c in COUNT_KEYS}},
                 d5_status="fit" if rec == "A" else None, verdict="INTERVALS REPRODUCED (EXACT)",
                 values_dir=f"/o/opt3_{rec}_20261021_120000_0123abcd")
        return r

    lid = "1792345678123"
    full = {"launch_epoch": 1792345678.123, "launch_id": lid, "fan_out": 2, "driver_stop": None, "workers": [blank_report("A"), blank_report("B")]}
    ex = dict(full, workers=[exact("A"), exact("B")])
    stp = {"launch_epoch": 1792345678.123, "launch_id": lid, "fan_out": 0, "driver_stop": "STOP: summary schema", "workers": []}

    def sc(basis, mut=None):
        s = json.loads(json.dumps(basis))
        if mut:
            mut(s)
        return "ok" if not validate_summary(s, nodes_ab, allow) else "fail"

    def w0(**kv):
        return lambda s: s["workers"][0].update(**kv)

    def nan0(**kv):
        return lambda s: s["workers"][0]["nan_by_node"]["nA"].update(**kv)

    crashed = [blank_report("A", "STOP: worker exception RemoteError"), blank_report("B", "STOP: worker exception other")]
    return {"valid_full": sc(full), "valid_stop": sc(stp), "valid_exact": sc(ex), "crashed_worker": sc(full, lambda s: s.update(workers=crashed)),
            "extra_key": sc(full, w0(extra=1)), "float_in_int": sc(full, w0(c4_n=1.5, c4_n_stopped=1.5)), "int_as_float": sc(full, w0(seconds=1.0)),
            "c4_unequal": sc(full, w0(c4_n=3, c4_n_stopped=2)), "c4_null_vs_3": sc(full, w0(c4_n=None, c4_n_stopped=3)),
            "c4_zero": sc(full, w0(c4_n=0, c4_n_stopped=0)), "compare_without_c4": sc(full, w0(stages=["compare"])),
            "verdict_free_text": sc(full, w0(stop=None, verdict="NOT REPRODUCED: 0.52")), "stop_free_text": sc(full, w0(stop="STOP: 0.52")),
            "stop_exception_unknown": sc(full, w0(stop="STOP: exception Foo")), "stop_null_verdict_stop": sc(full, w0(stop=None)),
            "workers_order": sc(full, lambda s: s.update(workers=list(reversed(s["workers"])))),
            "fan_out_false": sc(stp, lambda s: s.update(fan_out=False)), "fan_out_float": sc(stp, lambda s: s.update(fan_out=0.0)),
            "launch_id_free": sc(full, lambda s: s.update(launch_id="0.52")), "values_dir_free": sc(ex, w0(values_dir="/o/0.52")),
            "id_free_text": sc(full, w0(d1_candidate_failures=["0.52"])),
            "id_valid": sc(full, w0(d1_candidate_failures=["#3 joint FIBS|uncut C tau_B@0.01@draw7"])),
            "gap_id_valid": sc(full, w0(c1_first_failures=["gap nA@draw17"])), "gap_id_unknown_node": sc(full, w0(c1_first_failures=["gap zz@draw17"])),
            "failing_node_unknown": sc(full, w0(failing_nodes=["zz"])), "nan_missing_node": sc(full, w0(stages=["draws"])),
            "nan_extra_key": sc(ex, nan0(extra=0)), "nan_negative": sc(ex, nan0(d7=-1)), "exact_without_c1": sc(ex, w0(c1_ok=False)),
            "exact_a_d5_ambiguous": sc(ex, w0(d5_status="ambiguous")),
            "c0_verdict_without_c0_false": sc(full, w0(stop=None, verdict="NOT LIKE-FOR-LIKE (C0 scale unidentified)")),
            "driver_stop_free": sc(stp, lambda s: s.update(driver_stop="STOP: x")),
            "verdict_with_null_c4": sc(full, w0(stop=None, verdict="UNRESOLVED (stream gate)")),
            "stream_gate_with_c4": sc(full, w0(stop=None, verdict="UNRESOLVED (stream gate)", c4_n=5, c4_n_stopped=5,
                                                stages=["code_pins", "selftests", "pins", "manifests", "nodes", "c4", "draw0", "c0"])),
            "c4_stage_without_counts": sc(full, w0(stages=["c4"])), "counts_without_c4_stage": sc(full, w0(c4_n=5, c4_n_stopped=5)),
            "launch_epoch_infinite": sc(full, lambda s: s.update(launch_epoch=float("inf"))),
            "planted_selftest": sc(full, w0(selftests_failed=["0.52"])), "selftest_valid": sc(full, w0(selftests_failed=["c5.failures"])),
            "planted_slot": sc(full, w0(d5_not_reproduced=["0.52"])), "slot_valid": sc(full, w0(d5_not_reproduced=["s1.max_all"])),
            "planted_level": sc(full, w0(draw0_failures=["FIBS|uncut C:tau_B@0.52"])),
            "level_valid": sc(full, w0(draw0_failures=["FIBS|uncut C:tau_B@0.01"])),
            "stop_line_valid": sc(full, w0(stop="STOP: exception KeyError @c4", stop_file="boot", stop_line=10,
                                         stages=list(STAGES[:STAGES.index("resolve") + 1]))),
            "stop_line_zero": sc(full, w0(stop_line=0)), "stop_line_too_big": sc(full, w0(stop_line=BOOT_NLINES + 1)),
            "stop_line_float": sc(full, w0(stop_line=10.0)), "stop_line_on_non_stop": sc(ex, w0(stop_line=10)),
            "stage_suffix_valid": sc(full, w0(stop="STOP: reducer blob @code_pins")),
            "stage_suffix_unknown": sc(full, w0(stop="STOP: reducer blob @c9")),
            "stage_suffix_completed": sc(full, w0(stop="STOP: exception KeyError @nodes", stages=list(STAGES[:STAGES.index("nodes") + 1]))),
            "n_degenerate_negative": sc(full, w0(n_degenerate=-1)), "n_degenerate_float": sc(full, w0(n_degenerate=1.0)),
            "stop_file_valid": sc(full, w0(stop_file="reducer", stop_line=100)), "stop_file_unknown": sc(full, w0(stop_file="q1_eval", stop_line=10)),
            "stop_line_reducer_too_big": sc(full, w0(stop_file="reducer", stop_line=REDUCER_NLINES + 1)),
            "stop_line_without_file": sc(full, w0(stop_line=10)),
            "input_file_valid": sc(full, w0(stop="STOP: input missing @code_pins", input_file="override_S_old_C_2026-10-17_151056_gated.json")),
            "input_file_unknown": sc(full, w0(input_file="x.json")),
            "input_file_without_stop": sc(ex, w0(input_file="override_S_old_C_2026-10-17_151056_gated.json"))}


def run_selftests(R):
    """Every new or changed check's fixtures (Codex r1 / r2; 09), run before any producer read; the outcomes must equal the frozen
    predictions exactly (a missing, extra or different outcome fails). Returns (names of failing fixtures, C5's result)."""
    import numpy as np

    if file_sha(Path("/", PREDICTIONS[0])) != PREDICTIONS[1]:
        stop("predictions file sha")
    pred = json.loads(Path("/", PREDICTIONS[0]).read_bytes())
    got = {}
    got["d5_fit"] = {"zero_tokens": d5_fit("no numbers here", 0.1234), "two_equal_tokens": d5_fit("x 0.123 and 0.123", 0.1234),
                     "other_rounding": d5_fit("hw 0.120", 0.1234), "percent_ignored": d5_fit("95% interval half-width 0.123", 0.1234),
                     "both_scales_fit": d5_fit("0.5 or 50", 0.5), "zero_invariant": d5_fit("0", 0.0),
                     "per_scale_fraction": d5_fit("0.12 and 12 and 12", 0.12)}

    def dc(x):
        st, pr, nr, un, _fl = d5_classify(x)
        return f"{st}|{'-' if pr is None else pr[0] + ':' + pr[1]}|{','.join(nr)}|{','.join(un)}"

    H_, M_ = "half_range", "max_side"
    got["d5_classify"] = {"mixed_invariant": dc({"e1": {H_: "fit:invariant", M_: "none"}, "e2": {H_: "fit:fraction", M_: "none"}}),
                          "all_invariant": dc({"e1": {H_: "fit:invariant", M_: "none"}}),
                          "two_readings": dc({"e1": {H_: "fit:fraction", M_: "fit:fraction"}}),
                          "both_invariant": dc({"e1": {H_: "fit:invariant", M_: "fit:invariant"}}),
                          "scale_mix": dc({"e1": {H_: "fit:fraction", M_: "none"}, "e2": {H_: "fit:points", M_: "none"}}),
                          "one_none": dc({"e1": {H_: "fit:fraction", M_: "none"}, "e2": {H_: "none", M_: "none"}}),
                          "one_unresolved": dc({"e1": {H_: "fit:fraction", M_: "none"}, "e2": {H_: "unresolved", M_: "none"}}),
                          "nr_beats_unresolved": dc({"e1": {H_: "none", M_: "none"}, "e2": {H_: "unresolved", M_: "none"}}),
                          "no_elements": dc({})}
    ok_rows, nr_rows, nl_rows = ({"n": {"verdict": v}} for v in ("REPRODUCED", "NOT REPRODUCED", "NOT LIKE-FOR-LIKE"))
    got["verdict_line"] = {"b_exact": verdict_line(ok_rows, True, None), "a_fit": verdict_line(ok_rows, True, {"status": "fit"}),
                           "a_ambiguous": verdict_line(ok_rows, True, {"status": "ambiguous"}),
                           "a_mixed": verdict_line(ok_rows, True, {"status": "mixed"}),
                           "a_unresolved": verdict_line(ok_rows, True, {"status": "unresolved"}),
                           "a_error": verdict_line(ok_rows, True, {"status": "error"}),
                           "a_not_reproduced": verdict_line(ok_rows, True, {"status": "not_reproduced"}),
                           "node_nr_d5_ambiguous": verdict_line(nr_rows, True, {"status": "ambiguous"}),
                           "node_nlfl_d5_nr": verdict_line(nl_rows, True, {"status": "not_reproduced"}),
                           "stream_gate": verdict_line(ok_rows, False, {"status": "fit"})}
    sk = "10.halfwidth_C.max_all"
    sl = {sk: "hw 0.123"}
    ln = [{"element": "halfwidth_x", "key_path": ["slots", sk], "a_node": "nA|ci95[0]"}]

    def rdy(slots, lines, slot_list=frozenset({sk})):
        return _val(lambda: (d5_ready(slots, lines, {"nA"}, slot_list), "ok")[1])

    got["d5_ready"] = {"ready": rdy(sl, ln), "key_path_form": rdy(sl, [dict(ln[0], key_path=["x", sk])]), "slot_missing": rdy({}, ln),
                       "slot_not_string": rdy({sk: 0.5}, ln),
                       "aggregator_unknown": rdy({"10.halfwidth_C.mean": "hw 0.123"}, [dict(ln[0], key_path=["slots", "10.halfwidth_C.mean"])]),
                       "node_not_in_a": rdy(sl, [dict(ln[0], a_node="zz|ci95[0]")]), "no_elements": rdy(sl, []),
                       "slot_list_differs": rdy(sl, ln, frozenset({"other.max_all"}))}
    lab = "X|y C"
    node = f"pooled|{lab}|{COMBO}|joint|0.01|gap|delta_full"

    def cmp(rec_over, mine_over, exact=True, se=(0.01, 0.01)):
        rec = {"ci95": [0.1, 0.3], "draws": N_DRAWS, "finite_draws": N_DRAWS, "point": 0.2}
        rec.update(rec_over)
        H = {"pooled": {lab: {COMBO: {"joint": {"0.01": {"gap": {"delta_full": rec}}}}}}}
        m = {"lo": 0.1, "hi": 0.3, "draws": N_DRAWS, "finite": N_DRAWS, "point": 0.2, "flip_bound": 0.0, "flip_bound_point": 0.0, "c0": True}
        m.update(mine_over)
        try:
            r = compare_core(H, [node], {node: m}, 1.0, exact, {node: se})[node]
        except Exception:
            return "raise"
        return r["mc_diagnostic_agrees"] if not exact else f"{r['verdict']}|{r['label']}"

    got["compare"] = {"exact_equal": cmp({}, {}), "flip_only": cmp({}, {"lo": 0.1 + 1e-6, "flip_bound": 1e-5}),
                      "beyond_flip": cmp({}, {"lo": 0.1 + 1e-3, "flip_bound": 1e-5}), "record_finite_9499": cmp({"finite_draws": 9499}, {}),
                      "mine_finite_9499": cmp({}, {"finite": 9499}), "record_draws_float": cmp({"draws": float(N_DRAWS)}, {}),
                      "record_draws_9999": cmp({"draws": 9999}, {}), "c0_false": cmp({}, {"c0": False}),
                      "mc_ends_ok_counts_far": cmp({}, {"finite": 9600}, exact=False), "mc_both_ok": cmp({}, {}, exact=False),
                      "mine_lo_nan": cmp({}, {"lo": float("nan"), "finite": 9600}), "record_finite_10001": cmp({"finite_draws": 10001}, {})}
    # addendum 8d: a co-headline node in the REAL record shape ("<group>|<readout>" one key) and in r2.5's assumed split shape
    co = f"coheadline_signed|{COMBO}|joint|0.01|headline_six_full|C|S"
    co_rec = {"ci95": [0.1, 0.3], "draws": N_DRAWS, "finite_draws": N_DRAWS, "point": 0.2}
    co_m = {"lo": 0.1, "hi": 0.3, "draws": N_DRAWS, "finite": N_DRAWS, "point": 0.2, "flip_bound": 0.0, "flip_bound_point": 0.0, "c0": True}
    H_co = {"coheadline_signed": {COMBO: {"joint": {"0.01": {"headline_six_full|C": {"S": dict(co_rec)}}}}}}
    H_split = {"coheadline_signed": {COMBO: {"joint": {"0.01": {"headline_six_full": {"C": {"S": dict(co_rec)}}}}}}}

    def cmp_h(H, n):
        try:
            r = compare_core(H, [n], {n: co_m}, 1.0, True, {n: (0.01, 0.01)})[n]
        except Exception:
            return "raise"
        return f"{r['verdict']}|{r['label']}"

    got["compare"].update(coheadline_real_shape=cmp_h(H_co, co), coheadline_split_shape=cmp_h(H_split, co))
    H_gap = {"pooled": {lab: {COMBO: {"joint": {"0.01": {"gap": {"delta_full": dict(co_rec)}}}}}}}

    def c4n(H, n):
        fx = _val(lambda: c4_fixtures(H, [n], {n: co_m}, 1.0, True, {n: (0.01, 0.01)}))
        return f"{len(fx)}|{sum(fx.values())}" if isinstance(fx, dict) else fx

    got["c4_shape"] = {"coheadline_real_shape": c4n(H_co, co), "pooled_gap_shape": c4n(H_gap, node)}
    got["c0_scale"] = {"fraction_unique": c0_identify({"a": (0.2, 0.0, 0.2)}), "points_unique": c0_identify({"a": (0.2, 0.0, 20.0)}),
                       "neither": c0_identify({"a": (0.2, 0.0, 0.5)}), "both": c0_identify({"a": (0.0, 0.0, 0.0)})}
    N1 = N_DRAWS + 1
    got["c2"] = {"empty": _val(lambda: c2_select({}, {0: {}})),
                 "missing_member": _val(lambda: c2_select({0: np.zeros(N1)}, {0: {}, 1: {}})),
                 "rule_i1": _val(lambda: c2_index_rule({0: np.zeros(N1), 1: np.zeros(N1)})),
                 "rule_i2": _val(lambda: c2_index_rule({0: np.zeros(N_DRAWS)})),
                 "rule_mixed_lengths": _val(lambda: c2_index_rule({0: np.zeros(N1), 1: np.zeros(N_DRAWS)})),
                 "rule_2d": _val(lambda: c2_index_rule({0: np.zeros((N1, 2))}))}

    def npz(name, arrays):
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        p = SELFTEST_DIR / name
        p.write_bytes(buf.getvalue())
        return p

    zeros, ones, okv = np.zeros(N1), np.ones(N1), np.ones(N1, bool)
    ev = {"tau": {"B": np.zeros((N1, 1)), "S": np.zeros((N1, 1))}, "ok": {"B": okv, "S": okv}, "mom": [{"B": {}, "S": {}}]}
    momB = {str(k): [zeros, ones, okv, okv] for k in (0, 1, 2, 3, 4, -1)}
    evm = dict(ev, mom=[{"B": momB, "S": {}}])
    f1 = npz("opt3_st_c1.npz", {"joint__0.01__B__T": np.zeros(N_DRAWS), "joint__0.01__S__T": zeros, "joint__B__ok": okv, "joint__S__ok": okv})
    f2 = npz("opt3_st_c2.npz", {"joint__0.01__B__T": np.zeros((N1, 5)), "joint__0.01__S__T": zeros, "joint__B__ok": okv, "joint__S__ok": okv})
    base3 = {"joint__0.01__B__T": zeros, "joint__0.01__S__T": zeros, "joint__S__ok": okv, "joint__B__mu": np.zeros((5, N1)),
             "joint__B__sd": np.ones((5, N1)), "joint__B__sd0": ones}
    f3 = npz("opt3_st_c3.npz", dict(base3, joint__B__ok=okv, joint__B__mu0=np.zeros((N1, 5))))
    f4 = npz("opt3_st_c4.npz", dict(base3, joint__B__ok=np.ones((N1, 7), bool), joint__B__mu0=zeros))
    _CAL_CACHE.clear()
    got["calib"] = {"strict_layout": _val(lambda: calib_check(f1, ev, [1], ["0.01"], "I1", strict=True)),
                    "lenient_layout_mismatch": _val(lambda: calib_check(f1, ev, [1], ["0.01"], "I1")[1] > 0),
                    "strict_t_2d": _val(lambda: calib_check(f2, ev, [1], ["0.01"], "I1", strict=True)),
                    "lenient_t_2d": _val(lambda: calib_check(f2, ev, [1], ["0.01"], "I1")[1] > 0),
                    "strict_mu0_2d": _val(lambda: calib_check(f3, evm, [1], ["0.01"], "I1", strict=True)),
                    "good_strict": _val(lambda: "{1}|{3}".format(*calib_check(f4, evm, [1], ["0.01"], "I1", strict=True)))}
    _CAL_CACHE.clear()
    gp = {node: parse_node(node)}
    gkey = (lab, "joint", 0.01, "delta_full")
    x7, x9, fb9 = np.zeros(N1), np.zeros(N1), np.zeros(N1)
    x7[7], x9[9], fb9[9] = 1.0, 5e-9, 1e-8

    def gc(stored, vals, fb):
        return gap_compare(stored, gp, {node: vals}, {node: fb}, 1.0, "I1")

    got["gap"] = {"one_draw_off": _val(lambda: "{1}|{3[0]}".format(*gc({gkey: np.zeros(N1)}, x7, np.zeros(N1)))),
                  "flip_only": _val(lambda: "{1}|{2}".format(*gc({gkey: np.zeros(N1)}, x9, fb9))),
                  "missing_member": _val(lambda: gc({}, x7, np.zeros(N1))),
                  "wrong_length": _val(lambda: gc({gkey: np.zeros(N_DRAWS)}, x7, np.zeros(N1)))}
    got["summary_schema"] = _summary_fixtures()

    class _Unlisted(Exception):
        pass

    def code_of(fn):
        try:
            fn()
        except Stop as e:
            return e.code
        return "no stop"

    got["stop_codes"] = {"known_code": code_of(lambda: stop("reducer blob")), "unknown_code": code_of(lambda: stop("0.52")),
                         "exc_known": exc_code("worker ", KeyError()), "exc_unknown": exc_code("worker ", _Unlisted())}
    G, K = 2 ** 30, 2 ** 20

    def mo(*a):
        return "stop" if mem_over(mem_projection_mib(*a)) else "pass"

    got["pilot_mem"] = {"cgroup_small": mo("cgroup_v2", 1 * G, 3 * G, 0, 0, 0, 16, 10 ** 6),
                        "cgroup_over": mo("cgroup_v2", 2 * G, 10 * G, 0, 0, 0, 2, 10 ** 6),
                        "getrusage_small": mo("getrusage", 0, 0, 4 * K, 4.5 * K, 4 * K, 16, 10 ** 6),
                        "getrusage_over": mo("getrusage", 0, 0, 4 * K, 8 * K, 4 * K, 16, 10 ** 6)}
    got["d7_main_path"] = _d7_fixture(R)
    # addenda 8d / 8e: self-locating STOPs (the line of the deepest frame here, the running stage), the template records, the leaf gate
    def line_of(fn):
        try:
            fn()
        except Exception as e:
            return _stop_where(e)
        return None, None

    k_ = "STOP: exception KeyError"
    upto = list(STAGES[:STAGES.index("nodes") + 1])
    got["stop_where"] = {"line_raise": line_of(_fx_raise) == ("boot", _fx_raise.__code__.co_firstlineno + 1),
                         "line_stop_call_site": line_of(_fx_stop) == ("boot", _fx_stop.__code__.co_firstlineno + 1),
                         "line_library_call_site": line_of(_fx_library) == ("boot", _fx_library.__code__.co_firstlineno + 1),
                         "reducer_frame": line_of(lambda: R.jload("/nonexistent/opt3_fixture.json")) == ("reducer", R.jload.__code__.co_firstlineno + 1),
                         "stage_suffix_c4": stop_code_at(k_, upto + ["resolve"], "A"), "stage_suffix_resolve": stop_code_at(k_, upto, "A"),
                         "stage_suffix_code_pins": stop_code_at(k_, [], "A"), "stage_suffix_all_done": stop_code_at(k_, list(STAGES), "A"),
                         "stage_suffix_b_after_values": stop_code_at(k_, list(STAGES[:STAGES.index("values") + 1]), "B")}
    got["templates"] = _template_fixtures()
    got["inputs_gate"] = _inputs_fixtures()

    def rl(**kv):
        leaf = {"point": 0.2, "ci95": [0.1, 0.3], "draws": N_DRAWS, "finite_draws": N_DRAWS, "degenerate": False}
        leaf.update(kv)
        for k in [k for k, v in kv.items() if v is _DROP]:
            del leaf[k]
        H = {"pooled": {lab: {COMBO: {"joint": {"0.01": {"gap": {"delta_full": leaf}}}}}}}
        try:
            return f"ok|{resolve_check(H, [node])}"
        except Stop as e:
            return e.code

    got["resolve_leaf"] = {"missing_key": rl(point=_DROP), "extra_key": rl(extra=1), "degenerate_not_bool": rl(degenerate=0),
                           "draws_float": rl(draws=float(N_DRAWS)), "ci95_null_ok": rl(ci95=[None, 0.3]), "point_null_ok": rl(point=None),
                           "degenerate_counted": rl(degenerate=True)}
    # addendum 8c §4: D5 under flip-sensitive constituents; the rounding case: half-range 0.1149999 prints "0.11", the slot says "0.12"
    def dfl(x, fl):
        st, _pr, nr, _un, fls = d5_classify(x, fl)
        return f"{st}|{','.join(fls)}|{','.join(nr)}"

    NN, FR = {H_: "none", M_: "none"}, {H_: "fit:fraction", M_: "none"}
    rc_mine = {"nA": {"lo": 0.0, "hi": 0.2299998, "point": 0.1149999}}
    rc_slot = {"s1.max_all": {"nA"}}
    got["d5_flip"] = {"flip_relabels_nr": dfl({"e1": NN}, {"e1"}), "flip_with_other_nr": dfl({"e1": NN, "e2": NN}, {"e1"}),
                      "flip_fit_stays_fit": dfl({"e1": FR}, {"e1"}), "no_flip_nr": dfl({"e1": NN}, set()),
                      "rounding_crosses_flip": d5_halfwidth(rc_mine, rc_slot, {"s1.max_all": "hw 0.12"}, {"nA"})["status"],
                      "rounding_crosses_no_flip": d5_halfwidth(rc_mine, rc_slot, {"s1.max_all": "hw 0.12"}, set())["status"],
                      "verdict_flip": verdict_line(ok_rows, True, {"status": "flip_sensitive"})}
    # addendum 8c: Codex's tie case (two tied units of one run, multiplicity 2, alpha 0.5 -> k = 2) and a no-tie control
    pos_ = {("r", 0): 0, ("r", 1): 1}
    ks_, us_ = ["r#0", "r#0", "r#1", "r#1"], np.array([0, 1, 0, 1])
    _tv, ti = weighted_tau(np.array([1.0, 1.0]), np.array([2.0, 2.0]), ["0.5"])
    eng = ("r", ti[0])
    sh = ref_selected_unit(np.ones(4), ks_, us_, 2, pos_)
    cm = ref_selected_unit(np.ones(4), ks_, us_, 2, pos_, order="copy_major")
    _nv, ni = weighted_tau(np.array([3.0, 2.0]), np.array([2.0, 2.0]), ["0.5"])
    nz = np.array([3.0, 2.0, 3.0, 2.0])
    got["c5_tie"] = {"engine": f"{eng[0]}:{eng[1]}", "shared": f"{sh[0]}:{sh[1]}", "copy_major": f"{cm[0]}:{cm[1]}",
                     "match_shared": eng == sh[:2], "match_copy_major": eng == cm[:2], "shared_value_is_t": sh[2] == 1.0,
                     "no_tie_shared": "{0}:{1}".format(*ref_selected_unit(nz, ks_, us_, 2, pos_)),
                     "no_tie_copy_major": "{0}:{1}".format(*ref_selected_unit(nz, ks_, us_, 2, pos_, order="copy_major"))}
    if ("r", ni[0]) != ("r", 1):
        got["c5_tie"]["no_tie_shared"] = "engine differs"
    c5 = c5_weighted_path(R)
    c5_forced = c5_weighted_path(R, force_skip=True)  # addendum 7 §2 (d)
    got["c5"] = {"failures": len(c5["failures"]), "negatives_detected": c5["negatives_detected"],
                 "guard_real": "pass" if not c5_guard(c5) else "stop", "guard_all_skipped": "stop" if c5_guard(c5_forced) else "pass",
                 "compared": c5["compared"], "skipped_d7": c5["skipped_d7"], "empty": c5["empty"]}
    failed = []
    for grp in sorted(set(pred) | set(got)):
        if grp.startswith("_"):
            continue
        want, g = pred.get(grp), got.get(grp)
        if isinstance(want, dict) and isinstance(g, dict):
            failed += [f"{grp}.{k}" for k in sorted(set(want) | set(g)) if k not in want or k not in g or g[k] != want[k]]
        elif g != want:
            failed.append(grp)
    return failed, c5


# ---------------------------------------------------------------- the worker (one per record) and the parallel draws
_G = {}  # read by forked worker processes: {'LB': {label: arrays}, 'M': multiplicities, 'levels': [...], 'blas': this process's check}
_TPL, _BLAS_SEEN = [], []  # the threadpool_limits objects kept alive (limits stay in effect); every (pools, max threads, blas pools) seen


def _blas_one():
    """09 (addendum 8a): one BLAS / OpenMP thread in THIS process. threadpool_limits(limits=1), kept in effect, then threadpool_info()
    read back. Returns (n pools, the largest num_threads any pool reports, n BLAS pools); _blas_gate decides."""
    import numpy  # noqa: F401  (loads the BLAS library, so threadpoolctl can see and limit it)
    from threadpoolctl import threadpool_info, threadpool_limits

    _TPL.append(threadpool_limits(limits=1))
    info = threadpool_info()
    return len(info), max((int(x.get("num_threads", 0)) for x in info), default=0), sum(1 for x in info if x.get("user_api") == "blas")


def _child_init():
    _G["blas"] = _blas_one()


def _blas_gate(seen):
    """STOP unless every process checked has >= 1 BLAS pool and every pool reports num_threads == 1 (not an env-string check)."""
    _BLAS_SEEN.extend(seen)
    if any(nb < 1 for _n, _mx, nb in seen):
        stop("no BLAS pool found")
    if any(mx != 1 for _n, mx, _nb in seen):
        stop("BLAS threads not 1")


def _staged_expect(job):
    import cleanroom_reduce_v2 as R

    return R.staged_expect(job), _G["blas"]


def _task(args):
    label, mode, lo, hi = args
    return label, mode, lo, eval_chunk(_G["LB"][label], _G["M"], list(range(lo, hi)), _G["levels"], mode), _G["blas"]


def run_draws(labels_modes, M, levels, n_last, chunk=CHUNK, first=0):
    """Draw rows first..n_last for every (label, mode), split in chunks over WORKERS forked processes; concatenated in row order."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    _G["M"], _G["levels"] = M, levels
    tasks = [(lab, mode, lo, min(lo + chunk, n_last + 1)) for lab, mode in labels_modes for lo in range(first, n_last + 1, chunk)]
    parts = {}
    blas = []
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("fork"), initializer=_child_init) as px:
        for lab, mode, lo, ev, b in px.map(_task, tasks):
            parts.setdefault((lab, mode), []).append((lo, ev))
            blas.append(b)
    _blas_gate(blas)
    per = {}
    for (lab, mode), lst in parts.items():
        lst.sort(key=lambda t: t[0])

        def cat(key, sub, lst=lst):
            return np.concatenate([ev[key][sub] for _lo, ev in lst])

        arms = ("run1", "run2", "sim", "simB")
        per.setdefault(lab, {})[mode] = {"tau": {sd: cat("tau", sd) for sd in ("B", "S")}, "tpr": {a: cat("tpr", a) for a in arms},
                                         "flip": {a: cat("flip", a) for a in arms}, "reason": {a: cat("reason", a) for a in arms},
                                         "ok": {sd: cat("ok", sd) for sd in ("B", "S")}, "mom": [ev["mom"] for _lo, ev in lst]}
    return per


# ---------------------------------------------------------------- pilot memory (09, R1 #19; addendum 8 §3.7)
CGROUP = (("cgroup_v2", "/sys/fs/cgroup/memory.current"), ("cgroup_v1", "/sys/fs/cgroup/memory/memory.usage_in_bytes"))


def _read_int(path):
    """The first integer of a cgroup / proc file; None if the file is absent or unreadable (a probe of optional files only)."""
    try:
        return int(Path(path).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def mem_source():
    for src, p in CGROUP:
        if _read_int(p) is not None:
            return src, p
    return "getrusage", None


def rss_now_kib():
    """This process's current RSS in KiB (/proc/self/statm); its peak (getrusage) if statm is unreadable."""
    import resource

    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") // 1024
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


class MemSampler:
    """The container's current memory, read every 0.25 s on a daemon thread while the pilot runs; .peak is the max seen."""

    def __init__(self, path):
        import threading

        self.path, self.peak, self._done = path, 0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._done.is_set():
            v = _read_int(self.path)
            self.peak = max(self.peak, v or 0)
            self._done.wait(0.25)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        self._t.join()


def mem_projection_mib(src, base_b, peak_b, self_kib, child_kib, rss_base_kib, conc, res_bytes):
    """The full run's memory from the pilot (MiB). cgroup: base + (sampled peak - base) x WORKERS / pilot concurrency + growth;
    getrusage: parent peak + WORKERS x max(0, child peak - parent RSS at the pool's start) + growth. growth = the pilot's result
    bytes scaled to N draws, doubled for the concatenation."""
    growth = 2.0 * res_bytes * N_DRAWS / PILOT_DRAWS
    if src in ("cgroup_v2", "cgroup_v1"):
        b = base_b + max(0, peak_b - base_b) * WORKERS / max(1, conc) + growth
    else:
        b = self_kib * 1024.0 + WORKERS * max(0.0, child_kib - rss_base_kib) * 1024.0 + growth
    return b / 2 ** 20


def mem_over(proj_mib):
    return proj_mib > MEM_FRACTION * MEM_LIMIT_MIB


def result_nbytes(x):
    """Bytes held in the numpy arrays of a nested result (dicts / lists)."""
    import numpy as np

    n, stack = 0, [x]
    while stack:
        y = stack.pop()
        if isinstance(y, np.ndarray):
            n += y.nbytes
        elif isinstance(y, dict):
            stack.extend(y.values())
        elif isinstance(y, (list, tuple)):
            stack.extend(y)
    return n


def read_nodes(rec):
    path, want = NODES[rec]
    if file_sha(Path("/", path)) != want:
        stop("node list sha")
    return [ln for ln in Path("/", path).read_text().splitlines() if ln]


def _stamp():
    import secrets

    return f"{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"


def run_record(rec: str) -> dict:
    """One record end to end; returns the exact-schema report (booleans, counts, fixed codes, validated identifiers), validated
    against the schema before it is returned. Values go to /o."""
    import gzip
    import resource

    import numpy as np

    t0 = time.time()
    out = blank_report(rec, None)  # stop is None while running; every failure path sets a fixed code
    stg = out["stages"]
    nodes, allow = [], NO_ALLOW
    out["thread_env_pre"] = dict(THREAD_ENV_PRE)
    try:
        # 0. one BLAS / OpenMP thread in this process, checked by threadpool_info (09, addendum 8a)
        _G["blas"] = _blas_one()
        _blas_gate([_G["blas"]])
        out["blas_pools_n"] = _G["blas"][0]
        # 1. code pins (Codex r1 #4) and the node list; the self-tests BEFORE any producer read (Codex r2 #5)
        if file_sha(Path("/root", REDUCER[0])) != REDUCER[1]:
            stop("reducer blob")
        if file_sha(Path("/run.json")) != PINS_SHA[rec] or file_sha(Path("/root/q1_eval.py")) != Q1_EVAL_SHA:
            stop("pins config / q1_eval sha")
        nodes = read_nodes(rec)
        allow = load_allow()  # the pinned self-test and slot lists the report's identifiers are checked against (addendum 8c)
        sys.path.insert(0, "/root")
        import cleanroom_reduce_v2 as R

        inputs_gate(json.loads(Path("/run.json").read_bytes()), R.IN)  # 09 (addendum 8f): before the self-tests and any record read
        stg.append("code_pins")
        failed, c5 = run_selftests(R)
        out["selftests_failed"], out["selftests_ok"] = failed, not failed
        out["c5_trials"], out["c5_compared"], out["c5_skipped_d7"] = c5["n_trials"], c5["compared"], c5["skipped_d7"]  # addendum 7 §2 (a)
        if failed:
            stop("self-tests differ from the frozen predictions")
        stg.append("selftests")
        # 2. input pins; A's rc1a pin and D5 parse-readiness before the draws (09, R1 #18); the producer record
        for code, (name, want) in (("pin scope", SCOPE), ("pin calib_json", CALIB_JSON), ("pin perdraw_json", PERDRAW_JSON)):
            if file_sha(Path("/opt3in", name)) != want:
                stop(code)
        scope_lines = [json.loads(ln) for ln in Path("/opt3in", SCOPE[0]).read_text().splitlines() if ln.strip()]
        slots = by_slot = None
        if rec == "A":
            if file_sha(Path("/opt3in", RC1A[0])) != RC1A[1]:
                stop("pin rc1a")
            slots = json.loads(Path("/opt3in", RC1A[0]).read_bytes())["slots"]
            by_slot = d5_ready(slots, scope_lines, set(nodes), allow["slot"])
        run_cfg_raw = json.loads(Path("/run.json").read_bytes())
        raw = gzip.decompress(Path("/cfg", run_cfg_raw["pooled_A"]["name"]).read_bytes())
        if sha(raw) != run_cfg_raw["pooled_A"]["decompressed_sha256"]:
            stop("record sha")
        H = json.loads(raw)
        del raw
        stg.append("pins")
        # 3. the union stream's universe (meta.merge, 09 ruling 1), the calibration manifest, the per-draw npz pin, the node rebuild
        mm = H["meta"]["merge"]
        mains, env_of = list(mm["mains"]), dict(mm["env_of_main"])
        if list(mm["seed"]) != SEED or mains != sorted(set(mains)) or set(env_of) != set(mains):
            stop("meta.merge layout")
        main_idx = {m: i for i, m in enumerate(mains)}
        cal2e = json.loads(Path("/opt3in", CALIB_JSON[0]).read_bytes())[rec]
        cal_rec = H["meta"]["pooled_world"]["calibrations"]["by_label_combo"]
        parsed = {}
        for n in nodes:
            try:
                parsed[n] = parse_node(n)
            except (AssertionError, ValueError):
                stop("unparsed node", [n])
        labels = sorted({m for p in parsed.values() for m in members(p)})
        modes = sorted({p["mode"] for p in parsed.values()})
        cal_ok = True
        for lab in labels:
            k = f"{lab} | {COMBO}"
            same = cal_rec[k]["path"] == cal2e[k]["path"] and cal_rec[k]["sha256"] == cal2e[k]["sha256"]
            cal_ok &= bool(same and file_sha(Path("/p", cal2e[k]["path"])) == cal2e[k]["sha256"])
        out["calib_sha_ok"] = cal_ok
        if not cal_ok:
            stop("calibration npz sha / manifest")
        gap_pin = GAP_NPZ[rec]
        pj = json.loads(Path("/opt3in", PERDRAW_JSON[0]).read_bytes())
        if [pj[f"{rec}_npz"]["path"], pj[f"{rec}_npz"]["sha256"]] != list(gap_pin) or file_sha(Path("/p", gap_pin[0])) != gap_pin[1]:
            stop("per-draw npz pin")
        stg.append("manifests")
        if sorted(rebuild_nodes(rec)) != sorted(nodes):
            stop("node list rebuild")
        stg.append("nodes")
        # the resolve gate (09, addendum 8e): every in-scope node through the shared path to a leaf of 09's shape, before C4 or any draw
        out["n_degenerate"] = resolve_check(H, nodes)
        stg.append("resolve")
        # C4 BEFORE any verdict (09, addendum 8b): each record key compare_core reads, removed in turn, must make it raise. My side
        # is a NaN placeholder: the record keys are read first, so a removal raises before the placeholder is touched.
        nan_ = float("nan")
        ph = {n: {"lo": nan_, "hi": nan_, "draws": 0, "finite": 0, "point": nan_, "flip_bound": 0.0, "flip_bound_point": 0.0, "c0": False}
              for n in nodes}
        fx = c4_fixtures(H, nodes, ph, 1.0, False, {n: (nan_, nan_) for n in nodes})
        if not fx or not all(fx.values()):  # recorded only when every fixture stopped
            stop("C4 fixture did not stop")
        out["c4_n"], out["c4_n_stopped"] = len(fx), sum(fx.values())
        stg.append("c4")
        # C2 on A's fixed-mode delta_full with A's OWN index rule (Codex r2 #8), in both workers, BEFORE D1 (09, addendum 8c §5)
        method, n_c2, _a_rule = c2_identify()
        out["c2_method"], out["c2_n_nodes"] = method, n_c2
        stg.append("c2")
        # 4. draw-0 inputs (the reducer, unchanged), the universe, per-label arrays, the engine-vs-reducer draw-0 identity
        L, run_cfg = load_labels(R, set(labels))
        levels = [str(x) for x in run_cfg["levels"]]
        if not set(levels) <= set(LEVEL_LIST):  # identifiers carry levels by membership in the pinned list (addendum 8c §6)
            stop("a record level is not in the pinned level list")
        if not all(any(float(lv) == float(p["level"]) for lv in levels) for p in parsed.values()):
            stop("a node level is not among the record's levels")
        outside = sorted(({x["main"] for lab in labels for x in L[lab]["pool"].values()}
                          | {L[lab]["main"][c] for lab in labels for _s, c in L[lab]["used"]}) - set(main_idx))
        out["mains_outside_universe"] = len(outside)
        if outside:
            stop("PLUMBING: pool / cohort mains outside meta.merge.mains")  # addendum 4 §8 (09's wording)
        LB = {lab: build_label(R, L[lab], main_idx) for lab in labels}
        ones = np.ones((1, len(mains)), dtype=np.int32)
        e0, bad0 = {}, []
        for lab in labels:
            b, e0[lab] = draw0_consistency(LB[lab], ones, levels, L[lab]["d0"])
            bad0 += [f"{lab}:{x}" for x in b]
        out["draw0_failures"], out["draw0_ok"] = bad0, not bad0
        if bad0:
            stop("engine draw 0 differs from compute_label")
        stg.append("draw0")
        # 5. C0 BEFORE the expensive draws (Codex r1 #5; 09): one scale or NOT LIKE-FOR-LIKE
        per0 = {lab: {"joint": e0[lab], "fixed": e0[lab]} for lab in labels}
        scale_name = c0_identify(c0_points(H, parsed, per0, levels))
        out["c0_scale"], out["c0_ok"] = scale_name, scale_name is not None
        if scale_name is None:
            out["n_nodes"], out["n_not_like_for_like"], out["failing_nodes"] = len(nodes), len(nodes), sorted(set(nodes))
            out["verdict"] = "NOT LIKE-FOR-LIKE (C0 scale unidentified)"
            raise Unresolved()
        scale = 1.0 if scale_name == "fraction" else 100.0
        stg.append("c0")
        _G["LB"] = LB
        # 6. D1: #1..#12 on draws 1..20; first match wins; for B, D8's readings in order and B's fixed tau (Codex r1 #7)
        out["d1_s1_equals_s2"] = strata_order(mains, env_of, "S1") == strata_order(mains, env_of, "S2")  # pre-statement §1
        readings = ["joint", "joint_s507"] if rec == "B" else ["joint"]
        mats, ident, cand_fail = {}, None, []
        for num, (i_rule, s_rule, p_rule) in enumerate(D1_LIST, 1):
            if (s_rule, p_rule) not in mats:
                mats[(s_rule, p_rule)] = multiplicities(mains, env_of, s_rule, p_rule, SEED, N_DRAWS)
            for rd in readings:
                lm20 = [(lab, rd) for lab in labels] + ([(lab, "fixed") for lab in labels] if "fixed" in modes else [])
                per20 = run_draws(lm20, mats[(s_rule, p_rule)], levels, ID_DRAWS)
                miss = None
                for lab in labels:
                    r_ = calib_check(Path("/p", cal2e[f"{lab} | {COMBO}"]["path"]), per20[lab][rd], range(1, ID_DRAWS + 1), levels, i_rule,
                                     fixed_ev=per20[lab].get("fixed"), label=lab)
                    if r_[1]:
                        miss = f"#{num} {rd} {lab} {r_[2]}"
                        break
                if miss is None:
                    ident = (num, i_rule, s_rule, p_rule, rd)
                    break
                cand_fail.append(miss)
            if ident:
                break
        out["d1_candidate_failures"] = cand_fail
        if ident is None:
            out["verdict"] = "UNRESOLVED (stream gate)"
            raise Unresolved()
        out["d1_identified"], out["d1_reading"] = ident[0], ident[4]
        M, rd = mats[(ident[2], ident[3])], ident[4]
        stg.append("d1")
        # 7. the pilot, checked DURING the run for time (09 (b)) and, at its end, memory (09, R1 #19); then all draws
        lm = [(lab, rd) for lab in labels] + ([(lab, "fixed") for lab in labels] if "fixed" in modes else [])
        src, cg_path = mem_source()
        base_b, rss_base, res_b = _read_int(cg_path) if cg_path else 0, rss_now_kib(), 0
        tp, done = time.time(), 0
        with MemSampler(cg_path or "/nonexistent") as ms:
            for lo in range(1, PILOT_DRAWS + 1, CHUNK):
                hi = min(lo + CHUNK, PILOT_DRAWS + 1)
                res_b += result_nbytes(run_draws(lm, M, levels, hi - 1, first=lo))
                done += hi - lo
                proj = (time.time() - tp) * N_DRAWS / done
                out["pilot_projected_s"], out["pilot_elapsed_s"] = int(round(proj)), int(round(time.time() - t0))
                if (time.time() - t0) + proj > PILOT_FRACTION * TIMEOUT:
                    stop("pilot projection over 0.8 x timeout")
        self_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        child_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        mem = mem_projection_mib(src, base_b or 0, ms.peak, self_kib, child_kib, rss_base, min(WORKERS, len(lm)), res_b)
        out.update(pilot_rss_peak_mib=int(self_kib // 1024), pilot_rss_child_mib=int(child_kib // 1024),
                   pilot_mem_projected_mib=int(round(mem)), pilot_mem_source=src)
        if mem_over(mem):
            stop("pilot memory projection over 0.8 x 64 GiB")
        stg.append("pilot")
        per = run_draws(lm, M, levels, N_DRAWS)
        for lab in labels:
            per[lab]["joint"] = per[lab][rd]
        values, fbs, lvi = {}, {}, {}
        for n, p in parsed.items():  # node series, flip bounds and NaN-reason counts right after the draws (addendum 8 §3.4)
            lvi[n] = [i for i, x in enumerate(levels) if float(x) == float(p["level"])][0]
            values[n], fbs[n] = node_series(p, per, lvi[n]), node_flip(p, per, lvi[n])
            out["nan_by_node"][n] = node_nan_counts(p, per, lvi[n], values[n])
        stg.append("draws")
        # 8. C1: the calibration npz at all draws (strict per-quantity layouts), then EVERY in-scope gap member (no intersection)
        c1 = {lab: calib_check(Path("/p", cal2e[f"{lab} | {COMBO}"]["path"]), per[lab]["joint"], range(1, N_DRAWS + 1), levels, ident[1],
                               fixed_ev=per[lab].get("fixed"), strict=True, label=lab) for lab in labels}
        stored = {}
        for lab in labels:
            for mode in modes:
                for (lvs, nm), arr in gap_arrays(Path("/p", gap_pin[0]), lab, mode).items():
                    stored[(lab, mode, float(lvs), nm)] = arr
        gap_cmp, gap_mis, gap_flip, gap_first, gap_flip_by_node = gap_compare(stored, parsed, values, fbs, scale, ident[1])
        del stored
        c1_ok = all(v[1] == 0 for v in c1.values()) and gap_mis == 0
        out.update(c1_ok=c1_ok, c1_n_labels=len(c1), c1_n_labels_ok=sum(v[1] == 0 for v in c1.values()), c1_gap_compared=gap_cmp,
                   c1_gap_mismatch=gap_mis, c1_gap_flip=gap_flip, c1_ok_mask_diffs=sum(v[3] for v in c1.values()),
                   c1_first_failures=sorted(f"{lab} {v[2]}" for lab, v in c1.items() if v[1]) + gap_first)
        stg.append("c1")
        mine, mcse = {}, {}
        for n in parsed:
            lo_, hi_, dr, fin = interval(values[n], method)
            fb = fbs[n]
            mine[n] = {"lo": lo_, "hi": hi_, "draws": int(dr), "finite": int(fin), "point": float(values[n][0]), "c0": True,
                       "flip_bound": float(np.max(fb[1:])) if len(fb) > 1 else 0.0, "flip_bound_point": float(fb[0])}
            mcse[n] = mc_se(values[n], method)
        # the comparison (C4 ran before any verdict, stage "c4")
        rows = compare_core(H, nodes, mine, scale, c1_ok, mcse)
        vc = [r["verdict"] for r in rows.values()]
        out.update(n_nodes=len(rows), n_reproduced=vc.count("REPRODUCED"),
                   n_reproduced_flip=sum(1 for r in rows.values() if r["label"] == "EXACT (flip-attributable)"),
                   n_not_reproduced=vc.count("NOT REPRODUCED"), n_not_like_for_like=vc.count("NOT LIKE-FOR-LIKE"),
                   n_unresolved=vc.count("UNRESOLVED (stream gate)"),
                   n_incomplete_draws=sum(1 for r in rows.values() if not (r["mine_complete"] and r["record_meta_ok"])),
                   mc_n_agree=(None if c1_ok else sum(bool(r.get("mc_diagnostic_agrees")) for r in rows.values())),
                   failing_nodes=sorted(n for n, r in rows.items() if r["verdict"] in ("NOT REPRODUCED", "NOT LIKE-FOR-LIKE")))
        stg.append("compare")
        # the values, saved BEFORE D5 (09, R1 #18)
        vb = io.BytesIO()
        np.savez(vb, **{f"node{i:03d}": values[n] for i, n in enumerate(nodes)})
        dest = Path("/o", f"opt3_{rec}_{_stamp()}")
        dest.mkdir(parents=True, exist_ok=False)
        (dest / "node_series.npz").write_bytes(vb.getvalue())
        (dest / "compare.json").write_text(json.dumps({"rows": rows, "mine": mine, "nodes": nodes, "method": method, "scale": scale_name,
                                                      "d1": list(ident), "c1": {k: list(v) for k, v in c1.items()},
                                                      "nan_by_node": out["nan_by_node"]}, default=float))
        OUT_VOL.commit()
        out.update(values_dir=str(dest), values_sha256_node_series=sha(vb.getvalue()), values_sha256_compare=file_sha(dest / "compare.json"))
        stg.append("values")
        d5 = None
        if rec == "A":
            try:
                flip_nodes = {n for n, r in rows.items() if r["ends_flip_attributable"]} | {n for n, c in gap_flip_by_node.items() if c > 0}
                d5 = d5_halfwidth(mine, by_slot, slots, flip_nodes)  # 09 (addendum 8c §4): flip-sensitive elements
            except Exception:  # recorded as D5's status 'error' -> UNRESOLVED (D5 error); the values are already saved
                d5 = {"status": "error", "pair": None, "not_reproduced": [], "unresolved": [], "flip_sensitive": [], "per_slot": {}}
            out.update(d5_status=d5["status"], d5_pair=None if d5["pair"] is None else f"{d5['pair'][0]}:{d5['pair'][1]}",
                       d5_ok=d5["status"] == "fit", d5_not_reproduced=d5["not_reproduced"], d5_unresolved=d5["unresolved"],
                       n_d5_flip_sensitive=len(d5["flip_sensitive"]))
            (dest / "d5.json").write_text(json.dumps(d5))
            OUT_VOL.commit()
            stg.append("d5")
        out["verdict"] = verdict_line(rows, c1_ok, d5)
    except Unresolved:
        pass
    except Stop as e:  # a fixed code, the running stage, the file and line (addenda 8d, 8f), the inputs gate's file; no message text
        out["stop"], out["verdict"] = stop_code_at(e.code, stg, rec), "STOP"
        out["stop_file"], out["stop_line"] = _stop_where(e)
        out["input_file"] = e.input_file if e.input_file in INPUT_FILES else None
        out["stop_ids"] = [x for x in e.ids if _id_ok("stop_id", x, nodes, allow)]
    except Exception as e:  # nothing but a fixed code leaves on failure
        out["stop"], out["verdict"] = stop_code_at(exc_code("", e), stg, rec), "STOP"
        out["stop_file"], out["stop_line"] = _stop_where(e)
    if "draws" not in stg:  # the shared nan_by_node rule: filled only once the draws stage completed
        out["nan_by_node"] = {}
    out["blas_threads_max"] = max((mx for _n, mx, _nb in _BLAS_SEEN), default=None)
    out["seconds"] = int(round(time.time() - t0))
    if validate_report(out, nodes, allow):  # the worker never returns an off-schema report (addendum 8 §6)
        out = blank_report(rec, "STOP: report schema")
        out["seconds"] = int(round(time.time() - t0))
    return out


def _keystr(parts):
    """A figure source's `key` array joined as the frozen node lists were (jq: map(tostring) | join("|")); numbers as JSON prints them."""
    return "|".join(x if isinstance(x, str) else json.dumps(x) for x in parts)


PIN_CODES = {SCOPE[0]: "pin scope", FIGS["fig1_main"][0]: "pin fig1_main", FIGS["fig1_main_R"][0]: "pin fig1_main_R", FIGS["fig4"][0]: "pin fig4"}


def rebuild_nodes(rec):
    """The in-scope nodes rebuilt from the pinned sources: A = the scope file's a_nodes (minus |ci95[i]) united with Figs 1/6's
    record-input interval keys; B = Fig 4's interval keys. Every source is sha-checked first."""
    for name, want in [SCOPE, FIGS["fig1_main"], FIGS["fig1_main_R"], FIGS["fig4"]]:
        if file_sha(Path("/opt3in", name)) != want:
            stop(PIN_CODES[name])
    if rec == "B":
        src = json.loads(Path("/opt3in", FIGS["fig4"][0]).read_bytes())
        return sorted({_keystr(v["key"]) for v in src["values"] if "ci95" in v})
    out = set()
    for ln in Path("/opt3in", SCOPE[0]).read_text().splitlines():
        if ln.strip():
            a = json.loads(ln)["a_node"]
            if not re.search(r"\|ci95\[[01]\]$", a):
                stop("scope a_node form")
            out.add(re.sub(r"\|ci95\[[01]\]$", "", a))
    for f in ("fig1_main", "fig1_main_R"):
        src = json.loads(Path("/opt3in", FIGS[f][0]).read_bytes())
        out |= {_keystr(v["key"]) for v in src["values"] if "ci95" in v and v["input"] == "record"}
    return sorted(out)


def d5_halfwidth(mine, by_slot, slots, flip_nodes=frozenset()):
    """D5 (addenda 2 §2, 3 §2, 8 §3.1, 8c §4): per halfwidth slot (by_slot from d5_ready, checked before the draws), my two readings per
    node aggregated by the slot's name, the per-scale token rule per element, then d5_classify with the flip-sensitive elements (a slot
    with a node in flip_nodes: one that passed C1 / the end check only via the flip term). Slot strings stay in the container."""
    import math

    per = {}
    for slot, nds in sorted(by_slot.items()):
        how = D5_AGG[slot.rsplit(".", 1)[-1]]
        if any(not (math.isfinite(mine[n]["lo"]) and math.isfinite(mine[n]["hi"])) for n in nds):
            per[slot] = {"half_range": "unresolved", "max_side": "unresolved"}
            continue
        hr = how((mine[n]["hi"] - mine[n]["lo"]) / 2 for n in nds)
        ms = how(max(mine[n]["hi"] - mine[n]["point"], mine[n]["point"] - mine[n]["lo"]) for n in nds)
        per[slot] = {"half_range": d5_fit(slots[slot], hr), "max_side": d5_fit(slots[slot], ms)}
    flip_el = {slot for slot, nds in by_slot.items() if set(nds) & set(flip_nodes)}
    status, pair, nr, unres, fl = d5_classify(per, flip_el)
    return {"status": status, "pair": pair, "not_reproduced": nr, "unresolved": unres, "flip_sensitive": fl, "per_slot": per}


D5_VERDICT = {"ambiguous": "UNRESOLVED (D5 ambiguous)", "mixed": "UNRESOLVED (D5 mixed readings)",
              "unresolved": "UNRESOLVED (D5 unresolved elements)", "error": "UNRESOLVED (D5 error)",
              "flip_sensitive": "UNRESOLVED (D5 flip-sensitive)"}


def verdict_line(rows, exact, d5=None):
    """The overall verdict, from the fixed list (addendum 8 §5): C1 failed -> UNRESOLVED (stream gate); a NOT LIKE-FOR-LIKE node ->
    NOT LIKE-FOR-LIKE; a NOT REPRODUCED node or D5 not_reproduced -> NOT REPRODUCED; D5 ambiguous / mixed / unresolved / error ->
    UNRESOLVED (D5 ...); else EXACT. B has no D5 (d5 None)."""
    if not exact:
        return "UNRESOLVED (stream gate)"
    vs = [r["verdict"] for r in rows.values()]
    st = None if d5 is None else d5["status"]
    if "NOT LIKE-FOR-LIKE" in vs:
        return "NOT LIKE-FOR-LIKE"
    if "NOT REPRODUCED" in vs or st == "not_reproduced":
        return "NOT REPRODUCED"
    if st in D5_VERDICT:
        return D5_VERDICT[st]
    return "INTERVALS REPRODUCED (EXACT)"


# ---------------------------------------------------------------- Modal functions: two workers and one driver (§6; addendum 4 §11; 09 #3)
DRIVER_DEADLINE_S, WORKER_DEADLINE_S = 300, 600  # entry deadlines after launch: driver startup; driver + worker startup


def launch_id(launch_epoch):
    return str(int(round(launch_epoch * 1000)))


def _worker(rec, launch_epoch):
    if time.time() > launch_epoch + WORKER_DEADLINE_S:  # a platform retry after the window exits at once (09 #5 ii)
        return blank_report(rec, "STOP: past the launch deadline")
    return run_record(rec)


@app.function(image=images["A"], cpu=(16, 16), memory=(65536, 65536), timeout=TIMEOUT, startup_timeout=300, scaledown_window=60,
              retries=0, volumes=VOLS)
def worker_A(launch_epoch: float) -> dict:
    return _worker("A", launch_epoch)


@app.function(image=images["B"], cpu=(16, 16), memory=(65536, 65536), timeout=TIMEOUT, startup_timeout=300, scaledown_window=60,
              retries=0, volumes=VOLS)
def worker_B(launch_epoch: float) -> dict:
    return _worker("B", launch_epoch)


def stop_summary(launch_epoch, lid, code):
    return {"launch_epoch": launch_epoch, "launch_id": lid, "fan_out": 0, "driver_stop": code, "workers": []}


def _write_summary(summ):
    """Codex r2 #4: never raises. Name opt3_summary_<launch_id>_<YYYYMMDD_HHMMSS>_<8 hex>.json, exclusive create retried with a
    fresh nonce on a clash (5 tries), the commit retried (3 tries). Returns the name, or None if no write succeeded."""
    import secrets

    name = None
    try:
        body = json.dumps(summ)
    except Exception:
        body = json.dumps(stop_summary(0.0, "0000000000", "STOP: summary schema"))
    for _ in range(5):
        try:
            cand = f"opt3_summary_{summ.get('launch_id', '0000000000')}_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}.json"
            with open(Path("/o", cand), "x") as fh:
                fh.write(body)
            name = cand
            break
        except FileExistsError:
            continue
        except Exception:
            break
    for _ in range(3 if name else 0):
        try:
            OUT_VOL.commit()
            break
        except Exception:
            time.sleep(2)
    return name


@app.function(image=images["driver"], cpu=(1, 1), memory=(2048, 2048), timeout=4200, startup_timeout=300, scaledown_window=60,
              retries=0, volumes={"/o": OUT_VOL})
def driver(launch_epoch: float) -> dict:
    """ONE spawn from the laptop (`modal run --detach`). Idempotent (09 #3): exclusive-creates /o/opt3_spawned_<launch_id> and commits
    it before spawning; if it exists, no spawn. Exactly two workers. The summary is validated against the exact schema in the
    container (with both node lists) before the write, a STOP summary is written on EVERY other exit, and the write never raises."""
    try:
        lid = launch_id(launch_epoch)
    except Exception:
        lid = "0000000000"
    try:
        if time.time() > launch_epoch + DRIVER_DEADLINE_S:
            summ = stop_summary(launch_epoch, lid, "STOP: past the launch deadline")
        else:
            nodes_ab, allow = {"A": read_nodes("A"), "B": read_nodes("B")}, load_allow()
            OUT_VOL.reload()
            try:
                with open(Path("/o", f"opt3_spawned_{lid}"), "x"):
                    pass
            except FileExistsError:
                summ = stop_summary(launch_epoch, lid, "STOP: spawn marker exists (idempotency)")
            else:
                OUT_VOL.commit()
                calls = {"A": worker_A.spawn(launch_epoch), "B": worker_B.spawn(launch_epoch)}
                reps = []
                for rec in ("A", "B"):
                    try:
                        r = calls[rec].get()
                    except Exception as e:
                        r = blank_report(rec, exc_code("worker ", e))
                    if validate_report(r, nodes_ab[rec], allow):
                        r = blank_report(rec, "STOP: worker report schema")
                    reps.append(r)
                summ = {"launch_epoch": launch_epoch, "launch_id": lid, "fan_out": 2, "driver_stop": None, "workers": reps}
                if validate_summary(summ, nodes_ab, allow):
                    summ = stop_summary(launch_epoch, lid, "STOP: summary schema")
    except Stop as e:
        summ = stop_summary(launch_epoch, lid, e.code)
    except Exception as e:
        summ = stop_summary(launch_epoch, lid, exc_code("driver ", e))
    return {"summary": _write_summary(summ)}


@app.local_entrypoint()
def main():
    t = time.time()
    call = driver.spawn(t)  # exactly one remote call; launch with `modal run --detach`
    print(json.dumps({"driver_call_id": call.object_id, "launch_id": launch_id(t)}))
