"""recheck (b) v3: an independent in-container re-comparison of option (3) record A (build agent b, 2026-10-21 PDT).

Pre-statement: PRESTATEMENT_b_v3.md (a delta note on PRESTATEMENT_b_v2.md; both shas pinned below; frozen before this file was
written). Supersedes recheck_b_v2.py / recheck_b.py (not run). One CPU Modal function, one .remote() call, reads only (both Volumes
mounted read-only; nothing is written to any Volume).
v2 (Codex r1): the remote function's whole body runs inside a sanitizing boundary (_guarded): ANY exception becomes a code from closed
lists, returned as data; exception messages and tracebacks never cross back.
v3 (Codex on v2): the local entrypoint's whole body (preparation and file reads, the remote call, result processing and printing) runs
inside its twin (_local_guarded): ANY exception becomes one line with code=fail:<stage>:<Class>; no message, no re-raise, no chain.
In both boundaries the code is built after the except block (no __context__) in its own try, with a constant fallback
(stop:sanitizer:Other / fail:sanitizer:Other). main() exits with sys.exit(rc) outside every except block. On success the output
holds booleans and ints only and is written create-only to results/recheck_b_v3_out_<STAMP>.json.
Written fresh: nothing is imported or copied from a1's code (cleanroom_boot_opt3.py was read for file layouts only; PRESTATEMENT_b_v2 §7).
Launched only by RUN_ME_b_launch_v3.sh:
  "$M" run recheck_b_v3.py --stamp <STAMP> --run-dir <this folder>/runs/<STAMP> --trailer '<Claude-Session line>'
"""

import contextlib
import hashlib
import io
import json
import math
import re
import sys
import time
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-opt3recheck_b_kill-2026-11-20"
A1_VOL = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"  # a1's OUT_VOL -> /o (read-only)
PD_VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"  # the producer per-draw npz Volume -> /p (read-only)
TIMEOUT, STARTUP, SCALEDOWN = 600, 180, 2
CPU, MEM_MIB = 2, 4096
PRESTATEMENT_SHA256 = "e299e560b7a01b70482bf30201cdb850d0da79ba95090bc939f6768b5a8148b4"  # PRESTATEMENT_b_v3.md (delta note)
PRESTATEMENT_BASE_SHA256 = "36e12f41e39dc7ffbbf3a3ed6fdaa09a13932880ba017b38419bbdcd0851d710"  # PRESTATEMENT_b_v2.md (its base)
NUMPY_PIN, THREADPOOLCTL_PIN = "2.3.3", "3.6.0"
VALUES_DIR = "/o/opt3_A_20261021_202527_acbf3404"  # the summary's values_dir for record A (7d787550b)
NS_SHA = "9493e2b24caa34b42b8a04e5501ee1dc8213697a3fcaa10a1fdbbe9d24c88f4d"  # values_sha256_node_series
CMP_SHA = "6d81961657f7075f50c008b0921aaed6403501b70d2b951768684e9c6556a208"  # values_sha256_compare
PD_PATH = "/p/xfer/fpr507prep/alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327_perdraw.npz"
PD_SHA = "5e6cd10e382d8b03d409982c67efd72512176d7e95d2f3c94752d2b90e5650d1"  # opt3_perdraw_npz_AB.json A_npz.sha256
REC_FILE = "routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz"
REC_GZ_SHA = "191c467ab095b0f9ddfe2271822655bc5b49ee02a1a2c5a5b0e11bc855a00865"
REC_JSON_SHA = "54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a"
NODES_FILE, NODES_SHA = "opt3_nodes_A.txt", "8f6f42788ceb2d48a375e8dec21106ee8859debca7f75e41d5a15c1ecad5702b"
TPL_FILE, TPL_SHA = "opt3_record_path_templates.tsv", "58785b3064ac5a5d21666bc9f9b7365e5f85d02a8812734296196270c3b5649c"
D, MIN_FINITE, TOL, SCALE = 10000, 9500, 1e-9, 1.0
N_NODES, N_GAP, N_SAMPLE, SAMPLE_SEED, NC_DELTA = 60, 28, 2000, 20261021, 1e-6
COMBO = "pooled_all_origins"
LEAF_KEYS = {"ci95", "degenerate", "draws", "finite_draws", "point"}
BLAS_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                             "NUMEXPR_NUM_THREADS")}
TRAILER_RE = re.compile(r"Claude-Session: https://claude\.ai/code/session_[A-Za-z0-9]+")
STAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{6}")
# The closed lists every failure code is built from (PRESTATEMENT_b_v2 §3a; v3 adds the local stages, sanitizer, FileExistsError and
# the local reasons). Nothing outside them is ever returned or printed.
STAGES = ("entry", "selftest", "pins", "shas", "nodes", "record", "a1_files", "ends", "sample", "controls", "output",
          "prep", "remote", "result", "print", "sanitizer", "unknown")
STOP_REASONS = ("deadline", "meta_malformed", "pins", "threadpools", "sha_record_gz", "sha_record_json", "sha_nodes", "sha_templates",
                "a1_files_missing", "sha_node_series", "sha_compare", "perdraw_missing", "sha_perdraw", "nodes_newline", "nodes_count",
                "template_row", "node_parse", "templates_absent", "record_path", "record_leaf_keys", "record_leaf_layout",
                "reader_control", "compare_keys", "compare_nodes", "compare_d1", "node_series_keys", "node_series_shape", "gap_count",
                "gap_member_missing", "gap_lengths", "nc_magnitude", "output_schema", "selftest_sanitizer",
                "local_selftest", "local_stamp_rundir", "local_trailer", "local_results_dir", "local_output_exists", "local_input_sha",
                "return_sha", "return_layout", "unlisted")
EXC_NAMES = ("KeyError", "ValueError", "TypeError", "IndexError", "AttributeError", "LookupError", "OSError", "FileNotFoundError",
             "FileExistsError", "PermissionError", "IsADirectoryError", "MemoryError", "BadZipFile", "BadGzipFile", "JSONDecodeError",
             "UnicodeDecodeError", "EOFError", "ZeroDivisionError", "OverflowError", "FloatingPointError", "RuntimeError",
             "RecursionError", "AssertionError", "NotImplementedError", "TimeoutError", "ConnectionError", "KeyboardInterrupt",
             "SystemExit", "CancelledError", "FunctionTimeoutError", "RemoteError", "ExecutionError", "InvalidError", "NotFoundError",
             "InputCancellation", "ClientClosed", "ReturnShaMismatch", "OutputWriteFailed", "Other")
CODE_RE = re.compile("(stop|fail):(" + "|".join(STAGES) + "):(" + "|".join(EXC_NAMES) + "|Stop:(" + "|".join(STOP_REASONS) + "))")
REMOTE_FALLBACK_CODE = "stop:sanitizer:Other"  # v3: the constant used when building a remote code itself fails
FALLBACK_LINE = "RECHECK_B_STATUS fail app_id=unknown code=fail:sanitizer:Other"  # v3: the local constant
OUT_KEYS = {  # the exact output schema (PRESTATEMENT_b_v2 §4): booleans and ints only
    "sha_ok_record_gz": bool,
    "sha_ok_record_json": bool,
    "sha_ok_nodes_file": bool,
    "sha_ok_templates": bool,
    "sha_ok_node_series": bool,
    "sha_ok_compare": bool,
    "sha_ok_perdraw_npz": bool,
    "sanitizer_selftest_ok": bool,
    "pins_ok": bool,
    "blas_env_all_one": bool,
    "threadpools_n": int,
    "threadpools_all_one": bool,
    "reader_control_ok": bool,
    "templates_ok": bool,
    "nodes_same_order_as_frozen": bool,
    "method_is_linear": bool,
    "scale_is_fraction": bool,
    "d1_num_is_1": bool,
    "i_rule_is_I1": bool,
    "n_nodes_checked": int,
    "n_ends_ok": int,
    "n_counts_ok": int,
    "n_a1_compare_ok": int,
    "n_withheld_mine": int,
    "n_gap_nodes": int,
    "n_gap_members_found": int,
    "n_gap_lengths_ok": int,
    "n_sample": int,
    "n_sample_ok": int,
    "n_sample_both_nonfinite": int,
    "negative_control_ends_baseline_ok": bool,
    "negative_control_ends_flagged": bool,
    "negative_control_draw_baseline_ok": bool,
    "negative_control_draw_flagged": bool,
    "negative_control_flagged": bool,
    "elapsed_s": int,
}

app = modal.App(APP)
a1_vol = modal.Volume.from_name(A1_VOL, version=2)
pd_vol = modal.Volume.from_name(PD_VOL, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "threadpoolctl==3.6.0")
         .env(BLAS_ENV))  # also set in-function before numpy loads (the runtime may override image ENV)


class Stop(Exception):  # my own STOPs: the only argument is a reason from STOP_REASONS (never a value)
    pass


def _stop(reason):
    raise Stop(reason)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _isint(x):
    return type(x) is int


def _isnum(x):
    return type(x) in (int, float) and math.isfinite(x)


def _close(a, ref):  # |a - ref| <= t = 1e-9 * max(1, |ref|), both finite (PO:261)
    return math.isfinite(a) and math.isfinite(ref) and abs(a - ref) <= TOL * max(1.0, abs(ref))


def _build_code(prefix, stage, etype, reason):
    """<prefix>:<stage>:<ExceptionClass> or <prefix>:<stage>:Stop:<reason>, every part from a closed list. Called OUTSIDE any except
    block (v3), so an exception here has no __context__; callers wrap it in their own try with a constant fallback."""
    st = stage if stage in STAGES else "unknown"
    if etype is Stop:
        code = f"{prefix}:{st}:Stop:{reason if reason in STOP_REASONS else 'unlisted'}"
    else:
        n = getattr(etype, "__name__", None)
        code = f"{prefix}:{st}:{n if n in EXC_NAMES else 'Other'}"
    return code if CODE_RE.fullmatch(code) else f"{prefix}:sanitizer:Other"


def _guarded(body, *args):
    """THE sanitizing boundary (Codex r1 #1; v3 fallback): run body(st, *args); ANY exception becomes {"status": "stop", "code": <code>}.
    Inside the except block only the stage, the class and (for Stop) the reason are captured, in a nested try; the exception object,
    its message and its traceback are dropped there: never returned, printed or logged. The code is built after the block ends."""
    st = ["entry"]
    failed, stage, etype, reason = False, "unknown", None, None
    try:
        ok = {"status": "ok", **body(st, *args)}
    except BaseException as e:  # noqa: BLE001  (the sanitizing boundary)
        failed = True
        try:
            stage, etype = st[0], type(e)
            if etype is Stop and len(e.args) == 1:
                reason = e.args[0]
        except BaseException:  # noqa: BLE001
            pass
    if not failed:
        return ok
    try:
        return {"status": "stop", "code": _build_code("stop", stage, etype, reason)}
    except BaseException:  # noqa: BLE001  (sanitizer failure: a constant, built from nothing the exception carried)
        return {"status": "stop", "code": REMOTE_FALLBACK_CODE}


def _fixture_value_error(st, secret):  # the sanitizer fixture: an exception whose message carries a value
    st[0] = "selftest"
    raise ValueError(f"leak {secret!r}")


def _fixture_stop(st):
    st[0] = "selftest"
    _stop("deadline")


def _body(st, rec_gz, nodes_b, tpl_b, meta_in):
    import os

    os.environ.update(BLAS_ENV)  # before numpy (or anything that loads BLAS) is imported

    import gzip
    import io
    import zipfile

    t0 = time.time()
    deadline = float(meta_in["launch_epoch"]) + TIMEOUT + STARTUP

    def enter(stage):
        st[0] = stage
        if time.time() > deadline:
            _stop("deadline")

    enter("entry")
    stamp = str(meta_in["stamp"])
    if (not STAMP_RE.fullmatch(stamp) or meta_in["name"] != f"recheck_b_v3_out_{stamp}.json"
            or not TRAILER_RE.fullmatch(str(meta_in["trailer"])) or not re.fullmatch(r"[0-9a-f]{64}", str(meta_in["script_sha256"]))):
        _stop("meta_malformed")

    # [selftest] The sanitizer fixture, through the same _guarded boundary (PRESTATEMENT_b_v2 §3a).
    enter("selftest")
    r1 = _guarded(_fixture_value_error, 0.123456789)
    r2 = _guarded(_fixture_stop)
    if (r1 != {"status": "stop", "code": "stop:selftest:ValueError"} or r2 != {"status": "stop", "code": "stop:selftest:Stop:deadline"}
            or "0.123" in json.dumps(r1) or not CODE_RE.fullmatch(r1["code"]) or not CODE_RE.fullmatch(r2["code"])):
        _stop("selftest_sanitizer")
    out = {"sanitizer_selftest_ok": True}

    enter("pins")
    import numpy as np
    import threadpoolctl
    from threadpoolctl import threadpool_info, threadpool_limits

    if np.__version__ != NUMPY_PIN or threadpoolctl.__version__ != THREADPOOLCTL_PIN:
        _stop("pins")
    _tp = threadpool_limits(limits=1)  # noqa: F841  (held for the whole call)
    pools = threadpool_info()
    if not pools or any(p_.get("num_threads") != 1 for p_ in pools):
        _stop("threadpools")
    out.update(pins_ok=True, blas_env_all_one=all(os.environ.get(k) == "1" for k in BLAS_ENV), threadpools_n=len(pools),
               threadpools_all_one=True)

    # 1. [shas] Shas, before any use (STOP on any mismatch).
    enter("shas")
    if _sha(rec_gz) != REC_GZ_SHA:
        _stop("sha_record_gz")
    rec_js = gzip.decompress(rec_gz)
    if _sha(rec_js) != REC_JSON_SHA:
        _stop("sha_record_json")
    if _sha(nodes_b) != NODES_SHA:
        _stop("sha_nodes")
    if _sha(tpl_b) != TPL_SHA:
        _stop("sha_templates")
    ns_p, cmp_p = Path(VALUES_DIR, "node_series.npz"), Path(VALUES_DIR, "compare.json")
    if not ns_p.is_file() or not cmp_p.is_file():
        _stop("a1_files_missing")
    ns_bytes, cmp_bytes = ns_p.read_bytes(), cmp_p.read_bytes()
    if _sha(ns_bytes) != NS_SHA:
        _stop("sha_node_series")
    if _sha(cmp_bytes) != CMP_SHA:
        _stop("sha_compare")
    if not Path(PD_PATH).is_file():
        _stop("perdraw_missing")
    h = hashlib.sha256()
    with open(PD_PATH, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 24), b""):
            enter("shas")
            h.update(ch)
    if h.hexdigest() != PD_SHA:
        _stop("sha_perdraw")
    out.update(sha_ok_record_gz=True, sha_ok_record_json=True, sha_ok_nodes_file=True, sha_ok_templates=True, sha_ok_node_series=True,
               sha_ok_compare=True, sha_ok_perdraw_npz=True)

    # 2. [nodes] Nodes -> record paths (my parser; template-checked); [record] the record and the reader control.
    enter("nodes")
    lines = nodes_b.decode("utf-8").split("\n")
    if lines[-1] != "":
        _stop("nodes_newline")
    nodes = lines[:-1]
    if len(nodes) != N_NODES or len(set(nodes)) != N_NODES:
        _stop("nodes_count")
    tpl = tpl_b.decode("utf-8").split("\n")
    tpl_set = set(tpl)

    def row(prefix):
        r = [x for x in tpl if x.startswith(prefix)]
        if len(r) != 1:
            _stop("template_row")
        return r[0]

    lab_row = row(f"A\t<label> keys under pooled having {COMBO}: ")
    mode_row = row(f"A\t<mode> keys under pooled.<label>.{COMBO} (union): ")
    plvl_row = row(f"A\t<lvl> keys under pooled.<label>.{COMBO}.joint (union): ")
    clvl_row = row(f"A\t<lvl> keys under coheadline_signed.{COMBO}.joint: ")

    def parse(n):
        if n.startswith("pooled|") and f"|{COMBO}|" in n:
            label, rest = n[len("pooled|"):].split(f"|{COMBO}|", 1)
            parts = rest.split("|")
            if not label or len(parts) != 4 or parts[2] not in ("gap", "catch"):
                _stop("node_parse")
            mode, level, fam, leaf = parts
            ok = (f'A\tpooled.<label>.{COMBO}.<mode>.<lvl>."{fam}"."{leaf}"' in tpl_set and f'"{label}"' in lab_row
                  and f'"{mode}"' in mode_row and f'"{level}"' in plvl_row)
            return {"family": fam, "label": label, "mode": mode, "level": level, "leaf": leaf, "tpl_ok": ok,
                    "path": ["pooled", label, COMBO, mode, level, fam, leaf]}
        parts = n.split("|")
        if len(parts) == 7 and parts[0] == "coheadline_signed" and parts[1] == COMBO:
            mode, level, group, readout, sy = parts[2:]
            key = f"{group}|{readout}"
            ok = f'A\tcoheadline_signed.{COMBO}.<mode>.<lvl>."{key}"."{sy}"' in tpl_set and f'"{level}"' in clvl_row
            return {"family": "coheadline", "label": None, "mode": mode, "level": level, "leaf": sy, "tpl_ok": ok,
                    "path": ["coheadline_signed", COMBO, mode, level, key, sy]}
        _stop("node_parse")

    P = [parse(n) for n in nodes]
    if not all(p_["tpl_ok"] for p_ in P):
        _stop("templates_absent")
    out["templates_ok"] = True
    enter("record")
    H = json.loads(rec_js)
    del rec_js

    def leaf_of(p_):
        d = H
        for k in p_["path"]:
            if not isinstance(d, dict) or k not in d:
                _stop("record_path")
            d = d[k]
        if not isinstance(d, dict) or set(d) != LEAF_KEYS:
            _stop("record_leaf_keys")
        if not (isinstance(d["ci95"], list) and len(d["ci95"]) == 2 and _isint(d["draws"]) and _isint(d["finite_draws"])):
            _stop("record_leaf_layout")
        return d

    if leaf_of(P[0])["draws"] != D:
        _stop("reader_control")
    out["reader_control_ok"] = True

    # 3. [a1_files] a1's files: compare.json fields and order; node_series.npz arrays (draws 0..10,000).
    enter("a1_files")
    C = json.loads(cmp_bytes)  # a1 wrote it with json.dumps defaults (NaN literals allowed)
    if not isinstance(C, dict) or any(k not in C for k in ("nodes", "mine", "method", "scale", "d1")) or not isinstance(C["mine"], dict):
        _stop("compare_keys")
    cn = C["nodes"]
    if not isinstance(cn, list) or len(cn) != N_NODES or set(cn) != set(nodes):
        _stop("compare_nodes")
    idx = {n: i for i, n in enumerate(cn)}
    d1 = C["d1"]
    if not (isinstance(d1, list) and len(d1) == 5 and d1[1] in ("I1", "I2")):
        _stop("compare_d1")
    i_rule = d1[1]
    out.update(nodes_same_order_as_frozen=cn == nodes, method_is_linear=C["method"] == "linear", scale_is_fraction=C["scale"] == "fraction",
               d1_num_is_1=_isint(d1[0]) and d1[0] == 1, i_rule_is_I1=i_rule == "I1")
    with np.load(io.BytesIO(ns_bytes), allow_pickle=False) as Z:
        if set(Z.files) != {f"node{i:03d}" for i in range(N_NODES)}:
            _stop("node_series_keys")
        NS = {n: np.asarray(Z[f"node{idx[n]:03d}"], dtype=float) for n in nodes}
    if any(x.ndim != 1 or x.shape[0] != D + 1 for x in NS.values()):
        _stop("node_series_shape")

    # 4. [ends] Check 1: ends, all 60 nodes (RULE-b on a1's values vs the record; vs a1's own compare.json).
    enter("ends")

    def ends_check(lo, hi, nf, leaf):  # used by the real check AND the negative control
        ci = leaf["ci95"]
        if nf < MIN_FINITE or leaf["finite_draws"] < MIN_FINITE or not all(_isnum(v) for v in ci):
            return False
        return _close(SCALE * lo, float(ci[0])) and _close(SCALE * hi, float(ci[1]))

    def same_a1(mine_v, a1_v):  # within t(a1 value), or both NaN
        if type(a1_v) not in (int, float):
            return False
        if math.isnan(mine_v) and math.isnan(a1_v):
            return True
        return _close(mine_v, float(a1_v))

    mine = {}
    n_ends = n_counts = n_a1 = n_withheld = 0
    for n, p_ in zip(nodes, P):
        y = NS[n][1:]  # draw 0 excluded (TB:79)
        fin = y[np.isfinite(y)]
        nf, nd = int(fin.size), int(y.size)
        if nf:
            q = np.percentile(fin, [2.5, 97.5], method="linear")
            lo, hi = float(q[0]), float(q[1])
        else:
            lo = hi = math.nan
        mine[n] = (lo, hi, nf, nd)
        leaf = leaf_of(p_)
        n_withheld += int(nf < MIN_FINITE)
        n_ends += int(ends_check(lo, hi, nf, leaf))
        n_counts += int(leaf["draws"] == D and nd == D and leaf["finite_draws"] == nf)
        a = C["mine"].get(n)
        n_a1 += int(isinstance(a, dict) and same_a1(lo, a.get("lo")) and same_a1(hi, a.get("hi")) and _isint(a.get("draws"))
                    and a["draws"] == nd and _isint(a.get("finite")) and a["finite"] == nf)
    out.update(n_nodes_checked=len(mine), n_ends_ok=n_ends, n_counts_ok=n_counts, n_a1_compare_ok=n_a1, n_withheld_mine=n_withheld)

    # 5. [sample] Check 2: the seeded per-draw sample over the 28 gap nodes (a1's value vs the producer's stored array).
    enter("sample")
    gap = [(n, p_) for n, p_ in zip(nodes, P) if p_["family"] == "gap"]
    if len(gap) != N_GAP:
        _stop("gap_count")

    def member(p_):  # my builder: <label with "|" and " " -> "_">__pooled_all_origins__<mode>__<level>__pooled__gap_<leaf>.npy
        lab = p_["label"].replace("|", "_").replace(" ", "_")
        return f"{lab}__{COMBO}__{p_['mode']}__{p_['level']}__pooled__gap_{p_['leaf']}.npy"

    want_len = D + 1 if i_rule == "I1" else D  # PO:246 (I1: draws + 1; I2: draws)
    arrs, n_found, n_len = [], 0, 0
    with zipfile.ZipFile(PD_PATH) as zf:
        names = zf.namelist()
        for _, p_ in gap:
            nm = member(p_)
            if names.count(nm) != 1:
                continue
            n_found += 1
            with zf.open(nm) as fh:
                a = np.load(io.BytesIO(fh.read()), allow_pickle=False)
            n_len += int(a.ndim == 1 and a.shape[0] == want_len)
            arrs.append(np.asarray(a, dtype=float))
    if n_found != N_GAP:
        _stop("gap_member_missing")
    if n_len != N_GAP:
        _stop("gap_lengths")
    out.update(n_gap_nodes=len(gap), n_gap_members_found=n_found, n_gap_lengths_ok=n_len)

    def pair_ok(v, s):  # used by the real check AND the negative control
        fv, fs = math.isfinite(v), math.isfinite(s)
        if not fv and not fs:
            return True
        return fv and fs and abs(v - s) <= TOL * max(1.0, abs(s))

    rng = np.random.default_rng(SAMPLE_SEED)
    ks = rng.choice(N_GAP * D, size=N_SAMPLE, replace=False)
    n_ok = n_nan = 0
    nc_pair = None
    for k in ks:
        j, d = int(k) // D, int(k) % D + 1  # draws 1..10,000
        v = SCALE * float(NS[gap[j][0]][d])
        s = float(arrs[j][d] if i_rule == "I1" else arrs[j][d - 1])
        n_ok += int(pair_ok(v, s))
        n_nan += int(not math.isfinite(v) and not math.isfinite(s))
        if nc_pair is None and math.isfinite(v) and math.isfinite(s):
            nc_pair = (v, s)
    out.update(n_sample=len(ks), n_sample_ok=n_ok, n_sample_both_nonfinite=n_nan)

    # 6. [controls] Check 3: negative controls (in memory, after the real counts, through the same functions), with the magnitude
    #    assertion (Codex r1 #3): delta > 2 t(ref) for each selected reference, so a baseline within t cannot hide the perturbation.
    enter("controls")
    lo0, hi0, nf0, _ = mine[nodes[0]]
    leaf0 = leaf_of(P[0])
    ref_e = leaf0["ci95"][0]
    if _isnum(ref_e) and not NC_DELTA > 2 * TOL * max(1.0, abs(float(ref_e))):
        _stop("nc_magnitude")
    nc_e_base, nc_e_flag = ends_check(lo0, hi0, nf0, leaf0), not ends_check(lo0 + NC_DELTA, hi0, nf0, leaf0)
    if nc_pair is None:
        nc_d_base = nc_d_flag = False
    else:
        if not NC_DELTA > 2 * TOL * max(1.0, abs(nc_pair[1])):
            _stop("nc_magnitude")
        nc_d_base, nc_d_flag = pair_ok(*nc_pair), not pair_ok(nc_pair[0] + NC_DELTA, nc_pair[1])
    out.update(negative_control_ends_baseline_ok=nc_e_base, negative_control_ends_flagged=nc_e_flag,
               negative_control_draw_baseline_ok=nc_d_base, negative_control_draw_flagged=nc_d_flag,
               negative_control_flagged=nc_e_flag and nc_d_flag, elapsed_s=int(round(time.time() - t0)))

    # 7. [output] Check 4: the exact schema (booleans and ints only), then the bytes and their sha256.
    enter("output")
    out = {k: (bool(v) if OUT_KEYS.get(k) is bool and isinstance(v, (bool, np.bool_)) else v) for k, v in out.items()}
    if set(out) != set(OUT_KEYS) or any(type(out[k]) is not t for k, t in OUT_KEYS.items()):
        _stop("output_schema")
    body = (json.dumps(out, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    return {"json_bytes": body, "sha256": _sha(body)}


@app.function(image=image, cpu=(CPU, CPU), memory=(MEM_MIB, MEM_MIB), timeout=TIMEOUT, startup_timeout=STARTUP, scaledown_window=SCALEDOWN,
              retries=0, max_containers=1, volumes={"/o": a1_vol.read_only(), "/p": pd_vol.read_only()})
def recheck(rec_gz: bytes, nodes_b: bytes, tpl_b: bytes, meta_in: dict) -> dict:
    return _guarded(_body, rec_gz, nodes_b, tpl_b, meta_in)


RESULTS_DIR = Path("/Users/aswinsurya/probe-shift-reruns-2026-10-20/opt3_recheck/b_incontainer/results")  # the only place written


def _app_id():
    try:
        a = str(getattr(app, "app_id", None) or "unknown")
    except Exception:  # informational only; the status line must still print
        return "unknown"
    return a if re.fullmatch(r"ap-[A-Za-z0-9]+", a) else "unknown"


def _local_guarded(body, *args):
    """v3: the local twin of _guarded. Runs body(st, *args), which prints the status line as its LAST statement and returns rc.
    Returns (rc, None) on success, or (3, <one fail line with a closed-list code>) on ANY exception; never raises, never prints.
    Inside the except block only the stage, the class and (for Stop) the reason are captured, in a nested try; the code is built
    after the block ends, in its own try, with the constant FALLBACK_LINE if building it fails."""
    st = ["prep"]
    failed, stage, etype, reason = False, "unknown", None, None
    try:
        rc = body(st, *args)
    except BaseException as e:  # noqa: BLE001  (the local sanitizing boundary)
        failed = True
        try:
            stage, etype = st[0], type(e)
            if etype is Stop and len(e.args) == 1:
                reason = e.args[0]
        except BaseException:  # noqa: BLE001
            pass
    if not failed:
        return rc, None
    try:
        return 3, f"RECHECK_B_STATUS fail app_id={_app_id()} code={_build_code('fail', stage, etype, reason)}"
    except BaseException:  # noqa: BLE001  (sanitizer failure: a constant, built from nothing the exception carried)
        return 3, FALLBACK_LINE


def _local_fixture_value_error(st, secret):  # v3 local fixture: a local-side exception whose message carries a value
    st[0] = "selftest"
    raise ValueError(f"leak {secret!r}")


def _local_body(st, stamp, run_dir, trailer):
    """Everything the local entrypoint does, inside _local_guarded: preparation and file reads, the remote call, result processing
    and the create-only write, then printing the one status line (the LAST statement)."""
    st[0] = "prep"
    launch_epoch = time.time()
    buf = io.StringIO()  # v3 fixture: through the same boundary, with stdout / stderr captured (nothing may be printed)
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        fx = _local_guarded(_local_fixture_value_error, 0.123456789)
    if fx != (3, f"RECHECK_B_STATUS fail app_id={_app_id()} code=fail:selftest:ValueError") or "0.123" in fx[1] or buf.getvalue():
        _stop("local_selftest")
    here = Path(__file__).resolve().parent
    rd = Path(run_dir).resolve()
    name = f"recheck_b_v3_out_{stamp}.json"
    if not STAMP_RE.fullmatch(stamp) or rd != here / "runs" / stamp or not rd.is_dir():
        _stop("local_stamp_rundir")
    if not TRAILER_RE.fullmatch(trailer):
        _stop("local_trailer")
    if here / "results" != RESULTS_DIR or not RESULTS_DIR.is_dir():
        _stop("local_results_dir")
    out_local = RESULTS_DIR / name
    if out_local.exists():
        _stop("local_output_exists")
    blobs = {}
    for key, fname, want in (("rec", REC_FILE, REC_GZ_SHA), ("nodes", NODES_FILE, NODES_SHA), ("tpl", TPL_FILE, TPL_SHA)):
        b = (rd / fname).read_bytes()  # sha check only; the laptop never parses these
        if _sha(b) != want:
            _stop("local_input_sha")
        blobs[key] = b
    meta_in = {"stamp": stamp, "name": name, "trailer": trailer, "launch_epoch": launch_epoch,
               "script_sha256": _sha(Path(__file__).read_bytes())}
    st[0] = "remote"
    res = recheck.remote(blobs["rec"], blobs["nodes"], blobs["tpl"], meta_in)
    st[0] = "result"
    if isinstance(res, dict) and res.get("status") == "ok":
        body = res.get("json_bytes")
        if not isinstance(body, bytes) or _sha(body) != res.get("sha256"):
            _stop("return_sha")
        with out_local.open("xb") as fh:  # create-only
            fh.write(body)
        rc, line = 0, f"RECHECK_B_STATUS ok app_id={_app_id()} name={name} sha256={_sha(body)} local=results/{name}"
    elif isinstance(res, dict) and res.get("status") == "stop":
        code = res.get("code")
        code = code if isinstance(code, str) and code.startswith("stop:") and CODE_RE.fullmatch(code) else "stop:unknown:Other"
        rc, line = 2, f"RECHECK_B_STATUS stop app_id={_app_id()} code={code}"
    else:
        _stop("return_layout")
    st[0] = "print"
    print(line, flush=True)
    return rc


@app.local_entrypoint()
def main(stamp: str, run_dir: str, trailer: str):
    try:
        rc, line = _local_guarded(_local_body, stamp, run_dir, trailer)
    except BaseException:  # noqa: BLE001  (cannot normally happen: _local_guarded never raises)
        rc, line = 3, FALLBACK_LINE
    if line is not None:  # the failure line (the success line was printed inside the body)
        try:
            print(line, flush=True)
        except BaseException:  # noqa: BLE001  (nothing more can be printed)
            rc = 3
    if rc:
        sys.exit(rc)  # outside every except block: no exception chain; modal's CLI lets SystemExit through with this code
