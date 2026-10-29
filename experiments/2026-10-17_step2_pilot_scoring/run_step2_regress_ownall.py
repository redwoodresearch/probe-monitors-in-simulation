"""STEP-2 bitwise regression for own_all (09 via 2e, 10-17), keys only. Runs ONCE per label, at the scorer's final pin-fill commit.

(A) OLD vs NEW in ONE container and ONE Python process. run_own_all.py at the scorer's base commit (OLD: its git blob) and at HEAD (NEW: the
    file) are each exec'd into their own module object (module name run_own_all_old / run_own_all_new, registered in sys.modules, __file__ = the
    scorer's repo path, so HERE / REPO resolve as for the real file). Both import the same cutfull.py and run_famexp.py, which must be byte-identical
    between base and HEAD (else STOP). The jobs come from 2e's step (2): jobs_compare_in_wrapper.py ran each version's OWN _launch (cutfull
    --followup path, --refuse only=simh_q4f_v1_full_b1,only=simh_q4f_v1_full_b2) in the records-step wrapper, its score spawns captured, and
    published the plans (OLD and NEW score calls, writer dropped from the comparison, old_equals_new). The laptop sends the COMMITTED plans as
    bytes with their sha256 and never parses them; this container checks them (_plans_jobs: sha, family, driver sha == STEP2_DRIVER, OLD / NEW ==
    the sources it scores, the label's kwargs == the runner's, every entry equal) and decodes each version's score calls. One Modal CPU container (the scorer's image:
    torch 2.8.0, numpy 2.3.3, safetensors 0.8.0, the pinned 2e build at /root/x2e) exec's both sources the same way and runs OLD's score.local()
    on OLD's jobs, moves the container-local /work aside, then runs NEW's score.local() on NEW's jobs. The DATA Volumes are mounted READ-ONLY
    (scores at /out, cutfull activations at /b, the production work Volume at /prod); the ONLY writable mount is the raw-text Volume
    probe-shift_aswin-surya_sub-step2pilot_rawtext_kill-2026-11-14 at /raw (one exclusive-create raw file per attempt under the family's subdir:
    ownall_loo / ownall_simt / ownall_loo3); the scorers' OUT / WORK objects are replaced by no-op stubs, and all score outputs go to the
    container's local disk, never to a Volume. Every scored file (probe x readout x folder) is compared OLD vs NEW: every column
    bitwise in file order AND key by key (key, run_key, unit, call_index), every world label, runs_without_rows, and the counted A25 runs (count
    from the pinned counted-key file) covered per probe x readout across b1 + b2. ANY difference: STOP (exit 1) to 09.
(B) NEW vs STORED PRODUCTION, by key: the S_old R / C override files pinned in INPUTS_FULL 2026-10-16_110447 @ 927fabb0d (each cited by path +
    sha256, through its override record's commit, path and sha256) for own_all_rs / own_all_rs_uncut; for probes INPUTS_FULL does not pin
    (simtrained, LOO3), the probe's own committed score record -> manifest (sha) -> npz (sha). Per file: max |delta| (%.3e) and the count of
    differing units over every column; pass if bitwise or within the dtype bound (float32 1e-5*max(1,|s|), float64 1e-9*max(1,|s|), s = stored);
    any unit beyond the bound (or a non-float column difference, or another key set): STOP (exit 1) to 09; within the bound but not bitwise:
    reported. No per-unit value is printed or written. The bound follows the ARITHMETIC precision (09, 10-17): own_all scores are computed in
    float64 (ARITH cites the lines), so 1e-9 * max(1, |s|) on score / score_by_model; every other column must be equal exactly.
Machine metadata (CPU model, OpenBLAS core type and build config, modal / numpy / torch versions, env): written in full to the raw file; relayed
only as its sha256 plus, under 2e's ruling (b), the OpenBLAS core type (if allowlisted) and the modal / numpy versions (if x.y.z), else "other". In the
container, modal and numpy must EQUAL the literal pins 1.6.1 / 2.3.3 before any scoring (else VersionPinError, relayed as its type only).
Logs and the keys-only record: /Users/aswinsurya/probe-shift-cut-step2-logs-2026-10-17/own_all_regress/<label>/ (record also under results/).

REFUSES before run() does anything (the module-level import only writes the sha-checked BUILD) while STEP2_DRIVER is unset, names another path
than STEP2_DRIVER_PATH, or its file does not hash to its sha256 (2e, 10-17: second review R1, then step (2): no scorer runs on the laptop; the jobs
come from the plans). Then REFUSES before any Modal call (2e, 10-17) unless this interpreter's modal is the literal pin 1.6.1, HEAD == the given commit (full sha), the tracked tree is clean, HEAD is the scorer's origin
branch (its tip on origin by `git ls-remote`, and the local ref), the scorer's blob at HEAD is NEW_SCORER_BLOB (run step; 507-prep, 10-17; the record
stores it), and every pilot pin in the NEW scorer (PILOT_ROUTING / _ATTEST / _INVENTORY / _KEYS / _STEP0_GATE, parsed by AST) is set and its file at the
pinned commit has the pinned sha256. All local work (checks, the committed plans' bytes and sha256, citations) finishes before the one app.run().
One container: 8 CPU / 64 GiB (REGRESS_MEMORY_MIB, 2e 10-17), timeout 3600 s (REGRESS_TIMEOUT_S, 2e ruling (ii), 10-17), retries 0; worst
case at the timeout 3600 s x $0.00024688/s (8 x $0.0000131 per core-s + 64 x $0.00000222 per GiB-s, Modal list prices) = $0.888768. An OOM is
a typed _FAILED record (no result). Peak RSS after each score call is recorded (an int).

  # REFUSED until STEP2_DRIVER is set (its own pin-fill delta). Before it: the wrapper's ownall_regress_census step, its two census files
  # committed, then the ownall_regress_plans step, its plans file committed (the launch NOTE). Then ONCE per label:
  #    with the named modal environment's interpreter (2e's rule: never `uv run modal`; the venv of M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal)
  PYTHONPATH=src caffeinate -is /Users/aswinsurya/.cache/probe-shift-modal161/bin/python experiments/2026-10-17_step2_pilot_scoring/run_step2_regress_ownall.py \\
      run <label> <commit> <plans repo path> <plans sha256>
  labels: own_all_rs_uncut, own_all_rs (OWN_LABELS, on aswin/own-all-loo; the plans cover only these two; the other LABELS entries are kept
  for their record and are refused here)
"""

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import types
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

MODAL_PIN = "1.6.1"  # 2e (10-17): the LITERAL modal pin, asserted equal client side (here at import, and in run()) and in the container (_pin_check)
if modal.is_local() and modal.__version__ != MODAL_PIN:  # client side, BEFORE any Modal object is built or the scorer imported (Astra, 10-17): drift STOPs
    raise SystemExit(f"REFUSED (no Modal call made): this interpreter's modal is not the literal pin {MODAL_PIN} (use the named environment's python)")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
SCORER = "experiments/2026-10-14_p2_family_expansion/run_own_all.py"
DEPS = ("experiments/2026-10-14_p2_family_expansion/cutfull.py", "experiments/2026-10-14_p2_family_expansion/run_famexp.py")  # imported by both versions
APP = "probe-shift_aswin-surya_sub-step2regress_kill-2026-11-14"
LOGS = Path("/Users/aswinsurya/probe-shift-cut-step2-logs-2026-10-17/own_all_regress")
BUILD = Path("/tmp/famexp_build")  # the scorer's pinned 2e build (run_famexp materializes it locally, byte-checked), at /root/x2e as in the scorer
RES = "experiments/2026-10-14_p2_family_expansion/results"
INV_FIX = "09b070617:experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json:4a5d0fb3cad32740f849df5def3c7a03a40a18aa4c42bc360709f0c740973d03"
FOLDERS = ("simh_q4f_v1_full_b1_cutfull_v1__sim_honest", "simh_q4f_v1_full_b2_cutfull_v1__sim_honest")  # S_old A25 (S_old's H_sim)
COUNTED = ("572f67dbb", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_primary_v3.json",
           "80753f44aa66e78445f233e43e18c96fd01913246afcc03420b9322c447f4082")  # the counted A25 runs (own_all's COUNTED_SHA)
INPUTS_FULL = ("927fabb0d", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-16_110447.json",
               "d6923a75aa8f914cd3b2e92d8403f125dd095219997aebea8b1d6e4e2b67f77c")  # the FROZEN inputs pooled A read
COMMON = {"mode": "score", "inputs": f"{RES}/own_all_inputs_2026-10-15.json", "cutfull": True, "inventory_b": INV_FIX,
          "refuse": "only=simh_q4f_v1_full_b1,only=simh_q4f_v1_full_b2"}
LABELS = {  # base = the scorer commit this branch was cut from; follow-up parents score no S_old A25 message (relaunch / ep2 only), same fit
    "own_all_rs_uncut": {"raw_family": "ownall_loo", "base": "82f6e77ba67ea74fa1291bd428ddb4a56d11ca6f", "branch": "origin/aswin/own-all-loo", "sub": "own_all_rs_uncut", "prod": "inputs_full",
                         "orig": f"{RES}/own_all_rs_uncut_cutfull_score_2026-10-16_033743.json",
                         "kw": {"fit_stamp": "2026-10-16_030614", "rs": True, "uncut": True, "routing_commit": "d38fb15ec21b",
                                "followup": f"{RES}/own_all_rs_uncut_cutfull_score_2026-10-16_043127.json"}},
    "own_all_rs": {"raw_family": "ownall_loo", "base": "82f6e77ba67ea74fa1291bd428ddb4a56d11ca6f", "branch": "origin/aswin/own-all-loo", "sub": "own_all_rs", "prod": "inputs_full",
                   "orig": f"{RES}/own_all_rs_cutfull_score_2026-10-16_023727.json",
                   "kw": {"fit_stamp": "2026-10-16_003049", "rs": True, "routing_commit": "d38fb15ec21b", "followup": f"{RES}/own_all_rs_cutfull_score_2026-10-16_043125.json"}},
    "own_all_rs_simtrained_uncut": {"raw_family": "ownall_simt", "base": "460acc84dac4fcd256fa222b6ff632532e18bdc1", "branch": "origin/aswin/own-all-simtrained-2e", "sub": "own_all_rs_simtrained_uncut",
                                    "prod": "record", "orig": f"{RES}/own_all_rs_simtrained_uncut_cutfull_score_2026-10-16_154236.json",
                                    "kw": {"fit_stamp": "2026-10-16_152335", "rs": True, "uncut": True, "simtrained": True, "routing_commit": "d38fb15ec21b",
                                           "simt_keys": f"{RES}/own_all_rs_simtrained_uncut_training_keys_2026-10-16_152226.json",
                                           "followup": f"{RES}/own_all_rs_simtrained_uncut_cutfull_score_2026-10-16_154830.json"}},
    "own_all_loo3_uncut": {"raw_family": "ownall_loo3", "base": "e2003d463894b9688494cfea586e8338922aa3df", "branch": "origin/aswin/own-all-loo3-uncut-4f", "sub": "own_all_loo3_uncut", "prod": "record",
                           "orig": f"{RES}/own_all_loo3_uncut_cutfull_score_2026-10-16_034101.json",
                           "kw": {"fit_stamp": "2026-10-16_033003", "loo": True, "uncut": True, "uncut_rows": "2026-10-16_030614",
                                  "routing_commit": "e9c9ebf964154921df735feb1fc866b4ca196e7b", "followup": f"{RES}/own_all_loo3_uncut_cutfull_score_2026-10-16_044020.json"}},
    "own_all_loo4_uncut": {"raw_family": "ownall_loo3", "base": "e2003d463894b9688494cfea586e8338922aa3df", "branch": "origin/aswin/own-all-loo3-uncut-4f", "sub": "own_all_loo4_uncut", "prod": "record",
                           "orig": f"{RES}/own_all_loo4_uncut_cutfull_score_2026-10-16_041727.json",  # LOO4's fit 2026-10-16_040431 names its S_paper493 inputs:
                           "kw": {"fit_stamp": "2026-10-16_040431", "loo": True, "uncut": True, "loo4": True, "uncut_rows": "2026-10-16_030614",
                                  "sp493_inv": "85447113f:experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w1_v1_2026-10-16_025411.json:"
                                               "3dc8a91a52d694c8959db640b1de5688d2a6a657641d8ee7f79d4cff8aa2d067,"
                                               "102b3741e:experiments/2026-10-15_cut_sensitivity/results/inventory_sp493_w23_v1_2026-10-16_033210.json:"
                                               "b11a63f5914c1c5d199c3401104acd9697b6d3e7095802f24e67635895dc98f5",
                                  "sp493_routing": "e9c9ebf964154921df735feb1fc866b4ca196e7b", "routing_commit": "e9c9ebf964154921df735feb1fc866b4ca196e7b",
                                  "followup": f"{RES}/own_all_loo4_uncut_cutfull_score_2026-10-16_044049.json"}}}
PIN_NAMES = ("PILOT_ROUTING", "PILOT_ATTEST", "PILOT_INVENTORY", "PILOT_KEYS", "PILOT_STEP0_GATE")  # every pilot pin of the NEW scorer
NEW_SCORER_BLOB = "e69ee2c7c862858042a3e667bc1d54f39479dd62"  # 507-prep (10-17): `git rev-parse HEAD:<SCORER>` (40 hex) after the pin-fill; recompute on any scorer edit
STEP2_DRIVER = ("2e0e20f49f9397a86f64c61ae95d7b44aadb2ae7", "experiments/2026-10-17_step2_pilot_scoring/jobs_compare_in_wrapper.py",
                "126def416acb7c5471f1b25f78d45499774729a813c2c8d5eeb3b1ef11f58e19")  # the PIN FILL (2e, after D and the step (2) commits PASS): the commit that added the driver, its path, its sha256
# 2e's step (2), 10-17: the reviewed OLD-vs-NEW job driver, as (commit, STEP2_DRIVER_PATH, sha256). run() is REFUSED (_step2_gate,
# before it does anything; module-level imports only write the sha-checked BUILD) while it is unset; it is set ONLY in its own small pin-fill delta,
# after D (aswin/step2-recstep-ownfam) and the driver pass review. The regression container checks the plans' driver_sha256 against it.
STEP2_DRIVER_PATH = "experiments/2026-10-17_step2_pilot_scoring/jobs_compare_in_wrapper.py"  # the one file STEP2_DRIVER may pin
OWN_LABELS = ("own_all_rs_uncut", "own_all_rs")  # the labels the driver's plans cover (route C); the others are not regressed here
PLANS_RE = re.compile(r"experiments/2026-10-17_step2_pilot_scoring/results/regress_compare_ownall_plans_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}\.json")
ROW_KEY = ("key", "run_key", "unit", "call_index")
SCORE_COLS = ("score", "score_by_model")  # the computed columns; every other column must be equal exactly
ARITH_TOL = {"float32": 1e-5, "float64": 1e-9}  # 09 (via CUT, 10-17): the (B) bound follows the ARITHMETIC precision of the score, not the stored dtype
ARITH = {"82f6e77ba": "float64: run_own_all.py L803 X64 = X.astype(np.float64); L690 _logits = ((X64 - mean) / std) @ W.astype(float64) + b; score / score_by_model "
                      "float64 (np.full(n, nan), L807), L817 / L820 (same lines at 82f6e77ba and after the STEP-2 edit)",
         "460acc84d": "float64: run_own_all.py L1037 X64 = X.astype(np.float64); L922 _logits = ((X64 - mean) / std) @ W.astype(float64) + b; L1041, L1051 / L1054 (same lines after the edit)",
         "e2003d463": "float64: run_own_all.py L1186 X64 = X.astype(np.float64); L1056 _logits = ((X64 - mean) / std) @ W.astype(float64) + b; L1190, L1200 / L1203 (same lines after the edit)"}

if modal.is_local():  # the scorer's own import materializes the pinned 2e build into BUILD (byte-checked) before the app is created
    sys.path.insert(0, str(REPO / Path(SCORER).parent))
    import run_famexp  # noqa: F401

app = modal.App(APP)
VOLS = {"/out": modal.Volume.from_name("probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", version=2).read_only(),
        "/b": modal.Volume.from_name("probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14", version=2).read_only(),
        "/prod": modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2).read_only()}
img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy==2.3.3", "safetensors==0.8.0").add_local_dir(BUILD, "/root/x2e")
RAW_NAME = "probe-shift_aswin-surya_sub-step2pilot_rawtext_kill-2026-11-14"  # 2e (10-17): ONE raw-text Volume for every STEP-2 family, each under its own subdir
RAW = modal.Volume.from_name(RAW_NAME, create_if_missing=True, version=2)  # the only writable mount (/raw), for the raw-text write; every DATA Volume is read-only
CORE_ALLOW = re.compile(r"^(Haswell|SkylakeX|Zen|Cooperlake|SapphireRapids|Prescott|Nehalem|Sandybridge|Skylake|Atom)$")  # 2e ruling (b): relayable cores
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$", re.ASCII)  # 2e ruling (b): a relayable version string; it gates the RELAY only (_machine_relay), never an assert
NUMPY_PIN = "2.3.3"  # 2e (10-17): the LITERAL numpy pin, asserted equal in the container (_pin_check); drift STOPs
REGRESS_TIMEOUT_S = 3600  # 2e ruling (ii), 10-17: fits four score calls on every committed bound (largest: own_all_rs_uncut 4 x 768 = 3,072 s)
REGRESS_MEMORY_MIB = 65536  # 2e (10-17): 64 GiB, request = limit (CPU stays 8): four score calls share ONE process here; production ran one per container
WORK_DIR, OLD_DIR, PROD_DIR, RAW_ROOT = Path("/work"), Path("/regress_old_work"), Path("/prod"), Path("/raw")  # container paths (a test points them elsewhere)
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


class _NoVol:  # the scorers' OUT / WORK inside the regression: nothing is reloaded or committed (read-only mounts; outputs on local disk)
    reload = commit = staticmethod(lambda: None)


def _load(name, src, path):
    """One scorer version as its own module in this process: exec of its source, __file__ = the scorer's path (HERE / REPO as for the real file)."""
    mod = types.ModuleType(name)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(src, path, "exec"), mod.__dict__)  # noqa: S102
    return mod


def _blas() -> dict:
    """CPU model, numpy's OpenBLAS core type (runtime, via ctypes) and build config, numpy / torch versions."""
    import ctypes
    import glob

    import numpy as np

    try:
        cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "?")
    except OSError:
        cpu = "unavailable (no /proc/cpuinfo)"
    out = {"cpu_model": cpu, "cpu_count": os.cpu_count(), "modal": modal.__version__,
           "numpy": np.__version__, "env": {k: v for k, v in os.environ.items() if k.startswith(("OPENBLAS", "OMP_", "MKL_"))}}
    try:
        import torch

        out["torch"] = torch.__version__
    except ImportError:
        out["torch"] = None
    try:
        out["numpy_blas_build"] = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    except Exception as e:  # informational only
        out["numpy_blas_build"] = f"unavailable: {type(e).__name__}"
    for lib in sorted(glob.glob(str(Path(np.__file__).parent.parent / "numpy.libs" / "*openblas*"))):
        so = ctypes.CDLL(lib)
        for sym in ("scipy_openblas_get_corename64_", "scipy_openblas_get_corename", "openblas_get_corename64_", "openblas_get_corename"):
            if hasattr(so, sym):
                f = getattr(so, sym)
                f.restype = ctypes.c_char_p
                out |= {"openblas_lib": Path(lib).name, "openblas_core_type": f().decode("utf-8", errors="backslashreplace"), "openblas_symbol": sym}
                break
    out.setdefault("openblas_core_type", "not found (no OpenBLAS get_corename symbol in numpy.libs)")
    return out


def _plain(o):
    """json default: a numpy integer / bool scalar becomes its Python value, a set a sorted list; anything else fails loudly (and leaves as a typed
    failure), a numpy float included (Astra (b), 10-17). A numpy float64 is a Python float, which json writes without asking: _no_float refuses it.
    A TYPE validator, not a free-text validator: string safety comes from the fixed field set _regress_body returns (keys, shas, labels, counts,
    %.3e maxima), never from this check."""
    if type(o).__module__ == "numpy" and getattr(o, "ndim", None) == 0 and getattr(getattr(o, "dtype", None), "kind", None) in ("b", "i", "u"):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(type(o).__name__)


def _no_float(o):
    """Astra (b), 10-17: the relayed result holds no float anywhere (key or value; a numpy float64 is one), so no measured value can leave by
    accident; measured maxima leave only as the sanctioned %.3e strings. Returns o, else raises TypeError (relayed as its type only). A TYPE
    validator, not a free-text validator: a string passes whatever it says; string safety comes from the fixed field set _regress_body returns."""
    if isinstance(o, float):
        raise TypeError("a float in the relayed result")
    if isinstance(o, dict):
        for k, v in o.items():
            _no_float(k)
            _no_float(v)
    elif isinstance(o, (list, tuple, set, frozenset)):
        for v in o:
            _no_float(v)
    return o


def _attempt_token():
    """The per-attempt part of a raw file's name: the Modal input id when there is one, plus a uuid4, so a retried or re-run input never reuses a name."""
    try:
        iid = modal.current_input_id()
    except Exception:  # noqa: BLE001  outside a Modal input
        iid = None
    return f"{iid or 'noinput'}_{uuid.uuid4().hex}"


def _raw_write(path, b):
    """Exclusive create ("xb"): a second write to the same name raises (FileExistsError), never overwrites."""
    with path.open("xb") as f:
        f.write(b)


def _machine_relay(m):
    """2e ruling (b), the ONE machine exception (09 needs it to judge ulp differences): the OpenBLAS core type only if it fullmatches CORE_ALLOW,
    the modal and numpy versions only if each fullmatches VERSION_RE; anything else is the literal "other" (the full value is in the raw file)."""
    pick = lambda v, rx: v if isinstance(v, str) and rx.fullmatch(v) else "other"  # noqa: E731
    m = m or {}
    return {"openblas_core_type": pick(m.get("openblas_core_type"), CORE_ALLOW), "modal_version": pick(m.get("modal"), VERSION_RE),
            "numpy_version": pick(m.get("numpy"), VERSION_RE)}


class VersionPinError(Exception):
    """A tool version differs from its literal pin; _contained relays it as this type and a count only (the versions are in the raw file)."""


def _pin_check(m):
    """2e (10-17), in the container: modal == MODAL_PIN and numpy == NUMPY_PIN, compared EXACTLY (literal pins, no regex), else VersionPinError.
    VERSION_RE never asserts: it only gates what _machine_relay may relay, so 1.6.2 / 2.3.4 STOP here and would still relay as their strings."""
    bad = [f"{k} {m.get(k)!r} is not {want!r}" for k, want in (("modal", MODAL_PIN), ("numpy", NUMPY_PIN)) if m.get(k) != want]
    if bad:
        raise VersionPinError("; ".join(bad))


def _flush_native():
    """libc fflush(NULL): C-level stdio buffers (a native printf; os.write bypasses them) are written to the fds 1 / 2 OF THE MOMENT, so _contained
    calls it while they still point at its capture (Astra, 10-17). fflush reports errors by its return value: nonzero raises (Astra, 10-17)."""
    import ctypes

    if ctypes.CDLL(None).fflush(None) != 0:
        raise OSError("libc fflush(NULL) returned nonzero")


def _fds_to_devnull():
    """Fail closed (Astra, 10-17): fds 1 and 2 to /dev/null, so output still pending in a buffer can never reach the real fds."""
    nul = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(nul, 1)
        os.dup2(nul, 2)
    finally:
        os.close(nul)


def _peak_rss():
    """This process's peak resident set size so far, an int (resource.getrusage(RUSAGE_SELF).ru_maxrss: KiB on Linux, the container's OS).
    _regress_body records it after each score call, inside the capture (2e, 10-17: four calls share one process here)."""
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _contained(raw_dir, fn, *a, commit=None):
    """Run fn in this container; ONE json string of structured fields leaves (booleans, counts, keys, shas, dtype names, max |delta| as %.3e; no
    float, _no_float), so the local side gets plain Python only (it has no numpy). The CAPTURE (sys.stdout / sys.stderr redirected to a buffer,
    fds 1 / 2 dup2'd onto a temporary file, libc's stdio flushed into it) is active through fn AND all finalization (encoding, serialization, the
    raw-file create and write, commit = the raw Volume's commit), and is undone only in the outermost cleanup, after the sanitized outcome is
    chosen, each restore step in its own try (a failing step never skips the next restore; its TYPE is added as cleanup_error_types) (Astra,
    10-17). fn's Python-level and fd-level output (native printf included), any exception's traceback and the FULL machine metadata go to one raw
    file, <raw_dir>/raw_<attempt token>.txt on the raw-text Volume, created exclusively ("xb"), its text encoded as UTF-8 with
    errors="backslashreplace"; output during finalization itself stays in the capture and is discarded. Relayed: the raw file's {volume, path,
    sha256}; on success the machine metadata's sha256 and _machine_relay's three fields; on a failure of fn only the exception TYPE and a count
    (stage "body"); a failure in finalization leaves as its type and a count only (stage "finalize"), never its text. If any cleanup flush
    fails (or libc fflush returns nonzero), the cleanup FAILS CLOSED: fds 1 / 2 go to /dev/null instead of back to the real ones, the capture is
    closed undrained, and the outcome is failed (stage "cleanup" unless already failed; cleanup_error_types) (CUT / Astra / 507-prep / 2e ruling (b),
    10-17)."""
    enc = lambda t: t.encode("utf-8", errors="backslashreplace")  # noqa: E731  the one encoding policy for raw text
    err, raw, buf, fd_tmp, saved, redirect = None, None, io.StringIO(), None, [], contextlib.ExitStack()
    outcome = json.dumps({"failed": True, "stage": "finalize", "error_type": "NoOutcome", "errors": 1, "body_error_type": None, "raw": None})
    try:
        fd_tmp = tempfile.TemporaryFile()
        for s_ in (sys.stdout, sys.stderr):
            s_.flush()
        _flush_native()  # the caller's own pending native output goes to the caller's fds, not into the capture
        saved = [os.dup(1), os.dup(2)]
        os.dup2(fd_tmp.fileno(), 1)
        os.dup2(fd_tmp.fileno(), 2)
        redirect.enter_context(contextlib.redirect_stdout(buf))
        redirect.enter_context(contextlib.redirect_stderr(buf))
        out, machine = None, None
        try:
            res = fn(*a)
            machine = res.pop("machine", None)
            out = json.loads(json.dumps(_no_float(res), default=_plain))
        except BaseException as e:  # noqa: BLE001  any failure of fn, SystemExit included, leaves as its type only
            err = type(e).__name__
            buf.write("".join(traceback.format_exception(e)))
        _flush_native()  # fn's buffered native output into the capture, so into the raw file (the failure path too)
        for s_ in (sys.__stdout__, sys.__stderr__):
            s_.flush()
        mj = enc(json.dumps(machine, sort_keys=True, default=repr))
        fd_tmp.seek(0)
        b = enc("== python stdout / stderr ==\n" + buf.getvalue() + "\n== fd 1 / 2 ==\n") + fd_tmp.read() + enc("\n== machine (json) ==\n") + mj + b"\n"
        raw_dir.mkdir(parents=True, exist_ok=True)  # the parent directory only; the file itself is exclusive-create
        path = raw_dir / f"raw_{_attempt_token()}.txt"
        _raw_write(path, b)
        raw = {"volume": RAW_NAME, "path": str(path), "sha256": hashlib.sha256(b).hexdigest()}
        if commit is not None:
            commit()  # still inside the capture: whatever it prints, warns or writes to fd 1 / 2 stays here
        if err:
            outcome = json.dumps({"failed": True, "stage": "body", "error_type": err, "errors": 1, "raw": raw})
        else:
            outcome = json.dumps(out | {"failed": False, "raw": raw, "machine_sha256": hashlib.sha256(mj).hexdigest(), "machine_relay": _machine_relay(machine)})
    except BaseException as e:  # noqa: BLE001  a finalization failure: its type and a count only (with fn's error type, if fn failed too)
        outcome = json.dumps({"failed": True, "stage": "finalize", "error_type": type(e).__name__, "errors": 1 + (err is not None), "body_error_type": err, "raw": raw})
    finally:  # the outermost cleanup, after the outcome is chosen: native and Python buffers into the capture, then the restore, each in its own try
        cleanup_err = []

        def step(f):
            try:
                f()
                return True
            except BaseException as e:  # noqa: BLE001  a failing step never skips the next one; its type only
                cleanup_err.append(type(e).__name__)
                return False
        flushed = all([step(_flush_native), step(lambda: sys.__stdout__.flush()), step(lambda: sys.__stderr__.flush())])  # every flush is attempted
        step(redirect.close)
        if flushed:
            for i, fd in zip((1, 2), saved):
                step(lambda fd=fd, i=i: os.dup2(fd, i))
        else:  # FAIL CLOSED (Astra, 10-17): output may still be pending in a buffer: fds 1 / 2 go to /dev/null, never back to the real ones
            step(_fds_to_devnull)
        for fd in saved:
            step(lambda fd=fd: os.close(fd))
        if fd_tmp is not None:
            step(fd_tmp.close)  # the capture is closed, never drained to the real fds
    if cleanup_err:  # any cleanup failure fails the outcome, type-only (a success's result fields are dropped)
        o = json.loads(outcome)
        outcome = json.dumps(o | {"errors": o["errors"] + len(cleanup_err), "cleanup_error_types": cleanup_err} if o.get("failed") else
                             {"failed": True, "stage": "cleanup", "error_type": cleanup_err[0], "errors": len(cleanup_err), "raw": o.get("raw"),
                              "cleanup_error_types": cleanup_err})
    return outcome


def _remote_json(call, *a):
    """The local side of the one Modal call (507-prep, 10-17): its json string, parsed. A client-side failure (timeout, OOM, preemption, a lost
    connection) becomes a typed failure, its type and a count only, never its message, so the _FAILED record is still written."""
    try:
        return json.loads(call(*a))
    except BaseException as e:  # noqa: BLE001
        return {"failed": True, "stage": "remote", "error_type": type(e).__name__, "errors": 1, "raw": None}


def _app_lifecycle(app_, fn_, *a):
    """The WHOLE client-side app lifecycle (Astra, 10-17): entering app_.run() (auth, hydration), the app id, the one remote call (its own
    failures typed by _remote_json) and leaving it (teardown), under one type-only boundary. Any failure there is {failed, stage "app_lifecycle",
    error_type, errors, raw None, remote_result_received}, never its message. Returns (result, app id); app id is "" until the app is entered.
    Not captured: the Modal SDK's own client-side startup / teardown output (Modal's text, none of ours; no result or raw text of this job)."""
    app_id, res = "", None
    try:
        with app_.run():
            app_id = app_.app_id or ""
            res = _remote_json(fn_.remote, *a)
    except BaseException as e:  # noqa: BLE001
        return {"failed": True, "stage": "app_lifecycle", "error_type": type(e).__name__, "errors": 1, "raw": None, "remote_result_received": res is not None}, app_id
    return res, app_id


def _cmp(a_dir, b_dir, name, tol=None):
    """One scored file, A vs B (A = OLD or the stored production file): column set, bitwise in order, key by key; with tol (the arithmetic bound,
    1e-9 for float64 / 1e-5 for float32 arithmetic), per-unit max |delta| on the score columns against tol * max(1, |s|), s = A's (stored) value,
    every other column exactly. max |delta| covers every numeric column in both modes and reads inf for a non-finite mismatch, a column of another
    dtype or shape, or files that are not comparable (columns, per-column structure, key set or key uniqueness differ; checked before any
    permutation, whatever the row count), so a difference never reads as 0 and never passes; a non-numeric difference shows in
    units_not_bitwise_equal. Booleans and counts only (max |delta| as a %.3e string)."""
    import numpy as np

    za, zb = np.load(io.BytesIO(Path(a_dir, name).read_bytes()), allow_pickle=False), np.load(io.BytesIO(Path(b_dir, name).read_bytes()), allow_pickle=False)
    ca, cb = sorted(za.files), sorted(zb.files)
    na, nb = (z["key"].shape[0] if "key" in z.files and z["key"].ndim >= 1 else -1 for z in (za, zb))
    # Astra R4: structure BEFORE any permutation, whatever the row count: the same columns, and per column the same dtype, the same trailing shape and
    # exactly one row per key row on each side; a structural mismatch is not comparable (FAIL, max |delta| inf), never an IndexError or a pass
    struct = ca == cb and na >= 0 and nb >= 0 and all(za[c].ndim >= 1 and zb[c].ndim >= 1 and za[c].dtype == zb[c].dtype and za[c].shape[1:] == zb[c].shape[1:]
                                                     and za[c].shape[0] == na and zb[c].shape[0] == nb for c in ca)
    na, nb = max(na, 0), max(nb, 0)
    r = {"same_columns": ca == cb, "same_structure": struct, "units_a": na, "units_b": nb,
         "bitwise_in_order": struct and all(za[c].shape == zb[c].shape and za[c].tobytes() == zb[c].tobytes() for c in ca)}
    kc = [c for c in ROW_KEY if c in ca and c in cb]
    ka = [tuple(str(za[c][i]) for c in kc) for i in range(na)] if struct else []
    kb = [tuple(str(zb[c][i]) for c in kc) for i in range(nb)] if struct else []
    r |= {"row_key_columns": kc, "row_keys_unique": struct and len(set(ka)) == na and len(set(kb)) == nb, "same_row_keys": struct and set(ka) == set(kb)}
    diff, beyond, mx = np.zeros(na, bool), np.zeros(na, bool), 0.0
    shaped = struct and r["row_keys_unique"] and r["same_row_keys"]
    if shaped and na:
        pos = {k: i for i, k in enumerate(kb)}
        perm = np.array([pos[k] for k in ka], np.int64)
        for c in ca:
            x, y = np.ascontiguousarray(za[c]), np.ascontiguousarray(zb[c][perm])
            if x.dtype != y.dtype or x.shape != y.shape:
                diff[:] = beyond[:] = True
                mx = float("inf")  # a column of another dtype or shape is not comparable: max |delta| reads inf, never 0
                continue
            neq = ~(x.reshape(na, -1).view(np.uint8) == y.reshape(na, -1).view(np.uint8)).all(axis=1)
            diff |= neq
            if np.issubdtype(x.dtype, np.number):  # max |delta| over EVERY numeric column, in (A) and (B); a non-finite mismatch reads inf, never 0
                xx, yy = x.reshape(na, -1).astype(np.float64), y.reshape(na, -1).astype(np.float64)
                both_nan = np.isnan(xx) & np.isnan(yy)
                fin = np.isfinite(xx) & np.isfinite(yy)
                d = np.where(fin, np.abs(xx - yy), 0.0)
                nonfin = ~fin & ~both_nan & ~((xx == yy) & ~np.isnan(xx))
                mx = max(mx, float("inf") if nonfin.any() else float(d.max()) if d.size else 0.0)
            if tol is not None and c in SCORE_COLS and np.issubdtype(x.dtype, np.floating):
                bad = nonfin | (d > tol * np.maximum(1.0, np.abs(np.where(fin, xx, 0.0))))
                beyond |= bad.any(axis=1)
            else:
                beyond |= neq
    if not shaped:  # not comparable (structure, key set or key uniqueness differ), even with zero rows: max |delta| reads inf, never 0
        mx = float("inf")
    r |= {"units_not_bitwise_equal": int(diff.sum()) if shaped else max(na, nb), "units_beyond_bound": int(beyond.sum()) if shaped else max(na, nb),
          "max_abs_delta": f"{mx:.3e}"}
    r["bitwise_by_key"] = shaped and r["units_not_bitwise_equal"] == 0
    keys_a, keys_b = (set(z["key"].astype(str).reshape(-1)) if "key" in z.files else set() for z in (za, zb))
    wa, wb = sorted({k.split("|", 1)[0] for k in keys_a}), sorted({k.split("|", 1)[0] for k in keys_b})
    r |= {"world_labels_a": wa, "world_labels_b": wb, "world_labels_unchanged": wa == wb}
    if tol is not None:
        r["pass"] = shaped and r["units_beyond_bound"] == 0 and r["world_labels_unchanged"]
    else:
        r["pass"] = r["bitwise_in_order"] and r["bitwise_by_key"] and r["world_labels_unchanged"] and wb == ["simh_full"]
    return r, keys_a, keys_b


@app.function(image=img, cpu=(8, 8), memory=(REGRESS_MEMORY_MIB, REGRESS_MEMORY_MIB), timeout=REGRESS_TIMEOUT_S, retries=0, scaledown_window=2, volumes=VOLS | {"/raw": RAW})
def regress(label: str, sources: dict, plans: bytes, plans_sha: str, kw_want: dict, prod: list, expected: list, tol: float, stamp: str) -> str:
    """The one container: the step (2) plans checked and decoded HERE (_plans_jobs), then _regress_body, both under _contained, the raw Volume's
    commit inside its type-only boundary; returns ONE json string (structured fields only; raw text stays on the raw Volume). label is in
    OWN_LABELS (run() checks it before any Modal call)."""
    return _contained(RAW_ROOT / LABELS[label]["raw_family"] / f"{label}_{stamp}", _plans_then_body, label, sources, plans, plans_sha, kw_want, prod, expected,
                      tol, commit=RAW.commit)


def _dec(o):
    """jobs_compare_in_wrapper.py's lossless encoding back to the call arguments: {"__tuple__": [...]} a tuple, {"__dict__": [[k, v], ...]} a dict."""
    if type(o) is list:
        return [_dec(x) for x in o]
    if type(o) is dict:
        if set(o) == {"__tuple__"}:
            return tuple(_dec(x) for x in o["__tuple__"])
        if set(o) == {"__dict__"}:
            return {_dec(k): _dec(v) for k, v in o["__dict__"]}
        return {k: _dec(v) for k, v in o.items()}
    return o


def _plans_jobs(label, sources, plans, plans_sha, kw_want):
    """2e's step (2): the plans the driver published (run in the records wrapper), parsed ONLY here, in the container. STOP unless: the bytes
    have plans_sha; family ownall, mode plans; driver_sha256 == STEP2_DRIVER's; OLD / NEW == the sources this container scores (sha256), NEW's
    blob == NEW_SCORER_BLOB; the label's kwargs == the runner's; old_equals_new and every entry of the label equal; exactly b1 / b2. Then each
    version's score calls for the label (the payload put back into slot 2)."""
    if not isinstance(plans, bytes) or sha(plans) != plans_sha:
        raise SystemExit("STOP: the plans bytes differ from their sha256")
    p = json.loads(plans)
    if p.get("family") != "ownall" or p.get("mode") != "plans" or not (isinstance(STEP2_DRIVER, tuple) and p.get("driver_sha256") == STEP2_DRIVER[2]) \
            or p.get("old_equals_new") is not True or p.get("folders") != list(FOLDERS):
        raise SystemExit("STOP: the plans are not the pinned driver's own_all plans with old_equals_new true over b1 / b2")
    if p["old"]["sha256"] != sha(sources["old"].encode()) or p["new"]["sha256"] != sha(sources["new"].encode()) or p["new"]["blob"] != NEW_SCORER_BLOB:
        raise SystemExit("STOP: the plans' OLD / NEW scorers are not the ones this container scores")
    if p["kwargs"].get(label) != kw_want:
        raise SystemExit("STOP: the plans were built with other kwargs than this runner's for the label")
    keys = [f"{label}|payload"] + [f"{label}|{fo}" for fo in FOLDERS]
    if any(p["body"][k]["equal"] is not True for k in keys):
        raise SystemExit("STOP: an OLD / NEW entry of the label differs")
    jobs = {}
    for v in ("old", "new"):
        pay = _dec(p["body"][f"{label}|payload"][v])
        jobs[v] = [a[:2] + (pay,) + a[3:] for fo in FOLDERS for a in _dec(p["body"][f"{label}|{fo}"][v])]
        if not jobs[v] or any(len(a) != 8 or a[2] is not pay for a in jobs[v]):
            raise SystemExit("STOP: the plans' jobs are not 8-argument score calls")
    return jobs


def _plans_then_body(label, sources, plans, plans_sha, kw_want, prod, expected, tol):
    jobs = _plans_jobs(label, sources, plans, plans_sha, kw_want)
    return _regress_body(label, sources, jobs, prod, expected, tol) | {"plans": {"sha256": plans_sha, "score_calls": {v: len(j) for v, j in jobs.items()}}}


def _regress_body(label, sources, jobs, prod, expected, tol):
    """(A) OLD then NEW score.local() in this one process on a local /work; (B) NEW vs the stored production files on /prod. Keys only."""
    blas = _blas()
    _pin_check(blas)  # 2e (10-17): the literal tool pins, before any scoring; drift STOPs (VersionPinError, type-only relay)
    mods, ran, rss = {}, {}, {}
    for v in ("old", "new"):
        mods[v] = _load(f"run_own_all_{v}", sources[v], f"/root/step2/{v}/{SCORER}")
        mods[v].OUT = mods[v].WORK = _NoVol()
        WORK_DIR.mkdir(exist_ok=True)
        ran[v], rss[v] = [], []
        for a_ in jobs[v]:  # one score call per job, then this process's peak RSS (2e, 10-17)
            ran[v].append(mods[v].score.local(*a_))
            rss[v].append(_peak_rss())
        if v == "old":
            shutil.move(str(WORK_DIR), str(OLD_DIR))
    dirs = {v: {fo: x["path"].replace(f"{WORK_DIR}/", f"{OLD_DIR}/", 1) if v == "old" else x["path"] for r in ran[v] for fo, x in r["folders"].items()} for v in ran}
    a_out, runs = {}, {}
    for fo in FOLDERS:
        mo, mn = (json.loads(Path(dirs[v][fo], "manifest_own_all.json").read_bytes()) for v in ("old", "new"))
        strip = lambda m: {k: x for k, x in m.items() if k not in ("writer", "files")}  # noqa: E731  writer: each run's own stamp; files: npz shas (zip timestamps)
        f_ = {"same_file_names": sorted(mo["files"]) == sorted(mn["files"]), "manifest_equal_except_writer_and_npz_shas": strip(mo) == strip(mn), "files": {}}
        for name in sorted(mo["files"]):
            r, ka, kb = _cmp(dirs["old"][fo], dirs["new"][fo], name)
            rd = name.removesuffix(".npz").rsplit("__", 1)[1]
            for v, m, ks in (("old", mo, ka), ("new", mn, kb)):
                runs.setdefault((v, name), set()).update(ks | set(m["counts"][rd]["runs_without_rows"]))
            r["runs_without_rows_unchanged"] = mo["counts"][rd]["runs_without_rows"] == mn["counts"][rd]["runs_without_rows"]
            r["pass"] = r["pass"] and r["runs_without_rows_unchanged"]
            f_["files"][name] = r
        f_["pass"] = f_["same_file_names"] and f_["manifest_equal_except_writer_and_npz_shas"] and bool(f_["files"]) and all(x["pass"] for x in f_["files"].values())
        a_out[fo] = f_
    cover = {f"{v}|{n}": {"runs": len(s), "equals_counted": s == set(expected)} for (v, n), s in sorted(runs.items())}
    b_out = {}
    for p in prod:  # (B): the stored production file (sha-checked against its citation, its manifest sha-checked too) vs NEW, by key
        pdir = PROD_DIR / p["dir"]
        mb = Path(pdir, p["manifest"]).read_bytes()
        files = json.loads(mb)["files"] if sha(mb) == p["manifest_sha256"] else {}
        for name in ([p["file"]] if p.get("file") else sorted(files)):
            got = sha(Path(pdir, name).read_bytes()) if name in files else None
            if got is None or got != files[name] or (p.get("sha256") and got != p["sha256"]):
                b_out[f"{p['folder']}/{name}"] = {"pass": False, "problem": "production_manifest_or_npz_sha256_mismatch"}
                continue
            r, _, _ = _cmp(pdir, dirs["new"][p["folder"]], name, tol=tol)
            b_out[f"{p['folder']}/{name}"] = r | {"production_sha256": got}
    new_files = {f"{fo}/{n}" for fo in FOLDERS for n in a_out[fo]["files"]}
    return {"label": label, "machine": blas, "peak_rss_kib_after_each_score_call": rss, "A_old_vs_new": a_out, "A_coverage_b1_b2": cover, "counted_runs": len(expected),
            "A_pass": all(x["pass"] for x in a_out.values()) and bool(cover) and all(c["equals_counted"] for c in cover.values()),
            "B_new_vs_production": b_out, "B_files_cover_new": set(b_out) == new_files,
            "B_pass": set(b_out) == new_files and all(x["pass"] for x in b_out.values()),
            "B_bitwise_all": all(x.get("bitwise_by_key") for x in b_out.values())}


def _git(*a, text=True):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=text, check=True).stdout


def _pilot_pins(src_new, names=PIN_NAMES):
    """2e (10-17): every pilot pin of the NEW scorer (by AST, never imported) is a set (commit, path, sha256) and its file at that commit has that
    sha256; else REFUSE. Returns the pins for the record."""
    import ast

    vals = {t.id: n.value for n in ast.parse(src_new).body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name) and t.id in names}
    out = {}
    for nm in names:
        v = ast.literal_eval(vals[nm]) if nm in vals else None
        if not (isinstance(v, tuple) and len(v) == 3 and all(isinstance(x, str) and x for x in v)):
            raise SystemExit(f"REFUSED (no Modal call made): pilot pin {nm} is not set in the NEW scorer (commit, path, sha256)")
        b = subprocess.run(["git", "-C", str(REPO), "show", f"{v[0]}:{v[1]}"], capture_output=True)
        if b.returncode != 0 or sha(b.stdout) != v[2]:
            raise SystemExit(f"REFUSED (no Modal call made): pilot pin {nm}: {v[1]} @ {v[0]} is missing or does not have the pinned sha256")
        out[nm] = {"commit": v[0], "path": v[1], "sha256": v[2]}
    return out


def _step2_gate():
    """2e (10-17, second review R1; step (2)): REFUSE before run() does anything (no scorer _launch is run, no Modal call) unless STEP2_DRIVER is set,
    names STEP2_DRIVER_PATH and its file at its commit has its sha256. The module-level import of run_famexp (own_all's runner, L130-132) has run by
    then: it only writes the sha-checked pinned BUILD from git show (no Modal call, no record parse). The first statement of run()."""
    v = STEP2_DRIVER
    if not (isinstance(v, tuple) and len(v) == 3 and all(isinstance(x, str) for x in v) and re.fullmatch(r"[0-9a-f]{40}", v[0]) and v[1] == STEP2_DRIVER_PATH
            and re.fullmatch(r"[0-9a-f]{64}", v[2])):
        raise SystemExit("REFUSED (no scorer _launch run, no Modal call made): STEP2_DRIVER is unset or is not (40-hex commit, STEP2_DRIVER_PATH, 64-hex sha256)")
    b = subprocess.run(["git", "-C", str(REPO), "show", f"{v[0]}:{v[1]}"], capture_output=True)
    if b.returncode != 0 or sha(b.stdout) != v[2]:
        raise SystemExit("REFUSED (no scorer _launch run, no Modal call made): STEP2_DRIVER's file at its commit does not have the pinned sha256")


def _pre(label, commit, run=False):
    """HEAD == --commit (full sha), tracked tree clean, HEAD == the scorer's origin branch, base an ancestor, the shared imports byte-identical;
    with run (the Modal step), the scorer's blob at HEAD == NEW_SCORER_BLOB and every pilot pin of the NEW scorer set and sha-verified at its commit.
    origin's branch tip is read with git ls-remote (507-prep, 10-17), not only from the local ref, which may be stale."""
    if label not in LABELS:
        raise SystemExit(f"--label one of {sorted(LABELS)}")
    c = LABELS[label]
    head = _git("rev-parse", "HEAD").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or head != commit:
        raise SystemExit(f"HEAD {head} is not --commit {commit} (a full sha)")
    br = c["branch"].removeprefix("origin/")
    if [ln.split("\t")[0] for ln in _git("ls-remote", "origin", f"refs/heads/{br}").splitlines()] != [head]:
        raise SystemExit(f"REFUSED (no Modal call made): origin's refs/heads/{br} (git ls-remote) is not HEAD {head}")
    if _git("status", "--porcelain", "--untracked-files=no").strip() or head != _git("rev-parse", c["branch"]).strip():
        raise SystemExit(f"uncommitted tracked changes, or HEAD is not {c['branch']}")
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", c["base"], head]).returncode != 0:
        raise SystemExit("the scorer's base commit is not an ancestor of HEAD")
    if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", c["base"], head, "--", *DEPS]).returncode != 0:
        raise SystemExit(f"STOP: {DEPS} differ between the base and HEAD; OLD and NEW would not share their imports")
    log = LOGS / label
    log.mkdir(parents=True, exist_ok=True)
    src = {"old": _git("show", f"{c['base']}:{SCORER}"), "new": (REPO / SCORER).read_text()}
    if src["new"] != _git("show", f"HEAD:{SCORER}"):
        raise SystemExit("the scorer file differs from HEAD's")
    blob = _git("rev-parse", f"HEAD:{SCORER}").strip() if run else None
    if run and not (isinstance(NEW_SCORER_BLOB, str) and re.fullmatch(r"[0-9a-f]{40}", NEW_SCORER_BLOB) and blob == NEW_SCORER_BLOB):
        raise SystemExit(f"REFUSED (no Modal call made): NEW_SCORER_BLOB is unset or is not the scorer's blob at HEAD ({blob})")
    return c, log, src, (_pilot_pins(src["new"]) if run else None), blob


def _prod(label, c):
    """(B) citations: INPUTS_FULL's S_old R / C overrides (pinned), or the probe's committed score record -> manifest sha."""
    if c["prod"] == "inputs_full":
        fb = _git("show", f"{INPUTS_FULL[0]}:{INPUTS_FULL[1]}", text=False)
        if sha(fb) != INPUTS_FULL[2]:
            raise SystemExit("INPUTS_FULL differs from its pin")
        out = []
        for rd in ("R", "C"):
            ov = json.loads(fb)["overrides"][f"S_old|{rd}"]
            ob = _git("show", f"{ov['commit']}:{ov['path']}", text=False)
            if sha(ob) != ov["sha256"]:
                raise SystemExit(f"the S_old {rd} override differs from INPUTS_FULL's pin")
            for k, xs in json.loads(ob)["simh"].items():
                if k.split("/", 1)[0] != c["sub"]:
                    continue
                for x in xs:
                    if x["folder"] in FOLDERS:
                        out.append({"folder": x["folder"], "dir": str(Path(x["path"]).parent), "file": Path(x["path"]).name, "sha256": x["sha256"],
                                    "manifest": Path(x["manifest"]["path"]).name, "manifest_sha256": x["manifest"]["sha256"],
                                    "cited_by": {"override": ov, "inputs_full": dict(zip(("commit", "path", "sha256"), INPUTS_FULL, strict=True)), "key": k}})
        return out
    rb = _git("show", f"HEAD:{c['orig']}", text=False)
    return [{"folder": fo, "dir": x["path"].removeprefix("/work/"), "manifest": "manifest_own_all.json", "manifest_sha256": x["manifest_sha256"],
             "cited_by": {"score_record": c["orig"], "record_sha256": sha(rb), "note": "not pinned in INPUTS_FULL 110447 (pooled A did not read this probe)"}}
            for r in json.loads(rb)["results"].values() for fo, x in r["folders"].items() if fo in FOLDERS and x["path"] == f"/work/cutfull/sold/{c['sub']}/{fo}"]


def run(label: str, commit: str, plans: str, plans_sha: str):
    _step2_gate()  # second review R1 / step (2): REFUSED until the driver pin is set, whatever NEW_SCORER_BLOB and the pilot pins say
    if modal.__version__ != MODAL_PIN:  # 2e (10-17): the client is the named environment's modal, the literal pin, else REFUSE before any Modal call
        raise SystemExit(f"REFUSED (no Modal call made): this interpreter's modal is not the literal pin {MODAL_PIN} (use the named environment's python)")
    if label not in OWN_LABELS or not PLANS_RE.fullmatch(plans) or not re.fullmatch(r"[0-9a-f]{64}", plans_sha):
        raise SystemExit(f"REFUSED (no Modal call made): --label one of {list(OWN_LABELS)}, plans a {PLANS_RE.pattern} path, plans_sha 64 hex")
    c, log, src, pins, blob = _pre(label, commit, run=True)  # REFUSES here, before any Modal call, unless HEAD == commit, the blob and every pilot pin verified
    pb = _git("show", f"HEAD:{plans}", text=False)  # step (2)'s plans, committed at HEAD: bytes only; parsed only in the container (_plans_jobs)
    if sha(pb) != plans_sha:
        raise SystemExit("REFUSED (no Modal call made): the committed plans differ from plans_sha (the wrapper's reported output sha256)")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    lg = (log / f"run_{stamp}.log").open("x")
    say = lambda **k: (lg.write(json.dumps(k) + "\n"), lg.flush(), print(json.dumps(k)))  # noqa: E731
    say(step="start", label=label, commit=commit, base=c["base"], scorer_blob=blob, untracked=len(_git("status", "--porcelain").splitlines()), pilot_pins=pins,
        plans={"path": plans, "sha256": plans_sha})
    kw_want = COMMON | LABELS[label]["kw"]  # the container checks the plans' kwargs for the label against these
    cb = _git("show", f"{COUNTED[0]}:{COUNTED[1]}", text=False)
    if sha(cb) != COUNTED[2]:
        raise SystemExit("the counted A25 key file differs from its pin")
    expected = sorted(f"simh_full|{x['task']}" for x in json.loads(cb)["keys"] if x.get("usable") is True and x.get("world") == "simh_full")
    prod = _prod(label, c)
    if len({p["folder"] for p in prod}) != len(FOLDERS):
        raise SystemExit("the production citations do not cover both S_old A25 folders")
    say(step="launch", app=APP, counted_runs=len(expected), production_citations=len(prod))
    arith = ARITH[c["base"][:9]]
    tol = ARITH_TOL[arith.split(":", 1)[0]]
    # the only Modal call, after every local check: the whole app lifecycle under one type-only boundary (Astra, 10-17); no modal.enable_output()
    # (507-prep, 10-17): no container log reaches the laptop; one json string back, plain Python only
    res, app_id = _app_lifecycle(app, regress, label, src, pb, plans_sha, kw_want, prod, expected, tol, stamp)
    if res["failed"]:  # a structured failure: only the exception TYPE and a count are shown; the raw text stays on the raw Volume (path + sha in the record)
        fail = {"label": f"STEP-2 regression {label}: the container FAILED; raw text on the raw Volume only", "stamp (PT)": stamp, "app": APP, "app_id": app_id,
                "commit": commit, "base": c["base"], "scorer_blob": blob, "failed": True, "stage": res.get("stage"),
                "error_type": res["error_type"], "errors": res["errors"], "body_error_type": res.get("body_error_type"), "raw": res["raw"],
                "remote_result_received": res.get("remote_result_received"), "cleanup_error_types": res.get("cleanup_error_types")}
        (HERE / "results").mkdir(exist_ok=True)
        for d_ in (HERE / "results", log):
            (d_ / f"step2_regress_{label}_{stamp}_FAILED.json").open("x").write(json.dumps(fail, indent=1))
        say(step="failed", stage=res.get("stage"), error_type=res["error_type"], errors=res["errors"])
        raise SystemExit(f"STOP to 09: the regression container failed ({res['error_type']}); its raw text is on the raw Volume (path and sha256 in the record)")
    rec = {"label": f"STEP-2 bitwise regression, {label}: (A) OLD @ {c['base'][:9]} vs NEW @ {commit[:9]} in one container; (B) NEW vs stored production; "
                    "keys, booleans, counts and %.3e maxima only", "stamp (PT)": stamp, "app": APP, "app_id": app_id, "commit": commit, "base": c["base"], "scorer_blob": blob,
           "pilot_pins": pins, "plans": {"path": plans, "sha256": plans_sha, "score_calls": res.get("plans", {}).get("score_calls")}, "production_citations": prod,
           "counted_key_file": dict(zip(("commit", "path", "sha256"), COUNTED, strict=True)), "B_arithmetic": arith, "B_bound": f"{tol:.0e} * max(1, |s|) on {SCORE_COLS}",
           "result": res,
           "pass": bool(res["A_pass"] and res["B_pass"])}
    js = json.dumps(rec, indent=1, default=sorted)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"step2_regress_{label}_{stamp}.json").open("x").write(js)
    (log / f"step2_regress_{label}_{stamp}.json").open("x").write(js)
    say(step="done", A_pass=res["A_pass"], B_pass=res["B_pass"], B_bitwise_all=res["B_bitwise_all"], machine_relay=res["machine_relay"], machine_sha256=res["machine_sha256"],
        A_files={fo: [x["pass"] for x in v["files"].values()].count(True) for fo, v in res["A_old_vs_new"].items()},
        B_max_abs_delta={k: v.get("max_abs_delta") for k, v in res["B_new_vs_production"].items()},
        B_units_differing={k: v.get("units_not_bitwise_equal") for k, v in res["B_new_vs_production"].items()},
        record=f"results/step2_regress_{label}_{stamp}.json")
    if not rec["pass"]:
        raise SystemExit(f"STOP to 09: (A) pass {res['A_pass']}, (B) pass {res['B_pass']}")


if __name__ == "__main__":
    if len(sys.argv) != 6 or sys.argv[1] != "run":
        raise SystemExit("usage: run <label> <full commit sha> <plans repo path> <plans sha256>")
    run(*sys.argv[2:])
