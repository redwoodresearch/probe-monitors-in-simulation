"""A48a regression (09 via 2e, 10-17), on S_old's two A25 folders (b1 145 runs, b2 4 runs; the written records 041701 / 041726 @ 7eaec9f55).

(A) OLD vs NEW, bitwise. The OLD p2 scorer (2313b6eb1) and the NEW one (the commit under test) score the same inputs in ONE Modal container and
    ONE Python process, so the CPU, numpy and OpenBLAS kernels cannot differ. Every file each version writes is compared bit for bit, and every
    npz row and key (all 149 runs). ANY difference, even one ulp, is a STOP to 09. (A) covers arm() (epochs, plan, scoring, files) only, not
    main(): main()'s launch checks are not re-run here; the spec is the written record's dump.
(B) NEW vs STORED production, by key. The NEW files are compared with the 16 stored p2 npz that pooled A read: INPUTS_FULL @ 927fabb0d ->
    overrides S_old|R and S_old|C @ bb33b808e -> each file's path and sha256. A file passes if it is bitwise equal, or within the bound of the
    ARITHMETIC precision (09, A48b): p2 computes score and logits in float32 (see COMPUTE), so the bound is 1e-5·max(1,|s|) although the files
    store float64. Within the bound but not bitwise is reported, not a STOP; the count beyond 1e-9·max(1,|s|) is reported as information.
    Beyond the bound, rows that do not align, or a changed non-score field is a STOP to 09. Fail-safe caveat: float32 cancellation over the
    5,120-term dot products could in principle exceed 1e-5·max(1,|s|) between kernels; that would be a false STOP, never a false pass.

Runs only at the final pin-fill commit: main() parses PILOT_PINS from the NEW blob (never imported) and STOPs before any Modal call unless every
field of every pin is set and each pin's bytes at its commit match its sha256.

The output holds booleans, counts and one max |Δ| per scored file (%.3e). No per-unit value appears in stdout, exceptions or records. Q2 (Astra
on af135ef1c): no free text leaves the container. Arm fails and exceptions are relayed only as counts and exception type names; their raw text
(fail strings, messages, tracebacks), the full host metadata (2e ruling (b)) and all fd 1/2 output of the body go into one raw file, written in
the container, exclusive-create, to p2/<stamp>/fails_raw_<input id>_<uuid4>.json on 2e's one step-2 raw-text Volume (RAWVOL, not the score-free
records Volume OUTVOL), relayed by volume, path and sha256 only (_arm_safe, regress). Of the host metadata only the OpenBLAS core type and the
modal and numpy versions are relayed, each only if it fullmatches its constant pattern, else "other" (_env_relay). regress() returns one
json.dumps string, so only plain Python crosses to the local client (it has no numpy).

How both versions load in one process: the container writes the two scorer sources (git blobs at the two commits) to
/tmp/a48_src/{old,new}/run_simh_scores.py. It imports them as two module objects (p2_old, p2_new) via importlib, with `modal` swapped for an
inert stub during those imports only, so their Modal decorators return plain functions (.local and .starmap run in-process).
run_score_linear.py and run_fibs_score.py are byte-identical at both commits (checked before launch); they are imported once, under the same
stub, and shared. Each version's own arm() runs on the same spec (the written record's dump) and the same payload (the pinned weights). Its
projector layer passes run in-process, one after the other. arm() writes to /cw, which in this container is a symlink to a fresh /tmp folder
(no Volume is mounted at /cw). The honest_b gate is not re-run: a stub returns pass, because this compares code, not the gate's numerics.

  run only through launchers/cutfull/regress_a48.sh <final commit> <name> (refuses unless HEAD == <final commit>, the tree is clean, HEAD is pushed)
"""

import ast
import copy
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

HERE = Path(__file__).resolve().parent
# 0d (10-17, R attempt 1 crash-looped at import): in a container this file is /root/regress_oldnew.py, which has one parent, so the
# repo paths fall back to HERE there (only main() and _write_rec use them, on the laptop), as run_simh_scores.py L28 does
P2 = HERE.parents[1] if len(HERE.parents) > 3 else HERE  # experiments/2026-10-14_p2_simh_scores
REPO = HERE.parents[3] if len(HERE.parents) > 3 else HERE
APP = "probe-shift_aswin-surya_sub-p2pilotreg_oldnew_kill-2026-11-14"
ACTS_VOL = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"  # S_old's full-length inputs, at /b (read-only), as in production
HF_VOL = "probe-shift_aswin-surya_hf-cache"  # the tokenizer vocab _plan reads, at /hf (read-only)
CWVOL = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"  # the stored production files, at /stored (read-only; never at /cw)
OUTVOL = "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"  # p2's score-free records Volume: this run's result (survives a sleeping laptop)
OLD = "2313b6eb11d9d8925ef0889a6c603712204bb601"  # the scorer before A48 (branch aswin/p2-simh-scores)
OLD_SHA = "addf75079c895cf3335fc55976e35973aa4556fb1ce88d004995e229724cca05"  # A48c: that scorer blob's sha256, pinned
SCORER = "experiments/2026-10-14_p2_simh_scores/run_simh_scores.py"
LAUNCHER = "experiments/2026-10-14_p2_simh_scores/launchers/cutfull/launch_cutfull.py"
PILOT_PIPE = "experiments/2026-10-14_p2_simh_scores/launchers/cutfull/pilot_pipe.py"  # the launcher's named binary and pilot pipe
UPLOAD_PY = "experiments/2026-10-14_p2_simh_scores/launchers/cutfull/stdin_to_rawvol.py"  # the pilot's stdin -> raw-text Volume uploader
NEW_BLOBS = {SCORER: "9bb11121135809b29d09a9dc15fb12edb7d6caf3", LAUNCHER: "7a0d1580aebdc623b0da77f1e969f49697fc6124", PILOT_PIPE: "b86e3cd855dc90f9cc81077528fd4b2e4d0e7915", UPLOAD_PY: "adce59f891ca473223b4bf421a24b321227ebe77"}  # 2e: EVERY file on the launch path, by git blob id at the pin-fill commit; None = refuse
LIBS = ("experiments/2026-10-14_p2_refusal_harm/run_score_linear.py", "experiments/2026-10-14_p2_fibs_pair/run_fibs_score.py")
ORIG = {  # A47b's two A25 writes: folder -> (commit, record, sha256)
    "/cutfull/sold/p2/simh_q4f_v1_full_b1_cutfull_v1__sim_honest": ("7eaec9f55", "experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-16_041701.json",
                                                                   "ed7d6507d247a537b1b5670948ce587d1bf305917fa6e71dfcfde292807db230"),
    "/cutfull/sold/p2/simh_q4f_v1_full_b2_cutfull_v1__sim_honest": ("7eaec9f55", "experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-16_041726.json",
                                                                   "8ab1ef90a38ac4d8c33d3416abb8096a0957ec01f141673c23efd77063164a70")}
INPUTS_FULL = ("927fabb0d", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-16_110447.json", "d6923a75aa8f914cd3b2e92d8403f125dd095219997aebea8b1d6e4e2b67f77c")
OVERRIDES = ("S_old|R", "S_old|C")  # the overrides holding the p2 files pooled A read for S_old
ROWF = ("key", "unit", "call_index", "score", "logits", "run_key", "token", "n_tokens", "original_source_sha256", "eval_sha256")  # the scorer's per-row fields
TOL = {"float64": 1e-9, "float32": 1e-5}  # 09: |Δ| <= tol · max(1, |stored|), tol by the ARITHMETIC precision (A48b), not the stored dtype
COMPUTE = {  # A48b (09): the dtype each field is COMPUTED in (both are stored as float64). Lines: the two libraries are identical at 2313b6eb1 and on
             # this branch; run_simh_scores.py lines are given as 2313b6eb1 / this branch
    "score": "float32",  # linear: run_score_linear.py L125-126 direction and bias float32, L140 X float32, L144 (X @ v + B[n]) in float32, then
                         # .astype(float64); run_simh_scores.py L352 / L373 stores it as float64. FIBS: the float64 sigmoid mean at run_simh_scores.py
                         # L369 / L390 of the float32-computed logits below, so its precision is float32's
    "logits": "float32"}  # run_fibs_score.py L126 W, b, scale float32, L137 X float32, L141 ((X / scale) @ W + b) in float32, then .astype(float64);
                          # run_simh_scores.py L363 / L384 stacks them

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0", "threadpoolctl==3.5.0")  # production's numpy and safetensors
o_vol = modal.Volume.from_name(OUTVOL, version=2)


VALUE_DIGEST_LOCAL = False  # 0d (ii), 10-17: False -> value_digest in a container (production); True -> in-process (synthetic tests only)


def _value_digest(kind: str, b: bytes, sha256: str, arg: str = ""):
    """0d (ii), 10-17: records that hold measured values are parsed HERE, never on the laptop; only paths, shas, layers, keys and counts
    return. The bytes must hash to sha256, else STOP. regress_oldnew.py carries this function byte-identically (check_a48_pilot.py).
    linear: a stage-1 fit record (AUROCs, cosines) -> directions[arg]'s weights file and layer. conv: the FIBS conversion record (Platt
    values) -> per layer its weights file. own_all_rs: A47b's score record -> per folder its R run counts, and each folder's original.
    manifest: a written p2 manifest (gate diffs) -> its dump, folders' joins and files, probes without role, verdict and slice."""
    if hashlib.sha256(b).hexdigest() != sha256:
        raise SystemExit(f"value_digest {kind}: the bytes differ from their sha256")
    d = json.loads(b)
    if kind == "linear":
        x = d["directions"][arg]
        return {"path": x["file"]["path"], "sha256": x["file"]["sha256"], "layer": x["layer"]}
    if kind == "conv":
        return {L: {"output": v["output"], "output_sha256": v["output_sha256"]} for L, v in d["layers"].items()}
    if kind == "own_all_rs":
        oa_f = {n_: x_ for v_ in d["results"].values() for n_, x_ in v_.get("folders", {}).items()}
        return {"folders": {n_: {z: c_[z] for z in ("runs", "runs_without_rows") if z in c_} for n_, x_ in oa_f.items()
                            for c_ in [x_.get("counts", {}).get("R", {})]},
                "original_of": {n_: o_ for v_ in d["messages"].values() for n_, o_ in (v_.get("original_of") or {}).items()}}
    if kind == "manifest":
        return {"dump": d["dump"], "folders": {od: {"joins": x.get("joins"), "files": x.get("files")} for od, x in d["folders"].items()},
                "probes": {k: {x: y for x, y in v.items() if x != "role"} for k, v in d["probes"].items()}, "verdict": d.get("verdict"),
                "slice": d["sizing"]["slice"]}
    raise SystemExit(f"value_digest: unknown kind {kind!r}")


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=600, retries=0, block_network=True)
def value_digest(reqs: list) -> list:
    """0d (ii): _value_digest per request [kind, bytes, sha256, arg], in this container (the scorer's copy, byte-identical; no network)."""
    return [_value_digest(*r) for r in reqs]


def _value_read(reqs: list) -> list:
    """The value digests of one batch: one value_digest call (production; main() parses none of these records), or in-process (tests)."""
    return [_value_digest(*r) for r in reqs] if VALUE_DIGEST_LOCAL else value_digest.remote(reqs)


def rows_neq(x, y):
    """Per row, True where the row's bytes differ (any dtype and trailing shape; exact bits, NaN payloads included); every row if dtype or shape differs.
    With zero rows a dtype or shape mismatch returns an empty array, so callers check structure first (_field_struct; R4)."""
    import numpy as np

    if x.dtype != y.dtype or x.shape != y.shape:
        return np.ones(max(len(x) if x.ndim else 1, len(y) if y.ndim else 1), dtype=bool)
    if x.ndim == 0:
        return np.array([x.tobytes() != y.tobytes()])
    if len(x) == 0:
        return np.zeros(0, dtype=bool)
    xb, yb = (np.ascontiguousarray(a).reshape(len(a), -1).view(np.uint8) for a in (x, y))
    return (xb != yb).any(axis=1)


def _row_struct_ok(m: dict) -> bool:
    """R4 (Astra class, via CUT): the file has a 1-d key column and every row field's leading dimension is len(key). Checked before any row is
    indexed, whatever the row count."""
    k = m.get("key")
    return k is not None and k.ndim == 1 and all(m[f].ndim >= 1 and m[f].shape[0] == k.shape[0] for f in m if f in ROWF)


def _field_struct(a: dict, b: dict) -> dict:
    """R4: per field of a, True if b has it with the same dtype and the same full shape (zero rows included); separate from, and before, the bits."""
    return {f: f in b and a[f].dtype == b[f].dtype and a[f].shape == b[f].shape for f in a}


def _nrows(m: dict) -> int:
    return int(m["key"].shape[0]) if "key" in m and m["key"].ndim == 1 else 0


def _key_rows(m: dict) -> tuple:
    """A file's rows grouped by key with a stable sort (row order kept within each key): (row order, {key: (start, stop) in that order})."""
    import numpy as np

    o = np.argsort(m["key"], kind="stable")
    k, st, c = np.unique(m["key"][o], return_index=True, return_counts=True)
    return o, {x: (s, s + n) for x, s, n in zip(k.tolist(), st.tolist(), c.tolist())}


def keys_not_identical(a: dict, b: dict, rf: list) -> int:
    """(A) R3 (Astra, d26947c96): the number of keys whose rows differ. A key fails if it is on one side only, has another row count, or (a
    common key with the same count) any row field's bits differ in its rows, in row order. A row field whose length is not the key column's
    makes every key fail."""
    oa, ga = _key_rows(a)
    ob, gb = _key_rows(b)
    ks = set(ga) | set(gb)
    if any(m[f].ndim == 0 or len(m[f]) != len(m["key"]) for m in (a, b) for f in rf):
        return len(ks)
    return sum(k not in ga or k not in gb or ga[k][1] - ga[k][0] != gb[k][1] - gb[k][0]
               or any(rows_neq(a[f][oa[ga[k][0]:ga[k][1]]], b[f][ob[gb[k][0]:gb[k][1]]]).any() for f in rf) for k in ks)


def compare_npz(a_b: bytes, b_b: bytes) -> dict:
    """(A) one scored file, OLD vs NEW: fields, per-field bits, rows, and keys (each key's rows, in order, in every row field). Booleans and counts only."""
    import numpy as np

    a, b = (dict(np.load(io.BytesIO(x)).items()) for x in (a_b, b_b))  # loaded once (an NpzFile re-reads an array on every access)
    fa, fb = sorted(a), sorted(b)
    rf = [f for f in fa if f in ROWF and f in fb]
    st = _field_struct(a, b)  # R4: structure first; a field's bits are compared only if its dtype and full shape match
    r = {"fields_equal": fa == fb, "field_struct_equal": st, "row_fields_match_key_length": _row_struct_ok(a) and _row_struct_ok(b),
         "rows": _nrows(a), "rows_new": _nrows(b)}
    if not r["row_fields_match_key_length"]:  # R4: a structural mismatch FAILs with counts, before any bits are compared or any row indexed
        return r | {"bits_compared": False, "rows_not_identical": max(r["rows"], r["rows_new"]), "pass": False}
    r |= {"bits_compared": True, "field_bits_equal": {f: st[f] and not rows_neq(a[f], b[f]).any() for f in fa}}  # structure before bits
    neq = np.zeros(r["rows"], dtype=bool)
    if a["key"].shape == b["key"].shape:  # every row field is len(key) long on both sides (checked above)
        for f in rf:
            neq |= rows_neq(a[f], b[f]) if st[f] else True
    r["rows_not_identical"] = int(neq.sum()) if a["key"].shape == b["key"].shape else max(r["rows"], r["rows_new"])
    ka, kb = np.unique(a["key"]), np.unique(b["key"])
    bad = keys_not_identical(a, b, rf)
    r |= {"keys": int(len(ka)), "keys_new": int(len(kb)), "key_set_equal": set(ka.tolist()) == set(kb.tolist()), "keys_not_identical": bad, "key_list": sorted(ka.tolist())}
    r["pass"] = r["fields_equal"] and all(r["field_bits_equal"].values()) and r["rows"] == r["rows_new"] and r["rows_not_identical"] == 0 and r["key_set_equal"] and bad == 0
    return r


def bound_compare(new_b: bytes, stored_b: bytes) -> dict:
    """(B) one scored file, NEW vs STORED: rows must align (key and unit, bitwise) and every non-score field must be bitwise equal; score and
    logits are bitwise, or within the bound of their compute precision (COMPUTE; 09, A48b). Counts and one max |Δ| per field (%.3e) only.
    R2 (Astra, d26947c96): an element is within the bound only if the STORED value and |Δ| are both finite and |Δ| <= the bound, or if NEW and
    STORED are the same bits. So a non-finite element passes only bit-identical (inf = inf of the same sign; NaN = NaN only with the same bits,
    as (A)'s rows_neq compares exact bits, NaN payloads included), and any other non-finite mismatch is beyond the bound: a STOP. The bit rule
    can STOP on two NaNs with different payloads; that is a fail-safe false STOP, never a false pass."""
    import numpy as np

    n, s = (dict(np.load(io.BytesIO(x)).items()) for x in (new_b, stored_b))
    st = _field_struct(s, n)  # R4: per STORED field, NEW has it with the same dtype and full shape (zero rows included)
    r = {"bitwise": new_b == stored_b, "fields_equal": sorted(n) == sorted(s), "field_struct_equal": st,
         "row_fields_match_key_length": _row_struct_ok(n) and _row_struct_ok(s)}  # R4: checked before any row is indexed
    al = r["fields_equal"] and r["row_fields_match_key_length"] and st["key"] and not rows_neq(n["key"], s["key"]).any() \
        and ("unit" not in s or (st["unit"] and not rows_neq(n["unit"], s["unit"]).any()))
    r["rows_aligned"] = bool(al)
    r["other_fields_bitwise"] = bool(al) and all(st[f] and not rows_neq(n[f], s[f]).any() for f in s if f not in ("score", "logits"))
    for f in ("score", "logits"):
        if f not in s:
            continue
        a, b = n.get(f), s[f]
        if not al or a is None or not st[f] or b.dtype.name not in TOL:
            r[f] = {"comparable": False}
            continue
        with np.errstate(invalid="ignore", over="ignore"):  # inf - inf and the like: handled below, not warned
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            same = np.ascontiguousarray(a).view(f"u{a.itemsize}") == np.ascontiguousarray(b).view(f"u{b.itemsize}")  # per element, exact bits
            fin_ok = np.isfinite(b) & np.isfinite(d)
            ok = {k: same | (fin_ok & (d <= k * np.maximum(1.0, np.abs(b.astype(np.float64))))) for k in (TOL[COMPUTE[f]], TOL["float64"])}  # R2
        per_unit = lambda m: m.any(axis=tuple(range(1, m.ndim))) if m.ndim > 1 else m  # noqa: E731  (R4: reshape(0, -1) raises on zero rows)
        fin = d[np.isfinite(d)]
        r[f] = {"comparable": True, "stored_dtype": b.dtype.name, "compute_dtype": COMPUTE[f], "tol": TOL[COMPUTE[f]], "units": int(len(b)),
                "units_differing": int(rows_neq(a, b).sum()), "units_beyond_bound": int(per_unit(~ok[TOL[COMPUTE[f]]]).sum()),
                "max_abs_delta": f"{(float(fin.max()) if fin.size else 0.0):.3e}", "nonfinite_deltas": int((~np.isfinite(d)).sum()),
                "units_beyond_float64_bound (information)": int(per_unit(~ok[TOL["float64"]]).sum())}
    fs = [r[f] for f in ("score", "logits") if f in r]
    r["pass"] = r["row_fields_match_key_length"] and (r["bitwise"] or (r["other_fields_bitwise"] and bool(fs) and all(x["comparable"] and x["units_beyond_bound"] == 0 for x in fs)))
    r["status"] = "STOP to 09" if not r["pass"] else "bitwise" if r["bitwise"] else "within the float32 compute bound (reported, not a STOP)"
    return r


def _stub():
    """An inert `modal` for importing the scorers in-process: decorators return the function itself (.local / .starmap call it here)."""
    class _Any:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, n):
            return lambda *a, **k: self

    def function(**k):
        def deco(f):
            f.local = f
            f.starmap = lambda jobs: [f(*j) for j in jobs]
            return f
        return deco

    gate = types.SimpleNamespace(from_id=lambda _id: types.SimpleNamespace(get=lambda: {"pass": True}))  # the honest_b gate is not re-run here
    return types.SimpleNamespace(App=lambda *a, **k: types.SimpleNamespace(function=function, local_entrypoint=lambda *a, **k: (lambda f: f)),
                                 Image=_Any(), Volume=types.SimpleNamespace(from_name=lambda *a, **k: _Any()), FunctionCall=gate)


def _load_versions(src: dict, base: Path = Path("/tmp/a48_src")) -> dict:
    import importlib
    import importlib.util

    (base / "lib").mkdir(parents=True)
    for name, b in src["libs"].items():
        (base / "lib" / name).write_bytes(b)
    sys.path.insert(0, str(base / "lib"))
    real, mods = sys.modules.get("modal"), {}
    sys.modules["modal"] = _stub()
    try:
        for lib in ("run_score_linear", "run_fibs_score"):  # shared by both versions (byte-identical at both commits)
            importlib.import_module(lib)
        for ver in ("old", "new"):
            p = base / ver / "run_simh_scores.py"
            p.parent.mkdir(parents=True)
            p.write_bytes(src[ver])
            sp = importlib.util.spec_from_file_location(f"p2_{ver}", p)
            m = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(m)
            mods[ver] = m
    finally:
        sys.modules["modal"] = real
    return mods


def _host() -> dict:
    import numpy as np
    from threadpoolctl import threadpool_info

    lines = Path("/proc/cpuinfo").read_text().splitlines() if Path("/proc/cpuinfo").exists() else []
    field = lambda n: next((x.split(":", 1)[1].strip() for x in lines if x.split(":", 1)[0].strip() == n), None)  # noqa: E731
    from numpy._core._multiarray_umath import __cpu_features__ as feats

    return {"cpu_model": field("model name"), "cpu_vendor": field("vendor_id"), "cpu_count": os.cpu_count(), "numpy": np.__version__,
            "numpy_simd": sorted(k for k, v in feats.items() if v and k.startswith(("AVX", "FMA"))),
            "blas": [{k: d.get(k) for k in ("internal_api", "version", "architecture", "num_threads", "threading_layer")} for d in threadpool_info()]}  # architecture = the OpenBLAS core type


def _git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()


def _gshow(c: str, p: str) -> bytes:
    return subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=HERE).stdout


def _blob_problems(head: str, git=None) -> list:
    """[] only if every NEW_BLOBS pin is set and equals HEAD's git blob id for that path; else the problems (paths only)."""
    git = git or _git
    return [f"{p}: blob pin unset" if not want else f"{p}: blob at HEAD differs from its pin" for p, want in NEW_BLOBS.items()
            if not want or git("rev-parse", f"{head}:{p}") != want]


BRANCH_REF = "refs/heads/aswin/step2-pilot-p2"  # the ONE origin ref the regression may run from (fixed, so a detached worktree at its tip passes)


def _remote_problems(head: str, git) -> list:
    """[] only if origin's BRANCH_REF tip (git ls-remote, the only origin compare; no local remote-tracking ref is read) is exactly HEAD.
    git is required (main() passes _git), so no caller reaches the network by omission."""
    return [] if git("ls-remote", "origin", BRANCH_REF).split()[:1] == [head] else [f"origin's {BRANCH_REF} is not HEAD (git ls-remote)"]


def _pin_problems(new_src: bytes, gshow=None) -> list:
    """A48c: [] only if the NEW scorer's PILOT_PINS (parsed, never imported) has every field of every pin set and each pin's bytes at its commit
    match its sha256; else the problems (pin names only)."""
    gshow = gshow or _gshow
    pins = next((ast.literal_eval(n.value) for n in ast.parse(new_src).body if isinstance(n, ast.Assign)
                 for t in n.targets if isinstance(t, ast.Name) and t.id == "PILOT_PINS"), None)
    if not isinstance(pins, dict) or not pins:
        return ["PILOT_PINS not found in the NEW scorer"]
    out = []
    for k, v in pins.items():
        if not (isinstance(v, tuple) and len(v) == 3 and all(v)):
            out.append(f"{k}: unset")
            continue
        try:
            b = gshow(v[0], v[1])
        except subprocess.CalledProcessError:
            out.append(f"{k}: not at its commit")
            continue
        if hashlib.sha256(b).hexdigest() != v[2]:
            out.append(f"{k}: sha256 differs at its commit")
    return out


def _arm_safe(m, spec: dict, payload: dict, root: Path, raw: dict, where: str) -> tuple:
    """One version's arm() with CW -> root. Returns (result, None), or (None, fail). Q2 (Astra on af135ef1c, via CUT): fail is structured only
    ({n_fails, exception_type}); the arm's fail strings, or the exception's message and traceback, go ONLY to raw[where] (fails_raw.json)."""
    os.symlink(root, CW)
    try:
        r = m.arm(copy.deepcopy(spec), copy.deepcopy(payload), "a48-harness")
    except (Exception, SystemExit) as e:
        raw[where] = {"n_fails": 1, "exception_type": type(e).__name__} | _exc_raw(e)
        return None, {"n_fails": 1, "exception_type": type(e).__name__}
    finally:
        os.unlink(CW)
    if r.get("fails"):
        raw[where] = {"n_fails": len(r["fails"]), "exception_type": None, "fails": [_text(lambda: str(x)) for x in r["fails"]]}
        return None, {"n_fails": len(r["fails"]), "exception_type": None}
    return r, None


def _text(f) -> str:
    """Raw text for the raw file only; never raises (an odd __str__ cannot escape the type-only boundary)."""
    try:
        return f()
    except Exception as e:
        return f"<unprintable: {type(e).__name__}>"


def _exc_raw(e: BaseException) -> dict:
    """An exception's type, message and traceback, for the raw file only (call inside its except block). Never raises."""
    return {"type": type(e).__name__, "message": _text(lambda: str(e)), "traceback": _text(traceback.format_exc)}


# in-container paths (module constants so the synthetic test can point them at a temp folder)
CW, SRC_DIR, ARM_OUT, STORED, OUT_ROOT, RAW_ROOT = "/cw", Path("/tmp/a48_src"), Path("/tmp/a48_out"), Path("/stored"), Path("/o"), Path("/raw")
# 2e's ONE step-2 raw-text Volume for all families (create_if_missing, version 2; mounted only in regress): fails_raw.json goes here under
# p2/<run id>/, exclusive-create, NEVER on OUTVOL. The data Volumes (/b, /hf, /stored) stay read-only.
RAWVOL = "probe-shift_aswin-surya_sub-step2pilot_rawtext_kill-2026-11-14"
raw_vol = modal.Volume.from_name(RAWVOL, create_if_missing=True, version=2)
OBLAS_CORES = r"^(Haswell|SkylakeX|Zen|Cooperlake|SapphireRapids|Prescott|Nehalem|Sandybridge|Skylake|Atom)$"  # 2e ruling (b): the only relayed core types
SEMVER = r"^\d+\.\d+\.\d+$"


def _env_relay(host: dict, modal_version) -> dict:
    """2e ruling (b) on host metadata: the ONE exception to "no host field relayed". The OpenBLAS core type, and the modal and numpy version
    strings, each relayed only if it fullmatches its constant pattern, else the literal "other". Everything else about the host stays raw."""
    cores = {b.get("architecture") for b in host.get("blas", []) if b.get("internal_api") == "openblas"}
    core = next(iter(cores)) if len(cores) == 1 else None
    ok = lambda v, pat: v if isinstance(v, str) and re.fullmatch(pat, v) else "other"  # noqa: E731
    return {"openblas_core": ok(core, OBLAS_CORES), "modal_version": ok(modal_version, SEMVER), "numpy_version": ok(host.get("numpy"), SEMVER)}


RG_KW = dict(image=image, cpu=(2, 2), memory=(32768, 32768), timeout=3900, retries=0,  # A48c: 3,900 s x ~$0.35/h (2 CPU, 32 GiB) ~= $0.38, the guard cap
             volumes={"/b": modal.Volume.from_name(ACTS_VOL, version=2).read_only(), "/hf": modal.Volume.from_name(HF_VOL).read_only(),
                      "/stored": modal.Volume.from_name(CWVOL).read_only(), "/o": o_vol, "/raw": raw_vol})  # regress and regress_ptr: one container shape


@app.function(**RG_KW)
def regress(src: dict, specs: dict, payload: dict, stored: dict, joins: dict, stamp: str) -> str:
    """The container's work (_regress). Q2 (Astra on af135ef1c): NO free text leaves the container. The result relays only structured
    fields: per arm the fail count and exception type, the harness exception's type, their counts, env (_env_relay's three allowlisted fields)
    and the volume, path and sha256 of the raw file. The raw file holds the fail strings, exception messages and tracebacks, the FULL host
    metadata before and after (2e ruling (b)), and everything written to fds 1 and 2 (and native C stdio) up to the raw write. fds 1 and 2 stay
    on the capture through ALL finalization and are restored in the outermost cleanup (_run_and_write). The raw file goes to RAWVOL under
    p2/<run id>/ with a per-attempt token in its name (Modal input id + uuid4), exclusive-create, never to OUTVOL (score-free records), never
    printed or relayed. Returns ONE json.dumps string: plain Python only, because the local modal client has no numpy (json.dumps fails here,
    loudly, on any numpy value).
    The type-only boundary (Astra's encode-error class, via CUT): the body's exceptions are caught in _finish; anything raised after that,
    in ALL finalization (serialization, the raw and records writes, both Volume commits, capture restore), leaves only as a new RuntimeError
    naming the original exception's TYPE, raised outside the except block so it carries no context, message or traceback of the original.
    The raw file is serialized with ensure_ascii, so a lone surrogate is escaped (\\udXXX) instead of raising UnicodeEncodeError."""
    failed = None
    try:
        return _run_and_write(src, specs, payload, stored, joins, stamp)
    except (Exception, SystemExit) as e:
        failed = type(e).__name__
    raise RuntimeError(f"A48 regression harness: finalization failed with {failed} (raw text withheld)")


def _libc_fflush() -> None:
    """fflush(NULL): push every C stdio buffer (a native library's printf) into the CURRENT fds 1 and 2. Raises OSError if it fails."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.fflush(None) != 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))


def _set_std(pair: tuple) -> None:
    sys.stdout, sys.stderr = pair


def _run_steps(steps: list):
    """Runs every step, each in its own try, so one failure never skips the rest; returns the first exception (or None)."""
    first = None
    for s in steps:
        try:
            s()
        except BaseException as e:  # noqa: BLE001  (restoration must reach every step; the first error is re-raised by the caller)
            first = first or e
    return first


def _run_and_write(src: dict, specs: dict, payload: dict, stored: dict, joins: dict, stamp: str) -> str:
    """Astra (af135ef1c..21d6b4c9a) R4/R5 and CUT (a): fds 1 and 2 and sys.stdout/stderr point at a capture file for the WHOLE body AND all
    finalization (serialization, the raw and records writes, both Volume commits), so a noisy finalizer leaks nothing. The restore is the
    outermost cleanup: our wrapper, the original Python streams and C stdio are flushed while fds 1 and 2 still point at the capture (R4;
    _libc_fflush checks fflush's RETURN value), and every step runs in its own try (R5), so a failing flush never leaves the streams on a
    closed file. If any flush fails, fds 1 and 2 go to /dev/null instead of the real ones (fail closed). The first error reaches regress()
    (type only)."""
    sys.stdout.flush()
    sys.stderr.flush()
    cap = tempfile.TemporaryFile()
    saved = (os.dup(1), os.dup(2))
    cap_txt = open(cap.fileno(), "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)
    py_saved = (sys.stdout, sys.stderr)
    failure, body, state = None, None, {"flush_failed": False}
    try:
        try:
            os.dup2(cap.fileno(), 1)  # fd-level: anything the body, a finalizer or a library writes to fd 1 or 2 lands in the capture, not the log
            os.dup2(cap.fileno(), 2)
            sys.stdout = sys.stderr = cap_txt  # and Python-level, in case the runtime's sys.stdout does not write to fd 1
            body = _finish(src, specs, payload, stored, joins, stamp, cap, cap_txt, state)
        except (Exception, SystemExit) as e:
            failure = e
    finally:
        # 1. every pending buffer (our wrapper, the ORIGINAL Python streams, C stdio) goes into the capture while fds 1 and 2 still point at it
        flush_err = _run_steps([cap_txt.flush, py_saved[0].flush, py_saved[1].flush, lambda: sys.__stdout__ and sys.__stdout__.flush(),
                                lambda: sys.__stderr__ and sys.__stderr__.flush(), _libc_fflush])
        # 2. fail closed (Astra via CUT): only if every flush succeeded do fds 1 and 2 go back to the real ones; otherwise they point at
        #    /dev/null, so whatever is still pending (a native printf without a newline, sys.__stdout__) can never reach the log. The capture
        #    is never drained to the real fds. If /dev/null cannot be opened either, fds 1 and 2 stay on the capture.
        try:  # sticky (Astra astra_p2pilot5, optional): a flush that failed EARLIER in _finish also counts, even if this retry succeeded
            dest = saved if flush_err is None and not state["flush_failed"] else (os.open(os.devnull, os.O_WRONLY),) * 2
        except OSError:
            dest = None
        restore_err = _run_steps([lambda: _set_std(py_saved)] + ([lambda: os.dup2(dest[0], 1), lambda: os.dup2(dest[1], 2)] if dest else [])
                                 + [lambda: os.close(saved[0]), lambda: os.close(saved[1])]
                                 + ([lambda: os.close(dest[0])] if dest and dest is not saved else []) + [cap_txt.close, cap.close])
        restore_err = flush_err or restore_err
    if failure is not None:
        raise failure
    if restore_err is not None:
        raise restore_err
    return body


def _finish(src: dict, specs: dict, payload: dict, stored: dict, joins: dict, stamp: str, cap, cap_txt, state: dict) -> str:
    """The body and all finalization, inside the capture. The raw file's stdio is what the capture holds when it is written; anything a later
    finalizer prints stays in the capture, which is discarded (never relayed)."""
    raw = {"arm_fails": {}, "harness_exception": None}
    try:
        res = _regress(src, specs, payload, stored, joins, raw["arm_fails"])
    except (Exception, SystemExit) as e:
        raw["harness_exception"] = _exc_raw(e)
        res = {"host": {}, "host_after": {}, "slice": {}, "A": {}, "B": {}, "harness_exception": {"type": type(e).__name__},
               "arm_fails": {w: {"n_fails": v["n_fails"], "exception_type": v["exception_type"]} for w, v in raw["arm_fails"].items()}}
    try:
        cap_txt.flush()
        _libc_fflush()  # R4: native buffers into the capture before it is read
    except BaseException:
        state["flush_failed"] = True  # sticky: the cleanup then sends fds 1 and 2 to /dev/null even if its own retry succeeds
        raise
    raw["stdio"] = os.pread(cap.fileno(), os.fstat(cap.fileno()).st_size, 0).decode("utf-8", "replace")  # pread: fds 1/2 keep their offset
    mv = getattr(modal, "__version__", None)
    raw["env"] = {"host": res.pop("host"), "host_after": res.pop("host_after"), "modal_version": mv}  # 2e ruling (b): full host metadata raw only
    res["env"] = _env_relay(raw["env"]["host"], mv)
    res["n_arm_fails"] = sum(v["n_fails"] for v in res["arm_fails"].values())
    res["n_harness_exceptions"] = int(raw["harness_exception"] is not None)
    iid = getattr(modal, "current_input_id", lambda: None)()
    token = f"{iid if isinstance(iid, str) and re.fullmatch(r'[A-Za-z0-9_-]+', iid) else 'noinput'}_{uuid.uuid4().hex}"  # per attempt
    raw_dir = RAW_ROOT / "p2" / stamp  # p2's own subdir, one folder per run id (mkdir exist_ok for the parent only)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rb = json.dumps(raw, indent=1, ensure_ascii=True).encode("ascii")  # a lone surrogate is escaped, never an encode error
    with open(raw_dir / f"fails_raw_{token}.json", "xb") as fh:  # exclusive-create: an existing name is refused, never overwritten
        fh.write(rb)
    raw_vol.commit()
    res["fails_raw"] = {"volume": RAWVOL, "path": str((raw_dir / f"fails_raw_{token}.json").relative_to(RAW_ROOT)), "sha256": hashlib.sha256(rb).hexdigest()}
    out_dir = OUT_ROOT / "2026-10-14_p2_simh_scores" / f"a48_regress_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    body = json.dumps(res, indent=1, ensure_ascii=True)
    with open(out_dir / "result.json", "x", encoding="ascii") as fh:
        fh.write(body)
    o_vol.commit()
    return body


def _regress(src: dict, specs: dict, payload: dict, stored: dict, joins: dict, raw_fails: dict) -> dict:
    if os.path.lexists(CW):
        raise SystemExit("/cw exists in this container: refusing (arm writes there; it must be a fresh local folder)")
    host0, mods = _host(), _load_versions(src, SRC_DIR)
    res = {"host": host0, "slice": {v: m.SLICE for v, m in mods.items()}, "A": {}, "B": {}, "arm_fails": {}}
    new_files = {}
    for od, spec in specs.items():
        out = {}
        for ver, m in mods.items():  # OLD first, then NEW: same process, same container
            root = ARM_OUT / ver / od.strip("/").replace("/", "__")
            root.mkdir(parents=True)
            r, fail = _arm_safe(m, spec, payload, root, raw_fails, f"{ver} {od}")
            if fail:
                res["arm_fails"][f"{ver} {od}"] = fail
                continue
            d_ = root / od.strip("/")
            out[ver] = {p.name: p.read_bytes() for p in sorted(d_.iterdir())}
            res.setdefault("arm_hashes_match_bytes", {})[f"{ver} {od}"] = r["folders"][od]["files"] == {n: hashlib.sha256(b).hexdigest() for n, b in out[ver].items()}
            shutil.rmtree(root)
        if set(out) != {"old", "new"}:
            continue
        o, n = out["old"], out["new"]
        files = {}
        for name in sorted(set(o) | set(n)):
            f = {"in_both": name in o and name in n, "bytes_equal": o.get(name) == n.get(name)}
            if name.endswith(".npz") and f["in_both"]:
                f |= compare_npz(o[name], n[name])
            files[name] = f
        ks = set().union(*[set(f.get("key_list", [])) for f in files.values()])
        res["A"][od] = {"file_names_equal": sorted(o) == sorted(n), "files": {k: {x: y for x, y in v.items() if x != "key_list"} for k, v in files.items()},
                        "npz_keys_union_equals_written_joins": ks == set(joins[od]), "runs": len(joins[od])}
        new_files[od] = n
    for od, pins in stored.items():  # (B) the 16 stored production npz
        for name, (path, want) in sorted(pins.items()):
            sb = (STORED / path).read_bytes()
            f = {"stored_path": path, "stored_sha256_is_pin": hashlib.sha256(sb).hexdigest() == want, "new_present": name in new_files.get(od, {})}
            if f["new_present"]:
                f |= bound_compare(new_files[od][name], sb)
            res["B"][f"{od}/{name}"] = f
    res["host_after"] = _host()
    return res


def _pointer(body: str, stamp: str) -> str:
    """0d (ii), 10-17: in the container, after regress's body is on OUTVOL (_finish writes result.json): the (A) and (B) verdicts, as main()
    computed them before (ii), with counts, type names and pointers only. No per-file value leaves; the full result stays on the Volume,
    by path and sha256 (the body is ASCII, so its sha256 is the file's)."""
    res = json.loads(body)
    a_ok = not res.get("harness_exception") and not res["arm_fails"] and all(res.get("arm_hashes_match_bytes", {}).values()) and set(res["A"]) == set(ORIG) and all(
        v["file_names_equal"] and v["npz_keys_union_equals_written_joins"] and all(f["in_both"] and f["bytes_equal"] and f.get("pass", True) for f in v["files"].values())
        for v in res["A"].values())
    b_ok = len(res["B"]) == 16 and all(f["stored_sha256_is_pin"] and f["new_present"] and f.get("pass") for f in res["B"].values())
    verdict = "PASS" if a_ok and b_ok else "STOP to 09: " + ", ".join(x for x, ok in (("(A) OLD vs NEW differ", a_ok), ("(B) beyond the bound", b_ok),
                                                                               ("a harness exception (type only; raw text in fails_raw.json)", not res.get("harness_exception"))) if not ok)
    return json.dumps({"A_pass": a_ok, "B_pass": b_ok, "verdict": verdict, "arm_fails": res["arm_fails"], "n_arm_fails": res["n_arm_fails"],
                       "n_harness_exceptions": res["n_harness_exceptions"], "harness_exception_type": (res.get("harness_exception") or {}).get("type"),
                       "arm_hashes_match_bytes": all(res.get("arm_hashes_match_bytes", {}).values()), "env": res["env"], "fails_raw": res["fails_raw"],
                       "result": {"volume": OUTVOL, "path": f"2026-10-14_p2_simh_scores/a48_regress_{stamp}/result.json",
                                  "sha256": hashlib.sha256(body.encode("ascii")).hexdigest()}})


@app.function(**RG_KW)
def regress_ptr(src: dict, specs: dict, payload: dict, stored: dict, joins: dict, stamp: str) -> str:
    """0d (ii): regress's body in this same container (regress.local), then only _pointer's verdict and pointers return to main()."""
    return _pointer(regress.local(src, specs, payload, stored, joins, stamp), stamp)


def _write_rec(rec: dict, stamp: str, log_dir: str = "", res_dir: Path = None) -> Path:
    """The local record, results/a48_regress_<stamp>.json (and a copy in log_dir), each written once ("x")."""
    body = json.dumps(rec, indent=1, ensure_ascii=True)
    out = (res_dir or P2 / "results") / f"a48_regress_{stamp}.json"
    for p in [out] + ([Path(log_dir) / out.name] if log_dir else []):
        with open(p, "x", encoding="ascii") as fh:
            fh.write(body)
    return out


@app.local_entrypoint()
def main(commit: str, log_dir: str = ""):
    git, gshow, sha = _git, _gshow, (lambda b: hashlib.sha256(b).hexdigest())
    head = git("rev-parse", "HEAD")
    probs = _pin_problems(gshow(head, SCORER))  # A48c: first, before anything else and before any Modal call
    if probs:
        raise SystemExit(f"STOP: A48 pins are not all set and verified at HEAD (run only at the final pin-fill commit): {probs}")
    bprobs = _blob_problems(head)  # the NEW scorer and launcher, pinned by git blob id (filled at pin-fill; None = refuse)
    if bprobs:
        raise SystemExit(f"STOP: NEW_BLOBS: {bprobs}")
    if head != git("rev-parse", "--verify", f"{commit}^{{commit}}") or git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit(f"STOP: HEAD must be {commit} and the tree clean (tracked files)")
    rprobs = _remote_problems(head, git)  # pushed = origin's BRANCH_REF tip is exactly HEAD (git ls-remote): the only origin compare
    if rprobs:
        raise SystemExit(f"STOP: {rprobs}")
    blob_ids = {"new": {p: git("rev-parse", f"{head}:{p}") for p in NEW_BLOBS}, "old_scorer": git("rev-parse", f"{OLD}:{SCORER}")}
    src = {"old": gshow(OLD, SCORER), "new": gshow(head, SCORER), "libs": {Path(p).name: gshow(head, p) for p in LIBS}}
    if sha(src["old"]) != OLD_SHA:
        raise SystemExit("STOP: the OLD scorer blob's sha256 differs from its pin")
    if any(gshow(OLD, p) != src["libs"][Path(p).name] for p in LIBS):
        raise SystemExit("STOP: run_score_linear.py or run_fibs_score.py differs between the two commits (one shared copy is not valid)")
    consts = {v: {t.id: ast.literal_eval(n.value) for n in ast.parse(src[v]).body if isinstance(n, ast.Assign) for t in n.targets
                  if isinstance(t, ast.Name) and t.id in ("LINEAR", "CONV")} for v in ("old", "new")}
    if consts["old"] != consts["new"]:
        raise SystemExit("STOP: the two scorers pin different weights (LINEAR / CONV)")
    payload, dirs = {"linear": {}, "fibs": []}, {}  # exactly main()'s payload, from the pinned stage-1 and conversion records
    lin, cvp, ob_ = consts["new"]["LINEAR"], consts["new"]["CONV"], {}
    for od, (c, p, s) in ORIG.items():
        ob_[od] = gshow(c, p)
        if sha(ob_[od]) != s:
            raise SystemExit(f"STOP: {p} @ {c}: sha256 differs from its pin")
    lb, cb = {p: (REPO / rp).read_bytes() for p, rp in lin.items()}, (REPO / cvp).read_bytes()
    dg = _value_read([["linear", lb[p], sha(lb[p]), p] for p in lin] + [["conv", cb, sha(cb), ""]]  # 0d (ii): ONE value_digest call;
                     + [["manifest", ob_[od], ORIG[od][2], ""] for od in ORIG])  # none of these records is parsed here
    dl, cv, mf = dg[:len(lin)], dg[len(lin)], dict(zip(ORIG, dg[len(lin) + 1:], strict=True))
    for (p, rp), d in zip(lin.items(), dl, strict=True):
        wb = (REPO / rp).parent.joinpath(d["path"]).read_bytes()
        if sha(wb) != d["sha256"]:
            raise SystemExit(f"STOP: {p}: direction file sha256 differs from its stage-1 record")
        payload["linear"][p], dirs[p] = (d["layer"], wb), {"layer": d["layer"], "sha256": d["sha256"]}
    for L, v in sorted(cv.items(), key=lambda x: int(x[0])):
        wb = (REPO / cvp).parent.joinpath(v["output"]).read_bytes()
        if sha(wb) != v["output_sha256"]:
            raise SystemExit(f"STOP: FIBS L{L}: weight sha256 differs from the conversion record")
        payload["fibs"].append((int(L), wb))
    dirs["fibs"] = {f"L{L}": cv[L]["output_sha256"] for L in cv}
    specs, joins, recs, pre = {}, {}, {}, {}
    for od, (c, p, s) in ORIG.items():
        rj = mf[od]  # 0d (ii): the written record's digest (dump, folders' joins and files, probes without role, verdict, slice)
        specs[od] = dict(rj["dump"]) | {"entries": [], "cells": {}, "cell_main": {}, "gate_thresholds": {}, "dry_run": False, "test_write": False}  # the spec arm() received
        joins[od], recs[od] = rj["folders"][od]["joins"], {"commit": c, "path": p, "sha256": s}
        pre[f"{od}: weights equal the written record's probes"] = rj["probes"] == dirs
        pre[f"{od}: written verdict, one folder, mode simh, cutfull"] = rj.get("verdict") == "written" and list(rj["folders"]) == [od] and rj["dump"]["mode"] == "simh" and rj["dump"].get("cutfull") is True
        pre[f"{od}: written slice = the scorer default 300"] = rj["slice"] == 300
    ib = gshow(*INPUTS_FULL[:2])
    if sha(ib) != INPUTS_FULL[2]:
        raise SystemExit("STOP: INPUTS_FULL sha256 differs from its pin")
    stored, cite = {od: {} for od in ORIG}, []
    for ov in OVERRIDES:
        o_ = json.loads(ib)["overrides"][ov]
        ob = gshow(o_["commit"], o_["path"])
        if sha(ob) != o_["sha256"]:
            raise SystemExit(f"STOP: override {ov} sha256 differs from INPUTS_FULL's pin")
        cite.append({"override": ov} | o_)
        for key_, ents in json.loads(ob)["simh"].items():
            for e in (x for x in ents if isinstance(x, dict) and str(x.get("path", "")).endswith(".npz")):
                od = "/" + str(Path(e["path"]).parent)
                if od in stored:
                    stored[od][Path(e["path"]).name] = (e["path"], e["sha256"])
    for od in ORIG:
        rf = {n: h for n, h in mf[od]["folders"][od]["files"].items() if n.endswith(".npz")}
        pre[f"{od}: the overrides pin the written record's 8 npz (path and sha256)"] = {n: h for n, (_, h) in stored[od].items()} == rf and len(rf) == 8
    if not all(pre.values()):
        raise SystemExit(f"STOP before launch: {[k for k, v in pre.items() if not v]}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head_rec = {"git": head, "git_blob_ids": blob_ids, "old": OLD, "old_sha256": OLD_SHA, "stamp (PT)": stamp, "app": APP}
    failed = None
    try:
        ptr = json.loads(regress_ptr.remote(src, specs, payload, stored, joins, stamp))  # 0d (ii): the pointer only (verdicts, counts, types, paths, shas)
    except Exception as e:  # timeout, OOM, preemption or the harness's own type-only RuntimeError: a FAILED record, the type only
        failed = type(e).__name__
    if failed:
        _write_rec({"label": "A48a regression: FAILED, regress_ptr.remote raised (exception type only)"} | head_rec | {"remote_exception_type": failed,
                    "verdict": f"FAILED: regress_ptr.remote raised {failed}"}, stamp, log_dir)
        raise SystemExit(f"FAILED: regress_ptr.remote raised {failed} (record a48_regress_{stamp}.json)")
    a_ok, b_ok = ptr["A_pass"], ptr["B_pass"]  # computed in the container (_pointer), exactly as here before 0d (ii)
    rec = {"label": "A48a regression: (A) OLD 2313b6eb1 vs NEW in one container and process, bitwise, arm() only (not main()); (B) NEW vs the stored "
                    "production p2 files pooled A read, STOP only beyond the float32 compute bound (a float32-cancellation excess would be a fail-safe "
                    "false STOP); this record: booleans, counts, types and pointers only; the full result (per-file max |Δ|) on OUTVOL by path and "
                    "sha256 (0d (ii))"} | head_rec | {
           "scorer_blob_sha256": {"old": sha(src["old"]), "new": sha(src["new"])}, "shared_libs_blob_sha256": {n: sha(b) for n, b in src["libs"].items()},
           "written_records": recs, "inputs_full": dict(zip(("commit", "path", "sha256"), INPUTS_FULL)), "overrides": cite,
           "stored_files": {od: {n: {"path": p, "sha256": h} for n, (p, h) in v.items()} for od, v in stored.items()}, "prelaunch": pre, "result": ptr,
           "A_pass": a_ok, "B_pass": b_ok, "verdict": ptr["verdict"]}  # host metadata: raw file only (2e ruling (b)); ptr.env holds the three allowlisted fields
    out = _write_rec(rec, stamp, log_dir)
    print(f"env (2e ruling (b) allowlist only; full host metadata in the raw file): {json.dumps(ptr['env'])}")
    print(f"arm fails (counts and exception types only): {ptr['arm_fails'] or 'none'}; harness exceptions {ptr['n_harness_exceptions']} "
          f"(type {ptr['harness_exception_type']}); raw text NOT shown: {ptr['fails_raw']['volume']}:/{ptr['fails_raw']['path']} "
          f"sha256 {ptr['fails_raw']['sha256']}; arm hashes = bytes: {ptr['arm_hashes_match_bytes']}")
    print(f"result (not read here): {ptr['result']['volume']}:/{ptr['result']['path']} sha256 {ptr['result']['sha256']}")
    print(f"record {out}; (A) {a_ok}; (B) {b_ok}; {rec['verdict']}")
    if not (a_ok and b_ok):
        raise SystemExit(1)
