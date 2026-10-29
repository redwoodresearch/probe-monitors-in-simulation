"""rc1b W1 (route C reruns, 2026-10-20): INDEPENDENT re-derivation of rc1b slots 21 and 22.

Slots: 21.kappa_group.headline_six|C.joint.* and 22.kappa_five.C.joint.* (provenance v17 rows 231-240).
Definitions come only from the spec text quoted in W1_PRESTATED (RULE.md @ aee5aee09 L4535 and L4806-4813;
READ_LOG.md @ 3fbc0e451 L1427-1430; the records' own definition strings read by key). The producer's code was
not opened.

Route (code v11): per-member points from Stage D's per-task `caught` flags at draw 0, each cell weighted W_main / n_main
(the main-deduplicated weights; 09's ruling, G3 reversed); joint intervals from the kappa row store's per-draw margins
(D, po, p1, q1), read through the keys-only v11 layout file (form "margins"; W1's p1, the compared arm's rate, is the
row-store key <arm>|q1, and W1's q1, run 1's rate, is <arm>|p1), pooled over origins by D-weighting (summing
(D, po*D, p1*D, q1*D)), and verified at draw 0 against W1's weighted Stage D table by L0w (09's ruling). The 24 rows are
found by 09's matching rule: label == the kappa record's rows[L][o].full_variant_of, readout == "C", origin byte-equal to
one of the 4 pooled_all_origins strings.
The client ships sha256-checked bytes from `git show` and never parses a record. Every value is parsed and
compared inside the Modal container. Printed output: MATCH / MISMATCH booleans, counts, keys, shas, the Volume
path. Errors print the stage and the exception type only; the traceback goes to the Volume.

Modes: test (toy inputs, known answers), schema (keys-only layout of the row store), identity (keys-only identity
probe: whitelisted label / origin / readout strings, has__kappa, top-level key names), perturb (the predicted-FAIL
controls), real, l0diag (v10: the booleans-only diagnosis of PC2's L0 failure; pre-statement v11 section 11).

Run only from the committed blob in the item's worktree (the client refuses otherwise), with
W1_rowstore_listing.json (the frozen keys-only listing of the row store) committed beside this script:
  M=/Users/aswinsurya/.cache/probe-shift-modal161/bin/modal
  "$M" run --detach W1_check_kappa_slots.py --mode test
  "$M" run --detach W1_check_kappa_slots.py --mode perturb --layout W1_layout_rowstore_v11.json --layout-sha256 <hex>
  "$M" run --detach W1_check_kappa_slots.py --mode l0diag --layout W1_layout_rowstore.json --layout-sha256 <hex> \
      --rowstore-manifest-sha256 <the manifest_sha256 printed by the first launch that read the row store>
  "$M" run --detach W1_check_kappa_slots.py --mode real --layout W1_layout_rowstore_v11.json --layout-sha256 <hex> \
      --rowstore-manifest-sha256 <the manifest_sha256 printed by the first launch that read the row store>
"""

import datetime
import hashlib
import json
import os
import subprocess
import sys

import modal

APP_NAME = "probe-shift_aswin-surya_sub-rerun_rc1bw1_kill-2026-11-19"
OUT_VOL_NAME = APP_NAME
OUT_MOUNT = "/out"
OUT_SUBDIR = "rc1bw1"
RS_VOL_NAME = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
RS_MOUNT = "/rs"
RS_DIR = "xfer/fpr507prep/rowstore/kappa_CRD_2026-10-20_004454"
RS_N_FILES = 126
RS_TOTAL_BYTES = 796330995
# keys-only reference: the row store's (file, bytes) listing from `modal volume ls`, 13:3x and 14:01 PDT (identical)
RS_LISTING_FILE = "W1_rowstore_listing.json"
RS_LISTING_SHA256 = "92f34d48d7a66bc081f8c6280ac335c6a710c2817e8bc83378cf31b65b0a583e"
MODES = ("test", "schema", "identity", "perturb", "real", "l0diag")
PIN_MODES = ("real", "l0diag")  # modes that require the CLI row-store manifest pin
LAYOUT_MODES = ("real", "perturb", "l0diag")  # modes that read the row store through the layout file
# identity probe (09's ruling (a)(i), 2026-10-20 15:3x): a string prints only if it matches this (charset and length), and holds no
# decimal number; anything else prints as <redacted len=n>.
IDENT_CHARSET = r"[A-Za-z0-9 _|().:@+/-]{1,80}"
NAMES_SEP = "; "  # ";" is outside the charset, so it cannot occur inside a printed name

INPUTS = {
    "stageD": {
        "commit": "c22edd8d3cc1221225dec418ab00b40515d5ba43",
        "path": "experiments/2026-10-15_fpr_sweep_507_prep/results/routec_records/pertask_full_2026-10-20_004115.json",
        "sha256": "468655154f826ec1a93e08333eb802e68abea6db65a8e2f80a3e0f23465f36d9",
    },
    "kappa_gz": {
        "commit": "947ec6d4535377f9b1b204a53519a2f4af0a2646",
        "path": "experiments/2026-10-15_fpr_sweep_507_prep/results/routec_records/kappa_CRD_2026-10-20_004454.json.gz",
        "sha256": "b290f71c457609949555812faed78a1c18294d38e3e0138a6ff31f0a3d1516e8",
        "decompressed_sha256": "40ca1bf7a6deaf51175fb56cd89d7277b64ca1ec1ffdf87292d6ae0feca3cce9",
    },
    "kappa_extra": {
        "commit": "93ffca1ffcf76aef105940658af080939ffa1bf6",
        "path": "experiments/2026-10-15_fpr_sweep_507_prep/results/kappa_extra_kappa_CRD_2026-10-20_004454_2026-10-20_011442_ap-lSODwCn5gn3jLeCaCx38fX.json",
        "sha256": "85b995c9d640eab80b84c23bfac1b3945d49336047e7b48721507dc54e1e5a3f",
    },
    "rc1b": {
        "commit": "600f63614c0ab77f74fa7e326461e2e73086c2ac",
        "path": "experiments/2026-10-17_routec_slot_readout/results/slots_routeC_writing_rc1b.json",
        "sha256": "ab8dcb439b942c92d30b784ae1e6a3a193b18127391ff1753426663b726e728b",
    },
}

SIX = [
    "sol_a_uncut C",
    "FIBS|uncut C",
    "harmfulness|uncut C",
    "refusal|uncut C",
    "high-stakes|uncut C",
    "own_all_rs_uncut_L48 C",
]
FIVE = [m for m in SIX if m != "own_all_rs_uncut_L48 C"]
PAIRS = ("run2_vs_run1", "sim_vs_run1")
# v9 (margins form): every 2x2 table is held as SUMS = (N, A, X, Y): N cells, A agreeing cells (both caught or neither), X cells
# where the compared arm (run 2 or sim) is caught, Y cells where run 1 is caught. In counts: N = n11+n10+n01+n00, A = n11+n00,
# X = n11+n10, Y = n11+n01. The row store keeps per-draw MARGINS (D, po, p1, q1) per pair; SUMS = (D, po*D, p1*D, q1*D), so summing
# SUMS over origins is the D-weighted pooling of the margins, which equals pooling the counts (G6).
SUMS = ("N", "A", "X", "Y")
MARGINS = ("D", "po", "p1", "q1")
L0_INT_TOL = 1e-9  # 09's L0: D equals Stage D's n as an integer; each D*p within 1e-9 of an integer that equals Stage D's count
QUANTS = ("mean_kappa_run2_vs_run1", "mean_kappa_sim_vs_run1", "mean_kappa_sim_minus_run2")
MEMBER_Q = {
    "mean_kappa_run2_vs_run1": "kappa_run2_vs_run1",
    "mean_kappa_sim_vs_run1": "kappa_sim_vs_run1",
    "mean_kappa_sim_minus_run2": "kappa_sim_minus_run2",
}
GROUPS = {
    "six": {
        "members": SIX,
        "record": "kappa",
        "path": ["kappa_groups", "headline_six|C|pooled_all_origins"],
        "slot_prefix": "21.kappa_group.headline_six|C.joint.",
    },
    "five": {
        "members": FIVE,
        "record": "kappa_extra",
        "path": ["five_probe_sensitivity", "groups", "five_probe_sensitivity|C|pooled_all_origins"],
        "slot_prefix": "22.kappa_five.C.joint.",
    },
}
# provenance v17 @ 98b5054fb, rows 231-240 (predicate.component read by key)
PRINTED = {
    231: ("six", "mean_kappa_run2_vs_run1", "point"),
    232: ("six", "mean_kappa_sim_vs_run1", "point"),
    233: ("six", "mean_kappa_sim_minus_run2", "point"),
    234: ("six", "mean_kappa_sim_minus_run2", "interval_low"),
    235: ("six", "mean_kappa_sim_minus_run2", "interval_high"),
    236: ("five", "mean_kappa_run2_vs_run1", "point"),
    237: ("five", "mean_kappa_sim_vs_run1", "point"),
    238: ("five", "mean_kappa_sim_minus_run2", "point"),
    239: ("five", "mean_kappa_sim_minus_run2", "interval_low"),
    240: ("five", "mean_kappa_sim_minus_run2", "interval_high"),
}
COMPONENTS = ("point", "interval_low", "interval_high")
PERTURB = {"member": "FIBS|uncut C", "origin_prefix": "S_old ", "pair": "run2_vs_run1"}  # the one table PC2 perturbs
# PC2's controls, as additions to SUMS (N, A, X, Y) on that table at every draw (draw 0 included): one weight-1 cell with the arm
# caught and run 1 not (n10 + 1) is (1, 0, 1, 0); n10 + K and n01 + K is (2K, 0, K, K). v11: K is the member's number of distinct
# mains over the 4 origins (from Stage D's main keys), which is its pooled draw-0 weighted total under the dedup weights (each main
# sums to weight 1); v10 used the cell count 1313, which is the unweighted total. L0 on the row store only (vs the unperturbed
# Stage D table); C1 and C2 on both sources. With X = the compared arm and Y = run 1 on both sources (v11 layout), the vectors keep
# their meaning.
def ctrl_vectors(k_mains):
    k = float(k_mains)
    return {"L0": (1.0, 0.0, 1.0, 0.0), "C1": (1.0, 0.0, 1.0, 0.0), "C2": (2.0 * k, 0.0, k, k)}


CTRL = "ctrl_vectors(K): L0 and C1 (1, 0, 1, 0); C2 (2K, 0, K, K), K = the perturbed member's distinct mains over the 4 origins"
PRED_C2_FAIL_ROWS = (231, 233, 234, 235, 236, 238, 239, 240)  # predicted MISMATCH under the C2 vector; 232 and 237 MATCH
FLOAT_TOL = 1e-9
LAYOUT_VERSION = "v11"  # code v11 reads only the v11 layout (p1 = the compared arm's rate from the row-store key <arm>|q1)

image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.3", "tzdata==2025.2")
app = modal.App(APP_NAME)
out_vol = modal.Volume.from_name(OUT_VOL_NAME, create_if_missing=True)
rs_vol = modal.Volume.from_name(RS_VOL_NAME).read_only()


def _sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- container helpers (numpy imported lazily)

STAGE = ["init"]


class W1Stop(Exception):
    """A value-free stop: its message holds keys, counts or class paths only, so it may be printed."""


def stage(name):
    STAGE[0] = name


def kappa_of(t):
    """Cohen's kappa per row of t[..., 4] = SUMS (N, A, X, Y). RULE L4535: (p_o - p_e) / (1 - p_e), p_o = A/N,
    p_e = p1 q1 + (1 - p1)(1 - q1) with p1 = X/N (the compared arm's caught rate) and q1 = Y/N (run 1's)."""
    import numpy as np

    t = np.asarray(t, dtype=float)
    n = t[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        p_o = t[..., 1] / n
        p1 = t[..., 2] / n
        q1 = t[..., 3] / n
        p_e = p1 * q1 + (1.0 - p1) * (1.0 - q1)
        k = (p_o - p_e) / (1.0 - p_e)
    return np.where(np.isfinite(k), k, np.nan)


def table_of(x, y):
    """SUMS (N, A, X, Y) for boolean arrays x (the compared arm) and y (run 1), one element per cell."""
    import numpy as np

    x = np.asarray(x, dtype=bool)
    y = np.asarray(y, dtype=bool)
    return np.array([x.size, (x == y).sum(), x.sum(), y.sum()], dtype=float)


def match_rows(entries, fvo, origin_table, readout="C"):
    """09's matching rule (v10). entries: (file, label, readout, origin) raw values from the row-store pickles; fvo: {member: the
    kappa record's rows[member][o].full_variant_of}. A file matches (member, o) iff its label == fvo[member], its readout == "C",
    and its origin is byte-equal to o, one of origin_table. Returns (rows {(member, o): file}, n_duplicates, n_excluded), where
    n_excluded counts files with a C-member label and readout whose origin is NOT in the table (the paper_cells rows): never matched."""
    by_label = {v: k for k, v in fvo.items()}
    rows, dup, excl = {}, 0, 0
    for f, lab, ro, org in entries:
        if not (isinstance(lab, str) and lab in by_label and isinstance(ro, str) and ro == readout):
            continue
        m = by_label[lab]
        if isinstance(org, str) and org in origin_table:
            if (m, org) in rows:
                dup += 1
            else:
                rows[(m, org)] = f
        else:
            excl += 1
    return rows, dup, excl


def l0_int_ok(sums0, sd):
    """09's L0 on one (row, pair) at draw 0: N is integral and equals Stage D's N; each of A, X, Y (= D*po, D*p1, D*q1) is within
    L0_INT_TOL of an integer that equals Stage D's count exactly. sd = Stage D's SUMS (exact integers)."""
    n0 = float(sums0[0])
    if n0 != round(n0) or int(round(n0)) != int(sd[0]):
        return False
    for j in (1, 2, 3):
        v = float(sums0[j])
        k = round(v)
        if abs(v - k) > L0_INT_TOL or int(k) != int(sd[j]):
            return False
    return True


def l0w_ok(sums0, sdw, n):
    """v11's L0w (09's ruling) on one (row, pair) at draw 0: the row store's SUMS (D, po*D, p1*D, q1*D), read through the v11 layout
    (p1 = the compared arm's rate, from the row-store key <arm>|q1; q1 = run 1's, from <arm>|p1), equal W1's weighted Stage D table
    sdw = (sum w, sum w[agree], sum w[arm caught], sum w[run 1 caught]) in all four components, within 1e-9 * max(1, n), n = cells."""
    return allclose(sums0, sdw, SUM_TOL_REL * max(1.0, float(n)))


def is_degenerate(t):
    """A draw-0 table (N, A, X, Y) is degenerate (D3, READ_LOG L1429: "kappa is undefined or the catch is at the ceiling") if kappa is
    undefined (1 - p_e = 0) or either arm's catch is at 0 or 1 (to 1e-9 * max(1, N)). A degenerate check is never counted as an L0w
    pass: its match is required, but it is reported apart and every origin must also have an informative check."""
    import numpy as np

    n, _, x, y = (float(v) for v in t)
    tol = SUM_TOL_REL * max(1.0, n)
    ceiling = x <= tol or y <= tol or x >= n - tol or y >= n - tol
    return bool(ceiling or not np.isfinite(kappa_of(t)))


def group_stats(m_draws, points):
    """m_draws: (members, D) with draw 0 at column 0; points: per-member draw-0 values.
    Withhold a bootstrap draw if any member is not finite (G7); percentile interval (G8); point = mean (G9)."""
    import numpy as np

    m_draws = np.asarray(m_draws, dtype=float)
    boot = m_draws[:, 1:]
    ok = np.isfinite(boot).all(axis=0)
    vals = boot[:, ok].mean(axis=0)
    if vals.size:
        lo, hi = (float(v) for v in np.percentile(vals, [2.5, 97.5]))
    else:
        lo, hi = float("nan"), float("nan")
    pts = np.asarray(points, dtype=float)
    point = float(pts.mean()) if np.isfinite(pts).all() else float("nan")
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "finite_draws": int(ok.sum()), "draws": int(boot.shape[1])}


class _Missing:
    def __repr__(self):
        return "MISSING"


MISSING = _Missing()  # an absent record field: never MATCHes anything


def dig(obj, *keys):
    """Record lookup that returns MISSING (never raises) when a key or index is absent."""
    for k in keys:
        try:
            obj = obj[k]
        except (KeyError, IndexError, TypeError):
            return MISSING
    return obj


def ci_pair(r):
    """(lo, hi) from a record block: an explicit null ci95 gives (None, None); absent or malformed gives MISSING."""
    ci = dig(r, "ci95")
    if ci is None:
        return None, None
    if isinstance(ci, list) and len(ci) == 2:
        return ci[0], ci[1]
    return MISSING, MISSING


def _finite(x):
    import math

    return x is not None and not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def _undef(x):
    """Undefined = explicit null or NaN only (not infinity, not MISSING)."""
    import math

    return x is None or (isinstance(x, float) and math.isnan(x))


def cmp_float(mine, rec, tol=FLOAT_TOL):
    if rec is MISSING:
        return False
    if _undef(mine) and _undef(rec):
        return True  # both undefined
    if not _finite(mine) or not _finite(rec):
        return False
    return abs(float(mine) - float(rec)) <= tol


def cmp_int(mine, rec):
    return isinstance(rec, int) and not isinstance(rec, bool) and int(mine) == rec


# Number rule (v8, after Codex rounds 9-13), closed by prefixes: split the string into runs of [A-Za-z0-9_] (so ".", "-", "+",
# spaces etc. separate them) and drop the underscores in each run. A run that then starts with a digit, or (in any case) begins
# with "nan", "snan" or "inf", or is exactly "j", counts as a number, and the whole string is redacted. Every numeric form a Python
# float / int / complex / Decimal / Fraction, numpy, JSON or YAML parser accepts begins, once signs and dots have split it, with a
# digit (3, 1., .5, 1e-3, 1_000, 0x1F, 0x1p0, 1j), with one of those three words (nan, NaN123, sNaN, inf, Infinity, nanj), or is a
# bare j. Identifiers that begin otherwise (L48, run2, run_2, S_paper493, calibration_e12, s4_orphans_used, FIBS|uncut) print;
# words such as 1st, info or nanny are redacted too, which is the safe direction.
NUM_PREFIXES = ("nan", "snan", "inf")


def has_number(s):
    import re

    for run in re.findall(r"[A-Za-z0-9_]+", s):
        r = run.replace("_", "").lower()
        if r[:1].isdigit() or r.startswith(NUM_PREFIXES) or r == "j":
            return True
    return False


def ident_str(x):
    """(display, status) for an identifier under 09's whitelist. status: ok | redacted | absent | not_str. Never a value.
    A string prints only if it fullmatches IDENT_CHARSET and has_number() is false."""
    import re

    if x is MISSING:
        return "<absent>", "absent"
    if not isinstance(x, str):
        return f"<not_str type={type(x).__name__}>", "not_str"
    if re.fullmatch(IDENT_CHARSET, x) and not has_number(x):
        return x, "ok"
    return f"<redacted len={len(x)}>", "redacted"


def ident_disc(x, display, status):
    """A non-lossy grouping discriminator, used internally and never printed: the string when printable, else a sha256 of the
    redacted string or of the pickled non-string value, so that different values never merge into one group."""
    import pickle

    if status == "ok":
        return "ok:" + display
    if status == "absent":
        return "absent"
    if isinstance(x, str):
        return "redacted:" + _sha(x.encode("utf-8", "surrogatepass"))  # a lone surrogate must not crash the probe
    return "not_str:" + _sha(pickle.dumps(x, protocol=4))


def name_set(names):
    """(discriminator tuple, display string, n_names, n_hidden) for a collection of top-level key names, independent of order:
    names are sorted by their discriminators (never printed), and the displays are joined in that order with NAMES_SEP."""
    pairs = []
    for k in names:
        d, st = ident_str(k)
        pairs.append((ident_disc(k, d, st), d, st))
    pairs.sort(key=lambda p: p[0])
    return (tuple(p[0] for p in pairs), NAMES_SEP.join(p[1] for p in pairs), len(pairs),
            sum(p[2] != "ok" for p in pairs))


def _ident(s):
    """A pickle module/name operand printed only if it looks like a dotted identifier."""
    import re

    return s if isinstance(s, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,120}", s) else "<non-identifier>"


_NUM = r"(?<![\w.])([-+]?\d+(?:\.\d+)?)(?![\w.%])"


def parse_slot(s):
    """G11 grammar. Returns ([(value, decimals)] * 3, None) or (None, reason); reason never holds a value."""
    import re

    if not isinstance(s, str):
        return None, "not_str"
    t = s.replace(chr(0x2212), "-")  # U+2212 MINUS SIGN
    if re.search(r"\d\.?\d*[eE][-+]?\d", t):
        return None, "scientific"
    if re.search(r"(?i)\b(nan|inf|infinity)\b", t):
        return None, "nonfinite_word"
    toks = re.findall(_NUM, t)
    if len(toks) != 3:
        return None, f"n_numbers={len(toks)}"
    return [(float(x), len(x.split(".")[1]) if "." in x else 0) for x in toks], None


def cmp_str(mine, tok):
    val, d = tok
    if not _finite(mine):
        return False
    return abs(float(mine) - val) <= 0.5 * 10.0 ** (-d) + 1e-12


def safe_load(path):
    """Unpickle with an allow-list: builtins, collections, numpy reconstruction. Anything else stops."""
    import pickle

    allowed = {("builtins", n) for n in (
        "dict", "list", "tuple", "set", "frozenset", "str", "int", "float", "bool", "complex",
        "bytes", "bytearray", "slice", "range")}
    allowed |= {("collections", "OrderedDict"), ("collections", "defaultdict"), ("codecs", "encode"),
                ("_codecs", "encode")}  # protocol-2 bytes reconstruction
    allowed |= {(m, n) for m in ("numpy", "numpy.core.multiarray", "numpy._core.multiarray",
                                 "numpy.core.numeric", "numpy._core.numeric")
                for n in ("_reconstruct", "ndarray", "dtype", "scalar", "_frombuffer")}

    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in allowed:
                return super().find_class(module, name)
            raise W1Stop(f"disallowed_class {_ident(module)}.{_ident(name)}")

    with open(path, "rb") as f:
        return _U(f).load()


def get_path(obj, path):
    for p in path:
        if isinstance(p, int) and isinstance(obj, (list, tuple)):
            obj = obj[p]
        elif isinstance(obj, dict) and p in obj:
            obj = obj[p]
        else:
            raise W1Stop(f"layout_path_missing {json.dumps(path)}")
    return obj


def extract_sums(obj, spec):
    """spec: {"D": path, "po": path, "p1": path, "q1": path} (paths are key lists) for one pair. Reads the per-draw margins (draw on
    axis 0), checks them (finite; D >= 0; each p in [0, 1]; and at EVERY draw the four cell weights they imply are >= 0), and
    returns SUMS (Dn, 4) = (D, po*D, p1*D, q1*D)."""
    import numpy as np

    cols = []
    for c in MARGINS:
        col = np.asarray(get_path(obj, spec[c]), dtype=float)
        if col.ndim != 1:
            raise W1Stop(f"layout_margin_not_1d {c} ndim={col.ndim}")
        cols.append(col)
    if len({c.shape[0] for c in cols}) != 1:
        raise W1Stop("layout_margins_unequal_length")
    m = np.stack(cols, axis=1)
    if not np.isfinite(m).all() or (m[:, 0] < 0).any() or (m[:, 1:] < -1e-12).any() or (m[:, 1:] > 1 + 1e-12).any():
        raise W1Stop("layout_margins_not_finite_or_out_of_range")
    s = np.stack([m[:, 0], m[:, 1] * m[:, 0], m[:, 2] * m[:, 0], m[:, 3] * m[:, 0]], axis=1)
    if not cells_nonneg(s):
        raise W1Stop("layout_margins_imply_negative_cell")
    return s


def cells_nonneg(s):
    """The cell weights implied by SUMS rows (N, A, X, Y): n11 = (A + X + Y - N) / 2, n10 = X - n11, n01 = Y - n11, n00 = A - n11.
    True iff all are >= -1e-9 * max(1, N) at every row (floating tolerance only), so the margins are a possible 2x2 table."""
    import numpy as np

    s = np.asarray(s, dtype=float).reshape(-1, 4)
    n, a, x, y = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    n11 = (a + x + y - n) / 2.0
    tol = 1e-9 * np.maximum(1.0, n)
    return bool(((n11 >= -tol) & (x - n11 >= -tol) & (y - n11 >= -tol) & (a - n11 >= -tol)).all())


def dtype_sig(dt):
    """A dtype's structure without field titles (a title can be any object, so it is never printed)."""
    if dt.names is not None:
        return {"fields": {n: dtype_sig(dt.fields[n][0]) for n in dt.names}}
    if dt.subdtype is not None:
        return {"subdtype": dtype_sig(dt.subdtype[0]), "shape": list(dt.subdtype[1])}
    return dt.str


def sig(x, depth=0, full=False):
    """Keys-only signature: key names, Python types, array shapes and dtypes. Never a number or a string value."""
    import numpy as np

    # full=True: the untruncated grouping key (every element, every distinct signature; deeper than 32 levels STOPs);
    # full=False: the display form (first 2000 elements, 5 distinct signatures, depth <= 8).
    if full and depth > 32:
        raise W1Stop("schema_depth_exceeds_32")
    if not full and depth > 8:
        return "depth_limit"
    cap_n, cap_s = (None, None) if full else (2000, 5)
    if isinstance(x, dict):
        ks = list(x.keys())
        if len(ks) > 40 or not all(isinstance(k, str) for k in ks):
            subs = sorted({json.dumps(sig(v, depth + 1, full), sort_keys=True) for v in list(x.values())[:cap_n]})
            return {"dict_n_entries": len(ks), "name_types": sorted({type(k).__name__ for k in ks}),
                    "value_sigs": [json.loads(s) for s in subs[:cap_s]], "n_value_sigs": len(subs)}
        return {"dict": {k: sig(x[k], depth + 1, full) for k in ks}}
    if isinstance(x, (list, tuple)):
        subs = sorted({json.dumps(sig(v, depth + 1, full), sort_keys=True) for v in x[:cap_n]})
        return {type(x).__name__: len(x), "elem_sigs": [json.loads(s) for s in subs[:cap_s]], "n_elem_sigs": len(subs)}
    if isinstance(x, np.ndarray):
        return {"ndarray": list(x.shape), "dtype": dtype_sig(x.dtype)}
    if isinstance(x, np.generic):
        return "np." + type(x).__name__
    if isinstance(x, str):
        return "str"
    return type(x).__name__


GATE_PREFIXES = ("P.", "L0.", "C1.", "C2.", "C2x.")
DIAG_PREFIXES = ("D-a.", "D-b.", "D-c.", "D-d.", "D-e.", "D-f.", "D-g.", "L0.fixed")


def cnt(d, pre):
    ks = [k for k in d if k.startswith(pre)]
    return sum(d[k] for k in ks), len(ks), sorted(k for k in ks if not d[k])


def count_lines(gates, diags):
    out = []
    for pre in GATE_PREFIXES:
        a, n, bad = cnt(gates, pre)
        out.append(f"W1_COUNT gate={pre.rstrip('.')} match={a}/{n} failing={json.dumps(bad)}")
    for pre in DIAG_PREFIXES:
        a, n, bad = cnt(diags, pre)
        if n:
            out.append(f"W1_COUNT diag={pre.rstrip('.')} match={a}/{n} failing={json.dumps(bad)}")
    return out


def gate_summary(gates):
    return {pre.rstrip("."): list(cnt(gates, pre)[:2]) for pre in GATE_PREFIXES}


def jf(x):
    """JSON-safe float (non-finite -> None)."""
    return float(x) if _finite(x) else None


def clean(x):
    """Recursively JSON-safe: non-finite floats -> None, numpy scalars and arrays -> Python."""
    import numpy as np

    if x is MISSING:
        return "MISSING"
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float):
        return jf(x)
    return x


# ---------------------------------------------------------------- modes

def run_test():
    """PC1: toy inputs with known answers, plus planted-wrong sub-checks that must come out MISMATCH."""
    import pickle
    import re
    import tempfile

    import numpy as np

    stage("test")
    t = {}
    # SUMS (N, A, X, Y): counts (n11, n10, n01, n00) = (4, 1, 1, 4) are (10, 8, 5, 5); (0, 5, 5, 0) are (10, 0, 5, 5)
    t["kappa_0.6"] = abs(float(kappa_of([10, 8, 5, 5])) - 0.6) <= 1e-12
    t["kappa_minus1"] = abs(float(kappa_of([10, 0, 5, 5])) + 1.0) <= 1e-12
    t["kappa_undef_allcaught"] = bool(np.isnan(kappa_of([5, 5, 5, 5])))
    t["kappa_undef_nonecaught"] = bool(np.isnan(kappa_of([5, 5, 0, 0])))
    x1, y1 = [True, True, False, False], [True, False, True, False]
    x2, y2 = [True, False], [True, True]
    t["table_cells"] = table_of(x1, y1).tolist() == [4.0, 2.0, 2.0, 2.0]
    t["pooled_is_sum"] = (table_of(x1 + x2, y1 + y2) == table_of(x1, y1) + table_of(x2, y2)).all()
    m = np.array([[0.5, 0.1, 0.2, 0.3, 0.4], [0.5, 0.2, 0.3, np.nan, 0.5], [0.5, 0.3, 0.4, 0.5, 0.6]])
    g = group_stats(m, [0.1, 0.2, 0.6])
    exp_vals = np.array([0.2, 0.3, 0.5])  # draws 1, 2, 4; draw 3 withheld
    t["group_withhold"] = g["finite_draws"] == 3 and g["draws"] == 4
    t["group_ci"] = abs(g["ci_lo"] - float(np.percentile(exp_vals, 2.5))) <= 1e-12 and \
        abs(g["ci_hi"] - float(np.percentile(exp_vals, 97.5))) <= 1e-12
    t["group_point"] = abs(g["point"] - 0.3) <= 1e-12
    t["group_point_undef"] = np.isnan(group_stats(m, [0.1, np.nan, 0.6])["point"])
    lo, hi = np.percentile(np.arange(101.0), [2.5, 97.5])
    t["percentile_0_100"] = abs(lo - 2.5) <= 1e-12 and abs(hi - 97.5) <= 1e-12
    p, r = parse_slot("0.123 [" + chr(0x2212) + "0.045, 0.210]")
    t["grammar_minus"] = r is None and [v for v, _ in p] == [0.123, -0.045, 0.21] and [d for _, d in p] == [3, 3, 3]
    p, r = parse_slot("0.12 (95% CI 0.05" + chr(0x2013) + "0.19)")  # U+2013 EN DASH
    t["grammar_pct_endash"] = r is None and [v for v, _ in p] == [0.12, 0.05, 0.19]
    p, r = parse_slot("0.12 [0.05-0.19] x")
    t["grammar_hyphen_range"] = r is None and [v for v, _ in p] == [0.12, 0.05, 0.19]
    t["grammar_4numbers"] = parse_slot("n=5 0.1 [0.2, 0.3]")[1] == "n_numbers=4"
    t["grammar_sci"] = parse_slot("1e-05 [0.1, 0.2]")[1] == "scientific"
    t["grammar_nan"] = parse_slot("nan [0.1, 0.2]")[1] == "nonfinite_word"
    t["cmp_str_ok"] = cmp_str(0.1234, (0.12, 2)) and cmp_str(0.125, (0.12, 2)) and cmp_str(0.125, (0.13, 2))
    t["cmp_float_ok"] = cmp_float(0.5, 0.5 + 1e-12) and cmp_float(float("nan"), None) and cmp_int(7, 7)
    # planted-wrong sub-checks: each must come out MISMATCH (False)
    t["planted_str_off_by_one_unit"] = not cmp_str(0.13, (0.12, 2))
    t["planted_str_just_over_half_unit"] = not cmp_str(0.1251, (0.12, 2))
    t["planted_float_off_1e-6"] = not cmp_float(0.5, 0.5 + 1e-6)
    t["planted_float_undef_vs_value"] = not cmp_float(0.5, None)
    t["planted_int_bool"] = not cmp_int(1, True)
    t["ident_ok_and_redactions"] = (ident_str("FIBS|uncut") == ("FIBS|uncut", "ok")
                                    and ident_str("S_old (x) [y: z]") == ("<redacted len=16>", "redacted")
                                    and ident_str("p 0.123")[1] == "redacted" and ident_str("x" * 81)[1] == "redacted")
    t["ident_absent_not_str"] = ident_str(MISSING) == ("<absent>", "absent") and ident_str(3) == ("<not_str type=int>", "not_str")
    t["ident_sep_not_in_charset"] = re.fullmatch(IDENT_CHARSET, ";") is None and re.fullmatch(IDENT_CHARSET, "a;b") is None
    t["ident_numeric_literals_redacted"] = all(ident_str(s)[1] == "redacted"
                                               for s in ("p .123", "p 1.", "p 1e-3", "p 3", "0.01", "joint|0.01", "x -2",
                                                         "a 1_000", "a 1_000.25", "0x1F", "p nan", "a.1", "v2.2", "1e5",
                                                         "a 1j", "a 1_000j", "0x1p0", "p nanj", "p INF", "x 1b", "_1",
                                                         "j", "a -J", "nanJ", "InfinityJ", "sNaN", "NaN123", "sNaN_123", "a -NaN123",
                                                         "info", "nanny", "Inf_x"))
    t["ident_identifier_digits_ok"] = all(ident_str(s)[1] == "ok"
                                          for s in ("own_all_rs10_uncut_L48", "run2", "run_2", "S_paper493 (all-origins draws)",
                                                    "calibration_e12_key_sets", "H_B2", "s4_orphans_used",
                                                    "fold_minus1_draws", "FIBS|uncut", "C", "jump", "joint", "harmfulness|uncut", "high-stakes|uncut"))
    surrogate = "a" + chr(0xD800)
    t["ident_disc_surrogate_safe"] = (ident_str(surrogate)[1] == "redacted"
                                      and ident_disc(surrogate, *ident_str(surrogate)).startswith("redacted:"))
    names_a = ["1", 1] + [f"name_{c}" for c in "abcdefghijklmnopqrstuvwxyz"] + ["z" + c for c in "abcdefghijklmnop"]
    names_b = [1, "1"] + list(reversed(names_a[2:]))
    t["name_set_order_independent"] = name_set(names_a) == name_set(names_b) and name_set(names_a)[2] == 44 \
        and name_set(names_a)[3] == 2
    t["ident_disc_separates_same_display"] = (ident_str("[a]")[0] == ident_str("[b]")[0]
                                              and ident_disc("[a]", *ident_str("[a]")) != ident_disc("[b]", *ident_str("[b]"))
                                              and ident_disc(3, *ident_str(3)) != ident_disc(4, *ident_str(4)))
    t["planted_float_nan_vs_inf"] = not cmp_float(float("nan"), float("inf"))
    t["planted_float_missing"] = not cmp_float(float("nan"), MISSING) and not cmp_int(3, MISSING)
    t["grammar_sci_trailing_dot"] = parse_slot("0.1 [0.0, 0.2] 1.e2")[1] == "scientific"
    t["ident_sanitized"] = _ident("a b") == "<non-identifier>" and _ident("numpy.ndarray") == "numpy.ndarray"
    t["dig_missing"] = dig({"a": [1]}, "a", 5) is MISSING and dig({"a": 1}, "b") is MISSING and dig({"a": [1]}, "a", 0) == 1
    t["ci_pair_rules"] = (ci_pair({"ci95": [1, 2]}) == (1, 2) and ci_pair({}) == (MISSING, MISSING)
                          and ci_pair({"ci95": []}) == (MISSING, MISSING) and ci_pair({"ci95": None}) == (None, None))
    # margins extraction (dyadic fractions, so D*p is exact) and the allow-list on toy pickles
    marg = {"D": np.array([8.0, 16.0, 4.0]), "run2|po": np.array([0.75, 0.5, 0.25]),
            "run2|p1": np.array([0.5, 0.25, 0.5]), "run2|q1": np.array([0.5, 0.75, 0.25])}
    want = np.array([[8.0, 6.0, 4.0, 4.0], [16.0, 8.0, 4.0, 12.0], [4.0, 1.0, 2.0, 1.0]])
    toy = {"label": "toy C", "readout": "C", "origin": "S_x", "_kappa": {"joint": marg}}
    spec = {"D": ["_kappa", "joint", "D"], "po": ["_kappa", "joint", "run2|po"], "p1": ["_kappa", "joint", "run2|p1"],
            "q1": ["_kappa", "joint", "run2|q1"]}
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "toy.pkl")
        with open(fp, "wb") as f:
            pickle.dump(toy, f)
        t["layout_extract"] = (extract_sums(safe_load(fp), spec) == want).all()
        fp3 = os.path.join(d, "toy_p2.pkl")
        with open(fp3, "wb") as f:
            pickle.dump(toy, f, protocol=2)
        t["allowlist_protocol2"] = (extract_sums(safe_load(fp3), spec) == want).all()
    bad = {"_kappa": {"joint": dict(marg, **{"run2|po": np.array([0.75, 1.5, 0.25])})}}
    try:
        extract_sums(bad, spec)
        t["margins_out_of_range_stops"] = False
    except W1Stop:
        t["margins_out_of_range_stops"] = True
    # Codex round 15's example: (D, po, p1, q1) = (8, 1, 0.5, 0) at a later draw implies n01 = -2 and must STOP
    impossible = {"_kappa": {"joint": dict(marg, **{"run2|po": np.array([0.75, 1.0, 0.25]),
                                                     "run2|p1": np.array([0.5, 0.5, 0.5]),
                                                     "run2|q1": np.array([0.5, 0.0, 0.25]),
                                                     "D": np.array([8.0, 8.0, 4.0])})}}
    try:
        extract_sums(impossible, spec)
        t["margins_negative_cell_stops"] = False
    except W1Stop:
        t["margins_negative_cell_stops"] = cells_nonneg(want)  # and every possible toy table passes
    # 09's integer L0, and planted-wrong cases that must fail
    sd0 = np.array([8.0, 6.0, 4.0, 4.0])
    t["l0_int_exact"] = l0_int_ok(np.array([8.0, 6.0, 4.0, 4.0]), sd0) and l0_int_ok(np.array([8.0, 6.0 + 1e-10, 4.0, 4.0]), sd0)
    t["planted_l0_int"] = not any([l0_int_ok(np.array([9.0, 6.0, 4.0, 4.0]), sd0), l0_int_ok(np.array([8.5, 6.0, 4.0, 4.0]), sd0),
                                   l0_int_ok(np.array([8.0, 6.0 + 1e-6, 4.0, 4.0]), sd0),
                                   l0_int_ok(np.array([8.0, 7.0, 4.0, 4.0]), sd0)])
    # 09's matching rule on toy entries: 2 members x 2 table origins, one paper_cells row, one duplicate, one R row
    fvo = {"m1 C": "M1 C", "m2 C": "M2 C"}
    table = ["o1 [p]", "o2 [p]"]
    ents = [("f1", "M1 C", "C", "o1 [p]"), ("f2", "M1 C", "C", "o2 [p]"), ("f3", "M2 C", "C", "o1 [p]"),
            ("f4", "M2 C", "C", "o2 [p]"), ("f5", "M1 C", "C", "o1 [paper]"), ("f6", "M2 R", "R", "o1 [p]")]
    rows, dup, excl = match_rows(ents, fvo, table)
    t["match_rows_rule"] = len(rows) == 4 and dup == 0 and excl == 1 and rows[("m1 C", "o2 [p]")] == "f2"
    rows2, dup2, _ = match_rows(ents + [("f7", "M2 C", "C", "o2 [p]")], fvo, table)
    rows3, _, _ = match_rows(ents, fvo, ["o1 [p] "] + table[1:])  # a near-miss origin string (trailing space) must not match
    t["match_rows_dup_and_bytes"] = dup2 == 1 and len(rows2) == 4 and len(rows3) == 2
    ents_bad_ro = [(f, lab, np.array([ro]), org) for f, lab, ro, org in ents]  # a readout that is not a str must never match
    t["match_rows_readout_must_be_str"] = len(match_rows(ents_bad_ro, fvo, table)[0]) == 0
    titled = np.zeros(2, dtype=np.dtype([(("TITLEX", "fld"), "<f8")]))
    st = json.dumps(sig({"a": titled}), sort_keys=True)
    t["schema_dtype_no_title"] = "TITLEX" not in st and "fld" in st
    la, lb = [np.zeros(1)] * 2001, [np.zeros(1)] * 2000 + [np.zeros(2)]  # differ only after element 2000
    t["schema_full_sig_distinguishes"] = sig(la) == sig(lb) and sig(la, full=True) != sig(lb, full=True)
    deep = np.zeros(1)
    for _ in range(40):
        deep = [deep]
    try:
        sig(deep, full=True)
        t["schema_full_depth_stops"] = False
    except W1Stop:
        t["schema_full_depth_stops"] = sig(deep) is not None  # the display form still returns
    with tempfile.TemporaryDirectory() as d:
        fp2 = os.path.join(d, "bad.pkl")
        with open(fp2, "wb") as f:
            pickle.dump({"x": datetime.date(2020, 1, 1)}, f)  # a class outside the allow-list
        try:
            safe_load(fp2)
            t["allowlist_rejects_custom"] = False
        except W1Stop:
            t["allowlist_rejects_custom"] = True
    t.update(diag_toys())  # v10: the l0diag helpers' toys (the same set runs inside every l0diag launch)
    t.update(v11_toys())  # v11: the weighted estimator, L0w with the corrected orientation, degenerate rows, the C2 size
    t = {k: bool(v) for k, v in t.items()}
    lines = [f"W1_TEST t={k} ok={'true' if v else 'false'}" for k, v in t.items()]
    n_ok = sum(t.values())
    lines.append(f"W1_TEST_SUMMARY pass={n_ok}/{len(t)}")
    return {"exit": 0 if n_ok == len(t) else 1, "lines": lines, "record": {"tests": t}}


def load_inputs(blobs):
    import gzip

    stage("inputs")
    for name, spec in INPUTS.items():
        if _sha(blobs[name]) != spec["sha256"]:
            raise W1Stop(f"input_sha256_mismatch {name}")
    kb = gzip.decompress(blobs["kappa_gz"])
    if _sha(kb) != INPUTS["kappa_gz"]["decompressed_sha256"]:
        raise W1Stop("input_sha256_mismatch kappa_decompressed")
    return {
        "stageD": json.loads(blobs["stageD"]),
        "kappa": json.loads(kb),
        "kappa_extra": json.loads(blobs["kappa_extra"]),
        "rc1b": json.loads(blobs["rc1b"]),
    }


PIN_REL = os.path.join(OUT_SUBDIR, "W1_rowstore_manifest_pin.json")


def volume_pin(msha, mode, task):
    """Continuity across launches: the schema launch (the first reader) exclusive-created the pin file on the output Volume;
    every later launch must match it. From v8 only schema mode may create the pin; any other mode STOPs if it is absent."""
    out_vol.reload()
    fp = os.path.join(OUT_MOUNT, PIN_REL)
    if os.path.exists(fp):
        with open(fp, "rb") as f:
            pinned = json.loads(f.read()).get("manifest_sha256")
        return "match" if pinned == msha else "MISMATCH"
    if mode != "schema":
        return f"ABSENT_FOR_{mode.upper()}"
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "x") as f:
        json.dump({"manifest_sha256": msha, "created_by_mode": mode, "modal_task_id": task,
                   "created_pt": datetime.datetime.now(datetime.timezone.utc).isoformat()}, f, sort_keys=True)
    out_vol.commit()  # durable before any row data is processed
    return "created"


def rowstore_manifest(listing_bytes, want_manifest_sha256, mode, task):
    """sha256-pin every row-store file. STOP unless the (file, bytes) set equals the frozen listing, the Volume pin
    (volume_pin) holds, and, when a CLI pin is given (required for the real run), the manifest equals it."""
    stage("rowstore_manifest")
    if _sha(listing_bytes) != RS_LISTING_SHA256:
        raise W1Stop("rowstore_listing_sha256_mismatch")
    want = sorted((e["file"], e["bytes"]) for e in json.loads(listing_bytes))
    d = os.path.join(RS_MOUNT, RS_DIR)
    names = sorted(os.listdir(d))
    man = []
    for n in names:
        fp = os.path.join(d, n)
        if not os.path.isfile(fp) or not n.endswith(".pkl"):
            raise W1Stop(f"rowstore_unexpected_entry {_ident(n)}")
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        man.append({"file": n, "bytes": os.path.getsize(fp), "sha256": h.hexdigest()})
    msha = _sha(json.dumps(man, sort_keys=True, separators=(",", ":")).encode())
    ok_list = sorted((m["file"], m["bytes"]) for m in man) == want and len(want) == RS_N_FILES
    ok_b = sum(m["bytes"] for m in man) == RS_TOTAL_BYTES
    pin = "none" if not want_manifest_sha256 else ("match" if msha == want_manifest_sha256 else "MISMATCH")
    vpin = volume_pin(msha, mode, task) if ok_list and ok_b and pin != "MISMATCH" else "not_checked"
    line = (f"W1_ROWSTORE n_files={len(man)} listing_match={'true' if ok_list else 'false'} "
            f"total_bytes_ok={'true' if ok_b else 'false'} manifest_sha256={msha} cli_pin={pin} volume_pin={vpin}")
    return man, msha, ok_list and ok_b and pin != "MISMATCH" and vpin in ("match", "created"), line


def run_schema(inputs, man):
    """L-schema: keys-only structure of every row-store pickle, plus how many identify as a needed row."""
    stage("schema")
    labels = set(inputs["kappa"]["rows"].keys())
    origins = set(next(iter(inputs["kappa"]["rows"].values())).keys())
    needed = {(m, o) for m in SIX for o in inputs["stageD"]["meta"]["pertask"]["sets"]["pooled_all_origins"]}
    groups, ident, found = {}, 0, {}
    for m in man:
        obj = safe_load(os.path.join(RS_MOUNT, RS_DIR, m["file"]))
        s = sig(obj)  # display form
        key = _sha(json.dumps(sig(obj, full=True), sort_keys=True).encode())  # untruncated grouping key
        groups.setdefault(key, {"n_files": 0, "first_full_sig": s, "files": [], "group_sig_sha256": key})
        groups[key]["n_files"] += 1
        groups[key]["files"].append(m["file"])
        if isinstance(obj, dict) and isinstance(obj.get("label"), str) and isinstance(obj.get("origin"), str) \
                and obj["label"] in labels and obj["origin"] in origins:
            ident += 1
            if (obj["label"], obj["origin"]) in needed:
                found.setdefault((obj["label"], obj["origin"]), []).append(m["file"])
        del obj
    lines = [f"W1_SCHEMA n_sig_groups={len(groups)} identified_by_label_origin={ident}/{len(man)} "
             f"needed_found={len(found)}/{len(needed)} needed_dup={sum(len(v) > 1 for v in found.values())}"]
    for i, (_k, g) in enumerate(sorted(groups.items(), key=lambda kv: -kv[1]["n_files"])):
        lines.append(f"W1_SCHEMA_SIG id={i} n_files={g['n_files']} sig_sha256={g['group_sig_sha256'][:16]} sig={json.dumps(g['first_full_sig'], sort_keys=True)}")
    for (lab, org), files in sorted(found.items()):
        lines.append(f"W1_SCHEMA_NEEDED label={json.dumps(lab)} origin={json.dumps(org)} files={json.dumps(files)}")
    rec = {"groups": list(groups.values()), "needed_found": {f"{a} || {b}": v for (a, b), v in found.items()}}
    return {"exit": 0, "lines": lines, "record": rec}


def run_identity(man):
    """Keys-only identity probe (09's ruling (a)(i)). For every row-store file: the label, origin and readout strings under the
    whitelist (ident_str), has__kappa, and, for files with more than 40 top-level keys (summarized in schema mode), the top-level
    key names under the same whitelist. Files are grouped; only strings that pass the whitelist and counts are printed."""
    stage("identity")
    by_id, by_names, counts = {}, {}, {}
    n_dict = n_summarized = n_kappa = 0
    for m in man:
        obj = safe_load(os.path.join(RS_MOUNT, RS_DIR, m["file"]))
        top = obj if isinstance(obj, dict) else {}
        n_dict += isinstance(obj, dict)
        disp, disc = {}, {}
        for field in ("label", "origin", "readout"):
            raw = top.get(field, MISSING)
            d, st = ident_str(raw)
            disp[field], disc[field] = d, ident_disc(raw, d, st)
            counts[f"{field}_{st}"] = counts.get(f"{field}_{st}", 0) + 1
        hk = isinstance(obj, dict) and "_kappa" in obj
        n_kappa += hk
        # group on the non-lossy discriminators; print only the displays
        gid = (disc["label"], disc["origin"], disc["readout"], hk)
        by_id.setdefault(gid, {"display": (disp["label"], disp["origin"], disp["readout"], hk), "files": []})
        by_id[gid]["files"].append(m["file"])
        if isinstance(obj, dict) and len(obj) > 40:
            n_summarized += 1
            nid, names, n_names, n_hidden = name_set(obj.keys())
            counts["name_redacted_or_not_str"] = counts.get("name_redacted_or_not_str", 0) + n_hidden
            by_names.setdefault(nid, {"names": names, "files": [], "n_names": n_names, "n_hidden": n_hidden})
            by_names[nid]["files"].append(m["file"])
        del obj
    lines = [f"W1_IDENT n_files={len(man)} top_dict={n_dict} summarized_over_40={n_summarized} has__kappa={n_kappa} "
             f"n_id_groups={len(by_id)} n_name_groups={len(by_names)} counts={json.dumps(counts, sort_keys=True)}"]
    rec = {"id_groups": [], "name_groups": [], "counts": counts}
    for i, g in enumerate(sorted(by_id.values(), key=lambda g: (-len(g["files"]), str(g["display"]), g["files"][0]))):
        lab, org, ro, hk = g["display"]
        lines.append(f"W1_IDENT_GROUP id={i} n_files={len(g['files'])} has__kappa={'true' if hk else 'false'} "
                     f"label={json.dumps(lab)} readout={json.dumps(ro)} origin={json.dumps(org)}")
        rec["id_groups"].append({"label": lab, "origin": org, "readout": ro, "has__kappa": hk, "n_files": len(g["files"]),
                                 "files": g["files"]})
    for j, g in enumerate(sorted(by_names.values(), key=lambda g: (-len(g["files"]), g["names"], g["files"][0]))):
        lines.append(f"W1_IDENT_NAMES id={j} n_files={len(g['files'])} n_names={g['n_names']} n_hidden={g['n_hidden']} "
                     f"names={json.dumps(g['names'])}")
        rec["name_groups"].append({"names": g["names"], "n_names": g["n_names"], "n_hidden": g["n_hidden"],
                                   "n_files": len(g["files"]), "files": g["files"]})
    return {"exit": 0, "lines": lines, "record": rec}


def run_check(inputs, man, layout, perturb):
    import numpy as np

    stage("layout")
    for k in ("form", "identity", "joint", "draw0_index"):
        if k not in layout:
            raise W1Stop(f"layout_missing_field {k}")
    if layout["form"] != "margins":
        raise W1Stop("layout_form_not_margins")
    if layout["draw0_index"] != 0:
        raise W1Stop("layout_draw0_index_not_0")
    if layout.get("version") != LAYOUT_VERSION:  # v11: refuse the v10 layout, whose p1 / q1 reading the l0diag disproved
        raise W1Stop("layout_version_not_v11")
    ident = layout["identity"]
    if not (isinstance(ident, dict) and ident.get("readout") == "C" and isinstance(ident.get("origin_table"), list)
            and all(isinstance(s, str) for s in ident["origin_table"]) and ident.get("label") == "kappa_record_full_variant_of"):
        raise W1Stop("layout_identity_fields")
    modes = ["joint"] + (["fixed"] if layout.get("fixed") else [])
    for md in modes:
        for p in PAIRS:
            for c in MARGINS:
                spec = layout[md].get(p, {}).get(c) if isinstance(layout[md], dict) else None
                if not isinstance(spec, list) or not spec or not all(isinstance(s, str) for s in spec):
                    raise W1Stop(f"layout_missing {md}.{p}.{c}")
    sd, kp, kx, rc = inputs["stageD"], inputs["kappa"], inputs["kappa_extra"], inputs["rc1b"]
    gates, diags, lines = {}, {}, []
    rec = {"mine": {}, "theirs": {}}

    def gate(k, ok):
        gates[k] = bool(ok)

    def diag(k, ok):
        diags[k] = bool(ok)

    # ---- preconditions (P)
    stage("preconditions")
    def same_list(got, want):  # exact multiset equality, no duplicates allowed
        return isinstance(got, list) and len(got) == len(want) == len(set(want)) and sorted(got) == sorted(want)

    origins = list(sd["meta"]["pertask"]["sets"]["pooled_all_origins"])
    gate("P.n_origins_4_distinct", len(origins) == 4 and len(set(origins)) == 4)
    for m in SIX:
        gate(f"P.combo_members|{m}", same_list(dig(kp, "kappa", m, "combos", "pooled_all_origins", "members"), origins))
    for g, G in GROUPS.items():
        src = kp if G["record"] == "kappa" else kx
        gate(f"C1.{g}|members", same_list(dig(src, *G["path"], "members"), G["members"]))
    gate("P.five_base_members", same_list(dig(kx, "five_probe_sensitivity", "members"), [m[: -len(" C")] for m in FIVE]))
    for nm, ik in (("kappa_gz", "KAPPA"), ("kappa_extra", "KAPPA_EXTRA")):
        want_sha = INPUTS[nm].get("decompressed_sha256", INPUTS[nm]["sha256"])
        gate(f"P.rc1b_input|{ik}", dig(rc, "inputs", ik, "commit") == INPUTS[nm]["commit"]
             and dig(rc, "inputs", ik, "path") == INPUTS[nm]["path"] and dig(rc, "inputs", ik, "sha256") == want_sha)

    # ---- Stage D draw-0 tables per (member, origin) and pair. v11 (09's ruling; G3 reversed): each cell carries the dedup weight
    # W_main / n_main with W_main = 1 at draw 0 (inv_nmain, from Stage D's own main keys over the 4 origins), so a main in several
    # origins (or with several cells) counts once. Label: "a correction to the registered estimator (D2 L4807); G3's earlier
    # 'weight 1' reading was 09's error". Rows are parsed by stage_d_row (the same parser l0diag used; a non-finite max_minus_tau
    # entry is kept with its caught flag).
    stage("stageD_tables")
    sd_tab, sd_deg, sd_n, n_disagree, n_nonfinite_mmt = {}, {}, {}, 0, 0
    sdr = {(m, o): stage_d_row(sd["rows"][m][o]["per_task"], o) for m in SIX for o in origins}
    for m in SIX:
        wd = inv_nmain({o: sdr[(m, o)][2] for o in origins})
        for o in origins:
            cells, caught, _, nonfin, ndis = sdr[(m, o)]
            n_disagree += ndis
            n_nonfinite_mmt += int(sum(int(nonfin[a].sum()) for a in HIT_ARMS))
            # D-f (v11): the cell KEY sets are equal (Stage D's cells vs the kappa record's rows[m][o].cohort_cell_keys)
            rk = dig(kp, "rows", m, o, "cohort_cell_keys")
            diag(f"D-f.cohort_cells_equal|{m}|{o}", isinstance(rk, list) and all(isinstance(k, str) for k in rk)
                 and len(set(rk)) == len(rk) and sorted(rk) == cells)
            w = np.array([wd[(o, c)] for c in cells])
            sd_n[(m, o)] = len(cells)
            for p in PAIRS:
                t = wtable(caught[ARM_OF_PAIR[p]], caught["run1"], w)
                sd_tab[(m, o, p)] = t
                sd_deg[(m, o, p)] = is_degenerate(t)
    diags["D-e.caught_vs_mmt_disagree_is_0"] = n_disagree == 0
    diags["D-e.nonfinite_mmt_is_0"] = n_nonfinite_mmt == 0
    rec["counts"] = {"caught_vs_mmt_disagree": n_disagree, "nonfinite_max_minus_tau": n_nonfinite_mmt}
    lines.append(f"W1_DE caught_vs_mmt_disagree={n_disagree} nonfinite_max_minus_tau={n_nonfinite_mmt}")

    # ---- row store: identify the 24 needed rows by 09's rule (v10), then extract their per-draw margins as SUMS
    stage("rowstore_match")
    needed = {(m, o) for m in SIX for o in origins}
    gate("P.needed_rows_is_24", len(needed) == len(SIX) * 4 == 24)
    origin_table = ident["origin_table"]
    gate("P.origin_table_is_stageD_pooled_all_origins", same_list(origin_table, origins))
    # the label each member's rows carry: the kappa record's rows[m][o].full_variant_of, constant over the 4 origins and 1:1
    fvo = {}
    for m in SIX:
        vals = {dig(kp, "rows", m, o, "full_variant_of") for o in origins if isinstance(dig(kp, "rows", m, o, "full_variant_of"), str)}
        n_str = sum(isinstance(dig(kp, "rows", m, o, "full_variant_of"), str) for o in origins)
        gate(f"P.fvo_constant_and_str|{m}", n_str == 4 and len(vals) == 1)
        if len(vals) == 1:
            fvo[m] = vals.pop()
    gate("P.fvo_one_to_one", len(fvo) == len(SIX) and len(set(fvo.values())) == len(SIX))
    if not all(v for k, v in gates.items() if k.startswith(("P.fvo", "P.origin_table", "P.needed"))):
        bad = sorted(k for k, v in gates.items() if not v)
        raise W1Stop(f"matching_preconditions failing={json.dumps(bad)}")
    stage("rowstore_load")
    entries, rs_tab, first_file = [], {}, {}
    for f in man:
        obj = safe_load(os.path.join(RS_MOUNT, RS_DIR, f["file"]))
        top = obj if isinstance(obj, dict) else {}
        ent = (f["file"], top.get("label"), top.get("readout"), top.get("origin"))
        entries.append(ent)
        hit, _, _ = match_rows([ent], fvo, origin_table)
        for (m, o) in hit:  # this file matches (m, o); only the first such file is read (a duplicate STOPs below)
            if (m, o) not in first_file:
                first_file[(m, o)] = f["file"]
                for md in modes:
                    for p in PAIRS:
                        rs_tab[(md, m, o, p)] = extract_sums(obj, layout[md][p])
        del obj
    rs_file, n_dup, n_excl = match_rows(entries, fvo, origin_table)
    per_origin = [sum(1 for (m, o) in rs_file if o == origin_table[i]) for i in range(len(origin_table))]
    per_label = {m: sum(1 for (mm, o) in rs_file if mm == m) for m in SIX}
    gate("P.match_total_24", len(rs_file) == 24 and set(rs_file) == needed)
    gate("P.match_no_duplicate", n_dup == 0)
    for i, c in enumerate(per_origin):
        gate(f"P.match_per_origin_6|{i}", c == 6)
    for m in SIX:
        gate(f"P.match_per_label_4|{m}", per_label[m] == 4)
    lines.append(f"W1_MATCH total={len(rs_file)}/24 duplicates={n_dup} per_origin_by_table_index={json.dumps(per_origin)} "
                 f"per_label={json.dumps([per_label[m] for m in SIX])} excluded_c_member_non_table_origin={n_excl} "
                 f"unmatched_files={len(man) - len(rs_file) - n_dup - n_excl}")
    if not all(v for k, v in gates.items() if k.startswith("P.match")):
        bad = sorted(k for k, v in gates.items() if not v)
        lines.append(f"W1_STOP stage=matching failing={json.dumps(bad)}")
        lines += count_lines(gates, diags)
        return {"exit": 1, "lines": lines, "summary": gate_summary(gates),
                "record": {"gates": gates, "diags": diags, "entries_n": len(entries), **rec}}
    rec["rowstore_files_used"] = [[f"{a} || {b}", v] for (a, b), v in sorted(rs_file.items())]
    d_set = {t.shape[0] for t in rs_tab.values()}
    gate("P.rowstore_equal_D", len(d_set) == 1)
    D = d_set.pop()
    B = D - 1
    for g, G in GROUPS.items():
        src = kp if G["record"] == "kappa" else kx
        for md in modes:
            for q in QUANTS:
                gate(f"P.B_eq_draws|{g}|{md}|{q}", cmp_int(B, dig(src, *G["path"], md, q, "draws")))
    diag("D-g.stageD_draws_eq_B", cmp_int(B, dig(sd, "meta", "pertask", "draws")))

    # ---- L0 (09's integer rule): at draw 0, D equals Stage D's n and each D*p is within 1e-9 of an integer that equals Stage D's
    # count (A, X, Y), for 24 rows x 2 pairs; this checks the layout, the p1/q1 orientation and the matching rule by data
    stage("L0")
    # v11 (09's ruling): L0w, the row store's draw-0 SUMS against W1's weighted Stage D table in all four components. The v11 layout
    # reads the compared arm's rate from the row-store key <arm>|q1 and run 1's from <arm>|p1. Label: "identified by the draw-0
    # structural check (margins0_eq_row_hits_swapped 12/12 per origin; X/Y_eq_swapped 12/12 in S_newmains); no registered text names
    # it; kappa is invariant to it" (kappa_extra's degeneracy.rule names p1 and q1 without saying which arm each is).
    for md in modes:
        for (m, o) in sorted(needed):
            for p in PAIRS:
                ok = l0w_ok(rs_tab[(md, m, o, p)][0], sd_tab[(m, o, p)], sd_n[(m, o)])
                (gate if md == "joint" else diag)(f"L0.{md}|{m}|{o}|{p}", ok)
    # degenerate checks (ceiling or undefined at draw 0, W1's table) are never counted as passes: each must still match, they are
    # reported apart, and every origin must have at least one informative check
    for o in origins:
        gate(f"L0.informative_in_origin|{o}", any(not sd_deg[(m, o, p)] for m in SIX for p in PAIRS))
    n_l0 = sum(v for k, v in gates.items() if k.startswith("L0.joint|"))
    n_inf = sum(1 for d in sd_deg.values() if not d)
    n_inf_match = sum(bool(gates[f"L0.joint|{m}|{o}|{p}"]) for (m, o, p), d in sd_deg.items() if not d)
    n_orient = sum(1 for t in sd_tab.values() if abs(float(t[2]) - float(t[3])) > SUM_TOL_REL * max(1.0, float(t[0])))
    rec["l0w"] = {"degenerate": {f"{m}|{o}|{p}": d for (m, o, p), d in sd_deg.items()}, "informative": n_inf,
                  "informative_match": n_inf_match, "orientation_informative": n_orient}
    lines.append(f"W1_L0 draw0_joint_weighted_match={n_l0}/48 informative_match={n_inf_match}/{n_inf} "
                 f"degenerate={len(sd_deg) - n_inf} orientation_informative={n_orient}/{len(sd_tab)}")
    if n_l0 != 48 or not all(v for k, v in gates.items() if k.startswith(("P.", "L0."))):
        bad = sorted(k for k, v in gates.items() if not v)
        lines.append(f"W1_STOP stage=preconditions_or_L0 failing={json.dumps(bad)}")
        lines += count_lines(gates, diags)
        return {"exit": 1, "lines": lines, "summary": gate_summary(gates),
                "record": {"gates": gates, "diags": diags, **rec}}

    def evaluate(sd_t, rs_t):
        """Member values, groups (C1, D-d), slot strings (C2, C2x, D-c) and D-a, D-b for one pair of tables."""
        g_, d_, r_, fmt_fail_ = {}, {}, {"mine": {}, "theirs": {}}, []
        # per-member values: points from Stage D (combo = pooled cells, G3), draws from the row store (G6)
        stage("member_values")
        mem = {}
        for md in modes:
            for m in SIX:
                pts, drw = {}, {}
                for p in PAIRS:
                    pts[p] = float(kappa_of(sum(sd_t[(m, o, p)] for o in origins)))
                    drw[p] = kappa_of(sum(rs_t[(md, m, o, p)] for o in origins))
                mem[(md, m)] = {
                    "kappa_run2_vs_run1": (pts["run2_vs_run1"], drw["run2_vs_run1"]),
                    "kappa_sim_vs_run1": (pts["sim_vs_run1"], drw["sim_vs_run1"]),
                    "kappa_sim_minus_run2": (pts["sim_vs_run1"] - pts["run2_vs_run1"],
                                             drw["sim_vs_run1"] - drw["run2_vs_run1"]),
                }
        # D-a: per-member combo vs the kappa record (joint)
        for m in SIX:
            for mq, (pt, dr) in mem[("joint", m)].items():
                st = group_stats(dr[None, :], [pt])
                r = dig(kp, "kappa", m, "combos", "pooled_all_origins", "joint", mq)
                lo, hi = ci_pair(r)
                d_[f"D-a.{m}|{mq}|point"] = cmp_float(st["point"], dig(r, "point"))
                d_[f"D-a.{m}|{mq}|ci_lo"] = cmp_float(st["ci_lo"], lo)
                d_[f"D-a.{m}|{mq}|ci_hi"] = cmp_float(st["ci_hi"], hi)
                d_[f"D-a.{m}|{mq}|finite_draws"] = cmp_int(st["finite_draws"], dig(r, "finite_draws"))
                r_["mine"][f"member|{m}|{mq}"] = {k: jf(v) if isinstance(v, float) else v for k, v in st.items()}
        # D-b: per-member per-origin draw-0 points vs the kappa record
        for m in SIX:
            for o in origins:
                k2 = float(kappa_of(sd_t[(m, o, "run2_vs_run1")]))
                ks = float(kappa_of(sd_t[(m, o, "sim_vs_run1")]))
                for mq, v in (("kappa_run2_vs_run1", k2), ("kappa_sim_vs_run1", ks), ("kappa_sim_minus_run2", ks - k2)):
                    d_[f"D-b.{m}|{o}|{mq}"] = cmp_float(v, dig(kp, "kappa", m, "origins", o, "joint", mq, "point"))
        # groups (C1 for joint, D-d for fixed)
        stage("groups")
        mine = {}
        for g, G in GROUPS.items():
            src = kp if G["record"] == "kappa" else kx
            for md in modes:
                for q in QUANTS:
                    mq = MEMBER_Q[q]
                    st = group_stats(np.stack([mem[(md, m)][mq][1] for m in G["members"]]),
                                     [mem[(md, m)][mq][0] for m in G["members"]])
                    mine[(g, md, q)] = st
                    r = dig(src, *G["path"], md, q)
                    lo, hi = ci_pair(r)
                    put = g_ if md == "joint" else d_
                    pre = "C1" if md == "joint" else "D-d"
                    put[f"{pre}.{g}|{q}|point"] = cmp_float(st["point"], dig(r, "point"))
                    put[f"{pre}.{g}|{q}|ci_lo"] = cmp_float(st["ci_lo"], lo)
                    put[f"{pre}.{g}|{q}|ci_hi"] = cmp_float(st["ci_hi"], hi)
                    put[f"{pre}.{g}|{q}|finite_draws"] = cmp_int(st["finite_draws"], dig(r, "finite_draws"))
                    if md == "joint":
                        g_[f"C1.{g}|{q}|draws"] = cmp_int(st["draws"], dig(r, "draws"))
                    r_["mine"][f"group|{g}|{md}|{q}"] = {k: jf(v) if isinstance(v, float) else v for k, v in st.items()}
                    r_["theirs"][f"group|{g}|{md}|{q}"] = {k: dig(r, k) for k in ("point", "ci95", "finite_draws", "draws")}
        # slot strings (C2 printed components, C2x unprinted ones, D-c the `full` block)
        stage("slots")
        printed_keys = {(g, q, c) for (g, q, c) in PRINTED.values()}
        for g, G in GROUPS.items():
            for q in QUANTS:
                st = mine[(g, "joint", q)]
                vals = {"point": st["point"], "interval_low": st["ci_lo"], "interval_high": st["ci_hi"]}
                for block in ("slots", "full"):
                    key = G["slot_prefix"] + q
                    s = dig(rc, block, key)
                    toks, why = parse_slot(s)
                    r_["theirs"][f"{block}|{key}"] = s
                    if why is not None:
                        if block == "slots":
                            fmt_fail_.append(f"{key} reason={why}")
                            for c in COMPONENTS:
                                g_[f"{'C2' if (g, q, c) in printed_keys else 'C2x'}.{key}|{c}"] = False
                        else:
                            d_[f"D-c.{key}|format"] = False
                        continue
                    for c, tok in zip(COMPONENTS, toks, strict=True):
                        ok = cmp_str(vals[c], tok)
                        if block == "slots":
                            g_[f"{'C2' if (g, q, c) in printed_keys else 'C2x'}.{key}|{c}"] = ok
                        else:
                            d_[f"D-c.{key}|{c}"] = ok
        return {k: bool(v) for k, v in g_.items()}, {k: bool(v) for k, v in d_.items()}, r_, fmt_fail_

    def gate_lines(gs, fmt_fail_):
        out = []
        for row, (g, q, c) in sorted(PRINTED.items()):
            key = GROUPS[g]["slot_prefix"] + q
            out.append(f"W1_GATE v17_row={row} slot={json.dumps(key)} component={c} "
                       f"{'MATCH' if gs[f'C2.{key}|{c}'] else 'MISMATCH'}")
        for k in sorted(gs):
            if k.startswith("C1."):
                out.append(f"W1_GATE {k} {'MATCH' if gs[k] else 'MISMATCH'}")
        for f in fmt_fail_:
            out.append(f"W1_FORMAT_FAIL {f}")
        return out

    # ---- PC3, the real run: one evaluation of the unperturbed tables
    if not perturb:
        g2, d2, r2, fmt_fail = evaluate(sd_tab, rs_tab)
        gates.update(g2)
        diags.update(d2)
        rec["mine"].update(r2["mine"])
        rec["theirs"].update(r2["theirs"])
        lines += gate_lines(gates, fmt_fail)
        lines += count_lines(gates, diags)
        return {"exit": 0 if all(gates.values()) else 1, "lines": lines, "summary": gate_summary(gates),
                "record": {"gates": gates, "diags": diags, "modes": modes, "B": B, **rec}}

    # ---- PC2: a predicted-FAIL control for every gate, all in this launch (the unperturbed comparison is not run here)
    stage("perturb")
    po = [o for o in origins if o.startswith(PERTURB["origin_prefix"])]
    if len(po) != 1:
        raise W1Stop("perturb_origin_not_unique")
    po = po[0]
    # v11: the C2 size K = the perturbed member's distinct mains over the 4 origins (its pooled draw-0 weighted total)
    k_mains = len({mn for o in origins for mn in sdr[(PERTURB["member"], o)][2].values()})
    ctrl = ctrl_vectors(k_mains)

    def perturbed(add, stage_d_too):
        """Copies of the tables with the SUMS vector `add` (N, A, X, Y) added to one table (FIBS|uncut C, S_old, run2_vs_run1), at
        every draw; the originals are not changed."""
        sd_c, rs_c = dict(sd_tab), dict(rs_tab)
        k_sd, k_rs = (PERTURB["member"], po, PERTURB["pair"]), ("joint", PERTURB["member"], po, PERTURB["pair"])
        vec = np.asarray(add, dtype=float)
        rs_c[k_rs] = rs_c[k_rs] + vec[None, :]
        if stage_d_too:
            sd_c[k_sd] = sd_c[k_sd] + vec
        return sd_c, rs_c

    run2, sim, diff = QUANTS
    pred = {
        "L0": [f"L0.joint|{PERTURB['member']}|{po}|{PERTURB['pair']}"],
        "C1_core": sorted(f"C1.{g}|{q}|{c}" for g in GROUPS for q in (run2, diff) for c in ("point", "ci_lo", "ci_hi")),
        "C1_may_also_fail": sorted(f"C1.{g}|{q}|finite_draws" for g in GROUPS for q in (run2, diff)),
        "C2": sorted(f"C2.{GROUPS[g]['slot_prefix'] + q}|{c}" for row, (g, q, c) in PRINTED.items()
                     if row in PRED_C2_FAIL_ROWS),
        "C2x": sorted(f"C2x.{GROUPS[g]['slot_prefix'] + run2}|{c}" for g in GROUPS for c in ("interval_low", "interval_high")),
    }
    controls = {}
    # L0 control: the perturbed row-store draw 0 against the UNPERTURBED Stage D table
    _, rs_l0 = perturbed(ctrl["L0"], stage_d_too=False)
    l0c = {f"L0.joint|{m}|{o}|{p}": l0w_ok(rs_l0[("joint", m, o, p)][0], sd_tab[(m, o, p)], sd_n[(m, o)])
           for (m, o) in sorted(needed) for p in PAIRS}
    l0_fail = sorted(k for k, v in l0c.items() if not v)
    controls["L0"] = {"match": [sum(l0c.values()), len(l0c)], "failing": l0_fail, "as_predicted": l0_fail == pred["L0"]}
    # C1 control: +1 to n10 in both sources
    sd1, rs1 = perturbed(ctrl["C1"], stage_d_too=True)
    g1, d1, r1, ff1 = evaluate(sd1, rs1)
    g1.update({k: v for k, v in gates.items() if k.startswith("C1.")})  # the two member checks
    c1_fail = sorted(k for k, v in g1.items() if k.startswith("C1.") and not v)
    controls["C1"] = {"match": [sum(v for k, v in g1.items() if k.startswith("C1.")), sum(k.startswith("C1.") for k in g1)],
                      "failing": c1_fail,
                      "as_predicted": set(pred["C1_core"]) <= set(c1_fail) <= set(pred["C1_core"]) | set(pred["C1_may_also_fail"])}
    # C2 / C2x control: +K to n10 and to n01 in both sources (pre-statement section 6 and v12 section 13 say why both)
    sd2, rs2 = perturbed(ctrl["C2"], stage_d_too=True)
    g2, d2, r2, ff2 = evaluate(sd2, rs2)
    c2_fail = sorted(k for k, v in g2.items() if k.startswith("C2.") and not v)
    c2x_fail = sorted(k for k, v in g2.items() if k.startswith("C2x.") and not v)
    controls["C2"] = {"match": [sum(v for k, v in g2.items() if k.startswith("C2.")), sum(k.startswith("C2.") for k in g2)],
                      "failing": c2_fail, "as_predicted": c2_fail == pred["C2"] and not ff2}
    controls["C2x"] = {"match": [sum(v for k, v in g2.items() if k.startswith("C2x.")), sum(k.startswith("C2x.") for k in g2)],
                       "failing": c2x_fail, "as_predicted": c2x_fail == pred["C2x"] and not ff2}
    lines.append(f"W1_PERTURB member={json.dumps(PERTURB['member'])} origin={json.dumps(po)} pair={PERTURB['pair']} "
                 f"controls={json.dumps(ctrl, sort_keys=True)} all_draws=true")
    for name in ("L0", "C1", "C2", "C2x"):
        cr = controls[name]
        lines.append(f"W1_CONTROL name={name} match={cr['match'][0]}/{cr['match'][1]} failing={json.dumps(cr['failing'])} "
                     f"as_predicted={'true' if cr['as_predicted'] else 'false'}")
    all_pred = all(cr["as_predicted"] for cr in controls.values())
    lines += count_lines(gates, diags)  # the unperturbed P and L0 gates, and the two member checks
    summary = {**gate_summary(gates), "controls_as_predicted": {k: v["as_predicted"] for k, v in controls.items()}}
    return {"exit": 1 if all_pred else 3, "lines": lines, "summary": summary,
            "record": {"gates": gates, "diags": diags, "modes": modes, "B": B, "controls": controls, "predicted": pred,
                       "variants": {"C1": {"gates": g1, "diags": d1, **r1, "format_fail": ff1},
                                    "C2": {"gates": g2, "diags": d2, **r2, "format_fail": ff2}}, **rec}}


# ---------------------------------------------------------------- l0diag (code v10; pre-statement v11 section 11)
# Booleans and integer counts only. Every value is parsed and compared inside the container; the Volume record holds booleans,
# counts and pattern letters per key, never a margin, weight, rate or kappa.

LEVEL = "0.01"  # the record's per-action level key (cell_hits_draw0.levels); Stage D's f is the same level
ARM_OF_PAIR = {"run2_vs_run1": "run2", "sim_vs_run1": "sim"}
HIT_ARMS = ("run1", "run2", "sim")
MARGIN_TOL = 1e-12  # rates and weights vs rates and weights (one job, float64)
SUM_TOL_REL = 1e-9  # weighted sums vs D: 1e-9 * max(1, n) (09)
PC2_L0_PASS = {("own_all_rs_uncut_L48 C", "S_newmains", p) for p in PAIRS}  # PC2's 2/48 (its log, line 22), origin by short name
DIAG_CTL = {  # the four positive controls, each on an in-container copy (section 11.6)
    "d_plus_1": ("FIBS|uncut C", "S_newmains", "run2_vs_run1"),
    "flip_run1": ("sol_a_uncut C", "S_old", None),
    "roll_by_1": ("FIBS|uncut C", "S_old", "run2_vs_run1"),
    "weights_ones": ("sol_a_uncut C", "S_old", None),
}
PATTERNS = ("a", "b", "c", "W", "other")
CHECK_BOOLS = ("N_integral", "N_eq_stageD_n", "A_int_ok", "A_eq_ok", "X_int_ok", "X_eq_ok", "Y_int_ok", "Y_eq_ok",
               "X_eq_swapped", "Y_eq_swapped", "L0_int", "L0_int_eq_l0_int_ok", "idx0_eq_record_draw0",
               "idxlast_eq_record_draw0", "D0_eq_row_weight_sum", "margins0_eq_row_hits", "margins0_eq_row_hits_swapped",
               "hits_agree", "L0w", "L0w_row_weights")
ROW_BOOLS = ("weights_all_one", "weights_eq_inv_nmain", "weights_eq_inv_nmain_within_origin", "stageD_has_nonfinite_mmt",
             "ceiling_all3", "mains_align_stageD") + tuple(f"{k}|{a}" for k in ("hits_eq_stageD", "stageD_catch_eq_unweighted",
                                                                                "stageD_catch_eq_weighted",
                                                                                "stageD_catch_eq_row_hits_weighted")
                                                          for a in HIT_ARMS)
ROW_COUNTS = ("n_nonfinite_mmt_entries",) + tuple(f"{k}|{a}" for k in ("n_hit_mismatch", "n_hit_mismatch_on_nonfinite_mmt")
                                                  for a in HIT_ARMS)


def oshort(o):
    """An origin's short name (S_old, S_new, S_newmains, S_paper493): the text before the first space."""
    return o.split(" ")[0]


def check_layout(layout):
    """The layout file's structure (the same checks run_check makes). Returns the modes it declares."""
    for k in ("form", "identity", "joint", "draw0_index"):
        if k not in layout:
            raise W1Stop(f"layout_missing_field {k}")
    if layout["form"] != "margins":
        raise W1Stop("layout_form_not_margins")
    if layout["draw0_index"] != 0:
        raise W1Stop("layout_draw0_index_not_0")
    ident = layout["identity"]
    if not (isinstance(ident, dict) and ident.get("readout") == "C" and isinstance(ident.get("origin_table"), list)
            and all(isinstance(s, str) for s in ident["origin_table"]) and ident.get("label") == "kappa_record_full_variant_of"):
        raise W1Stop("layout_identity_fields")
    modes = ["joint"] + (["fixed"] if layout.get("fixed") else [])
    for md in modes:
        for p in PAIRS:
            for c in MARGINS:
                spec = layout[md].get(p, {}).get(c) if isinstance(layout[md], dict) else None
                if not isinstance(spec, list) or not spec or not all(isinstance(s, str) for s in spec):
                    raise W1Stop(f"layout_missing {md}.{p}.{c}")
    return modes


def hits_of(s, n):
    """A record hit string ('0' / '1' per cohort cell, in sorted cohort_cell_keys order) as a bool array of length n. Any other
    character, length or type is PLUMBING: the record's definition allows '-' only for the extra arms (sim2, run3)."""
    import numpy as np

    if not isinstance(s, str) or len(s) != n or set(s) - {"0", "1"}:
        raise W1Stop("PLUMBING hit_string_not_01_or_wrong_length")
    return np.frombuffer(s.encode("ascii"), dtype=np.uint8) == ord("1")


def wtable(x, y, w):
    """Weighted SUMS (N, A, X, Y) = (sum w, sum w[x == y], sum w[x], sum w[y]) for boolean arrays x (the compared arm) and y
    (run 1); with every w = 1 it equals table_of(x, y)."""
    import numpy as np

    x = np.asarray(x, dtype=bool)
    y = np.asarray(y, dtype=bool)
    w = np.asarray(w, dtype=float)
    return np.array([w.sum(), w[x == y].sum(), w[x].sum(), w[y].sum()], dtype=float)


def inv_nmain(cells_by_origin, within_origin=False):
    """Main-deduplicated draw-0 cell weights, W1's own derivation from Stage D's main keys (meta.merge.rule: "cell weight
    W_main / n_main", with W_main = 1 at draw 0). cells_by_origin: {origin: {cell: main}} over the 4 pooled_all_origins origins.
    n_main counts the (origin, cell) pairs of that main over all the given origins, or, with within_origin=True (the alternative
    reading, reported beside), within the cell's own origin. Returns {(origin, cell): 1 / n_main}."""
    from collections import Counter

    if within_origin:
        c = Counter((o, mn) for o, d in cells_by_origin.items() for mn in d.values())
        return {(o, cl): 1.0 / c[(o, mn)] for o, d in cells_by_origin.items() for cl, mn in d.items()}
    c = Counter(mn for d in cells_by_origin.values() for mn in d.values())
    return {(o, cl): 1.0 / c[mn] for o, d in cells_by_origin.items() for cl, mn in d.items()}


def allclose(a, b, tol):
    """Equal shapes, all finite, and |a - b| <= tol elementwise."""
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return bool(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and (np.abs(a - b) <= tol).all())


def l0_parts(sums0, sd):
    """09's per-check booleans for one (row, pair) at draw 0: N_integral, N_eq_stageD_n, and for each of A, X, Y int_ok (within
    L0_INT_TOL of an integer) and eq_ok (that integer equals Stage D's count). X_eq_swapped / Y_eq_swapped read p1 and q1 the
    other way round. L0_int is the conjunction that l0_int_ok computes (checked against it per key)."""
    n0 = float(sums0[0])
    ni = n0 == round(n0)
    out = {"N_integral": ni, "N_eq_stageD_n": ni and int(round(n0)) == int(sd[0])}
    for j, nm in ((1, "A"), (2, "X"), (3, "Y")):
        v = float(sums0[j])
        k = round(v)
        io = abs(v - k) <= L0_INT_TOL
        out[f"{nm}_int_ok"] = io
        out[f"{nm}_eq_ok"] = io and int(k) == int(sd[j])
    out["X_eq_swapped"] = out["X_int_ok"] and int(round(float(sums0[2]))) == int(sd[3])
    out["Y_eq_swapped"] = out["Y_int_ok"] and int(round(float(sums0[3]))) == int(sd[2])
    out["L0_int"] = out["N_eq_stageD_n"] and out["A_eq_ok"] and out["X_eq_ok"] and out["Y_eq_ok"]
    return out


def classify(c):
    """The exclusive pattern of one integer-L0 failure, from its booleans (pre-statement v11 section 11.4):
      c      index 0 of the row store is not the record's draw 0 (idx0_eq_record_draw0 false);
      other  the row's own draw-0 margins disagree with its own weights or hits (D0_eq_row_weight_sum or margins0_eq_row_hits
             false);
      a      Stage D's caught differs from the row's hits (arm or run 1), while the weights are the dedup weights;
      b      the hits agree, but the weights are not the dedup weights (draw-0 multiplicities not 1);
      W      the hits agree, the weights are the dedup weights, some are below 1, and the weighted L0 (L0w) passes;
      other  anything else (e.g. both a and b, or unit weights with agreeing hits, where L0 should have passed)."""
    if not c["idx0_eq_record_draw0"]:
        return "c"
    if not (c["D0_eq_row_weight_sum"] and c["margins0_eq_row_hits"]):
        return "other"
    hits, wd = c["hits_agree"], c["weights_eq_inv_nmain"]
    if not hits and wd:
        return "a"
    if hits and not wd:
        return "b"
    if hits and wd and not c["weights_all_one"] and c["L0w"]:
        return "W"
    return "other"


def stage_d_row(entries, o):
    """Stage D per_task entries of one (member, origin) row -> (cells sorted, {arm: caught bool array}, {cell: main},
    {arm: bool array, max_minus_tau non-finite}, caught_vs_mmt disagreements). A non-finite max_minus_tau entry is KEPT with its
    `caught` flag (the definition: caught = max > tau). STOPs as PLUMBING on a foreign origin, a non-bool caught, a duplicate
    (cell, arm), a cell whose main is not one str, or a cell without all three arms."""
    import numpy as np

    by_cell, main_of, nf = {}, {}, {}
    n_dis = 0
    for e in entries:
        if not isinstance(e, dict) or e.get("origin") != o:
            raise W1Stop("PLUMBING stageD_entry_origin")
        if not isinstance(e.get("caught"), bool):
            raise W1Stop("PLUMBING stageD_caught_not_bool")
        c, a, mn = e.get("cell"), e.get("arm"), e.get("main")
        if not (isinstance(c, str) and isinstance(a, str) and isinstance(mn, str)):
            raise W1Stop("PLUMBING stageD_cell_arm_main_not_str")
        arms = by_cell.setdefault(c, {})
        if a in arms:
            raise W1Stop("PLUMBING stageD_dup_cell_arm")
        arms[a] = e["caught"]
        if main_of.setdefault(c, mn) != mn:
            raise W1Stop("PLUMBING stageD_cell_two_mains")
        mmt = e.get("max_minus_tau")
        fin = _finite(mmt)
        nf.setdefault(c, {})[a] = not fin
        if fin:
            n_dis += int(e["caught"] != (mmt > 0))
    cells = sorted(by_cell)
    if not cells or any(set(by_cell[c]) != set(HIT_ARMS) for c in cells):
        raise W1Stop("PLUMBING stageD_incomplete_cell")
    caught = {a: np.array([by_cell[c][a] for c in cells], dtype=bool) for a in HIT_ARMS}
    nonfin = {a: np.array([nf[c][a] for c in cells], dtype=bool) for a in HIT_ARMS}
    return cells, caught, main_of, nonfin, n_dis


PLUMB_ROW_PATHS = (["cohort_cell_keys"], ["cell_hits_draw0", "weight"], ["cell_hits_draw0", "mains"],
                   ["cell_hits_draw0", "arms_def"], ["cell_hits_draw0", "levels", LEVEL], ["kappa_draw0_direct", "run2"],
                   ["kappa_draw0_direct", "sim"])


def num_array(x, what):
    """A 1-D array of finite numbers from a numpy array of a numeric dtype (float, signed or unsigned int) or a list / tuple whose
    every element is a real number (int / float or a numpy integer / floating; never a bool, str or None). Anything else, or a
    non-finite element, STOPs as PLUMBING (Codex round 17: no silent coercion of strings or None)."""
    import numpy as np

    if isinstance(x, np.ndarray):
        if x.dtype.kind not in "fiu":
            raise W1Stop(f"PLUMBING not_numeric {what}")
        a = x.astype(float)
    elif isinstance(x, (list, tuple)) and all(isinstance(v, (int, float, np.integer, np.floating))
                                              and not isinstance(v, (bool, np.bool_)) for v in x):
        a = np.asarray(x, dtype=float)
    else:
        raise W1Stop(f"PLUMBING not_numeric {what}")
    if a.ndim != 1 or not np.isfinite(a).all():
        raise W1Stop(f"PLUMBING not_finite_1d {what}")
    return a


def row_fields(obj, rec_row, layout, modes):
    """Read and type-check one matched row-store pickle; any missing key or wrong type STOPs as PLUMBING. Identity against the
    trusted kappa record's row (the job's own output for the same row): the pickle's kappa_draw0_direct and cohort_cell_keys must
    EQUAL the record's, else PLUMBING. Returns {"marg": {(mode, pair): (Dn, 4) margins (D, po, p1, q1)}, "w", "mains", "keys",
    "hits": {arm: bool array}, "catch_key": {arm: str}, "kd": {arm: (po, p1, q1)}}."""
    import numpy as np

    top = obj if isinstance(obj, dict) else {}
    for path in PLUMB_ROW_PATHS:
        if dig(top, *path) is MISSING:
            raise W1Stop(f"PLUMBING missing_field path_index={PLUMB_ROW_PATHS.index(path)}")
    keys = top["cohort_cell_keys"]
    if not (isinstance(keys, list) and keys and all(isinstance(k, str) for k in keys)):
        raise W1Stop("PLUMBING cohort_cells_not_str_list")
    n = len(keys)
    ch = top["cell_hits_draw0"]
    w = num_array(ch["weight"], "weight")
    if w.shape != (n,) or (w < 0).any():
        raise W1Stop("PLUMBING weight_shape_or_range")
    mains = ch["mains"]
    if not (isinstance(mains, list) and len(mains) == n and all(isinstance(x, str) for x in mains)):
        raise W1Stop("PLUMBING mains_not_str_list_of_n")
    lev = ch["levels"][LEVEL]
    hits, catch_key = {}, {}
    for a in HIT_ARMS:
        if not isinstance(lev, dict) or a not in lev:
            raise W1Stop(f"PLUMBING missing_field levels_arm {a}")
        hits[a] = hits_of(lev[a], n)
        ck = dig(ch, "arms_def", a, "catch_key")
        if not isinstance(ck, str):
            raise W1Stop(f"PLUMBING catch_name_not_str {a}")
        catch_key[a] = ck
    kd = {}
    same = True
    for a in ("run2", "sim"):
        v = [dig(top, "kappa_draw0_direct", a, c) for c in ("po", "p1", "q1")]
        r = [dig(rec_row, "kappa_draw0_direct", a, c) for c in ("po", "p1", "q1")]
        if not all(_finite(x) for x in v + r):
            raise W1Stop(f"PLUMBING kappa_draw0_direct_not_finite {a}")
        kd[a] = np.array(v, dtype=float)
        same = same and bool((kd[a] == np.array(r, dtype=float)).all())
    if not same or keys != dig(rec_row, "cohort_cell_keys"):
        raise W1Stop("PLUMBING pickle_vs_record_identity")
    marg = {}
    for md in modes:
        for p in PAIRS:
            cols = []
            for c in MARGINS:
                x = dig(top, *layout[md][p][c])
                if x is MISSING:
                    raise W1Stop(f"PLUMBING missing_field margin {md}.{p}.{c}")
                cols.append(num_array(x, f"margin {md}.{p}.{c}"))
            if len({c.shape[0] for c in cols}) != 1 or cols[0].shape[0] < 2:
                raise W1Stop(f"PLUMBING margins_shape {md}.{p}")
            mg = np.stack(cols, axis=1)
            # the same ranges as extract_sums, at every draw: D >= 0, each rate in [0, 1] (to 1e-12)
            if (mg[:, 0] < 0).any() or (mg[:, 1:] < -1e-12).any() or (mg[:, 1:] > 1 + 1e-12).any():
                raise W1Stop(f"PLUMBING margins_range {md}.{p}")
            marg[(md, p)] = mg
    return {"marg": marg, "w": w, "mains": mains, "keys": keys, "hits": hits, "catch_key": catch_key, "kd": kd}


def sums_of(m):
    """SUMS (N, A, X, Y) = (D, po*D, p1*D, q1*D) for margins rows m[..., 4] = (D, po, p1, q1)."""
    import numpy as np

    m = np.asarray(m, dtype=float)
    return np.stack([m[..., 0], m[..., 1] * m[..., 0], m[..., 2] * m[..., 0], m[..., 3] * m[..., 0]], axis=-1)


def diag_toys():
    """Toy checks of the l0diag helpers (also in PC1). Run inside the container, after the real-data plumbing step."""
    import copy

    import numpy as np

    t = {}

    def stops(f):  # True iff f() raises a PLUMBING stop
        try:
            f()
        except W1Stop as e:
            return str(e).startswith("PLUMBING")
        return False

    t["d_hits_of"] = hits_of("0110", 4).tolist() == [False, True, True, False]
    t["d_hits_of_rejects_dash"] = stops(lambda: hits_of("01-0", 4))
    t["d_hits_of_rejects_short"] = stops(lambda: hits_of("011", 4))
    t["d_hits_of_rejects_char"] = stops(lambda: hits_of("0120", 4))
    x, y = [True, True, False, False], [True, False, True, False]
    t["d_wtable_unit_eq_table_of"] = bool((wtable(x, y, [1, 1, 1, 1]) == table_of(x, y)).all())
    t["d_wtable_weighted"] = wtable(x, y, [0.5, 0.5, 1.0, 2.0]).tolist() == [4.0, 2.5, 1.0, 1.5]
    # real structure: main m1 shared by two origins, main m2 with two cells in one origin, main m3 alone
    cbo = {"o1": {"c1": "m1", "c2": "m2", "c3": "m2"}, "o2": {"c4": "m1", "c5": "m3"}}
    t["d_inv_nmain_shared_main"] = inv_nmain(cbo) == {("o1", "c1"): 0.5, ("o1", "c2"): 0.5, ("o1", "c3"): 0.5,
                                                      ("o2", "c4"): 0.5, ("o2", "c5"): 1.0}
    t["d_inv_nmain_within_origin"] = inv_nmain(cbo, within_origin=True) == {("o1", "c1"): 1.0, ("o1", "c2"): 0.5,
                                                                            ("o1", "c3"): 0.5, ("o2", "c4"): 1.0,
                                                                            ("o2", "c5"): 1.0}
    # weights below 1: the integer L0 fails while the weighted check passes on the same consistent row
    wf = np.array([0.5, 0.5, 0.5])
    xs, ys = np.array([True, False, True]), np.array([True, False, False])
    ws = wtable(xs, ys, wf)
    lp = l0_parts(ws, table_of(xs, ys))
    t["d_l0_int_fails_on_fractional_weights"] = (not lp["L0_int"]) and (not lp["N_integral"])
    # the same row as stored margins (D, po, p1, q1) = (1.5, 2/3, 2/3, 1/3): back to SUMS it matches W1's weighted table
    t["d_l0w_passes_on_fractional_weights"] = allclose(sums_of(np.array([1.5, 1.0 / 1.5, 1.0 / 1.5, 0.5 / 1.5])),
                                                       wtable(xs, ys, wf), SUM_TOL_REL * 3)
    sd0 = np.array([10.0, 8.0, 6.0, 4.0])
    for name, s in (("exact", [10.0, 8.0, 6.0, 4.0]), ("swapped", [10.0, 8.0, 4.0, 6.0]), ("n_off", [11.0, 8.0, 6.0, 4.0]),
                    ("frac", [10.0, 8.5, 6.0, 4.0]), ("near", [10.0, 8.0 + 1e-10, 6.0, 4.0])):
        t[f"d_l0_parts_eq_l0_int_ok_{name}"] = l0_parts(np.array(s), sd0)["L0_int"] == l0_int_ok(np.array(s), sd0)
    lp = l0_parts(np.array([10.0, 8.0, 4.0, 6.0]), sd0)
    t["d_l0_parts_swapped"] = (lp["N_eq_stageD_n"] and lp["A_eq_ok"] and not lp["X_eq_ok"] and not lp["Y_eq_ok"]
                               and lp["X_eq_swapped"] and lp["Y_eq_swapped"])
    # Stage D: a non-finite max_minus_tau entry is kept with its caught flag, flagged, and counted in N
    ents = [{"origin": "o", "cell": "c1", "main": "m1", "arm": a, "caught": cg, "max_minus_tau": mmt}
            for a, cg, mmt in (("run1", True, 0.5), ("run2", True, 0.2), ("sim", False, -0.1))]
    ents += [{"origin": "o", "cell": "c2", "main": "m2", "arm": a, "caught": False, "max_minus_tau": None}
             for a in ("run1", "run2", "sim")]
    cells, caught, main_of, nonfin, ndis = stage_d_row(ents, "o")
    t["d_stage_d_row_nonfinite_kept"] = (cells == ["c1", "c2"] and caught["run1"].tolist() == [True, False]
                                         and nonfin["run1"].tolist() == [False, True] and ndis == 0
                                         and main_of == {"c1": "m1", "c2": "m2"}
                                         and table_of(caught["sim"], caught["run1"]).tolist() == [2.0, 1.0, 0.0, 1.0])
    t["d_stage_d_row_stops_two_mains"] = stops(lambda: stage_d_row(ents + [dict(ents[0], arm="x", main="m9")], "o"))
    t["d_stage_d_row_stops_foreign_origin"] = stops(lambda: stage_d_row([dict(ents[0], origin="o2")], "o"))
    t["d_stage_d_row_stops_missing_arm"] = stops(lambda: stage_d_row(ents[:5], "o"))
    allc = np.ones(3, dtype=bool)
    t["d_ceiling_kappa_undefined"] = bool(np.isnan(kappa_of(table_of(allc, allc))))
    base = {"idx0_eq_record_draw0": True, "D0_eq_row_weight_sum": True, "margins0_eq_row_hits": True, "hits_agree": True,
            "weights_eq_inv_nmain": True, "weights_all_one": False, "L0w": True}
    t["d_classify_W"] = classify(base) == "W"
    t["d_classify_a"] = classify(dict(base, hits_agree=False)) == "a"
    t["d_classify_b"] = classify(dict(base, weights_eq_inv_nmain=False)) == "b"
    t["d_classify_c"] = classify(dict(base, idx0_eq_record_draw0=False)) == "c"
    t["d_classify_other_both"] = classify(dict(base, hits_agree=False, weights_eq_inv_nmain=False)) == "other"
    t["d_classify_other_unit_agree"] = classify(dict(base, weights_all_one=True)) == "other"
    t["d_classify_other_row_inconsistent"] = classify(dict(base, D0_eq_row_weight_sum=False)) == "other"
    t["d_allclose"] = (allclose([1.0, 2.0], [1.0, 2.0 + 1e-13], 1e-12) and not allclose([1.0], [1.0 + 1e-6], 1e-12)
                       and not allclose([np.nan], [np.nan], 1e-12) and not allclose([1.0, 2.0], [1.0], 1.0))
    t["d_sums_of"] = sums_of(np.array([[8.0, 0.75, 0.5, 0.25]])).tolist() == [[8.0, 6.0, 4.0, 2.0]]
    # a toy row-store object through row_fields: a good one, then each PLUMBING case
    kj = {"D": np.array([2.0, 2.0, 2.0])}
    for a in ("run2", "sim"):
        kj[f"{a}|po"], kj[f"{a}|p1"], kj[f"{a}|q1"] = np.full(3, 0.5), np.full(3, 0.5), np.full(3, 1.0)
    kd = {a: {"po": 0.5, "p1": 0.5, "q1": 1.0} for a in ("run2", "sim")}
    good = {"_kappa": {"joint": kj}, "cohort_cell_keys": ["k1", "k2"], "kappa_draw0_direct": kd,
            "cell_hits_draw0": {"weight": [1.0, 1.0], "mains": ["m1", "m2"],
                                "arms_def": {a: {"catch_key": a + "@B"} for a in HIT_ARMS},
                                "levels": {LEVEL: {"run1": "11", "run2": "10", "sim": "10"}}}}
    rec_row = {"cohort_cell_keys": ["k1", "k2"], "kappa_draw0_direct": copy.deepcopy(kd)}
    lay = {"joint": {p: {"D": ["_kappa", "joint", "D"], "po": ["_kappa", "joint", ARM_OF_PAIR[p] + "|po"],
                         "p1": ["_kappa", "joint", ARM_OF_PAIR[p] + "|p1"], "q1": ["_kappa", "joint", ARM_OF_PAIR[p] + "|q1"]}
                     for p in PAIRS}}
    rf = row_fields(good, rec_row, lay, ["joint"])
    own = wtable(rf["hits"]["run2"], rf["hits"]["run1"], rf["w"])
    t["d_row_fields_good"] = (rf["hits"]["run2"].tolist() == [True, False]
                              and allclose(rf["marg"][("joint", "run2_vs_run1")][0, 1:], rf["kd"]["run2"], MARGIN_TOL)
                              and allclose(rf["marg"][("joint", "run2_vs_run1")][0, 1:], own[1:] / own[0], MARGIN_TOL))

    def bad(edit):
        o = copy.deepcopy(good)
        edit(o)
        return o

    t["d_row_fields_missing_direct"] = stops(lambda: row_fields(bad(lambda o: o.pop("kappa_draw0_direct")), rec_row, lay,
                                                                ["joint"]))
    t["d_row_fields_missing_margin"] = stops(lambda: row_fields(bad(lambda o: o["_kappa"]["joint"].pop("sim|q1")), rec_row,
                                                                lay, ["joint"]))
    t["d_row_fields_missing_level"] = stops(lambda: row_fields(bad(lambda o: o["cell_hits_draw0"]["levels"].pop(LEVEL)),
                                                               rec_row, lay, ["joint"]))
    t["d_row_fields_dash_in_main_arm"] = stops(lambda: row_fields(
        bad(lambda o: o["cell_hits_draw0"]["levels"][LEVEL].update(run1="1-")), rec_row, lay, ["joint"]))
    t["d_row_fields_identity_direct"] = stops(lambda: row_fields(
        good, dict(rec_row, kappa_draw0_direct={a: {"po": 0.25, "p1": 0.5, "q1": 1.0} for a in ("run2", "sim")}), lay,
        ["joint"]))
    t["d_row_fields_identity_cells"] = stops(lambda: row_fields(good, dict(rec_row, cohort_cell_keys=["k2", "k1"]), lay,
                                                               ["joint"]))
    t["d_row_fields_weight_length"] = stops(lambda: row_fields(bad(lambda o: o["cell_hits_draw0"].update(weight=[1.0])),
                                                               rec_row, lay, ["joint"]))
    # Codex round 17: no silent coercion; a numeric string in the weights and a None inside a margin array both STOP
    t["d_row_fields_weight_str"] = stops(lambda: row_fields(bad(lambda o: o["cell_hits_draw0"].update(weight=["1.0", 1.0])),
                                                            rec_row, lay, ["joint"]))
    t["d_row_fields_margin_none"] = stops(lambda: row_fields(bad(lambda o: o["_kappa"]["joint"].update(D=[2.0, None, 2.0])),
                                                             rec_row, lay, ["joint"]))
    return {k: bool(v) for k, v in t.items()}


def v11_toys():
    """PC1' toys for code v11, on real-structure inputs: Stage D per_task entries for two origins with a main shared across them
    (cell weight 0.5 < 1), a main with two cells in one origin, a non-finite max_minus_tau cell, a ceiling table, and a row-store
    object whose `<arm>|p1` key holds run 1's rate and `<arm>|q1` the arm's (p1 != q1), read through the v11 and the v10 mappings."""
    import numpy as np

    t = {}

    def ents(o, cell, main, r1, r2, sim, mmt=0.5):
        return [{"origin": o, "cell": cell, "main": main, "arm": a, "caught": c, "max_minus_tau": (mmt if c else -mmt)
                 if mmt is not None else None} for a, c in (("run1", r1), ("run2", r2), ("sim", sim))]

    # o1: c1 (main m1), c2 and c3 (main m2); o2: c4 (main m1, shared with o1), c5 (main m3; max_minus_tau non-finite, not caught)
    e1 = ents("o1", "c1", "m1", True, True, True) + ents("o1", "c2", "m2", False, True, False) + ents("o1", "c3", "m2", True, False, True)
    e2 = ents("o2", "c4", "m1", True, False, True) + ents("o2", "c5", "m3", False, False, False, mmt=None)
    rows = {"o1": stage_d_row(e1, "o1"), "o2": stage_d_row(e2, "o2")}
    wd = inv_nmain({o: rows[o][2] for o in rows})
    tab = {}
    for o, (cells, caught, _, _, _) in rows.items():
        w = np.array([wd[(o, c)] for c in cells])
        tab[o] = wtable(caught["run2"], caught["run1"], w)
    # o1, weights 0.5 each: N 1.5; agree c1 -> A 0.5; run2 caught c1, c2 -> X 1.0; run1 caught c1, c3 -> Y 1.0
    # o2, weights 0.5 (c4) and 1 (c5): N 1.5; agree c5 -> A 1.0; run2 caught none -> X 0; run1 caught c4 -> Y 0.5
    t["v11_weighted_tables_shared_main"] = tab["o1"].tolist() == [1.5, 0.5, 1.0, 1.0] and tab["o2"].tolist() == [1.5, 1.0, 0.0, 0.5]
    pooled = tab["o1"] + tab["o2"]
    t["v11_pooled_n_is_distinct_mains"] = float(pooled[0]) == 3.0 == len({mn for o in rows for mn in rows[o][2].values()})
    t["v11_nonfinite_cell_counted"] = (rows["o2"][3]["run1"].tolist() == [False, True] and rows["o2"][0] == ["c4", "c5"]
                                       and float(tab["o2"][0]) == 1.5)
    t["v11_degenerate_ceiling_and_undefined"] = (is_degenerate([3.0, 3.0, 3.0, 3.0]) and is_degenerate([4.0, 2.0, 4.0, 2.0])
                                                 and is_degenerate([4.0, 4.0, 0.0, 0.0]) and not is_degenerate(tab["o1"])
                                                 and not is_degenerate([1.5, 1.0, 0.5, 0.5]))
    # orientation: the row store's run2|p1 holds run 1's rate (1/3) and run2|q1 the arm's (0) for o2; D = 1.5, po = 2/3
    kj = {"D": np.full(2, 1.5), "run2|po": np.full(2, 1.0 / 1.5), "run2|p1": np.full(2, 0.5 / 1.5), "run2|q1": np.zeros(2)}
    new_spec = {"D": ["_kappa", "joint", "D"], "po": ["_kappa", "joint", "run2|po"], "p1": ["_kappa", "joint", "run2|q1"],
                "q1": ["_kappa", "joint", "run2|p1"]}
    old_spec = dict(new_spec, p1=["_kappa", "joint", "run2|p1"], q1=["_kappa", "joint", "run2|q1"])
    obj = {"_kappa": {"joint": kj}}
    t["v11_orientation_new_mapping_passes"] = l0w_ok(extract_sums(obj, new_spec)[0], tab["o2"], 2)
    t["v11_orientation_old_mapping_fails"] = not l0w_ok(extract_sums(obj, old_spec)[0], tab["o2"], 2)
    s0 = np.array([1.5, 1.0, 0.0, 0.5])
    t["v11_l0w_tolerance"] = l0w_ok(s0 + 1e-12, s0, 2) and not l0w_ok(s0 + np.array([0.0, 1e-6, 0.0, 0.0]), s0, 2)
    t["v11_l0w_n_off_by_one_fails"] = not l0w_ok(s0 + np.array([1.0, 0.0, 0.0, 0.0]), s0, 2)
    t["v11_ctrl_c2_vector"] = ctrl_vectors(3)["C2"] == (6.0, 0.0, 3.0, 3.0) and ctrl_vectors(3)["L0"] == (1.0, 0.0, 1.0, 0.0)
    t["v11_ctrl_c2_moves_kappa"] = float(kappa_of(pooled + np.array(ctrl_vectors(3)["C2"]))) <= -0.2
    return {k: bool(v) for k, v in t.items()}


def run_l0diag(inputs, man, layout):
    """Booleans-only diagnosis of PC2's integer-L0 failure (pre-statement v11 section 11). Order: one real row first (PLUMBING:
    keys and identity against the kappa record), then the helper toys, then Stage D, all 24 rows, the booleans, the patterns,
    PC2's replication and the four positive controls. Exit 0 = plumbing, toys, replication and controls all as stated (the
    patterns are the result, whatever they are); 3 = toys, replication or a control not as stated; 1 = STOP; 2 = error."""
    import numpy as np

    stage("l0diag_layout")
    modes = check_layout(layout)
    sd, kp = inputs["stageD"], inputs["kappa"]
    lines, rec = [], {}
    stage("l0diag_preconditions")
    origins = list(sd["meta"]["pertask"]["sets"]["pooled_all_origins"])
    origin_table = layout["identity"]["origin_table"]
    if not (len(origins) == 4 and len(set(origins)) == 4 and sorted(origin_table) == sorted(origins)
            and len(origin_table) == 4 and sorted({oshort(o) for o in origins}) == ["S_new", "S_newmains", "S_old",
                                                                                     "S_paper493"]):
        raise W1Stop("PLUMBING origin_table")
    fvo = {}
    for m in SIX:
        vals = [dig(kp, "rows", m, o, "full_variant_of") for o in origins]
        if not (all(isinstance(v, str) for v in vals) and len(set(vals)) == 1):
            raise W1Stop(f"PLUMBING full_variant_of {m}")
        fvo[m] = vals[0]
    if len(set(fvo.values())) != len(SIX):
        raise W1Stop("PLUMBING full_variant_of_not_one_to_one")

    # ---- 1. the first real step: one matched row-store pickle, its keys, and its identity against the kappa record
    stage("plumbing")
    first = None
    for i, f in enumerate(man):
        obj = safe_load(os.path.join(RS_MOUNT, RS_DIR, f["file"]))
        top = obj if isinstance(obj, dict) else {}
        hit, _, _ = match_rows([(f["file"], top.get("label"), top.get("readout"), top.get("origin"))], fvo, origin_table)
        if hit:
            (m, o), = hit.keys()
            first = (i, m, o, row_fields(obj, dig(kp, "rows", m, o), layout, modes))
            break
        del obj
    if first is None:
        raise W1Stop("PLUMBING no_matching_row_file")
    i, m, o, rf = first
    idx0 = {p: allclose(rf["marg"][("joint", p)][0, 1:], rf["kd"][ARM_OF_PAIR[p]], MARGIN_TOL) for p in PAIRS}
    rec["plumb"] = {"files_read": i + 1, "member": m, "origin": oshort(o), "fields_ok": True, "pickle_eq_record": True,
                    "idx0_eq_record_draw0": idx0}
    lines.append(f"W1_PLUMB files_read={i + 1} member={json.dumps(m)} origin={oshort(o)} fields_ok=true pickle_eq_record=true "
                 f"idx0_eq_record_draw0={json.dumps(idx0, sort_keys=True)}")
    del first, rf

    # ---- 2. the helper toys (in the container)
    stage("l0diag_toys")
    toys = diag_toys()
    n_toy = sum(toys.values())
    rec["toys"] = toys
    lines.append(f"W1_DIAG_TOYS pass={n_toy}/{len(toys)} failing={json.dumps(sorted(k for k, v in toys.items() if not v))}")
    if n_toy != len(toys):  # a helper is wrong: stop before the scan
        return {"exit": 3, "lines": lines, "summary": {"PLUMB": [1, 1], "TOYS": [n_toy, len(toys)]}, "record": rec}

    # ---- 3. Stage D rows and W1's own dedup weights
    stage("l0diag_stageD")
    sdr = {(mm, oo): stage_d_row(sd["rows"][mm][oo]["per_task"], oo) for mm in SIX for oo in origins}
    wdd, wdw = {}, {}
    for mm in SIX:
        cbo = {oo: sdr[(mm, oo)][2] for oo in origins}
        wdd[mm], wdw[mm] = inv_nmain(cbo), inv_nmain(cbo, within_origin=True)

    # ---- 4. all 24 rows (09's matching rule), each through row_fields
    stage("l0diag_rows")
    entries, rows = [], {}
    for f in man:
        obj = safe_load(os.path.join(RS_MOUNT, RS_DIR, f["file"]))
        top = obj if isinstance(obj, dict) else {}
        ent = (f["file"], top.get("label"), top.get("readout"), top.get("origin"))
        entries.append(ent)
        hit, _, _ = match_rows([ent], fvo, origin_table)
        for key in hit:
            if key not in rows:
                rows[key] = row_fields(obj, dig(kp, "rows", *key), layout, modes)
        del obj
    rs_file, n_dup, _ = match_rows(entries, fvo, origin_table)
    needed = {(mm, oo) for mm in SIX for oo in origins}
    if set(rs_file) != needed or set(rows) != needed or n_dup:
        raise W1Stop("PLUMBING row_match_not_24_unique")
    lines.append(f"W1_ROWS matched={len(rows)}/24 duplicates={n_dup}")

    # ---- 5. alignment: the record's per-cell arrays are in sorted cohort_cell_keys order, the same cells as Stage D's
    stage("l0diag_align")
    al = {}
    for (mm, oo), rf in rows.items():
        cells, _, main_of, _, _ = sdr[(mm, oo)]
        al[(mm, oo)] = {"cohort_cells_sorted": rf["keys"] == sorted(rf["keys"]),
                        "cohort_cells_eq_stageD": sorted(rf["keys"]) == cells,
                        "mains_align_stageD": rf["mains"] == [main_of[c] for c in cells]}
    bad = sorted(f"{k}|{mm}|{oshort(oo)}" for (mm, oo), d in al.items() for k, v in d.items()
                 if not v and k != "mains_align_stageD")
    lines.append(f"W1_ALIGN rows={len(al)} cells_sorted_and_eq_stageD={'true' if not bad else 'false'} "
                 f"mains_align_stageD={sum(d['mains_align_stageD'] for d in al.values())}/{len(al)} failing={json.dumps(bad)}")
    if bad:
        raise W1Stop("PLUMBING alignment")

    # ---- 6. per-row booleans and counts
    stage("l0diag_compute")
    R, WD = {}, {}
    for (mm, oo), rf in rows.items():
        cells, caught, _, nonfin, _ = sdr[(mm, oo)]
        w = rf["w"]
        WD[(mm, oo)] = np.array([wdd[mm][(oo, c)] for c in cells])
        ww = np.array([wdw[mm][(oo, c)] for c in cells])
        r = {"weights_all_one": bool((w == 1.0).all()),
             "weights_eq_inv_nmain": allclose(w, WD[(mm, oo)], MARGIN_TOL),
             "weights_eq_inv_nmain_within_origin": allclose(w, ww, MARGIN_TOL),
             "stageD_has_nonfinite_mmt": bool(any(nonfin[a].any() for a in HIT_ARMS)),
             "ceiling_all3": bool(all(rf["hits"][a].all() for a in HIT_ARMS)),
             "mains_align_stageD": al[(mm, oo)]["mains_align_stageD"],
             "n_nonfinite_mmt_entries": int(sum(int(nonfin[a].sum()) for a in HIT_ARMS))}
        sdc = sd["rows"][mm][oo].get("draw0_catch")
        for a in HIT_ARMS:
            mis = rf["hits"][a] != caught[a]
            r[f"hits_eq_stageD|{a}"] = not bool(mis.any())
            r[f"n_hit_mismatch|{a}"] = int(mis.sum())
            r[f"n_hit_mismatch_on_nonfinite_mmt|{a}"] = int((mis & nonfin[a]).sum())
            v = sdc.get(rf["catch_key"][a]) if isinstance(sdc, dict) else None
            if not _finite(v):
                raise W1Stop(f"PLUMBING stageD_draw0_catch_missing {a}")
            r[f"stageD_catch_eq_unweighted|{a}"] = abs(float(v) - float(caught[a].mean())) <= MARGIN_TOL
            r[f"stageD_catch_eq_weighted|{a}"] = abs(float(v) - float((w * caught[a]).sum() / w.sum())) <= MARGIN_TOL
            r[f"stageD_catch_eq_row_hits_weighted|{a}"] = abs(float(v) - float((w * rf["hits"][a]).sum() / w.sum())) <= MARGIN_TOL
        R[(mm, oo)] = r
    O = {}
    for oo in origins:
        w0, k0 = rows[(SIX[0], oo)]["w"], rows[(SIX[0], oo)]["keys"]
        O[oo] = {"weights_identical_across_6": all(np.array_equal(rows[(mm, oo)]["w"], w0) and rows[(mm, oo)]["keys"] == k0
                                                   for mm in SIX)}

    # ---- 7. per-check booleans (mode x row x pair) and the pattern of each integer-L0 failure
    C = {}
    for md in modes:
        for (mm, oo), rf in rows.items():
            cells, caught, _, _, _ = sdr[(mm, oo)]
            n = len(cells)
            tol = SUM_TOL_REL * max(1.0, n)
            for p in PAIRS:
                arm = ARM_OF_PAIR[p]
                M = rf["marg"][(md, p)]
                s0 = sums_of(M[0])
                sd_t = table_of(caught[arm], caught["run1"])
                c = l0_parts(s0, sd_t)
                c["L0_int_eq_l0_int_ok"] = c["L0_int"] == l0_int_ok(s0, sd_t)
                c["idx0_eq_record_draw0"] = allclose(M[0, 1:], rf["kd"][arm], MARGIN_TOL)
                c["idxlast_eq_record_draw0"] = allclose(M[-1, 1:], rf["kd"][arm], MARGIN_TOL)
                c["D0_eq_row_weight_sum"] = abs(float(M[0, 0]) - float(rf["w"].sum())) <= tol
                own = wtable(rf["hits"][arm], rf["hits"]["run1"], rf["w"])
                c["margins0_eq_row_hits"] = allclose(M[0, 1:], own[1:] / own[0], MARGIN_TOL)
                c["margins0_eq_row_hits_swapped"] = allclose(M[0, [1, 3, 2]], own[1:] / own[0], MARGIN_TOL)
                c["hits_agree"] = R[(mm, oo)][f"hits_eq_stageD|{arm}"] and R[(mm, oo)]["hits_eq_stageD|run1"]
                c["weights_all_one"] = R[(mm, oo)]["weights_all_one"]
                c["weights_eq_inv_nmain"] = R[(mm, oo)]["weights_eq_inv_nmain"]
                c["L0w"] = allclose(s0, wtable(caught[arm], caught["run1"], WD[(mm, oo)]), tol)
                c["L0w_row_weights"] = allclose(s0, wtable(caught[arm], caught["run1"], rf["w"]), tol)
                c["pattern"] = "pass" if c["L0_int"] else classify(c)
                C[(md, mm, oo, p)] = c

    # ---- 8. PC2's integer L0, replicated by key; the S_newmains outcome
    rep = {md: {(mm, oshort(oo), p) for (m_, mm, oo, p), c in C.items() if m_ == md and c["L0_int"]} == PC2_L0_PASS
           for md in modes}
    rep_ok = rep["joint"] and all(c["L0_int_eq_l0_int_ok"] for c in C.values())
    pat = {}
    for md in modes:
        for oo in origins:
            cnts = {k: 0 for k in PATTERNS}
            for (m_, _mm, o_, _p), c in C.items():
                if m_ == md and o_ == oo and c["pattern"] != "pass":
                    cnts[c["pattern"]] += 1
            pat[(md, oo)] = cnts
    o_nm = next(oo for oo in origins if oshort(oo) == "S_newmains")
    nm_pats = [c["pattern"] for (m_, mm, oo, p), c in C.items() if m_ == "joint" and oo == o_nm and c["pattern"] != "pass"]
    outcome = nm_pats[0] if len(nm_pats) == 10 and len(set(nm_pats)) == 1 and nm_pats[0] in ("a", "b", "c") else "other"
    stop_to_09 = outcome in ("a", "other")
    # 09's ask: per row, stageD_has_nonfinite_mmt, and whether it coincides with N_eq being false (any joint check of the row)
    neq_false = {(mm, oo): any(not C[("joint", mm, oo, p)]["N_eq_stageD_n"] for p in PAIRS) for (mm, oo) in rows}
    coincide = {key: R[key]["stageD_has_nonfinite_mmt"] == neq_false[key] for key in rows}

    # ---- 9. positive controls on in-container copies (each predicted to flip its own boolean)
    stage("l0diag_controls")
    ctl = {}

    def org(sn):
        return next(oo for oo in origins if oshort(oo) == sn)

    mm, sn, p = DIAG_CTL["d_plus_1"]
    oo = org(sn)
    rf, cells, caught = rows[(mm, oo)], sdr[(mm, oo)][0], sdr[(mm, oo)][1]
    M = rf["marg"][("joint", p)][:2].copy()
    M[0, 0] += 1.0
    sd_t = table_of(caught[ARM_OF_PAIR[p]], caught["run1"])
    before = (C[("joint", mm, oo, p)]["D0_eq_row_weight_sum"], C[("joint", mm, oo, p)]["N_eq_stageD_n"])
    after = (abs(float(M[0, 0]) - float(rf["w"].sum())) <= SUM_TOL_REL * max(1.0, len(cells)),
             l0_parts(sums_of(M[0]), sd_t)["N_eq_stageD_n"])
    ctl["d_plus_1"] = {"informative": all(before), "as_predicted": not any(after)}
    mm, sn, _ = DIAG_CTL["flip_run1"]
    oo = org(sn)
    c1 = sdr[(mm, oo)][1]["run1"].copy()
    c1[0] = not c1[0]
    n_after = int((rows[(mm, oo)]["hits"]["run1"] != c1).sum())
    # v11 (the reviewer's item): informative is a real boolean, the unperturbed hits_eq_stageD|run1; the flip must then make that
    # boolean false with exactly one mismatching cell
    ctl["flip_run1"] = {"informative": R[(mm, oo)]["hits_eq_stageD|run1"],
                        "as_predicted": n_after == 1 and not bool((rows[(mm, oo)]["hits"]["run1"] == c1).all())}
    mm, sn, p = DIAG_CTL["roll_by_1"]
    oo = org(sn)
    M = rows[(mm, oo)]["marg"][("joint", p)]
    ctl["roll_by_1"] = {"informative": C[("joint", mm, oo, p)]["idx0_eq_record_draw0"],
                        "as_predicted": not allclose(M[1, 1:], rows[(mm, oo)]["kd"][ARM_OF_PAIR[p]], MARGIN_TOL)}
    mm, sn, _ = DIAG_CTL["weights_ones"]
    oo = org(sn)
    ctl["weights_ones"] = {"informative": R[(mm, oo)]["weights_eq_inv_nmain"],
                           "as_predicted": not allclose(np.ones(len(sdr[(mm, oo)][0])), WD[(mm, oo)], MARGIN_TOL)}

    # ---- 10. printed lines (counts per origin) and the Volume record (booleans per key)
    for md in modes:
        for oo in origins:
            ks = [k for k in C if k[0] == md and k[2] == oo]
            cnt_s = " ".join(f"{b}={sum(bool(C[k][b]) for k in ks)}" for b in CHECK_BOOLS)
            lines.append(f"W1_DIAG_CHECK mode={md} origin={oshort(oo)} n={len(ks)} {cnt_s}")
    for oo in origins:
        ks = [k for k in R if k[1] == oo]
        cnt_b = " ".join(f"{b}={sum(bool(R[k][b]) for k in ks)}" for b in ROW_BOOLS)
        cnt_n = " ".join(f"{b}={sum(R[k][b] for k in ks)}" for b in ROW_COUNTS)
        lines.append(f"W1_DIAG_ROW origin={oshort(oo)} n={len(ks)} {cnt_b} {cnt_n} "
                     f"weights_identical_across_6={'true' if O[oo]['weights_identical_across_6'] else 'false'}")
    lines.append(f"W1_NONFINITE rows_with_nonfinite_mmt={sum(r['stageD_has_nonfinite_mmt'] for r in R.values())}/24 "
                 f"rows_with_N_eq_false={sum(neq_false.values())}/24 rows_where_they_coincide={sum(coincide.values())}/24")
    for (md, oo), cnts in pat.items():
        lines.append(f"W1_PATTERN mode={md} origin={oshort(oo)} " + " ".join(f"{k}={v}" for k, v in cnts.items()))
    lines.append("W1_PC2_REPLICATION " + " ".join(f"{md}={'true' if v else 'false'}" for md, v in rep.items())
                 + f" l0_parts_eq_l0_int_ok={'true' if all(c['L0_int_eq_l0_int_ok'] for c in C.values()) else 'false'}")
    for name, v in ctl.items():
        lines.append(f"W1_CONTROL name={name} informative={'true' if v['informative'] else 'false'} "
                     f"as_predicted={'true' if v['as_predicted'] else 'false'}")
    lines.append(f"W1_OUTCOME s_newmains_failures={len(nm_pats)} pattern={outcome} stop_to_09={'true' if stop_to_09 else 'false'}")
    # a control counts only if it was informative (its boolean was true unperturbed) AND flipped as predicted (Codex round 17)
    ok_ctl = all(v["informative"] and v["as_predicted"] for v in ctl.values())
    rec.update({
        "checks": {f"{md}|{mm}|{oshort(oo)}|{p}": c for (md, mm, oo, p), c in C.items()},
        "rows": {f"{mm}|{oshort(oo)}": {**R[(mm, oo)], "N_eq_false_any_pair": neq_false[(mm, oo)],
                                        "nonfinite_coincides_with_N_eq_false": coincide[(mm, oo)], **al[(mm, oo)]}
                 for (mm, oo) in rows},
        "origins": {oshort(oo): v for oo, v in O.items()},
        "patterns": {f"{md}|{oshort(oo)}": v for (md, oo), v in pat.items()},
        "replicates_pc2": rep, "controls": ctl, "outcome": {"pattern": outcome, "stop_to_09": stop_to_09,
                                                            "s_newmains_failures": len(nm_pats)},
    })
    summary = {"PLUMB": [1, 1], "TOYS": [n_toy, len(toys)], "PC2_REPLICATION": [int(rep_ok), 1],
               "CONTROLS": [sum(v["informative"] and v["as_predicted"] for v in ctl.values()), len(ctl)]}
    code = 0 if (n_toy == len(toys) and rep_ok and ok_ctl) else 3
    return {"exit": code, "lines": lines, "summary": summary, "record": rec}


@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=900, startup_timeout=120, scaledown_window=60, retries=0,
              volumes={RS_MOUNT: rs_vol, OUT_MOUNT: out_vol})
def w1(mode: str, blobs: dict, layout_bytes: bytes, meta: dict) -> dict:
    import traceback
    from zoneinfo import ZoneInfo

    t0 = datetime.datetime.now(ZoneInfo("America/Los_Angeles"))
    task = os.environ.get("MODAL_TASK_ID", "notask")
    record = {"meta": meta, "mode": mode, "started_pt": t0.isoformat(), "modal_task_id": task,
              "inputs": INPUTS, "config": {"SIX": SIX, "FIVE": FIVE, "PRINTED": PRINTED, "FLOAT_TOL": FLOAT_TOL,
                                           "PERTURB": PERTURB if mode == "perturb" else None,
                                           "CTRL": CTRL if mode == "perturb" else None, "RS_DIR": RS_DIR,
                                           "RS_VOL_NAME": RS_VOL_NAME, "RS_LISTING_SHA256": RS_LISTING_SHA256,
                                           "L0DIAG": {"LEVEL": LEVEL, "MARGIN_TOL": MARGIN_TOL, "SUM_TOL_REL": SUM_TOL_REL,
                                                      "L0_INT_TOL": L0_INT_TOL, "DIAG_CTL": DIAG_CTL,
                                                      "PC2_L0_PASS": sorted(PC2_L0_PASS)} if mode == "l0diag" else None}}
    lines = [f"W1_START mode={mode} task={task}"]
    summary = {}
    try:
        if mode not in MODES:
            raise W1Stop("bad_mode")
        if mode == "test":
            res = run_test()
        else:
            inputs = load_inputs(blobs)
            lines.append("W1_INPUTS container_sha256_ok=true n=4")
            pin = meta.get("rowstore_manifest_sha256")
            if mode in PIN_MODES and not pin:
                raise W1Stop("mode_requires_rowstore_manifest_sha256")
            man, msha, ok, line = rowstore_manifest(blobs["listing"], pin, mode, task)
            lines.append(line)
            record["rowstore_manifest"] = man
            record["rowstore_manifest_sha256"] = msha
            if not ok:
                raise W1Stop("rowstore_listing_bytes_or_manifest_pin")
            if mode == "schema":
                res = run_schema(inputs, man)
            elif mode == "identity":
                res = run_identity(man)
            else:
                if _sha(layout_bytes) != meta.get("layout_sha256"):
                    raise W1Stop("layout_sha256_mismatch")
                record["layout"] = json.loads(layout_bytes)
                if mode == "l0diag":
                    res = run_l0diag(inputs, man, record["layout"])
                else:
                    res = run_check(inputs, man, record["layout"], perturb=(mode == "perturb"))
        lines += res["lines"]
        record["result"] = res["record"]
        summary = res.get("summary", {})
        code = res["exit"]
    except W1Stop as e:
        lines.append(f"W1_STOP stage={STAGE[0]} reason={e}")
        record["error_traceback"] = traceback.format_exc()
        code = 1
    except Exception as e:  # value-free line; the traceback goes to the Volume only
        lines.append(f"W1_ERROR stage={STAGE[0]} type={type(e).__name__}")
        record["error_traceback"] = traceback.format_exc()
        code = 2
    record["exit"] = code
    record["lines"] = lines
    os.makedirs(os.path.join(OUT_MOUNT, OUT_SUBDIR), exist_ok=True)
    rel = os.path.join(OUT_SUBDIR, f"W1_{mode}_{t0.strftime('%Y-%m-%d_%H%M%S')}_{task}.json")
    body = json.dumps(clean(record), sort_keys=True, allow_nan=False, default=str).encode()
    with open(os.path.join(OUT_MOUNT, rel), "xb") as f:
        f.write(body)
    out_vol.commit()
    lines.append(f"W1_VOLUME vol={OUT_VOL_NAME} path={rel} sha256={_sha(body)}")
    lines.append("W1_RESULT " + json.dumps({"mode": mode, "exit": code, "gates_match_of_total": summary},
                                           sort_keys=True))
    return {"exit": code, "lines": lines}


@app.local_entrypoint()
def main(mode: str = "", layout: str = "", layout_sha256: str = "", rowstore_manifest_sha256: str = ""):
    if mode not in MODES:
        print(f"W1_CLIENT_STOP bad_mode modes={','.join(MODES)}")
        sys.exit(2)
    if mode in PIN_MODES and not rowstore_manifest_sha256:
        print("W1_CLIENT_STOP mode_requires_rowstore_manifest_sha256")
        sys.exit(2)
    me = os.path.abspath(__file__)
    here = os.path.dirname(me)
    git = ["git", "-C", here]

    def g(*a):
        r = subprocess.run(git + list(a), capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    def committed(path):  # the file equals its blob at HEAD and has no local change
        b = g("hash-object", path)
        h = g("rev-parse", f"HEAD:{g('rev-parse', '--show-prefix') or ''}{os.path.basename(path)}")
        return b, b is not None and b == h and g("status", "--porcelain", "--", path) == ""

    head = g("rev-parse", "HEAD")
    listing_path = os.path.join(here, RS_LISTING_FILE)
    blob_file, ok_script = committed(me)
    _, ok_listing = committed(listing_path)
    ok = head is not None and ok_script and ok_listing
    print(f"W1_CLIENT script_blob={blob_file} committed_blob_match={'true' if ok else 'false'}")
    if not ok:
        print("W1_CLIENT_STOP launch_only_from_the_committed_blob")
        sys.exit(2)
    with open(me, "rb") as f:
        script_bytes = f.read()
    meta = {"git_head": head, "script_blob": blob_file, "script_sha256": _sha(script_bytes), "script_path": me,
            "client_started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "layout_path": layout or None, "layout_sha256": layout_sha256 or None,
            "rowstore_manifest_sha256": rowstore_manifest_sha256 or None}
    blobs = {}
    if mode != "test":
        for name, spec in INPUTS.items():
            b = subprocess.run(git + ["show", f"{spec['commit']}:{spec['path']}"], check=True, capture_output=True).stdout
            ok = _sha(b) == spec["sha256"]
            print(f"W1_INPUT {name} commit={spec['commit'][:12]} sha256_ok={'true' if ok else 'false'}")
            if not ok:
                sys.exit(2)
            blobs[name] = b
        with open(listing_path, "rb") as f:
            blobs["listing"] = f.read()
        ok = _sha(blobs["listing"]) == RS_LISTING_SHA256
        print(f"W1_INPUT rowstore_listing sha256_ok={'true' if ok else 'false'}")
        if not ok:
            sys.exit(2)
    layout_bytes = b""
    if mode in LAYOUT_MODES:
        if not layout or not layout_sha256:
            print("W1_CLIENT_STOP layout_and_layout_sha256_required")
            sys.exit(2)
        with open(layout, "rb") as f:
            layout_bytes = f.read()
        if _sha(layout_bytes) != layout_sha256:
            print("W1_CLIENT_STOP layout_sha256_mismatch")
            sys.exit(2)
    print(f"W1_CLIENT mode={mode} git_head={head} script_sha256={meta['script_sha256']}")
    res = w1.remote(mode, blobs, layout_bytes, meta)
    for line in res["lines"]:
        print(line)
    sys.exit(res["exit"])
