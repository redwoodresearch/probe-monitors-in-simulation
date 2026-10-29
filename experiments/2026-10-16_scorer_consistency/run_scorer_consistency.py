"""(b) Scorer-consistency check, the release gate for the pooled headline (0d / 09 via 2e, 10-16; CUT fork). KEYS, IDS AND SHAS ONLY, plus
09's two diagnostics as a maximum and counts per probe x folder pair (never a per-unit value). Modal CPU, one detached driver.
(b) certifies the WEIGHTS each folder was scored with (from the bytes the scorers loaded and re-hashed), not the scoring code.

WHAT GATES (gate_pass): the controls; Part 1 for the six LEAD probes (sol_a_uncut = reselect, FIBS / harmfulness / refusal / high-stakes
|uncut, own_all_rs_uncut_L48); Part 2's REQUIRED uncut pairs of those six: their pinned pairs (the R and C readouts the lead reads). The p2
call_end_content pairs are INFORMATION: 09 re-scoped 601b6c19c:1761 to the readout the headline actually uses, and no lead label reads
call_end_content (plan() STOPs if INPUTS_FULL / IL ever pins one; then the switch CALL_END_CONTENT_GATES applies). Everything else
(sol_a_uncut_pinned, the other probes, the 32k line) is reported beside and does not gate. 09's rulings: NOTE_09_rulings_2026-10-16.md (verbatim, with READ_LOG commit:line).

LEAD POOLS are defined by the pinned override KEYS (INPUTS_FULL 110447's fits: sol_a_uncut <- "sola_uncut_reselect/sol_a_uncut__{avg_top2,
cmd_A}.npz", whose S_old entries live in solu_reselect/), not by physical directory. 09 (B6): every folder feeding one lead pool, in every set
and every physical scorer directory, must carry ONE weights checkpoint per fold, and (sol_a_uncut, own_all) that checkpoint must equal the
registered FIT record's (INPUTS_FULL fits, sha256-checked); the sol_a_uncut pool must draw on the reselect variant only (physical dirs
sola_uncut_reselect / solu_reselect) and cover S_old, S_new, S_newmains and S_paper493. pinned vs reselect: information only.

Part 1, weights per folder, from the pinned, sha-checked folder manifests. The scorers write these fields from loaded, re-hashed bytes:
own_all run_own_all.py @ e2003d463:1109-1117 (rows 1199-1204, manifest 1222); sola run_sola_uncut_score.py @ 2bbab1758:211-226, 418-419;
solu run_sold_cutfull_score.py @ 687696e61:192-218, 675-677; p2 run_simh_scores.py @ 2313b6eb1:1347-1357. Weights only decide; provenance
strings (tags, fit stamps, record shas) are reported beside as information:
  p2: the probe's weight-file sha256(s) and layer;  own_all_rs*: per fold the weights, preproc and training-manifest sha256 (all required),
  plus the npz's per-row weights and preproc columns (fold -1 rows: all five models' columns, required) checked against the manifest;
  sol: per fold the fold-model sha256 and the weight-file sha256 map (file names normalized: a dated tag dir -> the base name, _cmd kept).
Part 2, 09's diagnostics over the NATURAL-identical pairs of E2E's token-prefix record (@ 571883e36, "(1) natural"), same line + same scorer
directory + same file only (09's pairing rule). Pinned override files are read sha-checked with their pinned manifests (headline inputs);
others are labelled "not a headline input". Units compare only INSIDE the common prefix d (the in-prefix units of both sides must sit at
equal positions; units past d are counted, not compared) with equal folds (fold-scored probes: present on both sides, raw values in -1..4).
Per pair: the max |delta| over `score` and every `score_by_model` column (a non-finite value equals only an identical one) and the count of
values beyond 09's dtype bound |delta| <= TOL[dtype] * max(1, |x|, |y|) (601b6c19c:1761 item 2; the scorer's dtype by code read: DTYPE);
counts, row counts, hosts. PASS per lead probe = every REQUIRED pair ends "compared" (no unit inside the prefix, no jointly finite value, no
units for a key etc. FAIL a required pair), >= 1 compared pinned pair per readout, identical weights, own_all rows consistent with their
manifests (every own_all file read, fold -1 by-model columns included), and no value beyond the bound; errors, fold or in-prefix position
differences and one-sided score_by_model fail any pair of the group.
CONTROLS (all must pass): c1 FIBS single checkpoint; c2 a planted second p2 checkpoint flagged; c3 a planted reselect x pinned pair excluded;
c4 every uncut batch with a compared pair has a passing self / planted-delta control (a jointly finite unit moved by 10x its bound); c5 a
planted own_all fold-weight change flagged; c6 a planted sol fold-model change flagged against its fit record; c7 synthetic cases through
the checks themselves (fold 4.5 / -2 / short columns flagged, NaN vs 1.0 beyond the bound, identical NaN not compared, folds absent on a
fold-scored probe, positions differing inside the prefix). 32k: the directory above the folder is not always the scorer directory there
(pass*/stamp dirs); 32k rows are reported, never gating.

  modal run --detach experiments/2026-10-16_scorer_consistency/run_scorer_consistency.py [--dry]
"""

import hashlib
import io
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # in a Modal container the script sits at /root (REPO is used locally only)
INPUTS_FULL = ("927fabb0d", "experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-16_110447.json",
               "d6923a75aa8f914cd3b2e92d8403f125dd095219997aebea8b1d6e4e2b67f77c")
IL_COMMIT = "09bc5356f"
IL = {"IL:S_old|R": ("2026-10-16_121635", "392c113ea8c75cda4a57489b9ec15dd5864cb63b59a5207acb33ac45fe096cd3"),
      "IL:S_old|C": ("2026-10-16_121703", "807cc58e55e239be3405d911a100c4caa91e966c971888ccc7f09777df44958a"),
      "IL:S_new|R": ("2026-10-16_122010", "cad766a1f92c87b47621697dd1e1d0f220dfdcc941354cba343453756633864c"),
      "IL:S_new|C": ("2026-10-16_121937", "5fbae9d47fb4263d1cdd5e834eac0d327afc8e5e9a37e439c7289755911ffa5e"),
      "IL:S_newmains|R": ("2026-10-16_122746", "d573cf76ba9b0bfb17286035803d082152eba99b1bf1dc35148b5be5533cbad9"),
      "IL:S_newmains|C": ("2026-10-16_122814", "559cac256e9f2b67aa78f1559afcc6baa54977b73f243f8228e268e4b007ad27")}
E2E = ("571883e36", "experiments/2026-10-15_e2e_sim_pipeline/results/check_token_prefix_2026-10-16_192643.json",
       "c7b0da2f42cfece95ce1675983173d2c6383c80c92325d152fa10e0dbe06f7a6")
CONV = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
FOUR = ("S_old", "S_new", "S_newmains", "S_paper493")
GROUPS = {"sol_a_uncut": {"keys": {"R": "sola_uncut_reselect/sol_a_uncut__avg_top2.npz", "C": "sola_uncut_reselect/sol_a_uncut__cmd_A.npz"},
                          "dirs": ("sola_uncut_reselect", "solu_reselect"), "fit": "sol_a_uncut", "family": "sol", "gating": True},
          **{f"{p}|uncut": {"keys": {"R": f"p2/{p}__reply_mean.npz", "C": f"p2/{p}__command.npz"}, "dirs": ("p2",), "fit": None, "family": "p2",
                            "gating": True} for p in ("fibs", "harmfulness", "refusal", "highstakes")},
          "own_all_rs_uncut_L48": {"keys": {"R": "own_all_rs_uncut/own_all_rs_uncut_L48__R.npz", "C": "own_all_rs_uncut/own_all_rs_uncut_L48__C.npz"},
                                   "dirs": ("own_all_rs_uncut",), "fit": "own_all_rs_uncut_L48", "family": "own_all", "gating": True},
          "sol_a_uncut_pinned (sensitivity)": {"keys": {"R": "sola_uncut_pinned/sol_a_uncut__avg_top2.npz", "C": "sola_uncut_pinned/sol_a_uncut__cmd_A.npz"},
                                               "dirs": ("sola_uncut_pinned", "solu_pinned"), "fit": "sol_a_uncut_pinned", "family": "sol", "gating": False}}
CALL_END_CONTENT_GATES = False  # 09 (10-16 ~21:4x, via 2e): the gate holds only products that READ call_end_content; no lead label does
CALL_END_PROBES = ("fibs", "harmfulness", "refusal", "highstakes")  # the four p2 lead probes (reported; gated only with the switch on)
TOL = {"float64": 1e-9, "float32": 1e-5}  # 09 (601b6c19c:1761 item 2): |delta| <= TOL[dtype] * max(1, |s|), s = the larger |score| of the pair
DTYPE = {  # the ARITHMETIC dtype of each scorer's score computation (its lowest-precision step), by code read; not the on-disk dtype:
    # float32-computed scores stored as float64 keep the float32 bound (2e's reading of 09; NOTE_09_rulings has the lines)
    "own_all": ("float64", "run_own_all.py @ e2003d463:1186 (X64 = X.astype(np.float64)), 1056 (float64 standardize + matmul), 1200"),
    "sola": ("float32", "run_sola_uncut_score.py @ 2bbab1758:293 (float32 activations), 298 ((X - mu) / sd in float32 before the matmul)"),
    "solu": ("float32", "run_sold_cutfull_score.py @ 687696e61:591 (float32 activations), 596 ((X - mu) / sd in float32 before the matmul)"),
    "p2_linear": ("float32", "p2_refusal_harm/run_score_linear.py @ 2313b6eb1:144 ((X @ v + B) in float32, then cast to float64)"),
    "p2_fibs": ("float32", "p2_fibs_pair/run_fibs_score.py @ 2313b6eb1:141 (float32 logit; sigmoid + layer mean in float64 at "
                           "run_simh_scores.py:369)")}
MUST_COMPARE = (("honest_a_e2|nm305_user_tracking", "honest_b_e2|nm305_user_tracking"), "/conv/cutfull/ep2-nm/p2/ep2_nm_cutfull_v1__honest_{}/{}",
                ("apollo_d2_tight", "fibs", "highstakes", "refusal"))  # E2E via 2e: the rerun's same-raw-dir call_end_content pairs (information)
MUST_COMPARE_LEAD = tuple(k for G in GROUPS.values() if G["gating"] for k in G["keys"].values())  # 2e / 0d: every lead file kind (dir/file)
# the lead reads must be a pinned, same-directory, compared pair of MUST_COMPARE[0] (c8_lead, gating)


def call_end_in(blobs):
    """{source name: sha-checked bytes} -> the sources whose bytes mention call_end anywhere, any field ([] = the lead reads none;
    09 @ c5a4953b3:1786, d614e63b2:1791)."""
    return sorted(n for n, b in blobs.items() if b"call_end" in b)
FOLDS = (-1, 0, 1, 2, 3, 4)
TW, TD = 600, 2400  # worker / driver timeouts (s); the runner stops the app at 45 min
HEX = re.compile(r"[0-9a-f]{64}")
app = modal.App("probe-shift_aswin-surya_sub-scorercheck_b_kill-2026-11-16")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.1.3")
vol = modal.Volume.from_name(CONV).read_only()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=REPO).stdout


def scorer_dir(path):  # the directory above the file's folder (uncut: cutfull*/<batch>/<scorer>/<folder>/<file>)
    p = path.removeprefix("/conv/").removeprefix("conv/").split("/")
    return p[-3] if len(p) >= 4 else None


def line_of(path):
    return "uncut" if "/cutfull" in "/" + path.removeprefix("/conv/") else "32k"


def pair_ok(px, py):  # 09's pairing rule: same line, same scorer directory, same probe file name; never across directories
    return (scorer_dir(px) is not None and scorer_dir(px) == scorer_dir(py) and line_of(px) == line_of(py)
            and px.rsplit("/", 1)[-1] == py.rsplit("/", 1)[-1])


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def wfile_key(p):  # a weight file's normalized name: "cmd/" if its directory ends in _cmd, then the base name
    parts = p.split("/")
    return ("cmd/" if len(parts) > 1 and parts[-2].endswith("_cmd") else "") + parts[-1]


def sol_weights(doc):
    """Fold-model sha256 per fold and the weight-file map of a sol manifest or fit record, from every source in it (never the registered 32k
    control weights). -> ({fold: sha}, {file: sha}, conflicts between sources)"""
    folds, files = defaultdict(set), defaultdict(set)
    for p, v in leaves(doc):
        if not (isinstance(v, str) and HEX.fullmatch(v)) or "registered" in p:
            continue
        m = re.search(r"(?:fold_models?_sha256|fold_models)(?:\.sol_a_uncut)?(?:\.|\[)(\d)\]?$", p)
        if m and re.match(r"^(weights\.|gates\.(sol_a_uncut_weights|weights)\.)", p):
            folds[f"f{m.group(1)}"].add(v)
        elif re.match(r"^weights\.files_sha256\.", p):
            files[wfile_key(p.split(".", 2)[-1])].add(v)
        elif re.match(r"^gates\.(sol_a_uncut_weights\.files|weights\.files_sha256)\.", p):
            files[wfile_key(p.split(".", 3)[-1])].add(v)
    conf = [f"{k}: {len(s)} values" for k, s in list(folds.items()) + list(files.items()) if len(s) > 1]
    return {k: sorted(s)[0] for k, s in folds.items()}, {k: sorted(s)[0] for k, s in files.items()}, conf


def dtype_of(path):  # -> (dtype, cite) of the scorer that wrote an uncut score file (DTYPE, by code read); other lines / dirs: not read
    sd, fn = scorer_dir(path), path.rsplit("/", 1)[-1]
    if line_of(path) != "uncut" or sd is None:
        return "float32", "not read (32k or unknown scorer; information only)"
    k = ("p2_fibs" if fn.startswith("fibs__") else "p2_linear") if sd == "p2" else "own_all" if sd.startswith("own_all") else \
        "sola" if sd.startswith("sola_uncut") else "solu" if sd.startswith("solu_") else None
    return DTYPE[k] if k else ("float32", "not read (other scorer; information only)")


def batch_of(path):  # an uncut score file's cutfull*/<batch>
    return "/".join(path.removeprefix("/conv/").split("/")[:2])


def family(sdir):  # the scorer family that Part 2 / c4 use (deltas, folder_weights): unchanged since 74fba8fd3
    return "p2" if sdir == "p2" else "own_all" if sdir.startswith("own_all") else "sol" if sdir.startswith(("sola_uncut", "solu_")) else "other"


INFO_SOL_DIRS = ("sol", "sol_c", "casu_inherit", "casis_uncut_inherit")  # the S_old e1 / e2 dirs of non-lead keys (09, READ_LOG ccf127800:1812)


def family_info(sdir):  # manifest parsing for Part 1 / info tables / the cross-dir table only: those 4 dirs read as sol-schema manifests
    return "sol" if sdir in INFO_SOL_DIRS else family(sdir)  # (S_old: manifest_sol.json); never used by deltas() or c4


def weights_of(fam, fname, man):
    """-> (weights {fold or "all": canonical}, info {provenance}, missing reason or None). Weights only decide; a missing field fails closed."""
    if fam == "p2":  # the p2 scorer's schema (run_simh_scores.py @ 2313b6eb1:1350, 1357, 1704): a linear probe = {layer, sha256};
        name = fname.split("__")[0]  # FIBS = one weight sha256 per layer, keyed L<k> (the layer is the key)
        pr = (man.get("probes") or {}).get(name) or {}
        ws = {k: v for k, v in pr.items() if k != "role"}  # every field but the role label is validated, nothing dropped first
        hx = all(isinstance(v, str) and HEX.fullmatch(v) for k, v in ws.items() if k != "layer")
        ok = hx and ((bool(ws) and all(re.fullmatch(r"L\d+", k) for k in ws)) if name == "fibs" else (
            set(ws) == {"sha256", "layer"} and isinstance(pr["layer"], int) and not isinstance(pr["layer"], bool)))
        if not ok:
            return {}, {}, f"p2 manifest: probes.{name} lacks its schema (fibs: L<k> weight sha256s; linear: an integer layer + sha256)"
        return {"all": ws}, {
            "record": ((man.get("inputs") or {}).get(f"record_{name}") or {}).get("sha256")}, None
    if fam == "own_all":
        label, ro = fname.removesuffix(".npz").rsplit("__", 1)
        w = {}
        for k in range(5):
            m = (man.get("models") or {}).get(f"{label}|{ro}|f{k}") or {}
            t = {f: m.get(f"{f}_sha256") for f in ("weights", "preproc", "training_manifest")}
            if not all(isinstance(v, str) and HEX.fullmatch(v) for v in t.values()):
                return {}, {}, f"own_all manifest: models[{label}|{ro}|f{k}] lacks a weights / preproc / training_manifest sha256"
            w[f"f{k}"] = t
        ww = man.get("weights") or {}
        return w, {"fit_record_sha256": ww.get("fit_record_sha256") or man.get("fit_record_sha256"), "fit_stamp": ww.get("fit_stamp") or man.get("fit_stamp")}, None
    if fam == "sol":
        folds, files, conf = sol_weights(man)
        if conf:
            return {}, {}, f"sol manifest: its weight sources disagree ({conf[:3]})"
        if sorted(folds) != [f"f{k}" for k in range(5)] or not files:
            return {}, {}, f"sol manifest: fold models {sorted(folds)}, {len(files)} weight files"
        return {f: {"fold_model": folds[f]} for f in folds} | {"all": {"files": files}}, {
            "tag": (man.get("weights") or {}).get("tag"), "fit_record": ((man.get("cites") or {}).get("fit_record") or {}).get("sha256")
            or (((man.get("weights") or {}).get("spec_doc") or {}).get("fit_record") or {}).get("sha256")}, None
    ws = {p: v for p, v in leaves(man) if isinstance(v, str) and HEX.fullmatch(v) and re.search(r"(weight|fold_model|models\.|probes?\.|preproc|training_manifest)", p, re.I)
          and not re.search(r"(staged|^inputs\.|dump|attest|routing|admission|inventory|registered|stage_record|extraction|r36|^files\.|gate_honest_b|corrected_b)", p)}
    return ({"all": ws}, {}, None) if ws else ({}, {}, "no weight sha256 found")


def host_ids(man):
    hs = set()
    for p, v in leaves({k: man.get(k) for k in ("host_cpu", "projector_cpu") if k in man}):
        if re.search(r"(cpu_vendor|cpu_family|cpu_model_number|numpy_simd)", p):
            hs.add(re.sub(r".*[.]", "", re.sub(r"\[\d+\]", "", p)) + "=" + str(v))
    return sorted(hs)


@app.function(image=image, cpu=2, memory=8192, timeout=TW, retries=0, volumes={"/conv": vol})
def read_manifests(items: list) -> dict:
    """[(manifest path, expected sha256, scorer dir, [files])] -> path -> {sha_ok, per file [weights, info, missing]}."""
    out = {}
    for mp, want, sdir, files in items:
        try:
            b = (Path("/conv") / mp).read_bytes()
            man = json.loads(b)
            out[mp] = {"sha_ok": sha(b) == want, "per_file": {f: list(weights_of(family_info(sdir), f, man)) for f in files}}
        except Exception as ex:
            out[mp] = {"error": type(ex).__name__}
    return out


ROW_COLS = ("score", "score_by_model", "token", "fold", "unit", "call_index", "model_weights_sha256", "preproc_sha256",
            "model_weights_sha256_by_model", "preproc_sha256_by_model")  # per-row arrays of a score file (layer / layers are per file)


def short_row_cols(z):
    """Per-row arrays of a score file whose first axis is not len(key) (a 0-d array counts as short); [] = aligned."""
    import numpy as np

    n = len(z["key"])
    return [c for c in ROW_COLS if c in z.files and (np.ndim(z[c]) == 0 or np.shape(z[c])[0] != n)]


def bad_fold_values(f):
    """Raw fold values outside -1..4 (non-integral, non-finite, non-numeric included), checked BEFORE any cast; [] = all valid."""
    import numpy as np

    a = np.asarray(f).reshape(-1)
    if a.dtype.kind not in "iuf":
        return [f"dtype {a.dtype.str}"]
    return sorted({repr(float(v)) for v in a[~np.isin(a, FOLDS)].tolist()})[:5]


def row_identity(z):
    """own_all npz rows -> {used: {fold: {weights, preproc}}, fold -1 coverage} or the reason it can't be read (no rows, per-row columns not
    exactly len(key) long, bad raw folds, missing columns); every row, no truncation."""
    import numpy as np

    need = ("key", "fold", "model_weights_sha256", "preproc_sha256")
    if any(c not in z.files for c in need):
        return {"missing_columns": [c for c in need if c not in z.files]}
    n = len(z["key"])  # the npz's rows: every per-row column (score axis included) must cover exactly these
    if n == 0:
        return {"no_rows": True}
    bw, bp = (z[c] if c in z.files else None for c in ("model_weights_sha256_by_model", "preproc_sha256_by_model"))
    lm = sorted(set([c for c in need[1:] if z[c].ndim != 1] + short_row_cols(z) + ([] if "score" in z.files else ["score"])))
    if lm:
        return {"length_mismatch": lm}
    bf = bad_fold_values(z["fold"])
    if bf:
        return {"bad_folds": bf}
    fo = np.asarray(z["fold"]).astype(np.int64)
    used = defaultdict(lambda: {"weights": set(), "preproc": set()})
    for f_, w_, q_ in zip(fo.tolist(), z["model_weights_sha256"].astype(str).tolist(), z["preproc_sha256"].astype(str).tolist()):
        if f_ >= 0:
            used[f"f{f_}"]["weights"].add(w_)
            used[f"f{f_}"]["preproc"].add(q_)
    m1 = np.flatnonzero(fo == -1)
    cov = None
    if len(m1):
        cov = bool(bw is not None and bp is not None and bw.ndim == 2 and bw.shape[1] == 5 and bp.shape == bw.shape)
        if cov:
            for j in m1.tolist():
                for k_ in range(5):
                    used[f"f{k_}"]["weights"].add(str(bw[j, k_]))
                    used[f"f{k_}"]["preproc"].add(str(bp[j, k_]))
    return {"used": {k: {a: sorted(s_) for a, s_ in v.items()} for k, v in used.items()}, "fold_minus1_rows": int(len(m1)),
            "fold_minus1_five_models": cov}


def row_violations(r, w):
    """Why an own_all npz's rows (row_identity) disagree with its folder manifest's weights w (keys and fold names only); [] = consistent."""
    if "used" not in r:
        return [r.get("error") or ("no rows" if r.get("no_rows") else None) or (f"row folds outside -1..4: {r['bad_folds']}" if "bad_folds" in r else None)
                or (f"per-row columns not len(key) long: {r['length_mismatch']}" if "length_mismatch" in r else None)
                or f"missing per-row columns {r.get('missing_columns')}"]
    why = ["fold -1 rows without the five models' weights / preproc columns"] if r["fold_minus1_rows"] and not r["fold_minus1_five_models"] else []
    for f, u in sorted(r["used"].items()):
        if f not in w:
            why.append(f"{f}: a row's fold is not one of the manifest's f0-f4")
        elif set(u["weights"]) != {w[f]["weights"]} or set(u["preproc"]) != {w[f]["preproc"]}:
            why.append(f"{f}: per-row weights / preproc differ from the folder manifest")
    return why


@app.function(image=image, cpu=2, memory=8192, timeout=TW, retries=0, volumes={"/conv": vol})
def row_weights(items: list) -> dict:
    """own_all npz: [(path, expected sha256)] -> {sha_ok} | row_identity (every row; A4 / A5)."""
    import numpy as np

    out = {}
    for p, want in items:
        try:
            b = (Path("/conv") / p).read_bytes()
            out[p] = {"sha_ok": sha(b) == want} | row_identity(np.load(io.BytesIO(b), allow_pickle=False))
        except Exception as ex:
            out[p] = {"error": type(ex).__name__}
    return out


def diff_arr(a, b):
    """-> (|a - b| per element, n finite pairs). A non-finite value equals only an identical one (both NaN, or the same infinity)."""
    import numpy as np

    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    fin = np.isfinite(a) & np.isfinite(b)
    same_nf = (np.isnan(a) & np.isnan(b)) | (np.isinf(a) & np.isinf(b) & (a == b))
    return np.where(fin, np.abs(np.where(fin, a, 0.0) - np.where(fin, b, 0.0)), np.where(same_nf, 0.0, np.inf)), int(fin.sum())


def over_bound(a, b, tol):
    """-> (|a - b|, n jointly finite, mask beyond 09's bound |a - b| > tol * max(1, |a|, |b|)); unequal non-finite values are always beyond."""
    import numpy as np

    dd, nf = diff_arr(a, b)
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    s = np.maximum(1.0, np.maximum(np.abs(np.where(np.isfinite(a), a, 0.0)), np.abs(np.where(np.isfinite(b), b, 0.0))))
    return dd, nf, dd > tol * s


def unit_delta(zx, zy, kx, ky, d, tol, fam, plant=False, n_tokens=None, line="uncut"):
    """One file pair for one run pair: the units inside the common prefix d, compared over score and every score_by_model column against
    09's dtype bound; a maximum and counts (numbers only). plant=True: the c4 control (one jointly finite score moved by 10x its bound)."""
    import numpy as np

    ix, iy = np.flatnonzero(zx["key"].astype(str) == kx), np.flatnonzero(zy["key"].astype(str) == ky)
    rows_ = {"n_rows_x": int(len(ix)), "n_rows_y": int(len(iy)), "n_rows_file_x": int(len(zx["key"])), "n_rows_file_y": int(len(zy["key"]))}
    if not len(ix) or not len(iy):
        return rows_ | {"status": "no units for a key"}
    if "score" not in zx.files or "score" not in zy.files:
        return rows_ | {"status": "no score array", "arrays_x": sorted(zx.files), "arrays_y": sorted(zy.files)}
    if short_row_cols(zx) or short_row_cols(zy):
        return rows_ | {"status": "per-row arrays not len(key) long", "arrays": [short_row_cols(zx), short_row_cols(zy)]}

    def g(z, i, c):
        return z[c][i] if c in z.files else None
    tx, ty = g(zx, ix, "token"), g(zy, iy, "token")
    coarse = False
    if (tx is None) != (ty is None):
        return rows_ | {"status": "token positions on one side only"}
    if tx is None:  # no positions on either side (32k only): only token-identical runs (d >= both lengths), equal call_index, unit and fold arrays
        same_ = len(ix) == len(iy) and all(g(zx, ix, c) is not None and g(zy, iy, c) is not None and np.array_equal(g(zx, ix, c), g(zy, iy, c))
                                           for c in ("call_index", "unit", "fold") if c in zx.files or c in zy.files)
        if line != "32k" or not same_ or not (n_tokens and d >= max(n_tokens)):
            return rows_ | {"status": "no token positions, and not a 32k token-identical pair with equal call index / unit / fold arrays"}
        coarse, cx, cy = True, ix, iy
    else:  # the units whose every readout lies inside the common prefix; both sides must hold them at equal positions
        if not all((t.dtype.kind in "iu" or (t.dtype.kind == "f" and np.isfinite(t).all() and (t == np.round(t)).all())) and (t >= 0).all() for t in (tx, ty)):
            return rows_ | {"status": "token positions not integral and non-negative"}
        px_, py_ = tx.astype(np.int64).reshape(len(ix), -1), ty.astype(np.int64).reshape(len(iy), -1)
        inx, iny = px_.max(axis=1) + 1 <= d, py_.max(axis=1) + 1 <= d
        if not inx.any() and not iny.any():
            return rows_ | {"status": "no unit inside the common prefix", "n_units": [int(len(ix)), int(len(iy))]}
        if inx.sum() != iny.sum() or not np.array_equal(px_[inx], py_[iny]):
            return rows_ | {"status": "positions differ inside the common prefix"}
        cx, cy = ix[inx], iy[iny]
    fx, fy = g(zx, cx, "fold"), g(zy, cy, "fold")
    if fam in ("sol", "own_all") and (fx is None or fy is None):
        return rows_ | {"status": "fold missing on a fold-scored probe"}
    if (fx is None) != (fy is None) or (fx is not None and not np.array_equal(fx, fy)):
        return rows_ | {"status": "folds differ"}
    if fx is not None and (bad_fold_values(fx) or bad_fold_values(fy)):
        return rows_ | {"status": "fold values outside -1..4"}
    sx, sy = zx["score"][cx].astype(np.float64), zy["score"][cy].astype(np.float64)
    bx, by = g(zx, cx, "score_by_model"), g(zy, cy, "score_by_model")
    if (bx is None) != (by is None) or (bx is not None and bx.shape != by.shape):
        return rows_ | {"status": "score_by_model present on one side only, or shapes differ"}
    planted = None
    if plant:  # one jointly finite value moved by 10x its bound: in score, else in score_by_model
        for a_, b_ in ((sx, sy),) + (((bx, by),) if bx is not None else ()):
            fa, fb = np.asarray(a_, np.float64).reshape(-1), np.asarray(b_, np.float64).reshape(-1)
            j_ = np.flatnonzero(np.isfinite(fa) & np.isfinite(fb))
            if len(j_):
                planted = 10 * tol * max(1.0, abs(float(fa[j_[0]])), abs(float(fb[j_[0]])))
                fa = fa.copy()
                fa[j_[0]] += planted
                if a_ is sx:
                    sx = fa
                else:
                    bx = fa.reshape(np.shape(bx))
                break
        if planted is None:
            return rows_ | {"status": "control: no jointly finite value"}
    dd, nf, ov = over_bound(sx, sy, tol)
    d2, n_sbm, ov2 = over_bound(bx, by, tol) if bx is not None else (None, 0, None)
    allv = dd.reshape(-1) if d2 is None else np.concatenate([dd.reshape(-1), d2.reshape(-1)])
    if nf + n_sbm == 0 and np.isfinite(allv).all():  # only identical non-finite values: nothing was compared (a required pair FAILs)
        return rows_ | {"status": "no jointly finite element", "n_units_in_prefix": int(len(cx))}
    r = rows_ | {"status": "compared", "coarse_identical_runs": coarse, "n_units": [int(len(ix)), int(len(iy))], "n_units_compared": int(len(cx)),
                 "n_units_past_prefix": [int(len(ix) - len(cx)), int(len(iy) - len(cy))], "n_finite_score": nf, "n_finite_by_model": n_sbm,
                 "n_both_nonfinite_score": int(((dd == 0) & ~(np.isfinite(sx) & np.isfinite(sy))).sum()),
                 "max_abs_delta": float(allv.max()), "n_differ": int((dd != 0).sum()), "tol": tol, "n_over_bound": int(ov.sum()),
                 "n_by_model_over_bound": 0 if ov2 is None else int(ov2.sum()), "planted_delta": planted}
    if "model_weights_sha256" in zx.files and "model_weights_sha256" in zy.files:
        r["n_units_weights_sha_differ"] = int((zx["model_weights_sha256"][cx].astype(str) != zy["model_weights_sha256"][cy].astype(str)).sum())
    return r


def folder_weights(path, fname, pin):
    """(weights, missing, ids) of one score file's folder: the PINNED manifest (sha256 checked) when the file is an override entry, else the
    manifest beside it naming the file (labelled unpinned; never a headline input)."""
    sdir = scorer_dir(path)
    if pin:
        b = (Path("/conv") / pin["manifest"]["path"]).read_bytes()
        if sha(b) != pin["manifest"]["sha256"]:
            return {}, "pinned manifest sha256 mismatch", {"pinned": True}
        man, mp = json.loads(b), pin["manifest"]["path"]
    else:
        d = (Path("/conv") / path.removeprefix("/conv/")).parent
        cand = [(m, json.loads(m.read_bytes())) for m in sorted(d.glob("manifest*.json"))]
        cand = [c for c in cand if fname in (c[1].get("files") or {})] or (cand if len(cand) == 1 else [])
        if len(cand) != 1:
            return {}, f"{len(cand)} manifests name the file", {"pinned": False}
        mp, man = str(cand[0][0]).removeprefix("/conv/"), cand[0][1]
    w, info, why = weights_of(family(sdir), fname, man)
    return w, why, {"pinned": bool(pin), "manifest": mp, "hosts": host_ids(man), "info": info}


@app.function(image=image, cpu=4, memory=16384, timeout=TW, retries=0, volumes={"/conv": vol})
def deltas(jobs: list) -> list:
    import numpy as np

    out, ctl = [], []
    for j in jobs:
        base = {k: j[k] for k in ("pair", "line", "x_label", "y_label", "px", "py", "d", "dtype", "call_end_gate")} | {
            "headline_input": bool(j["x_pin"] and j["y_pin"])}
        if not pair_ok(j["px"], j["py"]):
            out.append(base | {"status": "excluded by the pairing rule (different scorer directories)"})
            continue
        if not (j["px"].startswith("/conv/") and j["py"].startswith("/conv/")):  # e.g. 32k /p2/ paths: not on the mounted Volume (information only)
            out.append(base | {"status": "not on the conv Volume (32k information; not read)" if j["line"] == "32k"
                               else "error: an uncut job's file is not on the conv Volume"})
            continue
        try:
            zs = []
            for p, pin in ((j["px"], j["x_pin"]), (j["py"], j["y_pin"])):
                b = (Path("/conv") / p.removeprefix("/conv/")).read_bytes()
                if pin and sha(b) != pin["sha256"]:
                    raise ValueError("pinned score file sha256 mismatch")
                zs.append(np.load(io.BytesIO(b), allow_pickle=False))
            zx, zy = zs
            fam = family(scorer_dir(j["px"]) or "")
            kw = {"n_tokens": j.get("n_tokens"), "line": j["line"]}
            r = unit_delta(zx, zy, j["x_key"], j["y_key"], j["d"], j["tol"], fam, **kw)
            fn_ = j["px"].rsplit("/", 1)[-1]
            (wx, mx_, ix_), (wy, my_, iy_) = folder_weights(j["px"], fn_, j["x_pin"]), folder_weights(j["py"], fn_, j["y_pin"])
            r |= {"weights_same": (wx == wy) if not (mx_ or my_) else None, "weights_missing": [mx_, my_], "ids_x": ix_, "ids_y": iy_,
                  "weights_x_sha256": sha(json.dumps(wx, sort_keys=True).encode()) if wx else None,
                  "weights_y_sha256": sha(json.dumps(wy, sort_keys=True).encode()) if wy else None}
            if fam == "own_all":  # both files' actual rows (fold -1: all five models) against their own manifests (Astra 8)
                r["row_identity_why"] = [row_violations(row_identity(z), w) if not m else [f"no manifest weights ({m})"]
                                         for z, w, m in ((zx, wx, mx_), (zy, wy, my_))]
            if r.get("status") == "compared" and j["line"] == "uncut":  # c4 on BOTH files of every compared uncut job
                for z_, k_, p_ in ((zx, j["x_key"], j["px"]), (zy, j["y_key"], j["py"])):
                    s0 = unit_delta(z_, z_, k_, k_, j["d"], j["tol"], fam, **kw)
                    s1 = unit_delta(z_, z_, k_, k_, j["d"], j["tol"], fam, plant=True, **kw)
                    if s1.get("status") != "control: no jointly finite value":
                        ctl.append({"file": p_, "batch": batch_of(p_), "call_end": p_.endswith("__call_end_content.npz"), "passes": bool(
                            s0.get("status") == "compared" and s0.get("max_abs_delta") == 0 and s0.get("n_differ") == 0
                            and s1.get("status") == "compared" and s1.get("n_over_bound", 0) + s1.get("n_by_model_over_bound", 0) == 1
                            and s1.get("planted_delta") and abs(s1["max_abs_delta"] - s1["planted_delta"]) <= 1e-6 * s1["planted_delta"])})
        except Exception as ex:
            r = {"status": f"error: {type(ex).__name__}" + (": pinned score file sha256 mismatch" if "pinned score file sha256" in str(ex) else "")}
        out.append(base | r)
    return [{"ctl": ctl}] + out


def distinct(rows):
    """rows: [(folder, weights {fold: canonical})] -> {fold: {distinct, folders_per_checkpoint, checkpoints {id: n folders}}}."""
    folds = sorted({f for _, w in rows for f in w} - {"all"}) or ["all"]
    t = {}
    for f in folds:
        ck, rep = defaultdict(set), {}
        for folder, w in rows:
            c_ = {k: w.get(k) for k in (f, "all")}
            i_ = sha(json.dumps(c_, sort_keys=True).encode())[:16]
            ck[i_].add(folder)
            rep.setdefault(i_, c_)  # the weights behind each checkpoint id, stored as the id is computed
        t[f] = {"distinct": len(ck), "folders_per_checkpoint": sorted((len(v) for v in ck.values()), reverse=True),
                "checkpoints": {k: len(v) for k, v in sorted(ck.items(), key=lambda kv: -len(kv[1]))}, "weights": rep}
    return t


def fit_match(fam, fname, w, fit):
    """Per fold: a folder's weights equal the registered fit record's (sol: the fold model, and the folder's weight-file map EQUAL to the fit
    record's tight_used_by_scorer map, same names and sha256s; own_all: (label, readout, fold)'s weights / preproc / training manifest)."""
    if fam == "sol":
        fl = (w.get("all") or {}).get("files", {})
        files_ok = bool(fl) and fl == fit["files"]
        return {f: (w.get(f) or {}).get("fold_model") == fit["folds"].get(f) and files_ok for f in (f"f{k}" for k in range(5))}
    if fam == "own_all":
        label, ro = fname.removesuffix(".npz").rsplit("__", 1)
        return {f"f{k}": w.get(f"f{k}") == fit.get(f"{label}|{ro}|f{k}") for k in range(5)}
    return {}


class _Z(dict):  # a stand-in for a loaded npz (synthetic control only)
    @property
    def files(self):
        return list(self)


def synthetic_checks():
    """c7: planted synthetic cases through the checks themselves -> {case: True when the check catches it (or passes a clean case)}."""
    import numpy as np

    h = "a" * 64
    w = {f"f{k}": {"weights": h, "preproc": h} for k in range(5)}

    def oa(fold, n_cols=None):
        n = len(fold) if n_cols is None else n_cols
        return _Z(key=np.array(["k"] * len(fold)), fold=np.asarray(fold), score=np.zeros(len(fold)), model_weights_sha256=np.array([h] * n),
                  preproc_sha256=np.array([h] * n))

    def sc(score, fold=None, token=None):
        z = _Z(key=np.array(["k"] * len(score)), score=np.asarray(score, np.float64), token=np.arange(len(score)) if token is None else np.asarray(token))
        return z if fold is None else z | {"fold": np.asarray(fold)}

    def u(zx, zy, fam="sol"):
        return unit_delta(_Z(zx), _Z(zy), "k", "k", 100, TOL["float32"], fam)

    return {"fold_4.5_rows_flagged": bool(row_violations(row_identity(oa([0.0, 4.5])), w)),
            "fold_-2_rows_flagged": bool(row_violations(row_identity(oa([0, -2])), w)),
            "short_columns_flagged": bool(row_violations(row_identity(oa([0, 1], n_cols=1)), w)),
            "zero_rows_flagged": bool(row_violations(row_identity(oa([])), w)),
            "short_score_axis_flagged": bool(row_violations(row_identity(_Z(oa([0, 0]) | {"score": np.zeros(1)})), w)),
            "short_score_in_pair_flagged": u(sc([1.0]) | {"key": np.array(["k", "other"]), "fold": np.array([0, 0]), "token": np.array([1, 2])},
                                             sc([1.0], [0])).get("status") == "per-row arrays not len(key) long",
            "rows_beyond_fold_flagged": bool(row_violations(row_identity(_Z(oa([0]) | {"key": np.array(["k", "other"])})), w)),
            "fibs_none_layer_flagged": weights_of("p2", "fibs__reply_mean.npz", {"probes": {"fibs": {"L40": h, "L48": None}}})[2] is not None,
            "linear_layer_string_flagged": weights_of("p2", "refusal__command.npz", {"probes": {"refusal": {"sha256": h, "layer": "48"}}})[2] is not None,
            "fold_less_pair_in_info_sol_dirs_still_compared": all(  # deltas()/c4 for those dirs behave as at 74fba8fd3 (family() unchanged)
                u(sc([1.0]), sc([1.0]), fam=family(d)).get("status") == "compared" for d in INFO_SOL_DIRS),
            "plant_into_by_model": (lambda z_: unit_delta(z_, z_, "k", "k", 100, TOL["float32"], "own_all", plant=True).get("n_by_model_over_bound") == 1)(
                _Z(sc([np.nan], [0]) | {"score_by_model": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])})),
            "bool_fold_one_side_flagged": u(sc([1.0], [1]), sc([1.0], np.array([True]))).get("status") == "fold values outside -1..4",
            "nonintegral_token_flagged": u(sc([1.0], [0], [2.5]), sc([1.0], [0], [2.5])).get("status") == "token positions not integral and non-negative",
            "clean_rows_pass": row_violations(row_identity(oa([0, 1, 2, 3, 4])), w) == [],
            "nan_vs_1_beyond_bound": u(sc([np.nan], [0]), sc([1.0], [0])).get("n_over_bound") == 1,
            "identical_nan_not_compared": u(sc([np.nan], [0]), sc([np.nan], [0])).get("status") == "no jointly finite element",
            "folds_absent_flagged": u(sc([1.0]), sc([1.0])).get("status") == "fold missing on a fold-scored probe",
            "fold_4.5_pair_flagged": u(sc([1.0], [4.5]), sc([1.0], [4.5])).get("status") == "fold values outside -1..4",
            "positions_differ_flagged": u(sc([1.0, 2.0], [0, 0], [3, 5]), sc([1.0, 2.0], [0, 0], [3, 6])).get("status")
            == "positions differ inside the common prefix",
            "past_prefix_unit_skipped": u(sc([1.0, 2.0], [0, 0], [3, 200]), sc([1.0, 9.0], [0, 0], [3, 200])).get("n_units_compared") == 1,
            "within_bound_passes": u(sc([1.0], [0]), sc([1.0 + 1e-7], [0])).get("n_over_bound") == 0,
            "beyond_bound_flagged": u(sc([1.0], [0]), sc([1.0 + 1e-3], [0])).get("n_over_bound") == 1}


@app.function(image=image, cpu=2, memory=16384, timeout=TD, retries=0, volumes={"/conv": vol})
def drive(entries: list, pairs_jobs: list, fits: dict) -> dict:
    import copy

    vol.reload()
    mans = defaultdict(lambda: {"sha": set(), "sdir": None, "files": set()})
    for e in entries:
        m = mans[e["manifest"]["path"]]
        m["sha"].add(e["manifest"]["sha256"])
        m["sdir"] = e["sdir"]
        m["files"].add(e["file"])
    conflicting = sorted(p for p, v in mans.items() if len(v["sha"]) > 1)  # N3 / A8: one manifest path pinned with two sha256s
    items = [(p, sorted(v["sha"])[0], v["sdir"], sorted(v["files"])) for p, v in sorted(mans.items()) if p not in conflicting]
    mr = {}
    for part in read_manifests.map([items[i:i + 25] for i in range(0, len(items), 25)]):
        mr |= part
    ow = sorted({(e["path"], e["sha256"]) for e in entries if family(e["sdir"]) == "own_all"})
    rw = {}
    for part in row_weights.map([ow[i:i + 40] for i in range(0, len(ow), 40)]):
        rw |= part

    def w_of(e):  # -> (weights, info, why) for one entry, fail closed
        if e["manifest"]["path"] in conflicting:
            return {}, {}, "manifest path pinned with two sha256s"
        v = mr.get(e["manifest"]["path"], {})
        if "error" in v or not v.get("sha_ok"):
            return {}, {}, f"manifest unreadable or sha256 mismatch ({v.get('error')})"
        return tuple(v["per_file"].get(e["file"], [{}, {}, "file not in the manifest read"]))

    groups = {}
    for gname, G in GROUPS.items():  # Part 1 per lead group and readout, across ALL physical dirs and sets feeding the pool (09 B6)
        g_out = {"gating": G["gating"], "readouts": {}}
        for ro, key in G["keys"].items():
            es = [e for e in entries if e["override_key"] == key]
            missing, rows, wrong_dir, rowviol, rows_dir = [], [], [], [], defaultdict(list)
            per_dir_fit = defaultdict(lambda: defaultdict(lambda: {"matches_fit": 0, "differs_from_fit": 0}))
            for e in es:
                if e["sdir"] not in G["dirs"]:
                    wrong_dir.append({"set": e["set"], "folder": e["folder"], "sdir": e["sdir"]})
                w, info, why = w_of(e)
                if why:
                    missing.append({"set": e["set"], "folder": e["folder"], "reason": why})
                    continue
                rows.append((e["folder"], w))
                rows_dir[e["sdir"]].append((e["folder"], w))
                if G["fit"]:
                    for f, ok in fit_match(G["family"], e["file"], w, fits[G["fit"]]).items():
                        per_dir_fit[e["sdir"]][f]["matches_fit" if ok else "differs_from_fit"] += 1
                if G["family"] == "own_all":
                    r = rw.get(e["path"]) or {"error": "npz not read"}
                    v_ = (["npz sha256 mismatch"] if r.get("sha_ok") is False else []) + row_violations(r, w)
                    if v_:
                        rowviol.append({"folder": e["folder"], "reasons": v_[:6]})
            t = distinct(rows)
            sets_ = sorted({e["set"].split("|")[0].removeprefix("IL:") for e in es})
            off_fit = {d: {f: c for f, c in v.items() if c["differs_from_fit"]} for d, v in per_dir_fit.items()}
            off_fit = {d: v for d, v in off_fit.items() if v}
            why = ([f"{len(missing)} of {len(es)} folders missing / unreadable (fail closed)"] if missing or not es else []) + (
                [f"distinct checkpoints per fold {dict((f, v['distinct']) for f, v in t.items() if v['distinct'] != 1)}"] if any(v["distinct"] != 1 for v in t.values()) else []) + (
                [f"{len(wrong_dir)} entries in a physical dir outside {G['dirs']}"] if wrong_dir else []) + (
                [f"{len(rowviol)} own_all row-weight violations"] if rowviol else []) + (
                [f"sets missing {sorted(set(FOUR) - set(sets_))}"] if G["family"] == "sol" and set(FOUR) - set(sets_) else []) + (
                [f"folders whose fold weights differ from the fit record: {off_fit}"] if off_fit else [])
            g_out["readouts"][ro] = {"key": key, "entries": len(es), "sets": sets_, "physical_dirs": sorted({e["sdir"] for e in es}),
                                     "distinct_per_fold": t, "per_dir_vs_fit": {d: dict(v) for d, v in per_dir_fit.items()},
                                     "checkpoints_per_dir_fold": {d: {f: v["checkpoints"] for f, v in distinct(rs).items()} for d, rs in sorted(rows_dir.items())},
                                     "missing": missing[:20], "n_missing": len(missing), "wrong_dir": wrong_dir[:20], "row_violations": rowviol[:20],
                                     "n_row_violations": len(rowviol), "verdict": "FAIL" if why else "PASS", "why": why}
        groups[gname] = g_out
    by_sf = defaultdict(list)  # information: weights-only distinct count per (scorer dir, file)
    for e in entries:
        w, _, why = w_of(e)
        if not why:
            by_sf[f"{e['sdir']}/{e['file']}"].append((e["folder"], w))
    info_tables = {k: {f: v["distinct"] for f, v in distinct(rs).items()} for k, rs in sorted(by_sf.items())}
    lead_keys = {k for G in GROUPS.values() for k in G["keys"].values()}
    xd = defaultdict(lambda: {"rows": [], "by_dir": defaultdict(list), "missing": 0})  # 09: B6 across the S_old e1 / e2 directories of
    for e in entries:                                                                # every NON-lead key too (casu_inherit <-> casis_uncut_
        if e["override_key"] in lead_keys:                                           # inherit, sol <-> sol_c); information, not gating
            continue
        w, _, why = w_of(e)
        x = xd[e["override_key"]]
        x["by_dir"][e["sdir"]].append((e["folder"], w))
        if why:
            x["missing"] += 1
        else:
            x["rows"].append((e["folder"], w))
    cross_dir = {k: {"physical_dirs": sorted(x["by_dir"]), "n_missing": x["missing"],
                     "per_dir_rows": {d: sum(bool(r[1]) for r in rs) for d, rs in sorted(x["by_dir"].items())},
                     "per_dir_missing": {d: sum(not r[1] for r in rs) for d, rs in sorted(x["by_dir"].items())},
                     "compared_across_dirs": all(any(r[1] for r in rs) for rs in x["by_dir"].values()),  # False: one dir unreadable
                     "distinct_per_fold": {f: v["distinct"] for f, v in distinct(x["rows"]).items()},
                     "checkpoints_per_dir_fold": {d: {f: v["checkpoints"] for f, v in distinct([r for r in rs if r[1]]).items()} for d, rs in sorted(x["by_dir"].items())}}
                 for k, x in sorted(xd.items()) if len(x["by_dir"]) > 1}
    pin_vs_res = {}
    for ro in ("R", "C"):
        a = {json.dumps(w, sort_keys=True) for e in entries if e["override_key"] == GROUPS["sol_a_uncut"]["keys"][ro] for w in [w_of(e)[0]] if w}
        b = {json.dumps(w, sort_keys=True) for e in entries if e["override_key"] == GROUPS["sol_a_uncut_pinned (sensitivity)"]["keys"][ro] for w in [w_of(e)[0]] if w}
        pin_vs_res[ro] = {"reselect_checkpoints": len(a), "pinned_checkpoints": len(b), "identical": a == b, "rule": "information only (different checkpoints by design)"}
    dres = []
    for part in deltas.map([pairs_jobs[i:i + 30] for i in range(0, len(pairs_jobs), 30)]):
        dres += part
    c4 = [c for x in dres if "ctl" in x for c in x["ctl"]]
    dres = [x for x in dres if "ctl" not in x]
    ce_info = not CALL_END_CONTENT_GATES  # 09's re-scope: call_end_content-derived controls and batches are information, not gating c4
    c4_ce = [c for c in c4 if c["call_end"]] if ce_info else []
    c4 = [c for c in c4 if not (ce_info and c["call_end"])]
    c4_need = sorted({batch_of(x[k]) for x in dres if x.get("status") == "compared" and x["line"] == "uncut"
                      and not (ce_info and x["px"].endswith("__call_end_content.npz")) for k in ("px", "py")})
    c4_by_batch = {b_: {"controls": sum(c["batch"] == b_ for c in c4), "failed": sum(c["batch"] == b_ and not c["passes"] for c in c4)}
                   for b_ in sorted(set(c4_need) | {c["batch"] for c in c4})}
    c7 = synthetic_checks()
    fibs = [(e["folder"], w_of(e)[0]) for e in entries if e["override_key"] == "p2/fibs__reply_mean.npz" and not w_of(e)[2]]
    c1 = bool(fibs) and all(v["distinct"] == 1 for v in distinct(fibs).values())
    c2 = False
    if fibs:
        pl = copy.deepcopy(fibs)
        k0 = sorted(k for k in pl[0][1]["all"] if k != "layer")[0]
        pl[0][1]["all"][k0] = "0" * 64
        c2 = any(v["distinct"] == 2 for v in distinct(pl).values())
    oa = [(e["folder"], w_of(e)[0]) for e in entries if e["override_key"] == GROUPS["own_all_rs_uncut_L48"]["keys"]["R"] and not w_of(e)[2]]
    c5 = False
    k5 = next((i for i, (_, w) in enumerate(oa) if fit_match("own_all", "own_all_rs_uncut_L48__R.npz", w, fits["own_all_rs_uncut_L48"])["f0"] is True), None)
    if k5 is not None:  # non-vacuous: the planted folder matches its fit before the plant
        pl = copy.deepcopy(oa)
        pl[k5][1]["f0"]["weights"] = "0" * 64
        c5 = distinct(pl)["f0"]["distinct"] == 2 and fit_match("own_all", "own_all_rs_uncut_L48__R.npz", pl[k5][1], fits["own_all_rs_uncut_L48"])["f0"] is False
    sl = [(e, w_of(e)[0]) for e in entries if e["override_key"] == GROUPS["sol_a_uncut"]["keys"]["R"] and not w_of(e)[2]]
    c6 = False
    s6 = next(((e, w) for e, w in sl if fit_match("sol", e["file"], w, fits["sol_a_uncut"])["f2"] is True), None)
    if s6:  # non-vacuous: the planted folder matches the fit before the plant
        w6 = copy.deepcopy(s6[1])
        w6["f2"]["fold_model"] = "0" * 64
        c6 = fit_match("sol", s6[0]["file"], w6, fits["sol_a_uncut"])["f2"] is False
    return {"manifests_read": len(mr), "manifest_paths_with_two_pins": conflicting, "groups": groups, "info_tables": info_tables,
            "pinned_vs_reselect": pin_vs_res, "cross_dir_non_lead": cross_dir, "deltas": dres,
            "controls": {"c1_fibs_single_checkpoint": c1, "c2_planted_second_checkpoint_flagged": c2, "c4_by_batch": c4_by_batch,
                         "c4_failed_files": [c["file"] for c in c4 if not c["passes"]][:20],
                         "c4_batches_without_control": sorted(set(c4_need) - {c["batch"] for c in c4}),
                         "c4_all_pass": bool(c4) and all(c["passes"] for c in c4) and set(c4_need) <= {c["batch"] for c in c4},
                         "c4_call_end_content_information": {"controls": len(c4_ce), "failed": [c["file"] for c in c4_ce if not c["passes"]][:20]},
                         "c5_own_all_planted_fold_flagged": c5, "c6_sol_planted_fold_model_vs_fit_flagged": c6,
                         "c7_synthetic": c7, "c7_all_pass": all(c7.values())}}


def plan():
    fb = git("show", f"{INPUTS_FULL[0]}:{INPUTS_FULL[1]}")
    if sha(fb) != INPUTS_FULL[2]:
        raise SystemExit("INPUTS_FULL sha256")
    F = json.loads(fb)
    blobs = {f"INPUTS_FULL @ {INPUTS_FULL[0]}": fb}  # every pinned lead source's bytes, for the call_end scan
    ovs = [(k, v["commit"], v["path"], v["sha256"]) for k, v in F["overrides"].items()] + [
        (k, IL_COMMIT, f"experiments/2026-10-15_fpr_sweep_507_prep/config/cutfull/override_IL-{k[3:].replace('|', '_')}_{t}_gated.json", s)
        for k, (t, s) in IL.items()]
    entries, ov_rec = [], []
    for name, c, p, want in ovs:
        b = git("show", f"{c}:{p}")
        if sha(b) != want:
            raise SystemExit(f"{name}: override sha256")
        ov_rec.append({"set": name, "commit": c, "path": p, "sha256": want})
        O = json.loads(b)
        blobs[f"{name} @ {c}"] = b
        flat = [(fo, fk, e) for fo, d in O.get("arms", {}).items() for fk, e in d.items()] + [
            (e.get("folder"), fk, e) for fk, lst in O.get("simh", {}).items() for e in lst]
        if not flat:  # a per-source guard: an override that yields no entry fails closed
            raise SystemExit(f"STOP: {name}: the override has no entries")
        for folder, fk, e in flat:
            sd, fn = scorer_dir(e["path"]), e["path"].rsplit("/", 1)[-1]
            mp_ = (e.get("manifest") or {}).get("path") or ""
            if sd is None or mp_.rsplit("/", 1)[0] != e["path"].rsplit("/", 1)[0]:
                raise SystemExit(f"{name}: entry {fk} of {folder}: no scorer dir, or its manifest is not in the file's folder")
            entries.append({"set": name, "sdir": sd, "file": fn, "override_key": fk, "folder": e["path"].rsplit("/", 1)[0], "path": e["path"],
                            "sha256": e["sha256"], "manifest": {"path": mp_, "sha256": e["manifest"]["sha256"]}})
    pins_ = defaultdict(set)
    for e in entries:
        pins_[e["path"]].add(e["sha256"])
    two = sorted(p for p, v in pins_.items() if len(v) > 1)
    if two:  # Astra 7: one score file pinned with two sha256s by different overrides
        raise SystemExit(f"STOP: {len(two)} score files pinned with two sha256s, e.g. {two[0]}")
    fits, fit_pins = {}, {}  # the registered fit records (INPUTS_FULL fits), sha256-checked, as canonical weights
    for gname, G in GROUPS.items():
        if not G["fit"] or G["fit"] in fits:
            continue
        fe = F["fits"][G["fit"]]["fit"]
        b = git("show", f"{fe['commit']}:{fe['path']}")
        if sha(b) != fe["sha256"]:
            raise SystemExit(f"fit record of {G['fit']}: sha256")
        fit_pins[G["fit"]] = fe
        d = json.loads(b)
        if G["family"] == "sol":
            folds, files, conf = sol_weights(d)
            tu = (d.get("weights") or {}).get("tight_used_by_scorer") or {}
            wf = (d.get("weights") or {}).get("files_sha256") or {}
            if not tu or any(wf.get(k) != v for k, v in tu.items()):  # each RAW used entry sits in weights.files_sha256 itself, same sha
                raise SystemExit(f"fit record of {G['fit']}: tight_used_by_scorer empty or not in weights.files_sha256 with the same sha256")
            used = {wfile_key(k): v for k, v in tu.items()}
            if len(used) != len(tu):  # two used files normalizing to one name would hide one
                raise SystemExit(f"fit record of {G['fit']}: tight_used_by_scorer names collide after normalization")
            if conf or sorted(folds) != [f"f{k}" for k in range(5)] or not used or any(files.get(k) != v for k, v in used.items()):
                raise SystemExit(f"fit record of {G['fit']}: fold models {sorted(folds)}, {len(used)} files used by the scorer (each must sit in "
                                 f"files_sha256 with the same sha256), conflicts {conf[:2]}")
            fits[G["fit"]] = {"folds": folds, "files": used}  # the files the scorer loads (its manifests record exactly these), not every fitted file
        else:
            fits[G["fit"]] = {f"{x['variant']}|{x['readout']}|f{x['fold']}": {"weights": x["weights_sha256"], "preproc": x["preproc_sha256"],
                                                                             "training_manifest": x["training_manifest_sha256"]} for x in d["fit"]["fits"]}
    eb = git("show", f"{E2E[0]}:{E2E[1]}")
    if sha(eb) != E2E[2]:
        raise SystemExit("E2E record sha256")
    R = json.loads(eb)
    pin_of = {"/conv/" + e["path"]: {"sha256": e["sha256"], "manifest": e["manifest"]} for e in entries}
    jobs = []
    for p in R["pairs"]:
        if not p["verdict"].startswith("(1)"):
            continue
        x, y = p["pair"]
        for line in ("uncut", "32k"):
            ln = p["lines"].get(line) or {}
            if "common_prefix" not in ln:
                continue
            fx = [f["path"] for f in R["runs_export"][x]["files"] if f["line"] == line]
            fy = [f["path"] for f in R["runs_export"][y]["files"] if f["line"] == line]
            for px in fx:
                for py in fy:
                    if px.rsplit("/", 1)[-1] == py.rsplit("/", 1)[-1]:
                        fn, dt = px.rsplit("/", 1)[-1], dtype_of(px)
                        jobs.append({"pair": [x, y], "line": line, "x_label": x, "y_label": y, "x_key": R["runs_export"][x]["npz_row"],
                                     "y_key": R["runs_export"][y]["npz_row"], "d": ln["common_prefix"], "n_tokens": ln.get("n_tokens"),
                                     "px": px, "py": py, "x_pin": pin_of.get(px), "y_pin": pin_of.get(py), "dtype": dt[0], "tol": TOL[dt[0]],
                                     "call_end_gate": bool(CALL_END_CONTENT_GATES and line == "uncut" and scorer_dir(px) == "p2"
                                                           and fn in {f"{q}__call_end_content.npz" for q in CALL_END_PROBES})})
    rx_ = next((j["px"] for j in jobs if "/sola_uncut_reselect/" in j["px"]), None)
    pn_ = next((j["py"] for j in jobs if "/sola_uncut_pinned/" in j["py"] and rx_ and j["py"].rsplit("/", 1)[-1] == rx_.rsplit("/", 1)[-1]), None)
    c3 = {"px": rx_, "py": pn_} if rx_ and pn_ else None
    c3_ok = c3 is not None and not pair_ok(c3["px"], c3["py"]) and pair_ok(c3["px"], c3["px"]) and pair_ok(c3["py"], c3["py"])
    if not call_end_in({"planted": b'{"p2/fibs__call_end_content.npz": 1}'}) or len(blobs) != 1 + len(ovs) or not all(blobs.values()):
        raise SystemExit("STOP: the call_end scan is vacuous (a planted entry not caught, or a pinned source missing / empty)")
    ce_pinned = call_end_in(blobs)
    if ce_pinned:  # 09: the call_end_content gate attaches to a readout the headline reads; if one is ever pooled, re-scope (switch on) first
        raise SystemExit(f"STOP: {len(ce_pinned)} pinned lead sources mention call_end (e.g. {ce_pinned[0]}); 09's 601b6c19c:1761 gate "
                         "applies: set CALL_END_CONTENT_GATES and re-review")
    miss, not_made, nw_batches = [], [], set()  # every gated-probe call_end_content pair is scheduled or reported, never dropped silently
    for p in R["pairs"]:
        if not p["verdict"].startswith("(1)") or "common_prefix" not in (p["lines"].get("uncut") or {}):
            continue
        sp = {lb: [f["path"] for f in R["runs_export"][lb]["files"] if f["line"] == "uncut" and scorer_dir(f["path"]) == "p2"] for lb in p["pair"]}
        side = {lb: {x.rsplit("/", 1)[-1] for x in v} for lb, v in sp.items()}
        made = {lb: any(x.endswith("__call_end_content.npz") for x in s_) for lb, s_ in side.items()}  # the batch writes the readout (any probe)
        for q in CALL_END_PROBES:
            ce = f"{q}__call_end_content.npz"
            scored = {lb: bool({f"{q}__reply_mean.npz", f"{q}__command.npz"} & s_) for lb, s_ in side.items()}
            if any(scored[lb] and made[lb] and ce not in side[lb] for lb in side):  # a batch that writes the readout omits this probe's file
                miss.append(f"{'~'.join(p['pair'])}:{ce} (omitted)")
            elif all(ce in s_ for s_ in side.values()):
                if not any(j["pair"] == p["pair"] and j["line"] == "uncut" and pair_ok(j["px"], j["py"]) and scorer_dir(j["px"]) == "p2"
                           and j["px"].endswith("/" + ce) for j in jobs):
                    miss.append(f"{'~'.join(p['pair'])}:{ce} (listed, not scheduled)")
            elif all(scored.values()):  # one side's batch writes no call_end_content at all: nothing to compare (reported, by key)
                not_made.append(f"{'~'.join(p['pair'])}:{ce}")
                nw_batches |= {batch_of(sp[lb][0]) for lb in side if not made[lb] and sp[lb]}
            elif any(scored.values()):  # the probe is scored on one side only: an incomplete inventory (reported, by key)
                miss.append(f"{'~'.join(p['pair'])}:{q} scored on one side only")
    if miss and CALL_END_CONTENT_GATES:
        raise SystemExit(f"STOP: {len(miss)} gated call_end_content comparisons missing from the schedule, e.g. {miss[:3]}")
    mc = [(MUST_COMPARE[1].format("a", f"{q}__call_end_content.npz"), MUST_COMPARE[1].format("b", f"{q}__call_end_content.npz")) for q in MUST_COMPARE[2]]
    mc_absent = [m[0] for m in mc if not any(j["pair"] == list(MUST_COMPARE[0]) and (j["px"], j["py"]) == m and pair_ok(*m) for j in jobs)]
    miss += [f"E2E same-raw-dir pair not scheduled: {m}" for m in mc_absent]  # information since 09's re-scope (recorded, not a STOP)
    lead_mc = [k for k in MUST_COMPARE_LEAD if not any(j["pair"] == list(MUST_COMPARE[0]) and j["line"] == "uncut" and pair_ok(j["px"], j["py"])
                                                       and j["x_pin"] and j["y_pin"] and j["px"].endswith("/" + k.split("/")[1])
                                                       and scorer_dir(j["px"]) == k.split("/")[0] for j in jobs)]
    if lead_mc:
        raise SystemExit(f"STOP: lead file kinds without a pinned same-directory job for {MUST_COMPARE[0]}: {lead_mc}")
    lead_counts = {g: {ro: sum(e["override_key"] == k for e in entries) for ro, k in G["keys"].items()} for g, G in GROUPS.items()}
    if any(n == 0 for v in lead_counts.values() for n in v.values()):
        raise SystemExit(f"a lead key has no entry: {lead_counts}")
    return entries, ov_rec, jobs, {"c3_planted_cross_directory_pair": c3, "c3_excluded": c3_ok,
                                   "call_end_content_not_written_by_one_batch": not_made, "batches_writing_no_call_end_content": sorted(nw_batches),
                                   "call_end_content_schedule_gaps": miss, "pinned_call_end_content_entries": 0,
                                   "pinned_call_end_scan": {"sources": [f"INPUTS_FULL @ {INPUTS_FULL[0]} ({INPUTS_FULL[2][:16]}...)"] + [
                                       f"{o['set']} @ {o['commit']} ({o['sha256'][:16]}...)" for o in ov_rec],
                                       "how": "b'call_end' in each source's sha-checked bytes (every field)", "sources_scanned": len(blobs),
                                       "entries_scanned": len(entries), "planted_entry_caught": True,
                                       "not_scanned_here": "the pooled catalogue manifest 6f428281... (not mounted by (b)); refresh2's keys-only scan "
                                                           "found 0 call_end / call_end_content strings (positive control: 8 __command, 72 .npz)"}}, \
        fits, fit_pins, lead_counts


BAD = ("error", "folds differ", "fold missing", "fold values outside", "score_by_model present", "token positions on one side",
       "token positions not integral and non-negative", "positions differ inside", "per-row arrays not len(key)")  # statuses that FAIL any pair of a group; any other non-"compared" status FAILs a REQUIRED pair


def part2_verdicts(dres):
    """Per lead group (uncut): its pinned pairs and the gated p2 call_end_content pairs are REQUIRED and must end "compared"; coverage needs
    a compared PINNED pair per readout; per (line, scorer dir, file) for the rest (information)."""
    def group_of(px):
        sd, fn = scorer_dir(px), px.rsplit("/", 1)[-1]
        g = next((g for g, G in GROUPS.items() if sd in G["dirs"] and fn in {k.rsplit("/", 1)[-1] for k in G["keys"].values()}), None)
        if g is None and CALL_END_CONTENT_GATES and sd == "p2" and fn in {f"{q}__call_end_content.npz" for q in CALL_END_PROBES}:
            g = f"{fn.split('__')[0]}|uncut"  # 09 (601b6c19c:1761): its call_end_content holds the probe's headline
        return g
    lead, info = defaultdict(list), defaultdict(list)
    for x in dres:
        if x.get("status", "").startswith("excluded"):
            continue
        g = group_of(x["px"]) if x["line"] == "uncut" else None
        (lead[g] if g else info[f"{x['line']}:{scorer_dir(x['px'])}/{x['px'].rsplit('/', 1)[-1]}"]).append(x)

    def verdict(xs, need_cover):
        comp = [x for x in xs if x.get("status") == "compared"]
        st = defaultdict(int)
        for x in xs:
            st[x.get("status", "?").split(":")[0]] += 1
        bad = [x for x in xs if x.get("status", "").startswith(BAD)]
        req_open = [x for x in xs if (x["headline_input"] or x.get("call_end_gate")) and x.get("status") != "compared"]
        cov_ = {x["px"].rsplit("/", 1)[-1] for x in comp if x["headline_input"]}
        miss_ = sorted(set(need_cover) - cov_) if need_cover else []
        over = [x for x in comp if (x.get("n_over_bound") or 0) + (x.get("n_by_model_over_bound") or 0) > 0]
        mx = max((x["max_abs_delta"] for x in comp), default=None)
        why = ([f"{len(bad)} pairs with a failing status ({sorted({x['status'].split(':')[0] for x in bad})})"] if bad else []) + (
            [f"{len(req_open)} REQUIRED pairs (pinned, or a gated call_end_content) did not end 'compared' "
             f"({sorted({x.get('status', '?').split(':')[0] for x in req_open})})"] if req_open else []) + (
            [f"Part 2 not covered for {miss_} (no compared pinned pair for that readout)"] if miss_ else []) + (
            ["weights differ between paired folders"] if any(x.get("weights_same") is False for x in comp) else []) + (
            ["a paired folder's weights record is missing"] if any(x.get("weights_same") is None for x in comp) else []) + (
            ["units with differing per-row weights"] if any((x.get("n_units_weights_sha_differ") or 0) > 0 for x in comp) else []) + (
            ["own_all rows differ from their folder manifest"] if any(any(v) for x in xs for v in (x.get("row_identity_why") or [])) else []) + (
            [f"{len(over)} compared pairs with values beyond the dtype bound (max |delta| {mx:.3e})"] if over else [])
        per_file = {fn: {"pairs": sum(x["px"].endswith("/" + fn) for x in xs), "compared": len(cs), "dtype": next((x.get("dtype") for x in cs), None),
                         "max_abs_delta": max((x["max_abs_delta"] for x in cs), default=None),
                         "n_values_beyond_bound": sum((x.get("n_over_bound") or 0) + (x.get("n_by_model_over_bound") or 0) for x in cs),
                         "bound_verdict": "PASS" if cs and not any((x.get("n_over_bound") or 0) + (x.get("n_by_model_over_bound") or 0) for x in cs) else
                         ("FAIL" if cs else "none compared")}
                    for fn in sorted({x["px"].rsplit("/", 1)[-1] for x in xs}) for cs in [[x for x in comp if x["px"].endswith("/" + fn)]]}
        return {"verdict": "FAIL" if why else "PASS", "why": why, "statuses": dict(st), "compared": len(comp), "max_abs_delta": mx, "per_file": per_file,
                "n_values_beyond_bound": sum((x.get("n_over_bound") or 0) + (x.get("n_by_model_over_bound") or 0) for x in comp),
                "per_folder_pair": [{"pair": x["pair"], "folder_x": x["px"].rsplit("/", 1)[0], "folder_y": x["py"].rsplit("/", 1)[0],
                                     "file": x["px"].rsplit("/", 1)[-1], "status": x.get("status"), "dtype": x.get("dtype"),
                                     "max_abs_delta": x.get("max_abs_delta"), "n_differ": x.get("n_differ"),
                                     "n_over_bound": x.get("n_over_bound"), "n_by_model_over_bound": x.get("n_by_model_over_bound"),
                                     "weights_same": x.get("weights_same"), "row_identity_why": x.get("row_identity_why"),
                                     "n_rows": [x.get("n_rows_x"), x.get("n_rows_y")], "headline_input": x["headline_input"],
                                     "call_end_gate": x.get("call_end_gate"),
                                     "hosts": [(x.get("ids_x") or {}).get("hosts"), (x.get("ids_y") or {}).get("hosts")]} for x in xs]}
    return ({g: verdict(lead.get(g, []), [k.rsplit("/", 1)[-1] for k in G["keys"].values()] if G["gating"] else []) | {"gating": G["gating"]}
             for g, G in GROUPS.items()}, {k: verdict(v, []) for k, v in sorted(info.items())})


@app.local_entrypoint()
def main(dry: bool = False):
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("dirty tree")
    entries, ov_rec, jobs, c3, fits, fit_pins, lead_counts = plan()
    plan_summary = {"overrides": len(ov_rec), "entries": len(entries), "manifests": len({e["manifest"]["path"] for e in entries}),
                    "lead_counts": lead_counts, "delta_candidates": len(jobs), "delta_same_dir": sum(pair_ok(j["px"], j["py"]) for j in jobs),
                    "delta_pinned_pairs": sum(bool(j["x_pin"] and j["y_pin"]) for j in jobs), "c3": c3["c3_excluded"],
                    "fits": {g: f["sha256"][:16] for g, f in fit_pins.items()}}  # the record only: stdout carries gate fields only (2e)
    if dry:
        print(json.dumps({"dry_run": "plan ok"}), flush=True)
        return
    call = drive.spawn(entries, jobs, fits)
    plan_summary["driver_call"] = call.object_id
    r = call.get()
    p2_lead, p2_info = part2_verdicts(r["deltas"])
    e2e2 = [{k: x.get(k) for k in ("pair", "line", "px", "py", "status", "weights_x_sha256", "weights_y_sha256", "weights_same", "max_abs_delta",
                                   "n_differ", "n_rows_x", "n_rows_y", "n_rows_file_x", "n_rows_file_y")}
            | {"hosts": [(x.get("ids_x") or {}).get("hosts"), (x.get("ids_y") or {}).get("hosts")]}
            for x in r["deltas"] if x["px"].endswith("__call_end_content.npz") and all(lb.split("|", 1)[0].endswith("_e2") for lb in x["pair"])]
    mc = {q: next((x.get("status") for x in r["deltas"] if x["pair"] == list(MUST_COMPARE[0])
                   and x["px"] == MUST_COMPARE[1].format("a", f"{q}__call_end_content.npz")
                   and x["py"] == MUST_COMPARE[1].format("b", f"{q}__call_end_content.npz")), "absent") for q in MUST_COMPARE[2]}
    mcl = {k: next((x.get("status") for x in r["deltas"] if x["pair"] == list(MUST_COMPARE[0]) and x["line"] == "uncut" and x["headline_input"]
                    and pair_ok(x["px"], x["py"]) and scorer_dir(x["px"]) == k.split("/")[0] and x["px"].endswith("/" + k.split("/")[1])), "absent")
           for k in MUST_COMPARE_LEAD}
    controls = r["controls"] | {k: v for k, v in c3.items() if not k.startswith(("call_end", "batches_writing", "pinned_call_end"))} | {
        "c8_lead_file_kinds": mcl, "c8_lead_all_compared": all(v == "compared" for v in mcl.values())}
    non_lead = {"call_end_content": {
        "rule": "information only: no lead label reads call_end_content (09 READ_LOG @ c5a4953b3:1786, 1789; d614e63b2:1791); plan() STOPs if "
                "INPUTS_FULL / IL ever pin one (checked non-vacuously: entries scanned and a planted entry caught)",
        "pinned_call_end_content_entries": c3["pinned_call_end_content_entries"], "pinned_call_end_scan": c3["pinned_call_end_scan"],
        "e2e_same_raw_dir_pairs": mc, "not_written_by_one_batch": c3["call_end_content_not_written_by_one_batch"],
        "batches_writing_no_call_end_content": c3["batches_writing_no_call_end_content"], "schedule_gaps": c3["call_end_content_schedule_gaps"],
        "made_judged_from": "every *__call_end_content.npz a side lists (any probe)",
        "pairs": {k: v for k, v in p2_info.items() if k.endswith("__call_end_content.npz")}}}
    controls["pass"] = bool(controls["c1_fibs_single_checkpoint"] and controls["c2_planted_second_checkpoint_flagged"] and controls["c3_excluded"]
                            and controls["c4_all_pass"] and controls["c5_own_all_planted_fold_flagged"] and controls["c6_sol_planted_fold_model_vs_fit_flagged"]
                            and controls["c7_all_pass"] and controls["c8_lead_all_compared"])  # call_end_content: non_lead_information
    gate = {g: {"part1": {ro: v["verdict"] for ro, v in r["groups"][g]["readouts"].items()}, "part2": p2_lead[g]["verdict"], "gating": G["gating"]}
            for g, G in GROUPS.items()}
    gate_pass = controls["pass"] and all(all(v == "PASS" for v in gate[g]["part1"].values()) and gate[g]["part2"] == "PASS"
                                         for g, G in GROUPS.items() if G["gating"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"what": "(b) scorer-consistency check: weights per folder (keys, ids, shas) + 09's max |delta| and counts over the natural pairs",
           "gate_pass": gate_pass, "gate_by_lead": gate,
           "scope": "gating = the controls + Part 1 of the six lead probes (sol_a_uncut reselect, FIBS / harmfulness / refusal / high-stakes |uncut, "
                    "own_all_rs_uncut_L48) + Part 2 REQUIRED uncut pairs of those six (their pinned R and C pairs); p2 call_end_content pairs, "
                    "sol_a_uncut_pinned, other probes and the 32k line reported, not gating (09: no lead label reads call_end_content)",
           "call_end_content_gates": CALL_END_CONTENT_GATES, "call_end_content_probes": CALL_END_PROBES,
           "rulings_09": {"grading_and_tolerance": "READ_LOG @ 601b6c19c:1761-1765", "b6": "READ_LOG @ 294a73e21:1733 + correction 62c2ab5c4:1738",
                          "original_b_rule": "READ_LOG @ 01517a573:1692 (its 1e-6 superseded by 601b6c19c:1761)",
                          "file": "experiments/2026-10-14_priority1_spec_09/READ_LOG.md (branch aswin/convergence-audit-09); verbatim in NOTE_09_rulings_2026-10-16.md"},
           "tolerance": {"rule": "|delta| <= TOL[dtype] * max(1, |x|, |y|) per value (score and every score_by_model column); unequal non-finite "
                                 "values are always beyond; pairs with no jointly finite value are not compared", "TOL": TOL,
                         "dtype_by_scorer": DTYPE, "dtype_meaning": "the arithmetic precision of the score computation, not the on-disk "
                                                                    "dtype (float32-computed, float64-stored scores keep the float32 bound)",
                         "32k_and_other": "float32 bound, information only (dtype not read)"},
           "tolerance_disclosure": "tolerance changed from 1e-6 (a placeholder) to the dtype-derived bound, set by code read before (b)'s output "
                                   "was read (09, READ_LOG @ 601b6c19c:1761 item 3). CUT confirms: (b) has never run before this record; no (b) "
                                   "output existed or was read when the bound and DTYPE were set.",
           "server_side_bound": "every function retries=0; drive (TD 2,400 s) is spawned once by the local entrypoint and is the only caller of the "
                                "worker .map()s (TW 600 s each), so the app's execution is capped at about 3,000 s (~50 min); run_b.sh stops it "
                                "locally at 45 min",
           "certifies": "the weights each folder was scored with (from loaded, re-hashed bytes), not the scoring code",
           "pairing_rule": "within one line + scorer directory + file (09); 32k: the directory above the folder is not always the scorer directory",
           "weights_cited": {"own_all": "run_own_all.py @ e2003d463:1109-1117 (rows 1199-1204, manifest 1222)",
                             "sola": "run_sola_uncut_score.py @ 2bbab1758:211-226, 418-419", "solu": "run_sold_cutfull_score.py @ 687696e61:192-218, 675-677",
                             "p2": "run_simh_scores.py @ 2313b6eb1:1347-1357"},
           "plan_summary": plan_summary, "git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": sha(Path(__file__).read_bytes()),
           "inputs_full": {"commit": INPUTS_FULL[0], "path": INPUTS_FULL[1], "sha256": INPUTS_FULL[2]}, "fit_records": fit_pins, "overrides": ov_rec,
           "e2e_record": {"commit": E2E[0], "path": E2E[1], "sha256": E2E[2]}, "lead_counts": lead_counts, "controls": controls,
           "part1_groups": r["groups"], "part2_lead": p2_lead, "part2_information": p2_info, "p2_call_end_content_e2_vs_e2": e2e2,
           "non_lead_information": non_lead,
           "pinned_vs_reselect": r["pinned_vs_reselect"], "info_tables": r["info_tables"], "cross_dir_non_lead_information": r["cross_dir_non_lead"], "manifest_paths_with_two_pins": r["manifest_paths_with_two_pins"],
           "deltas": r["deltas"]}
    dest = HERE / "results" / f"scorer_consistency_{stamp}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    b = json.dumps(rec, indent=1, default=str).encode()
    with dest.open("xb") as fh:
        fh.write(b)
    print(json.dumps({"record": str(dest.relative_to(REPO)), "sha256": sha(b), "gate_pass": gate_pass, "gate_by_lead": gate,  # gate fields only;
                      "controls": {k: v for k, v in controls.items() if isinstance(v, bool)}}, indent=1))  # maxima and counts: the record only
    if not gate_pass:
        raise SystemExit("(b) gate FAIL (see the record)")
