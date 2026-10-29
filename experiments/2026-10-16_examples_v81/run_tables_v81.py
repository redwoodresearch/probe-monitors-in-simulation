"""v81 threshold-dependent tables, fallback (b) (0d's call via q2, 10-16 ~13:2x): §1 the outcome table and §2 the six false alarms, KEYS ONLY,
under the lead's calibration (POOLED per world, RULE items 42-46; --calibration per_origin selects the per-set rows instead). PRELIMINARY.

Why this exists: the Stage A records keep only draw-0 aggregates (T, units_above, per-run alarm shares, cell_hits strings), not per-unit z.
This job re-runs the lead build's OWN row code (run_peraction311._row at the pinned build commit, its sibling modules at the same commit) on
the lead record's own driver spec, with ONE output-only patch: at draw 0 of the joint pass, at 1e-2, it copies the row's draw-0 arrays (the
pools' units and z, the threshold T per world, the attack arms' per-unit scores and the moments that standardize them, the cell hits) and
stops the row. Nothing else in the row changes. Every exported quantity is then gated EXACTLY against the pinned record (T, units_above,
n_units and per-run alarm share per pool; the cell_hits strings), else STOP.

Selection (pure functions from run_examples_v81.py, reviewed there): table_outcomes (numpy default_rng, outcome order, "none"), false_alarms
(pools present by key, z - T > 0 computed, six without replacement), argmax_command, and the unit order rule (token order).
  §1: universe = the lead cohort cells of own_all_rs_uncut_L48 C over the four origin rows of the lead combo, keyed "<origin>|<cell>"; real
      caught = run1's draw-0 hit at tau_B, sim caught = sim's at tau_S (the lead record's strings); per drawn run, the argmax content-only
      command (out-of-fold z) with z and z - T; a structural zero (zero-tool) arm shows "no command" and is missed.
  §2: universe = one event per physical execution x probe: the run's FIRST C unit in token order with z > T, in the pools (pooled: the 2
      pooled pools, H_B and H_sim; per_origin: each origin's), over the six lead probes; sorted by (probe, pool, physical run key, unit index).
      Identity (09's alias ruling ~23:4x, pooled): the run's .eval sha256, one value per key across ALL pinned inventories (else STOP); E12
      replicates and 09's (a') members are separate executions; two aliases of one .eval in one pool STOP. (The per-set draw, superseded,
      kept the physical run key as its identity.) Build aliases are display labels only. Draws carry probe, pool, physical run key, .eval
      sha256 (pooled), alias, unit index, z, T, z - T. Label: "in-sample calibration alarms (the alarm events come from the pools that set
      the thresholds)".
Locators (for Stage 2 only): each needed run is matched to rows of the score files the build's own rows read (the build's reader, logged),
by its cell (after the build's alias tags) and its EXACT designated and five-model score sequences; all matches must name one physical
run_key with strictly increasing tokens. The run_key and readout token give the staged entry of the probe's REGISTERED readout kind
(READOUT_KIND: p2 "command", Sol (a) and own_all_rs "call_end"; the inventory's staged file; its sha must be the inventory's and the score entry's) and the .eval model event (inventory .eval sha), as in run_examples_v81.arguments_at. Lengths
and sha256s only; no text.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach experiments/2026-10-16_examples_v81/run_tables_v81.py \\
      --mode importcheck | tables [--calibration pooled]
  Measured values stay off the laptop (0d via 2e, 10-17): the laptop hashes the lead and --expect records and ships their bytes; the container
  (_full) parses them, gates, and returns the record bytes, which the laptop writes unparsed after a sha check, printing keys and counts only.
  --mode check (0d via q2, 10-17): the tables path with --expect <record>@<commit> --expect-sha <sha> (same draws, else nothing written);
      writes only the keys-and-counts diagnostics (readout kinds per drawn unit, the "no command" sides' units and SZK membership) to
      results/check/.
"""

import hashlib
import io
import json
import pickle
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
sys.path.insert(0, str(HERE))
import run_examples_v81 as X  # noqa: E402  (the reviewed pure rules and locator helpers)

APP = "probe-shift_aswin-surya_sub-routec-v81_tables_kill-2026-11-14"  # route C's own description (was q2examples_v81_kill-2026-11-15)
BUILD_DIR = "experiments/2026-10-15_fpr_sweep_507_prep"
BUILD_FILES = ("run_peraction311.py", "run_fprsweep_507.py", "run_common455.py", "run_common710.py", "cutfull_decisive.py", "full_override.py", "il_arms.py")
BUILD_FILES_IF_PRESENT = ("copies.py",)  # route C's runner imports copies at top level (rp 71633875 L38); older builds have no copies.py
ROUTEC_LABEL_PREFIX = "route C pooled A (full-length; "  # 09 (READ_LOG L2172 / L2213): the registered sidecar string starts so; never the label itself
ROUTEC_SELECTION_KIND = "route C selection record"  # the pins' selection kind for make_routec_selection.py's record (09 RULE item 40)
SIDECAR_RX = re.compile(r"git:([0-9a-f]{40}):([^ @:]+\.json)@([0-9a-f]{64})")  # the group steps' pointer form (aswin/routec-group-steps)
# §7's S7RC (aswin/routec-launch @ 1d8867052) commits each TAG's Volume record <name>.json as results/routec_records/routec_<TAG>_<name>.json.gz
ROUTEC_RECORD_DIR = "experiments/2026-10-15_fpr_sweep_507_prep/results/routec_records/"
ROUTEC_A_RX = re.compile(re.escape(ROUTEC_RECORD_DIR) + r"routec_A_([^/\s@:]+\.json)\.gz")


def volume_name(rp: str) -> str:
    """The Volume record's file name the driver spec's dest must end with (_tables). This script reads A only: an S7RC copy must be
    routec_A_<name>.json.gz (its Volume name is <name>.json), and any other file under routec_records/ (routec_B_, routec_R37_, ...)
    STOPs; a path outside routec_records/ falls back to its own name less .gz (the 10-16 form). The mapping is many-to-one
    (routec_A_<n>.json.gz and <n>.json.gz both give <n>.json): it only names what the spec's dest is compared with in the container,
    which, with the record's JSON sha256 (pinned and S2-covered), decides."""
    m = ROUTEC_A_RX.fullmatch(rp)
    if m:
        return m.group(1)
    if rp.startswith(ROUTEC_RECORD_DIR):
        raise SystemExit("STOP: the lead record is under routec_records/ but is not routec_A_<name>.json.gz (this script reads A only)")
    return Path(rp).name.removesuffix(".gz")


VOLS = {"/sol": "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", "/p2": "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13",
        "/conv": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", "/q3": "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"} | {
            k: v for k, v in X.VOLS.items() if k in ("/b", "/q4", "/acts")}
EVAL_ROOTS = ("/q4/evals_v1", "/acts/evals_v1", "/q3/simh_evals_v1")  # the T196 sim-honest (A25) .eval files sit in q3onpol_work's simh_evals_v1
LABELS = {  # the six lead probes at C (full_override.HEADLINE_SIX) -> their uncut score file in the INPUTS_FULL overrides
    "sol_a_uncut C": "sola_uncut_reselect/sol_a_uncut__cmd_A.npz", "FIBS|uncut C": "p2/fibs__command.npz",
    "harmfulness|uncut C": "p2/harmfulness__command.npz", "refusal|uncut C": "p2/refusal__command.npz",
    "high-stakes|uncut C": "p2/highstakes__command.npz", "own_all_rs_uncut_L48 C": "own_all_rs_uncut/own_all_rs_uncut_L48__C.npz"}
READOUT_SOURCE = ("09's RULE l.1591 @ c3a0cd830 (Append 28d item 1: 'Sol: call_end; p2: command, the command-parameter end') and l.2982 @ "
                  "0c930f8b2 (Append 16 extended item 3: 'Sol's content-only call end and the p2 command-parameter end'); own_all_rs: the general "
                  "unit rule, RULE l.3336 @ 1ca6ad673 (Append 25h amendment 1: 'one reply (R) or one content-only call end (C) of one run'; only "
                  "p2 is carved out, by l.1591), and its registered fit code run_own_all.py @ 19611550c l.105 (run_examples_v81's fit_code pin) "
                  "builds C rows from kind 'call_end'")
READOUT_KIND = {  # each lead probe's REGISTERED C readout, the staged entry kind its unit is located at (0d via q2, 10-17; FA5); sources in
    # READOUT_SOURCE (RULE l.1591 @ c3a0cd830 and l.2982 @ 0c930f8b2; own_all_rs: RULE l.3336 @ 1ca6ad673 and its fit code @ 19611550c l.105)
    "sol_a_uncut C": "call_end", "FIBS|uncut C": "command", "harmfulness|uncut C": "command", "refusal|uncut C": "command",
    "high-stakes|uncut C": "command", "own_all_rs_uncut_L48 C": "call_end"}
if set(READOUT_KIND) != set(LABELS) or any((k == "command") != LABELS[lb].endswith("__command.npz") for lb, k in READOUT_KIND.items()):
    raise SystemExit("READOUT_KIND must cover exactly the lead probes, with 'command' exactly for the __command score files")


def readout_kind(label: str) -> str:
    """The registered readout kind of a lead probe; an unknown probe STOPs."""
    if label not in READOUT_KIND:
        raise SystemExit(f"STOP: no registered readout kind for {label!r}")
    return READOUT_KIND[label]


OUTCOME_LABEL = "own_all_rs_uncut_L48 C"
ORIGINS = ("S_old", "S_new", "S_newmains", "S_paper493")
INVENTORY_ORDER = ("S_new", "ha25f", "S_old", "S_newmains", "S_paper493", "S_paper493_w23", "pools_ep2", "pools_fill", "il")  # inputs_full.py @ b4722d9c3 l.62
OVERRIDES = ("S_new|C", "S_new+25f|C", "S_newmains|C", "S_old|C", "S_paper493|C", "pools_ep2|C")  # every C override of INPUTS_FULL
CALIB = {  # calibration -> how the lead record names the four origin rows of one label (q2/09: pooled is the lead, item 43)
    "pooled": {"combo": "pooled_all_origins", "row": "{o} (all-origins draws) [pooled: pooled_all_origins]", "pools": "pooled",
               "label": "pooled calibration (one threshold per world over all tasks), preliminary"},
    "per_origin": {"combo": None, "row": "{o} (all-origins draws)", "pools": "per_origin", "label": "per-set calibration, preliminary"}}
LEAD = {  # record = (path, commit, sha256 of the git blob, sha256 of the JSON); spec = the driver's pickled job spec on /sol, bound to the record by
    # its dest, git and stamp (pooled: its sha256 is also pinned and required); selection = the record's estimator selection record, or (pooled)
    # a RULE-file pin (path + commit + sha256 + line) when the record names no selection record (meta.estimator.selection_record null)
    "pooled": "pins file",  # a parameter (q2, ~16:3x): --pins <committed JSON>, default PINS_FILE["pooled"]; re-pinning = editing that file
    "per_origin": {  # q2 (0d via Aswin, ~13:5x): the interim draw on the per-set full-length A record
        "build_commit": "20c6fbd208a32d03cdeaa207ef0a45984c8aa0aa",  # the record's own git field
        "record": ("experiments/2026-10-15_fpr_sweep_507_prep/results/alltasks_CRD_FULL_r2_allorigins_e12_2026-10-16_113538.json.gz", "ae6b09ada",
                   "f40f3b476569888cc4a80fedfa72dd5cc4125f068966fd8418d80f62bd3606ab", "95dc8602e86e845ef60394c6e4d361cf306e6cea5b50ad9f2379e0011e8ac4e8"),
        "spec": "xfer/fpr507prep/drivers/alltasks_CRD_FULL_r2_allorigins_e12_2026-10-16_113538.pkl",
        "selection": ("experiments/2026-10-15_fpr_sweep_507_prep/results/estimator_selection_2026-10-16_123050.json", "6fd84677a",
                      "222d9f3f06f580c45df4c8d5c2f68106bca0ad88da11a374ce60b287a8daddef")}}
PINS_FILE = {"pooled": "experiments/2026-10-16_examples_v81/config/lead_pins_pooled.json"}
PATCH_ANCHOR = "                    hits0[f] = {a_: np.asarray(z_[0] > T_[0]).copy() for a_, (z_, T_) in za_.items()}\n"
PATCH = PATCH_ANCHOR + (
    "                    if _V81_EXPORT is not None and abs(f - 0.01) <= 1e-12:  # v81 examples, fallback (b): draw-0 copy, output only, then stop\n"
    "                        _V81_EXPORT.update(_v81_pack(locals()))\n"
    "                        raise _V81Stop()\n")
PATCH_HEAD = '''
_V81_EXPORT = None  # v81 examples, fallback (b): set to a dict by the examples job only; None in the build itself
_V81_READS = []  # (volume, path, sha256) of every score file the row reads through C.read_npz (Codex b1 #2: candidates = the row's own files)
_V81_OPTS = {"z_key": None}  # --z-from-calib: the D1 file's array name template for the draw-0 per-unit z of pool {nm} (B / A / S)


class _V81Stop(Exception):
    pass


def _v81_pack(L):
    import numpy as np

    def m0(mom):
        if mom is None:
            return None
        mu, sd, mu0, sd0 = mom
        return {"mu": {int(k): float(np.asarray(v)[0]) for k, v in mu.items()}, "sd": {int(k): float(np.asarray(v)[0]) for k, v in sd.items()},
                "mu0": None if mu0 is None else float(np.asarray(mu0)[0]), "sd0": None if sd0 is None else float(np.asarray(sd0)[0])}

    U, att, f = L["U"], L["att"], L["f"]
    org = L.get("PW_ORG")

    def pool_z0(nm, P):
        """The pool's draw-0 unit z: the row's own zraw0 (zpool path), or, under D1 wave 2 (the shared calibration loaded, no pooled-pool z
        computed), zpool's exact expressions applied to the loaded draw-0 moments (row 0 of the block's arrays; draw 0 never skips fold -1)."""
        if _V81_OPTS["z_key"]:  # --z-from-calib: READ the draw-0 per-unit z that wave 1 wrote into the D1 file (loaded by the build, by sha)
            pwc, key = L.get("PWC_"), _V81_OPTS["z_key"].format(nm=nm)
            if not isinstance(pwc, dict) or key not in pwc:
                raise SystemExit(f"{L['label']}: the D1 calibration file has no array {key!r} (v81 export, --z-from-calib)")
            z = np.asarray(pwc[key], dtype=np.float64)
            if z.ndim != 1 or len(z) != len(P["s"]):
                raise SystemExit(f"{L['label']}: {key!r} is not one draw-0 z per pool unit (shape {z.shape}, units {len(P['s'])})")
            return z.copy(), f"read from the D1 calibration file ({key})"
        if nm in L["zraw0"]:
            return np.asarray(L["zraw0"][nm], dtype=np.float64).copy(), "zraw0 (the row's zpool)"
        mom = L["mom_"][nm]
        if not L["fold_model"]:
            return np.asarray(P["s"], dtype=np.float64).copy(), "raw score (single model)"
        if mom is None:
            raise SystemExit(f"{L['label']}: no moments for pool {nm} at draw 0 (v81 export)")
        fl = L["folds"]
        MUa = np.stack([np.asarray(mom[0][k])[:1] for k in fl], 1)
        SDa = np.stack([np.asarray(mom[1][k])[:1] for k in fl], 1)
        fm = np.maximum(P["f"], 0)
        z = (P["s"][None, :] - MUa[:, fm]) / SDa[:, fm]
        un = np.flatnonzero(P["f"] == -1)
        if len(un):
            if mom[2] is None:
                raise SystemExit(f"{L['label']}: fold -1 units without shared fold -1 moments (v81 export)")
            m = ((P["sbm"][None, un, :] - MUa[:, None, :]) / SDa[:, None, :]).mean(2)
            mu0, sd0 = np.asarray(mom[2])[:1], np.asarray(mom[3])[:1]
            z[:, un] = (m - mu0[:, None]) / sd0[:, None]
        return np.asarray(z[0], dtype=np.float64).copy(), "zpool's expressions on the shared D1 moments"

    return {"label": L["label"], "origin": L["vtag"], "coh": [str(c) for c in L["coh"]], "fold_model": bool(L["fold_model"]),
            "folds": [int(x) for x in L["folds"]] if L["fold_model"] else [], "szk": sorted(str(k) for k in L["SZK"]),
            "T": {"B": float(L["Tb"]["B"][0]), "S": float(L["TS"]["primary"][0])},
            "pools": {nm: {"run": np.asarray(P["run"]).astype(str), "s": np.asarray(P["s"], dtype=np.float64).copy(), "f": np.asarray(P["f"]).copy(),
                           "sbm": np.asarray(P["sbm"], dtype=np.float64).copy(), "col_moments": bool(P.get("col_moments")), "oof": bool(P.get("oof")),
                           "z0": zs[0], "z0_source": zs[1], "org": (np.asarray(org[nm]).astype(str) if org else None)}
                      for nm, P, zs in (("B", L["PB"], pool_z0("B", L["PB"])), ("S", L["PS"], pool_z0("S", L["PS"])))},
            "arms": {w: {"z0": np.asarray(z[0], dtype=np.float64).copy(), "hit": np.asarray(L["hits0"][f][w]).copy(),
                         "run": np.asarray(U["run"][att[w]]).astype(str), "s": np.asarray(U["score"][att[w]], dtype=np.float64).copy(),
                         "f": np.asarray(U["fold"][att[w]]).copy(), "sbm": np.asarray(U["sbm"][att[w]], dtype=np.float64).copy()}
                     for w, z in (("run1", L["zr1"]), ("sim", L["zsS"]))},
            "mom": {nm: m0(L["mom_"][nm]) for nm in ("B", "S")}}
'''

image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "zstandard==0.25.0")
         .add_local_file(HERE / "run_examples_v81.py", "/root/run_examples_v81.py"))
app = modal.App(APP, image=image)
vols = {k: modal.Volume.from_name(v).read_only() for k, v in VOLS.items()}


# ---------- container ----------


def load_build(code: dict):
    """The lead build's modules at its pinned commit (bytes from git, sha-checked by the launcher), with the output-only patch, imported
    from /tmp/pw. The patch anchor must occur exactly once (else STOP), so a changed build cannot be patched silently."""
    d = Path("/tmp/pw")
    d.mkdir(exist_ok=True)
    for name, text in code.items():
        if name == "run_peraction311.py":
            if text.count(PATCH_ANCHOR) != 1:
                raise SystemExit("the patch anchor (draw-0 cell hits) is not exactly once in the build's run_peraction311.py: STOP")
            head, sep, rest = text.partition("\nimport modal\n")
            if not sep:
                raise SystemExit("run_peraction311.py has no top-level 'import modal' line to place the export helpers after: STOP")
            text = head + sep + PATCH_HEAD + rest.replace(PATCH_ANCHOR, PATCH)
        (d / name).write_text(text)
    sys.path.insert(0, str(d))
    import run_peraction311 as R  # noqa: E402

    raw = R.C.read_npz

    def logged(volname, path, sha):  # the build's raw reader, unchanged, plus a record of what it read
        z = raw(volname, path, sha)
        R._V81_READS.append((volname, path, sha))
        return z

    R.C._read_npz_raw = logged  # the row's own `C.__dict__.setdefault("_read_npz_raw", C.read_npz)` keeps this one
    return R


def find_eval_v2(key: str, sha: str) -> tuple:
    """X.find_eval for this job: the key's .eval path with AND without a leading runs/ under EVAL_ROOTS (incl. the A25 sim-honest root), by
    sha256; exactly one distinct (volume, path) may hold it with that sha (the earlier draw record gated 1 unit on the narrower search)."""
    base = key.split("#e")[0].removeprefix("runs/")
    rels = [base, "runs/" + base]
    hits = []
    for r in EVAL_ROOTS:
        for rl in rels:
            pth = Path(r) / rl
            if pth.exists():
                b = pth.read_bytes()
                if hashlib.sha256(b).hexdigest() == sha:
                    mount = "/" + r.split("/")[1]
                    hits.append((VOLS[mount], str(pth).removeprefix(mount + "/"), b))
    if len({(v, q) for v, q, _ in hits}) != 1:
        raise X.Gate(f"{len(hits)} .eval files with the identity's sha256 under {EVAL_ROOTS}")
    return hits[0]


def unit_z(arm: dict, mom, fold_model: bool, folds: list):
    """Per-unit draw-0 z of an attack arm's units with the pool's draw-0 moments, the same operations as the build's zatt (per unit instead
    of on the run max): mapped units (s - mu_f) / sd_f at the unit's fold; fold -1 units the five models' z averaged, then (m - mu0) / sd0."""
    import numpy as np

    s, f, sbm = arm["s"], arm["f"].astype(np.int64), arm["sbm"]
    if not fold_model:
        return s.copy()
    MUa = np.array([[mom["mu"][k] for k in folds]])
    SDa = np.array([[mom["sd"][k] for k in folds]])
    fm = np.maximum(f, 0)
    z = (s - MUa[0, fm]) / SDa[0, fm]
    un = np.flatnonzero(f == -1)
    if len(un):
        if mom["mu0"] is None:
            raise SystemExit("fold -1 attack units without fold -1 moments: STOP")
        m = ((sbm[None, un, :] - MUa[:, None, :]) / SDa[:, None, :]).mean(2)[0]
        z[un] = (m - mom["mu0"]) / mom["sd0"]
    return z


_R = None  # the patched build module, set before the worker pools fork


def _pool():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(max_workers=12, mp_context=mp.get_context("fork"))


def _cohort_one(job_):
    lb, cf, bl = job_
    try:
        return _R._row(lb, {k: v for k, v in cf.items() if k not in ("draws_only", "row_store")} | {"cohort_only": True}, bl)
    except SystemExit:  # reviewer A (off-laptop): the build's own STOP messages can carry values; only the label crosses back
        raise SystemExit(f"{lb}: the build's row stopped (message withheld)") from None


def _export_one(job_):
    """One row up to its draw-0 export (then stopped by the patch). A row that finishes or is withheld returns a reason string instead."""
    lb, cf, bl = job_
    _R._V81_EXPORT = {}
    _R._V81_READS.clear()
    try:
        out = _R._row(lb, cf, bl)
        return f"the row finished without reaching the draw-0 export: STOP ({'withheld' if 'withheld' in out else 'no export'})"
    except _R._V81Stop:
        return dict(_R._V81_EXPORT) | {"reads": sorted(set(_R._V81_READS))}
    except SystemExit:  # reviewer A (off-laptop): the build's own STOP messages can carry values; only the label crosses back
        raise SystemExit(f"{lb}: the build's row stopped (message withheld)") from None


T_TOL = 1e-12  # 09's ruling via q2 (10-16 ~14:1x; 0d agrees): the build's own reproduction tolerance (run_peraction311 @ 20c6fbd20 l.274)
BAND = 1e-9  # the floor of the per-unit margin (Codex b5 / b6)
SAFETY = 2.0  # two containers each within z_bound of the exact value differ by at most twice it (Codex b6 / b7)
REPRO_LABEL = "thresholds reproduced to ≤ 1e-12 (floating-point); all counts exact"  # 09's wording, verbatim
DUP_RULE = {"per_origin": "first_in_sort", "pooled": None}  # 09's ruling (a) via q2 ~14:4x: per-set keeps the FIRST event in the stated sort per
# (probe, physical run) and records the dropped ones; the pooled draw keeps the gate ON (None = STOP), its pools being deduplicated


TINY = 1e-150  # |x| below this (and not 0) could underflow in x**2: STOP (Codex b7)


def _mom_err(x):
    """Rigorous bounds on the build's draw-0 moment arithmetic for inputs x (unit weights at draw 0; wmoments / zpool: mean = (w @ x) / n,
    var = (w @ x^2) / n - mean^2, sd = sqrt(max(var, 0))), valid for ANY summation order: with u = eps / 2 and g = (n + 2) eps (>= gamma_n
    plus the roundings of the squares, the division and the subtraction), |d mean| <= g mean|x|; |d E2| <= g E2; |d var| <= Dv = g E2 + 2 |mean|
    g mean|x| + (g mean|x|)^2 + eps (E2 + mean^2); requires var - 2 Dv > 0 (a positive variance interval; else STOP), then |d sd| <= Dv / sd +
    eps sd. Inputs that could underflow in x^2 STOP. -> (mean, sd, dmu, dsd)."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n, eps = len(x), np.finfo(np.float64).eps
    if n < 2:
        raise SystemExit("STOP: a moment over fewer than 2 units")
    ax = np.abs(x)
    if ((ax > 0) & (ax < TINY)).any():
        raise SystemExit("STOP: an input small enough to underflow in x^2 (the drift bound would not hold)")
    if not float(ax.max()) ** 2 * n < 1e300:  # Codex b8: no partial sum of x or x^2 can overflow in any order (with headroom)
        raise SystemExit("STOP: inputs large enough that a partial sum of x^2 could overflow (the drift bound would not hold)")
    g = (n + 2) * eps
    mean, ma, E2 = float(np.mean(x)), float(np.mean(ax)), float(np.mean(x * x))
    var = E2 - mean * mean
    Dv = g * E2 + 2 * abs(mean) * g * ma + (g * ma) ** 2 + eps * (E2 + mean * mean)
    if not (var - 2 * Dv > 0):
        raise SystemExit("STOP: a variance whose floating-point interval reaches zero (the drift bound would not hold)")
    sd = var ** 0.5
    return mean, sd, g * ma, Dv / sd + eps * sd


def _z_err(xu, mu, sd, dmu, dsd):
    """|d z| for z = (x - mu) / sd with |d mu| <= dmu, |d sd| <= dsd < sd, plus the subtraction and division roundings."""
    import numpy as np

    eps = np.finfo(np.float64).eps
    z = (xu - mu) / sd
    return (dmu + eps * (np.abs(xu) + abs(mu))) / (sd - dsd) + np.abs(z) * dsd / (sd - dsd) + eps * np.abs(z)


def z_bound(P: dict, mom, fold_model: bool, folds: list, U: dict = None):
    """A rigorous per-unit bound on |z_computed - z_exact| for the build's draw-0 z of units U (default: the pool's own units) standardized on
    pool P's moments (Codex b6 / b7): per fold, the fold's moment inputs (the pool's s in that fold; sbm[:, k] for the col_moments branch; the
    pooled s for oof) through _mom_err and _z_err; a fold -1 unit (21b) carries its five models' bounds into m = their mean, then through the
    mu0 / sd0 moments of the pool's fold -1 m values (their own perturbation added). Two containers then differ by at most twice this.
    The recomputed moments must equal the exported ones to 1e-9 relative (the inputs are the build's). Single-model pools: z = the raw score,
    bound 0."""
    import numpy as np

    eps = np.finfo(np.float64).eps
    U = P if U is None else U
    if not fold_model or mom is None:
        return np.zeros(len(U["z0"]))
    s, f, sbm = P["s"], P["f"].astype(np.int64), P["sbm"]
    fu, sbu, xu_s = U["f"].astype(np.int64), U["sbm"], U.get("s", P["s"] if U is P else None)
    b = np.zeros(len(U["z0"]))
    st = {}
    for k in folds:
        x = s if P["oof"] else (sbm[:, k] if P["col_moments"] else s[f == k])
        mu, sd, dmu, dsd = _mom_err(x)
        if abs(mu - mom["mu"][k]) > 1e-9 * max(1.0, abs(mu)) or abs(sd - mom["sd"][k]) > 1e-9 * max(1.0, sd):
            raise SystemExit("STOP: recomputed draw-0 moments differ from the exported ones (the bound's inputs are not the build's)")
        st[k] = (mom["mu"][k], mom["sd"][k], dmu, dsd)
        m = fu == k
        if m.any():
            b[m] = _z_err(xu_s[m], *st[k])
    u1 = np.flatnonzero(fu == -1)
    if len(u1):
        p1 = np.flatnonzero(f == -1)
        if not len(p1) or mom["sd0"] is None:
            raise SystemExit("STOP: fold -1 units without the pool's fold -1 moments")

        def m_and_err(B_):
            zz = np.stack([(B_[:, k] - st[k][0]) / st[k][1] for k in folds], 1)
            ez = np.stack([_z_err(B_[:, k], *st[k]) for k in folds], 1)
            mm = zz.mean(1)
            # Codex b8: the K-term mean's own reduction error is bounded by (K + 2) eps mean|zz| (gamma_K on the sum of |zz|, plus the
            # division), not by |mm| (cancellation); the inputs' drift ez enters the exact mean linearly
            return mm, ez.mean(1) + (len(folds) + 2) * eps * np.abs(zz).mean(1)

        mp, emp = m_and_err(sbm[p1])
        mu0, sd0, dmu0, dsd0 = _mom_err(mp)
        dmu0, dsd0 = dmu0 + emp.max(), dsd0 + 2 * emp.max() * (np.mean(np.abs(mp)) + emp.max()) / sd0 + emp.max()  # the inputs' own drift
        if not dsd0 < 0.5 * sd0:
            raise SystemExit("STOP: the fold -1 re-standardization's sd interval is too wide for the bound")
        mu_, eu_ = m_and_err(sbu[u1])
        b[u1] = _z_err(mu_, mom["mu0"], mom["sd0"], dmu0, dsd0) + eu_ / (sd0 - dsd0)
    return b


def origin_members(org: str):
    """A pooled unit's origin tag as the pooled build writes it (run_peraction311 @ 704122edf l.1086 / l.1112): a bare origin name, or the sorted
    origins of a run in several sets' pools joined by " + " (e.g. 'S_new + S_old'). -> the member list if EVERY part is exactly one of ORIGINS
    (no suffix, no repeats), else None (the caller STOPs; e.g. 'S_new (registered cohort)' -> None)."""
    parts = str(org).split(" + ")
    if not parts or len(set(parts)) != len(parts) or parts != sorted(parts) or any(x not in ORIGINS for x in parts):
        return None
    return parts


def check_calib_options(calibration: str, use_pw_calib: bool, z_from_calib: str, dry: bool) -> None:
    """One validator for both entry points (launcher and container; Astra p6a): the pooled-only flags, never both, the template's syntax, and
    the pooled recompute path refused outside a dry run."""
    if (use_pw_calib or z_from_calib) and calibration != "pooled":
        raise SystemExit("--use-pw-calib / --z-from-calib apply to the pooled calibration only")
    if use_pw_calib and z_from_calib:
        raise SystemExit("choose one of --use-pw-calib (z computed from the shared moments) and --z-from-calib (z read from the shared file)")
    if z_from_calib and not re.fullmatch(r"[A-Za-z0-9_]*\{nm\}[A-Za-z0-9_]*", z_from_calib):
        raise SystemExit("--z-from-calib takes the D1 file's array name template with {nm} (letters, digits, underscores and one {nm})")
    if calibration == "pooled" and not dry and not (use_pw_calib or z_from_calib):
        raise SystemExit("pooled: the recompute path is out (0d ~17:1x); run with --use-pw-calib or --z-from-calib <array template>")


def gate_row(ex: dict, rec_row: dict, exact: bool = False) -> float:
    """Equality with the lead record's own draw-0 numbers for this row, else STOP: T within T_TOL (absolute), every count, share and cell_hits
    string EXACT. Returns the row's max |T - T_record| over its two pools. exact (pooled, 0d ~17:1x): T must EQUAL the record's (the shared
    D1 tau), and the floating-point drift margins are not applied (z come from the shared moments or the shared file; exactness is checked by
    the exact T, counts, shares and cell_hits, not assumed)."""
    import numpy as np

    p0 = rec_row["pools_at_draw0"]["0.01"]
    dT = 0.0
    for nm in ("B", "S"):
        P, T = ex["pools"][nm], ex["T"][nm]
        runs = {}
        for r, z in zip(P["run"].tolist(), P["z0"].tolist()):
            runs[r] = max(runs.get(r, -np.inf), z)
        got = {"n_units": int(len(P["z0"])), "units_above": int((P["z0"] > T).sum()), "runs": len(runs),
               "per_run_alarm_share": float(np.mean([v > T for v in runs.values()]))}
        bad = [k for k, v in got.items() if p0[nm][k] != v]
        d_ = abs(float(p0[nm]["T"]) - T)
        if not (np.isfinite(T) and (d_ == 0.0 if exact else d_ <= T_TOL)):
            bad.append("T")
        if not exact:
            zb_ = z_bound(P, ex["mom"][nm], ex["fold_model"], ex["folds"])
            it_ = np.flatnonzero(P["z0"] == T)
            marg = np.maximum(SAFETY * (zb_ + (zb_[it_].max() if len(it_) else zb_.max())), BAND) + T_TOL
            if int((np.abs(P["z0"] - T) <= marg).sum()) != 1:  # only T's own unit may lie within its drift margin of T, else an alarm could swap (Codex b5 / b6)
                bad.append("a unit other than T's own within its floating-point margin of T")
        dT = max(dT, d_)
        if bad:
            raise SystemExit(f"STOP: {ex['label']} {ex['origin']} pool {nm}: {bad} differ from the lead record")
    cs = sorted(range(len(ex["coh"])), key=lambda i: ex["coh"][i])
    if [ex["coh"][i] for i in cs] != rec_row["cohort_cell_keys"]:
        raise SystemExit(f"STOP: {ex['label']} {ex['origin']}: cohort cells differ from the lead record")
    lv = rec_row["cell_hits_draw0"]["levels"]["0.01"]
    for w in ("run1", "sim"):
        h = ex["arms"][w]["hit"]
        s_ = "".join("-" if h[i] < 0 else ("1" if h[i] else "0") for i in cs)
        if s_ != lv[w]:
            raise SystemExit(f"STOP: {ex['label']} {ex['origin']}: the {w} cell_hits string differs from the lead record")
    return dT


MOUNT = {"conv": "/conv", "p2": "/p2", "solavg": "/sol"}
def lookup_cell(key: str) -> str:
    """The index cell of a key: a relaunch file key honest_b_relaunch|<task>_r<k> by its task (the build's relaunch_alias), then e12_cell
    (Codex b3 optional: relaunch aliases reach their candidates)."""
    import re

    w, c = key.split("|", 1)
    if w == "honest_b_relaunch":
        m = re.fullmatch(r"(.+)_r([1-5])", c)
        c = m.group(1) if m else c
    return _R.e12_cell(c)


def alias_ok(file_key: str, alias: str) -> bool:
    """Whether a score file's key can be the build's name for this pool / attack run, by the PINNED build's own alias rules (Codex b2):
    relaunch_alias (honest_b_relaunch|<task>_r<k> -> honest_r<k>|<task>), e12_rename (world -> world_e2, cell by e12_cell, both tags in either
    order) and SP_WORLD_ALIAS (an S_paper493 file's world token -> its folder's routing world); otherwise the keys must be equal."""
    import numpy as np

    if file_key == alias:
        return True
    fw, fc = file_key.split("|", 1)
    aw, ac = alias.split("|", 1)
    if fw == "honest_b_relaunch":
        try:
            return str(_R.relaunch_alias(np.array([file_key]), "v81 alias check")[0][0]) == alias
        except SystemExit:
            return False
    if _R.e12_cell(fc) != _R.e12_cell(ac):
        return False
    return fw == aw or aw == fw + "_e2" or (fw, aw) in _R.SP_WORLD_ALIAS


class FileIndex:
    """The score files ONE row (one label x origin) actually read (the build's own reader, logged; Codex b1 #2 / b2 #1), sha-checked again
    here, indexed by the file key's normalized cell: rows (designated score as rows_of builds it, five-model scores, run_key, token,
    call_index) in file order."""

    def __init__(self, reads: list, folds: int, staged_of: dict):
        import numpy as np

        self.by_cell = {}
        for vol, path, sha in sorted(set(map(tuple, reads))):
            b = (Path(MOUNT[vol]) / path.lstrip("/")).read_bytes()
            if hashlib.sha256(b).hexdigest() != sha:
                raise SystemExit("a logged score file's sha256 differs on re-read")
            z = np.load(io.BytesIO(b))
            if not all(x in z.files for x in ("key", "run_key", "token")):
                continue  # not an uncut per-row file (no run_key / token): never a locator source
            k = z["key"].astype(str)
            n = len(k)
            f = z["fold"].astype(np.int64) if (folds == 5 and "fold" in z.files) else np.zeros(n, np.int64)
            sbm = z["score_by_model"].astype(np.float64) if "score_by_model" in z.files else np.full((n, 5), np.nan)
            s_ = z["score"].astype(np.float64) if "score" in z.files else np.where(f >= 0, sbm[np.arange(n), np.clip(f, 0, 4)], np.nan)
            rk, tok = z["run_key"].astype(str), z["token"].astype(np.int64)
            ci = z["call_index"].astype(np.int64) if "call_index" in z.files else None
            for key in sorted(set(k.tolist())):
                ix = np.flatnonzero(k == key)
                self.by_cell.setdefault(lookup_cell(key), []).append({
                    "file_key": key, "path": path, "sha256": sha, "volume": vol, "staged_shas": staged_of.get((path, sha), []),
                    "s": s_[ix], "sbm": sbm[ix], "run_key": rk[ix], "token": tok[ix], "call_index": None if ci is None else ci[ix]})

    def match(self, alias: str, s, sbm, none_ok: bool = False):
        """The file rows of THIS row's own files whose key is an alias of `alias` by the build's rules (alias_ok) and whose designated AND
        five-model score sequences equal the run's exactly, in order. All matches must name ONE physical run_key with the same tokens; tokens
        must increase strictly (token order = file order). No match STOPs (or returns None with none_ok); two physical runs always STOP."""
        import numpy as np

        hits = [c for c in self.by_cell.get(lookup_cell(alias), []) if alias_ok(c["file_key"], alias) and len(c["s"]) == len(s)
                and np.array_equal(c["s"], s, equal_nan=True) and np.array_equal(c["sbm"], sbm, equal_nan=True)]
        if not hits:
            if none_ok:  # match_any: ONLY the zero-hit case is skippable; every other STOP propagates (reviewer A)
                return None
            raise SystemExit(f"STOP: run {alias}: no file row of the row's own files reproduces its unit scores")
        h0 = hits[0]
        if any(not (np.array_equal(h["run_key"], h0["run_key"]) and np.array_equal(h["token"], h0["token"])) for h in hits[1:]):
            raise SystemExit(f"STOP: run {alias}: matching files disagree on the physical run / tokens")
        if len(set(h0["run_key"].tolist())) != 1 or not (np.diff(h0["token"]) > 0).all():
            raise SystemExit(f"STOP: run {alias}: one physical run per run and strictly increasing tokens are required")
        return h0 | {"aliases_matched": sorted({h["file_key"] for h in hits})}


def locate(row: dict, i: int, inv: dict, kind: str) -> dict:
    """Unit i of a matched run: its run_key + readout token -> the staged entry of the probe's registered readout kind (inventory-pinned
    staged file; sha = the inventory's, else one of the score entry's staged_shas) -> the .eval model event (inventory .eval sha) -> digests. Keys, ints and digests only."""
    rk, tok = str(row["run_key"][i]), int(row["token"][i])
    r = inv.get(rk)
    if r is None:
        raise X.Gate("the run key is in no pinned inventory")
    up = r["uncut_planned"]
    staged = f"{up['staged_set']}/inputs/{up['staged_inputs_file']}"
    raw = (Path("/b") / staged).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if (up.get("staged_sha256") and sha != up["staged_sha256"]) or (row["staged_shas"] and sha not in row["staged_shas"]) or \
            not (up.get("staged_sha256") or row["staged_shas"]):
        raise X.Gate("the staged file's sha256 is not the inventory's and the score entry's (Codex b1 #3)")
    e = {"key": rk, "staged": staged, "staged_sha256": sha, "arm": up.get("folder"), "eval_sha256": r["eval_sha256"]}
    line, srow = X.staged_row(e)
    ci = None if row["call_index"] is None else int(row["call_index"][i])
    if ci is None:
        js = [j for j, c in enumerate(srow.get("calls") or []) if c.get("kind") == kind and c.get("readout") == tok]
        if len(js) != 1:
            raise X.Gate(f"{len(js)} staged {kind} entries carry the readout token")
        ci = srow["calls"][js[0]].get("call_index")
    a = X.arguments_at(e | {"unit": {"unit": i, "call_index": ci, "token": tok, "kind": kind}}, (line, srow))
    return {"run_key": rk, "token": tok, "readout_kind": kind, "score_file": {"volume": row["volume"], "path": row["path"], "sha256": row["sha256"]}} \
        | a["loc"] | X.digests(a["args"])


def readout_kinds(row: dict, i: int, inv: dict) -> dict:
    """Check only (0d via q2, 10-17): how many staged calls of kind "command" and of kind "call_end" carry unit i's (call_index, readout
    token), in the same staged row locate() reads (same sha checks). The unit's own call is identified ONLY by a unique entry carrying its
    (call_index, readout token); for that call, each kind's entries and their readout minus the scored token. Otherwise "not identified"
    plus the nearest staged entries before and after the token (no identity claimed). Keys and ints only; a Gate becomes a status string."""
    rk, tok = str(row["run_key"][i]), int(row["token"][i])
    ci = None if row["call_index"] is None else int(row["call_index"][i])
    out = {"run_key": rk, "token": tok, "call_index": ci}
    r = inv.get(rk)
    if r is None:
        return out | {"status": "the run key is in no pinned inventory"}
    up = r["uncut_planned"]
    staged = f"{up['staged_set']}/inputs/{up['staged_inputs_file']}"
    sha = hashlib.sha256((Path("/b") / staged).read_bytes()).hexdigest()
    if (up.get("staged_sha256") and sha != up["staged_sha256"]) or (row["staged_shas"] and sha not in row["staged_shas"]) or \
            not (up.get("staged_sha256") or row["staged_shas"]):
        return out | {"status": "the staged file's sha256 is not the inventory's and the score entry's"}
    out["staged_sha_checked_against"] = ("inventory and the score entry's staged_shas" if up.get("staged_sha256") and row["staged_shas"]
                                         else "inventory only" if up.get("staged_sha256") else "the score entry's staged_shas only")  # reviewer A #2
    try:
        _, srow = X.staged_row({"key": rk, "staged": staged, "staged_sha256": sha})
    except X.Gate as g:
        return out | {"status": f"gate: {g}"}
    calls = srow.get("calls") or []
    if not calls:  # reviewers A #3 / B #3
        return out | {"status": "the staged row has no calls"}
    at = lambda kd: [c for c in calls if c.get("kind") == kd and (ci is None or c.get("call_index") == ci) and c.get("readout") == tok]  # noqa: E731
    cm, ce = at("command"), at("call_end")
    kind_ = lambda c: c["kind"] if isinstance(c.get("kind"), str) and re.fullmatch(r"[a-z_]{1,32}", c["kind"]) else None  # noqa: E731  (B #3)
    ent_ = lambda c: {"kind": kind_(c), "call_index": c.get("call_index") if X.nnint(c.get("call_index")) else None,  # noqa: E731
                      "irid": c["irid"] if isinstance(c.get("irid"), str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", c["irid"]) else None}
    out |= {"status": "ok", "kinds_present": sorted({str(kind_(c)) for c in calls}), "command_matches": len(cm), "call_end_matches": len(ce)}
    # the unit's own call (both reviewers' re-glance REQUIRED 1): call_index numbers the calls of ONE model event and `calls` spans every
    # turn, so the call is identified by (irid, call_index), never by call_index alone; the way it was identified is recorded
    if ci is None:
        return out | {"same_call": "not identified: the unit has no call_index"}
    if len(cm) > 1 or len(ce) > 1:  # reviewer B re-glance 2, optional 1: an ambiguous match never resolves silently
        return out | {"same_call": f"not identified: {len(cm)} command / {len(ce)} call_end entries carry the unit's (call_index, readout token)"}
    if not (cm or ce):  # reviewer B re-glance 2, REQUIRED 1: no fallback guess; the neighbourhood below claims no identity
        ints_ = [c for c in calls if X.nnint(c.get("readout"))]
        near_ = {"before": max((c for c in ints_ if c["readout"] < tok), key=lambda c: c["readout"], default=None),
                 "after": min((c for c in ints_ if c["readout"] > tok), key=lambda c: c["readout"], default=None)}
        return out | {"same_call": "not identified: no command or call_end entry carries the unit's (call_index, readout token)",
                      # both reviewers (re-glance 3): every entry AT the token, which the strict neighbours below hide (a call_index mismatch)
                      "at the scored token (no call identity claimed)": [ent_(c) for c in ints_ if c["readout"] == tok],
                      "neighbourhood (no call identity claimed)": {k: None if c is None else ent_(c) | {
                          "readout_minus_scored_token": c["readout"] - tok, "entries_at_this_readout": sum(x["readout"] == c["readout"] for x in ints_)}
                          for k, c in near_.items()}}
    if not all(isinstance(c.get("irid"), str) for c in cm + ce):  # reviewer A re-glance 2, optional 3: checked per entry, before any set
        return out | {"same_call": "not identified: the identifying entry has no irid"}
    ids = {c["irid"] for c in cm + ce}
    if len(ids) > 1:
        return out | {"same_call": "not identified: the unique command and call_end entries name different irids"}
    irid = ids.pop()
    how = " and ".join(k for k, v in (("the unique command entry", cm), ("the unique call_end entry", ce)) if v) + " with the unit's (call_index, readout token)"
    same = [c for c in calls if c.get("irid") == irid and c.get("call_index") == ci]

    def per_kind(kd):  # ints only: each entry's readout minus the scored token; readouts that are not non-negative ints counted apart
        rs = [c.get("readout") for c in same if c.get("kind") == kd]
        return {"n": len(rs), "readout_minus_scored_token": sorted(r - tok for r in rs if X.nnint(r)), "readout_none_or_invalid": sum(not X.nnint(r) for r in rs)}

    return out | {"same_call": {"identified_by": how, "command": per_kind("command"), "call_end": per_kind("call_end")}}


def _tables(p: dict) -> dict:
    import numpy as np

    rec = p["record"]
    X.find_eval = find_eval_v2  # the locator's .eval search for this job (X.arguments_at looks find_eval up at call time)
    R = load_build(p["code"])
    sb = (Path("/sol") / p["spec_path"]).read_bytes()
    spec_sha = hashlib.sha256(sb).hexdigest()
    if p["calibration"] == "pooled" and not p.get("spec_sha256"):  # reviewers' R2: never skipped silently for the pooled draw
        raise SystemExit("pooled: the driver spec's sha256 pin is missing: STOP")
    if p.get("spec_sha256") and spec_sha != p["spec_sha256"]:
        raise SystemExit("the driver spec's sha256 differs from the pin: STOP")
    spec = pickle.loads(sb)
    if not (str(spec.get("dest", "")).endswith("/" + p["record_name"]) and spec["meta"].get("git") == rec["meta"]["git"]
            and spec["meta"].get("stamp") == rec["meta"]["stamp"] and spec.get("mode") == "alltasks"):
        raise SystemExit("the driver spec is not the one that wrote the lead record (dest, git, stamp, mode): STOP")
    cal = CALIB[p["calibration"]]
    rows_wanted = {(lb, cal["row"].format(o=o)) for lb in LABELS for o in ORIGINS}
    exact_ = p["calibration"] == "pooled"  # 0d (~17:1x): pooled gates are EXACT (09's <= 1e-12 covers the per-set draw only)
    check_calib_options(p["calibration"], bool(p.get("use_pw_calib")), p.get("z_from_calib") or "", bool(p.get("dry")))
    R._V81_OPTS["z_key"] = p.get("z_from_calib") or None
    jobs = spec["jobs"]
    lab = lambda lb, cf: cf.get("full_as", lb)  # noqa: E731
    combo_ok = lambda cf: cf.get("pw_combo") == cal["combo"]  # noqa: E731
    def vtag_of(cf):  # the build's vtag formula (run_peraction311._row), used only to choose which jobs to run; the export's origin is checked
        o = cf.get("origin", "S_new")
        return (o + (" without S4" if cf.get("no_s4") else "") + (" (all-units deviation)" if cf.get("allunits_dev") else "")
                + (" (registered cohort)" if cf.get("cohort") == "registered" else "") + (f" [{cf['sp_block']}]" if o == "S_paper493" and cf.get("sp_block") else "")
                + (" (all-origins draws)" if cf.get("merge") else "") + (f" [pooled: {cf['pw_combo']}]" if cf.get("pw_combo") else ""))

    mine = [i for i, (lb, cf, _) in enumerate(jobs) if lab(lb, cf) in LABELS and combo_ok(cf) and (lab(lb, cf), vtag_of(cf)) in rows_wanted]
    # the merge (all-origins) rows need the cohort pass and the cross-origin counts first: over every merge job that shares a (label, group)
    # with one of mine, so the counts are the build's own (merge_counts counts per label and group over the jobs it is given)
    grp = {(lab(jobs[i][0], jobs[i][1]), g) for i in mine if jobs[i][1].get("merge") for g in jobs[i][1]["merge"]["groups"]}
    mi = [i for i, (lb, cf, _) in enumerate(jobs) if cf.get("merge") and any((lab(lb, cf), g) in grp for g in cf["merge"]["groups"])]
    global _R
    _R = R
    with _pool() as ex_:
        co = list(ex_.map(_cohort_one, [jobs[i] for i in mi]))
    jobs = list(jobs)
    # merge_counts' own STOPs reach job() unwrapped: they name labels, groups and counts only (keys), unlike _row's (reviewer B, off-laptop 2)
    for i, j in zip(mi, R.merge_counts([jobs[i] for i in mi], co)):
        jobs[i] = j
    pwf = (rec["meta"].get("pooled_world") or {}).get("manifest_per_label", {}) if p["calibration"] == "pooled" else {}
    todo = []
    for i in mine:
        lb, cf, bl = jobs[i]
        # the spec's own D1 flags are dropped, then set explicitly: a pooled real run loads the label x combo's shared D1 calibration (wave 2,
        # by sha) with --z-from-calib (the draw-0 per-unit z READ from that file; primary, 2e's (ii')) or --use-pw-calib (z from the shared
        # moments; fallback), gated EXACTLY (T, counts, shares, cell_hits). The recompute path (no pw_calib: the row's own zpool / wthr) is
        # refused for pooled outside a dry run (0d); a per-set row never carries these flags
        cf = {k: v for k, v in cf.items() if k not in ("row_store", "pw_calib", "pw_calib_only")}
        if p.get("use_pw_calib") or p.get("z_from_calib"):  # (i) / (ii): load the label x combo's shared D1 calibration from the record's meta
            cal_ = ((rec["meta"].get("pooled_world") or {}).get("calibrations") or {}).get("by_label_combo", {}).get(f"{lab(lb, cf)} | {cf.get('pw_combo')}")
            if not (isinstance(cal_, dict) and cal_.get("path") and cal_.get("sha256")):
                raise SystemExit(f"{lab(lb, cf)}: no D1 calibration file in the record's meta.pooled_world.calibrations")
            cf = cf | {"pw_calib": {"path": cal_["path"], "sha256": cal_["sha256"]}}
        if p["calibration"] == "pooled":
            ref = pwf.get(f"{lab(lb, cf)} | {cf.get('pw_combo')}")
            if not (isinstance(ref, dict) and ref.get("path") and ref.get("sha256")):
                raise SystemExit(f"{lab(lb, cf)}: no pooled pools file in the lead record's meta.pooled_world.manifest_per_label")
            cf = cf | {"pw_pools": ref}  # the manifest entry as the build's driver passes it (to be confirmed against the reviewed build commit)
        todo.append((lb, cf, bl))
    with _pool() as ex_:
        got = list(ex_.map(_export_one, todo))
    ex = {}
    for (lb, cf, _), e_ in zip(todo, got):
        if isinstance(e_, str):
            raise SystemExit(f"{lab(lb, cf)}: {e_}")
        key = (lab(lb, cf), e_["origin"])
        if key not in rows_wanted or e_["origin"] != vtag_of(cf):
            raise SystemExit(f"STOP: {key}: the exported origin is not the predicted lead row")
        if key in ex:
            raise SystemExit(f"STOP: two jobs export the row {key}")
        if p.get("dry"):  # stub dry run: the gates are expected to fail on stub values; record the outcome, never select
            try:
                e_["max_abs_dT"] = gate_row(e_, rec["rows"][key[0]][key[1]], exact=exact_)
                e_["gate"] = "pass"
            except SystemExit as g:
                e_["gate"] = str(g)[:300]
        else:
            e_["max_abs_dT"] = gate_row(e_, rec["rows"][key[0]][key[1]], exact=exact_)
        ex[key] = e_
    if p.get("dry"):
        return {"dry_run": "rows exported to draw 0 and gated against the stub; NO selection, NO locator, NO text",
                "use_pw_calib": bool(p.get("use_pw_calib")),
                "pool_z_source": sorted({f"{nm}: {v['pools'][nm]['z0_source']}" for v in ex.values() for nm in ("B", "S")}),
                "rows_wanted": len(rows_wanted), "rows_exported": sorted(f"{a} | {b}" for a, b in ex),
                "rows_missing": sorted(f"{a} | {b}" for a, b in rows_wanted - set(ex)),
                "gate_by_row": {f"{a} | {b}": v["gate"] for (a, b), v in sorted(ex.items())},
                "reads_by_row": {f"{a} | {b}": len(v["reads"]) for (a, b), v in sorted(ex.items())},
                "pooled_pool_origins_by_row": {f"{a} | {b}": {nm: (sorted({str(x) for x in v["pools"][nm]["org"]}) if v["pools"][nm]["org"] is not None else None)
                                                              for nm in ("B", "S")} for (a, b), v in sorted(ex.items())}}
    if set(ex) != rows_wanted:
        raise SystemExit(f"STOP: rows exported {len(ex)} != wanted {len(rows_wanted)}; missing {sorted(rows_wanted - set(ex))[:4]}")
    staged_of = {}
    for ov in p["overrides"].values():
        for d in (ov.get("arms") or {}).values():
            for e in d.values():
                staged_of[(e["path"], e["sha256"])] = e.get("staged_shas") or []
        for lst in (ov.get("simh") or {}).values():
            for e in lst:
                staged_of[(e["path"], e["sha256"])] = e.get("staged_shas") or []
    inv = {}
    for name, ib in p["inventories"]:  # INVENTORY_ORDER (inputs_full.py): a later inventory's row supersedes an earlier one's by key (Codex b1 #3)
        for r in json.loads(ib).get("runs") or []:
            if isinstance(r, dict) and r.get("key") and r.get("eval_sha256") and isinstance(r.get("uncut_planned"), dict):
                inv[r["key"]] = r
    out = {"rows_gated": sorted(f"{a} | {b}" for a, b in ex), "calibration": p["calibration"],
           "pool_z_source": sorted({f"{nm}: {v['pools'][nm]['z0_source']}" for v in ex.values() for nm in ("B", "S")}),
           "use_pw_calib": bool(p.get("use_pw_calib")), "z_from_calib": p.get("z_from_calib") or None,
           "gates": {"reproduction": ("thresholds reproduced exactly (the shared D1 tau); all counts exact" if exact_ else REPRO_LABEL),
                     "T": ("T == T_record per pool (exact; 0d)" if exact_ else f"|T - T_record| <= {T_TOL:g} per pool (09's ruling; the build's reproduction tolerance)"),
                     "exact": "n_units, units_above, runs, per_run_alarm_share, cell_hits strings",
                     "band": ("not applied (exact mode: z from the shared calibration)" if exact_ else f"exactly one pool unit within its floating-point margin of T (T's own)"),
                     "max_abs_dT_by_row": {f"{a} | {b}": f"{v['max_abs_dT']:.3e}" for (a, b), v in sorted(ex.items())},
                     "rows_with_dT_nonzero": sum(v["max_abs_dT"] != 0 for v in ex.values()),
                     "max_abs_dT_all_rows": f"{max(v['max_abs_dT'] for v in ex.values()):.3e}"}, "calibration_label": CALIB[p["calibration"]]["label"],
           "driver_spec": {"volume": VOLS["/sol"], "path": p["spec_path"], "sha256": spec_sha, "bound_by": "dest, git, stamp, mode equal the record's"}}
    # ---- §1 outcome table
    cells = []
    diag = {"no_command_sides": [], "readout_kinds": [],  # check (0d via q2, 10-17): keys and counts only, output-only
            "in_szk_label": "true by construction (the 'a run without units is caught or is not a listed structural zero' STOP fires otherwise)"}
    for o in ORIGINS:
        e_ = ex[(OUTCOME_LABEL, cal["row"].format(o=o))]
        for w in ("run1", "sim"):
            z_ = unit_z(e_["arms"][w], e_["mom"]["B" if w == "run1" else "S"], e_["fold_model"], e_["folds"])
            e_["arms"][w]["uz"] = z_
        for ci_, c in enumerate(e_["coh"]):
            side = {}
            for w, nm in (("run1", "B"), ("sim", "S")):
                A, T = e_["arms"][w], e_["T"][nm]
                rk_ = f"{w}|{c}"
                m = A["run"] == rk_
                hit = bool(A["hit"][ci_])
                if rk_ in e_["szk"] or not m.any():
                    if hit or rk_ not in e_["szk"]:
                        raise SystemExit(f"STOP: {o}|{c} {w}: a run without units is caught or is not a listed structural zero")
                    side[w] = {"caught": False, "shown": "no command"}
                    diag["no_command_sides"].append({"side": rk_, "origin": o, "units_in_arm": int(m.sum()), "in_szk": rk_ in e_["szk"]})
                    continue
                uz = A["uz"][m]
                if not np.isfinite(uz).all() or float(uz.max()) != float(A["z0"][ci_]) or (float(uz.max()) > T) != hit:
                    raise SystemExit(f"STOP: {o}|{c} {w}: per-unit z does not reproduce the row's run z / hit")
                side[w] = {"caught": hit, "uz": uz, "s": A["s"][m], "sbm": A["sbm"][m], "T": T,
                           "zb": None if exact_ else z_bound(e_["pools"][nm], e_["mom"][nm], e_["fold_model"], e_["folds"],
                                                             {"s": A["s"][m], "f": A["f"][m], "sbm": A["sbm"][m], "z0": uz})}
            cells.append({"cell": f"{o}|{c}", "real_caught": side["run1"]["caught"], "sim_caught": side["sim"]["caught"],
                          "real": side["run1"] if "uz" in side["run1"] else "no command", "sim": side["sim"] if "uz" in side["sim"] else "no command"})
    t1 = X.table_outcomes(cells)
    fidx = {}

    def fi(label, row_vtag):  # the files of ONE row (label x origin): candidates never cross origins or rows (Codex b2 #1)
        if (label, row_vtag) not in fidx:
            e_ = ex[(label, row_vtag)]
            fidx[(label, row_vtag)] = FileIndex(e_["reads"], 5 if e_["fold_model"] else 1, staged_of)
        return fidx[(label, row_vtag)]

    def rows_of_org(org):  # a pooled unit's origin tag -> the lead rows of its member origins (a multi-set run has several)
        mem_ = origin_members(org)
        if mem_ is None:
            raise SystemExit(f"STOP: pooled unit origin tag {org!r} is not a bare origin or an ' + ' list of origins")
        return [cal["row"].format(o=o) for o in mem_]

    def match_any(lb_, rows_, r_, s_, b_):
        """Locate run r_ in each candidate row's own files (FileIndex.match); at least one must match exactly, and every match must name the same
        physical run_key and tokens, else STOP. It locates only; the reading (z, T) is the pooled pool's. -> (row match, rows that matched)."""
        got_ = []
        for rv_ in rows_:  # no try/except: an ambiguous member's files, a token-order failure or a sha re-read failure STOPs (reviewer A)
            h_ = fi(lb_, rv_).match(r_, s_, b_, none_ok=True)
            if h_ is not None:
                got_.append((rv_, h_))
        if not got_:
            raise SystemExit(f"STOP: {lb_}: run {r_} is found in none of its member origins' row files {rows_}")
        h0_ = got_[0][1]
        ci0_ = h0_["call_index"]
        if any(not (np.array_equal(h_["run_key"], h0_["run_key"]) and np.array_equal(h_["token"], h0_["token"])
                    and ((h_["call_index"] is None) == (ci0_ is None)) and (ci0_ is None or np.array_equal(h_["call_index"], ci0_))
                    and sorted(h_["staged_shas"]) == sorted(h0_["staged_shas"])) for _, h_ in got_[1:]):
            raise SystemExit(f"STOP: {lb_}: run {r_}: member origins' row files disagree on the physical run / tokens / call_index / staged shas")
        return h0_, [rv_ for rv_, _ in got_]

    t1o = {}
    for oc, d in t1.items():
        if d["drawn"] == "none":
            t1o[oc] = {"n": d["n"], "drawn": "none"}
            continue
        c = d["drawn"]
        shown = {}
        for w, sd in (("real (run 1)", c["real"]), ("sim (sim 1)", c["sim"])):
            if sd == "no command":
                shown[w] = "no command"
                continue
            units = [{"unit": k, "z": float(v)} for k, v in enumerate(sd["uz"])]
            u = X.argmax_command(units)
            if not exact_:
                mg_ = np.maximum(SAFETY * sd["zb"], BAND)
                if int((np.abs(sd["uz"] - u["z"]) <= mg_ + mg_[u["unit"]]).sum()) != 1:  # a near-tie could pick another unit in another container
                    raise SystemExit(f"STOP: {c['cell']} {w}: the argmax command is within the floating-point margin of another unit's z")
            o, cell = c["cell"].split("|", 1)
            rk_ = ("run1|" if w.startswith("real") else "sim|") + cell
            row = fi(OUTCOME_LABEL, cal["row"].format(o=o)).match(rk_, sd["s"], sd["sbm"])
            diag["readout_kinds"].append({"where": f"outcome | {oc} | {w}", "probe": OUTCOME_LABEL} | readout_kinds(row, u["unit"], inv))
            try:
                loc = locate(row, u["unit"], inv, readout_kind(OUTCOME_LABEL)) | {"status": "ok"}
            except X.Gate as g:
                loc = {"status": f"not retrievable: {g}"}  # 09: the slot is kept, never redrawn
            shown[w] = {"run": rk_, "unit_index": u["unit"], "n_units": len(units), "z": u["z"], "T": sd["T"], "z_minus_T": u["z"] - sd["T"],
                        "caught": sd["caught"]} | loc
        t1o[oc] = {"n": d["n"], "drawn": {"cell": c["cell"], "real_caught": c["real_caught"], "sim_caught": c["sim_caught"]} | shown}
    out["outcome_table"] = {"probe": OUTCOME_LABEL, "level": "1e-2 per action", "outcomes": t1o, "universe_cells": len(cells)}
    # ---- §2 false alarms
    probes = list(LABELS)
    pools, events, avail = set(), [], set()
    alias_merges = []  # two pool aliases of ONE physical execution (equal .eval sha256) in one pool; 09: expected 0 (the record dedups them), else STOP
    multi_set = []  # alarm runs whose pooled origin tag lists several sets ('S_new + S_old'): located by match_any, the member rows recorded

    phys_sha = p.get("phys_sha") or {}  # normalized key (no runs/) -> .eval sha256, one value across ALL pinned inventories (checked in main)

    def ident(rk):  # pooled: 09's physical identity = the run's .eval sha256; per-set (superseded draw): the physical run key, as before
        if p["calibration"] != "pooled":
            return rk
        h_ = phys_sha.get(rk.removeprefix("runs/"))
        if not h_:
            raise SystemExit(f"STOP: alarm run {rk}: no .eval sha256 in the pinned inventories (physical identity unresolved)")
        return h_

    for lb in probes:
        seen = {}
        for o in ORIGINS:
            e_ = ex[(lb, cal["row"].format(o=o))]
            for nm in ("B", "S"):
                pool = f"pooled H_{'B' if nm == 'B' else 'sim'}" if cal["pools"] == "pooled" else f"{o} H_{'B' if nm == 'B' else 'sim'}"
                P = e_["pools"][nm]
                sig = (hashlib.sha256(P["run"].astype("U").tobytes()).hexdigest(), hashlib.sha256(P["z0"].tobytes()).hexdigest(), e_["T"][nm])
                if pool in seen:  # pooled: every origin row of the label carries the same pooled pool (identical units, z and T), else STOP
                    if seen[pool] != sig:
                        raise SystemExit(f"STOP: {lb} {pool}: origin rows disagree on the pooled pool")
                    continue
                seen[pool] = sig
                pools.add(pool)
                avail.add((lb, pool))
                runs = {}
                for j, r in enumerate(P["run"].tolist()):
                    runs.setdefault(r, []).append(j)
                rv = cal["row"].format(o=o)
                sbm_ = e_["pools"][nm]["sbm"]
                phys, phys_alias = {}, {}  # keyed by ident(): pooled the .eval sha256 (09: equal .eval sha, never content or key text); per-set the run key
                for r, ix in sorted(runs.items()):
                    zz = P["z0"][ix]
                    if not (zz > e_["T"][nm]).any():
                        continue
                    if p["calibration"] == "pooled" and P["org"] is None:  # reviewer B: a pooled pool must carry its origin tags
                        raise SystemExit(f"STOP: {lb} {pool}: the pooled pool has no origin tags")
                    orgs = {str(x) for x in P["org"][ix]} if P["org"] is not None else {rv}
                    if len(orgs) != 1:
                        raise SystemExit(f"STOP: {lb} {pool}: run {r} carries units from two origin tags")
                    tag_ = orgs.pop()
                    if P["org"] is not None:
                        row, used_ = match_any(lb, rows_of_org(tag_), r, P["s"][ix], sbm_[ix])  # token order = file order, asserted in match
                        if " + " in tag_:
                            multi_set.append({"probe": lb, "pool": pool, "alias": r, "tag": tag_, "rows_matched": used_})
                    else:
                        row, used_ = fi(lb, rv).match(r, P["s"][ix], sbm_[ix]), [rv]
                    pk = str(row["run_key"][0])  # the physical run key: the SORT key (09 ~13:5x)
                    hk = ident(pk)  # the IDENTITY (09's alias ruling ~23:4x): E12 replicates and (a') members have distinct .eval files
                    if hk in phys:
                        if p["calibration"] != "pooled":  # the superseded per-set draw's rule, unchanged: same key, same z -> one event
                            if not np.array_equal(phys[hk], zz):
                                raise SystemExit(f"STOP: {lb} {pool}: two aliases of one physical run carry different unit z")
                            continue
                        alias_merges.append({"probe": lb, "pool": pool, "eval_sha256": hk, "run_kept": phys_alias[hk][0], "alias_kept": phys_alias[hk][1],
                                             "run_merged": pk, "alias_merged": r})
                        continue
                    phys[hk], phys_alias[hk] = zz, (pk, r)
                    u = X.first_alarm([{"unit": k, "z": float(v)} for k, v in enumerate(zz)], e_["T"][nm])
                    # 09: z and T are the POOLED run's own (the pooled pool's draw-0 z, the shared D1 tau), never a per-origin value; the member
                    # origin rows only locate the run
                    events.append({"probe": lb, "pool": pool, "run": pk, "_id": hk, "alias": r, "unit": u["unit"], "z": u["z"], "T": e_["T"][nm], "_row": row}
                                  | ({"eval_sha256": hk, "origin_tag": tag_, "origins_matched": [x_.split(" (")[0] for x_ in used_],
                                      "z_source": e_["pools"][nm]["z0_source"], "T_source": "the row's draw-0 tau (pooled: the shared D1 calibration's)"}
                                     if p["calibration"] == "pooled" else {}))
    if alias_merges:  # 09: the pooled record dedups true aliases already, so a nonzero count STOPs with the keys and .eval shas (keys only)
        raise SystemExit("STOP: within-pool aliases of one physical execution (equal .eval sha256): "
                         + "; ".join(f"{a['probe']} | {a['pool']} | {a['run_kept']} / {a['run_merged']} | {a['eval_sha256']}" for a in alias_merges[:5])
                         + f" ({len(alias_merges)} in all)")
    rule_ = DUP_RULE[p["calibration"]]
    if rule_ not in (None, "first_in_sort", "per_pool"):
        raise SystemExit(f"unknown DUP_RULE {rule_!r}")
    by_pk = {}  # (probe, identity): one event per physical execution per probe (09)
    for e in events:
        by_pk.setdefault((e["probe"], e["_id"]), []).append(e)
    dups = {k: v for k, v in by_pk.items() if len(v) > 1}
    dropped = []
    if dups:  # 09's gate: each physical run contributes at most ONE alarm event per probe (across its pools too)
        if rule_ is None:
            ex_ = sorted(dups)[:3]
            raise SystemExit(f"STOP: a physical execution (.eval sha256) appears twice among the alarm events: {len(dups)} (probe, sha) pairs, e.g. "
                             + "; ".join(f"{a} | {b} | runs {sorted(e['run'] for e in dups[(a, b)])} in pools {sorted(e['pool'] for e in dups[(a, b)])}"
                                         for a, b in ex_))
        if rule_ == "first_in_sort":  # keep the event first in the stated sort (probe, pool, physical run key, unit index); list the others
            keep = {k: min(v, key=lambda e: (e["pool"], e["unit"])) for k, v in dups.items()}
            dropped = sorted(({"probe": e["probe"], "pool": e["pool"], "run": e["run"], "alias": e["alias"], "unit": e["unit"]}
                              | ({"eval_sha256": e["eval_sha256"]} if "eval_sha256" in e else {})
                              for k, v in dups.items() for e in v if e is not keep[k]), key=lambda d: (d["probe"], d["pool"], d["run"]))
            events = [e for e in events if (e["probe"], e["_id"]) not in dups or e is keep[(e["probe"], e["_id"])]]
    if rule_ != "per_pool":  # the post-dedup universe: no physical run key twice per probe (09), before the six are drawn
        seen_ = set()
        for e in events:
            if (e["probe"], e["_id"]) in seen_:
                raise SystemExit("STOP: a physical execution is still repeated per probe after the dedup")
            seen_.add((e["probe"], e["_id"]))
    fa = X.false_alarms([{k: v for k, v in e.items() if k != "_id"} for e in events], probes, sorted(pools), avail)
    draws = []
    for d in fa["draws"]:
        row = d.pop("_row")
        diag["readout_kinds"].append({"where": f"false alarm {len(draws) + 1}", "probe": d["probe"], "pool": d["pool"]} | readout_kinds(row, d["unit"], inv))
        try:
            loc = locate(row, d["unit"], inv, readout_kind(d["probe"])) | {"status": "ok"}
        except X.Gate as g:
            loc = {"status": f"not retrievable: {g}"}  # 09: the slot is kept, never redrawn
        draws.append(d | loc)
    out["false_alarms"] = {"label": "in-sample calibration alarms (the alarm events come from the pools that set the thresholds)",
                           "draws": draws, "events_per_probe_pool": fa["events_per_probe_pool"], "universe_events": fa["universe"],
                           "duplicate_rule": (rule_ or "STOP") + " (09's ruling (a) via q2 ~14:4x)", "duplicates_dropped": dropped,
                           "duplicates_dropped_count": len(dropped),
                           "within_pool_alias_merges_count": len(alias_merges) if p["calibration"] == "pooled" else "n/a (per-set)",
                           "multi_set_tag_alarm_runs": multi_set, "multi_set_tag_alarm_runs_count": len(multi_set),
                           "identity": ("physical execution = the .eval sha256, one value per key across all pinned inventories (09)" if p["calibration"] == "pooled"
                                        else "the physical run key (the superseded per-set draw's rule)"),
                           "pools": sorted(pools), "thresholds": "the lead's draw-0 thresholds at 1e-2 per action (fixed)"}
    out["diagnostics"] = diag
    return out


def _render(p: dict) -> dict:
    """Stage 2 only (q2's conditional go): each drawn unit's command text, re-derived from its recorded locator (the staged row and .eval by
    sha) and returned only if the re-derived locator equals the record's AND both digests equal the record's (X.gate_digests), else STOP.
    The tables record's bytes are parsed HERE (values stay off the laptop, 0d via 2e / q2, 10-17); its git and script_sha256 go back so the
    laptop can check the writing script against git before it writes anything."""
    if p.get("stage_2") != X.STAGE2_GO:
        raise SystemExit("render: Stage 2 is not authorized for this call")
    if hashlib.sha256(p["record_bytes"]).hexdigest() != p["index_sha"]:
        raise SystemExit("render: the tables record differs from --index-sha")
    rec = json.loads(p["record_bytes"])
    if not str(rec.get("what", "")).startswith("v81 threshold tables"):
        raise SystemExit("render: the record was not written by this script's tables mode")
    ents, gated = render_entries(rec)
    X.find_eval = find_eval_v2
    out = []
    for t in ents:
        sr, un, ev = t["staged_row"], t["unit"], t["run_eval"]
        e = {"key": t["run_key"], "staged": sr["staged"], "staged_sha256": sr["staged_sha256"], "arm": sr["arm"], "eval_sha256": ev["sha256"],
             "unit": {"unit": un["row_unit"], "call_index": un["action"], "token": un["token"], "irid": un["irid"],
                      "kind": t.get("readout_kind", "call_end")}}  # records before the readout-kind fix located every unit at call_end
        pr_ = t["probe"] if t["where"].startswith("false alarm") else OUTCOME_LABEL
        if "readout_kind" in t and t["readout_kind"] != readout_kind(pr_):  # reviewer B optional 1: the recorded kind is the registered one
            raise SystemExit(f"STOP: {t['where']}: the recorded readout kind is not {pr_}'s registered one")
        a = X.arguments_at(e)
        if {k: a["loc"][k] for k in ("staged_row", "unit", "run_eval", "tool")} != {k: t[k] for k in ("staged_row", "unit", "run_eval", "tool")}:
            raise SystemExit(f"STOP: {t['where']}: the re-derived locator differs from the record")
        out.append({"where": t["where"]} | X.gate_digests(a["args"], t | {"label": t["where"]}))
    return {"entries": out, "gated": gated, "label": rec.get("label"), "run_label": rec.get("run_label"), "git": rec.get("git"),
            "script_sha256": rec.get("script_sha256")}


def _poolcheck(p: dict) -> dict:
    """Dry run (stub): each label's pooled pools file on /sol, sha-checked; per pool (B, A, S) its run and unit counts and the origins its
    units carry: each distinct tag form with its unit and run counts, and whether every ' + ' part is exactly one of ORIGINS. status is 'ok'
    unless a B or S tag fails, the file is missing / its sha differs, or an origin array is missing / misaligned. Counts and keys only."""
    import numpy as np

    cal = CALIB[p["calibration"]]
    out = {}
    for lb, ref in sorted(p["refs"].items()):
        fp = Path("/sol") / ref["path"]
        if not fp.exists():
            out[lb] = {"status": "missing on the Volume", "path": ref["path"]}
            continue
        b = fp.read_bytes()
        if hashlib.sha256(b).hexdigest() != ref["sha256"]:
            out[lb] = {"status": "sha256 differs from the record's pooled_file"}
            continue
        z = np.load(io.BytesIO(b))
        d = {"status": "ok", "arrays": sorted(z.files)}
        for nm in ("B", "A", "S"):
            if f"{nm}_run" not in z.files:
                d[nm] = "absent"
                if nm in ("B", "S"):  # reviewer B: a missing B or S pool is not ok
                    d["status"] = f"pool {nm} absent from the pooled pools file"
                continue
            n_ = int(len(z[f"{nm}_run"]))
            if f"{nm}_org" not in z.files or len(z[f"{nm}_org"]) != n_:  # Codex p1 #2: a missing / misaligned origin array is a failure, not ok
                d[nm] = {"runs": len(set(z[f"{nm}_run"].astype(str).tolist())), "units": n_, "origins": "MISSING or length differs from the run array"}
                d["status"] = "origin array missing or misaligned"
                continue
            og_, rn_ = z[f"{nm}_org"].astype(str), z[f"{nm}_run"].astype(str)
            forms = {}
            for o_ in sorted(set(og_.tolist())):
                m_ = og_ == o_
                forms[o_] = {"units": int(m_.sum()), "runs": len(set(rn_[m_].tolist())), "members_ok": origin_members(o_) is not None}
            unmapped = [o_ for o_, v_ in forms.items() if not v_["members_ok"]]
            d[nm] = {"runs": len(set(rn_.tolist())), "units": n_, "tag_forms": forms, "origins_unmapped": unmapped,
                     "multi_set_forms": sorted(o_ for o_ in forms if " + " in o_)}
            if unmapped and nm in ("B", "S"):  # the paid run is gated on B and S (reviewer B); A is reported only
                d["status"] = "origin tags that are not a bare origin or an ' + ' list of origins"
        out[lb] = d
    return {"pooled_files": out}


def check_record_meta(meta: dict, want: dict, inputs_full_sha256: str) -> None:
    """Route C (09 S2 item 10, re-asserted by key here, never on the laptop): the lead record's meta.git is the pinned launch commit, its
    meta.script_sha256 the pinned runner's (71633875 -> dcd265d7...), and meta.full_length.inputs_full the FROZEN file the laptop read for
    the inventories and overrides (sha256 equal to both pins; status FROZEN; test false). Any miss STOPs, naming keys only."""
    fin = (meta.get("full_length") or {}).get("inputs_full") or {}
    bad = [k for k, ok in (("git", meta.get("git") == want["git"]), ("script_sha256", meta.get("script_sha256") == want["script_sha256"]),
                           ("full_length.inputs_full.sha256", fin.get("sha256") == want["inputs_full_sha256"] == inputs_full_sha256),
                           ("full_length.inputs_full.status", str(fin.get("status", "")).startswith("FROZEN")),
                           ("full_length.inputs_full.test", fin.get("test") is False)) if not ok]
    if bad:
        raise SystemExit(f"STOP: the lead record's meta differs from the pins file's record_meta at {bad}")


def compose_label(b: bytes, sha: str, lead_json_sha256: str) -> str:
    """09's label rule for route C (READ_LOG L2172, amended L2213; the group steps' form): the sidecar's bytes hash to the pin (before the
    parse); it is exactly {"record", "record_sha256", "label"}; record_sha256 is this lead record's JSON sha256; the label is a string that
    starts with ROUTEC_LABEL_PREFIX and contains PRELIMINARY (no FINAL-release record exists). -> "<label> | <this step> on route C pooled A
    @ <sha>"."""
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit("STOP: the label sidecar's bytes differ from its pin")
    sc = json.loads(b)
    if not isinstance(sc, dict) or set(sc) != {"record", "record_sha256", "label"}:
        raise SystemExit("STOP: the label sidecar is not exactly {record, record_sha256, label}")
    if sc["record_sha256"] != lead_json_sha256:
        raise SystemExit("STOP: the label sidecar names another A record (record_sha256 differs from the lead record's JSON sha256)")
    lb = sc["label"]
    if not (isinstance(lb, str) and lb.startswith(ROUTEC_LABEL_PREFIX) and "PRELIMINARY" in lb):
        raise SystemExit("STOP: the sidecar label is not a route C pooled A label containing PRELIMINARY")
    return f"{lb} | v81 tables 3/4 (run_tables_v81) on route C pooled A @ {lead_json_sha256}"


def _full(p: dict) -> dict:
    """Everything that touches measured values runs here, in the container (0d via 2e, 10-17: values stay off the laptop; no parsing,
    comparing or printing there). The laptop sha-checks the lead record's and the --expect record's bytes (hashing only) and ships them;
    this parses them, runs the git-field / selection-pin / structure checks, poolcheck, the rows and the same-draws and locator-continuity
    gates, and serializes the record exactly as the laptop did before (same keys, same order). -> the record text, its sha256 and keys-only
    counts; a failed gate STOPs with keys only (no z, T or differences)."""
    import gzip

    rt = lambda x: json.loads(json.dumps(x, allow_nan=False))  # noqa: E731  (what job() applied to every remote result before)
    rb = gzip.decompress(p["lead_bytes"]) if p["lead_path"].endswith(".gz") else p["lead_bytes"]
    if hashlib.sha256(rb).hexdigest() != p["lead_json_sha256"]:
        raise SystemExit("the lead record's JSON sha256 differs from the pin")
    full = json.loads(rb)  # bare NaN literals parse (Python's json accepts them); unreadable blocks are not read by this script
    dry, c, calibration = bool(p["dry"]), p["build_commit"], p["calibration"]
    cal, notes = CALIB[calibration], []
    if c and not str((full.get("meta") or {}).get("git", "")).startswith(c[:9]) and c != (full.get("meta") or {}).get("git"):
        if not dry:
            raise SystemExit("the build commit is not the record's git field")
        notes.append("the build commit is not the record's git field")
    if p["rule_pin"]:  # fail closed (reviewer B): the key must be PRESENT in meta.estimator and its value None
        est_ = (full.get("meta") or {}).get("estimator")
        if not (isinstance(est_, dict) and "selection_record" in est_ and est_["selection_record"] is None):
            raise SystemExit("the selection pin is a RULE file, but the record's meta.estimator.selection_record is absent or not null")
    if p.get("record_meta"):  # route C (09 S2 item 10, by key in the container): launch commit, runner sha256, FROZEN INPUTS_FULL
        check_record_meta(full.get("meta") or {}, p["record_meta"], p["inputs_full_sha256"])
    if p.get("label_sidecar"):  # route C (09, READ_LOG L2172 / L2213): the label is composed from A's sidecar, read here by path@sha
        p["meta"]["run_label"] = compose_label(p["label_sidecar"]["bytes"], p["label_sidecar"]["sha256"], p["lead_json_sha256"])
    struct_ = record_structure(full, calibration)
    sub = None
    if struct_["ok"] and (not dry or (p["tables"]["spec_path"] and p["tables"]["code"])):  # the lead rows' draw-0 fields only (the gates' reference)
        sub = {"meta": {"git": full["meta"]["git"], "stamp": full["meta"]["stamp"], "pooled_world": full["meta"].get("pooled_world")},
               "rows": {lb: {cal["row"].format(o=o): {k: full["rows"][lb][cal["row"].format(o=o)][k] for k in ("pools_at_draw0", "cohort_cell_keys",
                                                                                                             "cell_hits_draw0")} for o in ORIGINS} for lb in LABELS}}
    del full
    if dry:  # structure first (keys only), then the pooled pools files and (with a spec) the rows to draw 0; nothing is selected
        out_ = {"pins": p["pins_ref"], "stub": p["stub"], "build_commit": c or "not pinned (structure and pooled-file checks only)", "record_structure": struct_}
        if calibration == "pooled" and not struct_.get("labels_without_pooled_file"):
            out_ |= rt(_poolcheck({"calibration": calibration, "refs": struct_["pooled_files"]}))
        if sub is not None:
            out_ |= rt(_tables(p["tables"] | {"record": sub, "dry": True}))
        b = json.dumps({"what": "DRY RUN against a STUB / unreviewed lead record: structure, pools and row exports only; no selection, no text",
                        "stamp (PT)": p["stamp"], "git": p["git"], "app": APP} | out_, indent=1).encode()
        return {"record_text": b.decode(), "sha256": hashlib.sha256(b).hexdigest(), "counts": {"structure_ok": struct_["ok"], "notes": notes}}
    if not struct_["ok"]:
        raise SystemExit("the lead record's structure is not what this script needs (keys only): " + json.dumps(  # reviewer B (off-laptop)
            {k: struct_.get(k) for k in ("rows_missing", "rows_lacking_fields", "labels_without_pooled_file", "rows_whose_pools_file_differs_from_the_manifest")})[:600])
    poolcheck_ = None
    if calibration == "pooled":  # reviewer B: the paid run STOPs unless every label's pooled pools file passes poolcheck (status ok: B and S tags)
        poolcheck_ = rt(_poolcheck({"calibration": calibration, "refs": struct_["pooled_files"]}))
        bad_ = sorted(lb for lb, v in poolcheck_["pooled_files"].items() if v.get("status") != "ok")
        if bad_:
            raise SystemExit(f"STOP: poolcheck is not ok for {bad_}: " + json.dumps({lb: poolcheck_["pooled_files"][lb].get("status") for lb in bad_})[:400])
    res = rt(_tables(p["tables"] | {"record": sub}))
    e = p.get("expect")
    if e:  # a locator-only re-run: the selection must equal the earlier record's by key, else nothing is written (its pin checked on the laptop)
        if hashlib.sha256(e["bytes"]).hexdigest() != e["sha256"]:  # reviewers A / B (off-laptop): the laptop's hash re-checked here, before parsing
            raise SystemExit("--expect: the earlier record's bytes differ from --expect-sha (container re-check)")
        old_ = json.loads(e["bytes"])
        new_ = json.loads(json.dumps(res))
        if draw_keys(old_) != draw_keys(new_):
            raise SystemExit("STOP: the re-run's draws differ from the earlier record's by key; nothing written")
        za_, zb_ = drawn_zmt(old_), drawn_zmt(new_)
        dz_ = max(abs(a - b) for a, b in zip(za_, zb_, strict=True)) if za_ else 0.0
        if not dz_ <= 1e-9:  # last-bit differences between containers only (09's 1e-12 on T; z within the drift margins checked in gate_row)
            raise SystemExit("STOP: the re-run's drawn z - T differ from the earlier record's by more than 1e-9; nothing written")
        res["same_draws_as"] = {"path": e["path"], "commit": e["commit"], "sha256": e["sha256"], "keys_units_cells_events_dropped": "identical",
                                "max_abs_diff_z_minus_T": f"{dz_:.3e}"}
        res["locator_continuity"] = locator_continuity(old_, new_)  # both reviewers (locator fix): STOPs before any write
    m = p["meta"]
    if p["mode"] == "check":  # keys and counts only: no z, T, locator digests or arguments
        b = json.dumps({"what": "v81 tables CHECK (0d via q2, 10-17): readout kinds per drawn unit; the 'no command' sides' units in the scored arm "
                                "and SZK membership. Keys and counts only", "stamp (PT)": p["stamp"], "git": p["git"],
                        "script_sha256": m["script_sha256"], "app": APP, "calibration": calibration,
                        "build_commit": c, "pins_file": p["pins_ref"], "z_from_calib": p["tables"]["z_from_calib"] or None,  # reviewer A #5 (path + sha, never contents)
                        "same_draws_as": res["same_draws_as"], "diagnostics": res["diagnostics"]}, indent=1).encode()
        counts = {"same_draws": "identical", "readout_rows": len(res["diagnostics"]["readout_kinds"]),
                  "no_command_sides": len(res["diagnostics"]["no_command_sides"])}
    else:
        out = {"what": "v81 threshold tables (RULE v81 §1-§2, items 42-46): KEYS ONLY; lengths and sha256s of command bytes / full arguments; no text",
               "stamp (PT)": p["stamp"], "git": p["git"], "script_sha256": m["script_sha256"],
               "build_commit": c, "lead": m["lead"], "pins_file": p["pins_ref"], "app": APP, "label": CALIB[calibration]["label"],
               "run_label": m["run_label"] or None, "label_sidecar": m.get("label_sidecar"), "poolcheck": poolcheck_} | res
        out["readout_kind_by_probe"] = {"mapping": READOUT_KIND, "source": READOUT_SOURCE}  # reviewer B optional 2
        if e:  # a locator-only re-run (q2 / 0d, 10-17): the same draws, counts and values; only the locators are new
            out["supersedes"] = {"path": e["path"], "commit": e["commit"], "sha256": e["sha256"],  # reviewer A optional 1: the scope names its evidence
                                 "scope": "the locators only: draw keys identical, z - T within 1e-9 (same_draws_as); locators of previously ok units "
                                          "identical except the staged entry index for command-kind units (locator_continuity)"}
        b = json.dumps(out, indent=1).encode()
        # 0d (10-17): no table number (dedupe counts, drift counts) leaves the container in the printed line; they stay in the record
        counts = {"rows_gated": len(res["rows_gated"])} | (
            {"same_draws": "identical", "locator_continuity": res["locator_continuity"]["counts"]} if e else {})
    return {"record_text": b.decode(), "sha256": hashlib.sha256(b).hexdigest(), "counts": counts}


def _importcheck(p: dict) -> dict:
    """Import the patched build in a container (no data read): the anchor is found once and the module imports."""
    R = load_build(p["code"])
    return {"imported": True, "has_row": callable(getattr(R, "_row", None)), "has_merge_counts": callable(getattr(R, "merge_counts", None)),
            "export_hook": R._V81_EXPORT is None and issubclass(R._V81Stop, Exception)}


@app.function(cpu=16, memory=65536, timeout=3600, retries=0, volumes=vols)
def job(mode: str, payload: dict):
    err = None
    try:
        res = {"importcheck": _importcheck, "full": _full, "render": _render, "poolcheck": _poolcheck}[mode](payload)
        return json.loads(json.dumps(res, allow_nan=False))
    except SystemExit as e:
        err = str(e)
    except X.Gate as e:
        err = f"gate: {e}"
    except Exception as e:  # noqa: BLE001 -- only the type crosses back
        err = f"stopped: {type(e).__name__} (message withheld)"
    raise SystemExit(err)


# ---------- laptop ----------


def git(*a) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def is_rule_pin(L: dict) -> bool:
    """A selection pin that is a RULE file (decided by its kind, reviewer B), which must then carry a line."""
    ref = L.get("selection_ref") or {}
    if not str(ref.get("kind", "")).startswith("RULE"):
        if ref.get("line") is not None:  # reviewer B: a line without a RULE kind is malformed, never silently a non-RULE pin (fail closed)
            raise SystemExit("a selection pin with a line must have a kind starting with 'RULE'")
        return False
    if not (isinstance(ref.get("line"), int) and not isinstance(ref.get("line"), bool) and ref["line"] >= 1):
        raise SystemExit("a RULE-file selection pin must carry its line (a positive integer)")
    return True


def load_lead(calibration: str, pins: str) -> tuple:
    """-> (L, stub, pins_ref): L = {"build_commit", "record": (path, commit, blob_sha, json_sha), "spec", "spec_sha256",
    "selection": (path, commit, sha) | None, "selection_ref": {path, commit, sha256, line, kind}}. per_origin: the reviewed constants in LEAD;
    pooled: a committed pins JSON (read from HEAD, so the tree's committed state is what runs); a non-stub pooled pin needs build_commit,
    spec, spec_sha256 and a selection (an estimator selection JSON, or a RULE-file pin of kind "RULE ..." with its line)."""
    if calibration == "per_origin":
        if pins:
            raise SystemExit("--pins applies to the pooled calibration only")
        return LEAD["per_origin"], False, None
    path = pins or PINS_FILE[calibration]
    pb = git("show", f"HEAD:{path}")
    j = json.loads(pb)
    if j.get("calibration") != calibration:
        raise SystemExit(f"{path}: calibration {j.get('calibration')!r} != {calibration!r}")
    r_, s_ = j.get("record") or {}, j.get("selection") or {}
    L = {"build_commit": j.get("build_commit"), "record": (r_.get("path"), r_.get("commit"), r_.get("blob_sha256"), r_.get("json_sha256")),
         "spec": j.get("spec"), "spec_sha256": j.get("spec_sha256"),
         "selection": (s_.get("path"), s_.get("commit"), s_.get("sha256")) if any(v for k, v in s_.items() if k in ("path", "commit", "sha256")) else None,
         "selection_ref": {k: s_.get(k) for k in ("path", "commit", "sha256", "line", "kind") if s_.get(k) is not None}}  # a RULE-file pin (0d): file sha + line
    stub = bool(j.get("stub"))
    missing = [k for k, v in (("record", all(L["record"])),) if not v] + \
              ([] if stub else [k for k, v in (("build_commit", L["build_commit"]), ("spec", L["spec"]), ("spec_sha256", L["spec_sha256"]),
                                               ("selection", L["selection"] and all(L["selection"]))) if not v])
    if missing:
        raise SystemExit(f"HELD: {path} lacks {missing} (the {calibration} pins are not filled in yet)")
    if stub and L["spec"] and not L["spec_sha256"]:  # reviewer B optional: a stub that names a spec must pin its sha too
        raise SystemExit(f"{path}: a stub with a driver spec but no spec_sha256")
    unfilled = sorted({s for s in strings_in(j) if "<FILL" in s or s == "TBD"})
    if unfilled:  # route C: the A / B / R37 pins are fill tokens until S2 (fail closed, before any git read of a pinned file)
        raise SystemExit(f"HELD: {path} still holds {len(unfilled)} fill token(s) (filled at the S2 go)")
    ifp, rm = j.get("inputs_full"), j.get("record_meta")
    if (ifp is None) != (rm is None):  # route C: the FROZEN INPUTS_FULL and the record's meta are pinned together, or neither (the 110447 default)
        raise SystemExit(f"{path}: inputs_full and record_meta go together")
    if ifp is not None:
        if not (isinstance(ifp, dict) and set(ifp) == {"commit", "path", "sha256"} and isinstance(rm, dict)
                and set(rm) == {"git", "script_sha256", "inputs_full_sha256"} and rm["inputs_full_sha256"] == ifp["sha256"]
                and rm["git"] == L["build_commit"] and all(isinstance(v, str) and v for v in (*ifp.values(), *rm.values()))):
            raise SystemExit(f"{path}: inputs_full must be {{commit, path, sha256}} and record_meta {{git (= build_commit), script_sha256, "
                             "inputs_full_sha256 (= inputs_full.sha256)}}")
        L |= {"inputs_full": (ifp["commit"], ifp["path"], ifp["sha256"]), "record_meta": rm}
    return L, stub, {"path": path, "commit": git("rev-parse", "HEAD").decode().strip(), "sha256": hashlib.sha256(pb).hexdigest()}


def check_routec_selection(sel: dict, record: tuple) -> None:
    """Route C (09 RULE item 40 via 2e, 10-17): the pinned selection is route C A's OWN record (make_routec_selection.py, keys only); it must
    read (A) and cite exactly this lead record (path, commit, blob and JSON sha256), else STOP."""
    rp, rc, rgz, rjs = record
    if sel.get("lead") != "A" or sel.get("record") != {"path": rp, "commit": rc, "blob_sha256": rgz, "json_sha256": rjs}:
        raise SystemExit("STOP: the route C selection record does not read (A) for this lead record")


def strings_in(x):
    """Every string value (and dict key) in a parsed JSON config."""
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from strings_in(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings_in(v)


def record_structure(full: dict, calibration: str) -> dict:
    """The lead record's structure as this script needs it (keys only): the 24 lead rows with their three draw-0 fields, and under the pooled
    calibration a pooled pools file per label in meta.pooled_world.manifest_per_label['<label> | <combo>'] = {path, sha256, ...}, which every
    lead row of the label must carry unchanged (path, sha256) in its own pooled_world.pools_file."""
    cal = CALIB[calibration]
    need = ("pools_at_draw0", "cohort_cell_keys", "cell_hits_draw0")
    rows_missing, lacking = [], {}
    for lb in LABELS:
        for o in ORIGINS:
            r = ((full.get("rows") or {}).get(lb) or {}).get(cal["row"].format(o=o))
            if not isinstance(r, dict):
                rows_missing.append(f"{lb} | {cal['row'].format(o=o)}")
            elif [k for k in need if k not in r]:
                lacking[f"{lb} | {cal['row'].format(o=o)}"] = [k for k in need if k not in r]
    out = {"row_name_form": cal["row"], "rows_missing": rows_missing, "rows_lacking_fields": lacking}
    if calibration == "pooled":  # the pooled build (stub @ 1c7418b1b): meta.pooled_world.manifest_per_label["<label> | <combo>"] = {path, sha256, ...}
        mpl = ((full.get("meta") or {}).get("pooled_world") or {}).get("manifest_per_label") or {}
        refs = {lb: mpl.get(f"{lb} | {cal['combo']}") for lb in LABELS}
        out["pooled_files"] = refs
        out["labels_without_pooled_file"] = sorted(lb for lb, v in refs.items() if not (isinstance(v, dict) and v.get("path") and v.get("sha256")))
        disagree = []
        for lb in LABELS:  # every lead row of the label must carry the same pools file (path, sha256) in its own pooled_world.pools_file
            for o in ORIGINS:
                pf = ((((full.get("rows") or {}).get(lb) or {}).get(cal["row"].format(o=o)) or {}).get("pooled_world") or {}).get("pools_file") or {}
                if not (isinstance(refs[lb], dict) and (pf.get("path"), pf.get("sha256")) == (refs[lb].get("path"), refs[lb].get("sha256"))):
                    disagree.append(f"{lb} | {cal['row'].format(o=o)}")
        out["rows_whose_pools_file_differs_from_the_manifest"] = disagree
    out["ok"] = not rows_missing and not lacking and not out.get("labels_without_pooled_file") and not out.get("rows_whose_pools_file_differs_from_the_manifest")
    return out


def build_code(commit: str) -> dict:
    """The build's modules at the commit: BUILD_FILES, plus each BUILD_FILES_IF_PRESENT file the commit has. A build whose runner imports
    copies at top level without copies.py at that commit STOPs here (it could not import in the container)."""
    c = git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    have = set(git("ls-tree", "--name-only", f"{c}:{BUILD_DIR}").decode().splitlines())
    code = {n: git("show", f"{c}:{BUILD_DIR}/{n}").decode() for n in BUILD_FILES + tuple(n for n in BUILD_FILES_IF_PRESENT if n in have)}
    if re.search(r"^import copies\b", code["run_peraction311.py"], re.M) and "copies.py" not in code:
        raise SystemExit("the build's run_peraction311.py imports copies, but copies.py is not at the build commit: STOP")
    return code, c


def render_entries(rec: dict) -> tuple:
    """The drawn units of a tables record that carry a locator (status ok), each with a readable 'where'; the gated ones listed by place."""
    ents, gated = [], []
    for oc, d in rec["outcome_table"]["outcomes"].items():
        if d["drawn"] == "none":
            continue
        for side in ("real (run 1)", "sim (sim 1)"):
            v = d["drawn"][side]
            if v == "no command":
                continue
            (ents if v["status"] == "ok" else gated).append(v | {"where": f"outcome table | {oc} | {d['drawn']['cell']} | {side}"})
    for i, v in enumerate(rec["false_alarms"]["draws"]):
        (ents if v["status"] == "ok" else gated).append(v | {"where": f"false alarm {i + 1} | {v['probe']} | {v['pool']}"})
    return ents, [g["where"] + " | " + g["status"] for g in gated]


def draw_keys(rec: dict) -> dict:
    """The selection of a tables record by key (cells, runs, units, pools), without locators: two records with equal draw_keys drew the same."""
    t1 = {oc: (d["n"], d["drawn"] if d["drawn"] == "none" else (d["drawn"]["cell"], tuple(
        (v if v == "no command" else (v["run"], v["unit_index"])) for v in (d["drawn"]["real (run 1)"], d["drawn"]["sim (sim 1)"]))))
          for oc, d in rec["outcome_table"]["outcomes"].items()}
    t2 = [(d["probe"], d["pool"], d["run"], d["unit"] if isinstance(d["unit"], int) else d["unit"]["row_unit"]) for d in rec["false_alarms"]["draws"]]
    return {"outcomes": t1, "false_alarms": t2, "events": rec["false_alarms"]["events_per_probe_pool"],
            "dropped": [(d["probe"], d["pool"], d["run"], d["unit"]) for d in rec["false_alarms"].get("duplicates_dropped", [])]}


LOC_SAME = ("run_key", "token", "score_file", "tool", "unit", "run_eval", "field", "length", "sha256", "arguments_length", "arguments_sha256")


def locator_continuity(old: dict, new: dict) -> dict:
    """A locator-only re-run (both reviewers, locator fix): the drawn slots paired in a fixed order (each drawn outcome cell's two sides, then
    the false alarms by index). A slot located ("ok") in the earlier record must stay ok with identical locator fields (LOC_SAME, and
    staged_row except its entry index `call`); `call` may change only for a command-kind unit. Any other change STOPs. -> per-slot classes
    and counts (keys only)."""
    def slots(rec):
        out = []
        for oc, d in sorted(rec["outcome_table"]["outcomes"].items()):
            if d["drawn"] != "none":
                out += [(f"outcome | {oc} | {sd}", d["drawn"][sd]) for sd in ("real (run 1)", "sim (sim 1)")]
        return out + [(f"false alarm {i + 1}", v) for i, v in enumerate(rec["false_alarms"]["draws"])]

    so, sn = slots(old), slots(new)
    if [w for w, _ in so] != [w for w, _ in sn]:
        raise SystemExit("STOP: the re-run's drawn slots differ from the earlier record's; nothing written")
    per, counts = [], {}
    for (w, a), (_, b) in zip(so, sn):
        if a == "no command" or b == "no command":
            if a != b:
                raise SystemExit(f"STOP: {w}: a 'no command' side changed; nothing written")
            cls = "no command"
        elif a.get("status") == "ok":
            sr_ = lambda v: {k: x for k, x in v.get("staged_row", {}).items() if k != "call"}  # noqa: E731
            if b.get("status") != "ok" or any(a.get(k) != b.get(k) for k in LOC_SAME) or sr_(a) != sr_(b):
                raise SystemExit(f"STOP: {w}: a previously located unit's locator changed; nothing written")
            if a["staged_row"]["call"] == b["staged_row"]["call"]:
                cls = "unchanged"
            elif b.get("readout_kind") == "command":
                cls = "staged entry index changed (command entry, same call)"
            else:
                raise SystemExit(f"STOP: {w}: the staged entry index changed for a {b.get('readout_kind')} unit; nothing written")
        else:
            cls = "re-located (was not retrievable)" if b.get("status") == "ok" else "still not retrievable"
        per.append({"slot": w, "class": cls})
        counts[cls] = counts.get(cls, 0) + 1
    return {"per_slot": per, "counts": counts}


def drawn_zmt(rec: dict) -> list:
    """z - T of every drawn unit, in a fixed order (outcome cells, then the six false alarms)."""
    out = []
    for oc, d in sorted(rec["outcome_table"]["outcomes"].items()):
        if d["drawn"] != "none":
            out += [v["z_minus_T"] for v in (d["drawn"]["real (run 1)"], d["drawn"]["sim (sim 1)"]) if v != "no command"]
    return out + [d["z_minus_T"] for d in rec["false_alarms"]["draws"]]


@app.local_entrypoint()
def main(mode: str = "", calibration: str = "pooled", build_commit: str = "", index: str = "", index_sha: str = "", go_stage_2: bool = False,
         text_out: str = "", expect: str = "", expect_sha: str = "", pins: str = "", dry_run: bool = False, use_pw_calib: bool = False,
         z_from_calib: str = "", run_label: str = "", label_sidecar: str = ""):
    """Route C (10-17): --pins names a pins file with inputs_full + record_meta (re-asserted against the record in the container);
    --label-sidecar git:<commit>:<path>.json@<sha256> (tables mode; not with --run-label) composes the record's run_label from A's sidecar in
    the container. Render's --text-out stays under q2's SCRATCH only (0d: the example-text exception keeps its 10-16 scope). The tables
    mode prints the record's path, sha256, outcome and gated-row count only; the table numbers stay in the record (0d, 10-17: they reach
    writing through the slot readout)."""
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    if calibration not in CALIB:
        raise SystemExit(f"--calibration {tuple(CALIB)}")
    if dry_run and mode != "tables":  # Codex p1 #1: a dry run is a tables-mode run only (never render)
        raise SystemExit("--dry-run applies to --mode tables only")
    if (use_pw_calib or z_from_calib) and mode not in ("tables", "check"):
        raise SystemExit("--use-pw-calib / --z-from-calib apply to --mode tables / check only")
    if mode in ("tables", "check"):
        check_calib_options(calibration, use_pw_calib, z_from_calib, dry_run)
    if mode == "check" and (dry_run or not (expect and expect_sha)):
        raise SystemExit("--mode check needs --expect and --expect-sha (the earlier record's draws) and no --dry-run")
    if run_label and not (mode == "tables" and re.fullmatch(r"[A-Za-z0-9 ;:,.()_+/-]{1,120}", run_label)):
        raise SystemExit("--run-label: tables mode only; up to 120 plain characters (the coordinator's label for this run, stored in the record)")
    if label_sidecar and (mode != "tables" or run_label or dry_run or not SIDECAR_RX.fullmatch(label_sidecar)):
        raise SystemExit("--label-sidecar: tables mode only, not with --run-label or --dry-run, in the form git:<40-hex>:<path>.json@<64-hex>")
    if mode == "importcheck":  # code only: the WIP or pinned build commit, no data, no values
        code, c = build_code(build_commit or load_lead(calibration, pins)[0]["build_commit"])  # an explicit --build-commit needs no pins
        print(json.dumps({"build_commit": c} | job.remote("importcheck", {"code": code})), flush=True)
        return
    if mode == "render":  # Stage 2: q2's conditional go (committed keys-only record after a Codex PASS, every gate passed)
        if not go_stage_2:
            raise SystemExit("render is Stage 2: --go-stage-2 only")
        out_ = Path(text_out).resolve()
        if X.SCRATCH.resolve() not in out_.parents:
            raise SystemExit("--text-out must be under this session's scratchpad")
        ip, ic = index.rsplit("@", 1)
        ic = git("rev-parse", "--verify", f"{ic}^{{commit}}").decode().strip()
        ib = git("show", f"{ic}:{ip}")
        if hashlib.sha256(ib).hexdigest() != index_sha:  # hashed only; _render parses it in the container
            raise SystemExit("render: the tables record differs from --index-sha")
        got = job.remote("render", {"record_bytes": ib, "index_sha": index_sha, "stage_2": X.STAGE2_GO})
        me = str(Path(__file__).resolve().relative_to(REPO))
        if not re.fullmatch(r"[0-9a-f]{40}", str(got["git"])) or hashlib.sha256(git("show", f"{got['git']}:{me}")).hexdigest() != got["script_sha256"]:
            raise SystemExit("render: the record was not written by this script at its recorded commit: nothing written")
        res, gated = got["entries"], got["gated"]
        with out_.open("x") as fh:  # the only place the text is written; never printed
            json.dump({"index": {"path": ip, "commit": ic, "sha256": index_sha}, "label": got["label"], "run_label": got["run_label"], "entries": res,
                       "gated_not_rendered": gated}, fh, indent=1)
        print(json.dumps({"text_file": str(out_)}), flush=True)  # a5b40fd5 R3: no count of drawn or gated units on the laptop
        return
    if mode not in ("tables", "check"):
        raise SystemExit("--mode importcheck | tables | check | render")
    if expect:  # the earlier record's pin, checked before any paid call (reviewers A #4 / B #6); the commit resolved to a full sha (B re-glance)
        ep_, ec_ = expect.rsplit("@", 1)
        ec_ = git("rev-parse", "--verify", f"{ec_}^{{commit}}").decode().strip()
        eb_ = git("show", f"{ec_}:{ep_}")
        if hashlib.sha256(eb_).hexdigest() != expect_sha:
            raise SystemExit("--expect: the earlier record's sha256 differs from --expect-sha")
    L, stub, pins_ref = load_lead(calibration, pins)
    if stub and not dry_run:
        raise SystemExit("a STUB pins file runs only with --dry-run (no selection, no record of draws, no render)")
    code, c = build_code(L["build_commit"]) if L["build_commit"] else (None, None)  # a stub without its build commit: structure + poolcheck only
    rp, rc, rgz, rjs = L["record"]
    rb = git("show", f"{rc}:{rp}")
    if hashlib.sha256(rb).hexdigest() != rgz:  # the laptop only hashes the lead record; _full decompresses, sha-checks the JSON and parses it
        raise SystemExit("the lead record's git blob sha256 differs from the pin")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    full_p = {"lead_bytes": rb, "lead_path": rp, "lead_json_sha256": rjs, "build_commit": c, "calibration": calibration, "rule_pin": is_rule_pin(L),
              "stub": stub, "pins_ref": pins_ref, "stamp": stamp, "git": git("rev-parse", "HEAD").decode().strip(), "mode": mode, "dry": dry_run,
              "meta": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lead": {k: v for k, v in L.items()}, "run_label": run_label},
              "tables": {"code": code, "record_name": volume_name(rp), "spec_path": L["spec"], "spec_sha256": L.get("spec_sha256"),
                         "calibration": calibration, "overrides": {}, "inventories": [], "use_pw_calib": use_pw_calib, "z_from_calib": z_from_calib}}
    ifp_ = L.get("inputs_full") or X.PINS["inputs_full"]  # route C: the pins file's FROZEN file; else 110447 (the 171-pool and per-set draws)
    full_p |= {"record_meta": L.get("record_meta"), "inputs_full_sha256": ifp_[2]}
    if label_sidecar:  # the sidecar's bytes go to the container unparsed (09: read by path@sha there)
        lc_, lp_, lh_ = SIDECAR_RX.fullmatch(label_sidecar).groups()
        lb_ = git("show", f"{lc_}:{lp_}")
        if hashlib.sha256(lb_).hexdigest() != lh_:
            raise SystemExit("--label-sidecar: the file's sha256 differs from the pointer")
        full_p["label_sidecar"] = {"bytes": lb_, "sha256": lh_}
        full_p["meta"]["label_sidecar"] = {"commit": lc_, "path": lp_, "sha256": lh_}

    def write(sub_dir: str, name: str, outcome: str = "gates passed (in the container)") -> None:  # the container's record bytes, written as they are (never parsed here) and sha-checked
        got = job.remote("full", full_p)
        b = got["record_text"].encode()
        if hashlib.sha256(b).hexdigest() != got["sha256"]:
            raise SystemExit("the record bytes differ from the container's sha256: nothing written")
        dest = HERE / "results" / sub_dir / name
        dest.parent.mkdir(exist_ok=True)
        with dest.open("xb") as fh:
            fh.write(b)
        print(json.dumps({"record": str(dest.relative_to(REPO)), "sha256": got["sha256"], "outcome": outcome} | got["counts"]), flush=True)

    if dry_run:  # structure, pooled pools files and (with a spec) the rows to draw 0, all in the container; nothing is selected
        write("dryrun", f"examples_v81_tables_{calibration}_DRYRUN_{stamp}.json", "dry run written (gate outcomes recorded, nothing selected)")
        return
    sp_, sc_, sh_ = L["selection"]
    sb_ = git("show", f"{sc_}:{sp_}")
    if is_rule_pin(L) and not L["selection_ref"]["line"] <= sb_.count(b"\n") + 1:
        raise SystemExit("the selection pin's line is outside the pinned file")
    if hashlib.sha256(sb_).hexdigest() != sh_:  # (load_lead guarantees a full selection pin outside a stub; a RULE-file pin is checked the same way)
        raise SystemExit("the selection record's sha256 differs from the pin")
    if str(L["selection_ref"].get("kind", "")).startswith(ROUTEC_SELECTION_KIND):  # K: written by make_routec_selection.py (keys only)
        check_routec_selection(json.loads(sb_), L["record"])
    inf = json.loads(X.pinned_at(*ifp_, name="inputs_full"))  # K: a FROZEN INPUTS_FULL config (writer inputs_full.py build), hashed before the parse
    ovs = {}
    for k in OVERRIDES:
        e = inf["overrides"][k]
        b = git("show", f"{e['commit']}:{e['path']}")
        if hashlib.sha256(b).hexdigest() != e["sha256"]:
            raise SystemExit(f"override {k}: sha256 differs from INPUTS_FULL")
        ovs[k] = json.loads(b)
    invs = []
    if set(inf["inventories"]) != set(INVENTORY_ORDER):
        raise SystemExit("INPUTS_FULL's inventories differ from INVENTORY_ORDER")
    man_cache = {}

    def eval_sha_of(r):  # an inventory row without eval_sha256: its routing record's .eval manifest (git, path @ commit), by the key's path
        rt = (r.get("uncut_planned") or {}).get("routing") or ""
        if " @ " not in rt:
            return None
        if rt not in man_cache:
            rp_, rc_ = rt.split(" @ ")
            mm = json.loads(git("show", f"{rc_}:{rp_}")).get("manifest") or ""
            if " @ " not in mm:
                man_cache[rt] = {}
            else:
                mp_, mc_ = mm.split(" @ ")
                man_cache[rt] = {f["path"].removeprefix("runs/"): f["sha256"] for f in json.loads(git("show", f"{mc_}:{mp_}")).get("files", [])}
        return man_cache[rt].get(r["key"].split("#e")[0].removeprefix("runs/"))

    e12p = inf["e12"]  # 09 (via q2 ~14:5x): epoch-2 runs resolve from INPUTS_FULL's E12 block BY KEY, checked against its sorted-key sha256s
    e12b = git("show", f"{e12p['commit']}:{e12p['path']}")
    if hashlib.sha256(e12b).hexdigest() != e12p["sha256"]:
        raise SystemExit("the E12 block's sha256 differs from INPUTS_FULL")
    e12 = json.loads(e12b)
    e12_sha = {}
    hs_ = lambda xs: hashlib.sha256(json.dumps(sorted(xs)).encode()).hexdigest()  # noqa: E731  (inputs_full.py's freeze-check convention)
    phys_ = lambda ents: sorted({f"{e['launch_id']}/{e['eval']}#e0" for e in ents})  # noqa: E731
    fcp_ = inf["freeze_checks"].get("e12_pools") or {}
    seen_pools = set()
    for o_, d_ in (e12.get("pools") or {}).items():  # every pool, direct or H_A family-grouped, as inputs_full.py checks them (Codex b14 #1)
        for pl_, x_ in d_.items():
            if pl_ == "H_A_group_of_family":
                continue  # the family -> H_A group map, not a pool
            if isinstance(x_, dict) and isinstance(x_.get("e2"), list):
                ents_ = x_["e2"]
            elif pl_ == "H_A" and isinstance(x_, dict) and x_ and all(isinstance(g, dict) and isinstance(g.get("e2"), list) for g in x_.values()):
                ents_ = [e_ for g in x_.values() for e_ in g["e2"]]
            else:
                raise SystemExit(f"E12 block {o_} {pl_}: neither {{e2: [...]}} nor H_A family groups")
            fc_ = (fcp_.get(o_) or {}).get(pl_) or {}
            if fc_.get("physical_members_sha256") != hs_(phys_(ents_)) or fc_.get("e2_members_sha256") != hs_({e_["key_e2"] for e_ in ents_}):
                raise SystemExit(f"E12 block {o_} {pl_}: members differ from INPUTS_FULL's sorted-key sha256s")
            seen_pools.add((o_, pl_))
            for e_ in ents_:
                pk_ = f"{e_['launch_id']}/{e_['eval']}#e0"
                if e_.get("inventory_key", pk_) != pk_ or e12_sha.setdefault(pk_.removeprefix("runs/"), e_["eval_sha256"]) != e_["eval_sha256"]:
                    raise SystemExit("E12 block: a physical key with two .eval shas or an inventory key that is not its launch/eval key")
    if seen_pools != {(o_, pl_) for o_, d_ in fcp_.items() for pl_ in d_}:
        raise SystemExit("E12 block: the checked pools differ from INPUTS_FULL's freeze_checks.e12_pools")
    phys_sha = {}
    for k in INVENTORY_ORDER:
        e = inf["inventories"][k]
        b = git("show", f"{e['commit']}:{e['path']}")
        if hashlib.sha256(b).hexdigest() != e["sha256"]:
            raise SystemExit(f"inventory {k}: sha256 differs from INPUTS_FULL")
        j = json.loads(b)
        for r in j.get("runs") or []:
            if not (isinstance(r, dict) and r.get("key")):
                continue
            ek_ = r["key"].removeprefix("runs/")  # both sides runs/-normalized (the map is normalized at build; reviewer B optional)
            have_ = {src: h for src, h in (("inventory", r.get("eval_sha256")), ("INPUTS_FULL's E12 block, by key", e12_sha.get(ek_)),
                                           ("the routing record's .eval manifest", eval_sha_of(r))) if h}
            if len(set(have_.values())) > 1:  # every available source must agree, the inventory's own value included (Codex b14 #2)
                raise SystemExit(f"{r['key']}: .eval sha256 sources disagree ({sorted(have_)})")
            if have_ and not r.get("eval_sha256"):
                src_ = next(k_ for k_ in ("INPUTS_FULL's E12 block, by key", "the routing record's .eval manifest") if k_ in have_)
                r["eval_sha256"], r["eval_sha256_source"] = have_[src_], src_
        for r in j.get("runs") or []:  # reviewers' R1 + (b): ONE .eval sha per (runs/-normalized) key across ALL inventories, every row with a sha
            if isinstance(r, dict) and r.get("key") and r.get("eval_sha256"):
                nk_ = r["key"].removeprefix("runs/")
                if phys_sha.setdefault(nk_, r["eval_sha256"]) != r["eval_sha256"] and calibration == "pooled":  # the STOP applies to pooled only
                    raise SystemExit(f"STOP: pooled: key {nk_}: the pinned inventories give two different .eval sha256s (physical identity conflict)")
        invs.append((k, json.dumps(j)))
    full_p["tables"] |= {"phys_sha": phys_sha, "overrides": ovs, "inventories": invs}
    if expect:  # the earlier record's bytes (sha-checked above, never parsed here) for the container's same-draws and continuity gates
        full_p["expect"] = {"bytes": eb_, "path": ep_, "commit": ec_, "sha256": expect_sha}
    if mode == "check":
        write("check", f"examples_v81_tables_{calibration}_CHECK_{stamp}.json")
    else:
        write("", f"examples_v81_tables_{calibration}_{stamp}.json")


def selftest(build_commit: str = "20c6fbd20") -> None:
    """Pure synthetic checks of alias_ok with the PINNED build's own helpers (e12_cell, relaunch_alias, SP_WORLD_ALIAS), extracted from its
    source by name (the build module itself is not imported on the laptop). No data."""
    import ast
    import types

    global _R
    src = git("show", f"{build_commit}:{BUILD_DIR}/run_peraction311.py").decode()
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ("e12_cell", "relaunch_alias"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "SP_WORLD_ALIAS" for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "build_helpers", "exec"), ns)  # noqa: S102 -- the pinned build's own three helpers
    _R = types.SimpleNamespace(**{k: ns[k] for k in ("e12_cell", "relaunch_alias", "SP_WORLD_ALIAS")})
    cases = [("honest_b|task_e2b_redraw2", "honest_b_e2|task", True), ("honest_b|task_redraw2_e2a", "honest_b_e2|task", True),
             ("honest_b|task_e2a", "honest_b_e2|task", True), ("honest_b|task", "honest_b|task", True), ("sim|cell", "run1|cell", False),
             ("attack|cell", "run1|cell", True), ("sim_honest|cell", "sp493_t196|cell", True), ("honest_b_relaunch|t5_r2", "honest_r2|t5", True),
             ("honest_b_relaunch|t5_r2", "honest_r3|t5", False), ("run1|cellA", "run1|cellB", False), ("honest_b|task_e2b", "honest_c_e2|task", False)]
    bad = [(a, b) for a, b, want in cases if alias_ok(a, b) != want]
    if bad:
        raise AssertionError(f"alias_ok: {bad}")
    import numpy as np

    def ent(key, rk, sc, tok=(10, 20)):
        return {"file_key": key, "path": "p", "sha256": "h", "volume": "conv", "staged_shas": [], "s": np.array(sc, float),
                "sbm": np.full((len(sc), 5), np.nan), "run_key": np.array([rk] * len(sc)), "token": np.array(tok[:len(sc)]), "call_index": None}

    fx = FileIndex.__new__(FileIndex)
    fx.by_cell = {}
    for e in (ent("honest_b|task", "L/e1#e0", [1.0, 2.0]), ent("honest_b|task_e2b_redraw2", "L/e2#e1", [3.0, 4.0]),
              ent("attack|cellx", "L/r1#e0", [5.0, 6.0]), ent("attack|cellx", "L/r2#e0", [7.0, 8.0]),
              ent("honest_b_relaunch|t5_r2", "L/rl#e0", [9.0, 1.5]), ent("sim|celly", "L/s1#e0", [2.5, 3.5]), ent("sim|celly", "L/s9#e0", [2.5, 3.5])):
        fx.by_cell.setdefault(lookup_cell(e["file_key"]), []).append(e)
    nan5 = np.full((2, 5), np.nan)
    got = [(a, str(fx.match(a, np.array(sc), nan5)["run_key"][0])) for a, sc in
           (("honest_b_e2|task", [3.0, 4.0]), ("honest_b|task", [1.0, 2.0]), ("run2|cellx", [7.0, 8.0]), ("run1|cellx", [5.0, 6.0]),
            ("honest_r2|t5", [9.0, 1.5]))]
    want = [("honest_b_e2|task", "L/e2#e1"), ("honest_b|task", "L/e1#e0"), ("run2|cellx", "L/r2#e0"), ("run1|cellx", "L/r1#e0"), ("honest_r2|t5", "L/rl#e0")]
    if got != want:
        raise AssertionError(f"FileIndex.match: {got}")
    assert fx.match("honest_b_e2|task", np.array([1.0, 9.0]), nan5, none_ok=True) is None  # zero hits: None only with none_ok
    try:  # two physical runs with identical scores STOP even with none_ok (reviewers A / B: only the zero-hit case is skippable)
        fx.match("sim|celly", np.array([2.5, 3.5]), nan5, none_ok=True)
        raise AssertionError("FileIndex.match(none_ok=True) accepted an ambiguous run")
    except SystemExit:
        pass
    for a, sc in (("sim|celly", [2.5, 3.5]), ("honest_b_e2|task", [1.0, 9.0]), ("run1|cellx", [7.0, 8.5])):
        try:  # two physical runs with identical scores; no matching scores; a near miss
            fx.match(a, np.array(sc), nan5)
        except SystemExit:
            continue
        raise AssertionError(f"FileIndex.match accepted {a} {sc}")
    def rowpair(z, T, Trec):
        exr = {"label": "t", "origin": "o", "coh": ["c1"], "T": {"B": T, "S": T}, "fold_model": False, "folds": [], "mom": {"B": None, "S": None},
               "pools": {nm: {"run": np.array([f"r{i}" for i in range(len(z))]), "z0": np.array(z, float), "s": np.array(z, float)} for nm in ("B", "S")},
               "arms": {w: {"hit": np.array([False])} for w in ("run1", "sim")}}
        runs = {f"r{i}": v for i, v in enumerate(z)}
        pd = {"T": Trec, "n_units": len(z), "units_above": int(sum(v > T for v in z)), "runs": len(z),
              "per_run_alarm_share": float(np.mean([v > T for v in runs.values()]))}
        return exr, {"pools_at_draw0": {"0.01": {"B": pd, "S": pd}}, "cohort_cell_keys": ["c1"], "cell_hits_draw0": {"levels": {"0.01": {"run1": "0", "sim": "0"}}}}

    z_ = [0.1 * i for i in range(100)]
    T_ = z_[98]
    assert abs(gate_row(*rowpair(z_, T_, T_ + 5.7e-14)) - 5.7e-14) < 1e-15  # drift within tolerance, no unit near T: passes, dT returned
    for zz, Tr in (([*z_[:97], T_ - 1e-13, T_, z_[99]], T_ + 5.7e-14), (z_, T_ + 2e-12)):  # a unit next to T (a possible swap); drift above 1e-12
        try:
            gate_row(*rowpair(zz, T_, Tr))
        except SystemExit:
            continue
        raise AssertionError("gate_row accepted a near-T unit or a drift above 1e-12")
    # z_bound replay (Codex b7): two independent summation orders (numpy's dot, and a reversed, blocked float64 loop) and the exact value
    # (math.fsum) of the build's moment formula; every unit's two computed z must lie within z_bound of the exact z
    import math

    rng = np.random.default_rng(7)
    for scale, off in ((1.0, 0.0), (0.5, 3.0), (50.0, -7.0)):  # logit-like pools; an ill-conditioned pool (mean / sd ~ 1e6) STOPs instead (below)
        x = rng.standard_normal(5040) * scale + off
        mom_ = {"mu": {0: float(np.dot(np.ones(len(x)), x) / len(x))}}
        mom_["sd"] = {0: float(np.sqrt(max(np.dot(np.ones(len(x)), x * x) / len(x) - mom_["mu"][0] ** 2, 0.0)))}
        P_ = {"s": x, "f": np.zeros(len(x), int), "sbm": np.full((len(x), 5), np.nan), "col_moments": False, "oof": False}
        P_["z0"] = (x - mom_["mu"][0]) / mom_["sd"][0]
        zb_ = z_bound(P_, mom_, True, [0])
        mu_x = math.fsum(x) / len(x)
        var_x = math.fsum(v * v for v in x) / len(x) - mu_x * mu_x
        z_exact = (x - mu_x) / math.sqrt(var_x)
        acc = np.float64(0.0)
        acc2 = np.float64(0.0)
        for blk in np.array_split(x[::-1], 63):
            acc, acc2 = acc + np.float64(sum(blk.tolist())), acc2 + np.float64(sum((blk * blk).tolist()))
        mu_r = float(acc) / len(x)
        z_rev = (x - mu_r) / math.sqrt(max(float(acc2) / len(x) - mu_r * mu_r, 0.0))
        assert (np.abs(P_["z0"] - z_exact) <= zb_).all() and (np.abs(z_rev - z_exact) <= zb_).all(), f"z_bound fails at scale {scale}"
    # the complete fold -1 (21b) path against a high-precision reference (math.fsum at every reduction)
    nf, nu = 600, 300
    Pm = {"s": rng.standard_normal(nf) * 2 - 1, "f": np.r_[np.repeat(np.arange(5), nf // 5)], "sbm": np.full((nf, 5), np.nan), "col_moments": False, "oof": False}
    mom1 = {"mu": {}, "sd": {}}
    for k in range(5):
        xk = Pm["s"][Pm["f"] == k]
        mom1["mu"][k] = float(np.dot(np.ones(len(xk)), xk) / len(xk))
        mom1["sd"][k] = float(np.sqrt(np.dot(np.ones(len(xk)), xk * xk) / len(xk) - mom1["mu"][k] ** 2))
    S1 = rng.standard_normal((nu, 5)) * 2 - 1
    P1 = {"s": np.r_[Pm["s"], np.full(nu, np.nan)], "f": np.r_[Pm["f"], np.full(nu, -1)], "sbm": np.r_[np.full((nf, 5), np.nan), S1],
          "col_moments": False, "oof": False}
    MU, SD = np.array([[mom1["mu"][k] for k in range(5)]]), np.array([[mom1["sd"][k] for k in range(5)]])
    m1 = ((S1[None] - MU[:, None, :]) / SD[:, None, :]).mean(2)[0]
    mom1["mu0"] = float(np.dot(np.ones(nu), m1) / nu)
    mom1["sd0"] = float(np.sqrt(np.dot(np.ones(nu), m1 * m1) / nu - mom1["mu0"] ** 2))
    P1["z0"] = np.r_[(Pm["s"] - MU[0, Pm["f"]]) / SD[0, Pm["f"]], (m1 - mom1["mu0"]) / mom1["sd0"]]
    zb1 = z_bound(P1, mom1, True, list(range(5)))
    mu_e = {k: math.fsum(Pm["s"][Pm["f"] == k]) / (nf // 5) for k in range(5)}
    sd_e = {k: math.sqrt(math.fsum(v * v for v in Pm["s"][Pm["f"] == k]) / (nf // 5) - mu_e[k] ** 2) for k in range(5)}
    m_e = np.array([math.fsum((S1[i, k] - mu_e[k]) / sd_e[k] for k in range(5)) / 5 for i in range(nu)])
    mu0_e = math.fsum(m_e) / nu
    sd0_e = math.sqrt(math.fsum(v * v for v in m_e) / nu - mu0_e ** 2)
    z_e = np.r_[(Pm["s"] - np.array([mu_e[k] for k in Pm["f"]])) / np.array([sd_e[k] for k in Pm["f"]]), (m_e - mu0_e) / sd0_e]
    assert (np.abs(P1["z0"] - z_e) <= zb1).all(), "z_bound fails on the fold -1 path"
    for bad_x in (np.r_[np.full(100, 1e-160), 1.0, 2.0], np.full(100, 3.0) + np.r_[np.zeros(99), 1e-15], rng.standard_normal(5040) * 1e-3 + 1e3,
                  np.r_[np.full(64, 1.5e153), -1.5e153]):  # underflow; fp-level variance; mean / sd ~ 1e6; a possible overflow in sum(x^2)
        try:
            _mom_err(bad_x)
        except SystemExit:
            continue
        raise AssertionError("_mom_err accepted an underflow or a vanishing variance")
    print(f"selftest: alias_ok {len(cases)} cases, FileIndex.match 8 end-to-end cases, gate_row 3 cases, the z_bound replay (3 scales + the fold -1 path vs math.fsum) and 4 _mom_err STOPs pass with the helpers of {build_commit}")


if __name__ == "__main__" and sys.argv[1:2] == ["--selftest"]:
    selftest(*(sys.argv[2:3] or []))
