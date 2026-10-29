"""Route C sim-pool numbers (2e's SIM-POOL owner, 2026-10-20; POST HOC; keys and counts; NOT RUN until a review PASS and refresh3's ARMED
line). For each post number that depends on the simulated-honest pool, the record holds the 10-16 (v83) count recomputed from its own source by
its own code, the route C count by the same code, and a changed / unchanged flag at the post's printed precision.

Route C's pool: the role-S runs of the rcCAT catalogue (@ 16682ef28, decompressed sha256 72432318...), which pooled record A (@ a936a5d97,
decompressed 54afad04...) names in meta.pooled_world.catalogue. It is the 10-16 pool plus the 20 A25 pilot runs (world simh_pilot, S_old);
gate G_DELTA checks that by key on the honest detector's two records.
- Item 1, [^sim] (share of simulated honest runs where GPT-5.6-Sol wrote part of the output): item (c)'s sol_flags / cell_arrays / wauc,
  imported from run_detector_solfree_auroc.py's blob (@ d1db55766). The 20 pilot .evals are in none of item (c)'s usage sources, so their
  Inspect headers (header_only: model names with one any-tokens bit, status, sample count; no message, score or text) are read here with
  check_simh_cont_headers.headers, imported from its blob (@ 8ac62e6d6), over the pilot's upload manifest (@ 4af62db74). A pilot file whose
  header is not read is UNKNOWN and stays in the denominator. Shares are given three ways: UNKNOWN counted as not Sol (all-runs denominator,
  the 10-17 basis, where UNKNOWN was 0), UNKNOWN counted as Sol (upper bound), and UNKNOWN left out.
- Item 2, [^failed] (failed sim-honest share): appendix_a_rates (@ 73cd0f2be) by make_appendix_a_rates.py (@ 35dff7132, imported from its
  blob for COMMON). Its headline counts only the COMMON sets. Every pilot run must be a role-S run with sets exactly [S_old] (G_PILOT_SETS:
  any exception STOPs). The route C headline is the 10-16 headline plus the pilot runs in COMMON sets.
- Item 4: the honest detector's sim class per cell (honest_all_routec @ 596bc767e) against the catalogue's role S, by cell. Every extra or
  missing key is named.
- Item 5, Table 7 row 5: the a_vs_b control's cohort (run key -> eval path, fold, main) in the route C record against the 10-16 record
  (@ 8b0b18e2a), fail-closed, and its run-level AUROC at the post's 2 decimals.
Every input is git show'd and sha256-checked on the laptop and again in the container. The record bytes are built in the container and written to
this job's own Volume (exclusive create, committed) before they are returned (review R1). The laptop
writes them (exclusive create) and prints only keys, gate results, changed / unchanged counts and paths.
Usage (NOTE_launch_2026-10-20.md): "$M" run experiments/2026-10-20_routec_simpool_numbers/run_routec_simpool_numbers.py --out <results/...json>
"""

import gzip
import hashlib
import json
import re
import subprocess
import sys
import types
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-routecsim_numbers_kill-2026-11-19"
VOL = "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"  # check_simh_cont_headers.VOL (checked in the container)
HDR_ROOT = "/a25/simh_evals_v1"  # check_simh_cont_headers.ROOT (checked in the container)
OUT_VOL = "probe-shift_aswin-surya_sub-routecsim_results_kill-2026-11-19"  # this job's own results Volume (review R1: the record survives a client drop)
OUT_NAME = re.compile(r"routec_simpool_numbers_\d{4}-\d{2}-\d{2}_\d{6}\.json")  # the record's file name, on the Volume and in results/
N_PILOT = 20
GZ = ("catalogue", "record_A")
# name: (commit, path, sha256 of the git show bytes, sha256 of the decompressed bytes for GZ names)
PINS = {
    "solfree_script": ("d1db55766", "experiments/2026-10-15_e2e_sim_pipeline/run_detector_solfree_auroc.py",
                       "a82879f4c4159352812ec69c23db40ee0d747e31bc7c1e4396c3813b5c76e36e", None),
    "headers_script": ("8ac62e6d6", "experiments/2026-10-15_e2e_sim_pipeline/check_simh_cont_headers.py",
                       "b07159df3f426fce3fd502da60528ac478edb2005532b0599297609bca9b57ae", None),
    "rates_script": ("35dff7132", "experiments/2026-10-15_fpr_sweep_507_prep/make_appendix_a_rates.py",
                     "1689800e47843a910b6641928ac3806545be001b32df6cc62ac6c685d4fd2863", None),
    "scan_rows": ("ff8ea35a7", "experiments/2026-10-16_attack_prompt_provenance/results/openai_usage_2026-10-16_185845_scan_rows.json",
                  "0d0392fb773b1804a3ded3b4f3fd2f4dc8e7ae33b1ee660dd7200d0550410197", None),
    "usage": ("de0bf0301", "experiments/2026-10-16_attack_prompt_provenance/results/openai_usage_reanalysis_2026-10-16_191817.json",
              "a0bcb2cef1ec21d598dda1795cc8b027eaf49235ba191f5d4c9643e108701102", None),
    "cont_headers": ("4326d76e9", "experiments/2026-10-15_e2e_sim_pipeline/results/simh_cont_headers_2026-10-17_081840.json",
                     "a7638c2d19f6a56f2953babca0ea98a7fd63854cbec21be6f1861ea452941e5d", None),
    "pilot_manifest": ("4af62db74", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/manifest_simh_eval_upload_pilot.json",
                       "875167b9916560444f80be78bf516beba6711837a811bd677b82f66e704d182b", None),
    "honest_171": ("8b0b18e2a", "experiments/2026-10-16_honest_sim_detector_all/results/honest_all_2026-10-16_120353.json",
                   "c9ec2530e368c990136d536aa5ff74aefc3be44615698ed00f913f7b52d97b14", None),
    "honest_routec": ("596bc767e", "experiments/2026-10-16_honest_sim_detector_all/results/honest_all_routec_2026-10-18_121211.json",
                      "10ed5645e0a8357150e07ac6416a06c6196e995d01eb5088b94de1d554d73c46", None),
    "rates_171": ("73cd0f2be", "experiments/2026-10-15_fpr_sweep_507_prep/results/appendix_a_rates_2026-10-16_121443.json",
                  "3c034ec9c74cc6989936a84e15ae07124cc93d94d0fe6fcb4bc6a4e958fd6e64", None),
    "catalogue": ("16682ef28", "experiments/2026-10-15_fpr_sweep_507_prep/results/routec_pwm_2026-10-18/"
                  "alltasks_CRD_FULL_r2_allorigins_pooledworld_catalogue_e12_2026-10-18_112818_catalogue.json.gz",
                  "4dc103729860dccffbffb4bfce522fbf2d3124502a080a79324f4fe29126f5a3",
                  "72432318a3703d4675a5c4530d77f598a003af351340d895d305b9bd1050adc8"),
    "record_A": ("a936a5d97", "experiments/2026-10-15_fpr_sweep_507_prep/results/routec_records/"
                 "routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz",
                 "191c467ab095b0f9ddfe2271822655bc5b49ee02a1a2c5a5b0e11bc855a00865",
                 "54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a"),
}
CHAIN = ("1df288e61", "experiments/2026-10-14_q3_overnight_chain/config")  # make_appendix_a_rates.CHAIN: the sim launch manifests it reads
PILOT_LAUNCH = "140858_simh_pilot"  # the pilot launch 2026-10-14_140858_simh_pilot (its short id: the broader grep)
# Where the post's v83 numbers were read (sources only; this tool recomputes each count from the record named, never from these files)
V83_SOURCES = {
    "item1": "detector_solfree_2026-10-17_082153.json (sha256 8282cb5e..., outside git) .detectors.honest.cells[<cell>].n_class1_sol_using / "
             ".n_class1_all (key map KEY_MAP_detector_solfree_2026-10-17.md @ 2dc08fd89), from honest_all @ 8b0b18e2a; writing "
             "changes_pending 10-17 08:33 'honest 50/848 (5.9%), 0 unknown' -> 'about 6%'",
    "item2": "appendix_a_rates_2026-10-16_121443.json @ 73cd0f2be .headline_common_sets.Y_sim.{failed, launched, rate} and "
             ".by_category['infrastructure (rate limit)']; 4.3% = failed / launched, 2.9% = (failed - rate limit) / launched",
    "item4": "honest_all @ 8b0b18e2a .cells[<cell>].problems.simh_vs_b.run_scores (keys with prefix 'sim|')",
    "item5": "honest_all @ 8b0b18e2a .cells['command|L48' | 'reply|L48'].problems.a_vs_b.run_level.auroc (0d 10-18 11:15: 'row 5 unchanged')",
}
PRINTED = {"item1": 0, "item2": 1, "item5": 2}  # the post's printed decimals: "about 6%", "4.3% (2.9% ...)", "0.52"
# The post's printed v83 values (draft_v84 L69 [^sim], L28 [^failed], App E / Table 7 row 5) and writing's 10-17 relay of item (c)'s counts:
# the v83 recompute must reproduce them (review O3) before any route C comparison means anything. Compared as booleans only.
PRINTED_V83 = {"item1_percent": "6", "item1_counts_sol_n": (50, 848), "item2_percent": "4.3", "item2_percent_without_rate_limit": "2.9",
               "item5_auroc": "0.52"}
ROW5_CELLS = ("command|L48", "reply|L48")
RATE_LIMIT = "infrastructure (rate limit)"

image = modal.Image.debian_slim(python_version="3.12").pip_install("inspect_ai==0.3.251", "zstandard==0.25.0", "numpy==2.1.3")
app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOL, version=2).read_only()
out_vol = modal.Volume.from_name(OUT_VOL, version=2, create_if_missing=True)
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


class Stop(SystemExit):
    """A gate failure: the message names the gate and keys, never a value."""


# ---------------------------------------------------------------- pinned scripts, imported from their blobs
class _Lazy:
    """Stands in for the modal objects a pinned script builds at import (Image, Volume): any attribute or call gives another _Lazy."""

    def __getattr__(self, name):
        return _Lazy()

    def __call__(self, *a, **k):
        return _Lazy()


class _App:
    """Stands in for modal.App in a pinned script: its decorators return the plain function, so nothing is registered or run on Modal."""

    def __init__(self, *a, **k):
        pass

    def function(self, *a, **k):
        return lambda f: f

    def local_entrypoint(self, *a, **k):
        return lambda f: f


def load_pinned(name: str, src: bytes) -> types.ModuleType:
    """Import a pinned script from its bytes as a plain module, with a stand-in `modal` only while it imports."""
    fake = types.ModuleType("modal")
    fake.App, fake.Image, fake.Volume = _App, _Lazy(), _Lazy()
    real = sys.modules.get("modal")
    sys.modules["modal"] = fake
    try:
        mod = types.ModuleType(f"pinned_{name}")
        mod.__file__ = f"/root/pinned_{name}.py"
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    finally:
        if real is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = real
    return mod


def decode(blobs: dict) -> dict:
    """sha256-check every input (raw bytes, and the decompressed bytes for GZ names) and parse the JSON ones."""
    out = {}
    for name, (_c, _p, want, want_gz) in PINS.items():
        b = blobs[name]
        if sha(b) != want:
            raise Stop(f"STOP: input {name} differs from its pin")
        if name in GZ:
            b = gzip.decompress(b)
            if sha(b) != want_gz:
                raise Stop(f"STOP: input {name} (decompressed) differs from its pin")
        out[name] = b if name.endswith("_script") else json.loads(b)
    return out


# ---------------------------------------------------------------- helpers
def rnd(x, places: int):
    """Half-up rounding of a float at the post's printed precision, as a string (None stays None)."""
    return None if x is None else str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def pct(x):
    return None if x is None else 100 * x


def sim_runs(rec: dict) -> dict:
    """{cell: run key} over the record's sim run keys; STOP on a cell with two sim runs."""
    out = {}
    for k in rec["run_keys"]:
        if k.startswith("sim|"):
            c = k.split("|")[1]
            if c in out:
                raise Stop(f"STOP: two sim runs on cell {c}")
            out[c] = k
    return out


def shares(sol: int, unk: int, n: int) -> dict:
    return {"all_runs_denominator": sol / n if n else None, "unknown_counted_sol_upper": (sol + unk) / n if n else None,
            "unknown_excluded": sol / (n - unk) if n > unk else None}


def sol_count(keys_paths: dict, flags: dict, norm) -> dict:
    """keys_paths: {run key: eval path or None}. A run with no path or no Sol verdict is UNKNOWN (kept in n)."""
    fl = {k: (flags.get(norm(p)) if p is not None else None) for k, p in keys_paths.items()}
    sol = sum(v is True for v in fl.values())
    free = sum(v is False for v in fl.values())
    unk = sorted(k for k, v in fl.items() if v is None)
    return {"n": len(fl), "sol_using": sol, "sol_free": free, "unknown": len(unk), "unknown_keys": unk, "shares": shares(sol, len(unk), len(fl))}


def flag(old, new, places: int) -> str:
    if old is None or new is None:
        return "not comparable"
    return "unchanged" if rnd(old, places) == rnd(new, places) else "changed"


# ---------------------------------------------------------------- the reduction (pure; the container calls it, the tests call it)
def build(x: dict, sf, rt, pilot_rows: list) -> dict:
    import numpy as np

    gates = {}
    cat, A, h171, hrc = x["catalogue"], x["record_A"], x["honest_171"], x["honest_routec"]
    # ---- G_TIE: the catalogue is the one A calibrated on
    if A["meta"]["pooled_world"]["catalogue"]["sha256"] != PINS["catalogue"][3]:
        raise Stop("STOP: G_TIE: record A's meta.pooled_world.catalogue.sha256 is not the pinned catalogue")
    gates["G_TIE"] = "PASS"
    S = {k: r for k, r in cat["runs"].items() if "S" in r["roles"]}
    s_cell = {}
    for k in S:
        c = k.split("|", 1)[1]
        if c in s_cell:
            raise Stop(f"STOP: two role-S catalogue runs on cell {c}")
        s_cell[c] = k
    a_sizes = {lab: v["sizes"]["S"]["after"]["runs"] for lab, v in sorted(A["meta"]["pooled_world"]["per_label"].items())
               if lab.endswith("| pooled_all_origins")}
    # ---- G_PILOT_SETS (item 2): every simh_pilot catalogue run is a role-S run with sets exactly [S_old]; any exception STOPs
    try:
        pilot = {k: r for k, r in cat["runs"].items() if r["world"] == "simh_pilot"}
        bad = sorted(k for k, r in pilot.items() if not ("S" in r["roles"] and r["sets"] == ["S_old"]))
    except Exception as e:  # noqa: BLE001  (fail-closed by design: any malformed catalogue row STOPs)
        raise Stop(f"STOP: G_PILOT_SETS raised {type(e).__name__}") from None
    if len(pilot) != N_PILOT or bad:
        raise Stop(f"STOP: G_PILOT_SETS: {len(pilot)} simh_pilot runs (want {N_PILOT}); not role S with sets == ['S_old']: {bad}")
    gates["G_PILOT_SETS"] = "PASS"
    pilot_cells = {k.split("|", 1)[1] for k in pilot}
    # ---- G_DELTA: route C's detector sim class = the 10-16 class + exactly the pilot cells, shared runs unchanged by key and path
    sim171, simrc = sim_runs(h171), sim_runs(hrc)
    shared = set(sim171) & set(simrc)
    if (set(sim171) - set(simrc)) or (set(simrc) - set(sim171)) != pilot_cells or any(
            sim171[c] != simrc[c] or h171["run_keys"][sim171[c]] != hrc["run_keys"][simrc[c]] for c in shared):
        raise Stop("STOP: G_DELTA: route C's sim class is not the 10-16 class plus exactly the pilot cells")
    gates["G_DELTA"] = "PASS"
    # ---- G_PILOT_PATHS: the pilot's detector eval paths = its upload manifest's paths = the header rows' paths
    man = sorted(f["path"] for f in x["pilot_manifest"]["files"])
    det = sorted(sf.norm(hrc["run_keys"][simrc[c]]) for c in pilot_cells)
    if len(man) != N_PILOT or man != det or sorted(r["path"] for r in pilot_rows) != man:
        raise Stop("STOP: G_PILOT_PATHS: pilot manifest, detector run paths and header rows differ")
    gates["G_PILOT_PATHS"] = "PASS"
    # ---- item 1 gates: item (c)'s own (self-test, header-record pins, positive control), then Sol flags
    st = sf.self_test()
    hdr = x["cont_headers"]
    if any(hdr.get(k) != v for k, v in sf.HEADER_PINS.items()) or len(hdr.get("rows", [])) != sf.HEADER_PINS["n_files"]:
        raise Stop("STOP: the 149-file header record is not item (c)'s pinned one")
    # G_POSITIVE_CONTROL: each cell's run-level AUROC recomputed with item (c)'s wauc equals the record's, both records: simh_vs_b (class 1 =
    # "sim", item (c)'s cell_arrays) and, review O2, a_vs_b (class 1 = "A", class 0 = "B": run_honest_all.py @ 596bc767e L35, L53)
    pc = {}
    for tag, rec in (("v83", h171), ("routec", hrc)):
        for cell, c in sorted(rec["cells"].items()):
            for prob, cfg in (("simh_vs_b", sf.DETECTORS["honest"]), ("a_vs_b", {"problem": "a_vs_b", "pos": "A", "neg": "B"})):
                keys, y, s = sf.cell_arrays("honest", cfg, rec, cell, c)
                a = sf.wauc(s, y, np.ones(len(keys)))
                ref = c["problems"][prob]["run_level"]["auroc"]
                pc[f"{tag} | {cell} | {prob}"] = bool(a is not None and isinstance(ref, (int, float)) and abs(a - ref) <= sf.TOL)
    if not all(pc.values()):
        raise Stop(f"STOP: positive control failed in {sorted(k for k, v in pc.items() if not v)}")
    gates["G_POSITIVE_CONTROL"] = f"PASS {sum(pc.values())} of {len(pc)}"
    flags_v83, src_v83 = sf.sol_flags(x["scan_rows"], x["usage"], hdr)  # item (c)'s three sources, as on 10-17
    flags_rc, src_rc = sf.sol_flags(x["scan_rows"], x["usage"], {"rows": hdr["rows"] + pilot_rows})  # + the pilot's header rows
    pilot_unread = sorted(r["path"] for r in pilot_rows if "read_error" in r or not r.get("has_header"))
    n_read = len(pilot_rows) - len(pilot_unread)  # review O1: the laptop sees the header coverage, never an all-PASS over unread files
    gates["G_PILOT_HEADERS"] = f"read {n_read} of {N_PILOT}" + ("" if n_read == N_PILOT else " (INCOMPLETE: the unread files are UNKNOWN, kept in n)")

    def cells_count(rec, flags):
        out = {}
        for cell, c in sorted(rec["cells"].items()):
            ks = [k for k in c["problems"]["simh_vs_b"]["run_scores"] if k.startswith("sim|")]
            out[cell] = sol_count({k: rec["run_keys"][k] for k in ks}, flags, sf.norm)
        return out

    c83, crc = cells_count(h171, flags_v83), cells_count(hrc, flags_rc)
    for tag, cc in (("v83", c83), ("routec", crc)):
        if not any(v["sol_using"] for v in cc.values()):
            raise Stop(f"STOP: no Sol-using simulated honest run in {tag} (model-name positive control)")
    gates["G_SOL_MODEL_NAME"] = "PASS"
    pool83 = sol_count({k: h171["run_keys"].get(sim171.get(k.split("|", 1)[1])) for k in S if k.split("|", 1)[1] not in pilot_cells},
                       flags_v83, sf.norm)
    poolrc = sol_count({k: hrc["run_keys"].get(simrc.get(k.split("|", 1)[1])) for k in S}, flags_rc, sf.norm)
    basis = "all_runs_denominator"
    v83_cells_agree = len({(v["n"], v["sol_using"], v["unknown"]) for v in c83.values()}) == 1
    item1 = {
        "v83_source": V83_SOURCES["item1"],
        "footnote_uses": f"route C: pool.routec.shares.{basis} (UNKNOWN counted as not Sol, in the denominator; the 10-17 basis, where UNKNOWN "
                         f"was 0). If pool.routec.unknown > 0, the footnote's honest range is {basis} to unknown_counted_sol_upper.",
        "printed_decimals_percent": PRINTED["item1"],
        "pilot_header_scan": {"n_files": len(pilot_rows), "n_header_read": len(pilot_rows) - len(pilot_unread), "n_unknown": len(pilot_unread),
                              "unknown_paths": pilot_unread, "rows": sorted(pilot_rows, key=lambda r: r["path"])},
        "n_paths_by_number_of_sources": {"v83": {str(k): v for k, v in sorted(src_v83.items())},
                                         "routec": {str(k): v for k, v in sorted(src_rc.items())}},
        "detector_cells": {"v83": c83, "routec": crc, "v83_cells_agree": v83_cells_agree,
                           "routec_cells_agree": len({(v["n"], v["sol_using"], v["unknown"]) for v in crc.values()}) == 1},
        "pool": {"v83": pool83 | {"what": "role-S catalogue runs minus the pilot, paths from honest_all @ 8b0b18e2a (whether this equals the 10-16 "
                                          "detector's sim class, cell by cell, is items.4_honest_detector_cohort.v83.<cell>.equal and gate "
                                          "G_POOL_V83_EQUALS_DETECTOR_171)"},
                 "routec": poolrc | {"what": "role-S catalogue runs (route C pool), paths from honest_all_routec @ 596bc767e"}},
        "percent": {"v83_pool": pct(pool83["shares"][basis]), "routec_pool": pct(poolrc["shares"][basis]),
                    "routec_pool_upper": pct(poolrc["shares"]["unknown_counted_sol_upper"]),
                    "routec_pool_unknown_excluded": pct(poolrc["shares"]["unknown_excluded"])},
    }
    item1["routec_pool_bounds_print_the_same"] = rnd(item1["percent"]["routec_pool"], PRINTED["item1"]) == rnd(
        item1["percent"]["routec_pool_upper"], PRINTED["item1"])
    item1["flags"] = {
        "pool_v83_vs_routec": flag(item1["percent"]["v83_pool"], item1["percent"]["routec_pool"], PRINTED["item1"]),
        **{f"cell {c} v83_vs_routec": flag(pct(c83[c]["shares"][basis]), pct(crc[c]["shares"][basis]), PRINTED["item1"]) for c in sorted(crc)},
    }
    item1["v83_reproduces_printed"] = {  # review O3: booleans only
        "pool percent matches the printed v83": rnd(item1["percent"]["v83_pool"], PRINTED["item1"]) == PRINTED_V83["item1_percent"],
        **{f"cell {c} percent matches the printed v83": rnd(pct(c83[c]["shares"][basis]), PRINTED["item1"]) == PRINTED_V83["item1_percent"] for c in sorted(c83)},
        **{f"cell {c} counts equal the 10-17 relay (sol, n)": (c83[c]["sol_using"], c83[c]["n"]) == PRINTED_V83["item1_counts_sol_n"]
           for c in sorted(c83)},
        "pool counts equal the 10-17 relay (sol, n)": (pool83["sol_using"], pool83["n"]) == PRINTED_V83["item1_counts_sol_n"]}
    # ---- item 2: the failed sim-honest share
    rates = x["rates_171"]
    hl = rates["headline_common_sets"]
    if list(hl["sets"]) != list(rt.COMMON):
        raise Stop("STOP: item 2: the rates record's headline sets are not make_appendix_a_rates.COMMON")
    y = hl["Y_sim"]
    if y["launched"] <= 0 or y["rate"] != y["failed"] / y["launched"]:
        raise Stop("STOP: item 2: Y_sim.rate != failed / launched in the rates record")
    rl = y["by_category"][RATE_LIMIT]
    in_common = sorted(k for k, r in pilot.items() if set(r["sets"]) & set(rt.COMMON))
    if in_common:
        raise Stop(f"STOP: item 2: pilot runs in the headline sets {in_common}; this tool does not recount their attempts")
    chain_hits = x["chain_pilot_files"]
    rates_manifests = sorted(rates["sim"]["launch_source"]["manifests_rows"])
    pilot_in_rates = sorted(n for n in rates_manifests if "pilot" in n)
    if chain_hits or pilot_in_rates:
        raise Stop("STOP: item 2: a pilot launch row is among the rates script's sim launch manifests")
    gates["G_RATES_SCOPE"] = "PASS"
    d_failed, d_launched = 0, len(in_common)  # the pilot runs in COMMON sets (none, by the two checks above)
    v83 = {"failed": y["failed"], "launched": y["launched"], "rate_limit": rl}
    rc = {"failed": y["failed"] + d_failed, "launched": y["launched"] + d_launched, "rate_limit": rl}
    pc2 = lambda d: {"percent": pct(d["failed"] / d["launched"]), "percent_without_rate_limit": pct((d["failed"] - d["rate_limit"]) / d["launched"])}  # noqa: E731
    pm = x["pilot_manifest"]
    cawe = pm["counted_attempts_without_eval"]  # review R2: an object of lists ({"simh": [...]} in the manifest @ 4af62db74), counted per key
    if not isinstance(cawe, dict) or not all(isinstance(v, list) for v in cawe.values()):
        raise Stop("STOP: item 2: the pilot manifest's counted_attempts_without_eval is not an object of lists")
    item2 = {
        "v83_source": V83_SOURCES["item2"], "printed_decimals_percent": PRINTED["item2"],
        "headline_sets": list(rt.COMMON), "rule": "make_appendix_a_rates.py @ 35dff7132 L43 (COMMON), L247-249 (Y_sim over COMMON), L254-256 "
                                                  "(other sets apart)",
        "pilot_runs_in_headline_sets": in_common, "chain_manifest_files_naming_the_pilot": chain_hits,
        "rates_sim_manifests_naming_pilot": pilot_in_rates,
        "v83": v83 | pc2(v83), "routec": rc | pc2(rc),
        "pilot_attempts_S_old_apart": {"manifest_counts": pm.get("counts"), "files": len(pm["files"]),
                                       "counted_attempts_without_eval": {k: len(v) for k, v in sorted(cawe.items())},
                                       "note": "S_old is outside the headline sets; listed for App I only"},
    }
    item2["flags"] = {"with_rate_limit": flag(item2["v83"]["percent"], item2["routec"]["percent"], PRINTED["item2"]),
                      "without_rate_limit": flag(item2["v83"]["percent_without_rate_limit"], item2["routec"]["percent_without_rate_limit"],
                                                 PRINTED["item2"])}
    item2["v83_reproduces_printed"] = {  # review O3
        "percent matches the printed v83": rnd(item2["v83"]["percent"], PRINTED["item2"]) == PRINTED_V83["item2_percent"],
        "percent_without_rate_limit matches the printed v83": rnd(item2["v83"]["percent_without_rate_limit"], PRINTED["item2"])
        == PRINTED_V83["item2_percent_without_rate_limit"]}
    # ---- item 4: the honest detector's sim class per cell against the pool, by cell, naming every difference
    def cohort4(rec, pool_cells: dict):
        out = {}
        zr = {cell: {r["name"] for r in v["runs"]} for cell, v in rec["zero_row_runs"].items()}  # run keys with no rows, per cell
        for cell, c in sorted(rec["cells"].items()):
            det = {k.split("|")[1]: k for k in c["problems"]["simh_vs_b"]["run_scores"] if k.startswith("sim|")}
            extra = sorted(det[x_] for x_ in set(det) - set(pool_cells))
            out[cell] = {"n_detector_sim": len(det), "n_pool": len(pool_cells), "n_both": len(set(det) & set(pool_cells)),
                         "detector_only_keys": extra, "pool_only_keys": sorted(pool_cells[x_] for x_ in set(pool_cells) - set(det)),
                         "detector_only_in_zero_row_runs": sorted(k for k in extra if k in zr.get(cell, set())),
                         "equal": set(det) == set(pool_cells)}
        allsim = sim_runs(rec)
        out["run_keys (all sim runs of the record)"] = {
            "n": len(allsim), "detector_only_keys": sorted(allsim[c] for c in set(allsim) - set(pool_cells)),
            "pool_only_keys": sorted(pool_cells[c] for c in set(pool_cells) - set(allsim)),
            "zero_row_cells_of_detector_only": {k: sorted(cell for cell, v in zr.items() if k in v)
                                                for k in sorted(allsim[c] for c in set(allsim) - set(pool_cells))}}
        return out

    s83 = {c: k for c, k in s_cell.items() if c not in pilot_cells}
    item4 = {"v83_source": V83_SOURCES["item4"], "v83": cohort4(h171, s83), "routec": cohort4(hrc, s_cell),
             "pool_n": {"v83": len(s83), "routec": len(S)}, "record_A_sizes_S_after_runs": a_sizes,
             "record_A_sizes_equal_pool": bool(a_sizes) and all(v == len(S) for v in a_sizes.values())}
    item4["flags"] = {"pool_n": "unchanged" if len(s83) == len(S) else "changed",
                      **{f"cell {c} n_detector_sim": "unchanged" if item4["v83"][c]["n_detector_sim"] == item4["routec"][c]["n_detector_sim"]
                         else "changed" for c in sorted(hrc["cells"])}}
    bad4 = sorted(c for c in h171["cells"] if not item4["v83"][c]["equal"])  # review O5: stated as a gate result, not assumed
    gates["G_POOL_V83_EQUALS_DETECTOR_171"] = "PASS" if not bad4 else f"FAIL in cells {bad4} (see items.4_honest_detector_cohort.v83)"
    # ---- item 5: the a_vs_b control, fail-closed cohort identity, and its run-level AUROC at 2 decimals
    try:
        def coh(rec, cell):
            rs = rec["cells"][cell]["problems"]["a_vs_b"]["run_scores"]
            if not rs or any(k.split("|", 1)[0] not in ("A", "B") for k in rs):
                raise ValueError("a_vs_b run keys")
            return {k: (rec["run_keys"][k], rec["run_folds"][k], rec["run_main"][k]) for k in rs}

        cells5 = sorted(set(h171["cells"]) | set(hrc["cells"]))
        same = {cell: coh(h171, cell) == coh(hrc, cell) for cell in cells5}
        d_score = {cell: max(abs(float(h171["cells"][cell]["problems"]["a_vs_b"]["run_scores"][k])
                                 - float(hrc["cells"][cell]["problems"]["a_vs_b"]["run_scores"][k]))
                             for k in hrc["cells"][cell]["problems"]["a_vs_b"]["run_scores"]) if same[cell] else None for cell in cells5}
        au = {tag: {cell: float(rec["cells"][cell]["problems"]["a_vs_b"]["run_level"]["auroc"]) for cell in cells5}
              for tag, rec in (("v83", h171), ("routec", hrc))}
    except Exception as e:  # noqa: BLE001  (fail-closed: any missing key or malformed row STOPs)
        raise Stop(f"STOP: item 5 cohort check raised {type(e).__name__}") from None
    item5 = {"v83_source": V83_SOURCES["item5"], "printed_decimals": PRINTED["item5"], "post_cells": list(ROW5_CELLS),
             "cohort_identical_by_key_path_fold_main": same, "max_abs_run_score_difference": d_score,
             "run_level_auroc": au, "rounded": {tag: {c: rnd(v, PRINTED["item5"]) for c, v in d.items()} for tag, d in au.items()}}
    item5["flags"] = {c: ("unchanged" if same[c] and rnd(au["v83"][c], 2) == rnd(au["routec"][c], 2) else
                          "changed" if same[c] else "NOT CONFIRMED (cohort differs)") for c in ROW5_CELLS}
    item5["v83_reproduces_printed"] = {f"cell {c} matches the printed v83": rnd(au["v83"][c], PRINTED["item5"]) == PRINTED_V83["item5_auroc"]
                                       for c in ROW5_CELLS}  # review O3
    v83p = {f"{k} | {b}": v for k, it in (("1", item1), ("2", item2), ("5", item5)) for b, v in it["v83_reproduces_printed"].items()}
    nbad = sorted(k for k, v in v83p.items() if not v)
    gates["G_V83_PRINTED"] = (f"PASS {len(v83p)} of {len(v83p)}" if not nbad else
                              f"FAIL {len(v83p) - len(nbad)} of {len(v83p)}; not reproduced: {nbad}")  # recorded, not a STOP: the reader decides
    flags = [v for it in (item1, item2, item4, item5) for v in it["flags"].values()]
    return {"what": __doc__.split("\nUsage")[0], "gates": gates, "self_test": st,
            "items": {"1_sim_footnote": item1, "2_failed_footnote": item2, "4_honest_detector_cohort": item4, "5_table7_row5_control": item5},
            "flag_counts": {k: flags.count(k) for k in sorted(set(flags))}}


def summary(rec: dict) -> dict:
    """What the laptop prints: gate results, flag counts and the item keys (no value)."""
    return {"gates": rec["gates"], "flag_counts": rec["flag_counts"], "items": sorted(rec["items"]),
            "flag_keys": {k: sorted(v["flags"]) for k, v in rec["items"].items()}}


def check_header_module(hd) -> None:
    if hd.VOL != VOL or hd.ROOT != HDR_ROOT:
        raise Stop("STOP: the header script's Volume or root is not this tool's mount")


@app.function(volumes={"/a25": vol, "/out": out_vol}, cpu=1, memory=8192, timeout=1200, retries=0)
def compute(blobs: dict, prov: dict) -> dict:
    if not OUT_NAME.fullmatch(prov["out_name"]):
        raise Stop("STOP: the record name is not routec_simpool_numbers_<stamp>.json")
    x = decode(blobs)
    sf, hd, rt = (load_pinned(n, x[n]) for n in ("solfree_script", "headers_script", "rates_script"))
    check_header_module(hd)
    x["chain_pilot_files"] = prov["chain_pilot_files"]
    files = [{"path": f["path"], "sha256": f["sha256"]} for f in x["pilot_manifest"]["files"]]
    pilot_rows = hd.headers(files)  # header_only reads, sha256-checked per file against the manifest (check_simh_cont_headers L43-71)
    rec = build(x, sf, rt, pilot_rows) | {"provenance": prov}
    b = json.dumps(rec, indent=1, sort_keys=True).encode()
    with open(f"/out/{prov['out_name']}", "xb") as fh:  # review R1: the record is durable on this job's Volume before it is returned
        fh.write(b)
    out_vol.commit()
    return {"record_bytes": b, "summary": summary(rec), "volume_path": f"{OUT_VOL}:/{prov['out_name']}"}


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, check=False, cwd=REPO)


@app.local_entrypoint()
def main(out: str):
    if git("status", "--porcelain", "--", str(Path(__file__).resolve())).stdout:
        raise Stop("STOP: the script is not committed")
    o = Path(out).resolve()
    if o.exists() or o.parent != HERE / "results" or not o.parent.is_dir() or not OUT_NAME.fullmatch(o.name):  # review O6: the folder exists
        raise Stop("STOP: --out must be a new results/routec_simpool_numbers_<stamp>.json in this folder")
    head = git("rev-parse", "HEAD").stdout.decode().strip()  # read once, before any spawn
    blobs = {}
    for name, (c, p, want, _) in PINS.items():
        r = git("show", f"{c}:{p}")
        if r.returncode or sha(r.stdout) != want:
            raise Stop(f"STOP: {name} @ {c} missing or differs from its pin")
        blobs[name] = r.stdout
    g = git("grep", "-l", PILOT_LAUNCH, CHAIN[0], "--", CHAIN[1])
    if g.returncode not in (0, 1):
        raise Stop("STOP: git grep over the chain manifests failed")
    prov = {"git": head, "script_sha256": sha(Path(__file__).resolve().read_bytes()), "app": APP, "volume": VOL, "out_volume": OUT_VOL,
            "out_name": o.name,
            "inputs": {n: {"commit": c, "path": p, "sha256": s, "sha256_decompressed": z} for n, (c, p, s, z) in PINS.items()},
            "chain_pilot_files": sorted(g.stdout.decode().split()), "chain": list(CHAIN)}
    reply = compute.remote(blobs, prov)
    b = reply["record_bytes"]
    with open(o, "xb") as fh:  # exclusive create; the laptop writes the bytes and reads no field of them
        fh.write(b)
    s = reply["summary"]
    print(f"gates {json.dumps(s['gates'], sort_keys=True)}")
    print(f"flag counts {json.dumps(s['flag_counts'], sort_keys=True)}; flag keys {json.dumps(s['flag_keys'], sort_keys=True)}")
    print(f"record {o.relative_to(REPO)} sha256 {sha(b)} (also on Volume {reply['volume_path']})")
