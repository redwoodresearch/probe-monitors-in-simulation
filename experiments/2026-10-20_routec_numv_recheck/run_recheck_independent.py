"""Independent recheck of the rows the route C numbers verifier marked "equal": its seeded samples (seed 20261020,
k 20) from two passes, batch 015950 and batch 020604.

Own implementation. It does not import, read or copy the verifier's verify_core.py / run_numbers_verify.py, and it
does not use the slot producer's slot_table.py. NOTE_recheck_independent_2026-10-20.md (next to this file) states the
rules, controls and spend.

Laptop (local entrypoint): reads git blobs by commit:path, checks sha256 of the plain bytes, passes them in (gzipped in
transit), prints one RECHECK_RESULT line of booleans, row ids, rule names, counts and error classes. It parses no
value file. Container (one CPU function): sha-checks every input again, extracts each printed value from its
artifact and each source value from its record, compares under the row's rule, runs the controls, writes the full
per-row detail (values included) to the Volume, and returns no values.
"""

import fnmatch
import gzip
import hashlib
import json
import math
import re
import subprocess
from decimal import Decimal
from pathlib import Path

import modal

APP_NAME = "probe-shift_aswin-surya_sub-routec-numv_recheck_kill-2026-11-19"
VOL_NAME = "probe-shift_aswin-surya_sub-routec-numv_rows_kill-2026-11-19"
BRANCH = "refs/heads/aswin/routec-numv-recheck"
SEED, K = 20261020, 20
RES = "experiments/2026-10-20_routec_numbers_verify/results"
PREP = "experiments/2026-10-15_fpr_sweep_507_prep/results"
SLOTS = "experiments/2026-10-17_routec_slot_readout/results"
SLOTNOTE = "experiments/2026-10-17_routec_slot_readout/NOTE_slot_readout.md"

# name -> (commit, path, sha256 of the plain bytes; for .gz paths, of the decompressed bytes, as in the verifier config)
PINS = {
    "result_015950": ("4b117b54b6c1d64a3e51ac6e94d17c282423b7e2", f"{RES}/numbers_verify_2026-10-20_015950.json",
                      "6adb8c3aee33caea60bbea1864b1bf0b96ea7124b477e88a8bfd0849bace6932"),
    "result_020604": ("ce8e0508a1c307b5d8e63746bddc9a926dc6a10d", f"{RES}/numbers_verify_2026-10-20_020604.json",
                      "3345ce4878a3d0c2b475ecae6c24a1a0be05e32756d32a8a655e0f0694d1d73a"),
    # artifacts
    "rc2": ("d6ff04c9c0951c417f6536b7ccc5e6fe3afcbb1d", f"{SLOTS}/slots_routeC_writing_rc2.json",
            "de6636afa74098cd69ab7bf630f031ef53ed0dd87a0c3fb24f28281f6e83e66c"),
    "rc1b": ("600f63614c0ab77f74fa7e326461e2e73086c2ac", f"{SLOTS}/slots_routeC_writing_rc1b.json",
             "ab8dcb439b942c92d30b784ae1e6a3a193b18127391ff1753426663b726e728b"),
    "fig4_src": ("6d56caeb83f641b009f363313fcbfa092e98203a", f"{PREP}/fig4_pooled_fig4_routecB_source.json",
                 "0de398ad8a6e6115a67e473c42a772f4def525dae11a481e23654e4d504e9e5d"),
    "fig1_src": ("7488001c2e9eb3bffd779e0ed325bd5ffc8a209b", f"{PREP}/fig1_main_figs156_routecA_source.json",
                 "4f31f71b9d8b92fe25845d715c91d1dcfbd9910fbd9c11f8ab185b3cb97c74e1"),
    "fig5_src": ("7488001c2e9eb3bffd779e0ed325bd5ffc8a209b", f"{PREP}/fig5_excess_C_figs156_routecA_source.json",
                 "1e9c1b1cd74cadb83ac6713a21b136793cd8aec792f2cbfb1f2aaf4ed99877c1"),
    "figs78_src": ("3efcd4a05e086ce32f8af67a9c4fda0db78abc42", f"{PREP}/fig_pertask_onpolicy_figs78_routecA_source.json",
                   "62cb3739c9058a6c183df69526fabd0bc1a5e7fc925a8560864666c6b3050065"),
    # source records
    "B": ("f4e21b0f76acde02d1ae4e7408028d6eaeb559d8", f"{PREP}/routec_records/"
          "routec_B_alltasks_CRD_FULL_r2_allorigins_sweep15_pooledworld_scored_e12_2026-10-19_180652.json.gz",
          "ef5c765716ee338a1e8d207a820eecd5c083ba1d51262ff72361d6d254bf6ad7"),
    "R37": ("55a2fb20e1206b37ffe85656901bcf4aa70edfde", f"{PREP}/routec_records/"
            "routec_R37_alltasks_CRD_FULL_r2_allorigins_rule37_pooledworld_scored_e12_2026-10-19_181100.json.gz",
            "8e6635e7413eb0202e469e3af84af70e72b801de9fc0c9d6b7505848a21807e4"),
    "A": ("a936a5d9762763f9feed351aebdeeec2d8114f21", f"{PREP}/routec_records/"
          "routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz",
          "54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a"),
    "A_group_excess": ("5eb9888d4dd6d7b9ddda5f8822471050bba11217",
                       f"{PREP}/group_excess_2026-10-20_003418_a6cada8b0fa546d890b6976f675f7246.json",
                       "f4fe20de538d63c1d2e769755220b3d76574e8b80213c9d2b0439d2b70fe8169"),
    "KAPPA": ("947ec6d4535377f9b1b204a53519a2f4af0a2646", f"{PREP}/routec_records/kappa_CRD_2026-10-20_004454.json.gz",
              "40ca1bf7a6deaf51175fb56cd89d7277b64ca1ec1ffdf87292d6ae0feca3cce9"),
    "KAPPA_EXTRA": ("93ffca1ffcf76aef105940658af080939ffa1bf6",
                    f"{PREP}/kappa_extra_kappa_CRD_2026-10-20_004454_2026-10-20_011442_ap-lSODwCn5gn3jLeCaCx38fX.json",
                    "85b995c9d640eab80b84c23bfac1b3945d49336047e7b48721507dc54e1e5a3f"),
    "ANCHOR": ("2a41e5f3bf133aaf54329e1c72dde2d63757da37",
               f"{PREP}/pertask_full_2026-10-20_004115_anchor_2026-10-20_011851_ap-Ul4ZPPM7A0z6LFMNMfGTqh.json",
               "2763ad99b7d37ee25784d9b58563a60d5d6c4b425656e45a97f71b82684cfbad"),
    "PLANT": ("699f214035a4f15b9e3025307eb21ee6705c3b7b", "experiments/2026-10-15_panel_noise_matched/results/"
              "panel_plant_rows_plant_rows_POOLED_FULL_routeC_2026-10-20_004247_2026-10-20_004833.json",
              "a864fad74436f0b5052adfaae7ba4ff54a0f192670b74b9b88ec8235b7048a21"),
    "PERTASK": ("c22edd8d3cc1221225dec418ab00b40515d5ba43", f"{PREP}/routec_records/pertask_full_2026-10-20_004115.json",
                "468655154f826ec1a93e08333eb802e68abea6db65a8e2f80a3e0f23465f36d9"),
    # the slot readout's registered slot-key table (definitions; git blobs 90af10473 and 194698115)
    "slotnote_rc2": ("d6ff04c9c0951c417f6536b7ccc5e6fe3afcbb1d", SLOTNOTE,
                     "f14ff20a39d92f94ed5ad05fe4b2cca92aeae23448e543446d919b58212aeeaa"),
    "slotnote_rc1b": ("600f63614c0ab77f74fa7e326461e2e73086c2ac", SLOTNOTE,
                      "ab6c740d7bd281d0f72ed6985cca53951a04f710d7305338aa195585c8acbdc4"),
}
# batch -> its result file pin and its Volume files (read in the container only; rows.jsonl is sha-checked, never parsed)
BATCHES = {
    "2026-10-20_015950": {"result": "result_015950",
                          "selection": "d28ae12d08f11bea24cde19d1df1d18f157dc8fa0249b30d92e8f7257ee39035",
                          "rows": "e6701fcabd0dec39d154f2583ebd33108f23ae894d092903c0d4c086f768a0e1"},
    "2026-10-20_020604": {"result": "result_020604",
                          "selection": "ef6b66e4930328cd2ce49901feca873777b9053810ec6a77da05d47841aaaf4e",
                          "rows": "b1ca7c12b9029ab141cd37a10d8166fa03949692b4dfc037f09ea5d1ba94d062"},
}
# Summary slots ("x (lo to hi)"): (artifact, id regex, record, path template). The template is filled from the id's
# parsed fields. Every family is DEF: its template comes from the slot readout's registered slot-key table,
# NOTE_slot_readout.md (rc2: blob 90af10473 at d6ff04c9c; rc1b: blob 194698115 at 600f63614; pinned below as
# slotnote_rc2 / slotnote_rc1b and checked row by row in the container, see DEFINITIONS), relayed by 2e from 09 on
# 10-20. No template has a "*" any more; candidates() still supports one. Every candidate is a {point, ci95} object.
_POOLED = ["pooled", "{pp}", "pooled_all_origins", "{mode}", "{fpr}"]
SUMM_FAMILIES = [
    # DEF rc2 L45 (row 18): `18.<delta_full|Y>.<joint|fixed>.<label>@<lvl>` -> the pooled B (sweep15) record
    # `pooled["<label>"].pooled_all_origins.<mode>["<lvl>"].gap.{delta_full, run2_minus_run1_at_tauB}`, so
    # Y = gap.run2_minus_run1_at_tauB (row 11, L37). B only.
    ("rc2", re.compile(r"^(?P<slot>18)\.(?P<metric>Y)\.(?P<mode>fixed|joint)\.(?P<pp>.+ [CR])@(?P<fpr>\d+\.\d+)$"),
     {"18": "B"}, _POOLED + ["gap", "run2_minus_run1_at_tauB"]),
    # DEF rc2 L45 (row 18, B) and L47 (row 20): `20.{delta_full, catch.<sim|real_mean>, readable}.<mode>.<label>@<lvl>`
    # -> "R37: the same block at 1e-4, 2e-4, 5e-4".
    ("rc2", re.compile(r"^(?P<slot>18|20)\.delta_full\.(?P<mode>fixed|joint)\.(?P<pp>.+ [CR])@(?P<fpr>\d+\.\d+)$"),
     {"18": "B", "20": "R37"}, _POOLED + ["gap", "delta_full"]),
    # DEF rc2 L47 (row 20): catch is defined for slot 20 only (row 18 defines no catch).
    ("rc2", re.compile(r"^(?P<slot>20)\.catch\.(?P<mode>fixed|joint)\.(?P<pp>.+ [CR])@(?P<fpr>\d+\.\d+)"
                       r"\.(?P<sub>real_mean|sim)$"),
     {"20": "R37"}, _POOLED + ["catch", "{sub}"]),
    # DEF rc1b L166 (row 21, the kappa row, not kappa_group at L167):
    # `21.kappa.<label>.<joint|fixed>.<kappa_sim_vs_run1|kappa_run2_vs_run1|kappa_sim_minus_run2>` ->
    # `kappa["<label>"].combos.pooled_all_origins.<mode>.<key>` (never the per-S-definition `origins`).
    ("rc1b", re.compile(r"^21\.kappa\.(?P<pp>.+ [CR])\.(?P<mode>fixed|joint)"
                        r"\.(?P<q>kappa_sim_vs_run1|kappa_run2_vs_run1|kappa_sim_minus_run2)$"),
     "KAPPA", ["kappa", "{pp}", "combos", "pooled_all_origins", "{mode}", "{q}"]),
    # DEF rc1b L168 (row 22): `22.kappa_five.<ro>.<mode>.<mean_kappa_…>` ->
    # `five_probe_sensitivity.groups["five_probe_sensitivity|<ro>|pooled_all_origins"].<mode>.<key>`.
    ("rc1b", re.compile(r"^22\.kappa_five\.(?P<pool>[CR])\.(?P<mode>fixed|joint)\.(?P<q>mean_kappa_[a-z0-9_]+)$"),
     "KAPPA_EXTRA", ["five_probe_sensitivity", "groups", "five_probe_sensitivity|{pool}|pooled_all_origins", "{mode}",
                     "{q}"]),
    # DEF rc1b L170 (row 24): `24.r2.lead.<label>.<main|yardstick|delta_main_minus_yardstick>` ->
    # `union_stream.<block>["<label>"].sets["pooled_all_origins (union stream)"].r2.<key>`, block = lead; the record is
    # the R² anchor (2763ad99 @ 2a41e5f3b), not detector_vs_drop (L171).
    ("rc1b", re.compile(r"^24\.r2\.(?P<block>lead)\.(?P<pp>.+ [CR])\.(?P<q>main|yardstick|delta_main_minus_yardstick)$"),
     "ANCHOR", ["union_stream", "{block}", "{pp}", "sets", "pooled_all_origins (union stream)", "r2", "{q}"]),
    # DEF rc1b L172 (row 25): `25.plant.aggregate.<f>.<ro>.<δ>.<S|excess>` -> `aggregate.by_level.<f>.<ro>.<δ>.<S|excess>`
    # (verdict, `shift`); shift already in points (x100), 1 decimal signed.
    ("rc1b", re.compile(r"^25\.plant\.aggregate\.(?P<level>\d+\.\d+)\.(?P<pool>[A-Z+]+)\.(?P<offset>[+\-]\d+)"
                        r"\.(?P<set>S|excess)$"),
     "PLANT", ["aggregate", "by_level", "{level}", "{pool}", "{offset}", "{set}", "shift"]),
]
# figure artifact -> (source JSON pin, {input name in that JSON: record pin})
FIGS = {
    "fig4:fig4_pooled_fig4_routecB": ("fig4_src", {"rec": "B"}),
    "figs156:fig1_main_figs156_routecA": ("fig1_src", {"record": "A"}),
    "figs156:fig5_excess_C_figs156_routecA": ("fig5_src", {"record": "A", "four": "A_group_excess"}),
    "figs78:fig_pertask_onpolicy_figs78_routecA": ("figs78_src", {"pertask": "PERTASK"}),
}
SCATTER_KEY = 'rows["{probe}"][<origin>]["per_task"][*]["max_minus_tau"]'

SLOT_ID = re.compile(r"^(?P<slot>\d+)\.(?P<rest>.+)#(?P<k>\d+)$")
UNITS19_ID = re.compile(r"^19\.real_units_above_tau_B\.(?P<pp>.+ [CR])@(?P<fpr>\d+\.\d+)$")
PLANT_COUNT_ID = re.compile(r"^25\.plant\.count\.(?P<level>\d+\.\d+)\.(?P<offset>[+\-]\d+)\.(?P<pool>[A-Z+]+)$")
AFTER_RUNS_ID = re.compile(r"^14\.after_runs\.(?P<pp>.+ [CR])\.(?P<set>[BS])$")
FIG_VALUE_ID = re.compile(r"^values\[(?P<i>\d+)\]\|(?:(?P<field>point|value)|ci95\[(?P<j>[01])\])$")
SCATTER_ID = re.compile(r"^src\|scatter_points/(?P<probe>[^/]+)/points/(?P<n>\d+)/(?P<arm>run1|run2|sim)$")
PRINTED_NUM = re.compile("[+\\-−]?\\d+(?:\\.\\d+)?%?")
FULL_NUM = re.compile(r"[+\-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+\-]?\d+)?|\bnan\b|\binf\b")
INT_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)|\d+")  # integers, thousands separators allowed (row 14 format)
PLANT_FIELDS = {0: ("detected", None), 1: ("cases", "detected"), 2: ("not_detected", "missed"),
                3: ("unavailable", "unavailable")}  # token k -> (counts field, printed word after token k)
# Definition guard: (pin, row prefix = the row's first cell plus the start of its key text, substrings the row must
# contain). Rows are selected by that key text, not by line number or slot number alone (rc1b has two "| 21 |" rows);
# each prefix must match exactly one line, or the run stops before any row is checked.
DEFINITIONS = [
    ("slotnote_rc2", "| 18 | `18.<delta_full", [
        "`18.<delta_full\\|Y>.<joint\\|fixed>.<label>@<lvl>`", "the pooled B (sweep15) record",
        '`pooled["<label>"].pooled_all_origins.<mode>["<lvl>"].gap.{delta_full, run2_minus_run1_at_tauB}`']),
    ("slotnote_rc2", "| 11 | `11.run2_minus_run1_at_tauB", ["`….gap.run2_minus_run1_at_tauB`"]),
    ("slotnote_rc2", "| 19 | `19.{units_above_tau_sim", [
        "`19.{units_above_tau_sim, distinct_runs_above_tau_sim, real_units_above_tau_B}.<label C>@0.001`",
        "the pooled B (sweep15) record's rows", '`pools_at_draw0["0.001"].S.units_above`', "`B.units_above`"]),
    ("slotnote_rc2", "| 20 | `20.{delta_full", [
        "`20.{delta_full, catch.<sim\\|real_mean>, readable}.<mode>.<label>@<lvl>`",
        "R37: the same block at 1e-4, 2e-4, 5e-4"]),
    ("slotnote_rc1b", "| 14 | `14.after_runs", [
        "`14.after_runs.<label>.<B\\|S>`", "never the pooled B (sweep15) record",
        '`meta.pooled_world.per_label["<label> \\| pooled_all_origins"].sizes.<pool>.after.runs`']),
    ("slotnote_rc1b", "| 21 | `21.kappa.<label>", [
        "`21.kappa.<label>.<joint\\|fixed>.<kappa_sim_vs_run1\\|kappa_run2_vs_run1\\|kappa_sim_minus_run2>`",
        '`kappa["<label>"].combos.pooled_all_origins.<mode>.<key>`']),
    ("slotnote_rc1b", "| 22 | `22.kappa_five", [
        "`22.kappa_five.<ro>.<mode>.<mean_kappa_…>`",
        '`five_probe_sensitivity.groups["five_probe_sensitivity\\|<ro>\\|pooled_all_origins"].<mode>.<key>`']),
    ("slotnote_rc1b", "| 24 | `24.r2.lead", [
        "`24.r2.lead.<label>.<main\\|yardstick\\|delta_main_minus_yardstick>`",
        '`union_stream.<block>["<label>"].sets["pooled_all_origins (union stream)"].r2.<key>`']),
    ("slotnote_rc1b", "| 25 | `25.plant.verdict", [
        "`25.plant.aggregate.<f>.<ro>.<δ>.<S\\|excess>`", "`aggregate.by_level.<f>.<ro>.<δ>.<S\\|excess>` (verdict, `shift`)",
        "`25.plant.count.<f>.<δ>.<C\\|R\\|C+R>`", "`counts.<f>.<δ>.<ro>` (detected, cases, misses)",
        "from `detected`, `not_detected`, `unavailable`, `cases`"]),
]

image = modal.Image.debian_slim(python_version="3.12")
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME)


class RowError(Exception):
    """A classified per-row failure; its message is the error class (never a value)."""


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def walk(obj, path):
    for p in path:
        if isinstance(obj, list) and isinstance(p, int) and 0 <= p < len(obj):
            obj = obj[p]
        elif isinstance(obj, dict) and p in obj:
            obj = obj[p]
        else:
            raise RowError("key_missing")
    return obj


def triple_index(rec):
    """CROSS-CHECK ONLY (never used to choose a leaf or decide equality): (point, ci95[0], ci95[1]) -> key paths of
    every {point, ci95} object in the whole record with three finite numbers."""
    index, stack = {}, [((), rec)]
    while stack:
        path, o = stack.pop()
        if isinstance(o, dict):
            ci = o.get("ci95")
            if is_num(o.get("point")) and isinstance(ci, list) and len(ci) == 2 and all(map(is_num, ci)):
                index.setdefault((float(o["point"]), float(ci[0]), float(ci[1])), []).append(list(path))
            stack.extend((path + (key,), val) for key, val in o.items())
        elif isinstance(o, list):
            stack.extend((path + (i,), val) for i, val in enumerate(o))
    return index


def candidates(rec, template):
    """[(path, (point, ci95[0], ci95[1]))] for every {point, ci95} object at a path matching the filled template.
    Elements without "*" must match exactly; "*" elements match by fnmatchcase on that one element."""
    level = [((), rec)]
    for el in template:
        nxt = []
        for path, o in level:
            if not isinstance(o, dict):
                continue
            if "*" not in el:
                if el in o:
                    nxt.append((path + (el,), o[el]))
            else:
                nxt.extend((path + (key,), val) for key, val in o.items() if fnmatch.fnmatchcase(key, el))
        level = nxt
    out = []
    for path, o in level:
        ci = o.get("ci95") if isinstance(o, dict) else None
        if isinstance(o, dict) and is_num(o.get("point")) and isinstance(ci, list) and len(ci) == 2 \
                and all(map(is_num, ci)):
            out.append((list(path), (float(o["point"]), float(ci[0]), float(ci[1]))))
    return out


def rounds_to(token: str, x: float, scale: int) -> bool:
    """Printed token == scale*x rounded to the token's own decimals (Python's correctly rounded format() of the
    double scale*x). Numeric compare: sign style and -0.0 vs +0.0 are not compared."""
    num = token.rstrip("%").replace("−", "-")
    d = len(num.split(".")[1]) if "." in num else 0
    return Decimal(num) == Decimal(format(scale * x, f".{d}f"))


def summ_compare(token: str, x: float) -> tuple:
    """Rule slot.summ: x100 if the token carries %, else the unique scale in (1, 100) that reproduces it."""
    if token.endswith("%"):
        return rounds_to(token, x, 100), "slot.summ_pct_x100"
    hits = [s for s in (1, 100) if rounds_to(token, x, s)]
    if len(hits) == 2:
        return True, "slot.summ_x1_and_x100"
    return (True, f"slot.summ_x{hits[0]}") if hits else (False, "slot.summ_x1_or_x100")


def bump_last_digit_of(token_or_repr: str) -> Decimal:
    """+1 in the last printed digit (as a Decimal step)."""
    dec = Decimal(token_or_repr.rstrip("%").replace("−", "-"))
    return Decimal(1).scaleb(dec.as_tuple().exponent)


def check_slot(det, artifact, slot_id, arts, recs, xindex=None):
    m = SLOT_ID.match(slot_id)
    if not m:
        raise RowError("id_unparsed")
    slot_key, k = slot_id.rsplit("#", 1)[0], int(m["k"])
    art = arts[artifact]
    if slot_key not in art["slots"] or slot_key not in art["full"]:
        raise RowError("slot_missing")
    printed, full = art["slots"][slot_key], art["full"][slot_key]
    det.update(slot_key=slot_key, k=k, printed=printed, full=full)

    if UNITS19_ID.match(slot_key) or PLANT_COUNT_ID.match(slot_key) or AFTER_RUNS_ID.match(slot_key):
        p_int, f_int = INT_NUM.findall(printed), INT_NUM.findall(full)
        if k >= len(p_int) or k >= len(f_int):
            raise RowError("printed_format")
        det["printed_token"] = p_int[k]
        x, rule = int_source(det, slot_key, k, printed, recs)
        det.update(rule=rule, source_value=repr(x), _x=x,
                   equal=int(p_int[k].replace(",", "")) == int(x) == int(f_int[k].replace(",", "")))
        return

    p_tok, f_tok = PRINTED_NUM.findall(printed), FULL_NUM.findall(full)
    if k >= len(p_tok) or k >= len(f_tok):
        raise RowError("printed_format")
    det["printed_token"] = p_tok[k]

    # summ rows: "x (lo to hi)" printed, "x [lo, hi]" full. Candidates come only from the path the id's parsed fields
    # name; exactly one candidate's (point, ci95[0], ci95[1]) must equal the full triple, as doubles.
    fam = [(rx.match(slot_key), rec, tpl) for art, rx, rec, tpl in SUMM_FAMILIES if art == artifact]
    fam = [(fm, rec, tpl) for fm, rec, tpl in fam if fm]
    if len(fam) != 1:
        raise RowError("unsupported_row")
    fm, rec, tpl = fam[0]
    g = fm.groupdict()
    rec_name = rec[g["slot"]] if isinstance(rec, dict) else rec
    template = [el.format(**g) for el in tpl]
    if len(p_tok) != 3 or len(f_tok) != 3:
        raise RowError("printed_format")
    full_vals = tuple(float(t) for t in f_tok)
    if xindex is not None:  # cross-check over every loaded record, reported next to the template result; decides nothing
        det["record_match_paths"] = {r: idx[full_vals] for r, idx in xindex.items() if full_vals in idx}
        det["record_match_counts"] = {r: len(p) for r, p in det["record_match_paths"].items()}  # reply: counts only
    cands = candidates(recs[rec_name], template)
    matched = [p for p, trip in cands if trip == full_vals]
    det.update(source_record=rec_name, template=template, candidates=[p for p, _ in cands],
               full_values=list(full_vals), matched=matched, n_candidates=len(cands), n_matches=len(matched))
    if not cands:
        raise RowError("leaf_missing")
    if len(matched) != 1:
        raise RowError("no_match" if not matched else "multi_match")
    if xindex is not None:
        det["template_match_in_cross_check"] = matched[0] in det["record_match_paths"].get(rec_name, [])
    # Value-free provenance flag for the "*" the id cannot fill (reviewer N1). match_pooled_all_origins: None when no
    # candidate path names a pooling at all; False fails the row, because route C's records are pooled-world (2e and
    # the reviewer, 10-20; not taken from the artifact). (match_panel_pooled went with the panel.*.Y guess, amendment 5.)
    path = [str(e) for e in matched[0]]
    if any("pooled_all_origins" in str(e) or "paper_cells" in str(e) for p, _ in cands for e in p):
        det["match_pooled_all_origins"] = any("pooled_all_origins" in e for e in path)
        if not det["match_pooled_all_origins"]:
            raise RowError("pooling_mismatch")
    x = full_vals[k]
    det.update(source_index=("point", "ci95[0]", "ci95[1]")[k], source_value=repr(x))
    equal, rule = summ_compare(p_tok[k], x)
    det.update(equal=equal, rule=rule, _x=x)


def int_source(det, slot_key, k, printed, recs):
    """Integer slots; returns (value, rule). All three are DEF: row 19 (rc2 L46), row 25 count (rc1b L172) and row 14
    (rc1b L40). A leaf is never picked by matching its value."""
    rule = "slot.int_exact_def"
    if m := UNITS19_ID.match(slot_key):
        # DEF rc2 L46 (row 19): `19.{…, real_units_above_tau_B}.<label C>@0.001` -> the pooled B (sweep15) record's rows
        # (honest pool B within it): `pools_at_draw0["0.001"]` … `B.units_above`. The row names no S definition, so
        # every pooled_all_origins S-definition row must agree (else leaf_ambiguous).
        per_s = walk(recs["B"], ["rows", m["pp"]])
        names = sorted(s for s in per_s if s.endswith("[pooled: pooled_all_origins]"))
        paths = [["rows", m["pp"], s, "pools_at_draw0", m["fpr"], "B", "units_above"] for s in names]
        det["source_record"] = "B"
    elif m := PLANT_COUNT_ID.match(slot_key):
        # DEF rc1b L172 (row 25): `25.plant.count.<f>.<δ>.<C|R|C+R>` -> `counts.<f>.<δ>.<ro>`, printed "k/n detected,
        # m missed, u unavailable" from `detected`, `cases`, `not_detected`, `unavailable` (= PLANT_FIELDS).
        field, word = PLANT_FIELDS.get(k, (None, None))
        if field is None:
            raise RowError("printed_format")
        if word:
            after = printed[[t.end() for t in INT_NUM.finditer(printed)][k]:].split()
            if not after or after[0].strip(",;") != word:
                raise RowError("printed_format")
        paths = [["counts", m["level"], m["offset"], m["pool"], field]]
        det["source_record"] = "PLANT"
    else:
        # DEF rc1b NOTE_slot_readout.md blob 194698115 L40 (row 14): `14.after_runs.<label>.<B|S>` (B = honest pool B,
        # S = sim honest pool S; never the pooled B (sweep15) record) ->
        # `meta.pooled_world.per_label["<label> | pooled_all_origins"].sizes.<pool>.after.runs`, in pooled A.
        m = AFTER_RUNS_ID.match(slot_key)
        paths = [["meta", "pooled_world", "per_label", f"{m['pp']} | pooled_all_origins", "sizes", m["set"], "after",
                  "runs"]]
        det["source_record"] = "A"
    vals = [walk(recs[det["source_record"]], p) for p in paths]
    det["source_paths"] = paths
    if not vals:
        raise RowError("leaf_missing")
    if len({repr(v) for v in vals}) != 1:
        raise RowError("leaf_ambiguous")
    x = vals[0]
    if not is_num(x) or x != int(x):
        raise RowError("not_an_integer")
    return x, rule


def check_fig(det, artifact, slot_id, figsrc, recs):
    src_pin, inputs = FIGS[artifact]
    src = figsrc[src_pin]
    det["rule"] = "fig.exact_float"
    if m := FIG_VALUE_ID.match(slot_id):
        i = int(m["i"])
        v = walk(src, ["values", i])
        rec_name = inputs.get(v.get("input"))
        if rec_name is None:
            raise RowError("fig_input_unpinned")
        if walk(src, ["inputs", v["input"], "sha256"]) != PINS[rec_name][2]:
            raise RowError("fig_input_pin_mismatch")
        leaf = walk(recs[rec_name], v["key"])
        if m["field"] == "value":
            printed, x = v["value"], leaf
        elif m["field"] == "point":
            printed, x = v["point"], walk(leaf, ["point"])
        else:
            j = int(m["j"])
            printed, x = walk(v, ["ci95", j]), walk(leaf, ["ci95", j])
        det.update(source_record=rec_name, key=v["key"])
    elif m := SCATTER_ID.match(slot_id):
        sp = walk(src, ["scatter_points", m["probe"]])
        if sp.get("key") != SCATTER_KEY.format(probe=m["probe"]):
            raise RowError("fig_key_unexpected")
        if walk(src, ["inputs", "pertask", "sha256"]) != PINS["PERTASK"][2]:
            raise RowError("fig_input_pin_mismatch")
        pt = walk(sp, ["points", int(m["n"])])
        per_task = walk(recs["PERTASK"], ["rows", m["probe"], pt["origin"], "per_task"])
        hits = [e for e in per_task if e.get("cell") == pt["cell"] and e.get("arm") == m["arm"]]
        det.update(source_record="PERTASK", origin=pt["origin"], cell=pt["cell"], n_entries=len(hits))
        if len(hits) != 1:
            raise RowError("leaf_missing" if not hits else "leaf_ambiguous")
        printed, x = pt[m["arm"]], hits[0]["max_minus_tau"]
    elif slot_id.startswith("text#"):
        det.update(equal=None, rule="not_checked")  # the verifier's text tokenizer is not ours to reuse
        raise RowError("not_checked_text_token")
    else:
        raise RowError("id_unparsed")
    det.update(printed_value=repr(printed), source_value=repr(x))
    if not (is_num(printed) and is_num(x)):
        raise RowError("not_a_number")
    det.update(equal=float(printed) == float(x), _x=x, _printed=printed)


def definitions_ok(data) -> bool:
    """Each DEFINITIONS row, selected by its key text (first cell plus the start of the key) in the pinned slot-key
    NOTE, exists exactly once and holds every substring the templates above are built from."""
    for pin, prefix, needles in DEFINITIONS:
        rows = [ln for ln in data[pin].decode().splitlines() if ln.startswith(prefix)]
        if len(rows) != 1 or not all(n in rows[0] for n in needles):
            return False
    return True


def controls(details):
    """Positive: first summ slot row and first fig row, source +1 in the last printed digit -> must be unequal.
    Negative: every checked row's source value against itself under its own rule -> must be equal."""
    pc = []
    for kind in ("slot.summ", "fig."):
        det = next((d for d in details if d.get("rule", "").startswith(kind) and "_x" in d and d["equal"]), None)
        if det is None:
            pc.append({"kind": kind, "i": None, "unequal": False, "error": "no_row"})
            continue
        x = float(det["_x"])
        if kind == "slot.summ":
            scale = 100 if det["rule"] in ("slot.summ_pct_x100", "slot.summ_x100") else 1
            x_pert = float(Decimal(repr(x)) + bump_last_digit_of(det["printed_token"]) / scale)
        else:
            x_pert = float(Decimal(repr(x)) + bump_last_digit_of(repr(x)))
            if x_pert == x:  # a 17-digit repr step can be below half an ulp: fall back to the next double up
                x_pert = math.nextafter(x, math.inf)
        if x_pert == x or not math.isfinite(x_pert):  # must change the double, else the control is void (2e, 10-20)
            det["positive_control"] = {"perturbed_source": repr(x_pert), "unequal": False, "error": "pc_invalid"}
            pc.append({"kind": kind, "i": det["i"], "unequal": False, "error": "pc_invalid"})
            continue
        if kind == "slot.summ":
            unequal = not rounds_to(det["printed_token"], x_pert, scale)
        else:
            unequal = float(det["_printed"]) != x_pert
        det["positive_control"] = {"perturbed_source": repr(x_pert), "unequal": unequal}
        pc.append({"kind": kind, "i": det["i"], "unequal": bool(unequal), "error": None})
    nc_n = nc_eq = 0
    for det in details:
        if "_x" not in det:
            continue
        x = det["_x"]
        if det["rule"].startswith("slot.summ"):
            scale = 100 if det["rule"] in ("slot.summ_pct_x100", "slot.summ_x100") else 1
            d = len(det["printed_token"].rstrip("%").split(".")[1]) if "." in det["printed_token"] else 0
            same = rounds_to(format(scale * x, f".{d}f"), x, scale)
        elif det["rule"].startswith("slot.int_exact"):
            same = int(str(int(x))) == int(x)
        else:
            same = float(x) == float(x)
        det["negative_control_equal"] = same
        nc_n, nc_eq = nc_n + 1, nc_eq + bool(same)
    return pc, {"n": nc_n, "n_equal": nc_eq}


def selection_strings(obj):
    out, stack = set(), [obj]
    while stack:  # every string anywhere in the selection file, keys included
        o = stack.pop()
        if isinstance(o, str):
            out.add(o)
        elif isinstance(o, dict):
            stack.extend(o.keys())
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
    return out


@app.function(image=image, cpu=2.0, memory=16384, timeout=1200, retries=0, volumes={"/vol": vol})
def recheck(blobs: dict, head: str, stamp: str, script_sha: str) -> dict:
    pins_ok = {"script": sha256(Path(__file__).read_bytes()) == script_sha}
    data = {}
    for name, (_, _, want) in PINS.items():
        data[name] = gzip.decompress(blobs[name])
        pins_ok[name] = sha256(data[name]) == want
    vol.reload()
    for batch, b in BATCHES.items():
        for f in ("selection", "rows"):
            p = Path("/vol") / batch / ("recheck_selection.json" if f == "selection" else "rows.jsonl")
            pins_ok[f"vol:{batch}:{f}"] = p.is_file() and sha256(p.read_bytes()) == b[f]
    pins_ok["definitions"] = definitions_ok(data)
    if not all(pins_ok.values()):
        return {"stop": "pin_fail", "pins_ok": pins_ok}

    load = lambda n: json.loads(data.pop(n))  # noqa: E731
    arts = {"rc2": load("rc2"), "rc1b": load("rc1b")}
    figsrc = {n: load(n) for n in ("fig4_src", "fig1_src", "fig5_src", "figs78_src")}
    recs = {n: load(n) for n in ("B", "R37", "A", "A_group_excess", "KAPPA", "KAPPA_EXTRA", "ANCHOR", "PLANT",
                                 "PERTASK")}
    xindex = {n: triple_index(r) for n, r in recs.items()}  # cross-check only

    reply = {"stop": None, "pins_ok": pins_ok, "batches": {}}
    for batch, b in BATCHES.items():
        rc = json.loads(data[b["result"]])["recheck"]
        meta_ok = (rc["seed"], rc["k"], len(rc["rows"])) == (SEED, K, K)
        meta_ok = meta_ok and rc["path"] == f"{batch}/recheck_selection.json" and rc["sha256"] == b["selection"]
        rows = [tuple(r) for r in rc["rows"]]
        sel = selection_strings(json.loads((Path("/vol") / batch / "recheck_selection.json").read_bytes()))
        details, out_rows = [], []
        for i, (artifact, slot_id) in enumerate(rows):
            det = {"i": i, "artifact": artifact, "slot_id": slot_id, "rule": "none", "equal": False, "error": None}
            try:
                if artifact in arts:
                    check_slot(det, artifact, slot_id, arts, recs, xindex)
                elif artifact in FIGS:
                    check_fig(det, artifact, slot_id, figsrc, recs)
                else:
                    raise RowError("unsupported_artifact")
                if det["equal"] is False:
                    det["error"] = "unequal"
            except RowError as e:
                det["error"] = str(e)
                if det["equal"] is not None:
                    det["equal"] = False
            details.append(det)
            out_rows.append({key: det.get(key) for key in ("i", "artifact", "slot_id", "equal", "rule", "error",
                                                           "n_candidates", "n_matches", "match_pooled_all_origins",
                                                           "record_match_counts",
                                                           "template_match_in_cross_check")})
        pc, nc = controls(details)
        for det in details:
            det.pop("_x", None)
            det.pop("_printed", None)
        record = {
            "what": f"Independent recheck of the numbers verifier's seeded 'equal' rows, batch {batch}; values "
                    "included; Volume only.",
            "git": head, "stamp": stamp, "script_sha256": script_sha, "pins": PINS, "batch": batch, "batch_pins": b,
            "pins_ok": pins_ok, "selection_meta_ok": meta_ok, "rows": details, "positive_control": pc,
            "negative_control": nc, "n_rows_in_selection_file": sum(a in sel and s in sel for a, s in rows),
        }
        out = Path("/vol") / batch / "recheck_independent" / f"recheck_independent_{stamp}.json"
        if out.exists():
            raise FileExistsError(f"{out} exists; results are append-only")
        out.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(record, indent=1, sort_keys=True).encode()
        out.write_bytes(body)
        vol.commit()
        reply["batches"][batch] = {
            "selection_meta_ok": bool(meta_ok), "n_rows_in_selection_file": record["n_rows_in_selection_file"],
            "rows": out_rows, "n_equal": sum(r["equal"] is True for r in out_rows),
            "n_unequal": sum(r["equal"] is False for r in out_rows),
            "n_not_checked": sum(r["equal"] is None for r in out_rows), "positive_control": pc,
            "negative_control": nc, "detail_path": f"{VOL_NAME}:{batch}/recheck_independent/{out.name}",
            "detail_sha256": sha256(body),
        }
    return reply


def git(*args, binary=False):
    out = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent, check=True, capture_output=True).stdout
    return out if binary else out.decode().strip()


@app.local_entrypoint()
def main():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    me = Path(__file__).resolve()
    rel = me.relative_to(git("rev-parse", "--show-toplevel"))
    if git("symbolic-ref", "-q", "HEAD") != BRANCH:
        raise SystemExit(f"STOP: not on {BRANCH}")
    if git("status", "--porcelain", "--", str(me.parent)):
        raise SystemExit("STOP: uncommitted changes in the experiment folder")
    head = git("rev-parse", "HEAD")
    script_sha = sha256(me.read_bytes())
    if sha256(git("show", f"{head}:{rel}", binary=True)) != script_sha:
        raise SystemExit("STOP: script on disk differs from HEAD")
    blobs = {}
    for name, (commit, path, want) in PINS.items():
        raw = git("show", f"{commit}:{path}", binary=True)
        if path.endswith(".gz"):
            blobs[name] = raw  # sha of the decompressed bytes is checked in the container
        elif sha256(raw) != want:
            raise SystemExit(f"STOP: pin sha mismatch for {name}")
        else:
            blobs[name] = gzip.compress(raw)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    reply = recheck.remote(blobs, head, stamp, script_sha)
    print("RECHECK_RESULT " + json.dumps({"git": head, "stamp": stamp, "script_sha256": script_sha, **reply}))
    ok = reply["stop"] is None
    for b in reply.get("batches", {}).values():
        nc = b["negative_control"]
        ok = ok and all(p["unequal"] for p in b["positive_control"]) and nc["n"] >= 1 and nc["n"] == nc["n_equal"]
        ok = ok and b["selection_meta_ok"] is True and b["n_rows_in_selection_file"] == K  # reviewer N2
    if not ok or len(reply.get("batches", {})) != len(BATCHES):
        raise SystemExit("STOP: pins, selection or controls failed (see RECHECK_RESULT)")
