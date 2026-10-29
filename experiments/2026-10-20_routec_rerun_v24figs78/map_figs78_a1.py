"""figs78 element tables + slot-24 quantity map for the route C FINAL-release record v2.4 (09's 14:05 conditions 1-3; 22:46 Q4-Q6).

One Modal CPU call. a1's addendum 24 values file is mounted READ-ONLY. Only keys, a1 verdict / match flags, value-identity booleans,
gate booleans and counts leave the container; no value, and nothing is sent to a1. Pre-statement: PRESTATED_v24figs78_*.md + AMEND1/2.

  "$M" run --detach map_figs78_a1.py --inputs-dir <frozen inputs dir> --stamp <YYYY-mm-dd_HHMMSS>

Exit labels (pre-stated): PLUMBING (an input / values sha, the real-data identity gate, a container exception) -> STOP to 09, no mapping;
STOP_TO_09 (any other gate false; the record is still written, keys only); OK (all 10 gates true).
r2 (Codex r1 REQUIRED 1-3, 5, 10, 11 and OPTIONAL 4, 9): a1's own printed-count arithmetic; a closed element inventory with expected counts
and unique identities; type-safe value identity; release figure status after all gates; stamp validation; bound <= cap by code.
"""

import copy
import hashlib
import json
import os
import re
import traceback
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-rerun_v24figs78_kill-2026-11-19"
A1_VOL = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"  # a1's output Volume (cleanroom_anchor.py @ 44f0cae2c L15)
VALUES = ("cleanroom_anchor_values_2026-10-20_135936.json", "8d501fd89a2a3faa860cf378e13a59973030b49fec37602173fa55ecffaf872d")
ONPOL, SUMM = "fig_pertask_onpolicy_figs78_routecA_source.json", "fig_pertask_summary_figs78_routecA_source.json"
PINS = {  # input file name -> (sha256, source commit:path); extracted from git by the launch RUN_ME, re-checked here
    ONPOL: ("62cb3739c9058a6c183df69526fabd0bc1a5e7fc925a8560864666c6b3050065",
            "3efcd4a05e086ce32f8af67a9c4fda0db78abc42:experiments/2026-10-15_fpr_sweep_507_prep/results/fig_pertask_onpolicy_figs78_routecA_source.json"),
    SUMM: ("1f9a873e39192d2258eeafb51f22b50b17ce274b8755708730f00d256096e19e",
           "3efcd4a05e086ce32f8af67a9c4fda0db78abc42:experiments/2026-10-15_fpr_sweep_507_prep/results/fig_pertask_summary_figs78_routecA_source.json"),
    "cleanroom_anchor_2026-10-20_135936.json": ("069b5dfa19646946c37e1e8608b43062e64d10bbc42613c6b91d3f643f5c0871",
        "44f0cae2c78a2464ecaaf548cb907e80c4ffcabe:experiments/2026-10-16_cleanroom_pooled_repro/results/cleanroom_anchor_2026-10-20_135936.json"),
    "anchor_followup24_2026-10-20_141527.json": ("bdd927eb9f5416e7fa65083a5e0ec2042fb03d105d7de077ffc6b24eb3fc63da",
        "b42018292ae332346ddd49e5e3885864821cd747:experiments/2026-10-16_cleanroom_pooled_repro/results/anchor_followup24_2026-10-20_141527.json"),
    "provenance_v84_final17_0141.jsonl": ("220d25d21943226791637581afda92faac595a3b15b9b6d6bbd3fd1a1e575971",
        "98b5054fbf8306e54711b1a7bacd72b5f5e0af2a:experiments/2026-10-20_routec_numbers_verify/post/provenance_v84_final17_0141.jsonl"),
}
# a1 cleanroom_anchor.py @ 44f0cae2c (blob sha256 02db267f…): L40-42 input pins, L45 COMBO, L46 SETS, L72-73 origin_key, L491-495 printed
A1_AN_PINS = {"stage_d.json": "468655154f826ec1a93e08333eb802e68abea6db65a8e2f80a3e0f23465f36d9",
              "anchor.json": "2763ad99b7d37ee25784d9b58563a60d5d6c4b425656e45a97f71b82684cfbad",
              "rc1b.json": "ab8dcb439b942c92d30b784ae1e6a3a193b18127391ff1753426663b726e728b"}
COMBO = "pooled_all_origins"
UNION_SET = f"{COMBO} (union stream)"
SETS = ["S_old", "S_new", "S_newmains", "S_paper493"]
PRINTED10 = ["24.n_cells.lead.own_all_rs_uncut_L48 C", "24.r2.lead.own_all_rs_uncut_L48 C.yardstick",
             "24.r2.yardstick_halfwidth.lead.own_all_rs_uncut_L48 C", "24.r2.lead.own_all_rs_uncut_L48 C.main",
             "24.r2.lead.own_all_rs_uncut_L48 C.delta_main_minus_yardstick",
             "24.r2.sensitivity_run2_anchor.own_all_rs_uncut_L48 C.delta_main_minus_yardstick"] + [
             f"24.r2.{b}.own_all_rs_uncut_L48 C.delta_main_minus_yardstick.ci95[{i}]" for b in ("lead", "sensitivity_run2_anchor") for i in (0, 1)]
SLOT24_ROWS = [69, 70, 71, 72, 73, 74] + list(range(241, 249))
SLOT24_RC1B = "experiments/2026-10-17_routec_slot_readout/results/slots_routeC_writing_rc1b.json@600f63614"
COMP_SUFFIX = {None: "", "point": "", "interval_low": ".ci95[0]", "interval_high": ".ci95[1]"}
HALFWIDTH_KEY = "24.r2.yardstick_halfwidth.lead.own_all_rs_uncut_L48 C"
META = {("anchor", ("gate", "pass")): "metadata: gate boolean (not drawn; disclosure only)",
        ("pertask", ("bootstrap", "sets", COMBO)): "metadata: bootstrap set descriptor (not drawn; disclosure only)"}
NOT_DRAWN = "disclosure only (not drawn)"
# PRESTATED §5: the closed inventory per figure, by class (Codex r1 REQUIRED 2)
C_PT, C_CI, C_NC, C_SC, C_NP, C_DP = ("union-stream r2 point -> a1 anchor_pt (pooled_all_origins point)",
                                      "union-stream r2 interval end -> a1 anchor_ci", "union-stream n_cells -> a1 anchor_ci",
                                      "Fig 7 scatter coordinate (Stage D max_minus_tau) -> a1 stage_d_pt",
                                      "point count -> a1 anchor_pt lead pooled_all_origins n_cells",
                                      "drawn-point count -> a1 anchor_pt lead pooled_all_origins n_cells")
EXPECT = {ONPOL: {C_PT: 2, C_CI: 4, C_NC: 1, C_SC: 3924, C_NP: 1, C_DP: 2, META[("anchor", ("gate", "pass"))]: 1,
                  META[("pertask", ("bootstrap", "sets", COMBO))]: 1},
          SUMM: {C_PT: 12, C_CI: 24, C_NC: 6, META[("anchor", ("gate", "pass"))]: 1, META[("pertask", ("bootstrap", "sets", COMBO))]: 1}}
STAMP_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}$")


def origin_key(s):
    return f"{s} (all-origins draws) [pooled: {COMBO}]"


app = modal.App(APP)
a1_vol = modal.Volume.from_name(A1_VOL, version=2)
out_vol = modal.Volume.from_name(APP, create_if_missing=True, version=2)
image = modal.Image.debian_slim(python_version="3.12")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def a1_index(rows):
    """(family, key) -> the rows with that key. Lookups require exactly one row."""
    idx = {}
    for r in rows:
        idx.setdefault((r.get("family"), r.get("key")), []).append(r)
    return idx


def a1_fam_counts(rows):
    """a1's own by_family / n_verdict arithmetic (cleanroom_anchor.py L601-610), recomputed from the values file."""
    fam = {}
    for r in rows:
        f_ = fam.setdefault(r["family"], {"rows": 0, "verdict": 0, "verdict_match": 0, "unresolved": 0})
        f_["rows"] += 1
        if r["verdict"]:
            f_["verdict"] += 1
            f_["verdict_match"] += r["match"] is True
            f_["unresolved"] += r["kind"] == "unresolved"
    vr = [r for r in rows if r["verdict"]]
    return fam, len(vr), sum(r["match"] is True for r in vr)


def a1_printed_count(rows):
    """a1's printed-10 arithmetic exactly (L611-613, L623): dicts with last-row-wins, then pboth per printed key (Codex r1 REQUIRED 1)."""
    pm = {r["key"]: r["match"] for r in rows if r["family"] == "rc1b_24"}
    pdm = {r["key"]: r["match"] for r in rows if r["family"] == "rc1b_24_display"}
    return sum(pm.get(k) is True and pdm.get(f"display|{k}") is True for k in PRINTED10)


def same(v, ref, kind):
    """Value identity, type-safe (Codex r1 REQUIRED 3): counts are ints on both sides (never bool); values are int / float, not bool."""
    if isinstance(v, bool) or isinstance(ref, bool):
        return False
    if kind == "int":
        return type(v) is int and type(ref) is int and v == ref
    return isinstance(v, (int, float)) and isinstance(ref, (int, float)) and v == ref


def map_one(idx, fam, key, v, kind):
    hits = idx.get((fam, key), [])
    if len(hits) != 1:
        return {"a1_family": fam, "a1_key": key, "a1_rows": len(hits), "a1_verdict": None, "a1_match": None, "value_identity": None,
                "status": "PRELIMINARY", "reason": "no unique a1 row"}
    r = hits[0]
    vid = same(v, r.get("ref"), kind)
    ok = r.get("verdict") is True and r.get("match") is True and r.get("kind") == kind and vid
    out = {"a1_family": fam, "a1_key": key, "a1_rows": 1, "a1_verdict": r.get("verdict") is True, "a1_match": r.get("match") is True,
           "a1_kind_ok": r.get("kind") == kind, "value_identity": vid, "status": "FINAL" if ok else "PRELIMINARY"}
    if not ok:
        out["reason"] = "a1 row not a matched verdict row of the expected kind, or the plotted value differs from a1's reference"
    return out


def element_table(name, src, idx):
    inv = {origin_key(s): s for s in SETS}
    els = []
    for i, v in enumerate(src.get("values") or []):
        k = tuple(v.get("key") or [])
        inp = v.get("input")
        base = {"source": "values", "value_index": i, "input": inp, "key_path": "/".join(map(str, k))}
        if (inp, k) in META and set(v) == {"input", "key", "value"}:
            els.append(base | {"id": f"values/{base['key_path']}", "class": META[(inp, k)], "status": NOT_DRAWN})
        elif (inp == "anchor" and len(k) == 7 and k[0] == "union_stream" and k[3] == "sets" and k[4] == UNION_SET and k[5] == "r2"
              and set(v) == {"input", "key", "point", "ci95"}):
            blk, lab, q = k[1], k[2], k[6]
            els.append(base | {"id": f"values/{base['key_path']}/point", "component": "point", "class": C_PT}
                       | map_one(idx, "anchor_pt", f"{blk}|{lab}|{COMBO}|r2.{q}", v.get("point"), "value"))
            ci = v.get("ci95") if isinstance(v.get("ci95"), list) and len(v.get("ci95")) == 2 else [None, None]
            for j in (0, 1):
                els.append(base | {"id": f"values/{base['key_path']}/ci95[{j}]", "component": f"ci95[{j}]", "class": C_CI}
                           | map_one(idx, "anchor_ci", f"{blk}|{lab}|union|r2.{q}.ci95[{j}]", ci[j], "value"))
        elif (inp == "anchor" and len(k) == 6 and k[0] == "union_stream" and k[3] == "sets" and k[4] == UNION_SET and k[5] == "n_cells"
              and set(v) == {"input", "key", "value"}):
            blk, lab = k[1], k[2]
            els.append(base | {"id": f"values/{base['key_path']}", "component": "value", "class": C_NC}
                       | map_one(idx, "anchor_ci", f"{blk}|{lab}|union|n_cells", v.get("value"), "int"))
        else:
            els.append(base | {"id": f"values/{i}", "class": "UNCLASSIFIED", "status": "PRELIMINARY", "reason": "no pre-stated class"})
    for lab, node in (src.get("scatter_points") or {}).items():
        for p in node.get("points") or []:
            s = inv.get(p.get("origin"))
            for arm in ("run1", "run2", "sim"):
                base = {"source": "scatter_points", "label": lab, "cell": p.get("cell"), "origin": p.get("origin"), "arm": arm,
                        "id": f"scatter/{lab}/{p.get('cell')}/{p.get('origin')}/{arm}", "class": C_SC}
                if s is None:
                    els.append(base | {"status": "PRELIMINARY", "reason": "origin not one of a1's 4 set keys"})
                else:
                    els.append(base | map_one(idx, "stage_d_pt", f"{lab}|{s}|{p.get('cell')}|{arm}|max_minus_tau", p.get(arm), "value"))
    for lab, n in (src.get("n_points") or {}).items():
        els.append({"source": "n_points", "label": lab, "id": f"n_points/{lab}", "class": C_NP}
                   | map_one(idx, "anchor_pt", f"lead|{lab}|{COMBO}|n_cells", n, "int"))
    for lab, node in (src.get("drawn_points_per_panel") or {}).items():
        for panel, n in node.items():
            els.append({"source": "drawn_points_per_panel", "label": lab, "panel": panel, "id": f"drawn/{lab}/{panel}", "class": C_DP}
                       | map_one(idx, "anchor_pt", f"lead|{lab}|{COMBO}|n_cells", n, "int"))
    cls = {}
    for e in els:
        c = cls.setdefault(e["class"], {"n": 0, "FINAL": 0, "PRELIMINARY": 0, NOT_DRAWN: 0})
        c["n"] += 1
        c[e["status"]] += 1
    ids = [e["id"] for e in els]
    drawn = [e for e in els if e["status"] != NOT_DRAWN]
    complete = ({k: v["n"] for k, v in cls.items()} == EXPECT[name] and len(set(ids)) == len(ids)
                and (name != ONPOL or (list(src.get("scatter_points") or {}) == list(src.get("n_points") or {}) == list(src.get("drawn_points_per_panel") or {})
                                       and all(sorted(n) == ["main", "yardstick"] for n in (src.get("drawn_points_per_panel") or {}).values()))))
    return {"file": PINS[name][1].split(":", 1)[1], "commit": PINS[name][1].split(":", 1)[0], "sha256": PINS[name][0],
            "n_elements": len(els), "n_drawn": len(drawn), "classes": cls, "expected_class_counts": EXPECT[name],
            "inventory_complete": complete, "ids_unique": len(set(ids)) == len(ids),
            "elements_all_final": bool(drawn) and all(e["status"] == "FINAL" for e in drawn),
            "metadata_not_drawn": sorted(e["key_path"] for e in els if e["status"] == NOT_DRAWN),
            "elements": els}


def slot24_table(prov, idx, fu):
    seen = {}
    for r in prov:
        seen[r.get("row")] = seen.get(r.get("row"), 0) + 1
    rows = {r["row"]: r for r in prov}
    out = []
    for n in SLOT24_ROWS:
        r = rows.get(n) or {}
        kp = r.get("key_path")
        slot = kp[1] if isinstance(kp, list) and len(kp) == 2 and kp[0] == "slots" else None
        comp = (r.get("predicate") or {}).get("component")
        key = (slot + COMP_SUFFIX[comp]) if slot and comp in COMP_SUFFIX else None
        val, disp = idx.get(("rc1b_24", key), []), idx.get(("rc1b_24_display", f"display|{key}"), [])
        e = {"row": n, "provenance_row_unique": seen.get(n) == 1, "record_is_rc1b_600f63614": r.get("record") == SLOT24_RC1B,
             "record_sha_is_a1_pin": r.get("record_sha256") == A1_AN_PINS["rc1b.json"], "slot": slot, "component": comp, "a1_key": key,
             "in_a1_printed10": key in PRINTED10, "a1_value_rows": len(val),
             "a1_value_match": len(val) == 1 and val[0].get("match") is True and val[0].get("verdict") is True,
             "a1_display_rows": len(disp),
             "a1_display_match": len(disp) == 1 and disp[0].get("match") is True and disp[0].get("verdict") is True}
        if key == HALFWIDTH_KEY:
            e["halfwidth_followup"] = {"identified_readings_is_half_range": fu.get("identified_readings") == ["half_range"],
                                       "ident_match": fu.get("ident_match") is True, "all_pairs_match": fu.get("all_pairs_match") is True}
        ok = (e["provenance_row_unique"] and e["record_is_rc1b_600f63614"] and e["record_sha_is_a1_pin"] and e["in_a1_printed10"]
              and e["a1_value_match"] and e["a1_display_match"] and (key != HALFWIDTH_KEY or all(e["halfwidth_followup"].values())))
        e["status"] = "FINAL" if ok else "PRELIMINARY"
        out.append(e)
    return out


# Bound <= cap by code (Codex r1 REQUIRED 11): (240 + 120 + 2) s x $0.00002198/s = $0.00795676 <= the $0.008 cap (AMEND2)
@app.function(image=image, cpu=(1, 1), memory=(4096, 4096), timeout=240, startup_timeout=120, scaledown_window=2, retries=0,
              volumes={"/a1": a1_vol.read_only(), "/out": out_vol})
def run(blobs: dict, stamp: str) -> dict:
    if not _export_fixture():  # AMEND10 (b): the export is checked on a synthetic raise before any data is read
        raise RuntimeError("STOP PLUMBING: export_fixture")
    res, err = None, None
    try:
        res = _run(blobs, stamp)
    except Exception as e:  # only the export leaves the container: type, our deepest line, the message with every digit masked
        err = _export(e)
    if err:
        raise RuntimeError(f"STOP PLUMBING: {err}") from None  # F-N2: never chained, so no original message can print
    return res


def _export(e):
    """AMEND10 (b) + AMEND11 (F-N1, F-N2): `<type>@L<deepest map_figs78_a1.py line>: <message>`, where every non-printable or
    non-ASCII character is "?" first, then every ASCII digit is "#"; one line, at most 120 characters. It cannot fail."""
    try:
        ours = [f.lineno for f in traceback.extract_tb(e.__traceback__) if Path(f.filename).name == Path(__file__).name]
        msg = re.sub(r"[0-9]", "#", re.sub(r"[^\x20-\x7e]", "?", str(e)))
        msg = msg.replace("\r", " ").replace("\n", " ")
        return f"{type(e).__name__}@L{ours[-1] if ours else 0}: {msg}"[:120]
    except Exception:
        return f"{type(e).__name__}@export_failed"


def _export_fixture():
    """Synthetic raises (AMEND10 (b) + AMEND11 F-N1, F-N3): ASCII digits, a superscript, Arabic-Indic digits and U+2028 must export as
    `ValueError@L<line >= 1>: <masked text>` within 120 characters."""
    cases = [("synthetic 12345", "synthetic #####"), ("sup \u00b2", "sup ?"), ("arabic \u0661\u0662", "arabic ??"), ("sep \u2028 x", "sep ? x")]
    for raw, want in cases:
        try:
            raise ValueError(raw)
        except ValueError as e:
            s = _export(e)
        if re.fullmatch(r"ValueError@L[1-9][0-9]*: " + re.escape(want), s) is None or len(s) > 120:
            return False
    return True


def _run(blobs, stamp):
    gates, rec = [], {"stamp": stamp, "app": APP, "inputs": {k: {"sha256": v[0], "source": v[1]} for k, v in PINS.items()},
                      "a1_values_file": {"volume": A1_VOL, "path": VALUES[0], "sha256": VALUES[1], "mount": "read-only"}}

    def gate(name, ok):
        gates.append({"name": name, "ok": bool(ok)})
        return bool(ok)

    # ---- 0. output boundary (Codex r1 REQUIRED 10) ----
    if not (isinstance(stamp, str) and STAMP_RE.fullmatch(stamp)):
        raise ValueError("stamp")
    outdir = Path("/out", stamp)
    # AMEND10 (a): the first launch failed here (/out is the Volume mount; resolve() follows it). Compare with the resolved mount, and keep
    # the escape check: the folder name is exactly the stamp and carries no path separator.
    if not (outdir.name == stamp and "/" not in stamp and "\\" not in stamp and os.sep not in stamp
            and outdir.resolve().parent == Path("/out").resolve()):
        raise ValueError("outdir")
    # ---- 1. input identity (plumbing) ----
    if not gate("inputs_sha", sorted(blobs) == sorted(PINS) and all(sha(blobs[k]) == PINS[k][0] for k in PINS)):
        return rec | {"gates": gates, "exit": "PLUMBING"}
    raw = Path("/a1", VALUES[0]).read_bytes()
    if not gate("a1_values_sha", sha(raw) == VALUES[1]):
        return rec | {"gates": gates, "exit": "PLUMBING"}
    rows = json.loads(raw)["rows"]
    del raw
    anchor_rec = json.loads(blobs["cleanroom_anchor_2026-10-20_135936.json"])
    fu = json.loads(blobs["anchor_followup24_2026-10-20_141527.json"])
    srcs = {k: json.loads(blobs[k]) for k in (ONPOL, SUMM)}
    prov = [json.loads(x) for x in blobs["provenance_v84_final17_0141.jsonl"].decode().splitlines() if x.strip()]

    # ---- 2. real data first: a1's own counts recomputed from the values file == a1's committed record (integer identity) ----
    fam, nv, nvm = a1_fam_counts(rows)
    real = (gate("real_by_family_equal", fam == anchor_rec["by_family"])
            & gate("real_n_verdict_equal", nv == anchor_rec["n_verdict"] and nvm == anchor_rec["n_verdict_match"])
            & gate("real_printed10_equal", len(PRINTED10) == anchor_rec["printed"]["n"] and a1_printed_count(rows) == anchor_rec["printed"]["n_match"]))
    if not real:
        return rec | {"gates": gates, "exit": "PLUMBING"}
    idx = a1_index(rows)

    # ---- 3. the figures read the same record bytes a1 pinned ----
    gate("figs78_inputs_are_a1_pins", all(s["inputs"]["anchor"]["sha256"] == A1_AN_PINS["anchor.json"]
                                          and s["inputs"]["pertask"]["sha256"] == A1_AN_PINS["stage_d.json"] for s in srcs.values()))
    rec["a1_duplicate_keys_by_family"] = {f: sum(1 for (f2, _), v in idx.items() if f2 == f and len(v) > 1)
                                          for f in sorted({str(r.get("family")) for r in rows})}

    # ---- 4. element tables (closed inventory; every drawn element must map) ----
    tables = [element_table(k, srcs[k], idx) for k in (ONPOL, SUMM)]
    for t in tables:
        gate(f"element_table_complete_and_final:{Path(t['file']).name}", t["inventory_complete"] and t["elements_all_final"])
    rec["element_tables"] = tables

    # ---- 5. slot 24: 14 printed rows -> 10 a1 quantities ----
    s24 = slot24_table(prov, idx, fu)
    rec["slot24_rows"] = s24
    rec["slot24_quantities"] = sorted({e["a1_key"] for e in s24 if e["a1_key"]})
    gate("slot24_14_rows_10_quantities_all_final", len(s24) == 14 and rec["slot24_quantities"] == sorted(PRINTED10)
         and all(e["status"] == "FINAL" for e in s24))

    # ---- 6. negative controls on in-memory copies (each predicted to FAIL its check; Codex r1 OPTIONAL 9) ----
    nc = {}
    base_onpol = {e["id"]: e["status"] for e in tables[0]["elements"]}
    s1 = copy.deepcopy(srcs[ONPOL])
    lab1 = next(iter(s1["scatter_points"]))
    p0 = s1["scatter_points"][lab1]["points"][0]
    old_ids = {f"scatter/{lab1}/{p0.get('cell')}/{p0.get('origin')}/{a}" for a in ("run1", "run2", "sim")}
    p0["cell"] = "__planted_cell_absent_from_a1__"
    new_ids = {f"scatter/{lab1}/{p0['cell']}/{p0.get('origin')}/{a}" for a in ("run1", "run2", "sim")}
    t1 = element_table(ONPOL, s1, idx)
    st1 = {e["id"]: (e["status"], e.get("a1_rows")) for e in t1["elements"]}
    nc["NC1_planted_key_absent"] = {
        "predicted": "the 3 planted elements have no a1 row; the 3 replaced ids were FINAL; every other element unchanged; table not all FINAL",
        "as_predicted": (all(base_onpol.get(i) == "FINAL" for i in old_ids) and all(st1.get(i) == ("PRELIMINARY", 0) for i in new_ids)
                         and all(st1[i][0] == base_onpol[i] for i in st1 if i not in new_ids) and not t1["elements_all_final"])}
    used = next(e for e in tables[0]["elements"] if e["status"] == "FINAL" and e.get("a1_key"))
    kk = (used["a1_family"], used["a1_key"])
    idx2 = dict(idx)
    idx2[kk] = [dict(idx[kk][0], match=False)]
    t2 = element_table(ONPOL, srcs[ONPOL], idx2)
    flipped = [e["id"] for e in t2["elements"] if e["status"] != base_onpol[e["id"]]]
    nc["NC2_a1_match_flipped"] = {"predicted": "exactly the elements mapped to the flipped a1 row turn PRELIMINARY (>= 1); table not all FINAL",
                                  "as_predicted": len(flipped) >= 1 and all(base_onpol[i] == "FINAL" for i in flipped)
                                  and {e["id"] for e in tables[0]["elements"] if (e.get("a1_family"), e.get("a1_key")) == kk} == set(flipped)
                                  and not t2["elements_all_final"]}
    rec["negative_controls"] = nc
    gate("negative_controls_fail_as_predicted", all(v["as_predicted"] for v in nc.values()))

    # ---- 7. release figure status AFTER all gates (Codex r1 REQUIRED 5; AMEND1) ----
    allg = all(g["ok"] for g in gates)
    for t in tables:
        t["figure_status"] = "FINAL" if allg and t["inventory_complete"] and t["elements_all_final"] else "PRELIMINARY"
    rec["gates"] = gates
    rec["exit"] = "OK" if allg else "STOP_TO_09"
    outdir.mkdir(parents=False, exist_ok=False)
    (outdir / "figs78_map.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    out_vol.commit()
    return rec


@app.local_entrypoint()
def main(inputs_dir: str, stamp: str):
    if not STAMP_RE.fullmatch(stamp):
        raise SystemExit("STOP stamp format")
    blobs = {k: Path(inputs_dir, k).read_bytes() for k in PINS}
    rec = run.remote(blobs, stamp)
    out = Path(__file__).parent / "results" / f"figs78_map_{stamp}.json"
    out.write_text(json.dumps(rec, indent=1, sort_keys=True))
    g = rec.get("gates", [])
    print(f"FIGS78_MAP exit={rec.get('exit')} gates_true={sum(x['ok'] for x in g)}/{len(g)} out={out.name}")
