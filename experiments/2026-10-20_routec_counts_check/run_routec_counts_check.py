"""COUNTSCHECK launch: route C counts check (pre-stated in PRESTATED_OUTCOME_2026-10-20.md @ aef74386b; 09 OK with 2 conditions).

One `modal run --detach` with three steps, all driven by the local entrypoint:
  1. laptop: load every pinned input by git show + sha256; reduce the value records with the committed jq filters (countscheck.py);
  2. Modal `read_pilot_headers`: the 20 pilot .eval HEADERS (Inspect header_only) on the read-only Volume, one "any token count > 0" bit
     per model name per file (no token count, message, transcript, score or text leaves the container);
  3. laptop: every count derivation and count comparison (countscheck.derive_all), then Modal `compare_shares`: the record's share,
     percent, printed-precision and percent-based flag leaves are compared with my counts INSIDE the container (09's condition 1) and
     only booleans come back.
The laptop prints my counts and booleans only, never a record value. Results are written with exclusive create.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import modal

APP_NAME = "probe-shift_aswin-surya_sub-rccount_pilothdr_kill-2026-11-19"  # 60 chars (<= 63)
PILOT_VOLUME = "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"
MOUNT, ROOT = "/a25", "simh_evals_v1"
N_PILOT = 20
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "input_tokens_cache_write", "input_tokens_cache_read",
                "reasoning_tokens")
# Printed numbers, from NOTE_SP L11-L12 and L112-L113 (never from the record)
I1_DP, I1_V83_PRINTED = 0, Decimal("6")
I2_DP, I2_V83_PRINTED, I2_V83_PRINTED_WO = 1, Decimal("4.3"), Decimal("2.9")
TOL = Fraction(1, 10**9)
MISSING = object()

image = modal.Image.debian_slim(python_version="3.12").pip_install("inspect_ai==0.3.276", "zstandard==0.25.0")
app = modal.App(APP_NAME)
pilot_volume = modal.Volume.from_name(PILOT_VOLUME).with_mount_options(read_only=True)


# ---------------------------------------------------------------- container: pilot headers

def bits_from_log(log) -> dict:
    """Header fields kept: status, completed sample count, and per model name one bit (any token count > 0)."""
    usage = (log.stats.model_usage if getattr(log, "stats", None) else None) or {}
    models = {}
    for name, u in usage.items():
        vals = [getattr(u, f, None) for f in TOKEN_FIELDS]
        models[str(name)] = any(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in vals)
    results = getattr(log, "results", None)
    return {"status": str(log.status), "completed_samples": int(results.completed_samples) if results else None,
            "models_any_tokens": models}


@app.function(image=image, volumes={MOUNT: pilot_volume}, cpu=1.0, memory=2048, timeout=600, retries=0)
def read_pilot_headers(files_json: str) -> str:
    from inspect_ai.log import read_eval_log

    rows = []
    for f in json.loads(files_json):
        p = Path(MOUNT) / ROOT / f["path"]
        sha_ok = hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"]
        if not sha_ok:
            raise RuntimeError(f"STOP: pilot file sha256 mismatch (row {len(rows)})")
        row = {"path": f["path"], "sha_ok": sha_ok, "has_header": False, "read_error": None, "status": None,
               "completed_samples": None, "models_any_tokens": {}}
        try:
            log = read_eval_log(str(p), header_only=True)
        except Exception as e:  # noqa: BLE001 - a header that can't be read is counted UNKNOWN downstream (NOTE_SF L18), never dropped
            row["read_error"] = type(e).__name__
        else:
            row.update(has_header=True, **bits_from_log(log))
        rows.append(row)
    return json.dumps({"rows": rows})


# ---------------------------------------------------------------- container: share / percent comparisons (09's condition 1)

def get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return MISSING
        d = d[k]
    return d


def rhu(x: Fraction, dp: int) -> Decimal:
    with localcontext() as c:
        c.prec = 60
        d = Decimal(x.numerator) / Decimal(x.denominator)
    return d.quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)


def close(rv, mine) -> bool:
    if mine is None or rv is MISSING or isinstance(rv, bool) or not isinstance(rv, (int, float)):
        return False
    return abs(Fraction(rv) - mine) <= TOL


def first_word(s):
    return s.split()[0].strip(":;,.()") if isinstance(s, str) and s.strip() else None


def share_set(c: dict) -> dict:
    n, s, u = c["n"], c["sol_using"], c["unknown"]
    return {"all_runs_denominator": Fraction(s, n) if n else None,
            "unknown_counted_sol_upper": Fraction(s + u, n) if n else None,
            "unknown_excluded": Fraction(s, n - u) if n - u else None}


def compare_record_shares(record_bytes: bytes, payload: dict) -> dict:
    """Booleans only. `payload` holds my counts (countscheck.derive_all); the record is read here, never on the laptop."""
    out = {"sha_ok": hashlib.sha256(record_bytes).hexdigest() == payload["simpool_sha256"], "items": {}}
    if not out["sha_ok"]:
        return out
    items = json.loads(record_bytes)["items"]
    i1, i2, res = items.get("1_sim_footnote", {}), items.get("2_failed_footnote", {}), out["items"]
    I1, I2 = '.items["1_sim_footnote"]', '.items["2_failed_footnote"]'

    def put(iid, key, rv, mine, label, cite, dp=None, share=False):
        r = {"record_key": key, "label": label, "cite": cite, "record_key_present": rv is not MISSING, "equal": close(rv, mine)}
        if dp is not None:
            r["equal_printed_precision"] = mine is not None and close(rv, Fraction(rhu(mine, dp)))
        if share:
            r["diag_record_is_percent"] = mine is not None and close(rv, mine * 100)
        res[iid] = r

    def put_bool(iid, key, rv, mine: bool, label, cite):
        res[iid] = {"record_key": key, "label": label, "cite": cite, "record_key_present": rv is not MISSING,
                    "equal": isinstance(rv, bool) and rv == mine, "mine": mine}

    def put_flag(iid, key, rv, unchanged: bool, label, cite):
        res[iid] = {"record_key": key, "label": label, "cite": cite, "record_key_present": rv is not MISSING,
                    "equal": first_word(rv) == ("unchanged" if unchanged else "changed"), "mine_unchanged": unchanged}

    IND, INT = "INDEPENDENT", "INTERNAL"
    sh = {sc: share_set(c) for sc, c in payload["item1"]["pool"].items()}
    for sc, d in sh.items():
        for name, m in d.items():
            put(f"S2.{sc}.{name}", f"{I1}.pool.{sc}.shares.{name}", get(i1, "pool", sc, "shares", name), m, IND,
                "REC_TEXT .what (three ways)", share=True)
    cell_sh = {sc: {cell: share_set(c) for cell, c in cells.items()} for sc, cells in payload["item1"]["cells"].items()}
    for sc, cells in cell_sh.items():
        for cell, d in cells.items():
            for name, m in d.items():
                put(f"S6.share.{sc}.{cell}.{name}", f"{I1}.detector_cells.{sc}[{cell}].shares.{name}",
                    get(i1, "detector_cells", sc, cell, "shares", name), m, IND, "REC_TEXT .v83_source", share=True)
    res["S3.printed_decimals_percent"] = {"record_key": f"{I1}.printed_decimals_percent", "label": IND, "cite": "NOTE_SP L112",
                                          "equal": get(i1, "printed_decimals_percent") == I1_DP}
    rc_all, rc_up, rc_ex = (sh["routec"][k] for k in ("all_runs_denominator", "unknown_counted_sol_upper", "unknown_excluded"))
    v_all = sh["v83"]["all_runs_denominator"]
    pct = {"routec_pool": rc_all, "routec_pool_upper": rc_up, "routec_pool_unknown_excluded": rc_ex, "v83_pool": v_all}
    for k, m in pct.items():
        put(f"S3.{k}", f"{I1}.percent.{k}", get(i1, "percent", k), None if m is None else m * 100, IND, "NOTE_SP L112", dp=I1_DP)
    put_bool("S4.bounds_print_the_same", f"{I1}.routec_pool_bounds_print_the_same", get(i1, "routec_pool_bounds_print_the_same"),
             rhu(rc_all * 100, I1_DP) == rhu(rc_up * 100, I1_DP), IND, "REC_TEXT .footnote_uses")
    put_bool("S9.v83_percent_matches_printed", f'{I1}.v83_reproduces_printed["pool percent matches the printed v83"]',
             get(i1, "v83_reproduces_printed", "pool percent matches the printed v83"), rhu(v_all * 100, I1_DP) == I1_V83_PRINTED,
             IND, "NOTE_SP L11")
    put_flag("S10.pool", f"{I1}.flags.pool_v83_vs_routec", get(i1, "flags", "pool_v83_vs_routec"),
             rhu(rc_all * 100, I1_DP) == rhu(v_all * 100, I1_DP), IND, "NOTE_SP L112")
    for cell in cell_sh["routec"]:
        a, b = cell_sh["routec"][cell]["all_runs_denominator"], cell_sh["v83"][cell]["all_runs_denominator"]
        put_flag(f"S10.{cell}", f'{I1}.flags["cell {cell} v83_vs_routec"]', get(i1, "flags", f"cell {cell} v83_vs_routec"),
                 a is not None and b is not None and rhu(a * 100, I1_DP) == rhu(b * 100, I1_DP), IND, "NOTE_SP L112")

    res["F2.printed_decimals_percent"] = {"record_key": f"{I2}.printed_decimals_percent", "label": INT, "cite": "NOTE_SP L113",
                                          "equal": get(i2, "printed_decimals_percent") == I2_DP}
    p = {}
    for sc in ("v83", "routec"):
        c = payload["item2"][sc]
        if c is None:  # pilot runs in COMMON: route C not derivable (pre-stated F3)
            continue
        p[sc] = (Fraction(c["failed"], c["launched"]) * 100, Fraction(c["failed"] - c["rate_limit"], c["launched"]) * 100)
        fid = "F2" if sc == "v83" else "F3"
        put(f"{fid}.{sc}.percent", f"{I2}.{sc}.percent", get(i2, sc, "percent"), p[sc][0], INT, "RATES_DEF; NOTE_SP L12", dp=I2_DP)
        put(f"{fid}.{sc}.percent_without_rate_limit", f"{I2}.{sc}.percent_without_rate_limit",
            get(i2, sc, "percent_without_rate_limit"), p[sc][1], INT, "RATES_DEF; NOTE_SP L12", dp=I2_DP)
    put_bool("F6.percent", f'{I2}.v83_reproduces_printed["percent matches the printed v83"]',
             get(i2, "v83_reproduces_printed", "percent matches the printed v83"), rhu(p["v83"][0], I2_DP) == I2_V83_PRINTED,
             INT, "NOTE_SP L12")
    put_bool("F6.percent_without_rate_limit", f'{I2}.v83_reproduces_printed["percent_without_rate_limit matches the printed v83"]',
             get(i2, "v83_reproduces_printed", "percent_without_rate_limit matches the printed v83"),
             rhu(p["v83"][1], I2_DP) == I2_V83_PRINTED_WO, INT, "NOTE_SP L12")
    if "routec" in p:
        for j, name in enumerate(("with_rate_limit", "without_rate_limit")):
            put_flag(f"F7.{name}", f"{I2}.flags.{name}", get(i2, "flags", name),
                     rhu(p["routec"][j], I2_DP) == rhu(p["v83"][j], I2_DP), INT, "NOTE_SP L113")
    return out


@app.function(image=image, cpu=1.0, memory=1024, timeout=300, retries=0)
def compare_shares(record_bytes: bytes, payload_json: str) -> str:
    return json.dumps(compare_record_shares(record_bytes, json.loads(payload_json)))


# ---------------------------------------------------------------- laptop: orchestration

NOT_COVERED = [
    "every AUROC (item 5 / Table 7 rows 4-5, App E, record (i) run_level, call_level, catch, null, paired, bootstrap blocks)",
    "item 1 detector_cells.{routec,v83}_cells_agree and the per-cell v83_reproduces_printed booleans (no readable definition)",
    "item 2 chain_manifest_files_naming_the_pilot, rates_sim_manifests_naming_pilot (gates; would need a key-like id search)",
    "item 4 record_A_sizes_S_after_runs, record_A_sizes_equal_pool (need record A, a value record)",
    "gates, flag_counts (includes item 5), self_test, record (i) inputs.pool_hashes, skip.dropped_rows, row totals vs activations",
]


@app.local_entrypoint()
def main():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    exp = here.relative_to(repo)

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()

    head = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=all", "--", str(exp), f":(exclude){exp}/results"):
        raise SystemExit("STOP: uncommitted files in the experiment folder")
    sys.path.insert(0, str(here))
    import countscheck as cc

    ctx = cc.load_all(repo, here)  # every sha and jq extract before any remote call
    cc.derive_all(ctx, None)  # dry pass, discarded: every key-level STOP fires before any remote call; nothing printed
    files = [{"path": f["path"], "sha256": f["sha256"]} for f in ctx["pilot_manifest"]["files"]]
    if len(files) != N_PILOT:
        raise SystemExit("STOP: pilot manifest does not list 20 files")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    results = here / "results"

    bits = json.loads(read_pilot_headers.remote(json.dumps(files)))
    if [r["path"] for r in bits["rows"]] != [f["path"] for f in files] or not all(r["sha_ok"] for r in bits["rows"]):
        raise SystemExit("STOP: pilot header rows do not match the manifest")
    bits_rec = {"what": "COUNTSCHECK pilot header bits (header_only; one any-token bit per model name; no token counts)",
                "app": APP_NAME, "git": head, "manifest": dict(zip(("commit", "path", "sha256"), cc.PINS["pilot_manifest"])),
                "volume": PILOT_VOLUME, "root": f"{MOUNT}/{ROOT}", "rows": bits["rows"]}
    bits_path = results / f"pilot_header_bits_{stamp}.json"
    bits_bytes = (json.dumps(bits_rec, indent=1) + "\n").encode()
    with open(bits_path, "xb") as fh:
        fh.write(bits_bytes)

    it, payload = cc.derive_all(ctx, bits)
    cmp = json.loads(compare_shares.remote(ctx["simpool_bytes"], json.dumps(payload)))
    if not cmp["sha_ok"]:
        raise SystemExit("STOP: simpool record sha256 mismatch in the container")
    for iid, r in cmp["items"].items():
        it.rows.append({"id": iid, "record": "ii", "where": "modal", **r})

    result = {
        "what": "COUNTSCHECK: independent counts / shares check of honest_all_routec (i) and routec_simpool_numbers (ii). "
                "Per item: my derived count (laptop) or a container boolean (shares, percents), equal to the record or not, "
                "label INDEPENDENT / INTERNAL / CROSS-RECORD (09 condition 2: only INDEPENDENT items are an independent half; "
                "[^failed] stays on Aswin's list; printed AUROCs are not covered and stay PRELIMINARY).",
        "prestated": "experiments/2026-10-20_routec_counts_check/PRESTATED_OUTCOME_2026-10-20.md @ aef74386b (sha256 e65848ce...)",
        "git": head, "app": APP_NAME, "stamp_pt": stamp,
        "inputs": {n: dict(zip(("commit", "path", "sha256"), v)) for n, v in cc.PINS.items()},
        "filters_sha256": ctx["filter_sha256"],
        "jq": subprocess.run(["jq", "--version"], capture_output=True, text=True, check=True).stdout.strip(),
        "pilot_header_bits": {"path": str(bits_path.relative_to(repo)), "sha256": hashlib.sha256(bits_bytes).hexdigest()},
        "items": it.rows, "summary": cc.summary(it.rows), "not_covered": NOT_COVERED,
        "s7_note_sf_l28": ctx["s7_note_sf_l28"],  # PRESTATED Amendment 1: information only
    }
    out_path = results / f"routec_counts_check_{stamp}.json"
    out_bytes = (json.dumps(result, indent=1) + "\n").encode()
    with open(out_path, "xb") as fh:
        fh.write(out_bytes)
    for r in it.rows:
        mine = r.get("mine")
        shown = f" mine={mine}" if r["where"] == "laptop" and isinstance(mine, (int, str)) else ""
        print(f"item {r['id']} [{r['label']}] equal={r['equal']}{shown}"
              + (f" registered_ok={r['equal_registered']}" if "equal_registered" in r else ""))
    s = result["summary"]
    print(f"summary items={s['n_items']} equal={s['n_equal']} false={len(s['false_ids'])} "
          f"false_not_prestated={len(s['false_ids_not_prestated'])} "
          f"prestated_false_but_equal={len(s['prestated_false_ids_that_were_equal'])} "
          f"registered_false={len(s['registered_false_ids'])}")
    print(f"record {out_path.relative_to(repo)} sha256 {hashlib.sha256(out_bytes).hexdigest()}")
    print(f"bits {bits_path.relative_to(repo)} sha256 {hashlib.sha256(bits_bytes).hexdigest()}")
