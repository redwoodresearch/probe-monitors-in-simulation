"""Clean-room re-derivation of row 92's [^sim] "in about 6% of simulated honest runs" on route C's pool (addendum 19, a1, 2026-10-20).

Definitions (registered; no production code read):
  pool       = the catalogue's role-S runs (rcCAT 72432318; "Route C's pool: the role-S runs of the rcCAT catalogue"), n = 868.
  run -> path: honest_all_routec @ 596bc767e .run_keys, sim-class key with the run's cell and epoch, "#e<n>" removed (09's fallback (b):
               a SHARED INPUT with the first re-derivation, disclosed; STOP on 0 or 2+ matches).
  copies     = every row for that path in scan_rows (parts joined by "/"), usage (.runs[].path), cont_headers (.rows[].path) and the
               20 pilot .eval headers read here (inspect header_only; model names with one any-token bit; nothing else).
  per copy   (NOTE_detector_solfree v2 L13-18): no header / empty or missing model list -> INCOMPLETE; Sol (openai/gpt-5.6-sol) tokens > 0
               -> SOL; lacks tokens > 0 of openai/gpt-6-astra or vllm/Qwen/Qwen3.5-27B -> INCOMPLETE; else FREE.
  per path   (L19): SOL and FREE together -> STOP; any SOL -> SOL; any INCOMPLETE -> UNKNOWN; no copy -> UNKNOWN (simpool NOTE L29-30);
               else FREE. UNKNOWN stays in the denominator, never Sol-free (L34-35).
  percent    = 100 x sol_using / n, printed to 0 dp (simpool NOTE L112), half-up.
Compared with routec_simpool_numbers_2026-10-20_020501.json (2e17de0c) .items["1_sim_footnote"] inside the container; the laptop gets
booleans and counts only (n, unknown, paths with copies, gates). The full values go to the output Volume.

  SP_CFG=<folder of the 7 pinned inputs> uv run --no-project --python 3.12 --with modal==1.6.1 modal run cleanroom_simpool.py
"""

import json
import os
import time
from pathlib import Path

import modal

OUT = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
PILOT_VOL = "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"
PINS = {"cat.json": "72432318a3703d4675a5c4530d77f598a003af351340d895d305b9bd1050adc8",
        "honest_all_routec.json": "10ed5645e0a8357150e07ac6416a06c6196e995d01eb5088b94de1d554d73c46",
        "scan_rows.json": "0d0392fb773b1804a3ded3b4f3fd2f4dc8e7ae33b1ee660dd7200d0550410197",
        "usage.json": "a0bcb2cef1ec21d598dda1795cc8b027eaf49235ba191f5d4c9643e108701102",
        "cont_headers.json": "a7638c2d19f6a56f2953babca0ea98a7fd63854cbec21be6f1861ea452941e5d",
        "pilot_manifest.json": "875167b9916560444f80be78bf516beba6711837a811bd677b82f66e704d182b",
        "simpool_record.json": "2e17de0c852612d8a2b1d83468997c0cb6f787dcfee9a3c93eef6edee711fd11"}
SOL, ASTRA, QWEN = "openai/gpt-5.6-sol", "openai/gpt-6-astra", "vllm/Qwen/Qwen3.5-27B"
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "input_tokens_cache_write", "input_tokens_cache_read", "reasoning_tokens")
SRC = {"scan_rows": "scan_rows", "usage": "usage", "cont_headers": "header_scan", "pilot_header": "header_scan"}

app = modal.App(OUT)
OUT_VOL = modal.Volume.from_name(OUT, version=2)
PVOL = modal.Volume.from_name(PILOT_VOL)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("inspect-ai==0.3.276")
if modal.is_local():
    for f in PINS:
        image = image.add_local_file(str(Path(os.environ["SP_CFG"]) / f), f"/sp/{f}")


def bit(usage, m):
    """Any token count > 0 for model m (a dict of token fields, or a bare bit)."""
    v = usage.get(m)
    if isinstance(v, bool):
        return v
    return isinstance(v, dict) and any(isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in v.values())


def copy_verdict(usage, has_header=True):
    if not has_header or not isinstance(usage, dict) or not usage:
        return "INCOMPLETE"
    if bit(usage, SOL):
        return "SOL"
    if not (bit(usage, ASTRA) and bit(usage, QWEN)):
        return "INCOMPLETE"
    return "FREE"


def path_verdict(vs):
    if "SOL" in vs and "FREE" in vs:
        return "STOP"
    if "SOL" in vs:
        return "SOL"
    if not vs or "INCOMPLETE" in vs:
        return "UNKNOWN"
    return "FREE"


@app.function(image=image, cpu=1, memory=4096, timeout=900, retries=0, volumes={"/o": OUT_VOL, "/a25": PVOL.read_only()})
def derive(stamp: str) -> dict:
    import hashlib
    from decimal import ROUND_HALF_UP, Decimal

    from inspect_ai.log import read_eval_log

    d = {}
    for f, want in PINS.items():
        raw = Path("/sp", f).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == want, f"input sha {f}"
        d[f] = json.loads(raw)
    S = {k: v for k, v in d["cat.json"]["runs"].items() if "S" in v["roles"]}
    sim = [(k.split("|")[1], k.split("|")[2], v.split("#")[0]) for k, v in d["honest_all_routec.json"]["run_keys"].items() if k.startswith("sim|")]
    stops, path_of = [], {}
    for k, v in S.items():
        cell = k.split("|", 1)[1]
        m = [p for c, e, p in sim if c == cell and e == v["epoch"]]
        if len(m) != 1:
            stops.append(f"map {k}: {len(m)} matches")
        else:
            path_of[k] = m[0]
    copies = {p: [] for p in path_of.values()}
    for r in d["scan_rows.json"]["rows"]:
        p = "/".join(r["parts"])
        if p in copies:
            copies[p].append(("scan_rows", copy_verdict(r.get("model_usage"))))
    for r in d["usage.json"]["runs"]:
        if r["path"] in copies:
            copies[r["path"]].append(("usage", copy_verdict(r.get("usage"))))
    for r in d["cont_headers.json"]["rows"]:
        if r["path"] in copies:
            copies[r["path"]].append(("cont_headers", copy_verdict(r.get("models_any_tokens"), bool(r.get("has_header")))))
    man = {f["path"]: f["sha256"] for f in d["pilot_manifest.json"]["files"]}
    pilot = {"files": len(man), "sha_ok": 0, "header_read": 0, "in_pool": sum(1 for p in man if p in copies)}
    unreadable = []
    for p, want in man.items():
        fp = Path("/a25/simh_evals_v1", p)
        ok = fp.exists() and hashlib.sha256(fp.read_bytes()).hexdigest() == want
        pilot["sha_ok"] += ok
        assert ok, f"pilot file sha or presence: {p}"
        try:
            h = read_eval_log(str(fp), header_only=True)
            mu = {m: any((getattr(u, f, None) or 0) > 0 for f in TOKEN_FIELDS)
                  for m, u in ((h.stats.model_usage or {}) if h.stats else {}).items()}
            pilot["header_read"] += 1
            v = copy_verdict(mu, True)
        except Exception as ex:  # registered: a header that can't be read is a copy without a header (INCOMPLETE -> UNKNOWN), named
            unreadable.append(f"{p}: {type(ex).__name__}")
            v = "INCOMPLETE"
        if p in copies:
            copies[p].append(("pilot_header", v))
    verdict = {k: path_verdict([v for _, v in copies[p]]) for k, p in path_of.items()}
    stops += [f"path {k}: SOL and FREE copies" for k, v in verdict.items() if v == "STOP"]
    n = len(S)
    cnt = {c: sum(1 for v in verdict.values() if v == c) for c in ("SOL", "FREE", "UNKNOWN")}
    pct = 100 * cnt["SOL"] / n
    p0 = int(Decimal(str(pct)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    nsrc = {}
    for p in path_of.values():
        k_ = len({SRC[s] for s, _ in copies[p]})  # distinct sources: scan_rows, usage, header scan (cont_headers or pilot headers)
        nsrc[str(k_)] = nsrc.get(str(k_), 0) + 1
    rec = d["simpool_record.json"]["items"]["1_sim_footnote"]
    rp = rec["pool"]["routec"]
    rpct = rec["percent"]["routec_pool"]
    eq = {"n": n == rp["n"], "sol_using": cnt["SOL"] == rp["sol_using"], "sol_free": cnt["FREE"] == rp["sol_free"],
          "unknown": cnt["UNKNOWN"] == rp["unknown"], "percent_exact": abs(pct - rpct) <= 1e-9 * max(1.0, abs(rpct)),
          "percent_0dp": p0 == int(Decimal(str(rpct)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)),
          "n_paths_by_number_of_sources": nsrc == {str(k): v for k, v in (rec["n_paths_by_number_of_sources"].get("routec") or {}).items()}}
    values = {"n": n, "counts": cnt, "percent": pct, "percent_0dp": p0, "n_paths_by_number_of_sources": nsrc,
              "record": {"pool_routec": rp, "percent_routec_pool": rpct}, "stops": stops, "pilot_header_unreadable": unreadable}
    vpath = Path("/o", f"cleanroom_simpool_values_{stamp}.json")
    with vpath.open("x") as fh:
        json.dump(values, fh)
    OUT_VOL.commit()
    return {"stamp": stamp, "n": n, "n_mapped": len(path_of), "n_unknown": cnt["UNKNOWN"],
            "n_paths_with_copies": sum(1 for p in path_of.values() if copies[p]), "pilot": pilot, "pilot_header_unreadable": unreadable, "n_stops": len(stops), "stops": stops[:20],
            "equal": eq, "values_file": {"volume": OUT, "path": vpath.name},
            "verdict_OK": not stops and all(eq[k] for k in ("n", "sol_using", "sol_free", "unknown", "percent_0dp"))}


@app.local_entrypoint()
def main():
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out = derive.remote(stamp)
    path = Path(__file__).parent / "results" / f"cleanroom_simpool_{stamp}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("n", "n_unknown", "n_paths_with_copies", "n_stops", "equal", "verdict_OK")}))
    print("wrote", path)
