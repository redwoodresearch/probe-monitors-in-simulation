"""Derived-slot clean-room half (fresh Claude agent, standing in for a1).

The six-probe mean of real_line (sim catch at the real threshold minus mean real catch), per readout C (commands) and R (tool
outputs), at joint mode and alarm level 0.01, re-derived from the clean-room engine's OWN regenerated per-draw series (option (3),
cleanroom_boot_opt3.py, record A), summarised with the record's summ rule, and compared with the producer half's stored value.

Rule (v2.7's derived-slot rule): "a derived slot is covered iff every input slot is covered by both halves (each half derives it
from its own run ...), then the halves intersect". This script is the clean-room half's derivation and its comparison.

Inputs
- The engine run's values directory on the clean-room Volume (mounted read-only): node_series.npz (members node000..node071, the
  engine's series for compare.json's nodes[i], draw 0 first) and compare.json ("nodes", "scale", "mine"); both sha-checked
  against the run summary's values_sha256_node_series / values_sha256_compare, on the exact bytes that are parsed.
- The producer half's results JSON, shipped in the image and sha-pinned; its keys results.six_probe["<C|R>|0.01"].real_line_mean
  = {point, ci95, draws, finite_draws}, in points (fraction x 100).

Node location: the 12 nodes are built as exact strings "pooled|<label> <C|R>|pooled_all_origins|joint|0.01|gap|real_line" for
the six labels below; each must occur exactly once in compare.json's nodes (whose sha must equal the pinned rlext node list),
and its npz member is node<index>. Cross-file check: each member's draw 0 must equal compare.json mine[node].point exactly,
and its finite count over draws 1..D must equal mine[node].finite.

Units: the engine's TPR is a weighted proportion ((V * caught).sum / V.sum in run_draws), so its series are FRACTIONS and
real_line = simB - real lies in [-1, 1] (checked here). compare.json's "scale" relates the record to the engine (record ~=
scale x engine; gap_compare, c0_identify). The producer half multiplies record A's stored FRACTIONS by 100, so only scale == "fraction"
is like-for-like (Codex d1); "points" STOPs (code 30). Points = 100 x engine fraction. The scale is
validated and reported with the record unit it implies. Comparison is in FRACTION units: producer_fraction = producer_points /
100; EXACT iff |mine - producer| <= 1e-9 x max(1, |producer_fraction|) (= 1e-9 fraction = 1e-7 points, since |x| <= 1), else
DIFF; |diff| is reported in both units. A readout is EXACT iff point, ci95 low and ci95 high are each EXACT and its draws and
finite_draws counts equal the producer's.

summ: the record's rule, a936a5d97:experiments/2026-10-15_fpr_sweep_507_prep/run_fprsweep_507.py l.113-120 (point = v[0];
f = finite v[1:]; withheld if len(f) < 0.95 D; ci95 = np.percentile(f, 2.5), np.percentile(f, 97.5), numpy defaults; D = 10000).

Run (after the engine's summary is in hand; one CPU container, about a minute):
  modal run run_derived_cleanroom_half.py --values-dir <values_dir> --series-sha <values_sha256_node_series> \
      --compare-sha <values_sha256_compare> --out-dir results
Prints one line, no values: DERIVED_OK <path> <sha256> C=<EXACT|DIFF> R=<EXACT|DIFF>.
Fixed exit codes (no message text): 10 bad arguments, 11 out-dir not a directory, 12 local producer file sha, 13 output exists;
20 values_dir form or sha argument form (container), 21 values files missing, 22 node_series sha, 23 compare.json sha,
24 producer file sha (container), 25 producer block shape, 30 compare.json shape or scale, 31 node list vs pin,
32 a node missing or duplicated, 33 npz member set, 34 series dtype or length, 35 non-finite draw 0, 36 value outside [-1, 1],
37 npz vs compare.json mine mismatch; 38 an input node not REPRODUCED EXACT in compare.json rows; 40 container script sha differs from the local file, 41 result shape.
"""

import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

import modal

LABEL = "derived-slot clean-room half (fresh Claude agent, standing in for a1)"
RULE = ("a derived slot is covered iff every input slot is covered by both halves (each half derives it from its own run ...), "
        "then the halves intersect")
APP = "probe-shift_aswin-surya_sub-realline_derived_kill-2026-11-25"
CLEANROOM_VOL = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
MOUNT = "/cr"  # the engine's /o maps to this read-only mount
PRODUCER = ("realline_boot_read_2026-10-25_180356.json", "cf86f3d916a77124eaf3d11ca8f67f65a9e805da74dbba14ffea6b7851b3aecc")
PRODUCER_REMOTE = f"/producer/{PRODUCER[0]}"
NODES_PIN = ("opt3/opt3_nodes_A_rlext.txt", "a5c12f43cfa9d2358ea189409701a2f6067f79870a524f08715b440495c1b5d7")
SUMM_SOURCE = "a936a5d97:experiments/2026-10-15_fpr_sweep_507_prep/run_fprsweep_507.py summ (l.113-120)"
SIX = ["sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48"]
READOUTS = ("C", "R")
LEVEL = "0.01"
D = 10000
RTOL = 1e-9
TO_POINTS = 100.0  # engine series are fractions
VALUES_DIR_RE = re.compile(r"/o/opt3_A_[0-9]{8}_[0-9]{6}_[0-9a-f]{8}")  # used with fullmatch
SHA_RE = re.compile(r"[0-9a-f]{64}")
HERE = Path(__file__).resolve().parent  # /root in the container; only used locally

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if modal.is_local():
    image = image.add_local_file(str(HERE / "results" / PRODUCER[0]), PRODUCER_REMOTE)


class Stop(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def need(ok, code):
    if not ok:
        raise Stop(code)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def node_id(label, rd):
    return f"pooled|{label} {rd}|pooled_all_origins|joint|{LEVEL}|gap|real_line"


def summ(v):  # the record's summ (SUMM_SOURCE), unchanged
    import numpy as np

    f = v[1:][np.isfinite(v[1:])]
    o = {"point": float(v[0]), "finite_draws": int(len(f)), "draws": D}
    if len(f) < 0.95 * D:
        return o | {"ci95": None, "withheld": "fewer than 95% complete draws"}
    return o | {"ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))], "degenerate": bool(np.ptp(f) == 0)}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def producer_block(prod, rd):
    """results.six_probe["<rd>|0.01"].real_line_mean, shape-checked: finite point, two finite ci95 ends, integral counts."""
    r = prod.get("results") if isinstance(prod, dict) else None
    sp = r.get("six_probe") if isinstance(r, dict) else None
    b = sp.get(f"{rd}|{LEVEL}") if isinstance(sp, dict) else None
    m = b.get("real_line_mean") if isinstance(b, dict) else None
    need(isinstance(m, dict), 25)
    ci = m.get("ci95")
    need(_num(m.get("point")) and isinstance(ci, list) and len(ci) == 2 and all(_num(x) for x in ci), 25)
    need(all(_num(m.get(k)) and float(m[k]).is_integer() for k in ("draws", "finite_draws")), 25)
    return m


def cmp_one(mine_fraction, producer_points):
    """|mine - producer| in fraction units against 1e-9 x max(1, |producer|)."""
    p = producer_points / TO_POINTS
    d = abs(mine_fraction - p)
    tol = RTOL * max(1.0, abs(p))
    return {"abs_diff_fraction": d, "abs_diff_points": d * TO_POINTS, "tol_fraction": tol, "label": "EXACT" if d <= tol else "DIFF"}


def _derive(values_dir, series_sha, compare_sha):
    import numpy as np

    need(VALUES_DIR_RE.fullmatch(values_dir) and SHA_RE.fullmatch(series_sha) and SHA_RE.fullmatch(compare_sha), 20)
    vdir = Path(MOUNT, values_dir[len("/o/"):])
    npz_p, cmp_p = vdir / "node_series.npz", vdir / "compare.json"
    need(vdir.is_dir() and npz_p.is_file() and cmp_p.is_file(), 21)
    nb, cb = npz_p.read_bytes(), cmp_p.read_bytes()  # hash exactly the bytes that are parsed
    need(sha(nb) == series_sha, 22)
    need(sha(cb) == compare_sha, 23)
    pb = Path(PRODUCER_REMOTE).read_bytes()
    need(sha(pb) == PRODUCER[1], 24)
    prod_json = json.loads(pb)
    prod = {rd: producer_block(prod_json, rd) for rd in READOUTS}

    comp = json.loads(cb)
    need(isinstance(comp, dict), 30)
    nodes, scale, eng = comp.get("nodes"), comp.get("scale"), comp.get("mine")
    need(isinstance(nodes, list) and all(isinstance(n, str) for n in nodes) and scale == "fraction", 30)  # Codex d1: the producer multiplies stored fractions by 100, so only "fraction" is like-for-like
    need(isinstance(eng, dict), 30)
    need(sha(("\n".join(nodes) + "\n").encode()) == NODES_PIN[1], 31)
    want = {rd: [node_id(lab, rd) for lab in SIX] for rd in READOUTS}
    idx = {}
    for rd in READOUTS:
        for n in want[rd]:
            need(nodes.count(n) == 1, 32)
            idx[n] = nodes.index(n)
    rows_ = comp.get("rows")  # Codex d1: each input node must itself be reproduced EXACT by the engine, or the mean is not covered
    need(isinstance(rows_, dict), 38)
    for n in idx:
        r_ = rows_.get(n)
        need(isinstance(r_, dict) and r_.get("verdict") == "REPRODUCED" and r_.get("label") == "EXACT", 38)
    with np.load(io.BytesIO(nb), allow_pickle=False) as z:
        need(sorted(z.files) == [f"node{i:03d}" for i in range(len(nodes))], 33)
        ser = {n: z[f"node{i:03d}"] for n, i in idx.items()}

    per_node = {}
    for n, v in ser.items():
        need(v.dtype == np.float64 and v.shape == (D + 1,), 34)
        need(bool(np.isfinite(v[0])), 35)
        fin = v[1:][np.isfinite(v[1:])]
        need(abs(float(v[0])) <= 1.0 + 1e-12 and bool(np.all(np.abs(fin) <= 1.0 + 1e-12)), 36)  # fraction-unit check
        e = eng.get(n)
        need(isinstance(e, dict) and e.get("point") == float(v[0]) and e.get("finite") == len(fin) and e.get("draws") == D, 37)
        per_node[n] = {"npz_member": f"node{idx[n]:03d}", "draw0_engine_fraction": float(v[0]),
                       "draw0_points": float(v[0]) * TO_POINTS, "length": int(v.shape[0]), "finite_draws_1_to_D": int(len(fin))}

    readouts = {}
    for rd in READOUTS:
        m = np.stack([ser[n] for n in want[rd]]).mean(axis=0)  # equal weight, draw by draw; NaN in any probe -> NaN
        s, p = summ(m), prod[rd]
        rows = {"point": cmp_one(s["point"], p["point"])}
        if s["ci95"] is None:
            rows |= {k: {"label": "DIFF", "reason": "ci95 withheld on the clean-room half"} for k in ("ci95_lo", "ci95_hi")}
        else:
            rows |= {"ci95_lo": cmp_one(s["ci95"][0], p["ci95"][0]), "ci95_hi": cmp_one(s["ci95"][1], p["ci95"][1])}
        counts_equal = s["draws"] == int(p["draws"]) and s["finite_draws"] == int(p["finite_draws"])
        ok = counts_equal and all(r["label"] == "EXACT" for r in rows.values())
        readouts[rd] = {"producer_key": f'results.six_probe["{rd}|{LEVEL}"].real_line_mean', "nodes": want[rd],
                        "comparisons": rows, "draws": s["draws"], "finite_draws": s["finite_draws"], "counts_equal": counts_equal,
                        "withheld": s["ci95"] is None, "degenerate": s.get("degenerate"), "label": "EXACT" if ok else "DIFF"}

    return {"label": LABEL, "rule": RULE, "summ_source": SUMM_SOURCE, "D": D,
            "inputs": {"values_dir": values_dir, "values_sha256_node_series": series_sha, "values_sha256_compare": compare_sha,
                       "producer_file": PRODUCER[0], "producer_sha256": PRODUCER[1], "node_list_pin": list(NODES_PIN)},
            "units": {"engine": "fraction (weighted TPR proportion)", "scale": scale,
                      "record_unit_implied": scale, "record_over_engine": 1.0 if scale == "fraction" else 100.0,
                      "to_points": TO_POINTS, "compare_in": "fraction", "tolerance": "1e-9 x max(1, |producer_fraction|)"},
            "readouts": readouts, "per_node": per_node,
            "container": {"numpy": np.__version__, "python": sys.version.split()[0],
                          "script_sha256": sha(Path(__file__).read_bytes())}}


@app.function(image=image, volumes={MOUNT: modal.Volume.from_name(CLEANROOM_VOL, version=2).read_only()}, cpu=(1.0, 1.0),
              memory=(4096, 4096), timeout=600, startup_timeout=300, scaledown_window=10, retries=0)
def derive(values_dir: str, series_sha: str, compare_sha: str) -> dict:
    try:
        return _derive(values_dir, series_sha, compare_sha)
    except Stop as e:  # only this script's own fixed-code checks; anything else propagates
        return {"stop": e.code}


@app.local_entrypoint()
def main(values_dir: str, series_sha: str, compare_sha: str, out_dir: str):
    import subprocess
    from datetime import datetime
    from zoneinfo import ZoneInfo

    pt = ZoneInfo("America/Los_Angeles")
    if not (VALUES_DIR_RE.fullmatch(values_dir) and SHA_RE.fullmatch(series_sha) and SHA_RE.fullmatch(compare_sha)):
        raise SystemExit(10)
    od = Path(out_dir).resolve()
    if not od.is_dir():
        raise SystemExit(11)
    if sha((HERE / "results" / PRODUCER[0]).read_bytes()) != PRODUCER[1]:
        raise SystemExit(12)
    script_sha = sha(Path(__file__).read_bytes())
    git = ["git", "-C", str(HERE)]
    head = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    st = subprocess.run(git + ["status", "--porcelain", "--", Path(__file__).name], check=True, capture_output=True, text=True)
    started = datetime.now(pt).isoformat(timespec="seconds")

    res = derive.remote(values_dir, series_sha, compare_sha)
    if "stop" in res:
        raise SystemExit(int(res["stop"]))
    if res.get("container", {}).get("script_sha256") != script_sha:
        raise SystemExit(40)
    rds = res.get("readouts")
    if not (isinstance(rds, dict) and set(rds) == set(READOUTS) and all(rds[rd].get("label") in ("EXACT", "DIFF") for rd in READOUTS)):
        raise SystemExit(41)

    now = datetime.now(pt)
    res["local"] = {"script": Path(__file__).name, "script_sha256": script_sha, "git_head": head,
                    "git_status_script": st.stdout.strip(), "started_pt": started, "finished_pt": now.isoformat(timespec="seconds"),
                    "argv": {"values_dir": values_dir, "series_sha": series_sha, "compare_sha": compare_sha, "out_dir": str(od)}}
    path = od / f"derived_cleanroom_half_{now.strftime('%Y-%m-%d_%H%M%S')}.json"
    if path.exists():
        raise SystemExit(13)
    body = (json.dumps(res, indent=2, allow_nan=False) + "\n").encode()
    with open(path, "xb") as fh:  # create-only
        fh.write(body)
    print(f"DERIVED_OK {path} {sha(body)} C={rds['C']['label']} R={rds['R']['label']}")
