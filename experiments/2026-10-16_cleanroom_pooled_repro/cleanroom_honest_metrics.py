"""Clean-room 4(a) (addendum 22, a1, 2026-10-20): the honest-sim detector's printed run-level AUROCs, recomputed from the record's
STORED held-out run scores (checks everything after the fit, not the fit; 4(b) refits). Printed (provenance v84 / v17 by key_path):
  rows 75-78: honest_all_routec cells.{reply|L48, command|L48}.problems.simh_vs_b.run_level.auroc (Table 7 row 4, App E);
  rows 95-98: routec_simpool_numbers items.5_table7_row5_control.run_level_auroc.routec.{reply|L48, command|L48} (Table 7 row 5).
Definitions: honest_sim_detector_all RULE @ 596bc767e L40-55 (run score = mean of held-out row logits; 5 folds by main) and the record's
direction text: class 1 = simulated honest (simh_vs_b) or real honest A (a_vs_b), class 0 = real honest B, class 1 scored higher.
AUROC = P(class-1 score > class-0 score) + 0.5 P(tie) over the held-out run scores (Mann-Whitney), all runs pooled; per fold beside.
The laptop gets booleans and counts only; values go to the output Volume.

  HM_CFG=<folder with the 2 records> uv run --no-project --python 3.12 --with modal==1.6.1 modal run cleanroom_honest_metrics.py
"""

import json
import os
import time
from pathlib import Path

import modal

OUT = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
PINS = {"honest_all_routec.json": "10ed5645e0a8357150e07ac6416a06c6196e995d01eb5088b94de1d554d73c46",
        "simpool_record.json": "2e17de0c852612d8a2b1d83468997c0cb6f787dcfee9a3c93eef6edee711fd11"}
CELLS = ("command|L48", "command|L63", "reply|L48", "reply|L63")
PROBLEMS = {"simh_vs_b": "sim", "a_vs_b": "A"}  # class-1 run-key prefix; class 0 = "B"
PRINTED = [("honest_all", c, "simh_vs_b") for c in ("reply|L48", "command|L48")]
RTOL = 1e-12
SCOPE_LABEL = "AUROC recomputed from the producer's stored held-out scores; the fit is not re-done"  # 22b (09; Codex P2)

app = modal.App(OUT)
OUT_VOL = modal.Volume.from_name(OUT, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if modal.is_local():
    for f in PINS:
        image = image.add_local_file(str(Path(os.environ["HM_CFG"]) / f), f"/hm/{f}")


def auroc(pos, neg):
    """Mann-Whitney AUROC with ties counted half, by ranks (average ranks for ties)."""
    import numpy as np

    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


@app.function(image=image, cpu=1, memory=2048, timeout=600, retries=0, volumes={"/o": OUT_VOL})
def run(stamp: str) -> dict:
    import hashlib
    import math

    import numpy as np

    raw = {f: Path("/hm", f).read_bytes() for f in PINS}
    for f, want in PINS.items():
        assert hashlib.sha256(raw[f]).hexdigest() == want, f"input sha {f}"
    H, SP = json.loads(raw["honest_all_routec.json"]), json.loads(raw["simpool_record.json"])
    folds = H.get("run_folds") or {}
    rows, stops, ctl = [], [], {}

    def num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    # control: my AUROC on a synthetic case with a known answer (ties included)
    ctl["synthetic_auroc"] = abs(auroc(np.array([3.0, 2.0, 2.0]), np.array([2.0, 1.0])) - (5.0 / 6.0)) < 1e-15  # 22b (R1 P0): 5 of 6 pairs
    mine = {}
    for c in CELLS:
        for prob, pfx in PROBLEMS.items():
            node = ((H["cells"].get(c) or {}).get("problems") or {}).get(prob) or {}
            rs = node.get("run_scores") or {}
            pos = np.array([v for k, v in rs.items() if k.split("|", 1)[0] == pfx], dtype=float)
            neg = np.array([v for k, v in rs.items() if k.split("|", 1)[0] == "B"], dtype=float)
            other = sum(1 for k in rs if k.split("|", 1)[0] not in (pfx, "B"))
            if other or not len(pos) or not len(neg) or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(neg)):
                stops.append(f"{c} {prob}: run_scores prefixes / finiteness (other {other}, pos {len(pos)}, neg {len(neg)})")
                continue
            a = auroc(pos, neg)
            mine[(c, prob)] = a
            rl = node.get("run_level") or {}
            nr = node.get("n_runs") or {}
            rows.append({"key": f"{c}|{prob}|n_runs.class1", "match": nr.get("class1") == len(pos), "verdict": True})
            rows.append({"key": f"{c}|{prob}|n_runs.class0", "match": nr.get("class0") == len(neg), "verdict": True})
            ref = rl.get("auroc")
            rows.append({"key": f"{c}|{prob}|run_level.auroc", "match": num(ref) and abs(a - ref) <= RTOL * max(1.0, abs(ref)), "verdict": True,
                         "mine": a, "ref": ref})
            pf = rl.get("auroc_per_fold") or {}
            fk = {k: str(folds.get(k)) for k in rs}
            for f_ in sorted(pf):
                pp = np.array([v for k, v in rs.items() if k.split("|", 1)[0] == pfx and fk[k] == str(f_)], dtype=float)
                nn = np.array([v for k, v in rs.items() if k.split("|", 1)[0] == "B" and fk[k] == str(f_)], dtype=float)
                af = auroc(pp, nn) if len(pp) and len(nn) else None
                rows.append({"key": f"{c}|{prob}|run_level.auroc_per_fold.{f_}", "verdict": False, "mine": af, "ref": pf[f_],
                             "match": af is not None and num(pf[f_]) and abs(af - pf[f_]) <= RTOL * max(1.0, abs(pf[f_]))})
    # Table 7 row 5: simpool's route C control AUROC per cell, identified against my AUROCs by problem (pre-stated: a_vs_b)
    sp = ((SP.get("items") or {}).get("5_table7_row5_control") or {}).get("run_level_auroc", {}).get("routec") or {}
    for c in ("reply|L48", "command|L48"):
        ref = sp.get(c)
        for prob in PROBLEMS:
            m_ = mine.get((c, prob))
            rows.append({"key": f"simpool.row5.{c}|{prob}", "verdict": prob == "a_vs_b", "mine": m_, "ref": ref,
                         "match": m_ is not None and num(ref) and abs(m_ - ref) <= RTOL * max(1.0, abs(ref))})
    vr = [r for r in rows if r["verdict"]]
    printed = [r for r in rows if r["key"] in {f"{c}|{p}|run_level.auroc" for _, c, p in PRINTED} or r["key"].startswith("simpool.row5.") and r["verdict"]]
    vals = {"scope_label": SCOPE_LABEL, "rows": rows, "controls": ctl, "stops": stops}
    vb = json.dumps(vals).encode()
    vpath = Path("/o", f"cleanroom_honest_metrics_values_{stamp}.json")
    with vpath.open("xb") as fh:
        fh.write(vb)
    OUT_VOL.commit()
    return {"scope_label": SCOPE_LABEL, "stamp": stamp, "n_stops": len(stops), "stops": stops, "controls": ctl, "n_rows": len(rows), "n_verdict": len(vr),
            "n_verdict_match": sum(bool(r["match"]) for r in vr), "failing_verdict_keys": [r["key"] for r in vr if not r["match"]],
            "diag_per_fold": {"n": sum(1 for r in rows if not r["verdict"] and ".auroc_per_fold." in r["key"]),
                              "n_match": sum(1 for r in rows if not r["verdict"] and ".auroc_per_fold." in r["key"] and r["match"])},
            "simpool_row5_simh_reading_match": [r["key"] for r in rows if r["key"].startswith("simpool.row5.") and not r["verdict"] and r["match"]],
            "printed": {"n": len(printed), "n_match": sum(bool(r["match"]) for r in printed)},
            "values_file": {"volume": OUT, "path": vpath.name, "sha256": hashlib.sha256(vb).hexdigest()},
            "PASS": not stops and all(ctl.values()) and all(bool(r["match"]) for r in vr)}


@app.local_entrypoint()
def main():
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out = run.remote(stamp)
    path = Path(__file__).parent / "results" / f"cleanroom_honest_metrics_{stamp}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("scope_label", "n_stops", "n_verdict", "n_verdict_match", "printed", "PASS")}))
    print("wrote", path)
