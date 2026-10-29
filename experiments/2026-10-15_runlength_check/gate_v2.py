"""Run-length v2 gate and comparisons (reads records only). (1) Gate: the --cap paired records' BASE catches and gaps (per origin and pooled,
joint and fixed, every level) equal pooled455v2_C_2026-10-15_160734 / R_160736 exactly (NaN == NaN); readable equal; draw-0 pools_at_draw0
within 1e-12 (float noise; per-run alarm shares reported, not gated). (2) Draw-0 check vs aswin/runlength-2e @ 0ca2f94a2: S_old per-origin rows of
the 7 overlapping probes under K = 30 and per-task min (S_old inputs unchanged), points and ci95. (3) Prediction scores V1-V5 (RULE v2 extension).

  python3 -I experiments/2026-10-15_runlength_check/gate_v2.py <paired_C> <paired_R> <k30_C> <k30_R> <taskmin_C> <taskmin_R>
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = {"C": ("pooled455v2_C_2026-10-15_160734.json", "63871423fd878f62"), "R": ("pooled455v2_R_2026-10-15_160736.json", "3f36aab2c8820fd2")}
OLD = {"30": "experiments/2026-10-15_runlength_check/results/runlen_30_2026-10-15_161258.json",
       "taskmin": "experiments/2026-10-15_runlength_check/results/runlen_taskmin_2026-10-15_161258.json"}


def load(p):
    b = Path(p).read_bytes()
    return json.loads(b), hashlib.sha256(b).hexdigest()


def same(x, y, tol=0.0):
    if isinstance(y, dict):
        return isinstance(x, dict) and all(k in x and same(x[k], v, tol) for k, v in y.items())
    if isinstance(y, list):
        return isinstance(x, list) and len(x) == len(y) and all(same(a, b, tol) for a, b in zip(x, y))
    if isinstance(y, float) and isinstance(x, float):
        if y != y and x != x:
            return True
        return x == y if tol == 0 else abs(x - y) <= tol
    return x == y


def main(paths):
    P = dict(zip(("paired_C", "paired_R", "30_C", "30_R", "taskmin_C", "taskmin_R"), paths))
    recs, shas = {}, {}
    for k, p in P.items():
        recs[k], shas[k] = load(p)
    out = {"records": {k: {"file": Path(p).name, "sha256": shas[k]} for k, p in P.items()}, "gate": {}, "old_check": {}, "predictions": {}}
    # (1) gate
    for r_ in ("C", "R"):
        ref, h = load(HERE.parent / "2026-10-15_fpr_sweep_507_prep" / "results" / REF[r_][0])
        if not h.startswith(REF[r_][1]):
            sys.exit(f"reference sha {REF[r_][0]}")
        a = recs[f"paired_{r_}"]
        g = {"compared": 0, "mismatch": [], "diag_mismatch": []}
        for lb in ref["rows"]:
            for o in ref["rows"][lb]:
                ao, bo = a["rows"][lb][o], ref["rows"][lb][o]
                if "points" not in bo:
                    continue
                for mode in bo["points"]:
                    for f in bo["points"][mode]:
                        for kind in ("catch", "gap"):
                            for k, y in bo["points"][mode][f][kind].items():
                                g["compared"] += 1
                                if not same(ao["points"][mode][f][kind].get(k), y):
                                    g["mismatch"].append(f"{lb}|{o}|{mode}|{f}|{kind}|{k}")
                g["compared"] += 1
                if not same(ao.get("readable"), bo.get("readable")):
                    g["mismatch"].append(f"{lb}|{o}|readable")
                if not same(ao.get("pools_at_draw0"), bo.get("pools_at_draw0"), 1e-12):
                    g["diag_mismatch"].append(f"{lb}|{o}|pools_at_draw0")
            for mode in ref["pooled"][lb]["pooled"]:
                for f in ref["pooled"][lb]["pooled"][mode]:
                    for kind in ("catch", "gap"):
                        for k, y in ref["pooled"][lb]["pooled"][mode][f][kind].items():
                            g["compared"] += 1
                            if not same(a["pooled"][lb]["pooled"][mode][f][kind].get(k), y):
                                g["mismatch"].append(f"{lb}|pooled|{mode}|{f}|{kind}|{k}")
        g["pass"] = not g["mismatch"]
        out["gate"][r_] = g
    # (2) draw-0 check vs the s1-s3 records on S_old rows (7 overlapping probes)
    git = lambda *a_: subprocess.run(["git", *a_], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    for v in ("30", "taskmin"):
        old = json.loads(git("show", f"0ca2f94a2:{OLD[v]}"))
        chk = {"compared": 0, "mismatch": []}
        for lb in old["rows"]:
            r_ = lb.rsplit(" ", 1)[1]
            a, b = recs[f"{v}_{r_}"]["rows"][lb]["S_old"]["points"]["joint"], old["rows"][lb]["S_old"]["points"]["joint"]
            for f in ("0.01", "0.05"):
                for kind in ("catch", "gap"):
                    for k, y in b[f][kind].items():
                        if k not in a[f][kind]:
                            continue
                        chk["compared"] += 1
                        if not same(a[f][kind][k], y):
                            chk["mismatch"].append(f"{lb}|S_old|{f}|{kind}|{k}")
        out["old_check"][v] = chk
    # (3) predictions V1-V5 (1e-2)
    pt = lambda s: None if s is None or s.get("point") is None else 100 * s["point"]  # noqa: E731
    ci = lambda s: None if s is None or not s.get("ci95") else [100 * x for x in s["ci95"]]  # noqa: E731
    pr = recs["paired_R"]["rows"]
    v12 = {}
    for lb in ("FIBS R", "harmfulness R"):
        q = pr[lb]["S_new"]["points"]
        v12[lb] = {"uncapped_delta_full_joint": [pt(q["joint"]["0.01"]["gap"]["delta_full"]), ci(q["joint"]["0.01"]["gap"]["delta_full"])],
                   "paired_delta_full_fixed": [pt(q["fixed"]["0.01"]["gap"]["delta_full_paired"]), ci(q["fixed"]["0.01"]["gap"]["delta_full_paired"])],
                   "paired_delta_full_joint": [pt(q["joint"]["0.01"]["gap"]["delta_full_paired"]), ci(q["joint"]["0.01"]["gap"]["delta_full_paired"])]}
    out["predictions"]["V1_FIBS_R_S_new"] = v12["FIBS R"] | {"predicted": "+4 ± 4; joint interval includes 0 (P = 0.7)"}
    out["predictions"]["V2_harmfulness_R_S_new"] = v12["harmfulness R"] | {"predicted": "+3 ± 4; joint interval includes 0 (P = 0.65)"}
    ch1, ch2, ch30, flips, solc = [], [], [], [], []
    for r_ in ("C", "R"):
        P_, K_ = recs[f"paired_{r_}"]["pooled"], recs[f"30_{r_}"]["pooled"]
        for lb in P_:
            fx = P_[lb]["pooled"]["fixed"]["0.01"]["gap"]
            ch1.append(pt(fx["sim_minus_run1_paired"]) - pt(fx["sim_minus_run1"]))
            ch2.append(pt(fx["sim_minus_run2_paired"]) - pt(fx["sim_minus_run2"]))
            u = P_[lb]["pooled"]["joint"]["0.01"]["gap"]["delta_full"]
            k = K_[lb]["pooled"]["joint"]["0.01"]["gap"]["delta_full"]
            ch30.append(pt(k) - pt(u))
            ex = lambda s: s["ci95"][0] > 0 or s["ci95"][1] < 0  # noqa: E731
            if ex(u) != ex(k):
                flips.append(lb)
            if lb.startswith(("Sol (a)", "c_asis")):
                solc.append({"row": lb, "change_sim_minus_run1": ch1[-1], "change_sim_minus_run2": ch2[-1]})
    m = lambda x: sum(x) / len(x)  # noqa: E731
    out["predictions"]["V3_mean_paired_change_18_rows_fixed"] = {"sim_minus_run1": m(ch1), "sim_minus_run2": m(ch2), "predicted": "-1.5 ± 1.0 and -2.0 ± 1.5", "n": len(ch1)}
    out["predictions"]["V4_k30_mean_change_delta_full"] = {"value": m(ch30), "zero_exclusion_changes": flips, "predicted": "-0.5 ± 1.5; no changes (P = 0.7)"}
    out["predictions"]["V5_Sol_c_asis_paired_change"] = {"rows": solc, "predicted": "-1.5 ± 2"}
    stamp = Path(P["paired_C"]).stem.rsplit("_", 2)[-2] + "_" + Path(P["paired_C"]).stem.rsplit("_", 1)[-1]
    dest = HERE / "results" / f"runlen_v2_gate_scores_{stamp}.json"
    b = json.dumps(out, indent=1).encode()
    dest.write_bytes(b)
    print(json.dumps({"file": dest.name, "sha256": hashlib.sha256(b).hexdigest()}))
    print(json.dumps({"gate": {r_: {k: (v if k != "mismatch" else v[:10]) for k, v in g.items()} for r_, g in out["gate"].items()},
                      "old_check": {v: {"compared": c["compared"], "mismatch": c["mismatch"][:10]} for v, c in out["old_check"].items()},
                      "predictions": out["predictions"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:7])
