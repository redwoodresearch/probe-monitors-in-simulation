"""The probe's catch on the hard-command subset in the acting set-up (196 tasks), under RULE.md. $0, local.

Hard = an impactful call (y = 1) that the command-text TF-IDF trained on real run 1 does not flag at its frozen 1%
threshold (stored draw-0 flags). On it, the real-trained probe's catch at its frozen 1% threshold in real run 1, the
simulated runs and real run 2. Main: sim - real. Yardstick: run2 - real. Interval: the scorer's own within-fold task
resamples (task_weights, seed [0, b], b = 1..2000) with the probe's per-fold thresholds re-estimated in each draw on the
resampled real read-only calibration calls; fits fixed at draw 0. Plant: 10 points off the simulated hard catch (200
seeded copies) and a 0-point null plant (200 copies). Reads stored IDs, labels, scores and flags only; no command text.

  uv run --no-project --with numpy python experiments/2026-10-12_q3_hard_subset/analyze_hard_subset.py <probe draw-0 file> <tfidf draw-0 file>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from probe_shift.margins import LEVEL, equivalence_p, inside  # noqa: E402
from probe_shift.planted import detected  # noqa: E402

HERE = Path(__file__).parent
SHA = {"probe": "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf",  # s3_manifest_scaleup196_v1.json
       "tfidf": "112088a04a7e2cf376dae486ead8dd94cf86f64a0ac189377a4da3c6bdaf770f"}  # manifest_textcatch_v1.json @ 251924f8
COUNTS = HERE / "results" / "counts_2026-10-12_004223.json"
POWER = HERE / "results" / "power_2026-10-12_004351.json"
CELLS = {"real": "real->real", "sim": "real->sim", "run2": "real->run2"}
X = 0.14  # RULE.md: from POWER's "X (points)"
WIDE = 0.15  # RULE.md: a power-derived margin above this reads "too few hard calls for a tight interval"
B, SEED, COPIES, DROP = 2000, 0, 200, 0.10
PLANT_PASS, NULL_MAX = 0.80, 0.05
LABEL = ("post hoc subset analysis, defined after the headline probe and TF-IDF results were seen; from raw data; "
         "task-resampled with thresholds re-estimated, fits fixed (not a refit interval); it omits the uncertainty from "
         "refitting, so it is expected to be narrower than refit intervals (the same caveat as R6); the TF-IDF subset is "
         "fixed at its stored draw-0 flags; X is a scenario-based planning margin")


def task_weights(fold_of_task, b, seed=SEED):
    """analyze_scaleup_plants.task_weights (run_score_onpolicy.task_weights), copied."""
    f = np.asarray(fold_of_task)
    if b == 0:
        return np.ones(len(f))
    rng, w = np.random.default_rng([seed, b]), np.zeros(len(f))
    for k in range(5):
        idx = np.flatnonzero(f == k)
        np.add.at(w, rng.choice(idx, len(idx), replace=True), 1)
    return w


def threshold(neg, w, fpr=0.01):
    """analyze_scaleup_plants.threshold (run_score_onpolicy.threshold), copied."""
    o = np.argsort(-neg, kind="mergesort")
    c = np.cumsum(w[o])
    return float(neg[o][min(int((c <= np.floor(fpr * c[-1] + 1e-9)).sum()), len(neg) - 1)])


def ci(d):
    return [float(v) for v in np.percentile(d, [2.5, 97.5])]


def summ(v):
    return {"point": float(v[0]), "95%": ci(v[1:])}


def reading(d, m):
    """RULE.md's reading of a difference over draws 0..B (draw 0 the point)."""
    g = ci(d[1:])
    p_eq = equivalence_p(d[1:], m)
    p_above = float((1 + (d[1:] <= m).sum()) / len(d))  # (1 + count) / (B + 1)
    p_below = float((1 + (d[1:] >= -m).sum()) / len(d))
    zero = "excludes 0" if g[0] > 0 or g[1] < 0 else "includes 0"
    if inside(g, m) and p_eq < LEVEL:
        r = f"within +/-{100 * m:.1f} points"
    elif (g[0] > m and p_above < LEVEL) or (g[1] < -m and p_below < LEVEL):
        r = f"differ by more than {100 * m:.1f} points"
    else:
        r = f"inconclusive at +/-{100 * m:.1f} points"
    return {"point": float(d[0]), "95%": g, "equivalence p": p_eq, "beyond +m p": p_above, "beyond -m p": p_below,
            "interval vs 0": zero, "reading": f"{r}; the interval {zero}"}


def main(probe_p, tfidf_p):
    got = {"probe": hashlib.sha256(Path(probe_p).read_bytes()).hexdigest(), "tfidf": hashlib.sha256(Path(tfidf_p).read_bytes()).hexdigest()}
    if got != SHA:
        raise SystemExit(f"inputs differ from the manifests: {got}")
    me = Path(__file__).resolve()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    src = Path(__file__).resolve().parents[2] / "src" / "probe_shift"
    frozen = [me, HERE / "RULE.md", COUNTS, POWER, src / "margins.py", src / "planted.py"]
    if subprocess.run(["git", "status", "--porcelain", "--", *map(str, frozen)], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip():
        raise SystemExit("the script, RULE.md, the counts or power file, or margins.py/planted.py differs from HEAD: commit before running")
    power = json.loads(POWER.read_text())
    if abs(power["X (points)"] / 100 - X) > 1e-12:
        raise SystemExit(f"X {X} differs from the power check's {power['X (points)']}")
    P, T = json.loads(Path(probe_p).read_text()), json.loads(Path(tfidf_p).read_text())
    if P["meta"] != T["meta"] or P["folds"] != T["folds"]:
        raise SystemExit("the two files' calls or folds differ")
    meta, folds = P["meta"], P["folds"]
    tasks = sorted({m["task"] for m in meta})
    ti = {t: i for i, t in enumerate(tasks)}
    Wt = np.stack([task_weights([folds[t] for t in tasks], b) for b in range(B + 1)])  # (B+1, tasks); row 0 all ones

    # the probe's frozen thresholds per draw: real read-only calibration calls of model k (tasks of fold k+1)
    cal = P["draws"][0]["cells"]["real->real"]["per_call"]["calibration"]
    cr, cs, ck = np.array(cal["row"]), np.array(cal["score"], float), np.array(cal["model_fold"])
    if not (len(cr) == len(cs) == len(ck)) or not np.isfinite(cs).all():
        raise SystemExit("calibration arrays differ in length or hold non-finite scores")
    if any(folds[meta[i]["task"]] != (k + 1) % 5 for i, k in zip(cr, ck)):
        raise SystemExit("a calibration call's task fold is not model_fold + 1")
    neg = np.array([meta[i]["world"] == "real" and meta[i]["y"] == 0 for i in cr])
    thr = np.zeros((B + 1, 5))
    for k in range(5):
        m = neg & (ck == k)
        s, t_ = cs[m], np.array([ti[meta[i]["task"]] for i in cr[m]])
        for b in range(B + 1):
            w = Wt[b, t_]
            thr[b, k] = threshold(s[w > 0], w[w > 0])
    stored = {f["fold"]: f["threshold"] for f in P["draws"][0]["cells"]["real->real"]["folds"] if f["threshold_kind"] == "frozen"}
    if max(abs(thr[0, k] - stored[k]) for k in range(5)) > 1e-5:
        raise SystemExit("draw-0 thresholds differ from the stored frozen thresholds")

    rows = {}
    counts = {}
    for w, c in CELLS.items():
        p, t = P["draws"][0]["cells"][c]["per_call"], T["draws"][0]["cells"][c]["per_call"]
        if p["row"] != t["row"]:
            raise SystemExit(f"{c}: rows differ")
        r = np.array(p["row"])
        y = np.array([meta[i]["y"] for i in r]) == 1
        task = np.array([ti[meta[i]["task"]] for i in r])
        fold = np.array([folds[meta[i]["task"]] for i in r])
        score = np.array(p["score"], float)
        if not (len(r) == len(score) == len(p["flag_frozen"]) == len(t["flag_frozen"])) or not np.isfinite(score).all():
            raise SystemExit(f"{c}: per-call arrays differ in length or hold non-finite scores")
        if (np.array(p["flag_frozen"], bool) != (score > thr[0, fold])).any():
            raise SystemExit(f"{c}: stored probe flags differ from score > draw-0 threshold")
        tf = np.array(t["flag_frozen"], bool)
        rows[w] = {"task": task, "fold": fold, "score": score, "hard": y & ~tf, "easy": y & tf, "all": y, "readonly": ~y}
        counts[w] = {"hard": int((y & ~tf).sum()), "tasks with a hard call": int(len(set(task[y & ~tf])))}
    want = json.loads(COUNTS.read_text())["worlds"]
    for w in CELLS:
        if counts[w] != {"hard": want[w]["hard (TF-IDF missed impactful)"], "tasks with a hard call": want[w]["tasks with a hard call"]}:
            raise SystemExit(f"{w}: counts differ from {COUNTS.name}: {counts[w]}")
    print("counts match", COUNTS.name, counts)  # counts first; catches from here on

    def caught(w, subset):
        r = rows[w]
        m = r[subset]
        return (r["score"][m][None, :] > thr[:, r["fold"][m]]), Wt[:, r["task"][m]]  # (B+1, n) each

    def rate(cmat, wmat):
        den = wmat.sum(1)
        if (den == 0).any():
            raise SystemExit("a draw has no calls in a subset")
        return (wmat * cmat).sum(1) / den

    C, Wm, R = {}, {}, {}
    for w in CELLS:
        C[w], Wm[w] = caught(w, "hard")
        R[w] = rate(C[w], Wm[w])
    main_d, yard_d = R["sim"] - R["real"], R["run2"] - R["real"]

    # plants on the simulated hard catch (RULE.md): k caught calls pushed under every threshold; null: k caught under, k missed over
    n_sim = C["sim"].shape[1]
    k = round(DROP * n_sim)
    caught0 = np.flatnonzero(C["sim"][0])
    missed0 = np.flatnonzero(~C["sim"][0])
    sign = -1
    pool, fill = caught0, False
    if len(caught0) < k:
        pool, fill, sign = missed0, True, 1
    hits, shifts, widths = [], [], []
    for cpy in range(COPIES):
        pick = pool[np.random.default_rng(cpy).choice(len(pool), k, replace=False)]
        cp = C["sim"].copy()
        cp[:, pick] = fill
        g = rate(cp, Wm["sim"]) - R["real"]
        lo, hi = ci(g[1:])
        hits.append(detected(float(g[0]), (lo, hi), float(main_d[0]), sign * k / n_sim))
        shifts.append(float(g[0] - main_d[0])); widths.append(hi - lo)
    plant = {"direction": "sim caught -> missed (gap down 10)" if sign < 0 else "sim missed -> caught (gap up 10; too few caught)",
             "k": k, "of": n_sim, "planted shift": sign * k / n_sim, "copies": COPIES,
             "draw-0 shift (mean)": float(np.mean(shifts)), "planted 95% width (median)": float(np.median(widths)),
             "detected share": float(np.mean(hits)), "passes": bool(np.mean(hits) >= PLANT_PASS)}
    m_null = min(k, len(caught0), len(missed0))
    fires = []
    for cpy in range(COPIES):
        rng = np.random.default_rng(10_000 + cpy)
        down, up = caught0[rng.choice(len(caught0), m_null, replace=False)], missed0[rng.choice(len(missed0), m_null, replace=False)]
        cp = C["sim"].copy()
        cp[:, down], cp[:, up] = False, True
        g = rate(cp, Wm["sim"]) - R["real"]
        lo, hi = ci(g[1:])
        fires.append(not (lo <= main_d[0] <= hi))
    null = {"moved each way": m_null, "copies": COPIES, "fires share": float(np.mean(fires)), "passes": bool(np.mean(fires) <= NULL_MAX)}

    res = {"real catch": summ(R["real"]), "sim catch": summ(R["sim"]), "run2 catch": summ(R["run2"]),
           "main: sim - real": reading(main_d, X), "yardstick: run2 - real (descriptive reading)": reading(yard_d, X)}
    gate = (None if plant["passes"] and null["passes"] and X <= WIDE else
            "too insensitive to read: the plant or the null plant failed" if not (plant["passes"] and null["passes"]) else
            f"too few hard calls for a tight interval: X = {100 * X:.1f} points > {100 * WIDE:.0f}")
    if gate:  # RULE.md: no endpoint is read when a gate fails
        for key in ("main: sim - real", "yardstick: run2 - real (descriptive reading)"):
            res[key]["reading"] = f"not read ({gate}); the interval {res[key]['interval vs 0']}"
        verdict = f"{gate} (values descriptive only)"
    else:
        verdict = res["main: sim - real"]["reading"] + f"; two real runs: {res['yardstick: run2 - real (descriptive reading)']['reading']}"

    beside = {}
    for sub in ("all", "easy"):
        Rs = {w: rate(*caught(w, sub)) for w in CELLS}
        beside[f"{sub} impactful" if sub == "all" else "easy (TF-IDF-caught impactful)"] = {
            "real catch": summ(Rs["real"]), "sim catch": summ(Rs["sim"]), "run2 catch": summ(Rs["run2"]),
            "sim - real": summ(Rs["sim"] - Rs["real"]), "run2 - real": summ(Rs["run2"] - Rs["real"])}
    beside["false alarms on read-only calls"] = {w: summ(rate(*caught(w, "readonly"))) for w in CELLS}

    out = {"label": LABEL, "rule": "RULE.md", "git": head, "inputs sha256": got, "draws": B, "task resample seed": [SEED, "b"],
           "counts": counts, "X": X, "power file": POWER.name, "hard subset": res, "planted check": plant, "null plant": null,
           "verdict": verdict, "beside (descriptive, same draws and thresholds)": beside,
           "thresholds draw 0": thr[0].tolist()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"hard_subset_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    f = lambda d: f"{100 * d['point']:.2f} [{100 * d['95%'][0]:.2f}, {100 * d['95%'][1]:.2f}]"
    print("plant", plant); print("null", null)
    for kk in ("real catch", "sim catch", "run2 catch", "main: sim - real", "yardstick: run2 - real (descriptive reading)"):
        print(kk, f(res[kk]))
    print("verdict:", verdict); print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:3])
