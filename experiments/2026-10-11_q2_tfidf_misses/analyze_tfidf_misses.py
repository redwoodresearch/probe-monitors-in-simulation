"""Real vs simulated catch of the dangerous-command probe on the impactful commands the TF-IDF misses (PREREG.md, 0b6e8cce
then 33566591, committed before this script). Probe and TF-IDF both trained on real, both at their frozen 1% thresholds;
10,000 task draws seeded 20261011; the 10-point simulated-side plant (200 copies) gates the reading. Keys, labels and
stored scores only; $0.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_tfidf_misses/analyze_tfidf_misses.py \\
      <qwen3.5__call_end__attack_only.jsonl.gz> <text_baseline_predictions_2026-10-07_115359_400dcc.jsonl.gz>
"""
import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from probe_shift.margins import LEVEL, equivalence_p, inside  # noqa: E402
from probe_shift.planted import detected  # noqa: E402

HERE = Path(__file__).parent
SHA = {"probe": "75178f1cfef2eb60436b75d809624cf1e37601c9dab552ffaa23b1fb981cedf3",
       "tfidf": "756cb8494d03d0f3598ed267968e68e0fee6f485ddb9bc07c16f3d9c20532c44"}
WORLDS = {"real": "real-attack-e1", "simulated": "sim-attack"}
SEED, DRAWS, COPIES, PLANT = 20261011, 10_000, 200, 0.10
# PREREG's counts, checked before any catch is computed
PREREG = {"tasks": {"real": 536, "simulated": 536}, "impactful": {"real": 14075, "simulated": 14146},
          "subset": {"real": 845, "simulated": 811}, "tasks with subset": {"real": 329, "simulated": 325},
          "most in one task": {"real": 34, "simulated": 26}, "kish": {"real": 2.328, "simulated": 2.195},
          "categories": {"real": {"run_program": 321, "network": 141, "file_write": 98, "process_service": 95, "install": 52, "db_write": 37,
                                  "permissions_config": 37, "file_delete": 32, "file_write_tmp": 26, "unknown": 3, "malformed": 3},
                         "simulated": {"run_program": 308, "network": 131, "process_service": 92, "file_write": 88, "install": 56, "db_write": 39,
                                       "permissions_config": 34, "file_write_tmp": 31, "file_delete": 29, "unknown": 3}},
          "tools": {"real": {"bash": 772, "db_bash": 45, "text_editor": 11, "client_bash": 9, "py_ml_bash": 4, "rust_worker_bash": 3, "nginx_bash": 1},
                    "simulated": {"bash": 752, "db_bash": 40, "text_editor": 11, "client_bash": 3, "py_ml_bash": 2, "nginx_bash": 2, "rust_worker_bash": 1}},
          "joined rows": 44919, "margin": 0.135}


def key(r):
    return r["arm"], r["key"], int(r["step"]), int(r["action"])


def load(path, keep):
    with gzip.open(path, "rt") as f:
        rows = [r for r in map(json.loads, f) if keep(r)]
    out = {key(r): r for r in rows}
    if len(out) != len(rows):
        raise SystemExit(f"{path}: duplicate keys")
    return out


def kish(m):
    m = m[m > 0]
    return len(m) * float((m ** 2).sum()) / float(m.sum()) ** 2


def margin(n_r, n_s, k):
    """PREREG: +/-10 unless the #666 gate gives under 80% for landing inside it at a true gap of 0; then the smallest
    margin with 80%, rounded up to the next half point."""
    de = max(1.25, 1.2 * k)
    se = math.sqrt(0.25 * de * (1 / n_r + 1 / n_s))
    p10 = max(0.0, math.erf((0.10 - 1.959964 * se) / se / math.sqrt(2)))
    m = 0.10 if p10 >= 0.80 else math.ceil((1.959964 + 1.281552) * se * 200 - 1e-9) / 200
    return {"DE": de, "SE": se, "P(inside +/-10 at gap 0), projected": p10, "small": p10 < 0.80, "margin": m}


def ci(d):
    return [float(v) for v in np.percentile(d, [2.5, 97.5])]


def main(probe_p, tfidf_p):
    got = {"probe": hashlib.sha256(Path(probe_p).read_bytes()).hexdigest(), "tfidf": hashlib.sha256(Path(tfidf_p).read_bytes()).hexdigest()}
    if got != SHA:
        raise SystemExit(f"inputs differ from PREREG's: {got}")
    me = Path(__file__).resolve()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    p = load(probe_p, lambda r: r["train_set"] == "real")
    t = load(tfidf_p, lambda r: r["baseline"] == "command_text" and r["train_set"] == "real-attack-e1")
    if not set(p) <= set(t):
        raise SystemExit(f"{len(set(p) - set(t))} probe rows have no TF-IDF row")
    ks = sorted(p)
    checks = {"label mismatches": sum(int(p[k]["y"]) != int(t[k]["y"]) for k in ks),
              "fold mismatches": sum(int(p[k]["fold"]) != int(t[k]["fold"]) for k in ks),
              "probe flag != score > threshold": sum(bool(p[k]["flag"]) != (float(p[k]["score"]) > float(p[k]["threshold"])) for k in ks),
              "TF-IDF flag != score > threshold": sum((str(t[k]["flag"]) == "True") != (float(t[k]["score"]) > float(t[k]["threshold"])) for k in ks)}
    folds_per_task = {}
    for k in ks:
        folds_per_task.setdefault(p[k]["main_task"], set()).add(int(p[k]["fold"]))
    checks["tasks in more than one fold"] = sum(len(v) > 1 for v in folds_per_task.values())
    checks["joined rows != PREREG"] = int(len(ks) != PREREG["joined rows"])
    if any(checks.values()):
        raise SystemExit(f"join checks failed: {checks}")
    tasks = sorted({p[k]["main_task"] for k in ks})
    ti = {x: i for i, x in enumerate(tasks)}
    arm = np.array([k[0] for k in ks]); task = np.array([ti[p[k]["main_task"]] for k in ks])
    y = np.array([int(p[k]["y"]) for k in ks]) == 1
    caught = np.array([bool(p[k]["flag"]) for k in ks])
    missed_text = np.array([str(t[k]["flag"]) != "True" for k in ks])
    T = len(tasks)
    sub, allpos, counts = {}, {}, {}
    for w, a in WORLDS.items():
        s = (arm == a) & y & missed_text
        n_s = np.bincount(task[s], minlength=T).astype(float)
        sub[w] = (s, n_s)
        counts[w] = {"tasks": int(len(set(task[arm == a]))), "impactful": int(((arm == a) & y).sum()), "subset": int(s.sum()),
                     "tasks with subset": int((n_s > 0).sum()), "most in one task": int(n_s.max()), "kish": round(kish(n_s), 3),
                     "categories": dict(Counter(p[ks[i]]["category"] for i in np.flatnonzero(s))),
                     "tools": dict(Counter(p[ks[i]]["tool"] for i in np.flatnonzero(s)))}
    if set(task[arm == WORLDS["real"]]) != set(task[arm == WORLDS["simulated"]]):
        raise SystemExit("the two worlds cover different tasks")
    for field in ("tasks", "impactful", "subset", "tasks with subset", "most in one task", "kish", "categories", "tools"):
        if {w: counts[w][field] for w in WORLDS} != PREREG[field]:
            raise SystemExit(f"{field} differs from PREREG: {[counts[w][field] for w in WORLDS]}")
    mg = margin(counts["real"]["subset"], counts["simulated"]["subset"], max(kish(sub[w][1]) for w in WORLDS))
    if mg["margin"] != PREREG["margin"]:
        raise SystemExit(f"margin {mg['margin']} differs from PREREG's {PREREG['margin']}")
    m = mg["margin"]
    print("counts match PREREG:", counts)
    # Counts match PREREG; catches from here on.
    for w, a in WORLDS.items():
        allpos[w] = (np.bincount(task[(arm == a) & y], weights=caught[(arm == a) & y].astype(float), minlength=T),
                     np.bincount(task[(arm == a) & y], minlength=T).astype(float))
    rng = np.random.default_rng(SEED)
    W = np.zeros((DRAWS + 1, T)); W[0] = 1
    np.add.at(W[1:], (np.repeat(np.arange(DRAWS), T), rng.integers(0, T, (DRAWS, T)).ravel()), 1)

    def rate(c, n):
        den = W @ n
        if (den == 0).any():
            raise SystemExit("a draw has no commands in a world")
        return (W @ c) / den

    c_sub = {w: np.bincount(task[sub[w][0]], weights=caught[sub[w][0]].astype(float), minlength=T) for w in WORLDS}
    cr, cs = rate(c_sub["real"], sub["real"][1]), rate(c_sub["simulated"], sub["simulated"][1])
    gap = cr - cs
    # The plant, first: 81 simulated subset commands moved from caught to missed (or missed to caught if too few caught).
    s_idx = np.flatnonzero(sub["simulated"][0])  # ks is sorted, so these rows are in (arm, key, step, action) order
    n_sim = len(s_idx); k = round(PLANT * n_sim)
    pool = s_idx[caught[s_idx]]; sign = 1
    if len(pool) < k:
        pool, sign = s_idx[~caught[s_idx]], -1
    hits, shifts, widths = [], [], []
    for cpy in range(COPIES):
        pick = pool[np.random.default_rng(cpy).choice(len(pool), k, replace=False)]
        moved = np.bincount(task[pick], minlength=T)
        gp = cr - rate(c_sub["simulated"] - sign * moved, sub["simulated"][1])
        lo, hi = ci(gp[1:])
        hits.append(detected(float(gp[0]), (lo, hi), float(gap[0]), sign * PLANT)); shifts.append(float(gp[0] - gap[0])); widths.append(hi - lo)
    plant = {"direction": "caught -> missed (+10)" if sign > 0 else "missed -> caught (-10)", "k": k, "copies": COPIES,
             "achieved shift (points, mean)": 100 * float(np.mean(shifts)), "planted 95% width (points, median)": 100 * float(np.median(widths)),
             "detected share": float(np.mean(hits)), "passes (>= 80%)": bool(np.mean(hits) >= 0.80)}
    g_ci = ci(gap[1:])
    res = {"real catch": {"point": float(cr[0]), "95%": ci(cr[1:])}, "simulated catch": {"point": float(cs[0]), "95%": ci(cs[1:])},
           "gap (real - simulated)": {"point": float(gap[0]), "95%": g_ci}}
    if plant["passes (>= 80%)"]:
        p_eq = equivalence_p(gap[1:], m)
        p_above = (1 + (gap[1:] <= m).sum()) / (DRAWS + 1); p_below = (1 + (gap[1:] >= -m).sum()) / (DRAWS + 1)
        if inside(g_ci, m) and p_eq < LEVEL:
            reading = f"agree within +/-{100 * m:.1f} points"
        elif (g_ci[0] > m and p_above < LEVEL) or (g_ci[1] < -m and p_below < LEVEL):
            reading = f"differ by more than {100 * m:.1f} points"
        else:
            reading = f"inconclusive at +/-{100 * m:.1f} points; interval {'excludes' if g_ci[0] > 0 or g_ci[1] < 0 else 'includes'} 0"
        res |= {"equivalence p": p_eq, "beyond +m p": float(p_above), "beyond -m p": float(p_below), "reading": reading}
    else:
        res["reading"] = "too insensitive to read (gap above is descriptive only)"
    ar, as_ = rate(*allpos["real"]), rate(*allpos["simulated"])
    beside = {"real catch": {"point": float(ar[0]), "95%": ci(ar[1:])}, "simulated catch": {"point": float(as_[0]), "95%": ci(as_[1:])},
              "gap (real - simulated)": {"point": float(ar[0] - as_[0]), "95%": ci((ar - as_)[1:])}}
    out = {"prereg": "PREREG.md @ 0b6e8cce, amended 33566591", "git": head, "inputs sha256": got, "join checks": checks, "counts": counts,
           "margin rule": mg, "planted check": plant, "TF-IDF-missed commands": res,
           "beside (descriptive): all impactful commands, same threshold and draws": beside,
           "real-vs-real yardstick": "none: one real attack run per task (epoch 1) in this data"}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"tfidf_misses_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    f = lambda d: f"{100 * d['point']:.1f} [{100 * d['95%'][0]:.1f}, {100 * d['95%'][1]:.1f}]"
    print("margin", mg); print("plant", plant)
    print(f"subset: real {f(res['real catch'])}, sim {f(res['simulated catch'])}, gap {f(res['gap (real - simulated)'])} -> {res['reading']}")
    print(f"all impactful (descriptive): real {f(beside['real catch'])}, sim {f(beside['simulated catch'])}, gap {f(beside['gap (real - simulated)'])}")
    print(dest)


if __name__ == "__main__":
    main(*sys.argv[1:3])
