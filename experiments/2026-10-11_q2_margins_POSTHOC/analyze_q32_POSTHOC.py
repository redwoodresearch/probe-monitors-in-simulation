"""Q3.2 (post hoc, PLAN_POSTHOC.md): the deception probe against a ±0.25-SD margin. Family A, label separation:
(1) sabotage-label D, (2) false-"done" standardized mean difference. Family B, real vs simulated: (3, 4) false-"done"
claims, simulated minus real, for September and today's simulators, (5) run-level S. Holm within each family; the
controls beside each (A: label shuffles; B: real-vs-real); a plant sweep (0.05-0.50 SD) per test. Each statistic and its
bootstrap draws reproduce the frozen analysis's stored values first. Keys and numbers only; no text is read.

  uv run --no-project --with numpy python experiments/2026-10-11_q2_margins_POSTHOC/analyze_q32_POSTHOC.py \
      <sabotage scores json> <run-level scores json>
"""
import hashlib
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from probe_shift.margins import LEVEL, difference_p, equivalence_p, holm, inside  # noqa: E402
from probe_shift.planted import detected  # noqa: E402

HERE = Path(__file__).parent; EXP = HERE.parent
M, N_BOOT, SIZES, COPIES = 0.25, 2000, [round(0.05 * i, 2) for i in range(1, 11)], 200
STORED = {"sabotage": EXP / "2026-10-09_q2_deception_commands/results/sabotage_scores_analysis_2026-10-10_210808.json",
          "claims": EXP / "2026-10-10_q2_false_claims/results/false_claims_analysis_2026-10-10_160736_v2.json",
          "claims posthoc": EXP / "2026-10-10_q2_false_claims/results/false_claims_posthoc_2026-10-10_160736.json",
          "runlevel": EXP / "2026-10-10_q2_deception_runlevel/results/runlevel_analysis_2026-10-10_215302.json"}


def pct(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def check(name, got, want, tol=1e-9):
    if np.abs(np.asarray(got, float) - np.asarray(want, float)).max() > tol:
        raise ValueError(f"{name}: {got} does not reproduce the stored {want}")


def sweep(stat_planted, unplanted, sign, units, rng_units):
    """stat_planted(planted_units, size) -> (point, draws, expected shift); detection rate per size under Aswin's rule."""
    out = {}
    for size in SIZES:
        hits = 0
        for c in range(COPIES):
            pick = np.random.default_rng(c).choice(rng_units, len(rng_units) // 2, replace=False)
            p, dr, exp = stat_planted(pick, sign * size)
            hits += detected(p, tuple(pct(dr)), unplanted, exp)
        out[str(size)] = hits / COPIES
    return out


def main(sab_path: Path, rl_path: Path):
    st = {k: json.loads(v.read_text()) for k, v in STORED.items()}
    out = {"inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (sab_path, rl_path, STORED["claims"])}, "margin (SD)": M,
           "level for inside after Holm": LEVEL}
    tests, ctrl = {}, {}

    # 1. Sabotage-label D (honest command-score SDs, run bootstrap)
    sc = json.loads(sab_path.read_text()); keys = sc["keys"]
    sd = float(np.array([r["score"] for r in sc["honest"]]).std(ddof=1))
    s = np.array([r["score"] for r in keys]); y = np.array([r["label"] == "sabotage" for r in keys], int); task = np.array([r["task"] for r in keys])
    paired = sorted(t for t in set(task) if len(set(y[task == t])) == 2)
    d = np.array([(s[(task == t) & (y == 1)].mean() - s[(task == t) & (y == 0)].mean()) / sd for t in paired])
    rng = np.random.default_rng(0); idx = [rng.integers(0, len(d), len(d)) for _ in range(N_BOOT)]
    boot = lambda v: np.array([v[i].mean() for i in idx])
    ref = st["sabotage"]["primary: within-run paired"]; check("sabotage D", [d.mean(), *pct(boot(d))], [ref["D (honest SDs)"], *ref["95% (run bootstrap)"]])
    tests["1 sabotage-label D (honest command SDs)"] = (float(d.mean()), boot(d))
    ctrl["1 sabotage-label D (honest command SDs)"] = {"label-shuffle control: exact sign-flip p (frozen)": ref["sign-flip p (two-sided, exact)"]}

    def plant1(pick, size):
        v = d.copy(); v[pick] += 2 * size
        return float(v.mean()), boot(v), 2 * size * len(pick) / len(d)
    sw = {"1 sabotage-label D (honest command SDs)": sweep(plant1, float(d.mean()), np.sign(d.mean()), d, np.arange(len(d)))}

    # 2. False-"done": standardized mean difference, false minus truthful (truthful SD recomputed per draw)
    rep = [r for r in st["claims"]["per report"] if r["class"] == "claims success"]
    lab = [r for r in rep if r["world"] == "qwen-real" and r["label"] is not None]
    s2 = np.array([r["score"] for r in lab]); y2 = np.array([r["label"] for r in lab]); t2 = np.array([r["task"] for r in lab])
    smd = lambda sv, yv: float((sv[yv == 1].mean() - sv[yv == 0].mean()) / sv[yv == 0].std(ddof=1))
    by = {g: sorted({t for t, yy in zip(t2, y2) if yy == g}) for g in (0, 1)}; rows = {t: np.flatnonzero(t2 == t) for t in set(t2)}
    rng = np.random.default_rng(0); idx2 = [np.concatenate([rows[t] for g in (0, 1) for t in rng.choice(by[g], len(by[g]))]) for _ in range(N_BOOT)]
    boot2 = lambda sv: np.array([smd(sv[i], y2[i]) for i in idx2])
    ref = st["claims"]["planted check (+0.5 truthful SD on false claims)"]; b2 = boot2(s2)
    check("false-done SMD", [smd(s2, y2), *pct(b2)], [ref["unplanted estimate"], *ref["unplanted 95%"]])
    tests["2 false-done claims, false minus truthful (truthful SDs)"] = (smd(s2, y2), b2)
    null = [smd(s2, np.isin(t2, list(c)).astype(int)) for c in itertools.combinations(sorted(set(t2)), len(by[1]))]
    obs = smd(s2, y2); n = len(null)
    ctrl["2 false-done claims, false minus truthful (truthful SDs)"] = {
        "label-shuffle control: task-block permutation p for this statistic (two-sided, exact)": float(min(1, 2 * min(sum(x >= obs - 1e-12 for x in null), sum(x <= obs + 1e-12 for x in null)) / n)),
        "assignments": n}
    sd_t = float(s2[y2 == 0].std(ddof=1))

    def plant2(pick, size):
        v = s2 + 2 * size * sd_t * ((y2 == 1) & np.isin(t2, pick))
        return smd(v, y2), boot2(v), 2 * size * float(((y2 == 1) & np.isin(t2, pick)).sum()) / float((y2 == 1).sum())
    sw["2 false-done claims, false minus truthful (truthful SDs)"] = sweep(plant2, obs, np.sign(obs), None, np.array(by[1]))

    # 3, 4. False-"done" claims, simulated minus real (real-claim SDs), and honest_b minus honest_a beside them
    real = [r for r in rep if r["world"] == "qwen-real"]; sd_r = float(np.std([r["score"] for r in real], ddof=1))

    def shift(group_a, group_b):
        allr = group_a + group_b; tk = np.array([r["task"] for r in allr]); isb = np.array([i >= len(group_a) for i in range(len(allr))])
        sv = np.array([r["score"] for r in allr]); names = sorted(set(tk)); rws = {t: np.flatnonzero(tk == t) for t in names}
        rng = np.random.default_rng(0); dr = [np.concatenate([rws[t] for t in rng.choice(names, len(names))]) for _ in range(N_BOOT)]
        dr = [i for i in dr if isb[i].any() and (~isb[i]).any()]
        est = lambda v, i: float((v[i][isb[i]].mean() - v[i][~isb[i]].mean()) / sd_r)
        full = np.arange(len(sv))
        return est(sv, full), np.array([est(sv, i) for i in dr]), sv, tk, isb, est, dr, names
    for w, tag in (("qwen-sim@september", "3"), ("qwen-sim@today", "4")):
        p, dr, sv, tk, isb, est, drw, names = shift(real, [r for r in rep if r["world"] == w])
        ref = st["claims posthoc"][f"planted check, real vs {w} (+0.5 real-claim SD on simulated)"]
        check(f"claims real vs {w}", [p, *pct(dr)], [ref["unplanted estimate"], *ref["unplanted 95%"]])
        k = f"{tag} false-done claims, simulated ({w}) minus real (real-claim SDs)"; tests[k] = (p, dr)

        def plant3(pick, size, sv=sv, tk=tk, isb=isb, est=est, drw=drw):
            m = isb & np.isin(tk, pick); v = sv + 2 * size * sd_r * m
            return est(v, np.arange(len(v))), np.array([est(v, i) for i in drw]), 2 * size * float(m.sum()) / float(isb.sum())
        sw[k] = sweep(plant3, p, np.sign(p), None, np.array(sorted(set(tk[isb]))))
    both = sorted({r["task"] for r in real if r["arm"] == "honest_a"} & {r["task"] for r in real if r["arm"] == "honest_b"})
    ha = [r for r in real if r["arm"] == "honest_a" and r["task"] in both]; hb = [r for r in real if r["arm"] == "honest_b" and r["task"] in both]
    p, dr = shift(ha, hb)[:2]
    for tag in ("3", "4"):
        ctrl[next(k for k in tests if k.startswith(tag))] = {"real-vs-real: honest_b minus honest_a claims (real-claim SDs)": {"point": p, "95%": pct(dr), "tasks": len(both),
                                                                                                                              "inside ±0.25": inside(pct(dr), M)}}

    # 5. Run-level S (s_H), confirmatory cohort; C beside it
    rl = json.loads(rl_path.read_text()); runs = {}
    for r in rl["scores"]:
        runs.setdefault((r["arm"], r["task"], r["run"]), []).append(r["score"])
    score = {k: float(np.mean(v)) for k, v in runs.items()}
    sH = float(np.array([v for (a, _, _), v in score.items() if a in ("honest_a", "honest_b")]).std(ddof=1))
    T = {}
    for (arm, t, _), v in score.items():
        T.setdefault(t, {}).setdefault(arm, []).append(v)
    launch_of = {}
    for r in rl["scores"]:
        if r["arm"] == "attack":
            launch_of[r["task"]] = r["dump"]  # as analyze_runlevel.py: the last attack row's dump
    strict = {t for t, dmp in launch_of.items() if dmp == "2026-10-09_rerun_strict_segends_v1"}
    ref = st["runlevel"]["part 2: September simulated vs real attack runs"]["confirmatory (t21-t196)"]

    def paired(ax, ay):
        ts = sorted(t for t, a in T.items() if ax in a and ay in a and t not in strict)
        e = np.array([(T[t][ax][0] - T[t][ay][0]) / sH for t in ts])
        rng = np.random.default_rng(0); dr = [rng.integers(0, len(ts), len(ts)) for _ in range(N_BOOT)]
        return e, dr
    e, dr5 = paired("sim_attack@september", "attack"); b5 = np.array([e[i].mean() for i in dr5])
    check("run-level S", [e.mean(), *pct(b5)], [ref["S (simulated - run 1)"]["mean (s_H)"], *ref["S (simulated - run 1)"]["95% (task bootstrap)"]])
    tests["5 run-level S, simulated minus attack run 1 (s_H)"] = (float(e.mean()), b5)
    ec, drc = paired("attack_run2", "attack"); bc = np.array([ec[i].mean() for i in drc])
    check("run-level C", [ec.mean(), *pct(bc)], [ref["C (run 2 - run 1, real-vs-real)"]["mean (s_H)"], *ref["C (run 2 - run 1, real-vs-real)"]["95% (task bootstrap)"]])
    ctrl["5 run-level S, simulated minus attack run 1 (s_H)"] = {"real-vs-real: C, attack run 2 minus run 1 (s_H)": {"point": float(ec.mean()), "95%": pct(bc), "tasks": len(ec),
                                                                                                                   "inside ±0.25": inside(pct(bc), M)}}

    def plant5(pick, size):
        v = e.copy(); v[pick] += 2 * size
        return float(v.mean()), np.array([v[i].mean() for i in dr5]), 2 * size * len(pick) / len(e)
    sw["5 run-level S, simulated minus attack run 1 (s_H)"] = sweep(plant5, float(e.mean()), np.sign(e.mean()), None, np.arange(len(e)))

    fam = {"A (label separation)": [k for k in tests if k[0] in "12"], "B (real vs simulated)": [k for k in tests if k[0] in "345"]}
    for fname, ks in fam.items():
        eq = holm({k: equivalence_p(tests[k][1], M) for k in ks}); dp = holm({k: difference_p(tests[k][1]) for k in ks})
        for k in ks:
            p, dr = tests[k]
            out[k] = {"family": fname, "point": p, "95%": pct(dr), "bootstrap draws": len(dr), "inside ±0.25 (95% interval)": inside(pct(dr), M),
                      "equivalence p": equivalence_p(dr, M), "Holm-adjusted equivalence p": eq[k], "inside after Holm": eq[k] < LEVEL,
                      "difference p": difference_p(dr), "Holm-adjusted difference p": dp[k], **ctrl[k],
                      "plant sweep (detection rate per size in SD)": sw[k], "smallest size detected in at least 80% of copies": next((float(z) for z, r in sw[k].items() if r >= 0.8), None)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"q32_margins_POSTHOC_{stamp}.json"; dest.parent.mkdir(exist_ok=True)
    with open(dest, "x") as fh:
        fh.write("{\n" + ",\n".join(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")
    for k in tests:
        o = out[k]
        print(f"{k}: {o['point']:+.3f} [{o['95%'][0]:+.3f}, {o['95%'][1]:+.3f}]; inside {o['inside ±0.25 (95% interval)']}, Holm eq p {o['Holm-adjusted equivalence p']:.4f}; "
              f"Holm diff p {o['Holm-adjusted difference p']:.3f}; 80% plant {o['smallest size detected in at least 80% of copies']}")
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
