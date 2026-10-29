"""Headline robustness pack (PLAN_POSTHOC_headline_robustness_2026-10-12.md; POST HOC, rules committed first). $0, local,
stored draw-0 scores only; no sealed runs. (a) catch at stricter alarms with the frozen real threshold, (b) catch at
matched false alarms (each world's own test-fold read-only calls), (c) the paired per-task shift in mean score on
state-changing commands, (d) a log-false-alarm figure, and a planted check on (a), (b) and (c).

  uv run --no-project --with numpy --with matplotlib python analyze_POSTHOC_headline_robustness.py <draws0-0 file>
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_scaleup_plants import threshold

HERE = Path(__file__).parent
N_BOOT, SEED = 2000, 20261012
WORLDS = {"real": "real->real", "sim": "real->sim", "run2": "real->run2"}
STORED = {"real": 0.8915766738660907, "sim": 0.8870244466946333, "run2": 0.8717836257309941}
TARGETS_A, TARGETS_B = (0.01, 0.005, 0.0025), (0.01, 0.005)


def main(path: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    raw = Path(path).read_bytes()
    D = json.loads(raw)
    meta, folds = D["meta"], D["folds"]
    tasks = sorted({m["task"] for m in meta})
    ti = {t: i for i, t in enumerate(tasks)}
    fold_of = np.array([folds[t] for t in tasks])
    cells = D["draws"][0]["cells"]
    cal = cells["real->real"]["per_call"]["calibration"]
    cr = np.array(cal["row"])
    keep = np.array([meta[i]["world"] == "real" and meta[i]["y"] == 0 for i in cr])  # real run 1's read-only calls
    c_t, c_s, c_k = np.array([ti[meta[i]["task"]] for i in cr[keep]]), np.array(cal["score"])[keep], np.array(cal["model_fold"])[keep]
    test = {}
    for v, cell in WORLDS.items():
        pc = cells[cell]["per_call"]
        rows = np.array(pc["row"])
        test[v] = {"t": np.array([ti[meta[i]["task"]] for i in rows]), "f": np.array([folds[meta[i]["task"]] for i in rows]),
                   "y": np.array([meta[i]["y"] for i in rows]), "s": np.array(pc["score"]), "flag": np.array(pc["flag_frozen"])}

    def frozen_t(w, fpr):
        return np.array([threshold(c_s[(c_k == k) & (w[c_t] > 0)], w[c_t[(c_k == k) & (w[c_t] > 0)]], fpr) for k in range(5)])

    def rate(v, w, flags, y):
        d = test[v]
        m = d["y"] == y
        ww = w[d["t"][m]]
        return float((ww * flags[m]).sum() / ww.sum())

    def a_vals(w, fpr, s_sim=None):
        th = frozen_t(w, fpr)
        out = {}
        for v in WORLDS:
            s = s_sim if (v == "sim" and s_sim is not None) else test[v]["s"]
            fl = s > th[test[v]["f"]]
            out[v] = (rate(v, w, fl, 1), rate(v, w, fl, 0))
        return out, th

    def b_vals(w, fpr, s_sim=None):
        out = {}
        for v in WORLDS:
            d = test[v]
            s = s_sim if (v == "sim" and s_sim is not None) else d["s"]
            th = np.full(5, np.inf)
            for k in range(5):
                m = (d["f"] == k) & (d["y"] == 0) & (w[d["t"]] > 0)
                th[k] = threshold(s[m], w[d["t"][m]], fpr)
            fl = s > th[d["f"]]
            out[v] = (rate(v, w, fl, 1), rate(v, w, fl, 0))
        return out

    real_pos = (test["real"]["y"] == 1)
    resid = np.concatenate([test["real"]["s"][real_pos & (test["real"]["f"] == k)] - test["real"]["s"][real_pos & (test["real"]["f"] == k)].mean() for k in range(5)])
    SD = float(np.sqrt((resid ** 2).sum() / (len(resid) - 5)))

    def task_means(v, s=None):
        d = test[v]
        s = d["s"] if s is None else s
        m = d["y"] == 1
        n = np.bincount(d["t"][m], minlength=len(tasks))
        tot = np.bincount(d["t"][m], weights=s[m], minlength=len(tasks))
        return np.where(n > 0, tot / np.maximum(n, 1), np.nan)

    base_mean = {v: task_means(v) for v in WORLDS}

    def c_vals(w, s_sim=None):
        out = {}
        for v in ("sim", "run2"):
            mv = task_means(v, s_sim) if (v == "sim" and s_sim is not None) else base_mean[v]
            ok = ~np.isnan(mv) & ~np.isnan(base_mean["real"]) & (w > 0)
            out[v] = float((w[ok] * (mv[ok] - base_mean["real"][ok])).sum() / w[ok].sum())
        return out

    # reproduction guard: the frozen 1% flags and catch
    one = np.ones(len(tasks))
    a1, th1 = a_vals(one, 0.01)
    for v in WORLDS:
        fl = test[v]["s"] > th1[test[v]["f"]]
        mism = int((fl != test[v]["flag"]).sum())
        if mism or abs(a1[v][0] - STORED[v]) > 1e-9:
            raise SystemExit(f"{v}: the frozen 1% flags or catch are not reproduced ({mism} flag mismatches, catch {a1[v][0]} vs {STORED[v]})")

    # planted: the smallest delta that drops simulated catch at 0.5% (frozen) by 5 points on draw 0
    sim = test["sim"]
    base05 = a_vals(one, 0.005)[0]["sim"][0]

    def planted_scores(delta):
        return np.where(sim["y"] == 1, sim["s"] - delta, sim["s"])

    lo, hi = 0.0, 20.0
    if base05 - a_vals(one, 0.005, planted_scores(hi))[0]["sim"][0] < 0.05:
        raise SystemExit("no shift up to 20 drops simulated catch at 0.5% by 5 points: the plant can't be built")
    for _ in range(60):
        mid = (lo + hi) / 2
        if base05 - a_vals(one, 0.005, planted_scores(mid))[0]["sim"][0] >= 0.05:
            hi = mid
        else:
            lo = mid
    delta = hi
    s_pl = planted_scores(delta)
    drop = base05 - a_vals(one, 0.005, s_pl)[0]["sim"][0]
    if not 0.05 <= drop <= 0.06:  # catch is a step function: the smallest delta reaching 5 points may overshoot a little
        raise SystemExit(f"the plant's drop at 0.5% is {drop:.4f}, outside [0.05, 0.06]")

    draws = []
    for b in range(1, N_BOOT + 1):
        rng, w = np.random.default_rng([SEED, 2, b]), np.zeros(len(tasks))
        for k in range(5):
            idx = np.flatnonzero(fold_of == k)
            np.add.at(w, rng.choice(idx, len(idx), replace=True), 1)
        draws.append(w)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    inside = lambda iv, m: bool(-m < iv[0] and iv[1] < m)  # noqa: E731
    out = {"label": "POST HOC headline robustness pack (rules: PLAN_POSTHOC_headline_robustness_2026-10-12.md)", "git": git,
           "input": {Path(path).name: hashlib.sha256(raw).hexdigest()}, "resamples": N_BOOT, "sd_real_positive_scores": SD,
           "auroc_gap_quoted": {"real_minus_sim": 0.0045, "ci95_500_refits": [-0.0028, 0.0156], "source": "the 196-task scoring's draws 0-500"},
           "a_stricter_alarms": {}, "b_matched_false_alarms": {}, "c_score_shift": {}, "planted": {"delta": delta}}
    # (a)
    for fpr in TARGETS_A:
        p0, th = a_vals(one, fpr)
        bs = [a_vals(w, fpr)[0] for w in draws]
        support = [int(((c_k == k) & (c_s > th[k])).sum()) for k in range(5)]
        rs, rr = [x["real"][0] - x["sim"][0] for x in bs], [x["real"][0] - x["run2"][0] for x in bs]
        out["a_stricter_alarms"][f"{fpr:.4f}"] = {
            "catch": {v: p0[v][0] for v in WORLDS}, "false_alarm": {v: p0[v][1] for v in WORLDS},
            "real_calibration_read_only_above_threshold_per_fold": support,
            "real_minus_sim": p0["real"][0] - p0["sim"][0], "real_minus_sim_ci95": ci(rs),
            "run1_minus_run2": p0["real"][0] - p0["run2"][0], "run1_minus_run2_ci95": ci(rr),
            "inside_10pt": inside(ci(rs), 0.10), "inside_5pt": inside(ci(rs), 0.05)}
    # (b)
    for fpr in TARGETS_B:
        p0 = b_vals(one, fpr)
        bs = [b_vals(w, fpr) for w in draws]
        rs, rr = [x["real"][0] - x["sim"][0] for x in bs], [x["real"][0] - x["run2"][0] for x in bs]
        out["b_matched_false_alarms"][f"{fpr:.4f}"] = {
            "catch": {v: p0[v][0] for v in WORLDS}, "false_alarm": {v: p0[v][1] for v in WORLDS},
            "real_minus_sim": p0["real"][0] - p0["sim"][0], "real_minus_sim_ci95": ci(rs),
            "run1_minus_run2": p0["real"][0] - p0["run2"][0], "run1_minus_run2_ci95": ci(rr),
            "inside_10pt": inside(ci(rs), 0.10), "inside_5pt": inside(ci(rs), 0.05)}
    # (c)
    c0 = c_vals(one)
    cb = [c_vals(w) for w in draws]
    out["c_score_shift"] = {"sim_minus_real": c0["sim"], "sim_minus_real_ci95": ci([x["sim"] for x in cb]),
                            "run2_minus_run1": c0["run2"], "run2_minus_run1_ci95": ci([x["run2"] for x in cb]),
                            "sim_minus_real_sd": c0["sim"] / SD, "sim_minus_real_sd_ci95": [x / SD for x in ci([x["sim"] for x in cb])],
                            "run2_minus_run1_sd": c0["run2"] / SD, "run2_minus_run1_sd_ci95": [x / SD for x in ci([x["run2"] for x in cb])],
                            "inside_0p2_sd": inside([x / SD for x in ci([x["sim"] for x in cb])], 0.2)}
    # planted (same resamples)
    pa0, pb0, pc0 = a_vals(one, 0.005, s_pl)[0], b_vals(one, 0.005, s_pl), c_vals(one, s_pl)
    ua, ub = out["a_stricter_alarms"]["0.0050"], out["b_matched_false_alarms"]["0.0050"]
    pa = ci([(lambda x: x["real"][0] - x["sim"][0])(a_vals(w, 0.005, s_pl)[0]) for w in draws])
    pb = ci([(lambda x: x["real"][0] - x["sim"][0])(b_vals(w, 0.005, s_pl)) for w in draws])
    pc_ = ci([c_vals(w, s_pl)["sim"] for w in draws])
    for name, unpl, pl_point, pl_ci, amount in (
            ("a_0.5pct", ua["real_minus_sim"], pa0["real"][0] - pa0["sim"][0], pa, 0.05),
            ("b_0.5pct", ub["real_minus_sim"], pb0["real"][0] - pb0["sim"][0], pb, (pb0["real"][0] - pb0["sim"][0]) - ub["real_minus_sim"]),
            ("c", c0["sim"], pc0["sim"], pc_, -delta)):
        shift = pl_point - unpl
        ok_size = amount != 0 and 0.5 <= shift / amount <= 1.5
        excl = not (pl_ci[0] <= unpl <= pl_ci[1])
        out["planted"][name] = {"planted_amount": amount, "shift": shift, "planted_ci95": pl_ci, "unplanted_estimate": unpl,
                                "detected": bool(ok_size and excl)}
    out["planted"]["achieved_drop_a_0.5pct"] = drop
    # a measure whose plant isn't detected is "not testable" and isn't read (plan)
    testable = {"a": out["planted"]["a_0.5pct"]["detected"], "b": out["planted"]["b_0.5pct"]["detected"], "c": out["planted"]["c"]["detected"]}
    for f in out["a_stricter_alarms"].values():
        f["testable"] = testable["a"]
    for f in out["b_matched_false_alarms"].values():
        f["testable"] = testable["b"]
    out["c_score_shift"]["testable"] = testable["c"]
    ok = all(testable.values())
    bolsters = ok and all(out["a_stricter_alarms"][f]["inside_10pt"] for f in ("0.0050", "0.0025")) \
        and out["b_matched_false_alarms"]["0.0050"]["inside_10pt"] and out["c_score_shift"]["inside_0p2_sd"]
    out["reading"] = ("bolsters: the score distribution matches, not just catch at 1%" if bolsters
                      else "scoped to the 1% operating point (see which measure fails)")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"headline_robustness_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    # (d) figure: catch against achieved false alarms, each world's own per-fold thresholds
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.logspace(-3, 0, 60)[:-1]  # the 100% endpoint is added as (1, 1): the rule's threshold at 1 flags no minimum
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for v, col, lab in (("real", "#1f77b4", "real run 1"), ("sim", "#d62728", "simulated"), ("run2", "#7f7f7f", "real run 2")):
        pts = [b_vals(one, x)[v] for x in grid]
        ax.plot([p[1] for p in pts] + [1.0], [p[0] for p in pts] + [1.0], color=col, label=lab, lw=1.8)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_xlabel("False alarms on read-only commands (log; left is better)")
    ax.set_ylabel("Catch on state-changing commands (higher is better)")
    ax.axvline(0.01, color="k", lw=0.6, ls=":")
    ax.legend(frameon=False)
    ax.set_title("Real-trained probe: catch vs false alarms")
    fig.tight_layout()
    fdst = HERE / "figures" / f"headline_robustness_roc_POSTHOC_{stamp}.png"
    fig.savefig(fdst, dpi=150)
    pct = lambda x: round(100 * x, 2)  # noqa: E731
    for f, r in out["a_stricter_alarms"].items():
        print(f"(a) target {float(f) * 100:.2f}%: catch real {pct(r['catch']['real'])} sim {pct(r['catch']['sim'])} run2 {pct(r['catch']['run2'])}; "
              f"FA real {pct(r['false_alarm']['real'])} sim {pct(r['false_alarm']['sim'])}; real-sim {pct(r['real_minus_sim'])} {[pct(x) for x in r['real_minus_sim_ci95']]}; "
              f"run1-run2 {pct(r['run1_minus_run2'])} {[pct(x) for x in r['run1_minus_run2_ci95']]}; support {r['real_calibration_read_only_above_threshold_per_fold']}")
    for f, r in out["b_matched_false_alarms"].items():
        print(f"(b) matched {float(f) * 100:.2f}%: catch real {pct(r['catch']['real'])} sim {pct(r['catch']['sim'])} run2 {pct(r['catch']['run2'])}; "
              f"FA real {pct(r['false_alarm']['real'])} sim {pct(r['false_alarm']['sim'])} run2 {pct(r['false_alarm']['run2'])}; "
              f"real-sim {pct(r['real_minus_sim'])} {[pct(x) for x in r['real_minus_sim_ci95']]}; run1-run2 {pct(r['run1_minus_run2'])} {[pct(x) for x in r['run1_minus_run2_ci95']]}")
    c = out["c_score_shift"]
    print(f"(c) sim-real {c['sim_minus_real']:.4f} {[round(x, 4) for x in c['sim_minus_real_ci95']]} = {c['sim_minus_real_sd']:.3f} SD "
          f"{[round(x, 3) for x in c['sim_minus_real_sd_ci95']]}; run2-run1 {c['run2_minus_run1_sd']:.3f} SD {[round(x, 3) for x in c['run2_minus_run1_sd_ci95']]}; SD {SD:.4f}")
    print("planted:", json.dumps(out["planted"]))
    print("reading:", out["reading"])
    print("->", dst, fdst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    main(ap.parse_args().path)
