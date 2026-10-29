"""Priority 1 first cut (RULE.md in this folder): paired sim - run2 and run2 - run1 tests on STORED run scores, pooled
honest A u B fixed thresholds, binary (paired t, McNemar, Tango score interval) and continuous (standardized paired t).
The laptop only reads the pinned committed JSONs, checks their sha256 and passes the score maps; every number is computed on
Modal CPU. Gate first: each record's stored nominal threshold and catches are reproduced exactly, or no number is written.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run \\
      experiments/2026-10-14_priority1_firstcut/run_firstcut.py --labels <claimc_labels_reduced.json>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-p1firstcut_paired_kill-2026-11-13"
R = "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/"
FILES = {  # name: (commit, path, sha256)
    "avg_e2": ("f47920a74", R + "avg_transfer_avg_v1_cueex_2026-10-14_104113.json", "f2c1c0e43c0e1376214b97b6b84a66136d6fff5cfe1ebbe8d3503369f1681144"),
    "avg_all": ("aa41408f6", R + "avg_transfer_avg_v1_2026-10-14_011347.json", "cc90ed914bdee4119e9648f7bc85f0e835f9fcd61e2ade3a6185b448b95fd4f4"),
    "recheck": ("e6081f344", "experiments/2026-10-14_headline_recheck_09/results/recheck_2026-10-14_015553.json", "5503cced397c0438b135154ee35abc6755ebe3b4814bbd5a7909609bf7f4ae33"),
    "cmd_all": ("737e6a92a", R + "cmd_readout_2026-10-14_033117.json", "16c0e11d0de3b34b25eb6057f4bca48efc10a989a2a6b4cf7b91e9c5920dd165"),
    "cmd_e2": ("af0c0ec41", R + "cmd_readout_cueex_2026-10-14_104112.json", "918966e49a916ea645d3cb4e620c9a9abd1d9194dce751b72f301c5256a89a96"),
    "armrs": ("185e27988", R + "armrs_top2_2026-10-14_022925.json", "cbcf73bad7908dac3b279ab37fa8f0a794a7cc7d444a16b63ae19344a747207e"),
    "item1": ("2d94e482f", R + "test_item1_scores_2026-10-14_001808.json", "2797c8fc79426553ed5e229ceb7045c6f72470b37337ac83ec878a462abe9ea1"),
    "e2": ("f2717b130", R + "cue_exclusion_E2_2026-10-14_104038.json", "c6f33dc81389160ada71e623f2e84c4dac1914c5277b727d819b0ddb0e393a29"),
    "recipe": ("e7820ab48", "experiments/2026-10-13_recipe_transfer_POSTHOC/results/recipe_transfer_POSTHOC_2026-10-14_000532.json",
               "bca6a8b3b9d49e65bececa43cf6b3e043a5bdd41b0b69cd2333a03ade144a6a1"),
}
PRIOR = "results/firstcut_2026-10-14_115701.json"  # Amendment 3: every existing row must equal this record exactly
REPLAY = "as-is Sol-trained, last token, replay layout (secondary)"
LABELS_SHA = "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995"
E2 = "E2: corrected B restricted to unchanged runs"
ALL = "all runs (unfixed render)"
COLS = [f"{p}k{k}_d{d}" for p in ("", "S_") for k in (10, 30) for d in range(5)]
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scipy==1.16.2")


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True).stdout


@app.function(image=image, cpu=(4, 4), memory=(4096, 4096), timeout=900, retries=0)
def compute(probes: list, labels: dict, prior_rows: list) -> dict:
    import numpy as np
    from scipy import stats

    def thr(h, a):  # spec section 3, unit weights: first score (desc, stable) whose cumulative weight exceeds floor(aW + 1e-9)
        h = np.asarray(h, float)
        o = h[np.argsort(-h, kind="mergesort")]
        return float(o[min(int(np.floor(a * len(o) + 1e-9)), len(o) - 1)])

    def tango(b, c, n, z=1.959963984540054):  # Tango (1998) score interval for (b - c)/n, paired proportions
        def T(d0):
            A, B, C = 2 * n, -b - c + (2 * n - b + c) * d0, -c * d0 * (1 - d0)
            q = (np.sqrt(max(B * B - 4 * A * C, 0.0)) - B) / (2 * A)
            den = np.sqrt(n * (2 * q + d0 * (1 - d0)))
            return (b - c - n * d0) / den if den > 0 else (np.inf if b - c - n * d0 > 0 else -np.inf if b - c - n * d0 < 0 else 0.0)

        def root(lo, hi, target):  # T decreases in d0
            for _ in range(200):
                mid = (lo + hi) / 2
                if T(mid) > target:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        est = (b - c) / n
        lo = -1.0 if T(-1 + 1e-12) <= z else root(-1 + 1e-12, est, z)
        hi = 1.0 if T(1 - 1e-12) >= -z else root(est, 1 - 1e-12, -z)
        return [float(lo), float(hi)]

    # toy checks before any data row
    t1 = tango(10, 10, 100)
    assert abs(t1[0] + t1[1]) < 1e-9, ("tango symmetry", t1)
    t0 = tango(0, 0, 100)
    assert abs(t0[0] + t0[1]) < 1e-9 and t0[1] > 0, ("tango zero discordant", t0)
    t2 = tango(15, 5, 100)
    assert t2[0] < 0.10 < t2[1], ("tango covers estimate", t2)
    zz = 1.959963984540054
    assert abs(t0[1] - zz * zz / (100 + zz * zz)) < 1e-9, ("tango zero-discordance closed form", t0)

    def paired_t(d):
        d = np.asarray(d, float)
        n, m, s = len(d), float(np.mean(d)), float(np.std(d, ddof=1))
        h = float(stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)) if s > 0 else 0.0
        tt = m / (s / np.sqrt(n)) if s > 0 else (0.0 if m == 0 else float(np.sign(m)) * np.inf)
        p = float(2 * stats.t.sf(abs(tt), n - 1)) if s > 0 else (1.0 if m == 0 else 0.0)
        return {"mean": m, "ci95": [m - h, m + h], "sd": s, "t": float(tt), "p": p}

    def cohort(m):
        has = lambda w: {u.split("|", 1)[1] for u, v in m.items() if u.startswith(w + "|") and v is not None and np.isfinite(v)}  # noqa: E731
        return sorted(t for t in has("run1") & has("run2") & has("sim") if all(labels.get(f"{w}|{t}") == "attempted" for w in ("run1", "run2", "sim")))

    gates, rows = [], []
    for P in probes:
        m, coh = P["scores"], cohort(P["scores"])
        ha = np.array([v for u, v in m.items() if u.startswith("honest_a|") and np.isfinite(v)])
        for a, g in P["gate"].items():
            t_rec = thr(ha, float(a))
            got = {"threshold": t_rec} | {f"catch_{w}": float(np.mean([m[f"{w}|{t}"] > g["threshold"] for t in coh])) for w in ("run1", "run2", "sim")}
            assert all(np.isfinite(v) for v in list(got.values()) + [g[k] for k in got]), ("nonfinite gate value", P["name"], P["base"], a)
            diffs = {k: abs(got[k] - g[k]) for k in got}
            gates.append({"probe": P["name"], "base": P["base"], "alpha": a, "cohort": len(coh), "stored_cohort": g.get("cohort_tasks"),
                          "max_abs_diff": max(diffs.values()), "diffs": diffs, "tolerance": P.get("gate_tol", {"threshold": 1e-9, "catch": 1e-9}),
                          "pass": diffs["threshold"] <= P.get("gate_tol", {}).get("threshold", 1e-9)
                          and max(v for k, v in diffs.items() if k != "threshold") <= 1e-9})
    if not all(g["pass"] for g in gates):
        return {"gates": gates, "gates_pass": False}

    for P in probes:
        m, coh = P["scores"], cohort(P["scores"])
        H = np.array([v for u, v in m.items() if u.startswith(("honest_a|", "honest_b|")) and np.isfinite(v)])
        HB = np.array([v for u, v in m.items() if u.startswith("honest_b|") and np.isfinite(v)])
        mu, sd = float(H.mean()), float(H.std(ddof=1))
        for a in (0.01, 0.05):
            T_ = thr(H, a)
            hit = {w: np.array([float(m[f"{w}|{t}"] > T_) for t in coh]) for w in ("run1", "run2", "sim")}
            z = {w: np.array([(m[f"{w}|{t}"] - mu) / sd for t in coh]) for w in ("run1", "run2", "sim")}
            row = {"probe": P["name"], "readout": P["readout"], "base": P["base"], "family": P["family"], "alpha": a, "n": len(coh),
                   "threshold_pooled": T_, "honest_pooled_n": int(len(H)), "honest_b_n": int(len(HB)),
                   "realized_fa_pooled_in_sample": float((H > T_).mean()), "realized_fa_honest_b": float((HB > T_).mean()),
                   "catch": {w: float(hit[w].mean()) for w in hit}, "z_ref": {"mean": mu, "sd": sd}}
            for name, x, y in (("sim_minus_run2", "sim", "run2"), ("run2_minus_run1", "run2", "run1")):
                d = hit[x] - hit[y]
                b, c = int((d > 0).sum()), int((d < 0).sum())
                row[name] = {"binary": paired_t(d) | {"x_only": b, "y_only": c, "discordant": b + c,
                                                      "mcnemar_p": float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0,
                                                      "tango95": tango(b, c, len(d))},
                             "continuous_z": paired_t(z[x] - z[y])}
            rows.append(row)
    # Amendment 3: the existing rows must equal the 115701 record exactly (same order, same values)
    same = [json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float) for a, b in zip(rows, prior_rows)]
    if len(prior_rows) > len(rows) or not all(same):
        return {"gates": gates, "gates_pass": False, "prior_rows_unchanged": False, "prior_mismatch_index": [i for i, ok in enumerate(same) if not ok]}
    return {"gates": gates, "gates_pass": True, "prior_rows_unchanged": True, "prior_rows_checked": len(prior_rows),
            "tango_toy": {"b=c=10,n=100": t1, "b=c=0,n=100": t0, "b=15,c=5,n=100": t2}, "rows": rows}


@app.local_entrypoint()
def main(labels: str):
    assert not subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, check=True).stdout, "dirty"
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    J = {}
    for k, (c, p, s) in FILES.items():
        b = git_bytes(c, p)
        assert hashlib.sha256(b).hexdigest() == s, (k, "sha256 mismatch")
        J[k] = json.loads(b)
    lb = Path(labels).read_bytes()
    assert hashlib.sha256(lb).hexdigest() == LABELS_SHA, "labels sha256 mismatch"
    lab = json.loads(lb)["labels"]

    def gate_of(cell):
        return {a: {"threshold": cell[f"at_{a}"]["threshold"], "catch_run1": cell[f"at_{a}"]["catch_run1"],
                    "catch_run2": cell[f"at_{a}"]["catch_run2"], "catch_sim": cell[f"at_{a}"]["catch_sim"], "cohort_tasks": cell.get("cohort_tasks")}
                for a in ("0.01", "0.05")}

    fam = {"a": "primary (Sol-trained; all B arms held out)", "b_ff": "repaired-trained (parked extension)", "c_ff": "repaired-trained (parked extension)",
           "b": "primary (Sol-trained on as-is sims; all B arms held out)", "c": "primary (Sol-trained on as-is sims; all B arms held out)",
           "own_a": "secondary (trained on B run1)"}
    probes = []
    for p in ("a", "own_a"):
        probes.append({"name": f"{p}", "readout": "averaged avg_top2", "base": E2, "family": fam[p], "scores": J["avg_e2"]["analysis"]["scores"]["avg_top2"][p],
                       "gate": gate_of(J["avg_e2"]["analysis"]["cells"]["avg_top2"][p])})
        probes.append({"name": f"{p}", "readout": "averaged avg_top2", "base": ALL, "family": fam[p], "scores": J["recheck"]["scores"][p],
                       "gate": gate_of(J["avg_all"]["analysis"]["cells"]["avg_top2"][p])})
    for p in ("a", "own_a", "b_ff", "c_ff"):
        for k, base in (("cmd_e2", E2), ("cmd_all", ALL)):
            probes.append({"name": p, "readout": "command A (content-only call_end, L63)", "base": base, "family": fam[p],
                           "scores": J[k]["run_scores"]["A_content_only_call_end"][p], "gate": gate_of(J[k]["cells"]["A_content_only_call_end"][p])})
    drop = set(J["e2"]["dropped_runs"])
    for p in ():  # Amendment 2: item 1 (b), (c) failed the gate (11:56 run) and are excluded; was ("b", "c"). Amendment 1 (Aswin via 0d): item 1's as-is (b), (c), last token L63; E2 = all-runs map minus E2 (Sol-trained, not refit)
        sc = J["item1"]["scores_L63"][p]
        probes.append({"name": p, "readout": "last token L63 (item 1 test models)", "base": E2, "family": fam[p],
                       "scores": {u: v for u, v in sc.items() if u not in drop}, "gate": gate_of(J["avg_e2"]["analysis"]["cells"]["last_L63"][p])})
        probes.append({"name": p, "readout": "last token L63 (item 1 test models)", "base": ALL, "family": fam[p],
                       "scores": sc, "gate": gate_of(J["avg_all"]["analysis"]["cells"]["last_L63"][p])})
    for col in COLS:
        probes.append({"name": col, "readout": "averaged avg_top2 (curve column)", "base": ALL,
                       "family": "secondary (Arm S: trained on sim)" if col.startswith("S_") else "secondary (Arm R: trained on run1)",
                       "scores": J["armrs"]["run_scores"][col], "gate": gate_of(J["armrs"]["cells"][col])})
    for p in ("a", "b", "c"):  # Amendment 3: item 1 at last token on the REPLAY layout, gated on item 1's own record
        probes.append({"name": f"item1_{p}", "readout": "last token L63 (replay)", "base": ALL, "family": REPLAY, "label": REPLAY,
                       "scores": J["item1"]["scores_L63"][p], "gate": gate_of(J["recipe"]["analysis"]["cells"]["full"]["output_end"][p]),
                       "gate_tol": {"threshold": 1e-6, "catch": 1e-9}})
    pb = (HERE / PRIOR).read_bytes()
    assert hashlib.sha256(pb).hexdigest() == "f84952ecd73fe8091a46ab251a64615720831be18ecd3d8a602e89cd4e0b4c46", "prior record sha256 mismatch"
    prior = json.loads(pb)["rows"]
    assert len(prior) == 64, ("prior rows", len(prior))
    out = compute.remote(probes, lab, prior)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    meta = {"git": git, "label": "PRELIMINARY first cut (RULE.md in this folder); stored scores only", "inputs": {k: {"commit": c, "path": p, "sha256": s} for k, (c, p, s) in FILES.items()},
            "labels_sha256": LABELS_SHA, "bases": {"primary": E2, "sensitivity": ALL}, "curve_note": "k10/k30 on the all-runs base only (E2 curve refit failed, dropped by 0d)",
            "not_applied": "Append 3 section 6 task_disjoint gate; bootstrap; TOST", "app": APP}
    dest = HERE / "results" / f"firstcut_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(meta | out, indent=1, default=float))
    if not out["gates_pass"]:
        print("GATE FAILED; no numbers. prior_rows_unchanged:", out.get("prior_rows_unchanged"), "mismatch index:", out.get("prior_mismatch_index"), "Failing gates:")
        for g in out["gates"]:
            if not g["pass"]:
                print(json.dumps(g))
        raise SystemExit(1)
    pct = lambda x: f"{100 * x:6.2f}"  # noqa: E731
    ci = lambda v: f"[{100 * v[0]:+6.2f}, {100 * v[1]:+6.2f}]"  # noqa: E731
    lines = [f"git {git}; gates pass ({len(out['gates'])} gates, max |diff| {max(g['max_abs_diff'] for g in out['gates']):.1e}); tango toy {json.dumps(out['tango_toy'])}",
             "Threshold FIXED on pooled honest A u B per probe and base (calibrated to target alpha); hits strictly above. Points are percentage points.",
             "probe | readout | base | alpha | N | catch run1/run2/sim | d(sim-run2) [paired-t 95%] | Tango 95% | sim-only/real-only | McNemar p | z-diff [95%] "
             "|| d(run2-run1) [paired-t 95%] | Tango 95% | run2-only/run1-only | McNemar p | z-diff [95%] || realized FA pooled(in-sample)/honest B"]
    for r in out["rows"]:
        s, q = r["sim_minus_run2"], r["run2_minus_run1"]
        lines.append(f"{r['probe']:10} | {r['readout'][:24]:24} | {'E2' if r['base'] == E2 else 'ALL':3} | {r['alpha']:.0%} | {r['n']} | "
                     f"{pct(r['catch']['run1'])}/{pct(r['catch']['run2'])}/{pct(r['catch']['sim'])} | "
                     f"{100 * s['binary']['mean']:+6.2f} {ci(s['binary']['ci95'])} | {ci(s['binary']['tango95'])} | {s['binary']['x_only']}/{s['binary']['y_only']} | "
                     f"{s['binary']['mcnemar_p']:.3f} | {s['continuous_z']['mean']:+.3f} [{s['continuous_z']['ci95'][0]:+.3f}, {s['continuous_z']['ci95'][1]:+.3f}] || "
                     f"{100 * q['binary']['mean']:+6.2f} {ci(q['binary']['ci95'])} | {ci(q['binary']['tango95'])} | {q['binary']['x_only']}/{q['binary']['y_only']} | "
                     f"{q['binary']['mcnemar_p']:.3f} | {q['continuous_z']['mean']:+.3f} [{q['continuous_z']['ci95'][0]:+.3f}, {q['continuous_z']['ci95'][1]:+.3f}] || "
                     f"{pct(r['realized_fa_pooled_in_sample'])}/{pct(r['realized_fa_honest_b'])}" + (f" || {r['family']}" if r["family"] == REPLAY else ""))
    (HERE / "results" / f"firstcut_{stamp}.printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
