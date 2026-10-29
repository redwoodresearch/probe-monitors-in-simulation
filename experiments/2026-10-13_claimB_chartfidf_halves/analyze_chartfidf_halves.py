"""POST HOC: apply RULE.md (this folder) to run_chartfidf_halves_modal.py's run scores; numbers only. Per variant, the
threshold, shares, draws, pooling, planted gate and reading are analyze_tfidf_halves.py's @ faafd24c, unchanged; added:
the variants, the paired-draw interval for character-only 2-5 minus word + character (0d's "(A)"), and the arm-W change.
The probe's numbers are read from halves_e2_direct_POSTHOC_2026-10-11_172057.json @ 53da7f48, the probe's arm W from
halves_splitW_2026-10-12_143656.json @ e064ed40, and faafd24c's word + character result from its committed JSON.

  uv run --no-project --python 3.12 --with numpy --with scikit-learn python \\
      experiments/2026-10-13_claimB_chartfidf_halves/analyze_chartfidf_halves.py data/chartfidf_halves_run_scores_<stamp>.json.gz
"""

import gzip
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
SIM, E1, E2, PL2 = "sim-attack", "real-attack-e1", "real-attack-e2", "sim-attack-plant2"
DIRS = ("dev_to_test", "test_to_dev")
DRAWS, SEED, ALARM_MARGIN, PLANT_FLOOR = 2000, 20261011, 0.10, 0.910  # analyze_halves.py's
VARIANTS = ("wordchar", "char1_5", "char2_5", "char1_5_W")
PROBE = ("53da7f48", "experiments/2026-10-11_claimB_halves/results/halves_e2_direct_POSTHOC_2026-10-11_172057.json")
PROBE_W = ("e064ed40", "experiments/2026-10-12_claimB_halves_mask_split/results/halves_splitW_2026-10-12_143656.json")
WORDCHAR = ("faafd24c", "experiments/2026-10-12_claimB_tfidf_halves/results/tfidf_halves_POSTHOC_2026-10-12_112423.json")
N_HELD = {"dev_to_test": 473, "test_to_dev": 466}  # held-out cells per direction (test half, dev half)


def thr(neg):  # analyze_halves.py
    s = np.sort(neg)[::-1]
    return s[int(np.floor(0.01 * len(s)))]


def share_above(pos, t):  # analyze_halves.py
    return float((pos > t).mean())


def git_json(commit, path):
    return json.loads(subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, text=True, check=True, cwd=HERE).stdout)


def main(path):
    raw = Path(path).read_bytes()
    res = json.loads(gzip.decompress(raw))
    for g in ("gate", "unit_checks", "gate2"):
        if not res.get(g, {}).get("passed"):
            raise SystemExit(f"{g} failed or missing: no numbers (RULE.md)")
    recs = res["records"]

    def arr(variant, comp, d):
        rr = [r for r in recs if r["variant"] == variant and r["comparison"] == comp and r["direction"] == d]
        return (np.array([r["score"] for r in rr]), np.array([r["arm"] for r in rr]), np.array([r["main_task"] for r in rr]))

    # the draws: analyze_tfidf_halves.py's, made once from the held-out half's three arms' main tasks, sorted
    fx0 = {d: arr("wordchar", "fixed_vs_e1", d) for d in DIRS}
    rng = np.random.default_rng(SEED)
    halves = {d: sorted(set(fx0[d][2][np.isin(fx0[d][1], [SIM, E1, E2])])) for d in DIRS}
    draws = {d: [rng.integers(0, len(halves[d]), len(halves[d])) for _ in range(DRAWS)] for d in DIRS}

    def analyse(variant):
        fx = {d: arr(variant, "fixed_vs_e1", d) for d in DIRS}
        for d in DIRS:  # every variant scores the same runs, so the draws apply unchanged
            if sorted(set(fx[d][2][np.isin(fx[d][1], [SIM, E1, E2])])) != halves[d]:
                raise SystemExit(f"{variant} {d}: main tasks differ from the draws'")

        def shares(d, dr=None):  # analyze_POSTHOC_halves_e2.py's shares()
            s, a, t = fx[d]
            if dr is not None:
                pos = {x: j for j, x in enumerate(halves[d])}
                w = np.bincount(dr, minlength=len(halves[d]))[[pos[x] for x in t]]
                s, a = np.repeat(s, w), np.repeat(a, w)
            th = thr(s[a == E1])
            return {"sim": share_above(s[a == SIM], th), "e2": share_above(s[a == E2], th), "e1_fpr": share_above(s[a == E1], th),
                    "n_sim": int((a == SIM).sum()), "n_e2": int((a == E2).sum()), "n_e1": int((a == E1).sum()),
                    "sim_flagged": int((s[a == SIM] > th).sum()), "e2_flagged": int((s[a == E2] > th).sum()), "e1_flagged": int((s[a == E1] > th).sum())}

        out = {}
        for d in DIRS:
            s, a, _ = fx[d]
            s0 = shares(d)
            g = np.array([(lambda x: x["sim"] - x["e2"])(shares(d, dr)) for dr in draws[d]])
            ps, pa, _ = arr(variant, "plant2", d)
            es, ea, _ = arr(variant, "e2_vs_e1", d)
            th2 = thr(es[ea == E1])
            plant_auc = float(roc_auc_score(pa == PL2, ps))
            out[d] = {
                "plant2_auroc": plant_auc, "plant2_pairs": int((pa == PL2).sum()), "planted_gate_passed": plant_auc >= PLANT_FLOOR,
                "catch_1pct": s0 | {"gap": s0["sim"] - s0["e2"], "gap_ci95": [float(x) for x in np.percentile(g, [2.5, 97.5])]},
                "auroc": {"fixed_vs_e1": float(roc_auc_score(a[np.isin(a, [SIM, E1])] == SIM, s[np.isin(a, [SIM, E1])])),
                          "e2_vs_e1_direct": float(roc_auc_score(a[np.isin(a, [E2, E1])] == E2, s[np.isin(a, [E2, E1])]))},
                "separate_yardstick": {"e2_flagged_by_e2_detector": share_above(es[ea == E2], th2), "n_e2": int((ea == E2).sum()),
                                       "e2_vs_e1_auroc": float(roc_auc_score(ea == E2, es))}}
        xs = [shares(d) for d in DIRS]
        ns, ne = sum(x["n_sim"] for x in xs), sum(x["n_e2"] for x in xs)
        psim, pe2 = sum(x["sim_flagged"] for x in xs) / ns, sum(x["e2_flagged"] for x in xs) / ne

        def pooled(b):
            ys = [shares(d, draws[d][b]) for d in DIRS]
            return sum(y["sim"] * y["n_sim"] for y in ys) / sum(y["n_sim"] for y in ys) - sum(y["e2"] * y["n_e2"] for y in ys) / sum(y["n_e2"] for y in ys)

        pg = np.array([pooled(b) for b in range(DRAWS)])
        lo, hi = (float(x) for x in np.percentile(pg, [2.5, 97.5]))
        both = all(out[d]["planted_gate_passed"] for d in DIRS)
        reading = ("TF-IDF flags simulated runs more than the second real run" if lo > 0 else
                   "no excess detected; upper bound below 10 percentage points" if lo <= 0 <= hi < ALARM_MARGIN else "unclear")
        out["pooled"] = {"sim_flagged": sum(x["sim_flagged"] for x in xs), "n_sim": ns, "e2_flagged": sum(x["e2_flagged"] for x in xs), "n_e2": ne,
                         "sim": psim, "e2": pe2, "gap": psim - pe2, "gap_ci95": [lo, hi], "planted_gate_both": both,
                         "reading": reading if both else f"descriptive only (a planted gate failed): {reading}"}
        return out, pg

    sig = {v: sorted((r["direction"], r["comparison"], r["key"], r["arm"], r["cell"], r["main_task"], r["rows"]) for r in recs if r["variant"] == v)
           for v in VARIANTS}
    if any(sig[v] != sig["wordchar"] for v in VARIANTS) or len(set(map(tuple, sig["wordchar"]))) != len(sig["wordchar"]):
        raise SystemExit("variants do not score the same runs with the same metadata, once each (Codex, 10-13)")
    results, per_draw = {}, {}
    for v in VARIANTS:
        results[v], per_draw[v] = analyse(v)

    # check: the refit word + character variant reproduces faafd24c's committed pooled numbers and interval
    wc = git_json(*WORDCHAR)["results"]["pooled"]
    repro = {k: results["wordchar"]["pooled"][k] == wc[k] for k in ("sim_flagged", "e2_flagged", "gap", "gap_ci95", "reading")}
    if not all(repro.values()):
        raise SystemExit(f"word + character pooled numbers differ from faafd24c's committed result: {repro}")

    # 0d's "(A)": character-only 2-5 minus word + character, paired on the same draws (descriptive)
    diff = per_draw["char2_5"] - per_draw["wordchar"]
    paired = {"point": results["char2_5"]["pooled"]["gap"] - results["wordchar"]["pooled"]["gap"],
              "ci95": [float(x) for x in np.percentile(diff, [2.5, 97.5])], "draws": DRAWS,
              "label": "descriptive; pooled sim-minus-e2 gap, character-only 2-5 minus word + character, same main-task draws"}
    arm_w_change = {"point": results["char1_5_W"]["pooled"]["gap"] - results["char1_5"]["pooled"]["gap"],
                    "label": "descriptive, no interval (RULE.md); character-only 1-5, arm W minus unmasked"}

    probe = git_json(*PROBE)["results"]
    pw = git_json(*PROBE_W)["primary"]
    side = {}
    for d in DIRS + ("pooled",):
        n = N_HELD[d] if d in N_HELD else sum(N_HELD.values())
        p = probe[d]
        row = {"probe": {"sim": f"{round(p['sim_catch'] * n)}/{n}", "e2_direct": f"{round(p['direct']['e2_flagged_by_sim_detector'] * n)}/{n}",
                         "gap": p["direct"]["gap"], "gap_ci95": p["direct"]["gap_ci95"]}}
        for v in VARIANTS:
            t = results[v][d]
            tf = t["catch_1pct"] if d in DIRS else t
            ok = t["planted_gate_passed"] if d in DIRS else t["planted_gate_both"]
            row[v] = {"sim": f"{tf['sim_flagged']}/{tf['n_sim']}", "e2_direct": f"{tf['e2_flagged']}/{tf['n_e2']}", "gap": tf["gap"],
                      "gap_ci95": tf["gap_ci95"], "planted_gate": ok, "label": "read" if ok else "descriptive only (planted gate failed)"}
        side[d] = row
    pwr = git_json(*PROBE_W)["results"]
    for d in DIRS + ("pooled",):  # the probe's arm W, flagged/total per direction and pooled (Codex, 10-13)
        n = N_HELD[d] if d in N_HELD else sum(N_HELD.values())
        c = pwr[d]["catch_1pct"]
        side[d]["probe_arm_W"] = {"sim": f"{round(c['sim_catch'] * n)}/{n}", "e2_direct": f"{round(c['direct_e2_flagged'] * n)}/{n}",
                                  "gap": c["direct_gap"], "gap_ci95": c.get("direct_gap_ci95")}
    side["pooled"]["probe_arm_W"] |= {"reading": pw["reading"], "planted_both_pass": pw["planted_both_pass"]}
    out = {"label": "POST HOC (RULE.md): character-only TF-IDF through the halves design; descriptive, preliminary until reviewed",
           "gate": res["gate"], "unit_checks": res["unit_checks"], "gate2": res["gate2"], "wordchar_reproduces_faafd24c": repro,
           "counts": res["counts"], "results": results, "paired_char2_5_minus_wordchar": paired, "arm_W_change_char1_5": arm_w_change,
           "side_by_side": side | {"sources": {"probe": f"{PROBE[1]} @ {PROBE[0]}", "probe_arm_W": f"{PROBE_W[1]} @ {PROBE_W[0]}",
                                               "wordchar_committed": f"{WORDCHAR[1]} @ {WORDCHAR[0]}"}},
           "draws": DRAWS, "seed": SEED, "inputs": {Path(path).name: hashlib.sha256(raw).hexdigest()},
           "runner_git": res.get("provenance", {}).get("launch_git"), "runner_sha256": res.get("script_sha256"), "code_sha256": res.get("code_sha256"),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"chartfidf_halves_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({v: {k: results[v]["pooled"][k] for k in ("sim_flagged", "e2_flagged", "gap", "gap_ci95", "reading")} for v in VARIANTS}, indent=1))
    print(json.dumps({v: {d: round(results[v][d]["plant2_auroc"], 4) for d in DIRS} for v in VARIANTS}))
    print("paired char2_5 - wordchar:", json.dumps(paired))
    print("arm W change (char1_5):", json.dumps(arm_w_change))
    print("->", dst)


if __name__ == "__main__":
    main(sys.argv[1])
