"""Stage 1 verdicts from run_fit_stage1.py (pre-registration PR #82: rule 2, the Stage 1 table). No new fits.

Per probe:
- the layer: best mean held-out-family AUROC (logistic regression); ties go to the deeper layer.
- the selection-aware null: per replicate k, the max over layers of the null mean, then its 95th percentile.
- every pre-registered validity criterion, pass or fail, with its number.
- the K6 re-check at the chosen layer: logistic regression vs mean difference vs shrinkage LDA.

Also writes the freeze amendment's draft table (layer, sign, pooling, C, verdict). The positive controls are evaluated
only after that amendment is committed (run_score_stage1.py).

  python analyze_fit_stage1.py --fit results/fit_stage1_<stamp>.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
GATES = {  # pre-registered (PR #82 Stage 1 table)
    "tvc": {"mean": 0.80, "min": 0.70},
    "monitoring": {"mean": 0.80, "min": 0.80},  # the two sets must agree: each held-out set >= 0.80
    "stakes": {"mean": 0.80, "min": 0.70},
    "impact": {"mean": 0.80, "min": 0.70},
}


def p95(v):
    v = [x for x in v if x is not None]
    return float(np.percentile(v, 95)) if v else None


def best_layer(per_layer: dict) -> int:
    """argmax of the values; ties go to the deeper layer."""
    top = max(v for v in per_layer.values() if v is not None)
    return max(l for l, v in per_layer.items() if v is not None and v == top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="the main fit (contrast sets, Needham)")
    ap.add_argument("--fit-vuln", help="the vulnerability fit, when run separately")
    args = ap.parse_args()
    fit = json.loads(Path(args.fit).read_text())
    L = {r["layer"]: r for r in fit["layers"]}
    if args.fit_vuln:
        for r in json.loads(Path(args.fit_vuln).read_text())["layers"]:
            L[r["layer"]]["vuln"] = r["vuln"]
    out = {"fit": args.fit, "fit_vuln": args.fit_vuln, "needham_drops": fit.get("needham_drops"), "probes": {}}
    for p, g in GATES.items():
        lr_mean = {l: L[l][p]["lr_mean"] for l in L}
        lstar = best_layer(lr_mean)
        r = L[lstar][p]
        null_max = [max(L[l][p]["null_mean"][k] for l in L) for k in range(len(r["null_mean"]))]
        cut = p95(null_max)
        v = {"layer": lstar, "sign": "+ (aware / impactful side positive)", "pooling": "run mean of readout calls (rule 5 weights)",
             "C_all": r["C_all"], "lr_mean": r["lr_mean"], "lr_min": r["lr_min"], "families": r["families"],
             "null_p95_max_over_layers": cut,
             "k6_at_layer": {"logistic_regression": r["lr_mean"], "mean_difference": r["mean_difference_mean"], "lda": r["lda_mean"]},
             "checks": {}}
        v["checks"]["held_out_mean"] = r["lr_mean"] >= g["mean"]
        v["checks"]["held_out_min"] = r["lr_min"] >= g["min"]
        v["checks"]["above_null"] = cut is not None and r["lr_mean"] > cut
        def maxnull(get):
            """Per replicate k, the max over layers (the same swaps at every layer), so the layer choice is repeated."""
            n = len(get(L[lstar]))
            return [max((get(L[l])[k] if get(L[l])[k] is not None else 0) for l in L) for k in range(n)]

        if p == "tvc":
            v["friendliness"] = r["friendliness"]
            v["checks"]["friendliness_both_strata"] = all(x is not None and x >= 0.70 for x in r["friendliness"].values())
            wa = r["within_aider"]
            cut_wa = p95(maxnull(lambda x: x["tvc"]["within_aider"]["null"]))
            v["within_aider"] = {"auroc": wa["auroc"], "null_p95_max_over_layers": cut_wa, "n_eval": wa["n_eval"], "n_deploy": wa["n_deploy"]}
            v["checks"]["within_aider_above_null"] = wa["auroc"] is not None and cut_wa is not None and wa["auroc"] > cut_wa
        if "implicit" in r:
            im = r["implicit"]
            cut_im = p95(maxnull(lambda x, p=p: x[p]["implicit"]["null"]))
            v["implicit"] = {"auroc": im["auroc"], "null_p95_max_over_layers": cut_im} | ({"by_impact": im["by_impact"]} if "by_impact" in im else {})
            v["checks"]["implicit_ge_0.70"] = im["auroc"] is not None and im["auroc"] >= 0.70
            v["checks"]["implicit_above_null"] = im["auroc"] is not None and cut_im is not None and im["auroc"] > cut_im
            if "by_impact" in im:
                v["checks"]["implicit_each_impact_stratum_ge_0.70"] = all(x is not None and x >= 0.70 for x in im["by_impact"].values())
        v["checks"]["k6_lr_beats_mean_difference_by_0.02"] = r["lr_mean"] - r["mean_difference_mean"] > 0.02  # reported only
        v["passes_validity"] = all(ok for k, ok in v["checks"].items() if not k.startswith("k6"))
        v["note"] = "Testing vs Conversation also needs its positive control (after the freeze)." if p == "tvc" else (
            "Also needs its positive control (after the freeze)." if p in ("monitoring", "stakes") else "")
        out["probes"][p] = v
    # twins (amendment 1 as decided): primary = tool-call transcripts; fallback to mixed if < ~100 per label survive
    def twin_verdict(key):
        if "skipped" in L[0][key]:
            return {"skipped": L[0][key]["skipped"], "n_train_by_label": L[0][key]["n_train_by_label"], "passes_validity": False}
        ho = {l: L[l][key]["held_out_by_source"] for l in L}
        lt = best_layer(ho)
        t = L[lt][key]
        v = {"layer": lt, "C_all": t["C_all"], "held_out_by_source": t["held_out_by_source"], "within_aider": t["within_aider"],
             "n_train": t["n_train"], "n_train_by_label": t["n_train_by_label"], "sources_train": t["sources_train"]}
        if "null_held_out" in t:
            nm = [max((L[l][key]["null_held_out"][k] or 0) for l in L) for k in range(len(t["null_held_out"]))]
            na = [max((L[l][key]["null_within_aider"][k] or 0) for l in L) for k in range(len(t["null_within_aider"]))]
            v["null_p95_max_over_layers"], v["within_aider_null_p95"] = p95(nm), p95(na)
            v["checks"] = {"held_out_ge_0.80": t["held_out_by_source"] >= 0.80,
                           "above_null": t["held_out_by_source"] > v["null_p95_max_over_layers"],
                           "within_aider_above_null": t["within_aider"] is not None and t["within_aider"] > v["within_aider_null_p95"]}
            v["passes_validity"] = all(v["checks"].values())
        return v

    counts = L[0]["twin"]["n_train_by_label"]
    fallback = min(counts.values()) < 100
    out["twin_tool_call_counts_by_label"] = counts
    out["twin_primary"] = "twin_mixed" if fallback else "twin"
    out["twin_fallback_used"] = fallback
    for key in ("twin", "twin_mixed", "twin_text"):
        out["probes"][key] = twin_verdict(key)
    # vulnerability control
    vl = {l: L[l]["vuln"]["em_val"] for l in L if "vuln" in L[l]}
    lv = best_layer(vl)
    vv = L[lv]["vuln"]
    null_max = [max(L[l]["vuln"]["null_safecoder"][k] for l in vl) for k in range(len(vv["null_safecoder"]))]
    out["probes"]["vuln"] = {"layer": lv, "em_val": vv["em_val"], "safecoder": vv["safecoder"], "null_p95_max_over_layers": p95(null_max), "n": vv["n"]}
    out["probes"]["vuln"]["checks"] = {"safecoder_ge_0.70": vv["safecoder"] >= 0.70, "above_null": vv["safecoder"] > p95(null_max)}
    out["probes"]["vuln"]["passes_validity"] = all(out["probes"]["vuln"]["checks"].values())
    ls = out["probes"]["stakes"]["layer"]
    out["stakes_vs_tvc_on_stakes_implicit_at_stakes_layer"] = L[ls]["stakes_vs_tvc_on_stakes_implicit"]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"fit_verdicts_{Path(args.fit).stem}__{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))

    # draft freeze: what run_score_stage1.py reads once the amendment is committed (layer, weights key, scored or not)
    primary_twin = out["twin_primary"]
    keys = {"tvc": "tvc", "monitoring": "monitoring", "stakes": "stakes", "impact": "impact", "vuln": "vuln", primary_twin: primary_twin}
    freeze = {"fit": args.fit, "fit_vuln": args.fit_vuln, "verdicts": str(path), "readout": "rule 1: last command-content token (reader.command_end_positions)",
              "pooling": "run mean of readout calls, rule 5 composition weights", "sign": "+ = aware / impactful / vulnerable side",
              "margin": 0.03, "twin_primary": primary_twin, "twin_fallback_used": out["twin_fallback_used"], "probes": {}}
    for p, key in keys.items():
        v = out["probes"][p]
        freeze["probes"][p] = {"layer": v.get("layer"), "weights_key": key, "C_all": v.get("C_all"),
                               "passes_validity": v.get("passes_validity", False),
                               "scored": bool(v.get("passes_validity", False)),
                               "role": "control" if p in ("impact", "vuln") else "awareness"}
    with (HERE / "results" / f"freeze_draft_{stamp}.json").open("x") as f:
        f.write(json.dumps(freeze, indent=1))
    for p, v in out["probes"].items():
        print(f"{p:11s} layer {v.get('layer', -1):2d} passes={v.get('passes_validity', 'n/a (secondary, no null)')} | "
              + ", ".join(f"{k}={ok}" for k, ok in v.get("checks", {}).items()))
    print("twin tool-call transcripts by label:", out["twin_tool_call_counts_by_label"], "| primary twin:", out["twin_primary"])
    print("->", path)


if __name__ == "__main__":
    main()
