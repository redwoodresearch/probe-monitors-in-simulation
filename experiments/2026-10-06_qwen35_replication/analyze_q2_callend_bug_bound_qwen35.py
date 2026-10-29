"""How much could the #47 call_end bug move the committed Qwen3.5 Q2 numbers (2026-10-07)?

The bug left 305 calls per five-arm text with no call_end (their last XML argument has no token of its own; fixed in
6c6fc56, rerun pending). Q2 needs all three readouts, so each such call is missing from every readout, and so is the
next call of the same turn, whose prev_content is that call_end. This counts those calls per Q2 view and domain (by
label, under the view's task rules), and bounds each committed headline gap if they were added back: every dropped
impactful or read-only call flagged, or none (recall and FPR), and ranked at either extreme (AUROC). The bound on a
gap (a probe's estimate gap: test on the other domain minus test at home) is set against that gap's 95% half-width. The dropped calls were
never scored, so their place in the calibration tail is unknown; instead this reports how many read-only calls each
domain would gain, which bounds how far the 1% threshold's rank can move. Saved data only.

What it bounds (Codex): a composition bound with the committed probes and frozen thresholds held fixed. Adding the
calls back would also retrain the probes, re-pick layers and C, and recalibrate the thresholds; those effects are not
bounded here. The added calls are a superset of what each base would regain: the arm, task and run rules of each view
apply, but the matched base's 32k-cut rule and any other missing readout do not, so matched counts equal whole ones.

  uv run --with pandas python experiments/2026-10-06_qwen35_replication/analyze_q2_callend_bug_bound_qwen35.py
"""

import argparse
import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
AUDIT = HERE / "results" / "callend_fix_positions_2026-10-07_110722.json.gz"
RESULTS = {"matched": HERE / "results" / "q2_q2_threeway_q2_alllayers_qwen35_matched_2026-10-06_235327.json",
           "whole": HERE / "results" / "q2_q2_threeway_q2_alllayers_qwen35_whole_2026-10-07_002445.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / ".run_cache" / "q2_call_labels" / "2026-10-06_212047.jsonl"))
    a = ap.parse_args()
    audit = json.loads(gzip.decompress(AUDIT.read_bytes()))
    v4 = next(t for t in audit["texts"] if t["text"] == "v4")
    bug = {(r["arm"], r["key"], c["step"], c["action"]) for r in v4["rows"] for c in r["changed"]}
    labels = {(r["arm"], r["key"], r["step"], r["action"]): r for r in map(json.loads, Path(a.labels).read_text().splitlines())}
    after = {(arm, key, step, action + 1) for arm, key, step, action in bug if (arm, key, step, action + 1) in labels}
    dropped = (bug | after) & set(labels)  # the calls Q2 lost: the call itself and the next call of its turn
    out = {"bug_calls": len(bug), "next_calls_in_turn": len(after - bug), "dropped_labeled_calls": len(dropped), "bases": {}}
    for base, path in RESULTS.items():
        res = json.loads(path.read_text())
        cfg = res["config"]
        honest, primary = set(cfg["honest_tasks"]), set(cfg["primary_real_honest"])
        rows = {}
        for view, v in cfg["views"].items():
            k = defaultdict(lambda: {"impactful": 0, "read_only": 0})
            for c in dropped:
                r = labels[c]
                dom = v["domains"].get(r["arm"])
                if dom is None:
                    continue
                honest_arm = r["arm"] in ("real-honest", cfg["extra_arm"])
                if honest_arm and r["main_task"] not in honest:
                    continue
                if r["arm"] == "real-honest" and v.get("one_run") and r["key"] not in primary:
                    continue
                if not honest_arm and not r[cfg["tasks"]]:  # attack calls: the paired attack tasks
                    continue
                k[dom]["impactful" if r["label"] == "impactful" else "read_only"] += 1
            for pos, e in res["views"][view]["positions"].items():
                h = e["headline"]
                for t in cfg["train_sets"]:
                    rng = {}
                    for dom in cfg["test_domains"]:
                        x = h[f"train {t} / test {dom}"]
                        Np, Nn = x["n"]["impactful"], x["n"]["read_only"]
                        kp, kn = k[dom]["impactful"], k[dom]["read_only"]
                        r_, f_, A = x["recall"]["value"], x["fpr"]["value"], x["auroc"]["value"]
                        rng[dom] = {"recall": (r_ * Np / (Np + kp), (r_ * Np + kp) / (Np + kp)),
                                    "fpr": (f_ * Nn / (Nn + kn), (f_ * Nn + kn) / (Nn + kn)),
                                    "auroc": (A * Np * Nn / ((Np + kp) * (Nn + kn)),
                                              (A * Np * Nn + kp * Nn + kn * Np + kp * kn) / ((Np + kp) * (Nn + kn))),
                                    "point": {"recall": r_, "fpr": f_, "auroc": A}, "added": {"impactful": kp, "read_only": kn},
                                    "n": {"impactful": Np, "read_only": Nn}}
                    home = cfg["home"][t]  # the reported estimate gap: test on the other domain minus test at home
                    other = next(d for d in cfg["test_domains"] if d != home)
                    gname = f"train {t} / test {other} minus test {home}"
                    g = h[gname]
                    for m in ("recall", "fpr", "auroc"):
                        lo = rng[other][m][0] - rng[home][m][1]
                        hi = rng[other][m][1] - rng[home][m][0]
                        point = rng[other]["point"][m] - rng[home]["point"][m]
                        half = (g[m]["ci95"][1] - g[m]["ci95"][0]) / 2
                        move = max(point - lo, hi - point)
                        rows[f"{view} {pos} {gname} {m}"] = {"reported": g[m]["value"], "ci_half_width": round(half, 4),
                                                             "worst_case_move": round(move, 4), "exceeds_ci_half": move > half,
                                                             "added_real": rng["real"]["added"], "added_sim": rng["sim"]["added"]}
        out["bases"][base] = rows
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q2_callend_bug_bound_qwen35_{stamp}.json"
    path.write_text(json.dumps({"bound": "composition only: committed probes and frozen thresholds fixed; retraining, layer and "
                                         "C selection and recalibration are not bounded. Added calls: a superset (view task and run "
                                         "rules applied; the matched 32k-cut rule and other missing readouts not)",
                                "audit": str(AUDIT.relative_to(HERE)), "labels": a.labels,
                                "results": {b: str(p.relative_to(HERE)) for b, p in RESULTS.items()}, **out}, indent=1))
    print({k: out[k] for k in ("bug_calls", "next_calls_in_turn", "dropped_labeled_calls")})
    for base, rows in out["bases"].items():
        ex = [k for k, v in rows.items() if v["exceeds_ci_half"]]
        worst = max(rows.items(), key=lambda kv: kv[1]["worst_case_move"] / max(kv[1]["ci_half_width"], 1e-9))
        print(f"{base}: {len(rows)} gaps; {len(ex)} with a worst-case move above the 95% half-width; "
              f"largest move/half-width: {worst[0]} {worst[1]['worst_case_move']} vs {worst[1]['ci_half_width']}")
        for k_ in ex[:12]:
            print(f"   exceeds: {k_}: move {rows[k_]['worst_case_move']} > half {rows[k_]['ci_half_width']} (reported {rows[k_]['reported']})")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
