"""Ties at the 1% threshold, 196-task monitor (POST HOC; rule in RULE_POSTHOC_ties_196.md, committed first). $0, stored
draw 0 only: per fold and pooled, the calibration set's size, target, threshold, achieved false alarms and ties at the
threshold, what random tie-breaking would change, and the same frozen thresholds on the held-out test read-only calls
of real run 1, simulated and real run 2.

Amendment 1: scores come from the by-layer run's full-precision layers_draw0.npz at each fold's chosen layer; the stored
draw 0 (6-decimal scores) supplies thresholds, flags and the stored rates, and its rounded ties are reported beside.

  uv run --no-project --with numpy python analyze_POSTHOC_ties_196.py <draws0-0 file> <layers draws file> <layers_draw0.npz>
"""

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from analyze_scaleup_plants import threshold

HERE = Path(__file__).parent
SHA = "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf"
SHA_LAYERS = "33198a5e"  # prefixes, from results/manifest_catch_by_layer_v1.json
SHA_NPZ = "c4bac187"
WORLDS = {"real": "real->real", "sim": "real->sim", "run2": "real->run2"}


def main(path: str, layers_draws: str, npz: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise SystemExit("input sha256 mismatch")
    D = json.loads(raw)
    meta, folds = D["meta"], D["folds"]
    lraw, zraw = Path(layers_draws).read_bytes(), Path(npz).read_bytes()
    if not hashlib.sha256(lraw).hexdigest().startswith(SHA_LAYERS) or not hashlib.sha256(zraw).hexdigest().startswith(SHA_NPZ):
        raise SystemExit("by-layer input sha256 mismatch")
    L = json.loads(lraw)
    if L["meta"] != meta or L["folds"] != folds:
        raise SystemExit("the by-layer run's rows differ from the stored draw 0")
    Z = np.load(npz, allow_pickle=False)
    world = np.array([m["world"] for m in meta])
    yy = np.array([m["y"] for m in meta])
    fold_of_row = np.array([folds[m["task"]] for m in meta])
    cells = D["draws"][0]["cells"]
    cal = cells["real->real"]["per_call"]["calibration"]
    rows, score, mf = np.array(cal["row"]), np.array(cal["score"]), np.array(cal["model_fold"])
    keep = np.array([meta[i]["world"] == "real" and meta[i]["y"] == 0 for i in rows])
    stored = {d["fold"]: d for d in cells["real->real"]["folds"] if d["threshold_kind"] == "frozen"}
    out = {"git": git, "input": Path(path).name, "input_sha256": SHA, "layers_input": [Path(layers_draws).name, Path(npz).name],
           "rule": "RULE_POSTHOC_ties_196.md (amendment 1)", "folds": [], "pooled": {}}
    T = {}
    agg = {"n": 0, "flag": 0, "eq": 0, "tb": 0, "target": 0, "e9": 0, "e6": 0, "r6": 0}
    for k in range(5):
        m = keep & (mf == k)
        assert all(folds[meta[i]["task"]] == (k + 1) % 5 for i in rows[m]), "calibration rows outside fold k+1"
        s_rounded = score[m]
        zr, zs = Z[f"real|{k}|rows"], Z[f"real|{k}|scores"][stored[k]["layer"]]
        full = dict(zip(zr.tolist(), zs.tolist()))
        crow = np.flatnonzero((world == "real") & (yy == 0) & (fold_of_row == (k + 1) % 5))
        if sorted(crow.tolist()) != sorted(rows[m].tolist()):
            raise SystemExit(f"fold {k}: calibration rows differ between the two inputs")
        s = np.array([full[i] for i in rows[m]])
        if np.abs(np.round(s, 6) - s_rounded).max() > 1.5e-6:
            raise SystemExit(f"fold {k}: full-precision scores don't round to the stored ones")
        n, t = len(s), threshold(s, np.ones(len(s)))
        if t != stored[k]["threshold"] or n != stored[k]["calibration_read_only_calls"]:
            raise SystemExit(f"fold {k}: gate failed (t {t} vs {stored[k]['threshold']}, n {n} vs {stored[k]['calibration_read_only_calls']})")
        T[k] = t
        target = math.floor(0.01 * n + 1e-9)
        gt, eq = int((s > t).sum()), int((s == t).sum())
        tb = target if gt <= target <= gt + eq else gt
        f = {"fold": k, "calibration_fold": (k + 1) % 5, "layer": stored[k]["layer"], "n": n, "target_k": target, "threshold": t,
             "flagged": gt, "achieved_false_alarm": gt / n, "tied_exact": eq, "tied_1e-9": int((abs(s - t) <= 1e-9).sum()),
             "tied_1e-6": int((abs(s - t) <= 1e-6).sum()), "at_or_above": int((s >= t).sum()),
             "stored_tail_calls_at_or_above": stored[k]["tail_calls_at_or_above"],
             "random_tie_break_flagged": tb, "random_tie_break_false_alarm": tb / n,
             "stored_6dp_tied_at_rounded_t": int((s_rounded == round(t, 6)).sum()), "test": {}}
        for v, cell in WORLDS.items():
            pc = cells[cell]["per_call"]
            r = np.array(pc["row"])
            fk = np.array([folds[meta[i]["task"]] for i in r]) == k
            neg = fk & np.array([meta[i]["y"] == 0 for i in r])
            fl = np.array(pc["flag_frozen"])[neg]
            sv = np.array([full[i] for i in r[neg]])
            mine = sv > t
            if not np.array_equal(mine, fl):
                raise SystemExit(f"fold {k} {v}: recomputed flags differ from the stored flag_frozen")
            f["test"][v] = {"n": int(neg.sum()), "flagged": int(mine.sum()), "achieved_false_alarm": float(mine.mean()), "tied_exact": int((sv == t).sum())}
        out["folds"].append(f)
        for key, val in (("n", n), ("flag", gt), ("eq", eq), ("tb", tb), ("target", target), ("e9", f["tied_1e-9"]), ("e6", f["tied_1e-6"]),
                         ("r6", f["stored_6dp_tied_at_rounded_t"])):
            agg[key] += val
    out["pooled"] = {"n": agg["n"], "target_k": agg["target"], "flagged": agg["flag"], "achieved_false_alarm": agg["flag"] / agg["n"],
                     "tied_exact": agg["eq"], "tied_1e-9": agg["e9"], "tied_1e-6": agg["e6"], "stored_6dp_tied_at_rounded_t": agg["r6"],
                     "random_tie_break_flagged": agg["tb"], "random_tie_break_false_alarm": agg["tb"] / agg["n"],
                     "test": {v: {"n": sum(f["test"][v]["n"] for f in out["folds"]), "flagged": sum(f["test"][v]["flagged"] for f in out["folds"]),
                                  "tied_exact": sum(f["test"][v]["tied_exact"] for f in out["folds"])} for v in WORLDS}}
    for v in WORLDS:
        p = out["pooled"]["test"][v]
        p["achieved_false_alarm"] = p["flagged"] / p["n"]
        stored_fa = cells[WORLDS[v]]["values"]["false_alarm_frozen"]
        if abs(p["achieved_false_alarm"] - stored_fa) > 1e-12:
            raise SystemExit(f"{v}: pooled test false alarms {p['achieved_false_alarm']} vs stored {stored_fa}")
    for v in WORLDS:  # added in amendment 2, post hoc: is the held-out false-alarm rate the same across folds?
        fl_ = np.array([f["test"][v]["flagged"] for f in out["folds"]], float)
        n_ = np.array([f["test"][v]["n"] for f in out["folds"]], float)
        obs = np.stack([fl_, n_ - fl_])
        exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
        chi2 = float(((obs - exp) ** 2 / exp).sum())
        out["pooled"]["test"][v]["fold_homogeneity_chi2_df4"] = chi2
        out["pooled"]["test"][v]["fold_homogeneity_p"] = float(math.exp(-chi2 / 2) * (1 + chi2 / 2))  # chi2 survival, 4 df
    print("fold | layer | cal n | k | t | flagged | cal FA | ties exact/1e-9/1e-6 | tie-break FA | test FA real / sim / run2")
    for f in out["folds"]:
        te = f["test"]
        print(f"{f['fold']} | {f['layer']} | {f['n']} | {f['target_k']} | {f['threshold']:.6f} | {f['flagged']} | {100 * f['achieved_false_alarm']:.3f}% | "
              f"{f['tied_exact']}/{f['tied_1e-9']}/{f['tied_1e-6']} | {100 * f['random_tie_break_false_alarm']:.3f}% | "
              + " / ".join(f"{100 * te[v]['achieved_false_alarm']:.2f}% ({te[v]['flagged']}/{te[v]['n']}, ties {te[v]['tied_exact']})" for v in WORLDS))
    p = out["pooled"]
    print(f"pooled | | {p['n']} | {p['target_k']} | | {p['flagged']} | {100 * p['achieved_false_alarm']:.3f}% | {p['tied_exact']}/{p['tied_1e-9']}/{p['tied_1e-6']} "
          f"(stored 6dp: {p['stored_6dp_tied_at_rounded_t']}) | {100 * p['random_tie_break_false_alarm']:.3f}% | "
          + " / ".join(f"{100 * p['test'][v]['achieved_false_alarm']:.2f}% ({p['test'][v]['flagged']}/{p['test'][v]['n']})" for v in WORLDS))
    print("fold homogeneity of test FA (chi2, 4 df; descriptive): " + "; ".join(
        f"{v} {p['test'][v]['fold_homogeneity_chi2_df4']:.1f} (p {p['test'][v]['fold_homogeneity_p']:.2g})" for v in WORLDS))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"ties_196_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("layers_draws")
    ap.add_argument("npz")
    main(**vars(ap.parse_args()))
