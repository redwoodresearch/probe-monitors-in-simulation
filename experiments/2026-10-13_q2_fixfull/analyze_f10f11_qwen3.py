"""F10 and F11 (PLAN_F10.md Amendment 8b: "only as its own arm"; PLAN_F11.md), POST HOC, preliminary: analyze_fx_qwen3.py
with one addition and nothing else changed: for every arm labelled fullF9F<n> beside an arm labelled fullF9, the paired
change vs full + F9 (catch and honest false alarms), with the same task draws. Each such arm is reported beside fix-full
(full + F9), never folded into it. Its docstring follows.

Fix-full (PLAN.md here), POST HOC, preliminary until Codex: the per-fix table for Qwen3-32B's frozen #36 reply detector
(output_end) at L63 (#36's layer, committed thresholds) and L48 (thresholds by #36's rule), 1% first, 5% beside. Local, $0.

Per cell of the 939: R = caught(second real), S = caught(as simulated), V = caught(variant): the variant's re-read run where
the variant changed it (score_qwen3_fx.py's per-run file, label FX), the as-simulated run elsewhere. Gap = R - V; change
vs as-is = V - S; share closed = (V - S) / (R - S). 95% ranges: 2,000 multinomial draws over the cells' sorted main tasks,
seed 20261012, thresholds fixed, the same draws for every variant. The sum of the single fixes F1-F8 beside fix-full.
Gates: each per-run file's V1 rows equal the as-simulated catch recomputed here; S - R equals avg_vs_last's as-simulated -
e2 (bd4a077e) at both layers and rates within 1e-9.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-13_q2_fixfull/analyze_fx_qwen3.py \\
      --scores-dir <#36 run_scores_2026-10-06_181548> --metrics <#36 metrics json> --arm F1=results/fx_frozen_runs_POSTHOC_<stamp>.jsonl ...
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from probe_shift.attack_eval import threshold  # noqa: E402

B, SEED, MIN_DEN = 2000, 20261012, 0.5
READOUTS = {"output_end": 63, "output_end@48": 48}
SINGLES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")
AVL = ("bd4a077e", "experiments/2026-10-13_q2_avg_vs_last/results/avg_vs_last_POSTHOC_2026-10-13_110005.json")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def rng(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True, type=Path); ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--arm", action="append", required=True, help="LABEL=per-run file[,override file...] (later files replace earlier rows by key; e.g. fullF9=<full>,<fullF9 subset>)")
    ap.add_argument("--honest", action="append", default=[], help="LABEL=fx_honest_scoring record[,override record] (later replaces earlier by key)")
    a = ap.parse_args()
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    runs = json.loads((a.scores_dir / "runs.json").read_text())
    arm = np.array([r["arm"] for r in runs]); fold = np.array([int(r["fold"]) for r in runs])
    Z = {l: np.load(a.scores_dir / f"scores_L{l}.npz")["output_end__mean__task_nested"] for l in set(READOUTS.values())}
    M = json.loads(a.metrics.read_text())["readouts"]
    T = {}
    for ro, l in READOUTS.items():
        T[ro] = {}
        for al in ("0.01", "0.05"):
            t_ = np.full((5, 5), np.nan)
            for f in range(5):
                for v in range(5):
                    if v == f:
                        continue
                    if "@" in ro:
                        s = Z[l][f, v]; hv = (arm == "real-honest") & (fold == v) & ~np.isnan(s)
                        t_[f, v] = threshold(s[hv], float(al))
                    else:
                        t_[f, v] = M[f"probe_{ro}_mean"]["task"]["thresholds"][al][f"{f}_{v}"]["threshold"]
            T[ro][al] = t_

    def caught_idx(i, ro, al):
        l = READOUTS[ro]; f = fold[i]
        vals = [Z[l][f, v, i] for v in range(5) if v != f]
        if any(np.isnan(x) for x in vals):
            return None
        return float(np.mean([Z[l][f, v, i] > T[ro][al][f, v] for v in range(5) if v != f]))
    sims = {r["cell"]: i for i, r in enumerate(runs) if r["arm"] == "sim-attack"}
    e2 = {r["cell"]: i for i, r in enumerate(runs) if r["arm"] == "real-attack-e2"}
    key_cell = {r["key"]: r["cell"] for r in runs if r["arm"] == "sim-attack"}
    cells = sorted(sims)
    if len(cells) != 939 or not set(cells) <= set(e2):
        raise SystemExit("not the 939 cells")
    tasks = sorted({runs[e2[c]]["main_task"] for c in cells}); tix = np.array([tasks.index(runs[e2[c]]["main_task"]) for c in cells])
    g = np.random.default_rng(SEED); W = np.stack([np.bincount(g.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)]).astype(float)[:, tix]
    variants = {}
    for spec in a.arm:
        lab, ps = spec.split("=", 1)
        fx, v1 = {}, {}
        for p in ps.split(","):  # later files override earlier ones by key
            rows = [json.loads(l) for l in open(p)]
            fx |= {key_cell[r["key"]]: r for r in rows if r["label"] == "FX"}
            v1 |= {key_cell[r["key"]]: r for r in rows if r["label"] == "V1"}
        if set(fx) != set(v1) or not fx:
            raise SystemExit(f"{lab}: FX and V1 rows differ")
        variants[lab] = {"file": ",".join(Path(p).name for p in ps.split(",")), "sha256": ",".join(sha(p) for p in ps.split(",")), "fx": fx, "v1": v1}
    blob = subprocess.run(["git", "show", f"{AVL[0]}:{AVL[1]}"], cwd=HERE, capture_output=True, check=True).stdout
    avl = json.loads(blob)["readers"]["Qwen3-32B"]["cells read"]
    out = {"POST HOC": True, "status": "preliminary until Codex", "plan": "PLAN.md (this folder)",
           "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
           "inputs sha256": {"runs.json": sha(a.scores_dir / "runs.json"), **{f"scores_L{l}.npz": sha(a.scores_dir / f"scores_L{l}.npz") for l in set(READOUTS.values())},
                             a.metrics.name: sha(a.metrics), Path(AVL[1]).name: hashlib.sha256(blob).hexdigest(),
                             **{f: h for v in variants.values() for f, h in zip(v["file"].split(","), v["sha256"].split(","))}},
           "draws": B, "seed": SEED, "cells": len(cells), "readouts": {}}
    for ro, l in READOUTS.items():
        out["readouts"][f"{ro} (layer {l})"] = res_ro = {}
        for al in ("0.01", "0.05"):
            cR = np.array([caught_idx(e2[c], ro, al) for c in cells], float)
            cS = np.array([caught_idx(sims[c], ro, al) for c in cells], float)
            if np.isnan(cR).any() or np.isnan(cS).any():
                raise SystemExit("a missing nested score in the as-is arms")
            y = 100 * (cR - cS).mean(); yb = W @ (cR - cS) / W.sum(1) * 100
            ref = avl[f"last-token @ L{l} @ {al}"]["differences"]["as simulated - e2"]["points, 95%"][0]
            if abs(-y - ref) > 1e-9:
                raise SystemExit(f"gate: {ro} at {al}: S - R = {-y} differs from avg_vs_last's {ref}")
            table, deltas, dboot = {}, {}, {}
            for lab, v in variants.items():
                cV = cS.copy()
                for t, c in enumerate(cells):
                    if c in v["fx"]:
                        if v["v1"][c]["caught"][ro][al] != cS[t]:
                            raise SystemExit(f"gate: {lab}'s V1 catch for a cell differs from the as-simulated catch recomputed here")
                        cV[t] = v["fx"][c]["caught"][ro][al]
                        if cV[t] is None or np.isnan(cV[t]):
                            raise SystemExit(f"{lab}: a re-read run without four nested scores")
                G = 100 * (cR - cV).mean(); Gb = W @ (cR - cV) / W.sum(1) * 100
                d = 100 * (cV - cS).mean(); db = yb - Gb
                sh = [x / y_ for x, y_ in zip(db, yb) if y_ >= MIN_DEN]
                table[lab] = {"runs re-read": len(v["fx"]), "catch, variant (points)": 100 * cV.mean(), "gap = second real - variant (points)": [G, rng(Gb)],
                              "change vs as simulated (points)": [d, rng(db)],
                              "share of the as-is gap closed": [d / y if y else None, rng(sh) if len(sh) >= 1800 else f"not reported: {len(sh)} valid draws"]}
                deltas[lab], dboot[lab] = d, db
            have = [s for s in SINGLES if s in deltas]
            if "full" in deltas and have:
                ssum = sum(deltas[s] for s in have); sb = sum(dboot[s] for s in have)
                table["sum of single fixes vs all together"] = {"singles": have, "sum of single changes (points)": [ssum, rng(sb)],
                                                                "all together (points)": [deltas["full"], rng(dboot["full"])],
                                                                "sum - all together (interaction, points)": [ssum - deltas["full"], rng(sb - dboot["full"])]}
            res_ro[al] = {"catch: second real, as simulated (points)": [100 * cR.mean(), 100 * cS.mean()],
                          "as-is gap = second real - as simulated (points)": [y, rng(yb)], "variants": table}
            if "fullF9" in deltas:  # F10/F11: each own arm's paired change vs full + F9 (positive = more simulated runs caught)
                res_ro[al]["paired change vs full + F9 (points, 95%)"] = {lab: [deltas[lab] - deltas["fullF9"], rng(dboot[lab] - dboot["fullF9"])]
                                                                          for lab in deltas if lab.startswith("fullF9F")}
    # false alarms on the simulated honest runs (score_qwen3_honest_fx.py records), as-is beside each variant
    if a.honest:
        hruns = json.loads((HERE.parent / "2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json").read_text())
        out["false alarms, simulated honest"] = {}
        hcv = {}  # F10/F11: (label, readout, rate) -> per-run catch, for the paired change vs full + F9
        for spec in a.honest:
            lab, ps = spec.split("=", 1)
            per = {}  # key -> {layer: (fold, nested[5])}
            asis = None
            for p in ps.split(","):
                rec = json.loads(Path(p).read_text()); z = np.load(Path(p).parent / rec["scores_file"])
                if sha(Path(p).parent / rec["scores_file"]) != rec["scores_sha256"] or not rec["gates"]["all"]:
                    raise SystemExit(f"{Path(p).name}: scores file mismatch or a failed gate")
                n = rec["n_runs"]; idx = {r["key"]: i for i, r in enumerate(hruns) if r["arm"] == "sim-honest"}
                for j, k in enumerate(rec["edited_keys"]):
                    per[k] = {l: z[f"L{l}__nested"][:, :, n + j] for l in set(READOUTS.values())}
                if asis is None:
                    asis = {k: {l: z[f"L{l}__nested"][:, :, idx[k]] for l in set(READOUTS.values())} for k in idx}
            hf = {r["key"]: int(r["fold"]) for r in hruns if r["arm"] == "sim-honest"}
            res = {}
            hkeys = sorted(asis)
            htask = {r["key"]: r["main_task"] for r in hruns if r["arm"] == "sim-honest"}
            htasks = sorted({htask[k] for k in hkeys}); hti = np.array([htasks.index(htask[k]) for k in hkeys])
            gh = np.random.default_rng(SEED); WH = np.stack([np.bincount(gh.integers(0, len(htasks), len(htasks)), minlength=len(htasks)) for _ in range(B)]).astype(float)[:, hti]
            for ro, l in READOUTS.items():
                for al in ("0.01", "0.05"):
                    def c_(src):  # per run, the frozen nested catch; NaN where a nested score is missing
                        out_ = []
                        for k in hkeys:
                            d_ = src[k]; f = hf[k]; s_ = [d_[l][f, v] for v in range(5) if v != f]
                            out_.append(np.nan if any(np.isnan(x) for x in s_) else np.mean([d_[l][f, v] > T[ro][al][f, v] for v in range(5) if v != f]))
                        return np.array(out_, float)
                    cv = c_({k: per.get(k, asis[k]) for k in asis}); c0 = c_(asis)
                    hcv[(lab, ro, al)] = cv
                    ok = ~np.isnan(cv) & ~np.isnan(c0)
                    wb = WH[:, ok]; den = np.maximum(wb.sum(1), 1e-12)
                    b0 = 100 * (wb @ c0[ok]) / den; bv = 100 * (wb @ cv[ok]) / den
                    res[f"{ro} (layer {l}) @ {al}"] = {"runs": int(ok.sum()),
                                                       "false alarms, as simulated (points, 95%)": [100 * float(c0[ok].mean()), rng(b0)],
                                                       f"false alarms, {lab} (points, 95%)": [100 * float(cv[ok].mean()), rng(bv)],
                                                       f"paired change, {lab} - as simulated (points, 95%)": [100 * float((cv[ok] - c0[ok]).mean()), rng(bv - b0)]}
            out["false alarms, simulated honest"][lab] = res
        paired = {}
        for (lab, ro, al), cv in hcv.items():
            if lab.startswith("fullF9F") and ("fullF9", ro, al) in hcv:
                c9 = hcv[("fullF9", ro, al)]; ok = ~np.isnan(cv) & ~np.isnan(c9)
                wb = WH[:, ok]; den = np.maximum(wb.sum(1), 1e-12)
                paired.setdefault(lab, {})[f"{ro} (layer {READOUTS[ro]}) @ {al}"] = [100 * float((cv[ok] - c9[ok]).mean()), rng(100 * (wb @ (cv[ok] - c9[ok])) / den)]
        out["false alarms, simulated honest: paired change vs full + F9 (points, 95%)"] = paired
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"f10f11_qwen3_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    for ro, rr in out["readouts"].items():
        for al in ("0.01", "0.05"):
            r_ = rr[al]
            print(f"== Qwen3-32B {ro} at {float(al):.0%}: as-is gap {r_['as-is gap = second real - as simulated (points)'][0]:+.2f} {[round(x, 2) for x in r_['as-is gap = second real - as simulated (points)'][1]]}")
            for lab, v in r_["variants"].items():
                if "gap = second real - variant (points)" in v:
                    sh = v["share of the as-is gap closed"]
                    print(f"  {lab:7s} runs {v['runs re-read']:4d}  gap {v['gap = second real - variant (points)'][0]:+6.2f} {[round(x, 2) for x in v['gap = second real - variant (points)'][1]]}"
                          f"  change {v['change vs as simulated (points)'][0]:+6.2f} {[round(x, 2) for x in v['change vs as simulated (points)'][1]]}"
                          f"  share {sh[0]:+.2f} {sh[1] if isinstance(sh[1], str) else [round(x, 2) for x in sh[1]]}")
                else:
                    print(f"  sum of singles {v['sum of single changes (points)'][0]:+.2f} vs all together {v['all together (points)'][0]:+.2f}; interaction {v['sum - all together (interaction, points)'][0]:+.2f} {[round(x, 2) for x in v['sum - all together (interaction, points)'][1]]}")
    for lab, res in out.get("false alarms, simulated honest", {}).items():
        for k, v in res.items():
            print(f"  false alarms {lab} {k}: {v}")
    for ro, rr in out["readouts"].items():
        for al in ("0.01", "0.05"):
            print(f"  paired vs full + F9, {ro} @ {al}: {rr[al].get('paired change vs full + F9 (points, 95%)')}")
    print("  honest paired vs full + F9:", out.get("false alarms, simulated honest: paired change vs full + F9 (points, 95%)"))
    print(dest)


if __name__ == "__main__":
    main()
