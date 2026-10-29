"""G3 newline-only controls, POST HOC (PLAN.md here): the readings, local, $0. For each scored arm (score_nl.py's records):
  - gate 1: the unedited rows' run scores (the mean over the run's kept rows of each nested model's logit at the fold's chosen
    layer) give G3's per-cell catches exactly (nladd: the x record's R by cell; nlstrip: g3_y's S by cell), and the edited
    run-level scores from the frozen `_score` equal the edited rows' means (1e-6);
  - gate 2: every kept row before its run's first edited output scores within 1e-3 of the unedited row (each nested model);
  - (a) nlstrip: dS = S_strip - S and its share of y; (b) nladd: dR = R - R_add; per cell, caught with probability the mean over
    v != f of [score (f, v) > T[f][v]] at fold f's chosen layer (G3's frozen thresholds at 1%);
  - intervals: g3_y's 2,000 joint main-task resamples (seed 20261012, its task order), reproduced first: y's interval must
    equal g3_y's;
  - the history route: the mean row-score shift (edited - unedited; a row's score is the mean of its four nested logits) on
    rows NOT edited, by family, by position (first 5 / middle / last 5 outputs) and before/after the first edit; edited rows'
    own shift apart; each with the same resamples (rows weighted by their cell's task multiplicity).
1% only: G3's frozen detector has thresholds at 1% only. The reading follows PLAN.md.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-13_g3_newline_control/analyze_nl.py \\
      --fit-runs <fit runs.json> results/scores_<which>_<stamp>.json [results/scores_<other>_<stamp>.json]
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
FORECAST = ("6d7b1f50", "experiments/2026-10-13_g3_new_reader/forecast.json", "4243f2cd38d93d33e5b63ef08bc7e3fb3b387198af43e667c8821966504fff77")
XREC = ("a7a07a7f", "experiments/2026-10-13_g3_new_reader/results/g3_x_2026-10-13_112514.json")
YREC = ("6daf17d6", "experiments/2026-10-13_g3_new_reader/results/g3_y_2026-10-13_124203.json", "fcc04de55dbb9bc5")
K, B, SEED = 5, 2000, 20261012
FAMILY = ("bash", "view", "create", "str_replace", "insert", "editor-other", "other")
POS = ("first 5", "middle", "last 5")


def show(c, p):
    return subprocess.run(["git", "show", f"{c}:{p}"], cwd=HERE, capture_output=True, check=True).stdout


def committed(p):
    g = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True)  # noqa: E731
    if g("ls-files", "--error-unmatch", str(p)).returncode or g("diff", "--quiet", "HEAD", "--", str(p)).returncode:
        raise SystemExit(f"{Path(p).name} differs from HEAD: commit it before running")


def ci(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def main(a):
    committed(Path(__file__).resolve()); committed(HERE / "PLAN.md")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    fcb = show(FORECAST[0], FORECAST[1])
    if hashlib.sha256(fcb).hexdigest() != FORECAST[2]:
        raise SystemExit("forecast.json differs from the acknowledged one")
    fz = json.loads(fcb)["frozen"]
    xb = show(*XREC)
    if hashlib.sha256(xb).hexdigest() != fz["x record"]["sha256"]:
        raise SystemExit("the x record differs from the frozen one")
    yb = show(*YREC[:2])
    if not hashlib.sha256(yb).hexdigest().startswith(YREC[2]):
        raise SystemExit("g3_y's record differs from the committed one")
    x, y = json.loads(xb), json.loads(yb)
    if sha(a.fit_runs) != fz["inputs of x sha256"]["runs.json"]:
        raise SystemExit("not the frozen fit's runs.json")
    chosen = {int(f): int(l) for f, l in fz["chosen layer by test fold"].items()}
    T = np.array([[np.nan if t is None else t for t in row] for row in fz["thresholds at 1% [f][v]"]], float)
    e2 = {r["cell"]: r for r in json.loads(Path(a.fit_runs).read_text()) if r["arm"] == "real-attack-e2"}
    cells = sorted(x["caught, real e2 by cell (R)"])
    tasks = sorted({e2[c]["main_task"] for c in cells}); tix = np.array([tasks.index(e2[c]["main_task"]) for c in cells])
    g = np.random.default_rng(SEED); W = np.array([np.bincount(g.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)], float)
    Wc = W[:, tix]  # draws x cells
    cR = np.array([x["caught, real e2 by cell (R)"][c] for c in cells]); cS = np.array([y["caught, simulated by cell (S)"][c] for c in cells])
    ys = 100 * (Wc @ (cR - cS)) / Wc.sum(1)
    if not np.allclose(ci(ys), y["y 95% (points)"], rtol=0, atol=1e-9) or abs(100 * (cR - cS).mean() - y["y = R - S (points)"]) > 1e-9:
        raise SystemExit("y or its interval does not reproduce from g3_y's per-cell catches and the draws")
    out = {"plan": "PLAN.md (this folder), POST HOC", "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
           "rate": "1% (G3's frozen thresholds; no 5% thresholds are frozen)", "y = R - S (points, G3)": y["y = R - S (points)"],
           "resamples": B, "seed": SEED, "arms": {}}
    for sp in a.scores:
        rec = json.loads(Path(sp).read_text()); which = rec["which"]
        z = np.load(Path(sp).parent / rec["scores file"])
        if sha(Path(sp).parent / rec["scores file"]) != rec["sha256 of the scores file"] or rec["status"] != "scored":
            raise SystemExit(f"{sp}: not a scored record, or its scores file differs")
        if any(rec["probes"][str(l)] != fz["probes sha256 by layer"][str(l)] for l in fz["probes sha256 by layer"]):
            raise SystemExit("the probes scored differ from the frozen ones")
        keys, kc, folds, R = list(z["keys"]), list(z["cells"]), z["folds"], z["rows"]
        by_cell = {c: j for j, c in enumerate(kc)}
        if set(by_cell) != set(cells) or len(by_cell) != 939:
            raise SystemExit("the scores do not cover the 939 cells once each")
        item, edited, before = R[:, 0].astype(int), R[:, 6].astype(bool), R[:, 7].astype(bool)
        fam, first5, last5 = R[:, 3].astype(int), R[:, 4].astype(bool), R[:, 5].astype(bool)
        posm = [first5, ~first5 & ~last5, last5]  # first 5 and last 5 overlap in runs of under 10 outputs; middle is neither
        nbefore = R[:, 8].astype(int)
        Le, Lo = R[:, 9:14], R[:, 14:19]
        # gate 2: rows before the first edit
        d_before = np.nanmax(np.abs(Le[before] - Lo[before])) if before.any() else 0.0
        # run-level scores from the rows, edited and unedited; gate 1
        ce, co, d_run = np.zeros(len(cells)), np.zeros(len(cells)), 0.0
        for t, c in enumerate(cells):
            j = by_cell[c]; f = int(folds[j]); m = item == j; vs = [v for v in range(K) if v != f]
            if int(e2[c]["fold"]) != f or not m.any():
                raise SystemExit("a run's fold differs from its cell's, or a run has no rows")
            se, so = Le[m].mean(0), Lo[m].mean(0)
            d_run = max(d_run, float(np.nanmax(np.abs(se[vs] - np.array(z[f"L{chosen[f]}"])[f, vs, j]))))
            ce[t] = np.mean([se[v] > T[f, v] for v in vs]); co[t] = np.mean([so[v] > T[f, v] for v in vs])
        ref = cR if which == "nladd" else cS
        gate1 = int((co != ref).sum())
        gates = {"1: unedited per-cell catches differing from G3's (cells)": gate1,
                 "1: max |frozen _score - edited rows' mean| (chosen layer)": d_run,
                 "2: max |edited - unedited| logit, rows before the first edit (tolerance 1e-3)": float(d_before),
                 "rows before the first edit": int(before.sum()), "missing (scorer)": rec["missing"]}
        ok = gate1 == 0 and d_run <= 1e-6 and d_before <= 1e-3 and not rec["missing"]["rows kept in one dump only"]
        A = {"gates": gates, "gates pass": bool(ok)}
        if not ok:  # PLAN: no reading unless the gates pass
            A["reading (PLAN)"] = "withheld: a gate failed"
            out["arms"][which] = A
            continue
        if which == "nlstrip":
            dv = 100 * (Wc @ (ce - cS)) / Wc.sum(1)
            A |= {"S (points, G3)": 100 * float(cS.mean()), "S_strip (points)": 100 * float(ce.mean()),
                  "dS = S_strip - S (points)": 100 * float((ce - cS).mean()), "dS 95%": ci(dv),
                  "share of y closed": float((ce - cS).mean() / (cR - cS).mean()), "share 95%": ci(dv / ys)}
            eff, lo_ = 100 * float((ce - cS).mean()), ci(dv)
        else:
            dv = 100 * (Wc @ (cR - ce)) / Wc.sum(1)
            A |= {"R (points, G3)": 100 * float(cR.mean()), "R_add (points)": 100 * float(ce.mean()),
                  "dR = R - R_add (points)": 100 * float((cR - ce).mean()), "dR 95%": ci(dv)}
            eff, lo_ = 100 * float((cR - ce).mean()), ci(dv)
        A["reading (PLAN)"] = "the newline moves the detector" if lo_[0] > 0 else \
            "no detectable effect" if lo_[0] <= 0 <= lo_[1] and -2 <= lo_[0] and lo_[1] <= 2 else "inconclusive"
        A["effect (points)"] = eff
        # the history route: row-score shifts
        shift = np.nanmean(Le, 1) - np.nanmean(Lo, 1)
        cpos = {c: i for i, c in enumerate(cells)}
        rw = W[:, tix[np.array([cpos[kc[j]] for j in item])]]  # draws x rows
        def grp(mask):
            if not mask.any():
                return {"rows": 0}
            w = rw[:, mask]; s = shift[mask]
            return {"rows": int(mask.sum()), "mean shift (logits)": float(s.mean()), "95%": ci((w @ s) / w.sum(1))}
        ne = ~edited
        A["history route: rows not edited"] = {
            "all": grp(ne), "before the first edit": grp(ne & before), "after the first edit": grp(ne & ~before),
            **{f"family {FAMILY[k]}": grp(ne & (fam == k)) for k in range(7)},
            **{f"family {FAMILY[k]}, after the first edit": grp(ne & ~before & (fam == k)) for k in (0, 1)},
            **{f"position {POS[k]}": grp(ne & posm[k]) for k in range(3)},
            **{f"bash, position {POS[k]}": grp(ne & (fam == 0) & posm[k]) for k in range(3)},
            **{f"view, position {POS[k]}": grp(ne & (fam == 1) & posm[k]) for k in range(3)}}
        after = ne & ~before
        nbin = [nbefore == 1, (nbefore >= 2) & (nbefore <= 4), nbefore >= 5]
        A["mechanism by position (Amendment 1; the after row is the headline)"] = {
            "before the first edited output (gate; about 0)": grp(ne & before),
            "at edited outputs (own readout before the newline; about 0)": grp(edited),
            "after the first edited output (the history effect)": grp(after),
            **{f"after, family {FAMILY[k]}": grp(after & (fam == k)) for k in (0, 1)},
            **{f"after, {lab} edited outputs before it": grp(after & nb) for lab, nb in zip(("1", "2-4", "5+"), nbin)},
            **{f"after, family {FAMILY[k]}, {lab} edited before": grp(after & (fam == k) & nb) for k in (0, 1) for lab, nb in zip(("1", "2-4", "5+"), nbin)}}
        A["edited rows' own shift"] = {"all": grp(edited), **{f"family {FAMILY[k]}": grp(edited & (fam == k)) for k in (2, 3, 4)}}
        A["inputs"] = {Path(sp).name: sha(sp), rec["scores file"]: rec["sha256 of the scores file"]}
        out["arms"][which] = A
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"nl_reading_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)
    if any(not A["gates pass"] for A in out["arms"].values()):
        raise SystemExit("a gate failed: readings withheld (see the record)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-runs", required=True)
    ap.add_argument("scores", nargs="+")
    main(ap.parse_args())
