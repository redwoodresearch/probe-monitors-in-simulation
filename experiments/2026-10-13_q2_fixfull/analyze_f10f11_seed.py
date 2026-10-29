"""F10 and F11 (PLAN_F10.md Amendment 8b: "only as its own arm"; PLAN_F11.md), POST HOC, preliminary: analyze_fx_seed.py
with three additions and nothing else changed: --f10-scores and --f11-scores (g3_score_f10.py / g3_score_f11.py records)
add the arms fullF9F10 and fullF9F11 (full + F9 with that fix's changed runs replacing it by cell); each one's paired
change vs full + F9 with the same task draws; and honest false alarms for full + F9 and each own arm (full + F9's honest
runs with that fix's honest runs replacing them by key), with the paired change vs full + F9. Requires --f9-scores. Each
own arm is reported beside fix-full (full + F9), never folded into it. Its docstring follows.

Fix-full (PLAN.md here), POST HOC, preliminary until Codex: the per-fix table for Seed-OSS-36B's frozen G3 detector, 1%
first, 5% beside (5% thresholds by the same rule from the fit's stored honest scores). Local, $0. A copy of
../2026-10-13_g3_followups/analyze_rep_g3.py (1c277bfc) generalized to every fx arm: per cell, R (second real), S (as
simulated), V (the variant's re-read run where the variant changed it, g3_score_fx.py; the as-simulated run elsewhere).
Gap = R - V; change = V - S; share closed = (V - S) / (R - S) (draws with y* >= 0.5 points). G3's draws (2,000, seed
20261012, the 939 cells' sorted main tasks, thresholds fixed). The sum of the single fixes F1-F8 beside fix-full; fix939
(the G3 fork's repaired arm, --rep-scores) and the newline fixes together (NL) as rows. False alarms: the frozen detector's
catch on the 437 simulated honest runs under fix-full (each in its main task's fold of the fit). Gate first: y and its range
at 1% equal g3_y's within 1e-9, and every per-cell R and S equal g3_x's and g3_y's.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-13_q2_fixfull/analyze_fx_seed.py \\
      --fx-scores results/g3_scores_fx_<stamp>.json --g3-results <G3's results dir> [--rep-scores <g3_scores_rep_<stamp>.json>]
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

B, SEED, LAYER_BY_FOLD, FIT, MIN_DEN = 2000, 20261012, (40, 63, 63, 63, 63), "2026-10-13_111525", 0.5
SINGLES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")
PINS = {"g3_y_2026-10-13_124203.json": "fcc04de55dbb9bc5df702c3c420814fe70ec736b4cabb8938cbc74e0631b2a69",
        "g3_x_2026-10-13_112514.json": "6fb7f787ebd4853a0f8b9f4e31c4585e602c88cf305e2fef11562420fe5baee2",
        "g3_scores_sim_2026-10-13_124117.json": "d828cc27bb87f07a2763948ec8371fe0ce6b95e3dec7895447a9485184b88798"}
FORECAST = ("6d7b1f50c2c08a5887962b5066681684d6e1ae63", "experiments/2026-10-13_g3_new_reader/forecast.json",
            "4243f2cd38d93d33e5b63ef08bc7e3fb3b387198af43e667c8821966504fff77")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def caught(S, F, T):
    valid = np.arange(5)[None, :] != F[:, None]
    return np.where(valid, S > T[F], False).sum(1) / valid.sum(1)


def rng(v):
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def fa(ch, tk):
    """share caught (points) and its 95% range over draws of the runs' sorted main tasks (seed SEED)"""
    ts = sorted(set(tk)); ti = np.array([ts.index(t) for t in tk])
    g = np.random.default_rng(SEED); W = np.stack([np.bincount(g.integers(0, len(ts), len(ts)), minlength=len(ts)) for _ in range(B)]).astype(float)[:, ti]
    return [100 * float(ch.mean()), rng(100 * (W @ ch) / W.sum(1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx-scores", required=True, type=Path); ap.add_argument("--g3-results", required=True, type=Path)
    ap.add_argument("--rep-scores", type=Path)
    ap.add_argument("--f9-scores", type=Path, help="g3_score_f9.py's record: adds F9 and fullF9 (= full with the F9 run's fullF9 runs overriding by cell)")
    ap.add_argument("--stage-record", required=True, type=Path, help="the staged run's stage_record.json (Volume copy): each run's changed messages")
    ap.add_argument("--extract-dir", required=True, type=Path, help="local copy of the fx run's chunks/ and stats/ (small JSONs): each run's shard and GPU")
    ap.add_argument("--f10-scores", type=Path); ap.add_argument("--f11-scores", type=Path)
    a = ap.parse_args()
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    G3R = a.g3_results
    for n, h in PINS.items():
        if sha(G3R / n) != h:
            raise SystemExit(f"{n} is not the pinned file")
    blob = subprocess.run(["git", "show", f"{FORECAST[0]}:{FORECAST[1]}"], cwd=HERE, capture_output=True, check=True).stdout
    if hashlib.sha256(blob).hexdigest() != FORECAST[2]:
        raise SystemExit("not the acknowledged forecast.json")
    fc = json.loads(blob)["frozen"]
    x, y = json.loads((G3R / "g3_x_2026-10-13_112514.json").read_text()), json.loads((G3R / "g3_y_2026-10-13_124203.json").read_text())
    sd = G3R / f"run_scores_{FIT}"
    if sha(sd / "runs.json") != fc["inputs of x sha256"]["runs.json"] or any(sha(sd / f"scores_L{l}.npz") != fc["inputs of x sha256"][f"scores_L{l}.npz"] for l in set(LAYER_BY_FOLD)):
        raise SystemExit("the fit's stored scores differ from the frozen ones")
    runs = json.loads((sd / "runs.json").read_text()); arm = np.array([r["arm"] for r in runs]); fold = np.array([int(r["fold"]) for r in runs])
    N = {l: np.load(sd / f"scores_L{l}.npz")["output_end__mean__task_nested"] for l in set(LAYER_BY_FOLD)}
    e2 = {r["cell"]: i for i, r in enumerate(runs) if r["arm"] == "real-attack-e2"}
    cells = sorted(x["caught, real e2 by cell (R)"]); ei = np.array([e2[c] for c in cells]); F = fold[ei]
    S_e2 = np.stack([N[LAYER_BY_FOLD[fold[i]]][fold[i], :, i] for i in ei])

    def load_rec(rec, base, mode_ok):
        z = np.load(base / rec["scores file"])
        if sha(base / rec["scores file"]) != rec["scores sha256"] or rec.get("status") != "scored" \
                or any(rec["probes sha256"][str(l)] != fc["probes sha256 by layer"][str(l)] for l in fc["probes sha256 by layer"]):
            raise SystemExit(f"{rec['scores file']}: not a scored record of the frozen detector")
        kc = {k: c for k, c in rec["keys and cells"]}
        return {kc[k]: (j, z) for j, k in enumerate(z["keys"])}
    srec = json.loads((G3R / "g3_scores_sim_2026-10-13_124117.json").read_text())
    if srec.get("frozen run") != FIT or srec.get("runs") != 939:
        raise SystemExit("not G3's simulated scores")
    sim_by = load_rec(srec, G3R, "sim")
    fx = json.loads(a.fx_scores.read_text())
    if fx.get("frozen run") != FIT:
        raise SystemExit("the fx scores are not of the frozen fit")
    arms = {}
    for arm_name, rec in fx["arms"].items():
        arms[arm_name] = load_rec(rec, a.fx_scores.parent, "fx")
    if a.f9_scores:
        f9 = json.loads(a.f9_scores.read_text())
        if f9.get("frozen run") != FIT:
            raise SystemExit("the F9 scores are not of the frozen fit")
        arms["sim-attack-fx-F9"] = load_rec(f9["arms"]["sim-attack-fx-F9"], a.f9_scores.parent, "fx")
        arms["sim-attack-fx-fullF9"] = dict(arms["sim-attack-fx-full"]) | load_rec(f9["arms"]["sim-attack-fx-fullF9"], a.f9_scores.parent, "fx")
        fx["arms"]["sim-attack-fx-F9"] = f9["arms"]["sim-attack-fx-F9"]
    own = {}  # F10/F11: own arms, each full + F9 with that fix's runs replacing it
    for tag, p in (("F10", a.f10_scores), ("F11", a.f11_scores)):
        if p is None:
            continue
        if not a.f9_scores:
            raise SystemExit("--f10-scores / --f11-scores need --f9-scores (their base is full + F9)")
        r = json.loads(p.read_text())
        if r.get("frozen run") != FIT:
            raise SystemExit(f"the {tag} scores are not of the frozen fit")
        arms[f"sim-attack-fx-fullF9{tag}"] = dict(arms["sim-attack-fx-fullF9"]) | load_rec(r["arms"][f"sim-attack-fx-fullF9{tag}"], p.parent, "fx")
        own[tag] = (r, p)
    if a.rep_scores:
        rrec = json.loads(a.rep_scores.read_text())
        if rrec.get("mode") != "rep" or rrec.get("frozen run") != FIT:
            raise SystemExit("not the G3 fork's repaired-arm scores")
        arms["fix939 (the G3 fork's repaired arm)"] = load_rec(rrec, a.rep_scores.parent, "rep")
    if set(sim_by) != set(cells):
        raise SystemExit("the as-simulated scores do not cover the frozen cells")
    pick = lambda by, t, c: by[c][1][f"L{LAYER_BY_FOLD[F[t]]}"][F[t], :, by[c][0]]  # noqa: E731
    S_sim = np.stack([pick(sim_by, t, c) for t, c in enumerate(cells)])
    # reproduction gate: a staged run identical to its as-simulated run (fix-full's and N's unchanged runs) must score as
    # G3's stored as-simulated scores within 1e-3, on every GPU type the extraction used (each run's shard from the chunk
    # index, each shard's GPU from its stats)
    srec = json.loads(a.stage_record.read_text())
    unchanged = {(r["arm"], r["key"]) for r in srec["runs"] if r["first_changed_message"] is None and r["arm"].startswith("sim-attack-fx-")}
    gpu_of_shard = {int(p.stem.split("_")[1]): json.loads(p.read_text()).get("gpu") for p in (a.extract_dir / "stats").glob("shard_*.json")}
    shard_of = {}
    for p in (a.extract_dir / "chunks").glob("shard*_chunk*.json"):
        for t_ in json.loads(p.read_text())["transcripts"]:
            shard_of[(t_["arm"], t_["key"])] = int(p.stem[5:8])
    kc_all = {arm_name: {k: c for k, c in rec["keys and cells"]} for arm_name, rec in fx["arms"].items()}
    cidx = {c: t for t, c in enumerate(cells)}
    repro = {}
    for arm_name, by in arms.items():
        for k, c in kc_all.get(arm_name, {}).items():
            if (arm_name, k) not in unchanged:
                continue
            t = cidx[c]; v = pick(by, t, c); s0 = S_sim[t]; m_ = np.arange(5) != F[t]
            gpu = gpu_of_shard.get(shard_of.get((arm_name, k)), "unknown")
            r_ = repro.setdefault(gpu, {"runs": 0, "max abs diff": 0.0})
            r_["runs"] += 1; r_["max abs diff"] = max(r_["max abs diff"], float(np.abs(v[m_] - s0[m_]).max()))
    if not repro or any(r_["max abs diff"] > 1e-3 for r_ in repro.values()):
        raise SystemExit(f"reproduction gate failed (unchanged runs vs G3's as-simulated scores): {repro}")
    T1 = np.array([[np.nan if v is None else v for v in row] for row in fc["thresholds at 1% [f][v]"]], float)
    T5 = np.full((5, 5), np.nan); hon = arm == "real-honest"
    for f in range(5):
        for v in range(5):
            if v != f:
                T5[f, v] = threshold(N[LAYER_BY_FOLD[f]][f, v][hon & (fold == v)], 0.05)
    e2r = {r["cell"]: r for r in runs if r["arm"] == "real-attack-e2"}
    tasks = sorted({e2r[c]["main_task"] for c in cells}); tix = np.array([tasks.index(e2r[c]["main_task"]) for c in cells])
    g = np.random.default_rng(SEED); W = np.stack([np.bincount(g.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)]).astype(float)[:, tix]
    out = {"POST HOC": True, "status": "preliminary until Codex", "plan": "PLAN.md (this folder)",
           "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
           "inputs sha256": PINS | {"forecast.json": FORECAST[2], a.fx_scores.name: sha(a.fx_scores), **({a.rep_scores.name: sha(a.rep_scores)} if a.rep_scores else {})},
           "draws": B, "seed": SEED, "reproduction gate (unchanged runs, by GPU)": repro,
           "GPU by shard": {str(k): v for k, v in sorted(gpu_of_shard.items())}, "rates": {}}
    attack_arms = {k: v for k, v in arms.items() if not k.startswith("sim-honest")}
    hmt = {r["key"]: r["main_task"] for r in json.loads((HERE.parent / "2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717/runs.json").read_text())
           if r["arm"] == "sim-honest"}
    for r_, T in ((0.01, T1), (0.05, T5)):
        cR, cS = caught(S_e2, F, T), caught(S_sim, F, T)
        if r_ == 0.01:
            if any(cR[t] != x["caught, real e2 by cell (R)"][c] or cS[t] != y["caught, simulated by cell (S)"][c] for t, c in enumerate(cells)):
                raise SystemExit("gate: a per-cell catch differs from g3_x's R or g3_y's S")
        yy = 100 * (cR - cS).mean(); yb = W @ (cR - cS) / W.sum(1) * 100
        if r_ == 0.01 and (abs(yy - y["y = R - S (points)"]) > 1e-9 or max(abs(rng(yb)[0] - y["y 95% (points)"][0]), abs(rng(yb)[1] - y["y 95% (points)"][1])) > 1e-9):
            raise SystemExit("gate: y or its range at 1% differs from g3_y's")
        table, deltas, dboot, cvs = {}, {}, {}, {}
        for name, by in attack_arms.items():
            if not set(by) <= set(cells):
                raise SystemExit(f"{name}: scores outside the frozen cells")
            S_v = np.stack([pick(by, t, c) if c in by else S_sim[t] for t, c in enumerate(cells)])
            vs = np.arange(5)[None, :] != F[:, None]
            if not np.isfinite(S_v[vs]).all():
                raise SystemExit(f"{name}: a non-finite nested score")
            cV = caught(S_v, F, T)
            G = 100 * (cR - cV).mean(); Gb = W @ (cR - cV) / W.sum(1) * 100
            d = 100 * (cV - cS).mean(); db = yb - Gb
            sh = [q / y_ for q, y_ in zip(db, yb) if y_ >= MIN_DEN]
            lab = name.removeprefix("sim-attack-fx-")
            table[lab] = {"runs re-read": len(by), "catch, variant (points)": 100 * cV.mean(), "gap = second real - variant (points)": [G, rng(Gb)],
                          "change vs as simulated (points)": [d, rng(db)],
                          "share of the as-is gap closed": [d / yy if yy else None, rng(sh) if len(sh) >= 1800 else f"not reported: {len(sh)} valid draws"]}
            deltas[lab], dboot[lab] = d, db
            cvs[lab] = cV
        have = [s for s in SINGLES if s in deltas]
        if "full" in deltas and have:
            ssum = sum(deltas[s] for s in have); sb = sum(dboot[s] for s in have)
            table["sum of single fixes vs all together"] = {"singles": have, "sum of single changes (points)": [ssum, rng(sb)],
                                                            "all together (points)": [deltas["full"], rng(dboot["full"])],
                                                            "sum - all together (interaction, points)": [ssum - deltas["full"], rng(sb - dboot["full"])]}
        res = {"catch: second real, as simulated (points)": [100 * cR.mean(), 100 * cS.mean()], "as-is gap = second real - as simulated (points)": [yy, rng(yb)], "variants": table}
        if "fullF9" in cvs:  # F10/F11: paired change vs full + F9 (positive = more simulated runs caught)
            res["paired change vs full + F9 (points, 95%)"] = {lab: [100 * float((cvs[lab] - cvs["fullF9"]).mean()), rng(W @ (cvs[lab] - cvs["fullF9"]) / W.sum(1) * 100)]
                                                               for lab in cvs if lab.startswith("fullF9F")}
        if own:  # F10/F11: honest false alarms, full + F9 and each own arm, per key, with the paired change vs full + F9
            def hz(rec, base):
                z = np.load(base / rec["scores file"])
                if sha(base / rec["scores file"]) != rec["scores sha256"] or rec.get("status") != "scored":
                    raise SystemExit(f"{rec['scores file']}: not a scored record")
                return {k: (int(z["folds"][j]), np.array(z[f"L{LAYER_BY_FOLD[int(z['folds'][j])]}"][int(z["folds"][j]), :, j])) for j, k in enumerate(z["keys"])}
            h9 = hz(fx["arms"]["sim-honest-fx-full"], a.fx_scores.parent) | hz(json.loads(a.f9_scores.read_text())["arms"]["sim-honest-fx-fullF9"], a.f9_scores.parent)
            hk = sorted(h9); htk = [hmt[k] for k in hk]
            ts_ = sorted(set(htk)); ti_ = np.array([ts_.index(t_) for t_ in htk])
            gh = np.random.default_rng(SEED); WH = np.stack([np.bincount(gh.integers(0, len(ts_), len(ts_)), minlength=len(ts_)) for _ in range(B)]).astype(float)[:, ti_]
            def hc(src):
                Fh = np.array([src[k][0] for k in hk]); Sh = np.stack([src[k][1] for k in hk])
                if not np.isfinite(np.where(np.arange(5)[None, :] != Fh[:, None], Sh, 0.0)).all():
                    raise SystemExit("an honest run without four finite nested scores")
                return caught(Sh, Fh, T)
            c9 = hc(h9)
            hres = {"fullF9": {"runs": len(hk), "share caught (points)": 100 * float(c9.mean()), "share caught (points, 95% over main-task draws)": fa(c9, htk)}}
            for tag, (r, p) in own.items():
                ho = hz(r["arms"][f"sim-honest-fx-fullF9{tag}"], p.parent)
                if not set(ho) <= set(h9):
                    raise SystemExit(f"{tag}: honest runs outside full + F9's")
                cO = hc(h9 | ho)
                hres[f"fullF9{tag}"] = {"runs": len(hk), "runs replaced": len(ho), "share caught (points)": 100 * float(cO.mean()),
                                        "share caught (points, 95% over main-task draws)": fa(cO, htk),
                                        "paired change vs full + F9 (points, 95%)": [100 * float((cO - c9).mean()), rng(100 * (WH @ (cO - c9)) / WH.sum(1))]}
            res["false alarms on simulated honest: full + F9 and own arms"] = hres
        for harm in ("sim-honest-fx-full", "sim-honest-fx-NL"):  # false alarms on the simulated honest runs under fix-full and N
            if harm not in arms:
                continue
            hrec = fx["arms"][harm]; z = np.load(a.fx_scores.parent / hrec["scores file"])
            Fh = z["folds"].astype(int); Sh = np.stack([z[f"L{LAYER_BY_FOLD[f_]}"][f_, :, j] for j, f_ in enumerate(Fh)])
            if not np.isfinite(np.where(np.arange(5)[None, :] != Fh[:, None], Sh, 0.0)).all():
                raise SystemExit(f"{harm}: a non-finite nested score")
            ch = caught(Sh, Fh, T)
            res[f"false alarms on simulated honest, {harm}"] = {"runs": int(len(ch)), "left out (main task not in the fit)": len(hrec["left out (main task not in the fit)"]),
                                                                "share caught (points)": 100 * float(ch.mean()), "runs caught in at least one fold": int((ch > 0).sum()),
                                                                "share caught (points, 95% over main-task draws)": fa(ch, [hmt[k] for k in z["keys"]])}
        # reference: the fit's real honest runs under the same rule (each in its own fold, so out of the thresholds' calibration fold)
        ri = np.flatnonzero(hon); Fr = fold[ri]; Sr = np.stack([N[LAYER_BY_FOLD[fold[i]]][fold[i], :, i] for i in ri])
        if not np.isfinite(np.where(np.arange(5)[None, :] != Fr[:, None], Sr, 0.0)).all():
            raise SystemExit("real honest: a non-finite nested score")
        cr = caught(Sr, Fr, T)
        res["false alarms on real honest (the fit's runs), reference"] = {"runs": int(len(ri)), "share caught (points, 95% over main-task draws)": fa(cr, [runs[i]["main_task"] for i in ri]),
                                                                          "runs caught in at least one fold": int((cr > 0).sum())}
        out["rates"][str(r_)] = res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"f10f11_seed_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    for al, r_ in out["rates"].items():
        print(f"== Seed-OSS-36B at {float(al):.0%}: as-is gap {r_['as-is gap = second real - as simulated (points)'][0]:+.2f} {[round(q, 2) for q in r_['as-is gap = second real - as simulated (points)'][1]]}")
        for lab, v in r_["variants"].items():
            if "gap = second real - variant (points)" in v:
                sh = v["share of the as-is gap closed"]
                print(f"  {lab:40s} runs {v['runs re-read']:4d}  gap {v['gap = second real - variant (points)'][0]:+6.2f} {[round(q, 2) for q in v['gap = second real - variant (points)'][1]]}"
                      f"  change {v['change vs as simulated (points)'][0]:+6.2f} {[round(q, 2) for q in v['change vs as simulated (points)'][1]]}"
                      f"  share {sh[0]:+.2f} {sh[1] if isinstance(sh[1], str) else [round(q, 2) for q in sh[1]]}")
            else:
                print(f"  sum of singles {v['sum of single changes (points)'][0]:+.2f} vs all together {v['all together (points)'][0]:+.2f}; interaction {v['sum - all together (interaction, points)'][0]:+.2f} {[round(q, 2) for q in v['sum - all together (interaction, points)'][1]]}")
        for k_, v_ in r_.items():
            if k_.startswith("false alarms") or k_.startswith("paired"):
                print(f"  {k_}:", v_)
    print(dest)


if __name__ == "__main__":
    main()
