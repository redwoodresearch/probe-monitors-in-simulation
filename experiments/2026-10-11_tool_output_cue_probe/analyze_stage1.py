"""Stage 1's gates (README section 3) for one probe, on Modal CPU where the rows are; only numbers and ids return.

Scores every row with the frozen probe at its chosen layer (logit = w . (x - mu) / sd, no intercept), then:
- G1 held-out implied pair accuracy (≥ 0.70, lower end > 0.60, above the task-grouped flip null's 95th; each
  mechanism ≥ 0.60); G1b held-out phrasings (≥ 0.80, lower end > 0.70, above the task-grouped flip null's 95th, as the
  frozen plan's G1b);
- G2 the neutral control: δ_cue = mean (sim - original) and δ_neutral = mean (neutral - original) on the same implied
  prefixes; δ_cue's lower end > 0; δ_neutral / δ_cue (joint bootstrap) within [-1/3, 1/3]; each neutral template's
  ratio (same prefixes) within ±1/2; the all-prefix neutral shift beside;
- G3 attack-panel implied pair accuracy, lower end > 0.5 and the point above the task-grouped flip null's 95th;
- G4 the planted check on P's runs: planted - original and planted - neutral run-score shifts above 0, and the
  symmetric rank change's lower end above m (0.07, or --m; stage 2 rechecks G4 against the m it uses).
Intervals: 2,000 main-task bootstrap draws (seed 20261011); null: 2,000 task-grouped label flips (seed 20261012).

  PYTHONPATH=src uv run modal deploy experiments/2026-10-11_tool_output_cue_probe/analyze_stage1.py
  PYTHONPATH=src uv run python experiments/2026-10-11_tool_output_cue_probe/analyze_stage1.py <probe> '{"C": <fit result>, "M": <fit result>}' <pairs extract> <planted extract> [--panel-from <probe A's pairs extract>] [--m <margin>]
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-toc_analysis_kill-2026-11-10"
VOL = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09"
DRAWS, SEED, NULL_SEED = 2000, 20261011, 20261012

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2")


@app.function(image=image, volumes={"/acts": vol}, cpu=8, memory=65536, timeout=3600, retries=0)
def analyze(runs: dict, fit: dict, own_sha: str, M: float) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != own_sha:
        raise ValueError("the deployed analysis is not the launching commit's: redeploy")
    vol.reload()
    rng = np.random.default_rng(SEED)

    def scored(run, probe):  # (meta, per-row logits) for a run, scored by a frozen probe; None if its layer wasn't read
        d = Path("/acts") / run
        if not (d / f"L{probe['layer']}.npy").exists():
            return None
        meta = json.loads((d / "meta.json").read_text())
        X = np.load(d / f"L{probe['layer']}.npy").astype(np.float64)
        return meta, ((X - np.array(probe["mu"])) / np.array(probe["sd"])) @ np.array(probe["w"])

    def boot(tasks, values):  # main-task bootstrap means of per-unit values (units grouped by task); values: units x k
        ut = sorted(set(tasks))
        ti = np.array([ut.index(t) for t in tasks])
        W = rng.multinomial(len(ut), np.full(len(ut), 1 / len(ut)), size=DRAWS).astype(float)
        cnt = np.bincount(ti, minlength=len(ut)).astype(float)
        V = np.atleast_2d(np.asarray(values, float).T).T
        return np.stack([(W @ np.bincount(ti, V[:, j], len(ut))) / (W @ cnt) for j in range(V.shape[1])], 1)

    ci = lambda x: [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]
    c = lambda a, b: float(a > b) + 0.5 * float(a == b)
    out = {}
    for readout in ("C", "M"):
        blob = (Path("/acts") / fit[readout]["run"] / "fits" / fit[readout]["fit_id"] / "probes.json").read_bytes()
        if hashlib.sha256(blob).hexdigest() != fit[readout]["probes_sha256"]:
            raise ValueError(f"{readout}: the frozen probes are not the fit result's")
        probe = json.loads(blob)["probes"]["last_token"]  # one row per dialogue: last token = per token
        meta, s = scored(runs[f"pairs_{readout}"], probe)  # pairs carry every layer
        if any(m["n_rows"] != 1 for m in meta):
            raise ValueError("a pair dialogue without exactly one row")
        by = {}
        for m, x in zip(meta, s):
            by.setdefault(m["pair"], {})[m["version"]] = (float(x), m)
        res = {"layer": probe["layer"]}

        def pairs(split, mech=None, byd=by):
            ps = [v for v in byd.values() if v["sim"][1]["split"] == split and (mech is None or v["sim"][1]["mechanism"] == mech)]
            return ps, [p["sim"][1]["task"] for p in ps], np.array([c(p["sim"][0], p["real"][0]) for p in ps])

        def accuracy(split, floor, lower, byd=by):  # pair accuracy, its bootstrap interval and the task-grouped flip null
            ps, tasks, acc = pairs(split, byd=byd)
            b = boot(tasks, acc)[:, 0]
            ut = sorted(set(tasks))
            flips = np.random.default_rng(NULL_SEED).integers(0, 2, (DRAWS, len(ut)))[:, [ut.index(t) for t in tasks]]
            null = np.where(flips, 1 - acc, acc).mean(1)
            y = np.r_[np.ones(len(ps)), np.zeros(len(ps))]
            r = {"pairs": len(ps), "tasks": len(ut), "pair accuracy": round(float(acc.mean()), 4), "95%": ci(b),
                 "null p95": round(float(np.percentile(null, 95)), 4),
                 "AUROC": round(float(roc_auc_score(y, [p["sim"][0] for p in ps] + [p["real"][0] for p in ps])), 4),
                 "pass": bool(acc.mean() >= floor and np.percentile(b, 2.5) > lower and acc.mean() > np.percentile(null, 95))}
            return r, ps

        res["G1 held-out implied"], held = accuracy("heldout", 0.70, 0.60)
        mech = {mm: float(pairs("heldout", mm)[2].mean()) for mm in sorted({p["sim"][1]["mechanism"] for p in held})}
        res["G1 held-out implied"] |= {"by mechanism": {k: round(v, 4) for k, v in mech.items()},
                                       "pass": res["G1 held-out implied"]["pass"] and min(mech.values()) >= 0.60}
        res["G1b held-out phrasings"], phr = accuracy("heldout_phrasing", 0.80, 0.70)
        res["G1b held-out phrasings"]["by phrasing"] = {mm: round(float(pairs("heldout_phrasing", mm)[2].mean()), 4)
                                                        for mm in sorted({p["sim"][1]["mechanism"] for p in phr})}
        allp = held + phr  # G2 decides on the 240 implied prefixes (tasks resampled among theirs); all 360 beside
        tasks = [p["sim"][1]["task"] for p in held]
        cue, neu = np.array([p["sim"][0] - p["original"][0] for p in held]), np.array([p["neutral"][0] - p["original"][0] for p in held])
        b = boot(tasks, np.c_[cue, neu])  # one set of draws for both shifts
        d_cue, d_neu = b[:, 0], b[:, 1]
        ratio = d_neu / d_cue
        point = neu.mean() / cue.mean()
        tk = np.array([p["neutral"][1].get("neutral_k") for p in held])
        tmpl = {k: float(neu[tk == k].mean() / cue[tk == k].mean()) for k in sorted(set(tk.tolist()))}  # both shifts on template k's prefixes
        tmpl_ci = {}  # each template's ratio on one set of joint draws with the cue shift; reported, the points decide (named caveat)
        for k in tmpl:
            bk = boot(tasks, np.c_[cue * (tk == k), neu * (tk == k)])
            tmpl_ci[k] = ci(bk[:, 1] / bk[:, 0])
        res["G2 neutral control"] = {"delta_cue": round(float(cue.mean()), 4), "delta_cue 95%": ci(d_cue),
                                     "delta_neutral": round(float(neu.mean()), 4), "ratio": round(float(point), 4), "ratio 95%": ci(ratio),
                                     "delta_neutral, all 360 held-out prefixes (beside)": round(float(np.mean([p["neutral"][0] - p["original"][0] for p in allp])), 4),
                                     "ratio by template": {k: round(v, 4) for k, v in tmpl.items()}, "ratio by template 95% (reported)": tmpl_ci,
                                     "neutral above real counterpart (share)": round(float(np.mean([c(p["neutral"][0], p["real"][0]) for p in allp])), 4),
                                     "neutral - real (mean)": round(float(np.mean([p["neutral"][0] - p["real"][0] for p in allp])), 4),
                                     "pass": bool(np.percentile(d_cue, 2.5) > 0 and -1 / 3 <= np.percentile(ratio, 2.5) and np.percentile(ratio, 97.5) <= 1 / 3
                                                  and all(abs(v) <= 0.5 for v in tmpl.values()))}
        res["shortcuts (held-out implied)"] = {
            "length-only guess accuracy": round(float(np.mean([c(p["sim"][1]["len_diff"], 0) for p in held])), 4),
            "accuracy by |length difference| (0-2, 3-6, >6)": [round(float(np.mean([c(p["sim"][0], p["real"][0]) for p in held if lo <= abs(p["sim"][1]["len_diff"]) <= hi] or [np.nan])), 4)
                                                               for lo, hi in ((0, 2), (3, 6), (7, 10**6))],
            "accuracy by cue-to-readout tokens (<=8, 9-16, >16)": [round(float(np.mean([c(p["sim"][0], p["real"][0]) for p in held if lo <= p["sim"][1]["distance"] <= hi] or [np.nan])), 4)
                                                                  for lo, hi in ((0, 8), (9, 16), (17, 10**6))]}
        if not runs.get(f"panel_{readout}"):
            raise ValueError("no attack panel run: G3 cannot be read")
        # G3: the attack panel (probe A's extraction, every layer), scored by this probe
        pm, ps_ = scored(runs[f"panel_{readout}"], probe)
        pby = {}
        for m, x in zip(pm, ps_):
            pby.setdefault(m["pair"], {})[m["version"]] = (float(x), m)
        res["G3 attack panel implied"], _ = accuracy("panel", 0.0, 0.5, pby)  # lower end above 0.5 and above the null's 95th
        sp = scored(runs[f"planted_{readout}"], probe) if runs.get(f"planted_{readout}") else None
        if sp is None and readout == "C":
            raise ValueError("the planted runs were not read at the content probe's layer: G4 cannot be read")
        if sp is None:
            res["G4 planted check"] = {"not read": f"the planted runs were read at the content layer, not M's layer {probe['layer']}"}
        else:  # G4: the planted check, run scores over matched outputs
            rm, rs = sp
            rows, start = {}, 0
            for m in rm:
                k2s = {k: float(x) for k, x in zip(map(tuple, m["keys"]), rs[start:start + m["n_rows"]]) if np.isfinite(x)}  # M: NaN past the cut
                rows[(m["cell"], m["arm"], m["version"])] = (m, k2s, m.get("task") or m.get("main_task"))
                start += m["n_rows"]
            score = {}
            for (cell, arm, v), (m, k2s, task) in rows.items():
                if v != "original":
                    continue
                vs = {x: rows.get((cell, arm, x)) for x in ("original", "planted", "neutral", "planted_every4")}
                if any(x is None for x in vs.values()):
                    raise ValueError(f"{cell} {arm}: a version is missing")
                common = set(vs["original"][1]) & set(vs["planted"][1]) & set(vs["neutral"][1])
                common4 = set(vs["original"][1]) & set(vs["planted_every4"][1])
                if common and common4:
                    score[(cell, arm)] = {x: float(np.mean([vs[x][1][k] for k in (common4 if x == "planted_every4" else common)])) for x in vs} | {
                        "original4": float(np.mean([vs["original"][1][k] for k in common4])), "task": task}
            cells = sorted({ca[0] for ca in score if (ca[0], "e1") in score and (ca[0], "e2") in score})
            runs_ = sorted(score)
            rt = [score[r]["task"] for r in runs_]
            sh = np.array([[score[r]["planted"] - score[r]["original"], score[r]["planted"] - score[r]["neutral"],
                            score[r]["planted_every4"] - score[r]["original4"]] for r in runs_])
            bs = boot(rt, sh)
            e = lambda cell, a, v: score[(cell, a)][v]
            rank = np.array([[(c(e(x, "e1", v), e(x, "e2", o)) - c(e(x, "e1", o), e(x, "e2", o))
                               + c(e(x, "e2", v), e(x, "e1", o)) - c(e(x, "e2", o), e(x, "e1", o))) / 2
                              for v, o in (("planted", "original"), ("planted_every4", "original4"))] for x in cells])
            br = boot([score[(x, "e1")]["task"] for x in cells], rank)
            swapped = np.array([(c(e(x, "e1", "original"), e(x, "e2", "planted")) - c(e(x, "e1", "planted"), e(x, "e2", "planted"))
                                 + c(e(x, "e2", "original"), e(x, "e1", "planted")) - c(e(x, "e2", "planted"), e(x, "e1", "planted"))) / 2 for x in cells])
            ct = sorted({score[(x, "e1")]["task"] for x in cells})  # one coin per main task swaps all its cells' labels
            coins = np.random.default_rng(NULL_SEED).integers(0, 2, (DRAWS, len(ct)))[:, [ct.index(score[(x, "e1")]["task"]) for x in cells]]
            g4null = np.where(coins, swapped, rank[:, 0]).mean(1)
            res["G4 planted check"] = {"runs": len(runs_), "cells": len(cells), "planted - original": round(float(sh[:, 0].mean()), 4),
                                       "95%": ci(bs[:, 0]), "planted - neutral": round(float(sh[:, 1].mean()), 4), "95% ": ci(bs[:, 1]),
                                       "rank change": round(float(rank[:, 0].mean()), 4), "rank change 95%": ci(br[:, 0]), "m": M,
                                       "rank change swap-null p95": round(float(np.percentile(g4null, 95)), 4),
                                       "every 4th: shift": round(float(sh[:, 2].mean()), 4), "every 4th: rank change": round(float(rank[:, 1].mean()), 4),
                                       "every 4th: rank change 95%": ci(br[:, 1]),
                                       "pass": bool(np.percentile(bs[:, 0], 2.5) > 0 and np.percentile(bs[:, 1], 2.5) > 0 and np.percentile(br[:, 0], 2.5) > M
                                                    and rank[:, 0].mean() > np.percentile(g4null, 95))}
        out[readout] = res
    gates = ["G1 held-out implied", "G1b held-out phrasings", "G2 neutral control", "G3 attack panel implied", "G4 planted check"]
    out["converged"] = all(f["max_grad"] <= f["accept"] for f in fit.values())
    out["m used by G4"] = M  # stage 2 rechecks G4 against its own m, from its eligible cells (README section 4, step 5)
    out["stage 1 passes (content readout decides)"] = bool(out["converged"] and all(out["C"].get(g, {}).get("pass") for g in gates))
    return out


if __name__ == "__main__":
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    st = git("status", "--porcelain", "--untracked-files=no")
    if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
        raise SystemExit("commit first: the result must record the commit of the code that made it")
    probe_name, fits, pairs_x, planted_x = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], sys.argv[4]
    px, tx = json.loads((HERE / pairs_x).read_text()), json.loads((HERE / planted_x).read_text())
    runs = {}
    panel_x = json.loads((HERE / sys.argv[sys.argv.index("--panel-from") + 1]).read_text()) if "--panel-from" in sys.argv else px
    build = json.loads((HERE / px["provenance"]["build"]).read_text())  # every extraction read here must come from this one build's files
    for x, want in ((px, build["dialogues"]["sha256"]), (panel_x, build["dialogues"]["sha256"]), (tx, build["planted"]["sha256"])):
        if x["provenance"]["build"] != px["provenance"]["build"] or x["job"].get("sha256") != want:
            raise ValueError(f"extraction {x['provenance']['stamp']} was not read from build {px['provenance']['build']}'s files")
    for name in px["assembled"]:
        if "Panel" not in name:
            runs[f"pairs_{name.split('_')[0][-1]}"] = name
    for name in panel_x["assembled"]:
        if "Panel" in name:
            runs[f"panel_{name.split('_')[0][-1]}"] = name
    for name in tx["assembled"]:
        runs[f"planted_{name.split('_')[3]}"] = name
    fit = {}
    for readout, path in fits.items():  # fit results as spawn_fit_toc.py wrote them
        f = json.loads((HERE / path).read_text())
        fit[readout] = {"run": f["run"], "fit_id": f["fit_id"], "probes_sha256": f["probes_sha256"], "max_grad": f["max_grad (all fits)"],
                        "accept": f["accept_grad"]}
    for readout in ("C", "M"):  # each fit must be the one made on this probe's own pair run, not another probe's or an older extraction
        if fit[readout]["run"] != runs[f"pairs_{readout}"] or not runs[f"pairs_{readout}"].startswith(f"toc{probe_name}{readout}_"):
            raise ValueError(f"{readout}: the fit's run {fit[readout]['run']} is not probe {probe_name}'s pair run {runs[f'pairs_{readout}']}")
    own = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    m = float(sys.argv[sys.argv.index("--m") + 1]) if "--m" in sys.argv else 0.07  # the plan's m; stage 2 passes its own
    out = modal.Function.from_name(APP, "analyze").remote(runs, fit, own, m)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"probe": probe_name, "git": git("rev-parse", "HEAD").stdout.strip(), "analysis_sha256": own, "runs": runs, "fits": fits, **out}
    path = HERE / "results" / f"stage1_{probe_name}_{stamp}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), f"\n-> {path}")
