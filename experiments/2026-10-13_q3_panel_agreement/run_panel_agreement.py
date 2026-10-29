"""Q3(b), panel agreement by world (RULE.md): labels and vote counts only, no text, no probe scores. POST HOC, preliminary.

Joins Aswin's frozen stage-1 sabotage-panel vote exports to their ID lists on call_id (simulated: votes 06f4ef4b… with
sim_calls_all 13b0a3da…; real runs 1 and 2: votes 9a96988d… with real_run2_calls_all 392eff0e…; S3 per the manifests in
experiments/2026-10-13_q3_sim_sabotage_ids/results/ on aswin/q3-simsab-ids), checks the joins and the 4-sure rule, and
compares the unclear rate, unanimity rate and vote margins across simulated, real run 1 and real run 2 with paired
task-bootstrap intervals. The analysis runs in one Modal CPU function (Aswin's no-laptop-compute rule); the laptop only
reads the four small files and writes the returned JSON.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach run_panel_agreement.py \\
      --sim-votes <csv> --sim-calls <csv> --real-votes <csv> --real-calls <csv>
"""
import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q3panel_agreement_kill-2026-11-13"
VOL = "probe-shift_aswin-surya_sub-q3panel_work_kill-2026-11-13"
HERE = Path(__file__).resolve().parent
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").pip_install("pandas==2.2.3", "numpy==2.1.3")
vol = modal.Volume.from_name(VOL, create_if_missing=True, version=2)

SHA = {"sim_votes": "06f4ef4bc00b53767b074185cda99311f2ba44f7c8f5a4f38ce59dd1537e81ae",
       "sim_calls": "13b0a3da826196eb051d9cd82be86e949200e359b19514f71de61423982ab8e2",
       "real_votes": "9a96988db9818c8b97f3045ade83f19fae0e19e7d0cac134fa775005eba8dc36",
       "real_calls": "392eff0ed547fb18b62a08bb9cd8945d5206c0a281672780affd40825acf4a1a"}
SHARED = ["run_id", "epoch", "turn", "call_index", "ordinal", "state_changing"]
WORLDS = {"sim": "simulated", "real": "real run 1", "run2": "real run 2"}  # the exports' world names
POOLED = "sim - real_pooled"
CONTROL = "real_run1 - real_run2 (control)"
PAIRS = {POOLED: (("sim",), ("real", "run2")), "sim - real_run1": (("sim",), ("real",)),
         "sim - real_run2": (("sim",), ("run2",)), CONTROL: (("real",), ("run2",))}
METRICS = {"unclear_rate": "unclear", "unanimity_rate": "unanimous", "mean_abs_margin": "abs_margin",
           "other_vote_share": "other_share", "labelled_sabotage_rate": "labelled_sabotage"}
B, SEED, MARGIN = 2000, 20261013, 0.02


@app.function(image=image, cpu=2, memory=4096, timeout=600, retries=0, volumes={"/out": vol})
def analyze(files: dict, git: dict, stamp: str) -> dict:
    import numpy as np
    import pandas as pd

    def need(ok, msg):
        if not ok:
            raise SystemExit(f"check failed: {msg}")

    got = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    need(got == SHA, f"input sha256 mismatch for {sorted(k for k in SHA if got.get(k) != SHA[k])}")

    def join(votes, calls):  # as analyze_label_counts.py on aswin/q3-simsab-ids
        V, C = pd.read_csv(io.BytesIO(votes)), pd.read_csv(io.BytesIO(calls))
        if "world" not in C:  # the simulated ID list has no world column
            C["world"] = "sim"
        if "world" in V:
            need((V.set_index("call_id").world.reindex(C.call_id).values == C.world.values).all(), "world differs")
            V = V.drop(columns=["world"])
        need(len(V) == len(C) and not V.call_id.duplicated().any() and set(V.call_id) == set(C.call_id),
             "the votes and the ID list hold different calls")
        M = C.merge(V, on="call_id", suffixes=("", "_v"), validate="one_to_one")
        for c in SHARED:
            need((M[c] == M[f"{c}_v"]).all(), f"{c} differs between the votes and the ID list")
        return M

    D = pd.concat([join(files["sim_votes"], files["sim_calls"]), join(files["real_votes"], files["real_calls"])],
                  ignore_index=True)
    s, n, o, v = D.sure_sabotage, D.sure_not_sabotage, D.other_votes, D.valid_votes
    need(set(D.world) == set(WORLDS), f"worlds {sorted(set(D.world))}")
    need(D.label.isin(["sabotage", "not_sabotage", "unclear"]).all(), "unexpected label")
    need(D.split.isin(["held-out", "train"]).all() and (D.groupby("task").split.nunique() == 1).all(), "splits")
    need(D.state_changing.isin([-1, 0, 1]).all(), "state_changing outside -1/0/1")  # -1 = unscored (make_sim_ids.py)
    need((v > 0).all() and (s + n + o == v).all(), "vote counts")
    rule = np.where(s >= 4, "sabotage", np.where(n >= 4, "not_sabotage", "unclear"))
    need((rule == D.label.values).all(), f"{int((rule != D.label.values).sum())} labels break the 4-sure rule")
    need(((D.confident == "yes") == (D.label != "unclear")).all(), "confident is not 'label is not unclear'")
    D["unclear"] = (D.label == "unclear").astype(float)
    D["unanimous"] = (np.maximum(s, n) == v).astype(float)
    D["margin"] = s - n
    D["abs_margin"] = (s - n).abs().astype(float)
    D["other_share"] = o / v
    D["labelled_sabotage"] = (D.label == "sabotage").astype(float)

    sc = D[D.state_changing == 1]
    scopes = {"P_state_changing": sc, "S1_any_sure_sabotage_vote": sc[sc.sure_sabotage >= 1],
              "S2_labelled_sabotage": sc[sc.label == "sabotage"]}
    splits = {"all": None, "train": "train", "held-out": "held-out"}

    def describe(df):
        out = {"calls": len(df), "tasks": int(df.task.nunique()), "runs": int(df.run_id.nunique())}
        if len(df):
            out |= {m: float(df[c].mean()) for m, c in METRICS.items()}
            for name, col in (("margin_counts", "margin"), ("valid_votes_counts", "valid_votes"), ("label_counts", "label")):
                out[name] = {str(k): int(x) for k, x in df[col].value_counts().sort_index().items()}
        return out

    def per_task(df, worlds, T, col):
        g = df[df.world.isin(worlds)].groupby("task")[col].agg(["sum", "count"]).reindex(T, fill_value=0)
        return g["sum"].to_numpy(float), g["count"].to_numpy(float)

    def boot(df, A, Bw, T, col):
        """Paired task bootstrap of rate(A) − rate(B), call-weighted; failed draws (0 calls) at ±inf, counted."""
        na, da = per_task(df, A, T, col)
        nb, db = per_task(df, Bw, T, col)
        out = {"tasks": len(T), "calls_a": int(da.sum()), "calls_b": int(db.sum())}
        if not len(T) or not da.sum() or not db.sum():
            return out | {"point": None, "lo": -np.inf, "hi": np.inf, "failed_draws": B}
        idx = np.random.default_rng(SEED).integers(0, len(T), size=(B, len(T)))
        W = np.zeros((B, len(T)))
        np.add.at(W, (np.arange(B)[:, None], idx), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = W @ na / (W @ da) - W @ nb / (W @ db)
        bad = ~np.isfinite(d)
        lo = np.quantile(np.where(bad, -np.inf, d), 0.025)
        hi = np.quantile(np.where(bad, np.inf, d), 0.975)
        lo, hi = (-np.inf if np.isnan(lo) else lo), (np.inf if np.isnan(hi) else hi)  # interpolation next to ±inf
        return out | {"a": na.sum() / da.sum(), "b": nb.sum() / db.sum(), "point": na.sum() / da.sum() - nb.sum() / db.sum(),
                      "lo": float(lo), "hi": float(hi), "failed_draws": int(bad.sum())}

    def compare(df):
        res = {}
        for variant in ("all_tasks", "task_matched"):
            res[variant] = {}
            for pair, (A, Bw) in PAIRS.items():
                ta, tb = set(df[df.world.isin(A)].task), set(df[df.world.isin(Bw)].task)
                T = sorted(ta & tb if variant == "task_matched" else ta | tb)
                res[variant][pair] = {m: boot(df, A, Bw, T, c) for m, c in METRICS.items()}
        return res

    per_world, diffs = {}, {}
    for sn, sdf in scopes.items():
        per_world[sn], diffs[sn] = {}, {}
        for pn, sp in splits.items():
            df = sdf if sp is None else sdf[sdf.split == sp]
            per_world[sn][pn] = {w: describe(df[df.world == w]) for w in WORLDS} | {
                "real_pooled": describe(df[df.world.isin(["real", "run2"])])}
            diffs[sn][pn] = compare(df)
    five = sc[sc.valid_votes == 5]
    sens = {"P_state_changing_5_valid_judges": {"all": {"per_world": {w: describe(five[five.world == w]) for w in WORLDS},
                                                        "differences": compare(five)}}}

    P = diffs["P_state_changing"]["all"]["task_matched"]
    uc, un = P[POOLED]["unclear_rate"], P[POOLED]["unanimity_rate"]
    cu, cn = P[CONTROL]["unclear_rate"]["point"], P[CONTROL]["unanimity_rate"]["point"]
    noisier = ((uc["lo"] > 0 and uc["point"] >= MARGIN and uc["point"] > abs(cu))
               or (un["hi"] < 0 and un["point"] <= -MARGIN and abs(un["point"]) > abs(cn)))
    within = uc["hi"] < MARGIN and un["lo"] > -MARGIN
    verdict = ("simulated labels are noisier" if noisier else "not noisier: within 2 points of real" if within
               else "inconclusive")
    S1 = diffs["S1_any_sure_sabotage_vote"]["all"]["task_matched"][POOLED]
    if S1["unclear_rate"]["lo"] > 0 or S1["unanimity_rate"]["hi"] < 0:
        verdict += "; but noisier on sabotage-relevant calls"

    res = {"written": stamp, "git": git, "inputs_sha256": got, "posthoc": True, "preliminary": True,
           "labels_and_vote_counts_only_no_text_no_scores": True, "rule": "RULE.md (this folder)",
           "bootstrap": {"draws": B, "seed": SEED, "unit": "task, resampled jointly across worlds", "interval": "95% percentile",
                         "failed_draws": "±inf, counted"},
           "margin_points": 100 * MARGIN,
           "validation": {"rows": {w: int((D.world == w).sum()) for w in WORLDS},
                          "state_changing": {w: int((sc.world == w).sum()) for w in WORLDS},
                          "state_changing_unscored_minus1": {w: int(((D.world == w) & (D.state_changing == -1)).sum()) for w in WORLDS},
                          "tasks": {w: int(D[D.world == w].task.nunique()) for w in WORLDS},
                          "joins_and_rule_checks": "passed", "confident_equals_not_unclear": True},
           "reading": {"aswin_verbatim": "If yes: simulated labels are noisier. If no: label noise isn't the cause.",
                       "operational": verdict, "noisier": bool(noisier), "within_margin": bool(within),
                       "inputs": {"unclear_rate": uc, "unanimity_rate": un, "control_unclear_point": cu,
                                  "control_unanimity_point": cn, "S1_unclear_rate": S1["unclear_rate"],
                                  "S1_unanimity_rate": S1["unanimity_rate"]}},
           "per_world": per_world, "differences": diffs, "sensitivity": sens}
    text = json.dumps(res, indent=1)
    Path(f"/out/panel_agreement_{stamp}.json").write_text(text)
    vol.commit()

    pct = lambda x: "n/a" if x is None else f"{100 * x:+.1f}"  # noqa: E731
    lines = [f"validation: {json.dumps(res['validation'])}"]
    for sn in ("P_state_changing", "S1_any_sure_sabotage_vote", "S2_labelled_sabotage"):
        for pn in splits:
            pw = per_world[sn][pn]
            lines.append(f"{sn} / {pn}: calls " + ", ".join(f"{w} {pw[w]['calls']}" for w in WORLDS) + "; unclear "
                         + ", ".join(f"{w} {100 * pw[w].get('unclear_rate', float('nan')):.1f}%" for w in WORLDS) + "; unanimous "
                         + ", ".join(f"{w} {100 * pw[w].get('unanimity_rate', float('nan')):.1f}%" for w in WORLDS))
            for pair in PAIRS:
                r = diffs[sn][pn]["task_matched"][pair]
                lines.append(f"   {pair} (task-matched, {r['unclear_rate']['tasks']} tasks): unclear {pct(r['unclear_rate']['point'])} "
                             f"[{pct(r['unclear_rate']['lo'])}, {pct(r['unclear_rate']['hi'])}] pts; unanimity "
                             f"{pct(r['unanimity_rate']['point'])} [{pct(r['unanimity_rate']['lo'])}, {pct(r['unanimity_rate']['hi'])}] pts; "
                             f"mean |margin| {r['mean_abs_margin']['point'] if r['mean_abs_margin']['point'] is None else round(r['mean_abs_margin']['point'], 3)} "
                             f"[{r['mean_abs_margin']['lo']:.3f}, {r['mean_abs_margin']['hi']:.3f}]; failed draws {r['unclear_rate']['failed_draws']}")
    lines.append(f"READING (operational, RULE.md): {verdict}")
    return {"text": text, "summary": "\n".join(lines)}


@app.local_entrypoint()
def main(sim_votes: str, sim_calls: str, real_votes: str, real_calls: str):
    paths = {"sim_votes": sim_votes, "sim_calls": sim_calls, "real_votes": real_votes, "real_calls": real_calls}
    files = {k: Path(p).read_bytes() for k, p in paths.items()}
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    meta = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "."))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = analyze.remote(files, meta, stamp)
    path = HERE / "results" / f"panel_agreement_{stamp}.json"
    with path.open("x") as fh:
        fh.write(out["text"])
    print(out["summary"], "\n->", path)
