"""Fit and validate eval-awareness probes on the Needham activations (Modal CPU; no GPU).

Per readout (asst_mean, asst_last, all_mean, all_last) and layer (0..63): one row per transcript, Apollo's recipe
(standardize, L2 logistic regression without intercept, C from the grid; primary C = 0.1 as in the templated probe).
- Held-out AUROC: 5 folds grouped by SOURCE dataset (a source is never on both sides), stratified by chat/agentic and
  label. The primary metric is the within-fold AUROC (each fold's held-out sources ranked against each other), averaged
  over the folds that hold both labels; overall, agentic only, chat only. Pooling out-of-fold scores across folds is
  biased here: labels are constant within a source, so each fold's training class balance mirrors its test fold's
  (eval share 0.32-0.70 by fold), which shifts per-fold offsets (the first run, fit_2026-10-06_172901, put TF-IDF and
  length below 0.5 that way). Pooled AUROC is kept as a secondary number. 95% intervals resample sources.
- Baselines on the same folds: TF-IDF of the rendered text within the cut, and a length-only model. Human ratings
  (the dataset's two annotators, who helped build it) for reference.
- Label-shuffle null: labels permuted across sources within the chat and agentic strata, refit with the same folds.
- The templated probe (experiments/2026-10-06_q3_offpolicy_probes, layers 16-48) scored on these transcripts, and
  each final Needham probe scored on the templated contrast pairs (call-JSON tokens; probe_shift.eval_awareness).
- Final probes on all transcripts (primary C) -> /data/probes/<probe_run>/<readout>/probe_L<l>.npz, with the cosine
  to the templated probe in raw activation space (coef / scale) next to label-shuffled probes' cosines as the null.
- The primary probe for LinuxArena, fixed here before any LinuxArena reading: the assistant readout (asst_last or
  asst_mean) and layer with the best within-fold agentic AUROC at the primary C.
- Probes already on the Volume are reused only if refitting gives the same weights (a determinism check).

  uv run modal run experiments/2026-10-06_q3_eval_awareness_needham/run_fit.py::main     (the fit)
  uv run modal run experiments/2026-10-06_q3_eval_awareness_needham/run_fit.py::family   (leave-scaffold-family-out)
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3evalaw_fit_kill-2026-11-06"
VOLUME = "probe-shift_aswin-surya_sub-q3evalaw_data_kill-2026-11-06"
Q3_VOLUME = "probe-shift_aswin-surya_sub-q3_outputs_kill-2026-11-06"   # templated probe, its pairs' activations (read only)
RUN = "2026-10-06_needham_qwen3-32b_v1"
PROBE_RUN = "needham_v1"
CFG = {
    "readouts": ["asst_last", "asst_mean", "all_last", "all_mean"],
    "primary_readouts": ["asst_last", "asst_mean"],   # the primary probe is the best of these (agentic, within-fold)
    "C_grid": [0.001, 0.01, 0.1],
    "primary_C": 0.1,                 # Apollo's reg_coeff 10, as the templated probe
    "max_iter": 2000,
    "folds": 5, "fold_seed": 0,
    "null_draws": 20, "null_layers": [8, 16, 24, 32, 40, 48, 56, 63],
    "ci_layers": [16, 24, 32, 40, 48, 56, 63], "bootstrap": 1000, "seed": 0,
    "templated_probe": {"dir": "probes/eval_awareness_v1", "layers": [16, 24, 32, 40, 48], "acts": "acts/eval_awareness_v1.npz"},
    "tfidf": {"token_pattern": r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]", "ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True, "C": 1.0},
}

app = modal.App(APP)
vol = modal.Volume.from_name(VOLUME)
q3_vol = modal.Volume.from_name(Q3_VOLUME).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2", "joblib", "threadpoolctl")
         .add_local_python_source("probe_shift"))


def oof_scores(X, y, folds, C, max_iter):
    """Out-of-fold probe scores (standardize on the training folds, logistic regression without intercept)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    s, conv = np.full(len(y), np.nan), True
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=C, fit_intercept=False, max_iter=max_iter, random_state=42).fit(sc.transform(X[tr]), y[tr])
        conv &= bool(lr.n_iter_[0] < max_iter)
        s[te] = sc.transform(X[te]) @ lr.coef_[0]
    return s, conv


def aurocs(s, y, agentic, folds, w=None):
    """Within-fold AUROC (mean over folds holding both labels) and pooled AUROC, overall / agentic / chat."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    w = np.ones(len(y)) if w is None else w
    out = {"within_fold": {}, "pooled": {}, "n_folds": {}}
    for name, m in (("all", np.ones(len(y), bool)), ("agentic", agentic), ("chat", ~agentic)):
        m = m & (w > 0)
        per = [roc_auc_score(y[m & (folds == f)], s[m & (folds == f)], sample_weight=w[m & (folds == f)])
               for f in np.unique(folds) if len(set(y[m & (folds == f)])) == 2]
        out["within_fold"][name] = round(float(np.mean(per)), 4) if per else None
        out["n_folds"][name] = len(per)
        out["pooled"][name] = round(float(roc_auc_score(y[m], s[m], sample_weight=w[m])), 4) if len(set(y[m])) == 2 else None
    return out


def boot_ci(s, y, agentic, source, folds, n, seed):
    """95% intervals of the within-fold AUROCs, resampling sources with replacement."""
    import numpy as np

    srcs = np.unique(source)
    sid = np.searchsorted(srcs, source)
    draws = np.random.default_rng(seed).integers(0, len(srcs), (n, len(srcs)))
    vals = {k: [] for k in ("all", "agentic", "chat")}
    for d in draws:
        w = np.bincount(d, minlength=len(srcs))[sid].astype(float)
        for k, v in aurocs(s, y, agentic, folds, w)["within_fold"].items():
            if v is not None:
                vals[k].append(v)
    return {k: [round(float(q), 4) for q in np.percentile(v, [2.5, 97.5])] if len(v) > n // 2 else None for k, v in vals.items()}


def permuted(y, source, agentic, rng):
    """Labels permuted across sources within the chat and agentic strata (every transcript keeps its source's label)."""
    import numpy as np

    out = y.copy()
    for stratum in (True, False):
        srcs = np.unique(source[agentic == stratum])
        lab = np.array([y[source == s][0] for s in srcs])
        for s, l in zip(srcs, rng.permutation(lab)):
            out[source == s] = l
    return out


def layer_job(readout, layer, X, y, agentic, source, folds, cfg):
    """Every fit for one (readout, layer): the C grid, the shuffle null, the final probe and its shuffled twins."""
    import numpy as np
    from threadpoolctl import threadpool_limits

    from probe_shift.offpolicy import fit_token_probe

    res = {}
    with threadpool_limits(1):
        for C in cfg["C_grid"]:
            s, conv = oof_scores(X, y, folds, C, cfg["max_iter"])
            res[f"C{C}"] = {"auroc": aurocs(s, y, agentic, folds), "converged": conv}
            if C == cfg["primary_C"]:
                primary_oof = s
        probe, conv = fit_token_probe(X, y, 1 / cfg["primary_C"], cfg["max_iter"])
        res["final_converged"] = conv
        null, null_probes = [], []
        if layer in cfg["null_layers"]:
            rng = np.random.default_rng(1000 + layer)
            for _ in range(cfg["null_draws"]):
                yp = permuted(y, source, agentic, rng)
                s, _ = oof_scores(X, yp, folds, cfg["primary_C"], cfg["max_iter"])
                null.append(aurocs(s, yp, agentic, folds)["within_fold"])
                null_probes.append(fit_token_probe(X, yp, 1 / cfg["primary_C"], cfg["max_iter"])[0])
    if layer in cfg["ci_layers"]:
        res["ci95_primary_C"] = boot_ci(primary_oof, y, agentic, source, folds, cfg["bootstrap"], cfg["seed"])
    if null:
        vals = {k: [d[k] for d in null if d[k] is not None] for k in ("all", "agentic", "chat")}
        res["null_primary_C"] = {k: {"mean": round(float(np.mean(v)), 4), "p95": round(float(np.percentile(v, 95)), 4),
                                     "max": round(float(np.max(v)), 4), "n": len(v), "values": v} if v else None for k, v in vals.items()}
    return readout, layer, res, probe, null_probes, primary_oof


def baseline_job(name, docs, L, y, folds, tf):
    """Out-of-fold TF-IDF (raw rendered text) and length-only scores on the same folds."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    s_tf, s_len = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        vec = TfidfVectorizer(token_pattern=tf["token_pattern"], ngram_range=tuple(tf["ngram_range"]), min_df=tf["min_df"],
                              sublinear_tf=tf["sublinear_tf"]).fit([d for d, k in zip(docs, tr) if k])
        clf = LogisticRegression(C=tf["C"], max_iter=2000).fit(vec.transform([d for d, k in zip(docs, tr) if k]), y[tr])
        s_tf[te] = clf.decision_function(vec.transform([d for d, k in zip(docs, te) if k]))
        mu, sd = L[tr].mean(0), L[tr].std(0) + 1e-9
        s_len[te] = LogisticRegression(max_iter=2000).fit((L[tr] - mu) / sd, y[tr]).decision_function((L[te] - mu) / sd)
    return name, s_tf, s_len


def raw_direction(p):
    import numpy as np

    w = p["coef"] / p["scaler_scale"]
    return w / np.linalg.norm(w)


@app.function(image=image, volumes={"/data": vol, "/q3": q3_vol}, cpu=32, memory=98304, timeout=4 * 3600)
def fit(cfg: dict) -> dict:
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    from probe_shift.eval_awareness import pairs
    from probe_shift.offpolicy import load_probe, probe_scores

    pdir = Path("/data/probes") / PROBE_RUN
    items = {}
    for line in Path("/data/inputs/items.jsonl").read_text().splitlines():
        it = json.loads(line)
        items[it["uid"]] = {k: v for k, v in it.items() if k not in ("ids", "assistant", "all")} | {"n_assistant": len(it["assistant"]),
                                                                                                   "n_all": len(it["all"]), "kept": len(it["ids"])}
    texts = {t["uid"]: t["text"] for t in map(json.loads, Path("/data/inputs/texts.jsonl").read_text().splitlines())}
    acts_dir = Path("/data/acts") / RUN
    uids = sorted(p.stem for p in acts_dir.glob("*.json"))
    if len(uids) != len(items):
        raise ValueError(f"{len(uids)} transcripts extracted of {len(items)}")
    arrays = {r: {} for r in cfg["readouts"]}
    for u in uids:
        z = np.load(acts_dir / f"{u}.npz")
        for r in cfg["readouts"]:
            if r in z.files:
                arrays[r][u] = z[r]
    meta = lambda us, k: np.array([items[u][k] for u in us])

    # Folds over sources, stratified by (agentic, label); every readout's subset inherits them.
    src_all = meta(uids, "source")
    strat = meta(uids, "agentic").astype(int) * 2 + meta(uids, "label")
    fold_of = dict(zip(uids, np.empty(len(uids), int)))
    for f, (_, te) in enumerate(StratifiedGroupKFold(cfg["folds"], shuffle=True, random_state=cfg["fold_seed"]).split(src_all, strat, src_all)):
        for i in te:
            fold_of[uids[i]] = f
    for s in np.unique(src_all):
        if len({fold_of[u] for u in uids if items[u]["source"] == s}) != 1:
            raise ValueError(f"source {s} split across folds")

    jobs, subsets = [], {}
    for r in cfg["readouts"]:
        us = sorted(arrays[r])
        sub = {"uids": us, "y": meta(us, "label").astype(int), "agentic": meta(us, "agentic").astype(bool),
               "source": meta(us, "source"), "folds": np.array([fold_of[u] for u in us])}
        subsets[r] = sub
        stack = np.stack([arrays[r][u] for u in us])  # [n, 64, H] fp16
        for l in range(stack.shape[1]):
            jobs.append(delayed(layer_job)(r, l, stack[:, l].astype(np.float32), sub["y"], sub["agentic"], sub["source"], sub["folds"], cfg))
        if not any(subsets[q]["uids"] == us for q in subsets if q != r):  # asst_* (all_*) readouts share their transcripts
            L = np.log1p(np.array([[items[u]["kept"], items[u]["n_all"], items[u]["n_assistant"], items[u]["n_assistant_turns"]] for u in us], float))
            jobs.insert(0, delayed(baseline_job)(r, [texts[u] for u in us], L, sub["y"], sub["folds"], cfg["tfidf"]))
    outs = Parallel(n_jobs=32, backend="loky")(jobs)

    by, probes, null_probes, oof, base_scores = {r: {} for r in cfg["readouts"]}, {}, {}, {}, {}
    for o in outs:
        if len(o) == 3:
            base_scores[o[0]] = o[1:]
            continue
        r, l, res, probe, nps, s = o
        by[r][l], probes[(r, l)], null_probes[(r, l)], oof[(r, l)] = res, probe, nps, s
    reused = 0
    for (r, l), p in probes.items():  # never overwrite: existing weights must be what this fit gives
        f = pdir / r / f"probe_L{l}.npz"
        if f.exists():
            old = load_probe(f.read_bytes())
            if any(not np.allclose(old[k], p[k], rtol=1e-5, atol=1e-7) for k in p):
                raise FileExistsError(f"{f} holds other weights; change PROBE_RUN")
            reused += 1
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, **p)

    # Baselines on the same folds, per readout subset (TF-IDF and length), and human ratings.
    baselines = {}
    for r, sub in subsets.items():
        owner = next(q for q in base_scores if subsets[q]["uids"] == sub["uids"])
        if owner in baselines:
            baselines[r] = baselines[owner]
            continue
        us, y = sub["uids"], sub["y"]
        s_tf, s_len = base_scores[owner]
        hum = np.array([np.mean([h for h in items[u]["human"] if h is not None]) if any(h is not None for h in items[u]["human"]) else np.nan for u in us])
        ok = ~np.isnan(hum)
        baselines[r] = {"n": len(us), "n_by_cell": {f"{'agentic' if a else 'chat'}/{'eval' if l else 'deploy'}": int(((sub["agentic"] == a) & (y == l)).sum())
                                                    for a in (True, False) for l in (1, 0)},
                        "sources": int(len(np.unique(sub["source"]))),
                        "tfidf": {"auroc": aurocs(s_tf, y, sub["agentic"], sub["folds"]),
                                  "ci95": boot_ci(s_tf, y, sub["agentic"], sub["source"], sub["folds"], cfg["bootstrap"], cfg["seed"])},
                        "length": {"auroc": aurocs(s_len, y, sub["agentic"], sub["folds"])},
                        "human_mean_rating": {"n": int(ok.sum()), "auroc": aurocs(hum[ok], y[ok], sub["agentic"][ok], sub["folds"][ok]) if ok.any() else None},
                        "oof": {"tfidf": [round(float(v), 4) for v in s_tf], "length": [round(float(v), 4) for v in s_len]}}

    # The templated probe on the Needham transcripts (no training), and cosines between directions.
    tp = cfg["templated_probe"]
    templ = {l: load_probe((Path("/q3") / tp["dir"] / f"probe_L{l}.npz").read_bytes()) for l in tp["layers"]}
    templated_on_needham = {r: {str(l): aurocs(probe_scores(np.stack([arrays[r][u][l] for u in subsets[r]["uids"]]), p),  # no training:
                                               subsets[r]["y"], subsets[r]["agentic"], subsets[r]["folds"])  # pooled is unbiased
                                for l, p in templ.items()} for r in cfg["readouts"]}
    cosine = {}
    for r in cfg["readouts"]:
        cosine[r] = {}
        for l, p in templ.items():
            d = raw_direction(p)
            nulls = [float(raw_direction(q) @ d) for q in null_probes[(r, l)]]
            cosine[r][str(l)] = {"cos": round(float(raw_direction(probes[(r, l)]) @ d), 4),
                                 "null_shuffled_labels": {"mean_abs": round(float(np.mean(np.abs(nulls))), 4), "max_abs": round(float(np.max(np.abs(nulls))), 4)} if nulls else None}
    cos_readouts = {f"L{l}": {f"{a}~{b}": round(float(raw_direction(probes[(a, l)]) @ raw_direction(probes[(b, l)])), 4)
                              for a, b in (("asst_last", "asst_mean"), ("asst_last", "all_last"), ("asst_mean", "all_mean"))} for l in cfg["ci_layers"]}

    # Each final Needham probe on the templated contrast pairs (per dialogue: last call-JSON token and mean over them).
    z = np.load(Path("/q3") / tp["acts"])
    owner = z["dialogue"]
    dl = pairs(0)
    if owner.max() + 1 != len(dl):
        raise ValueError("templated activations do not match probe_shift.eval_awareness.pairs(0)")
    lab, fam = np.array([d["label"] for d in dl]), np.array([d["family"] for d in dl])
    last_row = np.array([np.flatnonzero(owner == i).max() for i in range(len(dl))])
    on_pairs = {}
    for r in cfg["readouts"]:
        on_pairs[r] = {}
        for l in tp["layers"]:
            s_tok = probe_scores(z[f"L{l}"], probes[(r, l)])
            s_mean = np.bincount(owner, weights=s_tok) / np.bincount(owner)
            on_pairs[r][str(l)] = {f"{fam_}_{agg}": round(float(roc_auc_score(lab[fam == fam_], s[fam == fam_])), 4)
                                   for fam_ in ("explicit", "implicit") for agg, s in (("last_token", s_tok[last_row]), ("token_mean", s_mean))}

    pc = f"C{cfg['primary_C']}"
    agentic_auc = {(r, l): by[r][l][pc]["auroc"]["within_fold"]["agentic"] for r in cfg["primary_readouts"] for l in by[r]}
    primary = max(agentic_auc, key=agentic_auc.get)
    best_layer = {r: int(max((l for rr, l in agentic_auc if rr == r), key=lambda l: agentic_auc[(r, l)])) for r in cfg["primary_readouts"]}
    keep_layers = sorted(set(cfg["ci_layers"]) | set(best_layer.values()))
    oof_all = {r: {"uids": subsets[r]["uids"], **{f"L{l}": [round(float(v), 4) for v in oof[(r, l)]] for l in range(64)}} for r in cfg["readouts"]}
    oof_rows = {r: {"uids": v["uids"], **{f"L{l}": v[f"L{l}"] for l in keep_layers}} for r, v in oof_all.items()}
    transcripts = [{"uid": u, "id": items[u]["id"], "source": items[u]["source"], "label": items[u]["label"], "agentic": items[u]["agentic"],
                    "fold": int(fold_of[u]), "has_assistant_token": u in arrays["asst_last"]} for u in uids]
    result = {"by_readout": {r: {str(l): v for l, v in sorted(d.items())} for r, d in by.items()}, "baselines": baselines,
              "templated_on_needham": templated_on_needham, "cosine_to_templated": cosine, "cosine_between_readouts": cos_readouts,
              "needham_probe_on_templated_pairs": on_pairs, "primary": {"readout": primary[0], "layer": int(primary[1])},
              "best_layer_per_readout": best_layer, "primary_layer": int(primary[1]),
              "primary_rule": f"best within-fold agentic AUROC at C={cfg['primary_C']} over {cfg['primary_readouts']} x layers 0-63",
              "probes_reused": reused, "transcripts": transcripts,
              "probe_dir": f"{VOLUME}/probes/{PROBE_RUN}", "activations": f"{VOLUME}/acts/{RUN}"}
    (pdir / f"fit_{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')}.json").write_text(
        json.dumps({"config": cfg, "result": result, "oof": oof_all}))
    vol.commit()
    return result | {"oof": oof_rows}


@app.local_entrypoint()
def main():
    from probe_shift.offpolicy import provenance

    res = fit.remote(CFG)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"fit_{stamp}.json"
    path.write_text(json.dumps({"config": CFG, "app": APP, "provenance": provenance(__file__, HERE.parent.parent / "src/probe_shift/offpolicy.py")} | res, indent=1))
    pc = f"C{CFG['primary_C']}"
    wf = lambda a: a["within_fold"]
    for r in CFG["readouts"]:
        b = res["baselines"][r]
        print(f"\n== {r}: n={b['n']} {b['n_by_cell']} sources={b['sources']} | within-fold TF-IDF {wf(b['tfidf']['auroc'])} "
              f"length {wf(b['length']['auroc'])} | human (pooled) {b['human_mean_rating']['auroc']['pooled'] if b['human_mean_rating']['auroc'] else None}")
        for l in sorted(set(CFG["ci_layers"]) | set(res["best_layer_per_readout"].values())):
            d = res["by_readout"][r][str(l)]
            n = d.get("null_primary_C") or {}
            print(f"  L{l}: within-fold {wf(d[pc]['auroc'])} ci {d.get('ci95_primary_C')} | null agentic "
                  f"{ {k: n['agentic'][k] for k in ('mean', 'p95', 'max')} if n.get('agentic') else None} | pooled {d[pc]['auroc']['pooled']} "
                  f"| agentic C.001 {wf(d['C0.001']['auroc'])['agentic']} C.01 {wf(d['C0.01']['auroc'])['agentic']}")
    for r in ("asst_last", "asst_mean"):
        print(f"\n{r}: templated probe on Needham (pooled):", json.dumps({l: v["pooled"] for l, v in res["templated_on_needham"][r].items()}))
        print(f"{r}: cosine to templated:", json.dumps(res["cosine_to_templated"][r]))
        print(f"{r}: Needham probe on templated pairs:", json.dumps(res["needham_probe_on_templated_pairs"][r]))
    print(f"\nprimary: {res['primary']} ({res['primary_rule']}); best layer per readout {res['best_layer_per_readout']}; "
          f"probes reused {res['probes_reused']}\n-> {path}")


# Leave-scaffold-family-out: sources that share a scaffold sit on both sides of the label, so a held-out family asks
# whether the probe ranks evaluation above deployment within one scaffold, which source style alone cannot do.
FAMILIES = {"smolagents": {"eval": ["smol_gaia_manager", "smol_gaia_search"], "deploy": ["smol_real_manager", "smol_real_search"]},
            "aider": {"eval": ["aider_full"], "deploy": ["aider_commit", "aider_multi", "aider_manual"]}}
FAMILY_LAYERS = [16, 24, 27, 28, 32, 40, 48, 56, 63]


def family_job(readout, layer, X, y, agentic, source, test, cfg, n_null):
    """Train on every source outside the family, AUROC on the family; null: training labels permuted across sources."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from threadpoolctl import threadpool_limits

    from probe_shift.offpolicy import fit_token_probe, probe_scores

    with threadpool_limits(1):
        p, _ = fit_token_probe(X[~test], y[~test], 1 / cfg["primary_C"], cfg["max_iter"])
        auc = roc_auc_score(y[test], probe_scores(X[test], p))
        rng, null = np.random.default_rng(2000 + layer), []
        for _ in range(n_null):
            yp = permuted(y[~test], source[~test], agentic[~test], rng)
            q, _ = fit_token_probe(X[~test], yp, 1 / cfg["primary_C"], cfg["max_iter"])
            null.append(roc_auc_score(y[test], probe_scores(X[test], q)))
    return readout, layer, round(float(auc), 4), [round(float(v), 4) for v in null]


@app.function(image=image, volumes={"/data": vol}, cpu=32, memory=65536, timeout=2 * 3600)
def family_holdout(cfg: dict, n_null: int) -> dict:
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    items = {it["uid"]: it for it in map(json.loads, Path("/data/inputs/items.jsonl").read_text().splitlines())}
    texts = {t["uid"]: t["text"] for t in map(json.loads, Path("/data/inputs/texts.jsonl").read_text().splitlines())}
    us = sorted(u for u, it in items.items() if it["assistant"])
    y = np.array([items[u]["label"] for u in us])
    ag = np.array([items[u]["agentic"] for u in us], bool)
    src = np.array([items[u]["source"] for u in us])
    X = {r: np.stack([np.load(Path("/data/acts") / RUN / f"{u}.npz")[r][FAMILY_LAYERS] for u in us]).astype(np.float32)
         for r in cfg["primary_readouts"]}
    out = {}
    for fam, sides in FAMILIES.items():
        test = np.isin(src, sides["eval"] + sides["deploy"])
        if set(y[test & np.isin(src, sides["eval"])]) != {1} or set(y[test & np.isin(src, sides["deploy"])]) != {0}:
            raise ValueError(f"{fam}: family labels are not as listed")
        jobs = [delayed(family_job)(r, l, X[r][:, i], y, ag, src, test, cfg, n_null) for r in X for i, l in enumerate(FAMILY_LAYERS)]
        res = Parallel(n_jobs=32, backend="loky")(jobs)
        docs = [texts[u] for u in us]
        tf = cfg["tfidf"]
        vec = TfidfVectorizer(token_pattern=tf["token_pattern"], ngram_range=tuple(tf["ngram_range"]), min_df=tf["min_df"],
                              sublinear_tf=tf["sublinear_tf"]).fit([d for d, k in zip(docs, ~test) if k])
        clf = LogisticRegression(C=tf["C"], max_iter=2000).fit(vec.transform([d for d, k in zip(docs, ~test) if k]), y[~test])
        s_tf = clf.decision_function(vec.transform([d for d, k in zip(docs, test) if k]))
        out[fam] = {"sources": sides, "n_eval": int((y[test] == 1).sum()), "n_deploy": int((y[test] == 0).sum()),
                    "tfidf_auroc": round(float(roc_auc_score(y[test], s_tf)), 4),
                    "probe": {r: {str(l): {"auroc": a, "null_mean": round(float(np.mean(nl)), 4), "null_p95": round(float(np.percentile(nl, 95)), 4),
                                           "null_max": max(nl), "null": nl} for rr, l, a, nl in res if rr == r} for r in X}}
    return out


@app.local_entrypoint()
def family(n_null: int = 20):
    """uv run modal run experiments/2026-10-06_q3_eval_awareness_needham/run_fit.py::family"""
    from probe_shift.offpolicy import provenance

    res = family_holdout.remote(CFG, n_null)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"family_holdout_{stamp}.json"
    path.write_text(json.dumps({"config": CFG | {"families": FAMILIES, "layers": FAMILY_LAYERS, "n_null": n_null}, "app": APP,
                                "provenance": provenance(__file__)} | res, indent=1))
    for fam, r in res.items():
        print(f"\n== {fam}: {r['n_eval']} eval vs {r['n_deploy']} deploy; TF-IDF {r['tfidf_auroc']}")
        for rd, by_l in r["probe"].items():
            print(f"  {rd}: " + "  ".join(f"L{l} {v['auroc']:.3f} (null p95 {v['null_p95']:.2f})" for l, v in by_l.items()))
    print(f"-> {path}")
