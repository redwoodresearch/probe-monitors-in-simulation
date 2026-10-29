"""Vulnerability probe trained on real fixes: SafeCoder + SVEN vulnerable vs fixed functions, with grouped 5-fold
cross-validation by repository; Emergent Misalignment (EM) is the reverse transfer test. Modal CPU only.

Reads the activations run_vuln_probe.py extracted (acts/vuln_v1.npz: the call_end and mean-over-code vectors of every
item), after checking that this script rebuilds exactly the items they came from (fingerprint). One row per function,
not per token: a vulnerable and a fixed version differ in a few lines, so per-token labels would mostly be noise.
  - Folds: pairs grouped by repository, merged with any pair sharing a function version (probe_shift.vuln.repo_groups),
    5 folds balanced by size (seed 0). Both versions of a pair always share a fold.
  - Probe: standardized features, L2 logistic regression without intercept (Apollo's recipe), C from a small grid.
    Readouts: call_end (LinuxArena's readout) and the mean over code tokens.
  - Out-of-fold AUROC with a 95% interval resampling repository groups, pair accuracy, per language; a label-shuffle
    null (each pair's labels swapped with probability 1/2) at the chosen setting; TF-IDF (raw text) on the same folds.
  - The (layer, C) for LinuxArena: best out-of-fold call_end AUROC (chosen on the same folds, so slightly optimistic).
  - Final probes: all SafeCoder pairs, each layer at its best call_end C -> probes/<probe_run>/probe_L{l}.npz;
    scored on all 6,000 EM samples.

  PYTHONPATH=src uv run --no-project --python 3.13 --with 'modal>=1.2' --with numpy modal run \
      experiments/2026-10-06_q3_offpolicy_probes/run_vuln_safecoder_probe.py --data-dir <dir with em/ and safecoder/>
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3_vuln-sc_kill-2026-11-06"
OUTPUT = "probe-shift_aswin-surya_sub-q3_outputs_kill-2026-11-06"
PIP = ["numpy", "scikit-learn==1.7.2", "joblib", "threadpoolctl"]
CFG = {
    # The extraction these activations came from (run_vuln_probe.py CFG); the fingerprint must match.
    "acts_run": "vuln_v1", "model": "Qwen/Qwen3-32B", "dtype": "bfloat16", "layers": [16, 24, 32, 40, 48, 56, 63],
    "readout": "block_output", "tokens_per_sample": 8, "seed": 0,
    "em_per_class": 3000, "em_val_fraction": 0.2, "em_group_max_key_share": 8, "max_code_chars": 6000,
    # This run.
    "probe_run": "vuln_sc_v1", "folds": 5, "C_grid": [0.001, 0.01, 0.1], "max_iter": 2000, "bootstrap": 1000,
    "shuffle_null": 5,
}
EXTRACT_KEYS = ("model", "dtype", "layers", "readout", "tokens_per_sample", "seed")

app = modal.App(APP)
out_vol = modal.Volume.from_name(OUTPUT)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).add_local_python_source("probe_shift")


def oof_scores(X, y, fold, C, max_iter):
    """Out-of-fold scores of a standardized, intercept-free L2 logistic regression."""
    import numpy as np
    from threadpoolctl import threadpool_limits

    from probe_shift.offpolicy import fit_token_probe, probe_scores

    s = np.full(len(y), np.nan)
    with threadpool_limits(2):
        for f in np.unique(fold):
            probe, _ = fit_token_probe(X[fold != f], y[fold != f], 1 / C, max_iter)
            s[fold == f] = probe_scores(X[fold == f], probe)
    return s


def make_folds(group, n_folds: int, seed: int):
    """Repository groups shuffled (seeded), then each, largest first, to the currently smallest fold. Returns the fold
    of each row and the generator (cv keeps drawing its bootstrap from it)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    sizes = {g: int((group == g).sum()) for g in np.unique(group)}
    order = rng.permutation(sorted(sizes))
    load, fold_of = np.zeros(n_folds), {}
    for g in sorted(order, key=lambda g: -sizes[g]):  # ties keep the shuffled order (stable sort)
        fold_of[g] = int(load.argmin())
        load[fold_of[g]] += sizes[g]
    return np.array([fold_of[g] for g in group]), rng


@app.function(image=image, volumes={"/out": out_vol}, cpu=32, memory=65536, timeout=3600)
def mean_null(cfg: dict, meta: dict, items_fingerprint: str, best_c: dict, draws: int) -> dict:
    """Label-shuffle null for the mean-over-code probes: the same grouped folds and per-layer C, each pair's labels
    swapped with probability 1/2 per draw; out-of-fold AUROC next to the real one."""
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score

    out_vol.reload()
    path = Path("/out") / "acts" / f"{cfg['acts_run']}.npz"
    if path.with_suffix(".fingerprint").read_text() != items_fingerprint:
        raise ValueError(f"{path} was extracted from other items or settings")
    z = np.load(path)
    src = np.array(meta["source"])
    sc = np.char.startswith(src.astype(str), "safecoder")
    y, pair, group = np.array(meta["label"])[sc], np.array(meta["pair"])[sc], np.array(meta["group"])[sc]
    fold, _ = make_folds(group, cfg["folds"], cfg["seed"])
    pairs = sorted(set(pair))
    ys = [y]
    for k in range(draws):  # each pair's labels swapped with probability 1/2 (same seeds as cv's null)
        flip = dict(zip(pairs, np.random.default_rng(100 + k).random(len(pairs)) < 0.5))
        ys.append(np.array([1 - v if flip[p] else v for v, p in zip(y, pair)]))
    out = {}
    for l, C in best_c.items():
        X = z[f"L{l}_mean"].astype(np.float32)[sc]
        s = Parallel(n_jobs=16)(delayed(oof_scores)(X, yy, fold, C, cfg["max_iter"]) for yy in ys)
        aucs = [float(roc_auc_score(yy, ss)) for yy, ss in zip(ys, s)]
        null = np.array(aucs[1:])
        out[str(l)] = {"C": C, "auroc": round(aucs[0], 4), "null_mean": round(float(null.mean()), 4),
                       "null_p95": round(float(np.percentile(null, 95)), 4), "null_max": round(float(null.max()), 4), "draws": draws}
    return out


@app.function(image=image, volumes={"/out": out_vol}, cpu=32, memory=65536, timeout=3600)
def cv(cfg: dict, meta: dict, texts: list[str], items_fingerprint: str) -> dict:
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from probe_shift.offpolicy import fit_token_probe, probe_scores

    out_vol.reload()
    path = Path("/out") / "acts" / f"{cfg['acts_run']}.npz"
    if path.with_suffix(".fingerprint").read_text() != items_fingerprint:
        raise ValueError(f"{path} was extracted from other items or settings")
    pdir = Path("/out") / "probes" / cfg["probe_run"]
    if pdir.exists() and any(pdir.iterdir()):
        raise FileExistsError(f"{pdir} already holds probes; change cfg['probe_run']")
    z = np.load(path)
    src, y_all = np.array(meta["source"]), np.array(meta["label"])
    sc, em = np.char.startswith(src.astype(str), "safecoder"), src == "em"
    y, lang = y_all[sc], np.array(meta["language"])[sc]
    pair, group = np.array(meta["pair"])[sc], np.array(meta["group"])[sc]

    fold, rng = make_folds(group, cfg["folds"], cfg["seed"])
    sizes = {g: int((group == g).sum()) for g in np.unique(group)}
    gid = {g: i for i, g in enumerate(sorted(sizes))}
    gi = np.array([gid[g] for g in group])
    W = np.stack([np.bincount(d, minlength=len(gid)) for d in rng.integers(0, len(gid), (cfg["bootstrap"], len(gid)))]).astype(float)

    def metrics(s):
        vals = [roc_auc_score(y, s, sample_weight=w[gi]) for w in W if len(set(y[w[gi] > 0])) == 2]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        by = {}
        for p, v, lab in zip(pair, s, y):
            by.setdefault(p, {})[lab] = v
        return {"auroc": round(float(roc_auc_score(y, s)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "pair_acc": round(float(np.mean([d[1] > d[0] for d in by.values() if len(d) == 2])), 4),
                "by_language": {g: round(float(roc_auc_score(y[lang == g], s[lang == g])), 4) for g in sorted(set(lang))}}

    X = {(l, r): z[f"L{l}_{'last' if r == 'call_end' else 'mean'}"].astype(np.float32) for l in cfg["layers"] for r in ("call_end", "mean_code")}
    settings = [(l, r, C) for l in cfg["layers"] for r in ("call_end", "mean_code") for C in cfg["C_grid"]]
    scores = Parallel(n_jobs=16)(delayed(oof_scores)(X[(l, r)][sc], y, fold, C, cfg["max_iter"]) for l, r, C in settings)
    grid = {f"L{l}/{r}/C{C}": metrics(s) for (l, r, C), s in zip(settings, scores)}

    # TF-IDF (raw text) on the same folds, vocabulary from each training fold only.
    t_sc = [t for t, keep in zip(texts, sc) if keep]
    s_tfidf = np.full(len(y), np.nan)
    for f in range(cfg["folds"]):
        tr = [t for t, k in zip(t_sc, fold != f) if k]
        vec = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tr)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(vec.transform(tr), y[fold != f])
        s_tfidf[fold == f] = clf.decision_function(vec.transform([t for t, k in zip(t_sc, fold == f) if k]))
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(t_sc)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(vec.transform(t_sc), y)
    t_em = [t for t, keep in zip(texts, em) if keep]
    tfidf = {"cv": metrics(s_tfidf), "em_auroc": round(float(roc_auc_score(y_all[em], clf.decision_function(vec.transform(t_em)))), 4)}

    # Best C per layer (call_end readout); the primary layer is the best of those.
    best_c = {l: max(cfg["C_grid"], key=lambda C: grid[f"L{l}/call_end/C{C}"]["auroc"]) for l in cfg["layers"]}
    primary = max(cfg["layers"], key=lambda l: grid[f"L{l}/call_end/C{best_c[l]}"]["auroc"])
    null = []
    for k in range(cfg["shuffle_null"]):  # each pair's labels swapped with probability 1/2
        flip = {p: f for p, f in zip(sorted(set(pair)), np.random.default_rng(100 + k).random(len(set(pair))) < 0.5)}
        ys = np.array([1 - v if flip[p] else v for v, p in zip(y, pair)])
        s = oof_scores(X[(primary, "call_end")][sc], ys, fold, best_c[primary], cfg["max_iter"])
        null.append(float(roc_auc_score(ys, s)))

    # Final probes on all SafeCoder pairs; reverse transfer to EM.
    pdir.mkdir(parents=True, exist_ok=True)
    final = {}
    for l in cfg["layers"]:
        probe, conv = fit_token_probe(X[(l, "call_end")][sc], y, 1 / best_c[l], cfg["max_iter"])
        np.savez(pdir / f"probe_L{l}.npz", **probe)
        mprobe, _ = fit_token_probe(X[(l, "mean_code")][sc], y, 1 / best_c[l], cfg["max_iter"])
        final[str(l)] = {"C": best_c[l], "converged": conv,
                         "em_auroc_call_end": round(float(roc_auc_score(y_all[em], probe_scores(X[(l, "call_end")][em], probe))), 4),
                         "em_auroc_mean_code": round(float(roc_auc_score(y_all[em], probe_scores(X[(l, "mean_code")][em], mprobe))), 4)}
    result = {"cv_grid": grid, "tfidf": tfidf, "best_C_call_end": {str(l): c for l, c in best_c.items()}, "primary_layer": int(primary),
              "label_shuffle_null_primary": {"n": len(null), "mean": round(float(np.mean(null)), 4), "max": round(float(np.max(null)), 4)},
              "final": final, "folds": {"n_groups": len(gid), "pairs_per_fold": [int(len(set(pair[fold == f]))) for f in range(cfg["folds"])]},
              "probe_dir": f"{OUTPUT}/probes/{cfg['probe_run']}"}
    (pdir / "fit.json").write_text(json.dumps({"config": cfg, "result": result}))
    out_vol.commit()
    return result


@app.function(image=image, volumes={"/out": out_vol}, cpu=16, memory=65536, timeout=1800)
def fit_mean(cfg: dict, meta: dict, items_fingerprint: str, best_c: dict, probe_run: str) -> dict:
    """Mean-over-code probes on all SafeCoder pairs (one per layer, at that layer's best out-of-fold C for this readout),
    for reading LinuxArena edits by the mean over their new code. EM AUROC as a check."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from probe_shift.offpolicy import fit_token_probe, probe_scores

    out_vol.reload()
    path = Path("/out") / "acts" / f"{cfg['acts_run']}.npz"
    if path.with_suffix(".fingerprint").read_text() != items_fingerprint:
        raise ValueError(f"{path} was extracted from other items or settings")
    pdir = Path("/out") / "probes" / probe_run
    if pdir.exists() and any(pdir.iterdir()):  # same acts and per-layer C: return the stored fit (to record it locally)
        stored = json.loads((pdir / "fit.json").read_text())
        if stored["config"] == cfg and {str(k): v for k, v in stored["best_c"].items()} == {str(k): v for k, v in best_c.items()}:
            return stored["result"] | {"reused_stored_fit": True}
        raise FileExistsError(f"{pdir} already holds other probes")
    z = np.load(path)
    src, y_all = np.array(meta["source"]), np.array(meta["label"])
    sc, em = np.char.startswith(src.astype(str), "safecoder"), src == "em"
    pdir.mkdir(parents=True)
    out = {}
    for l, C in best_c.items():
        X = z[f"L{l}_mean"].astype(np.float32)
        probe, conv = fit_token_probe(X[sc], y_all[sc], 1 / C, cfg["max_iter"])
        np.savez(pdir / f"probe_L{l}.npz", **probe)
        out[str(l)] = {"C": C, "converged": conv, "em_auroc": round(float(roc_auc_score(y_all[em], probe_scores(X[em], probe))), 4)}
    (pdir / "fit.json").write_text(json.dumps({"config": cfg, "probe_run": probe_run, "items_fingerprint": items_fingerprint,
                                               "readout": "mean_code", "best_c": best_c, "result": out}))
    out_vol.commit()
    return out


@app.local_entrypoint()
def main(data_dir: str, mean_probes_from: str = "", mean_null_draws: int = 0):
    """`--mean-probes-from results/vuln_safecoder_probe_<stamp>.json`: fit and save mean-over-code probes (layers
    16-48, each at its best out-of-fold C in that run) to probes/<probe_run>_mean/, instead of the CV run."""
    from probe_shift.vuln import fingerprint, load_items, repo_groups

    em, sc, items, meta, counts = load_items(data_dir, CFG)
    if mean_probes_from:
        grid = json.loads((HERE / mean_probes_from).read_text())["cv_grid"]
        best_c = {l: max(CFG["C_grid"], key=lambda C: grid[f"L{l}/mean_code/C{C}"]["auroc"]) for l in (16, 24, 32, 40, 48)}
    if mean_probes_from and mean_null_draws:  # the label-shuffle null for those mean-over-code settings
        from probe_shift.offpolicy import provenance

        groups = repo_groups(sc)
        meta["group"] = [str(groups[p]) if p in groups else None for p in meta["pair"]]
        res = mean_null.remote(CFG, meta, fingerprint(CFG, items, EXTRACT_KEYS), best_c, mean_null_draws)
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        (HERE / "results" / f"vuln_safecoder_mean_null_{stamp}.json").write_text(json.dumps(
            {"config": CFG, "cv_results": mean_probes_from, "best_c": best_c, "by_layer": res,
             "provenance": provenance(__file__, HERE.parent.parent / "src/probe_shift/vuln.py", HERE.parent.parent / "src/probe_shift/offpolicy.py")}, indent=1))
        print(json.dumps(res))
        return
    if mean_probes_from:
        res = fit_mean.remote(CFG, meta, fingerprint(CFG, items, EXTRACT_KEYS), best_c, CFG["probe_run"] + "_mean")
        from probe_shift.offpolicy import provenance

        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        record = {"config": CFG, "readout": "mean_code", "cv_results": mean_probes_from, "best_c": best_c,
                  "probe_dir": f"{OUTPUT}/probes/{CFG['probe_run']}_mean", "em_auroc_by_layer": res,
                  "provenance": provenance(__file__, HERE.parent.parent / "src/probe_shift/vuln.py", HERE.parent.parent / "src/probe_shift/offpolicy.py")}
        (HERE / "results" / f"vuln_safecoder_mean_probes_{stamp}.json").write_text(json.dumps(record, indent=1))
        print(json.dumps(res))
        return
    groups = repo_groups(sc)
    meta["group"] = [str(groups[p]) if p in groups else None for p in meta["pair"]]
    print(f"SafeCoder {len(groups)} pairs in {len(set(groups.values()))} repository groups; EM {len(em)}; {counts}", flush=True)
    res = cv.remote(CFG, meta, [it["code"] for it in items], fingerprint(CFG, items, EXTRACT_KEYS))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"vuln_safecoder_probe_{stamp}.json"
    from probe_shift.offpolicy import provenance

    prov = provenance(__file__, HERE.parent.parent / "src/probe_shift/vuln.py", HERE.parent.parent / "src/probe_shift/offpolicy.py")
    path.write_text(json.dumps({"config": CFG | {"modal": {"app": APP, "pip": PIP, "output_volume": OUTPUT}}, "counts": counts,
                                "provenance": prov} | res, indent=1))
    f = lambda m: f"{m['auroc']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] pair acc {m['pair_acc']:.2f}"
    print(f"TF-IDF (raw text): CV {f(res['tfidf']['cv'])}; EM {res['tfidf']['em_auroc']}")
    for l in CFG["layers"]:
        c = res["best_C_call_end"][str(l)]
        print(f"L{l} (C={c}): call_end CV {f(res['cv_grid'][f'L{l}/call_end/C{c}'])} | mean_code CV "
              f"{f(res['cv_grid'][f'L{l}/mean_code/C{c}'])} | EM transfer call_end {res['final'][str(l)]['em_auroc_call_end']} "
              f"mean_code {res['final'][str(l)]['em_auroc_mean_code']}")
    print(f"primary layer {res['primary_layer']}; shuffle null {res['label_shuffle_null_primary']}\n-> {path}")
