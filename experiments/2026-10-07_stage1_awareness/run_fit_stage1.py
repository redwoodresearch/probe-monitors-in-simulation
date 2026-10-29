"""Stage 1 validation fits (pre-registration PR #82: rules 2-3, the Stage 1 table, amendment 1), Modal CPU, every core.
No LinuxArena data is read: only contrast sets, implicit cases, Needham and the vulnerability items. One job per layer
(0-63) writes per-layer statistics; analyze_fit_stage1.py chooses layers, applies the selection-aware nulls and writes
the verdicts the freeze amendment needs. Every probe's weights at every layer are saved to the Volume.

Per layer:
- Testing vs Conversation, monitoring, stakes, impactful command: held-out framing families.
  - Each family is held out once; C ∈ {0.001, 0.01, 0.1} is chosen inside that outer fold by 3-fold CV on the
    training families, grouped by scenario (`task`).
  - Standardized L2 logistic regression, no intercept. Mean difference (raw) and shrinkage LDA (Ledoit-Wolf) are
    reported alongside.
  - Statistic: the mean (and min) held-out-family AUROC.
- Null replicate k (k < 100): the same evaluation, C choice included, with each pair's labels swapped with probability
  1/2. The swaps come from a seed of (probe, k) only, so every layer sees the same swaps. Taking the max over layers per
  k (analyze step) repeats the layer choice.
- Implicit cases (monitoring, stakes): trained on every stated family (C by grouped CV), tested on the implicit set.
  Stakes is also tested within each command-impact stratum. The null is the same refit on swapped labels per k.
- Testing vs Conversation:
  - friendliness: with the card family held out, AUROC within its friendly and within its terse items;
  - within-aider: aider_full (evaluation) vs aider_commit / multi / manual (deployment) at asst_last, with a null from
    the swapped refits.
- Twin (amendment 1 as decided): the primary trains only on Needham transcripts with tool calls (the mean of their
  rule-1 rows). Secondary transfer checks: a mixed version (rule-1 rows, else asst_last) and a text-only version
  (asst_last).
  - Held out by source: 5 source-grouped folds, aider excluded from training. C is chosen inside each fold by
    source-grouped CV. Statistic: the mean of per-fold AUROCs.
  - Null: 100 source-level label permutations seeded by k, for the primary and the mixed version (the fallback).
  - Within-aider at asst_last for each version (a validity exception).
- Vulnerability (every 4th layer, plus 63): trained on EM train, C = 0.1 (the #42 recipe). Reported: EM val, and
  SafeCoder (both splits, overlap excluded). Null: 100 EM label permutations seeded by k.
- The stakes probe vs the Testing vs Conversation probe on stakes_implicit: correlation, and the Testing probe's AUROC
  for the stakes labels.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.2' --with numpy modal run --detach \
      experiments/2026-10-07_stage1_awareness/run_fit_stage1.py --sets <sets_*.jsonl> --vuln-data <dir>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-stage1_fit_kill-2026-11-06"
OUT = "probe-shift_aswin-surya_sub-stage1_acts_kill-2026-11-06"
NEEDHAM = "probe-shift_aswin-surya_sub-q3evalaw_data_kill-2026-11-06"
RUN = "2026-10-07_stage1_qwen35-27b_v1"
CPU, NULLS, CS = 64, 100, (0.001, 0.01, 0.1)
FAMILIES = {"tvc": ("explicit", "implicit", "card"), "monitoring": ("mon_A", "mon_B"), "stakes": ("stakes_A", "stakes_B"),
            "impact": ("impact_bash", "impact_editor", "impact_db")}
IMPLICIT = {"monitoring": "mon_implicit", "stakes": "stakes_implicit"}
AIDER_EVAL, AIDER_DEPLOY = ("aider_full",), ("aider_commit", "aider_multi", "aider_manual")
VULN_LAYERS = tuple(range(0, 64, 4)) + (63,)

app = modal.App(APP)
out_vol = modal.Volume.from_name(OUT, version=2)
needham_vol = modal.Volume.from_name(NEEDHAM)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scikit-learn==1.7.2", "joblib", "threadpoolctl")
         .add_local_python_source("probe_shift"))


def seed_of(*parts) -> int:
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


def load_layer(d: Path, layer: int):
    import numpy as np

    index = [json.loads(x) for x in (d / "index.jsonl").read_text().splitlines()]
    blocks = {c: np.load(d / f"chunk{c:03d}_L{layer}.npy") for c in sorted({r["chunk"] for r in index})}
    return index, np.stack([blocks[r["chunk"]][r["row"]] for r in index]).astype("float32")


def layer_job(layer: int, sets_meta: dict, vuln_meta: dict, needham_meta: dict, parts: tuple = ("main", "vuln")) -> dict:
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from threadpoolctl import threadpool_limits

    root = Path(__import__("os").environ.get("STAGE1_OUT", "/out")) / RUN  # STAGE1_OUT: local smoke tests only

    def fit_lr(X, y, C):
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        m = LogisticRegression(C=C, fit_intercept=False, max_iter=2000).fit((X - mu) / sd, y)
        return m.coef_[0] / sd, float(-(mu / sd) @ m.coef_[0])

    def auc(y, s):
        return float(roc_auc_score(y, s)) if len(set(np.asarray(y).tolist())) == 2 else None

    def choose_c(X, y, groups, n_splits=3):
        best, best_auc = CS[0], -1.0
        for C in CS:
            v = []
            for tr, te in GroupKFold(n_splits=min(n_splits, len(set(groups)))).split(X, y, groups):
                if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                    continue
                w, b = fit_lr(X[tr], y[tr], C)
                v.append(roc_auc_score(y[te], X[te] @ w + b))
            if v and np.mean(v) > best_auc:
                best, best_auc = C, float(np.mean(v))
        return best

    def held_out_families(X, y, fam, task, fams, extra=False):
        """Per family: train on the others (C chosen inside), AUROC on it. extra: also mean difference and LDA."""
        res = {}
        for f in fams:
            tr, te = fam != f, fam == f
            C = choose_c(X[tr], y[tr], task[tr])
            w, b = fit_lr(X[tr], y[tr], C)
            r = {"C": C, "lr": auc(y[te], X[te] @ w + b), "w": (w, b)}
            if extra:
                md = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
                lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[tr], y[tr])
                r |= {"mean_difference": auc(y[te], X[te] @ md), "lda": auc(y[te], lda.decision_function(X[te]))}
            res[f] = r
        return res

    def swapped(y, pair, probe_name, k):
        r = np.random.default_rng(seed_of("swap", probe_name, k))
        up = np.unique(pair)
        flip = dict(zip(up.tolist(), (r.random(len(up)) < 0.5).tolist()))
        return np.where([flip[q] for q in pair.tolist()], 1 - y, y)

    out = {"layer": layer}
    weights, vweights = {}, {}
    with threadpool_limits(1):
        if "main" in parts:
            idx, X = load_layer(root / "sets", layer)
            meta = [sets_meta[r["item"]] for r in idx]
            probe = np.array([m["probe"] for m in meta]); fam = np.array([m["family"] for m in meta])
            y = np.array([m["label"] for m in meta]); task = np.array([m.get("task") or m["pair"] for m in meta])
            pair = np.array([m["pair"] for m in meta]); friendly = np.array([m.get("friendly") for m in meta], dtype=object)
            impact = np.array([m.get("impact") for m in meta], dtype=object)
            for p, fams in FAMILIES.items():
                on = (probe == p) & np.isin(fam, fams)
                Xp, yp, fp, tp, pp = X[on], y[on], fam[on], task[on], pair[on]
                ho = held_out_families(Xp, yp, fp, tp, fams, extra=True)
                res = {"families": {f: {k: v for k, v in r.items() if k != "w"} for f, r in ho.items()}}
                for est in ("lr", "mean_difference", "lda"):
                    v = [r[est] for r in ho.values() if r[est] is not None]
                    res[f"{est}_mean"], res[f"{est}_min"] = float(np.mean(v)), float(np.min(v))
                if p == "tvc":
                    w, b = ho["card"]["w"]
                    te = fp == "card"
                    fr = friendly[on][te]
                    res["friendliness"] = {str(k): auc(yp[te][fr == k], (Xp[te] @ w + b)[fr == k]) for k in (True, False)}
                C_all = choose_c(Xp, yp, tp)
                w, b = fit_lr(Xp, yp, C_all)
                weights[p] = (w, b)
                res["C_all"] = C_all
                im = None
                if p in IMPLICIT:
                    im = (probe == p) & (fam == IMPLICIT[p])
                    s_im = X[im] @ w + b
                    res["implicit"] = {"auroc": auc(y[im], s_im)}
                    if p == "stakes":
                        res["implicit"]["by_impact"] = {k: auc(y[im][impact[im] == k], s_im[impact[im] == k]) for k in ("impactful", "read_only")}
                # null replicates: same swaps at every layer (seeded by probe and k), C choice repeated
                null_mean, null_min, null_im = [], [], []
                for k in range(NULLS):
                    ys = swapped(yp, pp, p, k)
                    hn = held_out_families(Xp, ys, fp, tp, fams)
                    v = [r["lr"] for r in hn.values() if r["lr"] is not None]  # trained and evaluated on the swapped labels
                    null_mean.append(float(np.mean(v))); null_min.append(float(np.min(v)))
                    if im is not None:
                        wn, bn = fit_lr(Xp, ys, choose_c(Xp, ys, tp))
                        null_im.append(auc(y[im], X[im] @ wn + bn))
                res["null_mean"], res["null_min"] = null_mean, null_min
                if im is not None:
                    res["implicit"]["null"] = null_im
                out[p] = res
            im = (probe == "stakes") & (fam == "stakes_implicit")
            st, tv = X[im] @ weights["stakes"][0], X[im] @ weights["tvc"][0]
            out["stakes_vs_tvc_on_stakes_implicit"] = {"pearson": float(np.corrcoef(st, tv)[0, 1]), "tvc_auroc_on_stakes_labels": auc(y[im], tv)}
            tvc_X, tvc_y, tvc_pair, tvc_task = X[probe == "tvc"], y[probe == "tvc"], pair[probe == "tvc"], task[probe == "tvc"]

            # Needham
            nidx, NX = load_layer(root / "needham", layer)
            items = {}
            for r, x in zip(nidx, NX):
                d = items.setdefault(r["item"], {"calls": [], "asst_last": None})
                if r["kind"] == "command_end":
                    d["calls"].append(x)
                else:
                    d["asst_last"] = x
            uids = sorted(u for u in items if u in needham_meta)
            src = np.array([needham_meta[u]["source"] for u in uids]); lab = np.array([int(needham_meta[u]["label"]) for u in uids])
            zero = np.zeros(NX.shape[1], "float32")
            has_calls = np.array([bool(items[u]["calls"]) for u in uids])
            has_al = np.array([items[u]["asst_last"] is not None for u in uids])
            calls_f = np.stack([np.mean(items[u]["calls"], 0) if items[u]["calls"] else zero for u in uids])
            al = np.stack([items[u]["asst_last"] if items[u]["asst_last"] is not None else zero for u in uids])
            mixed_f = np.where(has_calls[:, None], calls_f, al)
            aider_all = np.isin(src, AIDER_EVAL + AIDER_DEPLOY)
            aider = aider_all & has_al
            ya = np.isin(src[aider], AIDER_EVAL).astype(int)
            w, b = weights["tvc"]
            nn = []
            for k in range(NULLS):
                ys = swapped(tvc_y, tvc_pair, "tvc", k)
                wn, bn = fit_lr(tvc_X, ys, choose_c(tvc_X, ys, tvc_task))
                nn.append(auc(ya, al[aider] @ wn + bn))
            out["tvc"]["within_aider"] = {"auroc": auc(ya, al[aider] @ w + b), "n_eval": int(ya.sum()), "n_deploy": int((1 - ya).sum()), "null": nn}

            def twin(feat, keep, name, nulls):
                """Held out by source (5 source-grouped folds, aider never trained on, C chosen inside each fold); statistic =
                mean of per-fold AUROCs. Null: source-level label permutations seeded by k. Within-aider at asst_last."""
                train = np.flatnonzero(keep & ~aider_all)
                counts = {"n_train": int(len(train)), "n_train_by_label": {str(l): int((lab[train] == l).sum()) for l in (0, 1)},
                          "sources_train": int(len(set(src[train])))}
                if counts["sources_train"] < 5 or len(set(lab[train])) < 2:
                    return {"skipped": "fewer than 5 sources or one label", **counts}
                folds = list(GroupKFold(n_splits=5).split(feat[train], lab[train], src[train]))

                def cv(labels):
                    v = []
                    for tr, te in folds:
                        a2, b2 = train[tr], train[te]
                        if len(set(labels[a2])) < 2 or len(set(labels[b2])) < 2:
                            continue
                        wt, bt = fit_lr(feat[a2], labels[a2], choose_c(feat[a2], labels[a2], src[a2]))
                        v.append(roc_auc_score(labels[b2], feat[b2] @ wt + bt))
                    return float(np.mean(v)) if v else None

                Ct = choose_c(feat[train], lab[train], src[train])
                wt, bt = fit_lr(feat[train], lab[train], Ct)
                weights[name] = (wt, bt)
                sources = np.unique(src[train])
                src_label = {s_: int(lab[src == s_][0]) for s_ in sources}
                res = {"C_all": Ct, "held_out_by_source": cv(lab), "within_aider": auc(ya, al[aider] @ wt + bt),
                       "n_train": int(len(train)), "n_train_by_label": {str(l): int((lab[train] == l).sum()) for l in (0, 1)},
                       "sources_train": int(len(sources))}
                if nulls:
                    null_t, null_a = [], []
                    for k in range(NULLS):
                        r = np.random.default_rng(seed_of("twin", name, k))
                        perm = dict(zip(sources.tolist(), r.permutation([src_label[s_] for s_ in sources]).tolist()))
                        ln = np.array([perm.get(s_, lab[i]) for i, s_ in enumerate(src)])
                        null_t.append(cv(ln))
                        wn, bn = fit_lr(feat[train], ln[train], choose_c(feat[train], ln[train], src[train]))
                        null_a.append(auc(ya, al[aider] @ wn + bn))
                    res["null_held_out"], res["null_within_aider"] = null_t, null_a
                return res

            # amendment 1 as decided: primary = tool-call transcripts at rule 1; mixed (fallback) and text-only are secondary
            out["twin"] = twin(calls_f, has_calls, "twin", True)
            out["twin_mixed"] = twin(mixed_f, has_calls | has_al, "twin_mixed", True)
            out["twin_text"] = twin(al, has_al, "twin_text", False)

        if "vuln" in parts and layer in VULN_LAYERS:
            vidx, VX = load_layer(root / "vuln", layer)
            vm = [vuln_meta[r["item"]] for r in vidx]
            vsrc = np.array([m["source"] for m in vm]); vsplit = np.array([str(m.get("split")) for m in vm]); vy = np.array([m["label"] for m in vm])
            em_tr, em_va = (vsrc == "em") & (vsplit == "train"), (vsrc == "em") & (vsplit == "val")
            sc = np.isin(vsrc, ["safecoder_train", "safecoder_val"])
            if not em_tr.any() or not sc.any():
                raise ValueError(f"vuln split empty: em_train {em_tr.sum()}, safecoder {sc.sum()}")
            wv, bv = fit_lr(VX[em_tr], vy[em_tr], 0.1)
            nul_va, nul_sc = [], []
            for k in range(NULLS):
                yn = np.random.default_rng(seed_of("vuln", k)).permutation(vy[em_tr])
                wn, bn = fit_lr(VX[em_tr], yn, 0.1)
                nul_va.append(auc(vy[em_va], VX[em_va] @ wn + bn)); nul_sc.append(auc(vy[sc], VX[sc] @ wn + bn))
            out["vuln"] = {"em_val": auc(vy[em_va], VX[em_va] @ wv + bv), "safecoder": auc(vy[sc], VX[sc] @ wv + bv),
                           "null_em_val": nul_va, "null_safecoder": nul_sc,
                           "n": {"em_train": int(em_tr.sum()), "em_val": int(em_va.sum()), "safecoder": int(sc.sum())}}
            vweights["vuln"] = (wv, bv)
    wdir = root / "probes"
    wdir.mkdir(exist_ok=True)
    for name, ws in (("L{}.npz", weights), ("L{}_vuln.npz", vweights)):  # vuln weights in their own file
        if ws:
            np.savez(wdir / name.format(layer), **{f"{k}__w": v[0].astype("float32") for k, v in ws.items()},
                     **{f"{k}__b": np.float32(v[1]) for k, v in ws.items()})
    return out


@app.function(image=image, cpu=CPU, memory=196608, timeout=4 * 3600, volumes={"/out": out_vol, "/needham": needham_vol.read_only()})
def fit_all(sets_meta: dict, vuln_meta: dict, parts: list) -> dict:
    import time

    from joblib import Parallel, delayed

    t0 = time.time()
    needham_meta = {}
    for x in Path("/needham/inputs/items.jsonl").read_text().splitlines():
        it = json.loads(x)
        needham_meta[it["uid"]] = {"source": it["source"], "label": it["label"], "agentic": it.get("agentic")}
    # the drop table by label, source and reason, reported next to the twin results (amendment 4)
    from collections import Counter

    nd = Path("/out") / RUN / "needham"
    fails = [json.loads(x) for x in (nd / "failures.jsonl").read_text().splitlines()]
    read = {json.loads(x)["item"] for x in (nd / "index.jsonl").read_text().splitlines()}
    drops = Counter((needham_meta[f["item"]]["source"], int(needham_meta[f["item"]]["label"]), f["error"].split(":")[0][:40]) for f in fails)
    kept = Counter((needham_meta[u]["source"], int(needham_meta[u]["label"])) for u in read)
    needham_drops = {"dropped": [{"source": a, "label": b, "reason": c, "n": n} for (a, b, c), n in sorted(drops.items())],
                     "read": [{"source": a, "label": b, "n": n} for (a, b), n in sorted(kept.items())]}
    done = Path("/out") / RUN / ("fit_layers_" + "_".join(parts))
    done.mkdir(parents=True, exist_ok=True)
    res, n_layers = [], 64 if "main" in parts else len(VULN_LAYERS)
    gen = Parallel(n_jobs=CPU, return_as="generator_unordered")(delayed(layer_job)(l, sets_meta, vuln_meta, needham_meta, tuple(parts)) for l in (range(64) if "main" in parts else VULN_LAYERS))
    for r in gen:  # each layer saved and committed as it finishes, so a timeout keeps the finished ones
        (done / f"L{r['layer']}.json").write_text(json.dumps(r))
        out_vol.commit()
        res.append(r)
        print(f"layer {r['layer']} done ({len(res)}/{n_layers}) at {time.time() - t0:.0f} s", flush=True)
    return {"layers": sorted(res, key=lambda r: r["layer"]), "needham_drops": needham_drops, "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def main(sets: str, vuln_data: str, parts: str = "main,vuln"):
    from probe_shift.vuln import load_items

    sets_meta = {}
    for x in Path(sets).read_text().splitlines():
        it = json.loads(x)
        sets_meta[it["id"]] = {k: it.get(k) for k in ("probe", "set", "family", "pair", "label", "implicit", "friendly", "impact", "task")}
    VULN_CFG = {"seed": 0, "em_per_class": 3000, "em_val_fraction": 0.2, "em_group_max_key_share": 8, "max_code_chars": 6000}
    _, _, vuln, vmeta, _ = load_items(vuln_data, VULN_CFG)
    vuln_meta = {vmeta["id"][i]: {"source": vmeta["source"][i], "label": vmeta["label"][i], "split": vmeta["split"][i]}
                 for i in range(len(vuln))}
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO).stdout.strip())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(f"{len(sets_meta)} set items, {len(vuln_meta)} vulnerability items; commit {head[:8]}{' (dirty)' if dirty else ''}", flush=True)
    out = fit_all.remote(sets_meta, vuln_meta, parts.split(","))
    rec = {"stamp": stamp, "git": head, "git_dirty": dirty, "app": APP, "run": RUN, "cpu": CPU, "nulls": NULLS, "C_grid": CS, "parts": parts,
           "sets": {"file": sets, "sha256": hashlib.sha256(Path(sets).read_bytes()).hexdigest()}} | out
    with (HERE / "results" / f"fit_stage1_{stamp}_{parts.replace(',', '_')}.json").open("x") as f:
        f.write(json.dumps(rec))
    print(f"-> results/fit_stage1_{stamp}.json ({out['seconds']} s)")
