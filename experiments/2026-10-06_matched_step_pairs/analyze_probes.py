"""Probe scores for the matched pairs, on Modal, at all 64 layers and both readouts (output_end of step k; call_end of
the next call): the Q1-recipe probe frozen on the cohorts, a fresh probe fit within the pairs, and the saved Q2
impactful-call probes. Writes raw scores; analyze_pairs.py turns them into metrics and curves.

1. Q1 recipe, frozen (the main session saved no probe weights, so this refits its recipe and saves the weights): per
   readout and layer, real attack epoch 1 (0) vs simulated attack (1), every output_end (or call_end) row of the 939
   paired cells inside the 32k cut, from the main session's all-layers dump (read only); standardized features, L2
   logistic regression C = 0.01 (scikit-learn's objective, torch L-BFGS, max_iter 2000); one probe per held-out
   main-task fold (folds_v3). A pair is scored by the probe whose held-out fold holds its main task, so no probe saw the
   pair's task. Weights go to the pairs Volume.
   - Label-shuffle null at every layer: the same fits with training labels shuffled as Q1 does (per main task, real
     and sim swap with probability 1/2), N_NULL_FROZEN draws, batched into the same fit.
   - Layer choice without the pairs: separate fold fits on the cohort with every main task used in the pairs left out
     of training and scoring; their out-of-fold AUROC per layer picks the layer (analyze_pairs.py).
   - Our real-version activations are checked against the dump's row for the same transcript and position.
2. Fresh, within the pairs: per layer and position, leave-one-environment-out over the pairs' 8 environments, same
   probe; a label-shuffle null at every layer (N_NULL draws: within each pair, real and sim swap labels with probability
   1/2); and, for the two readouts, nested inner CV: for each held-out environment, inner leave-one-environment-out over
   the other 7 gives an AUROC per layer, from which analyze_pairs.py picks that environment's layer.
3. Q2 impactful-call probes as saved by the Q2 session (one per outer fold, layer chosen there): prev_content on our
   output_end of step k (the last content token before the next call), pre_call and call_end on the next call.

  PYTHONPATH=src python -m modal run experiments/2026-10-06_matched_step_pairs/analyze_probes.py --run <read run> \\
    --pairs data/pairs_<stamp>.jsonl [--test]
"""

import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE  # the container mounts this file at /root
APP = "probe-shift_aswin-surya_sub-matched_probes_kill-2026-11-06"
PAIRS_VOL = "probe-shift_aswin-surya_sub-matched_pairs_kill-2026-11-06"
ACTS_VOL = "probe-shift_aswin-surya_sub-la-activations"  # the main session's; read only
Q2_VOL = "probe-shift_aswin-surya_sub-q2_probe-features_kill-2026-11-06"  # the Q2 session's; read only
DUMP = "2026-10-06_la_segment-end_qwen3-32b_alllayers_v1"
Q2_RUN = "2026-10-06_q2_features_846b53c5c1bcd8a7_curl_read_only"  # features run of the Q2 headline (q2_probes_2026-10-06_154527)
KINDS = ["output_end", "output_marker", "pre_call", "call_end", "call_marker"]
READOUTS = {"output_end": "turn", "call_end": "next_turn"}  # readout -> which turn of the pair it sits in (action 0)
C, MAX_ITER, N_NULL, N_NULL_FROZEN = 0.01, 2000, 100, 20
PT = ZoneInfo("America/Los_Angeles")

app = modal.App(APP)
pairs_vol = modal.Volume.from_name(PAIRS_VOL, version=2)
acts_vol = modal.Volume.from_name(ACTS_VOL, version=2)
q2_vol = modal.Volume.from_name(Q2_VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scikit-learn==1.7.2")


def fit(X, Y, C: float, max_iter: int):
    """L2 logistic regression with scikit-learn's objective for every label column at once (probe_shift.torch_probe on
    the main session's branch): minimize C * sum logloss + 0.5 ||w||^2, intercept unpenalized. X standardized."""
    import torch
    import torch.nn.functional as F

    Y = Y.reshape(len(Y), -1).to(X.dtype)
    W = torch.zeros(X.shape[1], Y.shape[1], device=X.device, dtype=X.dtype, requires_grad=True)
    b = torch.zeros(Y.shape[1], device=X.device, dtype=X.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, tolerance_grad=1e-5, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (C * F.binary_cross_entropy_with_logits(X @ W + b, Y, reduction="sum") + 0.5 * (W * W).sum()) / len(X)
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def standardize(train, *others):
    mean, sd = train.mean(0), train.std(0, unbiased=False)
    sd = sd.where(sd > 0, sd.new_ones(()))
    return mean, sd, [(x - mean) / sd for x in (train, *others)]


def pair_acts(run: str):
    """[items, kind, layer, hidden] fp16 and the item list, from the read run on the pairs Volume."""
    import numpy as np

    d = Path("/pairs") / run / "acts"
    metas = sorted((json.loads(p.read_text()) for p in d.glob("*.json")), key=lambda m: (m["pair"], m["version"]))
    versions = {}
    for m in metas:
        versions.setdefault(m["pair"], set()).add(m["version"])
    complete = {p for p, v in versions.items() if v == {"real", "sim"}}
    if len(complete) < len(versions):  # a pair the reader read only one version of stays out of every fit
        print(f"dropping {len(versions) - len(complete)} incomplete pairs: {sorted(set(versions) - complete)}", flush=True)
    metas = [m for m in metas if m["pair"] in complete]
    assert all(m["kinds"] == KINDS for m in metas)
    X = np.stack([np.load(d / f"{m['pair']:03d}_{m['version']}.npz")["acts"] for m in metas])
    bad = [(m["pair"], m["version"]) for m, x in zip(metas, X) if not np.isfinite(x).all()]
    assert not bad, f"non-finite activations in {bad}"
    return X, metas


def within_pair_flips(pair, y0, n: int, seed: int):
    """n label columns: per pair, real and sim swap with probability 1/2 (pairs in sorted order, one draw each)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    up = np.unique(pair)
    cols = []
    for _ in range(n):
        flip = dict(zip(up, rng.random(len(up)) < 0.5))
        cols.append(np.where([flip[p] for p in pair], 1 - y0, y0))
    return cols


@app.function(image=image, volumes={"/acts": acts_vol, "/pairs": pairs_vol}, cpu=8, memory=32768, timeout=3600)
def index_dump(pairs: list[dict]) -> dict:
    """Where each training row and each pair's own row live in the dump, per readout. Training rows:
    {readout: {chunk: [[row, label, fold, main index], ...]}} for real-attack-e1 and sim-attack in the paired cells.
    Own rows: {readout: {pair: [chunk, row, token]}} of the pair's real transcript at that readout."""
    folds = pairs[0]["_folds"]
    paired = set(pairs[0]["_paired_cells"])
    mains = sorted(folds)
    main_idx = {m: i for i, m in enumerate(mains)}
    run = Path("/acts") / DUMP
    train = {r: {} for r in READOUTS}
    own = {r: {} for r in READOUTS}
    wanted = {(p["split_arm"], p["key"]): p for p in pairs}
    n_meta = 0
    for arm in ("real-attack-e1", "sim-attack", "real-honest"):
        for f in sorted((run / "meta" / arm).glob("*.json")):
            m = json.loads(f.read_text())
            n_meta += 1
            if arm != "real-honest" and m["cell"] in paired:
                for p in m["positions"]:
                    if p["kind"] in READOUTS and p["kept"]:
                        train[p["kind"]].setdefault(m["chunk"], []).append(
                            [m["row_start"] + p["row"], int(arm == "sim-attack"), folds[m["main_task"]], main_idx[m["main_task"]]])
            if (arm, m["key"]) in wanted:
                pr = wanted[(arm, m["key"])]
                for readout, which in READOUTS.items():
                    hit = [p for p in m["positions"] if p["kind"] == readout and p["step"] == pr[which] and p["action"] == 0]
                    assert len(hit) == 1, (m["key"], readout, pr[which])
                    if hit[0]["kept"]:
                        own[readout][str(pr["pair"])] = [m["chunk"], m["row_start"] + hit[0]["row"], hit[0]["token"]]
    assert len(own["output_end"]) == len(pairs), f"found the dump rows of {len(own['output_end'])} of {len(pairs)} pairs"
    # Every paired cell must be in the dump in both arms, or the frozen fit would mix in unpaired transcripts.
    arms_by_cell = {}
    for arm in ("real-attack-e1", "sim-attack"):
        for f in (run / "meta" / arm).glob("*.json"):
            cell = json.loads(f.read_text())["cell"]
            if cell in paired:
                arms_by_cell.setdefault(cell, []).append(arm)
    unpaired = sorted(c for c in paired if sorted(arms_by_cell.get(c, [])) != ["real-attack-e1", "sim-attack"])
    assert not unpaired, f"{len(unpaired)} paired cells lack exactly one transcript per arm in the dump: {unpaired[:5]}"
    return {"train": train, "own": own, "n_meta": n_meta, "mains": mains}


@app.function(image=image, gpu="A10G", volumes={"/acts": acts_vol, "/pairs": pairs_vol}, memory=65536, timeout=3 * 3600,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def frozen(run: str, out_name: str, layers: list[int], index: dict, pair_folds: dict, pair_mains: list[str],
           seed: int = 0) -> dict:
    """Per layer and readout: Q1-recipe probes per held-out fold with N_NULL_FROZEN training-label shuffles in the same
    batched fit; scores for every pair item (all columns); cohort out-of-fold AUROC; dump agreement of real items."""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    pairs_vol.reload()
    X_pairs, metas = pair_acts(run)
    item_fold = np.array([pair_folds[str(m["pair"])] for m in metas])
    skip_main = {index["mains"].index(m) for m in pair_mains}
    out = {"items": [[m["pair"], m["version"]] for m in metas], "layers": {}}
    wdir = Path("/pairs") / out_name / "frozen_q1_weights"
    wdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_mains = len(index["mains"])
    flips = rng.random((N_NULL_FROZEN, n_mains)) < 0.5  # the same draws at every layer and readout
    for layer in layers:
        t0 = time.time()
        res = {}
        for readout in READOUTS:
            rows, meta = [], []
            for chunk, entries in sorted(index["train"][readout].items()):
                arr = np.load(Path("/acts") / DUMP / "by_layer" / f"L{layer}" / f"{chunk}.npy", mmap_mode="r")
                e = np.array(entries)
                e = e[np.argsort(e[:, 0])]
                block = np.asarray(arr[e[:, 0]])
                bad = ~np.isfinite(block).all(1)
                assert not bad.any(), f"non-finite dump rows in L{layer}/{chunk}: rows {e[bad, 0][:5].tolist()}"
                rows.append(block)
                meta.append(e)
            e = np.concatenate(meta)
            X = torch.from_numpy(np.concatenate(rows)).cuda().float()
            y, fold, main = e[:, 1], e[:, 2], e[:, 3]
            Y = np.stack([y] + [np.where(flips[d][main], 1 - y, y) for d in range(N_NULL_FROZEN)], 1)
            Y = torch.from_numpy(Y).cuda()
            k = KINDS.index(readout)
            P = torch.from_numpy(X_pairs[:, k, layer]).cuda().float()
            scores = np.full((len(metas), Y.shape[1]), np.nan)
            oof = np.full(len(y), np.nan)
            for f in sorted(set(fold.tolist())):
                tr = torch.from_numpy(fold != f).cuda()
                mean, sd, (Xtr,) = standardize(X[tr])
                W, b = fit(Xtr, Y[tr], C, MAX_ITER)
                sel = item_fold == f
                if sel.any():
                    scores[sel] = (((P[torch.from_numpy(sel).cuda()] - mean) / sd) @ W + b).cpu().numpy()
                np.savez(wdir / f"{readout}_L{layer:02d}_fold{f}.npz", coef=W[:, 0].cpu().numpy(), intercept=b[:1].cpu().numpy(),
                         mean=mean.cpu().numpy(), sd=sd.cpu().numpy(), n_train=int(tr.sum()), C=C)
            # Layer choice without the pairs: separate fits on the cohort with every pair main task left out of both
            # training and scoring, so no pair's transcript influences which layer is picked.
            keep = ~np.isin(main, list(skip_main))
            for f in sorted(set(fold.tolist())):
                tr = torch.from_numpy(keep & (fold != f)).cuda()
                te_np = keep & (fold == f)
                _, _, (Xsel, Xval) = standardize(X[tr], X[torch.from_numpy(te_np).cuda()])
                Wsel, bsel = fit(Xsel, Y[tr][:, :1], C, MAX_ITER)
                oof[te_np] = (Xval @ Wsel + bsel)[:, 0].cpu().numpy()
            agree = []
            for i, m in enumerate(metas):
                if m["version"] != "real" or str(m["pair"]) not in index["own"][readout]:
                    continue
                chunk, row, token = index["own"][readout][str(m["pair"])]
                ref = np.load(Path("/acts") / DUMP / "by_layer" / f"L{layer}" / f"{chunk}.npy", mmap_mode="r")[row].astype(np.float32)
                ours = X_pairs[i, k, layer].astype(np.float32)
                cos = float(ours @ ref / (np.linalg.norm(ours) * np.linalg.norm(ref) + 1e-9))
                agree.append([m["pair"], token == m["tokens"][k], cos])
            assert len(agree) == sum(m["version"] == "real" for m in metas), "dump agreement missed real items"
            res[readout] = {"scores": scores[:, 0].tolist(), "null_scores": scores[:, 1:].T.round(4).tolist(),
                            "cohort_oof_auroc": float(roc_auc_score(y[keep], oof[keep])), "cohort_rows_scored": int(keep.sum()),
                            "n_train": int(len(y)), "agreement": agree}
        out["layers"][layer] = res
        print(f"layer {layer}: {time.time() - t0:.0f}s, cohort OOF AUROC "
              f"{res['output_end']['cohort_oof_auroc']:.3f} / {res['call_end']['cohort_oof_auroc']:.3f}", flush=True)
    pairs_vol.commit()
    return out


@app.function(image=image, gpu="A10G", volumes={"/pairs": pairs_vol}, memory=65536, timeout=3 * 3600)
def fresh(run: str, kind: str, pair_env: dict, n_null: int, seed: int = 0) -> dict:
    """Leave-one-environment-out probes within the pairs at one position, every layer: the real labels and n_null
    within-pair label swaps, one batched fit per held-out environment."""
    import numpy as np
    import torch

    pairs_vol.reload()
    X, metas = pair_acts(run)
    pair = np.array([m["pair"] for m in metas])
    y0 = np.array([float(m["version"] == "sim") for m in metas])
    env = np.array([pair_env[str(p)] for p in pair])
    Y = torch.from_numpy(np.stack([y0] + within_pair_flips(pair, y0, n_null, seed), 1)).cuda()
    k = KINDS.index(kind)
    out = {"items": [[m["pair"], m["version"]] for m in metas], "kind": kind, "layers": {}}
    for layer in range(X.shape[2]):
        A = torch.from_numpy(X[:, k, layer]).cuda().float()
        S = np.full((len(metas), Y.shape[1]), np.nan)
        for e in np.unique(env):
            tr = torch.from_numpy(env != e).cuda()
            _, _, (Atr, Ate) = standardize(A[tr], A[~tr])
            W, b = fit(Atr, Y[tr], C, MAX_ITER)
            S[env == e] = (Ate @ W + b).cpu().numpy()
        out["layers"][layer] = {"scores": S[:, 0].tolist(), "null_scores": S[:, 1:].T.round(4).tolist()}
    print(f"fresh {kind} done", flush=True)
    return out


@app.function(image=image, gpu="A10G", volumes={"/pairs": pairs_vol}, memory=65536, timeout=3 * 3600)
def fresh_inner(run: str, kind: str, pair_env: dict, outer_env: str) -> dict:
    """Inner CV for one held-out environment: leave-one-environment-out over the other environments, AUROC per layer
    (real labels), so the outer environment's layer is chosen without its own data."""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    pairs_vol.reload()
    X, metas = pair_acts(run)
    pair = np.array([m["pair"] for m in metas])
    y0 = np.array([float(m["version"] == "sim") for m in metas])
    env = np.array([pair_env[str(p)] for p in pair])
    inside = env != outer_env
    k = KINDS.index(kind)
    Y = torch.from_numpy(y0).cuda()
    auroc = {}
    for layer in range(X.shape[2]):
        A = torch.from_numpy(X[:, k, layer]).cuda().float()
        s = np.full(len(metas), np.nan)
        for e in np.unique(env[inside]):
            tr_np, te_np = inside & (env != e), env == e
            tr, te = torch.from_numpy(tr_np).cuda(), torch.from_numpy(te_np).cuda()
            _, _, (Atr, Ate) = standardize(A[tr], A[te])
            W, b = fit(Atr, Y[tr], C, MAX_ITER)
            s[te_np] = (Ate @ W + b)[:, 0].cpu().numpy()
        auroc[layer] = float(roc_auc_score(y0[inside], s[inside]))
    return {"kind": kind, "outer_env": outer_env, "inner_auroc": auroc}


@app.function(image=image, volumes={"/pairs": pairs_vol, "/q2": q2_vol}, cpu=4, memory=32768, timeout=1800)
def q2_scores(run: str, pair_folds: dict) -> dict:
    """The saved Q2 probes on the pairs: the probe for outer fold f scores the pairs whose main task is in fold f (out
    of fold, as in Q2), at the layer Q2 chose for that fold."""
    import numpy as np

    pairs_vol.reload()
    X, metas = pair_acts(run)
    pair = np.array([m["pair"] for m in metas])
    q2 = {}
    item_fold = np.array([pair_folds[str(p)] for p in pair])
    for q2pos, kind in (("prev_content", "output_end"), ("pre_call", "pre_call"), ("call_end", "call_end")):
        for train_set in ("real-attack-e1", "sim-attack", "real-attack-e1+real-honest"):
            z = np.load(Path("/q2") / Q2_RUN / "fits" / f"{q2pos}__{train_set}__real_probes.npz")
            s = np.full(len(metas), np.nan)
            for f in range(len(z["fold"])):
                sel = item_fold == int(z["fold"][f])
                a = X[sel, KINDS.index(kind), int(z["layer"][f])].astype(np.float32)
                s[sel] = ((a - z["mean"][f]) / z["sd"][f]) @ z["coef"][f] + z["intercept"][f]
            q2[f"{q2pos}|{train_set}"] = {"scores": s.tolist(), "layer": z["layer"].tolist(),
                                          "threshold": z["threshold"].tolist(), "item_fold": item_fold.tolist()}
    d = Path("/pairs") / run / "acts"
    lp = [np.load(d / f"{m['pair']:03d}_{m['version']}.npz")["logprobs"] for m in metas]
    return {"items": [[m["pair"], m["version"]] for m in metas], "q2": q2,
            "logprob_mean": [float(x.mean()) if len(x) else float("nan") for x in lp], "logprob_n": [len(x) for x in lp]}


@app.local_entrypoint()
def main(run: str, pairs: str, test: bool = False):
    import hashlib

    pairs_path = Path(pairs)
    recs = [json.loads(l) for l in pairs_path.read_text().splitlines()]
    # Pair numbers restart at 0 in every selection: the read run must come from this very pairs file.
    pairs_sha = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    read_cfg = json.loads(b"".join(pairs_vol.read_file(f"{run}/config.json")))
    assert read_cfg["pairs_sha256"] == pairs_sha, f"read run {run} was made from another pairs file"
    shared = json.loads((ROOT / "splits" / "exclusions_v3.json").read_text())
    folds = json.loads((ROOT / "splits" / "folds_v3.json").read_text())["task_folds"]["main_task_to_fold"]
    slim = [{k: r[k] for k in ("pair", "split_arm", "key", "turn", "next_turn")} for r in recs]
    slim[0] |= {"_folds": folds, "_paired_cells": shared["paired_attack_cells"]["cells"]}
    pair_folds = {str(r["pair"]): r["fold"] for r in recs}
    pair_env = {str(r["pair"]): r["environment"] for r in recs}
    pair_mains = sorted({r["main_task"] for r in recs})
    envs = sorted(set(pair_env.values()))
    stamp = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
    out_name = f"{run}__probes_{stamp}"
    t0 = time.time()
    kinds = ["output_end"] if test else KINDS
    fr = [fresh.spawn(run, kind, pair_env, N_NULL) for kind in kinds]
    inner = [fresh_inner.spawn(run, kind, pair_env, e) for kind in (["output_end"] if test else READOUTS) for e in envs]
    q2 = q2_scores.spawn(run, pair_folds)
    index = index_dump.remote(slim)
    print(f"dump index: {sum(len(v) for v in index['train']['output_end'].values())} output_end and "
          f"{sum(len(v) for v in index['train']['call_end'].values())} call_end training rows; "
          f"{len(index['own']['output_end'])}/{len(index['own']['call_end'])} pair rows", flush=True)
    layers = [40, 63] if test else list(range(64))
    groups = [layers[i::16] for i in range(min(16, len(layers)))]
    parts = list(frozen.starmap([(run, out_name, g, index, pair_folds, pair_mains) for g in groups]))
    # The frozen workers each commit their own weight files; make sure every (readout, layer, fold) file landed.
    expected = len(READOUTS) * len(layers) * len(set(folds.values()))
    landed = [e.path for e in pairs_vol.listdir(f"{out_name}/frozen_q1_weights")]
    assert len(landed) == expected, f"{len(landed)} of {expected} frozen weight files on the Volume"
    res = {"run": run, "pairs_file": pairs_path.name, "pairs_sha256": pairs_sha, "dump": DUMP, "q2_run": Q2_RUN, "C": C, "max_iter": MAX_ITER,
           "n_null_fresh": N_NULL, "n_null_frozen": N_NULL_FROZEN, "kinds": KINDS, "readouts": list(READOUTS),
           "weights": f"{PAIRS_VOL}/{out_name}/frozen_q1_weights",
           "frozen": {"items": parts[0]["items"], "layers": {str(l): v for p in parts for l, v in p["layers"].items()}},
           "fresh": {r["kind"]: r for r in (c.get() for c in fr)},
           "fresh_inner": [c.get() for c in inner], **q2.get(), "wall_seconds": round(time.time() - t0)}
    # 100+ MB of per-item scores (nulls at every layer): git-ignored data/ and the pairs Volume, not results/.
    path = HERE / "data" / "probe_scores" / f"probe_scores_{stamp}{'_test' if test else ''}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        f.write(json.dumps(res))
    print(f"wrote {path.relative_to(ROOT)} in {res['wall_seconds']} s", flush=True)
