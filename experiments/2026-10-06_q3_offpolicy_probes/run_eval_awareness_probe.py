"""Train an eval-awareness probe on Qwen3-32B (Modal) from the contrast pairs in probe_shift.eval_awareness.

1. extract (one H100): Qwen3-32B reads each dialogue (no thinking, no tool definitions, as the LinuxArena reader
   renders), and the residual stream after decoder layers 16..48 (block output, as in the LinuxArena extraction and
   the Apollo probe) is saved for the tokens inside the final tool call's JSON. Saved to the Q3 Volume; reused if there.
2. fit (CPU): Apollo's recipe per layer (per-token logistic regression on standardized activations, no intercept,
   C = 1/10). Train on 80% of core tasks (all their pairs), validate on the rest (dialogue score = mean per-token
   score). Probes: combined (explicit + implicit; saved for LinuxArena), explicit only, implicit only. Transfer AUROC
   (explicit-only probe on implicit pairs and the reverse) asks whether one "test vs real" direction is shared rather
   than a framing-specific one. Label-shuffle null: refits with each training pair's labels swapped with prob. 1/2.
   The primary layer for LinuxArena is the one with the best mean transfer AUROC on held-out tasks, fixed here before
   any LA reading. Reruns need a new cfg["run"]: cached activations are reused only if their fingerprint (dialogues
   and extraction settings) matches, and existing probe weights are never overwritten.

3. cv (CPU, validation only; no weights written): the same fits (layer_job) on the saved activations, with tasks held
   out in FOLDS task-grouped folds instead of one 12-task split. Fold 0 is the original split: `--scheme holdout`
   runs it alone and must reproduce the committed fit result exactly before `--scheme grouped` runs the other folds.
   Metrics are averaged over folds, and the layer rule is applied to the fold means. The LinuxArena probes are still
   the ones fit() saved (trained on the original 48 tasks), so LinuxArena scores do not change.

  PYTHONPATH=src uv run --no-project --python 3.13 --with 'modal>=1.2' --with pyyaml --with numpy \
      modal run experiments/2026-10-06_q3_offpolicy_probes/run_eval_awareness_probe.py::main
  PYTHONPATH=src uv run --no-project --python 3.13 --with 'modal>=1.2' --with pyyaml --with numpy \
      modal run --detach experiments/2026-10-06_q3_offpolicy_probes/run_eval_awareness_probe.py::cv \
      --scheme holdout   # then: --scheme grouped --repro results/eval_awareness_cv_holdout_<stamp>.json
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

from probe_shift.models import QWEN3_32B

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3_eval-awareness_kill-2026-11-06"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"   # shared weights cache: mounted read-only, offline
OUTPUT = "probe-shift_aswin-surya_sub-q3_outputs_kill-2026-11-06"
GPU = "H100!"
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "scikit-learn==1.7.2", "joblib", "threadpoolctl"]
CFG = {
    "run": "eval_awareness_v1",       # Volume paths: acts/eval_awareness_v1.npz, probes/eval_awareness_v1/probe_L{l}.npz
    "model": QWEN3_32B["name"],
    "revision": QWEN3_32B["revision"],  # pinned; the only snapshot on the HF cache Volume, so the 2026-10-06 run used it too
    "dtype": "bfloat16",
    "layers": [16, 24, 32, 40, 48],
    "readout": "block_output",        # residual stream after decoder layer L (0-indexed), as the LA extraction
    "detect": "tokens entirely inside the final tool call's JSON",
    "pairs_seed": 0,
    "split_seed": 0,
    "val_fraction": 0.2,              # of core tasks; all of a task's pairs go to one side
    "reg_coeff": 10,                  # C = 1 / reg_coeff, Apollo's repe.yaml
    "max_iter": 2000,
    "shuffle_null": 10,               # refits per layer for the combined probe
}

app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE).read_only()
out_vol = modal.Volume.from_name(OUTPUT, create_if_missing=True)
out_ro = modal.Volume.from_name(OUTPUT).read_only()  # cv only reads
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP)
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache, "/out": out_vol}, timeout=3600, memory=65536)
def extract(cfg: dict, dialogues: list[dict]) -> dict:
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.eval_awareness import render_call

    path = Path("/out") / "acts" / f"{cfg['run']}.npz"
    stamp = path.with_suffix(".fingerprint")
    if path.exists():  # reuse only activations of exactly these dialogues and extraction settings
        if not stamp.exists() or stamp.read_text() != fingerprint(cfg, dialogues):
            raise FileExistsError(f"{path} was extracted from other inputs or settings; change cfg['run']")
        return {"status": "reused", "path": str(path)}
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=getattr(torch, cfg["dtype"]),
                                                device_map="cuda", attn_implementation="sdpa").model.eval()
    load_s = time.time() - t0
    acts, owner, n_ctx = {l: [] for l in cfg["layers"]}, [], []
    for i, d in enumerate(dialogues):
        ids, det = render_call(tok, d)
        captured = {}

        def grab(l):
            def hook(m, a, o):
                captured[l] = (o[0] if isinstance(o, tuple) else o)[0, det].to(torch.float16).cpu().numpy()
            return hook

        hooks = [base.layers[l].register_forward_hook(grab(l)) for l in cfg["layers"]]
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        for l in cfg["layers"]:
            acts[l].append(captured[l])
        owner += [i] * len(det)
        n_ctx.append(len(ids))
    arrays = {f"L{l}": np.concatenate(v) for l, v in acts.items()}
    nonfinite = int(sum((~np.isfinite(v.astype(np.float32))).sum() for v in arrays.values()))
    if nonfinite:  # never save (or fingerprint) bad activations: a retry would reuse them
        raise ValueError(f"{nonfinite} non-finite activations; nothing saved")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, dialogue=np.array(owner), **arrays)
    stamp.write_text(fingerprint(cfg, dialogues))
    out_vol.commit()
    return {"status": "extracted", "path": str(path), "tokens": len(owner), "nonfinite": nonfinite, "load_seconds": round(load_s),
            "total_seconds": round(time.time() - t0), "context_tokens_mean": float(np.mean(n_ctx)), "device": torch.cuda.get_device_name()}


def primary_layer_metric(r: dict) -> float:
    """Mean cross-family transfer AUROC on held-out tasks only (the primary-layer rule)."""
    return round((r["explicit_only"]["transfer_auroc_implicit_val_tasks"] + r["implicit_only"]["transfer_auroc_explicit_val_tasks"]) / 2, 4)


EXTRACT_KEYS = ("model", "revision", "dtype", "layers", "readout", "detect")
FIT_KEYS = EXTRACT_KEYS + ("split_seed", "val_fraction", "reg_coeff", "max_iter", "shuffle_null")
FOLDS = 5  # cv: 60 core tasks, 12 held out per fold, as in the original 20% split


def holdout_tasks(task, cfg: dict) -> list[int]:
    """The original single split: val_fraction of the core tasks, drawn with split_seed."""
    import numpy as np

    tasks = np.unique(task)
    return sorted(int(t) for t in np.random.default_rng(cfg["split_seed"]).choice(
        tasks, int(round(cfg["val_fraction"] * len(tasks))), replace=False))


def grouped_folds(task, cfg: dict) -> list[list[int]]:
    """FOLDS task-grouped folds; every task is held out once. Fold 0 is the original split (holdout_tasks); the other
    tasks are shuffled with split_seed + 1 and split evenly."""
    import numpy as np

    first = holdout_tasks(task, cfg)
    rest = np.random.default_rng(cfg["split_seed"] + 1).permutation(sorted(set(np.unique(task).tolist()) - set(first)))
    return [first] + [sorted(int(t) for t in g) for g in np.array_split(rest, FOLDS - 1)]


def fold_mean(results: list[dict]) -> dict:
    """layer_job's metrics averaged over folds: `converged` must hold in every fold, the label-shuffle null pools its
    refits (n summed, max over folds), and every other number is the mean over folds."""
    import numpy as np

    out = {}
    for k, v in results[0].items():
        vals = [r[k] for r in results]
        if isinstance(v, dict):
            out[k] = fold_mean(vals)
        elif isinstance(v, bool):
            out[k] = all(vals)
        else:
            out[k] = sum(vals) if k == "n" else max(vals) if k == "max" else round(float(np.mean(vals)), 4)
    return out


def fingerprint(cfg: dict, dialogues: list[dict], keys=EXTRACT_KEYS) -> str:
    """sha256 of the dialogues and the settings a step depends on."""
    import hashlib

    # Not sort_keys: the chat template renders tool arguments in insertion order, so order is part of the input.
    return hashlib.sha256(json.dumps([{k: cfg[k] for k in keys}, dialogues]).encode()).hexdigest()


def layer_job(X, owner, label, family, pair, is_val, cfg) -> tuple[dict, dict]:
    """All fits for one layer. X: [tokens, H] float32; owner: dialogue of each token; the rest per dialogue."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from threadpoolctl import threadpool_limits

    from probe_shift.offpolicy import fit_token_probe, probe_scores

    n = len(label)

    def fit_score(train_dialogues, y):
        rows = train_dialogues[owner]
        with threadpool_limits(3):
            probe, conv = fit_token_probe(X[rows], y[owner[rows]], cfg["reg_coeff"], cfg["max_iter"])
        s = probe_scores(X, probe)
        cnt = np.bincount(owner, minlength=n)
        return probe, conv, np.bincount(owner, weights=s, minlength=n) / np.maximum(cnt, 1)

    def auroc(s, mask):
        return round(float(roc_auc_score(label[mask], s[mask])), 4)

    def pair_acc(s, mask):
        hi = {p: v for p, v, y in zip(pair[mask], s[mask], label[mask]) if y == 1}
        lo = {p: v for p, v, y in zip(pair[mask], s[mask], label[mask]) if y == 0}
        return round(float(np.mean([hi[p] > lo[p] for p in hi])), 4)

    res, combined = {}, None
    for name, fams in (("combined", ("explicit", "implicit")), ("explicit_only", ("explicit",)), ("implicit_only", ("implicit",))):
        probe, conv, s = fit_score(~is_val & np.isin(family, fams), label)
        r = {"converged": conv, "val_auroc_all": auroc(s, is_val),
             **{f"val_auroc_{f}": auroc(s, is_val & (family == f)) for f in ("explicit", "implicit")},
             **{f"val_pair_acc_{f}": pair_acc(s, is_val & (family == f)) for f in ("explicit", "implicit")}}
        if name != "combined":  # transfer to the other family: on held-out tasks, and on all tasks (each call carries
            other = "implicit" if fams == ("explicit",) else "explicit"  # both labels, so calls seen in training do not leak)
            r[f"transfer_auroc_{other}_val_tasks"] = auroc(s, is_val & (family == other))
            r[f"transfer_auroc_{other}_all_tasks"] = auroc(s, family == other)
        res[name] = r
        combined = probe if name == "combined" else combined
    null = []
    for k in range(cfg["shuffle_null"]):  # swap each training pair's labels with probability 1/2
        flip = np.random.default_rng(1000 + k).random(pair.max() + 1) < 0.5
        _, _, s = fit_score(~is_val, np.where(flip[pair], 1 - label, label))
        null.append(auroc(s, is_val))
    res["combined"]["label_shuffle_null_val_auroc"] = {"n": len(null), "mean": round(float(np.mean(null)), 4),
                                                       "max": round(float(np.max(null)), 4)}
    return res, combined


@app.function(image=image, volumes={"/out": out_vol}, cpu=16, memory=65536, timeout=3600)
def fit(cfg: dict, dialogues: list[dict]) -> dict:
    import numpy as np
    from joblib import Parallel, delayed

    out_vol.reload()
    pdir = Path("/out") / "probes" / cfg["run"]
    if pdir.exists() and any(pdir.iterdir()):  # never refit over weights that scores and results point at
        summary = pdir / "fit.json"  # written with the weights: a retry after a lost client gets the same result back
        if summary.exists() and json.loads(summary.read_text())["fingerprint"] == fingerprint(cfg, dialogues, FIT_KEYS):
            return json.loads(summary.read_text())["result"]
        raise FileExistsError(f"{pdir} already holds probes from other inputs or settings; change cfg['run']")
    z = np.load(Path("/out") / "acts" / f"{cfg['run']}.npz")
    bad = {k: int((~np.isfinite(z[k].astype(np.float32))).sum()) for k in z.files if k.startswith("L")}
    if any(bad.values()):  # also covers activations reused from an earlier extraction
        raise ValueError(f"non-finite activations: {bad}")
    owner = z["dialogue"]
    label = np.array([d["label"] for d in dialogues])
    family = np.array([d["family"] for d in dialogues])
    task = np.array([d["task"] for d in dialogues])
    pair = np.array([d["pair"] for d in dialogues])
    val_tasks = holdout_tasks(task, cfg)
    is_val = np.isin(task, val_tasks)
    outs = Parallel(n_jobs=len(cfg["layers"]))(
        delayed(layer_job)(z[f"L{l}"].astype(np.float32), owner, label, family, pair, is_val, cfg) for l in cfg["layers"])
    pdir.mkdir(parents=True, exist_ok=True)
    by_layer = {}
    for l, (res, probe) in zip(cfg["layers"], outs):
        np.savez(pdir / f"probe_L{l}.npz", **probe)
        by_layer[str(l)] = res
    transfer = {l: primary_layer_metric(r) for l, r in by_layer.items()}
    result = {"by_layer": by_layer, "mean_transfer_auroc": transfer, "primary_layer": int(max(transfer, key=transfer.get)),
              "val_tasks": val_tasks, "n_dialogues": len(dialogues), "n_tokens": int(len(owner)),
              "probe_dir": f"{OUTPUT}/probes/{cfg['run']}"}
    (pdir / "fit.json").write_text(json.dumps({"fingerprint": fingerprint(cfg, dialogues, FIT_KEYS), "config": cfg, "result": result}))
    out_vol.commit()
    return result


@app.function(image=image, volumes={"/out": out_ro}, cpu=16, memory=65536, timeout=3600)
def cv_fold(cfg: dict, dialogues: list[dict], val_tasks: list[int], compare_saved: bool, n_tokens: int) -> dict:
    """fit()'s fits (layer_job) with val_tasks held out, on the saved activations; writes nothing. compare_saved: also
    the largest difference between this fold's combined probe and the saved LinuxArena probe (the same split).
    eval_awareness_v1 predates the activation fingerprint, so the file is checked against the committed fit's token
    count and dialogue count instead, and its sha256 is recorded."""
    import hashlib
    import time

    import numpy as np
    from joblib import Parallel, delayed

    t0 = time.time()
    path = Path("/out") / "acts" / f"{cfg['run']}.npz"
    z = np.load(path)
    owner = z["dialogue"]
    if len(owner) != n_tokens or int(owner.max()) + 1 != len(dialogues):
        raise ValueError(f"{path}: {len(owner)} tokens over {int(owner.max()) + 1} dialogues; expected {n_tokens} over {len(dialogues)}")
    label, family, task, pair = (np.array([d[k] for d in dialogues]) for k in ("label", "family", "task", "pair"))
    is_val = np.isin(task, val_tasks)
    outs = Parallel(n_jobs=len(cfg["layers"]))(
        delayed(layer_job)(z[f"L{l}"].astype(np.float32), owner, label, family, pair, is_val, cfg) for l in cfg["layers"])
    out = {"val_tasks": sorted(val_tasks), "acts_file": f"{OUTPUT}/acts/{cfg['run']}.npz",
           "acts_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n_val_dialogues": int(is_val.sum()),
           "n_val_tokens": int(is_val[owner].sum()), "n_tokens": int(len(owner)),
           "by_layer": {str(l): res for l, (res, _) in zip(cfg["layers"], outs)}}
    if compare_saved:
        saved = {str(l): np.load(Path("/out") / "probes" / cfg["run"] / f"probe_L{l}.npz") for l in cfg["layers"]}
        out["max_abs_diff_vs_saved_probe"] = {l: {k: float(np.abs(probe[k] - saved[str(l)][k]).max()) for k in saved[str(l)].files}
                                              for l, (_, probe) in zip(cfg["layers"], outs)}
    return out | {"seconds": round(time.time() - t0)}


@app.local_entrypoint()
def main():
    from collections import Counter

    from probe_shift.eval_awareness import TASKS, pairs

    dialogues = pairs(CFG["pairs_seed"])
    print(f"{len(TASKS)} core tasks, {len(dialogues) // 2} pairs:", dict(Counter(d["family"] for d in dialogues[::2])), flush=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / f"eval_awareness_pairs_{stamp}.jsonl", "w") as f:  # the exact training data
        f.writelines(json.dumps(d) + "\n" for d in dialogues)
    ext = extract.remote(CFG, dialogues)
    print("extract:", ext, flush=True)
    if ext.get("nonfinite"):
        raise ValueError(f"{ext['nonfinite']} non-finite activations")
    res = fit.remote(CFG, dialogues)
    out = {"extract": ext, "config": CFG | {"modal": {"app": APP, "gpu": GPU, "pip": PIP, "hf_cache": HF_CACHE, "output_volume": OUTPUT}},
           "pairs_file": f"results/eval_awareness_pairs_{stamp}.jsonl"} | res
    path = HERE / "results" / f"eval_awareness_probe_{stamp}.json"
    path.write_text(json.dumps(out, indent=1))
    for l, r in res["by_layer"].items():
        c, e, i = r["combined"], r["explicit_only"], r["implicit_only"]
        print(f"L{l}: combined val AUROC explicit {c['val_auroc_explicit']:.3f} implicit {c['val_auroc_implicit']:.3f} "
              f"(shuffle null mean {c['label_shuffle_null_val_auroc']['mean']:.3f} max {c['label_shuffle_null_val_auroc']['max']:.3f}) | "
              f"transfer explicit->implicit {e['transfer_auroc_implicit_all_tasks']:.3f} implicit->explicit {i['transfer_auroc_explicit_all_tasks']:.3f}")
    print(f"primary layer (best mean held-out transfer): {res['primary_layer']}\n-> {path}")


def differences(a, b, path=""):
    """Every leaf where two nested results differ, as (path, a, b)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(a.keys() | b.keys()) for d in differences(a.get(k), b.get(k), f"{path}/{k}")]
    return [] if a == b else [(path, a, b)]


@app.local_entrypoint()
def cv(scheme: str = "holdout", repro: str = "", committed: str = "results/eval_awareness_probe_2026-10-06_143206.json",
       record: str = "results/eval_awareness_probe_2026-10-06_143206_record.json"):
    """Validation only (module docstring, step 3). holdout: fold 0 alone, which must equal `committed` in every metric
    and give `record`'s layer under the held-out rule. grouped: folds 1.. on top of fold 0 from `repro`, the holdout
    result, which must have reproduced with the same code. Writes results/eval_awareness_cv_<scheme>_<stamp>.json."""
    import numpy as np

    from probe_shift.eval_awareness import pairs
    from probe_shift.offpolicy import provenance

    prov = provenance(__file__, HERE.parent.parent / "src/probe_shift/eval_awareness.py", HERE.parent.parent / "src/probe_shift/offpolicy.py")
    if prov["git_dirty"]:
        raise RuntimeError("commit first: the result records the commit")
    old, rec = json.loads((HERE / committed).read_text()), json.loads((HERE / record).read_text())
    dialogues = pairs(CFG["pairs_seed"])
    if dialogues != [json.loads(line) for line in (HERE / old["pairs_file"]).read_text().splitlines()]:
        raise ValueError(f"pairs({CFG['pairs_seed']}) differs from {old['pairs_file']}, the data of the committed fit")
    if any(old["config"].get(k, rec["model"]["revision"]) != CFG[k] for k in FIT_KEYS):  # the run predates the revision key
        raise ValueError("CFG differs from the committed fit's settings")
    folds = grouped_folds(np.array([d["task"] for d in dialogues]), CFG)
    if folds[0] != old["val_tasks"]:
        raise ValueError(f"fold 0 {folds[0]} is not the committed split {old['val_tasks']}")
    rule = lambda by_layer: {l: primary_layer_metric(r) for l, r in by_layer.items()}
    pick = lambda m: int(max(m, key=m.get))
    out = {"scheme": scheme, "provenance": prov, "committed": committed, "record": record, "folds": folds,
           "config": CFG | {"folds": FOLDS, "modal": {"app": APP, "pip": PIP, "output_volume": OUTPUT}}}
    if scheme == "holdout":
        r = cv_fold.remote(CFG, dialogues, folds[0], True, old["n_tokens"])
        diffs = differences(old["by_layer"], r["by_layer"])
        metric = rule(r["by_layer"])
        out |= {"fold0": r, "mean_heldout_transfer_auroc": metric, "primary_layer": pick(metric),
                "differences_from_committed": diffs,
                "reproduces": not diffs and metric == rec["mean_heldout_transfer_auroc"] and pick(metric) == rec["primary_layer"]}
    elif scheme == "grouped":
        rep = json.loads((HERE / repro).read_text())
        if not rep["reproduces"] or rep["folds"] != folds or rep["provenance"]["source_sha256"] != prov["source_sha256"]:
            raise ValueError(f"{repro} did not reproduce the committed fit, or used other folds or code")
        per_fold = [rep["fold0"]] + list(cv_fold.starmap([(CFG, dialogues, f, False, old["n_tokens"]) for f in folds[1:]]))
        if len({f["acts_sha256"] for f in per_fold}) != 1:
            raise ValueError("folds read different activations")
        by_layer = {l: fold_mean([f["by_layer"][l] for f in per_fold]) for l in per_fold[0]["by_layer"]}
        metric = rule(by_layer)
        out |= {"repro": repro, "per_fold": per_fold, "by_layer": by_layer, "mean_heldout_transfer_auroc": metric,
                "primary_layer": pick(metric), "primary_layer_per_fold": [pick(rule(f["by_layer"])) for f in per_fold]}
    else:
        raise ValueError(f"unknown scheme {scheme}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"eval_awareness_cv_{scheme}_{stamp}.json"
    with path.open("x") as f:
        f.write(json.dumps(out, indent=1))
    by_layer = out["fold0"]["by_layer"] if scheme == "holdout" else out["by_layer"]
    for l, r in by_layer.items():
        o, c = old["by_layer"][l]["combined"], r["combined"]
        print(f"L{l}: held-out transfer {rec['mean_heldout_transfer_auroc'][l]:.4f} -> {out['mean_heldout_transfer_auroc'][l]:.4f} | "
              f"combined val AUROC explicit {o['val_auroc_explicit']:.4f} -> {c['val_auroc_explicit']:.4f}, "
              f"implicit {o['val_auroc_implicit']:.4f} -> {c['val_auroc_implicit']:.4f} | "
              f"null mean {o['label_shuffle_null_val_auroc']['mean']:.4f} -> {c['label_shuffle_null_val_auroc']['mean']:.4f}")
    print(f"primary layer {rec['primary_layer']} -> {out['primary_layer']}" + (f"; reproduces: {out['reproduces']}" if scheme == "holdout" else
          f"; per fold {out['primary_layer_per_fold']}") + f"\n-> {path}")
    if scheme == "holdout" and not out["reproduces"]:
        raise RuntimeError(f"fold 0 does not reproduce the committed fit: {diffs[:10]}")
