"""PLAN.md section 2: a sabotage probe trained on the public white-box-control datasets, frozen before any of our data is
scored. Three Modal functions, run in order; no stage-1 label and no stage-2 task or sealed run is touched here.

  stage    CPU: downloads the 12 datasets at pinned revisions INSIDE the container (never on the laptop), records each
           parquet's sha256, renders every row's [user, assistant] messages with Qwen3.5-27B's chat template (no system
           prompt, no generation prompt), and finds the readout: the last content token of the assistant turn (the token
           before its closing <|im_end|>, stepping back over whitespace-only tokens; never a special token). Problem
           group = sha256(problem_description); held-out = the full code_* rows whose group is not a training group;
           halves by sha256(seed|group). Saves token ids and metadata (no text) to the score Volume.
  extract  one H200: the output of every decoder block (base.layers[l] forward hook, 0-based) at the readout, all 64
           layers, fp16, as run_contrast.py and run_extract_replay.py do.
  fit      CPU: layers 16-48. C by 5-fold grouped CV on training rows only (fold = first byte of sha256(group) mod 5,
           standardization fit inside each training fold, mean fold AUROC, ties to the smaller C); final fit on all
           training rows; layer = best selection-half AUROC (ties: nearest 32, then lower; selection-half sabotage-only
           AUROC beside it). Validity gate, computed once at the chosen layer only: AUROC >= 0.85 on the validity half's
           code_sabotage rows. Saves the frozen probe only if the gate passes, and returns numbers only.
  fetch_probe  the frozen probe's bytes (weights only) for the S3 freeze upload.

  modal deploy experiments/2026-10-11_whitebox_dataset_probe/run_whitebox.py
  python -c "import modal; print(modal.Function.from_name('<app>', 'stage').spawn().object_id)"   (then extract, then fit)
"""

import hashlib
import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-wbprobe_run_kill-2026-11-10"
SCORE_VOL, HF_CACHE = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "probe-shift_aswin-surya_hf-cache"
MODEL = {"name": "Qwen/Qwen3.5-27B", "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654"}
OUT, SEED, C_GRID, LAYERS_FIT, GATE = "whitebox_probe_v1", 20261014, [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], range(16, 49), 0.85
DATASETS = {  # name -> pinned Hugging Face revision (candywal/<name>), read from the HF API's `sha` on 2026-10-11
    "train_in_distribution_code_sabotage_safe": "7f2616377cb85a34cfb7710591f673f2d2beb0a3",
    "train_in_distribution_code_sabotage_unsafe": "09373c478325b622ded38d9d38a94c54ac343014",
    "train_in_distribution_code_deception_safe": "9db246f59bdd32d18e74dffc31ff866c5c84f413",
    "train_in_distribution_code_deception_unsafe": "6f31c2c8aa6db2edead446b3188ca5d0e03bb3b9",
    "train_in_distribution_code_rule_violation_safe": "3939a3180f15200eafb29ab5a24f75a9164b42dd",
    "train_in_distribution_code_rule_violation_unsafe": "92781575706d0d666b1eca56b9dee8187c193024",
    "code_sabotage_safe": "b1162c54336adbed67373df339016ee25167789f",
    "code_sabotage_unsafe": "96aa1fdbaadcbce81ed1dd692cbcaef044991c19",
    "code_deception_safe": "d42a31df7de1a0e4c6b19a4cf3c646fdd9c16fc5",
    "code_deception_unsafe": "3594dd72259f4d2bb64464b5a9b7c83bd0a80eb4",
    "code_rule_violation_safe": "3407d4f817354eace15257d41568f6855c841fca",
    "code_rule_violation_unsafe": "29934f00f6288f9a63cf0671a56777760746e0eb"}
PARQUET = "data/train-00000-of-00001.parquet"
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]

app = modal.App(APP)
score_vol = modal.Volume.from_name(SCORE_VOL, version=2)
hf = modal.Volume.from_name(HF_CACHE).read_only()
gpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
stage_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy", "pandas", "pyarrow", "huggingface_hub")
fit_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2", "joblib")
             .env({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}))  # 16 workers, one BLAS thread each


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@app.function(image=stage_image, cpu=4, memory=16384, timeout=900, retries=0, scaledown_window=2, volumes={"/hf": hf, "/score": score_vol})
def stage(code_commit: str) -> dict:
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    out = Path("/score") / OUT
    if (out / "staged.json").exists():
        raise FileExistsError(out / "staged.json")
    tok = AutoTokenizer.from_pretrained(MODEL["name"], revision=MODEL["revision"], cache_dir="/hf/hub", local_files_only=True)
    end, special = tok.convert_tokens_to_ids("<|im_end|>"), set(tok.all_special_ids)
    think_end, header = tok.convert_tokens_to_ids("</think>"), tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    files, items, ids_all = {}, [], []
    for name, rev in DATASETS.items():
        p = Path(hf_hub_download(f"candywal/{name}", PARQUET, repo_type="dataset", revision=rev, cache_dir="/tmp/hfds"))
        files[name] = {"revision": rev, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        df = pd.read_parquet(p)
        has_pd = "problem_description" in df.columns  # the deception sets have none: their group is the user turn (amendment 1)
        files[name].update(rows=len(df), group_key="problem_description" if has_pd else "user_turn")
        for k, (msgs, prob) in enumerate(zip(df["messages"], df["problem_description"] if has_pd else [None] * len(df))):
            msgs = [{"role": str(m["role"]), "content": str(m["content"])} for m in msgs]
            prob = prob if has_pd else msgs[0]["content"]
            if [m["role"] for m in msgs] != ["user", "assistant"]:
                raise ValueError(f"{name} row {k}: roles {[m['role'] for m in msgs]}")
            ids = tok.apply_chat_template(msgs, tokenize=True)
            ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
            pre = tok.apply_chat_template(msgs[:1], tokenize=True)  # the user turn alone (no generation prompt)
            pre = list(pre["input_ids"] if hasattr(pre, "keys") else pre)
            seg = len(pre)  # the assistant turn starts here: the boundary comes from the template, not a text search
            ends = [j for j in range(seg, len(ids)) if ids[j] == end]
            if ids[:seg] != pre or ids[seg:seg + len(header)] != header or not ends:
                raise ValueError(f"{name} row {k}: the user turn, assistant header or closing <|im_end|> is not where the template puts it")
            thinks = [j for j in range(seg, ends[0]) if ids[j] == think_end]  # Qwen3.5 renders an empty thinking block before the answer
            a0 = thinks[0] + 1 if thinks else seg + len(header)
            pos = ends[0] - 1  # the assistant turn's own closing <|im_end|>
            while pos >= a0 and not tok.decode([ids[pos]]).strip():
                pos -= 1
            if pos < a0 or ids[pos] in special:
                raise ValueError(f"{name} row {k}: no content readout in the assistant turn")
            items.append({"dataset": name, "row": k, "label": int(name.endswith("_unsafe")),
                          "behavior": next(b for b in ("sabotage", "deception", "rule_violation") if f"code_{b}_" in name),
                          "set": "train" if name.startswith("train_in_distribution_") else "full", "group": h(str(prob)),
                          "n_tokens": len(ids), "pos": pos})
            ids_all.append(np.array(ids, np.int32))
    train_groups = {it["group"] for it in items if it["set"] == "train"}
    for it in items:
        if it["set"] == "train":
            it["split"] = "train"
        elif it["group"] in train_groups:
            it["split"] = "dropped_overlap"
        else:
            it["split"] = "selection" if int(h(f"{SEED}|{it['group']}")[:2], 16) % 2 == 0 else "validity"
    n_dropped = sum(it["split"] == "dropped_overlap" for it in items)
    keep = [i for i, it in enumerate(items) if it["split"] != "dropped_overlap"]  # overlap rows are counted, never extracted
    items, ids_all = [items[i] for i in keep], [ids_all[i] for i in keep]
    y = np.array([it["label"] for it in items])
    folds = np.array([int(h(it["group"])[:2], 16) % 5 if it["split"] == "train" else -1 for it in items])
    checks = {f"fold{f}_{side}": sorted(set(y[(folds >= 0) & ((folds == f) if side == "val" else (folds != f))].tolist())) for f in range(5) for side in ("train", "val")}
    checks["selection"] = sorted(set(y[[it["split"] == "selection" for it in items]].tolist()))
    checks["selection_sabotage"] = sorted(set(y[[it["split"] == "selection" and it["behavior"] == "sabotage" for it in items]].tolist()))
    checks["validity_sabotage"] = sorted(set(y[[it["split"] == "validity" and it["behavior"] == "sabotage" for it in items]].tolist()))
    if any(v != [0, 1] for v in checks.values()):  # both classes everywhere, before any GPU spend
        raise ValueError(f"a set lacks a class: {checks}")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "token_ids.npz", ids=np.concatenate(ids_all), offsets=np.cumsum([0] + [len(x) for x in ids_all]))
    meta = json.dumps({"model": MODEL, "files": files, "items": items, "code_commit": code_commit, "dropped_overlap": n_dropped})
    (out / "staged.json").write_text(meta)
    score_vol.commit()
    from collections import Counter
    sums = {"staged_json_sha256": hashlib.sha256(meta.encode()).hexdigest(),
            "token_ids_sha256": hashlib.sha256((out / "token_ids.npz").read_bytes()).hexdigest()}
    return {"files": files, "items": len(items), "dropped_overlap": n_dropped, **sums, "class_checks": checks,
            "tokens": int(sum(it["n_tokens"] for it in items)),
            "max_tokens": max(it["n_tokens"] for it in items),
            "by_split": {f"{s}|{b}|{l}": n for (s, b, l), n in sorted(Counter((it["split"], it["behavior"], it["label"]) for it in items).items())}}


@app.function(image=gpu_image, gpu="H200", memory=65536, timeout=1500, retries=0, scaledown_window=2, volumes={"/hf": hf, "/score": score_vol})
def extract() -> dict:  # 25-minute cap: about $2.3 at most; the whole-job watcher stops the app at $2.89
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    score_vol.reload()
    out = Path("/score") / OUT
    if (out / "acts.npy").exists():
        raise FileExistsError(out / "acts.npy")
    st = json.loads((out / "staged.json").read_text())
    z = np.load(out / "token_ids.npz")
    flat, off, N = z["ids"], z["offsets"], len(st["items"])
    lm = AutoModelForCausalLM.from_pretrained(MODEL["name"], revision=MODEL["revision"], dtype=torch.bfloat16, device_map="cuda",
                                              attn_implementation="sdpa").eval()
    base, layers, d = lm.model, list(range(lm.config.num_hidden_layers)), lm.config.hidden_size
    part, prog = out / "acts.partial.npy", out / "progress.json"
    start = json.loads(prog.read_text())["done"] if part.exists() and prog.exists() else 0  # resumable after a stop
    acts = np.lib.format.open_memmap(part, mode="r+" if start else "w+", dtype=np.float16, shape=(N, len(layers), d))
    cur, got = {"pos": 0}, {}
    hooks = [base.layers[l].register_forward_hook(lambda m, x, o, l=l: got.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0, cur["pos"]].detach()))
             for l in layers]  # GPU slices only; one stack and one copy per item
    try:
        for n in range(start, N):
            cur["pos"] = st["items"][n]["pos"]
            got.clear()
            with torch.inference_mode():
                base(input_ids=torch.tensor([flat[off[n]:off[n + 1]].tolist()], device="cuda"), use_cache=False)
            a = torch.stack([got[l] for l in layers]).to(torch.float16).cpu().numpy()
            if not np.isfinite(a).all():  # checked after the float16 conversion that is stored
                raise ValueError(f"item {n}: non-finite activations")
            acts[n] = a
            if (n + 1) % 500 == 0:
                acts.flush()
                prog.write_text(json.dumps({"done": n + 1}))
                score_vol.commit()
    finally:
        for hk in hooks:
            hk.remove()
    acts.flush()
    del acts
    part.rename(out / "acts.npy")
    prog.write_text(json.dumps({"done": N}))
    score_vol.commit()
    return {"items": N, "resumed_from": start, "layers": len(layers), "hidden": int(d)}


def _fit_layer(L, X, y, folds, sel, sel_sab, Cs):
    """C by grouped CV on training rows; the final fit; selection-half AUROCs. The validity half is never touched here."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    tr = np.flatnonzero(folds >= 0)
    cv = {}
    for C in Cs:
        aucs = []
        for f in range(5):
            a, b = tr[folds[tr] != f], tr[folds[tr] == f]
            m, s = X[a].mean(0), X[a].std(0)
            s[s == 0] = 1.0
            clf = LogisticRegression(C=C, max_iter=20000).fit((X[a] - m) / s, y[a])
            aucs.append(roc_auc_score(y[b], clf.decision_function((X[b] - m) / s)))
        cv[C] = float(np.mean(aucs))
    best = max(cv.values())
    C = min(c for c in Cs if cv[c] == best)
    m, s = X[tr].mean(0), X[tr].std(0)
    s[s == 0] = 1.0
    clf = LogisticRegression(C=C, max_iter=20000).fit((X[tr] - m) / s, y[tr])
    score = lambda idx: clf.decision_function((X[idx] - m) / s)  # noqa: E731
    return {"layer": L, "cv_auroc": cv, "C": C, "n_iter": int(clf.n_iter_[0]), "max_iter_hit": bool(clf.n_iter_[0] >= 20000),
            "selection_auroc": float(roc_auc_score(y[sel], score(sel))), "selection_sabotage_auroc": float(roc_auc_score(y[sel_sab], score(sel_sab))),
            "probe": {"coef": clf.coef_[0].astype(np.float64), "intercept": float(clf.intercept_[0]), "mean": m.astype(np.float64), "std": s.astype(np.float64)}}


@app.function(image=fit_image, cpu=16, memory=65536, timeout=1200, retries=0, scaledown_window=2, volumes={"/score": score_vol})
def fit() -> dict:
    import io

    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score

    score_vol.reload()
    out = Path("/score") / OUT
    if (out / "probe.npz").exists() or (out / "fit_summary.json").exists():
        raise FileExistsError(out / "fit_summary.json")
    it = json.loads((out / "staged.json").read_text())["items"]
    y = np.array([x["label"] for x in it])
    folds = np.array([int(h(x["group"])[:2], 16) % 5 if x["split"] == "train" else -1 for x in it])
    sel = np.flatnonzero([x["split"] == "selection" for x in it])
    sel_sab = np.flatnonzero([x["split"] == "selection" and x["behavior"] == "sabotage" for x in it])
    val_sab = np.flatnonzero([x["split"] == "validity" and x["behavior"] == "sabotage" for x in it])
    acts = np.load(out / "acts.npy")  # whole file (about 1.6 GB): row-strided mmap reads on a Volume are slow
    res = Parallel(n_jobs=16)(delayed(_fit_layer)(L, np.asarray(acts[:, L, :], np.float32), y, folds, sel, sel_sab, C_GRID) for L in LAYERS_FIT)
    by = {r["layer"]: r for r in res}
    top = max(r["selection_auroc"] for r in res)
    L = sorted((abs(l - 32), l) for l, r in by.items() if r["selection_auroc"] == top)[0][1]  # ties: nearest 32, then lower
    p = by[L]["probe"]
    X = np.asarray(acts[val_sab, L, :], np.float64)  # the validity half: used once, at the chosen layer only
    val_auc = float(roc_auc_score(y[val_sab], ((X - p["mean"]) / p["std"]) @ p["coef"] + p["intercept"]))
    passed = val_auc >= GATE
    summary = {"layers": {l: {k: v for k, v in rr.items() if k != "probe"} for l, rr in by.items()}, "chosen_layer": L, "chosen_C": by[L]["C"],
               "validity_sabotage_auroc_at_chosen_layer": val_auc, "gate": GATE, "gate_passed": bool(passed),
               "counts": {"train": int((folds >= 0).sum()), "selection": len(sel), "selection_sabotage": len(sel_sab),
                          "validity_sabotage": len(val_sab), "validity_sabotage_positives": int(y[val_sab].sum())}}
    if passed:
        buf = io.BytesIO()
        np.savez(buf, layer=np.int64(L), C=np.float64(by[L]["C"]), **p)
        b = buf.getvalue()
        (out / "probe.npz").write_bytes(b)
        summary["probe_sha256"] = hashlib.sha256(b).hexdigest()
    (out / "fit_summary.json").write_text(json.dumps(summary))
    score_vol.commit()
    return summary


@app.function(image=fit_image, timeout=120, retries=0, scaledown_window=2, volumes={"/score": score_vol})
def fetch_probe() -> bytes:
    """The frozen probe's bytes (about 120 KB of weights, no data), for the S3 freeze upload from the laptop."""
    score_vol.reload()
    return (Path("/score") / OUT / "probe.npz").read_bytes()
