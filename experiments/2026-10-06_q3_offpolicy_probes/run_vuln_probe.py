"""Train a vulnerability probe on Qwen3-32B (Modal) from Emergent Misalignment insecure/secure code, validate it on
held-out prompts, and test transfer on SafeCoder + SVEN real vulnerability fixes.

Data (probe_shift.vuln): every code sample is the file_text of a text_editor create call after the same neutral
messages, rendered as the LinuxArena reader renders a mid-transcript call; label 1 = insecure / vulnerable.
  - Emergent Misalignment (EM): 3,000 insecure + 3,000 secure Python completions (seed 0), split 80/20 by task group
    (probe_shift.vuln.em_split_groups: paraphrased prompts, shared templates and duplicate completions stay together).
  - SafeCoder: its val split picks the primary layer, minus pairs sharing a commit or code with its train split; its
    train split (never trained on) reports transfer, overall and per language. Functions over 6,000 characters are
    dropped (cost).
1. extract (one H100): residual stream after layers 16..63 at the call_end token (the call's last content token,
   LinuxArena's readout), the mean over code tokens, and 8 sampled code tokens per EM sample (training rows).
2. fit (CPU): Apollo's recipe (per-token logistic regression on standardized activations, no intercept, C = 1/10)
   on EM training tokens; a last-token variant (one call_end row per sample) for comparison. Evaluated with the
   mean-over-code readout and the call_end readout. Label-shuffle null (EM labels permuted within the training
   set, 5 refits per layer). Baselines: TF-IDF (raw text) logistic regression and code length.
The primary layer for LinuxArena is the best SafeCoder-val AUROC of the Apollo-recipe probe at call_end, fixed here.

  PYTHONPATH=src uv run --no-project --python 3.13 --with 'modal>=1.2' --with numpy modal run \
      experiments/2026-10-06_q3_offpolicy_probes/run_vuln_probe.py --data-dir <dir with em/ and safecoder/>
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

from probe_shift.models import QWEN3_32B

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q3_vuln_kill-2026-11-06"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
OUTPUT = "probe-shift_aswin-surya_sub-q3_outputs_kill-2026-11-06"
GPU = "H100!"
PIP = ["torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "scikit-learn==1.7.2", "joblib", "threadpoolctl"]
CFG = {
    "acts_run": "vuln_v1",                    # Volume acts/<acts_run>.npz (split-independent, so reusable)
    "probe_run": "vuln_v3",                   # Volume probes/<probe_run>/; v1 leaky EM split, v2 missed identical prompts
    "em_group_max_key_share": 8,              # probe_shift.vuln.em_split_groups
    "model": QWEN3_32B["name"], "dtype": "bfloat16",
    "revision": QWEN3_32B["revision"],        # pinned; the only snapshot on the HF cache Volume, so vuln_v1 used it too
    "layers": [16, 24, 32, 40, 48, 56, 63],   # every saved LinuxArena layer from 16 up
    "readout": "block_output",
    "em_per_class": 3000, "em_val_fraction": 0.2, "seed": 0,
    "tokens_per_sample": 8,                   # sampled code tokens per EM sample for per-token training
    "max_code_chars": 6000,
    "reg_coeff": 10, "max_iter": 2000, "shuffle_null": 5,
    "sources": {"emergent_misalignment": "github.com/emergent-misalignment/emergent-misalignment@80c11967 data/{insecure,secure}.jsonl",
                "safecoder": "github.com/eth-sri/SafeCoder@1bf9d57b data_train_val/{train,val}/{sec-desc,sec-new-desc}.jsonl"},
}
# The revision is passed to the loaders but not fingerprinted: acts/vuln_v1 was made from the pinned snapshot (the only
# one cached) before the pin existed, and run_vuln_safecoder_probe.py checks that same fingerprint.
EXTRACT_KEYS = ("model", "dtype", "layers", "readout", "tokens_per_sample", "seed")

app = modal.App(APP)
hf_cache = modal.Volume.from_name(HF_CACHE).read_only()
out_vol = modal.Volume.from_name(OUTPUT, create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP)
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


def fingerprint(cfg: dict, items, keys) -> str:
    from probe_shift.vuln import fingerprint as fp  # one definition, shared with run_vuln_safecoder_probe.py

    return fp(cfg, items, keys)


FIT_META_KEYS = ("id", "label", "source", "split", "language", "pair")  # what the fit reads; meta also carries "commit"


def fit_fingerprint(cfg: dict, meta: dict, texts: list[str]) -> str:
    """Fingerprint of a fit: every setting but the sources and the pinned revision (both fixed for the cached runs;
    probes/vuln_v3 was fit before either was recorded), the metadata the fit reads, and the code texts."""
    keys = tuple(k for k in cfg if k not in ("sources", "revision"))
    return fingerprint(cfg, [{k: meta[k] for k in FIT_META_KEYS}, texts], keys)


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache, "/out": out_vol}, timeout=3600, memory=65536)
def extract(cfg: dict, items: list[dict]) -> dict:
    import random
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.vuln import render

    path = Path("/out") / "acts" / f"{cfg['acts_run']}.npz"
    stamp = path.with_suffix(".fingerprint")
    if path.exists():
        if not stamp.exists() or stamp.read_text() != fingerprint(cfg, items, EXTRACT_KEYS):
            raise FileExistsError(f"{path} was extracted from other inputs or settings; change cfg['acts_run']")
        return {"status": "reused"}
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=getattr(torch, cfg["dtype"]),
                                                device_map="cuda", attn_implementation="sdpa").model.eval()
    rng = random.Random(cfg["seed"])
    L = cfg["layers"]
    last, mean, toks, owner, n_code = {l: [] for l in L}, {l: [] for l in L}, {l: [] for l in L}, [], []
    for i, it in enumerate(items):
        ids, code, end = render(tok, it)
        sample = sorted(rng.sample(code, min(len(code), cfg["tokens_per_sample"]))) if it["source"] == "em" else []
        captured = {}

        def grab(l):
            def hook(m, a, o):
                h = (o[0] if isinstance(o, tuple) else o)[0]
                captured[l] = (h[end], h[code].float().mean(0), h[sample] if sample else None)
            return hook

        hooks = [base.layers[l].register_forward_hook(grab(l)) for l in L]
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        for l in L:
            e, m, s = captured[l]
            last[l].append(e.to(torch.float16).cpu().numpy())
            mean[l].append(m.to(torch.float16).cpu().numpy())
            if s is not None:
                toks[l].append(s.to(torch.float16).cpu().numpy())
        owner += [i] * len(sample)
        n_code.append(len(code))
    arrays = {"tok_owner": np.array(owner), "n_code_tokens": np.array(n_code)}
    for l in L:
        arrays |= {f"L{l}_last": np.stack(last[l]), f"L{l}_mean": np.stack(mean[l]), f"L{l}_tok": np.concatenate(toks[l])}
    nonfinite = int(sum((~np.isfinite(v.astype(np.float32))).sum() for k, v in arrays.items() if k.startswith("L")))
    if nonfinite:  # never save (or fingerprint) bad activations: a retry would reuse them
        raise ValueError(f"{nonfinite} non-finite activations; nothing saved")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    stamp.write_text(fingerprint(cfg, items, EXTRACT_KEYS))
    out_vol.commit()
    return {"status": "extracted", "items": len(items), "token_rows": len(owner), "nonfinite": nonfinite,
            "total_seconds": round(time.time() - t0), "device": torch.cuda.get_device_name()}


def layer_job(l: int, z: dict, meta: dict, cfg: dict) -> tuple[dict, dict, dict]:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from threadpoolctl import threadpool_limits

    from probe_shift.offpolicy import fit_token_probe, probe_scores

    y, src, split, lang, pair = (np.array(meta[k]) for k in ("label", "source", "split", "language", "pair"))
    em_tr, em_val = (src == "em") & (split == "train"), (src == "em") & (split == "val")
    sc = {"safecoder_val": src == "safecoder_val", "safecoder_train": src == "safecoder_train"}
    owner = z["tok_owner"]
    Xtok, Xlast, Xmean = (z[f"L{l}_{k}"].astype(np.float32) for k in ("tok", "last", "mean"))

    def auc(s, m):
        return round(float(roc_auc_score(y[m], s[m])), 4) if len(set(y[m])) == 2 else None

    def pair_acc(s, m):
        d = {}
        for p, v, lab in zip(pair[m], s[m], y[m]):
            d.setdefault(p, {})[lab] = v
        return round(float(np.mean([v[1] > v[0] for v in d.values() if len(v) == 2])), 4)

    def evaluate(probe):
        r = {}
        for readout, X in (("call_end", Xlast), ("mean_code", Xmean)):
            s = probe_scores(X, probe)
            r[readout] = {"em_val_auroc": auc(s, em_val), **{f"{k}_auroc": auc(s, m) for k, m in sc.items()},
                          **{f"{k}_pair_acc": pair_acc(s, m) for k, m in sc.items()},
                          "safecoder_train_by_language": {g: auc(s, sc["safecoder_train"] & (lang == g))
                                                          for g in sorted(set(lang[sc["safecoder_train"]]))}}
        return r

    rows = em_tr[owner]
    with threadpool_limits(4):
        token_probe, conv = fit_token_probe(Xtok[rows], y[owner[rows]], cfg["reg_coeff"], cfg["max_iter"])
        last_probe, conv_last = fit_token_probe(Xlast[em_tr], y[em_tr], cfg["reg_coeff"], cfg["max_iter"])
    res = {"apollo_recipe": {"converged": conv, **evaluate(token_probe)},
           "last_token": {"converged": conv_last, **evaluate(last_probe)},
           "label_shuffle_null": shuffle_null(l, z, meta, cfg)}
    return res, token_probe, last_probe


def shuffle_null(l: int, z: dict, meta: dict, cfg: dict) -> dict:
    """Apollo-recipe refits with labels permuted among EM training samples (tokens follow their sample), scored on
    every evaluation set and readout the real probe is: EM val, SafeCoder val (layer choice), SafeCoder train."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from threadpoolctl import threadpool_limits

    from probe_shift.offpolicy import fit_token_probe, probe_scores

    y, src, split = (np.array(meta[k]) for k in ("label", "source", "split"))
    em_tr = (src == "em") & (split == "train")
    sets = {"em_val": (src == "em") & (split == "val"), "safecoder_val": src == "safecoder_val", "safecoder_train": src == "safecoder_train"}
    owner = z["tok_owner"]
    rows = em_tr[owner]
    X = {"call_end": z[f"L{l}_last"].astype(np.float32), "mean_code": z[f"L{l}_mean"].astype(np.float32)}
    Xtok = z[f"L{l}_tok"].astype(np.float32)
    draws = {f"{r}/{s}": [] for r in X for s in sets}
    with threadpool_limits(4):
        for k in range(cfg["shuffle_null"]):
            perm, idx = y.copy(), np.flatnonzero(em_tr)
            perm[idx] = np.random.default_rng(100 + k).permutation(y[idx])
            p, _ = fit_token_probe(Xtok[rows], perm[owner[rows]], cfg["reg_coeff"], cfg["max_iter"])
            for r, Xr in X.items():
                s = probe_scores(Xr, p)
                for name, m in sets.items():
                    draws[f"{r}/{name}"].append(float(roc_auc_score(y[m], s[m])))
    return {k: {"n": len(v), "mean": round(float(np.mean(v)), 4), "max": round(float(np.max(v)), 4)} for k, v in draws.items()}


@app.function(image=image, volumes={"/out": out_vol}, cpu=32, memory=131072, timeout=3600)
def null_only(cfg: dict, meta: dict, items_fingerprint: str) -> dict:
    """The label-shuffle null on every evaluation set, for a run whose probes are already fit. The activations must
    come from exactly these items (checked against the extraction's fingerprint), or the masks would not line up."""
    import numpy as np
    from joblib import Parallel, delayed

    out_vol.reload()
    path = Path("/out") / "acts" / f"{cfg['acts_run']}.npz"
    if path.with_suffix(".fingerprint").read_text() != items_fingerprint:
        raise ValueError(f"{path} was extracted from other items or settings")
    z = dict(np.load(path))
    outs = Parallel(n_jobs=len(cfg["layers"]))(delayed(shuffle_null)(l, {k: v for k, v in z.items() if k.startswith(f"L{l}_") or not k.startswith("L")},
                                                                       meta, cfg) for l in cfg["layers"])
    return {str(l): r for l, r in zip(cfg["layers"], outs)}


@app.function(image=image, volumes={"/out": out_vol}, cpu=32, memory=131072, timeout=3600)
def fit(cfg: dict, meta: dict, texts: list[str], items_fingerprint: str) -> dict:
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    pdir = Path("/out") / "probes" / cfg["probe_run"]
    summary = pdir / "fit.json"
    fp = fit_fingerprint(cfg, meta, texts)
    if pdir.exists() and any(pdir.iterdir()):
        if summary.exists() and json.loads(summary.read_text())["fingerprint"] == fp:
            return json.loads(summary.read_text())["result"]
        raise FileExistsError(f"{pdir} already holds probes from other inputs or settings; change cfg['probe_run']")
    out_vol.reload()
    path = Path("/out") / "acts" / f"{cfg['acts_run']}.npz"
    if path.with_suffix(".fingerprint").read_text() != items_fingerprint:  # the masks below must line up with its rows
        raise ValueError(f"{path} was extracted from other items or settings")
    z = dict(np.load(path))
    bad ={k: int((~np.isfinite(v.astype(np.float32))).sum()) for k, v in z.items() if k.startswith("L")}
    if any(bad.values()):  # also covers activations reused from an earlier (e.g. preempted) extraction
        raise ValueError(f"non-finite activations: {bad}")
    outs = Parallel(n_jobs=len(cfg["layers"]))(delayed(layer_job)(l, {k: v for k, v in z.items() if k.startswith(f"L{l}_") or not k.startswith("L")},
                                                                     meta, cfg) for l in cfg["layers"])
    pdir.mkdir(parents=True, exist_ok=True)
    by_layer = {}
    for l, (res, token_probe, last_probe) in zip(cfg["layers"], outs):
        np.savez(pdir / f"probe_L{l}.npz", **token_probe)          # Apollo recipe: the probe read on LinuxArena
        np.savez(pdir / f"probe_last_L{l}.npz", **last_probe)
        by_layer[str(l)] = res

    # Baselines on the code text: TF-IDF (raw text) logistic regression, and code length, trained on EM train.
    y, src, split = (np.array(meta[k]) for k in ("label", "source", "split"))
    tr = (src == "em") & (split == "train")
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vec.fit([t for t, keep in zip(texts, tr) if keep])  # vocabulary from training texts only
    Xt = vec.transform(texts)
    s_tfidf = LogisticRegression(C=1.0, max_iter=2000).fit(Xt[tr], y[tr]).decision_function(Xt)
    length = np.array([len(t) for t in texts], float)
    sign = 1.0 if roc_auc_score(y[tr], length[tr]) >= 0.5 else -1.0
    baselines = {}
    for name, s in (("tfidf_raw_text", s_tfidf), ("code_length", sign * length)):
        baselines[name] = {k: round(float(roc_auc_score(y[m], s[m])), 4) for k, m in
                           (("em_val", (src == "em") & (split == "val")), ("safecoder_val", src == "safecoder_val"),
                            ("safecoder_train", src == "safecoder_train"))}
    sel = {l: r["apollo_recipe"]["call_end"]["safecoder_val_auroc"] for l, r in by_layer.items()}
    result = {"by_layer": by_layer, "baselines": baselines, "primary_layer_metric": sel,
              "primary_layer": int(max(sel, key=sel.get)), "probe_dir": f"{OUTPUT}/probes/{cfg['probe_run']}"}
    summary.write_text(json.dumps({"fingerprint": fp, "config": cfg, "result": result}))
    out_vol.commit()
    return result


@app.local_entrypoint()
def main(data_dir: str, limit: int = 0, null_only_run: bool = False):
    """`--limit N`: a test with N EM samples per class and N SafeCoder pairs per split, under run name <run>_test."""
    from collections import Counter

    from probe_shift.vuln import load_items

    global CFG
    if limit:
        CFG = CFG | {"acts_run": CFG["acts_run"] + "_test", "probe_run": CFG["probe_run"] + "_test", "em_per_class": limit,
                     "shuffle_null": 1}
    # EM: task-grouped split (paraphrased prompts, shared templates, duplicate prompts and completions stay together).
    # SafeCoder: val picks the layer and train reports transfer, so val pairs sharing a commit or a function with train
    # are set aside from both roles (items stay unchanged, so cached activations still match).
    em, sc, items, meta, counts = load_items(data_dir, CFG, limit)
    print(f"EM {len(em)} ({Counter(it['split'] for it in em)}), SafeCoder {len(sc)} items; {counts}:",
          dict(Counter((it["source"], it["language"]) for it in sc)), flush=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S") + ("_test" if limit else "")
    (HERE / "results").mkdir(exist_ok=True)
    if null_only_run:  # the label-shuffle null on every evaluation set, from the saved activations (probes untouched)
        null = null_only.remote(CFG, meta, fingerprint(CFG, items, EXTRACT_KEYS))
        path = HERE / "results" / f"vuln_null_{stamp}.json"
        path.write_text(json.dumps({"config": CFG, "label_shuffle_null": null}, indent=1))
        for l, r in null.items():
            print(f"L{l}: " + "  ".join(f"{k} mean {v['mean']:.3f} max {v['max']:.3f}" for k, v in r.items()))
        print(f"-> {path}")
        return
    ext = extract.remote(CFG, items)
    print("extract:", ext, flush=True)
    if ext.get("nonfinite"):
        raise ValueError(f"{ext['nonfinite']} non-finite activations")
    res = fit.remote(CFG, meta, [it["code"] for it in items], fingerprint(CFG, items, EXTRACT_KEYS))
    from probe_shift.offpolicy import provenance

    out = {"extract": ext, "config": CFG | {"modal": {"app": APP, "gpu": GPU, "pip": PIP, "output_volume": OUTPUT}},
           "provenance": provenance(__file__, HERE.parent.parent / "src/probe_shift/vuln.py", HERE.parent.parent / "src/probe_shift/offpolicy.py"),
           "items": {"ids": meta["id"], "split": meta["split"], **counts}} | res
    path = HERE / "results" / f"vuln_probe_{stamp}.json"
    path.write_text(json.dumps(out, indent=1))
    print("baselines:", json.dumps(res["baselines"]))
    for l, r in res["by_layer"].items():
        a, b = r["apollo_recipe"], r["last_token"]
        print(f"L{l}: apollo-recipe call_end EM val {a['call_end']['em_val_auroc']} SC val {a['call_end']['safecoder_val_auroc']} "
              f"SC train {a['call_end']['safecoder_train_auroc']} (pair acc {a['call_end']['safecoder_train_pair_acc']}) | mean-code "
              f"EM val {a['mean_code']['em_val_auroc']} SC train {a['mean_code']['safecoder_train_auroc']} | last-token probe "
              f"EM val {b['call_end']['em_val_auroc']} SC train {b['call_end']['safecoder_train_auroc']} | null SC train call_end {r['label_shuffle_null']['call_end/safecoder_train']}")
    print(f"primary layer (best SafeCoder-val AUROC at call_end): {res['primary_layer']}\n-> {path}")
