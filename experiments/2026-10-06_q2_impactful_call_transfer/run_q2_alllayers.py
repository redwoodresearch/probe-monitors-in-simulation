"""Q2 at all 64 layers: per-layer 2x2 curves for call_end, pre_call and prev_content, and the inner-CV headline layer.

Uses the identical eligible calls (and labels) as a run_q2_probes.py features run, read from the main session's
all-layer segment-end run (one fp16 file per layer and chunk). Steps, all on Modal:
  1. index (CPU): map each eligible call's three positions to (chunk, row) in the all-layer files.
  2. pack (CPU, one container per layer): gather those rows into [call, hidden] arrays per layer and position.
  3. fit (GPU): per layer, probe_shift.transfer with C chosen on each outer fold's training-domain validation; for
     the real labels and for within-transcript label shuffles. Saves per-layer test scores and fold validation AUROCs.
  4. summary (CPU): per-layer 2x2 with main-task bootstrap intervals; the headline layer chosen inside each outer
     fold (earliest within layer_tolerance of the best validation AUROC across all 64 layers), with the same choice
     repeated per shuffle draw for its null.
An orchestrator runs the steps on Modal (launch with modal run --detach); --fetch reads its saved result.

--config q2_threeway (with the simulated honest arm, #31) adds that arm and fits three views (attack only, honest only, pooled), each
mapping arms to a real and a simulated domain. Its extra arm is indexed and packed on its own and stacked after the
attack-only rows, which are reused as packed, so the attack-only view reproduces the attack-only run.

  uv run modal run --detach experiments/2026-10-06_q2_impactful_call_transfer/run_q2_alllayers.py [--draws 1-5] [--config q2_threeway]
"""

import hashlib
import json
import re
import secrets
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q2_alllayers_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"  # read only
FEATS = "probe-shift_aswin-surya_sub-q2_probe-features_kill-2026-11-06"
POSITIONS = ["call_end", "pre_call", "prev_content"]
FIT_KEYS = ["fit_source_sha", "task_set_sha", "labeler_sha", "label_rule", "alllayers_run", "extra_alllayers_run",
            "reference_alllayers_run", "C_grid", "max_iter", "fpr", "seed",
            "train_sets", "test_domains", "home", "views", "honest_tasks", "primary_real_honest"]

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
feats_vol = modal.Volume.from_name(FEATS, version=2)
cpu_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2", "joblib")
             .add_local_python_source("probe_shift"))
gpu_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2", "torch==2.8.0")
             .add_local_python_source("probe_shift"))


def out_dir(cfg, base: bool = False) -> Path:
    return Path("/feats") / cfg["features_run"] / ("alllayers" if base else cfg.get("out_sub", "alllayers"))


# Packed before packs recorded their source (pack_source.json): (pack directory, the activations run it holds).
LEGACY_PACKS = {("alllayers", "2026-10-06_la_segment-end_qwen3-32b_alllayers_v1"),
                ("threeway", "2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1")}


def pack_dirs(cfg) -> list[tuple[Path, str]]:
    """The pack directories fit stacks, in row order, each with the activations run its rows must come from: the run's
    own out_sub (alllayers_run); with views, the attack-only pack of alllayers_run, then the extra arm's pack."""
    if cfg.get("views"):
        return [(out_dir(cfg, base=True), cfg["alllayers_run"]), (out_dir(cfg), cfg["extra_alllayers_run"])]
    return [(out_dir(cfg), cfg["alllayers_run"])]


def check_pack_source(layer_dir: Path, run: str) -> None:
    """Raise unless the pack in layer_dir was made from activations run `run`. A pack without pack_source.json is
    accepted only where LEGACY_PACKS names it."""
    f = layer_dir / "pack_source.json"
    if f.exists():
        got = json.loads(f.read_text())["activations_run"]
        if got != run:
            raise ValueError(f"{layer_dir}: packed from {got}, but this config reads {run}")
    elif (layer_dir.parent.name, run) not in LEGACY_PACKS:
        raise ValueError(f"{layer_dir}: no pack_source.json, so its activations run cannot be checked against {run}")


def views_of(cfg) -> dict:
    """View name -> {domains: arm -> domain, one_run}. Without `views`: one unnamed view, each arm its own domain."""
    return cfg.get("views") or {"": {"domains": None}}


def view_domain(cfg, manifest: list[dict], view: str):
    """Each row's domain in the view; "-" leaves the row out (never trained on, never tested). Honest rows outside the
    view's tasks, and real honest runs other than each task's chosen one when the view has one_run, are left out."""
    import numpy as np

    v = views_of(cfg)[view]
    if v["domains"] is None:
        return np.array([m["arm"] for m in manifest])
    keep = lambda m: m.get("in_honest_tasks", True) and (m.get("primary_run", True) or not v.get("one_run"))
    return np.array([v["domains"].get(m["arm"], "-") if keep(m) else "-" for m in manifest])


def fit_name(view: str, position: str, train_set: str, layer: int, seed) -> str:
    return f"{view + '__' if view else ''}{position}__{train_set}__L{layer}__{'real' if seed is None else f'null{seed}'}"


def positions_of(meta: dict, step: int, action: int) -> dict:
    """The three readouts of a call in an all-layer meta file (None where missing)."""
    at = {(q["kind"], q["step"], q["action"]): q for q in meta["positions"]}
    last_output = {}
    for q in meta["positions"]:
        if q["kind"] == "output_end":
            last_output[q["step"]] = max(last_output.get(q["step"], -1), q["action"])
    return {"call_end": at.get(("call_end", step, action)), "pre_call": at.get(("pre_call", step, action)),
            "prev_content": at.get(("call_end", step, action - 1)) if action > 0 else
            (at.get(("output_end", step - 1, last_output[step - 1])) if step - 1 in last_output else None)}


def same_readouts(meta: dict, ref: dict, step: int, action: int) -> dict | None:
    """The call's three readouts in `meta` if each is kept and sits on the same (kind, step, action) as in the
    reference read `ref`, else None. A scrubbed text can empty an output (level 4 drops stderr-looking lines), which
    removes its output_end and would leave prev_content missing or moved to an earlier output."""
    qs, rs = positions_of(meta, step, action), positions_of(ref, step, action)
    at = lambda q: q and (q["kind"], q["step"], q["action"])
    return qs if all(qs[p] is not None and qs[p]["kept"] and at(qs[p]) == at(rs[p]) for p in POSITIONS) else None


@app.function(image=cpu_image, volumes={"/acts": acts_vol, "/feats": feats_vol}, cpu=16, memory=65536, timeout=2 * 3600)
def index(cfg: dict, extra: list[dict] | None = None) -> dict:
    """Rows of each eligible call's three positions in the all-layer files, in manifest order. With `extra` (calls of
    the extra arm): index only those, keeping the calls whose three readouts are inside the reader cut, and write the
    view manifest (the attack-only manifest with the view fields, then the kept extra calls)."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    feats_vol.reload()
    base = json.loads((Path("/feats") / cfg["features_run"] / "manifest.json").read_text())
    manifest = base if extra is None else extra
    run = Path("/acts") / (cfg["alllayers_run"] if extra is None else cfg["extra_alllayers_run"])
    want = {(m["arm"], m["key"]) for m in manifest}

    def read_metas(run):
        paths = [p for arm in sorted({a for a, _ in want}) for p in (run / "meta" / arm).glob("*.json")]
        with ThreadPoolExecutor(32) as pool:
            return {(m["arm"], m["key"]): m for m in pool.map(lambda p: json.loads(p.read_text()), paths)}

    metas = read_metas(run)
    ref_run = cfg.get("reference_alllayers_run") if extra is None else None
    if ref_run:  # keep the calls whose readouts sit where they do in the reference read, and write that manifest
        refs = read_metas(Path("/acts") / ref_run)
        missing = sorted(k for k in want if k not in metas or k not in refs)
        if missing:
            raise ValueError(f"{len(missing)} transcripts missing from {run.name} or {ref_run}, e.g. {missing[:3]}")
        keep = [same_readouts(metas[(c["arm"], c["key"])], refs[(c["arm"], c["key"])], c["step"], c["action"]) is not None
                for c in manifest]
        dropped = {a: sum(1 for c, k in zip(manifest, keep) if c["arm"] == a and not k) for a in sorted({a for a, _ in want})}
        manifest = [c for c, k in zip(manifest, keep) if k]
    chunks = sorted({m["chunk"] for m in metas.values()})
    cid = {c: i for i, c in enumerate(chunks)}
    if extra is not None:
        missing = sorted({k for k in want if k not in metas})
        if missing:
            raise ValueError(f"{len(missing)} extra-arm transcripts missing from the all-layer run, e.g. {missing[:3]}")
        qs = [positions_of(metas[(c["arm"], c["key"])], c["step"], c["action"]) for c in manifest]
        kept = [i for i, q in enumerate(qs) if all(q[p] is not None and q[p]["kept"] for p in POSITIONS)]
        dropped = len(manifest) - len(kept)
        manifest = [manifest[i] for i in kept]
    chunk_of, row_of = np.full((3, len(manifest)), -1, np.int32), np.full((3, len(manifest)), -1, np.int64)
    for i, c in enumerate(manifest):
        m = metas[(c["arm"], c["key"])]
        qs = positions_of(m, c["step"], c["action"])
        for k, pos in enumerate(POSITIONS):
            q = qs[pos]
            if q is None or not q["kept"] or (pos != "prev_content" and q["tool"] != c["tool"]):
                raise ValueError(f"{c['arm']} {c['key']} step {c['step']} action {c['action']}: {pos} missing or mismatched in the all-layer run")
            chunk_of[k, i], row_of[k, i] = cid[m["chunk"]], m["row_start"] + q["row"]
    d = out_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "index.npz", chunk_of=chunk_of, row_of=row_of)
    (d / "chunks.json").write_text(json.dumps(chunks))
    report = {"calls": len(manifest), "chunks": len(chunks)}
    if ref_run:
        (d / "manifest.json").write_text(json.dumps(manifest))
        report |= {"reference_alllayers_run": ref_run, "calls_offered": len(base), "dropped_by_arm": dropped}
    if extra is not None:  # the view manifest: attack-only rows first, in their packed order, then the extra arm
        honest, primary = set(cfg["honest_tasks"]), set(cfg["primary_real_honest"])
        slices = {name: set(ts) for name, ts in cfg["slice_tasks"].items()}
        rows = []
        for m in base + manifest:
            honest_arm = m["arm"] in ("real-honest", cfg["extra_arm"])
            rows.append(m | {"in_honest_tasks": m["main_task"] in honest or not honest_arm,
                             "primary_run": m["arm"] != "real-honest" or m["key"] in primary,
                             **{name: m["main_task"] in ts for name, ts in slices.items()}})
        (d / "manifest.json").write_text(json.dumps(rows))
        report |= {"extra_calls_offered": len(extra), "extra_calls_outside_cut": dropped, "base_calls": len(base)}
    feats_vol.commit()
    return report


def view_manifest(cfg) -> list[dict]:
    d = out_dir(cfg)
    own = cfg.get("views") or cfg.get("reference_alllayers_run")  # index wrote this run's own manifest
    return json.loads(((d if own else Path("/feats") / cfg["features_run"]) / "manifest.json").read_text())


def gpu_device() -> dict:
    """The fit container's GPU. Modal's "A10G" pool mixes A10 and A10G cards, and fits match bit for bit only on the same
    card model, so every fit record carries it. Provenance only: not part of the fit hash."""
    import subprocess

    import torch

    props = torch.cuda.get_device_properties(0)
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                            text=True, check=True).stdout.splitlines()[0].strip()
    return {"name": props.name, "sm_count": props.multi_processor_count, "driver": driver}


@app.function(image=cpu_image, volumes={"/acts": acts_vol, "/feats": feats_vol}, cpu=8, memory=65536, timeout=2 * 3600)
def pack(cfg: dict, layer: int) -> int:
    import numpy as np

    feats_vol.reload()
    d = out_dir(cfg)
    idx = np.load(d / "index.npz")
    chunk_of, row_of = idx["chunk_of"], idx["row_of"]
    chunks = json.loads((d / "chunks.json").read_text())
    n = chunk_of.shape[1]
    out = {pos: np.zeros((n, 5120), np.float16) for pos in POSITIONS}
    run = cfg["extra_alllayers_run" if cfg.get("views") else "alllayers_run"]
    src = Path("/acts") / run / "by_layer" / f"L{layer}"
    for c, name in enumerate(chunks):
        sel = np.nonzero(chunk_of == c)
        if not len(sel[0]):
            continue
        arr = np.load(src / f"{name}.npy", mmap_mode="r")
        for k, pos in enumerate(POSITIONS):
            rows = np.flatnonzero(chunk_of[k] == c)
            out[pos][rows] = arr[row_of[k, rows]]
    (d / f"L{layer}").mkdir(exist_ok=True)
    for pos in POSITIONS:
        if not np.isfinite(out[pos]).all():
            raise ValueError(f"non-finite activations at layer {layer}, {pos}")
        np.save(d / f"L{layer}" / f"{pos}.npy", out[pos])
    (d / f"L{layer}" / "pack_source.json").write_text(json.dumps({"activations_run": run, "positions": POSITIONS, "rows": int(n)}))
    feats_vol.commit()
    return layer


@app.function(image=gpu_image, gpu="A10G", volumes={"/feats": feats_vol}, memory=16384, timeout=6 * 3600)
def fit(cfg: dict, view: str, position: str, train_set: str, layers: list[int], seeds: list) -> list[dict]:
    """Per layer and seed (None = real labels): transfer with C chosen per outer fold on validation AUROC."""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from probe_shift.probes import fit_logreg
    from probe_shift.transfer import transfer

    device = gpu_device()  # provenance, not hashed
    feats_vol.reload()
    d = out_dir(cfg)
    manifest = view_manifest(cfg)
    y0 = np.array([m["y"] for m in manifest])
    domain = view_domain(cfg, manifest, view)
    fold = np.array([m["fold"] for m in manifest])
    tkey = np.array([f"{m['arm']}|{m['key']}" for m in manifest])
    test = np.isin(domain, cfg["test_domains"])
    out = []
    for layer in layers:
        parts = []  # this run's own pack; with views, the attack-only rows as packed, then the extra arm's rows
        for pack_dir, run in pack_dirs(cfg):
            check_pack_source(pack_dir / f"L{layer}", run)
            parts.append(np.load(pack_dir / f"L{layer}" / f"{position}.npy"))
        X = torch.from_numpy(np.concatenate(parts)).cuda().float()
        if len(X) != len(manifest):
            raise ValueError(f"layer {layer} {position}: {len(X)} packed rows for {len(manifest)} manifest rows")
        for seed in seeds:
            y = y0.copy()
            if seed is not None:  # the same within-transcript permutation as run_q2_probes.py for this seed
                rng = np.random.default_rng(seed)
                for k in np.unique(tkey):
                    rows = np.flatnonzero(tkey == k)
                    y[rows] = rng.permutation(y0[rows])
            chosen, kept = [], []  # per outer fold: the selection, and the selected probe (real labels only)

            def fit_score(train, score_on):
                val, tst = score_on
                mu, sd = X[train].mean(0), X[train].std(0).clamp_min(1e-6)
                Z = (X - mu) / sd
                ytr = torch.tensor(y[train], dtype=torch.float32, device="cuda")
                best = None
                for C in cfg["C_grid"]:
                    w, g = fit_logreg(Z[train], ytr, C, cfg["max_iter"])
                    s = (Z @ w[:-1] + w[-1]).cpu().numpy()
                    auc = float(roc_auc_score(y[val], s[val]))
                    if best is None or auc > best[0]:
                        best = (auc, C, g, s, w)
                chosen.append({"C": best[1], "val_auroc": best[0], "grad_max": best[2]})
                if seed is None:
                    kept.append([t.cpu().numpy().astype(np.float32) for t in (best[4], mu, sd)])
                return [best[3][val], best[3][tst]]

            r = transfer(fit_score, y, domain, fold, set(cfg["train_sets"][train_set]), fpr=cfg["fpr"])
            name = fit_name(view, position, train_set, layer, seed)
            (d / "fits").mkdir(exist_ok=True)
            np.savez(d / "fits" / f"{name}.npz", score=r["score"][test].astype(np.float32 if seed is None else np.float16),
                     flag=r["flag"][test], threshold=r["threshold"][test].astype(np.float32),
                     **({"y": y[test].astype(np.int8)} if seed is not None else
                        {k: np.stack([f[i] for f in kept]) for i, k in enumerate(("w", "mu", "sd"))}))  # w[-1]: bias
            print(f"fit {view or 'attack'} {position} {train_set} L{layer} {'real' if seed is None else f'null{seed}'}", flush=True)
            out.append({"view": view, "position": position, "train_set": train_set, "layer": layer, "seed": seed, "file": f"{name}.npz",
                        "folds": [f | c for f, c in zip(r["folds"], chosen)]})
            out[-1]["device"] = device  # provenance, not hashed
        del X
        torch.cuda.empty_cache()
    feats_vol.commit()
    return out


@app.function(image=cpu_image, volumes={"/feats": feats_vol}, cpu=32, memory=131072, timeout=6 * 3600)  # schedules faster than 64 CPUs
def summary(cfg: dict, fits: list[dict]) -> dict:
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score

    from probe_shift.transfer import two_by_two

    feats_vol.reload()
    d = out_dir(cfg)
    manifest = view_manifest(cfg)
    layers, tol = cfg["layers"], cfg["layer_tolerance"]
    by_all = {(f.get("view", ""), f["position"], f["train_set"], f["layer"], f["seed"]): f for f in fits}
    load = lambda f: dict(np.load(d / "fits" / f["file"]))
    home = {t: cfg["home"][t] for t in cfg["train_sets"]}

    def one_view(view):
        by = {k[1:]: f for k, f in by_all.items() if k[0] == view}
        dom = view_domain(cfg, manifest, view)
        test = np.isin(dom, cfg["test_domains"])
        rows = [m for m, t in zip(manifest, test) if t]
        y = np.array([m["y"] for m in rows])
        domain = dom[test]
        groups = np.array([m["main_task"] for m in rows])
        fold = np.array([m["fold"] for m in rows])

        def per_layer(pos, layer):
            res = {t: load(by[(pos, t, layer, None)]) for t in cfg["train_sets"]}
            return layer, two_by_two(res, home, y, domain, groups, cfg["test_domains"], cfg["n_boot"], cfg["seed"])

        def headline(pos, t, seed):
            """Per outer fold, the earliest layer within tol of the best validation AUROC; its test rows from that layer."""
            score, flag, thr, picks = np.zeros(len(y)), np.zeros(len(y), bool), np.zeros(len(y)), []
            for f in sorted({fo["fold"] for fo in by[(pos, t, layers[0], seed)]["folds"]}):
                val = {l: next(fo["val_auroc"] for fo in by[(pos, t, l, seed)]["folds"] if fo["fold"] == f) for l in layers}
                top = max(val.values())
                pick = min(l for l in layers if val[l] >= top - tol)
                a = load(by[(pos, t, pick, seed)])
                m = fold == f
                score[m], flag[m], thr[m] = a["score"][m], a["flag"][m], a["threshold"][m]
                picks.append({"fold": f, "layer": pick, "val_auroc": round(val[pick], 4)})
            return {"score": score, "flag": flag, "threshold": thr}, picks, (np.array(load(by[(pos, t, layers[0], seed)])["y"]) if seed is not None else y)

        out = {"positions": {}}
        for pos in POSITIONS:
            curves = dict(Parallel(n_jobs=cfg["n_jobs"])(delayed(per_layer)(pos, l) for l in layers))
            heads, picks = {}, {}
            for t in cfg["train_sets"]:
                heads[t], picks[t], _ = headline(pos, t, None)
            head_rep = two_by_two(heads, home, y, domain, groups, cfg["test_domains"], cfg["n_boot"], cfg["seed"])
            # Sensitivity: the headline probes scored without a slice's main tasks, in every arm (no refit).
            slices = {}
            for name in cfg.get("slices", []):  # the slice's tasks are dropped before resampling tasks
                keep = ~np.array([m[name] for m in rows])
                sub = {t: {k: v[keep] for k, v in h.items()} for t, h in heads.items()}
                slices[f"without {name}"] = two_by_two(sub, home, y[keep], domain[keep], groups[keep], cfg["test_domains"],
                                                       cfg["n_boot"], cfg["seed"])
            seeds = sorted({s for (p, _, _, s) in by if p == pos and s is not None})
            nulls = {}
            def null_aucs(t, s):  # one draw: each saved fit is read once and scored on every test domain
                masks = [domain == te for te in cfg["test_domains"]]
                h, _, ys = headline(pos, t, s) if all((pos, t, l, s) in by for l in layers) else (None, None, None)
                per = {l: load(by[(pos, t, l, s)]) for l in layers if (pos, t, l, s) in by}
                return ([None if h is None else float(roc_auc_score(ys[m], h["score"][m])) for m in masks],
                        {l: [round(float(roc_auc_score(a["y"][m], a["score"][m].astype(np.float32))), 4) for m in masks]
                         for l, a in per.items()})

            for t in cfg["train_sets"]:
                if not any((pos, t, layers[0], s) in by for s in seeds):
                    continue
                draws = Parallel(n_jobs=cfg["n_jobs"])(delayed(null_aucs)(t, s) for s in seeds)
                for i, te in enumerate(cfg["test_domains"]):
                    head_null = [h[i] for h, _ in draws if h[i] is not None]
                    layer_null = {f"L{l}": [per[l][i] for _, per in draws if l in per] for l in layers}
                    nulls[f"train {t} / test {te}"] = {"headline": {"n": len(head_null), "max": round(max(head_null), 4) if head_null else None,
                                                                    "mean": round(float(np.mean(head_null)), 4) if head_null else None},
                                                       "per_layer": layer_null}
            out["positions"][pos] = {"per_layer": {f"L{l}": curves[l] for l in layers}, "headline": head_rep,
                                     "headline_layers": picks, "null": nulls, **({"slices": slices} if slices else {})}
        arms = np.array([m["arm"] for m in rows])
        out["calls_by_arm"] = {te: {a: {"calls": int(((domain == te) & (arms == a)).sum()),
                                        "main_tasks": len(set(groups[(domain == te) & (arms == a)]))}
                                    for a in sorted(set(arms[domain == te]))} for te in cfg["test_domains"]}
        return out

    if not cfg.get("views"):
        return {"layers": layers} | one_view("")
    out = {"layers": layers, "views": {v: one_view(v) for v in views_of(cfg)}}
    rep = cfg.get("reproduces")  # the attack-only view against the attack-only run's own fits: same calls, same scores
    if rep:
        base, check = out_dir(cfg, base=True) / "fits", {}
        for pos in POSITIONS:
            for t, t0 in rep["train_sets"].items():
                diffs = [(np.abs(load(by_all[(rep["view"], pos, t, l, None)])["score"] - np.load(base / f"{fit_name('', pos, t0, l, None)}.npz")["score"]).max(),
                          int((load(by_all[(rep["view"], pos, t, l, None)])["flag"] != np.load(base / f"{fit_name('', pos, t0, l, None)}.npz")["flag"]).sum()))
                         for l in layers if (rep["view"], pos, t, l, None) in by_all and (base / f"{fit_name('', pos, t0, l, None)}.npz").exists()]
                check[f"{pos} {t} vs {t0}"] = {"layers": len(diffs), "max_score_diff": float(max((x for x, _ in diffs), default=np.nan)),
                                               "flags_differ": sum(n for _, n in diffs)}
        out["reproduction"] = check
    return out


@app.function(image=cpu_image, volumes={"/feats": feats_vol}, timeout=12 * 3600)
def orchestrate(cfg: dict, seeds: list, do_index: bool, extra: list[dict] | None = None) -> dict:
    d = out_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    fit_cfg = {k: cfg.get(k) for k in FIT_KEYS}  # later draws and summaries must reuse the fits' own procedure
    saved = d / "fit_config.json"
    old_cfg = json.loads(saved.read_text()) if saved.exists() else {}
    clash = [k for k in FIT_KEYS if old_cfg.get(k) is not None and fit_cfg[k] is not None and old_cfg[k] != fit_cfg[k]]
    unknown = [k for k in FIT_KEYS if saved.exists() and fit_cfg[k] is not None and old_cfg.get(k) is None and
               k not in ("extra_alllayers_run", "views", "honest_tasks", "primary_real_honest")]  # not view-only keys
    if not seeds:  # --summary-only fits nothing, so only the inputs it reads must match; results name the fits' own code
        clash, unknown = [k for k in clash if k != "fit_source_sha"], [k for k in unknown if k != "fit_source_sha"]
        cfg = cfg | {"fit_source_sha": old_cfg.get("fit_source_sha"), "summary_fit_source_sha": cfg["fit_source_sha"]}
    if clash or unknown:  # fits whose code or task set is unknown are not merged with new ones
        raise ValueError(f"{d}: fits were made with other {clash} or unrecorded {unknown}: {old_cfg}, not {fit_cfg}")
    feats_vol.reload()
    prev = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
    report = index.remote(cfg, extra) if do_index else prev.get("index")  # an incremental run keeps the index report
    if do_index:
        list(pack.map([cfg] * len(cfg["layers"]), cfg["layers"]))
    if seeds:  # a summary-only run records nothing about how fits were made
        # recorded once the inputs exist, just before the first fits; keys the saved record lacks are filled in
        saved.write_text(json.dumps({k: old_cfg.get(k) if old_cfg.get(k) is not None else fit_cfg[k] for k in FIT_KEYS}))
    feats_vol.commit()
    chunk = cfg["layers_per_job"]
    groups = [cfg["layers"][i:i + chunk] for i in range(0, len(cfg["layers"]), chunk)]
    null_views = cfg.get("null_views", list(views_of(cfg)))
    per = cfg["null_draws_per_job"]  # a job loads its layers once, then fits each of its seeds
    jobs = [(cfg, v, pos, t, g, ss[i:i + per]) for v in views_of(cfg) for pos in POSITIONS for t in cfg["train_sets"] for g in groups
            for ss in [[s for s in seeds if s is None or (t in cfg["null_train_sets"] and v in null_views)]] for i in range(0, len(ss), per)]
    fits = [f for part in fit.starmap(jobs) for f in part] if jobs else []  # --summary-only: nothing to fit
    feats_vol.reload()
    shards = d / "fits_index"  # one index file per launch, so concurrent draw ranges never overwrite each other
    shards.mkdir(exist_ok=True)
    (shards / f"{cfg['launch_id']}.jsonl").write_text("".join(json.dumps(f) + "\n" for f in fits))
    feats_vol.commit()
    logs = [d / "fits_index.jsonl"] + sorted(shards.glob("*.jsonl"))  # the single file older runs wrote, then shards
    old = [json.loads(l) for log in logs if log.exists() for l in log.read_text().splitlines()]
    keyed = {(f.get("view", ""), f["position"], f["train_set"], f["layer"], f["seed"]): f for f in old}
    out = summary.remote(cfg, list(keyed.values())) | {"index": report, "config": cfg}  # --fetch reports this config
    (d / "result.json").write_text(json.dumps(out))
    feats_vol.commit()
    return out


def view_inputs(cfg: dict, labels: Path) -> list[dict]:
    """For a config with views: the calls to index from the extra run, and, in cfg, the honest tasks, each task's
    chosen real honest run and the slices' main tasks. Honest tasks: every main task with both a real and a simulated
    honest run (the paired honest tasks of the shared splits). The extra calls are the simulated honest ones, plus the
    real honest calls of honest tasks without an attack pair, which the attack-only features run does not hold.
    Chosen run (fixed before any result): lowest epoch, then lowest sample ID, then lowest .eval file name."""
    if cfg["label_rule"] != "as_labeled":
        raise ValueError("views support label_rule as_labeled only")
    folds = json.loads((ROOT / "splits" / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    paired = {r["main_task"] for r in rows if r[cfg["tasks"]]}  # tasks whose real honest calls are already packed
    honest = sorted({r["main_task"] for r in rows if r["arm"] == "real-honest"} & {r["main_task"] for r in rows if r["arm"] == cfg["extra_arm"]})
    runs = {}
    for r in rows:
        if r["arm"] == "real-honest" and r["main_task"] in honest:
            file, sample_id, epoch = r["key"].rsplit("__", 2)
            runs.setdefault(r["main_task"], set()).add((int(epoch), sample_id, file, r["key"]))
    meta = json.loads(labels.with_suffix(".meta.json").read_text())
    cfg |= {"honest_tasks": honest, "primary_real_honest": sorted(min(v)[3] for v in runs.values()),
            "slice_tasks": {name: meta["slices"][name] for name in cfg["slices"]}}
    return [{k: r[k] for k in ("arm", "key", "cell", "step", "action", "tool", "category", "main_task")} |
            {"fold": folds[r["main_task"]], "y": int(r["label"] == "impactful")}
            for r in rows if r["main_task"] in honest and (r["arm"] == cfg["extra_arm"] or (r["arm"] == "real-honest" and
                                                                                             r["main_task"] not in paired))]


@app.local_entrypoint()
def main(draws: str = "", fetch: bool = False, skip_index: bool = False, layers: str = "", summary_only: bool = False,
         config: str = ""):
    """--draws a-b: fit shuffle draws a..b (the real labels are fit on the first call, without --draws). --fetch: read
    the saved result instead of computing. --layers 16,40: a pilot on those layers only. --summary-only: recompute the
    summary from every fit saved so far (after more draws land). --config q2_threeway: merge that config on top (views)."""
    import yaml

    from probe_shift import provenance
    from probe_shift.labels import resolve_labels

    cfg = yaml.safe_load((HERE / "config" / "q2_probes.yaml").read_text()) | \
        yaml.safe_load((HERE / "config" / "q2_alllayers.yaml").read_text())
    if config:
        cfg |= yaml.safe_load((HERE / "config" / f"{config}.yaml").read_text())
    labels, made_by = resolve_labels(ROOT, cfg["labels_file"])
    cfg["labels_file"], cfg["labeler_sha"] = str(labels.relative_to(ROOT)), made_by
    cfg["features_run"] += f"_{made_by}_{cfg['label_rule']}"  # the run_q2_probes.py features run for these labels
    cfg["provenance"] = provenance.capture(ROOT, [Path(__file__)] + [ROOT / "src" / "probe_shift" / f for f in
                                                   ("probes.py", "transfer.py", "labels.py")])  # at launch
    if cfg.get("views") and not cfg.get("extra_alllayers_run", "").startswith("20"):
        raise ValueError(f"set extra_alllayers_run in config/{config}.yaml (now {cfg.get('extra_alllayers_run')!r})")
    cfg["launch_id"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S_") + secrets.token_hex(3)
    src = Path(__file__).read_text()  # the fitting code: fit, the helpers it calls, probes.py and transfer.py
    fit_src = "".join(re.search(rf"(?ms)^def {f}\(.*?(?=^\S)", src).group(0) for f in
                      ("out_dir", "pack_dirs", "check_pack_source", "views_of", "view_domain", "fit_name", "view_manifest", "fit"))
    fit_src = re.sub(r"(?m)^.*# provenance, not hashed\n", "", fit_src)  # recording the device does not change a fit
    cfg["fit_source_sha"] = hashlib.sha256((fit_src + "".join((ROOT / "src" / "probe_shift" / f).read_text()
                                                              for f in ("probes.py", "transfer.py"))).encode()).hexdigest()[:16]
    folds = json.loads((ROOT / "splits" / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    tasks = sorted({r["main_task"] for r in map(json.loads, labels.read_text().splitlines()) if r[cfg["tasks"]]})
    cfg["task_set_sha"] = hashlib.sha256(json.dumps({t: folds[t] for t in tasks}).encode()).hexdigest()[:16]  # as the probes
    extra = view_inputs(cfg, labels) if cfg.get("views") else None
    if layers:  # a pilot never touches the full run's fits or result
        cfg["layers"] = [int(l) for l in layers.split(",")]
        cfg["out_sub"] = cfg.get("out_sub", "alllayers") + "_pilot"
    seeds = [None] if not draws else []
    if draws:
        a, b = map(int, draws.split("-"))
        seeds = list(range(cfg["seed"] + a, cfg["seed"] + b + 1))
    if fetch:  # the saved result's own config and provenance, not this process's
        out = json.loads(b"".join(feats_vol.read_file(f"{cfg['features_run']}/{cfg.get('out_sub', 'alllayers')}/result.json")))
        cfg = out.get("config", cfg)
    elif summary_only:
        out = orchestrate.remote(cfg, [], False)
    else:
        out = orchestrate.remote(cfg, seeds, not skip_index and not draws, extra)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"{config or 'q2_alllayers'}_{stamp}{'_pilot' if layers else ''}.json"
    cfg = out.pop("config", cfg)
    path.write_text(json.dumps({"config": cfg | {"modal": {"app": APP}}, "labeler_sha": cfg["labeler_sha"], **out}, indent=1))
    for view, v in (out["views"] if "views" in out else {"": out}).items():
        t0 = list(cfg["train_sets"])[0]
        for pos, e in v["positions"].items():
            h = e["headline"]
            print(f"\n=== {view} {pos}: headline layers {[p['layer'] for p in e['headline_layers'][t0]]}  " + "  ".join(
                f"{k.removeprefix('train ')}: AUROC {h[k]['auroc']['value']:.3f} recall {h[k]['recall']['value']:.3f}"
                for k in h if k.startswith("train ") and " minus " not in k))
    if "reproduction" in out:
        print("\nattack-only view vs the attack-only run:", json.dumps(out["reproduction"]))
    print(f"\nresults -> {path.relative_to(ROOT)}")
