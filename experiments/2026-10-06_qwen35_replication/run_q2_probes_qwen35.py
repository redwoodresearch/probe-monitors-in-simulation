"""Q2 probes: does a probe that flags impactful calls, trained on one domain's activations, work on the other?

The qwen35 track's copy of experiments/2026-10-06_q2_impactful_call_transfer/run_q2_probes.py (PR #16 at e9c3b9e), for
Qwen3.5-27B as the reader. Changes: this track's app and Volumes; activations from the all-layers extraction's per-layer
chunk files (run_extract_qwen35.py) instead of per-transcript .npz; `--config`; and `match_cut_of`, which keeps only the
calls whose three positions the Qwen3-32B reader also read inside its 32k cut, for a like-for-like comparison (whole
transcripts otherwise). For all 64 layers (Aswin, 2026-10-06): `features` decides eligibility from metadata alone and
`features_layer` writes one file per position and layer (`<position>_L<layer>.npy`, one container per layer); `fit`
runs on an H200 with every layer on the GPU and also saves each layer's out-of-fold scores, from which `summary` draws
per-layer AUROC curves for every cell of the 2x2. The original docstring follows, written for Qwen3-32B.

Reads the Q2 labels and the main session's segment-end run (Qwen3-32B residual stream, layers 8..63). Three
positions per call:
  - call_end (readout a): the call's last content token. It sees the command and everything before it, not the
    output.
  - pre_call (readout b): the `<tool_call>` token opening the call. It sees only what came before the command.
  - prev_content: the last content token before the call, i.e. the previous call's call_end in the same turn, else
    the previous turn's last output_end. Same context as pre_call without template tokens. pre_call is a
    byte-identical template slot, where the BashArena end-of-sequence artifact could reappear, so it is reported
    only next to this.
A call is eligible if all three positions sit inside the 32k cut. That leaves out first calls, calls right after an
empty output, and calls past the cut. A task is kept only if both attack arms' transcripts were read. The
eligible-call manifest is saved so the text baselines score exactly these calls.

Steps, all on Modal:
  1. features (CPU): read each transcript's layer arrays once, write [layer, call, hidden] fp16 per position to the
     Q2 Volume, plus the manifest and an exclusion report with reasons.
  2. fit (GPU, one container per position and training set): probe_shift.transfer with torch L2 logistic regression
     (L-BFGS, sklearn's objective). Inside each outer fold, the layer and C are chosen on that fold's
     training-domain validation AUROC (earliest layer within `layer_tolerance` of the best), so no test task
     influences the choice.
  3. null (GPU): the same procedure, selection included, with labels permuted within each transcript.
  4. summary (CPU): the 2x2 with intervals (all calls; without unseen-effect calls; without text_editor; by
     category), the layer choices, and the null.

  uv run modal run experiments/2026-10-06_qwen35_replication/run_q2_probes_qwen35.py --config q2_probes_qwen35.yaml [--limit-tasks N]
"""

import gzip
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-qwen35_q2_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06"  # this track's Qwen3.5 extraction; read only
QWEN3_ACTS = "probe-shift_aswin-surya_sub-la-activations"  # the main session's Volume (Qwen3-32B runs); read only
FEATS = "probe-shift_aswin-surya_sub-qwen35_q2feats_kill-2026-11-06"
POSITIONS = ["call_end", "pre_call", "prev_content"]

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2).read_only()
qwen3_vol = modal.Volume.from_name(QWEN3_ACTS, version=2).read_only()
feats_vol = modal.Volume.from_name(FEATS, create_if_missing=True, version=2)

def wait(call):
    """A spawned call's result, retrying our network for up to an hour (a laptop DNS blip must not kill a run); remote
    errors are raised. As in the Q1 GPU analysis, plus grpclib's StreamTerminatedError, which ended a Q2 run here."""
    import socket
    import time

    from grpclib.exceptions import StreamTerminatedError

    last = None
    for _ in range(120):
        try:
            return call.get()
        except (socket.gaierror, ConnectionError, TimeoutError, StreamTerminatedError, modal.exception.ConnectionError,
                modal.exception.ServiceError) as e:  # a transport error here, or the same type raised by the remote call
            last = e
            print(f"lost the connection ({e!r}); retrying in 30 s", flush=True)
            time.sleep(30)
    raise TimeoutError(f"an hour of {last!r} from {call.object_id}: our connection, or a remote failure of that type; "
                       "check the call on Modal (finished pieces are on the Volume)") from last

cpu_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2")
             .add_local_python_source("probe_shift"))
gpu_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2", "torch==2.8.0")
             .add_local_python_source("probe_shift"))


def check_features(cfg: dict, d: Path):
    """Refuse a cached features run built with other layers, positions, labels or label rule. (Runs before the
    rename record `http_rule`, which then reads as a mismatch: rebuild them.)"""
    meta = json.loads((d / "features_meta.json").read_text())
    want = {"layers": cfg["layers"], "positions": POSITIONS, "labeler_sha": cfg["labeler_sha"], "label_rule": cfg["label_rule"],
            "activations_run": cfg["activations_run"], "tasks": cfg["tasks"], "folds_file": cfg["folds_file"],
            "task_set_sha": cfg["task_set_sha"], "match_cut_of": cfg["match_cut_of"]}
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise ValueError(f"features run {d.name} does not match the config (cached, wanted): {bad}")


@app.function(image=cpu_image, volumes={"/acts": acts_vol, "/qwen3": qwen3_vol, "/feats": feats_vol}, cpu=32, memory=65536,
              timeout=3 * 3600)
def features(cfg: dict, calls_gz: bytes) -> dict:
    """Which calls are eligible, from transcript metadata alone. Writes the manifest, and `plan.json`: per eligible call,
    its transcript's chunk and the chunk row of each position, which features_layer copies one layer at a time."""
    from collections import Counter, defaultdict
    from concurrent.futures import ThreadPoolExecutor

    calls = json.loads(gzip.decompress(calls_gz))
    layers = cfg["layers"]
    run_dir = Path("/acts") / cfg["activations_run"]
    out_dir = Path("/feats") / cfg["features_run"]
    out_dir.mkdir(parents=True, exist_ok=True)
    index = defaultdict(dict)  # (arm, key) -> {(step, action): call index}
    for i, c in enumerate(calls):
        index[(c["arm"], c["key"])][(c["step"], c["action"])] = i

    def read_metas(arm_dir):
        paths = [p for arm in sorted({c["arm"] for c in calls}) for p in (arm_dir / arm).glob("*.json")]
        with ThreadPoolExecutor(32) as pool:  # many small files: read in parallel
            ms = list(pool.map(lambda p: json.loads(p.read_text()), paths))
        return {(m["arm"], m["key"]): (p, m) for p, m in zip(paths, ms)}

    def position_of(meta):
        at = {(q["kind"], q["step"], q["action"]): q for q in meta["positions"]}
        last_output = {}
        for q in meta["positions"]:
            if q["kind"] == "output_end":
                last_output[q["step"]] = max(last_output.get(q["step"], -1), q["action"])

        def position(kind, step, action):
            if kind != "prev_content":
                return at.get((kind, step, action))
            if action > 0:
                return at.get(("call_end", step, action - 1))
            return at.get(("output_end", step - 1, last_output[step - 1])) if step - 1 in last_output else None
        return position

    metas = read_metas(run_dir / "meta")  # all-layers runs keep transcript metadata under meta/<arm>/
    missing = sorted(k for k in index if k not in metas)
    qwen3_metas = read_metas(Path("/qwen3") / cfg["match_cut_of"] / "acts") if cfg["match_cut_of"] else None
    if qwen3_metas is not None and not qwen3_metas:
        raise ValueError(f"no transcript metadata in {cfg['match_cut_of']} on {QWEN3_ACTS}")

    def load(arm_key):
        p, meta = metas[arm_key]
        mine = index[arm_key]
        position = position_of(meta)
        qwen3_position = None
        if qwen3_metas is not None and arm_key in qwen3_metas:
            qwen3_position = position_of(qwen3_metas[arm_key][1])

        rows, reasons = [], []
        for (step, action), i in mine.items():
            qs = {kind: position(kind, step, action) for kind in POSITIONS}
            for kind in ("call_end", "pre_call"):
                if qs[kind] is not None and qs[kind]["tool"] != calls[i]["tool"]:  # both pipelines must agree on the call
                    raise ValueError(f"tool mismatch {arm_key} step {step} action {action}: {qs[kind]['tool']} vs {calls[i]['tool']}")
            if qs["prev_content"] is None:
                reasons.append((i, "no previous content token (first call, or after an empty output)"))
            elif not all(q["kept"] for q in qs.values() if q is not None) or any(q is None for q in qs.values()):
                reasons.append((i, "a position with no token (empty after the template's trim) or missing"))  # no cut here
            elif qwen3_metas is not None and (qwen3_position is None or not all(
                    (q := qwen3_position(kind, step, action)) is not None and q["kept"] for kind in POSITIONS)):
                reasons.append((i, "outside what the Qwen3-32B reader read (its 32k cut, or not in its run)"))
            else:  # the chunk, and each position's row in it (metadata rows are relative to the transcript's block)
                rows.append((i, [meta["chunk"], [meta["row_start"] + qs[k]["row"] for k in POSITIONS]]))
        return rows, reasons

    with ThreadPoolExecutor(32) as pool:
        results = list(pool.map(load, [k for k in index if k in metas]))
    got = {i: x for rows, _ in results for i, x in rows}
    reasons = {i: r for _, rs in results for i, r in rs}
    for arm_key in missing:
        for i in index[arm_key].values():
            reasons[i] = "transcript not in the activation run"
    # Paired 2x2: keep a task's attack calls only if both attack arms' transcripts were read (by both readers, when
    # matching Qwen3-32B's cut).
    cell_ok = defaultdict(set)
    for i, c in enumerate(calls):
        if (c["arm"], c["key"]) in metas and (qwen3_metas is None or (c["arm"], c["key"]) in qwen3_metas):
            cell_ok[c["cell"]].add(c["arm"])
    eligible = []
    for i, c in enumerate(calls):
        if i not in got:
            continue
        if c["arm"] in cfg["test_domains"] and not set(cfg["test_domains"]) <= cell_ok[c["cell"]]:
            reasons[i] = "task's other attack arm not read"
            continue
        eligible.append(i)
    n = len(eligible)
    manifest = [calls[i] | {"row": r} for r, i in enumerate(eligible)]
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    (out_dir / "plan.json").write_text(json.dumps([got[i] for i in eligible]))
    report = {"calls_sent": len(calls), "eligible": n, "by_arm": dict(Counter(calls[i]["arm"] for i in eligible)),
              "excluded": dict(Counter(reasons.values())), "missing_transcripts": [list(k) for k in missing]}
    (out_dir / "report.json").write_text(json.dumps(report))
    (out_dir / "features_meta.json").write_text(json.dumps(  # what the arrays hold; fit and summary check it
        {"layers": layers, "positions": POSITIONS, "labeler_sha": cfg["labeler_sha"], "label_rule": cfg["label_rule"],
         "tasks": cfg["tasks"], "folds_file": cfg["folds_file"], "task_set_sha": cfg["task_set_sha"],
         "activations_run": cfg["activations_run"], "match_cut_of": cfg["match_cut_of"], "n_calls": n}))
    feats_vol.commit()
    return report


@app.function(image=cpu_image, volumes={"/acts": acts_vol, "/feats": feats_vol}, cpu=8, memory=32768, timeout=3 * 3600)
def features_layer(cfg: dict, layer: int) -> dict:
    """One layer's activations at every eligible call and position, from `plan.json`: `<position>_L<layer>.npy`,
    [call, hidden] fp16, rows in manifest order. Each file is written under a temporary name and renamed."""
    import os
    from collections import defaultdict

    import numpy as np

    feats_vol.reload()
    d = Path("/feats") / cfg["features_run"]
    plan = json.loads((d / "plan.json").read_text())
    by_chunk = defaultdict(list)
    for r, (chunk, rows) in enumerate(plan):
        by_chunk[chunk].append((r, rows))
    src = Path("/acts") / cfg["activations_run"] / "by_layer" / f"L{layer}"
    out = None
    for chunk, items in sorted(by_chunk.items()):
        x = np.load(src / f"{chunk}.npy", mmap_mode="r")
        if out is None:
            out = np.zeros((len(POSITIONS), len(plan), x.shape[1]), np.float16)
        rows = np.array([rs for _, rs in items])  # [calls in this chunk, position]
        out[:, [r for r, _ in items]] = np.asarray(x[rows.ravel()]).reshape(len(items), len(POSITIONS), -1).transpose(1, 0, 2)
    if not np.isfinite(out).all():  # fail loudly rather than feed NaN or Inf to the fits
        raise ValueError(f"non-finite activations at layer {layer}: {int((~np.isfinite(out)).sum())} values")
    for pi, kind in enumerate(POSITIONS):
        tmp = d / f".{kind}_L{layer}.tmp.npy"
        np.save(tmp, out[pi])
        os.replace(tmp, d / f"{kind}_L{layer}.npy")
    feats_vol.commit()
    return {"layer": layer, "calls": len(plan), "hidden": int(out.shape[-1])}


def select_with_curves(X, y, train, val, layers: list[int], grid: list[float], tol: float, max_iter: int):
    """probe_shift.probes.select (PR #16), copied so the shared function stays as it is, that also returns each layer's
    scores for every call ([layer, call], that layer's best C on `val`): the per-layer curves."""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from probe_shift.probes import fit_logreg

    ytr = torch.tensor(y[train], dtype=torch.float32, device=X.device)
    best = {}
    for li in range(len(layers)):
        Xl = X[li].float()
        mu, sd = Xl[train].mean(0), Xl[train].std(0).clamp_min(1e-6)
        Z = (Xl - mu) / sd
        for C in grid:
            w, g = fit_logreg(Z[train], ytr, C, max_iter)
            s = (Z @ w[:-1] + w[-1]).cpu().numpy()
            auc = float(roc_auc_score(y[val], s[val]))
            if li not in best or auc > best[li]["val_auroc"]:
                best[li] = {"li": li, "layer": layers[li], "C": C, "val_auroc": auc, "grad_max": g, "scores": s,
                            "w": w, "mean": mu, "sd": sd}
        del Xl, Z
    top = max(b["val_auroc"] for b in best.values())
    chosen = best[min(l for l, b in best.items() if b["val_auroc"] >= top - tol)]
    per_layer = np.stack([best[li]["scores"] for li in range(len(layers))]).astype(np.float32)
    return chosen | {"layer_curve": {f"L{layers[l]}": round(b["val_auroc"], 4) for l, b in sorted(best.items())}}, per_layer


@app.function(image=gpu_image, gpu="H200", volumes={"/feats": feats_vol}, memory=65536, timeout=8 * 3600)
def fit(cfg: dict, position: str, train_set: str, shuffle_seeds: list[int]) -> list[dict]:
    """Fits for one position and training set: the real labels when shuffle_seeds is [None], else one fit per seed
    with labels permuted within each transcript."""
    import numpy as np
    import torch

    from probe_shift.transfer import transfer

    feats_vol.reload()
    d = Path("/feats") / cfg["features_run"]
    manifest = json.loads((d / "manifest.json").read_text())
    check_features(cfg, d)
    first = np.load(d / f"{position}_L{cfg['layers'][0]}.npy", mmap_mode="r")
    if first.shape[0] != len(manifest):
        raise ValueError(f"{position}: {first.shape[0]} feature rows for {len(manifest)} manifest calls")
    X = torch.empty((len(cfg["layers"]), *first.shape), dtype=torch.float16, device="cuda")  # [layer, call, hidden]
    for li, l in enumerate(cfg["layers"]):  # one layer at a time through host memory
        X[li] = torch.from_numpy(np.load(d / f"{position}_L{l}.npy"))
    y0 = np.array([m["y"] for m in manifest])
    domain = np.array([m["arm"] for m in manifest])
    fold = np.array([m["fold"] for m in manifest])
    tkey = np.array([f"{m['arm']}|{m['key']}" for m in manifest])
    layers, grid, tol = cfg["layers"], cfg["C_grid"], cfg["layer_tolerance"]

    out = []
    for seed in shuffle_seeds:
        y = y0.copy()
        if seed is not None:
            rng = np.random.default_rng(seed)
            for k in np.unique(tkey):
                rows = np.flatnonzero(tkey == k)
                y[rows] = rng.permutation(y0[rows])
        choices, weights = [], []
        layer_scores = np.full((len(layers), len(y)), np.nan, np.float32)  # each layer's out-of-fold scores

        def fit_score(train, score_on):
            val, test = score_on
            ch, per_layer = select_with_curves(X, y, train, val, layers, grid, tol, cfg["max_iter"])  # chosen on validation
            layer_scores[:, test] = per_layer[:, test]
            choices.append({k: ch[k] for k in ("layer", "C", "val_auroc", "grad_max", "layer_curve")})
            weights.append({"coef": ch["w"][:-1].cpu().numpy(), "intercept": float(ch["w"][-1]), "mean": ch["mean"].cpu().numpy(),
                            "sd": ch["sd"].cpu().numpy(), "layer": ch["layer"], "C": ch["C"]})
            return [ch["scores"][val], ch["scores"][test]]

        r = transfer(fit_score, y, domain, fold, set(cfg["train_sets"][train_set]), fpr=cfg["fpr"])
        name = f"{position}__{train_set}__{'real' if seed is None else f'null{seed}'}"
        (d / "fits").mkdir(exist_ok=True)
        np.savez(d / "fits" / f"{name}.npz", y=y, score=r["score"], flag=r["flag"], threshold=r["threshold"])
        if seed is None:  # the trained probes, one per outer fold: score = ((x - mean) / sd) . coef + intercept on `layer`
            np.savez(d / "fits" / f"{name}_probes.npz", **{k: np.stack([wt[k] for wt in weights]) for k in weights[0]},
                     fold=np.arange(len(weights)), threshold=np.array([f["threshold"] for f in r["folds"]]))
            np.savez(d / "fits" / f"{name}_layers.npz", score=layer_scores, layers=np.array(layers))
        out.append({"position": position, "train_set": train_set, "seed": seed, "file": f"{name}.npz",
                    "layers_file": f"{name}_layers.npz" if seed is None else None,
                    "folds": [f | c for f, c in zip(r["folds"], choices)]})
    feats_vol.commit()
    return out


@app.function(image=cpu_image, volumes={"/feats": feats_vol}, cpu=16, memory=65536, timeout=3600)
def summary(cfg: dict, fits: list[dict]) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from probe_shift.calls import CLEAR_IMPACTFUL, READ_ONLY, UNCLEAR
    from probe_shift.transfer import two_by_two

    feats_vol.reload()
    check_features(cfg, Path("/feats") / cfg["features_run"])
    manifest = json.loads((Path("/feats") / cfg["features_run"] / "manifest.json").read_text())
    y = np.array([m["y"] for m in manifest])
    domain = np.array([m["arm"] for m in manifest])
    groups = np.array([m["main_task"] for m in manifest])
    category = np.array([m["category"] for m in manifest])
    editor = np.array([m["tool"] == "text_editor" for m in manifest])
    tests, home = cfg["test_domains"], cfg["home"]
    fit_dir = Path("/feats") / cfg["features_run"] / "fits"
    arrays = {f["file"]: dict(np.load(fit_dir / f["file"])) for f in fits}
    test_rows = np.flatnonzero(np.isin(domain, tests))
    out = {"positions": {}, "predictions": [], "calls": [{k: manifest[i][k] for k in ("arm", "key", "step", "action", "tool", "category", "fold", "y")}
                                                         for i in test_rows]}
    for f in fits:
        if f["seed"] is None:
            a = arrays[f["file"]]
            out["predictions"].append({"position": f["position"], "train_set": f["train_set"],
                                       **{k: a[k][test_rows].tolist() for k in ("score", "flag", "threshold")}})
    for pos in POSITIONS:
        real = {f["train_set"]: {k: arrays[f["file"]][k] for k in ("score", "flag", "threshold")} | {"folds": f["folds"]}
                for f in fits if f["position"] == pos and f["seed"] is None}
        if not real:
            continue
        rep = lambda n_boot, mask=None: two_by_two(real, {t: home[t] for t in real}, y, domain, groups, tests, n_boot, cfg["seed"], mask)
        entry = {"folds": {t: r["folds"] for t, r in real.items()},
                 "all_calls": rep(cfg["n_boot"]),
                 "robustness_no_unseen_effects": rep(cfg["n_boot"], ~np.isin(category, UNCLEAR)),
                 "no_text_editor": rep(cfg["n_boot"], ~editor),
                 "by_category": {c: rep(cfg["n_boot_category"], category == c) for c in CLEAR_IMPACTFUL + UNCLEAR + READ_ONLY
                                 if (category == c).any()}}
        curves = {}
        for f in fits:  # per-layer AUROC on each test domain, from each layer's out-of-fold scores (for interpretation)
            if f["position"] == pos and f["seed"] is None:
                S = np.load(fit_dir / f["layers_file"])["score"]
                for subset, keep in (("all_calls", np.ones(len(y), bool)), ("robustness_no_unseen_effects", ~np.isin(category, UNCLEAR))):
                    for te in tests:
                        m = (domain == te) & keep
                        curves.setdefault(subset, {})[f"train {f['train_set']} / test {te}"] = {
                            f"L{l}": round(float(roc_auc_score(y[m], S[li, m])), 4) for li, l in enumerate(cfg["layers"])}
        entry["layer_curves"] = curves
        for t in real:
            nulls = [f for f in fits if f["position"] == pos and f["train_set"] == t and f["seed"] is not None]
            for te in tests:
                m = domain == te
                draws = np.array([roc_auc_score(arrays[f["file"]]["y"][m], arrays[f["file"]]["score"][m]) for f in nulls])
                if len(draws):
                    obs = entry["all_calls"][f"train {t} / test {te}"]["auroc"]["value"]
                    entry["all_calls"][f"train {t} / test {te}"]["label_shuffle_null"] = {
                        "n": len(draws), "mean": round(float(draws.mean()), 4), "max": round(float(draws.max()), 4),
                        "p_value": round(float((1 + (draws >= obs).sum()) / (1 + len(draws))), 4)}
        out["positions"][pos] = entry
    return out


def task_set_sha(tasks, folds) -> str:
    """Fingerprint of the analyzed main tasks and their folds."""
    import hashlib

    return hashlib.sha256(json.dumps({t: folds[t] for t in sorted(tasks)}).encode()).hexdigest()[:16]


def source_sha() -> str:
    """Fingerprint of the code that produced a result: this script and the probe_shift modules it uses."""
    import hashlib

    files = [Path(__file__)] + [ROOT / "src" / "probe_shift" / f for f in ("probes.py", "transfer.py", "calls.py")]
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16]


@app.local_entrypoint()
def main(config: str = "q2_probes_qwen35.yaml", limit_tasks: int = 0, skip_features: bool = False):
    """--limit-tasks N: pilot on the first N main tasks (features, one fit per position for the first training set,
    two null draws), written under a separate features run."""
    import yaml

    from probe_shift.calls import IMPACTFUL
    from probe_shift.labels import resolve_labels

    cfg = yaml.safe_load((HERE / "config" / config).read_text())
    if cfg["layers"] == "all":
        cfg["layers"] = list(range(64))
    labels, made_by = resolve_labels(ROOT, cfg["labels_file"])  # `latest`: newest table from the current labeler
    cfg["labels_file"] = str(labels.relative_to(ROOT))
    folds = json.loads((ROOT / "splits" / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    tasks = sorted({r["main_task"] for r in rows if r[cfg["tasks"]]})
    cfg["labeler_sha"] = made_by
    cfg["features_run"] += f"_{made_by}_{cfg['label_rule']}"  # features and labels never mix across labelers
    if limit_tasks:  # a pilot: the first tasks of every fold, so each outer fold has training, validation and test tasks
        by_fold = {}
        for t in tasks:
            by_fold.setdefault(folds[t], []).append(t)
        per = -(-limit_tasks // len(by_fold))
        tasks = sorted(t for ts in by_fold.values() for t in ts[:per])
        if len({folds[t] for t in tasks}) != len(by_fold):
            raise ValueError("the pilot does not cover every fold")
        cfg["features_run"] += f"_pilot{limit_tasks}"
    tasks = set(tasks)
    cfg["task_set_sha"] = task_set_sha(tasks, folds)  # the cohort and its folds; cached features must match

    def y_of(r):
        """cfg label_rule: as_labeled (every network call impactful; Aswin, 2026-10-06), or the robustness rule
        localhost_reads_read_only (a call impactful only through localhost GET/HEAD sending no data is read-only)."""
        if cfg["label_rule"] == "localhost_reads_read_only" and {c for c in r["categories"] if c in IMPACTFUL} == {"network"} \
                and r["net_method"] in ("GET", "HEAD") and r["net_dest"] == "localhost" and not r["net_sends_data"]:
            return 0
        if cfg["label_rule"] not in ("as_labeled", "localhost_reads_read_only"):
            raise ValueError(f"unknown label_rule {cfg['label_rule']}")
        return int(r["label"] == "impactful")

    arms = set(cfg["test_domains"]) | {a for s in cfg["train_sets"].values() for a in s}
    calls = [{k: r[k] for k in ("arm", "key", "cell", "step", "action", "tool", "category", "main_task")} |
             {"fold": folds[r["main_task"]], "y": y_of(r)}
             for r in rows if r["arm"] in arms and r["main_task"] in tasks and (r[cfg["tasks"]] or r["arm"] not in cfg["test_domains"])]
    print(f"{len(calls)} labeled calls ({cfg['tasks']} attack tasks; honest calls of the same main tasks) -> {APP}", flush=True)
    if not skip_features:
        report = wait(features.spawn(cfg, gzip.compress(json.dumps(calls).encode())))
        print("features:", json.dumps(report)[:2000], flush=True)
        done = [wait(c) for c in [features_layer.spawn(cfg, l) for l in cfg["layers"]]]
        if sorted(x["layer"] for x in done) != sorted(cfg["layers"]) or {x["calls"] for x in done} != {report["eligible"]}:
            raise ValueError(f"per-layer features incomplete: {done}")
        print(f"features: {len(done)} layers x {report['eligible']} calls, hidden {done[0]['hidden']}", flush=True)

    train_sets = list(cfg["train_sets"])[:1] if limit_tasks else list(cfg["train_sets"])
    n_null = 2 if limit_tasks else cfg["shuffle_null"]
    jobs = [(cfg, pos, t, [None]) for pos in POSITIONS for t in train_sets]
    per = cfg["null_draws_per_job"]
    jobs += [(cfg, pos, t, list(range(cfg["seed"] + 1 + s, cfg["seed"] + 1 + min(s + per, n_null))))
             for pos in POSITIONS for t in cfg["null_train_sets"] if t in train_sets for s in range(0, n_null, per)]
    fits = [f for c in [fit.spawn(*j) for j in jobs] for f in wait(c)]
    out = wait(summary.spawn(cfg, fits))

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    tag = f"_{Path(config).stem}" + (f"_pilot{limit_tasks}" if limit_tasks else "")
    pred_path = HERE / "results" / f"q2_probe_predictions_{stamp}{tag}.jsonl.gz"
    ids = out.pop("calls")
    with gzip.open(pred_path, "wt") as f:  # per test call and probe: score, flag at the frozen threshold, threshold
        for p in out.pop("predictions"):
            for c, s, fl, t in zip(ids, p["score"], p["flag"], p["threshold"]):
                f.write(json.dumps({"position": p["position"], "train_set": p["train_set"], **c, "score": s, "flag": fl, "threshold": t}) + "\n")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    cfg["provenance"] = {"git_dirty": dirty, "source_sha": source_sha()}  # the exact code, even when uncommitted
    path = HERE / "results" / f"q2_probes_{stamp}{tag}.json"
    path.write_text(json.dumps({"config": cfg | {"modal": {"app": APP, "activations_volume": ACTS, "qwen3_volume": QWEN3_ACTS, "features_volume": FEATS}},
                                "git": git, "labeler_sha": made_by, "predictions_file": pred_path.name, **out}, indent=1))
    for pos, e in out["positions"].items():
        print(f"\n=== {pos}")
        for t, fl in e["folds"].items():
            print(f"  {t}: chosen per fold", [(c["layer"], c["C"], round(c["val_auroc"], 3)) for c in fl],
                  "max |grad|", max(c["grad_max"] for c in fl))
        for subset in ("all_calls", "robustness_no_unseen_effects"):
            print(f"  [{subset}]")
            for cell, m in e[subset].items():
                fmt = lambda k: f"{m[k]['value']:.3f} [{m[k]['ci95'][0]:.3f}, {m[k]['ci95'][1]:.3f}]"
                null = m.get("label_shuffle_null")
                extra = f"  null max {null['max']:.3f}" if null else ""
                print(f"    {cell:66} recall {fmt('recall')}  FPR {fmt('fpr')}  AUROC {fmt('auroc')}{extra}")
    print(f"\nresults -> {path.relative_to(ROOT)}, {pred_path.relative_to(ROOT)}")
