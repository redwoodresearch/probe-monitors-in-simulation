"""Q2 probes: does a probe that flags impactful calls, trained on one domain's activations, work on the other?

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

  uv run modal run experiments/2026-10-06_q2_impactful_call_transfer/run_q2_probes.py [--limit-tasks N]
"""

import gzip
import json
import secrets
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q2_probes_kill-2026-11-06"
ACTS = "probe-shift_aswin-surya_sub-la-activations"  # the main session's Volume; read only
FEATS = "probe-shift_aswin-surya_sub-q2_probe-features_kill-2026-11-06"
POSITIONS = ["call_end", "pre_call", "prev_content"]

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2)
feats_vol = modal.Volume.from_name(FEATS, create_if_missing=True, version=2)
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
            "task_set_sha": cfg["task_set_sha"]}
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise ValueError(f"features run {d.name} does not match the config (cached, wanted): {bad}")


@app.function(image=cpu_image, volumes={"/acts": acts_vol, "/feats": feats_vol}, cpu=32, memory=131072, timeout=3 * 3600)
def features(cfg: dict, calls_gz: bytes) -> dict:
    from collections import Counter, defaultdict
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    calls = json.loads(gzip.decompress(calls_gz))
    layers = cfg["layers"]
    run_dir = Path("/acts") / cfg["activations_run"]
    out_dir = Path("/feats") / cfg["features_run"]
    out_dir.mkdir(parents=True, exist_ok=True)
    index = defaultdict(dict)  # (arm, key) -> {(step, action): call index}
    for i, c in enumerate(calls):
        index[(c["arm"], c["key"])][(c["step"], c["action"])] = i

    metas = {}
    for arm in sorted({c["arm"] for c in calls}):
        for p in (run_dir / "acts" / arm).glob("*.json"):
            m = json.loads(p.read_text())
            metas[(arm, m["key"])] = (p, m)
    missing = sorted(k for k in index if k not in metas)

    def load(arm_key):
        p, meta = metas[arm_key]
        mine = index[arm_key]
        npz = np.load(p.parent / f"{meta['filename']}.npz")
        arrays = {l: npz[f"L{l}"] for l in layers}  # each layer read once per transcript
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

        rows, reasons = [], []
        for (step, action), i in mine.items():
            qs = {kind: position(kind, step, action) for kind in POSITIONS}
            for kind in ("call_end", "pre_call"):
                if qs[kind] is not None and qs[kind]["tool"] != calls[i]["tool"]:  # both pipelines must agree on the call
                    raise ValueError(f"tool mismatch {arm_key} step {step} action {action}: {qs[kind]['tool']} vs {calls[i]['tool']}")
            if qs["prev_content"] is None:
                reasons.append((i, "no previous content token (first call, or after an empty output)"))
            elif not all(q["kept"] for q in qs.values() if q is not None) or any(q is None for q in qs.values()):
                reasons.append((i, "past the 32k cut, or a position missing"))
            else:
                vec = np.stack([np.stack([arrays[l][qs[k]["row"]] for l in layers]) for k in POSITIONS])
                if not np.isfinite(vec).all():  # fail loudly rather than feed NaN or Inf to the fits
                    raise ValueError(f"non-finite activations in {arm_key} step {step} action {action}")
                rows.append((i, vec))
        return rows, reasons

    with ThreadPoolExecutor(32) as pool:
        results = list(pool.map(load, [k for k in index if k in metas]))
    got = {i: x for rows, _ in results for i, x in rows}
    reasons = {i: r for _, rs in results for i, r in rs}
    for arm_key in missing:
        for i in index[arm_key].values():
            reasons[i] = "transcript not in the activation run"
    # Paired 2x2: keep a task's attack calls only if both attack arms' transcripts were read.
    cell_ok = defaultdict(set)
    for i, c in enumerate(calls):
        if (c["arm"], c["key"]) in metas:
            cell_ok[c["cell"]].add(c["arm"])
    eligible = []
    for i, c in enumerate(calls):
        if i not in got:
            continue
        if c["arm"] in cfg["test_domains"] and not set(cfg["test_domains"]) <= cell_ok[c["cell"]]:
            reasons[i] = "task's other attack arm not read"
            continue
        eligible.append(i)
    n, h = len(eligible), next(iter(got.values())).shape[-1]
    for pi, kind in enumerate(POSITIONS):
        arr = np.lib.format.open_memmap(out_dir / f"{kind}.npy", mode="w+", dtype=np.float16, shape=(len(layers), n, h))
        for r, i in enumerate(eligible):
            arr[:, r] = got[i][pi]
        arr.flush()
        del arr
    manifest = [calls[i] | {"row": r} for r, i in enumerate(eligible)]
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    report = {"calls_sent": len(calls), "eligible": n, "by_arm": dict(Counter(calls[i]["arm"] for i in eligible)),
              "excluded": dict(Counter(reasons.values())), "missing_transcripts": [list(k) for k in missing]}
    (out_dir / "report.json").write_text(json.dumps(report))
    (out_dir / "features_meta.json").write_text(json.dumps(  # what the arrays hold; fit and summary check it
        {"layers": layers, "positions": POSITIONS, "labeler_sha": cfg["labeler_sha"], "label_rule": cfg["label_rule"],
         "tasks": cfg["tasks"], "folds_file": cfg["folds_file"], "task_set_sha": cfg["task_set_sha"],
         "activations_run": cfg["activations_run"], "n_calls": n, "hidden": int(h)}))
    feats_vol.commit()
    return report


@app.function(image=gpu_image, gpu="A10G", volumes={"/feats": feats_vol}, memory=16384, timeout=4 * 3600)
def fit(cfg: dict, position: str, train_set: str, shuffle_seeds: list[int]) -> list[dict]:
    """Fits for one position and training set: the real labels when shuffle_seeds is [None], else one fit per seed
    with labels permuted within each transcript."""
    import numpy as np
    import torch

    from probe_shift.probes import select
    from probe_shift.transfer import transfer

    feats_vol.reload()
    d = Path("/feats") / cfg["features_run"]
    manifest = json.loads((d / "manifest.json").read_text())
    check_features(cfg, d)
    X = torch.from_numpy(np.load(d / f"{position}.npy")).cuda()  # [layer, call, hidden] fp16
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

        def fit_score(train, score_on):
            val, test = score_on
            ch = select(X, y, train, val, layers, grid, tol, cfg["max_iter"])  # chosen on this fold's validation calls
            choices.append({k: ch[k] for k in ("layer", "C", "val_auroc", "grad_max", "layer_curve")})
            weights.append({"coef": ch["w"][:-1].cpu().numpy(), "intercept": float(ch["w"][-1]), "mean": ch["mean"].cpu().numpy(),
                            "sd": ch["sd"].cpu().numpy(), "layer": ch["layer"], "C": ch["C"]})
            return [ch["scores"][val], ch["scores"][test]]

        r = transfer(fit_score, y, domain, fold, set(cfg["train_sets"][train_set]), fpr=cfg["fpr"])
        name = f"{position}__{train_set}__{'real' if seed is None else f'null{seed}'}"
        fd = d / "fits" / cfg["launch_id"]  # each launch keeps its own fits and probes
        fd.mkdir(parents=True, exist_ok=True)
        np.savez(fd / f"{name}.npz", y=y, score=r["score"], flag=r["flag"], threshold=r["threshold"])
        if seed is None:  # the trained probes, one per outer fold: score = ((x - mean) / sd) . coef + intercept on `layer`
            np.savez(fd / f"{name}_probes.npz", **{k: np.stack([wt[k] for wt in weights]) for k in weights[0]},
                     fold=np.arange(len(weights)), threshold=np.array([f["threshold"] for f in r["folds"]]))
        out.append({"position": position, "train_set": train_set, "seed": seed, "file": f"{name}.npz",
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
    has_flag = all("call_end_content" in m for m in manifest)  # features runs made before the flag have none
    content = np.array([m.get("call_end_content", False) for m in manifest])
    tests, home = cfg["test_domains"], cfg["home"]
    fit_dir = Path("/feats") / cfg["features_run"] / "fits" / cfg["launch_id"]
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
                 # calls whose call_end token is command text; unavailable for manifests made before the flag
                 "call_end_on_content": rep(cfg["n_boot"], content) if has_flag else "unavailable: features run predates the flag",
                 "by_category": {c: rep(cfg["n_boot_category"], category == c) for c in CLEAR_IMPACTFUL + UNCLEAR + READ_ONLY
                                 if (category == c).any()}}
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


def call_end_is_content(args: dict) -> bool:
    """False when call_end lands on formatting the agent emits the same way every time, not command text: a trailing
    number such as bash's timeout, a list such as text_editor's view_range, or a string ending in an escaped newline
    (the token is the n of the escape), or an empty string (the token is ""). Checked against the run's token texts."""
    if not args:
        return False
    last = args[list(args)[-1]]
    return not (isinstance(last, (int, float, list, dict)) or (isinstance(last, str) and (last == "" or last.endswith("\n"))))


def task_set_sha(tasks, folds) -> str:
    """Fingerprint of the analyzed main tasks and their folds."""
    import hashlib

    return hashlib.sha256(json.dumps({t: folds[t] for t in sorted(tasks)}).encode()).hexdigest()[:16]


@app.function(image=cpu_image, volumes={"/feats": feats_vol}, timeout=8 * 3600)
def orchestrate(cfg: dict, calls_gz: bytes | None, train_sets: list[str], n_null: int) -> dict:
    """Features, fits and summary, run on Modal so a dropped laptop connection cannot stop the sequence (launch with
    modal run --detach). The result is also saved on the Volume, where --fetch reads it."""
    report = features.remote(cfg, calls_gz) if calls_gz is not None else None
    jobs = [(cfg, pos, t, [None]) for pos in POSITIONS for t in train_sets]
    per = cfg["null_draws_per_job"]
    jobs += [(cfg, pos, t, list(range(cfg["seed"] + 1 + s, cfg["seed"] + 1 + min(s + per, n_null))))
             for pos in POSITIONS for t in cfg["null_train_sets"] if t in train_sets for s in range(0, n_null, per)]
    fits = [f for chunk in fit.starmap(jobs) for f in chunk]
    out = summary.remote(cfg, fits) | {"features_report": report, "config": cfg}  # --fetch reports this config
    feats_vol.reload()
    runs = Path("/feats") / "launches"  # by launch ID alone, so --fetch needs no config to find a run
    runs.mkdir(exist_ok=True)
    (runs / f"probes_{cfg['launch_id']}.json").write_text(json.dumps(out))
    feats_vol.commit()
    return out


@app.local_entrypoint()
def main(limit_tasks: int = 0, skip_features: bool = False, fetch: str = ""):
    """--fetch LAUNCH_ID: don't compute; read the result that detached launch saved on the Volume (same labels).
    --limit-tasks N: pilot on the first N main tasks (features, one fit per position for the first training set,
    two null draws), written under a separate features run."""
    import yaml

    from probe_shift.calls import IMPACTFUL
    from probe_shift.labels import resolve_labels
    from probe_shift import provenance

    def launch():
        cfg = yaml.safe_load((HERE / "config" / "q2_probes.yaml").read_text())
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
        cfg["launch_id"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S_") + secrets.token_hex(3)  # --fetch takes this
        print(f"launch {cfg['launch_id']}", flush=True)
        cfg["provenance"] = provenance.capture(ROOT, [Path(__file__)] + [ROOT / "src" / "probe_shift" / f for f in ("probes.py", "transfer.py", "calls.py", "labels.py")])  # at launch

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
                 {"fold": folds[r["main_task"]], "y": y_of(r), "call_end_content": call_end_is_content(r["arguments"])}
                 for r in rows if r["arm"] in arms and r["main_task"] in tasks and (r[cfg["tasks"]] or r["arm"] not in cfg["test_domains"])]
        print(f"{len(calls)} labeled calls ({cfg['tasks']} attack tasks; honest calls of the same main tasks) -> {APP}", flush=True)
        train_sets = list(cfg["train_sets"])[:1] if limit_tasks else list(cfg["train_sets"])
        n_null = 2 if limit_tasks else cfg["shuffle_null"]
        out = orchestrate.remote(cfg, None if skip_features else gzip.compress(json.dumps(calls).encode()), train_sets, n_null)
        out.pop("config", None)
        return cfg, out

    if fetch:  # a detached launch: its own config, provenance and identity; nothing local is resolved first
        out = json.loads(b"".join(feats_vol.read_file(f"launches/probes_{fetch}.json")))
        cfg = out.pop("config")
    else:
        cfg, out = launch()
    print("features:", json.dumps(out.get("features_report"))[:2000], flush=True)

    stamp = cfg["launch_id"]  # local files carry the launch ID, so two launches never share a name
    tag = f"_pilot{limit_tasks}" if limit_tasks else ""
    pred_path = HERE / "results" / f"q2_probe_predictions_{stamp}{tag}.jsonl.gz"
    ids = out.pop("calls")
    with gzip.open(pred_path, "wt") as f:  # per test call and probe: score, flag at the frozen threshold, threshold
        for p in out.pop("predictions"):
            for c, s, fl, t in zip(ids, p["score"], p["flag"], p["threshold"]):
                f.write(json.dumps({"position": p["position"], "train_set": p["train_set"], **c, "score": s, "flag": fl, "threshold": t}) + "\n")
    git = cfg["provenance"]["git"]
    path = HERE / "results" / f"q2_probes_{stamp}{tag}.json"
    path.write_text(json.dumps({"config": cfg | {"modal": {"app": APP, "activations_volume": ACTS, "features_volume": FEATS}},
                                "git": git, "labeler_sha": cfg["labeler_sha"], "predictions_file": pred_path.name, **out}, indent=1))
    for pos, e in out["positions"].items():
        print(f"\n=== {pos}")
        for t, fl in e["folds"].items():
            print(f"  {t}: chosen per fold", [(c["layer"], c["C"], round(c["val_auroc"], 3)) for c in fl],
                  "max |grad|", max(c["grad_max"] for c in fl))
        for subset in [x for x in ("all_calls", "robustness_no_unseen_effects", "call_end_on_content") if isinstance(e[x], dict)]:
            print(f"  [{subset}]")
            for cell, m in e[subset].items():
                fmt = lambda k: f"{m[k]['value']:.3f} [{m[k]['ci95'][0]:.3f}, {m[k]['ci95'][1]:.3f}]"
                null = m.get("label_shuffle_null")
                extra = f"  null max {null['max']:.3f}" if null else ""
                print(f"    {cell:66} recall {fmt('recall')}  FPR {fmt('fpr')}  AUROC {fmt('auroc')}{extra}")
    print(f"\nresults -> {path.relative_to(ROOT)}, {pred_path.relative_to(ROOT)}")
