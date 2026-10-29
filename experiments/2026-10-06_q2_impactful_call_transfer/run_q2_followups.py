"""Q2 follow-ups after the 2x2, on the features run_q2_probes.py built (Modal GPU, then a CPU summary).

1. Adaptation, for pre_call and prev_content, in both directions (source -> target). Per outer task fold, the
   probe is fit and its layer and C are chosen on source data, as in the 2x2. Then on the held-out target test calls:
     - frozen: the source validation threshold (the 2x2 cell);
     - recalibrated (doc follow-up a): the threshold reset to 1% false positives on target validation calls (the
       target's calls in the fold the source used for validation);
     - updated (doc follow-up b): the probe refit on `n_shots` random target validation calls, starting from and
       shrunk toward the source weights (same objective, prior centred on the source probe). C is the source probe's
       C times each multiplier. Scored with the frozen source-validation threshold and by AUROC; `n_draws` random
       draws of the shots.
2. Held-out environment, for call_end and pre_call, trained on real or simulated attack: leave-one-environment-out
   outer folds (splits environment_folds), the next environment as validation, same selection and threshold rule.

  uv run modal run experiments/2026-10-06_q2_impactful_call_transfer/run_q2_followups.py
"""

import hashlib
import json
import secrets
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q2_followups_kill-2026-11-06"
FEATS = "probe-shift_aswin-surya_sub-q2_probe-features_kill-2026-11-06"

app = modal.App(APP)
feats_vol = modal.Volume.from_name(FEATS, version=2)
cpu_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2")
             .add_local_python_source("probe_shift"))
gpu_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2", "torch==2.8.0")
             .add_local_python_source("probe_shift"))


def load(cfg, position):
    import numpy as np

    d = Path("/feats") / cfg["features_run"]
    meta = json.loads((d / "features_meta.json").read_text())
    want = {k: cfg[k] for k in ("layers", "labeler_sha", "label_rule", "activations_run", "tasks", "folds_file", "task_set_sha")}
    want["positions"] = ["call_end", "pre_call", "prev_content"]
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise ValueError(f"features run {d.name} does not match the config (cached, wanted): {bad}")
    manifest = json.loads((d / "manifest.json").read_text())
    arr = lambda k: np.array([m[k] for m in manifest])
    return manifest, np.load(d / f"{position}.npy"), arr("y"), arr("arm"), arr("fold"), arr("main_task")


@app.function(image=gpu_image, gpu="A10G", volumes={"/feats": feats_vol}, memory=16384, timeout=4 * 3600)
def adapt(cfg: dict, position: str, source: str, target: str) -> dict:
    import numpy as np
    import torch

    from probe_shift.probes import fit_logreg, select
    from probe_shift.transfer import threshold_at_fpr

    _, X, y, domain, fold, _ = load(cfg, position)
    X = torch.from_numpy(X).cuda()
    n_folds = len(set(fold.tolist()))
    out = {"rows": [], "frozen": [], "recalibrated": [], "updated": {}, "folds": []}
    for f in range(n_folds):
        v = (f + 1) % n_folds
        src = domain == source
        train, val = np.flatnonzero(src & (fold != f) & (fold != v)), np.flatnonzero(src & (fold == v))
        tval, ttest = np.flatnonzero((domain == target) & (fold == v)), np.flatnonzero((domain == target) & (fold == f))
        ch = select(X, y, train, val, cfg["layers"], cfg["C_grid"], cfg["layer_tolerance"], cfg["max_iter"])
        s = ch["scores"]
        t_src, t_tgt = threshold_at_fpr(s[val][y[val] == 0], cfg["fpr"]), threshold_at_fpr(s[tval][y[tval] == 0], cfg["fpr"])
        out["rows"] += ttest.tolist()
        out["frozen"] += list(zip(s[ttest].tolist(), (s[ttest] > t_src).tolist()))
        out["recalibrated"] += list(zip(s[ttest].tolist(), (s[ttest] > t_tgt).tolist()))
        Z = (X[ch["li"]].float() - ch["mean"]) / ch["sd"]
        grads = []
        for mult in cfg["update_C_multipliers"]:
            for draw in range(cfg["n_draws"]):
                rng = np.random.default_rng(cfg["seed"] + 1000 * f + draw)
                shots = rng.choice(tval, cfg["n_shots"], replace=False)
                w, g = fit_logreg(Z[shots], torch.tensor(y[shots], dtype=torch.float32, device="cuda"), ch["C"] * mult,
                                  cfg["max_iter"], w0=ch["w"])
                grads.append(g)
                su = (Z @ w[:-1] + w[-1]).cpu().numpy()
                tu = threshold_at_fpr(su[val][y[val] == 0], cfg["fpr"])  # deployer's threshold: source validation
                out["updated"].setdefault(f"Cx{mult}/draw{draw}", []).extend(zip(su[ttest].tolist(), (su[ttest] > tu).tolist()))
        out["folds"].append({"fold": f, "layer": ch["layer"], "C": ch["C"], "val_auroc": ch["val_auroc"],
                             "grad_max": ch["grad_max"], "update_grad_max": max(grads), "threshold_source": t_src,
                             "threshold_target": t_tgt})
        del Z
    return {"position": position, "source": source, "target": target} | out


@app.function(image=gpu_image, gpu="A10G", volumes={"/feats": feats_vol}, memory=16384, timeout=4 * 3600)
def heldout_env(cfg: dict, position: str, train_set: str) -> dict:
    import numpy as np
    import torch

    from probe_shift.probes import select
    from probe_shift.transfer import transfer

    _, X, y, domain, _, main_task = load(cfg, position)
    X = torch.from_numpy(X).cuda()
    env_fold = np.array([cfg["main_task_env_fold"][t] for t in main_task])
    choices = []

    def fit_score(train, score_on):
        val, test = score_on
        ch = select(X, y, train, val, cfg["layers"], cfg["C_grid"], cfg["layer_tolerance"], cfg["max_iter"])
        choices.append({k: ch[k] for k in ("layer", "C", "val_auroc", "grad_max")})
        return [ch["scores"][val], ch["scores"][test]]

    r = transfer(fit_score, y, domain, env_fold, set(cfg["train_sets"][train_set]), fpr=cfg["fpr"],
                 n_folds=len(set(env_fold.tolist())))
    return {"position": position, "train_set": train_set, "score": r["score"].tolist(), "flag": r["flag"].tolist(),
            "folds": [f | c for f, c in zip(r["folds"], choices)]}


@app.function(image=cpu_image, volumes={"/feats": feats_vol}, cpu=16, memory=32768, timeout=3600)
def summarize(cfg: dict, adapted: list[dict], env: list[dict]) -> dict:
    import numpy as np

    from probe_shift.transfer import two_by_two

    manifest = json.loads((Path("/feats") / cfg["features_run"] / "manifest.json").read_text())
    y = np.array([m["y"] for m in manifest])
    domain = np.array([m["arm"] for m in manifest])
    groups = np.array([m["main_task"] for m in manifest])

    def as_result(rows, pairs):
        score, flag = np.full(len(y), np.nan), np.zeros(len(y), bool)
        score[rows] = [p[0] for p in pairs]
        flag[rows] = [p[1] for p in pairs]
        return {"score": score, "flag": flag}

    out = {"adaptation": {}, "heldout_environment": {}}
    for a in adapted:
        res = {"frozen": as_result(a["rows"], a["frozen"]), "recalibrated": as_result(a["rows"], a["recalibrated"])}
        res |= {f"updated {k}": as_result(a["rows"], v) for k, v in a["updated"].items()}
        rep = two_by_two(res, {k: a["source"] for k in res}, y, domain, groups, [a["target"]], cfg["n_boot"], cfg["seed"])
        point = {k: rep[f"train {k} / test {a['target']}"] for k in res}
        updated = {}
        for mult in cfg["update_C_multipliers"]:
            draws = [point[f"updated Cx{mult}/draw{d}"] for d in range(cfg["n_draws"])]
            updated[f"Cx{mult}"] = {m: {"mean": round(float(np.mean([d[m]["value"] for d in draws])), 4),
                                        "min": round(float(np.min([d[m]["value"] for d in draws])), 4),
                                        "max": round(float(np.max([d[m]["value"] for d in draws])), 4)}
                                    for m in ("recall", "fpr", "auroc")}
        out["adaptation"][f"{a['position']}: {a['source']} -> {a['target']}"] = {
            "frozen": point["frozen"], "recalibrated": point["recalibrated"], "updated_over_draws": updated,
            "updated_draws": {k: v for k, v in point.items() if k.startswith("updated")}, "folds": a["folds"]}
    for e in env:
        res = {e["train_set"]: {"score": np.array(e["score"], float), "flag": np.array(e["flag"])}}
        rep = two_by_two(res, {e["train_set"]: e["train_set"]}, y, domain, groups, cfg["test_domains"], cfg["n_boot"], cfg["seed"])
        out["heldout_environment"][f"{e['position']}: train {e['train_set']}"] = {"cells": rep, "folds": e["folds"]}
    return out


@app.function(image=cpu_image, volumes={"/feats": feats_vol}, timeout=8 * 3600)
def orchestrate(cfg: dict, adapt_jobs: list, env_jobs: list) -> dict:
    """Adaptation, held-out environment and summary, run on Modal so a dropped laptop connection cannot stop the
    sequence. The result is also saved next to the features."""
    adapted = list(adapt.starmap(adapt_jobs))
    env = list(heldout_env.starmap(env_jobs))
    out = summarize.remote(cfg, adapted, env) | {"config": {k: v for k, v in cfg.items() if k != "main_task_env_fold"}}
    feats_vol.reload()
    runs = Path("/feats") / "launches"  # by launch ID alone, so --fetch needs no config to find a run
    runs.mkdir(exist_ok=True)
    (runs / f"followups_{cfg['launch_id']}.json").write_text(json.dumps(out))
    feats_vol.commit()
    return out


@app.local_entrypoint()
def main(fetch: str = ""):
    """--fetch LAUNCH_ID: read the result that detached launch saved next to the features, instead of computing."""
    import yaml

    from probe_shift import provenance
    from probe_shift.labels import resolve_labels

    def launch():
        cfg = yaml.safe_load((HERE / "config" / "q2_probes.yaml").read_text()) | \
            yaml.safe_load((HERE / "config" / "q2_followups.yaml").read_text())
        labels, cfg["labeler_sha"] = resolve_labels(ROOT, cfg["labels_file"])  # `latest`: newest table from the current labeler
        cfg["labels_file"] = str(labels.relative_to(ROOT))
        cfg["features_run"] += f"_{cfg['labeler_sha']}_{cfg['label_rule']}"
        splits = json.loads((ROOT / "splits" / cfg["folds_file"]).read_text())
        folds = splits["task_folds"]["main_task_to_fold"]
        tasks = {r["main_task"] for r in map(json.loads, labels.read_text().splitlines()) if r[cfg["tasks"]]}
        cfg["task_set_sha"] = hashlib.sha256(json.dumps({t: folds[t] for t in sorted(tasks)}).encode()).hexdigest()[:16]
        env_to_fold = splits["environment_folds"]["environment_to_fold"]
        cfg["main_task_env_fold"] = {t: env_to_fold[e] for t, e in splits["main_task_environment"].items()}

        cfg["launch_id"] = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S_") + secrets.token_hex(3)  # --fetch takes this
        print(f"launch {cfg['launch_id']}", flush=True)
        cfg["provenance"] = provenance.capture(ROOT, [Path(__file__)] + [ROOT / "src" / "probe_shift" / f for f in ("probes.py", "transfer.py", "calls.py", "labels.py")])  # at launch
        adapt_jobs = [(cfg, pos, s, t) for pos in cfg["adapt_positions"] for s, t in cfg["directions"]]
        env_jobs = [(cfg, pos, t) for pos in cfg["env_positions"] for t in cfg["env_train_sets"]]
        out = orchestrate.remote(cfg, adapt_jobs, env_jobs)  # on Modal: launch with --detach
        return out.pop("config"), out

    if fetch:  # a detached launch: its own config, provenance and identity; nothing local is resolved first
        out = json.loads(b"".join(feats_vol.read_file(f"launches/followups_{fetch}.json")))
        cfg = out.pop("config")
    else:
        cfg, out = launch()
    cfg |= {"main_task_env_fold": None}

    stamp = cfg["launch_id"]  # local files carry the launch ID, so two launches never share a name
    git = cfg["provenance"]["git"]
    path = HERE / "results" / f"q2_followups_{stamp}.json"
    cfg.pop("main_task_env_fold")
    path.write_text(json.dumps({"config": cfg | {"modal": {"app": APP, "features_volume": FEATS}}, "git": git, **out}, indent=1))
    fmt = lambda d: f"{d['value']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
    for name, a in out["adaptation"].items():
        print(f"\n=== adaptation {name}")
        for k in ("frozen", "recalibrated"):
            print(f"  {k:14} recall {fmt(a[k]['recall'])}  FPR {fmt(a[k]['fpr'])}  AUROC {fmt(a[k]['auroc'])}")
        for k, u in a["updated_over_draws"].items():
            print(f"  updated {k:6} recall {u['recall']['mean']:.3f} ({u['recall']['min']:.3f}-{u['recall']['max']:.3f})  "
                  f"FPR {u['fpr']['mean']:.3f}  AUROC {u['auroc']['mean']:.3f}")
    for name, e in out["heldout_environment"].items():
        print(f"\n=== held-out environment {name}")
        for cell, m in e["cells"].items():
            print(f"  {cell:56} recall {fmt(m['recall'])}  FPR {fmt(m['fpr'])}  AUROC {fmt(m['auroc'])}")
    print(f"\nresults -> {path.relative_to(ROOT)}")
