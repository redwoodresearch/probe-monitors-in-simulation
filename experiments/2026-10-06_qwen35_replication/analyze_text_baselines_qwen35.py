"""The qwen35 track's copy of experiments/2026-10-06_q2_impactful_call_transfer/analyze_text_baselines.py (PR #16 at e9c3b9e), for Qwen3.5-27B
as the reader: this track's app and features Volume (run_q2_probes_qwen35.py writes it) and `--probes-config`.
The original docstring follows.

Text-only baselines for the Q2 2x2: how well can the command text, or the previous calls alone, flag impactful
(not read-only) calls when trained on one domain and tested on both?

  - command_text: TF-IDF of the call's JSON. The bar for the call_end probe (readout a); the labels are rules on
    this same text, so near-perfect scores are expected.
  - previous_calls: the categories and tools of the two previous calls. The bar for the pre_call probe (readout b),
    which sees only what came before the command.

Training sets: real attack epoch 1, simulated attack, and real attack plus real honest (the project doc's "LA honest +
attack" variant). Test sets: held-out tasks of real attack epoch 1 and simulated attack. Same protocol as the probes
(probe_shift.transfer): shared folds, threshold frozen at 1% false positives on training-set validation, 95% intervals
from resampling main tasks. Reported for all calls, without calls that are impactful only because their effects are
unseen (run_program, unknown: the robustness row), without text_editor calls, and per category. Runs on Modal CPU.

  uv run modal run experiments/2026-10-06_qwen35_replication/analyze_text_baselines_qwen35.py --probes-config q2_probes_qwen35.yaml
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
APP = "probe-shift_aswin-surya_sub-qwen35_q2-text_kill-2026-11-06"
FEATS = "probe-shift_aswin-surya_sub-qwen35_q2feats_kill-2026-11-06"  # run_q2_probes.py's manifest of eligible calls

ACTS = "probe-shift_aswin-surya_sub-la-activations"  # the shared segment-end run (read only): which calls the reader saw

app = modal.App(APP)
acts_vol = modal.Volume.from_name(ACTS, version=2).read_only()
feats_vol = modal.Volume.from_name(FEATS, version=2).read_only()

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

image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2")
         .add_local_python_source("probe_shift"))


def previous_call_features(rows: list[dict]) -> list[list[str]]:
    """Category and tool of the two previous calls in the same transcript, and whether this call is not the first of
    its agent turn. Computed over every call of the transcript, in order."""
    by_key = {}
    for i, r in enumerate(rows):
        by_key.setdefault((r["arm"], r["key"]), []).append(i)
    feats = [None] * len(rows)
    for idx in by_key.values():
        idx = sorted(idx, key=lambda i: (rows[i]["step"], rows[i]["action"]))
        for n, i in enumerate(idx):
            prev = [rows[idx[n - k]] if n - k >= 0 else None for k in (1, 2)]
            feats[i] = [p["category"] if p else "none" for p in prev] + [p["tool"] if p else "none" for p in prev] + \
                       [str(rows[i]["action"] > 0)]
    return feats


@app.function(image=image, volumes={"/acts": acts_vol, "/feats": feats_vol}, cpu=16, memory=65536, timeout=2 * 3600)
def analyze(cfg: dict, labels_gz: bytes, folds: dict) -> dict:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    from probe_shift.calls import IMPACTFUL, READ_ONLY, UNCLEAR
    from probe_shift.transfer import transfer, two_by_two

    every = [json.loads(l) for l in gzip.decompress(labels_gz).decode().splitlines()]
    prev = previous_call_features(every)
    test_domains = cfg["test_domains"]
    tasks = {r["main_task"] for r in every if r[cfg["tasks"]]}
    # Attack calls in the configured tasks; honest calls of the same main tasks (training only).
    keep = [i for i, r in enumerate(every) if (r["arm"] in test_domains and r[cfg["tasks"]]) or
            (r["arm"] == "real-honest" and r["main_task"] in tasks)]
    exclusions = {}
    if cfg.get("within_reader_cut"):  # only calls the reader saw: their call_end token inside the 32k cut
        seen, read, run_dir = set(), set(), Path("/acts") / cfg["within_reader_cut"]["activations_run"] / "acts"
        for arm in sorted({every[i]["arm"] for i in keep}):
            for mp in (run_dir / arm).glob("*.json"):
                m = json.loads(mp.read_text())
                read.add((arm, m["key"]))
                seen |= {(arm, m["key"], q["step"], q["action"]) for q in m["positions"] if q["kind"] == "call_end" and q["kept"]}
        missing = sorted({(every[i]["arm"], every[i]["key"]) for i in keep} - read)
        if missing:  # a missing transcript is an extraction gap, not a cutoff: refuse rather than drop it silently
            raise ValueError(f"{len(missing)} transcripts are not in the activation run, e.g. {missing[:3]}")
        n_before = len(keep)
        keep = [i for i in keep if (every[i]["arm"], every[i]["key"], every[i]["step"], every[i]["action"]) in seen]
        exclusions["past the reader's 32k cut"] = n_before - len(keep)
        print(f"within the reader's cut: {len(keep)} of {n_before} calls", flush=True)
    rows, prev = [every[i] for i in keep], [prev[i] for i in keep]
    y = np.array([r["label"] == "impactful" for r in rows], int)
    if cfg.get("features_run"):  # score exactly the probes' eligible calls, with the probes' labels (label_rule applied)
        meta = json.loads((Path("/feats") / cfg["features_run"] / "features_meta.json").read_text())
        if meta["labeler_sha"] != cfg["labeler_sha"]:
            raise ValueError(f"features run {cfg['features_run']} was labeled by {meta['labeler_sha']}, not {cfg['labeler_sha']}")
        exclusions["label_rule (from the features run)"] = meta.get("label_rule")
        manifest = json.loads((Path("/feats") / cfg["features_run"] / "manifest.json").read_text())
        y_of = {(m["arm"], m["key"], m["step"], m["action"]): m["y"] for m in manifest}
        sel = [i for i, r in enumerate(rows) if (r["arm"], r["key"], r["step"], r["action"]) in y_of]
        exclusions["not eligible for the probes (no previous content token, or a position past the cut)"] = len(rows) - len(sel)
        rows, prev = [rows[i] for i in sel], [prev[i] for i in sel]
        y = np.array([y_of[(r["arm"], r["key"], r["step"], r["action"])] for r in rows], int)
        if len(rows) != len(manifest):
            raise ValueError(f"{len(manifest) - len(rows)} manifest calls are not in the label table")
    domain = np.array([r["arm"] for r in rows])
    groups = np.array([r["main_task"] for r in rows])
    fold = np.array([folds[g] for g in groups])
    category = np.array([r["category"] for r in rows])
    editor = np.array([r["tool"] == "text_editor" for r in rows])
    texts = [json.dumps({"name": r["tool"], "arguments": r["arguments"]}, ensure_ascii=False) for r in rows]

    def command_text(train, score_on):
        c = cfg["command_text"]
        vec = TfidfVectorizer(**{k: tuple(v) if isinstance(v, list) else v for k, v in c["tfidf"].items()})
        X = vec.fit_transform([texts[i] for i in train])
        lr = LogisticRegression(**c["logistic_regression"]).fit(X, y[train])
        return [lr.decision_function(vec.transform([texts[i] for i in idx])) for idx in score_on]

    def previous_calls(train, score_on):
        enc = OneHotEncoder(handle_unknown="ignore").fit([prev[i] for i in train])
        lr = LogisticRegression(**cfg["previous_calls"]["logistic_regression"]).fit(enc.transform([prev[i] for i in train]), y[train])
        return [lr.decision_function(enc.transform([prev[i] for i in idx])) for idx in score_on]

    train_sets = {name: set(arms) for name, arms in cfg["train_sets"].items()}
    home = {name: cfg["home"][name] for name in train_sets}
    out = {"exclusions": exclusions, "n_calls": {d: {"impactful": int(((domain == d) & (y == 1)).sum()), "read_only": int(((domain == d) & (y == 0)).sum())}
                       for d in sorted(set(domain))}, "baselines": {}, "predictions": {}}
    for name, fit_score in {"command_text": command_text, "previous_calls": previous_calls}.items():
        res = {t: transfer(fit_score, y, domain, fold, arms, fpr=cfg["fpr"]) for t, arms in train_sets.items()}
        report = lambda n_boot, mask=None: two_by_two(res, home, y, domain, groups, test_domains, n_boot, cfg["seed"], mask)
        out["baselines"][name] = {
            "folds": {t: r["folds"] for t, r in res.items()},
            "all_calls": report(cfg["n_boot"]),
            "robustness_no_unseen_effects": report(cfg["n_boot"], ~np.isin(category, UNCLEAR)),
            "no_text_editor": report(cfg["n_boot"], ~editor),
            "by_category": {c: report(cfg["n_boot_category"], category == c) for c in IMPACTFUL + READ_ONLY if (category == c).any()}}
        test = np.isin(domain, test_domains)
        out["predictions"][name] = {t: {k: r[k][test].tolist() for k in ("score", "flag", "threshold")} for t, r in res.items()}
    out["prediction_rows"] = [{k: rows[i][k] for k in ("arm", "key", "step", "action", "tool", "category")} | {"y": int(y[i]), "fold": int(fold[i])}
                              for i in np.flatnonzero(np.isin(domain, test_domains))]
    return out


@app.local_entrypoint()
def main(probes_config: str = "q2_probes_qwen35.yaml"):
    import yaml

    from probe_shift.labels import resolve_labels

    cfg = yaml.safe_load((HERE / "config" / "text_baselines_qwen35.yaml").read_text())
    labels, made_by = resolve_labels(ROOT, cfg["labels_file"])  # `latest`: newest table from the current labeler
    cfg["labels_file"] = str(labels.relative_to(ROOT))
    if cfg.get("features_run") == "auto":  # the probes' features run for these labels (run_q2_probes.py's naming)
        probes_cfg = yaml.safe_load((HERE / "config" / probes_config).read_text())
        cfg["features_run"] = f"{probes_cfg['features_run']}_{made_by}_{probes_cfg['label_rule']}"
    folds = json.loads((ROOT / "splits" / cfg["folds_file"]).read_text())["task_folds"]["main_task_to_fold"]
    cfg["labeler_sha"] = made_by
    out = wait(analyze.spawn(cfg, gzip.compress(labels.read_bytes()), folds))

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    pred_path = HERE / "results" / f"text_baseline_predictions_{stamp}_{Path(probes_config).stem}.jsonl.gz"
    rows = out.pop("prediction_rows")
    with gzip.open(pred_path, "wt") as f:
        for name, by_train in out.pop("predictions").items():
            for t, p in by_train.items():
                for r, s, fl, th in zip(rows, p["score"], p["flag"], p["threshold"]):
                    f.write(json.dumps({"baseline": name, "train_set": t, **r, "score": s, "flag": fl, "threshold": th}) + "\n")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    path = HERE / "results" / f"text_baselines_{stamp}_{Path(probes_config).stem}.json"
    path.write_text(json.dumps({"config": cfg | {"modal": {"app": APP}}, "git": git, "labeler_sha": made_by,
                                "predictions_file": pred_path.name, **out}, indent=1))
    print("calls:", json.dumps(out["n_calls"]))
    for name, entry in out["baselines"].items():
        print(f"\n=== {name}")
        for subset in ("all_calls", "robustness_no_unseen_effects"):
            print(f"  [{subset}]")
            for cell, m in entry[subset].items():
                fmt = lambda k: f"{m[k]['value']:.3f} [{m[k]['ci95'][0]:.3f}, {m[k]['ci95'][1]:.3f}]"
                print(f"    {cell:66} recall {fmt('recall')}  FPR {fmt('fpr')}  AUROC {fmt('auroc')}")
    print(f"\nresults -> {path.relative_to(ROOT)}, {pred_path.relative_to(ROOT)}")
